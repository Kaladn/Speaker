"""6-1-6 Context-Cloud NLP Core Engine.

Implements the CONTEXT-CLOUD NLP CORE pseudocode contract.
Every answer derivable from stored count structure only.
No POS tagging. No embeddings. No probabilistic sampling.
No grammar labels. Meaning is cloud behavior.

Sections map 1:1 to the pseudocode contract (0-20).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger("ClearboxAI.reasoning_616.engine")


# ══════════════════════════════════════════════════════════════
# SECTION 0: HARD RULES (enforced, not advisory)
# ══════════════════════════════════════════════════════════════
#
# RULE_01: Symbol is identity only. No meaning in symbol.
# RULE_02: Meaning = cloud behavior + stability + similarity + diff.
# RULE_03: Raw text temporary. Discard after counting.
# RULE_04: No duplicate symbol assignments.
# RULE_05: No symbol collisions.
# RULE_06: All counts deterministic. Same input = same output.
# RULE_07: Count structure IS the map.
# RULE_08: Never ask "is this a noun/verb?" Ask: what cloud surrounds it.
# RULE_09: Homographs solved by cloud comparison, not surface.
# RULE_10: Every answer derivable from stored counts.


# ══════════════════════════════════════════════════════════════
# SECTION 1: CORE DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class StabilityMetrics:
    """Section 5: how trustworthy is this cloud?"""
    recurrence_score: float = 0.0
    document_spread_score: float = 0.0
    positional_consistency_score: float = 0.0
    cohesion_score: float = 0.0
    rarity_score: float = 0.0
    focal_weight: float = 0.0


@dataclass
class SymbolRecord:
    """Permanent identity for one word/token."""
    symbol_id: str              # hex address
    surface_forms: Set[str] = field(default_factory=set)
    total_occurrences: int = 0
    doc_occurrences: int = 0


@dataclass
class PositionalNeighborCount:
    neighbor_symbol_id: str
    count: int


@dataclass
class ContextCloud:
    """12-position context cloud for one anchor."""
    anchor_symbol_id: str
    before: Dict[int, Dict[str, int]] = field(default_factory=dict)  # pos -> {neighbor_id: count}
    after: Dict[int, Dict[str, int]] = field(default_factory=dict)
    total_count: int = 0
    doc_spread: int = 0
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)


# ── Query structures ─────────────────────────────────────────

class QueryType(Enum):
    """Section 7: query shape classification."""
    SINGLE_ANCHOR = "single_anchor"
    MULTI_ANCHOR = "multi_anchor"
    FORWARD_RELATION = "forward_relation"
    BACKWARD_RELATION = "backward_relation"
    RELATEDNESS = "relatedness"
    CONTRAST = "contrast"
    UNKNOWN = "unknown"


@dataclass
class QueryCloud:
    """The query mapped into symbol/cloud space."""
    query_symbols: List[str] = field(default_factory=list)
    before: Dict[int, Dict[str, int]] = field(default_factory=dict)
    after: Dict[int, Dict[str, int]] = field(default_factory=dict)
    pair_links: List[Tuple[str, str]] = field(default_factory=list)
    query_type: QueryType = QueryType.UNKNOWN
    focal_symbols: List[str] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    """Section 12: diff output showing why a match was scored."""
    matched_neighbors: List[Dict] = field(default_factory=list)
    matched_positions: List[int] = field(default_factory=list)
    exclusive_neighbors: List[Dict] = field(default_factory=list)
    diff_neighbors: List[Dict] = field(default_factory=list)
    supporting_counts: Dict[str, int] = field(default_factory=dict)
    stability_notes: List[str] = field(default_factory=list)


@dataclass
class CloudMatch:
    """Section 8: scored candidate from cloud search."""
    candidate_symbol_id: str
    candidate_word: str = ""
    local_similarity_score: float = 0.0
    layered_similarity_score: float = 0.0
    overlap_score: float = 0.0
    contrast_score: float = 0.0
    directional_match_score: float = 0.0
    stability_adjustment: float = 0.0
    final_score: float = 0.0
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)


@dataclass
class SupportMetrics:
    """Section 14: honest, grounded confidence."""
    overlap_strength: float = 0.0
    directional_consistency: float = 0.0
    cloud_stability: float = 0.0
    ambiguity: float = 0.0
    query_coverage: float = 0.0


@dataclass
class AnswerResult:
    """Section 13: final output."""
    answer_text: str = ""
    supporting_symbols: List[str] = field(default_factory=list)
    evidence_bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    uncertainty_notes: List[str] = field(default_factory=list)
    support_metrics: SupportMetrics = field(default_factory=SupportMetrics)
    score: float = 0.0
    query_type: QueryType = QueryType.UNKNOWN
    ranked_matches: List[CloudMatch] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# SECTION 9: POSITION WEIGHT (contract: nearest = heaviest)
# ══════════════════════════════════════════════════════════════

POSITION_WEIGHTS = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}


def _pos_weight(position: int) -> int:
    """Contract Section 9: GET_POSITION_WEIGHT."""
    return POSITION_WEIGHTS.get(abs(position), 0)


# ══════════════════════════════════════════════════════════════
# SECTION 2 + 3: LEXICON + NORMALIZATION (delegates to bridge)
# ══════════════════════════════════════════════════════════════

class LexiconAdapter:
    """Adapts the Clearbox lexicon bridge to the contract interface.

    Contract functions: ASSIGN_SYMBOL, GET_SYMBOL, VERIFY_UNIQUENESS.
    Delegates to bridge.word_index / bridge.entries.
    """

    def __init__(self, bridge):
        self._bridge = bridge

    def get_symbol(self, surface_text: str) -> Optional[str]:
        """Contract: GET_SYMBOL. Returns symbol_id or None."""
        if self._bridge and hasattr(self._bridge, "word_index"):
            return self._bridge.word_index.get(surface_text.lower())
        return None

    def get_word(self, symbol_id: str) -> Optional[str]:
        """Reverse lookup: symbol_id → word."""
        if not self._bridge:
            return None
        if hasattr(self._bridge, "entries"):
            entry = self._bridge.entries.get(symbol_id)
            if entry and hasattr(entry, "word") and entry.word:
                return entry.word
        return None

    def get_all_symbols(self) -> Set[str]:
        """All known symbol_ids."""
        if self._bridge and hasattr(self._bridge, "word_index"):
            return set(self._bridge.word_index.values())
        return set()

    @property
    def word_count(self) -> int:
        if self._bridge and hasattr(self._bridge, "word_index"):
            return len(self._bridge.word_index)
        return 0


# ══════════════════════════════════════════════════════════════
# SECTION 4 + EVIDENCE STORE ADAPTER
# ══════════════════════════════════════════════════════════════

class EvidenceAdapter:
    """Reads from the cumulative evidence store.

    Maps EvidenceCell → ContextCloud for the contract interface.
    """

    def __init__(self):
        self._store = None

    def _ensure(self):
        if self._store is None:
            from bridges.evidence_store import get_evidence_store
            self._store = get_evidence_store()

    def get_cloud(self, symbol_id: str) -> Optional[ContextCloud]:
        """Read evidence cell, return as ContextCloud."""
        self._ensure()
        cell = self._store.read_cell(symbol_id)
        if cell is None:
            return None

        before: Dict[int, Dict[str, int]] = {}
        after: Dict[int, Dict[str, int]] = {}

        for offset, entries in cell.buckets.items():
            bucket = {e.hex_addr: e.count for e in entries}
            if offset < 0:
                before[abs(offset)] = bucket
            else:
                after[offset] = bucket

        return ContextCloud(
            anchor_symbol_id=symbol_id,
            before=before,
            after=after,
            total_count=cell.total_count,
        )

    def cell_count(self) -> int:
        self._ensure()
        return self._store.count()

    def get_all_neighbor_symbols(self, symbol_id: str) -> Set[str]:
        """All neighbor symbol_ids across all positions."""
        cloud = self.get_cloud(symbol_id)
        if not cloud:
            return set()
        neighbors = set()
        for bucket in cloud.before.values():
            neighbors.update(bucket.keys())
        for bucket in cloud.after.values():
            neighbors.update(bucket.keys())
        return neighbors


# ══════════════════════════════════════════════════════════════
# SECTION 5: STABILITY / FOCAL WEIGHT
# ══════════════════════════════════════════════════════════════

def compute_stability(cloud: ContextCloud, symbol: SymbolRecord) -> StabilityMetrics:
    """Contract Section 5: COMPUTE_STABILITY_METRICS.

    Rare alone is NOT enough.
    Rare + stable recurring cloud = high focal value.
    Rare + weak chaotic cloud = low trust.
    """
    # Recurrence: how often does this symbol appear?
    recurrence = min(math.log(symbol.total_occurrences + 1) / 10.0, 1.0)

    # Document spread: appears in how many docs?
    doc_spread = min(symbol.doc_occurrences / 100.0, 1.0) if symbol.doc_occurrences > 0 else 0.0

    # Positional consistency: do the same neighbors recur in the same positions?
    position_scores = []
    for pos_buckets in [cloud.before, cloud.after]:
        for pos, bucket in pos_buckets.items():
            if not bucket:
                continue
            counts = list(bucket.values())
            if len(counts) < 2:
                position_scores.append(1.0)  # single neighbor = fully consistent
                continue
            top = max(counts)
            total = sum(counts)
            position_scores.append(top / total if total > 0 else 0)

    positional_consistency = sum(position_scores) / len(position_scores) if position_scores else 0.0

    # Cohesion: is the cloud concentrated or randomly spread?
    all_counts = []
    for pos_buckets in [cloud.before, cloud.after]:
        for bucket in pos_buckets.values():
            all_counts.extend(bucket.values())
    if all_counts:
        top_5 = sum(sorted(all_counts, reverse=True)[:5])
        total = sum(all_counts)
        cohesion = top_5 / total if total > 0 else 0
    else:
        cohesion = 0.0

    # Rarity: inverse frequency, bounded
    rarity = 1.0 / (1.0 + math.log(symbol.total_occurrences + 1))

    # Focal weight: combine all
    # Rare + stable = high. Rare + chaotic = low.
    focal = (
        rarity * 0.15 +
        recurrence * 0.15 +
        doc_spread * 0.20 +
        positional_consistency * 0.25 +
        cohesion * 0.25
    )

    return StabilityMetrics(
        recurrence_score=round(recurrence, 4),
        document_spread_score=round(doc_spread, 4),
        positional_consistency_score=round(positional_consistency, 4),
        cohesion_score=round(cohesion, 4),
        rarity_score=round(rarity, 4),
        focal_weight=round(focal, 4),
    )


# ══════════════════════════════════════════════════════════════
# SECTION 6: QUERY INTAKE
# ══════════════════════════════════════════════════════════════

def process_query(
    raw_question: str,
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> QueryCloud:
    """Contract Section 6: PROCESS_QUERY."""
    from core.lakespeak.text.normalize import extract_anchors

    words = extract_anchors(raw_question)
    symbol_list = []
    for w in words:
        sid = lexicon.get_symbol(w)
        if sid:
            symbol_list.append(sid)

    # Build query cloud (same 6-before/6-after counting inside query)
    qc = _build_query_cloud(symbol_list)
    qc.query_type = _detect_query_shape(symbol_list, qc, lexicon, evidence)
    qc.focal_symbols = _select_focal_symbols(symbol_list, lexicon, evidence)

    return qc


def _build_query_cloud(symbol_list: List[str]) -> QueryCloud:
    """Contract Section 6: BUILD_QUERY_CLOUD.
    Apply same positional counting logic within the query itself.
    """
    qc = QueryCloud(query_symbols=list(symbol_list))
    n = len(symbol_list)

    for i in range(n):
        anchor = symbol_list[i]
        for offset in range(1, 7):
            before_idx = i - offset
            after_idx = i + offset

            if before_idx >= 0:
                neighbor = symbol_list[before_idx]
                qc.before.setdefault(offset, {}).setdefault(neighbor, 0)
                qc.before[offset][neighbor] += 1

            if after_idx < n:
                neighbor = symbol_list[after_idx]
                qc.after.setdefault(offset, {}).setdefault(neighbor, 0)
                qc.after[offset][neighbor] += 1

    # Pair links: adjacent symbol pairs
    for i in range(n - 1):
        qc.pair_links.append((symbol_list[i], symbol_list[i + 1]))

    return qc


# ══════════════════════════════════════════════════════════════
# SECTION 7: QUERY SHAPE DETECTION
# ══════════════════════════════════════════════════════════════

# Directional cue words (resolved to symbols at runtime)
_FORWARD_CUES = {"cause", "causes", "produce", "produces", "create", "creates",
                  "lead", "leads", "result", "results", "affect", "affects",
                  "does", "do", "make", "makes"}
_BACKWARD_CUES = {"caused", "from", "because", "due", "origin", "source",
                   "why", "precede", "precedes", "before"}
_RELATEDNESS_CUES = {"relate", "related", "relation", "relationship", "connect",
                      "connection", "between", "link", "linked"}
_CONTRAST_CUES = {"difference", "differ", "differs", "different", "versus",
                   "compare", "compared", "contrast", "unlike"}


def _detect_query_shape(
    symbol_list: List[str],
    qc: QueryCloud,
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> QueryType:
    """Contract Section 7: DETECT_QUERY_SHAPE.

    This is routing, NOT POS tagging.
    """
    # Resolve cue words to check presence
    words_in_query = set()
    for sid in symbol_list:
        w = lexicon.get_word(sid)
        if w:
            words_in_query.add(w.lower())

    has_forward = bool(words_in_query & _FORWARD_CUES)
    has_backward = bool(words_in_query & _BACKWARD_CUES)
    has_relate = bool(words_in_query & _RELATEDNESS_CUES)
    has_contrast = bool(words_in_query & _CONTRAST_CUES)

    focal = qc.focal_symbols if qc.focal_symbols else symbol_list
    n_focal = len([s for s in focal if lexicon.get_word(s)])

    if has_contrast and n_focal >= 2:
        return QueryType.CONTRAST
    if has_relate and n_focal >= 2:
        return QueryType.RELATEDNESS
    if has_forward:
        return QueryType.FORWARD_RELATION
    if has_backward:
        return QueryType.BACKWARD_RELATION
    if n_focal >= 2:
        return QueryType.MULTI_ANCHOR
    if n_focal == 1:
        return QueryType.SINGLE_ANCHOR
    return QueryType.UNKNOWN


def _select_focal_symbols(
    symbol_list: List[str],
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> List[str]:
    """Contract Section 6: SELECT_FOCAL_SYMBOLS.

    Score by rarity, stability, and query position relevance.
    """
    scored = []
    for sid in set(symbol_list):
        word = lexicon.get_word(sid)
        if not word:
            continue

        cloud = evidence.get_cloud(sid)
        if cloud:
            # Use basic stability proxy: total_count (more = more stable)
            stability = min(cloud.total_count / 100.0, 1.0)
            # Rarity: appears less = more focal
            rarity = 1.0 / (1.0 + math.log(cloud.total_count + 1))
        else:
            stability = 0.0
            rarity = 1.0  # unknown = rare but untrusted

        # Query position: how central in query?
        positions = [i for i, s in enumerate(symbol_list) if s == sid]
        centrality = 1.0 - abs(sum(positions) / len(positions) - len(symbol_list) / 2) / max(len(symbol_list), 1)

        score = rarity * 0.3 + stability * 0.4 + centrality * 0.3
        scored.append((sid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:5]]


# ══════════════════════════════════════════════════════════════
# SECTION 8: CLOUD SEARCH
# ══════════════════════════════════════════════════════════════

def search_clouds(
    query_cloud: QueryCloud,
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
    max_candidates: int = 50,
) -> List[CloudMatch]:
    """Contract Section 8: SEARCH_CLOUDS."""
    candidate_set = _build_candidate_set(query_cloud, lexicon, evidence)

    if not candidate_set:
        return []

    matches = []
    for cid in list(candidate_set)[:max_candidates]:
        match = _score_cloud_match(query_cloud, cid, lexicon, evidence)
        if match.final_score > 0:
            matches.append(match)

    matches.sort(key=lambda m: m.final_score, reverse=True)
    return matches


def _build_candidate_set(
    query_cloud: QueryCloud,
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> Set[str]:
    """Contract Section 8: BUILD_CANDIDATE_SET."""
    candidates = set()

    for focal_sid in query_cloud.focal_symbols:
        candidates.add(focal_sid)
        # Add top neighbors from this focal symbol's cloud
        neighbors = evidence.get_all_neighbor_symbols(focal_sid)
        candidates.update(neighbors)

    # Also add all query symbols
    candidates.update(query_cloud.query_symbols)

    filtered = set()
    for sid in candidates:
        word = lexicon.get_word(sid)
        if word:
            filtered.add(sid)

    return filtered


# ══════════════════════════════════════════════════════════════
# SECTION 9: LOCAL CLOUD SCORING
# ══════════════════════════════════════════════════════════════

def _score_local_cloud(query_cloud: QueryCloud, candidate_cloud: ContextCloud) -> float:
    """Contract Section 9: SCORE_LOCAL_CLOUD.

    Compare query cloud buckets against candidate cloud buckets.
    Position-weighted overlap.
    """
    score = 0.0
    max_possible = 0.0

    for pos in range(1, 7):
        w = _pos_weight(pos)

        # Before
        q_before = query_cloud.before.get(pos, {})
        c_before = candidate_cloud.before.get(pos, {})
        for sid, q_count in q_before.items():
            c_count = c_before.get(sid, 0)
            overlap = min(q_count, c_count)
            score += overlap * w
            max_possible += q_count * w

        # After
        q_after = query_cloud.after.get(pos, {})
        c_after = candidate_cloud.after.get(pos, {})
        for sid, q_count in q_after.items():
            c_count = c_after.get(sid, 0)
            overlap = min(q_count, c_count)
            score += overlap * w
            max_possible += q_count * w

    return score / max_possible if max_possible > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# SECTION 10: CLOUD-OF-CLOUD SEARCH
# ══════════════════════════════════════════════════════════════

def _score_layered_cloud(
    query_cloud: QueryCloud,
    candidate_sid: str,
    evidence: EvidenceAdapter,
    top_k: int = 5,
) -> float:
    """Contract Section 10: SCORE_LAYERED_CLOUD.

    Second-order matching: compare the clouds of query's neighbors
    against the clouds of candidate's neighbors.
    """
    # Get top neighbors of candidate
    candidate_cloud = evidence.get_cloud(candidate_sid)
    if not candidate_cloud:
        return 0.0

    c_neighbors = _get_top_neighbors(candidate_cloud, top_k)
    q_neighbors = list(set(query_cloud.query_symbols))[:top_k]

    if not c_neighbors or not q_neighbors:
        return 0.0

    score = 0.0
    comparisons = 0

    for q_sid in q_neighbors:
        q_cloud = evidence.get_cloud(q_sid)
        if not q_cloud:
            continue
        for c_sid in c_neighbors:
            c_cloud = evidence.get_cloud(c_sid)
            if not c_cloud:
                continue
            # Second-order: positional overlap between neighbor clouds
            score += _positional_overlap(q_cloud, c_cloud)
            comparisons += 1

    return score / comparisons if comparisons > 0 else 0.0


def _get_top_neighbors(cloud: ContextCloud, top_k: int) -> List[str]:
    """Get top-K neighbor symbol_ids by weighted count."""
    scored: Dict[str, float] = {}
    for pos, bucket in cloud.before.items():
        w = _pos_weight(pos)
        for sid, count in bucket.items():
            scored[sid] = scored.get(sid, 0) + count * w
    for pos, bucket in cloud.after.items():
        w = _pos_weight(pos)
        for sid, count in bucket.items():
            scored[sid] = scored.get(sid, 0) + count * w

    ranked = sorted(scored, key=scored.get, reverse=True)
    return ranked[:top_k]


def _positional_overlap(cloud_a: ContextCloud, cloud_b: ContextCloud) -> float:
    """Positional overlap score between two clouds."""
    score = 0.0
    total = 0.0

    for pos in range(1, 7):
        w = _pos_weight(pos)
        for direction_a, direction_b in [(cloud_a.before, cloud_b.before), (cloud_a.after, cloud_b.after)]:
            a_bucket = direction_a.get(pos, {})
            b_bucket = direction_b.get(pos, {})
            shared = set(a_bucket.keys()) & set(b_bucket.keys())
            for sid in shared:
                score += min(a_bucket[sid], b_bucket[sid]) * w
            total += sum(a_bucket.values()) * w if a_bucket else 0

    return score / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# SECTION 8 (continued): SCORE_CLOUD_MATCH
# ══════════════════════════════════════════════════════════════

def _score_cloud_match(
    query_cloud: QueryCloud,
    candidate_sid: str,
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> CloudMatch:
    """Contract Section 8: SCORE_CLOUD_MATCH."""
    candidate_cloud = evidence.get_cloud(candidate_sid)
    word = lexicon.get_word(candidate_sid) or ""

    if not candidate_cloud:
        return CloudMatch(candidate_symbol_id=candidate_sid, candidate_word=word)

    # Section 9: local cloud scoring
    local_sim = _score_local_cloud(query_cloud, candidate_cloud)

    # Section 10: cloud-of-cloud
    layered_sim = _score_layered_cloud(query_cloud, candidate_sid, evidence)

    # Overlap: what fraction of query symbols appear in candidate's cloud?
    candidate_neighbors = evidence.get_all_neighbor_symbols(candidate_sid)
    query_syms = set(query_cloud.query_symbols)
    overlap = len(query_syms & candidate_neighbors) / len(query_syms) if query_syms else 0

    # Directional: do before/after alignments match query intent?
    directional = _score_directional(query_cloud, candidate_cloud)

    # Stability adjustment
    stability_adj = candidate_cloud.stability.focal_weight if candidate_cloud.stability else 0.0

    # Combine
    final = (
        local_sim * 0.30 +
        layered_sim * 0.20 +
        overlap * 0.20 +
        directional * 0.15 +
        stability_adj * 0.15
    )

    # Build evidence
    evidence_bundle = _diff_query_to_match(query_cloud, candidate_cloud, lexicon)

    return CloudMatch(
        candidate_symbol_id=candidate_sid,
        candidate_word=word,
        local_similarity_score=round(local_sim, 4),
        layered_similarity_score=round(layered_sim, 4),
        overlap_score=round(overlap, 4),
        contrast_score=0.0,  # populated for CONTRAST queries
        directional_match_score=round(directional, 4),
        stability_adjustment=round(stability_adj, 4),
        final_score=round(final, 4),
        evidence=evidence_bundle,
    )


def _score_directional(query_cloud: QueryCloud, candidate_cloud: ContextCloud) -> float:
    """Score how well before/after direction aligns with query intent."""
    q_before_syms = set()
    q_after_syms = set()
    for bucket in query_cloud.before.values():
        q_before_syms.update(bucket.keys())
    for bucket in query_cloud.after.values():
        q_after_syms.update(bucket.keys())

    c_before_syms = set()
    c_after_syms = set()
    for bucket in candidate_cloud.before.values():
        c_before_syms.update(bucket.keys())
    for bucket in candidate_cloud.after.values():
        c_after_syms.update(bucket.keys())

    # Before alignment
    before_match = len(q_before_syms & c_before_syms) / max(len(q_before_syms), 1)
    # After alignment
    after_match = len(q_after_syms & c_after_syms) / max(len(q_after_syms), 1)

    return (before_match + after_match) / 2.0


# ══════════════════════════════════════════════════════════════
# SECTION 12: DIFF CONTRACT
# ══════════════════════════════════════════════════════════════

def _diff_query_to_match(
    query_cloud: QueryCloud,
    candidate_cloud: ContextCloud,
    lexicon: LexiconAdapter,
) -> EvidenceBundle:
    """Contract Section 12: DIFF_QUERY_TO_MATCH."""
    matched = []
    exclusive_query = []
    exclusive_candidate = []
    matched_positions = []

    for pos in range(1, 7):
        for q_dir, c_dir, direction in [
            (query_cloud.before, candidate_cloud.before, "before"),
            (query_cloud.after, candidate_cloud.after, "after"),
        ]:
            q_bucket = q_dir.get(pos, {})
            c_bucket = c_dir.get(pos, {})

            shared = set(q_bucket.keys()) & set(c_bucket.keys())
            q_only = set(q_bucket.keys()) - set(c_bucket.keys())
            c_only = set(c_bucket.keys()) - set(q_bucket.keys())

            for sid in shared:
                word = lexicon.get_word(sid) or sid[:10]
                matched.append({
                    "word": word, "position": pos if direction == "after" else -pos,
                    "query_count": q_bucket[sid], "candidate_count": c_bucket[sid],
                })
                matched_positions.append(pos if direction == "after" else -pos)

            for sid in q_only:
                word = lexicon.get_word(sid) or sid[:10]
                exclusive_query.append({"word": word, "position": pos if direction == "after" else -pos, "source": "query"})

            for sid in c_only:
                word = lexicon.get_word(sid) or sid[:10]
                exclusive_candidate.append({"word": word, "position": pos if direction == "after" else -pos, "source": "candidate"})

    return EvidenceBundle(
        matched_neighbors=matched,
        matched_positions=sorted(set(matched_positions)),
        exclusive_neighbors=exclusive_query,
        diff_neighbors=exclusive_candidate,
        supporting_counts={"matched": len(matched), "query_exclusive": len(exclusive_query), "candidate_exclusive": len(exclusive_candidate)},
    )


# ══════════════════════════════════════════════════════════════
# SECTION 13: ANSWER GENERATION
# ══════════════════════════════════════════════════════════════

def generate_answer(
    query_cloud: QueryCloud,
    ranked_matches: List[CloudMatch],
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> AnswerResult:
    """Contract Section 13: GENERATE_ANSWER.

    Only say what the structure supports.
    No invented claims. No semantic fantasy.
    """
    if not ranked_matches:
        return AnswerResult(
            answer_text="No matching clouds found in evidence.",
            uncertainty_notes=["no_candidates"],
            query_type=query_cloud.query_type,
        )

    best = ranked_matches[0]
    evidence_bundle = best.evidence

    # Section 14: support metrics
    support = _compute_support_metrics(ranked_matches, evidence_bundle)

    # Render based on query type
    answer_text = _render_answer(query_cloud, best, evidence_bundle, ranked_matches, lexicon, evidence)
    uncertainty = _build_uncertainty_notes(ranked_matches, evidence_bundle, support)

    return AnswerResult(
        answer_text=answer_text,
        supporting_symbols=[m["word"] for m in evidence_bundle.matched_neighbors[:10]],
        evidence_bundle=evidence_bundle,
        uncertainty_notes=uncertainty,
        support_metrics=support,
        score=best.final_score,
        query_type=query_cloud.query_type,
        ranked_matches=ranked_matches[:10],
    )


def _render_answer(
    qc: QueryCloud,
    best: CloudMatch,
    evidence: EvidenceBundle,
    ranked: List[CloudMatch],
    lexicon: LexiconAdapter,
    ev_adapter: EvidenceAdapter,
) -> str:
    """Contract Section 13: RENDER_ANSWER. Only what structure supports."""
    cloud = ev_adapter.get_cloud(best.candidate_symbol_id)
    if not cloud:
        return f"'{best.candidate_word}' found but no cloud evidence."

    # Get top neighbors by direction for summary
    top_before = _get_top_neighbors_by_direction(cloud, "before", lexicon, 4)
    top_after = _get_top_neighbors_by_direction(cloud, "after", lexicon, 4)

    if qc.query_type == QueryType.SINGLE_ANCHOR:
        parts = [f"'{best.candidate_word}':"]
        if top_before:
            parts.append(f"preceded by {', '.join(top_before)}.")
        if top_after:
            parts.append(f"Followed by {', '.join(top_after)}.")
        parts.append(f"({cloud.total_count} total counts, score {best.final_score:.3f})")
        return " ".join(parts)

    if qc.query_type == QueryType.FORWARD_RELATION:
        if top_after:
            return f"'{best.candidate_word}' leads to: {', '.join(top_after)}. (score {best.final_score:.3f})"
        return f"'{best.candidate_word}': no strong forward relations in evidence."

    if qc.query_type == QueryType.BACKWARD_RELATION:
        if top_before:
            return f"'{best.candidate_word}' is preceded by: {', '.join(top_before)}. (score {best.final_score:.3f})"
        return f"'{best.candidate_word}': no strong backward relations in evidence."

    if qc.query_type == QueryType.RELATEDNESS:
        if len(ranked) >= 2:
            a, b = ranked[0], ranked[1]
            shared = [m["word"] for m in evidence.matched_neighbors[:5]]
            return (f"'{a.candidate_word}' and '{b.candidate_word}': "
                    f"overlap {best.overlap_score:.1%}. "
                    f"Shared context: {', '.join(shared) if shared else 'none'}.")
        return f"Insufficient evidence for relatedness."

    if qc.query_type == QueryType.CONTRAST:
        if len(ranked) >= 2:
            a, b = ranked[0], ranked[1]
            shared = [m["word"] for m in evidence.matched_neighbors[:3]]
            q_only = [m["word"] for m in evidence.exclusive_neighbors[:3]]
            c_only = [m["word"] for m in evidence.diff_neighbors[:3]]
            parts = [f"'{a.candidate_word}' vs '{b.candidate_word}':"]
            if shared:
                parts.append(f"Shared: {', '.join(shared)}.")
            if q_only:
                parts.append(f"First exclusive: {', '.join(q_only)}.")
            if c_only:
                parts.append(f"Second exclusive: {', '.join(c_only)}.")
            return " ".join(parts)
        return f"Insufficient evidence for contrast."

    if qc.query_type == QueryType.MULTI_ANCHOR:
        words = [lexicon.get_word(s) or "?" for s in qc.focal_symbols[:3]]
        return (f"Multi-anchor: {', '.join(words)}. "
                f"Best match: '{best.candidate_word}' (score {best.final_score:.3f}). "
                f"Cloud has {cloud.total_count} counts.")

    # UNKNOWN fallback
    parts = [f"'{best.candidate_word}' (score {best.final_score:.3f}):"]
    if top_after:
        parts.append(f"after: {', '.join(top_after)}")
    if top_before:
        parts.append(f"before: {', '.join(top_before)}")
    return " ".join(parts)


def _get_top_neighbors_by_direction(
    cloud: ContextCloud,
    direction: str,
    lexicon: LexiconAdapter,
    top_k: int,
) -> List[str]:
    """Get top-K neighbor words from before or after buckets."""
    buckets = cloud.before if direction == "before" else cloud.after
    scored: Dict[str, float] = {}
    for pos, bucket in buckets.items():
        w = _pos_weight(pos)
        for sid, count in bucket.items():
            scored[sid] = scored.get(sid, 0) + count * w

    ranked = sorted(scored, key=scored.get, reverse=True)
    words = []
    for sid in ranked:
        word = lexicon.get_word(sid)
        if word:
            words.append(word)
            if len(words) >= top_k:
                break
    return words


# ══════════════════════════════════════════════════════════════
# SECTION 14: SUPPORT METRICS
# ══════════════════════════════════════════════════════════════

def _compute_support_metrics(ranked: List[CloudMatch], evidence: EvidenceBundle) -> SupportMetrics:
    """Contract Section 14: honest confidence."""
    best = ranked[0] if ranked else None

    overlap_strength = len(evidence.matched_neighbors) / max(
        len(evidence.matched_neighbors) + len(evidence.exclusive_neighbors) + len(evidence.diff_neighbors), 1
    )

    directional = best.directional_match_score if best else 0.0
    stability = best.stability_adjustment if best else 0.0

    # Ambiguity: how close is runner-up?
    if len(ranked) >= 2:
        gap = ranked[0].final_score - ranked[1].final_score
        ambiguity = max(0, 1.0 - gap * 10)  # small gap = high ambiguity
    else:
        ambiguity = 0.0

    query_coverage = best.overlap_score if best else 0.0

    return SupportMetrics(
        overlap_strength=round(overlap_strength, 4),
        directional_consistency=round(directional, 4),
        cloud_stability=round(stability, 4),
        ambiguity=round(ambiguity, 4),
        query_coverage=round(query_coverage, 4),
    )


def _build_uncertainty_notes(
    ranked: List[CloudMatch],
    evidence: EvidenceBundle,
    support: SupportMetrics,
) -> List[str]:
    """Contract Section 13: BUILD_UNCERTAINTY_NOTES."""
    notes = []

    if not ranked or ranked[0].final_score < 0.1:
        notes.append("low structural support")

    if support.ambiguity > 0.7:
        notes.append("ambiguous cloud match — top candidates very close")

    if support.cloud_stability < 0.1:
        notes.append("focal anchor cloud not stable")

    if support.query_coverage < 0.2:
        notes.append("query partially outside known cloud space")

    if len(evidence.matched_neighbors) == 0:
        notes.append("no shared neighbors between query and candidate clouds")

    return notes


# ══════════════════════════════════════════════════════════════
# SECTION 16: QUERY LOOP (top-level entry point)
# ══════════════════════════════════════════════════════════════

def answer_question(
    raw_question: str,
    lexicon: LexiconAdapter,
    evidence: EvidenceAdapter,
) -> AnswerResult:
    """Contract Section 16: ANSWER_QUESTION.

    The full loop: query intake → cloud search → answer generation.
    """
    query_cloud = process_query(raw_question, lexicon, evidence)

    if not query_cloud.focal_symbols:
        return AnswerResult(
            answer_text=f"No known symbols found in query.",
            uncertainty_notes=["all_oov"],
            query_type=query_cloud.query_type,
        )

    ranked_matches = search_clouds(query_cloud, lexicon, evidence)
    result = generate_answer(query_cloud, ranked_matches, lexicon, evidence)
    return result

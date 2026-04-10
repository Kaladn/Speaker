from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from bridges.evidence_store import get_evidence_store
from core.lakespeak.text.normalize import extract_anchors
from security.data_paths import EVIDENCE_DIR, MAPS_DIR, RECEIPTS_DIR

TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.text'}


def make_receipt_id() -> str:
    now = datetime.now(timezone.utc)
    return f"rcpt_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def source_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _sanitize_source_name(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {' ', '-', '_', '/'}:
            cleaned.append('_')
    name = ''.join(cleaned).strip('_')
    return name[:80] or 'document'


def _merge_counts(target: Dict[str, Dict[int, Dict[str, int]]], source: Dict[str, Dict[int, Dict[str, int]]]) -> None:
    for focus_hex, offsets in source.items():
        target_offsets = target.setdefault(focus_hex, {})
        for offset, neighbors in offsets.items():
            target_neighbors = target_offsets.setdefault(int(offset), {})
            for neighbor_hex, count in neighbors.items():
                target_neighbors[neighbor_hex] = target_neighbors.get(neighbor_hex, 0) + int(count)


def _result_without_audit(
    source: str,
    text: str,
    mapped: Dict[str, Any],
    *,
    window: int,
    updated_cells: int | None,
    evidence_cells: int | None,
) -> Dict[str, Any]:
    return {
        'source': source,
        'source_hash': source_hash(text),
        'window': window,
        **mapped['stats'],
        'updated_cells': updated_cells,
        'evidence_cells': evidence_cells,
        'evidence_path': str(EVIDENCE_DIR),
        'receipt_path': None,
        'map_path': None,
    }


def map_text_to_counts(text: str, lexicon: Any, window: int = 6) -> Dict[str, Any]:
    blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    if not blocks and text.strip():
        blocks = [text.strip()]

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_words = 0
    total_symbols = 0
    unknown_words = 0
    unique_symbols = set()

    for block in blocks:
        words = extract_anchors(block)
        total_words += len(words)

        symbols = []
        for word in words:
            symbol = lexicon.word_index.get(word)
            if symbol:
                symbols.append(symbol)
                unique_symbols.add(symbol)
            else:
                unknown_words += 1

        total_symbols += len(symbols)
        if len(symbols) < 2:
            continue

        for idx, focus in enumerate(symbols):
            for offset in range(1, window + 1):
                before_idx = idx - offset
                after_idx = idx + offset
                if before_idx >= 0:
                    counts[focus][-offset][symbols[before_idx]] += 1
                if after_idx < len(symbols):
                    counts[focus][offset][symbols[after_idx]] += 1

    plain_counts = {
        focus_hex: {
            offset: dict(neighbors)
            for offset, neighbors in offsets.items()
        }
        for focus_hex, offsets in counts.items()
    }

    return {
        'counts_by_hex': plain_counts,
        'stats': {
            'blocks': len(blocks),
            'total_words': total_words,
            'total_symbols': total_symbols,
            'unknown_words': unknown_words,
            'coverage': round((total_symbols / total_words) * 100, 2) if total_words else 0.0,
            'unique_symbols': len(unique_symbols),
            'focus_cells': len(plain_counts),
            'window': window,
        },
    }


def _write_receipt_and_map(
    receipt_id: str,
    source: str,
    text: str,
    mapped: Dict[str, Any],
    *,
    window: int,
    write_map: bool,
    updated_cells: int | None,
    evidence_cells: int | None,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    receipt = {
        'receipt_id': receipt_id,
        'source': source,
        'source_hash': source_hash(text),
        'created_at_utc': timestamp,
        'window': window,
        **mapped['stats'],
        'updated_cells': updated_cells,
        'evidence_cells': evidence_cells,
        'evidence_path': str(EVIDENCE_DIR),
    }

    safe_name = _sanitize_source_name(source)
    receipt_path = RECEIPTS_DIR / f'{receipt_id}.{safe_name}.receipt.json'
    _write_json(receipt_path, receipt)

    map_path = None
    if write_map:
        map_report = {
            'receipt_id': receipt_id,
            'source': source,
            'source_hash': receipt['source_hash'],
            'created_at_utc': timestamp,
            'window': window,
            'stats': mapped['stats'],
            'counts_by_hex': mapped['counts_by_hex'],
        }
        map_path = MAPS_DIR / f'{receipt_id}.{safe_name}.map.json'
        _write_json(map_path, map_report)

    result = dict(receipt)
    result['receipt_path'] = str(receipt_path)
    result['map_path'] = str(map_path) if map_path else None
    return result


def ingest_text(
    text: str,
    lexicon: Any,
    *,
    source: str = 'inline',
    window: int = 6,
    write_map: bool = True,
    write_audit: bool = True,
) -> Dict[str, Any]:
    receipt_id = make_receipt_id()
    mapped = map_text_to_counts(text, lexicon, window=window)
    counts_by_hex = mapped['counts_by_hex']

    store = get_evidence_store()
    updated_cells = store.append_counts(counts_by_hex) if counts_by_hex else 0
    health = store.health()
    if not write_audit:
        return _result_without_audit(
            source,
            text,
            mapped,
            window=window,
            updated_cells=updated_cells,
            evidence_cells=health['cells'],
        )

    return _write_receipt_and_map(
        receipt_id,
        source,
        text,
        mapped,
        window=window,
        write_map=write_map,
        updated_cells=updated_cells,
        evidence_cells=health['cells'],
    )


def ingest_file(
    path: Path,
    lexicon: Any,
    *,
    source: str | None = None,
    window: int = 6,
    write_map: bool = True,
    write_audit: bool = True,
) -> Dict[str, Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding='utf-8')
    return ingest_text(
        text,
        lexicon,
        source=source or str(file_path.resolve()),
        window=window,
        write_map=write_map,
        write_audit=write_audit,
    )


def ingest_folder(
    folder: Path,
    lexicon: Any,
    *,
    window: int = 6,
    write_map: bool = True,
    flush_every: int = 10,
    recursive: bool = False,
    write_audit: bool = True,
    start_at: int = 0,
) -> Dict[str, Any]:
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f'Folder not found: {base}')

    walker = base.rglob('*') if recursive else base.glob('*')
    all_files = [path for path in sorted(walker) if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS]
    if not all_files:
        raise FileNotFoundError(f'No supported text files found in {base}')
    if start_at < 0:
        raise ValueError('start_at must be >= 0')

    files = all_files[start_at:]
    total = len(all_files)
    remaining = len(files)
    processed = 0
    errors = 0
    total_words = 0
    total_symbols = 0
    updated_cells_total = 0
    receipts: List[str] = []
    failures: List[Dict[str, str]] = []
    pending_counts: Dict[str, Dict[int, Dict[str, int]]] = {}
    flushes = 0
    store = get_evidence_store()

    def flush_pending() -> None:
        nonlocal pending_counts, updated_cells_total, flushes
        if not pending_counts:
            return
        updated = store.append_counts(pending_counts)
        updated_cells_total += updated
        flushes += 1
        pending_counts = {}

    print(f'Text files: {total}', flush=True)
    print(f'Start at: {start_at}', flush=True)
    print(f'Remaining: {remaining}', flush=True)
    print(f'Flush every: {flush_every}', flush=True)
    print(f'Audit trail: {"on" if write_audit else "off"}', flush=True)

    for file_path in files:
        source = str(file_path.resolve())
        try:
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            mapped = map_text_to_counts(text, lexicon, window=window)
            _merge_counts(pending_counts, mapped['counts_by_hex'])

            processed += 1
            total_words += int(mapped['stats'].get('total_words', 0))
            total_symbols += int(mapped['stats'].get('total_symbols', 0))

            if write_audit:
                receipt_id = make_receipt_id()
                result = _write_receipt_and_map(
                    receipt_id,
                    source,
                    text,
                    mapped,
                    window=window,
                    write_map=write_map,
                    updated_cells=None,
                    evidence_cells=None,
                )
                receipts.append(result['receipt_id'])

            if processed % flush_every == 0:
                flush_pending()
                print(
                    f'[{start_at + processed}/{total}] flushes={flushes} ev={store.count():,} receipts={len(receipts):,}',
                    flush=True,
                )
        except Exception as exc:
            errors += 1
            failures.append({'source': source, 'error': str(exc)[:200]})
            if errors <= 5:
                print(f'ERR {source}: {str(exc)[:200]}', flush=True)

    flush_pending()
    print(f'Done folder ingest: processed={processed} errors={errors} ev={store.count():,}', flush=True)

    return {
        'folder': str(base),
        'processed_files': processed,
        'start_at': start_at,
        'remaining_files': remaining,
        'errors': errors,
        'total_words': total_words,
        'total_symbols': total_symbols,
        'coverage': round((total_symbols / total_words) * 100, 2) if total_words else 0.0,
        'updated_cells_total': updated_cells_total,
        'receipts_written': len(receipts),
        'sample_receipts': receipts[:10],
        'failures': failures[:10],
        'flushes': flushes,
        'recursive': recursive,
        'audit_trail': write_audit,
        'evidence': store.health(),
    }


def _conversation_text(conversation: Dict[str, Any]) -> str:
    parts: List[str] = []
    for node in conversation.get('mapping', {}).values():
        if not isinstance(node, dict):
            continue
        message = node.get('message')
        if not isinstance(message, dict):
            continue
        content = message.get('content') or {}
        if not isinstance(content, dict):
            continue
        for part in content.get('parts', []):
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
    return '\n\n'.join(parts)


def ingest_chatgpt_export(
    export_dir: Path,
    lexicon: Any,
    *,
    window: int = 6,
    write_map: bool = True,
    flush_every: int = 10,
    write_audit: bool = True,
) -> Dict[str, Any]:
    base = Path(export_dir)
    files = sorted(base.glob('conversations-*.json'))
    if not files:
        raise FileNotFoundError(f'No conversations-*.json files found in {base}')

    conversations: List[tuple[str, int, Dict[str, Any]]] = []
    for file_path in files:
        data = json.loads(file_path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            continue
        for idx, conversation in enumerate(data):
            conversations.append((file_path.name, idx, conversation))

    total = len(conversations)
    processed = 0
    errors = 0
    total_words = 0
    total_symbols = 0
    updated_cells_total = 0
    receipts: List[str] = []
    failures: List[Dict[str, str]] = []
    pending_counts: Dict[str, Dict[int, Dict[str, int]]] = {}
    flushes = 0
    store = get_evidence_store()

    def flush_pending() -> None:
        nonlocal pending_counts, updated_cells_total, flushes
        if not pending_counts:
            return
        updated = store.append_counts(pending_counts)
        updated_cells_total += updated
        flushes += 1
        pending_counts = {}

    print(f'Conversations: {total}', flush=True)
    print(f'Flush every: {flush_every}', flush=True)
    print(f'Audit trail: {"on" if write_audit else "off"}', flush=True)

    for file_name, idx, conversation in conversations:
        title = str(conversation.get('title') or f'conversation_{processed + errors + 1}')
        text = _conversation_text(conversation)
        if not text.strip():
            continue
        source = f'chat_export/{file_name}/{idx:03d}/{title[:80]}'
        try:
            mapped = map_text_to_counts(text, lexicon, window=window)
            _merge_counts(pending_counts, mapped['counts_by_hex'])
            processed += 1
            total_words += int(mapped['stats'].get('total_words', 0))
            total_symbols += int(mapped['stats'].get('total_symbols', 0))
            if write_audit:
                receipt_id = make_receipt_id()
                result = _write_receipt_and_map(
                    receipt_id,
                    source,
                    text,
                    mapped,
                    window=window,
                    write_map=write_map,
                    updated_cells=None,
                    evidence_cells=None,
                )
                receipts.append(result['receipt_id'])
            if processed % flush_every == 0:
                flush_pending()
                print(
                    f'[{start_at + processed}/{total}] flushes={flushes} ev={store.count():,} receipts={len(receipts):,}',
                    flush=True,
                )
        except Exception as exc:
            errors += 1
            failures.append({'source': source, 'error': str(exc)[:200]})
            if errors <= 5:
                print(f'ERR {source}: {str(exc)[:200]}', flush=True)

    flush_pending()
    print(f'Done export ingest: processed={processed} errors={errors} ev={store.count():,}', flush=True)

    return {
        'export_dir': str(base),
        'processed_conversations': processed,
        'errors': errors,
        'total_words': total_words,
        'total_symbols': total_symbols,
        'coverage': round((total_symbols / total_words) * 100, 2) if total_words else 0.0,
        'updated_cells_total': updated_cells_total,
        'receipts_written': len(receipts),
        'sample_receipts': receipts[:10],
        'failures': failures[:10],
        'flushes': flushes,
        'audit_trail': write_audit,
        'evidence': store.health(),
    }


def health() -> Dict[str, Any]:
    store = get_evidence_store()
    return {
        'evidence': store.health(),
        'maps': len(list(MAPS_DIR.glob('*.map.json'))),
        'receipts': len(list(RECEIPTS_DIR.glob('*.receipt.json'))),
    }
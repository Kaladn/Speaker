from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.lakespeak.text.normalize import extract_anchors
from security.data_paths import REPORTS_DIR
from standalone616.lexicon import append_overlay_words, load_lexicon, save_alias_map

TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.text'}
_NUMBER_RE = re.compile(r'^\d+$')
_HEX_RE = re.compile(r'^0x[0-9a-f]+$')
_GUID_RE = re.compile(r'^[0-9a-f]{16,}$')
_UNICODE_PREFIX_RE = re.compile(r'^(?:u[0-9a-f]{4})+(?=[a-z_])')
_UNICODE_CODE_RE = re.compile(r'^u[0-9a-f]{4,}$')


def _sanitize_name(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {' ', '-', '_', '/'}:
            cleaned.append('_')
    name = ''.join(cleaned).strip('_')
    return name[:80] or 'folder'


def _iter_text_files(folder: Path, recursive: bool) -> List[Path]:
    base = Path(folder)
    walker: Iterable[Path] = base.rglob('*') if recursive else base.glob('*')
    return [path for path in sorted(walker) if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def _clean_unknown_token(token: str, lexicon: Any) -> tuple[str | None, str | None]:
    value = token.strip().lower()
    was_quoted = value.startswith("'") or value.endswith("'")

    cleaned = _UNICODE_PREFIX_RE.sub('', value)
    if cleaned.endswith("'s") and len(cleaned) > 2:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("s'") and len(cleaned) > 2:
        cleaned = cleaned[:-1]
    cleaned = cleaned.strip("'")

    if _UNICODE_PREFIX_RE.match(value) and cleaned and re.fullmatch(r'[0-9a-f]+', cleaned):
        return None, 'unicode_code'

    if not cleaned:
        return None, 'empty_after_clean'
    if set(cleaned) == {'_'}:
        return None, 'underscore_only'
    if any(ord(ch) > 127 for ch in cleaned):
        return None, 'non_ascii'
    if _NUMBER_RE.fullmatch(cleaned):
        return None, 'number'
    if _HEX_RE.fullmatch(cleaned):
        return None, 'hex'
    if _GUID_RE.fullmatch(cleaned):
        return None, 'guid'
    if _UNICODE_CODE_RE.fullmatch(cleaned):
        return None, 'unicode_code'
    if len(cleaned) == 1:
        return None, 'single_char'
    return cleaned, None


def prepare_folder(
    folder: Path,
    lexicon: Any,
    *,
    recursive: bool = False,
    apply: bool = True,
) -> Dict[str, Any]:
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f'Folder not found: {base}')

    files = _iter_text_files(base, recursive)
    if not files:
        raise FileNotFoundError(f'No supported text files found in {base}')

    raw_unknown = Counter()
    known_before = 0
    total_words = 0

    for file_path in files:
        text = file_path.read_text(encoding='utf-8', errors='ignore')
        for token in extract_anchors(text):
            total_words += 1
            if token in lexicon.word_index:
                known_before += 1
            else:
                raw_unknown[token] += 1

    alias_updates: Dict[str, str] = {}
    corrected_to_known = Counter()
    additions = Counter()
    ignored_unique = Counter()
    ignored_total = Counter()

    for token, count in raw_unknown.items():
        cleaned, ignore_reason = _clean_unknown_token(token, lexicon)
        if ignore_reason:
            ignored_unique[ignore_reason] += 1
            ignored_total[ignore_reason] += count
            continue

        if cleaned != token:
            alias_updates[token] = cleaned

        if cleaned in lexicon.word_index:
            corrected_to_known[token] = count
        else:
            additions[cleaned] += count

    corrected_known_total = sum(corrected_to_known.values())
    additions_total = sum(additions.values())
    coverage_before = round((known_before / total_words) * 100, 2) if total_words else 0.0
    coverage_after = round(((known_before + corrected_known_total + additions_total) / total_words) * 100, 2) if total_words else 0.0

    alias_rows = []
    for raw_token, cleaned in alias_updates.items():
        alias_rows.append(
            {
                'raw_token': raw_token,
                'cleaned_token': cleaned,
                'count': raw_unknown[raw_token],
                'resolution': 'known' if cleaned in lexicon.word_index else 'overlay',
            }
        )
    alias_rows.sort(key=lambda row: (-int(row['count']), row['raw_token']))

    addition_rows = [
        {'token': token, 'count': count}
        for token, count in additions.most_common()
    ]

    report_slug = _sanitize_name(base.name)
    summary_path = REPORTS_DIR / f'{report_slug}_dry_ingest_report.json'
    uncovered_path = REPORTS_DIR / f'{report_slug}_uncovered_words.tsv'
    alias_path = REPORTS_DIR / f'{report_slug}_alias_corrections.tsv'

    aliases_written = 0
    overlay_words_added = 0
    post_prepare_word_count = lexicon.word_count
    if apply:
        aliases_written = save_alias_map(alias_updates)
        overlay_words_added = append_overlay_words([row['token'] for row in addition_rows], lexicon)
        refreshed = load_lexicon(lexicon.lexicon_root, force_reload=True)
        post_prepare_word_count = refreshed.word_count

    summary = {
        'folder': str(base),
        'recursive': recursive,
        'files_processed': len(files),
        'total_words': total_words,
        'known_before': known_before,
        'raw_unknown_unique': len(raw_unknown),
        'raw_unknown_total': sum(raw_unknown.values()),
        'coverage_before': coverage_before,
        'corrected_to_known_unique': len(corrected_to_known),
        'corrected_to_known_total': corrected_known_total,
        'candidate_additions_unique': len(additions),
        'candidate_additions_total': additions_total,
        'ignored_unique': dict(sorted(ignored_unique.items())),
        'ignored_total': dict(sorted(ignored_total.items())),
        'estimated_coverage_after': coverage_after,
        'aliases_written': aliases_written,
        'overlay_words_added': overlay_words_added,
        'post_prepare_word_count': post_prepare_word_count,
        'top_aliases': alias_rows[:200],
        'top_additions': addition_rows[:200],
        'top_raw_unknowns': [
            {'token': token, 'count': count}
            for token, count in raw_unknown.most_common(200)
        ],
        'report_paths': {
            'summary': str(summary_path),
            'uncovered_words': str(uncovered_path),
            'alias_corrections': str(alias_path),
        },
    }

    _write_json(summary_path, summary)
    _write_lines(
        uncovered_path,
        ['token\tcount'] + [f"{row['token']}\t{row['count']}" for row in addition_rows],
    )
    _write_lines(
        alias_path,
        ['raw_token\tcleaned_token\tcount\tresolution']
        + [
            f"{row['raw_token']}\t{row['cleaned_token']}\t{row['count']}\t{row['resolution']}"
            for row in alias_rows
        ],
    )
    return summary
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from standalone616.lexicon import (
    append_overlay_words,
    load_lexicon,
    save_alias_map,
)
from standalone616.config import load_settings


REQUIRED_HEADER = ('token', 'count', 'suggested_target', 'reason', 'absolute_correct_version')


def _parse_tsv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open('r', encoding='utf-8') as handle:
        header = handle.readline().rstrip('\n').split('\t')
        if tuple(header) != REQUIRED_HEADER:
            raise ValueError(
                f'Unexpected header. Expected {REQUIRED_HEADER}, got {tuple(header)}'
            )
        for line_no, raw in enumerate(handle, start=2):
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 5:
                raise ValueError(f'Line {line_no}: expected 5 fields, got {len(fields)}')
            token = fields[0].strip().lower()
            correct = fields[4].strip()
            if not token or not correct:
                continue
            rows.append((token, correct))
    return rows


def apply_corrections(tsv_path: Path) -> Dict[str, int]:
    print(f'[apply-corrections] reading {tsv_path}', flush=True)
    rows = _parse_tsv(tsv_path)
    print(f'[apply-corrections] parsed {len(rows)} rows', flush=True)

    settings = load_settings()
    print(f'[apply-corrections] loading lexicon from {settings.lexicon_root}', flush=True)
    lexicon = load_lexicon(settings.lexicon_root, force_reload=True)
    print(f'[apply-corrections] lexicon ready: {lexicon.word_count} words', flush=True)

    delete_tokens: List[str] = []
    alias_updates: Dict[str, str] = {}
    overlay_words: List[str] = []

    for token, correct in rows:
        if correct == 'DELETE':
            delete_tokens.append(token)
            continue
        target = correct.strip().lower()
        if target == token:
            overlay_words.append(token)
        else:
            alias_updates[token] = target
            if target not in lexicon.word_index:
                overlay_words.append(target)

    print(
        f'[apply-corrections] classified: '
        f'{len(alias_updates)} aliases, '
        f'{len(overlay_words)} overlay-candidates, '
        f'{len(delete_tokens)} deletes (skipped)',
        flush=True,
    )

    print('[apply-corrections] writing alias_map.json', flush=True)
    aliases_added = save_alias_map(alias_updates)

    print('[apply-corrections] writing local_overlay.json', flush=True)
    overlay_added = append_overlay_words(overlay_words, lexicon)

    print('[apply-corrections] reloading lexicon to verify', flush=True)
    refreshed = load_lexicon(settings.lexicon_root, force_reload=True)

    summary = {
        'rows_total': len(rows),
        'aliases_in_input': len(alias_updates),
        'aliases_actually_added': aliases_added,
        'overlay_in_input': len(overlay_words),
        'overlay_actually_added': overlay_added,
        'deletes_skipped': len(delete_tokens),
        'lexicon_word_count_after': refreshed.word_count,
        'alias_count_after': refreshed.alias_count,
    }

    print('[apply-corrections] done', flush=True)
    for key, value in summary.items():
        print(f'  {key}: {value}', flush=True)

    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='standalone616.apply_corrections')
    parser.add_argument('tsv', help='Path to corrected TSV with absolute_correct_version column')
    args = parser.parse_args(argv)

    tsv_path = Path(args.tsv).expanduser().resolve()
    if not tsv_path.exists():
        print(f'ERROR: file not found: {tsv_path}', file=sys.stderr)
        return 2

    apply_corrections(tsv_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

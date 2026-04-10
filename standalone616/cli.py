from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from core.reasoning_616.engine import LexiconAdapter, EvidenceAdapter, answer_question
from standalone616.config import load_settings
from standalone616.lexicon import load_lexicon
from standalone616.pipeline import health, ingest_chatgpt_export, ingest_file, ingest_folder, ingest_text
from standalone616.prepare import prepare_folder


def _dump(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_runtime(*, force_reload: bool = False):
    settings = load_settings()
    lexicon = load_lexicon(settings.lexicon_root, force_reload=force_reload)
    return settings, lexicon


def cmd_info(_args: argparse.Namespace) -> int:
    settings, lexicon = _load_runtime()
    _dump(
        {
            'project_root': str(Path(__file__).resolve().parents[1]),
            'lexicon_root': str(settings.lexicon_root),
            'data_root': str(settings.data_root),
            'window': settings.window,
            'lexicon_words': lexicon.word_count,
            'alias_count': lexicon.alias_count,
            **health(),
        }
    )
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    _settings, lexicon = _load_runtime()
    entry = lexicon.lookup(args.word)
    if entry is None:
        _dump({'word': args.word, 'found': False})
        return 0
    _dump(
        {
            'word': args.word,
            'found': True,
            'symbol': entry.symbol,
            'status': entry.status,
            'payload': entry.payload,
        }
    )
    return 0


def cmd_prepare_folder(args: argparse.Namespace) -> int:
    _settings, lexicon = _load_runtime(force_reload=True)
    result = prepare_folder(
        Path(args.path),
        lexicon,
        recursive=args.recursive,
        apply=not args.dry_only,
    )
    _dump(result)
    return 0


def cmd_ingest_file(args: argparse.Namespace) -> int:
    settings, lexicon = _load_runtime(force_reload=args.force_reload_lexicon)
    result = ingest_file(
        Path(args.path),
        lexicon,
        source=args.source,
        window=args.window or settings.window,
        write_map=not args.no_map,
        write_audit=not args.no_audit,
    )
    _dump(result)
    return 0


def cmd_ingest_folder(args: argparse.Namespace) -> int:
    settings, lexicon = _load_runtime(force_reload=args.force_reload_lexicon)
    result = ingest_folder(
        Path(args.path),
        lexicon,
        window=args.window or settings.window,
        write_map=not args.no_map,
        flush_every=args.flush_every,
        recursive=args.recursive,
        write_audit=not args.no_audit,
        start_at=args.start_at,
    )
    _dump(result)
    return 0


def cmd_ingest_text(args: argparse.Namespace) -> int:
    settings, lexicon = _load_runtime(force_reload=args.force_reload_lexicon)
    result = ingest_text(
        args.text,
        lexicon,
        source=args.source or 'inline',
        window=args.window or settings.window,
        write_map=not args.no_map,
        write_audit=not args.no_audit,
    )
    _dump(result)
    return 0


def cmd_ingest_export(args: argparse.Namespace) -> int:
    settings, lexicon = _load_runtime(force_reload=args.force_reload_lexicon)
    result = ingest_chatgpt_export(
        Path(args.path),
        lexicon,
        window=args.window or settings.window,
        write_map=not args.no_map,
        flush_every=args.flush_every,
        write_audit=not args.no_audit,
    )
    _dump(result)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    _settings, lexicon = _load_runtime(force_reload=args.force_reload_lexicon)
    result = answer_question(args.question, LexiconAdapter(lexicon), EvidenceAdapter())
    supporting_words = [
        lexicon.entries[symbol].word
        for symbol in result.supporting_symbols
        if symbol in lexicon.entries and lexicon.entries[symbol].word
    ]
    ranked_matches: List[Dict[str, Any]] = []
    for match in result.ranked_matches[:5]:
        ranked_matches.append(
            {
                'symbol': match.candidate_symbol_id,
                'word': match.candidate_word,
                'score': round(match.final_score, 4),
                'overlap': round(match.overlap_score, 4),
                'directional': round(match.directional_match_score, 4),
            }
        )

    _dump(
        {
            'question': args.question,
            'query_type': result.query_type.value,
            'answer': result.answer_text,
            'score': round(result.score, 4),
            'supporting_symbols': result.supporting_symbols,
            'supporting_words': supporting_words,
            'support_metrics': asdict(result.support_metrics),
            'uncertainty_notes': result.uncertainty_notes,
            'ranked_matches': ranked_matches,
        }
    )
    return 0


def cmd_ui(_args: argparse.Namespace) -> int:
    from standalone616.ui import main as launch_ui

    return launch_ui()


def cmd_health(_args: argparse.Namespace) -> int:
    _dump(health())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='standalone616')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('info', help='Show standalone configuration and data health')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('lookup', help='Look up one word in the lexicon')
    p.add_argument('word')
    p.set_defaults(func=cmd_lookup)

    p = sub.add_parser('prepare-folder', help='Dry-ingest a folder, clean unknowns, and update the local standalone lexicon overlay')
    p.add_argument('path')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--dry-only', action='store_true')
    p.set_defaults(func=cmd_prepare_folder)

    p = sub.add_parser('ingest-file', help='Ingest a UTF-8 text file into local evidence')
    p.add_argument('path')
    p.add_argument('--source')
    p.add_argument('--window', type=int)
    p.add_argument('--no-map', action='store_true')
    p.add_argument('--no-audit', action='store_true')
    p.add_argument('--force-reload-lexicon', action='store_true')
    p.set_defaults(func=cmd_ingest_file)

    p = sub.add_parser('ingest-folder', help='Ingest .txt/.md files from a folder into local evidence')
    p.add_argument('path')
    p.add_argument('--window', type=int)
    p.add_argument('--flush-every', type=int, default=10)
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--start-at', type=int, default=0)
    p.add_argument('--no-map', action='store_true')
    p.add_argument('--no-audit', action='store_true')
    p.add_argument('--force-reload-lexicon', action='store_true')
    p.set_defaults(func=cmd_ingest_folder)

    p = sub.add_parser('ingest-text', help='Ingest inline text into local evidence')
    p.add_argument('--text', required=True)
    p.add_argument('--source')
    p.add_argument('--window', type=int)
    p.add_argument('--no-map', action='store_true')
    p.add_argument('--no-audit', action='store_true')
    p.add_argument('--force-reload-lexicon', action='store_true')
    p.set_defaults(func=cmd_ingest_text)

    p = sub.add_parser('ingest-export', help='Ingest a ChatGPT export folder into local evidence')
    p.add_argument('path')
    p.add_argument('--window', type=int)
    p.add_argument('--flush-every', type=int, default=10)
    p.add_argument('--no-map', action='store_true')
    p.add_argument('--no-audit', action='store_true')
    p.add_argument('--force-reload-lexicon', action='store_true')
    p.set_defaults(func=cmd_ingest_export)

    p = sub.add_parser('query', help='Ask the 6-1-6 reasoning engine a question')
    p.add_argument('--question', required=True)
    p.add_argument('--force-reload-lexicon', action='store_true')
    p.set_defaults(func=cmd_query)

    p = sub.add_parser('ui', help='Launch the simple desktop reasoning UI')
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser('health', help='Show local evidence/maps/receipts health')
    p.set_defaults(func=cmd_health)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
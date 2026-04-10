from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from security.data_paths import ALIAS_MAP_PATH, OVERLAY_PATH


@dataclass
class LexiconEntry:
    word: Optional[str]
    symbol: Optional[str]
    payload: Dict[str, Any]
    status: str = 'ASSIGNED'


class LexiconRuntime:
    def __init__(self, lexicon_root: Path):
        self.lexicon_root = Path(lexicon_root)
        self.word_index: Dict[str, str] = {}
        self.entries: Dict[str, LexiconEntry] = {}
        self.alias_map: Dict[str, str] = {}
        self.loaded = False

    def load(self) -> 'LexiconRuntime':
        self.word_index = {}
        self.entries = {}
        self.alias_map = {}

        self.lexicon_root.mkdir(parents=True, exist_ok=True)
        files = sorted(self.lexicon_root.glob('canonical_*.json'))
        if not files:
            stub = self.lexicon_root / 'canonical_MISC.json'
            stub.write_text('[]', encoding='utf-8')
            files = [stub]

        for file_path in files:
            self._load_records(_read_json(file_path, default=[]))

        self._load_overlay()
        self._load_aliases()
        self.loaded = True
        return self

    def _load_records(self, records: Any) -> None:
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            word = str(record.get('word') or '').strip().lower()
            symbol = str(record.get('symbol') or record.get('hex') or '').strip()
            if not word or not symbol:
                continue
            payload = dict(record)
            self.word_index[word] = symbol
            self.entries[symbol] = LexiconEntry(
                word=word,
                symbol=symbol,
                payload=payload,
                status=str(payload.get('status', 'ASSIGNED')).upper(),
            )

    def _load_overlay(self) -> None:
        self._load_records(_read_json(OVERLAY_PATH, default=[]))

    def _load_aliases(self) -> None:
        raw = load_alias_map()
        for alias, canonical in raw.items():
            canonical_symbol = self.word_index.get(canonical)
            if not canonical_symbol:
                continue
            self.alias_map[alias] = canonical
            self.word_index.setdefault(alias, canonical_symbol)

    @property
    def word_count(self) -> int:
        return len(self.entries)

    @property
    def alias_count(self) -> int:
        return len(self.alias_map)

    def resolve_token(self, token: str) -> str:
        key = token.strip().lower()
        return self.alias_map.get(key, key)

    def lookup(self, word: str) -> Optional[LexiconEntry]:
        symbol = self.word_index.get(word.strip().lower())
        if not symbol:
            return None
        return self.entries.get(symbol)


def _read_json(path: Path, *, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def load_alias_map() -> Dict[str, str]:
    raw = _read_json(ALIAS_MAP_PATH, default={})
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in raw.items():
        misspelling = str(key or '').strip().lower()
        canonical = str(value or '').strip().lower()
        if misspelling and canonical and misspelling != canonical:
            result[misspelling] = canonical
    return result


def save_alias_map(updates: Dict[str, str]) -> int:
    existing = load_alias_map()
    changed = 0
    for misspelling, canonical in sorted(updates.items()):
        key = str(misspelling or '').strip().lower()
        value = str(canonical or '').strip().lower()
        if not key or not value or key == value:
            continue
        if existing.get(key) != value:
            existing[key] = value
            changed += 1
    ALIAS_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIAS_MAP_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
    _CACHE.clear()
    return changed


def load_overlay_records() -> list[Dict[str, Any]]:
    raw = _read_json(OVERLAY_PATH, default=[])
    if not isinstance(raw, list):
        return []
    return [record for record in raw if isinstance(record, dict)]


def _next_symbol(word: str, taken_symbols: set[str]) -> str:
    digest = hashlib.sha256(word.encode('utf-8')).hexdigest().upper()
    for size in (16, 20, 24, 28, 32, 40, 48, 56, 64):
        candidate = '0x' + digest[:size]
        if candidate not in taken_symbols:
            return candidate
    raise RuntimeError(f'Unable to allocate unique symbol for {word!r}')


def append_overlay_words(words: list[str], lexicon: LexiconRuntime) -> int:
    existing_records = load_overlay_records()
    overlay_words = {
        str(record.get('word') or '').strip().lower(): record
        for record in existing_records
        if str(record.get('word') or '').strip()
    }
    taken_symbols = set(lexicon.entries.keys())
    for record in existing_records:
        symbol = str(record.get('symbol') or record.get('hex') or '').strip()
        if symbol:
            taken_symbols.add(symbol)

    added = 0
    for raw_word in sorted({str(word or '').strip().lower() for word in words if str(word or '').strip()}):
        if raw_word in lexicon.word_index or raw_word in overlay_words:
            continue
        symbol = _next_symbol(raw_word, taken_symbols)
        taken_symbols.add(symbol)
        existing_records.append(
            {
                'word': raw_word,
                'hex': symbol,
                'symbol': symbol,
                'binary': '',
                'font_symbol': '',
                'status': 'LOCAL',
                'source': 'local_overlay',
            }
        )
        overlay_words[raw_word] = existing_records[-1]
        added += 1

    existing_records.sort(key=lambda record: str(record.get('word') or ''))
    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY_PATH.write_text(json.dumps(existing_records, ensure_ascii=False, indent=2), encoding='utf-8')
    _CACHE.clear()
    return added


_CACHE: Dict[str, LexiconRuntime] = {}


def load_lexicon(lexicon_root: Path, *, force_reload: bool = False) -> LexiconRuntime:
    resolved = str(Path(lexicon_root).resolve())
    runtime = None if force_reload else _CACHE.get(resolved)
    if runtime is None:
        runtime = LexiconRuntime(Path(resolved)).load()
        _CACHE[resolved] = runtime
    return runtime
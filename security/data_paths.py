from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'standalone616.json'


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_local_path(raw: str | None, default_name: str) -> Path:
    value = (raw or default_name).strip()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


_CONFIG = _load_config()
DATA_ROOT = _resolve_local_path(_CONFIG.get('data_root'), 'data')
EVIDENCE_DIR = DATA_ROOT / 'evidence'
MAPS_DIR = DATA_ROOT / 'maps'
RECEIPTS_DIR = DATA_ROOT / 'receipts'
EXPORTS_DIR = DATA_ROOT / 'exports'
REPORTS_DIR = DATA_ROOT / 'reports'
LEXICON_STATE_DIR = DATA_ROOT / 'lexicon'
OVERLAY_PATH = LEXICON_STATE_DIR / 'local_overlay.json'
ALIAS_MAP_PATH = LEXICON_STATE_DIR / 'alias_map.json'

for _path in (DATA_ROOT, EVIDENCE_DIR, MAPS_DIR, RECEIPTS_DIR, EXPORTS_DIR, REPORTS_DIR, LEXICON_STATE_DIR):
    _path.mkdir(parents=True, exist_ok=True)
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "standalone616.json"


@dataclass(frozen=True)
class Settings:
    lexicon_root: Path
    data_root: Path
    window: int = 6


def load_settings() -> Settings:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Standalone config not found: {CONFIG_PATH}")

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    lexicon_root = Path(raw["lexicon_root"]).expanduser()
    if not lexicon_root.is_absolute():
        lexicon_root = (PROJECT_ROOT / lexicon_root).resolve()

    data_root = Path(raw.get("data_root", "data")).expanduser()
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    return Settings(
        lexicon_root=lexicon_root,
        data_root=data_root,
        window=int(raw.get("window", 6)),
    )

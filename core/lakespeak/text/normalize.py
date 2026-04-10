"""Canonical text normalization for LakeSpeak.

ONE definition of anchor extraction, used everywhere.

Rules (ANCHOR_VERSION = "v2"):
  1. Unicode NFKC normalization
  2. Lowercase
  3. Split on whitespace
  4. Keep words exactly as they appear after lowercase
  5. No punctuation stripping — words with apostrophes are words
  6. No boundary stripping — 'don't' stays 'don't'

Words are words. don't is a word. isn't is a word. they're is a word.
The lexicon has them. The anchor extractor must not destroy them.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spellcheck import AliasMap, SpellChecker, SpellResult

_LOGGER = logging.getLogger(__name__)

ANCHOR_VERSION = "v3"

# Emoji → emjnull (single canonical placeholder, before any other processing)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+", re.UNICODE,
)

# Split on anything that isn't a letter, digit, or apostrophe
# This keeps contractions intact: don't, isn't, they're, o'clock
_SPLIT_RE = re.compile(r"[^\w']+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize raw text to canonical lowercase string.

    Steps: emoji→emjnull → NFKC → lowercase → collapse whitespace → strip.
    """
    text = _EMOJI_RE.sub(" emjnull ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _SPLIT_RE.sub(" ", text)
    return text.strip()


def extract_anchors(text: str) -> list[str]:
    """Extract anchor words from text.

    Words are words. don't stays don't. isn't stays isn't.
    Returns a list of non-empty lowercase words.
    """
    normalized = normalize_text(text)
    return [w for w in normalized.split() if w]




def extract_anchors_with_spellcheck(
    text: str,
    spellchecker: SpellChecker,
    alias_map: AliasMap,
) -> tuple[list[str], list[SpellResult]]:
    """Extract anchors, alias-resolve, then spell-check remaining.

    Pipeline:  raw text → extract_anchors → alias map → SymSpell
    """
    anchors = extract_anchors(text)
    resolved = alias_map.resolve_tokens(anchors)
    results = spellchecker.check_tokens(resolved)
    return resolved, results



"""
Clearbox Binary Store — common store header + lexicon binary format.

All binary stores share a fixed 64-byte header:
  Offset  Size   Field
  0       4      magic bytes (0xCB010001 for lexicon, etc.)
  4       2      format version (uint16 BE)
  6       2      schema version (uint16 BE)
  8       4      record count (uint32 BE)
  12      4      flags (uint32 BE, reserved)
  16      8      created_utc (uint64 BE, unix microseconds)
  24      8      updated_utc (uint64 BE, unix microseconds)
  32      4      payload checksum — CRC32 of everything after header (uint32 BE)
  36      28     reserved (zero-filled, future use)
  ─────────────
  64      ...    payload bytes

Store types (magic):
  0xCB010001  Lexicon store
  0xCB010002  Evidence store (cascade cells)
  0xCB010003  Scratch evidence store
  0xCB010004  Governed evidence store

Self-contained. No external imports beyond stdlib.
"""

from __future__ import annotations

import json
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Magic constants ─────────────────────────────────────────────────────────

MAGIC_LEXICON    = 0xCB010001
MAGIC_EVIDENCE   = 0xCB010002
MAGIC_SCRATCH    = 0xCB010003
MAGIC_GOVERNED   = 0xCB010004
MAGIC_CITATIONS  = 0xCB010005

MAGIC_NAMES = {
    MAGIC_LEXICON:   "lexicon",
    MAGIC_EVIDENCE:  "evidence",
    MAGIC_SCRATCH:   "scratch",
    MAGIC_GOVERNED:  "governed",
    MAGIC_CITATIONS: "citations",
}

HEADER_SIZE = 64
HEADER_STRUCT = struct.Struct(">I HH I I Q Q I 28s")  # 64 bytes total

FORMAT_VERSION = 1
SCHEMA_VERSION = 1

# ── Status / Category / Pool enums ──────────────────────────────────────────

STATUS_MAP = {
    "AVAILABLE": 0, "ASSIGNED": 1, "TEMP_ASSIGNED": 2,
    "STRUCTURAL": 3, "REJECTED": 4, "CHECKED_OUT": 5,
}
STATUS_REVERSE = {v: k for k, v in STATUS_MAP.items()}

CATEGORY_MAP = {
    "content": 0, "function": 1, "structural": 2,
    "punctuation": 3, "emoji": 4,
}
CATEGORY_REVERSE = {v: k for k, v in CATEGORY_MAP.items()}

POOL_MAP = {
    "canonical": 0, "medical": 1, "structural": 2,
    "temp_pool": 3, "spare_slots": 4,
}
POOL_REVERSE = {v: k for k, v in POOL_MAP.items()}


# ── Store Header ────────────────────────────────────────────────────────────

@dataclass
class StoreHeader:
    magic: int
    format_version: int = FORMAT_VERSION
    schema_version: int = SCHEMA_VERSION
    record_count: int = 0
    flags: int = 0
    created_utc: int = 0      # unix microseconds
    updated_utc: int = 0      # unix microseconds
    payload_crc: int = 0      # CRC32 of payload region

    @staticmethod
    def now_us() -> int:
        return int(time.time() * 1_000_000)

    def to_bytes(self) -> bytes:
        return HEADER_STRUCT.pack(
            self.magic,
            self.format_version,
            self.schema_version,
            self.record_count,
            self.flags,
            self.created_utc,
            self.updated_utc,
            self.payload_crc,
            b"\x00" * 28,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "StoreHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} bytes (need {HEADER_SIZE})")
        magic, fmt_v, sch_v, count, flags, created, updated, crc, _reserved = (
            HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        )
        if magic not in MAGIC_NAMES:
            raise ValueError(f"Unknown magic: {magic:#010x}")
        return cls(
            magic=magic, format_version=fmt_v, schema_version=sch_v,
            record_count=count, flags=flags,
            created_utc=created, updated_utc=updated,
            payload_crc=crc,
        )

    def verify_payload(self, payload: bytes) -> bool:
        return zlib.crc32(payload) == self.payload_crc

    def describe(self) -> Dict[str, Any]:
        return {
            "store_type": MAGIC_NAMES.get(self.magic, "unknown"),
            "format_version": self.format_version,
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "created": datetime.fromtimestamp(
                self.created_utc / 1_000_000, tz=timezone.utc
            ).isoformat() if self.created_utc else None,
            "updated": datetime.fromtimestamp(
                self.updated_utc / 1_000_000, tz=timezone.utc
            ).isoformat() if self.updated_utc else None,
            "payload_crc": f"{self.payload_crc:#010x}",
        }


# ── Lexicon Binary Record ──────────────────────────────────────────────────
#
# Variable-length record layout:
#   [2] record_len   (uint16 BE — total bytes including this field)
#   [1] hex_len      (uint8)
#   [N] hex_bytes    (utf-8, e.g. "0x563F49ABF2")
#   [1] word_len     (uint8, 0 if AVAILABLE/no word)
#   [N] word_bytes   (utf-8, normalized lowercase)
#   [1] display_len  (uint8, 0 if same as word)
#   [N] display_bytes(utf-8, original case)
#   [1] status       (uint8, STATUS_MAP)
#   [1] category     (uint8, CATEGORY_MAP)
#   [1] pool         (uint8, POOL_MAP)
#   [4] tone_sig     (uint32 BE)
#   [1] font_len     (uint8)
#   [N] font_bytes   (utf-8, e.g. "CHAR_0101011000")
#   [5] binary_repr  (5 bytes = 40 bits, the binary field from JSON)
#   [8] mapped_at    (int64 BE, unix microseconds, 0 if unmapped)
#   [8] updated_at   (int64 BE, unix microseconds)
#   [4] crc32        (uint32 BE, of everything before this field in this record)
#

def _encode_str(s: str, max_len: int = 255) -> bytes:
    """Encode a string as [len_byte][utf8_bytes]."""
    raw = s.encode("utf-8")[:max_len]
    return struct.pack("B", len(raw)) + raw


def _decode_str(data: bytes, pos: int) -> Tuple[str, int]:
    """Decode [len_byte][utf8_bytes] → (string, new_pos)."""
    slen = data[pos]; pos += 1
    val = data[pos:pos + slen].decode("utf-8"); pos += slen
    return val, pos


def _parse_binary_field(binary_str: str) -> bytes:
    """Convert 40-char binary string to 5 bytes. Pads/truncates if needed."""
    if not binary_str:
        return b"\x00" * 5
    binary_str = binary_str[:40].ljust(40, "0")
    return int(binary_str, 2).to_bytes(5, "big")


def _format_binary_field(raw: bytes) -> str:
    """Convert 5 bytes back to 40-char binary string."""
    n = int.from_bytes(raw, "big")
    return f"{n:040b}"


def _iso_to_us(iso_str: str) -> int:
    """Convert ISO timestamp string to unix microseconds. Returns 0 on failure."""
    if not iso_str:
        return 0
    try:
        dt = datetime.fromisoformat(iso_str)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return 0


def _us_to_iso(us: int) -> str:
    """Convert unix microseconds to ISO string. Returns '' if 0."""
    if not us:
        return ""
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).isoformat()


@dataclass
class LexiconRecord:
    """Single lexicon entry in binary form."""
    hex_addr: str              # e.g. "0x563F49ABF2"
    word: str = ""             # normalized lowercase, "" if AVAILABLE
    display: str = ""          # original case, "" to default to word
    status: str = "AVAILABLE"
    category: str = "content"
    pool: str = "canonical"
    tone_sig: int = 0
    font_symbol: str = ""
    binary_repr: str = ""      # 40-char binary string
    mapped_at_us: int = 0      # unix microseconds
    updated_at_us: int = 0     # unix microseconds

    def to_bytes(self) -> bytes:
        """Serialize to variable-length binary record."""
        buf = bytearray()
        buf += b"\x00\x00"  # placeholder for record_len

        buf += _encode_str(self.hex_addr)
        buf += _encode_str(self.word)
        buf += _encode_str(self.display if self.display != self.word else "")
        buf += struct.pack("B", STATUS_MAP.get(self.status, 0))
        buf += struct.pack("B", CATEGORY_MAP.get(self.category, 0))
        buf += struct.pack("B", POOL_MAP.get(self.pool, 0))
        buf += struct.pack(">I", self.tone_sig)
        buf += _encode_str(self.font_symbol)
        buf += _parse_binary_field(self.binary_repr)
        buf += struct.pack(">q", self.mapped_at_us)
        buf += struct.pack(">q", self.updated_at_us)

        # CRC32 of record body (after the 2-byte length)
        crc = zlib.crc32(bytes(buf[2:]))
        buf += struct.pack(">I", crc)

        # Write record_len at offset 0
        record_len = len(buf)
        struct.pack_into(">H", buf, 0, record_len)

        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["LexiconRecord", int]:
        """Deserialize from bytes. Returns (record, next_offset)."""
        pos = offset
        record_len = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2

        record_end = offset + record_len
        stored_crc = struct.unpack(">I", data[record_end - 4:record_end])[0]
        body = data[offset + 2:record_end - 4]
        computed_crc = zlib.crc32(body)
        if stored_crc != computed_crc:
            raise ValueError(
                f"Record CRC mismatch at offset {offset}: "
                f"{stored_crc:#010x} vs {computed_crc:#010x}"
            )

        hex_addr, pos = _decode_str(data, pos)
        word, pos = _decode_str(data, pos)
        display_raw, pos = _decode_str(data, pos)
        display = display_raw if display_raw else word

        status_b = data[pos]; pos += 1
        cat_b = data[pos]; pos += 1
        pool_b = data[pos]; pos += 1
        tone_sig = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
        font_symbol, pos = _decode_str(data, pos)
        binary_raw = data[pos:pos + 5]; pos += 5
        mapped_at = struct.unpack(">q", data[pos:pos + 8])[0]; pos += 8
        updated_at = struct.unpack(">q", data[pos:pos + 8])[0]; pos += 8

        return cls(
            hex_addr=hex_addr,
            word=word,
            display=display,
            status=STATUS_REVERSE.get(status_b, "AVAILABLE"),
            category=CATEGORY_REVERSE.get(cat_b, "content"),
            pool=POOL_REVERSE.get(pool_b, "canonical"),
            tone_sig=tone_sig,
            font_symbol=font_symbol,
            binary_repr=_format_binary_field(binary_raw),
            mapped_at_us=mapped_at,
            updated_at_us=updated_at,
        ), record_end


# ── Lexicon Store Writer ────────────────────────────────────────────────────

class LexiconStoreWriter:
    """Builds lexicon.bin + lexicon.idx + lexicon.reverse.idx from records."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[LexiconRecord] = []

    def add(self, record: LexiconRecord) -> None:
        self._records.append(record)

    def add_from_json_entry(self, entry: Dict[str, Any]) -> None:
        """Convert a canonical/spare JSON entry to LexiconRecord and add it."""
        hex_addr = entry.get("hex", entry.get("symbol", ""))
        if not hex_addr:
            return

        word = entry.get("word", "")
        if isinstance(word, str):
            word = word.strip().lower()
        else:
            word = ""

        display = entry.get("display", "")
        if not isinstance(display, str):
            display = ""

        status = str(entry.get("status", "AVAILABLE")).upper()
        pool = entry.get("pack", "canonical")
        tone_str = entry.get("tone_signature", "TONE_0")
        try:
            tone_sig = int(tone_str.replace("TONE_", "")) if isinstance(tone_str, str) else int(tone_str)
        except (ValueError, AttributeError):
            tone_sig = 0

        mapped_at_us = _iso_to_us(entry.get("mapped_at", ""))

        # Infer category from pool
        if pool == "structural":
            category = "structural"
        elif pool == "medical":
            category = "content"
        else:
            category = "content"

        self.add(LexiconRecord(
            hex_addr=hex_addr,
            word=word,
            display=display,
            status=status,
            category=category,
            pool=pool,
            tone_sig=tone_sig,
            font_symbol=entry.get("font_symbol", ""),
            binary_repr=entry.get("binary", ""),
            mapped_at_us=mapped_at_us,
            updated_at_us=StoreHeader.now_us(),
        ))

    def write(self) -> Dict[str, Any]:
        """Write lexicon.bin, lexicon.idx, lexicon.reverse.idx. Returns stats."""
        bin_path = self.output_dir / "lexicon.bin"
        idx_path = self.output_dir / "lexicon.idx"
        rev_path = self.output_dir / "lexicon.reverse.idx"

        now_us = StoreHeader.now_us()

        # Serialize all records
        payload = bytearray()
        index_entries: Dict[str, Dict[str, int]] = {}
        word_to_hex: Dict[str, str] = {}
        hex_to_words: Dict[str, List[str]] = {}

        assigned = 0
        available = 0

        for rec in self._records:
            rec_bytes = rec.to_bytes()
            offset = len(payload)
            payload += rec_bytes
            index_entries[rec.hex_addr] = {"offset": offset, "length": len(rec_bytes)}

            if rec.word:
                word_to_hex[rec.word] = rec.hex_addr
                hex_to_words.setdefault(rec.hex_addr, []).append(rec.word)

            if rec.status in ("ASSIGNED", "STRUCTURAL", "TEMP_ASSIGNED", "CHECKED_OUT"):
                assigned += 1
            else:
                available += 1

        payload_bytes = bytes(payload)
        payload_crc = zlib.crc32(payload_bytes)

        header = StoreHeader(
            magic=MAGIC_LEXICON,
            record_count=len(self._records),
            created_utc=now_us,
            updated_utc=now_us,
            payload_crc=payload_crc,
        )

        # Write binary store
        with open(bin_path, "wb") as f:
            f.write(header.to_bytes())
            f.write(payload_bytes)

        # Write index (hex → offset/length + word→hex lookup)
        idx_data = {
            "store_type": "lexicon",
            "format_version": FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "record_count": len(self._records),
            "entries": index_entries,
            "word_to_hex": word_to_hex,
        }
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx_data, f, separators=(",", ":"))

        # Write reverse index (word→hex, hex→words, surface→hex)
        rev_data = {
            "word_to_hex": word_to_hex,
            "hex_to_words": hex_to_words,
        }
        with open(rev_path, "w", encoding="utf-8") as f:
            json.dump(rev_data, f, separators=(",", ":"))

        stats = {
            "records": len(self._records),
            "assigned": assigned,
            "available": available,
            "bin_size": HEADER_SIZE + len(payload_bytes),
            "bin_path": str(bin_path),
            "idx_path": str(idx_path),
            "rev_path": str(rev_path),
        }
        return stats


# ── Lexicon Store Reader ────────────────────────────────────────────────────

class LexiconStoreReader:
    """Read-only access to lexicon.bin via index. mmap-backed."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self._bin_path = self.store_dir / "lexicon.bin"
        self._idx_path = self.store_dir / "lexicon.idx"
        self._rev_path = self.store_dir / "lexicon.reverse.idx"
        self.header: Optional[StoreHeader] = None
        self._data: Optional[bytes] = None
        self._index: Dict[str, Tuple[int, int]] = {}
        self._word_to_hex: Dict[str, str] = {}
        self._hex_to_words: Dict[str, List[str]] = {}

    def load(self) -> StoreHeader:
        """Load index + mmap the binary store. Returns header."""
        # Load index
        with open(self._idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        for hex_addr, info in idx["entries"].items():
            self._index[hex_addr] = (info["offset"], info["length"])
        self._word_to_hex = idx.get("word_to_hex", {})

        # Load reverse index if present
        if self._rev_path.exists():
            with open(self._rev_path, "r", encoding="utf-8") as f:
                rev = json.load(f)
            self._hex_to_words = rev.get("hex_to_words", {})

        # Read binary store
        raw = self._bin_path.read_bytes()
        self.header = StoreHeader.from_bytes(raw)
        self._data = raw

        # Verify payload integrity
        payload = raw[HEADER_SIZE:]
        if not self.header.verify_payload(payload):
            raise ValueError(
                f"Payload CRC mismatch in {self._bin_path}: "
                f"header says {self.header.payload_crc:#010x}, "
                f"computed {zlib.crc32(payload):#010x}"
            )

        return self.header

    def resolve_word(self, word: str) -> Optional[str]:
        """word (normalized) → hex. Returns None if not found."""
        return self._word_to_hex.get(word.lower())

    def resolve_hex(self, hex_addr: str) -> Optional[List[str]]:
        """hex → list of words. Returns None if not found."""
        return self._hex_to_words.get(hex_addr)

    def read_record(self, hex_addr: str) -> Optional[LexiconRecord]:
        """Read a single record by hex address."""
        loc = self._index.get(hex_addr)
        if loc is None:
            return None
        offset, length = loc
        abs_offset = HEADER_SIZE + offset
        rec, _ = LexiconRecord.from_bytes(self._data, abs_offset)
        return rec

    def read_by_word(self, word: str) -> Optional[LexiconRecord]:
        """Look up by normalized word."""
        hex_addr = self.resolve_word(word)
        if hex_addr is None:
            return None
        return self.read_record(hex_addr)

    def iter_records(self):
        """Iterate all records in store order. Generator."""
        if self._data is None:
            return
        pos = HEADER_SIZE
        end = len(self._data)
        while pos < end:
            rec, next_pos = LexiconRecord.from_bytes(self._data, pos)
            yield rec
            pos = next_pos

    @property
    def record_count(self) -> int:
        return self.header.record_count if self.header else 0

    @property
    def word_count(self) -> int:
        return len(self._word_to_hex)

    def close(self):
        self._data = None
        self._index.clear()
        self._word_to_hex.clear()
        self._hex_to_words.clear()


# ── Lexicon Store Appender (incremental writes) ────────────────────────────

class LexiconStoreAppender:
    """Append records to an existing lexicon.bin + update indexes.

    For single-word operations (assign slot, promote temp, etc.)
    without rebuilding the entire store.
    """

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self._bin_path = self.store_dir / "lexicon.bin"
        self._idx_path = self.store_dir / "lexicon.idx"
        self._rev_path = self.store_dir / "lexicon.reverse.idx"

    def append_record(self, record: LexiconRecord) -> None:
        """Append a single record to the binary store and update indexes."""
        rec_bytes = record.to_bytes()

        # Append to binary file
        with open(self._bin_path, "r+b") as f:
            # Read current header
            header = StoreHeader.from_bytes(f.read(HEADER_SIZE))

            # Seek to end, get payload offset for this record
            f.seek(0, 2)  # EOF
            file_end = f.tell()
            payload_offset = file_end - HEADER_SIZE

            # Write record
            f.write(rec_bytes)

            # Update header: count, timestamp, CRC
            header.record_count += 1
            header.updated_utc = StoreHeader.now_us()

            # Recompute CRC over entire payload (header excluded)
            f.seek(HEADER_SIZE)
            all_payload = f.read()
            header.payload_crc = zlib.crc32(all_payload)

            # Write updated header
            f.seek(0)
            f.write(header.to_bytes())

        # Update JSON index
        idx = self._load_idx()
        idx["entries"][record.hex_addr] = {
            "offset": payload_offset,
            "length": len(rec_bytes),
        }
        if record.word:
            idx["word_to_hex"][record.word] = record.hex_addr
        idx["record_count"] = header.record_count
        self._save_idx(idx)

        # Update reverse index
        if record.word:
            rev = self._load_rev()
            rev["word_to_hex"][record.word] = record.hex_addr
            rev["hex_to_words"].setdefault(record.hex_addr, [])
            if record.word not in rev["hex_to_words"][record.hex_addr]:
                rev["hex_to_words"][record.hex_addr].append(record.word)
            self._save_rev(rev)

    def remove_record(self, hex_addr: str) -> None:
        """Remove a record from indexes (binary file retains dead space).

        The binary file is not compacted — the record remains as dead bytes.
        Indexes no longer point to it, so it's invisible to readers.
        A full rebuild (via LexiconStoreWriter) compacts the file.
        """
        idx = self._load_idx()
        entry = idx["entries"].pop(hex_addr, None)
        # Remove from word_to_hex
        words_to_remove = [w for w, h in idx["word_to_hex"].items() if h == hex_addr]
        for w in words_to_remove:
            del idx["word_to_hex"][w]
        idx["record_count"] = len(idx["entries"])
        self._save_idx(idx)

        # Update reverse index
        rev = self._load_rev()
        for w in words_to_remove:
            rev["word_to_hex"].pop(w, None)
        rev["hex_to_words"].pop(hex_addr, None)
        self._save_rev(rev)

    def _load_idx(self) -> dict:
        with open(self._idx_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_idx(self, idx: dict) -> None:
        with open(self._idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, separators=(",", ":"))

    def _load_rev(self) -> dict:
        if self._rev_path.exists():
            with open(self._rev_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"word_to_hex": {}, "hex_to_words": {}}

    def _save_rev(self, rev: dict) -> None:
        with open(self._rev_path, "w", encoding="utf-8") as f:
            json.dump(rev, f, separators=(",", ":"))


# ── Hex Uniqueness Invariant ─────────────────────────────────────────────────
#
# HARD RULE: One hex, one truth.
#   - If a hex is active (ASSIGNED/TEMP_ASSIGNED/STRUCTURAL), it must NOT
#     exist in spare pool or temp pool.
#   - If a hex is in temp-assigned, it must NOT still exist in spare pool.
#   - If a hex is in spare pool, it must be unused everywhere else.
#   - No hex may appear more than once in ANY file, or across files.
#
# This invariant was established after discovering 254,448 stale AVAILABLE
# collisions in the JSON-era spare pools that silently shadowed active
# canonical records.  Never again.
#

def verify_hex_uniqueness(lexicon_root: Path) -> Dict[str, Any]:
    """Verify the one-hex-one-truth invariant across all pool files.

    Returns {"ok": True/False, "collisions": [...], "total": N, "unique": N}.
    Raises no exceptions — reports only.
    """
    all_hexes: Dict[str, Tuple[str, str]] = {}  # hex -> (source, status)
    collisions: List[Dict[str, str]] = []
    total = 0

    for subdir in ("Canonical", "Medical", "Structural", "Spare_Slots", "Temp_Pool"):
        d = lexicon_root / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                if not isinstance(entry, dict) or not entry.get("hex"):
                    continue
                h = entry["hex"]
                total += 1
                source = f"{subdir}/{f.name}"
                status = entry.get("status", "UNKNOWN")
                if h in all_hexes:
                    prev_source, prev_status = all_hexes[h]
                    collisions.append({
                        "hex": h,
                        "first": f"{prev_source}({prev_status})",
                        "second": f"{source}({status})",
                    })
                all_hexes[h] = (source, status)

    return {
        "ok": len(collisions) == 0,
        "collisions": collisions,
        "total": total,
        "unique": len(all_hexes),
    }


def enforce_hex_uniqueness(lexicon_root: Path) -> Dict[str, Any]:
    """Enforce the one-hex-one-truth invariant.

    Active records (canonical/medical/structural) win absolutely.
    Removes any hex from spare/temp pools that already exists as active.
    Also removes intra-file and cross-file spare pool duplicates.

    Returns cleanup stats.
    """
    # Step 1: Collect all active hex addresses
    active_hexes: set = set()
    for subdir in ("Canonical", "Medical", "Structural"):
        d = lexicon_root / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                if isinstance(entry, dict) and entry.get("hex"):
                    active_hexes.add(entry["hex"])

    # Step 2: Purge from spare and temp pools
    removed_spare = 0
    removed_temp = 0
    seen_pool_hexes: set = set()

    for subdir, counter_name in [("Spare_Slots", "spare"), ("Temp_Pool", "temp")]:
        d = lexicon_root / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue

            cleaned = []
            for entry in data:
                if not isinstance(entry, dict) or not entry.get("hex"):
                    cleaned.append(entry)
                    continue
                h = entry["hex"]
                # Remove if active elsewhere or already seen in another pool file
                if h in active_hexes or h in seen_pool_hexes:
                    if counter_name == "spare":
                        removed_spare += 1
                    else:
                        removed_temp += 1
                    continue
                seen_pool_hexes.add(h)
                cleaned.append(entry)

            if len(cleaned) != len(data):
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(cleaned, fh, separators=(",", ":"))

    return {
        "active_hexes": len(active_hexes),
        "removed_spare": removed_spare,
        "removed_temp": removed_temp,
        "total_removed": removed_spare + removed_temp,
    }


# ── Conversion utility: JSON lexicon → binary ──────────────────────────────

# Active statuses that belong in the lexicon authority store.
# AVAILABLE/spare slots are allocation inventory — separate concern.
_ACTIVE_STATUSES = frozenset({"ASSIGNED", "TEMP_ASSIGNED", "STRUCTURAL", "CHECKED_OUT", "REJECTED"})


def convert_json_lexicon_to_binary(
    lexicon_root: Path,
    output_dir: Path,
    include_spare: bool = False,
) -> Dict[str, Any]:
    """
    Reads canonical_*.json, med_*.json, structural.json, pool_temp_*.json
    from lexicon_root and writes lexicon.bin + indexes to output_dir.

    By default only ACTIVE records (ASSIGNED, TEMP_ASSIGNED, STRUCTURAL,
    CHECKED_OUT, REJECTED) are included.  Spare slots (AVAILABLE) stay in
    their original JSON pool files — they are allocation inventory, not
    lexicon authority.

    Set include_spare=True to include everything (testing/migration only).

    Returns stats dict.
    """
    writer = LexiconStoreWriter(output_dir)
    files_read = 0
    skipped_spare = 0

    # Only read from known subdirectories — never the root level.
    _PACK_DIRS = {
        "Canonical": "canonical",
        "Medical": "medical",
        "Structural": "structural",
        "Temp_Pool": "temp_pool",
        "Spare_Slots": "spare_slots",
    }
    for subdir_name, pack in _PACK_DIRS.items():
        subdir = lexicon_root / subdir_name
        if not subdir.exists():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(data, list):
                continue

            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if "pack" not in entry:
                    entry["pack"] = pack

                # Filter: skip AVAILABLE spare slots unless explicitly requested
                status = str(entry.get("status", "AVAILABLE")).upper()
                if not include_spare and status not in _ACTIVE_STATUSES:
                    skipped_spare += 1
                    continue

                writer.add_from_json_entry(entry)

            files_read += 1

    stats = writer.write()
    stats["json_files_read"] = files_read
    stats["skipped_spare"] = skipped_spare
    return stats


# ── Citation Binary Store ──────────────────────────────────────────────────
#
# Citation Record Layout (variable-length):
#   [2] record_len       (uint16 BE — total bytes including this field)
#   [1] cite_id_len      (uint8)
#   [N] cite_id          (utf-8, e.g. "cite_f3a2b8c1d4e6f9h2")
#   [2] coord_len        (uint16 BE — coords can exceed 255 chars)
#   [N] coord            (utf-8)
#   [1] source           (uint8, CITE_SOURCE_MAP)
#   [1] flags            (uint8, bitfield: 0=unresolved, 1=auto_generated,
#                          2=has_cross_links, 3=derived, 4=has_emoji, 5-7=reserved)
#   [1] subject_len      (uint8)
#   [N] subject          (utf-8, topic tag, max 255)
#   [1] note_len         (uint8)
#   [N] note             (utf-8, human annotation, max 250)
#   [1] context_len      (uint8, Phase 2 — 0 until built)
#   [N] context          (utf-8)
#   [1] subcontext_len   (uint8, Phase 2 — 0 until built)
#   [N] subcontext       (utf-8)
#   [1] field_tag_len    (uint8, Phase 2 — 0 until built)
#   [N] field_tag        (utf-8)
#   [1] receipt_id_len   (uint8)
#   [N] receipt_id       (utf-8)
#   [1] orig_marker_len  (uint8)
#   [N] original_marker  (utf-8, e.g. "[1]")
#   [2] orig_ref_len     (uint16 BE — refs can be long)
#   [N] original_ref     (utf-8, max 1000 chars, truncated at migration)
#   [8] created_at_us    (int64 BE, unix microseconds)
#   [8] updated_at_us    (int64 BE, unix microseconds)
#   [4] crc32            (uint32 BE — CRC of everything before this field)
#

CITE_SOURCE_MAP = {
    "ui": 0, "manual": 1, "system": 2, "lakespeak": 3,
    "import": 4, "linker": 5, "autocite": 6, "note": 7,
}
CITE_SOURCE_REVERSE = {v: k for k, v in CITE_SOURCE_MAP.items()}

# Max length for original_ref in binary records (full text stays in JSON/chunks)
CITE_ORIGINAL_REF_MAX = 1000


def _encode_str16(s: str, max_len: int = 65535) -> bytes:
    """Encode a string as [uint16 BE len][utf8_bytes]."""
    raw = s.encode("utf-8")[:max_len]
    return struct.pack(">H", len(raw)) + raw


def _decode_str16(data: bytes, pos: int) -> Tuple[str, int]:
    """Decode [uint16 BE len][utf8_bytes] → (string, new_pos)."""
    slen = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
    val = data[pos:pos + slen].decode("utf-8"); pos += slen
    return val, pos


@dataclass
class CitationRecord:
    """Single data citation in binary form."""
    cite_id: str = ""
    coord: str = ""
    source: str = "lakespeak"
    flags: int = 0                    # bitfield
    subject: str = ""
    note: str = ""
    context: str = ""                 # Phase 2
    subcontext: str = ""              # Phase 2
    field_tag: str = ""               # Phase 2
    receipt_id: str = ""
    original_marker: str = ""
    original_ref: str = ""            # truncated to CITE_ORIGINAL_REF_MAX
    created_at_us: int = 0
    updated_at_us: int = 0

    @property
    def unresolved(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def auto_generated(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def has_emoji(self) -> bool:
        return bool(self.flags & 0x10)

    def to_bytes(self) -> bytes:
        """Serialize to variable-length binary record."""
        buf = bytearray()
        buf += b"\x00\x00"  # placeholder for record_len

        buf += _encode_str(self.cite_id)
        buf += _encode_str16(self.coord)
        buf += struct.pack("B", CITE_SOURCE_MAP.get(self.source, 0))
        buf += struct.pack("B", self.flags)
        buf += _encode_str(self.subject)
        buf += _encode_str(self.note[:250] if self.note else "")
        buf += _encode_str(self.context)
        buf += _encode_str(self.subcontext)
        buf += _encode_str(self.field_tag)
        buf += _encode_str(self.receipt_id)
        buf += _encode_str(self.original_marker)
        buf += _encode_str16(self.original_ref[:CITE_ORIGINAL_REF_MAX] if self.original_ref else "")
        buf += struct.pack(">q", self.created_at_us)
        buf += struct.pack(">q", self.updated_at_us)

        # CRC32 of record body (after the 2-byte length)
        crc = zlib.crc32(bytes(buf[2:]))
        buf += struct.pack(">I", crc)

        # Write record_len at offset 0
        record_len = len(buf)
        struct.pack_into(">H", buf, 0, record_len)

        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["CitationRecord", int]:
        """Deserialize from bytes. Returns (record, next_offset)."""
        pos = offset
        record_len = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2

        record_end = offset + record_len
        stored_crc = struct.unpack(">I", data[record_end - 4:record_end])[0]
        body = data[offset + 2:record_end - 4]
        computed_crc = zlib.crc32(body)
        if stored_crc != computed_crc:
            raise ValueError(
                f"Citation record CRC mismatch at offset {offset}: "
                f"{stored_crc:#010x} vs {computed_crc:#010x}"
            )

        cite_id, pos = _decode_str(data, pos)
        coord, pos = _decode_str16(data, pos)
        source_b = data[pos]; pos += 1
        flags_b = data[pos]; pos += 1
        subject, pos = _decode_str(data, pos)
        note, pos = _decode_str(data, pos)
        context, pos = _decode_str(data, pos)
        subcontext, pos = _decode_str(data, pos)
        field_tag, pos = _decode_str(data, pos)
        receipt_id, pos = _decode_str(data, pos)
        original_marker, pos = _decode_str(data, pos)
        original_ref, pos = _decode_str16(data, pos)
        created_at = struct.unpack(">q", data[pos:pos + 8])[0]; pos += 8
        updated_at = struct.unpack(">q", data[pos:pos + 8])[0]; pos += 8

        return cls(
            cite_id=cite_id,
            coord=coord,
            source=CITE_SOURCE_REVERSE.get(source_b, "lakespeak"),
            flags=flags_b,
            subject=subject,
            note=note,
            context=context,
            subcontext=subcontext,
            field_tag=field_tag,
            receipt_id=receipt_id,
            original_marker=original_marker,
            original_ref=original_ref,
            created_at_us=created_at,
            updated_at_us=updated_at,
        ), record_end

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict for API responses."""
        return {
            "cite_id": self.cite_id,
            "coord": self.coord,
            "source": self.source,
            "unresolved": self.unresolved,
            "auto_generated": self.auto_generated,
            "has_emoji": self.has_emoji,
            "subject": self.subject,
            "note": self.note,
            "context": self.context,
            "subcontext": self.subcontext,
            "field_tag": self.field_tag,
            "receipt_id": self.receipt_id,
            "original_marker": self.original_marker,
            "original_ref": self.original_ref,
            "created_at_utc": _us_to_iso(self.created_at_us),
            "updated_at_utc": _us_to_iso(self.updated_at_us),
        }

    @classmethod
    def from_json(cls, record: Dict[str, Any]) -> "CitationRecord":
        """Construct from a JSON citation record (sidecar or library)."""
        flags = 0
        if record.get("unresolved", True):
            flags |= 0x01
        if record.get("auto_generated", False):
            flags |= 0x02

        # Extract receipt_id from INGEST coord if present
        receipt_id = record.get("receipt_id", "")
        coord = record.get("coord", record.get("canonical", ""))
        if not receipt_id and coord.startswith("INGEST:"):
            parts = coord.replace("INGEST:", "").split("#", 1)
            receipt_id = parts[0]

        # original_ref: from note field, or content preview for library cites
        orig_ref = record.get("original_ref", "")
        if not orig_ref:
            orig_ref = record.get("note", "") or ""
        if not orig_ref and record.get("content"):
            orig_ref = record["content"][:CITE_ORIGINAL_REF_MAX]

        return cls(
            cite_id=record.get("cite_id", ""),
            coord=coord,
            source=record.get("source", "lakespeak"),
            flags=flags,
            subject=record.get("subject", "") or "",
            note=record.get("note", "") or "",
            receipt_id=receipt_id,
            original_marker=record.get("original_marker", record.get("subject", "")) or "",
            original_ref=orig_ref[:CITE_ORIGINAL_REF_MAX],
            created_at_us=_iso_to_us(record.get("created_at_utc", "")),
            updated_at_us=StoreHeader.now_us(),
        )


# ── Citation Store Writer ──────────────────────────────────────────────────

class CitationStoreWriter:
    """Builds citations.bin + indexes from records."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[CitationRecord] = []

    def add(self, record: CitationRecord) -> None:
        self._records.append(record)

    def add_from_json(self, json_record: Dict[str, Any]) -> None:
        """Convert a JSON citation record and add it."""
        rec = CitationRecord.from_json(json_record)
        if rec.cite_id and rec.coord:
            self._records.append(rec)

    def write(self) -> Dict[str, Any]:
        """Write citations.bin + citations.idx + citations.coord.idx + citations.rcpt.idx."""
        bin_path = self.output_dir / "citations.bin"
        idx_path = self.output_dir / "citations.idx"
        coord_idx_path = self.output_dir / "citations.coord.idx"
        rcpt_idx_path = self.output_dir / "citations.rcpt.idx"

        now_us = StoreHeader.now_us()

        payload = bytearray()
        index_entries: Dict[str, Dict[str, int]] = {}
        coord_index: Dict[str, str] = {}
        rcpt_index: Dict[str, List[str]] = {}

        for rec in self._records:
            rec_bytes = rec.to_bytes()
            offset = len(payload)
            payload += rec_bytes
            index_entries[rec.cite_id] = {"offset": offset, "length": len(rec_bytes)}
            coord_index[rec.coord] = rec.cite_id
            if rec.receipt_id:
                rcpt_index.setdefault(rec.receipt_id, []).append(rec.cite_id)

        payload_bytes = bytes(payload)
        payload_crc = zlib.crc32(payload_bytes)

        header = StoreHeader(
            magic=MAGIC_CITATIONS,
            record_count=len(self._records),
            created_utc=now_us,
            updated_utc=now_us,
            payload_crc=payload_crc,
        )

        with open(bin_path, "wb") as f:
            f.write(header.to_bytes())
            f.write(payload_bytes)

        idx_data = {
            "store_type": "citations",
            "format_version": FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "record_count": len(self._records),
            "entries": index_entries,
        }
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx_data, f, separators=(",", ":"))

        with open(coord_idx_path, "w", encoding="utf-8") as f:
            json.dump(coord_index, f, separators=(",", ":"))

        with open(rcpt_idx_path, "w", encoding="utf-8") as f:
            json.dump(rcpt_index, f, separators=(",", ":"))

        return {
            "records": len(self._records),
            "bin_size": HEADER_SIZE + len(payload_bytes),
            "bin_path": str(bin_path),
            "idx_path": str(idx_path),
            "coord_idx_path": str(coord_idx_path),
            "rcpt_idx_path": str(rcpt_idx_path),
        }


# ── Citation Store Reader ──────────────────────────────────────────────────

class CitationStoreReader:
    """Read-only access to citations.bin via index."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self._bin_path = self.store_dir / "citations.bin"
        self._idx_path = self.store_dir / "citations.idx"
        self._coord_idx_path = self.store_dir / "citations.coord.idx"
        self._rcpt_idx_path = self.store_dir / "citations.rcpt.idx"
        self.header: Optional[StoreHeader] = None
        self._data: Optional[bytes] = None
        self._index: Dict[str, Tuple[int, int]] = {}
        self._coord_index: Dict[str, str] = {}
        self._rcpt_index: Dict[str, List[str]] = {}

    def load(self) -> StoreHeader:
        """Load indexes + read the binary store. Returns header."""
        with open(self._idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        for cite_id, info in idx["entries"].items():
            self._index[cite_id] = (info["offset"], info["length"])

        if self._coord_idx_path.exists():
            with open(self._coord_idx_path, "r", encoding="utf-8") as f:
                self._coord_index = json.load(f)

        if self._rcpt_idx_path.exists():
            with open(self._rcpt_idx_path, "r", encoding="utf-8") as f:
                self._rcpt_index = json.load(f)

        raw = self._bin_path.read_bytes()
        self.header = StoreHeader.from_bytes(raw)
        self._data = raw

        payload = raw[HEADER_SIZE:]
        if not self.header.verify_payload(payload):
            raise ValueError(
                f"Citation payload CRC mismatch in {self._bin_path}: "
                f"header says {self.header.payload_crc:#010x}, "
                f"computed {zlib.crc32(payload):#010x}"
            )
        return self.header

    def get(self, cite_id: str) -> Optional[CitationRecord]:
        """Read a single record by cite_id."""
        loc = self._index.get(cite_id)
        if loc is None:
            return None
        offset, length = loc
        abs_offset = HEADER_SIZE + offset
        rec, _ = CitationRecord.from_bytes(self._data, abs_offset)
        return rec

    def resolve(self, coord: str) -> Optional[CitationRecord]:
        """Lookup by coord string."""
        cite_id = self._coord_index.get(coord)
        if cite_id is None:
            return None
        return self.get(cite_id)

    def by_receipt(self, receipt_id: str) -> List[CitationRecord]:
        """All citations for a document receipt."""
        cite_ids = self._rcpt_index.get(receipt_id, [])
        results = []
        for cid in cite_ids:
            rec = self.get(cid)
            if rec:
                results.append(rec)
        return results

    def count(self) -> int:
        return len(self._index)

    def health(self) -> Dict[str, Any]:
        """Basic stats."""
        return {
            "ok": self.header is not None,
            "records": self.count(),
            "coords": len(self._coord_index),
            "receipts": len(self._rcpt_index),
            "header": self.header.describe() if self.header else None,
        }


# ── Citation Store Appender ────────────────────────────────────────────────

class CitationStoreAppender:
    """Append records to an existing citations.bin + update indexes."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self._bin_path = self.store_dir / "citations.bin"
        self._idx_path = self.store_dir / "citations.idx"
        self._coord_idx_path = self.store_dir / "citations.coord.idx"
        self._rcpt_idx_path = self.store_dir / "citations.rcpt.idx"

    def append_record(self, record: CitationRecord) -> None:
        """Append a single record to the binary store and update indexes."""
        rec_bytes = record.to_bytes()

        with open(self._bin_path, "r+b") as f:
            header = StoreHeader.from_bytes(f.read(HEADER_SIZE))
            f.seek(0, 2)
            file_end = f.tell()
            payload_offset = file_end - HEADER_SIZE
            f.write(rec_bytes)

            header.record_count += 1
            header.updated_utc = StoreHeader.now_us()
            f.seek(HEADER_SIZE)
            all_payload = f.read()
            header.payload_crc = zlib.crc32(all_payload)
            f.seek(0)
            f.write(header.to_bytes())

        # Update primary index
        idx = self._load_json(self._idx_path)
        idx["entries"][record.cite_id] = {
            "offset": payload_offset,
            "length": len(rec_bytes),
        }
        idx["record_count"] = header.record_count
        self._save_json(self._idx_path, idx)

        # Update coord index
        coord_idx = self._load_json(self._coord_idx_path)
        coord_idx[record.coord] = record.cite_id
        self._save_json(self._coord_idx_path, coord_idx)

        # Update receipt index
        if record.receipt_id:
            rcpt_idx = self._load_json(self._rcpt_idx_path)
            rcpt_idx.setdefault(record.receipt_id, [])
            if record.cite_id not in rcpt_idx[record.receipt_id]:
                rcpt_idx[record.receipt_id].append(record.cite_id)
            self._save_json(self._rcpt_idx_path, rcpt_idx)

    def remove_record(self, cite_id: str) -> None:
        """Remove a record from indexes (binary file retains dead space)."""
        idx = self._load_json(self._idx_path)
        idx["entries"].pop(cite_id, None)
        idx["record_count"] = len(idx["entries"])
        self._save_json(self._idx_path, idx)

        # Remove from coord index
        coord_idx = self._load_json(self._coord_idx_path)
        to_remove = [c for c, cid in coord_idx.items() if cid == cite_id]
        for c in to_remove:
            del coord_idx[c]
        self._save_json(self._coord_idx_path, coord_idx)

        # Remove from receipt index
        rcpt_idx = self._load_json(self._rcpt_idx_path)
        for rcpt, cids in rcpt_idx.items():
            if cite_id in cids:
                cids.remove(cite_id)
        rcpt_idx = {k: v for k, v in rcpt_idx.items() if v}
        self._save_json(self._rcpt_idx_path, rcpt_idx)

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

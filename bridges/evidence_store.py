"""Evidence Store — cumulative 6-1-6 positional co-occurrence authority.

One binary file. Hex-keyed cells. 12 positional buckets (-6..-1, +1..+6).
Each bucket holds (neighbor_hex, count) pairs. Counts accumulate across
every document ever ingested. Nothing decrements. Append-only.

Files:
    evidence.bin      — cell records (variable-length, CRC32 per cell)
    evidence.idx      — JSON index: {hex_addr: {offset, length}}

Usage:
    store = EvidenceStore(EVIDENCE_DIR)
    store.load()                          # load index, mmap binary
    store.append_counts(counts_by_hex)    # merge new counts into cells
    cell = store.read_cell(hex_addr)      # read one cell
    store.health()                        # stats
"""

from __future__ import annotations

import json
import logging
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bridges.binary_store import (
    MAGIC_EVIDENCE,
    StoreHeader,
    HEADER_SIZE,
)

LOGGER = logging.getLogger("ClearboxAI.evidence_store")

# ── Cell layout ──────────────────────────────────────────────────
#
# Per cell (variable-length):
#   [1]  hex_len       (uint8)
#   [N]  hex_addr      (utf-8, e.g. "0A1B2C3D4E")
#   [4]  total_count   (uint32 BE — sum of all neighbor counts across all buckets)
#   [12 buckets, each]:
#     [2]  entry_count  (uint16 BE)
#     [entry_count entries, each]:
#       [1]  neighbor_hex_len  (uint8)
#       [N]  neighbor_hex      (utf-8)
#       [4]  count             (uint32 BE)
#   [4]  crc32          (uint32 BE — of everything before this field)
#
# Buckets ordered: -6, -5, -4, -3, -2, -1, +1, +2, +3, +4, +5, +6
# (zero position excluded — that's the anchor itself)

BUCKET_OFFSETS = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
BUCKET_COUNT = 12


@dataclass
class NeighborEntry:
    hex_addr: str
    count: int


@dataclass
class EvidenceCell:
    hex_addr: str
    total_count: int
    buckets: Dict[int, List[NeighborEntry]] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize cell to binary."""
        parts: list[bytes] = []
        # hex_addr
        raw_hex = self.hex_addr.encode("utf-8")
        parts.append(struct.pack("B", len(raw_hex)))
        parts.append(raw_hex)
        # total_count
        parts.append(struct.pack(">I", self.total_count))
        # 12 buckets in order
        for offset in BUCKET_OFFSETS:
            entries = self.buckets.get(offset, [])
            parts.append(struct.pack(">H", len(entries)))
            for e in entries:
                raw_n = e.hex_addr.encode("utf-8")
                parts.append(struct.pack("B", len(raw_n)))
                parts.append(raw_n)
                parts.append(struct.pack(">I", e.count))
        # CRC32
        payload = b"".join(parts)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return payload + struct.pack(">I", crc)

    @classmethod
    def from_bytes(cls, data: bytes, pos: int = 0) -> Tuple["EvidenceCell", int]:
        """Deserialize cell from binary. Returns (cell, next_pos)."""
        start = pos
        # hex_addr
        hex_len = data[pos]; pos += 1
        hex_addr = data[pos:pos + hex_len].decode("utf-8"); pos += hex_len
        # total_count
        total_count = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
        # 12 buckets
        buckets: Dict[int, List[NeighborEntry]] = {}
        for offset in BUCKET_OFFSETS:
            entry_count = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
            entries: List[NeighborEntry] = []
            for _ in range(entry_count):
                n_len = data[pos]; pos += 1
                n_hex = data[pos:pos + n_len].decode("utf-8"); pos += n_len
                count = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
                entries.append(NeighborEntry(hex_addr=n_hex, count=count))
            buckets[offset] = entries
        # CRC32
        expected_crc = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
        actual_crc = zlib.crc32(data[start:pos - 4]) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"CRC mismatch for cell {hex_addr}: expected {expected_crc:#x}, got {actual_crc:#x}")
        return cls(hex_addr=hex_addr, total_count=total_count, buckets=buckets), pos

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation for API responses."""
        return {
            "hex_addr": self.hex_addr,
            "total_count": self.total_count,
            "buckets": {
                str(offset): [{"hex": e.hex_addr, "count": e.count} for e in entries]
                for offset, entries in self.buckets.items()
            },
        }


# ── Evidence Store ───────────────────────────────────────────────

class EvidenceStore:
    """Cumulative 6-1-6 evidence authority. Hex-keyed, append-only counts."""

    def __init__(self, store_dir: Path) -> None:
        self._dir = Path(store_dir)
        self._bin_path = self._dir / "evidence.bin"
        self._idx_path = self._dir / "evidence.idx"
        self._index: Dict[str, dict] = {}  # hex_addr -> {offset, length}
        self._loaded = False

    def load(self) -> None:
        """Load index from disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._idx_path.exists():
            try:
                raw = json.loads(self._idx_path.read_text(encoding="utf-8"))
                self._index = raw.get("entries", {})
            except (json.JSONDecodeError, OSError) as e:
                LOGGER.warning("Failed to load evidence index: %s", e)
                self._index = {}
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def count(self) -> int:
        """Number of cells in the store."""
        self._ensure_loaded()
        return len(self._index)

    def read_cell(self, hex_addr: str) -> Optional[EvidenceCell]:
        """Read a single evidence cell by hex address."""
        self._ensure_loaded()
        entry = self._index.get(hex_addr)
        if entry is None:
            return None
        try:
            with open(self._bin_path, "rb") as f:
                f.seek(HEADER_SIZE + entry["offset"])
                data = f.read(entry["length"])
            cell, _ = EvidenceCell.from_bytes(data)
            return cell
        except Exception as e:
            LOGGER.warning("Failed to read cell %s: %s", hex_addr, e)
            return None

    def append_counts(
        self,
        counts: Dict[str, Dict[int, Dict[str, int]]],
    ) -> int:
        """Merge new co-occurrence counts into the evidence store.

        Args:
            counts: {focus_hex: {offset: {neighbor_hex: count}}}
                    offset in {-6..-1, +1..+6}

        Returns:
            Number of cells updated.
        """
        self._ensure_loaded()
        if not counts:
            return 0

        # Read all existing cells that need updating
        cells: Dict[str, EvidenceCell] = {}
        for focus_hex in counts:
            existing = self.read_cell(focus_hex)
            if existing:
                cells[focus_hex] = existing
            else:
                cells[focus_hex] = EvidenceCell(
                    hex_addr=focus_hex, total_count=0, buckets={}
                )

        # Merge counts
        for focus_hex, offsets in counts.items():
            cell = cells[focus_hex]
            for offset, neighbors in offsets.items():
                if offset not in BUCKET_OFFSETS:
                    continue
                bucket = cell.buckets.setdefault(offset, [])
                # Build lookup for existing neighbors
                existing_map = {e.hex_addr: e for e in bucket}
                for n_hex, count in neighbors.items():
                    if n_hex in existing_map:
                        existing_map[n_hex].count += count
                    else:
                        entry = NeighborEntry(hex_addr=n_hex, count=count)
                        bucket.append(entry)
                        existing_map[n_hex] = entry
                cell.buckets[offset] = bucket

        # Recompute total_count per cell
        for cell in cells.values():
            cell.total_count = sum(
                e.count for entries in cell.buckets.values() for e in entries
            )

        # Rewrite the entire binary (full rebuild for now — compact and correct)
        # For scale, this becomes append+compaction. For current size, rewrite is fine.
        self._dir.mkdir(parents=True, exist_ok=True)

        # Load ALL existing cells first
        all_cells: Dict[str, EvidenceCell] = {}
        if self._bin_path.exists() and self._index:
            with open(self._bin_path, "rb") as f:
                raw = f.read()
            for hex_addr, idx_entry in self._index.items():
                if hex_addr not in cells:  # Not being updated
                    try:
                        cell, _ = EvidenceCell.from_bytes(
                            raw, HEADER_SIZE + idx_entry["offset"]
                        )
                        all_cells[hex_addr] = cell
                    except Exception:
                        pass

        # Merge updated cells
        all_cells.update(cells)

        # Write
        header = StoreHeader(
            magic=MAGIC_EVIDENCE,
            record_count=len(all_cells),
            created_utc=StoreHeader.now_us(),
            updated_utc=StoreHeader.now_us(),
        )

        payload_parts: list[bytes] = []
        new_index: Dict[str, dict] = {}
        offset = 0
        for hex_addr in sorted(all_cells):
            cell_bytes = all_cells[hex_addr].to_bytes()
            new_index[hex_addr] = {"offset": offset, "length": len(cell_bytes)}
            payload_parts.append(cell_bytes)
            offset += len(cell_bytes)

        payload = b"".join(payload_parts)
        header.payload_crc = zlib.crc32(payload) & 0xFFFFFFFF

        with open(self._bin_path, "wb") as f:
            f.write(header.to_bytes())
            f.write(payload)

        # Write index
        idx_data = {
            "store_type": "evidence",
            "record_count": len(new_index),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": new_index,
        }
        self._idx_path.write_text(
            json.dumps(idx_data, separators=(",", ":")),
            encoding="utf-8",
        )

        self._index = new_index
        LOGGER.info(
            "Evidence store updated: %d cells (%d updated), %.1f KB",
            len(new_index), len(counts), len(payload) / 1024,
        )
        return len(counts)

    def health(self) -> Dict[str, Any]:
        """Return store health stats."""
        self._ensure_loaded()
        bin_size = self._bin_path.stat().st_size if self._bin_path.exists() else 0
        return {
            "ok": True,
            "cells": len(self._index),
            "bin_size_kb": round(bin_size / 1024, 1),
            "path": str(self._bin_path),
        }


# ── Singleton ────────────────────────────────────────────────────

_store: Optional[EvidenceStore] = None


def get_evidence_store() -> EvidenceStore:
    """Get or create the singleton evidence store."""
    global _store
    if _store is None:
        from security.data_paths import EVIDENCE_DIR
        _store = EvidenceStore(EVIDENCE_DIR)
        _store.load()
    return _store

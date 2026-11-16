"""The fingerprint index on disk: immutable segments, merged in the background.

A library of 30,000 hours holds about 1.1 billion hashes — a table in
SQLite would be rewritten by every batch of new rows (each touches every
leaf of a B-tree that size). So the hashes live in **segments**: files
written once, sorted by hash, read through mmap:

    header   64 bytes     magic, hash bits, bytes per entry, entries
    offsets  8 × (2^21 + 1) bytes   where each hash's entries begin
    entries  6 bytes each           recording id (22 bits) · frame (26 bits)

A hash's entries are one contiguous run: a lookup reads a few kB per hash
(on average a few hundred entries at that size). New fingerprints collect
in memory and are written as a new segment (store.py); segments of a
similar size are merged into one — four of a size make one of the next,
so every entry is rewritten a few times over the life of the index, not
with every recording. A merge drops the entries of recordings analyzed
again or gone (``dead``).

Limits: 4,194,303 recording ids (the store never reuses one) and frames up
to 2^26 — 298 hours in one recording; a longer one keeps the first 298.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import numpy as np

from whisper_tm.audio.fingerprint import HASH_BITS

MAGIC = b"WXAFP\x00\x00\x01"
HEADER = 64
#: Bytes of an entry
REC = 6
ID_BITS, T_BITS = 22, 26
MAX_ID = (1 << ID_BITS) - 1
MAX_T = (1 << T_BITS) - 1
BUCKETS = 1 << HASH_BITS
_HEAD = struct.Struct("<8sIIQd")
#: Buckets merged at a time — bounds a merge's memory to a 128th of the
#: entries (16-bit bucket numbers sort by radix)
_MERGE_STEP = 1 << 14


def pack(ids: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Entries (uint64, the low 48 bits used) of recording ``ids`` at
    ``frames``."""
    return (np.asarray(ids, np.uint64) << np.uint64(T_BITS)) | np.asarray(frames, np.uint64)


def unpack(entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recording ids and frames (int64) of entries."""
    e = np.asarray(entries, np.uint64)
    return (e >> np.uint64(T_BITS)).astype(np.int64), (e & np.uint64(MAX_T)).astype(np.int64)


def _to_rows(entries: np.ndarray) -> np.ndarray:
    """uint64 entries → (n, REC) bytes, little endian."""
    return np.ascontiguousarray(entries.astype("<u8").view(np.uint8).reshape(-1, 8)[:, :REC])


def _from_rows(rows: np.ndarray) -> np.ndarray:
    """(n, REC) bytes → uint64 entries."""
    out = np.zeros((len(rows), 8), np.uint8)
    out[:, :REC] = rows
    return out.view("<u8").ravel()


def _write_header(f, count: int) -> None:
    f.write(_HEAD.pack(MAGIC, HASH_BITS, REC, count, time.time()).ljust(HEADER, b"\0"))


def write(path: Path, hashes: np.ndarray, entries: np.ndarray) -> int:
    """A segment of ``entries`` under ``hashes`` at ``path`` — written
    whole under a temporary name first, then renamed: a crash leaves no half
    segment. Returns the number of entries."""
    hashes = np.asarray(hashes, np.uint32)
    order = np.argsort(hashes, kind="stable")
    counts = np.bincount(hashes, minlength=BUCKETS).astype(np.uint64)
    offsets = np.zeros(BUCKETS + 1, "<u8")
    np.cumsum(counts, out=offsets[1:])
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        _write_header(f, len(hashes))
        f.write(offsets.tobytes())
        for i in range(0, len(order), 4_000_000):
            f.write(_to_rows(np.asarray(entries)[order[i : i + 4_000_000]]).tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(hashes)


class Segment:
    """A segment, read through mmap."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic, bits, rec, count, created = _HEAD.unpack(f.read(_HEAD.size))
        if magic != MAGIC or bits != HASH_BITS or rec != REC:
            raise ValueError(f"{self.path.name}: not a fingerprint segment of this version")
        self.count = int(count)
        self.created = created
        mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        start = HEADER + 8 * (BUCKETS + 1)
        if len(mm) != start + REC * self.count:
            raise ValueError(f"{self.path.name}: truncated")
        self.offsets = mm[HEADER:start].view("<u8")
        self.rows = mm[start:].reshape(-1, REC) if self.count else np.zeros((0, REC), np.uint8)
        self._mm = mm

    @property
    def nbytes(self) -> int:
        return HEADER + 8 * (BUCKETS + 1) + REC * self.count

    def bucket_sizes(self, hashes: np.ndarray) -> np.ndarray:
        h = np.asarray(hashes, np.int64)
        return (self.offsets[h + 1] - self.offsets[h]).astype(np.int64)

    def lookup(self, hashes: np.ndarray, max_bucket: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """The entries under ``hashes`` (uint64), and for each the index of
        its hash in ``hashes``. Buckets larger than ``max_bucket`` (0 = no
        limit) are skipped: a hash that common says next to nothing."""
        h = np.asarray(hashes, np.int64)
        lo = self.offsets[h].astype(np.int64)
        n = self.offsets[h + 1].astype(np.int64) - lo
        if max_bucket:
            n[n > max_bucket] = 0
        total = int(n.sum())
        if not total:
            return np.zeros(0, np.uint64), np.zeros(0, np.int64)
        rows = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(total)
        return _from_rows(self.rows[rows]), np.repeat(np.arange(len(h)), n)

    def close(self) -> None:
        self.offsets = self.rows = None
        mm, self._mm = self._mm, None
        if mm is not None and getattr(mm, "_mmap", None) is not None:
            try:
                mm._mmap.close()
            except (BufferError, ValueError):
                pass  # a lookup still holds a view; the mapping goes with it


def merge(sources: list[Segment], path: Path, dead: np.ndarray | None = None, stop=None) -> int:
    """One segment of ``sources`` at ``path``, without the entries of the
    recording ids in ``dead``. Returns the number of entries; raises
    InterruptedError when ``stop()`` turns true (nothing is left behind)."""
    dead = np.unique(np.asarray(dead if dead is not None else [], np.int64))
    counts = np.zeros(BUCKETS, np.uint64)
    tmp = path.with_name(path.name + ".part")
    total = 0
    try:
        with open(tmp, "wb") as f:
            _write_header(f, 0)
            f.write(b"\0" * (8 * (BUCKETS + 1)))
            for b0 in range(0, BUCKETS, _MERGE_STEP):
                if stop is not None and stop():
                    raise InterruptedError("stopped")
                b1 = min(BUCKETS, b0 + _MERGE_STEP)
                parts, buckets = [], []
                for s in sources:
                    lo, hi = int(s.offsets[b0]), int(s.offsets[b1])
                    if hi <= lo:
                        continue
                    parts.append(np.asarray(s.rows[lo:hi]))
                    n = np.diff(s.offsets[b0 : b1 + 1].astype(np.int64))
                    buckets.append(np.repeat(np.arange(b1 - b0, dtype=np.uint16), n))
                if not parts:
                    continue
                rows = np.concatenate(parts)
                bk = np.concatenate(buckets)
                order = np.argsort(bk, kind="stable")
                rows, bk = rows[order], bk[order]
                if len(dead):
                    ids, _ = unpack(_from_rows(rows))
                    keep = ~np.isin(ids, dead)
                    rows, bk = rows[keep], bk[keep]
                counts[b0:b1] = np.bincount(bk, minlength=b1 - b0)
                f.write(np.ascontiguousarray(rows).tobytes())
                total += len(rows)
            offsets = np.zeros(BUCKETS + 1, "<u8")
            np.cumsum(counts, out=offsets[1:])
            f.seek(0)
            _write_header(f, total)
            f.write(offsets.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return total

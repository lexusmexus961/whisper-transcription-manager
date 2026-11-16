"""``audio.db`` next to the main database, and the fingerprint segments.

* ``files`` — one row per analyzed recording, by its absolute path
  (docs/PATHS.md): the size and modification time it had (another one
  means analyze again), its loudness figures, how many hashes it has and
  whether they are in a segment yet (``stored``). Its ``id`` is the
  recording id inside the segments — never reused (AUTOINCREMENT): an id
  analyzed again or deleted goes to ``dead`` until a merge has dropped its
  entries.
* ``waveforms`` — the waveform, 20 values a second (levels.py).
* ``relations`` — recordings that are the same, or contain one another,
  found when the later of the two was analyzed (analyze.py).

The segments live in the folder ``audio.db.fp`` beside it (index.py).
New fingerprints collect in memory — a search sees them at once — and are
written as a segment when ``FLUSH_ENTRIES`` have come together, ten minutes
after the first, or when the server stops. A server killed before that
loses them: such rows are found at the next start (``stored`` 0) and
analyzed again by the next run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from whisper_tm.audio import index
from whisper_tm.audio.fingerprint import VERSION

logger = logging.getLogger(__name__)

#: Entries in memory before they become a segment (~110 hours of sound)
FLUSH_ENTRIES = 4_000_000
#: ... or this long after the first of them came
FLUSH_AFTER_SEC = 600.0
#: Four segments of one size make one of the next size up; sizes step by 4
#: from this
_TIER_BASE = 1_000_000
_TIER_FAN = 4
#: Never more segments than this — the smallest are merged first
_MAX_SEGMENTS = 12

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    size        INTEGER NOT NULL DEFAULT 0,
    mtime       REAL NOT NULL DEFAULT 0,
    duration    REAL,
    status      TEXT NOT NULL,
    error       TEXT,
    source      TEXT,
    analyzed_at REAL,
    seconds     REAL,
    lufs        REAL,
    lra         REAL,
    peak_db     REAL,
    silence     REAL,
    clipped     REAL,
    hashes      INTEGER NOT NULL DEFAULT 0,
    stored      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS files_status ON files(status);
CREATE TABLE IF NOT EXISTS waveforms (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    levels  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS dead (id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS relations (
    a        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    b        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL,
    offset   REAL NOT NULL,
    matched  REAL NOT NULL,
    score    INTEGER NOT NULL,
    found_at REAL,
    PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS relations_b ON relations(b);
CREATE INDEX IF NOT EXISTS relations_kind ON relations(kind);
"""

#: The columns of a file row the API hands out
FILE_COLUMNS = (
    "id",
    "path",
    "size",
    "mtime",
    "duration",
    "status",
    "error",
    "source",
    "analyzed_at",
    "seconds",
    "lufs",
    "lra",
    "peak_db",
    "silence",
    "clipped",
    "hashes",
    "stored",
)


class Snapshot:
    """What a search reads: the segments and the entries in memory at one
    moment, and the ids no longer valid."""

    def __init__(self, segments, pend_h, pend_e, dead):
        self.segments = segments
        self.pend_h = pend_h
        self.pend_e = pend_e
        self.dead = dead
        self._pend_counts: tuple[np.ndarray, np.ndarray] | None = None

    def bucket_sizes(self, hashes: np.ndarray) -> np.ndarray:
        """How many entries each of ``hashes`` has — in every segment and in
        memory (dead ones included: an upper bound)."""
        h = np.asarray(hashes, np.int64)
        n = np.zeros(len(h), np.int64)
        for seg in self.segments:
            n += seg.bucket_sizes(h)
        if len(self.pend_h):
            if self._pend_counts is None:
                self._pend_counts = np.unique(self.pend_h, return_counts=True)
            uh, cnt = self._pend_counts
            pos = np.minimum(np.searchsorted(uh, h), len(uh) - 1)
            n += np.where(uh[pos] == h, cnt[pos], 0)
        return n

    def lookup(self, hashes: np.ndarray, max_bucket: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Entries under ``hashes`` (unique, uint32) and the index of each
        one's hash — from every segment and from memory."""
        parts, which = [], []
        for seg in self.segments:
            e, w = seg.lookup(hashes, max_bucket)
            if len(e):
                parts.append(e)
                which.append(w)
        if len(self.pend_h):
            pos = np.searchsorted(hashes, self.pend_h)
            pos = np.minimum(pos, len(hashes) - 1)
            hit = hashes[pos] == self.pend_h
            if hit.any():
                parts.append(self.pend_e[hit])
                which.append(pos[hit].astype(np.int64))
        if not parts:
            return np.zeros(0, np.uint64), np.zeros(0, np.int64)
        e, w = np.concatenate(parts), np.concatenate(which)
        if len(self.dead):
            ids, _ = index.unpack(e)
            keep = ~np.isin(ids, self.dead)
            e, w = e[keep], w[keep]
        return e, w


class AudioStore:
    """``audio.db`` and its segments — one per database file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.folder = self.path.with_name(self.path.name + ".fp")
        self._lock = threading.RLock()
        self._local = threading.local()
        self._segments: list[index.Segment] = []
        self._pend_h: list[np.ndarray] = []
        self._pend_e: list[np.ndarray] = []
        self._pend_ids: list[int] = []
        self._pend_n = 0
        self._pend_since = 0.0
        self._pend_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._dead = np.zeros(0, np.int64)
        self._merging = False
        self._stop = threading.Event()
        self._closed = False
        self._counts: tuple[float, dict] | None = None
        self._lost_segment = False
        self._seg_no: int | None = None
        self._flushing = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)
            kind = c.execute("SELECT value FROM meta WHERE key='version'").fetchone()
            if kind is None:
                c.execute("INSERT INTO meta(key, value) VALUES('version', ?)", (str(VERSION),))
            elif kind[0] != str(VERSION):
                # Fingerprints of another kind can't be searched with these:
                # everything is analyzed again
                logger.warning(
                    "Audio analysis: fingerprints of version %s, this is %s — "
                    "the index starts over",
                    kind[0],
                    VERSION,
                )
                self._wipe(c)
                c.execute("UPDATE meta SET value=? WHERE key='version'", (str(VERSION),))
        self._open_segments()
        self._recover()
        self._maintainer = threading.Thread(target=self._maintain, name="audio index", daemon=True)
        self._maintainer.start()

    # ── Connections ──────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def meta(self, key: str, default=None):
        with self.connect() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_meta(self, key: str, value) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    # ── Segments ─────────────────────────────────────────────────────────

    def _seg_paths(self) -> list[Path]:
        return sorted(self.folder.glob("seg-*.afp"))

    def _open_segments(self) -> None:
        # A merge that was interrupted: its journal names the segments it
        # replaced — gone once the new one is complete, kept otherwise
        for journal in self.folder.glob("merge-*.json"):
            try:
                info = json.loads(journal.read_text())
                if (self.folder / info["output"]).exists():
                    for name in info["sources"]:
                        (self.folder / name).unlink(missing_ok=True)
            except (OSError, ValueError, KeyError):
                pass
            journal.unlink(missing_ok=True)
        for part in self.folder.glob("*.part"):
            part.unlink(missing_ok=True)
        segs = []
        for p in self._seg_paths():
            try:
                segs.append(index.Segment(p))
            except (OSError, ValueError) as e:
                logger.warning(
                    "Audio analysis: segment %s unreadable (%s) — removed; "
                    "its recordings are analyzed again",
                    p.name,
                    e,
                )
                p.unlink(missing_ok=True)
                self._lost_segment = True
        self._segments = segs
        with self.connect() as c:
            self._dead = np.array([r[0] for r in c.execute("SELECT id FROM dead")], np.int64)

    def _recover(self) -> None:
        """Rows whose fingerprints never reached a segment are analyzed
        again — and all rows, when a segment had to be dropped."""
        with self.connect() as c:
            if self._lost_segment:
                ids = [r[0] for r in c.execute("SELECT id FROM files WHERE stored=1 AND hashes>0")]
                c.executemany("INSERT OR IGNORE INTO dead(id) VALUES(?)", [(i,) for i in ids])
                c.execute("DELETE FROM files WHERE stored=1 AND hashes>0")
            lost = c.execute(
                "DELETE FROM files WHERE status='done' AND hashes>0 AND stored=0"
            ).rowcount
        if lost:
            logger.info(
                "Audio analysis: %d recording(s) whose fingerprints were not written "
                "yet will be analyzed again",
                lost,
            )
        with self.connect() as c:
            self._dead = np.array([r[0] for r in c.execute("SELECT id FROM dead")], np.int64)

    def _wipe(self, c) -> None:
        c.execute("DELETE FROM files")
        c.execute("DELETE FROM dead")
        c.execute("DELETE FROM relations")
        for p in self.folder.glob("seg-*.afp"):
            p.unlink(missing_ok=True)

    def _next_name(self) -> str:
        """A segment name never used before — counted in memory, under the
        lock: a merge writes its output for minutes, and a flush meanwhile
        that read the folder again would pick the same name and lose one of
        the two."""
        with self._lock:
            if self._seg_no is None:
                used = [
                    int(p.stem.split("-")[1])
                    for p in self._seg_paths()
                    if p.stem.split("-")[1].isdigit()
                ]
                self._seg_no = max(used) if used else 0
            self._seg_no += 1
            return f"seg-{self._seg_no:06d}.afp"

    def flush(self) -> int:
        """Write the fingerprints in memory as a segment. Returns entries."""
        with self._lock:
            if not self._pend_n:
                return 0
            h = np.concatenate(self._pend_h)
            e = np.concatenate(self._pend_e)
            ids = list(self._pend_ids)
            name = self._next_name()
            n = index.write(self.folder / name, h, e)
            seg = index.Segment(self.folder / name)
            with self.connect() as c:
                c.executemany("UPDATE files SET stored=1 WHERE id=?", [(i,) for i in ids])
            self._segments = self._segments + [seg]
            self._pend_h, self._pend_e, self._pend_ids = [], [], []
            self._pend_n = 0
            self._pend_cache = None
            self._counts = None
        logger.info(
            "Audio analysis: %s entries of %d recording(s) written (%s)", f"{n:,}", len(ids), name
        )
        return n

    def _tier(self, seg: index.Segment) -> int:
        n, tier = seg.count, 0
        while n >= _TIER_BASE * _TIER_FAN ** (tier + 1):
            tier += 1
        return tier

    def _merge_candidates(self) -> list[index.Segment]:
        segs = list(self._segments)
        tiers: dict[int, list] = {}
        for s in segs:
            tiers.setdefault(self._tier(s), []).append(s)
        for tier in sorted(tiers):
            if len(tiers[tier]) >= _TIER_FAN:
                return tiers[tier]
        if len(segs) > _MAX_SEGMENTS:
            return sorted(segs, key=lambda s: s.count)[:_TIER_FAN]
        return []

    def merge(self, sources: list[index.Segment] | None = None) -> int:
        """Merge ``sources`` (default: what the tiers call for) into one
        segment. Returns entries written (0: nothing to merge)."""
        with self._lock:
            if self._merging:
                return 0
            sources = list(sources) if sources is not None else self._merge_candidates()
            if len(sources) < 2:
                return 0
            self._merging = True
            dead = self._dead.copy()
            everything = len(sources) == len(self._segments)
            name = self._next_name()
        try:
            out = self.folder / name
            journal = self.folder / f"merge-{out.stem}.json"
            journal.write_text(
                json.dumps({"output": name, "sources": [s.path.name for s in sources]})
            )
            t0 = time.monotonic()
            n = index.merge(sources, out, dead, stop=self._stop.is_set)
            merged = index.Segment(out)
            with self._lock:
                keep = [s for s in self._segments if s not in sources]
                self._segments = keep + [merged]
                if everything and len(dead):
                    # No segment holds these ids any more
                    with self.connect() as c:
                        c.executemany("DELETE FROM dead WHERE id=?", [(int(i),) for i in dead])
                    self._dead = np.setdiff1d(self._dead, dead)
                self._counts = None
            # Not closed: a search may still read them from its snapshot —
            # the mapping lives on until the last reference goes
            for s in sources:
                s.path.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
            logger.info(
                "Audio analysis: %d segments merged into %s (%s entries, %.0f s)",
                len(sources),
                name,
                f"{n:,}",
                time.monotonic() - t0,
            )
            return n
        except InterruptedError:
            (self.folder / f"merge-{Path(name).stem}.json").unlink(missing_ok=True)
            return 0
        finally:
            with self._lock:
                self._merging = False

    def _maintain(self) -> None:
        """Every minute: write what is in memory once it is due, merge what
        the tiers call for."""
        while not self._stop.wait(60.0):
            try:
                with self._lock:
                    due = self._pend_n and (
                        self._pend_n >= FLUSH_ENTRIES
                        or time.monotonic() - self._pend_since >= FLUSH_AFTER_SEC
                    )
                if due:
                    self.flush()
                while not self._stop.is_set() and self.merge():
                    pass
            except Exception:  # noqa: BLE001 — the next minute tries again
                logger.exception("Audio analysis: index maintenance failed")

    def close(self) -> None:
        """Write what is in memory; stop the maintenance."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            logger.exception("Audio analysis: writing the fingerprints in memory failed")
        for s in self._segments:
            s.close()

    # ── Reading ──────────────────────────────────────────────────────────

    def snapshot(self) -> Snapshot:
        with self._lock:
            if self._pend_cache is None:
                if self._pend_n:
                    self._pend_cache = (np.concatenate(self._pend_h), np.concatenate(self._pend_e))
                else:
                    self._pend_cache = (np.zeros(0, np.uint32), np.zeros(0, np.uint64))
            return Snapshot(list(self._segments), *self._pend_cache, self._dead)

    def file(self, path: str) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                f"SELECT {', '.join(FILE_COLUMNS)} FROM files WHERE path=?", (path,)
            ).fetchone()
        return dict(row) if row else None

    def files_by_id(self, ids) -> dict[int, dict]:
        ids = [int(i) for i in ids]
        out: dict[int, dict] = {}
        with self.connect() as c:
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                for row in c.execute(
                    f"SELECT {', '.join(FILE_COLUMNS)} FROM files WHERE id IN "
                    f"({','.join('?' * len(chunk))})",
                    chunk,
                ):
                    out[row["id"]] = dict(row)
        return out

    def is_current(self, path: str, size: int, mtime: float) -> bool:
        """Has ``path`` been analyzed as it is now (done or failed)?"""
        row = self.file(path)
        return bool(row) and row["size"] == size and abs(row["mtime"] - mtime) < 0.01

    def levels(
        self, path: str, size: int | None = None, mtime: float | None = None
    ) -> bytes | None:
        """The stored waveform of ``path`` — None when there is none, or it
        is of another version of the file."""
        with self.connect() as c:
            row = c.execute(
                "SELECT f.size, f.mtime, w.levels FROM files f JOIN waveforms w "
                "ON w.file_id = f.id WHERE f.path=?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        if size is not None and row[0] != size:
            return None
        if mtime is not None and abs(row[1] - mtime) >= 0.01:
            return None
        return bytes(row[2])

    def counts(self, ttl: float = 5.0) -> dict:
        """Recordings, hours, entries and disk use — kept ``ttl`` seconds."""
        now = time.monotonic()
        if self._counts and now - self._counts[0] < ttl:
            return self._counts[1]
        with self.connect() as c:
            row = c.execute(
                "SELECT count(*), coalesce(sum(duration), 0), coalesce(sum(hashes), 0) "
                "FROM files WHERE status='done'"
            ).fetchone()
            errors = c.execute("SELECT count(*) FROM files WHERE status='error'").fetchone()[0]
            same = c.execute("SELECT count(*) FROM relations WHERE kind='same'").fetchone()[0]
            contains = c.execute("SELECT count(*) FROM relations WHERE kind='contains'").fetchone()[
                0
            ]
        with self._lock:
            segs = list(self._segments)
            pending = self._pend_n
        disk = sum(s.nbytes for s in segs)
        try:
            disk += self.path.stat().st_size
            wal = self.path.with_name(self.path.name + "-wal")
            if wal.exists():
                disk += wal.stat().st_size
        except OSError:
            pass
        out = {
            "files": row[0],
            "hours": round(row[1] / 3600.0, 2),
            "entries": int(row[2]),
            "errors": errors,
            "segments": len(segs),
            "pending": pending,
            "disk_bytes": disk,
            "same": same,
            "contains": contains,
        }
        self._counts = (now, out)
        return out

    # ── Writing ──────────────────────────────────────────────────────────

    def _retire(self, c, row) -> None:
        """A recording's old row goes: its entries are dropped from memory,
        or marked dead in the segments."""
        rid = row["id"]
        with self._lock:
            if rid in self._pend_ids:
                keep_h, keep_e = [], []
                for h, e in zip(self._pend_h, self._pend_e, strict=True):
                    ids, _ = index.unpack(e)
                    mask = ids != rid
                    keep_h.append(h[mask])
                    keep_e.append(e[mask])
                self._pend_h, self._pend_e = keep_h, keep_e
                self._pend_n = sum(len(h) for h in keep_h)
                self._pend_ids = [i for i in self._pend_ids if i != rid]
                self._pend_cache = None
            elif row["stored"] and row["hashes"]:
                c.execute("INSERT OR IGNORE INTO dead(id) VALUES(?)", (rid,))
                self._dead = np.union1d(self._dead, [rid])
        c.execute("DELETE FROM files WHERE id=?", (rid,))

    def add(
        self,
        path: str,
        size: int,
        mtime: float,
        *,
        result: dict | None = None,
        hashes: np.ndarray | None = None,
        frames: np.ndarray | None = None,
        relations: list[dict] | None = None,
        source: str = "",
        seconds: float = 0.0,
        error: str = "",
    ) -> int:
        """Store a recording's analysis (or ``error``) — replacing an earlier
        one of the same path. Returns its id."""
        hashes = np.zeros(0, np.uint32) if hashes is None else np.asarray(hashes, np.uint32)
        frames = np.zeros(0, np.uint32) if frames is None else np.asarray(frames, np.int64)
        if len(frames) and frames.max() > index.MAX_T:
            keep = frames <= index.MAX_T
            hashes, frames = hashes[keep], frames[keep]
        r = result or {}
        with self._lock, self.connect() as c:
            old = c.execute("SELECT id, stored, hashes FROM files WHERE path=?", (path,)).fetchone()
            if old is not None:
                self._retire(c, old)
            cur = c.execute(
                "INSERT INTO files(path, size, mtime, duration, status, error, source, "
                "analyzed_at, "
                "seconds, lufs, lra, peak_db, silence, clipped, hashes, stored) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    path,
                    int(size),
                    float(mtime),
                    r.get("duration"),
                    "error" if error else "done",
                    error or None,
                    source,
                    time.time(),
                    round(seconds, 2),
                    r.get("lufs"),
                    r.get("lra"),
                    r.get("peak_db"),
                    r.get("silence"),
                    r.get("clipped"),
                    0 if error else len(hashes),
                ),
            )
            rid = int(cur.lastrowid)
            if rid > index.MAX_ID:
                raise RuntimeError("The audio index has run out of recording ids — rebuild it")
            if not error and r.get("levels"):
                c.execute(
                    "INSERT INTO waveforms(file_id, levels) VALUES(?, ?)",
                    (rid, sqlite3.Binary(r["levels"])),
                )
            for rel in relations or []:
                # ``a`` holds ``b`` (or is the same): the new recording is
                # ``a`` when it is the one that contains the other
                other = int(rel["other"])
                a, b = (rid, other) if rel.get("new_contains") else (other, rid)
                if c.execute("SELECT 1 FROM files WHERE id=?", (other,)).fetchone() is None:
                    continue
                c.execute(
                    "INSERT OR REPLACE INTO relations(a, b, kind, offset, matched, score, "
                    "found_at) VALUES(?,?,?,?,?,?,?)",
                    (
                        a,
                        b,
                        rel["kind"],
                        float(rel["offset"]),
                        float(rel["matched"]),
                        int(rel["score"]),
                        time.time(),
                    ),
                )
            if not error and len(hashes):
                if not self._pend_n:
                    self._pend_since = time.monotonic()
                self._pend_h.append(hashes)
                self._pend_e.append(index.pack(np.full(len(frames), rid, np.uint64), frames))
                self._pend_ids.append(rid)
                self._pend_n += len(hashes)
                self._pend_cache = None
            self._counts = None
        if self._pend_n >= FLUSH_ENTRIES and not self._flushing:
            self._flushing = True
            threading.Thread(target=self._flush_quietly, name="audio flush", daemon=True).start()
        return rid

    def _flush_quietly(self) -> None:
        try:
            self.flush()
        except Exception:  # noqa: BLE001 — the maintenance tries again
            logger.exception("Audio analysis: writing a segment failed")
        finally:
            self._flushing = False

    def remove(self, path: str) -> bool:
        """Forget ``path`` (deleted from the library)."""
        with self._lock, self.connect() as c:
            row = c.execute("SELECT id, stored, hashes FROM files WHERE path=?", (path,)).fetchone()
            if row is None:
                return False
            self._retire(c, row)
            self._counts = None
        return True

    def clear(self) -> None:
        """Empty the index: every row, every segment."""
        with self._lock:
            # (not closed: a search may still read them — see merge())
            self._segments = []
            self._pend_h, self._pend_e, self._pend_ids = [], [], []
            self._pend_n = 0
            self._pend_cache = None
            self._dead = np.zeros(0, np.int64)
            with self.connect() as c:
                self._wipe(c)
            self._counts = None


_stores: dict[str, AudioStore] = {}
_stores_lock = threading.Lock()


def store_path() -> Path:
    from whisper_tm.config import config

    p = config.path_setting("audio_analysis", "path")
    if p:
        return p
    return Path(config.db_path).with_name("audio.db")


def get_store() -> AudioStore:
    """The store of the configured path — one per path, kept."""
    p = str(store_path())
    with _stores_lock:
        if p not in _stores:
            _stores[p] = AudioStore(p)
        return _stores[p]


def close_all() -> None:
    """At shutdown: what is in memory goes to disk."""
    with _stores_lock:
        stores = list(_stores.values())
        _stores.clear()
    for s in stores:
        s.close()

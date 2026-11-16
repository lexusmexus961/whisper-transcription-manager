"""The media index: ``media.db`` next to the main database.

Three tables:

* ``files`` — one row per indexed recording: how far it got, with which
  settings (``settings``: what its moments depend on; ``kinds``: which
  vectors its moments have), the last kept still (where a continued run
  picks up the repeat check);
* ``moments`` — a moment of a recording: its time, its sound window, its
  thumbnail, what is said around it;
* ``items`` — a vector of a moment, one per kind (image, video,
  videoaudio, audio, speech, description).

**Stored as 8-bit numbers.** A vector keeps all ``dims`` numbers of the
model (768), each rounded to an 8-bit integer of the vector's own scale —
768 bytes a vector, less than 512 numbers as float16 (1,024 bytes), and
closer to the model: on a labeled sound test set the mean AP was
0.251 with 768 numbers, 0.189 with 512 and 0.201 with 256. Rounding moves a
cosine by about 0.001.

**Searching.** RAM keeps the stored 8-bit vectors themselves, all their
numbers, in blocks of ``BLOCK`` rows packed for fbgemm (PyTorch's int8
matrix kernel): a query scores every vector — 26 ms for a million on a
desktop CPU, against 40 ms as float16 with twice the RAM (768
bytes a vector instead of 1,536). The block being filled stays float16
until it is full. The best ``SHORTLIST`` are scored again in float32 from
the database, which also hands over their vectors. A first pass on fewer
numbers was measured and dropped: on a few thousand sound windows and stills,
the true top 10 by all numbers sat as deep as 2–3 % of the collection
(median query) when ranked by the first 128, 33 % at worst — at millions of
vectors no shortlist reaches that far; the signs of all numbers (1 bit
each) did worse. New rows go onto the last block in place — no matrix is
copied while an index run adds thousands an hour — and removed rows simply
aren't found in the second pass (the matrices are read again once a tenth
of their rows are gone). Where fbgemm is missing, the blocks stay float16.

**Counting** is kept in memory and moved with every write: counting
millions of rows takes seconds, and the settings page asks every two.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: ``description``: the language model's sentence about a moment's still,
#: embedded as text (media/describe.py)
KINDS = ("image", "video", "videoaudio", "audio", "speech", "description")
DIMS = (768, 512, 256, 128)
#: How new vectors are stored
FORMAT = "int8"
#: Rows of a block of the kept matrices
BLOCK = 65536
#: Rows scored again exactly from the database, at least (and 10 per wanted
#: hit): the first pass already scores all numbers, rounded to 8 bits
SHORTLIST = 1000
#: The same when a search asks with fewer numbers than stored (the first
#: pass still uses all of them)
SHORTLIST_FEWER = 20_000
#: How long the per-status file counts are kept
FILES_TTL_SEC = 10.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    video       INTEGER NOT NULL,
    duration    REAL,
    settings    TEXT,
    status      TEXT NOT NULL DEFAULT 'partial',
    next_t      REAL NOT NULL DEFAULT 0,
    prev        TEXT,
    moments     INTEGER NOT NULL DEFAULT 0,
    repeats     INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    seconds     REAL NOT NULL DEFAULT 0,
    indexed_at  REAL,
    kinds       TEXT
);
CREATE TABLE IF NOT EXISTS moments (
    id       INTEGER PRIMARY KEY,
    file_id  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    t        REAL NOT NULL,
    start    REAL NOT NULL,
    dur      REAL NOT NULL,
    repeat   INTEGER NOT NULL DEFAULT 0,
    thumb    TEXT,
    text     TEXT
);
CREATE INDEX IF NOT EXISTS moments_file ON moments(file_id, t);
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY,
    moment_id  INTEGER NOT NULL REFERENCES moments(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS items_kind ON items(kind);
CREATE INDEX IF NOT EXISTS items_moment ON items(moment_id);
CREATE TABLE IF NOT EXISTS descriptions (
    moment_id  INTEGER PRIMARY KEY REFERENCES moments(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    model      TEXT,
    at         REAL
);
"""


def unit(v: np.ndarray, dims: int) -> np.ndarray:
    """The first ``dims`` numbers, unit length again (float32)."""
    v = np.asarray(v, dtype=np.float32)[..., :dims]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def encode(v: np.ndarray, dims: int) -> bytes:
    """A vector as stored: its first ``dims`` numbers, as 8-bit integers of
    its own scale."""
    u = unit(v, dims)
    m = float(np.max(np.abs(u))) or 1e-12
    return np.clip(np.rint(u * (127.0 / m)), -127, 127).astype(np.int8).tobytes()


def decode(blob: bytes) -> np.ndarray:
    """A stored vector, unit length (float32). The scale of an 8-bit vector
    doesn't matter: only directions are compared."""
    v = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def bytes_per_vector(dims: int) -> int:
    return dims


class _Matrix:
    """Every vector of one kind, all its numbers: full blocks packed for
    fbgemm (8-bit), the block being filled as float16."""

    def __init__(self, dims: int):
        self.dims = dims
        self.blocks: list[dict] = []  # full: {"packed" | "X16", "n", "ids", "moments", "files"}
        self.tail = _block(dims, 1024)
        self.dead = 0

    def __len__(self) -> int:
        return sum(b["n"] for b in self.blocks) + self.tail["n"]

    def append(self, ids, moments, files, X) -> None:
        """``X``: unit float16 rows. The data goes in before the row count
        moves, so a search running alongside reads only finished rows."""
        k, total = 0, len(ids)
        while k < total:
            t = self.tail
            take = min(BLOCK - t["n"], total - k)
            if t["n"] + take > len(t["ids"]):
                _grow(t, min(BLOCK, max(t["n"] + take, 2 * len(t["ids"]))))
            sl = slice(t["n"], t["n"] + take)
            t["X16"][sl] = X[k : k + take]
            t["ids"][sl] = ids[k : k + take]
            t["moments"][sl] = moments[k : k + take]
            t["files"][sl] = files[k : k + take]
            t["n"] += take
            k += take
            if t["n"] == BLOCK:
                self.blocks.append(_seal(t))
                self.tail = _block(self.dims, 1024)

    def add_full(self, ids, moments, files, Q8) -> None:
        """A full block of stored 8-bit rows (reading the index)."""
        b = {"n": len(ids), "ids": ids, "moments": moments, "files": files}
        packed = _pack(Q8)
        if packed is None:
            b["X16"] = unit(Q8.astype(np.float32), Q8.shape[1]).astype(np.float16)
        else:
            b["packed"] = packed
        self.blocks.append(b)

    def ram_bytes(self) -> int:
        total = 0
        for b in list(self.blocks) + [self.tail]:
            total += sum(b[k].nbytes for k in ("ids", "moments", "files"))
            # packed: a byte a number and the row's scale and zero point
            total += b["X16"].nbytes if "X16" in b else b["n"] * (self.dims + 16)
        return total


def _block(dims: int, cap: int) -> dict:
    return {
        "n": 0,
        "X16": np.empty((cap, dims), dtype=np.float16),
        "ids": np.empty(cap, dtype=np.int64),
        "moments": np.empty(cap, dtype=np.int64),
        "files": np.empty(cap, dtype=np.int64),
    }


def _grow(b: dict, cap: int) -> None:
    n = b["n"]
    for key in ("X16", "ids", "moments", "files"):
        old = b[key]
        new = np.empty((cap,) + old.shape[1:], dtype=old.dtype)
        new[:n] = old[:n]
        b[key] = new


def _seal(t: dict) -> dict:
    """A full float16 block becomes a packed 8-bit one."""
    n = t["n"]
    X = t["X16"][:n].astype(np.float32)
    m = np.maximum(np.abs(X).max(axis=1, keepdims=True), 1e-12)
    Q8 = np.clip(np.rint(X * (127.0 / m)), -127, 127).astype(np.int8)
    b = {
        "n": n,
        "ids": t["ids"][:n].copy(),
        "moments": t["moments"][:n].copy(),
        "files": t["files"][:n].copy(),
    }
    packed = _pack(Q8)
    if packed is None:
        b["X16"] = t["X16"][:n].copy()
    else:
        b["packed"] = packed
    return b


_FBGEMM: dict = {"ok": None}


def _pack(Q8: np.ndarray):
    """8-bit rows packed for ``torch.ops.quantized.linear_dynamic``, each
    scaled to unit length — or None where fbgemm is missing."""
    if _FBGEMM["ok"] is False:
        return None
    try:
        import torch

        n = len(Q8)
        norms = np.linalg.norm(Q8.astype(np.float32), axis=1)
        scales = torch.from_numpy((1.0 / np.maximum(norms, 1e-12)).astype(np.float64))
        rows = Q8 if Q8.flags.writeable and Q8.flags.c_contiguous else np.array(Q8, dtype=np.int8)
        W = torch._make_per_channel_quantized_tensor(
            torch.from_numpy(rows), scales, torch.zeros(n, dtype=torch.long), 0
        )
        packed = torch.ops.quantized.linear_prepack(W, None)
        _FBGEMM["ok"] = True
        return packed
    except Exception as e:  # noqa: BLE001 — float16 blocks then
        if _FBGEMM["ok"] is None:
            logger.info("Media index: no fbgemm (%s) — blocks stay float16", e)
        _FBGEMM["ok"] = False
        return None


def _scores(b: dict, n: int, q: np.ndarray) -> np.ndarray:
    """The cosines of a block's first ``n`` rows with the unit query."""
    if "packed" in b:
        import torch

        out = torch.ops.quantized.linear_dynamic(torch.from_numpy(q).unsqueeze(0), b["packed"])
        return out.numpy().reshape(-1)[:n]
    from whisper_tm.rag.vector_db import similarities

    return np.asarray(similarities(b["X16"][:n], q), dtype=np.float32).reshape(-1)


class MediaStore:
    """``media.db`` and the kept matrices — one per database file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._mats: dict[str, _Matrix] | None = None
        self._mat_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._loading = False
        self._pending: list[dict] = []
        self._stale = False
        self._tally: dict | None = None
        self._tally_lock = threading.Lock()
        self._files_cache: tuple[float, dict] | None = None
        self.generation = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)

    # ── Connections ──────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
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

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ── Meta ─────────────────────────────────────────────────────────────

    def meta(self, key: str, default=None):
        with self.connect() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except ValueError:
            return row[0]

    def set_meta(self, key: str, value) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def dims(self) -> int | None:
        """The size the index stores — None while it is empty and unset."""
        d = self.meta("dims")
        return int(d) if d else None

    def format(self) -> str:
        """How the vectors are stored (``int8``)."""
        return FORMAT

    def ensure_dims(self, dims: int) -> int:
        """Fix the stored size (and layout) on the first write; later ones
        keep it."""
        have = self.dims()
        if have:
            return have
        self.set_meta("dims", int(dims))
        self.set_meta("format", FORMAT)
        return int(dims)

    # ── Files ────────────────────────────────────────────────────────────

    def file(self, path: str) -> dict | None:
        with self.connect() as c:
            c.row_factory = sqlite3.Row
            try:
                row = c.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
            finally:
                c.row_factory = None
        if not row:
            return None
        out = dict(row)
        out["kinds"] = _kinds(out.get("kinds"))
        return out

    def start_file(
        self,
        path: str,
        *,
        video: bool,
        duration: float,
        settings: str,
        kinds: list[str] | None = None,
        fresh: bool = False,
    ) -> dict:
        """The file's row — new, or reset when its moments were made with
        other settings (they go). ``kinds``: the vectors this pass writes;
        ``fresh``: reset it anyway."""
        cur = self.file(path)
        if cur and cur["settings"] == settings and not fresh:
            return cur
        with self._lock, self.connect() as c:
            if cur:
                gone = self._file_tally(c, cur["id"])
                c.execute("DELETE FROM files WHERE id=?", (cur["id"],))
            c.execute(
                "INSERT INTO files(path, video, duration, settings, status, kinds) "
                "VALUES(?,?,?,?, 'partial', ?)",
                (path, int(video), duration, settings, json.dumps(sorted(kinds or []))),
            )
        if cur:
            self._move(gone, -1)
            self._forget(gone["kinds"])
        self._files_cache = None
        return self.file(path)

    def set_file_kinds(self, file_id: int, kinds: list[str]) -> None:
        with self.connect() as c:
            c.execute("UPDATE files SET kinds=? WHERE id=?", (json.dumps(sorted(kinds)), file_id))

    def add_moments(
        self,
        file_id: int,
        rows: list[dict],
        *,
        next_t: float,
        prev: str | None,
        done: bool,
        seconds: float,
    ) -> int:
        """Store a chunk: ``rows`` = ``[{"t", "start", "dur", "repeat",
        "thumb", "text", "vectors": {kind: float32 vector}}]``. Returns the
        number of vectors written."""
        dims = self.dims()
        if not dims:
            raise RuntimeError("index size not set")
        added: dict[str, list] = {}
        moved = {"kinds": Counter(), "moments": 0, "repeats": 0, "stills": 0}
        with self._lock, self.connect() as c:
            for r in rows:
                rep = bool(r.get("repeat"))
                cur = c.execute(
                    "INSERT INTO moments(file_id, t, start, dur, repeat, thumb, text) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        file_id,
                        r["t"],
                        r["start"],
                        r["dur"],
                        int(rep),
                        r.get("thumb"),
                        r.get("text") or None,
                    ),
                )
                mid = cur.lastrowid
                moved["moments"] += 1
                moved["repeats"] += int(rep)
                moved["stills"] += int(not rep and bool(r.get("thumb")))
                for kind, vec in (r.get("vectors") or {}).items():
                    cur = c.execute(
                        "INSERT INTO items(moment_id, kind, vector) VALUES(?,?,?)",
                        (mid, kind, encode(vec, dims)),
                    )
                    added.setdefault(kind, []).append((cur.lastrowid, mid, file_id, vec))
                    moved["kinds"][kind] += 1
            c.execute(
                "UPDATE files SET next_t=?, prev=?, status=?, moments=moments+?, "
                "repeats=repeats+?, seconds=seconds+?, indexed_at=?, error=NULL WHERE id=?",
                (
                    next_t,
                    prev,
                    "done" if done else "partial",
                    len(rows),
                    moved["repeats"],
                    seconds,
                    time.time(),
                    file_id,
                ),
            )
        self._move(moved, +1)
        self._append(added)
        if done:
            self._files_cache = None
        return sum(moved["kinds"].values())

    def finish_file(self, file_id: int, seconds: float = 0.0) -> None:
        """A pass over a file's moments that wrote no new moments (vectors
        of a kind added later) is done."""
        with self.connect() as c:
            c.execute(
                "UPDATE files SET status='done', seconds=seconds+?, indexed_at=?, error=NULL "
                "WHERE id=?",
                (seconds, time.time(), file_id),
            )
        self._files_cache = None

    def add_items(self, rows: list[tuple[int, str, np.ndarray]]) -> int:
        """Vectors for moments that exist already: ``[(moment id, kind,
        vector)]`` — a kind added later, or a description. One of a kind per
        moment: an older one is replaced."""
        dims = self.dims()
        if not dims or not rows:
            return 0
        added: dict[str, list] = {}
        replaced: Counter = Counter()
        with self._lock, self.connect() as c:
            for mid, kind, vec in rows:
                row = c.execute("SELECT file_id FROM moments WHERE id=?", (mid,)).fetchone()
                if not row:
                    continue
                replaced[kind] += c.execute(
                    "DELETE FROM items WHERE moment_id=? AND kind=?", (mid, kind)
                ).rowcount
                cur = c.execute(
                    "INSERT INTO items(moment_id, kind, vector) VALUES(?,?,?)",
                    (mid, kind, encode(vec, dims)),
                )
                added.setdefault(kind, []).append((cur.lastrowid, mid, row[0], vec))
        new = Counter({k: len(v) for k, v in added.items()})
        self._move({"kinds": new}, +1)
        if sum(replaced.values()):
            self._move({"kinds": replaced}, -1)
            self._forget(replaced)
        self._append(added)
        return sum(new.values())

    def file_moments(self, file_id: int) -> list[dict]:
        """A file's moments in time order: ``[{"id", "t", "start", "dur",
        "repeat", "thumb", "text"}]``."""
        with self.connect() as c:
            rows = c.execute(
                "SELECT id, t, start, dur, repeat, thumb, text FROM moments "
                "WHERE file_id=? ORDER BY t",
                (file_id,),
            ).fetchall()
        keys = ("id", "t", "start", "dur", "repeat", "thumb", "text")
        return [dict(zip(keys, r)) for r in rows]

    def update_moments(self, file_id: int, rows: list[tuple[int, bool, str | None]]) -> None:
        """``[(moment id, repeat, thumb)]`` — what a later look at the
        pictures found (stills added to a file indexed without them)."""
        if not rows:
            return
        with self._lock, self.connect() as c:
            ids = [r[0] for r in rows]
            old = {}
            for k in range(0, len(ids), 900):
                part = ids[k : k + 900]
                for mid, rep, thumb in c.execute(
                    "SELECT id, repeat, thumb FROM moments "
                    f"WHERE id IN ({','.join('?' * len(part))})",
                    part,
                ):
                    old[mid] = (bool(rep), thumb)
            c.executemany(
                "UPDATE moments SET repeat=?, thumb=? WHERE id=?",
                [(int(bool(rep)), thumb, mid) for mid, rep, thumb in rows],
            )
            c.execute(
                "UPDATE files SET repeats=(SELECT coalesce(sum(repeat), 0) FROM moments "
                "WHERE file_id=?) WHERE id=?",
                (file_id, file_id),
            )
        d_rep = d_still = 0
        for mid, rep, thumb in rows:
            o_rep, o_thumb = old.get(mid, (False, None))
            d_rep += int(bool(rep)) - int(o_rep)
            d_still += int(not rep and bool(thumb)) - int(not o_rep and bool(o_thumb))
        self._move({"repeats": d_rep, "stills": d_still}, +1)

    def delete_kinds(self, file_id: int, kinds: list[str]) -> int:
        """Remove a file's vectors of these kinds (before they are made
        again, with other settings)."""
        if not kinds:
            return 0
        with self._lock, self.connect() as c:
            gone = Counter(
                dict(
                    c.execute(
                        "SELECT i.kind, count(*) FROM items i "
                        "JOIN moments m ON m.id = i.moment_id "
                        f"WHERE m.file_id=? AND i.kind IN ({','.join('?' * len(kinds))}) "
                        "GROUP BY i.kind",
                        (file_id, *kinds),
                    ).fetchall()
                )
            )
            c.execute(
                f"DELETE FROM items WHERE kind IN ({','.join('?' * len(kinds))}) "
                "AND moment_id IN (SELECT id FROM moments WHERE file_id=?)",
                (*kinds, file_id),
            )
        if gone:
            self._move({"kinds": gone}, -1)
            self._forget(gone)
        return sum(gone.values())

    def undescribed(self, limit: int, prefixes: list[str] | None = None) -> list[dict]:
        """Moments with a still of their own and no description vector
        yet, newest recordings first: ``[{"id", "thumb", "path", "t", "text"}]``
        (``text``: a sentence written before, not embedded yet; ``path`` and
        ``t`` to cut a still that was never written — the thumbnail cap)."""
        where, args = "", []
        if prefixes:
            where = " AND (" + " OR ".join("f.path LIKE ? ESCAPE '\\'" for _ in prefixes) + ")"
            args = [_like_prefix(p) for p in prefixes]
        with self.connect() as c:
            rows = c.execute(
                "SELECT m.id, m.thumb, f.path, m.t, d.text FROM moments m "
                "JOIN files f ON f.id = m.file_id "
                "LEFT JOIN descriptions d ON d.moment_id = m.id "
                "WHERE m.repeat = 0 AND m.thumb IS NOT NULL AND NOT EXISTS "
                "(SELECT 1 FROM items i WHERE i.moment_id = m.id AND i.kind = 'description')"
                f"{where} ORDER BY f.indexed_at DESC, m.t LIMIT ?",
                (*args, int(limit)),
            ).fetchall()
        return [
            {"id": mid, "thumb": thumb, "path": path, "t": t, "text": text}
            for mid, thumb, path, t, text in rows
        ]

    def description_counts(self) -> dict:
        c = self.counts()
        return {"described": c["kinds"].get("description", 0), "stills": c["stills"]}

    def fail_file(self, file_id: int, error: str) -> None:
        with self.connect() as c:
            c.execute("UPDATE files SET status='error', error=? WHERE id=?", (error[:500], file_id))
        self._files_cache = None

    def remove_files(self, paths: list[str]) -> int:
        with self._lock, self.connect() as c:
            n = 0
            for p in paths:
                n += c.execute("DELETE FROM files WHERE path=?", (p,)).rowcount
        if n:
            self._invalidate()
        return n

    def clear(self) -> None:
        with self._lock, self.connect() as c:
            c.execute("DELETE FROM descriptions")
            c.execute("DELETE FROM items")
            c.execute("DELETE FROM moments")
            c.execute("DELETE FROM files")
            c.execute(
                "DELETE FROM meta WHERE key IN ('dims', 'format', 'clip_frames', 'thumb_bytes')"
            )
        with self.connect() as c:
            c.execute("VACUUM")
        self._invalidate()

    def file_states(self) -> dict[str, dict]:
        """``{path: {"status", "settings", "kinds", "next_t", "duration"}}``
        of every indexed file."""
        with self.connect() as c:
            rows = c.execute(
                "SELECT path, status, settings, kinds, next_t, duration FROM files"
            ).fetchall()
        return {
            p: {
                "status": st,
                "settings": se,
                "kinds": _kinds(k),
                "next_t": nt or 0.0,
                "duration": du or 0.0,
            }
            for p, st, se, k, nt, du in rows
        }

    # ── Counting ─────────────────────────────────────────────────────────

    def _file_tally(self, c, file_id: int) -> dict:
        kinds = Counter(
            dict(
                c.execute(
                    "SELECT i.kind, count(*) FROM items i JOIN moments m ON m.id = i.moment_id "
                    "WHERE m.file_id=? GROUP BY i.kind",
                    (file_id,),
                ).fetchall()
            )
        )
        n, rep, stills = c.execute(
            "SELECT count(*), coalesce(sum(repeat), 0), "
            "coalesce(sum(CASE WHEN repeat = 0 AND thumb IS NOT NULL THEN 1 ELSE 0 END), 0) "
            "FROM moments WHERE file_id=?",
            (file_id,),
        ).fetchone()
        return {"kinds": kinds, "moments": n, "repeats": rep, "stills": stills}

    def _count_all(self) -> dict:
        with self.connect() as c:
            kinds = Counter(
                dict(c.execute("SELECT kind, count(*) FROM items GROUP BY kind").fetchall())
            )
            n, rep, stills = c.execute(
                "SELECT count(*), coalesce(sum(repeat), 0), "
                "coalesce(sum(CASE WHEN repeat = 0 AND thumb IS NOT NULL THEN 1 ELSE 0 END), 0) "
                "FROM moments"
            ).fetchone()
        return {"kinds": kinds, "moments": int(n), "repeats": int(rep), "stills": int(stills)}

    def _move(self, d: dict, sign: int) -> None:
        with self._tally_lock:
            t = self._tally
            if t is None:
                return
            for k, v in (d.get("kinds") or {}).items():
                t["kinds"][k] += sign * v
            for key in ("moments", "repeats", "stills"):
                t[key] += sign * int(d.get(key) or 0)

    def counts(self, fresh: bool = False) -> dict:
        """What the index holds. Vector and moment counts are kept in memory
        and moved with every write; the per-status file counts are kept
        ``FILES_TTL_SEC``."""
        with self._tally_lock:
            t = None if fresh else self._tally
        if t is None:
            t = self._count_all()
            with self._tally_lock:
                self._tally = t
        now = time.monotonic()
        fc = self._files_cache
        if fresh or fc is None or now - fc[0] > FILES_TTL_SEC:
            with self.connect() as c:
                files = dict(
                    c.execute("SELECT status, count(*) FROM files GROUP BY status").fetchall()
                )
                secs = c.execute(
                    "SELECT coalesce(sum(seconds), 0), coalesce(sum(duration), 0) FROM files "
                    "WHERE status='done'"
                ).fetchone()
            fc = (now, {"files": files, "secs": secs})
            self._files_cache = fc
        size = 0
        for suffix in ("", "-wal"):
            p = Path(str(self.path) + suffix)
            if p.exists():
                size += p.stat().st_size
        with self._tally_lock:
            kinds = {k: max(0, int(t["kinds"].get(k, 0))) for k in KINDS}
            moments, repeats, stills = t["moments"], t["repeats"], t["stills"]
        return {
            "kinds": kinds,
            "files": dict(fc[1]["files"]),
            "moments": max(0, int(moments)),
            "repeats": max(0, int(repeats)),
            "stills": max(0, int(stills)),
            "dims": self.dims(),
            "format": self.format(),
            "gpu_seconds": round(fc[1]["secs"][0], 1),
            "media_seconds": round(fc[1]["secs"][1], 1),
            "bytes": size,
        }

    # ── Moments ──────────────────────────────────────────────────────────

    def moment(self, moment_id: int) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT m.id, m.file_id, m.t, m.start, m.dur, m.repeat, m.thumb, m.text, "
                "f.path, f.video "
                "FROM moments m JOIN files f ON f.id = m.file_id WHERE m.id=?",
                (moment_id,),
            ).fetchone()
        if not row:
            return None
        keys = ("id", "file_id", "t", "start", "dur", "repeat", "thumb", "text", "path", "video")
        return dict(zip(keys, row))

    def item(self, item_id: int) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT id, moment_id, kind, vector FROM items WHERE id=?", (item_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "moment_id": row[1],
            "kind": row[2],
            "vector": decode(row[3]),
        }

    def vectors(self, item_ids: list[int]) -> dict[int, np.ndarray]:
        """``{item id: unit vector}`` of the items that still exist."""
        out: dict[int, np.ndarray] = {}
        ids = list(dict.fromkeys(int(i) for i in item_ids))
        with self.connect() as c:
            for k in range(0, len(ids), 900):
                part = ids[k : k + 900]
                for iid, blob in c.execute(
                    f"SELECT id, vector FROM items WHERE id IN ({','.join('?' * len(part))})", part
                ):
                    out[iid] = decode(blob)
        return out

    def describe(self, moment_id: int) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT text, model, at FROM descriptions WHERE moment_id=?", (moment_id,)
            ).fetchone()
        return {"text": row[0], "model": row[1], "at": row[2]} if row else None

    def set_description(self, moment_id: int, text: str, model: str) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT INTO descriptions(moment_id, text, model, at) VALUES(?,?,?,?) "
                "ON CONFLICT(moment_id) DO UPDATE SET text=excluded.text, "
                "model=excluded.model, at=excluded.at",
                (moment_id, text, model, time.time()),
            )

    # ── The kept matrices ────────────────────────────────────────────────

    def _invalidate(self) -> None:
        with self._mat_lock:
            self._mats = None
            self._pending = []
            self._stale = self._loading
            self.generation += 1
        with self._tally_lock:
            self._tally = None
        self._files_cache = None

    def _forget(self, gone: dict) -> None:
        """Rows were removed (``{kind: n}``): the first pass may still meet
        them, the second doesn't find them. Many of them → read again."""
        with self._mat_lock:
            self.generation += 1
            if self._mats is None:
                return
            for kind, n in gone.items():
                if kind in self._mats:
                    self._mats[kind].dead += int(n)
            total = sum(len(m) for m in self._mats.values())
            dead = sum(m.dead for m in self._mats.values())
            if total and dead > total / 10:
                self._mats = None

    def _append(self, added: dict[str, list]) -> None:
        """New rows go onto the kept matrices in place — or wait for a
        reading that is under way."""
        if not added:
            return
        with self._mat_lock:
            self.generation += 1
            if self._mats is None:
                if self._loading:
                    self._pending.append(added)
                return
            self._apply(self._mats, added)

    def _apply(self, mats: dict[str, _Matrix], added: dict[str, list]) -> None:
        dims = self.dims() or 768
        for kind, rows in added.items():
            m = mats.setdefault(kind, _Matrix(dims))
            X = unit(np.stack([np.asarray(r[3], dtype=np.float32) for r in rows]), m.dims)
            m.append(
                np.array([r[0] for r in rows], dtype=np.int64),
                np.array([r[1] for r in rows], dtype=np.int64),
                np.array([r[2] for r in rows], dtype=np.int64),
                X.astype(np.float16),
            )

    def matrices(self) -> dict[str, _Matrix]:
        """``{kind: _Matrix}`` — read once, kept, grown in place. Rows
        written while they are read are added after (a row read twice is
        sorted out when a search scores it)."""
        with self._mat_lock:
            if self._mats is not None:
                return self._mats
        with self._load_lock:
            with self._mat_lock:
                if self._mats is not None:
                    return self._mats
                self._loading, self._pending, self._stale = True, [], False
            try:
                mats = self._load()
            finally:
                with self._mat_lock:
                    self._loading = False
            with self._mat_lock:
                if self._stale:
                    self._stale = False
                    self._pending = []
                    stale = True
                else:
                    for added in self._pending:
                        self._apply(mats, added)
                    self._pending = []
                    self._mats = mats
                    stale = False
        if stale:  # emptied or files removed while reading
            return self.matrices()
        return mats

    def _load(self) -> dict[str, _Matrix]:
        t0 = time.time()
        dims = self.dims() or 768
        width = bytes_per_vector(dims)
        mats: dict[str, _Matrix] = {}
        with self.connect() as c:
            for kind in KINDS:
                m = _Matrix(dims)
                cur = c.execute(
                    "SELECT i.id, i.moment_id, m.file_id, i.vector FROM items i "
                    "JOIN moments m ON m.id = i.moment_id WHERE i.kind=? ORDER BY i.id",
                    (kind,),
                )
                while True:
                    rows = cur.fetchmany(BLOCK)
                    if not rows:
                        break
                    rows = [r for r in rows if len(r[3]) == width]
                    if not rows:
                        continue
                    n = len(rows)
                    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=n)
                    mids = np.fromiter((r[1] for r in rows), dtype=np.int64, count=n)
                    fids = np.fromiter((r[2] for r in rows), dtype=np.int64, count=n)
                    raw = b"".join(r[3] for r in rows)
                    Q8 = np.frombuffer(raw, dtype=np.int8).reshape(n, dims)
                    if n == BLOCK and not m.tail["n"]:
                        m.add_full(ids, mids, fids, Q8)
                    else:
                        m.append(
                            ids, mids, fids, unit(Q8.astype(np.float32), dims).astype(np.float16)
                        )
                mats[kind] = m
        total = sum(len(m) for m in mats.values())
        logger.info(
            "Media index: %d vectors loaded (%d numbers, %s; %s in RAM) in %.2f s",
            total,
            dims,
            FORMAT,
            "8-bit, fbgemm" if _FBGEMM["ok"] else "float16",
            time.time() - t0,
        )
        return mats

    def preload(self) -> None:
        threading.Thread(target=self._preload, name="media-matrix-preload", daemon=True).start()

    def _preload(self) -> None:
        try:
            self.matrices()
        except Exception as e:  # noqa: BLE001 — the first search loads it then
            logger.info("Media index preload: %s", e)

    def ram_bytes(self) -> int:
        """What the kept matrices take (0 while not loaded)."""
        with self._mat_lock:
            mats = self._mats
        if not mats:
            return 0
        return sum(m.ram_bytes() for m in mats.values())

    def file_ids_under(self, prefixes: list[str]) -> np.ndarray:
        """Ids of the files whose path starts with one of ``prefixes``."""
        if not prefixes:
            return np.empty(0, dtype=np.int64)
        with self.connect() as c:
            ids = [
                fid
                for fid, p in c.execute("SELECT id, path FROM files")
                if any(p.startswith(pre) for pre in prefixes)
            ]
        return np.asarray(ids, dtype=np.int64)

    def file_ids_of(self, paths) -> np.ndarray:
        """Ids of the indexed files among ``paths`` (absolute paths)."""
        want = set(p for p in paths if p)
        if not want:
            return np.empty(0, dtype=np.int64)
        with self.connect() as c:
            ids = [fid for fid, p in c.execute("SELECT id, path FROM files") if p in want]
        return np.asarray(ids, dtype=np.int64)

    def rank(
        self,
        q: np.ndarray,
        kind: str,
        k: int,
        *,
        dims: int | None = None,
        files: np.ndarray | None = None,
        exclude: int | None = None,
    ) -> list[tuple[int, int, float, np.ndarray]]:
        """The best ``k`` items of ``kind`` for the query ``q``: ``[(item
        id, moment id, score, vector)]`` — ``vector`` the item's, unit,
        ``dims`` numbers. ``dims``: fewer numbers than stored (Matryoshka);
        ``files``: only these files; ``exclude``: an item id."""
        m = self.matrices().get(kind)
        if m is None:
            return []
        # what is written now — an index run may add more while this runs
        blocks = list(m.blocks) + [m.tail]
        ns = [b["n"] for b in blocks]
        total = sum(ns)
        if not total:
            return []
        stored = self.dims() or len(q)
        d = min(dims or stored, stored, len(q))
        # The first pass scores all stored numbers; asked with fewer, the
        # query is cut and the shortlist goes deeper before the exact order
        q1 = np.zeros(m.dims, dtype=np.float32)
        q1[:d] = unit(q, d)
        parts = []
        for b, n in zip(blocks, ns):
            if not n:
                continue
            sc = _scores(b, n, q1)
            if files is not None:
                sc = np.where(np.isin(b["files"][:n], files), sc, -np.inf)
            parts.append((b, sc))
        if not parts:
            return []
        s = np.concatenate([sc for _, sc in parts])
        live = int(np.isfinite(s).sum())
        if not live:
            return []
        dead_share = min(0.5, m.dead / max(total, 1))
        base = SHORTLIST_FEWER if d < m.dims else max(SHORTLIST, 10 * k)
        want = int(base / (1.0 - dead_share)) + (1 if exclude else 0)
        take = min(live, want)
        part = np.argpartition(-s, take - 1)[:take]
        part = part[np.isfinite(s[part])]
        starts = np.cumsum([0] + [len(sc) for _, sc in parts])
        bi = np.searchsorted(starts, part, side="right") - 1
        rows = part - starts[bi]
        cand = {}
        for j, r in zip(bi, rows):
            b = parts[j][0]
            cand.setdefault(int(b["ids"][r]), int(b["moments"][r]))
        vecs = self.vectors(list(cand))
        qd = unit(q, d)
        out = []
        for iid, mid in cand.items():
            v = vecs.get(iid)
            if v is None or iid == exclude:
                continue
            vd = unit(v, d)
            out.append((iid, mid, float(vd @ qd), vd))
        out.sort(key=lambda x: (-x[2], x[0]))
        return out[:k]


def _kinds(raw) -> list[str]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(x) for x in v] if isinstance(v, list) else []


def _like_prefix(p: str) -> str:
    return p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


_stores: dict[str, MediaStore] = {}
_stores_lock = threading.Lock()


def store_path() -> Path:
    from whisper_tm.config import config

    p = config.path_setting("media_search", "path")
    if p:
        return p
    return Path(config.db_path).with_name("media.db")


def get_store() -> MediaStore:
    """The store of the configured path — one per path, kept."""
    p = str(store_path())
    with _stores_lock:
        if p not in _stores:
            _stores[p] = MediaStore(p)
        return _stores[p]

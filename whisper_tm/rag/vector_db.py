"""Vector storage in SQLite — brute force on CPU instead of vector index.

**Why no Postgres/pgvector.** Measured on a few hundred thousand chunks × 2560
dimensions, same machine, same data:

===================  =====================  ==================
                     Postgres + pgvector    SQLite + numpy
===================  =====================  ==================
Write                          301.2 s               41.2 s
Build HNSW index               481.7 s             n/a
Space                          4.79 GB              1.87 GB
— thereof index                2.77 GB                 —
Search                           2.8 ms             81.3 ms
Load matrix (once)                  —               3.53 s
===================  =====================  ==================

81 instead of 2.8 milliseconds sounds like a lot and it's not: For a search
form, both are "instant". In return, there's no server service, no backup, no
synchronization problem, and no eight-minute index build on every rebuild — and
the file is 2.6 times smaller. The index was larger than the data it indexed.

All numbers on this line come from the ACTUAL code path. This is not a side note:
An earlier benchmark measured the same search on an already converted matrix and
reported 82 ms — while the code in truth converted anew on every request and took
2,461 ms. A benchmark that doesn't take the same path as production measures
something different.

**Why no index.** With a few hundred thousand vectors, a single matrix product finishes faster
than an approximation algorithm needs to select its candidates. Exactly the same
calculation the voice card does (``core/speaker_knn``), there with about
50,000 vectors.

**Why CPU and not GPU.** Moving the matrix to the card costs 110.7 ms — more
than the entire CPU search. It would only be worth it if it stayed there
permanently, and that would be 1.73 GB permanently gone on a card that Whisper
and pyannote work on. VRAM is the scarce resource; as long as the CPU search
is fast enough, the GPU buys nothing here.

**Where the boundary lies.** The cost grows linearly with inventory: a few hundred
thousand chunks are about 80 ms, a million would be around 240 ms. If it gets more, the
dimension can be shortened — Qwen3 is Matryoshka-trained, and shortened to 1024
(measured on 200 real chunks) Spearman 0.965 and 87% Top-10 overlap remain with
2.5× less memory.

**Half precision in memory too.** At over half a million chunks the float32
matrix held 6.65 GB and took 31 s to load. It is kept as float16 now — the
file's own format, 3.33 GB. numpy has no fast float16 path (2.6 s per
question at that size, even block-wise through float32), torch's CPU half
GEMV does (32 ms at 200k rows against numpy float32's 53 ms — it reads half
the bytes, accumulates in float32 and rounds its output to float16, ≤2.5e-4).
So the whole matrix is ranked approximately and the best ``k`` plus a margin
are scored again exactly in float32: the returned values and their order are
exact, and the top 1,000 agreed with the float32 ranking 99.8 % before the
rescoring. Reading went from 31 s to ~5.5 s as well (count(*) without a
WHERE — 0.01 s against 4.4 s — and every blob copied straight into place
instead of joined, converted and copied). The server loads it in the
background after its start (__main__) and after an index run.

**Full text** is handled by SQLite itself via FTS5 — the same technique that the
app's existing search runs on. Fusing both results (Reciprocal Rank Fusion)
happens in Python; in SQL it would just be more awkward.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Number format for vectors — in the file and in the kept matrix. float16
#: halves space and load time; the precision loss with normalized embeddings
#: is smaller than the distance between two adjacent matches.
DTYPE = np.float16

#: What similarities come back as (see :func:`similarities`).
COMPUTE_DTYPE = np.float32

#: Rows per step where numpy ranks without torch.
SCORE_BLOCK = 32768

#: How many rows per INSERT round. 20,000 was measured as the point where
#: more yields nothing more (under 20 s for a few hundred thousand rows).
WRITE_BLOCK = 20_000


class RagConfigError(RuntimeError):
    """Misconfiguration (dimension unknown, schema doesn't match …)."""


def _nrm(A: np.ndarray) -> np.ndarray:
    """Normalize to length 1 row by row. Cosine becomes dot product."""
    n = np.linalg.norm(A, axis=-1, keepdims=True)
    return A / np.maximum(n, 1e-12)


_torch_mod = None
_torch_lock = threading.Lock()


def _torch():
    """torch, when it imports (it does wherever the app runs) — else None."""
    global _torch_mod
    with _torch_lock:
        if _torch_mod is None:
            try:
                import torch

                _torch_mod = torch
            except Exception as e:  # noqa: BLE001 — numpy still ranks, slower
                logger.info(
                    "Vector search without torch (%s) — numpy ranks the "
                    "float16 matrix block by block",
                    e,
                )
                _torch_mod = False
        return _torch_mod or None


def similarities(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """``X @ Q`` as float32 — ``X`` the kept matrix (n × d), ``Q`` one
    question (d,) or several (d × m).

    For a float16 ``X`` through torch's half GEMV: float32 accumulation, the
    result rounded to float16 (≤2.5e-4 off). Good for ranking; whoever
    returns values scores the chosen rows again exactly (:meth:`VectorDB.
    search_vector`). Without torch (or when it fails) numpy, block by block
    through float32 — exact, ~50× slower.
    """
    Q = np.asarray(Q, dtype=np.float32)
    if X.dtype == np.float16 and len(X):
        torch = _torch()
        if torch is not None:
            try:
                xt = torch.from_numpy(X)
                qt = torch.from_numpy(np.ascontiguousarray(Q.astype(np.float16)))
                return (xt @ qt).float().numpy()
            except Exception as e:  # noqa: BLE001 — numpy still answers
                logger.warning("Vector search: torch failed (%s) — numpy instead", e)
        out = np.empty((len(X),) + Q.shape[1:], dtype=COMPUTE_DTYPE)
        for i in range(0, len(X), SCORE_BLOCK):
            out[i : i + SCORE_BLOCK] = X[i : i + SCORE_BLOCK].astype(np.float32) @ Q
        return out
    return (np.asarray(X, dtype=np.float32) @ Q).astype(COMPUTE_DTYPE, copy=False)


class VectorDB:
    """Chunks, their vectors, and their full text — all in ONE SQLite file.

    The matrix is loaded on first access and kept in the process.
    It's discarded via ``generation``, which counts every change —
    the same procedure as for the voice inventory (``core/speaker_inventory``).
    A timestamp would be less precise: two changes in the same second
    would look like one.
    """

    TABLE = "chunks"

    def __init__(self, path: str | Path, *, dimensions: int = 0, fts_language: str = "english"):
        self.path = str(path)
        self.dimensions = int(dimensions or 0)
        # FTS5 has no German stemmer; the language is carried along
        # because it belongs in the meta, but it doesn't act like
        # Postgres' tsvector.
        self.fts_language = fts_language
        self._lock = threading.RLock()
        self._matrix: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._matrix_gen = -1
        # Serializes loading the matrix — see `matrix`.
        self._load_lock = threading.Lock()

    # ── Connections ──────────────────────────────────────────────────────
    def connect(self) -> sqlite3.Connection:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """A connection that **commits** on clean exit.

        Without the commit, write operations are silently lost: ``sqlite3``
        opens an implicit transaction on INSERT, and a ``close()`` without
        commit discards it silently — no error, no indication, the row is
        just gone.

        Found after a complete, "error-free" run over
        thousands of transcripts: In ``rag_meta`` only the values from ``ensure_schema``
        (which commits itself) were present, but neither ``embedding_model``
        nor ``index_fingerprint``. The fingerprint is the more dangerous loss
        — without it, a changed chunk configuration is never seen as changed,
        and the indexer doesn't re-embed even though it should.

        On exception, the transaction is rolled back: a half-written state
        is worse than none. A failing commit (disk full, I/O error) raises:
        swallowing it was the very silent loss described above.
        """
        c = self.connect()
        try:
            yield c
        except BaseException:
            with suppress(sqlite3.Error):
                c.rollback()
            raise
        else:
            c.commit()
        finally:
            c.close()

    @staticmethod
    @contextmanager
    def transaction(conn) -> Iterator[None]:
        """All or nothing — per transcript.

        psycopg brings ``conn.transaction()`` with it, ``sqlite3`` doesn't. Instead
        of teaching the indexer which database underlies it, the bracket is here:
        It says "this belongs together", and how that's achieved is the storage's
        business.

        ``BEGIN IMMEDIATE`` not the default deferred: Otherwise SQLite only
        acquires the write lock at the first INSERT, and two simultaneous runs
        can run each other into a ``database is locked`` after both have already
        read.
        """
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    # ── Schema ───────────────────────────────────────────────────────────
    def ensure_schema(self, conn=None, *, dimensions: int | None = None) -> dict:
        dim = int(dimensions if dimensions is not None else self.dimensions or 0)
        own = conn is None
        c = conn or self.connect()
        try:
            present = self.current_schema_info(c)
            if present.get("exists"):
                self.dimensions = present["dimensions"] or dim
                return present
            if dim <= 0:
                raise RagConfigError(
                    "Embedding dimension unknown — schema cannot be "
                    "created. Run a test embedding first."
                )
            c.executescript(f"""
                CREATE TABLE IF NOT EXISTS rag_meta (
                    key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id             INTEGER PRIMARY KEY,
                    transcript_key TEXT NOT NULL,
                    chunk_index    INTEGER NOT NULL,
                    content        TEXT NOT NULL,
                    speaker        TEXT,
                    start_sec      REAL,
                    end_sec        REAL,
                    seg_start      INTEGER,
                    seg_end        INTEGER,
                    language       TEXT,
                    model          TEXT,
                    media_path     TEXT,
                    metadata       TEXT NOT NULL DEFAULT '{{}}',
                    vector         BLOB,
                    created_at     REAL NOT NULL DEFAULT (unixepoch()),
                    UNIQUE (transcript_key, chunk_index));
                CREATE INDEX IF NOT EXISTS idx_chunks_key
                    ON {self.TABLE}(transcript_key);
                CREATE INDEX IF NOT EXISTS idx_chunks_speaker
                    ON {self.TABLE}(speaker);
                CREATE TABLE IF NOT EXISTS indexed_transcripts (
                    transcript_key TEXT PRIMARY KEY,
                    seg_count      INTEGER,
                    updated_at     REAL,
                    chunk_count    INTEGER,
                    indexed_at     REAL NOT NULL DEFAULT (unixepoch()));
                /* Full text. `content=` turns this into a pure index without
                   a second copy of the text — otherwise every chunk would
                   sit twice in the file. The triggers keep it in sync. */
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content, content='{self.TABLE}', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2');
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON {self.TABLE}
                    BEGIN INSERT INTO chunks_fts(rowid, content)
                          VALUES (new.id, new.content); END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON {self.TABLE}
                    BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, content)
                          VALUES ('delete', old.id, old.content); END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON {self.TABLE}
                    BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, content)
                          VALUES ('delete', old.id, old.content);
                          INSERT INTO chunks_fts(rowid, content)
                          VALUES (new.id, new.content); END;
            """)
            self._set_meta(c, "dimensions", dim)
            self._set_meta(c, "fts_language", self.fts_language)
            self._set_meta(c, "generation", 1)
            c.commit()
            self.dimensions = dim
            self._discard_matrix()
            return self.current_schema_info(c)
        finally:
            if own:
                c.close()

    def current_schema_info(self, conn=None) -> dict:
        def _q(c):
            da = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (self.TABLE,)
            ).fetchone()
            if not da:
                return {"exists": False}
            meta = self.get_meta(c)
            return {
                "exists": True,
                "vector_type": str(np.dtype(DTYPE)),
                "dimensions": int(meta.get("dimensions") or 0),
            }

        if conn is not None:
            return _q(conn)
        with self.connection() as c:
            return _q(c)

    def drop_all(self, conn=None):
        own = conn is None
        c = conn or self.connect()
        try:
            c.executescript(f"""
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS {self.TABLE};
                DROP TABLE IF EXISTS indexed_transcripts;
                DROP TABLE IF EXISTS rag_meta;
            """)
            c.commit()
        finally:
            if own:
                c.close()
        self._discard_matrix()

    # ── Meta ─────────────────────────────────────────────────────────────
    @staticmethod
    def _set_meta(c, key: str, value):
        c.execute(
            "INSERT INTO rag_meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def get_meta(self, conn=None) -> dict:
        def _q(c):
            try:
                return {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM rag_meta")}
            except sqlite3.OperationalError:
                return {}

        if conn is not None:
            return _q(conn)
        with self.connection() as c:
            return _q(c)

    def _generation(self, c) -> int:
        try:
            return int(self.get_meta(c).get("generation") or 0)
        except (TypeError, ValueError):
            return 0

    def _bump_generation(self, c):
        self._set_meta(c, "generation", self._generation(c) + 1)

    # ── Writing ──────────────────────────────────────────────────────────
    def upsert_chunks(self, conn, rows: list[dict]) -> int:
        if not rows:
            return 0
        dim = self.dimensions or len(rows[0]["embedding"])
        data = []
        for r in rows:
            v = np.asarray(r["embedding"], dtype=np.float32)
            if v.shape[-1] != dim:
                raise RagConfigError(f"Vector has {v.shape[-1]} dimensions, schema expects {dim}")
            # Non-finite values would poison the whole matrix: a single
            # NaN turns EVERY dot product into NaN, and the search then
            # silently returns nonsense instead of failing.
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            data.append(
                (
                    r["transcript_key"],
                    int(r["chunk_index"]),
                    r["content"],
                    r.get("speaker"),
                    r.get("start_sec"),
                    r.get("end_sec"),
                    r.get("seg_start"),
                    r.get("seg_end"),
                    r.get("language"),
                    r.get("model"),
                    r.get("media_path"),
                    json.dumps(r.get("metadata") or {}, ensure_ascii=False),
                    _nrm(v).astype(DTYPE).tobytes(),
                )
            )
        conn.executemany(
            f"""INSERT INTO {self.TABLE}
                (transcript_key, chunk_index, content, speaker, start_sec,
                 end_sec, seg_start, seg_end, language, model, media_path,
                 metadata, vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(transcript_key, chunk_index) DO UPDATE SET
                  content=excluded.content, speaker=excluded.speaker,
                  start_sec=excluded.start_sec, end_sec=excluded.end_sec,
                  seg_start=excluded.seg_start, seg_end=excluded.seg_end,
                  language=excluded.language, model=excluded.model,
                  media_path=excluded.media_path, metadata=excluded.metadata,
                  vector=excluded.vector""",
            data,
        )
        self._bump_generation(conn)
        return len(data)

    def delete_transcript(self, conn, transcript_key: str):
        conn.execute(f"DELETE FROM {self.TABLE} WHERE transcript_key=?", (transcript_key,))
        conn.execute("DELETE FROM indexed_transcripts WHERE transcript_key=?", (transcript_key,))
        self._bump_generation(conn)

    def prune_except(self, conn, keep_keys) -> int:
        """Delete every transcript whose key is NOT in ``keep_keys``.

        For transcripts that left the source (deleted, no longer complete,
        a key from an older formula): their chunks would otherwise stay
        searchable forever. Returns the number of transcripts removed.
        Through a TEMP table — hundreds of thousands of keys don't fit an ``IN (?, …)``.
        """
        conn.execute("DROP TABLE IF EXISTS temp._rag_keep")
        conn.execute("CREATE TEMP TABLE _rag_keep (k TEXT PRIMARY KEY)")
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO _rag_keep VALUES (?)", ((k,) for k in keep_keys)
            )
            gone = [
                r[0]
                for r in conn.execute(
                    f"SELECT transcript_key FROM indexed_transcripts "
                    f"WHERE transcript_key NOT IN (SELECT k FROM _rag_keep) "
                    f"UNION SELECT DISTINCT transcript_key FROM {self.TABLE} "
                    f"WHERE transcript_key NOT IN (SELECT k FROM _rag_keep)"
                )
            ]
            # The temp inserts opened an implicit transaction; the delete
            # below needs its own (BEGIN IMMEDIATE).
            conn.commit()
            if gone:
                with self.transaction(conn):
                    for k in gone:
                        self.delete_transcript(conn, k)
            return len(gone)
        finally:
            conn.execute("DROP TABLE IF EXISTS temp._rag_keep")

    def mark_indexed(
        self, conn, transcript_key: str, *, seg_count: int, updated_at: float, chunk_count: int
    ):
        conn.execute(
            """INSERT INTO indexed_transcripts
               (transcript_key, seg_count, updated_at, chunk_count, indexed_at)
               VALUES (?,?,?,?,unixepoch())
               ON CONFLICT(transcript_key) DO UPDATE SET
                 seg_count=excluded.seg_count, updated_at=excluded.updated_at,
                 chunk_count=excluded.chunk_count, indexed_at=unixepoch()""",
            (transcript_key, seg_count, updated_at, chunk_count),
        )

    def get_indexed_map(self, conn=None) -> dict[str, dict]:
        def _q(c):
            try:
                rows = c.execute(
                    "SELECT transcript_key, seg_count, updated_at,"
                    " chunk_count FROM indexed_transcripts"
                ).fetchall()
            except sqlite3.OperationalError:
                # Table missing (never indexed yet) → empty map. Do NOT
                # swallow other errors, or the indexer will re-embed everything.
                return {}
            return {
                r["transcript_key"]: {
                    "seg_count": r["seg_count"],
                    "updated_at": r["updated_at"],
                    "chunk_count": r["chunk_count"],
                }
                for r in rows
            }

        if conn is not None:
            return _q(conn)
        with self.connection() as c:
            return _q(c)

    # ── The matrix ───────────────────────────────────────────────────────
    def _discard_matrix(self):
        with self._lock:
            self._matrix = None
            self._ids = None
            self._matrix_gen = -1

    def matrix(self, conn=None) -> tuple[np.ndarray, np.ndarray]:
        """``(X, ids)`` — all vectors as a float16 matrix, loaded once.

        Measured: over half a million × 2560 from SQLite in **5.5 s** (from the page
        cache). That's the cost on first access after a restart or a
        change; every further search no longer pays it — and the server
        pays it in the background right after its start (:meth:`preload`).

        Two rules keep a reload from eating the machine's memory:

        * **One load at a time.** Two searches right after a change used to
          load the matrix twice side by side (3.5 GB float32 each).
        * **Read in blocks into a preallocated array.** ``fetchall()`` +
          ``b"".join`` + ``astype`` held the inventory three times over
          (bytes objects, the joined buffer, the float32 copy) — next to
          the matrix still in use, about 10 GB at a few hundred thousand chunks. Each blob
          now goes straight to its row.

        A changed inventory always reloads (``generation``): a held matrix
        that answers for a newer inventory would be silently stale results.
        """
        own = conn is None
        c = conn or self.connect()
        try:
            gen = self._generation(c)
            if (hit := self._cached(gen)) is not None:
                return hit
            with self._load_lock:
                # Whoever waited here usually finds the matrix the thread
                # before it just loaded.
                if (hit := self._cached(gen)) is not None:
                    return hit
                t0 = time.time()
                X, ids = self._read_matrix(c)
                if len(ids):
                    logger.info(
                        "Vector matrix loaded: %d × %d in %.2f s (%.2f GB)",
                        X.shape[0],
                        X.shape[1],
                        time.time() - t0,
                        X.nbytes / 1e9,
                    )
                with self._lock:
                    self._matrix, self._ids, self._matrix_gen = X, ids, gen
                return X, ids
        finally:
            if own:
                c.close()

    def preload(self) -> threading.Thread | None:
        """Load the matrix in a background thread (low priority) so that the
        first search doesn't wait for it — after the server's start and
        after an index run. ``None`` when there is no index to load."""
        if not Path(self.path).is_file():
            return None

        def run():
            try:
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
            except (AttributeError, OSError):
                pass
            try:
                with self.connection() as c:
                    if not self.current_schema_info(c).get("exists"):
                        return
                    self.matrix(c)
            except Exception as e:  # noqa: BLE001 — the first search loads it then
                logger.info("Vector matrix not preloaded: %s", e)

        t = threading.Thread(target=run, name="rag-matrix-preload", daemon=True)
        t.start()
        return t

    def _cached(self, gen: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The held matrix if it may answer for generation ``gen``."""
        with self._lock:
            if self._matrix is None:
                return None
            if self._matrix_gen == gen:
                return self._matrix, self._ids
            return None

    def _read_matrix(self, c) -> tuple[np.ndarray, np.ndarray]:
        """Read all vectors in blocks into one preallocated float16 array.

        Count and rows come from ONE read snapshot. The count is the plain
        ``count(*)`` — SQLite answers it from the smallest index (0.01 s at
        over half a million rows); with ``WHERE vector IS NOT NULL`` it walked every row
        (4.4 s). Rows without a vector, or with one of another length, are
        skipped and the array cut to what was read.

        The matrix stays float16 (3.33 GB at over half a million chunks instead of 6.65 GB
        as float32): :func:`similarities` ranks it without converting.
        """
        own_txn = not c.in_transaction
        if own_txn:
            c.execute("BEGIN")
        try:
            n = c.execute(f"SELECT count(*) FROM {self.TABLE}").fetchone()[0]
            if not n:
                return (
                    np.zeros((0, self.dimensions or 1), dtype=DTYPE),
                    np.zeros(0, dtype=np.int64),
                )
            cur = c.execute(
                f"SELECT id, vector FROM {self.TABLE} WHERE vector IS NOT NULL ORDER BY id"
            )
            X: np.ndarray | None = None
            ids = np.empty(n, dtype=np.int64)
            buf = None
            # The row width the schema promises; the first vector's only
            # where the schema doesn't say (one broken row must not decide)
            try:
                dim = int(self.get_meta(c).get("dimensions") or 0)
            except (TypeError, ValueError):
                dim = 0
            width = dim * np.dtype(DTYPE).itemsize
            pos = skipped = 0
            while pos < n:
                block = cur.fetchmany(WRITE_BLOCK)
                if not block:
                    break
                if X is None:
                    width = width or len(block[0][1])
                    if width % np.dtype(DTYPE).itemsize:
                        raise RagConfigError(
                            f"A stored vector has {width} bytes — not {np.dtype(DTYPE).name}"
                        )
                    X = np.empty((n, width // np.dtype(DTYPE).itemsize), dtype=DTYPE)
                    buf = memoryview(X).cast("B")
                for cid, blob in block:
                    if pos >= n:
                        break
                    if len(blob) != width:
                        skipped += 1
                        continue
                    buf[pos * width : (pos + 1) * width] = blob
                    ids[pos] = cid
                    pos += 1
            if skipped:
                logger.warning(
                    "Vector matrix: %d stored vectors of another length left out", skipped
                )
            if X is None:
                return (
                    np.zeros((0, self.dimensions or 1), dtype=DTYPE),
                    np.zeros(0, dtype=np.int64),
                )
            if pos < n:
                # Release the view before the array is cut (a copy of the rest)
                buf.release()
                return X[:pos].copy(), ids[:pos].copy()
            buf.release()
            return X, ids
        finally:
            if own_txn:
                c.rollback()

    # ── Search ───────────────────────────────────────────────────────────
    _FIELDS = (
        "id",
        "transcript_key",
        "chunk_index",
        "content",
        "speaker",
        "start_sec",
        "end_sec",
        "seg_start",
        "seg_end",
        "language",
        "model",
        "media_path",
    )

    def _lines(self, c, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        slot = ",".join("?" * len(ids))
        rows = c.execute(
            f"SELECT {', '.join(self._FIELDS)} FROM {self.TABLE} WHERE id IN ({slot})", ids
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def search_vector(
        self, qvec, *, k: int = 20, conn=None, min_value: float = 0.0, among=None
    ) -> list[dict]:
        """The ``k`` most similar chunks. A matrix product, nothing else.

        ``min_value`` isn't decorative: without it, EVERY query returns
        hits, even if the inventory contains nothing relevant — then the
        least-bad chunk comes back, and on screen that looks like an
        answer. Measured against real queries, a good hit sat at 0.64,
        the best miss at 0.49.

        ``among``: only these chunk ids compete (the passages one person
        speaks in — rag/retrieve.candidates); ids without a vector are left
        out.
        """
        own = conn is None
        c = conn or self.connect()
        try:
            X, ids = self.matrix(c)
            if not len(X):
                return []
            q = _nrm(np.asarray(qvec, dtype=np.float32).ravel())
            if q.shape[0] != X.shape[1]:
                raise RagConfigError(
                    f"Query has {q.shape[0]} dimensions, inventory has {X.shape[1]}"
                )
            rows = None
            if among is not None:
                want = np.unique(np.fromiter((int(i) for i in among), dtype=np.int64))
                pos = np.searchsorted(ids, want)
                ok = pos < len(ids)
                ok[ok] = ids[pos[ok]] == want[ok]
                rows = pos[ok]
                if not len(rows):
                    return []
            n = len(X) if rows is None else len(rows)
            k = min(int(k), n)
            if k <= 0:
                return []
            if rows is None:
                approx = similarities(X, q)
            else:
                approx = np.concatenate(
                    [
                        similarities(X[rows[i : i + SCORE_BLOCK]], q)
                        for i in range(0, n, SCORE_BLOCK)
                    ]
                )
            # The ranking is approximate (similarities): the best k plus a
            # margin are scored again exactly, and those values decide.
            take = min(n, k + max(64, k // 8))
            # argpartition gets the best without sorting everything.
            chosen = np.argpartition(-approx, take - 1)[:take]
            if rows is not None:
                chosen = rows[chosen]
            exact = X[chosen].astype(np.float32) @ q
            # Ties (copies of one recording have the same vector) by id
            order = np.lexsort((ids[chosen], -exact))[:k]
            hits = [(int(ids[chosen[j]]), float(exact[j])) for j in order if exact[j] >= min_value]
            lines = self._lines(c, [t[0] for t in hits])
            off = []
            for cid, value in hits:
                r = lines.get(cid)
                if r:
                    r["score"] = value
                    r["match"] = "vector"
                    off.append(r)
            return off
        finally:
            if own:
                c.close()

    def search_fts(self, query: str, *, k: int = 20, conn=None) -> list[dict]:
        """Full text via FTS5 — the same technique as the existing search."""
        own = conn is None
        c = conn or self.connect()
        try:
            # Natural-language queries contain characters that FTS5 reads
            # as syntax ("what does that cost?"). Quoting each word
            # individually turns it into a valid OR query instead of
            # failing with a syntax error.
            words = [
                w
                for w in "".join(
                    ch if ch.isalnum() or ch.isspace() else " " for ch in (query or "")
                ).split()
                if len(w) > 1
            ]
            if not words:
                return []
            expression = " OR ".join(f'"{w}"' for w in words)
            try:
                rows = c.execute(
                    f"""SELECT {", ".join("c." + f for f in self._FIELDS)},
                               bm25(chunks_fts) AS bm
                        FROM chunks_fts JOIN {self.TABLE} c
                          ON c.id = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        ORDER BY bm LIMIT ?""",
                    (expression, int(k)),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("FTS5 query rejected (%s): %s", expression, e)
                return []
            off = []
            for r in rows:
                d = dict(r)
                # bm25 is negative, smaller is better. For display this is
                # turned into a value that points upward like the others.
                d["score"] = -float(d.pop("bm") or 0.0)
                d["match"] = "fts"
                off.append(d)
            return off
        finally:
            if own:
                c.close()

    def search_match(self, expression: str, *, k: int = 20, conn=None) -> list[dict]:
        """Full text for a prepared FTS5 expression (rag/retrieve.py builds
        it, every token quoted) — best bm25 first, at most ``k``. A rejected
        expression is logged and finds nothing."""
        if not (expression or "").strip() or k <= 0:
            return []
        own = conn is None
        c = conn or self.connect()
        try:
            try:
                rows = c.execute(
                    f"""SELECT {", ".join("c." + f for f in self._FIELDS)},
                               bm25(chunks_fts) AS bm
                        FROM chunks_fts JOIN {self.TABLE} c
                          ON c.id = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        ORDER BY bm LIMIT ?""",
                    (expression, int(k)),
                ).fetchall()
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    logger.debug("Full text: no index yet (%s)", e)  # nothing built
                else:
                    logger.warning("FTS5 expression rejected (%s): %s", expression[:300], e)
                return []
            off = []
            for r in rows:
                d = dict(r)
                d["score"] = -float(d.pop("bm") or 0.0)
                d["match"] = "fts"
                off.append(d)
            return off
        finally:
            if own:
                c.close()

    def rows_by_id(self, ids, conn=None) -> dict[int, dict]:
        """The chunks with these ids (without their vectors) — in blocks, so
        a long list stays below SQLite's placeholder limit."""
        ids = [int(i) for i in dict.fromkeys(ids)]
        own = conn is None
        c = conn or self.connect()
        try:
            out: dict[int, dict] = {}
            for i in range(0, len(ids), 900):
                out.update(self._lines(c, ids[i : i + 900]))
            return out
        finally:
            if own:
                c.close()

    def vectors_for(self, ids, conn=None) -> np.ndarray:
        """The stored (unit) vectors of these chunk ids from the held matrix,
        one row each in their order; NaN for an id without a vector."""
        ids = np.asarray(list(ids), dtype=np.int64)
        X, all_ids = self.matrix(conn)
        out = np.full((len(ids), X.shape[1] if len(X) else 1), np.nan, dtype=np.float32)
        if not len(X) or not len(ids):
            return out
        pos = np.searchsorted(all_ids, ids)
        pos_c = np.minimum(pos, len(all_ids) - 1)
        found = all_ids[pos_c] == ids
        out[found] = X[pos_c[found]].astype(np.float32)
        return out

    def search_hybrid(
        self,
        qvec,
        query: str,
        *,
        k: int = 20,
        limit: int = 20,
        rrf_k: int = 60,
        conn=None,
        min_value: float = 0.0,
    ) -> list[dict]:
        """Vector + full text, fused via Reciprocal Rank Fusion.

        ``score = 1/(rrf_k + vrank) + 1/(rrf_k + frank)`` — in Python,
        because both lists are already in memory anyway, and SQL would
        only turn this into a more cumbersome formulation of the same
        calculation.
        """
        own = conn is None
        c = conn or self.connect()
        try:
            vec = self.search_vector(qvec, k=k, conn=c, min_value=min_value)
            txt = self.search_fts(query, k=k, conn=c)
            together: dict[int, dict] = {}
            for rank, r in enumerate(vec, start=1):
                d = together.setdefault(r["id"], dict(r))
                d["vrank"], d["vscore"] = rank, r["score"]
            for rank, r in enumerate(txt, start=1):
                d = together.setdefault(r["id"], dict(r))
                d["frank"], d["fscore"] = rank, r["score"]
            off = []
            for d in together.values():
                v, f = d.get("vrank"), d.get("frank")
                d["score"] = (1.0 / (rrf_k + v) if v else 0.0) + (1.0 / (rrf_k + f) if f else 0.0)
                d["match"] = "hybrid" if (v and f) else ("vector" if v else "fts")
                off.append(d)
            off.sort(key=lambda d: d["score"], reverse=True)
            return off[:limit]
        finally:
            if own:
                c.close()

    # ── Info ─────────────────────────────────────────────────────────────
    def stats(self, conn=None) -> dict:
        def _q(c):
            info = self.current_schema_info(c)
            if not info.get("exists"):
                return {"exists": False, "chunks": 0, "transcripts": 0}
            r = c.execute(
                f"SELECT count(*) AS n, count(DISTINCT transcript_key) AS t FROM {self.TABLE}"
            ).fetchone()
            size = 0
            try:
                size = Path(self.path).stat().st_size
            except OSError:
                pass
            meta = self.get_meta(c)
            return {
                "exists": True,
                "chunks": r["n"],
                "transcripts": r["t"],
                "size_bytes": size,
                "vector_type": info["vector_type"],
                "dimensions": info["dimensions"],
                "embedding_model": meta.get("embedding_model", ""),
            }

        if conn is not None:
            return _q(conn)
        with self.connection() as c:
            return _q(c)

    def ping(self) -> dict:
        """Reachability — never raises."""
        try:
            with self.connection() as c:
                v = c.execute("SELECT sqlite_version() AS v").fetchone()["v"]
                return {
                    "reachable": True,
                    "server_version": f"SQLite {v}",
                    "schema": self.current_schema_info(c),
                    "path": self.path,
                    "detail": "ok",
                }
        except Exception as e:  # noqa
            return {"reachable": False, "detail": str(e)[:200]}

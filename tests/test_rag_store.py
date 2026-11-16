"""
RAG vector store: vanished transcripts leave the index, commit failures
are not swallowed, and loading the matrix neither duplicates nor triples it
in memory.
"""

import hashlib
import math
import sqlite3
import threading

import numpy as np
import pytest

from whisper_tm.database import Database
from whisper_tm.rag.indexer import RagIndexer
from whisper_tm.rag.vector_db import VectorDB

_DIM = 32


def _embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for tok in (t or "").lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _tx(sentences):
    return {
        "complete": True,
        "language": "en",
        "segments": [
            {"speaker": "S0", "start": i * 3.0, "end": i * 3.0 + 3, "text": s}
            for i, s in enumerate(sentences)
        ],
    }


@pytest.fixture
def stack(tmp_path):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    db.upsert_transcript(
        "pasta.mp4",
        _tx(["tomatoes basil pasta noodles"]),
        media_dir=str(tmp_path),
        abs_path=str(tmp_path / "pasta.mp4"),
    )
    db.upsert_transcript(
        "ev.mp4",
        _tx(["battery motor kilowatts range"]),
        media_dir=str(tmp_path),
        abs_path=str(tmp_path / "ev.mp4"),
    )
    rag = VectorDB(str(tmp_path / "rag.db"))
    idx = RagIndexer(
        rag, db, _embed, chunk_max_chars=500, chunk_min_chars=1, embedding_model="fake"
    )
    yield db, rag, idx
    db.close_all()


class TestVanishedTranscripts:
    def test_deleted_transcript_leaves_the_index(self, stack, tmp_path):
        db, rag, idx = stack
        idx.run()
        assert rag.stats()["transcripts"] == 2
        db.delete_transcript(str(tmp_path / "ev.mp4"), media_dir=str(tmp_path))
        summary = idx.run()
        assert summary["removed"] == 1
        assert rag.stats()["transcripts"] == 1
        assert rag.search_fts("battery kilowatts") == []
        assert set(rag.get_indexed_map()) == {str(tmp_path / "pasta.mp4")}

    def test_empty_source_prunes_nothing(self, stack, monkeypatch):
        """No candidates at all is a wrong/unreadable DB, not an empty library."""
        db, rag, idx = stack
        idx.run()
        monkeypatch.setattr(db, "get_rag_candidates", lambda: [])
        idx.run()
        assert rag.stats()["transcripts"] == 2


class TestCommitNotSwallowed:
    def test_failed_commit_raises(self, tmp_path):
        class _Conn:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def commit(self):
                raise sqlite3.OperationalError("disk I/O error")

        class _DB(VectorDB):
            def connect(self):
                return _Conn(super().connect())

        rag = _DB(str(tmp_path / "rag.db"))
        with pytest.raises(sqlite3.OperationalError), rag.connection() as c:
            c.execute("CREATE TABLE IF NOT EXISTS t (x)")


class TestMatrixLoading:
    def test_matrix_matches_stored_vectors(self, stack):
        db, rag, idx = stack
        idx.run()
        X, ids = rag.matrix()
        # Kept as stored: float16 (half the memory of float32)
        assert X.dtype == np.float16 and X.shape == (len(ids), _DIM)
        with rag.connection() as c:
            rows = c.execute("SELECT id, vector FROM chunks ORDER BY id").fetchall()
        assert list(ids) == [r[0] for r in rows]
        for i, r in enumerate(rows):
            np.testing.assert_array_equal(X[i], np.frombuffer(r[1], dtype=np.float16))

    def test_reads_in_blocks_not_fetchall(self, stack, monkeypatch):
        """fetchall() + join held the inventory three times over."""
        import whisper_tm.rag.vector_db as vdb

        db, rag, idx = stack
        idx.run()
        monkeypatch.setattr(vdb, "WRITE_BLOCK", 1)
        X, ids = rag._read_matrix(rag.connect())
        assert len(ids) == rag.stats()["chunks"] and X.shape[0] == len(ids)

    def test_concurrent_searches_load_once(self, stack, monkeypatch):
        db, rag, idx = stack
        idx.run()
        loads = []
        real = rag._read_matrix

        def counting(c):
            loads.append(1)
            return real(c)

        monkeypatch.setattr(rag, "_read_matrix", counting)
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            rag.matrix()

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(loads) == 1


def _random_store(tmp_path, n=3000, dim=48, seed=1):
    """A store of ``n`` random unit vectors (one transcript per 50 rows)."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, dim)).astype(np.float32)
    rag = VectorDB(str(tmp_path / "rand.db"))
    with rag.connection() as c:
        rag.ensure_schema(c, dimensions=dim)
        rag.upsert_chunks(
            c,
            [
                {
                    "transcript_key": f"t{i // 50}",
                    "chunk_index": i % 50,
                    "content": f"text {i}",
                    "embedding": V[i],
                }
                for i in range(n)
            ],
        )
    return rag, V / np.linalg.norm(V, axis=1, keepdims=True)


class TestHalfPrecision:
    """The matrix is ranked in half precision and the chosen rows are scored
    again exactly: the answer is the float32 answer."""

    def test_the_best_k_and_their_values_are_exact(self, tmp_path):
        rag, V = _random_store(tmp_path)
        X, ids = rag.matrix()
        rng = np.random.default_rng(7)
        for _ in range(5):
            q = rng.standard_normal(V.shape[1]).astype(np.float32)
            q /= np.linalg.norm(q)
            exact = X.astype(np.float32) @ q
            want = np.argsort(-exact, kind="stable")[:200]
            got = rag.search_vector(q, k=200, min_value=-1.0)
            assert [r["id"] for r in got] == [int(ids[i]) for i in want]
            np.testing.assert_allclose([r["score"] for r in got], exact[want], rtol=0, atol=1e-6)

    def test_min_value_cuts_on_the_exact_value(self, tmp_path):
        rag, V = _random_store(tmp_path, n=500)
        q = V[17]
        got = rag.search_vector(q, k=50, min_value=0.999)
        assert [r["content"] for r in got] == ["text 17"]
        assert got[0]["score"] == pytest.approx(1.0, abs=1e-3)

    def test_torch_and_numpy_rank_alike(self, tmp_path, monkeypatch):
        import whisper_tm.rag.vector_db as vdb

        rag, V = _random_store(tmp_path, n=2000)
        X, _ids = rag.matrix()
        q = V[3] + 0.1
        q = (q / np.linalg.norm(q)).astype(np.float32)
        with_torch = vdb.similarities(X, q)
        assert with_torch.dtype == np.float32 and with_torch.shape == (len(X),)
        monkeypatch.setattr(vdb, "_torch", lambda: None)
        with_numpy = vdb.similarities(X, q)
        np.testing.assert_allclose(with_torch, with_numpy, atol=3e-3)
        np.testing.assert_allclose(with_numpy, X.astype(np.float32) @ q, atol=1e-5)
        Q = np.stack([q, V[9]], axis=1).astype(np.float32)
        assert vdb.similarities(X, Q).shape == (len(X), 2)

    def test_a_failing_torch_falls_back_to_numpy(self, tmp_path, monkeypatch):
        import whisper_tm.rag.vector_db as vdb

        rag, V = _random_store(tmp_path, n=300)
        X, ids = rag.matrix()

        class Broken:
            @staticmethod
            def from_numpy(_a):
                raise RuntimeError("no half GEMV here")

        monkeypatch.setattr(vdb, "_torch", lambda: Broken)
        got = rag.search_vector(V[5], k=3, min_value=-1.0)
        assert got[0]["content"] == "text 5"

    def test_stored_vectors_come_back_as_float32(self, tmp_path):
        rag, V = _random_store(tmp_path, n=200)
        X, ids = rag.matrix()
        out = rag.vectors_for([int(ids[4]), 999999, int(ids[0])])
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out[0], X[4].astype(np.float32))
        assert np.isnan(out[1]).all()
        np.testing.assert_array_equal(out[2], X[0].astype(np.float32))


class TestReading:
    def test_a_vector_of_another_length_is_left_out(self, tmp_path, caplog):
        rag, V = _random_store(tmp_path, n=120, dim=16)
        with rag.connection() as c:
            c.execute(
                "UPDATE chunks SET vector = ? WHERE id = 7",
                (np.zeros(8, dtype=np.float16).tobytes(),),
            )
            c.execute("UPDATE chunks SET vector = NULL WHERE id = 9")
            rag._bump_generation(c)
        X, ids = rag.matrix()
        assert X.shape == (118, 16) and 7 not in ids and 9 not in ids
        assert "another length" in caplog.text

    def test_an_empty_store_gives_an_empty_matrix(self, tmp_path):
        rag = VectorDB(str(tmp_path / "empty.db"))
        with rag.connection() as c:
            rag.ensure_schema(c, dimensions=8)
        X, ids = rag.matrix()
        assert X.shape[0] == 0 and len(ids) == 0
        assert rag.search_vector(np.ones(8), k=5) == []


class TestPreload:
    def test_the_matrix_is_loaded_in_the_background(self, tmp_path, monkeypatch):
        rag, _V = _random_store(tmp_path, n=100)
        loads = []
        real = rag._read_matrix
        monkeypatch.setattr(rag, "_read_matrix", lambda c: loads.append(1) or real(c))
        t = rag.preload()
        assert t is not None and t.name == "rag-matrix-preload"
        t.join(10)
        assert loads == [1]
        rag.matrix()  # the first search finds it loaded
        assert loads == [1]

    def test_no_index_no_preload(self, tmp_path):
        assert VectorDB(str(tmp_path / "missing.db")).preload() is None
        rag = VectorDB(str(tmp_path / "noschema.db"))
        with rag.connection() as c:
            c.execute("CREATE TABLE other (x)")
        t = rag.preload()
        t.join(10)
        assert rag._matrix is None


def test_a_broken_first_vector_does_not_set_the_width(tmp_path):
    rag, _V = _random_store(tmp_path, n=50, dim=16)
    with rag.connection() as c:
        first = c.execute("SELECT min(id) FROM chunks").fetchone()[0]
        c.execute(
            "UPDATE chunks SET vector = ? WHERE id = ?",
            (np.zeros(3, dtype=np.float16).tobytes(), first),
        )
        rag._bump_generation(c)
    X, ids = rag.matrix()
    assert X.shape == (49, 16) and first not in ids

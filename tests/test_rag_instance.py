"""The vector store is KEPT — otherwise its cache is useless.

``VectorDB`` loads the vector matrix on first access and keeps it in
memory; it is discarded via ``generation`` as soon as something in the inventory
changes. But this cache lives on the **instance** — and
``make_rag_db()`` used to build a new one on every call. Every
route, every search, every coverage question started from scratch.

Rough orders of magnitude on a large inventory (a few hundred thousand
chunks, a matrix of a few GB):

    Load matrix                 seconds       ← on EVERY search
    Similarity + Top-k          < 0.1 s
    Embed question (CPU)        ~0.1 s
    same matrix again           milliseconds  ← when instance persists

Most of a RAG search's time was reloading. The oft-invoked question
"GPU or CPU for embedding" decided over only a small fraction.

What is being tested: that it is the same instance, that a changed
configuration still takes effect, and that the cache via
``generation`` still drops when the inventory changes. The last
is the part that would be dangerous with sharing — a shared instance with
stale cache silently delivers outdated results.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.config import config
from whisper_tm.rag import service


@pytest.fixture
def fresh(tmp_path):
    """Own path, own instance — and cleaned up afterwards."""
    old_value = config.get("rag", "path", default="")
    old_dim = config.get("rag", "dimensions", default=0)
    config.set("rag", "path", str(tmp_path / "rag.db"))
    # Without a known dimension, `ensure_schema` doesn't create a schema — the
    # vector column width is otherwise not fixed.
    config.set("rag", "dimensions", 4)
    service.forget_rag_db()
    yield tmp_path
    service.forget_rag_db()
    config.set("rag", "path", old_value)
    config.set("rag", "dimensions", old_dim)


def test_asked_twice_is_the_same_object(fresh):
    assert service.make_rag_db() is service.make_rag_db()


def test_the_matrix_survives_the_second_call(fresh):
    """The actual purpose. Before, the cache was gone after every call."""
    rag = service.make_rag_db()
    with rag.connection() as c:
        rag.ensure_schema(c)
    X, _ids = service.make_rag_db().matrix()
    # A second access via a NEW make_rag_db() call must get the same
    # matrix — identical, not just equal.
    X2, _ = service.make_rag_db().matrix()
    assert X2 is X


def test_a_different_path_gives_a_different_instance(fresh, tmp_path):
    first_one = service.make_rag_db()
    config.set("rag", "path", str(tmp_path / "elsewhere.db"))
    second_one = service.make_rag_db()
    assert second_one is not first_one
    assert second_one.path != first_one.path


def test_forgetting_throws_the_instance_away(fresh):
    """Necessary because the key is the PATH: if the file is rebuilt
    at the same location, the key doesn't notice."""
    first_one = service.make_rag_db()
    service.forget_rag_db()
    assert service.make_rag_db() is not first_one


def test_a_changed_inventory_still_drops_the_cache(fresh):
    """The danger with sharing: silently stale results.

    The protection against it is ``generation`` — it counts every change, and
    the matrix is discarded as soon as it no longer fits. Without this
    proof, keeping the instance would be a data bug instead of a
    speedup.
    """

    def line(key, text, vector):
        return {
            "transcript_key": key,
            "chunk_index": 0,
            "content": text,
            "speaker": "A",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "seg_start": 0,
            "embedding": vector,
        }

    rag = service.make_rag_db()
    with rag.connection() as c:
        rag.ensure_schema(c)
        rag.upsert_chunks(c, [line("file-a", "first text", np.ones(4, dtype=np.float32))])
    X1, ids1 = rag.matrix()
    assert len(ids1) == 1

    with rag.connection() as c:
        rag.upsert_chunks(c, [line("file-b", "second text", np.full(4, 0.5, dtype=np.float32))])

    X2, ids2 = service.make_rag_db().matrix()
    assert len(ids2) == 2, "the cache swallowed the change"
    assert X2 is not X1


def test_the_instance_holds_no_connection(fresh):
    """Sharing is only allowed because no connection state hangs on it.

    ``connect()`` opens its own ``sqlite3`` connection per call. If one
    were held on the instance, it would migrate between the web server's
    threads — exactly the bug that ``check_same_thread`` normally
    prevents.
    """
    rag = service.make_rag_db()
    a, b = rag.connect(), rag.connect()
    try:
        assert a is not b
    finally:
        a.close()
        b.close()
    assert not any(isinstance(v, type(a)) for v in vars(rag).values())


# ── The question goes through the search path, not the index path ────────


def test_the_question_reports_purpose_search(monkeypatch):
    """`make_query_embed_fn()` must pass `purpose='search'` through to server
    startup — otherwise the search starts the model on the GPU again.

    What's checked is the WIRING, because exactly that can get lost
    along the way: each function looks correct on its own, and the effect
    only shows up at the next CUDA OOM.
    """
    from pathlib import Path

    from whisper_tm.core import embed_server

    config.set("embedding", "local", True)
    seen = {}

    class Dummy:
        def ensure(self, *, wait_time=None, purpose="index"):
            seen["purpose"] = purpose
            return "http://127.0.0.1:1/v1"

        def used(self):
            pass

    monkeypatch.setattr(embed_server, "SERVER", Dummy())
    # Without this the test would read the real user config and the
    # gigabyte-sized GGUF file next to it — on a fresh checkout (and on CI)
    # there is none, and the wiring under test would never be reached.
    monkeypatch.setattr(embed_server, "model_file", lambda: Path("qwen3-embedding.gguf"))
    monkeypatch.setattr(service, "embedding_ready", lambda: (True, ""))
    monkeypatch.setattr(
        service.llm_client, "embed_texts", lambda texts, **k: [[0.0] * 4 for _ in texts]
    )

    service.make_query_embed_fn()(["someone is insulted"])
    assert seen["purpose"] == "search"

    service.make_embed_fn()(["a chunk from a transcript"])
    assert seen["purpose"] == "index", "the index run must not go to the CPU"

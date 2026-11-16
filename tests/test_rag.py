"""
Tests for the RAG subsystem (whisper_tm/rag).

ALL run always. This is the actual benefit of switching from
pgvector to SQLite: The integration tests previously needed a reachable
Postgres, were skipped without one — and there never was one. Precisely
because of this, the coverage bug (indexer stores `abs_path`, coverage
looked for `rel_path`) could go unnoticed for months.
"""

import hashlib
import math

import pytest

from whisper_tm.database import Database
from whisper_tm.rag.chunker import chunk_segments
from whisper_tm.rag.indexer import RagIndexer, _close, _transcript_key
from whisper_tm.rag.vector_db import VectorDB


# ══════════════════════════════════════════════════════════════════════════
#  Chunker (pure logic, no DB)
# ══════════════════════════════════════════════════════════════════════════
def _segs(texts, speakers=None):
    out = []
    for i, t in enumerate(texts):
        spk = speakers[i] if speakers else ("A" if i % 2 else "B")
        out.append(
            {
                "seg_index": i,
                "speaker": spk,
                "start_sec": i * 2.0,
                "end_sec": i * 2.0 + 2.0,
                "text": t,
            }
        )
    return out


def test_chunk_empty():
    assert chunk_segments([]) == []


def test_chunk_basic_grouping():
    segs = _segs([f"Sentence number {i} with some text." for i in range(12)])
    chunks = chunk_segments(segs, max_chars=100, min_chars=30, overlap_segments=0)
    assert len(chunks) >= 2
    # chunk_index runs consecutively
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Time bounds monotonic
    assert chunks[0].start_sec == 0.0
    assert chunks[-1].end_sec == segs[-1]["end_sec"]
    # seg indices covered without gaps (without overlap)
    assert chunks[0].seg_start == 0


def test_chunk_overlap_carries_segments():
    segs = _segs([f"A somewhat longer sentence number {i} here." for i in range(9)])
    no_ov = chunk_segments(segs, max_chars=80, min_chars=20, overlap_segments=0)
    ov = chunk_segments(segs, max_chars=80, min_chars=20, overlap_segments=1)
    # With overlap the 2nd chunk starts earlier (overlapping segment)
    assert len(ov) >= 2
    assert ov[1].seg_start <= no_ov[1].seg_start


def test_chunk_single_long_segment_exceeds_max():
    segs = _segs(["Word " * 200])  # one very long segment
    chunks = chunk_segments(segs, max_chars=100, min_chars=20)
    assert len(chunks) == 1
    assert len(chunks[0].content) > 100  # soft limit


def test_chunk_dominant_speaker_and_labels():
    segs = _segs(
        [
            "I talk a lot and at length here in the first segment.",
            "Short.",
            "And I also talk a lot in the third segment here.",
        ],
        speakers=["ALICE", "BOB", "ALICE"],
    )
    chunks = chunk_segments(segs, max_chars=10000, min_chars=0, include_speaker_labels=True)
    assert len(chunks) == 1
    assert chunks[0].speaker == "ALICE"  # dominant speaker (characters)
    assert set(chunks[0].speakers) == {"ALICE", "BOB"}
    assert "[ALICE]" in chunks[0].content  # label inserted


def test_chunk_no_labels():
    segs = _segs(["Hello world.", "Second sentence."], speakers=["A", "B"])
    chunks = chunk_segments(segs, max_chars=10000, min_chars=0, include_speaker_labels=False)
    assert "[" not in chunks[0].content


def test_chunk_flexible_field_names():
    # start/end instead of start_sec/end_sec
    segs = [
        {"speaker": "X", "start": 1.0, "end": 2.0, "text": "Test one."},
        {"speaker": "X", "start": 2.0, "end": 3.0, "text": "Test two."},
    ]
    chunks = chunk_segments(segs, max_chars=10000, min_chars=0)
    assert chunks[0].start_sec == 1.0 and chunks[0].end_sec == 3.0


# ══════════════════════════════════════════════════════════════════════════
#  Pure helper functions
# ══════════════════════════════════════════════════════════════════════════


def test_close_tolerance():
    assert _close(100.0, 100.4)
    assert not _close(100.0, 101.0)


# ══════════════════════════════════════════════════════════════════════════
#  The vector store: schema, indexing run, search
# ══════════════════════════════════════════════════════════════════════════
_DIM = 256


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for tok in (t or "").lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _make_txt(sents):
    segs = [
        {"speaker": "S0" if i % 2 else "S1", "start": i * 3.0, "end": i * 3.0 + 3, "text": s}
        for i, s in enumerate(sents)
    ]
    return {
        "language": "de",
        "model": "turbo",
        "complete": True,
        "speakers": ["S0", "S1"],
        "segments": segs,
        "total_duration_sec": len(segs) * 3,
    }


def _key(db, rel_path: str) -> str:
    """The key under which a transcript is stored in the RAG index.

    Do NOT hard-code the ``rel_path``. That is exactly what these tests
    used to do, and because they were always skipped for lack of
    Postgres, nobody noticed that the indexer had long since switched to
    the identity (canonical ``abs_path``). The real bug behind it —
    coverage kept looking for ``rel_path`` and permanently reported
    "0 embedded" — sat unnoticed in the code for months.

    That's why the test now asks the same function as the indexer. A
    future schema change then won't break the tests falsely; it will
    break exactly where the two sides diverge.
    """
    for c in db.get_rag_candidates():
        if c["rel_path"] == rel_path:
            return _transcript_key(c)
    raise AssertionError(f"{rel_path} is not even a candidate")


@pytest.fixture
def rag_stack(tmp_path):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    db.upsert_transcript(
        "cooking/pasta.mp4",
        _make_txt(
            [
                "We're cooking pasta with tomatoes and basil.",
                "Sauté the onions and add the tomatoes.",
                "Parmesan on top of the noodles at the end.",
            ]
        ),
        media_dir=str(tmp_path),
    )
    db.upsert_transcript(
        "auto/ev.mp4",
        _make_txt(
            [
                "The electric motor delivers three hundred kilowatts.",
                "The battery reaches six hundred kilometers of range.",
                "The suspension is tuned for the race track.",
            ]
        ),
        media_dir=str(tmp_path),
    )
    rag = VectorDB(str(tmp_path / "rag.db"), fts_language="english")
    rag.drop_all()
    idx = RagIndexer(
        rag,
        db,
        _fake_embed,
        chunk_max_chars=200,
        chunk_min_chars=40,
        batch_size=8,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    yield db, rag, idx
    try:
        rag.drop_all()
        db.close_all()
    except Exception:
        pass


def test_build_and_schema(rag_stack):
    db, rag, idx = rag_stack
    summary = idx.run()
    assert summary["indexed"] == 2
    assert summary["chunks"] > 0
    assert summary["dimension"] == _DIM
    assert summary["error_count"] == 0
    info = rag.current_schema_info()
    assert info["exists"] and info["dimensions"] == _DIM
    st = rag.stats()
    assert st["transcripts"] == 2 and st["chunks"] == summary["chunks"]


def test_vector_search(rag_stack):
    db, rag, idx = rag_stack
    idx.run()
    qv = _fake_embed(["tomatoes noodles recipe"])[0]
    res = rag.search_vector(qv, k=5)
    assert res and res[0]["transcript_key"] == _key(db, "cooking/pasta.mp4")
    assert 0.0 <= res[0]["score"] <= 1.0001


def test_fts_search(rag_stack):
    db, rag, idx = rag_stack
    idx.run()
    res = rag.search_fts("battery range kilometers", k=5)
    assert res and res[0]["transcript_key"] == _key(db, "auto/ev.mp4")


def test_hybrid_rrf(rag_stack):
    db, rag, idx = rag_stack
    idx.run()
    qv = _fake_embed(["tomatoes basil pasta"])[0]
    res = rag.search_hybrid(qv, "tomatoes basil", k=10, limit=5, rrf_k=60)
    assert res and res[0]["transcript_key"] == _key(db, "cooking/pasta.mp4")
    # Top hit is backed by both modalities
    assert res[0]["match"] == "hybrid"
    assert res[0]["vrank"] == 1 and res[0]["frank"] == 1


def test_resume_and_incremental(rag_stack, tmp_path):
    db, rag, idx = rag_stack
    idx.run()
    s2 = idx.run()
    assert s2["indexed"] == 0 and s2["skipped"] == 2
    # Change to one transcript → only that one re-embedded
    db.upsert_transcript(
        "cooking/pasta.mp4",
        _make_txt(
            [
                "We're cooking pasta with tomatoes and basil.",
                "Sauté the onions and add the tomatoes.",
                "Parmesan on top of the noodles at the end.",
                "Tiramisu with mascarpone for dessert.",
            ]
        ),
        media_dir=str(tmp_path),
    )
    s3 = idx.run()
    assert s3["indexed"] == 1 and s3["skipped"] == 1
    assert rag.search_fts("Tiramisu mascarpone", k=3)[0]["transcript_key"] == _key(
        db, "cooking/pasta.mp4"
    )


def test_force_and_clear(rag_stack):
    db, rag, idx = rag_stack
    idx.run()
    s = idx.run(force=True)
    assert s["indexed"] == 2 and s["skipped"] == 0
    rag.drop_all()
    assert rag.current_schema_info().get("exists") is False


def test_coverage_tracking(rag_stack, tmp_path):
    """Coverage: which transcripts are embedded, which are pending."""
    from whisper_tm.api.rag import _compute_coverage

    db, rag, idx = rag_stack

    # Before embedding: both pending
    cov0 = _compute_coverage(rag, db)
    assert cov0["total"] == 2
    assert cov0["embedded"] == 0
    assert cov0["pending"] == 2

    idx.run()
    cov1 = _compute_coverage(rag, db)
    assert cov1["total"] == 2
    assert cov1["embedded"] == 2
    assert cov1["pending"] == 0
    assert set(cov1["embedded_keys"]) == {_key(db, "cooking/pasta.mp4"), _key(db, "auto/ev.mp4")}

    # Change one transcript → counts as stale/pending
    db.upsert_transcript(
        "cooking/pasta.mp4",
        _make_txt(
            [
                "Completely new content here.",
                "Second sentence.",
                "Third.",
                "Fourth added.",
            ]
        ),
        media_dir=str(tmp_path),
    )
    cov2 = _compute_coverage(rag, db)
    assert cov2["embedded"] == 1
    assert cov2["pending"] == 1
    assert cov2["stale"] == 1
    assert _key(db, "cooking/pasta.mp4") in cov2["pending_keys"]


def test_coverage_uses_the_same_key_as_the_indexer(rag_stack):
    """The key mismatch bug, as a test — indexer and coverage must agree.

    Until then, `_compute_coverage` built the key as `rel_path`, while
    the indexer used the canonical `abs_path`. Each was correct on its
    own; together they meant: after a complete, error-free indexing run,
    coverage permanently reported "0 embedded, everything pending".

    The existing tests couldn't catch this — they checked counts, and
    those were each plausible on their own. This test checks what
    actually matters: that both sides' key sets are IDENTICAL.
    """
    from whisper_tm.api.rag import _compute_coverage

    db, rag, idx = rag_stack
    idx.run()

    from_the_index = set(rag.get_indexed_map())
    from_the_coverage = set(_compute_coverage(rag, db)["embedded_keys"])

    assert from_the_index, "The indexing run stored nothing at all"
    assert from_the_index == from_the_coverage, (
        "Indexer and coverage build different keys — "
        f"index: {sorted(from_the_index)}, coverage: {sorted(from_the_coverage)}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  Batching across transcripts
# ══════════════════════════════════════════════════════════════════════════
class _CountingEmbedder:
    """Remembers how big the batches actually sent out were."""

    def __init__(self, fail_from: int | None = None):
        self.calls: list[int] = []
        self.fail_from = fail_from

    def __call__(self, texts):
        self.calls.append(len(texts))
        if self.fail_from is not None and len(texts) >= self.fail_from:
            raise RuntimeError("Batch too big — deliberately failed")
        return _fake_embed(texts)


def _many_transcripts(db, tmp_path, count=30, sentences=2):
    """Many SMALL transcripts — exactly the real-world case.

    In a typical library, a transcript has only a few chunks. A test
    with one large transcript wouldn't show the problem at all.
    """
    for i in range(count):
        db.upsert_transcript(
            f"folder/file{i:03d}.mp4",
            _make_txt(
                [
                    f"Sentence {j} in recording {i}, with enough text in it."
                    for j in range(sentences)
                ]
            ),
            media_dir=str(tmp_path),
        )


def test_a_batch_spans_transcripts(tmp_path):
    """`batch_size` was never reached: batching happened only per transcript.

    In a typical library a transcript has only a few chunks; nearly all are
    below batch_size=64. So instead of batches of 64, tiny batches went out.

    IMPORTANT for the expectation: on a local server the change brings
    practically NO speedup (chunks/s stays about the same) —
    there, GPU compute time per token dominates everything. What it does
    bring is the number of requests, and that matters as soon as the
    endpoint is over the network. This test therefore checks the batch
    size, not the speed.
    """
    db = Database(tmp_path / "wx.db")
    db.initialize()
    _many_transcripts(db, tmp_path, count=30)
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    counter = _CountingEmbedder()
    idx = RagIndexer(
        rag,
        db,
        counter,
        chunk_max_chars=10_000,
        chunk_min_chars=0,
        batch_size=16,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        s = idx.run()
        assert s["indexed"] == 30 and s["error_count"] == 0
        # Without the batching, EVERY call would be as big as one transcript.
        large = [n for n in counter.calls if n > 4]
        assert large, (
            f"not a single batch over 4 chunks — embedding happened per transcript: {counter.calls}"
        )
        assert max(counter.calls) >= 10, f"batches stayed small: {counter.calls}"
    finally:
        rag.drop_all()
        db.close_all()


def test_a_failed_batch_retries_one_by_one(tmp_path):
    """A broken chunk must not take down 64 healthy transcripts with it."""
    db = Database(tmp_path / "wx.db")
    db.initialize()
    _many_transcripts(db, tmp_path, count=12)
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    # Fails on anything that looks like a bulk batch; individual calls
    # (1-2 chunks) go through.
    counter = _CountingEmbedder(fail_from=5)
    idx = RagIndexer(
        rag,
        db,
        counter,
        chunk_max_chars=10_000,
        chunk_min_chars=0,
        batch_size=16,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        s = idx.run()
        assert s["error_count"] == 0, s["errors"]
        assert s["indexed"] == 12, "transcripts were lost in the batch failure"
        assert rag.stats()["transcripts"] == 12
    finally:
        rag.drop_all()
        db.close_all()


# ══════════════════════════════════════════════════════════════════════════
#  CJK and chunks too long for the embedding server
# ══════════════════════════════════════════════════════════════════════════
# Counted as characters, 1,200 of Japanese came to ~2,000 tokens with the
# speaker labels — over what a llama.cpp embedding server takes per input
# ("Context size has been exceeded"), and the recording stayed out of the
# index, on every run.


def test_text_weight_counts_cjk_heavier_and_nothing_else():
    from whisper_tm.rag.chunker import text_weight

    assert text_weight("") == 0
    latin = "Alice brings the cake, and Bob sets up the long table. Ünïcödé — ok!"
    assert text_weight(latin) == len(latin), "other languages must chunk as before"
    assert text_weight("日本語") == 3 + 5 and text_weight("ひらがな") == 4 + 7


def test_japanese_chunks_stay_within_the_servers_context():
    from whisper_tm.core.llm.budget import estimate_tokens

    sentence = "今日はみんなで公園に集まって誕生日のパーティーを準備しました。"  # invented
    segs = _segs([sentence * 2] * 60)
    chunks = chunk_segments(segs, max_chars=1200, min_chars=250, overlap_segments=1)
    assert len(chunks) > 5
    assert max(estimate_tokens(c.content) for c in chunks) < 1000


def test_latin_chunks_are_unchanged():
    segs = _segs([f"Sentence number {i} of a long recording, with words." for i in range(80)])
    chunks = chunk_segments(segs, max_chars=300, min_chars=100, overlap_segments=1)
    # Measured in plain characters, as before: every chunk closes right
    # after it reaches 300 characters of segment text
    for c in chunks[:-1]:
        segs_in = segs[c.seg_start : c.seg_end + 1]
        total = sum(len(x["text"]) for x in segs_in)
        assert total >= 300 and total - len(segs_in[-1]["text"]) < 300


class _ContextLimitedEmbedder:
    """Refuses any input longer than its 'context', like llama.cpp."""

    def __init__(self, limit: int = 7000):
        self.limit, self.inputs = limit, []

    def __call__(self, texts):
        self.inputs.append([len(t) for t in texts])
        if any(len(t) > self.limit for t in texts):
            raise RuntimeError(
                'Embedding-API 500: {"error":{"code":500,'
                '"message":"Context size has been exceeded."}}'
            )
        return _fake_embed(texts)


def test_a_chunk_too_long_for_the_server_is_embedded_from_its_beginning(tmp_path):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    _many_transcripts(db, tmp_path, count=3)
    long_text = " ".join(f"word{i}" for i in range(2500))  # ~17,000 characters
    db.upsert_transcript("folder/long.mp4", _make_txt([long_text]), media_dir=str(tmp_path))
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    emb = _ContextLimitedEmbedder()
    idx = RagIndexer(
        rag,
        db,
        emb,
        chunk_max_chars=1200,
        chunk_min_chars=0,
        batch_size=16,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        s = idx.run()
        assert s["error_count"] == 0, s["errors"]
        assert s["indexed"] == 4 and rag.stats()["transcripts"] == 4
        with rag.connection() as conn:
            stored = [r[0] for r in conn.execute("SELECT content FROM chunks")]
        assert any(len(c) > 15000 for c in stored), "the chunk keeps its whole text"
        from whisper_tm.rag.indexer import EMBED_FIT_WEIGHT

        assert any(
            len(x) == 1 and EMBED_FIT_WEIGHT * 0.8 <= x[0] <= EMBED_FIT_WEIGHT for x in emb.inputs
        ), "its beginning was embedded on its own"
    finally:
        rag.drop_all()
        db.close_all()


def test_other_embedding_errors_still_fail_the_recording(tmp_path):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    _many_transcripts(db, tmp_path, count=2)
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()

    def broken(texts):
        if any("Sentence" in t for t in texts):  # the probe passes
            raise RuntimeError("Embedding-API 503: model is loading")
        return _fake_embed(texts)

    idx = RagIndexer(
        rag,
        db,
        broken,
        chunk_max_chars=1200,
        chunk_min_chars=0,
        batch_size=16,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        s = idx.run()
        assert s["error_count"] == 2 and s["indexed"] == 0
    finally:
        rag.drop_all()
        db.close_all()


def test_a_run_gives_up_when_the_embedding_keeps_failing(tmp_path):
    """The embedding server gone: a run over a whole library logged a failure
    per transcript for hours. After FAILS_IN_A_ROW in a row it stops and
    says why — the next run picks up where it stopped."""
    from whisper_tm.rag import indexer as ix

    db = Database(tmp_path / "wx.db")
    db.initialize()
    _many_transcripts(db, tmp_path, count=60)
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    tried = []

    def gone(texts):
        if any("Sentence" in t for t in texts):  # the probe passes
            tried.append(len(texts))
            raise RuntimeError("Connection refused")
        return _fake_embed(texts)

    idx = RagIndexer(
        rag,
        db,
        gone,
        chunk_max_chars=10_000,
        chunk_min_chars=0,
        batch_size=4,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        s = idx.run()
        assert s["gave_up"].startswith(f"Stopped: {ix.FAILS_IN_A_ROW} transcripts in a row")
        assert "Connection refused" in s["gave_up"]
        assert s["indexed"] == 0 and s["error_count"] == ix.FAILS_IN_A_ROW
    finally:
        rag.drop_all()
        db.close_all()


def test_failures_with_successes_between_them_go_on(tmp_path):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    _many_transcripts(db, tmp_path, count=60)
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    calls = []

    def flaky(texts):
        calls.append(1)
        if len(calls) % 3 == 0:  # every third call fails
            raise RuntimeError("Embedding-API 503: busy")
        return _fake_embed(texts)

    idx = RagIndexer(
        rag,
        db,
        flaky,
        chunk_max_chars=10_000,
        chunk_min_chars=0,
        batch_size=1,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        s = idx.run()
        assert s["gave_up"] == "" and s["indexed"] + s["error_count"] == 60
    finally:
        rag.drop_all()
        db.close_all()


def test_batching_yields_the_same_as_one_by_one(tmp_path):
    """The change must not alter the RESULT, only the speed."""
    results = []
    for run, size in (("batched", 64), ("one_by_one", 1)):
        db = Database(tmp_path / f"wx_{run}.db")
        db.initialize()
        _many_transcripts(db, tmp_path, count=15)
        rag = VectorDB(str(tmp_path / "rag.db"))
        rag.drop_all()
        idx = RagIndexer(
            rag,
            db,
            _fake_embed,
            chunk_max_chars=10_000,
            chunk_min_chars=0,
            batch_size=size,
            embedding_model="fake",
            default_media_dir=str(tmp_path),
        )
        idx.run()
        with rag.connection() as c:
            rows = c.execute(
                f"SELECT transcript_key, chunk_index, content "
                f"FROM {rag.TABLE} ORDER BY transcript_key, chunk_index"
            ).fetchall()
        results.append(
            [(r["transcript_key"].split("wx_")[-1], r["chunk_index"], r["content"]) for r in rows]
        )
        rag.drop_all()
        db.close_all()
    assert results[0] == results[1], "batched and individual runs produce different chunks"


def test_an_empty_transcript_counts_as_done(tmp_path):
    """Otherwise the indexer keeps re-tackling the same empty ones every run."""
    db = Database(tmp_path / "wx.db")
    db.initialize()
    db.upsert_transcript(
        "empty/nothing.mp4",
        {
            "language": "de",
            "model": "turbo",
            "complete": True,
            "speakers": [],
            "segments": [],
            "total_duration_sec": 0,
        },
        media_dir=str(tmp_path),
    )
    _many_transcripts(db, tmp_path, count=3)
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    idx = RagIndexer(
        rag,
        db,
        _fake_embed,
        chunk_max_chars=10_000,
        chunk_min_chars=0,
        batch_size=16,
        embedding_model="fake",
        default_media_dir=str(tmp_path),
    )
    try:
        idx.run()
        assert _key(db, "empty/nothing.mp4") in rag.get_indexed_map()
        s2 = idx.run()
        assert s2["indexed"] == 0, "the empty transcript was touched again"
    finally:
        rag.drop_all()
        db.close_all()


def test_the_metadata_survives_the_close(tmp_path):
    """Writes must not be lost when the connection ends.

    `sqlite3` opens an implicit transaction on an INSERT; a `close()`
    without a commit silently discards it. In practice this meant that
    after a complete, apparently error-free run over a whole library,
    both `embedding_model` and `index_fingerprint` were missing from the
    meta — and without the fingerprint, a changed chunking configuration
    never counts as changed.
    """
    rag = VectorDB(str(tmp_path / "rag.db"), dimensions=4)
    rag.ensure_schema()
    with rag.connection() as c:
        rag._set_meta(c, "embedding_model", "qwen3-embedding-4b")
        rag._set_meta(c, "index_fingerprint", "abc123")
    meta = rag.get_meta()
    assert meta.get("embedding_model") == "qwen3-embedding-4b"
    assert meta.get("index_fingerprint") == "abc123"


def test_an_error_rolls_back(tmp_path):
    """A half-written state is worse than none at all."""
    rag = VectorDB(str(tmp_path / "rag.db"), dimensions=4)
    rag.ensure_schema()
    with pytest.raises(RuntimeError):
        with rag.connection() as c:
            rag._set_meta(c, "half", "written")
            raise RuntimeError("midway")
    assert "half" not in rag.get_meta()


def test_the_index_run_stores_model_and_fingerprint(rag_stack):
    """The run must record WHAT it embedded with.

    Otherwise the next run doesn't notice a model or config change and
    leaves stale vectors in place.
    """
    _db, rag, idx = rag_stack
    idx.run()
    meta = rag.get_meta()
    assert meta.get("embedding_model") == "fake"
    assert meta.get("index_fingerprint"), "no fingerprint stored"

    # Change the chunking configuration → the next run must re-embed.
    idx.chunk_max_chars = 999
    s = idx.run()
    assert s["indexed"] == 2 and s["skipped"] == 0, "config change went unnoticed"


# ── Which model the fingerprint names ───────────────────────────────────────
#
# Locally llama.cpp runs the GGUF file (embedding.local_model) and ignores
# the model name in the request. The fingerprint used `embedding.model` —
# a name in the settings — so swapping the GGUF for another model of the
# same dimension went unnoticed: new vectors joined an index of old ones.


def test_locally_the_fingerprint_names_the_gguf_file(monkeypatch):
    from whisper_tm.config import config
    from whisper_tm.rag.service import index_model_name

    monkeypatch.setitem(
        config._data,
        "embedding",
        {
            "local": True,
            "model": "qwen3-embedding",
            "local_model": "models/gguf/Qwen3-Embedding-4B-Q8_0.gguf",
        },
    )
    # The file name, not the path: moving the file changes no vector
    assert index_model_name() == "Qwen3-Embedding-4B-Q8_0.gguf"
    # Remote the request names the model, and the server runs that one
    config._data["embedding"]["local"] = False
    assert index_model_name() == "qwen3-embedding"


def test_the_old_form_covers_only_what_it_recorded(rag_stack):
    """Taken as unchanged means: same chunking, same settings name. A
    chunking change in the same step still re-embeds."""
    _db, rag, idx = rag_stack
    idx.embedding_model = "qwen3-embedding"
    idx.run()
    idx.embedding_model = "Qwen3-Embedding-4B-Q8_0.gguf"
    idx.legacy_embedding_model = "qwen3-embedding"
    idx.chunk_max_chars = 999
    assert idx.run()["indexed"] == 2


# ── From the index back into the UI ────────────────────────────────────────
#
# The vector index knows transcripts by their `abs_path` (the identity,
# docs/PATHS.md). But the search view jumps via `rel_path` + root. Without
# translation at the boundary, a semantically related hit is readable but
# not clickable — you see the spot and can't get there.


def test_display_paths_translate_the_key(rag_stack, monkeypatch):
    db, rag, idx = rag_stack
    import whisper_tm.api.rag as api

    monkeypatch.setattr(api, "db", db)

    key = _key(db, "cooking/pasta.mp4")
    result = api._display_paths([key])
    assert result[key]["rel_path"] == "cooking/pasta.mp4"
    assert result[key]["file_dir"], "without a root the jump goes nowhere"


def test_display_paths_know_legacy_rows_too(rag_stack, monkeypatch):
    """Before the switch to `abs_path`, `rel_path` was the key.

    Such rows are still in the index. If the translation didn't find
    them, the oldest transcripts of all would end up not clickable.
    """
    db, rag, idx = rag_stack
    import whisper_tm.api.rag as api

    monkeypatch.setattr(api, "db", db)

    result = api._display_paths(["auto/ev.mp4"])
    assert result["auto/ev.mp4"]["rel_path"] == "auto/ev.mp4"


def test_display_paths_do_not_choke_on_the_unknown(rag_stack, monkeypatch):
    """A hit whose transcript has meanwhile vanished from the DB must not
    take down the whole search — it just becomes not clickable."""
    db, rag, idx = rag_stack
    import whisper_tm.api.rag as api

    monkeypatch.setattr(api, "db", db)

    result = api._display_paths(["/does/not/exist.mp4", None, ""])
    assert result == {}


def test_display_paths_stay_below_the_placeholder_limit(rag_stack, monkeypatch):
    """SQLite caps the number of placeholders per query. The translation
    must therefore query in blocks — otherwise it fails for large k."""
    db, rag, idx = rag_stack
    import whisper_tm.api.rag as api

    monkeypatch.setattr(api, "db", db)

    many = [f"/nothing/{i}.mp4" for i in range(500)]
    many.append(_key(db, "cooking/pasta.mp4"))
    result = api._display_paths(many)
    assert len(result) == 1


# ── Querying the progress of a running indexing job ────────────────────────
#
# `/api/rag/build` is a stream AND a start command: a second call while a
# run is in progress gets 409. Anyone who only opens settings after the
# start had no way to see progress — the card stood still even though
# a whole library of transcripts was running through in the background.


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config as cfg

    cfg.load(None)
    cfg._data.setdefault("directories", {})["media"] = str(tmp_path)
    cfg._resolve_directories()
    cfg._ensure_directories()
    from whisper_tm.app import create_app

    return create_app().test_client()


def test_progress_without_a_run_reports_quiet(api_client):
    d = api_client.get("/api/rag/build/progress").get_json()
    assert d["running"] is False
    assert d["progress"] is None


def test_progress_hands_out_the_last_message(api_client):
    """The endpoint reads the shared state, not the SSE stream."""
    import whisper_tm.api.rag as api

    with api._build_lock:
        api._build_state.update(
            running=True,
            started_at=1.0,
            progress={
                "phase": "index",
                "done": 2000,
                "total": 10000,
                "indexed": 1900,
                "skipped": 100,
                "chunks": 3500,
            },
        )
    try:
        d = api_client.get("/api/rag/build/progress").get_json()
        assert d["running"] is True
        assert d["progress"]["done"] == 2000
        assert d["progress"]["indexed"] == 1900
    finally:
        with api._build_lock:
            api._build_state.update(running=False, progress=None, started_at=0.0)


def test_progress_has_no_side_effect(api_client):
    """Pure reporting: the call must not start a run — otherwise an open
    settings window would kick off a second one on every tick."""
    import whisper_tm.api.rag as api

    before = dict(api._build_state)
    api_client.get("/api/rag/build/progress")
    api_client.get("/api/rag/build/progress")
    assert api._build_state["running"] == before["running"]
    assert api._build_state["thread"] is before["thread"]


def test_the_build_hands_the_indexer_the_model_that_runs(api_client, monkeypatch):
    """The wiring for the fingerprint: locally the GGUF file name, remote
    the settings name."""
    import whisper_tm.api.rag as api
    from whisper_tm import extensions
    from whisper_tm.config import config

    # The module holds the database it saw at its first import — pin the
    # app's own, whatever imported the module earlier in the run
    monkeypatch.setattr(api, "db", extensions.db)
    seen = {}

    class Indexer:
        def __init__(self, *a, **kw):
            seen.clear()
            seen.update(kw)

        def run(self, **kw):
            return {"indexed": 0, "errors": []}

    monkeypatch.setattr(api, "RagIndexer", Indexer)
    monkeypatch.setattr(api, "embedding_ready", lambda: (True, ""))
    monkeypatch.setattr(api, "make_embed_fn", lambda *a, **k: lambda texts: [])
    monkeypatch.setattr(api, "make_rag_db", lambda: object())
    monkeypatch.setitem(
        config._data,
        "embedding",
        {
            "local": True,
            "model": "qwen3-embedding",
            "local_model": "models/gguf/Qwen3-Embedding-4B-Q8_0.gguf",
        },
    )

    body = api_client.get("/api/rag/build").get_data(as_text=True)
    assert '"phase": "complete"' in body, body
    assert seen["embedding_model"] == "Qwen3-Embedding-4B-Q8_0.gguf"
    assert "legacy_embedding_model" not in seen

    config._data["embedding"]["local"] = False
    body = api_client.get("/api/rag/build").get_data(as_text=True)
    assert '"phase": "complete"' in body, body
    assert seen["embedding_model"] == "qwen3-embedding"


def test_the_export_leaves_no_copy_behind(api_client, tmp_path, monkeypatch):
    """send_file marked the response direct_passthrough, so its close
    callback never ran: every export left a copy of the whole store in the
    temp folder."""
    import tempfile

    from whisper_tm.rag.service import forget_rag_db

    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    forget_rag_db()
    r = api_client.get("/api/rag/export")
    try:
        assert r.status_code == 200, r.get_data(as_text=True)[:200]
        assert r.get_data()[:16] == b"SQLite format 3\x00"
    finally:
        r.close()
        forget_rag_db()
    assert not [p for p in temp.iterdir() if p.name.startswith("wtm_ragexport_")]

"""The situation search: query words, full-text expressions, fusion, copies,
the quick search and its route (rag/retrieve.py, /api/rag/search).

What is pinned:

* query terms — stopwords out, long words as the prefix of their stem,
  names (the voice library's, and capitalised words inside a sentence) and
  quoted words as must terms; every FTS5 token quoted, so no query can break
  the expression (checked against a real FTS5 table);
* Reciprocal Rank Fusion of the dense and full-text lists;
* copies: same text (speaker labels aside) or vectors at cosine ≥ 0.97 are
  one group, the best-ranked row its representative — a shared intro is not
  a copy;
* the quick search on a real small index with a fake embedding and a fake
  reranker: reranked order, the reranker unavailable → fused order and a
  note, copies collapsed with their places for the jump;
* ``/api/rag/search``: with rerank/group the new shape, without them the old
  one, unchanged.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3

import numpy as np
import pytest

from whisper_tm.core import rerank_server as rs
from whisper_tm.core.stopwords import stopwords
from whisper_tm.rag import retrieve as rt

STOP = stopwords(["de", "en"])


# ══════════════════════════════════════════════════════════════════════════
#  Query terms
# ══════════════════════════════════════════════════════════════════════════


class TestQueryTerms:
    def test_names_are_must_terms_the_rest_content_words(self):
        t = rt.query_terms("bob gets insulted", stop=STOP, known_names={"bob"})
        assert t["must"] == ["bob"] and t["names"] == ["bob"]
        assert t["words"] == ["gets", "insult*"]

    def test_capitalised_inside_a_sentence_is_a_must_term(self):
        t = rt.query_terms("Someone insults Carol. Then Dave laughs", stop=STOP)
        # "Someone" and "Then" start a sentence — they are no names
        assert t["must"] == ["carol", "dave"]
        assert t["capitalised"] == ["carol", "dave"]
        assert t["words"] == ["someon*", "insult*", "laugh*"]

    def test_quoted_words_and_phrases(self):
        t = rt.query_terms('they talk about "Erin Smith" and “money”', stop=STOP)
        assert t["must"] == ["erin smith", "money"]
        assert t["quoted"] == ["erin smith", "money"]
        assert t["words"] == ["talk"]

    def test_stopwords_short_words_and_numbers(self):
        t = rt.query_terms("he was at it in 2016 with an ox", stop=STOP)
        assert t["must"] == [] and t["words"] == ["2016"]

    @pytest.mark.parametrize(
        "word,form",
        [
            ("insulted", "insult*"),
            ("insults", "insult*"),
            ("screaming", "scream*"),
            ("lied", "lied"),
            ("money", "money*"),
            ("the2nd", "the2nd"),
            ("someone", "someon*"),
            ("admitted", "admitt*"),
            ("lying", "lying*"),
        ],
    )
    def test_the_prefix_form(self, word, form):
        assert rt.prefix_form(word) == form

    def test_caps(self):
        many = " ".join(f"word{i}x" for i in range(60))
        assert len(rt.query_terms(many, stop=STOP)["words"]) == rt.MAX_WORDS
        names = " ".join(f"a Name{chr(97 + i)}x" for i in range(10))
        assert len(rt.query_terms(names, stop=STOP)["must"]) == rt.MAX_MUST

    def test_the_library_names_are_read_once_and_kept(self, monkeypatch):
        calls = []

        class Db:
            def people(self):
                calls.append(1)
                return [{"name": "Bob Smith"}, {"name": "The Boss"}, {"name": None}]

        import whisper_tm.database.speaker as sp

        monkeypatch.setattr(sp, "speaker_db", lambda: Db())
        monkeypatch.setitem(rt._names_cache, "at", -1e9)
        assert rt.known_names() == frozenset({"bob", "smith", "boss"})
        assert rt.known_names() == frozenset({"bob", "smith", "boss"})
        assert calls == [1]

    def test_without_the_library_no_names(self, monkeypatch):
        import whisper_tm.database.speaker as sp

        def broken():
            raise sqlite3.OperationalError("no such table")

        monkeypatch.setattr(sp, "speaker_db", broken)
        monkeypatch.setitem(rt._names_cache, "at", -1e9)
        assert rt.known_names() == frozenset()


class TestFtsExpressions:
    def test_every_token_quoted(self):
        assert rt.fts_term("insult*") == '"insult"*'
        assert rt.fts_term("erin smith") == '"erin smith"'
        assert rt.fts_term('"*') is None
        assert rt.fts_or(["bob", "insult*"]) == '("bob" OR "insult"*)'
        assert rt.fts_and(["bob"]) == '"bob"'
        assert rt.fts_must_any(["bob"], ["insult*", "idiot"]) == '"bob" AND ("insult"* OR "idiot")'
        assert rt.fts_must_any([], ["idiot"]) == '"idiot"'
        assert rt.fts_must_any(["bob", "carol"], []) == '("bob" AND "carol")'
        assert rt.fts_or([]) == ""

    @pytest.mark.parametrize(
        "query",
        [
            'what does "that cost?',
            "NEAR(a b) OR AND NOT",
            "bob* -x ^y {col}: z",
            "a\"b'c",
            "na\u00efve r\u00e9sum\u00e9 caf\u00e9",
            "(((",
            "* * *",
        ],
    )
    def test_no_query_breaks_the_expression(self, query):
        c = sqlite3.connect(":memory:")
        c.execute(
            "CREATE VIRTUAL TABLE t USING fts5(content, tokenize='unicode61 remove_diacritics 2')"
        )
        c.execute("INSERT INTO t(content) VALUES ('bob near the cost of anger')")
        t = rt.query_terms(query, stop=STOP, known_names={"bob"})
        for expr in (rt.fts_must_any(t["must"], t["words"]), rt.fts_or(t["must"] + t["words"])):
            if expr:
                c.execute("SELECT count(*) FROM t WHERE t MATCH ?", (expr,)).fetchone()

    def test_diacritics_match_both_ways(self):
        c = sqlite3.connect(":memory:")
        c.execute(
            "CREATE VIRTUAL TABLE t USING fts5(content, tokenize='unicode61 remove_diacritics 2')"
        )
        c.execute("INSERT INTO t(content) VALUES ('the caf\u00e9 was closed')")
        for word in ("cafe", "caf\u00e9", "CAF\u00c9"):
            expr = rt.fts_or([rt.prefix_form(word)])
            assert (
                c.execute("SELECT count(*) FROM t WHERE t MATCH ?", (expr,)).fetchone()[0] == 1
            ), word


# ══════════════════════════════════════════════════════════════════════════
#  Fusion and copies
# ══════════════════════════════════════════════════════════════════════════


def _row(i, content="", key="k", start=0.0, score=0.5):
    return {
        "id": i,
        "content": content,
        "transcript_key": key,
        "chunk_index": 0,
        "start_sec": start,
        "end_sec": start + 5,
        "media_path": key,
        "score": score,
    }


class TestFusion:
    def test_reciprocal_rank_fusion(self):
        dense = [_row(1, score=0.9), _row(2, score=0.8)]
        fts = [_row(2, score=7.0), _row(3, score=5.0)]
        out = rt.fuse([("dense", dense), ("any", fts)], rrf_k=60)
        assert [r["id"] for r in out] == [2, 1, 3]
        two = out[0]
        assert two["fused"] == pytest.approx(1 / 62 + 1 / 61)
        assert two["match"] == "hybrid" and two["vrank"] == 2 and two["frank"] == 1
        assert two["vscore"] == 0.8 and two["fscore"] == 7.0
        assert out[1]["match"] == "vector" and out[2]["match"] == "fts"
        assert out[1]["lists"] == ["dense"] and two["lists"] == ["dense", "any"]
        assert "score" not in two

    def test_best_full_text_rank_counts(self):
        out = rt.fuse(
            [("must_any", [_row(5, score=1.0)]), ("any", [_row(4, score=9.0), _row(5, score=3.0)])]
        )
        five = next(r for r in out if r["id"] == 5)
        assert five["frank"] == 1 and five["fscore"] == 3.0
        assert five["fused"] == pytest.approx(1 / 61 + 1 / 62)

    def test_ties_go_by_id(self):
        out = rt.fuse([("a", [_row(9)]), ("b", [_row(3)])])
        assert [r["id"] for r in out] == [3, 9]


class TestCopies:
    def test_same_text_other_folder_is_one_group(self):
        rows = [
            _row(1, "[Bob] You are such an idiot.", "/a/x.mp4", 10),
            _row(2, "Something else entirely", "/a/y.mp4"),
            _row(3, "[SPEAKER_00] You are such an IDIOT!", "/backup/x.mp4", 10),
        ]
        groups = rt.group_copies(rows)
        assert [g["id"] for g in groups] == [1, 2]
        assert groups[0]["copies"] == [
            {
                "id": 3,
                "transcript_key": "/backup/x.mp4",
                "chunk_index": 0,
                "start_sec": 10,
                "end_sec": 15,
                "media_path": "/backup/x.mp4",
            }
        ]
        assert groups[1]["copies"] == []

    def test_near_identical_vectors_are_one_group(self):
        rows = [
            _row(1, "alpha one", key="k1"),
            _row(2, "beta two", key="k2"),
            _row(3, "gamma three", key="k3"),
        ]
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.999, 0.02, 0.0]], np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        groups = rt.group_copies(rows, v)
        assert [g["id"] for g in groups] == [1, 2]
        assert [c["id"] for c in groups[0]["copies"]] == [3]

    def test_the_best_rank_represents_and_copies_chain(self):
        rows = [
            _row(1, "unique", key="k1"),
            _row(2, "chain a", key="k2"),
            _row(3, "chain b", key="k3"),
            _row(4, "chain a", key="k4"),
        ]
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0.99, 0.14], [0, 0.98, 0.2]], np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        groups = rt.group_copies(rows, v, threshold=0.97)
        assert [g["id"] for g in groups] == [1, 2]
        assert [c["id"] for c in groups[1]["copies"]] == [3, 4]

    def test_a_shared_intro_does_not_make_a_copy(self):
        intro = "Hello and welcome to another episode of our show " * 8
        rows = [
            _row(1, intro + "today we talk about cars and engines"),
            _row(2, intro + "today we talk about cooking and pasta"),
        ]
        assert len(rt.group_copies(rows)) == 2

    def test_rows_without_a_vector_are_compared_by_text_only(self):
        rows = [_row(1, "one"), _row(2, "two")]
        v = np.array([[1.0, 0.0], [np.nan, np.nan]], np.float32)
        assert len(rt.group_copies(rows, v)) == 2

    def test_the_same_video_in_another_container_is_one_group(self):
        """One video as .mp4, .mkv and .webm, each transcribed on
        its own — slightly different words and chunk borders."""
        base = "she said that most of the work was done by then and now"
        rows = [
            dict(_row(1, base + " is that true", key="/a/Part 1.mp4", start=200.0)),
            dict(_row(2, base + " is it true", key="/b/Part 1.mkv", start=230.0)),
            dict(_row(3, base + " yes", key="/c/Part 1.webm", start=3000.0)),
            dict(
                _row(
                    4,
                    "a completely different topic about cooking and pasta",
                    key="/d/Part 1.webm",
                    start=240.0,
                )
            ),
        ]
        rows[1]["start_sec"], rows[1]["end_sec"] = 202.0, 207.0  # mostly the same span
        g = rt.group_copies(rows)
        assert [x["id"] for x in g] == [1, 3, 4]
        assert [c["id"] for c in g[0]["copies"]] == [2]

    def test_neighbouring_passages_of_one_file_stay_apart(self):
        rows = [
            dict(_row(1, "bob is such an idiot he says", key="/a/x.mp4", start=10.0)),
            dict(_row(2, "bob is such an idiot she says", key="/a/x.mp4", start=40.0)),
        ]
        assert len(rt.group_copies(rows)) == 2

    def test_same_name_uses_the_vectors_when_there_are_some(self):
        rows = [
            dict(_row(1, "a b c", key="/a/v.mp4", start=0.0)),
            dict(_row(2, "x y z", key="/b/v.mkv", start=1.0)),
        ]
        near = [[1.0, 0.0], [0.8, 0.6]]  # cosine 0.8 — the same moment
        far = [[1.0, 0.0], [0.6, 0.8]]  # cosine 0.6 — a shifted neighbour
        assert len(rt.group_copies(rows, near)) == 1
        assert len(rt.group_copies(rows, far)) == 2

    def test_a_group_never_takes_two_passages_of_one_file(self):
        """Copies chain: .mkv passage ~ .webm passage ~ the next .mkv passage.
        Two passages of one recording are never copies of each other."""
        rows = [
            _row(1, "first passage", key="/a/v.mkv", start=0.0),
            _row(2, "something", key="/b/v.webm", start=2.0),
            _row(3, "second passage", key="/a/v.mkv", start=4.0),
        ]
        v = np.array([[1, 0, 0], [0.99, 0.14, 0], [0.98, 0.2, 0]], np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        groups = rt.group_copies(rows, v, threshold=0.97)
        assert sorted(g["id"] for g in groups) == [1, 3]
        assert sum(len(g["copies"]) for g in groups) == 1

    def test_the_same_length_under_another_name_is_one_group(self):
        """A video saved twice under different names (.mkv and .mp4) — another
        name, the same length; Find folds them (core/copies), so the meaning
        list does too."""
        base = "well let us just see how the new garden takes shape and then"
        rows = [
            dict(_row(1, base + " one", key="/a/funny_clip.mp4", start=40.0)),
            dict(_row(2, base + " two", key="/b/video_download_0001.mkv", start=41.0)),
            dict(_row(3, base + " three", key="/c/Episode 2.mp4", start=40.0)),
        ]
        lengths = {
            "/a/funny_clip.mp4": 120.05,
            "/b/video_download_0001.mkv": 120.0,
            "/c/Episode 2.mp4": 200.0,
        }
        g = rt.group_copies(rows, lengths=lengths)
        assert [x["id"] for x in g] == [1, 3] and [c["id"] for c in g[0]["copies"]] == [2]
        assert len(rt.group_copies(rows)) == 3  # without the lengths: names only

    def test_same_name_but_a_shifted_span_stays_apart(self):
        rows = [
            dict(_row(1, "bob is such an idiot no doubt", key="/a/v.mp4", start=0.0)),
            dict(_row(2, "bob is such an idiot no doubt", key="/b/v.mkv", start=4.0)),
        ]
        rows[0]["content"] += " one"
        rows[1]["content"] += " two"  # not text-identical
        assert len(rt.group_copies(rows)) == 2, "80 % of the span apart: neighbours, not copies"

    def test_text_key(self):
        assert rt.text_key("[Bob] Hi, you!  [Carol] What?") == "hiyouwhat"
        long = "x" * 1000 + "END"
        assert rt.text_key(long).endswith("|" + ("x" * 97) + "end")


# ══════════════════════════════════════════════════════════════════════════
#  A real small index
# ══════════════════════════════════════════════════════════════════════════

_DIM = 256


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for tok in rt._WORD.findall((t or "").lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _txt(sents, speaker="SPEAKER_00"):
    segs = [
        {"speaker": speaker, "start": i * 3.0, "end": i * 3.0 + 3, "text": s}
        for i, s in enumerate(sents)
    ]
    return {
        "language": "en",
        "model": "turbo",
        "complete": True,
        "speakers": [speaker],
        "segments": segs,
        "total_duration_sec": len(segs) * 3,
    }


INSULT = [
    "Bob, you are such an idiot, everybody insults you.",
    "Stop insulting Bob all the time, he did nothing.",
]


@pytest.fixture
def index(tmp_path, monkeypatch):
    from whisper_tm.database import Database
    from whisper_tm.rag.indexer import RagIndexer
    from whisper_tm.rag.vector_db import VectorDB

    db = Database(tmp_path / "wx.db")
    db.initialize()
    root = tmp_path / "media"
    for rel, sents in (
        ("videos/bob_day.mp4", INSULT),
        ("backup/videos/bob_day.mp4", INSULT),
        (
            "videos/cooking.mp4",
            ["We are cooking pasta with tomatoes.", "Parmesan goes on top at the end."],
        ),
        (
            "videos/bob_garden.mp4",
            ["Bob talks about the weather today.", "The garden needs water, says Bob."],
        ),
    ):
        db.upsert_transcript(rel, _txt(sents), media_dir=str(root), abs_path=str(root / rel))
    rag = VectorDB(str(tmp_path / "rag.db"))
    rag.drop_all()
    RagIndexer(
        rag,
        db,
        _fake_embed,
        chunk_max_chars=2000,
        chunk_min_chars=0,
        batch_size=8,
        embedding_model="fake",
        default_media_dir=str(root),
        include_speaker_labels=True,
    ).run()
    monkeypatch.setattr(rt, "known_names", lambda: frozenset({"bob"}))
    yield {"db": db, "rag": rag, "root": str(root)}
    db.close_all()


def _fake_rerank(calls=None):
    def rerank(query, docs, **kw):
        if calls is not None:
            calls.append((query, list(docs), kw))
        return [
            0.95
            if "insult" in d.lower()
            else (0.3 if "bob" in rt._WORD.findall(d.lower()) else 0.001)
            for d in docs
        ]

    return rerank


class TestCandidates:
    def test_the_lists_and_their_fusion(self, index):
        rows, info = rt.candidates(
            "bob gets insulted", k_dense=10, k_fts=10, db=index["rag"], embed_fn=_fake_embed
        )
        # "bob" AND an action word: the two copies; any word: Bob's garden too
        assert info["lists"] == {"dense": 4, "must_any": 2, "any": 3}
        assert info["must"] == ["bob"] and "insult*" in info["words"]
        top = rows[0]
        assert "insult" in top["content"].lower() and top["match"] == "hybrid"
        assert set(top["lists"]) == {"dense", "must_any", "any"}

    def test_a_failing_embedding_leaves_the_full_text(self, index):
        def broken(texts):
            raise RuntimeError("embedding server not reachable")

        rows, info = rt.candidates(
            "bob gets insulted", k_dense=10, k_fts=10, db=index["rag"], embed_fn=broken
        )
        assert "not reachable" in info["dense_error"]
        assert rows and all(r["match"] == "fts" for r in rows)

    def test_a_replaced_must_keeps_the_query_word(self, index):
        _rows, info = rt.candidates(
            "bob gets insulted", k_dense=0, k_fts=10, must=[], db=index["rag"]
        )
        assert info["must"] == [] and "bob" in info["words"]

    def test_the_must_list_of_the_deep_search(self, index):
        _rows, info = rt.candidates(
            "bob gets insulted", k_dense=0, k_fts=10, k_must=50, or_list=False, db=index["rag"]
        )
        assert info["lists"] == {"must_any": 2, "must": 3}


class TestCommonWords:
    def test_a_word_in_most_excerpts_is_left_out(self, tmp_path):
        """A plan's everyday words ("conversation", "sentence") fill the lists with
        everyday talk; a word in more than 5 % of the excerpts is no clue."""
        from whisper_tm.rag.vector_db import VectorDB

        db = VectorDB(str(tmp_path / "rag.db"))
        db.ensure_schema(dimensions=4)
        rows = [
            {
                "transcript_key": f"k{i}",
                "chunk_index": 0,
                "content": ("we talk a lot " if i % 2 else "nothing here ")
                + ("idiot " if i < 3 else "")
                + f"word{i}",
                "embedding": [1.0, 0.0, 0.0, 0.0],
            }
            for i in range(400)
        ]
        with db.connection() as c:
            db.upsert_chunks(c, rows)
        kept, dropped = rt.common_terms_out(db, ["talk", "idiot", "nothing"])
        assert kept == ["idiot"] and dropped == ["talk", "nothing"]

    def test_a_small_index_keeps_every_word(self, index):
        assert rt.common_terms_out(index["rag"], ["bob", "insult*"]) == (["bob", "insult*"], [])

    def test_only_a_plan_s_extra_words_are_filtered(self, index, monkeypatch):
        seen = []
        monkeypatch.setattr(
            rt,
            "common_terms_out",
            lambda db, words, **kw: seen.append(list(words)) or (list(words), []),
        )
        rt.candidates("bob gets insulted", k_dense=0, k_fts=10, db=index["rag"])
        assert seen == []
        rt.candidates(
            "bob gets insulted", k_dense=0, k_fts=10, db=index["rag"], extra_terms=["idiot"]
        )
        assert seen and "idiot" in seen[0]


class TestQuickSearch:
    def test_reranked_grouped_and_cut(self, index):
        calls = []
        out = rt.quick_search(
            "bob gets insulted",
            k=10,
            db=index["rag"],
            embed_fn=_fake_embed,
            rerank_fn=_fake_rerank(calls),
        )
        assert out["reranked"] is True and out["note"] is None and out["grouped"]
        res = out["results"]
        # Four recordings, one of them a copy: three groups
        assert len(res) == out["candidates"] == 3
        top = res[0]
        assert top["relevance"] == 0.95 and top["score"] == 0.95
        pair = {index["root"] + "/videos/bob_day.mp4", index["root"] + "/backup/videos/bob_day.mp4"}
        assert {top["transcript_key"]} | {c["transcript_key"] for c in top["copies"]} == pair
        assert len(top["copies"]) == 1
        assert [r["relevance"] for r in res] == sorted((r["relevance"] for r in res), reverse=True)
        assert out["min_score"] == rs.DEFAULTS["quick_min_score"]
        query, docs, kw = calls[0]
        assert query == "bob gets insulted" and len(docs) == 3
        assert kw["wait_sec"] == rs.QUICK_WAIT_SEC

    def test_only_the_best_groups_are_reranked(self, index, monkeypatch):
        monkeypatch.setattr(
            rs,
            "setting_int",
            lambda name: 1 if name == "quick_candidates" else int(rs.DEFAULTS[name]),
        )
        calls = []
        out = rt.quick_search(
            "bob gets insulted",
            k=10,
            db=index["rag"],
            embed_fn=_fake_embed,
            rerank_fn=_fake_rerank(calls),
        )
        assert len(calls[0][1]) == 1
        assert out["results"][0]["relevance"] is not None
        assert all(r["relevance"] is None for r in out["results"][1:])
        assert out["results"][1]["score"] == out["results"][1]["fused"]

    def test_without_the_reranker_the_fused_order_and_a_note(self, index):
        def busy(*a, **k):
            raise rs.RerankUnavailable("busy", "the GPU is in use by the queue")

        plain = rt.quick_search(
            "bob gets insulted", k=10, db=index["rag"], embed_fn=_fake_embed, rerank=False
        )
        out = rt.quick_search(
            "bob gets insulted", k=10, db=index["rag"], embed_fn=_fake_embed, rerank_fn=busy
        )
        assert out["reranked"] is False
        assert out["note"] == "Reranker busy (the GPU is in use) — sorted by meaning and wording"
        assert [r["id"] for r in out["results"]] == [r["id"] for r in plain["results"]]
        assert all(r["relevance"] is None for r in out["results"])
        assert plain["note"] is None

    def test_a_broken_reranker_does_not_take_the_search(self, index):
        def broken(*a, **k):
            raise ValueError("something unexpected")

        out = rt.quick_search(
            "bob gets insulted", k=10, db=index["rag"], embed_fn=_fake_embed, rerank_fn=broken
        )
        assert out["reranked"] is False and len(out["results"]) == 3
        assert (
            out["note"] == "Reranker failed (something unexpected) — sorted by meaning and wording"
        )

    def test_ungrouped_and_full_text_only(self, index):
        def never(texts):
            raise AssertionError("full text needs no embedding")

        out = rt.quick_search(
            "bob gets insulted",
            k=10,
            mode="fts",
            group=False,
            rerank=False,
            db=index["rag"],
            embed_fn=never,
        )
        keys = [r["transcript_key"] for r in out["results"]]
        # Both copies of Bob's day (not collapsed) and his garden
        assert len(keys) == 3 and all(r["copies"] == [] for r in out["results"])
        assert all(r["match"] == "fts" for r in out["results"])

    def test_k_cuts(self, index):
        out = rt.quick_search(
            "bob gets insulted",
            k=1,
            db=index["rag"],
            embed_fn=_fake_embed,
            rerank_fn=_fake_rerank(),
        )
        assert len(out["results"]) == 1 and out["candidates"] == 3

    def test_places_for_the_jump(self, index):
        out = rt.quick_search(
            "bob gets insulted",
            k=10,
            db=index["rag"],
            embed_fn=_fake_embed,
            rerank_fn=_fake_rerank(),
        )
        rt.place_copies(out["results"], index["db"])
        top = out["results"][0]
        assert top["rel_path"] in ("videos/bob_day.mp4", "backup/videos/bob_day.mp4")
        assert top["file_dir"] == index["root"]
        assert top["copies"][0]["rel_path"] in ("videos/bob_day.mp4", "backup/videos/bob_day.mp4")
        assert top["copies"][0]["file_dir"] == index["root"]

    def test_the_place_takes_its_root_from_the_identity(self, index, tmp_path):
        """Some transcripts carry their file's own folder as
        media_dir while rel_path is relative to a root above it: the hit
        must hand on that root, or its context, playback and opening point
        at a path that doesn't exist (the reading column said "Transcript
        not found")."""
        db = index["db"]
        root = tmp_path / "library"
        f = root / "videos" / "Archive" / "clip.mp4"
        db.upsert_transcript(
            "videos/Archive/clip.mp4",
            {
                "complete": True,
                "language": "de",
                "segments": [{"speaker": "A", "start": 0, "end": 2, "text": "hello"}],
            },
            media_dir=str(root),
            abs_path=str(f),
        )
        with db.connection() as conn:
            conn.execute(
                "UPDATE transcripts SET media_dir = ? WHERE abs_path = ?", (str(f.parent), str(f))
            )
        got = rt.display_paths([str(f)], db)
        assert got == {str(f): {"rel_path": "videos/Archive/clip.mp4", "file_dir": str(root)}}

    def test_unknown_keys_have_no_place(self, index):
        assert rt.display_paths(["/nowhere.mp4", None, ""], index["db"]) == {}
        assert rt.display_paths(["x"], None) == {}


# ══════════════════════════════════════════════════════════════════════════
#  The route
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def client(index, monkeypatch):
    from flask import Flask

    import whisper_tm.api.rag as api

    monkeypatch.setattr(api, "db", index["db"])
    monkeypatch.setattr(api, "make_rag_db", lambda: index["rag"])
    monkeypatch.setattr(api, "make_query_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(api, "embedding_ready", lambda: (True, ""))
    monkeypatch.setattr(rs, "rerank", _fake_rerank())
    app = Flask(__name__)
    app.register_blueprint(api.rag_bp)
    return app.test_client()


class TestTheRoute:
    def test_rerank_and_group(self, client, index):
        d = client.get(
            "/api/rag/search?mode=vector&k=50&rerank=1&group=1&q=bob%20gets%20insulted"
        ).get_json()
        assert d["ok"] and d["mode"] == "vector" and d["count"] == 3
        assert d["reranked"] is True and d["note"] is None and d["grouped"] is True
        assert d["min_score"] == 0.05 and d["candidates"] == 3
        top = d["results"][0]
        assert top["relevance"] == 0.95 and top["score"] == 0.95
        assert top["rel_path"] and top["file_dir"] == index["root"]
        assert len(top["copies"]) == 1
        copy = top["copies"][0]
        assert set(copy) == {
            "transcript_key",
            "rel_path",
            "file_dir",
            "start_sec",
            "end_sec",
            "media_path",
            "chunk_index",
        }
        assert copy["rel_path"] and copy["file_dir"] == index["root"]
        for key in (
            "fused",
            "vscore",
            "fscore",
            "match",
            "content",
            "speaker",
            "start_sec",
            "end_sec",
            "chunk_index",
        ):
            assert key in top

    def test_meaning_mode_has_both_on(self, client):
        d = client.get("/api/rag/search?mode=meaning&q=bob%20gets%20insulted").get_json()
        assert d["reranked"] is True and d["grouped"] is True

    def test_a_busy_reranker_says_so(self, client, monkeypatch):
        def busy(*a, **k):
            raise rs.RerankUnavailable("busy")

        monkeypatch.setattr(rs, "rerank", busy)
        d = client.get("/api/rag/search?mode=vector&rerank=1&group=1&q=bob").get_json()
        assert d["ok"] and d["reranked"] is False
        assert d["note"].startswith("Reranker busy")
        assert all(r["relevance"] is None for r in d["results"])

    def test_the_classic_answer_is_unchanged(self, client):
        d = client.get("/api/rag/search?mode=vector&k=12&q=bob%20gets%20insulted").get_json()
        assert d["ok"] and d["mode"] == "vector"
        assert set(d) == {"ok", "mode", "count", "elapsed_ms", "results"}
        assert "relevance" not in d["results"][0] and "copies" not in d["results"][0]
        h = client.get("/api/rag/search?q=bob%20gets%20insulted").get_json()
        assert h["mode"] == "hybrid" and h["results"][0]["match"] in ("hybrid", "vector", "fts")

    def test_long_queries_and_unknown_modes_are_refused(self, client):
        # Regression: 5,000 characters took long to embed; mode=bogus ran hybrid
        r = client.get("/api/rag/search?q=" + "bob " * 200)
        assert r.status_code == 400 and "too long" in r.get_json()["error"]
        r = client.get("/api/rag/search?mode=bogus&q=bob")
        assert r.status_code == 400 and "mode" in r.get_json()["error"]

    def test_k_is_bounded(self, client):
        d = client.get("/api/rag/search?mode=vector&rerank=1&k=5000&q=bob").get_json()
        assert d["ok"] and d["count"] <= 300

    def test_without_an_embedding_it_says_why(self, client, monkeypatch):
        import whisper_tm.api.rag as api

        monkeypatch.setattr(api, "embedding_ready", lambda: (False, "Embedding endpoint not set"))
        r = client.get("/api/rag/search?mode=vector&rerank=1&q=bob")
        assert r.status_code == 400 and "not set" in r.get_json()["error"]
        # Full text needs none
        d = client.get("/api/rag/search?mode=fts&rerank=1&q=bob").get_json()
        assert d["ok"] and d["count"] >= 1

    def test_an_empty_index_says_how_to_build_it(self, client, monkeypatch, tmp_path):
        """A new library: every search said "no such table: chunks"."""
        import whisper_tm.api.rag as api
        from whisper_tm.rag.vector_db import VectorDB

        empty = VectorDB(str(tmp_path / "empty_rag.db"))
        monkeypatch.setattr(api, "make_rag_db", lambda: empty)
        for url in (
            "/api/rag/search?mode=meaning&q=bob",
            "/api/rag/search?q=bob",
            "/api/rag/search?mode=fts&q=bob",
        ):
            r = client.get(url)
            assert r.status_code == 400, url
            assert r.get_json()["error"] == api.EMPTY_INDEX


def test_the_status_names_the_reranker(client, monkeypatch):
    monkeypatch.setattr(rs, "available", lambda: (False, "switched off (rerank.enabled)"))
    d = client.get("/api/rag/status").get_json()
    r = d["rerank"]
    assert r["available"] is False and r["detail"] == "switched off (rerank.enabled)"
    for key in ("enabled", "running", "endpoint", "model", "pid", "idle_limit_sec"):
        assert key in r
    # Counted, not loaded (a whole library's rows for a number)
    import whisper_tm.api.rag as api

    want = len(api.db.get_rag_candidates())
    monkeypatch.setattr(api.db, "get_rag_candidates", lambda: pytest.fail("loaded to count"))
    assert client.get("/api/rag/status").get_json()["source"] == {"transcripts": want}

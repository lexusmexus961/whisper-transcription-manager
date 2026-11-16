"""The deep search — "find everything" for a situation (rag/deep_search.py).

What is pinned, on a real small index with a fake embedding, a fake
reranker and a fake language model:

* the plan: read from the model's JSON, or from what is left of a cut-off
  answer; keywords deduplicated, stopwords out, the query's own words
  first; must terms only where the query has them; without a model the
  fallback;
* the run: candidates → groups (copies collapsed) → the reranker's order
  with the plan's criteria → one passage per call to the model, in that
  order, at the prefilter or above; YES a hit, UNCLEAR a maybe, NO gone;
  results with their places and copies; progress and timings;
* every verdict written at once — a restart goes on without asking again
  (no new plan, no new ranking, no passage twice);
* stop between passages; the early stop after a run of NO below 0.2; the
  reranker unavailable (fused order, all checked) or dropping out midway;
  no language model (ranked, unchecked); the model unreachable (an error,
  what was found stays);
* the routes and the LLM queue entry of kind ``deep_search``.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from whisper_tm.core import llm_client
from whisper_tm.core import rerank_server as rs
from whisper_tm.core.stopwords import stopwords
from whisper_tm.rag import deep_search as ds
from whisper_tm.rag import retrieve as rt

STOP = stopwords(["de", "en"])
QUERY = "bob gets insulted"

PLAN = {
    "situation": "Bob is insulted",
    "criteria_yes": "Only if someone insults Bob himself.",
    "criteria_no": "Not if Bob is only mentioned.",
    "rerank_criteria": "Relevant only if Bob himself is insulted. Not relevant if "
    "Bob is only mentioned.",
    "must_terms": ["bob"],
    "keywords": ["insult*", "idiot", "insult*", "the", "idiot", "fool"],
}


# ══════════════════════════════════════════════════════════════════════════
#  The plan
# ══════════════════════════════════════════════════════════════════════════


class TestThePlan:
    def test_from_the_models_json(self):
        p = ds.parse_plan(json.dumps(PLAN), QUERY, stop=STOP, known_names=())
        assert p["source"] == "llm" and p["situation"] == "Bob is insulted"
        assert p["must_terms"] == ["bob"]
        # The query's own words first; deduplicated; stopwords out
        assert p["keywords"] == ["gets", "insult*", "idiot", "fool"]
        assert p["criteria_yes"] == "Only if someone insults Bob himself."
        assert p["rerank_criteria"].startswith("Relevant only if")

    def test_must_terms_only_where_the_query_has_them(self):
        p = ds.parse_plan(
            json.dumps({**PLAN, "must_terms": ["bob", "idiot", "carol"]}),
            QUERY,
            stop=STOP,
            known_names=(),
        )
        assert p["must_terms"] == ["bob"]

    def test_quoted_words_and_library_names_are_must_terms_anyway(self):
        p = ds.parse_plan(
            json.dumps({**PLAN, "must_terms": []}),
            'bob says "sorry" to Dave',
            stop=STOP,
            known_names={"bob"},
        )
        assert p["must_terms"] == ["sorry", "bob"]
        # A capitalised word is a keyword in a deep search, not a must term
        assert "dave" in p["keywords"]

    def test_a_cut_off_looping_answer(self):
        cut = (
            '{"situation": "Bob is insulted", "criteria_yes": "Only if Bob is '
            'insulted.", "criteria_no": "Not \\"mentions\\".", "rerank_criteria": '
            '"Relevant only if.", "must_terms": ["bob"], "keywords": ["insult*", '
            '"idiot", ' + '"crazy", ' * 300 + '"cra'
        )
        p = ds.parse_plan(cut, QUERY, stop=STOP, known_names=())
        assert p["situation"] == "Bob is insulted"
        assert p["criteria_no"] == 'Not "mentions".'
        assert p["keywords"] == ["gets", "insult*", "idiot", "crazy"]

    def test_a_phrase_of_stopwords_is_no_keyword(self):
        """A real model returned "ich habe" among the keywords — it matches
        nearly every excerpt."""
        p = ds.parse_plan(
            json.dumps({**PLAN, "keywords": ["ich habe", "ich hab", "tut mir leid", "insult*"]}),
            QUERY,
            stop=STOP,
        )
        assert "ich habe" not in p["keywords"] and "ich hab" not in p["keywords"]
        assert "tut mir leid" in p["keywords"] and "insult*" in p["keywords"]

    def test_at_most_forty_keywords(self):
        many = {**PLAN, "keywords": [f"word{i}x" for i in range(80)]}
        assert len(ds.parse_plan(json.dumps(many), QUERY, stop=STOP)["keywords"]) == 40

    def test_nothing_usable(self):
        assert ds.parse_plan("I can't help with that.", QUERY, stop=STOP) is None
        assert ds.parse_plan('{"situation": "", "keywords": []}', "the", stop=STOP) is None

    def test_the_fallback(self, monkeypatch):
        monkeypatch.setattr(rt, "known_names", lambda: frozenset({"bob"}))
        p = ds.fallback_plan('bob gets "really" insulted by Carol', "no model")
        assert p["source"] == "fallback" and p["reason"] == "no model"
        assert p["must_terms"] == ["really", "bob"]
        assert p["keywords"] == ["gets", "insult*", "carol"]
        assert p["criteria_yes"] == "" and p["situation"].startswith("bob gets")


class FakeLlm:
    """The plan for a call with a schema; a verdict per passage otherwise."""

    def __init__(self, plan=PLAN, verdicts=None, fail=None):
        self.plan = plan
        self.verdicts = verdicts or {"insult": "YES", "garden": "UNCLEAR"}
        self.fail = fail or {}  # call number → exception
        self.plans, self.checks = [], []

    def chat(self, messages, **kw):
        n = len(self.plans) + len(self.checks)
        self.calls_kw = kw
        if n in self.fail:
            raise self.fail.pop(n)
        if kw.get("response_schema") is not None or "Query:" in messages[-1]["content"]:
            self.plans.append((messages, kw))
            return json.dumps(self.plan), {}
        self.checks.append((messages, kw))
        text = messages[-1]["content"].lower()
        for word, answer in self.verdicts.items():
            if word in text.split('"""')[1]:
                return answer, {}
        return "NO", {}


class TestMakePlan:
    def test_asked_once_with_the_schema_and_thinking_off(self, monkeypatch):
        fake = FakeLlm()
        monkeypatch.setattr(llm_client, "chat", fake.chat)
        monkeypatch.setattr(ds, "_llm_ready", lambda: True)
        monkeypatch.setattr(rt, "known_names", lambda: frozenset())
        p = ds.make_plan(QUERY)
        assert p["source"] == "llm" and p["must_terms"] == ["bob"]
        ((messages, kw),) = fake.plans
        assert messages[0]["content"] == ds.PLAN_SYSTEM
        assert messages[1]["content"] == f"Query: {QUERY}"
        assert kw["response_schema"] == ds.PLAN_SCHEMA
        assert kw["template_kwargs"] == {"enable_thinking": False}
        assert kw["thinking"] is False and kw["task"] == ds.TASK

    def test_a_rejected_schema_asks_for_plain_json(self, monkeypatch):
        calls = []

        def chat(messages, **kw):
            calls.append(kw)
            if kw.get("response_schema"):
                raise llm_client.LlmAnswerError("LLM error 400: response_format not supported")
            return "```json\n" + json.dumps(PLAN) + "\n```", {}

        monkeypatch.setattr(llm_client, "chat", chat)
        monkeypatch.setattr(ds, "_llm_ready", lambda: True)
        monkeypatch.setattr(rt, "known_names", lambda: frozenset())
        assert ds.make_plan(QUERY)["source"] == "llm"
        assert len(calls) == 2 and "response_schema" not in calls[1]

    @pytest.mark.parametrize(
        "error",
        [
            llm_client.LlmConnectionError("Timeout after 300s"),
            llm_client.LlmAnswerError("LLM error 500: x"),
        ],
    )
    def test_without_an_answer_the_fallback(self, monkeypatch, error):
        def chat(messages, **kw):
            raise error

        monkeypatch.setattr(llm_client, "chat", chat)
        monkeypatch.setattr(ds, "_llm_ready", lambda: True)
        monkeypatch.setattr(rt, "known_names", lambda: frozenset())
        p = ds.make_plan(QUERY)
        assert p["source"] == "fallback" and "did not answer" in p["reason"]

    def test_an_unusable_answer_the_fallback(self, monkeypatch):
        monkeypatch.setattr(llm_client, "chat", lambda m, **k: ("Sorry.", {}))
        monkeypatch.setattr(ds, "_llm_ready", lambda: True)
        monkeypatch.setattr(rt, "known_names", lambda: frozenset())
        assert ds.make_plan(QUERY)["reason"] == "the language model's plan was not usable"

    def test_no_model_set_up(self, monkeypatch):
        monkeypatch.setattr(ds, "_llm_ready", lambda: False)
        monkeypatch.setattr(rt, "known_names", lambda: frozenset())
        assert ds.make_plan(QUERY)["reason"] == "no language model set up"


# ══════════════════════════════════════════════════════════════════════════
#  The check of one passage
# ══════════════════════════════════════════════════════════════════════════


class TestTheCheck:
    @pytest.mark.parametrize(
        "answer,verdict",
        [
            ("YES", "hit"),
            ("Yes.", "hit"),
            (" yes", "hit"),
            ("**YES**", "hit"),
            ("NO", "no"),
            ("No!", "no"),
            ("UNCLEAR", "maybe"),
            ("Unclear", "maybe"),
            ("", "maybe"),
            ("Possibly", "maybe"),
            ("NOT SURE", "maybe"),
        ],
    )
    def test_the_verdict(self, answer, verdict):
        assert ds.parse_verdict(answer) == verdict

    def test_the_prompt(self):
        text = ds.verify_prompt("[Bob] you idiot", PLAN)
        assert '"""\n[Bob] you idiot\n"""' in text
        assert 'Does the situation "Bob is insulted" happen in this excerpt?' in text
        assert "Only if someone insults Bob himself. Not if Bob is only mentioned." in text
        assert text.endswith("Answer with exactly one English word: YES, UNCLEAR or NO.")
        generic = ds.verify_prompt("x" * 5000, {"situation": "s"})
        assert ds.GENERIC_CRITERIA in generic
        assert "x" * rs.DOC_CHARS in generic and "x" * (rs.DOC_CHARS + 1) not in generic

    def test_asked_with_five_tokens_and_thinking_off(self, monkeypatch):
        fake = FakeLlm()
        monkeypatch.setattr(llm_client, "chat", fake.chat)
        assert ds.ask_verdict("he insults Bob", PLAN) == ("hit", "YES")
        _m, kw = fake.checks[0]
        assert kw["max_tokens"] == 5 and kw["thinking"] is False
        assert kw["template_kwargs"] == {"enable_thinking": False}
        assert kw["temperature"] == 0.0

    def test_the_order(self):
        scores = [0.9, None, 0.01, 0.5, None, 0.05]
        assert ds.verify_order(scores, 0.05, 100) == [0, 3, 5, 1, 4]
        assert ds.verify_order(scores, 0.05, 2) == [0, 3]
        assert ds.verify_order([None, None, None], 0.05, 10) == [0, 1, 2]

    def test_the_rerank_instruction(self):
        assert ds.rerank_instruction(PLAN) == rs.CRITERIA_INSTRUCTION + PLAN["rerank_criteria"]
        own = {**PLAN, "rerank_criteria": "", "source": "llm"}
        assert ds.rerank_instruction(own).endswith("Not if Bob is only mentioned.")
        assert ds.rerank_instruction({"source": "fallback"}) == rs.default_instruction()


# ══════════════════════════════════════════════════════════════════════════
#  A run on a real small index
# ══════════════════════════════════════════════════════════════════════════

_DIM = 256


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for tok in rt._WORD.findall((t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _txt(sents):
    segs = [
        {"speaker": "SPEAKER_00", "start": i * 3.0, "end": i * 3.0 + 3, "text": s}
        for i, s in enumerate(sents)
    ]
    return {
        "language": "en",
        "model": "turbo",
        "complete": True,
        "speakers": ["SPEAKER_00"],
        "segments": segs,
        "total_duration_sec": len(segs) * 3,
    }


INSULT = [
    "Bob, you are such an idiot, everybody insults you.",
    "Stop insulting Bob all the time, he did nothing.",
]


class FakeRerank:
    """0.95 for an insult, 0.3 for Bob, 0.001 otherwise; ``fail_from``: the
    call (0-based) that finds the card busy."""

    def __init__(self, fail_from=None):
        self.calls = []
        self.fail_from = fail_from

    def __call__(self, query, docs, **kw):
        if self.fail_from is not None and len(self.calls) >= self.fail_from:
            raise rs.RerankUnavailable("busy", "the GPU is in use by the queue")
        self.calls.append((query, list(docs), kw))
        if kw.get("progress"):
            kw["progress"](len(docs), len(docs))
        return [
            0.95
            if "insult" in d.lower()
            else (0.3 if "bob" in rt._WORD.findall(d.lower()) else 0.001)
            for d in docs
        ]


class FakeQueue:
    def __init__(self):
        self.progresses = []

    def progress(self, entry_id, pct, msg):
        self.progresses.append((pct, msg))

    def abort_requested(self, entry_id):
        return False


@pytest.fixture
def lib(tmp_path, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.config import config
    from whisper_tm.core.llm_queue import llm_queue
    from whisper_tm.database import Database
    from whisper_tm.rag.indexer import RagIndexer
    from whisper_tm.rag.vector_db import VectorDB

    db = Database(tmp_path / "wx.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    # An LLM queue worker another test's app started may still run
    monkeypatch.setattr(llm_queue, "_paused", True)
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
    ).run()
    monkeypatch.setattr(ds, "_rag_db", lambda: rag)
    monkeypatch.setattr(ds, "_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(ds, "_llm_ready", lambda: True)
    monkeypatch.setattr(rt, "known_names", lambda: frozenset({"bob"}))
    rerank = FakeRerank()
    monkeypatch.setattr(rs, "rerank", rerank)
    llm = FakeLlm()
    monkeypatch.setattr(llm_client, "chat", llm.chat)
    ds._deleted.clear()
    ds._stops.clear()
    yield {
        "db": db,
        "rag": rag,
        "root": str(root),
        "tmp": tmp_path,
        "rerank": rerank,
        "llm": llm,
        "mp": monkeypatch,
    }
    db.close_all()


def _made(query=QUERY) -> str:
    """A search saved as create() saves it — without queueing it."""
    s = ds._new_record(query)
    ds.save(s)
    return s["id"]


class TestARun:
    def test_plan_candidates_order_check(self, lib):
        did = _made()
        q = FakeQueue()
        out = ds.run(did, q, 7)
        s = ds.load(did)
        assert s["status"] == "done" and s["stage"] == "done" and s["finished"]
        assert out == {
            "deep_search": did,
            "hits": 1,
            "maybe": 1,
            "checked": 2,
            "duration": out["duration"],
        }
        assert s["plan"]["source"] == "llm" and s["plan"]["must_terms"] == ["bob"]
        # Four passages, one a copy: three groups; the cooking one below the
        # prefilter is never checked
        assert s["progress"] == {
            "candidates": 4,
            "groups": 3,
            "ranked": 3,
            "rank_total": 3,
            "verified": 2,
            "verify_total": 2,
        }
        assert s["counts"] == {"hits": 1, "maybe": 1, "no": 0}
        assert s["reranked"] is True and s["note"] is None
        hit, maybe = s["results"]
        assert hit["verdict"] == "hit" and maybe["verdict"] == "maybe"
        assert hit["n"] == 0 and maybe["n"] == 1
        assert hit["score"] == hit["relevance"] == 0.95 and maybe["score"] == 0.3
        assert "insult" in hit["snippet"].lower() and "garden" in maybe["snippet"].lower()
        assert hit["rel_path"] in ("videos/bob_day.mp4", "backup/videos/bob_day.mp4")
        assert hit["file_dir"] == lib["root"]
        (copy,) = hit["copies"]
        assert {hit["rel_path"], copy["rel_path"]} == {
            "videos/bob_day.mp4",
            "backup/videos/bob_day.mp4",
        }
        assert copy["file_dir"] == lib["root"] and copy["start_sec"] == 0.0
        assert set(s["timings"]) == {"plan", "candidates", "ranking", "checking"}
        # The reranker: the query, the plan's criteria, a long wait, giving way
        ((query, docs, kw),) = lib["rerank"].calls
        assert query == QUERY and len(docs) == 3
        assert kw["instruction"] == rs.CRITERIA_INSTRUCTION + PLAN["rerank_criteria"]
        assert kw["wait_sec"] == rs.DEEP_WAIT_SEC and kw["yield_between"] is True
        # The model: one passage per call, best first
        checks = [m[-1]["content"] for m, _kw in lib["llm"].checks]
        assert len(checks) == 2 and "insult" in checks[0].lower()
        assert 'Does the situation "Bob is insulted" happen' in checks[0]
        assert [p for p, _m in q.progresses][-1] == 99
        assert any("Ranking" in m for _p, m in q.progresses)

    def test_hits_show_while_the_ranking_still_runs(self, lib):
        """Ranking thousands of groups can take many minutes: the best passages
        ranked so far are checked after each block, not after the last one."""
        lib["mp"].setattr(ds, "RANK_BLOCK", 1)
        events = []
        rerank, chat = lib["rerank"], lib["llm"].chat

        def ranking(query, docs, **kw):
            events.append("rank")
            return rerank(query, docs, **kw)

        def asking(messages, **kw):
            if "Does the situation" in messages[-1]["content"]:
                events.append("check")
            return chat(messages, **kw)

        lib["mp"].setattr(rs, "rerank", ranking)
        lib["mp"].setattr(llm_client, "chat", asking)
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert events.count("rank") == 3
        assert events.index("check") < len(events) - 1 - events[::-1].index("rank"), events
        # The insult passage (0.95) was checked early — and only once
        assert events.count("check") == 2 and s["counts"] == {"hits": 1, "maybe": 1, "no": 0}
        assert [x["verdict"] for x in s["results"]] == ["hit", "maybe"]

    def test_a_restart_goes_on_without_asking_again(self, lib):
        lib["mp"].setattr(ds, "LLM_FAILURES_MAX", 1)
        lib["mp"].setattr(ds, "LLM_RETRY_WAIT_SEC", 0.0)
        broken = FakeLlm(fail={2: llm_client.LlmConnectionError("Connection failed")})
        lib["mp"].setattr(llm_client, "chat", broken.chat)
        did = _made()
        with pytest.raises(RuntimeError, match="not reachable"):
            ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["status"] == "error" and "not reachable" in s["error"]
        assert [r["verdict"] for r in s["results"]] == ["hit"]  # kept
        again = FakeLlm()
        lib["mp"].setattr(llm_client, "chat", again.chat)
        ds.run(did, FakeQueue(), 8)
        s = ds.load(did)
        assert s["status"] == "done" and s["error"] is None
        assert again.plans == [], "planned again"
        assert len(lib["rerank"].calls) == 1, "ranked again"
        assert len(again.checks) == 1, "a passage was checked twice"
        assert [r["verdict"] for r in s["results"]] == ["hit", "maybe"]

    def test_a_model_outage_between_blocks_is_no_reranker_failure(self, lib):
        # Regression: the check after a block raised inside the reranker's try —
        # "Reranker failed", and the ranking was marked done for good
        lib["mp"].setattr(ds, "RANK_BLOCK", 1)
        lib["mp"].setattr(ds, "LLM_FAILURES_MAX", 1)
        lib["mp"].setattr(ds, "LLM_RETRY_WAIT_SEC", 0.0)
        broken = FakeLlm(fail={1: llm_client.LlmConnectionError("Connection failed")})
        lib["mp"].setattr(llm_client, "chat", broken.chat)
        did = _made()
        with pytest.raises(RuntimeError, match="not reachable"):
            ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["status"] == "error" and not s.get("note")
        assert ds._load_work(did)["ranked"] is False
        lib["mp"].setattr(llm_client, "chat", FakeLlm().chat)
        ds.run(did, FakeQueue(), 8)
        s = ds.load(did)
        assert s["status"] == "done" and s["reranked"] is True and not s.get("note")
        assert len(lib["rerank"].calls) == 3, "a block was ranked twice or not at all"

    def test_stop_between_passages(self, lib):
        did = _made()

        class Stopping(FakeLlm):
            def chat(self, messages, **kw):
                out = super().chat(messages, **kw)
                if self.checks:
                    ds._stops.add(did)
                return out

        lib["mp"].setattr(llm_client, "chat", Stopping().chat)
        with pytest.raises(InterruptedError):
            ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["status"] == "stopped" and s["finished"]
        assert [r["verdict"] for r in s["results"]] == ["hit"]
        assert did not in ds._stops

    def test_early_stop_after_a_run_of_no(self, lib):
        lib["mp"].setattr(ds, "EARLY_STOP_RUN", 1)
        lib["mp"].setattr(ds, "EARLY_STOP_SCORE", 0.5)
        no = FakeLlm(verdicts={"insult": "NO"})
        lib["mp"].setattr(llm_client, "chat", no.chat)
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert len(no.checks) == 1 and s["counts"]["no"] == 1
        assert "Stopped early" in s["note"] and s["results"] == []

    def test_without_the_reranker_all_in_fused_order(self, lib):
        lib["mp"].setattr(rs, "rerank", FakeRerank(fail_from=0))
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["reranked"] is False
        assert s["note"] == "Reranker busy (the GPU is in use) — sorted by meaning and wording"
        assert s["progress"]["verify_total"] == 3 and len(lib["llm"].checks) == 3
        assert all(r["score"] is None for r in s["results"])
        assert s["counts"] == {"hits": 1, "maybe": 1, "no": 1}

    def test_the_reranker_dropping_out_midway(self, lib):
        lib["mp"].setattr(ds, "RANK_BLOCK", 1)
        lib["mp"].setattr(rs, "rerank", FakeRerank(fail_from=1))
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["reranked"] is True
        assert "Reranker stopped after 1 of 3" in s["note"]
        # The ranked one first, the two others in fused order
        assert s["progress"]["verify_total"] == 3

    def test_a_broken_reranker_leaves_the_fused_order(self, lib):
        def broken(*a, **k):
            raise ValueError("something unexpected")

        lib["mp"].setattr(rs, "rerank", broken)
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["status"] == "done" and s["reranked"] is False
        assert "Reranker failed (something unexpected)" in s["note"]
        assert s["progress"]["verify_total"] == 3

    def test_without_a_language_model_ranked_but_unchecked(self, lib):
        lib["mp"].setattr(ds, "_llm_ready", lambda: False)

        def never(*a, **k):
            raise AssertionError("asked a model that is not there")

        lib["mp"].setattr(llm_client, "chat", never)
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["status"] == "done" and s["plan"]["source"] == "fallback"
        assert [r["verdict"] for r in s["results"]] == ["maybe", "maybe"]
        assert all(r["unchecked"] for r in s["results"])
        assert s["results"][0]["score"] == 0.95
        assert "ranked, not checked" in s["note"]
        assert s["counts"]["maybe"] == 2

    def test_a_passage_the_model_cannot_take_stays_unclear(self, lib):
        lib["mp"].setattr(
            llm_client,
            "chat",
            FakeLlm(fail={1: llm_client.LlmAnswerError("Empty response from LLM")}).chat,
        )
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert [r["verdict"] for r in s["results"]] == ["maybe", "maybe"]

    def test_nothing_in_the_index(self, lib):
        did = _made("zzzz qqqq")
        lib["mp"].setattr(lib["rag"], "search_vector", lambda *a, **k: [])
        lib["mp"].setattr(
            llm_client, "chat", FakeLlm(plan={**PLAN, "must_terms": [], "keywords": ["zzzz"]}).chat
        )
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        assert s["status"] == "done" and s["results"] == []
        assert "Nothing in the index matches" in s["note"]

    def test_since_is_a_cursor(self, lib):
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        p = ds.public(s, since=1)
        assert [r["n"] for r in p["results"]] == [1]
        assert p["next"] == p["total_results"] == 2
        assert ds.public(s, since=5)["results"] == []
        assert len(ds.public(s)["results"]) == 2
        assert p["plan"]["situation"] == "Bob is insulted" and "reason" not in p["plan"]
        assert p["elapsed_sec"] >= 0 and p["counts"]["hits"] == 1

    def test_stored_places_are_looked_up_again(self, lib):
        """A result stored with a wrong root (older records stored the
        media_dir — the file's own folder for many transcripts) is shown with
        the one the identity gives; without a database the stored one stands."""
        did = _made()
        ds.run(did, FakeQueue(), 7)
        s = ds.load(did)
        for r in s["results"]:
            r["file_dir"] = "/wrong/root"
            for c in r["copies"]:
                c["file_dir"] = "/wrong/root"
        shown = ds.public(s)["results"]
        assert {r["file_dir"] for r in shown} == {lib["root"]}
        assert all(c["file_dir"] == lib["root"] for r in shown for c in r["copies"])
        assert s["results"][0]["file_dir"] == "/wrong/root", "the record itself stays as stored"
        lib["mp"].setattr(ds, "_main_db", lambda: None)
        assert {r["file_dir"] for r in ds.public(s)["results"]} == {"/wrong/root"}


# ══════════════════════════════════════════════════════════════════════════
#  Routes and the LLM queue
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def client(lib):
    from flask import Flask

    from whisper_tm.api.llm_queue import llm_queue_bp
    from whisper_tm.api.rag import rag_bp

    app = Flask(__name__)
    app.register_blueprint(rag_bp)
    app.register_blueprint(llm_queue_bp)
    return app.test_client()


def _entry(lib, entry_id):
    with lib["db"].connection() as c:
        return dict(c.execute("SELECT * FROM llm_queue WHERE id=?", (entry_id,)).fetchone())


class TestTheRoutes:
    def test_start_list_get_and_the_queue_entry(self, client, lib):
        r = client.post("/api/rag/deep", json={"query": "  bob gets   insulted "}).get_json()
        assert r["ok"] and r["status"] == "queued" and r["entry_id"]
        e = _entry(lib, r["entry_id"])
        assert e["kind"] == "deep_search" and e["status"] == "pending"
        assert json.loads(e["params_json"]) == {"deep_id": r["id"]}
        assert e["filename"] == "Deep search · bob gets insulted"
        listed = client.get("/api/rag/deep").get_json()["searches"]
        assert listed[0]["id"] == r["id"] and listed[0]["query"] == QUERY
        assert listed[0]["counts"] == {"hits": 0, "maybe": 0}
        assert "results" not in listed[0]
        got = client.get(f"/api/rag/deep/{r['id']}").get_json()
        assert got["status"] == "queued" and got["stage"] == "plan" and got["plan"] is None
        assert got["results"] == [] and got["next"] == 0
        for key in (
            "progress",
            "counts",
            "created",
            "started",
            "finished",
            "elapsed_sec",
            "reranked",
            "note",
            "error",
            "message",
        ):
            assert key in got

    def test_the_queue_runs_it_without_an_llm_endpoint(self, client, lib, monkeypatch):
        from whisper_tm.config import config
        from whisper_tm.core import llm_tasks
        from whisper_tm.core.llm_queue import llm_queue

        monkeypatch.setitem(config._data, "llm", {**(config.get("llm") or {}), "endpoint": ""})
        r = client.post("/api/rag/deep", json={"query": QUERY}).get_json()
        entry = llm_queue._next_one()
        assert entry["kind"] == "deep_search"
        out = llm_tasks.execute(entry, llm_queue)
        assert out["deep_search"] == r["id"] and out["hits"] == 1
        d = client.get(f"/api/rag/deep/{r['id']}?since=1").get_json()
        assert d["status"] == "done" and [x["n"] for x in d["results"]] == [1]
        assert d["next"] == 2

    def test_refused(self, client):
        assert client.post("/api/rag/deep", json={"query": "  "}).status_code == 400
        assert client.post("/api/rag/deep", json={"query": "x" * 501}).status_code == 400
        assert client.post("/api/rag/deep", data="nonsense").status_code == 400
        assert client.get("/api/rag/deep/zzz").status_code == 404
        assert client.get("/api/rag/deep/aaaaaaaaaaaa").status_code == 404
        assert client.post("/api/rag/deep/aaaaaaaaaaaa/stop").status_code == 404
        assert client.delete("/api/rag/deep/aaaaaaaaaaaa").status_code == 404
        # The per-file queue route takes no deep search
        assert (
            client.post(
                "/api/llm/queue/add", json={"kind": "deep_search", "filename": "x"}
            ).status_code
            == 400
        )

    def test_stop_while_waiting_and_delete(self, client, lib):
        a = client.post("/api/rag/deep", json={"query": QUERY}).get_json()
        assert client.post(f"/api/rag/deep/{a['id']}/stop").get_json() == {"ok": True}
        got = client.get(f"/api/rag/deep/{a['id']}").get_json()
        assert got["status"] == "stopped" and got["finished"]
        assert _entry(lib, a["entry_id"])["status"] == "cancelled"
        assert client.post(f"/api/rag/deep/{a['id']}/stop").get_json() == {"ok": False}
        b = client.post("/api/rag/deep", json={"query": "someone lies"}).get_json()
        assert client.delete(f"/api/rag/deep/{b['id']}").get_json() == {"ok": True}
        assert client.get(f"/api/rag/deep/{b['id']}").status_code == 404
        assert _entry(lib, b["entry_id"])["status"] == "cancelled"
        assert not list((lib["tmp"] / "data" / "deep_searches").glob(f"{b['id']}*"))

    def test_an_entry_removed_in_the_llm_queue(self, client, lib):
        a = client.post("/api/rag/deep", json={"query": QUERY}).get_json()
        with lib["db"].connection() as c:
            c.execute("DELETE FROM llm_queue WHERE id=?", (a["entry_id"],))
        got = client.get(f"/api/rag/deep/{a['id']}").get_json()
        assert got["status"] == "stopped" and got["message"] == "Removed from the LLM queue"

    def test_retried_in_the_llm_queue_it_shows_as_waiting(self, client, lib):
        from whisper_tm.core.llm_queue import llm_queue

        a = client.post("/api/rag/deep", json={"query": QUERY}).get_json()
        assert client.post(f"/api/rag/deep/{a['id']}/stop").get_json()["ok"]
        assert llm_queue.retry(a["entry_id"])
        got = client.get(f"/api/rag/deep/{a['id']}").get_json()
        assert got["status"] == "queued" and got["finished"] is None
        # Regression: the stop of the waiting search stayed in _stops and stopped
        # the retry at once
        assert ds.run(a["id"], FakeQueue(), a["entry_id"])["hits"] == 1
        assert ds.load(a["id"])["status"] == "done"

    def test_without_an_index_it_says_so_and_does_not_start(self, lib, client, tmp_path):
        """Every list came back empty, and the full text logged a missing
        table — after the queue had taken the search."""
        from whisper_tm.rag.vector_db import VectorDB

        empty = VectorDB(str(tmp_path / "none.db"))
        lib["mp"].setattr(ds, "_rag_db", lambda: empty)
        r = client.post("/api/rag/deep", json={"query": QUERY})
        assert r.status_code == 400
        assert r.get_json()["error"].startswith("The meaning index is empty")
        assert ds.listing() == []

    def test_without_the_llm_queue_it_does_not_start(self, lib, monkeypatch):
        from whisper_tm.core.llm_queue import llm_queue

        monkeypatch.setattr(llm_queue, "enqueue", lambda *a, **k: 0)
        with pytest.raises(ds.DeepSearchError, match="LLM queue is not available"):
            ds.create(QUERY)
        (only,) = ds.listing()
        assert only["status"] == "error"

    def test_a_deleted_search_is_not_written_again(self, lib):
        did = _made()
        s = ds.load(did)
        assert ds.delete(did) is True
        ds.save(s)
        assert not (lib["tmp"] / "data" / "deep_searches" / f"{did}.json").exists()
        assert ds.delete(did) is False

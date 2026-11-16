"""Find anything (find/: plan, engines, fusion, judge, agent; api/find.py):
the plan checked against the library, the searches ranked by relevance and
inside the remembered files and people, the fusion with copies folded, the
judge with the facts it is told, a run, the routes, frames and clips.

The idea: one description — "Alice dances at a party, shouts, climbs
onto a table, says a certain phrase, I think a WhatsApp file" — every
search at once, a tile each, the language model ranking the results; the
spot playable, the suspected duplicates grouped (the same names are
sometimes different things). An independent review found: words came in
path order, hints and people only boosted, the plan wasn't grounded, the
judge invented facts, runs piled up on the shared language model.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path

import pytest
from flask import Flask

from whisper_tm.core import copies
from whisper_tm.find import agent, engines, fusion, judge
from whisper_tm.find import plan as planning


def _plan(**kw):
    p = {
        "summary": "s",
        "said_words": [],
        "said_meaning": "",
        "seen": "",
        "heard": "",
        "people": [],
        "file_hints": [],
        "media": "any",
        "media_stated": False,
        "language": "",
        "conversation": dict(planning.NO_CONVERSATION),
        "notes": [],
    }
    p.update(kw)
    return p


def _it(key, t, rank, kind, **kw):
    return engines._item(key, t=t, rank=rank, kind=kind, **kw)


class _Db:
    """A library in memory: media_files, transcripts, segments with their
    full-text index — what the searches, the folding and the clips read."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # as the app's database hands out rows
        self.conn.executescript(
            "CREATE TABLE media_files (abs_path TEXT, rel_path TEXT, file_size INT, duration_sec REAL,"
            " is_video INT, has_video INT, media_title TEXT);"
            "CREATE TABLE transcripts (id INTEGER PRIMARY KEY, abs_path TEXT, rel_path TEXT,"
            " media_dir TEXT, complete INT);"
            "CREATE TABLE segments (id INTEGER PRIMARY KEY, transcript_id INT, start_sec REAL,"
            " end_sec REAL, speaker TEXT, text TEXT);"
            "CREATE VIRTUAL TABLE segments_fts USING fts5(text, content='segments', content_rowid='id',"
            " tokenize='unicode61 remove_diacritics 2');"
        )

    def add(self, path, *, size=1000, length=480.35, video=True, said=(), speaker="SPEAKER_00"):
        root, _, rel = path.rpartition("/")
        self.conn.execute(
            "INSERT INTO media_files VALUES (?,?,?,?,?,?,?)",
            (path, rel, size, length, int(video), int(video), None),
        )
        if said:
            tid = self.conn.execute(
                "INSERT INTO transcripts (abs_path, rel_path, media_dir, complete) "
                "VALUES (?, ?, ?, 1)",
                (path, rel, root),
            ).lastrowid
            for start, text in said:
                sid = self.conn.execute(
                    "INSERT INTO segments (transcript_id, start_sec, end_sec, speaker, text)"
                    " VALUES (?,?,?,?,?)",
                    (tid, start, start + 5, speaker, text),
                ).lastrowid
                self.conn.execute(
                    "INSERT INTO segments_fts (rowid, text) VALUES (?, ?)", (sid, text)
                )
        return path

    def connection(self):
        @contextlib.contextmanager
        def c():
            yield self.conn

        return c()


#: Two stretches of speech — one video's, another episode's
_SAID = [
    (40, "well let us simply watch closely how clever people build themselves a wooden house"),
    (46, "today the builder uses his big digger to raise one whole wall from timber and stone"),
    (52, "meanwhile our cat sits right beside it and watches everything very carefully indeed"),
]
_OTHER = [
    (
        40,
        "this episode explains cheese plus why certain kinds smell so terribly strong after ripening",
    ),
    (46, "a friendly neighbour drops by carrying a basket full of fresh brown eggs for breakfast"),
    (52, "everybody laughs loudly while a playful dog runs off across that sunny garden lawn"),
]


# ══════════════════════════════════════════════════════════════════════════
#  The plan
# ══════════════════════════════════════════════════════════════════════════


class TestThePlan:
    def test_the_answer_of_the_language_model(self):
        text = (
            '{"summary": "Alice climbs onto a table at a party", "said_words": ["Crimson Harbor Lantern"],'
            ' "said_meaning": "Alice dances and shouts", "seen": "a woman climbing onto a table at a party",'
            ' "heard": "shouting", "people": ["Alice"], "file_hints": ["-WA"], "media": "video", "language": "German"}'
        )
        p = planning.parse_plan(text, "q")
        assert p["said_words"] == ["crimson harbor lantern"]  # lowercase, as transcribed
        assert p["seen"].startswith("a woman") and p["heard"] == "shouting"
        assert p["people"] == ["Alice"] and p["file_hints"] == ["-WA"] and p["media"] == "video"
        assert p["language"] == "German" and p["media_stated"] is False  # the grounding decides

    def test_a_useless_answer_is_none(self):
        assert planning.parse_plan("no idea", "q") is None
        assert planning.parse_plan('{"summary": "x"}', "q") is None
        assert planning.parse_plan('{"seen": "fire", "media": "tv"}', "q")["media"] == "any"

    def test_asking_for_the_duplicates_needs_no_search(self):
        # "which videos do I have twice" — nothing to search for, and still a plan
        p = planning.parse_plan(
            '{"summary": "x", "asks_for": "duplicates", "file_hints": ["Archive"]}', "q"
        )
        assert p["asks_for"] == "duplicates" and p["file_hints"] == ["Archive"]
        assert (
            planning.parse_plan('{"seen": "fire", "asks_for": "nonsense"}', "q")["asks_for"]
            == "moments"
        )
        assert planning.fallback_plan("a dog barks")["asks_for"] == "moments"
        assert "asks_for" in planning.PLAN_SCHEMA["required"]

    def test_asking_for_conversations_needs_no_search(self):
        # "a long talk with Alice and Bob in long sentences" — whole
        # conversations, not a moment; nothing to search for, and still a plan
        p = planning.parse_plan(
            '{"summary": "x", "people": ["Alice", "Bob"], "asks_for": "conversations",'
            ' "conversation": {"min_minutes": 30, "max_minutes": 0, "kind": "detailed",'
            ' "sentences": "longer"}}',
            "q",
        )
        assert p["asks_for"] == "conversations" and p["people"] == ["Alice", "Bob"]
        assert p["conversation"] == {
            "min_minutes": 30.0,
            "max_minutes": 0,
            "kind": "detailed",
            "sentences": "longer",
        }
        bare = planning.parse_plan('{"asks_for": "conversations"}', "q")
        assert (
            bare["asks_for"] == "conversations" and bare["conversation"] == planning.NO_CONVERSATION
        )

    def test_a_talk_with_people_is_planned_as_conversations(self, monkeypatch):
        """The language model's plan, grounded: the names as the voice
        library writes them, and the filters the Conversations tab opens
        with — the schema it answered to carries the conversation object."""
        from whisper_tm.core import llm_client

        seen = {}

        def chat(messages, **kw):
            seen.update(kw)
            return (
                '{"summary": "a long talk", "said_words": [], "said_meaning": "", "seen": "", "heard": "",'
                ' "people": ["alice", "Bob"], "file_hints": [], "media": "video", "language": "English",'
                ' "asks_for": "conversations", "conversation": {"min_minutes": 30, "max_minutes": 0,'
                ' "kind": "any", "sentences": "longer"}}'
            ), {}

        monkeypatch.setattr(llm_client, "chat", chat)
        monkeypatch.setattr(planning, "library_people", lambda: ["Alice", "Bob"])
        plan, source = planning.make_plan("a long talk with alice and bob in long sentences")
        assert source == "llm" and plan["asks_for"] == "conversations"
        assert plan["people"] == ["Alice", "Bob"] and not plan["notes"]
        assert planning.conversation_filters(plan) == {
            "people": ["Alice", "Bob"],
            "match": "all",
            "min_sec": 1800,
            "sentences": "longer",
        }
        assert seen["response_schema"] is planning.PLAN_SCHEMA and seen["max_tokens"] >= 800

    def test_a_bad_conversation_object_is_any(self):
        """The model may leave it out, swap the lengths or invent a kind —
        each field on its own falls back, the plan stays."""
        none = planning.NO_CONVERSATION
        bad = planning.parse_plan(
            '{"seen": "fire", "conversation": {"min_minutes": "long", "max_minutes": -5,'
            ' "kind": "chatty", "sentences": 3}}',
            "q",
        )
        assert bad["conversation"] == none and bad["asks_for"] == "moments"
        assert (
            planning.parse_plan('{"seen": "fire", "conversation": [1, 2]}', "q")["conversation"]
            == none
        )
        assert planning.parse_plan('{"seen": "fire"}', "q")["conversation"] == none
        swapped = planning.parse_plan(
            '{"asks_for": "conversations", "conversation": {"min_minutes": 90,'
            ' "max_minutes": 30, "kind": "LIVELY"}}',
            "q",
        )["conversation"]
        assert (swapped["min_minutes"], swapped["max_minutes"], swapped["kind"]) == (
            30.0,
            90.0,
            "lively",
        )
        # json reads 1e400 as infinity: no length at all
        endless = planning.parse_plan(
            '{"asks_for": "conversations", "conversation": {"min_minutes": 1e400}}', "q"
        )
        assert endless["conversation"]["min_minutes"] == 0
        # Without the language model nothing tells a talk with Alice from a
        # moment Alice is in: always moments
        fallback = planning.fallback_plan("a long talk with Alice")
        assert fallback["asks_for"] == "moments" and fallback["conversation"] == none

    def test_the_schema_is_strict_down_to_the_conversation(self):
        """llm_client sends PLAN_SCHEMA as a strict json_schema: an object
        needs every property required and no others, or the endpoint
        refuses the request."""

        def check(schema, path="plan"):
            if schema.get("type") == "object":
                assert schema.get("additionalProperties") is False, path
                assert set(schema["required"]) == set(schema["properties"]), path
                for k, v in schema["properties"].items():
                    check(v, f"{path}.{k}")
            elif schema.get("type") == "array":
                check(schema["items"], path + "[]")

        check(planning.PLAN_SCHEMA)
        props = planning.PLAN_SCHEMA["properties"]
        assert props["asks_for"]["enum"] == ["moments", "conversations", "duplicates", "contained"]
        assert props["conversation"]["properties"]["kind"]["enum"] == [
            "any",
            "detailed",
            "lively",
            "interview",
            "monologue",
            "group",
        ]
        assert props["conversation"]["properties"]["sentences"]["enum"] == [
            "any",
            "longer",
            "shorter",
        ]
        assert '"conversations" when the user wants whole conversations' in planning.PLAN_SYSTEM
        assert '- "conversation":' in planning.PLAN_SYSTEM

    def test_the_conversations_tabs_filters(self):
        """A plan that asks for conversations as /api/conversations'
        parameters — what the Conversations tab opens with (find.js)."""
        p = _plan(
            asks_for="conversations",
            people=["Alice", "Bob"],
            conversation={
                "min_minutes": 45,
                "max_minutes": 75,
                "kind": "lively",
                "sentences": "longer",
            },
        )
        assert planning.conversation_filters(p) == {
            "people": ["Alice", "Bob"],
            "match": "all",
            "min_sec": 2700,
            "max_sec": 4500,
            "kind": "lively",
            "sentences": "longer",
        }
        # With people named, "detailed" is their own measure: longer sentences
        # than usual for them — a kind is the library's, and people who always
        # talk quickly would have no detailed conversation at all
        p["conversation"] = dict(p["conversation"], kind="detailed", sentences="any")
        assert planning.conversation_filters(p) == {
            "people": ["Alice", "Bob"],
            "match": "all",
            "min_sec": 2700,
            "max_sec": 4500,
            "sentences": "longer",
        }
        nobody = _plan(
            asks_for="conversations", conversation=dict(planning.NO_CONVERSATION, kind="detailed")
        )
        assert planning.conversation_filters(nobody) == {"kind": "detailed"}
        # only what the plan sets
        assert planning.conversation_filters(_plan(asks_for="conversations")) == {}
        # one voice alone is a monologue too: the tab's two speakers are lifted
        mono = _plan(
            people=["Carol"], conversation=dict(planning.NO_CONVERSATION, kind="monologue")
        )
        assert planning.conversation_filters(mono) == {
            "people": ["Carol"],
            "match": "all",
            "kind": "monologue",
            "speakers_min": 0,
        }
        # the folder they are to be in, as a piece of the path
        assert planning.conversation_filters(_plan(file_hints=["podcasts", "-WA"])) == {
            "q": "podcasts"
        }

    def test_without_a_language_model_the_description_asks_itself(self, monkeypatch):
        from whisper_tm.rag import retrieve

        monkeypatch.setattr(retrieve, "known_names", lambda: frozenset({"alice"}))
        p = planning.fallback_plan('Alice says "crimson harbor" at night, a WhatsApp file')
        assert p["said_words"] == ["crimson harbor"] and p["people"] == ["Alice"]
        assert p["said_meaning"] and p["seen"] and p["file_hints"] == ["-WA"]

    def test_without_quotes_the_description_is_searched_word_for_word(self, monkeypatch):
        # Without a language model, the semantic or the Images & sounds index
        # (a fresh install) the word search was the only engine — and it
        # skipped everything that wasn't in quotes: zero hits
        from whisper_tm.rag import retrieve

        monkeypatch.setattr(retrieve, "known_names", lambda: frozenset())
        assert planning.fallback_plan("dog barking at night")["said_words"] == [
            "dog barking at night"
        ]

    def test_the_language_model_failing_gives_the_fallback(self, monkeypatch):
        from whisper_tm.core import llm_client

        def broken(*a, **k):
            raise llm_client.LlmError("down")

        monkeypatch.setattr(llm_client, "chat", broken)
        monkeypatch.setattr(planning, "library_people", lambda: [])
        p, source = planning.make_plan("a dog barks")
        assert source.startswith("fallback") and p["said_meaning"] == "a dog barks"

    def test_the_planner_knows_the_voice_library(self, monkeypatch):
        from whisper_tm.core import llm_client

        seen = {}

        def chat(messages, **kw):
            seen["system"] = messages[0]["content"]
            return (
                '{"summary": "s", "said_words": [], "said_meaning": "x", "seen": "", "heard": "",'
                ' "people": ["Alice"], "file_hints": [], "media": "any", "language": "English"}'
            ), {}

        monkeypatch.setattr(llm_client, "chat", chat)
        monkeypatch.setattr(planning, "library_people", lambda: ["Alice", "Bob"])
        planning.make_plan("alice dances around")
        assert "Known people (the voice library): Alice, Bob" in seen["system"]


class TestGrounding:
    """Review: "Alice dances around" became the person "Alice Dancing",
    "talking" a sound, "video" a hard filter the user never said."""

    def test_a_name_becomes_the_libraries(self):
        p = planning.ground(
            _plan(people=["Alice Dancing", "alice", "Nobody Known"]), "q", ["Alice", "Bob K"]
        )
        assert p["people"] == ["Alice"]
        assert any("“Alice Dancing” read as “Alice”" in n for n in p["notes"])
        assert any("“Nobody Known” is not in the voice library" in n for n in p["notes"])

    def test_speech_is_not_a_sound(self):
        assert planning.ground(_plan(heard="talking"), "q", [])["heard"] == ""
        assert planning.ground(_plan(heard="a man talking"), "q", [])["heard"] == ""
        assert planning.ground(_plan(heard="dog barking"), "q", [])["heard"] == "dog barking"
        assert planning.ground(_plan(heard="shouting"), "q", [])["heard"] == "shouting"

    def test_the_kind_of_recording_counts_only_when_said(self):
        said = planning.ground(_plan(media="video"), "in einem Video tanzt Alice im Garten", [])
        guessed = planning.ground(
            _plan(media="video"), "talking about a cat and showing a house", []
        )
        assert said["media_stated"] is True and guessed["media_stated"] is False
        assert any("only guessed" in n for n in guessed["notes"])
        voice = planning.ground(_plan(media="audio"), "eine Sprachnachricht von Alice", [])
        assert voice["media_stated"] is True
        assert fusion.media_factor(said, False) == fusion.WRONG_MEDIA
        assert fusion.media_factor(guessed, False) == fusion.WRONG_GUESS
        assert fusion.media_factor(guessed, True) == 1.0

    def test_conversations_keep_their_people_and_skip_the_kind_of_recording(self, monkeypatch):
        # The people are grounded as for a moment; a guessed "video" means
        # nothing to a conversation, and the run's log need not say it
        monkeypatch.setattr(planning, "conversation_names", lambda: {})
        p = planning.ground(
            _plan(asks_for="conversations", media="video", people=["alice", "Bob K."]),
            "a long talk with alice and bob",
            ["Alice", "Bob K"],
        )
        assert p["people"] == ["Alice", "Bob K"]
        assert not any("only guessed" in n for n in p["notes"])

    def test_a_conversation_keeps_the_names_typed_in_transcripts(self, monkeypatch):
        """The conversations know the names typed in transcripts too.
        Dropped, "a talk with Alice and Grace" listed every talk of Alice's
        as if Grace were in it — and a name nobody has stays, so the
        Conversations tab says it is unknown instead of showing the others'."""
        monkeypatch.setattr(
            planning, "conversation_names", lambda: {"alice": "Alice", "grace": "Grace"}
        )
        p = planning.ground(
            _plan(asks_for="conversations", people=["alice", "GRACE", "Frank"]),
            "a talk with alice, grace and frank",
            ["Alice", "Bob K"],
        )
        assert p["people"] == ["Alice", "Grace", "Frank"]
        assert any(
            "“Frank” is neither in the voice library nor typed in a transcript" in n
            for n in p["notes"]
        )
        assert not any("Grace" in n for n in p["notes"])
        assert planning.conversation_filters(p)["people"] == ["Alice", "Grace", "Frank"]
        # A moment's people stay the voice library's: the people search needs them
        m = planning.ground(_plan(people=["alice", "grace"]), "alice and grace", ["Alice", "Bob K"])
        assert m["people"] == ["Alice"]

    def test_the_typed_names_are_read_only_when_needed(self, monkeypatch):
        """The names index is built from every transcript's typed names —
        not read while the voice library knows everyone asked for."""

        def boom():
            raise AssertionError("the names index was read")

        monkeypatch.setattr(planning, "conversation_names", boom)
        p = planning.ground(
            _plan(asks_for="conversations", people=["alice", "Bob K"]), "q", ["Alice", "Bob K"]
        )
        assert p["people"] == ["Alice", "Bob K"]

    def test_the_typed_names_come_from_the_names_index(self, monkeypatch):
        from whisper_tm.rag import speakers

        monkeypatch.setattr(
            speakers,
            "index",
            lambda: {
                "names": {},
                "people": {},
                "shown": {"grace": "Grace", "dave": "Dave", "": ""},
            },
        )
        assert planning.conversation_names() == {"grace": "Grace", "dave": "Dave"}

        def broken():
            raise RuntimeError("no voice library")

        monkeypatch.setattr(speakers, "index", broken)
        assert planning.conversation_names() == {}


# ══════════════════════════════════════════════════════════════════════════
#  The searches
# ══════════════════════════════════════════════════════════════════════════


class _Rag:
    """The semantic index's passages table — what a scope reads of it."""

    TABLE = "chunks"

    def __init__(self, rows):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, transcript_key TEXT)")
        self.conn.executemany("INSERT INTO chunks VALUES (?, ?)", rows)
        self.asked = []

    def connection(self):
        @contextlib.contextmanager
        def c():
            yield self.conn

        return c()

    def search_vector(self, qvec, *, k=20, among=None):
        self.asked.append(sorted(among))
        return []


class TestTheWords:
    """Words by relevance (bm25) — they used to come in path order."""

    def _db(self):
        db = _Db()
        # many files whose paths sort first, the word only once in passing
        for n in range(8):
            db.add(
                f"/a/2025-0{n}-stream.mp4",
                said=[(10, f"yeah and then lantern or so whatever {n}")]
                + [(20 + k, "blah blah blah and even more endless talk") for k in range(6)],
            )
        db.add(
            "/z/VID-20200101-WA0001.mp4",
            said=[(5, "crimson harbor lantern is so cool crimson harbor")],
        )
        db.add("/z/VID-20200102-WA0002.mp4", said=[(9, "velvet lantern haha")])
        return db

    def test_the_best_match_leads_not_the_first_path(self):
        ctx = engines.Ctx(self._db())
        out = engines.search_words(_plan(said_words=["crimson harbor lantern"]), ctx)
        keys = [it["key"] for it in out["items"]]
        assert keys[0] == "/z/VID-20200101-WA0001.mp4"
        # the phrase, then its words alone: a transcript with "Velvet Lantern"
        assert "/z/VID-20200102-WA0002.mp4" in keys
        assert out["items"][0]["rel"] == "VID-20200101-WA0001.mp4" and out["items"][0]["dir"]

    def test_inside_a_file_name(self):
        ctx = engines.Ctx(self._db())
        rows, n = engines.fts_rows(ctx.db, '"lantern"', 60, engines.Scope("-WA", hint="-WA"))
        assert {r["key"] for r in rows} == {
            "/z/VID-20200101-WA0001.mp4",
            "/z/VID-20200102-WA0002.mp4",
        }
        assert n == 2
        # "-wa" in lower case would be every "-walk" — a piece with capitals matches as written
        sql, args = engines.hint_sql("-WA", ("t.rel_path",))
        assert "GLOB" in sql and args == ["*-WA*"]
        sql, args = engines.hint_sql("vacation_2", ("t.rel_path",))
        assert "LIKE" in sql and args == ["%vacation\\_2%"]

    def test_in_what_a_person_says(self):
        db = _Db()
        db.add("/v/talk.mp4", said=[(5, "alice says lantern")], speaker="SPEAKER_01")
        db.add("/v/talk2.mp4", said=[(5, "someone else says lantern")], speaker="SPEAKER_00")
        scope = engines.Scope(
            "Alice",
            voices=[("/v/talk.mp4", "SPEAKER_01"), ("/v/talk2.mp4", "SPEAKER_07")],
            people=["Alice"],
        )
        rows, _n = engines.fts_rows(db, '"lantern"', 60, scope)
        assert [r["key"] for r in rows] == ["/v/talk.mp4"]  # her label, not the other's
        rows, _n = engines.fts_rows(db, '"lantern"', 60)
        assert len(rows) == 2  # the temp table is gone again

    def test_the_files_search_finds_moments_itself(self, monkeypatch):
        """Review: a hint only lifted its (many thousands of) files, and
        hardly any of the best matches was one — a hint now searches inside
        its files."""
        db = self._db()
        ctx = engines.Ctx(db)
        rag = _Rag([(1, "/z/VID-20200101-WA0001.mp4"), (2, "/a/2025-01-stream.mp4")])
        monkeypatch.setattr(ctx, "rag", lambda: rag)
        monkeypatch.setattr(engines, "media_ready", lambda: "off")
        from whisper_tm.rag import service

        monkeypatch.setattr(service, "embedding_ready", lambda: (True, ""))
        monkeypatch.setattr(ctx, "query_vectors", lambda texts: [[0.1]])
        out = engines.search_files(
            _plan(said_words=["lantern"], said_meaning="alice says lantern", file_hints=["-WA"]),
            ctx,
        )
        assert out["count"] == 2 and out["paths"] == {
            "/z/VID-20200101-WA0001.mp4",
            "/z/VID-20200102-WA0002.mp4",
        }
        assert {it["key"] for it in out["items"]} == out["paths"] and all(
            it["kind"] == "files" for it in out["items"]
        )
        assert rag.asked == [[1]]  # the meaning search inside them too
        assert "searched inside them: words" in out["note"]


# ══════════════════════════════════════════════════════════════════════════
#  The fusion and the duplicates
# ══════════════════════════════════════════════════════════════════════════


class TestTheFusion:
    def test_searches_meeting_in_one_moment_lead(self):
        found = {
            "words": {"items": [_it("/a.mp4", 100, 0, "words"), _it("/b.mp4", 50, 1, "words")]},
            "meaning": {
                "items": [_it("/c.mp4", 10, 0, "meaning"), _it("/a.mp4", 130, 3, "meaning")]
            },
            "seen": {
                "items": [
                    _it("/a.mp4", 110, 5, "seen", thumb="/t/1", clip="/c/1", extra={"moment": 7})
                ]
            },
        }
        best = fusion.fuse(found, _plan())
        assert best[0]["key"] == "/a.mp4" and set(best[0]["engines"]) == {
            "words",
            "meaning",
            "seen",
        }
        assert best[0]["thumb"] == "/t/1" and best[0]["moment"] == 7
        assert {m["key"] for m in best} == {"/a.mp4", "/b.mp4", "/c.mp4"}

    def test_far_apart_in_one_recording_are_two_moments(self):
        found = {
            "words": {"items": [_it("/a.mp4", 10, 0, "words")]},
            "seen": {"items": [_it("/a.mp4", 900, 0, "seen")]},
        }
        best = fusion.fuse(found, _plan())
        assert len(best) == 2 and all(len(m["engines"]) == 1 for m in best)

    def test_a_moment_does_not_chain_on_and_on(self):
        """Two scenes 80 s apart are two moments — each gets judged; a chain
        of hits 40 s apart is cut where it would span more than 90 s."""
        apart = fusion.fuse(
            {"seen": {"items": [_it("/a.mp4", 110, 0, "seen"), _it("/a.mp4", 190, 1, "seen")]}},
            _plan(),
        )
        assert sorted(m["t"] for m in apart) == [110, 190]
        chain = fusion.fuse(
            {"seen": {"items": [_it("/a.mp4", 110 + 40 * k, k, "seen") for k in range(5)]}}, _plan()
        )
        assert sorted(m["t"] for m in chain) == [110, 230]

    def test_file_names_and_people_boost_and_the_kind_of_recording_counts(self):
        found = {
            "meaning": {
                "items": [
                    _it("/x/PTT-1.opus", 5, 0, "meaning"),
                    _it("/x/VID-WA1.mp4", 5, 1, "meaning"),
                ]
            },
            "files": {"paths": {"/x/VID-WA1.mp4"}},
            "people": {"paths": {"/x/VID-WA1.mp4"}},
        }
        best = fusion.fuse(found, _plan(media="video", media_stated=True))
        assert best[0]["key"] == "/x/VID-WA1.mp4" and best[0]["boosts"] == ["files", "people"]
        plain = fusion.fuse({"meaning": found["meaning"]}, _plan(media="video", media_stated=True))
        assert plain[0]["key"] == "/x/VID-WA1.mp4"  # the opus isn't a video
        assert fusion.fuse({"meaning": found["meaning"]}, _plan())[0]["key"] == "/x/PTT-1.opus"

    def test_a_scoped_hit_is_evidence_of_its_own(self):
        """The searches inside the remembered files bring moments the global
        lists don't have."""
        found = {
            "meaning": {"items": [_it(f"/s/{k}.mp4", 5, k, "meaning") for k in range(5)]},
            "files": {
                "items": [_it("/w/VID-WA9.mp4", 30, 0, "files", extra={"via": ["words"]})],
                "paths": {"/w/VID-WA9.mp4"},
            },
        }
        best = fusion.fuse(found, _plan())
        assert (
            best[0]["key"] == "/w/VID-WA9.mp4" and best[0]["engines"]["words"]["scope"] == "files"
        )

    def test_a_scoped_hit_is_no_second_vote_of_its_kind(self):
        """Alice's passages found by meaning globally AND inside what
        Alice says counted as two searches agreeing — and beat the one moment
        where the phrase was said word for word."""
        found = {
            "words": {"items": [_it("/s/stream.mp4", 9000, 0, "words")]},
            "meaning": {"items": [_it("/s/stream.mp4", 1500, 0, "meaning")]},
            "people": {
                "items": [_it("/s/stream.mp4", 1505, 0, "people", extra={"via": ["meaning"]})],
                "paths": {"/s/stream.mp4"},
            },
        }
        best = fusion.fuse(found, _plan())
        by_t = {round(m["t"]): m for m in best}
        assert (
            set(by_t[1500]["engines"]) == {"meaning"}
            and by_t[1500]["engines"]["meaning"]["rank"] == 0
        )
        assert by_t[9000]["score"] > by_t[1500]["score"]  # the phrase leads

    def test_at_most_three_moments_of_a_recording(self):
        found = {"words": {"items": [_it("/a.mp4", 1000 * k, k, "words") for k in range(6)]}}
        assert len(fusion.fuse(found, _plan())) == 3


class TestDuplicates:
    """The same moment in several files (one video as .mp4, .mkv and .webm
    used to take four of the best matches) is one hit with its suspected
    duplicates — found by what the files hold, never by their names."""

    def _words(self, key, t, rank):
        return _it(key, t, rank, "words", text="house")

    def test_copies_of_one_video_are_one_moment_with_their_evidence(self):
        db = _Db()
        mp4 = db.add("/a/house build.mp4", size=48_000_000, said=_SAID)
        mkv = db.add(
            "/b/youtube video #Ab12Cd34Ef5.mkv",
            size=49_000_000,
            length=480.36,
            said=[(s, t.replace("digger", "diggers")) for s, t in _SAID],
        )
        found = {
            "words": {"items": [self._words(mp4, 50, 0)]},
            "seen": {"items": [_it(mkv, 51, 0, "seen", thumb="/t/1", extra={"moment": 9})]},
        }
        best = fusion.fuse(found, _plan(), db)
        assert len(best) == 1
        m = best[0]
        assert [d["key"] for d in m["dups"]] == [mkv] and m["dups"][0][
            "why"
        ] == "the same length and words"
        assert m["dups"][0]["size"] == 49_000_000 and m["dups"][0]["length"] == 480.36
        assert (
            set(m["engines"]) == {"words", "seen"}
            and m["engines"]["seen"]["copy"] == "youtube video #Ab12Cd34Ef5.mkv"
        )
        alone = fusion.fuse({"words": found["words"]}, _plan(), db)[0]
        assert m["score"] > alone["score"]

    def test_copies_share_their_timeline(self):
        db = _Db()
        said = _SAID + [(s + 80, t) for s, t in _OTHER] + [(s + 140, t) for s, t in _SAID]
        webm = db.add("/a/v.webm", size=26_000_000, length=480.32, said=said)
        mkv = db.add("/a/v.mkv", size=49_000_000, length=480.36, said=said)
        mp4 = db.add("/b/v.mp4", size=48_000_000, length=480.35, said=said)
        found = {
            "meaning": {"items": [_it(webm, 110, 0, "meaning")]},
            "seen": {"items": [_it(mkv, 150, 0, "seen")]},
            "words": {"items": [self._words(mp4, 120, 3)]},
        }
        best = fusion.fuse(found, _plan(), db)
        assert len(best) == 1
        m = best[0]
        assert set(m["engines"]) == {"words", "meaning", "seen"}
        assert {d["key"] for d in m["dups"]} | {m["key"]} == {webm, mkv, mp4}

    def test_the_same_name_is_not_the_same_recording(self):
        db = _Db()
        a = db.add("/a/house.mp4", length=44.0, said=_SAID)
        b = db.add("/b/house.mp4", length=109.0, said=_OTHER)
        best = fusion.fuse(
            {"words": {"items": [self._words(a, 50, 0), self._words(b, 50, 1)]}}, _plan(), db
        )
        assert len(best) == 2 and not any(m["dups"] for m in best)

    def test_another_episode_at_the_same_moment_stays(self):
        db = _Db()
        a = db.add("/a/Episode 1.mp4", size=7_000_000, said=_SAID)
        b = db.add("/a/Episode 2.mp4", size=7_300_000, said=_OTHER)
        best = fusion.fuse(
            {"words": {"items": [self._words(a, 50, 0), self._words(b, 50, 1)]}}, _plan(), db
        )
        assert len(best) == 2
        c = db.add("/b/Episode 3.mp4", size=7_000_000, said=_OTHER)  # even the same size
        best = fusion.fuse(
            {"words": {"items": [self._words(a, 50, 0), self._words(c, 50, 1)]}}, _plan(), db
        )
        assert len(best) == 2

    def test_uncompressed_sound_of_one_length_is_not_one_recording(self):
        db = _Db()
        a = db.add("/game/line_02_03.wav", size=441_044, length=5.0, video=False)
        b = db.add("/game/line_07_01.wav", size=441_044, length=5.0, video=False)
        best = fusion.fuse(
            {"heard": {"items": [_it(a, 0, 0, "heard"), _it(b, 0, 1, "heard")]}}, _plan(), db
        )
        assert len(best) == 2

    def test_identical_files_need_no_words(self):
        db = _Db()
        a = db.add("/a/VID-1.mp4", size=256_000_000, length=520.0)
        b = db.add("/backup/VID-1 (1).mp4", size=256_000_000, length=520.0)
        best = fusion.fuse(
            {"seen": {"items": [_it(a, 30, 0, "seen"), _it(b, 31, 1, "seen")]}}, _plan(), db
        )
        assert len(best) == 1 and best[0]["dups"][0]["why"] == "the same size and length"

    def test_a_passage_cut_out_of_a_longer_recording(self):
        db = _Db()
        stream = db.add("/s/stream.mp4", length=11000.0, said=[(s + 7200, t) for s, t in _SAID])
        clip = db.add("/c/clip.mp4", length=58.0, said=_SAID)
        best = fusion.fuse(
            {
                "words": {"items": [self._words(stream, 7250, 0), self._words(clip, 50, 1)]},
                "seen": {"items": [_it(clip, 50, 0, "seen")]},
            },
            _plan(),
            db,
        )
        assert len(best) == 1 and best[0]["dups"][0]["why"] == "the same words"
        assert best[0]["key"] == clip and set(best[0]["engines"]) == {"words", "seen"}

    def test_the_same_sound_folds_what_words_cannot(self, monkeypatch):
        """Music without words, lengths the scan never read: the audio
        analysis' fingerprints said "the same recording"."""
        db = _Db()
        a = db.add("/a/song.mp4", size=10, length=0.0)
        b = db.add("/b/song (live).webm", size=20, length=0.0)
        same = {frozenset((a, b)): ("same", a, 0.0)}
        monkeypatch.setattr(copies, "sound_relation", lambda x, y: same.get(frozenset((x, y))))
        best = fusion.fuse(
            {"heard": {"items": [_it(a, 30, 0, "heard"), _it(b, 31, 1, "heard")]}}, _plan(), db
        )
        assert len(best) == 1 and best[0]["dups"][0]["why"] == "the same sound"
        monkeypatch.setattr(copies, "sound_relation", lambda x, y: None)
        assert (
            len(
                fusion.fuse(
                    {"heard": {"items": [_it(a, 30, 0, "heard"), _it(b, 31, 1, "heard")]}},
                    _plan(),
                    db,
                )
            )
            == 2
        )

    def test_a_passage_found_inside_a_longer_recording_by_its_sound(self, monkeypatch):
        db = _Db()
        stream = db.add("/s/stream.mp4", length=11000.0)
        clip = db.add("/c/clip.mp4", length=58.0)
        rel = {frozenset((stream, clip)): ("contains", stream, 7200.0)}
        monkeypatch.setattr(copies, "sound_relation", lambda x, y: rel.get(frozenset((x, y))))
        best = fusion.fuse(
            {"heard": {"items": [_it(stream, 7251, 0, "heard"), _it(clip, 50, 1, "heard")]}},
            _plan(),
            db,
        )
        assert len(best) == 1 and best[0]["dups"][0]["why"] == "the same passage (its sound)"
        # Another place of the stream is another moment
        best = fusion.fuse(
            {"heard": {"items": [_it(stream, 3000, 0, "heard"), _it(clip, 50, 1, "heard")]}},
            _plan(),
            db,
        )
        assert len(best) == 2

    def test_two_moments_of_one_recording_are_two(self):
        db = _Db()
        a = db.add("/a/x.mp4", said=_SAID + [(s + 300, t) for s, t in _SAID])
        best = fusion.fuse(
            {"words": {"items": [self._words(a, 50, 0), self._words(a, 350, 1)]}}, _plan(), db
        )
        assert len(best) == 2 and not any(m["dups"] for m in best)

    def test_at_most_three_moments_counting_the_copies(self):
        db = _Db()
        a = db.add(
            "/a/x.mp4",
            size=5,
            length=900.0,
            said=[(k * 200 + 40, f"part {k} " + _SAID[k % 3][1]) for k in range(4)],
        )
        b = db.add("/b/x.mkv", size=5, length=900.0)
        items = [
            self._words(a, 50, 0),
            self._words(b, 50, 1),
            self._words(a, 250, 2),
            self._words(b, 450, 3),
            self._words(a, 650, 4),
            self._words(b, 850, 5),
        ]
        best = fusion.fuse({"words": {"items": items}}, _plan(), db)
        assert len(best) == 3 and best[0]["dups"]

    def test_a_tile_keeps_its_order_and_counts_what_it_folded(self):
        db = _Db()
        a = db.add("/a/v.mp4", said=_SAID)
        b = db.add("/b/v.webm", length=480.32, said=_SAID)
        c = db.add("/c/w.mp4", length=100.0, said=_OTHER)
        items = [self._words(a, 50, 0), self._words(c, 50, 1), self._words(b, 50, 2)]
        shown, folded = fusion.fold_engine(items, db)
        assert [it["key"] for it in shown] == [a, c] and folded == 1
        assert shown[0]["dups"][0]["key"] == b and "_item" not in shown[0]["dups"][0]

    def test_a_flat_picture_says_nothing(self, tmp_path):
        from PIL import Image

        black, grad = tmp_path / "black.jpg", tmp_path / "grad.jpg"
        Image.new("RGB", (64, 36)).save(black)
        img = Image.new("L", (64, 36))
        img.putdata([(x * 4 + y * 3) % 256 for y in range(36) for x in range(64)])
        img.save(grad)
        assert copies.picture_hash(black) is None
        h = copies.picture_hash(grad)
        assert isinstance(h, int) and copies.picture_hash(grad) == h
        assert copies.picture_hash(tmp_path / "missing.jpg") is None


# ══════════════════════════════════════════════════════════════════════════
#  The judge
# ══════════════════════════════════════════════════════════════════════════


class TestTheJudge:
    @pytest.mark.parametrize(
        "text,want",
        [
            ("8 — Alice shouts it at night", (8.0, "Alice shouts it at night")),
            ("3: only the name is mentioned", (3.0, "only the name is mentioned")),
            ("Score 7 - a table, party", (7.0, "a table, party")),
            ("no digit", (None, "no digit")),
        ],
    )
    def test_a_line(self, text, want):
        assert judge.parse_verdict(text) == want

    def test_the_digits_probabilities_make_it_continuous(self):
        import math

        lp = [
            ("8", math.log(0.6)),
            ("9", math.log(0.3)),
            ("7", math.log(0.1)),
            ("—", math.log(0.001)),
        ]
        v, reason = judge.parse_verdict("8 — close", lp)
        assert v == pytest.approx(8.2, abs=0.01) and reason == "close"
        # a first token that is no digit: the line decides
        assert judge.parse_verdict("8 — close", [("**", -0.1)])[0] == 8.0

    def test_the_facts_it_is_told(self):
        """Review: a stream got a 9 for being "a WhatsApp file (WA in
        the title)" — now the program says whether it is."""
        m = {
            "key": "/v/2024-03-01 stream.mp4",
            "name": "2024-03-01 stream.mp4",
            "video": True,
            "t": 75.0,
            "dups": [{"key": "/w/VID-20200101-WA0001.mp4"}],
            "engines": {},
        }
        plan = _plan(
            file_hints=["-WA", "vacation"],
            people=["Alice", "Bob"],
            media="audio",
            media_stated=True,
        )
        facts, failed = judge.facts_for(
            m,
            plan,
            people_paths={"Alice": {"/v/2024-03-01 stream.mp4"}},
            lengths={"/v/2024-03-01 stream.mp4": 9000.0},
        )
        text = "\n".join(facts)
        assert "contains “-WA”: yes (VID-20200101-WA0001.mp4)" in text  # its copy has it
        assert "contains “vacation”: no" in text and "Alice speaks in this recording" in text
        assert "Bob's voice is not recognized" in text and "2:30:00 long" in text
        assert failed == ["file name without “vacation”", "not a sound recording"]
        assert judge.capped(9.0, failed) == judge.CAP_FAILED and judge.capped(9.0, []) == 9.0
        prompt = judge.judge_text(
            "q", plan, dict(m, engines={"words": {"rank": 0, "scope": "files"}}), "", facts
        )
        assert "FACT: the file name contains" in prompt
        assert "said word for word (rank 1 inside the files named like that)" in prompt

    def test_rejects_go_last(self):
        ms = [
            dict(key="a", score=5, verdict=2.0),
            dict(key="b", score=4, verdict=None),
            dict(key="c", score=3, verdict=8.5),
            dict(key="d", score=9, verdict=None),
            dict(key="e", score=1, verdict=4.0),
        ]
        assert [m["key"] for m in judge.ordered(ms)] == ["c", "e", "d", "b", "a"]

    def test_what_is_judged_next(self):
        """A recording's further moments count less once one of it was
        judged — little when it may be it, more when it was rejected (a
        stream's early moment rejected, a much later one was the match)."""
        a1, a2 = (
            {"key": "/a", "rec": "/a", "score": 24.3},
            {"key": "/a", "rec": "/a", "score": 10.1},
        )
        b, c = {"key": "/b", "rec": "/b", "score": 9.0}, {"key": "/c", "rec": "/c", "score": 7.0}
        best = [a1, b, a2, c]
        done = set()
        assert judge.next_to_judge(best, done) is a1
        done.add(id(a1))
        a1["verdict"] = 3.3  # rejected: its next moment stays in the race
        assert judge.next_to_judge(best, done) is b  # 9.0 > 10.1 × 0.8
        done.add(id(b))
        b["verdict"] = 1.0
        assert judge.next_to_judge(best, done) is a2  # 8.08 > 7.0
        a1["verdict"] = 8.0  # had it been good: the recording is shown
        assert judge.next_to_judge(best, done) is c  # 10.1 × 0.3 < 7.0
        done |= {id(a2), id(c)}
        assert judge.next_to_judge(best, done) is None


# ══════════════════════════════════════════════════════════════════════════
#  A run
# ══════════════════════════════════════════════════════════════════════════


class TestARun:
    @pytest.fixture
    def fakes(self, monkeypatch):
        plan = _plan(
            said_words=["crimson harbor"],
            said_meaning="alice says it",
            seen="a table at a party",
            file_hints=["-WA"],
        )
        monkeypatch.setattr(planning, "make_plan", lambda q: (plan, "llm"))
        calls = {"meaning": []}

        def meaning(text, ctx, rerank=True):
            calls["meaning"].append((text, rerank))
            return {"items": [_it("/v/other.mp4", 9, 0, "meaning")], "count": 1, "query": text}

        monkeypatch.setattr(engines, "search_meaning", meaning)
        monkeypatch.setattr(
            engines,
            "search_words",
            lambda p, ctx: {
                "items": [
                    _it("/v/VID-WA1.mp4", 75, 0, "words"),
                    _it("/v/other.mp4", 5, 1, "words"),
                ],
                "count": 2,
                "query": "x",
            },
        )
        monkeypatch.setattr(
            engines,
            "search_seen",
            lambda p, ctx: {"items": [_it("/v/VID-WA1.mp4", 80, 4, "seen")], "count": 1},
        )
        monkeypatch.setattr(
            engines,
            "search_heard",
            lambda p, ctx: (_ for _ in ()).throw(RuntimeError("service down")),
        )
        monkeypatch.setattr(
            engines,
            "search_files",
            lambda p, ctx: {"items": [], "count": 1, "paths": {"/v/VID-WA1.mp4"}},
        )
        monkeypatch.setattr(engines, "search_people", lambda p, ctx: {"skip": "nobody named"})
        judged = []

        def judge_(q, p, m, said, facts, vision):
            judged.append((m["key"], facts))
            return (9.0, "it is") if "WA1" in m["key"] else (2.0, "no")

        monkeypatch.setattr(judge, "judge", judge_)
        monkeypatch.setattr(agent, "_precut", lambda best, db, stop_flag=None: None)
        from whisper_tm.media import llm as mllm

        monkeypatch.setattr(mllm, "vision_available", lambda: False)
        return {"judged": judged, "calls": calls}

    def _wait(self, run):
        for _ in range(400):
            if run.state["ended"]:
                return run.snapshot()
            time.sleep(0.01)
        raise AssertionError("the run did not end")

    def test_every_search_reports_and_the_best_are_judged(self, fakes):
        s = self._wait(agent.start("Alice on a table at a party, says crimson harbor, WhatsApp"))
        e = s["engines"]
        assert e["words"]["status"] == "done" and e["words"]["count"] == 2
        assert e["people"]["status"] == "skipped" and e["heard"]["status"] == "failed"
        assert "service down" in e["heard"]["error"] and "paths" not in e["files"]
        assert s["status"] == "done" and s["judged"] == s["judge_total"] == 2
        assert s["best"][0]["key"] == "/v/VID-WA1.mp4" and s["best"][0]["verdict"] == 9.0
        assert s["best"][0]["boosts"] == ["files"] and fakes["judged"][0][0] == "/v/VID-WA1.mp4"
        assert any("FACT: the file name contains “-WA”: yes" in f for f in fakes["judged"][0][1])
        # the description by meaning before the plan — quick, without the reranker —
        # then the plan's sentence, reranked
        assert fakes["calls"]["meaning"][0] == (
            "Alice on a table at a party, says crimson harbor, WhatsApp",
            False,
        )
        assert ("alice says it", True) in fakes["calls"]["meaning"]

    def test_asking_for_the_duplicates_ends_with_the_plan(self, fakes, monkeypatch):
        plan = _plan(asks_for="contained", file_hints=["Archive"])
        monkeypatch.setattr(planning, "make_plan", lambda q: (plan, "llm"))
        s = self._wait(agent.start("which videos contain other videos in Archive"))
        assert (
            s["status"] == "duplicates" and s["best"] == [] and s["plan"]["asks_for"] == "contained"
        )
        assert {e["status"] for e in s["engines"].values()} == {"skipped"}
        assert fakes["judged"] == []

    def test_asking_for_conversations_ends_with_the_plan(self, fakes, monkeypatch):
        """Whole conversations are counted, not searched: no engine runs,
        and the run keeps the Conversations tab's filters — the page polls
        them from GET /api/find/<id> (find.js opens the tab with them)."""
        from whisper_tm.api.find import find_bp

        plan = _plan(
            asks_for="conversations",
            people=["Alice", "Bob"],
            notes=["person “Alice Smiles” read as “Alice” (the voice library's name)"],
            conversation={
                "min_minutes": 30,
                "max_minutes": 0,
                "kind": "any",
                "sentences": "longer",
            },
        )
        monkeypatch.setattr(planning, "make_plan", lambda q: (plan, "llm"))
        s = self._wait(agent.start("a long talk of Alice Smiles and Bob in long sentences"))
        assert s["status"] == "conversations" and s["best"] == [] and s["ended"]
        assert s["conversation_filters"] == {
            "people": ["Alice", "Bob"],
            "match": "all",
            "min_sec": 1800,
            "sentences": "longer",
        }
        assert {e["status"] for e in s["engines"].values()} == {"skipped"}
        assert fakes["judged"] == []
        log = [line["text"] for line in s["log"]]
        assert any("read as “Alice”" in t for t in log) and any(
            "Conversations tab" in t for t in log
        )
        app = Flask(__name__)
        app.register_blueprint(find_bp)
        state = app.test_client().get(f"/api/find/{s['id']}").get_json()
        assert state["status"] == "conversations"
        assert state["conversation_filters"] == s["conversation_filters"]

    def test_a_conversations_run_logs_before_it_ends(self, fakes, monkeypatch):
        """The page stops asking once a run has ended (find.js): a note
        logged after that — a name nobody has — never showed."""
        plan = _plan(
            asks_for="conversations",
            people=["Frank"],
            notes=["“Frank” is neither in the voice library nor typed in a transcript"],
        )
        monkeypatch.setattr(planning, "make_plan", lambda q: (plan, "llm"))
        logged_at_end = []
        real_set = agent.Run.set

        def spy(run, **kw):
            if "ended" in kw:
                logged_at_end.append(len(run.state["log"]))
            real_set(run, **kw)

        monkeypatch.setattr(agent.Run, "set", spy)
        s = self._wait(agent.start("a long talk with Frank"))
        assert s["status"] == "conversations"
        assert logged_at_end == [len(s["log"])]
        assert any("“Frank” is neither" in line["text"] for line in s["log"])

    def test_a_moment_run_has_no_conversation_filters(self, fakes):
        s = self._wait(agent.start("a dog barks at night"))
        assert s["status"] == "done" and s["conversation_filters"] is None

    def test_the_first_hits_come_while_it_plans(self, fakes, monkeypatch):
        """Review: nothing showed before the plan was there."""
        go = __import__("threading").Event()

        def slow(q):
            go.wait(3)
            return _plan(said_meaning="x"), "llm"

        monkeypatch.setattr(planning, "make_plan", slow)
        run = agent.start("a dog barks at night")
        for _ in range(300):
            if run.state["best"]:
                break
            time.sleep(0.01)
        early = run.snapshot()
        go.set()
        assert early["plan"] is None and early["status"] == "planning"
        assert (
            early["engines"]["meaning"]["status"] == "running"
            and early["engines"]["meaning"]["items"]
        )
        assert early["best"][0]["key"] == "/v/other.mp4"
        assert any("hits while planning" in line["text"] for line in early["log"])
        self._wait(run)

    def test_one_run_at_a_time(self, fakes, monkeypatch):
        go = __import__("threading").Event()

        def slow(q):
            go.wait(2)
            return _plan(said_meaning="x"), "llm"

        monkeypatch.setattr(planning, "make_plan", slow)
        first = agent.start("the first description")
        second = agent.start("the second description")
        assert first.stop_flag.is_set() and not second.stop_flag.is_set()
        go.set()
        assert self._wait(first)["status"] == "stopped"
        assert any("a new search started" in line["text"] for line in first.state["log"])
        self._wait(second)

    def test_the_tab_and_ask_do_not_stop_each_other(self, fakes, monkeypatch):
        # Ask's find_moments and the Find tab used to stop each other's runs
        go = __import__("threading").Event()

        def slow(q):
            go.wait(2)
            return _plan(said_meaning="x"), "llm"

        monkeypatch.setattr(planning, "make_plan", slow)
        tab = agent.start("the tab's description")
        chat = agent.start("Ask's description", "chat")
        assert not tab.stop_flag.is_set() and not chat.stop_flag.is_set()
        chat2 = agent.start("Ask's next description", "chat")
        assert chat.stop_flag.is_set() and chat.state["stopping"] is True
        assert not tab.stop_flag.is_set() and not chat2.stop_flag.is_set()
        assert chat2.state["owner"] == "chat" and tab.state["owner"] == "tab"
        go.set()
        for r in (tab, chat, chat2):
            self._wait(r)

    def test_a_stopped_run_cuts_no_more_clips(self, monkeypatch):
        # The background cuts of a stopped run used to go on
        import threading

        cut, stop = [], threading.Event()
        first = threading.Event()

        def cut_clip(key, start, length, video):
            cut.append(key)
            first.set()
            stop.wait(2)

        monkeypatch.setattr(agent, "recording", lambda db, key: {"video": True})
        monkeypatch.setattr(agent, "cut_clip", cut_clip)
        agent._precut([{"key": f"/v/{k}.mp4", "t": 1.0} for k in range(5)], object(), stop)
        assert first.wait(2)
        stop.set()
        time.sleep(0.2)
        assert cut == ["/v/0.mp4"]

    def test_stop_while_searching_ends_at_once(self, fakes, monkeypatch):
        gate = __import__("threading").Event()
        monkeypatch.setattr(engines, "search_seen", lambda p, ctx: (gate.wait(3), {"items": []})[1])
        run = agent.start("something at night")
        for _ in range(300):
            if run.state["status"] == "searching":
                break
            time.sleep(0.01)
        assert agent.stop(run.id)
        s = self._wait(run)
        gate.set()
        assert s["status"] == "stopped" and s["engines"]["seen"]["status"] == "stopped"
        assert any("Stopped after" in line["text"] for line in s["log"])
        assert agent.stop(run.id) is False

    def test_unchanged_lists_are_not_sent_again(self, fakes):
        s = self._wait(agent.start("a dog barks at night"))
        run = agent.get(s["id"])
        assert run.snapshot(s["erev"])["engines_same"] is True
        assert "items" not in run.snapshot(s["erev"])["engines"]["words"]
        assert run.snapshot(s["erev"] - 1)["engines"]["words"]["items"]


class TestTheRoutes:
    @pytest.fixture
    def client(self, monkeypatch):
        from whisper_tm.api.find import find_bp

        app = Flask(__name__)
        app.register_blueprint(find_bp)
        monkeypatch.setattr(agent, "_go", lambda run: run.set(status="done", ended=time.time()))
        return app.test_client()

    def test_start_poll_and_unchanged(self, client):
        assert client.post("/api/find", json={"query": "ab"}).status_code == 400
        r = client.post("/api/find", json={"query": "a dog barks at night"}).get_json()
        rid = r["id"]
        s = client.get(f"/api/find/{rid}").get_json()
        assert s["query"] == "a dog barks at night" and set(s["engines"]) == set(agent.ENGINES)
        same = client.get(f"/api/find/{rid}?rev={s['rev']}").get_json()
        assert same == {"rev": s["rev"], "same": True}
        assert client.get(f"/api/find/{rid}?erev={s['erev']}").get_json()["engines_same"] is True
        assert client.get("/api/find/nope").status_code == 404
        assert client.post(f"/api/find/{rid}/stop").get_json() == {
            "stopping": False
        }  # already ended

    def test_the_owner_of_a_run(self, client):
        r = client.post("/api/find", json={"query": "a dog barks", "owner": "chat"}).get_json()
        assert r["state"]["owner"] == "chat"
        assert (
            client.post("/api/find", json={"query": "a dog barks"}).get_json()["state"]["owner"]
            == "tab"
        )
        assert (
            client.post("/api/find", json={"query": "a dog barks", "owner": "x"}).status_code == 400
        )

    def test_ask_about_a_run(self, client, monkeypatch):
        from whisper_tm.core import chat

        saved = []
        monkeypatch.setattr(chat, "save", lambda conv: saved.append(conv))
        rid = client.post("/api/find", json={"query": "Alice on the table"}).get_json()["id"]
        run = agent.get(rid)
        run.set(
            plan=_plan(said_words=["lantern"]),
            best=[
                {
                    "key": "/v/VID-WA1.mp4",
                    "name": "VID-WA1.mp4",
                    "rel": "v/VID-WA1.mp4",
                    "dir": "/",
                    "t": 75.0,
                    "end": None,
                    "verdict": 8.6,
                    "reason": "Alice says it",
                    "engines": {"words": {"text": "lantern"}},
                    "dups": [{"key": "/b/VID-WA1.mp4"}],
                }
            ],
        )
        # Named as the tools take it back: relative to the open folder when
        # it lies there, else by its full path — relative to the hit's own
        # root, the transcript tool looked it up in the open folder (404)
        from whisper_tm.config import config

        monkeypatch.setattr(type(config), "media_dir", property(lambda self: Path("/")))
        d = client.post(f"/api/find/{rid}/ask").get_json()
        conv = saved[0]
        assert d["conversation"] == conv["id"] and conv["attached"]["kind"] == "find"
        assert (
            conv["refs"]["S1"]["abs_path"] == "/v/VID-WA1.mp4"
            and conv["refs"]["S1"]["start"] == 75.0
        )
        assert "S1: v/VID-WA1.mp4 at 1:15 — 8.6/9 — Alice says it" in conv["attached"]["text"]
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: Path("/b")))
        client.post(f"/api/find/{rid}/ask")
        assert "S1: /v/VID-WA1.mp4 at 1:15" in saved[1]["attached"]["text"]
        assert "1 copies" in conv["attached"]["text"] and "find" in chat.ATTACHED_RULES
        assert client.post("/api/find/nope/ask").status_code == 404
        assert client.post("/api/find/nope/stop").status_code == 404  # used to be a 200

    def test_a_body_that_is_not_an_object(self, client):
        # A JSON array used to be a 500; a list as the query started a run
        assert client.post("/api/find", json=["a dog barks"]).status_code == 400
        assert client.post("/api/find", json={"query": ["a dog barks"]}).status_code == 400


class TestFrames:
    """A still for every video match — cut from the recording where Images &
    sounds has none, for the list and for the language model's check."""

    @pytest.fixture
    def video(self, tmp_path, monkeypatch):
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")
        v = tmp_path / "VID-WA1.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x120:rate=5",
                "-t",
                "4",
                str(v),
            ],
            check=True,
        )
        monkeypatch.setattr(agent, "frames_dir", lambda: tmp_path / "frames")
        return str(v)

    def test_cut_once_and_kept(self, video):
        f = agent.cut_frame(video, 2.4)
        assert f and f.name == "2.jpg" and f.stat().st_size > 0
        inode = f.stat().st_ino  # a hit renews the time, a cut replaces the file
        assert agent.cut_frame(video, 2.9).stat().st_ino == inode  # the same second: kept
        assert agent.cut_frame("/nowhere/x.mp4", 1) is None

    def test_a_video_match_without_a_still_gets_a_frame(self):
        m = fusion.fuse({"words": {"items": [_it("/v/a b.mp4", 75, 0, "words")]}}, _plan())[0]
        assert m["thumb"] == "/api/find/frame?t=75&key=%2Fv%2Fa%20b.mp4"
        audio = fusion.fuse({"words": {"items": [_it("/v/a.opus", 5, 0, "words")]}}, _plan())[0]
        assert audio["thumb"] is None

    def test_only_recordings_of_the_library_are_cut(self, video, monkeypatch):
        from whisper_tm import extensions
        from whisper_tm.api.find import find_bp

        app = Flask(__name__)
        app.register_blueprint(find_bp)
        monkeypatch.setattr(extensions, "db", object())
        monkeypatch.setattr(
            agent,
            "recording",
            lambda db, key: {"video": True, "duration": 4.0} if key == video else None,
        )
        c = app.test_client()
        r = c.get("/api/find/frame", query_string={"key": video, "t": "1"})
        assert r.status_code == 200 and r.mimetype == "image/jpeg"
        assert (
            c.get("/api/find/frame", query_string={"key": "/etc/passwd", "t": "1"}).status_code
            == 404
        )
        # review: t=inf was a 500
        assert c.get("/api/find/frame", query_string={"key": video, "t": "inf"}).status_code == 400
        assert c.get("/api/find/frame", query_string={"key": video, "t": "nan"}).status_code == 400
        # t=1e308 was a 500 (file name too long); past the end, nothing to cut
        r = c.get("/api/find/frame", query_string={"key": video, "t": "1e308"})
        assert r.status_code == 400 and r.is_json
        r = c.get("/api/find/frame", query_string={"key": video, "t": "4"})
        assert r.status_code == 404 and r.is_json


class TestClips:
    """The moment plays in the preview: a clip cut from the recording,
    whatever its format, so the spot can be played right there."""

    def test_the_window(self):
        assert agent.clip_window(100) == (96.0, 30.0)
        assert agent.clip_window(2) == (0.0, 30.0)
        assert agent.clip_window(100, 150) == (96.0, 56.0)  # a passage: all of it
        assert agent.clip_window(100, 1000) == (96.0, agent.CLIP_MAX_SEC)
        assert agent.clip_window(100, 50) == (96.0, 30.0)  # an end before it means nothing

    @pytest.fixture
    def media(self, tmp_path, monkeypatch):
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")
        v, a = tmp_path / "VID-WA1.mkv", tmp_path / "PTT-1.opus"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x120:rate=5",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440",
                "-t",
                "8",
                "-pix_fmt",
                "yuv444p",
                "-c:v",
                "libx264",
                "-c:a",
                "libopus",
                str(v),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440",
                "-t",
                "6",
                "-c:a",
                "libopus",
                str(a),
            ],
            check=True,
        )
        monkeypatch.setattr(agent, "frames_dir", lambda: tmp_path / "find" / "frames")
        return str(v), str(a)

    def test_cut_once_whatever_the_format(self, media):
        import json
        import subprocess

        v, a = media
        clip = agent.cut_clip(v, 1.0, 5.0, True)
        assert clip and clip.suffix == ".mp4" and clip.parent.name == "clips"
        probe = json.loads(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(clip)],
                capture_output=True,
                text=True,
            ).stdout
        )
        kinds = {s["codec_type"]: s for s in probe["streams"]}
        assert kinds["video"]["codec_name"] == "h264" and kinds["video"]["pix_fmt"] == "yuv420p"
        assert kinds["audio"]["codec_name"] == "aac"
        inode = clip.stat().st_ino
        assert agent.cut_clip(v, 1.0, 5.0, True).stat().st_ino == inode  # kept
        sound = agent.cut_clip(a, 0.0, 30.0, False)
        assert sound and sound.suffix == ".m4a"
        assert agent.cut_clip("/nowhere/x.mp4", 0, 5, True) is None

    def test_only_recordings_of_the_library_play(self, media, monkeypatch):
        from whisper_tm import extensions
        from whisper_tm.api.find import find_bp

        v, a = media
        db = _Db()
        db.add(v, length=8.0)
        db.add(a, length=6.0, video=False)
        app = Flask(__name__)
        app.register_blueprint(find_bp)
        monkeypatch.setattr(extensions, "db", db)
        c = app.test_client()
        r = c.get("/api/find/clip", query_string={"key": v, "t": "3"})
        assert (
            r.status_code == 200
            and r.mimetype == "video/mp4"
            and r.headers.get("Accept-Ranges") == "bytes"
        )
        r = c.get("/api/find/clip", query_string={"key": a, "t": "1", "end": "4"})
        assert r.status_code == 200 and r.mimetype == "audio/mp4"
        assert (
            c.get("/api/find/clip", query_string={"key": "/etc/passwd", "t": "1"}).status_code
            == 404
        )
        assert c.get("/api/find/clip", query_string={"key": v, "t": "x"}).status_code == 400
        assert (
            c.get("/api/find/clip", query_string={"key": v, "t": "1", "end": "inf"}).status_code
            == 400
        )
        # Past the end an empty clip used to come back with a 200 and be kept
        r = c.get("/api/find/clip", query_string={"key": v, "t": "9"})
        assert r.status_code == 404 and r.is_json
        assert c.get("/api/find/clip", query_string={"key": v, "t": "1e308"}).status_code == 400
        r = c.get("/api/find/clip", query_string={"key": v, "t": "6", "end": "500"})
        assert r.status_code == 200
        assert any(
            p.name.endswith("-2.0-6.0.mp4") for p in agent.clips_dir().iterdir()
        )  # to the end

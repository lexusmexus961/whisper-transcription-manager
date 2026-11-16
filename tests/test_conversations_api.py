"""The Conversations routes (api/conversations.py).

Every route answers in JSON and says why it can't serve: no database (501),
the statistics switched off (503 — the status still answers, so the page can
say why the tab is empty). The search reads its parameters leniently: a
value it can't read turns off only its own filter and is named in
``ignored``; garbage, SQL and paths that climb out never get a 500 or
another file's numbers. The numbers themselves are pinned in
tests/test_conversations_query.py.
"""

from __future__ import annotations

import pytest
from flask import Flask

from tests.test_conversations_query import NAMES, _blocks, _hour, _write
from whisper_tm.api.conversations import SWITCHED_OFF, conversations_bp
from whisper_tm.conversations import query, runner, store
from whisper_tm.database import Database
from whisper_tm.database.speaker import SpeakerDB
from whisper_tm.rag import speakers

ROOT = "/media/library"
A, B = "SPEAKER_00", "SPEAKER_01"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: ROOT))
    query.forget()
    speakers.forget()
    yield
    query.forget()
    speakers.forget()


@pytest.fixture
def db(tmp_path, monkeypatch):
    voices = SpeakerDB(tmp_path / "speaker.db")
    voices.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: voices)
    d = Database(tmp_path / "wx.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.extensions.db", d)
    monkeypatch.setattr("whisper_tm.api.conversations.db", d)
    yield d
    d.close_all()


@pytest.fixture
def library(db):
    """A stream of an hour (Alice, Bob; Carol for a moment), a call dated in
    its name, and a note of Alice's."""
    _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
    _write(
        db,
        f"{ROOT}/calls/call 2024-03-01.mkv",
        _blocks([{A: 150, B: 150}]),
        names={A: "Alice", B: "Bob"},
    )
    _write(db, f"{ROOT}/note.mkv", _blocks([{A: 30}]), names={A: "Alice"})
    return db


def _app():
    app = Flask(__name__)
    app.register_blueprint(conversations_bp)
    app.config["TESTING"] = False  # a real error response, not a raised exception
    return app


@pytest.fixture
def client(library):
    return _app().test_client()


def _switched_off(monkeypatch):
    values = {"enabled": False}
    monkeypatch.setattr(runner, "_cfg", lambda name, default=None: values.get(name, default))


class _FakeRunner:
    def __init__(self, going=False):
        self.going = going
        self.started = []

    def start(self, reason="", recount=False):
        self.started.append((reason, recount))
        if self.going:
            return False
        self.going = True
        return True

    def stop(self):
        was, self.going = self.going, False
        return was

    def running(self):
        return self.going

    def status(self):
        return {"running": self.going}


# ── Finding ────────────────────────────────────────────────────────────────


class TestFind:
    def test_the_whole_library_with_the_open_folder(self, client):
        d = client.get("/api/conversations").get_json()
        assert (d["total"], d["offset"], d["limit"], d["truncated"]) == (3, 0, 50, False)
        # 3400 s of the stream, 300 of the call, 30 of the note — whatever the page holds
        assert d["total_sec"] == 3730.0
        assert client.get("/api/conversations?limit=1").get_json()["total_sec"] == 3730.0
        assert d["media_dir"] == ROOT
        assert d["ignored"] == [] and d["people_unknown"] == []
        assert set(d["status"]) == {"enabled", "counted", "stale", "running"}
        assert d["status"]["counted"] == 3
        assert set(d["baseline"]) == {
            "language",
            "source",
            "conversations",
            "wps",
            "turn_rate",
            "mean_turn_sec",
        }
        first = d["results"][0]
        assert (first["filename"], first["file_dir"], first["abs_path"]) == (
            "stream.mkv",
            ROOT,
            f"{ROOT}/stream.mkv",
        )
        call = next(r for r in d["results"] if "call" in r["filename"])
        assert (call["filename"], call["date"], call["date_source"]) == (
            "calls/call 2024-03-01.mkv",
            "2024-03-01",
            "name",
        )

    def test_people_repeated_or_comma_separated(self, client):
        a = client.get("/api/conversations?people=Alice&people=Bob").get_json()
        b = client.get("/api/conversations?people=alice,BOB").get_json()
        assert a["total"] == b["total"] == 2
        assert [(r["filename"], r["whole"]) for r in a["results"]] == [
            ("stream.mkv", False),
            ("calls/call 2024-03-01.mkv", True),
        ]
        any_ = client.get("/api/conversations?people=Bob&people=Grace&match=any").get_json()
        assert (any_["total"], any_["people_unknown"]) == (2, ["Grace"])

    def test_too_many_people_are_named_once_in_ignored(self, client, library):
        crowd = [f"Guest {i}" for i in range(1, query.PEOPLE_MAX + 2)]
        labels = [f"SPEAKER_{i:02d}" for i in range(len(crowd))]
        _write(
            library,
            f"{ROOT}/crowd.mkv",
            _blocks([dict.fromkeys(labels, 20)]),
            names=dict(zip(labels, crowd)),
        )
        # Too many values, one value with too many names, and both at once
        for people in (
            crowd,
            ",".join(crowd),
            crowd[:6] + [",".join(crowd[6:])],
            crowd[:11] + [",".join(crowd[11:])] + ["Alice"],
        ):
            d = client.get(
                "/api/conversations", query_string={"people": people, "match": "any"}
            ).get_json()
            assert (d["ignored"], d["people_unknown"], d["total"]) == (["people"], [], 1), people

    def test_filters_and_sort_reach_the_query(self, client):
        d = client.get(
            "/api/conversations",
            query_string={
                "min_sec": "100",
                "max_sec": "1000",
                "kind": "detailed",
                "sentences": "longer",
                "speakers_min": "2",
                "language": "en",
                "folder": "calls",
                "date_from": "2024-03-01",
                "date_to": "2024-03-01",
                "sort": "shortest",
                "copies": "all",
            },
        ).get_json()
        assert d["ignored"] == []
        assert [r["filename"] for r in d["results"]] == ["calls/call 2024-03-01.mkv"]

    def test_a_value_it_cannot_read_turns_off_only_its_filter(self, client):
        d = client.get(
            "/api/conversations",
            query_string={
                "min_sec": "a while",
                "max_sec": "500",
                "kind": "chatty",
                "sentences": "long",
                "speakers_min": "2.5",
                "questions_min": "2",
                "min_share": "0.9",
                "match": "most",
                "date_from": "2024-02-30",
                "sort": "sideways",
                "copies": "some",
                "language": "e n!",
                "offset": "first",
                "limit": "lots",
            },
        ).get_json()
        assert sorted(d["ignored"]) == sorted(
            [
                "min_sec",
                "kind",
                "sentences",
                "speakers_min",
                "questions_min",
                "min_share",
                "match",
                "date_from",
                "sort",
                "copies",
                "language",
                "offset",
                "limit",
            ]
        )
        # max_sec still holds: the call and the note
        assert sorted(r["filename"] for r in d["results"]) == [
            "calls/call 2024-03-01.mkv",
            "note.mkv",
        ]

    def test_paging(self, client):
        d = client.get("/api/conversations?limit=1&offset=1").get_json()
        assert (d["total"], len(d["results"]), d["truncated"]) == (3, 1, True)
        assert client.get("/api/conversations?limit=0").get_json()["limit"] == 1
        assert client.get("/api/conversations?limit=9999").get_json()["limit"] == 200
        d = client.get("/api/conversations?offset=99999999999999999999").get_json()
        assert d["results"] == [] and d["total"] == 3

    @pytest.mark.parametrize(
        "value",
        [
            "'; DROP TABLE transcripts; --",
            "Bob'); DROP TABLE conv_stats;--",
            "../../etc/passwd",
            "%",
            "_",
            "\x00",
            "é" * 5000,
            "1e309",
            "nan",
            "-1",
            "[]",
            # Whole numbers beyond SQLite's 64 bits
            "1e20",
            "9223372036854775808",
        ],
    )
    def test_garbage_never_answers_500(self, client, value):
        for name in (
            "people",
            "match",
            "min_share",
            "min_sec",
            "max_sec",
            "kind",
            "sentences",
            "wps_min",
            "wps_max",
            "rate_min",
            "rate_max",
            "speakers_min",
            "speakers_max",
            "questions_min",
            "language",
            "q",
            "folder",
            "date_from",
            "date_to",
            "sort",
            "copies",
            "offset",
            "limit",
        ):
            r = client.get("/api/conversations", query_string={name: value})
            assert r.status_code == 200, (name, value, r.status_code)
            assert "results" in r.get_json()

    def test_sql_in_a_value_is_a_value(self, client, library):
        for v in ("x' OR '1'='1", "'; DROP TABLE conv_stats; --"):
            for name in ("q", "people", "folder"):
                assert (
                    client.get("/api/conversations", query_string={name: v}).get_json()["total"]
                    == 0
                ), name
            # No language code: the filter is off, and says so
            d = client.get("/api/conversations", query_string={"language": v}).get_json()
            assert d["ignored"] == ["language"] and d["total"] == 3
        with library.connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM conv_stats").fetchone()[0] == 3
            assert conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 3

    def test_a_folder_that_climbs_out_reads_nothing_of_it(self, client):
        d = client.get("/api/conversations?folder=../../etc").get_json()
        assert d["ignored"] == ["folder"] and d["total"] == 3
        assert client.get("/api/conversations?folder=/etc").get_json()["total"] == 0


# ── One recording ──────────────────────────────────────────────────────────


class TestFile:
    def test_one_recording(self, client):
        d = client.get(f"/api/conversations/file?file={ROOT}/stream.mkv").get_json()
        assert (d["abs_path"], d["filename"], d["file_dir"], d["counted"]) == (
            f"{ROOT}/stream.mkv",
            "stream.mkv",
            ROOT,
            True,
        )
        assert d["kind"] == "detailed" and d["reasons"]
        assert [s["name"] for s in d["speakers"]] == ["Alice", "Bob", "Carol"]
        # Carol's 4.4 % makes no active voice
        assert [s["active"] for s in d["speakers"]] == [True, True, False]
        assert d["unnamed"] == 0
        assert [p["people"] for p in d["parts"]] == [["Alice"], ["Alice", "Bob"], ["Carol"]]
        assert {"wps_pct", "rate_pct", "turn_pct", "questions_share"} <= set(d["summary"])
        assert d["baseline"]["source"] == "defaults"

    def test_a_path_below_the_open_folder(self, client):
        d = client.get("/api/conversations/file", query_string={"file": "stream.mkv"})
        assert d.status_code == 200 and d.get_json()["abs_path"] == f"{ROOT}/stream.mkv"

    @pytest.mark.parametrize(
        "file,status",
        [
            ("", 400),
            ("../../etc/passwd", 400),
            ("/etc/passwd", 404),
            (f"{ROOT}/missing.mkv", 404),
            ("'; DROP TABLE conv_stats; --", 404),
            ("a" * 5000, 404),
            ("\x00", 404),
        ],
    )
    def test_nothing_else_is_read(self, client, file, status):
        r = client.get("/api/conversations/file", query_string={"file": file})
        assert r.status_code == status, file
        assert "error" in r.get_json()

    def test_not_counted_yet(self, client, library, monkeypatch):
        monkeypatch.setattr(store, "enabled", lambda: False)
        _write(library, f"{ROOT}/new.mkv", _blocks([{A: 60, B: 60}]))
        d = client.get(f"/api/conversations/file?file={ROOT}/new.mkv").get_json()
        assert (d["counted"], d["summary"], d["parts"], d["kind"]) == (False, {}, [], None)


# ── A person, and everyone ─────────────────────────────────────────────────


class TestPeople:
    def test_a_person(self, client):
        d = client.get("/api/conversations/person?person=alice").get_json()
        assert (d["name"], d["known"], d["recordings"], d["conversations"]) == ("Alice", True, 3, 2)
        assert [p["name"] for p in d["partners"]] == ["Bob"]
        assert len(d["longest"]) == 3 and d["media_dir"] == ROOT

    def test_a_name_nobody_has_is_no_error(self, client):
        r = client.get("/api/conversations/person?person=Grace")
        assert r.status_code == 200
        d = r.get_json()
        assert (d["known"], d["recordings"], d["longest"]) == (False, 0, [])

    def test_no_name(self, client):
        assert client.get("/api/conversations/person").status_code == 400
        assert client.get("/api/conversations/person?person=%20%5B%5D").status_code == 400

    def test_everyone_who_takes_part(self, client):
        d = client.get("/api/conversations/people").get_json()
        assert d["people"] == [
            {"name": "Alice", "recordings": 3, "conversations": 2},
            {"name": "Bob", "recordings": 2, "conversations": 2},
        ]
        assert client.get("/api/conversations/people?min_share=nonsense").status_code == 200


# ── Status and the background run ──────────────────────────────────────────


class TestStatusAndRun:
    def test_status(self, client):
        d = client.get("/api/conversations/status").get_json()
        assert d["enabled"] is True
        assert set(d["settings"]) == {"enabled", "auto_run", "min_share"}
        assert d["counts"] == {"transcripts": 3, "counted": 3, "stale": 0}
        assert "running" in d["run"]

    def test_the_run_counts_what_is_missing(self, db, monkeypatch):
        monkeypatch.setattr(store, "enabled", lambda: False)
        _write(db, f"{ROOT}/old.mkv", _blocks([{A: 60, B: 60}]))
        monkeypatch.setattr(store, "enabled", lambda: True)
        monkeypatch.setattr(runner, "PAUSE_SEC", 0)
        client = _app().test_client()
        assert client.get("/api/conversations/status").get_json()["counts"]["stale"] == 1
        d = client.post("/api/conversations/run/start", json={}).get_json()
        assert d["started"] is True
        runner.RUNNER.join(10)
        assert runner.RUNNER.status()["done"] == 1
        query.forget()  # the counts are held for ten seconds
        assert client.get("/api/conversations/status").get_json()["counts"]["stale"] == 0
        assert client.get("/api/conversations").get_json()["total"] == 1

    def test_one_run_at_a_time_and_stop(self, client, monkeypatch):
        fake = _FakeRunner(going=True)
        monkeypatch.setattr(runner, "RUNNER", fake)
        d = client.post("/api/conversations/run/start", json={"recount": True}).get_json()
        assert d == {"started": False, "run": {"running": True}}
        assert fake.started == [("count again", True)]
        assert client.post("/api/conversations/run/stop").get_json() == {"stopped": True}
        assert client.post("/api/conversations/run/stop").get_json() == {"stopped": False}

    @pytest.mark.parametrize(
        "body", [None, "not json", [1, 2], {"recount": "false"}, {"recount": 0}]
    )
    def test_a_body_it_cannot_read_counts_what_is_missing(self, client, monkeypatch, body):
        fake = _FakeRunner()
        monkeypatch.setattr(runner, "RUNNER", fake)
        if isinstance(body, str):
            r = client.post(
                "/api/conversations/run/start", data=body, content_type="application/json"
            )
        else:
            r = client.post("/api/conversations/run/start", json=body)
        assert r.status_code == 200 and r.get_json()["started"] is True
        assert fake.started == [("manual", False)]

    def test_writes_follow_the_origin_check(self, library, monkeypatch):
        """The app's check covers every POST — this blueprint needs none of
        its own."""
        from whisper_tm.core.security import register_origin_check

        fake = _FakeRunner()
        monkeypatch.setattr(runner, "RUNNER", fake)
        app = _app()
        register_origin_check(app)
        client = app.test_client()
        r = client.post(
            "/api/conversations/run/start", json={}, headers={"Origin": "http://evil.example"}
        )
        assert r.status_code == 403 and fake.started == []
        assert client.post("/api/conversations/run/start", json={}).get_json()["started"]


# ── Refusals ───────────────────────────────────────────────────────────────

ROUTES = [
    ("GET", "/api/conversations"),
    ("GET", f"/api/conversations/file?file={ROOT}/a.mkv"),
    ("GET", "/api/conversations/person?person=Alice"),
    ("GET", "/api/conversations/people"),
]


class TestRefusals:
    @pytest.mark.parametrize(
        "method,path",
        ROUTES
        + [
            ("GET", "/api/conversations/status"),
            ("POST", "/api/conversations/run/start"),
            ("POST", "/api/conversations/run/stop"),
        ],
    )
    def test_without_a_database(self, monkeypatch, method, path):
        monkeypatch.setattr("whisper_tm.api.conversations.db", None)
        r = _app().test_client().open(path, method=method)
        assert r.status_code == 501 and "error" in r.get_json()

    @pytest.mark.parametrize("method,path", ROUTES)
    def test_switched_off(self, client, monkeypatch, method, path):
        _switched_off(monkeypatch)
        r = client.open(path, method=method)
        assert r.status_code == 503
        assert r.get_json() == {"error": SWITCHED_OFF}

    def test_the_status_answers_while_switched_off(self, client, monkeypatch):
        _switched_off(monkeypatch)
        d = client.get("/api/conversations/status").get_json()
        assert d["enabled"] is False and d["counts"]["counted"] == 3

    def test_a_broken_query_is_json(self, client, monkeypatch):
        def broken(*a, **kw):
            raise RuntimeError("the query broke")

        monkeypatch.setattr(query, "find", broken)
        r = client.get("/api/conversations")
        assert r.status_code == 500 and "the query broke" in r.get_json()["error"]


# ── An empty library ───────────────────────────────────────────────────────


def test_an_empty_library_answers_with_zeros(db):
    client = _app().test_client()
    d = client.get("/api/conversations").get_json()
    assert (d["total"], d["results"], d["status"]["counted"]) == (0, [], 0)
    assert d["baseline"]["source"] == "defaults"
    assert client.get("/api/conversations/people").get_json() == {"people": []}
    assert client.get("/api/conversations/status").get_json()["counts"] == {
        "transcripts": 0,
        "counted": 0,
        "stale": 0,
    }
    assert client.get("/api/conversations/person?person=Alice").get_json()["known"] is False


# ── Registered, and documented ─────────────────────────────────────────────


def test_registered_with_the_app_and_its_parameters_documented():
    from whisper_tm.api import register_blueprints
    from whisper_tm.api.apidocs import collect

    app = Flask(__name__)
    register_blueprints(app)
    ref = {e["path"]: e for g in collect(app)["groups"] for e in g["endpoints"]}
    find = ref["/api/conversations"]
    assert {
        "people",
        "match",
        "min_sec",
        "max_sec",
        "kind",
        "sentences",
        "sort",
        "copies",
        "folder",
        "date_from",
        "offset",
        "limit",
    } <= set(find["query_params"])
    assert ref["/api/conversations/file"]["query_params"] == ["file"]
    assert ref["/api/conversations/run/start"]["body_fields"] == ["recount"]
    assert all(ref[p]["doc"] for p in ref if p.startswith("/api/conversations"))

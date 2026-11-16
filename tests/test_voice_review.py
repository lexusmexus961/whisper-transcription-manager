"""Listening through one person's voices — every voice, to check them.

A list to listen through everything, with autoplay, to check the names —
with many filters.

What these tests pin:

* every voice that carries the name is listed — the trashed ones are not;
* the fit of a voice comes from OTHER recordings: a wrong name given to all
  windows of one recording does not vouch for itself;
* a voice nearer to another person's centroid is flagged, and "doubtful
  first" puts it on top;
* the filters and orders do what they say, one voice per recording too;
* a decision (remove the name, trash, name as someone else) is one entry
  of the history with origin "people", touches only voices still named as
  the person, and can be undone;
* the API answers the page and refuses what it can't do.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.core import voice_review as review
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _vec(seed, basis=None, spread=0.05):
    r = np.random.default_rng(seed)
    v = basis if basis is not None else r.normal(size=DIM)
    v = v + spread * r.normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


ALICE, BOB = _vec(1, spread=0), _vec(2, spread=0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    for mod in ("whisper_tm.database.speaker", "whisper_tm.api.speaker", "whisper_tm.api.voices"):
        monkeypatch.setattr(f"{mod}.speaker_db", lambda: db)
    inventory.discard()
    review._held.clear()

    def rec(name, basis, n, seed, label="SPEAKER_00", spans=True, speech=30.0):
        path = media / name
        db.write(
            str(path),
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": label,
                    "windows": i + 1,
                    "start_sec": i * 40.0,
                    "speech_sec": speech,
                    "segments": 2,
                    "parts": [(i * 40.0, i * 40.0 + speech)] if spans else None,
                    "v_separated": _vec(seed + i, basis, 0.01),
                }
                for i in range(n)
            ],
        )
        return [w["id"] for w in db.windows(str(path))]

    ids = {
        "a": rec("2024-01-05 stream.mkv", ALICE, 3, 100),
        "b": rec("2024-06-10 talk.mkv", ALICE, 2, 200, speech=12.0),
        "c": rec("2025-02-01 old.mkv", ALICE, 2, 300, spans=False),
        "k": rec("2024-03-03 bob.mkv", BOB, 3, 400),
        # Named Alice, but sounds like Bob — and only in this one recording
        "wrong": rec("2024-09-09 wrong.mkv", BOB, 2, 500, label="SPEAKER_03"),
    }
    db.assign_many(ids["a"] + ids["b"], "Alice")
    db.write_names(
        [(v, "Alice", "assignment-spread") for v in ids["c"]],
        kind="name",
        origin="assignment",
        name="Alice",
    )
    db.assign_many(ids["k"], "Bob")
    db.assign_many(ids["wrong"], "Alice")
    yield {"db": db, "ids": ids, "media": media}
    inventory.discard()
    review._held.clear()


def _page(person="Alice", **f):
    return review.page(person, f, 0, 100)


class TestTheList:
    def test_every_named_voice_but_not_the_trashed(self, world):
        d = _page()
        assert d["total"] == 3 + 2 + 2 + 2
        world["db"].move_to_trash(world["ids"]["a"][:1])
        review._held.clear()
        assert _page()["total"] == 8
        assert _page("Bob")["total"] == 3

    def test_direct_and_spread(self, world):
        d = _page()
        assert d["counts"]["direct"] == 7 and d["counts"]["spread"] == 2
        assert {v["direct"] for v in _page(source="spread")["voices"]} == {False}

    def test_fit_comes_from_other_recordings(self, world):
        v = {x["id"]: x for x in _page()["voices"]}
        right = v[world["ids"]["a"][0]]["fit"]
        wrong = v[world["ids"]["wrong"][0]]["fit"]
        # The two "wrong" windows resemble each other — but that may not count
        assert right > 0.9 and wrong < 0.5, (right, wrong)

    def test_nearer_to_someone_else_comes_first(self, world):
        d = _page()
        first = d["voices"][:2]
        assert {x["id"] for x in first} == set(world["ids"]["wrong"])
        assert all(x["other"] == "Bob" for x in first)
        assert d["counts"]["closer"] == 2
        assert {x["id"] for x in _page(closer="1")["voices"]} == set(world["ids"]["wrong"])

    def test_filters(self, world):
        assert _page(date_from="2024-06-01", date_to="2024-12-31")["total"] == 2 + 2
        assert _page(file="TALK")["total"] == 2
        assert _page(folder="2025")["total"] == 2
        assert _page(label="speaker_03")["total"] == 2
        assert _page(min_speech="20")["total"] == 3 + 2 + 2
        assert _page(max_speech="20")["total"] == 2
        assert _page(spans="without")["total"] == 2
        assert _page(fit_max="0.5")["total"] == 2

    def test_one_per_recording_and_orders(self, world):
        d = _page(per_recording="1")
        assert d["total"] == 4
        dates = [v["date"] for v in _page(sort="date_desc")["voices"]]
        assert dates == sorted(dates, reverse=True)
        speech = [v["speech"] for v in _page(sort="speech_desc")["voices"]]
        assert speech == sorted(speech, reverse=True)

    def test_older_voices_are_only_a_topic_where_they_exist(self, world):
        assert _page()["legacy"] is True
        assert _page("Bob")["legacy"] is False

    def test_an_unknown_person(self, world):
        with pytest.raises(KeyError):
            _page("Nobody")


class TestDeciding:
    def test_remove_the_name(self, world):
        db = world["db"]
        ids = world["ids"]["wrong"]
        r = review.decide("Alice", ids, "unname")
        assert r["changed"] == 2 and r["decision"]
        assert all(v not in db.names_per_vector() for v in ids)
        e = db.decisions(limit=1)[0][0]
        assert e["kind"] == "unname" and e["origin"] == "people"
        db.undo(e["id"])
        assert all(db.names_per_vector()[v] == "Alice" for v in ids)

    def test_trash_and_name_as(self, world):
        db = world["db"]
        a, w = world["ids"]["a"], world["ids"]["wrong"]
        assert review.decide("Alice", a[:1], "trash")["changed"] == 1
        assert a[0] in db.trash_ids()
        assert review.decide("Alice", w, "name", "Bob")["changed"] == 2
        assert all(db.names_per_vector()[v] == "Bob" for v in w)
        assert db.decisions(limit=1)[0][0]["origin"] == "people"

    def test_only_voices_still_named_as_the_person(self, world):
        k = world["ids"]["k"]
        r = review.decide("Alice", k + world["ids"]["b"][:1], "unname")
        assert r["changed"] == 1 and r["skipped"] == len(k)
        assert all(world["db"].names_per_vector()[v] == "Bob" for v in k)

    def test_refused(self, world):
        with pytest.raises(ValueError):
            review.decide("Alice", [1], "explode")
        with pytest.raises(ValueError):
            review.decide("Alice", [1], "name", "")
        with pytest.raises(ValueError):
            review.decide("Alice", [1], "name", "alice")


class TestApi:
    @pytest.fixture
    def client(self, world):
        from whisper_tm.api.speaker import speaker_bp
        from whisper_tm.api.voices import voices_bp

        a = Flask(__name__)
        a.register_blueprint(speaker_bp)
        a.register_blueprint(voices_bp)
        return a.test_client()

    def test_page_and_decide(self, client, world):
        d = client.get(
            "/api/voices/review", query_string={"person": "Alice", "closer": "1"}
        ).get_json()
        assert d["total"] == 2 and d["voices"][0]["url"].startswith("/api/speaker/sample/")
        assert "abs_path" not in d["voices"][0]
        ids = [v["id"] for v in d["voices"]]
        r = client.post(
            "/api/voices/review/decide",
            json={"person": "Alice", "ids": ids, "action": "name", "to": "Bob"},
        ).get_json()
        assert r["ok"] and r["changed"] == 2
        d = client.get(
            "/api/voices/review", query_string={"person": "Alice", "closer": "1"}
        ).get_json()
        assert d["total"] == 0

    def test_errors(self, client):
        assert client.get("/api/voices/review").status_code == 400
        assert (
            client.get("/api/voices/review", query_string={"person": "Nobody"}).status_code == 404
        )
        assert client.post("/api/voices/review/decide", json={"person": "Alice"}).status_code == 400
        r = client.post(
            "/api/voices/review/decide", json={"person": "Alice", "ids": [1], "action": "x"}
        )
        assert r.status_code == 400

"""The history of decisions — and an undo that doesn't damage the rest.

Lots of voices get assigned, and a bit later it turns out a speaker
should have been kept after all. Going back must be possible without
damaging the rest of the network.

What these tests pin:

* every change to a name or to the trash leaves exactly one entry, written
  with the change — whichever path made it (map, guided assignment with
  its spread, trash map, rename);
* an undo puts back exactly what the entry changed, and only where this
  entry is still the last decision on that voice and the value is still
  the one it left — a later decision is never overwritten;
* redo mirrors it; an undo and its entry cancel out;
* what was decided before the history existed is rebuilt from the times
  of the trash and of the names — click by click;
* the API offers the freed voices as a question again.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_assignment as assignment
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _vec(seed, basis=None, spread=0.05):
    r = np.random.default_rng(seed)
    v = basis if basis is not None else r.normal(size=DIM)
    v = v + spread * r.normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    for mod in ("whisper_tm.database.speaker", "whisper_tm.api.speaker", "whisper_tm.api.voices"):
        monkeypatch.setattr(f"{mod}.speaker_db", lambda: db)
    inventory.discard()
    assignment.INVENTORY.discard()

    from whisper_tm.api.speaker import speaker_bp
    from whisper_tm.api.voices import voices_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    a.register_blueprint(voices_bp)
    a.db = db
    a.media = media
    yield a
    inventory.discard()
    assignment.INVENTORY.discard()


def _voices(app, files=3, per=4):
    """``files`` recordings with one speaker and ``per`` windows each."""
    ids = []
    for f in range(files):
        basis = _vec(100 + f, spread=0)
        path = str(app.media / f"rec{f}.mkv")
        app.db.write(
            path,
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": i,
                    "start_sec": i * 30.0,
                    "speech_sec": 30.0,
                    "segments": 2,
                    "v_separated": _vec(1000 + f * 10 + i, basis),
                }
                for i in range(per)
            ],
        )
        ids += [w["id"] for w in app.db.windows(path)]
    return ids


def _entries(db):
    return db.decisions(limit=100)[0]


class TestEveryChangeIsWrittenDown:
    def test_a_naming_is_one_entry(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:4], "Alice", source="assignment")
        e = _entries(app.db)
        assert len(e) == 1 and e[0]["kind"] == "name" and e[0]["name"] == "Alice"
        assert e[0]["voices"] == 4 and e[0]["origin"] == "assignment"

    def test_discarding_and_restoring(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:4], reason="assignment")
        app.db.restore_from_trash(ids[:2])
        kinds = [(e["kind"], e["voices"]) for e in _entries(app.db)]
        assert kinds == [("restore", 2), ("trash", 4)]

    def test_nothing_changed_nothing_written(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:2], "Alice")
        app.db.assign_many(ids[:2], "alice")  # same person, other case
        app.db.move_to_trash(ids[2:4])
        app.db.move_to_trash(ids[2:4])  # discarding twice
        app.db.restore_from_trash(ids[6:8])  # never discarded
        assert [e["kind"] for e in _entries(app.db)] == ["trash", "name"]

    def test_a_single_voice_and_its_removal(self, app):
        ids = _voices(app)
        app.db.assign(ids[0], "Bob")
        app.db.assign(ids[0], None)
        assert [e["kind"] for e in _entries(app.db)] == ["unname", "name"]

    def test_the_before_state_is_kept(self, app):
        ids = _voices(app)
        app.db.assign(ids[0], "Bob")
        app.db.assign(ids[0], "Alice")
        with app.db.connection() as c:
            row = c.execute(
                "SELECT name_before, name_after FROM speaker_decision_voices WHERE decision_id = ?",
                (_entries(app.db)[0]["id"],),
            ).fetchone()
        assert tuple(row) == ("Bob", "Alice")

    def test_a_failed_write_leaves_no_entry(self, app, monkeypatch):
        ids = _voices(app)
        monkeypatch.setattr(SpeakerDB, "_JOURNAL_SQL", "INSERT INTO nowhere VALUES (1)")
        with pytest.raises(sqlite3.OperationalError):
            app.db.assign_many(ids[:2], "Alice")
        assert app.db.names_per_vector() == {}, "the change stayed without its history"
        monkeypatch.undo()
        assert _entries(app.db) == []

    def test_deleting_a_recording_takes_its_rows(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:4], "Alice")
        app.db.remove(str(app.media / "rec0.mkv"))
        with app.db.connection() as c:
            assert c.execute("SELECT COUNT(*) FROM speaker_decision_voices").fetchone()[0] == 0

    def test_the_summary_names_speakers_and_samples(self, app):
        ids = _voices(app)
        app.db.assign_many(ids, "Alice")
        s = _entries(app.db)[0]["summary"]
        assert s["speakers"] == 3 and s["recordings"] == 3
        assert s["speech_sec"] == 12 * 30.0
        assert len(s["samples"]) == 3 and s["samples"][0]["vector_id"] in ids


class TestTheGuidedClick:
    """A named group and its spread are ONE decision."""

    def _world(self, app):
        base = _vec(7, spread=0)
        for f in range(4):
            app.db.write(
                str(app.media / f"s{f}.mkv"),
                speaker_core.MODEL_BRAND,
                [
                    {
                        "label": "SPEAKER_00",
                        "windows": i,
                        "start_sec": i * 60.0,
                        "speech_sec": 60.0,
                        "segments": 3,
                        "v_separated": _vec(f * 10 + i, base, 0.02),
                    }
                    for i in range(3)
                ],
            )
        return [r[0] for r in app.db._conn().execute("SELECT id FROM speaker_vectors ORDER BY id")]

    def test_click_and_spread_in_one_entry(self, app):
        ids = self._world(app)
        r = assignment.apply(ids[:3], "Alice")
        assert r["spread"] > 0, "nothing spread — the test needs the spread"
        e = _entries(app.db)
        assert len(e) == 1 and e[0]["id"] == r["decision"]
        assert e[0]["voices"] == 3 + r["spread"] and e[0]["spread"] == r["spread"]

    def test_undo_takes_the_spread_back_too(self, app):
        ids = self._world(app)
        r = assignment.apply(ids[:3], "Alice")
        app.db.undo(r["decision"])
        assert app.db.names_per_vector() == {}

    def test_a_typed_name_in_other_case_spreads(self, app):
        ids = self._world(app)
        assignment.apply(ids[:3], "Alice")
        app.db.undo(_entries(app.db)[0]["id"])
        app.db.assign_many(ids[3:6], "Alice")  # "Alice" exists
        r = assignment.apply(ids[:3], "alice")
        assert r["spread"] > 0, "the typed spelling didn't find its centroid"


class TestUndo:
    def test_puts_back_exactly_what_changed(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:4], reason="assignment")
        app.db.move_to_trash(ids[4:6])
        r = app.db.undo(_entries(app.db)[1]["id"])
        assert r["changed"] == 4
        assert app.db.trash_ids() == set(ids[4:6])

    def test_the_trash_reason_comes_back(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:2], reason="assignment")
        app.db.restore_from_trash(ids[:2])
        app.db.undo(_entries(app.db)[0]["id"])  # undo the restore
        with app.db.connection() as c:
            reasons = {r[0] for r in c.execute("SELECT reason FROM speaker_trash")}
        assert reasons == {"assignment"}

    def test_a_replaced_name_comes_back(self, app):
        ids = _voices(app)
        app.db.assign(ids[0], "Bob")
        app.db.assign(ids[0], "Alice")
        app.db.undo(_entries(app.db)[0]["id"])
        assert app.db.names_per_vector() == {ids[0]: "Bob"}

    def test_a_later_decision_is_not_overwritten(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:4], "Alice")
        app.db.assign(ids[0], "Bob")  # later, on one voice
        first = _entries(app.db)[1]["id"]
        r = app.db.undo(first)
        assert r["changed"] == 3
        assert dict(r["left"]) == {"changed by a later decision": 1}
        assert app.db.names_per_vector() == {ids[0]: "Bob"}

    def test_a_change_outside_the_history_is_left_alone(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        with app.db.connection() as c:  # e.g. an old tool
            c.execute("DELETE FROM speaker_trash WHERE vector_id = ?", (ids[0],))
        r = app.db.undo(_entries(app.db)[0]["id"])
        assert r["changed"] == 1 and dict(r["left"]) == {"trash changed since": 1}

    def test_an_undone_entry_and_its_undo_cancel_out(self, app):
        # discard, restore, undo the restore — the discard is last again
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        app.db.restore_from_trash(ids[:2])
        e = _entries(app.db)
        app.db.undo(e[0]["id"])
        r = app.db.undo(e[1]["id"])
        assert r["changed"] == 2 and app.db.trash_ids() == set()

    def test_the_other_thing_is_not_touched(self, app):
        # A discard is undone; the name the voice got since stays.
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        app.db.assign(ids[0], "Bob")
        app.db.undo(_entries(app.db)[1]["id"])
        assert app.db.trash_ids() == set()
        assert app.db.names_per_vector() == {ids[0]: "Bob"}

    def test_twice_is_refused(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        did = _entries(app.db)[0]["id"]
        app.db.undo(did)
        with pytest.raises(ValueError, match="already undone"):
            app.db.undo(did)

    def test_nothing_possible_is_refused_and_not_marked(self, app):
        ids = _voices(app)
        app.db.assign(ids[0], "Alice")
        app.db.assign(ids[0], "Bob")
        did = _entries(app.db)[1]["id"]
        with pytest.raises(ValueError, match="nothing could be undone"):
            app.db.undo(did)
        assert not app.db.decision_row(did)["undone_by"]

    def test_bookkeeping_is_not_listed(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        app.db.undo(_entries(app.db)[0]["id"])
        e = _entries(app.db)
        assert [x["kind"] for x in e] == ["trash"] and e[0]["undone_by"]


class TestRedo:
    def test_puts_it_in_force_again(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:4], "Alice")
        did = _entries(app.db)[0]["id"]
        app.db.undo(did)
        r = app.db.redo(did)
        assert r["changed"] == 4 and len(app.db.names_per_vector()) == 4
        assert not app.db.decision_row(did)["undone_by"]

    def test_not_over_what_was_decided_meanwhile(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:4])
        did = _entries(app.db)[0]["id"]
        app.db.undo(did)
        app.db.assign(ids[0], "Bob")
        app.db.move_to_trash([ids[1]])  # decided anew since
        r = app.db.redo(did)
        assert r["changed"] == 3
        assert dict(r["left"]) == {"changed by a later decision": 1}

    def test_only_an_undone_entry(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        with pytest.raises(ValueError, match="in force"):
            app.db.redo(_entries(app.db)[0]["id"])


class TestRename:
    def test_merging_two_people_and_back(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:4], "Carol Temp 1")
        app.db.assign_many(ids[4:6], "Carol")
        r = app.db.rename_person("Carol Temp 1", "Carol")
        assert r["merged"] and r["changed"] == 4
        assert set(app.db.names_per_vector().values()) == {"Carol"}
        assert "Carol Temp 1" not in {p["name"] for p in app.db.people()}
        app.db.undo(_entries(app.db)[0]["id"])
        names = app.db.names_per_vector()
        assert [names[i] for i in ids[:6]] == ["Carol Temp 1"] * 4 + ["Carol"] * 2

    def test_an_undone_rename_leaves_no_empty_person(self, app):
        """A rename undone must not leave "Bob 2" in the People list with no
        voice behind it."""
        ids = _voices(app)
        app.db.assign_many(ids[:2], "Bob")
        app.db.rename_person("Bob", "Bob 2")
        app.db.undo(_entries(app.db)[0]["id"])
        assert [p["name"] for p in app.db.people()] == ["Bob"]
        assert [p["name"] for p in app.db.people_stats()] == ["Bob"]

    def test_an_undone_naming_takes_its_new_person_along(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:2], "Brand New")
        app.db.undo(_entries(app.db)[0]["id"])
        assert app.db.people() == []
        # …but a person who still has voices stays
        app.db.assign_many(ids[:2], "Alice")
        app.db.assign_many(ids[2:4], "Alice")
        app.db.undo(_entries(app.db)[0]["id"])
        assert [p["name"] for p in app.db.people()] == ["Alice"]

    def test_only_the_spelling(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:2], "bob")
        r = app.db.rename_person("bob", "Bob")
        assert r["spelling_only"]
        assert set(app.db.names_per_vector().values()) == {"Bob"}


class TestApi:
    def test_list_and_undo_with_a_question(self, app):
        ids = _voices(app)
        c = app.test_client()
        c.post("/api/speaker/assignment/apply", json={"vector_ids": ids[:4], "trash": True})
        d = c.get("/api/voices/history").get_json()
        e = d["entries"][0]
        assert e["kind"] == "trash" and e["voices"] == 4 and e["at"].endswith("Z")
        assert d["counts"]["decisions"] == 1
        p = c.get(f"/api/voices/history/{e['id']}").get_json()
        assert p["moves"] == 4 and not p["redo"]
        r = c.post(f"/api/voices/history/{e['id']}/undo").get_json()
        assert r["ok"] and r["changed"] == 4 and r["entry"]["undone"]
        q = r["ask"]
        assert sorted(q["vector_ids"]) == sorted(ids[:4])
        assert q["speaker"] == 1 and q["samples"][0]["file"] == "rec0.mkv"
        assert q["decided"]["kind"] == "trash"

    def test_the_trash_route_answers_with_its_decision(self, app):
        ids = _voices(app)
        r = app.test_client().post("/api/voicemap/trash", json={"ids": ids[:2]}).get_json()
        assert r["decision"] == _entries(app.db)[0]["id"]

    def test_undo_twice_is_a_conflict(self, app):
        ids = _voices(app)
        app.db.move_to_trash(ids[:2])
        did = _entries(app.db)[0]["id"]
        c = app.test_client()
        assert c.post(f"/api/voices/history/{did}/undo").status_code == 200
        r = c.post(f"/api/voices/history/{did}/undo")
        assert r.status_code == 409 and "already" in r.get_json()["error"]
        assert c.post(f"/api/voices/history/{did}/redo").get_json()["changed"] == 2

    def test_undo_and_redo_move_the_progress(self, app):
        """The Assign page's bar follows an undo (it used to stand still) —
        from a grouping already held, never one computed for the undo."""
        ids = TestTheGuidedClick()._world(app)
        did = assignment.apply(ids[:3], "Alice")["decision"]
        c = app.test_client()
        assignment.INVENTORY.discard()
        assert c.post(f"/api/voices/history/{did}/undo").get_json()["progress"] is None
        assert assignment.INVENTORY.held() is None
        c.post(f"/api/voices/history/{did}/redo")
        named = assignment.progress(assignment.INVENTORY.fetch())["share"]
        r = c.post(f"/api/voices/history/{did}/undo").get_json()
        assert r["progress"]["share"] < named
        r = c.post(f"/api/voices/history/{did}/redo").get_json()
        assert r["progress"]["share"] == named

    def test_filter_by_kind_and_name(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:2], "Alice")
        app.db.move_to_trash(ids[4:6])
        c = app.test_client()
        assert [
            e["kind"] for e in c.get("/api/voices/history?kind=trash").get_json()["entries"]
        ] == ["trash"]
        assert [e["name"] for e in c.get("/api/voices/history?q=ali").get_json()["entries"]] == [
            "Alice"
        ]

    def test_people_and_rename(self, app):
        ids = _voices(app)
        app.db.assign_many(ids[:4], "Alice")
        c = app.test_client()
        p = c.get("/api/voices/people").get_json()["people"]
        assert p[0]["name"] == "Alice" and p[0]["voices"] == 4 and p[0]["recordings"] == 1
        r = c.post("/api/voices/people/rename", json={"from": "Alice", "to": "Grace"}).get_json()
        assert r["ok"] and r["decision"]
        assert set(app.db.names_per_vector().values()) == {"Grace"}

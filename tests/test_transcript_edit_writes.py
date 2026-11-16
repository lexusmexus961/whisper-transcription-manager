"""Editing a transcript from the page: writes that cannot lose it.

What these tests guard against:

* ``/api/save_speaker_names`` and ``_save_transcript_data`` (segment edits,
  find & replace) overwrote the transcript JSON in place — a crash, a full
  disk or a NAS hiccup mid-write left it truncated.
* A failed database write was logged at debug level and answered ``ok``:
  in ``transcript_mode: database`` (no JSON at all) the edit was gone.
* The ``?file=`` form checked a full path against the allowed roots, the
  path form and ``/api/delete_transcript`` did not.
* ``/api/unskip`` rewrote the skip list without the lock ``add_to_skiplist``
  holds — a lane's entry written meanwhile was lost.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest
from flask import Flask

from whisper_tm import extensions as ext
from whisper_tm.api import transcription as api
from whisper_tm.config import config
from whisper_tm.core import helpers
from whisper_tm.database import Database


def _seg(text, start):
    return {
        "start": start,
        "end": start + 1.0,
        "text": text,
        "speaker": "SPEAKER_00",
        "start_fmt": "",
        "end_fmt": "",
    }


@pytest.fixture
def lib(tmp_path, monkeypatch):
    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(api, "db", db)
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setitem(config._data.setdefault("directories", {}), "transcript_mode", "alongside")
    monkeypatch.setitem(config._data.setdefault("security", {}), "allowed_roots", [str(media)])
    (media / "talk.mp3").write_bytes(b"x")
    data = {
        "filename": "talk.mp3",
        "complete": True,
        "segments": [_seg("Hello there.", 0.0), _seg("Second line.", 2.0)],
    }
    db.upsert_transcript("talk.mp3", data, media_dir=str(media), abs_path=str(media / "talk.mp3"))
    tp = media / ".transcripts" / "talk.mp3.json"
    tp.parent.mkdir()
    tp.write_text(json.dumps(data), encoding="utf-8")
    app = Flask(__name__)
    app.register_blueprint(api.transcription_bp)
    return app.test_client(), db, media, tp


def _broken_dump(monkeypatch):
    real = json.dump

    def dump(obj, fp, *a, **k):
        fp.write('{"filename": "talk.mp3", "segm')  # half written
        raise OSError("No space left on device")

    monkeypatch.setattr(json, "dump", dump)
    return real


class TestAtomicJson:
    def test_speaker_names_crash_leaves_the_old_file(self, lib, monkeypatch):
        client, db, media, tp = lib
        before = tp.read_text(encoding="utf-8")
        _broken_dump(monkeypatch)
        client.post(
            "/api/save_speaker_names",
            json={"filename": "talk.mp3", "speaker_names": {"SPEAKER_00": "Alice"}},
        )
        assert tp.read_text(encoding="utf-8") == before
        assert not [p for p in tp.parent.iterdir() if p.name.endswith(".tmp")]

    def test_segment_edit_crash_leaves_the_old_file(self, lib, monkeypatch):
        client, db, media, tp = lib
        before = tp.read_text(encoding="utf-8")
        _broken_dump(monkeypatch)
        client.post(
            "/api/edit_segment", json={"filename": "talk.mp3", "index": 0, "text": "Changed."}
        )
        assert tp.read_text(encoding="utf-8") == before

    def test_the_edit_lands_in_the_file(self, lib):
        client, db, media, tp = lib
        r = client.post(
            "/api/edit_segment", json={"filename": "talk.mp3", "index": 1, "text": "Edited."}
        )
        assert r.get_json()["ok"] is True
        assert json.loads(tp.read_text(encoding="utf-8"))["segments"][1]["text"] == "Edited."


class TestDatabaseFailure:
    @pytest.fixture
    def db_only(self, lib, monkeypatch):
        monkeypatch.setitem(config._data["directories"], "transcript_mode", "database")
        client, db, media, tp = lib

        def locked(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(db, "upsert_transcript", locked)
        return client

    def test_segment_edit_reports_the_failure(self, db_only):
        r = db_only.post(
            "/api/edit_segment", json={"filename": "talk.mp3", "index": 0, "text": "Changed."}
        )
        assert r.status_code == 500 and not (r.get_json() or {}).get("ok")

    def test_speaker_names_report_the_failure(self, db_only):
        r = db_only.post(
            "/api/save_speaker_names",
            json={"filename": "talk.mp3", "speaker_names": {"SPEAKER_00": "Alice"}},
        )
        assert r.status_code == 500 and not (r.get_json() or {}).get("ok")


class TestRoots:
    """A full path names a recording anywhere — like ``?file=``, it must lie
    in the allowed roots. (The URL path form cannot carry one: the server
    folds the double slash and redirects to a relative path.)"""

    @pytest.fixture
    def outside(self, tmp_path):
        other = tmp_path / "other"
        (other / ".transcripts").mkdir(parents=True)
        victim = other / ".transcripts" / "y.mp3.json"
        victim.write_text(json.dumps({"segments": [_seg("Theirs.", 0.0)]}), encoding="utf-8")
        return other, victim

    def test_delete_checks_the_roots(self, lib, outside):
        client = lib[0]
        other, victim = outside
        r = client.post("/api/delete_transcript", json={"filename": str(other / "y.mp3")})
        assert r.status_code == 403 and victim.exists()

    def test_bulk_delete_checks_the_roots(self, lib, outside):
        client = lib[0]
        other, victim = outside
        r = client.post(
            "/api/delete_all_transcripts",
            json={"confirm_token": "DELETE_ALL_CONFIRMED", "filenames": [str(other / "y.mp3")]},
        )
        assert r.status_code == 403 and victim.exists()

    def test_edit_checks_the_roots(self, lib, outside):
        client = lib[0]
        other, victim = outside
        before = victim.read_text(encoding="utf-8")
        r = client.post(
            "/api/edit_segment",
            json={"filename": str(other / "y.mp3"), "index": 0, "text": "Mine."},
        )
        assert r.status_code == 403 and victim.read_text(encoding="utf-8") == before

    def test_a_full_path_inside_the_roots_works(self, lib):
        client, db, media, tp = lib
        r = client.post(
            "/api/edit_segment",
            json={"filename": str(media / "talk.mp3"), "index": 0, "text": "Full path."},
        )
        assert r.status_code == 200


def test_unskip_waits_for_the_skiplist_lock(lib, monkeypatch):
    """Without a database the JSON file IS the skip list."""
    client, db, media, tp = lib
    monkeypatch.setattr(ext, "db", None)
    monkeypatch.setattr(api, "db", None)
    helpers.add_to_skiplist("talk.mp3", "manual", abs_path=str(media / "talk.mp3"))
    sp = helpers._skiplist_path()
    before = sp.read_text(encoding="utf-8")
    done = threading.Event()

    def unskip():
        client.post("/api/unskip", json={"filename": "talk.mp3"})
        done.set()

    with helpers._skiplist_lock:
        t = threading.Thread(target=unskip)
        t.start()
        time.sleep(0.3)
        # A lane holding the lock is mid read-modify-write: nobody else
        # may rewrite the file under it
        assert sp.read_text(encoding="utf-8") == before, (
            "unskip rewrote the skip list without the lock"
        )
    t.join(5)
    assert done.is_set()
    assert not helpers.is_skipped("talk.mp3", media_dir=str(media))

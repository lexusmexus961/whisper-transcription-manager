"""
The JSON sync (``Database.sync_from_json_dir``) may never lose what only the
database holds, must retry files it couldn't handle, and must not invent an
identity from the wrong root.
"""

import json
import os
import sqlite3
import time

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


def _tx(name: str, text: str = "hello", **extra) -> dict:
    return {
        "filename": name,
        "complete": True,
        "language": "en",
        "processing_secs": 100.0,
        "segments": [{"speaker": "S0", "start": 0.0, "end": 1.0, "text": text}],
        **extra,
    }


def _stored(db, abs_path):
    with db.connection() as conn:
        row = conn.execute(
            "SELECT json_data, settings_json FROM transcripts WHERE abs_path = ?", (abs_path,)
        ).fetchone()
    return json.loads(row["json_data"]), json.loads(row["settings_json"] or "{}")


def _write(fp, data, mtime=None):
    fp.write_text(json.dumps(data))
    if mtime is not None:
        os.utime(fp, (mtime, mtime))


@pytest.fixture
def lib(tmp_path):
    media = tmp_path / "media"
    tdir = media / ".transcripts"
    tdir.mkdir(parents=True)
    (media / "a.mp4").write_bytes(b"x")
    return media, tdir


class TestDatabaseOnlyValuesSurvive:
    """The queue adds log_file and the lane-adjusted processing time to the
    database only; a voices-only run merges `voices` into settings_json only.
    The next scan's auto-sync re-imported the (newer) JSON and erased them."""

    def test_reimport_keeps_log_file_and_cost(self, db, lib):
        media, tdir = lib
        data = _tx("a.mp4", settings={"model": "turbo"})
        _write(tdir / "a.mp4.json", data)
        ab = str(media / "a.mp4")
        # what the queue writes after the run: DB only
        db.upsert_transcript(
            "a.mp4",
            {**data, "log_file": "run.log", "processing_secs": 33.0},
            media_dir=str(media),
            abs_path=ab,
        )
        db.merge_transcript_settings(ab, {"voices": {"windows": 3}})
        db.sync_from_json_dir(tdir, media_dir=str(media))
        stored, settings = _stored(db, ab)
        assert stored["log_file"] == "run.log"
        assert stored["processing_secs"] == 33.0
        assert settings["voices"] == {"windows": 3}
        assert settings["model"] == "turbo"

    def test_changed_json_still_imports_but_carries_db_values(self, db, lib):
        media, tdir = lib
        ab = str(media / "a.mp4")
        data = _tx("a.mp4", settings={"model": "turbo"})
        db.upsert_transcript(
            "a.mp4", {**data, "log_file": "run.log"}, media_dir=str(media), abs_path=ab
        )
        db.merge_transcript_settings(ab, {"voices": {"windows": 3}})
        # The file changed on disk (an edit made elsewhere)
        _write(tdir / "a.mp4.json", _tx("a.mp4", text="edited", settings={"model": "turbo"}))
        res = db.sync_from_json_dir(tdir, media_dir=str(media))
        assert res["imported"] == 1
        stored, settings = _stored(db, ab)
        assert stored["segments"][0]["text"] == "edited"
        assert stored["log_file"] == "run.log"
        assert settings["voices"] == {"windows": 3}


class TestSyncMarker:
    def test_failed_import_is_retried(self, db, lib, monkeypatch):
        """A database error on one file must not move the marker past it."""
        media, tdir = lib
        (media / "b.mp4").write_bytes(b"x")
        now = time.time()
        _write(tdir / "a.mp4.json", _tx("a.mp4"), now - 20)
        _write(tdir / "b.mp4.json", _tx("b.mp4"), now - 10)
        real = db.upsert_transcript

        def flaky(rel, data, **kw):
            if rel == "a.mp4":
                raise sqlite3.OperationalError("database is locked")
            return real(rel, data, **kw)

        monkeypatch.setattr(db, "upsert_transcript", flaky)
        res = db.sync_from_json_dir(tdir, media_dir=str(media))
        assert res["errors"] == 1 and res["imported"] == 1
        monkeypatch.setattr(db, "upsert_transcript", real)
        res = db.sync_from_json_dir(tdir, media_dir=str(media))
        assert res["imported"] == 1, res
        assert db.get_transcript("a.mp4", media_dir=str(media)) is not None

    def test_unreadable_json_does_not_hold_marker_forever(self, db, lib):
        """Broken content is permanent until rewritten — and rewriting gives
        a new mtime, so the marker may move on."""
        media, tdir = lib
        now = time.time()
        (tdir / "a.mp4.json").write_text("{broken")
        os.utime(tdir / "a.mp4.json", (now - 20, now - 20))
        db.sync_from_json_dir(tdir, media_dir=str(media))
        res = db.sync_from_json_dir(tdir, media_dir=str(media))
        assert res["errors"] == 0


class TestIdentityOfMovedMedia:
    def test_json_in_transcripts_dir_keeps_its_location(self, db, tmp_path):
        """Auto-sync passes the OPENED folder as media_dir; the JSON's
        filename is relative to the media root of back then. If the media
        file is gone from beside the JSON, the old code computed
        <opened folder>/<filename> — /nas/series/series/f1.mp4."""
        root = tmp_path / "nas"
        tdir = root / "series" / ".transcripts"
        tdir.mkdir(parents=True)
        _write(tdir / "f1.mp4.json", _tx("series/f1.mp4"))
        db.sync_from_json_dir(tdir, media_dir=str(root / "series"))
        with db.connection() as conn:
            rows = [r[0] for r in conn.execute("SELECT abs_path FROM transcripts")]
        assert rows == [str(root / "series" / "f1.mp4")]


class TestPermanentImportFailure:
    """A JSON that parses but always fails in the database (bad segment
    types) must not hold the marker back forever — every walk would re-read
    all newer JSONs over CIFS."""

    def test_marker_moves_on_after_repeated_failures(self, db, lib, monkeypatch, caplog):
        media, tdir = lib
        (media / "b.mp4").write_bytes(b"x")
        now = time.time()
        _write(tdir / "a.mp4.json", _tx("a.mp4"), now - 20)
        _write(tdir / "b.mp4.json", _tx("b.mp4"), now - 10)
        real = db.upsert_transcript
        reads = []

        def broken_a(rel, data, **kw):
            reads.append(rel)
            if rel == "a.mp4":
                raise TypeError("bad segment types")
            return real(rel, data, **kw)

        monkeypatch.setattr(db, "upsert_transcript", broken_a)
        for _ in range(db._SYNC_MAX_ATTEMPTS):
            assert db.sync_from_json_dir(tdir, media_dir=str(media))["errors"] == 1
        assert "a.mp4.json" in caplog.text
        reads.clear()
        res = db.sync_from_json_dir(tdir, media_dir=str(media))
        assert res["errors"] == 0 and "a.mp4" not in reads

    def test_a_rewritten_file_gets_fresh_attempts(self, db, lib, monkeypatch):
        media, tdir = lib
        now = time.time()
        _write(tdir / "a.mp4.json", _tx("a.mp4"), now - 20)
        real = db.upsert_transcript
        fail = {"on": True}

        def flaky(rel, data, **kw):
            if fail["on"]:
                raise TypeError("bad segment types")
            return real(rel, data, **kw)

        monkeypatch.setattr(db, "upsert_transcript", flaky)
        for _ in range(db._SYNC_MAX_ATTEMPTS):
            db.sync_from_json_dir(tdir, media_dir=str(media))
        fail["on"] = False
        _write(tdir / "a.mp4.json", _tx("a.mp4", text="fixed"), now - 5)
        assert db.sync_from_json_dir(tdir, media_dir=str(media))["imported"] == 1

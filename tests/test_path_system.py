"""
System tests for the path model from docs/PATHS.md.

These tests hold the guarantees that made the overhaul necessary at all.
They are deliberately scenario-driven — each describes
a process that went wrong before.
"""

import pytest

from whisper_tm.core import paths
from whisper_tm.database import Database


def _tx(filename, lang="de"):
    return {
        "filename": filename,
        "language": lang,
        "model": "turbo",
        "vocal_isolation": False,
        "vocal_model": None,
        "segments": [
            {
                "speaker": "S0",
                "start": 0.0,
                "end": 1.0,
                "start_fmt": "0:00",
                "end_fmt": "0:01",
                "text": "hello",
            }
        ],
        "speakers": ["S0"],
        "total_duration": "0:01",
        "total_duration_sec": 1.0,
        "processing_time": "1s",
        "processing_secs": 1.0,
        "total_chunks": 1,
        "complete": True,
        "warning_count": 0,
        "warnings": [],
    }


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.initialize()
    return d


class TestSameNameDifferentFolder:
    """The bug that all basename workarounds shared: two files
    with the same name in different folders were considered one."""

    @pytest.fixture
    def two(self, db):
        db.upsert_transcript("a/intro.mp4", _tx("a/intro.mp4", "de"), media_dir="/nas")
        db.upsert_transcript("b/intro.mp4", _tx("b/intro.mp4", "en"), media_dir="/nas")
        return db

    def test_two_rows_remain(self, two):
        with two.connection() as c:
            n = c.execute("SELECT COUNT(*) c FROM transcripts").fetchone()["c"]
        assert n == 2

    def test_each_one_is_found_separately(self, two):
        assert two.get_transcript("a/intro.mp4", "/nas")["language"] == "de"
        assert two.get_transcript("b/intro.mp4", "/nas")["language"] == "en"

    def test_deleting_hits_only_one(self, two):
        two.delete_transcript("a/intro.mp4", media_dir="/nas")
        assert two.get_transcript("a/intro.mp4", "/nas") is None
        assert two.get_transcript("b/intro.mp4", "/nas") is not None

    def test_a_bare_basename_finds_nothing(self, two):
        """Previously that returned one of the two files."""
        assert two.get_transcript("intro.mp4", "/nas") is None


class TestRootChangeAtRuntime:
    """config.media_dir changes when the folder is switched in the frontend.
    This is exactly where the duplicates came from."""

    def test_the_same_file_from_two_roots_stays_one(self, db):
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir="/nas/video")
        db.upsert_transcript("f1.mp4", _tx("f1.mp4"), media_dir="/nas/video/series")
        with db.connection() as c:
            rows = c.execute("SELECT abs_path FROM transcripts").fetchall()
        assert len(rows) == 1
        assert rows[0]["abs_path"] == "/nas/video/series/f1.mp4"

    def test_still_findable_after_a_root_change(self, db):
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir="/nas/video")
        # Frontend switches to the subfolder
        found = db.get_transcript("f1.mp4", media_dir="/nas/video/series")
        assert found is not None

    def test_realign_realigns_rel_path(self, db):
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir="/nas/video")
        db.realign_paths("/nas/video/series")
        with db.connection() as c:
            r = c.execute("SELECT rel_path, abs_path FROM transcripts").fetchone()
        assert r["rel_path"] == "f1.mp4"
        assert r["abs_path"] == "/nas/video/series/f1.mp4"  # Identity unchanged


class TestScanRootDiffersFromMediaDir:
    """database.scan_root may lie below media_dir. Then rel_path
    refers to the scan folder, but the lookup refers to media_dir."""

    def test_the_transcript_stays_findable(self, db):
        # Scan ran under /nas/video/series, media_dir is /nas/video
        db.upsert_transcript("f1.mp4", _tx("f1.mp4"), media_dir="/nas/video/series")
        # Resolution via identity, regardless of which root is queried
        assert db.get_transcript("f1.mp4", "/nas/video/series") is not None
        assert db.get_transcript("series/f1.mp4", "/nas/video") is not None

    def test_no_duplicate_entry(self, db):
        db.upsert_transcript("f1.mp4", _tx("f1.mp4"), media_dir="/nas/video/series")
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir="/nas/video")
        with db.connection() as c:
            n = c.execute("SELECT COUNT(*) c FROM transcripts").fetchone()["c"]
        assert n == 1


class TestApiTranscriptAfterRootChange:
    """GET /api/transcript was the endpoint where the failure became visible:
    the search found the transcript, but the click ran into nothing."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from whisper_tm.config import config

        media = tmp_path / "media"
        (media / "series").mkdir(parents=True)
        config.load(None)
        config._data.setdefault("directories", {})["media"] = str(media)
        config._data.setdefault("database", {})["path"] = str(tmp_path / "api.db")
        config._resolve_directories()
        config._ensure_directories()
        from whisper_tm import extensions as ext
        from whisper_tm.app import create_app

        app = create_app()
        # api/transcription.py binds `db` at import (from ... import db). In
        # a full run an earlier test imports it before any database exists,
        # and the binding stays None — both tests below were once skipped
        # for that, silently. Hand the endpoint this test's
        # database: the resolution logic is under test here, not the wiring.
        import whisper_tm.api.transcription as endpoint

        monkeypatch.setattr(endpoint, "db", ext.db)
        assert endpoint.db, "create_app() set up no database"
        return app.test_client(), media, endpoint.db

    def test_the_transcript_is_delivered(self, client):
        c, media, _db = client
        _db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir=str(media))
        r = c.get("/api/transcript/series/f1.mp4")
        assert r.status_code == 200
        assert r.get_json()["language"] == "de"

    def test_foreign_basename_returns_404(self, client):
        """Previously /api/transcript/f1.mp4 returned the transcript from
        series/f1.mp4 — a file that nobody asked for."""
        c, media, _db = client
        _db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir=str(media))
        assert c.get("/api/transcript/f1.mp4").status_code == 404


class TestNormalFormInTheDatabase:
    """Everything that goes into the DB as identity is canonical."""

    @pytest.mark.parametrize(
        "root,rel,expected",
        [
            ("/nas/video/", "a.mp4", "/nas/video/a.mp4"),
            ("/nas//video", "a.mp4", "/nas/video/a.mp4"),
            ("/nas/video", "/a.mp4", "/nas/video/a.mp4"),
            ("C:/video", "sub\\a.mp4", "C:/video/sub/a.mp4"),
            ("/nas/video/../video", "a.mp4", "/nas/video/a.mp4"),
        ],
    )
    def test_upsert_stores_the_canonical_form(self, db, root, rel, expected):
        db.upsert_transcript(rel, _tx(rel), media_dir=root)
        with db.connection() as c:
            r = c.execute("SELECT abs_path FROM transcripts").fetchone()
        assert r["abs_path"] == expected

    def test_odd_variants_are_the_same_file(self, db):
        db.upsert_transcript("a.mp4", _tx("a.mp4", "de"), media_dir="/nas/video")
        db.upsert_transcript("a.mp4", _tx("a.mp4", "en"), media_dir="/nas//video/")
        with db.connection() as c:
            rows = c.execute("SELECT language FROM transcripts").fetchall()
        assert len(rows) == 1
        assert rows[0]["language"] == "en"


class TestSingleScanEndpoint:
    """/api/scan_entry was untested — during the overhaul a bug there went
    unnoticed at first because no assertion covered it."""

    @pytest.fixture
    def structure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from whisper_tm.config import config

        media = tmp_path / "media"
        (media / "series").mkdir(parents=True)
        (media / "series" / "f1.mp4").write_bytes(b"x")
        config.load(None)
        config._data.setdefault("directories", {})["media"] = str(media)
        config._data.setdefault("database", {})["path"] = str(tmp_path / "e.db")
        config._resolve_directories()
        config._ensure_directories()
        from whisper_tm.app import create_app

        app = create_app()
        # Via `ext` not `whisper_tm.api.main.db`: The endpoint
        # reads the registry on every call, rather than
        # freezing the value at import. A test that still reads the old copy
        # was checking something different from what the application does.
        from whisper_tm import extensions as ext

        assert ext.db, "create_app() set up no database"
        return app.test_client(), media, ext.db

    def test_a_known_file_is_delivered(self, structure):
        c, media, _db = structure
        _db.bulk_upsert_media(
            [
                {
                    "rel_path": "series/f1.mp4",
                    "abs_path": paths.abs_from("series/f1.mp4", str(media)),
                    "display_name": "f1.mp4",
                    "subfolder": "series",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 1,
                    "modified_at": 0,
                    "status": "pending",
                }
            ]
        )
        r = c.get("/api/scan_entry/series/f1.mp4")
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["file"]["name"] == "series/f1.mp4"

    def test_traversal_is_rejected(self, structure):
        c, _media, _db = structure
        # Werkzeug normalizes a leading ../ in the URL — so a
        # path that exits the root only in the middle.
        r = c.get("/api/scan_entry/series/../../../etc/passwd")
        assert r.status_code in (400, 404), r.status_code

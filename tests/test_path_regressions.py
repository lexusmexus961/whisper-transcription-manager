"""
Regression tests for the two confirmed path bugs.

Both were previously uncovered and both led to *silent* corruption:
a wrong abs_path overwrote the correct one, and realign_paths then
pulled the rel_path along with it. See docs/PATHS.md.
"""

import json
from pathlib import Path

import pytest

from whisper_tm.core import paths
from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.initialize()
    return d


def _tx(filename):
    return {
        "filename": filename,
        "language": "de",
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
                "text": "hello world",
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


def _row(db, sql="SELECT rel_path, abs_path FROM transcripts"):
    with db.connection() as c:
        return dict(c.execute(sql).fetchone())


class TestNestedTranscriptsDirectory:
    """sync_from_json_dir must not build a duplicated abs_path from a local .transcripts/ (/media/sub/sub/b.mp3)."""

    @pytest.fixture
    def structure(self, tmp_path):
        media = tmp_path / "media"
        td = media / "sub" / ".transcripts"
        td.mkdir(parents=True)
        # This is how core/helpers.py stores local transcripts: <mediafile>.json,
        # and "filename" in the JSON is media-dir relative.
        (td / "b.mp3.json").write_text(json.dumps(_tx("sub/b.mp3")), encoding="utf-8")
        return media, td

    def test_abs_path_is_not_doubled(self, db, structure):
        media, td = structure
        # Root explicitly — this is how the application calls it too (app.py, scanner).
        db.sync_from_json_dir(td, media_dir=str(media))
        r = _row(db)
        assert r["abs_path"] == paths.abs_from("sub/b.mp3", str(media))
        assert "/sub/sub/" not in r["abs_path"]

    def test_a_correct_value_is_not_overwritten(self, db, structure):
        """The upsert overwrites abs_path — a wrong value would have
        displaced a previously correct one."""
        media, td = structure
        db.upsert_transcript("sub/b.mp3", _tx("sub/b.mp3"), media_dir=str(media))
        before = _row(db)["abs_path"]
        db.sync_from_json_dir(td, media_dir=str(media))
        assert _row(db)["abs_path"] == before

    def test_realign_does_not_pull_rel_path_into_the_wrong_place(self, db, structure):
        """The corruption propagated: realign_paths believed the
        wrong abs_path and wrote rel_path = sub/sub/b.mp3."""
        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        db.realign_paths(str(media))
        assert _row(db)["rel_path"] == "sub/b.mp3"

    def test_the_central_folder_stays_relative_to_media_dir(self, db, tmp_path):
        """Counter-check: the flat main transcript folder continues to
        refer to media_dir, not to its own parent folder."""
        media = tmp_path / "media"
        media.mkdir()
        tdir = tmp_path / "transcripts"
        tdir.mkdir()
        (tdir / "sub__b.mp3.json").write_text(json.dumps(_tx("sub/b.mp3")), encoding="utf-8")
        db.sync_from_json_dir(tdir, media_dir=str(media))
        assert _row(db)["abs_path"] == paths.abs_from("sub/b.mp3", str(media))


class TestRealignLikeEscaping:
    """realign_paths ran with unescaped LIKE — with a % in the root path
    it grabbed foreign rows and wrote them a false rel_path."""

    def test_percent_in_the_root(self, db):
        db.upsert_transcript(
            "real/clip.mp4", _tx("real/clip.mp4"), media_dir="/mnt/50-somewhere-else-data"
        )
        db.realign_paths("/mnt/50%data")
        assert _row(db)["rel_path"] == "real/clip.mp4"

    def test_underscore_in_the_root(self, db):
        db.upsert_transcript("real/clip.mp4", _tx("real/clip.mp4"), media_dir="/mnt/myXdir")
        db.realign_paths("/mnt/my_dir")
        assert _row(db)["rel_path"] == "real/clip.mp4"

    def test_real_rows_are_still_realigned(self, db):
        """Escaping must not break the actual task."""
        db.upsert_transcript(
            "old_value/clip.mp4", _tx("old_value/clip.mp4"), media_dir="/mnt/50%data"
        )
        with db.connection() as c:
            c.execute("UPDATE transcripts SET rel_path = 'wrong/clip.mp4'")
        db.realign_paths("/mnt/50%data")
        assert _row(db)["rel_path"] == "old_value/clip.mp4"

    def test_a_root_with_special_characters_and_a_subfolder(self, db):
        db.upsert_transcript("a/b/c.mp4", _tx("a/b/c.mp4"), media_dir="/mnt/100%_raw")
        with db.connection() as c:
            c.execute("UPDATE transcripts SET rel_path = 'x'")
        db.realign_paths("/mnt/100%_raw")
        assert _row(db)["rel_path"] == "a/b/c.mp4"


class TestCentralTranscriptsFolderAtTheRoot:
    """A ``.transcripts/`` DIRECTLY in the media directory is the CENTRAL
    storage — it contains transcripts for files from the entire tree, not
    just for neighboring files.

    The earlier assumption "every .transcripts/ is a local neighbor folder"
    dropped the subpath and entered ``/mnt/media/<basename>`` — paths that
    don't exist. On a real library, this created thousands of rows with
    fictitious identity.
    """

    @pytest.fixture
    def structure(self, tmp_path):
        media = tmp_path / "media"
        td = media / ".transcripts"
        td.mkdir(parents=True)
        low = media / "Archive" / "Phone" / "Voice"
        low.mkdir(parents=True)
        (low / "note.aac").write_bytes(b"x")
        # This is how central storage stores it: filename is media-dir relative
        (td / "note.aac.json").write_text(
            json.dumps(_tx("Archive/Phone/Voice/note.aac")), encoding="utf-8"
        )
        return media, td

    def test_the_subpath_is_preserved(self, db, structure):
        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        r = _row(db)
        assert r["abs_path"] == paths.abs_from("Archive/Phone/Voice/note.aac", str(media))

    def test_the_recorded_path_really_exists(self, db, structure):
        """The sharpest test: the identity must point to a real file."""
        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        assert Path(_row(db)["abs_path"]).exists()

    def test_the_filename_field_beats_the_storage_location(self, db, structure):
        """What is in the JSON is the reliable statement — not where the
        JSON happens to be located."""
        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        assert _row(db)["rel_path"] == "Archive/Phone/Voice/note.aac"


class TestSyncOnlyChangedFiles:
    """sync_from_json_dir read all JSONs fresh on EVERY call — on a
    large library many seconds and one transaction per transcript,
    on every scan. The timestamp of the last sync
    now lies in the database and survives a restart."""

    @pytest.fixture
    def structure(self, tmp_path):
        media = tmp_path / "m"
        td = media / ".transcripts"
        td.mkdir(parents=True)
        for n in ("a.mp4", "b.mp4"):
            (media / n).write_bytes(b"x")
            (td / f"{n}.json").write_text(json.dumps(_tx(n)), encoding="utf-8")
        return media, td

    def test_the_second_run_skips_everything(self, db, structure):
        media, td = structure
        assert db.sync_from_json_dir(td, media_dir=str(media))["imported"] == 2
        second = db.sync_from_json_dir(td, media_dir=str(media))
        assert second["imported"] == 0
        assert second["skipped"] == 2

    def test_survives_the_restart(self, db, structure, tmp_path):
        """The earlier cache was in memory — every server restart read everything fresh."""
        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        new_value = Database(tmp_path / "t.db")
        new_value._initialized = False
        new_value.initialize()
        assert new_value.sync_from_json_dir(td, media_dir=str(media))["imported"] == 0

    def test_a_changed_file_is_detected(self, db, structure):
        import time

        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        time.sleep(1.1)  # wait for mtime resolution
        (td / "a.mp4.json").write_text(
            json.dumps(_tx("a.mp4") | {"language": "en"}), encoding="utf-8"
        )
        r = db.sync_from_json_dir(td, media_dir=str(media))
        assert r["imported"] == 1 and r["skipped"] == 1
        with db.connection() as c:
            lang = c.execute("SELECT language FROM transcripts WHERE rel_path='a.mp4'").fetchone()[
                "language"
            ]
        assert lang == "en"

    def test_a_new_file_is_detected(self, db, structure):
        media, td = structure
        db.sync_from_json_dir(td, media_dir=str(media))
        (media / "c.mp4").write_bytes(b"x")
        (td / "c.mp4.json").write_text(json.dumps(_tx("c.mp4")), encoding="utf-8")
        assert db.sync_from_json_dir(td, media_dir=str(media))["imported"] == 1

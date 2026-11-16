"""
The schema: the first release's is the only one this version opens, and
the identity (abs_path is the key — docs/PATHS.md) in operation.

A database of an earlier build is refused with a message instead of being
upgraded, and so is one from a newer version. A new database gets the whole
schema at once — including the indexes the upserts rely on (ON
CONFLICT(abs_path) needs the partial unique index).
"""

import sqlite3

import pytest

from whisper_tm.database import (
    SCHEMA_VERSION,
    Database,
    DatabaseVersionError,
)
from whisper_tm.database.speaker import SCHEMA_VERSION as SPEAKER_VERSION
from whisper_tm.database.speaker import SpeakerDB

#: Indexes the queries and upserts rely on
INDEXES = {
    "idx_media_files_abs_unique",
    "idx_transcripts_abs_unique",
    "idx_skiplist_abs_unique",
    "idx_summaries_unique",
    "idx_media_abs",
    "idx_transcript_abs",
    "idx_skiplist_abs",
    "idx_summaries_abs",
    "idx_media_rel",
    "idx_transcript_rel",
    "idx_transcript_media",
    "idx_media_duration",
    "idx_media_modified",
    "idx_media_size",
    "idx_media_indexed",
    "idx_media_abs_nocase",
    "idx_queue_filename",
}


def _version(path) -> str:
    with sqlite3.connect(path) as c:
        return c.execute("SELECT value FROM schema_info WHERE key = 'version'").fetchone()[0]


class TestTheBaseline:
    def test_a_new_database_gets_the_whole_schema(self, tmp_path):
        Database(tmp_path / "new.db").initialize()
        with sqlite3.connect(tmp_path / "new.db") as c:
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            unique = c.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'idx_transcripts_abs_unique'"
            ).fetchone()[0]
        assert INDEXES <= names
        assert "UNIQUE" in unique and "WHERE abs_path != ''" in unique
        assert _version(tmp_path / "new.db") == str(SCHEMA_VERSION)
        assert not (tmp_path / "backups").exists()  # nothing to lose

    def test_the_index_names_are_english(self, tmp_path):
        Database(tmp_path / "new.db").initialize()
        SpeakerDB(tmp_path / "speaker.db").initialize()
        names = set()
        for f in ("new.db", "speaker.db"):
            with sqlite3.connect(tmp_path / f) as c:
                names |= {
                    r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")
                }
        assert {
            "idx_steps_transcript",
            "idx_steps_step",
            "idx_sv_path",
            "idx_sv_model",
            "idx_sa_person",
            "idx_sm_group",
        } <= names
        assert not names & {
            "idx_vs_transcript",
            "idx_vs_schritt",
            "idx_sv_pfad",
            "idx_sv_modell",
            "idx_sz_person",
            "idx_szg_gruppe",
        }

    @pytest.mark.parametrize(
        "version,words",
        [
            (SCHEMA_VERSION - 1, "development build"),
            (SCHEMA_VERSION + 1, "newer version"),
        ],
    )
    def test_a_database_it_cannot_open_is_refused_untouched(self, tmp_path, version, words):
        path = tmp_path / "other.db"
        Database(path).initialize()
        with sqlite3.connect(path) as c:
            c.execute("UPDATE schema_info SET value = ? WHERE key = 'version'", (str(version),))
            c.execute("CREATE TABLE marker (x)")
        with pytest.raises(DatabaseVersionError, match=words):
            Database(path).initialize()
        assert _version(path) == str(version)

    def test_a_library_without_a_version_is_refused(self, tmp_path):
        """Rows but no version: a database of a build older than the version row."""
        path = tmp_path / "old.db"
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE media_files (id INTEGER PRIMARY KEY, rel_path TEXT)")
            c.execute("INSERT INTO media_files (rel_path) VALUES ('a.mp4')")
        with pytest.raises(DatabaseVersionError, match="development build"):
            Database(path).initialize()


class TestTheVoiceDatabase:
    def test_a_new_one_is_made_at_its_version(self, tmp_path):
        db = SpeakerDB(tmp_path / "speaker.db")
        db.initialize()
        db.close()
        with sqlite3.connect(tmp_path / "speaker.db") as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SPEAKER_VERSION

    @pytest.mark.parametrize(
        "version,words", [(3, "development build"), (SPEAKER_VERSION + 1, "newer version")]
    )
    def test_one_it_cannot_open_is_refused_untouched(self, tmp_path, version, words):
        path = tmp_path / "speaker.db"
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE speaker_vectors (id INTEGER)")
            c.execute(f"PRAGMA user_version = {version}")
        with pytest.raises(DatabaseVersionError, match=words):
            SpeakerDB(path).initialize()
        with sqlite3.connect(path) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == version


def _tx(filename, complete=True, segs=1, lang="de"):
    return {
        "filename": filename,
        "language": lang,
        "model": "turbo",
        "vocal_isolation": False,
        "vocal_model": None,
        "segments": [
            {
                "speaker": "S0",
                "start": float(i),
                "end": float(i) + 1,
                "start_fmt": "0:00",
                "end_fmt": "0:01",
                "text": f"hello world {i}",
            }
            for i in range(segs)
        ],
        "speakers": ["S0"],
        "total_duration": "0:01",
        "total_duration_sec": 1.0,
        "processing_time": "1s",
        "processing_secs": 1.0,
        "total_chunks": 1,
        "complete": complete,
        "warning_count": 0,
        "warnings": [],
    }


class TestIdentityInOperation:
    """Normal operation must maintain the identity."""

    @pytest.fixture
    def db(self, tmp_path):
        d = Database(tmp_path / "t.db")
        d.initialize()
        return d

    def test_upsert_updates_instead_of_duplicating(self, db):
        db.upsert_transcript("a.mp4", _tx("a.mp4", lang="de"), media_dir="/nas")
        db.upsert_transcript("a.mp4", _tx("a.mp4", lang="en"), media_dir="/nas")
        with db.connection() as conn:
            rows = conn.execute("SELECT language FROM transcripts").fetchall()
        assert len(rows) == 1
        assert rows[0]["language"] == "en"

    def test_same_name_different_folders_stay_separate(self, db):
        db.upsert_transcript("a/intro.mp4", _tx("a/intro.mp4"), media_dir="/nas")
        db.upsert_transcript("b/intro.mp4", _tx("b/intro.mp4"), media_dir="/nas")
        with db.connection() as conn:
            n = conn.execute("SELECT COUNT(*) c FROM transcripts").fetchone()["c"]
        assert n == 2

    def test_a_root_change_creates_no_duplicate(self, db):
        """The same file, once seen from the parent root and once
        from the subfolder — exactly the case that created the duplicates."""
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4"), media_dir="/nas/video")
        db.upsert_transcript("f1.mp4", _tx("f1.mp4"), media_dir="/nas/video/series")
        with db.connection() as conn:
            rows = conn.execute("SELECT rel_path, abs_path FROM transcripts").fetchall()
        assert len(rows) == 1, [dict(r) for r in rows]
        assert rows[0]["abs_path"] == "/nas/video/series/f1.mp4"

    def test_an_explicit_abs_path_beats_the_root(self, db):
        db.upsert_transcript(
            "any.mp4", _tx("any.mp4"), media_dir="/wrong", abs_path="/nas/real/file.mp4"
        )
        with db.connection() as conn:
            r = conn.execute("SELECT abs_path FROM transcripts").fetchone()
        assert r["abs_path"] == "/nas/real/file.mp4"

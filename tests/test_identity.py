"""
Identity must apply EVERYWHERE, not just on upsert.

Schema v8 removed uniqueness from `rel_path`. Every query that still relies on
it is now ambiguous — and several of them delete foreign data. These tests
ensure that two files with the same name in different roots stay separated
throughout. See docs/PATHS.md.
"""

import pytest

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


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.initialize()
    return d


@pytest.fixture
def two_roots(db):
    """Same relative path under two different roots.

    After switching the media directory, this is the normal state.
    """
    db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4", "de"), media_dir="/nasA")
    db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4", "en"), media_dir="/nasB")
    return db


class TestDeleteHitsOnlyTheIntendedFile:
    def test_both_exist_separately(self, two_roots):
        with two_roots.connection() as c:
            n = c.execute("SELECT COUNT(*) c FROM transcripts").fetchone()["c"]
        assert n == 2

    def test_the_other_files_summary_survives(self, two_roots):
        """A1: `DELETE FROM summaries WHERE rel_path = ?` hit both files.
        Summaries are expensively generated and not recoverable."""
        with two_roots.connection() as c:
            c.execute(
                "INSERT INTO summaries (rel_path, abs_path, sum_index, text) "
                "VALUES ('series/f1.mp4', '/nasB/series/f1.mp4', 0, 'from nasB')"
            )
        two_roots.delete_transcript("series/f1.mp4", media_dir="/nasA")
        with two_roots.connection() as c:
            remaining = c.execute("SELECT text FROM summaries").fetchall()
        assert len(remaining) == 1
        assert remaining[0]["text"] == "from nasB"

    def test_the_other_files_status_stays(self, two_roots):
        """A1: the status reset also ran via rel_path."""
        with two_roots.connection() as c:
            c.execute("UPDATE media_files SET status='done' WHERE abs_path='/nasB/series/f1.mp4'")
        two_roots.delete_transcript("series/f1.mp4", media_dir="/nasA")
        with two_roots.connection() as c:
            r = c.execute(
                "SELECT status FROM media_files WHERE abs_path='/nasB/series/f1.mp4'"
            ).fetchone()
        assert r["status"] == "done"

    def test_only_the_intended_transcript_disappears(self, two_roots):
        two_roots.delete_transcript("series/f1.mp4", media_dir="/nasA")
        with two_roots.connection() as c:
            remaining = [r["abs_path"] for r in c.execute("SELECT abs_path FROM transcripts")]
        assert remaining == ["/nasB/series/f1.mp4"]


class TestSkiplist:
    def test_two_files_with_the_same_name_can_both_be_skipped(self, db):
        """A2: the `rel_path` primary key collided even though upsert resolves
        to `abs_path` → IntegrityError from add_skip."""
        db.add_skip("series/f1.mp4", "manual", media_dir="/nasA")
        db.add_skip("series/f1.mp4", "manual", media_dir="/nasB")
        with db.connection() as c:
            entries = sorted(r["abs_path"] for r in c.execute("SELECT abs_path FROM skiplist"))
        assert entries == ["/nasA/series/f1.mp4", "/nasB/series/f1.mp4"]

    def test_removal_hits_only_one(self, db):
        db.add_skip("series/f1.mp4", "manual", media_dir="/nasA")
        db.add_skip("series/f1.mp4", "manual", media_dir="/nasB")
        db.remove_skip("series/f1.mp4", media_dir="/nasA")
        with db.connection() as c:
            remaining = [r["abs_path"] for r in c.execute("SELECT abs_path FROM skiplist")]
        assert remaining == ["/nasB/series/f1.mp4"]


class TestMediaLink:
    def test_the_transcript_hangs_on_the_right_media_file(self, two_roots):
        """A3: the `media_id` subselect ran over rel_path without LIMIT →
        the transcript could end up attached to the wrong row. Since the
        foreign key cascade-deletes, deleting that row drags it along."""
        with two_roots.connection() as c:
            pairs = c.execute("""
                SELECT t.abs_path AS t_abs, m.abs_path AS m_abs
                  FROM transcripts t JOIN media_files m ON m.id = t.media_id
            """).fetchall()
        assert len(pairs) == 2
        for p in pairs:
            assert p["t_abs"] == p["m_abs"], dict(p)

    def test_deleting_the_media_file_does_not_take_the_foreign_transcript(self, two_roots):
        with two_roots.connection() as c:
            c.execute("DELETE FROM media_files WHERE abs_path='/nasA/series/f1.mp4'")
            remaining = [r["abs_path"] for r in c.execute("SELECT abs_path FROM transcripts")]
        assert remaining == ["/nasB/series/f1.mp4"]


class TestSummaries:
    def test_both_files_can_have_their_own_summaries(self, two_roots):
        """A4: `UNIQUE(rel_path, sum_index)` only allowed one of the two
        files to have a summary at all."""
        two_roots.replace_summaries("series/f1.mp4", [{"text": "from nasA"}], media_dir="/nasA")
        two_roots.replace_summaries("series/f1.mp4", [{"text": "from nasB"}], media_dir="/nasB")
        a = two_roots.get_summaries("series/f1.mp4", media_dir="/nasA")
        b = two_roots.get_summaries("series/f1.mp4", media_dir="/nasB")
        assert a and a[0]["text"] == "from nasA"
        assert b and b[0]["text"] == "from nasB"


class TestScanDisplay:
    def test_the_media_file_shows_its_own_transcript(self, db):
        """B4: the scan join ran over rel_path → a file showed the
        metadata of a same-named file from a different folder."""
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4", "de"), media_dir="/nasA")
        db.upsert_transcript("series/f1.mp4", _tx("series/f1.mp4", "en"), media_dir="/nasB")
        line = db.get_media_file_at("/nasB/series/f1.mp4")
        assert line is not None
        assert line.get("tx_lang") == "en", line


class TestResolverRecognizesPathsCorrectly:
    """resolve_transcript decided via `":" in p[:3]` whether a path is
    absolute. A colon is a legal filename character on Linux — and since
    canon() returns "" for non-absolute inputs, this turned into a query
    `WHERE abs_path = ''`, which hit ANY old row."""

    @pytest.fixture
    def with_legacy_row(self, db):
        with db.connection() as c:
            c.execute(
                "INSERT INTO transcripts (rel_path, abs_path, language, complete)"
                " VALUES ('other/place.mp4', '', 'LEGACYROW', 1)"
            )
        return db

    @pytest.mark.parametrize(
        "name",
        [
            "10:30 Meeting.mp4",
            "a:b.mp4",
            "S1:E2 Title.mp4",
        ],
    )
    def test_a_colon_in_the_name_finds_the_right_file(self, with_legacy_row, name):
        with_legacy_row.upsert_transcript(name, _tx(name, "de"), media_dir="/nas")
        r = with_legacy_row.get_transcript(name, media_dir="/nas")
        assert r is not None
        assert r["language"] == "de", "hit a foreign row"

    def test_an_unknown_file_finds_no_legacy_row(self, with_legacy_row):
        """A file without a transcript must NOT return the old row."""
        assert with_legacy_row.get_transcript("11:45 Missing.mp4", media_dir="/nas") is None

    def test_deleting_does_not_hit_the_legacy_row(self, with_legacy_row):
        with_legacy_row.upsert_transcript(
            "10:30 Meeting.mp4", _tx("10:30 Meeting.mp4"), media_dir="/nas"
        )
        with_legacy_row.delete_transcript("10:30 Meeting.mp4", media_dir="/nas")
        with with_legacy_row.connection() as c:
            remaining = [r["language"] for r in c.execute("SELECT language FROM transcripts")]
        assert remaining == ["LEGACYROW"], "the old row was deleted along with it"


class TestDoNotDeleteForeignSummaries:
    def test_writing_deletes_no_foreign_legacy_rows(self, db):
        """replace_summaries ALWAYS deleted the rel_path old rows too — even
        when the identity was known."""
        with db.connection() as c:
            c.execute(
                "INSERT INTO summaries (rel_path, abs_path, sum_index, text)"
                " VALUES ('intro.mp4', '', 0, 'old from elsewhere')"
            )
        db.replace_summaries("intro.mp4", [{"text": "new"}], media_dir="/nasA")
        with db.connection() as c:
            texts = sorted(r["text"] for r in c.execute("SELECT text FROM summaries"))
        assert "old from elsewhere" in texts, "foreign old row deleted"


class TestLikePrefixNeedsARoot:
    def test_an_empty_root_does_not_match_everything(self, db):
        """like_prefix('') resulted in '/', and the pattern '/%' matches EVERY row."""
        db.upsert_transcript("a.mp4", _tx("a.mp4"), media_dir="/nasA")
        db.upsert_transcript("b.mp4", _tx("b.mp4"), media_dir="/nasB")
        assert db.get_media_files_under("") == []
        assert db.count_media_files_under("") == 0

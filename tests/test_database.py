"""
Tests for database/__init__.py — SQLite + FTS5.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    """Fresh database for each test."""
    db_path = tmp_path / "test.db"
    d = Database(db_path)
    d.initialize()
    return d


@pytest.fixture
def sample_transcript():
    return {
        "filename": "test_video.mp4",
        "language": "de",
        "model": "turbo",
        "vocal_isolation": False,
        "vocal_model": None,
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.5,
                "start_fmt": "0:00:00",
                "end_fmt": "0:00:02",
                "text": "Hello world, this is a Test.",
            },
            {
                "speaker": "SPEAKER_01",
                "start": 3.0,
                "end": 5.0,
                "start_fmt": "0:00:03",
                "end_fmt": "0:00:05",
                "text": "Yes, a wonderful Test.",
            },
            {
                "speaker": "SPEAKER_00",
                "start": 5.5,
                "end": 8.0,
                "start_fmt": "0:00:05",
                "end_fmt": "0:00:08",
                "text": "Whisper Transcription Manager works brilliantly.",
            },
        ],
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "total_duration": "0:00:08",
        "total_duration_sec": 8.0,
        "processing_time": "5s",
        "processing_secs": 5.0,
        "total_chunks": 1,
        "complete": True,
        "warning_count": 0,
        "warnings": [],
    }


class TestDatabaseInit:
    def test_initialize(self, db):
        """Schema is correctly created."""
        with db.connection() as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = {r["name"] for r in tables}
            assert "transcripts" in table_names
            assert "segments" in table_names
            assert "segments_fts" in table_names
            assert "media_files" in table_names
            assert "skiplist" in table_names

    def test_wal_mode(self, db):
        with db.connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"

    def test_double_initialize(self, db):
        """Second initialize() must not crash."""
        db.initialize()


class TestTranscriptCRUD:
    def test_upsert_and_get(self, db, sample_transcript):
        tid = db.upsert_transcript("test_video.mp4", sample_transcript)
        assert tid > 0

        result = db.get_transcript("test_video.mp4")
        assert result is not None
        assert result["language"] == "de"
        assert result["model"] == "turbo"
        assert len(result["segments"]) == 3

    def test_app_version_roundtrip(self, db, sample_transcript):
        """App version of the transcription: DB column ↔ file-list field.

        Schema v13: app_version is stored on upsert and comes back as
        tx_app_version in the file-list query (display chip + filter
        "v2.1.0" in the UI). Re-importing an old JSON WITHOUT a version
        must not overwrite an existing value with an empty one.
        """
        sample_transcript["app_version"] = "2.1.0"
        sample_transcript["complete"] = True
        db.upsert_transcript("Games/clip.mp4", sample_transcript, media_dir="/mnt/nas")

        rows = db.get_media_files_under("/mnt/nas")
        by_rel = {r["rel_path"]: r for r in rows}
        assert by_rel["Games/clip.mp4"]["tx_app_version"] == "2.1.0"

        # Re-import an old JSON without a version (DB-sync case) → value stays
        old_value = dict(sample_transcript)
        old_value.pop("app_version")
        db.upsert_transcript("Games/clip.mp4", old_value, media_dir="/mnt/nas")
        rows = db.get_media_files_under("/mnt/nas")
        by_rel = {r["rel_path"]: r for r in rows}
        assert by_rel["Games/clip.mp4"]["tx_app_version"] == "2.1.0"

    def test_found_after_media_dir_change(self, db, sample_transcript):
        """The transcript stays findable when the media directory changes.

        Reported: clicking a search result → "no transcript found", while
        the video played normally. Cause: the row was created under
        media_dir=/mnt/nas/videos/Recordings, the query was made with the
        path relative to the new media directory /mnt/nas. Affected
        thousands of transcripts in a large library.
        """
        db.upsert_transcript(
            "Games/clip.mp4", sample_transcript, media_dir="/mnt/nas/videos/Recordings"
        )

        # Path relative to the NEW root — a direct lookup must fail ...
        new_value = "videos/Recordings/Games/clip.mp4"
        assert db.get_transcript(new_value) is None
        # ... via the absolute path it is found
        found = db.get_transcript(new_value, media_dir="/mnt/nas")
        assert found is not None
        assert len(found["segments"]) == 3

    def test_media_dir_hint_does_not_mismatch(self, db, sample_transcript):
        """The fallback must not return an unrelated file."""
        db.upsert_transcript(
            "Games/clip.mp4", sample_transcript, media_dir="/mnt/nas/videos/Recordings"
        )
        assert db.get_transcript("somewhere/else/clip.mp4", media_dir="/mnt/nas") is None
        # Same filename, different folder → no match
        assert db.get_transcript("videos/Other/Games/clip.mp4", media_dir="/mnt/nas") is None

    def test_abs_path_stored_on_upsert(self, db, sample_transcript):
        """The absolute path is stored along with it — it is the identity."""
        db.upsert_transcript("Games/clip.mp4", sample_transcript, media_dir="/mnt/nas/videos")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT abs_path FROM transcripts WHERE rel_path = 'Games/clip.mp4'"
            ).fetchone()
        assert row["abs_path"] == "/mnt/nas/videos/Games/clip.mp4"

    def test_survives_media_dir_change_end_to_end(self, db, sample_transcript):
        """A change of media directory must not make anything unfindable.

        That is the guarantee at stake here: the same file, addressed from
        a different root, is found through every path — loading, the
        completion check, and deletion.
        """
        db.upsert_transcript("Games/clip.mp4", sample_transcript, media_dir="/mnt/nas/videos")
        # Media directory moves up one level
        from_new = "videos/Games/clip.mp4"

        assert db.get_transcript(from_new, media_dir="/mnt/nas") is not None
        assert db.is_complete(from_new, media_dir="/mnt/nas")
        assert db.delete_transcript(from_new, media_dir="/mnt/nas")
        assert db.get_transcript(from_new, media_dir="/mnt/nas") is None

    def test_delete_cleans_up_under_stored_path(self, db, sample_transcript):
        """Deleting via the absolute path also cleans up around it.

        Summaries and file status hang off the STORED rel_path, not the
        requested one — otherwise orphans remain.
        """
        db.upsert_transcript("Games/clip.mp4", sample_transcript, media_dir="/mnt/nas/videos")
        db.bulk_upsert_media(
            [
                {
                    "rel_path": "Games/clip.mp4",
                    "abs_path": "/mnt/nas/videos/Games/clip.mp4",
                    "display_name": "clip.mp4",
                    "subfolder": "Games",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 1,
                    "modified_at": 1.0,
                    "status": "done",
                }
            ]
        )
        db.delete_transcript("videos/Games/clip.mp4", media_dir="/mnt/nas")
        with db.connection() as conn:
            st = conn.execute(
                "SELECT status FROM media_files WHERE rel_path = 'Games/clip.mp4'"
            ).fetchone()
        assert st["status"] == "pending", "file status not reset"

    def test_direct_lookup_still_wins(self, db, sample_transcript):
        """The normal case still works without a detour."""
        db.upsert_transcript("test_video.mp4", sample_transcript, media_dir="/mnt/nas")
        assert db.get_transcript("test_video.mp4", media_dir="/mnt/nas")

    def test_wrong_root_finds_nothing(self, db, sample_transcript):
        """Nothing may be found with the WRONG root.

        Previously the resolver fell back to `WHERE rel_path = ?` and
        returned the row anyway — without the restriction to
        `abs_path = ''` required by docs/PATHS.md. On read that was a wrong
        display; on delete it was data loss: delete_transcript() then hit
        the transcript of a same-named file from a different root.
        """
        db.upsert_transcript("test_video.mp4", sample_transcript, media_dir="/mnt/nasA")
        assert db.get_transcript("test_video.mp4", media_dir="/mnt/nasB") is None

    def test_delete_hits_only_the_intended_root(self, db, sample_transcript):
        """The data-loss case, reproduced."""
        db.upsert_transcript("series/f1.mp4", sample_transcript, media_dir="/mnt/nasB")
        # A delete attempt with the OTHER root must not touch anything
        assert db.delete_transcript("series/f1.mp4", media_dir="/mnt/nasA") is False
        assert db.get_transcript("series/f1.mp4", media_dir="/mnt/nasB")
        # With the correct root it takes effect
        assert db.delete_transcript("series/f1.mp4", media_dir="/mnt/nasB") is True
        assert db.get_transcript("series/f1.mp4", media_dir="/mnt/nasB") is None

    def test_old_row_without_identity_stays_findable(self, db, sample_transcript):
        """Level 2 still exists — but only for rows without abs_path."""
        db.upsert_transcript("old_value.mp4", sample_transcript, media_dir="/mnt/nas")
        with db.connection() as conn:
            conn.execute("UPDATE transcripts SET abs_path = '' WHERE rel_path = 'old_value.mp4'")
        assert db.get_transcript("old_value.mp4", media_dir="/mnt/somewhere/else")

    def test_is_complete(self, db, sample_transcript):
        assert not db.is_complete("test_video.mp4")
        db.upsert_transcript("test_video.mp4", sample_transcript)
        assert db.is_complete("test_video.mp4")

    def test_upsert_updates(self, db, sample_transcript):
        """A second upsert updates instead of inserting a duplicate."""
        db.upsert_transcript("test_video.mp4", sample_transcript)

        updated = dict(sample_transcript)
        updated["language"] = "en"
        updated["model"] = "large-v3"
        db.upsert_transcript("test_video.mp4", updated)

        result = db.get_transcript("test_video.mp4")
        assert result["language"] == "en"
        assert result["model"] == "large-v3"

    def test_delete_transcript(self, db, sample_transcript):
        db.upsert_transcript("test_video.mp4", sample_transcript)
        assert db.is_complete("test_video.mp4")

        db.delete_transcript("test_video.mp4")
        assert not db.is_complete("test_video.mp4")
        assert db.get_transcript("test_video.mp4") is None

    def test_delete_nonexistent(self, db):
        assert db.delete_transcript("nonexistent.mp4") is False

    def test_segments_stored(self, db, sample_transcript):
        """Segments are correctly stored in their own table."""
        db.upsert_transcript("test_video.mp4", sample_transcript)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT id FROM transcripts WHERE rel_path = ?", ("test_video.mp4",)
            ).fetchone()
            segs = conn.execute(
                "SELECT * FROM segments WHERE transcript_id = ? ORDER BY seg_index", (row["id"],)
            ).fetchall()
            assert len(segs) == 3
            assert segs[0]["text"] == "Hello world, this is a Test."
            assert segs[1]["speaker"] == "SPEAKER_01"

    def test_media_dir_stored(self, db, sample_transcript):
        db.upsert_transcript("test_video.mp4", sample_transcript, media_dir="/mnt/nas/videos")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT media_dir FROM transcripts WHERE rel_path = ?", ("test_video.mp4",)
            ).fetchone()
            assert row["media_dir"] == "/mnt/nas/videos"


class TestFTS5Search:
    @pytest.fixture(autouse=True)
    def setup_data(self, db, sample_transcript):
        db.upsert_transcript("video1.mp4", sample_transcript)

        tx2 = dict(sample_transcript)
        tx2["filename"] = "video2.mp4"
        tx2["segments"] = [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 3.0,
                "start_fmt": "0:00:00",
                "end_fmt": "0:00:03",
                "text": "Python and Machine Learning are exciting.",
            },
        ]
        db.upsert_transcript("video2.mp4", tx2)
        self.db = db

    def test_simple_search(self):
        result = self.db.search("Test")
        assert result["total"] > 0
        assert any("Test" in r["text"] for r in result["results"])

    def test_phrase_search(self):
        result = self.db.search('"Hello world"')
        assert result["total"] >= 1

    def test_multi_word(self):
        result = self.db.search("Machine Learning")
        assert result["total"] >= 1

    def test_file_filter(self):
        result = self.db.search("Test", file_filter="video1")
        assert all("video1" in r["filename"] for r in result["results"])

    def test_speaker_filter(self):
        result = self.db.search("Test", speaker_filter="SPEAKER_01")
        assert all(r["speaker"] == "SPEAKER_01" for r in result["results"])

    def test_no_results(self):
        result = self.db.search("xyznonexistent")
        assert result["total"] == 0

    def test_pagination(self):
        result = self.db.search("Test", limit=1, offset=0)
        assert len(result["results"]) <= 1

    def test_prefix_search(self):
        """Single-word search should match as a prefix."""
        result = self.db.search("Whisper")
        assert result["total"] >= 1

    def test_highlighting(self):
        """Hits are marked with \\x02/\\x03, NOT with finished HTML.

        If <mark> came from the DB, the frontend would have to insert the
        text unchecked — a <script> in the transcript would then be a hole.
        This way it is escaped first and swapped in afterward.
        """
        result = self.db.search("Test")
        assert result["results"]
        h = result["results"][0]["highlighted"]
        assert "\x02Test\x03" in h, h
        assert "<mark>" not in h

    def test_highlighting_marks_deaccented_match(self):
        """The index keeps words without diacritics — the highlight still
        sits on the original form in the text.

        This is exactly where the old browser highlighting used to fail:
        searching for 'ware' (index form), the text has 'wäre' — the regex
        found nothing.
        """
        tx = {
            "filename": "uml.mp4",
            "language": "de",
            "complete": True,
            "segments": [
                {
                    "speaker": "A",
                    "start": 0.0,
                    "end": 1.0,
                    "start_fmt": "0:00:00",
                    "end_fmt": "0:00:01",
                    "text": "Das wäre dramatisch.",
                }
            ],
        }
        self.db.upsert_transcript("uml.mp4", tx)
        r = self.db.search("ware")
        hit = next(x for x in r["results"] if x["filename"] == "uml.mp4")
        assert "\x02wäre\x03" in hit["highlighted"], hit["highlighted"]

    def test_special_chars_sanitized(self):
        """Special characters must not crash FTS5."""
        result = self.db.search("test (with) [brackets] {braces}")
        assert isinstance(result, dict)

    def test_empty_query(self):
        result = self.db.search("")
        assert result["total"] == 0

    # ── Fuzzy search (search.fuzzy_enabled / fuzzy_threshold) ──────────

    def test_fuzzy_finds_typo(self):
        """A typo should still find the hit."""
        exact = self.db.search("Machnie")  # 'Machine' scrambled
        assert exact["total"] == 0, "precondition: no exact hit"

        fuzzy = self.db.search("Machnie", fuzzy=0.7)
        assert fuzzy["total"] >= 1
        assert any("Machine" in r["text"] for r in fuzzy["results"])

    def test_fuzzy_off_by_default(self):
        """Without the fuzzy parameter, search stays exact."""
        assert self.db.search("Machnie")["total"] == 0

    def test_fuzzy_respects_threshold(self):
        """A high threshold no longer lets a coarse typo through."""
        assert self.db.search("Machnie", fuzzy=0.99)["total"] == 0

    def test_fuzzy_keeps_phrases_exact(self):
        """With quotation marks, matching is NOT loosened."""
        r = self.db.search('"Machine Learning"', fuzzy=0.6)
        assert r["total"] >= 1
        assert all("Machine Learning" in x["text"] for x in r["results"])

    def test_fuzzy_reports_matched_terms(self):
        """The matched words come back — otherwise the highlighting is missing.

        The frontend highlights via `terms`. Without this list it would
        highlight the TYPED word ('Machnie'), which does not even appear in
        the text — the fuzzy hit would remain visually unmarked.
        """
        r = self.db.search("Machnie", fuzzy=0.7)
        assert r["total"] >= 1
        terms = [t.lower() for t in r["terms"]]
        assert "machine" in terms, terms
        assert "machnie" in terms, "the original must stay in"

    def test_no_terms_without_fuzzy(self):
        """No terms key without fuzzy — the frontend highlights the input."""
        assert "terms" not in self.db.search("Machine")

    def test_no_terms_when_nothing_similar(self):
        """If fuzzy finds nothing similar, it stays with the input."""
        assert "terms" not in self.db.search("Machine", fuzzy=0.99)

    def test_fuzzy_rejects_distant_words(self):
        """The reported false hit: 'warkel' found 'wäre'.

        The index keeps 'wäre' as 'ware'; difflib scored that at exactly
        0.800 and returned thousands of hits for a very common word. Two changes
        across six characters is not a typo.
        """
        tx = {
            "filename": "w.mp4",
            "language": "de",
            "complete": True,
            "segments": [
                {
                    "speaker": "A",
                    "start": 0.0,
                    "end": 1.0,
                    "start_fmt": "0:00:00",
                    "end_fmt": "0:00:01",
                    "text": "Das wäre dramatisch.",
                }
            ],
        }
        self.db.upsert_transcript("w.mp4", tx)
        r = self.db.search("warkel", fuzzy=0.8)
        assert not [x for x in r["results"] if x["filename"] == "w.mp4"], (
            "'warkel' must not find 'wäre'"
        )

    def test_fuzzy_allows_single_edit(self):
        """A single change is the classic typo — it must go through."""
        r = self.db.search("Machin", fuzzy=0.8)  # 'Machine' minus 'e'
        assert r["total"] >= 1

    def test_edit_distance_bounded(self):
        """The distance calculation stops above the bound."""
        d = self.db._edit_distance_within
        assert d("machine", "machine", 2) == 0
        assert d("machin", "machine", 2) == 1
        assert d("warkel", "ware", 2) == 2
        assert d("waere", "werde", 2) == 2  # delete + insert, not 3x substitute
        assert d("short", "muchmuchlonger", 2) > 2  # aborted, not exact

    def test_max_edits_scale_with_word_length(self):
        """Two changes only for long words.

        'waere' (5 characters) and 'werde' are only two changes apart —
        but for such short words that is already a different word. The
        bound therefore grows with word length.
        """

        def max_edits(word, threshold=0.8):
            return max(1, min(2, int((1.0 - threshold) * len(word) + 1e-9)))

        assert self.db._edit_distance_within("waere", "werde", 2) == 2
        # ... but at five characters only one change is allowed
        assert max_edits("waere") == 1
        # Two only from ten characters on. Without the epsilon in the
        # formula, (1.0 - 0.8) * 10 would give 1.9999999999999996 here and
        # thus wrongly 1.
        assert max_edits("tenletters") == 2
        assert max_edits("ninechars") == 1

    def test_fuzzy_short_words_untouched(self):
        """Short words are too ambiguous for fuzzy — no runaway hit growth."""
        r = self.db.search("xyz", fuzzy=0.6)
        assert r["total"] == 0

    # ── Context segments (search.context_segments) ────────────────────

    def test_context_segments_attached(self):
        tx = {
            "filename": "ctx.mp4",
            "language": "de",
            "complete": True,
            "segments": [
                {
                    "speaker": "A",
                    "start": float(i),
                    "end": float(i + 1),
                    "start_fmt": f"0:00:0{i}",
                    "end_fmt": f"0:00:0{i + 1}",
                    "text": f"Sentence {i}" if i != 2 else "Here is Contextword",
                }
                for i in range(5)
            ],
        }
        self.db.upsert_transcript("ctx.mp4", tx)

        r = self.db.search("Contextword", context=2)
        hit = next(x for x in r["results"] if x["filename"] == "ctx.mp4")
        assert [s["text"] for s in hit["context_before"]] == ["Sentence 0", "Sentence 1"]
        assert [s["text"] for s in hit["context_after"]] == ["Sentence 3", "Sentence 4"]

    def test_context_zero_adds_no_keys(self):
        """Without context, the response stays as lean as before."""
        r = self.db.search("Test")
        for x in r["results"]:
            assert "context_before" not in x

    def test_context_at_transcript_start(self):
        """At the start there is nothing before it — no crash, empty list."""
        r = self.db.search("Hello", context=3)
        for x in r["results"]:
            assert isinstance(x["context_before"], list)


class TestSkiplist:
    def test_add_and_get(self, db):
        """get_skiplist() keys on identity (docs/PATHS.md).

        Via rel_path, a skipped intro.mp4 from folder A also locked out the
        same-named one from folder B.
        """
        db.add_skip("problem.mp4", "no_audio", media_dir="/nas/video")
        sl = db.get_skiplist()
        assert "/nas/video/problem.mp4" in sl
        assert sl["/nas/video/problem.mp4"]["reason"] == "no_audio"
        assert "problem.mp4" not in sl

    def test_files_with_the_same_name_stay_separate(self, db):
        """Two roots, same name → two independent entries."""
        db.add_skip("problem.mp4", "no_audio", media_dir="/nas/a")
        sl = db.get_skiplist()
        assert "/nas/a/problem.mp4" in sl
        assert "/nas/b/problem.mp4" not in sl

    def test_remove(self, db):
        db.add_skip("problem.mp4", "manual")
        db.remove_skip("problem.mp4")
        sl = db.get_skiplist()
        assert "problem.mp4" not in sl

    def test_media_status_updated(self, db):
        """Skip should also update media_files status."""
        db.upsert_media_file("problem.mp4", status="pending")
        db.add_skip("problem.mp4", "no_audio")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT status FROM media_files WHERE rel_path = ?", ("problem.mp4",)
            ).fetchone()
            assert row["status"] == "skipped"


class TestStats:
    def test_get_stats(self, db, sample_transcript):
        db.upsert_transcript("test.mp4", sample_transcript)
        stats = db.get_stats()
        assert stats["transcripts"]["total"] == 1
        assert stats["transcripts"]["total_segments"] == 3

    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["transcripts"]["total"] == 0


class TestSyncFromJson:
    def test_sync(self, db, sample_transcript, tmp_path):
        tx_dir = tmp_path / "transcripts"
        tx_dir.mkdir()
        with open(tx_dir / "test.mp4.json", "w") as f:
            json.dump(sample_transcript, f)

        result = db.sync_from_json_dir(tx_dir)
        assert result["imported"] == 1
        assert result["skipped"] == 0
        assert result["errors"] == 0

    def test_sync_skips_incomplete(self, db, sample_transcript, tmp_path):
        tx_dir = tmp_path / "transcripts"
        tx_dir.mkdir()
        incomplete = dict(sample_transcript)
        incomplete["complete"] = False
        with open(tx_dir / "partial.json", "w") as f:
            json.dump(incomplete, f)

        result = db.sync_from_json_dir(tx_dir)
        assert result["imported"] == 0
        assert result["skipped"] == 1


class TestUserSettings:
    def test_set_and_get(self, db):
        db.set_setting("theme", "dark")
        assert db.get_setting("theme") == "dark"

    def test_default(self, db):
        assert db.get_setting("nonexistent", default="light") == "light"

    def test_get_all(self, db):
        db.set_setting("a", "1")
        db.set_setting("b", "2")
        all_s = db.get_all_settings()
        assert all_s == {"a": "1", "b": "2"}


class TestBulkMethods:
    """Tests for bulk queries (scan performance at 50k+ files)."""

    # The bulk queries are keyed on abs_path — the identity
    # (docs/PATHS.md). A fixed root is enough for this; it does not need to
    # exist, abs_from() computes purely textually.
    ROOT = "/nas/video"

    def _id(self, rel: str) -> str:
        from whisper_tm.core import paths

        return paths.abs_from(rel, self.ROOT)

    @pytest.fixture(autouse=True)
    def setup_data(self, db, sample_transcript):
        """Create several transcripts for bulk tests."""
        db.upsert_transcript("video1.mp4", sample_transcript, media_dir=self.ROOT)

        tx2 = dict(sample_transcript)
        tx2["filename"] = "video2.mp4"
        tx2["model"] = "large-v3"
        db.upsert_transcript("video2.mp4", tx2, media_dir=self.ROOT)

        tx3 = dict(sample_transcript)
        tx3["filename"] = "subdir/video3.mp4"
        tx3["complete"] = False  # incomplete
        db.upsert_transcript("subdir/video3.mp4", tx3, media_dir=self.ROOT)

        self.db = db

    def test_get_completion_set(self):
        result = self.db.get_completion_set()
        assert isinstance(result, set)
        assert self._id("video1.mp4") in result
        assert self._id("video2.mp4") in result
        # incomplete should not be in set
        assert self._id("subdir/video3.mp4") not in result

    def test_completion_set_is_keyed_on_the_identity(self):
        """The display value does NOT show up as a key.

        Otherwise a file would count as done because a same-named one from
        a different root has a transcript (docs/PATHS.md).
        """
        result = self.db.get_completion_set()
        assert "video1.mp4" not in result

    def test_completion_set_separates_files_with_the_same_name(self):
        """Same rel_path, two roots → two separate entries."""
        tx = {"complete": True, "segments": [], "model": "turbo"}
        self.db.upsert_transcript("video1.mp4", tx, media_dir="/nas/other")
        result = self.db.get_completion_set()
        assert self._id("video1.mp4") in result
        assert "/nas/other/video1.mp4" in result

    def test_completion_set_empty_db(self, tmp_path):
        empty_db = Database(tmp_path / "empty.db")
        empty_db.initialize()
        result = empty_db.get_completion_set()
        assert result == set()

    def test_get_transcript_meta_batch_all(self):
        result = self.db.get_transcript_meta_batch()
        assert self._id("video1.mp4") in result
        assert self._id("video2.mp4") in result
        assert self._id("subdir/video3.mp4") not in result  # incomplete
        assert result[self._id("video1.mp4")]["tx_model"] == "turbo"
        assert result[self._id("video2.mp4")]["tx_model"] == "large-v3"

    def test_get_transcript_meta_batch_filtered(self):
        result = self.db.get_transcript_meta_batch([self._id("video1.mp4")])
        assert self._id("video1.mp4") in result
        assert self._id("video2.mp4") not in result

    def test_meta_batch_separates_files_with_the_same_name(self):
        """Two roots, same name → two metadata records.

        Via rel_path they used to share ONE, and which file got it depended
        on row order: the list showed duration, model and speakers of the
        respectively other file (docs/PATHS.md).
        """
        tx = {"complete": True, "segments": [], "model": "tiny"}
        self.db.upsert_transcript("video1.mp4", tx, media_dir="/nas/other")
        result = self.db.get_transcript_meta_batch()
        assert result[self._id("video1.mp4")]["tx_model"] == "turbo"
        assert result["/nas/other/video1.mp4"]["tx_model"] == "tiny"

    def test_get_transcript_meta_batch_missing(self):
        result = self.db.get_transcript_meta_batch(["/nas/video/nonexistent.mp4"])
        assert result == {}

    def test_meta_batch_has_expected_keys(self):
        result = self.db.get_transcript_meta_batch()
        meta = result[self._id("video1.mp4")]
        assert "processing_time" in meta
        assert "audio_duration_sec" in meta
        assert "tx_model" in meta
        assert "tx_lang" in meta
        assert "tx_seg_count" in meta
        assert "tx_word_count" in meta


class TestBackup:
    def test_backup_creates_file(self, db):
        bp = db.backup()
        assert bp is not None
        assert bp.exists()
        assert bp.stat().st_size > 0

    def test_backup_rotation(self, db):
        """Old backups get rotated out."""
        for _ in range(8):
            import time

            time.sleep(0.01)  # Ensure unique timestamps
            db.backup()
        backup_dir = Path(db._db_path).parent / "backups"
        backups = list(backup_dir.glob("*.db"))
        assert len(backups) <= 5  # max_backups default

    def test_vacuum_creates_backup(self, db, sample_transcript):
        db.upsert_transcript("test.mp4", sample_transcript)
        db.vacuum()
        backup_dir = Path(db._db_path).parent / "backups"
        assert backup_dir.exists()
        assert len(list(backup_dir.glob("*.db"))) >= 1

    def test_backup_to_custom_dir(self, db, tmp_path):
        """An external target (NAS path) gets created and written to."""
        dest = tmp_path / "nas" / "whisperx"
        bp = db.backup(dest)
        assert bp is not None
        assert bp.parent == dest
        assert bp.stat().st_size > 0

    def test_backup_unreachable_dest_returns_none(self, db, tmp_path):
        """Destination that cannot be created (NAS gone) → None, not an exception.

        The caller sits in the write path of a transcription; an exception
        there would take the finished work down with it.
        """
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        assert db.backup(blocker / "subfolders") is None

    def test_backup_staged_locally_not_written_page_by_page(self, db, tmp_path):
        """The destination is written ONCE in one go, not page by page.

        Writing the SQLite backup API page by page directly onto a CIFS
        share takes >5 min at ~500 MB and leaves behind a 0-byte file.
        Hence: stage locally, copy the finished file.
        """
        dest = tmp_path / "nas"
        dest.mkdir()
        opened: list[str] = []
        real_connect = sqlite3.connect

        def spy(target, *a, **kw):
            opened.append(str(target))
            return real_connect(target, *a, **kw)

        with patch("sqlite3.connect", side_effect=spy):
            bp = db.backup(dest)

        assert bp is not None and bp.stat().st_size > 0
        # No SQLite connection directly on the target directory
        assert not any(str(dest) in t for t in opened), opened

    def test_backup_leaves_no_empty_file_when_copy_fails(self, db, tmp_path):
        """If the copy to the target fails, no 0-byte remnant is left behind.

        An empty file in the backup folder looks like a
        valid backup at a glance — exactly the case that must never occur here.
        """
        dest = tmp_path / "nas"
        dest.mkdir()
        with (
            patch("os.replace", side_effect=OSError("EXDEV")),
            patch("shutil.copyfile", side_effect=OSError("NAS unreachable")),
        ):
            assert db.backup(dest) is None

        assert list(dest.glob("*.db")) == []
        # The local staging file is also gone
        assert list(Path(db._db_path).parent.glob(".*.tmp")) == []

    def test_backup_final_name_only_after_complete_copy(self, db, tmp_path):
        """If the copy is aborted midway, NO backup is created under the
        final name.

        A Ctrl-C during a network copy can leave behind a large file with
        a valid name — not recognizable as a torso from the outside.
        """
        dest = tmp_path / "nas"
        dest.mkdir()

        def the_mid_copy(src, dst):
            Path(dst).write_bytes(b"partial file")  # torso, as on abort
            raise KeyboardInterrupt("Ctrl-C mid-copy")

        with (
            patch("os.replace", side_effect=OSError("EXDEV")),
            patch("shutil.copyfile", side_effect=the_mid_copy),
            pytest.raises(KeyboardInterrupt),
        ):
            db.backup(dest)

        assert list(dest.glob("*.db")) == [], "torso under a valid name!"
        # The local staging file is as large as the DB — don't leave it lying around
        assert list(Path(db._db_path).parent.glob(".*.tmp")) == []

    def test_rotation_removes_part_and_wal_leftovers(self, db, tmp_path):
        """Leftovers from a crashed run get cleaned up too."""
        dest = tmp_path / "nas"
        dest.mkdir()
        stem = Path(db._db_path).stem
        (dest / f".{stem}_20200101_000000_000000.db.part").write_bytes(b"remainder")
        (dest / f"{stem}_20200101_000000_000000.db-wal").write_bytes(b"remainder")
        (dest / f"{stem}_20200101_000000_000000.db-shm").write_bytes(b"remainder")

        assert db.backup(dest) is not None
        assert list(dest.glob(".*.part")) == []
        assert list(dest.glob("*.db-wal")) == []
        assert list(dest.glob("*.db-shm")) == []


class TestExternalBackupCounter:
    """Automatic backup after N transcript changes."""

    @pytest.fixture
    def cfg(self, tmp_path, monkeypatch):
        """Point the config at a temporary NAS destination."""
        from whisper_tm.config import config

        dest = tmp_path / "nas"
        orig = config.get

        # config.get is variadic (get("a", "b") like get("a.b")) — the fake
        # must have the same shape, or callers elsewhere break.
        def fake_get(*keys, default=None):
            flat = ".".join(keys)
            if flat == "database.backup_path":
                return str(dest)
            if flat == "database.backup_every_changes":
                return 3
            return orig(*keys, default=default)

        monkeypatch.setattr(config, "get", fake_get)
        return dest

    def _wait_for_backup(self, db, dest, timeout=5.0):
        """The backup runs in a background thread — wait for completion."""
        import time as _t

        deadline = _t.time() + timeout
        while _t.time() < deadline:
            if not db._ext_backup_running and dest.is_dir() and list(dest.glob("*.db")):
                return True
            _t.sleep(0.02)
        return False

    def test_counter_increments_on_transcript_write(self, db, sample_transcript, cfg):
        db.upsert_transcript("a.mp4", sample_transcript)
        assert db.get_backup_counter() == 1
        db.upsert_transcript("b.mp4", sample_transcript)
        assert db.get_backup_counter() == 2

    def test_backup_fires_at_threshold_and_resets(self, db, sample_transcript, cfg):
        for name in ("a.mp4", "b.mp4", "c.mp4"):  # threshold = 3
            db.upsert_transcript(name, sample_transcript)
        assert self._wait_for_backup(db, cfg), "no backup appeared at the target"
        assert len(list(cfg.glob("*.db"))) == 1
        # Counter must be reset, otherwise every further change would trigger one
        assert db.get_backup_counter() == 0

    def test_delete_also_counts(self, db, sample_transcript, cfg):
        db.upsert_transcript("a.mp4", sample_transcript)
        db.delete_transcript("a.mp4")
        assert db.get_backup_counter() == 2

    def test_scan_writes_do_not_count(self, db, sample_transcript, cfg):
        """Index/scan rows must NOT trigger a backup — otherwise a scan run
        over tens of thousands of files would flood the destination."""
        db.bulk_upsert_media(
            [
                {
                    "rel_path": f"f{i}.mp4",
                    "abs_path": f"/m/f{i}.mp4",
                    "display_name": f"f{i}.mp4",
                    "subfolder": "",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 1,
                    "modified_at": 1.0,
                    "status": "pending",
                }
                for i in range(10)
            ]
        )
        assert db.get_backup_counter() == 0
        assert not cfg.exists() or not list(cfg.glob("*.db"))

    def test_bulk_import_does_not_count(self, db, sample_transcript, cfg):
        """A mass import does not count — otherwise a backup chain instead
        of a single backup."""
        with db.bulk_import():
            for i in range(10):  # threshold = 3
                db.upsert_transcript(f"f{i}.mp4", sample_transcript)
        assert db.get_backup_counter() == 0
        assert not cfg.exists() or not list(cfg.glob("*.db"))
        # Counts normally again afterward
        db.upsert_transcript("afterwards.mp4", sample_transcript)
        assert db.get_backup_counter() == 1

    def test_start_sync_does_not_flood_backups(self, db, sample_transcript, cfg, tmp_path):
        """The startup sync (JSON → DB) must not trigger a NAS backup.

        Otherwise thousands of imported JSONs at a low threshold would cause
        back-to-back backups of a large DB over the network for minutes.
        """
        tdir = tmp_path / ".transcripts"
        tdir.mkdir()
        for i in range(10):  # threshold = 3
            data = dict(sample_transcript, filename=f"f{i}.mp4", complete=True)
            (tdir / f"f{i}.json").write_text(json.dumps(data), encoding="utf-8")

        result = db.sync_from_json_dir(tdir, media_dir=str(tmp_path))

        assert result["imported"] == 10
        assert db.get_backup_counter() == 0
        assert not cfg.exists() or not list(cfg.glob("*.db"))

    def test_disabled_without_path(self, db, sample_transcript, monkeypatch):
        """Without a backup path, it doesn't even count."""
        from whisper_tm.config import config

        orig = config.get

        def fake_get(*keys, default=None):
            if ".".join(keys) == "database.backup_path":
                return ""
            return orig(*keys, default=default)

        monkeypatch.setattr(config, "get", fake_get)
        db.upsert_transcript("a.mp4", sample_transcript)
        assert db.get_backup_counter() == 0

    def test_rotation_applies_to_external_target(self, db, sample_transcript, cfg):
        """max_backups (5) also applies at the external target."""
        import time as _t

        for i in range(7):
            _t.sleep(0.01)  # unique timestamps in the filename
            db.backup(cfg)
        assert len(list(cfg.glob("*.db"))) <= 5


class TestPathAlignment:
    """realign_paths: the DB heals a media directory change on its own.

    Without this, the scanner creates NEW rows (ON CONFLICT operates on
    rel_path) when the media directory changes, instead of updating the
    existing ones — on a large library this produced thousands of duplicate
    file rows and hundreds of duplicate transcript rows. The skiplist lost its
    association in the process, so deliberately skipped files ended up back
    in the queue.
    """

    def _media_items(self, db, rel, abs_):
        db.bulk_upsert_media(
            [
                {
                    "rel_path": rel,
                    "abs_path": abs_,
                    "display_name": "x.mp4",
                    "subfolder": "",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 1,
                    "modified_at": 1.0,
                    "status": "pending",
                }
            ]
        )

    def test_realigns_after_media_dir_change(self, db, sample_transcript):
        self._media_items(db, "Games/clip.mp4", "/mnt/nas/videos/Games/clip.mp4")
        db.upsert_transcript("Games/clip.mp4", sample_transcript, media_dir="/mnt/nas/videos")
        db.add_skip("Games/clip.mp4", "manual", media_dir="/mnt/nas/videos")

        # Media directory moves up one level
        changed = db.realign_paths("/mnt/nas")
        assert changed, "nothing realigned"

        with db.connection() as conn:
            for table in ("media_files", "transcripts", "skiplist"):
                row = conn.execute(f"SELECT rel_path FROM {table}").fetchone()
                assert row["rel_path"] == "videos/Games/clip.mp4", table

    def test_skiplist_survives_the_change(self, db):
        """The real loss at stake: deliberately skipped files."""
        self._media_items(db, "Games/clip.mp4", "/mnt/nas/videos/Games/clip.mp4")
        db.add_skip("Games/clip.mp4", "no_audio", media_dir="/mnt/nas/videos")
        db.realign_paths("/mnt/nas")
        sl = db.get_skiplist()
        # The key is the identity — that stays unchanged across a root
        # change, which is exactly its purpose.
        assert "/mnt/nas/videos/Games/clip.mp4" in sl
        assert sl["/mnt/nas/videos/Games/clip.mp4"]["reason"] == "no_audio"

    def test_is_idempotent(self, db, sample_transcript):
        self._media_items(db, "Games/clip.mp4", "/mnt/nas/Games/clip.mp4")
        db.upsert_transcript("Games/clip.mp4", sample_transcript, media_dir="/mnt/nas")
        assert db.realign_paths("/mnt/nas") == {}, "unchanged DB touched"

    def test_leaves_real_duplicates_alone(self, db):
        """Where the target path is already taken, nothing is overwritten.

        A real duplicate exists there — that should be merged
        (tools/migrate_paths.py), not blindly flattened. The blocker here
        lies outside the root and therefore cannot move away.
        """
        self._media_items(db, "videos/Games/clip.mp4", "/somewhere/else/clip.mp4")
        self._media_items(db, "Games/clip.mp4", "/mnt/nas/videos/Games/clip.mp4")
        db.realign_paths("/mnt/nas")
        with db.connection() as conn:
            path_list = {r["rel_path"] for r in conn.execute("SELECT rel_path FROM media_files")}
        # Both unchanged: nothing overwritten, no UNIQUE violation
        assert path_list == {"videos/Games/clip.mp4", "Games/clip.mp4"}

    def test_converges_over_repeated_runs(self, db):
        """If a row is blocked by one still moving itself, a second run resolves it.

        Within ONE statement, the NOT-EXISTS check sees the old state — a
        row can therefore be temporarily blocked by another one that moves
        away within the same pass.
        """
        self._media_items(db, "videos/Games/clip.mp4", "/mnt/nas/elsewhere.mp4")
        self._media_items(db, "Games/clip.mp4", "/mnt/nas/videos/Games/clip.mp4")
        for _ in range(3):
            db.realign_paths("/mnt/nas")
        with db.connection() as conn:
            path_list = {r["rel_path"] for r in conn.execute("SELECT rel_path FROM media_files")}
        assert path_list == {"elsewhere.mp4", "videos/Games/clip.mp4"}

    def test_ignores_rows_outside_the_root(self, db):
        self._media_items(db, "Games/clip.mp4", "/somewhere/else/clip.mp4")
        assert db.realign_paths("/mnt/nas") == {}


def test_sync_skips_derivative_sidecars(tmp_path):
    """Start-sync finding: *_summaries.json/_translations.json are
    LISTS — the import mistook them for transcripts and logged
    an error per file ('list' object has no attribute 'get')."""
    import json as _json

    from whisper_tm.database import Database

    db = Database(tmp_path / "t.db")
    db.initialize()
    td = tmp_path / ".transcripts"
    td.mkdir()
    (td / "film.mp4.json").write_text(
        _json.dumps(
            {
                "complete": True,
                "filename": "film.mp4",
                "language": "de",
                "segments": [{"start": 0, "end": 1, "text": "Hello"}],
            }
        ),
        encoding="utf-8",
    )
    (td / "film.mp4_summaries.json").write_text(
        _json.dumps([{"text": "Summary.", "timestamp": 1}]), encoding="utf-8"
    )
    (td / "film.mp4_translations.json").write_text(
        _json.dumps([{"language": "de"}]), encoding="utf-8"
    )
    from pathlib import Path

    r = db.sync_from_json_dir(Path(td), media_dir=str(tmp_path))
    assert r["errors"] == 0, r
    assert r["imported"] == 1


def test_a_transcript_keeps_its_overlap_marker(tmp_path):
    db = Database(str(tmp_path / "library.db"))
    db.initialize()
    db.upsert_transcript(
        "a.mp4",
        {
            "segments": [
                {"start": 0, "end": 2, "text": "both talk", "overlapped": 1},
                {"start": 2, "end": 4, "text": "one talks"},
            ]
        },
        media_dir=str(tmp_path),
    )
    with db.connection() as conn:
        flags = [r[0] for r in conn.execute("SELECT overlapped FROM segments ORDER BY seg_index")]
    assert flags == [1, 0]

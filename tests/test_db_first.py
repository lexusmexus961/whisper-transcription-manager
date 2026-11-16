"""
Tests for the data policy: DB is the primary source, JSON remains as a
backup/fallback and heals the DB on a miss.

See docs/GUIDE.md → "Data policy: DB-first, JSON as backup".
"""

import json

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close_all()


SAMPLE_TDATA = {
    "filename": "video.mp4",
    "complete": True,
    "language": "de",
    "model": "turbo",
    "speakers": ["SPEAKER_00"],
    "segments": [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 2.0,
            "start_fmt": "0:00:00",
            "end_fmt": "0:00:02",
            "text": "Hello world",
        },
    ],
}


class TestSummariesDbFirst:
    def test_replace_and_get(self, db):
        db.replace_summaries(
            "a.mp4",
            [
                {"text": "S1", "prompt": "p", "model": "m", "timestamp": 1.0},
                {"text": "S2", "prompt": "", "model": "", "timestamp": 2.0},
            ],
        )
        rows = db.get_summaries("a.mp4")
        assert [r["text"] for r in rows] == ["S1", "S2"]

    def test_get_returns_none_without_entry(self, db):
        # None ≠ [] — None tells the caller to fall back to the JSON
        assert db.get_summaries("unknown.mp4") is None

    def test_replace_overwrites(self, db):
        db.replace_summaries("a.mp4", [{"text": "old_value"}])
        db.replace_summaries("a.mp4", [{"text": "new_value"}])
        rows = db.get_summaries("a.mp4")
        assert len(rows) == 1 and rows[0]["text"] == "new_value"

    def test_replace_empty_clears(self, db):
        db.replace_summaries("a.mp4", [{"text": "x"}])
        db.replace_summaries("a.mp4", [])
        # Emptied: no entry left → None (fallback semantics)
        assert db.get_summaries("a.mp4") is None

    def test_invalid_entries_skipped(self, db):
        db.replace_summaries(
            "a.mp4",
            [
                {"text": "ok"},
                {"no_text": True},
                "string",
                None,
            ],
        )
        rows = db.get_summaries("a.mp4")
        assert [r["text"] for r in rows] == ["ok"]


class TestTranscriptDbFirst:
    def test_upsert_then_get(self, db):
        db.upsert_transcript("video.mp4", SAMPLE_TDATA)
        data = db.get_transcript("video.mp4")
        assert data["segments"][0]["text"] == "Hello world"

    def test_fts_finds_upserted_segment(self, db):
        db.upsert_transcript("video.mp4", SAMPLE_TDATA)
        result = db.search("Hello")
        assert result["total"] >= 1
        assert result["results"][0]["filename"] == "video.mp4"

    def test_is_complete_from_db(self, db):
        db.upsert_transcript("video.mp4", SAMPLE_TDATA)
        assert db.is_complete("video.mp4") is True
        assert db.is_complete("missing_it.mp4") is False


class TestJsonHeal:
    """A JSON-only transcript → is_transcript_complete() heals the DB."""

    def test_json_hit_heals_db(self, db, tmp_path, monkeypatch):
        import whisper_tm.core.helpers as helpers
        import whisper_tm.extensions as ext

        # store the transcript ONLY as JSON
        tdir = tmp_path / "transcripts"
        tdir.mkdir()
        tp = tdir / "old_value.mp4.json"
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_TDATA, f)

        monkeypatch.setattr(helpers, "transcript_path_for", lambda rp, *a, **k: tdir / f"{rp}.json")
        monkeypatch.setattr(ext, "db", db)
        monkeypatch.setattr(
            helpers.config.__class__,
            "media_dir",
            property(lambda self: tmp_path),
            raising=False,
        )

        helpers.invalidate_complete_cache()
        assert db.get_transcript("old_value.mp4") is None  # before: no DB entry
        assert helpers.is_transcript_complete("old_value.mp4") is True
        healed = db.get_transcript("old_value.mp4")  # after: healed
        assert healed is not None
        assert healed["segments"][0]["text"] == "Hello world"

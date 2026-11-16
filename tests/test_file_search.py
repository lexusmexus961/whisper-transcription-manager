"""
Search over filenames and embedded titles.

Full-text search searches the SPOKEN word. Anyone searching for a series
title, a recording date, or the name of a downloaded video finds nothing
there — the term is in the filename or in the ``media_title`` from the
FFprobe metadata.

Typical case: a downloader's folder name returns no text matches but many
files, and many files carry an embedded title.
"""

from __future__ import annotations

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "search.db")
    d.initialize()
    with d.connection() as c:
        for rel, title, duration, video in [
            ("videos/game/recording_01.mp4", "Lunar Outpost Chronicles", 7200, 1),
            ("videos/game/recording_02.mp4", "Lunar Outpost Chronicles", 3600, 1),
            ("videos/yt/LmNoPqRsTuV.mp4", "How to bake bread", 300, 1),
            ("music/50%discount/song_a.mp3", "", 200, 0),
            ("music/other/tune.mp3", "Unrelated", 180, 0),
        ]:
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path, media_title, "
                "display_name, duration_sec, is_video, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                (rel, "/data/" + rel, title, rel.rsplit("/", 1)[-1], duration, video),
            )
    return d


class TestSearch:
    def test_finds_by_the_file_name(self, db):
        hits = db.search_media_files("recording_01")
        assert [t["rel_path"] for t in hits] == ["videos/game/recording_01.mp4"]

    def test_finds_by_the_embedded_title(self, db):
        """The key point: "Lunar Outpost" appears in NO filename."""
        hits = db.search_media_files("Lunar Outpost")
        assert len(hits) == 2
        assert all("recording_0" in t["rel_path"] for t in hits)

    def test_title_hits_come_first(self, db):
        """A title is the more deliberate label compared to "LmNoPqRsTuV.mp4"."""
        hits = db.search_media_files("bread")
        assert hits[0]["media_title"] == "How to bake bread"

    def test_finds_by_the_folder(self, db):
        hits = db.search_media_files("music/")
        assert len(hits) == 2

    def test_too_short_a_query_returns_nothing(self, db):
        assert db.search_media_files("a") == []
        assert db.search_media_files("") == []

    def test_limit_is_respected(self, db):
        assert len(db.search_media_files("videos", limit=2)) == 2


class TestWildcardsAreCharacters:
    """``%`` and ``_`` are allowed in names and must not act as
    LIKE wildcards (docs/PATHS.md, same rule as like_prefix)."""

    def test_percent_is_a_character(self, db):
        hits = db.search_media_files("50%discount")
        assert len(hits) == 1
        assert "50%discount" in hits[0]["rel_path"]

    def test_percent_does_not_act_as_a_wildcard(self, db):
        """`%discount` must NOT find everything that ends in "discount" …
        and certainly not everything."""
        assert db.search_media_files("videos%mp4") == []

    def test_underscore_is_a_character(self, db):
        """`_` as a wildcard would match "song_a" against "songXa" too."""
        assert len(db.search_media_files("song_a")) == 1
        assert db.search_media_files("songXa") == []


class TestTranscriptFlag:
    def test_reports_whether_a_transcript_exists(self, db):
        db.upsert_transcript(
            "videos/game/recording_01.mp4",
            {"complete": True, "segments": []},
            media_dir="/data",
            abs_path="/data/videos/game/recording_01.mp4",
        )
        by_path = {t["rel_path"]: t for t in db.search_media_files("recording")}
        assert by_path["videos/game/recording_01.mp4"]["has_transcript"]
        assert not by_path["videos/game/recording_02.mp4"]["has_transcript"]

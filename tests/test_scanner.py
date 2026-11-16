"""
Tests for scanner/ — Filesystem iteration, status resolution.
"""

import json

import pytest

from whisper_tm.core import paths
from whisper_tm.scanner.filesystem import _should_ignore, build_file_entry, iter_media_files
from whisper_tm.scanner.status import (
    _check_json,
    determine_file_status,
    resolve_transcript_status,
)


@pytest.fixture
def media_dir(tmp_path):
    """Temporary media directory with test files."""
    (tmp_path / "video1.mp4").write_text("fake video")
    (tmp_path / "video2.mkv").write_text("fake mkv")
    (tmp_path / "audio.mp3").write_text("fake audio")
    (tmp_path / "readme.txt").write_text("not media")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "deep.mp4").write_text("deep video")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.mp4").write_text("hidden")
    return tmp_path


class TestShouldIgnore:
    def test_hidden_files(self):
        assert _should_ignore(".git", [".*"])
        assert _should_ignore(".hidden", [".*"])

    def test_normal_files(self):
        assert not _should_ignore("video.mp4", [".*"])
        assert not _should_ignore("my_file.txt", [".*"])

    def test_custom_patterns(self):
        assert _should_ignore("node_modules", ["node_modules", ".*"])
        assert _should_ignore("__pycache__", ["__*", ".*"])

    def test_no_patterns(self):
        assert not _should_ignore("anything", [])


class TestIterMediaFiles:
    def test_flat_scan(self, media_dir):
        files = list(
            iter_media_files(
                media_dir,
                recursive=False,
                media_extensions={".mp4", ".mkv", ".mp3"},
            )
        )
        names = {f.name for f in files}
        assert "video1.mp4" in names
        assert "video2.mkv" in names
        assert "audio.mp3" in names
        assert "readme.txt" not in names
        assert "deep.mp4" not in names  # not recursive

    def test_recursive_scan(self, media_dir):
        files = list(
            iter_media_files(
                media_dir,
                recursive=True,
                media_extensions={".mp4", ".mkv", ".mp3"},
            )
        )
        names = {f.name for f in files}
        assert "deep.mp4" in names
        assert "secret.mp4" not in names  # .hidden ignored

    def test_no_limit_whole_tree(self, media_dir):
        """Limits removed — a recursive walk sees EVERYTHING."""
        files = list(
            iter_media_files(
                media_dir,
                recursive=True,
                media_extensions={".mp4", ".mkv", ".mp3"},
            )
        )
        names = {f.name for f in files}
        assert "deep.mp4" in names  # deepest entry comes through

    def test_transcript_dir_callback(self, media_dir):
        (media_dir / ".transcripts").mkdir()
        found = []
        list(
            iter_media_files(
                media_dir,
                recursive=False,
                media_extensions={".mp4"},
                on_transcript_dir=lambda p: found.append(p),
            )
        )
        assert len(found) == 1
        assert found[0].name == ".transcripts"

    def test_empty_dir(self, tmp_path):
        files = list(
            iter_media_files(
                tmp_path,
                media_extensions={".mp4"},
            )
        )
        assert len(files) == 0


class TestBuildFileEntry:
    def test_basic(self, media_dir):
        fp = media_dir / "video1.mp4"
        entry = build_file_entry(fp, media_dir)
        assert entry is not None
        assert entry["name"] == "video1.mp4"
        assert entry["display_name"] == "video1.mp4"
        assert entry["ext"] == ".mp4"
        assert entry["size"] > 0
        assert entry["subfolder"] == ""
        assert "_abs_path" in entry  # internal field

    def test_subdirectory(self, media_dir):
        fp = media_dir / "subdir" / "deep.mp4"
        entry = build_file_entry(fp, media_dir)
        assert entry["name"] == "subdir/deep.mp4"
        assert entry["subfolder"] == "subdir"

    def test_skiplist(self, media_dir):
        fp = media_dir / "video1.mp4"
        entry = build_file_entry(fp, media_dir, skiplist={"video1.mp4": {"reason": "no_audio"}})
        assert entry["skipped"] is True
        assert entry["skip_reason"] == "no_audio"

    def test_not_skipped(self, media_dir):
        fp = media_dir / "video1.mp4"
        entry = build_file_entry(fp, media_dir, skiplist={})
        assert entry["skipped"] is False

    def test_nonexistent_file(self, media_dir):
        fp = media_dir / "nonexistent.mp4"
        entry = build_file_entry(fp, media_dir)
        assert entry is None  # stat() fails


class TestCheckJson:
    def test_complete(self, tmp_path):
        tp = tmp_path / "test.json"
        tp.write_text(json.dumps({"complete": True, "segments": []}))
        complete, partial = _check_json(tp)
        assert complete is True
        assert partial is False

    def test_incomplete(self, tmp_path):
        tp = tmp_path / "test.json"
        tp.write_text(json.dumps({"complete": False}))
        complete, partial = _check_json(tp)
        assert complete is False
        assert partial is True

    def test_invalid_json(self, tmp_path):
        tp = tmp_path / "test.json"
        tp.write_text("not json at all")
        complete, partial = _check_json(tp)
        assert complete is False
        assert partial is True  # broken file = partial


class TestDetermineFileStatus:
    def test_complete(self):
        assert determine_file_status(True, False, False) == "done"

    def test_skipped(self):
        assert determine_file_status(False, False, True) == "skipped"

    def test_partial(self):
        assert determine_file_status(False, True, False) == "partial"

    def test_pending(self):
        assert determine_file_status(False, False, False) == "pending"

    def test_complete_overrides_skipped(self):
        """complete has highest priority."""
        assert determine_file_status(True, False, True) == "done"

    def test_skipped_overrides_partial(self):
        assert determine_file_status(False, True, True) == "skipped"


class TestResolveTranscriptStatus:
    @pytest.fixture(autouse=True)
    def _load_config(self):
        """Config must be loaded so transcript_path_for works."""
        from whisper_tm.config import config

        if not config.is_loaded:
            config.load()

    def test_no_transcript(self, media_dir):
        fp = media_dir / "video1.mp4"
        complete, partial = resolve_transcript_status("video1.mp4", fp)
        assert complete is False
        assert partial is False

    def test_local_transcript(self, media_dir):
        """Local .transcripts/ next to the file."""
        fp = media_dir / "video1.mp4"
        tx_dir = media_dir / ".transcripts"
        tx_dir.mkdir()
        tx = tx_dir / "video1.mp4.json"
        tx.write_text(json.dumps({"complete": True, "segments": []}))
        complete, partial = resolve_transcript_status("video1.mp4", fp)
        assert complete is True

    def test_local_incomplete(self, media_dir):
        fp = media_dir / "video1.mp4"
        tx_dir = media_dir / ".transcripts"
        tx_dir.mkdir()
        tx = tx_dir / "video1.mp4.json"
        tx.write_text(json.dumps({"complete": False}))
        complete, partial = resolve_transcript_status("video1.mp4", fp)
        assert complete is False
        assert partial is True

    def test_completion_set_hit(self, media_dir):
        """Pre-loaded completion_set is checked before JSON.

        The set is keyed on abs_path — the identity (docs/PATHS.md).
        """
        fp = media_dir / "video1.mp4"
        cset = {paths.canon(fp), paths.canon(media_dir / "other.mp4")}
        complete, partial = resolve_transcript_status("video1.mp4", fp, completion_set=cset)
        assert complete is True

    def test_completion_set_miss(self, media_dir):
        """File not in completion_set → not complete."""
        fp = media_dir / "video1.mp4"
        cset = {paths.canon(media_dir / "other.mp4")}
        complete, partial = resolve_transcript_status("video1.mp4", fp, completion_set=cset)
        assert complete is False

    def test_completion_set_does_not_match_on_the_basename(self, media_dir):
        """A mere name match does NOT count (docs/PATHS.md).

        Previously subdir/deep.mp4 was considered done as soon as some deep.mp4
        was done. With same-named files in multiple folders — Intro,
        Episode01, track01 — the scan reported files as transcribed that
        never were.
        """
        (media_dir / "subdir").mkdir(exist_ok=True)
        fp = media_dir / "subdir" / "deep.mp4"
        fp.write_text("test")
        # different folder, same name → different identity
        cset = {paths.canon(media_dir / "deep.mp4")}
        complete, partial = resolve_transcript_status("subdir/deep.mp4", fp, completion_set=cset)
        assert complete is False

    def test_completion_set_matches_the_full_path(self, media_dir):
        """Counter-check: with the right identity it still works."""
        (media_dir / "subdir").mkdir(exist_ok=True)
        fp = media_dir / "subdir" / "deep.mp4"
        fp.write_text("test")
        complete, partial = resolve_transcript_status(
            "subdir/deep.mp4", fp, completion_set={paths.canon(fp)}
        )
        assert complete is True

    def test_completion_set_ignores_rel_path(self, media_dir):
        """The display value alone is NOT enough anymore.

        This is exactly what the bug hung on: the same rel_path stands for
        different files depending on the scan root (docs/PATHS.md).
        """
        fp = media_dir / "video1.mp4"
        complete, _ = resolve_transcript_status("video1.mp4", fp, completion_set={"video1.mp4"})
        assert complete is False

    def test_completion_set_skips_json(self, media_dir):
        """When completion_set is passed, no JSON is read."""
        fp = media_dir / "video1.mp4"
        # JSON says "incomplete"
        tx_dir = media_dir / ".transcripts"
        tx_dir.mkdir(exist_ok=True)
        tx = tx_dir / "video1.mp4.json"
        tx.write_text(json.dumps({"complete": False}))
        # But completion_set says "complete" → DB wins
        cset = {paths.canon(fp)}
        complete, partial = resolve_transcript_status("video1.mp4", fp, completion_set=cset)
        assert complete is True

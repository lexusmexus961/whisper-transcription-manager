"""
Tests for scanner/watcher.py — Filesystem watcher.
"""

import time

import pytest

from whisper_tm.scanner.watcher import FileFingerprint, FileWatcher


@pytest.fixture
def media_dir(tmp_path):
    """Media directory with test files."""
    (tmp_path / "video1.mp4").write_text("fake video 1")
    (tmp_path / "video2.mkv").write_text("fake video 2")
    return tmp_path


class TestFileFingerprint:
    def test_equality(self):
        a = FileFingerprint(size=100, mtime=1234.0)
        b = FileFingerprint(size=100, mtime=1234.0)
        assert a == b

    def test_inequality_size(self):
        a = FileFingerprint(size=100, mtime=1234.0)
        b = FileFingerprint(size=200, mtime=1234.0)
        assert a != b


class TestFileWatcher:
    @pytest.fixture(autouse=True)
    def _load_config(self):
        from whisper_tm.config import config

        if not config.is_loaded:
            config.load()

    def test_initial_scan(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir, interval=999)
        watcher._fingerprints = watcher._scan_fingerprints()
        assert len(watcher._fingerprints) == 2
        assert "video1.mp4" in watcher._fingerprints

    def test_detect_added(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir)
        watcher._fingerprints = watcher._scan_fingerprints()

        # add new file
        (media_dir / "video3.mp4").write_text("new video")
        events = watcher._detect_changes()

        added = [e for e in events if e[0] == "added"]
        assert len(added) == 1
        assert added[0][1] == "video3.mp4"

    def test_detect_removed(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir)
        watcher._fingerprints = watcher._scan_fingerprints()

        # delete file
        (media_dir / "video1.mp4").unlink()
        events = watcher._detect_changes()

        removed = [e for e in events if e[0] == "removed"]
        assert len(removed) == 1
        assert removed[0][1] == "video1.mp4"

    def test_detect_modified(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir)
        watcher._fingerprints = watcher._scan_fingerprints()

        # modify file (size changes)
        (media_dir / "video1.mp4").write_text("fake video 1 - modified with more content")
        events = watcher._detect_changes()

        modified = [e for e in events if e[0] == "modified"]
        assert len(modified) == 1
        assert modified[0][1] == "video1.mp4"

    def test_no_changes(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir)
        watcher._fingerprints = watcher._scan_fingerprints()
        events = watcher._detect_changes()
        assert len(events) == 0

    def test_callback(self, media_dir):
        received_events = []

        def on_change(event_type, rel_path, abs_path):
            received_events.append((event_type, rel_path))

        watcher = FileWatcher(media_dir=media_dir, on_change=on_change)
        watcher._fingerprints = watcher._scan_fingerprints()

        (media_dir / "new.mp4").write_text("new")
        events = watcher._detect_changes()
        watcher._handle_events(events)

        assert len(received_events) == 1
        assert received_events[0] == ("added", "new.mp4")

    def test_start_stop(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir, interval=0.5)
        assert not watcher.is_running

        watcher.start()
        assert watcher.is_running
        time.sleep(0.1)

        watcher.stop()
        assert not watcher.is_running

    def test_status(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir, interval=10)
        status = watcher.status()
        assert status["running"] is False
        assert str(media_dir) in status["media_dir"]
        assert status["interval_sec"] == 10

    def test_force_scan(self, media_dir):
        watcher = FileWatcher(media_dir=media_dir)
        watcher._fingerprints = watcher._scan_fingerprints()

        (media_dir / "forced.mp4").write_text("forced")
        events = watcher.force_scan()

        assert len(events) == 1
        assert events[0][0] == "added"

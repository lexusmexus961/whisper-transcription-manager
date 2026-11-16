"""A file found corrupt while it was still being copied gets another chance.

A video queued while its copy stood still was found corrupt and put on the
skip list — for good: the entry kept no time to compare with, so the
finished file stayed skipped. Now a "corrupt" entry lapses once the file
changed after the verdict; the queue, the library scan and continuous
scanning then treat the file as one to transcribe.
"""

from __future__ import annotations

import os
import time

import pytest

import whisper_tm.extensions as ext
from whisper_tm.config import config
from whisper_tm.core import autoscan as au
from whisper_tm.core import helpers
from whisper_tm.database import Database
from whisper_tm.scanner.filesystem import build_file_entry


@pytest.fixture
def env(tmp_path, monkeypatch):
    media = tmp_path / "media"
    (media / "a").mkdir(parents=True)
    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(helpers, "_skiplist_path", lambda: tmp_path / "skiplist.json")
    helpers.invalidate_skiplist_cache()
    yield {"media": media, "db": db, "tmp": tmp_path}
    helpers.invalidate_skiplist_cache()


def _written(path, ago: float, data: bytes = b"x" * 100):
    path.write_bytes(data)
    t = time.time() - ago
    os.utime(path, (t, t))
    return path


def _found_corrupt(env, rel: str, path, ago: float, reason: str = "corrupt"):
    """The run's verdict, ``ago`` seconds back."""
    helpers.add_to_skiplist(rel, reason, abs_path=path)
    with env["db"].connection() as conn:
        conn.execute("UPDATE skiplist SET added_at = ?", (time.time() - ago,))
    helpers.invalidate_skiplist_cache()


def _copy_goes_on(path):
    with open(path, "ab") as f:
        f.write(b"rest of the file")


class TestTheQueue:
    def test_a_file_that_changed_after_the_verdict_is_tried_again(self, env):
        f = _written(env["media"] / "a" / "copying.mp4", ago=300)
        _found_corrupt(env, "a/copying.mp4", f, ago=120)
        assert helpers.is_skipped("a/copying.mp4", media_dir=env["media"])
        _copy_goes_on(f)
        assert not helpers.is_skipped("a/copying.mp4", media_dir=env["media"])
        assert env["db"].get_skiplist() == {}  # the entry is gone

    def test_an_unchanged_broken_file_stays_skipped(self, env):
        f = _written(env["media"] / "a" / "broken.mp4", ago=300)
        _found_corrupt(env, "a/broken.mp4", f, ago=120)
        for _ in range(2):
            helpers.invalidate_skiplist_cache()
            assert helpers.is_skipped("a/broken.mp4", media_dir=env["media"])

    def test_the_entry_just_written_holds(self, env):
        """Straight after the verdict, from the cache add_to_skiplist fills."""
        f = _written(env["media"] / "a" / "broken.mp4", ago=300)
        helpers.add_to_skiplist("a/broken.mp4", "corrupt", abs_path=f)
        assert helpers.is_skipped("a/broken.mp4", media_dir=env["media"])

    def test_only_corrupt_entries_lapse(self, env):
        """A skip by hand or for having no sound stays, whatever the file does."""
        f = _written(env["media"] / "a" / "silent.mp4", ago=300)
        _found_corrupt(env, "a/silent.mp4", f, ago=120, reason="no_audio")
        _copy_goes_on(f)
        assert helpers.is_skipped("a/silent.mp4", media_dir=env["media"])


class TestTheCheck:
    def test_times(self):
        now = time.time()
        entry = {"reason": "corrupt", "added_at": now - 600}
        assert not helpers.skip_lapsed("/x.mp4", entry, now - 900)  # before it
        assert not helpers.skip_lapsed("/x.mp4", entry, now - 600)  # the same time
        assert helpers.skip_lapsed("/x.mp4", entry, now - 60)  # after it
        # A NAS clock ahead of this one: counts once the time is reached
        assert not helpers.skip_lapsed("/x.mp4", entry, now + 3600)

    def test_an_entry_without_a_time_holds(self):
        """The JSON list of a run without a database kept only a text time."""
        old = {"reason": "corrupt", "time": "2026-01-01 00:00:00"}
        assert not helpers.skip_lapsed("/x.mp4", old, time.time())
        assert not helpers.skip_lapsed("/x.mp4", "corrupt", time.time())

    def test_a_file_that_is_gone_holds(self, tmp_path):
        entry = {"reason": "corrupt", "added_at": time.time() - 600}
        assert not helpers.skip_lapsed(str(tmp_path / "gone.mp4"), entry)


class TestTheLibraryScan:
    def test_it_is_listed_to_transcribe_again(self, env):
        media = env["media"]
        f = _written(media / "a" / "copying.mp4", ago=300)
        _found_corrupt(env, "a/copying.mp4", f, ago=120)
        assert build_file_entry(f, media, skiplist=helpers.load_skiplist())["skipped"]
        _copy_goes_on(f)
        entry = build_file_entry(f, media, skiplist=helpers.load_skiplist())
        assert not entry["skipped"] and env["db"].get_skiplist() == {}


class TestContinuousScanning:
    @pytest.fixture
    def scan(self, env, monkeypatch):
        monkeypatch.setitem(
            config._data.setdefault("security", {}), "allowed_roots", [str(env["media"])]
        )
        monkeypatch.setattr(type(config), "data_dir", property(lambda self: env["tmp"] / "data"))
        a = config._data.setdefault("autoscan", {})
        for k, v in {
            "enabled": False,
            "transcribe": True,
            "interval_min": 15,
            "folders": [],
            "settle_min": 0,
            "full_every_h": 6,
        }.items():
            monkeypatch.setitem(a, k, v)
        queued = []
        monkeypatch.setattr(
            "whisper_tm.core.queue.enqueue",
            lambda names, settings, base_dir=None, **k: queued.append(list(names)) or [{"id": 1}],
        )
        monkeypatch.setattr(
            "whisper_tm.scanner.register_single_file",
            lambda p, db=None, root=None: os.path.relpath(p, root),
        )
        monkeypatch.setattr(
            "whisper_tm.core.helpers.is_transcript_complete", lambda rel, media_dir=None: False
        )
        return {"scan": au.AutoScan(), "queued": queued}

    def test_a_copy_that_stood_still_is_queued_again(self, env, scan):
        s, media = scan["scan"], env["media"]
        s.round()  # the starting point
        f = _written(media / "a" / "copying.mp4", ago=300)
        t = time.time() + 5
        os.utime(media / "a", (t, t))
        s.round()  # seen
        assert s.round()["queued"] == 1  # settled: queued
        _found_corrupt(env, "a/copying.mp4", f, ago=120)  # … half-written
        assert s.round()["pending"] == 0  # nothing changed yet
        _copy_goes_on(f)  # the copy goes on
        assert s.round()["pending"] == 1
        assert s.state().counts() == {"pending": 1}
        assert s.round()["queued"] == 1  # settled again
        assert scan["queued"] == [["a/copying.mp4"], ["a/copying.mp4"]]

    def test_a_broken_file_is_not_queued_again(self, env, scan):
        s, media = scan["scan"], env["media"]
        s.round()
        f = _written(media / "a" / "broken.mp4", ago=300)
        t = time.time() + 5
        os.utime(media / "a", (t, t))
        s.round()
        s.round()
        _found_corrupt(env, "a/broken.mp4", f, ago=120)
        assert s.round()["pending"] == 0 and s.round()["queued"] == 0
        assert scan["queued"] == [["a/broken.mp4"]]

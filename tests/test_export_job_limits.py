"""Export and clip jobs have limits: a number that may wait, free space
before one starts, and results tidied by the clock, not only by the next
job."""

import time
from collections import namedtuple

import pytest

from whisper_tm.core import export_jobs as jobs


@pytest.fixture
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "folder", lambda: tmp_path)
    monkeypatch.setattr(jobs, "_jobs", {})
    yield tmp_path


def test_too_many_waiting_are_refused(clean):
    for i in range(jobs.MAX_WAITING):
        jobs._jobs[f"w{i}"] = jobs.Job(id=f"w{i}", label="x", total=1)
    with pytest.raises(jobs.Refused) as e:
        jobs.start("one more", [1], lambda *a: [])
    assert e.value.status == 429


def test_a_nearly_full_disk_is_refused(clean, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda p: Usage(10, 9, 100 * 1024**2))
    with pytest.raises(jobs.Refused) as e:
        jobs.start("big", [1], lambda *a: [])
    assert e.value.status == 507 and "GB" in str(e.value)


def test_old_results_go_without_a_new_job(clean):
    old = clean / "abc"
    old.mkdir()
    t = time.time() - jobs.KEEP_SECONDS - 10
    import os

    os.utime(old, (t, t))
    fresh = clean / "def"
    fresh.mkdir()
    jobs.tidy()
    assert not old.exists() and fresh.exists()

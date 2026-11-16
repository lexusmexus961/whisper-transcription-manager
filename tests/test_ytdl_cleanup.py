"""yt-dlp downloads: what a cancel leaves, and what reaches yt-dlp as a link
(core/ytdl.py).

* A cancelled or timed-out download left its partial ``ytdl_<id>*`` files
  in the work folder until the next server start — multi-GB per video —
  and a killed yt-dlp was never waited for (a zombie), its ffmpeg child
  left running.
* The link was the last argument with nothing before it: a playlist entry
  that came back as a bare video id beginning with "-" (YouTube ids do)
  was read by yt-dlp as an option.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from whisper_tm.core import ytdl


class FakeProc:
    """A yt-dlp that writes a partial file and never finishes."""

    def __init__(self, cmd, **kw):
        self.cmd, self.kw = cmd, kw
        out = Path(cmd[cmd.index("--output") + 1])
        Path(str(out).replace("%(ext)s", "mp4.part")).write_bytes(b"x" * 5000)
        self.stdout = io.StringIO("[download]  10.0% of 50MiB\n")
        self.pid = 4242
        self.returncode = None
        self.waited = False
        FakeProc.last = self

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        self.returncode = self.returncode if self.returncode is not None else -9
        return self.returncode


class Job:
    cancel = False


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(ytdl.subprocess, "Popen", FakeProc)
    groups = []
    monkeypatch.setattr(
        ytdl, "_kill_group", lambda proc, kill=False: groups.append("kill" if kill else "term")
    )
    monkeypatch.setattr(ytdl.time, "sleep", lambda s: None)
    tmp, dest = tmp_path / "tmp", tmp_path / "dest"
    tmp.mkdir()
    dest.mkdir()
    return tmp, dest, groups


def test_a_cancel_leaves_no_partial_file(fake):
    tmp, dest, groups = fake
    job = Job()
    job.cancel = True
    with pytest.raises(InterruptedError):
        ytdl._download_one(
            job, {"url": "https://youtu.be/abc", "title": "t", "id": "abc"}, tmp, dest
        )
    assert list(tmp.iterdir()) == []
    assert FakeProc.last.waited and groups


def test_a_timeout_leaves_no_partial_file_and_waits(fake, monkeypatch):
    tmp, dest, groups = fake
    monkeypatch.setattr(ytdl, "_DL_TIMEOUT", -1)
    with pytest.raises(RuntimeError, match="Timeout"):
        ytdl._download_one(
            Job(), {"url": "https://youtu.be/abc", "title": "t", "id": "abc"}, tmp, dest
        )
    assert list(tmp.iterdir()) == []
    assert FakeProc.last.waited


def test_its_own_process_group(fake):
    tmp, dest, _ = fake
    job = Job()
    job.cancel = True
    with pytest.raises(InterruptedError):
        ytdl._download_one(
            job, {"url": "https://youtu.be/abc", "title": "t", "id": "abc"}, tmp, dest
        )
    assert FakeProc.last.kw.get("start_new_session") is True


def test_the_link_comes_after_a_double_dash(fake):
    tmp, dest, _ = fake
    job = Job()
    job.cancel = True
    with pytest.raises(InterruptedError):
        ytdl._download_one(
            job, {"url": "-abc_DEF123", "title": "t", "id": "-abc_DEF123"}, tmp, dest
        )
    cmd = FakeProc.last.cmd
    assert cmd[-2] == "--"
    assert cmd[-1] == "https://www.youtube.com/watch?v=-abc_DEF123"


def test_enumeration_too(monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=('{"_type": "playlist", "entries": [{"id": "-xyzABC1234", "title": "a"}]}'),
            stderr="",
        )

    monkeypatch.setattr(ytdl.subprocess, "run", run)
    entries = ytdl.enumerate_entries("https://www.youtube.com/playlist?list=PL1")
    assert seen[0][-2:] == ["--", "https://www.youtube.com/playlist?list=PL1"]
    assert entries[0]["url"] == "https://www.youtube.com/watch?v=-xyzABC1234"

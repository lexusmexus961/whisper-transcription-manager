"""
Signals to a child's process group (core/procgroup.py).

The model servers and yt-dlp run in a session of their own, so a signal to
the group reaches their children too. Windows has no process groups in that
sense: os.killpg, os.getpgid and SIGKILL are missing, and the AttributeError
skipped everything after a stop.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from whisper_tm.core import procgroup

posix = pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX")


def _child() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )


@posix
@pytest.mark.parametrize("kill", [False, True])
def test_the_group_gets_the_signal(kill):
    p = _child()
    try:
        assert procgroup.signal_group(p.pid, kill=kill) is True
        assert p.wait(timeout=10) != 0
    finally:
        p.kill()


def test_without_process_groups_it_says_so(monkeypatch):
    # Above Linux' highest PID (2**22): without groups the children of the
    # PID are ended — never pick a number that may be a stranger's process
    monkeypatch.delattr(os, "killpg", raising=False)
    assert procgroup.signal_group(2**22 + 1) is False
    assert procgroup.signal_group(2**22 + 1, kill=True) is False


@posix
def test_a_gone_process_is_no_error():
    p = _child()
    p.kill()
    p.wait(timeout=10)
    assert procgroup.signal_group(p.pid) is False


def test_end_pid_falls_back_to_the_process(monkeypatch):
    p = _child()
    try:
        monkeypatch.delattr(os, "killpg", raising=False)
        procgroup.end_pid(p.pid)
        assert p.wait(timeout=10) != 0
    finally:
        p.kill()

"""A second server must not clean out the first server's workdir.

The scenario: a test server on another port gets a COPY of the
config — including ``work: ~/.cache/whisper-tm/work``. At startup,
``create_app()`` calls ``clean_workdir()``, and it deletes everything there
recursively. The production server's running job (a long recording, in the
middle of transcription) dies seconds later with

    Failed to load audio: … asr_block_001.wav: No such file or directory

The existing checks in ``clean_workdir`` couldn't see this: they protect USER DATA
from misconfiguration (work_dir == media_dir and similar). Here both servers were
configured correctly — they shared the directory completely legitimately. What was
missing was the question of whether someone was WORKING there right now.

The general lesson: a throwaway server next to a running system inherits more
than it should.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from whisper_tm.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path: Path) -> ConfigManager:
    """A configuration whose directories all live in tmp_path."""
    import yaml

    file = tmp_path / "config.yaml"
    file.write_text(
        yaml.safe_dump(
            {
                "directories": {
                    "media": str(tmp_path / "media"),
                    "work": str(tmp_path / "work"),
                    "transcripts": str(tmp_path / "tx"),
                },
            }
        ),
        encoding="utf-8",
    )
    for name in ("media", "work", "tx"):
        (tmp_path / name).mkdir()
    c = ConfigManager()
    c.load(file)
    return c


def _work_file(cfg: ConfigManager) -> Path:
    folder = cfg.work_dir / "videos_something_mp4_abc123"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "asr_block_001.wav"
    file.write_bytes(b"sound")
    return file


def test_without_a_marker_it_cleans_up(cfg):
    """Normal case remains: a restart cleans up leftovers."""
    file = _work_file(cfg)
    cfg.clean_workdir()
    assert not file.exists()
    assert not file.parent.exists()


def test_a_foreign_live_server_prevents_the_cleanup(cfg, tmp_path):
    """Real protection — on a TRULY running process.

    A made-up PID is not enough: we're checking if there's still something
    behind the number that looks like this program. So a real child process
    whose command line proves it.
    """
    import subprocess
    import sys

    file = _work_file(cfg)
    script = tmp_path / "whisper_tm_server.py"
    script.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(script)])
    try:
        # Right after the fork the child still carries this process's
        # command line — until its exec. Under load that window was long
        # enough for the cleanup to see "not a server" and delete.
        import time

        import psutil

        deadline = time.monotonic() + 10
        while script.name not in " ".join(psutil.Process(child.pid).cmdline()):
            assert time.monotonic() < deadline, "the child never started"
            time.sleep(0.02)
        (cfg.work_dir / cfg.USAGE_FILE).write_text(
            json.dumps({"pid": child.pid, "cmdline": f"{sys.executable} {script}"}),
            encoding="utf-8",
        )
        cfg.clean_workdir()
        assert file.exists(), "the running server's work file is gone"
    finally:
        child.kill()
        child.wait(timeout=10)


def test_your_own_marker_does_not_block(cfg):
    """Otherwise a server could lock itself out on restart
    — the workdir would then grow without bound."""
    file = _work_file(cfg)
    cfg.claim_workdir()
    cfg.clean_workdir()
    assert not file.exists()


def test_a_dead_pid_does_not_block(cfg):
    """After a crash the mark stays. It must not prevent cleanup forever."""
    file = _work_file(cfg)
    # PID that doesn't exist: beyond the maximum.
    (cfg.work_dir / cfg.USAGE_FILE).write_text(
        json.dumps({"pid": 4_000_000, "cmdline": "python -m whisper_tm"}), encoding="utf-8"
    )
    cfg.clean_workdir()
    assert not file.exists()


def test_a_broken_marker_does_not_block(cfg):
    """A half-written file is not a reason to never clean up again."""
    file = _work_file(cfg)
    (cfg.work_dir / cfg.USAGE_FILE).write_text("{no json", encoding="utf-8")
    cfg.clean_workdir()
    assert not file.exists()


def test_a_foreign_pid_without_whisperx_does_not_block(cfg):
    """A newly assigned PID belongs to some program. Just because the
    number matches doesn't mean a server is working there."""
    file = _work_file(cfg)
    (cfg.work_dir / cfg.USAGE_FILE).write_text(
        json.dumps({"pid": 1, "cmdline": "/sbin/init"}), encoding="utf-8"
    )
    cfg.clean_workdir()
    assert not file.exists()


def test_the_marker_is_written_after_the_cleanup(cfg):
    """Order is meaning here: the mark lies IN the workdir.
    Written before cleanup, it would be deleted immediately."""
    source = (Path(__file__).resolve().parent.parent / "whisper_tm" / "app.py").read_text(
        encoding="utf-8"
    )
    cleanup = source.index("config.clean_workdir()")
    occupy = source.index("config.claim_workdir()")
    assert cleanup < occupy  # cleanup comes before marking


def test_the_marker_survives_your_own_cleanup(cfg):
    """After cleanup + marking, the mark must be there — otherwise it won't
    protect the next startup."""
    _work_file(cfg)
    cfg.clean_workdir()
    cfg.claim_workdir()
    marker = cfg.work_dir / cfg.USAGE_FILE
    assert marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_a_second_app_leaves_a_live_servers_marker_alone(cfg, tmp_path):
    """create_app() cleans, then claims — twice in one test process.

    Once the first claim wrote the test's PID over the running server's;
    the second clean took the workdir for its own and deleted the running
    job's files, and the running jobs failed.
    """
    import subprocess
    import sys

    file = _work_file(cfg)
    script = tmp_path / "whisper_tm_server.py"
    script.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(script)])
    try:
        marker = cfg.work_dir / cfg.USAGE_FILE
        marker.write_text(
            json.dumps({"pid": child.pid, "cmdline": f"{sys.executable} {script}"}),
            encoding="utf-8",
        )
        for _ in range(2):  # two create_app() in one process
            cfg.clean_workdir()
            cfg.claim_workdir()
        assert json.loads(marker.read_text(encoding="utf-8"))["pid"] == child.pid
        assert file.exists(), "the running server's work file is gone"
    finally:
        child.kill()
        child.wait(timeout=10)


def test_the_cleanup_runs_before_the_queue_worker_starts():
    """The worker used to start first: a job it picked up in between lost
    its work files to the cleanup of its own server."""
    source = (Path(__file__).resolve().parent.parent / "whisper_tm" / "app.py").read_text(
        encoding="utf-8"
    )
    assert source.index("config.clean_workdir()") < source.index("queue_worker.start_worker()")

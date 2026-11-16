"""
Cleaning the work folder from the settings page.

``POST /api/clean_workdir`` deleted everything in the work folder: the files
of the job running right now — it died with "No such file or directory" —
and this server's marker ``.server-busy.json``, which nothing wrote back,
so a second server started afterwards took the folder as free. And it
answered ``ok`` even when ``clean_workdir()`` had refused to delete a thing.

The startup side of the marker (clean first, then claim) is covered in
test_workdir_usage.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from whisper_tm.config import ConfigManager, config


@pytest.fixture()
def cfg(tmp_path: Path) -> ConfigManager:
    """A configuration whose directories all live in tmp_path."""
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


def _job_folders(work: Path, *names: str) -> list[Path]:
    """Work folders as transcription_worker leaves them behind."""
    out = []
    for name in names:
        folder = work / name
        folder.mkdir(parents=True)
        (folder / "asr_block_001.wav").write_bytes(b"audio")
        out.append(folder)
    return out


# ── config.clean_workdir() ────────────────────────────────────────────────


def test_it_reports_what_it_removed_and_keeps_the_marker(cfg):
    cfg.claim_workdir()
    _job_folders(cfg.work_dir, "a_mp4_1", "b_mp4_2")
    (cfg.work_dir / "stray.tmp").write_bytes(b"x")

    result = cfg.clean_workdir()
    assert result == {"ok": True, "refused": False, "removed": 3, "failed": 0}
    # Both markers stay: the busy one and the one that makes it a work folder
    assert sorted(p.name for p in cfg.work_dir.iterdir()) == sorted(
        [cfg.USAGE_FILE, cfg.WORK_MARKER]
    )
    marker = json.loads((cfg.work_dir / cfg.USAGE_FILE).read_text(encoding="utf-8"))
    assert marker["pid"] == os.getpid()


def test_a_job_that_starts_meanwhile_stops_the_cleanup(cfg):
    """The busy question comes before every entry, not once at the start."""
    _job_folders(cfg.work_dir, "a", "b", "c")
    answers = iter(["", "The queue is processing new.mp4"])

    result = cfg.clean_workdir(busy=lambda: next(answers))
    assert result["ok"] is False and result["refused"] is True
    assert result["removed"] == 1
    assert result["error"] == "The queue is processing new.mp4"
    assert len([p for p in cfg.work_dir.iterdir() if p.name != cfg.WORK_MARKER]) == 2


def test_a_refusal_says_why(cfg, tmp_path, monkeypatch):
    """A work folder that contains the media folder: nothing may go."""
    monkeypatch.setattr(cfg, "_resolved_work", tmp_path)
    result = cfg.clean_workdir()
    assert result["ok"] is False and result["refused"] is True
    assert result["removed"] == 0
    assert "overlaps" in result["error"]
    assert (tmp_path / "media").is_dir()


def test_an_entry_that_would_not_go_is_reported(cfg, monkeypatch):
    _job_folders(cfg.work_dir, "stuck", "loose")
    real = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path).name == "stuck":
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    result = cfg.clean_workdir()
    assert result["ok"] is False and result["refused"] is False
    assert (result["removed"], result["failed"]) == (1, 1)
    assert "stuck" in result["error"]


def test_a_link_goes_but_not_what_it_points_to(cfg, tmp_path):
    keep = tmp_path / "elsewhere"
    keep.mkdir()
    (keep / "precious.txt").write_text("user data", encoding="utf-8")
    (cfg.work_dir / "link").symlink_to(keep, target_is_directory=True)

    result = cfg.clean_workdir()
    assert result["ok"] is True and result["removed"] == 1
    assert not (cfg.work_dir / "link").exists()
    assert (keep / "precious.txt").exists()


# ── POST /api/clean_workdir ───────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An app on a throwaway config and database; the queue stays paused."""
    monkeypatch.setenv("HOME", str(tmp_path))
    own = tmp_path / "config.yaml"
    own.write_text("queue:\n  autostart: false\n  lanes: 1\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "media")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "library.db")
    config._data["database"]["auto_backup"] = False
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client()  # cleans and claims the work folder
    monkeypatch.undo()
    config.load(None)


def test_the_api_refuses_while_the_queue_processes_a_file(client, monkeypatch):
    from whisper_tm.core import queue as q

    (running,) = _job_folders(config.work_dir, "long_recording_mp4_1a2b3c4d")
    monkeypatch.setattr(
        q,
        "_current",
        {
            "id": 1,
            "filename": "long recording.mp4",
            "dir": str(config.media_dir),
            "job_id": "long_recording_mp4_1a2b3c4d",
        },
    )

    r = client.post("/api/clean_workdir")
    assert r.status_code == 409
    d = r.get_json()
    assert d["ok"] is False and d["refused"] is True
    assert "long recording.mp4" in d["error"]
    assert (running / "asr_block_001.wav").exists(), "the running job's files are gone"


def test_the_refusal_names_how_many_more_run_beside_it(client, monkeypatch):
    """Queue lanes: three files run at once — the oldest is named, the
    other two counted."""
    from whisper_tm.core import queue as q

    (running,) = _job_folders(config.work_dir, "a_mp4_1a2b3c4d")
    monkeypatch.setattr(
        q,
        "snapshot",
        lambda: {
            "total": 5,
            "processed": 0,
            "pending": 2,
            "current_item_id": 1,
            "current_file": "a.mp4",
            "running_count": 3,
        },
    )

    r = client.post("/api/clean_workdir")
    assert r.status_code == 409
    error = r.get_json()["error"]
    assert "a.mp4 (+2 more)" in error and "running jobs have stopped" in error
    assert (running / "asr_block_001.wav").exists()


def test_the_api_says_what_it_removed_and_keeps_the_marker(client):
    _job_folders(config.work_dir, "old_job_1", "old_job_2")
    r = client.post("/api/clean_workdir")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "refused": False, "removed": 2, "failed": 0}
    marker = json.loads((config.work_dir / config.USAGE_FILE).read_text(encoding="utf-8"))
    assert marker["pid"] == os.getpid()


def _exec_done(proc, marker: str, timeout: float = 10.0) -> None:
    """Wait until the child's command line is its own. Right after Popen the
    exec may not be through — under load /proc/<pid>/cmdline was still
    empty, the check under test rightly took the PID for a foreign process,
    and this test failed now and then in the full suite."""
    import time

    import psutil

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if marker in " ".join(psutil.Process(proc.pid).cmdline()):
                return
        except psutil.Error:
            pass
        time.sleep(0.01)
    raise AssertionError(f"PID {proc.pid} never showed {marker!r} in its command line")


def test_a_refusal_is_not_answered_with_ok(client, tmp_path):
    """Another live server holds the folder: nothing goes, and the answer
    says so."""
    (leftover,) = _job_folders(config.work_dir, "its_job")
    script = tmp_path / "whisper_tm_server.py"
    script.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    other = subprocess.Popen([sys.executable, str(script)])
    try:
        _exec_done(other, script.name)
        (config.work_dir / config.USAGE_FILE).write_text(
            json.dumps({"pid": other.pid, "cmdline": f"{sys.executable} {script}"}),
            encoding="utf-8",
        )
        r = client.post("/api/clean_workdir")
        assert r.status_code == 409
        d = r.get_json()
        assert d["ok"] is False and d["refused"] is True
        assert "Another server" in d["error"]
        assert leftover.exists()
    finally:
        other.kill()
        other.wait(timeout=10)


def test_the_holder_is_recognised_by_this_apps_name_alone(tmp_path, monkeypatch):
    """The check looked for 'whisperx' — true only where the install path
    happened to contain it; a real install runs 'python -m whisper_tm'."""
    import psutil

    class P:
        def __init__(self, line):
            self._l = line

        def create_time(self):
            return 0.0

        def cmdline(self):
            return self._l.split()

    (config.work_dir).mkdir(parents=True, exist_ok=True)
    (config.work_dir / config.USAGE_FILE).write_text(json.dumps({"pid": 999999}), encoding="utf-8")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    for line, held in (
        ("/opt/app/.venv/bin/python -m whisper_tm --port 9860", True),
        ("/usr/bin/whisper-tm", True),
        ("/usr/bin/python3 -m http.server", False),
    ):
        monkeypatch.setattr(psutil, "Process", lambda pid, _l=line: P(_l))
        assert bool(config._workdir_taken_by_other()) is held, line

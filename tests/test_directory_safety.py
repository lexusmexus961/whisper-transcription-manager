"""
Directory settings the page may change, and what the app then does with them.

Background:

* ``directories.logs`` could be set to "/" from the settings page — and
  ``/api/logs/view/<path>`` then read any file the server can read (the
  config with every secret, SSH keys), ``/api/logs/delete/<path>`` deleted
  any file it can write (the database).
* ``clean_workdir()`` only refused a work folder that IS or CONTAINS a data
  folder. One INSIDE the media folder ("<media>/Series") was emptied —
  recordings and their .transcripts/ with it. And a work folder pointed at
  a folder full of someone's files (~/Documents) was emptied as well.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from whisper_tm.config import ConfigManager, config


@pytest.fixture()
def cfg(tmp_path: Path) -> ConfigManager:
    file = tmp_path / "cfg" / "config.yaml"
    file.parent.mkdir()
    file.write_text(
        yaml.safe_dump(
            {
                "directories": {"media": str(tmp_path / "media"), "work": str(tmp_path / "work")},
                "security": {"allowed_roots": [str(tmp_path / "media")]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "media" / "Series").mkdir(parents=True)
    c = ConfigManager()
    c.load(file)
    return c


# ── config.check_directory ────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["logs", "work", "clips", "transcripts", "uploads"])
@pytest.mark.parametrize("value", ["/", "/etc", "/usr/lib", "/home", "/proc/self"])
def test_system_folders_are_refused(cfg, name, value):
    assert cfg.check_directory(name, value)


def test_home_itself_and_folders_holding_the_library_are_refused(cfg, tmp_path):
    assert cfg.check_directory("logs", str(Path.home()))
    # contains the media folder
    assert cfg.check_directory("logs", str(tmp_path))
    assert cfg.check_directory("work", str(tmp_path / "media"))


def test_work_inside_the_library_is_refused(cfg, tmp_path):
    why = cfg.check_directory("work", str(tmp_path / "media" / "Series"))
    assert "inside" in why


def test_ordinary_folders_pass(cfg, tmp_path):
    assert cfg.check_directory("logs", str(tmp_path / "logs")) == ""
    assert cfg.check_directory("work", str(tmp_path / "scratch" / "work")) == ""
    # logs, clips and uploads may sit in the library tree (e.g. on a NAS)
    assert cfg.check_directory("logs", str(tmp_path / "media" / "logs")) == ""
    assert cfg.check_directory("uploads", str(tmp_path / "media")) == ""
    assert cfg.check_directory("work", "") == ""  # default


# ── clean_workdir ─────────────────────────────────────────────────────────


def test_clean_workdir_refuses_a_folder_inside_the_media(cfg, tmp_path):
    series = tmp_path / "media" / "Series"
    (series / "episode1.mp4").write_bytes(b"video")
    (series / ".transcripts").mkdir()
    (series / ".transcripts" / "episode1.mp4.json").write_text("{}")
    cfg._data["directories"]["work"] = str(series)
    cfg._resolve_directories()

    result = cfg.clean_workdir()
    assert result["refused"] is True
    assert (series / "episode1.mp4").exists()
    assert (series / ".transcripts" / "episode1.mp4.json").exists()


def test_clean_workdir_refuses_an_unmarked_folder_full_of_files(cfg, tmp_path):
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "taxes.pdf").write_bytes(b"pdf")
    cfg._data["directories"]["work"] = str(docs)
    cfg._resolve_directories()
    cfg._ensure_directories()  # must not mark a folder with content

    result = cfg.clean_workdir()
    assert result["refused"] is True
    assert (docs / "taxes.pdf").exists()
    assert not (docs / cfg.WORK_MARKER).exists()


def test_clean_workdir_empties_a_new_work_folder(cfg, tmp_path):
    wd = cfg.work_dir
    (wd / "job_1").mkdir()
    (wd / "job_1" / "asr_block_001.wav").write_bytes(b"x")
    result = cfg.clean_workdir()
    assert result["ok"] is True and result["removed"] == 1
    assert (wd / cfg.WORK_MARKER).exists()


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


def test_refusal_names_no_command_line(cfg, monkeypatch):
    """The reason goes out over HTTP; another process's command line must not."""
    import json
    import os
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "whisper_tm", "--secret-arg"]
    )
    try:
        _exec_done(child, "--secret-arg")
        (cfg.work_dir / cfg.USAGE_FILE).write_text(
            json.dumps({"pid": child.pid, "cmdline": "x"}), encoding="utf-8"
        )
        reason = cfg._workdir_taken_by_other()
        assert reason == f"PID {child.pid}"
        assert "--secret-arg" not in reason
    finally:
        child.kill()
        child.wait()
    assert os.getpid()


# ── HTTP ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    own = tmp_path / "config.yaml"
    own.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "media")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data["directories"]["logs"] = str(tmp_path / "logs")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "own.db")
    (tmp_path / "media").mkdir()
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


def test_put_refuses_logs_at_the_filesystem_root(client):
    c, tmp = client
    r = c.put("/api/settings", json={"directories": {"logs": "/"}})
    body = r.get_json()
    assert body["ok"] is False
    assert any("directories.logs" in e for e in body["errors"])
    assert config.dirs["logs"] == (tmp / "logs").resolve()


def test_put_refuses_work_inside_the_media(client):
    c, tmp = client
    (tmp / "media" / "Series").mkdir()
    r = c.put("/api/settings", json={"directories": {"work": str(tmp / "media" / "Series")}})
    assert r.get_json()["ok"] is False
    assert config.work_dir == (tmp / "work").resolve()


def test_put_accepts_an_ordinary_logs_folder(client):
    c, tmp = client
    r = c.put("/api/settings", json={"directories": {"logs": str(tmp / "other-logs")}})
    assert r.get_json()["ok"] is True
    assert config.dirs["logs"] == (tmp / "other-logs").resolve()


def test_log_viewer_serves_only_log_files(client):
    c, tmp = client
    logs = config.dirs["logs"]
    (logs / "20240301_job.log").write_text("line\n", encoding="utf-8")
    (logs / "secret.yaml").write_text("api_key: x\n", encoding="utf-8")
    assert c.get("/api/logs/view/20240301_job.log").status_code == 200
    assert c.get("/api/logs/view/secret.yaml").status_code in (403, 404)
    assert c.post("/api/logs/delete/secret.yaml").status_code in (403, 404)
    assert (logs / "secret.yaml").exists()


def test_log_viewer_does_not_follow_a_link_out(client):
    c, tmp = client
    logs = config.dirs["logs"]
    outside = tmp / "outside.log"
    outside.write_text("private\n", encoding="utf-8")
    (logs / "link.log").symlink_to(outside)
    assert c.get("/api/logs/view/link.log").status_code in (403, 404)
    assert c.post("/api/logs/delete/link.log").status_code in (403, 404)
    assert outside.exists()
    # …and the list doesn't offer what can't be opened
    names = [x["filename"] for x in c.get("/api/logs").get_json()["logs"]]
    assert "link.log" not in names

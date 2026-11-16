"""
The log explorer lists the newest job logs.

The job log folder can hold well over 100,000 files, e.g. on a NAS. ``GET /api/logs`` sorted
every entry and asked the file system for type, size and date of each one
(three round trips per file), only to keep 200 — the request did not answer
within 30 s. Job log names start with their timestamp
(``jobs/__init__.py::_persist_log``), so the names alone give the order,
and only the returned files are looked up.
"""

from __future__ import annotations

import os
import random
import time

import pytest
from flask import Flask

from whisper_tm.config import config


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """The settings routes on a log folder of the test's own."""
    from whisper_tm.api.settings import settings_bp

    folder = tmp_path / "logs"
    folder.mkdir()
    monkeypatch.setattr(config, "_resolved_logs", folder)
    app = Flask(__name__)
    app.register_blueprint(settings_bp)
    return app.test_client(), folder


def _fill(folder, count: int) -> list[str]:
    """Job logs named as jobs/__init__.py::_persist_log names them, one
    minute apart — written in shuffled order, so the directory order can't
    pass for the right one."""
    start = time.mktime((2026, 9, 1, 8, 0, 0, 0, 0, -1))
    names = [
        time.strftime("%Y%m%d_%H%M%S", time.localtime(start + 60 * i))
        + f"_videos_clip_{i:03d}.mp4.log"
        for i in range(count)
    ]
    for name in random.Random(7).sample(names, len(names)):
        (folder / name).write_text("line\n" * (len(name) % 5), encoding="utf-8")
    return names


def test_the_newest_come_first_and_only_200(logs):
    client, folder = logs
    names = _fill(folder, 450)
    (folder / "notes.txt").write_text("not a log", encoding="utf-8")

    r = client.get("/api/logs")
    assert r.status_code == 200
    got = r.get_json()["logs"]
    assert [x["filename"] for x in got] == sorted(names, reverse=True)[:200]
    # Still what the page shows per line
    newest = folder / got[0]["filename"]
    assert got[0]["size"] == newest.stat().st_size
    assert got[0]["modified"] == pytest.approx(newest.stat().st_mtime)


def test_only_the_returned_files_are_looked_up(logs, monkeypatch):
    """One stat per returned file — not three per file in the folder."""
    client, folder = logs
    names = _fill(folder, 450)
    # A folder whose name ends in .log sorts first — it is not a log
    (folder / "99991231_235959_not_a_file.log").mkdir()
    looked_up = []
    real = os.stat

    def counting(path, *args, **kwargs):
        if str(path).startswith(str(folder)):
            looked_up.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting)
    r = client.get("/api/logs")
    assert [x["filename"] for x in r.get_json()["logs"]] == sorted(names, reverse=True)[:200]
    assert len(looked_up) <= 201, f"{len(looked_up)} lookups for 200 entries"


def test_the_name_filter_keeps_the_order(logs):
    client, folder = logs
    names = _fill(folder, 450)
    r = client.get("/api/logs?q=CLIP_01")
    wanted = sorted((n for n in names if "clip_01" in n), reverse=True)
    assert len(wanted) == 10
    assert [x["filename"] for x in r.get_json()["logs"]] == wanted


def test_without_a_log_folder_the_list_is_empty(logs, monkeypatch):
    client, folder = logs
    monkeypatch.setattr(config, "_resolved_logs", folder / "not-there")
    r = client.get("/api/logs")
    assert r.status_code == 200
    assert r.get_json() == {"logs": []}


def test_the_download_takes_only_job_logs(tmp_path, monkeypatch):
    """GET /api/logs/download had weaker checks than the viewer: any file in
    the logs folder, links too."""
    from whisper_tm.api.transcription import transcription_bp

    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "job.log").write_text("ok\n", encoding="utf-8")
    (folder / "secrets.yaml").write_text("key: x\n", encoding="utf-8")
    outside = tmp_path / "outside.log"
    outside.write_text("not mine\n", encoding="utf-8")
    (folder / "link.log").symlink_to(outside)
    monkeypatch.setattr(config, "_resolved_logs", folder)
    app = Flask(__name__)
    app.register_blueprint(transcription_bp)
    c = app.test_client()
    assert c.get("/api/logs/download/job.log").status_code == 200
    assert c.get("/api/logs/download/secrets.yaml").status_code == 404
    assert c.get("/api/logs/download/link.log").status_code == 404

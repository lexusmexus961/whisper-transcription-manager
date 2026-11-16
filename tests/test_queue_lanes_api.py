"""Queue lanes in the API layer: several queue entries run at once.

core/queue.snapshot() reports them as ``running_items`` (oldest first) and
keeps current_* pointing at the oldest one. The routes add each entry's OWN
job progress — with only the oldest job's numbers, the browser would show
them over whichever running file it has open. The snapshot and the job
manager are stand-ins here: the lanes themselves are core/queue.py's tests.
"""

from __future__ import annotations

import json
import time

import pytest
from flask import Flask

import whisper_tm.extensions as ext
from whisper_tm.api import queue as queue_api
from whisper_tm.core import queue as q
from whisper_tm.jobs import JobManager

PHASE = {"index": 5, "total": 8, "title": "Diarization", "start": 1000.0}


def _snapshot(**over) -> dict:
    """Three running entries — the third one's lane has not made its job yet."""
    snap = {
        "running": True,
        "paused": False,
        "abort": False,
        "current_file": "a/one.mkv",
        "current_abs": "/m/a/one.mkv",
        "current_dir": "/m",
        "current_job_id": "j1",
        "current_item_id": 1,
        "running_items": [
            {
                "item_id": 1,
                "job_id": "j1",
                "filename": "a/one.mkv",
                "abs": "/m/a/one.mkv",
                "dir": "/m",
                "started_at": 100.0,
            },
            {
                "item_id": 2,
                "job_id": "j2",
                "filename": "two.mp3",
                "abs": "/n/two.mp3",
                "dir": "/n",
                "started_at": 101.0,
            },
            {
                "item_id": 3,
                "job_id": "",
                "filename": "three.wav",
                "abs": "/m/three.wav",
                "dir": "/m",
                "started_at": 102.0,
            },
        ],
        "running_count": 3,
        "lanes": 3,
        "total": 10,
        "processed": 4,
        "pending": 3,
        "done_count": 4,
        "fail_count": 0,
        "skip_count": 0,
        "cancel_count": 0,
        "retryable_count": 0,
        "started_at": 90.0,
        "finished_at": None,
        "log": [],
        "generation": 7,
    }
    snap.update(over)
    return snap


@pytest.fixture
def jm():
    m = JobManager()
    for jid, name, progress, message in (
        ("j1", "a/one.mkv", 40, "Transcribing…"),
        ("j2", "two.mp3", 75, "Finding speakers…"),
    ):
        job = m.create(jid, name)
        job.update(status="processing", progress=progress, message=message)
        job.start_time = time.time() - 12
    m.get("j2").phase = dict(PHASE)
    return m


@pytest.fixture
def client(monkeypatch, jm):
    monkeypatch.setattr(ext, "job_manager", jm)
    monkeypatch.setattr(q, "snapshot", lambda: _snapshot())
    monkeypatch.setattr(q, "list_items", lambda status=None, limit=None: [])
    app = Flask(__name__)
    app.register_blueprint(queue_api.queue_bp)
    return app.test_client()


def _ticks(client, n: int) -> list[dict]:
    """The first ``n`` events of the endless status stream."""
    resp = client.get("/api/queue/stream")
    out = []
    try:
        for chunk in resp.response:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            if text.startswith("data: "):
                out.append(json.loads(text[len("data: ") :]))
            if len(out) == n:
                break
    finally:
        resp.close()
    return out


class TestTheListRoute:
    def test_every_running_entry_comes_with_its_own_jobs_progress(self, client):
        d = client.get("/api/queue").get_json()
        one, two, three = d["running_items"]
        assert [r["item_id"] for r in (one, two, three)] == [1, 2, 3]
        assert (one["progress"], one["message"]) == (40, "Transcribing…")
        assert (two["progress"], two["message"]) == (75, "Finding speakers…")
        assert two["phase"] == PHASE and one["phase"] is None
        assert one["job_elapsed_sec"] == pytest.approx(12, abs=2)
        # No job yet: nothing to report, and nothing made up either
        assert "progress" not in three and three["job_id"] == ""
        assert (d["running_count"], d["lanes"]) == (3, 3)

    def test_current_stays_the_oldest(self, client):
        d = client.get("/api/queue").get_json()
        assert d["current_file"] == "a/one.mkv" and d["current_job_id"] == "j1"
        assert d["current_abs"] == "/m/a/one.mkv"

    def test_a_job_the_manager_no_longer_knows_adds_nothing(self, client, jm):
        jm.remove("j2")
        two = client.get("/api/queue").get_json()["running_items"][1]
        assert two["job_id"] == "j2" and "progress" not in two

    def test_a_phase_that_is_not_plain_data_is_dropped(self, client, jm):
        jm.get("j2").phase = {"title": "Diarization", "since": object()}
        r = client.get("/api/queue")
        assert r.status_code == 200
        two = r.get_json()["running_items"][1]
        assert two["phase"] is None and two["progress"] == 75

    def test_a_core_without_lanes_still_answers(self, client, monkeypatch):
        old = _snapshot()
        for key in ("running_items", "running_count", "lanes"):
            old.pop(key)
        monkeypatch.setattr(q, "snapshot", lambda: dict(old))
        d = client.get("/api/queue").get_json()
        # The browser then builds the list from current_* (batch.js)
        assert d["running_items"] == [] and d["running_count"] == 0 and d["lanes"] == 1
        assert d["current_file"] == "a/one.mkv"


class TestTheStream:
    def test_each_tick_carries_every_running_entry(self, client):
        (d,) = _ticks(client, 1)
        assert [r["filename"] for r in d["running_items"]] == ["a/one.mkv", "two.mp3", "three.wav"]
        assert d["running_items"][1]["progress"] == 75
        assert d["running_items"][1]["phase"] == PHASE
        assert (d["running_count"], d["lanes"]) == (3, 3)

    def test_the_old_fields_stay_the_oldest_jobs(self, client):
        (d,) = _ticks(client, 1)
        assert d["current_file"] == "a/one.mkv" and d["current_job_id"] == "j1"
        assert (d["job_progress"], d["job_message"]) == (40, "Transcribing…")

    def test_progress_goes_out_every_tick_the_lists_only_on_change(self, client, jm, monkeypatch):
        monkeypatch.setattr(queue_api.time, "sleep", lambda s: jm.get("j2").update(progress=76))
        first, second = _ticks(client, 2)
        assert first["running_items"][1]["progress"] == 75
        assert second["running_items"][1]["progress"] == 76
        # The waiting list didn't change: as before, only the first tick has it
        assert "items" in first and "items" not in second


class TestTheHealthCard:
    """/api/health/detail names the oldest running file and how many more."""

    def test_it_counts_every_running_entry(self, monkeypatch):
        from whisper_tm.api import health

        monkeypatch.setattr(health, "db", None)
        monkeypatch.setattr(q, "snapshot", lambda: _snapshot())
        out = health._queue_block()
        assert out["current_file"] == "a/one.mkv" and out["running_count"] == 3

    def test_a_core_without_lanes_counts_its_one_file(self, monkeypatch):
        from whisper_tm.api import health

        old = _snapshot()
        old.pop("running_count")
        monkeypatch.setattr(health, "db", None)
        monkeypatch.setattr(q, "snapshot", lambda: old)
        assert health._queue_block()["running_count"] == 1

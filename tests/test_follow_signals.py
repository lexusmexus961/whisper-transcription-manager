"""What the page following a queued file learns beyond "done" and "error".

Case: an upload went into the running queue with "summarize". Its first
attempt ran out of GPU memory — the page showed "❌ CUDA failed with error
out of memory" and the file's incidents an "error", although the queue
takes every OOM up again (_oom_retry) and the second attempt went through.
Then the transcript showed with "0 summaries": the summary followed a
minute later, and the page never learned of it.

* ``will_retry`` in the job's stream: the error is not the end of the file;
* no "error" incident for it — the queue books its own "oom" one;
* ``pending`` from /api/summaries while the queue still writes one.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import incidents
from whisper_tm.core import queue as q
from whisper_tm.jobs import JobManager, job_retries

OOM = "CUDA failed with error out of memory"


@pytest.fixture
def jm(tmp_path):
    return JobManager(logs_dir=tmp_path / "logs")


def _failed(jm, error, retry_if=q._is_oom_error):
    job = jm.create("j1", filename="talk.m4a")
    job.retry_if = retry_if
    jm.fail(job, error)
    return job


class TestTheStream:
    def test_an_oom_the_queue_takes_up_again(self, jm):
        d = jm.get_stream_data(_failed(jm, OOM))
        assert d["status"] == "error" and d["error"] == OOM and d["will_retry"] is True

    def test_any_other_error_is_the_end(self, jm):
        assert "will_retry" not in jm.get_stream_data(_failed(jm, "No audio track"))

    def test_outside_the_queue_an_oom_is_an_error(self, jm):
        assert "will_retry" not in jm.get_stream_data(_failed(jm, OOM, retry_if=None))

    def test_a_broken_check_counts_as_no(self, jm):
        def broken(error):
            raise ValueError("boom")

        assert "will_retry" not in jm.get_stream_data(_failed(jm, OOM, retry_if=broken))

    def test_only_an_error_retries(self, jm):
        job = jm.create("j2", filename="talk.m4a")
        job.retry_if = q._is_oom_error
        job.error = OOM  # a message, while it still runs
        assert job_retries(job) is False


class TestTheIncidents:
    def test_no_error_incident_for_an_error_taken_up_again(self, jm, monkeypatch):
        booked = []
        monkeypatch.setattr(incidents, "report_it", lambda kind, text, **k: booked.append(kind))
        incidents.job_event("error", _failed(jm, OOM))
        assert booked == []
        incidents.job_event("error", _failed(jm, "No audio track"))
        assert booked == ["error"]


class TestTheSummaries:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from whisper_tm.config import config

        config.load(None)
        config._data.setdefault("directories", {})["media"] = str(tmp_path)
        config._resolve_directories()
        config._ensure_directories()
        from whisper_tm.app import create_app

        return create_app().test_client(), tmp_path

    def test_pending_while_the_queue_writes_one(self, client, monkeypatch):
        c, root = client
        asked = []
        monkeypatch.setattr(q, "summary_pending", lambda abs_path: asked.append(abs_path) or True)
        d = c.get("/api/summaries/uploads/talk.m4a").get_json()
        assert d == {"summaries": [], "pending": True}
        assert asked == [str(root / "uploads" / "talk.m4a")]

    def test_not_pending_otherwise(self, client):
        c, _root = client
        assert c.get("/api/summaries/talk.m4a").get_json()["pending"] is False

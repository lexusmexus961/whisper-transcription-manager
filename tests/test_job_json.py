"""A job as JSON — without persistent stream.

Previously, there was no way to query the state of ONE job without keeping
a `text/event-stream` open:

* ``/api/status/<job_id>`` is SSE and runs for up to four hours,
* ``/api/active_jobs`` filters out completed jobs (``list_active`` only lets
  ``queued`` and ``processing`` through).

So the **result** of a job was unreachable for every non-streaming caller.
Specifically it hurts with YouTube downloads: which file was created is stored
only in ``job.result["files"]``. Names are built from
``secure_filename(title)[:80]`` with dedup counter and are not predictable —
without this information, only searching the upload directory by timestamp
and guessing remains.

Built for the MCP server, but not limited to it: every client that polls
instead of streaming needs exactly this.

**Two sources.** JobManager throws out jobs after ``max_age_secs`` (one hour)
from memory. Whoever asks later gets the row from ``job_history`` — fewer
fields, but it answers "did it run and how long did it take?" A 404 would be
a lie there: the job did exist.
"""

from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Mini app with real JobManager and real, empty database."""
    from whisper_tm.database import Database
    from whisper_tm.jobs import JobManager

    d = Database(tmp_path / "library.db")
    d.initialize()
    jm = JobManager()

    import whisper_tm.api.transcription as tr

    monkeypatch.setattr(tr, "db", d)
    monkeypatch.setattr(tr, "job_manager", jm)

    a = Flask(__name__)
    a.register_blueprint(tr.transcription_bp)
    # Do NOT propagate errors, but answer properly — otherwise the test
    # checks the debugger instead of the endpoint.
    a.config["TESTING"] = False
    a.jm = jm
    a.db = d
    return a


def _fetch(app, job_id):
    r = app.test_client().get(f"/api/jobs/{job_id}")
    return r.status_code, r.get_json()


class TestRunningJob:
    def test_a_running_job_is_reported(self, app):
        job = app.jm.create("j1", filename="a.mkv", job_type="transcription")
        job.update(status="processing", progress=42, message="running")
        code, d = _fetch(app, "j1")
        assert code == 200
        assert d["jobId"] == "j1" and d["status"] == "processing"
        assert d["progress"] == 42
        assert d["source"] == "running"

    def test_the_RESULT_comes_along(self, app):
        """The actual reason for this endpoint: `result` used to be visible
        only in the SSE stream."""
        job = app.jm.create("yt1", filename="https://…", job_type="youtube")
        app.jm.complete(
            job,
            {"total": 3, "ok": 3, "files": ["uploads/a.mp4", "uploads/b.mp4"], "in_library": True},
        )
        code, d = _fetch(app, "yt1")
        assert code == 200
        assert d["result"]["files"] == ["uploads/a.mp4", "uploads/b.mp4"]
        assert d["status"] == "done"

    def test_an_error_is_reported_as_an_error(self, app):
        job = app.jm.create("j2", filename="b.mkv")
        app.jm.fail(job, "CUDA failed with error out of memory")
        code, d = _fetch(app, "j2")
        assert d["status"] == "error"
        assert "out of memory" in d["error"]


class TestFallbackToTheHistory:
    """After an hour, the JobManager no longer knows the job."""

    def test_a_forgotten_job_comes_from_the_history(self, app):
        app.db.log_job("old1", "c.mkv", job_type="transcription", model="large-v3", language="de")
        app.db.update_job("old1", status="done", processing_sec=123.4, duration_sec=3600.0)
        code, d = _fetch(app, "old1")
        assert code == 200, "a forgotten job is reported as 404"
        assert d["source"] == "history"
        assert d["status"] == "done"
        assert d["filename"] == "c.mkv"
        assert d["elapsed_sec"] == 123.4
        assert d["model"] == "large-v3"

    def test_the_running_job_takes_precedence(self, app):
        """Both sources can know the same job — the fresher one wins."""
        app.db.log_job("j3", "d.mkv")
        app.db.update_job("j3", status="done")
        job = app.jm.create("j3", filename="d.mkv")
        job.update(status="processing", progress=7)
        code, d = _fetch(app, "j3")
        assert d["source"] == "running" and d["status"] == "processing"

    def test_a_failed_job_reports_no_progress(self, app):
        """Claiming 100% would simply be wrong for 'error'."""
        app.db.log_job("old2", "e.mkv")
        app.db.update_job("old2", status="error", error_message="broken")
        _, d = _fetch(app, "old2")
        assert d["progress"] == 0
        assert d["error"] == "broken"

    def test_a_truly_unknown_job_is_404(self, app):
        code, d = _fetch(app, "does_not_exist")
        assert code == 404
        assert "not found" in d["error"]


class TestContractWithTheStream:
    """The JSON path must carry the same fields as `to_dict()` — otherwise
    a client would have to know two forms."""

    def test_the_same_fields_as_to_dict(self, app):
        from whisper_tm.jobs import Job

        app.jm.create("j4", filename="f.mkv")
        _, d = _fetch(app, "j4")
        expected = set(Job("j4", "f.mkv").to_dict())
        assert expected <= set(d), f"missing: {expected - set(d)}"

    def test_it_really_is_not_a_stream(self):
        """A client that polls must not accidentally get stuck on a
        permanent stream."""
        from pathlib import Path

        q = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "api" / "transcription.py"
        ).read_text(encoding="utf-8")
        i = q.index("def job_status_json(")
        body = q[i : q.index("@transcription_bp.route", i)]
        assert "Response(" not in body and "yield" not in body
        assert "jsonify(" in body

"""The idle unloader must not kill a running job.

**Failure mode:** a very long recording whose diarization takes longer
than the 10-minute limit. In the log:

    Diarization worker terminated (PID 12345, freeing models)
    VRAM freed (limit: 10 min, hardware.model_idle_min)
    ❌ Error: … Diarization worker terminated unexpectedly.

Two recent changes came together:

1. `release_models` now takes the diarization worker with it —
   correct, because its VRAM counts the same. Only now it hits a
   PROCESS that is currently working.
2. The watchdog assumes a running job touches 'one of the models every
   few minutes'. With very long files it gets stuck for minutes in ONE
   step, without touching a model again.

Fixed on both levels: The worker reports usage while waiting,
and the watchdog never unloads anything while a job is running.
"""

from __future__ import annotations


class TestGuardAgainstUnloading:
    def test_nothing_is_unloaded_while_a_job_runs(self, monkeypatch):
        from whisper_tm.core import model_idle as ml

        class JM:
            @staticmethod
            def list_active():
                return [{"job_id": "x", "status": "processing"}]

        monkeypatch.setattr("whisper_tm.extensions.job_manager", JM)
        assert ml._job_running() is True

    def test_without_a_job_unloading_is_allowed(self, monkeypatch):
        from whisper_tm.core import model_idle as ml

        class JM:
            @staticmethod
            def list_active():
                return []

        monkeypatch.setattr("whisper_tm.extensions.job_manager", JM)
        assert ml._job_running() is False

    def test_when_in_doubt_nothing_is_unloaded(self, monkeypatch):
        """A model held too long costs VRAM, an unloaded one too early
        costs an entire job."""
        from whisper_tm.core import model_idle as ml

        class JM:
            @staticmethod
            def list_active():
                raise RuntimeError("job manager broken")

        monkeypatch.setattr("whisper_tm.extensions.job_manager", JM)
        assert ml._job_running() is True

    def test_the_watchdog_really_asks(self):
        """The query must be in the watchdog loop, not just beside it."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "model_idle.py"
        ).read_text(encoding="utf-8")
        run = source[source.index("def _watchdog_run") :]
        assert "_job_running()" in run, (
            "the watchdog still unloads without asking about running jobs"
        )
        assert run.index("_job_running()") < run.index("unload_all()"), (
            "the query is AFTER unloading — then it's useless"
        )


class TestManualUnload:
    """The "Unload VRAM" button in the top bar (POST /api/vram/unload)."""

    @staticmethod
    def _jm(monkeypatch, active):
        class JM:
            @staticmethod
            def list_active():
                return active

        monkeypatch.setattr("whisper_tm.extensions.job_manager", JM)

    @staticmethod
    def _spy(monkeypatch, loaded=True):
        from whisper_tm.core import model_idle as ml

        calls = []
        monkeypatch.setattr(ml, "unload_all", lambda: calls.append(1))
        monkeypatch.setattr(ml, "_something_loaded", lambda: loaded)
        monkeypatch.setattr(ml, "_stop_media_service", lambda: False)
        return calls

    def test_refused_while_a_job_runs(self, monkeypatch):
        from whisper_tm.core import model_idle as ml

        self._jm(monkeypatch, [{"job_id": "x", "status": "processing"}])
        calls = self._spy(monkeypatch)
        assert ml.unload_now() == {"ok": False, "reason": "job_running"}
        assert calls == []

    def test_unloads_when_idle(self, monkeypatch):
        from whisper_tm.core import model_idle as ml

        self._jm(monkeypatch, [])
        calls = self._spy(monkeypatch)
        assert ml.unload_now() == {"ok": True, "unloaded": True}
        assert calls == [1]

    def test_nothing_loaded_is_not_an_error(self, monkeypatch):
        from whisper_tm.core import model_idle as ml

        self._jm(monkeypatch, [])
        calls = self._spy(monkeypatch, loaded=False)
        assert ml.unload_now() == {"ok": True, "unloaded": False}
        assert calls == []

    def test_the_media_service_is_stopped_too(self, monkeypatch):
        """Its released model still leaves the process holding VRAM."""
        from whisper_tm.core import media_server
        from whisper_tm.core import model_idle as ml
        from whisper_tm.media import indexer

        real = ml._stop_media_service
        self._jm(monkeypatch, [])
        calls = self._spy(monkeypatch, loaded=False)
        monkeypatch.setattr(ml, "_stop_media_service", real)
        stops = []
        monkeypatch.setattr(
            media_server.SERVER, "stop", lambda reason="", force=False: stops.append(reason) or True
        )
        monkeypatch.setattr(indexer.INDEXER, "running", lambda: False)
        assert ml.unload_now() == {"ok": True, "unloaded": True}
        assert stops == ["unload on request"] and calls == []
        monkeypatch.setattr(indexer.INDEXER, "running", lambda: True)  # an index run keeps it
        stops.clear()
        assert ml.unload_now() == {"ok": True, "unloaded": False} and stops == []

    def test_endpoint_answers_409_while_a_job_runs(self, monkeypatch):
        from flask import Flask

        from whisper_tm.api.health import health_bp

        self._jm(monkeypatch, [{"job_id": "x", "status": "processing"}])
        self._spy(monkeypatch)
        app = Flask(__name__)
        app.register_blueprint(health_bp)
        r = app.test_client().post("/api/vram/unload")
        assert r.status_code == 409
        assert r.get_json()["ok"] is False

    def test_endpoint_unloads(self, monkeypatch):
        from flask import Flask

        from whisper_tm.api.health import health_bp

        self._jm(monkeypatch, [])
        calls = self._spy(monkeypatch)
        app = Flask(__name__)
        app.register_blueprint(health_bp)
        r = app.test_client().post("/api/vram/unload")
        assert r.status_code == 200
        d = r.get_json()
        assert d["ok"] is True and d["unloaded"] is True
        assert calls == [1]


class TestWorkerReportsUsage:
    def test_the_waiting_loop_reports_usage(self):
        """Otherwise the watchdog thinks the worker is idle while it's calculating."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "diar_process.py"
        ).read_text(encoding="utf-8")
        wait = source[source.index("def _wait") :]
        wait = wait[: wait.index("def diarize_it")]
        assert "report_usage" in wait, "the waiting worker reports no model usage"

    def test_reporting_usage_lifts_the_timestamp(self):
        import time

        from whisper_tm.core import model_idle as ml

        ml._last_use = time.time() - 3600
        old_value = ml._last_use
        ml.report_usage()
        assert ml._last_use > old_value

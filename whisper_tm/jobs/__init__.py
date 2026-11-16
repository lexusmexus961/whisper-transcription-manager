"""
Job manager: state, locking, logging, persistence, cleanup.

Manages all running and completed jobs with:
- Thread-safe state management
- Automatic cleanup of old jobs (TTL-based)
- Log persistence to disk
- WebSocket notifications
- Job recovery after restart
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ceilings for the two SSE delivery buffers. Both are emptied exclusively
# by one reading client (get_stream_data). Since the queue continues
# after the refactor also WITHOUT an open WebUI, its growth must be limited —
# otherwise a server watched by nobody for days eats up memory.
MAX_NEW_LOG = 1000
MAX_NEW_SEGMENTS = 2000


# ── Job state ─────────────────────────────────────────────────────────────


def job_retries(job) -> bool:
    """Whether the job's error is not the end — its queue entry is taken up
    again (Job.retry_if). A broken check counts as no."""
    check = getattr(job, "retry_if", None)
    if not check or getattr(job, "status", "") != "error":
        return False
    try:
        return bool(check(job.error or ""))
    except Exception:
        return False


#: What a job log's name doesn't take from the file's name: blanks and
#: separators, and what Windows refuses in a file name — a link's job is
#: named after its URL (":" even writes into a hidden stream there)
_LOG_NAME_RE = re.compile(r'[ /\\<>:"|?*\x00-\x1f]')


def log_name_part(filename: str) -> str:
    """The file's name as it goes into its job log's name."""
    return _LOG_NAME_RE.sub("_", filename)


@dataclass
class Job:
    """A single processing job."""

    job_id: str
    filename: str = ""
    job_type: str = "transcription"  # transcription, separation, batch
    status: str = "queued"  # queued, processing, done, error
    progress: int = 0  # 0-100
    message: str = "Queued…"
    result: dict | None = None
    error: str | None = None
    cancel: bool = False

    # Segments (live streaming)
    new_segments: list = field(default_factory=list)

    # Logging
    log: list[str] = field(default_factory=list)
    new_log: list[str] = field(default_factory=list)

    # Warnings
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)

    # Timing
    step_timings: list[dict] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0
    # Last worker heartbeat (progress/log). Stale cleanup follows
    # this — NOT start_time: a multi-hour transcription has not "hung"
    # just because it takes a long time.
    last_activity: float = 0

    # Persistence
    log_file: str = ""
    # File identity (docs/PATHS.md) — for the incidents table that
    # should survive reruns; `filename` is only the relative path.
    abs_path: str = ""

    # Set by the queue for its runs — what the page following the job needs
    # beyond the status. `retry_if(error)` is true for an error that does
    # not end the entry: the queue takes it up again (an OOM, _oom_retry).
    retry_if: Any = None

    def update(self, **kwargs) -> None:
        """Set multiple fields at once."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_activity = time.time()

    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time else 0

    def to_dict(self) -> dict:
        """Job as dict (for API responses)."""
        return {
            "jobId": self.job_id,
            "filename": self.filename,
            "type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "elapsed_sec": round(self.elapsed(), 1),
            "warning_count": self.warning_count,
            "log_file": self.log_file,
        }


# ── Job Manager ───────────────────────────────────────────────────────────


class JobManager:
    """Thread-safe job management with lifecycle handling.

    Features:
    - Automatic cleanup of old jobs (default: 1h after completion)
    - Log persistence to disk
    - Event callbacks for WebSocket integration
    - Maximum number of concurrent jobs can be limited
    """

    def __init__(
        self, logs_dir: Path | None = None, max_age_secs: int = 3600, cleanup_interval: int = 300
    ):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._logs_dir = logs_dir
        self._max_age = max_age_secs
        self._callbacks: list[Callable] = []
        self._cleanup_interval = cleanup_interval
        self._system_warnings: list[dict] = []
        self._cleanup_timer: threading.Timer | None = None
        self._cleanup_stopped = False
        # Give jobs that got stuck (worker died without complete()/fail())
        # a generous timeout — a multiple of the normal TTL.
        self._stale_max_age = max_age_secs * 6

    def start_cleanup_timer(self) -> None:
        """Start periodic cleanup."""
        with self._lock:
            self._cleanup_stopped = False
        self._run_cleanup()

    def _run_cleanup(self) -> None:
        """Remove old jobs."""
        self.cleanup()
        # Don't schedule a new timer after stop_cleanup_timer(), otherwise
        # cleanup keeps running forever despite "stop" (reschedule race).
        with self._lock:
            if self._cleanup_stopped:
                return
            self._cleanup_timer = threading.Timer(self._cleanup_interval, self._run_cleanup)
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

    def stop_cleanup_timer(self) -> None:
        with self._lock:
            self._cleanup_stopped = True
            timer = self._cleanup_timer
        if timer:
            timer.cancel()

    # ── Job CRUD ──────────────────────────────────────────────────────

    def create(self, job_id: str, filename: str = "", job_type: str = "transcription") -> Job:
        """Create a new job."""
        job = Job(
            job_id=job_id,
            filename=filename,
            job_type=job_type,
            start_time=time.time(),
            last_activity=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
        self._notify("created", job)
        return job

    def get(self, job_id: str) -> Job | None:
        """Fetch a job (None if not found)."""
        with self._lock:
            return self._jobs.get(job_id)

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def remove(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def list_active(self) -> list[dict]:
        """All active jobs as a list of dicts."""
        with self._lock:
            return [
                job.to_dict()
                for job in self._jobs.values()
                if job.status in ("queued", "processing")
            ]

    def log(self, job: Job, msg: str) -> None:
        """Append a timestamped log line to a job."""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._lock:
            job.last_activity = time.time()  # heartbeat for cleanup
            job.log.append(line)
            job.new_log.append(line)
            # new_log is a delivery buffer for the SSE stream — it is only
            # ever emptied when a client reads it. Without an open WebUI it
            # would otherwise grow without bound (the queue keeps running
            # regardless). Ring buffer: discard the oldest lines on overflow.
            if len(job.new_log) > MAX_NEW_LOG:
                del job.new_log[:-MAX_NEW_LOG]
            stragglers = job.status in ("done", "error") and bool(job.log_file)
        self._notify("log", job, line)
        # A line written AFTER completion: `complete()`/`fail()` have already
        # written the file, and without this catch-up it would be lost.
        #
        # Example: a "✅ Done!" line written one line after `complete()` would
        # be missing from every job log, which would then end in the middle of
        # post-processing — and nobody misses a "Done!" that never existed.
        #
        # This is the safeguard: whoever writes a line too late no longer
        # loses it silently.
        if stragglers:
            self._persist_log(job)

    def warn(self, job: Job, msg: str) -> None:
        """Append a warning to a job."""
        self.log(job, f"⚠ {msg}")
        with self._lock:
            job.warning_count += 1
            job.warnings.append(msg)
        # Dedicated event: the incidents table (core/incidents.py) listens
        # for it. Parsing a "log" event by its ⚠ prefix would be fragile.
        self._notify("warn", job, msg)

    # ── System-wide warnings (unmissable in the GUI) ──────────────
    # For states that do NOT concern just one job — e.g. when the
    # separation model falls back to the fallback. These warnings aren't
    # tied to any file, but to the server, and the WebUI shows them as a
    # red banner until they're acknowledged.
    def system_warning(self, msg: str, key: str | None = None) -> None:
        """Add a banner warning.

        ``key`` marks a warning that describes a STATE rather than an
        event: a newer one with the same key replaces the older. The memory
        warning fires after every job while the footprint stays high — without
        a key that stacks near-identical lines over the whole header.
        """
        with self._lock:
            if key is not None:
                self._system_warnings = [w for w in self._system_warnings if w.get("key") != key]
            self._system_warnings.append({"timestamp": time.time(), "text": msg, "key": key})
            del self._system_warnings[:-20]  # ring buffer
        # No job — the callback gets None and logs "system".
        self._notify("system", None, msg)

    def system_warnings(self) -> list[dict]:
        with self._lock:
            return list(self._system_warnings)

    def acknowledge_system_warnings(self) -> None:
        with self._lock:
            self._system_warnings.clear()

    def record_step(
        self, job: Job, label: str, duration_secs: float, group: str | None = None
    ) -> None:
        """Record step timing.

        The pipeline measures its steps anyway — the same values also flow
        from here into the central evaluation (debug.timings /
        /api/debug/timings), instead of instrumenting them a second time.
        """
        entry = {"label": label, "secs": round(duration_secs, 2)}
        if group:
            entry["group"] = group
        with self._lock:
            job.step_timings.append(entry)
        self.log(job, f"⏱ {label}: {self._fmt_duration(duration_secs)}")

        from whisper_tm.core import debug

        debug.record(f"pipeline: {label}", duration_secs)

    # ── Job Completion ────────────────────────────────────────────────

    def complete(self, job: Job, result: dict | None = None) -> None:
        """Mark a job as successfully completed.

        result is set BEFORE status, and both under lock — otherwise an SSE
        reader could see status="done" with result=None and the client
        would never get the result.
        """
        with self._lock:
            wc = job.warning_count
            msg = f"Done!{f' ({wc} warnings)' if wc > 0 else ''}"
            job.result = result
            job.end_time = time.time()
            job.progress = 100
            job.message = msg
            job.status = "done"
        self._persist_log(job)
        self._notify("done", job)

    def fail(self, job: Job, error: str) -> None:
        """Mark a job as failed."""
        with self._lock:
            job.error = error
            job.end_time = time.time()
            job.progress = 0
            job.message = error
            job.status = "error"
        self._persist_log(job)
        self._notify("error", job)

    # ── Log Persistence ───────────────────────────────────────────────

    def _persist_log(self, job: Job) -> str | None:
        """Save the job log as a text file."""
        if not self._logs_dir or not job.log:
            return None
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = log_name_part(job.filename or job.job_id)
        ts = time.strftime("%Y%m%d_%H%M%S")
        # The SAME file the second time around — the name carries the
        # timestamp, and a catch-up write must not create a second file for
        # the same job (with a six-figure number of logs in the directory,
        # that kind of thing would never be noticed).
        text = "\n".join(job.log) + "\n"
        if job.log_file:
            fp = self._logs_dir / job.log_file
            try:
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(text)
                return job.log_file
            except Exception as e:
                logger.warning("Could not save job log (%s): %s", fp, e)
                return None
        # A new name never takes another job's log: with queue lanes two
        # files whose names map to one safe name ('a/b.mp4', 'a_b.mp4') can
        # finish in the same second.
        for n in range(1, 100):
            fname = f"{ts}_{safe_name}.log" if n == 1 else f"{ts}_{safe_name}-{n}.log"
            fp = self._logs_dir / fname
            try:
                with open(fp, "x", encoding="utf-8") as f:
                    f.write(text)
                job.log_file = fname
                return fname
            except FileExistsError:
                continue
            except Exception as e:
                logger.warning("Could not save job log (%s): %s", fp, e)
                return None
        return None

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """Remove old completed jobs. Returns the number of jobs removed."""
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = []
            for jid, job in self._jobs.items():
                if job.status in ("done", "error"):
                    if job.end_time and (now - job.end_time) > self._max_age:
                        to_remove.append(jid)
                elif job.status in ("queued", "processing"):
                    # Worker died without complete()/fail() → the job would
                    # otherwise stay 'processing'/'queued' forever. Removed
                    # after a generous timeout.
                    #
                    # The yardstick is the LAST HEARTBEAT, not the start
                    # time: a long-running job (large-v3 + vocal isolation
                    # on a multi-hour recording) reports progress regularly
                    # and is very much alive. Going by start_time, it would
                    # have been cleaned up after 6h — mid-run — and the
                    # queue entry would afterward have wrongly ended up as
                    # "aborted".
                    alive_at = job.last_activity or job.start_time
                    if alive_at and (now - alive_at) > self._stale_max_age:
                        to_remove.append(jid)
            for jid in to_remove:
                del self._jobs[jid]
                removed += 1
        return removed

    # ── Event Callbacks ───────────────────────────────────────────────

    def on_event(self, callback: Callable) -> None:
        """Register a callback: fn(event_type, job, *args)"""
        with self._lock:
            self._callbacks.append(callback)

    def _notify(self, event: str, job: Job, *args) -> None:
        # Copy the list under lock, then iterate lock-free — otherwise a
        # concurrent on_event() could trigger "list changed during iteration".
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event, job, *args)
            except Exception as e:
                logger.debug("Unhandled exception: %s", e)

    # ── Live Segments ─────────────────────────────────────────────────

    def add_live_segments(self, job: Job, segments: list) -> None:
        """Thread-safely append live segments to a job.

        Public replacement for direct _lock access from external code:
        transcription workers stream new segments in here, the SSE stream
        (get_stream_data) consumes them and clears new_segments.

        Ring buffer: this list is a pure live preview for a watching
        browser. If nobody is watching, nobody clears it — for a 3-hour
        recording, tens of thousands of segment dicts would otherwise pile
        up here, file after file, while the queue keeps running in the
        background. The complete transcript is built independently of this,
        from all_segments in the worker.
        """
        if not segments:
            return
        with self._lock:
            job.new_segments.extend(segments)
            if len(job.new_segments) > MAX_NEW_SEGMENTS:
                del job.new_segments[:-MAX_NEW_SEGMENTS]

    # ── SSE Stream Data ───────────────────────────────────────────────

    def get_stream_data(self, job: Job, first: bool = False) -> dict:
        """Assemble data for the SSE stream.

        Reads ALL fields under the lock — status/result/error must be
        consistent with each other (see complete()/fail()).
        """
        with self._lock:
            data: dict[str, Any] = {
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "elapsed_sec": round(job.elapsed(), 1),
                "warning_count": job.warning_count,
                # For the progress map: current phase + completed steps
                # with duration. Small enough to send along every second.
                "phase": getattr(job, "phase", None),
                "step_timings": [dict(s) for s in job.step_timings],
            }

            if first:
                # Full log on reconnect
                new_log = list(job.log)
                job.new_log.clear()
            else:
                new_log = list(job.new_log)
                job.new_log.clear()

            new_segs = list(job.new_segments)
            job.new_segments.clear()

            if new_log:
                data["new_log"] = new_log
            if new_segs:
                data["new_segments"] = new_segs

            if job.status == "done":
                data["result"] = job.result
                if job.log_file:
                    data["log_file"] = job.log_file
            elif job.status == "error":
                data["error"] = job.error
                if job.log_file:
                    data["log_file"] = job.log_file
                if job_retries(job):
                    data["will_retry"] = True

        return data

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.1f}min"
        return f"{secs / 3600:.1f}h"

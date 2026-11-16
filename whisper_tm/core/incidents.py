"""Keep incidents permanently — the memory for everything that went wrong.

Previously there were three places where a problem could be, and
none of them stuck:

* the job in memory — an hour after completion cleanup removes it,
* the protocol file per job — one text file per job, by the hundred
  thousand, that nobody searches,
* ``job_history`` — capped at 10,000 lines, warnings not included at all.

The trigger: a file was missing most of its voice vectors (GPU full), it
appeared as ✅ in the list, and the log contained many similar lines
from other days — which files they were could only be reconstructed via
accidentally still available logs. Hence a record that keeps errors
forever, so it can be seen anytime what went wrong.

Now every warning, every error, every OOM retry and every system warning
lands as a line in ``incidents`` (library.db, schema v19), with file
identity ``abs_path``, the step and the time — and the table is never
capped. The way in is :func:`report`; job events come automatically
via :func:`attach`.

Writing must NEVER disturb a run: every error here is swallowed
and only logged.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("whisper_tm.incidents")

KINDS = ("warning", "error", "oom", "retry", "system")


def _db():
    import whisper_tm.extensions as ext

    return ext.db


def report_it(
    kind: str,
    text: str,
    *,
    abs_path: str = "",
    filename: str = "",
    job_id: str = "",
    step: str = "",
    count: int = 1,
    log_file: str = "",
) -> int | None:
    """Record an incident. Returns ID, None if it failed."""
    db = _db()
    if db is None:
        return None
    try:
        return db.report_incident(
            kind,
            text,
            abs_path=abs_path,
            filename=filename,
            job_id=job_id,
            step=step,
            count=count,
            log_file=log_file,
        )
    except Exception as e:
        logger.debug("Incident not saved (%s): %s", kind, e)
        return None


def _step(job) -> str:
    ph = getattr(job, "phase", None)
    if isinstance(ph, dict):
        return str(ph.get("title") or "")
    return ""


def job_event(event: str, job, *args) -> None:
    """Callback for ``JobManager.on_event``: log warnings and errors.

    "Cancelled" is not an incident — that was the user. An error at
    completion carries the log file name so you can go from incident
    to full protocol.
    """
    try:
        if event == "warn":
            text = args[0] if args else ""
            report_it(
                "warning",
                text,
                abs_path=getattr(job, "abs_path", ""),
                filename=job.filename,
                job_id=job.job_id,
                step=_step(job),
            )
        elif event == "error":
            error = str(job.error or "")
            if error.strip().lower() == "cancelled":
                return
            from whisper_tm.jobs import job_retries

            if job_retries(job):
                # Not an error of the file: the queue takes it up again and
                # books the incident itself (an OOM, with its attempt) —
                # an "error" beside it read as a failed file.
                return
            report_it(
                "error",
                error,
                abs_path=getattr(job, "abs_path", ""),
                filename=job.filename,
                job_id=job.job_id,
                step=_step(job),
                log_file=getattr(job, "log_file", ""),
            )
        elif event == "system":
            report_it("system", args[0] if args else "")
    except Exception as e:
        logger.debug("Incident callback: %s", e)


def attach(job_manager) -> None:
    """Call once at app startup."""
    job_manager.on_event(job_event)


def since(days: float) -> float:
    return time.time() - float(days) * 86400.0

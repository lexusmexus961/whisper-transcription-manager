"""
Exports that take long run as jobs: "Export · N files" with progress and
cancel, the result kept for a day to download.

Media exports (each file through ffmpeg) always run here, text exports
of very many files, and clips made in a recording's Clips tab (kind
"clip"). A job runs in a thread of its own — exports read the
disk and use the CPU, not the GPU the transcription queue works on, so
they don't wait behind it. Jobs live in memory; their results lie in the
data folder (``exports/``) until downloaded or a day old.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

KEEP_SECONDS = 24 * 3600
MAX_RUNNING = 2
#: Jobs that may wait for a slot — beyond, a new one is refused (60 requests
#: made 60 waiting jobs and threads)
MAX_WAITING = 8
#: Free space the exports folder must have before a job starts: a media
#: export copies its recordings there and then into a ZIP
MIN_FREE_BYTES = 2 * 1024**3
#: How often finished results are tidied without a new job
TIDY_EVERY = 3600


class Refused(RuntimeError):
    """A job not taken; ``status`` is the HTTP answer for it."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    id: str
    label: str
    total: int
    status: str = "waiting"  # waiting | running | done | error | cancelled
    done: int = 0
    progress: float = 0.0  # 0..100 over all files
    message: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    result: str = ""  # file name of the download
    size: int = 0
    skipped: list = field(default_factory=list)  # [name, why]
    kind: str = "export"  # export | clip (the Clips tab's "Make the clip")
    on_done: object = None  # called with the job when it is done
    zip_name: str = ""  # the ZIP's name when there are several files
    stop: bool = False

    def public(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "id",
                "label",
                "total",
                "status",
                "done",
                "progress",
                "message",
                "error",
                "created",
                "finished",
                "result",
                "size",
                "skipped",
                "kind",
            )
        }


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_slots = threading.Semaphore(MAX_RUNNING)


def folder() -> Path:
    from whisper_tm.config import config

    d = Path(config.data_dir) / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


_tidier: threading.Thread | None = None


def start_tidier() -> None:
    """Tidy now and then every TIDY_EVERY seconds: "kept a day" holds without
    a new job, and results left from before a restart go too."""
    global _tidier
    if _tidier is not None:
        return

    def loop():
        while True:
            try:
                tidy()
            except Exception as e:  # noqa: BLE001 — the next round tries again
                logger.debug("Export tidy: %s", e)
            time.sleep(TIDY_EVERY)

    _tidier = threading.Thread(target=loop, daemon=True, name="export-tidy")
    _tidier.start()


def tidy() -> None:
    """Results older than a day go, and their jobs."""
    cutoff = time.time() - KEEP_SECONDS
    with _lock:
        for jid, j in list(_jobs.items()):
            if j.finished and j.finished < cutoff:
                shutil.rmtree(folder() / jid, ignore_errors=True)
                _jobs.pop(jid, None)
    for d in folder().iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff and d.name not in _jobs:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def start(
    label: str, items: list, work, kind: str = "export", on_done=None, zip_name: str = ""
) -> Job:
    """A job over ``items``: ``work(item, out_dir, on_progress, should_stop)``
    writes one item's file(s) into ``out_dir`` and returns their paths (or
    raises to skip the item with its reason). Raises :class:`Refused` when
    too many wait or the disk is nearly full."""
    tidy()
    with _lock:
        waiting = sum(1 for j in _jobs.values() if j.status == "waiting")
    if waiting >= MAX_WAITING:
        raise Refused(f"{waiting} exports are waiting already — try again when they are done", 429)
    try:
        free = shutil.disk_usage(folder()).free
    except OSError:
        free = MIN_FREE_BYTES
    if free < MIN_FREE_BYTES:
        raise Refused(
            f"Only {free / 1024**3:.1f} GB free where exports are written "
            f"({folder()}) — at least {MIN_FREE_BYTES // 1024**3} GB are needed",
            507,
        )
    job = Job(
        id=uuid.uuid4().hex[:12],
        label=label,
        total=len(items),
        kind=kind,
        on_done=on_done,
        zip_name=zip_name,
    )
    with _lock:
        _jobs[job.id] = job
    threading.Thread(
        target=_run, args=(job, items, work), daemon=True, name=f"export-{job.id}"
    ).start()
    return job


def _run(job: Job, items: list, work) -> None:
    out_dir = folder() / job.id
    scratch = out_dir / "files"
    scratch.mkdir(parents=True, exist_ok=True)
    with _slots:
        if job.stop:
            job.status, job.finished = "cancelled", time.time()
            shutil.rmtree(out_dir, ignore_errors=True)
            return
        job.status = "running"
        made: list[Path] = []
        try:
            for i, item in enumerate(items):
                if job.stop:
                    break
                name = getattr(item, "name", None) or (
                    item.get("name") if isinstance(item, dict) else str(item)
                )
                job.message = f"{i + 1} of {job.total}: {name}"

                def progress(frac, i=i):
                    job.progress = round(
                        100 * (i + max(0.0, min(1.0, frac))) / max(1, job.total), 1
                    )

                try:
                    made += [Path(p) for p in work(item, scratch, progress, lambda: job.stop) or []]
                except Exception as e:  # noqa — one file that can't be exported doesn't stop the rest
                    if job.stop:
                        break
                    job.skipped.append([str(name), str(e)[:200]])
                job.done = i + 1
                job.progress = round(100 * job.done / max(1, job.total), 1)
            if job.stop:
                job.status = "cancelled"
                shutil.rmtree(out_dir, ignore_errors=True)
                return
            if not made:
                job.status, job.error = (
                    "error",
                    "nothing could be exported"
                    + (f" — {job.skipped[0][1]}" if job.skipped else ""),
                )
                shutil.rmtree(out_dir, ignore_errors=True)
                return
            if len(made) == 1:
                final = out_dir / made[0].name
                shutil.move(str(made[0]), final)
            else:
                final = out_dir / (job.zip_name or f"export_{time.strftime('%Y%m%d_%H%M')}.zip")
                taken: set[str] = set()
                with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as z:
                    for p in made:
                        arc = str(p.relative_to(scratch)) if p.is_relative_to(scratch) else p.name
                        k, n = arc, 2
                        while k in taken:
                            k = f"{Path(arc).stem} ({n}){Path(arc).suffix}"
                            n += 1
                        taken.add(k)
                        z.write(p, k)
                        if job.stop:
                            break
                if job.stop:
                    job.status = "cancelled"
                    shutil.rmtree(out_dir, ignore_errors=True)
                    return
            shutil.rmtree(scratch, ignore_errors=True)
            job.result = final.name
            job.size = final.stat().st_size
            job.status = "done"
            if callable(job.on_done):
                try:
                    job.on_done(job)
                except Exception as e:  # noqa — a note about the result must not fail it
                    logger.warning("Export %s: after-work failed: %s", job.id, e)
            job.message = f"{job.done} of {job.total} files" + (
                f" · {len(job.skipped)} skipped" if job.skipped else ""
            )
        except Exception as e:  # noqa
            logger.exception("Export %s failed", job.id)
            job.status, job.error = "error", str(e)[:300]
            shutil.rmtree(out_dir, ignore_errors=True)
        finally:
            job.finished = time.time()


def get(jid: str) -> Job | None:
    with _lock:
        return _jobs.get(jid)


def listing() -> list[dict]:
    with _lock:
        return [j.public() for j in sorted(_jobs.values(), key=lambda j: -j.created)]


def cancel(jid: str) -> bool:
    j = get(jid)
    if not j or j.status not in ("waiting", "running"):
        return False
    j.stop = True
    return True


def remove(jid: str) -> bool:
    j = get(jid)
    if not j or j.status in ("waiting", "running"):
        return False
    shutil.rmtree(folder() / jid, ignore_errors=True)
    with _lock:
        _jobs.pop(jid, None)
    return True


def result_path(jid: str) -> Path | None:
    j = get(jid)
    if not j or j.status != "done" or not j.result:
        return None
    p = folder() / jid / j.result
    return p if p.is_file() else None

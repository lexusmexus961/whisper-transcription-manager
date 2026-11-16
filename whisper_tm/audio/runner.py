"""Analyzing the library: during transcription, and in a background run.

**During transcription** (``audio_analysis.during_transcription``) the
transcription hands over the 16 kHz WAV it has just decoded
(:func:`after_decoding`): the analysis reads that — a minute of sound in a
few hundredths of a second — on a thread of its own, while the GPU works
on the transcript. A recording is never read twice for it.

**The background run** reads what was transcribed before, or never: every
recording the library scans found (``media_files``) with sound, below
``audio_analysis.folders`` (empty: the whole library), whose present
version (size, modification time) has no analysis yet — newest first,
``parallel_files`` at a time. It is a guest:

* its ffmpeg and its threads run at the lowest CPU priority (``nice 19``,
  and ``ionice`` idle where the disk honors it);
* ``max_read_mb_s`` caps what it reads (0: no cap) — on a network share
  the queue's own reads compete with it;
* ``pause_for_queue`` waits while a transcription runs;
* it waits while the disk the index is on has less than ``min_free_gb``.

It goes on where it stopped: a recording is stored when it is done, and a
run that was going when the server stopped starts again with the next start.
A run that ends looks for recordings that are gone (at most once a day) and
forgets them — only where their folder is still there, so a share that is
not mounted takes nothing along, and not at all when more than a fifth
seem gone.
``auto_run`` starts it by itself (and every ten minutes when new recordings
came) — off by default: on a large library on a network share the first
run reads everything once, which can take days.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

#: How long a scope is kept for the estimate
SCOPE_TTL_SEC = 60.0
#: Gone recordings are looked for at most this often, after a run
PRUNE_EVERY_SEC = 86400.0
#: ... and forgotten only while fewer than this share of them seem gone
PRUNE_MAX_SHARE = 0.2
_scope_lock = threading.Lock()
_scope: dict = {"key": None, "at": 0.0, "rows": []}


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("audio_analysis", name, default=default)


def _num(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(_cfg(name, default))
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


def settings() -> dict:
    from whisper_tm.core import paths
    from whisper_tm.core.helpers import parse_bool

    folders = _cfg("folders", []) or []
    if isinstance(folders, str):
        folders = [f for f in folders.splitlines() if f.strip()]
    return {
        "enabled": parse_bool(_cfg("enabled", True)),
        "during_transcription": parse_bool(_cfg("during_transcription", True)),
        "auto_run": parse_bool(_cfg("auto_run", False)),
        "folders": [
            paths.canon(Path(f).expanduser()) or str(Path(f).expanduser()).rstrip("/")
            for f in folders
            if str(f).strip()
        ],
        "parallel_files": int(_num("parallel_files", 2, 1, 16)),
        "max_read_mb_s": _num("max_read_mb_s", 0, 0, 100_000),
        "pause_for_queue": parse_bool(_cfg("pause_for_queue", False)),
        "min_free_gb": _num("min_free_gb", 5, 0, 100_000),
    }


def enabled() -> bool:
    return settings()["enabled"]


def _lower_this_thread() -> None:
    """The calling thread at the lowest CPU priority (Linux: per thread)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass


# ── During transcription ─────────────────────────────────────────────────

_hook_pool: ThreadPoolExecutor | None = None
_hook_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _hook_pool
    with _hook_lock:
        if _hook_pool is None:
            _hook_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio-analysis")
        return _hook_pool


def after_decoding(wav: str | Path, source: str | Path, log=None) -> bool:
    """The transcription has decoded ``source`` to ``wav`` (16 kHz mono):
    analyze it from there, in the background. The WAV is opened right
    away — the transcription may replace or delete it (separated vocals,
    the work folder emptied), an open file keeps its sound. Returns whether
    an analysis was started. Never raises: the transcript matters more."""
    try:
        s = settings()
        if not (s["enabled"] and s["during_transcription"]):
            return False
        from whisper_tm.audio.store import get_store
        from whisper_tm.core import paths

        path = paths.canon(source)
        if not path:
            return False
        st = Path(source).stat()
        store = get_store()
        if store.is_current(path, st.st_size, st.st_mtime):
            return False
        fh = open(wav, "rb")  # noqa: SIM115 — closed by the analysis
    except Exception as e:  # noqa: BLE001
        logger.info("Audio analysis of %s not started: %s", source, e)
        return False
    _pool().submit(_analyze_open_wav, fh, path, st.st_size, st.st_mtime, log)
    return True


def _analyze_open_wav(fh, path: str, size: int, mtime: float, log) -> None:
    from whisper_tm.audio.analyze import analyze_wav
    from whisper_tm.audio.store import get_store

    _lower_this_thread()
    try:
        res = analyze_wav(get_store(), fh, path, size, mtime)
        msg = (
            f"Audio analysis: {res['duration'] / 60:.1f} min of sound in "
            f"{res['seconds']:.1f} s (fingerprints, waveform, loudness)"
        )
        logger.info("%s — %s", msg, Path(path).name)
        if log:
            try:
                log(msg)
            except Exception:  # noqa: BLE001 — the job may be over
                pass
    except Exception as e:  # noqa: BLE001 — a background run tries it again
        logger.warning("Audio analysis of %s failed: %s", Path(path).name, e)
    finally:
        fh.close()


# ── The scope ────────────────────────────────────────────────────────────


def _main_db() -> Path:
    from whisper_tm.config import config

    return Path(config.db_path)


def scope(s: dict | None = None, fresh: bool = False) -> list[tuple]:
    """Recordings without a present analysis: ``[(path, size, mtime,
    duration)]``, newest first — read from the main database, compared
    with audio.db in SQL (one read-only connection, the store attached)."""
    from whisper_tm.audio.store import get_store

    s = s or settings()
    key = tuple(s["folders"])
    with _scope_lock:
        if not fresh and _scope["key"] == key and time.monotonic() - _scope["at"] < SCOPE_TTL_SEC:
            return _scope["rows"]
    store = get_store()
    main = _main_db()
    if not main.exists():
        return []
    con = sqlite3.connect(f"file:{main}?mode=ro", uri=True, timeout=30)
    try:
        con.execute("ATTACH DATABASE ? AS a", (f"file:{store.path}?mode=ro",))
    except sqlite3.OperationalError:
        con.close()
        con = sqlite3.connect(f"file:{main}?mode=ro", uri=True, timeout=30)
        con.execute("ATTACH DATABASE ? AS a", (str(store.path),))
    try:
        rows = con.execute(
            "SELECT m.abs_path, m.file_size, m.modified_at, m.duration_sec "
            "FROM media_files m LEFT JOIN a.files f ON f.path = m.abs_path "
            "WHERE m.abs_path != '' AND coalesce(m.has_audio, 1) != 0 "
            "AND (f.id IS NULL OR f.size != m.file_size OR abs(f.mtime - m.modified_at) >= 0.01) "
            "ORDER BY m.modified_at DESC"
        ).fetchall()
    finally:
        con.close()
    folders = [f + "/" for f in s["folders"]]
    if folders:
        rows = [r for r in rows if any(r[0].startswith(f) or r[0] == f[:-1] for f in folders)]
    with _scope_lock:
        _scope.update(key=key, at=time.monotonic(), rows=rows)
    return rows


def estimate(fresh: bool = False) -> dict:
    """What is left: recordings, hours of sound and bytes to read."""
    rows = scope(fresh=fresh)
    return {
        "files": len(rows),
        "hours": round(sum(float(r[3] or 0) for r in rows) / 3600.0, 1),
        "bytes": int(sum(int(r[1] or 0) for r in rows)),
    }


def forget_scope() -> None:
    with _scope_lock:
        _scope["at"] = 0.0


def prune_missing(store, stop=None) -> int:
    """Forget the recordings that are gone from disk (see above). Returns
    how many."""
    with store.connect() as c:
        paths = [r[0] for r in c.execute("SELECT path FROM files")]
    gone = []
    for p in paths:
        if stop is not None and stop():
            return 0
        if not os.path.exists(p) and os.path.isdir(os.path.dirname(p)):
            gone.append(p)
    if not gone:
        return 0
    if len(gone) > PRUNE_MAX_SHARE * len(paths):
        logger.warning(
            "Audio analysis: %d of %d recordings seem gone — none forgotten (a share half there?)",
            len(gone),
            len(paths),
        )
        return 0
    for p in gone:
        store.remove(p)
    logger.info("Audio analysis: %d recordings gone from disk, forgotten", len(gone))
    return len(gone)


# ── The run ──────────────────────────────────────────────────────────────


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def _queue_busy() -> bool:
    try:
        from whisper_tm.core import queue as _q

        return _q.busy() and not _q.is_paused()
    except Exception:  # noqa: BLE001
        return False


class Runner:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state: dict = {"running": False}

    def status(self) -> dict:
        with self._lock:
            st = dict(self.state)
            st["current"] = list(st.get("current") or [])
        if st.get("running") and st.get("started"):
            secs = max(1.0, time.time() - st["started"] - st.get("waited_sec", 0.0))
            done_h = st.get("hours_done", 0.0)
            st["speed"] = round(done_h * 3600.0 / secs, 1)  # seconds of sound a second
            left = max(0.0, st.get("hours_total", 0.0) - done_h)
            st["eta_sec"] = round(left * 3600.0 / st["speed"]) if st["speed"] > 0 else None
        return st

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def _add(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                self.state[k] = self.state.get(k, 0) + v

    def running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def start(self, reason: str = "") -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self.state = {
                "running": True,
                "stopping": False,
                "started": time.time(),
                "reason": reason,
                "files_done": 0,
                "files_total": 0,
                "hours_done": 0.0,
                "hours_total": 0.0,
                "bytes_total": 0,
                "errors": 0,
                "last_error": "",
                "skipped": 0,
                "current": [],
                "message": "Starting …",
                "waiting": None,
                "waited_sec": 0.0,
            }
            self._thread = threading.Thread(
                target=self._run, name="audio-analysis-run", daemon=True
            )
            self._thread.start()
        return True

    def stop(self, wanted: bool = False) -> bool:
        """Stop after the pieces in hand. ``wanted``: start again with the
        next server start (a shutdown, not the user's stop)."""
        with self._lock:
            alive = bool(self._thread and self._thread.is_alive())
            if alive:
                self.state["stopping"] = True
                self.state["message"] = "Stopping …"
        if not wanted:
            try:
                from whisper_tm.audio.store import get_store

                get_store().set_meta("run_wanted", False)
            except Exception:  # noqa: BLE001
                pass
        self._stop.set()
        return alive

    def join(self, timeout: float = 10.0) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _wait_reason(self, s: dict, store) -> str | None:
        if _free_gb(store.path.parent) < s["min_free_gb"]:
            return f"less than {s['min_free_gb']:g} GB free on the index's disk"
        if s["pause_for_queue"] and _queue_busy():
            return "a transcription is running"
        return None

    def _run(self) -> None:
        from whisper_tm.audio.store import get_store

        _lower_this_thread()
        store = get_store()
        store.set_meta("run_wanted", True)
        s = settings()
        try:
            rows = scope(s, fresh=True)
            self._set(
                files_total=len(rows),
                hours_total=round(sum(float(r[3] or 0) for r in rows) / 3600.0, 2),
                bytes_total=int(sum(int(r[1] or 0) for r in rows)),
                message=(f"{len(rows):,} recordings to analyze" if rows else "Nothing to analyze"),
            )
            todo = iter(rows)
            todo_lock = threading.Lock()
            per_file_rate = (
                s["max_read_mb_s"] * 1e6 / s["parallel_files"] if s["max_read_mb_s"] > 0 else 0.0
            )

            def take():
                with todo_lock:
                    return next(todo, None)

            def worker():
                _lower_this_thread()
                while not self._stop.is_set():
                    why = self._wait_reason(s, store)
                    if why:
                        self._set(waiting=why, message=f"Waiting: {why}")
                        t0 = time.time()
                        self._stop.wait(30.0)
                        self._add(waited_sec=time.time() - t0)
                        continue
                    self._set(waiting=None)
                    row = take()
                    if row is None:
                        return
                    self._one(store, row, per_file_rate)

            threads = [
                threading.Thread(target=worker, name=f"audio-run-{i}", daemon=True)
                for i in range(s["parallel_files"])
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            store.flush()
            stopped = self._stop.is_set()
            last_pruned = float(store.meta("pruned_at", 0) or 0)
            if not stopped and time.time() - last_pruned > PRUNE_EVERY_SEC:
                self._set(message="Looking for recordings that are gone …")
                n = prune_missing(store, stop=self._stop.is_set)
                if not self._stop.is_set():
                    store.set_meta("pruned_at", time.time())
                    self._set(forgotten=n)
            if not stopped:
                store.set_meta("run_wanted", False)
            st = self.status()
            self._set(
                message=("Stopped" if stopped else "Done")
                + f" — {st.get('files_done', 0):,} recordings analyzed"
                + (f", {st['errors']:,} unreadable" if st.get("errors") else "")
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Audio analysis run failed")
            self._set(message=f"Failed: {e}", last_error=str(e)[:300])
        finally:
            forget_scope()
            self._set(running=False, stopping=False, current=[], waiting=None, finished=time.time())

    def _one(self, store, row: tuple, per_file_rate: float) -> None:
        from whisper_tm.audio.analyze import analyze_file

        path, size, _mtime, duration = row
        name = Path(path).name
        try:
            st = os.stat(path)
        except OSError:
            self._add(skipped=1)
            return
        if store.is_current(path, st.st_size, st.st_mtime):
            self._add(skipped=1, hours_done=float(duration or 0) / 3600.0)
            return
        with self._lock:
            self.state["current"] = [*self.state.get("current", []), name]
        started = time.monotonic()

        def pace(seconds_read: float) -> None:
            # What ffmpeg has read of the file so far, about: its share of
            # the length. Sleeping here leaves the pipe full — ffmpeg waits.
            if per_file_rate <= 0 or not duration:
                return
            read = st.st_size * min(1.0, seconds_read / float(duration))
            ahead = read / per_file_rate - (time.monotonic() - started)
            if ahead > 0:
                self._stop.wait(min(ahead, 5.0))

        try:
            res = analyze_file(store, path, source="run", stop=self._stop.is_set, pace=pace)
            self._add(files_done=1, hours_done=float(res["duration"] or duration or 0) / 3600.0)
            if res["error"]:
                self._add(errors=1)
                self._set(last_error=f"{name}: {res['error']}"[:300])
        except InterruptedError:
            pass
        except Exception as e:  # noqa: BLE001 — the next recording
            logger.warning("Audio analysis of %s failed: %s", name, e)
            self._add(errors=1)
            self._set(last_error=f"{name}: {e}"[:300])
        finally:
            with self._lock:
                cur = list(self.state.get("current", []))
                if name in cur:
                    cur.remove(name)
                self.state["current"] = cur
            if not self._stop.is_set():
                st_ = self.status()
                self._set(
                    message=f"{st_.get('files_done', 0):,} of {st_.get('files_total', 0):,} "
                    f"recordings analyzed"
                )


RUNNER = Runner()
_auto_thread: threading.Thread | None = None


def on_server_start() -> None:
    """After the server has started: resume a run that was going when it
    stopped, and with ``auto_run`` look for new recordings now and then."""
    global _auto_thread
    try:
        s = settings()
        if not s["enabled"]:
            return
        from whisper_tm.audio.store import get_store

        store = get_store()
        if store.meta("run_wanted", False):
            RUNNER.start(reason="resumed")
    except Exception as e:  # noqa: BLE001
        logger.info("Audio analysis: no resume (%s)", e)
    if _auto_thread is None or not _auto_thread.is_alive():
        _auto_thread = threading.Thread(target=_auto_loop, name="audio-auto-run", daemon=True)
        _auto_thread.start()


def _auto_loop() -> None:
    time.sleep(120)
    while True:
        try:
            s = settings()
            if s["enabled"] and s["auto_run"] and not RUNNER.running():
                left = estimate(fresh=True)
                if left["files"]:
                    logger.info(
                        "Audio analysis: %d recordings without an analysis — analyzing them",
                        left["files"],
                    )
                    RUNNER.start(reason="automatic")
        except Exception as e:  # noqa: BLE001 — tried again in ten minutes
            logger.info("Audio analysis (automatic): %s", e)
        time.sleep(600)


def on_shutdown() -> None:
    """Stop a run (it starts again with the next start) and write what is
    in memory."""
    try:
        if RUNNER.running():
            RUNNER.stop(wanted=True)
            RUNNER.join(15.0)
    except Exception:  # noqa: BLE001
        pass
    try:
        if _hook_pool is not None:
            _hook_pool.shutdown(wait=True, cancel_futures=False)
    except Exception:  # noqa: BLE001
        pass
    from whisper_tm.audio.store import close_all

    close_all()

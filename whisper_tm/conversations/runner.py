"""Counting the library's conversation statistics in the background.

Every transcript write counts its own numbers (store.py), so this run is
for what was transcribed before the feature existed, for transcripts whose
numbers were counted by an older version of the rules, and for the rare
write that failed to count. It reads only the segments table and the small
columns of ``transcripts`` — never a transcript file, never ``json_data``
— counts outside any transaction and writes each batch in one short
transaction, so the queue's writers never wait long for the database.

It runs inside the server process, where pure Python competes with the
request threads for the interpreter: ``nice`` does not help there, a pause
after every batch does. A large library takes minutes of one CPU core.

``auto_run`` starts it about a minute after the server started when
something is missing, and looks again every half hour; the Conversations
tab can start it, and count everything again, too.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from whisper_tm.conversations import store

logger = logging.getLogger(__name__)

#: Transcripts asked for at a time …
BATCH_IDS = 200
#: … but at most about this many segments counted between two pauses
BATCH_SEGMENTS = 20_000
#: The pause after each batch — gives the request threads the interpreter
PAUSE_SEC = 0.05
#: First look after a start, then every …
START_DELAY_SEC = 60.0
LOOK_EVERY_SEC = 1800.0


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("conversations", name, default=default)


def settings() -> dict:
    from whisper_tm.core.helpers import parse_bool

    try:
        share = float(_cfg("min_share", 0.05))
    except (TypeError, ValueError):
        share = 0.05
    return {
        "enabled": parse_bool(_cfg("enabled", True)),
        "auto_run": parse_bool(_cfg("auto_run", True)),
        "min_share": min(max(share, 0.0), 0.5),
    }


def _db():
    from whisper_tm.extensions import db

    return db


def _lower_this_thread() -> None:
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass


class Runner:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state: dict = {"running": False}

    def status(self) -> dict:
        with self._lock:
            st = dict(self.state)
        if st.get("running") and st.get("started") and st.get("done"):
            secs = max(1.0, time.time() - st["started"])
            rate = st["done"] / secs
            left = max(0, st.get("total", 0) - st["done"])
            st["eta_sec"] = round(left / rate) if rate > 0 else None
        return st

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def start(self, reason: str = "", recount: bool = False) -> bool:
        """Start a run (False when one is going). ``recount``: count every
        transcript again, not only what is missing."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self.state = {
                "running": True,
                "stopping": False,
                "started": time.time(),
                "finished": None,
                "reason": reason,
                "recount": recount,
                "done": 0,
                "total": 0,
                "errors": 0,
                "last_error": "",
                "message": "Starting …",
            }
            self._thread = threading.Thread(
                target=self._run, args=(recount,), name="conversation-stats-run", daemon=True
            )
            self._thread.start()
        return True

    def stop(self) -> bool:
        with self._lock:
            alive = bool(self._thread and self._thread.is_alive())
            if alive:
                self.state["stopping"] = True
                self.state["message"] = "Stopping …"
        self._stop.set()
        return alive

    def join(self, timeout: float = 10.0) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _run(self, recount: bool) -> None:
        _lower_this_thread()
        db = _db()
        done = errors = 0
        try:
            if db is None:
                raise RuntimeError("no database")
            if recount:
                with db.connection() as conn:
                    store.recount_all(conn)
                store.changed()
            with db.connection() as conn:
                total = store.stale_count(conn)
            self._set(total=total, message=f"Counting {total:,} transcripts …")
            logger.info(
                "Conversation statistics: counting %d transcripts (%s)",
                total,
                self.state.get("reason") or "manual",
            )
            after = 0
            while not self._stop.is_set():
                with db.connection() as conn:
                    ids = store.stale_ids(conn, after, BATCH_IDS)
                    if ids:
                        marks = ",".join("?" * len(ids))
                        sizes = dict(
                            conn.execute(
                                f"SELECT id, segment_count FROM transcripts WHERE id IN ({marks})",
                                ids,
                            ).fetchall()
                        )
                if not ids:
                    break
                batch, segs = [], 0
                for tid in ids:
                    if batch and segs + (sizes.get(tid) or 0) > BATCH_SEGMENTS:
                        break
                    batch.append(tid)
                    segs += sizes.get(tid) or 0
                after = batch[-1]
                with db.connection() as conn:
                    loaded = store.load_segments(conn, batch)
                counted = []
                for tid in batch:
                    item = loaded.get(tid)
                    if item is None:
                        continue
                    try:
                        counted.append((tid, item, store.compute(item["data"])))
                    except Exception as e:  # noqa: BLE001 — one transcript, not the run
                        errors += 1
                        self._set(errors=errors, last_error=str(e)[:200])
                        logger.warning(
                            "Conversation statistics: transcript %s not counted: %s", tid, e
                        )
                written = self._write(db, counted)
                done += len(batch)
                self._set(done=done, message=f"Counted {done:,} of {total:,} transcripts")
                if written:
                    store.changed()
                self._stop.wait(PAUSE_SEC)
            stopped = self._stop.is_set()
            self._set(
                message=(
                    f"Stopped after {done:,} transcripts"
                    if stopped
                    else f"Done: {done:,} transcripts counted"
                )
            )
            logger.info(
                "Conversation statistics: %s %d transcripts (%d errors)",
                "stopped after" if stopped else "counted",
                done,
                errors,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Conversation statistics: run failed: %s", e)
            self._set(errors=errors + 1, last_error=str(e)[:200], message=f"Failed: {e}")
        finally:
            self._set(running=False, stopping=False, finished=time.time())
            try:
                if db is not None:
                    db.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _write(db, counted) -> int:
        """One short transaction per batch. A transcript written since it
        was read (other segment or character count) keeps the numbers its
        own write counted."""
        if not counted:
            return 0
        n = 0
        with db.connection() as conn:
            for tid, item, stats in counted:
                row = conn.execute(
                    "SELECT segment_count, char_count FROM transcripts WHERE id = ?", (tid,)
                ).fetchone()
                if (
                    row is None
                    or (row[0] or 0) != item["seg_count"]
                    or (row[1] or 0) != item["char_count"]
                ):
                    continue
                store.write(
                    conn,
                    tid,
                    stats,
                    seg_count=item["seg_count"],
                    char_count=item["char_count"],
                    abs_path=item["abs_path"],
                )
                n += 1
        return n


RUNNER = Runner()
_auto_thread: threading.Thread | None = None


def on_server_start() -> None:
    """After the server has started: with ``auto_run``, count what is
    missing now and every half hour."""
    global _auto_thread
    if _auto_thread is None or not _auto_thread.is_alive():
        _auto_thread = threading.Thread(
            target=_auto_loop, name="conversation-stats-auto", daemon=True
        )
        _auto_thread.start()


def _auto_loop() -> None:
    time.sleep(START_DELAY_SEC)
    while True:
        try:
            s = settings()
            db = _db()
            if s["enabled"] and s["auto_run"] and db is not None and not RUNNER.running():
                with db.connection() as conn:
                    left = store.stale_count(conn)
                if left:
                    RUNNER.start(reason="automatic")
        except Exception as e:  # noqa: BLE001 — tried again later
            logger.info("Conversation statistics (automatic): %s", e)
        time.sleep(LOOK_EVERY_SEC)


def on_shutdown() -> None:
    try:
        if RUNNER.running():
            RUNNER.stop()
            RUNNER.join(10.0)
    except Exception:  # noqa: BLE001
        pass

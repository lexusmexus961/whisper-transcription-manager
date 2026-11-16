"""
LLM queue: ONE worker for all LLM tasks.

Why a separate queue:

* **Closing must not abort.** Summaries were attached to the browser's
  SSE stream — close modal, work gone. Now each task continues server-side
  and the browser just hangs on.
* **One slot.** The llama.cpp server processes one request at a time.
  Ten parallel folder jobs would throttle each other (and cause timeouts) —
  so strictly sequential.
* **Folder batches.** "Summarize whole folder" simply queues n entries.

Persistent in table ``llm_queue`` (schema v17): restart loses nothing,
running entries are set back to ``pending`` on startup.

Intermediate results (map sections, chunk results) are also written —
the UI can expand progress and read along without waiting for the
final result.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

# Task types
TYPE_SUMMARY = "summary"
TYPE_CORRECTION = "correction"
TYPE_TRANSLATION = "translation"
#: One task over many recordings (core/analysis.py)
TYPE_ANALYSIS = "analysis"
#: "Find everything" for a situation, passage by passage (rag/deep_search.py)
TYPE_DEEP_SEARCH = "deep_search"
#: Describing stills of Images & sounds (media/describe.py)
TYPE_MEDIA_DESCRIBE = "media_describe"
#: Three short moments of a recording for a clip (the Clips tab)
TYPE_BEST_MOMENTS = "best_moments"
TYPES = (
    TYPE_SUMMARY,
    TYPE_CORRECTION,
    TYPE_TRANSLATION,
    TYPE_ANALYSIS,
    TYPE_DEEP_SEARCH,
    TYPE_MEDIA_DESCRIBE,
    TYPE_BEST_MOMENTS,
)
#: The kinds that are one file each — what the queue routes take; an
#: analysis and a deep search start through their own (api/analysis.py,
#: api/rag.py)
FILE_TYPES = (TYPE_SUMMARY, TYPE_CORRECTION, TYPE_TRANSLATION)

LABEL = {
    TYPE_SUMMARY: "Summary",
    TYPE_CORRECTION: "Correction",
    TYPE_TRANSLATION: "Translation",
    TYPE_ANALYSIS: "Analysis",
    TYPE_DEEP_SEARCH: "Deep search",
    TYPE_MEDIA_DESCRIBE: "Describe pictures",
    TYPE_BEST_MOMENTS: "Best moments",
}


def _db():
    from whisper_tm.extensions import db

    return db


class TaskRefused(RuntimeError):
    """A task that can't run, for a reason its message states (a set without
    transcripts …): failed like any error, logged without a traceback —
    a full ERROR for an expected refusal would be noise."""


def server_down(error: BaseException) -> bool:
    """The language model can't be reached right now — no connection, a
    name that doesn't resolve, the server still loading its model (503) —
    which is no fault of the entry. A slow answer (read timeout) or any
    other 5xx still is one."""
    from whisper_tm.core.llm_client import LlmConnectionError

    seen: set[int] = set()
    e: BaseException | None = error
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, LlmConnectionError):
            return str(e).startswith(
                ("Connection failed", "Timeout on connection", "LLM server 503")
            )
        e = e.__cause__ or e.__context__
    return False


#: Waiting for an unreachable language model: the next try after 30 s, then
#: twice as long each time, at most this long (seconds)
WAIT_MAX_SEC = 600


class LlmQueue:
    """Sequential queue for LLM tasks."""

    def __init__(self):
        self._lock = threading.Lock()
        # Held while the worker takes the next entry, and by held(): the
        # database can be swapped out without the worker inside it
        self._take_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._worker: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._current: dict | None = None  # {id, kind, filename, …}
        self._abort = False  # Abort the RUNNING entry
        # Cancels of entries the worker has taken but not started yet:
        # 'running' in the database, not yet _current
        self._cancelled: set[int] = set()
        self._log: list[str] = []
        # The language model can't be reached: the entries wait instead of
        # failing one after another — {since, next_try, tries, reason}
        self._waiting: dict | None = None

    # ── Start / Stop ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start worker and release stuck entries."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._cleanup_after_restart()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="llm-queue")
        self._worker.start()
        logger.info("LLM queue started")

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()

    @contextlib.contextmanager
    def held(self):
        """Nothing is taken from the queue inside this block (the database
        is being deleted). The pause state found is restored after."""
        was = self._paused
        self._paused = True
        try:
            with self._take_lock:
                yield
        finally:
            self._paused = was
            self._wakeup.set()

    def busy(self) -> bool:
        return self._current is not None

    def _cleanup_after_restart(self) -> None:
        """`running` from earlier process → back to `pending`."""
        db = _db()
        if not db:
            return
        try:
            with db.connection() as conn:
                cur = conn.execute(
                    "UPDATE llm_queue SET status='pending', progress=0, "
                    "message='Requeued after restart' "
                    "WHERE status='running'"
                )
                if cur.rowcount:
                    logger.info("LLM queue: %d interrupted entries requeued", cur.rowcount)
        except Exception as e:
            logger.debug("LLM queue cleanup: %s", e)

    # ── Queue / Manage ─────────────────────────────────────────

    def enqueue(
        self, kind: str, filename: str, params: dict | None = None, *, front: bool = False
    ) -> int:
        """Queue task; returns entry ID (0 = not possible)."""
        if kind not in TYPES:
            raise ValueError(f"Unknown LLM task type: {kind}")
        db = _db()
        if not db:
            return 0
        with db.connection() as conn:
            if front:
                row = conn.execute(
                    "SELECT MIN(position) p FROM llm_queue WHERE status='pending'"
                ).fetchone()
                pos = (row["p"] if row and row["p"] is not None else 0) - 1
            else:
                row = conn.execute("SELECT MAX(position) p FROM llm_queue").fetchone()
                pos = (row["p"] if row and row["p"] is not None else 0) + 1
            cur = conn.execute(
                "INSERT INTO llm_queue (kind, filename, params_json, status, "
                "position, created_at) VALUES (?, ?, ?, 'pending', ?, ?)",
                (kind, filename, json.dumps(params or {}, ensure_ascii=False), pos, time.time()),
            )
            entry_id = int(cur.lastrowid or 0)
        self._log_line(f"{LABEL[kind]} queued: {filename.split('/')[-1]}")
        self._wakeup.set()
        return entry_id

    def enqueue_many(
        self, kind: str, filenames: list[str], params: dict | None = None
    ) -> list[int]:
        return [self.enqueue(kind, fn, params) for fn in filenames]

    def cancel(self, entry_id: int) -> bool:
        """Remove pending entry or abort running one."""
        db = _db()
        if not db:
            return False
        with self._lock:
            currently_running = (self._current or {}).get("id") == entry_id
            if currently_running:
                self._abort = True
                return True
        with db.connection() as conn:
            cur = conn.execute(
                "UPDATE llm_queue SET status='cancelled', "
                "finished_at=? WHERE id=? AND status='pending'",
                (time.time(), entry_id),
            )
            if cur.rowcount > 0:
                return True
            taken = conn.execute(
                "SELECT 1 FROM llm_queue WHERE id=? AND status='running'", (entry_id,)
            ).fetchone()
        if not taken:
            return False
        # Taken by the worker between the two looks: it is either current
        # by now (abort it) or about to be (_run_one drops it)
        with self._lock:
            if (self._current or {}).get("id") == entry_id:
                self._abort = True
            else:
                self._cancelled.add(entry_id)
        return True

    def retry(self, entry_id: int) -> bool:
        db = _db()
        if not db:
            return False
        with db.connection() as conn:
            cur = conn.execute(
                "UPDATE llm_queue SET status='pending', progress=0, "
                "message='', error='' WHERE id=? AND status IN "
                "('error','cancelled')",
                (entry_id,),
            )
        if cur.rowcount:
            self._wakeup.set()
        return cur.rowcount > 0

    def clear(self, what: str = "done") -> int:
        """Remove finished entries (done/error/cancelled) or pending ones."""
        db = _db()
        if not db:
            return 0
        statuses = ("done", "error", "cancelled") if what == "done" else ("pending",)
        with db.connection() as conn:
            cur = conn.execute(
                f"DELETE FROM llm_queue WHERE status IN "  # noqa: S608
                f"({','.join('?' * len(statuses))})",
                statuses,
            )
            return cur.rowcount

    def pause(self, on: bool) -> None:
        if on != self._paused:
            self._log_line("Paused" if on else "Resumed")
        self._paused = on
        if not on:
            with self._lock:
                # Resume while waiting for the language model: try now
                if self._waiting:
                    self._waiting["next_try"] = 0.0
            self._wakeup.set()

    def _wait_for_server(self, entry: dict, error: BaseException) -> None:
        """The language model didn't answer: the entry goes back to wait,
        and so does every one after it, until a try reaches the server."""
        now = time.time()
        with self._lock:
            tries = (self._waiting or {}).get("tries", 0) + 1
            delay = min(WAIT_MAX_SEC, 30 * 2 ** (tries - 1))
            self._waiting = {
                "since": (self._waiting or {}).get("since", now),
                "next_try": now + delay,
                "tries": tries,
                "reason": str(error)[:200],
            }
        # It starts over: what it got so far would show twice
        self._update(
            entry["id"],
            status="pending",
            progress=0,
            started_at=0,
            parts_json="[]",
            partial_text="",
            message="Waiting for the language model",
        )
        if tries == 1:
            self._log_line(
                f"⏸ The language model can't be reached ({str(error)[:100]}) — "
                "the tasks wait and go on once it answers"
            )
        logger.info(
            "LLM queue: language model unreachable (%s) — next try in %d s", str(error)[:120], delay
        )

    def _server_answered(self) -> None:
        with self._lock:
            was = self._waiting
            self._waiting = None
        if was:
            self._log_line("▶ The language model answers again — going on")

    # ── State for UI ────────────────────────────────────────────

    def snapshot(self, limit: int = 100) -> dict:
        db = _db()
        entries: list[dict] = []
        if db:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, kind, filename, status, progress, message, "
                    "error, created_at, started_at, finished_at, params_json "
                    "FROM llm_queue ORDER BY "
                    "CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 "
                    "ELSE 2 END, position, id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            for r in rows:
                try:
                    params = json.loads(r["params_json"] or "{}")
                except json.JSONDecodeError:
                    params = {}
                entries.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "filename": r["filename"],
                        "status": r["status"],
                        "progress": r["progress"],
                        "message": r["message"],
                        "error": r["error"],
                        "created_at": r["created_at"],
                        "started_at": r["started_at"],
                        "finished_at": r["finished_at"],
                        "params": params,
                    }
                )
        # Counted over the whole table — the entry list above is capped, and a
        # folder batch of 1,000 tasks must not read "100 waiting".
        counter = {"pending": 0, "running": 0, "done": 0, "error": 0, "cancelled": 0}
        if db:
            with db.connection() as conn:
                for r in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM llm_queue GROUP BY status"
                ):
                    counter[r["status"]] = r["n"]
        with self._lock:
            current = dict(self._current) if self._current else None
            waiting = dict(self._waiting) if self._waiting else None
        if waiting:
            waiting["next_in"] = max(0, round(waiting.pop("next_try") - time.time()))
        return {
            "items": entries,
            "counts": counter,
            "current": current,
            "paused": self._paused,
            "waiting": waiting,
            "log": self._log[-40:],
        }

    def queue_position(self, entry_id: int) -> tuple[int, int, str]:
        """(Position, number waiting, file currently running).

        A "waiting..." without context is no information — the UI should
        tell WHAT is being waited for (e.g. an old large job blocking the
        queue, while the dialog only said "waiting").
        """
        db = _db()
        if not db:
            return (0, 0, "")
        with db.connection() as conn:
            pending = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM llm_queue WHERE status='pending' ORDER BY position, id"
                ).fetchall()
            ]
            running = conn.execute(
                "SELECT filename FROM llm_queue WHERE status='running' LIMIT 1"
            ).fetchone()
        slot = (pending.index(entry_id) + 1) if entry_id in pending else 0
        return (slot, len(pending), running["filename"] if running else "")

    def entry(self, entry_id: int) -> dict | None:
        """An entry including intermediate and final results."""
        db = _db()
        if not db:
            return None
        with db.connection() as conn:
            r = conn.execute("SELECT * FROM llm_queue WHERE id=?", (entry_id,)).fetchone()
        if not r:
            return None

        def _j(field_name, default):
            try:
                return json.loads(r[field_name] or "null") or default
            except (json.JSONDecodeError, IndexError, KeyError):
                return default

        slot, pending, running_file = (
            self.queue_position(entry_id) if r["status"] == "pending" else (0, 0, "")
        )
        return {
            "id": r["id"],
            "kind": r["kind"],
            "filename": r["filename"],
            "status": r["status"],
            "progress": r["progress"],
            "queue_slot": slot,
            "pending": pending,
            "currently_running": running_file,
            "message": r["message"],
            "error": r["error"],
            "params": _j("params_json", {}),
            "partial_results": _j("parts_json", []),
            "partial_text": r["partial_text"] if "partial_text" in r.keys() else "",
            "result": _j("result_json", None),
            "created_at": r["created_at"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
        }

    def open_entry(self, kind: str, filename: str) -> dict | None:
        """Is a task for this file already running or waiting?

        The UI attaches to it when opening a dialog — so a returning
        user finds their running job again.
        """
        db = _db()
        if not db:
            return None
        with db.connection() as conn:
            r = conn.execute(
                "SELECT id FROM llm_queue WHERE kind=? AND filename=? "
                "AND status IN ('pending','running') ORDER BY id DESC LIMIT 1",
                (kind, filename),
            ).fetchone()
        return self.entry(r["id"]) if r else None

    def summary_waiting(self, abs_path: str) -> bool:
        """Is a summary of the file at ``abs_path`` (canonical) waiting or
        running? An entry's name is relative to its ``media_dir`` — the
        root it came from — or to the open folder (core/llm_tasks.py)."""
        import os

        from whisper_tm.config import config
        from whisper_tm.core import paths

        db = _db()
        if not db or not abs_path:
            return False
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT filename, params_json FROM llm_queue WHERE kind = ? "
                "AND status IN ('pending', 'running')",
                (TYPE_SUMMARY,),
            ).fetchall()
        for r in rows:
            try:
                params = json.loads(r["params_json"] or "{}")
            except (TypeError, ValueError):
                params = {}
            root = (params.get("media_dir") if isinstance(params, dict) else "") or str(
                config.media_dir
            )
            if paths.canon(os.path.join(root, r["filename"] or "")) == abs_path:
                return True
        return False

    # ── Write Progress (from worker) ─────────────────

    def _update(self, entry_id: int, **fields) -> None:
        """Write an entry's fields. A status write (the outcome, the result)
        is retried on a locked database and its loss logged as a warning —
        swallowed at debug level, a locked database at the end of a long
        summary would leave the entry 'running' with its result gone, and
        nothing would say why. A progress write may simply miss."""
        db = _db()
        if not db or not fields:
            return
        allowed = {
            "status",
            "progress",
            "message",
            "error",
            "result_json",
            "parts_json",
            "partial_text",
            "started_at",
            "finished_at",
        }
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        sql = ", ".join(f"{k}=?" for k in fields)
        attempts = 4 if "status" in fields else 1
        for attempt in range(1, attempts + 1):
            try:
                with db.connection() as conn:
                    conn.execute(
                        f"UPDATE llm_queue SET {sql} WHERE id=?",  # noqa: S608
                        (*fields.values(), entry_id),
                    )
                return
            except sqlite3.OperationalError as e:
                if attempt < attempts:
                    time.sleep(0.5 * attempt)
                    continue
                if "status" in fields:
                    logger.warning(
                        "LLM queue: entry %s could not be set to %r (%s)",
                        entry_id,
                        fields["status"],
                        e,
                    )
                else:
                    logger.debug("LLM queue update: %s", e)
            except Exception as e:
                if "status" in fields:
                    logger.warning(
                        "LLM queue: entry %s could not be set to %r (%s)",
                        entry_id,
                        fields["status"],
                        e,
                    )
                else:
                    logger.debug("LLM queue update: %s", e)
                return

    def _log_line(self, text: str) -> None:
        self._log.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        del self._log[:-200]

    # ── Worker ────────────────────────────────────────────────────────

    def _next_one(self) -> dict | None:
        """Get next pending entry."""
        db = _db()
        if not db:
            return None
        # One statement: a SELECT, then an unguarded UPDATE would overwrite
        # a cancel that landed in between, and the cancelled task would run.
        with db.connection() as conn:
            r = conn.execute(
                "UPDATE llm_queue SET status='running', started_at=?, "
                "progress=0, message='Starting...' "
                "WHERE id = (SELECT id FROM llm_queue WHERE status='pending' "
                "ORDER BY position, id LIMIT 1) AND status='pending' "
                "RETURNING id, kind, filename, params_json",
                (time.time(),),
            ).fetchone()
            if not r:
                return None
        try:
            params = json.loads(r["params_json"] or "{}")
        except json.JSONDecodeError:
            params = {}
        return {"id": r["id"], "kind": r["kind"], "filename": r["filename"], "params": params}

    def _loop(self) -> None:
        while self._running:
            if self._paused:
                self._wakeup.wait(2.0)
                self._wakeup.clear()
                continue
            with self._lock:
                wait = self._waiting["next_try"] - time.time() if self._waiting else 0
            if wait > 0:
                # The language model was down: nothing is taken until the
                # next try (Resume brings it forward)
                self._wakeup.wait(min(wait, 5.0))
                self._wakeup.clear()
                continue
            try:
                with self._take_lock:
                    entry = None if self._paused else self._next_one()
            except Exception as e:
                # Transient DB errors (e.g. "database is locked" when the
                # transcription worker is writing) must not kill the ONLY
                # LLM worker thread — otherwise all pending entries stand at
                # "waiting..." until restart.
                logger.warning("LLM queue: DB not readable (%s) — retrying in 5 s", e)
                self._wakeup.wait(5.0)
                self._wakeup.clear()
                continue
            if not entry:
                self._wakeup.wait(5.0)
                self._wakeup.clear()
                continue
            self._run_one(entry)

    def _run_one(self, entry: dict) -> None:
        """Run one taken entry and book its outcome."""
        with self._lock:
            dropped = entry["id"] in self._cancelled
            self._cancelled.discard(entry["id"])
            if not dropped:
                self._current = entry
                self._abort = False
        if dropped:
            self._update(
                entry["id"], status="cancelled", message="Cancelled", finished_at=time.time()
            )
            self._log_line(f"✕ Cancelled: {entry['filename'].split('/')[-1]}")
            return
        self._log_line(f"▶ {LABEL[entry['kind']]}: {entry['filename'].split('/')[-1]}")
        try:
            from whisper_tm.core import llm_tasks

            result = llm_tasks.execute(entry, self)
            self._server_answered()
            self._update(
                entry["id"],
                status="done",
                progress=100,
                message="Done",
                result_json=json.dumps(result, ensure_ascii=False),
                finished_at=time.time(),
            )
            self._log_line(f"✅ {LABEL[entry['kind']]}: {entry['filename'].split('/')[-1]}")
        except InterruptedError:
            self._update(
                entry["id"], status="cancelled", message="Cancelled", finished_at=time.time()
            )
            self._log_line(f"✕ Cancelled: {entry['filename'].split('/')[-1]}")
        except Exception as e:
            if server_down(e):
                # Not the entry's fault: it fails no more than the ones
                # after it would — they wait for the server instead
                self._wait_for_server(entry, e)
                return
            if isinstance(e, TaskRefused):
                logger.warning("LLM queue entry %s failed: %s", entry["id"], e)
            else:
                logger.exception("LLM queue entry %s failed", entry["id"])
            self._update(
                entry["id"],
                status="error",
                message="Failed",
                error=str(e)[:500],
                finished_at=time.time(),
            )
            self._log_line(f"❌ {entry['filename'].split('/')[-1]}: {str(e)[:120]}")
        finally:
            with self._lock:
                self._current = None

    # ── Used by task code ─────────────────────────────────────────

    def progress(self, entry_id: int, percent: int, message: str) -> None:
        """Report progress."""
        self._update(entry_id, progress=max(0, min(99, int(percent))), message=message[:300])
        with self._lock:
            if self._current and self._current["id"] == entry_id:
                self._current["progress"] = percent
                self._current["message"] = message

    def partial_result(self, entry_id: int, part: dict) -> None:
        """Append intermediate result (map section, chunk ...)."""
        db = _db()
        if not db:
            return
        try:
            with db.connection() as conn:
                r = conn.execute(
                    "SELECT parts_json FROM llm_queue WHERE id=?", (entry_id,)
                ).fetchone()
                try:
                    parts = json.loads(r["parts_json"] or "[]") if r else []
                except json.JSONDecodeError:
                    parts = []
                parts.append(part)
                conn.execute(
                    "UPDATE llm_queue SET parts_json=? WHERE id=?",
                    (json.dumps(parts, ensure_ascii=False), entry_id),
                )
        except Exception as e:
            logger.debug("Intermediate result not savable: %s", e)

    def partial_text(self, entry_id: int, text: str) -> None:
        """Write text AS it is being created.

        Summaries no longer depend on the browser — without these
        intermediate states the dialog would see nothing until the end.
        The caller throttles (writing per token would be silly); here we
        just overwrite.
        """
        self._update(entry_id, partial_text=text[-20000:])

    def abort_requested(self, entry_id: int) -> bool:
        with self._lock:
            return self._abort and (self._current or {}).get("id") == entry_id


# Module singleton — like q in core/queue.py
llm_queue = LlmQueue()

"""
Persistent, server-side transcription queue.

Core idea: worker threads run permanently (not just during a "batch run")
and process the queue from the database. This means:

  * The browser is a pure observer. If the web UI closes, the queue keeps
    working — previously it lived in localStorage and was pushed by JS
    (see old transcribe.js/processNextQueued).
  * A server restart continues the queue: 'processing' is reset to 'pending'
    on boot (requeue_stale), pending entries continue running.

Lanes: ``queue.lanes`` worker threads each take one entry at a
time, so that many files are in flight at once. Every model is still loaded
once and serves one file at a time — core/gpu_gate.py decides who may use
which model when. What overlaps is everything around the models: copying
from the NAS, ffmpeg, padding, writing, booking. A mixed batch runs
noticeably faster with three lanes than with one, short files most of
all. One lane is the serial queue it always was.

The rules that keep lanes from tripping over each other:

  * an entry is claimed in ONE statement (UPDATE … RETURNING) and
    registered in ``_running`` under the same lock — never twice;
  * no two entries with the same name are in flight: the job id, the work
    folder and the job log hang on the name — nor two entries of the same
    file under another spelling (``_claim_next``);
  * at most one long file (``queue.long_file_sec``) is in flight — each
    holds its whole audio in memory, twice with the diarization worker;
  * stop, abort and shutdown reach every running entry, not just one.

Status values of an entry:
    pending → processing → done | error | skipped
    cancelled (deliberately aborted by user)

Abort is NOT the same as error: whoever pauses the queue (stop) or shuts down
the server (shutdown) wants the running entry later — it goes back to 'pending'.
A deliberate abort (explicitly via /api/cancel/<job_id> or as a running job
during total abort /api/queue/abort) ends as 'cancelled', not 'error' — visible
in history and re-queued via retry. See _requeue_on_abort.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from whisper_tm.config import VALID_WHISPER_MODELS, config
from whisper_tm.core import paths
from whisper_tm.core.gpu_gate import GATE
from whisper_tm.core.helpers import (
    is_skipped,
    is_transcript_complete,
    parse_bool,
    transcript_path_for,
)
from whisper_tm.core.security import root_allowed

logger = logging.getLogger(__name__)

# Roots of earlier configurations whose continued processing has already
# been noted in the log — otherwise the message would show up once per
# legacy entry, possibly hundreds of thousands of times.
_foreign_roots_reported: set[str] = set()

MAX_LOG = 200

# ── Worker state (runtime only; the queue itself lives in the DB) ─────────

_lock = threading.Lock()
_cv = threading.Condition(_lock)  # wakes the workers on new entries
_ctl_lock = threading.Lock()  # serializes start_worker/stop_worker
_worker: threading.Thread | None = None  # the first lane (older readers)
_workers: list[threading.Thread] = []
# Claim and registration happen under this lock together: a lane must never
# see an entry as 'processing' in the DB that _running doesn't know yet,
# or it could claim a second entry with the same name.
_claim_lock = threading.Lock()

#: Upper bound for queue.lanes: beyond 3 lanes a test GPU measured only
#: +3 %, and every lane holds a file's audio in memory.
MAX_LANES = 4

# ── Change counter + cache ───────────────────────────────────────────────
# The SSE stream polls the queue state every second — per open tab,
# endlessly. Without a cache that meant two DB transactions per second and
# tab, even when nothing is happening (measured: thousands of
# transactions within half an hour). Every mutation bumps _generation;
# reads only happen when the generation has changed since the last time.
#
# PRECONDITION: this module is the ONLY writer of the queue_items table.
# That holds (verified: no other module touches it) — anyone writing to it
# from outside in the future MUST call _bump(), otherwise the cache serves
# stale data.
# Bulk operations of the queue editor go exclusively through editor_where();
# whoever builds a second WHERE breaks the "hit count == effect" guarantee.
_generation = 0
_cache: dict = {"gen": -1, "data": None}
# Waiting lists per generation, split by requested length:
# {generation: {limit: [items]}}, limit 0 = full list.
# Previously the FULL list lived in _cache["data"] and was rebuilt on every
# cache miss — with a six-figure number of pending entries that meant one
# dict per entry, even though the stream only shows 200 of them. Now only what someone actually
# requested is built and kept.
_pending_cache: dict = {"gen": -1, "lists": {}}
_stopping = False
_paused = False
# Entries that run although the queue is paused: what was just brought in to
# be worked on now (upload, link download, the transcribe dialog, a video
# from the browser extension — "front"). When the queue is paused, only the
# new entry runs, the rest stays paused. Adding it used to resume the whole
# queue — a huge backlog on a card someone had paused it for. Gone from here
# once it ran (_release); stop, pause and resume empty it.
_solo: set[int] = set()
# Process is shutting down → the running entry gets re-queued instead of
# being marked as an error (see shutdown()).
_shutting_down = False

# Running entries by id, in claim order. ``_current`` stays the oldest of
# them for the readers that know only one (None = nothing runs).
_running: dict[int, dict] = {}
_current: dict | None = None
# Shared wall clock: advances 1/n per second while n entries run. An
# entry's share of it is its cost — with lanes its wall time is latency,
# up to three files sharing the card (see _cost_ratio).
_share_clock = {"v": 0.0, "t": time.monotonic()}
# Abort request for the entire run (not just the current job)
_abort = False
# abort() is emptying the queue right now: an entry a lane claims meanwhile
# is cancelled with the rest (_claim_next)
_aborting = False
# Item IDs whose abort originates from queue control (stop/shutdown) and
# which therefore belong back in the queue instead of ending as an error.
# A flag alone would not be enough here: between "stop" and the moment the
# worker notices the abort, seconds can pass (the GPU finishes computing the
# current chunk). If the user presses "resume" in that gap, a global flag
# would already have been reset and the entry would still fall into the
# error branch. The ID keeps the request tied to the entry.
_requeue_on_abort: set[int] = set()

#: How many times an entry has already failed on a CUDA OOM is stored
#: IN THE ENTRY (settings_json.oom_attempts) — it never
#: becomes an error, and the counter must survive a restart so the next run
#: proceeds with conserved VRAM and a small batch. See `_oom_retry`.
_log: list[str] = []
# Lines ever logged: the stream's cursor. The length of the ring stops at
# MAX_LOG — a cursor on it saw no new line once the ring was full.
_log_total = 0


def _db():
    """Fetch the DB handle fresh from extensions.

    Do not use the name bound at import time: create_app() only sets
    ext.db at runtime — a `from ... import db` would then be None.
    """
    import whisper_tm.extensions as ext

    return ext.db


def _jm():
    import whisper_tm.extensions as ext

    return ext.job_manager


def _bump() -> None:
    """Queue has changed → discard the cache."""
    global _generation
    with _lock:
        _generation += 1


def qlog(msg: str) -> None:
    """Line into the queue log (ring buffer for the SSE stream) AND into the log file.

    The ring buffer alone was not enough: it lives only in RAM and is
    discarded on shutdown. Of all things, the interesting events —
    aborted, skipped, back in the queue — showed up in no log at all. In
    the debug log, a user stop looked like a silent, wordless gap.
    """
    global _log_total
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        _log.append(entry)
        _log_total += 1
        if len(_log) > MAX_LOG:
            del _log[:-MAX_LOG]
    logger.info("Queue: %s", msg)


# ── Queue CRUD (DB) ───────────────────────────────────────────────────────


def _row_to_item(row) -> dict:
    settings = {}
    try:
        settings = json.loads(row["settings_json"] or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        settings = {}
    return {
        "id": row["id"],
        "filename": row["filename"],
        "dir": row["base_dir"] or "",
        # The identity (docs/PATHS.md), as the running entries carry it: the
        # name is relative to the entry's root, the browser's list to the
        # one open now — after the upload dialog opened the uploads folder,
        # a waiting upload showed "+Q" as if it weren't queued
        "abs": _item_abs_path({"base_dir": row["base_dir"], "filename": row["filename"]}),
        "settings": settings,
        "status": row["status"],
        "position": row["position"],
        "job_id": row["job_id"] or "",
        "error": row["error"] or "",
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "interrupted": _column(row, "interrupted", 0) or 0,
    }


def _column(row, name: str, default=None):
    """A column a row may lack (a SELECT that names its columns)."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _trim_overlap(filename: str, base_dir: str) -> str:
    """Strip a duplicated path prefix: the end of the root = the start of the name.

    A real-world case (hundreds of entries at once)::

        base_dir : /mnt/media/videos/Archive/Videos
        filename :             videos/Archive/Videos/meeting_2024-03-01.mp4
        combined : /mnt/media/videos/Archive/Videos/videos/Archive/Videos/…

    It happens because the frontend sends the currently DISPLAYED directory
    as ``base_dir``, but the filenames are relative to the media root
    (``batch.js``). Both values are correct on their own — just not
    relative to each other.

    We look for the LONGEST prefix of the filename that is at the same time
    the tail end of the root; exactly that is stripped off. The return
    value is the shortened name, or ``""`` if there is no overlap.

    The caller then checks whether the file is actually there — nothing is
    guessed, and the traversal protection of :func:`paths.abs_from` still
    sits in front of it.

    >>> _trim_overlap("videos/Archive/f.mp4", "/nas/videos/Archive")
    'f.mp4'
    >>> _trim_overlap("f.mp4", "/nas/videos")
    ''
    """
    parts = [t for t in str(filename or "").replace("\\", "/").split("/") if t]
    basis = [t for t in str(base_dir or "").replace("\\", "/").split("/") if t]
    if not parts or not basis:
        return ""
    # Stripping the whole prefix would be wrong: no name would be left.
    for k in range(min(len(parts) - 1, len(basis)), 0, -1):
        if basis[-k:] == parts[:k]:
            return "/".join(parts[k:])
    return ""


def enqueue(
    filenames: list[str],
    settings: dict,
    base_dir: str | None = None,
    dir_map: dict | None = None,
    front: bool = False,
) -> list[dict]:
    """Add files to the queue — at the end, or with ``front=True`` at the FRONT.

    ``front`` is for freshly brought-in files (upload/link download) and for
    a folder queued "next up" from the queue editor: whoever brings in a
    file or picks a folder right now wants to see it transcribed NOW —
    behind a huge overall run it would otherwise only be its turn in
    weeks.

    Duplicate entries (same file already pending/processing) are skipped —
    with ONE exception: if the file is already ``pending`` in the queue and
    comes in again with ``front=True``, it is PULLED TO THE FRONT instead of
    being silently discarded. Otherwise the promise "what I upload is next"
    would depend on something the user can't see: whether the same file
    happens to already be sitting somewhere in a huge run. Exactly then,
    ``front`` previously did nothing at all.

    An already RUNNING file (``processing``) is left untouched — it is
    already being worked on, and aborting a running model just to get to
    the front, where it already is, would waste compute.

    Returns the newly created AND the pulled-forward entries.
    """
    database = _db()
    if not database:
        return []

    dir_map = dir_map or {}
    added: list[dict] = []

    # Check root and filename BEFORE anything goes into the queue. This
    # function is the shared funnel for queueing (/api/queue/add);
    # `base_dir` and `dir_map` used to come from the request here
    # unchecked.
    #
    # The worker later builds `Path(item["dir"]) / filename`. Two holes
    # were hiding in there: an arbitrary root (`base_dir: "/etc"`) and an
    # absolute filename — `Path("/nas") / "/etc/shadow"` simply yields
    # `/etc/shadow` in pathlib. With `export_alongside` the pipeline also
    # writes next to the source, so this was also a write primitive. Fetch
    # the allowlist ONCE, not per file: root_allowed() internally calls
    # allowed_roots() and resolve() — for 20,000 files that was
    # measurably over 4 seconds in the request handler, on a NAS one
    # network round-trip per path component.
    _root_cache: dict[str, str] = {}

    def _check_root(value: str) -> str:
        """Normalize a root, or return '' if it is not allowed."""
        text = str(value or "").strip()
        if not text:
            return ""
        if text in _root_cache:
            return _root_cache[text]
        if not root_allowed(text):
            logger.warning("enqueue: root rejected (not in security.allowed_roots): %s", text)
            _root_cache[text] = "__REJECTED__"
            return "__REJECTED__"
        # canon() instead of resolve(): docs/PATHS.md rules out resolving
        # via the filesystem (symlinks stay as-is, no round-trips).
        _root_cache[text] = paths.canon(Path(text).expanduser())
        return _root_cache[text]

    checked: list[tuple[str, str]] = []
    for fn in filenames:
        if not fn:
            continue
        root = _check_root(dir_map.get(fn) or base_dir or "")
        if root == "__REJECTED__":
            continue
        basis = Path(root) if root else config.media_dir
        # The combined path MUST stay below the base. That defeats the
        # absolute filename and every "../".
        # abs_from() returns "" if the combined path leaves the base
        # (docs/PATHS.md) — that defeats the absolute filename and every
        # "../".
        #
        # An additional root_allowed(target) check used to sit here and
        # was REDUNDANT: the base is already checked above, and abs_from
        # guarantees the result lies below it. But it cost a Path.resolve()
        # per file — i.e. a NAS access. For a six-figure batch of files
        # that took minutes, during which the /api/queue/add call hung and
        # the UI showed nothing. Without it: well under a second.
        # An absolute filename is not a valid input: per the model, the
        # field is relative to the root. abs_from() would silently
        # reinterpret it — "/etc/shadow" becomes "<root>/etc/shadow"
        # because the leading slash is stripped. That stays within the
        # root and is thus harmless, but the entry would then sit in the
        # queue under a name that means something other than it looks
        # like. Better to reject it.
        if paths.is_absolute(str(fn)):
            logger.warning("enqueue: absolute filename rejected: %r", fn)
            continue
        target = paths.abs_from(str(fn), str(basis))
        if not target:
            logger.warning("enqueue: filename leaves the root, skipped: %r (base %s)", fn, basis)
            continue
        checked.append((fn, root))

    if not checked:
        return []

    # '' means "the media root" — the same file as its root spelled out.
    # As two keys, one file queued both ways was two entries.
    try:
        media_root = paths.canon(Path(config.media_dir))
    except Exception:  # config not loaded (tools/tests)
        media_root = ""

    def _key(fn: str, root: str) -> tuple[str, str]:
        return fn, (root or media_root)

    with database.connection() as conn:
        # Write lock BEFORE reading: Database.connection() runs with
        # isolation_level '', so the reads below were autocommit. Two
        # enqueues at once (autoscan, an upload, the extension) both read
        # an empty queue — the second one's "new run" DELETE wiped the
        # first one's fresh entries — or both missed the other's row and
        # queued the same file twice.
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        # If the queue is empty (no pending/processing), a new run begins:
        # completed entries of the previous run are cleared out so counters
        # and the progress bar start at 0. History lives in job_history
        # regardless.
        active = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM queue_items "
            "WHERE status IN ('pending', 'processing')) AS n"
        ).fetchone()["n"]
        if not active:
            conn.execute("DELETE FROM queue_items")

        # Dedup over (filename, root), not over the filename alone:
        # `series/f1.mp4` under /nasA and under /nasB are two different
        # files (docs/PATHS.md). Previously the second one was silently
        # discarded and never transcribed.
        # The id is needed for pulling forward, not just the key —
        # hence a dict instead of a set. `processing` is deliberately
        # included in the duplicate check, but does not get an id for
        # moving.
        #
        # Only the requested names are looked up (idx_queue_filename): the
        # write lock is held from the first read on, and reading every
        # waiting entry (tens of thousands) held it that long for each
        # autoscan enqueue.
        present: dict[tuple[str, str], int | None] = {}
        asked = sorted({fn for fn, _root in checked})
        for i in range(0, len(asked), 5000):
            for r in conn.execute(
                "SELECT id, filename, base_dir, status FROM queue_items "
                "WHERE filename IN (SELECT value FROM json_each(?)) "
                "AND status IN ('pending', 'processing')",
                (json.dumps(asked[i : i + 5000], ensure_ascii=False),),
            ):
                present[_key(r["filename"], r["base_dir"] or "")] = (
                    r["id"] if r["status"] == "pending" else None
                )
        existing = set(present)
        if front:
            # Before the first pending entry: positions are allowed to go
            # negative (the worker only sorts by ORDER BY position, id).
            row = conn.execute(
                "SELECT COALESCE(MIN(position), 1) AS p FROM queue_items WHERE status = 'pending'"
            ).fetchone()
            pos = row["p"] - len(filenames) - 1
        else:
            row = conn.execute("SELECT COALESCE(MAX(position), 0) AS p FROM queue_items").fetchone()
            pos = row["p"]

        payload = json.dumps(settings, ensure_ascii=False)

        # Collect first, then a single executemany. One INSERT per file
        # noticeably cost time for a six-figure batch of files, and the whole call hangs
        # off the /api/queue/add HTTP request — the UI shows nothing the
        # whole time.
        new_value: list[tuple[str, str]] = []
        # Already-pending files that are requested again with front=True:
        # (id, filename, root).
        move_to_front: list[tuple[int, str, str]] = []
        for fn, root in checked:
            key = _key(fn, root)
            if key in existing:
                eid = present.get(key)
                if front and eid is not None:
                    move_to_front.append((eid, fn, root))
                    present[key] = None  # pull forward only once
                continue
            existing.add(key)
            present[key] = None
            new_value.append((fn, root))
        if not new_value and not move_to_front:
            return []

        # The pulled-forward ones take the frontmost slots, the new ones
        # behind them. Their order among themselves stays that of the
        # request.
        for i, (eid, _fn, _root) in enumerate(move_to_front):
            conn.execute("UPDATE queue_items SET position = ? WHERE id = ?", (pos + i + 1, eid))
        pos += len(move_to_front)

        added = [
            {
                "id": eid,
                "filename": fn,
                "dir": root,
                "settings": settings,
                "status": "pending",
                "position": pos - len(move_to_front) + i + 1,
                "moved_up": True,
            }
            for i, (eid, fn, root) in enumerate(move_to_front)
        ]

        if new_value:
            value_list = [
                (fn, root, payload, pos + i + 1) for i, (fn, root) in enumerate(new_value)
            ]
            conn.executemany(
                "INSERT INTO queue_items "
                "(filename, base_dir, settings_json, status, position) "
                "VALUES (?, ?, ?, 'pending', ?)",
                value_list,
            )

            # The assigned IDs are contiguous: AUTOINCREMENT assigns them
            # ascending, and this module is, per the module header, the
            # ONLY writer of the table — within this one transaction
            # nothing else can interfere.
            #
            # Only read this inside the if branch: without a preceding
            # INSERT, last_insert_rowid() returns the id from an EARLIER
            # call on the same connection — `added` would then point at
            # someone else's row, and /api/transcribe would track the
            # wrong job.
            last_one = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            first_one = last_one - len(new_value) + 1
            added += [
                {
                    "id": first_one + i,
                    "filename": fn,
                    "dir": root,
                    "settings": settings,
                    "status": "pending",
                    "position": pos + i + 1,
                }
                for i, (fn, root) in enumerate(new_value)
            ]

    if added:
        _bump()
        # An abort request from an earlier run must not make the new one
        # show up as "aborted" — otherwise the worker only resets _abort
        # on the next idle cycle.
        global _abort
        with _cv:
            _abort = False
            _cv.notify_all()
        qlog(f"{len(added)} added to the queue")
    return added


def _db_state() -> dict:
    """Counters + times — read from the DB once per change.

    This is the hot path: the SSE stream calls it every second, per open
    tab, endlessly. Without the generation counter, every one of these
    ticks would land in the stats as a DB transaction.

    The waiting list is deliberately NOT included here anymore. It used to
    be fully materialized — with a six-figure number of pending entries
    that meant one dict per entry — even though snapshot() never touches it and the stream
    only shows 200 of them. Every queue change discarded the cache and
    rebuilt everything; a trace over a few minutes counted millions of
    calls to _row_to_item. The list now comes directly from the DB on
    demand, with LIMIT and via the idx_queue_status(status, position, id)
    index.
    """
    with _lock:
        if _cache["gen"] == _generation and _cache["data"] is not None:
            return _cache["data"]
        gen = _generation

    counts = {"pending": 0, "processing": 0, "done": 0, "error": 0, "skipped": 0, "cancelled": 0}
    started_at = finished_at = None

    database = _db()
    if database:
        with database.connection() as conn:
            for r in conn.execute("SELECT status, COUNT(*) AS n FROM queue_items GROUP BY status"):
                counts[r["status"]] = r["n"]
            row = conn.execute(
                "SELECT MIN(NULLIF(started_at, 0)) AS s, "
                "       MAX(finished_at) AS f FROM queue_items"
            ).fetchone()
            started_at = row["s"]
            finished_at = row["f"] or None

    data = {"counts": counts, "started_at": started_at, "finished_at": finished_at}

    with _lock:
        # Only cache if nothing has changed in the meantime — otherwise a
        # stale state would overwrite the new generation.
        if _generation == gen:
            _cache["gen"] = gen
            _cache["data"] = data
    return data


def list_items(status: str | None = None, limit: int = 0) -> list[dict]:
    """Entries of the queue. `limit` applies in the QUERY, not afterwards.

    The difference matters a lot for this queue: 'pending' used to run
    through a fully in-memory-built list, from which callers then took the
    first 200 — with a six-figure number of pending entries that meant one
    dict per entry created per call, for 200 shown.
    """
    database = _db()
    if not database:
        return []

    # Status lists are the hot path (SSE stream, every second per open
    # tab: pending + error + cancelled) and are therefore cached per
    # generation. An already existing, LONGER list of the same status also
    # serves a shorter request — then the stream tick costs nothing at
    # all anymore. Cache key: (status, limit).
    if status:
        with _lock:
            gen = _generation
            if _pending_cache["gen"] == gen:
                lists = _pending_cache["lists"]
                if (status, limit) in lists:
                    return list(lists[(status, limit)])
                if (status, 0) in lists:
                    full = lists[(status, 0)]
                    return list(full[:limit]) if limit else list(full)
                for (st, present), rows in lists.items():
                    if st == status and limit and present and present >= limit:
                        return list(rows[:limit])
            else:
                _pending_cache["gen"] = gen
                _pending_cache["lists"] = {}

    sql = "SELECT * FROM queue_items"
    params: tuple = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY position, id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with database.connection() as conn:
        items = [_row_to_item(r) for r in conn.execute(sql, params)]

    if status:
        with _lock:
            # Only store it if nothing has changed since our OWN read —
            # compared is `gen` (the state this list came from) against the
            # current generation.
            #
            # Previously this said `_pending_cache["gen"] == _generation`,
            # and that is not the same thing: if another reader has in the
            # meantime raised the marker to the new generation and stored
            # its fresh list, the condition was true — and the stale list
            # would overwrite the fresh one under the new generation. After
            # that the cache served the old state until something changed
            # the queue again. For pulling forward this would be fatal:
            # "right at the front" would not show up in the list.
            if gen == _generation:
                _pending_cache["lists"][(status, limit)] = items
    return items


def remove_item(item_id: int) -> tuple[bool, str]:
    """Remove an entry from the queue that has not started yet."""
    database = _db()
    if not database:
        return False, "No database"
    with database.connection() as conn:
        row = conn.execute("SELECT status FROM queue_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return False, "Entry not found"
        if row["status"] == "processing":
            return False, "Already running — please cancel instead of removing"
        if row["status"] != "pending":
            return False, "Entry is already finished"
        # Guarded: a lane may claim the row between the check and here.
        gone = conn.execute(
            "DELETE FROM queue_items WHERE id = ? AND status = 'pending'", (item_id,)
        ).rowcount
    _bump()
    if not gone:
        return False, "Already running — please cancel instead of removing"
    return True, "Removed"


def move_item(item_id: int, direction: int) -> tuple[bool, str]:
    """Move an entry in the remaining list up (-1) or down (+1) one position."""
    if direction not in (-1, 1):
        return False, "direction must be -1 or 1"
    database = _db()
    if not database:
        return False, "No database"
    with database.connection() as conn:
        pending = conn.execute(
            "SELECT id, position FROM queue_items WHERE status = 'pending' ORDER BY position, id"
        ).fetchall()
        ids = [r["id"] for r in pending]
        if item_id not in ids:
            return False, "Entry is not in the queue"
        i = ids.index(item_id)
        j = i + direction
        if j < 0 or j >= len(ids):
            return False, "Already at the edge"
        # Swap the positions of the two entries. With equal positions
        # (legacy data), a plain swap would have no effect → then
        # renumber the entire remaining list.
        pa, pb = pending[i]["position"], pending[j]["position"]
        if pa == pb:
            order = ids[:]
            order[i], order[j] = order[j], order[i]
            for n, iid in enumerate(order, start=1):
                conn.execute("UPDATE queue_items SET position = ? WHERE id = ?", (n, iid))
        else:
            conn.execute("UPDATE queue_items SET position = ? WHERE id = ?", (pb, item_id))
            conn.execute("UPDATE queue_items SET position = ? WHERE id = ?", (pa, ids[j]))
    _bump()
    return True, "Moved"


def move_to_top(item_id: int) -> tuple[bool, str]:
    """Put a pending entry at the VERY FRONT of the queue.

    move_item() moves by exactly ONE position. With a six-figure backlog,
    that is not a way forward — whoever wants a file now would have to
    click that many times. Hence this jump.

    The RUNNING entry is left untouched (same reasoning as in enqueue():
    pulling it forward would mean aborting a model just to reach a spot
    it's already at).

    Paused stays paused: this is pure reordering, not a job submission.
    Whoever brings in a file via "next up" goes through enqueue(
    front=True) — there the route deliberately resumes a paused queue.
    """
    database = _db()
    if not database:
        return False, "No database"
    with database.connection() as conn:
        row = conn.execute("SELECT status FROM queue_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return False, "Entry not found"
        if row["status"] == "processing":
            return False, "Already running — already at the very front"
        if row["status"] != "pending":
            return False, "Only pending entries can be pulled forward"
        first = conn.execute(
            "SELECT id, position FROM queue_items WHERE status = 'pending' "
            "ORDER BY position, id LIMIT 1"
        ).fetchone()
        if first["id"] == item_id:
            return True, "Already at the very front"
        # One position BEFORE — positions are allowed to go negative, the
        # worker only sorts by ORDER BY position, id (see
        # enqueue(front=True)). Renumbering the remaining list is out of
        # the question: that could be a six-figure number of UPDATEs for one
        # click.
        conn.execute(
            "UPDATE queue_items SET position = ? WHERE id = ?", (first["position"] - 1, item_id)
        )
    _bump()
    return True, "Moved to the very front"


def remove_pending_for(abs_paths: list[str]) -> int:
    """Remove pending entries whose file has disappeared.

    Called by the walk cleanup (scanner, Database.clean_orphaned_media): if the
    media_files row is gone, the associated pending entry must not be left
    lying around — it would only fail as "not found". The matching goes via
    base_dir + '/' + filename = abs_path (exactly how the worker builds the
    path). Lives here because this module is the sole writer of the table
    (cache generation!).
    """
    database = _db()
    if not database or not abs_paths:
        return 0
    n = 0
    with database.connection() as conn:
        for i in range(0, len(abs_paths), 500):
            part = abs_paths[i : i + 500]
            marks = ",".join("?" for _ in part)
            cur = conn.execute(
                f"DELETE FROM queue_items WHERE status = 'pending' "
                f"AND rtrim(base_dir, '/') || '/' || filename IN ({marks})",
                part,
            )
            n += cur.rowcount or 0
    if n:
        _bump()
        qlog(f"🧹 {n} pending entries of vanished files removed")
    return n


def clear_pending() -> int:
    """Discard all pending entries (the running one stays untouched)."""
    database = _db()
    if not database:
        return 0
    with database.connection() as conn:
        cur = conn.execute("DELETE FROM queue_items WHERE status = 'pending'")
        n = cur.rowcount or 0
    _bump()
    if n:
        qlog(f"{n} pending {'entry' if n == 1 else 'entries'} discarded")
    return n


def _item_abs_path(item: dict) -> str:
    """File identity of a queue entry (root + relative path)."""
    # Two forms are in circulation: raw DB row (base_dir) and _row_to_item
    # (dir). Previously only base_dir was checked — for the running entry
    # (dir), the identity silently fell back to the media root.
    try:
        basis = item.get("base_dir") or item.get("dir") or config.media_dir
        return paths.canon(Path(basis) / str(item.get("filename") or ""))
    except Exception:
        return ""


def _recommended_batch() -> int:
    try:
        import whisper_tm.extensions as ext

        return int(getattr(ext.hw, "recommended_batch_size", 16) or 16)
    except Exception:
        return 16


def _is_oom_error(error: str) -> bool:
    """A CUDA OOM, in whichever words it arrives.

    pyannote re-raises one as its own MemoryError — "batch_size (32) is
    probably too large. Try with a smaller value until memory error
    disappears." — without the words "out of memory". Unrecognised, a
    diarization that met a full card was booked as an error instead of
    waiting (seen while another process filled the card).
    """
    text = (error or "").lower()
    return "out of memory" in text or ("is probably too large" in text and "memory error" in text)


#: Errors of the machine, not of the file (the text an error carries, in
#: lower case → what is wrong). Every pending entry would fail the same way,
#: one after another, and land in the error bucket: the queue pauses instead
#: and the entry stays queued.
ENVIRONMENT_ERRORS = (
    ("no space left on device", "The disk is full"),
    ("disk quota exceeded", "The disk quota is used up"),
    ("read-only file system", "A folder the queue writes to is read-only"),
    ("transport endpoint is not connected", "A network share is gone (not mounted?)"),
    ("stale file handle", "A network share is gone (not mounted?)"),
    ("host is down", "A network share is gone (not mounted?)"),
    ("temporary failure in name resolution", "No connection to the internet"),
    ("name or service not known", "No connection to the internet"),
    ("network is unreachable", "No connection to the internet"),
    ("failed to establish a new connection", "No connection to the internet"),
    ("max retries exceeded with url", "No connection to the internet"),
    ("couldn't connect to 'https://huggingface.co'", "No connection to the internet"),
    ("outgoing traffic has been disabled", "Offline — a model it needs isn't downloaded"),
    # core/prefetch.download_error: a model's first download failed
    ("could not be downloaded from", "A model could not be downloaded (no internet?)"),
)


def environment_cause(error: str) -> str:
    """What about the machine an error says — "" when it is the file's own."""
    text = (error or "").lower()
    return next((cause for needle, cause in ENVIRONMENT_ERRORS if needle in text), "")


#: Below this much free space in the work folder no job starts (the decoded
#: sound alone is ~115 MB an hour, a vocal separation several GB):
#: ``queue.work_min_free_gb``
WORK_MIN_FREE_GB = 2.0


def _work_space_short() -> str:
    """Why the work folder has too little room for a job — "" when it has."""
    try:
        need = float(config.get("queue", "work_min_free_gb", default=WORK_MIN_FREE_GB) or 0)
    except (TypeError, ValueError):
        need = WORK_MIN_FREE_GB
    if need <= 0:
        return ""
    try:
        work = Path(config.work_dir)
        free = shutil.disk_usage(work if work.exists() else work.parent).free
    except Exception:  # noqa: BLE001 — unknown: the job finds out itself
        return ""
    if free < need * 1e9:
        return (
            f"only {free / 1e9:.1f} GB free in the work folder {work} "
            f"(queue.work_min_free_gb: {need:g})"
        )
    return ""


def _library_gone(folder) -> bool:
    """The library's folder missing, or empty where a share is mounted: every
    file of it would fail as "not found"."""
    try:
        folder = Path(folder)
        return not folder.is_dir() or next(folder.iterdir(), None) is None
    except OSError:
        return True


def _environment_pause(item: dict, error: str, job=None, cause: str = "") -> bool:
    """An error of the machine (``environment_cause``, or ``cause``): the
    queue pauses with the cause, the entry stays queued — instead of every
    pending file failing the same way, one after another (a NAS unmounted,
    a full disk, no internet for a model download). Fixed, "Resume" goes
    on where it stopped."""
    cause = cause or environment_cause(error)
    if not cause:
        return False
    filename = str(item.get("filename") or "")
    text = (
        f"{cause} — queue paused, {filename} stays queued. Fix it, then click "
        f'"Resume" in the queue. ({str(error or "")[:240]})'
    )
    # Paused first, then pending: the other way round another lane could
    # take the entry in between.
    pause()
    _back_to_pending(item["id"])
    try:
        _jm().system_warning(text)
    except Exception:
        pass
    try:
        from whisper_tm.core import incidents as _inc

        _inc.report_it(
            "system",
            text,
            abs_path=_item_abs_path(item),
            filename=filename,
            job_id=getattr(job, "job_id", "") or "",
            log_file=getattr(job, "log_file", "") or "",
        )
    except Exception:
        pass
    qlog(f"⏸ {text}")
    return True


def summary_pending(abs_path: str) -> bool:
    """Whether a summary of this file is still to come — waiting or running
    in the LLM queue (the transcription dialog's, see _queue_summary, or the
    summary dialog's). The file view asks, and shows it coming instead of
    "0 summaries"."""
    target = paths.canon(abs_path) if abs_path else ""
    if not target:
        return False
    from whisper_tm.core.llm_queue import llm_queue

    return llm_queue.summary_waiting(target)


def _oom_retry(item: dict, error: str, job=None) -> bool:
    """CUDA OOM: the entry NEVER becomes an error — it waits, or the queue pauses.

    Previously the rule was: retry once with "conserve VRAM", on the
    second OOM book it as an error. In practice the retry failed every
    time, because it started in the SAME second as the error, with a card
    that was just as full (another program held part of the VRAM). And
    because ``hardware.low_vram`` was already globally on, "conserve VRAM"
    changed nothing about the retry. Dozens of intact files were then
    marked as broken. They were not — intact files must never end up
    marked broken and lost.

    Hence three rules now:

    1. **Never an error.** An OOM is a state of the card, not a property of
       the file. The entry goes back to ``pending`` (at its position, it's
       next in line) — with ``low_vram``, a halved WhisperX batch, and the
       ``oom_attempts`` counter in the settings.
    2. **Wait first, then retry.** Before the retry run, the worker waits
       until ``hardware.oom_min_free_mb`` is free — at most
       ``hardware.oom_wait_min`` minutes. If the card doesn't free up,
       nothing is even attempted.
    3. **Then pause instead of burning on.** If the card doesn't free up,
       or if even the waited, conserved run fails too: pause the queue,
       raise a system warning, log an incident. The entry stays pending;
       "resume" tries again (with an even smaller batch). An OOM chain is a
       stop signal.

    The counter is deliberately stored in the database: after a restart the
    run should not start again at batch 16, and in the incidents table and
    the queue badge you should be able to see how many times it has already
    happened.

    With lanes the entry stays 'processing' while its lane waits — as
    'pending' another lane would take it straight away, onto the same full
    card. And the wait holds the GPU exclusively: the other lanes finish
    their current step and stop, so what is measured is what a retry would
    find. The retry itself runs low-VRAM, which is exclusive anyway.
    """
    if not _is_oom_error(error):
        return False
    database = _db()
    if not database:
        return False

    formerly = dict(item.get("settings") or {})
    attempts = int(formerly.get("oom_attempts") or 0) + 1
    formerly["oom_attempts"] = attempts
    formerly["low_vram"] = True
    basis = formerly.get("batch_size")
    try:
        basis = int(basis) if basis else _recommended_batch()
    except (TypeError, ValueError):
        basis = _recommended_batch()
    formerly["batch_size"] = max(2, basis // 2)
    try:
        with database.connection() as conn:
            conn.execute(
                "UPDATE queue_items SET settings_json = ? WHERE id = ? AND status = 'processing'",
                (json.dumps(formerly), item["id"]),
            )
    except Exception as e:  # noqa — a failed retry is no reason to
        logger.warning("OOM retry failed: %s", e)  # abort
        return False  # the run
    _bump()
    abs_path = _item_abs_path(item)
    filename = str(item.get("filename") or "")

    def _incident(kind, text):
        # The job's step and log go along: its own "error" incident is not
        # written for an error the queue takes up again (incidents.job_event).
        try:
            from whisper_tm.core import incidents as _inc

            step = _inc._step(job) if job is not None else ""
            _inc.report_it(
                kind,
                text,
                abs_path=abs_path,
                filename=filename,
                job_id=getattr(job, "job_id", "") or "",
                step=step,
                log_file=getattr(job, "log_file", "") or "",
            )
        except Exception:
            pass

    _incident(
        "oom",
        f"VRAM exhausted (attempt {attempts}) — back into the "
        f"queue with VRAM conservation, batch {formerly['batch_size']}: "
        f"{str(error or '')[:400]}",
    )
    qlog(
        f"🔁 VRAM exhausted — {filename} stays pending (attempt {attempts}, "
        f"conserving VRAM, batch {formerly['batch_size']})"
    )

    # Rule 2: wait first. Rule 3: after the second OOM, stop trying.
    try:
        with GATE.exclusive(
            "waiting after a VRAM overflow", cancel=lambda: _oom_wait_over(item, job)
        ):
            # Unloaded in the same section that measures: between the
            # pipeline's own cleanup and here the other lanes may have
            # loaded their models again, and the wait would count them as
            # foreign — ten minutes for nothing, then a pause with a false
            # "another GPU program" warning. The retry
            # runs low-VRAM and would unload them anyway.
            _unload_models()
            outcome = _oom_wait(item, attempts, job)
    except InterruptedError:
        # Stopped, paused or cancelled before the other lanes even let go
        outcome = _oom_cancel_outcome(item, job)
    if outcome == "cancelled":
        _finish(item["id"], "cancelled", "Cancelled")
        qlog(f"⚠ Aborted: {filename}")
        return True
    if outcome == "retry":
        _back_to_pending(item["id"])
        return True
    text = (
        f"VRAM permanently occupied — queue paused. {filename} failed "
        f"{outcome}; the entry stays queued. Is another GPU "
        f'program running? Then click "Resume" in the queue.'
    )
    # Paused first, then pending: the other way round another lane could
    # take the entry in between.
    pause()
    _back_to_pending(item["id"])
    try:
        _jm().system_warning(text)
    except Exception:
        pass
    _incident("system", text)
    qlog(f"⏸ {text}")
    return True


def _oom_wait(item: dict, attempts: int, job) -> str:
    """Rules 2 and 3 of _oom_retry: "retry", "cancelled", or why to pause."""
    from whisper_tm.core import vram as _vram

    min_free = int(config.get("hardware", "oom_min_free_mb", default=8000) or 0)
    wait_time = float(config.get("hardware", "oom_wait_min", default=10) or 0)
    # On a card with less VRAM than that the wait could never be met: it
    # ran its full time and paused the queue — on an 8 GB card after every
    # out-of-memory. At most 80 % of the card is waited for.
    total = _vram.total_mb()
    if total and min_free > 0.8 * total:
        min_free = int(0.8 * total)
    # A cancel, stop or abort that came while the lane waited for the other
    # lanes to let go of the GPU decides first — before lanes nothing could
    # come in between, and a second OOM paused the queue over an entry the
    # user had just cancelled.
    if _oom_wait_over(item, job):
        return _oom_cancel_outcome(item, job)
    if attempts >= 2:
        return "even with VRAM conservation and half the batch"
    if not (min_free > 0 and wait_time > 0):
        return "retry"  # waiting disabled: retry immediately
    reached, free = _vram.wait_for_free_vram(
        min_free,
        wait_time * 60,
        abort=lambda: _oom_wait_over(item, job),
        report_it=lambda f: qlog(
            f"⏳ Waiting for free VRAM: {f} MB free, need {min_free} MB (up to {wait_time:g} min)"
        ),
    )
    if _oom_wait_over(item, job):
        # The file's cancel, a stop, an abort or the shutdown ended it
        return _oom_cancel_outcome(item, job)
    if reached:
        if free is not None:
            qlog(f"✓ VRAM free ({free} MB) — retry starting")
        return "retry"
    return f"after {wait_time:g} min only {free} MB free (needs {min_free} MB)"


def _oom_wait_over(item: dict, job) -> bool:
    """Stop waiting for VRAM: shutdown, pause, abort — or the file's own
    cancel. job.cancel too: it is what the file's cancel button sets, and
    it stays set — a stop followed by "resume" within one poll tick left
    _paused/_abort False again and the wait went on. And the entry's own
    mark, which a cancel leaves when there is no job to set.

    A pause does not end the wait of an entry that runs although the queue
    is paused (run_now): it was paused before that entry started. Counted
    as "stop waiting" it went straight back to waiting, was taken again at
    once, ran into the full card again — forever, and never reached rule 3.
    It waits like any entry, and a second out-of-memory
    pauses it with the rest. A stop ends run-now, and with it the wait."""
    paused = _paused and item.get("id") not in _solo
    return (
        _shutting_down
        or paused
        or _abort
        or bool(item.get("_cancel"))
        or bool(job is not None and job.cancel)
    )


def _oom_cancel_outcome(item: dict, job) -> str:
    """How a wait that was broken off ends: "cancelled" for this one file
    or for an abort, "retry" (stays pending) when the queue was stopped,
    paused or shuts down."""
    with _lock:
        stopped = item["id"] in _requeue_on_abort
        _requeue_on_abort.discard(item["id"])
        cancelled = (
            bool(item.get("_cancel"))
            or bool(job is not None and job.cancel)
            # An abort, unlike a stop, leaves the queue unpaused —
            # and may still be on its way to setting job.cancel.
            or (_abort and not _paused)
        )
    if cancelled and not (stopped or _shutting_down):
        # Cancelled for this one file: it waits no more and does not
        # run again — the same end as cancelling a running job.
        return "cancelled"
    return "retry"  # stopped/paused: stays pending


def _unload_models() -> None:
    """Unload every resident model (caller holds the GPU exclusively)."""
    try:
        from whisper_tm.core.transcription import release_models
        from whisper_tm.core.vocal_isolation import release_separator

        release_models()
        release_separator()
    except Exception as e:  # noqa — measuring without is still better
        logger.warning("Unloading the models before the VRAM wait failed: %s", e)


def _back_to_pending(item_id: int) -> None:
    """The OOM'd entry joins the queue again, at its position."""
    database = _db()
    if not database:
        return
    with database.connection() as conn:
        conn.execute(
            "UPDATE queue_items SET status = 'pending', job_id = '', "
            "started_at = 0, finished_at = 0, error = '' "
            "WHERE id = ? AND status = 'processing'",
            (item_id,),
        )
    _bump()


def _requeue(item_id: int, fresh: bool = False) -> None:
    """Put an entry back into the queue (shutdown/restart). ``fresh``: the
    user's retry — the server going down while it ran is forgotten too."""
    database = _db()
    if not database:
        return
    with database.connection() as conn:
        conn.execute(
            "UPDATE queue_items SET status = 'pending', job_id = '', "
            "started_at = 0, finished_at = 0, error = ''"
            + (", interrupted = 0" if fresh else "")
            + " WHERE id = ? AND status IN ('processing', 'error', 'cancelled')",
            (item_id,),
        )
    _bump()


def retry_item(item_id: int) -> tuple[bool, str]:
    """Re-queue a failed/aborted entry.

    Until now, an 'error' entry was a dead end: _requeue() was private,
    there was no route and no button. Whoever wanted to know what went
    wrong had to re-queue the file by hand.
    """
    database = _db()
    if not database:
        return False, "No database"
    with database.connection() as conn:
        row = conn.execute("SELECT status FROM queue_items WHERE id = ?", (item_id,)).fetchone()
    if not row:
        return False, "Entry not found"
    if row["status"] not in ("error", "cancelled"):
        return False, "Only failed or aborted entries"
    _requeue(item_id, fresh=True)
    _wake()
    qlog(f"↩ Re-queued: entry {item_id}")
    return True, "Queued again"


def retry_items(item_ids: list[int]) -> int:
    """Re-queue several failed/aborted entries.

    For the filtered retry from the web UI (quick filter "Failed"/
    "Aborted" → only the shown entries go back into the queue, not
    indiscriminately all of them).
    """
    database = _db()
    if not database or not item_ids:
        return 0
    marks = ",".join("?" for _ in item_ids)
    with database.connection() as conn:
        cur = conn.execute(
            "UPDATE queue_items SET status = 'pending', job_id = '', "
            "started_at = 0, finished_at = 0, error = '', interrupted = 0 "
            f"WHERE id IN ({marks}) AND status IN ('error', 'cancelled')",
            tuple(item_ids),
        )
        n = cur.rowcount or 0
    _bump()
    if n:
        _wake()
        qlog(f"↩ {n} {'entry' if n == 1 else 'entries'} re-queued (filtered)")
    return n


def retry_failed() -> int:
    """Re-queue all failed/aborted entries."""
    database = _db()
    if not database:
        return 0
    with database.connection() as conn:
        cur = conn.execute(
            "UPDATE queue_items SET status = 'pending', job_id = '', "
            "started_at = 0, finished_at = 0, error = '', interrupted = 0 "
            "WHERE status IN ('error', 'cancelled')"
        )
        n = cur.rowcount or 0
    _bump()
    if n:
        _wake()
        qlog(f"↩ {n} failed/cancelled {'entry' if n == 1 else 'entries'} re-queued")
    return n


def _wake() -> None:
    """Wake the worker from idle (new work is available)."""
    with _cv:
        _cv.notify_all()


# ── Queue editor (bulk operations) ─────────────────────────────────────
# Reason: thousands of entries could
# only be removed via one individual call each, and enabling "conserve
# VRAM" for thousands of pending entries was impossible — there was no way
# to touch the settings of a pending entry at all.
#
# Rules of this block:
# * ONE WHERE builder (editor_where) for counting, listing, folders, dry
#   run and mutation. The number the user sees before clicking IS the
#   number the click hits — otherwise the dry run is just a claim.
# * Mutations exclusively hit status='pending'; the guard sits in the
#   UPDATE/DELETE WHERE, not in a SELECT before it. The running entry is
#   already in the worker's memory anyway (_claim_next copies the row,
#   _process reads item["settings"]).
# * BEGIN IMMEDIATE before counting: Database.connection() runs with
#   isolation_level '' — a SELECT before the UPDATE would otherwise be in
#   autocommit, and the expected check would be no guarantee.
# * No DB work under _lock; _bump()/qlog() only after the transaction.

EDITOR_STATUS = ("pending", "error", "cancelled", "done", "skipped")
EDITOR_IDS_MAX = 100_000
EDITOR_EXCEPT_MAX = 20_000
EDITOR_LIMIT_MAX = 1000
EDITOR_FOLDER_MAX = 500
_folder_cache: dict = {"gen": -1, "data": {}}


class TargetError(ValueError):
    """Invalid target/filter — the API responds 400 with str(e)."""


class HitCountChanged(Exception):
    """expected ≠ actual hit count — the API responds 409."""

    def __init__(self, actual: int):
        super().__init__(f"Match count has changed: now {actual}")
        self.actual = actual


def _gen() -> int:
    with _lock:
        return _generation


def _scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def editor_normalize_filter(filt: dict | None) -> dict:
    """Raw filter (API/GET args) → normalized FILTER dict.

    This is exactly the shape the server echoes back, and exactly the shape
    the client sends again — a filter is thus reproducible and comparable.
    """
    if filt is None:
        filt = {}
    if not isinstance(filt, dict):
        raise TargetError("filter must be an object")

    status = filt.get("status")
    if status is None or status == "" or status == []:
        status = ["pending"]
    if isinstance(status, str):
        status = [t for t in status.split(",")]
    if not isinstance(status, (list, tuple)):
        raise TargetError("status must be a list")
    out_status: list[str] = []
    for st in status:
        st = str(st or "").strip().lower()
        if not st:
            continue
        if st == "processing":
            raise TargetError("Status 'processing' is not filterable")
        if st not in EDITOR_STATUS:
            raise TargetError(f"Unknown status: {st}")
        if st not in out_status:
            out_status.append(st)
    if not out_status:
        out_status = ["pending"]

    d = filt.get("dir")
    if d is not None:
        d = str(d)
        if d != "":
            d = paths.canon(Path(d).expanduser())
            if not d:
                raise TargetError("dir must be absolute")

    prefix = str(filt.get("prefix") or "").strip()
    if len(prefix) > 1024:
        raise TargetError("prefix too long (max 1024)")
    if prefix.startswith("/") or ".." in prefix.split("/"):
        raise TargetError("prefix must be relative")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    rec = filt.get("recursive", True)
    if isinstance(rec, str):
        rec = rec.strip().lower() not in ("0", "false", "no", "off", "")
    rec = bool(rec)

    q = str(filt.get("q") or "")[:200]
    # The file name alone: "intro" as a name, not every file in an intro folder
    name = str(filt.get("name") or "").strip()[:200]

    setting = filt.get("setting")
    if setting is not None:
        if not isinstance(setting, dict):
            raise TargetError("setting must be an object")
        key = str(setting.get("key") or "")
        if not re.fullmatch(r"[A-Za-z_]+", key):
            raise TargetError("setting.key is invalid")
        value = setting.get("value")
        if not _scalar(value):
            raise TargetError("setting.value must be a scalar")
        setting = {"key": key, "value": value}

    return {
        "status": out_status,
        "dir": d,
        "prefix": prefix,
        "recursive": rec,
        "q": q,
        "name": name,
        "setting": setting,
    }


def _like_pattern(q: str) -> str:
    """Substring search: neutralize wildcards, `*` afterwards as the wildcard."""
    esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return "%" + esc.replace("*", "%") + "%"


def _status_clause(allowed: tuple[str, ...]) -> tuple[str, list]:
    if len(allowed) == 1:
        return "status = ?", [allowed[0]]
    placeholders = ",".join("?" * len(allowed))
    return f"status IN ({placeholders})", list(allowed)


def editor_where(target: dict, *, fixed_status: str | tuple[str, ...] | None) -> tuple[str, list]:
    """The ONE translation of a target into SQL.

    ``target`` is ``{'ids': [int]}`` or
    ``{'filter': FILTER, 'except_ids': [int]}``. Returns: WHERE fragment
    (without ``WHERE``) and parameter list. ``fixed_status`` restricts the
    status — a single one (mutations: 'pending'; counters: 'processing') or
    a tuple (retry: error+cancelled); with a filter, the INTERSECTION with
    ``filter.status`` then applies, so the error view only hits the errors.
    Without ``fixed_status``, ``filter.status`` applies.
    """
    parts: list[str] = []
    par: list = []
    if isinstance(fixed_status, str):
        fixed_status = (fixed_status,)

    ids = target.get("ids")
    filt = target.get("filter")
    if ids is not None:
        if not isinstance(ids, (list, tuple)):
            raise TargetError("ids must be numbers")
        if not ids:
            raise TargetError("ids is empty")
        if len(ids) > EDITOR_IDS_MAX:
            raise TargetError(f"Too many ids (max {EDITOR_IDS_MAX})")
        try:
            ids = [int(i) for i in ids]
        except (TypeError, ValueError):
            raise TargetError("ids must be numbers") from None
        k, kp = _status_clause(fixed_status or EDITOR_STATUS)
        parts.append(k)
        par.extend(kp)
        # ONE parameter for any number of ids — MAX_VARIABLE_NUMBER doesn't matter.
        parts.append("id IN (SELECT value FROM json_each(?))")
        par.append(json.dumps(ids))
        return " AND ".join(parts), par

    if filt is None:
        raise TargetError("Exactly one of ids, filter, or all is required")
    filt = editor_normalize_filter(filt)
    if fixed_status and len(fixed_status) == 1:
        allowed = fixed_status
    elif fixed_status:
        allowed = tuple(s for s in filt["status"] if s in fixed_status)
        if not allowed:
            raise TargetError(
                "This action only applies to entries with status " + " or ".join(fixed_status)
            )
    else:
        allowed = tuple(filt["status"])
    k, kp = _status_clause(allowed)
    parts.append(k)
    par.extend(kp)
    ex = target.get("except_ids") or []
    if ex:
        if len(ex) > EDITOR_EXCEPT_MAX:
            raise TargetError(f"Too many except_ids (max {EDITOR_EXCEPT_MAX})")
        try:
            ex = [int(i) for i in ex]
        except (TypeError, ValueError):
            raise TargetError("except_ids must be numbers") from None
        parts.append("id NOT IN (SELECT value FROM json_each(?))")
        par.append(json.dumps(ex))
    if filt["dir"] is not None:
        parts.append("base_dir = ?")
        par.append(filt["dir"])
    p = filt["prefix"]
    if p:
        # Range comparison instead of LIKE: uses idx_queue_filename/idx_queue_status.
        parts.append("filename >= ? AND filename < ?")
        par.extend([p, p[:-1] + chr(ord("/") + 1)])
    if not filt["recursive"]:
        # Also at the root (prefix ''): then "files with no folder part".
        parts.append("instr(substr(filename, ?), '/') = 0")
        par.append(len(p) + 1)
    if filt["q"]:
        parts.append("filename LIKE ? ESCAPE '\\'")
        par.append(_like_pattern(filt["q"]))
    if filt["name"]:
        # Both sides folded: case doesn't count, umlauts included
        # (database._name_folded)
        parts.append("wx_name_folded(filename) LIKE ? ESCAPE '\\'")
        par.append(_like_pattern(filt["name"].casefold()))
    st = filt["setting"]
    if st is not None:
        # json_valid guard: _row_to_item tolerates broken JSON,
        # json_extract would throw on it.
        expression = "CASE WHEN json_valid(settings_json) THEN json_extract(settings_json, ?) END"
        par.append("$." + st["key"])
        if st["value"] is None:
            parts.append(f"{expression} IS NULL")
        else:
            value = st["value"]
            if isinstance(value, bool):
                value = 1 if value else 0
            parts.append(f"{expression} = ?")
            par.append(value)
    return " AND ".join(parts), par


def _editor_description(target: dict) -> str:
    """Short description of the target — for qlog and the response."""
    if target.get("ids") is not None:
        return f"{len(target['ids'])} entries by selection"
    f = editor_normalize_filter(target.get("filter"))
    parts = []
    if f["status"] != ["pending"]:
        parts.append("Status " + "/".join(f["status"]))
    if f["prefix"]:
        parts.append(f"Folder {f['prefix']}")
    if not f["recursive"]:
        parts.append("without subfolders")
    if f["q"]:
        parts.append(f"Search '{f['q']}'")
    if f["name"]:
        parts.append(f"Name '{f['name']}'")
    if f["dir"] is not None:
        parts.append(f"Root {f['dir'] or '(media root)'}")
    if f["setting"]:
        parts.append(f"{f['setting']['key']}={f['setting']['value']}")
    text = " · ".join(parts) if parts else "all pending"
    ex = target.get("except_ids") or []
    if ex:
        text += f" minus {len(ex)}"
    return text


def _editor_count(conn, target: dict, *, fixed_status: str | tuple[str, ...]) -> dict:
    """Hit count, sample, roots — with the same WHERE as the mutation."""
    if isinstance(fixed_status, str):
        fixed_status = (fixed_status,)
    w, p = editor_where(target, fixed_status=fixed_status)
    affected = conn.execute(f"SELECT COUNT(*) FROM queue_items WHERE {w}", p).fetchone()[0]
    sample = [
        r[0]
        for r in conn.execute(
            f"SELECT filename FROM queue_items WHERE {w} ORDER BY position, id LIMIT 5", p
        )
    ]
    roots = [
        {"dir": r[0] or "", "count": r[1]}
        for r in conn.execute(
            f"SELECT base_dir, COUNT(*) FROM queue_items WHERE {w} "
            f"GROUP BY base_dir ORDER BY 2 DESC",
            p,
        )
    ]
    # "running and left untouched" only exists for actions on pending
    # entries: for retry (error/cancelled) a running entry was never a
    # candidate — reporting it would imply it got skipped.
    running = 0
    if fixed_status == ("pending",):
        wp, pp = editor_where(target, fixed_status="processing")
        running = conn.execute(f"SELECT COUNT(*) FROM queue_items WHERE {wp}", pp).fetchone()[0]
    ids = target.get("ids")
    return {
        "affected": int(affected),
        "sample": sample,
        "roots": roots,
        "skipped_processing": int(running),
        "requested": len(ids) if ids is not None else None,
        "description": _editor_description(target),
    }


def editor_list(filt: dict, offset: int = 0, limit: int = 200) -> dict:
    """Filtered, paginated list — its own SQL, no list_items() cache."""
    database = _db()
    filt = editor_normalize_filter(filt)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 200
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        offset = 0
    limit = max(1, min(limit, EDITOR_LIMIT_MAX))
    offset = max(0, offset)
    empty = {
        "total": 0,
        "offset": offset,
        "limit": limit,
        "generation": _gen(),
        "items": [],
        "processing": None,
        "processing_items": [],
        "roots": [],
        "filter": filt,
    }
    if not database:
        return empty
    target = {"filter": filt}
    w, p = editor_where(target, fixed_status=None)
    wp, pp = editor_where(target, fixed_status="processing")
    with database.connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM queue_items WHERE {w}", p).fetchone()[0]
        items = [
            _row_to_item(r)
            for r in conn.execute(
                f"SELECT * FROM queue_items WHERE {w} ORDER BY position, id LIMIT ? OFFSET ?",
                [*p, limit, offset],
            )
        ]
        roots = [
            {"dir": r[0] or "", "count": r[1]}
            for r in conn.execute(
                f"SELECT base_dir, COUNT(*) FROM queue_items WHERE {w} "
                f"GROUP BY base_dir ORDER BY 2 DESC",
                p,
            )
        ]
        runs = [
            _row_to_item(r)
            for r in conn.execute(
                f"SELECT * FROM queue_items WHERE {wp} ORDER BY position, id LIMIT ?",
                [*pp, MAX_LANES * 2],
            )
        ]
    return {
        "total": int(total),
        "offset": offset,
        "limit": limit,
        "generation": _gen(),
        "items": items,
        # 'processing' stays the first one; with lanes there are more.
        "processing": runs[0] if runs else None,
        "processing_items": runs,
        "roots": roots,
        "filter": filt,
    }


def editor_folders(filt: dict) -> dict:
    """Child folders (depth 1) under filter.prefix, per (root, segment).

    Tens of thousands of entries are not a list, they're a tree —
    "videos/shows (1,234)" is the useful row. Cached per generation and filter.
    """
    database = _db()
    filt = editor_normalize_filter(filt)
    filt["recursive"] = True
    prefix = filt["prefix"]
    st = filt["setting"]
    key = (
        tuple(filt["status"]),
        filt["dir"],
        prefix,
        filt["q"],
        filt["name"],
        st["key"] if st else None,
        json.dumps(st["value"]) if st else None,
    )
    with _lock:
        gen = _generation
        if _folder_cache["gen"] == gen and key in _folder_cache["data"]:
            return dict(_folder_cache["data"][key])
    try:
        media_dir = paths.canon(Path(config.media_dir)) if config.media_dir else ""
    except Exception:  # config not loaded (tools/tests)
        media_dir = ""
    empty = {
        "total": 0,
        "prefix": prefix,
        "dir": filt["dir"],
        "media_dir": media_dir,
        "folders": [],
        "files_here": 0,
        "roots": [],
        "truncated": False,
        "generation": gen,
    }
    if not database:
        return empty
    w, p = editor_where({"filter": filt}, fixed_status=None)
    # Roots ONLY by status: the root selection is a filter over the tree
    # and must stay in place once a root has been chosen or a folder
    # entered — otherwise it could never be deselected again.
    ww, wp = editor_where({"filter": {"status": filt["status"]}}, fixed_status=None)
    groups: dict[tuple[str, str], dict] = {}
    files_here = 0
    total = 0
    n = len(prefix)
    with database.connection() as conn:
        roots = [
            {"dir": r[0] or "", "count": r[1]}
            for r in conn.execute(
                f"SELECT base_dir, COUNT(*) FROM queue_items WHERE {ww} "
                f"GROUP BY base_dir ORDER BY 2 DESC, 1",
                wp,
            )
        ]
        for base_dir, filename in conn.execute(
            f"SELECT base_dir, filename FROM queue_items WHERE {w}", p
        ):
            total += 1
            bd = base_dir or ""
            remainder = filename[n:]
            i = remainder.find("/")
            if i < 0:
                files_here += 1
                continue
            name = remainder[:i]
            g = groups.get((bd, name))
            if g is None:
                g = groups[(bd, name)] = {
                    "name": name,
                    "prefix": prefix + name + "/",
                    "dir": bd,
                    "count": 0,
                    "has_children": False,
                }
            g["count"] += 1
            if not g["has_children"] and "/" in remainder[i + 1 :]:
                g["has_children"] = True
    folders = sorted(groups.values(), key=lambda g: (-g["count"], g["name"]))
    truncated = len(folders) > EDITOR_FOLDER_MAX
    result = {
        "total": total,
        "prefix": prefix,
        "dir": filt["dir"],
        "media_dir": media_dir,
        "folders": folders[:EDITOR_FOLDER_MAX],
        "files_here": files_here,
        "roots": roots,
        "truncated": truncated,
        "generation": gen,
    }
    with _lock:
        if _generation == gen:
            if _folder_cache["gen"] != gen:
                _folder_cache["gen"] = gen
                _folder_cache["data"] = {}
            _folder_cache["data"][key] = result
    return dict(result)


def _patch_text(patch: dict) -> str:
    parts = []
    for k, v in patch.items():
        if v is None:
            parts.append(f"{k}→default")
        elif isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _editor_mutation(
    target: dict,
    *,
    dry_run: bool,
    expected: int | None,
    execute,
    fixed_status: str | tuple[str, ...] = "pending",
) -> dict:
    """Shared frame: lock, count, expected check, mutation.

    ``fixed_status`` is the guard in the UPDATE/DELETE WHERE: 'pending'
    for settings/removal/moving, ('error', 'cancelled') for retry.
    """
    database = _db()
    if not database:
        raise TargetError("No database")
    with database.connection() as conn:
        if not dry_run:
            # Write lock NOW: counting and writing are one unit.
            conn.execute("BEGIN IMMEDIATE")
        counted = _editor_count(conn, target, fixed_status=fixed_status)
        if dry_run:
            return {**counted, "dry_run": True, "generation": _gen()}
        if expected is not None and counted["affected"] != expected:
            raise HitCountChanged(counted["affected"])
        w, p = editor_where(target, fixed_status=fixed_status)
        n = execute(conn, w, p)
    _bump()
    return {**counted, "affected": int(n), "dry_run": False, "generation": _gen()}


def editor_settings(
    target: dict, patch: dict, *, dry_run: bool = False, expected: int | None = None
) -> dict:
    """Merge settings of pending entries: only sent keys,
    ``None`` removes the key (config default applies again)."""
    if not isinstance(patch, dict) or not patch:
        raise TargetError("settings empty or only unknown keys")
    for _k, v in patch.items():
        if not _scalar(v):
            raise TargetError("Values must be scalars")

    def _apply_to(conn, w, p):
        cur = conn.execute(
            "UPDATE queue_items SET settings_json = json_patch("
            "CASE WHEN json_valid(settings_json) THEN settings_json ELSE '{}' END, ?) "
            f"WHERE {w}",
            [json.dumps(patch, ensure_ascii=False), *p],
        )
        return cur.rowcount or 0

    r = _editor_mutation(target, dry_run=dry_run, expected=expected, execute=_apply_to)
    r["patch"] = patch
    if not dry_run and r["affected"]:
        qlog(
            f"✎ Queue editor: {r['affected']} pending entries changed "
            f"({_patch_text(patch)}) — {r['description']}"
        )
    return r


def editor_remove(target: dict, *, dry_run: bool = False, expected: int | None = None) -> dict:
    """Remove pending entries of the target — never error/cancelled/done."""

    def _apply_to(conn, w, p):
        cur = conn.execute(f"DELETE FROM queue_items WHERE {w}", p)
        return cur.rowcount or 0

    r = _editor_mutation(target, dry_run=dry_run, expected=expected, execute=_apply_to)
    if not dry_run and r["affected"]:
        qlog(f"🧹 Queue editor: {r['affected']} pending entries removed — {r['description']}")
    return r


def editor_move(
    target: dict, where_to: str, *, dry_run: bool = False, expected: int | None = None
) -> dict:
    """Move the target as a block to front/back — order within the group stays."""
    if where_to not in ("top", "bottom"):
        raise TargetError("to must be 'top' or 'bottom'")

    def _apply_to(conn, w, p):
        if where_to == "top":
            basis = (
                "SELECT COALESCE(MIN(position), 1) AS p FROM queue_items WHERE status = 'pending'"
            )
            put = "basis.p - target.n - 1 + target.rn"
        else:
            # ALL rows, like enqueue append — behind everything that ever was.
            basis = "SELECT COALESCE(MAX(position), 0) AS p FROM queue_items"
            put = "basis.p + target.rn"
        conn.execute(
            "WITH target AS (SELECT id, ROW_NUMBER() OVER (ORDER BY position, id) AS rn, "
            f"COUNT(*) OVER () AS n FROM queue_items WHERE {w}), "
            f"basis AS ({basis}) "
            f"UPDATE queue_items SET position = {put} "
            "FROM target, basis WHERE queue_items.id = target.id "
            "AND queue_items.status = 'pending'",
            p,
        )
        # cursor.rowcount is -1 for WITH…UPDATE → ask changes() instead.
        return conn.execute("SELECT changes()").fetchone()[0]

    r = _editor_mutation(target, dry_run=dry_run, expected=expected, execute=_apply_to)
    r["to"] = where_to
    if not dry_run and r["affected"]:
        if where_to == "top":
            qlog(f"⤒ Queue editor: {r['affected']} entries moved to the front — {r['description']}")
        else:
            qlog(f"⤓ Queue editor: {r['affected']} entries moved to the back — {r['description']}")
    return r


def editor_retry(target: dict, *, dry_run: bool = False, expected: int | None = None) -> dict:
    """Re-queue failed/aborted entries of the target.

    Previously the editor sent an empty ``/api/queue/retry`` for "all"
    — that hit error AND cancelled, while the view only showed one status:
    the number on the button was not the number that got hit. Now the same
    WHERE as list and dry run, guard ('error', 'cancelled') in the UPDATE
    WHERE; with a filter, the intersection with filter.status counts.
    """

    def _apply_to(conn, w, p):
        cur = conn.execute(
            "UPDATE queue_items SET status = 'pending', job_id = '', "
            f"started_at = 0, finished_at = 0, error = '', interrupted = 0 WHERE {w}",
            p,
        )
        return cur.rowcount or 0

    r = _editor_mutation(
        target,
        dry_run=dry_run,
        expected=expected,
        execute=_apply_to,
        fixed_status=("error", "cancelled"),
    )
    if not dry_run and r["affected"]:
        _wake()
        qlog(f"↩ Queue editor: {r['affected']} entries re-queued — {r['description']}")
    return r


#: The server went down this often while an entry ran: it isn't started
#: again (requeue_stale)
MAX_INTERRUPTIONS = 2
INTERRUPTED_ERROR = (
    "The server went down twice while this file ran (a crash, the "
    "out-of-memory killer, a power cut) — it may be what takes it "
    "down, so it isn't started a third time. Retry runs it again."
)


def requeue_stale() -> int:
    """On startup: re-queue interrupted entries.

    A 'processing' entry after a restart means the server went down in the
    middle of the file (a clean shutdown puts its entries back itself) —
    the job is gone, so back to 'pending' so the queue picks it up again.
    Counted, though: a file that takes the process down ran first after
    every start, and a restart policy (Docker, systemd) made that a loop.
    Interrupted once, it runs alone next time (_claim_next) — so that the
    second time names the file, not the ones beside it — and interrupted
    twice it is booked as an error.
    """
    database = _db()
    if not database:
        return 0
    with database.connection() as conn:
        given_up = conn.execute(
            "UPDATE queue_items SET status = 'error', error = ?, job_id = '', "
            "finished_at = ?, interrupted = interrupted + 1 "
            "WHERE status = 'processing' AND interrupted >= ? RETURNING filename",
            (INTERRUPTED_ERROR, time.time(), MAX_INTERRUPTIONS - 1),
        ).fetchall()
        cur = conn.execute(
            "UPDATE queue_items SET status = 'pending', job_id = '', "
            "started_at = 0, interrupted = interrupted + 1 WHERE status = 'processing'"
        )
        n = cur.rowcount or 0
    _bump()
    for row in given_up:
        logger.warning(
            "Queue: %s — the server went down twice while it ran; booked as an error",
            row["filename"],
        )
        qlog(
            f"❌ {row['filename']}: the server went down twice while it ran — "
            "not started again (Retry runs it)"
        )
    if n:
        qlog(
            f"{n} interrupted {'entry' if n == 1 else 'entries'} re-queued after "
            "restart — each runs alone next"
        )
    return n


# ── Status / snapshot ─────────────────────────────────────────────────────


def snapshot() -> dict:
    """Full picture of the current run — basis for SSE and /api/queue.

    The DB portions come from _db_state() and thus from the cache, as long
    as the queue hasn't changed. The remainder (running job, pause, log) only
    lives in memory anyway.
    """
    state = _db_state()
    counts = state["counts"]
    started_at = state["started_at"]
    finished_at = state["finished_at"]

    with _lock:
        cur = dict(_current) if _current else None
        in_flight_rows = list(_running.values())
        paused = _paused
        solo = len(_solo)
        abort = _abort
        log = list(_log)
        log_total = _log_total
        gen = _generation
    # Read with _current in one go: between two reads a lane could claim,
    # and the UI got a current_file of None next to a running entry.
    in_flight = _running_view(in_flight_rows)

    total = sum(counts.values())
    # 'cancelled' counts as done: the entry does not come back on its own —
    # if it were missing here, progress would stay stuck below 100% forever.
    processed = counts["done"] + counts["error"] + counts["skipped"] + counts["cancelled"]
    # 'processing' MUST count: between _claim_next() (DB status is already
    # 'processing') and setting _current there is a brief window in which
    # cur is still None. Without this term the snapshot wrongly reported
    # "done" there — the frontend showed "queue finished" while a file was
    # actually just starting.
    running = bool(cur) or counts["pending"] > 0 or counts["processing"] > 0

    return {
        "running": running,
        "paused": paused,
        # Entries that run although the queue is paused (run_now)
        "solo": solo if paused else 0,
        "abort": abort,
        "current_file": cur["filename"] if cur else None,
        # Identity (docs/PATHS.md): the name is relative to the ENTRY's
        # root, the browser's file list relative to the one CURRENTLY
        # open — after jumping into the file's folder these are two names
        # for the same file. That's why comparison happens via abs_path.
        "current_abs": _item_abs_path(cur) if cur else None,
        "current_dir": (cur.get("base_dir") or cur.get("dir") or "") if cur else None,
        "current_job_id": cur["job_id"] if cur else None,
        "current_item_id": cur["id"] if cur else None,
        # Lanes: every running entry, oldest first. current_* above is its
        # first one, for the readers that know only one.
        "running_items": in_flight,
        "running_count": len(in_flight),
        "lanes": len(_workers) or lanes(),
        "total": total,
        "processed": processed,
        "pending": counts["pending"],
        "done_count": counts["done"],
        "fail_count": counts["error"],
        "skip_count": counts["skipped"],
        "cancel_count": counts["cancelled"],
        # Re-queueable (retry button in the frontend)
        "retryable_count": counts["error"] + counts["cancelled"],
        "started_at": started_at,
        "finished_at": None if running else finished_at,
        "log": log,
        "log_total": log_total,
        # Change counter: the open queue editor reloads on change.
        "generation": gen,
    }


# The media row of an entry. The same identity as _item_abs_path, in SQL:
# entries are stored with a canonical root and a relative name (enqueue
# checks both), so joining them with "/" gives the canonical abs_path —
# every pending entry finds its row this way.
_MEDIA_JOIN = "LEFT JOIN media_files m ON m.abs_path = rtrim(q.base_dir, '/') || '/' || q.filename"
_KIND_SQL = "COALESCE(json_extract(q.settings_json, '$.job'), 'transcription')"


def eta_inputs(now: float | None = None, per_group: int = 500) -> dict:
    """What core/eta.py needs for the queue's time left.

    Per kind of job ('voices' or 'transcription'): the pending entries with
    a known duration (count, audio seconds) and those without one (per
    extension: count, bytes); the running entry; and as samples the most
    recent finished entries — ``per_group`` short and long ones each, with
    their wall time. Recent, because the pace follows the settings: a
    voices run can change speed within hours (Spare VRAM, the
    diarization worker's VRAM limit).

    With lanes an entry's wall time is its latency, not its cost — three
    entries share the card. So a sample's seconds are its SHARE of the
    wall clock: every moment of its run counts 1/n while n entries ran
    (_clock_shares over every entry that ran in the samples' span, of any
    outcome). Shares add up to wall time, whatever the lanes were at the
    time — a pause, a change of lanes or a long file beside short ones
    don't skew it. ``parallelism`` (summed wall time ÷ summed shares of the
    samples) is only shown. ``running`` lists every running entry,
    ``current`` stays the newest of them.
    """
    database = _db()
    if not database:
        return {}
    now = now or time.time()
    out: dict = {
        "pending": {},
        "samples": {},
        "current": None,
        "running": [],
        "parallelism": 1.0,
        "last_hour": {},
    }
    with database.connection() as conn:
        for r in conn.execute(f"""
                SELECT {_KIND_SQL} AS kind,
                       SUM(COALESCE(m.duration_sec, 0) > 0) AS known,
                       SUM(CASE WHEN m.duration_sec > 0 THEN m.duration_sec END) AS audio
                  FROM queue_items q {_MEDIA_JOIN}
                 WHERE q.status = 'pending' GROUP BY kind"""):
            out["pending"][r["kind"]] = {
                "known": r["known"] or 0,
                "audio_sec": r["audio"] or 0.0,
                "no_duration": [],
            }
        for r in conn.execute(f"""
                SELECT {_KIND_SQL} AS kind, LOWER(COALESCE(m.extension, '')) AS ext,
                       COUNT(*) AS n, SUM(COALESCE(m.file_size, 0)) AS size
                  FROM queue_items q {_MEDIA_JOIN}
                 WHERE q.status = 'pending' AND COALESCE(m.duration_sec, 0) = 0
                 GROUP BY kind, ext"""):
            out["pending"][r["kind"]]["no_duration"].append((r["ext"], r["n"], r["size"]))
        running = conn.execute(f"""
                SELECT {_KIND_SQL} AS kind, m.duration_sec AS duration, q.started_at
                  FROM queue_items q {_MEDIA_JOIN}
                 WHERE q.status = 'processing' ORDER BY q.started_at DESC""").fetchall()
        out["running"] = [
            {
                "kind": r["kind"],
                "duration": r["duration"] or 0.0,
                "started_at": r["started_at"] or now,
            }
            for r in running
        ]
        out["current"] = out["running"][0] if out["running"] else None
        kinds = set(out["pending"]) | {r["kind"] for r in out["running"]}
        spans_of = {}
        for kind in kinds:
            spans_of[kind] = []
            for border in ("m.duration_sec < 60", "m.duration_sec >= 60"):
                spans_of[kind] += [
                    (r["dur"], r["a"], r["b"])
                    for r in conn.execute(
                        f"""
                        SELECT m.duration_sec AS dur, q.started_at AS a, q.finished_at AS b
                          FROM queue_items q {_MEDIA_JOIN}
                         WHERE q.status = 'done' AND {_KIND_SQL} = ?
                           AND q.started_at > 0 AND q.finished_at > q.started_at
                           AND m.duration_sec > 0 AND {border}
                         ORDER BY q.finished_at DESC LIMIT ?""",
                        (kind, per_group),
                    )
                ]
        every = [s for spans in spans_of.values() for s in spans]
        if every:
            lo = min(a for _, a, _ in every)
            hi = max(b for _, _, b in every)
            # Everything that held a lane in that span: finished in any way,
            # or still running (up to now)
            ran = [
                (r[0], r[1])
                for r in conn.execute(
                    """
                    SELECT started_at,
                           CASE WHEN status = 'processing' THEN ? ELSE finished_at END
                      FROM queue_items
                     WHERE started_at > 0 AND started_at < ?
                       AND (status = 'processing' OR finished_at > ?)""",
                    (now, hi, lo),
                )
            ]
            share = _clock_shares(ran)
            for kind, spans in spans_of.items():
                out["samples"][kind] = [(dur, share(a, b)) for dur, a, b in spans]
            cost = sum(share(a, b) for _, a, b in every)
            wall = sum(b - a for _, a, b in every)
            out["parallelism"] = round(wall / cost, 2) if cost > 0 else 1.0
        else:
            for kind in kinds:
                out["samples"][kind] = []
        r = conn.execute(
            f"""
                SELECT COUNT(*) AS files, SUM(COALESCE(m.duration_sec, 0)) AS audio
                  FROM queue_items q {_MEDIA_JOIN}
                 WHERE q.status = 'done' AND q.finished_at >= ?""",
            (now - 3600,),
        ).fetchone()
        out["last_hour"] = {"files": r["files"] or 0, "audio_sec": r["audio"] or 0.0}
    return out


def _clock_shares(spans: list[tuple[float, float]]):
    """``share(a, b)``: the wall-clock share of a run from a to b, where
    every moment counts 1/n while n of ``spans`` were running."""
    events: dict[float, int] = {}
    for a, b in spans:
        if b is None or b <= a:
            continue
        events[a] = events.get(a, 0) + 1
        events[b] = events.get(b, 0) - 1
    times = sorted(events)
    clock, rate = [], []
    acc, n = 0.0, 0
    for i, t in enumerate(times):
        if i and n > 0:
            acc += (t - times[i - 1]) / n
        clock.append(acc)
        n += events[t]
        rate.append(1.0 / n if n > 0 else 0.0)

    def at(t: float) -> float:
        i = bisect.bisect_right(times, t) - 1
        return 0.0 if i < 0 else clock[i] + (t - times[i]) * rate[i]

    def share(a: float, b: float) -> float:
        return max(0.0, at(b) - at(a))

    return share


# ── Control ─────────────────────────────────────────────────────────────
# Note: there is deliberately NO push channel for queue changes — the
# frontend hangs off the SSE long-lived stream /api/queue/stream, which
# reads the state once a second. The earlier SocketIO emit never had a
# client.


def pause() -> None:
    global _paused
    with _lock:
        _paused = True
        _solo.clear()
    qlog("Queue paused")


def resume() -> None:
    global _paused, _abort, _aborting
    with _cv:
        _paused = False
        _abort = False
        _aborting = False
        _solo.clear()
        _cv.notify_all()
    qlog("Queue resumed")


def run_now(item_ids) -> int:
    """Let these entries run although the queue is paused — nothing else of
    it starts. Without a pause nothing changes: they are at the front, and
    the queue runs. Returns how many run so."""
    ids = {int(i) for i in item_ids or [] if i}
    with _cv:
        if not _paused or not ids:
            return 0
        _solo.update(ids)
        _cv.notify_all()
    what = "the new entry runs" if len(ids) == 1 else f"{len(ids)} new entries run"
    qlog(f"Queue paused — {what} anyway, the rest waits")
    return len(ids)


def _prune_solo() -> None:
    """Forget the entries of _solo that will not run any more — removed,
    or finished another way."""
    database = _db()
    with _lock:
        ids = sorted(_solo)
    if not ids or not database:
        return
    with database.connection() as conn:
        left = {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM queue_items WHERE id IN ({','.join('?' * len(ids))}) "
                "AND status IN ('pending', 'processing')",
                ids,
            )
        }
    with _lock:
        _solo.intersection_update(left | (_solo - set(ids)))


def is_paused() -> bool:
    with _lock:
        return _paused


def _cancel_running(requeue: bool = False) -> list[dict]:
    """Set the abort flag on every running job. Returns the entries.

    An entry whose job doesn't exist yet (claimed, still checking the file
    on the NAS) gets the mark itself — _process hands it to the job when
    it makes one. Before lanes that window simply lost the cancel: the
    file ran through while "Stopping…" was shown.
    """
    jm = _jm()
    with _lock:
        hit = list(_running.values())
        for item in hit:
            item["_cancel"] = True
            if requeue:
                _requeue_on_abort.add(item["id"])
            else:
                # An abort after a stop: the stop's "put it back" no longer
                # holds — otherwise every running entry ran again.
                _requeue_on_abort.discard(item["id"])
        jids = [item.get("job_id") for item in hit]
    for jid in jids:
        job = jm.get(jid) if (jm and jid) else None
        if job:
            job.cancel = True
    return [dict(item) for item in hit]


def cancel_item(item_id: int) -> bool:
    """Cancel one running entry — also before its job exists.

    The ✕ on a running row of the Queue tab: its lane may still be checking
    the file on the NAS, with no job to cancel yet. The mark on the entry
    reaches the job when the lane makes it. Ends as 'cancelled', like
    /api/cancel/<job_id>. False when the entry isn't running (anymore).
    """
    with _lock:
        item = _running.get(item_id)
        if item is None:
            return False
        item["_cancel"] = True
        _requeue_on_abort.discard(item_id)  # cancel beats an earlier stop
        jid = item.get("job_id")
    jm = _jm()
    job = jm.get(jid) if (jm and jid) else None
    if job:
        job.cancel = True
    return True


def busy() -> bool:
    """Is any entry running right now? (for other GPU users)"""
    with _lock:
        return bool(_running)


def running_items() -> list[dict]:
    """The running entries, oldest first — what snapshot() reports."""
    with _lock:
        hit = list(_running.values())
    return _running_view(hit)


def _running_view(hit: list[dict]) -> list[dict]:
    return [
        {
            "item_id": item["id"],
            "job_id": item.get("job_id") or "",
            "filename": item["filename"],
            "abs": _item_abs_path(item),
            "dir": item.get("dir") or "",
            "started_at": item.get("started_at") or 0,
        }
        for item in hit
    ]


def stop(clear: bool = False) -> tuple[bool, str]:
    """Pause processing: abort the running job, pause the queue.

    The pending entries are KEPT. This is the important difference from the
    old in-memory batch queue: there, "stop == discard" had no lasting
    effect, because the list only lived in RAM anyway. Today it is
    persistent — an accidental click must not wipe out a thousand entries.
    For deliberately discarding, there's clear_pending() (/api/queue/clear).

    The RUNNING entry is part of this too: it gets aborted and put back to
    'pending', so it runs again after "resume". It used to end up as
    'error' ("Aborted") — whoever paused the queue briefly to do something
    else lost exactly the file that was in progress, and had no retry path
    to get it back.
    """
    global _abort, _paused
    with _cv:
        _abort = True
        _paused = True
        _solo.clear()

    hit = _cancel_running(requeue=True)

    if clear:
        clear_pending()

    qlog("Stopped (pending entries are kept)")
    if not hit:
        return True, "Queue paused"
    if len(hit) == 1:
        return True, "Running job is being cancelled, queue paused"
    return True, f"{len(hit)} running jobs are being cancelled, queue paused"


def abort() -> tuple[bool, int]:
    """Total abort: cancel the running job and delete ALL pending entries.

    After this the queue stands still. This is the deliberate "get rid of
    it" button — unlike stop(), which only PAUSES and keeps everything
    (including the running entry).

    The running entry stays in the history as 'cancelled' (same treatment
    as the targeted individual abort via /api/cancel). It used to be
    deleted here without a trace — then the file list no longer showed
    THAT a run had been aborted, and the file silently fell back to "open".
    Visible and re-queueable via retry is the
    more honest answer.

    Returns:
        (ok, number of removed pending entries)
    """
    global _abort, _paused, _aborting
    with _cv:
        _abort = True
        _aborting = True
        # Held while the queue empties: un-paused and woken first, the lanes
        # of a stopped queue took pending entries before the DELETE reached
        # them, and those ran to the end after a total abort.
        _paused = True
        _solo.clear()

    # Running entries first: an OOM wait that checks in between must
    # already see them cancelled, or it puts its entry back after the clear.
    try:
        hit = _cancel_running()  # set the abort flag on every running job
        n = clear_pending()
    finally:
        # Also when the DELETE fails (a locked database): left set, every
        # entry claimed after a Resume was booked 'cancelled'
        with _cv:
            _paused = False  # don't stay paused — nothing is waiting anymore
            _aborting = False
            _cv.notify_all()

    if hit:
        names = ", ".join(item["filename"] for item in hit)
        qlog(f"Aborted — queue cleared (running: {names} → aborted)")
    else:
        qlog("Aborted — queue cleared")
    return True, n


def shutdown(timeout: float = 30.0) -> None:
    """Process is shutting down: stop the worker, leave the queue UNTOUCHED.

    The running entry gets aborted and put back to 'pending' — it runs
    again on the next start. Pending entries stay untouched. That is
    exactly the point of a persistent queue: it should survive a restart,
    not be wiped out by one.
    """
    global _shutting_down
    with _lock:
        _shutting_down = True

    hit = _cancel_running()
    if not hit:
        stop_worker(timeout)
        return

    logger.info(
        "Queue: waiting for the running %s (%s)…",
        "job" if len(hit) == 1 else "jobs",
        ", ".join(item["filename"] for item in hit),
    )
    # stop_worker() waits for the worker loops to end — each lane writes its
    # entry back to 'pending' via _process() in the process.
    stop_worker(timeout)

    # Safety net: an entry whose lane didn't make it in time is still
    # stuck at 'processing'. requeue_stale() catches that on the next start
    # — reset it directly here anyway. As before lanes, an entry the
    # shutdown broke off as error/cancelled goes back too; 'done' stays.
    for item in hit:
        _requeue(item["id"])


# ── Worker ────────────────────────────────────────────────────────────────


def lanes() -> int:
    """How many entries run at once: ``queue.lanes``, 1 to MAX_LANES.

    Read when the workers start — a change takes effect with the next
    server start.
    """
    try:
        n = int(config.get("queue", "lanes", default=3) or 1)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(MAX_LANES, n))


def _long_file_sec() -> float:
    try:
        return float(config.get("queue", "long_file_sec", default=3600) or 0)
    except (TypeError, ValueError):
        return 3600.0


def start_worker() -> None:
    """Start the permanent workers, one per lane (idempotent). Called by create_app()."""
    global _worker, _stopping
    # Own lock for start/stop: _cv shares _lock, so a stop_worker() while
    # holding _lock would deadlock.
    with _ctl_lock:
        with _lock:
            alive = [t for t in _workers if t.is_alive()]
            stopping = _stopping
        if alive and not stopping:
            return  # already running
        if alive:
            # Workers are currently shutting down. Wait for them cleanly
            # first, otherwise _stopping=True would remain set and we'd
            # end up without workers.
            _stop_worker_locked()

        n = lanes()
        with _lock:
            _stopping = False
            _workers[:] = [
                threading.Thread(
                    target=_worker_loop,
                    daemon=True,
                    name=(f"queue-worker-{i + 1}" if n > 1 else "queue-worker"),
                )
                for i in range(n)
            ]
            _worker = _workers[0]
            started = list(_workers)
        for t in started:
            t.start()
    _tag_lanes(n)
    logger.info(
        "Queue worker started (runs permanently, independent of the browser; %d %s)",
        n,
        "lane" if n == 1 else "lanes",
    )


def stop_worker(timeout: float = 10.0) -> None:
    """Stop the workers and wait for them to end.

    The wait matters: without join(), an old thread keeps living and would
    snatch up entries in a freshly set-up context (e.g. in the next test).
    """
    with _ctl_lock:
        _stop_worker_locked(timeout)


def _stop_worker_locked(timeout: float = 10.0) -> None:
    global _stopping, _worker
    with _cv:
        _stopping = True
        _cv.notify_all()
        threads = list(_workers)

    # One deadline for all lanes, not one per lane: shutdown() promises
    # its caller ``timeout`` seconds, not lanes × timeout.
    deadline = time.monotonic() + timeout
    for t in threads:
        if t.is_alive() and t is not threading.current_thread():
            t.join(max(0.0, deadline - time.monotonic()))
    with _lock:
        if _workers == threads:
            _workers.clear()
            _worker = None


class _LaneTag(logging.Filter):
    """Pipeline lines carry their lane: with three files in flight, the
    phase banners and closing lines of all three interleave in one log."""

    def filter(self, record):
        name = threading.current_thread().name
        if name.startswith("queue-worker-") and not getattr(record, "_lane", False):
            record.msg = f"[{name.rsplit('-', 1)[1]}] {record.msg}"
            record._lane = True
        return True


_lane_filter = _LaneTag()


#: The loggers a lane writes through. A logger's filter only sees records
#: made on that very logger, not on its children.
_LANE_LOGGERS = (
    "whisper_tm.pipeline",
    __name__,
    "whisper_tm.core.gpu_gate",
    "whisper_tm.core.transcription",
    "whisper_tm.core.diar_process",
    "whisper_tm.core.vocal_isolation",
    "whisper_tm.core.speaker",
    "whisper_tm.jobs",
)


def _tag_lanes(n: int) -> None:
    for name in _LANE_LOGGERS:
        log = logging.getLogger(name)
        if n > 1 and _lane_filter not in log.filters:
            log.addFilter(_lane_filter)
        elif n <= 1 and _lane_filter in log.filters:
            log.removeFilter(_lane_filter)


def _set_current_locked() -> None:
    """``_current`` = the oldest running entry (caller holds _lock)."""
    global _current
    _current = next(iter(_running.values()), None)


def _advance_clock_locked() -> float:
    """Bring the shared clock to now (caller holds _lock); returns it."""
    now = time.monotonic()
    if _running:
        _share_clock["v"] += (now - _share_clock["t"]) / len(_running)
    _share_clock["t"] = now
    return _share_clock["v"]


def _cost_ratio(item: dict) -> float:
    """Share of the wall clock this entry had since its claim, per second
    of its own wall time: 1.0 alone, about 1/3 beside two others."""
    with _lock:
        v = _advance_clock_locked()
    wall = time.monotonic() - item.get("_t0", time.monotonic())
    if wall <= 0:
        return 1.0
    return max(0.0, min(1.0, (v - item.get("_v0", v)) / wall))


def _claim_next(only: set[int] | None = None) -> dict | None:
    """Claim the next pending entry for this lane, and register it —
    with ``only``, one of these entries (the paused queue's _solo).

    One statement claims it: the UPDATE only succeeds while the row is
    still 'pending', and RETURNING hands back exactly the row it changed.
    The SELECT-then-UPDATE before lanes was fine with one worker; with two
    both could read the same row and both run it.

    Passed over, so the order is "position, except":

    * entries whose NAME is already in flight — the job id, the work
      folder (rmtree'd at the end) and the job log hang on the name, and
      the same name under two roots would share all three;
    * entries whose FILE is already in flight under another spelling —
      ``sub/x.mp4`` under /media and ``x.mp4`` under /media/sub are one
      file (docs/PATHS.md); two lanes would write its transcript and
      voices at the same time;
    * other long files while one runs (``queue.long_file_sec``, 0 = off):
      each holds its whole audio in memory, and a 20-hour recording can
      push the diarization worker past 25 GB (core/debug.py).

    An entry the server went down under before (``interrupted``) runs
    alone: nothing starts beside it, and when it is next in line it waits
    for the running ones — should it take the server down again, it is the
    one booked for it (requeue_stale), not its neighbors.
    """
    database = _db()
    if not database:
        return None
    with _claim_lock:
        with _lock:
            if any(r.get("interrupted") for r in _running.values()):
                return None
            busy = bool(_running)
            names = sorted({r["filename"] for r in _running.values()})
            files = sorted({_item_abs_path(r) for r in _running.values()} - {""})
            long_in_flight = any(r.get("_long") for r in _running.values())
        long_sec = _long_file_sec()
        where, params = ["q.status = 'pending'"], []
        if only is not None:
            where.append(f"q.id IN ({','.join('?' * len(only)) or 'NULL'})")
            params += sorted(only)
        if names:
            where.append(f"q.filename NOT IN ({','.join('?' * len(names))})")
            params += names
        if files:
            # The same identity _MEDIA_JOIN uses; '' = the current root
            where.append(
                "(CASE WHEN q.base_dir = '' THEN ? ELSE rtrim(q.base_dir, '/') END"
                f" || '/' || q.filename) NOT IN ({','.join('?' * len(files))})"
            )
            params += [paths.canon(config.media_dir).rstrip("/"), *files]
        join = ""
        if long_in_flight and long_sec > 0:
            join = _MEDIA_JOIN
            where.append("COALESCE(m.duration_sec, 0) < ?")
            params.append(long_sec)
        with database.connection() as conn:
            rows = conn.execute(
                "UPDATE queue_items SET status = 'processing', started_at = ? "
                "WHERE id = (SELECT q.id FROM queue_items q "
                + join
                + " WHERE "
                + " AND ".join(where)
                + " ORDER BY q.position, q.id LIMIT 1) "
                "AND status = 'pending' AND (? OR COALESCE(interrupted, 0) = 0) "
                "RETURNING *",
                (time.time(), *params, 0 if busy else 1),
            ).fetchall()
            if not rows:
                return None
            item = _row_to_item(rows[0])
            # Its place in the run, counted inside the claim (one lane at a
            # time): read later from the snapshot it took in the other
            # lanes' claims too — three lanes starting together logged
            # "(2/11)", "(3/11)", "(3/11)".
            left, all_items = conn.execute(
                "SELECT COALESCE(SUM(status = 'pending'), 0), COUNT(*) FROM queue_items"
            ).fetchone()
            item["_pos"] = (int(all_items) - int(left), int(all_items))
            dur = 0.0
            if long_sec > 0:
                found = conn.execute(
                    "SELECT duration_sec FROM media_files WHERE abs_path = ?",
                    (_item_abs_path(item),),
                ).fetchone()
                dur = float(found[0] or 0) if found else 0.0
        item["status"] = "processing"
        item["_long"] = long_sec > 0 and dur >= long_sec
        with _lock:
            item["_v0"] = _advance_clock_locked()
            item["_t0"] = time.monotonic()
            _running[item["id"]] = item
            _set_current_locked()
            # A stop, pause, abort or shutdown that came while this lane
            # read the pause and ran its UPDATE missed the entry: it wasn't
            # in _running yet, and it ran to the end on a stopped queue.
            # It gets the mark _cancel_running would have
            # given it — an abort cancels, the rest put it back.
            if _aborting and not _shutting_down:
                item["_cancel"] = True
                _requeue_on_abort.discard(item["id"])
            elif _stopping or _shutting_down or (_paused and item["id"] not in _solo):
                item["_cancel"] = True
                _requeue_on_abort.add(item["id"])
    _bump()
    return item


def _finish(item_id: int, status: str, error: str = "") -> None:
    """Book the end of an entry.

    Unguarded on purpose: the lane that ran the entry knows its outcome
    best. Another lane can't hold the same row (_claim_next), and the one
    writer that may have moved it meanwhile — shutdown's safety net putting
    a slow entry back to 'pending' — is wrong when the entry did finish.
    """
    database = _db()
    if not database:
        return
    with database.connection() as conn:
        conn.execute(
            "UPDATE queue_items SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error or "", time.time(), item_id),
        )
    _bump()


def _release(item: dict) -> None:
    """The lane is done with this entry (whatever its outcome)."""
    global _abort
    with _lock:
        _advance_clock_locked()
        _running.pop(item["id"], None)
        _set_current_locked()
        # Even if the job finished cleanly despite a stop/abort request:
        # the tracking set must not keep growing.
        _requeue_on_abort.discard(item["id"])
        ran = item["id"] in _solo
    if ran:
        # It ran — unless it went back to wait (an out-of-memory retry):
        # then it still runs while paused, as it was asked to
        _prune_solo()


def _worker_loop() -> None:
    """Infinite loop of one lane: waits for work, works through it, waits again.

    Runs for the whole server lifetime — not per batch. That's exactly why
    the queue no longer needs an open browser.
    """
    global _abort
    while True:
        with _cv:
            if _stopping:
                return
            if _paused and not _solo:
                _cv.wait(timeout=1.0)
                continue
            # Paused: only what was asked to run anyway (run_now)
            only = set(_solo) if _paused else None

        try:
            item = _claim_next(only)
        except Exception as e:
            # A lane that dies on a locked database leaves the queue with
            # one lane less until the next restart — wait and try again.
            logger.warning("Queue: claiming the next entry failed: %s", e)
            with _cv:
                _cv.wait(timeout=5.0)
            continue
        if item is None:
            if only:
                _prune_solo()  # gone meanwhile: no longer waited for
            # Idle: sleep until enqueue() wakes it (timeout only as a
            # safety net, in case a notify gets lost). The abort flag is
            # spent once NO lane has anything left — another lane may
            # still be unwinding its cancelled entry.
            with _cv:
                if not _running:
                    _abort = False
                _cv.wait(timeout=2.0)
            continue

        try:
            _process(item)
        except BaseException as e:
            # BaseException too: a library's sys.exit() inside a step
            # (audio-separator does that when a model won't load) raised
            # SystemExit, which ended this thread silently — its entry
            # stayed 'processing' and the lane was gone until a restart.
            logger.exception("Queue worker: unexpected error on %s", item["filename"])
            text = str(e) if isinstance(e, Exception) else f"{type(e).__name__}: {e}"
            try:
                # The job too: one left 'processing' counts as running for
                # the idle unloader forever.
                jm = _jm()
                job = jm.get(item.get("job_id")) if (jm and item.get("job_id")) else None
                if job and job.status not in ("done", "error"):
                    jm.fail(job, text)
            except Exception as je:
                logger.debug("Queue: failing the job: %s", je)
            try:
                _finish(item["id"], "error", text)
            except Exception as fe:
                logger.warning(
                    "Queue: booking the error failed (%s) — the "
                    "entry is requeued at the next start",
                    fe,
                )
            qlog(f"❌ Exception: {item['filename']}: {text}")
        finally:
            _release(item)


def _batch_value(v) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _normalize(settings: dict) -> dict:
    """Map raw settings from the UI onto valid worker arguments.

    If a value is missing, the default configured under settings applies —
    not one hardcoded here. Otherwise the fields under
    Settings → Transcription would have no effect.
    """
    # "no transcription" from the dialog: 'none' is NOT a Whisper model.
    # Without this branch it would fall through to the unknown fallback
    # below and the run would transcribe anyway — exactly what the user
    # deselected.
    separation_only = str(settings.get("model") or "").strip().lower() == "none"
    # Job kind "voices": diarization + voice vectors for a file whose
    # transcript already exists (transcription.transcription_worker).
    voices_only = str(settings.get("job") or "").strip().lower() == "voices"

    model_default = config.get("transcription", "default_model", default="turbo") or "turbo"
    model_name = settings.get("model", model_default) or model_default
    if separation_only:
        model_name = model_default  # unused, but never 'none' further down
    # '/' allows HuggingFace models; anything else unknown → config default
    if model_name not in VALID_WHISPER_MODELS and "/" not in model_name:
        model_name = model_default
    if model_name not in VALID_WHISPER_MODELS and "/" not in model_name:
        model_name = "turbo"  # unusable config value → hard fallback

    num_sp = settings.get(
        "numSpeakers", config.get("transcription", "default_num_speakers", default=0)
    )
    min_sp = max_sp = None
    if num_sp and str(num_sp).isdigit() and int(num_sp) > 0:
        min_sp = max_sp = int(num_sp)

    models = config.get("vocal_isolation", "models", default={})
    default_model = config.get("vocal_isolation", "default_model", default="bs_roformer")
    vocal_model = settings.get("vocal_model", default_model) or default_model
    if vocal_model not in models:
        vocal_model = default_model

    lang_default = config.get("transcription", "default_language", default="auto") or "auto"

    opts = {
        "language": settings.get("language", lang_default) or lang_default,
        "model_name": model_name,
        "min_speakers": min_sp,
        "max_speakers": max_sp,
        # parse_bool instead of bool(): "false"/"0"/"no" would be True with bool()
        "diarize": parse_bool(
            settings.get("diarize", config.get("transcription", "default_diarize", default=True))
        ),
        "separation_only": separation_only,
        "voices_only": voices_only,
        # For a pure separation run both are non-negotiable: without
        # isolation there would be nothing to do, and without saving
        # alongside the source no result would remain from the run (the
        # old separation dialog with its own download folder is gone).
        "vocal_isolation": True
        if separation_only
        else parse_bool(
            settings.get(
                "vocal_isolation", config.get("vocal_isolation", "default_enabled", default=True)
            )
        ),
        "vocal_model": vocal_model,
        "save_audio_alongside": True
        if separation_only
        else parse_bool(
            settings.get(
                "save_audio_alongside",
                config.get("vocal_isolation", "default_save_alongside", default=False),
            )
        ),
        # Speaker vectors: the worker itself checks whether diarization is
        # happening — without speaker segments there's nothing to slice.
        # Only the request is passed through here, not the precondition.
        "speaker_vectors": parse_bool(
            settings.get(
                "speaker_vectors", config.get("transcription", "speaker_vectors", default=True)
            )
        ),
        # None = use the config default in the worker
        "word_timestamps": settings.get("word_timestamps"),
        "export_alongside": settings.get("export_alongside"),
        "low_vram": (
            None if settings.get("low_vram") is None else parse_bool(settings.get("low_vram"))
        ),
        # WhisperX batch: None = hardware recommendation; the OOM path halves it
        "batch_size": _batch_value(settings.get("batch_size")),
        # Preset NAME (not the text): resolved only in the worker — presets
        # are thus changeable without touching pending entries.
        "vocab_preset": (settings.get("vocab_preset") or "").strip() or None,
    }
    if voices_only:
        # Voices are only comparable when made the way a transcription
        # makes them: separated audio, diarized, vectors on. Nothing else
        # of the dialog applies — there is no Whisper run.
        opts.update(
            diarize=True,
            speaker_vectors=True,
            vocal_isolation=True,
            save_audio_alongside=False,
            # Renew existing voices, their decisions carried over
            refresh_voices=parse_bool(settings.get("refresh", False)),
        )
    return opts


def _queue_summary(filename: str, settings: dict, vdir, result=None) -> None:
    """The dialog's summary of a finished transcript, handed to the LLM queue
    (core/llm_queue.py).

    It used to run in the lane itself, one at a time, before the entry
    counted as done: while the language model was busy or out of reach — a
    server on another machine that didn't answer for half an hour — every
    lane waited behind it and no file was transcribed. The LLM queue runs
    one task at a time too, waits for an unreachable model without holding
    a lane, and keeps the task over a restart. A failure there is not the
    file's: the transcript stays done, and the summary can be made any time
    via the 🤖 button. That's why everything here is caught and only
    reported.
    """
    preset = (settings.get("summarize_preset") or "").strip()
    prompt = (settings.get("summarize_prompt") or "").strip()
    if not config.get("llm", "endpoint", default=""):
        qlog(f"⚠ No summary of {filename}: no language model is set up (Settings → AI models)")
        return
    # A recording without speech has nothing to summarize — queued, it
    # failed in the LLM queue as "Transcript not found"
    segments = (result or {}).get("segments") if isinstance(result, dict) else None
    if isinstance(segments, list) and not any(
        (seg.get("text") or "").strip() for seg in segments if isinstance(seg, dict)
    ):
        qlog(f"No summary of {filename}: the transcript is empty")
        return
    # What make_summary saved them as: the template, or "Custom"
    params = {
        "media_dir": str(vdir) if vdir else "",
        "auto": True,
        "label": "Custom" if prompt else preset or "Automatic",
    }
    for key, to in (
        ("summarize_preset", "preset"),
        ("summarize_model", "model"),
        ("summarize_language", "language"),
    ):
        value = (settings.get(key) or "").strip()
        if value:
            params[to] = value
    if prompt:
        params["system_prompt"] = prompt  # instead of the template's
    try:
        from whisper_tm.core.llm_queue import llm_queue

        if not llm_queue.enqueue("summary", filename, params):
            raise RuntimeError("the LLM queue is not available")
        qlog(
            f"🤖 Summary queued: {filename}"
            + (" (Custom)" if prompt else f" ({preset})" if preset else "")
        )
    except Exception as e:
        logger.warning("Auto-summary for %s not queued: %s", filename, e)
        qlog(f"⚠ Summary not queued: {filename} — {e}")


class _KeptTranscript:
    """What a re-transcription replaces — kept until the new run is done.

    Previously ``retranscribe`` deleted the old transcript (JSON, database
    row) and the file's voice vectors before the file was even looked for:
    a share that was briefly gone, a failed, cancelled or out-of-memory run
    — each ended with the old transcript and its named voices gone and
    nothing in their place. Now nothing is deleted up front.
    The run overwrites the transcript only at its very end, and the voices
    when it computes new ones; this keeps a copy of both:

    * ``settle(done=True)``: the new transcript stands. Old voices the run
      did not replace go, as before — they belong to the old transcript.
    * ``settle(done=False)``: whatever the run changed is put back — the
      JSON byte for byte, the database row, the voices with their names,
      trash, groups and history (every row that hangs on a vector id).
    """

    def __init__(self, filename: str, vdir, source: Path):
        self.filename = filename
        self.vdir = str(vdir)
        self.abs_path = paths.canon(source)
        self.json_path: Path | None = None
        self.json_bytes: bytes | None = None
        self.db_row: dict | None = None
        self.vectors: list[dict] = []
        self.children: dict[str, list[dict]] = {}

    @classmethod
    def take(cls, filename: str, vdir, source: Path) -> _KeptTranscript:
        kept = cls(filename, vdir, source)
        try:
            tp = transcript_path_for(filename, abs_path=source)
            if tp.is_file():
                kept.json_path, kept.json_bytes = tp, tp.read_bytes()
        except (OSError, ValueError) as e:
            logger.warning(
                "Re-transcription: the old transcript of %s is not readable (%s)", filename, e
            )
        database = _db()
        if database and kept.abs_path:
            with database.connection() as conn:
                row = conn.execute(
                    "SELECT rel_path, media_dir, json_data FROM transcripts "
                    "WHERE abs_path = ? LIMIT 1",
                    (kept.abs_path,),
                ).fetchone()
            kept.db_row = dict(row) if row else None
        kept._take_vectors()
        return kept

    # ── Voices ────────────────────────────────────────────────────────

    @staticmethod
    def _speaker_db():
        from whisper_tm.database.speaker import speaker_db

        sdb = speaker_db()
        sdb.initialize()
        return sdb

    @staticmethod
    def _child_tables(c) -> list[str]:
        """Every table with a column that references a vector id."""
        out = []
        for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
            for fk in c.execute(f'PRAGMA foreign_key_list("{name}")'):
                if fk["table"] == "speaker_vectors" and fk["from"] == "vector_id":
                    out.append(name)
                    break
        return out

    def _take_vectors(self) -> None:
        if not self.abs_path:
            return
        try:
            sdb = self._speaker_db()
            with sdb.connection() as c:
                self.vectors = [
                    dict(r)
                    for r in c.execute(
                        "SELECT * FROM speaker_vectors WHERE abs_path = ?", (self.abs_path,)
                    )
                ]
                ids = [v["id"] for v in self.vectors]
                if ids:
                    for t in self._child_tables(c):
                        self.children[t] = [
                            dict(r)
                            for r in c.execute(
                                f'SELECT * FROM "{t}" WHERE vector_id IN '
                                f"(SELECT value FROM json_each(?))",
                                (json.dumps(ids),),
                            )
                        ]
        except Exception as e:  # noqa — then the voices simply aren't touched
            logger.warning(
                "Re-transcription: the voices of %s could not be kept (%s)", self.filename, e
            )
            self.vectors = []

    @staticmethod
    def _key(row: dict) -> tuple:
        """A row by its whole content: ids are reused once rows are deleted
        (INTEGER PRIMARY KEY), so a new voice can carry an old one's id."""
        return tuple(sorted(row.items()))

    def _vectors_now(self, c) -> list[dict]:
        return [
            dict(r)
            for r in c.execute("SELECT * FROM speaker_vectors WHERE abs_path = ?", (self.abs_path,))
        ]

    def _restore_vectors(self) -> None:
        sdb = self._speaker_db()
        with sdb.connection(immediate=True) as c:
            if sorted(map(self._key, self._vectors_now(c))) == sorted(map(self._key, self.vectors)):
                return  # the run never touched them
            # The run's own vectors go (only THIS file's rows; theirs
            # cascade), the old ones come back — with their ids where those
            # are still free, so every name, trash mark, group and history
            # entry that hangs on an id holds again. Deleted rows free their
            # ids (plain INTEGER PRIMARY KEY), and meanwhile another lane may
            # have stored another file's voices under them: such an old row
            # gets a new id and its rows follow it. Never a REPLACE — that
            # took the other file's voices, names and all.
            c.execute("DELETE FROM speaker_vectors WHERE abs_path = ?", (self.abs_path,))
            new_id: dict[int, int] = {}
            for row in self.vectors:
                taken = c.execute(
                    "SELECT 1 FROM speaker_vectors WHERE id = ?", (row["id"],)
                ).fetchone()
                cols = [k for k in row if not (taken and k == "id")]
                cur = c.execute(
                    f"INSERT INTO speaker_vectors ({', '.join(cols)}) "
                    f"VALUES ({', '.join('?' * len(cols))})",
                    [row[k] for k in cols],
                )
                new_id[row["id"]] = cur.lastrowid if taken else row["id"]
            for table, rows in self.children.items():
                for row in rows:
                    row = {**row, "vector_id": new_id.get(row["vector_id"], row["vector_id"])}
                    cols = list(row)
                    try:
                        c.execute(
                            f'INSERT INTO "{table}" ({", ".join(cols)}) '
                            f"VALUES ({', '.join('?' * len(cols))})",
                            [row[k] for k in cols],
                        )
                    except sqlite3.IntegrityError as e:
                        # Its other end (a grouping run, a decision) went
                        # meanwhile — the voice is back without it
                        logger.debug("Re-transcription: %s row not restored: %s", table, e)
        sdb._changed(inventory=True, trash=True)

    def _drop_old_vectors(self) -> None:
        """The new transcript stands: old voices it did not replace go."""
        if not self.vectors:
            return
        kept = {self._key(v) for v in self.vectors}
        sdb = self._speaker_db()
        with sdb.connection(immediate=True) as c:
            stale = [v["id"] for v in self._vectors_now(c) if self._key(v) in kept]
            gone = (
                c.execute(
                    "DELETE FROM speaker_vectors WHERE id IN (SELECT value FROM json_each(?))",
                    (json.dumps(stale),),
                ).rowcount
                if stale
                else 0
            )
        if gone:
            sdb._changed(inventory=True)
            logger.info("Re-transcription: %d old voice windows of %s removed", gone, self.filename)

    # ── Transcript ────────────────────────────────────────────────────

    def _restore_transcript(self) -> None:
        from whisper_tm.core.helpers import invalidate_complete_cache

        if self.json_path is not None and self.json_bytes is not None:
            try:
                now = self.json_path.read_bytes() if self.json_path.is_file() else None
            except OSError:
                now = None
            if now != self.json_bytes:
                tmp = self.json_path.with_name(f".wtm-kept-{self.json_path.name[-40:]}.tmp")
                tmp.write_bytes(self.json_bytes)
                os.replace(tmp, self.json_path)
        database = _db()
        if database and self.abs_path:
            with database.connection() as conn:
                row = conn.execute(
                    "SELECT json_data FROM transcripts WHERE abs_path = ? LIMIT 1", (self.abs_path,)
                ).fetchone()
            now_json = row["json_data"] if row else None
            old = self.db_row
            if old is not None and old.get("json_data") and now_json != old["json_data"]:
                database.upsert_transcript(
                    old["rel_path"] or self.filename,
                    json.loads(old["json_data"]),
                    media_dir=old["media_dir"] or self.vdir,
                    abs_path=self.abs_path,
                )
            elif old is None and self.json_bytes is not None and row is not None:
                # Only the JSON knew the old transcript: the database follows it
                database.upsert_transcript(
                    self.filename,
                    json.loads(self.json_bytes),
                    media_dir=self.vdir,
                    abs_path=self.abs_path,
                )
        invalidate_complete_cache(self.filename, media_dir=self.vdir)

    def settle(self, done: bool) -> None:
        """Keep the new (``done``) or put the old back — never raises: the
        entry's outcome is booked either way, and a failure here is logged
        loudly instead."""
        steps = (
            (self._drop_old_vectors,) if done else (self._restore_transcript, self._restore_vectors)
        )
        for step in steps:
            try:
                step()
            except Exception:
                logger.exception("Re-transcription of %s: %s failed", self.filename, step.__name__)
        if not done:
            qlog(
                f"↩ Re-transcription did not finish — the old transcript and voices "
                f"of {self.filename} stay"
            )


def _already_handled(item: dict, filename: str, vdir, opts: dict, settings: dict) -> bool:
    """The checks before a run: True when the entry is finished without one
    (skipped). Asked again after a path repair, with the repaired name.

    Pure separation run: none of this concerns it. It must not be skipped
    as "transcript present" — an already-transcribed file is the normal
    case for separation. A re-transcription is never skipped either; its
    old transcript stays until the new one is done (_KeptTranscript).
    """
    if opts.get("separation_only"):
        return False
    if opts.get("voices_only"):
        # Never deletes the transcript. Skipped when the file already has
        # voices: storing them anew replaces the rows, and the names given
        # to the old ones hang on their ids — unless the entry renews them,
        # which carries the names over (SpeakerDB.replace_keeping_decisions).
        from whisper_tm.core import speaker as _speaker
        from whisper_tm.database.speaker import speaker_db

        if not opts.get("refresh_voices") and speaker_db().has_vectors(
            paths.canon(Path(vdir) / filename), _speaker.MODEL_BRAND
        ):
            _finish(item["id"], "skipped")
            qlog(f"Skipped (voices present): {filename}")
            return True
        # Voices belong to a transcribed file. Without a transcript there is
        # no speaker to align the labels to — and a later transcription with
        # diarization would replace these voices, names given to them
        # included, without the warning a re-transcription shows.
        if not is_transcript_complete(filename, media_dir=vdir):
            _finish(item["id"], "skipped")
            qlog(f"Skipped (no transcript — voices only work on transcribed files): {filename}")
            return True
        return False
    if parse_bool(settings.get("retranscribe", False)):
        return False
    if is_transcript_complete(filename, media_dir=vdir):
        _finish(item["id"], "skipped")
        qlog(f"Skipped (transcript present): {filename}")
        return True
    return False


def _process(item: dict) -> None:
    """Process one queue entry (blocks until done).

    A re-transcription's old transcript and voices are settled here,
    whatever way the run ends — a raised error included (_KeptTranscript).
    """
    box: dict = {}
    try:
        _process_entry(item, box)
    finally:
        kept = box.get("kept")
        if kept is not None:
            kept.settle(done=bool(box.get("done")))


def _process_entry(item: dict, box: dict) -> None:
    """The run of one entry; ``box`` gets what _process settles after it."""
    from whisper_tm.core.transcription import transcription_worker

    filename = item["filename"]
    settings = item["settings"]
    opts = _normalize(settings)
    database = _db()
    jm = _jm()

    # Determine THIS entry's root first — the checks below need it. If
    # they came before it, they ran against config.media_dir: for an entry
    # from a different root, a re-transcription would replace the transcript
    # of the same-named file under the CURRENT root, and the completeness
    # check would mark the job as "already transcribed". The queue is
    # persistent — exactly this case is the normal case (docs/PATHS.md
    # §writing).
    vdir = Path(item["dir"]) if item["dir"] else config.media_dir

    if _already_handled(item, filename, vdir, opts, settings):
        return

    # Second line of defense behind enqueue(): abs_from() builds the path
    # normalized and refuses to leave the entry's root — unlike
    # `vdir / filename` (docs/PATHS.md). That defeats traversal.
    target = paths.abs_from(filename, str(vdir))
    if not target:
        _finish(item["id"], "error", "Path not allowed")
        qlog(f"❌ Path leaves the entry's root: {filename}")
        return
    if not root_allowed(target):
        if item["dir"]:
            # THIS entry's root was checked and allowed at queuing time —
            # after a root change plus restart, it simply no longer shows
            # up in allowed_roots() (the anchor is the start root). The
            # queue is persistent and deliberately keeps running (otherwise
            # a path change made all entries die with "path not allowed").
            # Note it once per root.
            if item["dir"] not in _foreign_roots_reported:
                _foreign_roots_reported.add(item["dir"])
                qlog(f"ℹ Entries from an earlier root keep being processed: {item['dir']}")
        else:
            # dir='' means "current root" — if even that one isn't allowed,
            # the configuration is broken; that must not slip through.
            _finish(item["id"], "error", "Path not allowed")
            qlog(f"❌ Path outside the allowed directories: {filename}")
            return
    source = Path(target)

    # Check the skip list via the identity, using THIS entry's root. Via
    # the bare filename, a skipped same-named file from a different folder
    # locked this one out too — permanently and without notice
    # (docs/PATHS.md). This sits behind the vdir resolution because the
    # root is needed for the identity.
    if is_skipped(filename, media_dir=vdir):
        _finish(item["id"], "skipped")
        qlog(f"Skip list: {filename}")
        return

    if not source.exists():
        # Before the entry dies as "not found": the case where ROOT AND
        # FILENAME OVERLAP EACH OTHER.
        #
        # A real-world case — hundreds of files in one go:
        #
        #   filename : videos/Archive/Videos/meeting_2024-03-01.mp4   (from the media root)
        #   base_dir : /mnt/media/videos/Archive/Videos               (subfolder)
        #   yields   : /mnt/media/videos/Archive/Videos/
        #              videos/Archive/Videos/meeting_2024-03-01.mp4   ← duplicated
        #
        # This happens because the frontend sends the currently DISPLAYED
        # directory as `base_dir`, but the filenames are relative to the
        # media root (batch.js: `base_dir: dirInput.value`). If the two
        # diverge, EVERY entry of that batch is broken — and silently so:
        # hundreds of files that existed landed in the error bucket as "not found",
        # while they sat untouched on disk.
        #
        # The fix is narrowly scoped: ONLY the media root is tried as an
        # alternative, it must be allowed, and the file must actually be
        # there. This keeps the traversal protection of abs_from() intact
        # — nothing is guessed.
        candidates = []
        # 1. Shorten the overlap itself — the actual case.
        short = _trim_overlap(filename, str(vdir))
        if short:
            candidates.append((paths.abs_from(short, str(vdir)), vdir, short))
        # 2. The filename is relative to the media root, base_dir is a
        #    different folder. Only meaningful if the root is actually
        #    different — if the view is currently IN the subfolder, both
        #    are the same and this attempt would be the same mistake.
        candidates.append(
            (paths.abs_from(filename, str(config.media_dir)), config.media_dir, filename)
        )

        for cand, new_root, new_name in candidates:
            if cand and cand != target and root_allowed(cand) and Path(cand).exists():
                qlog(f"↳ Root and filename overlapped — found: {filename}")
                logger.info(
                    "Queue: path repaired (base_dir=%s, filename=%s) → %s",
                    item["dir"],
                    filename,
                    cand,
                )
                # The name goes with the root: every check and write
                # below (transcript present, the incomplete-transcript
                # cleanup, the job, the database) builds the identity from
                # the two. With the old name the overlap case asked
                # "transcript present?" of a path that doesn't exist — and
                # deleted the real file's COMPLETE transcript as
                # "incomplete".
                vdir = Path(new_root)
                filename = new_name
                target = cand
                source = Path(target)
                break
        else:
            # A whole library gone (a share not mounted) is no error of
            # the file: every entry of it would die as "not found"
            if _library_gone(vdir):
                _environment_pause(
                    item,
                    f"{filename} not found",
                    cause=(
                        f"The library folder {vdir} is gone or empty (a network share not mounted?)"
                    ),
                )
                return
            _finish(item["id"], "error", "Not found")
            qlog(f"❌ Not found: {filename}")
            return
        # The checks of the start, for the file actually found
        if _already_handled(item, filename, vdir, opts, settings):
            return

    # The work folder's disk (nearly) full: every job would fail on it, one
    # after another — paused before the first
    short = _work_space_short()
    if short:
        _environment_pause(item, short, cause="The disk of the work folder is (nearly) full")
        return

    # Remove an incomplete transcript JSON. The path is tied to the
    # identity and can therefore never hit an unrelated file (docs/PATHS.md).
    tp = transcript_path_for(filename, abs_path=source)
    if tp.exists() and not is_transcript_complete(filename, media_dir=vdir):
        tp.unlink()

    # Re-transcription: the old transcript and voices are kept until the new
    # run is done, and put back on any other outcome (_KeptTranscript). Only
    # now — the file has been found under its real name.
    if (
        parse_bool(settings.get("retranscribe", False))
        and not opts.get("separation_only")
        and not opts.get("voices_only")
    ):
        box["kept"] = _KeptTranscript.take(filename, vdir, source)

    # Job ID collision-free: a plain replacement of special characters would
    # map "Foo Bar.mp4" and "Foo.Bar.mp4" onto the same ID.
    import hashlib

    fhash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
    jid = filename.replace(" ", "_").replace(".", "_").replace("/", "_") + f"_{fhash}"

    # Keep a reference to the job object and do NOT fetch it again later via
    # jm.get(jid): the JobManager cleanup discards 'processing' jobs after a
    # few hours. For a long transcription (large-v3 + vocal isolation on a
    # multi-hour recording), get() would then return None — and the entry
    # would be wrongly marked "aborted" even though it ran through cleanly.
    job = jm.create(jid, filename=filename, job_type="transcription")
    # The dialog's summary follows the transcript (below). Without a
    # transcript there's nothing to summarize — the checkbox might still be
    # set from an earlier dialog state.
    wants_summary = (
        parse_bool(settings.get("summarize", False))
        and not opts.get("separation_only")
        and not opts.get("voices_only")
    )
    if job:
        # What the page following this job needs to know: an OOM is not the
        # end of the file (_oom_retry takes it up again)
        job.retry_if = _is_oom_error
    if database:
        try:
            database.log_job(
                jid,
                filename,
                "transcription",
                model=opts["model_name"],
                language=opts["language"],
                settings={
                    "vocal_isolation": opts["vocal_isolation"],
                    "vocal_model": opts["vocal_model"],
                    "diarize": opts["diarize"],
                    "queued": True,
                },
            )
            with database.connection() as conn:
                conn.execute("UPDATE queue_items SET job_id = ? WHERE id = ?", (jid, item["id"]))
            _bump()
        except Exception as e:
            logger.debug("Queue db.log_job: %s", e)

    # `item` is the very dict _running holds. A stop or cancel that came
    # before the job existed left its mark there — the job takes it over,
    # and transcription_worker stops at its first check.
    with _lock:
        item["job_id"] = jid
        cancel_early = bool(item.get("_cancel"))
    if cancel_early and job:
        job.cancel = True

    pos, total = item.get("_pos") or (0, 0)
    qlog(f"▶ Starting: {filename} ({pos}/{total})")

    from whisper_tm.core import debug
    from whisper_tm.core.helpers import format_size

    debug.banner(
        f"QUEUE {pos}/{total}",
        filename,
        Directory=str(vdir),
        Size=format_size(source.stat().st_size),
    )
    if debug.is_enabled():
        logger.debug("Queue entry %s | options: %s", item["id"], opts)

    # A low-VRAM run unloads every model before it starts and keeps none
    # — next to another lane it would pull the models out from under it.
    # It runs alone: the other lanes finish their current step and wait.
    low = opts.get("low_vram")
    if not isinstance(low, bool):
        from whisper_tm.core.vram import default_low_vram

        low = default_low_vram()
    # The run keeps this choice to its end: read live, a "Spare VRAM"
    # switched on in the settings mid-run made a lane that runs WITHOUT
    # the exclusive unload the separator and kill the diarization worker
    # another lane was waiting on.
    opts = {**opts, "low_vram": low}
    alone = GATE.exclusive("low-VRAM run", job=job) if low else contextlib.nullcontext()
    try:
        with debug.timed("queue: total transcription", file=filename), alone:
            transcription_worker(jid, source, filename, **opts)
    except InterruptedError:
        # Stopped or cancelled while it waited to have the GPU to itself:
        # the run never began. Booked below like any other cancel — and the
        # job ends too: left 'queued' it counted as running for the idle
        # unloader forever.
        if job:
            job.cancel = True
            jm.fail(job, "Cancelled")

    if job and database:
        # Write the transcript to the DB with the ACTUAL source directory:
        # the worker only knows config.media_dir, which would be wrong for
        # files from a deviating directory (dir_map).
        if job.status == "done" and opts.get("voices_only"):
            # The transcript was not touched and must not be now: the
            # re-read below would rewrite it from the JSON and point its
            # log link at this voices run. The voices keep their own link.
            try:
                voices = dict((job.result or {}).get("voices") or {})
                if voices and job.log_file:
                    voices["log_file"] = job.log_file
                    database.merge_transcript_settings(str(source), {"voices": voices})
            except Exception as e:
                logger.debug("Queue voices log link: %s", e)
        elif job.status == "done" and not opts.get("separation_only"):
            try:
                tp2 = transcript_path_for(filename, abs_path=source)
                if tp2.exists():
                    with open(tp2, encoding="utf-8") as f:
                        data = json.load(f)
                    if job.log_file:
                        data["log_file"] = job.log_file
                    # The database keeps the run's COST: with lanes the
                    # file shared the card, and its wall time is latency —
                    # the speed figures, the library's time left and the
                    # tempo filter read this column as compute time. Alone
                    # (one lane) the ratio is 1 and nothing changes; the
                    # transcript file keeps the wall time it measured.
                    try:
                        if data.get("processing_secs"):
                            data = {
                                **data,
                                "processing_secs": round(
                                    float(data["processing_secs"]) * _cost_ratio(item), 2
                                ),
                            }
                    except (TypeError, ValueError):
                        pass
                    # Pass abs_path ALONG. The worker in transcription.py
                    # gets it right directly before this (abs_path=source_path);
                    # this second upsert overwrote the same row without it
                    # and thus fell back to reconstructing from rel_path +
                    # root — exactly what docs/PATHS.md §writing rules out.
                    # `source` is the actual location.
                    database.upsert_transcript(
                        filename, data, media_dir=str(vdir), abs_path=str(source)
                    )
            except Exception as e:
                logger.warning("Queue upsert_transcript failed (transcript NOT in the DB): %s", e)
        try:
            database.update_job(
                jid, status=job.status, processing_sec=job.elapsed(), error_message=job.error or ""
            )
        except Exception as e:
            logger.debug("Queue update_job: %s", e)

    if job and job.status == "done":
        box["done"] = True
        _finish(item["id"], "done")
        proc = (job.result or {}).get("processing_time", "")
        qlog(f"✅ Done: {filename} ({proc})")
        # The summary follows in the LLM queue — the lane goes on
        if wants_summary:
            _queue_summary(filename, settings, vdir, job.result)
    elif job and job.status == "error" and not job.cancel:
        err = job.error or "Unknown"
        if _oom_retry(item, err, job):
            pass  # is back in the queue
        elif _environment_pause(item, err, job):
            pass  # back in the queue, the queue paused
        else:
            _finish(item["id"], "error", err)
            qlog(f"❌ Error: {filename}: {err}")
    else:
        # Abort. Detected via job.cancel, NOT via the status: the worker
        # reports an abort via job_manager.fail() — i.e. as 'error' with
        # the text "Cancelled". Whoever only checks the status counts
        # every abort as a failure (and never lands in this branch).
        # Without a job (JobManager cleanup was faster) also here: the
        # entry is then neither done nor demonstrably failed.
        with _lock:
            shutting = _shutting_down
            requeue_wanted = item["id"] in _requeue_on_abort
            _requeue_on_abort.discard(item["id"])
        if shutting or requeue_wanted:
            # Shutdown or "stop" in the web UI: not an error, but "not yet
            # finished" — back into the queue.
            _requeue(item["id"])
            reason = "Shutdown" if shutting else "Stopped"
            qlog(f"↩ Back in the queue ({reason}): {filename}")
        else:
            # Deliberate abort — targeted (/api/cancel/<job_id>) or as the
            # running job during a total abort (abort()). Not an error,
            # otherwise every deliberate abort would count as a failure in
            # the stats — but a visible 'cancelled' history and re-queueable
            # via retry.
            _finish(item["id"], "cancelled", "Cancelled")
            qlog(f"⚠ Aborted: {filename}")

"""Taking Whisper artifacts out of the transcripts that exist.

The artifact filter (``core/helpers.py``) works while a file is transcribed.
This applies the same rule — the patterns of
``transcription.hallucination_patterns``, and "no letter or digit" — to
transcripts already in the library, in two steps: a **dry run** that only
reads and counts, then **applying** what that dry run found. A third step
takes a cleanup back.

This changes the user's transcripts, so every step is built to lose nothing:

* The dry run only reads the database.
* Applying backs the database up first (``database.backup_path``) and does
  not start without that backup.
* Every segment it removes goes into a log BEFORE its transcript is written
  (write-ahead), with its position: "undo" puts it back. The log lives in
  the data folder, which is never emptied.
* A transcript with a translation or a correction run is left alone: both
  address segments by their position, and a removed segment would shift
  every one behind it.
* Each transcript is read again right before it is written and judged on
  its text of that moment — what changed since the dry run counts as it is
  now.
* One transcript at a time, each in its own transaction; the JSON file next
  to the media is replaced atomically. A stop, or a server restart, leaves
  every transcript either untouched or done.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core.helpers import (
    artifact_reason,
    atomic_write_json,
    compile_artifact_patterns,
    transcript_path_for,
)

logger = logging.getLogger(__name__)

#: How many of the most frequent texts, and of the most affected files, a
#: dry run lists — its counts cover everything
LIST_TEXTS = 30
LIST_FILES = 100
LIST_SKIPPED = 50

#: Rows per database read while scanning (about 10 million segments take
#: ~15 s)
_BATCH = 20000

_lock = threading.Lock()
_state: dict = {
    "running": False,
    "kind": "",
    "phase": "",
    "done": 0,
    "total": 0,
    "started_at": 0.0,
    "error": "",
    "stop": None,
}
_last: dict = {"dry_run": None, "applied": None, "undone": None}
#: The latest dry run's targets: run id → patterns and transcript ids
_runs: dict = {}


class Stopped(Exception):
    """The user pressed stop."""


def log_dir() -> Path:
    """Where the cleanup logs are kept — the data folder, never emptied."""
    return config.data_dir / "cleanups"


def _db():
    from whisper_tm.extensions import db

    if not db:
        raise RuntimeError("No database")
    return db


def _progress(phase: str, done: int, total: int) -> None:
    with _lock:
        _state.update(phase=phase, done=done, total=total)


def status() -> dict:
    """What runs now, how far it is, and the latest results."""
    with _lock:
        s = {k: v for k, v in _state.items() if k != "stop"}
        s.update(dry_run=_last["dry_run"], applied=_last["applied"], undone=_last["undone"])
    s["logs"] = list_logs()
    return s


def stop() -> bool:
    with _lock:
        ev = _state.get("stop")
    if ev:
        ev.set()
        return True
    return False


def start(kind: str, fn, *args) -> bool:
    """Run one job in the background; only one at a time."""
    with _lock:
        if _state["running"]:
            return False
        ev = threading.Event()
        _state.update(
            running=True,
            kind=kind,
            phase="starting",
            done=0,
            total=0,
            started_at=time.time(),
            error="",
            stop=ev,
        )

    def run():
        try:
            fn(*args, stop_event=ev)
        except Stopped:
            with _lock:
                _state["error"] = "stopped"
        except Exception as e:  # noqa: BLE001
            logger.exception("Artifact cleanup (%s) failed", kind)
            with _lock:
                _state["error"] = str(e)
        finally:
            with _lock:
                _state.update(running=False, phase="", stop=None)

    threading.Thread(target=run, name=f"artifacts-{kind}", daemon=True).start()
    return True


# ── Which transcripts must not be touched ────────────────────────────────


def _held(conn) -> dict[str, str]:
    """Transcripts whose segments are addressed by position elsewhere —
    by abs_path, or by rel_path for rows from before abs_path existed.

    A translation track stores ``[{id, text}]`` with id = segment position;
    a correction run stores its suggestions and, once applied, the undo
    history the same way. A rejected run changed nothing and holds nothing.
    """
    held: dict[str, str] = {}
    for ab, rel in conn.execute("SELECT abs_path, rel_path FROM translations"):
        for key in (ab, rel):
            if key:
                held.setdefault(key, "has a translation")
    for ab, rel in conn.execute(
        "SELECT abs_path, rel_path FROM llm_edits WHERE status != 'rejected'"
    ):
        for key in (ab, rel):
            if key:
                held.setdefault(key, "has a correction run")
    return held


def _why_held(held: dict, abs_path: str, rel_path: str) -> str:
    return held.get(abs_path or "") or held.get(rel_path or "") or ""


# ── Dry run ───────────────────────────────────────────────────────────────


def dry_run(patterns: list[str], stop_event: threading.Event) -> dict:
    """Count what the patterns would take out — reads only."""
    compiled, errors = compile_artifact_patterns(patterns)
    if errors:
        raise ValueError("not a valid regular expression — " + "; ".join(errors))
    db = _db()
    run_id = uuid.uuid4().hex[:12]
    reasons: Counter = Counter()
    texts: Counter = Counter()
    per_file: Counter = Counter()
    skipped: Counter = Counter()
    with db.connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        _progress("reading the transcripts", 0, total)
        files = {
            r[0]: (r[1] or "", r[2] or "")
            for r in conn.execute(
                "SELECT id, rel_path, abs_path FROM transcripts WHERE complete = 1"
            )
        }
        held = _held(conn)
        seen = 0
        cur = conn.execute("SELECT transcript_id, text FROM segments")
        while True:
            rows = cur.fetchmany(_BATCH)
            if not rows:
                break
            for tid, text in rows:
                if tid not in files:
                    continue  # not a finished transcript
                why = artifact_reason(text or "", compiled)
                if why is None:
                    continue
                rel, ab = files[tid]
                if _why_held(held, ab, rel):
                    skipped[tid] += 1
                    continue
                reasons[why] += 1
                texts[(text or "").strip()] += 1
                per_file[tid] += 1
            seen += len(rows)
            _progress("reading the transcripts", seen, total)
            if stop_event.is_set():
                raise Stopped()

    def name(tid):
        rel, ab = files[tid]
        return rel or ab

    result = {
        "run_id": run_id,
        "patterns": [str(p) for p in patterns if str(p).strip()],
        "at": time.time(),
        "scanned": seen,
        "segments": sum(reasons.values()),
        "transcripts": len(per_file),
        "reasons": [[why, n] for why, n in reasons.most_common()],
        "texts": [[t, n] for t, n in texts.most_common(LIST_TEXTS)],
        "files": [[name(tid), n] for tid, n in per_file.most_common(LIST_FILES)],
        "skipped": {
            "transcripts": len(skipped),
            "segments": sum(skipped.values()),
            "files": [
                [name(tid), n, _why_held(held, files[tid][1], files[tid][0])]
                for tid, n in skipped.most_common(LIST_SKIPPED)
            ],
        },
    }
    with _lock:
        _runs.clear()
        _runs[run_id] = {"patterns": result["patterns"], "targets": list(per_file)}
        _last["dry_run"] = result
    return result


# ── Applying ──────────────────────────────────────────────────────────────


def _write_json(path: Path, data: dict) -> None:
    """Replace a transcript file in one step — never half-written."""
    atomic_write_json(path, data)


def _save(row, tdata: dict) -> None:
    """Write a changed transcript the way an edit does: database, the JSON
    file next to the media (only one that exists — no new files or folders
    on the share), and the subtitle files if they are kept up to date."""
    db = _db()
    abs_path = row["abs_path"] or ""
    db.upsert_transcript(
        row["rel_path"], tdata, media_dir=row["media_dir"] or "", abs_path=abs_path
    )
    if not abs_path:
        return
    try:
        from whisper_tm.core.export import export_alongside

        export_alongside(Path(abs_path), tdata)
    except Exception as e:  # noqa: BLE001
        logger.debug("Re-export after the cleanup failed: %s", e)
    if config.transcript_mode == "database":
        return
    try:
        tp = transcript_path_for(
            row["rel_path"], media_dir=row["media_dir"] or None, abs_path=abs_path
        )
    except ValueError:
        return
    if tp.exists():
        _write_json(tp, tdata)


def _load(conn, tid: int):
    row = conn.execute(
        "SELECT id, rel_path, abs_path, media_dir, complete FROM transcripts WHERE id = ?", (tid,)
    ).fetchone()
    if not row or not row["complete"]:
        return None, None
    tdata = _db().get_transcript(
        row["abs_path"] or row["rel_path"], media_dir=row["media_dir"] or None
    )
    return row, tdata


def apply(run_id: str, stop_event: threading.Event) -> dict:
    """Remove what the dry run ``run_id`` found — backup first, log first."""
    with _lock:
        run = _runs.get(run_id)
    if not run:
        raise ValueError("That dry run is no longer there — run it again")
    compiled, errors = compile_artifact_patterns(run["patterns"])
    if errors:
        raise ValueError("not a valid regular expression — " + "; ".join(errors))
    db = _db()
    targets = run["targets"]

    _progress("backing up the database", 0, len(targets))
    backup = db.backup(dest_dir=db.backup_target())
    if not backup:
        raise RuntimeError(
            "The database backup failed — nothing was changed. "
            "Is the backup folder reachable (Database → Backups)?"
        )

    log_dir().mkdir(parents=True, exist_ok=True)
    log_path = log_dir() / f"cleanup-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    removed_total = changed = 0
    left: Counter = Counter()
    with open(log_path, "w", encoding="utf-8") as log, db.bulk_import():
        # bulk_import: every changed transcript would count toward the
        # change backups (database.backup_every_changes) — thousands of them
        # would copy the whole database to the backup folder again and again. The
        # backup above covers this run.
        log.write(
            json.dumps(
                {
                    "kind": "header",
                    "run_id": run_id,
                    "patterns": run["patterns"],
                    "backup": str(backup),
                    "started_at": time.time(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log.flush()
        for i, tid in enumerate(targets):
            if stop_event.is_set():
                left["stopped before"] = len(targets) - i
                break
            _progress("removing artifacts", i, len(targets))
            with db.connection() as conn:
                row, tdata = _load(conn, tid)
                why = _why_held(_held(conn), row["abs_path"], row["rel_path"]) if row else ""
            if not row or not tdata:
                left["gone or not readable"] += 1
                continue
            if why:
                left[why] += 1
                continue
            segments = tdata.get("segments") or []
            keep, removed = [], []
            for idx, seg in enumerate(segments):
                if artifact_reason(seg.get("text") or "", compiled) is None:
                    keep.append(seg)
                else:
                    removed.append([idx, seg])
            if not removed:
                left["nothing to remove any more"] += 1
                continue
            # Write-ahead: the log line is on disk before the transcript changes
            log.write(
                json.dumps(
                    {
                        "kind": "transcript",
                        "rel_path": row["rel_path"],
                        "abs_path": row["abs_path"],
                        "media_dir": row["media_dir"],
                        "segments_before": len(segments),
                        "removed": removed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log.flush()
            os.fsync(log.fileno())
            tdata["segments"] = keep
            _save(row, tdata)
            removed_total += len(removed)
            changed += 1
        _progress("done", len(targets), len(targets))
    result = {
        "run_id": run_id,
        "segments": removed_total,
        "transcripts": changed,
        "left": [[why, n] for why, n in left.most_common()],
        "log": log_path.name,
        "backup": str(backup),
        "at": time.time(),
    }
    with _lock:
        _runs.pop(run_id, None)
        _last["applied"] = result
    if changed:
        _invalidate_views()
    return result


def _invalidate_views() -> None:
    """File list and statistics cache counts that just changed."""
    try:
        from whisper_tm.core.helpers import invalidate_complete_cache

        invalidate_complete_cache()
    except Exception as e:  # noqa: BLE001
        logger.debug("File list cache not cleared: %s", e)
    try:
        from whisper_tm.api.settings import invalidate_stats_cache

        invalidate_stats_cache()
    except Exception as e:  # noqa: BLE001
        logger.debug("Statistics cache not cleared: %s", e)


# ── Undo ──────────────────────────────────────────────────────────────────


def list_logs() -> list[dict]:
    """The cleanup logs, newest first, with what each changed."""
    out = []
    try:
        files = sorted(log_dir().glob("cleanup-*.jsonl"), reverse=True)
    except OSError:
        return out
    for f in files[:20]:
        transcripts = segments = 0
        header: dict = {}
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    entry = json.loads(line)
                    if entry.get("kind") == "header":
                        header = entry
                    elif entry.get("kind") == "transcript":
                        transcripts += 1
                        segments += len(entry.get("removed") or [])
        except (OSError, ValueError):
            continue
        out.append(
            {
                "name": f.name,
                "undone": f.name.endswith(".undone.jsonl"),
                "transcripts": transcripts,
                "segments": segments,
                "patterns": header.get("patterns") or [],
                "backup": header.get("backup") or "",
                "at": header.get("started_at") or f.stat().st_mtime,
            }
        )
    return out


def _logged_row(conn, entry: dict):
    """The transcript row a log entry was written for — by its identity, or,
    for a row from before abs_path existed, by rel_path among the rows
    without one (and its root, when the log has it).

    An empty abs_path is never a search value: ``abs_path = ''`` matches
    every legacy row, and the first of them would have been given the
    logged transcript's segments.
    """
    cols = "id, rel_path, abs_path, media_dir, complete"
    ab = entry.get("abs_path") or ""
    if ab:
        return conn.execute(
            f"SELECT {cols} FROM transcripts WHERE abs_path = ? LIMIT 1", (ab,)
        ).fetchone()
    rel = entry.get("rel_path") or ""
    if not rel:
        return None
    sql = (
        f"SELECT {cols} FROM transcripts WHERE (abs_path = '' OR abs_path IS NULL) AND rel_path = ?"
    )
    args: list = [rel]
    if entry.get("media_dir"):
        sql += " AND media_dir = ?"
        args.append(entry["media_dir"])
    rows = conn.execute(sql + " LIMIT 2", args).fetchall()
    # Two legacy rows of the same name: which one is meant is a guess — none
    return rows[0] if len(rows) == 1 else None


def undo(name: str, stop_event: threading.Event) -> dict:
    """Put the segments of one cleanup back where they were.

    A transcript that changed since (transcribed again, edited) is left as
    it is: its segment count no longer fits, and a blind insert would put
    old lines between new ones.
    """
    if (
        "/" in name
        or "\\" in name
        or not name.startswith("cleanup-")
        or not name.endswith(".jsonl")
        or name.endswith(".undone.jsonl")
    ):
        raise ValueError("Not a cleanup log")
    path = log_dir() / name
    if not path.is_file():
        raise ValueError("Cleanup log not found")
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("kind") == "transcript":
                entries.append(entry)
    db = _db()
    restored = segments = 0
    left: Counter = Counter()
    with db.bulk_import():
        for i, entry in enumerate(entries):
            if stop_event.is_set():
                left["stopped before"] = len(entries) - i
                break
            _progress("putting segments back", i, len(entries))
            key = entry.get("abs_path") or entry.get("rel_path") or ""
            with db.connection() as conn:
                row = _logged_row(conn, entry)
            tdata = (
                db.get_transcript(key, media_dir=(row["media_dir"] if row else None) or None)
                if row
                else None
            )
            if not row or not tdata:
                left["transcript gone"] += 1
                continue
            current = tdata.get("segments") or []
            removed = entry.get("removed") or []
            if len(current) != int(entry.get("segments_before") or 0) - len(removed):
                left["changed since the cleanup"] += 1
                continue
            merged = list(current)
            for idx, seg in sorted(removed, key=lambda r: r[0]):
                merged.insert(int(idx), seg)
            tdata["segments"] = merged
            _save(row, tdata)
            restored += 1
            segments += len(removed)
        _progress("done", len(entries), len(entries))
    if not stop_event.is_set():
        path.rename(path.with_name(path.name[: -len(".jsonl")] + ".undone.jsonl"))
    result = {
        "log": name,
        "transcripts": restored,
        "segments": segments,
        "left": [[why, n] for why, n in left.most_common()],
        "at": time.time(),
    }
    with _lock:
        _last["undone"] = result
    if restored:
        _invalidate_views()
    return result

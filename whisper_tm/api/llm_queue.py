"""
API of the LLM queue: enqueue (single file or whole folder),
query state, read an entry including partial results, cancel.

All LLM tasks keep running server-side — closing a dialog
or browser no longer cancels anything (core/llm_queue.py).
"""

from __future__ import annotations

import json
import logging
import time

from flask import Blueprint, Response, jsonify, request

from whisper_tm.config import config
from whisper_tm.core.llm_queue import FILE_TYPES, LABEL, llm_queue

logger = logging.getLogger(__name__)

llm_queue_bp = Blueprint("llm_queue", __name__)


def _db():
    from whisper_tm.extensions import db

    return db


# ── Enqueue ───────────────────────────────────────────────────────────────


@llm_queue_bp.route("/api/llm/queue/add", methods=["POST"])
def queue_add():
    """Enqueue task(s).

    Body: {kind, filenames: [...] | filename, params: {...}, front?}
    """
    data = request.json if isinstance(request.json, dict) else {}
    kind = (data.get("kind") or "").strip()
    if kind not in FILE_TYPES:
        return jsonify({"error": f'Unknown type "{kind}" (allowed: {", ".join(FILE_TYPES)})'}), 400
    filenames = data.get("filenames")
    if filenames is None and data.get("filename"):
        filenames = [data["filename"]]
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"error": "No files"}), 400
    if len(filenames) > 5000:
        return jsonify({"error": f"Too many files ({len(filenames)})"}), 400
    params = data.get("params")
    if params is not None and not isinstance(params, dict):
        return jsonify({"error": "Field 'params' must be an object"}), 400
    if not config.get("llm", "endpoint", default=""):
        return jsonify({"error": "LLM not configured (Settings → AI models)"}), 400

    ids = [
        llm_queue.enqueue(kind, fn, params or {}, front=bool(data.get("front"))) for fn in filenames
    ]
    return jsonify(
        {
            "ok": True,
            "ids": [i for i in ids if i],
            "added": len([i for i in ids if i]),
            "label": LABEL[kind],
        }
    )


@llm_queue_bp.route("/api/llm/queue/folder", methods=["POST"])
def queue_folder():
    """Enqueue a whole folder — all files WITH a finished transcript.

    Body: {kind, dir: "subfolder" | "", recursive: true,
           skip_existing: true, params: {...}, dry_run_only: bool}

    ``dry_run_only`` only returns the preview (how many files would be
    affected) — the dialog shows that before anything is enqueued.
    """
    data = request.json if isinstance(request.json, dict) else {}
    kind = (data.get("kind") or "").strip()
    if kind not in FILE_TYPES:
        return jsonify({"error": f'Unknown type "{kind}"'}), 400
    db = _db()
    if not db:
        return jsonify({"error": "No database available"}), 500

    below = (data.get("dir") or "").strip().strip("/")
    recursive = bool(data.get("recursive", True))
    skip_it = bool(data.get("skip_existing", True))
    params = data.get("params") or {}
    language = (params.get("language") or "").strip()

    # Transcripts without speech stay out: every task on them fails
    empty: list[str] = []
    candidates = _files_with_transcript(db, below, recursive, empty=empty)
    total = len(candidates)
    if skip_it:
        candidates = [f for f in candidates if not _has_result(db, kind, f, language)]
    skipped = total - len(candidates)

    if data.get("dry_run_only"):
        return jsonify(
            {
                "found": total,
                "to_enqueue": len(candidates),
                "skipped": skipped,
                "empty": len(empty),
                "examples": [f.split("/")[-1] for f in candidates[:8]],
            }
        )

    if not config.get("llm", "endpoint", default=""):
        return jsonify({"error": "LLM not configured"}), 400
    ids = llm_queue.enqueue_many(kind, candidates, params)
    return jsonify(
        {
            "ok": True,
            "added": len([i for i in ids if i]),
            "skipped": skipped,
            "found": total,
            "empty": len(empty),
            "label": LABEL[kind],
        }
    )


def _files_with_transcript(db, below: str, recursive: bool, empty: list | None = None) -> list[str]:
    """Names of all fully transcribed files (optionally within a subfolder),
    relative to the open folder.

    Only transcripts of the CURRENT root — the DB holds several roots
    at once, so filter by abs_path (docs/PATHS.md). The name comes from
    the identity too: the stored rel_path is relative to the folder open
    when the file was transcribed, and a subfolder's batch missed every
    file transcribed with another folder open.

    Transcripts without a word go to ``empty`` instead, when it is given.
    """
    from whisper_tm.core import paths

    # The stored form (C:/… on Windows), not the folder's own spelling
    root = paths.canon(config.media_dir).rstrip("/") + "/"
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT rel_path, abs_path, word_count FROM transcripts WHERE complete=1 "
            "AND (abs_path IS NULL OR abs_path = '' OR (abs_path >= ? AND abs_path < ?))",
            (root, root[:-1] + "0"),
        ).fetchall()
    out = []
    for r in sorted(rows, key=lambda r: r["abs_path"] or r["rel_path"] or ""):
        ap = r["abs_path"] or ""
        if ap and not ap.startswith(root):
            continue
        rel = ap[len(root) :] if ap else (r["rel_path"] or "")
        if below:
            if not rel.startswith(below + "/") and rel != below:
                continue
            if not recursive and "/" in rel[len(below) + 1 :]:
                continue
        elif not recursive and "/" in rel:
            continue
        if r["word_count"] == 0:
            if empty is not None:
                empty.append(rel)
            continue
        out.append(rel)
    return out


def _has_result(db, kind: str, filename: str, language: str) -> bool:
    """Is there already a result of this kind for this file?"""
    try:
        if kind == "summary":
            return bool(db.get_summaries(filename, media_dir=str(config.media_dir)))
        if kind == "translation":
            if not language:
                return False
            return (
                db.get_translation(filename, language, media_dir=str(config.media_dir)) is not None
            )
        if kind == "correction":
            return bool(db.get_llm_edit_runs(filename, media_dir=str(config.media_dir)))
    except Exception as e:
        logger.debug("Existence check (%s/%s): %s", kind, filename, e)
    return False


# ── State ─────────────────────────────────────────────────────────────────


@llm_queue_bp.route("/api/llm/queue")
def queue_state():
    """The LLM queue: counts per status, paused flag and the entries (summary,
    translation and correction tasks) with progress and errors.
    """
    return jsonify(llm_queue.snapshot())


@llm_queue_bp.route("/api/llm/queue/item/<int:entry_id>")
def queue_item(entry_id):
    """One entry with partial results and final result."""
    e = llm_queue.entry(entry_id)
    if not e:
        return jsonify({"error": "Entry not found"}), 404
    return jsonify(e)


@llm_queue_bp.route("/api/llm/queue/open")
def queue_open():
    """Is a task of this kind for this file already running/waiting?

    Query: ?kind=summary&filename=…
    This lets a newly opened dialog attach to the running job.
    """
    kind = (request.args.get("kind") or "").strip()
    filename = (request.args.get("filename") or "").strip()
    if kind not in FILE_TYPES or not filename:
        return jsonify({"error": "kind and filename required"}), 400
    return jsonify({"entry": llm_queue.open_entry(kind, filename)})


@llm_queue_bp.route("/api/llm/queue/stream")
def queue_stream():
    """SSE: queue state, as long as something is moving."""

    def gen():
        last_one = None
        start = time.time()
        while time.time() - start < 3600:
            snap = llm_queue.snapshot(limit=40)
            fingerprint = json.dumps(
                [(i["id"], i["status"], i["progress"], i["message"]) for i in snap["items"]],
                ensure_ascii=False,
            )
            if fingerprint != last_one:
                last_one = fingerprint
                yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
            else:
                yield ": ping\n\n"
            time.sleep(1.5)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Manage ────────────────────────────────────────────────────────────────


@llm_queue_bp.route("/api/llm/queue/cancel", methods=["POST"])
def queue_cancel():
    """Cancel an LLM task. Body: ``id``. A running section still finishes."""
    data = request.json if isinstance(request.json, dict) else {}
    entry_id = data.get("id")
    if entry_id is None:
        return jsonify({"error": "id required"}), 400
    return jsonify({"ok": llm_queue.cancel(int(entry_id))})


@llm_queue_bp.route("/api/llm/queue/retry", methods=["POST"])
def queue_retry():
    """Put a failed LLM task back into the queue. Body: ``id``."""
    data = request.json if isinstance(request.json, dict) else {}
    entry_id = data.get("id")
    if entry_id is None:
        return jsonify({"error": "id required"}), 400
    return jsonify({"ok": llm_queue.retry(int(entry_id))})


@llm_queue_bp.route("/api/llm/queue/clear", methods=["POST"])
def queue_clear():
    """Remove LLM tasks from the list. Body: ``what`` — ``done`` (finished,
    failed, cancelled) or ``pending`` (everything still waiting).
    """
    data = request.json if isinstance(request.json, dict) else {}
    what = data.get("what") or "done"
    if what not in ("done", "pending"):
        return jsonify({"error": "what: done|pending"}), 400
    return jsonify({"ok": True, "removed": llm_queue.clear(what)})


@llm_queue_bp.route("/api/llm/queue/pause", methods=["POST"])
def queue_pause():
    """Pause or resume the LLM queue. Body: ``pause`` (default true). A running
    task finishes first.
    """
    data = request.json if isinstance(request.json, dict) else {}
    llm_queue.pause(bool(data.get("pause", True)))
    return jsonify({"ok": True, "paused": llm_queue._paused})

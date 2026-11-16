"""
Transcription routes: status/SSE, CRUD, skip.

Nothing is started here: queueing happens exclusively via
/api/queue/add into the persistent queue (core/queue.py), which a
permanently running worker processes — even without open WebUI.
"""

import json
import logging
import time
from pathlib import Path

from flask import Blueprint, Response, abort, jsonify, request, send_file

from whisper_tm.config import config
from whisper_tm.core import debug
from whisper_tm.core.helpers import (
    add_to_skiplist,
    atomic_write_json,
    find_transcript,
    invalidate_complete_cache,
    remove_from_skiplist,
    transcript_identity,
    transcript_path_for,
)
from whisper_tm.extensions import db, hw, job_manager

logger = logging.getLogger(__name__)

transcription_bp = Blueprint("transcription", __name__)


# Short-lived cache for log search: {(folder, suffix): (time, name)}
# Kept short because a running job can create a newer log file anytime.
_LOG_CACHE: dict = {}
_LOG_CACHE_SEC = 120.0


def _find_log_for_file(filename: str) -> str:
    """Find most recent log file for a media file — EXPENSIVE, on-demand only.

    **Why it is expensive:** the log directory can sit on a network share
    and hold well over 100,000 files. A single directory scan over CIFS
    can take **10–20 s**; an earlier version also did an ``is_file()`` per
    entry — one more network query per file.

    It used to be attached to ``/api/transcript/<file>``, so ran on EVERY
    file open. A typical log of that time:

        GET /api/video/…      → 206 in    1 ms   ← video available immediately
        GET /api/transcript/… → 200 in   ~50 s   ← THIS blocked

    Result only serves to fill the target name for the "📋 Log" button.
    So the search now sits behind its own endpoint and only runs when
    someone clicks the button.

    Two optimizations remain:

    * ``os.scandir`` and name comparison — no ``is_file()`` per entry.
      Only the few matches get touched.
    * short-lived cache: opening same log twice doesn't require two waits.
    """
    import os
    import time as _t

    logs_dir = config.dirs.get("logs")
    if not logs_dir or not logs_dir.exists():
        return ""
    from whisper_tm.jobs import log_name_part

    # Logs written before the Windows-safe names kept ":", "?" … as they were
    old = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    suffixes = tuple(dict.fromkeys([f"_{log_name_part(filename)}.log", f"_{old}.log"]))

    key = (str(logs_dir), suffixes)
    now = _t.monotonic()
    cached = _LOG_CACHE.get(key)
    if cached and now - cached[0] < _LOG_CACHE_SEC:
        return cached[1]

    mp = debug.MeasurePoints(f"Log search {Path(filename).name}")
    matches = []
    try:
        with os.scandir(logs_dir) as entries:
            for e in entries:
                # Only NAME is compared — it's already in the
                # directory entry and costs no extra network query.
                if e.name.endswith(suffixes):
                    matches.append(e)
    except OSError as e:
        logger.debug("Log directory not readable: %s", e)
        return ""
    mp.point("Read directory (NAS)")

    name = ""
    if matches:
        # stat() only here — for a handful of matches, not every file.
        name = max(matches, key=lambda e: e.stat().st_mtime).name
    mp.point("Choose most recent")
    mp.done(f"{len(matches)} matches")

    _LOG_CACHE[key] = (now, name)
    return name


def _checked(filename: str) -> str:
    """A filename from a request — a full path only within the allowed roots
    (403), as ``?file=`` checks it. Relative names stay below their root by
    construction (paths.abs_from refuses ``..``)."""
    from whisper_tm.core import paths
    from whisper_tm.core.security import check_root

    if paths.is_absolute(filename):
        check_root(filename, field_name="filename")
    return filename


def _file_param() -> str:
    """``?file=`` — a full path only within the allowed roots (403)."""
    from whisper_tm.core import paths
    from whisper_tm.core.security import check_root

    f = request.args.get("file") or ""
    if not f:
        abort(400, "file missing")
    if paths.is_absolute(f):
        check_root(f, field_name="file")
    return f


@transcription_bp.route("/api/transcript")
@transcription_bp.route("/api/transcript/<path:filename>")
def get_transcript(filename=None):
    """Load transcript. DB-first, JSON only as fallback.

    The path also comes as ``?file=`` — a full one can't travel in the URL
    path (see ``/api/speaker/windows``); it must lie in the allowed roots.
    """
    filename = _checked(filename) if filename else _file_param()
    # DB first. Root belongs in the call so it can be resolved by identity
    # (abs_path) — earlier basename loop here was a workaround for broken
    # rel_path values and would confuse a/intro.mp4 with b/intro.mp4 (docs/PATHS.md).
    if db:
        try:
            data = db.get_transcript(filename, media_dir=str(config.media_dir))
            if data:
                # NO log search here — see _find_log_for_file: it can cost
                # up to ~50 s per file call on a network share, for nothing but
                # labeling a button. Whoever clicks the button fetches the
                # name via /api/transcript/<file>/logfile.
                return jsonify(data)
        except Exception as e:
            logger.debug("DB get_transcript(%s): %s", filename, e)

    # Fallback: JSON (only if not database-only mode)
    if config.transcript_mode != "database":
        tp = find_transcript(filename)
        if tp:
            with open(tp, encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)

    return jsonify({"error": "No transcript available"}), 404


@transcription_bp.route("/api/transcript/<path:filename>/logfile")
def get_transcript_logfile(filename):
    """Find log file name — on demand, not in advance.

    This search used to run on every transcript fetch and was why opening
    a file could take up to ~50 seconds, even though video loaded in 1 ms
    (see _find_log_for_file).
    """
    return jsonify({"log_file": _find_log_for_file(filename)})


def _delete_transcript_everywhere(filename: str, media_dir: str | Path | None = None) -> None:
    """Delete transcript from storage AND database.

    Mirror of :func:`find_transcript`: there is exactly ONE storage location,
    so exactly one file here. (Earlier there were two delete attempts for two
    possible locations — the second search would find nothing after the first unlink() anyway.)

    Without this delete, /api/transcribe continues returning 409 "already exists".
    """
    from whisper_tm.core.helpers import find_transcript

    # Root MUST come along. Queue worker calls this for an entry whose
    # base_dir can differ from current config.media_dir — without it, the
    # re-transcription would delete the transcript of same-named file under
    # the current root (docs/PATHS.md).
    root = str(media_dir) if media_dir else str(config.media_dir)

    # Every file of the recording: central mode can hold an old flat-named
    # one beside a hashed one (helpers.transcript_files_for) — leaving one
    # had it found again right after the delete
    from whisper_tm.core.helpers import transcript_files_for

    tp = find_transcript(filename, media_dir=Path(root))
    files = [tp] if tp else []
    ident = transcript_identity(filename, media_dir=root)
    if ident:
        files += [p for p in transcript_files_for(ident) if p not in files]
    for fp in files:
        try:
            fp.unlink()
        except OSError as e:
            logger.debug("Transcript delete failed: %s", e)
    if db:
        try:
            # Delete only the intended file. Earlier basename lookup would
            # have pulled in transcripts of same-named files in other folders —
            # not fixable on delete.
            db.delete_transcript(filename, media_dir=root)
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)

    # Speaker vectors live in their own database and would otherwise know
    # nothing of the deletion. Orphaned records remained: windows of a file
    # that no longer exists — they appeared in the map, could be clicked but
    # not played. Errors here must not stop transcript deletion.
    try:
        from whisper_tm.database.speaker import speaker_db

        if ident:
            deleted = speaker_db().remove(ident)
            if deleted:
                logger.info("Speaker vectors also deleted: %d windows of %s", deleted, filename)
    except Exception as e:
        logger.debug("Speaker vectors not deleted: %s", e)

    invalidate_complete_cache(filename, media_dir=root)


def _delete_derivatives(
    filename: str,
    media_dir: str | Path | None = None,
    *,
    summaries: bool = False,
    translations: bool = False,
    edits: bool = False,
) -> dict:
    """Delete derivatives — from DB AND from JSON sidecar alongside.

    Sidecar must be included: read side heals from it ("DB primary,
    JSON heals"), so a summary deleted only in DB would be back on next load.
    """
    deleted = {"summaries": 0, "translations": 0, "edits": 0}
    if not (summaries or translations or edits):
        return deleted
    root = str(media_dir) if media_dir else str(config.media_dir)
    if db:
        try:
            deleted = db.delete_derivatives(
                filename,
                media_dir=root,
                summaries=summaries,
                translations=translations,
                edits=edits,
            )
        except Exception as e:
            logger.warning("Derivatives not deletable: %s", e)

    sidecars: list[Path] = []
    if summaries:
        from whisper_tm.core.summarize import summaries_path

        sidecars.append(summaries_path(filename))
    if translations:
        from whisper_tm.api.llm_jobs import _translations_json_path

        sidecars.append(_translations_json_path(filename))
    for fp in sidecars:
        try:
            fp.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Sidecar delete failed (%s): %s", fp, e)
    return deleted


@transcription_bp.route("/api/derivatives/<path:filename>")
def count_derivatives(filename):
    """What else hangs off the file besides the transcript? For delete dialog.

    Should only offer what actually exists — a line "Translations (0)"
    is just an opportunity to accidentally check something.
    """
    if not db:
        return jsonify({"transcript": False, "summaries": 0, "translations": 0, "edits": 0})
    root = str(config.media_dir)
    try:
        return jsonify(
            {
                "transcript": db.is_complete(filename, media_dir=root),
                "summaries": len(db.get_summaries(filename, media_dir=root) or []),
                "translations": len(db.get_translation_languages(filename, media_dir=root) or []),
                "edits": len(db.get_llm_edit_runs(filename, media_dir=root) or []),
            }
        )
    except Exception as e:
        logger.warning("Count derivatives failed: %s", e)
        return jsonify({"transcript": False, "summaries": 0, "translations": 0, "edits": 0})


@transcription_bp.route("/api/delete_transcript", methods=["POST"])
def delete_transcript():
    """Delete a file's transcript and, only if listed, its derivatives.

    Body: ``filename``; ``what`` — list of parts to delete (default
    ``["transcript"]``; summaries/translations only when named).
    """
    data = request.json if isinstance(request.json, dict) else {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename"}), 400
    _checked(str(filename))
    # Without specification: just the transcript. Derivatives are independent and
    # only deleted if explicitly checked in dialog — otherwise the old,
    # silent co-deletion would be back.
    what = data.get("what")
    if not isinstance(what, list):
        what = ["transcript"]
    deleted = {"transcript": False}
    if "transcript" in what:
        _delete_transcript_everywhere(filename)
        deleted["transcript"] = True
    deleted.update(
        _delete_derivatives(
            filename,
            summaries="summaries" in what,
            translations="translations" in what,
            edits="edits" in what,
        )
    )
    return jsonify({"ok": True, "deleted": deleted})


@transcription_bp.route("/api/delete_all_transcripts", methods=["POST"])
def delete_all_transcripts():
    """Delete transcripts by file list."""
    data = request.json if isinstance(request.json, dict) else {}
    token = data.get("confirm_token")
    if token != "DELETE_ALL_CONFIRMED":
        return jsonify({"error": "Invalid confirmation token"}), 400

    filenames = data.get("filenames", [])
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"error": "No files specified"}), 400

    for fn in filenames:
        _checked(str(fn))  # all or none: checked before the first delete
    deleted = 0
    for fn in filenames:
        # Consistent with single delete: removes DB + local .transcripts/
        # + basename entry, else 409 "already exists" after bulk delete.
        if find_transcript(fn) is not None:
            deleted += 1
        _delete_transcript_everywhere(fn)
    invalidate_complete_cache()  # Clear entire cache for bulk delete
    return jsonify({"ok": True, "deleted": deleted})


@transcription_bp.route("/api/save_speaker_names", methods=["POST"])
def save_speaker_names():
    """Save speaker name mappings. DB-first."""
    data = request.json if isinstance(request.json, dict) else {}
    filename = data.get("filename")
    names = data.get("speaker_names", {})
    if not filename:
        return jsonify({"error": "No filename"}), 400
    if not isinstance(names, dict):
        return jsonify({"error": "speaker_names must be an object"}), 400

    clean = {k: v for k, v in names.items() if v and v != k}
    _checked(str(filename))

    try:
        tdata = _load_transcript_data(filename)
        if not tdata:
            return jsonify({"error": "Transcript not found"}), 404

        # Update speaker names
        if clean:
            tdata["speaker_names"] = clean
        elif "speaker_names" in tdata:
            del tdata["speaker_names"]

        # The same write as an edit: identity resolved through the database,
        # the JSON replaced atomically, a failed database write reported
        _save_transcript_data(filename, tdata)
        # The meaning search shows these names from its next search on
        from whisper_tm.rag import speakers

        speakers.forget()

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@transcription_bp.route("/api/edit_segment", methods=["POST"])
def edit_segment():
    """Edit individual segment.

    Body: {"filename": "...", "index": 0, "text": "new text"}
    """
    data = request.json or {}
    filename = data.get("filename")
    index = data.get("index")
    new_text = data.get("text", "")

    if not filename or index is None:
        return jsonify({"error": "Filename and index required"}), 400

    try:
        index = int(index)
    except (TypeError, ValueError):
        return jsonify({"error": "Index must be an integer"}), 400
    _checked(str(filename))

    try:
        tdata = _load_transcript_data(filename)
        if not tdata:
            return jsonify({"error": "Transcript not found"}), 404

        segments = tdata.get("segments", [])
        if index < 0 or index >= len(segments):
            return jsonify({"error": f"Index {index} invalid (0-{len(segments) - 1})"}), 400

        old_text = segments[index].get("text", "")
        segments[index]["text"] = new_text.strip()

        _save_transcript_data(filename, tdata)

        return jsonify({"ok": True, "old": old_text, "new": new_text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@transcription_bp.route("/api/find_replace", methods=["POST"])
def find_replace():
    """Find and replace across all segments.

    Body: {
        "filename": "...",
        "find": "old word",
        "replace": "new word",
        "case_sensitive": false,
        "whole_word": false
    }
    Returns: {"ok": true, "replaced": 5, "segments_changed": 3}
    """
    data = request.json or {}
    filename = data.get("filename")
    find_str = data.get("find", "")
    replace_str = data.get("replace", "")
    case_sensitive = data.get("case_sensitive", False)
    whole_word = data.get("whole_word", False)

    if not filename or not find_str:
        return jsonify({"error": "Filename and search term required"}), 400
    _checked(str(filename))

    try:
        tdata = _load_transcript_data(filename)
        if not tdata:
            return jsonify({"error": "Transcript not found"}), 404

        import re

        if whole_word:
            # Not \b: a term that starts or ends with a non-letter ("C++",
            # "(laughs)") never matched. The page's preview uses the same rule.
            pattern = r"(?<!\w)" + re.escape(find_str) + r"(?!\w)"
            flags = 0 if case_sensitive else re.IGNORECASE
        else:
            pattern = re.escape(find_str)
            flags = 0 if case_sensitive else re.IGNORECASE

        total_replaced = 0
        segments_changed = 0

        for seg in tdata.get("segments", []):
            text = seg.get("text", "")
            count = len(re.findall(pattern, text, flags))
            if count > 0:
                seg["text"] = re.sub(pattern, lambda m: replace_str, text, flags=flags)
                total_replaced += count
                segments_changed += 1

        if total_replaced > 0:
            _save_transcript_data(filename, tdata)

        return jsonify(
            {
                "ok": True,
                "replaced": total_replaced,
                "segments_changed": segments_changed,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_transcript_data(filename: str) -> dict | None:
    """Load transcript data (prefer DB, fallback to JSON)."""
    tdata = None
    if db:
        # Pass media_dir: also finds transcripts still stored relative to
        # an earlier media directory.
        tdata = db.get_transcript(filename, media_dir=str(config.media_dir))
    if not tdata and config.transcript_mode != "database":
        tp = find_transcript(filename)
        if tp:
            with open(tp, encoding="utf-8") as f:
                tdata = json.load(f)
    return tdata


def _save_transcript_data(filename: str, tdata: dict):
    """Save transcript data (DB + JSON).

    Resolve identity FIRST through DB (docs/PATHS.md): `filename` comes
    from the request, and frontend also sends the filename from inside the
    TRANSCRIPT (activeTranscript.filename) — it's relative to the root that
    applied WHEN TRANSCRIBING. transcript_path_for without abs_path would guess
    /current_root/<old_relpath>/ and create an empty ghost tree there via mkdir.
    That's exactly how a ghost folder like "<some folder>/.transcripts"
    can appear (incident 1.3).

    A failed database write raises when the database is the only copy
    (``transcript_mode: database``) — the edit would otherwise be lost while
    the page heard "ok". With a JSON next to the media it is
    a warning: the file is written below and the start sync heals the row.
    The JSON is replaced atomically.
    """
    abs_path = ""
    if db:
        try:
            with db.connection() as conn:
                row = db.resolve_transcript(conn, filename, str(config.media_dir))
                if row and row["abs_path"]:
                    abs_path = row["abs_path"]
        except Exception as e:
            logger.debug("Identity lookup for %s: %s", filename, e)
        try:
            db.upsert_transcript(
                filename, tdata, media_dir=str(config.media_dir), abs_path=abs_path
            )
        except Exception as e:
            if config.transcript_mode == "database":
                raise RuntimeError(f"Transcript not saved (database): {e}") from e
            logger.warning("DB transcript save failed (%s), JSON still written: %s", filename, e)
    # Re-export next to file (SRT/TXT): After text edits (edit_segment,
    # find_replace, LLM correction) the exported SRT must not stay on the
    # old version. export_alongside checks itself whether the feature is
    # active (transcription.export_alongside) — call here only with known
    # identity (no guessing, ghost folder protection).
    if abs_path:
        try:
            from whisper_tm.core.export import export_alongside

            export_alongside(Path(abs_path), tdata)
        except Exception as e:
            logger.debug("Re-export after edit failed: %s", e)
    if config.transcript_mode != "database":
        try:
            tp = transcript_path_for(filename, abs_path=abs_path or None)
        except ValueError as e:
            # Without derivable identity there's no storage location. Transcript
            # already in DB — skip JSON copy, don't write to a guessed location
            # (docs/PATHS.md).
            logger.warning("Transcript JSON not storable (%s): %s", filename, e)
            return
        # Ghost folder protection: if DB has no identity, path is only GUESSED
        # from filename + current root. Then the media file must actually be there,
        # else no mkdir/write — transcript already in DB.
        if not abs_path:
            media_location = Path(transcript_identity(filename))
            if not media_location.exists():
                logger.warning(
                    "Transcript JSON not stored: media file not at "
                    "derived location (%s) — ghost folder protection "
                    "(docs/PATHS.md, incident 1.3)",
                    media_location,
                )
                return
        if tp.exists() or config.transcript_mode == "alongside":
            # For "alongside" also create new — .transcripts/ folder missing first time.
            tp.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(tp, tdata)


@transcription_bp.route("/api/skip", methods=["POST"])
def skip_file():
    """Put a file on the skip list so batches and the queue leave it alone.

    Body: ``filename``, optional ``reason`` (default ``manual``).
    """
    # Route decorator was missing — /api/skip was never registered (404)
    data = request.json if isinstance(request.json, dict) else {}
    filename = data.get("filename")
    reason = data.get("reason", "manual")
    if not filename:
        return jsonify({"error": "No filename"}), 400
    add_to_skiplist(filename, reason)
    return jsonify({"ok": True})


@transcription_bp.route("/api/unskip", methods=["POST"])
def unskip_file():
    """Take a file off the skip list again.

    Body: ``filename``, optional ``base_dir`` (default: the open media folder).
    """
    data = request.json if isinstance(request.json, dict) else {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename"}), 400
    # Like re-transcription: via identity, else the lookup in JSON skiplist
    # never hits (docs/PATHS.md). And clear cache — else file remains skipped
    # until TTL expires.
    root = data.get("base_dir") or str(config.media_dir)
    remove_from_skiplist(filename, root)
    return jsonify({"ok": True})


@transcription_bp.route("/api/vocal_isolation_status")
def vocal_isolation_status():
    """Is vocal isolation available, and which separation models exist
    (catalog grouped for the dialog, plus the default model).
    """
    from whisper_tm.core.vocal_isolation import GROUP_ORDER, model_catalog

    models = config.get("vocal_isolation", "models", default={})
    default = config.get("vocal_isolation", "default_model", default="bs_roformer")
    return jsonify(
        {
            "available": hw.has_audio_separator if hw else False,
            "models": models,  # name → filename (backward compat)
            "catalog": model_catalog(),  # annotated, grouped list for the GUI
            "group_order": GROUP_ORDER,
            "default_model": default,
        }
    )


@transcription_bp.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """Ask a running job to stop. It ends at the next checkpoint; 404 if the
    job id is unknown.
    """
    job = job_manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    job.cancel = True
    return jsonify({"ok": True})


@transcription_bp.route("/api/active_jobs")
def active_jobs():
    """All running jobs (for reconnect)."""
    return jsonify({"jobs": job_manager.list_active()})


@transcription_bp.route("/api/jobs/<job_id>")
def job_status_json(job_id):
    """One job as JSON — same info as SSE stream, without streaming.

    Why it's needed: `/api/status/<job_id>` is a persistent `text/event-stream`,
    and `/api/active_jobs` filters out finished jobs (`jobs/__init__.py`, `list_active`).
    So the **result** of a job was unreachable for any non-streaming caller — especially
    the `result.files` field of a YouTube download, i.e. the answer to "which
    file was actually created?". Filenames come from `secure_filename(title)[:80]`
    with dedup counter and are unpredictable; without this info, only option would be
    searching upload directory by timestamp and guessing.

    **Two sources, in this order.** JobManager keeps jobs only for a limited time
    (`max_age_secs`, default one hour) then evicts them. Someone asking an hour later
    gets the row from `job_history` — fewer fields, but it answers "did it finish
    and how long did it take?". A 404 here would be a lie: the job existed.
    """
    job = job_manager.get(job_id)
    if job:
        d = job.to_dict()
        d["source"] = "running"
        return jsonify(d)

    # `db` comes from the module import above — same instance used by all
    # other routes in this file.
    if db:
        try:
            with db.connection() as conn:
                r = conn.execute(
                    "SELECT job_id, filename, job_type, status, model, language, "
                    "       duration_sec, processing_sec, error_message, log_file, "
                    "       created_at, finished_at "
                    "  FROM job_history WHERE job_id = ? "
                    " ORDER BY id DESC LIMIT 1",
                    (job_id,),
                ).fetchone()
        except Exception as e:  # noqa — info is not mandatory
            logger.debug("job_history not readable: %s", e)
            r = None
        if r:
            return jsonify(
                {
                    "jobId": r["job_id"],
                    "filename": r["filename"],
                    "type": r["job_type"],
                    "status": r["status"],
                    # History has no progress anymore — only result.
                    # Claiming 100 would be wrong for 'error'.
                    "progress": 100 if r["status"] == "done" else 0,
                    "message": "",
                    "result": None,
                    "error": r["error_message"] or "",
                    "elapsed_sec": round(r["processing_sec"] or 0, 1),
                    "warning_count": 0,
                    "log_file": r["log_file"] or "",
                    "model": r["model"],
                    "language": r["language"],
                    "duration_sec": r["duration_sec"],
                    "created_at": r["created_at"],
                    "finished_at": r["finished_at"],
                    "source": "history",
                }
            )
    return jsonify({"error": "Job not found"}), 404


@transcription_bp.route("/api/status/<job_id>")
def job_status(job_id):
    """Job status as SSE stream with timeout and heartbeat."""
    MAX_DURATION = 4 * 3600  # 4h max
    HEARTBEAT_INTERVAL = 30  # 30s heartbeat

    def stream():
        first = True
        start = time.time()
        last_heartbeat = start
        while True:
            # Timeout: end SSE stream after MAX_DURATION
            elapsed = time.time() - start
            if elapsed > MAX_DURATION:
                timeout = {"status": "timeout", "message": "SSE stream timeout"}
                yield f"data: {json.dumps(timeout)}\n\n"
                break

            job = job_manager.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'unknown'})}\n\n"
                break

            data = job_manager.get_stream_data(job, first=first)
            first = False

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if job.status in ("done", "error"):
                break

            time.sleep(1)

            # Heartbeat comment (keeps proxies/load balancers alive)
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n"
                last_heartbeat = now

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Under /download/, not /api/logs/<path:filename>: catch-all collided
# with log management routes /api/logs/view|delete|delete_all in
# api/settings.py and only worked via Flask's precedence of static rules.
@transcription_bp.route("/api/logs/download/<path:filename>")
def download_log(filename):
    """Download a job log from the logs directory (path relative to it) —
    only *.log files, no links (the checks of the log viewer)."""
    from whisper_tm.api.settings import _log_file

    fp = _log_file(filename)
    if fp is None or not fp.exists():
        abort(404)
    return send_file(str(fp), as_attachment=True, download_name=fp.name, mimetype="text/plain")

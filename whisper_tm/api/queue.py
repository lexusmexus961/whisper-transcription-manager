"""
Queue routes: enqueue, reorder, remove, pause, status SSE.

The queue lives server-side (table queue_items) and is worked off by a
permanently running worker. The frontend is now just a viewer:
it enqueues here and watches via /api/queue/stream. If the user closes
the WebUI, the queue keeps running anyway.
"""

import json
import logging
import time

from flask import Blueprint, Response, jsonify, request

import whisper_tm.extensions as ext
from whisper_tm.core import queue as q

logger = logging.getLogger(__name__)

queue_bp = Blueprint("queue", __name__)

# Plain sanity limit against absurd requests — libraries with hundreds of
# thousands of files are realistic, the queue is DB-backed and can handle it.
MAX_QUEUE = 1_000_000
# Only send the head of the waiting list per tick — at 100k files the
# full list every second would be far too expensive.
ITEMS_PREVIEW = 200


# Only pass through what the caller actually sent. _normalize() fills
# missing values from the config (Settings → Transcription). A hard
# False/"turbo" here would make the configured default ineffective —
# exactly the bug that made half the settings page a dummy.
_SETTING_KEYS = (
    "language",
    "model",
    "diarize",
    "numSpeakers",
    "vocal_isolation",
    "vocal_model",
    "save_audio_alongside",
    "word_timestamps",
    "export_alongside",
    "retranscribe",
    "low_vram",
    "vocab_preset",
    # Voice vectors. Only takes effect together
    # with diarize; the acceptance question is handled by the worker itself.
    "speaker_vectors",
    # LLM summary as a post-step (like retranscribe: does NOT go to
    # transcription_worker, but is evaluated in core/queue.py itself)
    "summarize",
    "summarize_preset",
    "summarize_model",
    # … and the language it is written in ("" = the transcript's), and an
    # own instruction instead of the template (browser extension, "Custom…")
    "summarize_language",
    "summarize_prompt",
    # OOM path (core/queue._oom_retry): WhisperX batch per entry and
    # the counter of VRAM failures — filterable/resettable in the editor
    # (null = hardware recommendation resp. counter cleared).
    "batch_size",
    "oom_attempts",
    # Job kind: "voices" = diarization + voice vectors only, transcript
    # untouched (core/transcription.py, voices_only). Absent = transcribe.
    "job",
    # With job "voices": renew the file's voices, carrying names, trash and
    # history over to the new ones (SpeakerDB.replace_keeping_decisions)
    "refresh",
)


def _settings_from(data: dict) -> dict:
    return {k: data[k] for k in _SETTING_KEYS if k in data}


def _json_safe(value):
    """``value`` if it serializes, else None: job.phase is set by the worker
    and only passed through — one odd value must not break the stream for
    every open browser."""
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    return value


def _running_items(snap: dict) -> list[dict]:
    """The snapshot's running entries (queue lanes: several at once, oldest
    first), each with its own job's progress.

    current_* and job_* in the payloads stay the oldest entry's, for the
    readers that know only one. ``job_id`` is '' until the lane has made the
    job — such an entry has no progress fields yet.
    """
    jm = ext.job_manager
    out = []
    for item in snap.get("running_items") or []:
        entry = dict(item)
        job = jm.get(entry["job_id"]) if jm and entry.get("job_id") else None
        if job:
            entry["progress"] = job.progress
            entry["message"] = job.message
            entry["job_elapsed_sec"] = round(job.elapsed(), 1)
            entry["phase"] = _json_safe(getattr(job, "phase", None))
        out.append(entry)
    return out


@queue_bp.route("/api/queue/eta")
def api_queue_eta():
    """Time left for the queue (core/eta.py), with the run's progress.

    ``sec`` is None while nothing can be estimated yet. ``finish_at`` only
    while the queue runs — a paused queue finishes never.
    """
    from whisper_tm.core import eta

    snap = q.snapshot()
    est = {"sec": None, "kinds": [], "partial": False, "last_hour": {}}
    if ext.db and (snap["pending"] or snap["running"]):
        try:
            est = eta.cached_queue(ext.db, q.eta_inputs)
        except Exception as e:  # noqa — an estimate must not break the panel
            logger.warning("Queue time estimate failed: %s", e)
    now = time.time()
    return jsonify(
        {
            "ok": True,
            **est,
            "finish_at": (now + est["sec"]) if est["sec"] and not snap["paused"] else None,
            "paused": snap["paused"],
            "running": snap["running"],
            "total": snap["total"],
            "processed": snap["processed"],
            "pending": snap["pending"],
            "errors": snap["fail_count"],
            "started_at": snap["started_at"],
        }
    )


@queue_bp.route("/api/queue")
def api_queue_list():
    """Current queue state + the WAITING entries.

    'items' deliberately contains only pending — identical to the SSE
    stream. Previously list_items() ran here without a filter: the
    frontend adopted the list 1:1 as the queue (applyQueueState →
    queueItems) and thereby also showed long-finished and failed entries
    as "waiting". It also sent thousands of entries with their settings
    over the wire for a large queue. The tally of completed ones lives
    in the counters (done_count/fail_count).
    """
    snap = q.snapshot()
    snap["items"] = q.list_items("pending", limit=ITEMS_PREVIEW)
    # Include failed/cancelled entries (with error text) so the frontend
    # can name them and re-enqueue them selectively.
    snap["failed_items"] = q.list_items("error", limit=ITEMS_PREVIEW) + q.list_items(
        "cancelled", limit=ITEMS_PREVIEW
    )
    # Short names as in the SSE stream: applyQueueState() in the frontend
    # reads 'done'/'failed'/'skipped'. Without the aliases, a refresh via
    # this route reset the counters to 0 until the next stream tick
    # filled them again.
    snap["done"] = snap["done_count"]
    snap["failed"] = snap["fail_count"]
    snap["skipped"] = snap["skip_count"]
    snap["cancelled"] = snap["cancel_count"]
    snap["retryable"] = snap["retryable_count"]
    # Every running entry with its job's progress — the same list as the
    # SSE stream, so a refresh via this route doesn't drop the other lanes.
    snap["running_items"] = _running_items(snap)
    snap.setdefault("running_count", len(snap["running_items"]))
    snap.setdefault("lanes", 1)
    # System-wide warnings (e.g. separation-model fallback) — the WebUI
    # shows them as a red banner until /api/system_warnings/acknowledge
    # comes in.
    try:
        from whisper_tm.extensions import job_manager as _jm

        snap["system_warnings"] = _jm.system_warnings() if _jm else []
    except Exception:
        snap["system_warnings"] = []
    return jsonify(snap)


@queue_bp.route("/api/system_warnings/acknowledge", methods=["POST"])
def acknowledge_system_warnings():
    """Close the WebUI's red warning banner (clear warnings)."""
    from whisper_tm.extensions import job_manager as _jm

    if _jm:
        _jm.acknowledge_system_warnings()
    return jsonify({"ok": True})


@queue_bp.route("/api/queue/add", methods=["POST"])
def api_queue_add():
    """Enqueue files. Replaces the old client-side txQueue.push()."""
    data = request.json if isinstance(request.json, dict) else {}
    filenames = data.get("filenames")
    if filenames is None and data.get("filename"):
        filenames = [data["filename"]]
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"error": "No files"}), 400
    if len(filenames) > MAX_QUEUE:
        return jsonify({"error": f"Too many files ({len(filenames)}). Maximum: {MAX_QUEUE:,}"}), 400

    front = bool(data.get("front"))
    added = q.enqueue(
        filenames, _settings_from(data), data.get("base_dir"), data.get("dir_map"), front=front
    )
    # front=true comes from the add dialog (upload/link download, the
    # transcribe dialog): whoever just brought something in wants to see it
    # processed NOW. A paused queue runs just these — the rest stays paused
    # (resuming all of it would start the whole backlog).
    run_now = q.run_now([e.get("id") for e in added]) if front and added else 0
    # "items" is capped: at hundreds of thousands of enqueued files that
    # would be tens of MB of JSON the frontend doesn't even evaluate — enqueueFiles() only
    # reads `added` and reloads the state afterward via /api/queue.
    # Report fast-tracked entries separately: they're neither "newly
    # enqueued" nor "skipped". Without this number the UI reported either
    # nothing or something wrong when re-uploading the same file.
    moved_up = sum(1 for e in added if e.get("moved_up"))
    return jsonify(
        {
            "ok": True,
            "added": len(added),
            "items": added[:200],
            "items_truncated": len(added) > 200,
            "moved_up": moved_up,
            "skipped": len(filenames) - len(added),
            # The queue is paused and only these run
            "paused": q.is_paused(),
            "run_now": run_now,
        }
    )


@queue_bp.route("/api/queue/remove", methods=["POST"])
def api_queue_remove():
    """Remove one waiting entry from the transcription queue. Body: ``id``."""
    data = request.json if isinstance(request.json, dict) else {}
    try:
        item_id = int(data.get("id"))
    except (TypeError, ValueError):
        return jsonify({"error": "id required"}), 400
    ok, msg = q.remove_item(item_id)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "message": msg})


@queue_bp.route("/api/queue/move", methods=["POST"])
def api_queue_move():
    """Reorder one waiting entry. Body: ``id`` and either ``top: true`` (straight
    to the front) or ``direction`` (-1 up, 1 down).
    """
    data = request.json if isinstance(request.json, dict) else {}
    # `top`: all the way to the front instead of one position. Without this
    # path, the queue would only be theoretically sortable at hundreds of
    # thousands of entries — moving an entry from position 5000 to the front would take
    # 5000 clicks on ▲.
    if data.get("top"):
        try:
            item_id = int(data.get("id"))
        except (TypeError, ValueError):
            return jsonify({"error": "id required"}), 400
        ok, msg = q.move_to_top(item_id)
        if not ok:
            return jsonify({"error": msg}), 400
        return jsonify({"ok": True, "message": msg})
    try:
        item_id = int(data.get("id"))
        direction = int(data.get("direction"))
    except (TypeError, ValueError):
        return jsonify({"error": "id and direction (-1/1) required"}), 400
    ok, msg = q.move_item(item_id, direction)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True})


@queue_bp.route("/api/queue/cancel", methods=["POST"])
def api_queue_cancel():
    """Cancel one running entry by its queue id — also while its lane is
    still starting it and no job exists to cancel by job id."""
    data = request.json if isinstance(request.json, dict) else {}
    try:
        item_id = int(data.get("item_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "item_id required"}), 400
    if not q.cancel_item(item_id):
        return jsonify({"error": "Not running"}), 404
    return jsonify({"ok": True})


@queue_bp.route("/api/queue/clear", methods=["POST"])
def api_queue_clear():
    """Discard waiting entries; the running job is left untouched."""
    return jsonify({"ok": True, "removed": q.clear_pending()})


@queue_bp.route("/api/queue/retry", methods=["POST"])
def api_queue_retry():
    """Re-enqueue a failed or cancelled entry.

    Without 'id', ALL are re-enqueued — otherwise a failed entry was a
    dead end (no way back into the queue).
    """
    data = request.json if isinstance(request.json, dict) else {}
    # 'ids': filtered retry (quickfilter) — only the named entries.
    if isinstance(data.get("ids"), list):
        try:
            ids = [int(i) for i in data["ids"]]
        except (TypeError, ValueError):
            return jsonify({"error": "ids must be numbers"}), 400
        return jsonify({"ok": True, "requeued": q.retry_items(ids)})
    if data.get("id") is None:
        return jsonify({"ok": True, "requeued": q.retry_failed()})
    try:
        item_id = int(data["id"])
    except (TypeError, ValueError):
        return jsonify({"error": "id must be a number"}), 400
    ok, msg = q.retry_item(item_id)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "requeued": 1, "message": msg})


@queue_bp.route("/api/queue/stop", methods=["POST"])
def api_queue_stop():
    """Cancel the running job and pause the queue.

    The waiting entries are KEPT — /api/queue/clear exists for discarding
    them. A stop click must not destroy a thousand entries.
    """
    ok, msg = q.stop(clear=False)
    return jsonify({"ok": ok, "message": msg, "paused": True})


@queue_bp.route("/api/queue/abort", methods=["POST"])
def api_queue_abort():
    """Total abort: DISCARD the running job and delete ALL waiting entries.

    Afterward the queue is empty and stands still. A deliberate "get rid
    of it" button, as opposed to /api/queue/stop (pauses, keeps everything).
    """
    ok, removed = q.abort()
    return jsonify({"ok": ok, "removed": removed, "paused": False})


@queue_bp.route("/api/queue/resume", methods=["POST"])
def api_queue_resume():
    """Resume the queue.

    Returns the number of waiting entries. Without it, the frontend
    couldn't tell whether something was just starting up or whether the
    queue was simply empty — both looked identical: nothing happened.
    """
    q.resume()
    pending = 0
    try:
        pending = q.snapshot().get("pending", 0)
    except Exception as e:  # noqa — the number is a convenience, not required
        logger.debug("Could not determine queue state: %s", e)
    return jsonify({"ok": True, "paused": False, "pending": pending})


# ── Queue editor: filter, mark, bulk operations ──────────────────────
# Own namespace /api/queue/editor/*: the existing routes stay
# byte-identical. Every mutation takes EXACTLY ONE target ({ids} | {filter} |
# {all:true}), knows dry_run and expected (→ 409, nothing written) and
# only ever hits waiting entries. The translation to SQL is
# core/queue.editor_where — the only one.

_SWITCH_KEYS = (
    "diarize",
    "vocal_isolation",
    "save_audio_alongside",
    "word_timestamps",
    "export_alongside",
    "retranscribe",
    "low_vram",
    "speaker_vectors",
    "summarize",
)


def _filter_from_args(args) -> dict:
    raw: dict = {}
    if args.get("status") is not None:
        raw["status"] = [t for t in str(args.get("status")).split(",") if t.strip()]
    if args.get("dir") is not None:
        raw["dir"] = args.get("dir")
    raw["prefix"] = args.get("prefix", "")
    raw["recursive"] = args.get("recursive", "1")
    raw["q"] = args.get("q", "")
    raw["name"] = args.get("name", "")
    key = (args.get("setting_key") or "").strip()
    if key:
        raw["setting"] = {"key": key, "value": _setting_value(args.get("setting_value"))}
    filt = q.editor_normalize_filter(raw)
    if filt["setting"] and filt["setting"]["key"] not in _SETTING_KEYS:
        raise q.TargetError(f"Unknown setting key: {filt['setting']['key']}")
    return filt


def _setting_value(text):
    """GET value → true/false/null/string."""
    if text is None:
        return None
    t = str(text).strip()
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", ""):
        return None
    return t


def _target_from(data: dict) -> dict:
    """Exactly one of ids / filter / all — otherwise 400."""
    have = [k for k in ("ids", "filter", "all") if data.get(k) not in (None, False)]
    if len(have) != 1:
        raise q.TargetError("Exactly one of ids, filter, or all is required")
    ex = data.get("except_ids") or []
    if not isinstance(ex, list):
        raise q.TargetError("except_ids must be numbers")
    if have[0] == "ids":
        ids = data.get("ids")
        if not isinstance(ids, list):
            raise q.TargetError("ids must be numbers")
        if not ids:
            raise q.TargetError("ids is empty")
        if len(ids) > q.EDITOR_IDS_MAX:
            raise q.TargetError(f"Too many ids (max {q.EDITOR_IDS_MAX})")
        try:
            ids = [int(i) for i in ids]
        except (TypeError, ValueError):
            raise q.TargetError("ids must be numbers") from None
        return {"ids": ids}
    if have[0] == "filter":
        filt = data.get("filter")
        if not isinstance(filt, dict):
            raise q.TargetError("filter must be an object")
        filt = q.editor_normalize_filter(filt)
    else:
        filt = q.editor_normalize_filter({"status": ["pending"]})
    if filt["setting"] and filt["setting"]["key"] not in _SETTING_KEYS:
        raise q.TargetError(f"Unknown setting key: {filt['setting']['key']}")
    if len(ex) > q.EDITOR_EXCEPT_MAX:
        raise q.TargetError(f"Too many except_ids (max {q.EDITOR_EXCEPT_MAX})")
    try:
        ex = [int(i) for i in ex]
    except (TypeError, ValueError):
        raise q.TargetError("except_ids must be numbers") from None
    return {"filter": filt, "except_ids": ex}


def _expected_from(data: dict):
    e = data.get("expected")
    if e is None:
        return None
    try:
        return int(e)
    except (TypeError, ValueError):
        raise q.TargetError("expected must be a number") from None


def _settings_patch(data: dict) -> tuple[dict, list[str]]:
    s = data.get("settings")
    if not isinstance(s, dict):
        raise q.TargetError("settings must be an object")
    patch = {k: v for k, v in s.items() if k in _SETTING_KEYS}
    ignored = sorted(set(s) - set(_SETTING_KEYS))
    if not patch:
        raise q.TargetError("settings empty or only unknown keys")
    for k, v in patch.items():
        if v is not None and not isinstance(v, (bool, int, float, str)):
            raise q.TargetError("Values must be scalars")
        if k in _SWITCH_KEYS and v is not None and not isinstance(v, bool):
            raise q.TargetError(f"{k} must be true/false/null")
    return patch, ignored


def _editor_error(fn):
    """TargetError → 400, HitCountChanged → 409; everything else stays visible."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except q.TargetError as e:
            return jsonify({"error": str(e)}), 400
        except q.HitCountChanged as e:
            data = request.json if request.is_json and isinstance(request.json, dict) else {}
            expected = data.get("expected")
            return jsonify(
                {
                    "error": f"Match count has changed: expected {expected}, now {e.actual}",
                    "affected": e.actual,
                    "generation": q._gen(),
                }
            ), 409

    return wrapper


@queue_bp.route("/api/queue/editor/items")
@_editor_error
def api_queue_editor_items():
    """Queue editor: the entries matching a filter, paginated.

    Query: the editor filter fields plus ``offset`` and ``limit`` (default 200).
    """
    filt = _filter_from_args(request.args)
    offset = request.args.get("offset", default=0, type=int) or 0
    limit = request.args.get("limit", default=200, type=int) or 200
    return jsonify({"ok": True, **q.editor_list(filt, offset=offset, limit=limit)})


@queue_bp.route("/api/queue/editor/folders")
@_editor_error
def api_queue_editor_folders():
    """Queue editor: entry counts per folder for the current filter."""
    filt = _filter_from_args(request.args)
    return jsonify({"ok": True, **q.editor_folders(filt)})


def _mutation_data() -> dict:
    return request.json if request.is_json and isinstance(request.json, dict) else {}


@queue_bp.route("/api/queue/editor/settings", methods=["POST"])
@_editor_error
def api_queue_editor_settings():
    """Queue editor: change the transcription settings of every entry the target
    selects. Body: the target (filter or ids), the settings to set,
    ``dry_run`` for a match count first, ``expected`` to guard against a
    queue that changed in between.
    """
    data = _mutation_data()
    target = _target_from(data)
    patch, ignored = _settings_patch(data)
    r = q.editor_settings(
        target, patch, dry_run=bool(data.get("dry_run")), expected=_expected_from(data)
    )
    # ``hint`` stays in the response for the frontend; the old warning that
    # low_vram=false could not beat the global setting is obsolete — the
    # entry's choice wins (core/vram.low_vram).
    return jsonify({"ok": True, **r, "ignored_keys": ignored, "hint": None})


@queue_bp.route("/api/queue/editor/remove", methods=["POST"])
@_editor_error
def api_queue_editor_remove():
    """Queue editor: remove every entry the target selects. Body: the target,
    ``dry_run``, ``expected`` (see /api/queue/editor/settings).
    """
    data = _mutation_data()
    target = _target_from(data)
    r = q.editor_remove(target, dry_run=bool(data.get("dry_run")), expected=_expected_from(data))
    return jsonify({"ok": True, **r})


@queue_bp.route("/api/queue/editor/move", methods=["POST"])
@_editor_error
def api_queue_editor_move():
    """Queue editor: move every entry the target selects to the ``top`` or
    ``bottom``. Body: the target, ``to``, ``dry_run``, ``expected``.
    """
    data = _mutation_data()
    target = _target_from(data)
    where_to = data.get("to")
    if where_to not in ("top", "bottom"):
        raise q.TargetError("to must be 'top' or 'bottom'")
    r = q.editor_move(
        target, where_to, dry_run=bool(data.get("dry_run")), expected=_expected_from(data)
    )
    return jsonify({"ok": True, **r})


@queue_bp.route("/api/queue/editor/retry", methods=["POST"])
@_editor_error
def api_queue_editor_retry():
    """Re-enqueue the target's failed/cancelled entries —
    with dry-run and expected like every other editor action."""
    data = _mutation_data()
    target = _target_from(data)
    r = q.editor_retry(target, dry_run=bool(data.get("dry_run")), expected=_expected_from(data))
    return jsonify({"ok": True, **r})


@queue_bp.route("/api/queue/stream")
def api_queue_stream():
    """Permanent SSE stream with queue and job progress.

    Unlike before, this stream controls nothing — it only reports. If it
    breaks off (browser closed), the worker keeps going undeterred.
    """
    HEARTBEAT_INTERVAL = 30

    def stream():
        last_log_total = None
        last_heartbeat = time.time()
        last_items_sig = None
        last_failed_sig = None

        while True:
            snap = q.snapshot()
            # By the count of lines ever logged, not the ring's length:
            # that stops at its size, and a full ring then sent nothing new
            # — with lanes logging three files at once within minutes.
            log, total = snap["log"], snap.get("log_total", len(snap["log"]))
            if last_log_total is None:
                new_log = log
            else:
                fresh = total - last_log_total  # more than the ring: all of it
                new_log = log[-fresh:] if fresh > 0 else []
            last_log_total = total
            running = _running_items(snap)

            payload = {
                "running": snap["running"],
                "paused": snap["paused"],
                "solo": snap.get("solo", 0),
                "abort": snap["abort"],
                "current_file": snap["current_file"],
                "current_abs": snap.get("current_abs"),
                "current_dir": snap.get("current_dir"),
                "current_job_id": snap["current_job_id"],
                # Queue lanes: every running entry, oldest first, each with
                # its own job's progress (current_* is the first one). A
                # handful of short rows per tick — sent every tick, because
                # the progress in them moves every tick.
                "running_items": running,
                "running_count": snap.get("running_count", len(running)),
                "lanes": snap.get("lanes", 1),
                "total": snap["total"],
                "processed": snap["processed"],
                "pending": snap["pending"],
                "done": snap["done_count"],
                "failed": snap["fail_count"],
                "skipped": snap["skip_count"],
                "cancelled": snap["cancel_count"],
                "retryable": snap["retryable_count"],
                "started_at": snap["started_at"],
                "finished_at": snap["finished_at"],
                # Change counter from memory (0 DB cost): the open queue
                # editor reloads (debounced) on change — even for changes
                # that don't touch the preview's ID signature.
                "generation": snap["generation"],
            }

            # Only send the waiting list along if it has changed. The
            # stream runs permanently and ticks every second — re-
            # serializing 200 entries with their settings every second
            # and parsing them in the browser is pure waste as long as
            # nothing has moved.
            items = q.list_items("pending", limit=ITEMS_PREVIEW)
            items_sig = [i["id"] for i in items]
            if items_sig != last_items_sig:
                last_items_sig = items_sig
                payload["items"] = items

            # Same for failed/cancelled, only on change: the file list
            # marks rows from this (✕/⚠) and the quickfilters count from it.
            failed = q.list_items("error", limit=ITEMS_PREVIEW) + q.list_items(
                "cancelled", limit=ITEMS_PREVIEW
            )
            failed_sig = [i["id"] for i in failed]
            if failed_sig != last_failed_sig:
                last_failed_sig = failed_sig
                payload["failed_items"] = failed

            started = snap["started_at"]
            payload["elapsed_sec"] = round(time.time() - started, 1) if started else 0

            # Resolve ext.job_manager only here, not at import time:
            # create_app() sets the instance at runtime.
            jid = snap["current_job_id"]
            if jid and ext.job_manager:
                job = ext.job_manager.get(jid)
                if job:
                    payload["job_progress"] = job.progress
                    payload["job_message"] = job.message
                    payload["job_elapsed_sec"] = round(job.elapsed(), 1)

            if new_log:
                payload["new_log"] = new_log

            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            time.sleep(1)

            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n"
                last_heartbeat = now

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Library selection: pick files from the whole library ───────────────────
# core/library_select.py builds the query; these two routes count it and
# queue it. Queuing takes the count the user saw: if the library changed in
# between, the request fails instead of queuing something else.

#: The job kinds a selection can be queued as, and the settings they set.
_LIBRARY_JOBS = {
    "transcribe": {},
    "retranscribe": {"retranscribe": True},
    "voices": {"job": "voices"},
    "voices_refresh": {"job": "voices", "refresh": True},
}


@queue_bp.route("/api/queue/library/select", methods=["POST"])
def api_library_select():
    """Library selection: files in the database matching ``criteria`` (folder,
    status, duration, language, …), paginated by ``limit`` (max 500, 50 when
    left out, 0 for the count alone) and ``offset``.
    """
    from whisper_tm.core import library_select as ls

    data = request.json if isinstance(request.json, dict) else {}
    if not ext.db:
        return jsonify({"error": "No database"}), 503
    try:
        with ext.db.connection() as conn:
            limit = data.get("limit")
            out = ls.select(
                conn,
                data.get("criteria") or {},
                limit=max(min(int(50 if limit in (None, "") else limit), 500), 0),
                offset=max(int(data.get("offset") or 0), 0),
            )
    except ls.CriteriaError as e:
        return jsonify({"error": str(e)}), 400
    if data.get("estimate"):
        # How long transcribing them takes: the library's model (a fixed
        # part per file + a share of the audio, core/eta.py) — the same
        # estimate as the sidebar's "left" for a folder.
        from whisper_tm.core import eta

        try:
            model = eta.transcription_model(ext.db)
            out["estimate_sec"] = (
                round(model.seconds(out["count"], out["audio_sec"])) if model else None
            )
        except Exception as e:  # noqa — an estimate must not break the count
            logger.debug("Selection estimate: %s", e)
            out["estimate_sec"] = None
    return jsonify(out)


@queue_bp.route("/api/queue/library/facets")
def api_library_facets():
    """Library selection: the values the criteria can take (languages, models,
    folders, …) and the allowed media roots.
    """
    from whisper_tm.core import library_select as ls
    from whisper_tm.core.security import allowed_roots

    if not ext.db:
        return jsonify({"error": "No database"}), 503
    with ext.db.connection() as conn:
        out = ls.facets(conn)
    out["roots"] = [str(r) for r in allowed_roots()]
    return jsonify(out)


@queue_bp.route("/api/queue/library/enqueue", methods=["POST"])
def api_library_enqueue():
    """Queue every file matching ``criteria`` for a job (``job``: ``transcribe``,
    ``retranscribe``, ``voices`` — speaker vectors only — or ``voices_refresh``,
    voices made anew with their names kept). ``expected_count``
    must equal the current match count —
    the guard against queueing a different set than the one shown.

    ``front``: at the front of the queue, next up, in path order — files
    already waiting are pulled forward with them (core/queue.enqueue).
    Without it they join the end. ``settings``: the transcription options
    of the dialogs (model, language, separation, …); left out, the
    defaults under Settings apply.
    """
    from whisper_tm.core import library_select as ls
    from whisper_tm.core import paths as _paths
    from whisper_tm.core.security import allowed_roots

    data = request.json if isinstance(request.json, dict) else {}
    job = data.get("job") or "transcribe"
    if job not in _LIBRARY_JOBS:
        return jsonify({"error": f"unknown job kind: {job}"}), 400
    if not ext.db:
        return jsonify({"error": "No database"}), 503
    try:
        with ext.db.connection() as conn:
            files = ls.abs_paths(conn, data.get("criteria") or {})
    except ls.CriteriaError as e:
        return jsonify({"error": str(e)}), 400
    expected = data.get("expected_count")
    if not isinstance(expected, int) or expected != len(files):
        return jsonify(
            {
                "error": "The selection changed since it was counted "
                f"({expected} → {len(files)}). Count again, "
                "then queue.",
                "count": len(files),
            }
        ), 409
    settings = {**_settings_from(data.get("settings") or {}), **_LIBRARY_JOBS[job]}
    # Queue per allowed root, relative to it — the same funnel and checks
    # as every other way of queuing (root allowlist, duplicates).
    by_root: dict[str, list[str]] = {}
    outside = 0
    roots = sorted((_paths.canon(r) for r in allowed_roots()), key=len, reverse=True)
    for f in files:
        root = next((r for r in roots if _paths.is_inside(f, r)), None)
        if root is None:
            outside += 1
            continue
        by_root.setdefault(root, []).append(_paths.rel_from(f, root))
    front = bool(data.get("front"))
    parts = [
        (root, names[i : i + MAX_QUEUE])
        for root, names in by_root.items()
        for i in range(0, len(names), MAX_QUEUE)
    ]
    if front:
        # Each call goes before the one before it: the last part first, so
        # the files keep their order at the front.
        parts.reverse()
    added = moved_up = 0
    for root, part in parts:
        for entry in q.enqueue(part, settings, root, front=front):
            if entry.get("moved_up"):
                moved_up += 1
            else:
                added += 1
    return jsonify(
        {
            "ok": True,
            "job": job,
            "matched": len(files),
            "added": added,
            "moved_up": moved_up,
            "front": front,
            "paused": q.is_paused(),
            "already_queued": len(files) - outside - added - moved_up,
            "outside_allowed_roots": outside,
        }
    )

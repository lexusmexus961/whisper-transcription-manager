"""
LLM jobs: transcript correction and translation as background jobs.

Both tasks run chunk-by-chunk against the LLM and take minutes — they
therefore run as a job_manager job (job_type="llm") in a thread:
progress via the existing SSE route /api/status/<job_id>, cancellation
via /api/cancel/<job_id>, the browser may be closed in between.

Correction results are SUGGESTIONS (llm_edits, status=pending) and
are only applied to the transcript after review via /api/llm/edits/apply
(canonical dual write + SRT re-export). Translations are
a separate track per language (translations table + JSON backup).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core import llm_client
from whisper_tm.core.llm import correction as corr
from whisper_tm.core.llm import translation as trans
from whisper_tm.extensions import job_manager


def _db():
    """Resolve the database late.

    extensions.db is only set by create_app() (and freshly per
    app instance in tests) — a module import would lock in the first
    instance. Same pattern as core/summarize.py.
    """
    from whisper_tm.extensions import db

    return db


logger = logging.getLogger(__name__)

llm_jobs_bp = Blueprint("llm_jobs", __name__)


# ── Helpers ───────────────────────────────────────────────────────────────


def _load_transcript(filename: str):
    from whisper_tm.api.transcription import _load_transcript_data

    return _load_transcript_data(filename)


def _save_transcript(filename: str, tdata: dict):
    from whisper_tm.api.transcription import _save_transcript_data

    _save_transcript_data(filename, tdata)


def _translations_json_path(filename: str):
    """JSON backup: all translation tracks of a file in ONE file."""
    safe = filename.replace("/", "_").replace("\\", "_")
    return config.transcript_dir / f"{safe}_translations.json"


def _load_translations_json(filename: str) -> dict:
    fp = _translations_json_path(filename)
    if fp.exists():
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_translations_json(filename: str, data: dict) -> None:
    """Write the backup through a temporary file: written in place, a crash
    mid-write would truncate the backup of every track."""
    fp = _translations_json_path(filename)
    tmp = fp.with_name(fp.name + ".tmp")
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, fp)
    except OSError as e:
        logger.warning("translation JSON not writable: %s", e)
        with contextlib.suppress(OSError):
            tmp.unlink()


#: One lock per backup file: two translations of one file finishing together
#: would each read the file, add their language and write it back — the later
#: one without the other's language.
_json_locks: dict[str, threading.Lock] = {}
_json_locks_guard = threading.Lock()


def _json_lock(filename: str) -> threading.Lock:
    key = str(_translations_json_path(filename))
    with _json_locks_guard:
        return _json_locks.setdefault(key, threading.Lock())


def _store_translation_json(filename: str, language: str, entry: dict) -> None:
    """Put one language's track into the backup, the others untouched."""
    with _json_lock(filename):
        data = _load_translations_json(filename)
        data[language] = entry
        _save_translations_json(filename, data)


def _drop_translation_json(filename: str, language: str) -> bool:
    """Take one language out of the backup; False if it wasn't there."""
    with _json_lock(filename):
        data = _load_translations_json(filename)
        if language not in data:
            return False
        data.pop(language)
        _save_translations_json(filename, data)
        return True


def _llm_ready():
    """Precondition for all jobs: endpoint configured?"""
    if not config.get("llm", "endpoint", default=""):
        return jsonify(
            {"error": "LLM not configured. Please set the endpoint in Settings → AI models."}
        ), 400
    return None


def _start_job(job_kind_label: str, filename: str, work) -> str:
    """Create a job and start a thread. ``work(job)`` delivers the
    result dict for job_manager.complete()."""
    job_id = f"llm_{uuid.uuid4().hex[:10]}"
    job = job_manager.create(job_id, filename=filename, job_type="llm")
    job.update(status="processing", progress=0, message=f"{job_kind_label} starting…")

    def _run():
        try:
            result = work(job)
            job_manager.complete(job, result=result)
        except InterruptedError:
            job_manager.fail(job, "cancelled")
        except llm_client.LlmError as e:
            job_manager.fail(job, str(e)[:300])
        except Exception as e:
            logger.exception("%s failed (%s)", job_kind_label, filename)
            job_manager.fail(job, f"internal error: {str(e)[:200]}")

    threading.Thread(target=_run, daemon=True, name=f"llm-{job_kind_label}-{job_id}").start()
    return job_id


# ══════════════════════════════════════════════════════════════════════════
#  Correction
# ══════════════════════════════════════════════════════════════════════════


@llm_jobs_bp.route("/api/llm/correct", methods=["POST"])
def start_correction():
    """Generate correction suggestions (background job).

    Body: {filename, only_marked=true, model="", score_threshold?,
           logprob_threshold?}
    Returns: {job_id}
    """
    error = _llm_ready()
    if error:
        return error
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "no filename"}), 400

    tdata = _load_transcript(filename)
    if not tdata or not tdata.get("segments"):
        return jsonify({"error": "transcript not found"}), 404
    segments = tdata["segments"]

    only_marked = bool(data.get("only_marked", True))
    model = (data.get("model") or "").strip()

    def _to_float(key):
        v = data.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    score_threshold = _to_float("score_threshold")
    logprob_threshold = _to_float("logprob_threshold")

    def _work(job):
        result = corr.build_suggestions(
            segments,
            model=model,
            only_marked=only_marked,
            score_threshold=score_threshold,
            logprob_threshold=logprob_threshold,
            job=job,
        )
        # No model = the task assignment chose one; record that one
        used_model = model or llm_client.task_default_model("correction")
        result["model"] = used_model
        run_id = 0
        # Also save runs WITHOUT suggestions — the UI then shows what
        # was checked and what was discarded.
        if _db() and (result["suggestions"] or result["discarded"] or result["checked"]):
            try:
                run_id = _db().save_llm_edit_run(
                    filename, result, model=model, status="pending", media_dir=str(config.media_dir)
                )
            except Exception as e:
                logger.warning("correction run not storable: %s", e)
        return {
            "run_id": run_id,
            "suggestions": len(result["suggestions"]),
            "discarded": len(result["discarded"]),
            "checked": len(result["checked"]),
            "chunks": result["chunks"],
            "chunks_failed": result["chunks_failed"],
            "marked_segments": result["marked_segments"],
            "confirmed": result.get("confirmed", 0),
            "duration": result["duration"],
            # False = transcript carries no confidence data at all
            # (legacy stock) → UI recommends "whole transcript"
            "confidence_data": result.get("confidence_data", True),
        }

    return jsonify({"job_id": _start_job("Correction", filename, _work)})


@llm_jobs_bp.route("/api/llm/edits/<path:filename>")
def load_edits(filename):
    """Correction runs of a file: open review + history."""
    if not _db():
        return jsonify({"pending": None, "history": []})
    runs = _db().get_llm_edit_runs(filename, media_dir=str(config.media_dir))
    pending = next((r for r in runs if r["status"] == "pending"), None)
    history = [r for r in runs if r["status"] != "pending"]
    return jsonify({"pending": pending, "history": history[:20]})


def _get_run(filename: str, run_id, *, only_pending: bool = True):
    """Load a run; returns (run, error_response)."""
    if not _db():
        return None, (jsonify({"error": "no database available"}), 500)
    runs = _db().get_llm_edit_runs(filename, media_dir=str(config.media_dir))
    run = next((r for r in runs if r["id"] == run_id), None)
    if not run:
        return None, (jsonify({"error": "correction run not found"}), 404)
    if only_pending and run["status"] != "pending":
        return None, (jsonify({"error": f"run is already {run['status']}"}), 400)
    return run, None


def _set_suggestion_status(payload: dict, ids: set, status: str) -> None:
    """Track status per suggestion in the payload (open/applied/rejected).

    This keeps the changelog in the UI FULLY visible — nothing
    is removed from the list, only marked.
    """
    for v in payload.get("suggestions") or []:
        if v.get("id") in ids:
            v["status"] = status


def _finish_run_if_done(run_id: int, payload: dict) -> str:
    """Once all suggestions are decided, the run moves out of 'pending'."""
    vs = payload.get("suggestions") or []
    still_open = [v for v in vs if (v.get("status") or "open") == "open"]
    if still_open:
        _db().update_llm_edit_run(run_id, status="pending", payload=payload)
        return "pending"
    newer = "applied" if payload.get("applied") else "rejected"
    _db().update_llm_edit_run(run_id, status=newer, payload=payload)
    return newer


@llm_jobs_bp.route("/api/llm/edits/apply", methods=["POST"])
def apply_edits():
    """Apply a run's suggestions — individually or all.

    Body: {filename, run_id, ids: [segment-ids]}  — ids empty/missing = all
    still open. Partial apply is allowed: the run stays
    ``pending`` until every suggestion is decided (approve-per-line
    in the changelog).
    """
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    run_id = data.get("run_id")
    if not filename or run_id is None:
        return jsonify({"error": "filename and run_id required"}), 400

    run, error = _get_run(filename, run_id)
    if error:
        return error

    payload = dict(run.get("payload") or {})
    suggestions = payload.get("suggestions") or []
    still_open = [v for v in suggestions if (v.get("status") or "open") == "open"]

    ids = data.get("ids")
    if ids is not None and not isinstance(ids, list):
        return jsonify({"error": "field 'ids' must be a list"}), 400
    if ids:
        try:
            id_set = {int(i) for i in ids}
        except (TypeError, ValueError):
            return jsonify({"error": "invalid ids"}), 400
        selection = [v for v in still_open if v.get("id") in id_set]
    else:
        selection = still_open
    if not selection:
        return jsonify({"ok": True, "applied": 0, "status": run["status"]})

    tdata = _load_transcript(filename)
    if not tdata or not tdata.get("segments"):
        return jsonify({"error": "transcript not found"}), 404

    applied = corr.apply_edits_to(
        tdata, [{"id": v["id"], "new_value": v["new_value"]} for v in selection]
    )
    if applied:
        _save_transcript(filename, tdata)

    # Track cumulatively — a partial apply must not lose earlier edits
    # (otherwise the undo would be incomplete).
    payload["applied"] = (payload.get("applied") or []) + applied
    # Also done is what had nothing to do: if the target text already
    # sits in the segment (applied earlier, or typed that way by hand), the
    # suggestion would otherwise stay "open" forever and the run never close.
    segments = tdata.get("segments") or []
    done = {a["id"] for a in applied}
    for v in selection:
        sid = v.get("id")
        if sid in done or not isinstance(sid, int):
            continue
        if (
            0 <= sid < len(segments)
            and (segments[sid].get("text") or "").strip() == (v.get("new_value") or "").strip()
        ):
            done.add(sid)
    _set_suggestion_status(payload, done, "applied")
    status = _finish_run_if_done(run_id, payload)
    return jsonify({"ok": True, "applied": len(applied), "status": status})


@llm_jobs_bp.route("/api/llm/edits/dismiss", methods=["POST"])
def reject_edits_one_by_one():
    """Reject individual suggestions (stay in the changelog, marked).

    Body: {filename, run_id, ids: [segment-ids]}
    """
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    run_id = data.get("run_id")
    ids = data.get("ids")
    if not filename or run_id is None or not isinstance(ids, list) or not ids:
        return jsonify({"error": "filename, run_id and ids required"}), 400

    run, error = _get_run(filename, run_id)
    if error:
        return error
    try:
        id_set = {int(i) for i in ids}
    except (TypeError, ValueError):
        return jsonify({"error": "invalid ids"}), 400

    payload = dict(run.get("payload") or {})
    _set_suggestion_status(payload, id_set, "rejected")
    payload["rejected"] = sorted(set(payload.get("rejected") or []) | id_set)
    status = _finish_run_if_done(run_id, payload)
    return jsonify({"ok": True, "rejected": len(id_set), "status": status})


@llm_jobs_bp.route("/api/llm/edits/reject", methods=["POST"])
def discard_edits():
    """Discard a whole pending run."""
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    run_id = data.get("run_id")
    if not filename or run_id is None:
        return jsonify({"error": "filename and run_id required"}), 400
    if not _db():
        return jsonify({"error": "no database available"}), 500
    if not _db().update_llm_edit_run(run_id, status="rejected"):
        return jsonify({"error": "correction run not found"}), 404
    return jsonify({"ok": True})


@llm_jobs_bp.route("/api/llm/edits/undo", methods=["POST"])
def undo_edits():
    """Undo an applied run (write old texts back).

    Safety rule: a segment is only reset if its current text still
    matches the applied `new_value` — manual post-editing is
    never overwritten.
    """
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    run_id = data.get("run_id")
    if not filename or run_id is None:
        return jsonify({"error": "filename and run_id required"}), 400
    if not _db():
        return jsonify({"error": "no database available"}), 500

    run, error = _get_run(filename, run_id, only_pending=False)
    if error:
        return error
    # A still-open run (partial apply) can also be undone — then the
    # edits applied so far are simply reverted.
    if not (run.get("payload") or {}).get("applied"):
        return jsonify({"error": "this run applied nothing"}), 400

    tdata = _load_transcript(filename)
    if not tdata or not tdata.get("segments"):
        return jsonify({"error": "transcript not found"}), 404
    segments = tdata["segments"]

    back = 0
    skipped = 0
    for e in (run.get("payload") or {}).get("applied") or []:
        try:
            sid = int(e["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= sid < len(segments)):
            continue
        if (segments[sid].get("text") or "").strip() == (e.get("new_value") or ""):
            segments[sid]["text"] = e.get("old_value") or ""
            segments[sid].pop("edited", None)
            back += 1
        else:
            skipped += 1
    if back:
        _save_transcript(filename, tdata)
    # mark suggestions as open again — an undone run
    # should be decidable again in the changelog.
    payload = dict(run.get("payload") or {})
    for v in payload.get("suggestions") or []:
        if v.get("status") == "applied":
            v["status"] = "open"
    payload["applied"] = []
    _db().update_llm_edit_run(run_id, status="undone", payload=payload)
    return jsonify({"ok": True, "reverted": back, "skipped": skipped})


# ══════════════════════════════════════════════════════════════════════════
#  Translation
# ══════════════════════════════════════════════════════════════════════════


@llm_jobs_bp.route("/api/llm/translate", methods=["POST"])
def start_translation():
    """Translate a transcript (background job).

    Body: {filename, language, model=""}
    Returns: {job_id}
    """
    error = _llm_ready()
    if error:
        return error
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    language = (data.get("language") or "").strip()
    if not filename or not language:
        return jsonify({"error": "filename and language required"}), 400

    tdata = _load_transcript(filename)
    if not tdata or not tdata.get("segments"):
        return jsonify({"error": "transcript not found"}), 404
    segments = tdata["segments"]
    model = (data.get("model") or "").strip()

    def _work(job):
        t0 = time.time()
        result = trans.translate(segments, language, model=model, job=job)
        track = result["translations"]
        if not track:
            raise llm_client.LlmAnswerError("translation delivered no segments")
        used_model = model or llm_client.task_default_model("translation")
        saved = False
        if _db():
            try:
                saved = _db().replace_translation(
                    filename, language, track, model=used_model, media_dir=str(config.media_dir)
                )
            except Exception as e:
                logger.warning("translation not storable in DB: %s", e)
        # JSON backup (same pattern as summaries: DB primary, JSON heals)
        _store_translation_json(
            filename, language, {"model": used_model, "created_at": time.time(), "segments": track}
        )
        return {
            "language": language,
            "segments": len(track),
            "total": len(segments),
            "missing": len(result["missing"]),
            "missing_ids": result["missing"][:50],
            "chunks": result["chunks"],
            "chunks_failed": result["chunks_failed"],
            "duration": round(time.time() - t0, 1),
            "db": saved,
        }

    return jsonify({"job_id": _start_job(f"Translation → {language}", filename, _work)})


@llm_jobs_bp.route("/api/translations")
@llm_jobs_bp.route("/api/translations/<path:filename>")
def list_translations(filename=None):
    """Existing translation languages of a file (DB, JSON heals) — the
    path also as ``?file=`` (a full one, within the allowed roots)."""
    from whisper_tm.api.transcription import _file_param

    filename = filename or _file_param()
    languages = []
    if _db():
        try:
            languages = _db().get_translation_languages(filename, media_dir=str(config.media_dir))
        except Exception as e:
            logger.debug("get_translation_languages: %s", e)
    if not languages:
        all_items = _load_translations_json(filename)
        for lang, entry in sorted(all_items.items()):
            languages.append(
                {
                    "language": lang,
                    "model": entry.get("model", ""),
                    "created_at": entry.get("created_at", 0),
                }
            )
            # heal the DB
            if _db() and entry.get("segments"):
                try:
                    _db().replace_translation(
                        filename,
                        lang,
                        entry["segments"],
                        model=entry.get("model", ""),
                        media_dir=str(config.media_dir),
                    )
                except Exception:
                    pass
    return jsonify({"translations": languages})


@llm_jobs_bp.route("/api/translation/<language>")
@llm_jobs_bp.route("/api/translation/<language>/<path:filename>")
def load_translation(language, filename=None):
    """Load a translation track: {segments: [{id, text}]} — the path also
    as ``?file=`` (a full one, within the allowed roots)."""
    from whisper_tm.api.transcription import _file_param

    filename = filename or _file_param()
    track = None
    if _db():
        try:
            track = _db().get_translation(filename, language, media_dir=str(config.media_dir))
        except Exception as e:
            logger.debug("get_translation: %s", e)
    if track is None:
        entry = _load_translations_json(filename).get(language) or {}
        track = entry.get("segments")
    if track is None:
        return jsonify({"error": "translation not found"}), 404
    return jsonify({"language": language, "segments": track})


@llm_jobs_bp.route("/api/translations/delete", methods=["POST"])
def delete_translation():
    """Delete one stored translation of a transcript. Body: ``filename``,
    ``language``.
    """
    data = request.json if isinstance(request.json, dict) else {}
    filename = (data.get("filename") or "").strip()
    language = (data.get("language") or "").strip()
    if not filename or not language:
        return jsonify({"error": "filename and language required"}), 400
    deleted = False
    if _db():
        try:
            deleted = _db().delete_translation(filename, language, media_dir=str(config.media_dir))
        except Exception as e:
            logger.debug("delete_translation: %s", e)
    if _drop_translation_json(filename, language):
        deleted = True
    if not deleted:
        return jsonify({"error": "translation not found"}), 404
    return jsonify({"ok": True})

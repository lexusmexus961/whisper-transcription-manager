"""Whisper artifacts in the transcripts that exist: dry run, apply, undo.

The work is in ``core/artifact_cleanup.py``; these endpoints start it in the
background and report how far it is. Settings → Transcription → Clean-up
drives them.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core import artifact_cleanup as cleanup

artifacts_bp = Blueprint("artifacts", __name__)


@artifacts_bp.route("/api/artifacts/status")
def artifacts_status():
    """What runs now, how far it is, the latest dry run and cleanup, and the
    cleanup logs that can be undone."""
    return jsonify(cleanup.status())


@artifacts_bp.route("/api/artifacts/dry_run", methods=["POST"])
def artifacts_dry_run():
    """Count what the patterns would remove from the existing transcripts.

    Body: optional ``patterns`` (lines as typed on the page) — without them
    the saved ``transcription.hallucination_patterns``. Reads only.
    """
    data = request.get_json(silent=True) or {}
    patterns = data.get("patterns")
    if patterns is None:
        patterns = config.get("transcription", "hallucination_patterns", default=[]) or []
    if isinstance(patterns, str):
        patterns = patterns.splitlines()
    patterns = [str(p).strip() for p in patterns if str(p).strip()]
    from whisper_tm.core.helpers import compile_artifact_patterns

    _, errors = compile_artifact_patterns(patterns)
    if errors:
        return jsonify({"error": "not a valid regular expression — " + "; ".join(errors)}), 400
    if not cleanup.start("dry_run", cleanup.dry_run, patterns):
        return jsonify({"error": "Another cleanup step is running"}), 409
    return jsonify({"ok": True})


@artifacts_bp.route("/api/artifacts/apply", methods=["POST"])
def artifacts_apply():
    """Remove what one dry run found. Body: ``run_id`` of that dry run.

    Backs the database up first and logs every removed segment, so
    ``/api/artifacts/undo`` can put it back.
    """
    data = request.get_json(silent=True) or {}
    run_id = str(data.get("run_id") or "")
    if not run_id:
        return jsonify({"error": "run_id missing — run the dry run first"}), 400
    if not cleanup.start("apply", cleanup.apply, run_id):
        return jsonify({"error": "Another cleanup step is running"}), 409
    return jsonify({"ok": True})


@artifacts_bp.route("/api/artifacts/undo", methods=["POST"])
def artifacts_undo():
    """Put the segments of one cleanup back. Body: ``log`` (its file name)."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("log") or "")
    if not name:
        return jsonify({"error": "log missing"}), 400
    if not cleanup.start("undo", cleanup.undo, name):
        return jsonify({"error": "Another cleanup step is running"}), 409
    return jsonify({"ok": True})


@artifacts_bp.route("/api/artifacts/stop", methods=["POST"])
def artifacts_stop():
    """Stop the running step after the current transcript."""
    return jsonify({"ok": cleanup.stop()})

"""Analyses — one task over many recordings (core/analysis.py).

``POST /api/analyses/plan`` counts a set before anything runs (recordings,
words, parts, roughly how long); ``POST /api/analyses`` queues the analysis
in the LLM queue and answers at once; the page then asks
``GET /api/analyses/<id>`` until it is done.
"""

from __future__ import annotations

import io
import re

from flask import Blueprint, jsonify, request, send_file

from whisper_tm.core import analysis

analysis_bp = Blueprint("analysis", __name__)


def _body() -> dict:
    d = request.get_json(silent=True)
    return d if isinstance(d, dict) else {}


def _fail(e: analysis.AnalysisError):
    return jsonify({"error": str(e)}), (404 if str(e) == "unknown analysis" else 400)


@analysis_bp.route("/api/analyses")
def analyses_list():
    """The analyses, newest first, and the task templates."""
    return jsonify({"analyses": analysis.listing(), "presets": analysis.PRESETS})


@analysis_bp.route("/api/analyses/for")
def analyses_for_file():
    """The analyses that read one recording, newest first.
    Query: ``file`` — a name in the open folder or an absolute path."""
    from whisper_tm.config import config
    from whisper_tm.core.helpers import transcript_identity

    abs_path = transcript_identity(request.args.get("file", ""), media_dir=str(config.media_dir))
    if not abs_path:
        return jsonify({"error": "Path not resolvable"}), 400
    return jsonify({"analyses": analysis.analyses_of_file(abs_path)})


@analysis_bp.route("/api/analyses/plan", methods=["POST"])
def analyses_plan():
    """What an analysis of this set would read. Body: ``source``, ``order``,
    ``speakers``."""
    d = _body()
    try:
        return jsonify(
            analysis.plan(
                d.get("source") or {},
                str(d.get("order") or "folder"),
                bool(d.get("speakers", True)),
            )
        )
    except analysis.AnalysisError as e:
        return _fail(e)


@analysis_bp.route("/api/analyses", methods=["POST"])
def analyses_create():
    """Start an analysis. Body: ``source``, ``task``, and optionally
    ``title``, ``order`` (folder | name | dialogue), ``speakers``, ``merge``
    (one answer from the notes of all parts), ``model`` ('' = the summary
    task's model), ``preset``."""
    d = _body()
    try:
        a = analysis.create(
            d.get("source") or {},
            str(d.get("task") or ""),
            title=str(d.get("title") or ""),
            order=str(d.get("order") or "folder"),
            speakers=bool(d.get("speakers", True)),
            merge=bool(d.get("merge", True)),
            model=str(d.get("model") or ""),
            preset=str(d.get("preset") or ""),
        )
    except analysis.AnalysisError as e:
        return _fail(e)
    return jsonify({"ok": True, "analysis": analysis.public(a)})


@analysis_bp.route("/api/analyses/<aid>")
def analyses_get(aid):
    """One analysis: its answer, the notes of each part, the passages."""
    try:
        return jsonify(analysis.public(analysis.load(aid)))
    except analysis.AnalysisError as e:
        return _fail(e)


@analysis_bp.route("/api/analyses/<aid>/files")
def analyses_files(aid):
    """The set's files (the first ``limit``, 500 by default)."""
    try:
        files = analysis.files_of(aid)
        analysis.load(aid)
    except analysis.AnalysisError as e:
        return _fail(e)
    try:
        limit = max(1, min(int(request.args.get("limit", 500)), 20000))
    except ValueError:
        limit = 500
    return jsonify({"files": files[:limit], "total": len(files)})


@analysis_bp.route("/api/analyses/<aid>", methods=["DELETE"])
def analyses_delete(aid):
    """Delete an analysis — a running one is cancelled first."""
    try:
        if not analysis.delete(aid):
            return jsonify({"error": "unknown analysis"}), 404
        return jsonify({"ok": True})
    except analysis.AnalysisError as e:
        return _fail(e)


@analysis_bp.route("/api/analyses/<aid>/cancel", methods=["POST"])
def analyses_cancel(aid):
    """Cancel — a part being read still finishes."""
    try:
        return jsonify({"ok": analysis.cancel(aid)})
    except analysis.AnalysisError as e:
        return _fail(e)


@analysis_bp.route("/api/analyses/<aid>/rerun", methods=["POST"])
def analyses_rerun(aid):
    """Run it again from the start; a folder is looked at again."""
    try:
        return jsonify({"ok": True, "analysis": analysis.public(analysis.rerun(aid))})
    except analysis.AnalysisError as e:
        return _fail(e)


@analysis_bp.route("/api/analyses/<aid>/export")
def analyses_export(aid):
    """The analysis as a Markdown file — answer, notes, passages."""
    try:
        a = analysis.load(aid)
    except analysis.AnalysisError as e:
        return _fail(e)
    name = re.sub(r"[^\w\- ]+", "", a.get("title") or "analysis").strip()[:80] or "analysis"
    # send_file writes a name outside Latin-1 as filename*=UTF-8''… — a header
    # built by hand raised UnicodeEncodeError and the request hung
    return send_file(
        io.BytesIO(analysis.markdown(a).encode("utf-8")),
        mimetype="text/markdown",
        as_attachment=True,
        download_name=f"{name}.md",
    )


@analysis_bp.route("/api/analyses/<aid>/ask", methods=["POST"])
def analyses_ask(aid):
    """A conversation in "Ask" with this analysis attached."""
    try:
        return jsonify({"ok": True, "conversation": analysis.ask(aid)})
    except analysis.AnalysisError as e:
        return _fail(e)

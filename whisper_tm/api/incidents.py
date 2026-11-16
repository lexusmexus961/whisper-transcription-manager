"""Read incidents — what happened when with which file.

GET /api/incidents
    ?file=<relative or absolute path>       only this file
    &kind=warning|error|oom|retry|system
    &days=<n>                               only the last n days
    &q=<text>                               full text in text/step
    &limit=200&offset=0

The underlying table (library.db, `incidents`, schema v19) is never
truncated — see core/incidents.py.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from whisper_tm.core import incidents as _incidents
from whisper_tm.extensions import db

logger = logging.getLogger(__name__)

incidents_bp = Blueprint("incidents", __name__)


def _db():
    import whisper_tm.extensions as ext

    return ext.db or db


@incidents_bp.route("/api/incidents")
def api_incidents():
    """Recorded incidents (warnings, errors, OOM waits, retries), newest first.

    Query: ``file`` (absolute path or name), ``kind``, ``days``, ``q`` (text
    search), ``limit`` (default 200), ``offset``.
    """
    d = _db()
    if d is None:
        return jsonify({"error": "No database"}), 503
    file = (request.args.get("file") or "").strip()
    kind = (request.args.get("kind") or "").strip().lower() or None
    if kind and kind not in _incidents.KINDS:
        return jsonify({"error": f"kind must be one of {list(_incidents.KINDS)}"}), 400
    days = request.args.get("days", type=float)
    q = (request.args.get("q") or "").strip() or None
    limit = request.args.get("limit", default=200, type=int)
    offset = request.args.get("offset", default=0, type=int)
    from whisper_tm.core import paths

    abs_path = paths.canon(file) if paths.is_absolute(file) else None
    filename = None if abs_path else (file or None)
    try:
        out = d.incidents(
            abs_path=abs_path,
            filename=filename,
            kind=kind,
            since=_incidents.since(days) if days else None,
            search=q,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.warning("Reading incidents: %s", e)
        return jsonify({"error": str(e)}), 500
    out["limit"] = max(1, min(limit, 5000))
    out["offset"] = max(0, offset)
    return jsonify(out)

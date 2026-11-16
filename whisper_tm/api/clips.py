"""
The Clips tab: a recording's cuts (kept on the server, core/cuts.py),
making a clip from them as a job (core/clip_render.py through
core/export_jobs.py, kind "clip") and Best moments (an LLM queue task,
core/best_moments.py).

The recording is named by ``file``: absolute, or relative to the library
folder; it must lie in the allowed trees.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from whisper_tm.core import cuts as cut_store

logger = logging.getLogger(__name__)

clips_bp = Blueprint("clips", __name__)


class _Bad(ValueError):
    pass


def _file(value) -> str:
    from whisper_tm.api.export import _abs
    from whisper_tm.core.security import root_allowed

    a = _abs(str(value or ""))
    if not a or not root_allowed(a):
        raise _Bad("a recording inside the library is needed")
    return a


def _db():
    from whisper_tm.extensions import db

    return db


def _transcript(abs_path: str) -> dict | None:
    db = _db()
    return db.get_transcript(abs_path) if db else None


def _length(abs_path: str, data: dict | None) -> float:
    if data:
        try:
            return float(data.get("total_duration_sec") or 0)
        except (TypeError, ValueError):
            pass
    return 0.0


def _rel_for_queue(abs_path: str) -> str:
    """The name LLM queue entries go by: below the open folder if it is
    there, the absolute path otherwise."""
    from whisper_tm.config import config

    root = str(config.media_dir).rstrip("/")
    return abs_path[len(root) + 1 :] if abs_path.startswith(root + "/") else abs_path


@clips_bp.route("/api/clips")
def clips_get():
    """A recording's cuts, how its clip is to be made, and what was made."""
    try:
        a = _file(request.args.get("file"))
        return jsonify(cut_store.load(a))
    except (_Bad, cut_store.CutsError) as e:
        return jsonify({"error": str(e)}), 400


@clips_bp.route("/api/clips", methods=["PUT"])
def clips_put():
    """Replace the cuts and/or the settings. Body: {file, cuts?, settings?}."""
    body = request.get_json(silent=True) or {}
    try:
        a = _file(body.get("file"))
        d = cut_store.save(a, body.get("cuts"), body.get("settings"), _length(a, _transcript(a)))
        return jsonify(d)
    except (_Bad, cut_store.CutsError) as e:
        return jsonify({"error": str(e)}), 400


def _lines(data: dict | None) -> list[dict]:
    if not data:
        return []
    from whisper_tm.core import exporter as ex

    return ex.build_lines(data, ex.options({"names": False}))


@clips_bp.route("/api/clips/render", methods=["POST"])
def clips_render():
    """Make the clip(s) from the saved cuts and settings — a job (kind
    "clip") in the Queue, with progress, cancel and the download."""
    from whisper_tm.core import clip_render as cr
    from whisper_tm.core import export_jobs as jobs

    body = request.get_json(silent=True) or {}
    try:
        a = _file(body.get("file"))
        state = cut_store.load(a)
    except (_Bad, cut_store.CutsError) as e:
        return jsonify({"error": str(e)}), 400
    cuts, settings = state["cuts"], state["settings"]
    if not cuts:
        return jsonify({"error": "no cuts yet"}), 400
    src = Path(a)
    if not src.is_file():
        return jsonify({"error": "the recording is not there"}), 404
    data = _transcript(a)
    lines = _lines(data) if settings["subs"] else None
    video = cr.has_video(src)
    stem = src.stem
    if settings["stitch"] == "each" and len(cuts) > 1:
        items = [{"name": f"{stem}_clip{i + 1}", "cuts": [c]} for i, c in enumerate(cuts)]
    else:
        items = [{"name": f"{stem}_clip", "cuts": cuts}]

    def work(item, out_dir, progress, should_stop):
        parts = cr.plan(src, item["cuts"], settings)
        return [
            cr.render(
                src, parts, out_dir / item["name"], settings, lines, progress, should_stop, video
            )
        ]

    length = sum(c["end"] - c["start"] for c in cuts)
    fmt = (
        {"orig": "16:9", "vert": "vertical", "square": "square"}[settings["format"]]
        if video
        else "sound"
    )

    def done(job):
        cut_store.note_made(
            a,
            {
                "at": time.time(),
                "job": job.id,
                "n": len(items),
                "len": round(length, 1),
                "format": settings["format"] if video else "audio",
                "result": job.result,
                "size": job.size,
            },
        )

    m, s = divmod(int(length), 60)
    label = f"Clip · {src.name} · {m}:{s:02d} · {fmt}" + (
        f" · {len(items)} clips" if len(items) > 1 else ""
    )
    try:
        job = jobs.start(
            label, items, work, kind="clip", on_done=done, zip_name=f"{stem}_clips.zip"
        )
    except jobs.Refused as e:
        return jsonify({"error": str(e)}), e.status
    return jsonify({"job": job.public()}), 202


@clips_bp.route("/api/clips/best", methods=["POST"])
def clips_best():
    """Best moments: an LLM queue task that adds three moments as cuts.
    Body: {file}. Answers the queue entry's id."""
    from whisper_tm.config import config
    from whisper_tm.core.llm_queue import TYPE_BEST_MOMENTS, llm_queue

    body = request.get_json(silent=True) or {}
    try:
        a = _file(body.get("file"))
    except _Bad as e:
        return jsonify({"error": str(e)}), 400
    if not config.get("llm", "endpoint", default=""):
        return jsonify({"error": "No language model is set up (Settings → AI models)"}), 400
    if not _transcript(a):
        return jsonify({"error": "not transcribed yet"}), 400
    eid = llm_queue.enqueue(TYPE_BEST_MOMENTS, _rel_for_queue(a), {"abs": a}, front=True)
    if not eid:
        return jsonify({"error": "the AI queue is not available"}), 503
    return jsonify({"id": eid}), 202

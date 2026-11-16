"""
Export: one way out for transcripts, subtitles, documents, video and
audio — one recording or any set of them.

A selection is ``files`` (absolute, or relative to the library folder)
and/or ``folders`` (``{path, subfolders}``), optionally only what was
``added`` today, yesterday or in the last 7 / 30 days. Text formats
(core/exporter.py) come back at once for up to a few hundred recordings;
video and audio (core/media_export.py)
and bigger text exports run as jobs (core/export_jobs.py) with progress,
cancel and a download when done.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from whisper_tm.config import VIDEO_EXTENSIONS
from whisper_tm.core import export_jobs as jobs
from whisper_tm.core import exporter as ex
from whisper_tm.core import media_export as mx
from whisper_tm.core import paths

logger = logging.getLogger(__name__)

export_bp = Blueprint("export", __name__)

#: More recordings than this in a text export: a job instead of a download
DIRECT_LIMIT = 300
MAX_FILES = 20000
MEDIA_KINDS = ("video", "audio")


class _Bad(ValueError):
    pass


def _db():
    from whisper_tm.extensions import db

    if db is None:
        raise _Bad("No database configured")
    return db


def _added_range(added: str) -> tuple[float | None, float | None]:
    if not added:
        return None, None
    day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if added == "today":
        return day.timestamp(), None
    if added == "yesterday":
        return (day - timedelta(days=1)).timestamp(), day.timestamp()
    if added in ("7", "30"):
        return (day - timedelta(days=int(added) - 1)).timestamp(), None
    raise _Bad("added: today, yesterday, 7 or 30")


def _abs(p: str) -> str:
    """An absolute path, or one relative to the library folder."""
    p = str(p or "").strip()
    if p.startswith("/") or (len(p) > 2 and p[1] == ":"):
        return paths.canon(p)
    from whisper_tm.config import config

    root = str(config.media_dir)
    if ".." in p.replace("\\", "/").split("/"):
        return ""
    return paths.canon(root.rstrip("/") + ("/" + p.strip("/") if p.strip("/") else ""))


def selection(body: dict, capped: bool = True) -> tuple[list[dict], str]:
    """The recordings a selection names: [{abs, name, rel, status, video}]
    in a stable order, and the folder their ZIP paths start from.
    ``capped``: more than MAX_FILES is refused (an export); the count and the
    picker see any number."""
    from whisper_tm.core.security import root_allowed

    db = _db()
    files = body.get("files") or []
    folders = body.get("folders") or []
    if not isinstance(files, list) or not isinstance(folders, list):
        raise _Bad("files and folders must be lists")
    if not all(isinstance(f, dict) for f in folders):
        raise _Bad("folders: each one is {path, subfolders}")
    since, until = _added_range(str(body.get("added") or ""))
    rows: dict[str, dict] = {}
    with db.connection() as conn:

        def take(r):
            created = r["created_at"] or 0
            if since is not None and created < since:
                return
            if until is not None and created >= until:
                return
            # Not indexed yet (is_video 0): the extension tells
            rows[r["abs_path"]] = {
                "abs": r["abs_path"],
                "status": r["status"] or "pending",
                "video": bool(r["is_video"])
                or Path(r["abs_path"]).suffix.lower() in VIDEO_EXTENSIONS,
            }

        cols = "abs_path, status, is_video, created_at"
        for f in folders:
            path = _abs((f or {}).get("path") or "")
            if not path or not root_allowed(path):
                raise _Bad("a folder outside the library")
            lo = path.rstrip("/") + "/"
            for r in conn.execute(
                f"SELECT {cols} FROM media_files WHERE abs_path >= ? AND abs_path < ?",
                (lo, lo[:-1] + "0"),
            ):
                if not (f or {}).get("subfolders", True) and "/" in r["abs_path"][len(lo) :]:
                    continue
                take(r)
        for a in files:
            a = _abs(a)
            if not a or not root_allowed(a):
                raise _Bad("a file outside the library")
            r = conn.execute(f"SELECT {cols} FROM media_files WHERE abs_path = ?", (a,)).fetchone()
            if r:
                take(r)
            else:
                rows[a] = {
                    "abs": a,
                    "status": "done",
                    "video": Path(a).suffix.lower() in VIDEO_EXTENSIONS,
                }
    if capped and len(rows) > MAX_FILES:
        raise _Bad(f"more than {MAX_FILES:,} recordings — choose fewer")
    out = sorted(rows.values(), key=lambda x: x["abs"].lower())
    # The ZIP's folders start where the selection does
    base = ""
    if out:
        parents = [str(Path(x["abs"]).parent) for x in out]
        base = parents[0]
        for p in parents[1:]:
            while base and not (p == base or p.startswith(base.rstrip("/") + "/")):
                base = str(Path(base).parent) if base != "/" else ""
    for x in out:
        x["name"] = Path(x["abs"]).name
        x["rel"] = x["abs"][len(base.rstrip("/")) + 1 :] if base else x["name"]
    return out, base


def _doc(item: dict, opts: dict) -> ex.Doc | None:
    """A recording as an export document — None without a transcript."""
    db = _db()
    data = db.get_transcript(item["abs"])
    if not data:
        return None
    translation, tlang = None, ""
    if opts["what"] in ("translation", "both"):
        langs = [x.get("language") for x in (db.get_translation_languages(item["abs"]) or [])]
        tlang = opts["language"] if opts["language"] in langs else (langs[0] if langs else "")
        if tlang:
            translation = db.get_translation(item["abs"], tlang) or []
    summary = ""
    if opts["what"] == "summary" or opts["summary_on_top"]:
        sums = db.get_summaries(item["abs"]) or []
        summary = (sums[-1].get("text") or "") if sums else ""
    cuts = None
    if opts["part"] == "cuts":
        try:
            from whisper_tm.core import cuts as cut_store

            cuts = [(c["start"], c["end"]) for c in cut_store.get(item["abs"])]
        except ImportError:
            cuts = []
    lines = ex.build_lines(data, opts, translation, cuts)
    tx_date = 0.0
    try:
        with db.connection() as conn:
            r = conn.execute(
                "SELECT updated_at FROM transcripts WHERE abs_path = ? AND complete = 1",
                (item["abs"],),
            ).fetchone()
            tx_date = float(r["updated_at"] or 0) if r else 0.0
    except Exception:  # noqa — the header just goes without the date
        pass
    return ex.Doc(
        name=item["name"],
        rel=item["rel"],
        title=item["name"],
        length=float(data.get("total_duration_sec") or 0),
        language=data.get("language") or "",
        transcribed=tx_date,
        lines=lines,
        summary=summary,
        translation_language=tlang,
    )


def _text_export(items: list[dict], fmt: str, opts: dict) -> tuple[str, bytes, str, list]:
    docs, skipped = [], []
    for it in items:
        d = _doc(it, opts)
        if d is None:
            skipped.append([it["name"], "no transcript"])
            continue
        docs.append(d)
    name, data, mime = ex.render(docs, fmt, opts)
    return name, data, mime, skipped


def _media_work(kind: str, mopts: dict, topts: dict):
    """One recording through ffmpeg (for an export job)."""

    def work(item, out_dir: Path, progress, should_stop):
        src = Path(item["abs"])
        if not src.is_file():
            raise mx.MediaExportError("the file is not there")
        db = _db()
        data = (
            db.get_transcript(item["abs"])
            if (
                mopts.get("speakers")
                or mopts.get("remove_pauses")
                or mopts.get("subtitles")
                or mopts.get("with_transcript")
            )
            else None
        )
        parts = None
        if data and (mopts.get("speakers") or mopts.get("remove_pauses")):
            parts = mx.spans(
                data.get("segments") or [],
                mopts.get("speakers") or None,
                data.get("speaker_names") or {},
            )
            if not parts:
                raise mx.MediaExportError("nothing left after choosing the parts")
        elif mopts.get("speakers") or mopts.get("remove_pauses"):
            raise mx.MediaExportError("no transcript to find the parts by")
        made = []
        srt = None
        if kind == "video" and data and (mopts.get("subtitles") or mopts.get("with_transcript")):
            doc = ex.Doc(
                name=item["name"],
                rel=item["rel"],
                title=item["name"],
                lines=ex.build_lines(data, topts),
            )
            if parts:
                # Times move up with the cut-out pauses: shift each cue
                doc.lines = _shift(doc.lines, parts)
            srt = out_dir / f"{src.stem}.srt"
            srt.write_text(ex.render_srt(doc, topts), encoding="utf-8")
            if mopts.get("with_transcript"):
                made.append(srt)
        plan = (
            mx.plan_video(src, out_dir, mopts, parts, srt)
            if kind == "video"
            else mx.plan_audio(src, out_dir, mopts, parts)
        )
        made.insert(0, mx.run(plan, progress, should_stop))
        if srt and not mopts.get("with_transcript"):
            srt.unlink(missing_ok=True)
        return made

    return work


def _shift(lines: list[dict], parts: list[tuple[float, float]]) -> list[dict]:
    """Lines on the timeline of the kept parts only."""
    out = []
    offset_at = []
    acc = 0.0
    for a, b in parts:
        offset_at.append((a, b, acc))
        acc += b - a
    for ln in lines:
        for a, b, acc0 in offset_at:
            if ln["start"] < b and ln["end"] > a:
                s = max(ln["start"], a) - a + acc0
                e = min(ln["end"], b) - a + acc0
                out.append({**ln, "start": s, "end": e})
                break
    return out


def _parse() -> tuple[dict, str, dict, dict]:
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        raise _Bad("a JSON object is needed")
    fmt = str(body.get("format") or "txt")
    # A wrong type is the sender's mistake (400), not the server's (500)
    for key in ("options", "video", "audio"):
        if body.get(key) is not None and not isinstance(body[key], dict):
            raise _Bad(f"{key}: an object is needed")
    topts = ex.options(body.get("options") or {})
    mopts = {}
    if fmt == "video":
        mopts = mx.video_options(body.get("video") or {})
    elif fmt == "audio":
        mopts = mx.audio_options(body.get("audio") or {})
    elif fmt not in ex.TEXT_FORMATS:
        raise _Bad(f"format: {', '.join(ex.TEXT_FORMATS + MEDIA_KINDS)}")
    if mopts:
        # A video's subtitles follow the media choice of speakers, not the
        # text options' (which belong to text exports)
        topts = {
            **topts,
            "what": "transcript",
            "part": "whole",
            "lines_with": "",
            "speakers": list(mopts.get("speakers") or []),
        }
    return body, fmt, topts, mopts


def _fail(e: Exception):
    return jsonify({"error": str(e)}), 400


def _download_name(name: str) -> str:
    """A name for a download header: a line break or another control
    character in a recording's name (a file system allows them) is no part
    of a header."""
    return "".join(" " if ch < " " or ch == "\x7f" else ch for ch in str(name)).strip() or "export"


@export_bp.route("/api/export/count", methods=["POST"])
def export_count():
    """What a selection holds: recordings, transcribed ones, videos, their
    size on disk, and a few names. Body: the selection."""
    try:
        items, base = selection(request.get_json(silent=True) or {}, capped=False)
    except (_Bad, ex.ExportError) as e:
        return _fail(e)
    size = 0
    for it in items[:5000]:
        try:
            size += Path(it["abs"]).stat().st_size
        except OSError:
            pass
    return jsonify(
        {
            "count": len(items),
            "transcribed": sum(1 for i in items if i["status"] == "done"),
            "videos": sum(1 for i in items if i["video"]),
            "size": size,
            "base": base,
            "names": [i["rel"] for i in items[:8]],
            "too_many": len(items) > MAX_FILES,
            "max": MAX_FILES,
        }
    )


@export_bp.route("/api/export/files", methods=["POST"])
def export_files():
    """The recordings of a selection, for the export page's picker: each
    one's path, name below the selection's folder, status and whether it is
    a video — the first 3,000, filtered by ``q`` (in the name)."""
    body = request.get_json(silent=True) or {}
    try:
        items, base = selection(body, capped=False)
    except (_Bad, ex.ExportError) as e:
        return _fail(e)
    q = str(body.get("q") or "").strip().lower()
    if q:
        items = [i for i in items if q in i["rel"].lower()]
    return jsonify(
        {
            "total": len(items),
            "base": base,
            "files": [
                {"abs": i["abs"], "rel": i["rel"], "status": i["status"], "video": i["video"]}
                for i in items[:3000]
            ],
        }
    )


@export_bp.route("/api/export/preview", methods=["POST"])
def export_preview():
    """The first recording of a selection in a text format, as text (Word
    and PDF are shown as their plain text). Body: selection, format, options."""
    try:
        body, fmt, topts, _ = _parse()
        items, _base = selection(body, capped=False)
    except (_Bad, ex.ExportError, mx.MediaExportError) as e:
        return _fail(e)
    first = next((i for i in items if i["status"] == "done"), None)
    if not first:
        return jsonify(
            {"text": "", "name": "", "note": "No recording with a transcript in the selection."}
        )
    doc = _doc(first, topts)
    if doc is None:
        return jsonify({"text": "", "name": first["name"], "note": "No transcript."})
    shown = fmt if fmt in ex.RENDER_ONE else "txt"
    if fmt in MEDIA_KINDS:
        shown = "srt"
    text = ex.RENDER_ONE[shown](doc, topts)
    note = {
        "docx": "A Word document with this text.",
        "pdf": "A PDF with this text (standard font: characters outside Latin-1 are replaced).",
    }.get(fmt, "")
    if fmt in MEDIA_KINDS:
        note = "The subtitles of the first recording (burned in or beside the video when chosen)."
    # size: the whole first file, not the shown part — the page estimates
    # the download from it
    return jsonify(
        {
            "text": text[:6000],
            "name": first["name"],
            "truncated": len(text) > 6000,
            "note": note,
            "size": len(text.encode("utf-8")),
        }
    )


@export_bp.route("/api/export", methods=["POST"])
def export_run():
    """Export a selection. Body: the selection, ``format`` (txt, srt, vtt,
    md, csv, json, docx, pdf, video, audio), ``options`` (the text options),
    ``video`` / ``audio`` (the media options).

    Text for up to a few hundred recordings: the file itself (one file, or
    a ZIP with a file each). Video, audio and bigger text exports: a job —
    ``{"job": {...}}`` — to follow under /api/export/jobs.
    """
    try:
        body, fmt, topts, mopts = _parse()
        items, _base = selection(body)
    except (_Bad, ex.ExportError, mx.MediaExportError) as e:
        return _fail(e)
    if not items:
        return jsonify({"error": "nothing selected"}), 400
    if fmt in MEDIA_KINDS:
        # Recordings that can't be made into this (audio-only as video) are
        # listed as skipped in the job
        try:
            job = jobs.start(
                f"Export · {len(items)} {'file' if len(items) == 1 else 'files'} as {fmt}",
                items,
                _media_work(fmt, mopts, topts),
            )
        except jobs.Refused as e:
            return jsonify({"error": str(e)}), e.status
        return jsonify({"job": job.public()}), 202
    if len(items) > DIRECT_LIMIT:

        def work(item, out_dir, progress, should_stop):
            d = _doc(item, topts)
            if d is None:
                raise ex.ExportError("no transcript")
            name, data, _mime = ex.render([d], fmt, {**topts, "packing": "zip"})
            target = out_dir / (item["rel"] if topts["keep_folders"] else item["name"])
            target = target.with_suffix("." + ex.EXTENSION[fmt])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return [target]

        try:
            job = jobs.start(f"Export · {len(items):,} files as {fmt}", items, work)
        except jobs.Refused as e:
            return jsonify({"error": str(e)}), e.status
        return jsonify({"job": job.public()}), 202
    try:
        name, data, mime, skipped = _text_export(items, fmt, topts)
    except ex.ExportError as e:
        return _fail(e)
    # send_file writes the header for any name (RFC 6266: a plain form and
    # the UTF-8 one). Built by hand it had to fit Latin-1: a Korean or
    # Cyrillic name, even an en dash, raised while the headers were sent, and
    # the browser waited for an answer that never came.
    resp = send_file(
        io.BytesIO(data), mimetype=mime, as_attachment=True, download_name=_download_name(name)
    )
    resp.headers["X-Export-Skipped"] = str(len(skipped))
    return resp


@export_bp.route("/api/export/jobs")
def export_jobs():
    """The export jobs of the last day: progress, result, what was skipped."""
    return jsonify({"jobs": jobs.listing()})


@export_bp.route("/api/export/jobs/<jid>/download")
def export_job_download(jid):
    """The finished job's file."""
    p = jobs.result_path(jid)
    if not p:
        return jsonify({"error": "not ready or gone"}), 404
    return send_file(p, as_attachment=True, download_name=_download_name(p.name))


@export_bp.route("/api/export/jobs/<jid>/cancel", methods=["POST"])
def export_job_cancel(jid):
    """Stop a running or waiting export; what it wrote is removed."""
    return jsonify({"ok": True}) if jobs.cancel(jid) else (jsonify({"error": "not running"}), 409)


@export_bp.route("/api/export/jobs/<jid>", methods=["DELETE"])
def export_job_delete(jid):
    """Remove a finished export and its file."""
    return (
        jsonify({"ok": True})
        if jobs.remove(jid)
        else (jsonify({"error": "not finished or gone"}), 409)
    )

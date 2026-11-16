"""
Audio analysis — the search view's tab "Same audio" and its settings page.

Finds every recording that holds a snippet (fingerprints), recordings that
are the same or contain one another, and keeps each recording's waveform
and loudness (whisper_tm/audio).

Endpoints::

    GET  /api/audio/info                   counts, the run, the settings
    GET  /api/audio/estimate               what the background run has left
    POST /api/audio/run/start | run/stop   the background run
    POST /api/audio/search                 {"file", "start", "end"}: a stretch of a recording
    POST /api/audio/search-file            multipart "file": a sound or video of one's own
    GET  /api/audio/overview               the duplicates at a glance: counts, space, folders
    GET  /api/audio/duplicates             ?kind=same|contains &sort= &q= &min_sec= &pair= &file=
    GET  /api/audio/levels                 ?ids=1,2,3 &n= : stored waveforms
    GET  /api/audio/file                   ?file= : one recording's analysis
    POST /api/audio/retry-failed           failed recordings are tried again
    POST /api/audio/clear                  empty the index
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
import uuid
from pathlib import Path

from flask import Blueprint, abort, jsonify, request

from whisper_tm.config import MEDIA_EXTENSIONS, config
from whisper_tm.core import paths
from whisper_tm.core.security import check_root, safe_path

logger = logging.getLogger(__name__)

audio_bp = Blueprint("audio", __name__)

#: An uploaded snippet at most this large (a few minutes of video)
UPLOAD_MAX_BYTES = 300 * 1024 * 1024
#: Groups or wholes of the duplicates lists per page at most
DUP_PAGE_MAX = 200


def _enabled() -> bool:
    from whisper_tm.audio.runner import enabled

    return enabled()


def _store():
    from whisper_tm.audio.store import get_store

    return get_store()


def _source(filename: str) -> tuple[Path, str]:
    """The recording a request names — relative to the open folder or a full
    path in the allowed roots — and its identity (docs/PATHS.md)."""
    if paths.is_absolute(filename):
        return check_root(filename, field_name="file"), paths.canon(filename)
    return safe_path(config.media_dir, filename), paths.abs_from(filename, config.media_dir)


def _placed(rows: list[dict]) -> None:
    """Where each file row's recording lies for the page: ``file_dir`` (the
    root it is shown under) and ``filename`` (relative to it) — as the
    other searches hand them to goToSearchResult."""
    from whisper_tm.media.search import _places

    where = _places({r["path"] for r in rows})
    for r in rows:
        r["file_dir"], r["filename"] = where.get(
            r["path"], (str(Path(r["path"]).parent), Path(r["path"]).name)
        )
        r["name"] = Path(r["path"]).name


def _public(row: dict) -> dict:
    """A file row as the page gets it."""
    keep = (
        "id",
        "path",
        "duration",
        "status",
        "error",
        "lufs",
        "lra",
        "peak_db",
        "silence",
        "clipped",
        "analyzed_at",
        "source",
        "hashes",
    )
    return {k: row.get(k) for k in keep}


# ── Status ─────────────────────────────────────────────────────────────────


@audio_bp.route("/api/audio/info")
def audio_info():
    """What the tab and the settings page need: on, counts, the run."""
    from whisper_tm.audio.runner import RUNNER, settings

    s = settings()
    if not s["enabled"]:
        return jsonify(enabled=False, settings=s, run=RUNNER.status(), counts=None)
    try:
        counts = _store().counts()
    except Exception as e:  # noqa: BLE001 — shown as an error line
        logger.warning("Audio index unreadable: %s", e)
        return jsonify(
            enabled=True, settings=s, run=RUNNER.status(), counts=None, error=str(e)[:300]
        )
    return jsonify(enabled=True, settings=s, run=RUNNER.status(), counts=counts)


@audio_bp.route("/api/audio/estimate")
def audio_estimate():
    """What the background run has left: recordings, hours, bytes."""
    from whisper_tm.audio.runner import estimate

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    return jsonify(estimate(fresh=request.args.get("fresh") == "1"))


@audio_bp.route("/api/audio/run/start", methods=["POST"])
def audio_run_start():
    """Start the background run over the recordings without an analysis
    (``started`` false: one is going already)."""
    from whisper_tm.audio.runner import RUNNER

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    started = RUNNER.start(reason="started on the page")
    return jsonify(ok=True, started=started, run=RUNNER.status())


@audio_bp.route("/api/audio/run/stop", methods=["POST"])
def audio_run_stop():
    """Stop the background run after the pieces in hand — it doesn't go on
    by itself after a restart."""
    from whisper_tm.audio.runner import RUNNER

    return jsonify(ok=True, stopping=RUNNER.stop(), run=RUNNER.status())


# ── Searching ──────────────────────────────────────────────────────────────


def _answer(found: dict, *, exclude: tuple[int, float] | None = None) -> dict:
    """The hits with where to open them. ``exclude``: the recording the
    snippet was cut from and where — that place is the snippet itself."""
    hits = []
    for h in found["hits"]:
        places = h["places"]
        if exclude and h["id"] == exclude[0]:
            places = [p for p in places if abs(p["at"] - exclude[1]) > 1.0]
            if not places:
                continue
            best = max(places, key=lambda p: p["score"])
            h = {
                **h,
                "score": best["score"],
                "at": best["at"],
                "places": places,
                "strength": "strong" if best["score"] >= 25 else "match",
            }
        row = _public(h["file"])
        hits.append(
            {
                "score": h["score"],
                "strength": h["strength"],
                "at": h["at"],
                "q_start": h["q_start"],
                "q_end": h["q_end"],
                "places": places,
                **row,
            }
        )
    _placed(hits)
    hits.sort(key=lambda x: (-x["score"], x["path"]))
    return {"hits": hits, "hashes": found["hashes"], "seconds": found["seconds"], "ms": found["ms"]}


@audio_bp.route("/api/audio/search", methods=["POST"])
def audio_search():
    """Recordings holding ``start``…``end`` (seconds) of ``file`` — a
    stretch of 1 s to 5 min."""
    from whisper_tm.audio.analyze import QUERY_MAX_SEC, AnalysisError, search_snippet, snippet

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    body = request.get_json(silent=True) or {}
    filename = str(body.get("file") or "")
    try:
        start, end = float(body.get("start")), float(body.get("end"))
    except (TypeError, ValueError):
        return jsonify(error="start and end are seconds"), 400
    if not filename or start < 0 or end - start < 1.0:
        return jsonify(error="Choose at least one second"), 400
    if end - start > QUERY_MAX_SEC:
        return jsonify(error=f"At most {QUERY_MAX_SEC / 60:.0f} minutes at a time"), 400
    source, ident = _source(filename)
    if not source.exists():
        return jsonify(error="File not found"), 404
    try:
        x = snippet(source, start, end)
    except AnalysisError as e:
        return jsonify(error=f"The sound can't be read: {e}"), 422
    store = _store()
    own = store.file(ident)
    found = search_snippet(store, x)
    out = _answer(found, exclude=(own["id"], start) if own else None)
    out.update(
        source={
            "file": filename,
            "start": start,
            "end": end,
            "analyzed": bool(own and own["status"] == "done"),
        }
    )
    return jsonify(out)


@audio_bp.route("/api/audio/search-file", methods=["POST"])
def audio_search_file():
    """Recordings holding a sound or video file of one's own (its first
    five minutes)."""
    from whisper_tm.audio.analyze import AnalysisError, search_snippet, snippet

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file"), 400
    suffix = Path(f.filename).suffix.lower()
    kind = (f.mimetype or "").split("/")[0]
    if suffix not in MEDIA_EXTENSIONS and kind not in ("audio", "video"):
        return jsonify(error="A sound or a video, please"), 400
    if request.content_length and request.content_length > UPLOAD_MAX_BYTES:
        return jsonify(error=f"At most {UPLOAD_MAX_BYTES // (1024 * 1024)} MB"), 413
    with tempfile.TemporaryDirectory(prefix="wtm-audio-") as tmp:
        path = Path(tmp) / (uuid.uuid4().hex + suffix[:8])
        f.save(str(path))
        if path.stat().st_size > UPLOAD_MAX_BYTES:
            return jsonify(error=f"At most {UPLOAD_MAX_BYTES // (1024 * 1024)} MB"), 413
        try:
            x = snippet(path)
        except AnalysisError as e:
            return jsonify(error=f"The sound can't be read: {e}"), 422
    if len(x) < 16000:
        return jsonify(error="Less than a second of sound"), 422
    out = _answer(search_snippet(_store(), x))
    out.update(source={"name": f.filename})
    return jsonify(out)


# ── Duplicates ─────────────────────────────────────────────────────────────


def _filters():
    """The duplicates' filters from the query (audio/duplicates.py Filters):
    ``q``, ``q_whole``, ``q_part`` (letters of a path), ``person``,
    ``media`` (video|audio), ``ext`` (comma-separated), ``min_sec``,
    ``max_sec``, ``since``/``until`` (YYYY-MM-DD), ``lang``, ``transcribed``
    (yes|no), ``min_copies``, ``spread`` (across|within), ``min_parts``,
    ``pair``. Values that aren't numbers or dates are left out."""
    from whisper_tm.audio.duplicates import Filters

    a = request.args

    def text(name, n=200):
        return " ".join(str(a.get(name) or "").split())[:n]

    def number(name, kind=float):
        try:
            return max(kind(0), kind(a.get(name) or 0))
        except ValueError:
            return kind(0)

    def day(name):
        v = text(name, 10)
        return v if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) else ""

    return Filters(
        q=text("q"),
        q_whole=text("q_whole"),
        q_part=text("q_part"),
        person=text("person", 120),
        media=text("media", 8),
        ext=tuple(x for x in text("ext").split(",") if x)[:20],
        min_sec=number("min_sec"),
        max_sec=number("max_sec"),
        since=day("since"),
        until=day("until"),
        lang=text("lang", 16),
        transcribed=text("transcribed", 4),
        min_copies=number("min_copies", int),
        spread=text("spread", 8),
        min_parts=number("min_parts", int),
        pair=text("pair", 40),
    )


@audio_bp.route("/api/audio/overview")
def audio_overview():
    """The duplicates at a glance: how many recordings exist more than once
    and the space their extra copies take, how many contain another, the
    folders that hold the same recordings (the most shared first) and what
    the filters can choose from (``facets``: people, formats, languages, the
    span of the recording dates). Filters: ``q``, ``q_whole``, ``q_part``
    (letters of a path), ``person``, ``media`` (video|audio), ``ext``,
    ``min_sec``, ``max_sec``, ``since``/``until`` (YYYY-MM-DD), ``lang``,
    ``transcribed`` (yes|no), ``min_copies``, ``spread`` (across|within),
    ``min_parts``; ``folders``: how many folder pairs (30); ``fresh=1``:
    worked out anew."""
    from whisper_tm.audio import duplicates

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    try:
        top = int(request.args.get("folders") or 30)
    except ValueError:
        top = 30
    return jsonify(
        duplicates.overview(_store(), _filters(), top=top, fresh=request.args.get("fresh") == "1")
    )


@audio_bp.route("/api/audio/duplicates")
def audio_duplicates():
    """Recordings that are the same (``kind=same``: groups, each copy with
    its place on the group's timeline, size and kind) or contain others
    (``kind=contains``: each recording with the ones it holds and where).
    ``sort`` (same: space, copies, length, new; contains: parts, length,
    new), the filters (see the overview), ``file`` (only those of one
    recording), ``offset``, ``limit``."""
    from whisper_tm.audio import duplicates

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    kind = request.args.get("kind") or "same"
    if kind not in ("same", "contains"):
        return jsonify(error="kind is same or contains"), 400
    try:
        offset = max(0, int(request.args.get("offset") or 0))
        limit = min(DUP_PAGE_MAX, max(1, int(request.args.get("limit") or 50)))
    except ValueError:
        return jsonify(error="offset and limit are numbers"), 400
    flt = _filters()
    sort = request.args.get("sort") or ""
    store = _store()
    if request.args.get("file"):
        _, ident = _source(request.args["file"])
        own = store.file(ident)
        if not own:
            return jsonify(kind=kind, groups=[], wholes=[], total=0, offset=0, analyzed=False)
        flt.file_id = own["id"]
    if kind == "same":
        out = duplicates.group_page(store, flt, sort=sort or "space", offset=offset, limit=limit)
        _placed([f for grp in out["groups"] for f in grp["files"]])
    else:
        out = duplicates.whole_page(store, flt, sort=sort or "parts", offset=offset, limit=limit)
        _placed(
            [w["whole"] for w in out["wholes"]] + [p for w in out["wholes"] for p in w["parts"]]
        )
    return jsonify(out)


@audio_bp.route("/api/audio/levels")
def audio_levels():
    """The stored waveforms of recordings (``ids``: comma-separated), each in
    at most ``n`` values (400), base64: 0…255 over -60…0 dBFS, the loudest
    of each stretch."""
    from whisper_tm.audio import duplicates

    if not _enabled():
        return jsonify(error="Audio analysis is switched off"), 503
    try:
        ids = [int(x) for x in str(request.args.get("ids") or "").split(",") if x.strip()]
        n = int(request.args.get("n") or 400)
    except ValueError:
        return jsonify(error="ids and n are numbers"), 400
    if len(ids) > duplicates.LEVELS_MAX_IDS:
        return jsonify(error=f"At most {duplicates.LEVELS_MAX_IDS} recordings at a time"), 400
    return jsonify(levels={str(k): v for k, v in duplicates.levels(_store(), ids, n).items()})


# ── One recording ──────────────────────────────────────────────────────────


@audio_bp.route("/api/audio/file")
def audio_file():
    """One recording's analysis: loudness, and how many copies it has."""
    if not _enabled():
        return jsonify(enabled=False)
    filename = request.args.get("file") or ""
    if not filename:
        abort(400)
    _, ident = _source(filename)
    store = _store()
    row = store.file(ident)
    if not row:
        return jsonify(enabled=True, analyzed=False)
    with store.connect() as c:
        same = c.execute(
            "SELECT count(*) FROM relations WHERE kind='same' AND (a=? OR b=?)",
            (row["id"], row["id"]),
        ).fetchone()[0]
        holds = c.execute(
            "SELECT count(*) FROM relations WHERE kind='contains' AND a=?", (row["id"],)
        ).fetchone()[0]
        held = c.execute(
            "SELECT count(*) FROM relations WHERE kind='contains' AND b=?", (row["id"],)
        ).fetchone()[0]
    return jsonify(
        enabled=True,
        analyzed=row["status"] == "done",
        file=_public(row),
        copies=same,
        contains=holds,
        contained_in=held,
        current=abs(row["mtime"] - _mtime(ident)) < 0.01,
    )


def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return -1.0


# ── Upkeep ─────────────────────────────────────────────────────────────────


@audio_bp.route("/api/audio/retry-failed", methods=["POST"])
def audio_retry_failed():
    """Forget the recordings that could not be read — the next run tries
    them again."""
    from whisper_tm.audio.runner import forget_scope

    store = _store()
    with store.connect() as c:
        n = c.execute("DELETE FROM files WHERE status='error'").rowcount
    forget_scope()
    return jsonify(ok=True, forgotten=n)


@audio_bp.route("/api/audio/clear", methods=["POST"])
def audio_clear():
    """Empty the index: fingerprints, waveforms, loudness, relations."""
    from whisper_tm.audio.runner import RUNNER, forget_scope

    if RUNNER.running():
        return jsonify(error="Stop the background run first"), 409
    t0 = time.monotonic()
    _store().clear()
    forget_scope()
    return jsonify(ok=True, ms=round((time.monotonic() - t0) * 1000))

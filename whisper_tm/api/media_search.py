"""
Images & sounds — the search view's tab and its settings page.

Finds moments of the library by what is seen or heard: stills, 10 s clips,
clips with their sound, sound windows and spoken text of the recordings lie
in ONE vector space (EmbeddingGemma 2), so a text, a hit or a file of one's
own asks all of them. The parts:

* media/service.py — the model, in its own environment, a process of its
  own (core/media_server.py starts and stops it);
* media/store.py — the index, ``media.db``;
* media/indexer.py — index runs, a guest on the GPU (the gate);
* media/search.py, media/llm.py — searching, and what the language model
  adds (sound labels, the check of the best pictures, descriptions).

Endpoints::

    GET  /api/media-search/info
    GET  /api/media-search?q=|like=|qid=  &kinds=&dims=&limit=&folders=
    POST /api/media-search/query-file          file, q, with_sound
    GET  /api/media-search/thumb/<moment>
    GET  /api/media-search/clip/<moment>
    POST /api/media-search/check               {"query", "moments": [...]}
    GET  /api/media-search/check/<id>          POST …/<id>/cancel
    POST /api/media-search/describe/<moment>
    GET  /api/media-search/index               POST …/index/start|stop|clear
    GET  /api/media-search/environment         POST …/environment/install
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file

logger = logging.getLogger(__name__)

media_search_bp = Blueprint("media_search", __name__)

#: Uploaded queries are kept this long — they are only vectors
QUERY_TTL = 3600.0
#: A text query's length, as /api/search (a query of thousands of
#: characters took tens of seconds to embed)
QUERY_MAX = 500
#: The largest id SQLite stores
ID_MAX = 2**63 - 1
_queries: dict[str, dict] = {}
_lock = threading.Lock()


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("media_search", name, default=default)


def _enabled() -> bool:
    from whisper_tm.core.media_server import enabled

    return enabled()


# ── Status ─────────────────────────────────────────────────────────────────


@media_search_bp.route("/api/media-search/info")
def media_info():
    """What the tab needs to know: on, ready, what the index holds."""
    from whisper_tm.core.media_server import SERVER
    from whisper_tm.media import environment, indexer, llm
    from whisper_tm.media.store import get_store

    enabled = _enabled()
    try:
        counts = get_store().counts()
    except Exception as e:  # noqa: BLE001 — shown as an empty index
        logger.warning("Media index unreadable: %s", e)
        counts = {"kinds": {}, "files": {}, "moments": 0, "dims": None}
    env = environment.check()
    s = indexer.settings()
    configured = llm.configured()
    return jsonify(
        enabled=enabled,
        environment_ok=env["ok"],
        environment_problems=env["problems"],
        counts=counts,
        ready=enabled and env["ok"] and sum(counts["kinds"].values()) > 0,
        dims=counts.get("dims") or s["dims"],
        kinds_indexed=s["kinds"],
        service=SERVER.state() if enabled else None,
        indexer=indexer.INDEXER.status(),
        llm={
            "configured": configured,
            "vision": llm.vision_available() if configured else False,
            "translate_sounds": bool(_cfg("llm_translate_sounds", True)),
            "check_count": int(_cfg("llm_check_count", 24) or 24),
        },
        results_per_kind=int(_cfg("results_per_kind", 48) or 48),
        model=environment.model_id(),
    )


# ── Searching ──────────────────────────────────────────────────────────────


@media_search_bp.route("/api/media-search")
def media_search():
    """Search every kind for one query: ``q`` a text, ``like`` an item id
    ("Similar") or ``qid`` a file query; ``kinds``, ``dims`` (Matryoshka),
    ``limit`` per kind, ``folders`` (``|``-separated path prefixes)."""
    from whisper_tm.core.media_server import MediaUnavailable
    from whisper_tm.media import search
    from whisper_tm.media.store import DIMS

    if not _enabled():
        return jsonify(error="Images & sounds is switched off — Settings → Images & sounds"), 503
    try:
        limit = max(
            1, min(int(request.args.get("limit") or _cfg("results_per_kind", 48) or 48), 200)
        )
        dims = int(request.args.get("dims") or 0) or None
    except ValueError:
        return jsonify(error="limit and dims are numbers"), 400
    if dims is not None and dims not in DIMS:
        return jsonify(error=f"dims is one of {list(DIMS)}"), 400
    kinds = [k for k in (request.args.get("kinds") or "").split(",") if k] or None
    folders = [f for f in (request.args.get("folders") or "").split("|") if f] or None
    q = (request.args.get("q") or "").strip()
    if len(q) > QUERY_MAX:
        return jsonify(error=f"Search term too long (max {QUERY_MAX} characters)"), 400
    like, qid = request.args.get("like"), request.args.get("qid")
    # Parsed here: a ValueError of the search itself was reported as "like is
    # an item id"
    try:
        like_id = int(like) if like and not qid else None
    except ValueError:
        return jsonify(error="like is an item id"), 400
    if like_id is not None and not 0 <= like_id <= ID_MAX:
        return jsonify(error="No such item"), 404
    try:
        if qid:
            with _lock:
                held = _queries.get(qid)
            if not held:
                return jsonify(error="That file query has expired — drop the file in again"), 404
            out = search.search(
                vector=held["vector"], kinds=kinds, dims=dims, limit=limit, folders=folders
            )
            out["source"] = {
                "type": "file",
                "name": held["name"],
                "kind": held["kind"],
                "text": held["text"],
            }
        elif like_id is not None:
            out = search.search(like=like_id, kinds=kinds, dims=dims, limit=limit, folders=folders)
            out["source"] = {"type": "like", "item": like_id}
        elif len(q) >= 2:
            out = search.search(text=q, kinds=kinds, dims=dims, limit=limit, folders=folders)
            out["source"] = {"type": "text", "text": q}
        else:
            return jsonify(error="Enter at least 2 characters"), 400
    except LookupError as e:
        return jsonify(error=str(e)), 404
    except MediaUnavailable as e:
        return jsonify(error=str(e)), 503
    return jsonify(out)


@media_search_bp.route("/api/media-search/query-file", methods=["POST"])
def media_query_file():
    """A file as the query — an image, a sound or a video (``with_sound``:
    its sound too), and ``q``: a text that goes into the same vector."""
    from whisper_tm.core.media_server import SERVER, MediaUnavailable
    from whisper_tm.media import indexer
    from whisper_tm.media.service import unb64

    if not _enabled():
        return jsonify(error="Images & sounds is switched off"), 503
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file"), 400
    kind = {"image": "image", "audio": "audio", "video": "video"}.get(
        (f.mimetype or "").split("/")[0]
    )
    if kind is None:
        return jsonify(error="An image, a sound or a video, please"), 400
    up = indexer.cache_dir() / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    path = up / (uuid.uuid4().hex + Path(f.filename).suffix.lower()[:8])
    f.save(str(path))
    text = (request.form.get("q") or "").strip()[:500]
    try:
        frames = indexer.frames_for(indexer.settings(), 0)
        res = SERVER.call(
            "/embed-file",
            {
                "path": str(path),
                "kind": kind,
                "text": text,
                "with_sound": request.form.get("with_sound") == "1",
                "frames": frames,
            },
            timeout=300,
        )
    except MediaUnavailable as e:
        return jsonify(error=str(e)), 400 if getattr(e, "status", 0) == 400 else 503
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    qid = uuid.uuid4().hex
    now = time.monotonic()
    with _lock:
        for k in [k for k, v in _queries.items() if now - v["at"] > QUERY_TTL]:
            del _queries[k]
        _queries[qid] = {
            "vector": unb64(res["vector"]),
            "name": f.filename,
            "kind": kind,
            "text": text,
            "at": now,
        }
    return jsonify(qid=qid, kind=kind, name=f.filename, text=text, embed_ms=res.get("ms"))


# ── Pictures and clips ─────────────────────────────────────────────────────


def _moment(mid: int) -> dict:
    from whisper_tm.media.store import get_store

    if mid > ID_MAX:
        abort(404)  # past SQLite's integers: an OverflowError was a 500
    m = get_store().moment(mid)
    if not m:
        abort(404)
    return m


@media_search_bp.route("/api/media-search/thumb/<int:mid>")
def media_thumb(mid: int):
    """A moment's still (JPEG) — cut again from the recording when the
    cache lost it. Only files under the thumbnail folder are served."""
    from whisper_tm.media import indexer

    m = _moment(mid)
    thumb = m.get("thumb")
    if not thumb:
        abort(404)
    base = indexer.thumbs_dir().resolve()
    p = Path(thumb).resolve()
    if base not in p.parents:
        abort(404)
    if not p.is_file():
        # The cache was emptied: cut the still again from the recording
        from whisper_tm.media import cuts

        width = int(_cfg("thumb_width", 480) or 480)
        if not cuts.cached(p, lambda out: cuts.cut_still(m["path"], m["t"], out, width=width)):
            abort(404)
    return send_file(str(p), mimetype="image/jpeg", max_age=86400)


def _clip_window(m: dict) -> tuple[float, float, bool]:
    if m["video"]:
        sec = float(_cfg("clip_sec", 10) or 10)
        return max(0.0, m["t"] - sec * 0.3), sec, True
    return m["start"], m["dur"], False


@media_search_bp.route("/api/media-search/clip/<int:mid>")
def media_clip(mid: int):
    """The moment, playable: 10 s of the video with its sound around a
    still, or the sound window of an audio file. Cut once, then kept in
    the cache (``media_search.cache_max_mb``; the oldest go first)."""
    from whisper_tm.media import cuts, indexer

    m = _moment(mid)
    start, length, video = _clip_window(m)
    out = indexer.cache_dir() / "clips" / _clip_name(m["path"], start, length, video)
    if not os.path.isfile(m["path"]) and not out.is_file():
        abort(404)
    made = cuts.cached(
        out,
        lambda o: cuts.cut_clip(m["path"], start, length, video, o),
        after_cut=lambda: _trim_clips(out.parent),
    )
    if not made:
        # The recording can't be read (gone, broken): nothing to serve
        logger.warning("Media search: clip of moment %d not cut", mid)
        abort(404)
    return send_file(
        str(made), mimetype="video/mp4" if video else "audio/mp4", conditional=True, max_age=86400
    )


def _clip_name(path: str, start: float, length: float, video: bool) -> str:
    """The cached clip's name, by what it holds: named by the moment id, a new
    moment got an old recording's clip — SQLite reuses the ids of deleted
    rows."""
    import hashlib

    kind = "mp4" if video else "m4a"
    what = f"{path}|{start:.3f}|{length:.3f}|{kind}".encode("utf-8", "surrogatepass")
    return f"{hashlib.sha1(what).hexdigest()[:20]}.{kind}"


def _trim_clips(folder: Path) -> None:
    """Keep the clip cache under ``cache_max_mb`` — the oldest go first."""
    from whisper_tm.media import cuts

    cuts.trim(folder, float(_cfg("cache_max_mb", 2000) or 2000), suffixes=(".mp4", ".m4a"))


# ── The language model ─────────────────────────────────────────────────────


@media_search_bp.route("/api/media-search/check", methods=["POST"])
def media_check_start():
    """Let the vision model judge the given moments' stills for the query —
    one picture per request, in the background."""
    from whisper_tm.media import llm
    from whisper_tm.media.store import get_store

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    ids = [int(x) for x in (data.get("moments") or []) if str(x).isdigit()][:200]
    if not query or not ids:
        return jsonify(error="query and moments are needed"), 400
    if not llm.configured():
        return jsonify(error="No language model — Settings → AI models"), 503
    if llm.vision_available() is False:
        return jsonify(
            error="The language model can't see pictures (no vision projector loaded)"
        ), 503
    store = get_store()
    items = []
    for mid in ids:
        m = store.moment(mid)
        if m and m.get("thumb") and os.path.isfile(m["thumb"]):
            items.append({"key": mid, "thumb": m["thumb"]})
    if not items:
        return jsonify(error="None of these moments has a picture"), 400
    return jsonify(id=llm.start_check(query, items), total=len(items))


@media_search_bp.route("/api/media-search/check/<cid>")
def media_check_state(cid: str):
    """How far a check is: ``done``/``total`` and the scores so far
    (``{moment id: 0–9}``)."""
    from whisper_tm.media import llm

    st = llm.check_state(cid)
    if not st:
        return jsonify(error="No such check"), 404
    return jsonify(st)


@media_search_bp.route("/api/media-search/check/<cid>/cancel", methods=["POST"])
def media_check_cancel(cid: str):
    """Stop a running check after the picture in hand."""
    from whisper_tm.media import llm

    if llm.check_state(cid) is None:
        return jsonify(error="No such check"), 404
    return jsonify(cancelled=llm.cancel_check(cid))


@media_search_bp.route("/api/media-search/describe/<int:mid>", methods=["POST"])
def media_describe(mid: int):
    """One sentence about the moment's still — written once, kept."""
    from whisper_tm.config import config
    from whisper_tm.media import llm
    from whisper_tm.media.store import get_store

    store = get_store()
    m = _moment(mid)
    have = store.describe(mid)
    if have and not (request.get_json(silent=True) or {}).get("again"):
        return jsonify(text=have["text"], model=have["model"], cached=True)
    if not m.get("thumb") or not os.path.isfile(m["thumb"]):
        return jsonify(error="This moment has no picture"), 400
    if not llm.configured():
        return jsonify(error="No language model — Settings → AI models"), 503
    try:
        text = llm.describe(m["thumb"])
    except Exception as e:  # noqa: BLE001 — the model's answer, shown
        return jsonify(error=f"The language model did not answer: {e}"), 503
    model = _cfg("llm_model", "") or config.get("llm", "model", default="") or ""
    store.set_description(mid, text, model)
    # Searchable at once: the sentence as a vector of the kind "description"
    try:
        from whisper_tm.core.media_server import SERVER
        from whisper_tm.media.service import unb64

        res = SERVER.call("/embed", {"texts": [text], "prompt": "Document"}, timeout=60)
        store.add_items([(mid, "description", unb64(res["vectors"][0]))])
    except Exception as e:  # noqa: BLE001 — the text is kept; a describing run embeds it later
        logger.info("Media search: description of moment %d not embedded: %s", mid, e)
    return jsonify(text=text, model=model, cached=False)


# ── The index ──────────────────────────────────────────────────────────────


@media_search_bp.route("/api/media-search/index")
def media_index_state():
    """Counts, the run, and what indexing the scope takes and how far it is
    (``estimate``: kept and computed in the background — ``None`` until the
    first is ready, ``estimate_computing`` while a new one is made)."""
    from whisper_tm.media import indexer
    from whisper_tm.media.store import get_store

    store = get_store()
    run = indexer.INDEXER.status()
    out = {
        "counts": store.counts(),
        "descriptions": store.description_counts(),
        "describe_run": _describe_entry(),
        "run": run,
        "settings": indexer.settings(),
        "ram_bytes": store.ram_bytes(),
    }
    if request.args.get("estimate", "1") != "0":
        est = indexer.estimate_cached(running=bool(run.get("running")))
        out["estimate"] = est["value"]
        out["estimate_age"] = est["age"]
        out["estimate_computing"] = est["computing"]
        if est["error"]:
            out["estimate_error"] = est["error"]
    return jsonify(out)


@media_search_bp.route("/api/media-search/index/start", methods=["POST"])
def media_index_start():
    """Start an index run (``rebuild``: empty the index first)."""
    from whisper_tm.media import environment, indexer

    if not _enabled():
        return jsonify(error="Images & sounds is switched off"), 409
    if not environment.check()["ok"]:
        return jsonify(error="The media environment isn't ready — install it first"), 409
    rebuild = bool((request.get_json(silent=True) or {}).get("rebuild"))
    started = indexer.INDEXER.start(rebuild=rebuild, reason="rebuild" if rebuild else "manual")
    return jsonify(started=started, run=indexer.INDEXER.status())


@media_search_bp.route("/api/media-search/index/stop", methods=["POST"])
def media_index_stop():
    """Stop the index run after its current request — it gives the GPU back."""
    from whisper_tm.media import indexer

    return jsonify(stopping=indexer.INDEXER.stop(), run=indexer.INDEXER.status())


#: What the index writes into its cache folders. The folder is a setting
#: (media_search.cache_dir): pointed at someone else's folder, clearing the
#: index must not cost them a "thumbs" or "clips" folder of theirs.
_CACHE_FILES = {
    "thumbs": {".jpg", ".jpeg", ".part", ".tmp"},
    "clips": {".mp4", ".m4a", ".part", ".tmp"},
}


def _remove_cache(folder: Path, suffixes: set[str]) -> bool:
    """Delete ``folder`` if it holds nothing but files of these kinds (and
    no link); False — and nothing deleted — otherwise."""
    if not folder.exists():
        return True
    if folder.is_symlink() or not folder.is_dir():
        return False
    for p in folder.rglob("*"):
        if p.is_symlink() or (p.is_file() and p.suffix.lower() not in suffixes):
            logger.warning(
                "Not cleared: %s holds %s, which the index doesn't write", folder, p.name
            )
            return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


@media_search_bp.route("/api/media-search/index/clear", methods=["POST"])
def media_index_clear():
    """Empty the index — vectors, thumbnails and preview clips."""
    from whisper_tm.media import indexer
    from whisper_tm.media.store import get_store

    if indexer.INDEXER.running():
        return jsonify(error="An index run is going — stop it first"), 409
    get_store().clear()
    kept = [
        str(indexer.cache_dir() / sub)
        for sub, suffixes in _CACHE_FILES.items()
        if not _remove_cache(indexer.cache_dir() / sub, suffixes)
    ]
    get_store().set_meta("thumb_bytes", 0)
    indexer.forget_estimate()
    return jsonify(cleared=True, kept=kept, counts=get_store().counts(fresh=True))


def _describe_entry() -> dict | None:
    """The newest describing run in the LLM queue (pending or running
    first, else the last one)."""
    from whisper_tm.core.llm_queue import TYPE_MEDIA_DESCRIBE, llm_queue

    try:
        items = (llm_queue.snapshot(limit=200) or {}).get("items") or []
    except Exception:  # noqa: BLE001 — no queue, no run
        return None
    mine = [i for i in items if i.get("kind") == TYPE_MEDIA_DESCRIBE]
    live = [i for i in mine if i.get("status") in ("pending", "running")]
    pick = (live or mine or [None])[0]
    if not pick:
        return None
    return {k: pick.get(k) for k in ("id", "status", "progress", "message", "error")}


@media_search_bp.route("/api/media-search/describe-run", methods=["POST"])
def media_describe_run():
    """Describe the stills that have no description yet — a job of the LLM
    queue (``limit`` at most, the newest recordings first)."""
    from whisper_tm.media import describe, indexer, llm

    if not _enabled():
        return jsonify(error="Images & sounds is switched off"), 409
    if not llm.configured():
        return jsonify(error="No language model — Settings → AI models"), 409
    if llm.vision_available() is False:
        return jsonify(
            error="The language model can't see pictures (no vision projector loaded)"
        ), 409
    live = _describe_entry()
    if live and live.get("status") in ("pending", "running"):
        return jsonify(error="A describing run is already in the LLM queue", run=live), 409
    data = request.get_json(silent=True) or {}
    try:
        limit = int(data.get("limit") or _cfg("llm_describe_limit", 1000) or 1000)
    except (TypeError, ValueError):
        return jsonify(error="limit is a number"), 400
    entry = describe.enqueue(limit, indexer.settings()["folders"])
    return jsonify(entry=entry, run=_describe_entry())


# ── The environment ────────────────────────────────────────────────────────


@media_search_bp.route("/api/media-search/environment")
def media_environment():
    """The media environment (installed? the model fetched?), an
    install that runs, and the service process."""
    from whisper_tm.core.media_server import SERVER
    from whisper_tm.media import environment

    return jsonify(
        check=environment.check(force=request.args.get("fresh") == "1"),
        install=environment.install_state(),
        service=SERVER.state(),
    )


@media_search_bp.route("/api/media-search/environment/install", methods=["POST"])
def media_environment_install():
    """Install or repair the media environment in the background."""
    from whisper_tm.core.media_server import SERVER
    from whisper_tm.media import environment, indexer

    if indexer.INDEXER.running():
        return jsonify(error="An index run is going — stop it first"), 409
    SERVER.stop("the environment is being installed", force=True)
    return jsonify(started=environment.start_install(), install=environment.install_state())


@media_search_bp.route("/api/media-search/service/stop", methods=["POST"])
def media_service_stop():
    """Stop the service process (it starts again with the next search)."""
    from whisper_tm.core.media_server import SERVER
    from whisper_tm.media import indexer

    if indexer.INDEXER.running():
        return jsonify(error="An index run is going — stop it first"), 409
    return jsonify(stopped=SERVER.stop("stopped from the settings", force=True))


def warm_up() -> None:
    """After a start: load the index matrices in the background and begin
    looking for new recordings (``auto_index``) — when switched on."""
    if not _enabled():
        return
    from whisper_tm.media import indexer
    from whisper_tm.media.store import get_store, store_path

    if store_path().exists():
        get_store().preload()
    indexer.start_auto()
    # The environment check starts its Python (~5 s): once now, so the tab's
    # first look doesn't wait for it
    from whisper_tm.media import environment

    threading.Thread(
        target=environment.check, kwargs={"force": True}, name="media-env-check", daemon=True
    ).start()

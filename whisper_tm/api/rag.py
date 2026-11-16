"""
RAG API: status, indexer (SSE), hybrid search, schema management, export.

The main SQLite remains the source of truth; the vector store is a
rebuildable search index in its own SQLite file. The indexer runs in a
background thread and streams progress via Server-Sent Events.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time

from flask import Blueprint, Response, jsonify, request, send_file

from whisper_tm.config import config
from whisper_tm.core import paths, rerank_server
from whisper_tm.extensions import db
from whisper_tm.rag import (
    RagConfigError,
    RagIndexer,
    RagUnavailable,
    deep_search,
    retrieve,
    speakers,
)
from whisper_tm.rag.indexer import _close as _rag_close
from whisper_tm.rag.service import (
    embedding_config,
    embedding_ready,
    get_rag_config,
    index_model_name,
    make_embed_fn,
    make_query_embed_fn,
    make_rag_db,
    vector_db_path,
)

logger = logging.getLogger(__name__)

rag_bp = Blueprint("rag", __name__)

#: What /api/rag/search searches with
SEARCH_MODES = ("hybrid", "meaning", "vector", "fts")

# ── Global build state (only one build at a time) ─────────────────────────
_build_lock = threading.Lock()
_build_state: dict = {
    "running": False,
    "stop": None,  # threading.Event
    "thread": None,
    "last_summary": None,
    "started_at": 0.0,
    # Latest progress report of the running job. The SSE stream belongs
    # only to the tab that started it; this holds it for everyone else.
    "progress": None,
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_headers() -> dict:
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


# ── Status ──────────────────────────────────────────────────────────────
@rag_bp.route("/api/rag/status")
def rag_status():
    """Configuration + connection/schema status."""
    cfg = get_rag_config()
    econf = embedding_config()
    emb_ok, emb_msg = embedding_ready()

    # #104: copy build state consistently under _build_lock
    with _build_lock:
        build_info = {
            "running": _build_state["running"],
            "started_at": _build_state["started_at"],
            "last_summary": _build_state["last_summary"],
        }

    out = {
        "enabled": bool(cfg.get("enabled")),
        "embedding": {
            "configured": emb_ok,
            "detail": emb_msg,
            "model": econf["model"],
            "endpoint": econf["endpoint"],
        },
        "path": vector_db_path(),
        "chunking": {
            "max_chars": cfg.get("chunk_max_chars"),
            "min_chars": cfg.get("chunk_min_chars"),
            "overlap_segments": cfg.get("chunk_overlap_segments"),
        },
        "fts_language": cfg.get("fts_language"),
        "build": build_info,
    }
    try:
        out["rerank"] = rerank_server.status()
    except Exception as e:  # noqa
        out["rerank"] = {"available": False, "detail": str(e)[:200]}

    # SQLite source scope (how many transcripts could be indexed)
    try:
        if db:
            out["source"] = {"transcripts": db.count_rag_candidates()}
    except Exception as e:  # noqa
        out["source"] = {"transcripts": 0, "error": str(e)[:120]}

    try:
        rag = make_rag_db()
        out["db"] = rag.ping()
        if out["db"].get("reachable"):
            try:
                out["db"]["stats"] = rag.stats()
            except Exception as e:  # noqa
                out["db"]["stats"] = {"error": str(e)[:120]}
    except RagUnavailable as e:
        out["db"] = {"reachable": False, "detail": str(e)}
    except Exception as e:  # noqa
        out["db"] = {"reachable": False, "detail": str(e)[:200]}

    return jsonify(out)


# ── Coverage / tracking (which transcripts are embedded?) ─────────────────
def _compute_coverage(rag, sdb) -> dict:
    """Complete transcripts (main DB) vs. embedded ones (vector DB).

    A transcript counts as done when it appears unchanged in
    indexed_transcripts (seg_count + updated_at) — the same logic the
    indexer uses when skipping. Empty-indexed ones (chunk_count=0) also
    count as done: they were processed, just without chunks (#095). Changed
    ones count as "stale" and go into the pending set (get re-embedded).
    """
    # #101: The counting runs in Python because the candidates (main DB)
    # and the indexed map (vector DB) live in SEPARATE files — a single SQL
    # aggregate across both isn't possible. Also, the staleness check needs
    # a tolerance comparison (updated_at ± eps) per key, and
    # embedded_keys/pending_keys are needed as lists (API + tests). The
    # cheapest feasible option without a cross-DB query: two bulk loads
    # + a single O(N) pass.
    # The key MUST be built by the same function as the indexer.
    #
    # Previously this used `c["rel_path"]`. But since switching to the
    # identity (docs/PATHS.md), the indexer stores the canonical `abs_path`
    # — so the two never matched. Consequence: after a complete, error-free
    # indexing run, coverage kept reporting "0 embedded, everything
    # pending" — permanently. Demonstrated on one transcript: what gets
    # stored is `/tmp/…/cooking/pasta.mp4`, what was looked up was
    # `cooking/pasta.mp4`.
    #
    # This was only found once the integration tests ran against a real
    # database for the first time — before that they needed a Postgres that
    # was never there, were always skipped, and never noticed the switch.
    # Since moving to SQLite they run without an external service and
    # therefore always.
    from whisper_tm.rag.indexer import _transcript_key

    candidates = sdb.get_rag_candidates() if sdb else []
    indexed = rag.get_indexed_map()
    embedded_keys: list[str] = []
    pending_keys: list[str] = []
    stale = 0
    for c in candidates:
        key = _transcript_key(c)
        seg = c.get("seg_count") or 0
        upd = c.get("updated_at") or 0.0
        prev = indexed.get(key)
        if prev and prev.get("seg_count") == seg and _rag_close(prev.get("updated_at"), upd):
            # #095: unchanged, already processed → done. Empty-indexed
            # ones (chunk_count=0) also count as done (the indexer skips
            # them too), otherwise coverage stays below 100 %.
            embedded_keys.append(key)
        elif prev:
            # exists, but changed → stale, will be re-embedded
            stale += 1
            pending_keys.append(key)
        else:
            pending_keys.append(key)
    return {
        "total": len(candidates),
        "embedded": len(embedded_keys),
        "pending": len(pending_keys),
        "stale": stale,
        "embedded_keys": embedded_keys,
        "pending_keys": pending_keys,
    }


@rag_bp.route("/api/rag/coverage")
def rag_coverage():
    """Aggregated embedding progress: how many transcripts are done."""
    if not db:
        return jsonify({"error": "No database"}), 500
    try:
        rag = make_rag_db()
        cov = _compute_coverage(rag, db)
        pend = cov["pending_keys"]
        return jsonify(
            {
                "ok": True,
                "reachable": True,
                "total": cov["total"],
                "embedded": cov["embedded"],
                "pending": cov["pending"],
                "stale": cov["stale"],
                "pending_sample": pend[:300],
                "pending_truncated": len(pend) > 300,
            }
        )
    except (RagConfigError, RagUnavailable) as e:
        return jsonify({"error": str(e), "reachable": False}), 200
    except Exception as e:  # noqa
        return jsonify({"error": str(e)[:300], "reachable": False}), 200


@rag_bp.route("/api/rag/embedded_keys")
def rag_embedded_keys():
    """Which transcripts are embedded — as **rel_path** for the file list.

    The store keeps its keys as the canonical ``abs_path`` (the identity,
    docs/PATHS.md). The file list works with ``rel_path``. Translation
    therefore happens HERE, at the boundary, not in the browser.

    Previously this endpoint delivered the raw keys, and ``filelist.js``
    compared them against ``f.name``. That never matched. What sometimes
    still showed a checkmark was the fallback to the bare filename — and
    that is worse than no match: two identically named files in different
    folders both counted as embedded as soon as one of them was.

    ``chunks`` is included per entry, so the list can say not just "yes"
    but also "how much".
    """
    if not get_rag_config().get("enabled"):
        return jsonify({"keys": [], "count": 0, "active": False})
    if not db:
        return jsonify({"keys": [], "count": 0, "active": False, "error": "No database"})
    try:
        from whisper_tm.rag.indexer import _transcript_key

        rag = make_rag_db()
        indexed = rag.get_indexed_map()
        keys, chunks = [], {}
        # The file list matches on its entries' names, which are relative
        # to the folder open NOW. The stored rel_path is relative to the one
        # open at transcription time, so it is derived afresh from the
        # identity; files outside the open folder aren't in the list anyway.
        root = str(config.media_dir)
        for cand in db.get_rag_candidates():
            entry = indexed.get(_transcript_key(cand))
            if entry and (entry.get("chunk_count") or 0) > 0:
                name = paths.rel_from(cand["abs_path"], root)
                if name is None:
                    continue
                keys.append(name)
                chunks[name] = entry["chunk_count"]
        return jsonify({"keys": keys, "count": len(keys), "chunks": chunks, "active": True})
    except Exception as e:  # noqa
        return jsonify({"keys": [], "count": 0, "active": False, "error": str(e)[:200]})


# ── Create schema ───────────────────────────────────────────────────────
@rag_bp.route("/api/rag/schema", methods=["POST"])
def rag_schema():
    """Create extension + tables. Dimension from config or a sample."""
    try:
        rag = make_rag_db()
        cfg = get_rag_config()
        dim = int(cfg.get("dimensions") or 0)
        probed = False
        if dim <= 0:
            # Determine dimension via a test embedding
            embed = make_embed_fn()
            vec = embed(["Dimension sample."])
            if not vec or not vec[0]:
                return jsonify({"error": "Test embedding returned no vector"}), 400
            dim = len(vec[0])
            probed = True
        info = rag.ensure_schema(dimensions=dim)
        return jsonify({"ok": True, "dimension": dim, "probed": probed, "schema": info})
    except (RagConfigError, RagUnavailable) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa
        logger.warning("rag_schema: %s", e)
        return jsonify({"error": str(e)[:300]}), 500


# ── Build (SSE) ─────────────────────────────────────────────────────────
@rag_bp.route("/api/rag/build")
def rag_build():
    """Start an indexing run and stream progress as SSE.

    Query: force=1 (redo all), rebuild=1 (clear DB first).
    """
    if not db:
        return jsonify({"error": "No SQLite database"}), 500

    force = request.args.get("force", "false").lower() in ("1", "true", "yes")
    rebuild = request.args.get("rebuild", "false").lower() in ("1", "true", "yes")

    emb_ok, emb_msg = embedding_ready()
    if not emb_ok:
        return jsonify({"error": emb_msg}), 400

    with _build_lock:
        if _build_state["running"]:
            return jsonify({"error": "A build is already running"}), 409
        stop_ev = threading.Event()
        _build_state.update(running=True, stop=stop_ev, started_at=time.time())

    q: queue.Queue = queue.Queue()
    cfg = get_rag_config()

    def worker():
        try:
            rag = make_rag_db()
            if rebuild:
                q.put({"phase": "prepare", "message": "Clearing RAG DB (rebuild) …"})
                rag.drop_all()
            embed = make_embed_fn()
            indexer = RagIndexer(
                rag,
                db,
                embed,
                chunk_max_chars=cfg.get("chunk_max_chars", 1200),
                chunk_min_chars=cfg.get("chunk_min_chars", 250),
                chunk_overlap_segments=cfg.get("chunk_overlap_segments", 1),
                chunk_on_speaker_change=cfg.get("chunk_on_speaker_change", False),
                include_speaker_labels=cfg.get("include_speaker_labels", True),
                batch_size=cfg.get("batch_size", 64),
                # Locally the GGUF file name, not embedding.model: that's
                # the model that runs, so swapping it re-embeds
                embedding_model=index_model_name(),
                default_media_dir=str(config.media_dir),
            )

            def report_it(p):
                # Twice: into the queue for the SSE stream AND into the shared
                # state. The stream belongs only to the tab that started the
                # run — anyone opening settings later or reloading the page
                # would otherwise see a frozen progress card and have to
                # guess whether anything is still happening.
                with _build_lock:
                    _build_state["progress"] = p
                q.put(p)

            summary = indexer.run(
                progress_cb=report_it,
                force=force or rebuild,
                stop_flag=stop_ev,
            )
            with _build_lock:
                _build_state["last_summary"] = summary
            # The changed matrix loads now, not in the next search
            try:
                rag.preload()
            except Exception as e:  # noqa: BLE001 — the next search loads it then
                logger.debug("RAG build: matrix not preloaded: %s", e)
            q.put(
                {
                    "phase": "complete",
                    **{k: v for k, v in summary.items() if k != "errors"},
                    "errors": summary.get("errors", [])[:20],
                }
            )
        except (RagConfigError, RagUnavailable) as e:
            q.put({"phase": "error", "error": str(e)})
        except Exception as e:  # noqa
            logger.warning("RAG build error: %s", e)
            q.put({"phase": "error", "error": str(e)[:300]})
        finally:
            # #105: close this worker thread's thread-local SQLite
            # connection so it doesn't stay open until shutdown.
            try:
                db.close()
            except Exception as e:  # noqa
                logger.debug("RAG build: db.close() failed: %s", e)
            with _build_lock:
                _build_state["running"] = False
                _build_state["stop"] = None
                _build_state["progress"] = None
            q.put(None)

    t = threading.Thread(target=worker, daemon=True, name="rag-indexer")
    with _build_lock:
        _build_state["thread"] = t
    try:
        t.start()
    except Exception as e:  # noqa
        # #103: if the thread fails to start, running must not stay
        # True permanently — reset state and report the error.
        with _build_lock:
            _build_state["running"] = False
            _build_state["stop"] = None
        logger.warning("RAG build thread start failed: %s", e)
        return jsonify({"error": f"Thread start failed: {str(e)[:200]}"}), 500

    def stream():
        yield _sse({"phase": "started"})
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                # Heartbeat against proxy timeouts
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            yield _sse(item)
        yield _sse({"phase": "closed"})

    return Response(stream(), mimetype="text/event-stream", headers=_sse_headers())


@rag_bp.route("/api/rag/build/progress")
def rag_build_progress():
    """The state of the running indexing job — without SSE, for any viewer.

    ``/api/rag/build`` is a stream AND a start command: a second call while
    a run is in progress gets 409 and thus nothing at all. Anyone who opens
    settings only after the start (or reloads the page) would therefore
    have no way to see progress. This endpoint exists for that: pure
    reporting, no side effect.
    """
    with _build_lock:
        p = _build_state.get("progress")
        return jsonify(
            {
                "running": bool(_build_state["running"]),
                "started_at": _build_state["started_at"],
                "progress": p,
                "last_summary": _build_state["last_summary"],
            }
        )


@rag_bp.route("/api/rag/build/stop", methods=["POST"])
def rag_build_stop():
    """Stop a running embedding build after the current batch."""
    with _build_lock:
        ev = _build_state.get("stop")
        if _build_state.get("running") and ev is not None:
            ev.set()
            return jsonify({"ok": True, "message": "Cancellation requested"})
    return jsonify({"ok": False, "error": "No build running"})


# ── Clear ───────────────────────────────────────────────────────────────
@rag_bp.route("/api/rag/clear", methods=["POST"])
def rag_clear():
    """Delete everything in the vector store (409 while a build is running)."""
    try:
        # #096: check running under _build_lock and serialize drop_all()
        # with the build (don't clear while a build is indexing).
        with _build_lock:
            if _build_state.get("running"):
                return jsonify({"error": "Build running — please stop it first"}), 409
            rag = make_rag_db()
            rag.drop_all()
            _build_state["last_summary"] = None
        return jsonify({"ok": True})
    except Exception as e:  # noqa
        return jsonify({"error": str(e)[:300]}), 500


# ── Search ──────────────────────────────────────────────────────────────
def _flag(name: str, default: bool) -> bool:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@rag_bp.route("/api/rag/search")
def rag_search():
    """Hybrid/vector/full-text search in the RAG index.

    Query: q (required), mode=hybrid|vector|fts|meaning, k (result count),
    rerank=1 (the reranker orders the best groups — core/rerank_server.py),
    group=1 (copies of one passage in other folders collapsed into one hit).

    With rerank or group (both on by default for mode=meaning) it is the
    situation search (rag/retrieve.quick_search): candidates by meaning AND
    by wording (mode=fts: wording only), each hit with ``relevance`` (0..1,
    null where not reranked) and ``copies``; ``reranked``, ``note`` (why
    not, when asked) and ``min_score`` (rerank.quick_min_score) on top.
    Without them the answer is what it always was.
    """
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Empty search query"}), 400
    # Thousands of characters take tens of seconds to embed; an unknown mode ran hybrid
    if len(q) > 500:
        return jsonify({"error": "Search term too long (max 500 characters)"}), 400
    mode = (request.args.get("mode") or "hybrid").lower()
    if mode not in SEARCH_MODES:
        return jsonify({"error": f"mode is one of {', '.join(SEARCH_MODES)}"}), 400
    meaning = mode == "meaning"
    rerank, group = _flag("rerank", meaning), _flag("group", meaning)
    if rerank or group:
        return _situation_search(q, mode, rerank=rerank, group=group)
    cfg = get_rag_config()
    try:
        k = int(request.args.get("k") or cfg.get("top_k", 20))
    except ValueError:
        k = 20
    k = max(1, min(k, 100))

    try:
        rag = make_rag_db()
        if _index_empty(rag):
            return jsonify({"error": EMPTY_INDEX}), 400
        t0 = time.time()
        if mode == "fts":
            results = rag.search_fts(q, k=k)
        else:
            emb_ok, emb_msg = embedding_ready()
            if not emb_ok:
                return jsonify({"error": emb_msg}), 400
            # QUESTIONS get the instruction prefix, text passages don't —
            # see `prepare_question`. Measured on a test index, this
            # changed the best match for most questions.
            embed = make_query_embed_fn()
            # #108: handle an empty embedding result cleanly as 400 instead
            # of raising an IndexError (→ 500).
            qres = embed([q])
            if not qres or not qres[0]:
                return jsonify({"error": "Embedding returned no vector"}), 400
            qvec = qres[0]
            if mode == "vector":
                results = rag.search_vector(qvec, k=k)
            else:
                results = rag.search_hybrid(qvec, q, k=k, limit=k, rrf_k=cfg.get("rrf_k", 60))
        elapsed = round((time.time() - t0) * 1000)
        # Who speaks: names from the voice library instead of SPEAKER_xx
        speakers.name_rows(results)
        # Clean up results (JSON-serializable)
        display = _display_paths([r.get("transcript_key") for r in results])
        clean = []
        for r in results:
            place = display.get(r.get("transcript_key") or "", {})
            clean.append(
                {
                    "transcript_key": r.get("transcript_key"),
                    "chunk_index": r.get("chunk_index"),
                    "content": r.get("content"),
                    "speaker": r.get("speaker"),
                    "speaker_name": r.get("speaker_name"),
                    "start_sec": r.get("start_sec"),
                    "end_sec": r.get("end_sec"),
                    "language": r.get("language"),
                    "media_path": r.get("media_path"),
                    # For jumping into the transcript: the UI works with
                    # rel_path + root, the index with abs_path (docs/PATHS.md).
                    "rel_path": place.get("rel_path"),
                    "file_dir": place.get("file_dir"),
                    "score": _f(r.get("score")),
                    "vscore": _f(r.get("vscore")),
                    "fscore": _f(r.get("fscore")),
                    "match": r.get("match"),
                }
            )
        return jsonify(
            {"ok": True, "mode": mode, "count": len(clean), "elapsed_ms": elapsed, "results": clean}
        )
    except (RagConfigError, RagUnavailable) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa
        logger.warning("rag_search: %s", e)
        return jsonify({"error": str(e)[:300]}), 500


def _f(v):
    try:
        return round(float(v), 5) if v is not None else None
    except (TypeError, ValueError):
        return None


#: The answer of a meaning search before anything is embedded — as the deep
#: search gives it. A new library showed "no such table: chunks".
EMPTY_INDEX = (
    "The meaning index is empty — build it first: Settings → Search → "
    "Semantic search → Embed new transcripts"
)


def _index_empty(rag) -> bool:
    """No passage in the meaning index yet. Unknown (it can't be read) is
    not empty: the search then says what fails."""
    try:
        st = rag.stats()
    except Exception as e:  # noqa: BLE001
        logger.debug("rag_search: index not readable: %s", e)
        return False
    return not st.get("exists") or not st.get("chunks")


def _situation_search(q: str, mode: str, *, rerank: bool, group: bool):
    """``/api/rag/search`` with rerank and/or group — rag/retrieve.quick_search."""
    cfg = get_rag_config()
    try:
        k = int(request.args.get("k") or cfg.get("top_k", 20))
    except ValueError:
        k = 20
    k = max(1, min(k, 300))
    try:
        rag = make_rag_db()
        if _index_empty(rag):
            return jsonify({"error": EMPTY_INDEX}), 400
        embed = None
        if mode != "fts":
            emb_ok, emb_msg = embedding_ready()
            if not emb_ok:
                return jsonify({"error": emb_msg}), 400
            embed = make_query_embed_fn()
        out = retrieve.quick_search(
            q,
            k=k,
            mode=mode,
            rerank=rerank,
            group=group,
            db=rag,
            embed_fn=embed,
            rrf_k=int(cfg.get("rrf_k", 60) or 60),
        )
        results = out["results"]
        retrieve.place_copies(results, db)
        clean = []
        for r in results:
            clean.append(
                {
                    "transcript_key": r.get("transcript_key"),
                    "chunk_index": r.get("chunk_index"),
                    "content": r.get("content"),
                    "speaker": r.get("speaker"),
                    "speaker_name": r.get("speaker_name"),
                    "start_sec": r.get("start_sec"),
                    "end_sec": r.get("end_sec"),
                    "language": r.get("language"),
                    "media_path": r.get("media_path"),
                    "rel_path": r.get("rel_path"),
                    "file_dir": r.get("file_dir"),
                    "score": _f(r.get("score")),
                    "relevance": _f(r.get("relevance")),
                    "fused": _f(r.get("fused")),
                    "vscore": _f(r.get("vscore")),
                    "fscore": _f(r.get("fscore")),
                    "match": r.get("match"),
                    "copies": [
                        {
                            "transcript_key": c.get("transcript_key"),
                            "rel_path": c.get("rel_path"),
                            "file_dir": c.get("file_dir"),
                            "start_sec": c.get("start_sec"),
                            "end_sec": c.get("end_sec"),
                            "media_path": c.get("media_path"),
                            "chunk_index": c.get("chunk_index"),
                        }
                        for c in r.get("copies") or []
                    ],
                }
            )
        return jsonify(
            {
                "ok": True,
                "mode": mode,
                "count": len(clean),
                "elapsed_ms": out["elapsed_ms"],
                "reranked": out["reranked"],
                "grouped": out["grouped"],
                "note": out["note"],
                "min_score": out["min_score"],
                "candidates": out["candidates"],
                "voices": out.get("voices") or {},
                "results": clean,
            }
        )
    except (RagConfigError, RagUnavailable) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa
        logger.warning("rag_search (situation, %r): %s", q[:80], e)
        return jsonify({"error": str(e)[:300]}), 500


def _display_paths(keys: list) -> dict:
    """abs_path keys → ``rel_path`` and root, for jumping into the transcript
    (rag/retrieve.display_paths, with this module's database)."""
    return retrieve.display_paths(keys, db)


# ── Deep search ("find everything") ──────────────────────────────────────
@rag_bp.route("/api/rag/deep", methods=["POST"])
def rag_deep_start():
    """Start a deep search for a situation. Body: ``query``.

    It runs in the LLM queue (rag/deep_search.py): a plan, wide candidates,
    the reranker's order, then the language model checks one passage per
    call. Answers at once with the search's ``id``; the page then asks
    ``GET /api/rag/deep/<id>?since=n``.
    """
    data = request.get_json(silent=True)
    data = data if isinstance(data, dict) else {}
    try:
        s = deep_search.create(str(data.get("query") or ""))
    except deep_search.DeepSearchError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa
        logger.warning("Deep search not started: %s", e)
        return jsonify({"error": str(e)[:300]}), 500
    return jsonify(
        {"ok": True, "id": s["id"], "status": s["status"], "entry_id": s.get("entry_id")}
    )


@rag_bp.route("/api/rag/deep")
def rag_deep_list():
    """The deep searches, newest first — state and counts, no results."""
    return jsonify({"searches": deep_search.listing()})


@rag_bp.route("/api/rag/deep/<did>")
def rag_deep_get(did):
    """One deep search: state, plan, progress, and its results from
    ``since`` on (``next`` is the cursor for the next call)."""
    try:
        since = int(request.args.get("since") or 0)
    except ValueError:
        since = 0
    try:
        return jsonify(deep_search.public(deep_search.load(did), since))
    except deep_search.DeepSearchError as e:
        return jsonify({"error": str(e)}), 404


@rag_bp.route("/api/rag/deep/<did>/stop", methods=["POST"])
def rag_deep_stop(did):
    """Stop a deep search — what it found so far stays."""
    try:
        return jsonify({"ok": deep_search.stop(did)})
    except deep_search.DeepSearchError as e:
        return jsonify({"error": str(e)}), 404


@rag_bp.route("/api/rag/deep/<did>", methods=["DELETE"])
def rag_deep_delete(did):
    """Delete a deep search (a running one is stopped first)."""
    try:
        if not deep_search.delete(did):
            return jsonify({"error": "unknown deep search"}), 404
    except deep_search.DeepSearchError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"ok": True})


# ── Export: save the vector file ────────────────────────────────────────
@rag_bp.route("/api/rag/export")
def rag_export():
    """Download the vector store as a file.

    Previously this used a `pg_dump` call with a temp directory, a
    password in the environment, a 30-minute timeout, and cleanup paths
    for four kinds of errors. A SQLite file IS the backup — it comes down
    to copying a file.

    `VACUUM INTO` and not just send_file on the path: the indexer can be
    writing concurrently, and a file with a half-written WAL would be a
    copy that only turns out broken when restored. `VACUUM INTO` writes a
    self-consistent snapshot and is smaller on top of that.
    """
    try:
        rag = make_rag_db()
    except Exception as e:  # noqa
        return jsonify({"error": str(e)[:200]}), 500

    tmpdir = tempfile.mkdtemp(prefix="wtm_ragexport_")
    keep = False
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(tmpdir, f"whisper_tm_rag_{ts}.db")
        try:
            with rag.connection() as c:
                c.execute("VACUUM INTO ?", (outfile,))
        except Exception as e:  # noqa
            logger.warning("RAG export failed: %s", e)
            return jsonify({"error": str(e)[:300]}), 500

        resp = send_file(
            outfile,
            as_attachment=True,
            download_name=os.path.basename(outfile),
            mimetype="application/vnd.sqlite3",
        )
        # send_file marks the response direct_passthrough, and Werkzeug then
        # never calls its close() — the callback below would never run, and
        # every export left a copy of the whole store in the temp folder
        # (in RAM with a tmpfs /tmp). Same as the database export (api/settings.py).
        resp.direct_passthrough = False

        @resp.call_on_close
        def _cleanup():
            shutil.rmtree(tmpdir, ignore_errors=True)

        keep = True  # tmpdir stays alive until send_file finishes streaming
        return resp
    finally:
        # Clean up on all error paths; on success _cleanup does it.
        if not keep:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Export: transcripts as portable JSONL / ZIP ────────────────────────────
def _export_record(cand: dict) -> tuple[str, dict | None]:
    """Load one transcript for the export — by its identity.

    ``rel_path`` is relative to the root that was open when the file was
    transcribed, not to the one open now. Joining it with the current
    media directory, as an earlier version of this export did, found nothing
    as soon as another folder was open: every transcript was skipped and
    the download was an empty file.
    """
    key = cand["abs_path"] or cand["rel_path"]
    data = db.get_transcript(key, media_dir=cand["media_dir"] or None)
    if data:
        data.setdefault("filename", cand["rel_path"])
        data["_rel_path"] = cand["rel_path"]
        data["_abs_path"] = cand["abs_path"]
    return key, data


#: What one file of the ZIP holds: the transcript as JSON, as readable text
#: (times and speakers), or as subtitles
EXPORT_AS = ("json", "txt", "srt")


class _ExportFilter(ValueError):
    """A filter of the export that can't be read."""


def _export_filter() -> dict:
    """The export's filters from the query: ``folder=open`` — the folder open
    in the app (with ``subfolders=0`` only its own files) — and ``since`` /
    ``until`` (unix seconds) on when a recording came into the library."""
    out: dict = {"root": "", "subfolders": True, "since": None, "until": None}
    if (request.args.get("folder") or "") == "open":
        out["root"] = paths.canon(config.media_dir).rstrip("/")
        out["subfolders"] = request.args.get("subfolders", "1") not in ("0", "false")
    for k in ("since", "until"):
        v = request.args.get(k)
        if v not in (None, ""):
            try:
                out[k] = float(v)
            except ValueError:
                raise _ExportFilter(f"{k} must be a number (unix seconds)") from None
    return out


def _export_candidates(f: dict) -> list[dict]:
    if not (f["root"] or f["since"] is not None or f["until"] is not None):
        return db.get_rag_candidates()
    rows = db.export_candidates(f["root"], f["since"], f["until"])
    if f["root"] and not f["subfolders"]:
        rows = [r for r in rows if "/" not in r["abs_path"][len(f["root"]) + 1 :]]
    return rows


@rag_bp.route("/api/export/transcripts/count")
def export_transcripts_count():
    """How many transcripts an export with these filters holds (the export
    dialog shows it before the download), with a few names."""
    if not db:
        return jsonify({"error": "No database"}), 500
    try:
        f = _export_filter()
        rows = _export_candidates(f)
    except _ExportFilter as e:
        return jsonify({"error": str(e)}), 400
    names = [os.path.basename(r["abs_path"] or r["rel_path"]) for r in rows[:6]]
    return jsonify({"count": len(rows), "folder": f["root"], "examples": names})


@rag_bp.route("/api/export/transcripts")
def export_transcripts():
    """Export transcripts portably — all of them, or those the filters
    leave (see :func:`_export_filter`).

    Query: format=jsonl (default) | zip.
      jsonl – one JSON object per line (streaming, ideal for large amounts)
      zip   – one file per transcript in a ZIP; ``as`` = json (default) |
              txt (readable: times and speakers) | srt (subtitles)
    """
    if not db:
        return jsonify({"error": "No database"}), 500
    fmt = (request.args.get("format") or "jsonl").lower()
    as_ = (request.args.get("as") or "json").lower()
    if as_ not in EXPORT_AS:
        return jsonify({"error": f"as must be one of {', '.join(EXPORT_AS)}"}), 400
    ts = time.strftime("%Y%m%d_%H%M%S")

    try:
        f = _export_filter()
        candidates = _export_candidates(f)
    except _ExportFilter as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa
        return jsonify({"error": str(e)[:200]}), 500

    if fmt == "zip":
        return _export_transcripts_zip(candidates, ts, as_=as_, root=f["root"])

    # JSONL streaming
    def generate():
        errors = 0
        for cand in candidates:
            try:
                key, data = _export_record(cand)
            except Exception as e:  # noqa
                logger.warning("Export JSONL: skipping %s: %s", cand["abs_path"], str(e)[:120])
                errors += 1
                continue
            if not data:
                errors += 1
                continue
            yield json.dumps(data, ensure_ascii=False) + "\n"
        # #107: signal skipped/errored records (not silently)
        if errors:
            logger.warning(
                "Export JSONL: %d of %d transcripts skipped (error/empty)", errors, len(candidates)
            )

    headers = {
        "Content-Disposition": f'attachment; filename="whisper_tm_transcripts_{ts}.jsonl"',
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), mimetype="application/x-ndjson", headers=headers)


def _clock(sec) -> str:
    sec = max(0, int(sec or 0))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _speaker_of(data: dict):
    """A segment's speaker as the transcript names it — "Speaker 2" for an
    unnamed one, nothing without diarization."""
    import re as _re

    names = data.get("speaker_names") or {}

    def name(s: str) -> str:
        if not s or s in ("—", "-"):
            return ""
        if names.get(s):
            return str(names[s])
        m = _re.match(r"^SPEAKER_(\d+)$", s)
        return f"Speaker {int(m.group(1)) + 1}" if m else s

    return name


def _as_text(data: dict) -> str:
    """Readable: one line per segment, with its time and speaker — as the
    transcript view's own TXT export."""
    who = _speaker_of(data)
    lines = []
    for s in data.get("segments") or []:
        name = who(s.get("speaker") or "")
        lines.append(
            f"[{_clock(s.get('start'))} - {_clock(s.get('end'))}] "
            + (f"{name}: " if name else "")
            + (s.get("text") or "").strip()
        )
    return "\n".join(lines) + "\n"


def _as_srt(data: dict) -> str:
    from whisper_tm.core.export import _srt_time

    who = _speaker_of(data)
    out = []
    for i, s in enumerate(data.get("segments") or [], 1):
        start = s.get("start") or 0
        end = max(start, s.get("end") or 0)
        name = who(s.get("speaker") or "")
        out.append(
            f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n"
            + (f"{name}: " if name else "")
            + (s.get("text") or "").strip()
            + "\n"
        )
    return "\n".join(out)


def _export_transcripts_zip(candidates, ts, as_: str = "json", root: str = ""):
    import re as _re
    import zipfile

    tmpdir = tempfile.mkdtemp(prefix="wtm_txexport_")
    outfile = os.path.join(tmpdir, f"whisper_tm_transcripts_{ts}.zip")
    keep = False
    try:
        errors = 0
        try:
            taken: set[str] = set()
            with zipfile.ZipFile(outfile, "w", zipfile.ZIP_DEFLATED) as zf:
                for cand in candidates:
                    try:
                        key, data = _export_record(cand)
                    except Exception as e:  # noqa
                        logger.warning(
                            "Export ZIP: skipping %s: %s", cand["abs_path"], str(e)[:120]
                        )
                        errors += 1
                        continue
                    if not data:
                        errors += 1
                        continue
                    # Named after the absolute path: two files with the
                    # same rel_path under different roots would otherwise
                    # overwrite each other in the archive. One folder
                    # exported: below it, its subfolders kept.
                    name = key[len(root) + 1 :] if root and key.startswith(root + "/") else key
                    safe = _re.sub(r"[^\w.() /-]", "_", name).strip().lstrip("/")
                    if as_ == "json":
                        if not safe.endswith(".json"):
                            safe += ".json"
                    else:
                        safe = os.path.splitext(safe)[0] + "." + as_
                    # "talk.mp4" and "talk.m4a" both became "talk.txt"
                    stem, ext = os.path.splitext(safe)
                    n = 1
                    while safe in taken:
                        n += 1
                        safe = f"{stem} ({n}){ext}"
                    taken.add(safe)
                    text = (
                        json.dumps(data, ensure_ascii=False, indent=1)
                        if as_ == "json"
                        else _as_text(data)
                        if as_ == "txt"
                        else _as_srt(data)
                    )
                    zf.writestr(safe, text)
        except Exception as e:  # noqa
            return jsonify({"error": str(e)[:200]}), 500
        # #107: signal skipped/errored records (not silently)
        if errors:
            logger.warning(
                "Export ZIP: %d of %d transcripts skipped (error/empty)", errors, len(candidates)
            )

        resp = send_file(
            outfile,
            as_attachment=True,
            download_name=os.path.basename(outfile),
            mimetype="application/zip",
        )

        @resp.call_on_close
        def _cleanup():
            shutil.rmtree(tmpdir, ignore_errors=True)

        keep = True  # tmpdir stays alive until send_file finishes streaming (_cleanup)
        return resp
    finally:
        # #090: clean up tmpdir on error paths; on success via _cleanup.
        if not keep:
            shutil.rmtree(tmpdir, ignore_errors=True)

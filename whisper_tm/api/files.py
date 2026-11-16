"""
Paginated file list (UI rework 5.1).

Three lean endpoints replace the "whole library in one response"
pattern of /api/scan?walk=false (seconds of latency and a frozen browser
on large libraries):

- GET /api/files        — ONE page of the list (LIMIT/OFFSET, sort, tab, filter)
- GET /api/files/stats  — Aggregates for sidebar counters + directory statistics
- GET /api/files/names  — only the rel_paths of the active filter, for
                          "select all filtered" actions (Ctrl+A, batch, retry,
                          bulk delete, shift-range across page boundaries)

The scope is always the current media directory (config.media_dir) — the
DB holds several roots, scoping happens via abs_path (docs/PATHS.md).
The row shape matches exactly /api/scan_entry (scanner.db_row_to_entry).
"""

import logging
import time

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.extensions import db
from whisper_tm.scanner import db_row_to_entry

logger = logging.getLogger(__name__)

files_bp = Blueprint("files", __name__)

_TABS = {"all", "pending", "done", "processing", "error", "cancelled", "orphan", "unembedded"}

# Embedded RAG keys, cached briefly: the unembedded tab needs them
# on every page fetch, but the set only changes during an index run.
_emb_cache: dict = {"timestamp": 0.0, "keys": frozenset(), "active": False}
_EMB_TTL = 30.0


def _embedded_keys() -> tuple[frozenset, bool]:
    """(key set, rag_active) — from the vector store, with TTL cache."""
    now = time.monotonic()
    if now - _emb_cache["timestamp"] < _EMB_TTL:
        return _emb_cache["keys"], _emb_cache["active"]
    keys, active = frozenset(), False
    try:
        from whisper_tm.rag.service import get_rag_config, make_rag_db

        if get_rag_config().get("enabled"):
            rag = make_rag_db()
            with rag.connection() as c:
                rows = c.execute(
                    "SELECT transcript_key FROM indexed_transcripts WHERE chunk_count > 0"
                ).fetchall()
            keys = frozenset(r["transcript_key"] for r in rows)
            active = True
    except Exception as e:  # noqa — RAG off/unreachable: empty filter
        logger.debug("Embedded keys not loadable: %s", e)
    _emb_cache.update(timestamp=now, keys=keys, active=active)
    return keys, active


def _list_filters() -> dict | None:
    """The filter panel's parameters (all optional; a value that does not
    parse switches only that filter off):

    f_type=audio|video, f_len_min / f_len_max (seconds), f_added (days),
    f_lang (language code), f_has=summary,translation,warnings,transcript,
    f_sub (a subfolder of the open folder), f_people=Name,Name (recordings
    one of them speaks in, from the voice library).
    """
    a = request.args
    f: dict = {}
    if a.get("f_type") in ("audio", "video"):
        f["type"] = a["f_type"]
    for key, name in (
        ("f_len_min", "len_min"),
        ("f_len_max", "len_max"),
        ("f_added", "added_days"),
    ):
        try:
            v = float(a.get(key, ""))
            if 0 <= v < 1e9:
                f[name] = v
        except ValueError:
            pass
    lang = (a.get("f_lang") or "").strip()[:12]
    if lang:
        f["language"] = lang
    known = ("summary", "translation", "warnings", "transcript", "no_audio", "skipped", "empty")
    has = [h for h in (a.get("f_has") or "").split(",") if h in known]
    if has:
        f["has"] = has
    sub = (a.get("f_sub") or "").strip()[:1000]
    if sub:
        f["subfolder"] = sub
    people = [n.strip() for n in (a.get("f_people") or "").split(",") if n.strip()][:20]
    if people:
        spots: set[str] = set()
        try:
            from whisper_tm.database.speaker import speaker_db

            for name in people:
                spots.update(p for p, _ in speaker_db().person_spots(name))
        except Exception as e:  # noqa — no voice library: nobody matches
            logger.warning("Person filter not applicable: %s", e)
        f["people"] = people
        f["people_paths"] = sorted(spots)
    return f or None


def _list_params() -> dict:
    """Shared query parameters of /api/files and /api/files/names."""
    tab = request.args.get("tab", "all").strip() or "all"
    if tab not in _TABS:
        tab = "all"
    sort = request.args.get("sort", "name").strip() or "name"
    sort_dir = request.args.get("dir", "asc").strip() or "asc"
    q = request.args.get("q", "").strip()
    return {
        "tab": tab,
        "name_filter": q,
        "sort": sort,
        "sort_dir": sort_dir,
        "filters": _list_filters(),
    }


@files_bp.route("/api/files")
def files_page():
    """One page of the file list.

    Query: offset, limit (1–5000), sort (name|size|modified|tx_date|
    tx_word_count), dir (asc|desc), tab (quick filter), q (name filter),
    and the filter panel's f_type, f_len_min, f_len_max, f_added, f_lang,
    f_has, f_sub, f_people.
    """
    if not db:
        return jsonify({"error": "No database configured"}), 501
    p = _list_params()
    offset = request.args.get("offset", type=int, default=0)
    limit = request.args.get("limit", type=int, default=200)
    emb = None
    if p["tab"] == "unembedded":
        emb, active = _embedded_keys()
        if not active:
            return jsonify(
                {
                    "directory": str(config.media_dir),
                    "total": 0,
                    "offset": 0,
                    "limit": limit,
                    "files": [],
                }
            )
    root = str(config.media_dir)
    page = db.get_media_files_page(root, limit=limit, offset=offset, embedded_keys=emb, **p)
    entries = [db_row_to_entry(r, root) for r in page["rows"]]
    return jsonify(
        {
            "directory": root,
            "total": page["total"],
            "offset": offset,
            "limit": limit,
            "files": [e for e in entries if e],
        }
    )


@files_bp.route("/api/files/stats")
def files_stats():
    """Aggregates of the current media directory (counters + sums).

    Replaces the earlier client-side computation over the full list
    (_computeFileAggregates): with pagination the browser only ever sees one
    window and can no longer count itself.
    """
    if not db:
        return jsonify({"error": "No database configured"}), 501
    stats = db.get_media_stats_under(str(config.media_dir))
    try:
        from whisper_tm.core import eta

        stats["eta"] = eta.library(db, str(config.media_dir), stats)
    except Exception as e:  # noqa — an estimate must not break the counters
        logger.warning("Library time estimate failed: %s", e)
        stats["eta"] = None
    # No "unembedded" count any more: the sidebar chip it fed is
    # gone — how much is embedded is Settings → Search's business — and
    # counting it listed every finished file of the library on each call.
    return jsonify(stats)


@files_bp.route("/api/files/names")
def files_names():
    """All rel_paths of the active filter, in list order.

    For actions over "all filtered": the client only holds one
    window and can no longer build the set itself.
    """
    if not db:
        return jsonify({"error": "No database configured"}), 501
    p = _list_params()
    emb = None
    if p["tab"] == "unembedded":
        emb, _ = _embedded_keys()
    with_status = request.args.get("with_status") == "1"
    lines = db.get_media_names_under(
        str(config.media_dir), embedded_keys=emb, with_status=with_status, **p
    )
    if with_status:
        return jsonify(
            {"names": [z[0] for z in lines], "status": [z[1] for z in lines], "total": len(lines)}
        )
    return jsonify({"names": lines, "total": len(lines)})

"""
Search API: FTS5-based full-text search, stats, speaker filter.

Dual mode:
- Primary: SQLite FTS5 (milliseconds, highlighting, pagination)
- Fallback: JSON scan (if DB empty or disabled)
"""

import fnmatch
import json
import logging
import re
import sqlite3
from datetime import UTC
from pathlib import Path

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.database import _root_of
from whisper_tm.extensions import db

logger = logging.getLogger(__name__)

search_bp = Blueprint("search", __name__)


def _resolve_file_dir(rel_path: str) -> str:
    """Return the navigable directory for rel_path.

    Once a fourth, separate resolution path with basename search and
    directory walk. Since schema v8 there's exactly one: the identity
    (abs_path) via the central resolver (docs/PATHS.md). The walk
    disappeared without replacement — it ran per search hit and could
    slow the whole response on a NAS with missing files.
    """
    if db:
        try:
            with db.connection() as conn:
                row = db.resolve_transcript(
                    conn, rel_path, str(config.media_dir), columns="abs_path, media_dir"
                )
            if row:
                # The root rel_path is relative to — from the identity where
                # the stored media_dir doesn't fit (database._root_of)
                root = _root_of(rel_path, row["abs_path"], row["media_dir"])
                if root:
                    return root
                if row["abs_path"]:
                    # Without a stored root: the file's own folder.
                    return str(Path(row["abs_path"]).parent)
        except Exception as _e:  # noqa
            pass
    return str(config.media_dir)


#: What SQLite says when the FTS5 query itself is malformed — the user's
#: input, so a 400. Everything else (locked, I/O, a bug) is the server's.
_FTS_SYNTAX_MARKERS = (
    "fts5: syntax error",
    "unterminated string",
    "no such column",
    "unknown special query",
)


def _search_failed(exc: Exception, q: str):
    """The response for a failed search — 400 for a syntax error, else 500.

    Every error used to come back as "Invalid search syntax": a locked
    database (a long write elsewhere) looked like a typo in the query and
    left no trace in the log.
    """
    msg = str(exc)
    if isinstance(exc, sqlite3.OperationalError) and any(m in msg for m in _FTS_SYNTAX_MARKERS):
        return jsonify(
            {
                "error": "Invalid search syntax (FTS5) — check quotation marks.",
                "query": q,
            }
        ), 400
    logger.warning("Search for %r failed: %s", q, msg, exc_info=True)
    return jsonify({"error": f"Search failed: {msg[:300]}", "query": q}), 500


#: The file filter's length: a full path (the MCP server names a recording
#: outside the open folder that way) is often longer than 200 characters.
FILE_FILTER_CHARS = 1000
#: Numbers from a request are kept below SQLite's integers
INT_MAX = 2**62
#: The speakers' list for the filter is read again after this long — a
#: DISTINCT over every segment takes seconds per call on a large library
SPEAKERS_TTL_SEC = 60.0
_speakers_cache: dict = {"at": -1e9, "db": None, "speakers": []}


@search_bp.route("/api/search")
def global_search():
    """Full-text search across all transcripts.

    Query parameters:
        q: Search term (min 2 characters)
        file: File name filter (substring)
        speaker: Speaker filter (exact)
        limit: Max results (default 500)
        offset: Pagination offset
        mode: "fts" (DB) or "json" (file scan), default "auto"
    """
    q = request.args.get("q", "").strip()
    file_filter = request.args.get("file", "").strip()[:FILE_FILTER_CHARS]
    speaker_filter = request.args.get("speaker", "").strip()[:100]
    # search.max_results is the default when the caller sets no limit.
    # The hard cap of 5000 stays: a search must not make the response
    # arbitrarily large, no matter what the config says.
    _max_default = config.get("search", "max_results", default=500) or 500
    try:
        _max_default = max(1, int(_max_default))
    except (TypeError, ValueError):
        _max_default = 500
    # At least 1: SQLite reads LIMIT -5 as no limit (a response of many MB); the
    # offset below SQLite's integers (a huge one was an OverflowError, a 500)
    limit = max(
        1, min(request.args.get("limit", type=int, default=_max_default) or _max_default, 5000)
    )
    offset = min(max(request.args.get("offset", type=int, default=0) or 0, 0), INT_MAX)
    mode = request.args.get("mode", "auto")
    # file_exact=1: rel_path must match exactly (drill-down from the hit card),
    # instead of the usual substring search in the file name.
    file_exact = request.args.get("file_exact") in ("1", "true", "yes")

    if not q or len(q) < 2:
        # File mode: without search term search over METADATA —
        # "which files from then, so long, so fast processed"
        # instead of "where was X said". Triggers if any filter or
        # sorting is set.
        fo = _filter_opts_from_request()
        _sort, _direction = _sort_from_request()
        # mode=files forces the file list even without filter —
        # "just list and sort all files".
        if db is not None and (
            fo or file_filter or speaker_filter or _sort or request.args.get("mode") == "files"
        ):
            result = db.file_search(
                file_filter=file_filter,
                speaker_filter=speaker_filter,
                filter_opts=fo,
                sort=_sort,
                sort_dir=_direction,
                limit=limit,
                offset=offset,
            )
            for item in result.get("results", []):
                item["file_dir"] = item.get("media_dir") or _resolve_file_dir(item["filename"])
            result["query"] = q
            result["media_dir"] = str(config.media_dir)
            return jsonify(result)
        return jsonify({"results": [], "files": 0, "total": 0})
    if len(q) > 500:
        return jsonify({"error": "Search term too long (max 500 characters)"}), 400

    # Decide mode
    use_db = False
    if mode == "fts":
        use_db = True
    elif mode == "json":
        use_db = False
    elif mode == "auto":
        # Use DB when FTS is enabled and transcripts exist
        use_db = db is not None and config.get("database", "fts_enabled", default=True)

    if use_db and db:
        return _search_fts(q, file_filter, speaker_filter, limit, offset, file_exact)
    else:
        return _search_json(q, file_filter, speaker_filter, limit, offset)


def _filter_opts_from_request() -> dict:
    """Metadata filters from query parameters (all optional).

    Parse numbers defensively — a broken value disables only that one
    filter instead of answering the search with 400/500.
    Dates come as ISO date (YYYY-MM-DD) and map to Unix time;
    `date_to` includes the whole day.
    """
    fo: dict = {}
    for key in ("duration_min", "duration_max", "tempo_min", "tempo_max"):
        raw = request.args.get(key)
        if raw:
            try:
                fo[key] = float(raw)
            except ValueError:
                pass
    for key in ("word_count_min", "word_count_max"):
        raw = request.args.get(key)
        if raw:
            try:
                fo[key] = min(max(int(raw), 0), INT_MAX)  # SQLite's integers
            except ValueError:
                pass
    from datetime import datetime

    for key, day_end in (("date_from", False), ("date_to", True)):
        raw = (request.args.get(key) or "").strip()
        if raw:
            try:
                dt = datetime.strptime(raw, "%Y-%m-%d")
                ts = dt.replace(tzinfo=UTC).timestamp()
                fo[key] = ts + (86399 if day_end else 0)
            except ValueError:
                pass
    if request.args.get("transcribed_only") in ("1", "true", "yes"):
        fo["transcribed_only"] = True

    # ── Narrow to ONE PERSON ───────────────────────────────────────────
    #
    # "Search for X, but only where speaker Alice says it". Places are in
    # `speaker.db`, segments in `library.db` — two databases, so look up
    # here and pass the list on instead of joining in the query.
    #
    # Empty list is EXPLICITLY a filter: who searches for a person with no
    # voice assigned yet should see zero hits, not the whole library.
    person = (request.args.get("person") or "").strip()
    if person:
        try:
            from whisper_tm.database.speaker import speaker_db

            fo["person_spots"] = speaker_db().person_spots(person)
            fo["person_segments_only"] = request.args.get("person_segments") in ("1", "true", "yes")
        except Exception as e:  # noqa — without the speaker DB the search stays
            logger.warning("Person filter %r not applicable: %s", person, e)
    return fo


def _sort_from_request() -> tuple[str, str]:
    sort = (request.args.get("sort") or "").strip()
    if sort not in ("date", "duration", "size", "word_count", "speed"):
        sort = ""
    direction = (request.args.get("direction") or "desc").strip()
    return sort, ("asc" if direction == "asc" else "desc")


def _fuzzy_threshold() -> float:
    """Fuzzy threshold from config, 0.0 if disabled.

    Only when enabled; threshold clamped to a sensible range. Below ~0.5
    difflib returns nearly arbitrary words.
    """
    if not config.get("search", "fuzzy_enabled", default=False):
        return 0.0
    try:
        val = float(config.get("search", "fuzzy_threshold", default=0.8))
    except (TypeError, ValueError):
        val = 0.8
    return min(1.0, max(0.5, val))


def _search_fts(q, file_filter, speaker_filter, limit, offset, file_exact=False):
    """FTS5-based search (fast, with highlighting)."""
    try:
        try:
            _ctx = max(0, int(config.get("search", "context_segments", default=0) or 0))
        except (TypeError, ValueError):
            _ctx = 0
        _fuzzy = _fuzzy_threshold()
        _sort, _direction = _sort_from_request()
        result = db.search(
            query=q,
            file_filter=file_filter,
            speaker_filter=speaker_filter,
            limit=limit,
            offset=offset,
            context=_ctx,
            fuzzy=_fuzzy,
            file_exact=file_exact,
            filter_opts=_filter_opts_from_request(),
            sort=_sort,
            sort_dir=_direction,
        )
    except Exception as _e:  # noqa
        # Invalid FTS5 syntax (e.g. unbalanced '"' or bare AND) → clean 400;
        # anything else is a server error and is logged.
        return _search_failed(_e, q)
    for item in result.get("results", []):
        # Prefer stored media_dir; fallback to disk walk
        item["file_dir"] = item.get("media_dir") or _resolve_file_dir(item["filename"])
    result["query"] = q
    result["mode"] = "fts"
    result["media_dir"] = str(config.media_dir)
    result["files"] = _file_hits(q, filter_opts=_filter_opts_from_request())
    return jsonify(result)


def _file_hits(q: str, limit: int = 40, filter_opts: dict | None = None) -> list[dict]:
    """Hits in file name and embedded title.

    Complement to full-text search: that searches SPOKEN text. Who searches
    for a series title, recording date, or downloaded video name finds nothing
    there — the term is in the filename or file metadata.
    """
    if not db:
        return []
    try:
        hits = db.search_media_files(q, limit=limit, filter_opts=filter_opts)
    except Exception as e:  # noqa — full-text search must not depend on this
        logger.debug("File name search failed: %s", e)
        return []
    root = str(config.media_dir)
    return [
        {
            "filename": t["rel_path"],
            # The file's own root: rel_path is relative to it, not always to
            # media_dir (the joined path didn't exist)
            "file_dir": _root_of(t["rel_path"], t["abs_path"], None) or root,
            "title": t["media_title"] or "",
            "display_name": t["display_name"] or t["rel_path"].rsplit("/", 1)[-1],
            "duration_sec": t["duration_sec"] or 0,
            "is_video": bool(t["is_video"]),
            "has_transcript": bool(t["has_transcript"]),
            "status": t["status"] or "pending",
        }
        for t in hits
    ]


def _search_json(q, file_filter, speaker_filter, limit, offset=0):
    """JSON-based search (fallback, compatible with old code)."""
    # Quoted phrases → word-boundary match, unquoted → substring
    quoted_terms = []
    plain_terms = []

    for m in re.finditer(r'"([^"]+)"', q):
        phrase = m.group(1).strip()
        if len(phrase) >= 2:
            quoted_terms.append(phrase.lower())

    plain_part = re.sub(r'"[^"]*"', "", q).strip()
    for t in plain_part.split():
        t = t.strip()
        if len(t) >= 2:
            plain_terms.append(t.lower())

    if not quoted_terms and not plain_terms:
        return jsonify({"results": [], "files": 0, "total": 0})

    quoted_patterns = []
    for phrase in quoted_terms:
        escaped = re.escape(phrase)
        try:
            quoted_patterns.append(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE))
        except re.error:
            quoted_patterns.append(None)

    all_terms = plain_terms + quoted_terms
    results = []
    file_set = set()
    tdir = config.transcript_dir

    if not tdir.exists():
        return jsonify({"results": [], "files": 0, "total": 0, "mode": "json"})

    for tp in sorted(tdir.rglob("*.json")):
        if tp.name.startswith("."):
            continue
        try:
            with open(tp, encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("complete"):
                continue
            fname = data.get("filename", tp.stem)

            if file_filter:
                if not fnmatch.fnmatch(fname.lower(), f"*{file_filter.lower()}*"):
                    if not fnmatch.fnmatch(fname.lower(), file_filter.lower()):
                        continue

            sp_names = data.get("speaker_names", {})

            for seg in data.get("segments", []):
                text = seg.get("text", "")
                text_lower = text.lower()

                if not all(term in text_lower for term in plain_terms):
                    continue

                ok = True
                for i, pattern in enumerate(quoted_patterns):
                    found = pattern.search(text) if pattern else quoted_terms[i] in text_lower
                    if not found:
                        ok = False
                        break
                if not ok:
                    continue

                seg_speaker = seg.get("speaker", "—")
                if speaker_filter and seg_speaker != speaker_filter:
                    sp_display = sp_names.get(seg_speaker, seg_speaker)
                    if sp_display != speaker_filter:
                        continue

                file_set.add(fname)
                sp_display = sp_names.get(seg_speaker, seg_speaker)
                results.append(
                    {
                        "filename": fname,
                        "file_dir": _resolve_file_dir(fname),
                        "speaker": seg_speaker,
                        "speaker_name": sp_display,
                        "start": seg.get("start", 0),
                        "end": seg.get("end", seg.get("start", 0)),
                        "start_fmt": seg.get("start_fmt", ""),
                        "end_fmt": seg.get("end_fmt", ""),
                        "text": text,
                    }
                )
        except Exception as _e:  # noqa
            continue

    results.sort(key=lambda r: (r["filename"], r["start"]))
    total = len(results)
    return jsonify(
        {
            "query": q,
            "terms": all_terms,
            "results": results[offset : offset + limit],
            "files": len(file_set),
            "total": total,
            "truncated": total > offset + limit,
            "mode": "json",
            "media_dir": str(config.media_dir),
        }
    )


@search_bp.route("/api/search/facets")
def search_facets():
    """Hit distribution across ALL hits (not just the loaded page).

    Basis for heatmaps in the large search view: per file hit count
    plus time histogram, plus speaker distribution. Deliberately separate from
    /api/search so the result list can stay paginated while the overview
    still knows the full inventory.

    Query parameters:
        q, file, speaker: like /api/search
        buckets: time histogram resolution (default 40, max 200)
    """
    q = request.args.get("q", "").strip()
    file_filter = request.args.get("file", "").strip()[:FILE_FILTER_CHARS]
    speaker_filter = request.args.get("speaker", "").strip()[:100]
    buckets = request.args.get("buckets", type=int, default=40) or 40
    # Cap for file list. Search view works with 500 — it only shows the
    # densest files anyway. Who wants to DIVIDE the hits (MCP `search_plan`)
    # needs them all: a common word can have thousands of hits in over a
    # thousand files, and 500 files cover only part of them — the remainder
    # would silently fall off the table.
    max_files = request.args.get("max_files", type=int, default=500) or 500

    if not q or len(q) < 2:
        return jsonify({"files": [], "total": 0, "file_count": 0, "buckets": buckets})
    if len(q) > 500:
        return jsonify({"error": "Search term too long (max 500 characters)"}), 400
    if not db or not config.get("database", "fts_enabled", default=True):
        # Without FTS there's no affordable aggregation — search view manages
        # without facets and then hides the heatmap.
        return jsonify(
            {"files": [], "total": 0, "file_count": 0, "buckets": buckets, "unavailable": True}
        )

    try:
        result = db.search_facets(
            query=q,
            file_filter=file_filter,
            speaker_filter=speaker_filter,
            fuzzy=_fuzzy_threshold(),
            buckets=buckets,
            max_files=max_files,
            filter_opts=_filter_opts_from_request(),
        )
    except Exception as _e:  # noqa — same handling as /api/search
        return _search_failed(_e, q)

    # ONE connection for all files. Previously _resolve_file_dir opened its own
    # per file — with 500 facet files that's 500 connections per call,
    # and the call happens on every filter change.
    missing = [i for i in result.get("files", []) if not i.get("media_dir")]
    if missing and db:
        try:
            with db.connection() as conn:
                for item in missing:
                    row = db.resolve_transcript(
                        conn, item["filename"], str(config.media_dir), columns="abs_path, media_dir"
                    )
                    root = row and _root_of(item["filename"], row["abs_path"], row["media_dir"])
                    if root:
                        item["media_dir"] = root
                    elif row and row["abs_path"]:
                        item["media_dir"] = str(Path(row["abs_path"]).parent)
        except Exception as _e:  # noqa
            pass
    for item in result.get("files", []):
        item["file_dir"] = item.get("media_dir") or str(config.media_dir)
    result["query"] = q
    result["media_dir"] = str(config.media_dir)
    return jsonify(result)


@search_bp.route("/api/search/speakers")
def search_speakers():
    """All available speakers for filter dropdown."""
    if db:
        import time

        now = time.monotonic()
        c = _speakers_cache
        if c["db"] is db and now - c["at"] < SPEAKERS_TTL_SEC:
            return jsonify({"speakers": c["speakers"]})
        with db.connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT s.speaker, t.speaker_names
                FROM segments s
                JOIN transcripts t ON t.id = s.transcript_id
                WHERE s.speaker != '—' AND t.complete = 1
            """).fetchall()

            speakers = set()
            for row in rows:
                speakers.add(row["speaker"])
                names = json.loads(row["speaker_names"] or "{}")
                if row["speaker"] in names:
                    speakers.add(names[row["speaker"]])

        c.update(at=now, db=db, speakers=sorted(speakers))
        return jsonify({"speakers": c["speakers"]})

    # Fallback: JSON scan
    speakers = set()
    tdir = config.transcript_dir
    if tdir.exists():
        for tp in tdir.rglob("*.json"):
            if tp.name.startswith("."):
                continue
            try:
                with open(tp, encoding="utf-8") as f:
                    data = json.load(f)
                if not data.get("complete"):
                    continue
                sp_names = data.get("speaker_names", {})
                for seg in data.get("segments", []):
                    sp = seg.get("speaker")
                    if sp and sp != "—":
                        speakers.add(sp)
                        if sp in sp_names:
                            speakers.add(sp_names[sp])
            except Exception as _e:  # noqa
                continue

    return jsonify({"speakers": sorted(speakers)})


# ── Context of a hit ──────────────────────────────────────────────────────
@search_bp.route("/api/search/context")
def search_context():
    """The segments around a hit — for the reading column.

    Why its own endpoint and not ``context_segments`` in search:
    that setting adds context to EVERY hit. With 200 loaded hits per page
    that's 200 × 2 × 5 additional segments in a response the user 99% never
    sees. Here we fetch what someone actually opens — a small query per pick.

    Not via ``/api/transcript/<file>`` though it exists: that endpoint
    delivers the WHOLE transcript. For a recording many hours long
    that's megabytes for three paragraphs.

    Query: file (rel_path as the hit carries it), dir (the hit's
    ``file_dir``, the root that rel_path is relative to), start (seconds),
    n (segments per page, max 20). Without ``dir`` the open root is used.
    """
    rel = (request.args.get("file") or "").strip()
    root = (request.args.get("dir") or "").strip() or None
    if not rel or not db:
        return jsonify({"error": "No file specified"}), 400
    try:
        start = float(request.args.get("start") or 0)
    except ValueError:
        return jsonify({"error": "Invalid start time"}), 400
    try:
        n = max(1, min(int(request.args.get("n") or 3), 20))
    except ValueError:
        n = 3

    try:
        with db.connection() as conn:
            # By identity, like every other lookup (docs/PATHS.md). A bare
            # `WHERE rel_path = ?` took the newest of all files that share
            # the relative path — from any root.
            line = db.resolve_transcript(conn, rel, root, columns="id, speaker_names, abs_path")
            if not line:
                return jsonify({"error": "Transcript not found"}), 404
            tid = line["id"]
            try:
                names = json.loads(line["speaker_names"] or "{}")
            except (TypeError, ValueError):
                names = {}
            if not isinstance(names, dict):
                names = {}
            # The voice library's names for the labels the transcript leaves
            # unnamed — what the meaning search shows (rag/speakers.py)
            try:
                from whisper_tm.rag import speakers

                known = speakers.names_for([line["abs_path"]]).get(line["abs_path"]) or {}
                names = {**known, **names}
            except Exception as e:  # noqa: BLE001 — labels stay labels then
                logger.debug("search_context: voice names: %s", e)

            # Find the hit segment by start time. Rounded against floating
            # point noise — search and segment table read the same column,
            # but the value goes once through JSON.
            hits = conn.execute(
                "SELECT seg_index FROM segments WHERE transcript_id = ? "
                "ORDER BY ABS(start_sec - ?) LIMIT 1",
                (tid, start),
            ).fetchone()
            if not hits:
                return jsonify({"error": "No segments"}), 404
            middle = hits["seg_index"]

            rows = conn.execute(
                "SELECT seg_index, speaker, start_sec, end_sec, start_fmt, "
                "       end_fmt, text "
                "FROM segments WHERE transcript_id = ? "
                "  AND seg_index BETWEEN ? AND ? "
                "ORDER BY seg_index",
                (tid, middle - n, middle + n),
            ).fetchall()

        return jsonify(
            {
                "file": rel,
                "middle": middle,
                "segments": [
                    {
                        "index": r["seg_index"],
                        "speaker": names.get(r["speaker"]) or r["speaker"] or "—",
                        "start": r["start_sec"],
                        "end": r["end_sec"],
                        "start_fmt": r["start_fmt"] or "",
                        "end_fmt": r["end_fmt"] or "",
                        "text": r["text"] or "",
                        "is_hit": r["seg_index"] == middle,
                    }
                    for r in rows
                ],
            }
        )
    except Exception as e:  # noqa
        logger.warning("search_context: %s", e)
        return jsonify({"error": str(e)[:200]}), 500

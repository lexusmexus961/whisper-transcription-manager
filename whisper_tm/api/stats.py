"""
Statistics — one core, two scopes.

Previously there were three endpoints with three separate computation paths:
``/api/global_stats`` (text analysis), ``/api/dashboard_stats`` (global
metrics) and ``/api/directory_stats`` (directories, never called by the
frontend). On top of that the frontend computed the directory statistics a
fourth time itself, from the loaded file list — which would break anyway
once pagination is introduced, since then not all rows sit in the browser
any more.

Now: **one** aggregation, parametrized over a directory.

    GET /api/stats               → the entire active root
    GET /api/stats?path=Show/S1  → only this subtree
    GET /api/stats?text=1        → additionally the expensive word analysis
    GET /api/stats?scope=library → every row in the database, all roots

The library scope is the one exception to "only the active root": it is
asked for explicitly (the quick link in the stats view) and labelled as
such, so nobody mistakes it for the open folder. Counting still goes over
``abs_path`` — rows without an identity stay out, as everywhere else.

Both scopes return the same JSON shape. That is the real win: the UI only
needs one renderer, and "global" is simply the special case with no
subfolder.

**Counting happens over ``abs_path``, not over ``rel_path``.** ``docs/PATHS.md``
is unambiguous here: ``rel_path`` is a derived display value that may change
on a root switch, becomes stale for foreign rows after such a switch, and is
not unique across multiple roots — two roots each with a
``show/episode1.mp4`` collide. The database intentionally keeps old records
from foreign roots around; a statistic over ``rel_path`` would have silently
counted them in. This matches ``get_media_files_under``, the only access
already correctly restricted to the root.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from collections import Counter

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.core.stopwords import stopwords, words
from whisper_tm.extensions import db, hw

logger = logging.getLogger(__name__)

stats_bp = Blueprint("stats", __name__)

# How long a computed result is reused. Statistics are expensive (a large
# library's numbers take ten seconds and more). A finished or failed job
# clears the cache (app.py, clear_cache); files found by a rescan show
# within these minutes. So the Library opens at once when it was opened
# shortly before.
CACHE_TTL = 300.0

# Segment sample for the word analysis. The segment table has millions of
# rows for large libraries; a random sample delivers the same top lists at
# a fraction of the cost.
SAMPLE_SIZE = 4000

# Two levels: the state lock protects cache access (no torn reads), the
# compute lock enforces single-flight so that on parallel misses only one
# request computes. Never taken in reverse order → no deadlock.
_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
_compute_lock = threading.Lock()  # guards _compute_locks
# Single-flight per key, not one for all: the Library asks for the base
# numbers and the word analysis at once, and one lock for both made the
# base numbers wait for the word analysis (26 s instead of a few on a
# large library).
_compute_locks: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    with _compute_lock:
        lk = _compute_locks.get(key)
        if lk is None:
            if len(_compute_locks) > 4 * CACHE_MAX_ENTRIES:
                for k in [k for k, v in _compute_locks.items() if not v.locked()]:
                    del _compute_locks[k]
            lk = _compute_locks[key] = threading.Lock()
        return lk


#: At most this many results are kept. The key holds the requested folder,
#: and every folder used to stay in the cache for good — expired entries
#: were only skipped on read, never removed.
CACHE_MAX_ENTRIES = 64


def _cache_put(key: str, data: dict) -> None:
    """Store a result; drop expired ones, and the oldest beyond the cap."""
    now = time.time()
    with _cache_lock:
        for k in [k for k, (ts, _d) in _cache.items() if now - ts >= CACHE_TTL]:
            del _cache[k]
        _cache.pop(key, None)
        _cache[key] = (now, data)
        while len(_cache) > CACHE_MAX_ENTRIES:
            del _cache[next(iter(_cache))]


def clear_cache() -> None:
    """Call after a completed transcription."""
    try:
        from whisper_tm.api.library import clear_cache as _library_clear

        _library_clear()  # the Library's folder tree counts the same rows
    except Exception:  # noqa — the statistics' own cache is what matters here
        pass
    with _cache_lock:
        _cache.clear()


# ── Resolve scope ──────────────────────────────────────────────


def normalize_prefix(raw: str) -> str:
    """Turn user input into a clean relative subfolder.

    ``..`` components are stripped so the scope can't climb out of the
    root. The actual safeguard, though, is :func:`resolve` — that's
    where it's checked against the root.
    """
    path = (raw or "").strip().replace("\\", "/")
    parts = [t for t in path.split("/") if t and t not in (".", "..")]
    return "/".join(parts)


def resolve(prefix: str) -> tuple[str, str]:
    """(root, scope) as canonical absolute paths.

    ``scope`` is the root itself if no subfolder was requested.
    If the requested folder is not below the root, ``("", "")`` is
    returned — the caller then responds with an empty statistic instead
    of numbers from a foreign directory.
    """
    root = paths.canon(config.media_dir)
    if not root:
        return "", ""
    if not prefix:
        return root, root
    scope = paths.abs_from(prefix, root)
    if not scope or not paths.is_inside(scope, root):
        return root, ""
    return root, scope


def _filter(scope: str, column_name: str = "abs_path") -> tuple[str, list]:
    """WHERE fragment + parameters for everything below ``scope``.

    ``paths.like_prefix`` escapes ``%`` and ``_`` in the folder name and
    appends the separator — without it, root ``/mnt/show`` would also
    pick up ``/mnt/show2``.

    An empty ``scope`` is the library scope: every row with an identity.
    Only :func:`compute` in library mode gets here with it — the path
    scope bails out on an empty ``scope`` before filtering.
    """
    if not scope:
        return f"({column_name} IS NOT NULL AND {column_name} <> '')", []
    return f"{column_name} LIKE ? ESCAPE '\\'", [paths.like_prefix(scope) + "%"]


# ── Aggregation ───────────────────────────────────────────────────────────


def _number(row, key, default=0):
    """SQL aggregates return NULL for an empty set — here that becomes 0."""
    if row is None:
        return default
    # .keys() is intentional here, not a SIM118 case: `key in row` would
    # search the VALUES on a sqlite3.Row, not the column names.
    value = row[key] if key in row.keys() else None  # noqa: SIM118
    return default if value is None else value


def _empty(prefix: str, reason: str = "", library: bool = False) -> dict:
    """Response skeleton with no data — same shape so the frontend doesn't have to guess."""
    return {
        "scope": {
            "path": prefix,
            "label": "Whole library" if library else (prefix or "Whole open folder"),
            "global": not prefix,
            "library": library,
            "hint": reason,
        },
        "progress": {
            "total": 0,
            "done": 0,
            "pending": 0,
            "skipped": 0,
            "errors": 0,
            "without_audio": 0,
            "pct": 0.0,
        },
        "media": {"videos": 0, "audio_only": 0, "indexed": 0, "size": 0, "duration": 0},
        "transcripts": {
            "total": 0,
            "segments": 0,
            "words": 0,
            "chars": 0,
            "duration": 0,
            "compute_time": 0,
            "speed": 0.0,
            "wpm": 0,
            "avg_duration": 0,
            "avg_compute_time": 0,
            "avg_words": 0,
            "avg_segments": 0,
            "max_duration": 0,
        },
        "features": {"vocal_iso": 0, "diarized": 0, "with_warning": 0, "avg_speaker": 0},
        "duration_buckets": _duration_buckets(None, "", []),
        "models": [],
        "languages": [],
        "extensions": [],
        "activity": [],
        "history": [],
        "subfolders": [],
        "records": [],
        "system": {},
    }


def compute(
    prefix: str = "", with_text: bool = False, library: bool = False, folder: str = ""
) -> dict:
    """All metrics for one scope in a single connection.

    ``prefix`` empty = the entire active root. Otherwise a subfolder
    relative to it. ``library`` ignores both and covers every root in the
    database; there is no subfolder table then, because drilling down is
    defined relative to the open root.
    """
    if library:
        prefix = ""
    if not db:
        r = _empty(prefix, "No database configured", library)
        r["error"] = "No database configured"
        return r

    if folder:
        # Any folder of the library (the Library page's tree): it is the
        # root of its own statistics, its subfolders the drill-down
        root = scope = paths.canon(folder)
        prefix = ""
        if not root:
            return _empty("", "Folder not resolvable")
    elif library:
        root, scope = paths.canon(config.media_dir), ""
    else:
        root, scope = resolve(prefix)
        if not root:
            return _empty(prefix, "No media directory set")
        if not scope:
            return _empty(prefix, "Directory lies outside the root")

    result: dict = {
        "scope": {
            "path": prefix,
            "label": ("Whole library" if library else prefix or (root.rsplit("/", 1)[-1] or root)),
            "global": not prefix,
            "library": library,
            "root": root,
            "hint": "",
        },
    }
    where, args = _filter(scope)

    conn = None
    try:
        # Own read-only connection: the aggregation may run for seconds and
        # should not get in the way of the write path (running transcription).
        conn = sqlite3.connect(
            f"file:{db._db_path}?mode=ro",
            uri=True,
            timeout=5,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=3000")

        # ── Media files: progress and scope ─────────────────────────
        # COUNT(DISTINCT abs_path): after a root switch, two rows can
        # point to the same file (see get_media_files_under).
        row = conn.execute(
            f"""
            SELECT COUNT(DISTINCT abs_path) AS total,
                SUM(CASE WHEN status='done'    THEN 1 ELSE 0 END) AS done,
                SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skipped,
                SUM(CASE WHEN status='error'   THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN skip_reason='no_audio' THEN 1 ELSE 0 END) AS without_audio,
                SUM(CASE WHEN is_video=1 THEN 1 ELSE 0 END) AS videos,
                SUM(CASE WHEN is_video=0 THEN 1 ELSE 0 END) AS audio_only,
                SUM(CASE WHEN indexed_at > 0 THEN 1 ELSE 0 END) AS indexed,
                SUM(file_size)    AS size,
                SUM(duration_sec) AS media_duration
            FROM media_files WHERE {where}
        """,
            args,
        ).fetchone()
        total = _number(row, "total")
        done = _number(row, "done")
        result["progress"] = {
            "total": total,
            "done": done,
            "pending": _number(row, "pending"),
            "skipped": _number(row, "skipped"),
            "errors": _number(row, "errors"),
            "without_audio": _number(row, "without_audio"),
            "pct": round(done * 100.0 / total, 1) if total else 0.0,
        }
        result["media"] = {
            "videos": _number(row, "videos"),
            "audio_only": _number(row, "audio_only"),
            "indexed": _number(row, "indexed"),
            "size": _number(row, "size"),
            "duration": _number(row, "media_duration"),
        }

        # ── Transcripts: text volume and compute time ─────────────────────────
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS total,
                SUM(segment_count)   AS segments,
                SUM(word_count)      AS words,
                SUM(char_count)      AS chars,
                SUM(duration_sec)    AS duration,
                SUM(processing_secs) AS compute_time,
                AVG(CASE WHEN duration_sec    > 0 THEN duration_sec    END) AS avg_duration,
                AVG(CASE WHEN processing_secs > 0 THEN processing_secs END) AS avg_compute_time,
                AVG(CASE WHEN word_count      > 0 THEN word_count      END) AS avg_words,
                AVG(CASE WHEN segment_count   > 0 THEN segment_count   END) AS avg_segments,
                MAX(duration_sec) AS max_duration,
                SUM(CASE WHEN COALESCE(word_count, 0) = 0 THEN 1 ELSE 0 END) AS empty
            FROM transcripts WHERE complete = 1 AND {where}
        """,
            args,
        ).fetchone()
        duration = _number(row, "duration")
        compute_time = _number(row, "compute_time")
        words_total = _number(row, "words")
        result["transcripts"] = {
            "total": _number(row, "total"),
            "segments": _number(row, "segments"),
            "words": words_total,
            "chars": _number(row, "chars"),
            "duration": duration,
            "compute_time": compute_time,
            # Real-time factor: how many seconds of audio per second of compute time.
            "speed": round(duration / compute_time, 2) if compute_time > 0 else 0.0,
            # Speaking rate over the whole scope, not the average of the
            # individual values — short files should not distort the picture.
            "wpm": round(words_total / (duration / 60)) if duration > 60 else 0,
            "avg_duration": round(_number(row, "avg_duration", 0.0), 1),
            "avg_compute_time": round(_number(row, "avg_compute_time", 0.0), 1),
            "avg_words": round(_number(row, "avg_words", 0.0)),
            "avg_segments": round(_number(row, "avg_segments", 0.0)),
            "max_duration": _number(row, "max_duration"),
            # Transcripts without a single word (silence, music, a tone)
            "empty": _number(row, "empty"),
        }

        # ── Pipeline features ─────────────────────────────────────────
        # Diarized: what the transcription recorded, or a voices-only run
        # since (it diarizes too). Counting two or more speakers instead
        # told a library diarized at 88 % that it was 12 %; only rows
        # without recorded settings still go by the speakers.
        js = "CASE WHEN json_valid(settings_json) THEN settings_json END"
        row = conn.execute(
            f"""
            SELECT
                SUM(CASE WHEN vocal_isolation=1 THEN 1 ELSE 0 END) AS vocal_iso,
                SUM(CASE WHEN json_extract({js}, '$.diarized') = 1
                          OR json_extract({js}, '$.voices') IS NOT NULL
                          OR (json_extract({js}, '$.diarized') IS NULL
                              AND speaker_count > 1)
                         THEN 1 ELSE 0 END) AS diarized,
                SUM(CASE WHEN warning_count > 0 THEN 1 ELSE 0 END) AS with_warning,
                AVG(CASE WHEN speaker_count > 0 THEN CAST(speaker_count AS REAL) END) AS avg_speaker
            FROM transcripts WHERE complete=1 AND {where}
        """,
            args,
        ).fetchone()
        result["features"] = {
            "vocal_iso": _number(row, "vocal_iso"),
            "diarized": _number(row, "diarized"),
            "with_warning": _number(row, "with_warning"),
            "avg_speaker": round(_number(row, "avg_speaker", 0.0), 1),
        }

        # ── Distributions ──────────────────────────────────────────────────
        result["duration_buckets"] = _duration_buckets(conn, where, args)
        result["models"] = _grouped(conn, "model", where, args)
        result["languages"] = _grouped(conn, "language", where, args)
        result["extensions"] = _extensions(conn, where, args)
        result["activity"] = _activity(conn, where, args)
        result["history"] = _history(conn, where, args)
        result["subfolders"] = (
            [] if library else _subfolder(conn, root, scope, prefix, _indexed_map())
        )
        # Library scope: records may sit under any root, so they are shown
        # with their absolute path instead of one relative to the open root.
        result["records"] = _records(conn, where, args, "" if library else root)

        if with_text:
            result["text"] = _text_analysis(conn, scope)

        result["system"] = _system(conn)
        result["vector"] = _vector_coverage(conn, where, args)
    except Exception as e:
        logger.warning("Statistics failed: %s", e)
        result["error"] = str(e)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    return result


def _duration_buckets(conn, where, args) -> list[dict]:
    """Length distribution in fixed tiers.

    The tiers are always returned in full, even when one is empty —
    otherwise the bar chart jumps around differently at every data state.
    Boundaries are inclusive at the low end: 300 s counts as "5–30 min".
    """
    levels = [
        ("under_5min", "< 5 min"),
        ("5_30min", "5–30 min"),
        ("30min_2h", "30 min – 2 h"),
        ("over_2h", "> 2 h"),
    ]
    found: dict = {}
    if conn is not None:
        rows = conn.execute(
            f"""
            SELECT CASE
                     WHEN duration_sec <  300  THEN 'under_5min'
                     WHEN duration_sec <  1800 THEN '5_30min'
                     WHEN duration_sec <  7200 THEN '30min_2h'
                     ELSE 'over_2h' END AS bucket,
                   COUNT(*) AS cnt, SUM(duration_sec) AS duration
            FROM transcripts WHERE complete=1 AND duration_sec > 0 AND {where}
            GROUP BY bucket
        """,
            args,
        ).fetchall()
        found = {r["bucket"]: r for r in rows}
    return [
        {
            "key": key,
            "label": label,
            "cnt": _number(found.get(key), "cnt"),
            "duration": _number(found.get(key), "duration"),
        }
        for key, label in levels
    ]


def _grouped(conn, column_name: str, where, args) -> list[dict]:
    """Frequency distribution of a transcript column (model, language)."""
    rows = conn.execute(
        f"""
        SELECT {column_name} AS name, COUNT(*) AS cnt,
               SUM(duration_sec) AS duration, SUM(processing_secs) AS compute_time
        FROM transcripts
        WHERE complete=1 AND {column_name} != '' AND {column_name} IS NOT NULL AND {where}
        GROUP BY name ORDER BY cnt DESC LIMIT 12
    """,
        args,
    ).fetchall()
    return [
        {
            "name": r["name"],
            "cnt": r["cnt"],
            "duration": r["duration"] or 0,
            "speed": round((r["duration"] or 0) / r["compute_time"], 2)
            if (r["compute_time"] or 0) > 0
            else 0.0,
        }
        for r in rows
    ]


def _extensions(conn, where, args) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT extension AS name, COUNT(DISTINCT abs_path) AS cnt,
               SUM(file_size) AS size,
               SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done
        FROM media_files WHERE extension != '' AND {where}
        GROUP BY name ORDER BY cnt DESC LIMIT 12
    """,
        args,
    ).fetchall()
    return [dict(r) for r in rows]


def _activity(conn, where, args) -> list[dict]:
    """Completed transcripts per day, last 30 days — gap-free.

    SQL doesn't return days without activity; they're filled in here so
    the timeline has even spacing instead of bunched-up bars.
    """
    rows = conn.execute(
        f"""
        SELECT DATE(created_at, 'unixepoch') AS tag, COUNT(*) AS cnt,
               SUM(duration_sec) AS duration
        FROM transcripts
        WHERE complete=1 AND created_at > (unixepoch() - 30*86400) AND {where}
        GROUP BY tag
    """,
        args,
    ).fetchall()
    present = {r["tag"]: r for r in rows}
    days = conn.execute("""
        WITH RECURSIVE seq(value) AS (
            SELECT 0 UNION ALL SELECT value+1 FROM seq WHERE value < 29)
        SELECT DATE(unixepoch() - (29 - value) * 86400, 'unixepoch') AS tag FROM seq
    """).fetchall()
    return [
        {
            "tag": t["tag"],
            "cnt": _number(present.get(t["tag"]), "cnt"),
            "duration": _number(present.get(t["tag"]), "duration"),
        }
        for t in days
    ]


def _history(conn, where, args) -> list[dict]:
    """Completed transcripts per day over the whole history — only days
    with activity (the client fills the calendar; a gap-free list over
    years would be mostly zeros).

    Local dates, like the calendar the user reads: a transcript finished
    at 00:30 belongs to that day, not to the previous one in UTC.
    """
    rows = conn.execute(
        f"""
        SELECT DATE(created_at, 'unixepoch', 'localtime') AS tag, COUNT(*) AS cnt,
               SUM(duration_sec) AS duration, SUM(word_count) AS words
        FROM transcripts
        WHERE complete=1 AND created_at > 0 AND {where}
        GROUP BY tag ORDER BY tag
    """,
        args,
    ).fetchall()
    return [
        {
            "tag": r["tag"],
            "cnt": _number(r, "cnt"),
            "duration": _number(r, "duration"),
            "words": _number(r, "words"),
        }
        for r in rows
    ]


def _subfolder(conn, root: str, scope: str, prefix: str, indexed: dict | None = None) -> list[dict]:
    """Breakdown by direct subfolders of the scope.

    The folder name is cut from ``abs_path`` (not from ``rel_path``, which
    goes stale after a root switch). Files that sit directly in the scope
    are grouped under ``.``.

    Progress comes from ``media_files``, text volume from ``transcripts``.
    This matters more than it sounds: if progress were counted over the
    transcript rows, it would always come out to 100 % — a transcript only
    exists once the file is finished. That column would then measure
    itself and contradict the percentage in the header.
    """
    where, args = _filter(scope)
    offset = len(scope) + 2  # SUBSTR is 1-based, +1 for the separator
    cut = f"""CASE WHEN SUBSTR(abs_path, {offset}) LIKE '%/%'
                       THEN SUBSTR(SUBSTR(abs_path, {offset}), 1,
                                   INSTR(SUBSTR(abs_path, {offset}), '/') - 1)
                       ELSE '.' END"""

    media_items = conn.execute(
        f"""
        SELECT {cut} AS name,
               COUNT(DISTINCT abs_path) AS total,
               SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
               SUM(file_size) AS size
        FROM media_files WHERE {where}
        GROUP BY name
    """,
        args,
    ).fetchall()

    texts = conn.execute(
        f"""
        SELECT {cut} AS name,
               SUM(segment_count)   AS segments,
               SUM(word_count)      AS words,
               SUM(duration_sec)    AS duration,
               SUM(processing_secs) AS compute_time
        FROM transcripts WHERE complete=1 AND {where}
        GROUP BY name
    """,
        args,
    ).fetchall()
    by_name = {r["name"]: r for r in texts}

    # Embedding per folder.
    #
    # The vector store lives in a separate file — there's no GROUP BY across
    # both. So the transcripts are fetched once with their folder and
    # matched against the key set. A second pass over the same rows, but
    # without a second query over the large table.
    vec_in: dict[str, int] = {}
    vec_total: dict[str, int] = {}
    if indexed:
        from whisper_tm.core import paths as _paths

        for r in conn.execute(
            f"""
                SELECT {cut} AS name, abs_path, rel_path
                FROM transcripts WHERE complete=1 AND {where}""",
            args,
        ):
            n = r["name"]
            vec_total[n] = vec_total.get(n, 0) + 1
            ab = (r["abs_path"] or "").strip()
            if (_paths.canon(ab) or ab or r["rel_path"]) in indexed:
                vec_in[n] = vec_in.get(n, 0) + 1

    out = []
    for r in sorted(media_items, key=lambda x: -(x["total"] or 0))[:50]:
        name = r["name"]
        t = by_name.get(name)
        # "." = files directly in the scope; no further step leads there.
        path = prefix
        if name != ".":
            path = f"{prefix}/{name}" if prefix else name
        out.append(
            {
                "name": name,
                "total": r["total"] or 0,
                "done": r["done"] or 0,
                "size": r["size"] or 0,
                "segments": _number(t, "segments"),
                "words": _number(t, "words"),
                "duration": _number(t, "duration"),
                "compute_time": _number(t, "compute_time"),
                "path": path,
                # None means "no data available" (RAG off) and is shown in the
                # table as an empty cell — not as 0 %, which would be a lie.
                "vector_indexed": vec_in.get(name, 0) if indexed is not None else None,
                "vector_total": vec_total.get(name, 0) if indexed is not None else None,
            }
        )
    return out


# Record definitions: key, title, column, direction, condition.
_RECORDS = [
    ("longest", "Longest recording", "duration_sec", "DESC", "duration_sec > 0"),
    ("shortest", "Shortest recording", "duration_sec", "ASC", "duration_sec > 0"),
    ("most_words", "Most words", "word_count", "DESC", "word_count > 0"),
    ("most_segments", "Most segments", "segment_count", "DESC", "segment_count > 0"),
    ("most_speakers", "Most speakers", "speaker_count", "DESC", "speaker_count > 1"),
    ("longest_compute", "Longest compute time", "processing_secs", "DESC", "processing_secs > 0"),
]


def _display_path(abs_path: str, rel_path: str, root: str) -> str:
    """Path for display — derived from the identity, not stored.

    Without a root (library scope) the absolute path is the only
    unambiguous name.
    """
    if not root:
        return abs_path or rel_path
    return paths.rel_from(abs_path, root) or rel_path or abs_path


def _records(conn, where, args, root: str) -> list[dict]:
    """The one standout file per category."""
    out = []
    for key, title, column_name, direction, condition in _RECORDS:
        row = conn.execute(
            f"""
            SELECT abs_path, rel_path, {column_name} AS value
            FROM transcripts
            WHERE complete=1 AND {condition} AND {where}
            ORDER BY {column_name} {direction} LIMIT 1
        """,
            args,
        ).fetchone()
        if row:
            out.append(
                {
                    "key": key,
                    "title": title,
                    "value": row["value"],
                    "column_name": column_name,
                    "path": _display_path(row["abs_path"], row["rel_path"], root),
                }
            )

    # Fastest speaking rate: computed, therefore outside the table above.
    row = conn.execute(
        f"""
        SELECT abs_path, rel_path,
               CAST(word_count * 60.0 / duration_sec AS INTEGER) AS value
        FROM transcripts
        WHERE complete=1 AND duration_sec > 30 AND word_count > 0 AND {where}
        ORDER BY value DESC LIMIT 1
    """,
        args,
    ).fetchone()
    if row:
        out.append(
            {
                "key": "fastest",
                "title": "Highest speaking rate",
                "value": row["value"],
                "column_name": "wpm",
                "path": _display_path(row["abs_path"], row["rel_path"], root),
            }
        )
    return out


def _text_analysis(conn, scope: str) -> dict:
    """Top words, bigrams and speakers from a segment sample.

    The most expensive part of the statistics, hence only on request
    (``?text=1``). The numbers refer to the sample, not the total —
    ``sample`` states how many segments were actually read, so the UI can
    label it honestly.
    """
    stop = stopwords(config.get("search", "stopword_languages", default=["en"]))
    where, args = _filter(scope, "t.abs_path")
    rows = conn.execute(
        f"""
        SELECT s.text, s.speaker
        FROM segments s JOIN transcripts t ON t.id = s.transcript_id
        WHERE t.complete = 1 AND {where}
        ORDER BY RANDOM() LIMIT ?
    """,
        [*args, SAMPLE_SIZE],
    ).fetchall()

    word_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    speaker_counter: Counter = Counter()
    for r in rows:
        speaker = r["speaker"] or "—"
        if speaker != "—":
            speaker_counter[speaker] += 1
        ws = words(r["text"] or "", stop)
        word_counter.update(ws)
        for i in range(len(ws) - 1):
            # "the the": a repeated word is a stutter or a recognition
            # loop, not a phrase.
            if ws[i] != ws[i + 1]:
                bigram_counter[f"{ws[i]} {ws[i + 1]}"] += 1

    return {
        "sample": len(rows),
        "unique": len(word_counter),
        "top_words": [{"word": w, "cnt": c} for w, c in word_counter.most_common(40)],
        "top_bigrams": [{"phrase": p, "cnt": c} for p, c in bigram_counter.most_common(20)],
        "top_speaker": [{"name": s, "cnt": c} for s, c in speaker_counter.most_common(12)],
    }


def _system(conn) -> dict:
    """Hardware and database size — context for the numbers above."""
    info: dict = {"db_bytes": 0}
    with contextlib.suppress(Exception):
        info["db_bytes"] = db.get_db_size() if db else 0
    if hw:
        info.update(
            {
                "os": hw.os_name,
                "cpu": hw.cpu_name,
                "device": hw.best_device,
                "ram_mb": hw.ram_total_mb,
                "gpus": [g.name for g in hw.gpus],
            }
        )
    return info


# ── Endpoint ──────────────────────────────────────────────────────────────


@stats_bp.route("/api/stats")
def stats():
    """Statistics for the active root or a subfolder within it.

    Query parameters:
        path   Subfolder relative to the root. Empty/missing = whole root.
        text   ``1`` also runs the word analysis (expensive).
        scope  ``library`` covers every root in the database; ``path`` is
               ignored then.
        folder An absolute folder inside the allowed trees (the Library's
               tree): its statistics, whatever folder the list has open.
    """
    if not db:
        return jsonify({"error": "No database configured"}), 501

    folder = (request.args.get("folder") or "").strip()
    if folder:
        from whisper_tm.core.security import root_allowed

        if not root_allowed(folder):
            return jsonify({"error": "Folder outside the library"}), 403
        folder = paths.canon(folder)
    library = request.args.get("scope", "") == "library" and not folder
    prefix = "" if library else normalize_prefix(request.args.get("path", ""))
    with_text = request.args.get("text", "") in ("1", "true", "yes")
    # The root belongs in the key: after a directory switch, the statistics
    # of the previous root must not fall out of the cache.
    return jsonify(cached(prefix, with_text, library, folder))


def cached(
    prefix: str = "", with_text: bool = False, library: bool = False, folder: str = ""
) -> dict:
    """The statistics of one scope, from the cache or computed (once, however
    many ask at the same time)."""
    # The root belongs in the key: after a directory switch, the statistics
    # of the previous root must not fall out of the cache.
    key = f"{paths.canon(config.media_dir)}|{prefix}|{int(with_text)}|{int(library)}|{folder}"

    now = time.time()
    with _cache_lock:
        hits = _cache.get(key)
        if hits and (now - hits[0]) < CACHE_TTL:
            return hits[1]

    # Single-flight: on parallel misses of the same key only one computes,
    # the others then take the fresh result from the cache.
    with _key_lock(key):
        with _cache_lock:
            hits = _cache.get(key)
            if hits and (time.time() - hits[0]) < CACHE_TTL:
                return hits[1]

        data = compute(prefix, with_text, library, folder)

        # Errors deliberately NOT cached: otherwise a one-off database lock
        # timeout would sit around as an error message for 30 seconds.
        if "error" not in data:
            _cache_put(key, data)

    return data


def warm(folder: str) -> None:
    """Count the Library's first view ahead (the server's start, a finished
    job): the open folder's numbers, its word analysis and who speaks the
    most — all at once, so the Library opens without the wait."""
    folder = paths.canon(folder)
    t0 = time.time()

    def run(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a head start, never a failure
            logger.debug("Library warm-up: %s", e)

    from whisper_tm.api.library import warm_speakers

    threads = [
        threading.Thread(target=run, args=(f,), daemon=True, name="library-warm")
        for f in (
            lambda: cached(folder=folder),
            lambda: cached(with_text=True, folder=folder),
            lambda: warm_speakers(folder),
        )
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.info("Library counted ahead in %.1f s", time.time() - t0)


def _indexed_map() -> dict | None:
    """The keys in the vector store — once per statistics run.

    ``None`` means "no data available" (RAG off or store unreadable) and
    is explicitly different from an empty dict: the latter would mean
    "nothing embedded", and that's a claim you can't make without a store.
    """
    try:
        from whisper_tm.rag.service import get_rag_config, make_rag_db

        if not get_rag_config().get("enabled"):
            return None
        return make_rag_db().get_indexed_map()
    except Exception as e:  # noqa
        logger.debug("Embedding map not readable: %s", e)
        return None


def _vector_coverage(conn, where: str, args: list) -> dict:
    """How much of the scope is in the vector store?

    **Why this lives here and not in the RAG API.** Coverage is a metric
    over the collection like any other in this file, and it should have
    the same scope: whoever is looking at a subfolder wants to know how
    much of THIS folder is searchable — not how much of the whole library.

    The vector store lives in its own file; an SQL join across both would
    only work with ATTACH and would block the write side. So two lists and
    one pass: the transcripts in scope (with their identity) against the
    keys in the store.

    If the store is missing or RAG is off, a dict with ``active: False``
    comes back and no exception — the statistics must not fail because of
    that.
    """
    off = {
        "active": False,
        "embedded": 0,
        "open_count": 0,
        "stale": 0,
        "chunks": 0,
        "total": 0,
        "pct": 0.0,
    }
    try:
        from whisper_tm.rag.service import get_rag_config, make_rag_db

        # `rag.enabled` is "Show coverage", not a switch for the search:
        # the page says which of the two it is
        if not get_rag_config().get("enabled"):
            return {**off, "hidden": True}
        indexed = make_rag_db().get_indexed_map()
    except Exception as e:  # noqa
        logger.debug("Vector coverage not determinable: %s", e)
        return off
    if not indexed:
        off["active"] = True
        # Even with no entries, "total" (total) still matters: the display
        # should be able to show "0 of 4,000" and not just a zero.
        try:
            off["total"] = _number(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM transcripts WHERE complete=1 AND {where}", args
                ).fetchone(),
                "n",
            )
        except Exception:  # noqa
            pass
        return off

    from whisper_tm.core import paths as _paths
    from whisper_tm.rag.indexer import _close as _near

    try:
        rows = conn.execute(
            f"SELECT abs_path, rel_path, segment_count AS seg, updated_at "
            f"FROM transcripts WHERE complete=1 AND {where}",
            args,
        ).fetchall()
    except Exception as e:  # noqa
        logger.debug("Vector coverage: %s", e)
        return off

    on = open_count = stale = chunks = 0
    for r in rows:
        ab = (r["abs_path"] or "").strip()
        key = _paths.canon(ab) or ab or r["rel_path"]
        before = indexed.get(key)
        if not before:
            open_count += 1
            continue
        # Same rule as in the indexer: unchanged = done. Otherwise the
        # statistics would show "done" where the next run re-embeds.
        if before.get("seg_count") == (r["seg"] or 0) and _near(
            before.get("updated_at"), r["updated_at"] or 0.0
        ):
            on += 1
            chunks += int(before.get("chunk_count") or 0)
        else:
            stale += 1
            open_count += 1
    total = len(rows)
    off.update(
        {
            "active": True,
            "embedded": on,
            "open_count": open_count,
            "stale": stale,
            "chunks": chunks,
            "total": total,
            "pct": round(on * 100.0 / total, 1) if total else 0.0,
        }
    )
    return off

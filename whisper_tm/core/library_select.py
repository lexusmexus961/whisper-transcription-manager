"""
Library selection — pick files from the whole library by what is known about
them, and put them into the queue.

The queue editor manages what is already queued; this answers the question
before: *which* files. "Everything transcribed without diarization before
the 1st, under Archive/, longer than a minute, without voices" is one
selection here, counted live and queued as one job.

Criteria are building blocks combined with AND; every block is optional.
They run as one SQL query over ``media_files`` joined with ``transcripts``
(the recorded settings from schema v25 included), plus three sets that live
elsewhere and join as temporary tables: files with voices (``speaker.db``),
files already waiting in the queue, and files whose title exists in another
format that is already transcribed.

Unknown criteria are an error, not ignored: a typo in the interface would
otherwise widen the selection silently — and the next step queues it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from whisper_tm.core import paths

__all__ = ["CriteriaError", "select", "abs_paths", "facets", "CRITERIA"]


class CriteriaError(ValueError):
    """A criterion is unknown or has a value of the wrong kind."""


#: name → kind. The kinds decide validation and which SQL is built.
CRITERIA = {
    "folders": "paths",
    "exclude_folders": "paths",
    "name": "glob",
    "exclude_name": "glob",
    "extensions": "strings",
    "status": "strings",  # open | done | skipped | no_audio | error
    "duration_min": "number",
    "duration_max": "number",  # seconds
    "size_min": "number",
    "size_max": "number",  # bytes
    "video": "bool",
    "transcribed_from": "date",
    "transcribed_to": "date",
    "models": "strings",
    "languages": "strings",
    "vocal_models": "strings",
    "app_commits": "strings",
    "diarized": "bool",
    "voices": "bool",
    "voices_old": "bool",  # voices without stored speech spans
    "voices_run": "bool",  # a voices-only run is recorded (any result)
    "speech_for_voices": "bool",  # at least one segment of >= 4 s
    "segments_min": "number",
    "segments_max": "number",
    "words_min": "number",
    "words_max": "number",
    "speakers_min": "number",
    "speakers_max": "number",
    "summary": "bool",
    "translation": "bool",
    "corrections": "bool",
    "in_queue": "bool",
    "failed_min": "number",  # failed queue runs for this file
    "other_format_transcribed": "bool",
}

_STATUS = {
    "open": "((t.id IS NULL OR t.complete = 0) AND COALESCE(m.status, '') "
    "NOT IN ('skipped') AND COALESCE(m.has_audio, 1) != 0 "
    "AND sk.abs_path IS NULL)",
    "done": "(t.complete = 1)",
    "skipped": "(sk.abs_path IS NOT NULL OR m.status = 'skipped')",
    "no_audio": "(m.has_audio = 0)",
    "error": "(m.status = 'error')",
}

_SPEECH_FOR_VOICES_SEC = 4.0  # core/speaker.MIN_SEGMENT


def _validate(criteria: dict) -> dict:
    if not isinstance(criteria, dict):
        raise CriteriaError("criteria must be an object")
    out = {}
    for key, value in criteria.items():
        kind = CRITERIA.get(key)
        if kind is None:
            raise CriteriaError(f"unknown criterion: {key}")
        if value is None or value == "" or value == []:
            continue
        if kind == "bool":
            if not isinstance(value, bool):
                raise CriteriaError(f"{key}: expected true/false")
        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CriteriaError(f"{key}: expected a number")
        elif kind in ("strings", "paths"):
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise CriteriaError(f"{key}: expected a list of texts")
            if kind == "paths":
                value = [paths.canon(v) for v in value]
                if not all(value):
                    raise CriteriaError(f"{key}: folders must be absolute paths")
        elif kind in ("glob", "date"):
            if not isinstance(value, str):
                raise CriteriaError(f"{key}: expected text")
            if kind == "date":
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError as e:
                    raise CriteriaError(f"{key}: expected YYYY-MM-DD") from e
        out[key] = value
    bad = set(out.get("status", [])) - set(_STATUS)
    if bad:
        raise CriteriaError(f"status: unknown value(s) {sorted(bad)}")
    return out


def _glob_to_like(pattern: str) -> str:
    esc = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return "%" + esc.replace("*", "%").replace("?", "_") + "%"


def _epoch(day: str, end: bool = False) -> float:
    t = time.mktime(datetime.strptime(day, "%Y-%m-%d").timetuple())
    return t + 86400 if end else t


# ── Sets that live outside this query ────────────────────────────────────

_voiced_cache: dict = {}


def _voiced_paths() -> set[str]:
    """abs_paths with voice vectors — cached until speaker.db changes."""
    try:
        from whisper_tm.database.speaker import speaker_db

        sdb = speaker_db()
        stamp = tuple(
            os.path.getmtime(p) if os.path.exists(p) else 0
            for p in (sdb._db_path, sdb._db_path + "-wal")
        )
        if _voiced_cache.get("stamp") != stamp:
            sdb.initialize()
            with sdb.connection() as c:
                _voiced_cache["paths"] = {
                    r[0] for r in c.execute("SELECT DISTINCT abs_path FROM speaker_vectors")
                }
            _voiced_cache["stamp"] = stamp
        return _voiced_cache["paths"]
    except Exception:
        return set()


_old_cache: dict = {}


def _old_voice_paths() -> set[str]:
    """abs_paths with voices made before their speech spans were stored
    — the ones a renewal gives spans. A pass over every row,
    so only when the criterion is asked for, and kept until speaker.db
    changes."""
    try:
        from whisper_tm.database.speaker import speaker_db

        sdb = speaker_db()
        stamp = tuple(
            os.path.getmtime(p) if os.path.exists(p) else 0
            for p in (sdb._db_path, sdb._db_path + "-wal")
        )
        if _old_cache.get("stamp") != stamp:
            sdb.initialize()
            with sdb.connection() as c:
                _old_cache["paths"] = {
                    r[0]
                    for r in c.execute(
                        "SELECT DISTINCT abs_path FROM speaker_vectors "
                        "WHERE parts IS NULL OR parts IN ('', '[]', 'null')"
                    )
                }
            _old_cache["stamp"] = stamp
        return _old_cache["paths"]
    except Exception:
        return set()


_speech_cache: dict = {}


def _speech_ids(conn) -> set[int]:
    """Transcripts with at least one segment long enough for a voice.

    Asked per row it took seconds on a large library — too slow for a
    count that follows every click. One pass over the segments, kept until
    a transcript is added or changed.
    """
    stamp = conn.execute("SELECT COUNT(*), MAX(updated_at) FROM transcripts").fetchone()
    if _speech_cache.get("stamp") != tuple(stamp):
        _speech_cache["ids"] = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT transcript_id FROM segments "
                f"WHERE end_sec - start_sec >= {_SPEECH_FOR_VOICES_SEC}"
            )
        }
        _speech_cache["stamp"] = tuple(stamp)
    return _speech_cache["ids"]


def _queue_sets(conn) -> tuple[set[str], dict[str, int]]:
    queued, failed = set(), defaultdict(int)
    # base_dir '' is the media root (core/queue.enqueue) — skipped, such an
    # entry looked "not queued" and was offered, and queued, a second time
    try:
        from whisper_tm.config import config

        media_root = str(config.media_dir)
    except Exception:  # noqa: BLE001 — config not loaded (tools)
        media_root = ""
    for base, name, status in conn.execute(
        "SELECT base_dir, filename, status FROM queue_items "
        "WHERE status IN ('pending', 'processing', 'error')"
    ):
        base = base or media_root
        key = (
            paths.canon(name)
            if paths.is_absolute(name)
            else (paths.abs_from(name, base) if base else "")
        )
        if not key:
            continue
        if status == "error":
            failed[key] += 1
        else:
            queued.add(key)
    return queued, failed


def _other_format_transcribed(conn) -> set[str]:
    """Files whose title (same folder, same name without extension) is
    already transcribed in another format — .mkv next to a done .mp4."""
    done_stems = defaultdict(set)
    all_rows = conn.execute(
        "SELECT m.abs_path, t.complete FROM media_files m "
        "LEFT JOIN transcripts t ON t.abs_path = m.abs_path "
        "WHERE m.abs_path != ''"
    ).fetchall()
    for p, complete in all_rows:
        if complete:
            done_stems[os.path.splitext(p)[0]].add(p)
    return {p for p, _ in all_rows if done_stems.get(os.path.splitext(p)[0], set()) - {p}}


def _temp_table(conn, name: str, rows) -> None:
    conn.execute(f"DROP TABLE IF EXISTS temp.{name}")
    conn.execute(f"CREATE TEMP TABLE {name} (abs_path TEXT PRIMARY KEY, n INTEGER)")
    conn.executemany(f"INSERT OR IGNORE INTO temp.{name} VALUES (?, ?)", rows)


# ── Query ────────────────────────────────────────────────────────────────


def _build(conn, c: dict) -> tuple[str, list]:
    where, args = ["m.abs_path != ''"], []

    def num(expr, lo, hi):
        if lo in c:
            where.append(f"{expr} >= ?")
            args.append(c[lo])
        if hi in c:
            where.append(f"{expr} <= ?")
            args.append(c[hi])

    def flag(key, sql_true):
        if key in c:
            where.append(sql_true if c[key] else f"NOT {sql_true}")

    def one_of(key, expr):
        if key in c:
            where.append(f"{expr} IN ({','.join('?' * len(c[key]))})")
            args.extend(c[key])

    if "folders" in c:
        where.append("(" + " OR ".join("m.abs_path LIKE ? ESCAPE '\\'" for _ in c["folders"]) + ")")
        args.extend(paths.like_prefix(f) + "%" for f in c["folders"])
    for f in c.get("exclude_folders", []):
        where.append("m.abs_path NOT LIKE ? ESCAPE '\\'")
        args.append(paths.like_prefix(f) + "%")
    if "name" in c:
        where.append("m.abs_path LIKE ? ESCAPE '\\'")
        args.append(_glob_to_like(c["name"]))
    if "exclude_name" in c:
        where.append("m.abs_path NOT LIKE ? ESCAPE '\\'")
        args.append(_glob_to_like(c["exclude_name"]))
    if "extensions" in c:
        exts = [e.lower() if e.startswith(".") else "." + e.lower() for e in c["extensions"]]
        where.append(f"LOWER(m.extension) IN ({','.join('?' * len(exts))})")
        args.extend(exts)
    if "status" in c:
        where.append("(" + " OR ".join(_STATUS[s] for s in c["status"]) + ")")
    num("COALESCE(t.duration_sec, m.duration_sec, 0)", "duration_min", "duration_max")
    num("COALESCE(m.file_size, 0)", "size_min", "size_max")
    flag("video", "(COALESCE(m.is_video, 0) = 1)")
    if "transcribed_from" in c:
        where.append("t.created_at >= ?")
        args.append(_epoch(c["transcribed_from"]))
    if "transcribed_to" in c:
        where.append("t.created_at < ?")
        args.append(_epoch(c["transcribed_to"], end=True))
    one_of("models", "t.model")
    one_of("languages", "t.language")
    one_of("vocal_models", "t.vocal_model")
    one_of("app_commits", f"json_extract({_SETTINGS}, '$.app_commit')")
    flag("diarized", f"(COALESCE(json_extract({_SETTINGS}, '$.diarized'), 0) = 1)")
    # A voices-only run records itself under settings.voices even when it
    # found no speaker with enough speech — without this, "voices missing"
    # would offer those files again on every run.
    flag("voices_run", f"(json_extract({_SETTINGS}, '$.voices') IS NOT NULL)")
    num("COALESCE(t.segment_count, 0)", "segments_min", "segments_max")
    num("COALESCE(t.word_count, 0)", "words_min", "words_max")
    num("COALESCE(t.speaker_count, 0)", "speakers_min", "speakers_max")
    if "speech_for_voices" in c:
        conn.execute("DROP TABLE IF EXISTS temp.wx_speech")
        conn.execute("CREATE TEMP TABLE wx_speech (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO temp.wx_speech VALUES (?)", ((i,) for i in _speech_ids(conn)))
        flag("speech_for_voices", "(t.id IN (SELECT id FROM temp.wx_speech))")
    flag("summary", "EXISTS (SELECT 1 FROM summaries x WHERE x.abs_path = m.abs_path)")
    flag("translation", "EXISTS (SELECT 1 FROM translations x WHERE x.abs_path = m.abs_path)")
    flag("corrections", "EXISTS (SELECT 1 FROM llm_edits x WHERE x.abs_path = m.abs_path)")

    if "voices" in c:
        _temp_table(conn, "wx_voiced", ((p, 1) for p in _voiced_paths()))
        flag("voices", "(m.abs_path IN (SELECT abs_path FROM temp.wx_voiced))")
    if "voices_old" in c:
        _temp_table(conn, "wx_old_voices", ((p, 1) for p in _old_voice_paths()))
        flag("voices_old", "(m.abs_path IN (SELECT abs_path FROM temp.wx_old_voices))")
    if "in_queue" in c or "failed_min" in c:
        queued, failed = _queue_sets(conn)
        _temp_table(conn, "wx_queued", ((p, 1) for p in queued))
        _temp_table(conn, "wx_failed", failed.items())
        flag("in_queue", "(m.abs_path IN (SELECT abs_path FROM temp.wx_queued))")
        if "failed_min" in c:
            where.append(
                "COALESCE((SELECT n FROM temp.wx_failed f WHERE f.abs_path = m.abs_path), 0) >= ?"
            )
            args.append(c["failed_min"])
    if "other_format_transcribed" in c:
        _temp_table(conn, "wx_twins", ((p, 1) for p in _other_format_transcribed(conn)))
        flag("other_format_transcribed", "(m.abs_path IN (SELECT abs_path FROM temp.wx_twins))")
    return " AND ".join(where), args


# A transcript's recorded settings — NULL where they are not valid JSON.
# json_extract() throws on anything else, '' included, and one such row
# would fail the whole query: restored transcripts can carry '' and would
# take "From library"'s choice lists down with them.
_SETTINGS = "(CASE WHEN json_valid(t.settings_json) THEN t.settings_json END)"

_FROM = (
    "FROM media_files m "
    "LEFT JOIN transcripts t ON t.abs_path = m.abs_path AND t.abs_path != '' "
    "LEFT JOIN skiplist sk ON sk.abs_path = m.abs_path AND sk.abs_path != '' "
)


def select(conn: sqlite3.Connection, criteria: dict, limit: int = 50, offset: int = 0) -> dict:
    """Count, audio total and a page of the matching files."""
    c = _validate(criteria)
    where, args = _build(conn, c)
    n, secs = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM(COALESCE(t.duration_sec, m.duration_sec, 0)), 0) "
        f"{_FROM} WHERE {where}",
        args,
    ).fetchone()
    rows = conn.execute(
        f"SELECT m.abs_path, COALESCE(t.duration_sec, m.duration_sec, 0), "
        f"t.complete, t.created_at, t.model, t.settings_json {_FROM} WHERE {where} "
        f"ORDER BY m.abs_path LIMIT ? OFFSET ?",
        [*args, int(limit), int(offset)],
    ).fetchall()
    sample = []
    for p, dur, complete, created, model, raw in rows:
        try:
            s = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            s = {}
        sample.append(
            {
                "abs_path": p,
                "name": Path(p).name,
                "duration_sec": round(dur or 0, 1),
                "transcribed": bool(complete),
                "transcribed_at": created,
                "model": model or "",
                "diarized": s.get("diarized"),
            }
        )
    return {
        "count": n,
        "audio_sec": round(secs or 0),
        "criteria": c,
        "sample": sample,
        "offset": offset,
        "limit": limit,
    }


def abs_paths(conn: sqlite3.Connection, criteria: dict) -> list[str]:
    """Every matching file — what gets queued."""
    c = _validate(criteria)
    where, args = _build(conn, c)
    return [
        r[0]
        for r in conn.execute(f"SELECT m.abs_path {_FROM} WHERE {where} ORDER BY m.abs_path", args)
    ]


def facets(conn: sqlite3.Connection) -> dict:
    """The values the choice lists offer — what actually occurs, with counts."""

    def counted(sql):
        return [{"value": v, "count": n} for v, n in conn.execute(sql) if v not in (None, "")]

    return {
        "models": counted("SELECT model, COUNT(*) FROM transcripts GROUP BY 1 ORDER BY 2 DESC"),
        "languages": counted(
            "SELECT language, COUNT(*) FROM transcripts GROUP BY 1 ORDER BY 2 DESC"
        ),
        "vocal_models": counted(
            "SELECT vocal_model, COUNT(*) FROM transcripts GROUP BY 1 ORDER BY 2 DESC"
        ),
        "app_commits": counted(
            "SELECT json_extract(CASE WHEN json_valid(settings_json) "
            "THEN settings_json END, '$.app_commit'), COUNT(*) "
            "FROM transcripts GROUP BY 1 ORDER BY 2 DESC"
        ),
        "extensions": counted(
            "SELECT LOWER(extension), COUNT(*) FROM media_files GROUP BY 1 ORDER BY 2 DESC LIMIT 40"
        ),
    }

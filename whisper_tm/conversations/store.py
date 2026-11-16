"""The conversation statistics in the main database — written and read.

Three tables (created by SCHEMA_SQL in database/__init__.py), all hanging
off ``transcripts(id)`` with ``ON DELETE CASCADE``, so deleting a
transcript takes its numbers along without code of its own:

* ``conv_stats`` — one row per transcript: the summary of metrics.py, plus
  what it was counted from (``seg_count``, ``char_count``: the transcript's
  ``segment_count`` and ``char_count`` then) and by which rules
  (``version``);
* ``conv_speakers`` — one row per transcript and diarization label;
* ``conv_windows`` — the windows of a diarized recording longer than two
  windows, as one JSON text per transcript: only a query about people and
  a recording's "who talked when" read them, and a separate narrow table
  keeps them out of every scan.

**Kept current by every transcript write.** ``Database.upsert_transcript``
— the one place that writes segments: transcription, edits, speaker names,
corrections, imports — counts while it holds the segments and writes the
rows in its own transaction (:func:`write`), so the numbers are never older
than the transcript. A write that does not count (switched off, or the
counting failed) drops the old rows instead (:func:`drop`). The background
run (runner.py) counts what has no row yet, or a row of an older
``version``, or one whose ``seg_count`` or ``char_count`` no longer match.

``transcripts.updated_at`` is no help for that: it moves when only the
display path changes, and it sits behind the large ``json_data`` column,
so reading it for every transcript walks overflow pages.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime

from whisper_tm.conversations import metrics

logger = logging.getLogger(__name__)

#: The rules' version. Raise it when metrics.py counts differently: the
#: background run then counts every transcript again.
STATS_VERSION = 1
#: Windows are kept for diarized recordings longer than this
WINDOWS_FROM_SEC = 2 * metrics.WINDOW_SEC

_SUMMARY_COLS = (
    "start_sec",
    "end_sec",
    "duration_sec",
    "speech_sec",
    "attributed_sec",
    "unattributed_sec",
    "words",
    "sentences",
    "questions",
    "turns",
    "short_turns",
    "labels",
    "speakers",
    "top_share",
    "longest_turn_sec",
    "longest_turn_label",
    "longest_turn_start",
    "punctuated",
    "interview",
    "wps",
    "turn_rate",
    "mean_turn_sec",
)
_SPEAKER_COLS = (
    "talk_sec",
    "words",
    "sentences",
    "questions",
    "turns",
    "short_turns",
    "longest_turn_sec",
    "longest_turn_start",
    "first_sec",
    "last_sec",
)

_gen_lock = threading.Lock()
_generation = 0


def generation() -> int:
    """Moves with every write — query caches key on it."""
    return _generation


def changed() -> None:
    global _generation
    with _gen_lock:
        _generation += 1


def enabled() -> bool:
    try:
        from whisper_tm.config import config
        from whisper_tm.core.helpers import parse_bool

        return parse_bool(config.get("conversations", "enabled", default=True))
    except Exception:  # noqa: BLE001 — no configuration (tests, tools): counted
        return True


def name_date(abs_path: str) -> float | None:
    """The recording date in the file name (core/voice_extract.date_from_name)
    as the epoch of that day's local midnight — None without one. The file
    time is often a copy's date, years after the recording."""
    try:
        from whisper_tm.core.voice_extract import date_from_name

        day = date_from_name(abs_path or "")
        return datetime.strptime(day, "%Y-%m-%d").timestamp() if day else None
    except (ValueError, OverflowError, OSError):
        return None


def compute(data: dict) -> dict:
    """metrics.analyze for a transcript dict (as upsert_transcript gets it)."""
    return metrics.analyze(
        data.get("segments") or [], data.get("language"), data.get("total_duration_sec")
    )


def write(
    conn, transcript_id: int, stats: dict, *, seg_count: int, char_count: int, abs_path: str = ""
) -> None:
    """Replace a transcript's rows with ``stats`` (:func:`compute`), on the
    caller's connection and inside its transaction. A savepoint keeps the
    three tables in step: either all rows are replaced or none."""
    s = stats["summary"]
    windows = None
    if stats["windows"] and s["labels"] and s["duration_sec"] > WINDOWS_FROM_SEC:
        windows = json.dumps(
            {"w": metrics.WINDOW_SEC, "labels": stats["windows"]},
            separators=(",", ":"),
            ensure_ascii=False,
        )
    conn.execute("SAVEPOINT conv_write")
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO conv_stats (transcript_id, version, seg_count, "
            f"char_count, name_date, computed_at, {', '.join(_SUMMARY_COLS)}) "
            f"VALUES ({', '.join('?' * (6 + len(_SUMMARY_COLS)))})",
            (
                transcript_id,
                STATS_VERSION,
                seg_count,
                char_count,
                name_date(abs_path),
                time.time(),
                *[_sql(s[c]) for c in _SUMMARY_COLS],
            ),
        )
        conn.execute("DELETE FROM conv_speakers WHERE transcript_id = ?", (transcript_id,))
        if stats["speakers"]:
            conn.executemany(
                f"INSERT INTO conv_speakers (transcript_id, label, {', '.join(_SPEAKER_COLS)}) "
                f"VALUES ({', '.join('?' * (2 + len(_SPEAKER_COLS)))})",
                [
                    (transcript_id, label, *[_sql(d[c]) for c in _SPEAKER_COLS])
                    for label, d in stats["speakers"].items()
                ],
            )
        if windows is None:
            conn.execute("DELETE FROM conv_windows WHERE transcript_id = ?", (transcript_id,))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO conv_windows (transcript_id, window_sec, data) "
                "VALUES (?, ?, ?)",
                (transcript_id, metrics.WINDOW_SEC, windows),
            )
        conn.execute("RELEASE conv_write")
    except BaseException:
        # A full disk or an I/O error makes SQLite roll back the WHOLE
        # transaction, savepoint included: ROLLBACK TO would then fail with
        # "no such savepoint" and replace the error that matters. Undo the
        # savepoint only while it still exists, and always re-raise the
        # original error — the caller must see that its transaction is gone.
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK TO conv_write")
                conn.execute("RELEASE conv_write")
            except sqlite3.Error:
                pass
        raise


def drop(conn, transcript_id: int) -> int:
    """Remove a transcript's rows, inside the caller's transaction — for a
    write that was not counted. Kept, the old numbers could pass for current:
    merging two speaker labels changes neither ``seg_count`` nor
    ``char_count``. Without rows the background run counts it again."""
    n = conn.execute("DELETE FROM conv_stats WHERE transcript_id = ?", (transcript_id,)).rowcount
    n += conn.execute(
        "DELETE FROM conv_speakers WHERE transcript_id = ?", (transcript_id,)
    ).rowcount
    n += conn.execute("DELETE FROM conv_windows WHERE transcript_id = ?", (transcript_id,)).rowcount
    return n


def _sql(v):
    return int(v) if isinstance(v, bool) else v


# ── What is missing ────────────────────────────────────────────────────────

_STALE = """
    FROM transcripts t
    LEFT JOIN conv_stats s ON s.transcript_id = t.id
    WHERE t.complete = 1
      AND (s.transcript_id IS NULL OR s.version < ?
           OR s.seg_count != t.segment_count OR s.char_count != t.char_count)
"""


def stale_count(conn) -> int:
    """Transcripts whose numbers are missing or out of date."""
    return conn.execute("SELECT COUNT(*) " + _STALE, (STATS_VERSION,)).fetchone()[0]


def stale_ids(conn, after_id: int, limit: int) -> list[int]:
    """The next ``limit`` such transcripts after ``after_id``, by id."""
    rows = conn.execute(
        "SELECT t.id " + _STALE + " AND t.id > ? ORDER BY t.id LIMIT ?",
        (STATS_VERSION, after_id, limit),
    ).fetchall()
    return [r[0] for r in rows]


def counts(conn) -> dict:
    """``{"transcripts": n, "counted": n, "stale": n}`` — counted: rows of
    the present rules."""
    total = conn.execute("SELECT COUNT(*) FROM transcripts WHERE complete = 1").fetchone()[0]
    stale = stale_count(conn)
    return {"transcripts": total, "counted": total - stale, "stale": stale}


def recount_all(conn) -> int:
    """Mark every row out of date (the background run counts them again);
    the old numbers stay readable until then."""
    return conn.execute("UPDATE conv_stats SET version = 0").rowcount


def load_segments(conn, ids: list[int]) -> dict[int, dict]:
    """``{id: {"data": transcript-like dict, "seg_count", "char_count",
    "abs_path"}}`` for the background run — from the segments table and
    the columns before ``json_data``, never ``json_data`` itself."""
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    out: dict[int, dict] = {}
    for tid, lang, dur, segc, charc, abs_path in conn.execute(
        f"SELECT id, language, duration_sec, segment_count, char_count, abs_path "
        f"FROM transcripts WHERE id IN ({marks})",
        ids,
    ):
        out[tid] = {
            "data": {"language": lang, "total_duration_sec": dur, "segments": []},
            "seg_count": segc or 0,
            "char_count": charc or 0,
            "abs_path": abs_path or "",
        }
    for tid, a, b, sp, text, ph in conn.execute(
        f"SELECT transcript_id, start_sec, end_sec, speaker, text, phantom_score "
        f"FROM segments WHERE transcript_id IN ({marks}) "
        f"ORDER BY transcript_id, seg_index",
        ids,
    ):
        item = out.get(tid)
        if item is not None:
            item["data"]["segments"].append(
                {"start": a, "end": b, "speaker": sp, "text": text, "phantom_score": ph}
            )
    return out

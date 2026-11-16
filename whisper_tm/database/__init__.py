"""
Database engine: SQLite with FTS5, WAL mode, connection pooling.

Manages:
- Transcripts with segments (optionally alongside JSON files)
- FTS5 full-text index for ultra-fast search
- Job history and statistics
- File metadata cache (for fast scans)
- Settings and user preferences
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from whisper_tm.core import paths

logger = logging.getLogger(__name__)


# Columns of file list query: everything a list entry needs
# (scan base data, cached FFprobe metadata, transcript join) in ONE
# query. Shared by get_media_files_under (entire directory) and
# get_media_file_at (one file) — both MUST return the same row form,
# otherwise full scan and incremental update drift apart.
_MEDIA_SCAN_SELECT = """
            SELECT m.rel_path, m.abs_path, m.display_name, m.extension,
                   m.is_video, m.file_size, m.modified_at, m.status,
                   m.skip_reason, m.media_title, m.indexed_at,
                   m.duration_sec, m.has_audio, m.has_video, m.width, m.height,
                   m.audio_codec, m.video_codec, m.bitrate,
                   t.processing_time  AS tx_processing_time,
                   t.total_duration   AS tx_total_duration,
                   t.duration_sec     AS tx_duration_sec,
                   t.processing_secs  AS tx_processing_secs,
                   t.model            AS tx_model,
                   t.language         AS tx_lang,
                   t.vocal_isolation  AS tx_vocal_iso,
                   t.vocal_model      AS tx_vocal_model,
                   t.speakers_json    AS tx_speakers_json,
                   t.segment_count    AS tx_seg_count,
                   t.total_chunks     AS tx_chunks,
                   t.updated_at       AS tx_date,
                   t.word_count       AS tx_word_count,
                   t.char_count       AS tx_char_count,
                   t.speaker_count    AS tx_speaker_count,
                   t.warning_count    AS tx_warning_count,
                   t.app_version      AS tx_app_version,
                   d.abs_path         AS deriv_abs
            FROM media_files m
            LEFT JOIN transcripts t ON t.abs_path = m.abs_path
                       AND m.abs_path != '' AND t.complete = 1
            -- Are there derivations (summary/translation/correction run)?
            -- They survive transcript deletion, and the list must be able
            -- to show that — otherwise such a file looks untranscribed.
            -- UNION instead of three EXISTS: the three tables are tiny,
            -- join runs once instead of per row.
            LEFT JOIN (SELECT abs_path FROM summaries    WHERE abs_path != ''
                       UNION
                       SELECT abs_path FROM translations WHERE abs_path != ''
                       UNION
                       SELECT abs_path FROM llm_edits    WHERE abs_path != ''
                      ) d ON d.abs_path = m.abs_path AND m.abs_path != ''
"""


def _root_of(rel_path: str, abs_path: str | None, media_dir: str | None) -> str:
    """The folder ``rel_path`` is relative to — what a search hit hands on
    as its root (``file_dir``) to quote, open and play it.

    Taken from the identity where the two fit together. The stored
    ``media_dir`` doesn't always: some transcripts carry their file's own
    folder there while ``rel_path`` is relative to a root above it, and a
    hit's context ("Transcript not found"), its playback and its voices
    then point at a path that doesn't exist."""
    rel = (rel_path or "").strip("/")
    if abs_path and rel and not paths.is_absolute(rel_path):
        tail = "/" + rel
        if abs_path.endswith(tail):
            return abs_path[: -len(tail)] or "/"
    return media_dir or ""


def _name_folded(path: str | None) -> str | None:
    """SQL ``wx_name_folded(path)``: the file name — the part after the last
    "/" — case-folded, to match against a folded pattern. The queue editor
    narrows by it (core/queue.editor_where). SQLite has no function for the
    last "/": rtrim() over the whole path took 330 ms over 100,000 entries,
    this 40 — and LIKE alone folds only ASCII ("Éclair" ≠ "éclair")."""
    if not path:
        return path
    return path[path.rfind("/") + 1 :].casefold()


def _folded(text: str | None) -> str | None:
    """SQL ``wx_folded(text)``: the whole text case-folded. LIKE folds only
    ASCII, so a filter "été" missed "Été …" (core/_name_filter_sql)."""
    return text.casefold() if text else text


def _json_or(raw: Any, default):
    """Parse JSON defensively. On empty or corrupted value, ``default``
    is returned instead of raising an exception."""
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


class _Connection(sqlite3.Connection):
    """sqlite3.Connection subclass that allows weak references.

    The C base class doesn't support ``weakref`` — without this subclass,
    connections of dead threads couldn't be automatically freed via a
    WeakSet (#057).
    """

    pass


# ── Schema ────────────────────────────────────────────────────────────────

# The schema of the first release is the baseline: SCHEMA_SQL creates it
# whole. A database of an earlier build is refused (DatabaseVersionError),
# and so is one of a newer version. A later version that changes the layout
# adds its steps here, behind a backup.
# Tables of derived data that can be counted again from the segments at any
# time (conv_stats, conv_speakers, conv_windows: whisper_tm/conversations)
# are created by SCHEMA_SQL in every database at start.
SCHEMA_VERSION = 27


def _embedded(row, keys) -> bool:
    """Is this media row in the vector store? Its keys are the identity
    (rag.indexer._transcript_key); old rows keep their rel_path, and older
    ones the bare file name. Matched against the rel_path alone, no file
    ever counted as embedded."""
    rel = row["rel_path"] or ""
    return row["abs_path"] in keys or rel in keys or (bool(rel) and rel.rsplit("/", 1)[-1] in keys)


class DatabaseVersionError(RuntimeError):
    """A database this version can't open: from a development build before
    the baseline, or from a newer version."""


SCHEMA_SQL = """
-- Versioning
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Media files (cache for fast scans)
CREATE TABLE IF NOT EXISTS media_files (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Identity and display value, see docs/PATHS.md:
    --   abs_path = the identity (unique, see indexes below)
    --   rel_path = derived from it, relative to the currently active root
    abs_path     TEXT DEFAULT '',
    rel_path     TEXT NOT NULL,
    display_name TEXT,
    subfolder    TEXT DEFAULT '',
    extension    TEXT,
    is_video     INTEGER DEFAULT 0,
    file_size    INTEGER DEFAULT 0,
    modified_at  REAL DEFAULT 0,            -- Filesystem mtime
    duration_sec REAL DEFAULT 0,            -- FFprobe: exact duration (seconds)
    has_audio    INTEGER DEFAULT 1,
    media_title  TEXT DEFAULT '',            -- FFprobe metadata title
    -- Full media metadata (cached once per index, never recomputed)
    width        INTEGER DEFAULT 0,          -- video width (px)
    height       INTEGER DEFAULT 0,          -- video height (px)
    has_video    INTEGER DEFAULT 0,          -- real video stream (from FFprobe)
    audio_codec  TEXT DEFAULT '',
    video_codec  TEXT DEFAULT '',
    bitrate      INTEGER DEFAULT 0,          -- total bitrate (bps)
    -- Index tracking
    indexed_at   REAL DEFAULT 0,            -- time of the last FFprobe index (0 = never)
    index_mtime  REAL DEFAULT 0,            -- file mtime at index time (staleness check)
    status       TEXT DEFAULT 'pending',    -- pending, done, skipped, error
    skip_reason  TEXT DEFAULT '',
    last_scanned REAL DEFAULT 0,
    created_at   REAL DEFAULT (unixepoch()),
    updated_at   REAL DEFAULT (unixepoch())
);
-- The identity: unique where it is set. Partial, so a plain
-- "WHERE abs_path = ?" can't use it (SQLite can't know at planning time that
-- the parameter isn't '') — hence the plain lookup index beside it.
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_files_abs_unique
    ON media_files(abs_path) WHERE abs_path != '';
CREATE INDEX IF NOT EXISTS idx_media_abs ON media_files(abs_path);
CREATE INDEX IF NOT EXISTS idx_media_rel ON media_files(rel_path);
CREATE INDEX IF NOT EXISTS idx_media_status ON media_files(status);
CREATE INDEX IF NOT EXISTS idx_media_subfolder ON media_files(subfolder);
CREATE INDEX IF NOT EXISTS idx_media_ext ON media_files(extension);
-- The file list's sorts, and its name sort in index order (covering the
-- root check through abs_path) instead of sorting every row per page
CREATE INDEX IF NOT EXISTS idx_media_duration ON media_files(duration_sec);
CREATE INDEX IF NOT EXISTS idx_media_modified ON media_files(modified_at);
CREATE INDEX IF NOT EXISTS idx_media_size ON media_files(file_size);
CREATE INDEX IF NOT EXISTS idx_media_indexed ON media_files(indexed_at);
-- The file list's name sort goes by abs_path: rel_path is relative to the
-- folder that was open when a file was indexed, so one folder's files
-- sorted by it came out of order. The index on rel_path it replaces goes.
CREATE INDEX IF NOT EXISTS idx_media_abs_nocase ON media_files(abs_path COLLATE NOCASE);
DROP INDEX IF EXISTS idx_media_rel_abs_nocase;

-- Transcripts
CREATE TABLE IF NOT EXISTS transcripts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id         INTEGER REFERENCES media_files(id) ON DELETE CASCADE,
    -- As with media_files: abs_path is the identity, rel_path the
    -- derived display value (docs/PATHS.md).
    abs_path         TEXT DEFAULT '',
    rel_path         TEXT NOT NULL,
    language         TEXT DEFAULT '',
    model            TEXT DEFAULT '',
    vocal_isolation  INTEGER DEFAULT 0,
    vocal_model      TEXT DEFAULT '',
    lavasr           INTEGER DEFAULT 0,
    total_duration   TEXT DEFAULT '',
    duration_sec     REAL DEFAULT 0,
    processing_time  TEXT DEFAULT '',
    processing_secs  REAL DEFAULT 0,
    total_chunks     INTEGER DEFAULT 0,
    segment_count    INTEGER DEFAULT 0,
    word_count       INTEGER DEFAULT 0,
    char_count       INTEGER DEFAULT 0,
    speaker_count    INTEGER DEFAULT 0,
    warning_count    INTEGER DEFAULT 0,
    speakers_json    TEXT DEFAULT '[]',
    speaker_names    TEXT DEFAULT '{}',
    step_timings     TEXT DEFAULT '[]',
    warnings_json    TEXT DEFAULT '[]',
    complete         INTEGER DEFAULT 0,
    json_data        TEXT,                   -- optional full JSON
    media_dir        TEXT DEFAULT '',        -- config.media_dir at index time
    app_version      TEXT DEFAULT '',        -- app version that transcribed it (v13)
    settings_json    TEXT DEFAULT '',        -- how it was transcribed (v25)
    created_at       REAL DEFAULT (unixepoch()),
    updated_at       REAL DEFAULT (unixepoch())
);
-- As media_files: the identity (partial, unique) and the lookups; media_id
-- is the child side of ON DELETE CASCADE — without it every deleted
-- media_files row scanned the whole table
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_abs_unique
    ON transcripts(abs_path) WHERE abs_path != '';
CREATE INDEX IF NOT EXISTS idx_transcript_abs ON transcripts(abs_path);
CREATE INDEX IF NOT EXISTS idx_transcript_rel ON transcripts(rel_path);
CREATE INDEX IF NOT EXISTS idx_transcript_media ON transcripts(media_id);
CREATE INDEX IF NOT EXISTS idx_transcript_complete ON transcripts(complete);
CREATE INDEX IF NOT EXISTS idx_transcript_model ON transcripts(model);
CREATE INDEX IF NOT EXISTS idx_transcript_lang ON transcripts(language);

-- Segments (for fast search and analysis)
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    seg_index     INTEGER NOT NULL,
    speaker       TEXT DEFAULT '—',
    start_sec     REAL NOT NULL,
    end_sec       REAL NOT NULL,
    start_fmt     TEXT,
    end_fmt       TEXT,
    text          TEXT NOT NULL DEFAULT '',
    -- Schema v10: decoder confidence + reversible markers
    avg_logprob   REAL,
    phantom_score REAL DEFAULT 0,
    overlapped    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_seg_transcript ON segments(transcript_id);
CREATE INDEX IF NOT EXISTS idx_seg_speaker ON segments(speaker);

-- Schema v12: processing steps as their own table. Previously they only
-- existed as JSON in transcripts.step_timings — for analyses ("how much time
-- does separation via the library cost?") JSON had to be parsed every time.
-- Now one row per step, directly aggregatable.
CREATE TABLE IF NOT EXISTS processing_steps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    step       TEXT NOT NULL,
    group_nr        TEXT NOT NULL DEFAULT '',
    secs          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_transcript ON processing_steps(transcript_id);
CREATE INDEX IF NOT EXISTS idx_steps_step ON processing_steps(step);

-- Schema v19: incidents — permanent, per file, never truncated. One row per
-- warning, error, OOM retry or fallback maneuver, with the step
-- in which it happened. `abs_path` is the file's identity (docs/PATHS.md)
-- and survives re-runs: the history of a recording stays complete.
-- Empty abs_path = system-wide (not tied to a file).
CREATE TABLE IF NOT EXISTS incidents (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      REAL    NOT NULL DEFAULT (unixepoch()),
    abs_path  TEXT    NOT NULL DEFAULT '',
    filename  TEXT    NOT NULL DEFAULT '',
    job_id    TEXT    NOT NULL DEFAULT '',
    kind       TEXT    NOT NULL,          -- warning | error | oom | retry | system
    step   TEXT    NOT NULL DEFAULT '',
    text      TEXT    NOT NULL,
    count    INTEGER NOT NULL DEFAULT 1,
    log_file  TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_incidents_path ON incidents(abs_path, timestamp);
CREATE INDEX IF NOT EXISTS idx_incidents_time ON incidents(timestamp);
CREATE INDEX IF NOT EXISTS idx_incidents_kind  ON incidents(kind, timestamp);

-- FTS5 full-text index
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Word list of the FTS index. Costs no storage (view onto the index) and
-- provides the basis for fuzzy search: similar words are searched for in the
-- existing vocabulary, not across all segments.
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts_vocab
    USING fts5vocab(segments_fts, 'row');

-- Trigger for FTS sync
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text)
        VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text)
        VALUES('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Job history
CREATE TABLE IF NOT EXISTS job_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL,
    filename       TEXT,
    job_type       TEXT DEFAULT 'transcription',  -- transcription, separation, batch
    status         TEXT DEFAULT 'queued',
    model          TEXT DEFAULT '',
    language       TEXT DEFAULT '',
    duration_sec   REAL DEFAULT 0,
    processing_sec REAL DEFAULT 0,
    error_message  TEXT DEFAULT '',
    log_file       TEXT DEFAULT '',
    settings_json  TEXT DEFAULT '{}',
    created_at     REAL DEFAULT (unixepoch()),
    finished_at    REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_job_status ON job_history(status);
CREATE INDEX IF NOT EXISTS idx_job_type ON job_history(job_type);
CREATE INDEX IF NOT EXISTS idx_job_created ON job_history(created_at);
-- #063: every update_job() filters by job_id → index against full scan
CREATE INDEX IF NOT EXISTS idx_job_jobid ON job_history(job_id);

-- User settings (key-value for UI settings)
CREATE TABLE IF NOT EXISTS user_settings (
    key   TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL DEFAULT (unixepoch())
);

-- Skiplist
CREATE TABLE IF NOT EXISTS skiplist (
    -- Deliberately skipped files. The absolute path carries the identity
    -- (docs/PATHS.md) — without it, files would end up back in the queue
    -- after a root change, which is lost user intent, not
    -- just a cache miss.
    rel_path  TEXT NOT NULL,
    abs_path  TEXT DEFAULT '',
    reason    TEXT DEFAULT 'manual',
    added_at  REAL DEFAULT (unixepoch())
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_skiplist_abs_unique
    ON skiplist(abs_path) WHERE abs_path != '';
CREATE INDEX IF NOT EXISTS idx_skiplist_abs ON skiplist(abs_path);

-- Transcription queue (persistent, server-side).
-- The queue worker (core/queue.py) is the ONLY consumer: it works through the
-- pending entries sequentially. This means the queue keeps running when the
-- browser is closed, and resumes after a server restart.
CREATE TABLE IF NOT EXISTS queue_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    base_dir      TEXT DEFAULT '',
    settings_json TEXT DEFAULT '{}',
    status        TEXT DEFAULT 'pending',  -- pending|processing|done|error|skipped|cancelled
    position      INTEGER NOT NULL DEFAULT 0,
    job_id        TEXT DEFAULT '',
    error         TEXT DEFAULT '',
    created_at    REAL DEFAULT (unixepoch()),
    started_at    REAL DEFAULT 0,
    finished_at   REAL DEFAULT 0,
    interrupted   INTEGER DEFAULT 0          -- the server went down while it ran (v27)
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_items(status, position, id);
-- The file list's quick filters (failed, cancelled, in the queue) look up
-- every listed file's entries
CREATE INDEX IF NOT EXISTS idx_queue_filename ON queue_items(filename, status, id);

-- LLM summaries (DB is the primary source; JSON file remains as backup)
CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    -- As everywhere: abs_path is the identity, rel_path the display value.
    abs_path   TEXT DEFAULT '',
    rel_path   TEXT NOT NULL,
    sum_index  INTEGER NOT NULL DEFAULT 0,
    text       TEXT NOT NULL,
    prompt     TEXT DEFAULT '',
    model      TEXT DEFAULT '',
    created_at REAL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_summaries_relpath ON summaries(rel_path);
CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_unique
    ON summaries(abs_path, sum_index) WHERE abs_path != '';
CREATE INDEX IF NOT EXISTS idx_summaries_abs ON summaries(abs_path);

-- LLM translations (v16): one track per (file, language). segments_json =
-- [{id, text}] — id is the segment index in the original; timings/speakers
-- ALWAYS come from the original transcript when rendering/exporting.
CREATE TABLE IF NOT EXISTS translations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    abs_path      TEXT DEFAULT '',
    rel_path      TEXT NOT NULL,
    language      TEXT NOT NULL,
    segments_json TEXT NOT NULL,
    model         TEXT DEFAULT '',
    created_at    REAL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_translations_abs ON translations(abs_path, language);
CREATE INDEX IF NOT EXISTS idx_translations_rel ON translations(rel_path);

-- LLM correction runs (v16): payload_json carries the suggestions
-- [{id, old_value, new_value, reason, confidence}], or after applying, the
-- edits that were actually applied (old_value texts = undo history).
-- status: pending → applied/rejected (one run, one entry).
CREATE TABLE IF NOT EXISTS llm_edits (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    abs_path     TEXT DEFAULT '',
    rel_path     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    payload_json TEXT NOT NULL,
    model        TEXT DEFAULT '',
    created_at   REAL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_llm_edits_abs ON llm_edits(abs_path, status);

-- LLM queue (v17): ONE worker processes it sequentially.
-- parts_json collects intermediate results (map sections, chunk results)
-- so the UI can expand and follow the ongoing progress.
CREATE TABLE IF NOT EXISTS llm_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,
    filename     TEXT NOT NULL,
    params_json  TEXT DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'pending',
    position     INTEGER DEFAULT 0,
    progress     INTEGER DEFAULT 0,
    message      TEXT DEFAULT '',
    error       TEXT DEFAULT '',
    parts_json   TEXT DEFAULT '[]',
    partial_text     TEXT DEFAULT '',
    result_json  TEXT DEFAULT '',
    created_at   REAL DEFAULT (unixepoch()),
    started_at   REAL DEFAULT 0,
    finished_at  REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_queue_status
    ON llm_queue(status, position, id);

-- Conversation statistics (whisper_tm/conversations): counted from the
-- segments by every transcript write and by a background run. Derived
-- data — see the version log above SCHEMA_VERSION.
CREATE TABLE IF NOT EXISTS conv_stats (
    transcript_id      INTEGER PRIMARY KEY
                       REFERENCES transcripts(id) ON DELETE CASCADE,
    version            INTEGER NOT NULL,
    seg_count          INTEGER NOT NULL,
    char_count         INTEGER NOT NULL,
    name_date          REAL,
    computed_at        REAL,
    start_sec          REAL,
    end_sec            REAL,
    duration_sec       REAL,
    speech_sec         REAL,
    attributed_sec     REAL,
    unattributed_sec   REAL,
    words              INTEGER,
    sentences          INTEGER,
    questions          INTEGER,
    turns              INTEGER,
    short_turns        INTEGER,
    labels             INTEGER,
    speakers           INTEGER,
    top_share          REAL,
    longest_turn_sec   REAL,
    longest_turn_label TEXT,
    longest_turn_start REAL,
    punctuated         INTEGER,
    interview          INTEGER,
    wps                REAL,
    turn_rate          REAL,
    mean_turn_sec      REAL
);
CREATE INDEX IF NOT EXISTS idx_conv_stats_speakers ON conv_stats(speakers, duration_sec);
CREATE INDEX IF NOT EXISTS idx_conv_stats_duration ON conv_stats(duration_sec);
CREATE TABLE IF NOT EXISTS conv_speakers (
    transcript_id      INTEGER NOT NULL
                       REFERENCES transcripts(id) ON DELETE CASCADE,
    label              TEXT NOT NULL,
    talk_sec           REAL,
    words              INTEGER,
    sentences          INTEGER,
    questions          INTEGER,
    turns              INTEGER,
    short_turns        INTEGER,
    longest_turn_sec   REAL,
    longest_turn_start REAL,
    first_sec          REAL,
    last_sec           REAL,
    PRIMARY KEY (transcript_id, label)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS conv_windows (
    transcript_id      INTEGER PRIMARY KEY
                       REFERENCES transcripts(id) ON DELETE CASCADE,
    window_sec         INTEGER NOT NULL,
    data               TEXT NOT NULL
);
"""


# ── Database Engine ───────────────────────────────────────────────────────


class Database:
    """Thread-safe SQLite database with connection pool.

    Features:
    - WAL mode for concurrent read/write access
    - FTS5 full-text index for millisecond search
    - Automatic schema management
    - Bulk insert for fast batch operations
    - Thread-local connections
    """

    # Whitelist of allowed columns for dynamically built UPSERT/UPDATE statements
    # (#061: prevents arbitrary kwargs keys from ending up as column names in SQL)
    _MEDIA_FILE_COLS = frozenset(
        {
            "abs_path",
            "display_name",
            "subfolder",
            "extension",
            "is_video",
            "file_size",
            "modified_at",
            "duration_sec",
            "has_audio",
            "media_title",
            "width",
            "height",
            "has_video",
            "audio_codec",
            "video_codec",
            "bitrate",
            "indexed_at",
            "index_mtime",
            "status",
            "skip_reason",
            "last_scanned",
            "updated_at",
        }
    )
    _JOB_COLS = frozenset(
        {
            "filename",
            "job_type",
            "status",
            "model",
            "language",
            "duration_sec",
            "processing_sec",
            "error_message",
            "log_file",
            "settings_json",
            "finished_at",
            "created_at",
        }
    )
    # #064: upper bound for job_history — older entries are truncated on insert
    _JOB_HISTORY_MAX = 10000

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        # Registry of ALL created connections (across all threads) —
        # needed so reset()/close_all() can truly close all of them.
        # WeakSet: connections of dead request threads are automatically
        # freed instead of being held forever (#057).
        self._all_conns: weakref.WeakSet[sqlite3.Connection] = weakref.WeakSet()
        self._conns_lock = threading.Lock()
        # Generation counter: close_all() increments it so running threads
        # detect a globally closed connection and lazily recreate it (#059).
        self._generation = 0
        # External backup (NAS): only ONE copy at a time — two parallel
        # runs would write the same DB twice over the network.
        self._ext_backup_lock = threading.Lock()
        self._ext_backup_running = False
        # One backup at a time: each stages a copy as big as the database
        # next to it, two at once need twice the space
        self._backup_lock = threading.Lock()
        # JSON sync: failed imports per (file, mtime) — see sync_from_json_dir.
        # In memory: after a restart a broken file gets its attempts again.
        self._sync_failures: dict[tuple[str, float], int] = {}
        self._sync_failures_lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize the database and create the schema.

        A new database gets SCHEMA_SQL whole. An existing one is checked
        first — before the schema script, whose indexes would fail on a
        layout it doesn't know: it must be at SCHEMA_VERSION."""
        with self._init_lock:
            if self._initialized:
                return
            with self.connection() as conn:
                current = self._stored_version(conn)
                filled = self._has_rows(conn)
            if current > SCHEMA_VERSION:
                raise DatabaseVersionError(
                    f"The database {self._db_path} is from a newer version of "
                    f"Whisper Transcription Manager "
                    f"(schema {current}; this one knows up to {SCHEMA_VERSION}). "
                    f"Update the program, or restore a backup made by this version."
                )
            if (current or filled) and current < SCHEMA_VERSION:
                raise DatabaseVersionError(
                    f"The database {self._db_path} is from a development build before "
                    f"the first release (schema {current}); this version opens schema "
                    f"{SCHEMA_VERSION}. Start with a new database (database.path)."
                )
            with self.connection() as conn:
                conn.executescript(SCHEMA_SQL)
                if not current:
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                        ("version", str(SCHEMA_VERSION)),
                    )
            self._ensure_statistics()
            self._initialized = True

    @staticmethod
    def _stored_version(conn) -> int:
        """The schema version written into the database — 0 for a new one."""
        try:
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'version'").fetchone()
            return int(row["value"]) if row and row["value"] is not None else 0
        except (sqlite3.OperationalError, TypeError, ValueError):
            return 0

    @staticmethod
    def _has_rows(conn) -> bool:
        """Does it hold a library already (a database without a version)?"""
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('media_files', 'transcripts')"
            )
        }
        return any(
            conn.execute(f"SELECT 1 FROM {t} LIMIT 1").fetchone()  # noqa: S608
            for t in sorted(tables)
        )

    def _ensure_statistics(self) -> None:
        """Run ANALYZE if the database has no statistics.

        Without ``sqlite_stat1``, SQLite guesses the access paths — and here
        it guesses catastrophically wrong. The join for the file list

            media_files m LEFT JOIN transcripts t ON t.abs_path = m.abs_path
                                                 AND t.complete = 1

        went through ``idx_transcript_complete``: ``complete = 1`` matches
        practically every transcript and is worthless as a join key. On
        a large library, loading from the database took **tens of
        seconds**; the UI appeared frozen the whole time. With statistics,
        SQLite chooses ``idx_transcript_abs`` and it takes **milliseconds** —
        for a call that itself costs well under a second.

        The statistics are created once and afterward only refreshed
        when the dataset has changed significantly (see
        :meth:`refresh_statistics`).
        """
        try:
            with self.connection() as conn:
                present = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
                ).fetchone()
                if present:
                    return
                lines = conn.execute("SELECT COUNT(*) FROM media_files").fetchone()[0]
                if not lines:
                    return  # empty DB: nothing to analyze
                logger.info("Generating query statistics (ANALYZE)...")
                conn.execute("ANALYZE")
                logger.info("Query statistics generated (%d media files)", lines)
        except Exception as e:  # noqa — works without statistics, just slower
            logger.warning("ANALYZE failed: %s", e)

    #: Tables whose statistics steer the hot queries (file list, search,
    #: queue filters). The FTS shadow tables and the rest don't need them.
    _ANALYZE_TABLES = ("media_files", "transcripts", "segments", "queue_items")

    @classmethod
    def _analyze_hot_tables(cls, conn) -> None:
        """Refresh the planner statistics without a full ANALYZE.

        A plain ``ANALYZE`` reads every index of the whole database —
        including millions of segments and the FTS tables — and holds the
        write lock while doing it. ``analysis_limit`` samples each index
        instead (0.06 s instead of 1.9 s on a multi-GB database); the
        planner only needs the proportions.
        """
        conn.execute("PRAGMA analysis_limit=1000")
        for table in cls._ANALYZE_TABLES:
            conn.execute(f"ANALYZE {table}")

    def refresh_statistics(self, min_growth: int = 5000) -> None:
        """Refresh statistics when the dataset has grown significantly.

        Called after a scan. Without this, statistics become stale as
        a library grows, and the planner eventually goes wrong again.
        """
        try:
            with self.connection() as conn:
                now = conn.execute("SELECT COUNT(*) FROM media_files").fetchone()[0]
                last_one = conn.execute(
                    "SELECT value FROM schema_info WHERE key='analyzed_at_count'"
                ).fetchone()
                last_one = int(last_one["value"]) if last_one and last_one["value"] else 0
                if abs(now - last_one) < min_growth:
                    return
                self._analyze_hot_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                    ("analyzed_at_count", str(now)),
                )
                logger.info("Query statistics refreshed (%d media files)", now)
        except Exception as e:  # noqa
            logger.debug("ANALYZE refresh: %s", e)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the thread-local connection."""
        conn = getattr(self._local, "conn", None)
        # #059: if this connection was globally closed by close_all(),
        # the generation no longer matches → discard and recreate it,
        # instead of continuing to work with a closed connection.
        if conn is not None and getattr(self._local, "conn_gen", None) != self._generation:
            conn = None
        if conn is None:
            conn = sqlite3.connect(
                self._db_path,
                timeout=30,
                check_same_thread=False,
                factory=_Connection,  # #057: weakref-capable subclass
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")  # #060: enable ON DELETE CASCADE
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            conn.execute("PRAGMA busy_timeout=30000")  # #070: consistent with timeout=30
            conn.create_function("wx_name_folded", 1, _name_folded, deterministic=True)
            conn.create_function("wx_folded", 1, _folded, deterministic=True)

            # logging.trace_sql: log EVERY SQL statement. Very noisy,
            # but exactly what you want when you need to know what the DB is doing.
            from whisper_tm.config import config as _cfg

            if _cfg.get("logging", "trace_sql", default=False):
                _sql_log = logging.getLogger("whisper_tm.sql")
                conn.set_trace_callback(lambda stmt: _sql_log.debug("%s", " ".join(stmt.split())))

            self._local.conn = conn
            self._local.conn_gen = self._generation
            with self._conns_lock:
                self._all_conns.add(conn)
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for a database connection with auto-commit."""
        from whisper_tm.core import debug

        conn = self._get_connection()
        _t0 = time.perf_counter()
        try:
            yield conn
            conn.commit()
        except Exception as _e:  # noqa
            conn.rollback()
            raise
        finally:
            # Time per DB transaction. Slow transactions (>250ms) immediately
            # stand out in the debug log — with WAL + NAS this is often the culprit.
            _dt = time.perf_counter() - _t0
            debug.record("db: transaction", _dt)
            if debug.is_enabled() and _dt > 0.25:
                logger.debug("slow DB transaction: %s", debug.fmt_secs(_dt))

    def close(self) -> None:
        """Close the current thread's connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            with self._conns_lock:
                self._all_conns.discard(conn)
            self._local.conn = None

    def close_all(self) -> None:
        """Close ALL connections of all threads.

        Needed before deleting the DB file: otherwise worker/request threads
        keep the file open (Windows: unlink fails; Linux: orphaned
        handles write into the deleted file).
        """
        with self._conns_lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
            # #059: increment the generation so other threads recognize
            # their now-closed connection and lazily recreate it.
            self._generation += 1
        for conn in conns:
            try:
                conn.close()
            except Exception as e:
                logger.debug("Connection close failed: %s", e)

    def reset(self) -> None:
        """Fully reset the DB: close all connections,
        clear the initialized flag. Call initialize() again afterward.
        """
        # #072: under _init_lock, otherwise a race with initialize()
        with self._init_lock:
            self.close_all()
            self._initialized = False
            # Reset thread-local storage
            self._local = threading.local()

    # ── Transcript operations ────────────────────────────────────────

    def upsert_transcript(
        self, rel_path: str, data: dict, media_dir: str = "", abs_path: str = ""
    ) -> int:
        """Save/update a transcript with all segments in the DB.

        Args:
            rel_path: media file path relative to media_dir (display value)
            data: full transcript dict (like the JSON file)
            media_dir: root that rel_path is relative to
            abs_path: actual location of the media file. Whoever knows it should
                pass it — it's the identity (docs/PATHS.md). Only if it's
                missing is it reconstructed from media_dir + rel_path; this exact
                reconstruction produced, with nested .transcripts/ dirs, a
                doubled path (/media/sub/sub/b.mp3).
        Returns:
            transcript_id
        """
        # Determine the identity (docs/PATHS.md). Without it, the conflict
        # below doesn't apply and an update would become a duplicate — so as a
        # fallback, use the currently configured root instead of leaving it empty.
        if abs_path:
            abs_path = paths.canon(abs_path)
        else:
            root = media_dir
            if not root:
                try:
                    from whisper_tm.config import config as _cfg

                    root = str(_cfg.media_dir)
                except Exception as _e:  # noqa
                    root = ""
            abs_path = paths.abs_from(rel_path, root) if root else ""
        if not abs_path:
            logger.warning(
                "upsert_transcript without derivable abs_path (%s) — the row "
                "does not participate in identity enforcement",
                rel_path,
            )
        # Conversation statistics (whisper_tm/conversations), counted
        # before the transaction so the write lock isn't held for it. A
        # failure must never cost the transcript: the background run counts
        # a transcript without a row.
        conv = conv_store = None
        conv_moved = False
        try:
            from whisper_tm.conversations import store as conv_store

            if conv_store.enabled():
                conv = conv_store.compute(data)
        except Exception as e:  # noqa: BLE001
            logger.warning("Conversation statistics not counted for %s: %s", rel_path, e)
        with self.connection() as conn:
            segments = data.get("segments", [])
            speakers = data.get("speakers", [])
            all_text = " ".join(s.get("text", "") for s in segments)
            is_complete = int(bool(data.get("complete", False)))

            # #067/#076: create/update the media file row FIRST, so
            # the media_id subselect below applies. Derive status from the
            # complete flag instead of a blanket 'done' (otherwise inconsistent stats).
            conn.execute(
                """
                INSERT INTO media_files (rel_path, abs_path, status, updated_at)
                VALUES (?, ?, ?, unixepoch())
                ON CONFLICT(abs_path) WHERE abs_path != '' DO UPDATE SET
                    rel_path=excluded.rel_path,
                    status=excluded.status, updated_at=unixepoch()
            """,
                (rel_path, abs_path, "done" if is_complete else "pending"),
            )

            # Upsert transcript. media_id via subselect on media_files (#067),
            # transcript_id via RETURNING instead of a separate SELECT roundtrip (#065).
            cur = conn.execute(
                """
                INSERT INTO transcripts (
                    media_id, rel_path, language, model, vocal_isolation, vocal_model,
                    lavasr, total_duration, duration_sec,
                    processing_time, processing_secs, total_chunks,
                    segment_count, word_count, char_count, speaker_count,
                    warning_count, speakers_json, speaker_names,
                    step_timings, warnings_json, complete, json_data,
                    media_dir, abs_path, app_version, settings_json, updated_at
                ) VALUES (
                    (SELECT id FROM media_files
                      WHERE abs_path = ? AND abs_path != '' LIMIT 1),
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, unixepoch()
                ) ON CONFLICT(abs_path) WHERE abs_path != '' DO UPDATE SET
                    media_id=excluded.media_id,
                    rel_path=excluded.rel_path,
                    language=excluded.language, model=excluded.model,
                    vocal_isolation=excluded.vocal_isolation,
                    vocal_model=excluded.vocal_model,
                    lavasr=excluded.lavasr,
                    total_duration=excluded.total_duration,
                    duration_sec=excluded.duration_sec,
                    processing_time=excluded.processing_time,
                    processing_secs=excluded.processing_secs,
                    total_chunks=excluded.total_chunks,
                    segment_count=excluded.segment_count,
                    word_count=excluded.word_count,
                    char_count=excluded.char_count,
                    speaker_count=excluded.speaker_count,
                    warning_count=excluded.warning_count,
                    speakers_json=excluded.speakers_json,
                    speaker_names=excluded.speaker_names,
                    step_timings=excluded.step_timings,
                    warnings_json=excluded.warnings_json,
                    complete=excluded.complete,
                    json_data=excluded.json_data,
                    media_dir=COALESCE(NULLIF(excluded.media_dir, ''), transcripts.media_dir),
                    -- Like media_dir: don't replace an existing value with an
                    -- empty one when the caller doesn't know media_dir.
                    abs_path=COALESCE(NULLIF(excluded.abs_path, ''), transcripts.abs_path),
                    -- Never overwrite the version with empty: DB sync also
                    -- reads in old JSON files without app_version.
                    app_version=COALESCE(NULLIF(excluded.app_version, ''), transcripts.app_version),
                    -- Same rule: a JSON from before v25 has no settings, and
                    -- re-reading it must not erase what was recorded.
                    settings_json=COALESCE(NULLIF(excluded.settings_json, ''),
                                           transcripts.settings_json),
                    updated_at=unixepoch()
                RETURNING id
            """,
                (
                    abs_path,  # media_id subselect: via the identity
                    rel_path,  # transcripts.rel_path (display value)
                    data.get("language", ""),
                    data.get("model", ""),
                    int(bool(data.get("vocal_isolation", False))),
                    data.get("vocal_model", "") or "",
                    # Legacy column: LavaSR was removed. Kept so
                    # transcripts made with it keep their record when their
                    # JSON is imported again; new runs never set it.
                    int(bool(data.get("lavasr", False))),
                    data.get("total_duration", ""),
                    data.get("total_duration_sec", 0) or 0,
                    data.get("processing_time", ""),
                    data.get("processing_secs", 0) or 0,
                    data.get("total_chunks", 0) or 0,
                    len(segments),
                    len(all_text.split()),
                    len(all_text),
                    len(set(s.get("speaker", "") for s in segments if s.get("speaker"))),
                    data.get("warning_count", 0) or 0,
                    json.dumps(speakers),
                    json.dumps(data.get("speaker_names", {})),
                    json.dumps(data.get("step_timings", [])),
                    json.dumps(data.get("warnings", [])),
                    is_complete,
                    json.dumps(data, ensure_ascii=False),
                    media_dir,
                    # Without a root AND without a given abs_path, no
                    # absolute path can be formed; the field then stays empty and the
                    # lookup falls back to the legacy-row tier.
                    abs_path,
                    data.get("app_version", "") or "",
                    (
                        json.dumps(data["settings"], ensure_ascii=False, sort_keys=True)
                        if isinstance(data.get("settings"), dict) and data["settings"]
                        else ""
                    ),
                ),
            )
            tid = cur.fetchone()[0]

            # Replace segments
            conn.execute("DELETE FROM segments WHERE transcript_id = ?", (tid,))
            if segments:
                conn.executemany(
                    """
                    INSERT INTO segments (
                        transcript_id, seg_index, speaker,
                        start_sec, end_sec, start_fmt, end_fmt, text,
                        avg_logprob, phantom_score, overlapped
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            tid,
                            i,
                            s.get("speaker", "—"),
                            s.get("start", 0),
                            s.get("end", 0),
                            s.get("start_fmt", ""),
                            s.get("end_fmt", ""),
                            s.get("text", ""),
                            s.get("avg_logprob"),
                            s.get("phantom_score", 0) or 0,
                            1 if s.get("overlapped") else 0,
                        )
                        for i, s in enumerate(segments)
                    ],
                )

            # Replace processing steps (schema v12) — one row per
            # step, directly aggregatable for performance analyses.
            conn.execute("DELETE FROM processing_steps WHERE transcript_id = ?", (tid,))
            steps = data.get("step_timings") or []
            if steps:
                conn.executemany(
                    "INSERT INTO processing_steps "
                    "(transcript_id, step, group_nr, secs) VALUES (?, ?, ?, ?)",
                    [
                        (
                            tid,
                            s.get("label", "?"),
                            s.get("group", "") or "",
                            float(s.get("secs", 0) or 0),
                        )
                        for s in steps
                        if isinstance(s, dict)
                    ],
                )

            # In the same transaction: the numbers are never older than
            # the segments they were counted from. Not counted, the old
            # rows go: a merge of two speaker labels keeps the segment and
            # character counts, so they would pass for current.
            # A failure of the statistics alone is shrugged off — but when
            # SQLite rolled the whole transaction back (disk full, I/O
            # error), the transcript above is gone too: then the error has
            # to reach the caller instead of a "saved" that isn't true.
            if conv is not None:
                try:
                    conv_store.write(
                        conn,
                        tid,
                        conv,
                        seg_count=len(segments),
                        char_count=len(all_text),
                        abs_path=abs_path,
                    )
                    conv_moved = True
                except Exception as e:  # noqa: BLE001
                    if not conn.in_transaction:
                        raise
                    conv = None
                    logger.warning("Conversation statistics not written for %s: %s", rel_path, e)
            if conv is None and conv_store is not None:
                try:
                    conv_moved = conv_store.drop(conn, tid) > 0
                except Exception as e:  # noqa: BLE001
                    if not conn.in_transaction:
                        raise
                    logger.warning(
                        "Old conversation statistics not dropped for %s: %s", rel_path, e
                    )

        if conv_moved:
            conv_store.changed()
        # Outside the transaction: the counter writes to the DB itself and
        # the backup reads it — both would block each other with a still
        # open write transaction.
        self.note_transcript_change()
        return tid

    def delete_transcript(self, rel_path: str, media_dir: str | None = None) -> bool:
        """Delete the transcript and all segments."""
        with self.connection() as conn:
            row = self.resolve_transcript(conn, rel_path, media_dir, "id, rel_path, abs_path")
            if not row:
                return False
            # Cleanup runs via the identity, NOT via the found row's
            # rel_path: since v8 that's no longer unique, and a
            # rel_path-based cleanup would also delete the summaries of an
            # identically named file from another root — that's
            # unrecoverable (docs/PATHS.md).
            target_abs = row["abs_path"] or ""
            conn.execute("DELETE FROM segments WHERE transcript_id = ?", (row["id"],))
            conn.execute("DELETE FROM transcripts WHERE id = ?", (row["id"],))
            # Derivations (summaries/translations/llm_edits) are KEPT.
            # They hang off abs_path, not the transcript, and remain viewable
            # even without the original. Previously, this exact place
            # deleted the summaries too, but not translations or correction
            # runs — an inconsistent half-and-half state. Anyone who really
            # wants to get rid of them deletes them explicitly in the view.
            if target_abs:
                # #073: preserve skip status — only reset to 'pending'
                # if the file was not deliberately skipped.
                conn.execute(
                    """
                    UPDATE media_files SET status='pending', updated_at=unixepoch()
                     WHERE abs_path = ? AND status != 'skipped'
                """,
                    (target_abs,),
                )

        # Same as with upsert: only count after the transaction has finished.
        self.note_transcript_change()
        # The conversation rows went with the transcript (ON DELETE
        # CASCADE); the held search results go now, not when they expire
        from whisper_tm.conversations import store as conv_store

        conv_store.changed()
        return True

    # ── Resolution: which record is meant? ─────────────────────
    #
    # ONE path for all identity lookups. The model is in docs/PATHS.md:
    # abs_path is the identity, rel_path a derived display value. It
    # depends on the currently configured media directory, and that changes
    # at runtime — exactly that once caused a large share of transcripts to
    # become unfindable.
    #
    # Order:
    #   1. abs_path (unique, indexed)   — the normal case
    #   2. legacy rows without abs_path — counted, see _old_lines
    #
    # There is NO basename search. It was the stopgap for broken
    # rel_path values and invites a name collision: a/intro.mp4
    # and b/intro.mp4 are two files, not one.
    _old_lines = 0  # diagnostic: how often did tier 2 still apply?

    @classmethod
    def resolve_transcript(
        cls, conn, rel_path: str, media_dir: str | None, columns: str = "*", only_done: bool = False
    ):
        """Find the transcript row. Returns None if it doesn't exist."""
        done = " AND complete = 1" if only_done else ""

        # Tier 1: via the identity. If the caller doesn't know a root,
        # use the configured one — a lookup without a root is no longer
        # resolvable under the new model and would be silent guessing.
        root = media_dir
        if not root:
            try:
                from whisper_tm.config import config as _cfg

                root = str(_cfg.media_dir)
            except Exception as _e:  # noqa
                root = ""

        if root:
            # rel_path can already be absolute (caller with full path).
            # Detection MUST go through paths.is_absolute: a colon
            # is a legal filename character on Linux, so "10:30 Meeting.mp4"
            # is not a Windows drive (docs/PATHS.md).
            target = (
                paths.canon(rel_path)
                if paths.is_absolute(rel_path)
                else paths.abs_from(rel_path, root)
            )
            # An empty target value is NOT a search value: "WHERE abs_path = ''"
            # would match every legacy row without an identity and return a
            # completely unrelated file.
            if target:
                row = conn.execute(
                    f"SELECT {columns} FROM transcripts WHERE abs_path = ?{done} LIMIT 1",
                    (target,),
                ).fetchone()
                if row:
                    return row

        # Tier 2: via the derived rel_path. Last resort for
        # callers that don't know the root, and for legacy rows without
        # abs_path. Since v8, rel_path is NO LONGER unique — if multiple
        # rows match, the query is ambiguous and the answer is a guess. This is
        # reported instead of silently taking the first one.
        # ONLY legacy rows without an identity. This is exactly what docs/PATHS.md
        # §"Lookup and cleanup" requires — this restriction was missing here.
        #
        # Without it, tier 2 also found rows that carry a valid, but DIFFERENT
        # identity. For read calls that was only a wrong
        # display; for delete_transcript() it was data loss: the
        # queue worker deletes via this path on a re-transcription, and
        # thereby hit the transcript of the identically named file from another
        # root. Reproduced with two roots that both contain series/f1.mp4.
        hits = conn.execute(
            f"SELECT {columns} FROM transcripts WHERE rel_path = ? "
            f"AND (abs_path IS NULL OR abs_path = ''){done} LIMIT 2",
            (rel_path,),
        ).fetchall()
        if not hits:
            return None
        cls._old_lines += 1
        if len(hits) > 1:
            logger.warning(
                "Ambiguous lookup via rel_path: %s matches multiple "
                "files. The root belongs with the call (docs/PATHS.md).",
                rel_path,
            )
        elif cls._old_lines in (1, 10, 100, 1000):
            logger.info(
                "Transcript found via rel_path instead of abs_path (#%d): %s — "
                "tools/migrate_paths.py cleans up legacy records",
                cls._old_lines,
                rel_path,
            )
        return hits[0]

    def merge_transcript_settings(self, abs_path: str, patch: dict) -> bool:
        """Add keys to a transcript's recorded settings (schema v25).

        For work done on a transcript AFTER it was made — a voices-only run
        records what it did under ``voices``, next to how the transcript
        itself was produced, instead of overwriting that. Returns False if
        there is no transcript with this identity.
        """
        target = paths.canon(abs_path)
        with self.connection() as conn:
            row = conn.execute(
                "SELECT settings_json FROM transcripts WHERE abs_path = ?", (target,)
            ).fetchone()
            if row is None:
                return False
            try:
                current = json.loads(row["settings_json"] or "{}")
            except (TypeError, ValueError):
                current = {}
            if not isinstance(current, dict):
                current = {}
            current.update(patch)
            conn.execute(
                "UPDATE transcripts SET settings_json = ? WHERE abs_path = ?",
                (json.dumps(current, ensure_ascii=False, sort_keys=True), target),
            )
        return True

    def transcript_speakers(self, abs_path: str) -> list[tuple[str, float, float]]:
        """``(speaker, start, end)`` of every segment of the transcript with
        this identity — the labels a voices-only run aligns its fresh
        diarization to (core/speaker.align_labels). Empty without transcript.
        """
        target = paths.canon(abs_path)
        with self.connection() as conn:
            return [
                (r["speaker"] or "", float(r["start_sec"] or 0), float(r["end_sec"] or 0))
                for r in conn.execute(
                    "SELECT s.speaker, s.start_sec, s.end_sec FROM segments s "
                    "JOIN transcripts t ON t.id = s.transcript_id "
                    "WHERE t.abs_path = ? ORDER BY s.seg_index",
                    (target,),
                )
            ]

    def get_transcript(self, rel_path: str, media_dir: str | None = None) -> dict | None:
        """Load a transcript from the DB.

        Args:
            rel_path: path relative to the current media directory.
            media_dir: current media directory. If given, and lookup by rel_path
                fails, an additional search by absolute path is done — needed
                for transcripts from before a
                media directory change (see _find_transcript_by_abs).
        """
        with self.connection() as conn:
            row = self.resolve_transcript(conn, rel_path, media_dir, only_done=True)
            if not row:
                return None
            if row["json_data"]:
                # #068: corrupt JSON must not raise → fall back to segment
                # reconstruction instead of crashing.
                try:
                    return json.loads(row["json_data"])
                except (json.JSONDecodeError, ValueError, TypeError):
                    logger.warning(
                        "Corrupt json_data for %s — reconstructing from segments", rel_path
                    )
            # Fallback: reconstruct from segments.
            # IMPORTANT: map DB columns (start_sec/end_sec) to the JSON format
            # (start/end) — viewer/export expect seg["start"].
            segs = conn.execute(
                "SELECT * FROM segments WHERE transcript_id = ? ORDER BY seg_index", (row["id"],)
            ).fetchall()
            seg_list = [
                {
                    "speaker": s["speaker"] or "—",
                    "start": s["start_sec"],
                    "end": s["end_sec"],
                    "start_fmt": s["start_fmt"] or "",
                    "end_fmt": s["end_fmt"] or "",
                    "text": s["text"] or "",
                }
                for s in segs
            ]
            speakers = sorted(
                set(s["speaker"] for s in seg_list if s["speaker"] and s["speaker"] != "—")
            )
            return {
                "filename": rel_path,
                "language": row["language"],
                "model": row["model"],
                "speakers": speakers,
                "speaker_names": _json_or(row["speaker_names"], {}),
                "segments": seg_list,
                "processing_time": row["processing_time"],
                "total_duration": row["total_duration"],
                "total_duration_sec": row["duration_sec"],
                "processing_secs": row["processing_secs"],
                "vocal_isolation": bool(row["vocal_isolation"]),
                "vocal_model": row["vocal_model"],
                "lavasr": bool(row["lavasr"]),
                "step_timings": _json_or(row["step_timings"], []),
                "warning_count": row["warning_count"],
                "complete": True,
            }

    def is_complete(self, rel_path: str, media_dir: str | None = None) -> bool:
        """Check whether the transcript is complete in the DB."""
        with self.connection() as conn:
            row = self.resolve_transcript(conn, rel_path, media_dir, "complete")
            return bool(row and row["complete"])

    def get_completion_set(self, abs_paths=None) -> set[str]:
        """Return all completed **abs_paths** as a set.

        A single query instead of N individual is_complete() calls.
        For 50k transcripts: ~50ms instead of ~50s (1000x faster).

        Keyed on the identity, not the display value
        (docs/PATHS.md). Via ``rel_path`` there were two failure modes, both
        reproduced: a file counted as complete because an *identically named* one from
        another root had a transcript; and the same file counted as
        pending as soon as it was scanned from the parent folder (different
        ``rel_path``) — it then ran through the heal again on every scan.

        Rows without an identity (legacy records) don't show up here; for those
        the individual lookup via :meth:`resolve_transcript` still applies.

        Args:
            abs_paths: optionally restrict to these identities (an
                opened folder then doesn't load the whole library).
        """
        with self.connection() as conn:
            if abs_paths is not None:
                keys = [k for k in abs_paths if k]
                out: set[str] = set()
                for i in range(0, len(keys), 900):
                    batch = keys[i : i + 900]
                    ph = ",".join(["?"] * len(batch))
                    rows = conn.execute(
                        f"SELECT abs_path FROM transcripts "
                        f"WHERE complete = 1 AND abs_path != '' "
                        f"AND abs_path IN ({ph})",
                        batch,
                    ).fetchall()
                    out.update(r["abs_path"] for r in rows)
                return out
            rows = conn.execute(
                "SELECT abs_path FROM transcripts WHERE complete = 1 AND abs_path != ''"
            ).fetchall()
            return {r["abs_path"] for r in rows}

    # ── RAG source data (for the vector-store indexer) ─────────────────────────────
    def get_rag_candidates(self) -> list[dict]:
        """All complete transcripts as the RAG source list.

        Returns:
            List of dicts with id, abs_path (= identity and thus
            transcript_key), rel_path (display), language, model, media_dir,
            seg_count, updated_at, has_json.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, abs_path, rel_path, language, model, media_dir,
                       segment_count AS seg_count, updated_at,
                       CASE WHEN json_data IS NOT NULL AND json_data != ''
                            THEN 1 ELSE 0 END AS has_json
                FROM transcripts
                WHERE complete = 1
                ORDER BY id
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def export_candidates(
        self, root: str = "", since: float | None = None, until: float | None = None
    ) -> list[dict]:
        """The complete transcripts below ``root`` (an absolute folder, its
        stored spelling; "" = all), with ``added_at`` — when the recording
        came into the library (its media row; the transcript's own time
        without one) — within ``since`` ≤ added_at < ``until`` (unix
        seconds; None = open). Same keys as :meth:`get_rag_candidates`."""
        sql = (
            "SELECT t.id, t.abs_path, t.rel_path, t.language, t.model, t.media_dir, "
            "t.segment_count AS seg_count, t.updated_at, "
            "COALESCE(m.created_at, t.created_at) AS added_at "
            "FROM transcripts t LEFT JOIN media_files m ON m.id = t.media_id "
            "WHERE t.complete = 1"
        )
        args: list = []
        if root:
            root = root.rstrip("/") + "/"
            # A range, not LIKE: "_" and "%" are ordinary in folder names
            sql += " AND t.abs_path >= ? AND t.abs_path < ?"
            args += [root, root[:-1] + "0"]
        if since is not None:
            sql += " AND COALESCE(m.created_at, t.created_at) >= ?"
            args.append(since)
        if until is not None:
            sql += " AND COALESCE(m.created_at, t.created_at) < ?"
            args.append(until)
        with self.connection() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY t.abs_path", args).fetchall()]

    def count_rag_candidates(self) -> int:
        """``len(get_rag_candidates())`` without loading them — the status
        used to load every row just to count them."""
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM transcripts WHERE complete = 1").fetchone()[0]

    def get_segments_for_rag(self, transcript_id: int) -> list[dict]:
        """Load a transcript's segments for the RAG chunker.

        Primarily from the segments table; falls back to json_data for older
        imports without segment rows.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT seg_index, speaker, start_sec, end_sec, text "
                "FROM segments WHERE transcript_id = ? ORDER BY seg_index",
                (transcript_id,),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
            # Fallback: reconstruct from json_data
            row = conn.execute(
                "SELECT json_data FROM transcripts WHERE id = ?", (transcript_id,)
            ).fetchone()
            if row and row["json_data"]:
                try:
                    data = json.loads(row["json_data"])
                except Exception:
                    return []
                out = []
                for i, s in enumerate(data.get("segments", [])):
                    out.append(
                        {
                            "seg_index": i,
                            "speaker": s.get("speaker") or "—",
                            "start_sec": s.get("start", s.get("start_sec", 0.0)),
                            "end_sec": s.get("end", s.get("end_sec", 0.0)),
                            "text": s.get("text", ""),
                        }
                    )
                return out
            return []

    def get_transcript_meta_batch(self, abs_paths: list[str] | None = None) -> dict[str, dict]:
        """Load transcript metadata for multiple files at once.

        Keyed on ``abs_path`` — the identity (docs/PATHS.md). Via
        ``rel_path``, two identically named files from different
        roots shared ONE metadata record, and which of the two got it depended on
        row order: the file list showed the duration, model, and
        speakers of the other file.

        Args:
            abs_paths: list of identities. None = all completed.

        Returns:
            Dict: abs_path -> metadata dict
        """
        # Explicit column list WITHOUT json_data: SELECT * would otherwise load
        # the complete transcript JSON of all files into RAM — on every
        # directory scan (50k transcripts -> gigabytes).
        _META_COLS = (
            "abs_path, processing_time, total_duration, "
            "duration_sec, processing_secs, model, language, "
            "vocal_isolation, vocal_model, "
            "speakers_json, segment_count, total_chunks, "
            "updated_at, word_count, char_count, speaker_count, "
            "warning_count, step_timings, app_version"
        )
        with self.connection() as conn:
            if abs_paths is not None:
                keys = [k for k in abs_paths if k]
                # Batch query with an IN clause (SQLite allows at most 999 variables)
                result = {}
                for i in range(0, len(keys), 900):
                    batch = keys[i : i + 900]
                    placeholders = ",".join(["?"] * len(batch))
                    rows = conn.execute(
                        f"""
                        SELECT {_META_COLS} FROM transcripts
                        WHERE abs_path != '' AND abs_path IN ({placeholders})
                          AND complete = 1
                    """,
                        batch,
                    ).fetchall()
                    for row in rows:
                        result[row["abs_path"]] = self._row_to_scan_meta(row)
                return result
            else:
                rows = conn.execute(
                    f"SELECT {_META_COLS} FROM transcripts WHERE complete = 1 AND abs_path != ''"
                ).fetchall()
                return {row["abs_path"]: self._row_to_scan_meta(row) for row in rows}

    def _row_to_scan_meta(self, row) -> dict:
        """Convert a DB row into scan metadata (for batch queries)."""
        return {
            "processing_time": row["processing_time"],
            "audio_duration": row["total_duration"],
            "audio_duration_sec": row["duration_sec"],
            "processing_secs": row["processing_secs"],
            "tx_model": row["model"],
            "tx_lang": row["language"],
            "tx_vocal_iso": bool(row["vocal_isolation"]),
            "tx_vocal_model": row["vocal_model"],
            "tx_speakers": _json_or(row["speakers_json"], []),
            "tx_seg_count": row["segment_count"],
            "tx_chunks": row["total_chunks"],
            "tx_date": row["updated_at"],
            "tx_word_count": row["word_count"],
            "tx_char_count": row["char_count"],
            "tx_speaker_count": row["speaker_count"],
            "tx_warning_count": row["warning_count"],
            "tx_app_version": row["app_version"] or "",
            # For the hover over the processing time (exact per-step times)
            "tx_step_timings": _json_or(row["step_timings"], []),
        }

    # ── FTS5 search ────────────────────────────────────────────────────

    def _fts_from_where(
        self,
        conn,
        query: str,
        file_filter: str,
        speaker_filter: str,
        fuzzy: float,
        file_exact: bool = False,
        filter_opts: dict | None = None,
    ):
        """Builds the shared FROM/WHERE clause for search AND facets.

        filter_opts (all optional, None = no filter):
            duration_min/duration_max      file duration in seconds (t.duration_sec)
            word_count_min/word_count_max  transcript word count (t.word_count)
            tempo_min/tempo_max      processing speed as real-time factor
                                     (duration_sec / processing_secs)
            date_from/date_to      recording date as unix time
                                     (media_files.modified_at — the file
                                     date, not the transcription date)

        Returns:
            (from_where, params, match_terms), or None if the query is
            empty after cleanup (then there are by definition 0 matches).
        """
        # Build the FTS5 query. When fuzzy search is active, first try
        # the variant with similarity alternatives; the result is again
        # a normal FTS5 query, so the search run stays fast.
        fts_query, match_terms = "", []
        if fuzzy > 0:
            try:
                expanded = self._fuzzy_fts_query(conn, query, fuzzy)
                if expanded:
                    fts_query, match_terms = expanded
            except Exception as e:  # noqa — fuzzy is a convenience
                # The search must also work if the
                # expansion fails: better exact than nothing.
                logger.warning("Fuzzy expansion failed: %s", e)
                fts_query, match_terms = "", []
        if not fts_query:
            fts_query = self._build_fts_query(query)
        if not fts_query:
            return None

        # media_files always included via LEFT JOIN: provides the recording date
        # (date filter/sort) via the abs_path identity. LEFT, so
        # legacy rows without a media_files entry don't drop out of the search —
        # only the date filter inevitably excludes them.
        from_where = """
            FROM segments_fts
            JOIN segments s ON s.id = segments_fts.rowid
            JOIN transcripts t ON t.id = s.transcript_id
            LEFT JOIN media_files m ON m.abs_path = t.abs_path
            WHERE segments_fts MATCH ?
              AND t.complete = 1
        """
        params: list = [fts_query]

        fo = filter_opts or {}
        for condition, value in (
            (" AND t.duration_sec >= ?", fo.get("duration_min")),
            (" AND t.duration_sec <= ?", fo.get("duration_max")),
            (" AND t.word_count >= ?", fo.get("word_count_min")),
            (" AND t.word_count <= ?", fo.get("word_count_max")),
            (" AND m.modified_at >= ?", fo.get("date_from")),
            (" AND m.modified_at <= ?", fo.get("date_to")),
            # Real-time factor only where processing time is recorded — rows
            # without processing_secs are deliberately excluded by the speed filter.
            (
                " AND t.processing_secs > 0 AND t.duration_sec / t.processing_secs >= ?",
                fo.get("tempo_min"),
            ),
            (
                " AND t.processing_secs > 0 AND t.duration_sec / t.processing_secs <= ?",
                fo.get("tempo_max"),
            ),
        ):
            if value is not None:
                from_where += condition
                params.append(value)

        if file_filter:
            # file_exact for the hit card's drill-down: there the
            # path is known and must match EXACTLY. With LIKE %...% ,
            # "a/clip.mp4" would also count "b/a/clip.mp4" and the card would show
            # more matches than the file has.
            if file_exact:
                # The drill-down knows the exact path. Via the identity
                # if it's absolute — otherwise the card would also count matches
                # of an identically named file from another root
                # (docs/PATHS.md). A relative value remains allowed as a fallback for
                # legacy rows without abs_path.
                if paths.is_absolute(file_filter):
                    from_where += " AND t.abs_path = ?"
                    params.append(paths.canon(file_filter))
                else:
                    from_where += " AND t.rel_path = ?"
                    params.append(file_filter)
            elif paths.is_absolute(file_filter):
                # A full path — the MCP server names a recording outside
                # the open folder that way: that file, or everything
                # below that folder
                full = paths.canon(file_filter)
                from_where += " AND (t.abs_path = ? OR t.abs_path LIKE ? || '%' ESCAPE '\\')"
                params += [full, paths.like_prefix(full)]
            else:
                # % and _ are allowed in paths and must not act as
                # wildcards.
                from_where += " AND t.rel_path LIKE ? ESCAPE '\\'"
                params.append(
                    "%"
                    + (file_filter.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))
                    + "%"
                )

        if speaker_filter:
            from_where += " AND s.speaker = ?"
            params.append(speaker_filter)

        # ── Restrict to a PERSON (speaker recognition) ──────────────
        #
        # A text search can be limited to the recordings of a selected
        # speaker. The entries come from
        # `speaker.db` (a different database, hence as a parameter and
        # not via JOIN) and are given as (recording, id)-pairs.
        #
        # Via a TEMP table instead of `IN (?,?,...)`: a person can appear in
        # hundreds or thousands of recordings and would otherwise hit the
        # SQLite parameter limit. The table lives on this connection
        # and is refilled on every call.
        spots = fo.get("person_spots")
        if spots is not None:
            conn.execute("DROP TABLE IF EXISTS temp._person_places")
            conn.execute("CREATE TEMP TABLE _person_places (abs_path TEXT, label TEXT)")
            if spots:
                conn.executemany(
                    "INSERT INTO _person_places VALUES (?, ?)",
                    [(paths.canon(a), k) for a, k in spots],
                )
                conn.execute("CREATE INDEX temp._ps_path ON _person_places(abs_path)")
            if fo.get("person_segments_only"):
                # Strict: only the sentences assigned to this person.
                from_where += (
                    " AND EXISTS (SELECT 1 FROM _person_places ps "
                    "WHERE ps.abs_path = t.abs_path "
                    "AND ps.label = s.speaker)"
                )
            else:
                # Loose: "the recordings this person appears in" — there
                # every word counts, not just theirs.
                from_where += " AND t.abs_path IN (SELECT abs_path FROM _person_places)"

        return from_where, params, match_terms

    def file_search(
        self,
        file_filter: str = "",
        speaker_filter: str = "",
        filter_opts: dict | None = None,
        sort: str = "",
        sort_dir: str = "desc",
        limit: int = 500,
        offset: int = 0,
    ) -> dict:
        """File search WITHOUT full text: across the ENTIRE library.

        Basis is media_files (every known file, including untranscribed ones —
        recordings that are not transcribed yet must not be missing).
        Transcript metadata (word count, speed) is joined via LEFT JOIN;
        filters on it implicitly require a transcript.
        filter_opts, in addition to _fts_from_where:

            transcribed_only  True -> only files with a completed transcript
        """
        fo = filter_opts or {}
        where = " WHERE 1=1"
        params: list = []
        if fo.get("transcribed_only"):
            where += " AND t.id IS NOT NULL"
        if file_filter:
            cond, args = self._name_filter_sql(file_filter, "m.rel_path", ("m.media_title",))
            where += " AND " + cond
            params += args
        if speaker_filter:
            where += (
                " AND t.id IS NOT NULL AND EXISTS("
                "SELECT 1 FROM segments s "
                "WHERE s.transcript_id = t.id AND s.speaker = ?)"
            )
            params.append(speaker_filter)
        for condition, value in (
            (" AND m.duration_sec >= ?", fo.get("duration_min")),
            (" AND m.duration_sec <= ?", fo.get("duration_max")),
            (" AND t.word_count >= ?", fo.get("word_count_min")),
            (" AND t.word_count <= ?", fo.get("word_count_max")),
            (" AND m.modified_at >= ?", fo.get("date_from")),
            (" AND m.modified_at <= ?", fo.get("date_to")),
            (
                " AND t.processing_secs > 0 AND t.duration_sec / t.processing_secs >= ?",
                fo.get("tempo_min"),
            ),
            (
                " AND t.processing_secs > 0 AND t.duration_sec / t.processing_secs <= ?",
                fo.get("tempo_max"),
            ),
        ):
            if value is not None:
                where += condition
                params.append(value)

        _direction = "ASC" if sort_dir in ("auf", "asc") else "DESC"
        order_by = {
            "date": f"m.modified_at {_direction}",
            "duration": f"m.duration_sec {_direction}",
            "size": f"m.file_size {_direction}",
            "word_count": f"t.word_count {_direction}",
            "speed": (
                "(CASE WHEN t.processing_secs > 0 THEN "
                "t.duration_sec / t.processing_secs END) " + _direction
            ),
        }.get(sort, "m.rel_path ASC")
        if order_by != "m.rel_path ASC":
            order_by += ", m.rel_path ASC"

        basis = (
            " FROM media_files m "
            "LEFT JOIN transcripts t ON t.abs_path = m.abs_path "
            "AND m.abs_path != '' AND t.complete = 1" + where
        )
        with self.connection() as conn:
            total = conn.execute("SELECT COUNT(*) AS c" + basis, params).fetchone()["c"]
            rows = conn.execute(
                "SELECT m.rel_path, m.abs_path AS m_abs, m.duration_sec, m.file_size, m.is_video, "
                "m.modified_at AS file_date, "
                "t.media_dir, t.word_count, t.processing_secs, "
                "t.segment_count, (t.id IS NOT NULL) AS has_transcript"
                + basis
                + f" ORDER BY {order_by} LIMIT ? OFFSET ?",
                params + [max(1, min(int(limit or 500), 5000)), max(0, int(offset or 0))],
            ).fetchall()
        return {
            "mode": "files",
            "total": total,
            "files": total,
            "offset": offset,
            "limit": limit,
            "truncated": total > (offset + len(rows)),
            "results": [
                {
                    "filename": r["rel_path"],
                    "media_dir": _root_of(r["rel_path"], r["m_abs"], r["media_dir"]),
                    "file_duration": r["duration_sec"],
                    "file_size": r["file_size"],
                    "file_word_count": r["word_count"],
                    "file_date": r["file_date"],
                    "file_segments": r["segment_count"],
                    "has_transcript": bool(r["has_transcript"]),
                    "file_rate": (
                        round((r["duration_sec"] or 0) / r["processing_secs"], 1)
                        if r["processing_secs"]
                        else None
                    ),
                }
                for r in rows
            ],
        }

    def search_media_files(
        self, query: str, limit: int = 50, filter_opts: dict | None = None
    ) -> list[dict]:
        """Search files by path AND embedded title.

        Complements full-text search over the transcripts: here the search is
        in the filename and the ``media_title`` from the FFprobe metadata. This
        finds things that don't occur in the spoken word — the name
        of a series, a recording date, the title of a downloaded
        video.

        Matches in the title come first: it is the more deliberate label than
        a filename like ``LmNoPqRsTuV.mp4``.
        """
        if not query or len(query) < 2:
            return []
        # % and _ are allowed in names and must not act as wildcards
        # (docs/PATHS.md, same rule as like_prefix).
        search_term = (
            "%" + (query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")) + "%"
        )
        limit = max(1, min(int(limit or 50), 500))
        # The search view's metadata filters also apply here — otherwise
        # the name list would show 30-min files while "duration max 5 min"
        # is set on the right, and the filter would look broken. Duration/date come
        # from media_files; word count/speed need the transcript (rows
        # without a transcript inevitably drop out under these filters).
        fo = filter_opts or {}
        extra = ""
        params: list = [search_term, search_term]
        for condition, value in (
            (" AND m.duration_sec >= ?", fo.get("duration_min")),
            (" AND m.duration_sec <= ?", fo.get("duration_max")),
            (" AND m.modified_at >= ?", fo.get("date_from")),
            (" AND m.modified_at <= ?", fo.get("date_to")),
            (" AND t.word_count >= ?", fo.get("word_count_min")),
            (" AND t.word_count <= ?", fo.get("word_count_max")),
            (
                " AND t.processing_secs > 0 AND t.duration_sec / t.processing_secs >= ?",
                fo.get("tempo_min"),
            ),
            (
                " AND t.processing_secs > 0 AND t.duration_sec / t.processing_secs <= ?",
                fo.get("tempo_max"),
            ),
        ):
            if value is not None:
                extra += condition
                params.append(value)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT m.rel_path, m.abs_path, m.display_name, m.media_title,
                       m.duration_sec, m.status, m.is_video, m.file_size,
                       (t.id IS NOT NULL) AS has_transcript
                  FROM media_files m
                  LEFT JOIN transcripts t ON t.abs_path = m.abs_path
                       AND m.abs_path != '' AND t.complete = 1
                 WHERE (m.rel_path LIKE ? ESCAPE '\\'
                    OR m.media_title LIKE ? ESCAPE '\\'){extra}
                 ORDER BY (m.media_title LIKE ? ESCAPE '\\') DESC,
                          m.duration_sec DESC
                 LIMIT ?
            """,
                params + [search_term, limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def search_facets(
        self,
        query: str,
        file_filter: str = "",
        speaker_filter: str = "",
        fuzzy: float = 0.0,
        buckets: int = 40,
        max_files: int = 500,
        filter_opts: dict | None = None,
    ) -> dict:
        """Aggregated hit distribution for the large search view.

        Delivers — across ALL matches, not just the current result page —
        the hit count per file and a time histogram (`buckets` buckets over
        the file's runtime). This lets the search view show WHERE in a file
        a word occurs, without having to load all matches.

        Args:
            buckets: histogram resolution per file.
            max_files: cap for the file list (sorted by hit count).
        """
        buckets = max(1, min(int(buckets or 40), 200))
        max_files = max(1, min(int(max_files or 500), 5000))

        with self.connection() as conn:
            built = self._fts_from_where(
                conn, query, file_filter, speaker_filter, fuzzy, filter_opts=filter_opts
            )
            if built is None:
                return {
                    "files": [],
                    "speakers": [],
                    "total": 0,
                    "file_count": 0,
                    "buckets": buckets,
                    "files_truncated": False,
                }
            from_where, params, match_terms = built

            # One pass: count per (file, time bucket). The bucket is computed in
            # SQL so not every single match has to travel to Python —
            # for 10k+ matches that's the difference between
            # "instant" and "noticeable".
            bucket_expr = "CAST(MIN(? - 1, s.start_sec * ? / MAX(t.duration_sec, 1)) AS INTEGER)"
            # Per file by its identity: the rel_path is relative to the
            # folder open at transcription, and two recordings of the same
            # name in two folders opened that way were counted as one
            rows = conn.execute(
                f"SELECT COALESCE(NULLIF(t.abs_path, ''), t.rel_path) AS ident, "
                f"       MAX(t.rel_path) AS rel_path, "
                f"       MAX(t.duration_sec) AS duration_sec, "
                f"       MAX(t.media_dir) AS media_dir, "
                f"       MAX(t.abs_path) AS abs_path, "
                f"       {bucket_expr} AS bucket, "
                f"       COUNT(*) AS cnt " + from_where + " GROUP BY ident, bucket",
                [buckets, buckets] + params,
            ).fetchall()

            per_file: dict = {}
            total = 0
            for row in rows:
                rel = row["rel_path"]
                entry = per_file.get(row["ident"])
                if entry is None:
                    # Do NOT pre-create buckets: for a broad search across
                    # tens of thousands of files that would be millions of values, of which
                    # almost all would be discarded by the cap below anyway.
                    entry = per_file[row["ident"]] = {
                        "filename": rel,
                        "media_dir": _root_of(rel, row["abs_path"], row["media_dir"]),
                        "duration": row["duration_sec"] or 0,
                        "hits": 0,
                        "_bucket_counts": {},
                    }
                cnt = row["cnt"]
                entry["hits"] += cnt
                total += cnt
                b = row["bucket"]
                if b is None:
                    b = 0
                idx = max(0, min(buckets - 1, int(b)))
                entry["_bucket_counts"][idx] = entry["_bucket_counts"].get(idx, 0) + cnt

            files = sorted(per_file.values(), key=lambda f: (-f["hits"], f["filename"]))
            file_count = len(files)
            # Only now, for the files actually being returned, build the
            # complete histogram.
            for f in files[:max_files]:
                bucket = f.pop("_bucket_counts", {})
                hist = [0] * buckets
                for i, v in bucket.items():
                    hist[i] = v
                f["histogram"] = hist

            out = {
                "files": files[:max_files],
                "files_truncated": file_count > max_files,
                "file_count": file_count,
                "total": total,
                "buckets": buckets,
            }
            if match_terms:
                out["terms"] = match_terms
            return out

    def search(
        self,
        query: str,
        file_filter: str = "",
        speaker_filter: str = "",
        limit: int = 500,
        offset: int = 0,
        context: int = 0,
        fuzzy: float = 0.0,
        file_exact: bool = False,
        filter_opts: dict | None = None,
        sort: str = "",
        sort_dir: str = "desc",
    ) -> dict:
        """Full-text search over all segments via FTS5.

        Args:
            query: search term(s). Supports:
                   - simple words: hello world
                   - phrases: "exact phrase"
                   - boolean: word1 AND word2, word1 OR word2, NOT word
            file_filter: filename filter (LIKE pattern)
            speaker_filter: speaker filter (exact)
            limit: max results
            offset: offset for pagination
            context: how many segments before/after the match to include
                (0 = none). Ends up as context_before/context_after
                on the respective result.
            fuzzy: similarity threshold 0.0-1.0 for fuzzy search
                (0 = off, higher = stricter).
        Returns:
            Dict with results, total, files, truncated
        """
        with self.connection() as conn:
            # Shared FROM/WHERE clause for count AND data. The count deliberately
            # runs WITHOUT highlight() and without the selected extra columns
            # (#066: otherwise the expensive highlighting is computed for ALL matches
            # just to count them).
            built = self._fts_from_where(
                conn, query, file_filter, speaker_filter, fuzzy, file_exact, filter_opts=filter_opts
            )
            if built is None:
                return {"results": [], "total": 0, "files": 0, "truncated": False}
            from_where, params, match_terms = built

            # Count total — lean, without highlight()/extra columns.
            #
            # The file count comes from the SAME query, not from the
            # returned rows. Previously `files` only counted the files
            # of the loaded PAGE: with `limit=1` the search reported "1 file"
            # — even with hundreds of thousands of matches in over a thousand
            # files. The frontend worked around it via the facets
            # (`svSetStatus`), every other caller got the wrong number. The
            # batch split relies on it: this number is the basis of the
            # entire planning.
            line = conn.execute(
                "SELECT COUNT(*) AS cnt, COUNT(DISTINCT s.transcript_id) AS files " + from_where,
                params,
            ).fetchone()
            total = line["cnt"]
            files_total = line["files"]

            # Sort order: the default stays file + time (stable and
            # expected for the drill-down). The additional sort options run over the
            # file metadata; within a file, chronological order is kept.
            _direction = "ASC" if sort_dir in ("auf", "asc") else "DESC"
            _sort = {
                "date": f"m.modified_at {_direction}",
                "duration": f"t.duration_sec {_direction}",
                "word_count": f"t.word_count {_direction}",
                "speed": (
                    "(CASE WHEN t.processing_secs > 0 THEN "
                    "t.duration_sec / t.processing_secs END) " + _direction
                ),
            }.get(sort)
            order_by = (_sort + ", t.rel_path, s.start_sec") if _sort else "t.rel_path, s.start_sec"

            # Paginated results with highlighting
            data_sql = (
                "SELECT s.id, s.speaker, s.start_sec, s.end_sec, "
                "s.start_fmt, s.end_fmt, s.text, "
                "s.transcript_id, s.seg_index, "
                "t.rel_path, t.speaker_names, t.media_dir, t.abs_path, "
                "t.duration_sec AS file_duration, "
                "t.word_count AS file_word_count, "
                "t.processing_secs AS file_processing, "
                "m.modified_at AS file_date, "
                # Markers as control characters, NOT as <mark>: the frontend
                # escapes the text first and only then swaps the markers for
                # HTML. If finished HTML came out of the DB, it would have to be
                # inserted unchecked — a <script> in the transcript would then be a
                # hole. \x02/\x03 don't occur in transcript text.
                "highlight(segments_fts, 0, char(2), char(3)) AS highlighted"
                + from_where
                + f" ORDER BY {order_by} LIMIT ? OFFSET ?"
            )
            rows = conn.execute(data_sql, params + [limit, offset]).fetchall()

            ctx_map = self._fetch_context(conn, rows, context) if context > 0 else {}

            results = []
            file_set = set()
            for row in rows:
                # #069: a single corrupt speaker_names JSON must not abort the
                # entire search → parse defensively per row.
                sp_names = _json_or(row["speaker_names"], {})
                if not isinstance(sp_names, dict):
                    sp_names = {}
                sp_display = sp_names.get(row["speaker"], row["speaker"])
                file_set.add(row["rel_path"])
                entry = {
                    "filename": row["rel_path"],
                    "media_dir": _root_of(row["rel_path"], row["abs_path"], row["media_dir"]),
                    "speaker": row["speaker"],
                    "speaker_name": sp_display,
                    "start": row["start_sec"],
                    "end": row["end_sec"],
                    "start_fmt": row["start_fmt"],
                    "end_fmt": row["end_fmt"],
                    "text": row["text"],
                    "highlighted": row["highlighted"],
                    # File metadata for filter badges and sort display
                    "file_duration": row["file_duration"],
                    "file_word_count": row["file_word_count"],
                    "file_date": row["file_date"],
                    "file_rate": (
                        round(row["file_duration"] / row["file_processing"], 1)
                        if row["file_processing"]
                        else None
                    ),
                }
                if context > 0:
                    before, after = ctx_map.get((row["transcript_id"], row["seg_index"]), ([], []))
                    entry["context_before"] = before
                    entry["context_after"] = after
                results.append(entry)

            out = {
                "query": query,
                "results": results,
                "total": total,
                # All files with hits, not just those on this page.
                "files": files_total,
                "files_on_page": len(file_set),
                "truncated": total > limit + offset,
                "offset": offset,
                "limit": limit,
            }
            # Only include for fuzzy: otherwise the highlighted words would
            # deviate from what the user typed. Without the key,
            # the frontend continues to highlight the input itself.
            if match_terms:
                out["terms"] = match_terms
            return out

    # Fuzzy search: a typo ("Machnie") should still find the
    # match. FTS5 can't do that — it matches words exactly. Instead of
    # comparing every segment individually (hopeless with hundreds of
    # thousands of segments), the
    # index's VOCABULARY is searched: similar words are looked up and
    # appended to the FTS query as OR alternatives. The actual search run
    # thus remains a normal, fast FTS match.
    _FUZZY_MAX_ALTERNATIVES = 8  # per search word, otherwise the query becomes absurd
    _FUZZY_MIN_LEN = 4  # shorter words are too ambiguous

    @staticmethod
    def _edit_distance_within(a: str, b: str, max_d: int) -> int:
        """Levenshtein distance, but only up to ``max_d`` — beyond that, max_d + 1.

        Deliberately NOT difflib: its ``ratio()`` counts common subsequences
        and rates scattered random matches far too generously: two unrelated
        five-letter words came out at 0.8 that way (three changes!), and a
        nonsense word reached exactly 0.800 against the diacritic-free index
        form of a very common word — thousands of nonsensical matches. Edit
        distance separates cleanly, all usable matches sit at exactly one change.

        The cutoff at max_d keeps this fast: the loop aborts as soon as
        the whole row is above the bound.
        """
        if abs(len(a) - len(b)) > max_d:
            return max_d + 1
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(
                    min(
                        prev[j] + 1,  # delete
                        cur[j - 1] + 1,  # insert
                        prev[j - 1] + (ca != cb),
                    )
                )  # substitute
            if min(cur) > max_d:
                return max_d + 1
            prev = cur
        return prev[-1]

    def _fuzzy_terms(self, conn, word: str, threshold: float) -> list[str]:
        """Words similar to ``word`` from the FTS vocabulary.

        Pre-filter in SQL (same starting letter, length ±1), only then
        the expensive comparison in Python. Without the pre-filter, the
        distance computation would have to run over the entire vocabulary.

        ``threshold`` remains a tolerance of 0-1, but now controls how many
        changes are allowed — relative to word length: at 0.8 that's one
        change, for a long word two. Higher = stricter.
        """
        w = word.lower()
        if len(w) < self._FUZZY_MIN_LEN:
            return []
        try:
            rows = conn.execute(
                "SELECT term FROM segments_fts_vocab "
                "WHERE term >= ? AND term < ? "
                "  AND length(term) BETWEEN ? AND ?",
                # Length difference at most 1. At ±2, short,
                # very common words slipped through: a nonsense word matched
                # the de-diacriticized index entry of a common word with a
                # similarity of exactly 0.800 — thousands of nonsensical matches.
                # Real typos rarely change the length by more than one;
                # all usable matches (machnie->machine,
                # compter->computer, videio->video) fall at <= 1.
                (w[0], w[0] + "￿", len(w) - 1, len(w) + 1),
            ).fetchall()
        except Exception as e:  # vocabulary table missing (old DB)
            logger.debug("Fuzzy search not possible: %s", e)
            return []

        # Derive allowed changes from the tolerance. At least 1 (otherwise
        # fuzzy would be ineffective), at most 2 — from three changes on it's no
        # longer a typo but a different word.
        #
        # Rounded down, not to the nearest: at 0.8 and eight characters, rounding
        # would already give 2, and an eight-letter word would then also catch
        # unrelated words two letters away. This way two changes are only allowed
        # from ten characters on,
        # where they actually still pass as a typo.
        # The epsilon is not cosmetic: (1.0 - 0.8) * 10 comes out in
        # floating point as 1.9999999999999996 and would round down to 1 — the
        # bound would kick in one word length too late.
        max_d = max(1, min(2, int((1.0 - threshold) * len(w) + 1e-9)))

        scored = []
        for r in rows:
            term = r["term"]
            if term == w:
                continue  # the original is already in the query anyway
            d = self._edit_distance_within(w, term, max_d)
            if d <= max_d:
                # Sort by distance, then by word length on ties:
                # similarly long words are the more plausible typos.
                scored.append((d, abs(len(term) - len(w)), term))
        scored.sort()
        return [t for _, _, t in scored[: self._FUZZY_MAX_ALTERNATIVES]]

    def _fuzzy_fts_query(self, conn, query: str, threshold: float) -> tuple[str, list[str]] | None:
        """Finished FTS5 query with similarity alternatives — or None.

        Also returns the words that were actually matched on.
        The frontend needs this for highlighting: the text has "Machine",
        but "Machnie" was typed — without this list the find would stay unmarked.

        Deliberately builds the query itself instead of passing it through
        _build_fts_query: that sanitizes special characters and would remove the
        parentheses of the OR groups.

        None means "not applicable here, use the normal query". This applies
        to quotation marks and boolean operators: there the user has
        phrased things precisely, and an automatic loosening would break more
        than it helps.
        """
        import re as _re

        if '"' in query:
            return None
        clean = _re.sub(r"[^\w\s]", " ", query, flags=_re.UNICODE)
        words = [w for w in clean.split() if w]
        if not words:
            return None
        if any(w.upper() in ("AND", "OR", "NOT") for w in words):
            return None

        parts, terms, found_any = [], [], False
        for i, w in enumerate(words):
            # Last word still treated as a prefix — as in the normal query,
            # so typing in the search field feels unchanged.
            base = f"{w}*" if i == len(words) - 1 else w
            olds = self._fuzzy_terms(conn, w, threshold)
            terms.append(w)
            if olds:
                found_any = True
                terms.extend(olds)
                parts.append("(" + " OR ".join([base, *olds]) + ")")
            else:
                parts.append(base)
        # Join with an EXPLICIT "AND", not just a plain space.
        #
        # FTS5 reads a space between two simple terms as
        # implicit AND — but not next to a parenthesized group:
        #
        #     have you*                   -> matches
        #     (have OR has) you*          -> fts5: syntax error near "you"
        #     (have OR has) AND you*      -> matches
        #
        # With fuzzy search enabled (search.fuzzy_enabled), such a group
        # arose for EVERY multi-word input, and the entire search
        # responded with "Invalid search syntax". Single words worked because
        # no second group followed there — which made it look like the
        # search simply couldn't handle multiple words.
        # Nothing similar found -> normal query, saves a special case
        return (" AND ".join(parts), terms) if found_any else None

    @staticmethod
    def _fetch_context(conn, rows, context: int) -> dict:
        """Fetch surrounding segments for all matches (search.context_segments).

        Deliberately batched: one query per affected transcript instead of one
        per match. For 500 matches across 20 files, that's 20 queries
        against the primary key instead of 500 — for a DB of this
        size, the difference is between "unnoticeable" and "the search hangs".

        Returns:
            {(transcript_id, seg_index): (before, after)} — both lists containing
            dicts {speaker, text, start_fmt}.
        """
        # Collect, per transcript, which segment indices are needed.
        needed: dict[int, set[int]] = {}
        for row in rows:
            tid, idx = row["transcript_id"], row["seg_index"]
            if idx is None:
                continue
            want = needed.setdefault(tid, set())
            for d in range(-context, context + 1):
                if d != 0 and idx + d >= 0:
                    want.add(idx + d)

        # Store fetched segments by (transcript_id, seg_index).
        segs: dict[tuple, dict] = {}
        for tid, indices in needed.items():
            if not indices:
                continue
            ph = ",".join("?" * len(indices))
            for s in conn.execute(
                f"SELECT seg_index, speaker, text, start_fmt FROM segments "
                f"WHERE transcript_id = ? AND seg_index IN ({ph})",
                [tid, *indices],
            ):
                segs[(tid, s["seg_index"])] = {
                    "speaker": s["speaker"],
                    "text": s["text"],
                    "start_fmt": s["start_fmt"],
                }

        out: dict[tuple, tuple] = {}
        for row in rows:
            tid, idx = row["transcript_id"], row["seg_index"]
            if idx is None:
                continue
            before = [segs[(tid, idx + d)] for d in range(-context, 0) if (tid, idx + d) in segs]
            after = [segs[(tid, idx + d)] for d in range(1, context + 1) if (tid, idx + d) in segs]
            out[(tid, idx)] = (before, after)
        return out

    def _build_fts_query(self, query: str) -> str:
        """Convert a user query into FTS5 syntax. Sanitizes special characters."""
        import re as _re

        query = query.strip()
        if not query:
            return ""

        # Quoted phrases: check whether quotation marks are balanced
        if '"' in query:
            parts = query.split('"')
            if len(parts) % 2 == 0:
                # Odd number of parts = unbalanced -> remove quotation marks
                query = query.replace('"', " ")
            else:
                # Balanced: sanitize unquoted parts, keep quoted ones
                result_parts = []
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        # Outside quotation marks: remove special characters
                        clean = _re.sub(r"[^\w\s]", " ", part, flags=_re.UNICODE)
                        words = [w for w in clean.split() if w]
                        if words:
                            result_parts.append(" ".join(words))
                    else:
                        # Inside quotation marks: pass through as a phrase
                        if part.strip():
                            result_parts.append(f'"{part}"')
                result = " ".join(result_parts)
                return result if result.strip() else ""

        # No quotation marks: clean up FTS5 special characters
        # Allowed characters: letters, numbers, whitespace (Unicode words)
        clean = _re.sub(r"[^\w\s]", " ", query, flags=_re.UNICODE)
        words = [w for w in clean.split() if w]
        if not words:
            return ""

        if len(words) == 1:
            # Single word: prefix search
            return f"{words[0]}*"

        # Multiple words: AND combination with a prefix on the last word
        parts = list(words[:-1])
        parts.append(f"{words[-1]}*")
        return " ".join(parts)

    # ── Statistics ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Global statistics from the DB."""
        with self.connection() as conn:
            stats = {}
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_files,
                    COALESCE(SUM(CASE WHEN status='done' THEN 1 ELSE 0 END), 0) as done,
                    COALESCE(SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END), 0) as pending,
                    COALESCE(SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END), 0) as skipped,
                    COALESCE(SUM(file_size), 0) as total_size
                FROM media_files
            """).fetchone()
            stats["files"] = dict(row) if row else {}

            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    COALESCE(SUM(segment_count), 0) as total_segments,
                    COALESCE(SUM(word_count), 0) as total_words,
                    COALESCE(SUM(char_count), 0) as total_chars,
                    COALESCE(SUM(duration_sec), 0) as total_duration,
                    COALESCE(SUM(processing_secs), 0) as total_processing,
                    COALESCE(AVG(processing_secs), 0) as avg_processing
                FROM transcripts WHERE complete = 1
            """).fetchone()
            stats["transcripts"] = dict(row) if row else {}

            # Top models
            rows = conn.execute("""
                SELECT model, COUNT(*) as cnt
                FROM transcripts WHERE complete = 1 AND model != ''
                GROUP BY model ORDER BY cnt DESC LIMIT 10
            """).fetchall()
            stats["top_models"] = [dict(r) for r in rows]

            # Top languages
            rows = conn.execute("""
                SELECT language, COUNT(*) as cnt
                FROM transcripts WHERE complete = 1 AND language != ''
                GROUP BY language ORDER BY cnt DESC LIMIT 10
            """).fetchall()
            stats["top_languages"] = [dict(r) for r in rows]

            return stats

    # ── Media file cache ─────────────────────────────────────────────

    def upsert_media_file(self, rel_path: str, **kwargs) -> None:
        """Insert/update a media file in the cache.

        Without ``abs_path`` in kwargs, the identity constraint doesn't apply
        (the conflict runs via abs_path, docs/PATHS.md) and duplicates would
        result — so it's then filled in from the existing row or the current
        root.
        """
        with self.connection() as conn:
            if not kwargs.get("abs_path"):
                row = conn.execute(
                    "SELECT abs_path FROM media_files WHERE rel_path = ? "
                    "AND abs_path != '' LIMIT 1",
                    (rel_path,),
                ).fetchone()
                if row:
                    kwargs["abs_path"] = row["abs_path"]
                else:
                    try:
                        from whisper_tm.config import config as _cfg

                        kwargs["abs_path"] = paths.abs_from(rel_path, str(_cfg.media_dir))
                    except Exception as _e:  # noqa
                        pass
            fields = ["rel_path"]
            values = [rel_path]
            updates = ["rel_path=excluded.rel_path"]

            for key, val in kwargs.items():
                # #061: only allow known columns — otherwise an
                # arbitrary kwargs key would end up as a column name directly in SQL.
                if key not in self._MEDIA_FILE_COLS:
                    logger.warning("upsert_media_file: unknown column ignored: %s", key)
                    continue
                fields.append(key)
                values.append(val)
                updates.append(f"{key}=excluded.{key}")

            fields.append("last_scanned")
            values.append(time.time())
            updates.append("last_scanned=excluded.last_scanned")
            updates.append("updated_at=unixepoch()")

            placeholders = ",".join(["?"] * len(values))
            field_str = ",".join(fields)
            update_str = ",".join(updates)

            conn.execute(
                f"""
                INSERT INTO media_files ({field_str})
                VALUES ({placeholders})
                ON CONFLICT(abs_path) WHERE abs_path != '' DO UPDATE SET {update_str}
            """,
                values,
            )

    def bulk_upsert_media(self, files: list[dict]) -> None:
        """Batch update for media files (fast scan)."""
        with self.connection() as conn:
            now = time.time()
            # Heal legacy rows without abs_path first. Without this, the
            # conflict below doesn't apply (the partial identity excludes '')
            # and the scan creates a SECOND row instead of updating the
            # existing one — the old row's indexing data would be orphaned.
            # Only heal when EXACTLY ONE legacy row carries this display value.
            # Otherwise multiple rows would get the same identity — the partial
            # UNIQUE index would then break the entire scan batch.
            conn.executemany(
                "UPDATE media_files SET abs_path = ? WHERE rowid = ("
                "  SELECT rowid FROM media_files"
                "   WHERE rel_path = ? AND (abs_path IS NULL OR abs_path = '')"
                "   LIMIT 2 OFFSET 0)"
                " AND (SELECT COUNT(*) FROM media_files"
                "       WHERE rel_path = ? AND (abs_path IS NULL OR abs_path = '')) = 1",
                [
                    (f.get("abs_path", ""), f["rel_path"], f["rel_path"])
                    for f in files
                    if f.get("abs_path")
                ],
            )
            conn.executemany(
                """
                INSERT INTO media_files (
                    rel_path, abs_path, display_name, subfolder, extension,
                    is_video, file_size, modified_at, status,
                    last_scanned, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
                ON CONFLICT(abs_path) WHERE abs_path != '' DO UPDATE SET
                    rel_path=excluded.rel_path,
                    display_name=excluded.display_name,
                    subfolder=excluded.subfolder,
                    extension=excluded.extension,
                    is_video=excluded.is_video,
                    file_size=excluded.file_size,
                    modified_at=excluded.modified_at,
                    last_scanned=excluded.last_scanned,
                    -- The walk SAW the file, so it is no orphan any more.
                    -- Every other status (done, skipped, error ...) is kept:
                    -- those aren't statements about existence.
                    status=CASE WHEN media_files.status = 'orphan'
                                THEN excluded.status ELSE media_files.status END,
                    updated_at=unixepoch()
            """,
                [
                    (
                        f["rel_path"],
                        f.get("abs_path", ""),
                        f.get("display_name", ""),
                        f.get("subfolder", ""),
                        f.get("extension", ""),
                        int(f.get("is_video", False)),
                        f.get("file_size", 0),
                        f.get("modified_at", 0),
                        f.get("status", "pending"),
                        now,
                    )
                    for f in files
                ],
            )

    def clean_orphaned_media(self, root, seen_abs: set[str], max_share: float = 0.2) -> list[str]:
        """After a COMPLETE walk: delete rows of files that have disappeared.

        Previously, rows of deleted/renamed files stayed "pending"
        in the DB forever — every "queue all pending" batch pulled
        them into the queue again, where they failed as "not found"
        (e.g. every path under a share that was renamed).

        Safeguards against wrongful deletion:
        - Only rows below the scanned root that the walk did NOT
          see — and the caller confirms that the walk was complete
          (not aborted, not truncated, recursive).
        - Existence re-check per candidate: ignore patterns, hidden
          folders, or permissions can make the walk skip files that
          very much still exist — those are kept.
        - Volume guard: if more than ``max_share`` of the rows
          disappear (a half-dead mount looks the same), NOTHING is deleted.
        - Rows with a transcript are NEVER deleted (ON DELETE CASCADE
          would take the transcript along with it) — the file may be gone, the
          transcript remains part of the archive. They're instead
          marked as **'orphan'** (orphan filter in the GUI: view, then
          delete deliberately). If the file reappears, the next walk
          resets the status via the normal upsert.

        Returns:
            (deleted abs_paths — for queue cleanup,
             count of rows NEWLY marked 'orphan')
        """
        root = paths.canon(root)
        if not root or not seen_abs:
            return [], 0
        prefix = root.rstrip("/") + "/"
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, abs_path, status FROM media_files WHERE abs_path LIKE ? || '%'",
                (prefix,),
            ).fetchall()
        # Orphans the walk saw again: the file is back. The upsert of the
        # walk batch resets them too, but skips rows it deduplicates (same
        # identity under another rel_path) — this covers every one of them.
        revived = [r["id"] for r in rows if r["status"] == "orphan" and r["abs_path"] in seen_abs]
        if revived:
            with self.connection() as conn:
                for i in range(0, len(revived), 500):
                    part = revived[i : i + 500]
                    conn.execute(
                        "UPDATE media_files SET status = CASE WHEN EXISTS ("
                        "  SELECT 1 FROM transcripts t WHERE t.abs_path = "
                        "  media_files.abs_path AND t.complete = 1) "
                        "THEN 'done' ELSE 'pending' END, "
                        "updated_at = unixepoch() WHERE status = 'orphan' "
                        f"AND id IN ({','.join('?' for _ in part)})",
                        part,
                    )
        candidates = [
            (r["id"], r["abs_path"], r["status"]) for r in rows if r["abs_path"] not in seen_abs
        ]
        if not candidates:
            return [], 0

        # Check existence in parallel — on a NAS, one network roundtrip each.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=16) as pool:
            exists = list(pool.map(lambda k: os.path.lexists(k[1]), candidates))
        removed = [k for k, present in zip(candidates, exists) if not present]
        if not removed:
            return [], 0
        if len(removed) > max(50, int(max_share * len(rows))):
            logger.warning(
                "Walk cleanup skipped: %d of %d rows under %s "
                "would be gone (> %.0f%%) — this looks like a half-dead "
                "mount, not like deleted files.",
                len(removed),
                len(rows),
                root,
                max_share * 100,
            )
            return [], 0

        ids = [i for i, _, _ in removed]
        protected: set[int] = set()
        deleted: list[str] = []
        newly_orphaned = 0
        with self.connection() as conn:
            for i in range(0, len(ids), 500):
                part = ids[i : i + 500]
                marks = ",".join("?" for _ in part)
                protected.update(
                    r[0]
                    for r in conn.execute(
                        f"SELECT media_id FROM transcripts WHERE media_id IN ({marks})", part
                    )
                )
            for i in range(0, len(removed), 500):
                block = removed[i : i + 500]
                remove = [(mid, ap) for mid, ap, _ in block if mid not in protected]
                if remove:
                    marks = ",".join("?" for _ in remove)
                    conn.execute(
                        f"DELETE FROM media_files WHERE id IN ({marks})", [mid for mid, _ in remove]
                    )
                    deleted.extend(ap for _, ap in remove)
                # File gone, transcript present -> orphan marking
                mark = [mid for mid, _, st in block if mid in protected and st != "orphan"]
                if mark:
                    marks = ",".join("?" for _ in mark)
                    cur = conn.execute(
                        f"UPDATE media_files SET status = 'orphan', "
                        f"updated_at = unixepoch() WHERE id IN ({marks})",
                        mark,
                    )
                    newly_orphaned += cur.rowcount or 0
        if deleted or newly_orphaned:
            logger.info(
                "Walk cleanup under %s: %d disappeared files "
                "removed, %d newly marked orphan "
                "(transcript present, file missing)",
                root,
                len(deleted),
                newly_orphaned,
            )
        return deleted, newly_orphaned

    # ── Media metadata / indexing ────────────────────────────────

    # Columns cached via the FFprobe index
    _MEDIA_META_COLS = (
        "duration_sec",
        "has_audio",
        "has_video",
        "width",
        "height",
        "audio_codec",
        "video_codec",
        "bitrate",
        "media_title",
        "indexed_at",
        "index_mtime",
    )

    def bulk_set_media_metadata(self, items: list[dict]) -> None:
        """Save FFprobe metadata for multiple files in a batch.

        Each item: {rel_path, duration_sec, has_audio, has_video, width,
        height, audio_codec, video_codec, bitrate, media_title,
        indexed_at, index_mtime}. Creates missing media_files rows.
        """
        if not items:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO media_files (
                    rel_path, abs_path, duration_sec, has_audio, has_video, width, height,
                    audio_codec, video_codec, bitrate, media_title,
                    indexed_at, index_mtime, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
                ON CONFLICT(abs_path) WHERE abs_path != '' DO UPDATE SET
                    rel_path=excluded.rel_path,
                    duration_sec=excluded.duration_sec,
                    has_audio=excluded.has_audio,
                    has_video=excluded.has_video,
                    width=excluded.width, height=excluded.height,
                    audio_codec=excluded.audio_codec,
                    video_codec=excluded.video_codec,
                    bitrate=excluded.bitrate,
                    media_title=CASE WHEN excluded.media_title != ''
                                THEN excluded.media_title ELSE media_files.media_title END,
                    indexed_at=excluded.indexed_at,
                    index_mtime=excluded.index_mtime,
                    updated_at=unixepoch()
            """,
                [
                    (
                        it["rel_path"],
                        it.get("abs_path", "") or "",
                        it.get("duration_sec", 0) or 0,
                        int(bool(it.get("has_audio", True))),
                        int(bool(it.get("has_video", False))),
                        it.get("width", 0) or 0,
                        it.get("height", 0) or 0,
                        it.get("audio_codec", "") or "",
                        it.get("video_codec", "") or "",
                        it.get("bitrate", 0) or 0,
                        it.get("media_title", "") or "",
                        it.get("indexed_at", 0) or 0,
                        it.get("index_mtime", 0) or 0,
                    )
                    for it in items
                ],
            )

    def get_index_state_batch(self, keys=None) -> dict[str, tuple]:
        """Key -> (indexed_at, index_mtime) for ALL known files.

        Returns each record under both rel_path AND abs_path, so the
        indexer also recognizes and skips already indexed files after a
        media_dir change (parent -> child) (resumable, no FFprobe).

        Args:
            keys: optionally restrict to these abs_path/rel_path keys
                (an opened folder then doesn't load the whole library).
        """
        out: dict[str, tuple] = {}

        def _add(rows):
            for r in rows:
                val = (r["indexed_at"], r["index_mtime"])
                # With multiple rows sharing the same key (duplicate from a
                # media_dir change), the INDEXED one wins — otherwise
                # an indexed_at=0 row would overwrite the real skip info.
                prev = out.get(r["rel_path"])
                if prev is None or (r["indexed_at"] or 0) > (prev[0] or 0):
                    out[r["rel_path"]] = val
                ab = r["abs_path"]
                if ab:
                    prev = out.get(ab)
                    if prev is None or (r["indexed_at"] or 0) > (prev[0] or 0):
                        out[ab] = val

        base = "SELECT rel_path, abs_path, indexed_at, index_mtime FROM media_files"
        with self.connection() as conn:
            if keys is not None:
                klist = list(keys)
                for i in range(0, len(klist), 450):
                    batch = klist[i : i + 450]
                    ph = ",".join(["?"] * len(batch))
                    _add(
                        conn.execute(
                            f"{base} WHERE abs_path IN ({ph}) OR rel_path IN ({ph})", batch + batch
                        ).fetchall()
                    )
            else:
                _add(conn.execute(base).fetchall())
        return out

    def get_media_meta_batch(self, keys=None) -> dict[str, dict]:
        """Key -> cached media metadata (for fast scan enrichment).

        Only indexed files (indexed_at > 0). Each record is retrievable under
        rel_path AND abs_path — so scanning a subfolder recognizes
        the metadata that was saved when indexing the parent (with a different
        rel_path). No re-running FFprobe.

        Args:
            keys: optionally restrict to these abs_path/rel_path keys
                (an opened folder then doesn't load the whole library).
        """
        base = (
            "SELECT rel_path, abs_path, duration_sec, has_audio, has_video, "
            "width, height, audio_codec, video_codec, bitrate, "
            "media_title, indexed_at FROM media_files WHERE indexed_at > 0"
        )
        out: dict[str, dict] = {}

        def _add(rows):
            for r in rows:
                meta = {
                    "media_duration_sec": r["duration_sec"],
                    "media_has_audio": bool(r["has_audio"]),
                    "media_has_video": bool(r["has_video"]),
                    "media_width": r["width"],
                    "media_height": r["height"],
                    "media_audio_codec": r["audio_codec"],
                    "media_video_codec": r["video_codec"],
                    "media_bitrate": r["bitrate"],
                    "media_title": r["media_title"],
                    "indexed": True,
                }
                out[r["rel_path"]] = meta
                if r["abs_path"]:
                    out[r["abs_path"]] = meta

        with self.connection() as conn:
            if keys is not None:
                klist = list(keys)
                # Blocks of 450: the list goes into the query 2x (abs+rel) -> max
                # 900 variables (< SQLite limit 999).
                for i in range(0, len(klist), 450):
                    batch = klist[i : i + 450]
                    ph = ",".join(["?"] * len(batch))
                    _add(
                        conn.execute(
                            f"{base} AND (abs_path IN ({ph}) OR rel_path IN ({ph}))", batch + batch
                        ).fetchall()
                    )
            else:
                _add(conn.execute(base).fetchall())
        return out

    def get_media_files_under(self, abs_dir: str, status: str | None = None) -> list[dict]:
        """All known media files UNDER an absolute directory.

        ``status`` optionally restricts to a file status (e.g.
        'orphan' for the walk scan's orphan attachment).

        Basis of the walk-free scan (/api/scan?walk=false): delivers per
        file everything the file list needs (scan base data, cached
        FFprobe metadata, transcript metadata) in ONE query — not a
        single filesystem access. Duplicates (same abs_path under
        multiple rel_paths, e.g. after a media_dir change) are deduplicated;
        the "best" row wins (transcript > indexed > last modified).

        Rows without abs_path (legacy records) can't be scoped and
        are missing here — the next filesystem walk backfills abs_path.
        """
        # Without a canonical root, like_prefix would return "/" and the pattern
        # "/%" would match EVERY row — the whole library instead of one folder.
        if not paths.canon(abs_dir):
            return []
        esc = paths.like_prefix(abs_dir)
        filter_sql = " AND m.status = ?" if status else ""
        sql = (
            _MEDIA_SCAN_SELECT
            + f"""
            WHERE m.abs_path LIKE ? ESCAPE '\\'{filter_sql}
            ORDER BY (t.rel_path IS NOT NULL) ASC, m.indexed_at ASC,
                     m.updated_at ASC
        """
        )
        params: tuple = (esc + "%", status) if status else (esc + "%",)
        # Dedup via dict overwrite: due to the ORDER BY sort, the
        # "best" row comes LAST and wins.
        out: dict[str, dict] = {}
        with self.connection() as conn:
            for r in conn.execute(sql, params).fetchall():
                out[r["abs_path"]] = {k: r[k] for k in r.keys()}
        return list(out.values())

    def get_media_file_at(self, abs_path: str) -> dict | None:
        """ONE media file by absolute path — same columns as
        :meth:`get_media_files_under`.

        Basis of the incremental list update: when a single file
        finishes transcribing, reloading it costs one indexed point
        query instead of a full scan over the entire library.

        Multiple rows for the same abs_path (legacy records after a
        media_dir change) are resolved as there: the "best" row wins.
        """
        sql = (
            _MEDIA_SCAN_SELECT
            + """
            WHERE m.abs_path = ?
            ORDER BY (t.rel_path IS NOT NULL) ASC, m.indexed_at ASC,
                     m.updated_at ASC
        """
        )
        # Canonicalize the input: the stored value IS canonical
        # (docs/PATHS.md), a caller with a non-normalized path
        # (doubled separator, "..") would otherwise silently find nothing.
        target = paths.canon(abs_path)
        if not target:
            return None
        with self.connection() as conn:
            rows = conn.execute(sql, (target,)).fetchall()
        if not rows:
            return None
        r = rows[-1]  # ORDER BY: the "best" row comes last
        return {k: r[k] for k in r.keys()}

    def count_media_files_under(self, abs_dir: str) -> int:
        """Number of known files under an absolute directory.

        Serves the scan stream as an expected value for walk progress:
        the record from the last walk is a good estimate of how many
        files will be found this time (-> progress bar instead of flying blind).
        """
        # Without a canonical root, like_prefix would return "/" and the pattern
        # "/%" would match EVERY row — the whole library instead of one folder.
        if not paths.canon(abs_dir):
            return 0
        esc = paths.like_prefix(abs_dir)
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT abs_path) AS c FROM media_files "
                "WHERE abs_path LIKE ? ESCAPE '\\'",
                (esc + "%",),
            ).fetchone()
            return row["c"] or 0

    # ── Paginated file list (UI rework 5.1) ─────────────────────────────
    # Replaces the "load everything at once" approach of the main list: for
    # a large library, get_media_files_under delivers the whole library (seconds to
    # the browser); here, a PAGE via LIMIT/OFFSET comes directly from the indexes.

    # Sort whitelist: frontend key -> (ORDER BY expression, index hint).
    # The expression goes into the query via an f-string, so strictly ONLY via
    # this mapping — never insert the request value itself.
    #
    # The index hint (INDEXED BY) is necessary: otherwise the planner picks the
    # abs_path range index and then sorts every row in a TEMP B-TREE
    # (~60 ms per call, measured); the order scan over the sort index with
    # an early LIMIT stop is ~0.2 ms. For the tx sorts there is
    # no matching index (the sort key lives in the transcript join) —
    # there the TEMP B-TREE remains, those sorts are rare.
    _PAGE_SORTS = {
        "name": ("m.abs_path COLLATE NOCASE", "idx_media_abs_nocase"),
        "size": ("m.file_size", "idx_media_size"),
        "modified": ("m.modified_at", "idx_media_modified"),
        "tx_date": ("t.updated_at", None),
        "tx_word_count": ("t.word_count", None),
    }

    @staticmethod
    def _escape_like(text: str) -> str:
        """Defuse wildcards in a LIKE search term (with ESCAPE '\\')."""
        return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @classmethod
    def _name_filter_sql(
        cls, pattern: str, path_col: str, other_cols: tuple[str, ...] = ()
    ) -> tuple[str, list]:
        """SQL for the file-name filter of the list and the search panel.

        Without ``*`` or ``?`` it is a substring search, as always. With them
        it is a file-name pattern — ``*.opus``, ``*replay*``,
        ``meeting_2024*`` — the syntax both filter fields advertise. The
        pattern must match the whole name, so it is tried against the path
        as a whole and against its last segment (``…/name``); other columns
        (title, display name) are matched as a whole.

        LIKE ignores case for ASCII only: a pattern with other letters ("été",
        "ß") is compared case-folded on both sides (``wx_folded``) — only
        then, the plain LIKE stays the fast path.
        """
        if not pattern.isascii():
            pattern = pattern.casefold()
            path_col = f"wx_folded({path_col})"
            other_cols = tuple(f"wx_folded({c})" for c in other_cols)
        if "*" not in pattern and "?" not in pattern:
            like = f"%{cls._escape_like(pattern)}%"
            cols = (path_col, *other_cols)
            cond = " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for c in cols)
            return f"({cond})", [like] * len(cols)
        like = cls._escape_like(pattern).replace("*", "%").replace("?", "_")
        parts = [f"{path_col} LIKE ? ESCAPE '\\'", f"{path_col} LIKE ? ESCAPE '\\'"]
        params = [like, "%/" + like]
        for c in other_cols:
            parts.append(f"{c} LIKE ? ESCAPE '\\'")
            params.append(like)
        return "(" + " OR ".join(parts) + ")", params

    @staticmethod
    def _scope_range(abs_dir: str) -> tuple[str, str] | None:
        """Lower/upper bound for "all abs_paths under this folder".

        Range scan ``abs_path >= lo AND abs_path < hi`` instead of LIKE:
        the LIKE pattern can't use the (BINARY) index idx_media_abs
        (SQLite's prefix optimization only kicks in for case-sensitive LIKE) and
        therefore ran as a full table scan per page fetch. All children
        start with ``<canon>/``; '0' is the character after '/'.
        """
        canonical_form = paths.canon(abs_dir)
        if not canonical_form:
            return None
        lo = canonical_form.rstrip("/") + "/"
        return lo, lo[:-1] + "0"

    #: The file list's filter panel: which keys it takes and which of them
    #: read the transcript (t.*) — the page query joins it only then.
    LIST_FILTER_KEYS = (
        "type",
        "len_min",
        "len_max",
        "added_days",
        "language",
        "has",
        "subfolder",
        "people",
    )
    _LIST_FILTER_NEEDS_T = ("language", "len_min", "len_max")

    @staticmethod
    def _list_filter_sql(filters: dict | None, scope_dir: str = "") -> tuple[str, list]:
        """WHERE additions of the file list's filter panel (all optional,
        combined with AND):

        type: 'audio' | 'video'; len_min / len_max: seconds of media;
        added_days: added to the library within so many days; language:
        the transcript's language code; has: a list of 'summary',
        'translation', 'warnings', 'transcript', 'no_audio', 'skipped' (by
        hand), 'empty' (a transcript without words); subfolder: a folder below
        the open one (relative); people: names — recordings at least one of
        them speaks in (``people_paths``: their abs_paths, looked up by the
        caller in the voice library).
        """
        f = filters or {}
        sql, params = "", []
        if f.get("type") in ("video", "audio"):
            # Indexed files know it; the others go by their extension
            from whisper_tm.config import VIDEO_EXTENSIONS

            ext = " OR ".join("lower(m.rel_path) LIKE ?" for _ in VIDEO_EXTENSIONS)
            video = f"(m.is_video = 1 OR (COALESCE(m.indexed_at, 0) = 0 AND ({ext})))"
            sql += f" AND {video}" if f["type"] == "video" else f" AND NOT {video}"
            params.extend("%" + e.lower() for e in sorted(VIDEO_EXTENSIONS))
        for key, op in (("len_min", ">="), ("len_max", "<=")):
            v = f.get(key)
            if isinstance(v, (int, float)) and v >= 0:
                # The media's length once indexed, else the transcript's
                sql += f" AND COALESCE(NULLIF(m.duration_sec, 0), t.duration_sec, 0) {op} ?"
                params.append(float(v))
        days = f.get("added_days")
        if isinstance(days, (int, float)) and days > 0:
            import time as _time

            sql += " AND m.created_at >= ?"
            params.append(_time.time() - float(days) * 86400)
        lang = f.get("language")
        if isinstance(lang, str) and lang:
            sql += " AND t.language = ?"
            params.append(lang)
        has = f.get("has") or []
        if "summary" in has:
            sql += " AND m.abs_path IN (SELECT abs_path FROM summaries WHERE abs_path != '')"
        if "translation" in has:
            sql += " AND m.abs_path IN (SELECT abs_path FROM translations WHERE abs_path != '')"
        if "warnings" in has:
            sql += " AND t.warning_count > 0"
        if "transcript" in has:
            sql += " AND m.status = 'done'"
        if "no_audio" in has:
            sql += " AND m.status = 'skipped' AND m.skip_reason = 'no_audio'"
        if "skipped" in has:
            sql += " AND m.status = 'skipped' AND COALESCE(m.skip_reason, '') != 'no_audio'"
        if "empty" in has:
            sql += " AND m.status = 'done' AND COALESCE(t.word_count, 0) = 0"
        sub = f.get("subfolder")
        base = paths.canon(scope_dir) if scope_dir else ""
        if isinstance(sub, str) and sub.strip("/") and base and ".." not in sub.split("/"):
            prefix = base.rstrip("/") + "/" + sub.strip("/") + "/"
            sql += " AND m.abs_path >= ? AND m.abs_path < ?"
            params.extend([prefix, prefix[:-1] + "0"])  # "/" + 1 = "0"
        if f.get("people"):
            import json as _json

            sql += " AND m.abs_path IN (SELECT value FROM json_each(?))"
            params.append(_json.dumps(sorted(set(f.get("people_paths") or []))))
        return sql, params

    def _media_page_filter(
        self, tab: str, name_filter: str, filters: dict | None = None, scope_dir: str = ""
    ) -> tuple[str, list]:
        """WHERE additions for quick-filter tab + name filter of the file list.

        The tab definitions MUST match fileMatchesTab (static/js/filelist.js)
        and the counters in get_media_stats_under — a
        diverging definition shows counters that don't match the list.

        'error'/'cancelled' key the same way the frontend does, via the MOST RECENT
        error/cancelled queue entry per filename (queueOutcomes map):
        whoever was re-queued after an error and then cancelled, counts
        as cancelled, not twice.

        The queue filters are SET-based (IN over a once-materialized
        set) instead of correlated EXISTS subqueries: the latter ran once per
        media_files row (one index seek per row, ~300 ms per page fetch on a large library), the
        set on the other hand once over the few error/cancelled/processing rows.
        """
        sql, params = "", []
        _outcome_in = (
            " AND m.rel_path IN ("
            "SELECT qi.filename FROM queue_items qi"
            " JOIN (SELECT filename, MAX(id) AS mid FROM queue_items"
            "        WHERE status IN ('error','cancelled')"
            "        GROUP BY filename) latest ON qi.id = latest.mid"
            " WHERE qi.status = ?)"
        )
        _processing_in = (
            " m.rel_path IN (SELECT filename FROM queue_items  WHERE status = 'processing')"
        )
        if tab == "done":
            sql += " AND m.status = 'done'"
        elif tab == "orphan":
            sql += " AND m.status = 'orphan'"
        elif tab == "pending":
            sql += " AND m.status NOT IN ('done','skipped','orphan') AND NOT" + _processing_in
        elif tab == "processing":
            sql += " AND" + _processing_in
        elif tab in ("error", "cancelled"):
            sql += _outcome_in
            params.append(tab)
        elif tab == "unembedded":
            # Only the SQL half (fully transcribed); the comparison against
            # the RAG keys happens in the caller (separate rag.db).
            sql += " AND m.status = 'done'"

        if name_filter:
            # Same fields as the earlier client-side filter (_hay): rel_path
            # covers the subfolder too, plus display name, media title,
            # and the transcription version as a "v..." token.
            cond, args = self._name_filter_sql(
                name_filter,
                "m.rel_path",
                ("m.display_name", "m.media_title", "('v' || COALESCE(t.app_version,''))"),
            )
            sql += " AND " + cond
            params.extend(args)
        f_sql, f_params = self._list_filter_sql(filters, scope_dir)
        return sql + f_sql, params + f_params

    def get_media_files_page(
        self,
        abs_dir: str,
        tab: str = "all",
        name_filter: str = "",
        sort: str = "name",
        sort_dir: str = "asc",
        limit: int = 200,
        offset: int = 0,
        embedded_keys: set[str] | None = None,
        filters: dict | None = None,
    ) -> dict:
        """ONE page of the file list under an absolute directory.

        Same row shape as :meth:`get_media_files_under` /
        :meth:`get_media_file_at` (contract: scanner.db_row_to_entry) —
        paginated pages and incremental single-row updates must not
        drift apart.

        Return: {"total": <rows matching the filter>, "rows": [<page>]}.

        Special case tab='unembedded': ``embedded_keys`` (rel_paths + basenames
        from the RAG DB) are filtered out in Python AFTER the done
        restriction, because the keys live in rag.db. Offset/limit apply
        to the filtered result; the pass stays restricted to done
        rows.
        """
        scope = self._scope_range(abs_dir)
        if not scope:
            return {"total": 0, "rows": []}
        limit = min(max(int(limit or 200), 1), 5000)
        offset = max(int(offset or 0), 0)

        order, idx_hint = self._PAGE_SORTS.get(sort, self._PAGE_SORTS["name"])
        direction = "DESC" if str(sort_dir).lower() in ("desc", "ab") else "ASC"
        # m.id as tiebreaker: without a total ordering, rows with the same
        # sort value can jump between two pages (duplicate/missing).
        order_sql = f" ORDER BY {order} {direction}, m.id {direction}"

        where = " WHERE m.abs_path >= ? AND m.abs_path < ?"
        params: list = list(scope)
        if order.startswith("m.abs_path COLLATE NOCASE"):
            # The same range once more in the name index's collation: it can
            # seek to the folder instead of walking the whole library. Every
            # row of the binary range lies inside it.
            where += " AND m.abs_path COLLATE NOCASE >= ? AND m.abs_path COLLATE NOCASE < ?"
            params.extend(scope)
        f_sql, f_params = self._media_page_filter(tab, name_filter, filters, abs_dir)
        where += f_sql
        params.extend(f_params)

        with self.connection() as conn:
            if tab == "unembedded":
                keys = embedded_keys or set()
                rows, total, skipped = [], 0, 0
                for r in conn.execute(_MEDIA_SCAN_SELECT + where + order_sql, params):
                    if _embedded(r, keys):
                        continue
                    total += 1
                    if skipped < offset:
                        skipped += 1
                    elif len(rows) < limit:
                        rows.append({k: r[k] for k in r.keys()})
                return {"total": total, "rows": rows}

            # The transcript join is 1:0..1 and doesn't change the row count;
            # only the name filter (t.app_version) and tx sorts read t.*.
            needs_t = (
                bool(name_filter)
                or sort in ("tx_date", "tx_word_count")
                # "is set", not "is true": a length of 0 is a filter too, and
                # without the join its query had no t.duration_sec (a 500)
                or any((filters or {}).get(k) not in (None, "") for k in self._LIST_FILTER_NEEDS_T)
                or bool({"warnings", "empty"} & set((filters or {}).get("has") or []))
            )
            t_join = (
                " LEFT JOIN transcripts t ON t.abs_path = m.abs_path"
                "   AND m.abs_path != '' AND t.complete = 1"
                if needs_t
                else ""
            )
            total = conn.execute(
                f"SELECT COUNT(*) AS c FROM media_files m{t_join}{where}", params
            ).fetchone()["c"]
            # Two-phase: first determine ONLY the page's IDs (in the best
            # case runs entirely within the index), then fetch the wide rows
            # (38 columns + transcript join) only for those <=limit IDs.
            # In a single phase, the TEMP B-TREE would otherwise sort ALL the FULL rows.
            hint = f" INDEXED BY {idx_hint}" if idx_hint and not needs_t else ""
            inside = (
                f"SELECT m.id AS pid FROM media_files m{hint}{t_join}"
                f"{where}{order_sql} LIMIT ? OFFSET ?"
            )
            rows = conn.execute(
                _MEDIA_SCAN_SELECT + f" JOIN ({inside}) page ON m.id = page.pid" + order_sql,
                params + [limit, offset],
            ).fetchall()
            return {"total": total, "rows": [{k: r[k] for k in r.keys()} for r in rows]}

    def get_media_stats_under(self, abs_dir: str) -> dict:
        """Server-side aggregates for the sidebar counters + directory statistics.

        Replaces _computeFileAggregates (static/js/stats.js), which ran over
        the full client-side list — with pagination the client only sees
        one window and would show incorrect numbers.

        Scoped via abs_path prefix, never via rel_path (the DB holds several
        roots at once). The tab definitions match
        _media_page_filter — counters and list content must be consistent.
        """
        empty = {
            "total": 0,
            "done": 0,
            "pending": 0,
            "processing": 0,
            "skipped": 0,
            "orphan": 0,
            "error": 0,
            "cancelled": 0,
            "empty_count": 0,
            "no_audio_count": 0,
            "indexed": 0,
            "total_size": 0,
            "transcribed_audio_sec": 0.0,
            "total_proc_sec": 0.0,
            "total_words": 0,
            "total_segs": 0,
            "total_media_dur_sec": 0.0,
            # For honest percentages and the "remainder": only what's
            # still to come, and only the speed on recordings WITH speech.
            "pending_media_dur_sec": 0.0,
            "pending_without_dur": 0,
            "speech_audio_sec": 0.0,
            "speech_proc_sec": 0.0,
            # The time left (core/eta.py) costs per file, not only per second
            "pending_with_dur": 0,
        }
        scope = self._scope_range(abs_dir)
        if not scope:
            return empty
        with self.connection() as conn:
            r = conn.execute(
                """
                SELECT COUNT(*)                                   AS total,
                       SUM(m.status = 'done')                     AS done,
                       SUM(m.status = 'skipped')                  AS skipped,
                       SUM(m.status = 'orphan')                   AS orphan,
                       SUM(m.status = 'done'
                           AND COALESCE(t.word_count, 0) = 0)     AS empty_count,
                       SUM(m.status = 'skipped'
                           AND m.skip_reason = 'no_audio')        AS no_audio_count,
                       SUM(m.indexed_at > 0)                      AS indexed,
                       SUM(COALESCE(m.file_size, 0))              AS total_size,
                       SUM(CASE WHEN m.status = 'done'
                           THEN COALESCE(t.duration_sec, 0) END)  AS transcribed_audio_sec,
                       SUM(CASE WHEN m.status = 'done'
                           THEN COALESCE(t.processing_secs, 0) END) AS total_proc_sec,
                       SUM(CASE WHEN m.status = 'done'
                           THEN COALESCE(t.word_count, 0) END)    AS total_words,
                       SUM(CASE WHEN m.status = 'done'
                           THEN COALESCE(t.segment_count, 0) END) AS total_segs,
                       SUM(COALESCE(m.duration_sec,
                           CASE WHEN m.status = 'done'
                                THEN t.duration_sec END, 0))      AS total_media_dur_sec,
                       -- What's still left to transcribe: only pending durations.
                       -- The earlier remainder (total - transcribed) also counted the
                       -- duration of skipped files.
                       SUM(CASE WHEN m.status NOT IN ('done','skipped','orphan')
                           THEN COALESCE(m.duration_sec, 0) END) AS pending_media_dur_sec,
                       SUM(m.status NOT IN ('done','skipped','orphan')
                           AND COALESCE(m.duration_sec, 0) = 0)  AS pending_without_dur,
                       SUM(m.status NOT IN ('done','skipped','orphan')
                           AND COALESCE(m.duration_sec, 0) > 0)  AS pending_with_dur,
                       -- Speed ONLY on recordings with speech: silent day-long
                       -- recordings (processed in minutes each) pull the
                       -- average up and make the remainder too short.
                       SUM(CASE WHEN m.status = 'done' AND COALESCE(t.word_count, 0) > 0
                           THEN COALESCE(t.duration_sec, 0) END)  AS speech_audio_sec,
                       SUM(CASE WHEN m.status = 'done' AND COALESCE(t.word_count, 0) > 0
                           THEN COALESCE(t.processing_secs, 0) END) AS speech_proc_sec
                  FROM media_files m
                  LEFT JOIN transcripts t ON t.abs_path = m.abs_path
                       AND m.abs_path != '' AND t.complete = 1
                 WHERE m.abs_path >= ? AND m.abs_path < ?
            """,
                scope,
            ).fetchone()
            stats = dict(empty)
            for k in r.keys():
                if k in stats and r[k] is not None:
                    stats[k] = r[k]

            # Queue-dependent counters: same definition as _media_page_filter
            # (most recent error/cancelled entry per filename wins).
            qi = conn.execute(
                """
                SELECT SUM(qi.status = 'error')     AS error,
                       SUM(qi.status = 'cancelled') AS cancelled
                  FROM queue_items qi
                  JOIN (SELECT filename, MAX(id) AS mid FROM queue_items
                         WHERE status IN ('error','cancelled')
                         GROUP BY filename) latest ON qi.id = latest.mid
                 WHERE EXISTS (SELECT 1 FROM media_files m
                                WHERE m.rel_path = qi.filename
                                  AND m.abs_path >= ? AND m.abs_path < ?)
            """,
                scope,
            ).fetchone()
            stats["error"] = qi["error"] or 0
            stats["cancelled"] = qi["cancelled"] or 0
            stats["processing"] = conn.execute(
                "SELECT COUNT(DISTINCT qi.filename) AS c FROM queue_items qi"
                " WHERE qi.status = 'processing'"
                "   AND EXISTS (SELECT 1 FROM media_files m"
                "                WHERE m.rel_path = qi.filename"
                "                  AND m.abs_path >= ? AND m.abs_path < ?)",
                scope,
            ).fetchone()["c"]

            # 'pending' as in fileMatchesTab: not done/skipped/orphan and
            # not currently processing.
            stats["pending"] = max(
                0,
                stats["total"]
                - stats["done"]
                - stats["skipped"]
                - stats["orphan"]
                - stats["processing"],
            )
        return stats

    # ── Inputs of the time estimates (core/eta.py) ────────────────────

    def eta_transcription_groups(
        self, since: float, short_sec: float = 60.0
    ) -> tuple[tuple, tuple]:
        """Transcriptions since ``since`` as the two groups of an ETA fit:
        (files, audio seconds, processing seconds), short and long.

        Only recordings with speech: silent day-long recordings processed in
        minutes each would make long files look far cheaper than they are.
        """
        with self.connection() as conn:
            r = conn.execute(
                """
                SELECT SUM(duration_sec < :s)                             AS ns,
                       SUM(CASE WHEN duration_sec < :s THEN duration_sec END)    AS ds,
                       SUM(CASE WHEN duration_sec < :s THEN processing_secs END) AS ts,
                       SUM(duration_sec >= :s)                            AS nl,
                       SUM(CASE WHEN duration_sec >= :s THEN duration_sec END)    AS dl,
                       SUM(CASE WHEN duration_sec >= :s THEN processing_secs END) AS tl
                  FROM transcripts
                 WHERE complete = 1 AND COALESCE(word_count, 0) > 0
                   AND processing_secs > 0 AND duration_sec > 0
                   AND created_at >= :since
            """,
                {"s": short_sec, "since": since},
            ).fetchone()
        return (
            (r["ns"] or 0, r["ds"] or 0.0, r["ts"] or 0.0),
            (r["nl"] or 0, r["dl"] or 0.0, r["tl"] or 0.0),
        )

    def eta_byte_rates(self, min_files: int = 20) -> dict[str, float]:
        """Bytes per second of media per extension, from indexed files —
        gives files without a known duration one from their size."""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT LOWER(extension) AS ext, COUNT(*) AS n,
                       SUM(file_size) AS size, SUM(duration_sec) AS dur
                  FROM media_files
                 WHERE duration_sec > 0 AND file_size > 0
                 GROUP BY LOWER(extension)
            """).fetchall()
        return {r["ext"]: r["size"] / r["dur"] for r in rows if r["n"] >= min_files and r["dur"]}

    def eta_pending_without_duration(self, abs_dir: str) -> list[tuple]:
        """Open files under ``abs_dir`` without a known duration, per
        extension: (extension, files, bytes)."""
        scope = self._scope_range(abs_dir)
        if not scope:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT LOWER(extension) AS ext, COUNT(*) AS n,
                       SUM(COALESCE(file_size, 0)) AS size
                  FROM media_files
                 WHERE abs_path >= ? AND abs_path < ?
                   AND status NOT IN ('done','skipped','orphan')
                   AND COALESCE(duration_sec, 0) = 0
                 GROUP BY LOWER(extension)
            """,
                scope,
            ).fetchall()
        return [(r["ext"], r["n"], r["size"]) for r in rows]

    def get_media_names_under(
        self,
        abs_dir: str,
        tab: str = "all",
        name_filter: str = "",
        sort: str = "name",
        sort_dir: str = "asc",
        embedded_keys: set[str] | None = None,
        with_status: bool = False,
        filters: dict | None = None,
    ):
        """All names of the active filter, in list order — relative to the
        open folder, like the list's own entries (scanner.db_row_to_entry).

        For "select all filtered" actions (Ctrl+A, batch, filtered retry,
        bulk delete, shift-range across page boundaries): the client only holds
        one window and can no longer build the set itself. Just
        names — a small response even for a large library.

        The stored rel_path is relative to the folder open when a file was
        indexed: handed out as the name, a select-all under another folder
        held names the list didn't know ("samples/a.mp4" for "a.mp4") and
        acted on files that aren't there. The name comes from the identity.

        ``with_status=True`` returns ``(name, status)`` tuples: the selection
        bar also needs the status for rows outside the window.
        """
        scope = self._scope_range(abs_dir)
        if not scope:
            return []
        order, _ = self._PAGE_SORTS.get(sort, self._PAGE_SORTS["name"])
        direction = "DESC" if str(sort_dir).lower() in ("desc", "ab") else "ASC"
        where = " WHERE m.abs_path >= ? AND m.abs_path < ?"
        params: list = list(scope)
        f_sql, f_params = self._media_page_filter(tab, name_filter, filters, abs_dir)
        where += f_sql
        params.extend(f_params)
        # Every abs_path in the range starts with the folder and its "/"
        sql = (
            f"SELECT substr(m.abs_path, ?) AS name, m.abs_path, m.rel_path, m.status"
            f" FROM media_files m"
            f" LEFT JOIN transcripts t ON t.abs_path = m.abs_path"
            f"   AND m.abs_path != '' AND t.complete = 1"
            f"{where} ORDER BY {order} {direction}, m.id {direction}"
        )
        with self.connection() as conn:
            rows = conn.execute(sql, [len(scope[0]) + 1] + params).fetchall()
        if tab == "unembedded":
            keys = embedded_keys or set()
            rows = [r for r in rows if not _embedded(r, keys)]
        if with_status:
            return [(r["name"], r["status"] or "pending") for r in rows]
        return [r["name"] for r in rows]

    def get_index_stats(self, subfolder_prefix: str | None = None) -> dict:
        """Index coverage + aggregated statistics.

        Args:
            subfolder_prefix: only files whose rel_path starts with this (optional).
        Returns:
            total, indexed, unindexed, video_count, audio_count,
            total_duration_sec, indexed_duration_sec, total_size,
            transcribed (number of completed transcripts).
        """

        # scope via abs_path, NOT rel_path — the DB holds
        # multiple roots at once; the same rel_path can exist again under old
        # roots and would be double-counted. Rows without
        # abs_path (legacy records before schema v3) go through the rel_path
        # fallback instead of silently dropping out of the count.
        def _esc(s: str) -> str:
            # % and _ in a folder name are characters, not wildcards.
            return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        filter_sql = ""
        params: list = []
        if subfolder_prefix:
            sub = subfolder_prefix.rstrip("/")
            if sub.startswith("/"):
                filter_sql = "abs_path LIKE ? ESCAPE '\\'"
                params = [_esc(sub) + "/%"]
            else:
                try:
                    from whisper_tm.config import config as _cfg

                    root = str(_cfg.media_dir).rstrip("/")
                except Exception:
                    root = None
                if root:
                    filter_sql = (
                        "(abs_path LIKE ? ESCAPE '\\' OR "
                        "(COALESCE(abs_path, '') = '' AND "
                        "rel_path LIKE ? ESCAPE '\\'))"
                    )
                    params = [_esc(root + "/" + sub) + "/%", _esc(sub) + "/%"]
                else:
                    filter_sql = "rel_path LIKE ? ESCAPE '\\'"
                    params = [_esc(sub) + "/%"]
        where = f" WHERE {filter_sql}" if filter_sql else ""
        with self.connection() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                        AS total,
                    SUM(CASE WHEN indexed_at > 0 THEN 1 ELSE 0 END) AS indexed,
                    SUM(CASE WHEN has_video = 1 THEN 1 ELSE 0 END)  AS video_count,
                    SUM(duration_sec)                               AS total_duration_sec,
                    SUM(file_size)                                  AS total_size
                FROM media_files{where}
            """,
                params,
            ).fetchone()
            total = row["total"] or 0
            indexed = row["indexed"] or 0
            # The transcribed count uses the same abs_path scope —
            # ignoring the restriction would always count the
            # whole DB.
            t_query = "SELECT COUNT(*) AS c FROM transcripts WHERE complete = 1"
            if filter_sql:
                t_query += f" AND {filter_sql}"
            transcribed = conn.execute(t_query, params).fetchone()["c"]
            return {
                "total": total,
                "indexed": indexed,
                "unindexed": max(0, total - indexed),
                "video_count": row["video_count"] or 0,
                "total_duration_sec": round(row["total_duration_sec"] or 0, 1),
                "total_size": row["total_size"] or 0,
                "transcribed": transcribed,
            }

    # ── Skiplist ──────────────────────────────────────────────────────

    def realign_paths(self, media_root: str) -> dict:
        """Re-derive rel_path from abs_path.

        This is NOT a repair, but the intended path (docs/PATHS.md):
        abs_path is the identity, rel_path a display value derived from it,
        relative to the current root. When the root changes — because
        the library has moved, or a different folder was opened
        in the frontend — the display value is updated here. The identity
        remains untouched.

        Runs on every start (app.py) and normally costs one
        index scan, since only diverging rows are touched.

        Rows whose target value is already taken by another row
        stay unchanged: that's a real duplicate, which should be
        merged (tools/migrate_paths.py), not blindly overwritten.
        """
        # Canonical root (docs/PATHS.md): absolute, /-separated, no
        # trailing slash. n counts on the UNESCAPED form — substr computes
        # on the stored abs_path, not on the LIKE pattern.
        root = paths.canon(media_root)
        if not root:
            return {}
        n = len(root)
        # % and _ are allowed in folder names; without ESCAPE, a root
        # /mnt/50%data would also match rows under /mnt/50-completely-elsewhere-data
        # and write them a wrong rel_path.
        pattern = paths.like_prefix(root) + "%"
        result: dict[str, int] = {}
        with self.connection() as conn:
            for table in ("media_files", "transcripts", "skiplist"):
                try:
                    # First READ what needs to change, then write.
                    # A subquery that reads the same table the
                    # UPDATE is currently changing delivers a different result
                    # depending on optimizer order — for a tool
                    # that runs on every start, that's unacceptable.
                    candidates = conn.execute(
                        f"""
                        SELECT rowid AS rid, substr(abs_path, ? + 2) AS new_value
                          FROM {table}
                         WHERE abs_path LIKE ? ESCAPE '\\'
                           AND rel_path != substr(abs_path, ? + 2)
                    """,
                        (n, pattern, n),
                    ).fetchall()
                    if not candidates:
                        continue
                    # Target values that are already taken stay untouched:
                    # that's a real duplicate, which should be
                    # merged (tools/migrate_paths.py).
                    used = {r["rel_path"] for r in conn.execute(f"SELECT rel_path FROM {table}")}
                    to_change = [
                        (k["new_value"], k["rid"]) for k in candidates if k["new_value"] not in used
                    ]
                    if not to_change:
                        continue
                    try:
                        conn.executemany(
                            f"UPDATE {table} SET rel_path = ?, "
                            "updated_at = unixepoch() WHERE rowid = ?",
                            to_change,
                        )
                    except sqlite3.OperationalError as e:
                        if "updated_at" not in str(e):
                            raise
                        conn.executemany(
                            f"UPDATE {table} SET rel_path = ? WHERE rowid = ?", to_change
                        )
                    result[table] = len(to_change)
                except sqlite3.OperationalError as e:
                    logger.warning("realign_paths(%s): %s", table, e)
                    continue
        if result:
            logger.info(
                "Paths realigned to %s: %s",
                root,
                ", ".join(f"{v} x {k}" for k, v in result.items()),
            )
        return result

    def add_skip(
        self,
        rel_path: str,
        reason: str = "manual",
        media_dir: str | None = None,
        abs_path: str | None = None,
    ) -> None:
        """Put a file on the skiplist.

        ``abs_path`` is the actual location. Whoever knows it should pass it —
        that's more reliable than a root the caller often doesn't have
        (the transcription worker only knows ``source_path``).
        """
        # Identity via core/paths, not via os.path.join — the latter is
        # exactly the pattern docs/PATHS.md forbids and produced, on
        # Windows, a third spelling of the same path.
        abs_path = paths.canon(abs_path) if abs_path else self._identity(rel_path, media_dir)
        with self.connection() as conn:
            if not abs_path:
                # No root derivable: adopt the already known absolute
                # path of the media file.
                row = conn.execute(
                    "SELECT abs_path FROM media_files WHERE rel_path = ? "
                    "AND abs_path != '' LIMIT 1",
                    (rel_path,),
                ).fetchone()
                abs_path = (row["abs_path"] if row else "") or ""
            conn.execute(
                """
                INSERT INTO skiplist (rel_path, reason, added_at, abs_path)
                VALUES (?, ?, unixepoch(), ?)
                ON CONFLICT(abs_path) WHERE abs_path != '' DO UPDATE SET
                    rel_path=excluded.rel_path,
                    reason=excluded.reason, added_at=unixepoch()
            """,
                (rel_path, reason, abs_path),
            )
            if abs_path:
                conn.execute(
                    """
                    UPDATE media_files SET status='skipped', skip_reason=?,
                           updated_at=unixepoch()
                     WHERE abs_path = ?
                """,
                    (reason, abs_path),
                )

    def remove_skip(self, rel_path: str, media_dir: str | None = None) -> None:
        """Remove a file from the skiplist.

        Runs via the identity (docs/PATHS.md) — otherwise removing
        a file would also take along the identically named one from another root.
        """
        target = self._identity(rel_path, media_dir)
        with self.connection() as conn:
            if target:
                conn.execute("DELETE FROM skiplist WHERE abs_path = ?", (target,))
            else:
                conn.execute(
                    "DELETE FROM skiplist WHERE rel_path = ? "
                    "AND (abs_path IS NULL OR abs_path = '')",
                    (rel_path,),
                )
            # #085: derive status from transcript existence — files with
            # a completed transcript must be 'done' again, not 'pending'.
            row = (
                conn.execute(
                    "SELECT 1 FROM transcripts WHERE abs_path = ? AND complete = 1", (target,)
                ).fetchone()
                if target
                else None
            )
            new_status = "done" if row else "pending"
            if target:
                conn.execute(
                    """
                    UPDATE media_files SET status=?, skip_reason='', updated_at=unixepoch()
                     WHERE abs_path = ? AND status = 'skipped'
                """,
                    (new_status, target),
                )

    def get_skiplist(self) -> dict:
        """Skipped files, keyed by identity.

        Rows without ``abs_path`` (legacy records) keep ``rel_path`` as the
        key — for them the display value remains the only handle.

        Keyed by ``rel_path``, a broken ``intro.mp4`` in
        folder A would permanently also lock out ``intro.mp4`` in folder B: the job was
        checked off as "skipped" without any error appearing anywhere
        (docs/PATHS.md).
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT rel_path, abs_path, reason, added_at FROM skiplist"
            ).fetchall()
            return {
                (r["abs_path"] or r["rel_path"]): {
                    "reason": r["reason"],
                    "added_at": r["added_at"] or 0,
                }
                for r in rows
            }

    # ── User Settings ─────────────────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        with self.connection() as conn:
            row = conn.execute("SELECT value FROM user_settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_settings (key, value, updated_at)
                VALUES (?, ?, unixepoch())
            """,
                (key, value),
            )

    def get_all_settings(self) -> dict:
        with self.connection() as conn:
            rows = conn.execute("SELECT key, value FROM user_settings").fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ── Job history ───────────────────────────────────────────────────

    # ── Incidents (schema v19) ────────────────────────────────────────────────────

    INCIDENT_KINDS = ("warning", "error", "oom", "retry", "system")

    def report_incident(
        self,
        kind: str,
        text: str,
        *,
        abs_path: str = "",
        filename: str = "",
        job_id: str = "",
        step: str = "",
        count: int = 1,
        log_file: str = "",
    ) -> int:
        """Record an incident permanently. Returns the row id.

        Deliberately without deduplication: N skipped windows are N rows
        in the log, and that exact number can later be the information.
        Whoever wants to batch passes ``count``.
        """
        kind = str(kind or "warning").strip().lower()
        if kind not in self.INCIDENT_KINDS:
            kind = "warning"
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO incidents (abs_path, filename, job_id, kind, "
                "step, text, count, log_file) VALUES (?,?,?,?,?,?,?,?)",
                (
                    str(abs_path or ""),
                    str(filename or ""),
                    str(job_id or ""),
                    kind,
                    str(step or "")[:80],
                    str(text or "")[:2000],
                    max(1, int(count or 1)),
                    str(log_file or ""),
                ),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def incidents(
        self,
        *,
        abs_path: str | None = None,
        filename: str | None = None,
        kind: str | None = None,
        since: float | None = None,
        to_value: float | None = None,
        search: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        """Read incidents, newest first. ``filename`` matches as a suffix
        (relative path without root), ``abs_path`` exactly."""
        clauses, par = [], []
        if abs_path:
            clauses.append("abs_path = ?")
            par.append(abs_path)
        if filename:
            clauses.append("(filename = ? OR abs_path LIKE ?)")
            par += [filename, "%/" + filename.lstrip("/")]
        if kind:
            clauses.append("kind = ?")
            par.append(str(kind).lower())
        if since is not None:
            clauses.append("timestamp >= ?")
            par.append(float(since))
        if to_value is not None:
            clauses.append("timestamp <= ?")
            par.append(float(to_value))
        if search:
            clauses.append("(text LIKE ? OR step LIKE ?)")
            par += [f"%{search}%", f"%{search}%"]
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit = max(1, min(int(limit or 200), 5000))
        with self.connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(count),0) FROM incidents {where}", par
            ).fetchone()
            lines = conn.execute(
                f"SELECT id, timestamp, abs_path, filename, job_id, kind, step, "
                f"text, count, log_file FROM incidents {where} "
                f"ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                par + [limit, max(0, int(offset or 0))],
            ).fetchall()
            kinds = conn.execute(
                f"SELECT kind, COUNT(*) FROM incidents {where} GROUP BY kind", par
            ).fetchall()
        return {
            "total": int(total[0] or 0),
            "events": int(total[1] or 0),
            "kinds": {a: n for a, n in kinds},
            "items": [
                dict(
                    zip(
                        (
                            "id",
                            "timestamp",
                            "abs_path",
                            "filename",
                            "job_id",
                            "kind",
                            "step",
                            "text",
                            "count",
                            "log_file",
                        ),
                        z,
                    )
                )
                for z in lines
            ],
        }

    def incidents_per_file(self, abs_paths: list[str]) -> dict[str, int]:
        """Incident count per file — for lists (one query instead of N)."""
        if not abs_paths:
            return {}
        out: dict[str, int] = {}
        with self.connection() as conn:
            for i in range(0, len(abs_paths), 500):
                part = abs_paths[i : i + 500]
                q = ",".join("?" * len(part))
                for p, n in conn.execute(
                    f"SELECT abs_path, COUNT(*) FROM incidents "
                    f"WHERE abs_path IN ({q}) GROUP BY abs_path",
                    part,
                ):
                    out[p] = n
        return out

    def log_job(self, job_id: str, filename: str, job_type: str = "transcription", **kwargs) -> int:
        """Add a job to the history."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO job_history (
                    job_id, filename, job_type, status,
                    model, language, settings_json
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?)
            """,
                (
                    job_id,
                    filename,
                    job_type,
                    kwargs.get("model", ""),
                    kwargs.get("language", ""),
                    json.dumps(kwargs.get("settings", {})),
                ),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            # #064: limit the history to the last N entries (avoid unbounded
            # growth). Deletes everything older than the Nth-newest entry.
            conn.execute(
                """
                DELETE FROM job_history WHERE id < (
                    SELECT MIN(id) FROM (
                        SELECT id FROM job_history ORDER BY id DESC LIMIT ?
                    )
                )
            """,
                (self._JOB_HISTORY_MAX,),
            )
            return new_id

    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job status (only the most recent row for this job_id)."""
        # #087: without kwargs this would produce an invalid "UPDATE ... SET  WHERE"
        if not kwargs:
            return
        sets = []
        vals = []
        for key, val in kwargs.items():
            # #061: only allow known columns (no kwargs key goes directly into SQL)
            if key not in self._JOB_COLS:
                logger.warning("update_job: unknown column ignored: %s", key)
                continue
            sets.append(f"{key} = ?")
            vals.append(val)
        if not sets:
            return
        if "finished_at" not in kwargs and kwargs.get("status") in ("done", "error"):
            sets.append("finished_at = unixepoch()")
        with self.connection() as conn:
            vals.append(job_id)
            # #062: job_history is a history table (multiple rows per
            # job_id possible) -> specifically target the NEWEST row via PK,
            # instead of overwriting all rows with the same job_id.
            conn.execute(
                f"UPDATE job_history SET {', '.join(sets)} "
                "WHERE id = (SELECT id FROM job_history WHERE job_id = ? "
                "ORDER BY id DESC LIMIT 1)",
                vals,
            )

    # ── Sync: JSON <-> DB ──────────────────────────────────────────────

    # ── Summaries (DB = primary source) ─────────────────────

    def _identity(self, rel_path: str, media_dir: str | None) -> str:
        """rel_path + root -> canonical abs_path (docs/PATHS.md).

        If the caller doesn't know a root, the configured one is used.
        An empty return value means: no identity derivable, the caller
        must fall back to the ambiguous rel_path.
        """
        if paths.is_absolute(rel_path):
            return paths.canon(rel_path)
        root = media_dir
        if not root:
            try:
                from whisper_tm.config import config as _cfg

                root = str(_cfg.media_dir)
            except Exception as _e:  # noqa
                root = ""
        return paths.abs_from(rel_path, root) if root else ""

    def get_summaries(self, rel_path: str, media_dir: str | None = None) -> list | None:
        """Load summaries for a file.

        Returns:
            List of summary dicts, or None if NO entry exists
            (None != empty list — allows the caller to fall back to JSON).
        """
        with self.connection() as conn:
            target = self._identity(rel_path, media_dir)
            rows = (
                conn.execute(
                    "SELECT text, prompt, model, created_at FROM summaries "
                    "WHERE abs_path = ? ORDER BY sum_index",
                    (target,),
                ).fetchall()
                if target
                else []
            )
            if not rows:
                # Legacy rows without identity
                rows = conn.execute(
                    "SELECT text, prompt, model, created_at FROM summaries "
                    "WHERE rel_path = ? AND (abs_path IS NULL OR abs_path = '') "
                    "ORDER BY sum_index",
                    (rel_path,),
                ).fetchall()
            if not rows:
                return None
            return [
                {
                    "text": r["text"],
                    "prompt": r["prompt"],
                    "model": r["model"],
                    "timestamp": r["created_at"],
                }
                for r in rows
            ]

    def replace_summaries(
        self, rel_path: str, summaries: list, media_dir: str | None = None
    ) -> None:
        """Replace all summaries of a file (mirrors the JSON list)."""
        target = self._identity(rel_path, media_dir)
        with self.connection() as conn:
            if target:
                conn.execute("DELETE FROM summaries WHERE abs_path = ?", (target,))
            else:
                # Only clean up via the display value when no identity can be
                # derived — otherwise writing for /nasA/intro.mp4 would also
                # delete the summaries of /nasB/intro.mp4.
                conn.execute(
                    "DELETE FROM summaries WHERE rel_path = ? "
                    "AND (abs_path IS NULL OR abs_path = '')",
                    (rel_path,),
                )
            if summaries:
                conn.executemany(
                    "INSERT INTO summaries (abs_path, rel_path, sum_index, text, "
                    "prompt, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            target,
                            rel_path,
                            i,
                            s.get("text", ""),
                            s.get("prompt", ""),
                            s.get("model", ""),
                            s.get("timestamp", 0) or 0,
                        )
                        for i, s in enumerate(summaries)
                        if isinstance(s, dict) and s.get("text")
                    ],
                )

    # ── LLM translations (v16) ───────────────────────────────────────────

    def get_translation_languages(self, rel_path: str, media_dir: str | None = None) -> list:
        """Existing translation languages for a file (with metadata)."""
        with self.connection() as conn:
            target = self._identity(rel_path, media_dir)
            if not target:
                return []
            rows = conn.execute(
                "SELECT language, model, created_at, segments_json "
                "FROM translations WHERE abs_path = ? ORDER BY language",
                (target,),
            ).fetchall()
            out = []
            for r in rows:
                try:
                    count = len(json.loads(r["segments_json"]))
                except (TypeError, json.JSONDecodeError):
                    count = 0
                out.append(
                    {
                        "language": r["language"],
                        "model": r["model"],
                        "created_at": r["created_at"],
                        "segments": count,
                    }
                )
            return out

    def get_translation(
        self, rel_path: str, language: str, media_dir: str | None = None
    ) -> list | None:
        """Load a translation track: [{id, text}] or None."""
        with self.connection() as conn:
            target = self._identity(rel_path, media_dir)
            if not target:
                return None
            row = conn.execute(
                "SELECT segments_json FROM translations WHERE abs_path = ? AND language = ?",
                (target, language),
            ).fetchone()
            if not row:
                return None
            try:
                data = json.loads(row["segments_json"])
                return data if isinstance(data, list) else None
            except (TypeError, json.JSONDecodeError):
                return None

    def replace_translation(
        self,
        rel_path: str,
        language: str,
        segments: list,
        model: str = "",
        media_dir: str | None = None,
    ) -> bool:
        """Replace a language's translation track (one track per language)."""
        target = self._identity(rel_path, media_dir)
        if not target:
            return False
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM translations WHERE abs_path = ? AND language = ?", (target, language)
            )
            conn.execute(
                "INSERT INTO translations (abs_path, rel_path, language, "
                "segments_json, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    target,
                    rel_path,
                    language,
                    json.dumps(segments, ensure_ascii=False),
                    model,
                    time.time(),
                ),
            )
        return True

    def delete_derivatives(
        self,
        rel_path: str,
        media_dir: str | None = None,
        *,
        summaries: bool = False,
        translations: bool = False,
        edits: bool = False,
    ) -> dict:
        """Selectively delete a file's derivations; returns the counts.

        Counterpart to :meth:`delete_transcript`, which deliberately no longer
        touches them: whatever's removed here, the user checked off
        individually in the delete dialog.

        Runs exclusively via the identity (abs_path). Cleaning up
        via rel_path would also hit the identically named file from another root
        (docs/PATHS.md) — and deletion is not undoable.
        """
        empty = {"summaries": 0, "translations": 0, "edits": 0}
        target = self._identity(rel_path, media_dir)
        if not target:
            logger.warning(
                "delete_derivatives without a derivable identity (%s) — nothing will be deleted",
                rel_path,
            )
            return empty
        out = dict(empty)
        with self.connection() as conn:
            for name, table, enabled in (
                ("summaries", "summaries", summaries),
                ("translations", "translations", translations),
                ("edits", "llm_edits", edits),
            ):
                if not enabled:
                    continue
                cur = conn.execute(  # noqa: S608 — table name from a fixed list
                    f"DELETE FROM {table} WHERE abs_path = ?", (target,)
                )
                out[name] = cur.rowcount or 0
        return out

    def delete_translation(
        self, rel_path: str, language: str, media_dir: str | None = None
    ) -> bool:
        target = self._identity(rel_path, media_dir)
        if not target:
            return False
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM translations WHERE abs_path = ? AND language = ?", (target, language)
            )
            return cur.rowcount > 0

    # ── LLM correction runs (v16) ─────────────────────────────────────────

    def save_llm_edit_run(
        self,
        rel_path: str,
        payload: dict,
        model: str = "",
        status: str = "pending",
        media_dir: str | None = None,
    ) -> int:
        """Save a correction run; replaces an open pending run
        for the same file (there's at most ONE open review)."""
        target = self._identity(rel_path, media_dir)
        with self.connection() as conn:
            if target and status == "pending":
                conn.execute(
                    "DELETE FROM llm_edits WHERE abs_path = ? AND status = 'pending'", (target,)
                )
            cur = conn.execute(
                "INSERT INTO llm_edits (abs_path, rel_path, status, "
                "payload_json, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    target,
                    rel_path,
                    status,
                    json.dumps(payload, ensure_ascii=False),
                    model,
                    time.time(),
                ),
            )
            return int(cur.lastrowid or 0)

    def get_llm_edit_runs(
        self, rel_path: str, status: str | None = None, media_dir: str | None = None
    ) -> list:
        """Correction runs of a file (newest first)."""
        with self.connection() as conn:
            target = self._identity(rel_path, media_dir)
            if not target:
                return []
            sql = (
                "SELECT id, status, payload_json, model, created_at "
                "FROM llm_edits WHERE abs_path = ?"
            )
            params: list = [target]
            if status:
                sql += " AND status = ?"
                params.append(status)
            sql += " ORDER BY created_at DESC"
            out = []
            for r in conn.execute(sql, params).fetchall():
                try:
                    payload = json.loads(r["payload_json"])
                except (TypeError, json.JSONDecodeError):
                    payload = {}
                out.append(
                    {
                        "id": r["id"],
                        "status": r["status"],
                        "payload": payload,
                        "model": r["model"],
                        "created_at": r["created_at"],
                    }
                )
            return out

    def update_llm_edit_run(self, run_id: int, *, status: str, payload: dict | None = None) -> bool:
        with self.connection() as conn:
            if payload is not None:
                cur = conn.execute(
                    "UPDATE llm_edits SET status = ?, payload_json = ? WHERE id = ?",
                    (status, json.dumps(payload, ensure_ascii=False), run_id),
                )
            else:
                cur = conn.execute("UPDATE llm_edits SET status = ? WHERE id = ?", (status, run_id))
            return cur.rowcount > 0

    @staticmethod
    def _rel_path_from_json(fp: Path, transcript_dir: Path) -> str:
        """Derive rel_path from the JSON path when the JSON doesn't name one.

        Two storage locations, two forms:

        * Central transcript folder — the filename is encoded FLAT, the
          separators of the original path appear as ``__`` in it (see
          ``core/helpers.transcript_path_for``). These are converted back,
          otherwise ``sub__f.mp4.json`` would produce the unusable rel_path
          ``sub__f.mp4``, which no lookup would ever find again.
        * Local ``.transcripts/`` — there the plain filename sits
          next to the media file, no subpath is encoded.

        ``transcript_path_for`` shortens very long names to ``<hash>_<remainder>``.
        The original path can't be recovered from that; such files
        need the ``filename`` field in the JSON and aren't guessed here.
        """
        rel = paths.rel_from(fp, transcript_dir)
        if rel is None:
            rel = fp.name
        if rel.endswith(".json"):
            rel = rel[:-5]
        # Only in the flat main folder is "__" an encoded separator; in a
        # local .transcripts/ it would be part of the real filename.
        if "/" not in rel and "__" in rel:
            rel = rel.replace("__", "/")
        return rel

    def _set_sync_marker(self, key: str, value: float) -> None:
        """Record the timestamp of the last sync (survives a restart)."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                    (key, str(value)),
                )
        except Exception as e:  # noqa
            logger.debug("Sync marker not set: %s", e)

    @staticmethod
    def _abs_for_json(fp: Path, rel_path: str, media_dir: str, from_json_field: bool) -> str:
        """Determine the actual location of the media file for a transcript JSON.

        **The JSON's storage location takes priority**, provided the media file
        is actually next to it: a ``.transcripts/x.mp4.json`` belongs to
        ``../x.mp4``. That's a statement about the disk's actual current state and
        thus the strongest source available.

        The ``filename`` field in the JSON is NOT reliable. It was written during
        transcription relative to the THEN-CURRENT ``media_dir``, and that
        can be switched at runtime. Anyone who had a subfolder open
        at the time ends up with a bare filename there:

            JSON  /nas/videos/recordings/.transcripts/meeting_2024-03-01.mp4.json
            field  "filename": "meeting_2024-03-01.mp4"      <- just the name
            -> abs_from(field, "/nas") = "/nas/meeting_2024-03-01.mp4"

        This is exactly how the index run produced many rows with an identity
        under which nothing exists (docs/PATHS.md). The field now serves only
        as a fallback for the central collection folder, where the JSON isn't
        next to its media file and the storage location therefore reveals nothing.
        """
        # 1. Is the media file next to the JSON? Then that's the answer.
        if fp.name.endswith(".json") and fp.parent.name == ".transcripts":
            beside = paths.abs_from(fp.name[:-5], str(fp.parent.parent))
            if beside and Path(beside).exists():
                return beside
            # The media file is gone (moved, deleted) — the JSON's location
            # still says where it was, IF the field agrees with it: the
            # field is relative to the media root of THEN, so it must be a
            # path suffix of the location ("series/f1.mp4" of
            # /nas/series/f1.mp4). Computed against the folder auto-sync
            # passes, it became /nas/series/series/f1.mp4 — a phantom. A field
            # that is NOT such a suffix belongs to a central folder at the
            # root (subpath "voice_memos/note.aac" in
            # /media/.transcripts/note.aac.json) — step 2 handles that one.
            field = str(rel_path).replace("\\", "/").strip("/")
            if beside and (not from_json_field or beside == field or beside.endswith("/" + field)):
                return beside

        # 2. Otherwise compute the filename field against the root — the case of
        #    the central collection folder, where the storage location reveals nothing.
        #
        #    Deliberately WITHOUT an existence check: a temporarily unreachable
        #    NAS would otherwise cause transcripts to never make it into
        #    the database at all. The damage didn't come from here, but
        #    from this branch running BEFORE the location determination.
        if from_json_field and media_dir:
            return paths.abs_from(rel_path, media_dir)

        # 3. No usable field: last attempt via the name.
        name = PurePosixPath(str(rel_path).replace("\\", "/")).name
        if not name:
            name = fp.name[:-5] if fp.name.endswith(".json") else fp.name
        beside = paths.abs_from(name, str(fp.parent.parent))
        if beside and Path(beside).exists():
            return beside
        return paths.abs_from(rel_path, media_dir) if media_dir else ""

    def sync_from_json_dir(
        self, transcript_dir: Path, progress_callback=None, media_dir: str = ""
    ) -> dict:
        """Import all JSON transcripts from a directory into the DB.

        The FTS triggers stay active during the import, so the search index
        remains consistent even on a crash mid-import — no rebuild needed.
        A JSON whose content is already in the database is not rewritten
        (see :meth:`_merge_with_stored`).

        Returns:
            {"imported": int, "skipped": int, "errors": int}
        """
        result = {"imported": 0, "skipped": 0, "errors": 0}
        if not transcript_dir.exists():
            return result

        # No rglob: that would also descend into hidden directories —
        # on ZFS shares that means .zfs/snapshot/, where the complete tree
        # exists again per snapshot. Instead, os.walk with pruning: hidden
        # folders are cut off before descending; only .transcripts
        # stays traversable (that's exactly where the transcripts live).
        json_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(transcript_dir, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") or d == ".transcripts"]
            json_files.extend(
                Path(dirpath) / f
                for f in filenames
                if f.endswith(".json")
                and not f.startswith(".")
                # Derivation sidecars are LISTS, not transcripts —
                # they triggered "'list' object has no
                # attribute 'get'" errors during start-up sync.
                and not f.endswith(("_summaries.json", "_translations.json"))
            )
        total = len(json_files)

        if total == 0:
            return result

        # Only touch changed JSONs. The timestamp of the last sync for
        # this directory lives in schema_info and survives a restart —
        # the earlier cache existed only in memory, which is why EVERY server start
        # re-read all transcripts. On a library with thousands of JSONs
        # that meant many seconds and just as many individual transactions, every
        # scan over again.
        sync_key = "sync_mtime:" + paths.canon(transcript_dir)
        last_sync = 0.0
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT value FROM schema_info WHERE key = ?", (sync_key,)
                ).fetchone()
                if row and row["value"]:
                    last_sync = float(row["value"])
        except Exception as _e:  # noqa
            last_sync = 0.0

        # The marker may only move past files that were really handled.
        # A file that couldn't be stat'ed or read (CIFS hiccup) or whose
        # import failed in the database ("database is locked") holds it
        # back — otherwise that transcript would never be retried until its
        # JSON changes again. A JSON that doesn't parse does NOT hold it
        # back: it gets a new mtime when it is written again.
        newest = last_sync
        hold_back: float | None = None
        mtimes: dict[Path, float] = {}
        for fp in json_files:
            try:
                mtimes[fp] = fp.stat().st_mtime
            except OSError:
                # mtime unknown: the marker must not move at all
                hold_back = last_sync
                continue
            newest = max(newest, mtimes[fp])
        if last_sync > 0:
            fresh = [fp for fp in json_files if fp in mtimes and mtimes[fp] > last_sync]
            result["skipped"] += len(json_files) - len(fresh)
            json_files = fresh
            total = len(json_files)
            if total == 0:
                if hold_back is None:
                    self._set_sync_marker(sync_key, newest)
                logger.debug("Auto-sync %s: nothing changed", transcript_dir)
                return result

        # Determine the root. A local .transcripts/ sits NEXT TO its
        # media files — there the folder above is the root for the
        # files, but NOT the root that "filename" in the JSON
        # refers to: that's relative to media-dir (see core/transcription.py).
        # Conflating the two produced /media/sub/sub/b.mp3 — the wrong
        # value then even overwrote the correct one. That's why the
        # actual location is determined per file (see _abs_for_json below) instead of
        # being reconstructed from a guessed root.
        if not media_dir:
            try:
                from whisper_tm.config import config as _cfg

                media_dir = str(_cfg.media_dir)
            except Exception as _e:  # noqa
                media_dir = ""

        # #058: NO global DROP TRIGGER and NO full-text rebuild. The FTS
        # triggers keep the index consistent row by row, even on a crash
        # mid-import. The former 'rebuild' after every sync of more than ten
        # files re-indexed ALL segments under the write lock — 30 s on a
        # library with millions of segments, while the queue lanes wait at most
        # 30 s for that lock. A deliberate rebuild stays in the settings.
        try:
            # Silence the backup counter: otherwise the start-up sync triggers a chain
            # of NAS backups (see bulk_import).
            with self.bulk_import():
                for i, fp in enumerate(json_files):
                    try:
                        with open(fp, encoding="utf-8") as f:
                            data = json.load(f)
                        # Second line of defense behind the filename filter:
                        # what isn't a transcript object isn't an error,
                        # it's simply not our format.
                        if not isinstance(data, dict) or not data.get("complete"):
                            result["skipped"] += 1
                            continue
                        # #084: correctly derive rel_path from the path if the
                        # JSON has no (usable) "filename" field.
                        from_field = bool(data.get("filename"))
                        rel_path = data.get("filename") or self._rel_path_from_json(
                            fp, transcript_dir
                        )
                        abs_path = self._abs_for_json(fp, rel_path, media_dir, from_field)
                        merged = self._merge_with_stored(abs_path, data)
                        if merged is None:
                            # Everything in the JSON is already in the
                            # database — rewriting it would only throw away
                            # what the database alone holds.
                            result["skipped"] += 1
                            continue
                        self.upsert_transcript(
                            rel_path, merged, media_dir=media_dir, abs_path=abs_path
                        )
                        result["imported"] += 1
                    except ValueError as _e:
                        # Not parseable (JSONDecodeError) — permanent until
                        # the file is written again, which gives it a new mtime.
                        result["errors"] += 1
                        logger.warning("JSON import failed for %s: %s", fp, _e)
                    except Exception as _e:  # noqa
                        # #078: log file + reason instead of just counting the error.
                        # Read or database error: possibly transient — retry,
                        # but only a few times for the same file version: a
                        # JSON that always fails (bad segment types) would
                        # otherwise hold the marker forever, and every walk
                        # would re-read all newer JSONs over CIFS.
                        result["errors"] += 1
                        logger.warning("JSON import failed for %s: %s", fp, _e)
                        m = mtimes.get(fp, last_sync)
                        attempt_key = (str(fp), m)
                        with self._sync_failures_lock:
                            attempts = self._sync_failures.get(attempt_key, 0) + 1
                            self._sync_failures[attempt_key] = attempts
                        if attempts >= self._SYNC_MAX_ATTEMPTS:
                            logger.warning(
                                "JSON import of %s failed %d times — giving up "
                                "on it until the file changes",
                                fp,
                                attempts,
                            )
                        else:
                            # Just below this file, never below the old marker
                            cap = max(last_sync, m - 1e-3)
                            hold_back = cap if hold_back is None else min(hold_back, cap)

                    if progress_callback and (i % 50 == 0 or i == total - 1):
                        progress_callback(i + 1, total)
        finally:
            self._set_sync_marker(sync_key, newest if hold_back is None else min(newest, hold_back))

        return result

    #: How often the sync retries a JSON (same file, same mtime) that parses
    #: but fails to import, before the marker moves past it.
    _SYNC_MAX_ATTEMPTS = 3

    #: Values the database sets on top of what the transcript file says: the
    #: run's log (queue), the lane-adjusted compute time (queue). The file
    #: never carries the database's version, so a difference here is no
    #: reason to re-import.
    _DB_OWNED_KEYS = frozenset({"log_file", "processing_secs"})

    def _merge_with_stored(self, abs_path: str, data: dict) -> dict | None:
        """What a re-import of ``data`` may write — or None for "nothing".

        The database is the primary source; values that only it holds
        (``log_file``, the lane-adjusted ``processing_secs``, the
        ``voices`` record in the settings) are never in the JSON. A plain
        upsert of the JSON would erase them.

        * Everything in the JSON already stored (apart from the keys the
          database owns) -> None: the import is skipped, nothing is lost.
        * Otherwise the JSON wins where it says something, and everything
          only the database knows is carried over: keys missing from the
          JSON, the settings keys missing from its settings, and the
          database's processing time when the transcript itself is the same.
        """
        if not abs_path:
            return data
        with self.connection() as conn:
            row = conn.execute(
                "SELECT json_data, settings_json FROM transcripts WHERE abs_path = ?",
                (paths.canon(abs_path),),
            ).fetchone()
        if row is None:
            return data
        stored = _json_or(row["json_data"], None)
        stored = stored if isinstance(stored, dict) else {}
        settings = _json_or(row["settings_json"], None)
        settings = settings if isinstance(settings, dict) else {}

        own_keys = self._DB_OWNED_KEYS
        same = all(stored.get(k) == v for k, v in data.items() if k not in own_keys)
        if same and stored:
            new_settings = data.get("settings")
            if not isinstance(new_settings, dict) or all(
                settings.get(k) == v for k, v in new_settings.items()
            ):
                return None

        merged = {k: v for k, v in stored.items() if k not in data}
        merged.update(data)
        if same:
            for k in own_keys:
                if k in stored:
                    merged[k] = stored[k]
        if settings:
            incoming = data.get("settings")
            if isinstance(incoming, dict):
                merged["settings"] = {**settings, **incoming}
        return merged

    # ── FTS rebuild ───────────────────────────────────────────────────

    def rebuild_fts_index(self) -> None:
        """Rebuild the FTS5 index from scratch."""
        with self.connection() as conn:
            conn.execute("INSERT INTO segments_fts(segments_fts) VALUES('rebuild')")

    # ── Maintenance ───────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Compact the database. Creates a backup first."""
        self.backup()
        conn = self._get_connection()
        # #071: VACUUM must not run inside a transaction — close any
        # open transaction first, otherwise
        # "cannot VACUUM from within a transaction".
        conn.commit()
        conn.execute("VACUUM")

    def backup_target(self) -> Path:
        """Where a backup belongs.

        Only ONE destination, and that's the configured one (`database.backup_path`,
        typically a NAS). A second copy next to the database on
        the same disk is the weakest form of backup there is —
        if the disk fails, both are gone — and it can cost tens of GB.

        If the path is missing from the configuration, it falls back to the
        directory next to the database: a weak backup is better than none.
        """
        try:
            from whisper_tm.config import config as _cfg

            target = _cfg.backup_dir
        except Exception:
            target = None
        return target if target else Path(self._db_path).parent / "backups"

    def remove_backup_leftovers(self) -> int:
        """The staging copies of backups the process died in (a kill, a
        closed terminal, a restart during the start-up backup) — each as big
        as the database, next to it, and never looked at again.

        For the server's start: it holds the database's lock then, so no
        backup of this database is running.
        """
        db_path = Path(self._db_path)
        removed = 0
        for leftover in db_path.parent.glob(f".{db_path.stem}_????????_??????_??????.tmp*"):
            try:
                size = leftover.stat().st_size
                leftover.unlink()
            except OSError as e:
                logger.warning(
                    "Leftover of an interrupted backup not removed (%s): %s", leftover, e
                )
                continue
            removed += 1
            logger.info(
                "Removed the leftover of an interrupted backup: %s (%.1f GB)",
                leftover.name,
                size / 1e9,
            )
        return removed

    def backup_due(self) -> Path | None:
        """Back up IF the last backup is old enough. Otherwise do nothing.

        Return: path of the new backup, or ``None`` if skipped.

        The threshold lives in ``database.backup_min_hours`` (default 24).
        ``0`` means "on every start" — the old behavior, which is deliberately
        no longer the default.
        """
        try:
            from whisper_tm.config import config as _cfg

            hours = _cfg.get("database", "backup_min_hours", default=24)
        except Exception:
            hours = 24
        try:
            hours = float(hours)
        except (TypeError, ValueError):
            hours = 24.0
        target = self.backup_target()
        # Nothing to lose yet: the first start made a copy of an empty database
        try:
            with self.connection() as conn:
                if not self._has_rows(conn):
                    return None
        except sqlite3.Error:
            pass

        if hours > 0:
            stem = Path(self._db_path).stem
            try:
                present = sorted(
                    target.glob(f"{stem}_*.db"), key=lambda f: f.stat().st_mtime, reverse=True
                )
            except OSError as e:
                # Destination unreachable (NAS not mounted). Then do NOT
                # back up blindly — `backup()` reports this cleanly anyway.
                logger.warning("Backup destination unreadable (%s): %s", target, e)
                present = []
            if present:
                age_h = (time.time() - present[0].stat().st_mtime) / 3600
                if age_h < hours:
                    logger.info(
                        "DB backup skipped: the last one is %.1f h old "
                        "(threshold %.0f h, database.backup_min_hours)",
                        age_h,
                        hours,
                    )
                    return None
        return self.backup(dest_dir=target)

    def backup(self, dest_dir: str | Path | None = None) -> Path | None:
        """Create a DB backup (timestamped copy).

        Args:
            dest_dir: target directory. None = default (``<db>/backups``).
                For the external destination (NAS), ``database.backup_path``
                is passed through.

        Respects max_backups from the config — oldest ones are deleted.
        Returns: path to the backup file, or None.
        """
        with self._backup_lock:
            return self._backup(dest_dir)

    def _backup(self, dest_dir: str | Path | None) -> Path | None:
        db_path = Path(self._db_path)
        if not db_path.exists():
            return None

        backup_dir = Path(dest_dir) if dest_dir else db_path.parent / "backups"
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # External destination can be missing (NAS not mounted) — that's a
            # normal operating condition, no reason to raise.
            logger.warning("Backup destination unusable (%s): %s", backup_dir, e)
            return None

        # #082: microseconds in the name — otherwise two backups
        # within the same second collide (the second overwrites the first).
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_dir / f"{db_path.stem}_{ts}.db"

        # SQLite backup API instead of a file copy: copying a WAL DB with
        # copy2 loses all not-yet-checkpointed commits or
        # produces a corrupt backup (torn pages during ongoing writes).
        #
        # BUT: the backup API writes the destination page by page (4 KB). Directly onto
        # a CIFS/NFS share, that's over 100,000 individual
        # network round trips for a 500 MB database — the run makes practically
        # no progress (minutes without a single byte written) and leaves behind a
        # 0-byte file that looks like a finished backup. That's why it's ALWAYS
        # created locally next to the DB and the finished file copied to the
        # destination in one go (a local backup plus one sequential copy).
        import shutil

        staging = db_path.parent / f".{db_path.stem}_{ts}.tmp"
        try:
            src = self._get_connection()
            dst = sqlite3.connect(str(staging))
            try:
                src.backup(dst)
            finally:
                dst.close()
        except Exception as e:
            logger.warning("DB backup failed: %s", e)
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        # Move the finished file to the destination. Same filesystem (local
        # default backup) -> os.replace is atomic and free; across a
        # filesystem boundary (NAS) it raises EXDEV -> copy in one go.
        #
        # Over the network, the copy takes seconds. If the process dies during that time
        # (Ctrl-C, restart), a truncated file used to end up under the
        # final name in the folder — a fraction of the real size, not
        # distinguishable from a valid backup from the outside. That's why it's first
        # copied under a .part name and then renamed in the destination folder:
        # the final name only exists once all bytes are there.
        partial = backup_dir / f".{backup_path.name}.part"
        try:
            try:
                os.replace(staging, backup_path)
            except OSError:
                shutil.copyfile(staging, partial)
                os.replace(partial, backup_path)
        except OSError as e:
            # Destination gone (NAS unmounted), full, or read-only: don't leave
            # anything half-done lying around — it would pass as a valid backup
            # on the next look into the folder.
            logger.warning("Backup could not reach destination (%s): %s", backup_path, e)
            for leftover in (partial, backup_path):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            return None
        finally:
            # Even on KeyboardInterrupt/SIGTERM mid network copy: the
            # local staging file is as big as the DB and must not
            # be left behind. os.replace above may have already consumed it.
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass

        # Rotate old backups
        try:
            from whisper_tm.config import config as _cfg

            max_backups = _cfg.get("database", "max_backups", default=5)
        except Exception as _e:  # noqa
            max_backups = 5
        # #083: guard against invalid values (0, string, None) — otherwise
        # backups[max_backups:] would delete the just-created backup too.
        try:
            max_backups = int(max_backups)
        except (TypeError, ValueError):
            max_backups = 5
        if max_backups < 1:
            max_backups = 1

        # Also clean up remnants of a crashed earlier run: .part files and
        # the -wal/-shm that appear when someone opens a backup to check it.
        # Neither falls under the *.db glob and would otherwise lie around forever.
        for junk in (
            *backup_dir.glob(f".{db_path.stem}_*.db.part"),
            *backup_dir.glob(f"{db_path.stem}_*.db-wal"),
            *backup_dir.glob(f"{db_path.stem}_*.db-shm"),
        ):
            try:
                if junk.name != partial.name:
                    junk.unlink()
            except OSError:
                pass

        backups = sorted(
            backup_dir.glob(f"{db_path.stem}_*.db"), key=lambda f: f.stat().st_mtime, reverse=True
        )
        for old in backups[max_backups:]:
            try:
                old.unlink()
            except Exception as _e:  # noqa
                pass

        return backup_path

    # ── External backup after N transcript changes ────────────────────
    #
    # Purpose: the transcripts are the irreplaceable payload data (in
    # "database" mode they live ONLY in the DB). A counter over transcript
    # changes triggers a backup to an external destination
    # (NAS) at intervals. Deliberately NOT tied to scan/index write
    # operations: an index run over tens of thousands of files would otherwise
    # flood the destination with copies of a DB of several hundred MB.
    _BACKUP_COUNTER_KEY = "backup_change_counter"

    def _backup_settings(self) -> tuple[str, int]:
        """(destination path, threshold) from the config — defensive against garbage."""
        try:
            from whisper_tm.config import config as _cfg

            folder = _cfg.backup_dir
            every = _cfg.get("database", "backup_every_changes", default=0)
        except Exception:
            return "", 0
        try:
            every = int(every)
        except (TypeError, ValueError):
            every = 0
        return (str(folder) if folder else ""), max(0, every)

    def get_backup_counter(self) -> int:
        """Transcript changes since the last external backup."""
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT value FROM user_settings WHERE key = ?", (self._BACKUP_COUNTER_KEY,)
                ).fetchone()
            return int(row["value"]) if row and row["value"] else 0
        except Exception:
            return 0

    @contextmanager
    def bulk_import(self) -> Iterator[None]:
        """Bulk import: do NOT count transcript changes individually.

        The start-up sync pushes thousands of JSONs already on disk into the
        DB. Counted as "changes", this would blow through the backup threshold dozens of
        times over (a few thousand imports / 150 per backup) and back up to the
        NAS continuously for minutes — the same flood that's already deliberately
        avoided for scan/index runs. There's also no data at risk: the transcripts
        still exist as JSON, the DB is only being caught up here.

        Thread-local: a queue worker running in parallel should still
        record its real transcript changes normally.
        """
        self._local.skip_backup_counter = True
        try:
            yield
        finally:
            self._local.skip_backup_counter = False

    def note_transcript_change(self, n: int = 1) -> None:
        """Record one transcript change; trigger an external backup when the
        threshold is reached.

        Called from the write paths and must therefore NEVER
        raise and NEVER block: the actual backup runs in a
        background thread, otherwise the queue worker would stall for the
        duration of a copy over the network.
        """
        if getattr(self._local, "skip_backup_counter", False):
            return
        path, every = self._backup_settings()
        if not path or every <= 0 or n <= 0:
            return
        try:
            with self.connection() as conn:
                row = conn.execute(
                    """
                    INSERT INTO user_settings (key, value, updated_at)
                    VALUES (?, ?, unixepoch())
                    ON CONFLICT(key) DO UPDATE SET
                        value = CAST(CAST(value AS INTEGER) + ? AS TEXT),
                        updated_at = unixepoch()
                    RETURNING value
                """,
                    (self._BACKUP_COUNTER_KEY, str(n), n),
                ).fetchone()
            count = int(row["value"]) if row and row["value"] else 0
        except Exception as e:
            logger.debug("Backup counter could not be advanced: %s", e)
            return

        if count < every:
            return

        # Threshold reached: reset the counter and back up. The lock
        # prevents two simultaneously finished transcriptions from writing two
        # parallel copies of the same DB to the NAS.
        with self._ext_backup_lock:
            if self._ext_backup_running:
                return  # already running — no duplicate
            if self.get_backup_counter() < every:
                return  # another thread was faster
            self._reset_backup_counter()
            self._ext_backup_running = True

        t = threading.Thread(
            target=self._run_external_backup, args=(path,), name="db-backup", daemon=True
        )
        t.start()

    def _reset_backup_counter(self) -> None:
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO user_settings (key, value, updated_at)
                    VALUES (?, '0', unixepoch())
                    ON CONFLICT(key) DO UPDATE SET value='0',
                        updated_at = unixepoch()
                """,
                    (self._BACKUP_COUNTER_KEY,),
                )
        except Exception as e:
            logger.debug("Backup counter could not be reset: %s", e)

    def _run_external_backup(self, path: str) -> None:
        """Backup thread: backs up to the external destination and clears the flag."""
        try:
            t0 = time.time()
            bp = self.backup(path)
            if bp:
                logger.info("DB backup (external): %s . %.1fs", bp, time.time() - t0)
            else:
                # Destination unreachable (NAS gone): do NOT bump the
                # counter back up — the next time the threshold is reached
                # it'll be retried anyway.
                logger.warning("External DB backup failed (destination: %s)", path)
        except Exception as e:
            logger.warning("External DB backup failed: %s", e)
        finally:
            with self._ext_backup_lock:
                self._ext_backup_running = False

    def get_db_size(self) -> int:
        """Total DB size in bytes (including WAL and SHM files, #079)."""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(self._db_path + suffix)
            if p.exists():
                total += p.stat().st_size
        return total

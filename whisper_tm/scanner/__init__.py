"""
Scanner: scan media directory, determine status, load metadata.

Public API:
    scan_media_directory(vdir, ...) → ScanResult
    iter_media_files(root, ...) → Iterator[Path]
    resolve_transcript_status(rel_str, abs_path, db) → (complete, partial)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.core.helpers import format_size, load_skiplist
from whisper_tm.scanner.filesystem import (
    WalkAborted,
    build_file_entry,
    collect_media_files,
    iter_media_files,
)
from whisper_tm.scanner.status import determine_file_status, resolve_transcript_status

__all__ = [
    "scan_media_directory",
    "scan_media_directory_from_db",
    "db_row_to_entry",
    "iter_media_files",
    "determine_file_status",
]

logger = logging.getLogger(__name__)

# Serialize background DB work (auto-sync + heal) process-wide so
# multiple daemon threads don't write to SQLite simultaneously (#123 —
# "database is locked"). Applies across all scan threads.
_BG_DB_LOCK = threading.Lock()
# Per .transcripts/ directory the most recently synced JSON mtime,
# so auto-sync skips unchanged directories instead of reimporting all
# transcripts on EVERY scan (#115).
_SYNC_MTIME_CACHE: dict[str, float] = {}


@dataclass
class ScanResult:
    """Result of a directory scan."""

    directory: str
    files: list[dict] = field(default_factory=list)
    total: int = 0
    done: int = 0
    recursive: bool = False
    ultra: bool = False
    transcript_dirs_found: int = 0
    stats: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    source: str = "walk"  # "walk" = filesystem, "db" = DB entries only
    orphans: int = 0  # orphaned transcripts (file gone, transcript there)
    error: str | None = None


def scan_media_directory(
    vdir: Path | None = None,
    *,
    recursive: bool = False,
    ultra: bool = False,
    db=None,
    on_progress=None,
    should_stop=None,
) -> ScanResult:
    """Scan a media directory and return files with status.

    DB-first strategy for performance with 50k+ files:
    1. Load completion set from DB (1 query → set lookup per file)
    2. Load transcript metadata as a batch from DB (1 query)
    3. Filesystem scan only for file discovery, not for status
    4. JSON fallback only when no DB is present

    Args:
        vdir: directory (default: config.media_dir)
        recursive: include subdirectories
        ultra: ultra mode (higher limits)
        db: database instance for the status cache
        on_progress: optional callback for live progress (SSE stream).
            Receives dicts:
              {"phase": "walk", "files": n, "dirs_done": d, "dirs_known": k,
               "errors": e}   (+ waiting/wait_attempt/wait_secs/wait_reason,
                               when the walk is waiting on a dead network drive)
              {"phase": "db"}                      (DB preload starts)
              {"phase": "process", "done": n, "total": t}
            Note: the process events come from pool threads.
        should_stop: optional callable → True cleanly aborts the scan
            (e.g. an SSE client closed the connection).

    Returns:
        ScanResult with the file list and statistics
    """
    if vdir is None:
        vdir = config.media_dir

    result = ScanResult(directory=str(vdir), recursive=recursive, ultra=ultra)

    if not vdir.exists():
        result.error = f"Directory not found: {vdir}"
        return result

    skiplist = load_skiplist()

    # Track discovered transcript dirs for auto-DB-add
    found_transcript_dirs: list[Path] = []

    def _on_transcript_dir(path: Path):
        found_transcript_dirs.append(path)

    # ── Walk ONCE, materialize ────────────────────────────────────
    # The directory walk is the expensive part on NAS/SMB → walk it only ONCE
    # (not again for the DB keys) and IN PARALLEL (one directory read
    # per worker hides network latency — see collect_media_files).
    _t_walk = time.monotonic()
    _walk_prog = None
    if on_progress:
        # Pass walk progress (incl. waiting state on NAS outage) through
        # 1:1 as the walk phase.
        _walk_prog = lambda st: on_progress({"phase": "walk", **st})  # noqa: E731
    try:
        scanned = collect_media_files(
            vdir,
            recursive=recursive,
            on_transcript_dir=_on_transcript_dir,
            on_progress=_walk_prog,
            should_stop=should_stop,
        )
    except WalkAborted as e:
        # Dead network drive or similar: a clear error instead of silently
        # grinding on.
        result.error = str(e)
        return result

    if should_stop and should_stop():
        # Client gone (SSE closed): don't even start the expensive follow-up
        # phases (DB load, per-file stat, bulk upsert).
        result.error = "Scan aborted"
        return result
    walk_secs = time.monotonic() - _t_walk

    # ── Keys of the found files (for the scoped DB query) ──────────
    rel_keys: set[str] = set()
    abs_keys: set[str] = set()
    for fp in scanned:
        rk = paths.rel_from(fp, vdir)
        if rk is None:
            rk = fp.name
        rel_keys.add(rk)
        abs_keys.add(paths.canon(fp))

    # ── DB bulk preload (SCOPED to the found files) ─────────────────
    # Previously EVERY scan loaded the entire transcripts/media_files table —
    # for large libraries, many seconds PER opened folder, no matter how
    # small the folder is. Now only the rows of the files actually
    # scanned. With very many files (large recursive scan) the unscoped
    # full load is more efficient than tens of thousands of IN parameters
    # → fallback.
    completion_set: set[str] | None = None
    meta_batch: dict[str, dict] = {}
    media_meta_batch: dict[str, dict] = {}  # cached FFprobe metadata
    index_state: dict[str, tuple] = {}  # rel_path/abs_path → (indexed_at, mtime)
    scope_limit = 4000
    scoped = bool(db) and len(rel_keys) <= scope_limit
    if on_progress:
        on_progress({"phase": "db"})
    _t_db = time.monotonic()
    if db:
        try:
            if scoped:
                abs_list = list(abs_keys)
                key_list = list(abs_keys | rel_keys)
                # Transcript state and metadata via IDENTITY
                # (docs/PATHS.md), not via the display value.
                completion_set = db.get_completion_set(abs_paths=abs_list)
                meta_batch = db.get_transcript_meta_batch(abs_paths=abs_list)
                media_meta_batch = db.get_media_meta_batch(keys=key_list)
                index_state = db.get_index_state_batch(keys=key_list)
            else:
                completion_set = db.get_completion_set()
                meta_batch = db.get_transcript_meta_batch()
                media_meta_batch = db.get_media_meta_batch()
                index_state = db.get_index_state_batch()
        except Exception as e:
            logger.debug("DB preload failed, falling back to per-file: %s", e)
            completion_set = None
    db_secs = time.monotonic() - _t_db

    indexed_count = 0  # how many of the scanned files are FFprobe-indexed

    # ── Process files ────────────────────────────────────────────
    _t_proc = time.monotonic()
    files = []
    media_file_batch = []  # for bulk DB update
    # JSON-only transcripts → backfill into DB. (rel_path, abs_path) instead of
    # finished metadata: the JSON is only read during the background heal (#124).
    heal_batch: list[tuple[str, Path]] = []

    # The I/O-heavy parts PER FILE — build_file_entry (fp.stat()) and the
    # status detection (several .exists() checks in non-DB mode) — are each
    # a network round trip on NAS/SMB. SERIAL was THE bottleneck
    # (~100 ms/file → minutes for thousands of files). Doing this in parallel
    # via a thread pool hides the latency (the NAS serves dozens of accesses
    # concurrently). Only READ-ONLY shared state (vdir/skiplist/
    # completion_set) → thread-safe. Assembling the results afterwards
    # (in-memory lookups, dedup, counters) stays serial & deterministic
    # (map preserves order).
    def _pre(fp):
        entry = build_file_entry(fp, vdir, skiplist=skiplist)
        if entry is None:
            return None
        abs_path = entry.pop("_abs_path")
        rel_name = entry["name"]
        complete, has_partial = resolve_transcript_status(
            rel_name,
            abs_path,
            db=db,
            completion_set=completion_set,
            scan_root=vdir,
        )
        return (entry, abs_path, rel_name, complete, has_partial)

    # Progress of the processing phase (exact numbers — total is known).
    # Counter + throttle under lock, the callback itself outside (comes from
    # the pool threads; the consumer (SSE queue) is thread-safe).
    _pp = {"n": 0, "t": 0.0}
    _pp_lock = threading.Lock()

    def _pre_counted(fp):
        r = _pre(fp)
        fire = False
        with _pp_lock:
            _pp["n"] += 1
            now = time.monotonic()
            if now - _pp["t"] >= 0.25 or _pp["n"] == len(scanned):
                _pp["t"] = now
                fire = True
            n = _pp["n"]
        if fire:
            on_progress({"phase": "process", "done": n, "total": len(scanned)})
        return r

    _pre_fn = _pre_counted if on_progress else _pre
    scan_workers = int(config.get("scanner", "scan_workers", default=32) or 1)
    if scan_workers > 1 and len(scanned) > 4:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=scan_workers) as _pool:
            pre_results = list(_pool.map(_pre_fn, scanned))
    else:
        pre_results = [_pre_fn(fp) for fp in scanned]

    for pre in pre_results:
        if pre is None:
            continue
        entry, abs_path, rel_name, complete, has_partial = pre
        status = determine_file_status(complete, has_partial, entry.pop("skipped"))
        entry["status"] = status
        entry["size_fmt"] = format_size(entry["size"])

        # Transcript metadata from the batch (no per-file query/JSON read)
        if complete:
            # Look up via identity. Neither basename nor rel_path: both
            # showed the metadata of a same-named file from another
            # folder (docs/PATHS.md).
            tx_meta = meta_batch.get(paths.canon(abs_path))
            if not tx_meta:
                # Transcript exists only as JSON (legacy data).
                if db is not None:
                    # #124: do NOT read the JSON in the request path — only
                    # queue it for the background heal (which reads it and
                    # merges it into the DB). Display metadata comes from
                    # the DB on the next scan (then it's the primary source).
                    if len(heal_batch) < 500:
                        heal_batch.append((rel_name, abs_path))
                else:
                    # No DB → JSON is the primary source: read synchronously
                    # (fallback).
                    tx_meta, _tdata = _load_json_meta(rel_name, abs_path)
            if tx_meta:
                entry.update(tx_meta)

        # Attach cached media metadata (duration etc. WITHOUT FFprobe).
        # For ALL files — this way even non-transcribed files show their
        # duration. Look up via the absolute path first (stable across
        # media_dir changes: parent indexed → child recognizes it), then
        # fall back to rel_path.
        abs_str = paths.canon(abs_path)
        # ONLY via identity. The earlier rel_path fallback delivered the
        # FFprobe data of a same-named file from a different root —
        # duration, resolution and codecs of the wrong file, and the file
        # was also counted as indexed (indexed_count), so the "not indexed"
        # hint was missing (docs/PATHS.md).
        mm = media_meta_batch.get(abs_str)
        if mm:
            entry.update(mm)
            indexed_count += 1
            # Provide duration for display if there's no transcript
            if not entry.get("audio_duration") and mm.get("media_duration_sec"):
                entry["media_duration_sec"] = mm["media_duration_sec"]
        else:
            entry["indexed"] = False

        files.append(entry)

        # Collect for bulk DB update.
        # Do NOT create a duplicate if the file already exists in
        # media_files under its abs_path (with a DIFFERENT rel_path, e.g.
        # from the parent scan) — otherwise the table grows with every
        # subfolder visit and counts/index-skips get distorted.
        # #112: check across ALL known rows (index_state), not just the
        # FFprobe-indexed ones (media_meta_batch). Otherwise dedup doesn't
        # catch purely-scanned rows (indexed_at=0) and a media_dir change
        # creates duplicate rows.
        covered_by_abs = abs_str in index_state
        has_rel_row = rel_name in index_state
        if db and not (covered_by_abs and not has_rel_row):
            media_file_batch.append(
                {
                    "rel_path": rel_name,
                    "abs_path": abs_str,
                    "display_name": entry["display_name"],
                    "subfolder": entry["subfolder"],
                    "extension": entry["ext"],
                    "is_video": entry["is_video"],
                    "file_size": entry["size"],
                    "modified_at": entry["modified"],
                    "status": status,
                }
            )

    proc_secs = time.monotonic() - _t_proc

    # ── Bulk DB update (1 query instead of N) ───────────────────────
    _t_up = time.monotonic()
    if db and media_file_batch:
        try:
            db.bulk_upsert_media(media_file_batch)
            # Refresh statistics if the dataset has grown noticeably.
            # Without current statistics SQLite picks a catastrophic plan
            # for the file list (seconds instead of milliseconds, see Database).
            db.refresh_statistics()
        except Exception as e:
            # Never quietly: a failed write keeps every new file of this
            # scan out of the library. One row at a time then, so a single
            # bad row costs only itself.
            logger.warning(
                "The scan's files could not be stored in one go (%s) — storing them one at a time",
                e,
            )
            lost = 0
            for row in media_file_batch:
                try:
                    db.bulk_upsert_media([row])
                except Exception as e2:
                    lost += 1
                    if lost <= 5:
                        logger.warning(
                            "Not stored: %s (%s)",
                            str(row.get("abs_path", ""))
                            .encode("utf-8", "backslashreplace")
                            .decode(),
                            e2,
                        )
            if lost:
                logger.warning(
                    "%d of %d files of this scan could not be stored", lost, len(media_file_batch)
                )

    # ── Walk cleanup: remove vanished files from DB + queue ─────────
    # Only if this walk saw the tree COMPLETELY: recursive and not
    # aborted (otherwise we'd have exited long ago). There's no limit
    # anymore — a walk that ran to completion is always complete.
    # Otherwise rows of deleted/renamed files would stay "open" forever
    # and keep wandering back into the queue
    # (e.g. thousands of dead SMB alias paths).
    if db and recursive and scanned:
        try:
            vanished, _newly_orphaned = db.clean_orphaned_media(vdir, abs_keys)
            if vanished:
                from whisper_tm.core import queue as _queue

                _queue.remove_pending_for(vanished)
        except Exception as e:
            logger.debug("Walk cleanup failed: %s", e)
    up_secs = time.monotonic() - _t_up

    # Make the timing visible when a scan takes noticeably long — shows exactly
    # where the time goes (NAS walk vs. DB load vs. FFprobe enrichment vs. upsert).
    total_secs = walk_secs + db_secs + proc_secs + up_secs
    if total_secs > 3.0:
        logger.info(
            "Scan %s: %.1fs total — walk %.1fs, DB load %.1fs (%s), "
            "processing %.1fs, upsert %.1fs · %d files",
            vdir,
            total_secs,
            walk_secs,
            db_secs,
            f"scoped {len(rel_keys)} keys" if scoped else "full",
            proc_secs,
            up_secs,
            len(scanned),
        )

    # ── Statistics ───────────────────────────────────────────────
    total_size = sum(f["size"] for f in files)
    done_files = [f for f in files if f["status"] == "done"]
    done_count = len(done_files)
    total_proc_sec = sum(f.get("processing_secs", 0) for f in done_files)
    # Total audio duration: transcript duration preferred, else cached index duration
    total_audio_sec = 0.0
    for f in files:
        d = f.get("audio_duration_sec", 0) or f.get("media_duration_sec", 0) or 0
        total_audio_sec += d

    # ── Append orphans (transcript there, file gone) ─────────────
    # The walk only sees existing files — orphaned rows would therefore
    # never appear in the list and would be invisible. They are appended
    # from the DB so that the ghost filter shows them in BOTH modes
    # (walk and DB): review them, then delete deliberately.
    if db:
        try:
            for r in db.get_media_files_under(str(vdir), status="orphan"):
                # Seen by this walk: the file is back and already listed
                if r["abs_path"] in abs_keys:
                    continue
                entry = db_row_to_entry(r, str(vdir))
                if entry is not None:
                    files.append(entry)
                    result.orphans += 1
        except Exception as e:
            logger.debug("Appending orphans failed: %s", e)

    result.files = files
    result.total = len(files)
    result.done = done_count
    result.transcript_dirs_found = len(found_transcript_dirs)
    result.stats = {
        "total_size": total_size,
        "total_size_fmt": format_size(total_size),
        "done_pct": round(done_count / len(files) * 100, 1) if files else 0,
        "total_audio_sec": round(total_audio_sec, 1),
        "total_proc_sec": round(total_proc_sec, 1),
    }
    # Index coverage (for popup decision + GUI statistics)
    result.index = {
        "total": len(files),
        "indexed": indexed_count,
        "unindexed": max(0, len(files) - indexed_count),
        "indexed_pct": round(indexed_count / len(files) * 100, 1) if files else 100,
    }

    # ── Background: auto-sync (.transcripts/) + DB heal (JSON-only) ──────
    # #123: previously two uncoordinated daemon threads writing to SQLite in
    # parallel ("database is locked"). Now ONE thread whose DB work is
    # serialized via a process-wide lock (also against concurrent scans).
    auto_sync = bool(
        config.get("database", "auto_add_dirs", default=True) and found_transcript_dirs
    )
    if db and (auto_sync or heal_batch):

        def _background():
            with _BG_DB_LOCK:
                # Auto-sync: only import .transcripts/ directories whose JSONs
                # have changed since the last sync (#115 mtime skip) — no
                # longer ALL transcripts on EVERY scan.
                # sync_from_json_dir no longer drops FTS triggers globally
                # (#058), so it can be called normally.
                if auto_sync:
                    for tdir in found_transcript_dirs:
                        try:
                            # Key canonically: the same folder under two
                            # spellings would otherwise get two entries.
                            key = paths.canon(tdir) or str(tdir)
                            newest = _newest_json_mtime(tdir)
                            if newest and newest <= _SYNC_MTIME_CACHE.get(key, 0.0):
                                continue  # nothing changed → skip sync
                            # Pass the ROOT along. Without it,
                            # sync_from_json_dir falls back to config.media_dir —
                            # that may have changed between scan start and
                            # this background thread. "series/f1.mp4" then became
                            # "/nas/video/series/series/f1.mp4": a path that
                            # doesn't exist, in the identity column
                            # (docs/PATHS.md).
                            res = db.sync_from_json_dir(tdir, media_dir=str(vdir))
                            # Only remember the folder as done when every
                            # file went through — otherwise this cache would
                            # skip the retry the sync marker keeps open.
                            if not res.get("errors"):
                                _SYNC_MTIME_CACHE[key] = newest
                        except Exception as e:
                            logger.debug("Auto-sync failed: %s", e)
                # DB heal: merge JSON-only transcripts discovered during the
                # scan into the DB (DB = primary source). The JSON is read
                # only HERE (#124), the request path stays unblocked.
                if heal_batch:
                    healed = 0
                    for rel, ab in heal_batch:
                        try:
                            _meta, tdata = _load_json_meta(rel, ab)
                            if tdata is not None:
                                # The actual location is known (ab) — pass it
                                # along instead of letting upsert_transcript
                                # reconstruct it from rel_path + root
                                # (docs/PATHS.md §Writing).
                                db.upsert_transcript(
                                    rel, tdata, media_dir=str(vdir), abs_path=paths.canon(ab)
                                )
                                healed += 1
                        except Exception as e:
                            logger.debug("DB heal %s failed: %s", rel, e)
                    if healed:
                        logger.info("DB heal: %d JSON transcripts merged into DB", healed)

        threading.Thread(target=_background, daemon=True).start()

    return result


def scan_media_directory_from_db(
    vdir: Path | None = None,
    *,
    db=None,
) -> ScanResult:
    """Load the file list ONLY from the DB — NO filesystem walk.

    For the auto-walk-off mode: opening a directory (even the root of a
    huge NAS library) costs a single SQL query instead of minutes of
    directory traversal. The list reflects the last walk's
    state — new/deleted files appear only after the next
    manual refresh (walk) or index run.

    Returns the SAME entry/result shape as scan_media_directory, so the
    frontend doesn't need a second code path.
    """
    if vdir is None:
        vdir = config.media_dir
    vdir = Path(vdir)

    result = ScanResult(directory=str(vdir), recursive=True, ultra=True, source="db")
    if db is None:
        result.error = "No database configured"
        return result

    try:
        rows = db.get_media_files_under(str(vdir))
    except Exception as e:
        logger.exception("DB scan failed")
        result.error = f"DB scan failed: {e}"
        return result

    files: list[dict] = []
    indexed_count = 0

    for r in rows:
        entry = db_row_to_entry(r, str(vdir))
        if entry is None:
            continue
        if entry["indexed"]:
            indexed_count += 1
        files.append(entry)

    files.sort(key=lambda f: f["name"])
    result.orphans = sum(1 for f in files if f["status"] == "orphan")

    # ── Statistics (same shape as the walk scan) ──────────────────
    total_size = sum(f["size"] for f in files)
    done_count = sum(1 for f in files if f["status"] == "done")
    total_proc_sec = sum(f.get("processing_secs", 0) or 0 for f in files if f["status"] == "done")
    total_audio_sec = sum(
        (f.get("audio_duration_sec", 0) or f.get("media_duration_sec", 0) or 0) for f in files
    )

    result.files = files
    result.total = len(files)
    result.done = done_count
    result.stats = {
        "total_size": total_size,
        "total_size_fmt": format_size(total_size),
        "done_pct": round(done_count / len(files) * 100, 1) if files else 0,
        "total_audio_sec": round(total_audio_sec, 1),
        "total_proc_sec": round(total_proc_sec, 1),
    }
    result.index = {
        "total": len(files),
        "indexed": indexed_count,
        "unindexed": max(0, len(files) - indexed_count),
        "indexed_pct": round(indexed_count / len(files) * 100, 1) if files else 100,
    }
    return result


def register_single_file(abs_path_arg, db=None, root=None) -> str | None:
    """Register ONE freshly placed file (upload/download) into the DB right away.

    Replaces the earlier full walk after every upload: for a
    large NAS library, a single new file used to trigger a
    complete directory scan. Returns the rel_path relative to the root
    ``root`` (default: the current media root) — or None if the
    file lies outside it / the DB is missing. The entry is unique via
    ``abs_path``; the display derives its path from abs_path
    (db_row_to_entry), the rel_path is just the caller's key.
    """
    from whisper_tm.config import config

    if db is None:
        from whisper_tm.extensions import db as _db

        db = _db
    if not db:
        return None
    p = Path(abs_path_arg)
    try:
        st = p.stat()
    except OSError:
        return None
    root = str(root if root is not None else config.media_dir)
    rel = paths.rel_from(p, root)
    if not rel:
        return None
    ext = p.suffix.lower()
    # Read the title from the container tags right away (ONE file, one quick
    # ffprobe) — otherwise the indexer would only fetch it on the next run.
    title = ""
    try:
        from whisper_tm.core.audio_utils import get_media_title_fast

        title = get_media_title_fast(p)
    except Exception as e:  # noqa — title is a nicety, not a must
        logger.debug("Metadata title: %s", e)
    db.upsert_media_file(
        rel,
        abs_path=paths.canon(p),
        display_name=p.name,
        # rel has "/" on every platform; str(Path(…)) would give "\\" on Windows
        subfolder=rel.rsplit("/", 1)[0] if "/" in rel else "",
        extension=ext,
        is_video=int(ext in config.video_extensions),
        file_size=st.st_size,
        modified_at=st.st_mtime,
        media_title=title,
        status="pending",
    )
    return rel


def db_row_to_entry(r: dict, root: str) -> dict | None:
    """A DB row (columns from ``_MEDIA_SCAN_SELECT``) → list entry.

    Shared by the full scan (scan_media_directory_from_db) and the
    incremental single-file update (/api/scan_entry). Both paths MUST
    produce EXACTLY the same entry shape — otherwise a freshly
    reloaded row would differ from the same row after a full scan.

    ``root`` is the directory the display path is relative to.
    Rows outside it (or the root itself) return None.
    """
    import json as _json

    # The display path is derived, not truncated via character
    # arithmetic (docs/PATHS.md): unvalidated slicing silently produced
    # a broken-off path for a row outside the root.
    rel = paths.rel_from(r["abs_path"], root)
    if not rel:
        return None

    # Transcript join hit? (tx_date is t.updated_at — only set on a hit)
    has_tx = r.get("tx_date") is not None
    status = r.get("status") or "pending"
    # DB status can lag behind a freshly imported transcript
    # — the join is the more reliable source for "done". 'orphan' stays
    # though: transcript present, file gone — that's exactly what the status says.
    if has_tx and status not in ("skipped", "orphan"):
        status = "done"

    entry = {
        "name": rel,
        # Identity for the browser: the running queue file is recognized
        # via abs_path, not via the root-dependent name.
        "abs_path": r["abs_path"],
        # Deliberately string arithmetic instead of pathlib: rel is already
        # canonical (/-separated). A Path object per field and row cost
        # over a second for hundreds of thousands of files, just to strip off a name.
        "display_name": r.get("display_name") or rel.rpartition("/")[2],
        "metadata_title": r.get("media_title") or "",
        "subfolder": rel.rpartition("/")[0],
        "size": r.get("file_size") or 0,
        "size_fmt": format_size(r.get("file_size") or 0),
        "ext": r.get("extension")
        or ("." + rel.rpartition(".")[2].lower() if "." in rel.rpartition("/")[2] else ""),
        "is_video": bool(r.get("is_video")),
        "modified": r.get("modified_at") or 0,
        "status": status,
        "skip_reason": r.get("skip_reason") or "",
    }

    # Derivatives without original: only set when there are any — otherwise
    # the field would ride along for nothing across hundreds of thousands of entries. Without
    # a transcript it's the only way the list knows something is still
    # attached to the file.
    if r.get("deriv_abs") and not has_tx:
        entry["has_deriv"] = True

    # Cached FFprobe metadata (same field names as get_media_meta_batch)
    if (r.get("indexed_at") or 0) > 0:
        entry.update(
            {
                "indexed": True,
                "media_duration_sec": r.get("duration_sec") or 0,
                "media_has_audio": bool(r.get("has_audio")),
                "media_has_video": bool(r.get("has_video")),
                "media_width": r.get("width") or 0,
                "media_height": r.get("height") or 0,
                "media_audio_codec": r.get("audio_codec") or "",
                "media_video_codec": r.get("video_codec") or "",
                "media_bitrate": r.get("bitrate") or 0,
            }
        )
    else:
        entry["indexed"] = False

    # Transcript metadata (same field names as _row_to_scan_meta)
    if has_tx:
        entry.update(
            {
                "processing_time": r.get("tx_processing_time") or "",
                "audio_duration": r.get("tx_total_duration") or "",
                "audio_duration_sec": r.get("tx_duration_sec") or 0,
                "processing_secs": r.get("tx_processing_secs") or 0,
                "tx_model": r.get("tx_model") or "",
                "tx_lang": r.get("tx_lang") or "",
                "tx_vocal_iso": bool(r.get("tx_vocal_iso")),
                "tx_vocal_model": r.get("tx_vocal_model") or "",
                "tx_speakers": _json_or_default(r.get("tx_speakers_json"), _json),
                "tx_seg_count": r.get("tx_seg_count") or 0,
                "tx_chunks": r.get("tx_chunks") or 0,
                "tx_date": r.get("tx_date") or 0,
                "tx_word_count": r.get("tx_word_count") or 0,
                "tx_char_count": r.get("tx_char_count") or 0,
                "tx_speaker_count": r.get("tx_speaker_count") or 0,
                "tx_warning_count": r.get("tx_warning_count") or 0,
                "tx_app_version": r.get("tx_app_version") or "",
            }
        )

    return entry


def _json_or_default(raw, _json, default=None):
    """Defensive JSON parse for speaker lists from the DB."""
    if not raw:
        return default if default is not None else []
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return default if default is not None else []


def _newest_json_mtime(transcript_dir: Path) -> float:
    """Newest mtime of all (non-hidden) *.json files under a directory.

    0.0 if no JSONs are present/readable. Used by auto-sync as the mtime
    skip (#115): if nothing changed, the directory doesn't need to be
    reimported. The selection matches that of sync_from_json_dir (walk
    with pruning of hidden folders except .transcripts, *.json without
    names starting with "."), so the skip never misses a real import.
    """
    newest = 0.0
    try:
        # Like sync_from_json_dir: prune hidden folders (.zfs snapshots
        # etc.) before descending, only .transcripts remains walkable.
        for dirpath, dirnames, filenames in os.walk(transcript_dir, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") or d == ".transcripts"]
            for fname in filenames:
                if not fname.endswith(".json") or fname.startswith("."):
                    continue
                try:
                    m = os.stat(os.path.join(dirpath, fname)).st_mtime
                except OSError:
                    continue
                if m > newest:
                    newest = m
    except OSError:
        pass
    return newest


def _load_json_meta(rel_name: str, abs_path: Path) -> tuple[dict | None, dict | None]:
    """Read scan metadata from a JSON transcript file (fallback path).

    Returns:
        (meta_dict, full_tdata) — both None if nothing found.
    """
    import json as _json

    from whisper_tm.core.helpers import transcript_path_for

    # Exactly one location, determined by transcript_mode. The actual
    # location of the file goes in, not rel_name plus a guessed root.
    for tp in (transcript_path_for(rel_name, abs_path=abs_path),):
        try:
            if not tp.exists():
                continue
            with open(tp, encoding="utf-8") as f:
                tdata = _json.load(f)
            if not tdata.get("complete"):
                continue
            segs = tdata.get("segments", []) or []
            all_text = " ".join(s.get("text", "") for s in segs)
            meta = {
                "processing_time": tdata.get("processing_time", ""),
                "audio_duration": tdata.get("total_duration", ""),
                "audio_duration_sec": tdata.get("total_duration_sec", 0) or 0,
                "processing_secs": tdata.get("processing_secs", 0) or 0,
                "tx_model": tdata.get("model", ""),
                "tx_lang": tdata.get("language", ""),
                "tx_vocal_iso": bool(tdata.get("vocal_isolation", False)),
                "tx_vocal_model": tdata.get("vocal_model", "") or "",
                "tx_speakers": tdata.get("speakers", []) or [],
                "tx_seg_count": len(segs),
                "tx_chunks": tdata.get("total_chunks", 0) or 0,
                "tx_date": tp.stat().st_mtime,
                "tx_word_count": len(all_text.split()),
                "tx_char_count": len(all_text),
                "tx_speaker_count": len(
                    set(s.get("speaker", "") for s in segs if s.get("speaker"))
                ),
                "tx_warning_count": tdata.get("warning_count", 0) or 0,
                "tx_app_version": tdata.get("app_version", "") or "",
            }
            return meta, tdata
        except Exception as e:
            logger.debug("JSON meta read %s failed: %s", tp, e)
            continue
    return None, None

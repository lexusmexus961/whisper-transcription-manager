"""
Filesystem scanner: directory iteration, ignore patterns, media discovery.

Extracted from api/main.py scan_directory() for better testability
and reuse (e.g. filesystem watcher, CLI scan).
"""

from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.core.helpers import lift_skip, skip_lapsed

logger = logging.getLogger(__name__)

#: Folders whose unreadable names were already named in a warning (per run:
#: a continuous scan would repeat it every few minutes)
_bad_names_reported: set[str] = set()


def _printable(name: str) -> str:
    """A file name for a message, its bytes that are not UTF-8 written out."""
    return os.fsencode(name).decode("utf-8", "backslashreplace")


def storable_entries(entries: list, directory) -> list:
    """The entries whose names are valid UTF-8; the others are skipped and
    named in a warning.

    The system hands a name with other bytes (Latin-1 from an old archive)
    over with stand-in characters that no database, JSON answer or URL
    takes: one such name made the scan's single write fail, and every new
    file of that scan stayed out of the library without a word.
    """
    good, bad = [], []
    for entry in entries:
        try:
            entry.name.encode("utf-8")
        except UnicodeEncodeError:
            bad.append(entry.name)
        else:
            good.append(entry)
    if bad:
        where = _printable(str(directory))
        if where not in _bad_names_reported:
            if len(_bad_names_reported) < 10000:
                _bad_names_reported.add(where)
            logger.warning(
                "Skipped %d name(s) in %s that are not valid UTF-8: %s%s. Rename them "
                "(for example with convmv) so they can be listed and transcribed.",
                len(bad),
                where,
                ", ".join(_printable(n) for n in bad[:5]),
                " …" if len(bad) > 5 else "",
            )
    return good


def iter_media_files(
    root: Path,
    *,
    recursive: bool = False,
    ignore_patterns: list[str] | None = None,
    media_extensions: set[str] | None = None,
    on_transcript_dir: callable | None = None,
) -> Iterator[Path]:
    """Iterate media files in directory.

    Without file/depth limits: previous max_files/max_depth ceilings are
    removed — a scan ALWAYS sees the whole tree.
    Protection against infinite loops is symlink cycle detection alone.

    Args:
        root: base directory
        recursive: include subdirectories
        ignore_patterns: glob patterns for names to ignore
        media_extensions: allowed file extensions (lowercase with period)
        on_transcript_dir: callback when .transcripts/ found

    Yields:
        Path to each found media file
    """
    if ignore_patterns is None:
        ignore_patterns = config.get("scanner", "ignore_patterns", default=[".*"])
    if media_extensions is None:
        media_extensions = config.media_extensions

    count = 0  # progress log only (every 5000 files)
    # Global-visited set of canonical directory paths: each real directory
    # is entered EXACTLY ONCE. Catches ALL symlink cycles — even reciprocal
    # links A→B→A between non-ancestors (pure ancestor check doesn't).
    # For normal directories the canonical path is known by construction
    # (parent path + name) → no realpath/stat syscall; only real
    # symlinks are resolved via realpath.
    try:
        root_real = os.path.realpath(root)
    except OSError:
        root_real = str(root)
    visited: set[str] = {root_real}

    def _walk(directory: Path, depth: int, dir_real: str) -> Iterator[Path]:
        # `dir_real` = canonical (symlink-resolved) path of `directory`,
        # passed per level so symlink cycles are detected WITHOUT expensive
        # stat/realpath per directory.
        nonlocal count
        try:
            # os.scandir instead of Path.iterdir(): is_dir()/is_symlink()/is_file()
            # of DirEntry objects are served from ONE directory read
            # (no extra stat syscall per entry except resolving symlink target).
            # On NAS/SMB this saves thousands of network roundtrips vs
            # repeated Path.is_dir()/.stat().
            with os.scandir(directory) as it:
                entries = sorted(storable_entries(list(it), directory), key=lambda e: e.name)
        except OSError as e:
            # OSError covers both PermissionError AND transient I/O errors
            # (NAS mount briefly gone) — an error in a subdirectory must not
            # abort the complete scan.
            logger.debug("Directory not readable: %s (%s)", directory, e)
            return

        for entry in entries:
            name = entry.name
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue  # e.g. broken symlink

            # .transcripts must be checked BEFORE the ignore check,
            # because the ".*" pattern would otherwise swallow ".transcripts"
            if is_dir and name == ".transcripts":
                if on_transcript_dir:
                    on_transcript_dir(Path(entry.path))
                continue

            if _should_ignore(name, ignore_patterns):
                continue

            if is_dir:
                if recursive:
                    child = entry.path
                    # Determine the canonical path: real symlinks via realpath
                    # (syscall, rare), normal directories by construction
                    # (parent path + name → NO syscall). inode-free, so it
                    # stays correct on Windows/NAS (st_ino often 0/ambiguous).
                    try:
                        is_link = entry.is_symlink()
                    except OSError:
                        is_link = False
                    if is_link:
                        try:
                            child_real = os.path.realpath(child)
                        except OSError:
                            continue
                    else:
                        child_real = dir_real + os.sep + name
                    # Enter each real directory EXACTLY ONCE — catches all
                    # cycles (including mutual/sibling symlinks), prevents the
                    # exponential re-scanning that made the scan "hang".
                    if child_real in visited:
                        logger.debug("Cycle/duplicate link skipped: %s", child)
                        continue
                    visited.add(child_real)
                    yield from _walk(Path(child), depth + 1, child_real)
            else:
                try:
                    is_file = entry.is_file()
                except OSError:
                    continue
                if is_file and os.path.splitext(name)[1].lower() in media_extensions:
                    count += 1
                    # Live progress on large trees: otherwise a long NAS walk
                    # looks like a hang. One INFO line every 5000 files.
                    if count % 5000 == 0:
                        logger.info("Scan walk running… %d media files so far", count)
                    yield Path(entry.path)

    yield from _walk(root, 0, root_real)


def _should_ignore(name: str, patterns: list[str]) -> bool:
    """Check whether a name matches against the ignore patterns."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def iter_files_named(
    root: Path,
    basename: str,
    *,
    ignore_patterns: list[str] | None = None,
    max_dirs: int = 50000,
) -> Iterator[Path]:
    """Recursively search for files with exactly this name — WITH ignore patterns.

    Replacement for Path.rglob(basename): rglob descends into ALL
    directories, including hidden ones — on ZFS shares that means
    .zfs/snapshot/, where the entire tree sits again per snapshot. That
    multiplies the search and runs right into the special-file/SMB traps
    that the scanner already avoids via ignore_patterns. Here, ignored
    directories are pruned BEFORE descending, symlinks are not followed.

    Args:
        root: base directory
        basename: exact file name (no glob)
        ignore_patterns: same as the scanner (default from config)
        max_dirs: hard cap on visited directories

    Yields:
        Path to each found file with this name
    """
    if ignore_patterns is None:
        ignore_patterns = config.get("scanner", "ignore_patterns", default=[".*"])

    visited_dirs = 0
    # os.walk with topdown=True: shortening dirnames in-place → pruned
    # subtrees are never entered at all (not a single scandir roundtrip there).
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        visited_dirs += 1
        if visited_dirs > max_dirs:
            logger.debug("iter_files_named: max_dirs (%d) reached in %s", max_dirs, root)
            return
        dirnames[:] = [d for d in dirnames if not _should_ignore(d, ignore_patterns)]
        if basename in filenames:
            yield Path(dirpath) / basename


class WalkAborted(RuntimeError):
    """The walk was aborted because the filesystem appears to be dead
    (a series of transport errors, e.g. NAS 'Host is down') or has not
    responded for minutes. A clear error is better than hours of silent
    grinding on a dead mount."""


# OSError errnos that indicate a dead network drive (rather than a single
# broken directory): EIO, EAGAIN (cifs when the server is dead), ENOTCONN,
# EHOSTDOWN, EHOSTUNREACH, ESTALE, ETIMEDOUT.
_TRANSPORT_ERRNOS = {5, 11, 107, 110, 112, 113, 116}

# How often a transport-failed directory is retried after mount recovery
# before it finally counts as an error (prevents endless retries, e.g. on a
# permanently broken handle/ESTALE despite a live mount).
_MAX_DIR_RETRIES = 5


def _probe_dir_alive(path: str, timeout: float) -> bool:
    """Check whether a directory is (again) readable — WITH a time limit.

    The read attempt runs in a daemon thread: on a dead CIFS, scandir can
    block for minutes — the sample caller must not hang on that. A stuck
    sample thread simply runs out once the kernel ends the call."""
    import threading

    ok = {"v": False}

    def _p():
        try:
            with os.scandir(path) as it:
                next(iter(it), None)
            ok["v"] = True
        except OSError:
            pass

    t = threading.Thread(target=_p, daemon=True, name="walk-mount-sample")
    t.start()
    t.join(timeout)
    return ok["v"]


def _scan_one_dir(
    directory: str,
    ignore_patterns: list[str],
    media_extensions: set[str],
) -> tuple[list[str], list[tuple[str, str, bool]], list[str], int | None]:
    """Read ONE directory (worker function for the parallel walk).

    Returns:
        (media_paths, subdirs, transcript_dirs, errno)
        subdirs: list of (path, name, is_symlink) — the cycle/visited logic
        is handled by the coordinator (main thread), so no lock is needed.
        errno: None on success, otherwise the OSError errno (dead-mount detection).
    """
    media: list[str] = []
    subdirs: list[tuple[str, str, bool]] = []
    tdirs: list[str] = []
    try:
        with os.scandir(directory) as it:
            entries = storable_entries(list(it), directory)
    except OSError as e:
        # PermissionError/transient NAS errors: skip subdirectory, don't
        # abort the whole walk. The errno goes to the coordinator —
        # a SERIES of transport errors means: the mount is dead.
        logger.debug("Directory not readable: %s (%s)", directory, e)
        return media, subdirs, tdirs, (e.errno if e.errno is not None else -1)

    for entry in entries:
        name = entry.name
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue  # e.g. broken symlink

        # .transcripts BEFORE the ignore check (".*" would swallow it)
        if is_dir and name == ".transcripts":
            tdirs.append(entry.path)
            continue
        if _should_ignore(name, ignore_patterns):
            continue

        if is_dir:
            try:
                is_link = entry.is_symlink()
            except OSError:
                is_link = False
            subdirs.append((entry.path, name, is_link))
        else:
            try:
                is_file = entry.is_file()
            except OSError:
                continue
            if is_file and os.path.splitext(name)[1].lower() in media_extensions:
                media.append(entry.path)
    return media, subdirs, tdirs, None


def collect_media_files(
    root: Path,
    *,
    recursive: bool = True,
    ignore_patterns: list[str] | None = None,
    media_extensions: set[str] | None = None,
    on_transcript_dir: callable | None = None,
    on_progress: callable | None = None,
    workers: int | None = None,
    should_stop: callable | None = None,
    wait_for_mount: bool | None = None,
    max_error_streak: int | None = None,
    stall_timeout: float | None = None,
    probe_sec: float | None = None,
    wait_max_min: float | None = None,
) -> list[Path]:
    """Collect media files PARALLEL (recursive walk over a thread pool).

    Why parallel? The serial walk (iter_media_files) makes one os.scandir
    roundtrip per directory level — on NAS/SMB every one of those is
    network latency. With thousands of folders, the latency completely
    dominates the scan time. A pool that reads many directories at once
    hides this latency (the NAS serves dozens of reads in parallel) → the
    root scan of a large library drops from minutes to seconds.

    Semantics like iter_media_files (ignore patterns, .transcripts callback,
    symlink cycle protection, NO file/depth limit), but as a materialized,
    sorted list — all callers (scan, indexer, long-running scan) already
    collect the generator output into a list anyway.

    Coordination via a completion queue with a bounded frontier (instead of
    concurrent.futures.wait over ALL open futures): wait() is O(open) per
    call — with 100k+ directories that becomes quadratic and the walk
    apparently gets slower and slower. The queue is O(1) per directory, and
    the frontier limit keeps the number of live futures small (memory).
    The visited/cycle logic runs exclusively in the coordinator thread
    (no lock, deterministic); workers only do the plain scandir.

    on_progress: optional callback, receives a dict
    ``{"files", "dirs_done", "dirs_known", "errors"}`` (throttled, from the
    coordinator thread); in wait mode additionally ``waiting=True`` with
    ``wait_attempt``/``wait_secs``/``wait_reason``. An exact percentage is
    inherently impossible during the walk (the total is only known at the
    end) — but ``dirs_done/dirs_known`` converges towards 100% and serves
    as an estimate; it gets more accurate when the caller knows the
    expected file count (e.g. DB count from the last walk).

    should_stop: optional callable → True cleanly aborts the walk
    (e.g. client closed the SSE connection).

    wait_for_mount (default: config scanner.walk_wait_for_mount, True):
    if the walk detects a dead mount (transport error series or stall), it
    waits for its return (periodic root sample) and catches up the failed
    directories afterwards, instead of aborting — SMB services (e.g. on a NAS)
    like to temporarily lock up under load. The remaining thresholds
    (max_error_streak, stall_timeout, probe_sec, wait_max_min) also come
    from the config when None.

    Raises:
        WalkAborted: dead mount with wait mode disabled, or the network
        drive did not return within walk_wait_max_min.
    """
    if ignore_patterns is None:
        ignore_patterns = config.get("scanner", "ignore_patterns", default=[".*"])
    if media_extensions is None:
        media_extensions = config.media_extensions
    if workers is None:
        workers = int(config.get("scanner", "walk_workers", default=16) or 1)

    # Non-recursive (one level) or explicitly serial → existing iterator
    if not recursive or workers <= 1:
        return list(
            iter_media_files(
                root,
                recursive=recursive,
                ignore_patterns=ignore_patterns,
                media_extensions=media_extensions,
                on_transcript_dir=on_transcript_dir,
            )
        )

    import queue as _queue_mod
    import time as _time
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    try:
        root_real = os.path.realpath(root)
    except OSError:
        root_real = str(root)
    visited: set[str] = {root_real}

    results: list[str] = []
    stop = False  # should_stop → no NEW directories
    logged = 0  # progress log threshold (every 5000 files)
    dirs_done = 0  # directories finished reading
    dirs_known = 1  # directories discovered so far (incl. root)
    dir_errors = 0  # directories with a final read error
    err_streak = 0  # transport errors IN A ROW (dead-mount detection)
    last_prog = 0.0  # time throttle for on_progress
    abort_msg: str | None = None
    # Transport-failed directories: in wait mode they are retried AGAIN
    # after mount recovery (instead of failing silently).
    failed_transport: list[tuple[str, int, str]] = []
    retry_counts: dict[str, int] = {}  # path → retry attempts so far

    # Dead mount: N transport errors in a row without a single success,
    # or stall_timeout seconds without a single directory response
    # (all workers hanging in blocking scandir on a dead SMB).
    if max_error_streak is None:
        max_error_streak = int(config.get("scanner", "walk_max_error_streak", default=150) or 0)
    if stall_timeout is None:
        stall_timeout = float(config.get("scanner", "walk_stall_timeout_sec", default=180) or 0)
    # Wait mode: on a detected mount failure, do NOT abort but periodically
    # sample the root directory and continue after its return (SMB services
    # on a NAS like to temporarily lock up under load).
    if wait_for_mount is None:
        wait_for_mount = bool(config.get("scanner", "walk_wait_for_mount", default=True))
    if probe_sec is None:
        probe_sec = float(config.get("scanner", "walk_wait_probe_sec", default=15) or 15)
    if wait_max_min is None:
        wait_max_min = float(config.get("scanner", "walk_wait_max_min", default=0) or 0)
    wait_max_sec = wait_max_min * 60.0

    # Completion queue instead of wait(pending): O(1) per directory.
    done_q: _queue_mod.Queue = _queue_mod.Queue()
    to_visit: deque = deque()  # discovered, not-yet-submitted directories
    outstanding = 0  # submitted, not-yet-processed futures
    # Frontier limit: enough to saturate the pool, small enough not to hold
    # hundreds of thousands of future objects at 100k+ folders.
    max_outstanding = max(workers * 4, 8)

    pool = ThreadPoolExecutor(max_workers=workers)

    def _submit(path: str, depth: int, dir_real: str) -> None:
        nonlocal outstanding
        outstanding += 1
        fut = pool.submit(_scan_one_dir, path, ignore_patterns, media_extensions)
        # done callback runs in the worker thread; Queue.put is thread-safe.
        fut.add_done_callback(lambda f, p=path, d=depth, r=dir_real: done_q.put((f, p, d, r)))

    def _emit_progress(force: bool = False, **extra) -> None:
        nonlocal last_prog
        if not on_progress:
            return
        now = _time.monotonic()
        if not force and now - last_prog < 0.25:
            return
        last_prog = now
        on_progress(
            {
                "files": len(results),
                "dirs_done": dirs_done,
                "dirs_known": dirs_known,
                "errors": dir_errors,
                **extra,
            }
        )

    def _wait_for_recovery(reason: str) -> str:
        """Wait for the mount to return. → 'recovered'/'stopped'/'timeout'."""
        logger.warning(
            "Scan walk paused: %s — waiting for network drive (sample every %.0fs%s)…",
            reason,
            probe_sec,
            f", max. {wait_max_min:.0f}min" if wait_max_sec else "",
        )
        t0 = _time.monotonic()
        attempt = 0
        while True:
            if should_stop and should_stop():
                return "stopped"
            if wait_max_sec and (_time.monotonic() - t0) > wait_max_sec:
                return "timeout"
            attempt += 1
            _emit_progress(
                force=True,
                waiting=True,
                wait_attempt=attempt,
                wait_secs=round(_time.monotonic() - t0),
                wait_reason=reason,
            )
            if _probe_dir_alive(str(root), timeout=probe_sec):
                logger.info(
                    "Network drive reachable again — scan walk "
                    "continuing (%d folders being caught up)",
                    len(failed_transport),
                )
                return "recovered"
            # A sample that returns an error immediately (instead of
            # hanging) should not be hot-polled.
            _time.sleep(min(probe_sec, 5.0))

    def _recover_or_abort(reason: str) -> bool:
        """Run wait mode (if active). True = continue."""
        nonlocal abort_msg, err_streak, stop
        if not wait_for_mount:
            abort_msg = f"{reason} — scan aborted at {len(results)} files / {dirs_done} folders"
            return False
        outcome = _wait_for_recovery(reason)
        if outcome == "stopped":
            stop = True  # clean abort (client gone) — not an error
            return True
        if outcome == "timeout":
            abort_msg = (
                f"{reason} — network drive did not return within "
                f"{wait_max_min:.0f}min, scan aborted "
                f"at {len(results)} files / {dirs_done} folders"
            )
            return False
        # recovered: re-queue failed directories
        err_streak = 0
        to_visit.extend(failed_transport)
        failed_transport.clear()
        return True

    try:
        _submit(str(root), 0, root_real)
        while outstanding:
            try:
                fut, path, depth, dir_real = done_q.get(
                    timeout=stall_timeout if stall_timeout > 0 else None
                )
            except _queue_mod.Empty:
                # Not a single worker responding anymore → mount is completely stuck.
                if not _recover_or_abort(
                    f"Filesystem has not responded for {stall_timeout:.0f}s "
                    f"(network drive hanging?)"
                ):
                    break
                if stop:
                    break  # should_stop during the wait: do NOT keep waiting
                    # for the blocked workers (hanging on dead SMB)
                # After mount recovery: submit the caught-up directories,
                # in case the frontier was empty (all workers were hanging).
                while to_visit and outstanding < max_outstanding:
                    _submit(*to_visit.popleft())
                continue
            outstanding -= 1
            media, subdirs, tdirs, err = fut.result()

            # Dead-mount detection: SERIES of transport errors without a success
            if err is None:
                err_streak = 0
                dirs_done += 1
                if failed_transport:
                    # The mount is evidently alive (again) — catch up
                    # pending failures right away, even WITHOUT a wait
                    # cycle (a brief blip that resolved itself).
                    to_visit.extend(failed_transport)
                    failed_transport.clear()
            elif err in _TRANSPORT_ERRNOS:
                err_streak += 1
                tries = retry_counts.get(path, 0)
                if wait_for_mount and tries < _MAX_DIR_RETRIES:
                    # Note for retry after mount recovery — counts neither
                    # as done nor as a final error.
                    retry_counts[path] = tries + 1
                    failed_transport.append((path, depth, dir_real))
                else:
                    # Retries exhausted (broken handle despite a live
                    # mount) or wait mode off: final.
                    dirs_done += 1
                    dir_errors += 1
                if max_error_streak and err_streak >= max_error_streak:
                    if not _recover_or_abort(
                        f"{err_streak} directory transport errors in a row "
                        f"(network drive unreachable?)"
                    ):
                        break
                    if stop:
                        break  # should_stop during the wait → leave
            else:
                # Ordinary error (e.g. no permissions): final.
                dirs_done += 1
                dir_errors += 1

            if on_transcript_dir:
                for td in tdirs:
                    on_transcript_dir(Path(td))
            results.extend(media)
            if len(results) - logged >= 5000:
                logged = len(results)
                logger.info("Scan walk running… %d media files so far", logged)
            _emit_progress()

            if should_stop and not stop and should_stop():
                logger.info(
                    "Scan walk aborted (should_stop) — %d files / %d folders",
                    len(results),
                    dirs_done,
                )
                stop = True

            if not stop:
                for child, name, is_link in subdirs:
                    # Determine the canonical path: real symlinks via
                    # realpath (syscall, rare), normal directories by
                    # construction (parent path + name → NO syscall). Each
                    # real directory is entered EXACTLY ONCE → catches all
                    # symlink cycles.
                    if is_link:
                        try:
                            child_real = os.path.realpath(child)
                        except OSError:
                            continue
                    else:
                        child_real = dir_real + os.sep + name
                    if child_real in visited:
                        logger.debug("Cycle/duplicate link skipped: %s", child)
                        continue
                    visited.add(child_real)
                    dirs_known += 1
                    to_visit.append((child, depth + 1, child_real))

            # Refill the frontier — nothing new while stopping, the
            # already-running workers run out (while outstanding).
            while to_visit and outstanding < max_outstanding and not stop:
                _submit(*to_visit.popleft())

            # Everything else processed, but transport-failed folders still
            # open (streak threshold never reached, no further success
            # after that)? → treat like a mount failure: wait & catch up
            # (or abort), instead of silently losing the folders.
            if outstanding == 0 and failed_transport and not stop:
                if not _recover_or_abort(
                    f"{len(failed_transport)} directories open due to "
                    f"transport errors (network drive gone?)"
                ):
                    break
                if stop:
                    break
                while to_visit and outstanding < max_outstanding:
                    _submit(*to_visit.popleft())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if abort_msg:
        logger.warning("Scan walk: %s", abort_msg)
        raise WalkAborted(abort_msg)

    # Deterministic order (the parallel walk delivers unsorted)
    results.sort()
    final = [Path(p) for p in results]
    if on_progress:
        # ALWAYS send the completion event (unthrottled): small trees would
        # otherwise get no progress at all, and this way the caller sees
        # the final state.
        on_progress(
            {
                "files": len(final),
                "dirs_done": dirs_done,
                "dirs_known": dirs_done,
                "errors": dir_errors,
            }
        )
    return final


def build_file_entry(
    fp: Path,
    vdir: Path,
    *,
    skiplist: dict | None = None,
) -> dict | None:
    """Convert a single media file into a scan entry dict.

    Returns:
        Dict with file info, or None on error.
    """
    # rel_from returns None if the file is not below the root
    # (symlink pointing outside). Such entries have no valid rel_path.
    rel_str = paths.rel_from(fp, vdir)
    if rel_str is None:
        return None

    try:
        stat = fp.stat()
    except OSError:
        return None

    # The metadata title is read by the FFprobe INDEXER (📇) into the DB —
    # not the walk. The former scanner.scan_metadata_titles switch used to
    # start one ffprobe PER FILE here, whose result never made it into the
    # DB and, since the paginated list, nobody displayed anymore either.
    metadata_title = ""

    # subfolder from the canonical rel_str instead of via a Path object: the
    # value is already /-separated, a detour via pathlib would turn it back
    # into backslashes on Windows.
    parts = rel_str.split("/")
    subfolder = "/".join(parts[:-1]) if len(parts) > 1 else ""
    # Lookup via the IDENTITY: load_skiplist()/db.get_skiplist() are keyed
    # on abs_path (docs/PATHS.md). Looking up via rel_str never hit — skipped
    # files appeared as normal in the list, while the worker kept skipping
    # them. Old entries without an identity are still under the display
    # value, hence the second attempt.
    #
    # The skiplist value can be a dict ({"reason": ...}) or some other
    # truthy value — only read the reason for a dict, otherwise .get()
    # on e.g. a string would throw an AttributeError.
    _sl = skiplist or {}
    abs_str = paths.canon(fp)
    skip_entry = _sl.get(abs_str) or _sl.get(rel_str)
    # Found corrupt, changed since: most likely it was still being copied
    if skip_entry is not None and skip_lapsed(abs_str, skip_entry, stat.st_mtime):
        lift_skip(abs_str)
        skip_entry = None
    skipped = skip_entry is not None
    skip_reason = skip_entry.get("reason", "") if isinstance(skip_entry, dict) else ""

    return {
        "name": rel_str,
        "abs_path": abs_str,
        "display_name": fp.name,
        "metadata_title": metadata_title,
        "subfolder": subfolder,
        "size": stat.st_size,
        "ext": fp.suffix.lower(),
        "is_video": fp.suffix.lower() in config.video_extensions,
        "modified": stat.st_mtime,
        "skipped": skipped,
        "skip_reason": skip_reason,
        # abs_path for status resolution
        "_abs_path": fp,
    }

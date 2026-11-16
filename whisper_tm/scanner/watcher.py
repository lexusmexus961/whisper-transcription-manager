"""
Filesystem Watcher: Detects new/changed/deleted media files.

Polling-based (no watchdog needed) — cross-platform compatible.
Uses mtime + size fingerprints for change detection.

Usage:
    watcher = FileWatcher(media_dir, on_change=callback)
    watcher.start()   # Starts background thread
    watcher.stop()    # Stops cleanly

Events:
    on_change(event_type, rel_path, abs_path)
    event_type: "added", "modified", "removed"
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.scanner.filesystem import iter_media_files

logger = logging.getLogger(__name__)

# Type: (event_type, rel_path, abs_path)
ChangeCallback = Callable[[str, str, Path | None], None]


@dataclass
class FileFingerprint:
    """File fingerprint for change detection."""

    size: int
    mtime: float


class FileWatcher:
    """Polling-based filesystem watcher for media directories.

    Detects:
    - New files (added)
    - Changed files (modified) — size or mtime changed
    - Deleted files (removed)

    Args:
        media_dir: Directory to monitor
        interval: Polling interval in seconds
        recursive: Include subdirectories
        on_change: Callback for events
        debounce_sec: Wait time after a change before reporting it
    """

    def __init__(
        self,
        media_dir: Path | None = None,
        interval: float = 30.0,
        recursive: bool = True,
        on_change: ChangeCallback | None = None,
        debounce_sec: float = 5.0,
    ):
        self._media_dir = media_dir or config.media_dir
        self._interval = interval
        self._recursive = recursive
        self._on_change = on_change
        self._debounce = debounce_sec

        self._fingerprints: dict[str, FileFingerprint] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        # Protects _fingerprints + _media_dir against races between the
        # poll thread and request threads (start/stop/set_media_dir/status).
        self._lock = threading.Lock()

        # Debounce: collect events and fire after debounce_sec
        self._pending_events: list[tuple[str, str, Path | None]] = []
        self._last_event_time: float = 0

    @property
    def is_running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start watcher in background thread."""
        if self.is_running:
            logger.warning("FileWatcher already running")
            return

        self._stop_event.clear()
        self._started = True

        # Initial scan: capture current state as baseline (scan without lock)
        initial = self._scan_fingerprints() or {}
        with self._lock:
            self._fingerprints = initial
        logger.info(
            "FileWatcher started: %s (%d files, %ds interval)",
            self._media_dir,
            len(initial),
            self._interval,
        )

        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="file-watcher")
        self._thread.start()

    def stop(self) -> None:
        """Stop watcher cleanly."""
        if not self._started:
            return
        self._stop_event.set()
        # Without timeout join: The poll thread reacts immediately to the
        # stop event (the wait() in the loop returns). With timeout, a
        # still-running scan could be orphaned — a second start() would then
        # create a second poll thread. Don't join yourself (if stop() is called
        # from a callback within the poll thread).
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self._started = False
        self._thread = None
        logger.info("FileWatcher stopped")

    def force_scan(self) -> list[tuple[str, str, Path | None]]:
        """Trigger manual scan and return events."""
        return self._detect_changes()

    def set_media_dir(self, media_dir: Path) -> None:
        """Reorient watcher to a new directory (runtime switch).

        Resets fingerprints (new baseline, no "removed" flood
        for files in the old directory).
        """
        new_dir = Path(media_dir)
        # Scan without lock (long-running), then switch media_dir + fingerprints
        # atomically under lock. This ensures the poll thread never sees a mix of
        # new media_dir and old fingerprints (otherwise: event flood).
        fingerprints = self._scan_fingerprints(new_dir) or {}
        with self._lock:
            self._media_dir = new_dir
            self._fingerprints = fingerprints
        logger.info("FileWatcher reoriented: %s (%d files)", new_dir, len(fingerprints))

    # ── Internal ──────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Main loop: scan periodically.

        Adaptive timing: Since the removal of the max_files cap, each
        poll runs over the COMPLETE tree — on
        a large CIFS library that takes longer than the interval.
        To prevent this from becoming steady NAS load (walk ends, next starts
        immediately), the loop waits at least twice as long as the
        last scan took.
        """
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                events = self._detect_changes()
                if events:
                    self._handle_events(events)
            except Exception as e:
                logger.debug("FileWatcher poll error: %s", e)

            scan_duration = time.monotonic() - t0
            wait = max(self._interval, scan_duration * 2)
            if wait > self._interval:
                logger.info(
                    "FileWatcher: scan took %.0fs — next poll in %.0fs instead of %ds",
                    scan_duration,
                    wait,
                    self._interval,
                )
            # Wait, but react immediately to stop
            self._stop_event.wait(wait)

    def _scan_fingerprints(
        self,
        media_dir: Path | None = None,
        prev: dict[str, FileFingerprint] | None = None,
    ) -> dict[str, FileFingerprint] | None:
        """Capture current directory state as fingerprints.

        Args:
            media_dir: Directory to scan (default: current). Passed explicitly
                so a scan runs consistently against ONE directory, even if
                set_media_dir changes in parallel.
            prev: Previous fingerprints as fallback. If a single stat() call
                fails transiently (e.g., file briefly locked), the old
                fingerprint is kept instead of incorrectly marking the file
                as "removed" (prevents removed/added flapping).

        Returns:
            Fingerprint dict, or None on scan error (e.g., mount gone).
            None signals the caller to keep old fingerprints instead of
            reporting everything as "removed".
        """
        if media_dir is None:
            media_dir = self._media_dir
        prev = prev or {}
        result = {}
        try:
            if not media_dir.exists():
                logger.warning("FileWatcher: media directory not reachable: %s", media_dir)
                return None
            for fp in iter_media_files(
                media_dir,
                recursive=self._recursive,
            ):
                rel = paths.rel_from(fp, media_dir)
                if rel is None:
                    continue
                try:
                    stat = fp.stat()
                    result[rel] = FileFingerprint(size=stat.st_size, mtime=stat.st_mtime)
                except OSError:
                    # Transient stat error: keep old fingerprint instead of
                    # letting the file flap as "removed"/"added".
                    old = prev.get(rel)
                    if old is not None:
                        result[rel] = old
        except Exception as e:
            logger.debug("FileWatcher scan error: %s", e)
            return None
        return result

    def _detect_changes(self) -> list[tuple[str, str, Path | None]]:
        """Compare current state with stored fingerprints.

        Returns:
            List of (event_type, rel_path, abs_path) tuples
        """
        # Take a consistent snapshot of media_dir + baseline, then scan
        # WITHOUT the lock (long-running I/O).
        with self._lock:
            media_dir = self._media_dir
            baseline = self._fingerprints

        current = self._scan_fingerprints(media_dir, prev=baseline)
        if current is None:
            # Scan failed (mount gone?) → keep old fingerprints.
            # Otherwise a transient outage would report ALL files as "removed"
            # and then as "added" on the next poll (event storm).
            return []

        events: list[tuple[str, str, Path | None]] = []
        with self._lock:
            # If set_media_dir changed the directory during the scan, the
            # result belongs to the old directory → discard it. The new
            # baseline was already set; no event storm, no overwrite.
            if self._media_dir != media_dir:
                return []

            old = self._fingerprints
            old_keys = set(old.keys())
            new_keys = set(current.keys())

            # New files
            for rel in new_keys - old_keys:
                abs_path = media_dir / rel
                events.append(("added", rel, abs_path))

            # Deleted files
            for rel in old_keys - new_keys:
                events.append(("removed", rel, None))

            # Changed files
            for rel in old_keys & new_keys:
                old_fp = old[rel]
                new_fp = current[rel]
                if old_fp.size != new_fp.size or abs(old_fp.mtime - new_fp.mtime) > 0.5:
                    abs_path = media_dir / rel
                    events.append(("modified", rel, abs_path))

            # Update fingerprints
            self._fingerprints = current

        return events

    def _handle_events(self, events: list[tuple[str, str, Path | None]]) -> None:
        """Forward events to callback."""
        for event_type, rel_path, abs_path in events:
            logger.info("FileWatcher: %s → %s", event_type, rel_path)
            if self._on_change:
                try:
                    self._on_change(event_type, rel_path, abs_path)
                except Exception as e:
                    logger.debug("FileWatcher callback error: %s", e)

    # ── Status ────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Watcher status as a dict (for the API)."""
        with self._lock:
            media_dir = self._media_dir
            tracked_files = len(self._fingerprints)
        return {
            "running": self.is_running,
            "media_dir": str(media_dir),
            "interval_sec": self._interval,
            "recursive": self._recursive,
            "tracked_files": tracked_files,
        }

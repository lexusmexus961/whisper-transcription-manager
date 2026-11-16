"""
Transcript status resolution: check if a file was transcribed.

Prioritization (DB-first for scale):
  1. Database (fast, one query for all)
  2. JSON fallback (only if no DB or DB miss)

With 50k+ transcripts the DB is 1000x faster than JSON scan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.core.helpers import transcript_path_for

logger = logging.getLogger(__name__)


def resolve_transcript_status(
    rel_str: str,
    abs_path: Path,
    db=None,
    *,
    completion_set: set[str] | None = None,
    scan_root: str | Path | None = None,
) -> tuple[bool, bool]:
    """Determine transcript status for a media file.

    Args:
        rel_str: relative path from scan root (forward slashes)
        abs_path: absolute path to file — the identity (docs/PATHS.md)
        db: database instance (optional)
        completion_set: pre-loaded set of all completed **abs_paths** (from
                        db.get_completion_set()). If passed, no DB query
                        is made → bulk performance.
        scan_root: root that ``rel_str`` refers to. Needed because a scan
                   runs on the folder it was given, and ``config.media_dir``
                   can change while it runs — without it single lookup would
                   build identity against the wrong root and fall back to the
                   ambiguous rel_path branch.

    Returns:
        (complete, has_partial) — complete=True if fully transcribed
    """
    # ── 1. DB / Pre-loaded Set (fastest path) ─────────────────
    # Comparison via identity. Neither basename nor rel_path: a/intro.mp4
    # and b/intro.mp4 are two files, and both workarounds reported one
    # as complete because the other was (docs/PATHS.md).
    if completion_set is not None:
        if paths.canon(abs_path) in completion_set:
            return True, False
    elif db:
        try:
            # Pass actual location instead of rel_str + guessed
            # root — this ensures stage 1 of the resolver always hits.
            root = str(scan_root) if scan_root else str(config.media_dir)
            if db.is_complete(paths.canon(abs_path), media_dir=root):
                return True, False
        except Exception as e:
            logger.debug("DB status check failed: %s", e)

    # ── 2. JSON fallback (only if DB has no match) ─────────
    # At transcript_mode="database" we skip the JSON check
    if config.transcript_mode == "database":
        return False, False

    return _check_json_locations(rel_str, abs_path)


def _check_json_locations(rel_str: str, abs_path: Path) -> tuple[bool, bool]:
    """Check JSON transcript at all possible locations.

    Exactly ONE location, determined by transcript_mode (core.helpers). The
    actual location of the file is passed as abs_path — not rel_str plus
    guessed root (docs/PATHS.md).
    """
    has_partial = False

    try:
        tp = transcript_path_for(rel_str, abs_path=abs_path)
        if tp.exists():
            c, p = _check_json(tp)
            if c:
                return True, False
            has_partial = has_partial or p
    except OSError as e:
        logger.debug("JSON status check OSError for %s: %s", rel_str, e)

    return False, has_partial


def _check_json(path: Path) -> tuple[bool, bool]:
    """Read JSON transcript and check status.

    Returns:
        (complete, has_partial)
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        is_complete = data.get("complete", False) is True
        return is_complete, not is_complete
    except Exception as e:
        logger.debug("JSON parse failed %s: %s", path.name, e)
        return False, True


def determine_file_status(
    complete: bool,
    has_partial: bool,
    skipped: bool,
) -> str:
    """Derive file status from individual checks.

    Returns:
        "done", "skipped", "partial", or "pending"
    """
    if complete:
        return "done"
    elif skipped:
        return "skipped"
    elif has_partial:
        return "partial"
    return "pending"

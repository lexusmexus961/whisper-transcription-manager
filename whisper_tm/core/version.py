"""Version/status info for logging and GUI.

In case of error it must be clear WHICH code version was running. Version
number alone is not enough (it doesn't change between commits) —
therefore additionally the git commit including dirty flag, determined
once at startup and cached thereafter.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from whisper_tm import __version__

_cache: dict | None = None
_START = time.time()


def version_info() -> dict:
    """{"version", "commit", "dirty", "display", "start_time"} — cached."""
    global _cache
    if _cache is not None:
        return _cache
    commit, dirty = "", False
    try:
        root = Path(__file__).resolve().parents[2]
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            commit = r.stdout.strip()
            r2 = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            dirty = bool(r2.returncode == 0 and r2.stdout.strip())
    except Exception:
        pass  # no git (e.g. release archive)
    display = __version__ + (f" ({commit}{'+dirty' if dirty else ''})" if commit else "")
    _cache = {
        "version": __version__,
        "commit": commit,
        "dirty": dirty,
        "display": display,
        "start_time": _START,
    }
    return _cache

"""Signals to a child's whole process group — where there are any.

The model servers and yt-dlp start in a session of their own
(``start_new_session=True``), so a signal to the group reaches their
children too (llama-server's workers, yt-dlp's ffmpeg merge). Windows has
neither ``os.killpg`` nor ``os.getpgid`` nor ``SIGKILL``: there
:func:`signal_group` says False and the caller ends the process itself with
``terminate()``/``kill()``. The AttributeError that escaped before skipped
everything after a stop — the reranker's release of the GPU gate included.
"""

from __future__ import annotations

import os
import signal


def signal_group(pid: int, kill: bool = False) -> bool:
    """SIGTERM (``kill``: SIGKILL) to the process group of ``pid``.

    False when that isn't possible here — no process groups (Windows), or
    the process is gone or not ours: the caller then signals the process
    itself."""
    if not (hasattr(os, "killpg") and hasattr(os, "getpgid")):
        _signal_children(pid, kill)
        return False
    sig = getattr(signal, "SIGKILL", signal.SIGTERM) if kill else signal.SIGTERM
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _signal_children(pid: int, kill: bool) -> None:
    """Where there are no process groups: the children of ``pid`` — yt-dlp's
    ffmpeg merge outlived it and kept the partial file locked. The caller
    ends ``pid`` itself."""
    try:
        import psutil

        children = psutil.Process(pid).children(recursive=True)
    except Exception:  # noqa: BLE001 — gone, or not ours
        return
    for child in children:
        try:
            if kill:
                child.kill()
            else:
                child.terminate()
        except Exception:  # noqa: BLE001 — ended in between
            pass


def end_pid(pid: int) -> None:
    """Stop a process known only by its pid (a leftover from an earlier run):
    its group where there are groups, else the process."""
    if signal_group(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

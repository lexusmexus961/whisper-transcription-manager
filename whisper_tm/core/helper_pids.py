"""
The model servers this app started, written down — so that the next start
ends the ones a crash, a kill or a closed terminal left behind.

The embedding and rerank servers (llama-server) run in a session of their
own, so they outlive the app when it doesn't get to stop them. The next
start found the old one answering on its port and used it as "someone
else's server" — forever, as it never stops a server it didn't start: the
reranker kept its ~5 GB on the card. Now each start records the server's
pid next to the database (one server per database holds its lock), and the
next server start ends a recorded process that is still there — only when
it is that very process: same pid, same start time, a llama-server on the
recorded port.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

NAMES = ("embed-server", "rerank-server")


def _file(name: str) -> Path | None:
    """Next to the database — None without a loaded configuration (a tool
    that starts the embedding server on its own)."""
    from whisper_tm.config import config

    try:
        return Path(str(config.db_path) + f".{name}.pid")
    except Exception:  # noqa: BLE001 — never worth not starting a server
        return None


def record(name: str, pid: int, port: int) -> None:
    """``name`` was started as ``pid`` and listens on ``port``."""
    try:
        import psutil

        started = psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 — without it the leftover check refuses
        started = 0.0
    path = _file(name)
    if path is None:
        return
    try:
        path.write_text(
            json.dumps({"pid": pid, "port": int(port), "started": started}), encoding="utf-8"
        )
    except OSError as e:
        logger.debug("%s pid not recorded: %s", name, e)


def forget(name: str) -> None:
    """``name`` was stopped the regular way."""
    path = _file(name)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.debug("%s pid not removed: %s", name, e)


def _ours(entry: dict):
    """The recorded process, if it is still the one recorded — else None."""
    try:
        import psutil

        proc = psutil.Process(int(entry["pid"]))
        if abs(proc.create_time() - float(entry.get("started") or 0)) > 2.0:
            return None  # the pid went to another process
        cmd = proc.cmdline()
    except Exception:  # noqa: BLE001 — gone, not ours, or not readable
        return None
    exe = Path(cmd[0]).name if cmd else ""
    port = str(entry.get("port") or "")
    if "llama-server" not in exe or "--port" not in cmd:
        return None
    if cmd[cmd.index("--port") + 1 : cmd.index("--port") + 2] != [port]:
        return None
    return proc


def end_leftover(name: str) -> bool:
    """End the ``name`` server an earlier run of this app left behind."""
    path = _file(name)
    if path is None:
        return False
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    proc = _ours(entry)
    if proc is not None:
        from whisper_tm.core.procgroup import signal_group

        logger.info(
            "Ending the %s the last run left behind (pid %s, port %s)",
            name,
            entry["pid"],
            entry.get("port"),
        )
        if not signal_group(proc.pid):
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001 — psutil.TimeoutExpired
            if not signal_group(proc.pid, kill=True):
                proc.kill()
            time.sleep(1.0)
    forget(name)
    return proc is not None


def end_leftovers() -> int:
    """At the server's start: every helper server the last run left behind."""
    return sum(end_leftover(name) for name in NAMES)

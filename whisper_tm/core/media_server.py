"""The media service process: started on first use, stopped when idle.

The service (whisper_tm/media/service.py) runs EmbeddingGemma 2 in the
media environment (media/environment.py) — a Python of its own, because the
model needs newer libraries than the app's. This module starts it with that
Python, waits for ``/health``, talks to it, and stops it:

* after ``media_search.idle_stop_sec`` without a request and without an index
  run (the CPU copy of the model holds 1–3 GB of RAM);
* when the app shuts down (its own process group goes with it).

Queries keep the GPU out of it: the service answers them on the CPU. The GPU
is only used by an index run, which holds a stage of the GPU gate for it
(media/indexer.py) and tells the service to release the card whenever it
gives the gate back.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from whisper_tm.core import debug
from whisper_tm.core.procgroup import end_pid, signal_group

logger = logging.getLogger(__name__)

WATCH_TICK_SEC = 15.0


class MediaUnavailable(RuntimeError):
    """The service can't answer — not installed, switched off, failed to start."""


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("media_search", name, default=default)


def _num(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    from whisper_tm.core.helpers import parse_bool

    return parse_bool(_cfg("enabled", True))


def _host() -> str:
    h = str(_cfg("host", "127.0.0.1") or "127.0.0.1")
    return "127.0.0.1" if h in ("0.0.0.0", "::") else h


def _port() -> int:
    return int(_num("port", 8083))


def service_script() -> Path:
    return Path(__file__).resolve().parents[1] / "media" / "service.py"


def base_directory() -> Path:
    return Path(__file__).resolve().parents[2]


class MediaServer:
    def __init__(self):
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._address: tuple[str, int] | None = None
        #: /health answered since the start — set after it, cleared on stop
        self._ready = False
        self._log: Path | None = None
        self._last_used = 0.0
        self._started_at = 0.0
        self._busy = 0
        #: Set while an index run uses the service — no idle stop then
        self.indexing = False
        self._watchdog: threading.Thread | None = None

    # ── State ────────────────────────────────────────────────────────────

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def url(self) -> str:
        host, port = self._address or (_host(), _port())
        return f"http://{host}:{port}"

    def _responds(self, timeout: float = 2.0) -> dict | None:
        try:
            with urllib.request.urlopen(self.url() + "/health", timeout=timeout) as r:
                return json.load(r)
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def state(self) -> dict:
        with self._lock:
            alive = self._alive()
            pid = self._proc.pid if alive else None
            since = self._started_at if alive else None
            idle = time.monotonic() - self._last_used if alive else None
        health = self._responds(1.0) if alive else None
        return {
            "running": alive,
            "pid": pid,
            "started_at": since,
            "idle_sec": round(idle) if idle is not None else None,
            "loaded": (health or {}).get("loaded", []),
            "versions": (health or {}).get("versions", {}),
            "address": self.url(),
            "log": str(self._log) if self._log else None,
        }

    # ── Starting ─────────────────────────────────────────────────────────

    def ensure(self) -> None:
        """Running and answering — started now when it isn't."""
        if not enabled():
            raise MediaUnavailable("Images & sounds is switched off (media_search.enabled)")
        with self._lock:
            # Ready, not just alive: during a cold start the others wait on
            # the start lock instead of posting before /health answers
            if self._ready and self._alive() and self._address:
                return
        with self._start_lock:
            with self._lock:
                if self._alive() and self._address:
                    return
            self._start()

    def _start(self) -> None:
        from whisper_tm.media import environment

        py = environment.python_path()
        if not py.exists():
            raise MediaUnavailable(
                "The media environment is not installed — Settings → Images & sounds → Install"
            )
        script = service_script()
        argv = [
            str(py),
            str(script),
            "--host",
            _host(),
            "--port",
            str(_port()),
            "--model",
            environment.model_id(),
            "--threads",
            str(int(_num("cpu_threads", 8))),
        ]
        # A service left over from an earlier run of this app (a crash leaves
        # it running, its own process group): stopped, so the new one runs
        # with today's settings. Anything else on the port is not ours.
        ready = self._responds(1.0)
        if ready:
            pid = int(ready.get("pid") or 0)
            try:
                import psutil

                # psutil, not /proc: Windows has none
                cmd = " ".join(psutil.Process(pid).cmdline()) if pid else ""
            except Exception:  # noqa: BLE001 — gone, or not ours to read
                cmd = ""
            if str(script) not in cmd:
                raise MediaUnavailable(
                    f"Port {_port()} is taken by another program — media_search.port"
                )
            logger.info(
                "Media service: a leftover one answers on %s (pid %s) — stopping it",
                self.url(),
                pid,
            )
            end_pid(pid)
            for _ in range(60):
                if not self._responds(0.5):
                    break
                time.sleep(0.25)
        folder = debug.logs_root()
        folder.mkdir(parents=True, exist_ok=True)
        self._log = folder / "media-service.log"
        from whisper_tm.core import llama_server

        llama_server.trim_log(self._log)
        env = dict(os.environ)
        env.update(
            {
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONUNBUFFERED": "1",
            }
        )
        if environment.check().get("model_cached"):
            env["HF_HUB_OFFLINE"] = "1"
        # A key of this start, by environment (argv shows in ps): the
        # service answers only requests that carry it — a web page that
        # posts to 127.0.0.1:8083 can't make it read files or write stills
        import secrets

        self._token = secrets.token_hex(24)
        env["WHISPER_TM_MEDIA_TOKEN"] = self._token
        if str(_cfg("index_device", "cuda")).lower() == "cpu":
            # Nothing of it may reach the card then — not even a context
            env["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Media service starting: %s", " ".join(argv[:4]) + " …")
        with self._lock:
            with open(self._log, "ab") as fh:
                fh.write(f"\n=== Start {time.strftime('%F %T')} ===\n".encode())
                self._proc = subprocess.Popen(
                    argv,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                    cwd=str(script.parent),
                )
            self._address = (_host(), _port())
            self._started_at = time.time()
            self._last_used = time.monotonic()
        limit = _num("start_timeout_sec", 180)
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            if self._responds(2.0):
                logger.info("Media service ready after %.1f s", time.monotonic() - t0)
                with self._lock:
                    self._ready = self._alive()
                self._start_watchdog()
                return
            if not self._alive():
                tail = self._log_tail()
                self.stop("it ended during its start", force=True)
                raise MediaUnavailable(f"The media service ended during its start. {tail}")
            time.sleep(0.5)
        tail = self._log_tail()
        self.stop("not ready in time", force=True)
        raise MediaUnavailable(f"The media service was not ready after {limit:.0f} s. {tail}")

    def _log_tail(self, lines: int = 4) -> str:
        if not self._log:
            return ""
        from whisper_tm.core import llama_server

        return llama_server.log_tail(self._log, lines)

    # ── Talking ──────────────────────────────────────────────────────────

    def call(self, path: str, payload: dict | None = None, *, timeout: float = 120) -> dict:
        """POST ``payload`` to ``path`` — started first when needed."""
        self.ensure()
        with self._lock:
            self._busy += 1
            self._last_used = time.monotonic()
        try:
            req = urllib.request.Request(
                self.url() + path,
                data=json.dumps(payload or {}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Media-Token": getattr(self, "_token", ""),
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                try:
                    body = json.load(e)
                except ValueError:
                    body = {}
                err = MediaUnavailable(body.get("error") or f"HTTP {e.code}")
                err.status = e.code
                err.oom = bool(body.get("oom"))
                raise err from e
            except (urllib.error.URLError, OSError) as e:
                if not self._alive():
                    tail = self._log_tail()
                    self.stop("it ended", force=True)
                    raise MediaUnavailable(f"The media service ended. {tail}") from e
                raise MediaUnavailable(f"The media service did not answer: {e}") from e
        finally:
            with self._lock:
                self._busy -= 1
                self._last_used = time.monotonic()

    # ── Stopping ─────────────────────────────────────────────────────────

    def stop(self, reason: str = "", *, force: bool = False) -> bool:
        with self._lock:
            if self._busy and not force:
                return False
            p = self._proc
            self._proc = None
            self._address = None
            self._ready = False
        if p is None or p.poll() is not None:
            return False
        logger.info("Media service stopping (pid %s)%s", p.pid, f": {reason}" if reason else "")
        if not signal_group(p.pid):
            p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if not signal_group(p.pid, kill=True):
                p.kill()
        return True

    def _start_watchdog(self) -> None:
        with self._lock:
            if self._watchdog and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(
                target=self._watch, name="media-service-watcher", daemon=True
            )
            self._watchdog.start()

    def _watch(self) -> None:
        while True:
            time.sleep(WATCH_TICK_SEC)
            with self._lock:
                if self._proc is None:
                    return
                if not self._alive():
                    logger.warning(
                        "Media service ended on its own (exit %s). %s",
                        self._proc.returncode,
                        self._log_tail(),
                    )
                    self._proc = None
                    self._address = None
                    self._ready = False
                    return
                if self._busy or self.indexing:
                    continue
                quiet = time.monotonic() - self._last_used
                limit = _num("idle_stop_sec", 600)
            # Only when it stopped: a call that came in between keeps it
            # running, and the watcher must stay for its idle time
            if limit > 0 and quiet >= limit and self.stop(f"unused for {quiet:.0f} s"):
                return


SERVER = MediaServer()


def on_shutdown() -> None:
    try:
        SERVER.stop("the app shuts down", force=True)
    except Exception:  # noqa: BLE001
        pass


atexit.register(on_shutdown)

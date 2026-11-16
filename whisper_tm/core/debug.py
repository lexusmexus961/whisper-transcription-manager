"""
Central debug mode: verbose logging, timing everywhere, a log file.

Enable (one of these is enough):
    ./start.sh --debug
    WHISPER_TM_DEBUG=1 python -m whisper_tm
    config.yaml:  logging.debug: true

What debug mode does:
  * Log level DEBUG for the whole app
  * Verbose format: time (ms), thread, module:line
  * Each run writes its own log file (logging.debug_dir)
  * EVERYTHING is timed: HTTP requests, subprocesses (ffmpeg/ffprobe/
    audio-separator), DB accesses, queue entries, pipeline steps
  * At the end: an overview of where the time went (summary())

Timing always runs (it costs practically nothing); only the output
depends on the switch. That way, even in normal operation, one can check
via /api/debug/timings what was slow.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("whisper_tm.debug")

# Libraries that spam the console unusable at DEBUG level. They stay
# at INFO even in debug mode — otherwise you can no longer see your own lines.
#
# torio/torch are here because they log three full tracebacks on startup
# (searching for FFmpeg libraries: FFmpeg6 and FFmpeg5 fail,
# FFmpeg4 works). This is normal and not an error — but in the log it looked
# like a crash and obscured the first 30 lines.
_NOISY_LIBS = (
    "matplotlib",
    "PIL",
    "urllib3",
    "asyncio",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "numba",
    "speechbrain",
    "werkzeug.serving",
    "charset_normalizer",
    "git",
    "torio",
    "torch",
    "torchaudio",
    "pytorch_lightning",
    "lightning",
)


class _NoiseFilter(logging.Filter):
    """Suppresses chatter from third-party libraries below WARNING.

    As a FILTER on the handler, not as setLevel() on the logger: the
    libraries are only imported on the first job (whisperx → lightning) and
    set their logger level THEMSELVES then — a setLevel() at server start
    would long since have been overwritten by that point. A handler filter
    always applies.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(_NOISY_LIBS)


class _RequestLineFilter(logging.Filter):
    """Outside debug mode: Werkzeug's line for every request only when it
    failed (status 400 and up). The page polls the queue and the meters
    every few seconds, and request lines carry ?file=, ?path= and ?q= — a
    log users attach to bug reports filled with their library's paths and
    search terms."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "werkzeug" or record.levelno >= logging.WARNING:
            return True
        if "This is a development server" in str(record.msg):
            # Werkzeug's red start warning read as an error on every first
            # start; the app says what it means in plain words instead
            # (__main__, docs/OPERATING.md → Behind a reverse proxy)
            return False
        args = record.args if isinstance(record.args, tuple) else ()
        if len(args) != 3 or '%s" %s %s' not in str(record.msg):
            return True  # the start lines ("Running on …")
        try:
            return int(str(args[1])) >= 400
        except ValueError:
            return True


#: A run's main log is cut into files of this size, this many kept
LOG_FILE_BYTES = 20 * 1024 * 1024
LOG_FILE_BACKUPS = 5


# ── Splitting the log by topic ───────────────────────────────────────
#
# A single log with everything in it is no longer readable once debug mode
# is active: scanner, DB transactions, HTTP and pipeline write in a jumble.
# Each category therefore additionally gets its own file. The main file
# still keeps EVERYTHING — so nothing is lost, only views are gained.
#
# Order matters: the first matching rule wins. That's why the more
# specific `whisper_tm.core.queue` comes before the more general `whisper_tm.core`.
_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("db", ("whisper_tm.database", "whisper_tm.sql")),
    # ytdl belongs to "queue": downloads run as jobs in job_manager,
    # you look for their messages next to the other job events.
    (
        "queue",
        (
            "whisper_tm.core.queue",
            "whisper_tm.jobs",
            "whisper_tm.incidents",  # incidents table (job warnings/errors)
            "whisper_tm.core.ytdl",
            # links from the browser extension: downloads into the queue
            "whisper_tm.core.ingest",
            "whisper_tm.core.gpu_gate",
        ),
    ),  # which lane waited for the GPU
    # NOTE: this needs the ACTUAL logger names, not the
    # module paths. Part of the code names its logger freely
    # (getLogger("whisper_tm.pipeline") in core/transcription.py), the
    # other uses __name__. Initially only module paths were here — the
    # pipeline wrote dozens of lines per transcription, and pipeline.log stayed
    # empty. The test test_every_logger_has_a_category now pins this down.
    #
    # "whisperx." WITH the dot: without it, it's a prefix of "whisper_tm"
    # and this category swallows the entire app's own code.
    (
        "pipeline",
        (
            "whisper_tm.pipeline",  # steps [1/8] … [8/8]
            "whisper_tm.subprocess",  # ffmpeg/ffprobe/separator
            "whisper_tm.audio_utils",
            "whisper_tm.debug",  # VRAM peaks per model
            "whisper_tm.core.transcription",
            "whisper_tm.core.vram",  # spare VRAM: the card's size
            # The diarization worker is its own process, but
            # a step of THIS pipeline — you look for its messages
            # (start, abort, timeout) among the
            # other pipeline lines.
            "whisper_tm.core.diar_process",
            # Voice vectors run as a step WITHIN transcription
            # (right after diarization) — you look for their messages
            # next to the other pipeline lines, not separately.
            "whisper_tm.core.speaker",
            "whisper_tm.core.speaker_knn",
            "whisper_tm.core.speaker_inventory",
            "whisper_tm.core.vocabulary",  # initial_prompt presets
            "whisper_tm.core.vocal_isolation",
            # the models a job loads, fetched before the first one
            "whisper_tm.core.prefetch",
            # exports (core/exporter.py, export_jobs.py by the prefix),
            # video and audio exports and clips: ffmpeg runs
            "whisper_tm.core.export",
            "whisper_tm.core.media_export",
            "whisper_tm.core.clip_render",
            "whisper_tm.core.helpers",  # skip list, artifact filter
            # the same filter on the transcripts that exist
            "whisper_tm.core.artifact_cleanup",
            # cutting a person's voice out of the recordings
            "whisper_tm.core.voice_extract",
            "whisperx.",
            "pyannote",
            "faster_whisper",
        ),
    ),
    (
        "scanner",
        (
            "whisper_tm.scanner",
            # continuous scanning: its rounds and the files it found
            "whisper_tm.core.autoscan",
            # the audio analysis reads the library the same way: its
            # run, its index, and what it found gone
            "whisper_tm.audio",
            # the conversation statistics: their background count
            # over every transcript, and what it could not count
            "whisper_tm.conversations",
        ),
    ),
    # core.security belongs to "api": it reports rejected roots, and those
    # come exclusively from requests — that's where you look for them too.
    # The MCP server belongs to "api": it is an HTTP client AGAINST this API,
    # and you look for its messages (unreachable, 401, rejected path) among
    # the same lines as those of the requests it triggers. It does run
    # in its own process — then this assignment doesn't apply, but
    # it doesn't hurt either, and it applies when imported into the app.
    (
        "api",
        (
            "whisper_tm.api",
            "whisper_tm.core.auth",
            "whisper_tm.core.security",
            "whisper_tm.mcp_server",
            # the load sampler behind /api/system/load and /history,
            # and whose VRAM it is (/api/system/vram)
            "whisper_tm.core.metrics",
            "whisper_tm.core.vram_procs",
            # the clip editor's waveform, asked for by
            # /api/clip/waveform — the same
            "whisper_tm.core.waveform",
        ),
    ),
    # "rag" is the LLM category: indexer, client AND the summarization
    # core (core/summarize.py) — that's where you look for LLM errors, whether
    # they come from a request or from the queue post-step.
    (
        "rag",
        (
            "whisper_tm.rag",
            "whisper_tm.core.llm_client",
            "whisper_tm.core.summarize",
            # the chat (Ask): the model and its MCP tool calls
            "whisper_tm.core.chat",
            # analyses: one task over many recordings, in parts
            "whisper_tm.core.analysis",
            # Best moments for a clip (the Clips tab)
            "whisper_tm.core.best_moments",
            # LLM pipeline package (chunking/correction/translation,
            # docs/LLM-PIPELINE.md) — same category, same search.
            "whisper_tm.core.llm",
            # Local llama.cpp server (start/check) belongs to the
            # same search as the client that talks to it.
            "whisper_tm.core.llama_server",
            "whisper_tm.core.local_key",
            # The on-demand embedding server. Belongs here and not
            # to "pipeline": if an indexing run hangs, you look for the
            # reason next to the indexer that spun it up.
            "whisper_tm.core.embed_server",
            # the model servers a crash left behind, ended at start
            "whisper_tm.core.helper_pids",
            # The on-demand reranker of the situation search — next
            # to the searches (whisper_tm.rag) that start it
            "whisper_tm.core.rerank_server",
            # ... and the GGUF rewrite that prepares its model
            "whisper_tm.core.gguf",
            # Images & sounds: its index, index runs, the service
            # process and what the language model adds — next to the
            # other searches and the LLM they use
            "whisper_tm.media",
            "whisper_tm.find",
            # copies of a recording, told by content (Find, the meaning search)
            "whisper_tm.core.copies",
            "whisper_tm.core.media_server",
            "whisper_tm.core.llm_provider",
            "whisper_tm.core.llm_queue",
            "whisper_tm.core.llm_tasks",
        ),
    ),
    ("trace", ("whisper_tm.trace",)),
)


class _CategoryFilter(logging.Filter):
    """Only lets sentences of one category through.

    HTTP lines aren't recognized by logger name — they come from
    ``whisper_tm`` itself — but by the marker set by the request hook.
    """

    def __init__(self, prefixes: tuple[str, ...], only_http: bool = False):
        super().__init__()
        self._prefixes = prefixes
        self._only_http = only_http

    def filter(self, record: logging.LogRecord) -> bool:
        if self._only_http:
            return getattr(record, "http", False) is True
        if getattr(record, "http", False) is True:
            return False  # HTTP has its own file
        return record.name.startswith(self._prefixes)


_enabled = False
_log_file: Path | None = None
_log_dir_run: Path | None = None
#: The log folder as setup() resolved it (logs_root)
_logs_root: Path | None = None
_lock = threading.Lock()
# label -> list of durations (seconds)
_timings: dict[str, list[float]] = defaultdict(list)


# ── Switch ──────────────────────────────────────────────────────────────


def is_enabled() -> bool:
    return _enabled


def log_file() -> Path | None:
    """Path of this run's main log file (None if nothing is being written).

    It contains everything except the call trace — that's kept entirely in
    trace.log, and would otherwise bloat main.log many times over. The
    other category files are additional views, not a replacement.
    """
    return _log_file


def log_dir() -> Path | None:
    """This run's directory (contains main.log and the category files)."""
    return _log_dir_run


def _resolve_logs_root(value) -> Path:
    d = Path(str(value or "").strip() or "logs-debug").expanduser()
    if d.is_absolute():
        return d
    # A relative one lies in the data folder — from the working directory,
    # logs-debug/ turned up wherever the server was started, and a systemd
    # unit without a WorkingDirectory couldn't log at all
    from whisper_tm.config import config

    return Path(config.data_dir) / d


def logs_root() -> Path:
    """The log folder: ``--log-dir``, else ``logging.debug_dir``, else
    ``logs-debug`` — a relative one in the data folder. Each run's log goes
    in here, and so do the logs of the servers the app starts (llama.cpp,
    the media service)."""
    if _logs_root is not None:
        return _logs_root
    from whisper_tm.config import config

    return _resolve_logs_root(config.get("logging", "debug_dir", default=""))


def _wanted(cli_debug: bool) -> bool:
    """Debug active? CLI flag OR environment variable OR config."""
    if cli_debug:
        return True
    if os.environ.get("WHISPER_TM_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from whisper_tm.config import config

        return bool(config.get("logging", "debug", default=False))
    except Exception:
        return False


# ── Setup ─────────────────────────────────────────────────────────────


class MeasurePoints:
    """Where exactly does the time go WITHIN a request?

    The request timing in ``app.py`` says "2.1 min" and nothing else.
    A request like ``/api/speaker/sample/<id>`` can take minutes while
    the ffmpeg inside it is logged at a few milliseconds. The rest of the
    time sits between two log lines — and thus in the dark. This is
    exactly what this is for.

    Usage::

        mp = debug.MeasurePoints("Audio sample 42")
        row = db.read()           ; mp.point("DB")
        check_path()              ; mp.point("Path check")
        cut()                     ; mp.point("ffmpeg")
        mp.done()

    Result, one line::

        ⏱ Audio sample 42: 2.1min — DB 1ms · Path check 2.1min · ffmpeg 115ms

    **Nearly free outside debug mode**: it comes down to a few
    ``perf_counter`` calls and a list; logging only happens in
    debug mode. The sums are ALWAYS still forwarded to :func:`record` — so
    a slow step also shows up in ``/api/debug/timings``, without
    anyone having had to switch on debug mode beforehand.
    """

    __slots__ = ("name", "_t0", "_last", "_steps")

    def __init__(self, name: str):
        self.name = name
        self._t0 = self._last = time.perf_counter()
        self._steps: list[tuple[str, float]] = []

    def point(self, what: str) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self._steps.append((what, dt))
        record(f"{self.name} · {what}", dt)
        return dt

    def steps(self) -> list[tuple[str, float]]:
        """The intermediate points, as measured.

        So an endpoint can INCLUDE its own breakdown, instead of
        just writing it to the log — precise timing that shows in
        production what costs time where. Whoever has to
        search the log for the numbers first will never see them.
        """
        return list(self._steps)

    def done(self, extra: str = "") -> float:
        total = time.perf_counter() - self._t0
        record(self.name, total)
        if _enabled:
            parts = " · ".join(f"{w} {fmt_secs(d)}" for w, d in self._steps)
            _pipe_log.debug(
                "⏱ %s: %s — %s%s",
                self.name,
                fmt_secs(total),
                parts or "(no intermediate points)",
                f" · {extra}" if extra else "",
            )
        return total


def _tame_whisperx_logger() -> None:
    """whisperx hijacks its own logger — undo that.

    ``whisperx/log_utils.py`` calls ``setup_logging()`` on the FIRST
    ``get_logger(__name__)`` (i.e. on import of ``whisperx.diarize`` and
    siblings), and that does three things:

    1. ``logging.getLogger("whisperx").setLevel(INFO)``
    2. attaches its own StreamHandler to **stdout**
    3. ``propagate = False``

    Point 3 is the bad one: with that, **not a single whisperx message
    reaches the root's handlers anymore** — not even WARNING. Measured:
    neither ``main.log`` nor ``pipeline.log`` contained a
    warning from ``whisperx.diarize``; it only showed up on stdout. The
    ``"whisperx."`` category in :data:`_CATEGORIES` therefore ran into
    nothing, and the comment on :class:`_NoiseFilter` ("a handler filter
    always applies") was wrong for this one case: a filter on the root
    handler does nothing if the record never reaches the root.

    Two moves, both needed:

    * **Undo it**, in case whisperx has already struck (import before
      ``setup()``).
    * **Preempt it**: a ``NullHandler`` satisfies the
      ``if not whisperx_logger.handlers`` check in ``get_logger`` — so
      ``setup_logging()`` doesn't even run when whisperx is imported later
      (normally on the first job).
    """
    lg = logging.getLogger("whisperx")
    for h in list(lg.handlers):
        if not isinstance(h, logging.NullHandler):
            lg.removeHandler(h)
    lg.propagate = True
    lg.setLevel(logging.NOTSET)  # inherits from the root again
    lg.addHandler(logging.NullHandler())


def setup(
    cli_debug: bool = False, log_dir: str | Path | None = None, trace_calls: bool = False
) -> Path | None:
    """Configure logging. Returns the log file's path (or None).

    Called from __main__, BEFORE the app is built.
    """
    global _enabled, _log_file, _log_dir_run, _logs_root

    _enabled = _wanted(cli_debug)

    from whisper_tm.config import config

    if _enabled:
        level = logging.DEBUG
        fmt = (
            "%(asctime)s.%(msecs)03d [%(levelname)-7s] "
            "%(threadName)-14s %(name)s:%(lineno)d — %(message)s"
        )
    else:
        level_name = config.get("logging", "level", default="INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root = logging.getLogger()
    # basicConfig() no longer applies once handlers are already attached → set manually
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    noise = _NoiseFilter()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    console.addFilter(noise)
    # The call trace does NOT belong on the console: it generates tens of
    # thousands of lines per second and makes the terminal unusable. It's in
    # trace.log and in main.log — you can read it there at leisure.
    console.addFilter(lambda r: not r.name.startswith("whisper_tm.trace"))
    root.addHandler(console)

    # ── Log file ──────────────────────────────────────────────────
    if log_dir is None:
        log_dir = config.get("logging", "debug_dir", default="")
    _logs_root = _resolve_logs_root(log_dir)

    _log_file = None
    _log_dir_run = None
    if _enabled or config.get("logging", "persist", default=True):
        try:
            d = _logs_root
            d.mkdir(parents=True, exist_ok=True)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            file_format = logging.Formatter(
                "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(threadName)-14s "
                "%(name)s:%(lineno)d — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Splitting is only worth it in debug mode — without it, not much
            # is happening anyway, and a folder full of nearly empty files helps no one.
            split_up = _enabled and config.get("logging", "split", default=True)

            if split_up:
                _log_dir_run = d / f"whisper_tm_{stamp}"
                _log_dir_run.mkdir(parents=True, exist_ok=True)
                _log_file = _log_dir_run / "main.log"
            else:
                _log_file = d / f"whisper_tm_{stamp}.log"

            # Rotating: with restart: unless-stopped or systemd one run lasts
            # weeks, and a plain file grew without a bound
            from logging.handlers import RotatingFileHandler

            fh = RotatingFileHandler(
                _log_file, maxBytes=LOG_FILE_BYTES, backupCount=LOG_FILE_BACKUPS, encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG if _enabled else level)
            fh.setFormatter(file_format)
            fh.addFilter(noise)
            if split_up:
                # Do NOT additionally write the call trace to main.log: with
                # tracing active it makes up practically the entire content
                # and thus doubles the space needed. It's kept fully in
                # trace.log; main.log keeps everything else.
                fh.addFilter(lambda r: not r.name.startswith("whisper_tm.trace"))
            root.addHandler(fh)

            if split_up:
                for name, prefixes in (*_CATEGORIES, ("http", ())):
                    try:
                        kh = logging.FileHandler(_log_dir_run / f"{name}.log", encoding="utf-8")
                        kh.setLevel(logging.DEBUG)
                        kh.setFormatter(file_format)
                        kh.addFilter(noise)
                        kh.addFilter(_CategoryFilter(prefixes, only_http=(name == "http")))
                        root.addHandler(kh)
                    except OSError as e:
                        logging.getLogger("whisper_tm").warning(
                            "Could not create category log %s: %s", name, e
                        )

            _prune_old_logs(d, keep=config.get("logging", "keep_runs", default=20))
        except Exception as e:  # noqa
            logging.getLogger("whisper_tm").warning(
                "Log file could not be created (%s): %s", log_dir, e
            )
            _log_file = None
            _log_dir_run = None

    # Rein in the spammers. WARNING, not INFO: these libraries also chatter
    # irrelevant stuff at INFO (matplotlib builds its font cache, lightning
    # reports a checkpoint upgrade) — right in the middle of pipeline sections.
    # Domain-relevant output (whisperx, pyannote) is deliberately NOT here.
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _tame_whisperx_logger()

    if _enabled:
        # Werkzeug logs every request itself ("GET /api/health 200"). In
        # debug mode we do that with timing and client IP — otherwise
        # every request would show up twice in the log.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    else:
        wz = logging.getLogger("werkzeug")
        if not any(isinstance(f, _RequestLineFilter) for f in wz.filters):
            wz.addFilter(_RequestLineFilter())

    if _enabled:
        log = logging.getLogger("whisper_tm")
        log.debug("═══ DEBUG MODE ACTIVE ═══")
        if _log_dir_run:
            log.info(
                "Debug logs: %s/ (main.log + %s)",
                _log_dir_run,
                ", ".join(n for n, _ in (*_CATEGORIES, ("http", ()))),
            )
        elif _log_file:
            log.info("Debug log: %s", _log_file)

    # Call tracing only when explicitly requested — it halves the speed.
    if _enabled and (trace_calls or config.get("logging", "trace_calls", default=False)):
        start_call_tracing()

    return _log_file


def _prune_old_logs(d: Path, keep: int) -> None:
    """Only keep the last `keep` runs.

    A run is either a single file (without splitting) or a
    folder with the category files. Both are treated together here,
    otherwise changing the setting would leave the other form lying
    around forever.
    """
    if keep <= 0:
        return
    # A run's file and its rotated parts (.log.1 …) go together
    runs = [p for p in d.glob("whisper_tm_*") if p.is_dir() or p.suffix == ".log"]
    parts = {p for p in d.glob("whisper_tm_*.log.*") if p.suffix[1:].isdigit()}
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    kept = {p.name for p in runs[:keep]}
    for old in runs[keep:]:
        try:
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
            else:
                old.unlink()
        except OSError:
            pass
    for part in parts:
        if part.name.rsplit(".", 1)[0] not in kept:
            with contextlib.suppress(OSError):
                part.unlink()


# ── Call tracing: every function, without touching every function ────────────
#
# The goal is logs for everything in every function. By hand that
# would be thousands of lines that go stale again with the next refactor.
# Instead, a profiler hook attaches to the interpreter: it reports EVERY
# entry into and exit from a function, along with arguments, duration and
# return value.
#
# This is deliberately not something that runs along on the side:
#   * The hook fires on every single call. Expect roughly
#     half the speed; at call-heavy spots (a scan over a large
#     library) noticeably less too.
#   * The data volume is enormous. Hence a hard upper limit: once it's
#     reached, tracing switches itself off and reports it, instead of
#     filling up the disk.
# Enable via logging.trace_calls (or --trace-calls).

_trace_logger = logging.getLogger("whisper_tm.trace")
_trace_active = False
_trace_counter = 0
_trace_limit = 0
_trace_max_bytes = 0
# Don't check the file size on every event — a stat() per
# function call would cost more than the logging itself.
_TRACE_CHECK_INTERVAL = 50_000
_trace_args = False
_trace_package_path = ""
_trace_exceptions: tuple[str, ...] = ()
_trace_depth = threading.local()
# File name → module name. The hook runs per call, so the
# conversion may only happen once per file.
_trace_module_names: dict[str, str] = {}


def _module_name(file: str) -> str:
    """'…/whisper_tm/database/__init__.py' → 'database'.

    Not the file stem: for packages, every function would be called
    '__init__.foo', which doesn't distinguish anything in the trace.
    """
    name = _trace_module_names.get(file)
    if name is None:
        try:
            rel = Path(file).relative_to(_trace_package_path).with_suffix("")
            parts = [t for t in rel.parts if t != "__init__"]
            name = ".".join(parts) or "whisper_tm"
        except ValueError:
            name = Path(file).stem
        _trace_module_names[file] = name
    return name


#: Secret values from the configuration (API keys, tokens, the password):
#: a traced argument or return value that contains one is not written.
#: Collected when tracing starts, then every _SECRETS_REFRESH_S seconds by a
#: thread of its own — never from inside the hook, which must not take the
#: config lock or run package code.
_trace_secrets: frozenset[str] = frozenset()
_SECRETS_REFRESH_S = 10.0
_REDACTED = "<redacted>"
#: Per thread: "collecting the secrets right now — the hook ignores me".
_trace_local = threading.local()
_refresher_stop = threading.Event()


def _collect_secrets() -> frozenset[str]:
    """Every set secret-looking value of the configuration (and the MCP token)."""
    out: set[str] = set()
    try:
        from whisper_tm.config import config, is_secret

        def walk(node, key=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, str(k))
            elif isinstance(node, list):
                for v in node:
                    walk(v, key)
            elif isinstance(node, str) and is_secret(key) and len(node.strip()) >= 4:
                out.add(node.strip())

        walk(config.all())
    except Exception:  # noqa — redaction by name still applies
        pass
    try:
        from whisper_tm.mcp_server import token_file

        text = token_file().read_text(encoding="utf-8").strip()
        if len(text) >= 4:
            out.add(text)
    except Exception:  # noqa
        pass
    return frozenset(out)


_secret_names: dict[str, bool] = {}


def _is_secret_name(name: str) -> bool:
    """Is ``name`` (a parameter or function) named like a secret? Cached —
    the trace hook asks for every call."""
    hit = _secret_names.get(name)
    if hit is None:
        try:
            from whisper_tm.config import is_secret

            hit = is_secret(name)
        except Exception:  # noqa
            hit = False
        _secret_names[name] = hit
    return hit


def _brief(value, limit: int = 120) -> str:
    """Condense a value for the log line — never the whole data block.

    A transcript dict or an audio array as repr() in the log would be
    megabytes per line and would ruin all readability.
    """
    try:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<{type(value).__name__} {len(value)} B>"
        if isinstance(value, (list, tuple, set, dict)):
            return f"<{type(value).__name__} n={len(value)}>"
        if (
            isinstance(value, str)
            and _trace_secrets
            and any(sec in value for sec in _trace_secrets)
        ):
            return _REDACTED
        t = repr(value)
    except Exception:
        return "<repr failed>"
    if _trace_secrets and any(sec in t for sec in _trace_secrets):
        return _REDACTED
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _trace_file_bytes() -> int:
    """Current size of trace.log (0 if it doesn't exist)."""
    try:
        if _log_dir_run:
            return (_log_dir_run / "trace.log").stat().st_size
    except OSError:
        pass
    return 0


def _trace_hook(frame, event, arg):
    """sys.setprofile callback. Must be extremely cheap — it runs everywhere."""
    global _trace_counter

    # During shutdown CPython clears the module globals and sets them to
    # None — but the hook keeps running as long as it's installed. Without
    # this guard, every Ctrl-C at the end threw a traceback:
    #   TypeError: startswith first arg must be str …, not NoneType
    # (actually observed on shutdown, triggered from the logging teardown).
    if (
        not _trace_active
        or not isinstance(_trace_package_path, str)
        or not _trace_package_path
        or sys.is_finalizing()
    ):
        return

    if event not in ("call", "return"):
        return  # c_call/c_return: foreign code, uninteresting
    code = frame.f_code
    file = code.co_filename
    # Own code only. The check comes early, because it
    # discards the overwhelming majority of events right away.
    if not file.startswith(_trace_package_path):
        return
    if _trace_exceptions and any(a in file for a in _trace_exceptions):
        return
    # Tracing must not trace itself.
    if code.co_name.startswith("_trace") or "core/debug.py" in file:
        return

    if getattr(_trace_local, "collecting", False):
        return  # the secret collection's own calls
    _trace_counter += 1
    if _trace_limit and _trace_counter > _trace_limit:
        stop_call_tracing(reason=f"upper limit of {_trace_limit} events reached")
        return
    # Emergency brake via file size: that's the number that counts when it
    # comes to whether the disk fills up. Only check every few tens of
    # thousands of events, otherwise the check costs more than the writing.
    if _trace_max_bytes and _trace_counter % _TRACE_CHECK_INTERVAL == 0:
        if _trace_file_bytes() > _trace_max_bytes:
            stop_call_tracing(reason=f"trace.log over {_trace_max_bytes // (1024 * 1024)} MB")
        return

    # Start times per FRAME, not on a stack. A stack assumes strict
    # nesting — generators violate that: every next() is a
    # 'call', every yield a 'return', and other calls run in between.
    # The stack got thrown out of step by this, and a 'return' grabbed the
    # start time of a completely different call. So the trace showed
    # "_row_to_item 288.00s" for a function that takes microseconds.
    # (A run contains hundreds of thousands of generator calls — not an edge case.)
    #
    # NOT frame.f_locals: on optimized frames that's a snapshot, not
    # storage; from Python 3.13 (PEP 667) the value would vanish there.
    times = getattr(_trace_depth, "times", None)
    if times is None:
        times = _trace_depth.times = {}
    depth = getattr(_trace_depth, "depth", 0)
    place = f"{_module_name(file)}.{code.co_name}"

    if event == "call":
        times[id(frame)] = time.perf_counter()
        _trace_depth.depth = depth + 1
        if _trace_args:
            try:
                n = code.co_argcount
                names = code.co_varnames[:n]
                # A parameter named like a secret (password, token, api_key)
                # is never written, whatever it holds
                value_list = [
                    f"{k}={_REDACTED if _is_secret_name(k) else _brief(frame.f_locals.get(k))}"
                    for k in names
                    if k not in ("self", "cls")
                ]
                _trace_logger.debug(
                    "%s→ %s(%s)", "  " * min(depth, 20), place, ", ".join(value_list)
                )
            except Exception:
                _trace_logger.debug("%s→ %s(?)", "  " * min(depth, 20), place)
        else:
            _trace_logger.debug("%s→ %s", "  " * min(depth, 20), place)
    else:
        # A 'return' without a matching 'call' happens when tracing was
        # switched on in the middle of a call — then the duration stays
        # empty, instead of claiming a wrong one.
        t0 = times.pop(id(frame), None)
        depth = max(0, depth - 1)
        _trace_depth.depth = depth
        duration = f" {fmt_secs(time.perf_counter() - t0)}" if t0 is not None else ""
        # A function named like a secret (_get_password, access_token,
        # hf_token) returns one: its value is never written
        shown = _REDACTED if _is_secret_name(code.co_name) else _brief(arg)
        _trace_logger.debug("%s← %s%s = %s", "  " * min(depth, 20), place, duration, shown)


def start_call_tracing(with_arguments: bool | None = None) -> bool:
    """Switch tracing on. Returns whether it's actually running."""
    global _trace_active, _trace_limit, _trace_args, _trace_max_bytes
    global _trace_package_path, _trace_exceptions, _trace_counter

    if _trace_active:
        return True
    try:
        from whisper_tm.config import config

        # No event limit by default: millions can be used up by
        # startup and scan alone, before a transcription could even
        # begin. It's limited via file size instead — that's the
        # number that actually matters.
        _trace_limit = int(config.get("logging", "trace_calls_max", default=0) or 0)
        _trace_max_bytes = (
            int(config.get("logging", "trace_calls_max_mb", default=8000) or 0) * 1024 * 1024
        )
        _trace_args = (
            config.get("logging", "trace_call_args", default=True)
            if with_arguments is None
            else with_arguments
        )
        _trace_exceptions = tuple(config.get("logging", "trace_calls_exclude", default=[]) or [])
    except Exception:
        _trace_limit, _trace_args, _trace_exceptions = 0, True, ()
        _trace_max_bytes = 8000 * 1024 * 1024

    # Don't plan for more space than is free. The trace grows fast and
    # a system disk filled up is a much bigger problem
    # than a truncated trace. A fifth of the free space is the
    # upper limit; at least 200 MB is always allowed.
    if _log_dir_run and _trace_max_bytes:
        try:
            free = shutil.disk_usage(_log_dir_run).free
            cap = max(200 * 1024 * 1024, free // 5)
            if _trace_max_bytes > cap:
                logger.warning(
                    "trace_calls_max_mb (%d MB) is above what's free here "
                    "(%.1f GB) — limiting to %d MB.",
                    _trace_max_bytes // (1024 * 1024),
                    free / 1e9,
                    cap // (1024 * 1024),
                )
                _trace_max_bytes = cap
        except OSError:
            pass

    import whisper_tm

    _trace_package_path = str(Path(whisper_tm.__file__).parent)
    _refresh_trace_secrets()  # before the hook goes in: untraced
    _trace_counter = 0
    _trace_active = True
    _refresher_stop.clear()
    threading.Thread(target=_secrets_refresher, name="trace-secrets", daemon=True).start()

    # threading.setprofile FIRST: it applies to threads started from now on.
    # sys.setprofile only affects the calling one. Without both, of all
    # things queue worker and scan pool were missing from the trace.
    threading.setprofile(_trace_hook)
    sys.setprofile(_trace_hook)
    # Unhook the hook again in good time. atexit runs BEFORE the
    # interpreter clears the module globals — without this, the hook ran
    # into the teardown phase and threw tracebacks there.
    atexit.register(stop_call_tracing, "Program end")
    logger.warning(
        "Call tracing ACTIVE — every function call is being "
        "logged. This costs noticeable speed. Stops at %s%s.",
        f"{_trace_max_bytes // (1024 * 1024)} MB trace.log"
        if _trace_max_bytes
        else "unlimited file size",
        f" or {_trace_limit:,} events".replace(",", ".") if _trace_limit else "",
    )
    return True


def _refresh_trace_secrets() -> None:
    """Re-read the secret values (a key entered while tracing runs).

    The collection runs package code; a thread-local flag makes the hook
    skip it in this thread only. The global switch is never touched — a
    stop that came in meanwhile stays a stop, other threads stay traced."""
    global _trace_secrets
    _trace_local.collecting = True
    try:
        _trace_secrets = _collect_secrets()
    finally:
        _trace_local.collecting = False


def _secrets_refresher() -> None:
    """Thread: refresh the secret list while tracing runs."""
    _trace_local.collecting = True  # this thread is never traced
    while not _refresher_stop.wait(_SECRETS_REFRESH_S):
        if not _trace_active:
            return
        try:
            _refresh_trace_secrets()
        finally:
            _trace_local.collecting = True


def stop_call_tracing(reason: str = "") -> None:
    global _trace_active
    if not _trace_active:
        return
    # Switch off first, only then report: unhooking must not fail just
    # because logging is already half torn down during shutdown.
    _trace_active = False
    _refresher_stop.set()
    try:
        sys.setprofile(None)
        threading.setprofile(None)
    except Exception:
        pass
    try:
        logger.warning(
            "Call tracing stopped after %s events%s",
            f"{_trace_counter:,}".replace(",", "."),
            f" ({reason})" if reason else "",
        )
    except Exception:
        pass


def call_tracing_active() -> bool:
    return _trace_active


def trace_events() -> int:
    """How many call events have been logged so far."""
    return _trace_counter


def trace_log_bytes() -> int:
    """Current size of trace.log in bytes."""
    return _trace_file_bytes()


def record(label: str, secs: float) -> None:
    """Store a measurement (even without debug mode — it costs nothing)."""
    with _lock:
        _timings[label].append(secs)


@contextmanager
def timed(label: str, log_level: int = logging.DEBUG, **extra):
    """Time a block.

    with debug.timed("Audio extraction", file=name):
        ...
    """
    t0 = time.perf_counter()
    err = None
    try:
        yield
    except BaseException as e:
        err = e
        raise
    finally:
        dt = time.perf_counter() - t0
        record(label, dt)
        if _enabled:
            bits = " ".join(f"{k}={v}" for k, v in extra.items())
            status = f" ERROR({type(err).__name__})" if err else ""
            logger.log(
                log_level, "⏱ %s: %s%s%s", label, fmt_secs(dt), f" · {bits}" if bits else "", status
            )


def fmt_secs(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


def fmt_mb(mb: float) -> str:
    """Memory size in MB, human-readable (from 1 GB onwards in GB)."""
    if mb is None:
        return "?"
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


# ── GPU VRAM measurement ──────────────────────────────────────────────────────
# Goal: capture the PEAK GPU memory usage per model, in order to plan
# which model should move to a second GPU.
#
# Why NOT torch.cuda.max_memory_allocated()? The three most expensive models
# escape this measurement:
#   * WhisperX runs via ctranslate2 (faster-whisper) — its own CUDA
#     allocator, NOT torch. torch.cuda.*_allocated sees none of it.
#   * audio-separator runs as a SUBPROCESS with its own CUDA context — fully
#     invisible to the main process.
#   * Only alignment/diarization are genuine in-process torch models.
#
# So instead we measure the TOTAL memory used PER GPU via NVML/nvidia-smi
# (captures all processes). A background thread samples during the block and
# remembers the maximum — that way, short loading spikes and the subprocess
# consumption of separation are captured too.
#
# Deliberately NOT torch.cuda.mem_get_info(idx): that would create a
# CUDA context in the main process on a GPU that might not even be used
# (~300 MB) and would skew exactly the measurement we need for multi-GPU
# planning. NVML/nvidia-smi read without their own context.

_vram_lock = threading.Lock()
# label -> {gpu_index: {"peak_mb": int, "delta_mb": int}}
_vram: dict[str, dict[int, dict]] = defaultdict(dict)
#: Memory per step. Its own storage instead of a pseudo-GPU axis:
#: RAM has no device number, and smuggling it into `_vram` under index -1
#: would have tripped up every evaluation over it.
_ram: dict[str, dict] = {}
#: How often memory is read per GPU sampling tick.
RAM_TICKS_PER_VRAM_TICK = 5

_gpu_backend: str | None = None  # "pynvml" | "smi" | "none"
_pynvml = None
_gpu_static: dict[int, dict] | None = None  # {idx: {"name","total_mb"}} — static


def _init_gpu_backend() -> str:
    """Determine the fastest available read backend, once.

    pynvml (in-process, no subprocess) before nvidia-smi. Both read the
    memory WITHOUT creating their own CUDA context.
    """
    global _gpu_backend, _pynvml
    if _gpu_backend is not None:
        return _gpu_backend
    try:
        import pynvml  # nvidia-ml-py

        pynvml.nvmlInit()
        _pynvml = pynvml
        _gpu_backend = "pynvml"
        return _gpu_backend
    except Exception:
        pass
    _gpu_backend = "smi" if shutil.which("nvidia-smi") else "none"
    return _gpu_backend


def gpu_available() -> bool:
    return _init_gpu_backend() != "none"


def _smi_query(fields: str) -> list[list[str]]:
    """Query nvidia-smi --query-gpu → list of field values per GPU."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return []
        return [
            [c.strip() for c in line.split(",")]
            for line in r.stdout.strip().split("\n")
            if line.strip()
        ]
    except Exception:
        return []


def gpu_mem_used_mb() -> dict[int, int]:
    """Currently used memory PER GPU in MB (all processes). {} if no GPU."""
    be = _init_gpu_backend()
    if be == "pynvml":
        try:
            out = {}
            for i in range(_pynvml.nvmlDeviceGetCount()):
                h = _pynvml.nvmlDeviceGetHandleByIndex(i)
                out[i] = int(_pynvml.nvmlDeviceGetMemoryInfo(h).used) // (1024 * 1024)
            return out
        except Exception:
            pass
    if be in ("smi", "pynvml"):
        out = {}
        for parts in _smi_query("index,memory.used"):
            if len(parts) >= 2:
                try:
                    out[int(parts[0])] = int(float(parts[1]))
                except ValueError:
                    pass
        return out
    return {}


def gpu_used_and_util() -> dict[int, tuple[int, int | None]]:
    """Used memory (MB) and load (%) per GPU, in ONE query.

    The metrics sampler reads both every 2 s; through nvidia-smi each call
    costs ~19 ms, and two separate ones were double that — now the budget
    the per-process VRAM query (core/vram_procs.py) needs."""
    be = _init_gpu_backend()
    if be == "pynvml":
        try:
            out = {}
            for i in range(_pynvml.nvmlDeviceGetCount()):
                h = _pynvml.nvmlDeviceGetHandleByIndex(i)
                out[i] = (
                    int(_pynvml.nvmlDeviceGetMemoryInfo(h).used) // (1024 * 1024),
                    int(_pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
                )
            return out
        except Exception:
            pass
    if be in ("smi", "pynvml"):
        out = {}
        for parts in _smi_query("index,memory.used,utilization.gpu"):
            if len(parts) >= 3:
                with contextlib.suppress(ValueError):
                    util = None
                    with contextlib.suppress(ValueError):
                        util = int(float(parts[2]))
                    out[int(parts[0])] = (int(float(parts[1])), util)
        return out
    return {}


# ── This process's own memory ────────────────────────────────────────────
#
#  The diarization worker measures itself (`diar_process._rss_gb`) and
#  is discarded at 6 GiB. The SERVER PROCESS, where all the remaining
#  processing runs, was measured nowhere — for a run spanning days, that is
#  the number you need to even answer "is it growing?". In one measured
#  run it stood at several GiB RSS plus several GiB swapped out after hours
#  of jobs; whether that's a baseline or a trend couldn't be said for lack of
#  a record.
#
#  Swapped-out memory counts too: a process whose RSS looks calm
#  while its swap grows isn't frugal, it's being displaced.

_last_rss: float | None = None
_floor: float | None = None  # smallest footprint ever seen


def own_memory() -> tuple[float, float]:
    """(RSS, swap) of this process in GiB. (0,0) where /proc is missing."""
    rss = swap = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1048576
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1]) / 1048576
    except (OSError, ValueError, IndexError):
        pass
    return rss, swap


def memory_report() -> str:
    """One line for the job log — with the change since the last time.

    The change is the value that actually matters: an absolute level of
    7 GiB says nothing (models stay loaded on purpose), but a gain of
    0.4 GiB after every job says a lot.

    State is carried forward without a lock: jobs run one after another
    in the queue worker, and even a shifted comparison value would only be
    an imprecise log line, not harm.
    """
    global _last_rss, _floor
    rss, swap = own_memory()
    if rss <= 0:
        return ""
    parts = [f"{rss:.2f} GiB in RAM"]
    if swap > 0.05:
        parts.append(f"{swap:.2f} GiB swapped out")
    if _last_rss is not None:
        parts.append(f"Δ {rss - _last_rss:+.2f} GiB since the last job")
    _last_rss = rss

    # The FLOOR — the smallest footprint ever seen.
    #
    # The single value at the end of a job is unsuitable as a leak
    # indicator: across consecutive jobs it swings by whole GiB in both
    # directions. It hits a random point in the heap's ups and downs, not the consumption.
    #
    # The floor can do it: it rises ONLY when memory is genuinely held
    # permanently. Fluctuations around it are the normal coming and
    # going of processing; a drifting minimum is a leak.
    #
    # Computed on RSS + swapped-out: a process whose RSS drops because
    # the OS pushed pages into swap has released nothing.
    total = rss + swap
    if _floor is None or total < _floor:
        _floor = total
    parts.append(f"Floor {_floor:.2f} GiB")
    return " · ".join(parts)


#: Without a threshold of its own: warn above this share of the machine's
#: memory. A fixed 20 GB never fired on a 16 GB machine (it swapped first)
#: and fired after every job on one whose media index alone holds more.
MEMORY_WARNING_SHARE = 0.6


def memory_warning_gb() -> float:
    """The warning threshold in GiB: ``hardware.server_memory_warning_gb``,
    empty for 60 % of the machine's memory, 0 for off."""
    try:
        from whisper_tm.config import config as _cfg

        raw = _cfg.get("hardware", "server_memory_warning_gb", default="")
    except Exception:
        raw = ""
    if str(raw).strip():
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass  # not a number: automatic
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3 * MEMORY_WARNING_SHARE
    except Exception:
        return 20.0


def memory_limit_reached() -> str:
    """Is the footprint above the warning threshold? Returns: text or "".

    **Why a threshold and not a fix — and why one anyway.**

    Once it looked like a leak: the first few measured jobs each
    added a little over a tenth of a GiB, and that was extrapolated over a
    long queue. **That interpretation was wrong.** Over more jobs the
    spread of the per-job change (total: RSS + swap) turned out to be
    several times its mean, the footprint oscillated within a narrow
    band for hours, and some jobs GAVE BACK memory. That's a
    plateau with noise, not growth.

    Two lessons meant to stay here: three measurements pointing the
    same way are not a trend, and anyone looking only at RSS misses that
    swapping happened at the same time.

    The threshold stays anyway — as a safety net, not an alarm. It costs
    nothing, and should a later refactor actually accumulate something, it
    will speak up before the machine starts swapping. Deliberately NO
    automation: a server that restarts itself takes the decision out of
    the operator's hands at exactly the moment it matters.
    """
    limit = memory_warning_gb()
    if limit <= 0:
        return ""
    rss, swap = own_memory()
    total = rss + swap
    if total < limit:
        return ""
    # No growth rate in this text. It once claimed a fixed growth per job
    # as a fixed string — the very figure the docstring above
    # shows to be wrong — and once it fired during a single heavy
    # request while the jobs themselves stayed flat. The floor
    # is the measured value that tells the two apart.
    floor = f"{_floor:.1f} GiB" if _floor is not None else "not measured yet"
    return (
        f"The server process is using {total:.1f} GiB, more than the "
        f"{limit:.0f} GiB it is warned at "
        f"({rss:.1f} in RAM, {swap:.1f} swapped out; lowest level so "
        f"far {floor}). A lowest level that keeps rising from job to job "
        f"means memory is being held — a restart returns it and only "
        f"costs the currently running job."
    )


def gpu_static() -> dict[int, dict]:
    """Static GPU info (name, total memory) — determined once & cached."""
    global _gpu_static
    if _gpu_static is not None:
        return _gpu_static
    out: dict[int, dict] = {}
    be = _init_gpu_backend()
    if be == "pynvml":
        try:
            for i in range(_pynvml.nvmlDeviceGetCount()):
                h = _pynvml.nvmlDeviceGetHandleByIndex(i)
                name = _pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                total = int(_pynvml.nvmlDeviceGetMemoryInfo(h).total) // (1024 * 1024)
                out[i] = {"name": name, "total_mb": total}
        except Exception:
            out = {}
    if not out and be != "none":
        for parts in _smi_query("index,name,memory.total"):
            if len(parts) >= 3:
                try:
                    out[int(parts[0])] = {"name": parts[1], "total_mb": int(float(parts[2]))}
                except ValueError:
                    pass
    _gpu_static = out
    return out


def ram_used_mb() -> int:
    """Memory of THIS process including all children, in MB.

    The children are the actual point. Diarization runs in
    its own process (``core/diar_worker``), so it can
    be cancelled — and exactly that one once grew to **tens of GB** in a
    night run and triggered the kernel OOM killer. The server process
    itself looked harmless in the meantime. Whoever measures only
    ``self.memory_info()`` doesn't measure the place where it actually blows up.
    """
    try:
        import psutil

        p = psutil.Process()
        total_sum = p.memory_info().rss
        for k in p.children(recursive=True):
            try:
                total_sum += k.memory_info().rss
            except Exception:  # noqa: BLE001
                continue  # child already gone
        return total_sum // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0


def _record_vram(label: str, gpu_index: int, peak_mb: int, delta_mb: int) -> None:
    """Store peak values per (model, GPU) — always keep the maximum."""
    with _vram_lock:
        cur = _vram[label].get(gpu_index, {"peak_mb": 0, "delta_mb": 0})
        cur["peak_mb"] = max(cur["peak_mb"], int(peak_mb))
        cur["delta_mb"] = max(cur["delta_mb"], int(delta_mb))
        _vram[label][gpu_index] = cur


def _record_ram(label: str, peak_mb: int, delta_mb: int) -> None:
    """The same for memory — it has no GPU axis."""
    with _vram_lock:
        cur = _ram.get(label, {"peak_mb": 0, "delta_mb": 0})
        cur["peak_mb"] = max(cur["peak_mb"], int(peak_mb))
        cur["delta_mb"] = max(cur["delta_mb"], int(delta_mb))
        _ram[label] = cur


class _VramResult:
    """Result of a vram() block: peak and delta consumption per GPU."""

    def __init__(self):
        self.peak_mb: dict[int, int] = {}
        self.delta_mb: dict[int, int] = {}
        #: Memory (process tree) — peak and growth within the block.
        self.ram_peak_mb: int = 0
        self.ram_delta_mb: int = 0

    def note(self, only_active: bool = True) -> str:
        """Compact line for the log, e.g. 'GPU0 6.12 GB (Δ 5.80 GB)'.

        only_active: leave out GPUs without a notable delta change (<64 MB) —
        so the line shows only the GPU the model actually ran on.
        """
        static = gpu_static()
        bits = []
        for idx in sorted(self.peak_mb):
            d = self.delta_mb.get(idx, 0)
            if only_active and d < 64 and len(self.peak_mb) > 1:
                continue
            total = static.get(idx, {}).get("total_mb", 0)
            of_total = f"/{fmt_mb(total)}" if total else ""
            bits.append(f"GPU{idx} {fmt_mb(self.peak_mb[idx])}{of_total} (Δ {fmt_mb(d)})")
        # Memory belongs in the SAME line as VRAM. Logging them
        # separately would mean adding up two places when searching —
        # and that's exactly what didn't happen when a night run ran out of memory.
        if self.ram_peak_mb:
            bits.append(f"RAM {fmt_mb(self.ram_peak_mb)} (Δ {fmt_mb(self.ram_delta_mb)})")
        return " · ".join(bits)


@contextmanager
def vram(label: str, poll: float = 0.5):
    """Measure GPU VRAM peak during a block (only active in debug mode).

        with debug.vram("WhisperX turbo") as vr:
            model = whisperx.load_model(...)
        log(vr.note())

    A background thread polls the memory used per GPU and remembers the
    maximum. Also captures ctranslate2 and subprocess models (audio-separator),
    which torch.cuda.*_allocated doesn't see. Outside debug mode a pure no-op
    with no overhead; without a GPU only the memory is measured — on such a
    machine it is the number that matters.
    """
    result = _VramResult()
    if not _enabled:
        yield result
        return
    gpu = gpu_available()

    def _gpu_used() -> dict[int, int]:
        return gpu_mem_used_mb() if gpu else {}

    baseline = _gpu_used()
    peaks: dict[int, int] = dict(baseline)
    # Measure memory in the SAME cadence. It doesn't deserve its own
    # sampler: whoever wants to know what a step costs wants both numbers
    # side by side — and from the same time window.
    ram_basis = ram_used_mb()
    ram_peak = ram_basis
    stop = threading.Event()

    def _sample():
        nonlocal ram_peak
        while not stop.is_set():
            for idx, mb in _gpu_used().items():
                if mb > peaks.get(idx, 0):
                    peaks[idx] = mb
            # Sample memory more often than the GPU: a
            # /proc read costs nothing, `nvidia-smi` does. The cadence
            # determines how brief a spike may be to still get
            # caught — it reliably captures steps that last seconds to
            # minutes (diarization!), not a millisecond flash. That's
            # deliberate: that would need tracemalloc, and that costs
            # more than it's worth here.
            for _ in range(RAM_TICKS_PER_VRAM_TICK):
                r = ram_used_mb()
                if r > ram_peak:
                    ram_peak = r
                if stop.wait(poll / RAM_TICKS_PER_VRAM_TICK):
                    return

    t = threading.Thread(target=_sample, daemon=True, name="vram-sampler")
    t.start()
    try:
        yield result
    finally:
        stop.set()
        t.join(timeout=2)
        # Final sample — short blocks may end before the first poll tick.
        for idx, mb in _gpu_used().items():
            if mb > peaks.get(idx, 0):
                peaks[idx] = mb
        r = ram_used_mb()
        ram_peak = max(ram_peak, r)
        result.ram_peak_mb = ram_peak
        result.ram_delta_mb = max(0, ram_peak - ram_basis)
        _record_ram(label, ram_peak, result.ram_delta_mb)
        for idx, peak in peaks.items():
            delta = max(0, peak - baseline.get(idx, 0))
            result.peak_mb[idx] = peak
            result.delta_mb[idx] = delta
            _record_vram(label, idx, peak, delta)
        note = result.note()
        if note:
            logger.debug("📊 VRAM %s · %s", label, note)


def vram_stats() -> list[dict]:
    """Aggregated VRAM peaks per (model, GPU), sorted by peak."""
    static = gpu_static()
    rows = []
    with _vram_lock:
        for label, gpus in _vram.items():
            for idx, d in gpus.items():
                info = static.get(idx, {})
                rows.append(
                    {
                        "label": label,
                        "gpu": idx,
                        "gpu_name": info.get("name", f"GPU{idx}"),
                        "peak_mb": d["peak_mb"],
                        "peak": fmt_mb(d["peak_mb"]),
                        "delta_mb": d["delta_mb"],
                        "delta": fmt_mb(d["delta_mb"]),
                        "total_mb": info.get("total_mb", 0),
                        "ram_peak_mb": _ram.get(label, {}).get("peak_mb", 0),
                        "ram_delta_mb": _ram.get(label, {}).get("delta_mb", 0),
                    }
                )
    rows.sort(key=lambda r: r["peak_mb"], reverse=True)
    return rows


def ram_stats() -> list[dict]:
    """Memory peaks per step, largest first.

    Separate from `vram_stats()`, because there are steps that need RAM and
    never touch a GPU — they wouldn't show up in a GPU-indexed table at
    all. Exactly one of those once brought down the machine in a night run.
    """
    with _vram_lock:
        rows = [
            {
                "label": k,
                "peak_mb": v["peak_mb"],
                "peak": fmt_mb(v["peak_mb"]),
                "delta_mb": v["delta_mb"],
                "delta": fmt_mb(v["delta_mb"]),
            }
            for k, v in _ram.items()
        ]
    rows.sort(key=lambda r: r["peak_mb"], reverse=True)
    return rows


def vram_summary() -> str:
    """Memory tables: what eats how much — on the GPU AND in RAM.

    The early exit "no GPU rows → empty output" was a bug waiting to
    happen: a step that needs only memory has no line in the
    GPU table — and exactly one of those once brought down the machine.
    It would have stayed invisible even though its measured
    number was off.
    """
    rows = vram_stats()
    out: list[str] = []
    if rows:
        w = min(max([len(r["label"]) for r in rows] + [len("GPU VRAM per model (peak)")]), 46)
        out += [
            "",
            "═" * (w + 44),
            f"{'GPU VRAM per model (peak)':<{w}} {'GPU':>5} {'Peak':>11} "
            f"{'Δ (model)':>12} {'/ Total':>10}",
            "─" * (w + 44),
        ]
        for r in rows:
            label = r["label"][:w]
            total = fmt_mb(r["total_mb"]) if r["total_mb"] else "?"
            out.append(f"{label:<{w}} {r['gpu']:>5} {r['peak']:>11} {r['delta']:>12} {total:>10}")
        out.append("═" * (w + 44))

    # Memory as ITS OWN table below. Squeezing it into the same one
    # wouldn't work: a step without a GPU share has no line there, and
    # that was exactly what once brought the machine down.
    rram = ram_stats()
    if rram:
        # At least as wide as the heading — otherwise the header
        # sits above rather than over the numbers.
        rw = min(max([len(r["label"]) for r in rram] + [len("Memory per step (peak)")]), 46)
        total = _ram_total_mb()
        out += [
            "",
            "═" * (rw + 30),
            f"{'Memory per step (peak)':<{rw}} {'Peak':>11} {'Δ (step)':>13}",
            "─" * (rw + 30),
        ]
        for r in rram:
            out.append(f"{r['label'][:rw]:<{rw}} {r['peak']:>11} {r['delta']:>13}")
        if total:
            peak = max((r["peak_mb"] for r in rram), default=0)
            out.append("─" * (rw + 30))
            out.append(
                f"{'Machine total':<{rw}} {fmt_mb(total):>11} "
                f"{f'Peak = {peak * 100 // total} %':>13}"
            )
        out.append("═" * (rw + 30))
    return "\n".join(out)


def _ram_total_mb() -> int:
    """Total memory of the machine — the reference value.

    Without it, "25 GB" is just a number; with it, it's "82 % of the
    machine" and thus a statement.
    """
    try:
        with open("/proc/meminfo") as f:
            for z in f:
                if z.startswith("MemTotal:"):
                    return int(z.split()[1]) // 1024
    except OSError:
        pass
    return 0


# ── Sections in the log ─────────────────────────────────────────────────────
# Without structure, a pipeline log is a flat wasteland of lines where you
# can't find the start of a phase. Each phase therefore gets a header
# ("what's happening now, with which file") and a closing line with duration.

_RULE = "━" * 78
_pipe_log = logging.getLogger("whisper_tm.pipeline")


def short_name(name: str, width: int = 58) -> str:
    """Shorten a long path to display width — what matters is at the end."""
    name = str(name)
    if len(name) <= width:
        return name
    return "…" + name[-(width - 1) :]


def banner(title: str, subject: str = "", **fields) -> None:
    """Top level: a new file from the queue.

    Deliberately framed differently from section(), so you can immediately
    see while scrolling where a new FILE starts versus where only a phase changes.
    """
    _pipe_log.info("")
    _pipe_log.info("╔%s╗", "═" * (len(_RULE) - 2))
    _pipe_log.info("║ %s", title)
    if subject:
        _pipe_log.info("║ %s", short_name(subject, width=len(_RULE) - 4))
    for key, val in fields.items():
        if val not in (None, "", False):
            _pipe_log.info("║   %s: %s", key, val)
    _pipe_log.info("╚%s╝", "═" * (len(_RULE) - 2))


def section(
    title: str, subject: str = "", step: int | None = None, total: int | None = None, **fields
) -> None:
    """Header line of a new phase: "Now doing X with file Y".

    Only the FILE NAME goes here, not the full path — that's already in the
    banner() above. Otherwise, for deeply nested
    recording folders, the same 200-character path would repeat in every phase line.
    """
    num = f"[{step}/{total}] " if step and total else ""
    head = f"▶ {num}{title}"
    if subject:
        head += f"  ·  {short_name(Path(subject).name, width=48)}"

    _pipe_log.info("%s", _RULE)
    _pipe_log.info("%s", head)
    for key, val in fields.items():
        if val not in (None, "", False):
            _pipe_log.info("     %s: %s", key, val)


def section_done(title: str, secs: float, ok: bool = True, note: str = "") -> None:
    """Closing line of a phase — with duration, so outliers stand out immediately."""
    mark = "✔" if ok else "✖"
    line = f"{mark} {title} · {fmt_secs(secs)}"
    if note:
        line += f" · {note}"
    _pipe_log.info("%s", line)


# ── Evaluation ────────────────────────────────────────────────────────────


def timings(top: int = 0) -> list[dict]:
    """Aggregated measurements, sorted by total time."""
    with _lock:
        rows = [
            {
                "label": label,
                "count": len(vals),
                "total_sec": round(sum(vals), 3),
                "avg_sec": round(sum(vals) / len(vals), 3),
                "max_sec": round(max(vals), 3),
            }
            for label, vals in _timings.items()
            if vals
        ]
    rows.sort(key=lambda r: r["total_sec"], reverse=True)
    return rows[:top] if top else rows


def summary(top: int = 30) -> str:
    """Human-readable table: where did the time go?"""
    rows = timings(top)
    if not rows:
        return "No measurements."

    w = min(max((len(r["label"]) for r in rows), default=10), 60)
    out = [
        "",
        "═" * (w + 40),
        f"{'Where the time went':<{w}} {'Count':>7} {'Total':>10} {'Avg':>9} {'Max':>9}",
        "─" * (w + 40),
    ]
    for r in rows:
        label = r["label"][:w]
        out.append(
            f"{label:<{w}} {r['count']:>7} "
            f"{fmt_secs(r['total_sec']):>10} "
            f"{fmt_secs(r['avg_sec']):>9} {fmt_secs(r['max_sec']):>9}"
        )
    out.append("═" * (w + 40))
    return "\n".join(out)


def dump_summary() -> None:
    """Write the overview to the log (on shutdown)."""
    log = logging.getLogger("whisper_tm")
    if _timings:
        log.info("%s", summary())
    vram_tbl = vram_summary()
    if vram_tbl:
        log.info("%s", vram_tbl)


def reset() -> None:
    with _lock:
        _timings.clear()
    with _vram_lock:
        _vram.clear()

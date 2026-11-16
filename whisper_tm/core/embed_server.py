"""On-demand embedding server: starts when first needed, shuts down after idle time.

**Why a separate process at all.** The chat LLM (``core/llama_server.py``)
and embedding model are two different models with contradictory operating
parameters — one generates token by token, the other pushes the largest
batches possible. Forcing both into one process would mean taking the settings
of one for the other.

**Why on-demand.** The model occupies VRAM, specifically the GPU card that
Whisper and pyannote also use. A permanently running embedding server would
take several gigabytes away from batch processing — for a model that does
nothing most of the time. So: boot it when first embedding is needed, then
shut it down after a period of idle time.

The existing ``llama_server.py`` **only builds** the startup command and checks
the state; it was started manually via ``./start-llm.sh``. For the chat LLM
that's fine — the user decides when they want it. For embeddings it's not:
they run automatically during indexing, and no one should have to start a
script beforehand.

**What can go wrong while waiting, and how it's handled here.**

``/health`` only responds when the model is loaded — with 4 GB on the
GPU that takes seconds to minutes. If you ask just once and give up, you see
an error even though everything is fine. So it polls, and specifically with
an abort that **takes the child process with it**: a hanging llama-server
without a parent keeps occupying VRAM, and you only find it when the
next transcription fails due to insufficient memory.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path

from whisper_tm.core import debug
from whisper_tm.core.procgroup import signal_group

logger = logging.getLogger(__name__)

#: How often the guard checks whether idle time has been reached.
WATCH_TICK_SEC = 15.0


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("embedding", name, default=default)


def base_directory() -> Path:
    return Path(__file__).resolve().parents[2]


def binary() -> Path | None:
    """``llama-server`` — same search as for chat LLM."""
    from whisper_tm.core import llama_server

    return llama_server.binary()


def model_file() -> Path | None:
    """The configured GGUF file for the embedding model.

    Unlike with the chat LLM, we do NOT fall back to "the first file in models/gguf".
    That usually contains the chat model, and starting it as an embedding model
    produces vectors that *look* like embeddings — just without the training
    purpose behind them. Such an error only shows up in poor search results,
    if at all.
    """
    set_flag = (_cfg("local_model") or "").strip()
    if not set_flag:
        return None
    p = Path(set_flag)
    if not p.is_absolute():
        p = base_directory() / p
    return p if p.exists() else None


def _host() -> str:
    h = str(_cfg("local_host", "127.0.0.1") or "127.0.0.1")
    return "127.0.0.1" if h in ("0.0.0.0", "::") else h


def _port() -> int:
    return int(_cfg("local_port", 8081) or 8081)


def endpoint() -> str:
    """OpenAI-compatible base URL.

    ``/v1`` is intentional and not a detail: over the native Ollama path,
    EVERY text goes out as its own request, over the OpenAI path a whole
    batch at once. Measured with qwen3-embedding on real chunks:
    4.0 vs 73.0 chunks/s — factor 18 for the same calculation.
    """
    return f"http://{_host()}:{_port()}/v1"


def _free_vram_mb() -> int:
    """Free VRAM of the first GPU in MB, ``-1`` if not available.

    Via ``nvidia-smi`` rather than torch: this function runs before
    starting a foreign process, and building a CUDA context just for that
    would itself occupy memory — so avoid doing exactly what we're trying
    to prevent.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return -1
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        # No GPU, no nvidia-smi, unexpected output: then this function
        # makes no statement, and the caller should not assume anything.
        return -1


def _environment(on_gpu: bool) -> dict:
    """Environment of the child process.

    In CPU mode, the GPU card is HIDDEN, not just left unused.
    Measured: with `--n-gpu-layers 0` llama-server still occupied
    1,878 MiB of VRAM — the GPU-capable binary creates a CUDA context on
    startup whether it needs it or not. On a card where transcription fights
    for every hundred megabytes, that's not a remainder you can ignore.

    `CUDA_VISIBLE_DEVICES=""` means to CUDA: there is no card. Then
    no context is created either.
    """
    env = dict(os.environ)
    if not on_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["GGML_CUDA_NO_PINNED"] = "1"
    return env


#: llama.cpp's spellings of the layer count and the device list.
_LAYER_FLAGS = ("-ngl", "--gpu-layers", "--n-gpu-layers")
_DEVICE_FLAGS = ("-dev", "--device")


def _last_value(argv: list[str], flags: tuple[str, ...]) -> str | None:
    """The value of the LAST of ``flags`` in ``argv`` (``--x 5`` or
    ``--x=5``) — llama.cpp takes the last one, too."""
    found = None
    for i, a in enumerate(argv):
        name, eq, value = a.partition("=")
        if name not in flags:
            continue
        if eq:
            found = value
        elif i + 1 < len(argv):
            found = argv[i + 1]
    return found


def _runs_on_gpu(argv: list[str]) -> bool:
    """Does this command put any layer on the card?

    Read from the whole command, the extra arguments included. Previously
    only the first ``--n-gpu-layers`` counted: a CPU-only server
    set up through ``local_extra_args`` (``-ngl 0``, ``--device none``)
    counted as a GPU one and started with the card visible — and the
    GPU-capable binary then takes ≈1.9 GB of CUDA buffers it never uses
    (see ``_environment``).
    """
    if (_last_value(argv, _DEVICE_FLAGS) or "").strip().lower() == "none":
        return False
    layers = _last_value(argv, _LAYER_FLAGS)
    return layers is None or layers.strip() != "0"


def _job_running() -> bool:
    """Is the queue currently processing a file?

    Not "is there work" (that would be `running` in the queue snapshot, which
    also counts items waiting), but "is someone currently holding the GPU".

    The import is inside the function: `queue` pulls half the application
    with it, and this module is loaded at startup.
    """
    try:
        from whisper_tm.core import queue as _q

        return _q.busy()
    except Exception:  # noqa — don't block if uncertain
        return False


def _search_on_cpu() -> bool:
    """Should SEARCH run the server on the CPU by default?

    Default yes, and that's measured. With real chunks from
    ``rag.db`` (median 415 tokens) on a test machine:

    ============================ ============== ==============
    ..                           CPU (16 cores)  GPU (12 GB)
    ============================ ============== ==============
    Embed query (search)         **214 ms**     —
    Embed chunks (index)         0.2 chunks/s   ~18 chunks/s
    Corpus of 100k chunks        ~6 days        ~1.5 h
    ============================ ============== ==============

    So for search, the CPU costs about a fifth of a second — measured
    WHILE the queue was transcribing. For embeddings it's a factor of 80
    too slow, so it stays on the GPU.

    The reason for the difference is VOLUME, not the device: the CPU
    handles a constant 67 tokens/s, a chunk has ~415 tokens, a query ~30 —
    and of those ~22 are always the same instruction prefix that llama.cpp
    caches as prompt-prefix. What's new to compute per search is ~8 tokens.

    The benefit is not speed but that search no longer touches the GPU at all:
    a RAG search on the GPU can take several GB while a job is running
    and push that job into CUDA-OOM.
    """
    return bool(_cfg("local_search_on_cpu", True))


def _gpu_is_free() -> bool:
    """Should an index run be allowed the GPU RIGHT NOW?

    Same two conditions as in the startup command — no running job and
    enough free memory above the reserve. Intentionally a separate function
    rather than "build the startup command and check": the startup command has
    side effects in the log and reads half the configuration, while here we're
    just asking a question.

    An explicitly set ``local_gpu_layers`` means the user has decided themselves —
    then nothing is moved. ``0`` explicitly means "always CPU".
    """
    levels = int(_cfg("local_gpu_layers", -1))
    if levels >= 0:
        return False
    if _job_running():
        return False
    reserve = int(_cfg("local_vram_reserve_mb", 3000) or 0)
    if reserve <= 0:
        return True
    free = _free_vram_mb()
    # -1 means "not available" — then nothing is assumed and nothing
    # is moved. Moving into uncertainty can cost a transcription.
    return free >= reserve


def _start_command(force_cpu: bool = False) -> list[str]:
    """The llama-server invocation for embedding mode.

    ``force_cpu`` comes from search (see :func:`_search_on_cpu`) and
    acts like the VRAM guard, just without the condition.
    """
    bin_, model = binary(), model_file()
    if not bin_ or not model:
        return []
    argv = [
        str(bin_),
        "--model",
        str(model),
        "--host",
        _host(),
        "--port",
        str(_port()),
        # Without this, /v1/embeddings returns 501.
        "--embedding",
        # Ubatch and context must be at least as large as the longest input:
        # these models are non-causal and need the entire sequence in ONE pass.
        # If it doesn't fit, llama.cpp rejects the request with
        # "Context size has been exceeded" instead of chunking it — and in the
        # middle of the index run.
        #
        # 2048 is measured, not guessed: 4096 and 8192 are exactly as fast
        # (12.0 vs 12.2 chunks/s) and cost 7.4 and 10.5 GB VRAM instead of 5.9.
        # 1024 looked faster in initial testing — but failed in the cross-check
        # on a sample with longer chunks. Exactly such an error only shows up
        # hours into a run.
        "--batch-size",
        str(_cfg("local_batch", 2048) or 2048),
        "--ubatch-size",
        str(_cfg("local_batch", 2048) or 2048),
        "--ctx-size",
        str(_cfg("local_ctx", 2048) or 2048),
    ]
    # No prompt cache: an embedding never comes back to a prompt, and the
    # cache copied every passage's state off the card (llama_server)
    from whisper_tm.core import llama_server

    argv += llama_server.no_prompt_cache_args(bin_)
    # Reserve for transcription — the most important value in this file.
    #
    # "llama.cpp adapts to free memory" means on a SHARED card: it takes
    # everything that's currently free. That can cost a transcription:
    # the embedding server squeezes into the gap and uses exactly the buffer
    # that the separator needs right after, and the file ends up as an error
    # in the queue.
    #
    # So we check FIRST: if after subtracting the reserve there's not enough
    # left, the server runs on the CPU. Slower, but it doesn't take anything
    # from anyone — and that's the right priority, because transcription is
    # the work, embedding is just its postlude.
    reserve = int(_cfg("local_vram_reserve_mb", 3000) or 0)
    on_cpu = False
    if force_cpu or int(_cfg("local_gpu_layers", -1)) < 0:
        free = _free_vram_mb()
        running = _job_running()
        # A RUNNING job beats any snapshot of free memory.
        #
        # Otherwise: diarization finishes a second before and frees its
        # memory, the card looks empty — the embedding server takes it, and
        # Whisper finds almost nothing free in the next step; the file goes
        # back in the queue with VRAM exhausted.
        #
        # The reserve didn't help because it measures free memory EXACTLY when
        # it's largest: between two steps. Whoever's using the card now will
        # need it again right away.
        if force_cpu or running or (reserve > 0 and 0 <= free < reserve):
            logger.info(
                "Embedding server running on CPU: %s",
                "intended for search"
                if force_cpu
                else "queue is transcribing right now"
                if running
                else f"only {free} MB VRAM free, reserve {reserve} MB",
            )
            argv += ["--n-gpu-layers", "0"]
            on_cpu = True

    # Layers on the GPU. -1 means: do NOT set.
    #
    # We first had `--n-gpu-layers 99` hard-coded, and that was wrong — on a
    # shared card, fatal. llama.cpp adjusts the number of layers itself to
    # FREE memory, but only as long as no one sets the value. Observed
    # with a running batch:
    #
    #   W common_fit_params: failed to fit params to free device memory:
    #                        n_gpu_layers already set by user to 99, abort
    #   E cudaMalloc failed: out of memory (4075 MiB)
    #
    # The server died exactly when the card was needed — and on a shared
    # card that's the normal case, not the exception. Without the setting,
    # llama.cpp loads as much onto the GPU as fits, and the remainder onto the CPU.
    levels = int(_cfg("local_gpu_layers", -1))
    if not on_cpu and levels >= 0:  # the CPU path above has set 0
        argv += ["--n-gpu-layers", str(levels)]

    # Pooling and extra arguments go on EVERY path, the CPU one included.
    # Pooling decides how the token vectors of a text become its one vector,
    # and the device changes between starts: a search starts the server on
    # the CPU, an index run on the GPU (or on the CPU while the card is
    # busy). Previously the CPU path returned before this point, so with
    # a pooling set, questions were pooled the model's way and most stored
    # chunks the configured way — each search compared two different kinds
    # of vector, and nothing but worse hits showed it.
    pooling = (_cfg("local_pooling", "") or "").strip()
    if pooling:
        # Leaving empty means "use what's in the model" — correct for
        # Qwen3-Embedding (last-token pooling) and the safer assumption
        # for other models than a guessed value.
        argv += ["--pooling", pooling]
    from whisper_tm.core import local_key

    argv += local_key.launch_args(Path(argv[0]))
    extra = (_cfg("local_extra_args") or "").strip()
    if extra:
        from whisper_tm.core import commands

        argv += commands.split_args(extra)
    return argv


class EmbedServer:
    """A llama-server for embeddings that shuts itself down again."""

    def __init__(self):
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._last_used = 0.0
        self._watchdog: threading.Thread | None = None
        self._started_at = 0.0
        self._log: Path | None = None
        # Was this process started WITH GPU? Decides whether it must yield
        # when the queue starts up.
        self._on_gpu = False
        # (host, port) the process was started on — the configured address
        # can change under a running server (see `ensure`).
        self._address: tuple[str, int] | None = None
        # One provisioning at a time, the wait for /health included. The
        # state lock above is released while a start waits; a second caller
        # in that window found "not answering yet, process alive" and
        # started ANOTHER server, overwriting `_proc` — the first then ran
        # on unwatched, VRAM and all. Now it waits here and
        # finds the first one ready.
        self._start_lock = threading.Lock()

    # ── State ────────────────────────────────────────────────────────────
    def _responds(self, timeout: float = 2.0) -> bool:
        import requests

        try:
            r = requests.get(f"http://{_host()}:{_port()}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def _own_process_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def pid(self) -> int | None:
        """The running llama-server's PID — the performance view names it
        in the VRAM breakdown (core/vram_procs.py)."""
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def state(self) -> dict:
        with self._lock:
            empty = 0.0
            if self._last_used:
                empty = round(time.time() - self._last_used, 1)
            return {
                "managed": self._own_process_alive(),
                "reachable": self._responds(),
                "endpoint": endpoint(),
                "model": str(model_file() or ""),
                "binary": str(binary() or ""),
                "pid": self._proc.pid if self._own_process_alive() else 0,
                "running_for_sec": (
                    round(time.time() - self._started_at, 1) if self._started_at else 0.0
                ),
                "idle_sec": empty,
                "idle_limit_sec": int(_cfg("local_idle_stop_sec", 300) or 0),
            }

    # ── Startup ──────────────────────────────────────────────────────────
    def ensure(self, *, wait_time: float | None = None, purpose: str = "index") -> str:
        """Provision the server and return the endpoint (see :meth:`_ensure`).

        Serialized: a caller arriving while another one starts the server
        waits for that start instead of launching a second process.
        """
        with self._start_lock:
            return self._ensure(wait_time=wait_time, purpose=purpose)

    def _ensure(self, *, wait_time: float | None = None, purpose: str = "index") -> str:
        """Provision the server and return the endpoint.

        If something is already running on the port (e.g., started manually),
        use it and do NOT touch it — a second process on the same port would
        fail anyway, and killing someone else's would be overreaching.

        ``purpose='search'`` starts it on the CPU (see :func:`_search_on_cpu`).
        This only applies at STARTUP: if one is already running — say on the
        GPU for indexing — search uses it as-is instead of moving it. Moving
        in the middle of an index run would cost more than the 200 ms it saves.

        **But the opposite also happens**, and it's the more expensive case:
        if an index run finds a CPU server left behind by search, it would
        compute at 0.2 instead of 18 chunks/s — factor 90. If the card is free,
        it moves it there.
        """
        with self._lock:
            self._last_used = time.time()
            # Our own server, but not at the configured address: port or host
            # were changed in the settings while it ran. Everything below
            # only looks at the configured address; finding nothing there,
            # it used to start a second server and overwrite `_proc`. The
            # old one then ran on unwatched — the idle watchdog and `stop_it`
            # only know `_proc`, and its own session keeps it alive past the
            # app's exit, VRAM and all. So it goes first.
            if (
                self._address is not None
                and self._address != (_host(), _port())
                and self._own_process_alive()
            ):
                logger.info(
                    "Embedding server address changed (%s:%s → %s:%s) — stopping the old one",
                    *self._address,
                    _host(),
                    _port(),
                )
                self.stop_it()  # _lock is an RLock
            if self._responds():
                # Is it already running — but ON THE GPU, while the queue is
                # transcribing? Then it must yield.
                #
                # The guard in `_start_command` only applies at startup. If you
                # used the server before (e.g., for a RAG search) and then fire
                # up the queue, you'd otherwise have 10 GB of the card
                # permanently locked, without anyone touching it again.
                # Moving costs a few seconds for the next request; a cancelled
                # job costs minutes.
                if self._on_gpu and self._own_process_alive() and _job_running():
                    logger.info("Embedding server moving to CPU: queue needs the GPU")
                    self.stop_it()  # _lock is an RLock
                elif (
                    not self._on_gpu
                    and self._own_process_alive()
                    and purpose != "search"
                    and _gpu_is_free()
                ):
                    logger.info("Embedding server moving to GPU: index is 90x slower on CPU")
                    self.stop_it()
                else:
                    self._start_watchdog()
                    return endpoint()

            if self._proc is not None and not self._own_process_alive():
                self._cleanup()

            argv = _start_command(force_cpu=(purpose == "search" and _search_on_cpu()))
            if not argv:
                is_missing = (
                    "llama-server not found"
                    if not binary()
                    else "Embedding model not set or not available (embedding.local_model)"
                )
                raise RuntimeError(f"Embedding server cannot start: {is_missing}")

            folder = debug.logs_root()
            folder.mkdir(parents=True, exist_ok=True)
            self._log = folder / "embed-server.log"
            from whisper_tm.core import llama_server

            llama_server.trim_log(self._log)
            # Remember HOW it was started: no layer on the card means CPU —
            # and a CPU server starts with the card hidden (`_environment`).
            # Not set = llama.cpp takes the GPU.
            self._on_gpu = _runs_on_gpu(argv)
            logger.info(
                "Embedding server starting (%s): %s",
                "GPU" if self._on_gpu else "CPU",
                " ".join(argv[:6]) + " …",
            )
            with open(self._log, "ab") as fh:
                fh.write(f"\n=== Start {time.strftime('%F %T')} ===\n".encode())
                self._proc = subprocess.Popen(
                    argv,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    # Own process group: when stopping, the WHOLE group goes.
                    # Without this, a helper spawned by llama.cpp survives its
                    # parent and holds VRAM.
                    start_new_session=True,
                    env=_environment(self._on_gpu),
                )
            # Written down: should this app end without stopping it, the
            # next start ends it (core/helper_pids)
            from whisper_tm.core import helper_pids

            helper_pids.record("embed-server", self._proc.pid, _port())
            self._address = (_host(), _port())
            self._started_at = time.time()

        limit = float(
            wait_time if wait_time is not None else _cfg("local_start_timeout_sec", 180) or 180
        )
        end = time.time() + limit
        while time.time() < end:
            if self._responds(timeout=2.0):
                logger.info("Embedding server ready after %.1fs", time.time() - self._started_at)
                with self._lock:
                    self._last_used = time.time()
                self._start_watchdog()
                return endpoint()
            if not self._own_process_alive():
                tail = self._log_tail()
                self.stop_it()
                raise RuntimeError(f"Embedding server terminated during startup. {tail}")
            time.sleep(1.0)

        # Timeout: the process MUST go. A hanging llama-server keeps
        # occupying VRAM and is otherwise only noticed when the next
        # transcription fails because of it.
        tail = self._log_tail()
        self.stop_it()
        raise RuntimeError(f"Embedding server was not ready after {limit:.0f}s. {tail}")

    def _log_tail(self, lines: int = 3) -> str:
        from whisper_tm.core import llama_server

        return llama_server.log_tail(self._log, lines)

    # ── Shutdown ─────────────────────────────────────────────────────────
    def _cleanup(self):
        if self._proc is not None:
            from whisper_tm.core import helper_pids

            helper_pids.forget("embed-server")
        self._proc = None
        self._address = None
        self._started_at = 0.0

    def stop_it(self) -> bool:
        """Stop the self-started server. External ones are left untouched."""
        with self._lock:
            p = self._proc
            if p is None:
                return False
            if p.poll() is None:
                logger.info("Embedding server terminating (pid %s)", p.pid)
                if not signal_group(p.pid):
                    p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logger.warning("Embedding server not responding to SIGTERM — using SIGKILL")
                    if not signal_group(p.pid, kill=True):
                        p.kill()
                    with suppress(subprocess.TimeoutExpired):
                        p.wait(timeout=5)
            self._cleanup()
            return True

    # ── Idle Watcher ─────────────────────────────────────────────────────
    def used(self):
        """Remember that embedding just happened (extends runtime)."""
        with self._lock:
            self._last_used = time.time()

    def _start_watchdog(self):
        with self._lock:
            if self._watchdog and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(
                target=self._watch, name="embed-server-watcher", daemon=True
            )
            self._watchdog.start()

    def _watch(self):
        while True:
            time.sleep(WATCH_TICK_SEC)
            limit = float(_cfg("local_idle_stop_sec", 300) or 0)
            with self._lock:
                if not self._own_process_alive():
                    return
                if limit <= 0:  # 0 = never automatically stop
                    continue
                quiet = time.time() - self._last_used
                if quiet < limit:
                    continue
            logger.info("Embedding server unused for %.0fs → terminating (freeing VRAM)", quiet)
            self.stop_it()
            return


#: Exactly one per process — two would try to start on the same port.
SERVER = EmbedServer()


def available() -> tuple[bool, str]:
    """Can the local embedding server be operated at all?"""
    if not binary():
        from whisper_tm.core import commands

        return False, f"llama-server not found — {commands.llama_missing()}"
    if not model_file():
        set_flag = (_cfg("local_model") or "").strip()
        return False, (
            f"Model file not found: {set_flag}"
            if set_flag
            else "No embedding model set (embedding.local_model)"
        )
    return True, ""


def on_shutdown():
    """Clean up when this process exits."""
    try:
        SERVER.stop_it()
    except Exception as e:  # noqa
        logger.debug("Embedding server cleanup: %s", e)


# Register HERE, not in app.py.
#
# It used to be only in `create_app()`. That covered the app — but not
# scripts: a tool using `make_embed_fn()` started the server, ran, and left
# it running. The llama-server afterward
# had `PPid 1` and kept occupying VRAM even though its caller was long gone.
# The idle watcher didn't help: it's a daemon thread and dies with its process.
#
# Whoever starts the server must stop it — and that's everyone who imports
# this module.
atexit.register(on_shutdown)

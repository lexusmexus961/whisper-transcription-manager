"""On-demand reranking server: llama.cpp with Qwen3-Reranker, on the GPU only.

**Why a reranker.** A vector search compares two vectors made apart from each
other; a reranker (a cross-encoder) reads the query and the passage together
and judges the pair. Measured on a real index (a few hundred thousand chunks)
against careful gold labels, P@10 / P@50:

    =========================================  =========  =========
    query                                      Q1         Q2
    =========================================  =========  =========
    vector only (the meaning search until now) 0.40/0.56  0.40/0.18
    Qwen3-Reranker-0.6B                        ≤0.60/0.64 ≤0.20/0.16
    Qwen3-Reranker-4B, own instruction         1.00/0.82  0.60/0.30
    Qwen3-Reranker-4B, instruction + criteria  1.00/0.90  0.60/0.36
    =========================================  =========  =========

The 4B is a very good ORDER, not a junk filter (precision ≈0.4–0.5 at 0.5);
the deep search (rag/deep_search.py) adds a language model for that.

**GPU only.** On a 12 GB card it reads ≈10 passages a second (4 slots,
context 8192) in ≈4.5 GB of VRAM; on the CPU 3–4 a minute — useless. Without
a card the reranker is not available, and the callers keep their own order.

**The GPU gate.** The server holds the stage ``rerank`` of
``core/gpu_gate.GATE`` for as long as it runs — from a thread of its own,
because a stage belongs to the thread that took it. So Whisper (which runs
alone) never starts next to it, and the queue's other stages only start when
NVML shows the memory they need. It gives the card back: after
``rerank.idle_stop_sec`` without a call, after a few seconds without a call
while someone waits at the gate, and — in a long run — between two batches
once it has held the card for :data:`YIELD_AFTER_SEC` and someone waits.
Getting the card is bounded: a quick search waits ~5 s, a deep search
minutes; past that :class:`RerankUnavailable`.

**The instruction.** The model's GGUF bakes *"Given a web search query,
retrieve relevant passages that answer the query"* into its rerank template,
and llama.cpp only fills in ``{query}`` and ``{document}``. A copy of the file
whose template line reads ``<Instruct>: {query}`` (core/gguf.py rewrites the
header, the weights are copied as they are) lets each request bring its own
instruction: the query is sent as ``"<instruction>\\n<Query>: <text>"``. The
copy is made once and made again when the source file changes (size and
mtime in a sidecar). A model whose template lacks exactly that line is served
as it is, with the plain query.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from whisper_tm.core import commands, debug, gguf
from whisper_tm.core.procgroup import signal_group

logger = logging.getLogger(__name__)

#: The gate stage the server holds while it runs.
STAGE = "rerank"
#: The GGUF key holding the rerank prompt template.
TEMPLATE_KEY = "tokenizer.chat_template.rerank"
#: The instruction line of Qwen3-Reranker's template — exactly one wanted.
_INSTRUCT_LINE = re.compile(r"<Instruct>: [^\n{}]*\n<Query>: \{query\}")
DERIVED_LINE = "<Instruct>: {query}"

#: For a quick search: the situation must happen, a name is not enough.
GENERIC_INSTRUCTION = (
    "Judge whether this excerpt of a spoken conversation transcript shows the "
    "situation described in the query actually happening (merely mentioning the "
    "name or the topic is not enough)."
)
#: For a deep search: followed by that search's criteria.
CRITERIA_INSTRUCTION = (
    "Judge whether this excerpt of a spoken conversation transcript shows the "
    "situation described in the query actually happening. "
)

DEFAULTS = {
    "enabled": True,
    "local_model": "models/gguf/Qwen3-Reranker-4B-Q4_K_M.gguf",
    "host": "127.0.0.1",
    "port": 8082,
    "ctx": 8192,
    "parallel": 4,
    "batch": 2048,
    "idle_stop_sec": 300,
    "vram_need_mb": 4800,
    "start_timeout_sec": 120,
    "extra_args": "",
    "instruction": GENERIC_INSTRUCTION,
    "quick_candidates": 120,
    "quick_min_score": 0.05,
    "deep_dense": 2000,
    "deep_prefilter": 0.05,
    "deep_max_verify": 1500,
}

#: A passage is cut to this many characters before it is judged.
DOC_CHARS = 2000
#: Passages per request.
BATCH = 32
#: How long a quick search waits for the card; a deep search waits longer.
QUICK_WAIT_SEC = 5.0
DEEP_WAIT_SEC = 900.0
#: How often the watcher looks at idle time and the gate.
WATCH_TICK_SEC = 2.0
#: Idle this long while someone waits at the gate → the card goes back.
YIELD_GRACE_SEC = 2.0
#: A long run gives the card back between batches after holding it this
#: long, when someone waits at the gate.
YIELD_AFTER_SEC = 60.0
#: One request (a batch of 32 passages takes ~3 s on the card).
CALL_TIMEOUT_SEC = 300


class RerankUnavailable(RuntimeError):  # noqa: N818 - a state, like RagUnavailable
    """The reranker can't serve right now — the caller keeps its own order.

    ``reason``: ``off`` (switched off), ``missing`` (no llama-server or model
    file), ``no_gpu``, ``busy`` (the card is in use) or ``failed``.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail

    def note(self) -> str:
        """What a result list says about its order."""
        what = {
            "off": "Reranker switched off",
            "missing": f"Reranker not set up ({self.detail})",
            "no_gpu": "Reranker needs an NVIDIA GPU",
            "busy": "Reranker busy (the GPU is in use)",
        }.get(self.reason, f"Reranker failed ({self.detail[:160]})")
        return f"{what} — sorted by meaning and wording"


# ── Configuration ────────────────────────────────────────────────────────


def settings() -> dict:
    """The ``rerank`` section with its defaults filled in."""
    from whisper_tm.config import config

    out = dict(DEFAULTS)
    user = config.get("rerank") or {}
    if isinstance(user, dict):
        out.update({k: v for k, v in user.items() if v is not None})
    return out


def setting_int(name: str) -> int:
    try:
        return int(settings().get(name))
    except (TypeError, ValueError):
        return int(DEFAULTS[name])


def setting_float(name: str) -> float:
    try:
        return float(settings().get(name))
    except (TypeError, ValueError):
        return float(DEFAULTS[name])


def enabled() -> bool:
    from whisper_tm.core.helpers import parse_bool

    return parse_bool(settings().get("enabled", True))


def default_instruction() -> str:
    """``rerank.instruction`` — what a quick search asks the model to judge."""
    return str(settings().get("instruction") or GENERIC_INSTRUCTION).strip()


def base_directory() -> Path:
    return Path(__file__).resolve().parents[2]


def binary() -> Path | None:
    """``llama-server`` — the same one the chat and embedding servers use."""
    from whisper_tm.core import llama_server

    return llama_server.binary()


def model_file() -> Path | None:
    """The configured reranker GGUF, ``None`` when it isn't there.

    No fallback to another file in ``models/gguf``: the chat and embedding
    models live there, and neither ranks anything.
    """
    name = str(settings().get("local_model") or "").strip()
    if not name:
        return None
    p = Path(name)
    if not p.is_absolute():
        p = base_directory() / p
    return p if p.exists() else None


def _host() -> str:
    h = str(settings().get("host") or "127.0.0.1")
    return "127.0.0.1" if h in ("0.0.0.0", "::") else h


def _port() -> int:
    return setting_int("port")


def endpoint() -> str:
    return f"http://{_host()}:{_port()}"


def available() -> tuple[bool, str]:
    """Can the reranker be operated at all (switch, program, model file)?"""
    if not enabled():
        return False, "switched off (rerank.enabled)"
    if not binary():
        from whisper_tm.core import commands

        return False, f"llama-server not found — {commands.llama_missing()}"
    if not model_file():
        name = str(settings().get("local_model") or "").strip()
        return False, (
            f"model file not found: {name}"
            if name
            else "no reranker model set (rerank.local_model)"
        )
    return True, ""


def _free_vram_mb() -> int | None:
    """Free VRAM of the working GPU (NVML/nvidia-smi), ``None`` without one."""
    from whisper_tm.core import vram

    return vram.free_mb()


# ── The instruction: a template that takes one ─────────────────────────────


def derived_template(template) -> str | None:
    """``template`` with its instruction line turned into ``<Instruct>: {query}``.

    ``None`` unless it holds exactly one ``<Instruct>: …\\n<Query>: {query}``
    line (and a ``{document}``): another model's template is left alone.
    """
    if not isinstance(template, str) or "{document}" not in template:
        return None
    if len(_INSTRUCT_LINE.findall(template)) != 1:
        return None
    return _INSTRUCT_LINE.sub(DERIVED_LINE, template, count=1)


def build_query(text: str, instruction_text: str | None, instructable: bool) -> str:
    """The ``query`` of a rerank request: with an instruction where the
    template takes one, else the text alone."""
    if instructable and instruction_text:
        return f"{instruction_text.strip()}\n<Query>: {text}"
    return text


_prepare_lock = threading.Lock()


def _derived_targets(src: Path) -> list[Path]:
    """Where the instruction-ready copy may go: next to the model (every
    instance of the app shares it), else under the data folder."""
    name = f"{src.stem}.instruct.gguf"
    out = [src.parent / "derived" / name]
    try:
        from whisper_tm.config import config

        out.append(Path(config.data_dir) / "derived" / name)
    except Exception as e:  # noqa: BLE001 — the first place is enough then
        logger.debug("Reranker: no data folder for the derived model: %s", e)
    return out


def _sidecar(target: Path) -> Path:
    return target.with_name(target.name[: -len(".gguf")] + ".json")


#: The copy's layout: 2 = header padded to the original's data offset, so it
#: can share the weights (core/gguf.py). A copy of layout 1 (a full one,
#: 2.4 GB) is written again once — as a reflink where the disk allows.
LAYOUT = 2


def _fingerprint(src: Path, template: str) -> dict:
    st = src.stat()
    return {
        "source": str(src),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "layout": LAYOUT,
    }


def _is_current(target: Path, want: dict) -> bool:
    try:
        if not target.is_file() or target.stat().st_size < want["size"] // 2:
            return False
        return json.loads(_sidecar(target).read_text(encoding="utf-8")) == want
    except (OSError, ValueError):
        return False


def prepare_model(src: Path) -> tuple[Path, bool]:
    """``(file to serve, takes an instruction)`` for the model file ``src``.

    Writes the instruction-ready copy when there is none for this exact
    source yet (a header rewrite and a copy of the weights: seconds). Any
    failure serves the original with the plain query — worse ranking, but a
    ranking.
    """
    with _prepare_lock:
        try:
            template = gguf.read_metadata(src, keys={TEMPLATE_KEY}).get(TEMPLATE_KEY)
        except (OSError, gguf.GgufError) as e:
            logger.warning(
                "Reranker: the template of %s is not readable (%s) — "
                "serving it with the plain query",
                src.name,
                e,
            )
            return src, False
        new = derived_template(template)
        if new is None:
            logger.info(
                "Reranker: %s has no known instruction line — served as it "
                "is, with the plain query",
                src.name,
            )
            return src, False
        want = _fingerprint(src, new)
        targets = _derived_targets(src)
        for target in targets:
            if _is_current(target, want):
                return target, True
        for target in targets:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                free = shutil.disk_usage(target.parent).free
                # A reflink (btrfs, XFS) shares the weights and needs only the
                # header's room; a full copy needs the model's size
                roomy = free >= want["size"] + (512 << 20)
                t0 = time.monotonic()
                try:
                    gguf.rewrite_string(src, target, TEMPLATE_KEY, new, allow_copy=roomy)
                except gguf.GgufError as e:
                    if roomy:
                        raise
                    logger.warning(
                        "Reranker: %s has only %d MB free — no room for the "
                        "instruction-ready copy (%s)",
                        target.parent,
                        free >> 20,
                        e,
                    )
                    continue
                back = gguf.read_metadata(target, keys={TEMPLATE_KEY}).get(TEMPLATE_KEY)
                if back != new:
                    raise gguf.GgufError("the written template reads back differently")
                side = _sidecar(target)
                tmp = side.with_name(side.name + ".tmp")
                tmp.write_text(json.dumps(want), encoding="utf-8")
                tmp.replace(side)
                logger.info(
                    "Reranker: instruction-ready copy of %s written to %s in %.1f s",
                    src.name,
                    target,
                    time.monotonic() - t0,
                )
                return target, True
            except (OSError, gguf.GgufError) as e:
                logger.warning("Reranker: could not write %s (%s)", target, e)
        logger.warning(
            "Reranker: no instruction-ready copy of %s — serving it with the plain query", src.name
        )
        return src, False


# ── The start command ──────────────────────────────────────────────────────


def _start_command(model: Path) -> list[str]:
    """The llama-server invocation in rerank mode — every layer on the card."""
    s = settings()
    batch = str(setting_int("batch"))
    argv = [
        str(binary()),
        "--model",
        str(model),
        "--host",
        _host(),
        "--port",
        str(_port()),
        # /v1/rerank needs the embedding path with rank pooling
        "--embedding",
        "--pooling",
        "rank",
        "--reranking",
        "--ctx-size",
        str(setting_int("ctx")),
        "--parallel",
        str(setting_int("parallel")),
        # A passage is judged in one pass: the batch holds the longest one
        "--batch-size",
        batch,
        "--ubatch-size",
        batch,
        # GPU only: a partly offloaded 4B is nearly as slow as the CPU
        "--n-gpu-layers",
        "999",
    ]
    # No prompt cache: a passage is never judged twice, and the cache held
    # several GB of RAM (core/llama_server.py)
    from whisper_tm.core import llama_server, local_key

    argv += llama_server.no_prompt_cache_args(binary())
    argv += local_key.launch_args(binary())
    extra = str(s.get("extra_args") or "").strip()
    if extra:
        argv += commands.split_args(extra)
    return argv


# ── Holding the GPU gate ───────────────────────────────────────────────────


class _GateHold:
    """The stage ``rerank`` of the GPU gate, taken and held by a thread of
    its own until :meth:`release` — a stage belongs to the thread that took
    it, and the server outlives every single call."""

    def __init__(self, gate, need_mb: int):
        self.gate = gate
        self.need_mb = need_mb
        self.entered = threading.Event()
        self.done = threading.Event()
        self._off = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="rerank-gpu-gate", daemon=True)

    def start(self) -> _GateHold:
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            with self.gate.stage(STAGE, self.need_mb, cancel=self._off.is_set):
                self.entered.set()
                self._off.wait()
        except InterruptedError:
            pass
        except Exception as e:  # noqa: BLE001 — reported as "not entered"
            self.error = e
            logger.warning("Reranker: GPU gate: %s", e)
        finally:
            self.done.set()

    def wait(self, deadline: float, cancel=None) -> bool:
        """Until the stage is held (True) or the time is up / ``cancel()``."""
        while not self.entered.is_set():
            if self.done.is_set():
                return False
            if time.monotonic() >= deadline or (cancel is not None and cancel()):
                self.release()
                return False
            self.entered.wait(0.1)
        return True

    def release(self) -> None:
        self._off.set()


# ── The server ─────────────────────────────────────────────────────────────


def _post_rerank(ep: str, query: str, docs: list[str]) -> list[float]:
    """One ``/v1/rerank`` request: a score per document, in their order."""
    import requests

    try:
        from whisper_tm.core import local_key

        r = requests.post(
            f"{ep}/v1/rerank",
            headers=local_key.headers_for(f"{ep}/v1/rerank"),
            json={"query": query, "documents": docs, "top_n": len(docs)},
            timeout=(5, CALL_TIMEOUT_SEC),
        )
    except requests.RequestException as e:
        raise RerankUnavailable("failed", f"no answer from the reranker: {str(e)[:160]}") from e
    if r.status_code != 200:
        raise RerankUnavailable("failed", f"the reranker answered {r.status_code}: {r.text[:200]}")
    try:
        results = r.json().get("results")
        scores: list[float | None] = [None] * len(docs)
        for item in results:
            scores[int(item["index"])] = float(item["relevance_score"])
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
        raise RerankUnavailable("failed", f"unexpected answer from the reranker: {e}") from e
    if any(s is None for s in scores):
        raise RerankUnavailable("failed", "the reranker left documents without a score")
    return scores  # type: ignore[return-value]


class RerankServer:
    """A llama-server for reranking that starts on demand and steps aside."""

    def __init__(self, gate=None):
        self._lock = threading.RLock()
        # One start at a time, the waits for the card and the model included
        self._start_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._hold: _GateHold | None = None
        self._address: tuple[str, int] | None = None
        #: /health answered since the start — set after it, cleared on stop
        self._ready = False
        self._log: Path | None = None
        self._model: Path | None = None
        self._instructable = False
        self._started_at = 0.0  # wall clock, for the state
        self._held_since = 0.0  # monotonic: when it took the card
        self._last_used = 0.0  # monotonic
        self._active = 0  # calls in flight
        self._watchdog: threading.Thread | None = None
        self._external_logged = False
        #: The GPU gate — injectable for tests; ``None`` = core/gpu_gate.GATE
        self.gate = gate

    def _gate(self):
        if self.gate is not None:
            return self.gate
        from whisper_tm.core.gpu_gate import GATE

        return GATE

    # ── State ────────────────────────────────────────────────────────────
    def _own_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _responds(self, timeout: float = 2.0) -> bool:
        import requests

        try:
            return requests.get(f"{endpoint()}/health", timeout=timeout).status_code == 200
        except Exception:  # noqa: BLE001 — not answering is the answer
            return False

    def pid(self) -> int | None:
        """The running llama-server's PID (core/vram_procs.py names it)."""
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def state(self) -> dict:
        with self._lock:
            alive = self._own_alive()
            return {
                "running": alive,
                "endpoint": endpoint(),
                "model": str(self._model or model_file() or ""),
                "instructable": self._instructable if alive else None,
                "pid": self._proc.pid if alive else 0,
                "busy_calls": self._active,
                "running_for_sec": (
                    round(time.time() - self._started_at, 1) if alive and self._started_at else 0.0
                ),
                "idle_sec": (
                    round(time.monotonic() - self._last_used, 1)
                    if alive and self._last_used
                    else 0.0
                ),
                "idle_limit_sec": setting_int("idle_stop_sec"),
            }

    # ── A call ───────────────────────────────────────────────────────────
    @contextlib.contextmanager
    def session(self, *, wait_sec: float = QUICK_WAIT_SEC, cancel=None):
        """``(endpoint, takes_instruction)`` for one call; the server is not
        stopped while the call runs."""
        ep, instructable = self._checkout(wait_sec, cancel)
        try:
            yield ep, instructable
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                self._last_used = time.monotonic()

    def _take_call(self) -> tuple[str, bool]:
        """Count a call in (caller holds _lock)."""
        self._active += 1
        self._last_used = time.monotonic()
        return endpoint(), self._instructable

    def _checkout(self, wait_sec: float, cancel) -> tuple[str, bool]:
        wait_sec = max(0.0, float(wait_sec))
        deadline = time.monotonic() + wait_sec
        if not enabled():
            # Switched off while it ran: it goes now, not after its idle time
            self.stop("switched off (rerank.enabled)")
            raise RerankUnavailable("off", "switched off (rerank.enabled)")
        with self._lock:
            # Ready, not just alive: during a cold start the others wait on
            # the start lock instead of posting before /health answers
            if self._ready and self._own_alive() and self._address == (_host(), _port()):
                return self._take_call()
        if not self._acquire_start(deadline, cancel):
            if cancel is not None and cancel():
                raise InterruptedError("Cancelled")
            raise RerankUnavailable("busy", "another search is waiting for the GPU")
        try:
            with self._lock:
                if self._own_alive() and self._address == (_host(), _port()):
                    return self._take_call()
                if self._own_alive():
                    # host or port changed in the settings while it ran
                    self.stop("its address changed", force=True)
            if self._responds():
                # Not ours: started by hand, or by another instance. Used as it
                # is — its template is unknown, so it gets the plain query.
                if not self._external_logged:
                    logger.info(
                        "Reranker: a server this app did not start answers on "
                        "%s — using it as it is",
                        endpoint(),
                    )
                    self._external_logged = True
                with self._lock:
                    self._active += 1
                    self._last_used = time.monotonic()
                return endpoint(), False
            self._start(wait_sec, cancel)
            with self._lock:
                if not self._own_alive():
                    raise RerankUnavailable("failed", "the server stopped right after its start")
                return self._take_call()
        finally:
            self._start_lock.release()

    def _acquire_start(self, deadline: float, cancel) -> bool:
        while True:
            left = deadline - time.monotonic()
            if self._start_lock.acquire(timeout=max(0.0, min(0.2, left))):
                return True
            if left <= 0 or (cancel is not None and cancel()):
                return False

    # ── Starting ─────────────────────────────────────────────────────────
    def _take_gpu(self, need: int, deadline: float, cancel) -> _GateHold:
        """The gate stage, with ``need`` MB free on the card."""
        while True:
            hold = _GateHold(self._gate(), need).start()
            if not hold.wait(deadline, cancel):
                if cancel is not None and cancel():
                    raise InterruptedError("Cancelled")
                raise RerankUnavailable("busy", "the GPU is in use by the queue")
            free = _free_vram_mb()
            if free is None or free >= need:
                return hold
            hold.release()
            left = deadline - time.monotonic()
            if left < 1.0:
                raise RerankUnavailable(
                    "busy", f"only {free} MB of VRAM free, the reranker needs {need} MB"
                )
            end = time.monotonic() + min(5.0, left)
            while time.monotonic() < end:
                if cancel is not None and cancel():
                    raise InterruptedError("Cancelled")
                time.sleep(0.2)

    def _start(self, wait_sec: float, cancel) -> None:
        """Start the server (caller holds _start_lock). ``wait_sec``: how long
        to wait for the card — counted after the model file is ready, whose
        first preparation alone takes seconds."""
        ok, why = available()
        if not ok:
            raise RerankUnavailable("off" if why.startswith("switched off") else "missing", why)
        if _free_vram_mb() is None:
            raise RerankUnavailable("no_gpu", "no NVIDIA GPU found")
        model, instructable = prepare_model(model_file())
        need = setting_int("vram_need_mb")
        hold = self._take_gpu(need, time.monotonic() + wait_sec, cancel)
        try:
            # Inside the try: a bad rerank.extra_args (shlex) kept the stage
            # held forever
            argv = _start_command(model)
            folder = debug.logs_root()
            folder.mkdir(parents=True, exist_ok=True)
            self._log = folder / "rerank-server.log"
            from whisper_tm.core import llama_server

            llama_server.trim_log(self._log)
            with self._lock:
                logger.info("Reranker starting on the GPU: %s …", " ".join(argv[:6]))
                with open(self._log, "ab") as fh:
                    fh.write(f"\n=== Start {time.strftime('%F %T')} ===\n".encode())
                    self._proc = subprocess.Popen(
                        argv,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        # Its own process group: stopping takes the whole group
                        start_new_session=True,
                        env=dict(os.environ),
                    )
                # Should this app end without stopping it, the next start
                # ends it (core/helper_pids)
                from whisper_tm.core import helper_pids

                helper_pids.record("rerank-server", self._proc.pid, _port())
                self._hold = hold
                self._address = (_host(), _port())
                self._model = model
                self._instructable = instructable
                self._started_at = time.time()
                self._held_since = time.monotonic()
                self._last_used = time.monotonic()
        except Exception as e:
            hold.release()
            raise RerankUnavailable("failed", f"llama-server could not start: {e}") from e
        limit = setting_float("start_timeout_sec")
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            if self._responds(timeout=2.0):
                logger.info(
                    "Reranker ready after %.1f s (%s)",
                    time.monotonic() - t0,
                    "own instruction" if instructable else "plain query",
                )
                with self._lock:
                    self._last_used = time.monotonic()
                    self._ready = self._own_alive()
                self._start_watchdog()
                return
            if not self._own_alive():
                tail = self._log_tail()
                self.stop("it ended during its start", force=True)
                raise RerankUnavailable("failed", f"the server ended during its start. {tail}")
            if cancel is not None and cancel():
                self.stop("the search was stopped", force=True)
                raise InterruptedError("Cancelled")
            time.sleep(0.5)
        tail = self._log_tail()
        self.stop("not ready in time", force=True)
        raise RerankUnavailable("failed", f"not ready after {limit:.0f} s. {tail}")

    def _log_tail(self, lines: int = 3) -> str:
        from whisper_tm.core import llama_server

        return llama_server.log_tail(self._log, lines)

    # ── Stopping ─────────────────────────────────────────────────────────
    def stop(self, reason: str = "", *, force: bool = False) -> bool:
        """Stop the server this app started and give the card back. A call in
        flight keeps it running unless ``force``."""
        with self._lock:
            if self._active and not force:
                return False
            p, hold = self._proc, self._hold
            if p is None and hold is None:
                return False
            if p is not None and p.poll() is None:
                logger.info("Reranker stopping (pid %s)%s", p.pid, f": {reason}" if reason else "")
                if not signal_group(p.pid):
                    p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logger.warning("Reranker not answering SIGTERM — SIGKILL")
                    if not signal_group(p.pid, kill=True):
                        p.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        p.wait(timeout=5)
            # The card goes back only once the process is gone
            if hold is not None:
                hold.release()
            if p is not None:
                from whisper_tm.core import helper_pids

                helper_pids.forget("rerank-server")
            self._proc = None
            self._hold = None
            self._address = None
            self._ready = False
            self._started_at = 0.0
            return True

    def yield_if_wanted(self) -> bool:
        """Between two batches of a long run: give the card back when it was
        held for :data:`YIELD_AFTER_SEC` and someone waits at the gate. The
        next batch takes it again (and waits its turn)."""
        with self._lock:
            if not self._own_alive() or self._active:
                return False
            if time.monotonic() - self._held_since < YIELD_AFTER_SEC:
                return False
            if not self._gate().waiting():
                return False
        return self.stop("the queue waits for the GPU — back after its turn")

    # ── Idle watcher ─────────────────────────────────────────────────────
    def _start_watchdog(self) -> None:
        with self._lock:
            if self._watchdog and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(
                target=self._watch, name="rerank-server-watcher", daemon=True
            )
            self._watchdog.start()

    def _watch(self) -> None:
        while True:
            time.sleep(WATCH_TICK_SEC)
            reason = None
            with self._lock:
                if self._proc is None:
                    return
                if not self._own_alive():
                    logger.warning(
                        "Reranker ended on its own (exit %s). %s",
                        self._proc.returncode,
                        self._log_tail(),
                    )
                    self.stop("it ended", force=True)
                    return
                if self._active:
                    continue
                quiet = time.monotonic() - self._last_used
                limit = setting_float("idle_stop_sec")
                if limit > 0 and quiet >= limit:
                    reason = f"unused for {quiet:.0f} s"
                elif quiet >= YIELD_GRACE_SEC and self._gate().waiting():
                    reason = "the queue waits for the GPU"
            if reason and self.stop(reason):
                return

    # ── Ranking ──────────────────────────────────────────────────────────
    def rerank(
        self,
        query: str,
        docs: list[str],
        *,
        instruction: str | None = None,
        wait_sec: float = QUICK_WAIT_SEC,
        cancel=None,
        progress=None,
        batch: int = BATCH,
        yield_between: bool = False,
    ) -> list[float]:
        """A relevance score (0..1) per document, in their order.

        ``instruction``: what the model judges (default ``rerank.instruction``);
        sent with the query where the template takes one. ``wait_sec``: how
        long to wait for the card (each batch); ``cancel()`` ends waits and
        stops between batches (InterruptedError). ``progress(done, total)``
        after each batch. ``yield_between``: a long run lets the queue have
        the card between batches (:meth:`yield_if_wanted`).

        Raises RerankUnavailable when the card can't be had or the server
        fails.
        """
        if not docs:
            return []
        texts = [(d or "")[:DOC_CHARS] for d in docs]
        what = instruction if instruction is not None else default_instruction()
        out: list[float] = []
        t0 = time.monotonic()
        for i in range(0, len(texts), max(1, int(batch))):
            if cancel is not None and cancel():
                raise InterruptedError("Cancelled")
            part = texts[i : i + max(1, int(batch))]
            with self.session(wait_sec=wait_sec, cancel=cancel) as (ep, instructable):
                out.extend(_post_rerank(ep, build_query(query, what, instructable), part))
            if progress is not None:
                progress(len(out), len(texts))
            if yield_between and len(out) < len(texts):
                self.yield_if_wanted()
        logger.info("Reranked %d passages in %.1f s", len(texts), time.monotonic() - t0)
        return out


#: Exactly one per process — two would start on the same port.
SERVER = RerankServer()


def rerank(query: str, docs: list[str], **kw) -> list[float]:
    """:meth:`RerankServer.rerank` on the process's server."""
    return SERVER.rerank(query, docs, **kw)


def status() -> dict:
    """For the status page: set up, and running?"""
    ok, why = available()
    return {"enabled": enabled(), "available": ok, "detail": why, **SERVER.state()}


def on_shutdown() -> None:
    try:
        SERVER.stop("the app exits", force=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("Reranker cleanup: %s", e)


# Whoever imports this module may start the server, so it stops it too —
# the same reasoning as core/embed_server.py.
atexit.register(on_shutdown)

"""
Local llama.cpp server: start command from config, status check.

The setup can install llama.cpp alongside (``./setup.sh --llama``); the
operating parameters are then in the ``llama`` section of the config. Here
we build the start command from that (start-llm.sh writes it), and the
health check gets status assessment from here.

The config defaults suit the model the setup downloads (Qwen3.5-9B at
Q4_K_M) on an 8 GB card. It stays generic because every value can be
overridden individually and ``check()`` tells where assumptions about the
available machine don't fit — instead of silently delivering poor
performance.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import commands

logger = logging.getLogger(__name__)


def _cfg(name: str, default=None):
    return config.get("llama", name, default=default)


def base_directory() -> Path:
    """Project root (setup.sh creates llama.cpp/ and models/gguf/ there)."""
    return Path(__file__).resolve().parents[2]


def binary() -> Path | None:
    """Path to llama-server, if installed."""
    for candidate in (
        base_directory() / "llama.cpp" / "llama-server",
        base_directory() / "llama.cpp" / "llama-server.exe",
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else None


_help_lock = threading.Lock()
_help_cache: dict[tuple[str, int], str] = {}


def help_text(bin_: Path | None = None) -> str:
    """``llama-server --help`` of this binary — read once per file version
    (18 ms), with the card hidden so that it touches no GPU; "" when it
    can't be run."""
    bin_ = bin_ or binary()
    if not bin_:
        return ""
    try:
        key = (str(bin_), Path(bin_).stat().st_mtime_ns)
    except OSError:
        return ""
    with _help_lock:
        if key in _help_cache:
            return _help_cache[key]
    try:
        out = subprocess.run(
            [str(bin_), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        text = (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("llama-server --help: %s", e)
        text = ""
    with _help_lock:
        _help_cache[key] = text
    return text


def supports(flag: str, bin_: Path | None = None) -> bool:
    """Does this llama-server know ``flag`` (e.g. ``--cache-ram``)? An older
    build stops at an unknown argument instead of starting."""
    return (
        re.search(rf"(?:^|[\s,]){re.escape(flag)}(?=[\s,=]|$)", help_text(bin_), re.MULTILINE)
        is not None
    )


def _known(flag: str, bin_: Path | None = None) -> bool:
    """Does the build know ``flag``? When its help can't be read, taken as
    yes — as on a current build."""
    return not help_text(bin_) or supports(flag, bin_)


def _flash_attn_args(bin_: Path | None, on: bool) -> list[str]:
    """``-fa on|off``. Builds from before mid-2025 know ``-fa`` only as a
    switch and stop at the "on" after it: there the switch alone."""
    text = help_text(bin_)
    if text and supports("--flash-attn", bin_) and not re.search(r"--flash-attn\s+\[?on", text):
        return ["-fa"] if on else []
    return ["-fa", "on" if on else "off"]


def _no_reasoning_args(bin_: Path | None) -> list[str]:
    """Thinking off: ``--reasoning off``; on the builds before it
    ``--reasoning-budget 0``; older ones have neither and stop at it."""
    if _known("--reasoning", bin_):
        return ["--reasoning", "off"]
    if supports("--reasoning-budget", bin_):
        return ["--reasoning-budget", "0"]
    return []


def no_prompt_cache_args(bin_: Path | None = None) -> list[str]:
    """``--cache-ram 0`` where the build has it — for the embedding and the
    rerank server.

    Since late 2025 llama-server keeps the state of every finished prompt in
    RAM (``--cache-ram``, 8 GiB by default) so that a chat can pick it up
    again. An embedding or a rerank never comes back to a prompt: measured,
    the rerank server held several GB of RAM for it, and an embedding
    run copied 60–90 MB of state off the card for each of its passages
    ("making room for prompt cache entry" in its log).
    """
    return ["--cache-ram", "0"] if supports("--cache-ram", bin_) else []


#: A server log longer than this is cut at its next start, keeping the tail —
#: a long embedding run can leave a log of hundreds of MB behind.
LOG_MAX_BYTES = 20 << 20
LOG_KEEP_BYTES = 2 << 20


def trim_log(
    path: Path, *, max_bytes: int = LOG_MAX_BYTES, keep_bytes: int = LOG_KEEP_BYTES
) -> None:
    """Cut a server log that grew past ``max_bytes`` to its last
    ``keep_bytes`` (from a line start). Never raises."""
    try:
        if not path.is_file() or path.stat().st_size <= max_bytes:
            return
        with open(path, "rb") as fh:
            fh.seek(-keep_bytes, os.SEEK_END)
            tail = fh.read()
        cut = tail.find(b"\n")
        tail = tail[cut + 1 :] if cut >= 0 else tail
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(b"=== (older output cut) ===\n" + tail)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Trimming %s: %s", path, e)


def log_tail(path: Path | None, lines: int = 3) -> str:
    """The last ``lines`` lines of a server log, read from its end — the logs
    grow large, and a start error must not read all of one."""
    if not path:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 16384))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    last = [x.strip() for x in text.splitlines() if x.strip()][-lines:]
    return ("Last output: " + " | ".join(last)) if last else ""


#: GGUF files in models/gguf/ that are no chat model: the embedding and
#: reranker models of the semantic search, a vision projector, and every
#: part of a split model but the first. Picked by name order they came
#: first ("Qwen3-Embedding…" before "qwen2.5…"), and the chat server
#: loaded the embedding model.
_NOT_A_CHAT_MODEL = re.compile(r"embed|rerank|mmproj|-(?!00001)\d{5}-of-\d{5}", re.I)


def model_file() -> Path | None:
    """The configured model (a relative path is relative to the project
    folder), else the first chat model in models/gguf/."""
    set_flag = (_cfg("model") or "").strip()
    if set_flag:
        p = Path(set_flag).expanduser()
        if not p.is_absolute():
            p = base_directory() / p
        return p if p.exists() else None
    folder = base_directory() / "models" / "gguf"
    if not folder.is_dir():
        return None
    hits = sorted(p for p in folder.glob("*.gguf") if not _NOT_A_CHAT_MODEL.search(p.name))
    return hits[0] if hits else None


def endpoint() -> str:
    host = _cfg("host", "127.0.0.1") or "127.0.0.1"
    # 0.0.0.0 is a bind address, not a target address
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{_cfg('port', 8080)}/v1"


def recommended_threads() -> tuple[int, str]:
    """(Threads, reason) — conservative because overbooking is expensive.

    Measured: response speed drops by a factor of 6.6 as soon as more threads
    run than physical cores exist (spin-wait barriers against a hypervisor-
    preempted thread). Therefore: physical cores, capped at 8 — and one fewer
    in a VM.
    """
    set_flag = int(_cfg("threads", 0) or 0)
    if set_flag > 0:
        return set_flag, "from config"
    physical = 0
    try:
        # os.cpu_count() counts SMT siblings; /proc/cpuinfo has the true
        # cores (on Linux; otherwise we halve the threads).
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cores = {
                tuple(
                    z.split(":")[1].strip()
                    for z in block.splitlines()
                    if z.startswith(("physical id", "core id"))
                )
                for block in f.read().split("\n\n")
                if "core id" in block
            }
        physical = len({k for k in cores if k})
    except OSError:
        pass
    if not physical:
        physical = max(1, (os.cpu_count() or 2) // 2)
    if in_vm():
        physical = max(1, physical - 1)
        return min(physical, 8), "physical cores − 1, virtualized"
    return min(physical, 8), "physical cores, at most 8"


def in_vm() -> bool:
    try:
        out = subprocess.run(["systemd-detect-virt"], capture_output=True, text=True, timeout=3)
        return out.returncode == 0 and out.stdout.strip() not in ("", "none")
    except Exception:
        return False


def start_command() -> list[str]:
    """Complete llama-server call from config."""
    bin_ = binary()
    model = model_file()
    if not bin_ or not model:
        return []
    threads, _reason = recommended_threads()
    argv = [
        str(bin_),
        "--model",
        str(model),
        "--host",
        str(_cfg("host", "127.0.0.1")),
        "--port",
        str(_cfg("port", 8080)),
        "--ctx-size",
        str(_cfg("ctx_size", 32768)),
        "--batch-size",
        str(_cfg("batch_size", 2048)),
        "--ubatch-size",
        str(_cfg("ubatch_size", 512)),
        "--threads",
        str(threads),
        "--parallel",
        str(_cfg("parallel", 1)),
        "--cache-type-k",
        str(_cfg("cache_type_k", "f16")),
        "--cache-type-v",
        str(_cfg("cache_type_v", "f16")),
    ]
    # An older llama-server stops at a flag it doesn't know instead of
    # starting: what came later goes only to a build that has it
    if _known("--jinja", bin_):
        argv += ["--jinja"]
    # 0 means the CPU only — left out, the server would decide by itself
    # (newer builds fit as many layers as the free VRAM takes)
    ngl = _cfg("n_gpu_layers", "")
    if ngl not in (None, ""):
        argv += ["--n-gpu-layers", str(int(ngl))]
    ncmoe = int(_cfg("n_cpu_moe", 0) or 0)
    if ncmoe:
        # Only makes sense for MoE models; llama.cpp ignores it otherwise
        if _known("--n-cpu-moe", bin_):
            argv += ["--n-cpu-moe", str(ncmoe)]
        else:
            logger.warning(
                "llama.n_cpu_moe left out: this llama-server is older than "
                "--n-cpu-moe (./setup.sh --llama fetches a current one)"
            )
    argv += _flash_attn_args(bin_, bool(_cfg("flash_attn", True)))
    if not _cfg("reasoning", False):
        argv += _no_reasoning_args(bin_)
    from whisper_tm.core import local_key

    argv += local_key.launch_args(bin_)
    extra = (_cfg("extra_args") or "").strip()
    if extra:
        argv += commands.split_args(extra)
    return argv


def running() -> bool:
    """Does the server answer on its endpoint?"""
    import requests

    try:
        r = requests.get(endpoint().rsplit("/v1", 1)[0] + "/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


#: Bytes per element of a KV cache type (block sizes of llama.cpp's quants)
_KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}

_shape_cache: dict[tuple[str, float], dict] = {}

#: Hybrids whose header may leave out which layers have attention: llama.cpp
#: then gives it to every fourth layer (the others are linear attention with
#: a small fixed state instead of a KV cache)
_FULL_ATTENTION_EVERY = {"qwen3next": 4, "qwen35": 4, "qwen35moe": 4}


def _model_shape(model: Path) -> dict:
    """From the GGUF header: is it a mixture of experts, how many layers,
    and how many KV-cache numbers a token takes — in the layers with
    attention only (``hybrid``: the header says which those are). {} when
    it can't be read."""
    try:
        key = (str(model), model.stat().st_mtime)
        if key in _shape_cache:
            return _shape_cache[key]
        from whisper_tm.core import gguf

        arch = str(
            gguf.read_metadata(model, keys={"general.architecture"}).get("general.architecture")
            or ""
        )
        names = (
            "block_count",
            "expert_count",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.recurrent_layers",
            "full_attention_interval",
        )
        md = gguf.read_metadata(model, keys={f"{arch}.{n}" for n in names}, listed=4096)
    except Exception as e:  # noqa: BLE001 — an estimate, never worth an error
        logger.debug("GGUF header of %s not read: %s", model, e)
        return {}

    def num(name: str) -> int:
        try:
            v = md.get(f"{arch}.{name}") or 0
            # some models give a count per layer
            return int(max(v) if isinstance(v, list) else v)
        except (TypeError, ValueError):
            return 0

    def per_layer(name: str) -> list | None:
        v = md.get(f"{arch}.{name}")
        return v if isinstance(v, list) and len(v) == layers else None

    layers, heads = num("block_count"), num("attention.head_count")
    kv_heads = num("attention.head_count_kv") or heads
    key_len = num("attention.key_length") or (num("embedding_length") // heads if heads else 0)
    value_len = num("attention.value_length") or key_len
    # A hybrid says which layers keep a KV cache: by a flag per layer, by an
    # interval, or by no KV heads in the others
    kv_heads_each = per_layer("attention.head_count_kv") or [kv_heads] * layers
    recurrent = per_layer("attention.recurrent_layers")
    if recurrent is None:
        every = num("full_attention_interval") or _FULL_ATTENTION_EVERY.get(arch, 0)
        recurrent = [every > 1 and (i + 1) % every != 0 for i in range(layers)]
    kv_heads_sum = sum(int(n) for n, r in zip(kv_heads_each, recurrent) if not r)
    shape = {
        "moe": num("expert_count") > 0,
        "layers": layers,
        "hybrid": any(recurrent) or 0 in kv_heads_each,
        "kv_per_token": kv_heads_sum * (key_len + value_len),
    }
    _shape_cache[key] = shape
    return shape


def vram_estimate_mb(model: Path) -> int:
    """Roughly what the configured start takes on the card: weights, KV cache,
    compute buffers. The experts of ``n_cpu_moe`` layers stay in RAM — for a
    mixture of experts only: a dense model ignores the setting, and the
    estimate used to count it as nearly nothing on the card."""
    shape = _model_shape(model)
    size_mb = model.stat().st_size / 2**20
    ncmoe = int(_cfg("n_cpu_moe", 0) or 0)
    moe = shape.get("moe") if shape else bool(ncmoe)
    if moe and ncmoe:
        layers = shape.get("layers") or 0
        share = max(0.0, 1 - ncmoe / layers) if layers else 0.0
        on_gpu_mb = 512 + size_mb * share
    else:
        on_gpu_mb = size_mb
    kv_mb = 0.0
    if shape.get("kv_per_token") and (not moe or shape.get("hybrid")):
        # A mixture of experts may be a hybrid whose header doesn't say so
        # — counted in every layer, its small cache would look big
        ctx = int(_cfg("ctx_size", 32768) or 0)
        per = (
            _KV_BYTES.get(str(_cfg("cache_type_k", "f16")), 2.0)
            + _KV_BYTES.get(str(_cfg("cache_type_v", "f16")), 2.0)
        ) / 2
        kv_mb = ctx * shape["kv_per_token"] * per / 2**20
    return int(on_gpu_mb + kv_mb + int(_cfg("ubatch_size", 512)) / 8192 * 700)


def check() -> dict:
    """Status assessment for health check.

    Reports only what doesn't fit the *existing* machine — the config
    defaults suit the setup's model on an 8 GB card and are only a starting
    point elsewhere.
    """
    bin_ = binary()
    model = model_file()
    threads, threads_reason = recommended_threads()
    active = running()

    out = {
        "installed": bool(bin_),
        "binary": str(bin_) if bin_ else "",
        "model": model.name if model else "",
        "model_gb": round(model.stat().st_size / 1e9, 1) if model else 0,
        "endpoint": endpoint(),
        "running": active,
        "threads": threads,
        "threads_reason": threads_reason,
        "ctx_size": _cfg("ctx_size", 32768),
        "ubatch_size": _cfg("ubatch_size", 512),
        "n_cpu_moe": _cfg("n_cpu_moe", 0),
        "virtualized": in_vm(),
        "warnings": [],
        "hints": [],
    }
    if not bin_:
        out["hints"].append(
            f"llama.cpp is not installed — {commands.llama_missing()}, "
            "or set a different provider in settings."
        )
        return out
    if not model:
        out["warnings"].append("No GGUF model in models/gguf/ — the server won't start.")
    if not active:
        out["hints"].append(
            f"Server is not running. Start with {commands.start_llm()} (it uses VRAM "
            "as long as it runs)."
        )

    # ── VRAM check: does the config fit on the card at all? ──
    try:
        from whisper_tm.core import debug as _debug

        if _debug.gpu_available():
            idx = config.get("hardware", "gpu_index", default=0)
            static_text = _debug.gpu_static()
            info = static_text.get(idx) or next(iter(static_text.values()), {})
            total = info.get("total_mb", 0)
            out["vram_total_mb"] = total
            if total and model:
                need_mb = vram_estimate_mb(model)
                out["vram_needed_mb"] = need_mb
                ngl = _cfg("n_gpu_layers", "")
                if need_mb > total * 0.95 and ngl not in (None, "", 0, "0"):
                    moe = _model_shape(model).get("moe")
                    fix = (
                        "a smaller ubatch_size or a larger n_cpu_moe"
                        if moe
                        else "a smaller ctx_size or ubatch_size, or n_gpu_layers left "
                        "empty (llama.cpp then fits the layers to the free VRAM)"
                    )
                    out["warnings"].append(
                        f"Config needs roughly {need_mb} MiB, card has "
                        f"{total} MiB — llama-server crashes on start if it "
                        f"doesn't fit. Choose {fix}."
                    )
    except Exception as e:
        logger.debug("VRAM check: %s", e)

    # ── Thread overbooking: the most expensive single error ──
    visible = os.cpu_count() or 0
    set_flag = int(_cfg("threads", 0) or 0)
    if set_flag and visible and set_flag > visible:
        out["warnings"].append(
            f"threads={set_flag} exceeds {visible} visible CPUs — this "
            f"crashes response speed (measured up to factor 6)."
        )
    elif out["virtualized"] and set_flag == 0:
        out["hints"].append(
            f"Virtualized system detected: threads auto-set to {threads}. "
            f"If the guest has more vCPUs than the host has physical cores, "
            f"manually set it lower."
        )

    # ── Is the server running with different values than configured? ──
    if active:
        try:
            import requests

            from whisper_tm.core import local_key

            url = endpoint().rsplit("/v1", 1)[0] + "/props"
            r = requests.get(url, headers=local_key.headers_for(url), timeout=3)
            if r.status_code == 200:
                props = r.json()
                n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
                out["running_ctx"] = n_ctx
                if n_ctx and n_ctx != int(_cfg("ctx_size", 32768)):
                    out["hints"].append(
                        f"Running server has {n_ctx} token context, config "
                        f"says {_cfg('ctx_size', 32768)} — must restart after "
                        f"changing."
                    )
        except Exception as e:
            logger.debug("props: %s", e)
    return out

"""The media environment: its own Python for the embedding service.

EmbeddingGemma 2 needs transformers >= 5.19 and sentence-transformers >=
6.1; the app's own environment is pinned to older builds that WhisperX,
pyannote and the separator were tested with, and must not move. So the
service (media/service.py) runs in a virtual environment of its own —
``media_search.environment``, by default ``<data folder>/media-env``.

**Installing** goes through uv when it is there (it links the packages
from its cache: the environment takes far less new disk than its packages'
size, torch reused from the cache), otherwise through
``python -m venv`` and pip. torch comes from the PyTorch index matching the
machine: CUDA 12.8 wheels with an NVIDIA card, CPU wheels without. The model
(~1.5 GB) is fetched into the Hugging Face cache at the end; after that the
service runs offline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: What the service needs — the versions it was measured with.
PACKAGES = [
    "transformers==5.19.0",
    "sentence-transformers==6.1.0",
    "numpy>=1.26",
    "pillow>=10.1",
    "av>=14",
    "librosa>=0.10",
    "soundfile>=0.12",
    "huggingface_hub>=1.31",
]
TORCH = ["torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0"]
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

_install_lock = threading.Lock()
_install: dict = {
    "running": False,
    "lines": [],
    "ok": None,
    "error": None,
    "started": None,
    "ended": None,
}
_check_cache: dict = {"at": 0.0, "key": None, "value": None}


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("media_search", name, default=default)


def env_dir() -> Path:
    from whisper_tm.config import config

    p = config.expand_setting(_cfg("environment", ""), "media_search.environment")
    if p:
        return p
    return Path(config.data_dir) / "media-env"


def python_path() -> Path:
    d = env_dir()
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def model_id() -> str:
    return (_cfg("model", "") or "google/embeddinggemma-2").strip()


def _has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _uv() -> str | None:
    for cand in (shutil.which("uv"), str(Path.home() / ".local/bin/uv")):
        if cand and Path(cand).is_file():
            return cand
    return None


def check(force: bool = False) -> dict:
    """What the environment holds. It starts the environment's Python
    (~5 s), so it is kept: a call returns the last result at once and looks
    again in the background when that is a minute old. Only the very first
    call (or force) waits."""
    key = (str(python_path()), model_id())
    now = time.monotonic()
    if not force and _check_cache["key"] == key and _check_cache["value"] is not None:
        if now - _check_cache["at"] > 60 and not _check_cache.get("refreshing"):
            _check_cache["refreshing"] = True
            threading.Thread(target=_refresh, name="media-env-check", daemon=True).start()
        return _check_cache["value"]
    return _probe(key)


def _refresh() -> None:
    try:
        _probe((str(python_path()), model_id()))
    finally:
        _check_cache["refreshing"] = False


def _probe(key: tuple) -> dict:
    py = python_path()
    now = time.monotonic()
    out = {
        "path": str(env_dir()),
        "python": str(py),
        "exists": py.exists(),
        "packages": {},
        "model": model_id(),
        "model_cached": False,
        "ok": False,
        "problems": [],
    }
    if not py.exists():
        out["problems"].append("not installed")
    else:
        probe = (
            "import json, importlib\n"
            "out = {}\n"
            "for n in ('torch', 'transformers', 'sentence_transformers', 'av', 'PIL', 'numpy'):\n"
            "    try: out[n] = importlib.import_module(n).__version__\n"
            "    except Exception as e: out[n] = None\n"
            "try:\n"
            "    from huggingface_hub import try_to_load_from_cache\n"
            f"    hit = try_to_load_from_cache({model_id()!r}, 'model.safetensors')\n"
            "    out['model'] = isinstance(hit, str)\n"
            "except Exception: out['model'] = False\n"
            "print(json.dumps(out))\n"
        )
        try:
            r = subprocess.run(
                [str(py), "-c", probe],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            got = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
            got = {}
            out["problems"].append(f"the environment's Python does not run: {e}")
        out["model_cached"] = bool(got.pop("model", False))
        out["packages"] = got
        for name in ("torch", "transformers", "sentence_transformers", "av", "PIL", "numpy"):
            if not got.get(name):
                out["problems"].append(f"{name} missing")
        tv = got.get("transformers") or ""
        if tv and _version_tuple(tv) < (5, 19):
            out["problems"].append(f"transformers {tv} is too old (needs 5.19 or newer)")
        if not out["model_cached"]:
            out["problems"].append("model not downloaded yet")
    out["ok"] = not out["problems"]
    _check_cache.update(at=now, key=key, value=out)
    return out


def _version_tuple(v: str) -> tuple:
    parts = []
    for p in v.split(".")[:3]:
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def install_state() -> dict:
    with _install_lock:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in _install.items()} | {
            "lines": list(_install["lines"][-40:])
        }


def start_install() -> bool:
    """Install (or repair) the environment and fetch the model — in the
    background; :func:`install_state` follows it."""
    with _install_lock:
        if _install["running"]:
            return False
        _install.update(
            running=True, lines=[], ok=None, error=None, started=time.time(), ended=None
        )
    threading.Thread(target=_run_install, name="media-env-install", daemon=True).start()
    return True


def _say(line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    with _install_lock:
        _install["lines"].append(line[:300])
        del _install["lines"][:-400]
    logger.info("Media environment: %s", line[:300])


def _run(cmd: list[str], env: dict | None = None) -> None:
    _say("$ " + " ".join(cmd[:6]) + (" …" if len(cmd) > 6 else ""))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        _say(line)
    if p.wait() != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} ended with exit code {p.returncode}")


def _run_install() -> None:
    try:
        d = env_dir()
        py = python_path()
        uv = _uv()
        flavor = "cuda" if _has_nvidia() else "cpu"
        _say(
            f"Environment {d} — torch for {'CUDA 12.8' if flavor == 'cuda' else 'the CPU'}"
            f"{', through uv' if uv else ', through pip (no uv found)'}"
        )
        d.parent.mkdir(parents=True, exist_ok=True)
        if uv:
            if not py.exists():
                _run([uv, "venv", str(d), "--python", "3.12"])
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(py),
                    "--extra-index-url",
                    TORCH_INDEX[flavor],
                    "--index-strategy",
                    "unsafe-best-match",
                    *TORCH,
                    *PACKAGES,
                ]
            )
        else:
            if not py.exists():
                _run([sys.executable, "-m", "venv", str(d)])
            _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
            _run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "--extra-index-url",
                    TORCH_INDEX[flavor],
                    *TORCH,
                    *PACKAGES,
                ]
            )
        _say(f"Fetching the model {model_id()} …")
        _run(
            [
                str(py),
                "-c",
                f"from huggingface_hub import snapshot_download; "
                f"print(snapshot_download({model_id()!r}))",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_DISABLE_TELEMETRY": "1"},
        )
        state = check(force=True)
        if not state["ok"]:
            raise RuntimeError("; ".join(state["problems"]))
        with _install_lock:
            _install.update(ok=True)
        _say("Ready.")
    except Exception as e:  # noqa: BLE001 — shown on the settings page
        with _install_lock:
            _install.update(ok=False, error=str(e))
        _say(f"Failed: {e}")
    finally:
        with _install_lock:
            _install.update(running=False, ended=time.time())

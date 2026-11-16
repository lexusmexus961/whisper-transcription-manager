"""System check for the setup guide and ``--check``.

Two levels:

* :func:`quick_checks` — cheap, never touches the GPU. Reads what is
  installed and configured: FFmpeg, GPUs (sysfs + NVIDIA tools), the
  PyTorch and CTranslate2 builds, paths, optional features. Safe while a
  transcription runs.
* :func:`gpu_test` — actually computes on the GPU: a PyTorch matrix
  multiply, CTranslate2's device query and a short faster-whisper run with
  the ``tiny`` model. Run in a **subprocess** by the API
  (``python -m whisper_tm.core.diagnostics --gpu-test``) so a crashing
  driver takes down the test, not the server, and nothing stays resident.

The AMD part is **experimental and untested**: the project has never run
on an AMD GPU. Everything AMD-specific here is based
on public reports as of 2026-09 and says so in its output.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from whisper_tm.core import commands

# PCI vendor IDs as sysfs reports them.
_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}

# Radeon targets ROCm lists as supported on Linux (ROCm 7.14 system
# requirements, checked 2026-09). Instinct/Pro cards are left out on purpose:
# nobody runs this project on those by accident.
ROCM_SUPPORTED_RADEON = {
    "gfx1100": "RX 7900 XTX / XT / GRE",
    "gfx1101": "RX 7800 XT / 7700 XT / 7700",
    "gfx1200": "RX 9060 XT / 9060",
    "gfx1201": "RX 9070 XT / 9070 / 9070 GRE",
}

# Result states, in the order the UI sorts them.
OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"


def _check(
    group: str,
    title: str,
    status: str,
    detail: str = "",
    hint: str = "",
    experimental: bool = False,
) -> dict:
    return {
        "group": group,
        "title": title,
        "status": status,
        "detail": detail,
        "hint": hint,
        "experimental": experimental,
    }


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    """Output of a helper tool, or "" if it is missing/fails/hangs."""
    if not shutil.which(cmd[0]):
        return ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


# ── GPUs ─────────────────────────────────────────────────────────────────


def list_gpus() -> list[dict]:
    """Display GPUs from sysfs (Linux), independent of any ML library.

    This is what is physically in the machine — PyTorch may see only a
    subset of it (a CUDA build ignores AMD cards and vice versa), which is
    exactly the mismatch the check wants to point out.
    """
    out: list[dict] = []
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return _nvidia_from_driver()
    seen: set[str] = set()
    for card in sorted(drm.glob("card[0-9]*")):
        if "-" in card.name:  # connectors like card1-DP-1
            continue
        dev = card / "device"
        try:
            slot = dev.resolve().name
            vendor = _VENDORS.get((dev / "vendor").read_text().strip(), "other")
        except OSError:
            continue
        if slot in seen:
            continue
        seen.add(slot)
        vram_mb = 0
        with contextlib.suppress(OSError, ValueError):
            vram_mb = int((dev / "mem_info_vram_total").read_text()) // (1024 * 1024)
        name = ""
        line = _run(["lspci", "-s", slot])
        if line:
            name = line.split(":", 2)[-1].strip()
            name = re.sub(r"\s*\(rev \w+\)$", "", name)
            name = re.sub(
                r"^(Advanced Micro Devices, Inc\. \[AMD/ATI\]|NVIDIA Corporation)\s*", "", name
            )
        out.append({"vendor": vendor, "slot": slot, "name": name, "vram_mb": vram_mb})

    # NVIDIA does not publish VRAM in sysfs — ask the driver.
    if any(g["vendor"] == "nvidia" for g in out):
        with contextlib.suppress(Exception):
            from whisper_tm.core import debug

            nv = [g for g in out if g["vendor"] == "nvidia"]
            for i, info in sorted(debug.gpu_static().items()):
                if i < len(nv):
                    nv[i]["vram_mb"] = nv[i]["vram_mb"] or int(info.get("total_mb") or 0)
                    nv[i]["name"] = info.get("name") or nv[i]["name"]
    if not any(g["vendor"] == "nvidia" for g in out):
        # WSL2 and Docker Desktop's WSL2 backend: the card comes through
        # /dev/dxg without a DRM node — sysfs shows none, the driver does
        out += _nvidia_from_driver()
    return out


def _nvidia_from_driver() -> list[dict]:
    """The NVIDIA cards the driver (NVML or nvidia-smi) reports."""
    with contextlib.suppress(Exception):
        from whisper_tm.core import debug

        return [
            {
                "vendor": "nvidia",
                "slot": f"nvml:{i}",
                "name": info.get("name") or "NVIDIA GPU",
                "vram_mb": int(info.get("total_mb") or 0),
            }
            for i, info in sorted(debug.gpu_static().items())
        ]
    return []


def rocm_info() -> dict:
    """ROCm installation and the gfx targets it reports ("" / [] if none)."""
    version = ""
    for f in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        with contextlib.suppress(OSError):
            version = f.read_text().strip()
            break
    targets = sorted(set(re.findall(r"Name:\s+(gfx[0-9a-f]+)", _run(["rocminfo"], 10))))
    return {
        "installed": bool(version) or Path("/opt/rocm").is_dir() or bool(shutil.which("rocminfo")),
        "version": version,
        "gfx": targets,
        "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
    }


def _torch_info() -> dict:
    try:
        import torch
    except Exception as e:
        return {"installed": False, "error": str(e)}
    hip = getattr(torch.version, "hip", None) or ""
    cuda = torch.version.cuda or ""
    info = {
        "installed": True,
        "version": torch.__version__,
        "backend": "rocm" if hip else ("cuda" if cuda else "cpu"),
        "backend_version": hip or cuda,
        "available": False,
        "devices": [],
    }
    with contextlib.suppress(Exception):
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    return info


def _ct2_info() -> dict:
    try:
        import ctranslate2
    except Exception as e:
        return {"installed": False, "error": str(e)}
    count = 0
    with contextlib.suppress(Exception):
        count = int(ctranslate2.get_cuda_device_count())
    return {"installed": True, "version": ctranslate2.__version__, "gpu_count": count}


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


# ── Quick checks ─────────────────────────────────────────────────────────


def _fmt_gb(mb: int) -> str:
    return f"{mb / 1024:.0f} GB" if mb else "? GB"


def _gpu_checks(gpus: list[dict], torch_i: dict, ct2: dict) -> list[dict]:
    g = "GPU"
    out = []
    # Only NVIDIA and AMD cards compute here: an Intel iGPU or a virtual
    # machine's display adapter is a graphics card all the same, and a
    # correct CPU installation was told to reinstall for NVIDIA
    compute = [x for x in gpus if x["vendor"] in ("nvidia", "amd")]
    listed = "; ".join(
        f"{x['name'] or x['vendor'].upper() + ' ' + x['slot']} ({_fmt_gb(x['vram_mb'])})"
        for x in gpus
    )
    if compute:
        out.append(_check(g, "Graphics cards", INFO, listed))
    else:
        out.append(
            _check(
                g,
                "Graphics cards",
                WARN,
                f"No NVIDIA or AMD card{f' ({listed})' if listed else ''}",
                "Transcription runs on the CPU — many times slower.",
            )
        )

    vendors = {x["vendor"] for x in gpus}
    if not torch_i.get("installed"):
        out.append(
            _check(
                g,
                "PyTorch",
                FAIL,
                torch_i.get("error", "not installed"),
                f"Run {commands.setup()} again.",
            )
        )
    else:
        build = " ".join(x for x in (torch_i["backend"].upper(), torch_i["backend_version"]) if x)
        label = f"{torch_i['version']} ({build})"
        if torch_i["available"]:
            out.append(_check(g, "PyTorch", OK, f"{label} — sees: {', '.join(torch_i['devices'])}"))
        elif torch_i["backend"] == "cuda" and "nvidia" not in vendors and "amd" in vendors:
            out.append(
                _check(
                    g,
                    "PyTorch",
                    FAIL,
                    f"{label} — NVIDIA build, but only an AMD card is present",
                    "An AMD card needs the ROCm build of PyTorch — see the AMD section "
                    "(experimental).",
                )
            )
        elif torch_i["backend"] == "cpu" and compute:
            out.append(
                _check(
                    g,
                    "PyTorch",
                    WARN,
                    f"{label} — CPU-only build",
                    f"Reinstall with {commands.setup('--gpu nvidia')} (or amd, experimental).",
                )
            )
        elif not compute:
            # No card at all: a CPU installation as it should be. The line
            # used to send it looking for a driver ("nvidia-smi should list
            # the card") — for a card the machine doesn't have.
            out.append(_check(g, "PyTorch", OK, f"{label} — computes on the CPU"))
        else:
            out.append(
                _check(
                    g,
                    "PyTorch",
                    WARN,
                    f"{label} — no usable GPU",
                    "The container sees no GPU: it needs the NVIDIA Container Toolkit "
                    "on the host and the GPU section of docker-compose.yaml "
                    "(docker run --rm --gpus all ubuntu nvidia-smi shows the card)."
                    if commands.in_docker()
                    else "Driver loaded? nvidia-smi / rocminfo should list the card.",
                )
            )

    if not ct2.get("installed"):
        out.append(
            _check(g, "CTranslate2 (Whisper engine)", FAIL, ct2.get("error", "not installed"))
        )
    elif ct2["gpu_count"] > 0:
        out.append(
            _check(
                g,
                "CTranslate2 (Whisper engine)",
                OK,
                f"{ct2['version']} — {ct2['gpu_count']} GPU(s) usable",
            )
        )
    else:
        amd_only = "amd" in vendors and "nvidia" not in vendors
        if not compute:
            out.append(
                _check(
                    g,
                    "CTranslate2 (Whisper engine)",
                    INFO,
                    f"{ct2['version']} — Whisper runs on the CPU",
                )
            )
        else:
            out.append(
                _check(
                    g,
                    "CTranslate2 (Whisper engine)",
                    FAIL,
                    f"{ct2['version']} — no GPU usable, Whisper runs on the CPU",
                    (
                        "The pip package of CTranslate2 is built for NVIDIA only. AMD needs "
                        "a build with -DWITH_HIP=ON — see the AMD section (experimental)."
                    )
                    if amd_only
                    else "Check the NVIDIA driver and the CUDA libraries.",
                )
            )
    return out


def _machine_checks() -> list[dict]:
    """What the running server chose for this machine (model, vocal
    isolation, files at once) — nothing on the command line, where no
    hardware profile was made."""
    import whisper_tm.extensions as ext
    from whisper_tm.core import commands
    from whisper_tm.hardware import machine_defaults, machine_defaults_text, unsupported_card

    if getattr(ext, "hw", None) is None:
        return []
    out = []
    card = unsupported_card(ext.hw)
    if card:
        out.append(
            _check(
                "GPU",
                "PyTorch and this card",
                FAIL,
                card.split(": ", 1)[0],
                f"{card.split(': ', 1)[-1]} ({commands.setup()})",
            )
        )
    out.append(
        _check(
            "GPU",
            "Defaults for this machine",
            INFO,
            machine_defaults_text(*machine_defaults(ext.hw)),
        )
    )
    return out


def _amd_checks(gpus: list[dict], torch_i: dict) -> list[dict]:
    """Only when an AMD card is present or PyTorch is a ROCm build.

    With a CUDA build the AMD card is merely present (e.g. next to the
    NVIDIA card that does the work): everything here is then INFO — a
    missing ROCm is no problem for a setup that doesn't use it.
    """
    g = "AMD (experimental)"
    amd = [x for x in gpus if x["vendor"] == "amd"]
    in_use = torch_i.get("backend") == "rocm"
    if not amd and not in_use:
        return []
    out = [
        _check(
            g,
            "Status",
            WARN if in_use else INFO,
            "AMD support is experimental and untested — this project has never "
            "transcribed on an AMD GPU."
            + ("" if in_use else " The AMD card is present but not in use."),
            "Results below are hints, not guarantees. See the AMD section of the guide.",
            experimental=True,
        )
    ]
    r = rocm_info()
    if r["installed"]:
        version = f" — {r['version']}" if r["version"] else ""
        out.append(_check(g, "ROCm", OK, f"installed{version}", experimental=True))
    else:
        out.append(
            _check(
                g,
                "ROCm",
                FAIL if in_use else INFO,
                "not installed",
                "Only needed to transcribe on the AMD card: install ROCm from your "
                "distribution or AMD (rocm.docs.amd.com).",
                experimental=True,
            )
        )
    for gfx in r["gfx"]:
        if gfx.startswith("gfx1") or gfx.startswith("gfx9"):
            if gfx in ROCM_SUPPORTED_RADEON:
                out.append(
                    _check(
                        g,
                        f"Target {gfx}",
                        OK,
                        f"officially supported by ROCm ({ROCM_SUPPORTED_RADEON[gfx]})",
                        experimental=True,
                    )
                )
            else:
                out.append(
                    _check(
                        g,
                        f"Target {gfx}",
                        WARN if in_use else INFO,
                        "not on ROCm's official support list",
                        "Often works with HSA_OVERRIDE_GFX_VERSION=11.0.0 (RDNA3) — but there are "
                        "reports of it breaking after driver updates. Run the GPU test."
                        + (f" Currently set: {r['hsa_override']}" if r["hsa_override"] else ""),
                        experimental=True,
                    )
                )
    if amd and not r["gfx"]:
        out.append(_check(g, "Target", INFO, "unknown (rocminfo not available)", experimental=True))
    out.append(
        _check(
            g,
            "VRAM monitoring",
            INFO,
            "reads NVIDIA tools only — on AMD the VRAM display, the wait for free "
            "VRAM and the ⏏ figures stay empty",
            experimental=True,
        )
    )
    if not any(p.startswith(("ROCM", "MIGraphX")) for p in _onnx_providers()):
        out.append(
            _check(
                g,
                "Vocal isolation (.onnx models)",
                INFO,
                "older MDX models (.onnx) run on the CPU; the default Roformer model uses PyTorch",
                experimental=True,
            )
        )
    return out


def _storage_checks() -> list[dict]:
    from whisper_tm.config import config

    g = "Folders"
    out = []
    media = str(config.media_dir or "")
    if not media:
        out.append(
            _check(
                g, "Media folder", WARN, "not set", "Enter a folder in the path field at the top."
            )
        )
    elif not Path(media).is_dir():
        out.append(
            _check(
                g,
                "Media folder",
                WARN,
                f"{media} — not reachable",
                "Mounted? The path must exist on this machine.",
            )
        )
    else:
        out.append(_check(g, "Media folder", OK, media))
    work = Path(str(config.work_dir or ""))
    if work and work.is_dir() and os.access(work, os.W_OK):
        free = shutil.disk_usage(work).free // (1024**3)
        status = OK if free >= 20 else WARN
        out.append(
            _check(
                g,
                "Work folder",
                status,
                f"{work} — {free} GB free",
                "" if status == OK else "Long files need several GB of temporary space.",
            )
        )
    else:
        out.append(
            _check(
                g, "Work folder", FAIL, f"{work} — missing or not writable", "Settings → Folders."
            )
        )
    return out


def _feature_checks() -> list[dict]:
    from whisper_tm.config import config

    g = "Features"
    out = []
    import importlib.util as iu

    ffmpeg = shutil.which("ffmpeg")
    out.append(
        _check(
            g,
            "FFmpeg",
            OK if ffmpeg and shutil.which("ffprobe") else FAIL,
            ffmpeg or "not found",
            "" if ffmpeg else "Install ffmpeg (includes ffprobe).",
        )
    )
    out.append(
        _check(
            g,
            "Vocal isolation",
            OK if iu.find_spec("audio_separator") else INFO,
            "audio-separator installed"
            if iu.find_spec("audio_separator")
            else "audio-separator not installed — option unavailable",
        )
    )
    # The model is gated on Hugging Face: a token fetches it once, then the
    # cache is enough (core/hf_access)
    from whisper_tm.core import hf_access

    if hf_access.model_cached():
        token = " · Hugging Face token set" if config.hf_token else ""
        out.append(_check(g, "Speaker diarization", OK, f"model downloaded{token}"))
    elif config.hf_token:
        out.append(
            _check(
                g,
                "Speaker diarization",
                OK,
                "Hugging Face token set — the model downloads with the first file with speakers",
            )
        )
    else:
        out.append(
            _check(
                g,
                "Speaker diarization",
                INFO,
                "no Hugging Face token, model not downloaded",
                f"Needed once: accept the conditions at {hf_access.REPO_URL}, create "
                "a Read token, and enter it in the setup guide (Speaker diarization) "
                f"or with {commands.start('--hf-token')}.",
            )
        )
    # Every provider the tasks can use: the saved list, the single endpoint
    # setup.sh --llama writes (llm.endpoint — read as "Standard") and an
    # installed local server. Counting the list alone said "none configured"
    # right after setup.sh --llama.
    from whisper_tm.core import llm_provider

    try:
        providers = [p.name for p in llm_provider.all_items()]
    except Exception:  # noqa: BLE001 — a check, never worth an error
        providers = []
    if commands.in_docker():
        llm_hint = (
            f"Optional: Settings → AI models — a server on the host is {commands.DOCKER_HOST_URL}."
        )
    else:
        llm_hint = f"Optional: {commands.setup('--llama')} or Settings → AI models."
    out.append(
        _check(
            g,
            "Language model (summary, correction, translation)",
            OK if providers else INFO,
            ", ".join(providers) if providers else "none configured",
            "" if providers else llm_hint,
        )
    )
    return out


#: The modules a transcription really loads — ``import whisperx`` alone
#: defines lazy wrappers only, so a broken faster-whisper, pyannote or
#: audio-separator passed every check until the first job
PIPELINE_MODULES = (
    ("Speech recognition (faster-whisper)", "whisperx.asr"),
    ("Speaker diarization (pyannote)", "whisperx.diarize"),
    ("Vocal isolation (audio-separator)", "audio_separator.separator"),
    ("MCP server (fastmcp)", "fastmcp"),
)
_package_result: list[dict] | None = None


def _package_checks() -> list[dict]:
    """Each pipeline module imported in a process of its own (once per
    server run): the server doesn't keep pyannote and the rest in its
    memory just for a check, and a crash in one import stays there."""
    global _package_result
    if _package_result is not None:
        return _package_result
    g = "Packages"
    out = []
    for title, module in PIPELINE_MODULES:
        try:
            r = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True,
                text=True,
                timeout=180,
                encoding="utf-8",
                errors="replace",
            )
            ok, why = r.returncode == 0, (r.stderr or "").strip().splitlines()
        except (OSError, subprocess.TimeoutExpired) as e:
            ok, why = False, [str(e)]
        if ok:
            out.append(_check(g, title, OK, module))
        else:
            last = why[-1] if why else "it did not import"
            status = INFO if module == "fastmcp" else FAIL
            out.append(
                _check(
                    g,
                    title,
                    status,
                    f"{module}: {last[:200]}",
                    "Run ./setup.sh again — it installs what is missing.",
                )
            )
    _package_result = out
    return out


def quick_checks() -> dict:
    """Everything the guide shows without touching the GPU."""
    gpus = list_gpus()
    torch_i = _torch_info()
    ct2 = _ct2_info()
    checks = [
        _check(
            "System",
            "Python",
            OK if sys.version_info >= (3, 11) else FAIL,
            f"{platform.python_version()} on {platform.system()} {platform.release()}",
        ),
        *_feature_checks(),
        *_package_checks(),
        *_gpu_checks(gpus, torch_i, ct2),
        *_machine_checks(),
        *_amd_checks(gpus, torch_i),
        *_storage_checks(),
    ]
    worst = (
        FAIL
        if any(c["status"] == FAIL for c in checks)
        else (WARN if any(c["status"] == WARN for c in checks) else OK)
    )
    return {"checks": checks, "overall": worst, "gpus": gpus, "torch": torch_i, "ctranslate2": ct2}


# ── GPU test (runs in a subprocess) ──────────────────────────────────────


def gpu_test(device_index: int = 0, whisper: bool = True) -> dict:
    """Compute on the GPU for real. Each step reports on its own."""
    steps: list[dict] = []

    def step(name: str, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            steps.append(
                {
                    "step": name,
                    "status": OK,
                    "detail": detail,
                    "secs": round(time.perf_counter() - t0, 2),
                }
            )
        except Exception as e:
            steps.append(
                {
                    "step": name,
                    "status": FAIL,
                    "detail": f"{type(e).__name__}: {e}"[:500],
                    "secs": round(time.perf_counter() - t0, 2),
                }
            )

    def torch_kernel():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch sees no GPU")
        dev = torch.device("cuda", device_index)
        x = torch.randn(2048, 2048, device=dev)
        y = (x @ x).sum().item()  # .item() forces the kernel to finish
        if y != y:  # NaN = the kernel produced garbage
            raise RuntimeError("the GPU returned NaN")
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        return f"{torch.cuda.get_device_name(device_index)} via {backend}"

    def ct2_device():
        import ctranslate2

        types = sorted(ctranslate2.get_supported_compute_types("cuda", device_index))
        return "compute types: " + ", ".join(types)

    def whisper_tiny():
        import ctranslate2
        import numpy as np
        from faster_whisper import WhisperModel

        types = ctranslate2.get_supported_compute_types("cuda", device_index)
        ct = "float16" if "float16" in types else "float32"
        model = WhisperModel("tiny", device="cuda", device_index=device_index, compute_type=ct)
        # Five seconds of quiet noise: enough for the encoder to run on the GPU.
        audio = (np.random.default_rng(0).standard_normal(16000 * 5) * 0.01).astype("float32")
        t0 = time.perf_counter()
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        list(segments)
        return f"tiny model, {ct}, 5 s audio in {time.perf_counter() - t0:.2f} s"

    step("PyTorch computes on the GPU", torch_kernel)
    step("CTranslate2 reaches the GPU", ct2_device)
    if whisper:
        step("faster-whisper transcribes on the GPU", whisper_tiny)
    ok = all(s["status"] == OK for s in steps)
    return {"ok": ok, "steps": steps}


def run_gpu_test_subprocess(device_index: int = 0, timeout: float = 600) -> dict:
    """:func:`gpu_test` in its own process — the server's models stay untouched."""
    cmd = [
        sys.executable,
        "-m",
        "whisper_tm.core.diagnostics",
        "--gpu-test",
        "--device-index",
        str(device_index),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "steps": [], "error": f"no result after {timeout:.0f} s"}
    # The JSON is the last line; libraries may print before it.
    for line in reversed(r.stdout.strip().splitlines()):
        if line.startswith("{"):
            with contextlib.suppress(ValueError):
                return json.loads(line)
    err = (r.stderr or "").strip().splitlines()
    return {
        "ok": False,
        "steps": [],
        "error": f"test process ended with code {r.returncode}" + (f": {err[-1]}" if err else ""),
    }


def print_report(result: dict) -> None:
    """Plain-text version for ``--check``."""
    icon = {OK: "✓", INFO: "·", WARN: "⚠", FAIL: "✗"}
    group = None
    for c in result["checks"]:
        if c["group"] != group:
            group = c["group"]
            print(f"\n{group}")
        print(f"  {icon[c['status']]} {c['title']}: {c['detail']}")
        # Like the setup guide: every row that isn't fine says what to do
        if c["hint"] and c["status"] != OK:
            print(f"      → {c['hint']}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Whisper Transcription Manager GPU test")
    ap.add_argument("--gpu-test", action="store_true")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--no-whisper", action="store_true")
    a = ap.parse_args()
    if a.gpu_test:
        print(json.dumps(gpu_test(a.device_index, whisper=not a.no_whisper)))

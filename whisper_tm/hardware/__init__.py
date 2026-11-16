"""
Hardware detection: GPU (NVIDIA/AMD/Apple/DirectML), CPU, FFmpeg capabilities.

Automatically detects the best available compute platform and configures
all dependent components (WhisperX, audio-separator) accordingly.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import psutil

logger = logging.getLogger(__name__)


DeviceType = Literal["cuda", "rocm", "directml", "mps", "cpu"]


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    driver_version: str = ""
    compute_capability: str = ""
    vendor: str = ""  # "nvidia", "amd", "apple", "intel"


@dataclass
class HardwareProfile:
    """Complete hardware profile of the system."""

    # CPU
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0

    # GPU
    gpus: list[GPUInfo] = field(default_factory=list)
    best_device: DeviceType = "cpu"
    best_gpu_index: int = 0

    # Platform
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    arch: str = ""

    # Capabilities
    has_cuda: bool = False
    has_rocm: bool = False
    has_directml: bool = False
    has_mps: bool = False
    cuda_version: str = ""
    rocm_version: str = ""
    #: The card generations the installed PyTorch has code for ("sm_86", …)
    cuda_archs: list[str] = field(default_factory=list)

    # FFmpeg
    has_ffmpeg: bool = False
    ffmpeg_version: str = ""
    has_nvdec: bool = False
    has_vaapi: bool = False

    # Optional Tools
    has_audio_separator: bool = False
    has_ytdlp: bool = False
    has_fastmcp: bool = False
    fastmcp_version: str = ""

    # Recommended Settings
    recommended_compute_type: str = "float16"
    recommended_sep_chunk_duration: int = 1800
    recommended_batch_size: int = 16

    #: For testing: treat the card as having this much memory (MiB; 0 = its
    #: own). ``hardware.assume_vram_mb`` — the recommendations and the
    #: machine's defaults follow it, the card itself does not shrink.
    assume_vram_mb: int = 0

    def usable_vram_mb(self) -> int:
        """Memory of the card the jobs run on, in MiB — 0 without a GPU that
        computes (CPU, and Apple's MPS: WhisperX has no backend for it)."""
        if self.best_device not in ("cuda", "rocm") or not self.gpus:
            return 0
        if self.assume_vram_mb > 0:
            return int(self.assume_vram_mb)
        idx = min(max(self.best_gpu_index, 0), len(self.gpus) - 1)
        return int(self.gpus[idx].memory_total_mb or 0)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"OS: {self.os_name} {self.os_version} ({self.arch})",
            f"CPU: {self.cpu_name} ({self.cpu_cores}C/{self.cpu_threads}T)",
            f"RAM: {round(self.ram_total_mb / 1024)}GB ({self.ram_available_mb // 1024}GB free)",
        ]
        if self.gpus:
            for gpu in self.gpus:
                mem = (
                    f"{round(gpu.memory_total_mb / 1024)}GB"
                    if gpu.memory_total_mb
                    else "memory unknown"
                )
                lines.append(f"GPU {gpu.index}: {gpu.name} ({mem}) [{gpu.vendor}]")
        lines.append(f"Compute: {self.best_device} (type={self.recommended_compute_type})")
        if self.has_ffmpeg:
            lines.append(
                f"FFmpeg: {self.ffmpeg_version}"
                f"{' +NVDEC' if self.has_nvdec else ''}"
                f"{' +VAAPI' if self.has_vaapi else ''}"
            )
        if self.has_audio_separator:
            lines.append("audio-separator: available")
        if self.has_ytdlp:
            lines.append("yt-dlp: available")
        if self.has_fastmcp:
            lines.append(f"fastmcp: {self.fastmcp_version or 'available'}")
        return "\n".join(lines)


def detect_hardware(
    preferred_device: str = "auto", gpu_index: int = 0, assume_vram_mb: int = 0
) -> HardwareProfile:
    """Perform complete hardware detection.

    Args:
        preferred_device: "auto", "cuda", "rocm", "directml", "mps", "cpu"
        gpu_index: Preferred GPU index
        assume_vram_mb: treat the card as this big (testing; 0 = its own size)
    """
    profile = HardwareProfile()
    try:
        profile.assume_vram_mb = max(0, int(assume_vram_mb or 0))
    except (TypeError, ValueError):
        profile.assume_vram_mb = 0

    # ── Platform Info ─────────────────────────────────────────────
    profile.os_name = platform.system()
    profile.os_version = platform.version()
    profile.arch = platform.machine()
    profile.python_version = platform.python_version()

    # ── CPU Info ──────────────────────────────────────────────────
    profile.cpu_cores = psutil.cpu_count(logical=False) or 1
    profile.cpu_threads = psutil.cpu_count(logical=True) or 1
    mem = psutil.virtual_memory()
    profile.ram_total_mb = mem.total // (1024 * 1024)
    profile.ram_available_mb = mem.available // (1024 * 1024)

    try:
        if profile.os_name == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        profile.cpu_name = line.split(":")[1].strip()
                        break
        elif profile.os_name == "Windows":
            profile.cpu_name = platform.processor()
        elif profile.os_name == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            profile.cpu_name = r.stdout.strip()
    except Exception:
        profile.cpu_name = platform.processor() or "Unknown"

    # ── CUDA Detection ────────────────────────────────────────────
    profile.has_cuda = _detect_cuda(profile)

    # ── ROCm Detection ────────────────────────────────────────────
    profile.has_rocm = _detect_rocm(profile)

    # ── DirectML Detection (Windows) ──────────────────────────────
    if profile.os_name == "Windows":
        profile.has_directml = _detect_directml(profile)

    # ── MPS Detection (macOS) ─────────────────────────────────────
    if profile.os_name == "Darwin":
        profile.has_mps = _detect_mps(profile)

    # ── AMD APU/dGPU Fallback (for display, even without ROCm) ──
    if profile.os_name == "Linux":
        _detect_amd_any(profile)

    # ── Best Device Selection ─────────────────────────────────────
    # Don't accept an unknown preferred_device (e.g. a typo) verbatim —
    # check it against the valid DeviceType values, otherwise fall back
    # to auto-detection (with a warning).
    if preferred_device != "auto" and preferred_device not in get_args(DeviceType):
        logger.warning("Unknown device '%s' — falling back to auto-detection.", preferred_device)
        preferred_device = "auto"

    if preferred_device == "auto":
        if profile.has_cuda:
            profile.best_device = "cuda"
        elif profile.has_rocm:
            profile.best_device = "rocm"
        elif profile.has_mps:
            profile.best_device = "mps"
        elif profile.has_directml:
            profile.best_device = "directml"
        else:
            profile.best_device = "cpu"
    else:
        profile.best_device = preferred_device
        # Validate
        available = {
            "cuda": profile.has_cuda,
            "rocm": profile.has_rocm,
            "mps": profile.has_mps,
            "directml": profile.has_directml,
        }
        if not available.get(preferred_device, True):
            profile.best_device = "cpu"

    # Clamp gpu_index against the number of detected GPUs (0..n-1), so
    # 'cuda:N' doesn't hard-fail out-of-range/negative during model load.
    if profile.gpus:
        profile.best_gpu_index = max(0, min(gpu_index, len(profile.gpus) - 1))
    else:
        profile.best_gpu_index = 0

    # ── Compute Type & Chunk Recommendations ──────────────────────
    _compute_recommendations(profile)

    # ── FFmpeg ────────────────────────────────────────────────────
    _detect_ffmpeg(profile)

    # ── Optional Tools ────────────────────────────────────────────
    _detect_tools(profile)

    return profile


# ── GPU Detection Helpers ─────────────────────────────────────────────────


def _detect_cuda(profile: HardwareProfile) -> bool:
    """Detect NVIDIA CUDA."""
    try:
        import torch

        # On ROCm builds, torch.cuda.is_available() also returns True,
        # even though HIP/AMD is active — then do NOT record it as NVIDIA
        # (that's handled by _detect_rocm, otherwise the same GPUs would
        # be listed twice).
        if hasattr(torch.version, "hip") and torch.version.hip:
            return False
        if not torch.cuda.is_available():
            return False

        profile.cuda_version = torch.version.cuda or ""
        try:
            profile.cuda_archs = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        except Exception:  # noqa: BLE001 — then nothing is said about the build
            profile.cuda_archs = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpu = GPUInfo(
                index=i,
                name=props.name,
                memory_total_mb=total // (1024 * 1024),
                memory_free_mb=free // (1024 * 1024),
                compute_capability=f"{props.major}.{props.minor}",
                vendor="nvidia",
            )
            profile.gpus.append(gpu)
        return len(profile.gpus) > 0
    except Exception:
        # Fallback: nvidia-smi
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpu = GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_total_mb=int(float(parts[2])),
                            memory_free_mb=int(float(parts[3])),
                            driver_version=parts[4],
                            vendor="nvidia",
                        )
                        profile.gpus.append(gpu)
                return len(profile.gpus) > 0
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)
    return False


def _detect_rocm(profile: HardwareProfile) -> bool:
    """Detect AMD ROCm."""
    try:
        # Check ROCm via torch
        import torch

        if hasattr(torch.version, "hip") and torch.version.hip:
            profile.rocm_version = torch.version.hip
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total_mb=props.total_memory // (1024 * 1024),
                    vendor="amd",
                )
                profile.gpus.append(gpu)
            return True
    except Exception as e:
        logger.debug("Unhandled exception: %s", e)

    # Fallback: rocm-smi
    try:
        r = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and "GPU" in r.stdout:
            # Parse basic info
            gpu = GPUInfo(index=0, name="AMD GPU (ROCm)", vendor="amd")
            profile.gpus.append(gpu)
            return True
    except Exception as e:
        logger.debug("Unhandled exception: %s", e)
    return False


def _detect_directml(profile: HardwareProfile) -> bool:
    """Detect Windows DirectML."""
    try:
        import torch_directml

        return torch_directml.is_available()
    except ImportError:
        return False


def _detect_mps(profile: HardwareProfile) -> bool:
    """Detect Apple Metal Performance Shaders."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu = GPUInfo(index=0, name="Apple Silicon GPU", vendor="apple")
            profile.gpus.append(gpu)
            return True
    except Exception as e:
        logger.debug("Unhandled exception: %s", e)
    return False


def _detect_amd_any(profile: HardwareProfile) -> None:
    """Detect AMD GPU/APU via lspci — even without ROCm (display only).

    On AMD APUs (e.g. mobile Ryzen chips), ROCm isn't available,
    but we still want to show the GPU.
    """
    if any(g.vendor == "amd" for g in profile.gpus):
        return
    try:
        r = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                ll = line.lower()
                if ("vga" in ll or "3d" in ll or "display" in ll) and (
                    "amd" in ll or "radeon" in ll or "advanced micro" in ll
                ):
                    name = line.split(": ", 1)[-1].strip() if ": " in line else "AMD GPU/APU"
                    name = re.sub(r"\s*\(rev \w+\)$", "", name)
                    name = re.sub(r"^Advanced Micro Devices, Inc\. \[AMD/ATI\]\s*", "", name)
                    vram = 0
                    slot = line.split(" ", 1)[0]
                    with contextlib.suppress(OSError, ValueError):
                        # amdgpu publishes it in sysfs, whatever ROCm says
                        vram = int(
                            Path(
                                f"/sys/bus/pci/devices/0000:{slot}/mem_info_vram_total"
                            ).read_text()
                        ) // (1024 * 1024)
                    gpu = GPUInfo(
                        index=len(profile.gpus), name=name, vendor="amd", memory_total_mb=vram
                    )
                    profile.gpus.append(gpu)
    except Exception as e:
        logger.debug("Unhandled exception: %s", e)


# ── Recommendations ───────────────────────────────────────────────────────


def _compute_recommendations(profile: HardwareProfile):
    """Compute recommendations based on hardware."""
    if profile.best_device == "cpu":
        profile.recommended_compute_type = "int8"
        profile.recommended_sep_chunk_duration = 600
        profile.recommended_batch_size = 4
        return

    # GPU-based recommendations
    gpu_mem = 0
    if profile.gpus:
        idx = min(profile.best_gpu_index, len(profile.gpus) - 1)
        gpu_mem = profile.assume_vram_mb or profile.gpus[idx].memory_total_mb

    # The bounds lie between the sizes that are sold, as in _CLASSES: a card
    # reports a little less than its name says. They used to stand ON the
    # names (>= 12000, >= 8000) — a 12 GB card reports about 11,900 MiB and
    # got the 8 GB values, an 8 GB card the 4 GB ones.
    if gpu_mem >= 23000:
        # 24 GB and more
        profile.recommended_compute_type = "float16"
        profile.recommended_sep_chunk_duration = 3600
        profile.recommended_batch_size = 32
    elif gpu_mem >= 15000:
        # 16 GB
        profile.recommended_compute_type = "float16"
        profile.recommended_sep_chunk_duration = 1800
        profile.recommended_batch_size = 16
    elif gpu_mem >= 9500:
        # 10, 11 and 12 GB. Batch 8 is what the 12 GB card this project ran
        # on has used for thousands of recordings: large-v3 then holds
        # 5.6-6.8 GB while it transcribes.
        profile.recommended_compute_type = "float16"
        profile.recommended_sep_chunk_duration = 900
        profile.recommended_batch_size = 8
    elif gpu_mem >= 7500:
        # 8 GB: large-v3 with batch 8 ran out of memory with 7 GB free
        profile.recommended_compute_type = "float16"
        profile.recommended_sep_chunk_duration = 900
        profile.recommended_batch_size = 4
    elif gpu_mem >= 5500:
        # 6 GB (the machine's default model is turbo)
        profile.recommended_compute_type = "float16"
        profile.recommended_sep_chunk_duration = 600
        profile.recommended_batch_size = 4
    elif gpu_mem >= 2500:
        # 3 and 4 GB
        profile.recommended_compute_type = "int8"
        profile.recommended_sep_chunk_duration = 600
        profile.recommended_batch_size = 4
    else:
        profile.recommended_compute_type = "int8"
        profile.recommended_sep_chunk_duration = 300
        profile.recommended_batch_size = 2

    # MPS adjustments (Apple Silicon)
    if profile.best_device == "mps":
        profile.recommended_compute_type = "float32"

    # ROCm adjustments
    if profile.best_device == "rocm":
        profile.recommended_compute_type = "float16"


# ── A card the installed PyTorch has no code for ──────────────────────────


def build_misses_card(capability: str, archs: list[str]) -> bool:
    """Has a PyTorch build with these architectures no code for a card of
    this compute capability ("6.1")? Code for X.y runs on a card X.z with
    z >= y — and on no other generation. Unknown values say "no"."""
    try:
        major, minor = (int(x) for x in str(capability).split(".")[:2])
    except ValueError:
        return False
    have = [(int(a[3:-1]), int(a[-1])) for a in archs if a.startswith("sm_") and a[3:].isdigit()]
    if not have:
        return False
    return not any(m == major and n <= minor for m, n in have)


def unsupported_card(profile: HardwareProfile) -> str:
    """Why the card the jobs run on can't compute with the installed
    PyTorch ("" = it can, or nothing is known).

    PyTorch's CUDA 12.8 build has no code for cards older than compute
    capability 7.0 (GTX 10xx, GTX 9xx). Such a card is found and Whisper
    runs on it, but alignment, diarization, separation and the voice
    vectors stop with "no kernel image is available" — in the middle of the
    first job. setup.sh installs the CUDA 12.6 build for it.
    """
    if profile.best_device != "cuda" or not profile.gpus or not profile.cuda_archs:
        return ""
    gpu = profile.gpus[min(max(profile.best_gpu_index, 0), len(profile.gpus) - 1)]
    cap = gpu.compute_capability
    if not cap or not build_misses_card(cap, profile.cuda_archs):
        return ""
    newer = max(int(a[3:-1]) for a in profile.cuda_archs) < int(cap.split(".")[0])
    return (
        f"{gpu.name} (compute capability {cap}) is "
        f"{'newer' if newer else 'older'} than the installed PyTorch has code for "
        f"({', '.join(profile.cuda_archs)}): alignment, diarization, vocal isolation and "
        f'voice vectors would stop with "no kernel image is available". '
        + (
            "Update the install: run the setup again."
            if newer
            else "Run the setup again: it installs PyTorch's CUDA 12.6 build, the last one with "
            "code for this card."
        )
    )


# ── This machine's defaults ───────────────────────────────────────────────

#: The classes of machine, smallest first: (below this many MiB, model,
#: vocal isolation, voice vectors of recordings without vocal isolation,
#: files at once, what it is called). Cards report a little less than their
#: name says (a "12 GB" card about 12,000 MiB), so the bounds lie between
#: the sizes that are sold.
#:
#: Measured on the 12 GB card taken to be smaller (hardware.assume_vram_mb,
#: the rest of its memory held by another process), an 11-minute recording
#: with diarization: as an 8 GB card large-v3 with vocal isolation peaked
#: at 5.6 GB; as a 6 GB card turbo, with the 30-second windows separated for
#: the voice vectors, at 3.8 GB; as a 4 GB card that window separation ran
#: out of memory. Up to 11 GB the models are loaded for each step
#: (hardware.low_vram "auto"), and such an entry runs alone — more lanes
#: would only hold more recordings in RAM.
_CLASSES = (
    (5500, "turbo", False, False, 1, "a card with up to 4 GB"),
    (7500, "turbo", False, True, 1, "a 6 GB card"),
    (9500, "large-v3", True, True, 1, "an 8 GB card"),
    (11500, "large-v3", True, True, 1, "a 10 or 11 GB card"),
)
#: Without a GPU: on 16 cores an 11-minute recording took 2.2 min to
#: transcribe (turbo) and 4.4 min to diarize — and the window separation for
#: its voice vectors was not through after another 11
_CPU = ("turbo", False, False, 1, "no GPU, the CPU computes")


def machine_defaults(profile: HardwareProfile) -> tuple[dict, str]:
    """The settings that follow the machine, and what the machine is.

    ``({"section.key": value}, "an 8 GB card")`` for config.set_machine_defaults.
    A card of 12 GB and more gets the shipped values (large-v3, vocal
    isolation, three files at once); a machine without a GPU that computes
    gets what a CPU manages — large-v3 and the separation model take many
    times a recording's length there.
    """
    vram = profile.usable_vram_mb()
    if vram <= 0:
        found = _CPU
    else:
        found = next((rest for below, *rest in _CLASSES if vram < below), None)
        if found is None:
            return {}, "a card with 12 GB or more"
    model, isolation, windows, lanes, what = found
    return (
        {
            "transcription.default_model": model,
            "vocal_isolation.default_enabled": isolation,
            "transcription.speaker_window_separation": windows,
            "queue.lanes": lanes,
        },
        what,
    )


def apply_machine_defaults(profile: HardwareProfile | None = None) -> tuple[HardwareProfile, str]:
    """Detect the hardware (unless given) and make its defaults the config's.
    For the server's start and for the commands that go by those defaults —
    ``--prefetch`` downloaded large-v3 on a machine that then ran turbo.
    Returns the profile and the sentence that says what was chosen."""
    from whisper_tm.config import config

    if profile is None:
        profile = detect_hardware(
            preferred_device=config.get("hardware", "device", default="auto"),
            gpu_index=config.get("hardware", "gpu_index", default=0),
            assume_vram_mb=config.get("hardware", "assume_vram_mb", default=0),
        )
    values, what = machine_defaults(profile)
    config.set_machine_defaults(values)
    return profile, machine_defaults_text(values, what)


def machine_defaults_text(values: dict, what: str) -> str:
    """One sentence for the log, the system check and the settings page."""
    if not values:
        return f"This machine ({what}) runs with the standard values."
    model = values.get("transcription.default_model")
    lanes = values.get("queue.lanes")
    parts = [
        f"Whisper {model}",
        "vocal isolation on"
        if values.get("vocal_isolation.default_enabled")
        else "no vocal isolation",
        "one file at a time" if lanes == 1 else f"{lanes} files at once",
    ]
    if not values.get("vocal_isolation.default_enabled") and not values.get(
        "transcription.speaker_window_separation"
    ):
        # Both off: recordings get no voice vectors (config.default.yaml:
        # speaker_window_separation) — people are not found again across them
        parts.insert(2, "no voice vectors")
    return f"Chosen for this machine ({what}): {', '.join(parts)}. Each can be changed in Settings."


def _detect_ffmpeg(profile: HardwareProfile):
    """Detect FFmpeg installation and hardware features."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            profile.has_ffmpeg = True
            first_line = r.stdout.split("\n")[0]
            tokens = first_line.split(" ")
            profile.ffmpeg_version = tokens[2] if len(tokens) > 2 else "?"

        # Hardware Decoder
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10
        )
        accel = r.stdout.lower()
        profile.has_nvdec = "cuda" in accel or "nvdec" in accel
        profile.has_vaapi = "vaapi" in accel
    except Exception as e:
        logger.debug("Unhandled exception: %s", e)


def _detect_tools(profile: HardwareProfile):
    """Detect optional tools."""
    # audio-separator
    #
    # Check the PYTHON PACKAGE first, because that's exactly what the code
    # uses: `vocal_isolation.py:337` does `from audio_separator.separator
    # import Separator`. The earlier check only invoked the COMMAND
    # `audio-separator` — and that lives in `.venv/bin/`, so it's only on
    # PATH if the virtual environment was activated. If the server is
    # started via the interpreter path (`./.venv/bin/python -m
    # whisper_tm`), it isn't: detection then reported "not installed",
    # even though the import worked flawlessly. The run aborted with
    # "audio-separator not installed!" while the same library separated
    # without issue in the tools under `tools/`.
    #
    # Detection and usage must ask the same question.
    try:
        import importlib.util

        profile.has_audio_separator = importlib.util.find_spec("audio_separator") is not None
    except Exception as e:
        logger.debug("audio-separator (package check): %s", e)
    if not profile.has_audio_separator:
        # Fall back to the command — for installations that provide it
        # without an importable package.
        try:
            r = subprocess.run(
                ["audio-separator", "--help"], capture_output=True, text=True, timeout=10
            )
            profile.has_audio_separator = r.returncode == 0
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)

    # yt-dlp: the package in venv is enough (core/ytdl.py calls it via python -m);
    # the binary on PATH is only the fallback for system installations.
    try:
        from whisper_tm.core.ytdl import available as _ytdlp_present

        profile.has_ytdlp = _ytdlp_present()
    except Exception as e:
        logger.debug("yt-dlp not found: %s", e)

    # fastmcp — only needed for `python -m whisper_tm.mcp_server`; the app
    # itself runs without it. Therefore do NOT import it, just check for it:
    # fastmcp pulls in half a framework on import, and this detection
    # runs on every startup.
    try:
        if importlib.util.find_spec("fastmcp") is not None:
            profile.has_fastmcp = True
            try:
                from importlib.metadata import version

                profile.fastmcp_version = version("fastmcp")
            except Exception as e:
                logger.debug("fastmcp version unknown: %s", e)
    except Exception as e:
        logger.debug("fastmcp not found: %s", e)


def get_whisperx_device(profile: HardwareProfile) -> str:
    """Device string for WhisperX/ctranslate2 (accepts only 'cuda', 'cpu', 'auto').

    ctranslate2 does NOT understand 'cuda:0' — only 'cuda' or 'cpu'.
    """
    if profile.best_device == "cuda":
        return "cuda"
    if profile.best_device == "rocm":
        # PyTorch's ROCm build runs alignment, diarization and separation on
        # the AMD card, but the stock CTranslate2 wheel knows NVIDIA only:
        # Whisper on "cuda" then failed every file. Only a CTranslate2 built
        # with HIP sees the card (the setup guide's AMD page).
        if _ct2_gpu_count() > 0:
            return "cuda"
        logger.warning(
            "CTranslate2 sees no GPU (it needs a build with HIP for AMD) — "
            "Whisper runs on the CPU, the other steps on the AMD card"
        )
        return "cpu"
    return "cpu"


def _ct2_gpu_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 — no CTranslate2, no GPU for it
        return 0

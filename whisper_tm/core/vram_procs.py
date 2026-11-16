"""VRAM per process — whose memory is it?

The performance view showed VRAM as one number: the driver's total over
every process on the card. This module shows which process uses the
VRAM, so that foreign sources show too — a transcription that
runs out of memory at 11 of 12 GB can't tell from that total whether it
filled the card itself or a game, a browser or another AI tool did.

**Where the numbers come from.** The driver reports memory per process:
through NVML in-process where ``nvidia-ml-py`` is installed (compute AND
graphics processes), otherwise through ``nvidia-smi -q -d PIDS`` — 20 ms,
and unlike ``--query-compute-apps`` it lists graphics-only processes too:
a game or a desktop compositor would be missing from the compute list.

**Which GPU.** Only the one WhisperX actually runs on: the card on which
this app's processes hold memory, and while they hold none, the configured
one — the card the transcription will load onto. The driver asked is
NVIDIA's, so a second card of another make (e.g. an AMD card that drives
the desktop) is never counted. A
desktop compositor or a browser that shows up anyway holds memory on
*this* card: the driver reports per card.

**Own or foreign.** Own is this server and every process it started — the
diarization worker, the embedding server, anything below them in the
process tree. The known ones get their role; the rest go by program name.
What the driver holds for itself and attributes to no process is the
difference to the card's total: ``unattributed_mb``.

**No command lines leave this module.** The server listens on the whole
network, and another program's command line can carry a secret
(``llama-server --api-key …``). The program's name is derived from it here
and only the name is kept.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: How the roles of this app's own processes are shown.
ROLE_LABELS = {
    "server": "whisper-tm server",
    "diarization": "Diarization worker",
    "embedding": "Embedding server",
    "reranker": "Reranker",
}

_lock = threading.Lock()
#: (pid, start time) → description. A pid alone could be a new process by
#: now; the start time makes the key unique.
_described: dict[tuple[int, float], dict] = {}
_bus_to_index: dict[str, int] | None = None


# ── Reading the driver ───────────────────────────────────────────────────


def _smi_report() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True, text=True, timeout=5
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def parse_smi(report: str) -> dict[str, list[dict]]:
    """``nvidia-smi -q -d PIDS`` → ``{bus id: [{pid, type, name, mb}]}``.

    ``mb`` is ``None`` where the driver says "Not available" (a process in
    another container, some driver setups)."""
    gpus: dict[str, list[dict]] = {}
    current: list[dict] | None = None
    proc: dict | None = None
    for line in report.splitlines():
        if line.startswith("GPU ") and ":" in line:
            current = gpus.setdefault(line[4:].strip().lower(), [])
            proc = None
            continue
        if current is None or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Process ID":
            try:
                proc = {"pid": int(value), "type": "", "name": "", "mb": None}
            except ValueError:
                proc = None
                continue
            current.append(proc)
        elif proc is None:
            continue
        elif key == "Type":
            proc["type"] = value
        elif key == "Name":
            proc["name"] = value
        elif key == "Used GPU Memory":
            number = value.split()[0] if value else ""
            try:
                proc["mb"] = int(float(number))
            except ValueError:
                proc["mb"] = None
    return gpus


def _bus_index() -> dict[str, int]:
    """Bus id → the index nvidia-smi and the config count GPUs by (static)."""
    global _bus_to_index
    if _bus_to_index is None:
        from whisper_tm.core import debug

        out = {}
        for parts in debug._smi_query("index,pci.bus_id"):
            if len(parts) >= 2:
                try:
                    out[parts[1].strip().lower()] = int(parts[0])
                except ValueError:
                    pass
        _bus_to_index = out
    return _bus_to_index


def _nvml_processes(gpu_index: int) -> list[dict] | None:
    from whisper_tm.core import debug

    nvml = debug._pynvml
    try:
        h = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        merged: dict[int, dict] = {}
        for kind, read in (
            ("C", nvml.nvmlDeviceGetComputeRunningProcesses),
            ("G", nvml.nvmlDeviceGetGraphicsRunningProcesses),
        ):
            for p in read(h):
                used = getattr(p, "usedGpuMemory", None)
                mb = int(used) // (1024 * 1024) if isinstance(used, int) else None
                have = merged.get(p.pid)
                if have is None:
                    merged[p.pid] = {"pid": int(p.pid), "type": kind, "name": "", "mb": mb}
                else:
                    have["type"] = "C+G"
                    if mb is not None:
                        have["mb"] = max(have["mb"] or 0, mb)
        return list(merged.values())
    except Exception as e:  # noqa — fall back to nvidia-smi
        logger.debug("NVML processes: %s", e)
        return None


def gpu_lists() -> dict[int, list[dict]] | None:
    """The driver's process list of every NVIDIA card, by the index
    nvidia-smi and the config count them — ``None`` if it can't be read."""
    from whisper_tm.core import debug

    backend = debug._init_gpu_backend()
    if backend == "none":
        return None
    if backend == "pynvml":
        try:
            count = debug._pynvml.nvmlDeviceGetCount()
            lists = {i: _nvml_processes(i) for i in range(count)}
            if all(v is not None for v in lists.values()):
                return lists
        except Exception as e:  # noqa — nvidia-smi then
            logger.debug("NVML device count: %s", e)
    report = _smi_report()
    if not report:
        return None
    gpus = parse_smi(report)
    bus = _bus_index()
    if not bus and len(gpus) == 1:
        # Without the index table (a single card): the only GPU there is
        return {0: next(iter(gpus.values()))}
    return {bus[b]: procs for b, procs in gpus.items() if b in bus}


def raw_processes(gpu_index: int) -> list[dict] | None:
    """The driver's list for one GPU, ``None`` if it can't be read."""
    lists = gpu_lists()
    return None if lists is None else lists.get(gpu_index, [])


# ── Who is who ───────────────────────────────────────────────────────────


def _role_pids() -> dict[int, str]:
    """The processes this app knows by role."""
    roles = {os.getpid(): "server"}
    try:
        from whisper_tm.core import diar_process

        for pid in diar_process.worker_pids():
            roles[pid] = "diarization"
    except Exception:  # noqa
        pass
    try:
        from whisper_tm.core import embed_server

        pid = embed_server.SERVER.pid()
        if pid:
            roles[pid] = "embedding"
    except Exception:  # noqa
        pass
    try:
        from whisper_tm.core import rerank_server

        pid = rerank_server.SERVER.pid()
        if pid:
            roles[pid] = "reranker"
    except Exception:  # noqa
        pass
    return roles


def _program(name: str, cmdline: list[str]) -> str:
    """A process's program, as a person would name it.

    "python" says nothing: then the module (``-m``) or the script counts."""
    base = name or (Path(cmdline[0]).name if cmdline else "")
    if base.startswith("python") and cmdline:
        args = cmdline[1:]
        if "-m" in args and args.index("-m") + 1 < len(args):
            return f"python -m {args[args.index('-m') + 1]}"
        for a in args:
            if a.endswith(".py"):
                return f"python {Path(a).name}"
    return base


def describe(pid: int, smi_name: str = "", roles: dict[int, str] | None = None) -> dict:
    """What a process is: program, user, own or foreign, role."""
    roles = _role_pids() if roles is None else roles
    try:
        import psutil

        p = psutil.Process(pid)
        started = p.create_time()
    except Exception:
        # Gone, or in another PID namespace (a container): the driver's
        # name is all there is.
        words = smi_name.split()
        return {
            "program": _program(Path(words[0]).name if words else "", words) or f"PID {pid}",
            "user": None,
            "own": False,
            "role": None,
        }
    key = (pid, started)
    with _lock:
        hit = _described.get(key)
    if hit is not None and (hit["role"] or not roles.get(pid)):
        return hit
    try:
        cmdline = p.cmdline()
    except Exception:
        cmdline = smi_name.split() if smi_name else []
    try:
        name = p.name()
    except Exception:
        name = ""
    try:
        user = p.username()
    except Exception:
        user = None
    me = os.getpid()
    try:
        own = pid == me or any(a.pid == me for a in p.parents())
    except Exception:
        own = pid == me
    d = {
        "program": _program(name, cmdline) or f"PID {pid}",
        "user": user,
        "own": bool(own),
        "role": roles.get(pid),
    }
    with _lock:
        if len(_described) > 500:
            _described.clear()
        _described[key] = d
    return d


def resident_models() -> list[str]:
    """What the server process holds on the GPU by name — the ASR models of
    the transcription and the vocal-isolation separator."""
    out: list[str] = []
    try:
        from whisper_tm.core import transcription, vocal_isolation

        # The registry's keys are tuples: ("large-v3", "cuda", "float16")
        # reads as "large-v3 (cuda, float16)"
        out = sorted(
            f"{k[0]} ({', '.join(str(x) for x in k[1:])})"
            if isinstance(k, tuple) and len(k) > 1
            else str(k[0] if isinstance(k, tuple) and k else k)
            for k in transcription._models.keys()
        )
        if vocal_isolation._separator is not None:
            key = vocal_isolation._separator_key
            out.append(f"Separator ({key[0]})" if key else "Separator")
    except Exception as e:  # noqa
        logger.debug("Resident models: %s", e)
    return out


def _entry(r: dict, roles: dict[int, str]) -> dict:
    d = describe(r["pid"], r.get("name", ""), roles)
    if d["role"]:
        key, label = f"own:{d['role']}", ROLE_LABELS.get(d["role"], d["role"])
    elif d["own"]:
        key, label = f"own:{d['program']}", d["program"]
    else:
        key, label = f"foreign:{d['program']}", d["program"]
    return {
        "pid": r["pid"],
        "mb": r.get("mb"),
        "type": r.get("type", ""),
        **d,
        "key": key,
        "label": label,
    }


def snapshot(configured: int) -> tuple[int, list[dict] | None]:
    """``(gpu index, processes)`` of the GPU WhisperX runs on.

    That is the card with the most memory held by this app's processes;
    while they hold none, ``configured`` — ``hardware.gpu_index``, the card
    the transcription loads onto. ``processes`` is ``None`` if the driver
    can't be asked; each entry as in :func:`processes`, largest first.
    """
    lists = gpu_lists()
    if lists is None:
        return configured, None
    roles = _role_pids()
    described = {i: [_entry(r, roles) for r in procs] for i, procs in lists.items()}
    own = {i: sum(p["mb"] or 0 for p in procs if p["own"]) for i, procs in described.items()}
    idx = max(own, key=lambda i: (own[i], i == configured)) if own else configured
    if not own or own[idx] <= 0:
        idx = configured
    procs = sorted(described.get(idx, []), key=lambda p: -(p["mb"] or 0))
    return idx, procs


def processes(gpu_index: int) -> list[dict] | None:
    """Every process on the GPU with its VRAM, largest first — ``None`` if
    the driver can't be asked.

    Each: ``pid``, ``mb``, ``type`` (C compute, G graphics), ``program``,
    ``user``, ``own``, ``role``, and ``key``/``label``: how the
    history groups it — by role for this app's processes, by program for
    the rest (three renderer processes of one browser are one source).
    """
    raw = raw_processes(gpu_index)
    if raw is None:
        return None
    roles = _role_pids()
    out = [_entry(r, roles) for r in raw]
    out.sort(key=lambda p: -(p["mb"] or 0))
    return out

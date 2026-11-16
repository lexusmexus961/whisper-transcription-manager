"""Index runs of "Images & sounds": which recordings, which moments, the GPU.

**What is indexed.** Every transcribed recording in scope
(``media_search.folders``, empty = the whole library), newest first. A
video is cut into moments every ``video_interval_sec`` seconds; a moment
has a still, a clip around it, the sound around it and what is said there.
An audio file is cut into sound windows of ``sound_window_sec``. Which of
those become vectors is ``media_search.kinds`` — all of them by default.

**Kinds added later cost only themselves.** A file remembers which vectors
its moments have (``files.kinds``: ``image``, ``video@4x10`` — clips of 4
frames over 10 s —, ``videoaudio@4x10``, ``audio``, ``speech``). A kind
switched on later, or clips with other frames, are added to the moments
that are there (a "fill"); only what the moments themselves depend on —
the model, the interval, the sound window, the repeat threshold, the
stored size — makes a file start over. A kind switched off stays in the
index until "Rebuild everything".

**Repeats.** A moment whose still looks like the last kept one (cosine ≥
``dedupe_similarity``) — a webcam stream over hours — gets no still and no
clip of its own; its sound still counts. The run goes on where it stopped:
a file keeps its position (``next_t``) and its last kept still.

**Clip frames.** ``clip_frames`` 0 = automatic: 8 frames per clip when the
videos in scope add up to at most ``AUTO_FRAMES_HOURS`` hours, 4 above —
decided at the first run and kept (``meta.clip_frames``), so a growing
library doesn't make every clip again. Measured on a 12 GB card, 4 frames
ran at 5.5 clips/s with cosine 0.98 to 16 frames, 8 frames at half the
speed with 0.99.

**The GPU.** An index run is a guest on the card the queue works on. It
takes the stage ``media`` of core/gpu_gate.GATE (only when NVML shows
``vram_need_mb`` free while other stages run; never next to Whisper, which
runs alone), keeps it for a turn of several chunks, and gives it back —
releasing the model in the service first — when someone waits at the gate
and the turn lasted ``yield_after_sec``. With ``index_while_queue_runs``
off it waits until no transcription runs at all. Run outside the gate, an
index pass can push the queue's separator into an out-of-memory retry;
that is what the gate is for.

**Several recordings at once.** ``parallel_files`` recordings go at the
same time, each its chunks in order (a file's repeats depend on its last
kept still). With one, the card waited for ffmpeg most of the time — the
stills, clips and sounds of a chunk are decoded before it reaches the
model; with more, other files are decoded while one is on the card
(measured on a 12 GB card: busy ~35 % with one, ~63 % with two, ~76 %
with three, no better with four). The model runs one request at a time
either way, so it takes no more VRAM. A turn is given back only when no
request is at the service.

**The disk.** The index waits while the disk it is written to has less
than ``min_free_gb`` free (20 GB, the free space Diagnostics asks of the
work folder: the queue needs it for long recordings). Thumbnails are
written up to ``thumbs_max_mb``; above it a still is cut from the
recording when it is first shown.

**The estimate** — what the scope holds, what is left, the GPU time and the
disk it takes — reads the transcripts' durations (the columns before the
transcript JSON: 0.5 s on a large library, where reading past it took
20 s) and counts moments without making them. It is kept and computed in
the background; the settings page never waits for it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path

from whisper_tm.core import paths

logger = logging.getLogger(__name__)

VIDEO_EXT = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".flv",
    ".wmv",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ogv",
    ".vob",
}
#: The kinds an index run makes (``description`` is the language model's)
ALL_KINDS = ("image", "video", "videoaudio", "audio", "speech")
VISUAL = ("image", "video", "videoaudio")
CLIPS = ("video", "videoaudio")
AUTO_FRAMES_HOURS = 500.0
#: GPU throughput measured on a 12 GB card next to the running
#: queue (items per second) — what the estimate divides by
RATE = {"image": 13.0, "video4": 5.5, "videoaudio4": 4.5, "audio": 36.0, "speech": 200.0}
#: Mean size of a thumbnail (480 px wide, measured on a sample of stills)
THUMB_BYTES = 22_000
STAGE = "media"
#: Seconds between two looks whether the queue still runs (``index_while_queue_runs`` off)
QUEUE_WAIT_SEC = 10.0
#: Pause after the service ran out of GPU memory, before the same chunk again (smaller)
OOM_PAUSE_SEC = 30.0
#: Pause after giving the card back, so the waiting stage gets in first
YIELD_PAUSE_SEC = 2.0
#: Seconds between two looks at the disk while it is too full
DISK_WAIT_SEC = 60.0
#: Files in a row the service failed on before a run gives up
FAILS_IN_A_ROW = 5
#: How long the list of recordings in scope is kept
SCOPE_TTL_SEC = 300.0
#: How long an estimate is kept — while a run goes, and otherwise
ESTIMATE_TTL_RUNNING_SEC = 8.0
ESTIMATE_TTL_SEC = 120.0
GB = 1024**3


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("media_search", name, default=default)


def _num(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(_cfg(name, default))
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


def settings() -> dict:
    from whisper_tm.core.helpers import parse_bool
    from whisper_tm.media.store import DIMS

    kinds = _cfg("kinds", list(ALL_KINDS))
    if kinds is None:
        kinds = list(ALL_KINDS)
    if isinstance(kinds, str):
        kinds = [k.strip() for k in kinds.split(",")]
    kinds = [k for k in ALL_KINDS if k in set(kinds)]
    dims = int(_num("dims", 768, 128, 768))
    if dims not in DIMS:
        dims = min(DIMS, key=lambda d: abs(d - dims))
    folders = _cfg("folders", []) or []
    if isinstance(folders, str):
        folders = [f for f in folders.splitlines() if f.strip()]
    return {
        "kinds": kinds,
        "video_interval_sec": _num("video_interval_sec", 20, 2, 3600),
        "sound_window_sec": _num("sound_window_sec", 10, 2, 120),
        "clip_sec": _num("clip_sec", 10, 2, 60),
        "clip_frames": int(_num("clip_frames", 0, 0, 32)),
        "dedupe_similarity": _num("dedupe_similarity", 0.97, 0, 1),
        "dims": dims,
        # In the stored form (C:/… on Windows) — a folder that isn't absolute
        # stays as written and matches nothing, as before
        "folders": [
            paths.canon(Path(f).expanduser()) or str(Path(f).expanduser()).rstrip("/")
            for f in folders
            if str(f).strip()
        ],
        "index_device": "cpu" if str(_cfg("index_device", "cuda")).lower() == "cpu" else "cuda",
        "index_while_queue_runs": parse_bool(_cfg("index_while_queue_runs", True)),
        "vram_need_mb": int(_num("vram_need_mb", 2500, 0, 48000)),
        "yield_after_sec": _num("yield_after_sec", 20, 0, 3600),
        "batch": int(_num("batch", 4, 1, 64)),
        "grab_workers": int(_num("grab_workers", 4, 1, 32)),
        "parallel_files": int(_num("parallel_files", 3, 1, 8)),
        "chunk_moments": int(_num("chunk_moments", 16, 1, 512)),
        "thumb_width": int(_num("thumb_width", 480, 160, 1280)),
        "thumbs_max_mb": int(_num("thumbs_max_mb", 4000, 0, 10_000_000)),
        "min_free_gb": _num("min_free_gb", 20, 0, 100_000),
    }


def frames_for(s: dict, video_hours: float, kept: int | None = None) -> int:
    """Frames per clip: the setting, else what the index decided at its
    first run (``kept``), else by the hours of video in scope."""
    if s["clip_frames"] > 0:
        return s["clip_frames"]
    if kept:
        return int(kept)
    return 8 if video_hours <= AUTO_FRAMES_HOURS else 4


def layout(s: dict, store) -> tuple[int, str]:
    """``(numbers per vector, format)`` new vectors are written with: the
    stored ones once the index has some, else the settings'."""
    from whisper_tm.media.store import FORMAT

    d = store.dims()
    return (d, store.format()) if d else (s["dims"], FORMAT)


def fingerprint(s: dict, lay: tuple[int, str]) -> str:
    """What a file's moments depend on — another value makes it start over
    (the kinds are kept per file: see :func:`kind_specs`)."""
    from whisper_tm.media import environment

    key = {
        "model": environment.model_id(),
        "interval": s["video_interval_sec"],
        "window": s["sound_window_sec"],
        "dedupe": s["dedupe_similarity"],
        "dims": lay[0],
        "format": lay[1],
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def kind_specs(s: dict, frames: int, video: bool) -> list[str]:
    """The vectors a file's moments should have, with what clips depend on:
    ``["image", "video@4x10", "videoaudio@4x10", "audio", "speech"]``."""
    out = []
    for k in s["kinds"]:
        if k in VISUAL and not video:
            continue
        out.append(f"{k}@{frames}x{s['clip_sec']:g}" if k in CLIPS else k)
    return out


def spec_kind(spec: str) -> str:
    return spec.split("@", 1)[0]


def plan_file(state: dict | None, base: str, want: list[str]) -> tuple[str, list[str]]:
    """What a file needs: ``("done"|"full"|"continue"|"fill", specs)`` —
    ``specs`` the vectors that pass makes."""
    if state is None or state["settings"] != base:
        return "full", want
    have = set(state["kinds"])
    if state["status"] != "done":
        # a pass under way goes on with what it started with; started with
        # other kinds, it starts over
        return ("continue", sorted(have)) if have == set(want) else ("full", want)
    missing = [sp for sp in want if sp not in have]
    return ("fill", missing) if missing else ("done", [])


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT


def thumbs_dir() -> Path:
    return cache_dir() / "thumbs"


def thumb_path(path: str, t: float) -> str:
    """Where a moment's still goes: a folder per recording, named by its
    path — not by its row id, which SQLite gives again after a delete (a
    recording would show another one's stills)."""
    h = hashlib.sha1(path.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return str(thumbs_dir() / h[:2] / h / f"{int(round(t * 10))}.jpg")


def cache_dir() -> Path:
    from whisper_tm.config import config

    p = config.expand_setting(_cfg("cache_dir", ""), "media_search.cache_dir")
    if p:
        return p
    return Path(config.cache_dir) / "media"


def free_gb(path: Path) -> float | None:
    """Free space of the disk ``path`` is (or would be) on."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / GB
    except OSError:
        return None


def disk_free_gb() -> float | None:
    """The least free space of the disks the index and its thumbnails are
    written to."""
    from whisper_tm.media.store import store_path

    vals = [v for v in (free_gb(store_path().parent), free_gb(thumbs_dir())) if v is not None]
    return min(vals) if vals else None


def thumbs_bytes(store) -> int:
    """What the thumbnails take — counted once, then moved by the runs."""
    have = store.meta("thumb_bytes")
    if have is not None:
        return int(have)
    total = 0
    root = thumbs_dir()
    if root.is_dir():
        for dirpath, _dirs, names in os.walk(root):
            for n in names:
                try:
                    total += os.path.getsize(os.path.join(dirpath, n))
                except OSError:
                    pass
    store.set_meta("thumb_bytes", total)
    return total


# ── Scope ──────────────────────────────────────────────────────────────────

_scope: dict = {"key": None, "at": 0.0, "rows": []}
_scope_lock = threading.Lock()


def scope_files(s: dict, fresh: bool = False) -> list[dict]:
    """The transcribed recordings in scope, newest first:
    ``[{"path", "duration", "video"}]`` — kept ``SCOPE_TTL_SEC``.

    Only columns before ``json_data`` are read: a column after it makes
    SQLite walk every transcript's JSON (20 s instead of 0.5 s)."""
    from whisper_tm.extensions import db

    key = tuple(s["folders"])
    with _scope_lock:
        if not fresh and _scope["key"] == key and time.monotonic() - _scope["at"] < SCOPE_TTL_SEC:
            return _scope["rows"]
    if not db:
        return []
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, abs_path, duration_sec FROM transcripts "
            "WHERE complete=1 AND abs_path IS NOT NULL AND abs_path != ''"
        ).fetchall()
    folders = [f + "/" for f in s["folders"]]
    seen: dict[str, dict] = {}
    for tid, path, dur in rows:
        if folders and not any(path.startswith(f) or path == f[:-1] for f in folders):
            continue
        if not dur or dur <= 0:
            continue
        cur = seen.get(path)
        if cur is None or tid > cur["id"]:
            seen[path] = {"id": tid, "path": path, "duration": float(dur), "video": is_video(path)}
    out = sorted(seen.values(), key=lambda r: -r["id"])
    with _scope_lock:
        _scope.update(key=key, at=time.monotonic(), rows=out)
    return out


def plan_moments(duration: float, video: bool, s: dict, frm: float = 0.0) -> list[dict]:
    """The moments of a recording from ``frm`` seconds: ``[{"t", "start",
    "dur"}]`` — ``t`` the still, ``start``/``dur`` the sound window. ``frm``
    is a moment's ``t`` (what a pass stores as ``next_t``), for sounds too."""
    if duration <= 0:
        return []
    win = s["sound_window_sec"]
    out = []
    if video:
        step = s["video_interval_sec"]
        n = max(1, math.ceil(duration / step))
        for k in range(n):
            t = min(duration - 0.5, step * (k + 0.5)) if duration > 1 else 0.0
            if t < frm - 1e-6:
                continue
            start = max(0.0, min(t - win / 2, duration - win))
            out.append(
                {
                    "t": round(max(0.0, t), 3),
                    "start": round(start, 3),
                    "dur": round(min(win, duration - start), 3),
                }
            )
    else:
        n = max(1, math.ceil(duration / win))
        for k in range(n):
            start = k * win
            dur = min(win, duration - start)
            # By its centre ``t``, the stored next_t: by its start, a resumed
            # sound file skipped one window for good
            if start + dur / 2 < frm - 1e-6:
                continue
            if dur < 1.0 and k:
                continue
            out.append(
                {"t": round(start + dur / 2, 3), "start": round(start, 3), "dur": round(dur, 3)}
            )
    return out


def count_moments(duration: float, video: bool, s: dict, frm: float = 0.0) -> int:
    """``len(plan_moments(...))`` without making them."""
    if duration <= 0:
        return 0
    if video:
        step = s["video_interval_sec"]
        n = max(1, math.ceil(duration / step))
        if frm <= 0:
            return n
        # t_k = step * (k + 0.5), the last one pulled to duration - 0.5
        return sum(
            1
            for k in range(n)
            if (min(duration - 0.5, step * (k + 0.5)) if duration > 1 else 0.0) >= frm - 1e-6
        )
    win = s["sound_window_sec"]
    n = max(1, math.ceil(duration / win))
    if n > 1 and duration - (n - 1) * win < 1.0:
        n -= 1
    if frm <= 0:
        return n
    # By the windows' centres, as plan_moments
    return sum(1 for k in range(n) if k * win + min(win, duration - k * win) / 2 >= frm - 1e-6)


def _gpu_seconds(spec_kinds: set[str], moments: int, frames: int, stills: bool) -> float:
    """GPU time of a pass that makes these kinds for ``moments`` moments
    (``stills``: it looks at the stills — for repeats or for the kind)."""
    secs = moments / RATE["image"] if stills else 0.0
    if "video" in spec_kinds:
        secs += moments / (RATE["video4"] * 4 / frames)
    if "videoaudio" in spec_kinds:
        secs += moments / (RATE["videoaudio4"] * 4 / frames)
    if "audio" in spec_kinds:
        secs += moments / RATE["audio"]
    if "speech" in spec_kinds:
        secs += moments / RATE["speech"]
    return secs


def estimate(s: dict | None = None, fresh: bool = False) -> dict:
    """What the scope holds and what indexing what is left takes — files,
    hours, moments, vectors, GPU hours, disk; and how far it is (coverage:
    recordings and hours fully indexed)."""
    from whisper_tm.media.store import bytes_per_vector, get_store

    s = s or settings()
    store = get_store()
    files = scope_files(s, fresh=fresh)
    states = store.file_states()
    lay = layout(s, store)
    base = fingerprint(s, lay)
    vh = sum(f["duration"] for f in files if f["video"]) / 3600
    ah = sum(f["duration"] for f in files if not f["video"]) / 3600
    frames = frames_for(s, vh, store.meta("clip_frames"))
    todo = done = fills = 0
    hours_done = 0.0
    moments = 0
    gpu = 0.0
    vectors: dict[str, int] = {}
    new_stills = 0
    for f in files:
        st = states.get(f["path"])
        want = kind_specs(s, frames, f["video"])
        action, specs = plan_file(st, base, want)
        if action == "done":
            done += 1
            hours_done += f["duration"] / 3600
            continue
        todo += 1
        kinds = {spec_kind(sp) for sp in specs}
        if action == "fill":
            fills += 1  # its hours count once every wanted kind is there
            m = count_moments(f["duration"], f["video"], s)
            known = any(spec_kind(h) in VISUAL for h in st["kinds"])
            looks = f["video"] and ("image" in kinds or (not known and kinds & set(VISUAL)))
        else:
            frm = st["next_t"] if action == "continue" and st else 0.0
            if action == "continue" and st:
                hours_done += min(frm, f["duration"]) / 3600
            m = count_moments(f["duration"], f["video"], s, frm)
            looks = f["video"] and bool(kinds & set(VISUAL))
            if looks:
                new_stills += m
        moments += m
        gpu += _gpu_seconds(kinds, m, frames, looks)
        for k in kinds:
            if k in VISUAL and not f["video"]:
                continue
            vectors[k] = vectors.get(k, 0) + m
    n_vec = sum(vectors.values())
    vec_bytes = n_vec * bytes_per_vector(lay[0])
    # searching keeps every vector in RAM: 8-bit, a scale and three ids each
    have = sum(store.counts()["kinds"].values())
    ram = (have + n_vec) * (lay[0] + 40)
    ram_total = _ram_total()
    t_have = thumbs_bytes(store)
    t_cap = s["thumbs_max_mb"] * 1024**2
    t_new = new_stills * THUMB_BYTES
    # 0 = no limit, as for the writing (_thumbs_on)
    t_bytes = t_new if t_cap <= 0 else max(0, min(t_new, t_cap - t_have))
    total_hours = vh + ah
    return {
        "files": len(files),
        "todo": todo,
        "done": done,
        "fills": fills,
        "video_hours": round(vh, 3),
        "audio_hours": round(ah, 3),
        "hours_done": round(hours_done, 3),
        "share_done": round(hours_done / total_hours, 4) if total_hours else 0.0,
        "clip_frames": frames,
        "moments": moments,
        "vectors": n_vec,
        "vectors_by_kind": vectors,
        "gpu_hours": round(gpu / 3600, 1),
        "bytes": vec_bytes,
        "thumb_bytes": t_bytes,
        "ram_bytes": ram,
        "ram_total": ram_total,
        "ram_heavy": bool(ram_total) and ram > ram_total / 2,
        "thumbs_have": t_have,
        "thumbs_cap": t_cap,
        "free_gb": _round(disk_free_gb()),
        "min_free_gb": s["min_free_gb"],
        "layout": {
            "dims": lay[0],
            "format": lay[1],
            "wanted_dims": s["dims"],
            "differs": bool(store.dims()) and (lay[0] != s["dims"] or lay[1] != "int8"),
        },
        "at": time.time(),
    }


def _ram_total() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


_est: dict = {"value": None, "key": None, "at": 0.0, "busy": False, "error": None}
_est_lock = threading.Lock()


def estimate_cached(running: bool = False) -> dict:
    """The last estimate at once (``value`` None before the first), and a
    new one computed in the background when it is old or the settings
    changed: ``{"value", "age", "computing", "error"}``."""
    s = settings()
    key = json.dumps({k: s[k] for k in sorted(s)}, sort_keys=True)
    ttl = ESTIMATE_TTL_RUNNING_SEC if running else ESTIMATE_TTL_SEC
    with _est_lock:
        stale = _est["key"] != key or time.monotonic() - _est["at"] > ttl
        if stale and not _est["busy"]:
            _est["busy"] = True
            threading.Thread(
                target=_estimate_bg, args=(s, key), name="media-estimate", daemon=True
            ).start()
        val = _est["value"] if _est["key"] == key else None
        return {
            "value": val,
            "age": round(time.monotonic() - _est["at"], 1) if val else None,
            "computing": _est["busy"],
            "error": _est["error"],
        }


def _estimate_bg(s: dict, key: str) -> None:
    try:
        val = estimate(s)
        err = None
    except Exception as e:  # noqa: BLE001 — shown on the settings page
        logger.warning("Media index estimate failed: %s", e)
        val, err = None, str(e)
    with _est_lock:
        if val is not None:
            _est.update(value=val, key=key, at=time.monotonic())
        _est.update(busy=False, error=err)


def forget_estimate() -> None:
    with _est_lock:
        _est.update(key=None)


def _said(segments: list, t0: float, t1: float) -> str:
    out, last = [], None
    for seg in segments:
        if seg.get("end", 0) > t0 and seg.get("start", 0) < t1:
            sp = seg.get("speaker") or ""
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            out.append((f"[{sp}] " if sp and sp != last else "") + txt)
            last = sp
    return " ".join(out)[:600]


def _segments(path: str) -> list:
    from whisper_tm.extensions import db

    if not db:
        return []
    with db.connection() as conn:
        row = conn.execute(
            "SELECT json_data FROM transcripts WHERE abs_path=? AND complete=1 "
            "ORDER BY id DESC LIMIT 1",
            (path,),
        ).fetchone()
    try:
        return (json.loads(row[0]) or {}).get("segments", []) if row and row[0] else []
    except ValueError:
        return []


_LABEL_RE = re.compile(r"\[[^\]]{1,60}\]\s*")


def _spoken(shown: str | None) -> str:
    """What the model reads of a moment's text: without speaker labels."""
    return _LABEL_RE.sub("", shown or "").strip()


# ── The run ────────────────────────────────────────────────────────────────


class Indexer:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state: dict = {"running": False}
        self._thumb_bytes = 0
        self._thumb_cap = 0
        self._fails = 0
        self._failing = False

    def status(self) -> dict:
        with self._lock:
            st = dict(self.state)
        if st.get("running") and st.get("moments_done") and st.get("started"):
            secs = time.time() - st["started"] - st.get("paused_sec", 0.0)
            st["moments_per_hour"] = round(st["moments_done"] / max(secs, 1) * 3600)
        return st

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def start(self, *, rebuild: bool = False, reason: str = "") -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self.state = {
                "running": True,
                "stopping": False,
                "started": time.time(),
                "reason": reason,
                "files_done": 0,
                "files_total": 0,
                "fills": 0,
                "moments_done": 0,
                "vectors": 0,
                "repeats": 0,
                "file": None,
                "pass": None,
                "message": "Starting …",
                "errors": [],
                "missing": 0,
                "waiting": None,
                "paused_sec": 0.0,
            }
            self._thread = threading.Thread(
                target=self._run, args=(rebuild,), name="media-indexer", daemon=True
            )
            self._thread.start()
        forget_estimate()
        return True

    def stop(self) -> bool:
        with self._lock:
            if not (self._thread and self._thread.is_alive()):
                return False
            self.state["stopping"] = True
            self.state["message"] = "Stopping after this chunk …"
        self._stop.set()
        return True

    def running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    # ── The loop ─────────────────────────────────────────────────────────

    def _run(self, rebuild: bool) -> None:
        from whisper_tm.core.media_server import SERVER
        from whisper_tm.media.store import get_store

        store = get_store()
        turn = None
        try:
            s = settings()
            if not s["kinds"]:
                raise RuntimeError(
                    "Nothing to index — choose what under Settings → Images & sounds"
                )
            if rebuild:
                self._set(message="Emptying the index …")
                store.clear()
                shutil.rmtree(thumbs_dir(), ignore_errors=True)
                store.set_meta("thumb_bytes", 0)
            lay = layout(s, store)
            store.ensure_dims(lay[0])
            if lay[0] != s["dims"] or lay[1] != "int8":
                logger.info(
                    "Media index keeps %d numbers per vector (%s) — the settings say %d; "
                    "“Rebuild everything” applies them",
                    lay[0],
                    lay[1],
                    s["dims"],
                )
            files = scope_files(s, fresh=True)
            vh = sum(f["duration"] for f in files if f["video"]) / 3600
            frames = frames_for(s, vh, store.meta("clip_frames"))
            if s["clip_frames"] <= 0 and not store.meta("clip_frames"):
                store.set_meta("clip_frames", frames)
            base = fingerprint(s, lay)
            states = store.file_states()
            work = []
            for f in files:
                action, specs = plan_file(
                    states.get(f["path"]), base, kind_specs(s, frames, f["video"])
                )
                if action != "done":
                    work.append((f, action))
            self._thumb_bytes = thumbs_bytes(store)
            self._thumb_cap = s["thumbs_max_mb"] * 1024**2
            self._set(
                files_total=len(work),
                clip_frames=frames,
                message=f"{len(work)} recordings to index",
            )
            logger.info(
                "Media index run: %d of %d recordings in scope to do (%d to fill), "
                "%d frames per clip, kinds %s",
                len(work),
                len(files),
                sum(1 for _, a in work if a == "fill"),
                frames,
                ",".join(s["kinds"]),
            )
            SERVER.indexing = True
            turn = _Turn(s)
            self._fails = 0
            self._failing = False
            self._work(store, work, s, frames, base, turn)
            self._set(message="Stopped" if self._stop.is_set() else "Done")
            if not self._stop.is_set():
                _describe_after()
        except InterruptedError:
            self._set(message="Stopped")
        except Exception as e:  # noqa: BLE001 — shown on the settings page
            if self._stop.is_set() and not self._failing:
                # Asked to stop, then the service went away under the call:
                # the app shutting down stops the run, then the service, and
                # the chunk in flight broke — a stop, not a failure with a
                # traceback in the log at every restart
                logger.info("Media index run stopped (%s: %s)", type(e).__name__, e)
                self._set(message="Stopped")
            else:
                logger.exception("Media index run failed")
                self._set(message=f"Failed: {e}", error=str(e))
        finally:
            if turn is not None:
                turn.end()
            SERVER.indexing = False
            try:
                store.set_meta("thumb_bytes", self._thumb_bytes)
            except Exception:  # noqa: BLE001
                pass
            self._set(running=False, stopping=False, ended=time.time(), file=None, waiting=None)
            forget_estimate()

    def _work(self, store, work: list, s: dict, frames: int, base: str, turn: _Turn) -> None:
        """The recordings of a run, ``parallel_files`` at a time. A failure
        in one stops the others after their chunk in hand and is raised."""
        todo = iter(work)
        failed: list[BaseException] = []

        def worker():
            while not self._stop.is_set():
                with self._lock:
                    item = next(todo, None)
                if item is None:
                    return
                try:
                    self._one(store, item[0], s, frames, base, turn)
                    self._add(files_done=1)
                    if self._fails >= FAILS_IN_A_ROW:
                        last = (self.state.get("errors") or [{}])[-1]
                        raise RuntimeError(
                            f"{FAILS_IN_A_ROW} recordings in a row failed — "
                            f"the last: {last.get('error', '?')}"
                        )
                except BaseException as e:  # noqa: BLE001 — raised by the run below
                    with self._lock:
                        failed.append(e)
                    # A break after a stop (the service shut down under
                    # the request) stays a stop
                    if not isinstance(e, InterruptedError) and not self._stop.is_set():
                        self._failing = True
                    self._stop.set()
                    return

        n = max(1, min(s["parallel_files"], len(work)))
        if n == 1:
            worker()
        else:
            threads = [
                threading.Thread(target=worker, name=f"media-indexer-{k + 1}", daemon=True)
                for k in range(n)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if failed:
            raise next((e for e in failed if not isinstance(e, InterruptedError)), failed[0])

    def _one(self, store, f: dict, s: dict, frames: int, base: str, turn: _Turn) -> None:
        """A file until it has every wanted kind: its pass, then a fill. A
        file the service fails on is marked and left for the next run."""
        for _ in range(3):
            if self._stop.is_set():
                return
            if not os.path.isfile(f["path"]):
                self._add(missing=1)
                return
            want = kind_specs(s, frames, f["video"])
            st = store.file(f["path"])
            state = (
                None
                if st is None
                else {"settings": st["settings"], "status": st["status"], "kinds": st["kinds"]}
            )
            action, specs = plan_file(state, base, want)
            if action == "done":
                self._fails = 0
                return
            if action == "fill":
                ok = self._fill(store, f, st, specs, s, frames, turn)
            else:
                ok = self._file(
                    store,
                    f,
                    s,
                    frames,
                    base,
                    specs if action == "full" else st["kinds"],
                    turn,
                    fresh=action == "full" and st is not None and st["settings"] == base,
                )
            if not ok:
                return

    def _wait_for_room(self, s: dict, turn: _Turn) -> None:
        need = s["min_free_gb"]
        if need <= 0:
            return
        t0 = None
        while True:
            free = disk_free_gb()
            if free is None or free >= need:
                if t0 is not None:
                    self._add(paused_sec=time.time() - t0)
                    self._set(waiting=None)
                return
            if t0 is None:
                t0 = time.time()
                turn.end()
                logger.info("Media index: %.1f GB free — waiting until %g GB are", free, need)
            self._set(
                waiting="disk",
                free_gb=round(free, 1),
                message=(
                    f"⏳ Waiting for disk space — {free:.1f} GB free, the index leaves {need:g} GB"
                ),
            )
            if self._stop.wait(DISK_WAIT_SEC):
                raise InterruptedError("Stopped")

    def _thumbs_on(self) -> bool:
        return self._thumb_cap <= 0 or self._thumb_bytes < self._thumb_cap

    def _call(
        self, store, row: dict, path: str, payload: dict, s: dict, turn: _Turn
    ) -> dict | None:
        """One request to the service at the GPU — smaller after running
        out of memory; ``None`` when the file is gone or the run stops."""
        from whisper_tm.core.media_server import SERVER, MediaUnavailable

        for attempt in range(3):
            if self._stop.is_set():
                return None
            self._wait_for_room(s, turn)
            turn.begin(self)
            try:
                try:
                    res = SERVER.call("/index", payload, timeout=900)
                finally:
                    turn.done()
                if res.get("thumb_bytes"):
                    with self._lock:
                        self._thumb_bytes += int(res["thumb_bytes"])
                        total = self._thumb_bytes
                    store.set_meta("thumb_bytes", total)
                return res
            except MediaUnavailable as e:
                if getattr(e, "oom", False) and attempt < 2:
                    logger.info(
                        "Media index: out of GPU memory — giving the card back, retrying smaller"
                    )
                    turn.end()
                    payload["batch"] = max(1, payload["batch"] // 2)
                    self._stop.wait(OOM_PAUSE_SEC)
                    continue
                if getattr(e, "status", 0) in (400, 404, 500, 507):
                    # this recording (unreadable, gone, too big even in small
                    # passes) — marked, the run goes on
                    logger.warning("Media index: %s failed: %s", path, e)
                    store.fail_file(row["id"], str(e))
                    self._note_error(path, str(e))
                    self._fails += 1
                    return None
                raise
        return None

    def _payload(self, f: dict, chunk: list, kinds: list[str], s: dict, frames: int, prev) -> dict:
        return {
            "path": f["path"],
            "video": f["video"],
            "moments": chunk,
            "kinds": kinds,
            "clip_sec": s["clip_sec"],
            "clip_frames": frames,
            "dedupe": s["dedupe_similarity"],
            "prev": prev,
            "device": s["index_device"],
            "batch": s["batch"],
            "workers": s["grab_workers"],
            "thumb_width": s["thumb_width"],
            "thumbs": self._thumbs_on(),
        }

    def _file(
        self,
        store,
        f: dict,
        s: dict,
        frames: int,
        base: str,
        specs: list[str],
        turn: _Turn,
        fresh: bool = False,
    ) -> bool:
        """A file's pass: its moments, with the vectors of ``specs``
        (``fresh``: start over although the settings are the same — a pass
        that was under way with other kinds). False: the file failed or the
        run stops."""
        path = f["path"]
        row = store.start_file(
            path, video=f["video"], duration=f["duration"], settings=base, kinds=specs, fresh=fresh
        )
        if row["status"] == "done":
            return True
        kinds = sorted({spec_kind(sp) for sp in specs})
        moments = plan_moments(f["duration"], f["video"], s, frm=row["next_t"])
        segs = _segments(path)
        prev = row["prev"]
        last_thumb = _last_thumb(store, row["id"]) if row["next_t"] else None
        self._set(file=path, message=f"{Path(path).name}", **{"pass": "index"})
        n = s["chunk_moments"]
        t_file = time.time()
        for k in range(0, len(moments), n):
            if self._stop.is_set():
                return False
            chunk = moments[k : k + n]
            for m in chunk:
                m["shown"] = _said(segs, m["start"], m["start"] + m["dur"])
                m["text"] = _spoken(m["shown"])
                m["thumb"] = thumb_path(path, m["t"]) if f["video"] else None
            t0 = time.time()
            payload = self._payload(f, chunk, kinds, s, frames, prev)
            res = self._call(store, row, path, payload, s, turn)
            if res is None:
                return False
            rows = []
            for m, got in zip(chunk, res["moments"]):
                vecs = {kind: _unb64(v) for kind, v in (got.get("vectors") or {}).items()}
                if f["video"] and got.get("still"):
                    last_thumb = m["thumb"]
                rows.append(
                    {
                        "t": m["t"],
                        "start": m["start"],
                        "dur": m["dur"],
                        "repeat": got.get("repeat"),
                        "thumb": (
                            (m["thumb"] if got.get("still") else last_thumb) if f["video"] else None
                        ),
                        "text": m["shown"],
                        "vectors": vecs,
                    }
                )
            prev = res.get("prev") or prev
            done = k + n >= len(moments)
            nv = store.add_moments(
                row["id"],
                rows,
                next_t=(moments[k + n]["t"] if not done else f["duration"]),
                prev=prev,
                done=done,
                seconds=time.time() - t0,
            )
            self._progress(
                rows, nv, f"{Path(path).name} · {min(k + n, len(moments))}/{len(moments)} moments"
            )
            turn.maybe_yield(self)
        if not moments:
            store.add_moments(
                row["id"],
                [],
                next_t=f["duration"],
                prev=prev,
                done=True,
                seconds=time.time() - t_file,
            )
        self._fails = 0
        return True

    def _fill(
        self, store, f: dict, row: dict, specs: list[str], s: dict, frames: int, turn: _Turn
    ) -> bool:
        """Vectors of kinds a file doesn't have yet, for the moments it has.
        A kind made before with other clip settings goes first. False: the
        file failed or the run stops (the next run makes them again)."""
        path = f["path"]
        self._add(fills=1)
        have = list(row["kinds"])
        kinds = sorted({spec_kind(sp) for sp in specs})
        old = [h for h in have if spec_kind(h) in kinds]
        if old:
            store.delete_kinds(row["id"], sorted({spec_kind(h) for h in old}))
        moments = store.file_moments(row["id"])
        known = any(spec_kind(h) in VISUAL for h in have)
        looks = f["video"] and not known and bool(set(kinds) & set(VISUAL))
        self._set(
            file=path,
            message=f"{Path(path).name} · adding {', '.join(kinds)}",
            **{"pass": "fill"},
        )
        n = s["chunk_moments"]
        prev = None
        last_thumb = None
        t_file = time.time()
        for k in range(0, len(moments), n):
            if self._stop.is_set():
                return False
            chunk = moments[k : k + n]
            sent = []
            for m in chunk:
                thumb = m["thumb"]
                if looks:
                    thumb = thumb_path(path, m["t"])
                sent.append(
                    {
                        "t": m["t"],
                        "start": m["start"],
                        "dur": m["dur"],
                        "text": _spoken(m["text"]),
                        "thumb": thumb,
                    }
                )
            payload = self._payload(f, sent, kinds, s, frames, prev)
            if f["video"] and known:
                payload["repeats"] = [bool(m["repeat"]) for m in chunk]
            t0 = time.time()
            res = self._call(store, row, path, payload, s, turn)
            if res is None:
                return False
            items, marks = [], []
            for m, sm, got in zip(chunk, sent, res["moments"]):
                for kind, v in (got.get("vectors") or {}).items():
                    items.append((m["id"], kind, _unb64(v)))
                if looks:
                    if got.get("still"):
                        last_thumb = sm["thumb"]
                    marks.append(
                        (
                            m["id"],
                            bool(got.get("repeat")),
                            sm["thumb"] if got.get("still") else last_thumb,
                        )
                    )
            prev = res.get("prev") or prev
            nv = store.add_items(items)
            if marks:
                store.update_moments(row["id"], marks)
            self._progress(
                [{"repeat": r} for _, r, _ in marks] if marks else [{} for _ in chunk],
                nv,
                (
                    f"{Path(path).name} · adding {', '.join(kinds)} · "
                    f"{min(k + n, len(moments))}/{len(moments)} moments"
                ),
                time.time() - t0,
            )
            turn.maybe_yield(self)
        keep = [h for h in have if spec_kind(h) not in kinds]
        store.set_file_kinds(row["id"], keep + list(specs))
        store.finish_file(row["id"], time.time() - t_file)
        self._fails = 0
        return True

    def _add(self, **kw) -> None:
        """Counters of the state, added to — several files run at once."""
        with self._lock:
            for k, v in kw.items():
                self.state[k] = self.state.get(k, 0) + v

    def _progress(self, rows: list, nv: int, message: str, _secs: float = 0.0) -> None:
        self._add(
            moments_done=len(rows),
            vectors=nv,
            repeats=sum(1 for r in rows if r.get("repeat")),
        )
        self._set(message=message)

    def _note_error(self, path: str, err: str) -> None:
        with self._lock:
            self.state.setdefault("errors", []).append({"path": path, "error": err[:300]})
            del self.state["errors"][:-20]


def _unb64(v: str):
    from whisper_tm.media.service import unb64

    return unb64(v)


def _last_thumb(store, file_id: int) -> str | None:
    """The still a continued file's next repeats show: its last moment's."""
    ms = store.file_moments(file_id)
    return ms[-1]["thumb"] if ms else None


class _Turn:
    """The index run's turn at the GPU: a stage of the gate, held for
    several chunks, given back (model released first) when someone waits.

    Several recordings run at once: each request is counted from
    :meth:`begin` to :meth:`done`, and the turn ends only when none is at
    the service — the model must not be released under a running request."""

    def __init__(self, s: dict):
        self.s = s
        self._hold = None
        self._since = 0.0
        self._cv = threading.Condition()
        self._calls = 0
        self._ending = False
        self._back_at = 0.0

    def begin(self, ix: Indexer) -> None:
        """Before a request: the turn taken or kept, the request counted."""
        with self._cv:
            while self._ending:
                self._cv.wait(0.5)
                if ix._stop.is_set():
                    raise InterruptedError("Stopped")
            if self._hold is None and self.s["index_device"] == "cuda":
                self._take(ix)
            self._calls += 1

    def done(self) -> None:
        with self._cv:
            self._calls -= 1
            self._cv.notify_all()

    def _take(self, ix: Indexer) -> None:
        # After a yield the waiting stage gets in first — also when the
        # other recording asks for the card again right away
        while time.monotonic() < self._back_at:
            if ix._stop.wait(min(0.2, self._back_at - time.monotonic())):
                raise InterruptedError("Stopped")
        if not self.s["index_while_queue_runs"]:
            self._wait_queue_idle(ix)
        from whisper_tm.core.gpu_gate import GATE

        ix._set(waiting="gpu", message="⏳ Waiting for the GPU — the queue has it")
        hold = _GateHold(GATE, self.s["vram_need_mb"]).start()
        while not hold.wait(time.monotonic() + 1.0, ix._stop.is_set):
            if ix._stop.is_set():
                hold.release()
                raise InterruptedError("Stopped")
            hold = _GateHold(GATE, self.s["vram_need_mb"]).start() if hold.done.is_set() else hold
        self._hold = hold
        self._since = time.monotonic()
        ix._set(waiting=None)

    def _wait_queue_idle(self, ix: Indexer) -> None:
        from whisper_tm.core import queue as _q

        while _q.busy() and not _q.is_paused():
            ix._set(waiting="queue", message="⏳ Waiting until no transcription runs")
            if ix._stop.wait(QUEUE_WAIT_SEC):
                raise InterruptedError("Stopped")
        ix._set(waiting=None)

    def maybe_yield(self, ix: Indexer) -> None:
        with self._cv:
            if self._hold is None or self._ending:
                return
            if time.monotonic() - self._since < self.s["yield_after_sec"]:
                return
        from whisper_tm.core.gpu_gate import GATE

        if GATE.waiting() or (not self.s["index_while_queue_runs"] and _queue_busy()):
            logger.info("Media index: giving the GPU back to the queue")
            self.end(back_after=YIELD_PAUSE_SEC)

    def end(self, back_after: float = 0.0) -> None:
        """The turn given back once no request is at the service."""
        with self._cv:
            if self._hold is None or self._ending:
                return
            self._ending = True
            try:
                while self._calls:
                    self._cv.wait()
                from whisper_tm.core.media_server import SERVER

                try:
                    SERVER.call("/release", {"device": "cuda"}, timeout=60)
                except Exception as e:  # noqa: BLE001 — the stage goes back either way
                    logger.info("Media index: releasing the GPU model: %s", e)
                self._hold.release()
                self._hold = None
                self._back_at = time.monotonic() + back_after
            finally:
                self._ending = False
                self._cv.notify_all()


def _queue_busy() -> bool:
    try:
        from whisper_tm.core import queue as _q

        return _q.busy() and not _q.is_paused()
    except Exception:  # noqa: BLE001
        return False


class _GateHold:
    """The stage ``media`` of the GPU gate, held by a thread of its own — a
    stage belongs to the thread that took it (as core/rerank_server.py)."""

    def __init__(self, gate, need_mb: int):
        self.gate = gate
        self.need_mb = need_mb
        self.entered = threading.Event()
        self.done = threading.Event()
        self._off = threading.Event()
        self.thread = threading.Thread(target=self._run, name="media-gpu-gate", daemon=True)

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
        except Exception as e:  # noqa: BLE001
            logger.warning("Media index: GPU gate: %s", e)
        finally:
            self.done.set()

    def wait(self, deadline: float, cancel=None) -> bool:
        while not self.entered.is_set():
            if self.done.is_set():
                return False
            if time.monotonic() >= deadline or (cancel is not None and cancel()):
                return False
            self.entered.wait(0.1)
        return True

    def release(self) -> None:
        self._off.set()


def _describe_after() -> None:
    """With ``llm_describe_after_index``: queue the describing of the new
    stills (media/describe.py) — when a language model is there."""
    from whisper_tm.core.helpers import parse_bool

    if not parse_bool(_cfg("llm_describe_after_index", False)):
        return
    try:
        from whisper_tm.media import describe, llm
        from whisper_tm.media.store import get_store

        if not llm.configured() or llm.vision_available() is False:
            return
        if not get_store().undescribed(1, None):
            return
        describe.enqueue(int(_num("llm_describe_limit", 1000, 1, 1_000_000)), settings()["folders"])
    except Exception as e:  # noqa: BLE001 — the index run is done either way
        logger.info("Media index: describing not queued: %s", e)


INDEXER = Indexer()


# ── Automatic runs ─────────────────────────────────────────────────────────

_auto_thread: threading.Thread | None = None


def start_auto() -> None:
    """With ``media_search.auto_index``: look for new transcriptions in
    scope every ten minutes and index them."""
    global _auto_thread
    if _auto_thread and _auto_thread.is_alive():
        return
    _auto_thread = threading.Thread(target=_auto_loop, name="media-auto-index", daemon=True)
    _auto_thread.start()


def _auto_loop() -> None:
    from whisper_tm.core.helpers import parse_bool
    from whisper_tm.core.media_server import enabled

    time.sleep(60)
    while True:
        try:
            if enabled() and parse_bool(_cfg("auto_index", False)) and not INDEXER.running():
                est = estimate(fresh=True)
                if est["todo"]:
                    logger.info(
                        "Media index: %d new recordings in scope — indexing them", est["todo"]
                    )
                    INDEXER.start(reason="automatic")
        except Exception as e:  # noqa: BLE001 — tried again in ten minutes
            logger.info("Media index (automatic): %s", e)
        time.sleep(600)

"""Stills and clips cut out of recordings with ffmpeg.

One place for the commands, a lock per file being cut (two requests for the
same clip cut it once), and keeping a cache folder under its size. Images &
sounds (api/media_search.py) and Find (find/agent.py) cut with it.

A clip plays in any browser whatever the recording is (.mkv, .avi, .wmv,
.opus …): H.264 at 360 lines in 4:2:0 — 10-bit and other pixel formats play
nowhere — with AAC sound, or AAC alone for a recording without a picture;
``+faststart`` so playing starts before the file is read to its end.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from whisper_tm.core import media_formats

logger = logging.getLogger(__name__)

#: A file younger than this is never trimmed: it may be about to be served
#: (a concurrent trim could delete a clip right before send_file)
TRIM_MIN_AGE_SEC = 60.0

#: Output file → [its lock, how many requests hold or wait for it]. Counted:
#: dropping the entry while one waited let a third request make a new lock —
#: two ffmpeg wrote the same .part
_locks: dict[str, list] = {}
_locks_lock = threading.Lock()


@contextlib.contextmanager
def _file_lock(out):
    key = str(out)
    with _locks_lock:
        entry = _locks.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _locks_lock:
            entry[1] -= 1
            if entry[1] <= 0 and _locks.get(key) is entry:
                del _locks[key]


def cut_still(src: str, t: float, out, *, width: int = 480, timeout: int = 60) -> bool:
    """A JPEG of ``src`` at ``t`` seconds, ``width`` wide, written to ``out``
    (through a temporary file). False when the recording can't be read."""
    out = Path(out)
    if not os.path.isfile(src):
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-y",
                "-ss",
                f"{max(0.0, float(t)):.3f}",
                *media_formats.source(src),
                "-frames:v",
                "1",
                "-vf",
                f"scale={int(width)}:-2",
                str(tmp),
            ],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        r = None
    if r is None or r.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def cut_clip(
    src: str, start: float, length: float, video: bool, out, *, timeout: int = 180, threads: int = 2
) -> bool:
    """``start`` … ``start + length`` of ``src`` as an MP4 the browser plays
    (see the module), written to ``out`` through a ``.part`` file. False —
    with the reason in the log — when ffmpeg can't make it."""
    out = Path(out)
    if not os.path.isfile(src):
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        str(int(threads)),
        "-y",
        "-ss",
        f"{max(0.0, float(start)):.3f}",
        "-t",
        f"{max(0.1, float(length)):.3f}",
        *media_formats.source(src),
    ]
    if video:
        cmd += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale=-2:360,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ac",
            "2",
        ]
    else:
        cmd += ["-vn", "-map", "0:a:0", "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    tmp = out.with_name(out.name + ".part")
    try:
        r = subprocess.run(
            cmd + ["-movflags", "+faststart", "-f", "mp4", str(tmp)],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        r = None
    if r is None or r.returncode or not tmp.is_file() or tmp.stat().st_size == 0:
        logger.info(
            "No clip of %s at %.0f s: %s",
            src,
            start,
            (r.stderr or b"")[-300:].decode("utf-8", "replace") if r else "timed out",
        )
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def _touch(out: Path) -> bool:
    """A hit: its time renewed, so trim() takes the unused first (otherwise
    popular clips went first). False when it is gone."""
    try:
        os.utime(out)
        return True
    except OSError:
        return out.is_file()


def cached(out, make, *, after_cut=None) -> Path | None:
    """``out`` when it exists, else made once by ``make(out) -> bool`` under
    the file's lock (a second request waits for the first).
    ``after_cut()`` runs after a real cut only — trimming on every hit was
    wasted work."""
    out = Path(out)
    if out.is_file() and _touch(out):
        return out
    with _file_lock(out):
        if out.is_file() and _touch(out):
            return out
        ok = make(out) and out.is_file()
    if not ok:
        return None
    if after_cut is not None:
        after_cut()
    return out


def trim(folder, max_mb: float, suffixes=(".jpg",)) -> None:
    """Keep ``folder`` (and below) under ``max_mb`` — the oldest files go
    first. Files being written (``.part``) and those younger than
    :data:`TRIM_MIN_AGE_SEC` are left alone."""
    folder = Path(folder)
    young = time.time() - TRIM_MIN_AGE_SEC
    if not folder.is_dir():
        return
    files = []
    for p in folder.rglob("*"):
        if p.suffix not in suffixes or ".part" in p.name:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, p))
    total = sum(size for _, size, _ in files)
    for mtime, size, p in sorted(files):
        if total <= max_mb * 1024 * 1024 or mtime > young:
            break
        try:
            p.unlink()
            total -= size
        except OSError:
            pass

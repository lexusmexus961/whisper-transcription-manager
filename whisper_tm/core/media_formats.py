"""The containers a recording may come in, and how ffmpeg is told so.

ffmpeg opens what a file's bytes say it is, not what its name says. A DASH
or HLS manifest, an ffconcat list or an SDP file named talk.mp4 makes every
read of it open other files or URLs of the file's choosing. Every command
that opens a source file — a recording, an upload, a downloaded video, a
query file — therefore names the containers it may be (``-format_whitelist``,
an input option, before ``-i``). The app's own work files (WAV, MP3, FLAC)
are among them; the internal ``-f concat`` and ``-f lavfi`` inputs set their
format themselves and don't go through here.
"""

from __future__ import annotations

from pathlib import Path

#: As ffprobe names the demuxers (one name can be a list:
#: "mov,mp4,m4a,3gp,3g2,mj2"). Playlists, manifests, image sequences and
#: scripts are not among them.
MEDIA_CONTAINERS = frozenset(
    {
        "mov",
        "mp4",
        "m4a",
        "3gp",
        "3g2",
        "mj2",
        "matroska",
        "webm",
        "avi",
        "asf",
        "flv",
        "live_flv",
        "mpeg",
        "mpegts",
        "mpegvideo",
        "m4v",
        "h264",
        "hevc",
        "ogg",
        "mp3",
        "wav",
        "w64",
        "flac",
        "aac",
        "loas",
        "amr",
        "amrnb",
        "amrwb",
        "caf",
        "au",
        "rm",
        "gsm",
        "voc",
        "aiff",
        "ac3",
        "eac3",
        "dts",
        "dtshd",
        "truehd",
        "mlp",
        "wv",
        "tta",
        "ape",
        "mxf",
        "nut",
        "ivf",
        "dv",
        "gxf",
        "wtv",
        "mpc",
        "mpc8",
        "tak",
        "dsf",
        "oma",
    }
)

#: The value of ``-format_whitelist``
WHITELIST = ",".join(sorted(MEDIA_CONTAINERS))


def source(path: str | Path) -> list[str]:
    """``-format_whitelist … -i <path>`` for ffmpeg."""
    return ["-format_whitelist", WHITELIST, "-i", str(path)]


def probe(path: str | Path) -> list[str]:
    """``-format_whitelist … <path>`` for ffprobe (the input is positional)."""
    return ["-format_whitelist", WHITELIST, str(path)]


def is_recording(path: str | Path) -> bool:
    """Is it a recording: an audio or video stream in a real container
    (:data:`MEDIA_CONTAINERS`)? True when ffprobe can't be asked (nothing
    would transcribe then anyway).

    ffprobe may open those containers only (``-format_whitelist``), so a
    manifest is not even read — reading it would fetch what it names.
    """
    import json
    import subprocess

    try:
        # Not strict UTF-8: a file's foreign metadata in ffprobe's messages
        # would make the call explode (test_clip_cut)
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-format_whitelist",
                WHITELIST,
                "-show_entries",
                "stream=codec_type:format=format_name",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if r.returncode != 0:
        return False
    try:
        found = json.loads(r.stdout or "{}")
    except ValueError:
        return False
    kinds = {s.get("codec_type") for s in found.get("streams") or []}
    container = set(str((found.get("format") or {}).get("format_name") or "").split(","))
    return bool({"audio", "video"} & kinds) and bool(container & MEDIA_CONTAINERS)

"""
Exporting video and audio — one recording per ffmpeg run.

Video: the original unchanged, MP4 (H.264/AAC), WebM (VP9/Opus) or MKV;
a size (the original, 1080p, 720p, 480p); burned-in subtitles; only some
speakers' parts or the pauses removed. Audio: MP3, WAV, FLAC, M4A (AAC),
Opus or the original track; a quality (bitrate) or sample rate; mono; the
same choice of parts.

"Only some speakers" and "pauses removed" keep a list of time spans (from
the transcript's segments) and drop the rest — ffmpeg's select/aselect
filters with those spans, the timestamps closed up behind them.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from whisper_tm.core import media_formats
from whisper_tm.core.audio_utils import filter_path, keep_audio_filter

logger = logging.getLogger(__name__)

VIDEO_FORMATS = ("original", "mp4", "webm", "mkv")
AUDIO_FORMATS = ("mp3", "wav", "flac", "m4a", "opus", "track")
SIZES = {"original": None, "1080": 1080, "720": 720, "480": 480}
AUDIO_QUALITY = {"low": "96k", "medium": "160k", "high": "256k"}
SAMPLE_RATES = (16000, 22050, 44100, 48000)
VIDEO_DEFAULTS = {
    "format": "mp4",
    "size": "original",
    "subtitles": False,
    "speakers": [],
    "remove_pauses": False,
    "with_transcript": False,
}
AUDIO_DEFAULTS = {
    "format": "mp3",
    "quality": "medium",
    "sample_rate": 0,
    "mono": False,
    "speakers": [],
    "remove_pauses": False,
}


class MediaExportError(ValueError):
    """An option that does not fit, or a recording that can't be exported."""


def video_options(raw: dict | None) -> dict:
    o = dict(VIDEO_DEFAULTS)
    o.update({k: v for k, v in (raw or {}).items() if k in VIDEO_DEFAULTS})
    if o["format"] not in VIDEO_FORMATS:
        raise MediaExportError(f"video format: one of {', '.join(VIDEO_FORMATS)}")
    if str(o["size"]) not in SIZES:
        raise MediaExportError("video size: original, 1080, 720 or 480")
    o["size"] = str(o["size"])
    for k in ("subtitles", "remove_pauses", "with_transcript"):
        o[k] = bool(o[k])
    if not isinstance(o["speakers"], list):
        raise MediaExportError("speakers: a list")
    return o


def audio_options(raw: dict | None) -> dict:
    o = dict(AUDIO_DEFAULTS)
    o.update({k: v for k, v in (raw or {}).items() if k in AUDIO_DEFAULTS})
    if o["format"] not in AUDIO_FORMATS:
        raise MediaExportError(f"audio format: one of {', '.join(AUDIO_FORMATS)}")
    if o["quality"] not in AUDIO_QUALITY:
        raise MediaExportError("audio quality: low, medium or high")
    try:
        o["sample_rate"] = int(o["sample_rate"] or 0)
    except (TypeError, ValueError):
        raise MediaExportError("sample rate: a number") from None
    if o["sample_rate"] and o["sample_rate"] not in SAMPLE_RATES:
        raise MediaExportError(f"sample rate: one of {', '.join(map(str, SAMPLE_RATES))}")
    o["mono"] = bool(o["mono"])
    o["remove_pauses"] = bool(o["remove_pauses"])
    if not isinstance(o["speakers"], list):
        raise MediaExportError("speakers: a list")
    return o


# ── Which parts ──────────────────────────────────────────────────────────


def spans(
    segments: list[dict],
    speakers: list[str] | None = None,
    names: dict | None = None,
    pad: float = 0.15,
    join_gap: float = 0.6,
) -> list[tuple[float, float]]:
    """The time spans to keep: the segments (of some speakers, or all),
    padded a little and joined where only a short gap lies between."""
    want = {s.lower() for s in (speakers or [])}
    names = names or {}
    out: list[list[float]] = []
    for s in sorted(segments or [], key=lambda x: float(x.get("start") or 0)):
        label = s.get("speaker") or ""
        if want and label.lower() not in want and str(names.get(label, "")).lower() not in want:
            continue
        a = max(0.0, float(s.get("start") or 0) - pad)
        b = float(s.get("end") or 0) + pad
        if b <= a:
            continue
        if out and a - out[-1][1] <= join_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in out]


def _select_expr(parts: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in parts)


# ── Commands ─────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """One run: the ffmpeg command, where it writes, and how long the
    result is expected to be (seconds, for the progress)."""

    cmd: list[str]
    out: Path
    seconds: float
    copy_only: bool = False


def _probe(src: Path) -> dict:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name:format=duration",
                "-of",
                "json",
                *media_formats.probe(src),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        import json

        d = json.loads(r.stdout or "{}")
        return {
            "video": any(
                s.get("codec_type") == "video" and s.get("codec_name") not in ("mjpeg", "png")
                for s in d.get("streams", [])
            ),
            "audio_codec": next(
                (
                    s.get("codec_name")
                    for s in d.get("streams", [])
                    if s.get("codec_type") == "audio"
                ),
                "",
            ),
            "duration": float((d.get("format") or {}).get("duration") or 0),
        }
    except Exception as e:  # noqa — a file ffprobe can't read is refused below
        logger.debug("ffprobe %s: %s", src, e)
        return {"video": False, "audio_codec": "", "duration": 0.0}


_TRACK_EXT = {"aac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg", "flac": "flac"}


def plan_video(
    src: Path, out_dir: Path, o: dict, parts: list[tuple[float, float]] | None, srt: Path | None
) -> Plan:
    info = _probe(src)
    if not info["video"]:
        raise MediaExportError("no video in this recording")
    stem = src.stem
    height = SIZES[o["size"]]
    change = bool(height or (o["subtitles"] and srt) or parts)
    if o["format"] == "original" and not change:
        out = out_dir / src.name
        return Plan(["cp", str(src), str(out)], out, info["duration"], copy_only=True)
    fmt = o["format"] if o["format"] != "original" else "mp4"
    out = out_dir / f"{stem}.{fmt}"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        *media_formats.source(src),
    ]
    vf, af = [], []
    if parts:
        vf += [f"select='{_select_expr(parts)}'", "setpts=N/FRAME_RATE/TB"]
        af += [keep_audio_filter(parts, _select_expr(parts))]
    if height:
        vf.append(f"scale=-2:'min({height},ih)'")
    if o["subtitles"] and srt:
        vf.append("subtitles=" + filter_path(srt))
    if fmt == "mkv" and not vf and not af:
        cmd += ["-map", "0", "-c", "copy"]
    else:
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if af:
            cmd += ["-af", ",".join(af)]
        if fmt == "webm":
            cmd += [
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                "0",
                "-crf",
                "33",
                "-row-mt",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                "128k",
            ]
        else:
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
            if fmt == "mp4":
                cmd += ["-movflags", "+faststart"]
    cmd.append(str(out))
    seconds = sum(b - a for a, b in parts) if parts else info["duration"]
    return Plan(cmd, out, seconds)


def plan_audio(src: Path, out_dir: Path, o: dict, parts: list[tuple[float, float]] | None) -> Plan:
    info = _probe(src)
    if not info["audio_codec"]:
        raise MediaExportError("no audio track in this recording")
    fmt = o["format"]
    if fmt == "track":
        ext = _TRACK_EXT.get(info["audio_codec"], "mka")
    else:
        ext = fmt
    out = out_dir / f"{src.stem}.{ext}"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        *media_formats.source(src),
        "-vn",
    ]
    af = []
    if parts:
        af += [keep_audio_filter(parts, _select_expr(parts))]
    if af:
        cmd += ["-af", ",".join(af)]
    if o["mono"]:
        cmd += ["-ac", "1"]
    if o["sample_rate"]:
        cmd += ["-ar", str(o["sample_rate"])]
    rate = AUDIO_QUALITY[o["quality"]]
    if fmt == "track":
        cmd += ["-c:a", "copy"] if not (af or o["mono"] or o["sample_rate"]) else ["-c:a", "flac"]
        if af or o["mono"] or o["sample_rate"]:
            out = out.with_suffix(".flac")  # a changed track can't stay a copy
    elif fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", rate]
    elif fmt == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    elif fmt == "flac":
        cmd += ["-c:a", "flac"]
    elif fmt == "m4a":
        cmd += ["-c:a", "aac", "-b:a", rate]
    elif fmt == "opus":
        cmd += ["-c:a", "libopus", "-b:a", {"96k": "64k", "160k": "96k", "256k": "160k"}[rate]]
    cmd.append(str(out))
    seconds = sum(b - a for a, b in parts) if parts else info["duration"]
    return Plan(cmd, out, seconds)


def run(plan: Plan, on_progress=None, should_stop=None) -> Path:
    """Run one plan; ``on_progress(fraction)`` while it runs; ``should_stop()``
    true cancels it (the half-written file is removed)."""
    if plan.copy_only:
        shutil.copy2(plan.cmd[1], plan.cmd[2])
        if on_progress:
            on_progress(1.0)
        return plan.out
    # In the job's own folder: a relative path in a filter must never land
    # in the server's working directory
    proc = subprocess.Popen(
        plan.cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        cwd=str(plan.out.parent),
    )
    try:
        for line in proc.stdout:
            if should_stop and should_stop():
                proc.kill()
                break
            if line.startswith("out_time_us=") and plan.seconds > 0 and on_progress:
                try:
                    on_progress(min(1.0, int(line.split("=", 1)[1]) / 1e6 / plan.seconds))
                except ValueError:
                    pass
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
        raise
    if should_stop and should_stop():
        try:
            plan.out.unlink()
        except OSError:
            pass
        raise MediaExportError("cancelled")
    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "")[-400:]
        try:
            plan.out.unlink()
        except OSError:
            pass
        raise MediaExportError(f"ffmpeg failed: {err.strip() or proc.returncode}")
    if on_progress:
        on_progress(1.0)
    return plan.out


def scratch_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="wtm_mexport_"))


def remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (
        path.unlink() if path.exists() else None
    )

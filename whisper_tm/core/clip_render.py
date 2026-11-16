"""
Making a clip from a recording's cuts — the Clips tab's "Make the clip".

Each cut is cut out exactly (decoded from the keyframe before, written
from its start), optionally without its pauses (ffmpeg's silencedetect
finds them; the speech between stays, with a little air), brought to the
chosen frame (the original 16:9, vertical 9:16 or square 1:1 — a fixed
crop position, no face tracking), subtitled (burned in, from the
transcript's lines on the clip's own timeline) and faded softly in and
out. The parts are encoded alike and joined without another encode.

One clip from all cuts, or one clip per cut. A recording without a
picture becomes sound only (M4A).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from whisper_tm.core import media_formats
from whisper_tm.core.audio_utils import filter_path

logger = logging.getLogger(__name__)

FADE = 0.25  # seconds, at the start and end of each cut
PAUSE_AIR = 0.15  # kept on each side of a removed pause
SIZES = {"orig": None, "vert": (1080, 1920), "square": (1080, 1080)}
#: libass sizes on its 288-line canvas: a tall frame wants smaller letters,
#: or two lines of 28 characters wrap to four
SUB_STYLE = {"orig": "", "vert": "FontSize=11,MarginV=40", "square": "FontSize=14,MarginV=24"}


class ClipError(ValueError):
    """A clip that can't be made."""


@dataclass
class Part:
    start: float
    end: float
    fade_in: bool = False
    fade_out: bool = False

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Plan:
    """One clip: the parts in order and where it goes."""

    parts: list[Part] = field(default_factory=list)
    out: Path | None = None

    @property
    def seconds(self) -> float:
        return sum(p.length for p in self.parts)


def has_video(src: Path) -> bool:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "V",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                *media_formats.probe(src),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return "video" in (r.stdout or "")
    except Exception as e:  # noqa — unreadable: treated as sound only
        logger.debug("ffprobe %s: %s", src, e)
        return False


_SIL = re.compile(r"silence_(start|end): (-?[0-9.]+)")


def silences(
    src: Path, a: float, b: float, min_len: float, noise_db: int = -35
) -> list[tuple[float, float]]:
    """The pauses (silence of at least ``min_len`` seconds) between a and b,
    in the recording's time."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-ss",
        f"{a:.3f}",
        "-t",
        f"{b - a:.3f}",
        *media_formats.source(src),
        "-vn",
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_len}",
        "-f",
        "null",
        "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return []
    out, start = [], None
    for kind, val in _SIL.findall(r.stderr or ""):
        t = float(val) + a
        if kind == "start":
            start = max(a, t)
        elif start is not None:
            out.append((start, min(b, t)))
            start = None
    if start is not None:
        out.append((start, b))
    return out


def speech_parts(a: float, b: float, quiet: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[a, b] without its pauses, a little air left on both sides."""
    keep, at = [], a
    for s, e in sorted(quiet):
        cut_from, cut_to = s + PAUSE_AIR, e - PAUSE_AIR
        if cut_to - cut_from <= 0.05:
            continue
        if cut_from > at:
            keep.append((at, cut_from))
        at = max(at, cut_to)
    if b > at:
        keep.append((at, b))
    return [(x, y) for x, y in keep if y - x >= 0.1]


def plan(src: Path, cuts: list[dict], settings: dict, find_silences=silences) -> list[Part]:
    """The parts of one clip from these cuts, in order."""
    parts: list[Part] = []
    for c in cuts:
        a, b = float(c["start"]), float(c["end"])
        spans = [(a, b)]
        if settings.get("pauses"):
            spans = speech_parts(
                a, b, find_silences(src, a, b, float(settings.get("pause_min") or 0.8))
            ) or [(a, b)]
        for i, (x, y) in enumerate(spans):
            parts.append(
                Part(
                    x,
                    y,
                    fade_in=bool(settings.get("fade")) and i == 0,
                    fade_out=bool(settings.get("fade")) and i == len(spans) - 1,
                )
            )
    return parts


def frame_filter(fmt: str, crop: float) -> list[str]:
    """Video filters for the frame: the original (at most 1080 lines), or a
    crop at a fixed position scaled to 9:16 or 1:1."""
    size = SIZES.get(fmt)
    if not size:
        return ["scale=-2:'min(1080,ih)'"]
    w, h = size
    ratio = w / h
    c = max(0.0, min(1.0, float(crop)))
    return [
        f"crop='min(iw,ih*{ratio:.6f})':'min(ih,iw/{ratio:.6f})':'(iw-ow)*{c:.3f}':'(ih-oh)/2'",
        f"scale={w}:{h}",
        "setsar=1",
    ]


def subtitles_for(lines: list[dict], part: Part, path: Path, cpl: int, names: bool = False) -> bool:
    """An SRT of the transcript's lines within this part, on the part's own
    timeline. False when nothing is said in it."""
    from whisper_tm.core import exporter as ex

    mine = []
    for ln in lines:
        if ln["end"] <= part.start or ln["start"] >= part.end:
            continue
        mine.append(
            {
                **ln,
                "start": max(ln["start"], part.start) - part.start,
                "end": min(ln["end"], part.end) - part.start,
            }
        )
    if not mine:
        return False
    opts = ex.options({"names": names, "sub_cpl": cpl, "sub_lines": 2, "sub_max": 5})
    doc = ex.Doc(name=path.name, rel=path.name, title="", lines=mine)
    path.write_text(ex.render_srt(doc, opts), encoding="utf-8")
    return True


def part_cmd(
    src: Path,
    part: Part,
    out: Path,
    video: bool,
    vf: list[str],
    srt: Path | None,
    sub_style: str = "",
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-ss",
        f"{part.start:.3f}",
        *media_formats.source(src),
        "-t",
        f"{part.length:.3f}",
    ]
    af = []
    if part.fade_in:
        af.append(f"afade=t=in:st=0:d={FADE}")
    if part.fade_out and part.length > 2 * FADE:
        af.append(f"afade=t=out:st={part.length - FADE:.3f}:d={FADE}")
    if video:
        v = list(vf)
        if srt:
            v.append(
                "subtitles="
                + filter_path(srt)
                + (f":force_style='{sub_style}'" if sub_style else "")
            )
        if part.fade_in:
            v.append(f"fade=t=in:st=0:d={FADE}")
        if part.fade_out and part.length > 2 * FADE:
            v.append(f"fade=t=out:st={part.length - FADE:.3f}:d={FADE}")
        v.append("fps=30")
        cmd += ["-map", "0:V:0", "-map", "0:a:0?", "-vf", ",".join(v)]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(out),
        ]
    else:
        cmd += ["-vn", "-map", "0:a:0"]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(out)]
    return cmd


def _run(cmd: list[str], seconds: float, on_progress, should_stop, cwd: Path | None = None) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        cwd=str(cwd) if cwd else None,
    )
    try:
        for line in proc.stdout:
            if should_stop and should_stop():
                proc.kill()
                break
            if line.startswith("out_time_us=") and seconds > 0 and on_progress:
                try:
                    on_progress(min(1.0, int(line.split("=", 1)[1]) / 1e6 / seconds))
                except ValueError:
                    pass
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
        raise
    if should_stop and should_stop():
        raise ClipError("cancelled")
    if proc.returncode != 0:
        raise ClipError(
            "ffmpeg failed: "
            + ((proc.stderr.read() if proc.stderr else "")[-400:].strip() or str(proc.returncode))
        )


def render(
    src: Path,
    parts: list[Part],
    out: Path,
    settings: dict,
    lines: list[dict] | None,
    on_progress=None,
    should_stop=None,
    video: bool | None = None,
) -> Path:
    """Make one clip from its parts into ``out`` (its suffix is set here:
    .mp4, or .m4a for sound only). Temporary files lie beside it and go."""
    if not parts:
        raise ClipError("no cuts")
    if not src.is_file():
        raise ClipError("the recording is not there")
    video = has_video(src) if video is None else video
    # Added, not swapped: with_suffix() took "talk.v2_clip1" for a name with
    # the extension ".v2_clip1", and every clip of that recording was
    # written over the same "talk.mp4"
    out = out.with_name(out.name + (".mp4" if video else ".m4a"))
    work = out.parent / f".{out.stem}.parts"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    total = sum(p.length for p in parts) or 1.0
    done = 0.0
    try:
        vf = (
            frame_filter(settings.get("format", "orig"), settings.get("crop", 0.5)) if video else []
        )
        cpl = (
            28
            if settings.get("format") == "vert"
            else 32
            if settings.get("format") == "square"
            else 42
        )
        made = []
        for i, p in enumerate(parts):
            if should_stop and should_stop():
                raise ClipError("cancelled")
            srt = None
            if video and settings.get("subs") and lines:
                srt = work / f"{i:03d}.srt"
                if not subtitles_for(lines, p, srt, cpl):
                    srt = None
            piece = work / f"{i:03d}{out.suffix}"
            _run(
                part_cmd(src, p, piece, video, vf, srt, SUB_STYLE.get(settings.get("format"), "")),
                p.length,
                (lambda f, d=done, n=p.length: on_progress and on_progress((d + f * n) / total)),
                should_stop,
                cwd=work,
            )
            made.append(piece)
            done += p.length
        if len(made) == 1:
            shutil.move(str(made[0]), out)
        else:
            lst = work / "list.txt"
            lst.write_text("".join(f"file '{m.name}'\n" for m in made), encoding="utf-8")
            cmd = (
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(lst),
                    "-c",
                    "copy",
                ]
                + (["-movflags", "+faststart"] if video else [])
                + [str(out)]
            )
            r = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace", timeout=600, cwd=str(work)
            )
            if r.returncode != 0:
                raise ClipError("joining the parts failed: " + (r.stderr or "")[-300:].strip())
        if on_progress:
            on_progress(1.0)
        return out
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)

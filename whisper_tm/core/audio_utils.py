"""
Audio utilities: FFmpeg operations, extraction, splitting, media info.
Cross-platform (Windows/Linux/macOS).
"""

from __future__ import annotations

import functools
import json
import logging
import re
import subprocess
import time
from pathlib import Path

from whisper_tm.core import media_formats
from whisper_tm.core.helpers import format_ts

_dbg_log = logging.getLogger("whisper_tm.subprocess")


def filter_path(path) -> str:
    """A file path as an ffmpeg filter's ``filename`` option:
    ``"subtitles=" + filter_path(srt)``.

    ffmpeg reads a filter graph twice. The option value needs ``\\``, ``:``
    and ``'`` escaped; the graph around it needs the whole value in single
    quotes, in which a backslash escapes nothing — an apostrophe has to end
    the quote, stand escaped, and open it again. Escaped once only, a name
    like "Don't stop" ended the quote: the filter failed, and what followed
    the apostrophe was read as filters of its own.
    """
    value = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return "filename='" + value.replace("'", "'\\''") + "'"


def _run_probe(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    """subprocess.run with timing.

    The short ffprobe calls do NOT go through run_cancellable. Without this
    wrapper they were missing from the evaluation — and they add up
    precisely: on a scan across tens of thousands of files on a NAS, ffprobe
    is the lion's share.
    """
    from whisper_tm.core import debug

    tool = Path(str(cmd[0])).name if cmd else "?"
    t0 = time.perf_counter()
    try:
        # ffprobe writes UTF-8; Windows would read it in its ANSI code page
        return subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    finally:
        dt = time.perf_counter() - t0
        debug.record(f"subprocess: {tool}", dt)
        if debug.is_enabled():
            _dbg_log.debug("⏱ %s: %s · %s", tool, debug.fmt_secs(dt), cmd[-1] if cmd else "")


# ── FFprobe / media info ─────────────────────────────────────────────────


def get_media_info(filepath: Path) -> dict:
    """Complete media metadata via ffprobe (for the index).

    Returns duration, resolution, audio/video codec, has_audio/has_video,
    bitrate and metadata title — all in a single ffprobe call.
    """
    empty = {
        "duration": 0.0,
        "duration_fmt": "—",
        "width": 0,
        "height": 0,
        "has_video": False,
        "has_audio": False,
        "audio_codec": "",
        "video_codec": "",
        "bitrate": 0,
        "title": "",
        "ok": False,
    }
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            *media_formats.probe(filepath),
        ]
        r = _run_probe(cmd, timeout=30)
        if r.returncode != 0 or not r.stdout:
            return empty
        data = json.loads(r.stdout)
        fmt = data.get("format", {}) or {}
        try:
            duration = float(fmt.get("duration", 0) or 0)
        except (TypeError, ValueError):
            duration = 0.0
        try:
            bitrate = int(fmt.get("bit_rate", 0) or 0)
        except (TypeError, ValueError):
            bitrate = 0

        width = height = 0
        has_video = has_audio = False
        video_codec = audio_codec = ""
        for s in data.get("streams", []):
            ctype = s.get("codec_type")
            if ctype == "video" and not has_video:
                # Ignore cover kind (mjpeg/png as "video")
                if s.get("disposition", {}).get("attached_pic"):
                    continue
                has_video = True
                width = s.get("width", 0) or 0
                height = s.get("height", 0) or 0
                video_codec = s.get("codec_name", "") or ""
            elif ctype == "audio" and not has_audio:
                has_audio = True
                audio_codec = s.get("codec_name", "") or ""

        tags = fmt.get("tags", {}) or {}
        title = tags.get("title", "") or tags.get("TITLE", "")
        return {
            "duration": duration,
            "duration_fmt": format_ts(duration),
            "width": width,
            "height": height,
            "has_video": has_video,
            "has_audio": has_audio,
            "audio_codec": audio_codec,
            "video_codec": video_codec,
            "bitrate": bitrate,
            "title": title,
            "ok": True,
        }
    except Exception:
        return empty


def get_media_title_fast(filepath: Path) -> str:
    """Read only the metadata title (fast, no stream scan)."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format_tags=title",
            "-of",
            "json",
            *media_formats.probe(filepath),
        ]
        r = _run_probe(cmd, timeout=5)
        data = json.loads(r.stdout)
        tags = data.get("format", {}).get("tags", {})
        return tags.get("title", "") or tags.get("TITLE", "")
    except Exception:
        return ""


class CorruptMediaError(RuntimeError):
    """The file itself is corrupt — retrying will not change that.

    Distinguished from the plain RuntimeError, which also covers transient
    problems (NAS briefly unavailable, ffprobe timeout). Only THIS exception
    may permanently put a file on the skip list.
    """


# ffprobe messages that prove a file is corrupt, as opposed to a circumstance
# that might turn out differently on the next attempt. "moov atom not found"
# is the classic case of a recording cut off mid-way: for MP4/M4A the index
# sits at the end of the file and was never written.
_CORRUPT_MARKERS = (
    "moov atom not found",
    "invalid data found when processing input",
    "end of file",
    "could not find codec parameters",
)


def _is_corrupt_stderr(stderr: str) -> bool:
    low = (stderr or "").lower()
    return any(m in low for m in _CORRUPT_MARKERS)


def has_audio_stream(input_path: Path) -> bool:
    """Check whether a media file contains an audio stream.

    Distinguishes a genuine sample failure (returncode≠0 / unreadable output)
    from "runs fine, but no audio stream": on a real error an exception is
    raised instead of wrongly returning False — otherwise a transient
    ffprobe error would permanently put the file on the skip list.

    If the file is demonstrably corrupt (see _CORRUPT_MARKERS), a
    CorruptMediaError is raised: that one MAY go on the skip list, since it
    will not fix itself.
    """
    # "-v error" instead of "-v quiet": quiet also suppresses the error output
    # we write into the exception below — the log then only showed "Code 1):"
    # with no reason at all. With "error" it shows e.g. "moov atom not found"
    # (aborted recording), and it's immediately clear the file is corrupt.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        *media_formats.probe(input_path),
    ]
    r = _run_probe(cmd, timeout=30)
    if r.returncode != 0:
        reason = (r.stderr or "").strip().replace("\n", " ")[-300:]
        if _is_corrupt_stderr(r.stderr):
            raise CorruptMediaError(f"File is corrupt: {plain_reason(r.stderr)}")
        raise RuntimeError(
            f"File not readable (ffprobe code {r.returncode}): {reason or 'no error output'}"
        )
    try:
        data = json.loads(r.stdout)
    except Exception as e:
        raise RuntimeError(f"ffprobe output unreadable: {e}") from e
    return len(data.get("streams", [])) > 0


# ── Raw elementary streams (no container) ────────────────────────────────
# ADTS-AAC & co. carry neither an index nor a duration. libavformat then
# falls back to estimate_timings_from_bit_rate() — file size × 8 ÷ bitrate,
# where the bitrate only comes from the FIRST frames (libavformat/demux.c).
# For VBR material with a quiet start, i.e. exactly voice memos, this is off
# by a huge margin: e.g. 280 s instead of 240 s, up to a factor of
# 22 in the literature. That's why this counts instead of estimating.
RAW_STREAM_FORMATS = {"aac"}

# Output samples per AAC frame. With SBR (HE-AAC) it's 2048, and ffprobe
# already reports the DOUBLED sample rate — together they yield the correct
# duration, either one alone gives a factor-2 error.
_AAC_SAMPLES_PER_FRAME = {"HE-AAC": 2048, "HE-AACv2": 2048}
_AAC_SAMPLES_STANDARD = 1024


def is_raw_stream(format_name: str) -> bool:
    """Elementary stream without a duration in its header?

    ffprobe reports a list for real containers ("mov,mp4,m4a,…"); a single
    name without a comma is the signature of a raw stream.
    """
    name = (format_name or "").strip().lower()
    return bool(name) and "," not in name and name in RAW_STREAM_FORMATS


def duration_from_frames(fp: Path) -> float | None:
    """Exact duration of a raw AAC stream — counted, not estimated.

    ``-count_frames`` counts the DECODED frames. That's the difference from
    ``-count_packets``: an ADTS stream can contain packets without audio
    (FAAC writes its encoder tag as the first frame), which would otherwise
    be counted too.

    ``None`` means "cannot be determined" — then the estimate is kept.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-count_frames",
        "-show_entries",
        "stream=profile,sample_rate,nb_read_frames",
        "-of",
        "json",
        *media_formats.probe(fp),
    ]
    try:
        r = _run_probe(cmd, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return None
        s = (json.loads(r.stdout).get("streams") or [{}])[0]
        frames = int(s.get("nb_read_frames") or 0)
        rate = int(s.get("sample_rate") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError):
        return None
    if frames <= 0 or rate <= 0:
        return None
    per_frame = _AAC_SAMPLES_PER_FRAME.get(s.get("profile") or "", _AAC_SAMPLES_STANDARD)
    return frames * per_frame / rate


def get_duration(fp: Path) -> float:
    """Audio/video duration in seconds.

    Validates the return code and the presence of a duration; falls back to
    the stream duration if the format duration is missing (some files only
    have a stream duration but no format duration → otherwise
    ValueError/KeyError).

    For raw elementary streams, this counts instead of estimating — see
    ``duration_from_frames``.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=format_name,duration:stream=duration",
        "-of",
        "json",
        *media_formats.probe(fp),
    ]
    r = _run_probe(cmd, timeout=60)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(
            f"ffprobe duration failed (code {r.returncode}): {ffmpeg_failure(r.stderr, 300)}"
        )
    data = json.loads(r.stdout)
    if is_raw_stream((data.get("format", {}) or {}).get("format_name", "")):
        exact = duration_from_frames(fp)
        if exact is not None:
            return exact
    dur = (data.get("format", {}) or {}).get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    # Fallback: stream duration, if the format duration is missing/invalid
    for s in data.get("streams", []):
        sd = s.get("duration")
        if sd is not None:
            try:
                return float(sd)
            except (TypeError, ValueError):
                continue
    raise RuntimeError("No duration in ffprobe output (neither format nor stream)")


def get_audio_duration(fp: Path) -> float:
    """Duration of the AUDIO TRACK in seconds.

    Not the same as :func:`get_duration`. The container duration is based on
    the longest stream — for a recording whose audio cuts off early, that's
    the video track:

        recording.3gp    Video 8000.0 s    Audio 5400.0 s

    Anyone cutting audio chunks based on this produces empty files for the
    last 43 minutes. That is exactly what caused voice separation to fail
    with "chunk 5/5 extraction failed" — even though 90 minutes of audio had
    been extracted cleanly.

    Falls back to the container duration if the audio stream does not report
    its own duration (happens with some formats).
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "json",
        *media_formats.probe(fp),
    ]
    r = _run_probe(cmd, timeout=60)
    if r.returncode == 0 and r.stdout:
        try:
            for s in json.loads(r.stdout).get("streams", []):
                sd = s.get("duration")
                if sd is None:
                    continue
                value = float(sd)
                if value > 0:
                    return value
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return get_duration(fp)


def get_duration_safe(fp: Path) -> float:
    """Duration without raising an exception."""
    try:
        return get_duration(fp)
    except Exception:
        return 0


def _mean_volume_db(fp: Path) -> float | None:
    """Mean volume level (dBFS) via ffmpeg volumedetect. None on error."""
    try:
        r = _run_probe(
            ["ffmpeg", *media_formats.source(fp), "-af", "volumedetect", "-f", "null", "-"],
            timeout=120,
        )
        import re as _re

        m = _re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", r.stderr)
        return float(m.group(1)) if m else None
    except Exception:
        return None


# ── Process management ───────────────────────────────────────────────────


def run_cancellable(
    cmd: list, job=None, timeout: int = 14400, label: str | None = None
) -> subprocess.CompletedProcess:
    """Subprocess with cancel polling. Cross-platform safe.

    Uses SIGTERM instead of SIGKILL on Unix for a clean shutdown.
    On Windows, terminate() is used.

    ALL external tools go through here (ffmpeg, ffprobe, audio-separator)
    — that's exactly why the timing is hooked in at this point.
    """
    import platform
    import threading

    from whisper_tm.core import debug

    _t_start = time.perf_counter()
    _tool = Path(str(cmd[0])).name if cmd else "?"
    # A descriptive label ("audio track copy", "sep chunk 3/13") makes the
    # debug lines and the timing aggregation readable — the raw command line
    # is still shown alongside it.
    _op = f"{_tool} [{label}]" if label else _tool
    if debug.is_enabled():
        _dbg_log.debug("▶ %s: %s", _op, " ".join(str(c) for c in cmd))

    # creationflags for Windows, to start the subprocess without its own window
    kwargs = {}
    if platform.system() == "Windows":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)

    # Drain the pipes IN THE BACKGROUND — otherwise a deadlock: once the
    # OS pipe buffer (~64KB) fills up, the child process blocks on writing
    # and wait() never returns (ffmpeg/audio-separator log a lot to
    # stderr).
    _bufs = {"stdout": b"", "stderr": b""}

    def _drain(name, stream):
        try:
            _bufs[name] = stream.read()
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    _readers = [
        threading.Thread(target=_drain, args=(n, s), daemon=True)
        for n, s in (("stdout", proc.stdout), ("stderr", proc.stderr))
        if s is not None
    ]
    for t in _readers:
        t.start()

    deadline = time.time() + timeout
    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if job and job.cancel:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                raise InterruptedError("Cancelled") from None
            if time.time() > deadline:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                raise RuntimeError("FFmpeg timeout") from None

    for t in _readers:
        t.join(timeout=10)

    stdout = _bufs["stdout"].decode(errors="replace")
    stderr = _bufs["stderr"].decode(errors="replace")

    _dt = time.perf_counter() - _t_start
    debug.record(f"subprocess: {_op}", _dt)
    # Per-job total: the summary at the end of the job then shows how much
    # time in total went into external tools (ffmpeg/ffprobe/…).
    if job is not None:
        try:
            job.ffmpeg_secs = getattr(job, "ffmpeg_secs", 0.0) + _dt
            job.ffmpeg_calls = getattr(job, "ffmpeg_calls", 0) + 1
        except Exception:
            pass
    if debug.is_enabled():
        _dbg_log.debug("⏱ %s: %s (rc=%s)", _op, debug.fmt_secs(_dt), proc.returncode)
        if proc.returncode != 0 and stderr.strip():
            _dbg_log.debug("   stderr: %s", stderr.strip()[-500:])

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


# ── Audio extraction ─────────────────────────────────────────────────────

#: What ffmpeg says about the machine rather than the file — anywhere in its
#: output: its last line is often only "Conversion failed!"
FFMPEG_SYSTEM_ERRORS = (
    "no space left on device",
    "disk quota exceeded",
    "read-only file system",
    "permission denied",
    "input/output error",
    "transport endpoint is not connected",
    "stale file handle",
)


def ffmpeg_reason(stderr: str, limit: int = 200) -> str:
    """The line of ffmpeg's output that says why it failed: a system error
    anywhere in it, else the last line."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in lines:
        if any(e in ln.lower() for e in FFMPEG_SYSTEM_ERRORS):
            return ln[:limit]
    return lines[-1][:limit] if lines else ""


_FF_PREFIX = re.compile(r"^\[[^\]]*@ 0x[0-9a-fA-F]+\]\s*")


@functools.lru_cache(maxsize=1)
def aselect_works() -> bool:
    """Does this ffmpeg's ``aselect`` drop audio at all?

    Ubuntu 26.04's ffmpeg 8.0.1 passes every frame through ``aselect``,
    whatever the expression — even ``aselect='0'``. A cut made with it came
    out as long as the original, and the diarization then mapped its times
    back onto parts that were never cut. Asked once per run with one second
    of a sine: half of it selected must give half the samples."""
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=d=1:sample_rate=8000",
                "-af",
                "aselect='lt(t,0.5)',asetpts=N/SR/TB",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True  # no answer: the usual filter, as before
    works = r.returncode == 0 and len(r.stdout) < 12000  # 8,000 bytes when it works
    if not works:
        logging.getLogger("whisper_tm.audio_utils").warning(
            "This ffmpeg's aselect keeps every frame — cuts use atrim + concat instead"
        )
    return works


def keep_audio_filter(parts: list[tuple[float, float]], select_expr: str = "") -> str:
    """An audio filter that keeps exactly ``parts`` (seconds) and joins them.

    ``aselect`` with ``asetpts=N/SR/TB`` where it works (one pass, nothing
    held back); where it doesn't (``aselect_works``), one ``atrim`` per part
    after an ``asplit``, joined by ``concat`` — sample-exact in every ffmpeg,
    but the parts still to come wait in memory, so it is the fallback only.
    ``select_expr``: the aselect expression, if the caller builds its own
    (balanced for many parts)."""
    if not parts:
        return "aselect='0'"
    if aselect_works():
        expr = select_expr or "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in parts)
        return f"aselect='{expr}',asetpts=N/SR/TB"
    trims = [f"atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS" for a, b in parts]
    if len(trims) == 1:
        return trims[0]
    n = len(trims)
    split = "asplit=" + str(n) + "".join(f"[s{i}]" for i in range(n))
    chains = ";".join(f"[s{i}]{t}[p{i}]" for i, t in enumerate(trims))
    joined = "".join(f"[p{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1"
    return f"{split};{chains};{joined}"


def plain_reason(stderr: str, limit: int = 200) -> str:
    """ffmpeg's reasons in plain words, for a message a person reads: without
    the demuxer prefix ("[mov,mp4,m4a,3gp,3g2,mj2 @ 0x55…]") and without the
    file's path in front of a line — the message names the file already, and
    the path was the work copy's, deep in the cache."""
    out: list[str] = []
    for ln in (stderr or "").splitlines():
        ln = _FF_PREFIX.sub("", ln.strip())
        if ln.startswith("/") and ": " in ln:
            ln = ln.split(": ", 1)[1]
        if ln and ln not in out:
            out.append(ln)
    return "; ".join(out)[:limit]


def ffmpeg_failure(stderr: str, tail: int = 500) -> str:
    """What an error message carries of ffmpeg's output: a system error
    line when there is one (a full disk was hidden behind "Conversion
    failed!"), else the output's last ``tail`` characters."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    system = next((ln for ln in lines if any(e in ln.lower() for e in FFMPEG_SYSTEM_ERRORS)), "")
    return system[:tail] if system else (stderr or "")[-tail:]


def extract_audio(input_path: Path, output_path: Path, job=None) -> None:
    """Extract audio from video/audio (16kHz mono WAV)."""
    if not has_audio_stream(input_path):
        raise RuntimeError("No audio track present – file skipped")

    cmd = [
        "ffmpeg",
        "-y",
        *media_formats.source(input_path),
        "-vn",
        "-map",
        "0:a:0",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        # -rf64 auto: 4 GiB limit of the WAV format. At 16 kHz mono (32 kB/s)
        # that kicks in after 37.3 h — so it only affects very long
        # recordings, but large collections do contain some. Below the
        # limit, the file remains an ordinary RIFF/WAV.
        "-rf64",
        "auto",
        "-threads",
        "0",
        str(output_path),
    ]
    r = run_cancellable(cmd, job=job, timeout=14400, label="Audio extraction 16k")
    if r.returncode != 0:
        if "does not contain any stream" in r.stderr:
            raise RuntimeError("No audio track present – file skipped")
        raise RuntimeError(f"FFmpeg: {ffmpeg_failure(r.stderr)}")


# ── Output file search ───────────────────────────────────────────────────


def find_output_wav(
    directory: Path, stem_hint: str = "vocal", fallback: bool = True
) -> Path | None:
    """Find a WAV file in a directory by stem hint.

    Takes negative patterns into account: "no vocal" does not match for
    hint="vocal".
    """
    found = None
    hint_lower = stem_hint.lower()
    negative = f"no {hint_lower}"
    negative2 = f"no_{hint_lower}"

    # A missing directory simply means "nothing found" — the caller has a
    # way to handle that (insert silence, use the original). A raw
    # FileNotFoundError would instead take down the whole job.
    if not directory.is_dir():
        return None

    for f in sorted(directory.iterdir()):
        if f.suffix.lower() != ".wav":
            continue
        # Only check the suffix appended by the separator — the source stem
        # itself can contain keywords ("Background Story.mp4")
        fl = _separator_suffix(f)
        # Exclude negatives
        if negative in fl or negative2 in fl:
            continue
        if hint_lower in fl:
            found = f
            break
    if not found and fallback:
        wavs = list(directory.glob("*.wav"))
        if wavs:
            found = wavs[0]
    return found


def _separator_suffix(f: Path) -> str:
    """Extract the name suffix appended by the separator (lowercase).

    audio-separator appends the stem type to the end, e.g.
    "My Video_(Vocals).wav". The source stem itself can contain keywords
    ("Background Story.mp4") — so only the end is matched: preferably the
    last parenthesized block, otherwise the last 40 characters.
    """
    import re as _re

    stem = f.stem.lower()
    m = _re.search(r"\(([^()]*)\)[^()]*$", stem)
    if m:
        return m.group(1)
    return stem[-40:]


def find_output_wavs(directory: Path) -> tuple[Path | None, Path | None]:
    """Find Vocals + Instrumental WAVs.

    audio-separator output patterns:
      - "(Vocals).wav" / "(Instrumental).wav"    (MDX-NET)
      - "(No Vocals).wav" / "(Vocals).wav"        (Kim_Vocal, VR)
      - "(Primary Stem).wav" / "(Secondary Stem).wav"
    """
    import logging

    _log = logging.getLogger("whisper_tm.audio_utils")

    vocals = inst = None
    if not directory.is_dir():
        _log.warning("find_output_wavs: directory missing: %s", directory)
        return None, None
    wavs = sorted([f for f in directory.iterdir() if f.suffix.lower() == ".wav"])

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(
            "find_output_wavs: %d WAVs in %s: %s",
            len(wavs),
            directory.name,
            [(f.name, f.stat().st_size) for f in wavs],
        )

    for f in wavs:
        # Only match the separator suffix, not the source stem —
        # "Background Story (Vocals).wav" must not be classified as
        # Instrumental just because "background" is in the title.
        fl = _separator_suffix(f)

        # ── Negative patterns FIRST (highest priority) ─────
        # "No Instrumental" = Vocals (from some models)
        # Must be checked BEFORE "instrumental", since it's a substring!
        if any(
            p in fl
            for p in [
                "no instrumental",
                "no_instrumental",
                "(no instrumental)",
                "(no instrumentals)",
            ]
        ):
            vocals = f
            _log.debug("  → Vocals (via 'No Instrumental'): %s", f.name)
            continue

        # "No Vocals" / "No Vocal" = Instrumental (from vocal models)
        # Must be checked BEFORE "vocal", since it's a substring!
        if any(
            p in fl
            for p in [
                "no_vocal",
                "no vocal",
                "(no vocals)",
                "no_vocals",
            ]
        ):
            inst = f
            _log.debug("  → Instrumental (via 'No Vocals'): %s", f.name)
            continue

        # ── Positive patterns ─────────────────────────────────
        # Instrumental stem. "other": the Python API calls the Roformer
        # models' non-vocal stem "(other)" (the CLI called it
        # "(Instrumental)"); _separator_suffix returns the parenthesized
        # CONTENT, i.e. 'other'.
        if any(
            p in fl
            for p in [
                "instrumental",
                "accompaniment",
                "music",
                "background",
                "other",
            ]
        ):
            inst = f
            _log.debug("  → Instrumental: %s", f.name)
            continue

        # Vocals stem
        if any(p in fl for p in ["vocal", "voice", "speech"]):
            vocals = f
            _log.debug("  → Vocals: %s", f.name)
            continue

    # Fallback: no suffix recognition possible.
    # IMPORTANT: for PCM WAV the FILE SIZE is content-independent
    # (duration×rate×channels×bit depth) → Vocals and Instrumental are
    # practically the same size, so size is a coin flip as a criterion.
    # Instead, distinguish by mean volume level (RMS): the Vocals stem
    # typically has the higher mean level for speech. As a last resort,
    # deterministically alphabetical (reproducible instead of random).
    if not vocals and not inst and len(wavs) >= 2:
        two = sorted(wavs, key=lambda f: f.name)[:2]
        vols = [(f, _mean_volume_db(f)) for f in two]
        if all(v is not None for _, v in vols) and abs(vols[0][1] - vols[1][1]) > 0.5:
            vols.sort(key=lambda x: x[1], reverse=True)  # louder = Vocals
            vocals, inst = vols[0][0], vols[1][0]
            _log.warning(
                "  → Fallback (RMS): Vocals=%s (%.1f dB), Inst=%s (%.1f dB)",
                vols[0][0].name,
                vols[0][1],
                vols[1][0].name,
                vols[1][1],
            )
        else:
            vocals, inst = two[0], two[1]
            _log.warning(
                "  → Fallback (alphabetical, uncertain): Vocals=%s, Inst=%s", vocals.name, inst.name
            )
    elif not vocals and wavs:
        # inst may already be set → do NOT pick the same file as Vocals
        # (otherwise vocals == inst).
        candidates = [f for f in wavs if f != inst]
        if candidates:
            vocals = candidates[0]
            _log.debug("  → Single fallback: Vocals=%s", vocals.name)

    _log.debug(
        "find_output_wavs result: vocals=%s, inst=%s",
        vocals.name if vocals else None,
        inst.name if inst else None,
    )

    return vocals, inst

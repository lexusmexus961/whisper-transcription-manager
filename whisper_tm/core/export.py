"""
Export utilities: generate SRT/TXT/JSON files alongside videos.
For Plex, Jellyfin, and other media servers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from whisper_tm.config import config

logger = logging.getLogger(__name__)


def export_alongside(source_path: Path, transcript_data: dict, force: bool = False) -> list[str]:
    """Export transcript as SRT/TXT/JSON alongside the source file.

    Args:
        source_path: Path to the original media file
        transcript_data: Transcript dict with segments
        force: Skip config check (per-job override)
    Returns:
        List of generated file paths
    """
    if not force and not config.get("transcription", "export_alongside", default=False):
        return []

    raw_formats = config.get("transcription", "export_alongside_formats", default=["srt"])
    # Normalise: config might store as comma-separated string or list
    if isinstance(raw_formats, str):
        formats = [f.strip() for f in raw_formats.split(",") if f.strip()]
    else:
        formats = list(raw_formats) if raw_formats else ["srt"]
    if not formats:
        return []

    stem = source_path.stem
    parent = source_path.parent
    segments = transcript_data.get("segments", [])
    created = []

    for fmt in formats:
        fmt = fmt.lower().strip()
        try:
            if fmt == "srt":
                fp = parent / f"{stem}.srt"
                _atomic_write_text(fp, _to_srt(segments))
                created.append(str(fp))
            elif fmt == "txt":
                fp = parent / f"{stem}.txt"
                _atomic_write_text(fp, _to_txt(segments))
                created.append(str(fp))
            elif fmt == "json":
                fp = parent / f"{stem}.json"
                _atomic_write_text(
                    fp,
                    json.dumps(transcript_data, ensure_ascii=False, indent=2),
                )
                created.append(str(fp))
        except Exception as e:
            # Was debug-only before -> write errors went unnoticed ("Export empty").
            logger.warning("Export format '%s' failed: %s", fmt, e)

    return created


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically: temp file in same directory + os.replace.

    Prevents partially written/corrupted sidecars if the write operation
    is interrupted (crash, full disk). os.replace is atomic as long as the
    temp file is in the same directory (= same filesystem).

    Note (#028): A foreign-file guard (don't overwrite pre-existing, non-self-
    generated files) is intentionally NOT implemented here. Sidecars sit without
    origin markers directly next to the media file ("film.srt"), so a user-
    provided file cannot be reliably distinguished from ours. Unconditional
    overwriting = previous re-export behavior -> DEFERRED.
    """
    # dot-prefix + .tmp-suffix: unobtrusive for media scanner until os.replace.
    #
    # The prefix must NOT contain the full filename: mkstemp will append
    # random characters + ".tmp", and with already long names from nested
    # recording directories, that exceeds the filesystem's 255-byte limit
    # (OSError 36) — export would fail only at the very end. A short hash
    # over the target name is equally unique.
    tag = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".wtm-{tag}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        # mkstemp creates 0600; sidecars should remain readable as before
        # (media servers may run under different users).
        try:
            os.chmod(tmp, 0o644)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _to_srt(segments: list[dict]) -> str:
    """Convert segments to SRT format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start_s = seg.get("start", 0)
        end_s = seg.get("end", 0)
        # #031: end must not be before start, otherwise invalid time range.
        if end_s < start_s:
            end_s = start_s
        start = _srt_time(start_s)
        end = _srt_time(end_s)
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker", "")
        prefix = f"[{speaker}] " if speaker and speaker != "—" else ""
        lines.append(f"{i}\n{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(lines)


def _to_txt(segments: list[dict]) -> str:
    """Export segments as readable text."""
    lines = []
    current_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "—")
        text = (seg.get("text") or "").strip()
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"[{speaker}]")
            current_speaker = speaker
        lines.append(text)
    return "\n".join(lines)


def _srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    # #031: clamp negative values to 0, otherwise invalid timecodes are created.
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

"""
Regression: Very long file names must not break saving.

Background: Transcription ran completely through (minutes of GPU time) and
failed ONLY when writing the transcript with
    OSError: [Errno 36] File name too long
because the entire job_id was appended to the already long target filename.
All the work was lost.

The names here are shaped like a real error case (nested recording folders,
accented characters → 2 bytes per character).
"""

import os

import pytest

from whisper_tm.core.helpers import transcript_path_for

# ext4/XFS allow 255 BYTES per filename component (not characters!)
NAME_MAX = 255

# Shaped like the real case
LONG_REL = (
    "Recordings Archive etc/Capture4K/My Café Résumé Adventure/"
    "2024-03-01 10-00-00 - My Café Résumé Adventure - A Naïve Piñata Session/"
    "My Café Résumé Adventure - A Naïve Piñata Session (Live Commentary)_#1.m4a"
)


@pytest.fixture
def tdir(tmp_path, monkeypatch):
    from whisper_tm.config import config

    d = tmp_path / "transcripts"
    d.mkdir()
    monkeypatch.setattr(type(config), "transcript_dir", property(lambda self: d))
    return d


def _fits(name: str) -> bool:
    return len(name.encode("utf-8")) <= NAME_MAX


def test_transcript_name_fits_filesystem_limit(tdir):
    tp = transcript_path_for(LONG_REL)
    assert _fits(tp.name), f"{len(tp.name.encode())} Bytes: {tp.name}"


def test_temp_name_fits_and_is_actually_writable(tdir):
    """The temp name of atomic writing must also fit.

    That's exactly where the error was: tp.name (up to 240 bytes) + "." + job_id
    (~200 characters) + ".tmp" → well over 255 bytes.
    """
    import hashlib

    tp = transcript_path_for(LONG_REL)

    # This is how transcription.py builds the temp name today
    tmp_id = hashlib.sha256(tp.name.encode("utf-8")).hexdigest()[:16]
    tmp = tp.with_name(f".wtm-{tmp_id}.tmp")

    assert _fits(tmp.name), f"{len(tmp.name.encode())} Bytes: {tmp.name}"

    # And the decisive proof: it can actually be written.
    # Create the target folder like transcription.py does — when
    # transcript_mode "alongside" it's in the media file's folder.
    tp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("{}", encoding="utf-8")
    os.replace(tmp, tp)
    assert tp.exists()


def test_old_scheme_would_have_failed(tdir):
    """Proves the old temp name actually exceeded the limit."""
    job_id = LONG_REL.replace(" ", "_").replace(".", "_").replace("/", "_") + "_4221af2c"
    tp = transcript_path_for(LONG_REL)
    old_tmp_name = tp.name + f".{job_id}.tmp"

    assert not _fits(old_tmp_name), "test case no longer hits the error"
    with pytest.raises(OSError):
        (tp.parent / old_tmp_name).write_text("{}", encoding="utf-8")


def test_export_sidecar_tmp_name_fits(tmp_path):
    """The same error was in sidecar export (core/export.py)."""
    from whisper_tm.core.export import _atomic_write_text as _atomic_write

    # Target name just under the limit (umlauts = 2 bytes) — the old temp name
    # ("." + full target name + random + ".tmp") would certainly exceed it.
    long_name = "Ä" * 105 + "_long_recording_with_umlauts" + ".srt"
    target = tmp_path / long_name
    assert _fits(target.name), f"{len(target.name.encode())} Bytes"
    assert not _fits("." + target.name + ".XXXXXXXX.tmp"), "old temp name fit anyway"

    _atomic_write(target, "1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    assert target.read_text(encoding="utf-8").startswith("1")

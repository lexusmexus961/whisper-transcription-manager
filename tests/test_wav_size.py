"""
WAV has a 32-bit size field and is thus limited to 4 GiB.

At 48 kHz stereo s16 (192 kB/s) the limit is reached after **6.21 h**, at
16 kHz mono (32 kB/s) after 37.3 h. Beyond that, ffmpeg writes a file with
an overflowed header and even warns about it:

    Filesize 4800000000 invalid for wav, output file will be broken

The tricky part: ``ffprobe`` reads such a file to the end and reports the
correct duration — ``soundfile`` reads the header and reports **22,369.62 s**
(= exactly 4 GiB). The pipeline uses both methods, depending on the step.
Consequences on a recording past the limit: shifted timestamps, minutes
missing at the end, instrumental track unusable.

The countermeasure is ``-rf64 auto``: below the limit the file stays a
normal RIFF/WAV (thus fully compatible), above it an RF64 header is written,
which ffmpeg, soundfile and pydub can read.

These tests verify that the pipeline's write points set the flag. They do
**not** generate 4-GB files — that would take too long for a suite; here we
ensure the commands carry the flag and that a small RF64 is understood by
all readers.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")

# limits where WAV breaks without RF64
LIMIT_BYTES = 2**32
BYTES_PER_S_48K_STEREO = 48000 * 2 * 2
BYTES_PER_S_16K_MONO = 16000 * 1 * 2


def _flat(text: str) -> str:
    """Source text with the formatter's line breaks taken out: one space
    for every run of whitespace, none inside brackets, no trailing commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*([)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+([)\]}])", r"\1", text)


def test_the_limits_match_the_documentation():
    """Verify the numbers in the code comments."""
    hours_48k_stereo = LIMIT_BYTES / BYTES_PER_S_48K_STEREO / 3600
    hours_16k_mono = LIMIT_BYTES / BYTES_PER_S_16K_MONO / 3600
    assert hours_48k_stereo == pytest.approx(6.21, abs=0.01)
    assert hours_16k_mono == pytest.approx(37.3, abs=0.1)


# ── The pipeline commands carry the flag ───────────────────────


def _source_text(module: str) -> str:
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "whisper_tm" / "core" / module
    return _flat(p.read_text(encoding="utf-8"))


def test_concat_writes_rf64():
    """_concat_chunks concatenates all separation chunks — this is where
    the 4 GiB actually tore (48 kHz stereo)."""
    src = _source_text("vocal_isolation.py")
    block = src[src.index('"-f", "concat"') :]
    block = block[: block.index("], job=job")]
    assert '"-rf64", "auto"' in block, (
        "Concat writes without -rf64: files over 6.21 h get an overflowed WAV header."
    )


def test_audio_extraction_writes_rf64():
    """extract_audio pulls the complete audio track (16 kHz mono)."""
    src = _source_text("audio_utils.py")
    block = src[src.index('"-vn", "-map", "0:a:0"') :]
    block = block[: block.index("]", block.index("str(output_path)"))]
    assert '"-rf64", "auto"' in block


def test_downsampling_writes_rf64():
    """Vocals → 16 kHz for WhisperX, also over the full length."""
    src = _source_text("transcription.py")
    block = src[: src.index("str(wav_clean)")]
    block = block[block.rindex("run_cancellable([") :]
    assert '"-rf64", "auto"' in block, (
        "The vocals downsample writes without -rf64: files over 37.3 h get "
        "an overflowed WAV header."
    )


# ── An RF64 must be understood by all readers in the pipeline ────────


@pytest.fixture
def rf64_file(tmp_path):
    """Small but real RF64 (forced) — tests the readers, not the size."""
    p = tmp_path / "small.wav"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "2",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            "-rf64",
            "always",
            str(p),
            "-loglevel",
            "error",
        ],
        check=True,
    )
    assert p.read_bytes()[:4] == b"RF64", "ffmpeg did not write RF64"
    return p


def test_ffprobe_reads_rf64(rf64_file):
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(rf64_file),
        ],
        capture_output=True,
        text=True,
    )
    assert float(r.stdout.strip()) == pytest.approx(2.0, abs=0.05)


def test_soundfile_reads_rf64(rf64_file):
    """soundfile is the reader that reported 22,369 s for the broken WAV."""
    sf = pytest.importorskip("soundfile")
    assert sf.info(str(rf64_file)).duration == pytest.approx(2.0, abs=0.05)


def test_pydub_reads_rf64(rf64_file):
    """audio-separator writes and reads via pydub."""
    pydub = pytest.importorskip("pydub")
    seg = pydub.AudioSegment.from_file(str(rf64_file))
    assert len(seg) / 1000 == pytest.approx(2.0, abs=0.05)


def test_auto_stays_a_normal_wav_below_the_limit(tmp_path):
    """The most important compatibility point: for short files — i.e. the
    vast majority — -rf64 auto changes nothing. The header stays RIFF,
    every tool reads it as before."""
    p = tmp_path / "short.wav"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "2",
            "-acodec",
            "pcm_s16le",
            "-rf64",
            "auto",
            str(p),
            "-loglevel",
            "error",
        ],
        check=True,
    )
    assert p.read_bytes()[:4] == b"RIFF", (
        "-rf64 auto must not switch small files to RF64 — that would "
        "lose compatibility for no reason."
    )


def test_a_corrupt_files_reason_reads_plainly():
    from whisper_tm.core.audio_utils import plain_reason

    stderr = (
        "[mov,mp4,m4a,3gp,3g2,mj2 @ 0x55587893f0c0] moov atom not found\n"
        "/srv/demo/.cache/whisper-tm/work/job_1/team-call.mp4: Invalid data found when processing input\n"
    )
    assert plain_reason(stderr) == "moov atom not found; Invalid data found when processing input"

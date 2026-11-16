"""A chunk ffmpeg fails to read is an error, not silence.

Background: _extract_sep_chunks replaced ANY failed chunk extraction
(``returncode != 0``) with silence of the chunk's length — meant for the
legitimately empty stretch where a recording's audio track ends early, for
which ffmpeg exits 0 with an empty file ("Output file is empty"). A real
read error then removed up to 30 minutes from the transcript, logged as an
ordinary line, and the job succeeded.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from whisper_tm.core import vocal_isolation as vi

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


@pytest.fixture
def source(tmp_path, monkeypatch):
    src = tmp_path / "talk.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(src),
        ],
        check=True,
        timeout=60,
    )
    return src


def _failing_on(n, real):
    seen = {"i": 0}

    def run(cmd, job=None, timeout=0, label=None):
        if label and label.startswith("Sep chunk extraction"):
            seen["i"] += 1
            if seen["i"] == n:
                return subprocess.CompletedProcess(cmd, 1, "", "Input/output error")
        return real(cmd, job=job, timeout=timeout, label=label)

    return run


def test_a_failed_chunk_extraction_raises(source, tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "run_cancellable", _failing_on(2, vi.run_cancellable))
    with pytest.raises(RuntimeError, match="Chunk 2/3"):
        vi._extract_sep_chunks(source, tmp_path, 2, 0.5)


def test_an_empty_stretch_still_becomes_silence(tmp_path):
    # Video 6 s, audio only 2 s: chunks 2 and 3 are legitimately empty
    src = tmp_path / "v.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=d=6:s=64x48",
            "-f",
            "lavfi",
            "-i",
            "sine=d=2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            str(src),
        ],
        check=True,
        timeout=60,
    )
    chunks, total, silence = vi._extract_sep_chunks(src, tmp_path, 2, 0.0)
    assert len(chunks) == 3 and silence == {1, 2}

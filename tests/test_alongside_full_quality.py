"""Tracks kept next to the file are in full quality — the vocals too.

Vocal tracks saved next to the file must keep the full quality. A
``<name>_vocals.mp3`` used to come out at 16 kHz mono and 32 kbit/s beside a
44.1-kHz stereo instrumental: the separation writes its vocals as 16 kHz
mono, all Whisper needs — and that was the track saved.

What these tests pin, with real ffmpeg and a stand-in for the separator:

* without a listener the vocals stay 16 kHz mono (the intermediate file of
  a long recording is about a sixth of the full-quality size);
* with ``full_vocals`` they are in the format of the instrumental;
* the worker asks for that when the tracks go next to the file, or are the
  whole result; the MP3s there are VBR V0, keeping rate and channels.
"""

from __future__ import annotations

import inspect
import re
import shutil
import subprocess

import pytest

from whisper_tm.core import transcription as tr
from whisper_tm.core import vocal_isolation as vi

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available"
)


def _flat(text: str) -> str:
    """Source text with the formatter's line breaks taken out: one space
    for every run of whitespace, none inside brackets, no trailing commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*([)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+([)\]}])", r"\1", text)


def _probe(path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    rate, ch = out.split(",")
    return int(rate), int(ch)


@pytest.fixture
def separation(tmp_path, monkeypatch):
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
            "44100",
            "-ac",
            "2",
            str(src),
        ],
        check=True,
        timeout=60,
    )

    def separator(chunk, out_dir, model_file, job=None, single_stem=None, retry=0):
        # Both stems the chunk itself — the format is what matters here
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(chunk, out_dir / "chunk_(Vocals).wav")
        shutil.copy(chunk, out_dir / "chunk_(Instrumental).wav")

    monkeypatch.setattr(vi, "_run_separator", separator)
    monkeypatch.setattr(vi, "_chunk_prefilter", lambda *a, **k: "separate")
    # Chunks of 2 s: a file shorter than one chunk is separated in one piece
    # and keeps its format anyway — the 16 kHz came with the chunked path
    # every recording over 30 minutes takes
    real_get = vi.config.get
    monkeypatch.setattr(
        vi.config,
        "get",
        lambda section, key, default=None: (
            2
            if (section, key) == ("hardware", "sep_chunk_duration_sec")
            else 0.5
            if (section, key) == ("hardware", "sep_overlap_sec")
            else real_get(section, key, default=default)
        ),
    )
    for d in "abc":
        (tmp_path / d).mkdir()  # as isolate_vocals makes its folder
    return src


def test_for_whisper_alone_the_vocals_stay_small(separation, tmp_path):
    vocals, inst = vi.chunked_separate(separation, tmp_path / "a")
    assert _probe(vocals) == (16000, 1)
    assert _probe(inst) == (48000, 2)


def test_to_listen_to_they_are_like_the_instrumental(separation, tmp_path):
    vocals, inst = vi.chunked_separate(separation, tmp_path / "b", full_vocals=True)
    assert _probe(vocals) == _probe(inst) == (48000, 2)


def test_the_mp3s_next_to_the_file_keep_it(separation, tmp_path, monkeypatch):
    class _Jobs:
        def log(self, *a, **k):
            pass

        def warn(self, job, text):
            pass  # same stems: "nearly the same size"

    monkeypatch.setattr(tr, "job_manager", _Jobs())
    monkeypatch.setattr("whisper_tm.core.autoscan.note_made", lambda *a: None)
    vocals, inst = vi.chunked_separate(separation, tmp_path / "c", full_vocals=True)
    tr._save_audio_alongside(separation, vocals, inst, job=None)
    v = tmp_path / "talk_vocals.mp3"
    i = tmp_path / "talk_instrumental.mp3"
    assert _probe(v) == _probe(i) == (48000, 2)
    assert tr.ALONGSIDE_MP3 == ["-codec:a", "libmp3lame", "-q:a", "0"]


def test_the_worker_asks_for_full_vocals_when_they_are_kept():
    source = inspect.getsource(tr.transcription_worker)
    call = source[source.index("isolate_vocals(") :]
    call = call[: call.index(")\n")]
    assert "full_vocals=bool(save_audio_alongside or separation_only)" in call
    # and Whisper still gets 16 kHz from the worker's own conversion
    assert '"-ar", "16000", "-ac", "1"' in _flat(source[source.index("Vocals → 16kHz mono WAV") :])

"""The instrumental track keeps its timeline when chunks differ in kind.

Background: audio-separator writes its stems at 44.1 kHz, the chunks the
prefilter skips (silence, clean speech) got a 48-kHz silence placeholder —
and ffmpeg's concat demuxer does not resample. Every 48-kHz piece played
8.8 % too long: the instrumental saved next to the file drifted away from
the video by ~158 s per skipped 30-minute chunk. Real ffmpeg below, with a
stand-in separator that writes 44.1 kHz like the real one.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from whisper_tm.core import vocal_isolation as vi

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available"
)


def _probe(path) -> tuple[int, int, float]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    import json

    d = json.loads(out)
    s = d["streams"][0]
    return int(s["sample_rate"]), int(s["channels"]), float(d["format"]["duration"])


@pytest.fixture
def mixed(tmp_path, monkeypatch):
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
            "sine=frequency=440:duration=12",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(src),
        ],
        check=True,
        timeout=60,
    )

    def separator(chunk, out_dir, model_file, job=None, single_stem=None, retry=0):
        # Like audio-separator: stems at 44.1 kHz stereo
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in ("Vocals", "Instrumental"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(chunk),
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    str(out_dir / f"chunk_({stem}).wav"),
                ],
                check=True,
                timeout=60,
            )

    calls = iter(["separate", "silence", "clean", "separate", "silence", "separate"])
    monkeypatch.setattr(vi, "_run_separator", separator)
    monkeypatch.setattr(vi, "_chunk_prefilter", lambda *a, **k: next(calls))
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
    (tmp_path / "w").mkdir()
    return src, tmp_path / "w"


def test_instrumental_is_one_rate_and_as_long_as_the_source(mixed):
    src, work = mixed
    vocals, inst = vi.chunked_separate(src, work, full_vocals=True)
    rate, ch, dur = _probe(inst)
    assert (rate, ch) == (48000, 2)
    assert abs(dur - 12.0) < 0.1, f"instrumental is {dur:.2f} s, the source 12 s"
    v_rate, _v_ch, v_dur = _probe(vocals)
    assert v_rate == 48000 and abs(v_dur - 12.0) < 0.1


# ── _trim_chunk never hands back the separator's own file ────────────────
# The caller deletes the separator's output folder right after trimming
# (shutil.rmtree(chunk_out)); a returned path into it pointed at nothing
# by the time the concat ran (e.g. with sep_overlap_sec: 0).


def _tone(path, secs=2.0):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={secs}",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        timeout=60,
    )


def test_trim_without_work_still_writes_the_output(tmp_path):
    stem = tmp_path / "chunk_out" / "x_(Instrumental).wav"
    stem.parent.mkdir()
    _tone(stem)
    out = tmp_path / "trimmed.wav"
    got = vi._trim_chunk(stem, out, 0.0, 0.0, 2.0)
    shutil.rmtree(stem.parent)
    assert got == out and out.exists()


def test_trim_longer_than_the_chunk_is_an_error(tmp_path):
    stem = tmp_path / "s.wav"
    _tone(stem, 1.0)
    with pytest.raises(RuntimeError):
        vi._trim_chunk(stem, tmp_path / "o.wav", 0.6, 0.6, 1.0)

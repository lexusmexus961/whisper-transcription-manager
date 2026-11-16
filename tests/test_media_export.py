"""Video and audio export (whisper_tm/core/media_export.py) on media made
here with ffmpeg: formats, sizes, parts, subtitles, cancel."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from whisper_tm.core import media_export as mx

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _make(tmp: Path) -> tuple[Path, Path]:
    video = tmp / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=640x360:r=15:d=12",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=12",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    audio = tmp / "memo.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=f=300:d=10",
            "-c:a",
            "aac",
            str(audio),
        ],
        check=True,
    )
    return video, audio


def _info(p: Path) -> dict:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,height,channels,sample_rate:format=duration",
            "-of",
            "json",
            str(p),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def _dur(p: Path) -> float:
    return float(_info(p)["format"]["duration"])


SEGS = [
    {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
    {"start": 3.2, "end": 4.0, "speaker": "SPEAKER_00"},
    {"start": 7.0, "end": 9.0, "speaker": "SPEAKER_01"},
]


def test_spans_join_short_gaps_and_pick_speakers():
    assert mx.spans(SEGS, pad=0) == [(1.0, 4.0), (7.0, 9.0)]
    assert mx.spans(SEGS, ["Alice"], {"SPEAKER_01": "Alice"}, pad=0) == [(7.0, 9.0)]


def test_options_are_checked():
    with pytest.raises(mx.MediaExportError):
        mx.video_options({"format": "avi"})
    with pytest.raises(mx.MediaExportError):
        mx.audio_options({"sample_rate": 12345})
    assert mx.video_options({"size": 720})["size"] == "720"


def test_video_mp4_smaller_with_parts_and_subtitles(tmp_path):
    video, _ = _make(tmp_path)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nHello there\n")
    o = mx.video_options({"format": "mp4", "size": "480", "subtitles": True})
    parts = mx.spans(SEGS, pad=0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = mx.run(mx.plan_video(video, out_dir, o, parts, srt))
    info = _info(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v["codec_name"] == "h264" and v["height"] == 360, "never scaled up"
    assert abs(_dur(out) - 5.0) < 0.6, "only the 5 s of speech"


def test_video_webm_720_and_the_original_unchanged(tmp_path):
    video, _ = _make(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    webm = mx.run(
        mx.plan_video(
            video, out_dir, mx.video_options({"format": "webm", "size": "720"}), None, None
        )
    )
    assert {s["codec_name"] for s in _info(webm)["streams"]} == {"vp9", "opus"}
    orig = mx.run(
        mx.plan_video(video, out_dir, mx.video_options({"format": "original"}), None, None)
    )
    assert orig.read_bytes() == video.read_bytes()


def test_audio_only_recordings_are_refused_for_video(tmp_path):
    _, audio = _make(tmp_path)
    with pytest.raises(mx.MediaExportError, match="no video"):
        mx.plan_video(audio, tmp_path, mx.video_options({}), None, None)


@pytest.mark.parametrize(
    "fmt, codec",
    [
        ("mp3", "mp3"),
        ("wav", "pcm_s16le"),
        ("flac", "flac"),
        ("m4a", "aac"),
        ("opus", "opus"),
        ("track", "aac"),
    ],
)
def test_audio_formats(tmp_path, fmt, codec):
    video, _ = _make(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = mx.run(mx.plan_audio(video, out_dir, mx.audio_options({"format": fmt}), None))
    info = _info(out)
    assert [s["codec_name"] for s in info["streams"]] == [codec] and abs(_dur(out) - 12) < 0.5


def test_audio_mono_resampled_without_pauses(tmp_path):
    video, _ = _make(tmp_path)
    o = mx.audio_options(
        {"format": "wav", "mono": True, "sample_rate": 16000, "remove_pauses": True}
    )
    out = mx.run(mx.plan_audio(video, tmp_path, o, mx.spans(SEGS, pad=0)))
    a = _info(out)["streams"][0]
    assert a["channels"] == 1 and a["sample_rate"] == "16000" and abs(_dur(out) - 5.0) < 0.3


def test_cancel_removes_the_half_written_file(tmp_path):
    video, _ = _make(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    plan = mx.plan_video(video, out_dir, mx.video_options({"format": "webm"}), None, None)
    with pytest.raises(mx.MediaExportError, match="cancelled"):
        mx.run(plan, should_stop=lambda: True)
    assert not plan.out.exists()


# ── A recording's name inside a filter graph ──────────────────────────────
#
# The subtitle file is named after the recording, and its path goes into
# ffmpeg's filter graph. Escaped for one of ffmpeg's two parsers only, an
# apostrophe ended the quote: "Don't stop.mp4" could not get subtitles, and
# what followed the apostrophe was read as filters of its own.

SRT = "1\n00:00:01,000 --> 00:00:03,000\nHello there\n"
OPTS = {"format": "mp4", "size": "480", "subtitles": True}


@pytest.mark.parametrize(
    "stem",
    ["Don't stop", "a=b;c,d[e]", "It's 50%: 'half' off", "back\\slash", "Привет 한국어 – ’quotes’"],
)
def test_subtitles_burn_in_whatever_the_recording_is_called(tmp_path, stem):
    video, _ = _make(tmp_path)
    src = video.rename(tmp_path / f"{stem}.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    srt = out_dir / f"{stem}.srt"
    srt.write_text(SRT, encoding="utf-8")
    made = mx.run(mx.plan_video(src, out_dir, mx.video_options(OPTS), None, srt))
    assert made.name == f"{stem}.mp4" and _info(made)["streams"]


def test_a_name_cannot_add_filters_of_its_own(tmp_path, monkeypatch):
    # The name closes the quote and adds a filter that writes a file
    video, _ = _make(tmp_path)
    stem = "x':metadata=mode=print:file=owned.txt,drawtext=text='y"
    src = video.rename(tmp_path / f"{stem}.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    srt = out_dir / f"{stem}.srt"
    srt.write_text(SRT, encoding="utf-8")
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    made = mx.run(mx.plan_video(src, out_dir, mx.video_options(OPTS), None, srt))
    assert made.exists()
    assert not list(tmp_path.rglob("owned.txt"))
    assert not any(elsewhere.iterdir())


def test_the_filter_path_survives_both_of_ffmpegs_parsers():
    from whisper_tm.core.audio_utils import filter_path

    assert filter_path("/a/plain.srt") == "filename='/a/plain.srt'"
    assert filter_path("/a/Don't.srt") == "filename='/a/Don\\'\\''t.srt'"
    assert filter_path("C:\\x\\y.srt") == "filename='C\\:\\\\x\\\\y.srt'"

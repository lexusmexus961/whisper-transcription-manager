"""
Files whose audio track ends before the video.

Example — a VLC recording of a DVD:

    dvd_recording.3gp
      Video 8100.0 s
      Audio 5400.0 s      ← ends 45 minutes early

``get_duration()`` returns the container duration, based on the
LONGEST stream. Speaker separation cut five chunks from it; the last one lay
completely after the audio end and returned empty:

    ffmpeg: Output file is empty, nothing was encoded
    → Chunk 5/5 extraction failed

The entire run broke, even though 90 minutes of audio were cleanly extracted. The
file played without problems — it wasn't broken.

Two levels of fixes:
  1. Division is based on the AUDIO TRACK (``get_audio_duration``).
  2. Chunks without audio no longer break the run; instead they are filled with
     silence of exactly the same length. This also covers the more common
     case where audio only exists MIDWAY through the file (DVD/TV recordings
     where recording starts before the broadcast): there the first
     AND last chunks are empty. Omitting them would shift all following timestamps
     forward — silence preserves the time axis.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_tm.core import vocal_isolation as vi


class TestDivisionCoversFullFile:
    def test_chunks_cover_container_duration(self, tmp_path, monkeypatch):
        """Division is by CONTAINER duration, not by audio track.

        Dividing by audio track would be shorter and not address empty chunks
        at all — but would rely on a metadata value. If it reports too little,
        the remainder would be silently cut off and the job still counted as
        successful. A transcript that's too short and nobody notices
        is worse than a failure.
        """
        called = []

        def ffmpeg_dummy(cmd, **kw):
            # read -ss and -t from the call
            ss = float(cmd[cmd.index("-ss") + 1]) if "-ss" in cmd else 0.0
            called.append(ss)
            target = __import__("pathlib").Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * 5000)  # "has content"

            class R:
                returncode = 0
                stderr = ""

            return R()

        monkeypatch.setattr(vi, "get_duration", lambda p: 8100.0)
        monkeypatch.setattr(vi, "get_audio_duration", lambda p: 5400.0)
        monkeypatch.setattr(vi, "run_cancellable", ffmpeg_dummy)

        source = tmp_path / "film.3gp"
        source.write_bytes(b"x")
        chunks, duration, _silence = vi._extract_sep_chunks(
            source, tmp_path, chunk_dur=1800.0, overlap=6.0
        )

        assert duration == pytest.approx(8100.0), "nothing may be cut off"
        assert len(chunks) == 5


class TestGaps:
    """Chunks without audio are filled with silence, not discarded.

    A discarded chunk would shift all following timestamps forward when
    merging — transcript and video would drift apart.
    Silence of equal length preserves the time axis.
    """

    def _mock(self, empty_at):
        """ffmpeg mock: extraction provides nothing for `empty_at`,
        silence generation (anullsrc) always provides something."""
        counter = {"i": 0}

        def f(cmd, **kw):
            target = Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            is_silence = "lavfi" in cmd
            if is_silence:
                target.write_bytes(b"\x00" * 5000)
            else:
                i = counter["i"]
                counter["i"] += 1
                target.write_bytes(b"" if i in empty_at else b"\x00" * 5000)

            class R:
                returncode = 0
                stderr = "Output file is empty, nothing was encoded"

            return R()

        return f

    def test_gap_in_middle_preserves_timeline(self, tmp_path, monkeypatch):
        """The case "audio only in the middle": empty outside, audio inside."""
        monkeypatch.setattr(vi, "get_duration", lambda p: 9000.0)
        monkeypatch.setattr(vi, "get_audio_duration", lambda p: 9000.0)
        monkeypatch.setattr(vi, "run_cancellable", self._mock(empty_at={0, 1, 4}))

        source = tmp_path / "film.3gp"
        source.write_bytes(b"x")
        chunks, _, _silence = vi._extract_sep_chunks(
            source, tmp_path, chunk_dur=1800.0, overlap=6.0
        )

        assert len(chunks) == 5, "no chunk may be lost — otherwise desync"
        assert all(c[0].exists() for c in chunks)
        assert _silence == {0, 1, 4}, "silence chunks must be reported"

    def test_trailing_gap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vi, "get_duration", lambda p: 9000.0)
        monkeypatch.setattr(vi, "get_audio_duration", lambda p: 9000.0)
        monkeypatch.setattr(vi, "run_cancellable", self._mock(empty_at={3, 4}))

        source = tmp_path / "film.3gp"
        source.write_bytes(b"x")
        chunks, _, _silence = vi._extract_sep_chunks(
            source, tmp_path, chunk_dur=1800.0, overlap=6.0
        )
        assert len(chunks) == 5

    def test_no_audio_at_all_remains_error(self, tmp_path, monkeypatch):
        """If NOTHING carries audio, it must fail — otherwise a pure
        silence track would pass as a successful transcription."""
        monkeypatch.setattr(vi, "get_duration", lambda p: 9000.0)
        monkeypatch.setattr(vi, "get_audio_duration", lambda p: 9000.0)
        monkeypatch.setattr(vi, "run_cancellable", self._mock(empty_at=set(range(10))))

        source = tmp_path / "film.3gp"
        source.write_bytes(b"x")
        with pytest.raises(RuntimeError, match="No audio"):
            vi._extract_sep_chunks(source, tmp_path, chunk_dur=1800.0, overlap=6.0)


class TestGetAudioDuration:
    def test_falls_back_to_container(self, tmp_path, monkeypatch):
        """If audio stream reports no duration, container duration applies."""
        from whisper_tm.core import audio_utils

        class R:
            returncode = 0
            stdout = '{"streams": [{}]}'
            stderr = ""

        monkeypatch.setattr(audio_utils, "_run_probe", lambda *a, **k: R())
        monkeypatch.setattr(audio_utils, "get_duration", lambda p: 123.0)
        assert audio_utils.get_audio_duration(tmp_path / "x.mp4") == 123.0

    def test_prefers_the_audio_track(self, tmp_path, monkeypatch):
        from whisper_tm.core import audio_utils

        class R:
            returncode = 0
            stdout = '{"streams": [{"duration": "5400.0"}]}'
            stderr = ""

        monkeypatch.setattr(audio_utils, "_run_probe", lambda *a, **k: R())
        monkeypatch.setattr(audio_utils, "get_duration", lambda p: 8100.0)
        assert audio_utils.get_audio_duration(tmp_path / "x.mp4") == pytest.approx(5400.0)

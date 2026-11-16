"""Single-pass provisioning: pre-check and honest error messages.

Background: many queue entries failed with one and the same
message "Single-pass provisioning failed … check container/codec".
Behind it were four different causes, most common first:

    file has no audio track at all (videos without sound; photos named .MOV)
    file is 0 bytes in size
    ffmpeg ran through, but the durations of the two outputs don't match
    container broken (moov atom not found)

None of these was a bug in the single-pass command. The bug was that the
fast path skipped the pre-check that the slow path (extract_audio) has
always done — and that the summary message did not name any of the four
cases. Without the check, the single pass first read the entire file from
the NAS, only to fail for certain afterwards; without the distinction,
everything landed in the retry bucket instead of the skiplist and kept
coming back forever.
"""

import subprocess
from pathlib import Path

import pytest

from whisper_tm.core.audio_utils import CorruptMediaError, has_audio_stream


def _ffmpeg_present() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(not _ffmpeg_present(), reason="ffmpeg missing")


def _make_video_without_audio(target: Path) -> Path:
    """Short video with no audio track at all — the most common of the four cases."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            "-c:v",
            "libx264",
            "-t",
            "1",
            str(target),
        ],
        capture_output=True,
        timeout=60,
        check=True,
    )
    return target


def _make_tone(target: Path, duration: float = 1.0) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            str(target),
        ],
        capture_output=True,
        timeout=60,
        check=True,
    )
    return target


class TestThePrecheck:
    def test_a_video_without_an_audio_track_is_detected(self, tmp_path):
        """The most common scenario: no audio stream, so False instead of an exception."""
        v = _make_video_without_audio(tmp_path / "muted.mp4")
        assert has_audio_stream(v) is False

    def test_a_file_with_audio_gets_through(self, tmp_path):
        assert has_audio_stream(_make_tone(tmp_path / "tone.wav")) is True

    def test_an_empty_file_counts_as_broken(self, tmp_path):
        """0 bytes: ffprobe reports "End of file" — this is a skiplist case.

        It's important that CorruptMediaError comes back here, not
        RuntimeError: only the definitive exception may lock the file out
        permanently. A transient error (NAS briefly down) must never do that.
        """
        empty = tmp_path / "empty.aac"
        empty.write_bytes(b"")
        with pytest.raises(CorruptMediaError):
            has_audio_stream(empty)

    def test_a_broken_container_counts_as_broken(self, tmp_path):
        """MP4 without a moov atom, like an aborted recording."""
        broken = tmp_path / "cancelled.mp4"
        broken.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 2048)
        with pytest.raises(CorruptMediaError):
            has_audio_stream(broken)

    def test_a_missing_file_is_not_a_skiplist_case(self, tmp_path):
        """Missing ≠ broken. Otherwise a brief NAS outage causes a lockout."""
        with pytest.raises(Exception) as exc:
            has_audio_stream(tmp_path / "does_not_exist.mp4")
        assert not isinstance(exc.value, CorruptMediaError)


class TestOnePassCheckInTheWorker:
    """The single pass checks for the audio track before the expensive NAS read."""

    def test_the_one_pass_calls_has_audio_stream_before_ffmpeg(self):
        import inspect

        from whisper_tm.core import transcription

        source = inspect.getsource(transcription.transcription_worker)
        check_it = source.index("has_audio_stream(source_path)")
        one_pass = source.index("_single_pass_cmd(source_path")
        assert check_it < one_pass, (
            "The audio-track check must come BEFORE the single pass — "
            "otherwise the entire file is read from the NAS before it "
            "turns out it has no sound."
        )

    def test_the_precheck_continues_when_in_doubt(self):
        """An inconclusive ffprobe must not let a job fail.

        The check sits on the NAS path, where it didn't exist before, and
        its ffprobe has 30 s against the single pass's 14400 s. A CIFS
        hiccup would otherwise cause files to fail that used to go through
        fine — the check would then be a regression.
        """
        import inspect

        from whisper_tm.core import transcription

        source = inspect.getsource(transcription.transcription_worker)
        block = source[
            source.index("has_audio_track = has_audio_stream") : source.index(
                "if not has_audio_track"
            )
        ]
        assert "except CorruptMediaError:" in block, "Only the definitive exception may propagate."
        assert "has_audio_track = True" in block, (
            "On an inconclusive sample result, the single pass must still run."
        )

    def test_a_no_audio_message_triggers_the_skiplist(self):
        """The message must match the pattern the skiplist evaluates.

        transcription_worker attaches "no_audio" based on the substring
        "no audio" in the error text. If the message deviates from
        that, the file keeps coming back silently on every run.
        """
        import inspect

        from whisper_tm.core import transcription

        source = inspect.getsource(transcription.transcription_worker)
        assert "no audio" in source
        assert 'add_to_skiplist(filename, "no_audio"' in source


class TestErrorMessageNamesTheReason:
    """Four causes, four messages — no longer one summary message."""

    def test_a_broken_container_becomes_a_skiplist_exception(self):
        """Only CorruptMediaError locks out permanently — this is where it belongs."""
        from whisper_tm.core.transcription import _one_pass_error

        e = _one_pass_error(183, "[mov,mp4] moov atom not found", 0.0, 0.0)
        assert isinstance(e, CorruptMediaError)
        assert "corrupt" in str(e)

    def test_an_ffmpeg_abort_names_the_code_and_the_last_line(self):
        from whisper_tm.core.transcription import _one_pass_error

        e = _one_pass_error(234, "something\nStream map '' matches no streams.", 0.0, 0.0)
        assert isinstance(e, RuntimeError)
        assert not isinstance(e, CorruptMediaError)
        assert "234" in str(e)
        assert "matches no streams" in str(e)

    def test_a_duration_mismatch_is_named(self):
        """rc=0, but the outputs have different lengths."""
        from whisper_tm.core.transcription import _one_pass_error

        e = _one_pass_error(0, "", 99999.9, 0.6)
        assert isinstance(e, RuntimeError)
        assert not isinstance(e, CorruptMediaError)
        assert "99999.9" in str(e)
        assert "0.6" in str(e)
        assert "have different lengths" in str(e)

    def test_without_stderr_no_dangling_colon(self):
        from whisper_tm.core.transcription import _one_pass_error

        assert str(_one_pass_error(1, "", 0.0, 0.0)).endswith("(code 1)")

    def test_every_message_names_the_one_pass(self):
        """So the error can still be matched in the filter."""
        from whisper_tm.core.transcription import _one_pass_error

        for args in ((234, "broken", 0.0, 0.0), (0, "", 5.0, 9.0)):
            assert "Single-pass" in str(_one_pass_error(*args))

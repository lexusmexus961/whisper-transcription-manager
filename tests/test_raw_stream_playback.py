"""Raw elementary streams must be transcoded for playback.

Symptom: a file shows a far longer duration than it actually plays.
The cause wasn't a corrupt file: raw ADTS has neither index nor duration in
the header. When one is needed, it's ESTIMATED from file size ÷ bitrate — and the
bitrate comes only from the first frames (libavformat/demux.c,
estimate_timings_from_bit_rate). With VBR voice notes with quiet start
it's way off: the browser can report many times the real length, and
even ffprobe's format=duration can be too long. So ffprobe's value is
also estimated.

Instead of re-encoding, it's repackaged (`-c:a copy -bsf:a aac_adtstoasc`):
bit-exact, ~5x faster, smaller than original — and without the ~69 ms
encoder offset from libmp3lame, which would pull subtitles and audio apart.
"""

from whisper_tm.core.audio_utils import is_raw_stream as _raw_stream


class TestRawStream:
    def test_adts_aac_is_raw(self):
        assert _raw_stream("aac")

    def test_aac_in_an_mp4_container_is_not_raw(self):
        """Same codec, but with a container — that carries the duration."""
        assert not _raw_stream("mov,mp4,m4a,3gp,3g2,mj2")

    def test_ogg_and_webm_stay_untouched(self):
        assert not _raw_stream("ogg")
        assert not _raw_stream("matroska,webm")

    def test_mp3_is_deliberately_not_covered(self):
        """With a Xing header the duration is correct; preemptively
        transcoding the whole library would be more expensive than the
        problem."""
        assert not _raw_stream("mp3")

    def test_flac_is_not_covered(self):
        """FLAC carries total_samples in the STREAMINFO — duration is known."""
        assert not _raw_stream("flac")

    def test_empty_and_unknown_are_not_raw(self):
        assert not _raw_stream("")
        assert not _raw_stream(None)
        assert not _raw_stream("   ")
        assert not _raw_stream("does-not-exist")

    def test_case_does_not_matter(self):
        assert _raw_stream("AAC")
        assert _raw_stream("  aac  ")


# ── Exact duration: counted instead of estimated ─────────────────────────

import shutil
import subprocess

import pytest

from whisper_tm.core.audio_utils import duration_from_frames, get_duration

pytestmark_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not available"
)


@pytest.fixture
def adts(tmp_path):
    """Raw ADTS with a KNOWN length (30 s) and a quiet, sparse start.

    Two separately encoded parts are simply concatenated — that's
    allowed with ADTS, each frame carries its own header. This
    reproduces exactly the case from practice: the first frames are
    much sparser than the remainder, and a bitrate estimated from them is off.
    """

    def part(name, source, seconds, bitrate):
        target = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
                "-t",
                str(seconds),
                "-c:a",
                "aac",
                "-b:a",
                bitrate,
                "-f",
                "adts",
                str(target),
            ],
            check=True,
            timeout=120,
        )
        return target.read_bytes()

    target = tmp_path / "sample.aac"
    target.write_bytes(
        part("quiet.aac", "anullsrc=r=16000:cl=mono", 20, "8k")
        + part("loud.aac", "sine=frequency=440:r=16000", 10, "96k")
    )
    return target


@pytestmark_ffmpeg
class TestExactDuration:
    """Tolerance 0.4 s on purpose: what's measured is the length of the
    ENCODED stream, and every AAC encoding pass appends ~2112 priming
    samples at the front and up to 1024 padding samples at the back
    (Apple TN2258). Raw ADTS carries no gapless marker with which this
    could be backed out. The fixture encodes twice, so a good 0.2 s of
    overhang — that's correct and not a measurement error.
    """

    def test_the_counted_duration_matches_the_truth(self, adts):
        measured = duration_from_frames(adts)
        assert measured is not None
        assert abs(measured - 30.0) < 0.4, f"measured: {measured}"

    def test_get_duration_uses_the_count(self, adts):
        assert get_duration(adts) == pytest.approx(duration_from_frames(adts))

    def test_the_estimate_provably_deviates(self, adts):
        """Proves why the counting is needed at all."""
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(adts),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        estimated = float(r.stdout.strip())
        real = duration_from_frames(adts)
        assert abs(estimated - real) > 0.2, (
            f"the estimate was accurate this time ({estimated} vs {real}) "
            f"— then the test setup is no longer meaningful"
        )

    def test_after_repackaging_the_duration_is_right(self, adts, tmp_path):
        """The playback path: once repackaged, the container carries the
        length, and it's the same one the counting produces."""
        m4a = tmp_path / "sample.m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(adts),
                "-c:a",
                "copy",
                "-bsf:a",
                "aac_adtstoasc",
                "-movflags",
                "+faststart",
                str(m4a),
            ],
            check=True,
            timeout=120,
        )
        assert get_duration(m4a) == pytest.approx(duration_from_frames(adts), abs=0.05)

    def test_repackaging_is_lossless(self, adts, tmp_path):
        """Bit-exact: decoded PCM of both versions must be identical."""
        m4a = tmp_path / "sample.m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(adts),
                "-c:a",
                "copy",
                "-bsf:a",
                "aac_adtstoasc",
                str(m4a),
            ],
            check=True,
            timeout=120,
        )

        def md5(path):
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-f", "md5", "-"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return r.stdout.strip()

        assert md5(adts) == md5(m4a)

    def test_a_broken_file_returns_none(self, tmp_path):
        garbage = tmp_path / "broken.aac"
        garbage.write_bytes(b"this is not AAC")
        assert duration_from_frames(garbage) is None


@pytestmark_ffmpeg
class TestRepack:
    """The playback path itself.

    Regression: writing goes to ``*.m4a.tmp`` (atomic), and ffmpeg
    derives the output format from the extension — without ``-f mp4`` it
    chokes on ".tmp" and EVERY repackage silently fell back to the MP3
    transcode. The bug was only visible in a single log line.
    """

    def test_repackaging_returns_a_file(self, adts, tmp_path):
        from whisper_tm.api.media import _repack

        target = _repack(adts, tmp_path, "key")
        assert target is not None, "repackaging failed (missing -f mp4?)"
        assert target.exists() and target.suffix == ".m4a"
        assert not (tmp_path / "key.m4a.tmp").exists(), "temp file was left behind"

    def test_the_duration_is_right_after_repackaging(self, adts, tmp_path):
        from whisper_tm.api.media import _repack

        target = _repack(adts, tmp_path, "key")
        assert get_duration(target) == pytest.approx(duration_from_frames(adts), abs=0.05)

    def test_the_second_call_takes_the_cache(self, adts, tmp_path):
        from whisper_tm.api.media import _repack

        first = _repack(adts, tmp_path, "key")
        stamp = first.stat().st_mtime_ns
        second_val = _repack(adts, tmp_path, "key")
        assert second_val == first and second_val.stat().st_mtime_ns == stamp

    def test_a_broken_file_falls_back(self, tmp_path):
        """None means: the caller transcodes instead."""
        from whisper_tm.api.media import _repack

        garbage = tmp_path / "broken.aac"
        garbage.write_bytes(b"this is not AAC")
        assert _repack(garbage, tmp_path, "key") is None
        assert not (tmp_path / "key.m4a.tmp").exists()

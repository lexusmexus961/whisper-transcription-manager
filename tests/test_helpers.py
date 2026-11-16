"""
Tests for core/helpers.py — formatting, transcript paths, the artifact filter.
"""

import pytest

from whisper_tm.core.helpers import (
    fmt_seg,
    format_duration,
    format_size,
    format_ts,
    is_silent,
)


class TestFormatTs:
    def test_zero(self):
        assert format_ts(0) == "0:00:00"

    def test_seconds(self):
        assert format_ts(45) == "0:00:45"

    def test_minutes(self):
        assert format_ts(125) == "0:02:05"

    def test_hours(self):
        assert format_ts(3661) == "1:01:01"

    def test_large(self):
        assert format_ts(36000) == "10:00:00"


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500.0 B"

    def test_kb(self):
        assert format_size(1536) == "1.5 KB"

    def test_mb(self):
        assert format_size(10 * 1024 * 1024) == "10.0 MB"

    def test_gb(self):
        assert format_size(2.5 * 1024**3) == "2.5 GB"


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(30) == "30s"

    def test_minutes(self):
        assert format_duration(150) == "2.5min"

    def test_hours(self):
        assert format_duration(7200) == "2.0h"


class TestIsSilent:
    """The artifact filter: regular expressions against the whole text.

    Formerly a fixed set of exact phrases — "Untertitelung des ZDF,
    2020" never matched "Untertitelung", and a soundtrack came back full of
    credits. The two shipped patterns are exactly those credits (German,
    because that is what Whisper invents on German-trained silence).
    """

    @pytest.fixture(autouse=True)
    def shipped(self, monkeypatch):
        from whisper_tm.config import config

        t = config._data.setdefault("transcription", {})
        monkeypatch.setitem(t, "filter_hallucinations", True)
        monkeypatch.setitem(
            t, "hallucination_patterns", [r"Untertitelung.*ZDF.*", r"Untertitelung.*BR \d{4}"]
        )
        return t

    def test_empty_text(self):
        assert is_silent({"text": ""})

    def test_dots(self):
        assert is_silent({"text": "..."})

    def test_ellipsis(self):
        assert is_silent({"text": "…"})

    def test_music_notes_hold_no_letter(self):
        assert is_silent({"text": "♪ ♪"})

    @pytest.mark.parametrize(
        "text",
        [
            "Untertitelung des ZDF, 2020",
            "Untertitelung. BR 2018",
            "Untertitelung des ZDF für funk, 2017",
            "UNTERTITELUNG DES ZDF, 2020.",
            "„Untertitelung: BR 2018“",
        ],
    )
    def test_the_credits_of_the_soundtrack(self, text):
        assert is_silent({"text": text})

    @pytest.mark.parametrize(
        "text",
        [
            "This is a normal sentence.",
            "Untertitelung",  # not what the patterns say
            "And then, Untertitelung des ZDF, 2020 — we laughed.",  # only whole segments
        ],
    )
    def test_real_text_stays(self, text):
        assert not is_silent({"text": text})

    def test_own_patterns(self, shipped, monkeypatch):
        monkeypatch.setitem(shipped, "hallucination_patterns", [r"(Music ?)+"])
        assert is_silent({"text": "Music Music Music"})
        assert not is_silent({"text": "Music is great"})

    def test_a_broken_pattern_is_skipped_not_fatal(self, shipped, monkeypatch, caplog):
        monkeypatch.setitem(shipped, "hallucination_patterns", ["(unclosed", "End credits"])
        assert is_silent({"text": "End credits."})
        assert "(unclosed" in caplog.text

    def test_switched_off_nothing_is_matched(self, shipped, monkeypatch):
        monkeypatch.setitem(shipped, "filter_hallucinations", False)
        assert not is_silent({"text": "Untertitelung des ZDF, 2020"})
        assert not is_silent({"text": "…"})

    def test_min_chars(self):
        assert is_silent({"text": "Hi"}, min_chars=5)
        assert not is_silent({"text": "Hello World"}, min_chars=5)

    def test_only_punctuation(self, shipped, monkeypatch):
        # With min_chars=1 pure punctuation goes whatever the filter says
        monkeypatch.setitem(shipped, "filter_hallucinations", False)
        assert is_silent({"text": "!?."}, min_chars=1)
        # With min_chars=0 the length check passes it on…
        assert not is_silent({"text": "!?."}, min_chars=0)
        # …and the artifact filter drops it: no letter, no digit
        monkeypatch.setitem(shipped, "filter_hallucinations", True)
        assert is_silent({"text": "!?."}, min_chars=0)


class TestFmtSeg:
    def test_basic(self):
        seg = {"speaker": "SPEAKER_00", "start": 1.234, "end": 5.678, "text": "Test text"}
        result = fmt_seg(seg)
        assert result["speaker"] == "SPEAKER_00"
        assert result["start"] == 1.23
        assert result["end"] == 5.68
        assert result["text"] == "Test text"
        assert result["start_fmt"] == "0:00:01"
        assert "words" not in result

    def test_missing_speaker(self):
        seg = {"start": 0, "end": 1, "text": "No speaker"}
        result = fmt_seg(seg)
        assert result["speaker"] == "—"

    def test_with_words(self):
        seg = {
            "speaker": "A",
            "start": 0,
            "end": 1,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95},
                {"word": "world", "start": 0.5, "end": 0.9, "score": 0.88},
            ],
        }
        result = fmt_seg(seg, include_words=True)
        assert "words" in result
        assert len(result["words"]) == 2
        assert result["words"][0]["word"] == "Hello"
        assert result["words"][0]["score"] == 0.95

    def test_without_words_flag(self):
        seg = {
            "speaker": "A",
            "start": 0,
            "end": 1,
            "text": "Test",
            "words": [{"word": "Test", "start": 0, "end": 0.5, "score": 0.9}],
        }
        result = fmt_seg(seg, include_words=False)
        assert "words" not in result

    def test_text_stripped(self):
        seg = {"start": 0, "end": 1, "text": "  whitespace  "}
        result = fmt_seg(seg)
        assert result["text"] == "whitespace"


class TestCompletionCache:
    """Tests for the TTL-based transcript completion cache."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        from whisper_tm.config import config

        if not config.is_loaded:
            config.load()

    def test_invalidate_specific(self):
        from whisper_tm.core.helpers import (
            _complete_cache,
            invalidate_complete_cache,
        )

        _complete_cache["test.mp4"] = (True, 9999999999)
        invalidate_complete_cache("test.mp4")
        assert "test.mp4" not in _complete_cache

    def test_invalidate_all(self):
        from whisper_tm.core.helpers import (
            _complete_cache,
            invalidate_complete_cache,
        )

        _complete_cache["a.mp4"] = (True, 9999999999)
        _complete_cache["b.mp4"] = (False, 9999999999)
        invalidate_complete_cache()
        assert len(_complete_cache) == 0

    def test_cache_hit(self):
        import time

        # The cache is keyed to the IDENTITY, not to the display value
        # (docs/PATHS.md) — otherwise it would report a file as done because
        # a same-named one under the current root has one.
        from whisper_tm.core.helpers import (
            _complete_cache,
            is_transcript_complete,
            skiplist_key,
        )

        key = skiplist_key("cached.mp4")
        _complete_cache[key] = (True, time.time())
        assert is_transcript_complete("cached.mp4") is True
        _complete_cache.pop(key, None)

    def test_the_cache_separates_files_with_the_same_name(self):
        """A hit under root A must not pull along root B."""
        import time

        from whisper_tm.core.helpers import (
            _complete_cache,
            is_transcript_complete,
        )

        _complete_cache["/data/a/same.mp4"] = (True, time.time())
        try:
            assert is_transcript_complete("same.mp4", media_dir="/data/a") is True
            assert is_transcript_complete("same.mp4", media_dir="/data/b") is False
        finally:
            _complete_cache.pop("/data/a/same.mp4", None)
            _complete_cache.pop("/data/b/same.mp4", None)

    def test_cache_miss_expired(self):
        import time

        from whisper_tm.core.helpers import (
            _COMPLETE_CACHE_TTL,
            _complete_cache,
            is_transcript_complete,
        )

        # Pre-populate with expired entry
        _complete_cache["expired.mp4"] = (True, time.time() - _COMPLETE_CACHE_TTL - 1)
        # Should re-check (will find nothing → False)
        result = is_transcript_complete("expired.mp4")
        assert result is False
        _complete_cache.pop("expired.mp4", None)


class TestFindOutputWavs:
    """Tests for audio_utils.find_output_wavs — Vocals/Instrumental Detection."""

    def test_standard_mdx_naming(self, tmp_path):
        from whisper_tm.core.audio_utils import find_output_wavs

        (tmp_path / "input_(Vocals).wav").write_bytes(b"\0" * 2000)
        (tmp_path / "input_(Instrumental).wav").write_bytes(b"\0" * 2000)
        v, i = find_output_wavs(tmp_path)
        assert "Vocals" in v.name
        assert "Instrumental" in i.name

    def test_no_vocals_is_instrumental(self, tmp_path):
        """'No Vocals' must be recognized as Instrumental, not as Vocals."""
        from whisper_tm.core.audio_utils import find_output_wavs

        (tmp_path / "input_(Vocals).wav").write_bytes(b"\0" * 2000)
        (tmp_path / "input_(No Vocals).wav").write_bytes(b"\0" * 2000)
        v, i = find_output_wavs(tmp_path)
        assert "Vocals" in v.name and "No" not in v.name
        assert "No Vocals" in i.name

    def test_fallback_by_size(self, tmp_path):
        """No recognizable stem names → larger file as Vocals."""
        from whisper_tm.core.audio_utils import find_output_wavs

        (tmp_path / "output_a.wav").write_bytes(b"\0" * 5000)  # larger
        (tmp_path / "output_b.wav").write_bytes(b"\0" * 2000)  # smaller
        v, i = find_output_wavs(tmp_path)
        assert v.name == "output_a.wav"  # larger = Vocals
        assert i.name == "output_b.wav"

    def test_single_wav(self, tmp_path):
        from whisper_tm.core.audio_utils import find_output_wavs

        (tmp_path / "only_one.wav").write_bytes(b"\0" * 2000)
        v, i = find_output_wavs(tmp_path)
        assert v is not None
        assert i is None

    def test_empty_dir(self, tmp_path):
        from whisper_tm.core.audio_utils import find_output_wavs

        v, i = find_output_wavs(tmp_path)
        assert v is None
        assert i is None

    def test_missing_dir_is_not_an_exception(self, tmp_path):
        """Missing directory = "nothing found", no FileNotFoundError.

        If the chunk directory disappeared mid-run (cache cleared, disk full),
        a raw FileNotFoundError would tear down the entire transcription —
        after hours of GPU time. The caller can handle (None, None).
        """
        from whisper_tm.core.audio_utils import find_output_wav, find_output_wavs

        gone = tmp_path / "does-not-exist"
        assert find_output_wavs(gone) == (None, None)
        assert find_output_wav(gone) is None


class TestCorruptMedia:
    """Corrupt file vs. transient sample error — only the former is final."""

    def test_corrupt_markers_detected(self):
        from whisper_tm.core.audio_utils import _is_corrupt_stderr

        assert _is_corrupt_stderr("[mov,mp4] moov atom not found")
        assert _is_corrupt_stderr("Invalid data found when processing input")

    def test_transient_errors_are_not_corruption(self):
        """A failing NAS must not permanently exclude a file."""
        from whisper_tm.core.audio_utils import _is_corrupt_stderr

        assert not _is_corrupt_stderr("Input/output error")
        assert not _is_corrupt_stderr("Connection timed out")
        assert not _is_corrupt_stderr("")

    def test_has_audio_stream_raises_corrupt(self, monkeypatch, tmp_path):
        import subprocess

        from whisper_tm.core import audio_utils
        from whisper_tm.core.audio_utils import CorruptMediaError, has_audio_stream

        monkeypatch.setattr(
            audio_utils,
            "_run_probe",
            lambda *a, **k: subprocess.CompletedProcess([], 1, "", "moov atom not found"),
        )
        with pytest.raises(CorruptMediaError):
            has_audio_stream(tmp_path / "broken.m4a")

    def test_has_audio_stream_raises_plain_runtime_on_transient(self, monkeypatch, tmp_path):
        import subprocess

        from whisper_tm.core import audio_utils
        from whisper_tm.core.audio_utils import CorruptMediaError, has_audio_stream

        monkeypatch.setattr(
            audio_utils,
            "_run_probe",
            lambda *a, **k: subprocess.CompletedProcess([], 1, "", "Input/output error"),
        )
        with pytest.raises(RuntimeError) as exc:
            has_audio_stream(tmp_path / "nas-gone.mp4")
        assert not isinstance(exc.value, CorruptMediaError)

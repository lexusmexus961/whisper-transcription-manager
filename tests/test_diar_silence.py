"""Cutting silence out of diarization — without warping the timeline.

**Why this exists.** Diarization became the most expensive step (on a
day-long recording it takes longer than separation) and runs over the
whole file, including hours of guaranteed silence. On long recordings
more than half can be silent — two independent methods
(``silencedetect`` and the separation prefilter) agree on that.

**Why it's dangerous.** An offset in the timeline goes unnoticed by
anyone. Speakers then simply end up at the wrong places in the
transcript, and you only notice once someone listens back to a spot.
That's why the back-conversion is a pure function — and why this file
tests it harder than anything else in this project.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import diar_silence as ds


@pytest.fixture()
def aselect_filter(monkeypatch):
    """The aselect form of the cut, whatever this ffmpeg does with it — the
    tests below read that expression (the atrim fallback for an ffmpeg whose
    aselect keeps every frame: tests/test_aselect_fallback.py)."""
    from whisper_tm.core import audio_utils

    monkeypatch.setattr(audio_utils, "aselect_works", lambda: True)


class TestSpeechParts:
    def test_silence_at_the_start_and_in_the_middle(self):
        assert ds.speech_parts([(0.0, 10.0), (20.0, 30.0)], 40.0) == [(10.0, 20.0), (30.0, 40.0)]

    def test_without_silence_everything_stays(self):
        assert ds.speech_parts([], 100.0) == [(0.0, 100.0)]

    def test_complete_silence_yields_nothing(self):
        assert ds.speech_parts([(0.0, 100.0)], 100.0) == []

    def test_short_gaps_are_closed(self):
        """Keeping two parts separated by a 1.5 s gap costs more than
        the 1.5 s it would save."""
        parts = ds.speech_parts([(10.0, 11.5)], 20.0, glue=2.0)
        assert parts == [(0.0, 20.0)]

    def test_long_gaps_stay_gaps(self):
        parts = ds.speech_parts([(10.0, 40.0)], 50.0, glue=2.0)
        assert parts == [(0.0, 10.0), (40.0, 50.0)]

    def test_overlapping_silence_is_merged(self):
        """Two detectors can deliver the same range twice."""
        assert ds.speech_parts([(0.0, 10.0), (5.0, 15.0)], 20.0) == [(15.0, 20.0)]

    def test_silence_beyond_the_end_of_the_file(self):
        assert ds.speech_parts([(90.0, 999.0)], 100.0) == [(0.0, 90.0)]


class TestTimesBack:
    """The core function. Every bug here is invisible data corruption."""

    def test_a_part_shifts_by_its_start(self):
        assert ds.times_back([(0.0, 5.0, "A")], [(100.0, 110.0)]) == [(100.0, 105.0, "A")]

    def test_the_second_part_gets_its_own_offset(self):
        parts = [(0.0, 10.0), (100.0, 110.0)]
        # 12 s in the cut timeline = 2 s into the second part = 102 s
        assert ds.times_back([(12.0, 15.0, "A")], parts) == [(102.0, 105.0, "A")]

    def test_a_section_across_the_cut_is_split(self):
        """THE important case.

        A speaker segment that, in the cut version, runs across a seam
        covers two spots in the original — with minutes or hours of
        silence between them. Reporting it as continuous would amount to
        claiming that someone was talking there.
        """
        parts = [(0.0, 10.0), (100.0, 110.0)]
        assert ds.times_back([(8.0, 12.0, "A")], parts) == [(8.0, 10.0, "A"), (100.0, 102.0, "A")]

    def test_a_section_across_three_parts(self):
        parts = [(0.0, 10.0), (100.0, 110.0), (200.0, 210.0)]
        out = ds.times_back([(5.0, 25.0, "A")], parts)
        assert out == [(5.0, 10.0, "A"), (100.0, 110.0, "A"), (200.0, 205.0, "A")]

    def test_the_total_duration_is_preserved(self):
        """What was cut must not vanish — only move."""
        parts = [(0.0, 10.0), (100.0, 110.0)]
        out = ds.times_back([(0.0, 20.0, "A")], parts)
        assert sum(e - s for s, e, _ in out) == pytest.approx(20.0)

    def test_without_parts_everything_stays_unchanged(self):
        """No cut → no conversion. Otherwise the safe path
        (not cutting) would suddenly become the dangerous one."""
        segs = [(1.0, 2.0, "A"), (3.0, 4.0, "B")]
        assert ds.times_back(segs, []) == segs

    def test_speaker_names_survive(self):
        parts = [(0.0, 10.0), (100.0, 110.0)]
        out = ds.times_back([(8.0, 12.0, "SPEAKER_07")], parts)
        assert {x[2] for x in out} == {"SPEAKER_07"}

    def test_the_result_is_sorted_by_time(self):
        parts = [(0.0, 10.0), (100.0, 110.0)]
        out = ds.times_back([(12.0, 14.0, "B"), (1.0, 2.0, "A")], parts)
        assert [x[0] for x in out] == sorted(x[0] for x in out)

    def test_empty_sections_are_dropped(self):
        assert ds.times_back([(5.0, 5.0, "A")], [(0.0, 10.0)]) == []

    def test_no_section_lands_in_the_silence(self):
        """The acid test: after the back-conversion, no timestamp may
        fall within a range that was cut out as silent."""
        silence = [(10.0, 100.0), (150.0, 300.0)]
        parts = ds.speech_parts(silence, 400.0)
        length = sum(b - a for a, b in parts)
        # One segment spanning the entire cut file
        out = ds.times_back([(0.0, length, "A")], parts)
        for s, e, _ in out:
            for a, b in silence:
                assert e <= a or s >= b, f"segment {s}–{e} reaches into the silence {a}–{b}"


class TestIsTheCutWorthIt:
    def test_a_lot_of_silence_is_worth_it(self):
        parts = ds.speech_parts([(0.0, 500.0)], 1000.0)
        assert ds.cut_worth_it(parts, 1000.0) is True

    def test_little_silence_is_not_worth_it(self):
        parts = ds.speech_parts([(0.0, 50.0)], 1000.0)
        assert ds.cut_worth_it(parts, 1000.0) is False

    def test_too_many_parts_are_not_worth_it(self):
        """With a thousand snippets, ffmpeg's selection becomes unwieldy,
        and every conversion is another chance for a bug."""
        parts = [(float(i), float(i) + 0.5) for i in range(ds.MAX_PARTS + 1)]
        assert ds.cut_worth_it(parts, 10_000.0) is False

    def test_without_parts_no_cut(self):
        assert ds.cut_worth_it([], 100.0) is False


@pytest.mark.usefixtures("aselect_filter")
class TestFilterExpression:
    def test_resets_the_timestamps(self):
        """Without `asetpts`, the output keeps the gaps as time jumps —
        the cut file would end up as long as the original again."""
        f = ds.select_filter([(0.0, 10.0)])
        assert "asetpts=N/SR/TB" in f

    def test_every_part_is_in_there(self):
        f = ds.select_filter([(1.0, 2.0), (5.0, 6.0)])
        assert "between(t,1.000,2.000)" in f
        assert "between(t,5.000,6.000)" in f


@pytest.mark.usefixtures("aselect_filter")
class TestTheParenthesization:
    """Why the expression gets parenthesized — the project's costliest bug.

    Without it, the pipeline can diarize entire day-long files instead
    of just their speech parts. The cause isn't the logic here, but a
    hard limit in ffmpeg's expression parser: it reads `a+b+c+…`
    recursively and gives up past 101 terms.

    What makes it insidious is the path the error takes. ffmpeg exits
    with rc=244, the pipeline catches that, logs a WARNING, and carries
    on with the whole file — technically correct, but several times as
    expensive in time and RAM, enough to run a machine out of memory.
    An error disguised as a warning simply goes unnoticed.
    """

    def test_the_value_stays_the_same(self):
        """Parentheses may only change the reading order, not the selection."""
        parts = [(1.0, 2.0), (5.0, 6.0), (9.0, 10.0)]
        f = ds.select_filter(parts)
        for a, b in parts:
            assert f"between(t,{a:.3f},{b:.3f})" in f
        assert f.count("between(t,") == 3

    def test_the_depth_grows_logarithmically(self):
        """The crux of the matter: 1,000 parts must not end up 1,000
        levels deep. What's measured is the maximum nesting depth."""
        parts = [(float(i), i + 0.5) for i in range(1000)]
        f = ds.select_filter(parts)
        depth = t = 0
        for z in f:
            if z == "(":
                t += 1
                depth = max(depth, t)
            elif z == ")":
                t -= 1
        assert depth <= 20, f"{depth} levels — the parser gives up at 100"

    def test_a_single_part_gets_no_parentheses(self):
        """No decoration: the most common case stays as short as before."""
        assert ds.select_filter([(1.0, 2.0)]) == "aselect='between(t,1.000,2.000)',asetpts=N/SR/TB"

    def test_without_parts_no_broken_expression(self):
        """`cut_worth_it` catches this case beforehand — but an empty
        `aselect=''` would be a syntax error instead of an empty selection."""
        assert "aselect='0'" in ds.select_filter([])

    @pytest.mark.ffmpeg
    def test_REAL_ffmpeg_digests_150_parts(self):
        """The test that actually covers the failure.

        150 lies just past the parser's limit: long recordings commonly
        have anywhere between 1 and well over 100 speech parts — so the
        limit runs right through the middle of real-world inputs.

        The counter-check with the FLAT expression is deliberately placed
        alongside: without it, the test would stay green even if a future
        ffmpeg lifted the limit and the parenthesization became
        unnecessary — you just wouldn't know anymore why it's there.
        """
        import subprocess

        count = 150
        assert count > ds.FFMPEG_EXPR_DEPTH, (
            "the measured limit has grown larger than the case that "
            "failed — this test then checks nothing anymore"
        )

        parts = [(i * 10.0, i * 10.0 + 5.0) for i in range(count)]

        def run(af: str) -> int:
            return subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=16000:cl=mono",
                    "-t",
                    "0.2",
                    "-af",
                    af,
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
            ).returncode

        assert run(ds.select_filter(parts)) == 0, "parenthesized version must go through"

        flat = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in parts)
        if run(f"aselect='{flat}',asetpts=N/SR/TB") == 0:
            import pytest

            pytest.skip(
                "this ffmpeg no longer has the limit — the parenthesization is harmless regardless"
            )


@pytest.mark.ffmpeg
class TestOnRealAudio:
    """A run with real files — the pure arithmetic isn't enough."""

    def test_silence_is_found_in_a_generated_file(self, tmp_path):
        import subprocess

        p = tmp_path / "mixed.wav"
        # 10 s tone, 40 s silence, 10 s tone
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=10:sample_rate=16000",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=10:sample_rate=16000",
                "-filter_complex",
                "[1]atrim=duration=40[s];[0][s][2]concat=n=3:v=0:a=1",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(p),
            ],
            check=True,
            capture_output=True,
        )

        silence = ds.find_silence(p, min_duration=30.0)
        assert silence, "the 40 s of silence weren't found"
        a, b = silence[0]
        assert 9.0 < a < 11.0, f"silence starts at {a}, expected ~10"
        assert 49.0 < b < 51.0, f"silence ends at {b}, expected ~50"

    def test_the_cut_is_really_shorter(self, tmp_path):
        """The check: does the filter actually cut, or does the file
        stay the same length (the classic asetpts bug)?"""
        import subprocess

        p = tmp_path / "mixed.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=10:sample_rate=16000",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=10:sample_rate=16000",
                "-filter_complex",
                "[1]atrim=duration=40[s];[0][s][2]concat=n=3:v=0:a=1",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(p),
            ],
            check=True,
            capture_output=True,
        )

        parts = ds.speech_parts(ds.find_silence(p, min_duration=30.0), 60.0)
        target = tmp_path / "short.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(p),
                "-af",
                ds.select_filter(parts),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(target),
            ],
            check=True,
            capture_output=True,
        )

        duration = float(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(target),
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        assert 18.0 < duration < 23.0, f"cut file is {duration:.1f}s long, expected ~20"

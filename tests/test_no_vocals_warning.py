"""'No vocals' meant two different things — and both sounded like an error.

A finished file can end with an EMPTY transcript (0 segments) and carry
these two warnings:

    No vocals after chunk processing (all chunks without vocals output)
    No vocals found, using original

Often nothing is broken: the pre-filter classified every section as
silence for such a file — the recording simply had no audio, and an empty transcript is the correct
answer. But the sentence sends anyone who looks at the warnings later into
error diagnosis.

The other situation — it WAS separated and no separation provided voice — is
on the other hand really worth a look. Giving both the same sentence obscures
exactly the boundary where one should look.
"""

from __future__ import annotations

import inspect

from whisper_tm.core.vocal_isolation import chunked_separate, no_vocals_text


class TestText:
    def test_all_silence_says_so_too(self):
        t = no_vocals_text(40, 0)
        assert "without audio" in t
        assert "40" in t
        # No word that sounds like an error.
        assert "fail" not in t.lower()

    def test_separated_but_nothing_found_stays_a_finding(self):
        t = no_vocals_text(40, 12)
        assert "12 sections were separated" in t
        assert "none delivered voice" in t

    def test_the_two_situations_are_distinguishable(self):
        """The whole point: Two situations, two sentences."""
        assert no_vocals_text(40, 0) != no_vocals_text(40, 12)

    def test_a_single_separated_section_already_counts(self):
        """The boundary lies at 'was it separated at all?', not at a
        quantity. A section that was separated and yielded nothing is
        already the second situation."""
        assert "without audio" not in no_vocals_text(40, 1)

    def test_a_negative_number_falls_on_the_harmless_side(self):
        """Robustness against a missing balance key: Better to report 'no
        audio' than an invented number of separated sections."""
        assert "without audio" in no_vocals_text(40, -1)


class TestWiringUp:
    def test_the_pipeline_uses_the_helper(self):
        """Otherwise the old sentence would still be in the log and the test
        would prove nothing about what the user actually sees."""
        source = inspect.getsource(chunked_separate)
        assert "no_vocals_text(n_chunks" in source
        assert "all chunks without vocals output" not in source

    def test_the_helper_gets_the_number_of_separated_sections(self):
        """Not the number of silent ones: `separate` is the key on which the
        distinction depends."""
        source = inspect.getsource(chunked_separate)
        place = source.index("no_vocals_text(")
        call = source[place : source.index(")", place) + 1]
        assert '"separate"' in call, call

"""Vocabulary prompt must not come back as transcript text.

With a vocabulary preset (comma list) the list itself can appear as
transcript segments — disconnected, sometimes repeated multiple times. The
cause is WhisperX's mechanics: the vocabulary goes as initial_prompt into
EVERY batch window in the sot_prev slot, the model reads it as "just spoken"
and continues it. Extreme case: a short file with continuous speech
whose entire transcript consists of a single list segment.
"""

from whisper_tm.core.postprocessing import is_prompt_echo, remove_prompt_echoes

VOCABULARY = "Alpha, Bravo, Charlie, Delta"


class TestDetection:
    def test_a_bare_list_is_an_echo(self):
        assert is_prompt_echo("Alpha, Bravo, Charlie, Delta", VOCABULARY)

    def test_a_list_repeated_several_times_is_an_echo(self):
        assert is_prompt_echo("Alpha, Bravo, Charlie, Alpha, Bravo, Charlie", VOCABULARY)

    def test_a_subset_of_the_list_is_an_echo(self):
        """The extreme case: only part of the list."""
        assert is_prompt_echo("Bravo, Charlie, Delta", VOCABULARY)

    def test_a_real_sentence_with_one_term_stays(self):
        assert not is_prompt_echo("Alpha called yesterday", VOCABULARY)

    def test_a_sentence_of_pure_terms_with_a_filler_word_stays(self):
        """A single word outside the list is enough — the prompt reconstruction
        has no filler words."""
        assert not is_prompt_echo("Alpha and Bravo and Charlie", VOCABULARY)

    def test_a_single_interjection_stays(self):
        assert not is_prompt_echo("Alpha!", VOCABULARY)
        assert not is_prompt_echo("Alpha, Alpha", VOCABULARY)

    def test_without_a_vocabulary_nothing_is_detected(self):
        """Without a prompt there is no echo — then the list is real text."""
        assert not is_prompt_echo("Alpha, Bravo, Charlie, Delta", "")

    def test_case_and_punctuation_do_not_matter(self):
        assert is_prompt_echo("alpha; BRAVO — charlie.", VOCABULARY)


class TestRemove:
    def _segs(self):
        return [
            {"start": 0.0, "text": "Good morning everyone."},
            {"start": 5.0, "text": "Alpha, Bravo, Charlie, Delta"},
            {"start": 9.0, "text": "Alpha called yesterday"},
        ]

    def test_only_the_echo_is_dropped(self):
        out = remove_prompt_echoes(self._segs(), VOCABULARY)
        assert [s["start"] for s in out] == [0.0, 9.0]

    def test_without_a_vocabulary_everything_stays(self):
        assert len(remove_prompt_echoes(self._segs(), "")) == 3

    def test_a_whole_file_that_is_only_echo(self):
        """The real case: the transcript consists ONLY of the echo."""
        assert (
            remove_prompt_echoes([{"start": 0.34, "text": "Bravo, Charlie, Delta"}], VOCABULARY)
            == []
        )

def test_english_phantoms_score_like_the_german_ones():
    from whisper_tm.core.postprocessing import phantom_score

    assert phantom_score("Thanks for watching!", None, -1.0) >= 0.8
    assert phantom_score("Thank you.", "SPEAKER_00", -0.2) < 0.5  # a real one, with a speaker
    assert phantom_score("Subtitles by the Amara.org community", None, None) >= 0.7
    assert phantom_score("Thank you for the coffee.", None, None) == 0.0

"""
Word boundary segmentation — the step from "timestamp per word" to
precise subtitles, tested with synthetic Japanese.

Word-accurate SRT cues matter especially for Japanese. The logic used to
live untestable inside the worker (core/transcription.py) and has been
extracted to core/segmentation.py — these tests prove the four
phases on exactly the cases that occur in real CJK.
"""

from __future__ import annotations

import pytest

from whisper_tm.core.segmentation import word_boundary_segmentation


def _char(text: str, start: float, duration_per: float) -> list[dict]:
    """A CJK segment words array: each character a "word" with times."""
    return [
        {
            "word": ch,
            "start": round(start + i * duration_per, 3),
            "end": round(start + (i + 1) * duration_per, 3),
        }
        for i, ch in enumerate(text)
    ]


@pytest.fixture(autouse=True)
def _defaults():
    """Lock split parameters to documented defaults."""
    from whisper_tm.config import config

    old_value = {
        k: config.get("transcription", k, default=None)
        for k in (
            "max_segment_duration",
            "max_segment_words",
            "max_segment_words_cjk",
            "word_pause_threshold",
        )
    }
    config.set("transcription", "max_segment_duration", 3.5)
    config.set("transcription", "max_segment_words", 8)
    config.set("transcription", "max_segment_words_cjk", 15)
    config.set("transcription", "word_pause_threshold", 0.3)
    yield
    for k, v in old_value.items():
        if v is not None:
            config.set("transcription", k, v)


# ── Phase 1: Repair broken CJK times ────────────────────────────────


def test_mostly_broken_timings_are_distributed_evenly():
    # wav2vec2 classic with CJK: all characters start==end
    words = [{"word": ch, "start": 5.0, "end": 5.0} for ch in "こんにちは"]
    seg = [{"start": 5.0, "end": 7.5, "text": "こんにちは", "words": words}]
    out = word_boundary_segmentation(seg, "ja")
    w = out[0]["words"]
    assert w[0]["start"] == 5.0 and w[-1]["end"] == 7.5
    # strictly monotonic — each syllable has real duration
    assert all(b["start"] >= a["end"] for a, b in zip(w, w[1:], strict=False))
    assert all(x["end"] > x["start"] for x in w)


def test_a_single_broken_character_is_interpolated():
    words = _char("これはテスト", 0.0, 0.4)
    words[2]["start"] = words[2]["end"] = None  # a hole in the middle
    seg = [{"start": 0.0, "end": 2.4, "text": "これはテスト", "words": words}]
    out = word_boundary_segmentation(seg, "ja")
    w = out[0]["words"][2]
    assert w["start"] is not None and w["end"] is not None
    assert 0.8 <= w["start"] < w["end"] <= 1.2  # between the neighbors


# ── Phase 2: repetitions ─────────────────────────────────────────────────


def test_laughter_is_collapsed():
    words = _char("は" * 12, 0.0, 0.2)
    seg = [{"start": 0.0, "end": 2.4, "text": "は" * 12, "words": words}]
    out = word_boundary_segmentation(seg, "ja")
    # Greedy pattern takes the longest pattern that matches ≥6×: はは×6
    assert out[0]["text"] == "はは×6"
    assert out[0]["words"][0]["start"] == 0.0


def test_a_repetition_with_real_text_behind_it_stays():
    text = "ははは、それでは始めましょう"
    seg = [{"start": 0.0, "end": 3.0, "text": text, "words": _char(text, 0.0, 3.0 / len(text))}]
    out = word_boundary_segmentation(seg, "ja")
    assert "始めましょう" in "".join(s["text"] for s in out)  # nothing lost


# ── Phase 3: split — the actual subtitle cues ─────────────────────


def test_split_at_the_sentence_end_with_exact_timings():
    text = "今日は良い天気です。明日は雨です。"
    seg = [
        {
            "start": 0.0,
            "end": 5.1,
            "text": text,
            "speaker": "SPEAKER_00",
            "words": _char(text, 0.0, 0.3),
        }
    ]
    out = word_boundary_segmentation(seg, "ja")
    assert len(out) == 2
    # The punctuation stays with the sentence — even when the duration
    # limit (3.0s) hits one character before it (the earlier bug produced
    # "。明日は…").
    assert out[0]["text"] == "今日は良い天気です。"
    assert out[1]["text"] == "明日は雨です。"
    # Cue times come from the CHARACTERS, not from the original segment
    assert out[0]["end"] == pytest.approx(3.0, abs=0.01)
    assert out[1]["start"] == pytest.approx(3.0, abs=0.01)
    assert out[1]["speaker"] == "SPEAKER_00"


def test_kinsoku_no_cue_starts_with_a_combining_character():
    """行頭禁則: ょ/っ/ー & co. must never stand at the start of a cue.

    The sentence is longer than the 3.0s limit and has neither
    punctuation nor pauses — the duration limit MUST split somewhere, but
    never such that the next subtitle starts with a combining character
    (previously real: "でし|ょう" → cue "ょう。").
    """
    from whisper_tm.core.segmentation import CJK_NOT_AT_START

    text = "明日は雨が降るでしょうけれども試合はきっとやります"
    seg = [{"start": 0.0, "end": len(text) * 0.3, "text": text, "words": _char(text, 0.0, 0.3)}]
    out = word_boundary_segmentation(seg, "ja")
    assert len(out) >= 2  # limit has split
    for s in out:
        assert s["text"][0] not in CJK_NOT_AT_START, s["text"]


def test_max_15_chars_per_cue():
    # 40 DIFFERENT characters — a uniform run would be a repetition for
    # phase 2 and would (correctly) be collapsed instead of split.
    text = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらり"
    seg = [{"start": 0.0, "end": 4.0, "text": text, "words": _char(text, 0.0, 0.1)}]
    out = word_boundary_segmentation(seg, "ja")
    assert len(out) >= 3
    assert all(len(s["text"]) <= 15 for s in out)


def test_pause_split_at_the_cjk_threshold():
    # 0.2s pause: above the CJK threshold (0.15) → split
    w = _char("それでは", 0.0, 0.3) + _char("始めます", 1.4, 0.3)
    seg = [{"start": 0.0, "end": 2.6, "text": "それでは始めます", "words": w}]
    from whisper_tm.config import config

    # Force a split — CJK reads its own duration key
    config.set("transcription", "max_segment_duration_cjk", 2.0)
    try:
        out = word_boundary_segmentation(seg, "ja")
    finally:
        config.set("transcription", "max_segment_duration_cjk", 3.0)
    assert len(out) == 2
    assert out[0]["text"] == "それでは"
    assert out[1]["start"] == pytest.approx(1.4, abs=0.01)


# ── Phase 4: merge protection ─────────────────────────────────────────────────


def test_a_short_interjection_does_not_merge_across_minutes():
    segs = [
        {"start": 100.0, "end": 100.3, "text": "はい", "words": _char("はい", 100.0, 0.15)},
        {
            "start": 250.0,
            "end": 253.0,
            "text": "次の質問です",
            "words": _char("次の質問です", 250.0, 0.5),
        },
    ]
    out = word_boundary_segmentation(segs, "ja")
    assert len(out) == 2  # gap guard: NO merge
    assert out[0]["end"] == 100.3


def test_a_western_language_with_spaces():
    words = [
        {"word": w, "start": i * 0.5, "end": (i + 1) * 0.5}
        for i, w in enumerate(
            ["This", "is", "a", "test", ".", "And", "here", "comes", "more", "text", "now"]
        )
    ]
    seg = [{"start": 0.0, "end": 5.5, "text": "…", "words": words}]
    out = word_boundary_segmentation(seg, "en")
    assert len(out) >= 2
    assert " " in out[0]["text"]  # joined with a space


# ── Adjustable cue duration for CJK + whitespace filter ──────────────────


def test_the_cjk_cue_duration_is_freely_configurable():
    """max_segment_duration_cjk applies directly — the old min(…,3.0) cap
    made the setting a dummy for CJK (e.g. 1-second chunks were
    impossible)."""
    from whisper_tm.config import config

    text = "明日は雨が降るけれども試合はやります"  # 18 characters, no breaks
    seg = [{"start": 0.0, "end": 3.6, "text": text, "words": _char(text, 0.0, 0.2)}]
    config.set("transcription", "max_segment_duration_cjk", 1.0)
    try:
        out = word_boundary_segmentation(seg, "ja")
    finally:
        config.set("transcription", "max_segment_duration_cjk", 3.0)
    assert len(out) >= 3  # ~1s chunks
    # Phase 4 may merge splitters back (min 0.4s) — but nothing may
    # stay significantly over the limit (tolerance: one kinsoku deferral)
    assert max(s["end"] - s["start"] for s in out) <= 1.5


def test_spaces_between_cjk_characters_are_dropped():
    from whisper_tm.core.segmentation import strip_cjk_spaces

    assert strip_cjk_spaces("今 度 の 旅 行") == "今度の旅行"
    # Latin in a CJK context stays readably separated
    assert strip_cjk_spaces("今度の Mr. Smith 旅行") == "今度の Mr. Smith 旅行"
    assert strip_cjk_spaces("") == ""
    assert strip_cjk_spaces("plain english text") == "plain english text"


def test_korean_keeps_the_spaces_between_its_words():
    # Korean was on the list of languages without spaces: every transcript
    # was saved as one unbroken run, and search and subtitles broke with it
    from whisper_tm.core.segmentation import NO_SPACE_LANGS, strip_cjk_spaces

    sentence = "오늘 회의는 세 시에 시작합니다"
    assert "ko" not in NO_SPACE_LANGS
    assert strip_cjk_spaces(sentence) == sentence
    words = [
        {"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(sentence.split())
    ]
    seg = [{"start": 0.0, "end": 2.0, "text": sentence, "words": words}]
    out = word_boundary_segmentation(seg, "ko")
    assert " ".join(s["text"] for s in out) == sentence


def test_the_split_result_has_no_cjk_spaces():
    # Word strings with Whisper spaces (" 今") → cue text still clean
    text = "今度の旅行は家族みんなで行くそうです。まだ決まりません。"
    words = [
        {"word": " " + ch, "start": i * 0.2, "end": (i + 1) * 0.2} for i, ch in enumerate(text)
    ]
    seg = [{"start": 0.0, "end": len(text) * 0.2, "text": text, "words": words}]
    out = word_boundary_segmentation(seg, "ja")
    for s in out:
        assert " " not in s["text"], s["text"]


def test_a_katakana_word_is_never_cut_apart():
    """A katakana run = ONE loanword. Without protection the duration limit
    can cut right through it, e.g. "…のスマー" | "トフォン…" — one word
    split over two cues."""
    text = "毎日の生活で使われるのがスマートフォンの画面だと言われる"
    seg = [{"start": 0.0, "end": len(text) * 0.25, "text": text, "words": _char(text, 0.0, 0.25)}]
    out = word_boundary_segmentation(seg, "ja")
    assert len(out) >= 2  # limits have split
    for a, b in zip(out, out[1:], strict=False):
        # No cut in the middle of katakana: predecessor ending in katakana
        # AND successor starting with katakana would be exactly the
        # スマー|トフォン tear.
        from whisper_tm.core.segmentation import _is_katakana

        assert not (_is_katakana(a["text"][-1]) and _is_katakana(b["text"][0])), (
            f"katakana torn: …{a['text'][-4:]} | {b['text'][:4]}…"
        )
    assert "スマートフォン" in "".join(s["text"] for s in out)


# ── "Split only at natural boundaries" ─────────────────────────────────


def _with_natural_only(segs, lang):
    from whisper_tm.config import config

    config.set("transcription", "split_natural_only", True)
    try:
        return word_boundary_segmentation(segs, lang)
    finally:
        config.set("transcription", "split_natural_only", False)


def test_natural_only_cuts_at_the_pause_instead_of_the_limit():
    """Limit long since exceeded — the cut still only happens at the pause."""
    from whisper_tm.config import config

    # 24 characters in a row (4.8s), THEN a real pause, then more
    part1 = "明日は雨が降るけれども試合はやりますから全員"
    part2 = "集合してください"
    w = _char(part1, 0.0, 0.2) + _char(part2, 5.4, 0.2)  # 0.6s pause
    seg = [{"start": 0.0, "end": 7.0, "text": part1 + part2, "words": w}]
    config.set("transcription", "max_segment_duration_cjk", 1.0)
    try:
        out = _with_natural_only(seg, "ja")
    finally:
        config.set("transcription", "max_segment_duration_cjk", 3.0)
    assert len(out) == 2  # ONE cut — at the pause
    assert out[0]["text"] == part1  # no hack mid-sentence
    assert out[1]["start"] == pytest.approx(5.4, abs=0.01)


def test_natural_only_uses_weak_punctuation_more_eagerly():
    """Overdue + 、 → split at the comma (the 、 rule would otherwise need
    ≥4 characters since the last cut, or 1.5s)."""
    from whisper_tm.config import config

    text = "はい、そうです、わかりました、では始めます"
    seg = [{"start": 0.0, "end": len(text) * 0.4, "text": text, "words": _char(text, 0.0, 0.4)}]
    config.set("transcription", "max_segment_duration_cjk", 1.0)
    try:
        out = _with_natural_only(seg, "ja")
    finally:
        config.set("transcription", "max_segment_duration_cjk", 3.0)
    assert len(out) >= 3
    # Every cut sits AFTER a punctuation mark — never mid-flow
    for s in out[:-1]:
        assert s["text"][-1] == "、", s["text"]


def test_natural_only_without_boundaries_keeps_the_cue_whole():
    """No pause, no punctuation → prefer a too-long cue over a chopped
    sentence (documented trade-off of this mode)."""
    text = "明日は雨が降るけれども試合はやりますから全員集合"
    seg = [{"start": 0.0, "end": len(text) * 0.3, "text": text, "words": _char(text, 0.0, 0.3)}]
    out = _with_natural_only(seg, "ja")
    assert len(out) == 1
    assert out[0]["text"] == text


# ── Strong punctuation mid short segment ─────────────────────────


def test_a_sentence_end_inside_a_short_segment_is_split():
    """A short segment like "本当か?うれしいですね。" (~1 s, 12 characters)
    used to stay one cue — short segments never even entered the split phase,
    even though a sentence ends in the middle."""
    text = "本当か?うれしいですね。"
    seg = [{"start": 10.0, "end": 11.2, "text": text, "words": _char(text, 10.0, 0.1)}]
    out = word_boundary_segmentation(seg, "ja")
    texts = [s["text"] for s in out]
    assert texts == ["本当か?", "うれしいですね。"], texts


def test_punctuation_at_the_end_does_not_split():
    """Only MID-SEGMENT counts — a normal sentence cue stays one."""
    text = "うれしいですね。"
    seg = [{"start": 10.0, "end": 11.0, "text": text, "words": _char(text, 10.0, 0.12)}]
    out = word_boundary_segmentation(seg, "ja")
    assert [s["text"] for s in out] == [text]


# ── Phase 1b: implausible speaking rates (alignment gaps at digits) ─────


def test_an_implausible_speech_rate_is_redistributed():
    """wav2vec2 cannot align digits — a whole run of characters can be
    squeezed into 0.8 s, next to which the final "。" hangs on
    for 10 s. The times are formally valid (start<end), but must be
    redistributed."""
    text = "今日はみんなで駅に集まるんですからね3番7番2番9番います"
    # 29 characters in ~0.8 s … then a single character with a 10 s tail
    words = [
        {"word": ch, "start": round(100.0 + i * 0.028, 3), "end": round(100.0 + (i + 1) * 0.028, 3)}
        for i, ch in enumerate(text)
    ]
    words.append({"word": "。", "start": 100.8, "end": 111.0})
    seg = [{"start": 100.0, "end": 111.0, "text": text + "。", "words": words}]
    out = word_boundary_segmentation(seg, "ja")
    # No cue may keep an impossible speaking rate …
    for s in out:
        dur = s["end"] - s["start"]
        rate = len(s["text"]) / max(dur, 0.05)
        assert rate <= 16, (s["text"], round(rate, 1))
    # … and no cue swallows almost the whole segment as dead air
    longest = max(s["end"] - s["start"] for s in out)
    assert longest < 8, longest


def test_plausible_timings_stay_untouched():
    """A normal speaking rate (~6 char/s) must NOT be redistributed —
    real alignment times are better than any interpolation."""
    text = "今日はいい天気ですね"
    words = _char(text, 50.0, 0.17)
    seg = [{"start": 50.0, "end": 51.7, "text": text, "words": words}]
    out = word_boundary_segmentation(seg, "ja")
    assert out[0]["words"][0]["start"] == 50.0
    assert abs(out[0]["words"][1]["start"] - 50.17) < 0.01

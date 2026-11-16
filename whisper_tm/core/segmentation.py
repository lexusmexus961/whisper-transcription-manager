"""
Word-boundary segmentation: aligned words become subtitle cues.

This logic used to live as a ~250-line block in the middle of
``transcription_worker`` — inaccessible for tests, even though it's THE step
that makes precise subtitles from "timestamp per word" (e.g. for Japanese).
Extracted here, behavior unchanged.

For CJK languages (ja/zh/ko/yue), wav2vec2 alignment provides characters as
"words" — and often fails (start==end, missing times). Therefore four phases:

1. Repair broken word times (interpolation from segment/neighbor times)
2. Collapse repetitions (e.g. a syllable repeated many times → syllable×12)
3. Split at word boundaries: punctuation (hard for sentence-final marks;
   soft for others), duration, character and pause limits — CJK tighter
   (max 15 characters, ≤3 s, pause 0.15 s)
4. Merge overly short/overlapping splits back together (with a gap guard,
   so an isolated short interjection doesn't merge with a segment
   minutes later)
"""

from __future__ import annotations

import re as _re

# Bind the singleton at run time (like core/ytdl.py) — tests swap it out.
import whisper_tm.extensions as ext
from whisper_tm.config import config

CJK_BREAK_CHARS = set("。、！？!?,;:；：…）」』】》）")
CJK_STRONG_BREAK = set("。！？!?")
# Kinsoku (line-start prohibition): characters that in Japanese may NEVER
# stand at the start of a line — small kana (combining syllables), the
# long vowel mark, iteration marks. A split right before one would tear
# the syllable apart (would look broken in subtitles).
CJK_NOT_AT_START = set("ゃゅょぁぃぅぇぉっーゝゞ々ャュョァィゥェォッヶヵ")


def _is_katakana(ch: str) -> bool:
    """Katakana (full- and half-width) — a katakana RUN is ONE loanword."""
    return "゠" <= ch <= "ヿ" or "ｦ" <= ch <= "ﾟ"


def _log(job, msg: str) -> None:
    if job is not None and ext.job_manager:
        ext.job_manager.log(job, msg)


#: Languages written without spaces between words. Korean is not one of
#: them: it was treated as one, and every Korean transcript was saved as an
#: unbroken run of text.
NO_SPACE_LANGS = frozenset({"ja", "zh", "yue"})

# CJK script ranges: punctuation, kana, CJK ideographs (+ext. A/compat.),
# fullwidth forms, halfwidth katakana. Hangul is left out — Korean puts
# spaces between its words.
# For the space filter below — e.g. a space before a Japanese period
# must also be removed.
_CJK_CHARS = (
    r"\u3001-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uff01-\uff60\uff66-\uff9f"
)
_CJK_SPACE_RE = _re.compile(f"(?<=[{_CJK_CHARS}])\\s+(?=[{_CJK_CHARS}])")


def strip_cjk_spaces(text: str) -> str:
    """Remove spaces BETWEEN CJK characters.

    Whisper's BPE tokenizer likes to scatter spaces between tokens in
    Japanese — simply wrong in the written form. Only gaps between two
    CJK characters are removed; Latin text stays untouched (e.g. a
    Japanese phrase containing an embedded English name).
    """
    return _CJK_SPACE_RE.sub("", text or "")


def _word_speaker(w: dict) -> str | None:
    """The speaker whisperx.assign_word_speakers gave a word — None if none."""
    sp = w.get("speaker") if isinstance(w, dict) else None
    return sp if sp and sp != "—" else None


def _speaker_of(words: list[dict], fallback):
    """The speaker most of ``words`` belong to — the parent's when the words
    carry none."""
    counts: dict[str, int] = {}
    for w in words:
        sp = _word_speaker(w)
        if sp:
            counts[sp] = counts.get(sp, 0) + 1
    if not counts:
        return fallback
    return max(counts, key=lambda k: (counts[k], k == fallback))


def _same_speaker(a: dict, b: dict) -> bool:
    """May two cues be merged? Not when both name a speaker and they
    differ (a short answer would otherwise go into the other speaker's
    cue)."""
    sa, sb = a.get("speaker"), b.get("speaker")
    if not sa or not sb or sa == "—" or sb == "—":
        return True
    return sa == sb


def word_boundary_segmentation(final: list[dict], lang: str, job=None) -> list[dict]:
    """Split segments with ``words`` into subtitle-fine segments.

    ``final`` is not mutated and returned, but replaced (the input list
    itself does remain working material though — exactly like the
    earlier inline block that kept operating on ``final``).
    """
    max_seg_dur = config.get("transcription", "max_segment_duration", default=3.5)
    max_words = config.get("transcription", "max_segment_words", default=8)
    pause_threshold = config.get("transcription", "word_pause_threshold", default=0.3)
    min_seg_dur = 0.4  # minimum segment duration (sec.)

    is_cjk = lang in NO_SPACE_LANGS
    if is_cjk:
        max_words = config.get("transcription", "max_segment_words_cjk", default=15)
        # Dedicated CJK key instead of the previous hard min(…, 3.0):
        # the cap made the setting a dummy for CJK as soon as someone
        # wanted more than 3 s — and less (e.g. 1 s for very readable
        # cues) is now just as directly settable.
        max_seg_dur = config.get("transcription", "max_segment_duration_cjk", default=3.0)
        pause_threshold = min(pause_threshold, 0.15)

    # Split only at natural boundaries (pauses/punctuation): the
    # duration/character limits then no longer force a cut mid-sentence,
    # but instead make the soft boundaries more eager. Cues can thereby
    # become longer than the maximum — in exchange, no sentence flow
    # breaks off anymore.
    natural_only = bool(config.get("transcription", "split_natural_only", default=False))

    # ══════════════════════════════════════════════════════
    #  Phase 1: repair broken word timestamps
    #  WhisperX alignment (wav2vec2) often fails on CJK —
    #  many words have start==end or no timestamp at all.
    #  → interpolation from segment boundaries.
    # ══════════════════════════════════════════════════════
    repaired_count = 0
    for seg in final:
        words = seg.get("words", [])
        if not words:
            continue
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_dur = seg_end - seg_start

        # Count valid words (have real duration)
        valid = sum(
            1
            for w in words
            if w.get("start") is not None
            and w.get("end") is not None
            and w["end"] > w["start"] + 0.01
        )
        total = len(words)

        # >50% broken → distribute all evenly across the segment
        if total > 0 and valid < total * 0.5 and seg_dur > 0.1:
            per_word = seg_dur / total
            for wi, w in enumerate(words):
                w["start"] = round(seg_start + wi * per_word, 3)
                w["end"] = round(seg_start + (wi + 1) * per_word, 3)
            repaired_count += 1
        else:
            # Interpolate individual broken words
            for wi, w in enumerate(words):
                has_start = w.get("start") is not None
                has_end = w.get("end") is not None
                is_zero = has_start and has_end and abs(w["end"] - w["start"]) < 0.01

                if not has_start or not has_end or is_zero:
                    prev_end = seg_start
                    for pi in range(wi - 1, -1, -1):
                        pe = words[pi].get("end")
                        ps = words[pi].get("start")
                        if pe is not None and ps is not None and pe > ps + 0.01:
                            prev_end = pe
                            break
                    next_start = seg_end
                    for ni in range(wi + 1, total):
                        ns = words[ni].get("start")
                        ne = words[ni].get("end")
                        if ns is not None and ne is not None and ne > ns + 0.01:
                            next_start = ns
                            break
                    # Number of consecutive broken words
                    gap_words = 1
                    for ni in range(wi + 1, total):
                        ne = words[ni].get("end")
                        ns = words[ni].get("start")
                        if ne is not None and ns is not None and ne > ns + 0.01:
                            break
                        gap_words += 1
                    slot = (next_start - prev_end) / max(gap_words, 1)
                    idx_in_gap = 0
                    for gi in range(wi, min(wi + gap_words, total)):
                        ws = words[gi].get("start")
                        we = words[gi].get("end")
                        if ws is None or we is None or abs(we - ws) < 0.01:
                            words[gi]["start"] = round(prev_end + idx_in_gap * slot, 3)
                            words[gi]["end"] = round(prev_end + (idx_in_gap + 1) * slot, 3)
                        idx_in_gap += 1

    if repaired_count > 0:
        _log(
            job,
            f"Word timestamps: {repaired_count} segments "
            f"with faulty timestamps repaired (interpolated)",
        )

    # ══════════════════════════════════════════════════════
    #  Phase 1b: plausibility of word times (CJK)
    #  wav2vec2 cannot align digits/special characters (e.g. a
    #  string of spoken numbers):
    #  WhisperX then interpolates nonsense — 27 characters in 0.8 s
    #  next to a 10-second single character. The times are formally
    #  "valid" (start<end), so the 50% repair above doesn't catch it.
    #  Criterion here: a 4-word window with an impossible speech rate
    #  (>16 characters/s; real Japanese stays below ~12 mora/s even
    #  spoken fast)
    #  → redistribute all word times of the segment character-weighted
    #  across the segment boundaries.
    # ══════════════════════════════════════════════════════
    if is_cjk:
        implausible = 0
        for seg in final:
            words = seg.get("words", [])
            if len(words) < 4:
                continue
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_dur = seg_end - seg_start
            if seg_dur < 1.0:
                continue
            bad = False
            for i in range(len(words) - 3):
                windows = words[i : i + 4]
                duration = (windows[-1].get("end") or 0) - (windows[0].get("start") or 0)
                chars = sum(len((w.get("word") or "").strip()) for w in windows)
                if chars / max(duration, 0.05) > 16:
                    bad = True
                    break
            if not bad:
                continue
            total = sum(len((w.get("word") or "").strip()) for w in words) or len(words)
            pos = seg_start
            for w in words:
                share = (len((w.get("word") or "").strip()) or 1) / total
                w["start"] = round(pos, 3)
                pos = min(seg_end, pos + share * seg_dur)
                w["end"] = round(pos, 3)
            implausible += 1
        if implausible:
            _log(
                job,
                f"Word times: {implausible} segments with implausible "
                f"speech rate re-interpolated (alignment gaps, "
                f"e.g. digits)",
            )

    # ══════════════════════════════════════════════════════
    #  Phase 2: collapse repetitions
    #  (はっはっはっは... / hahaha... etc.)
    # ══════════════════════════════════════════════════════
    for seg in final:
        words = seg.get("words", [])
        if len(words) < 8:
            continue
        joined = "".join(w.get("word", "") for w in words)
        rep_match = _re.match(r"^(.{1,3})\1{5,}", joined)
        # Only collapse if the repetition makes up (almost) the WHOLE
        # segment — otherwise real text after the
        # repetition would be lost ("ははは、それでは始めましょう").
        if rep_match and rep_match.end() >= len(joined) * 0.9:
            pattern = rep_match.group(1)
            count = rep_match.end() // len(pattern)
            remainder = joined[rep_match.end() :].strip()
            display = f"{pattern}×{count}" + (f" {remainder}" if remainder else "")
            seg["words"] = [
                {
                    "word": display,
                    "start": words[0].get("start", seg["start"]),
                    "end": words[-1].get("end", seg["end"]),
                }
            ]
            seg["text"] = display

    # ══════════════════════════════════════════════════════
    #  Phase 3: split segments at word boundaries
    #
    #  ⚠ WARNING, NEW DICTS ARE BUILT HERE.
    #
    #  Phases 1, 2 and 4 modify the existing segments in place — only
    #  this one rebuilds. Any field not explicitly set on the child is
    #  gone afterward, and silently so: `fmt_seg` only writes what's
    #  present, the DB column stays NULL, nothing fails.
    #
    #  That's exactly how `avg_logprob`, `phantom_score` and `overlapped`
    #  once disappeared for months — every diarized segment held zeros,
    #  while the same pipeline without diarization (and thus without
    #  this split) delivered confidence values.
    #
    #  Anyone adding a field here: first consider whether it may be
    #  INHERITED. `avg_logprob` yes — it comes from the decoder and can't
    #  be recomputed after cutting. `phantom_score` and `overlapped`
    #  NO — they apply to the old boundaries and are redetermined by the
    #  caller after the split.
    # ══════════════════════════════════════════════════════
    split_final = []
    for seg in final:
        words = seg.get("words", [])
        seg_dur = seg.get("end", 0) - seg.get("start", 0)

        # A STRONG punctuation mark in the middle of the segment (sentence-
        # final marks — not at the end) means: two sentences sit in one
        # cue here. The split phase must check for this even when
        # duration and length are under the limits — otherwise such
        # cases stayed glued together forever.
        sentence_midway = is_cjk and any(
            (w.get("word") or "").strip()[-1:] in CJK_STRONG_BREAK for w in words[:-1]
        )
        # Two speakers in one segment (the words say so): its cues must
        # part where the speaker changes — a lasting change, two words or
        # the segment's end, not a single word assigned astray
        speaker_turns = {
            i
            for i in range(1, len(words))
            if _word_speaker(words[i])
            and _word_speaker(words[i - 1])
            and _word_speaker(words[i]) != _word_speaker(words[i - 1])
            and (
                i + 1 >= len(words)
                or _word_speaker(words[i + 1]) in (None, _word_speaker(words[i]))
            )
        }

        if not words or (
            seg_dur <= max_seg_dur
            and len(words) <= max_words
            and not sentence_midway
            and not speaker_turns
        ):
            split_final.append(seg)
            continue

        split_points = [0]
        for i in range(1, len(words)):
            words_since_split = i - split_points[-1]
            if i in speaker_turns:
                split_points.append(i)
                continue
            if words_since_split < 2:
                continue

            sub_start_word = words[split_points[-1]]
            curr_word = words[i]
            sub_dur = (curr_word.get("end") or curr_word.get("start", 0)) - (
                sub_start_word.get("start", 0)
            )

            prev_word_text = (words[i - 1].get("word", "") or "").strip()
            prev_end = words[i - 1].get("end")
            curr_start = curr_word.get("start")

            # Overdue = duration or count limit exceeded. In "natural
            # boundaries only" mode this no longer forces a cut, but
            # instead makes the soft punctuation rules more eager — the
            # cut then falls on the next natural boundary (hard limits
            # chop the sentence flow unnaturally).
            overdue = sub_dur >= max_seg_dur or words_since_split >= max_words

            # CJK: punctuation split
            if is_cjk and prev_word_text:
                last_char = prev_word_text[-1]
                if last_char in CJK_STRONG_BREAK and words_since_split >= 2:
                    split_points.append(i)
                    continue
                if last_char in CJK_BREAK_CHARS and (
                    words_since_split >= 4 or sub_dur >= 1.5 or overdue
                ):
                    split_points.append(i)
                    continue

            # Western punctuation
            if not is_cjk and prev_word_text:
                last_char = prev_word_text[-1] if prev_word_text else ""
                if last_char in ".!?:" and words_since_split >= 3:
                    split_points.append(i)
                    continue
                if last_char in ",;–—" and (words_since_split >= 5 or overdue):
                    split_points.append(i)
                    continue

            # Don't separate punctuation from the sentence: if the CURRENT
            # word ends with a break character, the duration/count limit
            # must not set the split EXACTLY before it — otherwise the
            # next subtitle would start with a stray punctuation mark
            # (happened for real: limit hit one character before the
            # sentence end). One step later the punctuation split kicks
            # in and the character stays with the sentence. Plus Kinsoku:
            # no split before characters that must not stand at the
            # start of a line.
            curr_text = (curr_word.get("word", "") or "").strip()
            trailing_chars = bool(curr_text) and curr_text[-1] in (
                CJK_BREAK_CHARS if is_cjk else set(".!?:,;")
            )
            kinsoku = is_cjk and bool(curr_text) and curr_text[0] in CJK_NOT_AT_START
            # Katakana run = ONE loanword — never cut in the middle
            # (happened for real, splitting a loanword mid-syllable).
            inside_katakana = (
                is_cjk
                and bool(prev_word_text)
                and bool(curr_text)
                and _is_katakana(prev_word_text[-1])
                and _is_katakana(curr_text[0])
            )
            split_forbidden = trailing_chars or kinsoku or inside_katakana

            # Hard limits: only when cuts mid-flow are allowed. In
            # natural-only mode, overdue segments wait for the next
            # pause/punctuation boundary (cues may then become longer).
            if not natural_only:
                # Duration limit
                if sub_dur >= max_seg_dur and words_since_split >= 2 and not split_forbidden:
                    split_points.append(i)
                    continue

                # Word count limit
                if words_since_split >= max_words and not split_forbidden:
                    split_points.append(i)
                    continue

            # Pause split (a real speaking pause doesn't trip Kinsoku —
            # a combining character CAN'T stand before a pause; the
            # guard stays anyway in case the alignment produces nonsense)
            if prev_end is not None and curr_start is not None:
                gap = curr_start - prev_end
                if (
                    gap >= pause_threshold
                    and words_since_split >= 2
                    and not kinsoku
                    and not inside_katakana
                ):
                    split_points.append(i)

        for si, sp in enumerate(split_points):
            end_idx = split_points[si + 1] if si + 1 < len(split_points) else len(words)
            sub_words = words[sp:end_idx]
            if not sub_words:
                continue
            sub_start = sub_words[0].get("start", seg.get("start", 0))
            sub_end = sub_words[-1].get("end", seg.get("end", 0))
            if is_cjk:
                # The space filter applies here too: the word strings
                # themselves can carry Whisper spaces.
                sub_text = strip_cjk_spaces("".join(w.get("word", "") for w in sub_words).strip())
            else:
                sub_text = " ".join(w.get("word", "") for w in sub_words).strip()
            if not sub_text:
                continue
            part = {
                "start": sub_start,
                "end": sub_end,
                "text": sub_text,
                "speaker": (
                    _speaker_of(sub_words, seg.get("speaker", "—"))
                    if speaker_turns
                    else seg.get("speaker", "—")
                ),
                "words": sub_words,
            }
            # Inherit the decoder confidence from the parent segment.
            #
            # This phase is the only one that builds NEW dicts — phases
            # 1, 2 and 4 modify the existing ones in place. That's
            # exactly why `avg_logprob` once got lost here: in the
            # database it was NULL for every diarized segment, because
            # diarization forces alignment and alignment triggers this
            # split. For non-diarized segments it was present — the
            # same pipeline, just without this one step.
            #
            # `phantom_score` and `overlapped` are deliberately NOT
            # inherited: they apply to the old boundaries. A 20-second
            # segment with 7 seconds of overlap would otherwise pass its
            # flag to all five subtitles, including the four that don't
            # touch the overlap at all. That's why the caller calls
            # `mark_segments` AFTER this split.
            if seg.get("avg_logprob") is not None:
                part["avg_logprob"] = seg["avg_logprob"]
            split_final.append(part)

    # ══════════════════════════════════════════════════════
    #  Phase 4: merge overly short / overlapping segments
    # ══════════════════════════════════════════════════════
    merged_final = []
    for seg in split_final:
        seg_dur = seg["end"] - seg["start"]
        if merged_final:
            prev = merged_final[-1]
            prev_dur = prev["end"] - prev["start"]
            overlap = seg["start"] < prev["end"] + 0.05
            should_merge = _same_speaker(prev, seg) and (
                (seg_dur < min_seg_dur and overlap)
                or (prev_dur < min_seg_dur and overlap)
                or (abs(seg["start"] - prev["start"]) < 0.1 and abs(seg["end"] - prev["end"]) < 0.1)
            )
            if should_merge:
                sep = "" if is_cjk else " "
                prev["text"] = (prev["text"] + sep + seg["text"]).strip()
                prev["end"] = max(prev["end"], seg["end"])
                # setdefault instead of get: otherwise the words end up
                # in a throwaway list if prev has no "words"
                prev.setdefault("words", []).extend(seg.get("words", []))
                continue
        merged_final.append(seg)

    # Last round: merge segments < min_seg_dur with their successor —
    # BUT only if temporally adjacent. Without a gap guard, an isolated
    # short interjection (e.g. a brief "yes" at 100.0-100.3s) would
    # merge with the next segment, possibly minutes later, and create a
    # wrongly timed giant subtitle.
    if len(merged_final) > 1:
        max_merge_gap = 1.0  # sec. — larger gap = do not merge
        final_pass = [merged_final[0]]
        for seg in merged_final[1:]:
            prev = final_pass[-1]
            gap = seg["start"] - prev["end"]
            if (
                prev["end"] - prev["start"] < min_seg_dur
                and gap <= max_merge_gap
                and _same_speaker(prev, seg)
            ):
                sep = "" if is_cjk else " "
                prev["text"] = (prev["text"] + sep + seg["text"]).strip()
                prev["end"] = max(prev["end"], seg["end"])
                prev.setdefault("words", []).extend(seg.get("words", []))
            else:
                final_pass.append(seg)
        merged_final = final_pass

    cjk_info = f" [CJK-Mode: {lang}]" if is_cjk else ""
    _log(
        job,
        f"Word split: {len(merged_final)} segments "
        f"(max {max_seg_dur}s / {max_words} Tokens, "
        f"min {min_seg_dur}s){cjk_info}",
    )
    return merged_final

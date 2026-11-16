"""Post-processing of transcript segments.

Principle: NOTHING is deleted or changed — only scores and markers,
reversible and searchable. Whether something is a phantom depends on a
MANUALLY calibrated phrase list (calibration test: large-v3 confirmed to
invent "Many thanks" etc. where there's coughing/humming/music) plus
context gates. avg_logprob alone is independently disproven as hallucination
detector (F1 = 23.6%, measured) — it enters only as ONE gate among many.

Markers per segment:
  avg_logprob    Decoder confidence (passed through from whisperx, 4.0)
  phantom_score  0.0–1.0: known hallucination phrase + gates (4.1)
  overlapped    1, if diarization found ≥2 simultaneous speakers there
                 (4.2) — Whisper is demonstrably incomplete there
                 (raw sanity check), gap is made visible.
"""

from __future__ import annotations

import re

# ── Phantom list ────────────────────────────────────────────────────────
# Source 1 (calibrated): large-v3 outro phantoms, confirmed
# on test clips. Source 2: documented German community strings
# (openai/whisper #679/#651/#928, whisperX #230, Gist riotbib) — YouTube
# subtitle artifacts that never appear in game/everyday audio. Source 3: the
# English outputs Whisper is known for on silence and music (openai/whisper
# discussions #928, #1783) — a match only scores; real thank-yous with a
# speaker stay under the usual threshold.
PHANTOM_EXACT = [
    "vielen dank",
    "vielen vielen dank",
    "danke",
    "danke schön",
    "dankeschön",
    "bis zum nächsten mal",
    "bis zum nächsten mal tschüss",
    "das wars für heute",
    "das wars",
    "danke fürs zuschauen",
    "danke fürs zusehen",
    "tschüss",
    "untertitel der amara org community",
    "untertitel von stephanie geiges",
    "mit untertiteln von swiss txt",
    "copyright wdr",
    "thank you",
    "thank you so much",
    "thank you very much",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching and see you next time",
    "see you next time",
    "please subscribe",
    "like and subscribe",
    "bye",
    "subtitles by the amara org community",
    "transcription by castingwords",
]
PHANTOM_PREFIX = [
    "untertitel im auftrag des",
    "untertitelung im auftrag des",
    "untertitel der amara",
    "untertitelung des zdf",
    "copyright wdr",
    "swiss txt",
    "amara org",
    "subtitles by",
    "captions by",
    "transcribed by",
    "transcription by",
    "please subscribe to",
]


def _normalize_it(text: str) -> str:
    t = re.sub(r"[^a-zäöüß0-9\s]", " ", (text or "").lower())
    return " ".join(t.split())


def phantom_score(text: str, speaker: str | None, avg_logprob: float | None) -> float:
    """0.0 = no suspicion … 1.0 = very sure phantom.

    Basis is phrase match; gates refine:
      +0.25 no speaker assigned — diarization found no speech energy
            there
      +0.15 avg_logprob < −0.8 (uncertain decoder)
    A phrase match WITH speaker stays under 0.5 — real thank-yous exist;
    the threshold for UI/filtering is up to the consumer.
    """
    n = _normalize_it(text)
    if not n:
        return 0.0
    hits = (
        n in PHANTOM_EXACT
        or any(n.startswith(p) for p in PHANTOM_PREFIX)
        # "Many, many thanks." repeated — repeat variants of the
        # classic
        or re.fullmatch(r"(vielen\s+)+dank", n) is not None
    )
    if not hits:
        return 0.0
    score = 0.45
    if not speaker or speaker == "—":
        score += 0.25
    if avg_logprob is not None and avg_logprob < -0.8:
        score += 0.15
    return round(min(score, 1.0), 2)


def overlapping_intervals(diar_segs) -> list[tuple[float, float]]:
    """Time ranges with ≥2 simultaneously active speakers from diarization.

    diar_segs is the pyannote/whisperx DataFrame (columns start/end/speaker).
    Sweep over start/end events; robust against None/empty.
    """
    try:
        starts = list(diar_segs["start"])
        ends = list(diar_segs["end"])
    except Exception:
        return []
    events = sorted([(float(s), 1) for s in starts] + [(float(e), -1) for e in ends])
    intervals = []
    active = 0
    begin = None
    for t, delta in events:
        active += delta
        if active >= 2 and begin is None:
            begin = t
        elif active < 2 and begin is not None:
            if t - begin > 0.2:  # ignore mini-overlaps
                intervals.append((begin, t))
            begin = None
    return intervals


def _overlap(seg_start, seg_end, intervals) -> float:
    """FRACTION of segment that overlaps — not a number of seconds.

    The distinction matters enough for its own sentence: the return value
    is held against ``overlaps_from`` (0.3), and what "30%" means depends
    entirely on how long the segment is.
    """
    duration = max(seg_end - seg_start, 1e-6)
    total_sum = 0.0
    for s, e in intervals:
        total_sum += max(0.0, min(seg_end, e) - max(seg_start, s))
    return total_sum / duration


def is_prompt_echo(text: str, vocabulary: str) -> bool:
    """Does the segment consist EXCLUSIVELY of words from the vocabulary prompt?

    Then Whisper didn't transcribe, it continued the prompt.
    WhisperX puts the vocabulary in EVERY batch window in the sot_prev slot
    (whisperx/asr.py: initial_prompt → all_tokens → previous_tokens), the
    model sees it as "what was just spoken". With a comma-separated list,
    the most likely continuation is the same list again — observed case:
    an 18-second file with continuous speech ended up as only the word list.

    Deliberately strict: EVERY token must come from the list. A real sentence
    with one of the words still has filler words — the prompt replica doesn't.
    Plus at least two different terms or four tokens, so a single genuine
    interjection ("damn!") survives.
    """
    if not vocabulary:
        return False
    terms = {w for w in (_normalize_it(vocabulary).split()) if w}
    if not terms:
        return False
    tokens = _normalize_it(text).split()
    if len(tokens) < 3 or not all(t in terms for t in tokens):
        return False
    return len(set(tokens)) >= 2 or len(tokens) >= 4


def remove_prompt_echoes(segments: list[dict], vocabulary: str) -> list[dict]:
    """Filter out prompt echoes; returns remaining segments.

    The exception to "mark only" principle (see module header): a phantom
    like "Many thanks" COULD have been spoken, but here we have demonstrable
    text we fed the model ourselves. This is not suspicion, it's an artifact
    of our own input.
    """
    if not vocabulary:
        return segments
    return [s for s in segments if not is_prompt_echo(s.get("text", ""), vocabulary)]


def mark_segments(segments: list[dict], diar_segs=None, overlaps_from: float = 0.3) -> dict:
    """Set all markers in-place. Returns: counts for log.

    ⚠ **MUST run AFTER word boundary split**, not before.

    Markers apply to the boundaries they were calculated on. Before split
    they judge 20-second blocks, after they judge subtitles — only the
    latter works. A block with 7 seconds overlap would otherwise pass its
    mark to all five subtitles, including the four without overlap. And
    "Many thanks" is suspicious as its own subtitle, not in twenty seconds
    of actual speech.

    An earlier version called it before the split — and the split threw
    both markers away (see warning at Phase 3 in ``segmentation.py``).

    ``overlaps_from`` is a **FRACTION** of segment duration, not seconds
    (see :func:`_overlap`). On unsplit segments, value 0.3 was useless:
    a 20-second block would need 6 seconds overlap in a row, real overlaps
    last 0.3–1.5 s. On cut subtitles it works: four seconds need 1.2 s.
    Value stays where it is — never wrong, just in wrong place in chain.

    **Only** ``phantom_score`` and ``overlapped`` are changed.
    Text, times, speakers and words remain untouched; this step can't
    worsen subtitles.
    """
    intervals = overlapping_intervals(diar_segs) if diar_segs is not None else []
    n_phantom = n_overlapping = 0
    for seg in segments:
        score = phantom_score(seg.get("text", ""), seg.get("speaker"), seg.get("avg_logprob"))
        if score > 0:
            seg["phantom_score"] = score
            n_phantom += 1
        if (
            intervals
            and _overlap(seg.get("start", 0.0), seg.get("end", 0.0), intervals) >= overlaps_from
        ):
            seg["overlapped"] = 1
            n_overlapping += 1
    return {"phantom": n_phantom, "overlapped": n_overlapping, "overlap_intervals": len(intervals)}

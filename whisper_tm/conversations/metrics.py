"""How people talk in one transcript — counted from its segments alone.

Pure: segments in, numbers out; no database, no configuration, no model.
store.py writes what :func:`analyze` returns, query.py adds it up.

What the numbers mean:

* **speech** — the time covered by segments (overlaps counted once);
  **attributed** speech is what a diarization label says who spoke.
* **words** — runs of letters of every script, combining marks included
  (the rule of core/stopwords.py, so "नमस्ते" is one word). Languages
  written without spaces (Chinese, Japanese, Thai …) count each letter:
  there is no space to count words by, and comparisons stay within one
  language (query.py's baselines are per language).
* **sentences** — cut at . ! ? … followed by a space, and right after
  。！？, inside one speaker's turn. Segments are not sentences: with word
  timestamps they are subtitle cues of a few words, and Whisper's own
  segments end mid-sentence too, so a sentence may span several segments.
  A single letter before the period ("z. B.", initials) does not end one.
  A transcript with fewer than one sentence end per
  :data:`PUNCTUATED_WORDS` words counts each segment as a sentence and
  says so (``punctuated`` false): some languages and models write no
  sentence punctuation, and one endless "sentence" says nothing.
* **turns** — consecutive segments of one speaker. Segments without a
  label ('—': not diarized) belong to no speaker's turn and don't end
  one; a pause of :data:`TURN_GAP_SEC` does (a new turn of the same
  speaker, not a change). A **speaker change** is a turn whose speaker is
  not the one of the turn before; ``turns`` counts the changes (the first
  turn included). A **short turn** is a change of at most
  :data:`SHORT_TURN_WORDS` words — "yes", "right": backchannels.
* **active speakers** — labels with at least :data:`ACTIVE_SHARE` of the
  attributed speech, and :data:`ACTIVE_MIN_SEC` of it unless their share
  is large (a short voice message has short turns): a label that said one
  word in an hour is diarization noise, not a participant.
* **interview** — two active speakers, one of them asking: at least
  :data:`INTERVIEW_QUESTIONS` of their sentences are questions while they
  speak at most :data:`INTERVIEW_SHARE` of the time.

Segments marked as phantoms (``phantom_score`` > 0: phrases Whisper is
known to invent over noise, core/postprocessing.py) are left out — "Thanks
for watching" over a cough is nobody's sentence.

**Windows** slice the timeline into :data:`WINDOW_SEC` pieces and keep,
per label, the talk, words, sentences, questions and changes in each. A
query adds them up to find the stretch of a long recording where given
people talk together — in a stream of several hours the people come and
go, and the recording as a whole says little about any hour of it.
"""

from __future__ import annotations

import re

from whisper_tm.core.stopwords import WORD_PATTERN

#: Seconds per window (query.py's stretches are made of whole windows)
WINDOW_SEC = 300
#: A pause this long ends a turn, also of the same speaker
TURN_GAP_SEC = 20.0
#: A speaker change of at most this many words is a backchannel
SHORT_TURN_WORDS = 3
#: An active speaker has at least this share of the attributed speech …
ACTIVE_SHARE = 0.05
#: … and at least this many seconds of it, unless they have this share
ACTIVE_MIN_SEC = 3.0
ACTIVE_LARGE_SHARE = 0.2
#: Interview: the asker's share of questions among their sentences …
INTERVIEW_QUESTIONS = 0.35
#: … at most this share of the talk …
INTERVIEW_SHARE = 0.35
#: … and at least this many sentences
INTERVIEW_MIN_SENTENCES = 5
#: With fewer than one sentence end per this many words, a transcript is
#: unpunctuated (each segment a sentence)
PUNCTUATED_WORDS = 60
#: Turn rate and mean turn need this much attributed speech to mean much
RATE_MIN_SEC = 60.0
#: The label of a segment nobody was assigned to (not diarized)
UNATTRIBUTED = "—"
#: Languages written without spaces between words: letters are counted
UNSPACED = frozenset({"ja", "zh", "yue", "th", "lo", "km", "my", "bo"})

_LETTER = re.compile(r"[^\W\d_]")
# A sentence ends at . ! ? … (several, then maybe closing quotes or
# brackets) when whitespace follows — "3.5" and "e.g.," don't end one —
# or right after a CJK full stop, which needs no space after it.
_END = re.compile(r"([.!?…‽]+[\"'»«”’)\]]*)(\s+)|([。！？]+[」』”’）)]*)")
_SINGLE_LETTER = re.compile(r"(?:^|\s)[^\W\d_]\.$")
_ANY_END = re.compile(r"[.!?…‽]+(?=[\"'»«”’)\]]*(?:\s|$))|[。！？]")
_CLOSERS = "\"'»«”’)]」』）"


def count_words(text: str, unspaced: bool = False) -> int:
    """Words in ``text`` — letters in a language without spaces."""
    if not text:
        return 0
    if unspaced:
        return len(_LETTER.findall(text))
    return len(WORD_PATTERN.findall(text))


def split_sentences(text: str) -> list[str]:
    """``text`` cut into sentences (see the module)."""
    out, last = [], 0
    for m in _END.finditer(text):
        if m.group(1):
            if _SINGLE_LETTER.search(text[last : m.start(1) + 1]):
                continue
            end = m.end(1)
        else:
            end = m.end(3)
        piece = text[last:end].strip()
        if piece:
            out.append(piece)
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    return out


def is_question(sentence: str) -> bool:
    s = sentence.rstrip(_CLOSERS + " ")
    return s.endswith("?") or s.endswith("？")


def _clean(segments) -> list[tuple[float, float, str | None, str]]:
    """(start, end, label or None, text), sorted; phantoms and empty
    segments left out."""
    out = []
    for s in segments or ():
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        try:
            if float(s.get("phantom_score") or 0) > 0:
                continue
            a = float(s.get("start") or 0)
            b = float(s.get("end") or 0)
        except (TypeError, ValueError):
            continue
        if not (a == a and b == b):  # NaN
            continue
        b = max(a, b)
        sp = s.get("speaker")
        label = sp.strip() if isinstance(sp, str) and sp.strip() not in ("", UNATTRIBUTED) else None
        out.append((a, b, label, text))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _union(segs) -> float:
    total, cs, ce = 0.0, None, None
    for a, b, _, _ in segs:
        if ce is None:
            cs, ce = a, b
        elif a > ce:
            total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    if ce is not None:
        total += ce - cs
    return total


def _empty_speaker(start: float) -> dict:
    return {
        "talk_sec": 0.0,
        "words": 0,
        "sentences": 0,
        "questions": 0,
        "turns": 0,
        "short_turns": 0,
        "longest_turn_sec": 0.0,
        "longest_turn_start": start,
        "first_sec": start,
        "last_sec": start,
    }


def analyze(segments, language: str | None = None, duration_sec: float | None = None) -> dict:
    """``{"summary": {...}, "speakers": {label: {...}}, "windows": {label:
    [[window, talk_sec, words, sentences, questions, changes], ...]}}``.

    ``segments`` are transcript segments (dicts with start, end, speaker,
    text, maybe phantom_score); ``duration_sec`` the recording's length
    (the last segment's end when unknown)."""
    unspaced = str(language or "").lower().split("-")[0] in UNSPACED
    segs = _clean(segments)

    all_words = sum(count_words(t, unspaced) for _, _, _, t in segs)
    ends = sum(len(_ANY_END.findall(t)) for _, _, _, t in segs)
    punctuated = all_words == 0 or (ends > 0 and all_words / ends <= PUNCTUATED_WORDS)

    # Runs: a speaker's turn, or a stretch nobody was assigned to. Each
    # holds its segments (start, end, text) for the windows below.
    runs: list[dict] = []
    for a, b, label, text in segs:
        cur = runs[-1] if runs else None
        if cur is not None and cur["label"] == label and a - cur["end"] < TURN_GAP_SEC:
            cur["end"] = max(cur["end"], b)
            cur["segs"].append((a, b, text))
        else:
            runs.append({"label": label, "start": a, "end": b, "segs": [(a, b, text)]})

    speakers: dict[str, dict] = {}
    windows: dict[str, dict[int, list]] = {}
    totals = {"words": 0, "sentences": 0, "questions": 0, "sentence_words": 0}
    unattributed_sec = 0.0
    prev_label = None
    for run in runs:
        texts = [t for _, _, t in run["segs"]]
        sents = split_sentences(" ".join(texts)) if punctuated else texts
        sents = [x for x in sents if count_words(x, unspaced)]
        n_sent = len(sents)
        n_quest = sum(1 for x in sents if is_question(x))
        seg_words = [count_words(t, unspaced) for _, _, t in run["segs"]]
        n_words = sum(seg_words)
        talk = sum(b - a for a, b, _ in run["segs"])
        totals["words"] += n_words
        totals["sentences"] += n_sent
        totals["questions"] += n_quest
        totals["sentence_words"] += sum(count_words(x, unspaced) for x in sents)
        label = run["label"]
        change = False
        if label is None:
            unattributed_sec += talk
        else:
            d = speakers.get(label)
            if d is None:
                d = speakers[label] = _empty_speaker(run["start"])
            d["talk_sec"] += talk
            d["words"] += n_words
            d["sentences"] += n_sent
            d["questions"] += n_quest
            change = label != prev_label
            if change:
                d["turns"] += 1
                if n_words <= SHORT_TURN_WORDS:
                    d["short_turns"] += 1
            prev_label = label
            span = run["end"] - run["start"]
            if span > d["longest_turn_sec"]:
                d["longest_turn_sec"] = span
                d["longest_turn_start"] = run["start"]
            d["last_sec"] = max(d["last_sec"], run["end"])
        # Windows: talk and words where each segment lies, the run's
        # sentences and questions shared out by its words (a monologue of
        # ten minutes is not all in the window it started in), the change
        # where the run started.
        key = label or UNATTRIBUTED
        per = windows.setdefault(key, {})
        for (a, b, _), w in zip(run["segs"], seg_words):
            row = per.setdefault(int(a // WINDOW_SEC), [0.0, 0, 0.0, 0.0, 0])
            row[0] += b - a
            row[1] += w
            if n_words:
                row[2] += n_sent * w / n_words
                row[3] += n_quest * w / n_words
        if n_words == 0 and run["segs"]:
            row = per.setdefault(int(run["segs"][0][0] // WINDOW_SEC), [0.0, 0, 0.0, 0.0, 0])
            row[2] += n_sent
            row[3] += n_quest
        if change:
            per.setdefault(int(run["start"] // WINDOW_SEC), [0.0, 0, 0.0, 0.0, 0])[4] += 1

    attributed = sum(d["talk_sec"] for d in speakers.values())
    for d in speakers.values():
        d["talk_sec"] = round(d["talk_sec"], 2)
        d["longest_turn_sec"] = round(d["longest_turn_sec"], 2)
    active = [
        sp
        for sp, d in speakers.items()
        if attributed > 0
        and d["talk_sec"] / attributed >= ACTIVE_SHARE
        and (d["talk_sec"] >= ACTIVE_MIN_SEC or d["talk_sec"] / attributed >= ACTIVE_LARGE_SHARE)
    ]
    shares = (
        sorted((d["talk_sec"] / attributed for d in speakers.values()), reverse=True)
        if attributed > 0
        else []
    )
    longest_label = max(speakers, key=lambda sp: speakers[sp]["longest_turn_sec"], default=None)
    changes = sum(d["turns"] for d in speakers.values())

    interview = False
    if len(active) == 2:
        asker = min(active, key=lambda sp: speakers[sp]["talk_sec"])
        d = speakers[asker]
        interview = (
            d["sentences"] >= INTERVIEW_MIN_SENTENCES
            and d["questions"] / d["sentences"] >= INTERVIEW_QUESTIONS
            and d["talk_sec"] / attributed <= INTERVIEW_SHARE
        )

    end = max((b for _, b, _, _ in segs), default=0.0)
    try:
        duration = float(duration_sec) if duration_sec else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    summary = {
        "start_sec": round(segs[0][0], 2) if segs else 0.0,
        "end_sec": round(end, 2),
        "duration_sec": round(max(duration, end), 2),
        "speech_sec": round(_union(segs), 2),
        "attributed_sec": round(attributed, 2),
        "unattributed_sec": round(unattributed_sec, 2),
        "words": totals["words"],
        "sentences": totals["sentences"],
        "questions": totals["questions"],
        "turns": changes,
        "short_turns": sum(d["short_turns"] for d in speakers.values()),
        "labels": len(speakers),
        "speakers": len(active),
        "top_share": round(shares[0], 4) if shares else 0.0,
        "longest_turn_sec": speakers[longest_label]["longest_turn_sec"] if longest_label else 0.0,
        "longest_turn_label": longest_label,
        "longest_turn_start": round(speakers[longest_label]["longest_turn_start"], 2)
        if longest_label
        else 0.0,
        "punctuated": bool(punctuated),
        "interview": bool(interview),
        # Derived: words per sentence over every sentence, speaker changes
        # per minute of attributed speech, attributed speech per change
        "wps": round(totals["sentence_words"] / totals["sentences"], 3)
        if punctuated and totals["sentences"]
        else None,
        "turn_rate": round(changes / (attributed / 60.0), 3)
        if attributed >= RATE_MIN_SEC and changes
        else None,
        "mean_turn_sec": round(attributed / changes, 2)
        if attributed >= RATE_MIN_SEC and changes
        else None,
    }
    out_windows = {
        label: [
            [w, round(r[0], 2), r[1], round(r[2], 2), round(r[3], 2), r[4]]
            for w, r in sorted(per.items())
        ]
        for label, per in windows.items()
    }
    return {"summary": summary, "speakers": speakers, "windows": out_windows}

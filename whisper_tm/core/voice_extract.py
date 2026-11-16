"""The voice extractor — one person's voice as clips, each with its sentence.

All voices of one person, or a part of them narrowed down by name, date
and many other filters — to download as one audio file or as single
samples in a ZIP, with the transcript pieces. Useful for TTS training:
the voice detection stores the exact stretches where only that voice is
heard, so the clips can be cut cleanly.

**What a clip is.** A sentence of the transcript (a segment) spoken by one
of the chosen voices. The sentence brings the text, its time range the
audio. A voice's diarization label and its sentences' speaker are the same
label by construction (``core/speaker.align_labels`` for voices-only runs),
so the sentences of a voice are the segments with its label.

**Where only this voice is heard.** Two sources, the better one first:

1. The voice's stored speech spans (``speaker_vectors.parts``): the times
   where diarization heard exactly one speaker, this one — overlap and
   silence are already cut out (``clean_spans``). A
   sentence counts when nearly all of it lies inside the spans of the
   CHOSEN windows of its voice.
2. A voice without stored spans: then the transcript's own marker decides:
   ``segments.overlapped`` is set where diarization found two speakers at
   once, and most of the voice's windows must belong to the person (a
   label that mixes two people is left out). Weaker — the check is per
   sentence, not per second — and every clip says which check it passed
   (``clean_by`` in the metadata).

**Which voices.** The named ones — and, if asked for, the voices the
collection recognizes as this person: their best similarity to one of the
person's named voices reaches the threshold, and among all named people
this one's centroid is the nearest. The same two measures the file view
(``where_it_occurs``) and the guided assignment's spread use. Discarded
voices never.

**From conversations.** The sentences can be narrowed to conversations the
conversation statistics found (whisper_tm/conversations; the search view's
Conversations tab hands its filters over): an hour where Alice, Bob and
Carol talk together in long sentences, say. A sentence counts when its
middle lies in one of them — in a stretch of a long recording, or anywhere
in a recording that is a conversation as a whole. Copies stay apart there,
as everywhere in the extractor.

**The text** is the transcript's sentence as it is. No word times are
stored (``transcription.word_timestamps`` is off by default), so a clip is
always a whole sentence: sentences are chosen, not cut — by length, by a
plausible speaking rate (two words over sixteen seconds means the times
are wrong), by the decoder's confidence, and without what the artifact
filter or the phantom check marks.

**The export** runs in the background, one at a time, and writes into
``<clips folder>/voice-exports`` — the work folder sits on the system disk,
and a dataset of a few hours is gigabytes — or into a folder you name
(instead of downloading), as a ZIP or unpacked, straight where a TTS trainer reads it. That
folder must lie in the allowed trees and already exist, and an export
never replaces anything in it: a name that is taken gets ``-2``, ``-3``.
Audio is cut from the original recording, recording by recording, so a
six-hour stream is not decoded once per sentence.
"""

from __future__ import annotations

import bisect
import collections
import io
import json
import logging
import math
import re
import shutil
import threading
import time
import uuid
import wave
import zipfile
from pathlib import Path

import numpy as np

from whisper_tm.core import media_formats, paths

logger = logging.getLogger(__name__)

# ── Small helpers ────────────────────────────────────────────────────────

_DATE_ISO = re.compile(r"(20\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})")
_DATE_DE = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.((?:19|20)?\d{2})(?!\d)")


def date_from_name(path: str) -> str:
    """Recording date from the FILE NAME — ``YYYY-MM-DD`` or ``""``.

    The same rule as the map's date filter (``graphDateFromName``): the file
    time is often only a backup or copy date (years after an old voice
    message), the name carries the recording date. Two spellings: year
    first (``AUD-20240301``, ``2024-03-01 14-30-00``, ``2024.03.01``) and
    the German day-first one (``interview 01.03.2024``, ``(01.03.24)``);
    many names carry no date at all.
    """
    name = str(path or "").rsplit("/", 1)[-1]
    m = _DATE_ISO.search(name)
    if m:
        y, mo, d = m.groups()
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
    for m in _DATE_DE.finditer(name):
        d, mo, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31 and "2000" <= y <= "2099":
            return f"{y}-{int(mo):02d}-{int(d):02d}"
    return ""


def parse_spans(raw) -> list[tuple[float, float]]:
    """Stored speech spans (JSON ``[[a,b],…]``) → sorted, merged pairs."""
    if not raw:
        return []
    try:
        pairs = json.loads(raw)
    except (TypeError, ValueError):
        return []
    out = []
    for x in pairs if isinstance(pairs, list) else []:
        try:
            a, b = float(x[0]), float(x[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a >= 0:
            out.append((a, b))
    return merge_spans(out)


def merge_spans(spans, gap: float = 0.05) -> list[tuple[float, float]]:
    """Sort and join spans that touch (the windows cut one long turn in
    thirty-second pieces; here it is one turn again)."""
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def covered(a: float, b: float, spans, starts=None) -> float:
    """How much of ``[a, b]`` lies inside the (merged) spans, 0…1."""
    if b <= a or not spans:
        return 0.0
    starts = starts if starts is not None else [s[0] for s in spans]
    i = max(0, bisect.bisect_right(starts, a) - 1)
    inside = 0.0
    while i < len(spans) and spans[i][0] < b:
        lo, hi = max(a, spans[i][0]), min(b, spans[i][1])
        if hi > lo:
            inside += hi - lo
        i += 1
    return inside / (b - a)


def _span_of(a: float, b: float, spans, starts) -> tuple[float, float] | None:
    """The span that holds most of ``[a, b]`` — padding stays inside it."""
    i = max(0, bisect.bisect_right(starts, a) - 1)
    best, best_len = None, 0.0
    while i < len(spans) and spans[i][0] < b:
        lo, hi = max(a, spans[i][0]), min(b, spans[i][1])
        if hi - lo > best_len:
            best, best_len = spans[i], hi - lo
        i += 1
    return best


def certainty(logprob) -> int | None:
    """Whisper's certainty of a sentence in percent — the decoder's mean
    token probability, ``exp(avg_logprob)``."""
    if logprob is None:
        return None
    try:
        return int(round(100 * math.exp(float(logprob))))
    except (TypeError, ValueError, OverflowError):
        return None


def slug(text: str, fallback: str = "voice") -> str:
    """A file-name-safe version of a name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")
    return s[:60] or fallback


# ── Options ──────────────────────────────────────────────────────────────

#: Every filter with its default. The page sends what it shows; anything
#: missing or out of range falls back to these.
DEFAULTS = {
    "include": "recognized",  # "named" or "recognized"
    # Recognized: best similarity to one of the person's named voices at
    # least this. The recognized voices of a person typically sit near 0.9;
    # from 0.75 to 0.80 the count barely drops, above 0.90 it falls
    # sharply. 0.80 keeps nearly all and leaves the doubtful edge out — a
    # dataset suffers more from a wrong voice than from a few clips less.
    "threshold": 0.80,
    "date_from": "",
    "date_to": "",
    "folder": "",  # display path starts with
    "file_text": "",  # in the file path
    "text": "",
    "text_not": "",  # in the sentence / not in the sentence
    "language": "",
    "min_sec": 2.0,
    "max_sec": 15.0,
    "rate_min": 5.0,
    "rate_max": 25.0,  # characters per second
    # How sure Whisper was of the sentence, in percent: the decoder's mean
    # token probability, exp(avg_logprob). No per-word certainty is stored;
    # this is the sentence's. On clear speech most sentences reach 75 %.
    "min_certainty": 75,
    "skip_phantoms": True,
    "skip_artifacts": True,
    "clean": "best",  # "strict": only voices with speech spans
    "coverage": 0.9,  # share of a sentence inside the spans
    "label_share": 0.6,  # no spans: share of the label's windows
    "join_gap": 0.0,  # join sentences closer than this (0 = off)
    "padding": 0.1,
    "per_recording": 0,
    "max_hours": 0.0,
    "max_clips": 0,
    "order": "time",  # "time", "best", "spread", "sure"
    # Only from these conversations: their filters by the names of
    # GET /api/conversations (people, match, min_sec, kind, sentences …);
    # empty = every recording
    "conversations": {},
}

#: The conversation filters the extractor passes on
CONVERSATION_KEYS = (
    "people",
    "match",
    "min_share",
    "min_sec",
    "max_sec",
    "kind",
    "sentences",
    "wps_min",
    "wps_max",
    "rate_min",
    "rate_max",
    "speakers_min",
    "speakers_max",
    "questions_min",
    "language",
    "q",
    "folder",
    "date_from",
    "date_to",
)

_BOUNDS = {
    "threshold": (0.5, 0.99),
    "min_sec": (0.3, 60.0),
    "max_sec": (0.5, 120.0),
    "rate_min": (0.0, 100.0),
    "rate_max": (1.0, 200.0),
    "min_certainty": (0, 100),
    "coverage": (0.5, 1.0),
    "label_share": (0.0, 1.0),
    "join_gap": (0.0, 2.0),
    "padding": (0.0, 1.0),
    "per_recording": (0, 100000),
    "max_hours": (0.0, 10000.0),
    "max_clips": (0, 10_000_000),
}


def clean_options(raw: dict | None) -> dict:
    """The options of a request, typed and within bounds."""
    raw = raw or {}
    out = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        if key not in raw or raw[key] is None:
            continue
        v = raw[key]
        if key == "conversations":
            out[key] = conversation_filters(v)
            continue
        try:
            if isinstance(default, bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                v = int(float(v))
            elif isinstance(default, float):
                v = float(v)
            else:
                v = str(v).strip()
        except (TypeError, ValueError):
            continue
        if key in _BOUNDS:
            lo, hi = _BOUNDS[key]
            v = min(max(v, lo), hi)
        out[key] = v
    if out["include"] not in ("named", "recognized"):
        out["include"] = DEFAULTS["include"]
    if out["clean"] not in ("best", "strict"):
        out["clean"] = DEFAULTS["clean"]
    if out["order"] not in ("time", "best", "spread", "sure"):
        out["order"] = DEFAULTS["order"]
    for k in ("date_from", "date_to"):
        if out[k] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", out[k]):
            out[k] = ""
    if out["max_sec"] < out["min_sec"]:
        out["max_sec"] = out["min_sec"]
    if out["rate_max"] < out["rate_min"]:
        out["rate_max"] = out["rate_min"]
    return out


def conversation_filters(raw) -> dict:
    """The conversation filters of a request: known names only, values as
    text (or a list of texts for the people) — the conversation search
    reads them leniently itself (query.parse_filters)."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for k in CONVERSATION_KEYS:
        v = raw.get(k)
        if isinstance(v, (list, tuple)):
            vals = [
                str(x).strip()[:200]
                for x in v[:20]
                if isinstance(x, (str, int, float)) and str(x).strip()
            ]
            if vals:
                out[k] = vals if k == "people" else ",".join(vals)
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            text = str(v).strip()[: 4096 if k == "folder" else 200]
            if text and text != "0":
                out[k] = text
    return out


def conversation_ranges(filters: dict) -> dict:
    """The time ranges of the conversations ``filters`` find, per recording
    (query.ranges) — every copy, as the extractor keeps copies apart."""
    from whisper_tm.config import config
    from whisper_tm.conversations import query, store
    from whisper_tm.extensions import db

    if not store.enabled():
        raise ValueError(
            "the conversation statistics are switched off (Settings → Search → Conversations)"
        )
    if db is None:
        raise ValueError("no database")
    f, ignored = query.parse_filters({**filters, "copies": "all"}, media_dir=config.media_dir)
    got = query.ranges(db, f)
    got["ignored"] = ignored + got["ignored"]
    return got


def _in_ranges(a: float, b: float, spans) -> bool:
    mid = (a + b) / 2
    return any(x <= mid <= y for x, y in spans)


# ── Which voices ─────────────────────────────────────────────────────────

#: The person's named voices the similarity is measured against, at most.
#: Hundreds of thousands of voices against a person's tens of thousands of named ones
#: are trillions of multiply-adds — seconds on the CPU for each change of a filter. Every
#: n-th named voice keeps the breadth (recordings, years) at a fraction.
REFERENCE_AT_MOST = 3000

_sim_lock = threading.Lock()
_sim_held: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
_label_windows: dict[tuple, dict] = {}


def _gallery_key(gallery: dict, person: str) -> str | None:
    from whisper_tm.database.speaker import fold_name

    want = fold_name(person)
    return next((n for n in gallery if fold_name(n) == want), None)


def similarity(b: dict, person: str) -> tuple[np.ndarray, np.ndarray]:
    """Per voice of the inventory: its best similarity to one of the
    person's named voices, and whether this person's centroid is the
    nearest of all named people. Held per inventory and naming state."""
    name = _gallery_key(b["gallery"], person)
    n = len(b["X"])
    if name is None or not n:
        return np.zeros(n, np.float32), np.zeros(n, bool)
    E = b["gallery"][name]
    key = (
        b["source"],
        b["inventory_generation"],
        name,
        len(E),
        tuple(sorted((k, len(v)) for k, v in b["gallery"].items())),
    )
    with _sim_lock:
        hit = _sim_held.get(key)
        if hit is not None:
            _sim_held.move_to_end(key)
            return hit
    if len(E) > REFERENCE_AT_MOST:
        E = E[np.linspace(0, len(E) - 1, REFERENCE_AT_MOST).astype(int)]
    X = b["X"]
    best = np.empty(n, np.float32)
    rows = max(1024, int(48_000_000 // max(1, len(E))))
    for a in range(0, n, rows):
        best[a : a + rows] = (X[a : a + rows] @ E.T).max(axis=1)
    names = sorted(b["gallery"])
    from whisper_tm.core.speaker import nrm

    C = nrm(np.stack([b["gallery"][k].mean(axis=0) for k in names])).astype(np.float32)
    nearest = np.empty(n, np.int32)
    for a in range(0, n, 65536):
        nearest[a : a + 65536] = (X[a : a + 65536] @ C.T).argmax(axis=1)
    mine = nearest == names.index(name)
    with _sim_lock:
        _sim_held[key] = (best, mine)
        while len(_sim_held) > 3:
            _sim_held.popitem(last=False)
    return best, mine


def _windows_per_label(b: dict) -> dict:
    """How many windows each (recording, label) has — for the share of a
    label that belongs to the person."""
    key = (b["source"], b["inventory_generation"])
    held = _label_windows.get(key)
    if held is None:
        held = collections.Counter((m["abs_path"], m["label"]) for m in b["meta"])
        _label_windows.clear()
        _label_windows[key] = held
    return held


def choose_voices(person: str, opts: dict) -> dict:
    """The windows that belong to the person — named, and recognized if
    asked for — with their similarity and speech spans."""
    from whisper_tm.core import speaker_inventory
    from whisper_tm.database.speaker import fold_name, speaker_db

    db = speaker_db()
    b = speaker_inventory.inventory()
    want = fold_name(person)
    names = db.names_per_vector()
    named = {v for v, n in names.items() if fold_name(n) == want}
    trash = db.trash_ids()
    best, mine = similarity(b, person)
    rows: dict[int, dict] = {}
    for vid in named:
        if vid in trash:
            continue
        try:
            i = b["id_to_row"][vid]
        except KeyError:
            continue
        rows[i] = {"named": True}
    recognized = 0
    if opts["include"] == "recognized" and len(best):
        take = np.flatnonzero((best >= opts["threshold"]) & mine)
        for i in take.tolist():
            if i in rows:
                continue
            vid = int(b["ids"][i])
            if vid in trash or vid in names:
                continue  # discarded, or named as someone else
            rows[i] = {"named": False}
            recognized += 1
    ids = [int(b["ids"][i]) for i in rows]
    spans = _spans_of(db, ids)
    out = []
    for i, r in rows.items():
        m = b["meta"][i]
        out.append(
            {
                "id": m["id"],
                "abs_path": m["abs_path"],
                "label": m["label"],
                "start_sec": m["start_sec"],
                "speech_sec": m["speech_sec"],
                "named": r["named"],
                "sim": float(best[i]) if len(best) else 0.0,
                "spans": spans.get(m["id"], []),
            }
        )
    return {
        "windows": out,
        "named": sum(1 for w in out if w["named"]),
        "recognized": recognized,
        "label_windows": _windows_per_label(b),
    }


def _spans_of(db, ids) -> dict[int, list[tuple[float, float]]]:
    """The stored speech spans of these windows."""
    if not ids:
        return {}
    out = {}
    with db.connection() as c:
        t = db._ids_table(c, ids)
        for vid, raw in c.execute(
            f"SELECT v.id, v.parts FROM speaker_vectors v JOIN {t} i "
            "ON i.id = v.id WHERE v.parts IS NOT NULL"
        ):
            spans = parse_spans(raw)
            if spans:
                out[vid] = spans
    return out


# ── Sentences ────────────────────────────────────────────────────────────


def _sentences(pairs) -> dict[tuple[str, str], list[dict]]:
    """The transcript's sentences of these (recording, label) pairs — without
    the quotation marks Whisper put around a line (helpers.edge_quotes):
    nobody speaks them, and a TTS dataset must not learn them."""
    from whisper_tm import extensions as ext
    from whisper_tm.core.helpers import edge_quotes

    if not pairs or ext.db is None:
        return {}
    out: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    with ext.db.connection() as c:
        c.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _voice_pairs "
            "(abs_path TEXT, label TEXT, PRIMARY KEY (abs_path, label))"
        )
        c.execute("DELETE FROM _voice_pairs")
        c.executemany("INSERT OR IGNORE INTO _voice_pairs VALUES (?, ?)", list(pairs))
        for abs_path, label, idx, a, b, text, logprob, phantom, overlapped, language in c.execute(
            "SELECT t.abs_path, s.speaker, s.seg_index, s.start_sec, s.end_sec, "
            "       s.text, s.avg_logprob, s.phantom_score, s.overlapped, "
            "       t.language "
            "FROM _voice_pairs w "
            "JOIN transcripts t ON t.abs_path = w.abs_path "
            "JOIN segments s ON s.transcript_id = t.id AND s.speaker = w.label "
            "ORDER BY t.abs_path, s.start_sec"
        ):
            out[(abs_path, label)].append(
                {
                    "seg": idx,
                    "start": float(a or 0.0),
                    "end": float(b or 0.0),
                    "text": edge_quotes(text)[0],
                    "confidence": logprob,
                    "phantom": float(phantom or 0.0),
                    "overlapped": bool(overlapped),
                    "language": language or "",
                }
            )
        c.execute("DELETE FROM _voice_pairs")
    return out


def _display(abs_path: str, root: str) -> str:
    return paths.shown(abs_path, root) if root else abs_path


# ── A run: the preview, and exactly what an export takes ─────────────────

_runs: collections.OrderedDict[str, dict] = collections.OrderedDict()
_runs_lock = threading.Lock()
#: The preview on the page and the one before it. A run holds every clip:
#: all voices of one well-covered person can be hundreds of thousands of
#: clips (hundreds of hours) — a few hundred MB in the server per run.
RUNS_HELD = 2


def build_run(person: str, raw_opts: dict | None, root: str = "") -> dict:
    """Choose the voices, their sentences and the clips — and hold the
    result, so the export takes exactly what the preview showed."""
    from whisper_tm.core.helpers import artifact_reason

    person = (person or "").strip()
    if not person:
        raise ValueError("choose a person")
    opts = clean_options(raw_opts)
    t0 = time.time()
    chosen = choose_voices(person, opts)
    windows = chosen["windows"]
    if not windows:
        raise ValueError(f'no voices of "{person}" — name some in the map first')
    conv = conversation_ranges(opts["conversations"]) if opts["conversations"] else None

    by_voice: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for w in windows:
        by_voice[(w["abs_path"], w["label"])].append(w)
    dropped: collections.Counter = collections.Counter()
    keep_voices = {}
    folder = opts["folder"].strip("/").lower()
    file_text = opts["file_text"].lower()
    for key, ws in by_voice.items():
        abs_path = key[0]
        shown = _display(abs_path, root)
        date = date_from_name(abs_path)
        if folder and not shown.lower().startswith(folder):
            dropped["folder"] += len(ws)
            continue
        if file_text and file_text not in shown.lower():
            dropped["file name"] += len(ws)
            continue
        if (opts["date_from"] or opts["date_to"]) and not date:
            dropped["no date in the file name"] += len(ws)
            continue
        if opts["date_from"] and date < opts["date_from"]:
            dropped["before the date range"] += len(ws)
            continue
        if opts["date_to"] and date > opts["date_to"]:
            dropped["after the date range"] += len(ws)
            continue
        if conv is not None and abs_path not in conv["ranges"]:
            dropped["not in the chosen conversations"] += len(ws)
            continue
        spans = merge_spans([s for w in ws for s in w["spans"]])
        if not spans:
            if opts["clean"] == "strict":
                dropped["voice without speech spans"] += len(ws)
                continue
            total = chosen["label_windows"].get(key, len(ws))
            if total and len(ws) / total < opts["label_share"]:
                dropped["label shared with other voices"] += len(ws)
                continue
        keep_voices[key] = {
            "spans": spans,
            "starts": [s[0] for s in spans],
            "sim": max(w["sim"] for w in ws),
            "named": any(w["named"] for w in ws),
            "display": shown,
            "date": date,
        }

    sentences = _sentences(list(keep_voices))
    no_transcript = [k for k in keep_voices if k not in sentences]
    if no_transcript:
        dropped["recording without transcript"] += len(no_transcript)
    text_in = opts["text"].lower()
    text_out = opts["text_not"].lower()
    lang = opts["language"].lower()
    clips: list[dict] = []
    considered = 0
    for key, rows in sentences.items():
        v = keep_voices[key]
        picked = []
        for s in rows:
            considered += 1
            dur = s["end"] - s["start"]
            text = s["text"]
            if not text or dur <= 0:
                dropped["empty sentence"] += 1
                continue
            if conv is not None and not _in_ranges(s["start"], s["end"], conv["ranges"][key[0]]):
                dropped["outside the chosen conversations"] += 1
                continue
            if lang and s["language"].lower() != lang:
                dropped["other language"] += 1
                continue
            if text_in and text_in not in text.lower():
                dropped["text filter"] += 1
                continue
            if text_out and text_out in text.lower():
                dropped["text filter"] += 1
                continue
            if opts["skip_phantoms"] and s["phantom"] > 0:
                dropped["marked as phantom"] += 1
                continue
            if opts["skip_artifacts"] and artifact_reason(text) is not None:
                dropped["artifact (Settings → Transcription)"] += 1
                continue
            sure_pct = certainty(s["confidence"])
            if sure_pct is not None and sure_pct < opts["min_certainty"]:
                dropped["Whisper unsure"] += 1
                continue
            if v["spans"]:
                if covered(s["start"], s["end"], v["spans"], v["starts"]) < opts["coverage"]:
                    dropped["outside the voice's speech spans"] += 1
                    continue
                clean_by = "spans"
            else:
                if s["overlapped"]:
                    dropped["two speakers at once"] += 1
                    continue
                clean_by = "transcript"
            picked.append({**s, "clean_by": clean_by})
        # Join neighbours, then judge length and rate on the joined clip
        if opts["join_gap"] > 0 and picked:
            joined = [dict(picked[0])]
            for s in picked[1:]:
                last = joined[-1]
                if (
                    s["start"] - last["end"] <= opts["join_gap"]
                    and s["end"] - last["start"] <= opts["max_sec"]
                    and s["clean_by"] == last["clean_by"]
                ):
                    last["end"] = s["end"]
                    last["text"] = f"{last['text']} {s['text']}".strip()
                    last["joined"] = last.get("joined", 1) + 1
                    if s["confidence"] is not None:
                        held = last["confidence"]
                        last["confidence"] = (
                            s["confidence"] if held is None else min(held, s["confidence"])
                        )
                else:
                    joined.append(dict(s))
            picked = joined
        for s in picked:
            dur = s["end"] - s["start"]
            if dur < opts["min_sec"]:
                dropped["too short"] += 1
                continue
            if dur > opts["max_sec"]:
                dropped["too long"] += 1
                continue
            rate = len(s["text"]) / dur
            if rate < opts["rate_min"] or rate > opts["rate_max"]:
                dropped["implausible timing"] += 1
                continue
            a, b = s["start"], s["end"]
            pad = opts["padding"]
            if pad:
                if v["spans"]:
                    span = _span_of(a, b, v["spans"], v["starts"]) or (a, b)
                    a, b = max(span[0], a - pad), min(span[1], b + pad)
                else:
                    a, b = max(0.0, a - pad), b + pad
            conf = s["confidence"]
            sure = min(1.0, max(0.0, (conf if conf is not None else -0.5) + 1.0))
            text = " ".join(s["text"].split())
            clips.append(
                {
                    # Which sentence this is, across previews: the hand edits
                    # are kept by it (`edit_clip`).
                    "key": f"{key[0]}\t{key[1]}\t{s['start']:.2f}\t{s.get('joined', 1)}",
                    "abs_path": key[0],
                    "label": key[1],
                    "display": v["display"],
                    "date": v["date"],
                    "start": round(a, 3),
                    "end": round(b, 3),
                    # The sentence's own bounds (without padding) — where the
                    # cut at the nearest pause looks.
                    "s_start": round(s["start"], 3),
                    "s_end": round(s["end"], 3),
                    "text": text,
                    "confidence": None if conf is None else round(float(conf), 3),
                    "certainty": certainty(conf),
                    "sim": round(v["sim"], 3),
                    "named": v["named"],
                    "clean_by": s["clean_by"],
                    "language": s["language"],
                    "score": v["sim"] + 0.25 * sure + (0.1 if s["clean_by"] == "spans" else 0.0),
                    "orig_start": round(a, 3),
                    "orig_end": round(b, 3),
                    "orig_text": text,
                    "state": "",
                    "edited": False,
                }
            )

    # Hand edits made before (a previous preview of this person) apply again
    _apply_edits(person, clips)
    active = [c for c in clips if c["state"] != "dropped"]
    by_hand = [c for c in clips if c["state"] == "dropped"]
    active = _order(active, opts)
    active, capped = _cap(active, opts)
    if capped:
        dropped["over the amount limit"] += capped
    if by_hand:
        dropped["dropped by hand"] += len(by_hand)
    clips = active + by_hand
    stats = _stats(chosen, keep_voices, active, considered, dropped)
    if conv is not None:
        stats["conversations"] = {
            "found": conv["conversations"],
            "recordings": len(conv["ranges"]),
            "people_unknown": conv["people_unknown"],
            "ignored": conv["ignored"],
            "sentences_baseline": conv["sentences_baseline"],
        }
    run = {
        "run_id": uuid.uuid4().hex[:12],
        "person": person,
        "options": opts,
        "created": time.time(),
        "clips": clips,
        "stats": stats,
        "computed_sec": round(time.time() - t0, 2),
    }
    with _runs_lock:
        _runs[run["run_id"]] = run
        while len(_runs) > RUNS_HELD:
            _runs.popitem(last=False)
    return run


def _order(clips: list[dict], opts: dict) -> list[dict]:
    if opts["order"] == "sure":
        return sorted(
            clips,
            key=lambda c: (
                -(c["certainty"] if c["certainty"] is not None else -1),
                -c["sim"],
                c["abs_path"],
                c["start"],
            ),
        )
    if opts["order"] == "best":
        return sorted(
            clips, key=lambda c: (-c["score"], c["date"] or "9999", c["abs_path"], c["start"])
        )
    by_time = sorted(clips, key=lambda c: (c["date"] or "9999", c["abs_path"], c["start"]))
    if opts["order"] != "spread":
        return by_time
    # Round-robin over the recordings: a cap then takes a little of
    # each instead of everything from the first ones.
    per: collections.OrderedDict[str, collections.deque] = collections.OrderedDict()
    for c in by_time:
        per.setdefault(c["abs_path"], collections.deque()).append(c)
    out = []
    while per:
        for k in list(per):
            out.append(per[k].popleft())
            if not per[k]:
                del per[k]
    return out


def _cap(clips: list[dict], opts: dict) -> tuple[list[dict], int]:
    """The per-recording cap, the number cap and the hours cap."""
    out, seen = [], collections.Counter()
    total = 0.0
    limit = opts["max_hours"] * 3600 if opts["max_hours"] else math.inf
    for c in clips:
        if opts["per_recording"] and seen[c["abs_path"]] >= opts["per_recording"]:
            continue
        if opts["max_clips"] and len(out) >= opts["max_clips"]:
            continue
        dur = c["end"] - c["start"]
        if total + dur > limit:
            continue
        seen[c["abs_path"]] += 1
        total += dur
        out.append(c)
    return out, len(clips) - len(out)


def _stats(
    chosen: dict, voices: dict, clips: list[dict], considered: int, dropped: collections.Counter
) -> dict:
    seconds = sum(c["end"] - c["start"] for c in clips)
    per_file: dict[str, list] = {}
    years: collections.Counter = collections.Counter()
    langs: collections.Counter = collections.Counter()
    sure: collections.Counter = collections.Counter()
    for c in clips:
        e = per_file.setdefault(c["abs_path"], [c["display"], 0, 0.0, c["date"]])
        e[1] += 1
        e[2] += c["end"] - c["start"]
        years[(c["date"] or "")[:4] or "unknown"] += c["end"] - c["start"]
        langs[c["language"] or "?"] += 1
        sure[_certainty_band(c["certainty"])] += 1
    return {
        "windows_named": chosen["named"],
        "windows_recognized": chosen["recognized"],
        "voices": len(voices),
        "recordings": len(per_file),
        "sentences": considered,
        "clips": len(clips),
        "seconds": round(seconds, 1),
        "by_spans": sum(1 for c in clips if c["clean_by"] == "spans"),
        "certainty": [[b, sure.get(b, 0)] for b in CERTAINTY_BANDS if sure.get(b)],
        "kept": sum(1 for c in clips if c["state"] == "kept"),
        "edited": sum(1 for c in clips if c["edited"]),
        "by_transcript": sum(1 for c in clips if c["clean_by"] == "transcript"),
        "years": sorted([[y, round(s, 1)] for y, s in years.items()]),
        "languages": langs.most_common(),
        "dropped": dropped.most_common(),
        "top": [
            {"display": d, "clips": n, "seconds": round(s, 1), "date": dt}
            for d, n, s, dt in sorted(per_file.values(), key=lambda e: -e[2])[:300]
        ],
    }


#: How sure Whisper was, in bands — the preview's distribution.
CERTAINTY_BANDS = ("95–100 %", "90–95 %", "80–90 %", "70–80 %", "50–70 %", "below 50 %", "unknown")


def _certainty_band(p: int | None) -> str:
    if p is None:
        return "unknown"
    for floor, band in (
        (95, CERTAINTY_BANDS[0]),
        (90, CERTAINTY_BANDS[1]),
        (80, CERTAINTY_BANDS[2]),
        (70, CERTAINTY_BANDS[3]),
        (50, CERTAINTY_BANDS[4]),
    ):
        if p >= floor:
            return band
    return CERTAINTY_BANDS[5]


def clip_view(c: dict, i: int, words: bool = False) -> dict:
    """One clip as the page shows it — with its words where the transcript
    has them (asked for page by page: a preview of thousands of clips
    would read every transcript)."""
    view = {
        "i": i,
        **{
            k: c[k]
            for k in (
                "key",
                "display",
                "date",
                "start",
                "end",
                "s_start",
                "s_end",
                "text",
                "confidence",
                "certainty",
                "sim",
                "named",
                "clean_by",
                "state",
                "edited",
                "orig_start",
                "orig_end",
                "orig_text",
            )
        },
    }
    if words:
        view["words"] = clip_words(c)
    return view


# ── The words of a clip ──────────────────────────────────────────────────
#
# When word timestamps are on, every word is coloured by how sure it is —
# and highlighted while it plays, a little like subtitles.
#
# Only transcripts made with word timestamps have words: the alignment
# gives each one its time and a score (0–1, how well the sound fits the
# word). They live in the transcript's JSON, not in the database — read
# once per recording and held.

_words_lock = threading.Lock()
_words_held: collections.OrderedDict[str, list | None] = collections.OrderedDict()
WORDS_HELD = 64


def _recording_words(abs_path: str) -> list[dict] | None:
    with _words_lock:
        if abs_path in _words_held:
            _words_held.move_to_end(abs_path)
            return _words_held[abs_path]
    from whisper_tm.core.helpers import find_transcript

    words = None
    try:
        tp = find_transcript(abs_path, abs_path=abs_path)
        if tp is not None:
            with open(tp, encoding="utf-8") as fh:
                data = json.load(fh)
            out = []
            for seg in data.get("segments") or []:
                for w in seg.get("words") or []:
                    # A quotation mark is never spoken — not at a word either
                    text = str(w.get("word") or "").strip().strip("»«‹›").strip()
                    if not text:
                        continue
                    a, b, p = w.get("start"), w.get("end"), w.get("score")
                    out.append(
                        {
                            "w": text,
                            "a": None if a is None else float(a),
                            "b": None if b is None else float(b),
                            "p": None if p is None else round(float(p), 3),
                        }
                    )
            words = out or None
    except (OSError, ValueError, TypeError) as e:
        logger.debug("No words for %s: %s", abs_path, e)
        words = None
    with _words_lock:
        _words_held[abs_path] = words
        while len(_words_held) > WORDS_HELD:
            _words_held.popitem(last=False)
    return words


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def clip_words(c: dict) -> list[dict] | None:
    """The words of the clip's sentence(s), with absolute times — or None:
    no word timestamps, or the text was corrected by hand, or the words
    don't spell the sentence (then colouring them would mislead)."""
    if c.get("text") != c.get("orig_text"):
        return None
    words = _recording_words(c["abs_path"])
    target = _letters(c["text"])
    if not words or not target:
        return None
    # The run of words that spells the sentence, starting near its start.
    # Not a time window: neighbouring sentences touch, and a window of a
    # few hundredths can take the next sentence's first word along. An
    # untimed word in the run (numbers often
    # get no time) still belongs to it.
    near = [
        k
        for k, w in enumerate(words)
        if w["a"] is not None and c["s_start"] - 0.6 <= w["a"] <= c["s_end"] + 0.6
    ]
    for k0 in sorted(near, key=lambda k: abs(words[k]["a"] - c["s_start"])):
        spelled = ""
        for k in range(k0, len(words)):
            spelled += _letters(words[k]["w"])
            if spelled == target:
                return words[k0 : k + 1]
            if not target.startswith(spelled):
                break
    return None


def run_view(run: dict, examples: int = 60) -> dict:
    """What the page shows of a run — the clip list stays on the server
    and comes page by page (`clips_page`)."""
    return {
        "run_id": run["run_id"],
        "person": run["person"],
        "options": run["options"],
        "stats": run["stats"],
        "computed_sec": run["computed_sec"],
        "total": len(run["clips"]),
        "examples": [clip_view(c, i) for i, c in enumerate(run["clips"][:examples])],
    }


CLIP_STATES = ("all", "open", "edited", "kept", "dropped")


def clips_page(run: dict, offset: int = 0, limit: int = 60, state: str = "all") -> dict:
    """A page of the run's clips — all, or those open / edited / kept /
    dropped by hand."""
    want = state if state in CLIP_STATES else "all"
    pick = [
        (i, c)
        for i, c in enumerate(run["clips"])
        if want == "all"
        or (want == "open" and c["state"] == "")
        or (want == "edited" and c["edited"])
        or (want in ("kept", "dropped") and c["state"] == want)
    ]
    offset = max(0, int(offset))
    limit = min(max(1, int(limit)), 500)
    return {
        "run_id": run["run_id"],
        "state": want,
        "total": len(pick),
        "offset": offset,
        "clips": [clip_view(c, i, words=True) for i, c in pick[offset : offset + limit]],
        "totals": totals(run),
    }


def totals(run: dict) -> dict:
    """What an export of the run would take, with the hand edits — and the
    whole recordings behind the clips, for an export of those."""
    active = [c for c in run["clips"] if c["state"] != "dropped"]
    kept = [c for c in active if c["state"] == "kept"]
    sizes = recording_sizes(run)
    recs = {c["abs_path"] for c in active}
    recs_kept = {c["abs_path"] for c in kept}
    return {
        "clips": len(active),
        "seconds": round(sum(c["end"] - c["start"] for c in active), 1),
        "kept": len(kept),
        "kept_seconds": round(sum(c["end"] - c["start"] for c in kept), 1),
        "dropped": len(run["clips"]) - len(active),
        "edited": sum(1 for c in run["clips"] if c["edited"]),
        "recordings": len(recs),
        "recordings_bytes": sum(sizes.get(r, 0) for r in recs),
        "kept_recordings": len(recs_kept),
        "kept_recordings_bytes": sum(sizes.get(r, 0) for r in recs_kept),
    }


def recording_sizes(run: dict) -> dict[str, int]:
    """The file size of every recording of a run — from the library
    (``media_files.file_size``), kept with the run. Looked up once: asking
    a network share for 3,000 files on every page of clips would take seconds."""
    held = run.get("_sizes")
    if held is not None:
        return held
    paths = sorted({c["abs_path"] for c in run["clips"]})
    sizes: dict[str, int] = {}
    try:
        from whisper_tm.extensions import db

        if db:
            with db.connection() as conn:
                for a in range(0, len(paths), 900):
                    chunk = paths[a : a + 900]
                    for r in conn.execute(
                        "SELECT abs_path, file_size FROM media_files WHERE abs_path IN "
                        f"({','.join('?' * len(chunk))})",
                        chunk,
                    ):
                        sizes[r[0]] = int(r[1] or 0)
    except Exception as e:  # noqa: BLE001 — an estimate only
        logger.debug("recording sizes: %s", e)
    run["_sizes"] = sizes
    return sizes


# ── Hand edits ───────────────────────────────────────────────────────────
#
# An in-memory clip editor: sometimes a clip still carries half a word;
# it can then be adjusted by hand and downloaded as a ZIP.
#
# An edit belongs to a SENTENCE (its key), not to a position in one
# preview: the preview is computed again whenever a filter changes, and
# the clip that was number 12 can be number 40 then — or not in it at all
# and back with the next filter. So the edits are kept per person, by key,
# and every preview of that person applies them. In memory: a restart of
# the server forgets them, nothing is written to the transcripts.

_edits_lock = threading.Lock()
_edits: dict[str, dict[str, dict]] = {}


def _edit_store(person: str) -> dict[str, dict]:
    from whisper_tm.database.speaker import fold_name

    with _edits_lock:
        return _edits.setdefault(fold_name(person.strip()), {})


def _apply_edits(person: str, clips: list[dict]) -> None:
    store = _edit_store(person)
    if not store:
        return
    for c in clips:
        e = store.get(c["key"])
        if e:
            _set_clip(c, e)


def _set_clip(c: dict, e: dict) -> None:
    c["start"], c["end"] = e.get("start", c["orig_start"]), e.get("end", c["orig_end"])
    c["text"] = e.get("text", c["orig_text"])
    c["state"] = e.get("state", "")
    c["edited"] = (
        abs(c["start"] - c["orig_start"]) > 0.0005
        or abs(c["end"] - c["orig_end"]) > 0.0005
        or c["text"] != c["orig_text"]
    )


def edit_clip(run: dict, i: int, data: dict) -> dict:
    """Change one clip of a run by hand — its bounds, its text, its state
    ("kept", "dropped" or open) — or put it back as it was (``reset``)."""
    if not 0 <= i < len(run["clips"]):
        raise ValueError("no such clip")
    c = run["clips"][i]
    store = _edit_store(run["person"])
    if data.get("reset"):
        with _edits_lock:
            store.pop(c["key"], None)
        _set_clip(c, {})
        return {"clip": clip_view(c, i, words=True), "totals": totals(run)}
    try:
        start = float(data.get("start", c["start"]))
        end = float(data.get("end", c["end"]))
    except (TypeError, ValueError):
        raise ValueError("start and end must be numbers") from None
    if not (0.0 <= start < end):
        raise ValueError("the clip must start before it ends")
    if end - start < 0.2:
        raise ValueError("a clip of less than 0.2 seconds")
    if end - start > 120 or abs(start - c["orig_start"]) > 60 or abs(end - c["orig_end"]) > 60:
        raise ValueError("that is far from the sentence — at most a minute either way")
    text = " ".join(str(data.get("text", c["text"]) or "").split())[:2000]
    if not text:
        raise ValueError("a clip needs its text")
    state = data.get("state", c["state"])
    if state not in ("", "kept", "dropped"):
        raise ValueError("state is kept, dropped or empty")
    e = {"start": round(start, 3), "end": round(end, 3), "text": text, "state": state}
    with _edits_lock:
        store[c["key"]] = e
    _set_clip(c, e)
    return {"clip": clip_view(c, i, words=True), "totals": totals(run)}


def forget_edits(person: str) -> int:
    from whisper_tm.database.speaker import fold_name

    with _edits_lock:
        return len(_edits.pop(fold_name(person.strip()), {}) or {})


def get_run(run_id: str) -> dict | None:
    with _runs_lock:
        return _runs.get(str(run_id or ""))


# ── Cutting audio ────────────────────────────────────────────────────────


class _Stop:
    """What ``run_cancellable`` asks (``job.cancel``) — tied to the stop
    button of the export."""

    def __init__(self, event: threading.Event):
        self._event = event

    @property
    def cancel(self) -> bool:
        return self._event.is_set()


def _decode(
    abs_path: str, a: float, b: float, rate: int, tmp_dir: Path, stop: threading.Event | None = None
) -> np.ndarray:
    """``[a, b]`` of a recording as mono int16 samples at ``rate``.

    Through a short WAV file, not a pipe: every external tool of this app
    runs through ``run_cancellable``, and that one hands back text.
    """
    from whisper_tm.core.audio_utils import run_cancellable

    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"cut-{uuid.uuid4().hex[:10]}.wav"
    try:
        r = run_cancellable(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{max(0.0, a):.3f}",
                "-t",
                f"{max(0.05, b - a):.3f}",
                *media_formats.source(abs_path),
                "-vn",
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                str(rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(target),
            ],
            job=_Stop(stop) if stop is not None else None,
            timeout=600,
            label="voice extract",
        )
        if r.returncode != 0 or not target.exists():
            raise RuntimeError((r.stderr or "ffmpeg failed").strip()[-200:])
        with wave.open(str(target), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
    finally:
        target.unlink(missing_ok=True)


def _normalize(x: np.ndarray, target_db: float = -20.0, peak_db: float = -1.0) -> np.ndarray:
    """Even loudness: RMS to ``target_db``, peaks kept below ``peak_db``."""
    if not len(x):
        return x
    f = x.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(f * f)))
    if rms < 1e-5:
        return x
    gain = 10 ** (target_db / 20) / rms
    peak = float(np.max(np.abs(f))) * gain
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        gain *= limit / peak
    return np.clip(f * gain * 32768.0, -32768, 32767).astype(np.int16)


def _wav_bytes(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


#: Cutting at the nearest pause. Whisper's sentence bounds are close, but
#: not exact, and padding reaches into what comes before and after: half a
#: word of the previous sentence is the most common flaw of a clip. So each
#: bound looks for the quietest moment near the sentence's own start and
#: end — a little before the start, a little after the end — and cuts
#: there when it is a pause. Without a pause close by,
#: the sentence's own bound is taken instead of the padding: padding into
#: ongoing speech is exactly what brings in half words. The editor on the
#: page uses the same rule (voices_clipedit.js).
#:
#: **What a pause is, measured against the recording.** A fixed share of
#: the speech's loudness finds none in a noisy recording: where the
#: background noise sits at a large share of the voice, nothing ever gets
#: quieter. So a
#: pause is close to the recording's own floor — the 10th percentile of the
#: loudness around the clip — a quarter of the way up to the voice's median.
#: When floor and voice don't part (the floor above 60 % of the voice:
#: music, a crowd, someone talking on), there is no pause to trust.
SNAP_FRAME = 0.02
SNAP_OUT, SNAP_IN = 0.25, 0.1  # look this far outside / inside the sentence
SNAP_DIP = 0.25  # a pause: this far from the floor to the voice
SNAP_FLOOR_MAX = 0.6  # floor above this share of the voice: no pauses
SNAP_MARGIN = 0.4  # decoded beyond a clip, to have room to look


def _energy(x: np.ndarray, rate: int) -> np.ndarray:
    """Loudness (RMS) per frame of ``SNAP_FRAME`` seconds."""
    n = max(1, int(rate * SNAP_FRAME))
    frames = len(x) // n
    if frames < 1:
        return np.zeros(0, np.float32)
    f = x[: frames * n].astype(np.float32).reshape(frames, n) / 32768.0
    return np.sqrt((f * f).mean(axis=1))


def snap_bounds(
    x: np.ndarray, rate: int, x_from: float, s_start: float, s_end: float, start: float, end: float
) -> tuple[float, float]:
    """A clip's bounds at the quietest moment near its sentence's start and
    end, where that is a pause; else the sentence's own bounds.

    ``x`` holds the audio from ``x_from`` seconds of the recording on.
    """
    e = _energy(x, rate)
    if not len(e):
        return start, end

    def frame(t: float) -> int:
        return int((t - x_from) / SNAP_FRAME)

    i0, i1 = max(0, frame(s_start)), min(len(e), frame(s_end))
    if i1 - i0 < 3:
        return start, end
    voice = float(np.median(e[i0:i1]))
    around = e[max(0, frame(s_start - 1.0)) : min(len(e), frame(s_end + 1.0))]
    floor = float(np.percentile(around, 10))
    if voice <= 0 or floor > SNAP_FLOOR_MAX * voice:
        quiet = -1.0  # no pause to trust: the sentence bounds
    else:
        quiet = max(floor + SNAP_DIP * (voice - floor), 1e-4)

    def quietest(lo: float, hi: float) -> float | None:
        a, b = max(0, frame(lo)), min(len(e), frame(hi) + 1)
        if b <= a:
            return None
        j = a + int(np.argmin(e[a:b]))
        return x_from + (j + 0.5) * SNAP_FRAME if e[j] <= quiet else None

    ns = quietest(s_start - SNAP_OUT, s_start + SNAP_IN)
    ne = quietest(s_end - SNAP_IN, s_end + SNAP_OUT)
    ns = s_start if ns is None else ns
    ne = s_end if ne is None else ne
    if ne - ns < 0.3:
        return start, end
    return round(ns, 3), round(ne, 3)


def _context(run: dict, i: int, pad: float) -> tuple[float, float]:
    c = run["clips"][i]
    return (max(0.0, min(c["start"], c["s_start"]) - pad), max(c["end"], c["s_end"]) + pad)


def context_wav(
    run: dict, i: int, pad: float = 2.5, rate: int = 16000, tmp_dir: Path | None = None
) -> tuple[bytes, float]:
    """The clip with ``pad`` seconds around it, as WAV — what the editor on
    the page holds in memory to show, play and trim without asking again.
    Returns the bytes and where in the recording they start."""
    from whisper_tm.config import config

    if not 0 <= i < len(run["clips"]):
        raise ValueError("no such clip")
    a, b = _context(run, i, max(0.0, min(float(pad), 10.0)))
    x = _decode(
        run["clips"][i]["abs_path"],
        a,
        b,
        rate,
        tmp_dir or Path(config.dirs["work"]) / "voice_export_tmp",
    )
    return _wav_bytes(x, rate), a


def clip_wav(run: dict, i: int, rate: int = 22050, tmp_dir: Path | None = None) -> bytes:
    """One clip as WAV, as it stands (with the hand edits)."""
    from whisper_tm.config import config

    if not 0 <= i < len(run["clips"]):
        raise ValueError("no such clip")
    c = run["clips"][i]
    x = _decode(
        c["abs_path"],
        c["start"],
        c["end"],
        rate,
        tmp_dir or Path(config.dirs["work"]) / "voice_export_tmp",
    )
    return _wav_bytes(x, rate)


def _clusters(
    clips: list[tuple[int, dict]], gap: float = 30.0, longest: float = 300.0, margin: float = 0.0
) -> list[tuple[float, float, list]]:
    """One recording's clips in pieces to decode at once: neighbours closer
    than ``gap``, at most ``longest`` seconds per piece, each piece reaching
    ``margin`` beyond its clips and sentences (room to look for a pause)."""

    def lo(c):
        return max(0.0, min(c["start"], c.get("s_start", c["start"])) - margin)

    def hi(c):
        return max(c["end"], c.get("s_end", c["end"])) + margin

    out: list[list] = []
    for i, c in sorted(clips, key=lambda ic: ic[1]["start"]):
        if out and lo(c) - out[-1][1] <= gap and hi(c) - out[-1][0] <= longest:
            out[-1][1] = max(out[-1][1], hi(c))
            out[-1][2].append((i, c))
        else:
            out.append([lo(c), hi(c), [(i, c)]])
    return [(a, b, cs) for a, b, cs in out]


def cut_clip(run: dict, i: int, target: Path, tmp_dir: Path) -> Path:
    """One clip as MP3 — for listening on the page before exporting."""
    from whisper_tm.core.audio_utils import run_cancellable

    c = run["clips"][i]
    tmp = target.with_suffix(".mp3.tmp")
    r = run_cancellable(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{c['start']:.3f}",
            "-t",
            f"{max(0.05, c['end'] - c['start']):.3f}",
            *media_formats.source(c["abs_path"]),
            "-vn",
            "-ac",
            "1",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            "-f",
            "mp3",
            str(tmp),
        ],
        job=None,
        timeout=120,
        label="voice clip",
    )
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 500:
        tmp.unlink(missing_ok=True)
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[-200:])
    tmp.replace(target)
    return target


# ── The export job ───────────────────────────────────────────────────────

#: How an export is written: a TTS dataset, one audio file — or the whole
#: recordings the person speaks in, as they are (all files the speaker
#: is in, with all the filters).
MODES = ("dataset", "single", "recordings")

FORMATS = {
    # extension, ffmpeg codec options, bytes per second of audio (estimate)
    "flac": ("flac", ["-c:a", "flac"], None),
    "wav": ("wav", ["-c:a", "pcm_s16le", "-rf64", "auto"], None),
    "mp3": ("mp3", ["-c:a", "libmp3lame", "-b:a", "192k"], 24_000),
    "opus": ("opus", ["-c:a", "libopus", "-b:a", "64k"], 8_000),
}
RATES = (16000, 22050, 24000, 44100, 48000)

_job_lock = threading.Lock()
_job: dict = {"running": False}
_stop = threading.Event()


def exports_dir() -> Path:
    """``<clips folder>/voice-exports`` — typically on the media storage, not
    on the system disk the work folder shares with everything else."""
    from whisper_tm.config import config

    return Path(config.dirs["clips"]) / "voice-exports"


def dest_dir(raw: str) -> Path:
    """The folder an export goes into: the exports folder when ``raw`` is
    empty, else ``raw`` — checked, since it comes from the request. It must
    be a full path to a folder that exists, lies in the allowed trees
    (``security.allowed_roots``, the ones the folder browser shows) and can
    be written. Raises ValueError with the reason."""
    import os

    from whisper_tm.core.security import root_allowed

    text = str(raw or "").strip()
    if not text:
        return exports_dir()
    p = Path(text).expanduser()
    if not p.is_absolute():
        raise ValueError("give the full path of the folder, starting with /")
    try:
        p = p.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"unusable path: {e}") from None
    if not p.is_dir():
        raise ValueError(f"there is no folder {p}")
    if not root_allowed(p):
        raise ValueError(
            f"{p} lies outside the allowed folders (security.allowed_roots in config.yaml)"
        )
    if not os.access(p, os.W_OK | os.X_OK):
        raise ValueError(f"{p} can't be written (read-only, or no permission)")
    return p


def dest_info(raw: str) -> dict:
    """For the page, while a folder is typed or picked: the folder as it
    will be used and its free space — or why it can't be."""
    try:
        p = dest_dir(raw)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    probe = p
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent  # the exports folder comes later
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = None
    return {"ok": True, "path": str(p), "free": free, "exports": not str(raw or "").strip()}


def _free_base(folder: Path, base: str, names) -> str:
    """``base``, or ``base-2``, ``base-3`` … — the first for which none of
    ``names(base)`` exists in ``folder``. An export never replaces a file
    or folder that is already there."""
    candidate, n = base, 1
    while any(
        (folder / name).exists() or (folder / name).is_symlink() for name in names(candidate)
    ):
        n += 1
        candidate = f"{base}-{n}"
    return candidate


class _ZipOut:
    """The dataset as one ZIP. Created exclusively (``x``): it never opens
    a file that is already there."""

    def __init__(self, path: Path):
        self.path = path
        self.zf = zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED, allowZip64=True)

    def add(self, name: str, data) -> None:
        self.zf.writestr(name, data)

    def close(self) -> None:
        self.zf.close()

    def abort(self) -> None:
        try:
            self.zf.close()
        except Exception:  # noqa: BLE001 — removed next
            pass
        self.path.unlink(missing_ok=True)


class _DirOut:
    """The dataset unpacked: a new folder with ``wavs/`` and the metadata.
    ``mkdir`` without ``exist_ok`` — it never writes into a folder that is
    already there, so removing it after a stop removes only its own."""

    def __init__(self, path: Path):
        self.path = path
        path.mkdir()
        try:
            (path / "wavs").mkdir()
        except BaseException:
            path.rmdir()
            raise

    def add(self, name: str, data) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(self.path / name, "xb") as f:
            f.write(data)

    def close(self) -> None:
        pass

    def abort(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def export_options(raw: dict | None) -> dict:
    raw = raw or {}
    mode = raw.get("mode") if raw.get("mode") in MODES else "dataset"
    fmt = raw.get("format") if raw.get("format") in FORMATS else "flac"
    try:
        rate = int(raw.get("sample_rate") or 22050)
    except (TypeError, ValueError):
        rate = 22050
    if rate not in RATES:
        rate = 22050
    try:
        gap = min(max(float(raw.get("gap", 0.4)), 0.0), 5.0)
    except (TypeError, ValueError):
        gap = 0.4

    def flag(key: str, default: bool) -> bool:
        v = raw.get(key, default)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on")

    dest = str(raw.get("dest") or "").strip()
    return {
        "mode": mode,
        "format": fmt,
        "sample_rate": rate,
        "gap": gap,
        "normalize": flag("normalize", True),
        # Cut every clip not trimmed by hand at the nearest pause
        "snap": flag("snap", True),
        # Only the clips marked "keep" in the editor
        "only_kept": flag("only_kept", False),
        "name": slug(raw.get("name") or "", ""),
        # A folder of your own instead of the exports folder, and there
        # the dataset as files instead of a ZIP (a download needs one file)
        "dest": dest,
        "unpacked": (bool(dest) and mode in ("dataset", "recordings") and flag("unpacked", False)),
    }


def chosen_clips(run: dict, eo: dict) -> list[tuple[int, dict]]:
    """The clips an export takes: none dropped by hand, and with
    ``only_kept`` only the ones kept."""
    return [
        (i, c)
        for i, c in enumerate(run["clips"])
        if c["state"] != "dropped" and (not eo.get("only_kept") or c["state"] == "kept")
    ]


def chosen_recordings(run: dict, eo: dict) -> list[str]:
    """The recordings behind the clips an export takes — in the order of
    their first clip."""
    return list(dict.fromkeys(c["abs_path"] for _i, c in chosen_clips(run, eo)))


def estimate_bytes(run: dict, eo: dict) -> int:
    if eo["mode"] == "recordings":
        sizes = recording_sizes(run)
        return sum(sizes.get(r, 0) for r in chosen_recordings(run, eo)) + 2_000_000
    clips = [c for _i, c in chosen_clips(run, eo)]
    seconds = sum(c["end"] - c["start"] for c in clips)
    pcm = eo["sample_rate"] * 2
    if eo["mode"] == "dataset":
        return int(seconds * pcm * 1.01 + 44 * len(clips) + 2_000_000)
    per_sec = FORMATS[eo["format"]][2]
    total = seconds + eo["gap"] * len(clips)
    if per_sec is None:
        per_sec = pcm * (0.62 if eo["format"] == "flac" else 1.0)
    return int(total * per_sec + 2_000_000)


def export_status() -> dict:
    with _job_lock:
        state = {k: v for k, v in _job.items() if k != "thread"}
    state["files"] = list_exports()
    return state


def start_export(run_id: str, raw: dict | None) -> dict:
    """Start the export of one preview run in the background."""
    run = get_run(run_id)
    if run is None:
        raise ValueError("the preview is gone — compute it again")
    eo = export_options(raw)
    if not chosen_clips(run, eo):
        raise ValueError(
            "no clips to export" + (" — none is marked “keep”" if eo["only_kept"] else "")
        )
    target_dir = dest_dir(eo["dest"])
    target_dir.mkdir(parents=True, exist_ok=True)
    need = estimate_bytes(run, eo)
    free = shutil.disk_usage(target_dir).free
    if need * 1.1 + 500_000_000 > free:
        raise ValueError(
            f"not enough space in {target_dir}: needs about "
            f"{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free"
        )
    with _job_lock:
        if _job.get("running"):
            raise RuntimeError("an export is already running")
        _stop.clear()
        _job.clear()
        _job.update(
            running=True,
            run_id=run_id,
            person=run["person"],
            mode=eo["mode"],
            format=eo["format"],
            total=(
                len(chosen_recordings(run, eo))
                if eo["mode"] == "recordings"
                else len(chosen_clips(run, eo))
            ),
            done=0,
            seconds_done=0.0,
            started=time.time(),
            error=None,
            result=None,
            estimate_bytes=need,
            current="",
            dest=str(target_dir) if eo["dest"] else "",
        )
        t = threading.Thread(
            target=_export_thread, args=(run, eo), name="voice-export", daemon=True
        )
        _job["thread"] = t
    t.start()
    return {"ok": True, "estimate_bytes": need}


def stop_export() -> bool:
    with _job_lock:
        running = bool(_job.get("running"))
    if running:
        _stop.set()
    return running


def _export_thread(run: dict, eo: dict) -> None:
    try:
        result = export(run, eo, _stop, progress=_progress)
        with _job_lock:
            _job.update(running=False, result=result, finished=time.time())
    except InterruptedError:
        with _job_lock:
            _job.update(running=False, error="stopped", finished=time.time())
    except Exception as e:  # noqa — reported on the page
        logger.exception("Voice export failed")
        with _job_lock:
            _job.update(running=False, error=str(e)[:300], finished=time.time())


def _progress(**kw) -> None:
    with _job_lock:
        _job.update(kw)


def export(
    run: dict,
    eo: dict,
    stop: threading.Event,
    progress=lambda **kw: None,
    tmp_dir: Path | None = None,
) -> dict:
    """Cut every clip of the run and write the dataset or the one file.

    Recording by recording, neighbouring clips decoded in one piece; a
    recording that can't be read is skipped and counted. Written to a
    ``.part`` name first and renamed at the end — a stopped or failed
    export leaves nothing that looks finished.
    """
    from whisper_tm.config import config

    rate = eo["sample_rate"]
    person = run["person"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = eo["name"] or f"{slug(person)}-{stamp}"
    target_dir = dest_dir(eo.get("dest") or "")
    target_dir.mkdir(parents=True, exist_ok=True)
    if eo["mode"] == "recordings":
        return _export_recordings(
            run,
            eo,
            stop,
            progress,
            target_dir,
            base if eo["name"] else f"{slug(person)}-recordings-{stamp}",
        )
    tmp_dir = tmp_dir or Path(config.dirs["work"]) / "voice_export_tmp"
    clips = list(run["clips"])
    take = [i for i, _c in chosen_clips(run, eo)]
    if eo["mode"] == "single":
        # One file runs through the recordings in time order.
        order = sorted(
            take,
            key=lambda i: (clips[i]["date"] or "9999", clips[i]["abs_path"], clips[i]["start"]),
        )
    else:
        order = take
    ids = {i: f"{slug(person)}_{n + 1:05d}" for n, i in enumerate(order)}
    per_file: collections.OrderedDict[str, list] = collections.OrderedDict()
    for i in order:
        per_file.setdefault(clips[i]["abs_path"], []).append((i, clips[i]))

    left: collections.Counter = collections.Counter()
    written: list[dict] = []
    done = 0
    seconds_done = 0.0

    def pieces():
        """Decoded clips, recording by recording, in the order above."""
        for abs_path, items in per_file.items():
            if stop.is_set():
                raise InterruptedError("stopped")
            if not Path(abs_path).exists():
                left["recording unreachable"] += len(items)
                yield abs_path, None
                continue
            for a, b, cs in _clusters(items, margin=SNAP_MARGIN if eo.get("snap") else 0.0):
                if stop.is_set():
                    raise InterruptedError("stopped")
                try:
                    x = _decode(abs_path, a, b, rate, tmp_dir, stop)
                except InterruptedError:
                    raise
                except Exception as e:
                    logger.warning(
                        "voice export: %s [%.1f-%.1f] not readable: %s", abs_path, a, b, e
                    )
                    left["recording not readable"] += len(cs)
                    yield abs_path, None
                    continue
                out = []
                for i, c in cs:
                    cs_, ce_ = c["start"], c["end"]
                    # A clip trimmed by hand is cut where the hand put it
                    if eo.get("snap") and not c.get("edited"):
                        cs_, ce_ = snap_bounds(x, rate, a, c["s_start"], c["s_end"], cs_, ce_)
                    c = {**c, "start": cs_, "end": ce_}
                    lo = int(round((cs_ - a) * rate))
                    hi = int(round((ce_ - a) * rate))
                    seg = x[max(0, lo) : max(0, min(hi, len(x)))]
                    if len(seg) < 0.5 * max(1, hi - lo):
                        left["cut short (end of the recording)"] += 1
                        continue
                    if eo["normalize"]:
                        seg = _normalize(seg)
                    out.append((i, c, seg))
                yield abs_path, sorted(out, key=lambda t: order_pos[t[0]])

    order_pos = {i: n for n, i in enumerate(order)}
    if eo["mode"] == "dataset":
        suffix = "" if eo.get("unpacked") else ".zip"
        base = _free_base(target_dir, base, lambda b: (b + suffix, b + suffix + ".part"))
        final = target_dir / f"{base}{suffix}"
        part = final.with_name(final.name + ".part")
        out = _DirOut(part) if eo.get("unpacked") else _ZipOut(part)
        try:
            for abs_path, items in pieces():
                progress(current=_display(abs_path, str(config.media_dir)))
                for i, c, seg in items or []:
                    name = f"wavs/{ids[i]}.wav"
                    out.add(name, _wav_bytes(seg, rate))
                    dur = len(seg) / rate
                    written.append(
                        {
                            "id": ids[i],
                            "file": name,
                            "i": i,
                            "words": _cut_words(c),
                            "duration": round(dur, 3),
                            **c,
                        }
                    )
                    done += 1
                    seconds_done += dur
                    progress(done=done, seconds_done=round(seconds_done, 1))
            if not written:
                raise RuntimeError(
                    "no clip could be cut — "
                    + ", ".join(f"{n} {why}" for why, n in left.most_common())
                )
            written.sort(key=lambda w: order_pos[w["i"]])
            out.add("metadata.csv", _metadata_csv(written))
            out.add("metadata.jsonl", _metadata_jsonl(written, person))
            out.add("README.txt", _readme(run, eo, written, left))
            out.close()
            part.rename(final)
        except BaseException:
            out.abort()
            raise
        files = [final.name]
    else:
        ext, codec, _bps = FORMATS[eo["format"]]
        base = _free_base(
            target_dir, base, lambda b: (f"{b}.{ext}", f"{b}.part.{ext}", f"{b}.srt", f"{b}.jsonl")
        )
        final = target_dir / f"{base}.{ext}"
        part = final.with_name(f"{base}.part.{ext}")
        srt_path = target_dir / f"{base}.srt"
        index_path = target_dir / f"{base}.jsonl"
        gap = np.zeros(int(eo["gap"] * rate), np.int16)
        enc = _Encoder(part, rate, codec)
        position = 0.0
        try:
            for abs_path, items in pieces():
                progress(current=_display(abs_path, str(config.media_dir)))
                for i, c, seg in items or []:
                    enc.write(seg)
                    dur = len(seg) / rate
                    written.append(
                        {
                            "id": ids[i],
                            "i": i,
                            "at": round(position, 3),
                            "words": _cut_words(c),
                            "duration": round(dur, 3),
                            **c,
                        }
                    )
                    position += dur
                    if len(gap):
                        enc.write(gap)
                        position += len(gap) / rate
                    done += 1
                    seconds_done += dur
                    progress(done=done, seconds_done=round(seconds_done, 1))
            if not written:
                raise RuntimeError(
                    "no clip could be cut — "
                    + ", ".join(f"{n} {why}" for why, n in left.most_common())
                )
            enc.close()
            part.replace(final)
        except BaseException:
            enc.abort()
            part.unlink(missing_ok=True)
            raise
        for path, text in (
            (srt_path, _srt(written)),
            (index_path, _metadata_jsonl(written, person)),
        ):
            with open(path, "x", encoding="utf-8") as f:
                f.write(text)
        files = [final.name, srt_path.name, index_path.name]
    shutil.rmtree(tmp_dir, ignore_errors=True)
    from whisper_tm.core.autoscan import note_made

    note_made(*(target_dir / f for f in files))
    size = sum(_size(target_dir / f) for f in files)
    logger.info(
        "Voice export %s: %d clips, %.1f min, %s → %s",
        person,
        len(written),
        seconds_done / 60,
        dict(left),
        target_dir / files[0],
    )
    return {
        "files": files,
        "clips": len(written),
        "seconds": round(seconds_done, 1),
        "bytes": size,
        "left": left.most_common(),
        "folder": str(target_dir),
        # Not in the exports folder: the page says where it went instead
        # of offering a download
        "dest": str(target_dir) if eo.get("dest") else "",
        "unpacked": bool(eo.get("unpacked")),
    }


def _arcname(abs_path: str, root: str) -> str:
    """Where a recording goes inside the export: its path below the media
    root (or its full path without the leading slash) — the folders keep
    files of the same name apart."""
    rel = _display(abs_path, root).lstrip("/")
    return "recordings/" + "/".join(part for part in rel.split("/") if part not in ("", ".", ".."))


def _export_recordings(
    run: dict, eo: dict, stop: threading.Event, progress, target_dir: Path, base: str
) -> dict:
    """The whole recordings behind the chosen clips, copied as they are —
    as one ZIP (stored, not compressed: audio and video don't shrink) or
    unpacked into a folder, with ``recordings.csv`` (path, the person's
    clips and speech in it, date) and a README."""
    import csv

    from whisper_tm.config import config

    root = str(config.media_dir)
    per: collections.OrderedDict[str, list] = collections.OrderedDict()
    for _i, c in chosen_clips(run, eo):
        per.setdefault(c["abs_path"], []).append(c)
    unpacked = bool(eo.get("unpacked"))
    suffix = "" if unpacked else ".zip"
    base = _free_base(target_dir, base, lambda b: (b + suffix, b + suffix + ".part"))
    final = target_dir / f"{base}{suffix}"
    part = final.with_name(final.name + ".part")
    left: collections.Counter = collections.Counter()
    rows, done, size = [], 0, 0
    zf = None
    if unpacked:
        part.mkdir()
    else:
        zf = zipfile.ZipFile(part, "x", compression=zipfile.ZIP_STORED, allowZip64=True)
    try:
        for abs_path, cs in per.items():
            if stop.is_set():
                raise InterruptedError("stopped")
            progress(current=_display(abs_path, root))
            src = Path(abs_path)
            if not src.is_file():
                left["recording unreachable"] += 1
                continue
            arc = _arcname(abs_path, root)
            try:
                if zf is not None:
                    with open(src, "rb") as fin, zf.open(arc, "w", force_zip64=True) as fout:
                        while chunk := fin.read(8 * 1024 * 1024):
                            if stop.is_set():
                                raise InterruptedError("stopped")
                            fout.write(chunk)
                else:
                    dst = part / arc
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            except InterruptedError:
                raise
            except OSError as e:
                logger.warning("voice export: %s not copied: %s", abs_path, e)
                left["recording not readable"] += 1
                continue
            n = src.stat().st_size
            size += n
            done += 1
            rows.append(
                {
                    "file": arc,
                    "source": abs_path,
                    "date": cs[0]["date"],
                    "clips": len(cs),
                    "seconds": round(sum(c["end"] - c["start"] for c in cs), 1),
                    "bytes": n,
                }
            )
            progress(done=done, seconds_done=round(sum(r["seconds"] for r in rows), 1))
        if not rows:
            raise RuntimeError(
                "no recording could be copied — "
                + ", ".join(f"{n} {why}" for why, n in left.most_common())
            )
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["file", "source", "date", "clips", "seconds", "bytes"])
        w.writeheader()
        w.writerows(rows)
        o = run["options"]
        readme = (
            "\n".join(
                [
                    f"Recordings with the voice of {run['person']} — exported by "
                    f"Whisper Transcription Manager on "
                    f"{time.strftime('%Y-%m-%d %H:%M')}",
                    "",
                    "Use a person's recordings and voice only with their consent.",
                    "",
                    f"{len(rows)} recordings, {size / 1e9:.2f} GB, copied as they are.",
                    "recordings/      the files, in their folders below the media folder",
                    "recordings.csv   per file: where it came from, its date (from the name),",
                    "                 how many of the person's sentences it holds and their",
                    "                 seconds — the filters of the preview decided which",
                    "",
                    "Voices: "
                    + (
                        "named ones only"
                        if o["include"] == "named"
                        else f"named ones and recognized ones (similarity >= {o['threshold']})"
                    ),
                    "Only the recordings of the clips marked to keep."
                    if eo.get("only_kept")
                    else "",
                ]
                + (
                    [""] + ["Left out:"] + [f"  {n} {why}" for why, n in left.most_common()]
                    if left
                    else []
                )
            )
            + "\n"
        )
        if zf is not None:
            zf.writestr("recordings.csv", out.getvalue())
            zf.writestr("README.txt", readme)
            zf.close()
        else:
            (part / "recordings.csv").write_text(out.getvalue(), encoding="utf-8")
            (part / "README.txt").write_text(readme, encoding="utf-8")
        part.rename(final)
    except BaseException:
        if zf is not None:
            try:
                zf.close()
            except Exception:  # noqa: BLE001 — removed next
                pass
            part.unlink(missing_ok=True)
        else:
            shutil.rmtree(part, ignore_errors=True)
        raise
    from whisper_tm.core.autoscan import note_made

    note_made(final)
    logger.info(
        "Voice export %s: %d recordings, %.1f GB → %s", run["person"], len(rows), size / 1e9, final
    )
    return {
        "files": [final.name],
        "clips": sum(r["clips"] for r in rows),
        "recordings": len(rows),
        "seconds": round(sum(r["seconds"] for r in rows), 1),
        "bytes": _size(final),
        "left": left.most_common(),
        "folder": str(target_dir),
        "dest": str(target_dir) if eo.get("dest") else "",
        "unpacked": unpacked,
    }


class _Encoder:
    """ffmpeg reading PCM from a pipe and writing the one file."""

    def __init__(self, target: Path, rate: int, codec: list[str]):
        import subprocess
        import tempfile

        # Errors to a file, not a pipe nobody reads: a full pipe would stop
        # ffmpeg halfway.
        self._err = tempfile.TemporaryFile()  # noqa: SIM115 — closed in close()/abort()
        self._p = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                str(rate),
                "-ac",
                "1",
                "-i",
                "-",
                *codec,
                str(target),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._err,
        )

    def write(self, x: np.ndarray) -> None:
        self._p.stdin.write(x.tobytes())

    def close(self) -> None:
        self._p.stdin.close()
        rc = self._p.wait(timeout=600)
        self._err.seek(0)
        err = self._err.read().decode(errors="replace").strip()
        self._err.close()
        if rc != 0:
            raise RuntimeError(err[-300:] or f"ffmpeg rc={rc}")

    def abort(self) -> None:
        try:
            self._p.kill()
            self._p.wait(timeout=10)
        except Exception:
            pass
        self._err.close()


def _cut_words(c: dict) -> list[dict] | None:
    """The clip's words with times from the start of the cut — for a
    trainer that wants to know where each word is. Words outside the cut
    are left out."""
    words = clip_words(c)
    if not words:
        return None
    a, b = c["start"], c["end"]
    out = []
    for w in words:
        if w["a"] is not None and (w["a"] < a - 0.02 or w["a"] > b):
            continue
        out.append(
            {
                "word": w["w"],
                "start": None if w["a"] is None else round(max(0.0, w["a"] - a), 3),
                "end": None if w["b"] is None else round(min(b, w["b"]) - a, 3),
                "score": w["p"],
            }
        )
    return out or None


def _clean_text(t: str) -> str:
    return " ".join(str(t or "").replace("|", " ").split())


def _metadata_csv(written: list[dict]) -> str:
    """LJSpeech layout — ``id|text|text``, what Coqui, Piper and most
    training scripts read."""
    return "".join(
        f"{w['id']}|{_clean_text(w['text'])}|{_clean_text(w['text'])}\n" for w in written
    )


def _metadata_jsonl(written: list[dict], person: str) -> str:
    keys = (
        "file",
        "at",
        "duration",
        "text",
        "display",
        "start",
        "end",
        "label",
        "date",
        "sim",
        "named",
        "clean_by",
        "certainty",
        "confidence",
        "edited",
        "language",
        "words",
    )
    lines = []
    for w in written:
        d = {"id": w["id"], "person": person}
        for k in keys:
            if k in w:
                d["source" if k == "display" else k] = w[k]
        lines.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _srt(written: list[dict]) -> str:
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for n, w in enumerate(written, 1):
        out.append(
            f"{n}\n{ts(w['at'])} --> {ts(w['at'] + w['duration'])}\n{_clean_text(w['text'])}\n"
        )
    return "\n".join(out)


def _amount(n: int, seconds: float) -> str:
    """ "1 clip, 2.1 seconds", "500 clips, 30.0 minutes", "400 clips, 12.00 hours"."""
    if seconds < 60:
        span = f"{seconds:.1f} seconds"
    elif seconds < 3600:
        span = f"{seconds / 60:.1f} minutes"
    else:
        span = f"{seconds / 3600:.2f} hours"
    return f"{n} clip{'' if n == 1 else 's'}, {span}"


def _readme(run: dict, eo: dict, written: list[dict], left) -> str:
    o = run["options"]
    seconds = sum(w["duration"] for w in written)
    edited = sum(1 for w in written if w.get("edited"))
    lines = [
        f"Voice of {run['person']} — exported by Whisper Transcription Manager on "
        f"{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Use a person's voice only with their consent: a dataset like this can",
        "train a model that speaks like them.",
        "",
        f"{_amount(len(written), seconds)}, "
        f"{eo['sample_rate']} Hz mono 16-bit WAV"
        + (", loudness evened out" if eo["normalize"] else ""),
        "",
        "wavs/<id>.wav    the clips",
        "metadata.csv     id|text|text (LJSpeech layout)",
        "metadata.jsonl   one line per clip: text, source recording, start and",
        "                 end in it, diarization label, similarity to the",
        "                 person, named or recognized, how it was checked",
        "",
        "Each clip is one sentence of the transcript (or neighbouring sentences",
        "joined), spoken by this voice. clean_by says how 'only this voice' was",
        "checked: 'spans' = inside the speech spans where diarization heard",
        "only this speaker; 'transcript' = the recording has no stored spans,",
        "the sentence is not marked as overlapping speech.",
        "",
        f"{edited} clip{'' if edited == 1 else 's'} trimmed or corrected by hand (edited: true)."
        + (
            " The others were cut at the nearest pause around their sentence"
            " — or at the sentence's own bounds where there was none."
            if eo.get("snap")
            else ""
        ),
        "Only the clips marked to keep." if eo.get("only_kept") else "",
        "certainty: how sure Whisper was of the sentence, in percent.",
        "",
        "Voices: "
        + (
            "named ones only"
            if o["include"] == "named"
            else f"named ones and recognized ones (similarity >= {o['threshold']})"
        ),
        f"Sentence length {o['min_sec']}-{o['max_sec']} s, speaking rate "
        f"{o['rate_min']}-{o['rate_max']} characters/s, Whisper certain to at "
        f"least {o['min_certainty']} %",
    ]
    if o["date_from"] or o["date_to"]:
        lines.append(
            f"Dates {o['date_from'] or '…'} to {o['date_to'] or '…'} (from the file names)"
        )
    for k, label in (
        ("folder", "Folder"),
        ("file_text", "File name contains"),
        ("text", "Text contains"),
        ("text_not", "Text without"),
        ("language", "Language"),
    ):
        if o[k]:
            lines.append(f"{label}: {o[k]}")
    if left:
        lines += ["", "Left out while cutting:"]
        lines += [f"  {n} {why}" for why, n in left.most_common()]
    return "\n".join(lines) + "\n"


def list_exports() -> list[dict]:
    d = exports_dir()
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        if not p.is_file() or ".part" in p.name:
            continue
        if p.suffix.lower() not in (".zip", ".flac", ".wav", ".mp3", ".opus", ".srt", ".jsonl"):
            continue
        st = p.stat()
        out.append({"name": p.name, "bytes": st.st_size, "modified": st.st_mtime})
    out.sort(key=lambda e: -e["modified"])
    return out


def delete_export(name: str) -> bool:
    from whisper_tm.core.security import safe_path_or_none

    p = safe_path_or_none(exports_dir(), name)
    if p is None or not p.is_file() or p.parent != exports_dir().resolve():
        return False
    p.unlink()
    return True

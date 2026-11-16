"""Finding conversations — by who takes part and how they talk.

store.py keeps every transcript's numbers; this module answers questions
with them: "an hour with Alice, Bob and Carol in long sentences", "how does
Carol talk, and with whom", "who talked when in this recording".

**Baselines.** "Longer sentences than usual" means longer than the library's
own conversations in the same language — languages are spoken in sentences
of different lengths, and a library of lectures talks differently from one
of phone calls. A baseline holds, per language, the deciles of words per
sentence, speaker changes per minute and seconds per turn over the
recordings that are real conversations: two active speakers or more, no
voice above :data:`MONOLOGUE_SHARE`, :data:`BASE_SENTENCES` sentences and
:data:`BASE_SPEECH_SEC` of attributed speech, punctuated. A language with
fewer than :data:`BASE_MIN` of them borrows the deciles of all languages
together (``source: "library"``), a library with fewer generic figures of
spoken conversation (:data:`DEFAULTS`, ``source: "defaults"``).

With people asked for, "longer sentences" means longer than usual for
*them*: the people talk in their own way (a quick-witted round of friends
in sentences of five words, a lecture in fifteen), and measured against the
library a round that always talks quickly would never have a conversation in
longer sentences — the hour where they talked things through at length
would not be found. Their own conversations together (or, with ``match:
any``, each in which one of them takes part) are the measure then
(``sentences_baseline.source: "people"``), with :data:`OWN_MIN` of them at
least; the kinds stay measured against the library — a kind says what a
conversation is, whoever asks.

**Kinds.** One per conversation, the first rule that fits: no speech;
speakers unknown (not diarized); monologue; interview; detailed (long
sentences and long turns); back-and-forth (many speaker changes, no long
sentences); conversation. ``group`` (three active speakers or more) is a
tag beside the kind. A query over whole recordings filters by kind in SQL
(:data:`_KIND_SQL` beside the per-language thresholds in a temporary
table), so a library of 200,000 transcripts is never read into Python for
it; :func:`kind_of` is the same rule for the stretches, which only exist in
Python.

**People.** Names are looked up when a query runs (rag/speakers.py: names
typed in the transcript first, then the voice map's majority), by absolute
path only — typed names are also kept under the display path, which a file
under another root may share. One person can hold several labels in one
recording; their talk is added up. A person takes part in a whole recording
with ``min_share`` of its attributed speech (conversations.min_share) and
:data:`PART_SEC`.

In a long recording people come and go — in a stream of six hours Alice,
Bob and Carol may talk together for one hour of it — so a query for people
together looks for **stretches**: the recording's windows (metrics.py) in
which each of them is active, a single window between two such windows
counted in (a pause, someone fetching coffee), joined into runs of at least
two windows. Only the recordings in which each of them talks somewhere are
read that far: their windows are the one large thing a query loads. Two
errors are accepted there: windows count changes per label, so a change
between two labels of one person counts as a speaker change, and a stretch
counts active speakers by label, so such a person can count twice.

**Copies.** One recording kept as .mp4 and .mkv, or once more in a backup
folder, has a transcript per copy. Results fold them: the same length
(core/copies), the same active speakers, words, sentences and speech within
:data:`COPY_SHARE`, and — copies share their timeline — the speech starting
and ending, and the longest turn starting and ending, within
:data:`COPY_EDGE_SEC` of each other. Counts tell recordings apart only when
there is something to count: length, speakers and words alone fold
thousands of different voice messages of one length into one, so a
conversation of fewer than :data:`COPY_MIN_WORDS` words is never folded (a
copy shown twice is the harmless mistake, a conversation hidden behind
another one is not). Nor is one without a longest turn (not diarized): the
parts of a stream cut to one length, talked through from start to end, say
about as many words in as much speech — only a moment of their own, the
longest turn, tells them apart. A row joins the first row of a group it is
a copy of, never a chain of rows each a little apart. The copy kept is the
one with more named people, then the shorter path, and it says how many
there were (``copies``). The sound's "same" relation and the words around a
moment (core/copies) are not asked: the one is a database of its own, the
other reads segments per pair — not for a list of every copy.

**Dates** come from the file name where it has one (``date_source:
"name"``) — the file time is often a copy's date — else from the file
(``"file"``).
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from array import array
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import astuple, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from whisper_tm.conversations import metrics, store
from whisper_tm.core import copies as copy_rules
from whisper_tm.core import paths

logger = logging.getLogger(__name__)

# ── Rules ──────────────────────────────────────────────────────────────────

#: One voice with this share of the attributed speech holds a monologue
MONOLOGUE_SHARE = 0.85
#: A group: this many active speakers or more
GROUP_SPEAKERS = 3
#: A baseline counts the conversations with this many sentences …
BASE_SENTENCES = 20
#: … and this much attributed speech (and two active speakers, nobody above
#: MONOLOGUE_SHARE, punctuation — sentence lengths need sentence ends)
BASE_SPEECH_SEC = 120.0
#: A language with fewer such conversations borrows the library's deciles,
#: a library with fewer the generic ones
BASE_MIN = 30
#: Baselines are worked out again after this long at the latest
BASE_MAX_AGE_SEC = 600.0
#: "Longer sentences than usual for them" needs this many of the people's
#: own conversations, else the library is the measure
OWN_MIN = 10
#: The percentiles a baseline holds
PCTS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
#: Generic figures of spoken conversation, for a library with too few
#: conversations of its own; filled linearly around p40…p60
DEFAULTS = {
    "wps": (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0),
    "turn_rate": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    "mean_turn_sec": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0),
}

#: The kinds, and what the page calls them
KINDS = ("detailed", "lively", "interview", "monologue", "conversation", "unknown", "silent")
KIND_NAMES = {
    "detailed": "Detailed",
    "lively": "Back-and-forth",
    "interview": "Interview",
    "monologue": "Monologue",
    "conversation": "Conversation",
    "unknown": "Speakers unknown",
    "silent": "No speech",
}

#: Taking part in a whole recording: min_share of its attributed speech and
#: at least this long
PART_SEC = 5.0
#: A long recording is looked into for stretches when each person talks this
#: long in it (or min_share of it): someone quiet for most of ten hours can
#: be very present in one of them
CANDIDATE_SEC = 60.0
#: Active in a window: this long, and max(min_share, WINDOW_SHARE) of all the
#: speech in it — a remark in someone else's monologue is no part in it
WINDOW_TALK_SEC = 15.0
WINDOW_SHARE = 0.05
#: A stretch over this share of the recording's windows with speech is the
#: whole recording
WHOLE_SHARE = 0.9
#: Stretches of one recording at most (the longest)
STRETCHES_MAX = 5
#: A result lists the named people with at least this share of it
PEOPLE_SHARE = 0.01
#: Copies say the same: words, sentences and speech within this share …
COPY_SHARE = 0.02
#: … and their speech starts and ends at the same moments (seconds)
COPY_EDGE_SEC = 2.0
#: Conversations of fewer words are never folded: among thousands of short
#: voice messages, a few of the same length say as many words by chance
COPY_MIN_WORDS = 300
#: "Who talked when" in at most this many parts
PARTS_MAX = 40
#: Result lists are kept this long, this many at most — paging through one
#: must not count it again
RESULTS_TTL_SEC = 60.0
RESULTS_MAX = 32
#: The library's counts are kept this long (the tab asks every two seconds
#: while the background run counts)
COUNTS_TTL_SEC = 10.0
#: Paging
LIMIT_DEFAULT = 50
LIMIT_MAX = 200
#: People asked for at once at most
PEOPLE_MAX = 12
#: A name is at most this long (rag/speakers.py cuts it there), a value of
#: the ``people`` parameter — maybe a comma list of names — at most that
NAME_CHARS = 60
LIST_CHARS = 4096
#: A whole-number filter (speakers) above this is this
WHOLE_MAX = 1_000_000
SORTS = (
    "longest",
    "shortest",
    "sentences",
    "rate",
    "questions",
    "monologue",
    "newest",
    "oldest",
    "together",
)
_DESCENDING = frozenset(
    {"longest", "sentences", "rate", "questions", "monologue", "newest", "together"}
)

# The whole-recording kind in SQL: the rules of kind_of, the thresholds from
# temp.cv_base (bl: the recording's language, bf: the fallback row '*').
# A NULL measure fails its comparison and falls through, as None does there.
_KIND_SQL = f"""CASE
    WHEN s.words = 0 THEN 'silent'
    WHEN s.labels = 0 THEN 'unknown'
    WHEN s.speakers <= 1 OR s.top_share >= {MONOLOGUE_SHARE!r} THEN 'monologue'
    WHEN s.interview = 1 THEN 'interview'
    WHEN s.wps >= COALESCE(bl.wps60, bf.wps60)
         AND s.mean_turn_sec >= COALESCE(bl.turn50, bf.turn50) THEN 'detailed'
    WHEN s.turn_rate >= COALESCE(bl.rate60, bf.rate60)
         AND (s.wps IS NULL OR s.wps <= COALESCE(bl.wps50, bf.wps50)) THEN 'lively'
    ELSE 'conversation' END"""

_DAY_SQL = "COALESCE(s.name_date, NULLIF(m.modified_at, 0))"
# The value each sort orders whole recordings by (NULL: last)
_SORT_SQL = {
    "longest": "s.duration_sec",
    "shortest": "s.duration_sec",
    "sentences": "s.wps",
    "rate": "s.turn_rate",
    "questions": "CASE WHEN s.sentences > 0 THEN CAST(s.questions AS REAL) / s.sentences END",
    "monologue": "s.longest_turn_sec",
    "newest": _DAY_SQL,
    "oldest": _DAY_SQL,
    "together": "NULL",  # worked out from the people, see _any_rows
}

# The stored root (transcripts.media_dir) only where the display path does
# not lead to the file: database._root_of needs it for nothing else, and the
# column lies behind json_data — the whole transcript, megabytes for a long
# one. Reading it walks that transcript's overflow pages, for every row of a
# person's recordings; SQLite reads the column of a CASE branch only when
# the branch is taken.
_MEDIA_DIR_SQL = """CASE WHEN t.rel_path != '' AND substr(t.rel_path, 1, 1) != '/'
             AND substr(t.rel_path, -1) != '/'
             AND substr(t.abs_path, -length(t.rel_path) - 1) = '/' || t.rel_path
        THEN NULL ELSE t.media_dir END"""

_REC_SQL = f"""
    SELECT t.id AS tid, t.abs_path, t.rel_path, {_MEDIA_DIR_SQL} AS media_dir,
           lower(COALESCE(t.language, '')) AS lang,
           s.name_date, NULLIF(m.modified_at, 0) AS modified_at,
           s.start_sec, s.end_sec, s.duration_sec, s.speech_sec, s.attributed_sec,
           s.unattributed_sec, s.words, s.sentences, s.questions, s.turns, s.short_turns,
           s.labels, s.speakers, s.top_share, s.longest_turn_sec, s.longest_turn_label,
           s.longest_turn_start, s.punctuated, s.interview, s.wps, s.turn_rate,
           s.mean_turn_sec, s.version, s.seg_count, s.char_count,
           t.segment_count AS t_seg_count, t.char_count AS t_char_count,
           EXISTS (SELECT 1 FROM conv_windows w WHERE w.transcript_id = t.id) AS has_windows
"""


# ── Names ──────────────────────────────────────────────────────────────────


def clean_name(name, limit: int = NAME_CHARS) -> str:
    """A name as the names index keeps it (rag/speakers.py): one line, no
    brackets, at most :data:`NAME_CHARS` characters (``limit``: a value that
    may list several, not cut to one name's length)."""
    return " ".join(re.sub(r"[\[\]]", " ", str(name or "")).split())[:limit]


def fold(name) -> str:
    """The identity of a name: ASCII letters folded, as the voice library's
    ``COLLATE NOCASE`` compares them (database/speaker.fold_name)."""
    from whisper_tm.database.speaker import fold_name

    return fold_name(clean_name(name)) or ""


def _index() -> dict:
    from whisper_tm.rag import speakers

    return speakers.index()


class _Names:
    """The names index as this feature reads it: by absolute path only."""

    def __init__(self, index: dict):
        self.index = index
        self._names = index.get("names") or {}
        self._people = index.get("people") or {}
        self._shown = index.get("shown") or {}
        self._labels: dict[str, dict[str, str]] = {}

    def labels(self, abs_path: str) -> dict[str, str]:
        """``{label: folded name}`` of a recording's named labels."""
        got = self._labels.get(abs_path)
        if got is None:
            got = {}
            for label, name in (self._names.get(abs_path) or {}).items():
                f = fold(name)
                if f:
                    got[label] = f
            self._labels[abs_path] = got
        return got

    def count(self, abs_path: str) -> int:
        """How many people are named in a recording."""
        return len(set(self.labels(abs_path).values()))

    def known(self, folded: str) -> bool:
        return bool(folded) and folded in self._people

    def shown(self, folded: str) -> str:
        return self._shown.get(folded) or folded

    def recordings(self, folded: str) -> set[str]:
        return {k for k in self._people.get(folded) or {} if paths.is_absolute(k)}

    def everyone(self) -> list[str]:
        """Everyone named in at least one recording (by absolute path)."""
        return [f for f, keys in self._people.items() if any(paths.is_absolute(k) for k in keys)]


def resolve_people(names: _Names, items) -> tuple[list[str], list[str]]:
    """``(folded names of the people known, the names asked for that are
    not)``. A value with commas is one person when one has that name, else a
    list of them (``people=Alice,Bob``) — so the known can be more than
    :data:`PEOPLE_MAX` (:func:`find` cuts them and says so)."""
    known: list[str] = []
    unknown: list[str] = []
    for item in items:
        whole = clean_name(item)
        if not whole:
            continue
        pieces = [whole]
        # Split the value as it came: cut to one name's length first, the
        # names after the 60th character would be lost or cut into another
        if "," in str(item) and not names.known(fold(whole)):
            pieces = [p for p in (clean_name(x) for x in str(item).split(",")) if p]
        for piece in pieces:
            f = fold(piece)
            if not names.known(f):
                if piece not in unknown:
                    unknown.append(piece)
            elif f not in known:
                known.append(f)
    return known, unknown


# ── Baselines ──────────────────────────────────────────────────────────────

_BASE_SQL = f"""
    SELECT lower(COALESCE(t.language, '')), s.wps, s.turn_rate, s.mean_turn_sec
    FROM conv_stats s JOIN transcripts t ON t.id = s.transcript_id
    WHERE t.complete = 1 AND t.abs_path != ''
      AND s.speakers >= 2 AND s.top_share < {MONOLOGUE_SHARE!r}
      AND s.sentences >= {BASE_SENTENCES} AND s.attributed_sec >= {BASE_SPEECH_SEC!r}
      AND s.punctuated = 1
      AND s.wps IS NOT NULL AND s.turn_rate IS NOT NULL AND s.mean_turn_sec IS NOT NULL
"""

_base_lock = threading.Lock()
_base_held: dict = {"key": None, "at": -1e18, "value": None}


def deciles(values) -> tuple[float, ...]:
    """p10 … p90 of ``values``, interpolated linearly between ranks."""
    v = sorted(float(x) for x in values)
    n = len(v)
    out = []
    for p in PCTS:
        pos = (n - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        out.append(v[lo] + (v[hi] - v[lo]) * (pos - lo))
    return tuple(out)


def _baseline(language, source: str, rows) -> dict:
    return {
        "language": language,
        "source": source,
        "conversations": len(rows),
        "wps": deciles(r[0] for r in rows),
        "turn_rate": deciles(r[1] for r in rows),
        "mean_turn_sec": deciles(r[2] for r in rows),
    }


def baselines(conn, db_path: str) -> dict:
    """``{"languages": {code: baseline}, "library": baseline}`` — held on the
    statistics' generation, :data:`BASE_MAX_AGE_SEC` at most."""
    key = (db_path, store.generation())
    now = time.monotonic()
    with _base_lock:
        if _base_held["key"] == key and now - _base_held["at"] < BASE_MAX_AGE_SEC:
            return _base_held["value"]
    by_lang: dict[str, list] = {}
    for lang, wps, rate, turn in conn.execute(_BASE_SQL):
        by_lang.setdefault(lang or "", []).append((wps, rate, turn))
    pooled = [r for rows in by_lang.values() for r in rows]
    if len(pooled) >= BASE_MIN:
        library = _baseline(None, "library", pooled)
    else:
        library = {"language": None, "source": "defaults", "conversations": 0, **DEFAULTS}
    # '*' is the fallback row of temp.cv_base, never a language of its own
    value = {
        "languages": {
            lang: _baseline(lang, "language", rows)
            for lang, rows in by_lang.items()
            if len(rows) >= BASE_MIN and lang != "*"
        },
        "library": library,
    }
    with _base_lock:
        _base_held.update(key=key, at=now, value=value)
    return value


def baseline_for(bases: dict, language) -> dict:
    """The baseline a recording in ``language`` is measured against."""
    own = bases["languages"].get(language or "")
    if own is not None:
        return own
    return {**bases["library"], "language": language or None}


def public_baseline(b: dict) -> dict:
    def pcts(values, digits):
        return {f"p{p}": round(v, digits) for p, v in zip(PCTS, values)}

    return {
        "language": b["language"],
        "source": b["source"],
        "conversations": b["conversations"],
        "wps": pcts(b["wps"], 1),
        "turn_rate": pcts(b["turn_rate"], 2),
        "mean_turn_sec": pcts(b["mean_turn_sec"], 1),
    }


def thresholds(b: dict) -> dict:
    """The deciles the kinds and the sentence filter compare with."""
    return {
        "wps60": b["wps"][5],
        "wps50": b["wps"][4],
        "wps40": b["wps"][3],
        "turn50": b["mean_turn_sec"][4],
        "rate60": b["turn_rate"][5],
    }


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def percentile(value, quantiles) -> int | None:
    """Where ``value`` lies among a baseline's deciles, 0…100: interpolated
    linearly between them, beyond them with the slope of the nearest rising
    step; a value equal to several deciles lies in their middle."""
    if value is None:
        return None
    v = float(value)
    pts = list(zip(PCTS, quantiles))
    same = [p for p, q in pts if q == v]
    if same:
        return _half_up((same[0] + same[-1]) / 2)
    for (p0, q0), (p1, q1) in zip(pts, pts[1:]):
        if q0 < v < q1:
            return _half_up(p0 + (p1 - p0) * (v - q0) / (q1 - q0))
    rising = [(p1 - p0) / (q1 - q0) for (p0, q0), (p1, q1) in zip(pts, pts[1:]) if q1 > q0]
    if v < pts[0][1]:
        pct = pts[0][0] - (pts[0][1] - v) * rising[0] if rising else 0.0
    else:
        pct = pts[-1][0] + (v - pts[-1][1]) * rising[-1] if rising else 100.0
    return _half_up(min(100.0, max(0.0, pct)))


# ── Kinds and reasons ──────────────────────────────────────────────────────


def kind_of(m: dict, th: dict) -> str:
    """The kind of a conversation — the rules of :data:`_KIND_SQL`."""
    if not m.get("words"):
        return "silent"
    if not m.get("labels"):
        return "unknown"
    if (m.get("speakers") or 0) <= 1 or (m.get("top_share") or 0.0) >= MONOLOGUE_SHARE:
        return "monologue"
    if m.get("interview"):
        return "interview"
    wps, turn, rate = m.get("wps"), m.get("mean_turn_sec"), m.get("turn_rate")
    if wps is not None and turn is not None and wps >= th["wps60"] and turn >= th["turn50"]:
        return "detailed"
    if rate is not None and rate >= th["rate60"] and (wps is None or wps <= th["wps50"]):
        return "lively"
    return "conversation"


_AMONG = {
    "language": "of conversations in this language",
    "library": "of the library's conversations",
    "defaults": "of typical conversations",
}


def _than(pct: int, more: str, less: str) -> str:
    return f"{more} than {pct} %" if pct >= 50 else f"{less} than {100 - pct} %"


def clock(sec) -> str:
    """Seconds as ``m:ss`` or ``h:mm:ss``."""
    s = int(round(float(sec or 0)))
    h, rest = divmod(s, 3600)
    mi, se = divmod(rest, 60)
    return f"{h}:{mi:02d}:{se:02d}" if h else f"{mi}:{se:02d}"


def reasons(kind: str, m: dict, base: dict, asker: tuple | None = None) -> list[str]:
    """One to three sentences with the numbers that decided the kind.
    ``asker``: ``(name or None, % of their sentences that are questions,
    % of the talk)`` of an interview's asking voice."""
    among = _AMONG.get(base.get("source"), _AMONG["defaults"])
    wps, rate, turn = m.get("wps"), m.get("turn_rate"), m.get("mean_turn_sec")

    def sentences():
        p = percentile(wps, base["wps"])
        return f"Sentences of {wps:.1f} words — {_than(p, 'longer', 'shorter')} {among}."

    def turns():
        p = percentile(turn, base["mean_turn_sec"])
        return f"Turns of {turn:.0f} s on average — {_than(p, 'longer', 'shorter')}."

    def changes():
        p = percentile(rate, base["turn_rate"])
        return f"{rate:.1f} speaker changes a minute — {_than(p, 'more', 'fewer')} {among}."

    if kind == "silent":
        return ["Nobody speaks in it."]
    if kind == "unknown":
        return ["Its speakers were not told apart: who spoke when is not known."]
    if kind == "monologue":
        share = m.get("top_share") or 0.0
        out = (
            [f"One voice speaks {_half_up(share * 100)} % of the time."]
            if share >= MONOLOGUE_SHARE
            else ["Only one voice speaks for more than a moment."]
        )
        if (m.get("longest_turn_sec") or 0) >= 60:
            out.append(f"Its longest turn lasts {clock(m['longest_turn_sec'])}.")
        return out
    if kind == "interview":
        if not asker:
            return ["Two voices, one of them asking most of the questions."]
        name, asks, share = asker
        who = f"{name}'s" if name else "the asker's"
        return [
            f"Two voices, one asking: {asks} % of {who} sentences are questions.",
            f"The asker speaks {share} % of the time.",
        ]
    if kind == "detailed":
        return [sentences(), turns()]
    if kind == "lively":
        return [changes()] + ([sentences()] if wps is not None else [])
    out = ([sentences()] if wps is not None else []) + ([changes()] if rate is not None else [])
    return out or ["Too little speech to say how they talk."]


def _asker(m: dict, named: dict, names: _Names) -> tuple | None:
    active = m.get("active") or []
    if len(active) != 2:
        return None
    per = m["per_label"]
    label = min(active, key=lambda x: per[x]["talk_sec"])
    d = per[label]
    if not d["sentences"] or not m["attributed_sec"]:
        return None
    name = names.shown(named[label]) if label in named else None
    return (
        name,
        _half_up(100 * d["questions"] / d["sentences"]),
        _half_up(100 * d["talk_sec"] / m["attributed_sec"]),
    )


# ── Paths and dates ────────────────────────────────────────────────────────


def place(abs_path: str, rel_path: str, media_dir: str) -> tuple[str, str]:
    """``(file_dir, filename)`` — what a search hit hands on to open and play
    a recording: the root its display path is relative to (database._root_of)
    and the path below it. Where the stored root and display path don't lead
    to the file (an older root's display path), its folder and name; the two
    always join to ``abs_path``."""
    from whisper_tm.database import _root_of

    root = _root_of(rel_path or "", abs_path, media_dir or "")
    if root:
        rel = paths.rel_from(abs_path, root)
        if rel:
            return root, rel
    folder, _, name = (abs_path or "").rpartition("/")
    return folder or "/", name


def iso_day(ts) -> str | None:
    """A time as the local day ``YYYY-MM-DD``."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None


def _day_start(day: str, after: bool = False) -> float | None:
    """Local midnight starting ``day`` (``after``: the day after it)."""
    try:
        d = datetime.strptime(day, "%Y-%m-%d")
        return (d + timedelta(days=1) if after else d).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _when(rec) -> tuple[float | None, str | None]:
    """A recording's time and where it comes from: the file name's date
    first, then the file."""
    if rec["name_date"] is not None:
        return rec["name_date"], "name"
    if rec["modified_at"] is not None:
        return rec["modified_at"], "file"
    return None, None


# ── Filters ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Filters:
    """What a query asks for — :func:`parse_filters` reads it leniently."""

    people: tuple[str, ...] = ()
    match: str = "all"
    min_share: float | None = None
    min_sec: float | None = None
    max_sec: float | None = None
    kinds: tuple[str, ...] = ()
    group: bool = False
    sentences: str = "any"
    wps_min: float | None = None
    wps_max: float | None = None
    rate_min: float | None = None
    rate_max: float | None = None
    speakers_min: int | None = None
    speakers_max: int | None = None
    questions_min: float | None = None
    language: str = ""
    q: str = ""
    folder: str = ""
    date_from: str = ""
    date_to: str = ""
    sort: str = "longest"
    copies: str = "fold"


_LANG = re.compile(r"[a-z0-9_-]{1,16}")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_filters(raw: dict, media_dir: str | None = None) -> tuple[Filters, list[str]]:
    """``(Filters, ignored)`` from request values (strings, or lists of them).

    Lenient like the searches: a value that can't be read turns off only its
    own filter and its name goes into ``ignored``. 0 for a maximum means no
    maximum (the page sends nothing else for an empty field)."""
    ignored: list[str] = []

    def values(name) -> list[str]:
        v = raw.get(name)
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v if x is not None]
        return [str(v)]

    def text(name, n=200, collapse=True) -> str:
        v = ",".join(values(name))
        return (" ".join(v.split()) if collapse else v.strip())[:n]

    def number(name, *, hi=None, whole=False, zero_off=False):
        s = text(name, 40)
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            ignored.append(name)
            return None
        if not math.isfinite(v) or v < 0 or (hi is not None and v > hi) or (whole and v != int(v)):
            ignored.append(name)
            return None
        if zero_off and v == 0:
            return None
        # SQLite binds integers of 64 bits — "1e20 speakers" would not bind
        # (a 500), and a million says the same
        return int(min(v, WHOLE_MAX)) if whole else v

    from whisper_tm.database.speaker import fold_name

    people: list[str] = []
    seen: set[str] = set()
    for v in values("people"):
        # Whole: one value may list several people (resolve_people splits it)
        name = clean_name(v, LIST_CHARS)
        if name and fold_name(name) not in seen:
            seen.add(fold_name(name))
            people.append(name)
    if len(people) > PEOPLE_MAX:
        people = people[:PEOPLE_MAX]
        ignored.append("people")

    match = text("match", 10).lower() or "all"
    if match not in ("all", "any"):
        ignored.append("match")
        match = "all"

    kinds: list[str] = []
    group = False
    bad_kind = False
    for k in (x.strip() for x in text("kind", 200).lower().split(",")):
        if not k or k in ("any", "all"):
            continue
        if k == "group":
            group = True
        elif k in KINDS:
            if k not in kinds:
                kinds.append(k)
        else:
            bad_kind = True
    if bad_kind:
        ignored.append("kind")

    sentences = text("sentences", 10).lower() or "any"
    if sentences not in ("longer", "shorter", "any"):
        ignored.append("sentences")
        sentences = "any"

    language = text("language", 40).lower()
    if language and not _LANG.fullmatch(language):
        ignored.append("language")
        language = ""

    folder = ""
    raw_folder = text("folder", 4096, collapse=False)
    if raw_folder:
        if paths.is_absolute(raw_folder):
            folder = paths.canon(raw_folder)
        elif media_dir:
            folder = paths.abs_from(raw_folder, media_dir)
        if not folder:
            ignored.append("folder")

    days = {}
    for name in ("date_from", "date_to"):
        d = text(name, 20)
        if d and not (_DAY.fullmatch(d) and _day_start(d) is not None):
            ignored.append(name)
            d = ""
        days[name] = d

    sort = text("sort", 20).lower() or "longest"
    if sort not in SORTS or (sort == "together" and not people):
        ignored.append("sort")
        sort = "longest"

    copies = text("copies", 10).lower() or "fold"
    if copies not in ("fold", "all"):
        ignored.append("copies")
        copies = "fold"

    return Filters(
        people=tuple(people),
        match=match,
        min_share=number("min_share", hi=0.5),
        min_sec=number("min_sec", zero_off=True),
        max_sec=number("max_sec", zero_off=True),
        kinds=tuple(kinds),
        group=group,
        sentences=sentences,
        wps_min=number("wps_min", zero_off=True),
        wps_max=number("wps_max", zero_off=True),
        rate_min=number("rate_min", zero_off=True),
        rate_max=number("rate_max", zero_off=True),
        speakers_min=number("speakers_min", whole=True, zero_off=True),
        speakers_max=number("speakers_max", whole=True, zero_off=True),
        questions_min=number("questions_min", hi=1.0, zero_off=True),
        language=language,
        q=text("q", 200, collapse=False),
        folder=folder,
        date_from=days["date_from"],
        date_to=days["date_to"],
        sort=sort,
        copies=copies,
    ), ignored


def parse_page(offset, limit) -> tuple[int, int, list[str]]:
    """``(offset, limit, ignored)``: limit 1 … :data:`LIMIT_MAX`."""
    ignored: list[str] = []

    def whole(name, raw, default):
        s = str(raw if raw is not None else "").strip()
        if not s:
            return default
        try:
            return int(float(s))
        except (ValueError, OverflowError):
            ignored.append(name)
            return default

    off = min(max(whole("offset", offset, 0), 0), 10**9)
    lim = min(max(whole("limit", limit, LIMIT_DEFAULT), 1), LIMIT_MAX)
    return off, lim, ignored


def _min_share(f: Filters) -> float:
    if f.min_share is not None:
        return f.min_share
    from whisper_tm.conversations import runner

    return runner.settings()["min_share"]


# ── Reading ────────────────────────────────────────────────────────────────


@contextmanager
def _reading(db):
    """A read-only connection of its own: a query may read for a second, and
    the writers (three queue lanes, edits) must not wait for it."""
    from whisper_tm.database import _folded, _name_folded

    p = Path(db._db_path)
    uri = (p.as_uri() if p.is_absolute() else f"file:{p}") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=3000")
        conn.execute("PRAGMA temp_store=MEMORY")
        # The name filter compares non-ASCII letters case-folded (wx_folded)
        conn.create_function("wx_folded", 1, _folded, deterministic=True)
        conn.create_function("wx_name_folded", 1, _name_folded, deterministic=True)
        yield conn
    finally:
        conn.close()


def _temp_paths(conn, keys) -> None:
    """The recordings of the people asked for, as ``temp.cv_paths``: too many
    for ``IN (?, ?, …)`` (a person can be in thousands)."""
    conn.execute("DROP TABLE IF EXISTS temp.cv_paths")
    conn.execute("CREATE TEMP TABLE cv_paths (abs_path TEXT PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO temp.cv_paths VALUES (?)", ((k,) for k in keys))


def _temp_ids(conn, ids) -> None:
    conn.execute("DROP TABLE IF EXISTS temp.cv_ids")
    conn.execute("CREATE TEMP TABLE cv_ids (tid INTEGER PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO temp.cv_ids VALUES (?)", ((i,) for i in ids))


def _temp_base(conn, bases: dict) -> None:
    """Each language's thresholds, and the fallback row ``'*'`` for the
    languages that borrow the library's — what :data:`_KIND_SQL` joins."""
    conn.execute("DROP TABLE IF EXISTS temp.cv_base")
    conn.execute(
        "CREATE TEMP TABLE cv_base (lang TEXT PRIMARY KEY, wps60 REAL, wps50 REAL, "
        "wps40 REAL, turn50 REAL, rate60 REAL)"
    )
    rows = [("*", bases["library"])] + list(bases["languages"].items())
    conn.executemany(
        "INSERT INTO temp.cv_base VALUES (?, ?, ?, ?, ?, ?)",
        [
            (lang, t["wps60"], t["wps50"], t["wps40"], t["turn50"], t["rate60"])
            for lang, t in ((lang, thresholds(b)) for lang, b in rows)
        ],
    )


def _from(*, people: bool = False, base: bool = False, media: bool = True) -> str:
    """The tables a query reads: the people's recordings (``temp.cv_paths``)
    or all; ``media`` for the file's date, ``base`` for the thresholds of
    the recording's language (temp.cv_base)."""
    out = (
        "FROM temp.cv_paths c JOIN transcripts t ON t.abs_path = c.abs_path "
        "JOIN conv_stats s ON s.transcript_id = t.id "
        if people
        else "FROM conv_stats s JOIN transcripts t ON t.id = s.transcript_id "
    )
    if media:
        out += "LEFT JOIN media_files m ON m.id = t.media_id "
    if base:
        out += (
            "LEFT JOIN temp.cv_base bl ON bl.lang = lower(COALESCE(t.language, '')) "
            "JOIN temp.cv_base bf ON bf.lang = '*' "
        )
    return out


def _labels_of(conn, tids) -> dict[int, dict[str, dict]]:
    """``{transcript id: {label: conv_speakers row}}``."""
    tids = list(tids)
    if not tids:
        return {}
    _temp_ids(conn, tids)
    out: dict[int, dict[str, dict]] = {}
    for r in conn.execute(
        "SELECT cs.* FROM temp.cv_ids i JOIN conv_speakers cs ON cs.transcript_id = i.tid"
    ):
        out.setdefault(r["transcript_id"], {})[r["label"]] = dict(r)
    return out


class _Windows:
    """A recording's windows (metrics.py) as columns per label — talk,
    words, sentences, questions, changes: one value per window — and the
    speech of everyone (``total``, the unattributed included)."""

    #: More windows than this is a broken row, not a recording (a month of
    #: five-minute windows — a day has 288): read, it would fill columns of
    #: this length for every label
    MAX = 10_000

    def __init__(self, data: dict):
        self.size = float(data.get("w") or metrics.WINDOW_SEC)
        raw = data.get("labels") or {}
        n = 0
        for rows in raw.values():
            for r in rows:
                n = max(n, int(r[0]) + 1)
        if n > self.MAX or self.size <= 0:
            raise ValueError(f"{n} windows of {self.size} s")
        self.n = n
        self.labels: dict[str, list[list[float]]] = {}
        for label, rows in raw.items():
            cols = [[0.0] * n for _ in range(5)]
            for r in rows:
                i = int(r[0])
                if 0 <= i < n:
                    for k in range(5):
                        cols[k][i] += float(r[k + 1] or 0)
            self.labels[str(label)] = cols
        self.total = [sum(c[0][i] for c in self.labels.values()) for i in range(n)]

    def talk(self, labels, i: int) -> float:
        return sum(self.labels[x][0][i] for x in labels if x in self.labels)


def _each_windows(conn, tids):
    """``(transcript id, its _Windows)`` one recording at a time — None for
    windows that can't be read. A person in thousands of long recordings
    would otherwise hold all their windows at once, hundreds of megabytes.
    The caller runs no SQL on ``conn`` before it is through."""
    tids = list(tids)
    if not tids:
        return
    _temp_ids(conn, tids)
    for tid, data in conn.execute(
        "SELECT w.transcript_id, w.data FROM temp.cv_ids i "
        "JOIN conv_windows w ON w.transcript_id = i.tid"
    ):
        try:
            win = _Windows(json.loads(data))
        # (json.loads reads Infinity, which int() refuses with OverflowError)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as e:
            logger.debug("Conversations: windows of transcript %s not readable: %s", tid, e)
            win = None
        yield tid, win


def _windows_of(conn, tids) -> dict[int, _Windows]:
    return {tid: win for tid, win in _each_windows(conn, tids) if win is not None}


# ── One conversation's numbers ─────────────────────────────────────────────


def _per_label(rows: dict) -> dict[str, dict]:
    return {
        label: {
            "talk_sec": float(d.get("talk_sec") or 0.0),
            "words": float(d.get("words") or 0),
            "sentences": float(d.get("sentences") or 0),
            "questions": float(d.get("questions") or 0),
            "turns": float(d.get("turns") or 0),
        }
        for label, d in rows.items()
    }


def _whole(rec, label_rows: dict) -> dict:
    """A whole recording's numbers, as a stretch has them."""
    per = _per_label(label_rows)
    att = float(rec["attributed_sec"] or 0.0)
    # Active as metrics.py counts them (a short voice message has short turns)
    active = [
        x
        for x, d in per.items()
        if att > 0
        and d["talk_sec"] / att >= metrics.ACTIVE_SHARE
        and (
            d["talk_sec"] >= metrics.ACTIVE_MIN_SEC
            or d["talk_sec"] / att >= metrics.ACTIVE_LARGE_SHARE
        )
    ]
    dur = float(rec["duration_sec"] or 0.0)
    return {
        "whole": True,
        "start_sec": 0.0,
        "end_sec": dur,
        "duration_sec": dur,
        "speech_sec": float(rec["speech_sec"] or 0.0),
        "attributed_sec": att,
        "words": rec["words"] or 0,
        "sentences": rec["sentences"] or 0,
        "questions": rec["questions"] or 0,
        "turns": rec["turns"] or 0,
        "labels": rec["labels"] or 0,
        "speakers": rec["speakers"] or 0,
        "top_share": float(rec["top_share"] or 0.0),
        "interview": bool(rec["interview"]),
        "punctuated": bool(rec["punctuated"]),
        "wps": rec["wps"],
        "turn_rate": rec["turn_rate"],
        "mean_turn_sec": rec["mean_turn_sec"],
        "longest_turn_sec": rec["longest_turn_sec"],
        "longest_turn_start": rec["longest_turn_start"],
        "longest_turn_label": rec["longest_turn_label"],
        "per_label": per,
        "active": active,
    }


def _stretch(win: _Windows, a: int, b: int, rec) -> dict:
    """The numbers of windows ``a`` … ``b``, added up (see the module)."""
    per: dict[str, dict] = {}
    words = sentences = questions = 0.0
    for label, cols in win.labels.items():
        talk = sum(cols[0][a : b + 1])
        w = sum(cols[1][a : b + 1])
        se = sum(cols[2][a : b + 1])
        q = sum(cols[3][a : b + 1])
        words += w
        sentences += se
        questions += q
        if label != metrics.UNATTRIBUTED and (talk > 0 or w > 0):
            per[label] = {
                "talk_sec": talk,
                "words": w,
                "sentences": se,
                "questions": q,
                "turns": sum(cols[4][a : b + 1]),
            }
    att = sum(d["talk_sec"] for d in per.values())
    changes = sum(d["turns"] for d in per.values())
    start = max(a * win.size, float(rec["start_sec"] or 0.0))
    end = (b + 1) * win.size
    if rec["end_sec"]:
        end = min(end, float(rec["end_sec"]))
    active = [
        x
        for x, d in per.items()
        if att > 0
        and d["talk_sec"] / att >= metrics.ACTIVE_SHARE
        and d["talk_sec"] >= metrics.ACTIVE_MIN_SEC
    ]
    interview = False
    if len(active) == 2:
        d = per[min(active, key=lambda x: per[x]["talk_sec"])]
        interview = (
            d["sentences"] >= metrics.INTERVIEW_MIN_SENTENCES
            and d["questions"] / d["sentences"] >= metrics.INTERVIEW_QUESTIONS
            and d["talk_sec"] / att <= metrics.INTERVIEW_SHARE
        )
    timed = att >= metrics.RATE_MIN_SEC and changes > 0
    lt_start = rec["longest_turn_start"]
    inside = bool(rec["longest_turn_sec"]) and lt_start is not None and start <= lt_start < end
    return {
        "whole": False,
        "start_sec": start,
        "end_sec": end,
        "duration_sec": max(0.0, end - start),
        "speech_sec": sum(win.total[a : b + 1]),
        "attributed_sec": att,
        "words": int(round(words)),
        "sentences": sentences,
        "questions": questions,
        "turns": changes,
        "labels": len(per),
        "speakers": len(active),
        "top_share": max((d["talk_sec"] for d in per.values()), default=0.0) / att if att else 0.0,
        "interview": interview,
        "punctuated": bool(rec["punctuated"]),
        "wps": words / sentences if rec["punctuated"] and sentences > 0 else None,
        "turn_rate": changes / (att / 60.0) if timed else None,
        "mean_turn_sec": att / changes if timed else None,
        "longest_turn_sec": rec["longest_turn_sec"] if inside else None,
        "longest_turn_start": lt_start if inside else None,
        "longest_turn_label": rec["longest_turn_label"] if inside else None,
        "per_label": per,
        "active": active,
    }


def _runs(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    """Maximal runs of true windows, a single false one between two true
    ones counted in, at least ``min_len`` long: ``[(first, last)]``."""
    n = len(flags)
    filled = [flags[i] or (0 < i < n - 1 and flags[i - 1] and flags[i + 1]) for i in range(n)]
    out, i = [], 0
    while i < n:
        if not filled[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and filled[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            out.append((i, j))
        i = j + 1
    return out


def _person_talk(per: dict, named: dict, people) -> dict[str, float]:
    """Each person's talk in a conversation, their labels added up."""
    talk = dict.fromkeys(people, 0.0)
    for label, d in per.items():
        f = named.get(label)
        if f in talk:
            talk[f] += d["talk_sec"]
    return talk


def _r(v, digits: int):
    return None if v is None else round(float(v), digits)


def _result(
    rec, m: dict, named: dict, names: _Names, base: dict, requested=(), kind: str | None = None
) -> dict:
    """A conversation as the API answers it."""
    att = m["attributed_sec"]
    by: dict[str, list[float]] = {}
    for label, d in m["per_label"].items():
        f = named.get(label)
        if f:
            x = by.setdefault(f, [0.0, 0.0, 0.0])
            x[0] += d["talk_sec"]
            x[1] += d["words"]
            x[2] += d["sentences"]
    people = []
    for f, (talk, words, sents) in by.items():
        share = talk / att if att > 0 else 0.0
        if share >= PEOPLE_SHARE:
            people.append(
                {
                    "name": names.shown(f),
                    "talk_sec": round(talk, 1),
                    "share": round(share, 3),
                    "wps": round(words / sents, 1) if m["punctuated"] and sents > 0 else None,
                }
            )
    people.sort(key=lambda p: (-p["talk_sec"], p["name"].lower()))
    together = None
    if requested:
        together = round(sum(by.get(f, (0.0,))[0] for f in requested) / att, 3) if att else 0.0
    kind = kind or kind_of(m, thresholds(base))
    file_dir, filename = place(rec["abs_path"], rec["rel_path"], rec["media_dir"])
    ts, source = _when(rec)
    day = iso_day(ts)
    lt_label = m.get("longest_turn_label")
    se = m["sentences"] or 0
    return {
        "abs_path": rec["abs_path"],
        "filename": filename,
        "file_dir": file_dir,
        "start_sec": _r(m["start_sec"], 1),
        "end_sec": _r(m["end_sec"], 1),
        "duration_sec": _r(m["duration_sec"], 1),
        "whole": m["whole"],
        "recording_sec": _r(rec["duration_sec"], 1),
        "language": rec["lang"] or None,
        "date": day,
        "date_source": source if day else None,
        "speech_sec": _r(m["speech_sec"], 1),
        "words": int(round(m["words"] or 0)),
        "sentences": int(round(se)),
        "questions": int(round(m["questions"] or 0)),
        "wps": _r(m["wps"], 1),
        "wps_pct": percentile(m["wps"], base["wps"]),
        "turn_rate": _r(m["turn_rate"], 2),
        "rate_pct": percentile(m["turn_rate"], base["turn_rate"]),
        "mean_turn_sec": _r(m["mean_turn_sec"], 1),
        "turn_pct": percentile(m["mean_turn_sec"], base["mean_turn_sec"]),
        # Without punctuation every segment is a sentence: no share to tell
        "questions_share": round(m["questions"] / se, 3) if se > 0 and m["punctuated"] else None,
        "longest_turn_sec": _r(m["longest_turn_sec"], 1) if m["longest_turn_sec"] else None,
        "longest_turn_start": _r(m["longest_turn_start"], 1) if m["longest_turn_sec"] else None,
        "longest_turn_name": names.shown(named[lt_label])
        if m["longest_turn_sec"] and lt_label in named
        else None,
        "speakers": m["speakers"],
        "unnamed": sum(1 for x in m["active"] if x not in named),
        "people": people,
        "kind": kind,
        "group": m["speakers"] >= GROUP_SPEAKERS,
        "interview": bool(m["interview"]),
        "reasons": reasons(kind, m, base, _asker(m, named, names) if kind == "interview" else None),
        "together": together,
        "copies": 1,
    }


def _passes(m: dict, kind: str, f: Filters, th: dict, sent: dict | None = None) -> bool:
    """The filters of :func:`_where` for a conversation worked out in Python
    (a stretch): the same comparisons, a missing measure fails them.
    ``sent``: the people's own measure of sentence length (:func:`_own`),
    else the baseline's (``th``)."""
    d = m["duration_sec"]
    if f.min_sec is not None and d < f.min_sec:
        return False
    if f.max_sec is not None and d > f.max_sec:
        return False
    if f.kinds and kind not in f.kinds:
        return False
    if f.group and m["speakers"] < GROUP_SPEAKERS:
        return False
    wps, rate = m["wps"], m["turn_rate"]
    st = sent or th
    if f.sentences == "longer" and not (wps is not None and wps >= st["wps60"]):
        return False
    if f.sentences == "shorter" and not (wps is not None and wps <= st["wps40"]):
        return False
    for value, lo, hi in ((wps, f.wps_min, f.wps_max), (rate, f.rate_min, f.rate_max)):
        if lo is not None and not (value is not None and value >= lo):
            return False
        if hi is not None and not (value is not None and value <= hi):
            return False
    if f.speakers_min is not None and m["speakers"] < f.speakers_min:
        return False
    if f.speakers_max is not None and m["speakers"] > f.speakers_max:
        return False
    if f.questions_min is not None:
        se = m["sentences"]
        if not (se and se > 0 and m["questions"] / se >= f.questions_min):
            return False
    return True


def _where(f: Filters, *, whole: bool = True, sent: dict | None = None) -> tuple[list[str], list]:
    """WHERE parts for :data:`_from` (parameterized; the sort and kind names
    come from whitelists). ``whole`` False: only what holds for a recording
    whatever stretch of it is asked for — where, when, and that it is not
    shorter than the shortest conversation asked for. ``sent``: as for
    :func:`_passes`."""
    w = ["t.complete = 1", "t.abs_path != ''"]
    a: list = []
    if f.language:
        w.append("lower(COALESCE(t.language, '')) = ?")
        a.append(f.language)
    if f.q:
        from whisper_tm.database import Database

        cond, args = Database._name_filter_sql(f.q, "t.abs_path")
        w.append(cond)
        a += args
    if f.folder:
        from whisper_tm.database import Database

        scope = Database._scope_range(f.folder)
        if scope:
            w.append("t.abs_path >= ? AND t.abs_path < ?")
            a += list(scope)
    lo = _day_start(f.date_from) if f.date_from else None
    hi = _day_start(f.date_to, after=True) if f.date_to else None
    if lo is not None:
        w.append(f"{_DAY_SQL} >= ?")
        a.append(lo)
    if hi is not None:
        w.append(f"{_DAY_SQL} < ?")
        a.append(hi)
    if f.min_sec is not None:
        w.append("s.duration_sec >= ?")
        a.append(f.min_sec)
    if not whole:
        return w, a
    if f.max_sec is not None:
        w.append("s.duration_sec <= ?")
        a.append(f.max_sec)
    if f.kinds:
        w.append(f"({_KIND_SQL}) IN ({','.join('?' * len(f.kinds))})")
        a += list(f.kinds)
    if f.group:
        w.append(f"s.speakers >= {GROUP_SPEAKERS}")
    if f.sentences == "longer" and sent:
        w.append("s.wps >= ?")
        a.append(sent["wps60"])
    elif f.sentences == "shorter" and sent:
        w.append("s.wps <= ?")
        a.append(sent["wps40"])
    elif f.sentences == "longer":
        w.append("s.wps >= COALESCE(bl.wps60, bf.wps60)")
    elif f.sentences == "shorter":
        w.append("s.wps <= COALESCE(bl.wps40, bf.wps40)")
    for col, lo_v, hi_v in (
        ("s.wps", f.wps_min, f.wps_max),
        ("s.turn_rate", f.rate_min, f.rate_max),
        ("s.speakers", f.speakers_min, f.speakers_max),
    ):
        if lo_v is not None:
            w.append(f"{col} >= ?")
            a.append(lo_v)
        if hi_v is not None:
            w.append(f"{col} <= ?")
            a.append(hi_v)
    if f.questions_min is not None:
        w.append("s.sentences > 0 AND CAST(s.questions AS REAL) / s.sentences >= ?")
        a.append(f.questions_min)
    return w, a


def _needs_base(f: Filters) -> bool:
    return bool(f.kinds) or f.sentences != "any"


def _needs_media(f: Filters) -> bool:
    # The file's date — a list of every transcript reads 200,000 rows less
    # without it
    return bool(f.date_from or f.date_to) or f.sort in ("newest", "oldest")


# ── Result rows, copies, order ─────────────────────────────────────────────


class _Row:
    """One conversation of a result list while it is made: what folding and
    ordering need, and its answer where it is worked out at once (a
    stretch's; a whole recording's waits for the page that shows it).
    ``rec`` is the recording's length, ``first``/``last`` when its speech
    starts and ends, ``lt_start``/``lt_sec`` its longest turn; ``dur``,
    ``speech``, ``words`` and ``speakers`` are the conversation's (a
    stretch's own)."""

    __slots__ = (
        "tid",
        "path",
        "lang",
        "rec",
        "first",
        "last",
        "lt_start",
        "lt_sec",
        "dur",
        "speech",
        "speakers",
        "words",
        "sentences",
        "wps",
        "start",
        "whole",
        "sort_v",
        "copies",
        "result",
    )

    def __init__(
        self,
        tid,
        path,
        lang,
        *,
        rec,
        first,
        last,
        dur,
        speech,
        speakers,
        words,
        sentences,
        wps=None,
        lt_start=0.0,
        lt_sec=0.0,
        start=0.0,
        whole=True,
        sort_v=None,
        result=None,
    ):
        # Numbers as SQL hands them (a list can hold every transcript of the
        # library: no conversions beyond a missing value)
        self.tid = tid
        self.path = path or ""
        self.lang = lang or ""
        self.rec = rec or 0.0
        self.first = first or 0.0
        self.last = last or 0.0
        self.lt_start = lt_start or 0.0
        self.lt_sec = lt_sec or 0.0
        self.dur = dur or 0.0
        self.speech = speech or 0.0
        self.speakers = speakers or 0
        self.words = words or 0
        self.sentences = sentences or 0.0
        self.wps = wps
        self.start = start or 0.0
        self.whole = whole
        self.sort_v = sort_v
        self.copies = 1
        self.result = result


def _rec_row(rec, **kw) -> _Row:
    """A whole recording's row from its record."""
    return _Row(
        rec["tid"],
        rec["abs_path"],
        rec["lang"],
        rec=rec["duration_sec"],
        first=rec["start_sec"],
        last=rec["end_sec"],
        lt_start=rec["longest_turn_start"],
        lt_sec=rec["longest_turn_sec"],
        dur=rec["duration_sec"],
        speech=rec["speech_sec"],
        speakers=rec["speakers"],
        words=rec["words"],
        sentences=rec["sentences"],
        wps=rec["wps"],
        **kw,
    )


def _word_bucket(words: int) -> int:
    # Words within COPY_SHARE of each other land in the same bucket or the
    # next one: a step of 2.5 % is wider than 1 / 0.98
    return int(math.log(words) / math.log(1.025))


def _turn_bucket(start: float) -> int:
    # Longest turns starting within COPY_EDGE_SEC of each other land in the
    # same bucket or the next one
    return int(start // COPY_EDGE_SEC)


# A cell of fold_copies as one number — speakers, word bucket, turn bucket —
# and its eight neighbours as offsets: a tuple per lookup cost a quarter of
# the folding of every transcript
_CELL_W = 1 << 20
_CELL_S = 1 << 40
_NEIGHBOURS = tuple(w * _CELL_W + t for w in (-1, 0, 1) for t in (-1, 0, 1))


def _cell(r: _Row) -> int:
    return r.speakers * _CELL_S + _word_bucket(r.words) * _CELL_W + _turn_bucket(r.lt_start)


def _close(x: float, y: float) -> bool:
    return abs(x - y) <= COPY_SHARE * max(x, y)


def _copies(a: _Row, b: _Row) -> bool:
    if a.path == b.path or a.whole != b.whole or a.speakers != b.speakers:
        return False
    if not (
        _close(a.words, b.words) and _close(a.sentences, b.sentences) and _close(a.speech, b.speech)
    ):
        return False
    if not copy_rules.same_length(a.rec, b.rec):
        return False
    # Copies share their timeline: their speech starts and ends together,
    # and so does their longest turn — where the parts of a stream cut to
    # one length differ, though they say about as many words in as much
    # speech from start to end
    if abs(a.first - b.first) > COPY_EDGE_SEC or abs(a.last - b.last) > COPY_EDGE_SEC:
        return False
    if (
        abs(a.lt_start - b.lt_start) > COPY_EDGE_SEC
        or abs(a.lt_start + a.lt_sec - b.lt_start - b.lt_sec) > COPY_EDGE_SEC
    ):
        return False
    # … and a stretch of one is the same stretch of the other
    return a.whole or (
        abs(a.start - b.start) <= metrics.WINDOW_SEC and abs(a.dur - b.dur) <= metrics.WINDOW_SEC
    )


def fold_copies(rows: list[_Row], names: _Names) -> list[_Row]:
    """The rows with copies folded (see the module), in their order: each
    kept row carries ``copies``, how many rows it stands for.

    Sorted by length, each row is compared only with the rows of nearly its
    length, words *and* longest turn — thousands of recordings of one length
    (a stream cut into parts of an hour) would be millions of pairs
    otherwise. A row joins the first of a group it is a copy of, never a
    chain of rows each a little apart: parts of one stream that talk alike
    would chain into one."""
    # Positions of the rows that say enough to be told apart, by length;
    # without a longest turn (not diarized) there is no moment to compare
    order = sorted(
        (i for i, r in enumerate(rows) if r.words >= COPY_MIN_WORDS and r.rec > 0 and r.lt_sec > 0),
        key=lambda i: rows[i].rec,
    )
    # The first row of each group, by cell (speakers, words, longest turn)
    near: dict[int, deque] = {}
    groups: dict[int, list[int]] = {}

    def first_of_group(r: _Row, cell: int) -> int | None:
        lo = r.rec - max(copy_rules.LENGTH_SEC, copy_rules.LENGTH_SHARE * r.rec)
        for step in _NEIGHBOURS:
            dq = near.get(cell + step)
            if not dq:
                continue
            while dq and rows[dq[0]].rec < lo:
                dq.popleft()
            for j in dq:
                if _copies(r, rows[j]):
                    return j
        return None

    for i in order:
        r = rows[i]
        cell = _cell(r)
        j = first_of_group(r, cell)
        if j is None:
            near.setdefault(cell, deque()).append(i)
            groups[i] = [i]
        else:
            groups[j].append(i)
    dropped: set[int] = set()
    for members in groups.values():
        if len(members) > 1:
            best = min(
                members,
                key=lambda i: (
                    -names.count(rows[i].path),
                    len(rows[i].path),
                    rows[i].path,
                    rows[i].start,
                ),
            )
            rows[best].copies = len(members)
            dropped.update(i for i in members if i != best)
    return [r for i, r in enumerate(rows) if i not in dropped] if dropped else list(rows)


def _order_sql(sort: str) -> str:
    """ORDER BY for whole recordings — the order :func:`_order` gives."""
    v = _SORT_SQL[sort]
    return f"ORDER BY ({v}) IS NULL, ({v}) {'DESC' if sort in _DESCENDING else 'ASC'}, t.abs_path"


def _order(rows: list[_Row], sort: str) -> None:
    desc = sort in _DESCENDING

    def key(r):
        v = r.sort_v
        if v is None:
            return (1, 0.0, r.path, r.start)
        return (0, -v if desc else v, r.path, r.start)

    rows.sort(key=key)


def _sort_value(sort: str, m: dict, rec, together):
    if sort in ("longest", "shortest"):
        return m["duration_sec"]
    if sort == "sentences":
        return m["wps"]
    if sort == "rate":
        return m["turn_rate"]
    if sort == "questions":
        return m["questions"] / m["sentences"] if m["sentences"] else None
    if sort == "monologue":
        return m["longest_turn_sec"]
    if sort in ("newest", "oldest"):
        return _when(rec)[0]
    return together


# ── Finding ────────────────────────────────────────────────────────────────

_ROW_SQL = (
    "SELECT t.id, t.abs_path, lower(COALESCE(t.language, '')), s.duration_sec, "
    "s.start_sec, s.end_sec, s.longest_turn_start, s.longest_turn_sec, s.speech_sec, "
    "s.speakers, s.words, s.sentences, s.attributed_sec, s.wps"
)


def _whole_rows(conn, f: Filters) -> list[_Row]:
    """Whole recordings, every filter in SQL — the query without people."""
    where, args = _where(f)
    sql = (
        f"{_ROW_SQL}, {_SORT_SQL[f.sort]} "
        f"{_from(base=_needs_base(f), media=_needs_media(f))} "
        f"WHERE {' AND '.join(where)} {_order_sql(f.sort)}"
    )
    return [
        _Row(
            tid,
            path,
            lang,
            rec=dur,
            first=first,
            last=last,
            lt_start=lt_start,
            lt_sec=lt_sec,
            dur=dur,
            speech=speech,
            speakers=speakers,
            words=words,
            sentences=sents,
            wps=wps,
            sort_v=k,
        )
        for (
            tid,
            path,
            lang,
            dur,
            first,
            last,
            lt_start,
            lt_sec,
            speech,
            speakers,
            words,
            sents,
            _att,
            wps,
            k,
        ) in conn.execute(sql, args)
    ]


def _any_rows(
    conn, f: Filters, people: list[str], min_share: float, names: _Names, sent: dict | None = None
) -> list[_Row]:
    """Whole recordings in which at least one of ``people`` takes part."""
    keys = set().union(*(names.recordings(p) for p in people))
    if not keys:
        return []
    _temp_paths(conn, keys)
    where, args = _where(f, sent=sent)
    sql = (
        f"{_ROW_SQL}, {_SORT_SQL[f.sort]} "
        f"{_from(people=True, base=_needs_base(f), media=_needs_media(f))} "
        f"WHERE {' AND '.join(where)} {_order_sql(f.sort)}"
    )
    found = conn.execute(sql, args).fetchall()
    labels = _labels_of(conn, (r[0] for r in found))
    out = []
    for (
        tid,
        path,
        lang,
        dur,
        first,
        last,
        lt_start,
        lt_sec,
        speech,
        speakers,
        words,
        sents,
        att,
        wps,
        k,
    ) in found:
        talk = _person_talk(_per_label(labels.get(tid, {})), names.labels(path), people)
        att = att or 0.0
        if not any(att > 0 and t / att >= min_share and t >= PART_SEC for t in talk.values()):
            continue
        together = sum(talk.values()) / att if att else 0.0
        out.append(
            _Row(
                tid,
                path,
                lang,
                rec=dur,
                first=first,
                last=last,
                lt_start=lt_start,
                lt_sec=lt_sec,
                dur=dur,
                speech=speech,
                speakers=speakers,
                words=words,
                sentences=sents,
                wps=wps,
                sort_v=together if f.sort == "together" else k,
            )
        )
    return out


def _conversation_row(rec, m, kind, named, names, base, people, sort) -> _Row:
    res = _result(rec, m, named, names, base, people, kind=kind)
    # (the recording's longest turn, also for a stretch: copies share it)
    return _Row(
        rec["tid"],
        rec["abs_path"],
        rec["lang"],
        rec=rec["duration_sec"],
        first=rec["start_sec"],
        last=rec["end_sec"],
        lt_start=rec["longest_turn_start"],
        lt_sec=rec["longest_turn_sec"],
        dur=m["duration_sec"],
        speech=m["speech_sec"],
        speakers=m["speakers"],
        words=m["words"],
        sentences=m["sentences"],
        wps=m["wps"],
        start=m["start_sec"],
        whole=m["whole"],
        result=res,
        sort_v=_sort_value(sort, m, rec, res["together"]),
    )


def _together_rows(
    conn,
    f: Filters,
    people: list[str],
    min_share: float,
    names: _Names,
    bases: dict,
    sent: dict | None = None,
) -> list[_Row]:
    """Conversations in which all of ``people`` take part: the stretches of
    long recordings, short ones whole (see the module)."""
    keys = set.intersection(*(names.recordings(p) for p in people))
    if not keys:
        return []
    _temp_paths(conn, keys)
    where, args = _where(f, whole=False)
    recs = conn.execute(
        _REC_SQL + _from(people=True) + " WHERE " + " AND ".join(where), args
    ).fetchall()
    labels = _labels_of(conn, (r["tid"] for r in recs))
    out: list[_Row] = []
    looked: dict[int, tuple] = {}

    def whole_one(rec, named, base):
        m = _whole(rec, labels.get(rec["tid"], {}))
        kind = kind_of(m, thresholds(base))
        if _passes(m, kind, f, thresholds(base), sent):
            out.append(_conversation_row(rec, m, kind, named, names, base, people, f.sort))

    def by_share(rec, named, base):
        talk = _person_talk(_per_label(labels.get(rec["tid"], {})), named, people)
        att = float(rec["attributed_sec"] or 0.0)
        if att > 0 and all(t / att >= min_share and t >= PART_SEC for t in talk.values()):
            whole_one(rec, named, base)

    for rec in recs:
        per = _per_label(labels.get(rec["tid"], {}))
        named = names.labels(rec["abs_path"])
        talk = _person_talk(per, named, people)
        att = float(rec["attributed_sec"] or 0.0)
        share = {p: (t / att if att > 0 else 0.0) for p, t in talk.items()}
        base = baseline_for(bases, rec["lang"])
        if rec["has_windows"]:
            if all(talk[p] >= CANDIDATE_SEC or share[p] >= min_share for p in people):
                looked[rec["tid"]] = (rec, named, base)
        elif all(share[p] >= min_share and talk[p] >= PART_SEC for p in people):
            whole_one(rec, named, base)
    for tid, win in _each_windows(conn, list(looked)):
        rec, named, base = looked.pop(tid)
        if win is None:  # unreadable: the whole recording, by the share rule
            by_share(rec, named, base)
        else:
            out.extend(
                _stretch_rows(
                    rec, win, labels.get(tid, {}), named, names, base, people, min_share, f, sent
                )
            )
    # Windows gone since the recordings were read (a write): the same
    for rec, named, base in looked.values():
        by_share(rec, named, base)
    return out


def _stretch_rows(
    rec,
    win: _Windows,
    label_rows: dict,
    named: dict,
    names: _Names,
    base: dict,
    people: list[str],
    min_share: float,
    f: Filters,
    sent: dict | None = None,
) -> list[_Row]:
    th = thresholds(base)
    held = {p: {x for x, g in named.items() if g == p} for p in people}
    share = max(min_share, WINDOW_SHARE)
    flags = [
        all(win.talk(held[p], i) >= max(WINDOW_TALK_SEC, share * win.total[i]) for p in people)
        for i in range(win.n)
    ]
    # A recording shorter than three windows has no room for two of them
    # around a pause: one window is a stretch there
    runs = _runs(flags, 1 if float(rec["duration_sec"] or 0) < 3 * win.size else 2)
    spoken = sum(1 for t in win.total if t > 0)
    whole_m, parts = None, []
    for a, b in runs:
        covered = sum(1 for i in range(a, b + 1) if win.total[i] > 0)
        if spoken and covered >= WHOLE_SHARE * spoken:
            whole_m = _whole(rec, label_rows)
        else:
            parts.append(_stretch(win, a, b, rec))
    found = []
    if whole_m is not None:
        kind = kind_of(whole_m, th)
        if _passes(whole_m, kind, f, th, sent):
            # The whole recording holds whatever else they said together
            return [_conversation_row(rec, whole_m, kind, named, names, base, people, f.sort)]
    for m in parts:
        kind = kind_of(m, th)
        if _passes(m, kind, f, th, sent):
            found.append((m, kind))
    found.sort(key=lambda x: (-x[0]["duration_sec"], x[0]["start_sec"]))
    return [
        _conversation_row(rec, m, kind, named, names, base, people, f.sort)
        for m, kind in found[:STRETCHES_MAX]
    ]


def _materialize(db, tids: list[int], names: _Names, bases: dict, people) -> dict[int, dict]:
    """``{transcript id: answer}`` of whole recordings — for the page shown
    only. One deleted since the list was made has none."""
    with _reading(db) as conn:
        _temp_ids(conn, tids)
        recs = {
            rec["tid"]: rec
            for rec in conn.execute(
                _REC_SQL + "FROM temp.cv_ids i JOIN transcripts t ON t.id = i.tid "
                "JOIN conv_stats s ON s.transcript_id = t.id "
                "LEFT JOIN media_files m ON m.id = t.media_id"
            )
        }
        labels = _labels_of(conn, tids)
    return {
        tid: _result(
            rec,
            _whole(rec, labels.get(tid, {})),
            names.labels(rec["abs_path"]),
            names,
            baseline_for(bases, rec["lang"]),
            people,
        )
        for tid, rec in recs.items()
    }


# Result lists per filter key: {"at", "index", "value"}
_cache: dict[tuple, dict] = {}
_cache_lock = threading.Lock()
_compute_lock = threading.RLock()


def _cached(key: tuple, index) -> dict | None:
    now = time.monotonic()
    with _cache_lock:
        e = _cache.get(key)
        if e is not None and now - e["at"] < RESULTS_TTL_SEC and e["index"] is index:
            return e["value"]
    return None


def _keep(key: tuple, index, value: dict) -> None:
    now = time.monotonic()
    with _cache_lock:
        for k in [k for k, e in _cache.items() if now - e["at"] >= RESULTS_TTL_SEC]:
            del _cache[k]
        _cache.pop(key, None)
        _cache[key] = {"at": now, "index": index, "value": value}
        while len(_cache) > RESULTS_MAX:
            del _cache[next(iter(_cache))]


def _held(key: tuple, index, make):
    """The value held for ``key`` — made once while it is current. The key
    carries the statistics' generation; the names index must be the same
    object (rebuilt when a name changes)."""
    got = _cached(key, index)
    if got is None:
        with _compute_lock:
            got = _cached(key, index)
            if got is None:
                got = make()
                _keep(key, index, got)
    return got


def forget() -> None:
    """Drop everything held (tests; a library switched underneath)."""
    with _cache_lock:
        _cache.clear()
    with _base_lock:
        _base_held.update(key=None, at=-1e18, value=None)
    with _counts_lock:
        _counts_held.update(key=None, at=-1e18, value=None)


def _own(
    conn, f: Filters, people: list[str], min_share: float, names: _Names, bases: dict, db_path: str
) -> dict:
    """The people's own measure of sentence length (see the module): the
    deciles of words per sentence over their conversations — together, or
    with ``match: any`` each in which one of them takes part — with as many
    voices as ``f`` asks for, in its language if it names one, copies
    folded, whatever else it filters (length, kind, place, time).
    ``{"conversations": n}``, with ``wps40``/``wps60`` from :data:`OWN_MIN`
    of them. Held like a result list: the tab asks again with every other
    filter."""
    key = (
        "own",
        db_path,
        store.generation(),
        tuple(people),
        f.match,
        min_share,
        f.speakers_min,
        f.speakers_max,
        f.language,
    )

    def make() -> dict:
        ref = Filters(
            people=f.people,
            match=f.match,
            min_share=f.min_share,
            speakers_min=f.speakers_min,
            speakers_max=f.speakers_max,
            language=f.language,
        )
        rows = (
            _together_rows(conn, ref, people, min_share, names, bases)
            if f.match == "all"
            else _any_rows(conn, ref, people, min_share, names)
        )
        wps = [r.wps for r in fold_copies(rows, names) if r.wps is not None]
        out: dict = {"conversations": len(wps)}
        if len(wps) >= OWN_MIN:
            d = deciles(wps)
            out.update(wps40=d[3], wps60=d[5])
        return out

    return _held(key, names.index, make)


def _sentences_baseline(
    f: Filters, own: dict | None, base: dict, people: list[str], names: _Names
) -> dict | None:
    """What "longer/shorter sentences" was measured against, for the
    answer: the people's own conversations, or the baseline (the majority
    language's; each recording is measured against its own language's)."""
    if f.sentences == "any":
        return None
    if own and own.get("wps60") is not None:
        return {
            "sentences": f.sentences,
            "source": "people",
            "people": [names.shown(p) for p in people],
            "conversations": own["conversations"],
            "longer_from": round(own["wps60"], 1),
            "shorter_to": round(own["wps40"], 1),
        }
    th = thresholds(base)
    out = {
        "sentences": f.sentences,
        "source": base["source"],
        "language": base["language"],
        "conversations": base["conversations"],
        "longer_from": round(th["wps60"], 1),
        "shorter_to": round(th["wps40"], 1),
    }
    if own is not None:
        # Too few of their own to measure against
        out["people_conversations"] = own["conversations"]
    return out


def _listed(db, f: Filters) -> tuple[dict, _Names, list[str], list[str], list[str]]:
    """The result list ``f`` asks for — made once and held: ``(held, names,
    people, unknown, ignored)``. :func:`find` pages through it,
    :func:`ranges` takes all of it."""
    names = _Names(_index())
    people, unknown = resolve_people(names, f.people)
    ignored = []
    if len(people) > PEOPLE_MAX:
        people = people[:PEOPLE_MAX]
        ignored.append("people")
    min_share = _min_share(f)
    key = (
        "find",
        db._db_path,
        store.generation(),
        astuple(replace(f, people=())),
        tuple(people),
        tuple(unknown),
        min_share,
    )

    def make() -> dict:
        with _reading(db) as conn:
            bases = baselines(conn, db._db_path)
            if _needs_base(f):
                _temp_base(conn, bases)
            # Whole recordings come from SQL in order; stretches, and the
            # people's share, are worked out here and sorted here
            ordered = True
            nobody = bool(f.people) and (not people or (f.match == "all" and bool(unknown)))
            own = (
                _own(conn, f, people, min_share, names, bases, db._db_path)
                if people and not nobody and f.sentences != "any"
                else None
            )
            sent = own if own and own.get("wps60") is not None else None
            if nobody:
                rows = []  # nobody known — or not all of them: nobody together
            elif people and f.match == "all":
                rows = _together_rows(conn, f, people, min_share, names, bases, sent)
                ordered = False
            elif people:
                rows = _any_rows(conn, f, people, min_share, names, sent)
                ordered = f.sort != "together"
            else:
                rows = _whole_rows(conn, f)
        rows = fold_copies(rows, names) if f.copies == "fold" else rows
        if not ordered:
            _order(rows, f.sort)
        langs = Counter(r.lang for r in rows).most_common(1)
        base = (
            baseline_for(bases, langs[0][0])
            if langs and 2 * langs[0][1] > len(rows)
            else bases["library"]
        )
        # Held: the order as transcript ids, the copies where there were
        # some, and the answers worked out so far (a stretch's at once) —
        # not the rows: a list of every transcript is 200,000 of them, near
        # 100 MB, and up to RESULTS_MAX lists are held
        return {
            "tids": array("q", (r.tid for r in rows)),
            "copies": {i: r.copies for i, r in enumerate(rows) if r.copies > 1},
            "results": {i: r.result for i, r in enumerate(rows) if r.result is not None},
            "bases": bases,
            "baseline": public_baseline(base),
            "sentences_baseline": _sentences_baseline(f, own, base, people, names),
            "total_sec": round(sum(r.dur for r in rows), 1),
        }

    return _held(key, names.index, make), names, people, unknown, ignored


def find(db, f: Filters, offset: int = 0, limit: int = LIMIT_DEFAULT) -> dict:
    """``GET /api/conversations`` without the request: one page of the
    conversations ``f`` asks for (see the module and api/conversations.py)."""
    held, names, people, unknown, ignored = _listed(db, f)
    tids, results = held["tids"], held["results"]
    shown = range(min(offset, len(tids)), min(offset + limit, len(tids)))
    todo = [i for i in shown if i not in results]
    if todo:
        got = _materialize(db, [tids[i] for i in todo], names, held["bases"], people)
        for i in todo:
            if tids[i] in got:
                results[i] = got[tids[i]]
    return {
        "total": len(tids),
        "offset": offset,
        "limit": limit,
        "truncated": offset + limit < len(tids),
        # How long all of them last together — the page's sum is no total
        "total_sec": held["total_sec"],
        "results": [
            dict(results[i], copies=held["copies"].get(i, 1)) for i in shown if i in results
        ],
        "baseline": held["baseline"],
        # What "longer/shorter sentences" was measured against (None
        # without that filter)
        "sentences_baseline": held["sentences_baseline"],
        "people_unknown": unknown,
        "ignored": ignored,
        "status": status_brief(db),
    }


def ranges(db, f: Filters) -> dict:
    """Every conversation ``f`` finds as time ranges per recording — what the
    voice extractor takes its sentences from (core/voice_extract.py): a
    stretch's own range, a whole recording all of it (``(0, inf)``).
    ``{"ranges": {abs_path: [(start, end), …]}, "conversations": n,
    "people_unknown": […], "ignored": […], "sentences_baseline": …}``."""
    held, _names, _people, unknown, ignored = _listed(db, f)
    tids, results = held["tids"], held["results"]
    out: dict[str, list[tuple[float, float]]] = {}
    whole: list[int] = []
    for i, tid in enumerate(tids):
        r = results.get(i)
        if r is None or r.get("whole"):
            whole.append(tid)
        else:
            out.setdefault(r["abs_path"], []).append(
                (float(r["start_sec"] or 0.0), float(r["end_sec"] or 0.0))
            )
    if whole:
        with _reading(db) as conn:
            _temp_ids(conn, whole)
            for (path,) in conn.execute(
                "SELECT t.abs_path FROM temp.cv_ids i JOIN transcripts t ON t.id = i.tid"
            ):
                out.setdefault(path, []).append((0.0, math.inf))
    return {
        "ranges": out,
        "conversations": len(tids),
        "people_unknown": unknown,
        "ignored": ignored,
        "sentences_baseline": held["sentences_baseline"],
    }


# ── One recording ──────────────────────────────────────────────────────────


def _merge_equal(parts: list[list]) -> list[list]:
    out: list[list] = []
    for p in parts:
        if out and out[-1][2] == p[2]:
            out[-1] = [out[-1][0], p[1], p[2]]
        else:
            out.append(list(p))
    return out


def _parts(win: _Windows, named: dict, names: _Names, rec, min_share: float) -> list[dict]:
    """Who talked when: per window the active named people and how many
    active voices have no name (the rule of the stretches); consecutive
    equal windows are one part, a single window between two equal ones
    takes theirs — a pause, an aside; talk between silence stays. Parts in
    which nobody is active are left out (the same people on both sides of
    one are one part); of more than :data:`PARTS_MAX`, the two neighbours
    shortest together become one, with the people of the longer."""
    share = max(min_share, WINDOW_SHARE)
    sets, talks = [], []
    for i in range(win.n):
        floor = max(WINDOW_TALK_SEC, share * win.total[i])
        talk: dict = {}
        for label, cols in win.labels.items():
            if label == metrics.UNATTRIBUTED:
                continue
            k = named.get(label) or ("", label)
            talk[k] = talk.get(k, 0.0) + cols[0][i]
        on = {k: t for k, t in talk.items() if t >= floor}
        sets.append(
            (
                frozenset(k for k in on if isinstance(k, str)),
                sum(1 for k in on if not isinstance(k, str)),
            )
        )
        talks.append(on)
    smooth = list(sets)
    for i in range(1, len(sets) - 1):
        nb = sets[i - 1]
        if nb == sets[i + 1] and sets[i] != nb and (nb[0] or nb[1]):
            smooth[i] = nb
    # [first window, last window, (people, unnamed)]
    parts = _merge_equal([i, i, s] for i, s in enumerate(smooth))
    parts = _merge_equal(p for p in parts if p[2][0] or p[2][1])
    while len(parts) > PARTS_MAX:
        # Shortest pair first: equal parts are paired off along the whole
        # recording instead of one part swallowing its neighbours one by one
        k = min(range(len(parts) - 1), key=lambda j: (parts[j + 1][1] - parts[j][0], j))
        left, right = parts[k], parts[k + 1]
        keep = left if left[1] - left[0] >= right[1] - right[0] else right
        parts[k : k + 2] = [[left[0], right[1], keep[2]]]
        parts = _merge_equal(parts)
    out = []
    first, last = float(rec["start_sec"] or 0.0), float(rec["end_sec"] or 0.0)
    for a, b, (people, unnamed) in parts:
        talk: dict[str, float] = {}
        for i in range(a, b + 1):
            for k, t in talks[i].items():
                if k in people:
                    talk[k] = talk.get(k, 0.0) + t
        end = (b + 1) * win.size
        ranked = sorted(people, key=lambda x: (-talk.get(x, 0.0), names.shown(x).lower()))
        out.append(
            {
                "start_sec": round(max(a * win.size, first), 1),
                "end_sec": round(min(end, last) if last else end, 1),
                "people": [names.shown(k) for k in ranked],
                "unnamed": unnamed,
            }
        )
    return out


def file_stats(db, abs_path: str, min_share: float | None = None) -> dict | None:
    """``GET /api/conversations/file``: one recording — None when it has no
    transcript."""
    names = _Names(_index())
    share = min_share if min_share is not None else _min_share(Filters())
    with _reading(db) as conn:
        t = conn.execute(
            f"SELECT t.id, t.abs_path, t.rel_path, {_MEDIA_DIR_SQL} AS media_dir, "
            "lower(COALESCE(t.language, '')) AS lang FROM transcripts t "
            "WHERE t.abs_path = ? AND t.complete = 1 LIMIT 1",
            (abs_path,),
        ).fetchone()
        if t is None:
            return None
        bases = baselines(conn, db._db_path)
        base = baseline_for(bases, t["lang"])
        rec = conn.execute(_REC_SQL + _from() + "WHERE t.id = ?", (t["id"],)).fetchone()
        label_rows = _labels_of(conn, [t["id"]]).get(t["id"], {}) if rec else {}
        win = _windows_of(conn, [t["id"]]).get(t["id"]) if rec and rec["has_windows"] else None
    file_dir, filename = place(t["abs_path"], t["rel_path"], t["media_dir"])
    out = {
        "abs_path": t["abs_path"],
        "filename": filename,
        "file_dir": file_dir,
        "counted": rec is not None,
        "stale": rec is None,
        "summary": {},
        "kind": None,
        "group": False,
        "reasons": [],
        "speakers": [],
        "people": [],
        "unnamed": 0,
        "parts": [],
        "baseline": public_baseline(base),
    }
    if rec is None:
        return out
    named = names.labels(t["abs_path"])
    m = _whole(rec, label_rows)
    res = _result(rec, m, named, names, base)
    att = m["attributed_sec"]
    punctuated = bool(rec["punctuated"])
    speakers = []
    for label, d in sorted(label_rows.items(), key=lambda x: -(x[1]["talk_sec"] or 0)):
        sents = d["sentences"] or 0
        speakers.append(
            {
                "label": label,
                "name": names.shown(named[label]) if label in named else None,
                "active": label in m["active"],
                "talk_sec": _r(d["talk_sec"] or 0.0, 1),
                "share": round((d["talk_sec"] or 0.0) / att, 3) if att else 0.0,
                "words": d["words"] or 0,
                "sentences": sents,
                "wps": round((d["words"] or 0) / sents, 1) if punctuated and sents else None,
                "questions": d["questions"] or 0,
                "turns": d["turns"] or 0,
                "short_turns": d["short_turns"] or 0,
                "longest_turn_sec": _r(d["longest_turn_sec"], 1),
                "longest_turn_start": _r(d["longest_turn_start"], 1),
                "first_sec": _r(d["first_sec"], 1),
                "last_sec": _r(d["last_sec"], 1),
            }
        )
    people: dict[str, dict] = {}
    for label, f in named.items():
        if label not in m["per_label"]:
            continue
        p = people.setdefault(f, {"name": names.shown(f), "labels": [], "talk_sec": 0.0})
        p["labels"].append(label)
        p["talk_sec"] += m["per_label"][label]["talk_sec"]
    ranked = sorted(
        (p for p in people.values() if p["talk_sec"] > 0),
        key=lambda p: (-p["talk_sec"], p["name"].lower()),
    )
    for p in ranked:
        p["labels"].sort()
        p["share"] = round(p["talk_sec"] / att, 3) if att else 0.0
        p["talk_sec"] = round(p["talk_sec"], 1)
    summary = {
        k: rec[k]
        for k in (
            "start_sec",
            "end_sec",
            "duration_sec",
            "speech_sec",
            "attributed_sec",
            "unattributed_sec",
            "words",
            "sentences",
            "questions",
            "turns",
            "short_turns",
            "labels",
            "speakers",
            "top_share",
            "longest_turn_sec",
            "longest_turn_label",
            "longest_turn_start",
            "wps",
            "turn_rate",
            "mean_turn_sec",
        )
    }
    summary.update(
        punctuated=punctuated,
        interview=bool(rec["interview"]),
        wps=_r(rec["wps"], 1),
        turn_rate=_r(rec["turn_rate"], 2),
        mean_turn_sec=_r(rec["mean_turn_sec"], 1),
        top_share=_r(rec["top_share"], 3),
        wps_pct=res["wps_pct"],
        rate_pct=res["rate_pct"],
        turn_pct=res["turn_pct"],
        questions_share=res["questions_share"],
        longest_turn_name=res["longest_turn_name"],
        language=res["language"],
        date=res["date"],
        date_source=res["date_source"],
    )
    out.update(
        stale=(
            rec["version"] != store.STATS_VERSION
            or rec["seg_count"] != rec["t_seg_count"]
            or rec["char_count"] != rec["t_char_count"]
        ),
        summary=summary,
        kind=res["kind"],
        group=res["group"],
        reasons=res["reasons"],
        speakers=speakers,
        people=ranked,
        unnamed=res["unnamed"],
        parts=_parts(win, named, names, rec, share) if win is not None else [],
    )
    return out


# ── People ─────────────────────────────────────────────────────────────────


def _taking_part(conn, names: _Names, people: list[str], min_share: float) -> dict:
    """``{person: [(record, their labels' numbers, named labels)]}`` — the
    recordings each of ``people`` takes part in."""
    keys = set().union(*(names.recordings(p) for p in people)) if people else set()
    out: dict[str, list] = {p: [] for p in people}
    if not keys:
        return out
    _temp_paths(conn, keys)
    recs = conn.execute(
        _REC_SQL + _from(people=True) + "WHERE t.complete = 1 AND t.abs_path != ''"
    ).fetchall()
    labels = _labels_of(conn, (r["tid"] for r in recs))
    want = set(people)
    for rec in recs:
        per = _per_label(labels.get(rec["tid"], {}))
        named = names.labels(rec["abs_path"])
        att = float(rec["attributed_sec"] or 0.0)
        if att <= 0:
            continue
        for p, talk in _person_talk(per, named, set(named.values()) & want).items():
            if talk / att >= min_share and talk >= PART_SEC:
                out[p].append((rec, per, named))
    return out


def _kept(items: list, names: _Names) -> list:
    """``items`` (record first) with copies folded."""
    return [
        r.result for r in fold_copies([_rec_row(item[0], result=item) for item in items], names)
    ]


def people_list(db, min_share: float | None = None) -> dict:
    """``GET /api/conversations/people``: everyone named who takes part in at
    least one counted recording — most conversations first."""
    names = _Names(_index())
    share = min_share if min_share is not None else _min_share(Filters())
    key = ("people", db._db_path, store.generation(), share)

    def make() -> dict:
        everyone = names.everyone()
        with _reading(db) as conn:
            taking = _taking_part(conn, names, everyone, share)
        out = []
        for p, items in taking.items():
            kept = _kept(items, names)
            if kept:
                out.append(
                    {
                        "name": names.shown(p),
                        "recordings": len(kept),
                        "conversations": sum(
                            1 for rec, _, _ in kept if (rec["speakers"] or 0) >= 2
                        ),
                    }
                )
        out.sort(key=lambda x: (-x["conversations"], x["name"].lower(), x["name"]))
        return {"people": out}

    return _held(key, names.index, make)


def person(db, name: str, min_share: float | None = None) -> dict:
    """``GET /api/conversations/person``: how one person talks, and with
    whom. A name nobody has: ``known`` false and zeros."""
    names = _Names(_index())
    f = fold(name)
    share = min_share if min_share is not None else _min_share(Filters())
    empty = {
        "name": clean_name(name),
        "known": False,
        "recordings": 0,
        "conversations": 0,
        "talk_sec": 0.0,
        "wps": None,
        "questions_share": None,
        "partners": [],
        "kinds": {},
        "first_date": None,
        "last_date": None,
        "longest": [],
    }
    if not names.known(f):
        return empty
    key = ("person", db._db_path, store.generation(), f, share)

    def make() -> dict:
        with _reading(db) as conn:
            bases = baselines(conn, db._db_path)
            kept = _kept(_taking_part(conn, names, [f], share)[f], names)
        talk = words = sents = quests = 0.0
        partners: dict[str, list] = {}
        kinds: Counter = Counter()
        times = []
        for rec, per, named in kept:
            att = float(rec["attributed_sec"] or 0.0)
            mine = [x for x, g in named.items() if g == f and x in per]
            talk += sum(per[x]["talk_sec"] for x in mine)
            if rec["punctuated"]:
                words += sum(per[x]["words"] for x in mine)
                sents += sum(per[x]["sentences"] for x in mine)
                quests += sum(per[x]["questions"] for x in mine)
            for g, t in _person_talk(per, named, set(named.values()) - {f}).items():
                if att > 0 and t / att >= share and t >= PART_SEC:
                    p = partners.setdefault(g, [0, 0.0])
                    p[0] += 1
                    p[1] += float(rec["speech_sec"] or 0.0)
            kinds[kind_of(_whole(rec, {}), thresholds(baseline_for(bases, rec["lang"])))] += 1
            ts, _ = _when(rec)
            if ts is not None:
                times.append(ts)
        ranked = sorted(
            partners.items(), key=lambda x: (-x[1][0], -x[1][1], names.shown(x[0]).lower())
        )[:12]
        longest = find(
            db, Filters(people=(names.shown(f),), sort="longest", min_share=share), 0, 5
        )["results"]
        return {
            "name": names.shown(f),
            "known": True,
            "recordings": len(kept),
            "conversations": sum(1 for rec, _, _ in kept if (rec["speakers"] or 0) >= 2),
            "talk_sec": round(talk, 1),
            "wps": round(words / sents, 1) if sents else None,
            "questions_share": round(quests / sents, 3) if sents else None,
            "partners": [
                {"name": names.shown(g), "recordings": n, "together_sec": round(sec, 1)}
                for g, (n, sec) in ranked
            ],
            "kinds": dict(kinds.most_common()),
            "first_date": iso_day(min(times)) if times else None,
            "last_date": iso_day(max(times)) if times else None,
            "longest": longest,
        }

    return _held(key, names.index, make)


# ── The library's counts ───────────────────────────────────────────────────

_counts_lock = threading.Lock()
_counts_held: dict = {"key": None, "at": -1e18, "value": None}


def counts(db) -> dict:
    """``{"transcripts", "counted", "stale"}`` — held :data:`COUNTS_TTL_SEC`."""
    now = time.monotonic()
    with _counts_lock:
        if _counts_held["key"] == db._db_path and now - _counts_held["at"] < COUNTS_TTL_SEC:
            return dict(_counts_held["value"])
    with _reading(db) as conn:
        value = store.counts(conn)
    with _counts_lock:
        _counts_held.update(key=db._db_path, at=now, value=value)
    return dict(value)


def status_brief(db) -> dict:
    """What a result list says about the counting behind it."""
    from whisper_tm.conversations import runner

    try:
        c = counts(db)
    except sqlite3.Error as e:
        logger.debug("Conversations: counts not readable: %s", e)
        c = {"counted": 0, "stale": 0}
    return {
        "enabled": runner.settings()["enabled"],
        "counted": c["counted"],
        "stale": c["stale"],
        "running": runner.RUNNER.running(),
    }

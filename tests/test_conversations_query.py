"""Finding conversations (conversations/query.py): baselines, kinds and
their reasons, who takes part, the stretches of long recordings, filters,
sorts, copies, dates, scope and the held result lists.

Every number can be checked on paper. Long recordings are built window by
window: each label talks its seconds as one block of 10-second segments,
each segment one sentence of 12 words — so the talk, words, sentences and
speaker changes of a stretch can be counted by hand. Baselines, kinds and
filters are checked on rows written straight into the tables, so each test
shows the numbers it expects.
"""

from __future__ import annotations

import re
import weakref
from array import array

import numpy as np
import pytest

from whisper_tm.conversations import query, store
from whisper_tm.core import speaker as sp
from whisper_tm.database import Database
from whisper_tm.database.speaker import SpeakerDB
from whisper_tm.rag import speakers

ROOT = "/media/library"
A, B, C, D = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"
NAMES = {A: "Alice", B: "Bob", C: "Carol"}
#: One sentence of 12 words
SENT = "one two three four five six seven eight nine ten eleven twelve."


# ── Setup ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: ROOT))
    _ROWS[0] = 0
    query.forget()
    speakers.forget()
    yield
    query.forget()
    speakers.forget()


@pytest.fixture
def voices(tmp_path, monkeypatch):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    return d


@pytest.fixture
def db(tmp_path, monkeypatch, voices):
    d = Database(tmp_path / "wx.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.extensions.db", d)
    yield d
    d.close_all()


def _seg(speaker, start, end, text=SENT):
    return {"speaker": speaker, "start": start, "end": end, "text": text}


def _blocks(windows, text=SENT):
    """Segments window by window (300 s each): ``windows`` holds per window
    ``{label: seconds}``; each label talks its seconds as one block of
    10-second segments, in the dict's order, from the window's start."""
    segs = []
    for i, talk in enumerate(windows):
        t = i * 300.0
        for label, sec in talk.items():
            left = float(sec)
            while left > 1e-9:
                d = min(10.0, left)
                segs.append(_seg(label, t, t + d, text))
                t += d
                left -= d
    return segs


def _write(db, path, segs, *, names=None, language="en", duration=None):
    """Write a transcript (the hook counts it) — and forget the held names,
    as the editor's save does."""
    data = {
        "language": language,
        "segments": segs,
        "complete": True,
        "total_duration_sec": duration or max((s["end"] for s in segs), default=0),
        "speakers": sorted({s["speaker"] for s in segs if s["speaker"] != "—"}),
        "speaker_names": names or {},
    }
    if path.startswith(ROOT + "/"):
        rel, root = path[len(ROOT) + 1 :], ROOT
    else:
        root, _, rel = path.rpartition("/")
    tid = db.upsert_transcript(rel, data, media_dir=root, abs_path=path)
    speakers.forget()
    return tid


def _find(db, **raw):
    f, ignored = query.parse_filters(raw, media_dir=ROOT)
    out = query.find(db, f, 0, query.LIMIT_MAX)
    out["ignored"] = ignored + out["ignored"]
    return out


def _paths(out):
    return [r["abs_path"] for r in out["results"]]


_STATS = dict(
    version=store.STATS_VERSION,
    seg_count=10,
    char_count=100,
    name_date=None,
    computed_at=0.0,
    start_sec=0.0,
    end_sec=600.0,
    duration_sec=600.0,
    speech_sec=500.0,
    attributed_sec=500.0,
    unattributed_sec=0.0,
    words=1000,
    sentences=100,
    questions=10,
    turns=50,
    short_turns=5,
    labels=2,
    speakers=2,
    top_share=0.6,
    longest_turn_sec=60.0,
    longest_turn_label=A,
    longest_turn_start=10.0,
    punctuated=1,
    interview=0,
    wps=10.0,
    turn_rate=6.0,
    mean_turn_sec=10.0,
)


_ROWS = [0]


def _row(db, path, *, language="en", modified_at=0.0, rel_path=None, media_dir=None, **stats):
    """A transcript and its numbers written straight into the tables. The
    rows of a test differ in length unless it says otherwise — rows alike
    in every number would fold as copies. ``rel_path``/``media_dir``: a
    display path and stored root other than the ones the path gives."""
    _ROWS[0] += 1
    s = {**_STATS, **stats}
    if "duration_sec" not in stats:
        s["duration_sec"] = 600.0 + 7 * _ROWS[0]
    if "end_sec" not in stats:
        s["end_sec"] = s["duration_sec"]
    root, _, rel = path.rpartition("/")
    if path.startswith(ROOT + "/"):
        root, rel = ROOT, path[len(ROOT) + 1 :]
    rel = rel if rel_path is None else rel_path
    root = root if media_dir is None else media_dir
    with db.connection() as conn:
        mid = conn.execute(
            "INSERT INTO media_files (rel_path, abs_path, modified_at, status) "
            "VALUES (?, ?, ?, 'done')",
            (rel, path, modified_at),
        ).lastrowid
        tid = conn.execute(
            "INSERT INTO transcripts (media_id, rel_path, abs_path, language, complete, "
            "duration_sec, segment_count, char_count, media_dir) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
            (mid, rel, path, language, s["duration_sec"], s["seg_count"], s["char_count"], root),
        ).lastrowid
        cols = list(s)
        conn.execute(
            f"INSERT INTO conv_stats (transcript_id, {', '.join(cols)}) "
            f"VALUES ({', '.join('?' * (len(cols) + 1))})",
            (tid, *[s[c] for c in cols]),
        )
    store.changed()
    return tid


def _conversations(db, n, language="de", start=1, **extra):
    """``n`` rows that count for the baseline: words per sentence
    ``start`` … ``start+n-1``, speaker changes per minute the same, seconds
    per turn twice that."""
    for i in range(start, start + n):
        _row(
            db,
            f"{ROOT}/{language}/talk{i:03d}.mkv",
            language=language,
            wps=float(i),
            turn_rate=float(i),
            mean_turn_sec=2.0 * i,
            **extra,
        )


def _hour():
    """Twelve windows of a recording of 3400 s: Alice alone (0–2), Alice and
    Bob (3–7), Alice alone (8), Alice and Bob (9–10), Carol (11)."""
    return _blocks(
        [{A: 200}] * 3 + [{A: 100, B: 100}] * 5 + [{A: 150}] + [{A: 100, B: 100}] * 2 + [{C: 100}]
    )


# ── Baselines ──────────────────────────────────────────────────────────────


class TestBaselines:
    def test_deciles_interpolate_between_ranks(self):
        # 30 values 1…30: p10 lies at rank 2.9 → 3 + 0.9
        assert query.deciles(range(1, 31)) == pytest.approx(
            (3.9, 6.8, 9.7, 12.6, 15.5, 18.4, 21.3, 24.2, 27.1)
        )
        assert query.deciles([7.0]) == (7.0,) * 9

    def test_a_language_with_enough_conversations_has_its_own(self, db):
        _conversations(db, 30)
        with query._reading(db) as conn:
            bases = query.baselines(conn, db._db_path)
        de = query.baseline_for(bases, "de")
        assert (de["source"], de["conversations"], de["language"]) == ("language", 30, "de")
        pub = query.public_baseline(de)
        assert pub["wps"]["p50"] == 15.5 and pub["wps"]["p60"] == 18.4
        assert pub["turn_rate"]["p40"] == 12.6
        assert pub["mean_turn_sec"]["p50"] == 31.0

    def test_a_language_with_few_borrows_the_library(self, db):
        _conversations(db, 30)
        _conversations(db, 5, language="en", start=100)
        with query._reading(db) as conn:
            bases = query.baselines(conn, db._db_path)
        en = query.baseline_for(bases, "en")
        assert (en["source"], en["conversations"], en["language"]) == ("library", 35, "en")
        # 35 values: p60 at rank 20.4 → 21.4 (the five of 100 on top)
        assert en["wps"][5] == pytest.approx(21.4)
        assert query.baseline_for(bases, "fi")["source"] == "library"
        assert query.baseline_for(bases, "de")["source"] == "language"

    def test_too_few_conversations_use_the_generic_figures(self, db):
        _conversations(db, 29)
        with query._reading(db) as conn:
            de = query.baseline_for(query.baselines(conn, db._db_path), "de")
        assert (de["source"], de["conversations"]) == ("defaults", 0)
        pub = query.public_baseline(de)
        assert [pub["wps"][f"p{p}"] for p in query.PCTS] == [5, 6, 7, 8, 9, 10, 11, 12, 13]
        assert (pub["turn_rate"]["p40"], pub["turn_rate"]["p50"], pub["turn_rate"]["p60"]) == (
            4,
            5,
            6,
        )
        assert (
            pub["mean_turn_sec"]["p40"],
            pub["mean_turn_sec"]["p50"],
            pub["mean_turn_sec"]["p60"],
        ) == (8, 10, 12)

    def test_only_real_conversations_count(self, db):
        _conversations(db, 30)
        for i, bad in enumerate(
            [
                dict(top_share=0.9),
                dict(speakers=1),
                dict(punctuated=0),
                dict(sentences=19),
                dict(attributed_sec=100.0),
                dict(turn_rate=None),
            ]
        ):
            _row(db, f"{ROOT}/de/bad{i}.mkv", language="de", **{"wps": 500.0, **bad})
        with query._reading(db) as conn:
            de = query.baseline_for(query.baselines(conn, db._db_path), "de")
        assert de["conversations"] == 30
        assert de["wps"][8] == pytest.approx(27.1)  # no 500 among them

    def test_held_until_the_numbers_change(self, db, monkeypatch):
        _conversations(db, 30)
        with query._reading(db) as conn:
            first = query.baselines(conn, db._db_path)
            assert query.baselines(conn, db._db_path) is first
            _conversations(db, 30, start=31)  # moves the generation
            second = query.baselines(conn, db._db_path)
            assert second is not first
            assert second["languages"]["de"]["conversations"] == 60
            monkeypatch.setattr(query, "BASE_MAX_AGE_SEC", 0.0)
            assert query.baselines(conn, db._db_path) is not second


class TestPercentiles:
    DEC = (3.9, 6.8, 9.7, 12.6, 15.5, 18.4, 21.3, 24.2, 27.1)

    def test_between_deciles_linearly(self):
        assert query.percentile(20.0, self.DEC) == 66  # 60 + 10 × 1.6 / 2.9
        assert query.percentile(21.3, self.DEC) == 70

    def test_beyond_the_deciles_clamped(self):
        assert query.percentile(3.0, self.DEC) == 7  # 10 - 0.9 × 10 / 2.9
        assert query.percentile(-50.0, self.DEC) == 0
        assert query.percentile(500.0, self.DEC) == 100

    def test_equal_deciles_meet_in_their_middle(self):
        assert query.percentile(8.0, (5, 6, 8, 8, 8, 9, 10, 11, 12)) == 40
        assert query.percentile(3.0, (3,) * 9) == 50
        assert query.percentile(2.0, (3,) * 9) == 0
        assert query.percentile(None, self.DEC) is None


# ── Kinds and reasons ──────────────────────────────────────────────────────

TH = {"wps60": 10.0, "wps50": 9.0, "wps40": 8.0, "turn50": 10.0, "rate60": 6.0}
#: One row per rule of the kinds (the generic thresholds above apply)
GRID = {
    "silent": dict(words=0, labels=0, speakers=0, wps=None),
    "unknown": dict(labels=0, speakers=0),
    "solo": dict(speakers=1, top_share=0.5),
    "dominated": dict(top_share=0.9, interview=1),
    "interview": dict(interview=1, wps=20.0, mean_turn_sec=30.0),
    "detailed": dict(wps=10.0, mean_turn_sec=10.0, turn_rate=9.0),
    "lively": dict(wps=9.0, mean_turn_sec=5.0, turn_rate=6.0),
    "lively_unpunctuated": dict(wps=None, punctuated=0, turn_rate=7.0),
    "long_sentences_short_turns": dict(wps=12.0, mean_turn_sec=9.9, turn_rate=6.5),
    "long_sentences_unknown_turns": dict(wps=12.0, mean_turn_sec=None, turn_rate=None),
    "plain": dict(wps=9.5, mean_turn_sec=12.0, turn_rate=5.0),
}
GRID_KINDS = {
    "silent": "silent",
    "unknown": "unknown",
    "solo": "monologue",
    "dominated": "monologue",
    "interview": "interview",
    "detailed": "detailed",
    "lively": "lively",
    "lively_unpunctuated": "lively",
    "long_sentences_short_turns": "conversation",
    "long_sentences_unknown_turns": "conversation",
    "plain": "conversation",
}


class TestKinds:
    def test_the_rules_in_their_order(self):
        for name, over in GRID.items():
            m = {**_STATS, **over}
            assert query.kind_of(m, TH) == GRID_KINDS[name], name

    def test_sql_and_python_say_the_same(self, db):
        for name, over in GRID.items():
            _row(db, f"{ROOT}/grid/{name}.mkv", **over)
        for kind in query.KINDS:
            want = {f"{ROOT}/grid/{n}.mkv" for n, k in GRID_KINDS.items() if k == kind}
            out = _find(db, kind=kind)
            assert set(_paths(out)) == want, kind
            assert {r["kind"] for r in out["results"]} <= {kind}
        # Every recording once, under the kind its answer names
        every = _find(db)
        assert {
            r["abs_path"].rsplit("/", 1)[1][:-4]: r["kind"] for r in every["results"]
        } == GRID_KINDS

    def test_each_language_its_own_thresholds(self, db):
        _conversations(db, 30)  # de: wps p60 18.4, turns p50 31
        _conversations(db, 5, language="en", start=100)  # en borrows: wps p60 21.4
        # 110 s of speech: not one of the baseline's conversations itself
        same = dict(wps=19.0, mean_turn_sec=40.0, turn_rate=1.0, attributed_sec=110.0)
        _row(db, f"{ROOT}/x/de.mkv", language="de", **same)
        _row(db, f"{ROOT}/x/en.mkv", language="en", **same)
        _row(db, f"{ROOT}/x/fi.mkv", language="fi", **same)
        got = {r["abs_path"]: r["kind"] for r in _find(db, folder=f"{ROOT}/x")["results"]}
        assert got == {
            f"{ROOT}/x/de.mkv": "detailed",
            f"{ROOT}/x/en.mkv": "conversation",
            f"{ROOT}/x/fi.mkv": "conversation",
        }
        assert _paths(_find(db, folder=f"{ROOT}/x", kind="detailed")) == [f"{ROOT}/x/de.mkv"]

    def test_a_language_named_like_the_fallback_row(self, db):
        """'*' keys the thresholds every language without its own borrows."""
        _conversations(db, 30, language="*")
        _conversations(db, 30, language="de", start=40)
        assert _find(db, kind="detailed")["total"] >= 1

    def test_the_group_tag(self, db):
        _row(db, f"{ROOT}/two.mkv", speakers=2)
        _row(db, f"{ROOT}/three.mkv", speakers=3, top_share=0.4, wps=9.0)
        _row(db, f"{ROOT}/four.mkv", speakers=4, top_share=0.3, wps=12.0, mean_turn_sec=20.0)
        assert set(_paths(_find(db, kind="group"))) == {f"{ROOT}/three.mkv", f"{ROOT}/four.mkv"}
        # With a kind: groups of that kind
        assert _paths(_find(db, kind="detailed,group")) == [f"{ROOT}/four.mkv"]
        assert {r["abs_path"]: r["group"] for r in _find(db)["results"]} == {
            f"{ROOT}/two.mkv": False,
            f"{ROOT}/three.mkv": True,
            f"{ROOT}/four.mkv": True,
        }


class TestReasons:
    def test_long_sentences_and_turns_are_named_with_their_numbers(self, db):
        _conversations(db, 30)
        _row(
            db,
            f"{ROOT}/x/talk.mkv",
            language="de",
            wps=20.0,
            mean_turn_sec=40.0,
            turn_rate=1.5,
            attributed_sec=110.0,
        )
        [r] = _find(db, folder=f"{ROOT}/x")["results"]
        assert r["kind"] == "detailed"
        assert (r["wps_pct"], r["turn_pct"]) == (66, 66)
        assert r["reasons"] == [
            "Sentences of 20.0 words — longer than 66 % of conversations in this language.",
            "Turns of 40 s on average — longer than 66 %.",
        ]

    def test_short_sentences_say_shorter(self, db):
        _row(db, f"{ROOT}/quick.mkv", wps=7.0, turn_rate=8.0, mean_turn_sec=7.5)
        [r] = _find(db)["results"]
        assert r["kind"] == "lively"
        # The generic figures: 8 changes = p80; 7 words = p30
        assert r["reasons"] == [
            "8.0 speaker changes a minute — more than 80 % of typical conversations.",
            "Sentences of 7.0 words — shorter than 70 % of typical conversations.",
        ]

    def test_a_monologue_names_its_share(self, db):
        _row(db, f"{ROOT}/lecture.mkv", top_share=0.91, longest_turn_sec=250.0)
        [r] = _find(db)["results"]
        assert r["reasons"] == [
            "One voice speaks 91 % of the time.",
            "Its longest turn lasts 4:10.",
        ]

    def test_an_interview_names_the_asker(self, db):
        segs, t = [], 0.0
        for _ in range(8):
            segs.append(_seg(A, t, t + 10, "What happened next?"))
            segs.append(
                _seg(B, t + 10, t + 40, "Well, it was a long story and it went on and on. " * 3)
            )
            t += 40
        _write(db, f"{ROOT}/interview.mkv", segs, names={A: "Bob"})
        [r] = _find(db)["results"]
        assert r["kind"] == "interview"
        assert r["reasons"] == [
            "Two voices, one asking: 100 % of Bob's sentences are questions.",
            "The asker speaks 25 % of the time.",
        ]

    def test_no_speech_and_no_speakers(self, db):
        _write(db, f"{ROOT}/plain.mkv", [_seg("—", 0, 5, "Nobody was told apart here.")])
        _write(db, f"{ROOT}/quiet.mkv", [_seg(A, 0, 5, "123 456.")])
        got = {r["abs_path"]: (r["kind"], r["reasons"]) for r in _find(db)["results"]}
        assert got[f"{ROOT}/plain.mkv"] == (
            "unknown",
            ["Its speakers were not told apart: who spoke when is not known."],
        )
        assert got[f"{ROOT}/quiet.mkv"] == ("silent", ["Nobody speaks in it."])


# ── Who takes part ─────────────────────────────────────────────────────────


class TestPeople:
    def test_taking_part_needs_a_share_and_seconds(self, db):
        # Bob: 6 s of 150 = 4 % — a remark, under the default 5 %
        _write(db, f"{ROOT}/call.mkv", _blocks([{A: 144, B: 6}]), names={A: "Alice", B: "Bob"})
        assert _find(db, people="Bob")["total"] == 0
        assert _find(db, people="Bob", min_share="0.03")["total"] == 1
        # … and 5 s at least, whatever the share
        _write(db, f"{ROOT}/short.mkv", _blocks([{A: 40, B: 4}]), names={A: "Alice", B: "Bob"})
        assert _paths(_find(db, people="Bob", min_share="0.03")) == [f"{ROOT}/call.mkv"]

    def test_all_of_them_together_or_any_of_them(self, db):
        _write(db, f"{ROOT}/ab.mkv", _blocks([{A: 60, B: 60}]), names=NAMES)
        _write(db, f"{ROOT}/ac.mkv", _blocks([{A: 60, C: 70}]), names=NAMES)
        _write(db, f"{ROOT}/b.mkv", _blocks([{B: 80}]), names=NAMES)
        assert _paths(_find(db, people=["Alice", "Bob"])) == [f"{ROOT}/ab.mkv"]
        assert set(_paths(_find(db, people="Alice,Bob", match="any"))) == {
            f"{ROOT}/ab.mkv",
            f"{ROOT}/ac.mkv",
            f"{ROOT}/b.mkv",
        }
        # Any case: names compare as the voice library does
        assert _paths(_find(db, people=["ALICE", "bob"])) == [f"{ROOT}/ab.mkv"]

    def test_each_result_lists_its_people(self, db):
        # Alice 90 s, Bob 60 s, an unnamed voice 30 s, Carol 1 s (under 1 %: left out)
        _write(
            db,
            f"{ROOT}/call.mkv",
            _blocks([{A: 90, B: 60, D: 30, C: 1}]),
            names={A: "Alice", B: "Bob", C: "Carol"},
        )
        [r] = _find(db)["results"]
        assert r["people"] == [
            {"name": "Alice", "talk_sec": 90.0, "share": round(90 / 181, 3), "wps": 12.0},
            {"name": "Bob", "talk_sec": 60.0, "share": round(60 / 181, 3), "wps": 12.0},
        ]
        assert (r["speakers"], r["unnamed"]) == (3, 1)
        assert r["together"] is None

    def test_voice_map_names_and_typed_names_first(self, db, voices):
        path = f"{ROOT}/meeting.mkv"
        ids = _vectors(voices, path, {A: 2, B: 2})
        voices.assign_many(ids[A], "Dave")
        voices.assign_many(ids[B], "Erin")
        _write(db, path, _blocks([{A: 60, B: 60}]), names={B: "Frank"})
        [r] = _find(db)["results"]
        assert [p["name"] for p in r["people"]] == ["Dave", "Frank"]
        assert _find(db, people="Erin")["total"] == 0
        assert _find(db, people=["Dave", "Frank"])["total"] == 1

    def test_unknown_people_are_named_not_failed(self, db):
        _write(db, f"{ROOT}/ab.mkv", _blocks([{A: 60, B: 60}]), names=NAMES)
        out = _find(db, people=["Alice", "Grace"])
        assert out["people_unknown"] == ["Grace"]
        assert out["total"] == 0  # nobody can say Grace was there
        out = _find(db, people=["Alice", "Grace"], match="any")
        assert (out["total"], out["people_unknown"]) == (1, ["Grace"])

    def test_a_name_with_a_comma_is_one_person_when_someone_has_it(self, db):
        _write(
            db, f"{ROOT}/ab.mkv", _blocks([{A: 60, B: 60}]), names={A: "Frank, Senior", B: "Bob"}
        )
        assert _find(db, people=["Frank, Senior", "Bob"])["total"] == 1
        assert _find(db, people="Frank, Senior")["total"] == 1
        assert _find(db, people="Bob,Senior")["people_unknown"] == ["Senior"]

    def test_more_people_than_can_be_asked_for_are_named_in_ignored(self, db):
        """One value can name any number of people: those after the first
        PEOPLE_MAX are left out, and the answer says so."""
        crowd = [f"Guest {i}" for i in range(1, query.PEOPLE_MAX + 2)]
        labels = [f"SPEAKER_{i:02d}" for i in range(len(crowd))]
        _write(
            db,
            f"{ROOT}/crowd.mkv",
            _blocks([dict.fromkeys(labels, 20)]),
            names=dict(zip(labels, crowd)),
        )
        out = _find(db, people=",".join(crowd), match="any")
        assert (out["total"], out["ignored"], out["people_unknown"]) == (1, ["people"], [])
        assert _find(db, people=",".join(crowd[:-1]), match="any")["ignored"] == []


def _vectors(voices, path, labels):
    """Voice windows of a recording: ``{label: n}`` → ``{label: [ids]}``."""
    rng = np.random.default_rng(len(path))
    rows = []
    for label, n in labels.items():
        for i in range(n):
            v = rng.normal(size=192).astype(np.float32)
            rows.append(
                {
                    "label": label,
                    "windows": i + 1,
                    "start_sec": i * 40.0,
                    "speech_sec": 30.0,
                    "segments": 2,
                    "parts": [(i * 40.0, i * 40.0 + 30)],
                    "v_separated": v / np.linalg.norm(v),
                }
            )
    voices.write(path, sp.MODEL_BRAND, rows)
    out: dict[str, list[int]] = {}
    for w in voices.windows(path):
        out.setdefault(w["label"], []).append(w["id"])
    return out


# ── Stretches of long recordings ───────────────────────────────────────────


class TestStretches:
    def test_the_stretch_where_they_talk_together(self, db):
        _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        out = _find(db, people=["Alice", "Bob"])
        [r] = out["results"]
        # Windows 3–10 (8 alone with Alice between): 900 … 11 × 300
        assert (r["whole"], r["start_sec"], r["end_sec"], r["duration_sec"]) == (
            False,
            900.0,
            3300.0,
            2400.0,
        )
        assert r["recording_sec"] == 3400.0
        # 7 windows of 100 + 100 and one of Alice's 150
        assert r["speech_sec"] == 1550.0
        assert (r["words"], r["sentences"], r["wps"]) == (1860, 155, 12.0)
        # Changes where each block starts: 1 (window 3: Bob), 2 × 4, 1 (8), 1 (9), 2 (10)
        assert r["turn_rate"] == round(13 / (1550 / 60), 2)
        assert r["mean_turn_sec"] == round(1550 / 13, 1)
        assert r["people"] == [
            {"name": "Alice", "talk_sec": 850.0, "share": round(850 / 1550, 3), "wps": 12.0},
            {"name": "Bob", "talk_sec": 700.0, "share": round(700 / 1550, 3), "wps": 12.0},
        ]
        assert (r["speakers"], r["unnamed"], r["together"]) == (2, 0, 1.0)
        # The recording's longest turn (Alice's first 200 s) lies before it
        assert r["longest_turn_sec"] is None and r["longest_turn_name"] is None
        assert r["kind"] == "detailed" and r["wps_pct"] == 80

    def test_the_whole_recording_answers_without_people(self, db):
        _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        [r] = _find(db)["results"]
        assert (r["whole"], r["start_sec"], r["end_sec"]) == (True, 0.0, 3400.0)
        assert (r["speech_sec"], r["words"], r["sentences"]) == (2250.0, 2700, 225)
        assert (r["turn_rate"], r["mean_turn_sec"]) == (0.4, 150.0)  # 15 changes
        assert [p["name"] for p in r["people"]] == ["Alice", "Bob", "Carol"]
        assert r["speakers"] == 2  # Carol's 4.4 % is under 5 %
        assert (r["longest_turn_sec"], r["longest_turn_start"], r["longest_turn_name"]) == (
            200.0,
            0.0,
            "Alice",
        )

    def test_a_single_window_between_counts_in_two_do_not(self, db):
        hole = [{A: 100, B: 100}] * 2 + [{A: 200}] + [{A: 100, B: 100}] * 2 + [{A: 200}] * 6
        _write(db, f"{ROOT}/one.mkv", _blocks(hole), names=NAMES)
        gap = [{A: 100, B: 100}] * 2 + [{A: 200}] * 2 + [{A: 100, B: 100}] * 2 + [{A: 200}] * 6
        _write(db, f"{ROOT}/two.mkv", _blocks(gap), names=NAMES)
        got = [
            (r["abs_path"].rsplit("/", 1)[1], r["start_sec"], r["end_sec"])
            for r in _find(db, people=["Alice", "Bob"])["results"]
        ]
        assert got == [
            ("one.mkv", 0.0, 1500.0),
            ("two.mkv", 0.0, 600.0),
            ("two.mkv", 1200.0, 1800.0),
        ]

    def test_several_stretches_longest_first(self, db):
        w = (
            [{A: 100, B: 100}] * 2
            + [{A: 200}] * 3
            + [{A: 100, B: 100}] * 4
            + [{A: 200}] * 3
            + [{A: 100, B: 100}] * 3
        )
        _write(db, f"{ROOT}/day.mkv", _blocks(w), names=NAMES)
        got = [(r["start_sec"], r["end_sec"]) for r in _find(db, people="Alice,Bob")["results"]]
        assert got == [(1500.0, 2700.0), (3600.0, 4400.0), (0.0, 600.0)]
        # The last stretch ends with the recording's speech: 14 × 300 + 200
        assert _find(db, people="Alice,Bob", max_sec="1000")["total"] == 2

    def test_nine_tenths_of_the_recording_is_the_whole_of_it(self, db):
        _write(db, f"{ROOT}/most.mkv", _blocks([{A: 100, B: 100}] * 9 + [{A: 200}]), names=NAMES)
        _write(
            db, f"{ROOT}/less.mkv", _blocks([{A: 100, B: 100}] * 8 + [{A: 200}] * 2), names=NAMES
        )
        got = {
            r["abs_path"].rsplit("/", 1)[1]: (r["whole"], r["start_sec"], r["end_sec"])
            for r in _find(db, people=["Alice", "Bob"])["results"]
        }
        assert got == {"most.mkv": (True, 0.0, 2900.0), "less.mkv": (False, 0.0, 2400.0)}

    def test_one_window_is_a_stretch_in_a_short_recording(self, db):
        # 700 s: three windows, only the middle one together
        _write(
            db,
            f"{ROOT}/short.mkv",
            _blocks([{A: 200}, {A: 100, B: 100}, {A: 50}]),
            names=NAMES,
            duration=700,
        )
        # 1000 s: four windows — one is not enough
        _write(
            db,
            f"{ROOT}/longer.mkv",
            _blocks([{A: 200}, {A: 100, B: 100}, {A: 200}, {A: 100}]),
            names=NAMES,
        )
        [r] = _find(db, people=["Alice", "Bob"])["results"]
        assert (r["abs_path"], r["start_sec"], r["end_sec"]) == (f"{ROOT}/short.mkv", 300.0, 600.0)

    def test_one_person_in_two_labels(self, db):
        # Alice talks as SPEAKER_00 and SPEAKER_02, 10 s each per window —
        # neither alone reaches the 15 s of an active part, both do
        w = [{A: 10, C: 10, B: 200}] * 4 + [{B: 200}] * 2
        _write(db, f"{ROOT}/split.mkv", _blocks(w), names={A: "Alice", C: "Alice", B: "Bob"})
        [r] = _find(db, people=["Alice", "Bob"])["results"]
        assert (r["start_sec"], r["end_sec"]) == (0.0, 1200.0)
        assert r["people"][1] == {
            "name": "Alice",
            "talk_sec": 80.0,
            "share": round(80 / 880, 3),
            "wps": 12.0,
        }

    def test_someone_quiet_in_a_long_recording_is_not_looked_for(self, db):
        # Bob: 20 s in 12 windows of Alice — under 60 s and under 5 %
        w = [{A: 290, B: 20}] + [{A: 290}] * 11
        _write(db, f"{ROOT}/lecture.mkv", _blocks(w), names=NAMES)
        assert _find(db, people=["Alice", "Bob"])["total"] == 0

    def test_windows_are_read_only_where_each_of_them_talks(self, db, monkeypatch):
        """The windows are the one large thing a query reads: only for the
        recordings in which every person asked for talks somewhere."""
        stream = _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        _write(db, f"{ROOT}/lecture.mkv", _blocks([{A: 290, B: 20}] + [{A: 290}] * 11), names=NAMES)
        # Bob is named here too (the typed names), but never speaks
        _write(db, f"{ROOT}/other.mkv", _blocks([{A: 100, C: 100}] * 4), names=NAMES)
        read = []
        real = query._each_windows
        monkeypatch.setattr(
            query, "_each_windows", lambda conn, tids: read.append(sorted(tids)) or real(conn, tids)
        )
        assert _find(db, people="Alice,Bob")["total"] == 1
        assert read == [[stream]]

    def test_one_recordings_windows_at_a_time(self, db, monkeypatch):
        """Someone in thousands of long recordings: their windows are read and
        dropped one recording after the other, never all held at once."""
        for i in range(4):
            _write(
                db,
                f"{ROOT}/stream{i}.mkv",
                _blocks([{A: 100 + 10 * i, B: 100}] * 4 + [{A: 200}] * 2),
                names=NAMES,
            )
        alive, most = weakref.WeakSet(), [0]
        real = query._Windows.__init__

        def counted(self, data):
            real(self, data)
            alive.add(self)
            most[0] = max(most[0], len(alive))

        monkeypatch.setattr(query._Windows, "__init__", counted)
        assert _find(db, people="Alice,Bob")["total"] == 4
        # (the one before is let go when the loop takes the next)
        assert most[0] <= 2

    def test_a_stretch_starts_with_the_speech(self, db):
        # Two minutes of nothing, then Alice and Bob together in windows
        # 0–3, then Alice alone until 3570 s
        segs = [
            dict(s, start=s["start"] + 120, end=s["end"] + 120)
            for s in _blocks([{A: 100, B: 60}] * 4 + [{A: 150}] * 8)
        ]
        _write(db, f"{ROOT}/late.mkv", segs, names=NAMES)
        [r] = _find(db, people="Alice,Bob")["results"]
        assert (r["whole"], r["start_sec"], r["end_sec"], r["recording_sec"]) == (
            False,
            120.0,
            1200.0,
            3570.0,
        )

    def test_a_stretch_must_pass_the_filters(self, db):
        _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        assert _find(db, people="Alice,Bob", min_sec="2400")["total"] == 1
        assert _find(db, people="Alice,Bob", min_sec="2401")["total"] == 0
        assert _find(db, people="Alice,Bob", kind="detailed", sentences="longer")["total"] == 1
        assert _find(db, people="Alice,Bob", kind="lively")["total"] == 0
        assert _find(db, people="Alice,Bob", speakers_min="3")["total"] == 0
        assert _find(db, people="Alice,Bob", wps_min="12", wps_max="12")["total"] == 1

    def test_at_most_five_stretches_of_a_recording(self, db):
        # Seven times: two windows together, two of Alice alone
        _write(
            db,
            f"{ROOT}/week.mkv",
            _blocks(([{A: 100, B: 100}] * 2 + [{A: 200}] * 2) * 7),
            names=NAMES,
        )
        got = [r["start_sec"] for r in _find(db, people="Alice,Bob")["results"]]
        assert got == [0.0, 1200.0, 2400.0, 3600.0, 4800.0]

    @pytest.mark.parametrize(
        "data",
        [
            "[1, 2]",
            "not json",
            '{"w": -5, "labels": {}}',
            '{"labels": {"SPEAKER_00": 5}}',
            '{"labels": {"SPEAKER_00": [["x", 1, 1, 1, 1, 1]]}}',
            # json.loads reads Infinity; int() of it is an OverflowError
            '{"labels": {"SPEAKER_00": [[Infinity, 1, 1, 1, 1, 1]]}}',
            # A window a year in: a broken row, not columns of that length
            '{"labels": {"SPEAKER_00": [[105120, 1, 1, 1, 1, 1]]}}',
        ],
    )
    def test_windows_it_cannot_read_leave_the_whole_recording(self, db, data):
        tid = _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        with db.connection() as conn:
            conn.execute("UPDATE conv_windows SET data = ? WHERE transcript_id = ?", (data, tid))
        store.changed()
        [r] = _find(db, people="Alice,Bob")["results"]
        assert (r["whole"], r["start_sec"], r["end_sec"]) == (True, 0.0, 3400.0)
        assert query.file_stats(db, f"{ROOT}/stream.mkv")["parts"] == []

    @pytest.mark.parametrize(
        "flags,min_len,runs",
        [
            ("TTFTTFFTT", 2, [(0, 4), (7, 8)]),  # one window between counts in, two don't
            ("TFT", 2, [(0, 2)]),
            ("FTF", 2, []),
            ("FTF", 1, [(1, 1)]),
            ("TFFT", 1, [(0, 0), (3, 3)]),
            ("FTFTF", 2, [(1, 3)]),
            ("", 2, []),
        ],
    )
    def test_runs_of_windows(self, flags, min_len, runs):
        assert query._runs([c == "T" for c in flags], min_len) == runs


# ── Filters and order ──────────────────────────────────────────────────────

FILTER_ROWS = {
    "short": dict(duration_sec=120.0, wps=6.0, turn_rate=9.0, speakers=2, questions=40),
    "medium": dict(duration_sec=1800.0, wps=9.0, turn_rate=5.0, speakers=3, top_share=0.5),
    "long": dict(
        duration_sec=7200.0,
        wps=14.0,
        turn_rate=2.0,
        speakers=4,
        top_share=0.4,
        mean_turn_sec=30.0,
        questions=5,
    ),
    "unpunctuated": dict(
        duration_sec=900.0, wps=None, punctuated=0, turn_rate=None, mean_turn_sec=None, speakers=2
    ),
}


class TestFilters:
    @pytest.fixture
    def rows(self, db):
        for name, over in FILTER_ROWS.items():
            _row(db, f"{ROOT}/f/{name}.mkv", **over)
        _row(db, f"{ROOT}/f/finnish.mkv", language="fi", duration_sec=300.0)
        return db

    @staticmethod
    def names(out):
        return sorted(r["abs_path"].rsplit("/", 1)[1][:-4] for r in out["results"])

    @pytest.mark.parametrize(
        "raw,want",
        [
            ({"min_sec": "900"}, ["long", "medium", "unpunctuated"]),
            ({"max_sec": "900"}, ["finnish", "short", "unpunctuated"]),
            ({"min_sec": "300", "max_sec": "1800"}, ["finnish", "medium", "unpunctuated"]),
            ({"sentences": "longer"}, ["finnish", "long"]),  # wps ≥ 10
            ({"sentences": "shorter"}, ["short"]),  # wps ≤ 8
            ({"wps_min": "9", "wps_max": "10"}, ["finnish", "medium"]),
            ({"rate_min": "5"}, ["finnish", "medium", "short"]),
            ({"rate_max": "5"}, ["long", "medium"]),
            ({"speakers_min": "3"}, ["long", "medium"]),
            ({"speakers_max": "2"}, ["finnish", "short", "unpunctuated"]),
            # More than SQLite's 64-bit integers hold: as many as there can be
            ({"speakers_min": "1e20"}, []),
            (
                {"speakers_max": "9223372036854775808"},
                ["finnish", "long", "medium", "short", "unpunctuated"],
            ),
            ({"questions_min": "0.2"}, ["short"]),  # 40 of 100 sentences
            ({"language": "FI"}, ["finnish"]),
            ({"kind": "lively"}, ["short"]),
        ],
    )
    def test_each_filter(self, rows, raw, want):
        assert self.names(_find(rows, **raw)) == want

    def test_sql_and_python_filter_alike(self, rows):
        """Stretches are filtered in Python: the same numbers must pass the
        same filters there as in SQL."""
        with query._reading(rows) as conn:
            bases = query.baselines(conn, rows._db_path)
            recs = conn.execute(query._REC_SQL + query._from() + "WHERE 1").fetchall()
        cases = [
            {"min_sec": "900"},
            {"max_sec": "900"},
            {"sentences": "longer"},
            {"sentences": "shorter"},
            {"wps_min": "9"},
            {"rate_max": "5"},
            {"speakers_min": "3"},
            {"questions_min": "0.2"},
            {"kind": "lively,detailed"},
            {"kind": "group"},
        ]
        for raw in cases:
            f, _ = query.parse_filters(raw)
            in_python = set()
            for rec in recs:
                th = query.thresholds(query.baseline_for(bases, rec["lang"]))
                m = query._whole(rec, {})
                if query._passes(m, query.kind_of(m, th), f, th):
                    in_python.add(rec["abs_path"])
            assert set(_paths(_find(rows, **raw))) == in_python, raw


def _words(n):
    """One sentence of ``n`` words."""
    return " ".join(["word"] * n) + "."


class TestOwnSentences:
    """With people asked for, "longer sentences" is measured against their
    own conversations: a round that always talks in short sentences still
    has its conversations in longer ones — measured against the library it
    would have none."""

    @pytest.fixture
    def rounds(self, db):
        # The library talks in long sentences (20 … 49 words) …
        _conversations(db, 30, language="en", start=20)
        # … Alice and Bob in short ones: twelve calls of 4 … 15 words
        for n in range(4, 16):
            _write(
                db,
                f"{ROOT}/ab/call{n:02d}.mkv",
                _blocks([{A: 60, B: 60}], text=_words(n)),
                names=NAMES,
            )
        return db

    def test_longer_and_shorter_than_usual_for_them(self, rounds):
        out = _find(rounds, people="Alice,Bob", sentences="longer")
        # Their deciles of 4 … 15: p60 at rank 6.6 → 10.6, p40 at 4.4 → 8.4
        assert sorted(r["wps"] for r in out["results"]) == [11.0, 12.0, 13.0, 14.0, 15.0]
        assert out["sentences_baseline"] == {
            "sentences": "longer",
            "source": "people",
            "people": ["Alice", "Bob"],
            "conversations": 12,
            "longer_from": 10.6,
            "shorter_to": 8.4,
        }
        short = _find(rounds, people="Alice,Bob", sentences="shorter")
        assert sorted(r["wps"] for r in short["results"]) == [4.0, 5.0, 6.0, 7.0, 8.0]

    def test_any_of_them_measures_each_they_take_part_in(self, rounds):
        out = _find(rounds, people="Alice", match="any", sentences="longer")
        assert out["sentences_baseline"]["source"] == "people"
        assert sorted(r["wps"] for r in out["results"]) == [11.0, 12.0, 13.0, 14.0, 15.0]

    def test_other_filters_leave_the_measure(self, rounds):
        out = _find(rounds, people="Alice,Bob", sentences="longer", wps_max="12")
        assert sorted(r["wps"] for r in out["results"]) == [11.0, 12.0]
        assert out["sentences_baseline"]["longer_from"] == 10.6

    def test_the_kinds_stay_measured_against_the_library(self, rounds):
        # Sentences of 15 words are short for this library: no call is detailed
        assert _find(rounds, people="Alice,Bob", kind="detailed")["total"] == 0

    def test_without_people_the_library_is_the_measure(self, rounds):
        out = _find(rounds, sentences="longer")
        b = out["sentences_baseline"]
        # The library's 30 conversations of 20 … 49 words: p60 at rank 17.4
        assert (b["source"], b["language"], b["longer_from"]) == ("language", "en", 37.4)
        assert out["results"] and all(r["wps"] >= 37.4 for r in out["results"])
        assert _find(rounds)["sentences_baseline"] is None

    def test_too_few_of_their_own_use_the_library(self, rounds):
        for n in (30, 40):
            _write(
                rounds,
                f"{ROOT}/ac/call{n}.mkv",
                _blocks([{A: 60, C: 60}], text=_words(n)),
                names=NAMES,
            )
        out = _find(rounds, people="Alice,Carol", sentences="longer")
        b = out["sentences_baseline"]
        assert (b["source"], b["people_conversations"]) == ("language", 2)
        assert [r["wps"] for r in out["results"]] == [40.0]


class TestSorts:
    @pytest.fixture
    def rows(self, db):
        _row(
            db,
            f"{ROOT}/s/a.mkv",
            duration_sec=100.0,
            wps=8.0,
            turn_rate=3.0,
            questions=50,
            longest_turn_sec=20.0,
            name_date=1_700_000_000.0,
        )
        _row(
            db,
            f"{ROOT}/s/b.mkv",
            duration_sec=300.0,
            wps=None,
            punctuated=0,
            turn_rate=9.0,
            questions=0,
            longest_turn_sec=300.0,
            modified_at=1_600_000_000.0,
        )
        _row(
            db,
            f"{ROOT}/s/c.mkv",
            duration_sec=200.0,
            wps=12.0,
            turn_rate=None,
            questions=10,
            longest_turn_sec=90.0,
        )
        return db

    @pytest.mark.parametrize(
        "sort,want",
        [
            ("longest", "bca"),
            ("shortest", "acb"),
            ("sentences", "cab"),
            ("rate", "bac"),
            ("questions", "acb"),
            ("monologue", "bca"),
            ("newest", "abc"),
            ("oldest", "bac"),
        ],
    )
    def test_each_order(self, rows, sort, want):
        got = "".join(r["abs_path"][-5] for r in _find(rows, sort=sort)["results"])
        assert got == want

    def test_together_by_the_share_of_the_people_asked_for(self, db):
        _write(db, f"{ROOT}/half.mkv", _blocks([{A: 30, B: 30, C: 60}]), names=NAMES)
        _write(db, f"{ROOT}/all.mkv", _blocks([{A: 30, B: 30}]), names=NAMES)
        out = _find(db, people="Alice,Bob", sort="together")
        assert [(r["abs_path"], r["together"]) for r in out["results"]] == [
            (f"{ROOT}/all.mkv", 1.0),
            (f"{ROOT}/half.mkv", 0.5),
        ]
        # Without people there is nothing to be together
        assert _find(db, sort="together")["ignored"] == ["sort"]

    def test_together_with_any_of_them(self, db):
        _write(db, f"{ROOT}/half.mkv", _blocks([{A: 30, C: 30}]), names=NAMES)
        _write(db, f"{ROOT}/most.mkv", _blocks([{A: 10, B: 80, C: 10}]), names=NAMES)
        out = _find(db, people="Alice,Bob", match="any", sort="together")
        assert [(r["abs_path"], r["together"]) for r in out["results"]] == [
            (f"{ROOT}/most.mkv", 0.9),
            (f"{ROOT}/half.mkv", 0.5),
        ]


# ── Copies ─────────────────────────────────────────────────────────────────


class TestCopies:
    def test_copies_are_one_result(self, db):
        segs = _blocks([{A: 150, B: 150}])  # 360 words
        _write(db, f"{ROOT}/talk.mkv", segs, names=NAMES)
        _write(db, f"{ROOT}/backup/talk (copy).mp4", segs, names=NAMES)
        # Another recording of the same length saying less
        _write(db, f"{ROOT}/other.mkv", _blocks([{A: 150, B: 150}], text="short words."))
        out = _find(db)
        assert out["total"] == 2
        got = {r["abs_path"]: r["copies"] for r in out["results"]}
        assert got == {f"{ROOT}/talk.mkv": 2, f"{ROOT}/other.mkv": 1}
        assert _find(db, copies="all")["total"] == 3

    def test_the_copy_kept_has_more_names_then_the_shorter_path(self, db):
        segs = _blocks([{A: 150, B: 150}])
        _write(db, f"{ROOT}/talk.mkv", segs, names={A: "Alice"})
        _write(db, f"{ROOT}/backup/talk.mp4", segs, names=NAMES)
        [r] = _find(db)["results"]
        assert (r["abs_path"], r["copies"]) == (f"{ROOT}/backup/talk.mp4", 2)

    def test_the_same_stretch_of_copies_is_one_result(self, db):
        _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        _write(db, f"{ROOT}/old/stream.webm", _hour(), names=NAMES)
        out = _find(db, people="Alice,Bob")
        assert [(r["abs_path"], r["copies"]) for r in out["results"]] == [(f"{ROOT}/stream.mkv", 2)]

    def test_short_copies_are_shown_twice(self, db):
        """Under 300 words counts can't tell copies from other recordings."""
        segs = _blocks([{A: 60, B: 60}])  # 144 words
        _write(db, f"{ROOT}/note.ogg", segs, names=NAMES)
        _write(db, f"{ROOT}/backup/note.ogg", segs, names=NAMES)
        assert [r["copies"] for r in _find(db)["results"]] == [1, 1]

    @staticmethod
    def _random_rows(n, seed=7, length=None, through=False):
        """``n`` different conversations: of 2 … 30 min, or all of
        ``length`` (parts of a stream) — ``through``: talked through from
        start to end, as alike in their counts as different parts get."""
        import random

        rnd = random.Random(seed)
        rows = []
        for i in range(n):
            rec = length or rnd.uniform(120, 1800)
            if through:
                first, last = rnd.uniform(0, 0.5), rec - rnd.uniform(0, 1)
                speech = (last - first) * rnd.uniform(0.92, 0.97)
                words = int(speech * rnd.uniform(2.4, 2.6))
            else:
                first, last = rnd.uniform(0, 5), rec - rnd.uniform(0, 60)
                speech = (last - first) * rnd.uniform(0.6, 0.95)
                words = int(speech * rnd.uniform(1.5, 3.0))
            turn = rnd.uniform(10, min(240, speech / 2))
            rows.append(
                query._Row(
                    i,
                    f"/m/{i}.mkv",
                    "en",
                    rec=rec,
                    first=first,
                    last=last,
                    lt_start=rnd.uniform(first, last - turn),
                    lt_sec=turn,
                    dur=rec,
                    speech=speech,
                    speakers=rnd.randint(1, 4),
                    words=words,
                    sentences=words / rnd.uniform(6, 14),
                )
            )
        return rows

    @staticmethod
    def _twins(rows):
        """Copies of ``rows`` — the same numbers, transcribed apart."""
        return [
            query._Row(
                100_000 + r.tid,
                f"/m/copy/{r.tid}.mkv",
                "en",
                rec=r.rec + 0.1,
                first=r.first + 0.5,
                last=r.last - 0.5,
                lt_start=r.lt_start + 0.5,
                lt_sec=r.lt_sec - 1.0,
                dur=r.dur + 0.1,
                speech=r.speech * 1.01,
                speakers=r.speakers,
                words=r.words + 2,
                sentences=r.sentences,
            )
            for r in rows
            if r.words >= query.COPY_MIN_WORDS
        ]

    NO_NAMES = query._Names({"names": {}, "people": {}, "shown": {}})

    def test_folding_compares_only_neighbours(self, monkeypatch):
        """Five thousand different conversations of 2 … 30 min: each is
        compared with the few of nearly its length and words, not with all
        the others — and none of them pass for copies."""
        calls = []
        real = query._copies
        monkeypatch.setattr(query, "_copies", lambda a, b: calls.append(1) or real(a, b))
        rows = self._random_rows(5000)
        assert len(query.fold_copies(rows, self.NO_NAMES)) == 5000
        assert len(calls) < 50_000  # all pairs: 12.5 million
        twins = self._twins(self._random_rows(5000)[:200])
        kept = query.fold_copies(self._random_rows(5000) + twins, self.NO_NAMES)
        assert len(kept) == 5000 and sum(r.copies for r in kept) == 5000 + len(twins)

    @pytest.mark.parametrize("through", [False, True])
    def test_parts_of_one_length_are_not_copies(self, monkeypatch, through):
        """A stream cut into parts of half an hour: as long as each other,
        and — talked through — about as many words in as much speech from
        start to end. Their longest turns differ; and a part is never folded
        into one that is folded into a third, so alike parts don't chain."""
        calls = []
        real = query._copies
        monkeypatch.setattr(query, "_copies", lambda a, b: calls.append(1) or real(a, b))
        parts = self._random_rows(2000, length=1800.0, through=through)
        assert len(query.fold_copies(parts, self.NO_NAMES)) == 2000
        assert len(calls) < 20_000  # all pairs: 2 million
        twins = self._twins(parts[:100])
        kept = query.fold_copies(
            self._random_rows(2000, length=1800.0, through=through) + twins, self.NO_NAMES
        )
        assert len(kept) == 2000 and sum(r.copies for r in kept) == 2000 + len(twins)

    def test_without_a_longest_turn_nothing_is_folded(self, db):
        """Not diarized: no moment of their own to tell parts apart by."""
        segs = [_seg("—", 10.0 * i, 10.0 * i + 10, SENT) for i in range(40)]  # 480 words
        _write(db, f"{ROOT}/lecture.mkv", segs)
        _write(db, f"{ROOT}/backup/lecture.mkv", segs)
        assert [r["copies"] for r in _find(db)["results"]] == [1, 1]


# ── Dates, scope, places ───────────────────────────────────────────────────


class TestDates:
    def test_the_date_in_the_name_first_then_the_file(self, db):
        _write(db, f"{ROOT}/call 2024-03-01 18-00.mkv", _blocks([{A: 60, B: 60}]))
        _write(db, f"{ROOT}/call.mkv", _blocks([{A: 70, B: 70}]))
        _write(db, f"{ROOT}/undated.mkv", _blocks([{A: 80, B: 80}]))
        when = query._day_start("2023-07-04") + 13 * 3600
        with db.connection() as conn:
            conn.execute(
                "UPDATE media_files SET modified_at = ? WHERE abs_path = ?",
                (when, f"{ROOT}/call.mkv"),
            )
            conn.execute(
                "UPDATE media_files SET modified_at = 0 WHERE abs_path = ?",
                (f"{ROOT}/undated.mkv",),
            )
        store.changed()
        got = {
            r["abs_path"].rsplit("/", 1)[1]: (r["date"], r["date_source"])
            for r in _find(db)["results"]
        }
        assert got == {
            "call 2024-03-01 18-00.mkv": ("2024-03-01", "name"),
            "call.mkv": ("2023-07-04", "file"),
            "undated.mkv": (None, None),
        }

    def test_whole_days_from_to(self, db):
        for i, day in enumerate(("2024-02-29", "2024-03-01", "2024-03-02")):
            _write(db, f"{ROOT}/call {day}.mkv", _blocks([{A: 60 + 10 * i, B: 60}]))
        names = [
            r["abs_path"][-14:-4]
            for r in _find(db, date_from="2024-03-01", date_to="2024-03-01")["results"]
        ]
        assert names == ["2024-03-01"]
        assert _find(db, date_from="2024-03-01")["total"] == 2
        assert _find(db, date_to="2024-03-01")["total"] == 2
        out = _find(db, date_from="2024-13-01", date_to="yesterday")
        assert out["total"] == 3 and out["ignored"] == ["date_from", "date_to"]


class TestScope:
    @pytest.fixture
    def tree(self, db):
        for path in (
            f"{ROOT}/Sub/a.mkv",
            f"{ROOT}/Sub2/b.mkv",
            f"{ROOT}/50%_off/c.mkv",
            "/media/elsewhere/Sub/d.mkv",
        ):
            _row(db, path, words=len(path) * 10)
        return db

    def test_a_folder_holds_what_lies_below_it(self, tree):
        assert _paths(_find(tree, folder=f"{ROOT}/Sub")) == [f"{ROOT}/Sub/a.mkv"]
        assert _paths(_find(tree, folder=f"{ROOT}/Sub/")) == [f"{ROOT}/Sub/a.mkv"]
        # Below the open folder
        assert _paths(_find(tree, folder="Sub")) == [f"{ROOT}/Sub/a.mkv"]
        assert _paths(_find(tree, folder="/media/elsewhere")) == ["/media/elsewhere/Sub/d.mkv"]
        # The library as a whole: every root
        assert _find(tree)["total"] == 4

    def test_a_folder_that_climbs_out_is_ignored(self, tree):
        out = _find(tree, folder="../elsewhere")
        assert out["ignored"] == ["folder"] and out["total"] == 4

    def test_letters_of_the_path_in_any_case(self, tree):
        assert set(_paths(_find(tree, q="sub"))) == {
            f"{ROOT}/Sub/a.mkv",
            f"{ROOT}/Sub2/b.mkv",
            "/media/elsewhere/Sub/d.mkv",
        }
        assert _paths(_find(tree, q="SUB2")) == [f"{ROOT}/Sub2/b.mkv"]
        # % and _ are letters here, not wildcards
        assert _paths(_find(tree, q="50%_")) == [f"{ROOT}/50%_off/c.mkv"]
        assert _find(tree, q="b%c")["total"] == 0
        assert _paths(_find(tree, q="*/c.mkv")) == [f"{ROOT}/50%_off/c.mkv"]

    def test_file_names_as_the_searches_hand_them_on(self, tree):
        got = {r["abs_path"]: (r["file_dir"], r["filename"]) for r in _find(tree)["results"]}
        assert got[f"{ROOT}/Sub/a.mkv"] == (ROOT, "Sub/a.mkv")
        assert got["/media/elsewhere/Sub/d.mkv"] == ("/media/elsewhere/Sub", "d.mkv")
        assert all(fd.rstrip("/") + "/" + fn == p for p, (fd, fn) in got.items())

    def test_the_stored_root_only_where_the_display_path_misses_the_file(self, db):
        """What the searches hand on, from the tables: the stored root
        (transcripts.media_dir) counts only where the display path doesn't
        lead to the file — it is read from there only then."""
        # The display path leads to the file: its root, whatever is stored
        _row(db, f"{ROOT}/a/fits.mkv", rel_path="a/fits.mkv", media_dir="/stale/root")
        # One of an older root: the stored root it lies below
        _row(db, f"{ROOT}/a/older.mkv", rel_path="x/older.mkv", media_dir=ROOT)
        # Neither leads to it: the file's own folder
        _row(db, f"{ROOT}/a/lost.mkv", rel_path="y/lost.mkv", media_dir="/other")
        got = {
            r["abs_path"].rsplit("/", 1)[1]: (r["file_dir"], r["filename"])
            for r in _find(db)["results"]
        }
        assert got == {
            "fits.mkv": (ROOT, "a/fits.mkv"),
            "older.mkv": (ROOT, "a/older.mkv"),
            "lost.mkv": (f"{ROOT}/a", "lost.mkv"),
        }
        for name, place in got.items():
            one = query.file_stats(db, f"{ROOT}/a/{name}")
            assert (one["file_dir"], one["filename"]) == place, name

    def test_nothing_behind_the_transcript_is_read_for_every_row(self, db):
        """json_data holds the whole transcript, megabytes for a long one:
        a column behind it costs a walk through its overflow pages for each
        row read. The lists read none — the stored root only in the branch
        for a display path that misses the file."""
        with db.connection() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(transcripts)")]
        behind = cols[cols.index("json_data") + 1 :]
        assert "media_dir" in behind
        for sql in (
            query._REC_SQL.replace(query._MEDIA_DIR_SQL, ""),
            query._ROW_SQL,
            query._BASE_SQL,
        ):
            assert [c for c in behind if re.search(rf"\bt\.{c}\b", sql)] == []
        assert query._MEDIA_DIR_SQL in query._REC_SQL
        assert re.search(r"ELSE t\.media_dir END$", query._MEDIA_DIR_SQL)

    def test_a_stale_display_path_still_leads_to_the_file(self):
        assert query.place(f"{ROOT}/a/b.mkv", "a/b.mkv", ROOT) == (ROOT, "a/b.mkv")
        # A display path relative to an older, deeper root
        assert query.place(f"{ROOT}/a/b.mkv", "b.mkv", ROOT) == (f"{ROOT}/a", "b.mkv")
        # One that fits nothing: the file's own folder
        assert query.place(f"{ROOT}/a/b.mkv", "x/y.mkv", "/other") == (f"{ROOT}/a", "b.mkv")


# ── Held result lists ──────────────────────────────────────────────────────


class TestHeld:
    def test_a_write_shows_at_once(self, db):
        _write(db, f"{ROOT}/one.mkv", _blocks([{A: 60, B: 60}]))
        assert _find(db)["total"] == 1
        _write(db, f"{ROOT}/two.mkv", _blocks([{A: 70, B: 70}]))
        assert _find(db)["total"] == 2

    def test_paging_counts_once(self, db, monkeypatch):
        for i in range(5):
            _write(db, f"{ROOT}/c{i}.mkv", _blocks([{A: 60 + i, B: 60}]))
        calls = []
        real = query._whole_rows
        monkeypatch.setattr(query, "_whole_rows", lambda *a: calls.append(1) or real(*a))
        f, _ = query.parse_filters({})
        first = query.find(db, f, 0, 2)
        second = query.find(db, f, 2, 2)
        assert len(calls) == 1
        assert first["total"] == second["total"] == 5
        assert first["truncated"] and second["truncated"]
        assert query.find(db, f, 4, 2)["truncated"] is False
        assert len({r["abs_path"] for r in first["results"] + second["results"]}) == 4

    def test_a_new_name_shows_at_once(self, db):
        segs = _blocks([{A: 60, B: 60}])
        _write(db, f"{ROOT}/ab.mkv", segs, names={A: "Alice"})
        assert _find(db, people="Bob")["people_unknown"] == ["Bob"]
        _write(db, f"{ROOT}/ab.mkv", segs, names={A: "Alice", B: "Bob"})
        assert _find(db, people="Bob")["total"] == 1

    def test_at_most_so_many_lists(self, db, monkeypatch):
        monkeypatch.setattr(query, "RESULTS_MAX", 3)
        for n in range(6):
            _find(db, min_sec=str(n + 1))
        assert len(query._cache) == 3

    def test_a_held_list_keeps_a_number_per_conversation(self, db):
        """Up to RESULTS_MAX lists are held, each maybe of every transcript
        of the library: the order as transcript ids, not a row object (with
        its path and numbers) per conversation."""
        for i in range(3):
            _write(db, f"{ROOT}/c{i}.mkv", _blocks([{A: 60 + i, B: 60}]))
        _find(db)
        [entry] = query._cache.values()
        held = entry["value"]
        assert isinstance(held["tids"], array) and len(held["tids"]) == 3
        assert held["copies"] == {}
        assert not any(isinstance(v, query._Row) for v in held.values())

    def test_a_recording_deleted_since_is_left_out_of_its_page(self, db):
        for i in range(3):
            _write(db, f"{ROOT}/c{i}.mkv", _blocks([{A: 60 + 10 * i, B: 60}]))
        f, _ = query.parse_filters({})
        assert query.find(db, f, 0, 1)["total"] == 3  # only the first page worked out
        assert db.delete_transcript("c1.mkv", ROOT)
        assert [r["abs_path"] for r in query.find(db, f, 0, 3)["results"]] == [
            f"{ROOT}/c2.mkv",
            f"{ROOT}/c0.mkv",
        ]


# ── One recording ──────────────────────────────────────────────────────────


class TestOneRecording:
    def test_how_they_talk_and_who_talked_when(self, db):
        _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        out = query.file_stats(db, f"{ROOT}/stream.mkv")
        assert (out["counted"], out["stale"], out["kind"]) == (True, False, "detailed")
        assert (out["file_dir"], out["filename"]) == (ROOT, "stream.mkv")
        s = out["summary"]
        assert (s["duration_sec"], s["speech_sec"], s["words"], s["sentences"]) == (
            3400.0,
            2250.0,
            2700,
            225,
        )
        assert (s["turns"], s["speakers"], s["labels"], s["top_share"]) == (15, 2, 3, 0.644)
        assert (s["wps"], s["turn_rate"], s["mean_turn_sec"], s["wps_pct"]) == (
            12.0,
            0.4,
            150.0,
            80,
        )
        assert s["longest_turn_name"] == "Alice" and s["punctuated"] is True
        alice = out["speakers"][0]
        assert {
            k: alice[k] for k in ("label", "name", "talk_sec", "words", "sentences", "wps", "turns")
        } == {
            "label": A,
            "name": "Alice",
            "talk_sec": 1450.0,
            "words": 1740,
            "sentences": 145,
            "wps": 12.0,
            "turns": 7,
        }
        assert [(p["name"], p["labels"], p["talk_sec"]) for p in out["people"]] == [
            ("Alice", [A], 1450.0),
            ("Bob", [B], 700.0),
            ("Carol", [C], 100.0),
        ]
        # Window 8 (Alice alone) lies between two of Alice and Bob: theirs
        assert out["parts"] == [
            {"start_sec": 0.0, "end_sec": 900.0, "people": ["Alice"], "unnamed": 0},
            {"start_sec": 900.0, "end_sec": 3300.0, "people": ["Alice", "Bob"], "unnamed": 0},
            {"start_sec": 3300.0, "end_sec": 3400.0, "people": ["Carol"], "unnamed": 0},
        ]

    def test_an_unnamed_voice_is_counted_in_its_part(self, db):
        w = [{A: 100, D: 100}] * 3 + [{A: 200}] * 2
        _write(db, f"{ROOT}/call.mkv", _blocks(w), names={A: "Alice"})
        out = query.file_stats(db, f"{ROOT}/call.mkv")
        assert out["unnamed"] == 1
        assert [(s["label"], s["name"], s["active"]) for s in out["speakers"]] == [
            (A, "Alice", True),
            (D, None, True),
        ]
        parts = out["parts"]
        assert parts == [
            {"start_sec": 0.0, "end_sec": 900.0, "people": ["Alice"], "unnamed": 1},
            {"start_sec": 900.0, "end_sec": 1400.0, "people": ["Alice"], "unnamed": 0},
        ]

    def test_at_most_forty_parts(self, db):
        w = []
        for i in range(60):  # two windows each: never a blip
            w += [{A: 100}] * 2 if i % 2 else [{B: 100}] * 2
        _write(db, f"{ROOT}/long.mkv", _blocks(w), names=NAMES)
        parts = query.file_stats(db, f"{ROOT}/long.mkv")["parts"]
        assert 30 <= len(parts) <= query.PARTS_MAX
        assert parts[0]["start_sec"] == 0.0 and parts[-1]["end_sec"] == 35800.0
        assert all(p["end_sec"] == q["start_sec"] for p, q in zip(parts, parts[1:]))

    def test_short_recordings_have_no_parts(self, db):
        _write(db, f"{ROOT}/call.mkv", _blocks([{A: 60, B: 60}]), names=NAMES)
        out = query.file_stats(db, f"{ROOT}/call.mkv")
        assert out["parts"] == [] and out["counted"] is True

    def test_not_counted_yet_and_no_transcript(self, db, monkeypatch):
        monkeypatch.setattr(store, "enabled", lambda: False)
        _write(db, f"{ROOT}/new.mkv", _blocks([{A: 60, B: 60}]))
        out = query.file_stats(db, f"{ROOT}/new.mkv")
        assert (out["counted"], out["summary"], out["speakers"], out["kind"]) == (
            False,
            {},
            [],
            None,
        )
        assert out["baseline"]["source"] == "defaults"
        assert query.file_stats(db, f"{ROOT}/missing.mkv") is None


# ── A person, and everyone ─────────────────────────────────────────────────


class TestPerson:
    @pytest.fixture
    def library(self, db):
        _write(db, f"{ROOT}/stream.mkv", _hour(), names=NAMES)
        call = _blocks([{A: 100, D: 200}])
        _write(db, f"{ROOT}/call.mkv", call, names={A: "Alice", D: "Dave"})
        _write(db, f"{ROOT}/backup/call.mkv", call, names={A: "Alice", D: "Dave"})
        _write(db, f"{ROOT}/note.mkv", _blocks([{A: 30}]), names={A: "Alice"})
        return db

    def test_how_a_person_talks_and_with_whom(self, library):
        p = query.person(library, "alice")
        assert (p["name"], p["known"]) == ("Alice", True)
        # The stream, the call (its copy folded) and the note
        assert (p["recordings"], p["conversations"]) == (3, 2)
        assert p["talk_sec"] == 1450.0 + 100.0 + 30.0
        assert p["wps"] == 12.0 and p["questions_share"] == 0.0
        assert p["partners"] == [
            {"name": "Bob", "recordings": 1, "together_sec": 2250.0},
            {"name": "Dave", "recordings": 1, "together_sec": 300.0},
        ]
        assert p["kinds"] == {"detailed": 2, "monologue": 1}
        assert [r["abs_path"] for r in p["longest"]] == [
            f"{ROOT}/stream.mkv",
            f"{ROOT}/call.mkv",
            f"{ROOT}/note.mkv",
        ]

    def test_a_name_nobody_has(self, library):
        p = query.person(library, "Grace")
        assert (p["known"], p["recordings"], p["partners"], p["longest"]) == (False, 0, [], [])

    def test_everyone_who_takes_part(self, library):
        # Carol's 4.4 % in the stream is no part in it
        assert query.people_list(library)["people"] == [
            {"name": "Alice", "recordings": 3, "conversations": 2},
            {"name": "Bob", "recordings": 1, "conversations": 1},
            {"name": "Dave", "recordings": 1, "conversations": 1},
        ]
        assert {"name": "Carol", "recordings": 1, "conversations": 1} in query.people_list(
            library, min_share=0.04
        )["people"]


# ── An empty library ───────────────────────────────────────────────────────


def test_an_empty_library_answers_in_the_same_shape(db):
    out = _find(db, people="Alice")
    assert (out["total"], out["results"], out["people_unknown"]) == (0, [], ["Alice"])
    assert out["baseline"]["source"] == "defaults" and out["baseline"]["language"] is None
    assert out["status"]["counted"] == 0 and out["status"]["stale"] == 0
    assert query.people_list(db) == {"people": []}

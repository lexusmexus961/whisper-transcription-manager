"""Renewing a recording's voices — the decisions go to the same voices.

Everything that has no speech spans yet can be queued again with voices
only. A voices-only run makes them anew, with spans — and the names, the
trash and the history of the old ones must reach the new ones.

What these tests pin:

* names and the trash follow the VOICE, window by window, not the label:
  the new run numbers its speakers after the transcript, the old voices
  after whichever diarization made them — and it groups differently, so a
  new label can hold windows of several old voices;
* the times stay (named at, discarded at), the spans arrive;
* the history follows: an old decision's undo and redo act on the new
  windows, its counts and samples are computed anew;
* when any decided old voice has no clear counterpart, the recording stays
  exactly as it was — no guess puts a name on someone else;
* old voices without a decision need no counterpart;
* a voice the new run split in two keeps its name in both halves; two old
  labels of one person merged by the new run hand on both histories;
* a click on a voice renewed since is skipped, not an error;
* the queue entry and carry_along route a renewal through all this.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import queue as q
from whisper_tm.core import speaker as sp
from whisper_tm.database.speaker import CARRY_AT_LEAST, CARRY_MARGIN, SpeakerDB, match_voices

DIM = 64
PATH = "/media/meeting_2024-03-01.mkv"
MODEL = sp.MODEL_BRAND


def _vec(seed, basis=None, spread=0.03):
    r = np.random.default_rng(seed)
    v = basis if basis is not None else r.normal(size=DIM)
    v = v + spread * r.normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


ALICE, BOB, TV, CAROL, NEW = (_vec(i, spread=0) for i in (1, 2, 3, 4, 5))


def _lines(spec, seed, spans=True):
    """``spec``: ``[(label, voice, (start, …))]`` → the rows a run stores."""
    out, k = [], 0
    for label, voice, starts in spec:
        for i, start in enumerate(starts):
            out.append(
                {
                    "label": label,
                    "windows": i + 1,
                    "start_sec": float(start),
                    "speech_sec": 30.0,
                    "segments": 3,
                    "parts": [(start, start + 30.0)] if spans else None,
                    "v_separated": _vec(seed + k, voice),
                }
            )
            k += 1
    return out


def _numbered(lines):
    """Number the windows per label, as a run does — for specs that name a
    label twice."""
    seen: dict[str, int] = {}
    for z in lines:
        seen[z["label"]] = z["windows"] = seen.get(z["label"], 0) + 1
    return lines


def _ids(db, label=None, path=PATH):
    return [w["id"] for w in db.windows(path) if label is None or w["label"] == label]


OLD = [
    ("SPEAKER_00", ALICE, (0, 100, 200)),
    ("SPEAKER_01", BOB, (1000, 1100)),
    ("SPEAKER_02", TV, (2000,)),
    ("SPEAKER_03", CAROL, (3000,)),
]
#: The new run: Alice and Bob numbered the other way round, Carol not found
#: (she carried no decision), someone new
RENEWED = [
    ("SPEAKER_00", BOB, (1000, 1100, 1200)),
    ("SPEAKER_01", ALICE, (0, 100)),
    ("SPEAKER_02", TV, (2000, 2100)),
    ("SPEAKER_04", NEW, (4000,)),
]


@pytest.fixture
def db(tmp_path):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    yield d
    d.close()


@pytest.fixture
def decided(db):
    """An old recording: Alice and Bob named, the TV discarded, Carol left."""
    db.write(PATH, MODEL, _lines(OLD, 100, spans=False))
    db.assign_many(_ids(db, "SPEAKER_00"), "Alice")
    name_alice = db.last_decision_id()
    db.assign_many(_ids(db, "SPEAKER_01"), "Bob")
    db.move_to_trash(_ids(db, "SPEAKER_02"), reason="tv")
    trash_tv = db.last_decision_id()
    with db.connection() as c:
        c.execute("UPDATE speaker_assignment SET set_at = '2025-01-02 03:04:05'")
        c.execute("UPDATE speaker_trash SET discarded_at = '2025-02-03 04:05:06'")
    return {"name_alice": name_alice, "trash_tv": trash_tv}


def _state(db, path=PATH):
    """Everything about a recording's voices, to compare before and after."""
    with db.connection() as c:
        return {
            "windows": [
                tuple(r)
                for r in c.execute(
                    "SELECT id, label, start_sec, parts FROM speaker_vectors "
                    "WHERE abs_path = ? ORDER BY id",
                    (path,),
                )
            ],
            "names": [
                tuple(r)
                for r in c.execute(
                    "SELECT z.vector_id, p.name, z.source, z.set_at FROM speaker_assignment z "
                    "JOIN speaker_people p ON p.id = z.person_id ORDER BY z.vector_id"
                )
            ],
            "trash": [
                tuple(r)
                for r in c.execute(
                    "SELECT vector_id, reason, discarded_at FROM speaker_trash ORDER BY vector_id"
                )
            ],
            "journal": [
                tuple(r)
                for r in c.execute(
                    "SELECT * FROM speaker_decision_voices ORDER BY decision_id, vector_id"
                )
            ],
        }


def _named(db, label):
    names = db.names_per_vector()
    return {names.get(i) for i in _ids(db, label)}


class TestByVoice:
    def test_names_and_trash_follow_the_voice_not_the_label(self, db, decided):
        r = db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        assert r["replaced"] and not r["unplaced"]
        assert _named(db, "SPEAKER_01") == {"Alice"}  # the voice, not the old label
        assert _named(db, "SPEAKER_00") == {"Bob"}
        assert _named(db, "SPEAKER_04") == {None}
        tv = _ids(db, "SPEAKER_02")
        assert len(tv) == 2 and set(tv) <= db.trash_ids()
        assert {c["new"]: c["name"] for c in r["carried"]} == {
            "SPEAKER_00": "Bob",
            "SPEAKER_01": "Alice",
            "SPEAKER_02": "trash",
        }
        assert (r["names"], r["trash"]) == (5, 2)
        # How long the write lock was held, for the log — a duration, not a time
        assert 0 <= r["secs"] < 60

    def test_the_old_windows_go_the_spans_arrive_the_times_stay(self, db, decided):
        old = _ids(db)
        db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        ws = db.windows(PATH)
        assert len(ws) == 3 + 2 + 2 + 1 and all(w["parts"] for w in ws)
        # No old id comes back as another voice — a page may still show it
        assert min(w["id"] for w in ws) > max(old)
        assert {w["model"] for w in ws} == {MODEL}
        with db.connection() as c:
            assert {r[0] for r in c.execute("SELECT set_at FROM speaker_assignment")} == {
                "2025-01-02 03:04:05"
            }
            assert {
                tuple(r) for r in c.execute("SELECT reason, discarded_at FROM speaker_trash")
            } == {("tv", "2025-02-03 04:05:06")}

    def test_old_voices_without_a_decision_need_no_counterpart(self, db):
        db.write(PATH, MODEL, _lines(OLD, 100, spans=False))
        r = db.replace_keeping_decisions(PATH, MODEL, _lines([("SPEAKER_09", NEW, (5,))], 7))
        assert r["replaced"] and [w["label"] for w in db.windows(PATH)] == ["SPEAKER_09"]

    def test_a_recording_without_voices_simply_gets_them(self, db):
        r = db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        assert r["replaced"] and len(db.windows(PATH)) == 8 and not r["carried"]

    def test_a_voice_split_in_two_keeps_its_name_in_both(self, db, decided):
        split = RENEWED + [("SPEAKER_05", ALICE, (300, 400))]
        db.replace_keeping_decisions(PATH, MODEL, _lines(split, 900))
        assert _named(db, "SPEAKER_01") == _named(db, "SPEAKER_05") == {"Alice"}

    def test_a_new_label_holding_two_people(self, db, decided):
        """A long recording: the new run can put windows of Bob and
        of Alice under one label — each window keeps its own person."""
        joined = [
            ("SPEAKER_00", BOB, (1000, 1100)),
            ("SPEAKER_01", ALICE, (0,)),
            ("SPEAKER_02", TV, (2000,)),
            ("SPEAKER_09", ALICE, (150, 250)),
            ("SPEAKER_09", BOB, (1150,)),
        ]
        r = db.replace_keeping_decisions(PATH, MODEL, _numbered(_lines(joined, 900)))
        assert r["replaced"]
        names = db.names_per_vector()
        got = sorted(
            (w["start_sec"], names.get(w["id"]))
            for w in db.windows(PATH)
            if w["label"] == "SPEAKER_09"
        )
        assert got == [(150.0, "Alice"), (250.0, "Alice"), (1150.0, "Bob")]
        assert {(c["new"], c["name"], c["windows"]) for c in r["carried"]} >= {
            ("SPEAKER_09", "Alice", 2),
            ("SPEAKER_09", "Bob", 1),
        }

    def test_one_window_of_a_label_was_someone_else(self, db):
        """A label named Alice, one window of it unnamed because it is Bob:
        the new Alice windows are Alice, the new Bob window stays unnamed —
        and keeps the history of that click."""
        db.write(
            PATH,
            MODEL,
            _numbered(
                _lines(
                    [("SPEAKER_00", ALICE, (0, 100)), ("SPEAKER_00", BOB, (200,))], 100, spans=False
                )
            ),
        )
        ids = _ids(db)
        db.assign_many(ids, "Alice")
        bob = [w["id"] for w in db.windows(PATH) if w["start_sec"] == 200.0]
        db.write_names([(bob[0], None, None)], kind="unname", origin="t")
        unname = db.last_decision_id()
        r = db.replace_keeping_decisions(
            PATH, MODEL, _lines([("SPEAKER_00", ALICE, (0, 100)), ("SPEAKER_03", BOB, (210,))], 9)
        )
        assert r["replaced"]
        assert _named(db, "SPEAKER_00") == {"Alice"} and _named(db, "SPEAKER_03") == {None}
        with db.connection() as c:
            got = [
                r[0]
                for r in c.execute(
                    "SELECT v.label FROM speaker_decision_voices d JOIN speaker_vectors v "
                    "ON v.id = d.vector_id WHERE d.decision_id = ?",
                    (unname,),
                )
            ]
        assert got == ["SPEAKER_03"]

    def test_one_person_under_two_old_labels_merges(self, db):
        db.write(
            PATH,
            MODEL,
            _lines(
                [("SPEAKER_00", ALICE, (0, 100)), ("SPEAKER_07", ALICE, (500,))], 100, spans=False
            ),
        )
        db.assign_many(_ids(db, "SPEAKER_00"), "Alice")
        first = db.last_decision_id()
        db.assign_many(_ids(db, "SPEAKER_07"), "Alice", source="assignment-spread")
        second = db.last_decision_id()
        r = db.replace_keeping_decisions(PATH, MODEL, _lines([("SPEAKER_00", ALICE, (0, 400))], 9))
        assert r["replaced"] and _named(db, "SPEAKER_00") == {"Alice"}
        assert sorted(r["carried"][0]["from"]) == ["SPEAKER_00", "SPEAKER_07"]
        with db.connection() as c:
            followed = {
                d for (d,) in c.execute("SELECT DISTINCT decision_id FROM speaker_decision_voices")
            }
        assert {first, second} <= followed


class TestTheHistoryFollows:
    def test_undo_and_redo_act_on_the_new_windows(self, db, decided):
        db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        alice = _ids(db, "SPEAKER_01")
        db.undo(decided["name_alice"])
        assert _named(db, "SPEAKER_01") == {None}
        assert _named(db, "SPEAKER_00") == {"Bob"}  # untouched
        db.redo(decided["name_alice"])
        assert all(db.names_per_vector()[v] == "Alice" for v in alice)
        db.undo(decided["trash_tv"])
        assert not set(_ids(db, "SPEAKER_02")) & db.trash_ids()

    def test_counts_and_samples_are_computed_anew(self, db, decided):
        r = db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        assert r["history_entries"] == 3
        entries = {e["id"]: e for e in db.decisions(limit=10)[0]}
        assert entries[decided["name_alice"]]["voices"] == 2
        assert entries[decided["trash_tv"]]["voices"] == 2
        here = set(_ids(db))
        for e in entries.values():
            assert all(s["vector_id"] in here for s in e["summary"]["samples"])

    def test_every_old_window_hands_its_entries_on(self, db):
        """Window by window in time: an entry that touched only a late
        window of a voice reaches the new window nearest to it."""
        db.write(PATH, MODEL, _lines([("SPEAKER_00", ALICE, (0, 1000, 2000))], 100, spans=False))
        ids = _ids(db)
        db.assign_many(ids, "Alice")
        db.move_to_trash(ids[2:], reason="x")
        db.restore_from_trash(ids[2:])
        late = db.last_decision_id()
        db.replace_keeping_decisions(PATH, MODEL, _lines([("SPEAKER_00", ALICE, (10, 1990))], 9))
        with db.connection() as c:
            got = [
                r[0]
                for r in c.execute(
                    "SELECT v.start_sec FROM speaker_decision_voices d "
                    "JOIN speaker_vectors v ON v.id = d.vector_id WHERE d.decision_id = ?",
                    (late,),
                )
            ]
        assert got == [1990.0]


class TestUnclearKeepsTheRecording:
    def test_a_decided_voice_the_new_run_lacks(self, db, decided):
        before = _state(db)
        without_alice = [s for s in RENEWED if s[1] is not ALICE]
        r = db.replace_keeping_decisions(PATH, MODEL, _lines(without_alice, 900))
        assert not r["replaced"]
        assert [u["old"] for u in r["unplaced"]] == ["SPEAKER_00"]
        assert r["unplaced"][0]["name"] == "Alice"
        assert _state(db) == before

    def test_only_one_half_of_a_voice_was_named(self, db):
        """A short voice message: the old diarization cut one voice
        in two, only one half was named — the new run hears one voice."""
        db.write(
            PATH,
            MODEL,
            _lines([("SPEAKER_00", ALICE, (0,)), ("SPEAKER_01", ALICE, (40,))], 100, spans=False),
        )
        db.assign_many(_ids(db, "SPEAKER_00"), "Alice")
        r = db.replace_keeping_decisions(PATH, MODEL, _lines([("SPEAKER_00", ALICE, (0,))], 9))
        assert r["replaced"] and _named(db, "SPEAKER_00") == {"Alice"}

    def test_two_people_too_alike_to_tell(self, db):
        bob_like_alice = _vec(77, ALICE, 0.04)
        db.write(
            PATH,
            MODEL,
            _lines(
                [("SPEAKER_00", ALICE, (0,)), ("SPEAKER_01", bob_like_alice, (500,))],
                100,
                spans=False,
            ),
        )
        db.assign_many(_ids(db, "SPEAKER_00"), "Alice")
        db.assign_many(_ids(db, "SPEAKER_01"), "Bob")
        before = _state(db)
        r = db.replace_keeping_decisions(PATH, MODEL, _lines([("SPEAKER_00", ALICE, (0,))], 9))
        assert not r["replaced"] and _state(db) == before

    def test_one_voice_named_and_unnamed(self, db, decided):
        """The same voice, one window of it unnamed by hand: two decisions
        on one voice — which one a new window of it should carry stays open."""
        db.write_names([(_ids(db, "SPEAKER_00")[0], None, None)], kind="unname", origin="t")
        before = _state(db)
        r = db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        assert not r["replaced"]
        assert {(u["old"], u["name"]) for u in r["unplaced"]} == {
            ("SPEAKER_00", "Alice"),
            ("SPEAKER_00", "history"),
        }
        assert _state(db) == before


def _pair(cos):
    """Two unit vectors with this cosine, and the one right between them."""
    a = np.zeros(DIM, np.float32)
    a[0] = 1.0
    b = np.zeros(DIM, np.float32)
    b[0], b[1] = cos, np.sqrt(1 - cos * cos)
    mid = (a + b) / np.linalg.norm(a + b)
    return a, b, mid.astype(np.float32)


class TestMatching:
    def test_the_operating_point(self):
        assert CARRY_AT_LEAST >= 0.75 and CARRY_MARGIN >= 0.05

    A, B = ("SPEAKER_00", 1, False), ("SPEAKER_01", None, False)

    def test_too_far_takes_nothing(self):
        far = _vec(9, spread=0)
        m = match_voices({self.A: ALICE}, [far], {self.A})
        assert m["takes"] == {} and m["best"][0][0] == self.A

    def test_without_vectors_nothing_is_matched(self):
        m = match_voices({self.A: ALICE}, [None], {self.A})
        assert m == {"takes": {}, "best": {}, "nearest": {}}

    def test_someone_else_just_as_alike_is_a_rival(self):
        """The new window lies between a named voice and another person
        (0.92 to both, the two only 0.70 alike): which one it is stays open
        — even though the other one was never named."""
        a, b, mid = _pair(0.70)
        assert match_voices({self.A: a, self.B: b}, [mid], {self.A})["takes"] == {}

    def test_the_unnamed_half_of_one_person_is_no_rival(self):
        a, b, mid = _pair(0.95)
        assert match_voices({self.A: a, self.B: b}, [mid], {self.A})["takes"] == {0: [self.A]}

    def test_an_undecided_best_hands_over_to_the_decided_half(self):
        a, b, _mid = _pair(0.95)
        assert match_voices({self.A: a, self.B: b}, [b], {self.A})["takes"] == {0: [self.A]}

    def test_each_window_for_itself(self):
        m = match_voices(
            {self.A: ALICE, ("SPEAKER_02", 2, False): BOB},
            [ALICE, BOB, NEW],
            {self.A, ("SPEAKER_02", 2, False)},
        )
        assert m["takes"] == {0: [self.A], 1: [("SPEAKER_02", 2, False)]}


class TestAroundIt:
    def test_a_click_on_a_renewed_voice_is_skipped(self, db, decided):
        gone = _ids(db, "SPEAKER_03")[0]
        db.replace_keeping_decisions(PATH, MODEL, _lines(RENEWED, 900))
        n = len(db.decisions(limit=50)[0])
        assert db.write_names([(gone, "Carol", "human")], kind="name", origin="t") == []
        assert db.write_trash([gone], True, reason="x") == []
        assert len(db.decisions(limit=50)[0]) == n  # no empty entry

    def test_the_write_lock_is_taken_before_the_first_read(self, db):
        with db.connection(immediate=True) as c:
            assert c.in_transaction

    def test_carry_along_renews_through_it(self, monkeypatch, db):
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
        db.write("/m/a.mkv", MODEL, _lines([("SPEAKER_03", ALICE, (0,))], 1, spans=False))
        db.assign_many(_ids(db, path="/m/a.mkv"), "Alice")
        monkeypatch.setattr(
            sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(3, ALICE)
        )
        audio = np.zeros(60 * sp.SR, dtype=np.float32)
        b = sp.carry_along(
            "/m/a.mkv",
            audio,
            [{"start": 0.0, "end": 60.0, "speaker": "SPEAKER_00"}],
            device="cpu",
            separated=True,
            keep_decisions=True,
        )
        assert b["renewal"]["replaced"] and b["windows"] == 2
        assert {w["label"] for w in db.windows("/m/a.mkv")} == {"SPEAKER_00"}
        assert set(db.names_per_vector().values()) == {"Alice"}

    def test_carry_along_says_when_it_kept_the_voices(self, monkeypatch, db):
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
        db.write("/m/a.mkv", MODEL, _lines([("SPEAKER_03", ALICE, (0,))], 1, spans=False))
        db.assign_many(_ids(db, path="/m/a.mkv"), "Alice")
        monkeypatch.setattr(
            sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(3, BOB)
        )
        audio = np.zeros(60 * sp.SR, dtype=np.float32)
        b = sp.carry_along(
            "/m/a.mkv",
            audio,
            [{"start": 0.0, "end": 60.0, "speaker": "SPEAKER_00"}],
            device="cpu",
            separated=True,
            keep_decisions=True,
        )
        assert b["windows"] == 0 and not b["renewal"]["replaced"] and "keeps" in b["reason"]
        assert {w["label"] for w in db.windows("/m/a.mkv")} == {"SPEAKER_03"}


class TestTheQueue:
    @pytest.fixture
    def cfg(self):
        from whisper_tm.config import config as c

        if not c.is_loaded:
            c.load()
        return c

    def test_the_setting(self, cfg):
        from whisper_tm.api.queue import _LIBRARY_JOBS, _SETTING_KEYS

        assert "refresh" in _SETTING_KEYS
        assert q._normalize({"job": "voices", "refresh": True})["refresh_voices"] is True
        assert q._normalize({"job": "voices"})["refresh_voices"] is False
        assert "refresh_voices" not in q._normalize({"refresh": True})
        assert _LIBRARY_JOBS["voices_refresh"] == {"job": "voices", "refresh": True}

    def test_a_renewal_is_not_skipped_for_its_voices(self, monkeypatch, cfg):
        from whisper_tm.database import speaker as speaker_db_module

        finished, logged = [], []
        monkeypatch.setattr(q, "_db", lambda: None)
        monkeypatch.setattr(q, "_jm", lambda: None)
        monkeypatch.setattr(q, "_finish", lambda item_id, status, *a: finished.append(status))
        monkeypatch.setattr(q, "qlog", lambda msg, *a, **k: logged.append(msg))
        monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)

        class _HasVoices:
            def has_vectors(self, *a, **k):
                return True

        monkeypatch.setattr(speaker_db_module, "speaker_db", lambda: _HasVoices())
        q._process(
            {
                "id": 1,
                "filename": "a.mp4",
                "dir": "/media",
                "settings": {"job": "voices", "refresh": True},
            }
        )
        # Past the "voices present" check; it still needs a transcript
        assert finished == ["skipped"] and "no transcript" in logged[-1]

    def test_the_worker_hands_it_on(self):
        import inspect

        from whisper_tm.core.transcription import transcription_worker

        assert inspect.signature(transcription_worker).parameters["refresh_voices"].default is False
        assert "keep_decisions=bool(voices_only and refresh_voices)" in inspect.getsource(
            transcription_worker
        )

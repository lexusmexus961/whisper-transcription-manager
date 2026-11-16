"""A voices-only run writes its speakers into the transcript — and only those.

The speakers come from the map, at its threshold: a recording transcribed
without telling speakers apart can get several voices from its voices run —
on the map, in "Who was there?" — while its transcript says "—" in every line.

What these tests pin:

* every sentence gets the label of the diarization turn it overlaps most —
  the labels the voices carry; between turns the nearest within a second;
* a name from the map wins; else the recognition's, if the windows that
  reach the threshold mostly agree; a doubtful speaker stays unnamed;
* a transcript that has speakers keeps them, a name someone gave is never
  replaced, the text and the times never change;
* JSON and database get the same, and nothing is written when nothing
  changed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from whisper_tm.core import speaker as sp
from whisper_tm.core import transcription as tr
from whisper_tm.database import Database


def _turns(*t):
    return [{"speaker": s, "start": a, "end": b} for s, a, b in t]


class TestLabelSegments:
    def test_the_most_overlap_wins(self):
        turns = _turns(("SPEAKER_00", 0, 5), ("SPEAKER_01", 5, 12))
        assert sp.label_segments([(0, 4), (4, 9), (11, 12)], turns) == [
            "SPEAKER_00",
            "SPEAKER_01",
            "SPEAKER_01",
        ]

    def test_between_turns_the_nearest_within_a_second(self):
        turns = _turns(("SPEAKER_00", 0, 5), ("SPEAKER_01", 10, 12))
        assert sp.label_segments([(5.4, 5.8), (9.5, 9.8), (7, 8)], turns) == [
            "SPEAKER_00",
            "SPEAKER_01",
            "—",
        ]

    def test_no_turns_no_speakers(self):
        assert sp.label_segments([(0, 1)], []) == ["—"]


class TestNamesForLabels:
    """The map first, then the recognition at the threshold."""

    @pytest.fixture
    def library(self, monkeypatch):
        alice, bob = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        x = np.array(
            [[1.0, 0, 0], [0.99, 0.14, 0], [0, 1.0, 0], [0.6, 0.8, 0], [0.97, 0.24, 0], [0, 0, 1.0]]
        )
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        windows = [
            {"id": 1, "label": "SPEAKER_00", "person": "Alice"},  # named on the map
            {"id": 2, "label": "SPEAKER_00", "person": None},
            {"id": 3, "label": "SPEAKER_01", "person": None},  # like Bob
            {"id": 4, "label": "SPEAKER_02", "person": None},  # 0.8 Bob, 0.6 Alice
            {"id": 5, "label": "SPEAKER_02", "person": None},  # like Alice
            {"id": 6, "label": "SPEAKER_03", "person": None},  # like no one
        ]

        class _Db:
            def windows(self, abs_path):
                return windows

        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: _Db())
        monkeypatch.setattr(
            "whisper_tm.core.speaker_inventory.inventory",
            lambda: {
                "gallery": {"Alice": alice[None, :], "Bob": bob[None, :]},
                "X": x,
                "id_to_row": {i + 1: i for i in range(6)},
            },
        )
        return windows

    def test_the_map_names_and_the_recognition_fills_in(self, library):
        got = sp.names_for_labels("/m/a.mp4", threshold=0.75)
        assert got["SPEAKER_00"]["name"] == "Alice" and got["SPEAKER_00"]["source"] == "map"
        assert got["SPEAKER_01"] == {"name": "Bob", "source": "recognition", "windows": 1, "of": 1}

    def test_a_doubtful_speaker_stays_unnamed(self, library):
        got = sp.names_for_labels("/m/a.mp4", threshold=0.75)
        # SPEAKER_02: one window for Bob (0.8), one for Alice (0.97) — no two thirds
        assert "SPEAKER_02" not in got and "SPEAKER_03" not in got

    def test_below_the_threshold_no_name(self, library):
        got = sp.names_for_labels("/m/a.mp4", threshold=0.999)
        assert set(got) == {"SPEAKER_00", "SPEAKER_01"}  # exact matches only
        got = sp.names_for_labels("/m/a.mp4", threshold=1.01)
        assert set(got) == {"SPEAKER_00"}  # the map's alone


class TestIntoTheTranscript:
    SEGMENTS = [
        {"start": 0.0, "end": 4.0, "text": "Hello there.", "speaker": "—"},
        {"start": 5.0, "end": 9.0, "text": "Hi, how are you?", "speaker": "—"},
        {"start": 30.0, "end": 31.0, "text": "(music)", "speaker": "—"},
    ]

    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        db = Database(tmp_path / "wx.db")
        db.initialize()
        tp = tmp_path / ".transcripts" / "a.mp4.json"
        tp.parent.mkdir()

        def put(segments, names=None):
            data = {
                "complete": True,
                "language": "en",
                "segments": [dict(s) for s in segments],
                "settings": {"source": "recorded"},
            }
            if names:
                data["speaker_names"] = names
            tp.write_text(json.dumps(data), encoding="utf-8")
            db.upsert_transcript(
                "a.mp4", data, media_dir=str(tmp_path), abs_path=str(tmp_path / "a.mp4")
            )

        monkeypatch.setattr(tr, "find_transcript", lambda *a, **k: tp)
        monkeypatch.setattr(type(tr.config), "transcript_mode", property(lambda self: "alongside"))
        monkeypatch.setattr(type(tr.config), "media_dir", property(lambda self: tmp_path))
        names = {}
        monkeypatch.setattr(sp, "names_for_labels", lambda abs_path: names)
        yield {"db": db, "tp": tp, "put": put, "names": names, "src": tmp_path / "a.mp4"}
        db.close_all()

    TURNS = _turns(("SPEAKER_00", 0, 4.5), ("SPEAKER_01", 4.5, 10))

    def _db_rows(self, db):
        with db.connection() as c:
            segs = [
                tuple(r)
                for r in c.execute(
                    "SELECT speaker, text, start_sec FROM segments ORDER BY seg_index"
                )
            ]
            names = c.execute("SELECT speaker_names FROM transcripts").fetchone()[0]
        return segs, json.loads(names or "{}")

    def test_speakers_and_names_go_into_json_and_database(self, setup):
        setup["put"](self.SEGMENTS)
        setup["names"].update(
            {"SPEAKER_01": {"name": "Bob", "source": "recognition", "windows": 3, "of": 4}}
        )
        got = tr._voices_into_transcript(setup["db"], setup["src"], "a.mp4", self.TURNS)
        assert got == {"segments": 2, "names": {"SPEAKER_01": "Bob (recognition, 3 of 4 windows)"}}
        j = json.loads(setup["tp"].read_text(encoding="utf-8"))
        assert [s["speaker"] for s in j["segments"]] == ["SPEAKER_00", "SPEAKER_01", "—"]
        assert [s["text"] for s in j["segments"]] == [s["text"] for s in self.SEGMENTS]
        assert j["speakers"] == ["SPEAKER_00", "SPEAKER_01"] and j["speaker_names"] == {
            "SPEAKER_01": "Bob"
        }
        segs, names = self._db_rows(setup["db"])
        assert [s[0] for s in segs] == ["SPEAKER_00", "SPEAKER_01", "—"] and names == {
            "SPEAKER_01": "Bob"
        }
        assert [s[1] for s in segs] == [s["text"] for s in self.SEGMENTS]

    def test_speakers_already_there_stay_and_a_given_name_is_never_replaced(self, setup):
        mine = [dict(s, speaker="SPEAKER_05") for s in self.SEGMENTS]
        setup["put"](mine, names={"SPEAKER_05": "Carol"})
        setup["names"].update(
            {"SPEAKER_05": {"name": "Bob", "source": "map", "windows": 2, "of": 2}}
        )
        before = setup["tp"].stat().st_mtime_ns
        got = tr._voices_into_transcript(setup["db"], setup["src"], "a.mp4", self.TURNS)
        assert got == {"segments": 0, "names": {}}
        assert setup["tp"].stat().st_mtime_ns == before  # not even rewritten
        assert self._db_rows(setup["db"])[1] == {"SPEAKER_05": "Carol"}

    def test_a_missing_name_is_added_to_speakers_already_there(self, setup):
        mine = [dict(s, speaker="SPEAKER_05") for s in self.SEGMENTS]
        setup["put"](mine)
        setup["names"].update(
            {
                "SPEAKER_05": {"name": "Bob", "source": "map", "windows": 2, "of": 2},
                "SPEAKER_09": {"name": "Alice", "source": "map", "windows": 1, "of": 1},
            }
        )
        got = tr._voices_into_transcript(setup["db"], setup["src"], "a.mp4", self.TURNS)
        assert got["segments"] == 0 and list(got["names"]) == [
            "SPEAKER_05"
        ]  # not one it doesn't have
        j = json.loads(setup["tp"].read_text(encoding="utf-8"))
        assert {s["speaker"] for s in j["segments"]} == {"SPEAKER_05"}
        assert j["speaker_names"] == {"SPEAKER_05": "Bob"}

    def test_without_a_transcript_nothing_is_written(self, setup, monkeypatch):
        monkeypatch.setattr(tr, "find_transcript", lambda *a, **k: None)
        got = tr._voices_into_transcript(setup["db"], setup["src"], "a.mp4", self.TURNS)
        assert got["segments"] == 0 and got.get("reason") == "no transcript"
        assert not [p for p in os.listdir(setup["tp"].parent) if p.endswith(".tmp")]

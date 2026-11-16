"""The voice extractor — one person's voice as clips with their sentences.

All voices of one person, or a part filtered by name, date and so on,
as one audio file or single samples in a ZIP, with the transcript pieces —
e.g. for TTS training.

What these tests pin:

* which voices: the named ones, and — only when asked for — the recognized
  ones (similar enough, and this person the nearest); discarded voices and
  voices named as someone else never;
* which sentences: inside the voice's speech spans where it has them, not
  marked as two speakers at once where it has none; the filters;
* the order and the caps;
* the export: a dataset ZIP (WAV at the asked rate, metadata.csv in the
  LJSpeech layout, metadata.jsonl, README) or one file with subtitles; an
  unreachable recording is skipped and counted; a stop leaves nothing
  that looks finished;
* saving into a folder of your own: the folder is checked, the
  dataset can go unpacked, and nothing already there is ever replaced.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from flask import Flask

from whisper_tm import extensions as ext
from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.core import voice_extract as vx
from whisper_tm.database import Database
from whisper_tm.database.speaker import SpeakerDB

DIM = 192
RATE = 16000


def _vec(seed, basis=None, spread=0.03):
    r = np.random.default_rng(seed)
    v = basis if basis is not None else r.normal(size=DIM)
    v = v + spread * r.normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _wav(path, seconds=80.0):
    t = np.arange(int(seconds * RATE)) / RATE
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(x.tobytes())


def _seg(sp, a, b, text, **kw):
    return {
        "speaker": sp,
        "start": a,
        "end": b,
        "text": text,
        "avg_logprob": kw.pop("logprob", -0.2),
        **kw,
    }


ALICE, BOB = _vec(1, spread=0), _vec(2, spread=0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    clips = tmp_path / "clips"
    real = type(config).dirs.fget

    def dirs(self):
        d = dict(real(self))
        d["clips"] = clips
        d["work"] = tmp_path / "work"
        d["cache"] = tmp_path / "cache"
        return d

    monkeypatch.setattr(type(config), "dirs", property(dirs))
    t = config._data.setdefault("transcription", {})
    monkeypatch.setitem(t, "hallucination_patterns", [r"Untertitelung.*ZDF.*"])

    lib = Database(tmp_path / "lib.db")
    lib.initialize()
    monkeypatch.setattr(ext, "db", lib)
    sdb = SpeakerDB(tmp_path / "speaker.db")
    sdb.initialize()
    for mod in ("whisper_tm.database.speaker", "whisper_tm.api.voices", "whisper_tm.api.speaker"):
        monkeypatch.setattr(f"{mod}.speaker_db", lambda: sdb)
    inventory.discard()
    vx._sim_held.clear()
    vx._runs.clear()
    vx._edits.clear()  # hand edits are kept per person, module-wide

    rec = {
        "a": media / "2024-06-02 stream.wav",  # spans, Alice named
        "b": media / "AUD-20150301-WA0001.wav",  # spans, Alice recognized
        "c": media / "old recording.wav",  # no spans (an older recording)
    }
    for p in rec.values():
        _wav(p)

    def windows(label, basis, spans, n, seed):
        return [
            {
                "label": label,
                "windows": i + 1,
                "start_sec": spans[0][0] if spans else i * 10.0,
                "speech_sec": 30.0,
                "segments": len(spans),
                "parts": spans or None,
                "v_separated": _vec(seed + i, basis),
            }
            for i in range(n)
        ]

    sdb.write(
        str(rec["a"]),
        speaker_core.MODEL_BRAND,
        windows("SPEAKER_00", ALICE, [(0.0, 20.0), (30.0, 50.0)], 2, 100)
        + windows("SPEAKER_01", BOB, [(20.0, 30.0)], 1, 200),
    )
    sdb.write(
        str(rec["b"]), speaker_core.MODEL_BRAND, windows("SPEAKER_00", ALICE, [(5.0, 40.0)], 2, 300)
    )
    sdb.write(str(rec["c"]), speaker_core.MODEL_BRAND, windows("SPEAKER_00", ALICE, [], 2, 400))
    ids = {(w["abs_path"], w["label"]): [] for w in []}
    for p in rec.values():
        for w in sdb.windows(str(p)):
            ids.setdefault((str(p), w["label"]), []).append(w["id"])
    sdb.assign_many(ids[(str(rec["a"]), "SPEAKER_00")], "Alice")
    sdb.assign_many(ids[(str(rec["c"]), "SPEAKER_00")], "Alice")
    sdb.assign_many(ids[(str(rec["a"]), "SPEAKER_01")], "Bob")

    tx = {
        "a": [
            _seg("SPEAKER_00", 1.0, 5.0, "Hello there, this is a test."),
            _seg("SPEAKER_00", 6.0, 11.0, "Another good sentence to keep."),
            _seg("SPEAKER_01", 21.0, 25.0, "Bob says something here."),
            # straddles the span end (20.0) into Bob's turn
            _seg("SPEAKER_00", 18.0, 24.0, "This one runs into Bob."),
            _seg("SPEAKER_00", 31.0, 31.8, "Too short."),
            _seg("SPEAKER_00", 32.0, 48.0, "Oh."),  # 16 s for 3 chars
            _seg("SPEAKER_00", 40.0, 44.0, "Untertitelung des ZDF, 2020"),
            _seg("SPEAKER_00", 44.5, 48.5, "Unsure words said here.", logprob=-2.0),
        ],
        "b": [
            _seg("SPEAKER_00", 6.0, 10.0, "From the old voice message one."),
            _seg("SPEAKER_00", 12.0, 16.0, "From the old voice message two."),
        ],
        "c": [
            _seg("SPEAKER_00", 2.0, 6.0, "An old recording without spans."),
            _seg("SPEAKER_00", 8.0, 12.0, "Two people at once here.", overlapped=1),
        ],
    }
    for k, p in rec.items():
        lib.upsert_transcript(
            p.name,
            {"filename": p.name, "complete": True, "language": "en", "segments": tx[k]},
            media_dir=str(media),
            abs_path=str(p),
        )
    yield {
        "lib": lib,
        "sdb": sdb,
        "rec": rec,
        "ids": ids,
        "media": media,
        "clips": clips,
        "tmp": tmp_path,
    }
    inventory.discard()
    vx._sim_held.clear()
    vx._runs.clear()
    vx._edits.clear()


def _texts(run):
    return [c["text"] for c in run["clips"]]


def _run(world, **opts):
    return vx.build_run("Alice", opts, root=str(world["media"]))


class TestWhichVoices:
    def test_named_only(self, world):
        run = _run(world, include="named")
        files = {c["abs_path"] for c in run["clips"]}
        assert files == {str(world["rec"]["a"]), str(world["rec"]["c"])}
        assert run["stats"]["windows_named"] == 4 and run["stats"]["windows_recognized"] == 0

    def test_recognized_ones_join(self, world):
        run = _run(world, include="recognized", threshold=0.8)
        assert str(world["rec"]["b"]) in {c["abs_path"] for c in run["clips"]}
        assert run["stats"]["windows_recognized"] == 2
        assert all(not c["named"] for c in run["clips"] if c["abs_path"] == str(world["rec"]["b"]))

    def test_the_threshold_decides(self, world):
        run = _run(world, include="recognized", threshold=0.99)
        assert str(world["rec"]["b"]) not in {c["abs_path"] for c in run["clips"]}

    def test_a_voice_named_as_someone_else_never(self, world):
        run = _run(world, include="recognized", threshold=0.5)
        assert "Bob says something here." not in _texts(run)

    def test_discarded_voices_never(self, world):
        sdb = world["sdb"]
        sdb.move_to_trash(world["ids"][(str(world["rec"]["b"]), "SPEAKER_00")])
        run = _run(world, include="recognized", threshold=0.8)
        assert str(world["rec"]["b"]) not in {c["abs_path"] for c in run["clips"]}

    def test_an_unknown_person(self, world):
        with pytest.raises(ValueError, match="no voices"):
            vx.build_run("Nobody", {}, root="")


class TestWhichSentences:
    def test_inside_the_spans_and_clean(self, world):
        texts = _texts(_run(world, include="named"))
        assert "Hello there, this is a test." in texts
        assert "Another good sentence to keep." in texts
        assert "This one runs into Bob." not in texts
        dropped = dict(_run(world, include="named")["stats"]["dropped"])
        assert dropped["outside the voice's speech spans"] >= 1

    def test_without_spans_the_transcript_marker(self, world):
        run = _run(world, include="named")
        c = [x for x in run["clips"] if x["abs_path"] == str(world["rec"]["c"])]
        assert [x["text"] for x in c] == ["An old recording without spans."]
        assert c[0]["clean_by"] == "transcript"
        assert all(
            x["clean_by"] == "spans"
            for x in run["clips"]
            if x["abs_path"] != str(world["rec"]["c"])
        )

    def test_strict_leaves_voices_without_spans_out(self, world):
        run = _run(world, include="named", clean="strict")
        assert str(world["rec"]["c"]) not in {c["abs_path"] for c in run["clips"]}

    def test_length_rate_confidence_artifacts(self, world):
        texts = _texts(_run(world, include="named"))
        for gone in ("Too short.", "Oh.", "Untertitelung des ZDF, 2020", "Unsure words said here."):
            assert gone not in texts, gone
        # -2.0 is 14 % sure — below the default of 75 %
        texts = _texts(_run(world, include="named", min_certainty=10, skip_artifacts=False))
        assert "Unsure words said here." in texts
        assert "Untertitelung des ZDF, 2020" in texts

    def test_date_folder_and_text(self, world):
        run = _run(world, include="recognized", threshold=0.8, date_from="2020-01-01")
        assert {c["date"] for c in run["clips"]} == {"2024-06-02"}
        run = _run(world, include="recognized", threshold=0.8, date_to="2016-01-01")
        assert {c["date"] for c in run["clips"]} == {"2015-03-01"}
        run = _run(world, include="named", text="good")
        assert _texts(run) == ["Another good sentence to keep."]
        run = _run(world, include="named", text_not="test", file_text="stream")
        assert _texts(run) == ["Another good sentence to keep."]

    def test_padding_stays_inside_the_span(self, world):
        run = _run(world, include="named", padding=0.5, date_from="2024-01-01")
        first = run["clips"][0]
        assert first["start"] == 0.5 and first["end"] == 5.5
        run = _run(world, include="named", padding=1.5, date_from="2024-01-01")
        assert run["clips"][0]["start"] == 0.0, "padding left the span"

    def test_joining_neighbours(self, world):
        run = _run(world, include="named", join_gap=1.5, date_from="2024-01-01")
        assert _texts(run)[0] == ("Hello there, this is a test. Another good sentence to keep.")


class TestFromConversations:
    """Only the sentences of conversations the conversation statistics
    found — the search view's Conversations tab hands its filters over."""

    def test_only_recordings_the_conversations_hold(self, world):
        from whisper_tm.conversations import query

        query.forget()
        # The recordings last 48.5 s (a), 16 s (b) and 12 s (c)
        run = _run(world, include="recognized", conversations={"min_sec": "40"})
        assert {c["abs_path"] for c in run["clips"]} == {str(world["rec"]["a"])}
        dropped = dict(run["stats"]["dropped"])
        assert dropped["not in the chosen conversations"] >= 1
        conv = run["stats"]["conversations"]
        assert (conv["found"], conv["recordings"]) == (1, 1)
        run = _run(world, include="recognized", conversations={"max_sec": "20"})
        assert {c["abs_path"] for c in run["clips"]} == {
            str(world["rec"]["b"]),
            str(world["rec"]["c"]),
        }
        query.forget()

    def test_a_stretch_takes_the_sentences_in_it(self, world, monkeypatch):
        """A sentence counts when its middle lies in the stretch."""
        from whisper_tm.conversations import query

        a = str(world["rec"]["a"])
        monkeypatch.setattr(
            query,
            "ranges",
            lambda db, f: {
                "ranges": {a: [(0.0, 9.0)]},
                "conversations": 1,
                "people_unknown": [],
                "ignored": [],
                "sentences_baseline": None,
            },
        )
        run = _run(world, include="named", conversations={"people": ["Alice", "Bob"]})
        # 1–5 (middle 3) and 6–11 (middle 8.5) lie in it; the rest later
        assert _texts(run) == ["Hello there, this is a test.", "Another good sentence to keep."]
        assert dict(run["stats"]["dropped"])["outside the chosen conversations"] >= 1

    def test_switched_off_says_so(self, world, monkeypatch):
        from whisper_tm.conversations import store

        monkeypatch.setattr(store, "enabled", lambda: False)
        with pytest.raises(ValueError, match="switched off"):
            _run(world, conversations={"min_sec": "40"})

    def test_the_filters_are_cleaned(self):
        assert vx.clean_options(
            {
                "conversations": {
                    "people": ["Alice", "", 3, None],
                    "min_sec": 2700,
                    "kind": ["detailed", "group"],
                    "sentences": "longer",
                    "speakers_min": 0,
                    "nonsense": "x",
                    "q": True,
                }
            }
        )["conversations"] == {
            "people": ["Alice", "3"],
            "min_sec": "2700",
            "kind": "detailed,group",
            "sentences": "longer",
        }
        assert vx.clean_options({"conversations": "people=Alice"})["conversations"] == {}
        assert vx.clean_options({})["conversations"] == {}


class TestOrderAndAmount:
    def test_in_time_order_by_default(self, world):
        run = _run(world, include="recognized", threshold=0.8)
        dates = [c["date"] or "9999" for c in run["clips"]]
        assert dates == sorted(dates)

    def test_caps(self, world):
        run = _run(world, include="recognized", threshold=0.8, per_recording=1)
        assert len(run["clips"]) == len({c["abs_path"] for c in run["clips"]})
        run = _run(world, include="recognized", threshold=0.8, max_clips=2)
        assert len(run["clips"]) == 2
        run = _run(world, include="recognized", threshold=0.8, max_hours=8.5 / 3600)
        assert sum(c["end"] - c["start"] for c in run["clips"]) <= 8.5
        assert dict(run["stats"]["dropped"])["over the amount limit"] >= 1

    def test_spread_takes_a_little_of_each(self, world):
        run = _run(world, include="recognized", threshold=0.8, order="spread", max_clips=3)
        assert len({c["abs_path"] for c in run["clips"]}) == 3


def _export(world, run, **eo):
    return vx.export(run, vx.export_options(eo), threading.Event(), tmp_dir=world["tmp"] / "cut")


@pytest.mark.ffmpeg
class TestExport:
    def test_a_dataset(self, world):
        run = _run(world, include="recognized", threshold=0.8)
        r = _export(world, run, mode="dataset", sample_rate=22050)
        z = zipfile.ZipFile(world["clips"] / "voice-exports" / r["files"][0])
        wavs = sorted(n for n in z.namelist() if n.startswith("wavs/"))
        assert len(wavs) == r["clips"] == len(run["clips"])
        with wave.open(io.BytesIO(z.read(wavs[0]))) as w:
            assert w.getframerate() == 22050 and w.getnchannels() == 1
            # A steady tone has no pause to cut at: the clip is cut at the
            # sentence's own bounds instead of reaching into the padding.
            first = run["clips"][0]
            assert abs(w.getnframes() / 22050 - (first["s_end"] - first["s_start"])) < 0.05
        csv_lines = z.read("metadata.csv").decode().splitlines()
        assert csv_lines[0].count("|") == 2 and csv_lines[0].startswith("Alice_00001|")
        meta = [json.loads(x) for x in z.read("metadata.jsonl").decode().splitlines()]
        assert meta[0]["person"] == "Alice" and meta[0]["file"] == "wavs/Alice_00001.wav"
        assert {m["clean_by"] for m in meta} == {"spans", "transcript"}
        assert "LJSpeech" in z.read("README.txt").decode()

    def test_one_file_with_subtitles(self, world):
        run = _run(world, include="named")
        r = _export(world, run, mode="single", format="flac", sample_rate=16000, gap=0.5)
        folder = world["clips"] / "voice-exports"
        audio, srt, index = (folder / f for f in r["files"])
        assert audio.suffix == ".flac" and audio.stat().st_size > 1000
        cues = srt.read_text(encoding="utf-8").strip().split("\n\n")
        assert len(cues) == r["clips"]
        # 1.0-5.0 s: no pause in a steady tone, so the sentence's own bounds
        assert cues[0].splitlines()[1] == "00:00:00,000 --> 00:00:04,000"
        assert len(index.read_text(encoding="utf-8").splitlines()) == r["clips"]

    def test_an_unreachable_recording_is_skipped(self, world):
        run = _run(world, include="recognized", threshold=0.8)
        world["rec"]["b"].unlink()
        r = _export(world, run, mode="dataset")
        assert dict(r["left"]) == {"recording unreachable": 2}
        assert r["clips"] == len(run["clips"]) - 2

    def test_nothing_cut_is_an_error(self, world):
        run = _run(world, include="named")
        for p in world["rec"].values():
            p.unlink()
        with pytest.raises(RuntimeError, match="no clip could be cut"):
            _export(world, run, mode="dataset")
        assert not list((world["clips"] / "voice-exports").iterdir())

    def test_a_stop_leaves_nothing(self, world):
        run = _run(world, include="named")
        stop = threading.Event()
        stop.set()
        with pytest.raises(InterruptedError):
            vx.export(run, vx.export_options({}), stop, tmp_dir=world["tmp"] / "cut")
        assert not list((world["clips"] / "voice-exports").iterdir())

    def test_not_enough_space_is_refused(self, world, monkeypatch):
        run = _run(world, include="named")
        monkeypatch.setattr(vx.shutil, "disk_usage", lambda p: type("U", (), {"free": 1000})())
        with pytest.raises(ValueError, match="not enough space"):
            vx.start_export(run["run_id"], {})


class TestApi:
    @pytest.fixture
    def client(self, world):
        from whisper_tm.api.voices import voices_bp

        a = Flask(__name__)
        a.register_blueprint(voices_bp)
        return a.test_client()

    @pytest.mark.ffmpeg
    def test_preview_listen_export_download_delete(self, client, world):
        d = client.post(
            "/api/voices/extract/preview", json={"person": "Alice", "options": {"include": "named"}}
        ).get_json()
        assert d["stats"]["clips"] >= 3 and d["examples"][0]["text"]
        assert "clips" not in d, "the clip list stays on the server"
        r = client.get(f"/api/voices/extract/clip/{d['run_id']}/0")
        assert r.status_code == 200 and r.mimetype == "audio/mpeg"
        r = client.post(
            "/api/voices/extract/export", json={"run_id": d["run_id"], "mode": "dataset"}
        ).get_json()
        assert r["ok"]
        vx._job["thread"].join(timeout=60)  # the export runs in the background
        s = client.get("/api/voices/extract/status").get_json()
        assert not s["running"] and s["result"]["clips"] == d["stats"]["clips"]
        name = s["files"][0]["name"]
        dl = client.get(f"/api/voices/extract/download/{name}")
        assert dl.status_code == 200 and dl.data[:2] == b"PK"
        assert client.post("/api/voices/extract/delete", json={"name": name}).get_json()["ok"]
        assert client.get("/api/voices/extract/status").get_json()["files"] == []

    def test_errors_say_something(self, client):
        assert client.post("/api/voices/extract/preview", json={}).status_code == 400
        assert client.get("/api/voices/extract/clip/nope/0").status_code == 404
        r = client.post("/api/voices/extract/export", json={"run_id": "nope"})
        assert r.status_code == 400 and "gone" in r.get_json()["error"]
        assert client.get("/api/voices/extract/download/../x").status_code in (403, 404)


class TestOptions:
    def test_out_of_range_falls_back(self):
        o = vx.clean_options(
            {
                "threshold": 7,
                "min_sec": "x",
                "order": "chaos",
                "date_from": "2024-13",
                "max_sec": 1,
                "min_sec2": 3,
            }
        )
        assert o["threshold"] == 0.99 and o["min_sec"] == 2.0
        assert o["order"] == "time" and o["date_from"] == ""
        assert o["max_sec"] == 2.0

    def test_the_date_in_the_name(self):
        assert vx.date_from_name("/a/AUD-20150301-WA0001.mp3") == "2015-03-01"
        assert vx.date_from_name("/a/2024-06-02 12-00-00.mp4") == "2024-06-02"
        assert vx.date_from_name("/a/hash-7f3a.mp4") == ""


# ── The clip editor: hand edits, the cut at the nearest pause ────────────


def _tone(seconds, rate=16000, gaps=()):
    """A tone, silent where ``gaps`` say ((from, to) in seconds)."""
    t = np.arange(int(seconds * rate)) / rate
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    for a, b in gaps:
        x[int(a * rate) : int(b * rate)] = 0.0
    return (x * 32767).astype(np.int16)


class TestTheCutAtThePause:
    """Padding can reach into half a word of the previous sentence.
    The cut moves into the pause
    before the sentence and after it — where there is one."""

    def test_into_the_pauses_around_the_sentence(self):
        x = _tone(8, gaps=[(0.92, 1.0), (5.0, 5.12)])
        a, b = vx.snap_bounds(x, 16000, 0.0, 1.0, 5.0, 0.9, 5.1)
        assert 0.92 <= a <= 1.0, a  # not into the word before
        assert 5.0 <= b <= 5.12, b

    def test_a_noisy_recording_still_has_pauses(self):
        """With background noise at ~40 % of the voice, a fixed share of
        the voice finds no pause anywhere. Measured
        against the recording's own floor, the dips show."""
        rng = np.random.default_rng(3)
        voice = _tone(8, gaps=[(0.92, 1.0), (5.0, 5.12)]).astype(np.float32)
        noise = rng.normal(0, 0.4 * 0.3 * 32767 / np.sqrt(2), size=len(voice))
        x = np.clip(voice + noise, -32768, 32767).astype(np.int16)
        a, b = vx.snap_bounds(x, 16000, 0.0, 1.0, 5.0, 0.9, 5.1)
        assert 0.92 <= a <= 1.0 and 5.0 <= b <= 5.12, (a, b)

    def test_without_a_pause_the_sentence_bounds(self):
        x = _tone(8)
        assert vx.snap_bounds(x, 16000, 0.0, 1.0, 5.0, 0.9, 5.1) == (1.0, 5.0)

    def test_audio_that_starts_later(self):
        x = _tone(4, gaps=[(0.42, 0.5)])  # from 0.5 s on: 0.92–1.0
        a, _b = vx.snap_bounds(x, 16000, 0.5, 1.0, 3.0, 0.9, 3.1)
        assert 0.92 <= a <= 1.0, a

    @pytest.mark.ffmpeg
    def test_a_clip_trimmed_by_hand_is_not_moved(self, world):
        run = _run(world, include="named", date_from="2024-01-01")
        vx.edit_clip(run, 0, {"start": 1.3, "end": 4.6})
        r = _export(world, run, mode="dataset", sample_rate=16000)
        z = zipfile.ZipFile(world["clips"] / "voice-exports" / r["files"][0])
        meta = [json.loads(x) for x in z.read("metadata.jsonl").decode().splitlines()]
        assert (meta[0]["start"], meta[0]["end"], meta[0]["edited"]) == (1.3, 4.6, True)


class TestHandEdits:
    def test_bounds_and_text(self, world):
        run = _run(world, include="named")
        d = vx.edit_clip(run, 0, {"start": 1.2, "end": 4.8, "text": "Hello there."})
        c = d["clip"]
        assert (c["start"], c["end"], c["text"], c["edited"]) == (1.2, 4.8, "Hello there.", True)
        assert d["totals"]["edited"] == 1

    def test_they_survive_a_new_preview(self, world):
        run = _run(world, include="named")
        key = run["clips"][0]["key"]
        vx.edit_clip(run, 0, {"start": 1.2, "end": 4.8, "state": "kept"})
        again = _run(world, include="recognized", threshold=0.8)
        c = next(c for c in again["clips"] if c["key"] == key)
        assert (c["start"], c["end"], c["state"]) == (1.2, 4.8, "kept")
        assert again["stats"]["kept"] == 1

    def test_reset(self, world):
        run = _run(world, include="named")
        vx.edit_clip(run, 0, {"start": 1.2, "end": 4.8})
        c = vx.edit_clip(run, 0, {"reset": True})["clip"]
        assert (c["start"], c["end"], c["edited"]) == (c["orig_start"], c["orig_end"], False)
        assert _run(world, include="named")["stats"]["edited"] == 0

    @pytest.mark.ffmpeg
    def test_a_dropped_clip_is_left_out(self, world):
        run = _run(world, include="named")
        n = len(run["clips"])
        vx.edit_clip(run, 0, {"state": "dropped"})
        again = _run(world, include="named")
        assert again["stats"]["clips"] == n - 1
        assert dict(again["stats"]["dropped"])["dropped by hand"] == 1
        r = _export(world, again, mode="dataset")
        assert r["clips"] == n - 1

    @pytest.mark.ffmpeg
    def test_only_the_kept_ones(self, world):
        run = _run(world, include="named")
        vx.edit_clip(run, 1, {"state": "kept"})
        r = _export(world, run, mode="dataset", only_kept=True)
        assert r["clips"] == 1

    def test_forgetting_puts_everything_back(self, world):
        run = _run(world, include="named")
        vx.edit_clip(run, 0, {"start": 1.2, "end": 4.8, "state": "kept"})
        assert vx.forget_edits("alice") == 1  # by person, case folded
        again = _run(world, include="named")
        assert again["stats"]["kept"] == 0 and again["stats"]["edited"] == 0

    def test_nothing_kept_nothing_to_export(self, world):
        run = _run(world, include="named")
        with pytest.raises(ValueError, match="none is marked"):
            vx.start_export(run["run_id"], {"only_kept": True})

    @pytest.mark.parametrize(
        "bad",
        [
            {"start": 5, "end": 4},
            {"start": 1, "end": 1.1},
            {"start": 1, "end": 300},
            {"text": "  "},
            {"state": "maybe"},
            {"start": "x"},
        ],
    )
    def test_nonsense_is_refused(self, world, bad):
        run = _run(world, include="named")
        with pytest.raises(ValueError):
            vx.edit_clip(run, 0, bad)

    def test_pages_and_states(self, world):
        run = _run(world, include="recognized", threshold=0.8)
        vx.edit_clip(run, 0, {"state": "kept"})
        vx.edit_clip(run, 1, {"state": "dropped"})
        vx.edit_clip(run, 2, {"start": run["clips"][2]["start"] + 0.1})
        assert vx.clips_page(run, 0, 2)["total"] == len(run["clips"])
        assert len(vx.clips_page(run, 0, 2)["clips"]) == 2
        assert [c["i"] for c in vx.clips_page(run, state="kept")["clips"]] == [0]
        assert [c["i"] for c in vx.clips_page(run, state="dropped")["clips"]] == [1]
        assert [c["i"] for c in vx.clips_page(run, state="edited")["clips"]] == [2]
        assert vx.clips_page(run, state="open")["total"] == len(run["clips"]) - 2


class TestCertainty:
    def test_in_percent(self):
        assert vx.certainty(-0.105) == 90 and vx.certainty(0) == 100
        assert vx.certainty(None) is None

    def test_on_every_clip_and_as_a_filter(self, world):
        run = _run(world, include="named", min_certainty=10, skip_artifacts=False)
        unsure = next(c for c in run["clips"] if c["text"] == "Unsure words said here.")
        assert unsure["certainty"] == 14
        assert "Unsure words said here." not in _texts(_run(world, include="named"))
        assert dict(_run(world, include="named")["stats"]["dropped"])["Whisper unsure"] >= 1

    def test_the_surest_first(self, world):
        run = _run(world, include="named", min_certainty=10, skip_artifacts=False, order="sure")
        sure = [c["certainty"] for c in run["clips"]]
        assert sure == sorted(sure, reverse=True)

    def test_the_bands(self, world):
        bands = dict(_run(world, include="named")["stats"]["certainty"])
        assert bands == {"80–90 %": len(_run(world, include="named")["clips"])}


class TestEditorApi:
    @pytest.fixture
    def client(self, world):
        from whisper_tm.api.voices import voices_bp

        a = Flask(__name__)
        a.register_blueprint(voices_bp)
        return a.test_client()

    @pytest.mark.ffmpeg
    def test_page_edit_context_and_wav(self, client, world):
        d = client.post(
            "/api/voices/extract/preview", json={"person": "Alice", "options": {"include": "named"}}
        ).get_json()
        rid = d["run_id"]
        page = client.get(f"/api/voices/extract/clips/{rid}?offset=1&limit=2").get_json()
        assert page["offset"] == 1 and len(page["clips"]) == 2 and page["clips"][0]["i"] == 1
        r = client.post(f"/api/voices/extract/edit/{rid}/0", json={"start": 1.25, "end": 4.75})
        assert r.get_json()["clip"]["edited"]
        assert (
            client.post(
                f"/api/voices/extract/edit/{rid}/0", json={"start": 5, "end": 1}
            ).status_code
            == 400
        )
        ctx = client.get(f"/api/voices/extract/context/{rid}/0?pad=1")
        assert ctx.status_code == 200 and ctx.mimetype == "audio/wav"
        frm = float(ctx.headers["X-Clip-From"])
        with wave.open(io.BytesIO(ctx.data)) as w:
            secs = w.getnframes() / w.getframerate()
        # around the clip AND its sentence (1.0–5.0 s), a second each side
        assert frm == pytest.approx(0.0) and secs == pytest.approx(6.0, abs=0.05)
        wav = client.get(f"/api/voices/extract/clip/{rid}/0?format=wav")
        assert wav.status_code == 200 and "attachment" in wav.headers["Content-Disposition"]
        with wave.open(io.BytesIO(wav.data)) as w:
            assert w.getnframes() / w.getframerate() == pytest.approx(3.5, abs=0.05)


@pytest.mark.ffmpeg
class TestSaveToAFolder:
    """Instead of downloading the clips, they can be saved to a path you
    give."""

    @pytest.fixture
    def target(self, world, monkeypatch):
        from whisper_tm.config import config

        # Only the media folder is allowed — a folder beside it is not
        monkeypatch.setitem(
            config._data.setdefault("security", {}), "allowed_roots", [str(world["media"])]
        )
        t = world["media"] / "tts"
        t.mkdir()
        return t

    @pytest.fixture
    def client(self, world):
        from whisper_tm.api.voices import voices_bp

        a = Flask(__name__)
        a.register_blueprint(voices_bp)
        return a.test_client()

    def test_the_folder_is_checked(self, world, target):
        assert vx.dest_dir("") == vx.exports_dir()
        with pytest.raises(ValueError, match="full path"):
            vx.dest_dir("tts")
        with pytest.raises(ValueError, match="no folder"):
            vx.dest_dir(str(target / "missing"))
        with pytest.raises(ValueError, match="no folder"):
            vx.dest_dir(str(world["rec"]["a"]))  # a file, not a folder
        beside = world["tmp"] / "beside"
        beside.mkdir()
        with pytest.raises(ValueError, match="outside the allowed folders"):
            vx.dest_dir(str(beside))
        with pytest.raises(ValueError, match="outside the allowed folders"):
            vx.dest_dir(str(target / ".." / ".." / "beside"))
        locked = target / "locked"
        locked.mkdir()
        locked.chmod(0o555)
        try:
            if not os.access(locked, os.W_OK):  # root may write anyway
                with pytest.raises(ValueError, match="can't be written"):
                    vx.dest_dir(str(locked))
        finally:
            locked.chmod(0o755)
        d = vx.dest_info(str(target))
        assert d["ok"] and d["path"] == str(target.resolve()) and d["free"] > 0
        assert not vx.dest_info("tts")["ok"]

    def test_unpacked_only_with_a_folder_and_a_dataset(self):
        assert not vx.export_options({"unpacked": True})["unpacked"]
        assert not vx.export_options({"unpacked": True, "dest": "/x", "mode": "single"})["unpacked"]
        assert vx.export_options({"unpacked": True, "dest": "/x"})["unpacked"]

    def test_unpacked_into_the_folder(self, world, target):
        run = _run(world, include="named")
        r = _export(world, run, mode="dataset", dest=str(target), unpacked=True, name="alice")
        out = target / "alice"
        assert r["files"] == ["alice"] and r["dest"] == str(target.resolve()) and r["unpacked"]
        wavs = sorted((out / "wavs").iterdir())
        assert len(wavs) == r["clips"] == len(run["clips"])
        with wave.open(str(wavs[0])) as w:
            assert w.getframerate() == 22050 and w.getnchannels() == 1
        assert (out / "metadata.csv").read_text(encoding="utf-8").startswith("Alice_00001|")
        meta = [
            json.loads(x) for x in (out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [m["file"] for m in meta] == [f"wavs/{w.name}" for w in wavs]
        assert "LJSpeech" in (out / "README.txt").read_text(encoding="utf-8")
        assert r["bytes"] == sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        exports = world["clips"] / "voice-exports"
        assert not exports.exists() or not list(exports.iterdir()), "nothing in the exports folder"

    def test_as_a_zip_into_the_folder(self, world, target):
        run = _run(world, include="named")
        r = _export(world, run, mode="dataset", dest=str(target), name="alice")
        z = zipfile.ZipFile(target / "alice.zip")
        assert r["files"] == ["alice.zip"] and not r["unpacked"]
        assert "metadata.csv" in z.namelist()

    def test_nothing_there_is_replaced(self, world, target):
        run = _run(world, include="named")
        (target / "alice").mkdir()
        (target / "alice" / "mine.txt").write_text("keep me")
        (target / "alice-2.zip").write_text("keep me too")
        r1 = _export(world, run, mode="dataset", dest=str(target), unpacked=True, name="alice")
        r2 = _export(world, run, mode="dataset", dest=str(target), name="alice")
        r3 = _export(world, run, mode="dataset", dest=str(target), name="alice")
        assert r1["files"] == ["alice-2"]
        assert r2["files"] == ["alice.zip"]
        assert r3["files"] == ["alice-3.zip"]
        assert sorted(p.name for p in (target / "alice").iterdir()) == ["mine.txt"]
        assert (target / "alice" / "mine.txt").read_text() == "keep me"
        assert (target / "alice-2.zip").read_text() == "keep me too"

    def test_the_exports_folder_does_not_replace_either(self, world):
        run = _run(world, include="named")
        r1 = _export(world, run, mode="dataset", name="same")
        r2 = _export(world, run, mode="dataset", name="same")
        assert (r1["files"], r2["files"]) == (["same.zip"], ["same-2.zip"])

    def test_one_file_into_the_folder(self, world, target):
        run = _run(world, include="named")
        (target / "alice.srt").write_text("mine")
        r = _export(world, run, mode="single", format="flac", dest=str(target), name="alice")
        assert r["files"] == ["alice-2.flac", "alice-2.srt", "alice-2.jsonl"]
        assert (target / "alice.srt").read_text() == "mine"
        assert all((target / f).stat().st_size > 0 for f in r["files"])

    def test_a_stop_leaves_only_what_was_there(self, world, target):
        run = _run(world, include="named")
        (target / "other").mkdir()
        stop = threading.Event()
        stop.set()
        with pytest.raises(InterruptedError):
            vx.export(
                run,
                vx.export_options({"dest": str(target), "unpacked": True, "name": "alice"}),
                stop,
                tmp_dir=world["tmp"] / "cut",
            )
        assert sorted(p.name for p in target.iterdir()) == ["other"]

    def test_a_failure_halfway_removes_its_own_folder(self, world, target, monkeypatch):
        run = _run(world, include="named")
        calls = {"n": 0}
        real = vx._wav_bytes

        def fail_second(x, rate):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk gone")
            return real(x, rate)

        monkeypatch.setattr(vx, "_wav_bytes", fail_second)
        with pytest.raises(OSError, match="disk gone"):
            _export(world, run, mode="dataset", dest=str(target), unpacked=True, name="alice")
        assert list(target.iterdir()) == []

    def test_start_refuses_a_bad_folder(self, world, target):
        run = _run(world, include="named")
        with pytest.raises(ValueError, match="no folder"):
            vx.start_export(run["run_id"], {"dest": str(target / "missing")})
        assert not vx._job.get("running")

    def test_api(self, client, world, target):
        d = client.get("/api/voices/extract/dest", query_string={"path": str(target)}).get_json()
        assert d["ok"] and d["path"] == str(target.resolve())
        d = client.get(
            "/api/voices/extract/dest", query_string={"path": "/nowhere/at/all"}
        ).get_json()
        assert not d["ok"] and "no folder" in d["error"]
        pv = client.post(
            "/api/voices/extract/preview", json={"person": "Alice", "options": {"include": "named"}}
        ).get_json()
        r = client.post(
            "/api/voices/extract/export",
            json={"run_id": pv["run_id"], "mode": "dataset", "dest": "tts"},
        )
        assert r.status_code == 400 and "full path" in r.get_json()["error"]
        r = client.post(
            "/api/voices/extract/export",
            json={
                "run_id": pv["run_id"],
                "mode": "dataset",
                "dest": str(target),
                "unpacked": True,
                "name": "alice",
            },
        ).get_json()
        assert r["ok"]
        vx._job["thread"].join(timeout=60)
        s = client.get("/api/voices/extract/status").get_json()
        assert s["result"]["dest"] == str(target.resolve()) and s["result"]["files"] == ["alice"]
        assert (target / "alice" / "metadata.csv").exists()
        assert not s["files"], "a folder of your own is not in the download list"


HELLO_WORDS = {
    "start": 1.0,
    "end": 5.0,
    "text": "Hello there, this is a test.",
    "words": [
        {"word": "Hello", "start": 1.0, "end": 1.4, "score": 0.95},
        {"word": "there,", "start": 1.5, "end": 1.9, "score": 0.9},
        {"word": "this", "start": 2.0, "end": 2.3, "score": 0.42},
        {"word": "is", "start": None, "end": None, "score": None},
        {"word": "a", "start": 2.6, "end": 2.7, "score": 0.8},
        {"word": "test.", "start": 2.8, "end": 3.4, "score": 0.99},
    ],
}


class TestWords:
    """When word timestamps are on, every word is coloured by how sure it
    is — and highlighted while it plays."""

    def _words_json(self, world, segs):
        from whisper_tm.core.helpers import transcript_path_for

        p = world["rec"]["a"]
        tp = transcript_path_for(p.name, str(world["media"]), abs_path=str(p))
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps({"segments": segs}), encoding="utf-8")
        vx._words_held.clear()

    def _clip(self, run, start):
        page = vx.clips_page(run, 0, 50)
        return next(c for c in page["clips"] if c["text"].startswith(start))

    def test_the_words_of_a_clip(self, world):
        self._words_json(
            world,
            [
                HELLO_WORDS,
                {
                    "start": 6.0,
                    "end": 11.0,
                    "text": "Another good sentence to keep.",
                    # incomplete: these words don't spell the sentence
                    "words": [{"word": "Another", "start": 6.0, "end": 6.5, "score": 0.9}],
                },
            ],
        )
        run = _run(world, include="named", date_from="2024-01-01")
        w = self._clip(run, "Hello")["words"]
        assert [x["w"] for x in w] == ["Hello", "there,", "this", "is", "a", "test."]
        assert w[2]["p"] == 0.42 and w[3]["a"] is None, "an untimed word stays in its place"
        assert self._clip(run, "Another")["words"] is None

    def test_the_next_sentence_touching_it_is_not_taken_along(self, world):
        # The next sentence starts right where this one ends — its first
        # word must not be counted to this clip
        nxt = {
            "start": 5.04,
            "end": 5.6,
            "text": "And more.",
            "words": [
                {"word": "And", "start": 5.04, "end": 5.3, "score": 0.9},
                {"word": "more.", "start": 5.31, "end": 5.6, "score": 0.9},
            ],
        }
        seg = dict(
            HELLO_WORDS,
            words=HELLO_WORDS["words"][:-1]
            + [{"word": "test.", "start": 2.8, "end": 5.0, "score": 0.99}],
        )
        self._words_json(world, [seg, nxt])
        run = _run(world, include="named", date_from="2024-01-01")
        w = self._clip(run, "Hello")["words"]
        assert w[-1]["w"] == "test.", [x["w"] for x in w]

    def test_none_without_word_timestamps_or_after_a_correction(self, world):
        run = _run(world, include="named", date_from="2024-01-01")
        assert self._clip(run, "Hello")["words"] is None
        self._words_json(world, [HELLO_WORDS])
        assert self._clip(run, "Hello")["words"]
        i = next(i for i, c in enumerate(run["clips"]) if c["text"].startswith("Hello"))
        r = vx.edit_clip(run, i, {"text": "Hello there, this was a test."})
        assert r["clip"]["words"] is None, "corrected text: the words would no longer match"

    @pytest.mark.ffmpeg
    def test_the_export_carries_them_from_the_start_of_the_cut(self, world):
        self._words_json(world, [HELLO_WORDS])
        run = _run(world, include="named", date_from="2024-01-01")
        r = _export(world, run, mode="dataset", snap=False)
        z = zipfile.ZipFile(world["clips"] / "voice-exports" / r["files"][0])
        meta = [json.loads(x) for x in z.read("metadata.jsonl").decode().splitlines()]
        hello = next(m for m in meta if m["text"].startswith("Hello"))
        assert [w["word"] for w in hello["words"]][:2] == ["Hello", "there,"]
        assert abs(hello["words"][0]["start"] - (1.0 - hello["start"])) < 0.01
        assert hello["words"][3]["start"] is None and hello["words"][2]["score"] == 0.42
        assert all(m.get("words") is None for m in meta if not m["text"].startswith("Hello"))


class TestWholeRecordings:
    """An option to simply download all files the speaker was in — with
    all the filters applied too."""

    def test_every_recording_as_it_is(self, world):
        run = _run(world, include="recognized", threshold=0.8)
        r = _export(world, run, mode="recordings")
        z = zipfile.ZipFile(world["clips"] / "voice-exports" / r["files"][0])
        files = sorted(n for n in z.namelist() if n.startswith("recordings/"))
        want = sorted({c["abs_path"] for c in run["clips"]})
        assert files == sorted("recordings/" + p.rsplit("/", 1)[-1] for p in want)
        assert r["recordings"] == len(want) and r["files"][0].startswith("Alice-recordings-")
        for name in files:  # byte for byte the original
            assert z.read(name) == (world["media"] / name.split("/", 1)[1]).read_bytes()
        rows = list(csv.DictReader(io.StringIO(z.read("recordings.csv").decode())))
        assert {r_["source"] for r_ in rows} == set(want)
        assert sum(int(r_["clips"]) for r_ in rows) == len(run["clips"])
        assert "copied as they are" in z.read("README.txt").decode()

    def test_the_filters_decide_which(self, world):
        run = _run(world, include="named", file_text="stream")
        r = _export(world, run, mode="recordings")
        z = zipfile.ZipFile(world["clips"] / "voice-exports" / r["files"][0])
        assert [n for n in z.namelist() if n.startswith("recordings/")] == [
            "recordings/2024-06-02 stream.wav"
        ]

    def test_the_totals_count_recordings_and_their_size(self, world):
        lib = world["lib"]
        for p in world["rec"].values():
            lib.upsert_media_file(
                p.name, abs_path=str(p), extension=".wav", file_size=p.stat().st_size
            )
        run = _run(world, include="recognized", threshold=0.8)
        t = vx.totals(run)
        recs = {c["abs_path"] for c in run["clips"]}
        assert t["recordings"] == len(recs)
        assert t["recordings_bytes"] == sum(Path(p).stat().st_size for p in recs)
        eo = vx.export_options({"mode": "recordings"})
        assert vx.estimate_bytes(run, eo) == t["recordings_bytes"] + 2_000_000

    def test_unpacked_into_a_folder(self, world, monkeypatch):
        from whisper_tm.config import config

        monkeypatch.setitem(
            config._data.setdefault("security", {}), "allowed_roots", [str(world["media"])]
        )
        target = world["media"] / "out"
        target.mkdir()
        run = _run(world, include="named")
        r = _export(world, run, mode="recordings", dest=str(target), unpacked=True, name="alice")
        out = target / "alice"
        assert r["unpacked"] and (out / "recordings.csv").is_file()
        assert sorted(p.name for p in (out / "recordings").iterdir()) == sorted(
            {c["abs_path"].rsplit("/", 1)[-1] for c in run["clips"]}
        )

    def test_stopped_leaves_nothing(self, world):
        run = _run(world, include="named")
        stop = threading.Event()
        stop.set()
        with pytest.raises(InterruptedError):
            vx.export(run, vx.export_options({"mode": "recordings"}), stop)
        exports = world["clips"] / "voice-exports"
        assert not exports.exists() or not list(exports.iterdir())

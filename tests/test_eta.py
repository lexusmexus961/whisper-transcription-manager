"""Time left for the library and the queue (core/eta.py).

The sidebar's old remainder divided the open audio by the all-time average
speed. It was off three ways: the recent
pace differed clearly from the all-time average; a file also costs a fixed
amount (a few-second voice note takes about as long as it plays); and the
unindexed MP4 videos among the open files were not counted at all.
"""

from __future__ import annotations

import json
import time

import pytest

from whisper_tm.core import eta
from whisper_tm.core import queue as q

# ── The model ───────────────────────────────────────────────────────────


class TestFit:
    def test_fixed_part_and_rate_come_back_exactly(self):
        # 2 s per file + 1/20 of the audio
        samples = [
            (d, 2.0 + d / 20) for d in (5, 8, 12, 30, 45) * 5 + (90, 300, 1200, 3600, 20000) * 5
        ]
        m = eta.fit(*eta.groups_of(samples))
        assert m.a == pytest.approx(2.0) and m.b == pytest.approx(0.05)
        assert not m.rough and m.samples == 50

    def test_a_plain_speed_misses_the_fixed_part(self):
        # A queue of mostly notes under 10 s
        samples = [(7, 2.5)] * 700 + [(150, 10.0)] * 60
        m = eta.fit(*eta.groups_of(samples))
        pending_files, pending_audio = 50000, 50000 * 7.0
        with_fixed = m.seconds(pending_files, pending_audio)
        plain = pending_audio * sum(t for _, t in samples) / sum(d for d, _ in samples)
        assert with_fixed == pytest.approx(50000 * 2.5, rel=0.02)
        assert plain < with_fixed / 2

    def test_too_few_of_one_group_falls_back_to_one_ratio(self):
        samples = [(10, 3.0)] * 50 + [(600, 40.0)] * 5
        m = eta.fit(*eta.groups_of(samples))
        assert m.rough and m.a == 0
        assert m.b == pytest.approx((50 * 3 + 5 * 40) / (50 * 10 + 5 * 600))

    def test_nothing_measured_is_no_model(self):
        assert eta.fit((0, 0.0, 0.0), (0, 0.0, 0.0)) is None

    def test_a_meaningless_solution_is_not_used(self):
        # Short files slower per file than long ones would give a < 0
        samples = [(30, 100.0)] * 30 + [(600, 20.0)] * 30
        assert eta.fit(*eta.groups_of(samples)).rough


class TestDurationFromSize:
    def test_size_at_the_rate_of_the_extension(self):
        est, audio, unknown = eta.audio_from_size(
            [(".mp4", 10, 10 * 150e6), (".xyz", 3, 1e6)], {".mp4": 150e6 / 1500}
        )
        assert est == 10 and audio == pytest.approx(15000) and unknown == 3

    def test_no_size_stays_unknown(self):
        assert eta.audio_from_size([(".mp4", 4, 0)], {".mp4": 1e5}) == (0, 0.0, 4)


# ── Database inputs ──────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path, monkeypatch):
    from whisper_tm.database import Database

    d = Database(tmp_path / "library.db")
    d.initialize()
    monkeypatch.setattr(q, "_db", lambda: d)
    eta.clear_cache()
    yield d
    eta.clear_cache()


def _media(db, rel, dur, status="pending", size=0, ext=".mp4", root="/m"):
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO media_files (rel_path, abs_path, extension, file_size, "
            "duration_sec, status) VALUES (?, ?, ?, ?, ?, ?)",
            (rel, f"{root}/{rel}", ext, size, dur, status),
        )


def _transcript(db, rel, dur, secs, words=100, age_days=1, root="/m"):
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO transcripts (rel_path, abs_path, complete, word_count, "
            "duration_sec, processing_secs, created_at) VALUES (?, ?, 1, ?, ?, ?, ?)",
            (rel, f"{root}/{rel}", words, dur, secs, time.time() - age_days * 86400),
        )


def _queued(db, rel, status="pending", settings=None, started=None, finished=None, root="/m"):
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, status, "
            "position, started_at, finished_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
            (rel, root, json.dumps(settings or {}), status, started, finished),
        )


class TestLibrary:
    def test_recent_transcriptions_set_the_pace(self, db):
        for i in range(100):
            _transcript(db, f"s{i}.mp4", 10, 4.0 + 10 / 15)
            _transcript(db, f"l{i}.mp4", 1800, 4.0 + 1800 / 15)
            # older and twice as fast: not what the pipeline does today
            _transcript(db, f"o{i}.mp4", 1800, 1800 / 30, age_days=90)
        m = eta.transcription_model(db)
        assert m.a == pytest.approx(4.0) and m.b == pytest.approx(1 / 15)
        assert m.window_days == 30

    def test_a_thin_window_takes_all_of_the_history(self, db):
        # Fewer than 200 in 30 days: rather everything than a guess from 20
        for i in range(10):
            _transcript(db, f"s{i}.mp4", 10, 4.0 + 10 / 15)
            _transcript(db, f"l{i}.mp4", 1800, 4.0 + 1800 / 15)
        for i in range(100):
            _transcript(db, f"os{i}.mp4", 10, 4.0 + 10 / 15, age_days=90)
            _transcript(db, f"ol{i}.mp4", 1800, 4.0 + 1800 / 15, age_days=90)
        m = eta.transcription_model(db)
        assert m.samples == 220 and m.window_days is None

    def test_silent_recordings_do_not_count(self, db):
        for i in range(30):
            _transcript(db, f"s{i}.mp4", 10, 4.0 + 10 / 15)
            _transcript(db, f"l{i}.mp4", 1800, 4.0 + 1800 / 15)
            _transcript(db, f"q{i}.mp4", 86400, 240, words=0)  # 24 h of silence in 4 min
        assert eta.transcription_model(db).b == pytest.approx(1 / 15)

    def test_the_open_files_including_the_unindexed(self, db):
        for i in range(30):
            _transcript(db, f"s{i}.mp4", 10, 4.0 + 10 / 15)
            _transcript(db, f"l{i}.mp4", 1800, 4.0 + 1800 / 15)
        # indexed MP4s: 1 MB per 10 s of video
        for i in range(20):
            _media(db, f"idx{i}.mp4", 10, status="done", size=1_000_000)
        _media(db, "open1.mp4", 1500)
        _media(db, "open2.mp4", 0, size=150_000_000)  # 1500 s by its size
        _media(db, "open3.xyz", 0, size=5, ext=".xyz")  # nothing to go by
        stats = db.get_media_stats_under("/m")
        e = eta.library(db, "/m", stats)
        assert (e["known"], e["by_size"], e["unknown"]) == (1, 1, 1)
        assert e["size_audio_sec"] == pytest.approx(1500)
        assert e["sec"] == pytest.approx(2 * 4.0 + 3000 / 15)


class TestQueue:
    def _run(self, db, now):
        # Finished voices runs: 1.5 s + 1/20 of the audio, one after the
        # other (the samples are shares of the wall clock — runs that
        # overlapped would each count only their share)
        t = now - 2000
        for i in range(25):
            for name, dur in ((f"d{i}.aac", 8), (f"e{i}.aac", 600)):
                _media(db, name, dur, status="done")
                _queued(db, name, "done", {"job": "voices"}, t, t + 1.5 + dur / 20)
                t += 1.5 + dur / 20 + 0.01

    def test_voices_left_from_the_run_itself(self, db):
        now = time.time()
        self._run(db, now)
        for i in range(100):
            _media(db, f"p{i}.aac", 5, status="done")
            _queued(db, f"p{i}.aac", settings={"job": "voices"})
        _media(db, "long.aac", 3600, status="done")
        _queued(db, "long.aac", settings={"job": "voices"})
        est = eta.queue(db, q.eta_inputs(now=now), now=now)
        (voices,) = est["kinds"]
        assert voices["kind"] == "voices" and voices["files"] == 101
        assert est["sec"] == pytest.approx(101 * 1.5 + (500 + 3600) / 20)
        assert voices["source"] == "queue" and not voices["model"]["rough"]

    def test_the_running_entry_counts_with_what_is_left_of_it(self, db):
        now = time.time()
        self._run(db, now)
        _media(db, "cur.aac", 600, status="done")
        _queued(db, "cur.aac", "processing", {"job": "voices"}, started=now - 10)
        est = eta.queue(db, q.eta_inputs(now=now), now=now)
        assert est["sec"] == pytest.approx(1.5 + 600 / 20 - 10)

    def test_a_transcription_without_history_borrows_the_library(self, db):
        for i in range(30):
            _transcript(db, f"s{i}.mp4", 10, 4.0 + 10 / 15)
            _transcript(db, f"l{i}.mp4", 1800, 4.0 + 1800 / 15)
        _media(db, "new.mp4", 900)
        _queued(db, "new.mp4", settings={"model": "large-v3"})
        est = eta.queue(db, q.eta_inputs())
        (t,) = est["kinds"]
        assert t["kind"] == "transcription" and t["source"] == "library"
        assert est["sec"] == pytest.approx(4.0 + 900 / 15)

    def test_a_fresh_voices_run_has_no_estimate_yet(self, db):
        _media(db, "p.aac", 5, status="done")
        _queued(db, "p.aac", settings={"job": "voices"})
        est = eta.queue(db, q.eta_inputs())
        assert est["sec"] is None and est["kinds"][0]["sec"] is None

    def test_one_kind_without_an_estimate_makes_the_sum_partial(self, db):
        for i in range(30):
            _transcript(db, f"s{i}.mp4", 10, 4.0 + 10 / 15)
            _transcript(db, f"l{i}.mp4", 1800, 4.0 + 1800 / 15)
        _media(db, "t.mp4", 900)
        _queued(db, "t.mp4", settings={"model": "large-v3"})
        _media(db, "v.aac", 5, status="done")
        _queued(db, "v.aac", settings={"job": "voices"})
        est = eta.queue(db, q.eta_inputs())
        assert est["sec"] == pytest.approx(4.0 + 900 / 15) and est["partial"]

    def test_the_last_hour(self, db):
        now = time.time()
        self._run(db, now)
        _media(db, "old.aac", 60, status="done")
        _queued(db, "old.aac", "done", {"job": "voices"}, now - 7300, now - 7200)
        assert q.eta_inputs(now=now)["last_hour"] == {"files": 50, "audio_sec": 25 * 8 + 25 * 600}


class TestEndpoints:
    @pytest.fixture
    def client(self, db, monkeypatch):
        import whisper_tm.extensions as ext
        from whisper_tm.app import create_app

        monkeypatch.setattr(ext, "db", db)
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def test_queue_eta(self, client, db, monkeypatch):
        now = time.time()
        TestQueue()._run(db, now)
        for i in range(10):
            _media(db, f"p{i}.aac", 5, status="done")
            _queued(db, f"p{i}.aac", settings={"job": "voices"})
        monkeypatch.setattr(
            q,
            "snapshot",
            lambda: {
                "paused": False,
                "running": True,
                "total": 60,
                "processed": 50,
                "pending": 10,
                "fail_count": 0,
                "started_at": now - 600,
            },
        )
        d = client.get("/api/queue/eta").get_json()
        assert d["sec"] == pytest.approx(10 * 1.5 + 50 / 20)
        assert d["finish_at"] == pytest.approx(time.time() + d["sec"], abs=5)
        assert (d["total"], d["processed"], d["pending"]) == (60, 50, 10)

    def test_a_paused_queue_has_no_finish_time(self, client, db, monkeypatch):
        now = time.time()
        TestQueue()._run(db, now)
        _media(db, "p.aac", 5, status="done")
        _queued(db, "p.aac", settings={"job": "voices"})
        monkeypatch.setattr(
            q,
            "snapshot",
            lambda: {
                "paused": True,
                "running": True,
                "total": 51,
                "processed": 50,
                "pending": 1,
                "fail_count": 0,
                "started_at": now - 600,
            },
        )
        d = client.get("/api/queue/eta").get_json()
        assert d["sec"] > 0 and d["finish_at"] is None and d["paused"] is True

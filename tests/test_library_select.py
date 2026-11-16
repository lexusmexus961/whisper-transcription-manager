"""
Library selection (core/library_select.py and its two queue routes).

A selection decides what gets queued — possibly tens of thousands of files.
Each criterion must match exactly what it says, an unknown criterion must be
refused rather than ignored, and queuing must not act on a set other than
the one that was counted.
"""

import json

import pytest

from whisper_tm.core import library_select as ls
from whisper_tm.database import Database

R = "/lib"


def _seg(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end, "text": "x " * 5}


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = Database(tmp_path / "wx.db")
    d.initialize()

    def media(name, **kw):
        d.upsert_media_file(
            name,
            abs_path=f"{R}/{name}",
            extension="." + name.rsplit(".", 1)[1],
            is_video=int(name.endswith((".mp4", ".mkv"))),
            duration_sec=kw.pop("duration", 120),
            file_size=kw.pop("size", 1000),
            **kw,
        )

    def done(name, settings, segments, created=1_780_000_000):
        media(name)
        tid = d.upsert_transcript(
            name,
            {
                "complete": True,
                "model": "large-v3",
                "language": "de",
                "vocal_model": "mel_roformer_kim",
                "total_duration_sec": 120,
                "segments": segments,
                "settings": settings,
            },
            media_dir=R,
            abs_path=f"{R}/{name}",
        )
        with d.connection() as c:
            c.execute("UPDATE transcripts SET created_at = ? WHERE id = ?", (created, tid))

    media("backups/open.mp3")
    media("backups/long_open.mp3", duration=7200)
    media("videos/skipped.mp4", status="skipped")
    done(
        "videos/talk.mp4",
        {"source": "recorded", "diarized": True, "app_commit": "abc"},
        [_seg("SPEAKER_00", 0, 6), _seg("SPEAKER_01", 6, 8)],
    )
    done(
        "videos/talk.mkv",
        {"source": "recorded", "diarized": False},
        [_seg("—", 0, 6)],
        created=1_770_000_000,
    )
    done("backups/short.mp3", {"source": "recorded", "diarized": True}, [_seg("SPEAKER_00", 0, 2)])
    with d.connection() as c:
        c.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, status, "
            "position) VALUES ('backups/open.mp3', ?, '{}', 'pending', 1)",
            (R,),
        )
        c.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, status, "
            "position) VALUES ('backups/long_open.mp3', ?, '{}', 'error', 2)",
            (R,),
        )
    monkeypatch.setattr(ls, "_voiced_paths", lambda: {f"{R}/videos/talk.mp4"})
    monkeypatch.setattr(ls, "_old_voice_paths", lambda: {f"{R}/videos/talk.mp4"})
    yield d
    d.close_all()


def names(db, **criteria):
    with db.connection() as c:
        return sorted(p.removeprefix(R + "/") for p in ls.abs_paths(c, criteria))


class TestCriteria:
    def test_status(self, db):
        assert names(db, status=["open"]) == ["backups/long_open.mp3", "backups/open.mp3"]
        assert names(db, status=["done"]) == [
            "backups/short.mp3",
            "videos/talk.mkv",
            "videos/talk.mp4",
        ]
        assert names(db, status=["skipped"]) == ["videos/skipped.mp4"]

    def test_folders_names_and_extensions(self, db):
        assert names(db, folders=[f"{R}/videos"], exclude_name="*skip*") == [
            "videos/talk.mkv",
            "videos/talk.mp4",
        ]
        assert names(db, exclude_folders=[f"{R}/videos"], name="*open*") == [
            "backups/long_open.mp3",
            "backups/open.mp3",
        ]
        assert names(db, extensions=["mkv"]) == ["videos/talk.mkv"]

    def test_the_folder_boundary_is_respected(self, db):
        assert names(db, folders=[f"{R}/video"]) == []

    def test_duration_and_video(self, db):
        assert names(db, duration_min=3600) == ["backups/long_open.mp3"]
        assert names(db, video=False, status=["done"]) == ["backups/short.mp3"]

    def test_recorded_settings(self, db):
        assert names(db, diarized=False, status=["done"]) == ["videos/talk.mkv"]
        assert names(db, app_commits=["abc"]) == ["videos/talk.mp4"]
        assert names(db, transcribed_to="2026-02-05") == ["videos/talk.mkv"]

    def test_voices_and_the_speech_they_need(self, db):
        """The voices job selection: transcribed, no voices yet, and enough
        speech for a window."""
        assert names(db, status=["done"], voices=False, speech_for_voices=True) == [
            "videos/talk.mkv"
        ]
        assert names(db, voices=True) == ["videos/talk.mp4"]

    def test_a_voices_run_without_result_is_not_offered_again(self, db):
        """settings.voices is recorded by every voices-only run, also when no
        speaker had enough speech — those files must not come back."""
        assert names(db, status=["done"], voices=False, voices_run=False) == [
            "backups/short.mp3",
            "videos/talk.mkv",
        ]
        db.merge_transcript_settings(
            f"{R}/backups/short.mp3", {"voices": {"windows": 0, "reason": "too little"}}
        )
        assert names(db, status=["done"], voices=False, voices_run=False) == ["videos/talk.mkv"]
        assert names(db, voices_run=True) == ["backups/short.mp3"]

    def test_older_voices(self, db):
        """Voices made before their speech spans were stored — the ones a
        renewal gives spans."""
        assert names(db, voices_old=True) == ["videos/talk.mp4"]
        assert "videos/talk.mp4" not in names(db, voices_old=False)

    def test_queue_state(self, db):
        assert names(db, in_queue=True) == ["backups/open.mp3"]
        assert names(db, status=["open"], in_queue=False) == ["backups/long_open.mp3"]
        assert names(db, failed_min=1) == ["backups/long_open.mp3"]

    def test_another_format_already_transcribed(self, db):
        assert names(db, other_format_transcribed=True) == ["videos/talk.mkv", "videos/talk.mp4"]

    def test_count_and_audio_total(self, db):
        with db.connection() as c:
            out = ls.select(c, {"status": ["open"]})
        assert out["count"] == 2 and out["audio_sec"] == 7320
        assert [s["name"] for s in out["sample"]] == ["long_open.mp3", "open.mp3"]


class TestRefusals:
    @pytest.mark.parametrize(
        "criteria",
        [
            {"stauts": ["open"]},  # typo
            {"status": ["opn"]},
            {"duration_min": "60"},
            {"voices": "no"},
            {"folders": ["relative/dir"]},
            {"transcribed_from": "01.02.2026"},
        ],
    )
    def test_a_wrong_criterion_is_an_error(self, db, criteria):
        with db.connection() as c, pytest.raises(ls.CriteriaError):
            ls.select(c, criteria)

    def test_empty_values_count_as_absent(self, db):
        assert len(names(db, status=[], name="")) == 6


class TestRoutes:
    @pytest.fixture
    def client(self, db, monkeypatch, tmp_path):
        from flask import Flask

        from whisper_tm import extensions as ext
        from whisper_tm.api import queue as queue_api
        from whisper_tm.core import queue as q

        monkeypatch.setattr(ext, "db", db)
        queued = []
        monkeypatch.setattr(
            q,
            "enqueue",
            lambda names, settings, base_dir=None, *a, **k: (
                queued.append((sorted(names), settings, base_dir))
                or [{"filename": n} for n in names]
            ),
        )
        monkeypatch.setattr(
            "whisper_tm.core.security.allowed_roots", lambda: [__import__("pathlib").Path(R)]
        )
        app = Flask(__name__)
        app.register_blueprint(queue_api.queue_bp)
        client = app.test_client()
        client.queued = queued
        return client

    def test_a_count_alone_sends_no_rows(self, client):
        d = client.post(
            "/api/queue/library/select", json={"criteria": {"status": ["open"]}, "limit": 0}
        ).get_json()
        assert d["count"] == 2 and d["sample"] == []
        d = client.post(
            "/api/queue/library/select", json={"criteria": {"status": ["open"]}}
        ).get_json()
        assert len(d["sample"]) == 2 and d["limit"] == 50

    def test_select_counts(self, client):
        d = client.post(
            "/api/queue/library/select", json={"criteria": {"status": ["done"], "voices": False}}
        ).get_json()
        assert d["count"] == 2

    def test_enqueue_as_voices_job(self, client):
        crit = {"status": ["done"], "voices": False, "speech_for_voices": True}
        r = client.post(
            "/api/queue/library/enqueue",
            json={"criteria": crit, "job": "voices", "expected_count": 1},
        )
        assert r.status_code == 200, r.get_json()
        assert client.queued == [(["videos/talk.mkv"], {"job": "voices"}, R)]

    def test_enqueue_as_renewal(self, client):
        """Renewing keeps the names: the queue entry carries the setting the
        worker hands on (SpeakerDB.replace_keeping_decisions)."""
        r = client.post(
            "/api/queue/library/enqueue",
            json={"criteria": {"voices_old": True}, "job": "voices_refresh", "expected_count": 1},
        )
        assert r.status_code == 200, r.get_json()
        assert client.queued == [(["videos/talk.mp4"], {"job": "voices", "refresh": True}, R)]

    def test_a_changed_selection_is_not_queued(self, client):
        r = client.post(
            "/api/queue/library/enqueue",
            json={"criteria": {"status": ["done"]}, "job": "voices", "expected_count": 99},
        )
        assert r.status_code == 409 and r.get_json()["count"] == 3
        assert client.queued == []

    def test_an_unknown_job_is_refused(self, client):
        r = client.post(
            "/api/queue/library/enqueue",
            json={"criteria": {}, "job": "everything", "expected_count": 0},
        )
        assert r.status_code == 400

    def test_settings_pass_the_same_whitelist(self, client):
        client.post(
            "/api/queue/library/enqueue",
            json={
                "criteria": {"status": ["open"], "in_queue": False},
                "job": "transcribe",
                "expected_count": 1,
                "settings": {"diarize": True, "rm_rf": True},
            },
        )
        assert client.queued[0][1] == {"diarize": True}


class TestAtTheFront:
    """The queue editor's "Add folder" tab: transcribe this folder and push
    it to the top — a folder's files without a transcript next up, before a
    long-running batch, in path order; files already waiting move up with
    them, keeping their own settings."""

    @pytest.fixture
    def client(self, db, monkeypatch):
        from flask import Flask

        from whisper_tm import extensions as ext
        from whisper_tm.api import queue as queue_api

        monkeypatch.setattr(ext, "db", db)
        monkeypatch.setattr(
            "whisper_tm.core.security.allowed_roots", lambda: [__import__("pathlib").Path(R)]
        )
        # The real enqueue: "/lib" is a symlink on Linux, which the root
        # check would resolve away from the allowed root
        monkeypatch.setattr("whisper_tm.core.queue.root_allowed", lambda p: True)
        for name in ("games/c.wav", "games/a.wav", "games/sub/b.wav"):
            db.upsert_media_file(
                name, abs_path=f"{R}/{name}", extension=".wav", duration_sec=4, file_size=100
            )
        with db.connection() as c:
            c.execute(
                "INSERT INTO queue_items (filename, base_dir, settings_json, status, "
                "position) VALUES ('x/old.mp4', ?, '{\"job\": \"voices\"}', "
                "'pending', 3)",
                (R,),
            )
        app = Flask(__name__)
        app.register_blueprint(queue_api.queue_bp)
        return app.test_client()

    def _order(self, db):
        with db.connection() as c:
            return [
                r[0]
                for r in c.execute(
                    "SELECT filename FROM queue_items WHERE status = 'pending' "
                    "ORDER BY position, id"
                )
            ]

    def test_a_folder_goes_next_up_in_path_order(self, client, db):
        crit = {"folders": [f"{R}/games"], "status": ["open"]}
        d = client.post(
            "/api/queue/library/select", json={"criteria": crit, "limit": 8, "estimate": True}
        ).get_json()
        assert d["count"] == 3 and d["audio_sec"] == 12 and "estimate_sec" in d
        r = client.post(
            "/api/queue/library/enqueue",
            json={
                "criteria": crit,
                "job": "transcribe",
                "front": True,
                "settings": {"model": "large-v3", "vocal_isolation": False},
                "expected_count": 3,
            },
        ).get_json()
        assert (r["added"], r["moved_up"], r["front"]) == (3, 0, True)
        assert self._order(db) == [
            "games/a.wav",
            "games/c.wav",
            "games/sub/b.wav",
            "backups/open.mp3",
            "x/old.mp4",
        ]
        with db.connection() as c:
            settings = json.loads(
                c.execute(
                    "SELECT settings_json FROM queue_items WHERE filename = 'games/a.wav'"
                ).fetchone()[0]
            )
        assert settings == {"model": "large-v3", "vocal_isolation": False}

    def test_already_waiting_ones_move_up_with_their_own_settings(self, client, db):
        r = client.post(
            "/api/queue/library/enqueue",
            json={
                "criteria": {"folders": [f"{R}/backups"], "status": ["open"]},
                "job": "transcribe",
                "front": True,
                "settings": {"model": "turbo"},
                "expected_count": 2,
            },
        ).get_json()
        assert (r["added"], r["moved_up"], r["already_queued"]) == (1, 1, 0)
        assert self._order(db)[:2] == ["backups/open.mp3", "backups/long_open.mp3"]
        with db.connection() as c:
            kept = c.execute(
                "SELECT settings_json FROM queue_items WHERE "
                "filename = 'backups/open.mp3' AND status = 'pending'"
            ).fetchone()[0]
        assert kept == "{}"

    def test_at_the_end_when_not_asked_otherwise(self, client, db):
        r = client.post(
            "/api/queue/library/enqueue",
            json={
                "criteria": {"folders": [f"{R}/games"], "status": ["open"]},
                "job": "transcribe",
                "expected_count": 3,
            },
        ).get_json()
        assert (r["added"], r["moved_up"], r["front"]) == (3, 0, False)
        assert self._order(db) == [
            "backups/open.mp3",
            "x/old.mp4",
            "games/a.wav",
            "games/c.wav",
            "games/sub/b.wav",
        ]


def test_json_roundtrip_of_a_saved_selection():
    """Selections are plain JSON — a saved one reads back identically."""
    crit = {"status": ["done"], "voices": False, "duration_min": 60}
    assert ls._validate(json.loads(json.dumps(crit))) == crit


def test_a_transcript_without_valid_settings_breaks_nothing(db):
    """Transcripts restored from a backup can carry settings_json = '' —
    json_extract() threw on them, and "From library"'s choice lists and
    every criterion on the recorded settings failed."""
    with db.connection() as c:
        c.execute(
            "UPDATE transcripts SET settings_json = '' WHERE abs_path = ?",
            (f"{R}/backups/short.mp3",),
        )
        f = ls.facets(c)
    assert f["app_commits"] == [{"value": "abc", "count": 1}]
    assert names(db, diarized=False, status=["done"]) == ["backups/short.mp3", "videos/talk.mkv"]
    assert names(db, app_commits=["abc"]) == ["videos/talk.mp4"]
    assert "backups/short.mp3" in names(db, voices_run=False, status=["done"])


def test_facets_list_what_occurs(db):
    with db.connection() as c:
        f = ls.facets(c)
    assert f["models"] == [{"value": "large-v3", "count": 3}]
    assert {"value": "abc", "count": 1} in f["app_commits"]
    assert {e["value"] for e in f["extensions"]} == {".mp3", ".mp4", ".mkv"}

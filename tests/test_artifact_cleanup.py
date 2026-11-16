"""Whisper artifacts in the transcripts that exist: dry run, apply, undo.

The filter works while transcribing; this applies the same patterns to the
library afterwards — first a dry run, then apply, so you can see what is
affected. It changes the user's transcripts, so
the tests pin the safety rules as much as the result:

* the dry run changes nothing;
* applying backs the database up first and refuses without a backup;
* every removed segment is logged before its transcript changes, and undo
  puts it back where it was — in the database and in the JSON file;
* a transcript with a translation or a correction run is left alone;
* the change counter of the backups is not flooded.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from whisper_tm import extensions as ext
from whisper_tm.config import config
from whisper_tm.core import artifact_cleanup as cleanup
from whisper_tm.database import Database

ZDF = r"Untertitelung.*ZDF.*"


def _seg(text, start):
    return {
        "start": start,
        "end": start + 1.0,
        "text": text,
        "speaker": "SPEAKER_00",
        "start_fmt": "",
        "end_fmt": "",
    }


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A library of three transcripts, with JSON files next to the media."""
    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    t = config._data.setdefault("transcription", {})
    monkeypatch.setitem(t, "hallucination_patterns", [ZDF])
    d = config._data.setdefault("directories", {})
    monkeypatch.setitem(d, "transcript_mode", "alongside")
    monkeypatch.setitem(
        config._data.setdefault("database", {}), "backup_path", str(tmp_path / "backups")
    )
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    media = tmp_path / "media"
    media.mkdir()
    files = {
        "a.mp3": ["Hello there.", "Untertitelung des ZDF, 2020", "How are you?", "…"],
        "b.mp3": ["Nothing to see here.", "Really nothing."],
        "c.mp3": ["Untertitelung des ZDF, 2020", "Translated line."],
    }
    for name, texts in files.items():
        (media / name).write_bytes(b"x")
        data = {
            "filename": name,
            "complete": True,
            "segments": [_seg(x, i * 2.0) for i, x in enumerate(texts)],
        }
        db.upsert_transcript(name, data, media_dir=str(media), abs_path=str(media / name))
        tx = media / ".transcripts"
        tx.mkdir(exist_ok=True)
        (tx / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    # c.mp3 has a translation: its segments are addressed by position
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO translations (abs_path, rel_path, language, segments_json)"
            " VALUES (?, 'c.mp3', 'en', '[]')",
            (str(media / "c.mp3"),),
        )
    cleanup._runs.clear()
    cleanup._last.update(dry_run=None, applied=None, undone=None)
    return db, media


def _texts(db, media, name):
    return [s["text"] for s in db.get_transcript(str(media / name))["segments"]]


def _dry(patterns=(ZDF,)):
    return cleanup.dry_run(list(patterns), stop_event=threading.Event())


class TestDryRun:
    def test_it_counts_and_changes_nothing(self, lib):
        db, media = lib
        before = _texts(db, media, "a.mp3")
        r = _dry()
        # a.mp3: the credit and the "…" (no letter or digit)
        assert r["segments"] == 2 and r["transcripts"] == 1
        assert dict(r["reasons"]) == {ZDF: 1, "no letter or digit": 1}
        assert ["Untertitelung des ZDF, 2020", 1] in r["texts"]
        assert _texts(db, media, "a.mp3") == before

    def test_a_translated_transcript_is_left_alone(self, lib):
        r = _dry()
        assert r["skipped"]["transcripts"] == 1 and r["skipped"]["segments"] == 1
        assert r["skipped"]["files"][0][0] == "c.mp3"
        assert "translation" in r["skipped"]["files"][0][2]

    def test_a_broken_pattern_is_refused(self, lib):
        with pytest.raises(ValueError, match="not a valid regular expression"):
            _dry(["(unclosed"])


class TestApply:
    def test_removes_what_the_dry_run_found(self, lib):
        db, media = lib
        r = _dry()
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        assert a["segments"] == 2 and a["transcripts"] == 1
        assert _texts(db, media, "a.mp3") == ["Hello there.", "How are you?"]
        # the JSON next to the media too
        data = json.loads((media / ".transcripts" / "a.mp3.json").read_text(encoding="utf-8"))
        assert [s["text"] for s in data["segments"]] == ["Hello there.", "How are you?"]
        # the others untouched
        assert _texts(db, media, "b.mp3") == ["Nothing to see here.", "Really nothing."]
        assert _texts(db, media, "c.mp3")[0] == "Untertitelung des ZDF, 2020"

    def test_the_database_is_backed_up_first(self, lib, tmp_path):
        r = _dry()
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        backup = Path(a["backup"])
        assert backup.exists() and backup.parent == tmp_path / "backups"

    def test_without_a_backup_nothing_changes(self, lib, monkeypatch):
        db, media = lib
        r = _dry()
        monkeypatch.setattr(Database, "backup", lambda self, dest_dir=None: None)
        with pytest.raises(RuntimeError, match="backup failed"):
            cleanup.apply(r["run_id"], stop_event=threading.Event())
        assert "Untertitelung des ZDF, 2020" in _texts(db, media, "a.mp3")

    def test_every_removed_segment_is_logged_first(self, lib, tmp_path):
        r = _dry()
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        lines = [
            json.loads(x)
            for x in (tmp_path / "data" / "cleanups" / a["log"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert lines[0]["kind"] == "header" and lines[0]["patterns"] == [ZDF]
        entry = lines[1]
        assert entry["segments_before"] == 4
        assert [i for i, _ in entry["removed"]] == [1, 3]
        assert entry["removed"][0][1]["text"] == "Untertitelung des ZDF, 2020"

    def test_it_judges_the_text_of_now(self, lib):
        # Edited after the dry run: the credit became real speech
        db, media = lib
        r = _dry()
        data = db.get_transcript(str(media / "a.mp3"))
        data["segments"][1]["text"] = "A real sentence now."
        data["segments"] = data["segments"][:3]
        db.upsert_transcript("a.mp3", data, media_dir=str(media), abs_path=str(media / "a.mp3"))
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        assert a["segments"] == 0
        assert ["nothing to remove any more", 1] in a["left"]

    def test_the_change_counter_is_not_flooded(self, lib, monkeypatch):
        seen = []
        monkeypatch.setattr(
            Database,
            "note_transcript_change",
            lambda self, n=1: seen.append(getattr(self._local, "skip_backup_counter", False)),
        )
        r = _dry()
        cleanup.apply(r["run_id"], stop_event=threading.Event())
        assert seen and all(seen), "a change outside bulk_import would count toward the backups"

    def test_a_dry_run_is_used_once(self, lib):
        r = _dry()
        cleanup.apply(r["run_id"], stop_event=threading.Event())
        with pytest.raises(ValueError, match="run it again"):
            cleanup.apply(r["run_id"], stop_event=threading.Event())


class TestUndo:
    def test_puts_the_segments_back_where_they_were(self, lib):
        db, media = lib
        before = _texts(db, media, "a.mp3")
        r = _dry()
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        u = cleanup.undo(a["log"], stop_event=threading.Event())
        assert u["segments"] == 2 and u["transcripts"] == 1
        assert _texts(db, media, "a.mp3") == before
        data = json.loads((media / ".transcripts" / "a.mp3.json").read_text(encoding="utf-8"))
        assert [s["text"] for s in data["segments"]] == before

    def test_the_log_is_marked_and_not_undone_twice(self, lib, tmp_path):
        r = _dry()
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        cleanup.undo(a["log"], stop_event=threading.Event())
        logs = cleanup.list_logs()
        assert logs[0]["undone"] and logs[0]["name"].endswith(".undone.jsonl")
        with pytest.raises(ValueError):
            cleanup.undo(a["log"], stop_event=threading.Event())

    def test_a_transcript_changed_since_is_left_alone(self, lib):
        db, media = lib
        r = _dry()
        a = cleanup.apply(r["run_id"], stop_event=threading.Event())
        data = db.get_transcript(str(media / "a.mp3"))
        data["segments"].append(_seg("Added later.", 9.0))
        db.upsert_transcript("a.mp3", data, media_dir=str(media), abs_path=str(media / "a.mp3"))
        u = cleanup.undo(a["log"], stop_event=threading.Event())
        assert u["transcripts"] == 0 and ["changed since the cleanup", 1] in u["left"]
        assert _texts(db, media, "a.mp3")[-1] == "Added later."

    @pytest.mark.parametrize("name", ["../x.jsonl", "cleanup-x.txt", "other.jsonl"])
    def test_only_cleanup_logs(self, lib, name):
        with pytest.raises(ValueError):
            cleanup.undo(name, stop_event=threading.Event())


class TestApi:
    @pytest.fixture
    def client(self, lib, monkeypatch):
        from flask import Flask

        from whisper_tm.api.artifacts import artifacts_bp

        app = Flask(__name__)
        app.register_blueprint(artifacts_bp)
        # Run the background step right here, so the test sees its end
        monkeypatch.setattr(
            cleanup.threading,
            "Thread",
            lambda target, name="", daemon=False: type("T", (), {"start": staticmethod(target)})(),
        )
        return app.test_client()

    def test_dry_run_apply_undo_through_the_api(self, client, lib):
        db, media = lib
        assert client.post("/api/artifacts/dry_run", json={"patterns": [ZDF]}).get_json()["ok"]
        s = client.get("/api/artifacts/status").get_json()
        assert not s["running"] and s["dry_run"]["segments"] == 2
        assert client.post(
            "/api/artifacts/apply", json={"run_id": s["dry_run"]["run_id"]}
        ).get_json()["ok"]
        s = client.get("/api/artifacts/status").get_json()
        assert s["applied"]["segments"] == 2 and s["logs"][0]["segments"] == 2
        assert client.post("/api/artifacts/undo", json={"log": s["logs"][0]["name"]}).get_json()[
            "ok"
        ]
        assert "Untertitelung des ZDF, 2020" in _texts(db, media, "a.mp3")

    def test_a_broken_pattern_is_refused_before_starting(self, client):
        r = client.post("/api/artifacts/dry_run", json={"patterns": ["(unclosed"]})
        assert r.status_code == 400 and "not a valid" in r.get_json()["error"]

    def test_without_patterns_the_saved_ones(self, client, lib):
        client.post("/api/artifacts/dry_run", json={})
        s = client.get("/api/artifacts/status").get_json()
        assert s["dry_run"]["patterns"] == [ZDF]

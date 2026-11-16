"""Undo of an artifact cleanup on legacy rows (no abs_path).

The undo lookup used to run ``WHERE abs_path = ? OR (abs_path = ''
AND rel_path = ?)`` with ``''`` as the first value when the logged row had
no identity — the first clause then matched EVERY legacy row, ``LIMIT 1``
took an unrelated one, and its transcript was overwritten with the logged
one's segments.
"""

from __future__ import annotations

import json
import threading

import pytest

from whisper_tm import extensions as ext
from whisper_tm.config import config
from whisper_tm.core import artifact_cleanup as cleanup
from whisper_tm.database import Database


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
def legacy(tmp_path, monkeypatch):
    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setitem(config._data.setdefault("directories", {}), "transcript_mode", "database")
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    media = tmp_path / "media"
    media.mkdir()
    for name, texts in (("x.mp3", ["Unrelated one.", "Unrelated two."]), ("y.mp3", ["Kept line."])):
        data = {
            "filename": name,
            "complete": True,
            "segments": [_seg(t, i * 2.0) for i, t in enumerate(texts)],
        }
        db.upsert_transcript(name, data, media_dir=str(media), abs_path=str(media / name))
    with db.connection() as conn:
        # Rows from before abs_path existed
        conn.execute("UPDATE transcripts SET abs_path = ''")
    return db, media


def test_undo_restores_the_logged_legacy_transcript_not_another(legacy):
    db, media = legacy
    log_dir = cleanup.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "cleanup-20240301-120000.jsonl"
    with open(log_dir / name, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "header", "patterns": [], "started_at": 0}) + "\n")
        f.write(
            json.dumps(
                {
                    "kind": "transcript",
                    "rel_path": "y.mp3",
                    "abs_path": "",
                    "media_dir": str(media),
                    "segments_before": 2,
                    "removed": [[0, _seg("Removed credit.", 0.0)]],
                }
            )
            + "\n"
        )
    r = cleanup.undo(name, stop_event=threading.Event())
    with db.connection() as conn:
        rows = {
            r_["rel_path"]: r_["json_data"]
            for r_ in conn.execute("SELECT rel_path, json_data FROM transcripts")
        }
    x = json.loads(rows["x.mp3"])
    y = json.loads(rows["y.mp3"])
    assert [s["text"] for s in x["segments"]] == ["Unrelated one.", "Unrelated two."]
    assert [s["text"] for s in y["segments"]] == ["Removed credit.", "Kept line."]
    assert r["transcripts"] == 1

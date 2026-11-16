"""
Correction review API: partial apply, individual dismiss, undo.

The changelog in the UI allows approve/dismiss PER LINE — the run must
stay open until each suggestion is decided, and the
applied edits must accumulate together (otherwise undo would be
incomplete). These tests check exactly that against the real routes.
"""

import pytest

from whisper_tm.config import config


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    app = create_app()
    return app.test_client()


@pytest.fixture
def run(app_client, monkeypatch, tmp_path):
    """Transcript + stored correction run with two suggestions."""
    from whisper_tm.api import llm_jobs
    from whisper_tm.extensions import db

    tdata = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "The bugdet is approved."},
            {"start": 1.0, "end": 2.0, "text": "We meet on Wensday."},
            {"start": 2.0, "end": 3.0, "text": "Unchanged."},
        ]
    }
    saved = {}

    monkeypatch.setattr(llm_jobs, "_load_transcript", lambda fn: tdata)
    monkeypatch.setattr(llm_jobs, "_save_transcript", lambda fn, d: saved.update({"tdata": d}))

    payload = {
        "suggestions": [
            {
                "id": 0,
                "old_value": "The bugdet is approved.",
                "new_value": "The budget is approved.",
                "reason": "Typo",
                "confidence": 0.9,
            },
            {
                "id": 1,
                "old_value": "We meet on Wensday.",
                "new_value": "We meet on Wednesday.",
                "reason": "Typo",
                "confidence": 0.9,
            },
        ],
        "discarded": [],
        "checked": [],
    }
    run_id = db.save_llm_edit_run(
        "a.mp4", payload, model="m", status="pending", media_dir=str(config.media_dir)
    )
    return {"client": app_client, "run_id": run_id, "tdata": tdata, "saved": saved, "db": db}


def _post(client, path, **body):
    return client.post(path, json=body)


def test_a_partial_apply_keeps_the_run_open(run):
    c, run_id = run["client"], run["run_id"]
    r = _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[0])
    assert r.status_code == 200
    d = r.get_json()
    assert d["applied"] == 1
    assert d["status"] == "pending"  # Suggestion 1 is still open
    assert run["tdata"]["segments"][0]["text"] == "The budget is approved."
    assert run["tdata"]["segments"][1]["text"] == "We meet on Wensday."

    # The changelog keeps both entries, one is marked
    runs = run["db"].get_llm_edit_runs("a.mp4", media_dir=str(config.media_dir))
    vs = runs[0]["payload"]["suggestions"]
    assert [v.get("status") for v in vs] == ["applied", None] or [v.get("status") for v in vs] == [
        "applied",
        "open",
    ]


def test_the_second_apply_closes_the_run_and_is_cumulative(run):
    c, run_id = run["client"], run["run_id"]
    _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[0])
    d = _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[1]).get_json()
    assert d["applied"] == 1
    assert d["status"] == "applied"
    runs = run["db"].get_llm_edit_runs("a.mp4", media_dir=str(config.media_dir))
    # Both edits are in the history → undo can revert both
    assert len(runs[0]["payload"]["applied"]) == 2


def test_dismiss_individually_and_completion(run):
    c, run_id = run["client"], run["run_id"]
    d = _post(c, "/api/llm/edits/dismiss", filename="a.mp4", run_id=run_id, ids=[0]).get_json()
    assert d["rejected"] == 1 and d["status"] == "pending"
    # Dismiss second → nothing applied → run is considered rejected
    d = _post(c, "/api/llm/edits/dismiss", filename="a.mp4", run_id=run_id, ids=[1]).get_json()
    assert d["status"] == "rejected"
    assert run["tdata"]["segments"][0]["text"] == "The bugdet is approved."


def test_dismiss_then_apply_yields_applied(run):
    c, run_id = run["client"], run["run_id"]
    _post(c, "/api/llm/edits/dismiss", filename="a.mp4", run_id=run_id, ids=[0])
    d = _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[1]).get_json()
    assert d["status"] == "applied"


def test_apply_without_ids_takes_all_open_ones(run):
    c, run_id = run["client"], run["run_id"]
    d = _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id).get_json()
    assert d["applied"] == 2 and d["status"] == "applied"


def test_undo_after_a_partial_apply(run):
    c, run_id = run["client"], run["run_id"]
    _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[0])
    d = _post(c, "/api/llm/edits/undo", filename="a.mp4", run_id=run_id).get_json()
    assert d["reverted"] == 1
    assert run["tdata"]["segments"][0]["text"] == "The bugdet is approved."
    # After undo, suggestions are decidable again
    runs = run["db"].get_llm_edit_runs("a.mp4", media_dir=str(config.media_dir))
    vs = runs[0]["payload"]["suggestions"]
    assert all((v.get("status") or "open") == "open" for v in vs)


def test_undo_leaves_manually_changed_segments_alone(run):
    c, run_id = run["client"], run["run_id"]
    _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[0])
    run["tdata"]["segments"][0]["text"] = "Edited by hand."
    d = _post(c, "/api/llm/edits/undo", filename="a.mp4", run_id=run_id).get_json()
    assert d["reverted"] == 0 and d["skipped"] == 1
    assert run["tdata"]["segments"][0]["text"] == "Edited by hand."


def test_apply_counts_as_done_when_the_text_already_matches(run):
    """If the target text is already in the segment (previously applied or
    typed by hand), the suggestion must still come out of open status —
    otherwise the run would remain forever non-closable."""
    c, run_id = run["client"], run["run_id"]
    run["tdata"]["segments"][0]["text"] = "The budget is approved."
    d = _post(c, "/api/llm/edits/apply", filename="a.mp4", run_id=run_id, ids=[0]).get_json()
    assert d["applied"] == 0  # nothing to change …
    runs = run["db"].get_llm_edit_runs("a.mp4", media_dir=str(config.media_dir))
    vs = runs[0]["payload"]["suggestions"]
    assert vs[0]["status"] == "applied"  # … still completed
    assert runs[0]["status"] == "pending"  # Suggestion 1 still open


def test_apply_on_an_unknown_run(app_client):
    r = app_client.post("/api/llm/edits/apply", json={"filename": "a.mp4", "run_id": 999999})
    assert r.status_code == 404

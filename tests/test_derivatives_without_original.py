"""Derivatives survive deletion of the transcript.

Background: deleting a transcript used to take its summaries with it,
while translations and correction runs remained in the database but were
no longer accessible. Decision: derivatives always remain
(they are tied to abs_path, not to the transcript) and are visible
even without the original. These tests verify both scenarios.
"""

import pytest

from whisper_tm.database import Database
from whisper_tm.scanner import db_row_to_entry


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


@pytest.fixture
def root(tmp_path):
    w = tmp_path / "media"
    w.mkdir()
    return str(w)


@pytest.fixture
def file_with_derivatives(db, root):
    """Transcript + all three types of derivatives for the same file."""
    rel = "folder/film.mp4"
    db.upsert_transcript(
        rel,
        {
            "filename": rel,
            "language": "de",
            "model": "turbo",
            "complete": True,
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.0,
                    "start_fmt": "0:00:00",
                    "end_fmt": "0:00:01",
                    "text": "Hello",
                }
            ],
        },
        media_dir=root,
    )
    db.replace_summaries(rel, [{"text": "Short summary", "model": "gpt"}], media_dir=root)
    db.replace_translation(
        rel, "English", [{"id": 0, "text": "Hello"}], model="gpt", media_dir=root
    )
    db.save_llm_edit_run(
        rel,
        {"suggestions": [{"id": 0, "new_value": "Hello!"}]},
        model="gpt",
        status="pending",
        media_dir=root,
    )
    return rel


class TestDeleteLeavesDerivativesInPlace:
    def test_summary_survives(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        assert db.delete_transcript(rel, media_dir=root) is True
        assert db.get_summaries(rel, media_dir=root), "Summary was deleted along with transcript"

    def test_translation_survives(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        db.delete_transcript(rel, media_dir=root)
        assert db.get_translation_languages(rel, media_dir=root)
        assert db.get_translation(rel, "English", media_dir=root)

    def test_correction_run_survives(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        db.delete_transcript(rel, media_dir=root)
        assert db.get_llm_edit_runs(rel, media_dir=root)

    def test_transcript_and_segments_are_still_gone(self, db, root, file_with_derivatives):
        """Counter-check: deletion does happen, just only the transcript."""
        rel = file_with_derivatives
        db.delete_transcript(rel, media_dir=root)
        assert db.get_transcript(rel, media_dir=root) is None
        with db.connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 0


class TestListShowsDerivatives:
    """Without a marker, such a file looks like one that was never transcribed
    — then the derivatives would never be found again."""

    def _line(self, db, root, rel):
        from whisper_tm.database import _MEDIA_SCAN_SELECT

        with db.connection() as conn:
            for r in conn.execute(_MEDIA_SCAN_SELECT):
                d = dict(r)
                if d["rel_path"] == rel:
                    return d
        return None

    def test_flag_set_after_deletion(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        db.upsert_media_file(
            rel,
            abs_path=f"{root}/{rel}",
            display_name="film.mp4",
            extension=".mp4",
            is_video=1,
            file_size=1,
            modified_at=1.0,
        )
        db.delete_transcript(rel, media_dir=root)

        line = self._line(db, root, rel)
        assert line is not None, "Test setup: row not found"
        entry = db_row_to_entry(line, root)
        assert entry["status"] == "pending"
        assert entry.get("has_deriv") is True, "List conceals the preserved derivatives"

    def test_no_flag_while_transcript_exists(self, db, root, file_with_derivatives):
        """With transcript, the marker is redundant — it should only
        point to the special case, otherwise every finished file carries it."""
        rel = file_with_derivatives
        db.upsert_media_file(
            rel,
            abs_path=f"{root}/{rel}",
            display_name="film.mp4",
            extension=".mp4",
            is_video=1,
            file_size=1,
            modified_at=1.0,
        )
        entry = db_row_to_entry(self._line(db, root, rel), root)
        assert entry["status"] == "done"
        assert "has_deriv" not in entry

    def test_no_flag_without_derivatives(self, db, root):
        rel = "folder/empty.mp4"
        db.upsert_media_file(
            rel,
            abs_path=f"{root}/{rel}",
            display_name="empty.mp4",
            extension=".mp4",
            is_video=1,
            file_size=1,
            modified_at=1.0,
        )
        entry = db_row_to_entry(self._line(db, root, rel), root)
        assert "has_deriv" not in entry


class TestDeleteSelection:
    """The delete dialog sends WHAT should be deleted. Without specification,
    only the transcript is deleted — otherwise the old silent co-deleter would be back."""

    def test_targeted_summaries_only(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        out = db.delete_derivatives(rel, media_dir=root, summaries=True)
        assert out["summaries"] == 1
        assert db.get_summaries(rel, media_dir=root) in (None, [])
        # The other two remain untouched.
        assert db.get_translation_languages(rel, media_dir=root)
        assert db.get_llm_edit_runs(rel, media_dir=root)
        assert db.get_transcript(rel, media_dir=root) is not None

    def test_targeted_translations_only(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        out = db.delete_derivatives(rel, media_dir=root, translations=True)
        assert out["translations"] == 1
        assert not db.get_translation_languages(rel, media_dir=root)
        assert db.get_summaries(rel, media_dir=root)

    def test_everything_together(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        out = db.delete_derivatives(
            rel, media_dir=root, summaries=True, translations=True, edits=True
        )
        assert out == {"summaries": 1, "translations": 1, "edits": 1}
        assert db.get_summaries(rel, media_dir=root) in (None, [])
        assert not db.get_translation_languages(rel, media_dir=root)
        assert not db.get_llm_edit_runs(rel, media_dir=root)

    def test_nothing_checked_deletes_nothing(self, db, root, file_with_derivatives):
        rel = file_with_derivatives
        assert db.delete_derivatives(rel, media_dir=root) == {
            "summaries": 0,
            "translations": 0,
            "edits": 0,
        }
        assert db.get_summaries(rel, media_dir=root)
        assert db.get_translation_languages(rel, media_dir=root)
        assert db.get_llm_edit_runs(rel, media_dir=root)

    def test_only_targets_intended_root(self, db, tmp_path, file_with_derivatives, root):
        """Same rel_path under two roots: deletion may only affect one
        (docs/PATHS.md) — in deletion, a mistake is not recoverable."""
        rel = file_with_derivatives
        other = str(tmp_path / "media2")
        db.replace_summaries(rel, [{"text": "Foreign", "model": "gpt"}], media_dir=other)

        db.delete_derivatives(rel, media_dir=root, summaries=True)
        assert db.get_summaries(rel, media_dir=other), (
            "Summary of the other root was deleted along with it"
        )


# ── The routes behind (delete dialog calls them) ────────────────────────


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    app = create_app()
    # Read AFTER create_app: before that, extensions.db is still None.
    from whisper_tm import extensions

    real_db = extensions.db
    # A second create_app() in the same process (second test) creates a new
    # Database, but the route modules keep their import-time binding.
    # Without this reconciliation, test setup would write to one and the
    # endpoint would read from the other — in production, there's only one.
    import whisper_tm.api.transcription as tx_api

    monkeypatch.setattr(tx_api, "db", real_db)

    rel = "a.mp4"
    root = str(config.media_dir)
    real_db.upsert_transcript(
        rel,
        {
            "filename": rel,
            "language": "de",
            "model": "turbo",
            "complete": True,
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.0,
                    "start_fmt": "0:00:00",
                    "end_fmt": "0:00:01",
                    "text": "Hello",
                }
            ],
        },
        media_dir=root,
    )
    real_db.replace_summaries(rel, [{"text": "Short", "model": "gpt"}], media_dir=root)
    real_db.replace_translation(
        rel, "English", [{"id": 0, "text": "Hi"}], model="gpt", media_dir=root
    )
    return {"client": app.test_client(), "db": real_db, "rel": rel, "root": root}


def test_counts_what_is_attached(api):
    d = api["client"].get("/api/derivatives/a.mp4").get_json()
    assert d["transcript"] is True
    assert d["summaries"] == 1
    assert d["translations"] == 1


def test_without_selection_only_transcript(api):
    """Old callers don't send `what` — they may not drag anything along."""
    r = api["client"].post("/api/delete_transcript", json={"filename": "a.mp4"})
    assert r.status_code == 200
    assert r.get_json()["deleted"]["transcript"] is True
    d = api["client"].get("/api/derivatives/a.mp4").get_json()
    assert d["transcript"] is False
    assert d["summaries"] == 1 and d["translations"] == 1


def test_selection_is_followed(api):
    r = api["client"].post(
        "/api/delete_transcript", json={"filename": "a.mp4", "what": ["summaries", "translations"]}
    )
    assert r.status_code == 200
    d = api["client"].get("/api/derivatives/a.mp4").get_json()
    assert d["summaries"] == 0 and d["translations"] == 0
    assert d["transcript"] is True, "Transcript was not checked"


def test_summary_does_not_come_back_from_sidecar(api):
    """DB primary, JSON heals — without sidecar deletion it would return immediately."""
    from whisper_tm.core.summarize import load_summaries, save_summaries

    save_summaries("a.mp4", [{"text": "Short", "model": "gpt"}])
    api["client"].post("/api/delete_transcript", json={"filename": "a.mp4", "what": ["summaries"]})
    assert not load_summaries("a.mp4")


def test_the_dialog_speaks_the_same_language_as_the_server():
    """The delete dialog once sent a different field name than the `what` the
    server read — the selection was ignored, and ticking only "Summaries" deleted the
    transcript instead. The dialog is checked against the server here:
    same field name, and only values the server knows."""
    import re
    from pathlib import Path

    js = (
        Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "transcripts.js"
    ).read_text(encoding="utf-8")
    assert "JSON.stringify({filename, what})" in js
    offered = set(re.findall(r"html \+= line\('(\w+)'", js))
    assert offered == {"transcript", "summaries", "translations", "edits"}, offered
    assert "d.transcript" in js, "the dialog reads the server's field name"

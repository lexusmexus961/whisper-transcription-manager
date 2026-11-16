"""
Endpoints must not answer with 500 on unusable path arguments.

Background: since the switch to a single storage location,
``core.helpers.transcript_path_for`` raises a ``ValueError`` when the
identity cannot be derived — instead of falling back to a guessed
location. That is correct, but it left nine call sites in the API with
an unhandled exception:

    POST /api/delete_transcript {"filename": "../../etc/passwd"}   → 500
    GET  /api/transcript/../../etc/passwd                          → 500

Access was correctly denied in the process — no file outside the root
was ever opened. But a stack trace is not an answer. The read sites now
use ``find_transcript()`` (returns ``None``), and the one write site
catches the exception.
"""

from __future__ import annotations

import pytest

from whisper_tm.core.helpers import find_transcript, transcript_path_for


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    (media / "real.mp4").write_bytes(b"x")
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "media_on_start", property(lambda self: media))
    monkeypatch.setattr(
        type(config), "transcript_dir", property(lambda self: media / ".transcripts")
    )

    from flask import Flask

    from whisper_tm.api.transcription import transcription_bp

    a = Flask(__name__)
    a.register_blueprint(transcription_bp)
    a.config["TESTING"] = False  # do NOT propagate errors, respond for real
    return a


UNUSABLE = [
    "../../etc/passwd",
    "../outside.mp4",
    "..",
    "../",
]


class TestNoFiveHundreds:
    @pytest.mark.parametrize("name", UNUSABLE)
    def test_delete_transcript(self, app, name):
        r = app.test_client().post("/api/delete_transcript", json={"filename": name})
        assert r.status_code < 500, f"{name!r} → {r.status_code}"

    @pytest.mark.parametrize("name", UNUSABLE)
    def test_transcript_read(self, app, name):
        r = app.test_client().get(f"/api/transcript/{name}")
        assert r.status_code < 500, f"{name!r} → {r.status_code}"

    @pytest.mark.parametrize("name", UNUSABLE)
    def test_skip(self, app, name):
        r = app.test_client().post("/api/skip", json={"filename": name})
        assert r.status_code < 500, f"{name!r} → {r.status_code}"


class TestPathResolution:
    """The functions underneath — so it stays clear who raises and who doesn't."""

    def test_transcript_path_for_raises_on_an_unusable_path(self, app):
        """No fallback to a guessed location."""
        with pytest.raises(ValueError, match="No transcript path derivable"):
            transcript_path_for("../../etc/passwd")

    def test_find_transcript_returns_none_instead_of_raising(self, app):
        """For "does it exist?" None is the correct answer, not an exception.

        Every read site in the API therefore uses this function.
        """
        assert find_transcript("../../etc/passwd") is None

    def test_find_transcript_finds_a_real_file(self, app, tmp_path):
        from whisper_tm.config import config

        tp = config.media_dir / ".transcripts" / "real.mp4.json"
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text("{}", encoding="utf-8")
        assert find_transcript("real.mp4") == tp

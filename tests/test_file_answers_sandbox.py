"""A document from the library never runs as the app's own page.

/api/video/<path> served any file inline: an .html (or .svg) in the media
folder — a share others write to, an upload — opened from there ran its
script on the app's origin, with the whole API at its disposal. Every
answer is nosniff now, and the answers that carry files are sandboxed:
a document shown from them runs no script and has no origin of the app.
The app's own pages carry a Content-Security-Policy of their own.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    own = tmp_path / "config.yaml"
    own.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "media")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "own.db")
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "x.html").write_text("<script>alert(1)</script>", encoding="utf-8")
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


def test_file_answers_are_sandboxed(client):
    c, _ = client
    r = c.get("/api/video/x.html")
    assert r.status_code == 200 and b"<script>" in r.data  # served — but inert
    for path in (
        "/api/video/x.html",
        "/api/download/x.html",
        "/api/clip/audio",
        "/api/media-search/thumb",
    ):
        r = c.get(path)
        assert r.headers.get("Content-Security-Policy") == "sandbox", path
        assert r.headers.get("X-Content-Type-Options") == "nosniff", path


def test_the_app_itself_is_not_sandboxed(client):
    """Its pages get a policy of their own: scripts, styles, pictures and
    requests from the app only, no plugins, no foreign <base> or form
    target — and no frame-ancestors, dashboards may show it in a frame."""
    c, _ = client
    r = c.get("/")
    csp = r.headers.get("Content-Security-Policy", "")
    assert "sandbox" not in csp
    rules = dict(part.strip().split(" ", 1) for part in csp.split(";"))
    assert rules["default-src"] == "'self'" and rules["connect-src"] == "'self'"
    assert rules["script-src"] == "'self' 'unsafe-inline'"
    assert rules["object-src"] == "'none'" and rules["base-uri"] == "'self'"
    assert "frame-ancestors" not in rules
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    health = c.get("/api/health")
    assert health.headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" not in health.headers  # JSON needs none

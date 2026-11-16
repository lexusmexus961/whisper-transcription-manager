"""
The stored YouTube cookies.txt.

``#HttpOnly_`` lines are cookies, not comments: the Netscape format marks
HttpOnly cookies this way, and browser exports write YouTube's login cookies
so. The status counted every line that starts with "#" out and undercounted
the file.

The file holds credentials. It was written first and restricted to 0600 only
afterwards — readable for others in between, and for good if the chmod
failed. Now it is created with 0600.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from flask import Flask

COOKIES = "\n".join(
    [
        "# Netscape HTTP Cookie File",
        "# This is a generated file! Do not edit.",
        "",
        ".youtube.com\tTRUE\t/\tTRUE\t1790000000\tPREF\tf6=40000000",
        ".youtube.com\tTRUE\t/\tTRUE\t1790000000\tVISITOR_INFO1_LIVE\tabc",
        "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1790000000\tHSID\tAaBbCc",
        "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1790000000\tSSID\tDdEeFf",
        "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1790000000\t__Secure-3PSID\tGgHh",
    ]
)


@pytest.fixture
def cookies(tmp_path, monkeypatch):
    """The cookie routes with the file in the test's own folder — never the
    real data directory."""
    from whisper_tm.api.media import media_bp
    from whisper_tm.core import ytdl

    target = tmp_path / "data" / "youtube_cookies.txt"
    monkeypatch.setattr(ytdl, "cookies_path", lambda: target)
    app = Flask(__name__)
    app.register_blueprint(media_bp)
    return app.test_client(), target


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_httponly_lines_count_as_cookies(cookies):
    client, target = cookies
    assert client.post("/api/youtube/cookies", json={"text": COOKIES}).status_code == 200
    d = client.get("/api/youtube/cookies").get_json()
    assert d["exists"] is True
    assert d["lines"] == 5


def test_the_file_is_created_private(cookies, monkeypatch):
    """With every chmod switched off, the mode it was created with shows —
    a write followed by chmod(0600) would be left readable here."""
    client, target = cookies
    monkeypatch.setattr(os, "chmod", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "chmod", lambda *args, **kwargs: None)
    previous = os.umask(0o022)
    try:
        r = client.post("/api/youtube/cookies", json={"text": COOKIES})
    finally:
        os.umask(previous)
    assert r.status_code == 200
    assert _mode(target) == 0o600
    assert [p.name for p in target.parent.iterdir()] == [target.name]


def test_an_old_readable_file_is_replaced_by_a_private_one(cookies):
    client, target = cookies
    target.parent.mkdir(parents=True)
    target.write_text("# Netscape HTTP Cookie File\nold\n", encoding="utf-8")
    target.chmod(0o644)

    assert client.post("/api/youtube/cookies", json={"text": COOKIES}).status_code == 200
    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == COOKIES + "\n"


def test_a_failed_save_leaves_no_temp_file(cookies, monkeypatch):
    client, target = cookies

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full)
    r = client.post("/api/youtube/cookies", json={"text": COOKIES})
    assert r.status_code == 500
    assert "No space left" in r.get_json()["error"]
    assert list(target.parent.iterdir()) == []

"""
/api/navigate_to_file stays inside the allowed trees.

It searched first and checked the root only at the very end: a hint like
``/home/<user>/x`` walked the home folder (and "/" the whole filesystem),
and with ``debug: true`` the 403 still carried a debug log that named the
file it had found outside — an oracle for any file on the machine.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    own = tmp_path / "config.yaml"
    own.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    media = tmp_path / "media"
    (media / "below").mkdir(parents=True)
    (media / "below" / "clip.mp4").write_bytes(b"x" * 2048)
    config._data.setdefault("directories", {})["media"] = str(media)
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "own.db")
    config._data.setdefault("security", {})["allowed_roots"] = [str(media)]
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


def test_no_search_and_no_trace_outside_the_allowed_roots(client, monkeypatch):
    c, tmp = client
    secret_dir = tmp / "private" / "deep"
    secret_dir.mkdir(parents=True)
    (secret_dir / "id_ed25519").write_text("key", encoding="utf-8")

    import whisper_tm.scanner.filesystem as fs

    searched = []
    real = fs.iter_files_named

    def spy(root, name):
        searched.append(str(root))
        return real(root, name)

    monkeypatch.setattr(fs, "iter_files_named", spy)
    r = c.post(
        "/api/navigate_to_file",
        json={"filename": "id_ed25519", "media_dir": str(tmp / "private" / "x"), "debug": True},
    )
    body = r.get_json()
    assert body["found"] is False
    text = r.get_data(as_text=True)
    assert str(secret_dir) not in text
    assert "exists=True" not in text
    assert all(str(tmp / "private") not in s for s in searched)


def test_a_full_path_outside_is_refused_without_a_debug_log(client):
    c, tmp = client
    outside = tmp / "private"
    outside.mkdir()
    (outside / "notes.mp3").write_bytes(b"x")
    r = c.post(
        "/api/navigate_to_file", json={"filename": str(outside / "notes.mp3"), "debug": True}
    )
    assert r.status_code == 403
    body = r.get_json()
    assert body["found"] is False
    assert "debug_log" not in body


def test_inside_still_works(client):
    c, tmp = client
    r = c.post(
        "/api/navigate_to_file",
        json={"filename": "clip.mp4", "media_dir": str(tmp / "media" / "below")},
    )
    body = r.get_json()
    assert body["found"] is True

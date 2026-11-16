"""Choosing the library folder in the app.

The only allowed tree of a fresh install is the folder found at start
(``~/Videos``). The path field and Settings refused every other folder, for
everybody — yet the setup guide and setup.sh say "choose it in the app", and
a NAS mount is the first thing many will enter.

The owner may choose anywhere now: a request from this machine itself, or
any request once the login is on (the rule of the settings only this machine
may change). A request from another machine without the login stays inside
the allowed trees, as before.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config
from whisper_tm.core import security

#: An address that is not this machine's (TEST-NET-3)
STRANGER = {"REMOTE_ADDR": "203.0.113.9"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    own = tmp_path / "config.yaml"
    own.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    (tmp_path / "videos").mkdir()
    for sub in ("talks", "music"):
        (tmp_path / "nas" / sub).mkdir(parents=True)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "videos")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "own.db")
    config._resolve_directories()
    config._ensure_directories()
    # The library as it stood when the config was loaded
    monkeypatch.setattr(config, "_media_on_start", (tmp_path / "videos").resolve())
    monkeypatch.setattr(security, "_owner_roots", [])
    from whisper_tm.api import main

    monkeypatch.setattr(main, "_saved_library", "")
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


def test_this_machine_may_open_a_folder_anywhere(client):
    c, tmp = client
    r = c.post("/api/set_directory", json={"path": str(tmp / "nas")})
    assert r.status_code == 200 and r.get_json()["directory"] == str(tmp / "nas")
    assert config.media_dir == (tmp / "nas").resolve()


def test_a_folder_chosen_in_the_path_field_is_saved_for_the_next_start(client):
    c, tmp = client
    c.post("/api/set_directory", json={"path": str(tmp / "nas"), "chosen": True})
    assert str(tmp / "nas") in (tmp / "config.yaml").read_text(encoding="utf-8")


def test_a_jump_the_page_makes_by_itself_is_not_saved(client):
    # After an upload the page opens the upload folder — which may lie
    # outside the library. It must not become the library by that.
    c, tmp = client
    before = (tmp / "config.yaml").read_text(encoding="utf-8")
    assert c.post("/api/set_directory", json={"path": str(tmp / "nas")}).status_code == 200
    assert (tmp / "config.yaml").read_text(encoding="utf-8") == before
    assert security.root_allowed(tmp / "nas" / "talks")  # open for this run all the same


def test_going_back_to_the_first_library_is_saved_too(client):
    # Otherwise a look into another tree made that tree the library after a restart
    c, tmp = client
    c.post("/api/set_directory", json={"path": str(tmp / "nas"), "chosen": True})
    c.post("/api/set_directory", json={"path": str(tmp / "nas" / "talks"), "chosen": True})
    assert (
        c.post("/api/set_directory", json={"path": str(tmp / "videos"), "chosen": True}).status_code
        == 200
    )
    import yaml

    saved = yaml.safe_load((tmp / "config.yaml").read_text(encoding="utf-8"))
    assert saved["directories"]["media"] == str(tmp / "videos")


def test_a_step_inside_the_library_saves_nothing(client):
    c, tmp = client
    (tmp / "videos" / "sub").mkdir()
    before = (tmp / "config.yaml").read_text(encoding="utf-8")
    r = c.post("/api/set_directory", json={"path": str(tmp / "videos" / "sub"), "chosen": True})
    assert r.status_code == 200
    assert (tmp / "config.yaml").read_text(encoding="utf-8") == before


def test_the_chosen_tree_stays_open_after_a_step_into_it(client):
    # Without that, opening nas/talks shut nas/music out for everybody
    c, tmp = client
    c.post("/api/set_directory", json={"path": str(tmp / "nas")})
    c.post("/api/set_directory", json={"path": str(tmp / "nas" / "talks")})
    assert security.root_allowed(tmp / "nas" / "music")
    r = c.post(
        "/api/set_directory", json={"path": str(tmp / "nas" / "music")}, environ_overrides=STRANGER
    )
    assert r.status_code == 200


def test_another_machine_without_the_login_stays_inside(client):
    c, tmp = client
    r = c.post("/api/set_directory", json={"path": str(tmp / "nas")}, environ_overrides=STRANGER)
    assert r.status_code == 403 and "allowed directories" in r.get_json()["error"]
    assert (
        c.post("/api/set_directory", json={"path": "/"}, environ_overrides=STRANGER).status_code
        == 403
    )
    assert config.media_dir == (tmp / "videos").resolve()
    r = c.get("/api/browse", query_string={"path": str(tmp / "nas")}, environ_overrides=STRANGER)
    assert r.status_code == 403
    r = c.put(
        "/api/settings",
        json={"directories": {"media": str(tmp / "nas")}},
        environ_overrides=STRANGER,
    )
    assert r.get_json()["ok"] is False and config.media_dir == (tmp / "videos").resolve()


def test_a_request_a_proxy_passed_on_is_not_this_machine(client):
    c, tmp = client
    r = c.post(
        "/api/set_directory",
        json={"path": str(tmp / "nas")},
        headers={"X-Forwarded-For": "203.0.113.9"},
    )
    assert r.status_code == 403


def test_the_folder_browser_goes_anywhere_for_this_machine(client):
    c, tmp = client
    r = c.get("/api/browse", query_string={"path": str(tmp / "nas")})
    names = [d["name"] for d in r.get_json()["dirs"]]
    assert r.status_code == 200 and names == ["..", "music", "talks"]


def test_settings_take_a_new_library_from_this_machine(client):
    c, tmp = client
    r = c.put("/api/settings", json={"directories": {"media": str(tmp / "nas")}})
    assert r.get_json()["ok"] is True, r.get_json()
    assert config.media_dir == (tmp / "nas").resolve()
    assert security.root_allowed(tmp / "nas" / "talks")


def test_a_folder_that_is_not_there_is_refused_for_everyone(client):
    c, tmp = client
    r = c.post("/api/set_directory", json={"path": str(tmp / "missing")})
    assert r.status_code == 400 and config.media_dir == (tmp / "videos").resolve()
    assert not security.root_allowed(tmp / "missing")


def test_with_the_login_on_the_person_logged_in_is_the_owner(client, monkeypatch):
    c, _tmp = client
    from whisper_tm.core import auth

    app = c.application
    with app.test_request_context(environ_base=STRANGER):
        assert not security.owner_request()
        monkeypatch.setattr(auth, "is_auth_enabled", lambda: True)
        assert security.owner_request()
    assert not security.owner_request()  # outside a request: nobody


def test_the_path_field_says_when_a_folder_was_chosen():
    from pathlib import Path

    js = Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js"
    assert "changeDir({chosen: true})" in (js / "pathpicker.js").read_text(encoding="utf-8")
    directory = (js / "directory.js").read_text(encoding="utf-8")
    assert "chosen: opts.chosen === true" in directory
    assert "await changeDir();" in (js / "upload.js").read_text(
        encoding="utf-8"
    )  # the jump: not chosen

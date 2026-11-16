"""/api/ui/prefs and /api/ui/recent: UI preferences kept on the server
(pins, the rail's fold, Home's groups, last tabs, export options and
presets, recently opened recordings)."""

from __future__ import annotations

import json

import pytest
from flask import Flask


@pytest.fixture
def env(tmp_path, monkeypatch):
    from whisper_tm.api import ui_prefs
    from whisper_tm.config import config

    data = tmp_path / "data"
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: data))
    monkeypatch.setattr(ui_prefs, "root_allowed", lambda p: str(p).startswith(str(lib) + "/"))
    app = Flask(__name__)
    app.register_blueprint(ui_prefs.ui_prefs_bp)
    return app.test_client(), data / "ui_prefs.json", lib


def _put(client, body):
    return client.put("/api/ui/prefs", data=json.dumps(body), content_type="application/json")


def test_nothing_stored_is_an_empty_object(env):
    client, _, _ = env
    r = client.get("/api/ui/prefs")
    assert r.status_code == 200 and r.get_json() == {}


def test_a_change_is_stored_and_merged(env):
    client, path, _ = env
    assert (
        _put(client, {"pins": ["voices:map", "find:audio"], "rail_folded": True}).status_code == 200
    )
    r = _put(client, {"home_groups": {"find": False}})
    assert r.get_json() == {
        "pins": ["voices:map", "find:audio"],
        "rail_folded": True,
        "home_groups": {"find": False},
    }
    assert json.loads(path.read_text()) == r.get_json()
    # a later read (another browser, after a restart) sees the same
    assert client.get("/api/ui/prefs").get_json() == r.get_json()


def test_unknown_keys_are_dropped(env):
    client, path, _ = env
    r = _put(client, {"pins": ["queue"], "theme": "pink", "__proto__": {"x": 1}})
    assert r.status_code == 200 and r.get_json() == {"pins": ["queue"]}
    assert "theme" not in path.read_text()


@pytest.mark.parametrize(
    "body, word",
    [
        ({"pins": ["a", "b", "c", "d", "e", "f", "g"]}, "at most 6"),
        ({"pins": ["Voices Map"]}, "not a place"),
        ({"pins": "voices"}, "list"),
        ({"rail_folded": "yes"}, "true or false"),
        ({"home_groups": {"find": "open"}}, "true or false"),
        ({"last_tab": {"voices": "<b>"}}, "window and tab"),
        ({"export_options": {"x": "y" * 9000}}, "larger than"),
        ({"export_presets": [{"options": {}}]}, "needs a name"),
        ({"recent": [{"path": "relative/a.mp3"}]}, "absolute path"),
    ],
)
def test_a_value_that_does_not_fit_is_refused_with_its_key(env, body, word):
    client, path, _ = env
    r = _put(client, body)
    assert r.status_code == 400 and word in r.get_json()["error"]
    assert not path.exists(), "nothing is written on a refusal"


def test_duplicate_pins_count_once(env):
    client, _, _ = env
    r = _put(client, {"pins": ["queue", "queue", "voices:map"]})
    assert r.get_json()["pins"] == ["queue", "voices:map"]


def test_the_body_is_size_limited(env):
    client, _, _ = env
    r = _put(client, {"export_options": {"x": "y" * 70_000}})
    assert r.status_code == 413


def test_not_an_object_is_refused(env):
    client, _, _ = env
    assert _put(client, ["pins"]).status_code == 400
    assert client.put("/api/ui/prefs", data="{", content_type="application/json").status_code == 400


def test_a_broken_file_starts_from_defaults(env):
    client, path, _ = env
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert client.get("/api/ui/prefs").get_json() == {}
    # and one bad stored key costs only that key
    path.write_text(json.dumps({"pins": ["queue"], "rail_folded": "nope"}))
    assert client.get("/api/ui/prefs").get_json() == {"pins": ["queue"]}


def test_recently_opened_moves_to_the_front_without_duplicates(env):
    client, _, lib = env
    a, b = lib / "a.mp3", lib / "b.mp3"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    for p in (a, b, a):
        assert client.post("/api/ui/recent", json={"path": str(p)}).status_code == 200
    recent = client.get("/api/ui/prefs").get_json()["recent"]
    assert [r["path"] for r in recent] == [str(a), str(b)]
    assert recent[0]["at"] >= recent[1]["at"] > 0


def test_recently_opened_keeps_twenty(env):
    client, _, lib = env
    for i in range(25):
        f = lib / f"{i}.mp3"
        f.write_bytes(b"x")
        client.post("/api/ui/recent", json={"path": str(f)})
    recent = client.get("/api/ui/prefs").get_json()["recent"]
    assert len(recent) == 20 and recent[0]["path"].endswith("/24.mp3")


def test_a_deleted_recording_drops_out(env):
    client, _, lib = env
    a = lib / "a.mp3"
    a.write_bytes(b"x")
    client.post("/api/ui/recent", json={"path": str(a)})
    a.unlink()
    assert client.get("/api/ui/prefs").get_json()["recent"] == []


def test_recently_opened_only_takes_files_in_the_library(env):
    client, _, _ = env
    assert client.post("/api/ui/recent", json={"path": "/etc/passwd"}).status_code == 403
    assert client.post("/api/ui/recent", json={"path": "a.mp3"}).status_code == 400
    assert client.post("/api/ui/recent", json={}).status_code == 400


def test_a_stored_path_outside_the_library_is_never_answered(env):
    client, path, _ = env
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"recent": [{"path": "/etc/passwd", "at": 1}]}))
    assert client.get("/api/ui/prefs").get_json()["recent"] == []


def test_a_recent_time_is_a_finite_number(monkeypatch):
    # 1e999 was kept and came back as "Infinity", which no browser reads as JSON
    from whisper_tm.api import ui_prefs

    monkeypatch.setattr(ui_prefs, "_good_path", lambda p: True)
    out = ui_prefs._recent(
        [
            {"path": "/lib/a.mp3", "at": float("inf")},
            {"path": "/lib/b.mp3", "at": 5},
            {"path": "/lib/c.mp3", "at": True},
        ]
    )
    assert [r["at"] for r in out] == [0.0, 5.0, 0.0]
    json.dumps(out, allow_nan=False)

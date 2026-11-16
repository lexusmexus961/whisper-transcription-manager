"""
Settings API hardening.

* A whole file-only section sent as a scalar ({"auth": false}) slipped past
  the "auth." prefix check and replaced the login section.
* Moving llm.endpoint / embedding.endpoint kept the stored key: the next
  request sent the saved API key to whatever server was entered.
* A reset set the config.yaml-only sections (llama, logging) back to their
  defaults, where the page can't restore them.
* /api/mcp/access handed the MCP token to anyone on the LAN, and the LAN
  could switch mcp.require_token off.
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
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


LAN = {"REMOTE_ADDR": "203.0.113.7"}  # not one of this machine's addresses


# ── #4 file-only sections ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "section",
    [
        "auth",
        "security",
        "server",
        "llama",
        "logging",
        "database",
        "mcp",
        "embedding",
        "audio_analysis",
    ],
)
def test_a_file_only_section_cannot_be_replaced_whole(client, section):
    c, _ = client
    before = config.get(section)
    r = c.put("/api/settings", json={section: False})
    assert r.get_json()["ok"] is False
    assert config.get(section) == before


@pytest.mark.parametrize(
    "key,value", [("extra_args", "--path /"), ("local_model", "/etc/passwd"), ("host", "0.0.0.0")]
)
def test_the_rerankers_start_command_is_file_only(client, key, value):
    """The same llama-server start as llama.* and embedding.local_extra_args."""
    c, _ = client
    before = config.get("rerank", key)
    r = c.put("/api/settings", json={"rerank": {key: value}})
    assert r.get_json()["ok"] is False
    assert config.get("rerank", key) == before


def test_the_audio_index_path_is_file_only(client):
    """A database written where the request says: the file's to decide —
    the rest of the section saves as usual."""
    c, tmp = client
    r = c.put("/api/settings", json={"audio_analysis": {"path": str(tmp / "x.db")}})
    assert r.get_json()["ok"] is False
    assert not config.get("audio_analysis", "path")
    r = c.put("/api/settings", json={"audio_analysis": {"parallel_files": 3}})
    assert r.get_json()["ok"] is True
    assert config.get("audio_analysis", "parallel_files") == 3


def test_an_ordinary_section_cannot_be_replaced_by_a_scalar(client):
    c, _ = client
    before = config.get("scanner")
    r = c.put("/api/settings", json={"scanner": 0})
    assert r.get_json()["ok"] is False
    assert config.get("scanner") == before


def test_ordinary_keys_still_save(client):
    c, _ = client
    r = c.put("/api/settings", json={"scanner": {"watch_interval_sec": 77}})
    assert r.get_json()["ok"] is True
    assert config.get("scanner", "watch_interval_sec") == 77


# ── #5 key follows its endpoint ───────────────────────────────────────────


@pytest.mark.parametrize("section", ["llm", "embedding"])
def test_moving_the_endpoint_forgets_the_key(client, section):
    c, _ = client
    config.set(section, "endpoint", "https://api.example.com/v1")
    config.set(section, "api_key", "sk-real-secret")
    r = c.put(
        "/api/settings",
        json={section: {"endpoint": "http://attacker.example/v1", "api_key": "__UNCHANGED__"}},
    )
    assert r.get_json()["ok"] is True
    assert config.get(section, "endpoint") == "http://attacker.example/v1"
    assert config.get(section, "api_key") == ""


@pytest.mark.parametrize("section", ["llm", "embedding"])
def test_same_endpoint_keeps_the_key_and_a_new_key_is_taken(client, section):
    c, _ = client
    config.set(section, "endpoint", "https://api.example.com/v1")
    config.set(section, "api_key", "sk-real-secret")
    c.put(
        "/api/settings",
        json={section: {"endpoint": "https://api.example.com/v1", "api_key": "__UNCHANGED__"}},
    )
    assert config.get(section, "api_key") == "sk-real-secret"
    c.put(
        "/api/settings",
        json={section: {"endpoint": "https://other.example.com/v1", "api_key": "sk-new"}},
    )
    assert config.get(section, "api_key") == "sk-new"


# ── #9 reset ──────────────────────────────────────────────────────────────


def test_reset_keeps_config_file_only_sections(client):
    c, _ = client
    config._data.setdefault("llama", {})["ctx_size"] = 12345
    config._data.setdefault("logging", {})["level"] = "DEBUG"
    config._data.setdefault("embedding", {})["local_extra_args"] = "--flash-attn"
    r = c.post("/api/settings/reset")
    assert r.status_code == 200
    assert config.get("llama", "ctx_size") == 12345
    assert config.get("logging", "level") == "DEBUG"
    assert config.get("embedding", "local_extra_args") == "--flash-attn"


# ── #11 MCP access ────────────────────────────────────────────────────────


def test_mcp_token_is_not_shown_to_the_lan(client):
    c, _ = client
    d = c.get("/api/mcp/access", environ_base=LAN).get_json()
    assert d["token"] == "" and d["token_hidden"] is True
    assert c.post("/api/mcp/access/renew", environ_base=LAN).status_code == 403
    # nor this machine's folder layout
    assert d["file"] == d["python"] == d["package_root"] == d["config"] == ""


def test_mcp_token_is_shown_on_this_machine(client):
    c, _ = client
    d = c.get("/api/mcp/access").get_json()  # 127.0.0.1
    assert d["token"] and d["token_hidden"] is False
    assert d["file"] and d["python"] and d["package_root"]


@pytest.mark.parametrize("header", ["X-Forwarded-For", "Forwarded", "X-Real-IP"])
def test_behind_a_reverse_proxy_nobody_is_this_machine(client, header):
    """The proxy's 127.0.0.1 is everyone's address there."""
    c, _ = client
    d = c.get("/api/mcp/access", headers={header: "203.0.113.7"}).get_json()
    assert d["token"] == "" and d["token_hidden"] is True


def test_the_lan_cannot_switch_the_mcp_token_off(client):
    c, _ = client
    r = c.put(
        "/api/settings", json={"mcp": {"require_token": False, "host": "0.0.0.0"}}, environ_base=LAN
    )
    body = r.get_json()
    assert body["ok"] is False
    assert config.get("mcp", "require_token") is True
    assert config.get("mcp", "host") == "127.0.0.1"
    # Unchanged values sent back by the page are fine
    r = c.put(
        "/api/settings",
        json={"mcp": {"require_token": "true", "host": "127.0.0.1"}},
        environ_base=LAN,
    )
    assert r.get_json()["ok"] is True
    # From this machine it can be changed
    r = c.put("/api/settings", json={"mcp": {"require_token": False}})
    assert r.get_json()["ok"] is True
    assert config.get("mcp", "require_token") is False


@pytest.mark.parametrize(
    "section,key,value",
    [
        # the MCP server's way to the app, its token and its player links: moved
        # elsewhere, the token goes there and the tools talk to an impostor
        ("mcp", "base_url", "http://203.0.113.9:9860"),
        ("mcp", "listen_base_url", "http://203.0.113.9:9860"),
        ("mcp", "api_token", "someone-elses"),
        # the Python environment Images & sounds starts, and its folders
        ("media_search", "environment", "/tmp/evil/bin/python"),
        ("media_search", "path", "/tmp/evil/media.db"),
        ("media_search", "cache_dir", "/tmp/evil"),
        # the app's folders: written to, or emptied
        ("directories", "uploads", "~/.ssh"),
        ("directories", "clips", "~/.config/autostart"),
        ("directories", "work", "~/.local/lib"),
    ],
)
def test_the_lan_cannot_move_what_the_app_runs_or_writes(client, section, key, value):
    c, tmp_path = client
    before = config.get(section, key)
    r = c.put("/api/settings", json={section: {key: value}}, environ_base=LAN)
    body = r.get_json()
    assert body["ok"] is False and "this machine's own browser" in " ".join(body["errors"])
    assert config.get(section, key) == before
    # The page sends every field back: unchanged, it passes
    r = c.put("/api/settings", json={section: {key: before}}, environ_base=LAN)
    assert r.get_json()["ok"] is True


def test_this_machine_can_still_set_a_folder(client):
    c, tmp_path = client
    clips = tmp_path / "my-clips"
    r = c.put("/api/settings", json={"directories": {"clips": str(clips)}})
    assert r.get_json()["ok"] is True and config.get("directories", "clips") == str(clips)


def test_debug_mode_never_switches_on_the_werkzeug_debugger():
    """--debug is the app's logging and timing. Passed on to app.run it
    opened Werkzeug's interactive debugger (a Python console behind a PIN)
    on every address the server listens on."""
    import re
    from pathlib import Path

    main = (Path(__file__).resolve().parents[1] / "whisper_tm" / "__main__.py").read_text(
        encoding="utf-8"
    )
    call = re.search(r"app\.run\((.*?)\)", main, re.S).group(1)
    assert re.search(r"\bdebug=False\b", call), call


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("embedding", "local_host", "0.0.0.0"),
        ("embedding", "local_port", 1),
        ("embedding", "local_model", "/etc/passwd"),
        ("media_search", "model", "someone/else"),
        ("media_search", "host", "0.0.0.0"),
        ("media_search", "port", 1),
        ("rag", "path", "/tmp/elsewhere.db"),
        ("database", "backup_path", "/tmp/elsewhere"),
    ],
)
def test_the_lan_cannot_move_services_or_where_things_are_written(client, section, key, value):
    """They start programs on an address of the request's choosing, load a
    file as a model, or write the vector store or every backup somewhere."""
    c, _ = client
    before = config.get(section, key)
    body = c.put("/api/settings", json={section: {key: value}}, environ_base=LAN).get_json()
    assert body["ok"] is False and config.get(section, key) == before
    assert c.put("/api/settings", json={section: {key: before}}, environ_base=LAN).get_json()["ok"]


@pytest.mark.parametrize(
    "args,ok",
    [
        ("--limit-rate 2M --no-playlist", True),
        ("--format bestaudio --retries=3", True),
        ("--proxy http://198.51.100.9:8080", False),
        ("--no-check-certificates", False),
        ("--legacy-server-connect", False),
        ("https://example.invalid/other", False),  # a bare word: an extra download
        ("--no-playlist https://example.invalid/x", False),
        ("--limit-rate", False),  # its value is missing
    ],
)
def test_ytdlp_options_from_the_page(args, ok):
    from whisper_tm.api.settings import _check_ytdlp_args

    assert (_check_ytdlp_args(args) == "") is ok, _check_ytdlp_args(args)


def test_a_password_inside_an_endpoint_is_not_shown(client):
    """GET /api/settings masked secrets by key name only: a user:password in
    llm.endpoint came back in clear."""
    c, _ = client
    config._data.setdefault("llm", {})["endpoint"] = "http://u:hunter2@192.0.2.9:8080/v1"
    config._data["llm"]["api_key"] = "sk-kept"
    got = c.get("/api/settings").get_json()
    shown = got["config"]["llm"]["endpoint"]
    assert "hunter2" not in shown and shown.startswith("http://u:")
    # Sent back unchanged: the stored password stays
    assert c.put("/api/settings", json={"llm": {"endpoint": shown}}).get_json()["ok"]
    assert config.get("llm", "endpoint") == "http://u:hunter2@192.0.2.9:8080/v1"
    assert config.get("llm", "api_key") == "sk-kept"  # not taken for a moved endpoint
    # Changed around the mask: refused, nothing written
    body = c.put(
        "/api/settings", json={"llm": {"endpoint": shown.replace("8080", "8081")}}
    ).get_json()
    assert body["ok"] is False and config.get("llm", "endpoint").endswith(":8080/v1")

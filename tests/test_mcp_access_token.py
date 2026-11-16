"""The access token of the MCP network services.

**Why.** MCP over HTTP and the OpenAPI service answer anyone who reaches
their port. Listening on 0.0.0.0 with `allow_writes` on, they let anyone
on the network transcribe, download and summarize. Only the OpenAPI service
could ask for a token, one typed in by hand; MCP over HTTP asked for
nothing.

**What it does now.** Both services expect the same token, by default
(`mcp.require_token`). It is made on first use and kept in `mcp_token` next
to config.yaml, readable only by the owner — config.yaml gets shared, a
credential doesn't belong there. The settings page shows it and makes a new
one; switching the requirement or renewing the token applies at once,
without restarting the services.
"""

from __future__ import annotations

import os
import stat

import pytest

from whisper_tm import mcp_server as mcp
from whisper_tm.config import config


@pytest.fixture
def own_config(tmp_path, monkeypatch):
    """A config file of this test, so the token lands in tmp_path."""
    path = tmp_path / "config.yaml"
    path.write_text("mcp:\n  require_token: true\n", encoding="utf-8")
    monkeypatch.setattr(config, "_config_path", path)
    monkeypatch.setitem(config._data.setdefault("mcp", {}), "require_token", True)
    monkeypatch.setitem(config._data["mcp"], "token", "")
    monkeypatch.setattr(mcp, "_token_kept", ("", ""))
    return tmp_path


class TestTheToken:
    def test_made_on_first_use_next_to_the_config(self, own_config):
        token = mcp.access_token()
        assert len(token) >= 32
        assert mcp.token_file() == own_config / "mcp_token"
        assert (own_config / "mcp_token").read_text().strip() == token

    def test_only_the_owner_can_read_it(self, own_config):
        mcp.access_token()
        mode = stat.S_IMODE(os.stat(own_config / "mcp_token").st_mode)
        assert mode == 0o600, oct(mode)

    def test_the_same_token_after_a_restart(self, own_config, monkeypatch):
        first = mcp.access_token()
        monkeypatch.setattr(mcp, "_token_kept", ("", ""))  # a new process
        assert mcp.access_token() == first

    def test_renewing_replaces_it_at_once(self, own_config):
        old = mcp.access_token()
        new = mcp.access_token(renew=True)
        assert new != old
        assert mcp.access_token() == new
        assert (own_config / "mcp_token").read_text().strip() == new

    def test_a_token_of_your_own_wins(self, own_config, monkeypatch):
        monkeypatch.setitem(config._data["mcp"], "token", "  mine  ")
        assert mcp.access_token() == "mine"
        assert not (own_config / "mcp_token").exists()

    def test_required_by_default(self):
        from pathlib import Path

        import yaml

        d = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert d["mcp"]["require_token"] is True
        assert d["mcp"]["token"] == ""

    def test_switched_off_nothing_is_required(self, own_config, monkeypatch):
        assert mcp.required_token() == mcp.access_token()
        monkeypatch.setitem(config._data["mcp"], "require_token", False)
        monkeypatch.setitem(config._data["mcp"], "host", "127.0.0.1")
        assert mcp.required_token() == ""

    def test_switched_off_only_on_this_machine(self, own_config, monkeypatch):
        """On the network every machine there — and through DNS rebinding
        every web page — could run the tools: the token stays."""
        monkeypatch.setitem(config._data["mcp"], "require_token", False)
        monkeypatch.setitem(config._data["mcp"], "host", "0.0.0.0")
        assert mcp.required_token() == mcp.access_token()


class TestCarryingIt:
    @pytest.mark.parametrize(
        "headers, ok",
        [
            ({"authorization": "Bearer tok"}, True),
            ({"authorization": "bearer tok"}, True),
            ({"authorization": "Bearer  tok "}, True),
            ({"x-api-token": "tok"}, True),
            ({"authorization": "Bearer wrong"}, False),
            ({"authorization": "tok"}, False),
            ({}, False),
        ],
    )
    def test_bearer_or_header(self, headers, ok):
        assert mcp.carries_token(headers, "tok") is ok

    def test_nothing_needed_without_a_token(self):
        assert mcp.carries_token({}, "") is True


class TestMcpOverHttp:
    """The gate around FastMCP's own app — the service that asked for
    nothing at all before."""

    INIT = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    ACCEPT = {"Accept": "application/json, text/event-stream"}

    @pytest.fixture
    def client(self, own_config):
        pytest.importorskip("fastmcp")
        from starlette.testclient import TestClient

        app = mcp.build_server().http_app(middleware=mcp.http_middleware())
        with TestClient(app) as c:
            yield c

    def test_without_the_token_refused(self, client):
        r = client.post("/mcp", json=self.INIT, headers=self.ACCEPT)
        assert r.status_code == 401
        assert r.headers.get("www-authenticate") == "Bearer"
        assert "Settings > MCP server" in r.json()["error"]

    def test_a_wrong_token_refused(self, client):
        r = client.post(
            "/mcp", json=self.INIT, headers={**self.ACCEPT, "Authorization": "Bearer nope"}
        )
        assert r.status_code == 401

    def test_with_the_token_it_answers(self, client):
        r = client.post(
            "/mcp",
            json=self.INIT,
            headers={**self.ACCEPT, "Authorization": f"Bearer {mcp.access_token()}"},
        )
        assert r.status_code == 200

    def test_a_renewed_token_applies_without_a_restart(self, client):
        old = mcp.access_token()
        new = mcp.access_token(renew=True)
        head = {**self.ACCEPT}
        assert (
            client.post(
                "/mcp", json=self.INIT, headers={**head, "Authorization": f"Bearer {old}"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/mcp", json=self.INIT, headers={**head, "Authorization": f"Bearer {new}"}
            ).status_code
            == 200
        )

    def test_switching_the_requirement_off_applies_at_once(self, client, monkeypatch):
        monkeypatch.setitem(config._data["mcp"], "require_token", False)
        monkeypatch.setitem(config._data["mcp"], "host", "127.0.0.1")
        assert (
            client.post(
                "/mcp", json=self.INIT, headers={**self.ACCEPT, "Host": "localhost:5400"}
            ).status_code
            == 200
        )

    def test_without_a_token_a_rebound_name_is_refused(self, client, monkeypatch):
        """DNS rebinding: a page of evil.example whose name now points here
        sends Origin and Host both its own."""
        monkeypatch.setitem(config._data["mcp"], "require_token", False)
        monkeypatch.setitem(config._data["mcp"], "host", "127.0.0.1")
        r = client.post(
            "/mcp",
            json=self.INIT,
            headers={
                **self.ACCEPT,
                "Host": "evil.example:5400",
                "Origin": "http://evil.example:5400",
            },
        )
        assert r.status_code == 403 and "rebinding" in r.json()["error"]
        # An address is fine: a page can't rebind to it
        assert (
            client.post(
                "/mcp", json=self.INIT, headers={**self.ACCEPT, "Host": "127.0.0.1:5400"}
            ).status_code
            == 200
        )

    def test_a_cors_preflight_passes_the_gate(self, client):
        # A browser sends it without credentials; it runs nothing
        assert client.options("/mcp").status_code != 401


class TestOpenApi:
    @pytest.fixture
    def client(self, own_config, monkeypatch):
        pytest.importorskip("fastmcp")
        from starlette.testclient import TestClient

        class Fake:
            def fetch(self, path, **kw):
                return {"people": []}

        monkeypatch.setattr(mcp, "CONN", Fake())
        return TestClient(mcp.build_openapi_app(mcp.build_server(), mcp.required_token))

    def test_the_live_token_is_required(self, client):
        assert client.post("/list_people").status_code == 401
        ok = client.post("/list_people", headers={"Authorization": f"Bearer {mcp.access_token()}"})
        assert ok.status_code == 200

    def test_the_spec_names_the_bearer_scheme(self, client):
        spec = client.get("/openapi.json").json()
        assert spec["security"] == [{"bearer": []}]

    def test_switched_off_open(self, client, monkeypatch):
        monkeypatch.setitem(config._data["mcp"], "require_token", False)
        monkeypatch.setitem(config._data["mcp"], "host", "127.0.0.1")
        assert client.post("/list_people", headers={"Host": "localhost:5500"}).status_code == 200
        r = client.post("/list_people", headers={"Host": "evil.example:5500"})
        assert r.status_code == 403


class TestAutostart:
    """The app's own start path uses the gate for both services."""

    @pytest.fixture
    def runs(self, own_config, monkeypatch):
        pytest.importorskip("fastmcp")
        from whisper_tm import app as app_mod

        started = []

        class FakeThread:
            def __init__(self, target=None, args=(), name="", daemon=False, **kw):
                started.append({"name": name, "args": args})

            def start(self):
                pass

        monkeypatch.setattr(app_mod.threading, "Thread", FakeThread)
        m = config._data["mcp"]
        monkeypatch.setitem(m, "autostart_http", True)
        monkeypatch.setitem(m, "autostart_openapi", True)
        app_mod.start_mcp_services()
        return {s["name"]: s for s in started}

    def test_the_token_exists_before_any_client_came(self, runs, own_config):
        assert (own_config / "mcp_token").exists()

    def test_mcp_over_http_runs_behind_the_gate(self, runs, monkeypatch):
        import fastmcp

        seen = {}
        monkeypatch.setattr(fastmcp.FastMCP, "run", lambda self, *a, **kw: seen.update(kw))
        runs["mcp-http"]["args"][1]()
        gates = [m for m in seen.get("middleware", []) if m.cls is mcp.AccessGate]
        assert gates and gates[0].kwargs["token_of"] is mcp.required_token

    def test_openapi_asks_for_the_live_token(self, runs, monkeypatch):
        import uvicorn
        from starlette.testclient import TestClient

        seen = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app))
        runs["mcp-openapi"]["args"][1]()
        c = TestClient(seen["app"])

        class Fake:  # no call leaves the test
            def fetch(self, path, **kw):
                return {"people": []}

        monkeypatch.setattr(mcp, "CONN", Fake())
        assert c.post("/list_people").status_code == 401
        # …the token of now, not the one from the start
        mcp.access_token(renew=True)
        head = {"Authorization": f"Bearer {mcp.access_token()}"}
        assert c.post("/list_people", headers=head).status_code == 200

    def test_the_old_per_service_token_is_gone(self):
        assert "openapi_token" not in mcp.load_settings(reload=False)


class TestSettingsApi:
    @pytest.fixture
    def client(self, own_config, monkeypatch):
        from whisper_tm.app import create_app

        app = create_app()
        # create_app loads the session's config again — point it back here
        monkeypatch.setattr(config, "_config_path", own_config / "config.yaml")
        monkeypatch.setattr(mcp, "_token_kept", ("", ""))
        return app.test_client()

    def test_the_page_gets_token_state_and_paths(self, client, own_config):
        d = client.get("/api/mcp/access").get_json()
        assert d["token"] == (own_config / "mcp_token").read_text().strip()
        assert d["required"] is True and d["source"] == "file"
        assert {s["key"] for s in d["services"]} == {"http", "openapi"}
        assert all(isinstance(s["listening"], bool) for s in d["services"])
        # What the stdio set-up needs: this Python, this code, this config
        assert d["python"] and d["package_root"] and d["config"]

    def test_renew_makes_a_new_one(self, client):
        old = client.get("/api/mcp/access").get_json()["token"]
        d = client.post("/api/mcp/access/renew").get_json()
        assert d["ok"] and d["token"] != old
        assert client.get("/api/mcp/access").get_json()["token"] == d["token"]

    def test_renew_replaces_a_token_of_your_own(self, client, monkeypatch):
        monkeypatch.setattr(config, "save", lambda *a, **kw: None)
        monkeypatch.setitem(config._data["mcp"], "token", "mine")
        assert client.get("/api/mcp/access").get_json()["source"] == "config"
        d = client.post("/api/mcp/access/renew").get_json()
        assert d["token"] != "mine" and config.get("mcp", "token") == ""

    def test_the_settings_page_cannot_set_it(self, client):
        # config.yaml only — like the login and the allowed folders
        r = client.put("/api/settings", json={"mcp": {"token": "chosen"}})
        assert config.get("mcp", "token") != "chosen"
        assert r.status_code in (200, 400)

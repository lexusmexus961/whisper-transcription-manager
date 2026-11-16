"""The same tools as an OpenAPI service — for clients without MCP.

Open WebUI integrates tools exclusively as an "OpenAPI Tool Server": it
fetches a specification at ``/openapi.json`` and calls the endpoints
described there. It does not understand MCP.

The usual way to do this is ``mcpo``, a proxy in front of the MCP server —
a second process, a second install, a second source of bugs. Here it is
built in, and from the SAME tool list: the specification is generated at
runtime from ``server.list_tools()``. A new tool therefore appears
automatically in both worlds.

That is exactly what these tests check: that there is no second list that
can drift apart, that the specification has the shape a client expects, and
that a call ultimately does the same thing as it would via MCP.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from whisper_tm import mcp_server as mcp

pytest.importorskip("fastmcp")
pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402


@pytest.fixture
def server():
    return mcp.build_server()


@pytest.fixture
def tools(server):
    return asyncio.run(server.list_tools())


@pytest.fixture
def client(server, monkeypatch):
    """The OpenAPI app without a real counterpart — what's tested is the
    service, not the app behind it."""
    # (no token: only this machine's names and addresses are answered)
    return TestClient(mcp.build_openapi_app(server), base_url="http://127.0.0.1")


class TestSpecification:
    def test_every_tool_has_an_endpoint(self, client, tools):
        """The one thing that justifies this path over a proxy:
        there is only ONE tool list."""
        d = client.get("/openapi.json").json()
        assert set(d["paths"]) == {f"/{t.name}" for t in tools}

    def test_the_description_comes_along(self, client, tools):
        """A model chooses based on this. Without a description, a tool
        is either never used or used wrongly."""
        d = client.get("/openapi.json").json()
        for t in tools:
            op = d["paths"][f"/{t.name}"]["post"]
            assert op["operationId"] == t.name
            assert op["summary"]
            assert (t.description or "").strip()[:30] in op["description"]

    def test_the_parameters_are_in_the_request_body(self, client):
        d = client.get("/openapi.json").json()
        s = d["paths"]["/transcribe"]["post"]["requestBody"]["content"]["application/json"][
            "schema"
        ]
        assert s["required"] == ["file"]
        assert "language" in s["properties"] and "wait" in s["properties"]

    def test_a_tool_without_parameters_has_no_request_body(self, client):
        """Some clients still show an empty schema with `required: false`
        as a mandatory field — then the form has a box that nobody can
        fill in."""
        d = client.get("/openapi.json").json()
        assert "requestBody" not in d["paths"]["/list_people"]["post"]

    def test_no_ref_in_the_schemas(self, tools):
        """The parameter schemas pass unchanged from FastMCP into the
        specification. That only holds as long as they contain no `$defs`:
        in OpenAPI 3.1 a `$ref` is resolved against the root of the WHOLE
        document, not against the sub-schema — an embedded `#/$defs/X`
        would therefore point at nothing, and the client would get a tool
        with unusable parameters.

        Today none of the tools produce anything like that (only str/int/
        float/bool/list[str]). Whoever introduces their own type will
        break this — and then knows that the schemas need to be hoisted
        into `components`.
        """
        for t in tools:
            raw = json.dumps(t.parameters or {})
            assert "$defs" not in raw and "$ref" not in raw, t.name

    def test_the_groups_are_complete(self, tools):
        """A new tool without a group lands silently under "other"."""
        missing = [t.name for t in tools if t.name not in mcp.GROUPS]
        assert not missing, missing

    def test_the_conversation_tools_offer_their_choices(self, client):
        """kind, sentences and sort are choices a client can offer as a list
        — inline, without $defs — and the service says the statistics exist."""
        d = client.get("/openapi.json").json()
        op = d["paths"]["/find_conversations"]["post"]
        s = op["requestBody"]["content"]["application/json"]["schema"]
        assert s["properties"]["kind"]["enum"] == [
            "any",
            "detailed",
            "lively",
            "interview",
            "monologue",
            "conversation",
            "group",
        ]
        assert s["properties"]["sentences"]["enum"] == ["any", "longer", "shorter"]
        assert {"type": "array", "items": {"type": "string"}} in s["properties"]["people"]["anyOf"]
        # every filter may be left out
        assert op["requestBody"]["required"] is False and op["tags"] == ["speakers"]
        assert "Conversation statistics" in d["info"]["description"]

    def test_it_is_valid_openapi_3_1(self, client):
        d = client.get("/openapi.json").json()
        assert d["openapi"].startswith("3.1")
        assert d["info"]["title"] and d["info"]["version"]
        for path, entry in d["paths"].items():
            assert set(entry) == {"post"}, path
            assert "200" in entry["post"]["responses"]


class TestCalls:
    def test_a_tool_can_be_called(self, client, monkeypatch):
        called = {}

        class Fake:
            def fetch(self, path, **kw):
                called["path"] = path
                return {"people": [{"name": "Alice", "voices": 3}]}

        monkeypatch.setattr(mcp, "CONN", Fake())
        r = client.post("/list_people")
        assert r.status_code == 200
        assert r.json() == {"people": [{"name": "Alice", "voices": 3}]}
        assert called["path"] == "/api/speaker/people"

    def test_the_parameters_arrive(self, client, monkeypatch):
        seen = {}

        class Fake:
            def fetch(self, path, **kw):
                seen.update(kw)
                return {"total": 0, "files": 0, "results": []}

        monkeypatch.setattr(mcp, "CONN", Fake())
        r = client.post("/search", json={"text": "hello world", "limit": 7})
        assert r.status_code == 200
        assert seen["q"] == "hello world" and seen["limit"] == 7

    def test_a_conversation_query_arrives_in_the_apis_words(self, client, monkeypatch):
        seen = {}

        class Fake:
            time_limit = 30.0

            def fetch(self, path, **kw):
                seen["path"] = path
                seen.update(kw)
                return {
                    "total": 0,
                    "results": [],
                    "media_dir": "/media",
                    "baseline": {},
                    "status": {"enabled": True, "counted": 5, "stale": 0, "running": False},
                }

        monkeypatch.setattr(mcp, "CONN", Fake())
        r = client.post(
            "/find_conversations",
            json={"people": ["Alice", "Bob"], "min_minutes": 60, "sort": "speaker_changes"},
        )
        assert r.status_code == 200 and r.json()["total"] == 0
        assert seen["path"] == "/api/conversations" and seen["people"] == ["Alice", "Bob"]
        assert seen["min_sec"] == 3600 and seen["sort"] == "rate" and seen["match"] == "all"

    def test_an_error_becomes_400_with_a_reason(self, client):
        """400 and not 500: the call was wrong or the thing isn't possible.
        The model should read the reason and try differently, instead of
        reporting a server error."""
        r = client.post("/search", json={"text": "a"})
        assert r.status_code == 400
        assert "two characters" in r.json()["error"]

    def test_an_unknown_tool_is_404(self, client):
        r = client.post("/does_not_exist", json={})
        assert r.status_code == 404

    def test_the_write_protection_applies_here_too(self, client, monkeypatch):
        """Otherwise the OpenAPI path would be the open back door to
        exactly what `allow_writes: false` is meant to prevent."""
        monkeypatch.setattr(mcp, "WRITING_ALLOWED", False)
        r = client.post("/transcribe", json={"file": "a.mkv"})
        assert r.status_code == 400
        assert "allow_writes" in r.json()["error"]

    def test_a_broken_body_does_not_become_a_server_error(self, client):
        r = client.post("/search", content=b"no json", headers={"content-type": "application/json"})
        assert r.status_code == 400


class TestAccess:
    def test_without_a_token_everything_is_open(self, server):
        c = TestClient(mcp.build_openapi_app(server, token=""), base_url="http://127.0.0.1")
        assert c.get("/openapi.json").status_code == 200
        assert "security" not in c.get("/openapi.json").json()

    def test_with_a_token_it_is_required(self, server, monkeypatch):
        c = TestClient(mcp.build_openapi_app(server, token="secret"))
        assert c.post("/list_people").status_code == 401
        assert c.post("/list_people", headers={"Authorization": "Bearer wrong"}).status_code == 401

    def test_with_the_right_token_it_works(self, server, monkeypatch):
        class Fake:
            def fetch(self, path, **kw):
                return {"people": []}

        monkeypatch.setattr(mcp, "CONN", Fake())
        c = TestClient(mcp.build_openapi_app(server, token="secret"))
        r = c.post("/list_people", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200

    def test_without_a_token_a_foreign_page_cannot_call_tools(self, server, monkeypatch):
        """CORS is open for Open WebUI; without a token any web page could
        otherwise drive the tools from the user's browser."""

        class Fake:
            def fetch(self, path, **kw):
                return {"people": []}

        monkeypatch.setattr(mcp, "CONN", Fake())
        c = TestClient(mcp.build_openapi_app(server, token=""), base_url="http://127.0.0.1")
        assert c.post("/list_people", headers={"Origin": "http://evil.example"}).status_code == 403
        assert c.post("/list_people", headers={"Origin": "http://127.0.0.1"}).status_code == 200
        assert c.post("/list_people").status_code == 200
        t = TestClient(mcp.build_openapi_app(server, token="secret"))
        assert (
            t.post(
                "/list_people",
                headers={"Origin": "http://evil.example", "Authorization": "Bearer secret"},
            ).status_code
            == 200
        )

    def test_the_token_is_in_the_specification(self, server):
        """Otherwise the client never sends it along and looks for the
        bug elsewhere."""
        d = TestClient(mcp.build_openapi_app(server, token="x")).get("/openapi.json").json()
        assert d["security"] == [{"bearer": []}]
        assert d["components"]["securitySchemes"]["bearer"]["scheme"] == "bearer"


class TestEntryPoint:
    def test_the_root_points_the_way(self, client):
        """Whoever types the address into the browser should not stare
        at a 404, but read how to proceed."""
        d = client.get("/").json()
        assert d["openapi"] == "/openapi.json"
        assert "transcribe" in d["tools"]

    def test_the_switches_exist(self):
        """`--openapi` must be in the parser, not only in the docs —
        the lesson from `--stdio`."""
        import argparse
        import inspect

        source = inspect.getsource(mcp.main)
        i = source.index("p = argparse.ArgumentParser(")
        j = source.index("a = p.parse_args(")
        space = {"argparse": argparse}
        exec(inspect.cleandoc(source[i:j]), space)  # noqa: S102
        a = space["p"].parse_args(["--openapi", "--port", "5400"])
        assert a.openapi and a.port == 5400
        with pytest.raises(SystemExit):
            space["p"].parse_args(["--openapi", "--http"])

    def test_the_config_knows_the_values(self):
        from pathlib import Path

        import yaml

        d = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert "openapi_port" in d["mcp"]
        # One token for both network services, required by default
        assert d["mcp"]["require_token"] is True and d["mcp"]["token"] == ""
        assert "openapi_token" not in d["mcp"]

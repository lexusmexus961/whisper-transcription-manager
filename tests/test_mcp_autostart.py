"""Starting the MCP server's network services together with the app.

**The reason.** The MCP server used to be started purely by hand.
Since it logs nothing during normal operation, its absence goes unnoticed —
you only notice when a tool doesn't respond. So it had to be started by
hand again after every machine reboot.

**What gets started along with it and what doesn't.** Only the two NETWORK
services (OpenAPI and MCP over HTTP). The stdio path doesn't need this:
there, the AI client starts the server itself and talks over standard
input.

**Why threads and not separate processes.** The MCP server talks to the app
over HTTP — it can run in the same process and then just calls itself. That
saves all the process management and ensures the services come up together
with the app.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def started_ones(monkeypatch):
    """Intercept threads instead of actually opening ports.

    A test that starts network services occupies ports on the machine of
    whoever runs it — and collides with a running application.
    """
    from whisper_tm import app as app_mod

    runs = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False, **kw):
            self._entry = {"name": name, "daemon": daemon, "target": target, "args": args}
            runs.append(self._entry)

        def start(self):
            pass

    monkeypatch.setattr(app_mod.threading, "Thread", FakeThread)
    return runs


class TestTheSwitch:
    def test_nothing_starts_by_default(self, started_ones, monkeypatch):
        """The safe default: whoever configures nothing opens no port.

        This matters for the test suite too — `create_app()` runs in many
        tests, and none of them may incidentally spin up a service.
        """
        from whisper_tm import app as app_mod

        assert app_mod.start_mcp_services() == []
        assert started_ones == []

    def test_openapi_is_started(self, started_ones, monkeypatch):
        from whisper_tm import app as app_mod
        from whisper_tm.config import config

        monkeypatch.setitem(config._data.setdefault("mcp", {}), "autostart_openapi", True)
        out = app_mod.start_mcp_services()
        assert len(out) == 1 and "OpenAPI" in out[0], out
        assert started_ones[0]["name"] == "mcp-openapi"

    def test_http_is_started(self, started_ones, monkeypatch):
        from whisper_tm import app as app_mod
        from whisper_tm.config import config

        monkeypatch.setitem(config._data.setdefault("mcp", {}), "autostart_http", True)
        out = app_mod.start_mcp_services()
        assert len(out) == 1 and "HTTP" in out[0], out
        assert started_ones[0]["name"] == "mcp-http"

    def test_both_run_at_the_same_time(self, started_ones, monkeypatch):
        """They sit on different ports — that is the reason these are two
        separate switches instead of an either-or."""
        from whisper_tm import app as app_mod
        from whisper_tm.config import config

        m = config._data.setdefault("mcp", {})
        monkeypatch.setitem(m, "autostart_openapi", True)
        monkeypatch.setitem(m, "autostart_http", True)
        assert len(app_mod.start_mcp_services()) == 2
        assert {g["name"] for g in started_ones} == {"mcp-openapi", "mcp-http"}


class TestEachOnItsOwnPort:
    """This only surfaced once both switches were on at the same time —
    up to then they never had been.

    Both services read the same variable `port`. A closure, however, does
    not capture a value but the variable: by the time the OpenAPI thread
    actually ran, the second branch had long since overwritten `port` with
    the MCP port. OpenAPI thus occupied 5399, the MCP service wanted the
    same port and died with "address already in use" — while the log said
    "OpenAPI on 127.0.0.1:5400", i.e. the opposite of what happened.
    Whoever addressed the MCP server got "Unknown tool: mcp" from the
    OpenAPI service.
    """

    @pytest.fixture
    def both(self, started_ones, monkeypatch):
        from whisper_tm import app as app_mod
        from whisper_tm.config import config

        m = config._data.setdefault("mcp", {})
        monkeypatch.setitem(m, "autostart_openapi", True)
        monkeypatch.setitem(m, "autostart_http", True)
        monkeypatch.setitem(m, "port", 5399)
        monkeypatch.setitem(m, "openapi_port", 5400)
        messages = app_mod.start_mcp_services()
        return messages, {g["name"]: g for g in started_ones}

    def test_the_message_names_two_different_ports(self, both):
        messages, _ = both
        assert any("5400" in z for z in messages), messages
        assert any("5399" in z for z in messages), messages

    def test_every_service_binds_the_port_from_its_message(self, both):
        """The actual proof: not what gets reported, but what the thread
        actually uses when it starts running."""
        _, threads = both
        bound = {}

        for name, key in (("mcp-openapi", "port"), ("mcp-http", "port")):
            entry = threads[name]
            labelling, target = entry["args"]
            # The target calls uvicorn.run resp. server.run — we intercept
            # both and just record which port comes through.
            bound[name] = _port_of_run(target)

        assert bound["mcp-openapi"] == 5400, bound
        assert bound["mcp-http"] == 5399, bound
        assert bound["mcp-openapi"] != bound["mcp-http"]


def _port_of_run(target) -> int:
    """Run the lambda once and intercept which port it starts its server
    with."""
    pytest.importorskip("fastmcp")
    uvicorn = pytest.importorskip("uvicorn")

    seen = {}

    def _remember(*a, **kw):
        seen["port"] = kw.get("port")

    real_uvicorn, real_run = uvicorn.run, None
    uvicorn.run = _remember
    try:
        # `server.run` belongs to the FastMCP instance captured in the
        # lambda — hence via the type, not via the module.
        import fastmcp

        real_run = fastmcp.FastMCP.run
        fastmcp.FastMCP.run = lambda self, *a, **kw: _remember(*a, **kw)
        target()
    finally:
        uvicorn.run = real_uvicorn
        if real_run is not None:
            import fastmcp

            fastmcp.FastMCP.run = real_run
    return seen.get("port")

    def test_they_do_not_linger_on_shutdown(self, started_ones, monkeypatch):
        """Daemon threads: otherwise the app would hang on shutdown because
        a uvicorn keeps running in the background."""
        from whisper_tm import app as app_mod
        from whisper_tm.config import config

        monkeypatch.setitem(config._data.setdefault("mcp", {}), "autostart_openapi", True)
        app_mod.start_mcp_services()
        assert started_ones[0]["daemon"] is True


class TestItDoesNotToppleTheServer:
    """The MCP server is an accessory. Whatever goes wrong here must not
    take the application down with it — otherwise you'd trade a missing
    service for an application that doesn't even start anymore."""

    def test_without_fastmcp_only_a_warning(self, started_ones, monkeypatch, caplog):
        import builtins

        from whisper_tm import app as app_mod
        from whisper_tm.config import config

        monkeypatch.setitem(config._data.setdefault("mcp", {}), "autostart_openapi", True)
        real = builtins.__import__

        def _no_fastmcp(name, *a, **kw):
            if name == "whisper_tm.mcp_server":
                raise ImportError("fastmcp missing")
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_fastmcp)
        assert app_mod.start_mcp_services() == []
        assert started_ones == []

    def test_an_error_while_building_is_caught(self, started_ones, monkeypatch):
        from whisper_tm import app as app_mod
        from whisper_tm import mcp_server
        from whisper_tm.config import config

        monkeypatch.setitem(config._data.setdefault("mcp", {}), "autostart_openapi", True)

        def _broken(*a, **kw):
            raise RuntimeError("no server buildable")

        monkeypatch.setattr(mcp_server, "build_server", _broken)
        assert app_mod.start_mcp_services() == []
        assert started_ones == []


class TestCompleteness:
    """Project rule: no setting without a default, an effect, and a UI.

    A switch that only exists in the config, nobody finds; one that only
    exists in the UI does nothing.
    """

    @pytest.mark.parametrize(
        "key", ["autostart_openapi", "autostart_http", "openapi_port", "require_token", "token"]
    )
    def test_present_in_default_code_and_interface(self, key):
        yaml_text = (REPO / "config.default.yaml").read_text(encoding="utf-8")
        py_text = (REPO / "whisper_tm" / "app.py").read_text(encoding="utf-8")
        mcp_text = (REPO / "whisper_tm" / "mcp_server.py").read_text(encoding="utf-8")
        js_text = "".join(
            (REPO / "whisper_tm" / "static" / "js" / name).read_text(encoding="utf-8")
            for name in ("settings.js", "settings_pages.js", "settings_tools.js")
        )

        assert f"{key}:" in yaml_text, "missing in config.default.yaml"
        assert f'"{key}"' in py_text or f'"{key}"' in mcp_text, "not read anywhere"
        assert f"mcp.{key}" in js_text, "missing on the settings page"

    def test_the_default_is_off(self):
        """Ports do not open by themselves, just because someone updated."""
        from whisper_tm.config import config

        assert config.get("mcp", "autostart_openapi", default=False) is False
        assert config.get("mcp", "autostart_http", default=False) is False


class TestTheConnectionHoldsToo:
    """The bug the first version had.

    `build_server()` alone is not enough. The tools access `CONN` at
    runtime; without a connection every single one answers "No connection
    configured." The service then runs, dutifully serves its spec
    (`/openapi.json` → 200), and still can't do anything.

    The tests above did not catch this, because they intercept the threads
    and thereby only check THAT something gets started. It only surfaced
    in the functional test against the running service. This test closes
    the gap.
    """

    def test_the_tools_have_a_connection(self, started_ones, monkeypatch):
        from whisper_tm import app as app_mod
        from whisper_tm import mcp_server
        from whisper_tm.config import config

        monkeypatch.setattr(mcp_server, "CONN", None, raising=False)
        monkeypatch.setitem(config._data.setdefault("mcp", {}), "autostart_openapi", True)
        app_mod.start_mcp_services()
        assert mcp_server.CONN is not None, (
            "the service runs without a connection — every tool reports 'No connection configured.'"
        )
        assert mcp_server.CONN.base.startswith("http")

    def test_and_the_remaining_controls_as_well(self, started_ones, monkeypatch):
        """`allow_writes` and the limits come from the same source —
        if only the connection were there, the service would run with the
        module defaults instead of what is in the settings."""
        from whisper_tm import app as app_mod
        from whisper_tm import mcp_server
        from whisper_tm.config import config

        m = config._data.setdefault("mcp", {})
        monkeypatch.setitem(m, "autostart_openapi", True)
        monkeypatch.setitem(m, "max_files_per_call", 7)
        monkeypatch.setattr(mcp_server, "MAX_FILES", -1, raising=False)
        app_mod.start_mcp_services()
        assert mcp_server.MAX_FILES == 7

    def test_both_start_paths_share_the_same_place(self):
        """`main()` and the autostart must not do this separately — otherwise
        one path drifts away from the other, and you only notice once a
        tool goes silent."""
        import inspect

        from whisper_tm import app as app_mod
        from whisper_tm import mcp_server

        assert "set_runtime" in inspect.getsource(mcp_server.main)
        assert "set_runtime" in inspect.getsource(app_mod.start_mcp_services)


def test_shipped_defaults_keep_mcp_closed():
    """As shipped, no MCP service starts and an MCP client may only read."""
    import yaml

    shipped = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(encoding="utf-8")
    )
    mcp = shipped["mcp"]
    assert mcp["autostart_http"] is False
    assert mcp["autostart_openapi"] is False
    assert mcp["allow_writes"] is False

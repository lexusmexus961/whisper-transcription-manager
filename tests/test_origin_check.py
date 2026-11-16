"""
Cross-site requests.

Without a login every endpoint answers anyone who reaches the port, and a
browser on the LAN reaches it from any page it shows. A body-less POST —
a "simple request", no CORS preflight — from a foreign page to
/api/db/delete deleted the database. State-changing requests from a browser
now have to come from the app's own page or from the browser extension.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config
from whisper_tm.core.security import origin_problem

# ── the rule ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "origin,host",
    [
        ("http://localhost:5077", "localhost:5077"),
        ("http://192.0.2.20:5077", "192.0.2.20:5077"),
        ("http://nas-box:5077", "NAS-BOX:5077"),
        ("https://wx.example.org", "wx.example.org"),  # default port 443
        ("chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd", "192.0.2.20:5077"),
    ],
)
def test_own_page_and_the_extension_pass(origin, host, monkeypatch):
    from whisper_tm.core import security

    # This machine as the user's: LAN address 192.0.2.20, host name nas-box;
    # wx.example.org is the trusted name of a reverse proxy
    monkeypatch.setattr(security, "_own_addresses", lambda: {"127.0.0.1", "::1", "192.0.2.20"})
    monkeypatch.setitem(security._own_names_cache, "names", frozenset({"localhost", "nas-box"}))
    assert origin_problem("POST", origin, host, trusted=["https://wx.example.org"]) == ""


def test_requests_without_origin_pass():
    """curl, scripts, the MCP server: no browser, no foreign page."""
    assert origin_problem("POST", None, "localhost:5077") == ""
    assert origin_problem("DELETE", "", "localhost:5077", "same-origin") == ""


@pytest.mark.parametrize(
    "origin,host",
    [
        ("http://evil.example", "192.0.2.20:5077"),
        ("http://192.0.2.20:8080", "192.0.2.20:5077"),  # other port
        ("http://localhost:5077", "192.0.2.20:5077"),
        ("null", "localhost:5077"),
    ],
)
def test_foreign_origins_are_refused(origin, host):
    assert origin_problem("POST", origin, host)


def test_cross_site_fetch_without_origin_is_refused():
    assert origin_problem("POST", None, "localhost:5077", "cross-site")
    # …unless it is the extension
    assert (
        origin_problem(
            "POST",
            "chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd",
            "localhost:5077",
            "cross-site",
        )
        == ""
    )


def test_reading_is_never_refused():
    assert origin_problem("GET", "http://evil.example", "localhost:5077") == ""


# ── GETs that start work ──────────────────────────────────────────────────
#
# An <img src=".../api/rag/build?rebuild=1"> on any page emptied the
# semantic index and started hours of GPU work: GETs were never checked.


@pytest.mark.parametrize(
    "path",
    [
        "/api/rag/build",
        "/api/index",
        "/api/scan_stream",
        "/api/scan_entry/a.mp4",
        "/api/db/export",
        "/api/media-search",
        "/api/rag/search",
        # ffmpeg (or a model) at work for an <img> or <audio>
        "/api/clip/waveform",
        "/api/find/frame",
        "/api/find/clip",
        "/api/media-search/thumb/12",
        "/api/speaker/sample/7",
        "/api/voices/extract/clip/r1/0",
    ],
)
@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_a_foreign_page_cannot_start_work_with_a_get(path, site):
    from whisper_tm.core.security import get_effect_problem

    assert get_effect_problem("GET", path, site)


@pytest.mark.parametrize(
    "site,origin",
    [
        ("same-origin", None),
        ("none", None),
        (None, None),
        ("cross-site", "chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd"),
    ],
)
def test_the_own_page_a_script_and_the_extension_still_can(site, origin):
    from whisper_tm.core.security import get_effect_problem

    assert get_effect_problem("GET", "/api/rag/build", site, origin) == ""


def test_other_gets_are_untouched():
    """Clips and stills are embedded by other pages (Open WebUI plays them)."""
    from whisper_tm.core.security import get_effect_problem

    for path in ("/api/clip/audio", "/api/rag/build/progress", "/api/health"):
        assert get_effect_problem("GET", path, "cross-site") == ""
    assert get_effect_problem("POST", "/api/rag/build", "cross-site") == ""


# ── in the app ────────────────────────────────────────────────────────────


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


def test_foreign_page_cannot_delete_the_database(client):
    c, tmp = client
    r = c.post(
        "/api/db/delete", headers={"Origin": "http://evil.example", "Sec-Fetch-Site": "cross-site"}
    )
    assert r.status_code == 403
    assert "error" in r.get_json()
    assert (tmp / "own.db").exists()


def test_foreign_page_cannot_empty_the_semantic_index(client):
    c, _ = client
    r = c.get("/api/rag/build?rebuild=1", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403 and "own page" in r.get_json()["error"]


def test_foreign_page_cannot_reset_settings(client):
    c, _ = client
    r = c.post("/api/settings/reset", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_own_page_still_works(client):
    c, _ = client
    r = c.post(
        "/api/autoscan/scan",
        headers={"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"},
    )
    assert r.status_code != 403
    r = c.post(
        "/api/setup/done",
        headers={
            "Origin": "chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    assert r.status_code == 200
    # Any other extension is a foreign origin
    r = c.post(
        "/api/setup/done",
        headers={"Origin": "chrome-extension://abcdefghijklmnop", "Sec-Fetch-Site": "cross-site"},
    )
    assert r.status_code == 403


# ── DNS rebinding: reading too ─────────────────────────────────────────────
#
# A foreign page whose name points at this machine read every GET of the API
# as its own — only state-changing requests were checked. The Host must be
# one of this machine's names or addresses for any /api/ request.


@pytest.fixture
def machine(monkeypatch):
    from whisper_tm.core import security

    monkeypatch.setattr(security, "_own_addresses", lambda: {"127.0.0.1", "::1", "192.0.2.20"})
    monkeypatch.setitem(
        security._own_names_cache, "names", frozenset({"localhost", "nas-box", "nas-box.local"})
    )
    return security


@pytest.mark.parametrize(
    "host",
    [
        "localhost:5077",
        "127.0.0.1:5077",
        "192.0.2.20:5077",
        "NAS-BOX:5077",
        "nas-box.local",
        "[::1]:5077",
    ],
)
def test_the_machines_own_names_read(machine, host):
    assert machine.host_problem("/api/search", None, host) == ""


def test_a_foreign_name_reads_nothing(machine):
    assert machine.host_problem("/api/search", None, "evil.example:5077")
    assert machine.host_problem(
        "/api/system/history", "http://evil.example:5077", "evil.example:5077"
    )
    # Pages and static files are not the API
    assert machine.host_problem("/", None, "evil.example:5077") == ""


def test_trusted_names_and_the_extension_read(machine):
    trusted = ["https://wx.example.org"]
    assert machine.host_problem("/api/search", None, "wx.example.org", trusted) == ""
    assert (
        machine.host_problem("/api/search", "https://wx.example.org", "backend:5077", trusted) == ""
    )
    assert (
        machine.host_problem(
            "/api/search",
            "chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd",
            "media.example:5077",
        )
        == ""
    )
    assert machine.host_problem("/api/search", "chrome-extension://abc", "media.example:5077")


def test_a_rebound_page_cannot_read_the_api(client, monkeypatch):
    from whisper_tm.core import security

    monkeypatch.setitem(security._own_names_cache, "names", frozenset({"localhost"}))
    c, _ = client
    r = c.get("/api/search/speakers", headers={"Host": "evil.example:5077"})
    assert r.status_code == 403 and "evil.example" in r.get_json()["error"]
    assert c.get("/api/search/speakers").status_code == 200


# ── An IP address is never a rebinding ────────────────────────────────────
#
# In a container the server sees only its own addresses: opened through the
# host's LAN address, every API request was refused until that address went
# into security.trusted_origins. Rebinding needs a name the attacker
# controls — a request sent to an IP address passes, a name still doesn't.


@pytest.mark.parametrize("host", ["198.51.100.7:9860", "198.51.100.7", "[2001:db8::7]:9860"])
def test_any_ip_address_reads(machine, host):
    assert machine.host_problem("/api/search", None, host) == ""


@pytest.mark.parametrize(
    "host", ["198.51.100.7.nip.io:9860", "nas:9860", "198.51.100.7.example:9860"]
)
def test_names_that_look_like_addresses_still_read_nothing(machine, host):
    assert machine.host_problem("/api/search", None, host)


def test_the_page_opened_by_ip_address_may_write(machine):
    assert machine.origin_problem("POST", "http://198.51.100.7:9860", "198.51.100.7:9860") == ""
    assert machine.origin_problem("POST", "http://[2001:db8::7]:9860", "[2001:db8::7]:9860") == ""


def test_a_foreign_page_still_cannot_write_to_an_ip_address(machine):
    assert machine.origin_problem("POST", "http://evil.example", "198.51.100.7:9860")
    # same address, another port: another site's page
    assert machine.origin_problem("POST", "http://198.51.100.7:8080", "198.51.100.7:9860")
    # a rebound name stays refused even when it equals the Host
    assert machine.origin_problem("POST", "http://evil.example:9860", "evil.example:9860")


def test_a_container_reached_by_the_hosts_address(client, monkeypatch):
    from whisper_tm.core import security

    monkeypatch.setattr(security, "_own_addresses", lambda: {"127.0.0.1", "172.17.0.2"})
    c, _ = client
    assert c.get("/api/search/speakers", headers={"Host": "198.51.100.7:9860"}).status_code == 200
    r = c.get("/api/search/speakers", headers={"Host": "nas.example:9860"})
    assert r.status_code == 403 and "trusted_origins" in r.get_json()["error"]


# ── Only our extension ─────────────────────────────────────────────────────
#
# Every installed browser extension passed the check and could change the
# server's state. The manifest's "key" fixes our extension's ID; others pass
# only when listed in security.trusted_extensions (a store build has its own).


@pytest.mark.parametrize(
    "origin",
    [
        "chrome-extension://abcdefghijklmnop",
        "moz-extension://1234-5678",
        "chrome-extension://" + "mloofppjoccofakkdoahcmjpppamkgcd" + "x",
    ],
)
def test_another_extension_is_refused(origin):
    from whisper_tm.core.security import get_effect_problem

    assert "not ours" in origin_problem("POST", origin, "localhost:5077", "cross-site")
    assert get_effect_problem("GET", "/api/rag/build", "cross-site", origin)


def test_a_listed_extension_passes():
    listed = ["abcdefghijklmnop", "moz-extension://1234-5678"]
    assert (
        origin_problem(
            "POST",
            "chrome-extension://abcdefghijklmnop",
            "localhost:5077",
            "cross-site",
            extensions=listed,
        )
        == ""
    )
    assert (
        origin_problem(
            "POST", "moz-extension://1234-5678/", "localhost:5077", "cross-site", extensions=listed
        )
        == ""
    )


def test_the_manifest_key_gives_our_id():
    import base64
    import hashlib
    import json
    from pathlib import Path

    from whisper_tm.core.security import OUR_EXTENSIONS

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "browser-extension" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    der = base64.b64decode(manifest["key"])
    ext_id = "".join(chr(ord("a") + int(c, 16)) for c in hashlib.sha256(der).hexdigest()[:32])
    assert OUR_EXTENSIONS == ("chrome-extension://" + ext_id,)


def test_a_media_search_clip_get_is_guarded():
    """It cuts and encodes on the first request, and the ids are sequential."""
    from whisper_tm.core.security import get_effect_problem

    assert get_effect_problem(
        "GET", "/api/media-search/clip/17", "cross-site", "https://evil.example"
    )
    assert get_effect_problem("GET", "/api/media-search/clip/17", "same-origin", None) == ""

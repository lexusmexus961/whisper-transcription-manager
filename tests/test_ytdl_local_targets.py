"""Links to this machine or the local network are not fetched.

Links reach the downloader from the add dialog, the browser extension, the
MCP tools and Ask's actions — and through the last two from text a language
model read in a transcript or on a page. A link to http://127.0.0.1:… or a
router's address had the server fetch from inside the network. Now the
host is resolved first and every address it has must be a public one; a
playlist page's entries are checked one by one as well.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config
from whisper_tm.core import ytdl


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9860/api/settings",
        "http://localhost:8080/v1/models",
        "https://10.0.0.1/admin",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # a cloud's instance data
        "http://100.64.0.1/",  # carrier-grade NAT
        "http://[::1]:9860/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0:9860/",
    ],
)
def test_local_targets_are_refused(url):
    assert "local network" in ytdl.local_target(url)
    with pytest.raises(RuntimeError, match="local network"):
        ytdl._as_link(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://93.184.216.34/watch",
        "https://[2606:4700::1111]/x",
        "https://nothing-here.invalid/v",
    ],
)
def test_public_or_unresolvable_ones_pass(url):
    """A name that doesn't resolve is left to yt-dlp, which fails on it."""
    assert ytdl.local_target(url) == ""


def test_the_setting_lets_them_through(monkeypatch):
    monkeypatch.setitem(config._data.setdefault("youtube", {}), "allow_private_hosts", True)
    assert ytdl.local_target("http://192.168.1.10/video.mp4") == ""


def test_the_download_route_says_why(monkeypatch):
    from flask import Flask

    import whisper_tm.extensions as ext
    from whisper_tm.api import media

    monkeypatch.setattr(ext, "hw", type("Hw", (), {"has_ytdlp": True})())
    started = []
    monkeypatch.setattr(ytdl, "start_download", lambda *a, **k: started.append(a) or "j1")
    app = Flask(__name__)
    app.register_blueprint(media.media_bp)
    r = app.test_client().post(
        "/api/youtube/download",
        json={"urls": ["https://93.184.216.34/v", "http://127.0.0.1:9860/api/settings"]},
    )
    assert r.status_code == 400 and "local network" in r.get_json()["error"]
    assert not started


def test_the_setting_is_file_only():
    from whisper_tm.api.settings import _file_only

    assert _file_only("youtube.allow_private_hosts")


# ── One lookup per host ───────────────────────────────────────────────────
#
# Fifty links of one job name the same host, and so do the entries of a
# playlist. Each was looked up on its own; with a name that doesn't resolve
# and a slow resolver that was seconds per link, in one request.


def test_a_host_is_looked_up_once_for_many_links(monkeypatch):
    import socket

    from whisper_tm.core import ytdl

    asked = []

    def lookup(host, *a, **k):
        asked.append(host)
        if host == "inside.example":
            return [(socket.AF_INET, 0, 0, "", ("192.168.1.20", 0))]
        raise socket.gaierror(socket.EAI_NONAME, "unknown")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setattr(ytdl, "_resolved", {})
    assert [ytdl.local_target(f"https://nowhere.example/{i}") for i in range(50)] == [""] * 50
    assert asked == ["nowhere.example"]
    assert all("local network" in ytdl.local_target(f"http://inside.example/{i}") for i in range(5))
    assert asked == ["nowhere.example", "inside.example"]


def test_an_answer_is_kept_only_for_a_short_while(monkeypatch):
    import socket

    from whisper_tm.core import ytdl

    asked = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: asked.append(host) or [(socket.AF_INET, 0, 0, "", ("8.8.8.8", 0))],
    )
    monkeypatch.setattr(ytdl, "_resolved", {})
    clock = [1000.0]
    monkeypatch.setattr(ytdl.time, "monotonic", lambda: clock[0])
    assert ytdl.local_target("https://public.example/a") == ""
    clock[0] += ytdl._RESOLVED_FOR + 1
    assert ytdl.local_target("https://public.example/b") == ""
    assert asked == ["public.example", "public.example"]


def test_a_missing_js_runtime_is_named_first(monkeypatch):
    """Without Deno the 403 said 'outdated yt-dlp' and sent people to the
    nightly build."""
    from whisper_tm.core import ytdl

    monkeypatch.setattr(ytdl, "js_runtime", lambda: "")
    assert "Deno" in ytdl.hint_for_error("HTTP Error 403: Forbidden")
    monkeypatch.setattr(ytdl, "js_runtime", lambda: "/usr/bin/deno")
    assert "nightly" in ytdl.hint_for_error("HTTP Error 403: Forbidden")


def test_a_title_keeps_its_script():
    """secure_filename kept ASCII only: a Japanese or Cyrillic title fell
    back to the video id."""
    from whisper_tm.core.ytdl import title_stem

    assert title_stem("日本語のタイトル") == "日本語のタイトル"
    assert title_stem("Москва — столица") == "Москва_—_столица"
    assert title_stem("Ваня/Маша: 1991") == "Ваня_Маша__1991"
    # ASCII as before (secure_filename), so earlier downloads are still found
    assert title_stem("How Fast Is It Really?") == "How_Fast_Is_It_Really"
    assert title_stem("...") == "" and title_stem("") == ""
    assert len(title_stem("x" * 300)) == 80

"""
Second review of the security hardening.

1. The default work folder (~/.cache/whisper-tm/work) was refused when an
   allowed root or the media folder was the home folder (or "/") — startup
   cleaning then stood back forever.
2. check_directory resolved NAS paths while holding config._lock: a hung
   CIFS mount froze every config.get().
3. A custom work folder from before the marker, holding only the app's own
   leftovers, was never marked — and never cleaned again.
4. Behind a reverse proxy that rewrites Host, every POST was refused.
5. DNS rebinding (Origin == Host == evil.example:5077) passed the check.
6. Refreshing the trace's secret list flipped the global tracing switch.
7. _write_private failed where a rename can't replace the file, and wrote
   under config._lock.
8. Public paths without a trailing slash lost their subtree silently.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import threading
import time
from pathlib import Path

import pytest
import yaml

from whisper_tm.config import ConfigManager


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch HOME — the real ~/.cache/whisper-tm/work is never touched."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    for var in ("XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    return h


def _load(path: Path, data: dict) -> ConfigManager:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    c = ConfigManager()
    c.load(path)
    return c


# ── 1 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "roots,media",
    [
        (["~"], None),
        (["/"], None),
        ([], "~"),
    ],
)
def test_default_work_folder_passes_with_home_as_root_or_media(
    home, tmp_path, monkeypatch, roots, media
):
    roots = [str(home) if r == "~" else r for r in roots]
    data = {"security": {"allowed_roots": roots}}
    if media:
        data["directories"] = {"media": str(home)}
    c = _load(tmp_path / "config.yaml", data)
    # allowed_roots() reads the app's global config; this test's instance is
    # its own — hand it the roots as the app would compute them
    monkeypatch.setattr(
        "whisper_tm.core.security.allowed_roots",
        lambda: [Path(r).resolve() for r in roots] + [c.media_dir.resolve()],
    )
    default = c.cache_dir / "work"
    assert c.work_dir.resolve() == default.resolve()
    assert c.check_directory("work", str(default)) == ""
    (default / "job_1").mkdir(parents=True)
    result = c.clean_workdir()
    assert result["refused"] is False and result["removed"] == 1


# ── 2 ─────────────────────────────────────────────────────────────────────


def test_a_hung_nas_path_does_not_hold_the_config_lock(home, tmp_path, monkeypatch):
    import whisper_tm.config as cfgmod

    c = _load(
        tmp_path / "config.yaml", {"directories": {"logs": str(tmp_path / "hungnas" / "logs")}}
    )
    entered, release = threading.Event(), threading.Event()
    real = Path.resolve

    def slow_resolve(self, strict=False):
        if "hungnas" in str(self):
            entered.set()
            release.wait(10)
        return real(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", slow_resolve)
    monkeypatch.setattr(cfgmod, "RESOLVE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(cfgmod, "_resolver", None)
    out = {}
    t = threading.Thread(
        target=lambda: out.setdefault("why", c.check_directory("work", str(tmp_path / "w2")))
    )
    try:
        t0 = time.monotonic()
        t.start()
        assert entered.wait(5)
        got = c._lock.acquire(timeout=0.5)
        assert got, "config lock held during NAS resolve"
        c._lock.release()
        t.join(5)
        assert not t.is_alive() and time.monotonic() - t0 < 3
        assert out["why"] == ""
    finally:
        release.set()


# ── 3 ─────────────────────────────────────────────────────────────────────


def test_old_custom_work_folder_with_app_leftovers_is_marked(home, tmp_path):
    wd = tmp_path / "mywork"
    (wd / "Show_ep1_mp4_1a2b3c4d").mkdir(parents=True)
    (wd / "Show_ep1_mp4_1a2b3c4d" / "asr_block_001.wav").write_bytes(b"x")
    (wd / "_sep_api_out").mkdir()
    (wd / "chunk_0001.wav").write_bytes(b"x")
    (wd / ".server-busy.json").write_text("{}")
    c = _load(tmp_path / "config.yaml", {"directories": {"work": str(wd)}})
    assert (wd / c.WORK_MARKER).exists()
    assert c.clean_workdir()["ok"] is True


def test_foreign_files_get_a_startup_warning_and_no_marker(home, tmp_path, caplog):
    wd = tmp_path / "docs"
    wd.mkdir()
    (wd / "taxes.pdf").write_bytes(b"x")
    caplog.set_level(logging.WARNING, logger="whisper_tm.config")
    c = _load(tmp_path / "config.yaml", {"directories": {"work": str(wd)}})
    assert not (wd / c.WORK_MARKER).exists()
    assert any(c.WORK_MARKER in r.getMessage() for r in caplog.records)


# ── 4 / 5 ─────────────────────────────────────────────────────────────────


def test_trusted_origin_passes_behind_a_proxy():
    from whisper_tm.core.security import origin_problem

    assert (
        origin_problem(
            "POST", "https://wx.example.org", "127.0.0.1:5077", trusted=["https://wx.example.org"]
        )
        == ""
    )
    assert origin_problem("POST", "https://wx.example.org", "127.0.0.1:5077")


def test_dns_rebinding_is_refused():
    from whisper_tm.core.security import origin_problem

    assert origin_problem("POST", "http://evil.example:5077", "evil.example:5077")


def test_own_names_and_addresses_still_pass(monkeypatch):
    import socket

    from whisper_tm.core import security

    monkeypatch.setattr(security, "_own_addresses", lambda: {"127.0.0.1", "::1", "192.0.2.20"})
    monkeypatch.setattr(socket, "gethostname", lambda: "mediabox")
    monkeypatch.setattr(socket, "getfqdn", lambda *a: "mediabox.home.arpa")
    security._own_names_cache.clear()
    for host in (
        "localhost:5077",
        "127.0.0.1:5077",
        "[::1]:5077",
        "192.0.2.20:5077",
        "mediabox:5077",
        "MEDIABOX.home.arpa:5077",
        "mediabox.local:5077",
    ):
        origin = "http://" + host.lower()
        assert security.origin_problem("POST", origin, host) == "", host
    # The extension sends from its own origin to whatever address it is set to
    assert (
        security.origin_problem(
            "POST",
            "chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd",
            "transcriber.example:5077",
        )
        == ""
    )
    security._own_names_cache.clear()


# ── 6 ─────────────────────────────────────────────────────────────────────


def test_refreshing_secrets_never_switches_tracing(monkeypatch):
    from whisper_tm.core import debug

    debug.start_call_tracing(with_arguments=False)
    try:

        def collect_and_stop():
            debug.stop_call_tracing(reason="stopped meanwhile")
            return frozenset()

        monkeypatch.setattr(debug, "_collect_secrets", collect_and_stop)
        debug._refresh_trace_secrets()
        assert debug.call_tracing_active() is False
    finally:
        debug.stop_call_tracing(reason="test")


def test_the_hook_never_collects(monkeypatch):
    from whisper_tm.config import coerce_value
    from whisper_tm.core import debug

    calls = []
    monkeypatch.setattr(debug, "_collect_secrets", lambda: calls.append(1) or frozenset())
    debug.start_call_tracing(with_arguments=False)
    try:
        n = len(calls)
        for _ in range(30000):
            coerce_value("1", 1)
        assert len(calls) == n
    finally:
        debug.stop_call_tracing(reason="test")


# ── 7 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("code", [errno.EBUSY, errno.EXDEV, errno.EACCES])
def test_write_private_falls_back_to_writing_in_place(tmp_path, monkeypatch, code):
    from whisper_tm import config as cfgmod

    target = tmp_path / "config.yaml"
    target.write_text("old\n", encoding="utf-8")
    os.chmod(target, 0o644)

    def refuse(*a, **k):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(cfgmod.os, "replace", refuse)
    cfgmod._write_private(target, "new: 1\n")
    assert target.read_text() == "new: 1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]


def test_save_writes_outside_the_config_lock(home, tmp_path, monkeypatch):
    from whisper_tm import config as cfgmod

    c = _load(tmp_path / "config.yaml", {})
    held = []
    real = cfgmod._write_private

    def spy(target, text):
        free = c._lock.acquire(blocking=False)
        if free:
            c._lock.release()
        # An RLock acquires for its owner: ask from another thread
        box = {}
        t = threading.Thread(
            target=lambda: box.setdefault("got", c._lock.acquire(timeout=0.5)) and c._lock.release()
        )
        t.start()
        t.join(2)
        held.append(box.get("got"))
        return real(target, text)

    monkeypatch.setattr(cfgmod, "_write_private", spy)
    c.save()
    assert held == [True]


# ── 8 ─────────────────────────────────────────────────────────────────────


def test_custom_public_path_without_slash_is_warned_about(monkeypatch, caplog):
    from whisper_tm.config import config
    from whisper_tm.core import auth

    monkeypatch.setitem(
        config._data, "auth", {"public_paths": ["/api/health", "/static/", "/api/public"]}
    )
    warnings = auth.public_path_warnings()
    assert len(warnings) == 1 and "/api/public" in warnings[0]

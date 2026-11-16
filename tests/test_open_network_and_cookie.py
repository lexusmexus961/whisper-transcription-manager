"""Two things a LAN install needs: a warning when anyone on the network can
reach it without a login, and a login cookie that another install on the
same host name doesn't overwrite."""

import pytest


@pytest.mark.parametrize(
    "host,auth,docker,warned",
    [
        ("127.0.0.1", False, False, False),
        ("localhost", False, False, False),
        ("0.0.0.0", True, False, False),
        ("0.0.0.0", False, True, False),  # the container always binds 0.0.0.0
        ("0.0.0.0", False, False, True),
        ("192.0.2.7", False, False, True),
    ],
)
def test_the_open_network_warning(monkeypatch, host, auth, docker, warned):
    from whisper_tm import __main__ as main
    from whisper_tm.core import auth as auth_mod
    from whisper_tm.core import commands

    monkeypatch.setattr(auth_mod, "is_auth_enabled", lambda: auth)
    monkeypatch.setattr(commands, "in_docker", lambda: docker)
    text = main.network_without_login(host)
    assert bool(text) is warned
    if warned:
        assert host in text and "auth.enabled" in text


def test_the_banner_lines_have_one_width():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "whisper_tm/__main__.py").read_text(
        encoding="utf-8"
    )
    assert "Whisper Transcription Manager v{__version__:<17}║" in src
    assert len("║  Whisper Transcription Manager v" + "0.9.0".ljust(17) + "║") == len(
        "╔" + "═" * 50 + "╗"
    )


def test_the_cookie_is_named_after_the_install(tmp_path, monkeypatch):
    from whisper_tm.app import create_app
    from whisper_tm.config import config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"directories:\n  media: {tmp_path}\nserver:\n  secret_key: one\n", encoding="utf-8"
    )
    config.load(str(cfg))
    a = create_app(str(cfg)).config["SESSION_COOKIE_NAME"]
    config._data["server"]["secret_key"] = "two"
    b = create_app(str(cfg)).config["SESSION_COOKIE_NAME"]
    assert a.startswith("wtm_session_") and len(a) == len("wtm_session_") + 8 and a != b

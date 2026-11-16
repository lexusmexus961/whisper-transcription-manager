"""Small security fixes: a password with an umlaut, the MCP token's temp
file, the server's paths on the access page (tests/test_settings_hardening.py)."""

from __future__ import annotations

import os
import stat

import pytest

from whisper_tm.core import auth


@pytest.mark.parametrize("password", ["Grüße-2026", "пароль", "パスワード", "plain"])
def test_any_password_can_log_in(monkeypatch, password):
    """compare_digest refuses str with anything but ASCII — the login with
    such a password raised a TypeError (an error 500) instead of checking."""
    monkeypatch.setattr(auth, "_get_password", lambda: password)
    assert auth._check_password(password) is True
    assert auth._check_password(password + "x") is False


def test_a_token_with_an_umlaut_is_checked_too(monkeypatch):
    monkeypatch.setattr(auth, "_get_api_token", lambda: "tökén")
    assert auth._check_token("tökén") is True and auth._check_token("token") is False


def test_the_token_file_is_private_even_beside_an_old_temp_file(tmp_path):
    """A world-readable mcp_token.tmp left over was opened again with its
    old mode, the token written into it, and only then moved in place."""
    from whisper_tm.mcp_server import _write_private

    stale = tmp_path / "mcp_token.tmp"
    stale.write_text("old")
    stale.chmod(0o644)
    target = tmp_path / "mcp_token"
    _write_private(target, "secret-token")
    assert target.read_text() == "secret-token\n"
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert stale.read_text() == "old"  # not used
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("mcp_token.") and p != stale]

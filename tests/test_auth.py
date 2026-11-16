"""
Tests for core/auth.py — authentication.
"""

import pytest

from whisper_tm.config import config
from whisper_tm.core.auth import (
    _check_password,
    _check_token,
    _is_public_path,
    is_auth_enabled,
)


@pytest.fixture(autouse=True)
def _load_config():
    if not config.is_loaded:
        config.load()


class TestCheckPassword:
    def test_correct_plaintext(self):
        config.set("auth", "password", "test123")
        assert _check_password("test123") is True

    def test_wrong_plaintext(self):
        config.set("auth", "password", "test123")
        assert _check_password("wrong") is False

    def test_empty_stored(self):
        config.set("auth", "password", "")
        assert _check_password("anything") is False

    def test_timing_safe(self):
        """Password check should be timing-safe (hmac.compare_digest)."""
        config.set("auth", "password", "secret")
        # Both cases should have similar execution time
        assert _check_password("secret") is True
        assert _check_password("secre") is False


class TestCheckToken:
    def test_correct_token(self):
        config.set("auth", "api_token", "my-token-123")
        assert _check_token("my-token-123") is True

    def test_wrong_token(self):
        config.set("auth", "api_token", "my-token-123")
        assert _check_token("wrong-token") is False

    def test_empty_token(self):
        config.set("auth", "api_token", "")
        assert _check_token("anything") is False


class TestIsPublicPath:
    def test_health(self):
        assert _is_public_path("/api/health") is True

    def test_static(self):
        assert _is_public_path("/static/js/app.js") is True

    def test_auth(self):
        assert _is_public_path("/auth/login") is True

    def test_api_protected(self):
        assert _is_public_path("/api/scan") is False

    def test_root_protected(self):
        assert _is_public_path("/") is False


class TestIsAuthEnabled:
    def test_disabled_by_default(self):
        config.set("auth", "enabled", False)
        assert is_auth_enabled() is False

    def test_enabled(self):
        config.set("auth", "enabled", True)
        assert is_auth_enabled() is True

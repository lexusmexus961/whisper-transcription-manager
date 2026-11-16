"""
Public paths match exactly.

The default public path "/api/health" used to match everything below it, so
with the login on, /api/health/detail stayed open: the name of the file
being transcribed, the media path, storage error texts and system warnings.
"""

from whisper_tm.core.auth import _is_public_path


def test_the_liveness_ping_is_public():
    assert _is_public_path("/api/health") is True


def test_the_detail_view_is_not():
    assert _is_public_path("/api/health/detail") is False
    assert _is_public_path("/api/healthz") is False


def test_directory_entries_still_match_by_prefix():
    assert _is_public_path("/static/js/app.js") is True
    assert _is_public_path("/auth/login") is True

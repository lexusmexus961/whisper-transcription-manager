"""Jumping in the transcript must not query the network share every time.

Symptom: clicking a point in the transcript sometimes took very long to
start playback.

``/api/debug/timings`` shows where the time goes while a batch is reading
from the same share: most jumps are fast, single ones stall for many
seconds — and nearly all of that is the file stat on the network share,
not ``send_file``.

Each click is a separate range request, and each triggered a new path resolution:
``safe_path`` calls ``.resolve()`` **twice** (over CIFS directory by
directory), plus an ``exists()``. On the tenth jump to the same file,
that's ten times the same answer.

So we test exactly that: the expensive resolution is skipped on the second
call — and the check is not lost. A cache that lets an escape attempt through
would be worse than the problem.
"""

from __future__ import annotations

import time

import pytest
from flask import Flask

from whisper_tm.api import media


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    root = tmp_path / "media"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "film.mp4").write_bytes(b"not really a film")
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: root))
    media.clear_path_cache()

    a = Flask(__name__)
    a.register_blueprint(media.media_bp)
    a.root = root
    yield a
    media.clear_path_cache()


def test_second_jump_does_not_query_nas_again(app, monkeypatch):
    """The core. Count REAL resolutions, not requests."""
    calls = []
    real = media.safe_path
    monkeypatch.setattr(media, "safe_path", lambda b, p: (calls.append(p), real(b, p))[1])

    c = app.test_client()
    for _ in range(5):
        assert c.get("/api/video/folder/film.mp4").status_code in (200, 206)
    assert len(calls) == 1, f"the share was queried {len(calls)}x instead of once"


def test_after_ttl_queried_again(app, monkeypatch):
    """30 seconds, matching a CIFS mount's `actimeo=30`. Without expiry,
    a deleted or replaced file would be answered wrong indefinitely."""
    calls = []
    real = media.safe_path
    monkeypatch.setattr(media, "safe_path", lambda b, p: (calls.append(p), real(b, p))[1])
    monkeypatch.setattr(media, "PATH_CACHE_TTL", 0.05)

    c = app.test_client()
    c.get("/api/video/folder/film.mp4")
    time.sleep(0.08)
    c.get("/api/video/folder/film.mp4")
    assert len(calls) == 2


def test_new_file_appears_after_ttl(app):
    """The NEGATIVE result is also cached — it took the same seconds.
    But it must not stick forever."""
    c = app.test_client()
    assert c.get("/api/video/folder/later.mp4").status_code == 404
    (app.root / "folder" / "later.mp4").write_bytes(b"there")
    # Within the TTL it stays as the old answer …
    assert c.get("/api/video/folder/later.mp4").status_code == 404
    media.clear_path_cache()  # … then not anymore
    assert c.get("/api/video/folder/later.mp4").status_code in (200, 206)


def test_escape_attempt_still_rejected(app):
    """The check is not skipped, only not repeated.

    Ask twice so a hit from cache is also checked —
    a cache that lets the second one through would be the worse kind of
    error: it never shows up in real use.
    """
    c = app.test_client()
    for _ in range(2):
        r = c.get("/api/video/../../etc/passwd")
        assert r.status_code in (400, 403, 404), r.status_code


def test_different_files_dont_share_entry(app, monkeypatch):
    """The key must be the whole input. If not, the second file
    gets the path of the first — and the user sees film A when clicking on
    recording B."""
    (app.root / "folder" / "second.mp4").write_bytes(b"different film")
    p1, da1 = media._resolved("folder/film.mp4")
    p2, da2 = media._resolved("folder/second.mp4")
    assert da1 and da2
    assert p1 != p2
    assert p1.name == "film.mp4" and p2.name == "second.mp4"


def test_memory_doesnt_grow_unbounded(app, monkeypatch):
    """A library with hundreds of thousands of files must not land in a dict."""
    monkeypatch.setattr(media, "PATH_CACHE_MAX", 8)
    for i in range(40):
        media._resolved(f"folder/nothere{i}.mp4")
    assert len(media._path_cache) <= 8

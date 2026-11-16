"""
Allowed roots: the application may only work within configured trees.

Why this was needed: ``safe_path()`` checks a path against a **base** —
and is therefore only as good as the base itself. In this application
the base can be switched at runtime (``config.media_dir``) and was, at
several endpoints, additionally taken straight from the request (``base_dir``).
Reproduced and confirmed live:

    POST /api/set_directory {"path": "/"}      -> 200
    GET  /api/download/etc/hostname            -> 200, real file content

Every correct ``safe_path`` check in the app was therefore a check against
a directory that the caller had chosen themselves.

The check sits at the HTTP boundary, not in the ``media_dir`` setter: the
setter is the programmatic path (startup, tools, tests), and it also writes
``_data["directories"]["media"]`` — an allowlist derived from that would
trail behind the media directory and legitimize itself.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config
from whisper_tm.core.security import allowed_roots, root_allowed


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A media directory that counts as the only allowed root."""
    media = tmp_path / "media"
    (media / "series").mkdir(parents=True)
    (media / "series" / "episode1.mp4").write_bytes(b"x")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "secret.txt").write_text("secret")

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "media_on_start", property(lambda self: media))
    monkeypatch.setattr(
        config,
        "get",
        lambda *a, **k: (
            [] if a[:2] == ("security", "allowed_roots") else config.__class__.get(config, *a, **k)
        ),
    )
    return {"media": media, "foreign": foreign}


class TestRootAllowed:
    def test_the_root_itself(self, library):
        assert root_allowed(library["media"])

    def test_subdirectory(self, library):
        assert root_allowed(library["media"] / "series")

    def test_file_below_it(self, library):
        assert root_allowed(library["media"] / "series" / "episode1.mp4")

    def test_sibling_folder_is_not(self, library):
        assert not root_allowed(library["foreign"])

    def test_filesystem_root_is_not(self, library):
        assert not root_allowed("/")

    def test_system_directory_is_not(self, library):
        assert not root_allowed("/etc")
        assert not root_allowed("/etc/shadow")

    def test_traversal_is_resolved(self, library):
        """``..`` must not lead outside the root."""
        assert not root_allowed(str(library["media"] / ".." / "foreign" / "secret.txt"))

    def test_name_prefix_is_not_a_match(self, tmp_path, monkeypatch):
        """``/media-evil`` is NOT under ``/media``.

        A plain ``startswith`` would have let it through — the same mistake
        ``safe_path`` already avoids with ``relative_to``.
        """
        media = tmp_path / "media"
        media.mkdir()
        evil = tmp_path / "media-evil"
        evil.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        monkeypatch.setattr(type(config), "media_on_start", property(lambda self: media))
        assert not root_allowed(evil)

    def test_empty_value_is_not(self, library):
        assert not root_allowed("")
        assert not root_allowed(None)


class TestAllowedRoots:
    def test_current_media_directory_counts_too(self, library):
        """The current value belongs to it — otherwise the programmatic
        path (startup, tools, tests) would lock itself out.

        This is not an extension: it can only be switched via the HTTP
        boundary, and that checks beforehand against exactly this list.
        """
        assert library["media"].resolve() in allowed_roots()

    def test_explicit_list_takes_precedence(self, tmp_path, monkeypatch):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: a))
        real = config.get

        def faked(*args, **kw):
            if args[:2] == ("security", "allowed_roots"):
                return [str(b)]
            return real(*args, **kw)

        monkeypatch.setattr(config, "get", faked)
        roots = allowed_roots()
        assert b.resolve() in roots
        assert root_allowed(b)

"""The cache buster must change when a file changes.

**Why this deserves its own test.** Browsers can continue with old JS from
cache after an update — "the button does nothing", while the code has long
since been fixed. The first remedy: attach version and Git commit to every
static inclusion.

That isn't enough. `version_info()` is cached for process lifetime; the
buster thus stays fixed for the entire SERVER RUN. Whoever changes a JS file
without restarting changes the content under an unchanged URL.

The failure mode: a fix to the voice map is built, tested and committed —
but not visible in an already open browser. It gets
`voicemap_graph.js?v=2.1.0-c7fcfb2` from its cache, while the server delivers
new content under exactly that URL. Checks with a freshly started browser
always see the new version, so the reported error looks like something else
when the real issue is the cache.

That's why the latest modification time under ``static/`` is now also
included in the buster.
"""

from __future__ import annotations

import re

import pytest

from whisper_tm.api import main as api_main


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Just the home page — no database, worker, or models.

    `create_app()` touches the REAL database; here only
    the blueprint is registered with the same folders that `app.py` sets.
    """
    from pathlib import Path

    from flask import Flask

    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))

    static_dir = Path(api_main.__file__).resolve().parents[1] / "static"
    a = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static",
        template_folder=str(static_dir),
    )
    a.register_blueprint(api_main.main_bp)
    return a.test_client()


def _buster(html: str) -> str:
    """Extract the value of ``?v=…`` from the first inclusion."""
    m = re.search(r'\.(?:js|css)\?v=([^"\']+)', html)
    assert m, "no static inclusion with ?v= found"
    return m.group(1)


class TestItChangesWithFiles:
    def test_changed_js_file_changes_buster(self, client, tmp_path, monkeypatch):
        """The core: no restart, no commit — just touch the file.

        This is the case described above. Everything else (version, commit)
        stays the same; if the buster didn't change, a browser with cache
        would get the old file.
        """
        before = _buster(client.get("/").get_data(as_text=True))

        # Make a real static file "later" — and later than ALL others.
        #
        # The buster is the maximum across all files. `old_value + 60` would
        # do nothing when another file (just edited) is even newer, and the
        # test would report an error that doesn't exist.
        import os
        from pathlib import Path

        js = Path(api_main.__file__).resolve().parents[1] / "static" / "js" / "voicemap_graph.js"
        old_value = js.stat().st_mtime
        new_value = api_main._static_marker() + 1000
        try:
            os.utime(js, (new_value, new_value))
            after = _buster(client.get("/").get_data(as_text=True))
        finally:
            os.utime(js, (old_value, old_value))

        assert after != before, (
            "buster didn't change — a browser with cache would still get the old file"
        )

    def test_without_change_stays_same(self, client):
        """Otherwise every delivery would be a cache miss, and the cache would be
        useless instead of wrong — no solution either."""
        a = _buster(client.get("/").get_data(as_text=True))
        b = _buster(client.get("/").get_data(as_text=True))
        assert a == b

    def test_version_and_commit_still_included(self, client):
        """The buster should stay readable: it must show which version
        is being delivered — otherwise it doesn't help debugging."""
        from whisper_tm import __version__

        b = _buster(client.get("/").get_data(as_text=True))
        assert b.startswith(__version__), b

    def test_all_inclusions_carry_same_version(self, client):
        """A partially updated set of JS and CSS is worse than a completely old one
        — then appearance and behavior don't match."""
        html = client.get("/").get_data(as_text=True)
        value_list = set(re.findall(r'\.(?:js|css)\?v=([^"\']+)', html))
        assert len(value_list) == 1, value_list


class TestTheMarkerItself:
    def test_it_finds_the_newest_file(self):
        from pathlib import Path

        root = Path(api_main.__file__).resolve().parents[1] / "static"
        real = max(
            p.stat().st_mtime for p in root.rglob("*") if p.suffix in (".js", ".css", ".html")
        )
        assert api_main._static_marker() == int(real)

    def test_read_error_does_not_throw(self, monkeypatch):
        """The home page must never fail at this point.

        An unusable buster is a cosmetic flaw; an exception would be
        total UI failure — because of a timestamp.
        """
        import pathlib

        def _broken(self, *a, **kw):
            raise OSError("Directory not readable")

        monkeypatch.setattr(pathlib.Path, "rglob", _broken)
        assert api_main._static_marker() == 0

    def test_page_comes_through_anyway(self, client, monkeypatch):
        """The proof is in the pudding: full response, not just the helper function."""
        import pathlib

        def _broken(self, *a, **kw):
            raise OSError("Directory not readable")

        monkeypatch.setattr(pathlib.Path, "rglob", _broken)
        r = client.get("/")
        assert r.status_code == 200
        assert "?v=" in r.get_data(as_text=True)

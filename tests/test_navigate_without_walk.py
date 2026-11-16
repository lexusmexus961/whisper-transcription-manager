"""
Clicking a search result must NOT trigger a filesystem walk.

Incident: /api/navigate_to_file found the file immediately
via hint_dir (step 1a), but then still ran scan_media_directory
(recursive) anyway — a full NAS walk over the whole library, just to build
the list entry of the ONE file. Three quick clicks = three parallel
full walks; the file wouldn't open for minutes. Since the scan limits
were removed, this would have become even more expensive.

Now a DB point query runs there instead (get_media_file_at → db_row_to_entry),
with register_single_file as a fallback for freshly stored files.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    media = tmp_path / "media"
    (media / "below").mkdir(parents=True)
    (media / "below" / "clip.mp4").write_bytes(b"x" * 2048)
    config._data.setdefault("directories", {})["media"] = str(media)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    app = create_app()
    # There used to be a patch on `api.main.db` here, with the note:
    # "api.main binds ext.db at import time — if another app test ran
    # before it in the same process, the module-level db still points at
    # THAT test's database."
    #
    # The note described a real bug in the application, not a quirk of the
    # tests: `api.main` used to read the registry only once at import time.
    # The module now reads it on every call (`ext.db`), so the crutch is
    # not just unnecessary but harmful — it would hide a regression.
    # See `tests/test_extensions_registry.py`.
    return app.test_client(), media


def test_navigate_does_no_walk_and_delivers_the_entry(client, monkeypatch):
    c, media = client

    # Get the file into the DB (as upload/download/indexer would)
    from whisper_tm.scanner import register_single_file

    register_single_file(media / "below" / "clip.mp4")

    # The actual safeguard: ANY walk in the navigation request is a bug.
    import whisper_tm.scanner as scanner_mod

    def no_walk(*a, **k):
        raise AssertionError(
            "navigate_to_file triggered a directory walk "
            "— exactly what blocked opening from search "
            "for minutes"
        )

    monkeypatch.setattr(scanner_mod, "scan_media_directory", no_walk)
    monkeypatch.setattr(scanner_mod.filesystem, "collect_media_files", no_walk)

    r = c.post(
        "/api/navigate_to_file", json={"filename": "below/clip.mp4", "media_dir": str(media)}
    )
    d = r.get_json()
    assert d["found"] is True, d
    # Folder jump: the root is the file's PARENT FOLDER —
    # the list shows the file along with its neighbors, not the whole search root.
    assert d["directory"] == str(media / "below")
    assert d["file"]["name"] == "clip.mp4"
    # The former full list is deliberately no longer part of the response
    assert "files" not in d


def test_a_full_path_outside_the_hint_folder_is_found(client):
    """The voice map sends the full path of a recording outside the folder it
    was loaded for. Step 1a found it; step 3 joined it to the hint folder
    ("<folder>/mnt/…"), switched the list there and said "File not found"
    for a file that was there (a name with "!!", parentheses and a stray
    " )" before the extension)."""
    c, media = client
    folder = media / "videos" / "some channel"
    folder.mkdir(parents=True)
    clip = folder / "Clip 12_ final cut!! (draft _) ).mp4"
    clip.write_bytes(b"z" * 2048)
    (media / "other").mkdir()

    r = c.post(
        "/api/navigate_to_file", json={"filename": str(clip), "media_dir": str(media / "other")}
    )
    d = r.get_json()
    assert d["found"] is True, d
    assert d["directory"] == str(folder)
    assert d["file"]["name"] == clip.name


def test_a_full_path_outside_the_allowed_roots_is_refused(client, tmp_path):
    c, media = client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.mp4").write_bytes(b"x" * 2048)
    from whisper_tm.config import config

    before = str(config.media_dir)

    r = c.post(
        "/api/navigate_to_file", json={"filename": str(outside / "x.mp4"), "media_dir": str(media)}
    )
    assert r.status_code == 403
    assert r.get_json()["found"] is False
    assert str(config.media_dir) == before


def test_navigate_registers_an_unknown_file_without_a_walk(client, monkeypatch):
    """Freshly stored file (not yet in the DB): point registration."""
    c, media = client
    (media / "new_value.mp4").write_bytes(b"y" * 2048)

    import whisper_tm.scanner as scanner_mod

    def no_walk(*a, **k):
        raise AssertionError("walk in the navigation request")

    monkeypatch.setattr(scanner_mod, "scan_media_directory", no_walk)

    r = c.post("/api/navigate_to_file", json={"filename": "new_value.mp4", "media_dir": str(media)})
    d = r.get_json()
    assert d["found"] is True, d
    assert d["file"]["name"] == "new_value.mp4"

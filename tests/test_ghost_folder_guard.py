"""
Ghost directories: write paths must not create trees in guessed locations.

Failure mode: empty directories after a root migration, or a stray
"<some folder>/.transcripts". Cause: `filename` from the request
is sometimes the filename from INSIDE THE TRANSCRIPT — relative to the root
that applied AT TRANSCRIPTION TIME. `transcript_path_for` without abs_path
then guessed /current_root/<old_relpath>/ and mkdir created the ghost tree.
Fix: resolve identity first via DB; without DB identity, only write if the
media file actually exists at the derived location.
"""

import pytest

from whisper_tm.api import transcription as api_tx
from whisper_tm.config import config


@pytest.fixture
def root(tmp_path, monkeypatch):
    if not config.is_loaded:
        config.load()
    w = tmp_path / "root"
    w.mkdir()
    old_value = config.media_dir
    config.media_dir = w
    # DB gone: the test checks exactly the case "DB knows no identity"
    monkeypatch.setattr(api_tx, "db", None)
    yield w
    config.media_dir = old_value


def _tdata():
    return {"filename": "x", "complete": True, "segments": []}


def test_a_guessed_location_is_not_created(root):
    """Old root-relative name, file not there → NO mkdir."""
    api_tx._save_transcript_data("Ghost/old-root-file.mp4", _tdata())
    assert not (root / "Ghost").exists(), "ghost tree was created at guessed location"


def test_a_real_file_gets_its_json(root):
    """If media file exists at the derived location, store normally."""
    (root / "clip.mp4").write_bytes(b"x")
    api_tx._save_transcript_data("clip.mp4", _tdata())
    assert (root / ".transcripts" / "clip.mp4.json").exists()


def test_subfolder_with_a_real_file(root):
    """Also nested: mkdir only where the file actually lies."""
    (root / "Games").mkdir()
    (root / "Games" / "clip.mp4").write_bytes(b"x")
    api_tx._save_transcript_data("Games/clip.mp4", _tdata())
    assert (root / "Games" / ".transcripts" / "clip.mp4.json").exists()

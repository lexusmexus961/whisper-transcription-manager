"""
Tests: hidden directories (.zfs snapshots, etc.) are skipped in ALL recursive
walks — not just in the media scanner, but also in file search (iter_files_named
replaces rglob), JSON import (sync_from_json_dir) and mtime skip
(_newest_json_mtime).

Background: rglob descends into hidden directories. On ZFS shares, the entire
tree under .zfs/snapshot/ exists again per snapshot — the search multiplies
and runs into SMB-fatal special files (FIFO recovery artifacts inside a
snapshot can make the mount die reproducibly).
"""

import pytest

from whisper_tm.database import Database
from whisper_tm.scanner import _newest_json_mtime
from whisper_tm.scanner.filesystem import iter_files_named

MEDIA_EXT = {".mp4", ".mp3"}


@pytest.fixture
def tree(tmp_path):
    """Tree with visible files and a .zfs snapshot duplicate."""
    (tmp_path / "movies").mkdir()
    (tmp_path / "movies" / "film.mp4").write_text("real")
    # ZFS snapshot directory: same content again (multiple)
    for snap in ("snap1", "snap2"):
        d = tmp_path / ".zfs" / "snapshot" / snap / "movies"
        d.mkdir(parents=True)
        (d / "film.mp4").write_text("snapshot-copy")
    # another hidden directory
    (tmp_path / ".Trash-1000").mkdir()
    (tmp_path / ".Trash-1000" / "film.mp4").write_text("trash")
    return tmp_path


def test_iter_files_named_finds_a_real_file(tree):
    hits = list(iter_files_named(tree, "film.mp4", ignore_patterns=[".*"]))
    assert hits == [tree / "movies" / "film.mp4"]


def test_iter_files_named_prunes_hidden_folders(tree):
    # Not a single hit may come from .zfs or .Trash-1000
    hits = list(iter_files_named(tree, "film.mp4", ignore_patterns=[".*"]))
    assert all(".zfs" not in hit.parts and ".Trash-1000" not in hit.parts for hit in hits)


def test_iter_files_named_max_dirs(tmp_path):
    # Hard limit: with max_dirs=1 only the root directory is read
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.mp4").write_text("x")
    (tmp_path / "x.mp4").write_text("x")
    hits = list(iter_files_named(tmp_path, "x.mp4", ignore_patterns=[".*"], max_dirs=1))
    assert hits == [tmp_path / "x.mp4"]


def test_iter_files_named_exact_name_with_metacharacters(tmp_path):
    # No glob matching (#119): square brackets in names are literal
    (tmp_path / "clip [1080p].mp4").write_text("x")
    hits = list(iter_files_named(tmp_path, "clip [1080p].mp4", ignore_patterns=[".*"]))
    assert hits == [tmp_path / "clip [1080p].mp4"]


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close_all()


def _fake_transcript(path, name):
    path.write_text(
        '{"complete": true, '
        '"segments": [{"start": 0, "end": 1, "text": "hello"}], '
        f'"language": "de", "filename": "{name}"}}'
    )


def test_sync_from_json_dir_ignores_hidden_folders(db, tmp_path):
    # Visible transcript + duplicate in .zfs snapshot
    _fake_transcript(tmp_path / "a.json", "a.mp4")
    snap = tmp_path / ".zfs" / "snapshot" / "s1"
    snap.mkdir(parents=True)
    _fake_transcript(snap / "b.json", "b.mp4")
    r = db.sync_from_json_dir(tmp_path)
    assert r["imported"] == 1


def test_sync_from_json_dir_enters_transcripts(db, tmp_path):
    # .transcripts remains passable (only hidden directory with useful data)
    td = tmp_path / "sub" / ".transcripts"
    td.mkdir(parents=True)
    _fake_transcript(td / "c.json", "c.mp4")
    r = db.sync_from_json_dir(tmp_path)
    assert r["imported"] == 1


def test_newest_json_mtime_ignores_hidden_folders(tmp_path):
    import os

    snap = tmp_path / ".zfs" / "snapshot" / "s1"
    snap.mkdir(parents=True)
    (snap / "old_value.json").write_text("{}")
    # Date snapshot JSON artificially into the future
    fut = 4102444800  # 01.01.2100
    os.utime(snap / "old_value.json", (fut, fut))
    (tmp_path / "real.json").write_text("{}")
    real_mtime = (tmp_path / "real.json").stat().st_mtime
    assert _newest_json_mtime(tmp_path) == pytest.approx(real_mtime)

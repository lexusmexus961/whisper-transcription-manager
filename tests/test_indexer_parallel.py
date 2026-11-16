"""
The index run's per-file filesystem checks must not run one after another:
on CIFS each is a network round trip, and an index run with nothing to do
still checked every file serially (stat for the resume check, an
exists() per file in the transcript discovery).
"""

import threading
import time

import pytest

from whisper_tm.database import Database
from whisper_tm.scanner import indexer


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


@pytest.fixture
def library(tmp_path, monkeypatch, db):
    files = []
    for i in range(24):
        f = tmp_path / f"f{i:02d}.mp4"
        f.write_bytes(b"x")
        files.append(f)
    monkeypatch.setattr(indexer, "collect_media_files", lambda root, **kw: list(files))
    monkeypatch.setattr(
        indexer, "get_media_info", lambda fp: {"ok": True, "duration": 10.0, "has_audio": True}
    )
    list(indexer.index_directory(tmp_path, db=db, discover_transcripts=False))
    return files


def _slow_and_recorded(real, threads):
    def wrapper(*a, **kw):
        threads.add(threading.get_ident())
        time.sleep(0.02)
        return real(*a, **kw)

    return wrapper


def test_resume_check_stats_in_parallel(library, monkeypatch, db, tmp_path):
    threads: set[int] = set()
    monkeypatch.setattr(indexer, "_mtime", _slow_and_recorded(indexer._mtime, threads))
    final = list(indexer.index_directory(tmp_path, db=db, workers=8, discover_transcripts=False))[
        -1
    ]
    assert final["skipped"] == 24
    assert len(threads) > 1


def test_transcript_discovery_checks_in_parallel(library, monkeypatch, db, tmp_path):
    threads: set[int] = set()
    monkeypatch.setattr(
        indexer, "_transcript_beside", _slow_and_recorded(indexer._transcript_beside, threads)
    )
    final = list(indexer.index_directory(tmp_path, db=db, workers=8))[-1]
    assert final["complete"] and final["healed_transcripts"] == 0
    assert len(threads) > 1

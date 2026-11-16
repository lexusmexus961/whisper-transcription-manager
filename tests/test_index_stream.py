"""
The index stream must not fall silent.

Background: ``index_directory`` delivered its first progress event only after
the complete walk AND ``get_index_state_batch()`` were done. On a library with
a few hundred thousand files, that is minutes of silence between the
/api/index request and the first ffprobe. The watchdog in the
frontend thought the stream was dead, closed it and reported "timeout" — the
server indexed invisibly on.

From the user's perspective, indexing was simply broken.
"""

from __future__ import annotations

import time

import pytest

from whisper_tm.database import Database
from whisper_tm.scanner import indexer as ix


@pytest.fixture(autouse=True)
def _config_loaded(tmp_path):
    """The indexer discovers transcripts on the side and needs
    ``config.transcript_dir`` for that — so a loaded configuration."""
    from whisper_tm.config import config

    if not config.is_loaded:
        config.load()
    return config


@pytest.fixture
def media_items(tmp_path):
    d = tmp_path / "media"
    d.mkdir()
    for i in range(6):
        (d / f"file{i}.mp3").write_bytes(b"\x00" * 64)
    return d


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "index.db")
    d.initialize()
    return d


def _run(vdir, db, **kw):
    """Run index, collect (timestamp, event) for each message."""
    t0 = time.monotonic()
    out = []
    for prog in ix.index_directory(vdir, db=db, workers=2, **kw):
        out.append((time.monotonic() - t0, prog))
    return out


def test_the_first_event_comes_before_the_walk_ends(media_items, db, monkeypatch):
    """Heartbeats must come while collecting.

    The walk is artificially slowed down. If the first event came only after,
    the old state would be restored.
    """
    real = ix.collect_media_files

    def slow(*a, **k):
        time.sleep(2.5)
        return real(*a, **k)

    monkeypatch.setattr(ix, "collect_media_files", slow)

    events = _run(media_items, db)
    assert events, "generator delivered nothing"
    assert events[0][0] < 2.0, (
        f"first event only after {events[0][0]:.2f}s — stream falls silent "
        f"during collection (walk takes 2.5s here)"
    )
    assert events[0][1].get("phase") == "collect"


def test_no_gap_exceeds_the_watchdog(media_items, db, monkeypatch):
    """Two events must never be further apart than the watchdog
    the frontend waits (static/js/index.js: _INDEX_WATCHDOG_MS)."""
    real = ix.collect_media_files

    def slow(*a, **k):
        time.sleep(2.5)
        return real(*a, **k)

    monkeypatch.setattr(ix, "collect_media_files", slow)

    events = _run(media_items, db)
    gaps = [events[i][0] - events[i - 1][0] for i in range(1, len(events))]
    assert max(gaps, default=0) < 3.0, f"largest gap: {max(gaps):.2f}s"


def test_the_run_ends_with_a_completion_event(media_items, db):
    events = _run(media_items, db)
    assert events[-1][1].get("complete") is True


def test_an_error_in_the_collector_thread_is_not_lost(media_items, db, monkeypatch):
    """The walk runs in a thread — an exception there must reach the caller
    and not pass through as an empty result."""

    def broken(*a, **k):
        raise OSError("mount gone")

    monkeypatch.setattr(ix, "collect_media_files", broken)

    with pytest.raises(OSError, match="mount gone"):
        _run(media_items, db)


def test_metadata_lands_in_the_db_with_its_identity(tmp_path, db):
    """Counter-check to progress: the index actually persists too.

    Without real media files FFprobe fails; what's checked is that
    such files count as 'failed' and are NOT considered indexed — otherwise
    their duration would stay 0 permanently, since they'd never be probed again.
    """
    d = tmp_path / "m"
    d.mkdir()
    (d / "broken.mp3").write_bytes(b"\x00" * 64)

    last_item = _run(d, db)[-1][1]
    assert last_item["failed"] == 1
    assert last_item["indexed"] == 0
    with db.connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM media_files WHERE indexed_at > 0").fetchone()[0]
    assert n == 0

"""
The statistics cache stays small.

Every distinct ``?path=`` became a cache entry that was never removed —
only its age was checked on read. A client walking many folders grew the
server's memory without end.
"""

from whisper_tm.api import stats


def test_old_entries_go_and_the_cache_is_capped(monkeypatch):
    stats.clear_cache()
    now = [1000.0]
    monkeypatch.setattr(stats.time, "time", lambda: now[0])
    for i in range(stats.CACHE_MAX_ENTRIES * 3):
        stats._cache_put(f"key{i}", {"n": i})
    assert len(stats._cache) <= stats.CACHE_MAX_ENTRIES
    assert "key%d" % (stats.CACHE_MAX_ENTRIES * 3 - 1) in stats._cache
    # Expired entries are dropped at the next insert
    now[0] += stats.CACHE_TTL + 1
    stats._cache_put("fresh", {"n": 0})
    assert list(stats._cache) == ["fresh"]
    stats.clear_cache()

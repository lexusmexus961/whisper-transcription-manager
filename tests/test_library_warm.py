"""The Library's first view: base numbers and word analysis compute side by
side (single-flight per key, not one lock for all), and the server counts
the open folder ahead at its start."""

import threading
import time
from pathlib import Path

from whisper_tm.api import stats

STATIC = Path(__file__).resolve().parents[1] / "whisper_tm" / "static" / "js"


def test_different_keys_compute_side_by_side(monkeypatch):
    running, peak = [0], [0]
    lock = threading.Lock()

    def slow(prefix, with_text, library, folder):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return {"ok": with_text}

    monkeypatch.setattr(stats, "compute", slow)
    stats.clear_cache()
    ts = [
        threading.Thread(target=stats.cached, kwargs={"with_text": w, "folder": "/x"})
        for w in (False, True)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 2
    stats.clear_cache()


def test_same_key_computes_once(monkeypatch):
    calls = []
    monkeypatch.setattr(stats, "compute", lambda *a: calls.append(1) or time.sleep(0.1) or {"n": 1})
    stats.clear_cache()
    ts = [threading.Thread(target=stats.cached, kwargs={"folder": "/y"}) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1
    stats.clear_cache()


def test_warm_fills_numbers_words_and_speakers(monkeypatch):
    from whisper_tm.api import library

    seen = []
    monkeypatch.setattr(stats, "compute", lambda p, w, lib, f: seen.append((w, f)) or {"w": w})
    monkeypatch.setattr(library, "warm_speakers", lambda f: seen.append(("speakers", f)))
    stats.clear_cache()
    stats.warm("/lib/media")
    assert (
        (False, "/lib/media") in seen
        and (True, "/lib/media") in seen
        and ("speakers", "/lib/media") in seen
    )
    assert stats.cached(folder="/lib/media") == {"w": False}  # from the cache now
    assert len([s for s in seen if s[0] is False]) == 1
    stats.clear_cache()


def test_the_page_asks_for_who_speaks_with_the_numbers():
    js = (STATIC / "statsview.js").read_text(encoding="utf-8")
    i = js.index("async function stvLoad(")
    assert "stvSpeakersAhead();" in js[i : i + 1500]


def test_library_opens_on_the_open_folder_after_the_root_is_known():
    js = (STATIC / "library.js").read_text(encoding="utf-8")
    i = js.index("async function show(path)")
    body = js[i : js.index("function openFolderRel", i)]
    assert body.index("await level('')") < body.index("path = openFolderRel()")

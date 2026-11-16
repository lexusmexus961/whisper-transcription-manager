"""Resident models: a new language or model replaces the old one.

Previously the residency cache kept every alignment model (one per
language, ~1.2 GB each) and every Whisper weight set (turbo AND large-v3)
until the idle unloader — which never runs while the queue is busy. A queue
mixing de/en/ja piled them up on a 12-GB card until a file died of
CUDA OOM. Now a family keeps a bounded number of members, evicted when a
NEW key loads (least recently used first). A lane still using an evicted
model keeps its own reference — eviction only drops the cache's.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import transcription as tr


@pytest.fixture
def cache(monkeypatch):
    from whisper_tm.core import model_idle, vram

    monkeypatch.setattr(vram, "low_vram", lambda: False)
    monkeypatch.setattr(model_idle, "report_usage", lambda: None)
    monkeypatch.setattr(tr, "_models", {})
    monkeypatch.setattr(tr, "_holders", {})
    loads = []

    def load(key):
        def fn():
            loads.append(key)
            return object()

        return fn

    return loads, load


def _align(lang, load, keep=False):
    """One file's alignment on this lane — let go afterwards, as the worker
    does at the end of the stage (``keep`` holds on)."""
    m = tr._resident_model(("align", lang, "cuda"), load(("align", lang)))
    if not keep:
        tr.release_held()
    return m


def test_alignment_models_are_bounded(cache, monkeypatch):
    loads, load = cache
    monkeypatch.setattr(tr, "_family_limit", lambda family: 2)
    de = _align("de", load)
    _align("en", load)
    assert _align("de", load) is de  # still resident, no reload
    _align("ja", load)  # evicts en (least recently used)
    keys = [k for k in tr._models if k[0] == "align"]
    assert sorted(k[1] for k in keys) == ["de", "ja"]
    assert len(loads) == 3


def test_the_same_key_never_evicts(cache, monkeypatch):
    loads, load = cache
    monkeypatch.setattr(tr, "_family_limit", lambda family: 1)
    a = _align("de", load)
    for _ in range(3):
        assert _align("de", load) is a
    assert len(loads) == 1


def test_new_weights_take_their_pipelines_along(cache):
    loads, load = cache
    w_key = ("asr-weights", "turbo", "cuda", 0, "float16")
    tr._resident_model(w_key, load("turbo"))
    tr._resident_model(("asr", "turbo", "cuda", 0, "float16", "de", None, ()), load("p-de"))
    tr.release_held()
    tr._resident_model(("asr-weights", "large-v3", "cuda", 0, "float16"), load("large"))
    assert w_key not in tr._models
    assert not [k for k in tr._models if k[0] == "asr" and k[1] == "turbo"]
    assert ("asr-weights", "large-v3", "cuda", 0, "float16") in tr._models


def test_another_card_is_another_family(cache):
    loads, load = cache
    tr._resident_model(("align", "de", "cuda"), load("a"))
    tr._resident_model(("align", "de", "cpu"), load("b"))
    tr._resident_model(("asr-weights", "turbo", "cuda", 0, "f"), load("c"))
    tr._resident_model(("asr-weights", "turbo", "cuda", 1, "f"), load("d"))
    assert len(tr._models) == 4


# ── Never evict what a lane still uses ───────────────────────────────────
# Regression: an evicted model a running lane still referenced stayed
# in VRAM anyway — and the next file of that language loaded a SECOND copy.
# A model held by a lane (from its _resident_model call until the lane lets
# go, release_held) is kept; only models nobody uses are evicted.


def _in_other_lane(fn):
    import threading

    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", fn()))
    t.start()
    t.join(5)
    return out["v"]


def test_a_model_another_lane_uses_is_kept(cache, monkeypatch):
    loads, load = cache
    monkeypatch.setattr(tr, "_family_limit", lambda family: 1)
    import threading

    hold = threading.Event()
    done = threading.Event()
    got = {}

    def lane():
        got["de"] = _align("de", load, keep=True)  # holds it until it lets go
        hold.set()
        done.wait(5)
        tr.release_held()

    t = threading.Thread(target=lane)
    t.start()
    hold.wait(5)
    _align("en", load)  # this lane: de is in use elsewhere
    assert ("align", "de", "cuda") in tr._models
    assert _in_other_lane(lambda: _align("de", load)) is got["de"]  # no second copy
    done.set()
    t.join(5)
    tr.release_held()
    _align("ja", load)  # nobody uses de or en now
    assert [k[1] for k in tr._models if k[0] == "align"] == ["ja"]


def test_weights_in_use_are_kept(cache):
    import threading

    loads, load = cache
    turbo = ("asr-weights", "turbo", "cuda", 0, "float16")
    held, done = threading.Event(), threading.Event()

    def lane():
        tr._resident_model(turbo, load("turbo"))
        held.set()
        done.wait(5)
        tr.release_held()

    t = threading.Thread(target=lane)
    t.start()
    held.wait(5)
    tr._resident_model(("asr-weights", "large-v3", "cuda", 0, "float16"), load("large"))
    assert turbo in tr._models
    done.set()
    t.join(5)


def test_a_lane_that_died_holds_nothing(cache, monkeypatch):
    loads, load = cache
    monkeypatch.setattr(tr, "_family_limit", lambda family: 1)
    _in_other_lane(lambda: _align("de", load, keep=True))  # never lets go, ends
    _align("en", load)
    assert [k[1] for k in tr._models if k[0] == "align"] == ["en"]

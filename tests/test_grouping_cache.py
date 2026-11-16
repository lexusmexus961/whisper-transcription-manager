"""The result cache in :func:`speaker_knn.group_voices`.

Why it exists: On a library with tens of thousands of voices the fine cut
took many seconds, almost all of them in ONE large component. The effort is stuck
in a Python loop in sklearn and cannot be made faster — only rarer. So
`group_voices` holds ready runs when the caller confirms with ``marker`` that
the same marker means the same vectors.

What is checked here is therefore not "is it fast", but the two ways a cache
can be wrong: It answers a question that was not asked (false positive), or
its answer can be changed by the caller (shared state).
"""

import numpy as np
import pytest

from whisper_tm.core import speaker_knn as knn

DIM = 192


@pytest.fixture(autouse=True)
def clean():
    knn.clear_cache()
    yield
    knn.clear_cache()


def points(count: int = 6, per_group: int = 8, seed: int = 3) -> np.ndarray:
    """A small inventory with real group structure (see `people`)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(count, DIM)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    true_val = np.repeat(np.arange(count), per_group)
    noise = rng.normal(size=(count * per_group, DIM)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    return knn._nrm(Z[true_val] + 0.5 * noise)


def run(X, threshold=0.55, marker=None, link_method="average"):
    return knn.group_voices(X, threshold, link_method=link_method, device="cpu", marker=marker)


# ── Without marker, compute ────────────────────────────────────────────


def test_without_a_marker_no_cache():
    """The default case remains as it was: each call computes."""
    X = points()
    run(X)
    run(X)
    assert len(knn._result_cache) == 0


# ── With marker: same content, but not the same field ─────────────────


def test_the_second_run_comes_from_the_cache():
    X = points()
    a = run(X, marker=("db", "m", 1))
    assert len(knn._result_cache) == 1
    b = run(X, marker=("db", "m", 1))
    assert np.array_equal(a["markers"], b["markers"])
    assert a["groups"] == b["groups"]
    assert a["components"] == b["components"]


def test_a_cache_hit_really_stops_computing():
    """Proof that the cache hit does not enter the computation at all.

    A time comparison would be too noisy at 48 points. Instead, the fine cut
    is replaced with a trap after the first run: if it runs again, the test
    fails with announcement.
    """
    X = points()
    first = run(X, marker=("db", "m", 1))

    def trap(*a, **k):
        raise AssertionError("fine cut ran despite cached run")

    real, knn._finer = knn._finer, trap
    try:
        second_val = run(X, marker=("db", "m", 1))
    finally:
        knn._finer = real
    assert np.array_equal(first["markers"], second_val["markers"])


def test_the_output_is_not_shared():
    """Whoever modifies the markers must not corrupt the cached run.

    This is the error most likely to happen to a cache and detected latest:
    all callers get the same field, someone writes to it, and from then on
    the map shows the corrupted state — until the database happens to change.
    """
    X = points()
    a = run(X, marker=("db", "m", 1))
    original = a["markers"].copy()
    a["markers"][:] = 0
    a["groups"] = -1

    b = run(X, marker=("db", "m", 1))
    assert np.array_equal(b["markers"], original)
    assert b["groups"] != -1
    # And the second output is also its own again.
    assert b["markers"] is not a["markers"]


# ── What MUST miss the cache ─────────────────────────────────────────


@pytest.mark.parametrize(
    "second_one,why",
    [
        (("db", "m", 2), "different generation — database has changed"),
        (("other.db", "m", 1), "different database, same generation"),
        (("db", "other-model", 1), "different embedding"),
    ],
)
def test_a_different_marker_does_not_hit(second_one, why):
    """`generation` alone does not identify an inventory.

    Two different databases both reach 1 at some point — even regularly in
    tests. That is why the source is included in the marker.
    """
    X = points()
    run(X, marker=("db", "m", 1))
    run(X, marker=second_one)
    assert len(knn._result_cache) == 2, why


def test_a_different_threshold_does_not_hit():
    """The slider must work — otherwise it shows the same for every value."""
    X = points()
    a = run(X, threshold=0.35, marker=("db", "m", 1))
    b = run(X, threshold=0.85, marker=("db", "m", 1))
    assert len(knn._result_cache) == 2
    assert a["groups"] != b["groups"]


def test_a_different_linkage_does_not_hit():
    X = points()
    run(X, marker=("db", "m", 1), link_method="average")
    run(X, marker=("db", "m", 1), link_method="complete")
    assert len(knn._result_cache) == 2


def test_a_different_count_does_not_hit():
    """Last safeguard in case a caller chooses the marker too coarsely."""
    X = points()
    run(X, marker=("db", "m", 1))
    run(X[:20], marker=("db", "m", 1))
    assert len(knn._result_cache) == 2


# ── Cache must not grow ─────────────────────────────────────────


def test_the_cache_stays_capped():
    X = points()
    for g in range(knn.RESULT_CACHE_MAX + 5):
        run(X, marker=("db", "m", g))
    assert len(knn._result_cache) == knn.RESULT_CACHE_MAX


def test_the_oldest_falls_out_first():
    X = points()
    for g in range(knn.RESULT_CACHE_MAX):
        run(X, marker=("db", "m", g))
    # Touch the oldest again — it must NOT be the next one to fly.
    run(X, marker=("db", "m", 0))
    run(X, marker=("db", "m", 999))
    present = {s[0][2] for s in knn._result_cache}
    assert 0 in present
    assert 1 not in present


def test_clearing_cleans_up():
    X = points()
    run(X, marker=("db", "m", 1))
    knn.clear_cache()
    assert len(knn._result_cache) == 0


# ── Empty inventory ────────────────────────────────────────────────────


def test_an_empty_inventory_does_not_land_in_the_cache():
    """The shortcut above computes nothing — it must not store anything."""
    empty = np.zeros((0, DIM), dtype=np.float32)
    e = run(empty, marker=("db", "m", 1))
    assert e["nodes"] == 0
    assert len(knn._result_cache) == 0


# ── And the source the caller relies on ───────────────────


def test_the_inventory_names_its_source(tmp_path, monkeypatch):
    """`speaker_inventory.inventory()` must provide the source.

    The map builds its marker from this. If the key is missing, a KeyError
    flies there — this test makes it a named failure.
    """
    from whisper_tm.core import speaker_inventory
    from whisper_tm.database.speaker import SpeakerDB

    db = SpeakerDB(str(tmp_path / "speaker.db"))
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    speaker_inventory.discard()
    try:
        state_info = speaker_inventory.inventory()
        assert state_info["source"] == db._db_path
        assert "generation" in state_info and "model" in state_info
    finally:
        speaker_inventory.discard()

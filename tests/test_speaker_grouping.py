"""Neighborhood, grouping and run storage.

Everything on CPU and on invented vectors with REAL structure — a known
pitfall when measuring at scale:
``Z + 0.35 * noise`` looks harmless, but 192-dimensional noise has
length 0.35·√192 ≈ 4.85 against a centroid of length 1. The "persons"
were completely drowned, and both grouping methods dutifully reported
"each point its own group" — which looked like a result.
That's why :func:`persons` builds the disturbance here normalized to its LENGTH.
"""

import numpy as np
import pytest

from whisper_tm.core import speaker_knn as knn
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def persons(
    count: int, each: int, scatter: float = 0.5, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """``count`` persons with ``each`` windows each. Returns (X, truth)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(count, DIM)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    truth = np.repeat(np.arange(count), each)
    noise = rng.normal(size=(count * each, DIM)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    X = Z[truth] + scatter * noise
    return knn._nrm(X), truth


def _same_partition(a, b) -> bool:
    """Two markings describe the same partition (numbers don't matter)."""
    a, b = np.asarray(a), np.asarray(b)
    return len(np.unique(a)) == len(np.unique(b)) and all(
        len(np.unique(b[a == m])) == 1 for m in np.unique(a)
    )


# ── Neighborhood ────────────────────────────────────────────────────────


def test_neighbors_find_own_person():
    X, truth = persons(8, 12)
    idx, dist, _ = knn.neighbours(X, k=5, device="cpu")
    assert idx.shape == (96, 5)
    # the five best neighbors belong to own person
    hits = (truth[idx] == truth[:, None]).mean()
    assert hits > 0.95, f"only {hits:.2f} of neighbors are correct"


def test_neighbors_exclude_self():
    X, _ = persons(4, 6)
    idx, dist, _ = knn.neighbours(X, k=3, device="cpu")
    for i in range(len(X)):
        assert i not in idx[i]
    assert dist.max() <= 1.001


def test_neighbors_sorted_descending():
    X, _ = persons(4, 6)
    _idx, dist, _ = knn.neighbours(X, k=4, device="cpu")
    assert (np.diff(dist, axis=1) <= 1e-5).all()


def test_neighbors_with_empty_inventory():
    idx, dist, _ = knn.neighbours(np.zeros((0, DIM), np.float32), device="cpu")
    assert idx.shape[0] == 0 and dist.shape[0] == 0


def test_neighbors_with_single_vector():
    """k is capped to n-1 — otherwise topk raises."""
    X = knn._nrm(np.random.default_rng(0).normal(size=(1, DIM)).astype(np.float32))
    idx, _dist, _ = knn.neighbours(X, k=30, device="cpu")
    assert idx.shape == (1, 0)


# ── Step 1: Connected components ──────────────────────────────────────


def test_components_separate_clearly_separated_persons():
    X, truth = persons(6, 10, scatter=0.35)
    idx, dist, _ = knn.neighbours(X, k=8, device="cpu")
    comp = knn.components(idx, dist, len(X), threshold=0.6)
    assert _same_partition(comp, truth)


def test_components_without_edges_each_is_separate():
    X, _ = persons(6, 10)
    idx, dist, _ = knn.neighbours(X, k=8, device="cpu")
    comp = knn.components(idx, dist, len(X), threshold=0.999)
    assert len(np.unique(comp)) == len(X)


# ── Step 2: Two-stage run ──────────────────────────────────────────────


def test_grouping_finds_built_in_persons():
    X, truth = persons(10, 15, scatter=0.4)
    e = knn.group_voices(X, threshold=0.6, k=10, device="cpu")
    assert e["nodes"] == 150
    assert _same_partition(e["markers"], truth), f"{e['groups']} groups instead of 10"


def test_fine_cut_separates_what_component_throws_together():
    """The core of the two-stage procedure.

    Two persons are connected via a bridge of intermediate points —
    exactly the chain that single linkage fails on. Step 1 throws
    both in ONE component, step 2 (`average`) separates them again.
    """
    rng = np.random.default_rng(3)
    a = knn._nrm(rng.normal(size=(1, DIM)).astype(np.float32))[0]
    b = knn._nrm(rng.normal(size=(1, DIM)).astype(np.float32))[0]

    def cloud(center, n, s):
        st = rng.normal(size=(n, DIM)).astype(np.float32)
        st /= np.linalg.norm(st, axis=1, keepdims=True)
        return knn._nrm(center + s * st)

    # two clouds plus a chain of bridge points between them
    bridge = knn._nrm(np.stack([(1 - t) * a + t * b for t in np.linspace(0.2, 0.8, 12)]))
    X = np.vstack([cloud(a, 25, 0.35), cloud(b, 25, 0.35), bridge])

    idx, dist, _ = knn.neighbours(X, k=10, device="cpu")
    comp = knn.components(idx, dist, len(X), threshold=0.6)
    assert len(np.unique(comp)) < 3, "bridge should chain everything"

    e = knn.group_voices(X, threshold=0.6, k=10, device="cpu")
    assert e["components"] < e["groups"], "fine cut did not split the chained component"


def test_grouping_reports_its_metrics():
    X, _ = persons(5, 8)
    e = knn.group_voices(X, threshold=0.6, k=6, device="cpu")
    for field in ("markers", "nodes", "groups", "components", "largest", "loners", "duration_sec"):
        assert field in e
    assert e["groups"] >= 1
    assert e["unsplit"] == []


def test_grouping_with_empty_inventory():
    e = knn.group_voices(np.zeros((0, DIM), np.float32), threshold=0.6, device="cpu")
    assert e["nodes"] == 0 and e["groups"] == 0


def test_progress_is_reported():
    X, _ = persons(4, 6)
    reported = []
    knn.group_voices(
        X, threshold=0.6, k=5, device="cpu", progress=lambda t, a: reported.append((t, a))
    )
    assert reported and reported[-1][1] == 1.0


# ── Arrangement ─────────────────────────────────────────────────────────


def test_arrangement_is_same_between_two_runs():
    """A force simulation would be in a different place each run — then the
    user would have to find their group anew each time."""
    marks = np.array([0, 0, 0, 1, 1, 2])
    x1, y1 = knn.arrange(marks)
    x2, y2 = knn.arrange(marks)
    assert np.allclose(x1, x2) and np.allclose(y1, y2)


def test_large_groups_are_inside():
    marks = np.repeat([0, 1, 2, 3], [40, 20, 10, 2])
    x, y = knn.arrange(marks)
    r = np.hypot(x, y)
    assert r[0] < r[3], "largest group belongs in the middle"


# ── Run storage ───────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    return d


# ── Operating point fixed ──────────────────────────────────────────────


def test_stripe_width_stays_small():
    """4,096 rows against a few hundred thousand vectors yield a VRAM peak
    of several GB and knock out any concurrent transcription."""
    assert knn.STRIPE <= 1024

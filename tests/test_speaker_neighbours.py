"""The exact neighbourhood (core/speaker_neighbours.py).

The GPU only proposes candidates; exact float64 values decide, ties by the
lower index. So the result is a function of the vectors alone: the GPU and
the CPU path agree to the bit, and an update after new or deleted voices
equals a computation from scratch — which is what lets the map follow a
running queue without recomputing hundreds of thousands squared
similarities every few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import speaker_neighbours as sn

pytestmark = pytest.mark.skipif(not sn.HAVE_NUMBA, reason="numba not installed")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


DEVICES = ["cpu"] + (["cuda"] if _cuda() else [])


def _unit(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def _voices(n: int, seed: int, dim: int = 48, dups: int = 0) -> np.ndarray:
    """Clustered voices; ``dups`` exact copies of one voice (ties)."""
    r = np.random.default_rng(seed)
    centres = r.normal(size=(max(2, n // 25), dim))
    X = centres[r.integers(0, len(centres), n)] + 0.35 * r.normal(size=(n, dim))
    if dups:
        X[r.choice(n, dups, replace=False)] = X[0]
    return _unit(X)


def _brute(X: np.ndarray, k: int):
    """Every pair exactly, ranked by value then index."""
    X64 = X.astype(np.float64)
    n = len(X)
    k = min(k, n - 1)
    idx = np.empty((n, k), np.int32)
    val = np.empty((n, k), np.float64)
    for r in range(n):
        # Term by term, like the module. (Python's `sum` would not do:
        # since 3.12 it compensates, and the last bit differs.)
        s = np.zeros(n)
        for t in range(X.shape[1]):
            s += X64[r, t] * X64[:, t]
        s[r] = -np.inf
        order = sorted(range(n), key=lambda j: (-s[j], j))[:k]
        idx[r] = order
        val[r] = s[order]
    return idx, val


class TestExact:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "n,k,seed", [(12, 5, 1), (40, 30, 2), (300, 30, 3), (700, 10, 4), (900, 30, 5)]
    )
    def test_against_every_pair(self, device, n, k, seed):
        X = _voices(n, seed)
        idx, v64, _ = sn.compute(X, k, device)
        want_i, want_v = _brute(X, k)
        assert np.array_equal(idx, want_i)
        assert np.array_equal(v64, want_v)

    @pytest.mark.parametrize("device", DEVICES)
    def test_more_duplicates_than_candidates(self, device):
        """200 copies of one voice: the candidates can't hold them all, the
        bound fails, and the row falls back to everything within it."""
        X = _voices(800, 6, dups=200)
        idx, v64, _ = sn.compute(X, 30, device)
        want_i, want_v = _brute(X, 30)
        assert np.array_equal(idx, want_i)
        assert np.array_equal(v64, want_v)

    @pytest.mark.skipif(not _cuda(), reason="no GPU")
    def test_gpu_and_cpu_agree_to_the_bit(self):
        X = _voices(3000, 7, dim=192, dups=40)
        a = sn.compute(X, 30, "cuda")
        b = sn.compute(X, 30, "cpu")
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_bound_that_is_too_tight_is_widened(self, device, monkeypatch):
        monkeypatch.setattr(sn, "EPS_GPU", 1e-12)
        monkeypatch.setattr(sn, "EPS_CPU", 1e-12)
        X = _voices(500, 8, dups=20)
        idx, v64, _ = sn.compute(X, 30, device)
        want_i, want_v = _brute(X, 30)
        assert np.array_equal(idx, want_i) and np.array_equal(v64, want_v)

    def test_the_first_k_are_the_k_strongest(self):
        X = _voices(400, 9)
        i30, v30, _ = sn.compute(X, 30, "cpu")
        i5, v5, _ = sn.compute(X, 5, "cpu")
        assert np.array_equal(i30[:, :5], i5) and np.array_equal(v30[:, :5], v5)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_tiny_inventories(self, n):
        X = _voices(max(n, 2), 10)[:n]
        idx, v64, _ = sn.compute(X, 30, "cpu")
        assert idx.shape == (n, max(0, min(30, n - 1)))


class TestUpdate:
    """An update equals a computation from scratch — to the bit.

    The small test sets are dense: one deleted voice sits in about thirty
    rows, so most rows would be recomputed — where the module rightly
    prefers to start over. The share is lifted here so the update path
    itself is what gets checked; `test_too_much_changed_says_so` covers
    the share.
    """

    @pytest.fixture(autouse=True)
    def _always_update(self, monkeypatch):
        monkeypatch.setattr(sn, "UPDATE_AT_MOST", 1.0)

    @staticmethod
    def _check(X_old, ids_old, X_new, ids_new, device, k=30):
        pi, pv, _ = sn.compute(X_old, k, device)
        got = sn.update(ids_old, pi, pv, X_new, ids_new, device)
        assert got is not None
        want = sn.compute(X_new, k, device)
        assert np.array_equal(got[0], want[0])
        assert np.array_equal(got[1], want[1])

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("seed", range(4))
    def test_new_voices(self, device, seed):
        X = _voices(1200, 20 + seed, dups=15)
        ids = np.arange(1200, dtype=np.int64) * 3
        old = np.random.default_rng(seed).random(1200) > 0.02
        self._check(X[old], ids[old], X, ids, device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("seed", range(4))
    def test_deleted_voices(self, device, seed):
        X = _voices(1200, 30 + seed, dups=15)
        ids = np.arange(1200, dtype=np.int64) * 3
        keep = np.random.default_rng(seed).random(1200) > 0.03
        self._check(X, ids, X[keep], ids[keep], device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_both_at_once_and_in_the_middle(self, device):
        """A re-transcribed file: its old voices go, new ones come — and
        not only at the end of the id order."""
        X = _voices(1500, 40, dups=10)
        ids = np.arange(1500, dtype=np.int64) * 2
        r = np.random.default_rng(40)
        before = r.random(1500) > 0.03
        after = r.random(1500) > 0.03
        self._check(X[before], ids[before], X[after], ids[after], device)

    def test_nothing_changed(self):
        X = _voices(500, 41)
        ids = np.arange(500, dtype=np.int64)
        self._check(X, ids, X, ids, "cpu")

    def test_too_much_changed_says_so(self, monkeypatch):
        monkeypatch.setattr(sn, "UPDATE_AT_MOST", 0.5)
        X = _voices(600, 42)
        ids = np.arange(600, dtype=np.int64)
        pi, pv, _ = sn.compute(X[:200], 30, "cpu")
        assert sn.update(ids[:200], pi, pv, X, ids, "cpu") is None

    def test_unsorted_ids_are_not_updated(self):
        X = _voices(300, 43)
        ids = np.arange(300, dtype=np.int64)[::-1].copy()
        pi, pv, _ = sn.compute(X, 30, "cpu")
        assert sn.update(ids, pi, pv, X, ids, "cpu") is None

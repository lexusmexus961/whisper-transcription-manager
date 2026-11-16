"""sklearn's structured agglomerative clustering, compiled
(core/speaker_agglomerative.py).

The map cut its large components with sklearn's merge loop, which runs in
Python: most of the map's grouping time in one measurement. The compiled loop
replays it step by step. These tests hold it to the labels of the
installed sklearn — ties, chains, both linkages — and to handing back
what it does not replay.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from whisper_tm.core import speaker_agglomerative as sa

pytestmark = pytest.mark.skipif(not sa.HAVE_NUMBA, reason="numba not installed")


def _graph(X: np.ndarray, k: int, path: bool = False):
    """The kNN graph as the map builds it; ``path`` adds i—i+1 so it is
    connected (clustered voices otherwise fall into islands — and a test
    over islands would compare nothing, see `test_the_labels_of_sklearn`)."""
    S = X @ X.T
    np.fill_diagonal(S, -2)
    nbr = np.argsort(-S, axis=1, kind="stable")[:, :k]
    m = len(X)
    rows, cols = np.repeat(np.arange(m), k), nbr.ravel()
    if path:
        rows = np.concatenate([rows, np.arange(m - 1)])
        cols = np.concatenate([cols, np.arange(1, m)])
    G = coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(m, m)).tocsr()
    return G.maximum(G.T)


def _voices(seed: int, m: int, dim: int = 24, dups: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    centres = r.normal(size=(max(2, m // 20), dim))
    X = centres[r.integers(0, len(centres), m)] + r.uniform(0.2, 0.7) * r.normal(size=(m, dim))
    if dups:
        X[r.choice(m, dups, replace=False)] = X[0]
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def _sklearn(X, G, d_cut, method):
    from sklearn.cluster import AgglomerativeClustering

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=d_cut,
            metric="euclidean",
            linkage=method,
            connectivity=G,
        ).fit_predict(X)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("method", ["average", "complete"])
def test_the_labels_of_sklearn(seed, method):
    X = _voices(seed, 150 + 40 * seed, dups=seed * 3)
    G = _graph(X, 5 + seed % 6, path=True)
    for d_cut in (0.2, 0.45, 0.7, 1.0, 1.4):
        mine = sa.fit_predict(X, G, d_cut, method)
        assert mine is not None, "handed back — this test would check nothing"
        assert np.array_equal(mine, _sklearn(X, G, d_cut, method)), d_cut


def test_many_equal_distances():
    """Voices on a grid: whole rows of identical merge distances, where
    only the heap's order decides who merges first."""
    g = np.stack(np.meshgrid(np.arange(12), np.arange(12)), -1).reshape(-1, 2)
    X = np.hstack([g, np.full((len(g), 1), 20.0)]).astype(np.float64)
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    G = _graph(X, 4)
    for d_cut in (0.001, 0.01, 0.05):
        for method in ("average", "complete"):
            mine = sa.fit_predict(X, G, d_cut, method)
            assert mine is not None
            assert np.array_equal(mine, _sklearn(X, G, d_cut, method))


def test_what_is_not_replayed_is_handed_back():
    X = _voices(1, 120)
    G = _graph(X, 6)
    assert sa.fit_predict(X, G, 0.5, "single") is None
    # Two islands: sklearn would add edges of its own
    a = _voices(2, 60)
    far = np.zeros((60, 24), np.float32)
    far[:, 0] = 1
    b = (far + 0.01 * np.random.default_rng(3).normal(size=far.shape)).astype(np.float32)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    X2 = np.vstack([a, b])
    G2 = coo_matrix(_graph(a, 5))
    G2 = coo_matrix(
        (
            np.concatenate([G2.data, G2.data]),
            (np.concatenate([G2.row, G2.row + 60]), np.concatenate([G2.col, G2.col + 60])),
        ),
        shape=(120, 120),
    ).tocsr()
    assert sa.fit_predict(X2, G2, 0.5, "average") is None


def test_the_map_uses_it(monkeypatch):
    """`_finer` above FULL_UNTIL goes through the compiled loop, and gets
    what sklearn would have given."""
    from whisper_tm.core import speaker_knn as sk

    monkeypatch.setattr(sk, "FULL_UNTIL", 50)
    X = _voices(9, 300)
    calls = []
    real = sa.fit_predict

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(sa, "fit_predict", spy)
    got = sk._finer(X, 0.8, "average")
    assert calls == [True]
    monkeypatch.setattr(sa, "fit_predict", lambda *a, **k: None)
    assert np.array_equal(got, sk._finer(X, 0.8, "average"))

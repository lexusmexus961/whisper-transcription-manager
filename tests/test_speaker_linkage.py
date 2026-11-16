"""Complete linkage without the distance matrix (core/speaker_linkage.py).

The guided assignment used to link one chain-connected component of tens
of thousands of centroids through scipy/fastcluster: an n² matrix and its
copies, tens of GB of RAM at peak — more than most machines have. The
sparse linkage looks only at the pairs within the cut — on the same
centroids: far less time and memory, and the identical partition.

These tests hold it to the matrix path's result and to its memory promise.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from whisper_tm.core import speaker_assignment as assignment
from whisper_tm.core import speaker_linkage as sl

pytestmark = pytest.mark.skipif(not sl.HAVE_NUMBA, reason="numba not installed")


def _unit(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def _matrix_path(C: np.ndarray, cut: float) -> np.ndarray:
    """The reference: scipy complete linkage on the full distance matrix."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(C) < 2:
        return np.ones(len(C), dtype=int)
    D = 1.0 - (C @ C.T)
    np.fill_diagonal(D, 0.0)
    np.clip(D, 0.0, 2.0, out=D)
    return fcluster(
        linkage(squareform(D, checks=False), method="complete"), t=1.0 - cut, criterion="distance"
    )


def _partition(labels) -> list[tuple]:
    groups: dict = {}
    for i, g in enumerate(np.asarray(labels).tolist()):
        groups.setdefault(g, []).append(i)
    return sorted(tuple(v) for v in groups.values())


def _clusters(seed: int, dim: int = 192) -> np.ndarray:
    r = np.random.default_rng(seed)
    cores = r.normal(size=(int(r.integers(2, 12)), dim))
    return _unit(
        np.vstack(
            [c + r.uniform(0.2, 0.6) * r.normal(size=(int(r.integers(3, 20)), dim)) for c in cores]
        )
    )


class TestSameResultAsTheMatrix:
    @pytest.mark.parametrize("seed", range(40))
    @pytest.mark.parametrize("cut", [0.3, 0.6, 0.71, 0.8])
    def test_clustered_voices(self, seed, cut):
        C = _clusters(seed, dim=16 if seed % 2 else 192)
        assert _partition(sl.complete_groups(C, cut)) == _partition(_matrix_path(C, cut))

    @pytest.mark.parametrize("seed", range(10))
    def test_one_long_chain(self, seed):
        """The real shape: everything connected through chains — the case
        that needed the full n² matrix."""
        r = np.random.default_rng(100 + seed)
        steps = np.cumsum(0.15 * r.normal(size=(120, 24)), axis=0) + 3.0
        C = _unit(steps)
        assert _partition(sl.complete_groups(C, 0.9)) == _partition(_matrix_path(C, 0.9))

    def test_the_chain_still_breaks_apart(self):
        # A~B, B~C, but A and C unrelated: complete linkage keeps A and C apart
        a = np.zeros(8)
        a[0] = 1
        c = np.zeros(8)
        c[1] = 1
        C = _unit([a, a + c, c])
        g = sl.complete_groups(C, 0.7)
        assert g[0] != g[2]
        assert _partition(g) == _partition(_matrix_path(C, 0.7))

    def test_nothing_within_the_cut(self):
        C = _unit(np.eye(20)[:, :20])
        assert len(set(sl.complete_groups(C, 0.5).tolist())) == 20

    def test_everything_within_the_cut(self):
        C = _unit(np.ones((30, 8)) + 1e-3 * np.random.default_rng(1).normal(size=(30, 8)))
        assert len(set(sl.complete_groups(C, 0.5).tolist())) == 1

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_tiny_inputs(self, n):
        C = _unit(np.random.default_rng(2).normal(size=(max(n, 1), 8)))[:n]
        g = sl.complete_groups(C, 0.5)
        assert len(g) == n and (n == 0 or g.min() == 1)


class TestTies:
    def test_duplicates_merge_first_and_stay_together(self):
        # Identical centroids (the same recording twice) are at distance 0
        r = np.random.default_rng(3)
        base = _unit(r.normal(size=(6, 32)))
        C = np.vstack([base, base, base[:2]])
        g = sl.complete_groups(C, 0.8)
        for i in range(6):
            assert g[i] == g[6 + i]

    def test_the_same_input_gives_the_same_groups(self):
        C = _clusters(7)
        assert np.array_equal(sl.complete_groups(C, 0.71), sl.complete_groups(C, 0.71))


class TestTheEdgeGraph:
    def test_symmetric_sorted_and_complete(self):
        C = _clusters(11, dim=32)
        tau = 1.0 - 0.6
        indptr, nbr, dist = sl.edges(C, tau)
        n = len(C)
        want = set()
        S = C @ C.T
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = max(np.float32(1.0) - S[i, j], np.float32(0.0))
                    if float(d) <= tau:
                        want.add((i, j))
        got = set()
        for i in range(n):
            row = nbr[indptr[i] : indptr[i + 1]]
            assert np.all(np.diff(row) > 0), "row not sorted"
            assert i not in row, "self loop"
            got.update((i, int(j)) for j in row)
        assert got == want
        assert all((j, i) in got for i, j in got)

    def test_blocks_change_nothing_but_the_last_bit(self, monkeypatch):
        """BLAS rounds a product in the last bit depending on the block
        shape. That may move a distance by float32 rounding — and so a pair
        lying exactly at the cut — never more (the matrix path had the same
        property)."""
        C = _clusters(12, dim=32)
        tau = 0.3

        def as_dict(csr):
            indptr, nbr, dist = csr
            return {
                (i, int(nbr[k])): float(dist[k])
                for i in range(len(indptr) - 1)
                for k in range(indptr[i], indptr[i + 1])
            }

        a = as_dict(sl.edges(C, tau))
        monkeypatch.setattr(sl, "BLOCK_BYTES", 4096)
        b = as_dict(sl.edges(C, tau))
        for pair in a.keys() ^ b.keys():
            assert abs((a.get(pair) or b.get(pair)) - tau) < 1e-6, pair
        for pair in a.keys() & b.keys():
            assert abs(a[pair] - b[pair]) < 1e-6, pair

    def test_a_cut_far_too_low_stops_instead_of_eating_memory(self):
        C = _unit(np.ones((200, 8)) + 1e-2 * np.random.default_rng(4).normal(size=(200, 8)))
        with pytest.raises(sl.TooManyEdges):
            sl.edges(C, 1.0 - 0.1, max_edges=1000)


def test_no_distance_matrix_is_built(monkeypatch):
    """The promise of the module: memory grows with the close pairs, not
    with centroids². 3,000 centroids as a matrix would be 36 MB."""
    monkeypatch.setattr(sl, "BLOCK_BYTES", 1 << 20)
    r = np.random.default_rng(5)
    C = _unit(r.normal(size=(3000, 64)))
    sl.complete_groups(C[:100], 0.5)  # compile outside the measurement
    tracemalloc.start()
    sl.complete_groups(C, 0.5)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 8 << 20, f"peak {peak / 2**20:.1f} MB — a dense matrix is back?"


class TestTheAssignmentUsesIt:
    def test_the_sparse_path_links(self, monkeypatch):
        """No scipy/fastcluster linkage on the way — that was the tens-of-GB peak."""
        import scipy.cluster.hierarchy as h

        monkeypatch.setattr(h, "linkage", lambda *a, **k: pytest.fail("matrix linkage used"))
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.6)
        C = _clusters(21, dim=32)
        g = assignment.Inventory._concat(C)
        monkeypatch.undo()
        assert _partition(g) == _partition(_matrix_path(C, 0.6))

    def test_without_numba_the_matrix_path_still_runs(self, monkeypatch):
        monkeypatch.setattr(sl, "HAVE_NUMBA", False)
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.6)
        C = _clusters(22, dim=32)
        assert _partition(assignment.Inventory._concat(C)) == _partition(_matrix_path(C, 0.6))

    def test_the_centroids_are_built_as_before(self):
        """The vectorized centroid step against the loop it replaced."""
        r = np.random.default_rng(23)
        X = _unit(r.normal(size=(400, 16)))
        meta = [
            {
                "id": 1000 + i,
                "abs_path": f"/m/f{i % 13}.mkv",
                "label": f"SPEAKER_0{i % 3}",
                "speech_sec": float(r.uniform(0, 9)),
            }
            for i in range(400)
        ]
        key, C, speech, windows, vectors = assignment.Inventory._centroids(X, meta)
        buckets: dict = {}
        for i, m in enumerate(meta):
            buckets.setdefault((m["abs_path"], m["label"]), []).append(i)
        want = [
            (k, idx)
            for k, idx in buckets.items()
            if sum(meta[i]["speech_sec"] for i in idx) >= assignment.MIN_SPEECH_SEC
        ]
        assert key == [k for k, _ in want]
        assert vectors == [[meta[i]["id"] for i in idx] for _, idx in want]
        assert windows.tolist() == [len(idx) for _, idx in want]
        ref = np.stack([assignment._nrm(X[idx].mean(axis=0)) for _, idx in want])
        assert np.abs(C - ref).max() < 1e-6
        assert np.allclose(speech, [sum(meta[i]["speech_sec"] for i in idx) for _, idx in want])

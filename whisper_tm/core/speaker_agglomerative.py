"""sklearn's structured agglomerative clustering, compiled — same result.

The voice map cuts each connected component of up to 50,000 voices with
``AgglomerativeClustering(linkage="average", connectivity=kNN graph)``.
sklearn runs that merge loop in *Python*: in one measurement a few components
of several thousand voices each took about two thirds of the map's grouping
time — and the queue changes the voices every few seconds.

This module runs the same loop compiled. Not a similar algorithm: the same
steps in the same order, so the labels are identical to sklearn's:

* the distances of the connected pairs come from sklearn itself
  (``paired_distances``), in blocks;
* the merge loop replays sklearn's ``linkage_tree``: its heap of edges
  compared by weight alone — with CPython's ``heapq`` rules for equal
  weights (``_siftdown``/``_siftup``) —, its skipping of merged nodes, its
  weighted mean (``average_merge``) or maximum (``max_merge``) of a
  merged cluster's distances, and the order its neighbours are pushed in;
* the tree is cut like ``_hc_cut``: the number of clusters from the merge
  distances at or above the threshold, the labels in the order of
  ``heapq``'s list of negated node numbers.

What it does not replay it hands back (``None``): single linkage, and a
connectivity graph in several pieces (sklearn then adds edges of its own).
A self-test against the installed sklearn switches it off if a new
sklearn version ever computed differently.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import numba as nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover — a lean install
    nb = None
    HAVE_NUMBA = False

#: Pairs per block when sklearn computes the distances (memory: two
#: blocks of this many rows × dim float32).
PAIR_BLOCK = 1 << 17

_checked: bool | None = None
_check_lock = threading.Lock()

if HAVE_NUMBA:
    _jit = nb.njit(cache=True)
else:  # pragma: no cover

    def _jit(f):
        return f


# ── CPython's heapq, on parallel arrays, compared by weight alone ─────────


@_jit
def _siftdown(hw, ha, hb, startpos, pos):
    nw, na, nb_ = hw[pos], ha[pos], hb[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        if nw < hw[parentpos]:
            hw[pos] = hw[parentpos]
            ha[pos] = ha[parentpos]
            hb[pos] = hb[parentpos]
            pos = parentpos
            continue
        break
    hw[pos] = nw
    ha[pos] = na
    hb[pos] = nb_


@_jit
def _siftup(hw, ha, hb, endpos, pos):
    startpos = pos
    nw, na, nb_ = hw[pos], ha[pos], hb[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not hw[childpos] < hw[rightpos]:
            childpos = rightpos
        hw[pos] = hw[childpos]
        ha[pos] = ha[childpos]
        hb[pos] = hb[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    hw[pos] = nw
    ha[pos] = na
    hb[pos] = nb_
    _siftdown(hw, ha, hb, startpos, pos)


@_jit
def _isiftdown(h, startpos, pos):
    new = h[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        if new < h[parentpos]:
            h[pos] = h[parentpos]
            pos = parentpos
            continue
        break
    h[pos] = new


@_jit
def _isiftup(h, endpos, pos):
    startpos = pos
    new = h[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not h[childpos] < h[rightpos]:
            childpos = rightpos
        h[pos] = h[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    h[pos] = new
    _isiftdown(h, startpos, pos)


# ── linkage_tree ─────────────────────────────────────────────────────────


@_jit
def _tree(m, indptr, indices, dist, average, children, distances):
    """The merge loop of sklearn's ``linkage_tree`` (connectivity given,
    full tree). ``children[t] = (b, a)`` of the t-th merged edge — sklearn's
    reversed pair — and ``distances[t]`` its weight. Returns False if the
    graph fell apart before the tree was complete."""
    n_nodes = 2 * m - 1
    keys = [np.empty(0, np.int64) for _ in range(n_nodes)]
    vals = [np.empty(0, np.float64) for _ in range(n_nodes)]
    length = np.zeros(n_nodes, np.int64)
    for x in range(m):
        s, e = indptr[x], indptr[x + 1]
        kk = np.empty(e - s + 4, np.int64)
        vv = np.empty(e - s + 4, np.float64)
        for p in range(s, e):
            kk[p - s] = indices[p]
            vv[p - s] = dist[p]
        keys[x] = kk
        vals[x] = vv
        length[x] = e - s

    cap = max(16, indptr[m] // 2 + 2 * m)
    hw = np.empty(cap, np.float64)
    ha = np.empty(cap, np.int64)
    hb = np.empty(cap, np.int64)
    size = 0
    # The upper triangle, row by row — the order sklearn builds its list in
    for ind in range(m):
        for p in range(indptr[ind], indptr[ind + 1]):
            r = indices[p]
            if r < ind:
                hw[size] = dist[p]
                ha[size] = ind
                hb[size] = r
                size += 1
    for i in range(size // 2 - 1, -1, -1):
        _siftup(hw, ha, hb, size, i)

    used = np.ones(n_nodes, np.int64)
    for k in range(m, n_nodes):
        while True:
            if size == 0:
                return False
            # heappop
            size -= 1
            lw, la, lb = hw[size], ha[size], hb[size]
            if size > 0:
                w, a, b = hw[0], ha[0], hb[0]
                hw[0] = lw
                ha[0] = la
                hb[0] = lb
                _siftup(hw, ha, hb, size, 0)
            else:
                w, a, b = lw, la, lb
            if used[a] != 0 and used[b] != 0:
                break
        i, j = a, b
        distances[k - m] = w
        children[k - m, 0] = j
        children[k - m, 1] = i
        n_i = used[i]
        n_j = used[j]
        used[k] = n_i + n_j
        used[i] = 0
        used[j] = 0

        # average_merge / max_merge of A[i] and A[j], masked by `used`
        ki, vi, li = keys[i], vals[i], length[i]
        kj, vj, lj = keys[j], vals[j], length[j]
        ok = np.empty(li + lj + 4, np.int64)
        ov = np.empty(li + lj + 4, np.float64)
        fi = np.float64(n_i)
        fj = np.float64(n_j)
        fo = np.float64(n_i + n_j)
        t = 0
        p = 0
        q = 0
        while p < li or q < lj:
            if q >= lj or (p < li and ki[p] < kj[q]):
                key = ki[p]
                if used[key] != 0:
                    ok[t] = key
                    ov[t] = vi[p]
                    t += 1
                p += 1
            elif p >= li or kj[q] < ki[p]:
                key = kj[q]
                if used[key] != 0:
                    ok[t] = key
                    ov[t] = vj[q]
                    t += 1
                q += 1
            else:
                key = ki[p]
                if used[key] != 0:
                    ok[t] = key
                    if average:
                        ov[t] = (fi * vi[p] + fj * vj[q]) / fo
                    else:
                        ov[t] = max(vi[p], vj[q])
                    t += 1
                p += 1
                q += 1
        keys[k] = ok
        vals[k] = ov
        length[k] = t
        keys[i] = np.empty(0, np.int64)
        vals[i] = np.empty(0, np.float64)
        keys[j] = np.empty(0, np.int64)
        vals[j] = np.empty(0, np.float64)
        length[i] = 0
        length[j] = 0

        for s in range(t):
            col = ok[s]
            d = ov[s]
            # A[col].append(k, d): k is larger than every key in there
            if length[col] == len(keys[col]):
                grown = max(8, 2 * len(keys[col]))
                k2 = np.empty(grown, np.int64)
                v2 = np.empty(grown, np.float64)
                k2[: length[col]] = keys[col][: length[col]]
                v2[: length[col]] = vals[col][: length[col]]
                keys[col] = k2
                vals[col] = v2
            keys[col][length[col]] = k
            vals[col][length[col]] = d
            length[col] += 1
            # heappush(WeightedEdge(d, k, col))
            if size == len(hw):
                hw2 = np.empty(2 * size, np.float64)
                ha2 = np.empty(2 * size, np.int64)
                hb2 = np.empty(2 * size, np.int64)
                hw2[:size] = hw
                ha2[:size] = ha
                hb2[:size] = hb
                hw, ha, hb = hw2, ha2, hb2
            hw[size] = d
            ha[size] = k
            hb[size] = col
            size += 1
            _siftdown(hw, ha, hb, 0, size - 1)
    return True


@_jit
def _cut(n_clusters, children, n_leaves):
    """sklearn's ``_hc_cut``: labels in the order of heapq's node list."""
    h = np.empty(n_clusters + 1, np.int64)
    last = children[len(children) - 1]
    h[0] = -(max(last[0], last[1]) + 1)
    size = 1
    for _ in range(n_clusters - 1):
        top = -h[0]
        c0 = children[top - n_leaves, 0]
        c1 = children[top - n_leaves, 1]
        # heappush(nodes, -c0)
        h[size] = -c0
        size += 1
        _isiftdown(h, 0, size - 1)
        # heappushpop(nodes, -c1)
        item = -c1
        if h[0] < item:
            h[0] = item
            _isiftup(h, size, 0)
    label = np.zeros(n_leaves, np.int64)
    stack = np.empty(2 * n_leaves, np.int64)
    for i in range(size):
        node = -h[i]
        top = 0
        stack[0] = node
        top = 1
        while top:
            top -= 1
            x = stack[top]
            if x < n_leaves:
                label[x] = i
            else:
                stack[top] = children[x - n_leaves, 0]
                stack[top + 1] = children[x - n_leaves, 1]
                top += 2
    return label


# ── The public part ──────────────────────────────────────────────────────


def _run(X: np.ndarray, G, d_cut: float, link_method: str) -> np.ndarray | None:
    from scipy.sparse.csgraph import connected_components
    from sklearn.metrics.pairwise import paired_distances

    m = len(X)
    # sklearn: the graph made symmetric; in several pieces it would add
    # edges of its own (`_fix_connectivity`) — not replayed here.
    C = (G + G.T).tocsr()
    C.sort_indices()
    n_cc, _ = connected_components(C)
    if n_cc > 1:
        return None
    rows = np.repeat(np.arange(m, dtype=np.int64), np.diff(C.indptr))
    cols = C.indices.astype(np.int64)
    off = rows != cols
    rows, cols = rows[off], cols[off]
    dist = np.empty(len(rows), np.float64)
    for a in range(0, len(rows), PAIR_BLOCK):
        dist[a : a + PAIR_BLOCK] = paired_distances(
            X[rows[a : a + PAIR_BLOCK]], X[cols[a : a + PAIR_BLOCK]], metric="euclidean"
        )
    indptr = np.zeros(m + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=m), out=indptr[1:])
    children = np.empty((m - 1, 2), np.int64)
    distances = np.empty(m - 1, np.float64)
    if not _tree(m, indptr, cols, dist, link_method == "average", children, distances):
        return None
    n_clusters = int(np.count_nonzero(distances >= d_cut)) + 1
    return _cut(n_clusters, children, m)


def fit_predict(X: np.ndarray, G, d_cut: float, link_method: str) -> np.ndarray | None:
    """The labels of ``AgglomerativeClustering(n_clusters=None,
    distance_threshold=d_cut, metric="euclidean", linkage=link_method,
    connectivity=G).fit_predict(X)`` — or ``None`` where not replayed."""
    if (
        not HAVE_NUMBA
        or link_method not in ("average", "complete")
        or len(X) < 2
        or not _agrees_with_sklearn()
    ):
        return None
    return _run(X, G, d_cut, link_method)


def _agrees_with_sklearn() -> bool:
    """Once per process: the compiled loop against the installed sklearn
    on inputs with ties and chains. A disagreement switches it off."""
    global _checked
    if _checked is not None:
        return _checked
    with _check_lock:
        if _checked is not None:
            return _checked
        ok = True
        try:
            import warnings

            from scipy.sparse import coo_matrix
            from sklearn.cluster import AgglomerativeClustering

            for seed in range(6):
                r = np.random.default_rng(seed)
                m = int(r.integers(60, 260))
                X = r.normal(size=(m // 6 + 1, 16))[r.integers(0, m // 6 + 1, m)]
                X = X + 0.3 * r.normal(size=(m, 16))
                X[: m // 10] = X[0]  # ties
                X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
                S = X @ X.T
                np.fill_diagonal(S, -2)
                nbr = np.argsort(-S, axis=1, kind="stable")[:, :8]
                # Plus a path through all voices: clustered voices fall
                # into islands otherwise, and a check over islands would
                # compare nothing (they are handed back to sklearn).
                rows = np.concatenate([np.repeat(np.arange(m), 8), np.arange(m - 1)])
                cols = np.concatenate([nbr.ravel(), np.arange(1, m)])
                G = coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(m, m)).tocsr()
                G = G.maximum(G.T)
                for method in ("average", "complete"):
                    for d_cut in (0.3, 0.6, 0.9):
                        mine = _run(X, G, d_cut, method)
                        if mine is None:
                            ok = False  # connected: it must answer
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            theirs = AgglomerativeClustering(
                                n_clusters=None,
                                distance_threshold=d_cut,
                                metric="euclidean",
                                linkage=method,
                                connectivity=G,
                            ).fit_predict(X)
                        if not np.array_equal(mine, theirs):
                            ok = False
        except Exception as e:  # noqa — a broken check means: don't use it
            logger.warning("Compiled agglomerative clustering disabled: %s", e)
            ok = False
        if not ok:
            logger.warning(
                "Compiled agglomerative clustering differs from "
                "the installed sklearn — using sklearn itself"
            )
        _checked = ok
        return ok


def warm_up() -> None:
    """Compile and check against sklearn before the first map needs it."""
    if HAVE_NUMBA:
        _agrees_with_sklearn()

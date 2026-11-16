"""Complete linkage cut at a threshold — without the distance matrix.

**Why this exists.** The guided assignment groups speaker centroids by
complete linkage cut at ``1 - cut`` (core/speaker_assignment.py). scipy /
fastcluster need the full distance matrix of what they link, and on a
large collection one chain-connected component can hold tens of thousands
of centroids: a matrix of billions of entries, its condensed copy and
fastcluster's float64 copy — **tens of gigabytes** peak and over a minute
for one grouping, with the server's own memory on top — more RAM than most
machines have.

**Why the matrix is not needed.** Only merges at a distance ``<= tau`` make
it into the cut. Complete linkage merges two groups at the LARGEST distance
between their members, so a merge below tau needs EVERY pair across the two
groups to be closer than tau — every pair an edge of the sparse graph of
pairs within tau. That graph is all the computation ever looks at:

* each group keeps the sorted list of points within tau of ALL its members,
  with the largest distance to them; merging two groups intersects their
  lists (a pair that is not complete can never become complete again);
* a group's best partner is a group whose points all lie in its list;
  complete-linkage distances only grow under merges, so a stored best
  partner is a lower bound and is recomputed only when it is popped stale
  (the lazy nearest-neighbour scheme of Müllner's generic algorithm).

Memory is linear in the number of edges (tens of millions of edges: about
a gigabyte), the edges come from blocked BLAS products in parallel, the
linking itself is compiled (numba). On the same collection: **seconds and
a small fraction of the memory instead of minutes and tens of gigabytes —
and the identical partition** (tests/test_speaker_linkage.py).

Ties (equal distances) are broken by index, deterministically; the matrix
path broke them its own way, so on data with exact ties at the deciding
merge the two may part — which on a real collection did not happen once.
Like the matrix path, the distances come from float32 BLAS products, whose
last bit depends on the block shape; a pair lying within that rounding of
the cut can therefore come and go as the collection grows — measured
across block sizes and against the matrix path: no group changed.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import numba as nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover — a lean install
    nb = None
    HAVE_NUMBA = False

#: Product block of the edge search, in bytes. A reused buffer of this size
#: instead of a fresh block per step: fresh 1-GB blocks spent more time in
#: page faults than in the multiplication (measured 3.3x).
BLOCK_BYTES = 64 << 20

#: Never allocate edges beyond this share of the memory that is free when a
#: grouping starts — a cut set far too low would otherwise connect nearly
#: every pair and ask for tens of gigabytes again.
EDGE_MEMORY_SHARE = 0.25

#: Parallel numba code must never run from two Python threads at once:
#: numba's fallback threading layer ("workqueue") is not thread-safe, and
#: two groupings side by side would only double the memory anyway.
_run_lock = threading.Lock()


class TooManyEdges(RuntimeError):
    """The cut connects so many pairs that the edges would not fit."""


if HAVE_NUMBA:
    _jit = nb.njit(cache=True)
    _pjit = nb.njit(cache=True, parallel=True)
    _prange = nb.prange
else:  # pragma: no cover

    def _jit(f):
        return f

    _pjit = _jit
    _prange = range


# ── Edges: pairs within tau, as a symmetric CSR with sorted rows ──────────


@_pjit
def _count_upper(S, tau, counts):
    """Per row of the block: how many columns to the right lie within tau.

    ``S`` holds rows ``r0..r0+B`` against columns ``r0..n``; the strictly
    upper part (column > row) is ``k > r``.
    """
    B, W = S.shape
    for r in _prange(B):
        c = 0
        for k in range(r + 1, W):
            d = np.float32(1.0) - S[r, k]
            if d < 0:
                d = np.float32(0.0)
            if np.float64(d) <= tau:
                c += 1
        counts[r] = c


@_pjit
def _fill_upper(S, r0, tau, offs, cols, ds):
    B, W = S.shape
    for r in _prange(B):
        p = offs[r]
        for k in range(r + 1, W):
            d = np.float32(1.0) - S[r, k]
            if d < 0:
                d = np.float32(0.0)
            if np.float64(d) <= tau:
                cols[p] = r0 + k
                ds[p] = d
                p += 1


@_pjit
def _symmetrize(n, fw_ptr, fw_col, fw_d, nthreads):
    """Upper lists -> full symmetric CSR, every row sorted by neighbour.

    Row p is [neighbours below p][neighbours above p]. The lower parts come
    from a parallel counting sort: chunk t of the rows writes behind chunks
    0..t-1, which keeps each lower part sorted too.
    """
    step = (n + nthreads - 1) // nthreads
    hist = np.zeros((nthreads, n), np.int64)
    for t in _prange(nthreads):
        lo = t * step
        hi = min(n, lo + step)
        for p in range(lo, hi):
            for e in range(fw_ptr[p], fw_ptr[p + 1]):
                hist[t, fw_col[e]] += 1
    bw = np.zeros(n, np.int64)
    for t in range(nthreads):
        for p in range(n):
            bw[p] += hist[t, p]
    indptr = np.empty(n + 1, np.int64)
    indptr[0] = 0
    for p in range(n):
        indptr[p + 1] = indptr[p] + bw[p] + (fw_ptr[p + 1] - fw_ptr[p])
    nbr = np.empty(indptr[n], np.int32)
    dist = np.empty(indptr[n], np.float32)
    pos = np.empty((nthreads, n), np.int64)
    for p in _prange(n):
        acc = indptr[p]
        for t in range(nthreads):
            pos[t, p] = acc
            acc += hist[t, p]
    for t in _prange(nthreads):
        lo = t * step
        hi = min(n, lo + step)
        for p in range(lo, hi):
            for e in range(fw_ptr[p], fw_ptr[p + 1]):
                q = fw_col[e]
                w = pos[t, q]
                nbr[w] = p
                dist[w] = fw_d[e]
                pos[t, q] = w + 1
    for p in _prange(n):
        w = indptr[p] + bw[p]
        for e in range(fw_ptr[p], fw_ptr[p + 1]):
            nbr[w] = fw_col[e]
            dist[w] = fw_d[e]
            w += 1
    return indptr, nbr, dist


def _free_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa — no psutil: assume 8 GB
        return 8 << 30


def edges(C: np.ndarray, tau: float, max_edges: int | None = None):
    """All pairs at a distance ``<= tau`` as a symmetric CSR.

    Distances are ``float32(1 - C_i·C_j)``, clipped at 0 — the values the
    matrix path computed — and a pair belongs in when that value, taken as
    float64, is ``<= tau``: the very test fcluster applied to the merges.
    """
    n = len(C)
    C = np.ascontiguousarray(C, dtype=np.float32)
    CT = np.ascontiguousarray(C.T)
    if max_edges is None:
        # an undirected edge ends up as two directed entries of 8 bytes,
        # plus its upper-list copy while the CSR is built
        max_edges = int(_free_bytes() * EDGE_MEMORY_SHARE) // 24
    # A constant byte budget per block, so the rows grow as the upper
    # triangle narrows: every block switches between the BLAS threads and
    # the compiled ones, and with a fixed 96 rows those 900 switches cost
    # more than the products themselves (measured: nearly twice the time).
    budget = max(min(BLOCK_BYTES, 4 * n * n), 4 * 64 * max(1, n))
    buf = np.empty(budget // 4, np.float32)
    fw_cnt = np.zeros(n, np.int64)
    cols_parts, d_parts = [], []
    total = 0
    r0 = 0
    while r0 < n:
        W = n - r0
        B = int(min(W, max(64, budget // (4 * W))))
        S = buf[: B * W].reshape(B, W)
        np.matmul(C[r0 : r0 + B], CT[:, r0:], out=S)
        counts = np.empty(B, np.int64)
        _count_upper(S, tau, counts)
        total += int(counts.sum())
        if total > max_edges:
            raise TooManyEdges(
                f"the cut connects more than {max_edges:,} pairs — too many "
                f"to hold (raise speaker.assignment_cut)"
            )
        offs = np.zeros(B, np.int64)
        np.cumsum(counts[:-1], out=offs[1:])
        cols = np.empty(int(counts.sum()), np.int32)
        ds = np.empty(len(cols), np.float32)
        _fill_upper(S, r0, tau, offs, cols, ds)
        fw_cnt[r0 : r0 + B] = counts
        cols_parts.append(cols)
        d_parts.append(ds)
        r0 += B
    del buf
    fw_ptr = np.zeros(n + 1, np.int64)
    np.cumsum(fw_cnt, out=fw_ptr[1:])
    fw_col = np.concatenate(cols_parts) if cols_parts else np.zeros(0, np.int32)
    del cols_parts
    fw_d = np.concatenate(d_parts) if d_parts else np.zeros(0, np.float32)
    del d_parts
    threads = nb.get_num_threads() if HAVE_NUMBA else 1
    return _symmetrize(n, fw_ptr, fw_col, fw_d, threads)


# ── Linking ────────────────────────────────────────────────────────────────


@_jit
def _find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:  # path compression
        nxt = parent[x]
        parent[x] = r
        x = nxt
    return r


@_jit
def _before(d1, x1, p1, d2, x2, p2):
    """Heap order: distance, then the ids — ties broken deterministically."""
    if d1 != d2:
        return d1 < d2
    if x1 != x2:
        return x1 < x2
    return p1 < p2


@_jit
def _push(hd, hx, hp, hsx, hsp, size, d, x, p, sx, sp):
    i = size
    hd[i] = d
    hx[i] = x
    hp[i] = p
    hsx[i] = sx
    hsp[i] = sp
    while i > 0:
        q = (i - 1) >> 1
        if not _before(hd[i], hx[i], hp[i], hd[q], hx[q], hp[q]):
            break
        hd[q], hd[i] = hd[i], hd[q]
        hx[q], hx[i] = hx[i], hx[q]
        hp[q], hp[i] = hp[i], hp[q]
        hsx[q], hsx[i] = hsx[i], hsx[q]
        hsp[q], hsp[i] = hsp[i], hsp[q]
        i = q
    return size + 1


@_jit
def _pop(hd, hx, hp, hsx, hsp, size):
    size -= 1
    hd[0] = hd[size]
    hx[0] = hx[size]
    hp[0] = hp[size]
    hsx[0] = hsx[size]
    hsp[0] = hsp[size]
    i = 0
    while True:
        m = 2 * i + 1
        if m >= size:
            break
        r = m + 1
        if r < size and _before(hd[r], hx[r], hp[r], hd[m], hx[m], hp[m]):
            m = r
        if not _before(hd[m], hx[m], hp[m], hd[i], hx[i], hp[i]):
            break
        hd[m], hd[i] = hd[i], hd[m]
        hx[m], hx[i] = hx[i], hx[m]
        hp[m], hp[i] = hp[i], hp[m]
        hsx[m], hsx[i] = hsx[i], hsx[m]
        hsp[m], hsp[i] = hsp[i], hsp[m]
        i = m
    return size


@_jit
def _best(x, seg0, seglen, nbr, dist, parent, size, cnt, mx, touched):
    """x's nearest COMPLETE partner: (distance, group) or (inf, -1).

    A group is complete with x when all of its points lie in x's list —
    within tau of every member of x; the distance is the largest of them.
    """
    nt = 0
    for k in range(seg0[x], seg0[x] + seglen[x]):
        r = _find(parent, nbr[k])
        if cnt[r] == 0:
            touched[nt] = r
            nt += 1
            mx[r] = dist[k]
        elif dist[k] > mx[r]:
            mx[r] = dist[k]
        cnt[r] += 1
    bd = np.float32(np.inf)
    bp = -1
    for t in range(nt):
        r = touched[t]
        if cnt[r] == size[r] and (mx[r] < bd or (mx[r] == bd and r < bp)):
            bd = mx[r]
            bp = r
        cnt[r] = 0
    return bd, bp


@_pjit
def _nearest_singletons(indptr, nbr, dist, bd, bp):
    n = len(indptr) - 1
    for x in _prange(n):
        d = np.float32(np.inf)
        p = -1
        for k in range(indptr[x], indptr[x + 1]):
            if dist[k] < d or (dist[k] == d and nbr[k] < p):
                d = dist[k]
                p = nbr[k]
        bd[x] = d
        bp[x] = p


@_jit
def _grow(a, size):
    b = np.empty(len(a) * 2, a.dtype)
    b[:size] = a[:size]
    return b


@_jit
def _link(n, indptr, nbr, dist, bd, bp):
    parent = np.arange(n)
    size = np.ones(n, np.int64)
    stamp = np.zeros(n, np.int64)
    alive = np.ones(n, np.bool_)
    seg0 = indptr[:-1].copy()
    seglen = indptr[1:] - indptr[:-1]
    cap = 2 * n + 16
    hd = np.empty(cap, np.float32)
    hx = np.empty(cap, np.int64)
    hp = np.empty(cap, np.int64)
    hsx = np.empty(cap, np.int64)
    hsp = np.empty(cap, np.int64)
    hsize = 0
    for x in range(n):
        if bp[x] >= 0:
            hsize = _push(hd, hx, hp, hsx, hsp, hsize, bd[x], x, bp[x], 0, 0)
    cnt = np.zeros(n, np.int64)
    mx = np.zeros(n, np.float32)
    touched = np.empty(n, np.int64)
    while hsize > 0:
        x = hx[0]
        p = hp[0]
        sx = hsx[0]
        sp = hsp[0]
        hsize = _pop(hd, hx, hp, hsx, hsp, hsize)
        if not alive[x] or stamp[x] != sx:
            continue  # x has changed: a newer entry of it exists
        if alive[p] and stamp[p] == sp:
            # ── merge p into x: intersect the sorted lists in place ─────────
            ia = seg0[x]
            ib = seg0[p]
            na = seglen[x]
            nb_ = seglen[p]
            i = 0
            j = 0
            w = ia
            while i < na and j < nb_:
                qa = nbr[ia + i]
                qb = nbr[ib + j]
                if qa == qb:
                    da = dist[ia + i]
                    db = dist[ib + j]
                    nbr[w] = qa
                    dist[w] = da if da > db else db
                    w += 1
                    i += 1
                    j += 1
                elif qa < qb:
                    i += 1
                else:
                    j += 1
            seglen[x] = w - ia
            seglen[p] = 0
            parent[p] = x
            alive[p] = False
            size[x] += size[p]
            stamp[x] += 1
        # x merged, or its partner has changed: its best partner anew
        d, q = _best(x, seg0, seglen, nbr, dist, parent, size, cnt, mx, touched)
        if q >= 0:
            if hsize >= len(hd):
                hd = _grow(hd, hsize)
                hx = _grow(hx, hsize)
                hp = _grow(hp, hsize)
                hsx = _grow(hsx, hsize)
                hsp = _grow(hsp, hsize)
            hsize = _push(hd, hx, hp, hsx, hsp, hsize, d, x, q, stamp[x], stamp[q])
    labels = np.zeros(n, np.int64)
    root_label = np.zeros(n, np.int64)
    nxt = 1
    for y in range(n):
        r = _find(parent, y)
        if root_label[r] == 0:
            root_label[r] = nxt
            nxt += 1
        labels[y] = root_label[r]
    return labels


def complete_groups(C: np.ndarray, cut: float, mp=None) -> np.ndarray:
    """Group numbers (1-based) of complete linkage cut at ``1 - cut``.

    ``C``: unit rows (float32). The partition is the one of
    ``fcluster(linkage(1 - C·Cᵀ, "complete"), t=1 - cut, "distance")``.
    ``mp``: optional debug.MeasurePoints for the step times.
    """
    n = len(C)
    if n == 0:
        return np.zeros(0, np.int64)
    if n == 1:
        return np.ones(1, np.int64)
    tau = 1.0 - float(cut)
    with _run_lock:
        indptr, nbr, dist = edges(C, tau)
        if mp is not None:
            mp.point(f"Edges ({len(nbr) // 2})")
        bd = np.empty(n, np.float32)
        bp = np.empty(n, np.int64)
        _nearest_singletons(indptr, nbr, dist, bd, bp)
        labels = _link(n, indptr, nbr, dist, bd, bp)
        if mp is not None:
            mp.point("Linking")
    return labels


def warm_up() -> None:
    """Compile (or load from the cache) before the first real grouping —
    so the first click does not pay a few seconds of compiling."""
    if not HAVE_NUMBA:
        return
    try:
        r = np.random.default_rng(0)
        X = r.normal(size=(64, 8)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        complete_groups(X, 0.5)
    except Exception as e:  # noqa — warming up must never break anything
        logger.debug("speaker linkage warm-up failed: %s", e)


def threads() -> int:
    """How many threads the compiled parts use."""
    if not HAVE_NUMBA:
        return 1
    return int(nb.get_num_threads()) if nb else (os.cpu_count() or 1)

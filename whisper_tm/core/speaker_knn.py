"""Neighborhood and grouping — the compute core behind the voice map.

The task: from hundreds of thousands of voice vectors, find the groups that are the
same person. Two things about this were measured, and both ruled out an
obvious solution.

**The full similarity matrix does not exist.** At 300,000 vectors that would
be 9·10^10 entries, 360 GB. So it is computed in stripes on the GPU, from
which only the best ``k`` neighbors per row survive: 300,000 vectors in
4.7 seconds.

**And agglomerative grouping does not scale to the corpus.** Measured:

===========  ==========================  =========================
Nodes        agglomerative (``average``)  Connected components
===========  ==========================  =========================
25,000       20.1 s · 3.4 GB              0.03 s
50,000       72.5 s · 6.1 GB              0.07 s
100,000      286.6 s · 12.3 GB            0.13 s
300,000      **out of memory**            ~0.4 s
===========  ==========================  =========================

Both find the same groups; one takes 2,200 times longer. Still, the faster
one is not simply the better one: connected components is **single
linkage**, and single linkage *chains* — two people merge as soon as any
two of their windows are close. A single mixed window is enough. That is
exactly why ``average`` is the chosen method.

The way out is **two-stage**:

.. code-block:: text

    1. Connected components over the whole corpus
       → fast, coarse, and it DECOMPOSES the task
    2. `average` WITHIN each component
       → the chosen linkage stays where it matters
    3. whatever became too large through chaining stands out

That way step 2 costs almost nothing (1,000 components of 100 nodes each
instead of one task with 100,000), and the quality stays that of ``average``.
"""

from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

#: Neighbors per node. 30 is taken from the scaling design and generous:
#: a person rarely has more than 30 windows that are closer to them than
#: all other voices in the corpus.
K = 30

#: Rows per GPU stripe. **Do not raise to 4,096.** At 300,000 vectors,
#: 4,096 rows are an intermediate result of 4.9 GB and a VRAM peak of
#: 5.2 GB — that tears down any concurrently running transcription. At
#: 1,024 it is 1.2 GB with practically the same runtime.
STRIPE = 1024

#: Above this component size, step 2 uses the kNN graph instead of a full
#: distance matrix. Below it, the full matrix is faster and exact:
#: 4,000² float32 is 64 MB.
FULL_UNTIL = 4000

#: How many finished runs are kept. A run is the labels per vector — at
#: ~50k voices, 200 KB. Eight of those are 1.6 MB and cover the slider
#: without anyone noticing the memory.
RESULT_CACHE_MAX = 8
_result_cache: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
_result_lock = threading.Lock()
#: One grouping at a time. Two map requests arriving together (tile and
#: graph view) computed the same fine cut twice, side by side.
_group_compute = threading.Lock()

#: Each component's fine cut, held by the ids of its members — for the
#: last few (threshold, linkage) settings. A cut depends on nothing but
#: its members' vectors, and a vector never changes under its id; so
#: when the queue adds voices, only the components they join are cut
#: again. At ~500k voices and threshold 0.94 that is ~15k components
#: of three or more, ~15 s to cut them all.
FINE_CACHE_MAX = 4
_fine_cache: collections.OrderedDict[tuple, dict] = collections.OrderedDict()

#: How many neighbourhoods are held. One is 30 × (int32 + float32) per
#: voice — 120 MB at ~500k voices — so only two: the map's and the
#: threshold search's.
KNN_CACHE_MAX = 2
_knn_cache: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
#: One kNN at a time: each occupies up to 3.5 GB of VRAM next to a running
#: transcription.
_knn_compute = threading.Lock()


def clear_cache() -> None:
    """Discard held runs and neighbourhoods — for tests and directory switches."""
    with _result_lock:
        _result_cache.clear()
        _knn_cache.clear()
        _fine_cache.clear()


def _cache_copy(entry: dict) -> dict:
    """An output the caller is allowed to modify.

    ``markers`` is **copied**. Without that, all callers would share the
    same field: a single ``markers[x] = 0`` somewhere in map building would
    silently corrupt the held run, and the map would show it that way to
    everyone from then on, until the database changes. The copy costs
    ~200 KB.
    """
    new_value = dict(entry)
    new_value["markers"] = entry["markers"].copy()
    return new_value


def _nrm(A: np.ndarray) -> np.ndarray:
    """The ONE normalization — see core/speaker.py::nrm."""
    from whisper_tm.core.speaker import nrm

    return nrm(A)


def neighbours(
    X: np.ndarray, k: int = K, stripe: int = STRIPE, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray, float]:
    """The ``k`` most similar neighbors per vector.

    Returns ``(index, value, vram_gb)``; ``X`` is assumed to already be
    **normalized**.

    **Exact** (core/speaker_neighbours.py): the GPU proposes
    candidates in half precision, exact float64 values decide, ties go to
    the lower index. The same vectors give the same neighbours to the bit,
    whatever GEMM kernel the GPU picks — which is what makes the held
    neighbourhood updatable (:func:`held_neighbours`). 17.8 s → 9.5 s at
    ~500k voices. Without numba the plain GPU product below remains.
    """
    from whisper_tm.core import speaker_neighbours

    if speaker_neighbours.HAVE_NUMBA:
        idx, v64, peak = speaker_neighbours.compute(X, k, device)
        return idx, v64.astype(np.float32), peak
    # The card only inside a stage of the GPU gate (speaker_neighbours.at_gpu_gate)
    with speaker_neighbours.at_gpu_gate(X, device) as dev:
        return _neighbours_torch(X, k, stripe, dev)


def _neighbours_torch(
    X: np.ndarray, k: int, stripe: int, device: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """The neighbours as a plain GPU product and ``topk`` — without numba.

    No CPU fallback on GPU out-of-memory. That would be the convenient
    path and the wrong one: the same computation needs the same amount in
    system RAM, the process grows by tens of GB, and the kernel's OOM
    killer takes the server down with it. Instead: shrink the stripes and
    try again.
    """
    import torch

    if not len(X):
        return (np.zeros((0, k), np.int32), np.zeros((0, k), np.float32), 0.0)
    k = min(k, len(X) - 1)
    if k < 1:
        return (np.zeros((len(X), 0), np.int32), np.zeros((len(X), 0), np.float32), 0.0)

    g = torch.device(device)
    t0 = time.time()
    while True:
        try:
            T = torch.from_numpy(np.ascontiguousarray(X)).to(g)
            idx = np.empty((len(X), k), dtype=np.int32)
            value = np.empty((len(X), k), dtype=np.float32)
            for a in range(0, len(X), stripe):
                b = min(a + stripe, len(X))
                S = T[a:b] @ T.T
                # exclude self — otherwise the best neighbor is always
                # oneself
                S[torch.arange(b - a, device=g), torch.arange(a, b, device=g)] = -2.0
                w, i = torch.topk(S, k, dim=1)
                idx[a:b] = i.cpu().numpy()
                value[a:b] = w.cpu().numpy()
                del S
            break
        except torch.cuda.OutOfMemoryError:
            del T
            torch.cuda.empty_cache()
            if stripe <= 64:
                raise
            stripe //= 4
            logger.warning("kNN: GPU memory tight, stripe shrunk to %d", stripe)
    on_gpu = str(device).startswith("cuda")
    peak = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0
    del T
    if on_gpu:
        torch.cuda.empty_cache()
    logger.info(
        "kNN k=%d over %d vectors in %.1f s (peak %.1f GB VRAM)", k, len(X), time.time() - t0, peak
    )
    return idx, value, peak


def held_neighbours(
    X: np.ndarray,
    marker,
    k: int = K,
    device: str = "cuda",
    ids: np.ndarray | None = None,
    family=None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`neighbours`, held per ``marker`` — ``(index, value)``.

    The neighbourhood is the map's largest single cost. Measured with
    ~500k voices on a consumer GPU: 17.8 s with the GPU to
    itself, 26–37 s next to a running transcription. And it was computed
    on EVERY map request — including the reload after each "Apply" in the
    guided assignment, which changes names and nothing else.

    It depends on the vectors alone. The caller passes a ``marker`` that
    changes exactly when they do — the map uses the database's
    ``inventory_generation`` — and promises that the same marker means
    the same ``X``, as with ``group_voices(marker=…)``.

    **When the vectors changed** — and with a running queue they change
    every few seconds — ``family`` names what stays the same (database,
    model, selection) and ``ids`` which voice each row is. The held
    neighbourhood of the same family is then *updated* to the new vectors
    instead of recomputed (core/speaker_neighbours.py): identical to a
    fresh computation, at the cost of the new and the affected rows.

    The arrays come back read-only: they are shared with every later
    caller, and a write into them would corrupt the map for everyone.
    """
    from whisper_tm.core import speaker_neighbours

    family = marker if family is None else family
    key = (marker, int(k), len(X))

    def held():
        with _result_lock:
            entry = _knn_cache.get(family)
            if entry is not None and entry["key"] == key:
                _knn_cache.move_to_end(family)
                return entry["idx"], entry["value"]
        return None

    hit = held()
    if hit is not None:
        return hit
    with _knn_compute:
        # The request that waited here usually finds the answer now.
        hit = held()
        if hit is not None:
            return hit
        with _result_lock:
            before = _knn_cache.get(family)
        idx = v64 = None
        if (
            before is not None
            and ids is not None
            and before["ids"] is not None
            and before["k"] == min(int(k), len(X) - 1)
            and speaker_neighbours.HAVE_NUMBA
        ):
            done = speaker_neighbours.update(
                before["ids"], before["idx"], before["v64"], X, ids, device
            )
            if done is not None:
                idx, v64 = done
        if idx is None:
            if speaker_neighbours.HAVE_NUMBA:
                idx, v64, _vram = speaker_neighbours.compute(X, k, device)
            else:
                with speaker_neighbours.at_gpu_gate(X, device) as dev:
                    idx, value, _vram = _neighbours_torch(X, k, STRIPE, dev)
        value = v64.astype(np.float32) if v64 is not None else value
        for a in (idx, value) + ((v64,) if v64 is not None else ()):
            a.flags.writeable = False
        entry = {
            "key": key,
            "k": idx.shape[1] if idx.ndim == 2 else 0,
            "ids": None if ids is None else np.array(ids, dtype=np.int64),
            "idx": idx,
            "v64": v64,
            "value": value,
        }
        with _result_lock:
            _knn_cache[family] = entry
            _knn_cache.move_to_end(family)
            while len(_knn_cache) > KNN_CACHE_MAX:
                _knn_cache.popitem(last=False)
    return idx, value


def components(idx: np.ndarray, value: np.ndarray, n: int, threshold: float) -> np.ndarray:
    """Step 1: connected components over edges at or above ``threshold``.

    This is single linkage, and it chains — which is exactly why it is
    explicitly only the **pre-decomposition** here, not the result.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if not n:
        return np.zeros(0, dtype=np.int32)
    good = value >= threshold
    # The rows are the neighbour lists already — as CSR directly, without
    # the sorting a COO conversion does (0.37 s → 0.12 s at ~500k
    # voices). The labels don't change: a component is numbered by its
    # first node, whatever order the neighbours are in.
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(good.sum(axis=1), out=indptr[1:])
    G = csr_matrix((np.ones(int(indptr[-1]), dtype=np.int8), idx[good], indptr), shape=(n, n))
    _count, markers = connected_components(G, directed=False)
    return markers.astype(np.int32)


def _finer(Xk: np.ndarray, threshold: float, link_method: str) -> np.ndarray:
    """Step 2 for ONE component: `average` on its members.

    Below :data:`FULL_UNTIL`, via the full distance matrix — exact and, for
    a few thousand points, faster than any approximation. Above it, via
    the component's kNN graph.

    The cosine → euclidean conversion is the spot that is easy to get
    wrong: for normalized vectors, ``d = sqrt(2 - 2·cos)``
    holds. Cosine 0.55 is **0.95**, not 0.45. With the wrong value
    nothing merges anymore, and that looks like a result.
    """
    m = len(Xk)
    if m < 3:
        return np.zeros(m, dtype=np.int32)
    d_cut = float(np.sqrt(max(0.0, 2.0 - 2.0 * threshold)))

    if m <= FULL_UNTIL:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        S = np.clip(Xk @ Xk.T, -1.0, 1.0)
        D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * S))
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method=link_method)
        return (fcluster(Z, t=d_cut, criterion="distance") - 1).astype(np.int32)

    from scipy.sparse import coo_matrix

    from whisper_tm.core import speaker_agglomerative

    i2, w2, _ = neighbours(Xk, k=min(K, m - 1), device=_device())
    lines = np.repeat(np.arange(m), i2.shape[1])
    G = coo_matrix(
        (np.ones(lines.size, dtype=np.float32), (lines, i2.ravel())), shape=(m, m)
    ).tocsr()
    G = G.maximum(G.T)
    # sklearn's merge loop, compiled: the same labels, 3.9 s → 0.1 s for
    # a component of ~10k voices (core/speaker_agglomerative.py). What it
    # doesn't replay — single linkage, a graph in pieces — sklearn does.
    labels = speaker_agglomerative.fit_predict(Xk, G, d_cut, link_method)
    if labels is None:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=d_cut,
            metric="euclidean",
            linkage=link_method,
            connectivity=G,
        ).fit_predict(Xk)
    return labels.astype(np.int32)


def _device() -> str:
    """The GPU if there is one. The neighbourhood is exact either way —
    the device decides the speed, not the result."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa — no torch: the CPU
        return "cpu"


def group_voices(
    X: np.ndarray,
    threshold: float,
    k: int = K,
    link_method: str = "average",
    device: str = "cuda",
    progress=None,
    neighbourhood=None,
    marker=None,
    ids: np.ndarray | None = None,
) -> dict:
    """The whole two-stage run.

    Returns a dict with ``markers`` (group number per vector) and the run's
    metrics. ``progress(text, share)`` is called along the way so a
    long run can report progress.

    ``neighbourhood`` accepts an already computed ``(idx, value)``. That is
    not a micro-tuning but the difference between "works" and "doesn't
    work" as soon as someone wants to group the SAME vectors over many
    thresholds (``/api/voicemap/threshold-search``). Measured on ~10k voices:

    ===========================  ========
    Neighborhood                  1.0 s
    Fine cut at threshold 0.4     2.0 s
    Fine cut at threshold 0.7     0.1 s
    ===========================  ========

    The neighborhood does NOT depend on the threshold — computing it 51
    times would be 51 seconds for a result that was already available
    after the first.

    **``marker`` turns on the result cache.** By passing it, the caller
    agrees that the same label means the same ``X``; the map passes the
    speaker database's ``inventory_generation``, which moves when vectors
    come or go (new recording, deleted recording, trash) — not on names:
    the map used ``generation`` before, and every "Apply" in the
    guided assignment threw the run away (~15 s at ~500k voices). Without
    ``marker``, it always recomputes. One run with a marker is computed at
    a time; a request for the same run waits and gets the held one.

    Why this became necessary — measured with ~50k voices,
    threshold 0.65:

    ==============================  ==========
    Load corpus                       0.32 s
    Neighborhood (kNN, GPU)           0.27 s
    Coarse decomposition              0.01 s
    **Fine cut**                    **18.98 s**
    ==============================  ==========

    And of that 18.98 s, **18.38 s falls on a single component** with
    ~30k voices (98.6%). The remaining ~900 components together cost
    0.26 s. The cost sits in sklearn's ``linkage_tree``, and specifically
    in *Python*: 7.2 million ``heappop`` calls in the loop (12.9 s) plus
    8.4 s in the heap itself. Fewer edges in the connectivity graph do NOT
    help — with k=10 it is the same 18.4 s, with k=5 even 23.1 s, because
    then more components have to be patched back together.

    So the fine cut cannot be made faster, only rarer. That is exactly
    what the label does.
    """
    if marker is None or not len(X):
        # Nothing to hold for an empty inventory: it computes nothing.
        return _group_voices(X, threshold, k, link_method, device, progress, neighbourhood, ids)
    key = (marker, round(float(threshold), 6), int(k), link_method, len(X))
    done = _held_run(key, marker, threshold)
    if done is not None:
        return done
    with _group_compute:
        # Whoever waited here for the same run finds it now.
        done = _held_run(key, marker, threshold)
        if done is not None:
            return done
        result = _group_voices(X, threshold, k, link_method, device, progress, neighbourhood, ids)
        with _result_lock:
            _result_cache[key] = _cache_copy(result)
            _result_cache.move_to_end(key)
            while len(_result_cache) > RESULT_CACHE_MAX:
                _result_cache.popitem(last=False)
    return result


def _held_run(key: tuple, marker, threshold: float) -> dict | None:
    with _result_lock:
        done = _result_cache.get(key)
        if done is None:
            return None
        _result_cache.move_to_end(key)
    logger.info("Grouping: held run (label %s, threshold %.3f)", marker, threshold)
    return _cache_copy(done)


def _group_voices(
    X: np.ndarray,
    threshold: float,
    k: int,
    link_method: str,
    device: str,
    progress,
    neighbourhood,
    ids: np.ndarray | None = None,
) -> dict:
    """The computation behind :func:`group_voices`, without its cache.

    ``ids`` (which voice each row is) lets each component's cut be held
    across changes of the inventory — see :data:`_fine_cache`."""

    def report_it(text, share):
        if progress:
            progress(text, share)
        logger.info("Grouping: %s", text)

    t0 = time.time()
    n = len(X)
    if not n:
        return {
            "markers": np.zeros(0, np.int32),
            "nodes": 0,
            "groups": 0,
            "components": 0,
            "duration_sec": 0.0,
            "largest": 0,
        }
    if neighbourhood is not None:
        idx, value = neighbourhood
    else:
        report_it("Computing neighborhood", 0.05)
        idx, value, _vram = neighbours(_nrm(X.astype(np.float32)), k=k, device=device)

    report_it("Coarse decomposition (connected components)", 0.35)
    comp = components(idx, value, n, threshold)
    component_count = int(comp.max()) + 1 if n else 0

    report_it(f"Fine cut in {component_count} components", 0.45)
    # START AT ONE. Zero is reserved: it means "belongs to no group",
    # and the map draws it gray.
    #
    # Before, the count started at zero — and `order` works through the
    # components by SIZE. The first subdivision of the LARGEST component
    # then got label 0 and was from then on treated as unbound.
    # Observed once: at threshold 0.62 the largest person stood as a
    # group of a few thousand voices; at 0.61 it disappeared without a
    # trace — and the count of singletons jumped by exactly that group.
    #
    # Whether the largest person in the corpus even exists thus hinged on
    # an internal number assignment. A nudge of the threshold slider
    # decided it anew, in either direction, without warning.
    next_one = 1
    large = []
    # Descending by size, so the expensive ones run first and an abort
    # does not leave exactly the most important ones open.
    order = np.argsort(np.bincount(comp, minlength=component_count))[::-1]

    # Members per component in ONE pass.
    #
    # Previously this used `np.flatnonzero(comp == kid)` INSIDE the loop
    # — a full comparison over all nodes per component, plus a bool field
    # of that length. At about a thousand voices and 16 components that is
    # unnoticeable; at ~50k voices and thousands of components it is
    # the single biggest cost in building the map.
    #
    # `argsort` over the count gives the order; `np.argsort(comp)` gives
    # the members of all components in one block — from which a slice can
    # be taken per component without searching again.
    _sorted = np.argsort(comp, kind="stable")
    _limits = np.cumsum(np.bincount(comp, minlength=component_count))
    _start = np.concatenate(([0], _limits[:-1]))
    # The fine cut total, and the most expensive single component next
    # to it.
    #
    # The second number is the important one: once it was 18.38 s of
    # 18.98 s for ONE component with ~30k voices. Logging only the sum
    # would have looked like general slowness and pointed in the wrong
    # direction — but it is chaining at a single spot.
    _t_fine = 0.0
    _most_expensive = (0.0, 0)

    # Phase 1: each component's own cut — held by its members' ids where
    # possible, computed otherwise.
    fine_key = (float(threshold), link_method)
    with _result_lock:
        held = _fine_cache.get(fine_key, {})
    kept: dict = {}
    cut_of: dict[int, np.ndarray] = {}
    reused = 0
    for counter, kid in enumerate(order):
        members = _sorted[_start[kid] : _limits[kid]]
        m = len(members)
        if m < 3 or m > 50_000:
            continue
        if ids is not None:
            mids = ids[members]
            key = (m, hash(mids.tobytes()))
            h = held.get(key)
            if h is not None and np.array_equal(h[0], mids):
                cut_of[int(kid)] = h[1]
                kept[key] = h
                reused += 1
                continue
        _t = time.perf_counter()
        # Normalized here, per component: the same rows as normalizing the
        # whole matrix (the normalization is per row) — without two
        # ~400-MB copies for the components that are held anyway.
        below = _finer(_nrm(X[members].astype(np.float32)), threshold, link_method)
        _duration = time.perf_counter() - _t
        _t_fine += _duration
        _most_expensive = max(_most_expensive, (_duration, m))
        cut_of[int(kid)] = below
        if ids is not None:
            below.flags.writeable = False
            kept[key] = (mids, below)
        if counter % 200 == 0:
            report_it(
                f"Fine cut {counter}/{component_count}",
                0.45 + 0.5 * counter / max(component_count, 1),
            )
    if ids is not None:
        # Only this run's components stay: the held cuts never outgrow
        # the inventory.
        with _result_lock:
            _fine_cache[fine_key] = kept
            _fine_cache.move_to_end(fine_key)
            while len(_fine_cache) > FINE_CACHE_MAX:
                _fine_cache.popitem(last=False)

    # Phase 2: numbering, in the order of size — as the loop that was here
    # did it one component at a time (over 100,000 components at threshold
    # 0.94: 0.4 s of Python), now in one pass: each component takes a
    # block of numbers, as many as it has groups; a voice gets its block's
    # start plus its place in the component's own cut.
    sizes_c = _limits - _start
    blocks = sizes_c.copy()  # under three: each voice alone
    local = np.empty(n, dtype=np.int64)
    local[_sorted] = np.arange(n) - _start[comp[_sorted]]
    for kid, below in cut_of.items():
        blocks[kid] = int(below.max()) + 1
        local[_sorted[_start[kid] : _limits[kid]]] = below
    for kid in order:
        if sizes_c[kid] > 50_000:
            # Something like this only arises from chaining. Not
            # decomposing it here would be convenient and wrong — but the
            # fine cut over 50,000 nodes is exactly the computation that
            # blows the memory. So report it and leave it whole, so it
            # can be SEEN.
            large.append((int(kid), int(sizes_c[kid])))
            blocks[kid] = 1
            local[_sorted[_start[kid] : _limits[kid]]] = 0
    first = np.empty(component_count, dtype=np.int64)
    in_order = blocks[order]
    first[order] = next_one + np.cumsum(in_order) - in_order
    markers = (first[comp] + local).astype(np.int32)
    next_one += int(blocks.sum())
    if ids is not None:
        logger.info("Grouping: %d component cuts held, %d computed", reused, len(cut_of) - reused)

    logger.info(
        "Grouping: %d components · fine cut %.2fs, of which "
        "%.2fs in the most expensive one (%d voices)",
        component_count,
        _t_fine,
        _most_expensive[0],
        _most_expensive[1],
    )

    sizes = np.bincount(markers, minlength=next_one)
    duration = time.time() - t0
    report_it("done", 1.0)
    if large:
        logger.warning(
            "Grouping: %d component(s) over 50,000 nodes stayed "
            "undivided — that is chaining, look there: %s",
            len(large),
            large[:5],
        )
    result = {
        "markers": markers,
        "nodes": n,
        # `next_one` is the NEXT free label, and counting starts at 1
        # (zero is reserved for "belongs to no group"). So there are
        # `next_one - 1` groups. Without that `- 1`, every run reported
        # exactly one group too many.
        "groups": max(0, int(next_one) - 1),
        "components": component_count,
        "largest": int(sizes.max()) if next_one else 0,
        "loners": int((sizes == 1).sum()),
        "unsplit": large,
        "duration_sec": round(duration, 1),
    }
    return result


def arrange(
    markers: np.ndarray, sizes: np.ndarray | None = None, distance: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Distribute groups over the plane — spiral instead of force simulation.

    The overview level does not need physics: what matters is that large
    groups sit in the middle, nothing overlaps, and the layout stays
    **stable** between two runs. A force simulation ends up somewhere
    different on every run, and then the user has to find their group
    again.

    Returns ``(x, y)`` per group; the radius grows with the square root of
    the running number, so the area fills evenly.
    """
    if sizes is None:
        sizes = np.bincount(markers)
    order = np.argsort(sizes)[::-1]
    x = np.zeros(len(sizes), dtype=np.float32)
    y = np.zeros(len(sizes), dtype=np.float32)
    gold = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    for slot, g in enumerate(order):
        r = distance * np.sqrt(slot + 0.5) * (1.0 + 0.15 * np.log1p(sizes[g]))
        x[g] = r * np.cos(gold * slot)
        y[g] = r * np.sin(gold * slot)
    return x, y

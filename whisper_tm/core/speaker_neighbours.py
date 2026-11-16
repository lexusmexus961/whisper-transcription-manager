"""The exact neighbourhood — and how it follows new voices without starting over.

The voice map needs, per voice, the ``k`` most similar other voices.
Previously that was one cuBLAS product per stripe of rows and ``torch.topk``
over it: tens of seconds for a large inventory, longer still next to
a running transcription — and all of it again for every new recording.
The queue adds recordings every few seconds, so the map was computed from
scratch on nearly every request.

**Two stages.** The GPU only proposes *candidates*: a half-precision
product per stripe and the ``CANDIDATES`` largest per row. The CPU then
decides with *exact* values — each candidate's dot product in float64,
in one fixed order — and ranks by value, ties by the lower row. The
result no longer depends on which GEMM kernel the GPU happened to pick:
two computations over the same vectors are identical to the bit.

**Why the candidates are enough — the bound.** A half-precision
similarity of two unit vectors is off by at most :data:`EPS_GPU` (inputs
rounded to 11 bits: ≤ 9.8e-4; the result rounded: ≤ 4.9e-4; float32
accumulation: ≤ 1.2e-5 — in practice clearly less).
If the smallest candidate's value plus that bound still lies below the
exact k-th value, no voice outside the candidates can reach the top k.
Where that is not certain (a small share of the rows, near-duplicates), the row
takes every voice within the bound of its k-th value and ranks those
exactly. The bound is checked against the exact values on every run; a
GPU that is less precise than assumed widens it instead of losing
neighbours.

On a consumer GPU the candidates take most of the time; the exact values
and the uncertain rows add only a fraction of it.

**Following new voices.** Because the result is exact, it can be updated
instead of recomputed, and the update is identical to a recomputation:

* a row whose neighbours all still exist keeps them, and only the new
  voices can enter — those within the bound of its k-th value, ranked
  exactly;
* a row that lost a neighbour (a deleted recording, the trash) and every
  new row are computed like at the start.

Twenty new voices then cost one product of n × 20 instead of n².
"""

from __future__ import annotations

import contextlib
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import numba as nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover — a lean install
    nb = None
    HAVE_NUMBA = False

#: How far a GPU similarity may be off — see the module docstring.
EPS_GPU = 2e-3
#: The same for the CPU's float32 product: at most 192 · 2⁻²⁴ ≈ 1.2e-5 for
#: unit vectors of 192 dimensions; generously above that.
EPS_CPU = 1e-4
#: Candidates per row before the exact values decide.
CANDIDATES = 64
#: Bytes of one product stripe on the GPU. 1.25 GB: the old limit that
#: kept a concurrently running transcription alive (see speaker_knn).
STRIPE_BYTES = 1_250_000_000
#: Bytes of one product block on the CPU.
BLOCK_BYTES = 256 << 20
#: Above this share of rows to recompute, an update is no cheaper than
#: starting over.
UPDATE_AT_MOST = 0.5
#: Up to this many new voices and rows to recompute, an update runs on the
#: CPU: a few hundred columns against a large inventory are milliseconds of BLAS,
#: while the GPU would first need the whole matrix (a few hundred MB) uploaded — and
#: it is busy with the transcription. The result is the same either way.
UPDATE_ON_CPU = 256
#: How long the full neighbourhood waits for the GPU gate before it
#: computes on the CPU. With three queue lanes Whisper holds the gate
#: (exclusively) almost all the time: five seconds sent the full kNN of
#: a large inventory to the CPU — minutes instead of seconds.
#: Its turn comes between two Whisper chunks; waiting for it is cheaper.
GPU_WAIT_SEC = 300.0
#: The same for an update too big for the CPU route: it is the case of a
#: running queue — a short wait, then the CPU, which is quick enough for
#: the share of rows an update recomputes.
GPU_WAIT_UPDATE_SEC = 5.0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa — no torch, no card
        return False


@contextlib.contextmanager
def at_gpu_gate(X: np.ndarray, device: str, wait_sec: float | None = None, cancel=None):
    """The device to compute on — the GPU only inside a stage of the gate.

    The map's neighbourhood loaded the whole inventory onto the card (half
    precision, plus product stripes of :data:`STRIPE_BYTES`) outside
    ``core/gpu_gate.GATE``: next to the queue's lanes and other processes
    on the card a map refresh could push a transcription into out-of-memory.
    Now it takes the stage "voicemap" with what it needs.

    ``wait_sec`` (default :data:`GPU_WAIT_SEC`): while the gate keeps it out
    longer than that, it yields "cpu" — the neighbours are exact either
    way. ``cancel()`` ends the wait early (also to the CPU). Without CUDA
    it yields "cpu" at once and never asks the gate.
    """
    if not str(device).startswith("cuda"):
        yield device
        return
    if not _cuda_available():
        yield "cpu"
        return
    from whisper_tm.core import gpu_gate

    need_mb = int(X.size * 2 / 2**20 + STRIPE_BYTES / 2**20 + 256)
    end = time.monotonic() + (GPU_WAIT_SEC if wait_sec is None else wait_sec)

    def give_up() -> bool:
        return time.monotonic() > end or bool(cancel is not None and cancel())

    stack = contextlib.ExitStack()
    try:
        stack.enter_context(gpu_gate.GATE.stage("voicemap", need_mb, cancel=give_up))
    except InterruptedError:
        logger.info("kNN: the GPU stayed busy (gate held) — computing on the CPU")
        device = "cpu"
    with stack:
        yield device


if HAVE_NUMBA:
    _pjit = nb.njit(cache=True, parallel=True)
    _prange = nb.prange
else:  # pragma: no cover

    def _pjit(f):
        return f

    _prange = range


# ── Exact values ──────────────────────────────────────────────────────────
#
# Each dot product in float64, term by term in index order. The products of
# two float32 numbers are exact in float64, and without fastmath the sum is
# never reordered — so the value is the same on every machine and in every
# call, whatever else is computed alongside.


@_pjit
def _rank_dense(X, rows, cand, approx, k, out_idx, out_v, err):
    """Top ``k`` of each row among its candidates: exact values, ties by
    the lower index. ``err[q]``: the largest |approx - exact| seen."""
    d = X.shape[1]
    for q in _prange(len(rows)):
        r = rows[q]
        m = cand.shape[1]
        order = np.argsort(cand[q], kind="mergesort")
        c = np.empty(m, np.int64)
        vals = np.empty(m, np.float64)
        worst = 0.0
        for a in range(m):
            j = cand[q, order[a]]
            c[a] = j
            s = 0.0
            for t in range(d):
                s += np.float64(X[r, t]) * np.float64(X[j, t])
            if j == r:
                s = -np.inf
            else:
                off = abs(np.float64(approx[q, order[a]]) - s)
                if off > worst:
                    worst = off
            vals[a] = s
        err[q] = worst
        o = np.argsort(-vals, kind="mergesort")
        for a in range(k):
            out_idx[q, a] = c[o[a]]
            out_v[q, a] = vals[o[a]]


@_pjit
def _rank_csr(X, rows, ptr, cols, k, out_idx, out_v):
    """As `_rank_dense`, with a candidate list of its own length per row."""
    d = X.shape[1]
    for q in _prange(len(rows)):
        r = rows[q]
        c = np.sort(cols[ptr[q] : ptr[q + 1]])
        m = len(c)
        vals = np.empty(m, np.float64)
        for a in range(m):
            j = c[a]
            s = 0.0
            for t in range(d):
                s += np.float64(X[r, t]) * np.float64(X[j, t])
            vals[a] = -np.inf if j == r else s
        o = np.argsort(-vals, kind="mergesort")
        for a in range(k):
            if a < m:
                out_idx[q, a] = c[o[a]]
                out_v[q, a] = vals[o[a]]
            else:
                # Cannot happen while the bound holds — marked, not guessed.
                out_idx[q, a] = -1
                out_v[q, a] = -np.inf


@_pjit
def _merge(X, rows, ptr, cols, idx, v64):
    """Let the new voices ``cols[ptr[q]:ptr[q+1]]`` into row ``rows[q]``'s
    top k (``idx``/``v64``, updated in place) where they rank higher."""
    d = X.shape[1]
    k = idx.shape[1]
    for q in _prange(len(rows)):
        a0, a1 = ptr[q], ptr[q + 1]
        if a1 == a0:
            continue
        r = rows[q]
        m = k + a1 - a0
        c = np.empty(m, np.int64)
        vals = np.empty(m, np.float64)
        for a in range(k):
            c[a] = idx[r, a]
            vals[a] = v64[r, a]
        for a in range(a1 - a0):
            j = cols[a0 + a]
            s = 0.0
            for t in range(d):
                s += np.float64(X[r, t]) * np.float64(X[j, t])
            c[k + a] = j
            vals[k + a] = s
        o1 = np.argsort(c, kind="mergesort")
        c = c[o1]
        vals = vals[o1]
        o = np.argsort(-vals, kind="mergesort")
        for a in range(k):
            idx[r, a] = c[o[a]]
            v64[r, a] = vals[o[a]]


def _locked(kernel, *args):
    """Run a parallel kernel under the lock all compiled speaker code shares
    (core/speaker_linkage.py: numba's fallback threading layer is not
    thread-safe)."""
    from whisper_tm.core.speaker_linkage import _run_lock

    with _run_lock:
        kernel(*args)


# ── Candidates ────────────────────────────────────────────────────────────


class _Products:
    """Approximate similarities of row blocks against columns — on the GPU
    in half precision, or on the CPU in float32. ``eps`` bounds the error."""

    def __init__(self, X: np.ndarray, device: str):
        self.X = X
        self.gpu = str(device).startswith("cuda")
        self.eps = EPS_GPU if self.gpu else EPS_CPU
        self.peak = 0.0
        if self.gpu:
            import torch

            self.torch = torch
            self.g = torch.device(device)
            self.T = torch.from_numpy(np.ascontiguousarray(X)).to(self.g).half()
            self.stripe_bytes = STRIPE_BYTES
        else:
            self.T = None

    def close(self) -> None:
        if self.gpu:
            torch = self.torch
            self.peak = torch.cuda.max_memory_allocated(self.g) / 1e9
            del self.T
            self.T = None
            torch.cuda.empty_cache()

    def _block_rows(self, n_cols: int, item: int) -> int:
        budget = self.stripe_bytes if self.gpu else BLOCK_BYTES
        return int(max(16, min(8192, budget // max(1, item * n_cols))))

    def top(self, rows: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        """The ``m`` largest approximate similarities per row (self excluded):
        ``(columns, values)``."""
        n = len(self.X)
        cand = np.empty((len(rows), m), np.int64)
        approx = np.empty((len(rows), m), np.float32)
        if not len(rows):
            return cand, approx
        if self.gpu:
            torch = self.torch
            step = self._block_rows(n, 2)
            a = 0
            while a < len(rows):
                b = min(a + step, len(rows))
                try:
                    rr = torch.from_numpy(rows[a:b]).to(self.g)
                    S = self.T[rr] @ self.T.T
                    S[torch.arange(b - a, device=self.g), rr] = -2.0
                    w, i = torch.topk(S, m, dim=1)
                    cand[a:b] = i.cpu().numpy()
                    approx[a:b] = w.float().cpu().numpy()
                    del S, w, i
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if step <= 16:
                        raise
                    step //= 4
                    self.stripe_bytes //= 4
                    logger.warning("kNN: GPU memory tight, stripe shrunk to %d rows", step)
                    continue
                a = b
            return cand, approx
        XT = self.X.T
        step = self._block_rows(n, 4)
        for a in range(0, len(rows), step):
            rr = rows[a : a + step]
            S = self.X[rr] @ XT
            S[np.arange(len(rr)), rr] = -2.0
            part = np.argpartition(S, n - m, axis=1)[:, n - m :]
            cand[a : a + step] = part
            approx[a : a + step] = np.take_along_axis(S, part, axis=1)
        return cand, approx

    def above(
        self, rows: np.ndarray, limit: np.ndarray, cols: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per row every column whose approximate similarity is at least
        ``limit[row]`` (self excluded) — as ``(ptr, columns)``. ``cols``
        restricts the columns (positions into X)."""
        n_cols = len(self.X) if cols is None else len(cols)
        counts = np.zeros(len(rows), np.int64)
        found = []
        if not len(rows) or not n_cols:
            return np.zeros(len(rows) + 1, np.int64), np.zeros(0, np.int64)
        # The limit rounded DOWN to float32, a little more: rounding it up
        # could drop a column that lies exactly on it.
        lim32 = (np.asarray(limit, np.float64) - 1e-6).astype(np.float32)
        if self.gpu:
            torch = self.torch
            TC = self.T if cols is None else self.T[torch.from_numpy(cols).to(self.g)]
            lim = torch.from_numpy(lim32).to(self.g)
            step = self._block_rows(n_cols, 2 + 4 + 1)
            a = 0
            while a < len(rows):
                b = min(a + step, len(rows))
                try:
                    rr = torch.from_numpy(rows[a:b]).to(self.g)
                    S = (self.T[rr] @ TC.T).float()
                    if cols is None:
                        S[torch.arange(b - a, device=self.g), rr] = -2.0
                    mask = S >= lim[a:b, None]
                    del S
                    counts[a:b] = mask.sum(dim=1).cpu().numpy()
                    found.append(torch.nonzero(mask)[:, 1].cpu().numpy())
                    del mask
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if step <= 16:
                        raise
                    step //= 4
                    logger.warning("kNN: GPU memory tight, stripe shrunk to %d rows", step)
                    continue
                a = b
            del TC
        else:
            XC = self.X if cols is None else self.X[cols]
            step = self._block_rows(n_cols, 4)
            for a in range(0, len(rows), step):
                rr = rows[a : a + step]
                S = self.X[rr] @ XC.T
                if cols is None:
                    S[np.arange(len(rr)), rr] = -2.0
                mask = S >= lim32[a : a + step, None]
                counts[a : a + step] = mask.sum(axis=1)
                found.append(np.nonzero(mask)[1])
        hit = np.concatenate(found).astype(np.int64)
        if cols is not None:
            hit = cols[hit]
        ptr = np.zeros(len(rows) + 1, np.int64)
        np.cumsum(counts, out=ptr[1:])
        return ptr, hit


# ── The neighbourhood of some rows, from scratch ─────────────────────────


def _rows(P: _Products, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    X = P.X
    n = len(X)
    idx = np.empty((len(rows), k), np.int32)
    v64 = np.empty((len(rows), k), np.float64)
    if not len(rows):
        return idx, v64
    m = min(max(CANDIDATES, 2 * k), n - 1)
    if m >= n - 1:
        # Every other row is a candidate: nothing can be missed.
        cand = np.broadcast_to(np.arange(n, dtype=np.int64), (len(rows), n)).copy()
        approx = np.zeros(cand.shape, np.float32)
        err = np.zeros(len(rows), np.float64)
        # One column too many (the row itself): ranked last, as -inf.
        _locked(_rank_dense, X, rows, cand, approx, k, idx, v64, err)
        return idx, v64
    cand, approx = P.top(rows, m)
    err = np.empty(len(rows), np.float64)
    _locked(_rank_dense, X, rows, cand, approx, k, idx, v64, err)
    eps = P.eps
    seen = float(err.max())
    if seen > eps / 2:
        # Less precise than assumed: widen the bound, never trust it blindly.
        logger.warning(
            "kNN: similarities off by up to %.1e (bound %.1e) — widening the bound", seen, eps
        )
        eps = 4 * seen
    low = approx.min(axis=1).astype(np.float64)
    unsure = np.flatnonzero(low + eps >= v64[:, k - 1])
    if len(unsure):
        ptr, cols = P.above(rows[unsure], v64[unsure, k - 1] - eps)
        ui = np.empty((len(unsure), k), np.int32)
        uv = np.empty((len(unsure), k), np.float64)
        _locked(_rank_csr, X, rows[unsure], ptr, cols, k, ui, uv)
        if (ui < 0).any():
            raise RuntimeError("kNN: the similarity bound did not hold")
        idx[unsure] = ui
        v64[unsure] = uv
    return idx, v64


def compute(X: np.ndarray, k: int, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, float]:
    """Exact top ``k`` per row: ``(idx int32, v64 float64, peak_vram_gb)``.

    ``X``: unit rows, float32. Ranked by exact value, ties by the lower
    index; the row itself never counts.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = len(X)
    k = min(k, n - 1)
    if n == 0 or k < 1:
        return (np.zeros((n, max(k, 0)), np.int32), np.zeros((n, max(k, 0)), np.float64), 0.0)
    t0 = time.time()
    with at_gpu_gate(X, device) as dev:
        P = _Products(X, dev)
        try:
            idx, v64 = _rows(P, np.arange(n, dtype=np.int64), k)
        finally:
            P.close()
    logger.info(
        "kNN k=%d over %d vectors in %.1f s (exact; peak %.1f GB VRAM)",
        k,
        n,
        time.time() - t0,
        P.peak,
    )
    return idx, v64, P.peak


def update(
    prev_ids: np.ndarray,
    prev_idx: np.ndarray,
    prev_v64: np.ndarray,
    X: np.ndarray,
    ids: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray] | None:
    """The neighbourhood of ``X`` (rows ``ids``) from the one of
    ``prev_ids`` — identical to :func:`compute`, or ``None`` where an
    update would not be cheaper (then compute).

    Both id lists must be ascending: that is the order the inventory is
    read in, and it keeps the ranking by index stable across the update.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = len(X)
    k = prev_idx.shape[1] if prev_idx.ndim == 2 else 0
    if (
        n < 2
        or not len(prev_ids)
        or k < 1
        or k != min(k, n - 1)
        or np.any(ids[1:] <= ids[:-1])
        or np.any(prev_ids[1:] <= prev_ids[:-1])
    ):
        return None
    t0 = time.time()
    pos = np.searchsorted(ids, prev_ids)
    pos_c = np.minimum(pos, n - 1)
    kept = ids[pos_c] == prev_ids
    new_of_old = np.where(kept, pos_c, -1)
    moved = new_of_old[prev_idx]  # -1: neighbour gone
    lost = kept & (moved < 0).any(axis=1)
    clean = kept & ~lost
    old_here = np.zeros(n, bool)
    old_here[new_of_old[kept]] = True
    added = np.flatnonzero(~old_here)
    again = np.sort(np.concatenate([new_of_old[lost], added])).astype(np.int64)
    if len(again) > UPDATE_AT_MOST * n:
        return None

    idx = np.empty((n, k), np.int32)
    v64 = np.empty((n, k), np.float64)
    clean_new = new_of_old[clean].astype(np.int64)
    idx[clean_new] = moved[clean]
    v64[clean_new] = prev_v64[clean]
    small = len(added) <= UPDATE_ON_CPU and len(again) <= UPDATE_ON_CPU
    with at_gpu_gate(X, "cpu" if small else device, wait_sec=GPU_WAIT_UPDATE_SEC) as dev:
        P = _Products(X, dev)
        try:
            if len(added) and len(clean_new):
                # A new voice enters a row only at or above its k-th value.
                ptr, cols = P.above(
                    clean_new, v64[clean_new, k - 1] - P.eps, cols=added.astype(np.int64)
                )
                _locked(_merge, X, clean_new, ptr, cols, idx, v64)
            if len(again):
                ri, rv = _rows(P, again, k)
                idx[again] = ri
                v64[again] = rv
        finally:
            P.close()
    logger.info(
        "kNN updated: %d new, %d gone, %d rows recomputed — %.2f s",
        len(added),
        int((~kept).sum()),
        len(again),
        time.time() - t0,
    )
    return idx, v64


def warm_up() -> None:
    """Load the compiled parts from numba's cache (or compile them) before
    the first map needs them."""
    if not HAVE_NUMBA:
        return
    try:
        r = np.random.default_rng(0)
        X = r.normal(size=(120, 8))
        X[:40] = X[0]  # ties past the candidates: the fallback
        X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
        ids = np.arange(120, dtype=np.int64)
        idx, v64, _ = compute(X[:100], 30, "cpu")
        update(ids[:100], idx, v64, X, ids, "cpu")
    except Exception as e:  # noqa — warming up must never break anything
        logger.debug("kNN warm-up failed: %s", e)

"""Assign with a few clicks — speaker centroids, complete linkage,
propagation.

**The problem this module solves.** The voice map groups windows via a
threshold: what's above 0.71 is connected, and what's connected forms a
group. On a large real collection this produced a single giant group,
starting from ONE voice: after a few expansion steps it had grown from
about a hundred voices in a dozen recordings to tens of thousands of
voices from hundreds of recordings — and was still growing.

Every single edge was above the threshold, but the beginning and end of
the chain had nothing to do with each other. A higher threshold doesn't
help: even at 0.83, a large part of the collection still hung together,
and the separation threshold for a single distinctive voice was 0.94 — a
value at which even one's own voice falls apart into dozens of little
groups.

**What's done instead**, in three steps. The methods aren't invented, but
taken from the literature on linking across recordings (radio archives,
VoxSRC systems) — and measured against a real collection.

1. **Centroids instead of windows.** Not individual 30-second windows are
   grouped, but one mean value per recording and ID. pyannote has already
   grouped within the recording; that work isn't thrown away, but used as
   a building block (two-stage procedure).

       Windows per speaker centroid: a handful
       Cohesion within a speaker: median above 0.95

2. **Complete linkage instead of connected components.** The component
   connects whatever touches at ONE point — that's the chain. Complete
   linkage requires that ALL pairs of two groups match, and is therefore
   chain-proof by construction. Measured:

       ============================  =====================
       Method                        largest cluster
       ============================  =====================
       Windows + threshold           about half of all windows
       Centroids + threshold         about half of all centroids
       **Centroids + complete**      **under 2% of all centroids**
       ============================  =====================

   And it's not just smaller, it's also correct: measured against several
   hundred manual assignments (which weren't part of the computation),
   **purity is above 99%** — two people practically never end up together.

3. **Propagate.** Pure groups are conservative: one person falls apart
   into many. That's why an assigned name is propagated to all centroids
   close enough to the **name centroid** (mean of all voices of this
   name, grows with every click). Measured against the same manual
   assignments:

       known    threshold   assigned    of which correct
          10%       0.65        ~80%              ~98%
          25%       0.65        ~95%             ~100%

The result of this interplay is the whole reason for the module:

    50% of recurring speaking time after  ~15 clicks
    80%                            after ~100 clicks
    95%                            after ~300 clicks

**What is NOT shown:** groups that only occur in a single recording. On a
real collection about a fifth of all speaking time is in voices that
appear exactly once — they never come back, and naming them would be
wasted time. So counting is only done against the recurring speaking
time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

#: Cutoff for complete linkage. 0.71 measured: largest cluster under 2% of
#: all centroids, purity above 99%. Higher fragments, lower doesn't gain much more grouping.
CUT = 0.71

#: Above this, a name gets propagated. 0.65 is the measured compromise —
#: near-perfect accuracy at 25% known nodes. 0.60 covers slightly more,
#: but is less accurate.
SPREAD = 0.65

#: A recount in the background starts at most this often.
#:
#: A running queue changes the inventory every few seconds, and every
#: suggestion after such a change started a recount — several seconds on all cores,
#: back to back for as long as the user kept clicking (measured next to
#: a running queue). The held grouping is delivered either way; new
#: voices simply join a minute later.
RECOUNT_EVERY_SEC = 60

#: Both are ADJUSTABLE, and that's not a formality.
#:
#: The slider in the voice map belongs to the MAP — it decides how
#: grouping happens there. The assignment path has its own two values,
#: and they used to be hidden constants in the code. A setting you can't
#: see and can't change is exactly what this project avoids.
#:
#: The defaults are measurement results (see module header), not
#: natural constants — the material gets a say too.
_CONTROL_VALUES = {
    "cut": ("speaker.assignment_cut", CUT),
    "spread": ("speaker.assignment_spread", SPREAD),
}


def control_value(kind: str) -> float:
    """The configured value — or the measured one.

    An unusable value in the configuration is silently ignored: it must
    not break an assignment run, and the default is well-established.
    """
    path, standard = _CONTROL_VALUES.get(kind, (None, 0.71))
    if not path:
        return standard
    try:
        from whisper_tm.config import config

        value = float(config.get(*path.split("."), default=standard))
    except Exception:  # noqa — when in doubt, use the measured value
        return standard
    return value if 0.0 < value <= 1.0 else standard


#: Less speaking time than this, and a centroid isn't reliable enough.
MIN_SPEECH_SEC = 10.0


def _nrm(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=-1, keepdims=True)
    return A / np.maximum(n, 1e-12)


class Inventory:
    """The centroids and their grouping, held in the process.

    Computing is linear in memory in the number of close pairs
    (core/speaker_linkage.py). Measured on a large collection (tens of
    thousands of centroids): seconds and about a gigabyte or two — the
    earlier distance-matrix linkage took over a minute and tens of GB
    there, for the identical grouping.

    **It depends on ``inventory_generation``, NOT on ``generation``.** A
    name doesn't change the grouping — centroids arise from path, ID,
    and vector. Before, the cache depended on `generation`, which moves
    on every change: every click on "that's Alice" threw away the
    computation, and the next suggestion computed from scratch. Exactly
    in the step you do a hundred times in a row.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = None
        self._computing = False  # is a recomputation running right now?

    def fetch(self, force: bool = False) -> dict:
        from whisper_tm.core import speaker_inventory
        from whisper_tm.database.speaker import speaker_db

        db = speaker_db()
        gen = db.inventory_generation
        trash_gen = db.trash_generation
        with self._lock:
            # The cutoff belongs in the key: it changes the grouping
            # without touching the underlying data. Without it, the old
            # result would stay in place after adjusting it — and the
            # slider would be a dummy.
            if (
                self._state
                and self._state["generation"] == gen
                and self._state.get("cut") == control_value("cut")
                and not force
            ):
                return self._state

            # DELIVER STALE, RECOMPUTE IN THE BACKGROUND.
            #
            # The everyday case: the queue drops new voices every few
            # minutes, and the next suggestion would pay many seconds for
            # that — right in the middle of clicking through, without
            # the user having done anything. For the question "which
            # group do I take next," a grouping from five minutes ago is
            # just as usable: it doesn't yet know the two newest
            # recordings, and those will simply show up next time.
            #
            # The FIRST time this doesn't work — then there's nothing to
            # deliver, and we wait.
            #
            # And it also doesn't work when the CUTOFF was changed:
            # whoever turns the slider wants to see exactly its effect.
            # A stale state would make it look like a dummy
            # (`test_changed_cut_discards_result`). Only new voices
            # are delivered later — the user wasn't waiting on those.
            cut_equal = self._state is not None and self._state.get("cut") == control_value("cut")
            if self._state is not None and cut_equal and not force:
                if time.time() - self._state.get("computed_at", 0.0) >= RECOUNT_EVERY_SEC:
                    self._trigger_recount()
                return self._state

        from whisper_tm.core import debug

        t0 = time.time()
        mp = debug.MeasurePoints("Assignment")
        state_info = speaker_inventory.inventory()
        mp.point("Load vectors")
        X, meta = state_info["X"], state_info["meta"]
        removed = db.trash_ids()
        if removed:
            keep = [i for i, m in enumerate(meta) if m["id"] not in removed]
            X, meta = X[keep], [meta[i] for i in keep]
        mp.point("Trash")

        new_value = self._compute(X, meta, mp)
        new_value["generation"] = gen
        new_value["trash_generation"] = trash_gen
        new_value["cut"] = control_value("cut")
        new_value["computed_sec"] = round(time.time() - t0, 2)
        mp.done(f"{len(X)} windows, {len(new_value['key'])} speakers")
        # Put the breakdown INTO the result too, not just in the log:
        # anyone who has to dig for the numbers in the log never sees
        # them.
        new_value["times"] = [{"step": w, "sec": round(d, 3)} for w, d in mp.steps()]
        new_value["computed_at"] = time.time()
        logger.info(
            "Assignment: %d windows → %d speakers → %d groups (%.1f s) — %s",
            len(X),
            len(new_value["key"]),
            new_value["n_groups"],
            new_value["computed_sec"],
            " · ".join(f"{w} {d:.1f}s" for w, d in mp.steps()),
        )
        with self._lock:
            self._state = new_value
        return new_value

    def _trigger_recount(self) -> None:
        """Kick off a thread that refreshes the grouping.

        At most one: two simultaneous computations cost double the
        memory and deliver the same result.
        """
        if self._computing:
            return
        self._computing = True

        def _run():
            try:
                self.fetch(force=True)
            except Exception:
                logger.exception("Assignment: recompute failed")
            finally:
                with self._lock:
                    self._computing = False

        threading.Thread(target=_run, name="assignment-recompute", daemon=True).start()

    def discard(self) -> None:
        with self._lock:
            self._state = None

    def held(self) -> dict | None:
        """The grouping if one is held — never computes one."""
        with self._lock:
            return self._state

    def refresh_soon(self) -> None:
        """Regroup in the background now, not after the usual minute.

        Voices that came back from the trash (an undo in the history) are
        not in the held grouping — it was computed without them. Until the
        next grouping they can be named from the history directly, but the
        guided assignment would not ask about them.
        """
        if self._state is not None:
            self._trigger_recount()

    # ── Computation ──────────────────────────────────────────────────────

    @staticmethod
    def _centroids(X, meta):
        """One centroid per recording and speaker label — vectorized.

        The loop that did this per group took 2 s on a large collection; one
        pass assigns group numbers in order of first appearance (the order
        the loop produced, so the grouping is numbered as before), a sparse
        product sums the rows.
        """
        codes = np.empty(len(meta), np.int64)
        first: dict = {}
        speech = np.empty(len(meta), np.float64)
        ids = np.empty(len(meta), np.int64)
        for i, m in enumerate(meta):
            codes[i] = first.setdefault((m["abs_path"], m["label"]), len(first))
            speech[i] = m.get("speech_sec") or 0
            ids[i] = m["id"]
        keys = list(first)
        k = len(keys)
        per_group_speech = np.bincount(codes, weights=speech, minlength=k)
        take = np.flatnonzero(per_group_speech >= MIN_SPEECH_SEC)
        if not len(take):
            return None
        from scipy.sparse import csr_matrix

        rows = csr_matrix(
            (np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(k, len(codes))
        )
        sums = np.asarray(rows[take] @ X, dtype=np.float32)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], take)
        counts = np.bincount(codes, minlength=k)[take]
        vectors = [ids[order[b : b + c]].tolist() for b, c in zip(bounds, counts)]
        return (
            [keys[g] for g in take],
            _nrm(sums).astype(np.float32),
            per_group_speech[take],
            counts,
            vectors,
        )

    @staticmethod
    def _compute(X, meta, mp=None) -> dict:
        """Build centroids and link them completely."""
        built = Inventory._centroids(X, meta)
        if built is None:
            return {
                "key": [],
                "C": np.zeros((0, 1), np.float32),
                "language": np.zeros(0),
                "windows": np.zeros(0, int),
                "vectors": [],
                "group_nr": np.zeros(0, int),
                "n_groups": 0,
                "recurring": set(),
            }
        key, C, language, windows, vectors = built
        if mp is not None:
            mp.point("Centroids")
        # A single centroid has no distance matrix — scipy throws
        # "empty distance matrix" there. It's trivially its own group,
        # and this case comes up in practice (a first recording).
        group_nr = np.zeros(1, dtype=int) if len(C) < 2 else Inventory._concat(C, mp)

        # Recurring = in at least two recordings. Everything else is
        # not shown (see module header).
        files = defaultdict(set)
        for i, g in enumerate(group_nr):
            files[int(g)].add(key[i][0])
        recurring = {g for g, d in files.items() if len(d) >= 2}
        if mp is not None:
            mp.point("Post-processing")

        return {
            "key": key,
            "C": C,
            "language": np.asarray(language, dtype=float),
            "windows": np.asarray(windows),
            "vectors": vectors,
            "group_nr": group_nr,
            "n_groups": int(len(set(group_nr.tolist()))),
            "recurring": recurring,
        }

    @staticmethod
    def _concat(C: np.ndarray, mp=None) -> np.ndarray:
        """Complete linkage cut at the configured value.

        The sparse linkage (core/speaker_linkage.py) — no distance matrix,
        compiled and parallel: on a large collection seconds and a gigabyte or
        two instead of over a minute and tens of GB, the identical partition. Without numba (a lean
        install) the matrix path below still runs.
        """
        from whisper_tm.core import debug, speaker_linkage

        if not speaker_linkage.HAVE_NUMBA:
            logger.warning(
                "Assignment: numba missing — linking with the "
                "distance matrix (slow, memory ~ centroids²)"
            )
            return Inventory._concat_matrix(C, mp)
        t0 = time.perf_counter()
        groups = speaker_linkage.complete_groups(C, control_value("cut"), mp)
        debug.record("Assignment · linking", time.perf_counter() - t0)
        return groups

    @staticmethod
    def _concatenator():
        """The fastest available linkage — and who provides it.

        `fastcluster` computes the same method as scipy, just in a
        faster form. Measured at tens of thousands of centroids:
        fastcluster takes well under half of scipy's time — **result identical, character
        for character, the same partition**. It's a recommendation, not
        a requirement; if it's missing, everything keeps running with
        scipy.
        """
        try:
            import fastcluster

            return fastcluster.linkage
        except ImportError:
            from scipy.cluster.hierarchy import linkage

            return linkage

    @staticmethod
    def _concat_matrix(C: np.ndarray, mp=None) -> np.ndarray:
        """Complete linkage with distance matrices, component by component —
        the earlier path, now only the fallback without numba.

        **Why this gives the same result.** The cut happens at
        ``1 - cut``. Two centroids only land in the same group under
        complete linkage if their distance is below the cutoff — every
        finished group is therefore a clique in the graph of edges below
        the cutoff, and thus lies entirely within ONE connected
        component. Across component boundaries, merging only happens
        above the cutoff, and `fcluster` throws that away anyway. A
        comparison on a real collection confirms it: identical partition
        (`tests/test_assignment_speed.py`).

        **What it gains.** At tens of thousands of centroids they fall apart into a few
        thousand components, most of them singletons that don't need to be
        linked at all. Measured:

            ============================  ======
            scipy, all at once (before)   ~30 s
            split + scipy                 ~15 s
            split + fastcluster           ~10 s
            ============================  ======

        Memory drops along with it: instead of one distance matrix over
        everything (several GiB, plus scipy's float64 copy), just one per
        component.

        **What it does NOT solve.** One component with most of the
        centroids remains — those voices are linked together via chains. It's the
        reason ~30 s doesn't become 3.

        The fallback without scipy still exists: an assignment path that
        wouldn't even start without a heavy dependency would be useless
        in a lean install.
        """
        try:
            from scipy.cluster.hierarchy import fcluster
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            from scipy.spatial.distance import squareform
        except ImportError:
            return Inventory._link_slow(C)

        cut = control_value("cut")
        tau = 1.0 - cut
        n = len(C)
        concat = Inventory._concatenator()

        # ── 1. Edges below the cutoff, block by block ────────────────────
        # Block-wise, because the full similarity matrix at tens of
        # thousands of centroids would be several GiB. This keeps peak memory at
        # BLOCK × n and the runtime the same (BLAS sees the same
        # multiplications).
        # Only compute the upper triangle: the similarity matrix is
        # symmetric, and every edge is only needed once anyway.
        # So instead of `C[block] @ C.T`, `C[block] @ C[from here].T` —
        # that halves the multiplications — and about halves the time.
        BLOCK = 4096
        lines, columns = [], []
        for start in range(0, n, BLOCK):
            S = C[start : start + BLOCK] @ C[start:].T
            a, b = np.nonzero(S >= cut)
            a = a + start
            b = b + start
            top = a < b  # each edge once
            lines.append(a[top])
            columns.append(b[top])
        lines = np.concatenate(lines) if lines else np.zeros(0, dtype=np.int64)
        columns = np.concatenate(columns) if columns else np.zeros(0, dtype=np.int64)
        if mp is not None:
            mp.point(f"Edges ({len(lines)})")

        # ── 2. Components ─────────────────────────────────────────────────
        graph = coo_matrix((np.ones(len(lines), np.int8), (lines, columns)), shape=(n, n))
        _count, marker = connected_components(graph, directed=False)
        if mp is not None:
            mp.point(f"Components ({_count})")

        # ── 3. Complete linkage per component ────────────────────────────
        from whisper_tm.core import debug

        order = np.argsort(marker, kind="stable")
        limits = np.flatnonzero(np.diff(marker[order])) + 1
        group_nr = np.zeros(n, dtype=int)
        next_one = 1
        largest = (0.0, 0)  # (seconds, centroids)
        t_part = time.perf_counter()
        for part in np.split(order, limits):
            if len(part) == 1:
                # A singleton is trivially its own group — scipy throws
                # "empty distance matrix" here.
                group_nr[part[0]] = next_one
                next_one += 1
                continue
            Cs = C[part]
            D = 1.0 - (Cs @ Cs.T)
            np.fill_diagonal(D, 0.0)
            np.clip(D, 0.0, 2.0, out=D)
            Z = concat(squareform(D, checks=False), method="complete")
            g = fcluster(Z, t=tau, criterion="distance")
            group_nr[part] = next_one + g - 1
            next_one += int(g.max())
            # Report the largest component separately: it's the reason
            # the computation doesn't get shorter, and without its own
            # number it would vanish into the total.
            if len(part) > largest[1]:
                largest = (time.perf_counter() - t_part, len(part))
            t_part = time.perf_counter()
        if mp is not None:
            mp.point("Linking")
            debug.record("Assignment · largest component", largest[0])
            if largest[1]:
                logger.debug("largest component: %d centroids, %.1f s", largest[1], largest[0])
        return group_nr

    @staticmethod
    def _link_slow(C: np.ndarray) -> np.ndarray:
        S = C @ C.T
        n = len(C)
        group_nr = np.arange(n)
        active = {i: [i] for i in range(n)}
        while True:
            best, bw = None, control_value("cut")
            key_ = list(active)
            for a in range(len(key_)):
                for b in range(a + 1, len(key_)):
                    ia, ib = active[key_[a]], active[key_[b]]
                    w = S[np.ix_(ia, ib)].min()  # complete = the WORST pair
                    if w > bw:
                        best, bw = (key_[a], key_[b]), w
            if best is None:
                break
            a, b = best
            active[a].extend(active.pop(b))
            for i in active[a]:
                group_nr[i] = a
        return group_nr


INVENTORY = Inventory()


# ── The pipeline ────────────────────────────────────────────────────────


def _flat(state_info: dict) -> tuple[np.ndarray, np.ndarray]:
    """All centroids' vector ids in one array, and where each one's start —
    built once per held grouping."""
    f = state_info.get("_flat")
    if f is None:
        vecs = state_info["vectors"]
        lens = np.fromiter((len(v) for v in vecs), dtype=np.int64, count=len(vecs))
        flat = np.fromiter((v for vs in vecs for v in vs), dtype=np.int64, count=int(lens.sum()))
        ptr = np.zeros(len(vecs) + 1, dtype=np.int64)
        np.cumsum(lens, out=ptr[1:])
        f = state_info["_flat"] = (flat, ptr)
    return f


def discarded(state_info: dict) -> np.ndarray | None:
    """Centroids whose voices have ALL gone to the trash since the grouping
    was computed — ``None`` while the trash hasn't changed since.

    The grouping leaves the trash out when it is computed. Discarding a
    suggestion used to recompute it right in the click — seconds of waiting
    per "Discard". Now the held grouping stays and this mask closes what
    was discarded, like a name closes it; the grouping follows in the
    background. A whole group discarded changes no other group of the
    complete linkage (its members never merged with anything outside).

    Held per trash state: a running queue makes every grouping stale
    within seconds, and all trashed ids were read four times per
    suggestion for nothing.
    """
    from whisper_tm.database.speaker import speaker_db

    db = speaker_db()
    tg = db.trash_generation
    if tg == state_info.get("trash_generation"):
        return None
    held = state_info.get("_discarded")
    if held is not None and held[0] == tg:
        return held[1]
    trash = db.trash_ids()
    if not trash or not len(state_info["C"]):
        mask = None
    else:
        flat, ptr = _flat(state_info)
        hit = np.isin(flat, np.fromiter(trash, dtype=np.int64, count=len(trash)))
        mask = np.add.reduceat(hit.astype(np.int64), ptr[:-1]) == np.diff(ptr)
    state_info["_discarded"] = (tg, mask)
    return mask


def _named(state_info: dict, assignment: dict) -> tuple[np.ndarray, dict[int, str]]:
    """Which centroids carry a name — and, for those whose voices all carry
    the same one, that name (in centroid order).

    One pass over all centroid ids in numpy instead of a Python loop over
    tens of thousands of centroids per call, twice per suggestion (0.8 s)."""
    flat, ptr = _flat(state_info)
    n = len(ptr) - 1
    if not assignment or not n:
        return np.zeros(n, bool), {}
    known = np.fromiter(assignment.keys(), dtype=np.int64, count=len(assignment))
    has = np.logical_or.reduceat(np.isin(flat, known), ptr[:-1])
    single: dict[int, str] = {}
    for i in np.flatnonzero(has).tolist():
        names = {assignment[v] for v in flat[ptr[i] : ptr[i + 1]].tolist() if v in assignment}
        if len(names) == 1:
            single[i] = names.pop()
    return has, single


def _named_centroids(
    state_info: dict, assignment: dict | None = None
) -> tuple[list[str], np.ndarray]:
    """One centroid per person, across ALL their voices.

    Propagating against the name centroid is more stable than against
    the nearest single node: a mean over many windows wobbles less, and
    it keeps getting better with every click. Measured (module header):
    near-perfect hit rate at 25% known nodes.
    """
    from whisper_tm.database.speaker import speaker_db

    if assignment is None:
        assignment = speaker_db().names_per_vector()
    if not assignment:
        return [], np.zeros((0, state_info["C"].shape[1]), np.float32)
    gone = discarded(state_info)
    by_name = defaultdict(list)
    # A centroid with two names is a contradiction — then it counts for
    # neither (`_named` leaves it out). Better not to propagate than to
    # propagate wrongly.
    for i, name in _named(state_info, assignment)[1].items():
        if gone is not None and gone[i]:
            continue
        by_name[name].append(i)
    if not by_name:
        return [], np.zeros((0, state_info["C"].shape[1]), np.float32)
    names = sorted(by_name)
    M = _nrm(np.stack([state_info["C"][by_name[n]].mean(axis=0) for n in names]))
    return names, M.astype(np.float32)


def open_mask(state_info: dict, threshold: float | None = None) -> np.ndarray:
    """Which centroids are still without a name — after propagation?

    "Open" does NOT mean "has no label in the database," but "doesn't
    get one even through propagation." Otherwise the UI would present
    groups that have long since been decided.
    """
    from whisper_tm.core import debug
    from whisper_tm.database.speaker import speaker_db

    if threshold is None:
        threshold = control_value("spread")
    mp = debug.MeasurePoints("Open mask")
    assignment = speaker_db().names_per_vector()
    still_open = ~_named(state_info, assignment)[0]
    gone = discarded(state_info)
    if gone is not None:
        still_open[gone] = False
    mp.point("Read names")
    names, M = _named_centroids(state_info, assignment)
    mp.point("Centroids per person")
    if len(M):
        idx = np.flatnonzero(still_open)
        if len(idx):
            S = state_info["C"][idx] @ M.T
            still_open[idx[S.max(axis=1) >= threshold]] = False
    mp.point("Propagate")
    mp.done(f"{int(still_open.sum())} of {len(still_open)} open")
    return still_open


def progress(
    state_info: dict, threshold: float | None = None, still_open: np.ndarray | None = None
) -> dict:
    """How far along is the assignment — measured against RECURRING time.

    ``still_open``: the open mask, if the caller has it already — the
    suggestion computed it twice (0.8 s each)."""
    if still_open is None:
        still_open = open_mask(state_info, threshold)
    # dtype explicit: with an empty collection the list would be empty
    # and numpy would make it float64 — as a mask that would then be an
    # IndexError.
    again_val = np.array(
        [int(g) in state_info["recurring"] for g in state_info["group_nr"]], dtype=bool
    )
    target = float(state_info["language"][again_val].sum())
    remaining = float(state_info["language"][again_val & still_open].sum())
    return {
        "target_hours": round(target / 3600, 1),
        "open_hours": round(remaining / 3600, 1),
        "share": round(1 - remaining / target, 4) if target else 1.0,
        "speakers_total": int(len(state_info["C"])),
        "speakers_open": int((again_val & still_open).sum()),
        "groups": state_info["n_groups"],
        "one_off_hours": round(float(state_info["language"][~again_val].sum()) / 3600, 1),
    }


#: What the next suggestion is picked by — by hours, by speaker, or by
#: recordings.
#:
#: All three only count the OPEN part of a group — what already has a
#: name doesn't count. They answer different questions:
#:
#: * **hours** — "where's most of the speaking time?" The
#:   default, and the fastest route to coverage: over a third in under ten clicks.
#: * **speaker** (speakers) — "where do the most centroids hang
#:   together?" Finds the fragmented cases: many short appearances of
#:   the same voice.
#: * **recordings** — "who shows up in the most recordings?"
#:   That's the question about REGULARS. A group from 40 two-minute
#:   recordings brings fewer hours than one from a single long one — but
#:   it makes cross-collection search useful.
ORDERINGS = ("hours", "speaker", "recordings")


#: How many speakers of a question come with a sample — the page plays them
#: one after another, so more than the eight it used to show.
SAMPLES_AT_MOST = 48
#: The named people listed as nearest to a question's group.
NEAREST_SHOWN = 5

_named_held: dict = {}


def _named_centroids_held(state_info: dict) -> tuple[list[str], np.ndarray]:
    """:func:`_named_centroids`, kept until a name or the grouping changes —
    a question is asked after every click, the people change far less."""
    from whisper_tm.database.speaker import speaker_db

    key = (id(state_info), speaker_db().generation)
    if _named_held.get("key") != key:
        _named_held.update(key=key, value=_named_centroids(state_info))
    return _named_held["value"]


def nearest_people(state_info: dict, idx: list[int]) -> list[dict]:
    """Who a question's group sounds like — the named people nearest to it.

    Two numbers per person: how alike the group's centroid and the person's
    centroid are, and how many of the group's speakers have that person as
    their own nearest name ("votes"). A group that is clearly one known
    person shows one high line; a new voice shows low ones, and two close
    lines say "one of these two — listen". Spreading names at 0.65
    (:data:`SPREAD`), so above that the name would already have reached it
    had it been open on its own.
    """
    names, M = _named_centroids_held(state_info)
    if not names or not idx:
        return []
    X = _nrm(state_info["C"][idx].astype(np.float32))
    weight = np.asarray(state_info["language"][idx], dtype=np.float64)
    c = _nrm(
        (X * weight[:, None]).sum(axis=0, keepdims=True)
        if weight.sum() > 0
        else X.mean(axis=0, keepdims=True)
    )[0]
    sims = M @ c
    per = X @ M.T
    votes = np.bincount(per.argmax(axis=1), minlength=len(names))
    order = np.argsort(-sims)[:NEAREST_SHOWN]
    return [
        {"name": names[i], "sim": round(float(sims[i]), 3), "votes": int(votes[i])}
        for i in order.tolist()
    ]


def question_stats(state_info: dict, idx: list[int]) -> dict:
    """What else helps deciding: when (from the file names), where (the
    folders that hold most of the speech), and how the speech is spread."""
    from whisper_tm.core.voice_extract import date_from_name

    speech = np.asarray(state_info["language"][idx], dtype=np.float64)
    paths = [state_info["key"][i][0] for i in idx]
    dates = sorted(d for d in (date_from_name(p) for p in set(paths)) if d)
    folders: dict[str, float] = defaultdict(float)
    for p, sec in zip(paths, speech.tolist()):
        folders[p.rsplit("/", 1)[0].rsplit("/", 1)[-1] or "/"] += sec
    total = float(speech.sum()) or 1.0
    top = sorted(folders.items(), key=lambda kv: -kv[1])[:3]
    return {
        "first_date": dates[0] if dates else "",
        "last_date": dates[-1] if dates else "",
        "dated": len(dates),
        "folders": [{"folder": f, "share": round(sec / total, 3)} for f, sec in top],
        "speech_median_sec": round(float(np.median(speech)), 1) if len(speech) else 0.0,
        "speech_max_sec": round(float(speech.max()), 1) if len(speech) else 0.0,
        "windows": int(sum(int(state_info["windows"][i]) for i in idx)),
    }


def _sort_value(kind: str, state_info: dict, idx: list[int]) -> float:
    """How much a group weighs — depending on the chosen question."""
    if kind == "speaker":
        return len(idx)
    if kind == "recordings":
        return len({state_info["key"][i][0] for i in idx})
    return float(state_info["language"][idx].sum())


def suggestion(
    threshold: float | None = None,
    without: set[int] | list[int] | None = None,
    sort_by: str = "hours",
) -> dict | None:
    """The next group that gains the most.

    Sorted by OPEN speaking time, not by size: a group with dozens of hours
    from over a hundred recordings is one decision, a thousand single voices
    are a thousand. That's the entire difference between 15 clicks and thousands.

    ``sort_by`` selects the question: ``hours`` (default), ``speaker``,
    or ``recordings`` — see :data:`ORDERINGS`.

    ``without`` are SKIPPED groups — given as vector IDs, not group
    numbers. That's not just for convenience, it's necessary: after
    every decision, regrouping happens, and `fcluster` reassigns its
    numbers in the process. A remembered number would then point to a
    DIFFERENT group — you'd blindly skip a group you never even saw.
    Vector IDs never change.
    """
    from whisper_tm.core import debug

    mp = debug.MeasurePoints("Suggestion")
    state_info = INVENTORY.fetch()
    # First checkpoint with meaning: if the collection is warm, this is
    # nearly zero — if there's a number here, it was recomputed, and the
    # breakdown for that is in `state_info["times"]`.
    mp.point("Collection")
    if not len(state_info["C"]):
        return None
    still_open = open_mask(state_info, threshold)
    mp.point("open mask")

    by_group: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(state_info["group_nr"]):
        if still_open[i] and int(g) in state_info["recurring"]:
            by_group[int(g)].append(i)
    if not by_group:
        return None

    mp.point("collect groups")
    removed = {int(v) for v in (without or ())}
    kind = sort_by if sort_by in ORDERINGS else "hours"
    # Speaking time stays the secondary key: for "by recordings", dozens
    # of groups share the same number, and then the heaviest one should
    # come first instead of an arbitrary one.
    row = sorted(
        by_group,
        key=lambda g: (
            -_sort_value(kind, state_info, by_group[g]),
            -float(state_info["language"][by_group[g]].sum()),
        ),
    )
    best = None
    for g in row:
        # Only check once something has actually been skipped: the
        # normal case is the empty set, and that shouldn't cost
        # anything.
        if removed and any(
            int(v) in removed for i in by_group[g] for v in state_info["vectors"][i]
        ):
            continue
        best = g
        break
    mp.point("sort")
    if best is None:  # everything open has been skipped
        return None
    idx = sorted(by_group[best], key=lambda i: -state_info["language"][i])

    samples = []
    for i in idx[:SAMPLES_AT_MOST]:
        path, label = state_info["key"][i]
        samples.append(
            {
                "vector_id": int(state_info["vectors"][i][0]),
                "all_vectors": [int(v) for v in state_info["vectors"][i]],
                "file": path,
                "label": label,
                "windows": int(state_info["windows"][i]),
                "speech_sec": round(float(state_info["language"][i]), 1),
            }
        )
    mp.point("Samples")
    try:
        nearest = nearest_people(state_info, idx)
        stats = question_stats(state_info, idx)
    except Exception:  # noqa: BLE001 — help for deciding, never a reason to fail
        logger.exception("Assignment: stats for the question failed")
        nearest, stats = [], {}
    mp.point("Nearest people")
    progress_now = progress(state_info, threshold, still_open=still_open)
    mp.point("Progress")
    mp.done()
    return {
        "group_nr": int(best),
        "speaker": len(idx),
        "recordings": len({state_info["key"][i][0] for i in idx}),
        "speech_hours": round(float(state_info["language"][idx].sum()) / 3600, 2),
        "samples": samples,
        "nearest": nearest,
        "stats": stats,
        "vector_ids": [int(v) for i in idx for v in state_info["vectors"][i]],
        "sort_by": kind,
        "progress": progress_now,
        # What this call cost, broken down — and, if it was recomputed,
        # the breakdown of that computation as well.
        "times": {
            "total_sec": round(sum(d for _, d in mp.steps()), 3),
            "steps": [{"step": w, "sec": round(d, 3)} for w, d in mp.steps()],
            "inventory": state_info.get("times"),
            "inventory_computed_sec": state_info.get("computed_sec"),
            # How old the grouping this suggestion is based on is, and
            # whether a new one is currently being computed. Without
            # these two numbers, "deliver stale" would be a silent
            # assumption.
            "inventory_age_sec": (
                round(time.time() - state_info["computed_at"], 1)
                if state_info.get("computed_at")
                else None
            ),
            "recomputing": bool(INVENTORY._computing),
        },
    }


def apply(vector_ids: list[int], name: str, source: str = "assignment") -> dict:
    """Set a name and propagate it.

    Propagation happens IMMEDIATELY, not just at the next suggestion:
    the user should see the payoff of their click right away —
    otherwise a click that decides 2,000 voices would look the same as
    one that decides 12.
    """
    from whisper_tm.database.speaker import speaker_db

    db = speaker_db()
    # The click and its spread are ONE decision: undoing it takes both
    # back (database/speaker.py, "History of decisions").
    with db.decision("name", origin=source, name=name) as d:
        result = _apply(db, vector_ids, name, source)
    result["decision"] = d.id
    return result


def _apply(db, vector_ids: list[int], name: str, source: str) -> dict:
    db.assign_many(vector_ids, name, source=source)
    # NO discard() here. There used to be one, with the reasoning
    # "generation has changed" — and that was the most expensive step in
    # the entire pipeline: every assignment made the grouping recompute
    # twice (once here, once after propagation), many seconds each.
    #
    # A name doesn't move a centroid. What it changes — which voices are
    # already assigned — is read fresh from the database on every
    # access anyway (`names_per_vector`), not from this collection state.
    state_info = INVENTORY.fetch()

    names, M = _named_centroids(state_info)
    if not len(M):
        return {"set_flag": len(vector_ids), "spread": 0, "progress": progress(state_info)}
    # The typed name as stored: "alice" names the voices of
    # "Alice" (the table folds case) — and must spread like it.
    from whisper_tm.database.speaker import fold_name

    j = next((k for k, n in enumerate(names) if fold_name(n) == fold_name(name)), None)
    spread = 0
    if j is not None:
        assignment = db.names_per_vector()
        free_mask = ~_named(state_info, assignment)[0]
        gone = discarded(state_info)
        if gone is not None:
            # Never spread a name onto discarded voices
            free_mask &= ~gone
        free = np.flatnonzero(free_mask).tolist()
        if free:
            S = state_info["C"][free] @ M.T
            takes = (S.argmax(axis=1) == j) & (S.max(axis=1) >= control_value("spread"))
            more = [v for k, i in enumerate(free) if takes[k] for v in state_info["vectors"][i]]
            if more:
                db.assign_many(more, name, source=source + "-spread")
                spread = len(more)
                # Also not here: propagation only sets names, nothing else.
    return {"set_flag": len(vector_ids), "spread": spread, "progress": progress(state_info)}

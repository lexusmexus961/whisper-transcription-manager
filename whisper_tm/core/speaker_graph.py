"""The voice map as a graph — the answer of ``/api/voicemap/graph``.

**Why it is built in held parts.** Measured with ~500k voices on a
consumer GPU, per map request:

=======================================  ============
Loading the inventory                      4.1 s
Neighbourhood (kNN, GPU)                   17.8–37 s
Grouping (fine cut)                        16–17 s
Collecting edges (Python loop)             2.1 s
Nodes, legend, JSON (~70 MB)               ~6 s
=======================================  ============

The guided assignment reloads the map after every "Apply", and an Apply
changes names — nothing else. Every row above was recomputed for it: 53 s
per click. Names reach exactly two things in the answer, the ``name``
column and the group legend; everything else depends on the vectors, the
trash selection and the cut. So the answer is held in three layers:

``view``
    What the vectors and the trash decide: which voices are on the map,
    and their columns (``id``, ``label``, ``start``, …) already as JSON.
    Held by the database's ``inventory_generation`` — new or deleted
    recordings and the trash move it, names don't.
``cut``
    What the threshold, the linkage and ``k`` decide on top: the groups,
    the edges, each group's mean edge strength — also as JSON.
per request
    The names: the ``name`` column and the legend entries of the groups
    that contain a named voice. Everything else is joined as it was
    encoded.

The answer is byte for byte what ``jsonify`` wrote before (compact, keys
sorted, ASCII) — the test compares both.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import threading

import numpy as np

from whisper_tm.core import paths

logger = logging.getLogger(__name__)

#: Views held: the map and the trash map.
VIEWS_MAX = 2
#: Cuts held: the last two slider positions. One is ~60 MB at ~500k
#: voices — ~25 MB of edges as JSON, the legend of ~150k groups, the
#: group numbers.
CUTS_MAX = 2

_lock = threading.Lock()
#: Building a view or a cut is heavy (seconds, and the kNN on the GPU):
#: one at a time. Whoever waited for the same one finds it held.
_build = threading.Lock()
_views: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
_cuts: collections.OrderedDict[tuple, dict] = collections.OrderedDict()


def clear_cache() -> None:
    """Discard held views and cuts — for tests and directory switches."""
    with _lock:
        _views.clear()
        _cuts.clear()


def _dump(value) -> str:
    """JSON exactly as ``jsonify`` writes it: compact, sorted keys, ASCII."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _num(x) -> str:
    """A legend number as `json.dumps` writes it — ``repr`` of a finite
    float is exactly that (NaN and infinity go through `_dump`)."""
    return "null" if x is None else repr(x)


def _legend_entry(nr: int, n: int, mean) -> str:
    """One group of the legend without names, as `_dump` would write it —
    at a tenth of the cost (~150k groups: 0.5 s → 0.07 s)."""
    if mean is None or math.isfinite(mean):
        return f'{{"mean":{_num(mean)},"n":{n},"name":null,"nr":{nr},"people":[]}}'
    return _dump({"nr": nr, "n": n, "mean": mean, "name": None, "people": []})


def _edge_list(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> str:
    """``[[a,b,w],…]`` as `_dump` writes it — joined directly instead of
    building over a million small lists first (1.16 s → 0.56 s)."""
    if not np.isfinite(w).all():
        return _dump([[x, y, z] for x, y, z in zip(a.tolist(), b.tolist(), w.tolist())])
    return (
        "["
        + ",".join([f"[{x},{y},{z!r}]" for x, y, z in zip(a.tolist(), b.tolist(), w.tolist())])
        + "]"
    )


def _obj(parts: dict[str, str]) -> str:
    """A JSON object from already encoded values, keys sorted as `_dump` does."""
    return "{" + ",".join(f"{json.dumps(k)}:{v}" for k, v in sorted(parts.items())) + "}"


def _held(cache: collections.OrderedDict, key: tuple):
    with _lock:
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
        return hit


def _hold(cache: collections.OrderedDict, key: tuple, value: dict, at_most: int) -> None:
    with _lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > at_most:
            cache.popitem(last=False)


# ── The view: which voices, and their fixed columns ──────────────────────


def view(state_info: dict, trash_only: bool, root: str) -> dict:
    """The voices on the map (or in the trash map), with fixed columns.

    ``root`` is part of the key because the file names are shortened
    against it, and it moves when the media folder is switched.
    """
    key = (
        state_info["source"],
        state_info["model"],
        state_info["inventory_generation"],
        trash_only,
        root,
    )
    hit = _held(_views, key)
    if hit is not None:
        return hit
    with _build:
        hit = _held(_views, key)
        if hit is not None:
            return hit
        v = _make_view(state_info, trash_only, root, key)
        _hold(_views, key, v, VIEWS_MAX)
        return v


def _make_view(state_info: dict, trash_only: bool, root: str, key: tuple) -> dict:
    from whisper_tm.database.speaker import speaker_db

    meta = state_info["meta"]
    # ── The trash acts BEFORE grouping ───────────────────────────────────
    #
    # Nuisance voices shouldn't just be invisible, they should disappear
    # from the computation. That's the
    # difference that matters — game noise and a TV in the background
    # glue two real speakers into one group. Just hiding it would leave
    # that lump standing.
    #
    # `?trash=1` flips the selection and returns exactly what was
    # discarded — the same map, a different slice. A second drawing
    # routine for that would be a second truth.
    removed = speaker_db().trash_ids()
    all_ids = state_info["ids"]
    if removed:
        in_trash = np.isin(all_ids, np.fromiter(removed, dtype=np.int64, count=len(removed)))
        keep = np.flatnonzero(in_trash if trash_only else ~in_trash)
        rows = [meta[i] for i in keep.tolist()]
        ids = all_ids[keep]
    elif trash_only:
        keep, rows = np.zeros(0, dtype=np.int64), []
        ids = all_ids[:0]
    else:
        # Everything: no copy of the matrix.
        keep, rows, ids = None, meta, all_ids

    def display(p: str) -> str:
        return paths.shown(p, root)

    # The file path is sent once per recording and referenced by index:
    # in a large collection voices outnumber recordings about 20 to 1,
    # and repeating the path per voice was close to half the answer.
    file_names: list[str] = []
    file_index: dict[str, int] = {}
    file_idx = []
    for m in rows:
        shown = display(m["abs_path"])
        if shown not in file_index:
            file_index[shown] = len(file_names)
            file_names.append(shown)
        file_idx.append(file_index[shown])

    return {
        "key": key,
        # What `speaker_knn` holds its results by: the same voices — and
        # what stays the same while voices come and go.
        "marker": key[:4],
        "family": (key[0], key[1], key[3]),
        "keep": keep,
        "n": len(rows),
        "ids": ids,
        # The inventory is read `ORDER BY id`, and the trash keeps the
        # order — so names can be placed by binary search.
        "ids_sorted": bool(np.all(ids[1:] > ids[:-1])),
        "trash": len(removed),
        "files": len({m["abs_path"] for m in rows}),
        "file_names": _dump(file_names),
        "nodes": {
            "id": _dump(ids.tolist()),
            "file_idx": _dump(file_idx),
            "label": _dump([m["label"] for m in rows]),
            "start": _dump([m["start_sec"] for m in rows]),
            # Window and speaking duration belong with it: without them
            # the selection field said "window 0 · 0 s speech" — a value
            # that looks like a measurement but isn't one.
            "windows": _dump([m["windows"] for m in rows]),
            "speech_sec": _dump([m["speech_sec"] for m in rows]),
        },
    }


def matrix(state_info: dict, v: dict) -> np.ndarray:
    """The view's vectors — a copy only when the trash takes some away."""
    X = state_info["X"]
    return X if v["keep"] is None else X[v["keep"]]


# ── The cut: groups and edges at one threshold ────────────────────────────


def cut(state_info: dict, v: dict, limit: float, link_method: str, k: int, device: str) -> dict:
    """Groups, edges and the name-free legend for one slider position."""
    from whisper_tm.core import debug, speaker_knn

    key = (v["key"], limit, link_method, k)
    hit = _held(_cuts, key)
    if hit is not None:
        return hit
    with _build:
        hit = _held(_cuts, key)
        if hit is not None:
            return hit
        X = matrix(state_info, v)
        # ONE neighbourhood for both.
        #
        # Previously kNN ran twice: once in `group_voices` with K=30, then
        # again with k=5 for the edges. `topk` returns results sorted in
        # descending order, so the k strongest neighbours are exactly the
        # first k columns of the 30-neighbourhood.
        #
        # `X` is already normalized (`speaker_inventory.inventory`), and
        # `group_voices` normalizes idempotently afterward — both see the
        # same values.
        # Held by the vectors, and when they changed — a running queue adds
        # some every few seconds — updated rather than recomputed: `family`
        # is what stays (database, model, selection), `ids` which voice
        # each row is.
        with debug.timed("Map: neighbourhood (kNN)", voices=len(X), device=device):
            full_idx, full_value = speaker_knn.held_neighbours(
                X, v["marker"], k=speaker_knn.K, device=device, ids=v["ids"], family=v["family"]
            )
        with debug.timed("Map: grouping", threshold=limit, link_method=link_method):
            run = speaker_knn.group_voices(
                X,
                limit,
                link_method=link_method,
                device=device,
                neighbourhood=(full_idx, full_value),
                marker=v["marker"],
                ids=v["ids"],
            )
        with debug.timed("Map: collecting edges", k=k):
            c = _make_cut(np.asarray(run["markers"]), full_idx[:, :k], full_value[:, :k], limit)
        c["key"] = key
        _hold(_cuts, key, c, CUTS_MAX)
        return c


def edges(
    idx: np.ndarray, value: np.ndarray, limit: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per node the ``k`` strongest neighbours at or above ``limit`` — each
    pair ONCE (``a < b``), otherwise the browser draws every line twice.

    ``(a, b, weight)`` in the order and with the values of the double loop
    this replaces (2.1 s at ~500k voices, as Python)::

        for i in range(n):
            for s in range(k):
                j, w = int(idx[i, s]), float(value[i, s])
                if j == i or w < limit:
                    continue
                edges[(min(i, j), max(i, j))] = round(w, 4)

    That is: a pair sits where it was FIRST met and keeps the weight it
    was LAST met with — from the other side it can differ in the last
    bit, because the GPU computed the two directions in different stripes.
    """
    n, k = idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = idx.reshape(-1).astype(np.int64)
    w = value.reshape(-1).astype(np.float64)
    # `~(w < limit)`, not `w >= limit`: the loop kept a NaN, so this does.
    keep = (cols != rows) & ~(w < limit)
    rows, cols, w = rows[keep], cols[keep], w[keep]
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    pair = a * max(n, 1) + b
    _, first = np.unique(pair, return_index=True)
    _, last_rev = np.unique(pair[::-1], return_index=True)
    last = len(pair) - 1 - last_rev
    order = np.argsort(first, kind="stable")
    first, last = first[order], last[order]
    # `round(w, 4)` of a float32 value, and `np.round` agree exactly: the
    # scaled float32 value never lands next to a .5 boundary it isn't on.
    return a[first], b[first], np.round(w[last], 4)


def _make_cut(markers: np.ndarray, idx: np.ndarray, value: np.ndarray, limit: float) -> dict:
    a, b, w = edges(idx, value, limit)

    # The legend's `mean` is the mean edge strength WITHIN the group: a
    # group at 0.91 is a different statement than one at 0.63, even if
    # both are the same size. Summed with Python's `sum` in edge order,
    # as before — the mean is rounded to three places, and a different
    # summation could tip one exactly there.
    ga, gb = markers[a], markers[b]
    within = (ga == gb) & (ga > 0)
    g, gw = ga[within], w[within]
    order = np.argsort(g, kind="stable")
    g, gw = g[order], gw[order].tolist()
    means: dict[int, float] = {}
    if len(g):
        bounds = np.flatnonzero(g[1:] != g[:-1]) + 1
        starts = [0, *bounds.tolist()]
        ends = [*bounds.tolist(), len(g)]
        for s, e in zip(starts, ends):
            means[int(g[s])] = round(sum(gw[s:e]) / (e - s), 3)

    # Groups by size, then number — the order the legend is sent in.
    sizes = np.bincount(markers)
    numbers = np.flatnonzero(sizes)
    numbers = numbers[numbers > 0]
    numbers = numbers[np.lexsort((numbers, -sizes[numbers]))]
    # Each group's legend entry without names, already encoded. A request
    # re-encodes only the groups that contain a named voice.
    legend = [
        _legend_entry(nr, n, means.get(nr))
        for nr, n in zip(numbers.tolist(), sizes[numbers].tolist())
    ]
    return {
        "markers": markers,
        "sizes": sizes,
        "means": means,
        "position": {nr: p for p, nr in enumerate(numbers.tolist())},
        "legend": legend,
        # Loners = voices that are ALONE in their group.
        #
        # Previously: `(markers == 0).sum()`. That counted whoever happened
        # to get label zero — and that was the first split of the largest
        # component. At threshold 0.61 that was thousands of voices, which
        # disappeared from every group view as a result
        # (`core/speaker_knn.py` — the cause is fixed there).
        "loners": int(np.count_nonzero(sizes[numbers] == 1)),
        "group_nr": _dump(markers.tolist()),
        "edges": _edge_list(a, b, w),
    }


# ── Per request: the names ────────────────────────────────────────────────


def answer(
    v: dict, c: dict, names: dict[int, str], *, limit: float, link_method: str, k: int, root: str
) -> str:
    """The JSON of ``/api/voicemap/graph`` — held parts plus today's names."""
    ids = v["ids"]
    n = v["n"]
    column: list = [None] * n
    named: list[tuple[int, str]] = []
    if names:
        vids = np.fromiter(names.keys(), dtype=np.int64, count=len(names))
        people = list(names.values())
        if v["ids_sorted"]:
            pos = np.searchsorted(ids, vids)
            pos[pos >= n] = 0
            found = ids[pos] == vids if n else np.zeros(len(vids), bool)
        else:
            where = {vid: i for i, vid in enumerate(ids.tolist())}
            pos = np.array([where.get(x, -1) for x in vids.tolist()], dtype=np.int64)
            found = pos >= 0
        named = sorted((int(p), people[j]) for j, p in enumerate(pos.tolist()) if found[j])
        for p, person in named:
            column[p] = person

    # The group's name is the most frequently assigned name in it — not
    # the group number. Numbers shift when the threshold moves, names
    # don't. Counted in node order, so ties go to the name met first.
    markers = c["markers"]
    counted: dict[int, dict[str, int]] = {}
    for p, person in named:
        nr = int(markers[p])
        if nr > 0 and person:
            here = counted.setdefault(nr, {})
            here[person] = here.get(person, 0) + 1
    legend = c["legend"]
    if counted:
        legend = list(legend)
        sizes, means = c["sizes"], c["means"]
        for nr, here in counted.items():
            legend[c["position"][nr]] = _dump(
                {
                    "nr": nr,
                    "n": int(sizes[nr]),
                    "mean": means.get(nr),
                    "name": max(here, key=here.get),
                    "people": sorted(here.items(), key=lambda t: -t[1]),
                }
            )

    nodes = dict(v["nodes"])
    nodes["group_nr"] = c["group_nr"]
    nodes["name"] = _dump(column)
    return (
        _obj(
            {
                "threshold": _dump(round(limit, 4)),
                "link_method": _dump(link_method),
                "k": _dump(k),
                "voices": _dump(n),
                "files": _dump(v["files"]),
                "group_count": _dump(len(legend)),
                "loners": _dump(c["loners"]),
                # The root that the file names were shortened against.
                #
                # Without it, the paths in the response can't be resolved: they
                # are relative to `config.media_dir` AT THE TIME OF LOADING, and
                # that moves when the folder changes. Whoever navigates with it
                # later would otherwise be guessing — that went wrong exactly this
                # way once ("recording not found").
                "root": _dump(root),
                # How many voices are in the trash — the map's switch shows it.
                # It came only with an EMPTY answer before, so the switch to the
                # trash map never showed on a map that had voices.
                "trash": _dump(v["trash"]),
                "nodes": _obj(nodes),
                "file_names": v["file_names"],
                "edges": c["edges"],
                "groups": "[" + ",".join(legend) + "]",
            }
        )
        + "\n"
    )

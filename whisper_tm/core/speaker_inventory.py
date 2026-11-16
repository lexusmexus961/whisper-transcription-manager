"""The voice inventory as an in-memory matrix — loaded once.

Measured: reading a few hundred thousand vectors from SQLite
costs **1.7 seconds**, querying them as a matrix **5.5
milliseconds**. This component's whole cost balance is in these two
numbers: computing is free, loading is not.

So it's loaded once and then held — 230 MB as ``float32``. That's
exactly why no vector database is needed here either: Qdrant or pgvector
would have to beat 5.5 milliseconds over a network connection.

**When the cache expires.** ``SpeakerDB`` counts every change
(``generation``) and, separately, every change to the vectors
(``inventory_generation``). A transcription running alongside pushes new
vectors in; an assigned name changes the gallery. The next request must
see both — but a name only changes the gallery. Previously it reloaded
everything, seconds of work, after every "Apply" in the guided
assignment. Now the matrix stays while ``inventory_generation`` holds,
and a name reloads only the gallery (a fraction of a second).
A timestamp would be less precise — two changes in the same second
would look like one.

**When vectors changed, the held matrix follows them.** A running queue
adds recordings every few seconds; reading all rows again for
each cost seconds. Instead the ids are compared (0.2 s through the model
index), and only new rows are read and normalized — the normalization is
per row, so the result is the same to the bit. The trash moves
``inventory_generation`` too, but not a single vector: then nothing is
read at all.

**What is NOT cached:** the raw BLOBs. What's held is exclusively the
already **normalized** matrix, because every query needs it that way; normalize
once instead of on every request.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping

import numpy as np

logger = logging.getLogger(__name__)


class _Rows(Mapping):
    """Vector id → row number, by binary search in the ascending ids.

    A dict of every vector id cost 0.1 s and tens of MB — rebuilt with every
    recording the queue added, for a handful of lookups per question
    ("who was in this recording")."""

    def __init__(self, ids: np.ndarray):
        self._ids = ids

    def __getitem__(self, vid) -> int:
        i = int(np.searchsorted(self._ids, vid))
        if i < len(self._ids) and self._ids[i] == vid:
            return i
        raise KeyError(vid)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids.tolist())


_lock = threading.Lock()
#: Exactly one entry — the inventory is singular. A dict keyed by model
#: would be possible, but would normally just double memory usage.
_state: dict = {}


def _nrm(A: np.ndarray) -> np.ndarray:
    """THE one normalization — see core/speaker.py::nrm."""
    from whisper_tm.core.speaker import nrm

    return nrm(A)


def inventory(model: str | None = None, force: bool = False) -> dict:
    """The normalized matrix, its metadata, and the person gallery.

    Returns:

    ``X``
        ``(n, dim)`` float32, **normalized**.
    ``meta``
        List of dicts per row (``id``, ``abs_path``, ``label``,
        ``windows``, ``start_sec``, ``speech_sec``, …).
    ``id_to_row``
        Reverse mapping vector ID → row number.
    ``ids``
        The vector ids per row, ascending (``int64``).
    ``gallery``
        ``{person: (m, dim)}``, likewise normalized.
    ``generation`` / ``inventory_generation``
        Database state this cache stems from — every change, and only
        the changes to the vectors. Whoever caches something computed
        from ``X`` keys it on the second (with ``source``).
    """
    from whisper_tm.core import speaker as speaker_core
    from whisper_tm.database.speaker import speaker_db

    db = speaker_db()
    db.initialize()
    model = model or speaker_core.MODEL_BRAND
    key = (db._db_path, model)
    # Both counters BEFORE loading: what is loaded afterwards is at least
    # that new, so a held state can only ever be labelled older than it is
    # — then it is loaded once too often, never once too rarely.
    inventory_gen = db.inventory_generation
    gen = db.generation

    with _lock:
        old_value = _state.get(key)
    if old_value and not force:
        if old_value["generation"] == gen:
            return old_value
        if old_value["inventory_generation"] == inventory_gen:
            # Only names changed: same matrix, same rows — a new gallery.
            t0 = time.time()
            new_value = {
                **old_value,
                "gallery": _gallery(db, model),
                "generation": gen,
                "loaded_sec": round(time.time() - t0, 2),
            }
            with _lock:
                _state[key] = new_value
            return new_value
        new_value = _follow(db, old_value, model, gen, inventory_gen)
        if new_value is not None:
            with _lock:
                _state[key] = new_value
            return new_value

    # Load outside the lock. For a few hundred thousand rows this takes 1.7 s, and no
    # other request may be stuck waiting that long — in the worst case
    # two load simultaneously and one discards its result. That is
    # cheaper than a blocked server.
    t0 = time.time()
    X, meta = db.all_vectors(model=model)
    ids = np.fromiter((m["id"] for m in meta), dtype=np.int64, count=len(meta))
    new_value = {
        "X": _nrm(X) if len(X) else X,
        "meta": meta,
        "id_to_row": _Rows(ids),
        "ids": ids,
        "gallery": _gallery(db, model),
        "generation": gen,
        "inventory_generation": inventory_gen,
        "model": model,
        # Where the inventory comes from. It's here because `generation`
        # alone doesn't identify an inventory: two different databases both
        # eventually land on 1 — regularly, even, in tests. Whoever caches
        # against the state (see `speaker_knn.group_voices(marker=…)`) needs
        # the origin for that.
        "source": db._db_path,
        "loaded_sec": round(time.time() - t0, 2),
    }
    logger.info(
        "Voice inventory loaded: %d vectors, %d named people, %.2f s",
        len(new_value["X"]),
        len(new_value["gallery"]),
        new_value["loaded_sec"],
    )
    with _lock:
        _state[key] = new_value
    return new_value


#: Above this share of changed rows, reading everything is as cheap.
FOLLOW_AT_MOST = 0.2


def _follow(db, old: dict, model: str, gen: int, inventory_gen: int) -> dict | None:
    """The held inventory, brought up to the database's vectors — or
    ``None`` where reading everything is simpler."""
    held = old["ids"]
    if not len(held):
        return None
    t0 = time.time()
    now = db.vector_ids(model)
    gone = ~np.isin(held, now, assume_unique=True)
    new_ids = now[~np.isin(now, held, assume_unique=True)]
    if int(gone.sum()) + len(new_ids) > FOLLOW_AT_MOST * len(held):
        return None
    X, meta, ids = old["X"], old["meta"], held
    if gone.any():
        keep = np.flatnonzero(~gone)
        X, meta, ids = X[keep], [meta[i] for i in keep.tolist()], held[keep]
    Xa, meta_a = db.vectors_by_id(new_ids, model=model) if len(new_ids) else (None, [])
    if meta_a:
        ids_a = np.fromiter((m["id"] for m in meta_a), dtype=np.int64, count=len(meta_a))
        X = np.concatenate([X, _nrm(Xa)]) if len(X) else _nrm(Xa)
        meta = meta + meta_a
        ids = np.concatenate([ids, ids_a])
        if len(ids) > 1 and not np.all(ids[1:] > ids[:-1]):
            # New ids below held ones (not how SQLite hands them out, but
            # possible): into id order, as a full read would have them.
            order = np.argsort(ids, kind="stable")
            X, ids = X[order], ids[order]
            meta = [meta[i] for i in order.tolist()]
    changed = X is not old["X"]
    new_value = {
        **old,
        "X": X,
        "meta": meta,
        "ids": ids,
        "id_to_row": _Rows(ids) if changed else old["id_to_row"],
        "gallery": _gallery(db, model),
        "generation": gen,
        "inventory_generation": inventory_gen,
        "loaded_sec": round(time.time() - t0, 2),
    }
    logger.info(
        "Voice inventory followed: %d new, %d gone, %d vectors, %.2f s",
        len(meta_a),
        int(gone.sum()),
        len(X),
        new_value["loaded_sec"],
    )
    return new_value


def _gallery(db, model: str) -> dict[str, np.ndarray]:
    """``{person: (m, dim)}``, normalized — the named voices of ``model``
    that are not in the trash (``SpeakerDB.person_vectors``).

    The result also carries ``gallery_ids`` — see :func:`gallery_ids`."""
    ids: dict = {}
    g = _Gallery({p: _nrm(V) for p, V in db.person_vectors(model=model, ids_out=ids).items()})
    g.ids = ids
    return g


class _Gallery(dict):
    """The gallery dict, with each person's vector ids (``ids``) in row
    order — so a caller can leave one recording's own voices out."""

    ids: dict


def gallery_ids(state: dict) -> dict[str, np.ndarray]:
    """``{person: vector ids}`` of the held gallery, in its rows' order —
    empty for a gallery that doesn't carry them."""
    return getattr(state.get("gallery"), "ids", None) or {}


def discard() -> None:
    """Clear the cache — for tests and directory switches."""
    with _lock:
        _state.clear()


def where_it_occurs(person: str, threshold: float | None = None, at_most: int = 500) -> dict:
    """**The inventory question: which recordings does this person appear in?**

    This is the question one level above "who was in this
    recording" — and the actual payoff of the whole build. Technically it's
    a single matrix product against the held inventory; at a few hundred
    thousand vectors it costs milliseconds.

    The decision is made on **individual voices**, not the centroid: measured,
    the best individual value beats the centroid at every operating point, by
    15 points at zero false alarms.

    The answer is sorted by hit count and carries the minutes along per
    recording — without timestamps you can't listen back to a hit.
    """
    from whisper_tm.core import speaker as speaker_core

    b = inventory()
    if not len(b["X"]) or person not in b["gallery"]:
        return {"person": person, "recordings": [], "found": 0, "searched": len(b["X"])}
    threshold = speaker_core.threshold("gallery") if threshold is None else threshold

    E = b["gallery"][person]
    best = (b["X"] @ E.T).max(axis=1)
    hits = np.flatnonzero(best >= threshold)

    per_file: dict[str, dict] = {}
    for i in hits:
        m = b["meta"][int(i)]
        d = per_file.setdefault(
            m["abs_path"],
            {
                "abs_path": m["abs_path"],
                "windows": 0,
                "best_value": 0.0,
                "timestamps": set(),
                "labels": set(),
            },
        )
        d["windows"] += 1
        d["best_value"] = max(d["best_value"], float(best[i]))
        d["timestamps"].add(int(m["start_sec"] // 60))
        d["labels"].add(m["label"])

    recordings = sorted(per_file.values(), key=lambda d: (-d["windows"], -d["best_value"]))
    for d in recordings:
        d["timestamps"] = sorted(d["timestamps"])[:40]
        d["labels"] = sorted(d["labels"])
        d["best_value"] = round(d["best_value"], 3)

    return {
        "person": person,
        "threshold": threshold,
        "found": len(recordings),
        "searched": len(b["X"]),
        "voices_in_gallery": int(len(E)),
        "truncated": len(recordings) > at_most,
        "recordings": recordings[:at_most],
    }

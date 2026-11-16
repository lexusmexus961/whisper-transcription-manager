"""Listening through one person's voices — every voice that carries the
name, not six samples.

A list to listen through everything with autoplay and check it — like
the assignment's play all, only the first second — with many filters.

**How well a voice fits the person.** Its best similarity to the person's
directly named voices in OTHER recordings. The windows of one recording
resemble each other anyway, so a wrong name given to all of them would
vouch for itself; another recording has to agree. Voices named by
spreading are measured against the direct ones, never against each other.

**Nearer to someone else.** The centroid of another named person lies
nearer than the person's own — the clearest hint of a wrong name.

Ordered by fit, the doubtful voices come first. The table is computed once
per person and naming state (milliseconds to a second for the largest
person) and then filtered and paged from memory; a decision changes the
state and the next request computes it anew.

Decisions go through the history like everywhere else: removing the name,
discarding, or naming as someone else is one entry each, undoable.
"""

from __future__ import annotations

import collections
import threading

import numpy as np

REFERENCE_AT_MOST = 3000
SORTS = ("fit", "date_desc", "date_asc", "recording", "speech_desc")

_lock = threading.Lock()
_held: collections.OrderedDict[tuple, dict] = collections.OrderedDict()


def _no_spans(parts) -> bool:
    return parts is None or str(parts).strip() in ("", "[]", "null")


def table(person: str) -> dict:
    """Every voice of the person with its fit — held per naming state."""
    from whisper_tm.core import speaker_inventory
    from whisper_tm.core.speaker import nrm
    from whisper_tm.core.voice_extract import date_from_name
    from whisper_tm.database.speaker import fold_name, speaker_db

    db = speaker_db()
    b = speaker_inventory.inventory()
    want = fold_name(person)
    key = (b["source"], db.generation, want)
    with _lock:
        hit = _held.get(key)
        if hit is not None:
            _held.move_to_end(key)
            return hit

    with db.connection() as c:
        named = c.execute(
            "SELECT z.vector_id, z.source, p.name FROM speaker_assignment z "
            "JOIN speaker_people p ON p.id = z.person_id"
        ).fetchall()
    mine = {int(r[0]): (r[1] or "") for r in named if fold_name(r[2]) == want}
    shown = next((r[2] for r in named if fold_name(r[2]) == want), person)
    if not mine:
        raise KeyError(person)
    trash = db.trash_ids()
    rows = []
    for vid in sorted(mine):
        if vid in trash:
            continue
        try:
            rows.append(b["id_to_row"][vid])
        except KeyError:
            continue
    n = len(rows)
    X = b["X"][rows] if n else np.zeros((0, 1), np.float32)
    meta = [b["meta"][i] for i in rows]
    direct = np.array([not mine[m["id"]].endswith("spread") for m in meta], bool)
    recs: dict[str, int] = {}
    rec = np.array([recs.setdefault(m["abs_path"], len(recs)) for m in meta], np.int64)

    # Fit: best similarity to direct voices of other recordings
    ref = np.flatnonzero(direct) if direct.any() else np.arange(n)
    if len(ref) > REFERENCE_AT_MOST:
        ref = ref[np.linspace(0, len(ref) - 1, REFERENCE_AT_MOST).astype(int)]
    fit = np.full(n, np.nan, np.float32)
    if n and len(ref):
        E, Erec = X[ref], rec[ref]
        step = max(256, int(24_000_000 // max(1, len(ref))))
        for a in range(0, n, step):
            S = X[a : a + step] @ E.T
            S[rec[a : a + step, None] == Erec[None, :]] = -2.0
            best = S.max(axis=1)
            fit[a : a + step] = np.where(best > -1.5, best, np.nan)

    # Nearer to someone else: the other people's centroids
    own = np.zeros(n, np.float32)
    other_name: list[str | None] = [None] * n
    other_sim = np.zeros(n, np.float32)
    if n:
        base = X[direct] if direct.any() else X
        c_own = nrm(base.mean(axis=0, keepdims=True))[0]
        own = X @ c_own
        others = [k for k in sorted(b["gallery"]) if fold_name(k) != want and len(b["gallery"][k])]
        if others:
            C = nrm(np.stack([b["gallery"][k].mean(axis=0) for k in others])).astype(np.float32)
            Co = X @ C.T
            k = Co.argmax(axis=1)
            other_sim = Co[np.arange(n), k]
            for j in np.flatnonzero(other_sim > own).tolist():
                other_name[j] = others[int(k[j])]

    no_spans = set()
    ids = [m["id"] for m in meta]
    with db.connection() as c:
        for a in range(0, len(ids), 900):
            chunk = ids[a : a + 900]
            marks = ",".join("?" * len(chunk))
            for r in c.execute(
                f"SELECT id, parts FROM speaker_vectors WHERE id IN ({marks})", chunk
            ):
                if _no_spans(r[1]):
                    no_spans.add(int(r[0]))

    from whisper_tm.config import config
    from whisper_tm.core import paths

    try:
        root = paths.canon(config.media_dir)  # the stored form
    except Exception:  # noqa: BLE001 — a display path only
        root = ""
    dates: dict[str, str] = {}
    voices = []
    for j, m in enumerate(meta):
        p = m["abs_path"]
        if p not in dates:
            dates[p] = date_from_name(p)
        display = paths.shown(p, root) if root else p
        voices.append(
            {
                "id": int(m["id"]),
                "abs_path": p,
                "display": display,
                "file": p.rsplit("/", 1)[-1],
                "date": dates[p],
                "label": m["label"],
                "start": round(float(m.get("start_sec") or 0.0), 2),
                "speech": round(float(m.get("speech_sec") or 0.0), 1),
                "direct": bool(direct[j]),
                "source": mine[m["id"]],
                "fit": None if np.isnan(fit[j]) else round(float(fit[j]), 3),
                "own": round(float(own[j]), 3),
                "other": other_name[j],
                "other_sim": round(float(other_sim[j]), 3) if other_name[j] else None,
                "spans": int(m["id"]) not in no_spans,
                "rec": int(rec[j]),
            }
        )
    held = {"person": shown, "voices": voices, "legacy": bool(no_spans)}
    with _lock:
        _held[key] = held
        while len(_held) > 4:
            _held.popitem(last=False)
    return held


def _num(raw, default=None):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def clean_filters(raw: dict) -> dict:
    """What the page asks for, made safe."""
    raw = raw or {}

    def flag(k):
        return str(raw.get(k, "")).lower() in ("1", "true", "on", "yes")

    sort = raw.get("sort") if raw.get("sort") in SORTS else "fit"
    source = raw.get("source") if raw.get("source") in ("direct", "spread") else "all"
    spans = raw.get("spans") if raw.get("spans") in ("with", "without") else "all"
    return {
        "sort": sort,
        "source": source,
        "spans": spans,
        "closer": flag("closer"),
        "per_recording": flag("per_recording"),
        "fit_max": _num(raw.get("fit_max")),
        "fit_min": _num(raw.get("fit_min")),
        "date_from": str(raw.get("date_from") or "")[:10],
        "date_to": str(raw.get("date_to") or "")[:10],
        "folder": str(raw.get("folder") or "").strip().strip("/"),
        "file": str(raw.get("file") or "").strip().lower(),
        "label": str(raw.get("label") or "").strip().lower(),
        "min_speech": _num(raw.get("min_speech")),
        "max_speech": _num(raw.get("max_speech")),
    }


def _keep(v: dict, f: dict) -> bool:
    if f["source"] == "direct" and not v["direct"]:
        return False
    if f["source"] == "spread" and v["direct"]:
        return False
    if f["spans"] == "with" and not v["spans"]:
        return False
    if f["spans"] == "without" and v["spans"]:
        return False
    if f["closer"] and not v["other"]:
        return False
    fit = v["fit"]
    if f["fit_max"] is not None and (fit is None or fit > f["fit_max"]):
        return False
    if f["fit_min"] is not None and (fit is None or fit < f["fit_min"]):
        return False
    if f["date_from"] and (not v["date"] or v["date"] < f["date_from"]):
        return False
    if f["date_to"] and (not v["date"] or v["date"] > f["date_to"]):
        return False
    if f["folder"] and not v["display"].startswith(f["folder"]):
        return False
    if f["file"] and f["file"] not in v["file"].lower():
        return False
    if f["label"] and f["label"] not in v["label"].lower():
        return False
    if f["min_speech"] is not None and v["speech"] < f["min_speech"]:
        return False
    return not (f["max_speech"] is not None and v["speech"] > f["max_speech"])


def _sort_key(sort: str):
    if sort == "date_desc":
        return lambda v: (v["date"] or "", v["abs_path"], v["start"])
    if sort == "date_asc":
        return lambda v: (v["date"] or "9999", v["abs_path"], v["start"])
    if sort == "recording":
        return lambda v: (v["abs_path"], v["start"])
    if sort == "speech_desc":
        return lambda v: (-v["speech"], v["abs_path"], v["start"])
    # fit: nearer to someone else first, then the least fitting
    return lambda v: (
        0 if v["other"] else 1,
        v["fit"] if v["fit"] is not None else 2.0,
        v["abs_path"],
        v["start"],
    )


def page(person: str, raw: dict | None, offset: int = 0, limit: int = 100) -> dict:
    """A page of the person's voices, filtered and ordered."""
    t = table(person)
    f = clean_filters(raw or {})
    picked = [v for v in t["voices"] if _keep(v, f)]
    picked.sort(key=_sort_key(f["sort"]), reverse=f["sort"] == "date_desc")
    if f["per_recording"]:
        seen, one = set(), []
        for v in picked:
            if v["rec"] in seen:
                continue
            seen.add(v["rec"])
            one.append(v)
        picked = one
    offset = max(0, int(offset))
    limit = min(max(1, int(limit)), 500)
    allv = t["voices"]
    dates = sorted(v["date"] for v in allv if v["date"])
    return {
        "person": t["person"],
        "filters": f,
        "total": len(picked),
        "offset": offset,
        "voices": [
            {k: v[k] for k in v if k not in ("abs_path", "rec")}
            | {"url": f"/api/speaker/sample/{v['id']}"}
            for v in picked[offset : offset + limit]
        ],
        "counts": {
            "voices": len(allv),
            "direct": sum(1 for v in allv if v["direct"]),
            "spread": sum(1 for v in allv if not v["direct"]),
            "closer": sum(1 for v in allv if v["other"]),
            "without_spans": sum(1 for v in allv if not v["spans"]),
            "recordings": len({v["rec"] for v in allv}),
            "matching_recordings": len({v["rec"] for v in picked}),
        },
        # Voices without stored speech spans — the filter and the
        # hint only appear where there are such voices at all
        "legacy": t["legacy"],
        "first_date": dates[0] if dates else "",
        "last_date": dates[-1] if dates else "",
    }


ACTIONS = ("unname", "trash", "name")


def decide(person: str, ids, action: str, to: str = "") -> dict:
    """Remove the name, discard, or name as someone else — one entry of the
    history. Only voices that still carry the person's name are touched: a
    page shown before another decision must not act on what changed since."""
    from whisper_tm.database.speaker import fold_name, speaker_db

    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    to = (to or "").strip()
    if action == "name" and not to:
        raise ValueError("no name to give")
    if action == "name" and fold_name(to) == fold_name(person):
        raise ValueError("that is the same name")
    db = speaker_db()
    names = db.names_per_vector()
    want = fold_name(person)
    try:
        asked = {int(v) for v in ids}
    except (TypeError, ValueError):
        raise ValueError("invalid voice ids") from None
    hit = sorted(v for v in asked if v in names and fold_name(names[v]) == want)
    if not hit:
        return {"changed": 0, "skipped": len(asked), "decision": None}
    if action == "unname":
        done = db.write_names(
            [(v, None, None) for v in hit], kind="unname", origin="people", name=person
        )
    elif action == "trash":
        done = db.write_trash(hit, True, reason="people review", origin="people")
    else:
        done = db.write_names(
            [(v, to, "human") for v in hit], kind="name", origin="people", name=to
        )
    return {
        "changed": len(done),
        "skipped": len(asked) - len(hit),
        "decision": db.last_decision_id() if done else None,
    }

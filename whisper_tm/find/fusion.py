"""Find's fusion — the searches' hits as moments, copies folded.

Hits of one recording close in time are one moment (``NEAR_SEC``, a moment
spanning at most ``NEAR_SPAN_SEC``); a moment's score is the reciprocal ranks
of the searches that found it, more where several meet in it or elsewhere
in its recording, times the boosts (the file name has a remembered piece,
a named person speaks in it) and the kind of recording.

Copies of a recording (core/copies — the same video as .mp4, .mkv and
.webm, a backup) are ONE recording here: their hits meet on the timeline
they share, the moment is shown in the file of its best hit and the other
files are its suspected duplicates. What is left of the same passage in
other recordings (a clip cut out of a stream) is folded in as well, its
evidence staying its own.
"""

from __future__ import annotations

import logging
import os

from whisper_tm.core import copies

logger = logging.getLogger(__name__)

#: How much a kind of evidence counts in the fusion
WEIGHT = {"words": 1.3, "meaning": 1.0, "seen": 1.0, "heard": 0.8}
KINDS = tuple(WEIGHT)
#: The searches inside a scope (the remembered files, what the named people
#: say). Their hits are evidence of the kind that found them inside — a
#: passage Alice speaks found by meaning is meaning, not a second search
#: agreeing with the global meaning search (they overlap: counted as two,
#: Alice's passages would beat a word-for-word hit of the very phrase).
SCOPED = ("files", "people")
CONTENT = KINDS + SCOPED
RRF_K = 20
#: Hits of one recording this close count as one moment (a passage of the
#: meaning search is ~60 s) — and a moment spans no more than NEAR_SPAN_SEC:
#: chained further, two scenes 80 s apart become one moment, and the
#: language model judges only the first (e.g. one scene at 1:50 and another
#: at 3:10 of one video)
NEAR_SEC = 60.0
NEAR_SPAN_SEC = 90.0
#: Boosts of a moment's score
FILE_BOOST = 1.6
PEOPLE_BOOST = 1.35
#: The other kind of recording than the description SAYS (a video, a voice
#: message) — and than the plan only guessed
WRONG_MEDIA = 0.25
WRONG_GUESS = 0.85
#: Moments a recording keeps — its copies counted in
PER_RECORDING = 3
#: A list's best this many are compared for duplicates (the rest is never shown)
COPY_CANDIDATES = 120


def is_video(key: str) -> bool:
    from whisper_tm.media.indexer import is_video as _is_video

    return _is_video(key or "")


def local_still(key: str, t: float, moment=None) -> str | None:
    """A still of a moment on this disk, if one exists — Images & sounds' or
    a frame Find cut before (nothing is cut for a comparison)."""
    if moment:
        try:
            from whisper_tm.media.store import get_store

            row = get_store().moment(int(moment))
            if row and row.get("thumb") and os.path.isfile(row["thumb"]):
                return row["thumb"]
        except Exception:  # noqa: BLE001
            pass
    try:
        from whisper_tm.find.agent import frame_path

        p = frame_path(key, t)
        return str(p) if p.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def looks(db, keys) -> copies.Looks:
    return copies.Looks(db, keys, still=local_still)


def dup_of(it: dict, why: str, facts: dict) -> dict:
    """What a folded duplicate keeps: enough to show, play and open it."""
    size, length = facts.get(it["key"]) or (0, 0.0)
    return {
        "key": it["key"],
        "name": it.get("name") or os.path.basename(it["key"]),
        "rel": it.get("rel"),
        "dir": it.get("dir"),
        "t": it.get("t"),
        "end": it.get("end"),
        "video": it.get("video"),
        "thumb": it.get("thumb"),
        "why": why,
        "size": size,
        "length": round(length, 2),
    }


def unique_dups(it: dict) -> list[dict]:
    """An item's duplicates once each, never the item's own file."""
    out, seen = [], {it["key"]}
    for d in it.get("dups") or []:
        if d.get("key") and d["key"] not in seen:
            seen.add(d["key"])
            out.append(d)
    return out


def media_factor(plan: dict, video: bool) -> float:
    want = plan.get("media") or "any"
    if want == "any" or (want == "video") == bool(video):
        return 1.0
    return WRONG_MEDIA if plan.get("media_stated") else WRONG_GUESS


def score(engines: dict, elsewhere, boosts, video: bool, plan: dict) -> float:
    """A moment's score: reciprocal ranks, more for several searches meeting
    in it (or elsewhere in its recording), the boosts, the kind of recording."""
    s = sum(WEIGHT.get(n, 1.0) / (RRF_K + e["rank"]) for n, e in engines.items())
    s *= 1.0 + 0.6 * (len(engines) - 1)  # several searches meet here
    s *= 1.0 + 0.15 * len(set(elsewhere) - set(engines))  # …or elsewhere in the recording
    if "files" in boosts:
        s *= FILE_BOOST
    if "people" in boosts:
        s *= PEOPLE_BOOST
    s *= media_factor(plan, video)
    return round(s * 100, 3)


# ══════════════════════════════════════════════════════════════════════════
#  Copies of a recording
# ══════════════════════════════════════════════════════════════════════════


def copy_classes(hits: dict[str, list[dict]], lk: copies.Looks) -> dict[str, tuple[str, str]]:
    """Which recordings with hits are copies of one another (core/copies).

    ``hits``: each recording's hits, best first, the recordings best first.
    Returns ``{key: (the class's best key, why)}`` for every recording that
    is a copy of a better one.
    """
    keys = list(hits)
    facts = lk.facts
    rank = {k: i for i, k in enumerate(keys)}
    by_length = sorted((k for k in keys if (facts.get(k) or (0, 0))[1]), key=lambda k: facts[k][1])
    root = {k: k for k in keys}
    why: dict[str, str] = {}

    def find(k):
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for i, a in enumerate(by_length):
        for b in by_length[i + 1 :]:
            if not copies.same_length(facts[a], facts[b]):
                gap = max(copies.LENGTH_SEC, copies.LENGTH_SHARE * facts[b][1])
                if facts[b][1] - facts[a][1] > gap:
                    break
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            probes = [it["t"] for it in hits[a][:2]] + [it["t"] for it in hits[b][:2]]
            moments = {
                (it["key"], round(it["t"])): (it.get("extra") or {}).get("moment")
                for it in hits[a] + hits[b]
                if (it.get("extra") or {}).get("moment")
            }
            reason = lk.copy_reason(a, b, probes, moments)
            if reason:
                keep, drop = (ra, rb) if rank[ra] <= rank[rb] else (rb, ra)
                root[drop] = keep
                for k in (a, b, drop):
                    if k != keep:
                        why.setdefault(k, reason)
    return {k: (find(k), why.get(k, "the same length")) for k in keys if find(k) != k}


def fold_duplicates(
    items: list[dict], db, *, limit: int = COPY_CANDIDATES, lk: copies.Looks | None = None
) -> tuple[list[dict], dict]:
    """Fold suspected duplicates into the item they repeat.

    ``items`` best first. An item that looks like a better one in another
    file (:meth:`copies.Looks.moment_reason`) goes into that one's ``dups``;
    copies of copies join the same group, and a group never holds two
    moments of one recording. Returns ``(the items left, best first;
    {recording: the recording it copies} for the copies found here)`` — a
    recording is an item's ``rec`` (its class of copies) or its file.
    """
    head = [dict(it) for it in items[:limit]]
    if len(head) < 2 or db is None:
        return head, {}
    lk = lk or looks(db, [it["key"] for it in head])
    lk.add([it["key"] for it in head])
    n = len(head)
    root = list(range(n))
    recs = [{it.get("rec") or it["key"]} for it in head]
    joined: dict[int, tuple] = {}

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for j in range(1, n):
        for i in range(j):
            ri, rj = find(i), find(j)
            if ri == rj or recs[ri] & recs[rj]:
                continue
            got = lk.moment_reason(_with_moment(head[i]), _with_moment(head[j]))
            if got:
                root[rj] = ri
                recs[ri] |= recs[rj]
                joined[j] = (i, *got)
                break
    kept, by_root, same_rec = [], {}, {}
    for k, it in enumerate(head):
        if find(k) == k:
            it["dups"] = list(it.get("dups") or [])
            by_root[k] = it
            kept.append(it)
    for k, it in enumerate(head):
        r = find(k)
        if r == k:
            continue
        _i, why, copy = joined[k]
        group = by_root[r]
        group["dups"].append(dict(dup_of(it, why, lk.facts), copy=copy, _item=it))
        # what was folded into it before (the semantic search's copies) goes along
        group["dups"].extend(it.get("dups") or [])
        if copy:
            same_rec[it.get("rec") or it["key"]] = group.get("rec") or group["key"]
    for it in kept:
        it["dups"] = unique_dups(it)
    return kept, same_rec


def fold_engine(items: list[dict], db) -> tuple[list[dict], int]:
    """One search's hits with their suspected duplicates folded (shown in a
    tile): ``(the hits left, in rank order; how many are folded in)``."""
    if not items or db is None:
        return items, 0
    kept, _ = fold_duplicates(items, db, limit=len(items))
    folded = 0
    for it in kept:
        for d in it["dups"]:
            d.pop("_item", None)
            d.pop("copy", None)
        folded += len(it["dups"])
    return kept, folded


# ══════════════════════════════════════════════════════════════════════════
#  The moments
# ══════════════════════════════════════════════════════════════════════════


def fuse(engines: dict, plan: dict, db=None) -> list[dict]:
    """The content searches' hits as moments, best first: ``[{"key", "t",
    "score", "engines": {name: {"rank", "t", "text"}}, "dups", "rec", …}]``
    (module doc). Without ``db`` no copies are looked for."""
    files = (engines.get("files") or {}).get("paths") or set()
    people = (engines.get("people") or {}).get("paths") or set()
    by_key: dict[str, list] = {}
    for name in CONTENT:
        for it in (engines.get(name) or {}).get("items") or []:
            if not it.get("key"):
                continue
            if name in SCOPED:
                for kind in (it.get("extra") or {}).get("via") or []:
                    if kind in WEIGHT:
                        hit = dict(it, kind=kind, scope=name)
                        by_key.setdefault(it["key"], []).append((kind, hit))
            else:
                by_key.setdefault(it["key"], []).append((name, it))
    lk = looks(db, list(by_key)) if db is not None and by_key else None
    classes: dict[str, tuple[str, str]] = {}
    if lk is not None:
        # the recordings best first: a class of copies is named after its best
        ordered = sorted(
            by_key, key=lambda k: -max(WEIGHT[n] / (RRF_K + it["rank"]) for n, it in by_key[k])
        )
        try:
            hits = {
                k: sorted((it for _, it in by_key[k]), key=lambda it: it["rank"]) for k in ordered
            }
            classes = copy_classes(hits, lk)
        except Exception as e:  # noqa: BLE001 — each file its own recording then
            logger.warning("Find: copies not compared: %s", e)
    by_rec: dict[str, list] = {}
    for key, hits in by_key.items():
        by_rec.setdefault(classes.get(key, (key,))[0], []).extend(hits)
    members: dict[str, list[str]] = {}
    for key, (rec, _why) in classes.items():
        members.setdefault(rec, [rec]).append(key)
    moments = []
    for rec, hits in by_rec.items():
        hits.sort(key=lambda x: x[1]["t"])
        clusters: list[list] = []
        for h in hits:
            if (
                clusters
                and h[1]["t"] - clusters[-1][-1][1]["t"] <= NEAR_SEC
                and h[1]["t"] - clusters[-1][0][1]["t"] <= NEAR_SPAN_SEC
            ):
                clusters[-1].append(h)
            else:
                clusters.append([h])
        rec_engines = sorted({n for n, _ in hits})
        group = members.get(rec, [rec])
        place = {}  # each file's name and root, from its hits
        for _n, it in hits:
            place.setdefault(it["key"], it)
        boosts = [
            b
            for b, paths in (("files", files), ("people", people))
            if any(k in paths for k in group)
        ]
        for cl in clusters:
            best: dict[str, dict] = {}
            for n, it in cl:
                if n not in best or it["rank"] < best[n]["rank"]:
                    best[n] = it
            lead = min(
                best.values(), key=lambda it: -WEIGHT.get(it["kind"], 1.0) / (RRF_K + it["rank"])
            )
            key = lead["key"]
            video = is_video(key)
            media = next((it for it in best.values() if it.get("thumb") or it.get("clip")), None)
            first = place.get(key) or lead
            ev = {n: _evidence(n, it, key) for n, it in best.items()}
            copies_here = (
                [
                    dup_of(
                        dict(
                            place.get(k) or {},
                            key=k,
                            t=lead["t"],
                            end=lead.get("end"),
                            video=is_video(k),
                            thumb=frame_url(k, lead["t"]) if is_video(k) else None,
                        ),
                        classes[k][1] if k in classes else classes[key][1],
                        lk.facts,
                    )
                    for k in group
                    if k != key
                ]
                if lk is not None
                else []
            )
            carried = [d for it in best.values() for d in it.get("dups") or []]
            moments.append(
                {
                    "key": key,
                    "name": os.path.basename(key),
                    "rel": first.get("rel") or lead.get("rel"),
                    "dir": first.get("dir") or lead.get("dir"),
                    "video": video,
                    "t": lead["t"],
                    "end": lead.get("end"),
                    "rec": rec,
                    "score": score(ev, rec_engines, boosts, video, plan),
                    "engines": ev,
                    "elsewhere": rec_engines,
                    "boosts": boosts,
                    # a copy's still is this moment's picture too
                    "thumb": (
                        media.get("thumb")
                        if media
                        else frame_url(key, lead["t"])
                        if video
                        else None
                    ),
                    "clip": media.get("clip") if media else None,
                    "moment": (media.get("extra") or {}).get("moment") if media else None,
                    "verdict": None,
                    "reason": None,
                    "dups": unique_dups({"key": key, "dups": copies_here + carried}),
                }
            )
    moments.sort(key=lambda m: -m["score"])
    same_rec: dict[str, str] = {}
    if lk is not None:
        moments, same_rec = fold_duplicates(moments, db, lk=lk)
        for m in moments:
            for d in m["dups"]:
                it = d.pop("_item", None)
                if not d.pop("copy", False) or it is None:
                    continue
                # a copy's evidence is the moment's: the transcript of one file,
                # the stills of another
                for n, e in it["engines"].items():
                    if n not in m["engines"] or e["rank"] < m["engines"][n]["rank"]:
                        m["engines"][n] = dict(e, copy=d["name"])
                m["elsewhere"] = sorted(set(m["elsewhere"]) | set(it["elsewhere"]))
                m["boosts"] = [
                    b for b in ("files", "people") if b in m["boosts"] or b in it["boosts"]
                ]
                if not m.get("moment") and it.get("moment"):
                    m["moment"] = it["moment"]
            m["score"] = score(m["engines"], m["elsewhere"], m["boosts"], m["video"], plan)
        moments.sort(key=lambda m: -m["score"])
    # one recording's moments beyond its best three are rarely what is meant —
    # counted with its copies
    seen: dict[str, int] = {}
    out = []
    for m in moments:
        rec = m.get("rec") or m["key"]
        rec = same_rec.get(rec, rec)
        m["rec"] = rec
        seen[rec] = seen.get(rec, 0) + 1
        if seen[rec] <= PER_RECORDING:
            out.append(m)
    return out


def frame_url(key: str, t: float) -> str:
    from whisper_tm.find.agent import frame_url as _frame_url

    return _frame_url(key, t)


def _with_moment(it: dict) -> dict:
    """An item with its Images & sounds moment at the top (for its still)."""
    return dict(it, moment=it.get("moment") or (it.get("extra") or {}).get("moment"))


def _evidence(name: str, it: dict, key: str) -> dict:
    """What a moment keeps of a search's hit: its rank, time and text — for
    a hit found inside a scope that scope, for a copy's hit the copy's name."""
    ev = {"rank": it["rank"], "t": it["t"], "text": (it.get("text") or "")[:300]}
    if it.get("scope"):
        ev["scope"] = it["scope"]
    if it["key"] != key:
        ev["copy"] = os.path.basename(it["key"])
    return ev

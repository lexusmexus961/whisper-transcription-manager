"""Duplicates at a glance: what the relations table holds (analyze.py,
store.py), put together for the duplicates view of the search view's tab
"Same audio" and for Ask's ``find_duplicates``.

* **Groups**: recordings that are the same sound from start to end, linked
  by "same" in any number (a pair, or one sound effect in twenty folders).
  Each copy has its place against the others (``pos``, seconds): where its
  sound begins on the group's common timeline.
* **Wholes**: a recording with every recording it contains, each at the
  second it begins there (a stream and the clips cut from it).
* **Folders**: the folders that hold the same recordings. A copied folder
  keeps its layout: ``Backup/2024/Trip`` and ``Phone/2024/Trip`` end alike in
  ``2024/Trip``, so the pair is the folders above that end (``Backup`` and
  ``Phone``), however many subfolders were copied along. Two copies in one
  folder make a pair of that folder with itself.
* **Space**: what the extra copies take: in each group, every copy but the
  largest of its kind. A video and a sound file with its sound are both
  kept: the picture is not in the sound.

Every list and count can be narrowed (``Filters``): letters of a path, the
person speaking (the voice map's names), video or sound only, the format,
the length, the recording date (from the file name, else the file's own
date), the transcript's language, transcribed or not, the number of copies,
copies across folders or inside one; for recordings holding others, the
holding recording and the parts each by letters of their path — the clips of
one folder inside the streams of another.

The groups and their places are worked out from the tables at once and kept
until the relations change; the folders and the counts follow each filter.
"""

from __future__ import annotations

import base64
import bisect
import hashlib
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import accumulate

import numpy as np

from whisper_tm.config import VIDEO_EXTENSIONS

#: Worked out again after this long even with the relations unchanged (a
#: recording's size or loudness may have changed)
TTL_SEC = 120.0
#: While the analysis finds new pairs: worked out again at most this often
MIN_AGE_SEC = 15.0
#: How many recordings each folder holds (for "how much of it is a copy"):
#: counted anew at most this often
TOTALS_TTL_SEC = 60.0
GROUP_SORTS = ("space", "copies", "length", "new")
WHOLE_SORTS = ("parts", "length", "new")
#: Copies of one group handed out at most (the rest is only counted)
GROUP_FILES_MAX = 200
#: Waveforms in one request, and values per waveform, at most
LEVELS_MAX_IDS = 300
LEVELS_MAX_POINTS = 2000
logger = logging.getLogger(__name__)


@dataclass
class Filters:
    """What a list and its counts are narrowed to. A group fits when one of
    its copies has each asked property (not necessarily the same copy: they
    are one recording); a recording holding others fits by its own
    properties, the person by it or one of its parts."""

    q: str = ""  # letters of a path (any recording of a card)
    q_whole: str = ""  # the holding recording's path
    q_part: str = ""  # a part's path: the other parts are left out
    person: str = ""  # speaks in it (the voice map's names)
    media: str = ""  # "video" | "audio"
    ext: tuple = ()  # formats, without the dot
    min_sec: float = 0.0
    max_sec: float = 0.0
    since: str = ""  # YYYY-MM-DD: the recording date (name, else file)
    until: str = ""
    lang: str = ""  # the transcript's language
    transcribed: str = ""  # "yes" | "no"
    min_copies: int = 0
    spread: str = ""  # "across" folders | "within" one folder
    min_parts: int = 0
    pair: str = ""  # one folder pair of the overview
    file_id: int | None = None  # only the duplicates of one recording

    def norm(self) -> Filters:
        low = lambda x: " ".join(str(x or "").split()).lower()  # noqa: E731
        return Filters(
            q=low(self.q),
            q_whole=low(self.q_whole),
            q_part=low(self.q_part),
            person=" ".join(str(self.person or "").split()),
            media=self.media if self.media in ("video", "audio") else "",
            ext=tuple(sorted({e.strip().lstrip(".").lower() for e in self.ext if e and e.strip()})),
            min_sec=max(0.0, float(self.min_sec or 0)),
            max_sec=max(0.0, float(self.max_sec or 0)),
            since=str(self.since or "")[:10],
            until=str(self.until or "")[:10],
            lang=str(self.lang or "").lower(),
            transcribed=self.transcribed if self.transcribed in ("yes", "no") else "",
            min_copies=max(0, int(self.min_copies or 0)),
            spread=self.spread if self.spread in ("across", "within") else "",
            min_parts=max(0, int(self.min_parts or 0)),
            pair=str(self.pair or ""),
            file_id=self.file_id,
        )

    def recording(self) -> list:
        """The tests of one recording's own properties that are asked for."""
        tests = []
        if self.media:
            tests.append(lambda r: r["video"] == (self.media == "video"))
        if self.ext:
            tests.append(lambda r: r["ext"] in self.ext)
        if self.since:
            tests.append(lambda r: bool(r["day"]) and r["day"] >= self.since)
        if self.until:
            tests.append(lambda r: bool(r["day"]) and r["day"] <= self.until)
        if self.lang:
            tests.append(lambda r: r["lang"] == self.lang)
        if self.transcribed:
            tests.append(lambda r: r["transcribed"] == (self.transcribed == "yes"))
        return tests

    def speaks(self, r: dict) -> bool:
        return not self.person or self.person.lower() in r["people_lower"]


_lock = threading.Lock()
#: One build at a time: the page asks for the overview and a list at once
_building = threading.Lock()
_graphs: dict[str, _Graph] = {}
_totals: dict[str, _Totals] = {}


@dataclass
class _Group:
    ids: list[int]
    pos: dict[int, float]
    #: its "same" pairs: (a, b, folder pair id, the file under the pair's
    #: first folder, the shared end of the layout)
    rels: list[tuple[int, int, str, int, str]] = field(default_factory=list)
    length: float = 0.0
    extra: int = 0
    extra_bytes: int = 0
    largest: set[int] = field(default_factory=set)
    found_at: float = 0.0
    text: str = ""
    folders: int = 1  # in how many folders its copies lie


class _Graph:
    """The relations of one moment, put together."""

    def __init__(self, sig: tuple):
        self.sig = sig
        self.at = time.monotonic()
        self.files: dict[int, dict] = {}
        self.groups: list[_Group] = []
        self.group_of: dict[int, _Group] = {}
        #: whole → [(part, at, matched, score, found_at)], by ``at``
        self.wholes: dict[int, list[tuple]] = {}
        #: folder pair id → (folder, folder)
        self.pairs: dict[str, tuple[str, str]] = {}


# ── Folders ────────────────────────────────────────────────────────────────


def _folder(path: str) -> tuple[bool, list[str]]:
    """Whether ``path`` is absolute, and its folder's names, top first."""
    p = path.replace("\\", "/")
    return p.startswith("/"), [n for n in p.split("/")[:-1] if n]


def _join(absolute: bool, names: list[str]) -> str:
    s = "/".join(names)
    return "/" + s if absolute else s


def folder_pair(path_a: str, path_b: str) -> tuple[str, str, str, bool]:
    """``(folder, folder, end, swapped)``: the two folders a copy links, in
    order; the end of the layout they share below them ("" when none); and
    whether ``path_b`` lies under the first.

    ``/bak/x/2024/trip/a.mp4`` and ``/phone/2024/trip/a (1).mp4`` give
    ``("/bak/x", "/phone", "2024/trip", False)``.
    """
    abs_a, a = _folder(path_a)
    abs_b, b = _folder(path_b)
    if a == b and abs_a == abs_b:
        d = _join(abs_a, a)
        return d, d, "", False
    k = 0
    # Each keeps at least one name of its own
    while k < min(len(a), len(b)) - 1 and a[-1 - k] == b[-1 - k]:
        k += 1
    while True:
        ra, rb = _join(abs_a, a[: len(a) - k]), _join(abs_b, b[: len(b) - k])
        # ... and neither becomes the other's parent: "/m/Photos" and
        # "/m/Backup/Photos", not "/m" and "/m/Backup" (all of /m)
        if not k or not (_inside(ra, rb) or _inside(rb, ra)):
            break
        k -= 1
    end = "/".join(a[len(a) - k :]) if k else ""
    return (ra, rb, end, False) if ra <= rb else (rb, ra, end, True)


def _inside(folder: str, other: str) -> bool:
    """Whether ``other`` is ``folder`` or lies in it."""
    return other == folder or other.startswith(folder.rstrip("/") + "/")


def pair_id(a: str, b: str) -> str:
    return hashlib.sha1(f"{a}\0{b}".encode("utf-8", "surrogatepass")).hexdigest()[:12]


def common_part(a: str, b: str) -> tuple[str, str, str]:
    """``(common, rest of a, rest of b)``: the folder both lie in, and the
    rest of each below it."""
    abs_a, na = a.startswith("/"), [n for n in a.split("/") if n]
    nb = [n for n in b.split("/") if n]
    k = 0
    while k < min(len(na), len(nb)) and na[k] == nb[k]:
        k += 1
    return _join(abs_a, na[:k]), "/".join(na[k:]), "/".join(nb[k:])


# ── The graph ──────────────────────────────────────────────────────────────


def _video(path: str) -> bool:
    dot = path.rfind(".")
    return dot > path.rfind("/") and path[dot:].lower() in VIDEO_EXTENSIONS


def _signature(store) -> tuple:
    with store.connect() as c:
        return tuple(
            c.execute(
                "SELECT count(*), coalesce(max(found_at), 0), coalesce(sum(a), 0), "
                "coalesce(sum(b), 0) FROM relations"
            ).fetchone()
        )


def _build(store, sig: tuple) -> _Graph:
    g = _Graph(sig)
    with store.connect() as c:
        rel = c.execute(
            "SELECT a, b, kind, offset, matched, score, coalesce(found_at, 0) FROM relations"
        ).fetchall()
    g.files = store.files_by_id({r[0] for r in rel} | {r[1] for r in rel})
    files = g.files
    adj: dict[int, list[tuple[int, float]]] = {}
    same: list[tuple] = []
    for a, b, kind, off, matched, score, found in rel:
        if a not in files or b not in files:
            continue
        if kind == "same":
            same.append((a, b, found))
            adj.setdefault(a, []).append((b, float(off)))
            adj.setdefault(b, []).append((a, -float(off)))
        elif kind == "contains":
            g.wholes.setdefault(a, []).append((b, float(off), float(matched), int(score), found))
    for parts in g.wholes.values():
        parts.sort(key=lambda p: (p[1], p[0]))
    # The groups, each copy placed against the first by the offsets
    for start in sorted(adj):
        if start in g.group_of:
            continue
        pos = {start: 0.0}
        todo = [start]
        while todo:
            x = todo.pop()
            for y, off in adj[x]:
                if y not in pos:
                    pos[y] = pos[x] + off
                    todo.append(y)
        lo = min(pos.values())
        grp = _Group(ids=sorted(pos), pos={i: round(p - lo, 2) for i, p in pos.items()})
        for i in grp.ids:
            g.group_of[i] = grp
        g.groups.append(grp)
    for a, b, found in same:
        grp = g.group_of[a]
        fa, fb, end, swapped = folder_pair(files[a]["path"], files[b]["path"])
        pid = pair_id(fa, fb)
        g.pairs[pid] = (fa, fb)
        grp.rels.append((a, b, pid, b if swapped else a, end))
        grp.found_at = max(grp.found_at, found)
    for grp in g.groups:
        rows = [files[i] for i in grp.ids]
        grp.length = max(float(r.get("duration") or 0.0) for r in rows)
        grp.text = "\n".join(r["path"] for r in rows).lower()
        grp.folders = len({r["path"].replace("\\", "/").rsplit("/", 1)[0] for r in rows})
        # The space of the extra copies: all but the largest of each kind
        for video in (True, False):
            kind = [r for r in rows if _video(r["path"]) == video]
            if not kind:
                continue
            # (of the same size: the oldest, the likely original — then the first path)
            keep = min(
                kind,
                key=lambda r: (-int(r.get("size") or 0), float(r.get("mtime") or 0), r["path"]),
            )
            grp.largest.add(keep["id"])
            grp.extra += len(kind) - 1
            grp.extra_bytes += sum(int(r.get("size") or 0) for r in kind) - int(
                keep.get("size") or 0
            )
    _describe(g)
    return g


def _people_names(paths: list[str]) -> dict[str, set[str]]:
    """``{recording: {person, …}}`` for ``paths`` — the names the voice
    map gives their speakers (database/speaker.py ``map_names``)."""
    try:
        from whisper_tm.database.speaker import speaker_db

        return {p: set(m.values()) for p, m in speaker_db().map_names(paths).items()}
    except Exception as e:  # noqa: BLE001 — no voice library: no people
        logger.debug("Duplicates: no names from the voice library: %s", e)
        return {}


def _languages(paths: list[str]) -> dict[str, str]:
    """``{recording: language}`` of the transcribed ones among ``paths``."""
    out: dict[str, str] = {}
    try:
        from whisper_tm.extensions import db
    except Exception:  # noqa: BLE001
        return out
    if not db:
        return out
    try:
        with db.connection() as conn:
            for k in range(0, len(paths), 500):
                part = paths[k : k + 500]
                for path, lang in conn.execute(
                    f"SELECT abs_path, language FROM transcripts WHERE abs_path IN "
                    f"({','.join('?' * len(part))})",
                    part,
                ):
                    out[path] = str(lang or "").lower()
    except Exception as e:  # noqa: BLE001 — without the database: none known
        logger.debug("Duplicates: transcripts not readable: %s", e)
    return out


def _describe(g: _Graph) -> None:
    """What the filters ask of each recording: its format, its recording
    date (from the name — a copy's file date is often only when it was
    copied — else the file's own), its transcript's language, the people
    speaking in it."""
    from whisper_tm.core.voice_extract import date_from_name

    paths = [r["path"] for r in g.files.values()]
    langs = _languages(paths)
    people = _people_names(paths)
    # A "date" in a name that lies ahead is no recording date (a number)
    tomorrow = time.strftime("%Y-%m-%d", time.localtime(time.time() + 86400))
    for r in g.files.values():
        path = r["path"]
        dot = path.rfind(".")
        r["ext"] = path[dot + 1 :].lower() if dot > path.replace("\\", "/").rfind("/") else ""
        r["video"] = _video(path)
        day = date_from_name(path)
        if day > tomorrow:
            day = ""
        if not day and r.get("mtime"):
            try:
                day = time.strftime("%Y-%m-%d", time.localtime(float(r["mtime"])))
            except (OverflowError, OSError, ValueError):
                day = ""
        r["day"] = day
        r["transcribed"] = path in langs
        r["lang"] = langs.get(path, "")
        r["people"] = sorted(people.get(path, ()))
        r["people_lower"] = {n.lower() for n in r["people"]}


def graph(store, fresh: bool = False) -> _Graph:
    """The relations put together: kept while they stay the same."""
    key = str(store.path)
    asked = time.monotonic()
    sig = _signature(store)

    def usable(g):
        if g is None:
            return False
        if fresh:
            return g.at >= asked  # built since it was asked for
        age = time.monotonic() - g.at
        return (g.sig == sig and age < TTL_SEC) or (age < MIN_AGE_SEC and sig[0] >= g.sig[0])

    with _lock:
        g = _graphs.get(key)
    if usable(g):
        return g
    with _building:
        with _lock:
            g = _graphs.get(key)
        if usable(g):  # another request built it meanwhile
            return g
        g = _build(store, sig)
        with _lock:
            _graphs[key] = g
    return g


def forget() -> None:
    with _lock:
        _graphs.clear()
        _totals.clear()


# ── What the page asks ─────────────────────────────────────────────────────


def _row(r: dict) -> dict:
    return {
        "id": r["id"],
        "path": r["path"],
        "size": int(r.get("size") or 0),
        "duration": r.get("duration"),
        "lufs": r.get("lufs"),
        "video": r["video"],
        "ext": r["ext"],
        "day": r["day"],
        "lang": r["lang"],
        "transcribed": r["transcribed"],
        "people": r["people"],
    }


def _filters(flt: Filters | None, kw: dict) -> Filters:
    return (flt or Filters(**kw)).norm()


def _groups(g: _Graph, flt: Filters) -> list[_Group]:
    tests = flt.recording()
    out = []
    for grp in g.groups:
        if flt.file_id is not None and flt.file_id not in grp.pos:
            continue
        if flt.min_sec and grp.length < flt.min_sec:
            continue
        if flt.max_sec and grp.length > flt.max_sec:
            continue
        if flt.min_copies and len(grp.ids) < flt.min_copies:
            continue
        if flt.spread == "across" and grp.folders < 2:
            continue
        if flt.spread == "within" and grp.folders > 1:
            continue
        if flt.q and flt.q not in grp.text:
            continue
        if flt.pair and not any(r[2] == flt.pair for r in grp.rels):
            continue
        rows = [g.files[i] for i in grp.ids]
        if not all(any(t(r) for r in rows) for t in tests):
            continue
        if flt.person and not any(flt.speaks(r) for r in rows):
            continue
        out.append(grp)
    return out


def _wholes(g: _Graph, flt: Filters) -> list[tuple[int, list[tuple]]]:
    """The recordings holding others that fit, each with the parts that do."""
    tests = flt.recording()
    out = []
    for w, parts in g.wholes.items():
        whole = g.files[w]
        path = whole["path"].lower()
        if flt.file_id is not None and w != flt.file_id and all(p[0] != flt.file_id for p in parts):
            continue
        dur = float(whole.get("duration") or 0.0)
        if (flt.min_sec and dur < flt.min_sec) or (flt.max_sec and dur > flt.max_sec):
            continue
        if flt.q_whole and flt.q_whole not in path:
            continue
        if flt.q_part:
            parts = [p for p in parts if flt.q_part in g.files[p[0]]["path"].lower()]
            if not parts:
                continue
        if (
            flt.q
            and flt.q not in path
            and all(flt.q not in g.files[p[0]]["path"].lower() for p in parts)
        ):
            continue
        if not all(t(whole) for t in tests):
            continue
        if (
            flt.person
            and not flt.speaks(whole)
            and not any(flt.speaks(g.files[p[0]]) for p in parts)
        ):
            continue
        if flt.min_parts and len(parts) < flt.min_parts:
            continue
        out.append((w, parts))
    return out


def _facets(g: _Graph) -> dict:
    """What the filters can choose from, with how many groups or recordings
    holding others each has: the people, the formats, the languages, the
    span of the recording dates."""
    people: Counter = Counter()
    exts: Counter = Counter()
    langs: Counter = Counter()
    days: list[str] = []
    cards = [[g.files[i] for i in grp.ids] for grp in g.groups]
    cards += [[g.files[w], *(g.files[p[0]] for p in parts)] for w, parts in g.wholes.items()]
    for rows in cards:
        people.update({n for r in rows for n in r["people"]})
        exts.update({r["ext"] for r in rows if r["ext"]})
        langs.update({r["lang"] for r in rows if r["lang"]})
        days.extend(r["day"] for r in rows if r["day"])
    return {
        "people": people.most_common(),
        "ext": exts.most_common(),
        "lang": langs.most_common(),
        "days": [min(days), max(days)] if days else [],
    }


class _Totals:
    """The analyzed recordings of every folder, counted in one pass."""

    def __init__(self, store):
        self.at = time.monotonic()
        direct: Counter = Counter()
        with store.connect() as c:
            for (path,) in c.execute("SELECT path FROM files WHERE status='done'"):
                absolute, names = _folder(path)
                direct[_join(absolute, names)] += 1
        self.direct = direct
        self.folders = sorted(direct)
        self.cum = [0, *accumulate(direct[f] for f in self.folders)]

    def under(self, folder: str, *, only_here: bool = False, but: str = "") -> int:
        """Recordings in ``folder`` and below it (``only_here``: in it alone;
        ``but``: not those of that folder inside it)."""
        if only_here:
            return self.direct.get(folder, 0)
        lo = folder.rstrip("/") + "/"
        i = bisect.bisect_left(self.folders, lo)
        j = bisect.bisect_left(self.folders, lo[:-1] + "0")  # the next name after "folder/…"
        n = self.cum[j] - self.cum[i] + (self.direct.get(folder, 0) if folder != lo else 0)
        if but and but.startswith(lo):
            n -= self.under(but)
        return n


def _folder_totals(store) -> _Totals:
    key = str(store.path)
    with _lock:
        t = _totals.get(key)
    if t is None or time.monotonic() - t.at >= TOTALS_TTL_SEC:
        t = _Totals(store)
        with _lock:
            _totals[key] = t
    return t


def _folders(store, g: _Graph, groups: list[_Group], top: int) -> tuple[list[dict], int]:
    """The folder pairs of ``groups``, the most shared first: ``top`` of
    them, and how many there are."""
    acc: dict[str, dict] = {}
    for grp in groups:
        for a, b, pid, first, end in grp.rels:
            s = acc.get(pid)
            if s is None:
                s = acc[pid] = {"groups": set(), "a": set(), "b": set(), "ends": set()}
            s["groups"].add(id(grp))
            second = b if first == a else a
            s["a"].add(first)
            s["b"].add(second)
            if end:
                s["ends"].add(end)
    ranked = sorted(
        acc.items(),
        key=lambda kv: (-len(kv[1]["groups"]), -len(kv[1]["a"] | kv[1]["b"]), g.pairs[kv[0]]),
    )
    totals = _folder_totals(store) if ranked else None
    out = []
    for pid, s in ranked[:top]:
        fa, fb = g.pairs[pid]
        within = fa == fb
        ids_a = s["a"] | s["b"] if within else s["a"]
        ids_b = ids_a if within else s["b"]
        common, rest_a, rest_b = common_part(fa, fb)
        # (counted up to a minute ago — never fewer than the copies found since)
        total_a = max(len(ids_a), totals.under(fa, only_here=within, but="" if within else fb))
        total_b = 0 if within else max(len(ids_b), totals.under(fb, but=fa))
        out.append(
            {
                "id": pid,
                "a": fa,
                "b": fb,
                "within": within,
                "common": common,
                "a_rest": rest_a,
                "b_rest": rest_b,
                "groups": len(s["groups"]),
                "a_files": len(ids_a),
                "b_files": len(ids_b),
                "a_bytes": sum(int(g.files[i].get("size") or 0) for i in ids_a),
                "b_bytes": sum(int(g.files[i].get("size") or 0) for i in ids_b),
                "a_total": total_a,
                "b_total": total_b,
                "layout": len(s["ends"]),
            }
        )
    return out, len(acc)


def overview(
    store, flt: Filters | None = None, *, top: int = 30, fresh: bool = False, **kw
) -> dict:
    """The tiles and the folders of the duplicates view, and what its
    filters can choose from (``facets``)."""
    flt = _filters(flt, kw)
    flt.pair = ""  # the tiles count every folder pair
    g = graph(store, fresh)
    groups = _groups(g, flt)
    wholes = _wholes(g, flt)
    folders, n_folders = _folders(store, g, groups, max(1, min(200, int(top))))
    with store.connect() as c:
        analyzed = c.execute("SELECT count(*) FROM files WHERE status='done'").fetchone()[0]
    return {
        "same": {
            "groups": len(groups),
            "files": sum(len(x.ids) for x in groups),
            "extra": sum(x.extra for x in groups),
            "extra_bytes": sum(x.extra_bytes for x in groups),
        },
        "contains": {
            "wholes": len(wholes),
            "parts": len({p[0] for _w, ps in wholes for p in ps}),
            "pairs": sum(len(ps) for _w, ps in wholes),
        },
        "folders": folders,
        "folders_total": n_folders,
        "analyzed": analyzed,
        "facets": _facets(g),
    }


def group_page(
    store,
    flt: Filters | None = None,
    *,
    sort: str = "space",
    offset: int = 0,
    limit: int = 50,
    **kw,
) -> dict:
    """A page of the groups: each copy with its place, size and kind."""
    g = graph(store)
    groups = _groups(g, _filters(flt, kw))
    key = {
        "space": lambda x: (-x.extra_bytes, -x.length, x.ids[0]),
        "copies": lambda x: (-len(x.ids), -x.extra_bytes, x.ids[0]),
        "length": lambda x: (-x.length, -len(x.ids), x.ids[0]),
        "new": lambda x: (-x.found_at, x.ids[0]),
    }[sort if sort in GROUP_SORTS else "space"]
    groups.sort(key=key)
    page = []
    for grp in groups[offset : offset + limit]:
        # Videos first, each kind's kept copy first, then the largest
        rows = sorted(
            (g.files[i] for i in grp.ids),
            key=lambda r: (
                not r["video"],
                r["id"] not in grp.largest,
                -int(r.get("size") or 0),
                r["path"],
            ),
        )
        page.append(
            {
                "files": [
                    {**_row(r), "pos": grp.pos[r["id"]], "largest": r["id"] in grp.largest}
                    for r in rows[:GROUP_FILES_MAX]
                ],
                "count": len(grp.ids),
                "length": grp.length,
                "extra": grp.extra,
                "extra_bytes": grp.extra_bytes,
                "found_at": grp.found_at,
            }
        )
    return {"kind": "same", "groups": page, "total": len(groups), "offset": offset, "sort": sort}


def whole_page(
    store,
    flt: Filters | None = None,
    *,
    sort: str = "parts",
    offset: int = 0,
    limit: int = 50,
    **kw,
) -> dict:
    """A page of the recordings that contain others, each with its parts
    (those that fit the filters; ``parts_all``: how many it holds in all)."""
    g = graph(store)
    wholes = _wholes(g, _filters(flt, kw))

    def dur(w):
        return float(g.files[w].get("duration") or 0.0)

    key = {
        "parts": lambda x: (-len(x[1]), -dur(x[0]), x[0]),
        "length": lambda x: (-dur(x[0]), -len(x[1]), x[0]),
        "new": lambda x: (-max(p[4] for p in x[1]), x[0]),
    }[sort if sort in WHOLE_SORTS else "parts"]
    wholes.sort(key=key)
    page = []
    for w, parts in wholes[offset : offset + limit]:
        # How much of it the parts cover (overlaps counted once)
        covered, reach = 0.0, 0.0
        for _p, at, matched, _s, _f in parts:
            lo, hi = max(at, reach), at + matched
            if hi > lo:
                covered += hi - lo
            reach = max(reach, hi)
        page.append(
            {
                "whole": _row(g.files[w]),
                "parts": [
                    {**_row(g.files[p]), "at": at, "matched": matched, "score": score}
                    for p, at, matched, score, _f in parts
                ],
                "parts_all": len(g.wholes[w]),
                "covered": round(covered, 1),
            }
        )
    return {
        "kind": "contains",
        "wholes": page,
        "total": len(wholes),
        "offset": offset,
        "sort": sort,
    }


def levels(store, ids, n: int) -> dict[int, str]:
    """The stored waveforms of ``ids``, each in at most ``n`` values (the
    loudest of each stretch; 0…255 over -60…0 dBFS), base64."""
    n = max(16, min(LEVELS_MAX_POINTS, int(n)))
    want = list(dict.fromkeys(int(i) for i in ids))[:LEVELS_MAX_IDS]
    out: dict[int, str] = {}
    with store.connect() as c:
        for k in range(0, len(want), 500):
            chunk = want[k : k + 500]
            for fid, blob in c.execute(
                "SELECT file_id, levels FROM waveforms WHERE file_id IN "
                f"({','.join('?' * len(chunk))})",
                chunk,
            ):
                v = np.frombuffer(bytes(blob), np.uint8)
                if len(v) > n:
                    edges = np.floor(np.arange(n) * (len(v) / n)).astype(np.int64)
                    v = np.maximum.reduceat(v, edges)
                out[int(fid)] = base64.b64encode(v.tobytes()).decode("ascii")
    return out

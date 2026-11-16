"""One recording analyzed, and a snippet searched.

**Analyzing** reads the sound once — from the recording through ffmpeg (16
kHz mono, the video read along and thrown away: core/transcription.py's
single-pass explains why that is faster on a network share), or from the
16 kHz WAV a transcription has made anyway — and feeds it, ten seconds at
a time, to the fingerprints (fingerprint.py) and the levels (levels.py).

**Relations** are found right then: the new recording, in windows of 10 s
back to back (at most 600 of them, spread over a longer one), is searched
in the index before its own fingerprints go in — each window with 32 of its
hashes: at a billion entries a window's every hash would read tens of
thousands of them. Which 32 is a fixed scramble of the hash values, never
the rarest: a recording that exists three times makes its own hashes
three times as common, and "rarest first" picked exactly the hashes its
copies did NOT share. Hashes nothing else has are skipped (they can't
match), and the commonest ones too. A recording found with nearly all
windows at one offset and the same length is the **same**; one that holds
all windows of the other is said to **contain** it. Each pair is found
once — when the second of the two is analyzed.

**Searching** a snippet: its landmarks (fingerprint.query_landmarks) are
looked up; every hit gives a recording and an offset (stored frame minus
the snippet's frame). The score of a recording at an offset is the number
of different snippet hashes found there, give or take a frame (see
fingerprint.py for why different ones). A recording may hold the snippet
more than once — each place counts on its own.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from whisper_tm.audio import fingerprint, index
from whisper_tm.audio.fingerprint import FPS, SR
from whisper_tm.audio.levels import Levels
from whisper_tm.core import media_formats

logger = logging.getLogger(__name__)

#: A recording at an offset is a match from this score on (different
#: snippet hashes): time-reversed snippets — which can't be in the index —
#: reached 5 at most beside several hundred hours of sound; real copies of
#: 3 s, 15 and more
MIN_SCORE = 10
#: ... and a sure one from here
STRONG_SCORE = 25
#: A hash with more entries than this says next to nothing — skipped
MAX_BUCKET = 200_000
#: Entries one lookup reads at most: past it the commonest hashes are left
#: out first (a minute of snippet against a billion entries would read tens
#: of millions)
MAX_ENTRIES = 4_000_000
#: Chance agreements grow with the snippet: the score needed is at least
#: this share of its different hashes (10 for up to ~1,700 of them, 10 s at
#: query density; 18 for a five-minute snippet)
MIN_SHARE = 0.006
#: Snippets longer than this are fingerprinted as the index is (sparser,
#: unshifted) — plenty for a match, a fraction of the lookups
DENSE_MAX_SEC = 30.0
#: A snippet hash is taken at most this often (a held tone repeats it)
MAX_TIMES = 8
#: The longest snippet searched
QUERY_MAX_SEC = 300.0
#: Places of the same snippet in one recording at least this far apart
_APART_SEC = 2.0
#: Candidates checked exactly (by their upper bound)
_MAX_CANDIDATES = 20_000
#: Relations: windows of the new recording (back to back, at most
#: _WIN_MAX of them), each searched with _WIN_HASHES of its hashes — none
#: with more entries than _WIN_MAX_BUCKET
_WIN_SEC = 10.0
_WIN_MAX = 600
_WIN_HASHES = 32
_WIN_MAX_BUCKET = 20_000
#: A window counts as found at an offset with this many different hashes
_WIN_FOUND = 4
#: Share of the windows found for "same" / "contains"
_REL_SHARE = 0.8
#: Lengths this close count as the same (seconds, or a share)
_SAME_LEN_SEC, _SAME_LEN_SHARE = 2.0, 0.02
_CHUNK_SAMPLES = SR * 10


class AnalysisError(Exception):
    """The sound of a recording can't be read."""


# ── Reading sound ────────────────────────────────────────────────────────


def _low_priority(cmd: list[str]) -> list[str]:
    """A command that yields the CPU (and the disk, where ionice works) to
    everything else: the queue beside it is what the machine is for."""
    if shutil.which("ionice"):
        cmd = ["ionice", "-c", "3", *cmd]
    if shutil.which("nice"):
        cmd = ["nice", "-n", "19", *cmd]
    return cmd


def decode(
    path: str | Path,
    *,
    start: float | None = None,
    length: float | None = None,
    stop=None,
    pace=None,
    low_priority: bool = True,
):
    """The sound of ``path`` as int16 at 16 kHz mono, ten seconds at a
    time. ``stop()`` true ends it (InterruptedError); ``pace(seconds)`` is
    called after every piece with the seconds read so far (and may sleep).
    Raises AnalysisError when ffmpeg can't read any sound."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += media_formats.source(path)
    if length is not None:
        cmd += ["-t", f"{max(0.0, length):.3f}"]
    cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
    if start is None and length is None:
        # The video read along, thrown away: one sequential read instead of
        # a seek between every two sound packets
        cmd += ["-map", "0:v:0?", "-c", "copy", "-f", "null", "-"]
    if low_priority:
        cmd = _low_priority(cmd)
    got = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        rest = b""
        try:
            while True:
                if stop is not None and stop():
                    proc.kill()
                    raise InterruptedError("stopped")
                data = proc.stdout.read(_CHUNK_SAMPLES * 2)
                if not data:
                    break
                data = rest + data
                whole = len(data) // 2 * 2
                rest = data[whole:]
                if whole:
                    x = np.frombuffer(data[:whole], dtype="<i2")
                    got += len(x)
                    yield x
                    if pace is not None:
                        pace(got / SR)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait()
        if proc.returncode not in (0, None) and not got:
            err.seek(0)
            text = err.read().decode("utf-8", "replace").strip()
            # A video without a sound track: ffmpeg's last line ("Error
            # opening output files: Invalid argument") says nothing of it
            if "matches no streams" in text:
                raise AnalysisError("no sound track")
            msg = text.splitlines()
            raise AnalysisError((msg[-1] if msg else "") or "ffmpeg could not read the sound")
        if not got:
            raise AnalysisError("no sound")


def read_wav(fh, *, stop=None):
    """The samples of a 16 kHz mono 16-bit WAV (RIFF or RF64, as the
    transcription writes it) from an open file — ten seconds at a time.
    Raises AnalysisError for any other kind of WAV."""
    head = fh.read(12)
    if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
        raise AnalysisError("not a WAV file")
    fmt_ok = False
    while True:
        ck = fh.read(8)
        if len(ck) < 8:
            raise AnalysisError("WAV without data")
        cid, size = ck[:4], struct.unpack("<I", ck[4:])[0]
        if cid == b"fmt ":
            body = fh.read(size + (size & 1))
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt_ok = tag == 1 and channels == 1 and rate == SR and bits == 16
        elif cid == b"data":
            if not fmt_ok:
                raise AnalysisError("not 16 kHz mono 16-bit")
            break
        else:
            fh.seek(size + (size & 1), os.SEEK_CUR)
    # The data runs to the end of the file (RF64 and unfinished RIFF headers
    # carry no usable size)
    rest = b""
    while True:
        if stop is not None and stop():
            raise InterruptedError("stopped")
        data = fh.read(_CHUNK_SAMPLES * 2)
        if not data:
            break
        data = rest + data
        whole = len(data) // 2 * 2
        rest = data[whole:]
        if whole:
            yield np.frombuffer(data[:whole], dtype="<i2")


# ── Analyzing ────────────────────────────────────────────────────────────


def measure(pieces) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fingerprints and levels of sound given in pieces (int16): the levels'
    figures (levels.Levels.finish) and the hashes with their frames."""
    fp = fingerprint.Fingerprinter(fingerprint.STORE)
    lv = Levels()
    for x in pieces:
        fp.feed(x)
        lv.feed(x)
    h, t = fp.finish()
    return lv.finish(), h, t


def _sorted_query(h: np.ndarray, t: np.ndarray):
    """Query hashes sorted, each kept at most MAX_TIMES times; the unique
    ones with where they start and how many there are."""
    order = np.lexsort((t, h))
    h, t = h[order], t[order]
    uh, first = np.unique(h, return_index=True)
    rank = np.arange(len(h)) - np.repeat(first, np.diff(np.append(first, len(h))))
    keep = rank < MAX_TIMES
    h, t = h[keep], t[keep]
    uh, first, count = np.unique(h, return_index=True, return_counts=True)
    return h, t, uh, first, count


def _matches(snap, h: np.ndarray, t: np.ndarray, exclude_ids=()):
    """Every hit of the query: recording id, offset (frames), index of the
    query hash (into its unique hashes), query frame — sorted by
    (recording, offset)."""
    h, t, uh, first, count = _sorted_query(np.asarray(h, np.uint32), np.asarray(t, np.int64))
    sizes = snap.bucket_sizes(uh)
    if int(sizes.sum()) > MAX_ENTRIES:
        # The commonest hashes go first until the lookup fits
        order = np.argsort(sizes, kind="stable")
        fits = np.cumsum(sizes[order]) <= MAX_ENTRIES
        keep = np.zeros(len(uh), bool)
        keep[order[fits]] = True
        sel = np.flatnonzero(keep)
        uh, first, count = uh[sel], first[sel], count[sel]
    entries, which = snap.lookup(uh, MAX_BUCKET)
    if not len(entries):
        z = np.zeros(0, np.int64)
        return z, z, z, z
    ids, frames = index.unpack(entries)
    if len(exclude_ids):
        keep = ~np.isin(ids, np.asarray(list(exclude_ids), np.int64))
        ids, frames, which = ids[keep], frames[keep], which[keep]
    n = count[which]
    rep = np.repeat(np.arange(len(ids)), n)
    k = np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)
    tq = t[first[which][rep] + k]
    rid, off, hidx = ids[rep], frames[rep] - tq, which[rep]
    key = (rid << 27) | (off + (1 << 26))
    order = np.argsort(key, kind="stable")
    return key[order], hidx[order], tq[order], rid[order]


def _places(key, hidx, tq, *, min_score: int, apart_frames: int):
    """The places a query was found: ``{rid: [(score, offset, q0, q1), …]}``
    best first — the score of a place is the number of different query
    hashes at its offset ±1 frame."""
    if not len(key):
        return {}
    ukey, kstart, kcount = np.unique(key, return_index=True, return_counts=True)
    kinv = np.repeat(np.arange(len(ukey), dtype=np.int64), kcount)
    pairs = np.unique((kinv << 21) | hidx)
    pk = pairs >> 21
    d0 = np.bincount(pk, minlength=len(ukey))
    n = len(ukey)
    has_p = np.zeros(n, bool)
    has_n = np.zeros(n, bool)
    has_p[1:] = ukey[1:] == ukey[:-1] + 1
    has_n[:-1] = ukey[:-1] + 1 == ukey[1:]
    bound = d0.copy()
    bound[has_p] += d0[np.flatnonzero(has_p) - 1]
    bound[has_n] += d0[np.flatnonzero(has_n) + 1]
    cand = np.flatnonzero(bound >= min_score)
    if not len(cand):
        return {}
    cand = cand[np.argsort(-bound[cand], kind="stable")][:_MAX_CANDIDATES]
    p_lo = np.searchsorted(pk, np.arange(n))
    p_hi = np.searchsorted(pk, np.arange(n), side="right")
    found: dict[int, list] = {}
    for j in cand:
        lo = j - 1 if has_p[j] else j
        hi = j + 1 if has_n[j] else j
        score = int(len(np.unique(pairs[p_lo[lo] : p_hi[hi]] & ((1 << 21) - 1))))
        if score < min_score:
            continue
        rid = int(ukey[j] >> 27)
        off = int(ukey[j] & ((1 << 27) - 1)) - (1 << 26)
        places = found.setdefault(rid, [])
        if any(abs(off - p[1]) < apart_frames for p in places):
            continue
        part = tq[kstart[lo] : kstart[hi] + kcount[hi]]
        places.append((score, off, int(part.min()), int(part.max())))
    for places in found.values():
        places.sort(key=lambda p: -p[0])
    return found


def search(
    snap,
    h: np.ndarray,
    t: np.ndarray,
    *,
    min_score: int = MIN_SCORE,
    exclude_ids=(),
    limit: int = 500,
) -> list[dict]:
    """Recordings holding the query: ``[{"id", "score", "at", "q_start",
    "q_end", "places": [{"at", "score"}, …]}]`` best first. ``at`` is
    where the query's start lies in the recording (seconds; negative when
    the recording starts inside the query)."""
    if len(h):
        min_score = max(min_score, int(np.ceil(MIN_SHARE * len(np.unique(h)))))
    key, hidx, tq, _ = _matches(snap, h, t, exclude_ids)
    found = _places(key, hidx, tq, min_score=min_score, apart_frames=int(_APART_SEC * FPS))
    # The query's length: its last frame, and the pair that starts there
    q_len = (int(np.max(t)) + fingerprint.DT_MAX) / FPS if len(t) else 0.0
    hits = []
    for rid, places in found.items():
        best = places[0]
        hits.append(
            {
                "id": rid,
                "score": best[0],
                "at": round(best[1] / FPS, 2),
                "q_start": round(best[2] / FPS, 2),
                "q_end": round(min(q_len, (best[3] + fingerprint.DT_MAX) / FPS), 2),
                "places": [{"at": round(p[1] / FPS, 2), "score": p[0]} for p in places],
            }
        )
    hits.sort(key=lambda x: (-x["score"], x["id"]))
    return hits[:limit]


def relations(snap, h: np.ndarray, t: np.ndarray, duration: float, durations) -> list[dict]:
    """Recordings in the index that are the same as this one, contain it,
    or are contained in it (see above). ``durations(ids)`` gives
    ``{id: seconds}`` of the recordings found."""
    if not len(h) or duration <= 0:
        return []
    frames = int(duration * FPS)
    win_fr = int(_WIN_SEC * FPS)
    every_fr = int(max(_WIN_SEC, duration / _WIN_MAX) * FPS)
    # Whole windows only (a recording shorter than one: itself)
    starts = np.arange(0, max(1, frames - win_fr + 1), every_fr)
    h, t = np.asarray(h, np.uint32), np.asarray(t, np.int64)
    w = t // every_fr
    inside = (w < len(starts)) & (t - w * every_fr < win_fr)
    h, t, w = h[inside], t[inside], w[inside]
    # Of each window's hashes that some other recording has at all (and not
    # too many): a fixed scramble's first ones
    sizes = snap.bucket_sizes(h)
    have = (sizes > 0) & (sizes <= _WIN_MAX_BUCKET)
    h, t, w = h[have], t[have], w[have]
    if not len(h):
        return []
    scramble = (h.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    order = np.lexsort((t, scramble, w))
    h, t, w = h[order], t[order], w[order]
    first = np.searchsorted(w, w, side="left")
    keep = (np.arange(len(w)) - first) < _WIN_HASHES
    key, hidx, tq, _ = _matches(snap, h[keep], t[keep])
    found = _places(key, hidx, tq, min_score=2 * _WIN_FOUND, apart_frames=int(_APART_SEC * FPS))
    if not found:
        return []
    lengths = durations(list(found))
    total_w = len(starts)
    out = []
    for other, places in found.items():
        score, off, _q0, _q1 = places[0]
        dur_o = float(lengths.get(other) or 0.0)
        if dur_o <= 0:
            continue
        # The windows found at this offset (``key`` is sorted)
        lo = np.searchsorted(key, (other << 27) | (off - 1 + (1 << 26)))
        hi = np.searchsorted(key, (other << 27) | (off + 1 + (1 << 26)), side="right")
        pairs = np.unique(np.stack([tq[lo:hi] // every_fr, hidx[lo:hi]]), axis=1)
        found_w = int(np.count_nonzero(np.bincount(pairs[0]) >= _WIN_FOUND))
        if not found_w:
            continue
        # ... and those that lie inside the other recording there
        in_other = ((starts + off) >= -FPS) & ((starts + off + win_fr) <= dur_o * FPS + FPS)
        expected = int(np.count_nonzero(in_other))
        shift = off / FPS  # where this recording starts in the other
        same_len = abs(duration - dur_o) <= max(
            _SAME_LEN_SEC, _SAME_LEN_SHARE * max(duration, dur_o)
        )
        whole = found_w >= max(1, _REL_SHARE * total_w)
        if whole and same_len and abs(shift) <= _SAME_LEN_SEC:
            out.append(
                {
                    "other": other,
                    "kind": "same",
                    "offset": round(shift, 2),
                    "matched": round(min(duration, dur_o), 1),
                    "score": score,
                    "new_contains": False,
                }
            )
        elif whole and dur_o > duration + _SAME_LEN_SEC:
            out.append(
                {
                    "other": other,
                    "kind": "contains",
                    "offset": round(shift, 2),
                    "matched": round(duration, 1),
                    "score": score,
                    "new_contains": False,
                }
            )
        elif (
            expected >= (1 if dur_o < 2.5 * _WIN_SEC else 2)
            and found_w >= max(1, _REL_SHARE * expected)
            and duration > dur_o + _SAME_LEN_SEC
        ):
            out.append(
                {
                    "other": other,
                    "kind": "contains",
                    "offset": round(-shift, 2),
                    "matched": round(dur_o, 1),
                    "score": score,
                    "new_contains": True,
                }
            )
    return out


def _store_result(
    store, path: str, size: int, mtime: float, result: dict, h, t, *, source: str, seconds: float
) -> int:
    snap = store.snapshot()

    def durations(ids):
        return {i: r.get("duration") for i, r in store.files_by_id(ids).items()}

    try:
        rels = relations(snap, h, t, float(result.get("duration") or 0), durations)
    except Exception:  # noqa: BLE001 — the fingerprints count more than the relations
        logger.exception("Audio analysis: relations of %s", path)
        rels = []
    return store.add(
        path,
        size,
        mtime,
        result=result,
        hashes=h,
        frames=t,
        relations=rels,
        source=source,
        seconds=seconds,
    )


def analyze_file(store, path: str, *, source: str = "run", stop=None, pace=None) -> dict:
    """Analyze the recording at ``path`` (absolute) and store it. Returns
    ``{"id", "duration", "seconds", "error"}``; an unreadable recording is
    stored as failed (it is tried again once it changes)."""
    p = Path(path)
    st = p.stat()
    t0 = time.monotonic()
    try:
        result, h, t = measure(decode(p, stop=stop, pace=pace))
    except AnalysisError as e:
        rid = store.add(
            path,
            st.st_size,
            st.st_mtime,
            error=str(e)[:300] or "unreadable",
            source=source,
            seconds=time.monotonic() - t0,
        )
        return {"id": rid, "duration": 0.0, "seconds": time.monotonic() - t0, "error": str(e)}
    secs = time.monotonic() - t0
    rid = _store_result(
        store, path, st.st_size, st.st_mtime, result, h, t, source=source, seconds=secs
    )
    return {"id": rid, "duration": result["duration"], "seconds": secs, "error": ""}


def analyze_wav(
    store, fh, path: str, size: int, mtime: float, *, source: str = "transcription", stop=None
) -> dict:
    """Analyze a recording from the 16 kHz WAV its transcription made
    (``fh``: the WAV, open) and store it under ``path``."""
    t0 = time.monotonic()
    result, h, t = measure(read_wav(fh, stop=stop))
    secs = time.monotonic() - t0
    rid = _store_result(store, path, size, mtime, result, h, t, source=source, seconds=secs)
    return {"id": rid, "duration": result["duration"], "seconds": secs, "error": ""}


# ── Snippets ─────────────────────────────────────────────────────────────


def snippet(path: str | Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """The sound of a snippet (int16, 16 kHz): ``start``…``end`` of a
    recording, or the beginning of a file of one's own — QUERY_MAX_SEC at
    most."""
    length = (
        QUERY_MAX_SEC if end is None or start is None else min(QUERY_MAX_SEC, max(0.0, end - start))
    )
    parts = list(
        decode(path, start=start if start is not None else 0.0, length=length, low_priority=False)
    )
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def search_snippet(store, x: np.ndarray, *, exclude_ids=(), limit: int = 500) -> dict:
    """Search a snippet's sound in the index: ``{"hits", "hashes", "ms"}``
    with each hit's file row (path, duration, loudness …) as ``file``."""
    t0 = time.monotonic()
    if len(x) > DENSE_MAX_SEC * SR:
        h, t = fingerprint.landmarks(x, fingerprint.STORE)
    else:
        h, t = fingerprint.query_landmarks(x)
    hits = search(store.snapshot(), h, t, exclude_ids=exclude_ids, limit=limit) if len(h) else []
    rows = store.files_by_id([x["id"] for x in hits])
    out = []
    for hit in hits:
        row = rows.get(hit["id"])
        if row is None or row["status"] != "done":
            continue
        out.append(
            {**hit, "file": row, "strength": "strong" if hit["score"] >= STRONG_SCORE else "match"}
        )
    return {
        "hits": out,
        "hashes": int(len(np.unique(h))) if len(h) else 0,
        "seconds": round(len(x) / SR, 2),
        "ms": round((time.monotonic() - t0) * 1000),
    }

"""Fingerprints (whisper_tm/audio/fingerprint.py) and the search for a
snippet (audio/analyze.py).

The sound here is synthetic: notes of one to three partials at random
pitches, a few a second, over faint noise — every seed a different
"recording" with plenty of spectral peaks, and nothing the tests need from
the network. ffmpeg (where present) makes the copies: mp3 at 32 kbit/s.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from whisper_tm.audio import analyze, fingerprint
from whisper_tm.audio.fingerprint import SR

FFMPEG = shutil.which("ffmpeg") is not None


def song(seed: int, seconds: float) -> np.ndarray:
    """A recording of its own for every seed: float32 at 16 kHz.

    Struck notes — a 5 ms attack, then a decay — as plucked strings or
    syllables have them: a note held at one level for 300 ms has its peak
    anywhere along the plateau, and a little noise moves it, while real
    sound has onsets that pin it."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR), np.float32)
    t = 0
    while t < len(out):
        dur = int(rng.uniform(0.08, 0.35) * SR)
        tt = np.arange(dur) / SR
        env = np.minimum(tt / 0.005, 1.0) * np.exp(-tt / rng.uniform(0.04, 0.2))
        note = sum(
            rng.uniform(0.1, 0.5)
            * np.sin(2 * np.pi * rng.uniform(150, 3500) * tt + rng.uniform(0, 6))
            for _ in range(int(rng.integers(1, 4)))
        )
        end = min(len(out), t + dur)
        out[t:end] += (note * env)[: end - t]
        t += int(dur * rng.uniform(0.6, 1.1))
    out += rng.normal(0, 0.003, len(out)).astype(np.float32)
    return (out / np.abs(out).max() * 0.5).astype(np.float32)


def as_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def write(x: np.ndarray, path: Path, *args: str) -> Path:
    """``x`` as a media file (ffmpeg; ``args``: encoder options)."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-i",
            "-",
            *args,
            str(path),
        ],
        input=as_int16(x).tobytes(),
        check=True,
    )
    return path


def reencoded(x: np.ndarray, *args: str) -> np.ndarray:
    """``x`` through an encoder and back (int16 at 16 kHz)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = write(x, Path(tmp) / "copy.mp3", *args)
        return np.concatenate(list(analyze.decode(p, low_priority=False)))


# ── Landmarks ────────────────────────────────────────────────────────────


def test_pieces_give_the_landmarks_of_the_whole():
    """A recording fed in pieces of any size is fingerprinted exactly as at
    once — the background run and the transcription feed ten seconds at a
    time, a search the whole snippet."""
    x = song(1, 200)
    h1, t1 = fingerprint.landmarks(x)
    fp = fingerprint.Fingerprinter()
    rng = np.random.default_rng(5)
    i = 0
    while i < len(x):
        n = int(rng.integers(1, 90_000))
        fp.feed(x[i : i + n])
        i += n
    h2, t2 = fp.finish()
    assert len(h1) > 1000
    a, b = np.lexsort((h1, t1)), np.lexsort((h2, t2))
    assert np.array_equal(h1[a], h2[b]) and np.array_equal(t1[a], t2[b])


def test_int16_and_float_give_the_same():
    x = song(2, 20)
    h1, _ = fingerprint.landmarks(x)
    h2, _ = fingerprint.landmarks(as_int16(x))
    assert abs(len(h1) - len(h2)) <= len(h1) * 0.02


def test_density_and_hash_fields():
    """About ten hashes a second are stored; every hash fits its 21 bits —
    first bin, bin step and frame step within their ranges."""
    x = song(3, 120)
    h, t = fingerprint.landmarks(x)
    per_sec = len(h) / 120
    assert 6 <= per_sec <= 14, per_sec
    assert int(h.max()) < (1 << fingerprint.HASH_BITS)
    dt = h & 63
    df = ((h >> 6) & 127).astype(int) - fingerprint.DF_MAX
    f1 = h >> 13
    assert dt.min() >= fingerprint.DT_MIN and dt.max() <= fingerprint.DT_MAX
    assert np.abs(df).max() <= fingerprint.DF_MAX
    assert f1.max() < fingerprint.FMAX - fingerprint.FMIN
    assert np.all(np.diff(t.astype(np.int64)) >= 0)


def test_silence_has_no_landmarks():
    h, t = fingerprint.landmarks(np.zeros(SR * 30, np.float32))
    assert len(h) == 0 and len(t) == 0
    quiet = np.random.default_rng(1).normal(0, 1e-5, SR * 30).astype(np.float32)
    assert len(fingerprint.landmarks(quiet)[0]) == 0


def test_short_sound_has_no_frames():
    h, t = fingerprint.landmarks(np.zeros(500, np.float32))
    assert len(h) == 0


def test_query_landmarks_are_denser_and_distinct():
    x = song(4, 8)
    h, t = fingerprint.query_landmarks(x)
    stored, _ = fingerprint.landmarks(x)
    assert len(h) > 2 * len(stored)
    pairs = set(zip(h.tolist(), t.tolist(), strict=True))
    assert len(pairs) == len(h)


# ── Searching ────────────────────────────────────────────────────────────


class _Snap:
    """A snapshot of landmarks in memory, as store.Snapshot hands them out."""

    def __init__(self, recordings: dict[int, np.ndarray]):
        from whisper_tm.audio import index

        hs, es = [], []
        for rid, x in recordings.items():
            h, t = fingerprint.landmarks(x)
            hs.append(h)
            es.append(index.pack(np.full(len(t), rid, np.uint64), t))
        self.h = np.concatenate(hs)
        self.e = np.concatenate(es)

    def lookup(self, hashes, max_bucket=0):
        pos = np.minimum(np.searchsorted(hashes, self.h), len(hashes) - 1)
        hit = hashes[pos] == self.h
        return self.e[hit], pos[hit].astype(np.int64)

    def bucket_sizes(self, hashes):
        u, c = np.unique(self.h, return_counts=True)
        pos = np.minimum(np.searchsorted(u, hashes), len(u) - 1)
        return np.where(u[pos] == hashes, c[pos], 0)


@pytest.fixture(scope="module")
def library():
    recs = {i: song(100 + i, 60) for i in range(1, 9)}
    return recs, _Snap(recs)


def _find(snap, x):
    h, t = fingerprint.query_landmarks(x)
    return analyze.search(snap, h, t)


def test_a_snippet_is_found_where_it_is(library):
    recs, snap = library
    x = recs[3][int(21.5 * SR) : int(26.5 * SR)]  # 5 s from 21.5 s on
    hits = _find(snap, x)
    assert hits and hits[0]["id"] == 3
    assert abs(hits[0]["at"] - 21.5) <= 0.1
    assert hits[0]["score"] >= analyze.STRONG_SCORE
    assert len(hits) == 1


def test_three_seconds_are_enough(library):
    recs, snap = library
    for rid, start in ((1, 4.0), (5, 40.3), (8, 55.0)):
        hits = _find(snap, recs[rid][int(start * SR) : int((start + 3) * SR)])
        assert hits and hits[0]["id"] == rid and abs(hits[0]["at"] - start) <= 0.1


def test_quieter_and_noisy_copies_are_found(library):
    recs, snap = library
    x = recs[6][10 * SR : 20 * SR]
    quiet = x * 0.1  # -20 dB
    rms = float(np.sqrt(np.mean(x**2)))
    noisy = x + np.random.default_rng(3).normal(0, rms, len(x)).astype(np.float32)  # 0 dB SNR
    for y in (quiet, noisy):
        hits = _find(snap, y)
        assert hits and hits[0]["id"] == 6 and abs(hits[0]["at"] - 10) <= 0.1


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_an_mp3_copy_is_found(library):
    recs, snap = library
    y = reencoded(recs[2][30 * SR : 36 * SR], "-b:a", "32k")
    hits = _find(snap, y)
    assert hits and hits[0]["id"] == 2 and abs(hits[0]["at"] - 30) <= 0.15


def test_unknown_sound_finds_nothing(library):
    _, snap = library
    assert _find(snap, song(999, 10)) == []
    # Every hash of the library, but backwards in time: no offset agrees
    recs, _ = library
    assert _find(snap, recs[4][5 * SR : 15 * SR][::-1].copy()) == []


def test_a_recording_holding_the_snippet_twice_has_two_places():
    a = song(7, 20)
    b = song(8, 30)
    twice = np.concatenate([a[: 10 * SR], b, a[2 * SR : 12 * SR]])
    snap = _Snap({1: twice, 2: b})
    hits = _find(snap, a[3 * SR : 8 * SR])
    assert hits and hits[0]["id"] == 1
    places = sorted(p["at"] for p in hits[0]["places"])
    assert len(places) == 2
    assert abs(places[0] - 3.0) <= 0.1 and abs(places[1] - (10 + 30 + 1)) <= 0.1


def test_relations_same_and_contains():
    """The windows of a new recording find what the index holds: a copy of
    the same length is "same"; a longer one holding it all "contains" it;
    a new recording holding an older one "contains" that one."""
    a, b, c = song(11, 40), song(12, 30), song(13, 35)
    compilation = np.concatenate([c, a * 0.5, b])
    snap = _Snap({1: a, 2: b, 3: compilation})
    h, t = fingerprint.landmarks(
        a + np.random.default_rng(1).normal(0, 0.01, len(a)).astype(np.float32)
    )
    durations = {1: 40.0, 2: 30.0, 3: 105.0}

    def lengths(ids):
        return {i: durations[i] for i in ids}

    rel = {r["other"]: r for r in analyze.relations(snap, h, t, 40.0, lengths)}
    assert rel[1]["kind"] == "same" and abs(rel[1]["offset"]) <= 0.1
    assert rel[3]["kind"] == "contains" and not rel[3]["new_contains"]
    assert abs(rel[3]["offset"] - 35.0) <= 0.1  # a starts at 35 s in it
    assert 2 not in rel
    # The other way round: the compilation is new, a and c are in the index
    snap2 = _Snap({1: a, 4: c})
    h, t = fingerprint.landmarks(compilation)
    rel = {
        r["other"]: r for r in analyze.relations(snap2, h, t, 105.0, lambda ids: {1: 40.0, 4: 35.0})
    }
    assert rel[1]["kind"] == "contains" and rel[1]["new_contains"]
    assert abs(rel[1]["offset"] - 35.0) <= 0.1
    assert rel[4]["kind"] == "contains" and abs(rel[4]["offset"]) <= 0.1


def test_relations_ignore_a_shared_intro():
    """Two different recordings that share ten seconds are neither the same
    nor does one contain the other."""
    intro = song(20, 10)
    a = np.concatenate([intro, song(21, 50)])
    b = np.concatenate([intro, song(22, 50)])
    snap = _Snap({1: a})
    h, t = fingerprint.landmarks(b)
    assert analyze.relations(snap, h, t, 60.0, lambda ids: {1: 60.0}) == []


def test_a_lookup_over_its_budget_leaves_the_commonest_out(library, monkeypatch):
    """A lookup reads at most MAX_ENTRIES: the commonest hashes go first —
    and the snippet is still found by the rest."""
    recs, snap = library
    x = recs[5][20 * SR : 30 * SR]
    h, t = fingerprint.query_landmarks(x)
    full = analyze._matches(snap, h, t)[0]
    monkeypatch.setattr(analyze, "MAX_ENTRIES", len(full) // 3)
    cut = analyze._matches(snap, h, t)[0]
    assert len(cut) < len(full)
    hits = analyze.search(snap, h, t)
    assert hits and hits[0]["id"] == 5


def test_the_score_needed_grows_with_the_snippet(library, monkeypatch):
    recs, snap = library
    h, t = fingerprint.query_landmarks(recs[2][10 * SR : 16 * SR])
    assert analyze.search(snap, h, t)
    monkeypatch.setattr(analyze, "MIN_SHARE", 1.0)  # every hash would have to agree
    assert analyze.search(snap, h, t) == []

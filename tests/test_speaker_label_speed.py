"""Labelling a long recording's sentences is not quadratic.

Previously ``label_segments`` and ``align_labels`` held every segment
against every turn — for a recording of many hours (10,000 segments ×
10,000 turns) that is 10⁸ Python steps in the worker of a voices-only run.

Both now only look at the turns near a segment — with exactly the results
of the full comparison, ties and sums included (checked here against the
old double loop on random material).
"""

from __future__ import annotations

import random
import time

from whisper_tm.core import speaker as sp


def _old_label_segments(segments, turns, reach=1.0):
    out = []
    for a, b in segments:
        acc: dict[str, float] = {}
        near, gap = "—", reach
        for t in turns:
            ov = min(t["end"], b) - max(t["start"], a)
            if ov > 0:
                acc[t["speaker"]] = acc.get(t["speaker"], 0.0) + ov
            else:
                d = -ov
                if d <= gap:
                    near, gap = t["speaker"], d
        out.append(max(acc, key=acc.get) if acc else near)
    return out


def _old_overlap(turns, labelled):
    overlap: dict[str, dict[str, float]] = {}
    for t in turns:
        acc = overlap.setdefault(t["speaker"], {})
        for s, a, b in labelled:
            ov = min(t["end"], b) - max(t["start"], a)
            if ov > 0:
                acc[s] = acc.get(s, 0.0) + ov
    return overlap


def _material(seed, n_turns=400, n_segs=500, length=3600.0):
    r = random.Random(seed)
    turns = []
    for _ in range(n_turns):
        a = round(r.uniform(0, length), 1)
        turns.append(
            {
                "speaker": f"SPEAKER_{r.randrange(5):02d}",
                "start": a,
                "end": round(a + r.choice([0.3, 2, 8, 40, 300]), 1),
            }
        )
    segs = []
    for _ in range(n_segs):
        a = round(r.uniform(0, length), 1)
        segs.append((a, round(a + r.choice([0.0, 0.5, 3, 12]), 1)))
    return turns, segs


def test_label_segments_matches_the_full_comparison():
    for seed in range(6):
        turns, segs = _material(seed)
        assert sp.label_segments(segs, turns) == _old_label_segments(segs, turns)
        assert sp.label_segments(segs, turns, reach=5.0) == _old_label_segments(
            segs, turns, reach=5.0
        )


def test_align_labels_matches_the_full_comparison():
    for seed in range(6):
        turns, segs = _material(seed)
        r = random.Random(seed + 100)
        transcript = [(f"SPEAKER_{r.randrange(4):02d}", a, b) for a, b in segs]
        labelled = [(s, a, b) for s, a, b in transcript if b > a]
        assert sp._overlap_by_label(turns, labelled) == _old_overlap(turns, labelled)


def test_a_long_recording_is_quick():
    r = random.Random(1)
    turns, segs, t = [], [], 0.0
    for i in range(12000):
        d = r.uniform(1, 6)
        turns.append({"speaker": f"SPEAKER_{i % 4:02d}", "start": t, "end": t + d})
        segs.append((t, t + d))
        t += d + r.uniform(0, 0.5)
    t0 = time.perf_counter()
    sp.label_segments(segs, turns)
    sp.align_labels(turns, [("SPEAKER_00", a, b) for a, b in segs])
    assert time.perf_counter() - t0 < 3.0


def test_endless_and_broken_turns_count_as_before():
    turns = [
        {"speaker": "A", "start": 0.0, "end": float("inf")},
        {"speaker": "B", "start": float("nan"), "end": 5.0},
        {"speaker": "C", "start": 100.0, "end": 101.0},
    ]
    segs = [(10.0, 12.0), (100.2, 100.8), (float("nan"), 1.0)]
    assert sp.label_segments(segs, turns) == _old_label_segments(segs, turns)

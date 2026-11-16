"""The largest group must not disappear because it got marker 0.

Symptom: lowering the threshold from 0.62 to 0.61 left only about half of
all voices visible, and the largest speaker was missing completely:

    Threshold 0.62   the largest speaker is a group of thousands of voices · few singletons
    Threshold 0.61   that speaker doesn't exist · thousands of singletons

The group didn't split — it was declared a singleton. The cause is in
``group_voices``: ``next_one`` started at **zero**, and ``order`` processes
components by SIZE. The first subdivision of the largest component got marker
0, and the API reads 0 as "belongs to no group".

Whether the largest person in the data existed at all depended on internal
numbering — a nudge of the threshold slider decided it anew, in both
directions, without any indication.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import speaker_knn as sk


def _clouds(sizes, spread=0.03, dim=192, seed=5):
    """Build clearly separated clouds — each cloud is a "speaker"."""
    rng = np.random.default_rng(seed)
    parts = []
    for n in sizes:
        middle = rng.normal(size=(1, dim))
        parts.append(middle + rng.normal(scale=spread, size=(n, dim)))
    X = np.vstack(parts).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class TestZeroIsReserved:
    def test_no_group_gets_the_zero(self):
        """The core. 0 means "unbound" — a real group must never carry it, or
        it disappears from every group view."""
        X = _clouds([300, 200, 120, 40, 12])
        m = sk.group_voices(X, threshold=0.6, k=5, device="cpu")["markers"]
        assert (m > 0).all(), f"{int((m == 0).sum())} voices carry the reserved zero"

    def test_the_largest_component_survives(self):
        """It is processed first and used to get zero earlier."""
        X = _clouds([500, 60, 30])
        m = sk.group_voices(X, threshold=0.6, k=5, device="cpu")["markers"]
        largest = np.bincount(m).max()
        assert largest >= 400, (
            f"largest group only {largest} — large cloud split apart or stored as unbound"
        )

    @pytest.mark.parametrize("threshold", [0.50, 0.55, 0.60, 0.65, 0.70])
    def test_at_every_threshold(self, threshold):
        """The error occurred on ADJUSTMENT — it depended on numbering, not
        on a specific value."""
        X = _clouds([300, 150, 80, 20])
        m = sk.group_voices(X, threshold=threshold, k=5, device="cpu")["markers"]
        assert (m > 0).all(), f"marker 0 assigned at threshold {threshold}"

    def test_small_components_do_not_collide_with_the_zero(self):
        """Even components under three members count from one."""
        X = _clouds([2, 2, 2, 2], spread=0.01)
        m = sk.group_voices(X, threshold=0.95, k=3, device="cpu")["markers"]
        assert (m > 0).all()

    def test_a_single_vector(self):
        X = _clouds([1])
        result = sk.group_voices(X, threshold=0.6, k=1, device="cpu")
        assert (result["markers"] > 0).all()

    def test_empty_input_does_not_crash(self):
        result = sk.group_voices(
            np.zeros((0, 192), dtype=np.float32), threshold=0.6, k=5, device="cpu"
        )
        assert result["nodes"] == 0


class TestLoners:
    """What "singleton" SHOULD mean: alone in its own group."""

    def test_the_api_counts_groups_of_size_one(self):
        """The map's legend (core/speaker_graph.py): marker 0 is no group,
        and a group of one is a loner however it is numbered."""
        from whisper_tm.core import speaker_graph

        markers = np.array([0, 0, 3, 1, 1, 2, 3, 3, 4], dtype=np.int32)
        idx = np.zeros((len(markers), 1), dtype=np.int32)
        value = np.zeros((len(markers), 1), dtype=np.float32)
        c = speaker_graph._make_cut(markers, idx, value, 0.5)
        assert c["loners"] == 2  # groups 2 and 4
        assert len(c["legend"]) == 4  # 0 is not a group
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "whisper_tm"
        for f in (root / "api" / "speaker.py", root / "core" / "speaker_graph.py"):
            assert '"loners": int((markers == 0).sum())' not in f.read_text(encoding="utf-8"), (
                "old counting counts whoever happened to get zero"
            )

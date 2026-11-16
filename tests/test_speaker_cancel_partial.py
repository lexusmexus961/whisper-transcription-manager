"""A cancel during the voice vectors must never store a partial set.

Background: on ``job.cancel`` the window loop stopped and handed back the
windows it had so far — and ``carry_along`` wrote them, deleting the
recording's complete set (and the names on it) in favour of a few windows.
A run with holes is not a valid run; a cancelled one writes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import speaker as sp
from whisper_tm.database.speaker import SpeakerDB


def _vec(seed, dim=192):
    r = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return r / np.linalg.norm(r)


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    return d


@pytest.fixture
def model(monkeypatch):
    calls = []

    def load(device="cuda", job=None):
        def compute(w):
            calls.append(1)
            return _vec(len(calls))

        return compute

    monkeypatch.setattr(sp, "load_model", load)
    return calls


def _turns():
    return [{"start": 0.0, "end": 120.0, "speaker": "SPEAKER_00"}]


def _old_set(db, path):
    db.write(
        path,
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": i + 1,
                "start_sec": i * 30.0,
                "speech_sec": 30.0,
                "segments": 2,
                "parts": [(i * 30.0, i * 30.0 + 30)],
                "v_separated": _vec(100 + i),
            }
            for i in range(4)
        ],
    )
    ids = [w["id"] for w in db.windows(path)]
    db.assign_many(ids, "Alice")
    return ids


def test_compute_vectors_raises_on_abort(model):
    audio = np.zeros(120 * sp.SR, dtype=np.float32)
    plan = sp.plan_windows(_turns())
    assert len(plan) == 4
    with pytest.raises(InterruptedError):
        sp.compute_vectors(audio, plan, device="cpu", abort=lambda: len(model) >= 2)


@pytest.mark.parametrize("keep", [False, True])
def test_cancel_keeps_the_existing_vectors_and_names(db, model, keep):
    path = "/media/talk.mkv"
    ids = _old_set(db, path)
    audio = np.zeros(120 * sp.SR, dtype=np.float32)
    with pytest.raises(InterruptedError):
        sp.carry_along(
            path,
            audio,
            _turns(),
            device="cpu",
            abort=lambda: len(model) >= 2,
            separated=True,
            keep_decisions=keep,
        )
    after = db.windows(path)
    assert [w["id"] for w in after] == ids
    assert all(w["person"] == "Alice" for w in after)

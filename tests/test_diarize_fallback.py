"""A failed speaker step keeps the transcript (transcription._diarize_or_none).

The transcript is done when the speakers are looked for. A failure that
another try won't mend — the model can't be downloaded, the token was
revoked, the worker was killed, the time limit on a slow CPU — used to fail
the whole job and throw the transcript away. Now it is saved without
speakers and the job says so, as without a token. A full card stays the
queue's (it waits and tries again, never books an error), a cancel stays a
cancel, and a voices-only run has nothing else to give.
"""

from __future__ import annotations

import inspect
import types

import pytest

from whisper_tm.core import transcription as tr


class _Jobs:
    def __init__(self):
        self.warnings = []

    def warn(self, job, msg):
        self.warnings.append(msg)


@pytest.fixture
def jobs(monkeypatch):
    j = _Jobs()
    monkeypatch.setattr(tr, "job_manager", j)
    return j


def call(monkeypatch, outcome, *, voices_only=False, cancel=False):
    def fake(*a, **k):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(tr, "_diarize", fake)
    job = types.SimpleNamespace(cancel=cancel)
    return tr._diarize_or_none(
        "a.wav",
        None,
        None,
        "cpu",
        "hf_x",
        job=job,
        total_dur=60.0,
        load_done=None,
        t_load=0.0,
        voices_only=voices_only,
    )


def test_the_speakers_when_it_works(monkeypatch, jobs):
    assert call(monkeypatch, ["turns"]) == ["turns"] and jobs.warnings == []


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("401 Client Error: Unauthorized for url: https://huggingface.co/pyannote/…"),
        TimeoutError("Diarization took longer than 1800 s"),
        RuntimeError("The diarization worker died (exit code -9)"),
        OSError("We couldn't connect to 'https://huggingface.co'"),
    ],
)
def test_a_failure_another_try_wont_mend_keeps_the_transcript(monkeypatch, jobs, error):
    assert call(monkeypatch, error) is None
    assert len(jobs.warnings) == 1 and jobs.warnings[0].startswith("Speaker recognition failed")


def test_a_full_card_stays_the_queues(monkeypatch, jobs):
    with pytest.raises(RuntimeError, match="out of memory"):
        call(monkeypatch, RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    with pytest.raises(MemoryError):
        call(
            monkeypatch,
            MemoryError(
                "batch_size (32) is probably too large. Try with a smaller "
                "value until memory error disappears."
            ),
        )
    assert jobs.warnings == []


def test_a_cancel_and_a_voices_only_run_fail_as_before(monkeypatch, jobs):
    with pytest.raises(InterruptedError):
        call(monkeypatch, InterruptedError("Cancelled"))
    with pytest.raises(InterruptedError):
        call(monkeypatch, RuntimeError("worker gone"), cancel=True)
    with pytest.raises(RuntimeError):
        call(monkeypatch, RuntimeError("401"), voices_only=True)
    assert jobs.warnings == []


def test_the_worker_saves_without_speakers_then():
    src = inspect.getsource(tr.transcription_worker)
    assert "_diarize_or_none(" in src and "voices_only=voices_only" in src
    after = src[src.index("_diarize_or_none(") :]
    assert "if diar_segs is None:" in after and "final = aligned" in after[: after.index("else:")]

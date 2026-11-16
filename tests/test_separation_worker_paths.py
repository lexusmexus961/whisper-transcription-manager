"""The worker's separation-only path, run for real up to the separator.

Background:

* ``_save_audio_alongside`` caught EVERY exception, cancellation included
  (``InterruptedError`` is an ``OSError``): a cancel during the MP3 step of
  a separation-only run completed the job as "separated", with a truncated
  MP3 left next to the recording. It also never looked at ffmpeg's exit
  code, wrote straight onto the final name, and called a failed write
  "saved".
* A short recording the prefilter found silent made a separation-only run
  fail ("Separation produced no vocals track") — there was nothing to
  separate, which is no error.

ffmpeg is real (provisioning, the MP3 encode); the separator is a stand-in.
"""

from __future__ import annotations

import shutil
import subprocess
import types

import pytest

from whisper_tm.core import transcription as tr
from whisper_tm.core import vocal_isolation as vi
from whisper_tm.jobs import Job

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


class _Jobs:
    def __init__(self, job):
        self.job = job
        self.done = None
        self.failed = None
        self.warnings = []

    def get(self, _jid):
        return self.job

    def log(self, job, msg):
        job.log.append(msg)

    def warn(self, job, msg):
        job.warning_count += 1
        job.warnings.append(msg)
        self.warnings.append(msg)

    def record_step(self, *a, **k):
        pass

    def add_live_segments(self, *a, **k):
        pass

    def system_warning(self, *a, **k):
        pass

    def complete(self, job, result=None):
        self.done = result
        job.status = "done"

    def fail(self, job, error):
        self.failed = error
        job.status = "error"


@pytest.fixture
def worker(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    src = media / "talk.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(src),
        ],
        check=True,
        timeout=60,
    )
    job = Job(job_id="talk_wav_test", filename="talk.wav")
    jobs = _Jobs(job)
    monkeypatch.setattr(tr, "job_manager", jobs)
    monkeypatch.setattr(vi, "job_manager", jobs)
    monkeypatch.setattr(
        tr,
        "hw",
        types.SimpleNamespace(
            has_audio_separator=True,
            best_device="cpu",
            best_gpu_index=0,
            recommended_compute_type="int8",
            recommended_batch_size=1,
        ),
    )
    monkeypatch.setattr(tr, "get_whisperx_device", lambda hw: "cpu")
    monkeypatch.setattr(tr, "_whisperx", lambda *parts: types.SimpleNamespace())
    monkeypatch.setattr(tr, "db", None)
    from whisper_tm.core import autoscan

    monkeypatch.setattr(autoscan, "note_made", lambda *a, **k: None)

    def run(**kw):
        tr.transcription_worker(
            job.job_id,
            src,
            "talk.wav",
            diarize=False,
            vocal_isolation=True,
            separation_only=True,
            save_audio_alongside=True,
            **kw,
        )
        return job, jobs, media

    return run


def _stems(work, kind="tone"):
    v = work / "vocal_iso_work" / "vocals.wav"
    i = work / "vocal_iso_work" / "inst.wav"
    v.parent.mkdir(parents=True, exist_ok=True)
    for p in (v, i):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=300:duration=3",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(p),
            ],
            check=True,
            timeout=60,
        )
    return v, i


def test_a_cancel_while_saving_the_tracks_cancels_the_run(worker, monkeypatch):
    def isolate(source, work, job=None, **k):
        return _stems(work)

    monkeypatch.setattr(vi, "isolate_vocals", isolate)
    real = tr.run_cancellable

    def run(cmd, job=None, timeout=0, label=None):
        if "vocals.wav" in " ".join(map(str, cmd)) and "libmp3lame" in cmd:
            job.cancel = True
            raise InterruptedError("Cancelled")
        return real(cmd, job=job, timeout=timeout, label=label)

    monkeypatch.setattr(tr, "run_cancellable", run)
    job, jobs, media = worker()
    assert jobs.done is None and jobs.failed == "Cancelled"
    assert not (media / "talk_vocals.mp3").exists()


def test_a_failed_mp3_leaves_nothing_and_is_not_called_saved(worker, monkeypatch):
    def isolate(source, work, job=None, **k):
        return _stems(work)

    monkeypatch.setattr(vi, "isolate_vocals", isolate)
    real = tr.run_cancellable

    def run(cmd, job=None, timeout=0, label=None):
        if "vocals.wav" in " ".join(map(str, cmd)) and "libmp3lame" in cmd:
            r = real(cmd, job=job, timeout=timeout, label=label)
            return subprocess.CompletedProcess(cmd, 1, r.stdout, "disk full")
        return real(cmd, job=job, timeout=timeout, label=label)

    monkeypatch.setattr(tr, "run_cancellable", run)
    job, jobs, media = worker()
    assert sorted(p.name for p in media.iterdir() if p.suffix == ".mp3") == [
        "talk_instrumental.mp3"
    ]
    assert not [p for p in media.iterdir() if p.name.endswith(".tmp")]
    assert any("vocals" in w.lower() for w in jobs.warnings)
    assert not any("Vocals: talk_vocals.mp3" in line for line in job.log)


def test_the_tracks_are_saved(worker, monkeypatch):
    monkeypatch.setattr(vi, "isolate_vocals", lambda source, work, job=None, **k: _stems(work))
    job, jobs, media = worker()
    assert jobs.done is not None
    assert (media / "talk_vocals.mp3").stat().st_size > 1000
    assert (media / "talk_instrumental.mp3").stat().st_size > 1000


def test_a_silent_recording_is_no_failed_separation(worker, monkeypatch):
    def isolate(source, work, job=None, **k):
        job.separation_silent = True  # what chunked_separate notes
        return None, None

    monkeypatch.setattr(vi, "isolate_vocals", isolate)
    job, jobs, media = worker()
    assert jobs.failed is None and jobs.done is not None
    assert jobs.done.get("silent") is True


def test_chunked_separate_notes_a_silent_recording(tmp_path, monkeypatch):
    src = tmp_path / "quiet.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "3",
            str(src),
        ],
        check=True,
        timeout=60,
    )
    monkeypatch.setattr(vi, "_chunk_prefilter", lambda *a, **k: "silence")
    monkeypatch.setattr(vi, "job_manager", _Jobs(None))
    job = Job(job_id="q")
    (tmp_path / "w").mkdir()
    assert vi.chunked_separate(src, tmp_path / "w", job=job) == (None, None)
    assert getattr(job, "separation_silent", False) is True

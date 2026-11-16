"""Diarization must be cancellable — otherwise a hang blocks the GPU.

Starting point: pyannote knows no time limit. If it hangs, the job cannot be
cancelled, and the GPU stays occupied until the server restarts. A watchdog
CANNOT solve this — Python cannot kill a thread, and an aborted call would
leave the CUDA context orphaned. For a batch run over a large library (days
of processing) that is not an edge case.

The solution is a dedicated, long-lived child process. These tests check it
with **real processes**, just without pyannote: the doubles below are small
Python scripts that speak the same line protocol — one hangs, one dies, one
chatters. Only this way is it actually verified that a hanging process really
dies; with a mocked object exactly that would remain untested.
"""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time

import pytest

from whisper_tm.core import diar_process

# ── Dummy workers ───────────────────────────────────────────────────────


def _dummy(tmp_path, name: str, body: str) -> list:
    """Write a script and return the command that starts it."""
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(path)]


WELL_BEHAVED = """
    import json, sys
    print(json.dumps({"kind": "ready"}), flush=True)
    for line in sys.stdin:
        job_ref = json.loads(line)
        if job_ref.get("kind") == "end":
            break
        print(json.dumps({"kind": "result", "ok": True, "segments": [
            [0.0, 1.5, "SPEAKER_00"], [1.5, 3.0, "SPEAKER_01"]]}), flush=True)
"""

HANGS = """
    import json, sys, time
    print(json.dumps({"kind": "ready"}), flush=True)
    for line in sys.stdin:
        # Exactly the behavior at stake here: the job arrives, and then
        # nothing ever returns.
        while True:
            time.sleep(3600)
"""

DIES_ON_START = """
    import sys
    sys.stderr.write("Diarization model not loadable: no cache\\n")
    sys.exit(1)
"""

CHATTERS = """
    import json, sys
    # A library that writes to stdout unprompted — that must not break the
    # protocol.
    print("Lightning automatically upgraded your loaded checkpoint", flush=True)
    print(json.dumps({"kind": "ready"}), flush=True)
    for line in sys.stdin:
        print("tqdm: 100%|##########|", flush=True)
        print(json.dumps({"kind": "result", "ok": True,
                          "segments": [[0.0, 2.0, "SPEAKER_00"]]}), flush=True)
"""

DOMAIN_ERRORS = """
    import json, sys
    print(json.dumps({"kind": "ready"}), flush=True)
    counter = 0
    for line in sys.stdin:
        counter += 1
        if counter == 1:
            print(json.dumps({"kind": "result", "ok": False,
                              "error": "File is corrupt"}), flush=True)
        else:
            print(json.dumps({"kind": "result", "ok": True,
                              "segments": [[0.0, 1.0, "SPEAKER_00"]]}),
                  flush=True)
"""


class Job:
    """The bare minimum of the real job object: a flippable ``cancel``."""

    def __init__(self):
        self.cancel = False


@pytest.fixture
def cleanup():
    worker = []
    yield worker
    for a in worker:
        a.terminate("Test ended")
    diar_process.terminate_all("Test ended")


# ── The normal case ───────────────────────────────────────────────────────


class TestNormalOperation:
    def test_a_job_delivers_the_sections(self, tmp_path, cleanup):
        a = diar_process.Worker(command=_dummy(tmp_path, "good", WELL_BEHAVED))
        cleanup.append(a)
        segments = a.diarize_it(tmp_path / "any.wav")
        assert segments == [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")]

    def test_the_worker_stays_alive_between_jobs(self, tmp_path, cleanup):
        """The whole reason for a LONG-LIVED process.

        Measured: a fresh process until the model is loaded costs ~5 s.
        Over a library of ~100k files that would be well over a hundred
        extra hours — a throwaway process per file is therefore not an option.
        """
        a = diar_process.Worker(command=_dummy(tmp_path, "good2", WELL_BEHAVED))
        cleanup.append(a)
        a.diarize_it(tmp_path / "one.wav")
        pid = a._proc.pid
        a.diarize_it(tmp_path / "two.wav")
        assert a._proc.pid == pid, "the second job started a new process"
        assert a.alive

    def test_speaker_counts_are_passed_through(self, tmp_path, cleanup):
        """min/max must arrive in the job — otherwise the input has no effect."""
        echo = _dummy(
            tmp_path,
            "echo",
            """
            import json, sys
            print(json.dumps({"kind": "ready"}), flush=True)
            for line in sys.stdin:
                job_ref = json.loads(line)
                print(json.dumps({"kind": "result", "ok": True,
                                  "segments": [[0.0, 1.0, json.dumps(job_ref)]]}),
                      flush=True)
        """,
        )
        a = diar_process.Worker(command=echo)
        cleanup.append(a)
        segments = a.diarize_it("/path/to/audio.wav", min_speakers=2, max_speakers=5)
        job = json.loads(segments[0][2])
        assert job["wav"] == "/path/to/audio.wav"
        assert job["min_speakers"] == 2
        assert job["max_speakers"] == 5

    def test_chatter_on_stdout_does_not_break_the_protocol(self, tmp_path, cleanup):
        """Libraries write to stdout unprompted (Lightning, tqdm).

        The real worker redirects file descriptor 1 to stderr for this
        reason; the parent side additionally skips anything that is not
        JSON. Here the second part is checked — it is the safety net in
        case something gets through anyway.
        """
        a = diar_process.Worker(command=_dummy(tmp_path, "loud", CHATTERS))
        cleanup.append(a)
        assert a.diarize_it(tmp_path / "x.wav") == [(0.0, 2.0, "SPEAKER_00")]


# ── What this is really about ─────────────────────────────────────────────


class TestCancelling:
    def test_a_hanging_worker_is_cancelled_and_dies(self, tmp_path, cleanup):
        """The core of the matter: ``job.cancel`` really terminates a hang.

        Previously this was impossible — the call sat in its own process,
        and Python cannot kill a thread.
        """
        a = diar_process.Worker(command=_dummy(tmp_path, "hangs", HANGS))
        cleanup.append(a)
        a.start_it()
        proc = a._proc

        job = Job()
        threading.Timer(0.6, lambda: setattr(job, "cancel", True)).start()

        t0 = time.time()
        with pytest.raises(InterruptedError):
            a.diarize_it(tmp_path / "hang.wav", job=job, deadline_s=600)
        duration = time.time() - t0

        assert duration < 10, f"abort took {duration:.1f}s"
        assert not a.alive
        assert proc.poll() is not None, "the process is still alive"

    def test_the_time_limit_kills_a_hang_on_its_own(self, tmp_path, cleanup):
        """Without a time limit, a batch run hits a hang overnight.

        Nobody sits in front of the cancel button at three in the morning.
        """
        a = diar_process.Worker(command=_dummy(tmp_path, "hangs2", HANGS))
        cleanup.append(a)
        a.start_it()
        proc = a._proc

        with pytest.raises(TimeoutError) as error:
            a.diarize_it(tmp_path / "hang.wav", deadline_s=1.0)

        assert "Diarization" in str(error.value)
        assert not a.alive
        assert proc.poll() is not None

    def test_cancelling_works_already_at_startup(self, tmp_path, cleanup):
        """Loading the model (~5 s) must not be a hole with no way out either."""
        silent = _dummy(
            tmp_path,
            "muted",
            """
            import time
            while True:
                time.sleep(3600)
        """,
        )
        a = diar_process.Worker(command=silent)
        cleanup.append(a)
        job = Job()
        threading.Timer(0.6, lambda: setattr(job, "cancel", True)).start()
        with pytest.raises(InterruptedError):
            a.start_it(job=job)
        assert not a.alive


class TestErrorPaths:
    def test_a_worker_that_does_not_start_says_so(self, tmp_path, cleanup):
        a = diar_process.Worker(command=_dummy(tmp_path, "dead", DIES_ON_START))
        cleanup.append(a)
        with pytest.raises(RuntimeError) as error:
            a.start_it()
        # The last stderr lines belong in the message — otherwise it just
        # says "died" and nobody knows why.
        assert "no cache" in str(error.value)

    def test_a_domain_error_does_not_end_the_worker(self, tmp_path, cleanup):
        """A corrupt file is not a reason to throw away the model.

        Otherwise every unreadable file in a batch run would cost 5 s of
        reloading.
        """
        a = diar_process.Worker(command=_dummy(tmp_path, "domain_error", DOMAIN_ERRORS))
        cleanup.append(a)
        with pytest.raises(RuntimeError, match="File is corrupt"):
            a.diarize_it(tmp_path / "broken.wav")
        assert a.alive, "the worker was killed because of a corrupt file"
        assert a.diarize_it(tmp_path / "intact.wav") == [(0.0, 1.0, "SPEAKER_00")]

    def test_after_a_kill_the_next_job_keeps_running(self, tmp_path, cleanup):
        """A killed worker must not stop the queue.

        What is checked is the PROCESS, not the object: ``terminate()`` resets
        the worker to the "never started" state (fresh response queue
        included), so the same object may be reused — just with a new
        process.
        """
        command = _dummy(tmp_path, "good3", WELL_BEHAVED)
        a = diar_process.take("cpu", command=command)
        cleanup.append(a)
        a.diarize_it(tmp_path / "a.wav")
        old_pid = a._proc.pid
        a.terminate("Test: killed")

        b = diar_process.take("cpu", command=command)
        cleanup.append(b)
        assert b.diarize_it(tmp_path / "b.wav") == [
            (0.0, 1.5, "SPEAKER_00"),
            (1.5, 3.0, "SPEAKER_01"),
        ]
        assert b._proc.pid != old_pid, "the old process was still running"

    def test_a_worker_that_died_in_between_is_replaced(self, tmp_path, cleanup):
        """If the process dies on its own (OOM killer), ``fetch_it`` notices."""
        short_lived = _dummy(
            tmp_path,
            "short_lived",
            """
            import json, sys
            print(json.dumps({"kind": "ready"}), flush=True)
            sys.exit(0)
        """,
        )
        a = diar_process.take("cpu", command=short_lived)
        cleanup.append(a)
        a.start_it()
        a._proc.wait(timeout=5)
        assert not a.alive

        b = diar_process.take("cpu", command=short_lived)
        cleanup.append(b)
        assert b is not a, "the dead worker was reused"


# ── Cleanup: VRAM must come back ────────────────────────────────


class TestVramRelease:
    def test_terminate_all_kills_the_workers(self, tmp_path):
        a = diar_process.take("cpu", command=_dummy(tmp_path, "good4", WELL_BEHAVED))
        a.diarize_it(tmp_path / "a.wav")
        proc = a._proc
        assert diar_process.something_running()

        diar_process.terminate_all("Test")

        assert proc.poll() is not None
        assert not diar_process.something_running()

    def test_releasing_models_takes_the_worker_along(self, tmp_path):
        """The idle unloader calls ``release_models`` — and must also
        catch the process that holds the model in ITS OWN ADDRESS SPACE.

        Otherwise the server would consider itself unloaded while ~1 GB is
        still sitting on the card.
        """
        from whisper_tm.core import transcription

        a = diar_process.take("cpu", command=_dummy(tmp_path, "good5", WELL_BEHAVED))
        a.diarize_it(tmp_path / "a.wav")
        proc = a._proc

        transcription.release_models()

        assert proc.poll() is not None

    def test_the_idle_watchdog_sees_the_worker(self, tmp_path):
        """``_something_loaded`` decides whether anything gets unloaded at all."""
        from whisper_tm.core import model_idle

        try:
            a = diar_process.take("cpu", command=_dummy(tmp_path, "good6", WELL_BEHAVED))
            a.diarize_it(tmp_path / "a.wav")
            assert model_idle._something_loaded()
        finally:
            diar_process.terminate_all("Test")
        assert not model_idle._something_loaded()


# ── What the caller expects ────────────────────────────────


class TestTheShapeOfTheResult:
    def test_the_dataframe_has_exactly_the_columns_used(self):
        df = diar_process.as_dataframe([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.5, "SPEAKER_01")])
        assert list(df.columns) == ["start", "end", "speaker"]
        assert len(df) == 2

    def test_an_empty_result_stays_a_dataframe(self):
        """A recording with no detected speech yields zero segments.

        If ``None`` or an empty list came back instead, ``assign_word_speakers``
        would die on the missing ``columns``.
        """
        df = diar_process.as_dataframe([])
        assert list(df.columns) == ["start", "end", "speaker"]
        assert len(df) == 0

    def test_the_three_consumers_cope_with_it(self):
        """Everything that touches the result must also be able to read it.

        The ``segment``/``label`` columns from pyannote are deliberately not
        carried across the process boundary — this test confirms that
        really nobody needs them.
        """
        from whisper_tm.core.postprocessing import overlapping_intervals
        from whisper_tm.core.speaker import normalize_turns

        df = diar_process.as_dataframe([(0.0, 2.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")])

        # 1. Overlap marker
        assert overlapping_intervals(df) == [(1.0, 2.0)]

        # 2. Voice vectors
        turns = normalize_turns(df)
        assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
        assert turns[0]["start"] == 0.0 and turns[0]["end"] == 2.0

        # 3. Speaker assignment in the transcript
        import whisperx

        result = whisperx.assign_word_speakers(
            df, {"segments": [{"start": 0.1, "end": 0.9, "words": []}]}
        )
        assert result["segments"][0]["speaker"] == "SPEAKER_00"


# ── The time limit ───────────────────────────────────────────────────


class TestDeadline:
    def test_short_files_get_the_lower_bound(self):
        """For a 30 s note, a 15 s limit would be nonsense — loading the
        model alone costs 5 s."""
        from whisper_tm.core.transcription import _diar_deadline

        assert _diar_deadline(30) == 900.0

    def test_long_files_scale_along(self):
        from whisper_tm.core.transcription import _diar_deadline

        # 7 h recording → half the audio duration as the limit.
        assert _diar_deadline(25200) == pytest.approx(12600.0)

    def test_factor_zero_turns_the_limit_off(self, monkeypatch):
        from whisper_tm.config import config
        from whisper_tm.core.transcription import _diar_deadline

        real = config.get

        def faked(*path, default=None):
            if path == ("hardware", "diarization_deadline_factor"):
                return 0
            return real(*path, default=default)

        monkeypatch.setattr(config, "get", faked)
        assert _diar_deadline(3600) == float("inf")

    def test_the_measured_duration_is_far_below_the_limit(self):
        """Proves that the limit is generous, rather than forcing speed.

        Measured on a consumer GPU: about 75x real time, i.e. 2 h of speech
        in roughly 96 s.
        """
        from whisper_tm.core.transcription import _diar_deadline

        measured, duration = 96.0, 7200.0
        assert _diar_deadline(duration) > measured * 30


# ── The wiring in the pipeline ──────────────────────────────────────
#
# Up to here, the WORKER is verified. What was still missing: that
# `transcription.py::_diarize` uses it correctly too — the place where
# the pipeline and the child process meet. Checking that against the real
# model needs a GPU, so the wiring is verified here with doubles.


class _JobDummy:
    cancel = False

    def update(self, **kw):
        pass


class _JobManagerDummy:
    """`extensions.job_manager` is None outside the app."""

    def __init__(self):
        self.lines = []

    def log(self, job, text):
        self.lines.append(text)

    def warn(self, job, text):
        self.lines.append(f"WARN {text}")


@pytest.fixture
def pipeline_environment(monkeypatch):
    """Make `_diarize` callable: job manager and timing."""
    from whisper_tm.core import transcription

    jm = _JobManagerDummy()
    monkeypatch.setattr(transcription, "job_manager", jm)
    return jm


class TestWiringInThePipeline:
    def test_the_normal_path_goes_through_the_worker(self, monkeypatch, pipeline_environment):
        """Result is a DataFrame with exactly the three columns."""
        from whisper_tm.core import transcription

        seen = {}

        class Worker:
            alive = False

            def start_it(self, job=None):
                seen["started"] = True

            def diarize_it(
                self, wav, min_speakers=None, max_speakers=None, job=None, deadline_s=None
            ):
                seen.update(wav=str(wav), min=min_speakers, max=max_speakers, deadline=deadline_s)
                return [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]

        monkeypatch.setattr(diar_process, "take", lambda *a, **k: Worker())

        df = transcription._diarize(
            "/audio/audio.wav",
            2,
            4,
            "cuda",
            None,
            job=_JobDummy(),
            total_dur=7200.0,
            load_done=lambda *a: None,
            t_load=0.0,
        )

        assert list(df.columns) == ["start", "end", "speaker"]
        assert len(df) == 2
        assert seen["started"] is True
        assert seen["wav"] == "/audio/audio.wav"
        assert seen["min"] == 2 and seen["max"] == 4
        # The limit must come from the audio duration, not out of nowhere.
        assert seen["deadline"] == pytest.approx(3600.0)

    def test_a_hang_comes_through_as_a_timeout(self, monkeypatch, pipeline_environment):
        """The error must not be swallowed along the way.

        Otherwise the job would just say "diarization failed" with no
        indication that a time limit kicked in — and nobody would know that
        the limit is set too tight.
        """
        from whisper_tm.core import transcription

        class Worker:
            alive = True

            def start_it(self, job=None):
                pass

            def diarize_it(self, *a, **k):
                raise TimeoutError("Diarization: without response after 900s")

        monkeypatch.setattr(diar_process, "take", lambda *a, **k: Worker())

        with pytest.raises(TimeoutError, match="without response"):
            transcription._diarize(
                "/audio/audio.wav",
                None,
                None,
                "cuda",
                None,
                job=_JobDummy(),
                total_dur=60.0,
                load_done=lambda *a: None,
                t_load=0.0,
            )

    def test_a_running_worker_reports_resident(self, monkeypatch, pipeline_environment):
        """Without this line, the log would look as if the model reloads every time."""
        from whisper_tm.core import transcription

        class Worker:
            alive = True

            def start_it(self, job=None):
                pass

            def diarize_it(self, *a, **k):
                return [(0.0, 1.0, "SPEAKER_00")]

        monkeypatch.setattr(diar_process, "take", lambda *a, **k: Worker())
        transcription._diarize(
            "/audio/a.wav",
            None,
            None,
            "cuda",
            None,
            job=_JobDummy(),
            total_dur=60.0,
            load_done=lambda *a: None,
            t_load=0.0,
        )

        assert any("resident" in z for z in pipeline_environment.lines)

    def test_the_emergency_exit_runs_in_its_own_process_and_warns(
        self, monkeypatch, pipeline_environment
    ):
        """`hardware.diarization_subprocess: false` is the old way.

        It must work — and it must SAY that a hang is then no longer
        cancellable. A silent fallback to the broken state would be worse
        than no switch at all.
        """
        import whisperx.diarize

        from whisper_tm.config import config
        from whisper_tm.core import transcription

        real = config.get

        def faked(*path, default=None):
            if path == ("hardware", "diarization_subprocess"):
                return False
            return real(*path, default=default)

        monkeypatch.setattr(config, "get", faked)

        # The worker must not be touched at all.
        def no_worker(*a, **k):
            raise AssertionError("the fallback used the subprocess anyway")

        monkeypatch.setattr(diar_process, "take", no_worker)

        class Pipeline:
            def __init__(self, *a, **k):
                pass

            def __call__(self, audio, **kw):
                import pandas as pd

                return pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

        monkeypatch.setattr(whisperx.diarize, "DiarizationPipeline", Pipeline)
        monkeypatch.setattr(transcription, "_resident_model", lambda key, load_fn, **k: load_fn())
        import whisperx

        monkeypatch.setattr(whisperx, "load_audio", lambda p: [0.0])

        df = transcription._diarize(
            "/audio/a.wav",
            None,
            None,
            "cpu",
            None,
            job=_JobDummy(),
            total_dur=60.0,
            load_done=lambda *a: None,
            t_load=0.0,
        )

        assert len(df) == 1
        assert any("not cancellable" in z for z in pipeline_environment.lines), (
            "the fallback stays silent about its cost"
        )


class TestSettingsAreComplete:
    """No orphaned setting: read, write, take effect — and in the UI.

    Project house rule. For these two keys a gap would be especially
    annoying: `diarization_subprocess` without a UI would be an emergency
    exit that nobody would find when it actually matters.
    """

    @pytest.mark.parametrize(
        "key",
        [
            "diarization_subprocess",
            "diarization_deadline_factor",
        ],
    )
    def test_present_in_config_code_and_interface(self, key):
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        yaml_text = (repo / "config.default.yaml").read_text(encoding="utf-8")
        py_text = (repo / "whisper_tm" / "core" / "transcription.py").read_text(encoding="utf-8")
        js_text = "".join(
            (repo / "whisper_tm" / "static" / "js" / name).read_text(encoding="utf-8")
            for name in ("settings.js", "settings_pages.js", "settings_tools.js")
        )

        assert f"{key}:" in yaml_text, "missing in config.default.yaml"
        assert f'"{key}"' in py_text, "not read anywhere"
        assert f"hardware.{key}" in js_text, "missing on the settings page"

    def test_the_default_is_the_safe_path(self):
        """By default, the CANCELLABLE variant must run."""
        from whisper_tm.config import config

        assert config.get("hardware", "diarization_subprocess", default=True) is True


# ── Measurement points: where does time go WITHIN a request? ─────────────


class TestMeasurementPoints:
    """The request timing says "a few minutes" and nothing else.

    A `/api/speaker/sample/<id>` request can take minutes while the ffmpeg
    call inside it is logged at around a hundred milliseconds. The rest
    lies between two log lines — so every step needs to be timed.
    """

    def test_every_step_lands_separately_in_the_report(self):
        from whisper_tm.core import debug

        mp = debug.MeasurePoints("Sample")
        mp.point("first")
        mp.point("second")
        total = mp.done()

        names = {z["label"] for z in debug.timings()}
        assert "Sample · first" in names
        assert "Sample · second" in names
        assert "Sample" in names, "the total time is missing"
        assert total >= 0

    def test_the_total_time_covers_the_steps(self):
        """Otherwise the line would be self-contradictory."""
        import time

        from whisper_tm.core import debug

        mp = debug.MeasurePoints("Sample2")
        time.sleep(0.02)
        a = mp.point("sleep")
        total = mp.done()
        assert a >= 0.015
        assert total >= a

    def test_without_debug_mode_nothing_is_logged_but_measured(self, caplog):
        """The measurement always runs — otherwise you'd have to reproduce
        the bug first, after already turning debug mode on."""
        from whisper_tm.core import debug

        was_enabled = debug._enabled
        debug._enabled = False
        try:
            with caplog.at_level("DEBUG"):
                mp = debug.MeasurePoints("Quiet")
                mp.point("step")
                mp.done()
            assert not [r for r in caplog.records if "Quiet" in r.getMessage()]
        finally:
            debug._enabled = was_enabled
        assert "Quiet" in {z["label"] for z in debug.timings()}


# ── Memory cap for the long-lived worker ─────────────────────────


class TestMemoryCap:
    """The long-lived process must be bounded.

    In practice it can grow to **around 10 GiB** after several very long
    recordings and stay there.

    It is not the audio track — numpy returns large arrays via ``mmap``
    immediately (measured: several GiB drop to almost nothing after ``del``). It is cached
    tensors from pyannote and torch, and no library gives those up
    voluntarily; even ``malloc_trim`` did not help.

    So the process gets thrown away when needed: 5 s restart against an
    OOM killer that would take down a very long file and possibly the server
    with it.
    """

    def _with_limit(self, monkeypatch, value):
        from whisper_tm.config import config

        real = config.get

        def faked(*path, default=None):
            if path == ("hardware", "diarization_worker_max_gb"):
                return value
            return real(*path, default=default)

        monkeypatch.setattr(config, "get", faked)

    def test_above_the_limit_the_worker_is_discarded(self, tmp_path, monkeypatch, cleanup):
        a = diar_process.Worker(command=_dummy(tmp_path, "good_d1", WELL_BEHAVED))
        cleanup.append(a)
        a.diarize_it(tmp_path / "a.wav")
        assert a.alive
        proc = a._proc

        self._with_limit(monkeypatch, 0.000001)  # any size is above this
        a.diarize_it(tmp_path / "b.wav")

        assert not a.alive, "the worker keeps running despite exceeding the limit"
        assert proc.poll() is not None

    def test_below_the_limit_it_stays_alive(self, tmp_path, monkeypatch, cleanup):
        """Otherwise the whole point of the long-lived process would be gone."""
        a = diar_process.Worker(command=_dummy(tmp_path, "good_d2", WELL_BEHAVED))
        cleanup.append(a)
        self._with_limit(monkeypatch, 999.0)
        a.diarize_it(tmp_path / "a.wav")
        pid = a._proc.pid
        a.diarize_it(tmp_path / "b.wav")
        assert a.alive and a._proc.pid == pid

    def test_limit_zero_turns_the_cap_off(self, tmp_path, monkeypatch, cleanup):
        a = diar_process.Worker(command=_dummy(tmp_path, "good_d3", WELL_BEHAVED))
        cleanup.append(a)
        self._with_limit(monkeypatch, 0)
        a.diarize_it(tmp_path / "a.wav")
        pid = a._proc.pid
        a.diarize_it(tmp_path / "b.wav")
        assert a.alive and a._proc.pid == pid

    def test_the_result_comes_back_despite_a_restart(self, tmp_path, monkeypatch, cleanup):
        """The cap kicks in AFTER the response — the job must not get lost
        just because cleanup happens afterward."""
        a = diar_process.Worker(command=_dummy(tmp_path, "good_d4", WELL_BEHAVED))
        cleanup.append(a)
        self._with_limit(monkeypatch, 0.000001)
        segments = a.diarize_it(tmp_path / "a.wav")
        assert segments == [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")]

    def test_rss_is_measured_at_all(self, tmp_path, cleanup):
        a = diar_process.Worker(command=_dummy(tmp_path, "good_d5", WELL_BEHAVED))
        cleanup.append(a)
        a.start_it()
        assert a._rss_gb() > 0, "the child's memory usage is not being read"
        a.terminate("Test")
        assert a._rss_gb() == 0.0

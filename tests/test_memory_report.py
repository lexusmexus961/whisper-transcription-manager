"""The server process now measures its own memory.

The diarization worker measures itself and gets discarded at 6 GiB
(`diar_process._memory_cap`) — the SERVER PROCESS, in which all other
processing runs, was never measured. After a long run with many jobs
its RSS and swap can be high, and whether that is a baseline or a trend
cannot be determined without records.

For a run over several days, that's exactly the question that must be answered
before the machine swaps. The CHANGE per job is the
value that matters, not the absolute level: models stay intentionally loaded, a
baseline of 7 GiB is normal — a plus of 0.4 GiB after each job
would not be.
"""

from __future__ import annotations

import inspect

from whisper_tm.core import debug


class TestMeasurement:
    def test_own_memory_is_actually_found(self):
        rss, swap = debug.own_memory()
        # A running Python process always uses something.
        assert rss > 0.01, f"RSS unbelievable: {rss}"
        assert swap >= 0

    def test_value_is_in_gibibytes(self):
        """A sign error in the unit (kB instead of GiB) would only
        be noticed if someone reads 7,400,000 in the log line."""
        rss, _ = debug.own_memory()
        assert 0.01 < rss < 200, f"{rss} doesn't match any unit"


class TestReport:
    def test_first_report_has_no_change_yet(self):
        debug._last_rss = None
        t = debug.memory_report()
        assert "GiB in RAM" in t
        assert "since the last job" not in t

    def test_second_report_names_the_change(self):
        debug._last_rss = None
        debug.memory_report()
        t = debug.memory_report()
        assert "since the last job" in t
        # Sign always present — "0.00" without sign would look like an error.
        assert "Δ +" in t or "Δ -" in t

    def test_change_calculates_against_previous_value(self):
        debug._last_rss = 1.0
        t = debug.memory_report()
        rss, _ = debug.own_memory()
        expected = f"{rss - 1.0:+.2f}"
        assert expected in t, t

    def test_without_swap_nothing_about_swap_appears(self):
        """Otherwise every line would have a "0.00 GiB swapped" — noise
        in a line that should be readable at a glance."""
        debug._last_rss = None
        real = debug.own_memory
        debug.own_memory = lambda: (3.0, 0.0)
        try:
            assert "swapped" not in debug.memory_report()
            debug.own_memory = lambda: (3.0, 2.5)
            assert "2.50 GiB swapped" in debug.memory_report()
        finally:
            debug.own_memory = real

    def test_without_measurement_no_report(self):
        """On a system without /proc no line "0.00 GiB" should appear
        — a made-up zero is worse than no information."""
        real = debug.own_memory
        debug.own_memory = lambda: (0.0, 0.0)
        try:
            assert debug.memory_report() == ""
        finally:
            debug.own_memory = real


class TestWiring:
    def test_pipeline_writes_report_to_job_log(self):
        from whisper_tm.core.transcription import transcription_worker

        source = inspect.getsource(transcription_worker)
        assert "debug.memory_report()" in source
        assert "Server memory" in source

    def test_error_during_measurement_does_not_fail_the_job(self):
        """The line is incidental. A job that fails on its own memory
        message after completed work would be absurd."""
        from whisper_tm.core.transcription import transcription_worker

        source = inspect.getsource(transcription_worker)
        position = source.index("debug.memory_report()")
        # Far enough that the warning threshold fits between:
        # it stands in the SAME try block, and that's what the test holds.
        context = source[position - 300 : position + 900]
        assert "try:" in context and "except Exception" in context
        assert "memory_limit_reached()" in context, "warning must be in same protection"


class TestWarningThreshold:
    """A net, not an alarm.

    A few small increases in a row can look like a leak. Over more jobs
    they average out near zero with large deviation — a plateau with
    noise. Extrapolating from too few data points is an error.

    The threshold remains as a net: it costs nothing, and if a later
    refactoring does collect something, it will speak up before the
    machine swaps. A value that merely sits in a log line among thousands
    of others warns nobody.
    """

    def _with(self, monkeypatch, rss, swap, threshold=20.0):
        monkeypatch.setattr(debug, "own_memory", lambda: (rss, swap))
        from whisper_tm import config as cfgmod

        real = cfgmod.config.get

        def mocked(*parts, default=None):
            if parts[-1] == "server_memory_warning_gb":
                return threshold
            return real(*parts, default=default)

        monkeypatch.setattr(cfgmod.config, "get", mocked)
        return debug.memory_limit_reached()

    def test_below_threshold_no_word(self, monkeypatch):
        assert self._with(monkeypatch, 6.0, 3.0) == ""

    def test_swapped_out_counts_too(self, monkeypatch):
        """A process with 12 GiB RSS and 9 GiB in swap occupies 21, not 12.
        Checking only RSS would silence the warning exactly when the
        machine is already swapping."""
        assert self._with(monkeypatch, 12.0, 9.0) != ""
        assert self._with(monkeypatch, 12.0, 3.0) == ""

    def test_warning_names_the_remedy(self, monkeypatch):
        t = self._with(monkeypatch, 15.0, 6.0)
        assert "restart" in t
        assert "21.0 GiB" in t

    def test_zero_disables_threshold(self, monkeypatch):
        assert self._with(monkeypatch, 30.0, 10.0, threshold=0) == ""

    def test_empty_means_a_share_of_the_machine(self, monkeypatch):
        """A fixed 20 GB never fired on a 16 GB machine (it swapped first)
        and fired after every job on a large one whose media index alone
        holds more."""
        from types import SimpleNamespace

        import psutil

        def machine(gib):
            monkeypatch.setattr(
                psutil, "virtual_memory", lambda: SimpleNamespace(total=gib * 1024**3)
            )

        machine(16)
        t = self._with(monkeypatch, 9.0, 1.0, threshold="")  # above 9.6
        assert t and "10 GiB it is warned at" in t
        machine(64)
        assert self._with(monkeypatch, 22.0, 6.0, threshold="") == ""  # below 38.4
        assert self._with(monkeypatch, 40.0, 0.0, threshold="") != ""

    def test_a_number_of_its_own_wins(self, monkeypatch):
        assert self._with(monkeypatch, 9.0, 0.0, threshold=8) != ""
        assert self._with(monkeypatch, 9.0, 0.0, threshold="8") != ""

    def test_garbage_is_automatic(self, monkeypatch):
        from types import SimpleNamespace

        import psutil

        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=64 * 1024**3))
        assert self._with(monkeypatch, 22.0, 0.0, threshold="lots") == ""

    def test_without_measurement_no_warning(self, monkeypatch):
        assert self._with(monkeypatch, 0.0, 0.0) == ""

    def test_warning_goes_to_all_three_places(self):
        """Job log, server log, and banner in the UI."""
        from whisper_tm.core.transcription import transcription_worker

        source = inspect.getsource(transcription_worker)
        position = source.index("memory_limit_reached()")
        body = source[position : position + 500]
        assert "job_manager.log(" in body
        assert "job_manager.system_warning(" in body
        assert "logger.warning(" in body


class TestFloor:
    """A single value at job end is not a leak indicator.

    Consecutive jobs can swing by several GiB down, up and down
    again. It hits a random point
    in the heap's ups and downs. The FLOOR — the smallest minimum ever seen —
    rises only when memory is truly held persistently.
    """

    def _sequence(self, monkeypatch, values):
        """Run through a sequence of (rss, swap) and collect the floors."""
        debug._last_rss = None
        debug._floor = None
        out = []
        for rss, swap in values:
            monkeypatch.setattr(debug, "own_memory", lambda r=rss, s=swap: (r, s))
            debug.memory_report()
            out.append(debug._floor)
        return out

    def test_fluctuation_does_not_raise_floor(self, monkeypatch):
        """Exactly the observed case: up, down, up — no leak."""
        floors = self._sequence(
            monkeypatch, [(6.0, 3.0), (5.0, 3.0), (8.0, 3.0), (6.5, 3.0), (7.0, 3.0)]
        )
        assert floors == [9.0, 8.0, 8.0, 8.0, 8.0]

    def test_real_holding_raises_floor(self, monkeypatch):
        """A leak looks different: the minimum moves along."""
        floors = self._sequence(monkeypatch, [(5.0, 0.0), (6.0, 0.0), (7.0, 0.0), (8.0, 0.0)])
        assert floors == [5.0, 5.0, 5.0, 5.0], "nothing is held yet here"
        floors = self._sequence(
            monkeypatch, [(5.0, 0.0), (4.0, 0.0), (6.0, 0.0), (5.5, 0.0), (7.0, 0.0), (6.5, 0.0)]
        )
        # The minimum stays at 4 — even an upward trend raises the floor
        # only when the process never comes back down.
        assert min(floors) == 4.0 and floors[-1] == 4.0

    def test_floor_accounts_for_swapped_out(self, monkeypatch):
        """A process whose RSS drops because the OS pushed pages to
        swap hasn't released anything."""
        floors = self._sequence(monkeypatch, [(8.0, 1.0), (5.0, 4.0)])
        assert floors == [9.0, 9.0], "the reshuffling was counted as release"

    def test_floor_appears_in_the_line(self, monkeypatch):
        debug._last_rss = None
        debug._floor = None
        monkeypatch.setattr(debug, "own_memory", lambda: (6.0, 3.0))
        assert "Floor 9.00 GiB" in debug.memory_report()

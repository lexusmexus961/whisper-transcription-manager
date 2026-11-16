"""
Tests for core/debug.py — the central debug switch with timing.
"""

import logging
import time
from pathlib import Path

import pytest

from whisper_tm.config import config
from whisper_tm.core import debug


@pytest.fixture(autouse=True)
def _clean():
    if not config.is_loaded:
        config.load()
    debug.reset()
    yield
    debug.reset()


def _row(label: str) -> dict:
    """One measurement by its label. The table is the process's: a worker
    an earlier test's app left running (the LLM queue polls the database —
    every poll a timed transaction) may add rows of its own mid-test, so
    the checks look for theirs instead of counting all."""
    return {r["label"]: r for r in debug.timings()}[label]


# ── Timing ────────────────────────────────────────────────────────────────


def test_timed_records_a_measurement():
    with debug.timed("test step"):
        time.sleep(0.02)

    rows = {r["label"]: r for r in debug.timings()}
    assert "test step" in rows
    assert rows["test step"]["count"] == 1
    assert rows["test step"]["total_sec"] >= 0.02


def test_measurements_aggregate():
    for _ in range(3):
        debug.record("repeated", 0.5)

    row = {r["label"]: r for r in debug.timings()}["repeated"]
    assert row["count"] == 3
    assert row["total_sec"] == pytest.approx(1.5, abs=0.01)
    assert row["avg_sec"] == pytest.approx(0.5, abs=0.01)
    assert row["max_sec"] == pytest.approx(0.5, abs=0.01)


def test_timing_survives_an_exception():
    """Even a failed step must be measured — otherwise the exact one that
    causes trouble is missing from the evaluation."""
    with pytest.raises(ValueError), debug.timed("broken"):
        time.sleep(0.01)
        raise ValueError("on purpose")

    assert _row("broken")["count"] == 1


def test_measurement_runs_even_when_debug_is_off():
    """The measurement always runs (costs nothing) — only the OUTPUT depends
    on the switch. Otherwise you'd have to restart the server to see what
    was slow."""
    assert debug.is_enabled() is False
    debug.record("also without debug", 1.0)
    assert _row("also without debug")["count"] == 1


def test_summary_is_sorted_by_total_time():
    debug.record("fast", 0.1)
    debug.record("slow", 10.0)
    debug.record("medium", 1.0)

    labels = [r["label"] for r in debug.timings() if r["label"] in ("slow", "medium", "fast")]
    assert labels == ["slow", "medium", "fast"]
    assert "slow" in debug.summary()


def test_fmt_secs():
    assert debug.fmt_secs(0.05) == "50ms"
    assert debug.fmt_secs(2.5) == "2.50s"
    assert debug.fmt_secs(90) == "1.5min"
    assert debug.fmt_secs(7200) == "2.00h"


# ── GPU VRAM measurement ──────────────────────────────────────────────────


def test_fmt_mb():
    assert debug.fmt_mb(0) == "0 MB"
    assert debug.fmt_mb(512) == "512 MB"
    assert debug.fmt_mb(1024) == "1.00 GB"
    assert debug.fmt_mb(6144) == "6.00 GB"
    assert debug.fmt_mb(None) == "?"


@pytest.fixture
def _fake_gpu(monkeypatch):
    """Debug on + a simulated GPU whose usage we control."""
    monkeypatch.setattr(debug, "_enabled", True)
    monkeypatch.setattr(debug, "gpu_available", lambda: True)
    monkeypatch.setattr(debug, "gpu_static", lambda: {0: {"name": "Fake GPU", "total_mb": 12288}})
    # changing "used" values that the sampler sees one after another
    state = {"used": [2000, 5000, 8000, 3000]}

    def _used():
        vals = state["used"]
        return {0: vals.pop(0) if len(vals) > 1 else vals[0]}

    monkeypatch.setattr(debug, "gpu_mem_used_mb", _used)
    return state


def test_vram_records_peak_and_delta(_fake_gpu):
    with debug.vram("Test model", poll=0.001):
        time.sleep(0.02)  # give the sampler time to see the peak

    stats = {r["label"]: r for r in debug.vram_stats()}
    assert "Test model" in stats
    row = stats["Test model"]
    assert row["gpu"] == 0
    assert row["gpu_name"] == "Fake GPU"
    # Peak is the maximum of the seen values, baseline = first value (2000)
    assert row["peak_mb"] >= 5000
    assert row["delta_mb"] >= 3000
    assert row["total_mb"] == 12288


def test_vram_keeps_the_maximum_across_blocks(_fake_gpu):
    """'always the max': a second, smaller run must not lower the peak."""
    with debug.vram("Model X", poll=0.001):
        time.sleep(0.02)
    first = {r["label"]: r for r in debug.vram_stats()}["Model X"]["peak_mb"]

    # second run with consistently smaller values
    _fake_gpu["used"] = [1000, 1500, 1200]
    with debug.vram("Model X", poll=0.001):
        time.sleep(0.02)
    second = {r["label"]: r for r in debug.vram_stats()}["Model X"]["peak_mb"]
    assert second == first  # maximum is preserved


def test_vram_is_noop_when_debug_off(monkeypatch):
    monkeypatch.setattr(debug, "_enabled", False)
    monkeypatch.setattr(debug, "gpu_available", lambda: True)
    called = []
    monkeypatch.setattr(debug, "gpu_mem_used_mb", lambda: called.append(1) or {0: 1})
    with debug.vram("must not measure") as vr:
        pass
    assert vr.note() == ""
    assert debug.vram_stats() == []
    assert called == []  # no sampling without debug mode


def test_vram_summary_lists_models(_fake_gpu):
    with debug.vram("MelBand Roformer", poll=0.001):
        time.sleep(0.02)
    text = debug.vram_summary()
    assert "MelBand Roformer" in text
    assert "GPU VRAM per model" in text


def test_reset_clears_vram(_fake_gpu):
    with debug.vram("throwaway", poll=0.001):
        time.sleep(0.02)
    assert debug.vram_stats()
    debug.reset()
    assert debug.vram_stats() == []


# ── Switch & log file ─────────────────────────────────────────────────────


def test_env_var_enables_debug(monkeypatch):
    monkeypatch.setenv("WHISPER_TM_DEBUG", "1")
    assert debug._wanted(cli_debug=False) is True


def test_config_enables_debug(monkeypatch):
    monkeypatch.delenv("WHISPER_TM_DEBUG", raising=False)
    config.set("logging", "debug", True)
    try:
        assert debug._wanted(cli_debug=False) is True
    finally:
        config.set("logging", "debug", False)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("WHISPER_TM_DEBUG", raising=False)
    config.set("logging", "debug", False)
    assert debug._wanted(cli_debug=False) is False


def test_setup_writes_a_log_file(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_TM_DEBUG", raising=False)
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        path = debug.setup(cli_debug=True, log_dir=tmp_path)
        # With splitting (default in debug mode) the main file sits in
        # one folder per run: <log_dir>/whisper_tm_<time>/main.log
        assert path is not None
        assert path.name == "main.log"
        assert path.parent.parent == tmp_path
        assert debug.log_dir() == path.parent

        logging.getLogger("whisper_tm").debug("hello from the test")
        for h in root.handlers:
            h.flush()

        assert "hello from the test" in path.read_text(encoding="utf-8")
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])
        debug._enabled = False


def test_old_logs_are_pruned(tmp_path):
    for i in range(8):
        f = tmp_path / f"whisper_tm_2026010{i}_120000.log"
        f.write_text("x")
        # different mtime, so the sorting is unambiguous
        import os

        os.utime(f, (1000 + i, 1000 + i))

    debug._prune_old_logs(tmp_path, keep=3)
    assert len(list(tmp_path.glob("whisper_tm_*.log"))) == 3


# ── Section structure in the log ──────────────────────────────────────────


def test_section_header_names_the_phase_and_the_file(caplog):
    with caplog.at_level(logging.INFO, logger="whisper_tm.pipeline"):
        debug.section(
            "Speech recognition (WhisperX)",
            subject="Folder/Subfolder/My Recording.mp4",
            step=4,
            total=6,
            Model="large-v3",
            Language="de",
        )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "▶ [4/6] Speech recognition (WhisperX)" in text
    assert "My Recording.mp4" in text  # filename is in the header
    assert "Folder/Subfolder" not in text  # ... but not the whole path
    assert "Model: large-v3" in text


def test_section_done_reports_duration_and_outcome(caplog):
    with caplog.at_level(logging.INFO, logger="whisper_tm.pipeline"):
        debug.section_done("Alignment", 34.2)
        debug.section_done("Diarization", 108.0, ok=False, note="no token")

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "✔ Alignment · 34.20s" in text
    assert "✖ Diarization · 1.8min · no token" in text


def test_banner_shows_the_full_path(caplog):
    """The full path belongs in the banner ONCE — the phases below it show
    only the filename, otherwise the same 200-character path keeps repeating."""
    with caplog.at_level(logging.INFO, logger="whisper_tm.pipeline"):
        debug.banner("QUEUE 13/1500", "Videos/2024/Recording.mp4", Size="3.5 GB")

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "QUEUE 13/1500" in text
    assert "Videos/2024/Recording.mp4" in text
    assert "Size: 3.5 GB" in text


def test_short_name_keeps_the_end():
    """When shortening, the end must be preserved — that's where the filename is."""
    long = "a" * 100 + "IMPORTANT.mp4"
    out = debug.short_name(long, width=30)
    assert len(out) == 30
    assert out.endswith("IMPORTANT.mp4")
    assert out.startswith("…")


def test_noisy_libs_cannot_reenable_themselves():
    """Third-party libraries set their logger level THEMSELVES on import.

    lightning is only imported on the first job (whisperx → lightning), and
    thereby undoes a setLevel() from server startup — its INFO messages
    ended up in the middle of the pipeline sections. The filter is
    therefore attached to the handler and can't be worked around that way.
    """
    noise = debug._NoiseFilter()

    lib = logging.getLogger("lightning.pytorch.utilities.migration.utils")
    lib.setLevel(logging.DEBUG)  # that's what the library does

    def rec(logger_name, level):
        return logging.LogRecord(logger_name, level, "x", 1, "msg", None, None)

    assert noise.filter(rec("lightning.pytorch.utilities", logging.INFO)) is False
    assert noise.filter(rec("matplotlib.font_manager", logging.INFO)) is False
    # Warnings and errors ALWAYS come through — even from noisy libraries
    assert noise.filter(rec("lightning.pytorch", logging.WARNING)) is True
    # Our own output and domain-relevant ones (whisperx, pyannote) stay
    assert noise.filter(rec("whisper_tm.pipeline", logging.INFO)) is True
    assert noise.filter(rec("whisperx.vads.pyannote", logging.INFO)) is True


# ── Splitting logs by topic ────────────────────────────────────────────────


@pytest.fixture
def _log_environment(tmp_path, monkeypatch):
    """Run setup() and restore root logging afterward.

    Without the reset, this test's FileHandlers stay attached to the root
    logger and write into every subsequent test.
    """
    root = logging.getLogger()
    old_handlers, old_level = list(root.handlers), root.level
    monkeypatch.setattr(
        config,
        "get",
        _config_with(
            config,
            {
                "logging.debug": True,
                "logging.split": True,
            },
        ),
    )
    yield tmp_path
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    for h in old_handlers:
        root.addHandler(h)
    root.setLevel(old_level)
    debug.stop_call_tracing()


def _config_with(cfg, value_list):
    """Replace config.get with a variant with overrides."""
    orig = cfg.get

    def fake(*keys, default=None):
        flat = ".".join(keys)
        if flat in value_list:
            return value_list[flat]
        return orig(*keys, default=default)

    return fake


def _run_folder(tmp_path):
    return sorted(tmp_path.glob("whisper_tm_*"))[-1]


def test_split_creates_one_file_per_category(_log_environment):
    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)
    d = _run_folder(tmp_path)
    assert d.is_dir()
    names = {f.name for f in d.iterdir()}
    assert "main.log" in names
    for expected in (
        "db.log",
        "queue.log",
        "pipeline.log",
        "scanner.log",
        "api.log",
        "rag.log",
        "trace.log",
        "http.log",
    ):
        assert expected in names, f"{expected} missing"


def test_lines_land_in_the_right_file(_log_environment):
    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)
    logging.getLogger("whisper_tm.database").debug("DBLINE")
    logging.getLogger("whisper_tm.scanner.filesystem").debug("SCANLINE")
    logging.getLogger("whisper_tm").debug("HTTPLINE", extra={"http": True})
    for h in logging.getLogger().handlers:
        h.flush()

    d = _run_folder(tmp_path)
    assert "DBLINE" in (d / "db.log").read_text()
    assert "SCANLINE" not in (d / "db.log").read_text()
    assert "SCANLINE" in (d / "scanner.log").read_text()
    assert "HTTPLINE" in (d / "http.log").read_text()
    # HTTP must NOT additionally end up in a topic file
    assert "HTTPLINE" not in (d / "api.log").read_text()
    # main.log keeps everything
    main_part = (d / "main.log").read_text()
    assert all(s in main_part for s in ("DBLINE", "SCANLINE", "HTTPLINE"))


def test_whisperx_prefix_does_not_swallow_own_code(_log_environment):
    """'whisperx' is a prefix of 'whisper_tm'.

    Without the dot in the rule, all of our own code ended up in
    pipeline.log and the other files stayed empty.
    """
    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)
    logging.getLogger("whisper_tm.database").debug("OWNCODE")
    logging.getLogger("whisperx.asr").debug("FOREIGNCODE")
    for h in logging.getLogger().handlers:
        h.flush()
    d = _run_folder(tmp_path)
    assert "OWNCODE" not in (d / "pipeline.log").read_text()
    assert "FOREIGNCODE" in (d / "pipeline.log").read_text()


def test_whisperx_does_not_hijack_its_logger(_log_environment):
    """whisperx sets ``propagate = False`` — otherwise it's missing from EVERY log.

    ``whisperx/log_utils.py`` calls a ``setup_logging()`` on the first
    ``get_logger``, which gives the ``whisperx`` logger its own stdout
    handler, sets the level to INFO, and **turns off propagation to the
    root**. That meant no whisperx message reached the log files anymore —
    even a WARNING was missing from main.log AND pipeline.log; it only
    appeared on stdout.

    For long batch runs this is not a cosmetic flaw: library
    warnings concerning a transcription would be invisible.
    ``debug.setup`` reverses that and preempts a later import
    (NullHandler).

    The test imports whisperx DELIBERATELY before the ``setup()`` call —
    that's exactly the order in which it broke.
    """
    import whisperx.diarize  # noqa: F401  (triggers whisperx' setup_logging)

    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)

    lg = logging.getLogger("whisperx")
    assert lg.propagate is True, "whisperx messages don't reach the root"

    logging.getLogger("whisperx.diarize").warning("WHISPERX-WARNING")
    for h in logging.getLogger().handlers:
        h.flush()
    d = _run_folder(tmp_path)
    assert "WHISPERX-WARNING" in (d / "main.log").read_text()
    assert "WHISPERX-WARNING" in (d / "pipeline.log").read_text()


# ── Call tracing ──────────────────────────────────────────────────────────


def test_call_tracing_records_calls(_log_environment):
    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)
    debug.start_call_tracing(with_arguments=True)
    try:
        debug.fmt_secs(1.5)  # a real function from the package
    finally:
        debug.stop_call_tracing()
    for h in logging.getLogger().handlers:
        h.flush()
    assert not debug.call_tracing_active()


def test_trace_is_kept_off_the_console(_log_environment, capsys):
    """The trace must not flood the terminal."""
    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)
    logging.getLogger("whisper_tm.trace").debug("TRACELINE")
    logging.getLogger("whisper_tm.database").debug("NORMALLINE")
    output = capsys.readouterr().err
    assert "TRACELINE" not in output
    assert "NORMALLINE" in output


def test_shortening_never_dumps_whole_payloads():
    """A transcript dict in the log would be megabytes big."""
    assert debug._brief([1] * 10_000) == "<list n=10000>"
    assert debug._brief(b"x" * 5000) == "<bytes 5000 B>"
    long_text = debug._brief("y" * 5000)
    assert len(long_text) <= 121 and long_text.endswith("…")


def test_module_name_is_not_dunder_init():
    """database/__init__.py must not appear as '__init__'."""
    import whisper_tm

    debug._trace_package_path = str(Path(whisper_tm.__file__).parent)
    debug._trace_module_names.clear()
    assert debug._module_name(f"{debug._trace_package_path}/database/__init__.py") == "database"
    assert debug._module_name(f"{debug._trace_package_path}/core/queue.py") == "core.queue"


def test_single_file_when_split_is_off(tmp_path, monkeypatch):
    """Without splitting it stays a single file as before."""
    monkeypatch.delenv("WHISPER_TM_DEBUG", raising=False)
    monkeypatch.setattr(
        config,
        "get",
        _config_with(
            config,
            {
                "logging.debug": True,
                "logging.split": False,
            },
        ),
    )
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        path = debug.setup(cli_debug=True, log_dir=tmp_path)
        assert path.parent == tmp_path
        assert path.name.startswith("whisper_tm_") and path.suffix == ".log"
        assert debug.log_dir() is None
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])


def test_pruning_removes_old_runs_of_both_shapes(tmp_path):
    """Pruning must catch individual files AND run folders.

    Otherwise whoever switches the setting would have a form that
    never gets cleaned up.
    """
    import time as _t

    for i in range(3):
        (tmp_path / f"whisper_tm_old{i}.log").write_text("x")
        _t.sleep(0.01)
    for i in range(3):
        d = tmp_path / f"whisper_tm_folder{i}"
        d.mkdir()
        (d / "main.log").write_text("x")
        _t.sleep(0.01)

    debug._prune_old_logs(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("whisper_tm_*"))
    assert len(remaining) == 2, remaining
    # The two most recent ones are folders
    assert all((tmp_path / n).is_dir() for n in remaining)


def test_trace_hook_survives_interpreter_shutdown():
    """On shutdown, CPython sets module globals to None.

    But the hook keeps running as long as it's installed. Observed for real
    on Ctrl-C: every shutdown ended with
      TypeError: startswith first arg must be str …, not NoneType
    raised out of the logging teardown.
    """
    import sys as _sys

    old_path, old_active = debug._trace_package_path, debug._trace_active
    try:
        debug._trace_active = True
        debug._trace_package_path = None  # this is how it looks during teardown
        frame = _sys._getframe()
        # Must NOT throw
        debug._trace_hook(frame, "call", None)
        debug._trace_hook(frame, "return", None)
    finally:
        debug._trace_package_path, debug._trace_active = old_path, old_active


def test_trace_hook_is_inert_when_stopped():
    """A disabled trace must not log anything anymore."""
    import sys as _sys

    debug.stop_call_tracing()
    assert not debug.call_tracing_active()
    debug._trace_hook(_sys._getframe(), "call", None)  # no effect, no error


def test_stop_is_registered_for_exit(monkeypatch):
    """The hook must be removed again before the globals get torn down."""
    registered = []
    monkeypatch.setattr("atexit.register", lambda fn, *a: registered.append((fn, a)))
    monkeypatch.setattr(debug, "_trace_active", False)
    try:
        debug.start_call_tracing(with_arguments=False)
        assert any(fn is debug.stop_call_tracing for fn, _ in registered)
    finally:
        debug.stop_call_tracing()


def test_size_cap_is_clamped_to_free_disk(tmp_path, monkeypatch):
    """The limit must never be bigger than the free space.

    A truncated trace is a smaller problem than a system disk that's
    written full.
    """
    import shutil as _sh

    monkeypatch.setattr(debug, "_log_dir_run", tmp_path)
    monkeypatch.setattr(debug, "_trace_active", False)
    monkeypatch.setattr(
        config,
        "get",
        _config_with(
            config,
            {
                "logging.trace_calls_max_mb": 100_000,  # 100 GB requested
                "logging.trace_calls_max": 0,
            },
        ),
    )
    # 10 GB free -> cap at one-fifth = 2 GB
    monkeypatch.setattr(_sh, "disk_usage", lambda p: _sh._ntuple_diskusage(10**10, 0, 10**10))
    try:
        debug.start_call_tracing(with_arguments=False)
        assert debug._trace_max_bytes == 10**10 // 5
    finally:
        debug.stop_call_tracing()


def test_no_event_limit_by_default(tmp_path, monkeypatch):
    """Two million events were gone before the first transcription."""
    monkeypatch.setattr(debug, "_log_dir_run", None)
    monkeypatch.setattr(debug, "_trace_active", False)
    try:
        debug.start_call_tracing(with_arguments=False)
        assert debug._trace_limit == 0
    finally:
        debug.stop_call_tracing()


def test_durations_survive_generators(tmp_path, monkeypatch):
    """Generators must not throw off the timing.

    A stack assumes strict nesting — generators violate it (every next() a
    'call', every yield a 'return'). That produced durations of minutes
    for a function that takes microseconds, across hundreds of thousands
    of generator calls.
    """
    import re

    root = logging.getLogger()
    old_ones, level = list(root.handlers), root.level
    monkeypatch.setattr(
        config,
        "get",
        _config_with(
            config,
            {
                "logging.debug": True,
                "logging.split": True,
                "logging.trace_calls_max": 0,
                "logging.trace_calls_max_mb": 0,
            },
        ),
    )
    try:
        debug.setup(cli_debug=True, log_dir=tmp_path)
        debug.start_call_tracing(with_arguments=False)
        # A real package function that runs through a generator
        for _ in range(50):
            debug.fmt_secs(0.5)
            list(debug.timings())
        debug.stop_call_tracing()
        for h in root.handlers:
            h.flush()

        text = (_run_folder(tmp_path) / "trace.log").read_text()
        sec = [float(m) for m in re.findall(r"← \S+ ([\d.]+)s ", text)]
        # Nothing in this test takes even remotely close to a second.
        assert not [s for s in sec if s > 1.0], f"nonsensical durations: {sec[:5]}"
    finally:
        debug.stop_call_tracing()
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        for h in old_ones:
            root.addHandler(h)
        root.setLevel(level)


def test_trace_is_not_duplicated_into_main_log(_log_environment):
    """The trace must not duplicate main.log.

    Otherwise a run's trace takes twice its size on disk, because main.log
    captures the same content again. With tight disk space that's pure
    waste — the trace is already fully in trace.log.
    """
    tmp_path = _log_environment
    debug.setup(cli_debug=True, log_dir=tmp_path)
    logging.getLogger("whisper_tm.trace").debug("TRACEENTRY")
    logging.getLogger("whisper_tm.database").debug("DBENTRY")
    for h in logging.getLogger().handlers:
        h.flush()

    d = _run_folder(tmp_path)
    assert "TRACEENTRY" in (d / "trace.log").read_text()
    main_part = (d / "main.log").read_text()
    assert "TRACEENTRY" not in main_part, "trace duplicated in main.log"
    assert "DBENTRY" in main_part, "main.log must keep everything else"


def test_every_logger_has_a_category():
    """Every logger used in the code must end up in a category.

    The bug behind this: the pipeline logs under `whisper_tm.pipeline`,
    but the categories listed module paths like
    `whisper_tm.core.transcription`. Result: pipeline.log stayed empty,
    even though a transcription writes 63 lines. Check the logger NAME,
    not the module path.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "whisper_tm"
    names = set()
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        names.update(re.findall(r'getLogger\("(whisper_tm[^"]*)"\)', text))
        # Modules that use getLogger(__name__) carry their module path
        if re.search(r"getLogger\(__name__\)", text):
            rel = py.relative_to(root.parent).with_suffix("")
            parts = [t for t in rel.parts if t != "__init__"]
            names.add(".".join(parts))

    all_prefixes = tuple(p for _, ps in debug._CATEGORIES for p in ps)
    # These deliberately end up in main.log only — no topic of their own.
    without_category = {
        "whisper_tm",
        "whisper_tm.hardware",
        "whisper_tm.config",
        "whisper_tm.extensions",
    }

    unassigned = sorted(
        n for n in names if not n.startswith(all_prefixes) and n not in without_category
    )
    assert not unassigned, (
        f"Logger without a category file: {unassigned}. Either add to "
        f"_CATEGORIES or list above as deliberately uncategorized."
    )


def test_the_log_folder_is_the_one_setup_used(tmp_path, monkeypatch):
    """The servers the app starts write next to its log (logs_root)."""
    monkeypatch.setattr(debug, "_logs_root", None)
    debug.setup(cli_debug=False, log_dir=tmp_path / "here")
    assert debug.logs_root() == tmp_path / "here"


def test_without_setup_the_log_folder_follows_the_config(tmp_path, monkeypatch):
    """A relative folder lies in the data folder — from the working
    directory, logs-debug/ turned up wherever the server was started, and a
    systemd unit without a WorkingDirectory couldn't log."""
    monkeypatch.setattr(debug, "_logs_root", None)
    monkeypatch.setattr(
        config,
        "get",
        lambda section, key=None, default=None: (
            "relative/logs" if (section, key) == ("logging", "debug_dir") else default
        ),
    )
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    assert debug.logs_root() == tmp_path / "data" / "relative" / "logs"
    absolute = tmp_path / "elsewhere"
    monkeypatch.setattr(
        config,
        "get",
        lambda section, key=None, default=None: (
            str(absolute) if (section, key) == ("logging", "debug_dir") else default
        ),
    )
    assert debug.logs_root() == absolute

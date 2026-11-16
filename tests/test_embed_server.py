"""The embedding server, on demand.

What's checked is the lifecycle, not llama.cpp: start command, startup
with waiting, shutdown on idle, and the cases where something goes
wrong. Instead of a 4 GB model, a tiny Python script runs as the server —
that keeps the tests fast and makes them work without a GPU too.

The point this is actually about: an embedding server that doesn't
reliably shut back DOWN is worse than none at all. It holds VRAM on
exactly the card the batch run works on, and you only notice it once a
transcription fails from too little memory.
"""

from __future__ import annotations

import os
import pathlib
import socket
import sys
import textwrap
import time
from contextlib import suppress

import pytest

from whisper_tm.config import config
from whisper_tm.core import embed_server

# A mini server that answers /health with 200 — that's all the
# lifecycle needs. `--delay` delays readiness, `--crash`
# makes it crash on startup.
MINI_SERVER = textwrap.dedent("""
    import sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    delay = float(sys.argv[sys.argv.index('--delay')+1]) if '--delay' in sys.argv else 0.0
    if '--crash' in sys.argv:
        sys.stderr.write('deliberately failed\\n')
        raise SystemExit(3)
    port = int(sys.argv[sys.argv.index('--port')+1])
    time.sleep(delay)
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def log_message(self, *a): pass
    HTTPServer(('127.0.0.1', port), H).serve_forever()
""")


def _free_port() -> int:
    """A port nothing listens on — a fixed one met a server of its own
    kind still running there, or another test's."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def place(tmp_path, monkeypatch):
    """An EmbedServer that starts the mini script instead of llama-server."""
    script = tmp_path / "miniserver.py"
    script.write_text(MINI_SERVER)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not actually a model")

    monkeypatch.setattr(embed_server, "binary", lambda: script)
    monkeypatch.setattr(embed_server, "model_file", lambda: model)
    monkeypatch.setattr(embed_server, "base_directory", lambda: tmp_path)

    # The real code's start command doesn't fit a Python script —
    # only the call itself is replaced here, the lifecycle stays real.
    def command(extra=()):
        return [sys.executable, str(script), "--port", str(embed_server._port()), *extra]

    # The signature must stay that of the real `_start_command`, otherwise
    # this scaffolding checks a call that doesn't actually exist.
    monkeypatch.setattr(embed_server, "_start_command", lambda force_cpu=False: command())

    old_value = dict(config.get("embedding") or {})
    config.set("embedding", "local_port", _free_port())
    config.set("embedding", "local_start_timeout_sec", 20)
    config.set("embedding", "local_idle_stop_sec", 300)

    s = embed_server.EmbedServer()
    yield s, command, monkeypatch
    s.stop_it()
    for k, v in old_value.items():
        config.set("embedding", k, v)


# ── The start command ────────────────────────────────────────────────────


def test_start_command_without_a_model_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(embed_server, "binary", lambda: tmp_path / "x")
    monkeypatch.setattr(embed_server, "model_file", lambda: None)
    assert embed_server._start_command() == []


def test_start_command_contains_what_is_needed(monkeypatch, tmp_path):
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    argv = embed_server._start_command()
    # Without --embedding, llama.cpp answers /v1/embeddings with 501.
    assert "--embedding" in argv
    # Ubatch MUST be set: non-causal models need the whole
    # sequence in one pass, otherwise the request is rejected.
    assert "--ubatch-size" in argv
    assert argv[0] == str(b)


def test_gpu_layers_are_NOT_prescribed_by_default(monkeypatch, tmp_path):
    """A fixed `--n-gpu-layers 99` kills the server.

    llama.cpp adjusts the layer count itself to the FREE VRAM — but only
    as long as nobody dictates the value. With a fixed value it aborts:

        W common_fit_params: failed to fit params to free device memory:
                             n_gpu_layers already set by user to 99, abort
        E cudaMalloc failed: out of memory (4075 MiB)

    On a card shared by Whisper, pyannote and the embedding model, little
    free memory is the normal case. That's why the default must not
    disable the adjustment.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    # DICTATE free VRAM instead of inheriting it from the machine.
    #
    # Without this, the test depended on the state of the card: if a
    # transcription is currently running, little is free, the VRAM
    # protection kicks in and sets `--n-gpu-layers 0` — the test would
    # then fail, even though both behaviors are correct. This happened in
    # practice, in the middle of a batch run.
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 12000)

    config.set("embedding", "local_gpu_layers", -1)
    assert "--n-gpu-layers" not in embed_server._start_command()

    # Explicit values must still come through.
    config.set("embedding", "local_gpu_layers", 0)
    argv = embed_server._start_command()
    assert argv[argv.index("--n-gpu-layers") + 1] == "0", "CPU mode doesn't work"
    config.set("embedding", "local_gpu_layers", 20)
    argv = embed_server._start_command()
    assert argv[argv.index("--n-gpu-layers") + 1] == "20"
    config.set("embedding", "local_gpu_layers", -1)


def test_pooling_is_set_on_every_start_path(monkeypatch, tmp_path):
    """Search starts the server on the CPU, an index run on the GPU (or on
    the CPU while the card is busy). The pooling decides how a text's token
    vectors become its ONE vector — set on only some paths, questions and
    stored chunks are pooled differently, and search quality drops without
    any error. An earlier version's CPU path returned before the pooling (and
    the extra arguments) were added."""
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)
    monkeypatch.setattr(embed_server, "_job_running", lambda: False)
    old = {
        k: config.get("embedding", k)
        for k in ("local_pooling", "local_extra_args", "local_gpu_layers")
    }
    config.set("embedding", "local_gpu_layers", -1)
    config.set("embedding", "local_pooling", "mean")
    config.set("embedding", "local_extra_args", "--threads 4")
    try:
        paths = {
            "GPU": embed_server._start_command(),
            "CPU for search": embed_server._start_command(force_cpu=True),
        }
        monkeypatch.setattr(embed_server, "_job_running", lambda: True)
        paths["CPU while a job runs"] = embed_server._start_command()
        monkeypatch.setattr(embed_server, "_job_running", lambda: False)
        config.set("embedding", "local_gpu_layers", 0)
        paths["CPU by hand"] = embed_server._start_command()
        for name, argv in paths.items():
            assert "--pooling" in argv, f"{name}: no pooling"
            assert argv[argv.index("--pooling") + 1] == "mean", name
            assert argv[-2:] == ["--threads", "4"], f"{name}: extra args lost"
        # The CPU paths still say CPU — exactly once
        for name in ("CPU for search", "CPU while a job runs", "CPU by hand"):
            argv = paths[name]
            assert argv.count("--n-gpu-layers") == 1, name
            assert argv[argv.index("--n-gpu-layers") + 1] == "0", name
        assert "--n-gpu-layers" not in paths["GPU"]

        # Empty = the model's own pooling, and that on every path alike
        config.set("embedding", "local_pooling", "")
        assert "--pooling" not in embed_server._start_command(force_cpu=True)
    finally:
        for k, v in old.items():
            config.set("embedding", k, v)


def test_no_fallback_to_just_any_gguf(monkeypatch, tmp_path):
    """The chat model must NOT be used as an embedder.

    It would deliver vectors that look like embeddings — the bug would
    then only show up as poor search results, i.e. practically never.
    """
    (tmp_path / "models" / "gguf").mkdir(parents=True)
    (tmp_path / "models" / "gguf" / "chat-model.gguf").write_text("")
    monkeypatch.setattr(embed_server, "base_directory", lambda: tmp_path)
    monkeypatch.setattr(
        embed_server, "_cfg", lambda name, default=None: "" if name == "local_model" else default
    )
    assert embed_server.model_file() is None


def test_endpoint_is_openai_compatible():
    assert embed_server.endpoint().endswith("/v1")


# ── Startup ──────────────────────────────────────────────────────────────


def test_starts_and_becomes_reachable(place):
    s, _command, _mp = place
    ep = s.ensure()
    assert ep.endswith("/v1")
    z = s.state()
    assert z["managed"] and z["reachable"] and z["pid"] > 0


def test_its_log_goes_to_the_log_folder(place, tmp_path):
    """Into ``logging.debug_dir`` (``--log-dir``) like the app's own log —
    it went to logs-debug in the install folder whatever that said."""
    from whisper_tm.core import debug

    s, _command, mp = place
    mp.setattr(debug, "_logs_root", tmp_path / "logs")
    s.ensure()
    log = tmp_path / "logs" / "embed-server.log"
    assert log.is_file() and "=== Start" in log.read_text()


def test_a_second_call_does_not_restart(place):
    s, _command, _mp = place
    s.ensure()
    pid = s.state()["pid"]
    s.ensure()
    assert s.state()["pid"] == pid


def test_waits_until_the_model_is_loaded(place):
    """/health only answers after loading — asking once isn't enough."""
    s, command, mp = place
    mp.setattr(embed_server, "_start_command", lambda force_cpu=False: command(["--delay", "3"]))
    t = time.time()
    s.ensure()
    duration = time.time() - t
    assert s.state()["reachable"]
    assert duration >= 3, "there was no waiting at all"


def test_a_crash_at_startup_reports_the_output(place):
    """A failed start must not run into the timeout."""
    s, command, mp = place
    mp.setattr(embed_server, "_start_command", lambda force_cpu=False: command(["--crash"]))
    t = time.time()
    with pytest.raises(RuntimeError, match="terminated"):
        s.ensure()
    assert time.time() - t < 15, "waited for the timeout instead of noticing"
    assert s.state()["pid"] == 0


def test_a_timeout_leaves_no_process_behind(place):
    """THE important case: after a timeout, nothing may keep running.

    A hanging llama-server holds several GB of VRAM and otherwise is
    only noticed once the next transcription fails because of it.
    """
    s, command, mp = place
    mp.setattr(embed_server, "_start_command", lambda force_cpu=False: command(["--delay", "60"]))
    with pytest.raises(RuntimeError, match="not ready"):
        s.ensure(wait_time=3)
    assert s._proc is None
    assert s.state()["pid"] == 0


def test_missing_pieces_report_themselves_clearly(monkeypatch):
    """If the binary is missing, startup must fail with a clear message.

    `_responds` is disabled, and that's not incidental: without it, the
    test checks whatever happens to be listening on the configured port.
    That's exactly how it once failed in a full suite run — a real
    embedding server was running on that port, `ensure()` found it and
    rightly raised nothing. A test whose result depends on foreign
    processes doesn't measure what it claims to.
    """
    s = embed_server.EmbedServer()
    monkeypatch.setattr(embed_server, "binary", lambda: None)
    monkeypatch.setattr(s, "_responds", lambda timeout=2.0: False)
    with pytest.raises(RuntimeError, match="llama-server not found"):
        s.ensure()


# ── Shutdown ─────────────────────────────────────────────────────────────


def test_stopping_really_ends_it(place):
    s, _b, _mp = place
    s.ensure()
    p = s._proc
    assert s.stop_it() is True
    assert p.poll() is not None, "the process is still alive"
    assert s.state()["managed"] is False


def test_stopping_without_a_start_is_harmless(place):
    s, _b, _mp = place
    assert s.stop_it() is False


def test_a_changed_port_stops_our_old_server_first(place):
    """Port changed in the settings while our own server still runs on the
    old one. `ensure` only looks at the configured address, found nothing
    there, started a second server and overwrote `_proc` — the old one ran
    on unwatched: neither the idle watchdog nor `stop_it` knew it anymore,
    and its own session kept it alive past the app's exit, VRAM and all."""
    s, _command, _mp = place
    s.ensure()
    old = s._proc
    config.set("embedding", "local_port", _free_port())
    try:
        s.ensure()
        assert old.poll() is not None, "the server on the old port still runs"
        assert s._proc is not old and s._proc.poll() is None
        assert s._responds(), "nothing answers on the new port"
        assert s.stop_it() is True
    finally:
        if old.poll() is None:  # leave nothing behind, whatever failed
            old.kill()
            old.wait(timeout=10)


def test_idling_shuts_the_server_down(place):
    """The deadline is the whole point: idle → VRAM back."""
    s, _b, _mp = place
    config.set("embedding", "local_idle_stop_sec", 1)
    _old, embed_server.WATCH_TICK_SEC = embed_server.WATCH_TICK_SEC, 0.3
    try:
        s.ensure()
        assert s.state()["managed"]
        end = time.time() + 15
        while time.time() < end and s.state()["managed"]:
            time.sleep(0.3)
        assert not s.state()["managed"], "the watchdog didn't shut it down"
    finally:
        embed_server.WATCH_TICK_SEC = _old


def test_use_extends_the_deadline(place):
    """A running index run must not shoot down its own server."""
    s, _b, _mp = place
    config.set("embedding", "local_idle_stop_sec", 3)
    _old, embed_server.WATCH_TICK_SEC = embed_server.WATCH_TICK_SEC, 0.3
    try:
        s.ensure()
        end = time.time() + 6
        while time.time() < end:  # regularly "use" it
            s.used()
            time.sleep(0.4)
        assert s.state()["managed"], "the server was shut down despite ongoing use"
    finally:
        embed_server.WATCH_TICK_SEC = _old


def test_zero_turns_the_automatic_off(place):
    s, _b, _mp = place
    config.set("embedding", "local_idle_stop_sec", 0)
    _old, embed_server.WATCH_TICK_SEC = embed_server.WATCH_TICK_SEC, 0.3
    try:
        s.ensure()
        s._last_used = 0.0  # as if it had been unused forever
        time.sleep(2)
        # Typographic quotation marks NOT in the Python string: the
        # closing character would end the literal (a project trap).
        assert s.state()["managed"], "0 must mean 'never shut down'"
    finally:
        embed_server.WATCH_TICK_SEC = _old


def test_a_foreign_server_is_not_shut_down(place):
    """If something is already running there, it's used — but not shot down."""
    s, command, _mp = place
    import subprocess

    foreign = subprocess.Popen(command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.time() + 15
        while time.time() < end and not s._responds():
            time.sleep(0.2)
        assert s._responds(), "the foreign server never came up"

        s.ensure()
        assert s._proc is None, "a second process was started"
        assert s.stop_it() is False
        assert foreign.poll() is None, "the foreign process was terminated"
    finally:
        foreign.terminate()
        foreign.wait(timeout=10)


# ── Availability ─────────────────────────────────────────────────────────


def test_availability_reports_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(embed_server, "binary", lambda: None)
    ok, msg = embed_server.available()
    assert not ok and "llama-server" in msg

    b = tmp_path / "llama-server"
    b.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: None)
    ok, msg = embed_server.available()
    assert not ok and "Model" in msg


# ── The VRAM reserve ─────────────────────────────────────────────────────


def test_the_reserve_pushes_to_the_cpu(monkeypatch, tmp_path):
    """A costly bug, as a test.

    An index run was running alongside a batch run.
    llama.cpp adapted to the free VRAM — and thereby took exactly the
    buffer that separation needed seconds later:

        [12:00:00] VRAM Separation: GPU0 11.6 GB/12.0 GB
        [12:00:06] Separation failed after 2 attempts

    The file ended up as an error in the queue. If too little is free,
    the embedding server belongs on the CPU — slower, but it doesn't
    take anything away from the transcription.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    config.set("embedding", "local_gpu_layers", -1)
    config.set("embedding", "local_vram_reserve_mb", 3000)
    try:
        monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 1200)
        argv = embed_server._start_command()
        assert argv[argv.index("--n-gpu-layers") + 1] == "0", (
            "the GPU was used despite too little free VRAM"
        )

        # Enough free → llama.cpp is allowed to decide for itself again.
        monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 9000)
        assert "--n-gpu-layers" not in embed_server._start_command()
    finally:
        config.set("embedding", "local_vram_reserve_mb", 3000)


def test_without_nvidia_smi_no_assumption(monkeypatch, tmp_path):
    """No nvidia-smi does NOT mean 'no memory free'.

    Otherwise the server would run in CPU mode for no reason on every
    machine without an NVIDIA card — and on one with an AMD card, the
    reasoning behind it would simply be wrong.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: -1)
    config.set("embedding", "local_gpu_layers", -1)
    assert "--n-gpu-layers" not in embed_server._start_command()


def test_reserve_zero_turns_the_check_off(monkeypatch, tmp_path):
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 10)
    config.set("embedding", "local_gpu_layers", -1)
    config.set("embedding", "local_vram_reserve_mb", 0)
    try:
        assert "--n-gpu-layers" not in embed_server._start_command()
    finally:
        config.set("embedding", "local_vram_reserve_mb", 3000)


def test_free_vram_handles_missing_tools(monkeypatch):
    def broken(*a, **k):
        raise FileNotFoundError("nvidia-smi doesn't exist here")

    monkeypatch.setattr(embed_server.subprocess, "run", broken)
    assert embed_server._free_vram_mb() == -1


def test_the_end_of_the_process_takes_the_server_along(tmp_path):
    """Whoever starts the server must shut it down — even a script.

    At first, the registration only lived in `create_app()`. The app was
    thereby covered, but no tool was: a script using `make_embed_fn()`
    would start the server, run through, and leave it running. This
    happened in practice — the llama-server afterward had PPid 1
    and kept holding VRAM. The idle watchdog didn't help: it's a daemon
    thread and dies with its process.

    So what's checked isn't whether something is registered in `atexit`
    (that can't be read out cleanly), but what actually counts: a
    dedicated process starts the server, ends normally — and afterward
    nothing is still alive.
    """
    import json
    import subprocess as sp

    script = tmp_path / "miniserver.py"
    script.write_text(MINI_SERVER)
    model = tmp_path / "m.gguf"
    model.write_text("")

    port = _free_port()
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(f"""
        import json, pathlib, sys
        sys.path.insert(0, {str(pathlib.Path.cwd())!r})
        from whisper_tm.config import config
        from whisper_tm.core import debug, embed_server
        debug._logs_root = pathlib.Path({str(tmp_path)!r})
        config.set("embedding", "local_port", {port})
        config.set("embedding", "local_start_timeout_sec", 25)
        config.set("embedding", "local_idle_stop_sec", 0)   # never on its own
        embed_server._start_command = lambda force_cpu=False: [
            sys.executable, {str(script)!r}, "--port", "{port}"]
        embed_server.SERVER.ensure()
        print(json.dumps({{"pid": embed_server.SERVER.state()["pid"]}}))
    """)
    )

    off = sp.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=120)
    assert off.returncode == 0, off.stderr[-600:]
    pid = json.loads(off.stdout.strip().splitlines()[-1])["pid"]
    assert pid > 0

    # The driver has ended. The server must have gone down with it.
    end = time.time() + 15
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return  # gone, as it should be
        time.sleep(0.5)
    # Not gone: clean up so the test leaves nothing behind, then report.
    with suppress(ProcessLookupError):
        os.kill(pid, 9)
    raise AssertionError(
        f"The server (pid {pid}) kept running after the process ended — "
        f"exactly the orphaned state that holds VRAM."
    )


def test_with_tight_vram_the_server_goes_to_the_cpu(monkeypatch, tmp_path):
    """The other half of the same rule — and untested until now.

    The embedding server shares the card with Whisper and pyannote. If
    little is free there, it must NOT occupy it too, but compute on the
    CPU instead. This exact case once cost a running transcription
    (CUDA out of memory), because the server started in the free moment
    between two steps and took the card.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    config.set("embedding", "local_gpu_layers", -1)
    config.set("embedding", "local_vram_reserve_mb", 3000)

    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 1200)
    argv = embed_server._start_command()
    assert argv[argv.index("--n-gpu-layers") + 1] == "0", (
        "with 1.2 GB free it must fall back to the CPU"
    )

    # And without information about free memory, it does NOT guess:
    # llama.cpp then adapts on its own.
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: -1)
    assert "--n-gpu-layers" not in embed_server._start_command()


def test_a_running_job_forces_the_server_to_the_cpu(monkeypatch, tmp_path):
    """**A finding from a real log.**

    The reserve didn't help, because it measures the free VRAM EXACTLY at
    the moment it's largest — between two steps:

        12:00:00.0  Diarization worker shuts down (10 GB free)
        12:00:00.5  Embedding server ready after 2.0s
        12:00:01.3  Queue: VRAM exhausted — file back in the queue

    Whoever is currently using the card will also need it again right
    away. A running job therefore beats any snapshot.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    config.set("embedding", "local_gpu_layers", -1)
    # Plenty of free memory — the reserve alone would allow the GPU.
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)

    monkeypatch.setattr(embed_server, "_job_running", lambda: False)
    assert "--n-gpu-layers" not in embed_server._start_command(), (
        "without a running job, llama.cpp decides for itself"
    )

    monkeypatch.setattr(embed_server, "_job_running", lambda: True)
    argv = embed_server._start_command()
    assert argv[argv.index("--n-gpu-layers") + 1] == "0", (
        "during a running transcription, the card belongs to the job"
    )


def test_an_explicit_value_is_left_untouched(monkeypatch, tmp_path):
    """Whoever sets the layers by hand means it that way — then nothing
    gets overridden, not even by a running job."""
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    monkeypatch.setattr(embed_server, "_job_running", lambda: True)
    config.set("embedding", "local_gpu_layers", 20)
    argv = embed_server._start_command()
    assert argv[argv.index("--n-gpu-layers") + 1] == "20"
    config.set("embedding", "local_gpu_layers", -1)


def test_the_question_about_the_job_never_throws(monkeypatch):
    """`_job_running` is asked in the start path. If it raises, the server
    doesn't start at all anymore — so when in doubt, do NOT block."""
    import whisper_tm.core.queue as _q

    def broken(*a, **k):
        raise RuntimeError("queue not reachable")

    monkeypatch.setattr(_q, "_lock", property(broken), raising=False)
    assert embed_server._job_running() in (True, False)


def test_in_cpu_mode_the_card_is_hidden(monkeypatch):
    """`--n-gpu-layers 0` alone is NOT enough.

    Measured: the server started that way still occupies
    about 1.9 GB VRAM — the GPU-capable binary creates a
    CUDA context at startup whether it needs one or not. On a card
    where transcription is fighting over a hundred megabytes, that's
    not nothing.
    """
    env = embed_server._environment(on_gpu=False)
    assert env.get("CUDA_VISIBLE_DEVICES") == "", "the card is still visible"

    # In GPU mode, the environment must NOT be touched — otherwise the
    # server starts where it belongs, without a card. (The variable from the
    # shell running the tests would decide otherwise.)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env2 = embed_server._environment(on_gpu=True)
    assert "CUDA_VISIBLE_DEVICES" not in env2 or env2["CUDA_VISIBLE_DEVICES"] != ""


# ── Search no longer touches the card ─────────────────────────────────────


def test_the_search_starts_on_the_cpu(monkeypatch, tmp_path):
    """Measured with real chunks:

        Embedding a question (search)     CPU ~200 ms  ← imperceptible
        Embedding chunks (index)          CPU 0.2/s vs GPU ~18/s

    For search, the CPU is thus free to use, for embedding it's a
    factor of 80 too slow. The gain isn't speed: a search taking
    several GB of the card while a job is running causes CUDA OOMs.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    config.set("embedding", "local_gpu_layers", -1)
    # Best conditions for the GPU: lots free, no job.
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)
    monkeypatch.setattr(embed_server, "_job_running", lambda: False)

    argv = embed_server._start_command(force_cpu=True)
    assert argv[argv.index("--n-gpu-layers") + 1] == "0"
    # … and the index run remains unaffected by it.
    assert "--n-gpu-layers" not in embed_server._start_command()


def test_the_switch_can_be_turned_off(monkeypatch, tmp_path):
    """A toggle that only knows one direction isn't one."""
    config.set("embedding", "local_search_on_cpu", False)
    try:
        assert embed_server._search_on_cpu() is False
    finally:
        config.set("embedding", "local_search_on_cpu", True)
    assert embed_server._search_on_cpu() is True


def test_purpose_search_reaches_the_start_command(monkeypatch, tmp_path):
    """The wiring, not the computation: `ensure(purpose='search')`
    must actually build the command differently. A parameter that gets
    lost along the way looks correct in every individual test.

    `_responds` is explicitly denied instead of relying on a free port.
    The test otherwise failed exactly when a REAL embedding server was
    running on the configured port (e.g. that of a running instance of
    the application): `ensure` then uses it and never gets past it
    to the start command. A test that depends on a foreign process
    measures that process's state, not the code.
    """
    b = tmp_path / "llama-server"
    b.write_text("")
    m = tmp_path / "m.gguf"
    m.write_text("")
    monkeypatch.setattr(embed_server, "binary", lambda: b)
    monkeypatch.setattr(embed_server, "model_file", lambda: m)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)
    monkeypatch.setattr(embed_server, "_job_running", lambda: False)

    seen = {}
    real = embed_server._start_command

    def remember(force_cpu=False):
        seen["cpu"] = force_cpu
        return []  # empty → ensure aborts right after

    monkeypatch.setattr(embed_server, "_start_command", remember)

    server = embed_server.EmbedServer()
    monkeypatch.setattr(server, "_responds", lambda timeout=2.0: False)
    for purpose, expected in (("search", True), ("index", False)):
        with pytest.raises(RuntimeError):
            server.ensure(purpose=purpose)
        assert seen["cpu"] is expected, purpose
    assert real is not remember


def test_a_running_server_is_not_moved(monkeypatch, tmp_path):
    """Search starts on the CPU — but it doesn't FORCE anyone there.

    If the server is already running on the card, say for an index run,
    a move would be more expensive than the 100 ms it's supposed to
    save: the model would have to be reloaded, and the index run would
    then keep running on the CPU afterward.
    """
    server = embed_server.EmbedServer()
    monkeypatch.setattr(server, "_responds", lambda timeout=2.0: True)
    monkeypatch.setattr(server, "_own_process_alive", lambda: True)
    monkeypatch.setattr(server, "_start_watchdog", lambda: None)
    monkeypatch.setattr(embed_server, "_job_running", lambda: False)
    server._on_gpu = True

    def must_not(*a, **k):
        raise AssertionError("the running server was restarted")

    monkeypatch.setattr(embed_server, "_start_command", must_not)

    assert server.ensure(purpose="search") == embed_server.endpoint()
    assert server._on_gpu is True


def test_the_index_run_brings_the_server_back_from_the_cpu(monkeypatch, tmp_path):
    """**A bug that a follow-up check itself found.**

    `ensure` uses any server that answers. After search starts it
    up on the CPU, it stays there for five minutes (the idle deadline).
    An index run in that window would have used it too and computed at
    0.2 instead of 18 chunks/s — a factor of 90, and nobody would have
    noticed: there's no error message, the run is just days longer.

    The rule for the opposite direction (a GPU server yields to a job)
    already existed; this one was missing.
    """
    server = embed_server.EmbedServer()
    monkeypatch.setattr(server, "_responds", lambda timeout=2.0: True)
    monkeypatch.setattr(server, "_own_process_alive", lambda: True)
    monkeypatch.setattr(server, "_start_watchdog", lambda: None)
    monkeypatch.setattr(embed_server, "_job_running", lambda: False)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)
    config.set("embedding", "local_gpu_layers", -1)

    stopped = []
    monkeypatch.setattr(server, "stop_it", lambda: stopped.append(True))
    monkeypatch.setattr(embed_server, "_start_command", lambda force_cpu=False: [])

    server._on_gpu = False
    with pytest.raises(RuntimeError):  # empty start command
        server.ensure(purpose="index")
    assert stopped, "the CPU server kept running, the index run would be 90× slower"

    # SEARCH does NOT bring it back — it's properly at home on the CPU.
    stopped.clear()
    server._on_gpu = False
    assert server.ensure(purpose="search") == embed_server.endpoint()
    assert not stopped


def test_it_only_moves_when_the_card_is_free(monkeypatch, tmp_path):
    """A move into uncertainty can cost a transcription.

    Three situations in which the index run must leave the CPU server
    alone: a running job, too little free memory, and a manually set
    `local_gpu_layers` (then the user has decided).
    """
    monkeypatch.setattr(embed_server, "_job_running", lambda: False)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)
    config.set("embedding", "local_gpu_layers", -1)
    config.set("embedding", "local_vram_reserve_mb", 3000)
    assert embed_server._gpu_is_free() is True

    monkeypatch.setattr(embed_server, "_job_running", lambda: True)
    assert embed_server._gpu_is_free() is False

    monkeypatch.setattr(embed_server, "_job_running", lambda: False)
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 1200)
    assert embed_server._gpu_is_free() is False

    # Not determinable (-1) does NOT mean "free".
    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: -1)
    assert embed_server._gpu_is_free() is False

    monkeypatch.setattr(embed_server, "_free_vram_mb", lambda: 11000)
    config.set("embedding", "local_gpu_layers", 0)
    assert embed_server._gpu_is_free() is False, "the user wanted the CPU"
    config.set("embedding", "local_gpu_layers", -1)


def test_two_callers_during_startup_start_ONE_server(place):
    """Background: the lock covered the start but not the wait for
    /health. A second caller (a search while the indexer starts the server)
    found "not answering yet, process alive" and started a SECOND server,
    overwriting ``_proc`` — the first then ran on unwatched, holding VRAM
    past the app's exit."""
    import threading

    s, command, monkeypatch = place
    starts = []

    def counted(force_cpu=False):
        starts.append(1)
        return command(["--delay", "1.5"])

    monkeypatch.setattr(embed_server, "_start_command", counted)
    errors = []

    def call():
        try:
            s.ensure()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(2)]
    threads[0].start()
    time.sleep(0.3)  # the first is waiting for /health
    threads[1].start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(starts) == 1
    assert s._own_process_alive()


# ── A CPU server never sees the card ───────────────────────────────────────


@pytest.mark.parametrize(
    "argv,on_gpu",
    [
        ([], True),
        (["--n-gpu-layers", "0"], False),
        (["-ngl", "0"], False),
        (["--gpu-layers=0"], False),
        (["--n-gpu-layers", "0", "-ngl", "20"], True),  # the last one counts
        (["-ngl", "99", "--n-gpu-layers", "0"], False),
        (["--device", "none"], False),
        (["-dev", "CUDA0"], True),
        (["-ngl", "auto"], True),
    ],
)
def test_whether_a_command_puts_layers_on_the_card(argv, on_gpu):
    """Read from the whole command, the extra arguments included: a CPU-only
    server set up through `local_extra_args` (`-ngl 0`) counted as a GPU one
    and started with the card visible — the GPU-capable binary then takes
    ≈1.9 GB of CUDA buffers it never uses (measured)."""
    assert embed_server._runs_on_gpu(["llama-server", "--model", "m.gguf", *argv]) is on_gpu


def test_a_cpu_server_from_the_extra_arguments_starts_without_the_card(place):
    s, command, mp = place
    mp.setattr(embed_server, "_start_command", lambda force_cpu=False: command(["-ngl", "0"]))
    seen = {}
    real = embed_server.subprocess.Popen

    def spy(argv, **kw):
        seen["env"] = kw.get("env") or {}
        return real(argv, **kw)

    mp.setattr(embed_server.subprocess, "Popen", spy)
    s.ensure()
    assert seen["env"].get("CUDA_VISIBLE_DEVICES") == "", "the card is still visible"
    assert s._on_gpu is False

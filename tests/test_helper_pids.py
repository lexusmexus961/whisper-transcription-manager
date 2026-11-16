"""Model servers a crash, a kill or a closed terminal left behind are ended
at the next start — only those, and only when they are still the process
that was recorded.

The embedding and rerank servers run in a session of their own and outlived
an app that didn't get to stop them; the next start found the old one
answering on its port and used it as someone else's server, forever — the
reranker kept its ~5 GB on the card.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import time

import pytest

from whisper_tm import __main__ as main_module
from whisper_tm.core import helper_pids

psutil = pytest.importorskip("psutil")


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(helper_pids, "_file", lambda name: tmp_path / f"{name}.pid")
    return tmp_path


def _server(name="llama-server", port="8123"):
    """A process that looks like a llama-server on ``port``."""
    proc = subprocess.Popen(
        [
            "bash",
            "-c",
            f'exec -a {name} "$0" -c "import time; time.sleep(60)" --port {port}',
            sys.executable,
        ],
        start_new_session=True,
    )
    for _ in range(50):
        try:
            if psutil.Process(proc.pid).cmdline()[:1] == [name]:
                break
        except psutil.Error:
            pass
        time.sleep(0.05)
    return proc


def test_a_leftover_is_ended(files):
    proc = _server()
    try:
        helper_pids.record("embed-server", proc.pid, 8123)
        assert helper_pids.end_leftovers() == 1
        assert proc.wait(timeout=10) is not None
        assert not (files / "embed-server.pid").exists()
    finally:
        proc.kill()


def test_another_process_under_the_pid_is_left_alone(files):
    """The pid went to another program meanwhile — or never was ours."""
    other = _server(name="something-else")
    wrong_port = _server(port="9999")
    try:
        helper_pids.record("embed-server", other.pid, 8123)
        helper_pids.record("rerank-server", wrong_port.pid, 8123)
        assert helper_pids.end_leftovers() == 0
        assert other.poll() is None and wrong_port.poll() is None
        assert not list(files.glob("*.pid"))
        # A recorded start time that isn't this process's
        mine = _server()
        try:
            helper_pids.record("embed-server", mine.pid, 8123)
            entry = json.loads((files / "embed-server.pid").read_text())
            entry["started"] -= 60
            (files / "embed-server.pid").write_text(json.dumps(entry))
            assert helper_pids.end_leftovers() == 0 and mine.poll() is None
        finally:
            mine.kill()
    finally:
        other.kill()
        wrong_port.kill()


def test_a_regular_stop_forgets_it(files):
    helper_pids.record("rerank-server", 4242, 8123)
    helper_pids.forget("rerank-server")
    assert helper_pids.end_leftovers() == 0


def test_the_servers_record_and_the_start_ends_leftovers():
    from whisper_tm.core import embed_server, rerank_server

    assert 'helper_pids.record("embed-server"' in inspect.getsource(embed_server)
    assert 'helper_pids.forget("embed-server")' in inspect.getsource(embed_server)
    assert 'helper_pids.record("rerank-server"' in inspect.getsource(rerank_server)
    assert 'helper_pids.forget("rerank-server")' in inspect.getsource(rerank_server)
    source = inspect.getsource(main_module.main)
    # after the database's lock: only its own server ends what it recorded
    assert source.index("_hold_database(config.db_path)") < source.index(
        "helper_pids.end_leftovers()"
    )
    # a closed terminal shuts down like Ctrl+C — unless started with nohup
    assert '"SIGHUP"' in source and "SIG_IGN" in source

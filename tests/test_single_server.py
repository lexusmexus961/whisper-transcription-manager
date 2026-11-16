"""One server per database and port (__main__._hold_database, _port_taken).

A second server started on the same database re-queued the entries the
first one was working on and worked them a second time; its port was only
bound at the very end, after the queue had taken up work. Now it refuses to
start while another live server holds the database, and asks for the port
first.
"""

from __future__ import annotations

import fcntl
import socket
import subprocess
import sys

import pytest

from whisper_tm import __main__ as main


@pytest.fixture(autouse=True)
def _release():
    yield
    if main._db_lock is not None:
        main._db_lock.close()
        main._db_lock = None


def test_the_first_server_gets_the_database(tmp_path):
    db = tmp_path / "library.db"
    assert main._hold_database(db) == ""
    lock = tmp_path / "library.db.lock"
    assert lock.exists() and lock.read_text().startswith("pid ")
    # Anyone else (another open of the file) can't take it now
    with open(lock, "a+") as other:
        with pytest.raises(BlockingIOError):
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_a_second_server_is_told_who_has_it(tmp_path):
    db = tmp_path / "library.db"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            f"f = open({str(db) + '.lock'!r}, 'a+'); fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "f.seek(0); f.truncate(); f.write('pid 4242: whisper-tm'); f.flush()\n"
            "print('held', flush=True); time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert main._hold_database(db) == "pid 4242: whisper-tm"
        assert main._db_lock is None
    finally:
        holder.kill()
        holder.wait()
    # The holder gone (also when killed), the database is free again
    assert main._hold_database(db) == ""


def test_a_port_in_use(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert main._port_taken("127.0.0.1", port)
    assert main._port_taken("127.0.0.1", port) == ""


def test_main_asks_before_the_app_is_made():
    import inspect

    src = inspect.getsource(main.main)
    assert src.index("_hold_database(config.db_path)") < src.index("app = create_app(args.config)")
    assert src.index("_port_taken(") < src.index("app = create_app(args.config)")


def test_an_install_without_the_apps_files_refuses_to_start(tmp_path, monkeypatch):
    """A wheel (pip install .) carried neither the pages nor the defaults and
    started a server without a page, on no defaults at all."""
    from whisper_tm import __main__ as main_module

    assert main_module._not_a_checkout() == ""  # this checkout has both
    fake = tmp_path / "site-packages" / "whisper_tm"
    fake.mkdir(parents=True)
    monkeypatch.setattr(main_module, "__file__", str(fake / "__main__.py"))
    said = main_module._not_a_checkout()
    assert "the pages and config.default.yaml" in said and "./setup.sh" in said


def test_the_check_reads_the_config_it_is_given():
    """--check ran before the configuration was loaded: it looked at the
    default folders, whatever --config said."""
    import inspect

    from whisper_tm import __main__ as main_module

    source = inspect.getsource(main_module.main)
    block = source[source.index("if args.check:") : source.index("_run_check(gpu_test")]
    assert "config.load(args.config)" in block

"""Shared test fixtures."""

import os
import shutil

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ffmpeg: runs ffmpeg/ffprobe — skipped where they are missing"
    )


def pytest_runtest_setup(item):
    """A test marked ffmpeg skips without ffmpeg and ffprobe instead of
    failing (CI requires them: pytest_sessionstart)."""
    if item.get_closest_marker("ffmpeg") and not (
        shutil.which("ffmpeg") and shutil.which("ffprobe")
    ):
        pytest.skip("needs ffmpeg and ffprobe")


def pytest_sessionstart(session):
    """In CI, node and ffmpeg are part of the run: hundreds of tests skip
    without them (the page's JavaScript, the media steps), and a run that
    skips them is green all the same. Locally they may be missing."""
    if os.environ.get("CI"):
        missing = [tool for tool in ("node", "ffmpeg", "ffprobe") if not shutil.which(tool)]
        if missing:
            pytest.exit(
                f"CI without {', '.join(missing)}: the tests that need them would only skip",
                returncode=4,
            )


def _refuse_real_data(config, home) -> None:
    """Stop the session before a test can touch the real installation.

    Every place the app writes to has to lie under the session's throwaway
    HOME (the XDG folders are pointed there too). One that does not would
    put the real database back in reach, so the suite does not start.
    """
    from pathlib import Path

    home = Path(home).resolve()
    from whisper_tm.core import debug

    for name, path in (
        ("database", config.db_path),
        ("work directory", config.work_dir),
        ("media directory", config.media_dir),
        ("transcript directory", config.transcript_dir),
        ("data folder", config.data_dir),
        ("log folder", debug.logs_root()),
    ):
        p = Path(path).resolve()
        if p != home and home not in p.parents:
            pytest.exit(
                f"Test isolation broken: the {name} would be {p}, "
                f"outside the test HOME {home} — not running "
                f"against real data.",
                returncode=3,
            )


@pytest.fixture(scope="session", autouse=True)
def _no_name_lookups():
    """Tests resolve no host names: an invented one ("https://a/1") was looked
    up for real, and with a slow resolver each lookup took seconds — fifty of
    them made one test run for minutes. Addresses, localhost and this
    machine's own name still resolve; a test that needs more patches
    ``socket.getaddrinfo`` itself."""
    import ipaddress
    import socket

    real = socket.getaddrinfo
    own = {"localhost", socket.gethostname().lower(), socket.getfqdn().lower()}

    def local_only(host, *args, **kwargs):
        name = host.decode("ascii", "replace") if isinstance(host, (bytes, bytearray)) else host
        if name is None or str(name).lower() in own or not str(name):
            return real(host, *args, **kwargs)
        try:
            ipaddress.ip_address(str(name).split("%", 1)[0])
        except ValueError:
            raise socket.gaierror(
                socket.EAI_NONAME, "Name or service not known (tests resolve no names)"
            ) from None
        return real(host, *args, **kwargs)

    socket.getaddrinfo = local_only
    yield
    socket.getaddrinfo = real


@pytest.fixture(autouse=True)
def _fake_downloads_count_as_recordings(request, monkeypatch):
    """The link-download tests write placeholder bytes where yt-dlp would
    write a video; the container check would remove them. Its own test
    (test_ytdl_refuses_non_media) runs the real check."""
    name = request.node.module.__name__.rsplit(".", 1)[-1]
    if name.startswith(("test_ytdl", "test_ingest")) and name != "test_ytdl_refuses_non_media":
        from whisper_tm.core import media_formats

        monkeypatch.setattr(media_formats, "is_recording", lambda p: True)
    yield


@pytest.fixture(autouse=True)
def _no_package_imports_in_checks(request, monkeypatch):
    """The system check imports the pipeline's modules in subprocesses —
    tens of seconds with torch each time. Tests get the result 'all there';
    test_diagnostics_packages runs the real thing."""
    if request.node.module.__name__.rsplit(".", 1)[-1] != "test_diagnostics_packages":
        from whisper_tm.core import diagnostics

        monkeypatch.setattr(diagnostics, "_package_result", [])
    yield


@pytest.fixture(autouse=True)
def _no_failed_logins_carry_over():
    """The guessing brake counts per client address — every test client is
    127.0.0.1, so one test's wrong passwords would pause the next test."""
    from whisper_tm.core import auth

    auth._failures.clear()
    yield
    auth._failures.clear()


@pytest.fixture(autouse=True)
def _no_queue_worker_outlives_its_test():
    """An app made in a test (create_app) starts the real queue workers, and
    they ran on for the rest of the session. A later test that put a pending
    entry into a queue of its own had it taken up by them — now and then,
    depending on the load: a stray "library folder is gone" warning where the
    test expected its own."""
    yield
    import sys

    q = sys.modules.get("whisper_tm.core.queue")
    if q is not None and any(t.is_alive() for t in list(getattr(q, "_workers", []))):
        try:
            q.stop_worker(timeout=5.0)
        except Exception:  # noqa: BLE001 — a test's leftovers must not fail the next one
            pass
    # The LLM queue's worker too: left running, it polled a later test's
    # database through a connection that test closed — a segfault in sqlite
    lq = sys.modules.get("whisper_tm.core.llm_queue")
    worker = getattr(getattr(lq, "llm_queue", None), "_worker", None) if lq else None
    if worker is not None and worker.is_alive():
        try:
            lq.llm_queue.stop()
            worker.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture(scope="session", autouse=True)
def _isolate_user_config(tmp_path_factory):
    """Isolate tests from the user's actual config — and from their data.

    Multiple test files call ``config.load()``. Without this safeguard,
    ``_find_config`` searches for the user's file under ``~/.config/whisper-tm/``
    and merges it with the defaults — the suite then tests not the code,
    but the personal settings of whoever runs it.

    For example, a ``database.max_backups`` changed in the UI from 5 to 25
    would make the rotation tests fail that expect the default 5. A test
    should not depend on what someone clicked in the settings.

    ``WHISPER_TM_CONFIG`` therefore points to a file of its own: it exists (so
    it is taken as the first candidate), and apart from the queue not
    starting on its own and running one lane it brings nothing — the
    defaults from ``config.default.yaml`` apply. One lane: the queue tests
    pin the order and the control of one worker; the lanes have tests of
    their own (tests/test_queue_lanes.py) that ask for more. The log folder
    lies in the throwaway HOME: the servers a test starts write their logs
    there, not next to a live installation's.

    And a throwaway HOME for the whole session. The defaults put the
    database, the work directory and ``~/Videos`` under HOME — the real
    ones, for any test that creates an app without moving HOME itself. Such
    a test running next to a live server would have its ``create_app()``
    import old transcripts from ``~/Videos/.transcripts`` into the real
    database and empty the real work directory under running jobs. The
    queue stays paused
    for the same reason: a test app would pick up whatever a test put into
    its queue, and run it on the GPU. Paused has one way through:
    ``queue.run_now`` lets a new entry run anyway — a test that calls it for
    real owns the workers (tests/test_queue_lanes.py), one that only passes
    through it stubs it (tests/test_ingest.py).
    """
    home = tmp_path_factory.mktemp("home")
    own = tmp_path_factory.mktemp("cfg") / "config.yaml"
    own.write_text(
        "queue:\n  autostart: false\n  lanes: 1\n"
        f"logging:\n  debug_dir: {str(home / 'logs-debug')!r}\n",
        encoding="utf-8",
    )

    import os

    from whisper_tm.config import config

    # The XDG folders go: set in a contributor's shell, they point the data,
    # config and cache folders back at the real ones (platformdirs reads them
    # before HOME). Unset — not pointed at this HOME: then the folders follow
    # HOME, also the one a test sets for a database of its own.
    xdg = ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
    saved = {k: os.environ.get(k) for k in ("WHISPER_TM_CONFIG", "HOME", *xdg)}
    os.environ["WHISPER_TM_CONFIG"] = str(own)
    os.environ["HOME"] = str(home)
    for key in xdg:
        os.environ.pop(key, None)
    # ALWAYS load, don't just discard an already loaded config: multiple
    # DB methods derive identity (abs_path) from ``config.media_dir`` if needed.
    # If the config remained unloaded, the result depended on which test file
    # ran before — ``tests/test_database.py`` failed in isolation (duplicate
    # instead of upsert, skip without status update), but not in the full suite
    # because ``test_auth.py`` had already loaded the config.
    config.load()
    # Pinned as well: a test that loads a config of its own would otherwise
    # put the default, ./logs-debug, back in force
    from whisper_tm.core import debug

    debug._logs_root = home / "logs-debug"
    _refuse_real_data(config, home)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

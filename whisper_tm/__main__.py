"""
Whisper Transcription Manager — Main entry point.

Usage:
    whisper-tm                           # Standard start
    whisper-tm --config /path/config.yaml
    whisper-tm --port 5001 --host 0.0.0.0   # open to the network
    whisper-tm --device cpu              # Force CPU mode
    whisper-tm --media ~/Videos          # Set media directory
    whisper-tm --remap-path OLD NEW      # Translate stored paths after a move
    whisper-tm --hf-token                # Set up speaker diarization (token)
    python -m whisper_tm                 # Alternative
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path


def _console_never_fails() -> None:
    """Characters the console can't show come out as "?" instead of an error.

    A redirected stdout in a narrow encoding (Windows' ANSI code page, a
    Latin-1 locale) has no box-drawing characters: the banner's first line
    ended the start with a UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # not a text stream of its own (a test's capture, None)


#: The lock on the database, held for the server's lifetime (_hold_database)
_db_lock = None


def _hold_database(db_path) -> str:
    """Take the database for this process: "" when it is ours, else who has
    it. A second server on the same database re-queued the first one's
    running entries at its start and worked them a second time. The lock
    goes with the process — also when it is killed: nothing stale is left."""
    global _db_lock
    try:
        import fcntl
    except ImportError:  # no fcntl (native Windows): unchecked
        return ""
    lock_path = Path(str(db_path) + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 — held until the end
    except OSError:
        return ""
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        who = f.read().strip()
        f.close()
        return who or "another process"
    except OSError:  # a filesystem without locks: unchecked
        f.close()
        return ""
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}: {' '.join(sys.argv)}")
    f.flush()
    _db_lock = f
    return ""


def _port_taken(host: str, port: int) -> str:
    """Why the server couldn't listen on ``host``:``port`` — "" when it can.
    Asked before anything starts: binding came last, after the queue had
    already taken up its work."""
    import socket

    family = socket.AF_INET6 if ":" in str(host) else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((str(host), int(port)))
    except OSError as e:
        return str(e.strerror or e)
    return ""


def network_without_login(host: str) -> str:
    """The warning for a server other machines reach with the login off — or
    "". In Docker the container always listens on 0.0.0.0; the published
    address decides there, so nothing is said."""
    from whisper_tm.core.auth import is_auth_enabled
    from whisper_tm.core.commands import in_docker

    if str(host) in ("127.0.0.1", "localhost", "::1") or str(host).startswith("127."):
        return ""
    if is_auth_enabled() or in_docker():
        return ""
    return (
        f"The server listens on {host} with the login off: everyone on the network "
        f"can read every transcript and change or delete the library. Turn on "
        f"auth.enabled, or listen on 127.0.0.1 (docs/GUIDE.md: Network Access)."
    )


def run():
    """The command line: :func:`main`, with a config.yaml that can't be read
    named in two lines instead of a traceback."""
    from whisper_tm.config import ConfigError

    try:
        return main()
    except ConfigError as e:
        print(f"\n{e}\n", file=sys.stderr)
        sys.exit(2)


def main():
    _console_never_fails()
    parser = argparse.ArgumentParser(
        description="Whisper Transcription Manager — Professional Transcription Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisper-tm                                Standard start
  whisper-tm --config ~/config.yaml         Custom configuration
  whisper-tm --port 5001                    Different port
  whisper-tm --device cpu                   CPU mode
  whisper-tm --media /mnt/nas/videos        NAS directory
  whisper-tm --workdir /mnt/ssd/whisper-tm    Fast workdir (a disk, not tmpfs)
  whisper-tm --hf-token                     Set up speaker diarization (Hugging Face token)
  whisper-tm --prefetch                     Download the models the settings need

Moving to another machine (server stopped):
  whisper-tm --show-roots                   Where do the stored paths point?
  whisper-tm --remap-path /mnt/nas /srv/nas             Dry run
  whisper-tm --remap-path /mnt/nas /srv/nas --apply     Translate
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="Server port (default: 9860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: 127.0.0.1, this machine only; 0.0.0.0: the network)",
    )
    parser.add_argument(
        "--media",
        "-m",
        type=str,
        default=None,
        help="Media directory",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        default=None,
        help="Working directory (SSD/NVMe recommended)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        choices=["auto", "cuda", "rocm", "mps", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: log everything, measure all timings, write log file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for this run's log file (default: logging.debug_dir, "
        "logs-debug in the data folder)",
    )
    parser.add_argument(
        "--trace-sql",
        action="store_true",
        help="Additionally log EVERY SQL statement (very verbose; switches debug mode on)",
    )
    parser.add_argument(
        "--trace-calls",
        action="store_true",
        help="Log EVERY function call (arguments, duration, return value). "
        "Goes to trace.log. Costs noticeable speed — for tracing "
        "a specific flow, not for continuous operation.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check hardware, installation and folders, then exit",
    )
    parser.add_argument(
        "--gpu-test",
        action="store_true",
        help="With --check: also compute on the GPU (PyTorch, CTranslate2, "
        "a short Whisper run with the tiny model)",
    )
    parser.add_argument(
        "--hf-token",
        action="store_true",
        help="Set up speaker diarization: ask for the Hugging Face token "
        "(or take it from HF_TOKEN), check that it may download the "
        "gated pyannote model, save it in the config and download the "
        "model, then exit",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download the models the settings need (Whisper, alignment, "
        "NLTK's sentence splitter, the separation model; with "
        "diarization on, its models), then exit — so the first job, or "
        "a machine offline later, has them",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="With --hf-token: never ask, use HF_TOKEN or the configured token",
    )
    parser.add_argument(
        "--show-roots",
        action="store_true",
        help="List the media roots stored in the database and whether they "
        "exist on this machine, then exit",
    )
    parser.add_argument(
        "--remap-path",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="After moving to another machine: translate every stored path "
        "under OLD to NEW (databases and config.yaml). Repeatable. "
        "Dry run unless --apply is given; the server must be stopped.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --remap-path: actually write (after a backup)",
    )
    parser.add_argument(
        "--drop-conflicting",
        action="store_true",
        help="With --remap-path: if the server already ran on the new machine "
        "and stored some files under the new path, delete those rows "
        "and keep the moved ones",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="With --remap-path --apply: skip the backup copies",
    )

    args = parser.parse_args()

    if args.version:
        from whisper_tm import __version__

        print(f"Whisper Transcription Manager v{__version__}")
        sys.exit(0)

    too_old = _sqlite_too_old() or _not_a_checkout()
    if too_old:
        print(too_old, file=sys.stderr)
        sys.exit(1)

    if args.check:
        # The configuration named with --config: the check reads its folders,
        # the GPU index and the language model's settings from it
        from whisper_tm.config import config

        config.load(args.config)
        _run_check(gpu_test=args.gpu_test)
        sys.exit(0)

    if args.hf_token:
        from whisper_tm.config import config
        from whisper_tm.core import hf_access

        config.load(args.config)
        sys.exit(hf_access.cli(interactive=False if args.no_input else None))
    if args.no_input:
        parser.error("--no-input only works with --hf-token")
    if args.prefetch:
        from whisper_tm.config import config
        from whisper_tm.core import prefetch

        config.load(args.config)
        sys.exit(prefetch.cli())

    if args.show_roots or args.remap_path:
        from whisper_tm.database import path_remap

        if args.remap_path:
            if args.apply:
                # Rewriting the paths under a running server would mix its
                # writes with ours: it must be stopped
                from whisper_tm.config import config

                config.load(args.config)
                who = _hold_database(config.db_path)
                if who:
                    print(
                        f"A Whisper Transcription Manager server uses this database "
                        f"({config.db_path}) — {who}. "
                        f"Stop it before --remap-path --apply.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            sys.exit(
                path_remap.run(
                    args.remap_path,
                    apply=args.apply,
                    backup=not args.no_backup,
                    drop_conflicting=args.drop_conflicting,
                    config_path=args.config,
                )
            )
        sys.exit(path_remap.show_roots(args.config))
    if args.apply or args.no_backup or args.drop_conflicting:
        parser.error("--apply, --no-backup and --drop-conflicting only work with --remap-path")

    # Suppress unnecessary warnings
    import warnings

    warnings.filterwarnings("ignore", message="torchcodec is not installed")
    warnings.filterwarnings("ignore", message="TensorFloat-32.*TF32")
    warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

    # ── Load config & CLI overrides ──────────────────────────
    # MUST run before logging setup: the debug flag may also come from
    # config.yaml (logging.debug), not just from --debug.
    from whisper_tm.config import config

    config.load(args.config)
    # Command-line flags hold for this run only (config.override): the
    # settings page saves the whole configuration, and a --port of a test
    # instance must not become the port of the next normal start.
    if args.trace_sql:
        config.override("logging.trace_sql", True)

    # ── Logging / Debug mode ─────────────────────────────────
    from whisper_tm.core import debug

    # --trace-calls and --trace-sql imply --debug: the calls need debug
    # mode's log file, and the statements are logged at DEBUG level.
    debug.setup(
        cli_debug=args.debug or args.trace_calls or args.trace_sql,
        log_dir=args.log_dir,
        trace_calls=args.trace_calls,
    )
    logger = logging.getLogger("whisper_tm")

    # Version info as one of the FIRST log lines — in case of error the
    # debug log must show which code version ran.
    try:
        from whisper_tm.core.version import version_info

        logger.info("Whisper Transcription Manager %s", version_info()["display"])
    except Exception:
        pass

    if config.config_path:
        # The settings page saves back into this file (config.save)
        logger.info(f"Config: {config.config_path}")
    else:
        logger.info("Config: Defaults (no config.yaml found)")

    if args.port:
        config.override("server.port", args.port)
    if args.host:
        config.override("server.host", args.host)
    if args.device:
        config.override("hardware.device", args.device)
    if args.media:
        config.override("directories.media", args.media)
    if args.workdir:
        config.override("directories.work", args.workdir)

    # Make directory overrides effective: config.set() only changes the
    # raw data — resolved paths (config.media_dir etc.) are created
    # in _resolve_directories() at load(). Without re-resolve
    # --media/--workdir would silently have no effect.
    if args.media or args.workdir:
        config._resolve_directories()
        config._ensure_directories()

    # ── One server per database and port ───────────────────
    # Before the app is made: making it re-queues the entries a running
    # server is working on and starts the queue on them.
    who = _hold_database(config.db_path)
    if who:
        print(
            f"Another Whisper Transcription Manager server uses this database "
            f"({config.db_path}) — {who}.\n"
            f"Stop it first, or give this one its own data folder (database.path).",
            file=sys.stderr,
        )
        sys.exit(1)
    taken = _port_taken(
        config.get("server", "host", default="127.0.0.1"),
        config.get("server", "port", default=9860),
    )
    if taken:
        print(
            f"Port {config.get('server', 'port', default=9860)} can't be used ({taken}) — "
            f"is another server running? Choose another with --port.",
            file=sys.stderr,
        )
        sys.exit(1)
    # The embedding and rerank servers a crash, a kill or a closed terminal
    # left behind: the last run of this database's server recorded them
    try:
        from whisper_tm.core import helper_pids

        helper_pids.end_leftovers()
    except Exception as _e:  # noqa — never worth not starting
        logger.debug(f"Leftover model servers not checked: {_e}")

    # ── Create app ─────────────────────────────────────────
    from whisper_tm.app import create_app
    from whisper_tm.database import DatabaseVersionError

    try:
        app = create_app(args.config)
    except DatabaseVersionError as e:
        # Not a crash: a database this version can't open, said as that
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # The compiled speaker parts — the assignment's linking, the map's
    # neighbourhood and fine cut: load them from numba's cache (or compile
    # them once, a few seconds) before the first click needs them
    try:
        import threading

        from whisper_tm.core import speaker_agglomerative, speaker_linkage, speaker_neighbours

        def _warm_speaker():
            speaker_linkage.warm_up()
            speaker_neighbours.warm_up()
            speaker_agglomerative.warm_up()

        _warm = threading.Timer(15, _warm_speaker)
        _warm.daemon = True
        _warm.start()
    except Exception as _e:  # noqa — warming up is an optimization only
        logger.debug(f"Speaker warm-up not scheduled: {_e}")

    # The meaning search's vector matrix (several GB for a large library,
    # seconds from the page cache, much longer from a cold disk): loaded in the
    # background so that the first search after a start doesn't wait for it
    try:
        import threading

        def _warm_search():
            from whisper_tm.core.helpers import parse_bool
            from whisper_tm.rag.service import get_rag_config, make_rag_db

            if parse_bool(get_rag_config().get("enabled", False)):
                make_rag_db().preload()

        _search = threading.Timer(10, _warm_search)
        _search.daemon = True
        _search.start()

        def _warm_media():
            from whisper_tm.api.media_search import warm_up

            warm_up()

        _media = threading.Timer(15, _warm_media)
        _media.daemon = True
        _media.start()
    except Exception as _e:  # noqa — warming up is an optimization only
        logger.debug(f"Search matrix warm-up not scheduled: {_e}")

    # Audio analysis: a background run that was going when the server
    # stopped goes on; with audio_analysis.auto_run new recordings are
    # analyzed by themselves
    try:
        import threading

        def _audio_start():
            from whisper_tm.audio.runner import on_server_start

            on_server_start()

        _audio = threading.Timer(20, _audio_start)
        _audio.daemon = True
        _audio.start()
    except Exception as _e:  # noqa — the run can be started from the page
        logger.debug(f"Audio analysis not scheduled: {_e}")

    # Conversation statistics: with conversations.auto_run, transcripts
    # without numbers (transcribed before, or counted by older rules) are
    # counted in the background — the runner waits a minute itself
    try:
        from whisper_tm.conversations.runner import on_server_start as _conv_start

        _conv_start()
    except Exception as _e:  # noqa — the run can be started from the tab
        logger.debug(f"Conversation statistics not scheduled: {_e}")

    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=9860)

    # ── Banner ────────────────────────────────────────────────
    from whisper_tm import __version__
    from whisper_tm.extensions import hw

    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  Whisper Transcription Manager v{__version__:<17}║")
    print("╠══════════════════════════════════════════════════╣")
    if hw:
        gpu_info = f"{hw.gpus[0].name}" if hw.gpus else "none"
        print(f"║  Device:  {hw.best_device:<39}║")
        print(f"║  GPU:     {gpu_info:<39}║")
    # 0.0.0.0 listens on every interface; a browser opens it as localhost
    # (Windows doesn't open 0.0.0.0 at all)
    url = (
        f"http://localhost:{port} (all interfaces)"
        if host in ("0.0.0.0", "::")
        else f"http://{host}:{port}"
    )
    print(f"║  Server:  {url:<39}║")
    print(f"║  Media:   {str(config.media_dir)[-39:]:<39}║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    open_warning = network_without_login(host)
    if open_warning:
        print(f"  ⚠ {open_warning}")
        print()
        logger.warning(open_warning)
    # Speaker diarization needs a gated model once — say how, every start
    # until it is there (a check of the local cache, no network)
    try:
        from whisper_tm.core import hf_access

        if not config.hf_token and not hf_access.model_cached():
            print("  Speaker diarization is not set up: it needs a free Hugging Face")
            from whisper_tm.core import commands

            if commands.in_docker():
                print("  token once — HF_TOKEN in .env next to docker-compose.yaml,")
            else:
                print(f"  token once — run  {commands.start('--hf-token')},")
            print("  or open the setup guide 🧭 → Speaker diarization.")
            print()
    except Exception as _e:  # noqa — a hint must not keep the server from starting
        logger.debug(f"Diarization hint skipped: {_e}")

    # ── Start server ────────────────────────────────────────
    # Treat SIGTERM (kill/container stop) the same as Ctrl+C, so
    # cleanup (_shutdown) also runs on container stop — not just on
    # KeyboardInterrupt.
    # SIGHUP too: a closed terminal ended the process without its cleanup,
    # and the model servers it had started ran on with their VRAM
    for _name in ("SIGTERM", "SIGHUP"):
        try:
            if _name == "SIGHUP" and signal.getsignal(signal.SIGHUP) == signal.SIG_IGN:
                continue  # started with nohup: hangups are meant to pass
            signal.signal(getattr(signal, _name), _handle_sigterm)
        except (AttributeError, ValueError, OSError, RuntimeError) as _e:  # noqa
            logger.debug(f"{_name} handler not registered: {_e}")

    # The Library's first view is counted ahead: on a large library its
    # statistics take half a minute, and the first open waited for them
    if config.get("ui", "warm_library", default=True):

        def _warm_library():
            time.sleep(3)
            try:
                from whisper_tm.api.stats import warm
                from whisper_tm.core.security import allowed_roots

                warm(str(config.media_dir))  # the open folder: where the Library opens
                roots = [str(r) for r in allowed_roots()]
                if roots and roots[0].rstrip("/") != str(config.media_dir).rstrip("/"):
                    warm(roots[0])  # the tree's top, one click up
            except Exception as _e:  # noqa — a head start only
                logger.debug(f"Library warm-up skipped: {_e}")

        threading.Thread(target=_warm_library, name="library-warm", daemon=True).start()

    logger.info(
        "Built-in web server — fine at home and on your own network; for access "
        "from outside it, put a reverse proxy in front (docs/OPERATING.md → "
        "Behind a reverse proxy)"
    )
    try:
        # Werkzeug server, threaded: the app lives on parallel requests
        # (SSE continuous streams + normal API calls simultaneously). Previously
        # Flask-SocketIO ran here in threading mode — that was the same
        # Werkzeug server, just with unused WebSocket attachment.
        # debug=False whatever --debug says: the app's debug mode is its
        # logging and timing (core/debug.py). Flask's would switch on
        # Werkzeug's interactive debugger — a Python console in the browser
        # behind a PIN, on every address the server listens on.
        app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped.")
    finally:
        _shutdown()


# Module guard: _shutdown() may only run ONCE. Both the finally block
# in main() and the SIGTERM handler trigger it — without the guard
# cleanup would run twice (idempotent/reentrant safe).
_shutting_down = False


def _shutdown():
    """Graceful shutdown: cancel batch, stop watcher, close DB, cleanup."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    from whisper_tm import extensions as ext

    logger = logging.getLogger("whisper_tm")

    try:
        from whisper_tm.core.autoscan import AUTOSCAN

        AUTOSCAN.stop()
    except Exception as _e:  # noqa
        pass

    try:
        # Stop watcher
        if ext.file_watcher and ext.file_watcher.is_running:
            ext.file_watcher.stop()
            logger.info("FileWatcher stopped")
    except Exception as _e:  # noqa
        pass

    try:
        # Cleanly stop and wait for queue worker — otherwise it gets killed
        # while writing → partial transcripts. Must run BEFORE
        # db.close_all() since the worker still writes to the DB.
        #
        # IMPORTANT: shutdown() discards NOTHING. The queue is
        # persistent; waiting entries stay in the DB and the
        # currently running entry gets re-queued so it resumes on
        # next start.
        from whisper_tm.core import queue as _queue

        logger.info("Stopping queue worker…")
        _queue.shutdown(timeout=30.0)
        logger.info("Queue worker stopped (queue preserved)")
    except Exception as _e:  # noqa
        pass

    try:
        # Audio analysis: stop the background run (it goes on with the next
        # start) and write the fingerprints still in memory — after the
        # queue, whose transcriptions hand recordings to it
        from whisper_tm.audio.runner import on_shutdown as _audio_shutdown

        _audio_shutdown()
    except Exception as _e:  # noqa
        pass

    try:
        # Conversation statistics: a run stops after its batch (before the
        # database closes); what is left is counted with the next start
        from whisper_tm.conversations.runner import on_shutdown as _conv_shutdown

        _conv_shutdown()
    except Exception as _e:  # noqa
        pass

    try:
        # Images & sounds: stop an index run (it gives the GPU back) and
        # the service process with its own process group
        from whisper_tm.core import media_server as _media_server
        from whisper_tm.media import indexer as _media_indexer

        _media_indexer.INDEXER.stop()
        _media_server.on_shutdown()
    except Exception as _e:  # noqa
        pass

    try:
        # Stop job manager cleanup timer
        if ext.job_manager:
            ext.job_manager.stop_cleanup_timer()
    except Exception as _e:  # noqa
        pass

    try:
        # Close DB: close_all() closes ALL thread connections
        # (worker/requests) + WAL checkpoint. close() closes only the
        # main thread connection → worker connections would leak.
        if ext.db:
            ext.db.close_all()
            logger.info("Database closed")
    except Exception as _e:  # noqa
        pass

    try:
        # Finally: where did the time go? (also goes to log file)
        from whisper_tm.core import debug

        debug.dump_summary()
        if debug.log_file():
            logger.info("Log for this run: %s", debug.log_file())
    except Exception as _e:  # noqa
        pass


def _handle_sigterm(signum, frame):
    """Treat SIGTERM (kill/container stop) like Ctrl+C: trigger the same
    _shutdown routine and end the process cleanly. _shutdown()
    is idempotent, so the finally block in main() runs as a no-op."""
    logging.getLogger("whisper_tm").info(
        "%s received — shutting down.", getattr(signal.Signals(signum), "name", signum)
    )
    _shutdown()
    sys.exit(0)


def _not_a_checkout() -> str:
    """Why this installation can't run ("" = it can).

    The app runs from its folder — a git checkout (setup.sh, or pip install
    -e) or the Docker image: its pages (whisper_tm/static) and the
    defaults with their reasons (config.default.yaml, beside the package)
    are read from there. A wheel (pip install .) has neither, and started a
    server without a page, on no defaults at all."""
    package = Path(__file__).resolve().parent
    missing = [
        name
        for name, p in (
            ("the pages", package / "static" / "index.html"),
            ("config.default.yaml", package.parent / "config.default.yaml"),
        )
        if not p.exists()
    ]
    if not missing:
        return ""
    return (
        f"This installation lacks {' and '.join(missing)} — Whisper Transcription Manager "
        f"runs from its "
        f"folder: git clone it and run ./setup.sh (or pip install -e . there), or use "
        f"Docker. A wheel (pip install .) doesn't carry them."
    )


def _sqlite_too_old() -> str:
    """Why this Python's SQLite can't hold the database ("" = it can).

    The tables' defaults and many queries use unixepoch(), new in SQLite
    3.38 — without it the first scan, queue entry and statistics fail with
    "no such function". Ubuntu 22.04's Python has 3.37."""
    import sqlite3

    if sqlite3.sqlite_version_info >= (3, 38):
        return ""
    return (
        f"SQLite {sqlite3.sqlite_version} is too old: the database needs 3.38 or newer. "
        f"Run ./setup.sh again — it then builds the venv with a Python whose SQLite "
        f"is new enough (uv's own Python 3.12)."
    )


def _run_check(gpu_test: bool = False):
    """Hardware check and diagnostics — the same checks as the setup guide."""
    from whisper_tm.hardware import detect_hardware

    print("Whisper Transcription Manager — Hardware Check")
    print("=" * 50)
    profile = detect_hardware()
    print(profile.summary())
    print()
    print("Recommendations:")
    print(f"  Compute Type:    {profile.recommended_compute_type}")
    print(f"  Sep Chunk:       {profile.recommended_sep_chunk_duration}s")
    print(f"  Batch Size:      {profile.recommended_batch_size}")
    from whisper_tm.hardware import machine_defaults, machine_defaults_text

    print(f"  {machine_defaults_text(*machine_defaults(profile))}")
    from whisper_tm.hardware import unsupported_card

    if unsupported_card(profile):
        print(f"\n  ✗ {unsupported_card(profile)}")

    # Validation
    issues = []
    if not profile.has_ffmpeg:
        issues.append("FFmpeg not found! Please install.")
    if profile.best_device == "cpu":
        issues.append("No GPU detected. CPU mode is slower.")
    if not profile.has_audio_separator:
        issues.append("audio-separator not installed (Vocal Isolation not available)")

    if issues:
        print()
        print("Notices:")
        for issue in issues:
            print(f"  ⚠ {issue}")

    from whisper_tm.config import config
    from whisper_tm.core import diagnostics

    if not config._loaded:
        config.load()
    print()
    print("System check")
    print("=" * 50, end="")
    diagnostics.print_report(diagnostics.quick_checks())
    if gpu_test:
        print()
        print("GPU test")
        print("=" * 50)
        idx = int(config.get("hardware", "gpu_index", default=0) or 0)
        r = diagnostics.gpu_test(idx)
        for st in r["steps"]:
            mark = "✓" if st["status"] == "ok" else "✗"
            print(f"  {mark} {st['step']}: {st['detail']} ({st['secs']} s)")
        print("  → GPU transcription works" if r["ok"] else "  → GPU transcription does NOT work")
    else:
        print()
        print("Compute on the GPU for real: --check --gpu-test")


if __name__ == "__main__":
    run()

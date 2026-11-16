"""
Flask App Factory: Blueprint registration, middleware.

IMPORTANT: Global instances (db, hw, job_manager) live in
extensions.py — NOT here. create_app() initializes them.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import suppress
from pathlib import Path

from flask import Flask, request

import whisper_tm.extensions as ext
from whisper_tm.config import config
from whisper_tm.database import Database
from whisper_tm.hardware import apply_machine_defaults, detect_hardware, unsupported_card
from whisper_tm.jobs import JobManager

logger = logging.getLogger("whisper_tm")


def _load_or_create_secret() -> str:
    """Flask secret: from the config, otherwise from a key file, otherwise new.

    Previously, without configuration, a new random key was generated on
    EVERY start — every restart thus invalidated all sessions, and the
    warning about it appeared on every run in the log, without anyone ever
    being able to turn it off short of inventing a key by hand.

    The key does not belong in config.yaml: for some people that lives in
    the repo or gets shared. Instead, a dedicated file next to the config,
    readable only by the owner (0600).
    """
    secret = config.get("server", "secret_key", default="")
    if secret:
        return secret

    import secrets

    from platformdirs import user_config_dir

    from whisper_tm.config import APP_NAME

    base = config.config_path.parent if config.config_path else Path(user_config_dir(APP_NAME))
    key_file = base / "secret_key"

    try:
        if key_file.exists():
            existing = key_file.read_text(encoding="utf-8").strip()
            if existing:
                # Whoever reads the key can forge a login session
                if os.name == "posix" and key_file.stat().st_mode & 0o077:
                    logger.warning(
                        "%s is readable by other accounts — chmod 600 it "
                        "(anyone who reads it can forge a login)",
                        key_file,
                    )
                return existing
    except OSError as e:
        logger.warning("secret_key file not readable (%s): %s", key_file, e)

    secret = secrets.token_hex(32)
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        # Written 0600 from the start — write_text and a chmod after it left
        # the key readable for a moment under the usual umask
        from whisper_tm.config import _write_private

        _write_private(key_file, secret)
        logger.info("New secret_key generated: %s (sessions now survive a restart)", key_file)
    except OSError as e:
        # Read-only filesystem (container): keep going, but be honest about it.
        logger.warning(
            "secret_key could not be saved (%s): %s — a random key is in "
            "effect, sessions will not survive a restart.",
            key_file,
            e,
        )
    return secret


def _warn_readable_secrets() -> None:
    """Name the files with passwords and tokens that other accounts can read
    — a config.yaml copied by hand or written by a container's cp is 0644."""
    if os.name != "posix" or not config.config_path:
        return
    folder = Path(config.config_path).parent
    for f in (
        Path(config.config_path),
        folder / "secret_key",
        folder / "mcp_token",
        folder / "llama_api_key",
    ):
        try:
            if f.exists() and f.stat().st_mode & 0o077:
                logger.warning("%s is readable by other accounts — chmod 600 it", f)
        except OSError:
            continue


def _backup_on_start() -> None:
    """The start's backup, if one is due (``Database.backup_due``)."""
    try:
        bp = ext.db.backup_due()
        if bp:
            logger.info(f"DB backup: {bp}")
    except Exception as e:
        logger.warning("Backup on start failed: %s", e)
    # The voice database holds what nobody can recompute: the names, the
    # trash and the decision history — a copy a week, next to the others
    try:
        from whisper_tm.database.speaker import speaker_db

        vp = speaker_db().backup_due(ext.db.backup_target())
        if vp:
            logger.info("Voice database backup: %s", vp)
    except Exception as e:
        logger.warning("Voice database backup on start failed: %s", e)


def create_app(config_path: str | None = None) -> Flask:
    """Create and configure the Flask app.

    Initializes all extensions in whisper_tm.extensions:
    - ext.db (SQLite + FTS5)
    - ext.hw (hardware profile)
    - ext.job_manager (job lifecycle)
    """

    # ── Load configuration ────────────────────────────────────────
    # Skip config.load() if already loaded (e.g. from __main__.py
    # with CLI overrides). Prevents CLI arguments from being lost.
    if not config.is_loaded:
        config.load(config_path)
    warnings = config.validate()
    for w in warnings:
        logger.warning(f"Config: {w}")

    # ── Detect hardware ─────────────────────────────────────────
    ext.hw = detect_hardware(
        preferred_device=config.get("hardware", "device", default="auto"),
        gpu_index=config.get("hardware", "gpu_index", default=0),
        assume_vram_mb=config.get("hardware", "assume_vram_mb", default=0),
    )
    logger.info(f"Hardware:\n{ext.hw.summary()}")
    # The defaults that follow the machine (model, vocal isolation, files at
    # once): chosen once, here — whatever the user set stays as it is
    _, chosen = apply_machine_defaults(ext.hw)
    logger.info(chosen)
    # A card the installed PyTorch has no code for: said at start, not by a
    # "no kernel image" in the middle of the first job
    _card = unsupported_card(ext.hw)
    if _card:
        logger.warning(_card)

    # ── Initialize database ──────────────────────────────────
    ext.db = Database(config.db_path)
    ext.db.initialize()
    logger.info(f"Database: {config.db_path} ({ext.db.get_db_size() // 1024}KB)")

    # Align paths to the current media directory. In the normal case
    # nothing happens here; after a change of the media directory it
    # prevents the scanner from creating duplicate rows and the skip list
    # from losing its mapping.
    try:
        ext.db.realign_paths(str(config.media_dir))
    except Exception as e:
        logger.warning("Path alignment failed: %s", e)

    # Backup on startup — but NOT on every startup.
    #
    # Previously a full copy was made on every server start. With a database
    # of, say, 1.5 GB and `max_backups: 25` that amounts to over 30 GB in the data
    # directory, and every restart added another one — several times a day
    # while developing. A backup on every server start isn't needed, only
    # the rotation as set in the settings.
    #
    # Now the AGE of the most recent existing backup decides
    # (`database.backup_min_hours`, default 24). A restart thus costs
    # nothing, while a daily copy is kept.
    #
    # In the background: a large database takes minutes to copy, and a
    # start that seemed to hang got killed — leaving the staging copy, as
    # big as the database, next to it. Such leftovers go first.
    ext.db.remove_backup_leftovers()
    if config.get("database", "auto_backup", default=True):
        threading.Thread(target=_backup_on_start, name="db-backup-start", daemon=True).start()

    # Export results kept a day — also those left from before this start
    try:
        from whisper_tm.core import export_jobs

        export_jobs.start_tidier()
    except Exception as e:  # noqa: BLE001 — exports tidy on their next job too
        logger.debug("Export tidy not started: %s", e)

    # Clean up stale jobs (from a previous crash)
    try:
        with ext.db.connection() as conn:
            stale = conn.execute(
                "UPDATE job_history SET status='error', error_message='Server restart' "
                "WHERE status IN ('queued', 'processing')"
            )
            if stale.rowcount > 0:
                logger.info(f"Stale jobs cleaned up: {stale.rowcount}")
    except Exception as e:
        logger.debug("Startup cleanup: %s", e)

    if config.get("database", "sync_on_start", default=True):
        tdir = config.transcript_dir
        # tdir.exists() can die with an OSError on a NAS path
        # (CIFS "Host is down", Errno 112) — the app must still be able to
        # start: the DB is the primary source, the JSON sync is only a
        # remedy (a NAS going offline used to crash the startup).
        try:
            tdir_exists = tdir.exists()
        except OSError as e:
            logger.warning("Transcript directory not checkable (%s): %s — DB sync skipped", tdir, e)
            tdir_exists = False
        if tdir_exists:
            result = ext.db.sync_from_json_dir(tdir, media_dir=str(config.media_dir))
            logger.info(
                f"DB sync: {result['imported']} imported, "
                f"{result['skipped']} skipped, "
                f"{result['errors']} errors"
            )

    # ── Job manager ───────────────────────────────────────────────
    ext.job_manager = JobManager(
        logs_dir=config.dirs["logs"],
        max_age_secs=3600,
    )
    ext.job_manager.start_cleanup_timer()

    # ── LLM queue ─────────────────────────────────────────
    # Own sequential worker: summarization/correction/
    # translation keep running server-side even if the browser
    # closes — and never in parallel (one llama.cpp slot).
    from whisper_tm.core.llm_queue import llm_queue

    llm_queue.start()
    # Analyses that wait for their transcripts (videos fetched for them)
    # go into the LLM queue once those are there
    from whisper_tm.core import analysis as _analysis

    _analysis.start_watcher()

    def _job_event(event, job, *args):
        # Invalidate the stats cache when a job finishes,
        # so dashboard data stays current.
        # (Replaces the former circular import of
        #  api.settings.invalidate_stats_cache in core.transcription)
        if event in ("done", "error"):
            try:
                from whisper_tm.api.settings import invalidate_stats_cache
                from whisper_tm.api.stats import clear_cache

                invalidate_stats_cache()
                clear_cache()
            except Exception as e:
                logger.debug("Unhandled exception: %s", e)

    ext.job_manager.on_event(_job_event)
    # Persist warnings and errors in library.db (schema v19, incidents).
    from whisper_tm.core import incidents as _incidents

    _incidents.attach(ext.job_manager)

    # ── Flask app ─────────────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static",
        template_folder=str(static_dir),
    )
    app.config["MAX_CONTENT_LENGTH"] = (
        config.get("server", "max_upload_mb", default=200) * 1024 * 1024
    )
    # Do NOT cache static assets long-term: otherwise the browser keeps
    # serving old JS/CSS from cache after an update (bugfixes never land).
    # max_age=0 + no-cache forces revalidation on every load (local app →
    # doesn't matter).
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["SECRET_KEY"] = _load_or_create_secret()
    _warn_readable_secrets()
    # Browsers share cookies across every port of a host name: two installs
    # on one machine (or another Flask app there) overwrote each other's
    # "session". A name of this install's own, from its key — not from the
    # port, which is 9860 inside every container.
    import hashlib

    app.config["SESSION_COOKIE_NAME"] = (
        "wtm_session_" + hashlib.sha256(app.config["SECRET_KEY"].encode()).hexdigest()[:8]
    )
    # Not sent along when another site's page makes the browser ask (forms,
    # images, frames): the origin check stops such writes already, this
    # stops them earlier and for reads too
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    # A login lasts auth.session_days from the moment it was made: Flask's
    # default renewed the cookie on every request, so one in use never ended
    from datetime import timedelta

    try:
        days = float(config.get("auth", "session_days", default=7) or 7)
    except (TypeError, ValueError):
        days = 7.0
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=max(days, 0.01))
    app.config["SESSION_REFRESH_EACH_REQUEST"] = False
    # Behind a TLS proxy: the cookie only over https (server.secure_cookie)
    app.config["SESSION_COOKIE_SECURE"] = bool(config.get("server", "secure_cookie", default=False))

    # ── Load history (performance view, top-bar meters) ──────────
    # Own small SQLite file next to the main DB — see core/metrics.py.
    from whisper_tm.core import metrics

    metrics.start(Path(config.db_path).parent)

    # ── Work folder ───────────────────────────────────────────────
    # Emptied BEFORE the queue worker starts: it used to happen further
    # down, after the worker had already started — a job picked up in
    # between lost its work files to the cleanup of its own server.
    config.clean_workdir()
    # After, not before: the marker lives INSIDE the workdir and would
    # otherwise be swept up again by the cleanup itself. It keeps the
    # next server from clearing our work files out from under us
    # (config.py).
    config.claim_workdir()
    # Uploads that never finished leave their chunks in the cache — they
    # lie outside the work folder, so its cleanup doesn't take them
    try:
        from whisper_tm.api.media import remove_stale_chunks

        remove_stale_chunks()
    except Exception as e:  # noqa: BLE001 — never worth not starting
        logger.debug("Old upload chunks not removed: %s", e)

    # ── Queue worker ──────────────────────────────────────────────
    # The worker runs for the entire server lifetime and works through the
    # persistent queue — regardless of whether a browser has the WebUI
    # open. Before starting, entries that were mid-processing at the last
    # shutdown are re-queued.
    from whisper_tm.core import queue as queue_worker

    queue_worker.requeue_stale()

    # queue.autostart: false → worker runs, but only picks up work once
    # "Resume" is clicked in the WebUI. Otherwise a server start
    # immediately jumps into the next file, which gets in the way while
    # developing/debugging (and, with a full queue, occupies the GPU
    # before you've even had a look). The queue itself is kept intact
    # in both cases.
    if config.get("queue", "autostart", default=True):
        queue_worker.start_worker()
    else:
        queue_worker.pause()
        queue_worker.start_worker()
        logger.info(
            "Queue worker started, but PAUSED (queue.autostart: false) — "
            "processing only starts via 'Resume' in the WebUI."
        )

    # ── Cross-site protection ─────────────────────────────────
    # First hook of all: a state-changing request from a foreign web page
    # (no login needed while auth is off) is refused before anything runs.
    from whisper_tm.core.security import register_origin_check

    register_origin_check(app)

    # ── Register blueprints ───────────────────────────────────
    from whisper_tm.api import register_blueprints

    register_blueprints(app)

    # ── Request timing ───────────────────────────────────────
    # Always runs (costs nothing); only output in debug mode.
    # The values end up in the evaluation under /api/debug/timings.
    from whisper_tm.core import debug as _debug

    @app.before_request
    def _req_start():
        request.environ["_wxp_t0"] = time.perf_counter()

    @app.after_request
    def _req_end(resp):
        t0 = request.environ.get("_wxp_t0")
        if t0 is None:
            return resp
        dt = time.perf_counter() - t0

        # Group by endpoint, not by raw URL — otherwise every
        # /api/status/<jobid> bloats the evaluation with its own row.
        ep = request.endpoint or request.path

        # SSE streams: after_request fires BEFORE the generator has sent
        # even a single line — the measured time would always be ~0ms and
        # thus a flat-out lie (the stream keeps running for hours
        # afterward). So clearly label what was actually measured here.
        streamed = resp.is_streamed
        label = f"http: {request.method} {ep}" + (" [stream start]" if streamed else "")
        _debug.record(label, dt)

        if _debug.is_enabled():
            # extra={"http": True} additionally routes the line into
            # http.log — by logger name alone it couldn't be told apart
            # from others.
            logger.debug(
                "⏱ HTTP %s %s → %s in %s%s · %s",
                request.method,
                request.path,
                resp.status_code,
                _debug.fmt_secs(dt),
                " (stream still running)" if streamed else "",
                request.remote_addr or "-",
                extra={"http": True},
            )
        return resp

    # ── Cache control: frontend always fresh ──────────────────────
    # HTML/JS/CSS must NOT be cached long-term, otherwise users end up
    # running on stale code after an update (e.g. the "Done" button
    # doesn't appear, a popup doesn't close). no-cache = revalidate,
    # don't re-download if unchanged (304).
    @app.after_request
    def _no_cache_frontend(resp):
        with suppress(Exception):
            ct = resp.headers.get("Content-Type", "")
            path = request.path or ""
            if (
                path == "/"
                or path.endswith((".html", ".js", ".css"))
                or "text/html" in ct
                or "javascript" in ct
                or "text/css" in ct
            ):
                resp.headers["Cache-Control"] = "no-cache, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
        return resp

    #: Answers that carry a file of the library or one the app made: an
    #: .html or .svg among them (a file in the media folder, an upload) must
    #: never run as the app's own page — with the API at its disposal
    _FILE_ANSWERS = ("/api/video/", "/api/download/", "/api/clip/", "/api/media-search/")

    #: The app's own pages: scripts, styles, pictures and requests from the
    #: app itself only (inline ones too — the pages' handlers are written
    #: into their HTML), no plugins, no foreign <base> or form target. An
    #: injected <script src> or fetch to elsewhere goes nowhere. No
    #: frame-ancestors: dashboards may show the app in a frame.
    _APP_CSP = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "media-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
        "worker-src 'self' blob:; object-src 'none'; base-uri 'self'; "
        "form-action 'self'"
    )

    @app.after_request
    def _file_answers_run_nothing(resp):
        with suppress(Exception):
            resp.headers.setdefault("X-Content-Type-Options", "nosniff")
            if (request.path or "").startswith(_FILE_ANSWERS):
                # A document shown from here gets an origin of its own and
                # runs no script; audio, video and pictures play as before
                resp.headers.setdefault("Content-Security-Policy", "sandbox")
            elif "text/html" in resp.headers.get("Content-Type", ""):
                resp.headers.setdefault("Content-Security-Policy", _APP_CSP)
        return resp

    # ── Authentication (optional) ──────────────────────────────
    if config.get("auth", "enabled", default=False):
        from whisper_tm.core.auth import register_auth

        register_auth(app)

    # ── Error handlers ────────────────────────────────────────────
    @app.errorhandler(403)
    def forbidden(e):
        # JSON instead of a Flask HTML page: the frontends do r.json() —
        # an HTML response ended up as "JSON.parse: unexpected character"
        # instead of a readable error message (e.g. a rejected root).
        return {"error": getattr(e, "description", None) or "Not allowed"}, 403

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Internal server error")
        return {"error": "Internal server error"}, 500

    @app.errorhandler(413)
    def too_large(e):
        return {"error": "File too large"}, 413

    # ── Health check ──────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        """Cheap liveness ping (the frontend polls it): status, compute device,
        whether the database is attached. The full picture is /api/health/detail.
        """
        return {
            "status": "ok",
            "device": ext.hw.best_device if ext.hw else "unknown",
            "db": ext.db is not None,
        }

    # ── Debug: timing evaluation ─────────────────────────────────────
    @app.route("/api/debug/timings")
    def debug_timings():
        """Where did the time go? Available even without debug mode.

        'vram' lists the GPU memory peak per model (only populated when
        run in debug mode with GPU) — the basis for multi-GPU planning.
        """
        return {
            "debug": _debug.is_enabled(),
            "log_file": str(_debug.log_file()) if _debug.log_file() else None,
            "timings": _debug.timings(),
            "vram": _debug.vram_stats(),
            "gpus": _debug.gpu_static(),
        }

    @app.route("/api/debug/trace", methods=["GET", "POST"])
    def debug_trace():
        """Turn call tracing on and off at runtime.

        The reason for this endpoint: when starting with --trace-calls,
        startup and directory scanning fill the file with hundreds of
        megabytes before a transcription even begins — the interesting
        part gets lost in the noise. This way the trace can be placed
        exactly around the section you want to inspect:

            curl -XPOST -d '{"on":true}'  -H 'Content-Type: application/json' …
            # ... start a transcription and wait ...
            curl -XPOST -d '{"on":false}' -H 'Content-Type: application/json' …
        """
        from flask import request as _req

        if _req.method == "POST":
            data = _req.json if isinstance(_req.json, dict) else {}
            turn_on = bool(data.get("on", True))
            if turn_on:
                if not _debug.is_enabled():
                    return {"error": "Only possible in debug mode (--debug)"}, 400
                _debug.start_call_tracing(with_arguments=data.get("args"))
            else:
                _debug.stop_call_tracing(reason="ended via /api/debug/trace")
        return {
            "active": _debug.call_tracing_active(),
            "events": _debug.trace_events(),
            "trace_log_mb": round(_debug.trace_log_bytes() / (1024 * 1024), 1),
            "log_dir": str(_debug.log_dir()) if _debug.log_dir() else None,
        }

    @app.route("/api/debug/timings/text")
    def debug_timings_text():
        """Same evaluation as a ready-made table (curl-friendly)."""
        from flask import Response as _Resp

        text = _debug.summary(top=50)
        vram_tbl = _debug.vram_summary()
        if vram_tbl:
            text += "\n" + vram_tbl
        return _Resp(text + "\n", mimetype="text/plain")

    @app.route("/api/debug/gpu")
    def debug_gpu():
        """Current GPU memory per card + model peaks measured so far.

        Handy for live-watching how full each GPU is — e.g. while
        weighing whether to put the separation model on a second card.
        """
        static = _debug.gpu_static()
        used = _debug.gpu_mem_used_mb()
        gpus = []
        for idx in sorted(static):
            info = static[idx]
            u = used.get(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": info.get("name", f"GPU{idx}"),
                    "used_mb": u,
                    "used": _debug.fmt_mb(u) if u is not None else "?",
                    "total_mb": info.get("total_mb", 0),
                    "total": _debug.fmt_mb(info["total_mb"]) if info.get("total_mb") else "?",
                }
            )
        return {
            "available": _debug.gpu_available(),
            "gpus": gpus,
            "vram_peaks": _debug.vram_stats(),
        }

    @app.route("/api/debug/timings/reset", methods=["POST"])
    def debug_timings_reset():
        """Reset the collected step and request timings of /api/debug/timings."""
        _debug.reset()
        return {"ok": True}

    # ── Startup info ──────────────────────────────────────────────
    dirs = config.dirs
    logger.info(f"Media:       {dirs['media']}")
    logger.info(f"Transcripts: {dirs['transcripts']}")
    logger.info(f"Work:        {dirs['work']}")
    logger.info(f"Device:      {ext.hw.best_device}")

    _warn_about_stale_roots()

    # ── Filesystem watcher (optional) ─────────────────────────────
    # No on_change callback: the frontend fetches changes via a 60s poll
    # of /api/watcher/status; there is no push channel
    # (the former SocketIO emit never had a client).
    if config.get("scanner", "watch_enabled", default=False):
        from whisper_tm.scanner.watcher import FileWatcher

        interval = config.get("scanner", "watch_interval_sec", default=30)
        ext.file_watcher = FileWatcher(
            media_dir=config.media_dir,
            interval=interval,
            recursive=True,
        )
        ext.file_watcher.start()
        logger.info(f"FileWatcher: active ({interval}s interval)")

        # Watcher follows runtime directory changes (/api/set_directory):
        # without this hook it would keep polling the old directory forever.
        def _on_config_change(key, value):
            if key == "directories.media" and ext.file_watcher:
                try:
                    ext.file_watcher.set_media_dir(Path(value))
                except Exception as e:
                    logger.debug("Watcher retarget failed: %s", e)

        config.on_change(_on_config_change)

    # ── Continuous scanning (core/autoscan.py) — only when switched on ──
    try:
        from whisper_tm.core.autoscan import AUTOSCAN

        AUTOSCAN.apply()
    except Exception as e:  # noqa: BLE001 — the app runs without it
        logger.warning("Autoscan not started: %s", e)

    start_mcp_services()

    return app


def _warn_about_stale_roots() -> None:
    """Banner if stored paths point to a root this machine doesn't have.

    After a move to another server with a different mount point, every
    transcript is still there but can no longer be opened — and the
    interface just looks empty. The banner names the command that
    translates the paths (database/path_remap.py).

    In a thread: `isdir` on a hung network mount can block for minutes.
    """

    def check():
        try:
            from whisper_tm.database.path_remap import stale_roots

            stale = stale_roots(config.db_path)
        except Exception as e:
            logger.debug("Root check skipped: %s", e)
            return
        if not stale:
            return
        root, (n, target) = next(iter(stale.items()))
        where = (
            f"which is a symlink to {target}" if target else "which does not exist on this machine"
        )
        text = (
            f"{n:,} stored rows point to {root}, {where}. If the media "
            f"moved, stop the server and run: whisper-tm --remap-path "
            f"{root} {target or '<new path>'}"
        )
        logger.warning(text)
        if ext.job_manager:
            ext.job_manager.system_warning(text)

    threading.Thread(target=check, name="root-check", daemon=True).start()


def start_mcp_services() -> list[str]:
    """Start the MCP server's network services alongside it, if requested.

    **Why this even lives here.** The MCP server used to be a purely
    manual start. Since it doesn't log anything during normal operation,
    its absence goes unnoticed — you only notice once a tool fails to
    respond. That happens after every machine restart, i.e. regularly.

    **Only the NETWORK services.** The stdio path doesn't need this: there
    the AI client starts the server itself and talks over standard input.

    **In threads, not as separate processes.** The MCP server talks to the
    app over HTTP — it can run in the same process and then simply call
    itself. This avoids process management (starting, monitoring,
    stopping) and ensures the services go down together with the app.
    As daemon threads: nothing lingers when the app exits.

    Return value: which services were started (for logging and tests).
    """
    started: list[str] = []
    wants_openapi = bool(config.get("mcp", "autostart_openapi", default=False))
    wants_http = bool(config.get("mcp", "autostart_http", default=False))
    if not (wants_openapi or wants_http):
        return started

    try:
        from whisper_tm.mcp_server import (
            access_token,
            build_openapi_app,
            build_server,
            http_middleware,
            load_settings,
            required_token,
            set_runtime,
        )
    except ImportError as e:  # fastmcp missing
        logger.warning(
            "MCP autostart skipped — fastmcp is missing (%s). "
            "Install it with: pip install 'whisper-tm[mcp]'",
            e,
        )
        return started

    try:
        # `reload=False`: the configuration is already in place here.
        # Fetching it from disk again in the middle of startup would
        # discard everything that has been set at runtime up to this point.
        settings = load_settings("", "", reload=False)
        # Without this, EVERY tool responds with "No connection configured."
        # — the service then runs, returns its spec, and yet can't do
        # anything.
        set_runtime(settings)
        server = build_server()
        # Made now rather than on the first request, so the settings page
        # can show it before any client has connected
        access_token()
    except Exception as error:  # never crash server startup
        logger.warning("MCP autostart failed: %s", error)
        return started

    def _run(name: str, target) -> None:
        try:
            target()
        except Exception as error:
            logger.warning("MCP service %s stopped: %s", name, error)

    # EVERY SERVICE BINDS ITS PORT IMMEDIATELY, not only once it starts running.
    #
    # Previously both branches said `host, port = ...` and both lambdas
    # read the same variable — a closure over the function body, not over
    # the value. By the time the OpenAPI thread actually ran, the second
    # branch had long since overwritten `port` with the MCP port. The
    # result, with both switches on at the same time: OpenAPI took 5399, the
    # MCP service wanted the same port and failed with "address already in
    # use" — meanwhile the log
    # said "OpenAPI on 127.0.0.1:5400", i.e. the opposite. Whoever
    # addressed the MCP server ended up at the OpenAPI service and got
    # "Unknown tool: mcp".
    #
    # The default values in the lambda bind the value at creation time.
    if wants_openapi:
        import uvicorn

        oa_host, oa_port = settings["host"], int(settings["openapi_port"])
        threading.Thread(
            target=_run,
            args=(
                "OpenAPI",
                lambda h=oa_host, p=oa_port: uvicorn.run(
                    build_openapi_app(server, required_token), host=h, port=p, log_level="warning"
                ),
            ),
            name="mcp-openapi",
            daemon=True,
        ).start()
        started.append(f"OpenAPI on {oa_host}:{oa_port}")

    if wants_http:
        mcp_host, mcp_port = settings["host"], int(settings["port"])
        threading.Thread(
            target=_run,
            args=(
                "HTTP",
                lambda h=mcp_host, p=mcp_port: server.run(
                    transport="http", host=h, port=p, middleware=http_middleware()
                ),
            ),
            name="mcp-http",
            daemon=True,
        ).start()
        started.append(f"MCP/HTTP on {mcp_host}:{mcp_port}")

    for z in started:
        logger.info("MCP autostart: %s", z)
    return started

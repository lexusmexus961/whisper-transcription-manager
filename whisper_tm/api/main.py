"""
Main routes: index, directory management, scan, browse.
"""

import json
import logging
import os
import threading
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request

from whisper_tm import extensions as ext
from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.core.security import (
    allowed_roots,
    check_root,
    root_allowed,
    safe_path,
    safe_path_or_none,
)

# `ext.db` and NOT `from …extensions import db`.
#
# Direct import binds the VALUE at import time. `create_app()` sets
# `ext.db` only after — anyone who imports this module before keeps forever
# the `None` from before and gets a 503 from every DB endpoint.
#
# In the app this works out, but only by chance of order: `app.py`
# sets `ext.db` before it imports the blueprints. A test file that imports
# the module during collection breaks that order — then a completely
# different test fails (traversal check reports 503 instead of 400), with
# the cause three files away.
#
# The module header of `extensions.py` already says it: "Only import this
# module for access to global instances."

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)

# Serializes writers of process-global config.media_dir
# (set_directory / navigate_to_file) so two parallel requests don't
# change the target directory for each other (#116).
# DEFERRED: The clean refactor passes the target directory PER REQUEST instead
# of mutating config.media_dir globally. But that affects many readers
# (scan/index/browse/…) across multiple files — too invasive for this file.
# So here only: mutation under lock + consistent response snapshot of the set
# path (the pure read race of other endpoints on the global state remains
# deliberately open).
_media_dir_lock = threading.Lock()


@main_bp.route("/")
def index():
    # Cache buster for all static includes: version + commit
    # change with every build — without it the browser happily continues
    # with old JS/CSS from cache ("button does nothing" while code is fixed).
    from whisper_tm.core.version import version_info

    vi = version_info()
    buster = f"{vi['version']}-{vi['commit'] or 'x'}"
    # …and the most recent timestamp of the static files themselves.
    #
    # Version and commit alone are NOT enough. `version_info()` is cached
    # for process lifetime, so the buster is fixed for the whole server run.
    # Anyone who changes a JS file without restarting changes CONTENT at an
    # unchanged URL — and every browser that loaded the page in this run
    # continues with the old file.
    #
    # Without the marker, a fix to a static file can be built, tested and
    # committed but stay invisible in an already open browser — it keeps
    # `some_file.js?v=<version>-<commit>` from its cache while the server
    # delivers new content at the same URL, and a check in a fresh browser
    # sees the new version. An error that hides as "still doesn't work"
    # costs more time than the error itself.
    buster += "-" + str(_static_marker())
    return render_template("index.html", video_dir=str(config.media_dir), v=buster)


def _static_marker() -> int:
    """Most recent modification time under ``static/`` as whole second.

    Costs around 40 ``stat`` calls per page load — only on ``/``, not on
    the files themselves. Against a cache holding back a fixed file,
    that's cheap.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "static"
    newest = 0.0
    try:
        for p in root.rglob("*"):
            if p.suffix in (".js", ".css", ".html"):
                m = p.stat().st_mtime
                if m > newest:
                    newest = m
    except OSError:
        return 0
    return int(newest)


#: The library folder as config.yaml holds it ("" = not looked at yet)
_saved_library = ""


def _keep_library_choice(directory: str) -> None:
    """Save the open folder as the library when it is another library than
    the saved one — not on a step into one of its subfolders.

    Somebody who enters the NAS mount in the path field has chosen their
    library: a restart must not go back to the old folder. A subfolder is
    only a place inside it, and saving that would make it the library.
    """
    global _saved_library
    if not _saved_library:
        _saved_library = str(getattr(config, "media_on_start", "") or "")
    if _saved_library and paths.is_inside(directory, _saved_library):
        return
    try:
        config.save()
        _saved_library = directory
    except Exception as e:  # noqa: BLE001 — the folder is open all the same
        logger.warning("The library folder could not be saved: %s", e)


@main_bp.route("/api/set_directory", methods=["POST"])
def set_directory():
    """Switch media directory."""
    data = request.json if isinstance(request.json, dict) else {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "No path given"}), 400
    # The root comes from the request. It's the basis against which every
    # later safe_path() check runs — anyone who can set it to "/" thus
    # disables all the app's path checks (abort(403) on violation).
    # The owner — this machine, or through the login — may choose a library
    # anywhere; everybody else stays inside the allowed trees.
    from whisper_tm.core.security import owner_request, remember_root, root_allowed

    new_tree = not root_allowed(path)
    if new_tree and not owner_request():
        return jsonify(
            {
                "error": "This folder is outside the allowed directories "
                "(security.allowed_roots). A new library folder can be chosen "
                "on the machine the server runs on, or with the login on."
            }
        ), 403
    try:
        # #116: serialize mutation of the process-global media_dir and read
        # the path back inside the same lock → a consistent response.
        with _media_dir_lock:
            config.media_dir = path
            directory = str(config.media_dir)
        if new_tree:
            # A new library, not a step inside one: allowed for this run
            remember_root(directory)
        # Kept only when somebody chose it in the path field. The page also
        # switches by itself — to the upload folder after an upload — and
        # that must not make the upload folder the library.
        if data.get("chosen") is True:
            _keep_library_choice(directory)
        return jsonify({"ok": True, "directory": directory})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@main_bp.route("/api/clean_workdir", methods=["POST"])
def api_clean_workdir():
    """Empty the work folder — never under a running job.

    The folder holds the files of the job in progress (audio blocks,
    separated stems); deleting them kills it with "No such file or
    directory". So this refuses with 409 while the queue works on a file,
    and ``config.clean_workdir`` asks again before every entry it deletes.
    The answer says what happened: ``removed`` and ``failed`` entries, and
    with ``ok: false`` the ``error`` — ``refused`` (409) when it stood back.
    It used to answer ``ok`` whatever happened.
    """
    from whisper_tm.core import queue as q

    def busy() -> str:
        try:
            snap = q.snapshot()
        except Exception as e:  # noqa — when in doubt, delete nothing
            return f"Queue state unknown ({e}) — the work folder was not cleaned."
        # Not the snapshot's `running`: that also counts waiting entries,
        # which have no files yet. 'processing' rows cover the moment in
        # which the worker has claimed an entry but not yet made it current.
        processing = snap["total"] - snap["processed"] - snap["pending"]
        if snap["current_item_id"] is None and processing <= 0:
            return ""
        # Queue lanes: name the oldest running file and how many run beside it.
        n = snap.get("running_count") or 0
        what = snap["current_file"] or "a file"
        if n > 1:
            what += f" (+{n - 1} more)"
        return (
            f"The queue is processing {what} — its files are in the work "
            f"folder. Pause the queue and clean up once the running "
            f"{'jobs have' if n > 1 else 'job has'} stopped."
        )

    reason = busy()
    if reason:
        return jsonify(
            {"ok": False, "refused": True, "removed": 0, "failed": 0, "error": reason}
        ), 409
    result = config.clean_workdir(busy=busy)
    # 409 when it stood back: another server works there, the folder
    # overlaps user data, or a job started meanwhile. Entries that would not
    # go (permissions) are a report with ok false, not a refusal.
    return jsonify(result), (409 if result["refused"] else 200)


def _library_unchosen() -> bool:
    """No library chosen yet: the folder is only the default (~/Videos). A
    first start on a machine without that folder is not an error to
    show in red, but a library still to choose."""
    return not str(config.get("directories", "media", default="") or "").strip()


@main_bp.route("/api/scan")
def scan_directory():
    """Scan media directory (filesystem walk → DB sync).

    ?walk=false → only read from DB (no filesystem walk). For the
    DISPLAY, the frontend has used paginated /api/files endpoints since
    the UI rework; this endpoint is the walk trigger (fallback without SSE —
    the normal way is /api/scan_stream) and now returns the list only for
    completeness.

    The old page/per_page "pagination" is removed: it only trimmed the
    JSON output AFTER the full walk+full upsert (#117) —
    true pagination delivers /api/files directly from indices.
    """
    from whisper_tm.scanner import scan_media_directory, scan_media_directory_from_db

    vdir = config.media_dir
    recursive = request.args.get("recursive", "true").lower() in ("true", "1", "yes")
    ultra = request.args.get("ultra", "true").lower() in ("true", "1", "yes")
    walk = request.args.get("walk", "true").lower() in ("true", "1", "yes")

    if walk or not ext.db:
        # db=None: without a database there is nothing to load → always walk
        result = scan_media_directory(vdir, recursive=recursive, ultra=ultra, db=ext.db)
    else:
        result = scan_media_directory_from_db(vdir, db=ext.db)

    if result.error:
        return jsonify(
            {
                "error": result.error,
                "files": [],
                "directory": result.directory,
                "unchosen": _library_unchosen(),
            }
        )

    resp = {
        "directory": result.directory,
        "files": result.files,
        "total": result.total,
        "done": result.done,
        "recursive": result.recursive,
        "ultra": result.ultra,
        "stats": result.stats,
        "index": result.index,
        "source": result.source,
        # Orphaned transcripts (file gone, transcript here) — the GUI reports
        # the finding and offers the filter.
        "orphans": result.orphans,
    }
    if result.transcript_dirs_found:
        resp["transcript_dirs_found"] = result.transcript_dirs_found

    return jsonify(resp)


@main_bp.route("/api/scan_entry")
@main_bp.route("/api/scan_entry/<path:filename>")
def scan_entry(filename=None):
    """ONE file from list fresh from DB — incremental list update.

    When a file finishes transcribing, exactly its row changes. Previously
    the frontend reloaded the ENTIRE list (/api/scan?walk=false): on a large
    library several seconds per call, every ~25s — right when the queue worker was locking
    the DB with write transactions. Here the same update costs one indexed
    point query.

    Response: {"file": <entry>} — same form as an entry from /api/scan.
    404 if the DB doesn't (yet) know the path.

    ``?abs=`` carries the identity of the file that just finished. The
    queue works across roots, while ``filename`` is relative to the queue
    entry's root — not to the folder open in the list. If the file lies
    outside the open folder, the list doesn't show it: the answer is
    ``{"file": null, "outside": true}`` and the frontend has nothing to do.
    Without this, every such job came back 404 and the frontend fell back
    to a full rescan of the open folder after each file.

    ``?file=`` takes a recording by its full path, wherever it lies within
    the allowed roots — the MCP server names a search hit outside the open
    folder that way; its ``name`` is then relative to the allowed
    root it lies in.
    """
    from whisper_tm.scanner import db_row_to_entry

    if not ext.db:
        return jsonify({"error": "No database configured"}), 503

    if filename is None:
        full = request.args.get("file") or ""
        if not paths.is_absolute(full):
            return jsonify({"error": "file must be a full path"}), 400
        check_root(full, field_name="file")
        abs_path = paths.canon(full)
        row = ext.db.get_media_file_at(abs_path)
        roots = [str(r) for r in allowed_roots() if paths.is_inside(abs_path, str(r))]
        entry = row and db_row_to_entry(row, roots[0] if roots else abs_path.rpartition("/")[0])
        if not entry:
            return jsonify({"error": "File not in the database"}), 404
        return jsonify({"file": entry})

    given = paths.canon(request.args.get("abs") or "")
    if given and not paths.is_inside(given, config.media_dir):
        return jsonify({"file": None, "outside": True})

    if given:
        # The identity names the file. ``filename`` is relative to the queue
        # entry's root, not always the open one: the upload dialog opens the
        # uploads folder, and "uploads/x.m4a" then named …/uploads/uploads/
        # x.m4a — a 404, and a full rescan after every finished upload.
        abs_path = given
    else:
        # Traversal protection, then form the identity (docs/PATHS.md).
        if safe_path_or_none(config.media_dir, filename) is None:
            return jsonify({"error": "Invalid path"}), 400
        abs_path = paths.abs_from(filename, config.media_dir)

    try:
        row = ext.db.get_media_file_at(abs_path)
    except Exception as e:
        logger.exception("Single-file scan failed")
        return jsonify({"error": f"DB query failed: {e}"}), 500

    if row is None:
        return jsonify({"error": "File not in the database"}), 404

    entry = db_row_to_entry(row, str(config.media_dir))
    if entry is None:
        return jsonify({"error": "File not in the database"}), 404
    return jsonify({"file": entry})


@main_bp.route("/api/scan_stream")
def scan_directory_stream():
    """Like /api/scan (walk mode), but as SSE stream with live progress.

    Events:
        {"phase": "walk", "files": n, "dirs_done": d, "dirs_known": k,
         "expected_files": e}     — Walk running (e = DB count from last
                                    walk as expected value for progress bar)
        {"phase": "db"}           — DB preload
        {"phase": "process", "done": n, "total": t}  — File processing
        {"complete": true, ...}   — full scan result (like /api/scan)

    No filtering/pagination — the stream serves the default file list.
    """
    import queue as _queue

    from whisper_tm.scanner import scan_media_directory

    vdir = config.media_dir
    if not vdir.exists():
        return jsonify(
            {"error": f"Directory not found: {vdir}", "unchosen": _library_unchosen()}
        ), 404

    # Expected value for the progress bar: count from the last walk.
    expected = 0
    if ext.db:
        try:
            expected = ext.db.count_media_files_under(str(vdir))
        except Exception as e:
            logger.debug("count_media_files_under failed: %s", e)

    def stream():
        q: _queue.Queue = _queue.Queue()
        holder: dict = {}
        # Client disconnect → generator dies (GeneratorExit on yield) →
        # finally sets stop flag → walk/scan breaks server-side instead of
        # running unobserved for hours.
        stop = {"v": False}

        def worker():
            try:
                holder["result"] = scan_media_directory(
                    vdir,
                    recursive=True,
                    ultra=True,
                    db=ext.db,
                    on_progress=q.put,
                    should_stop=lambda: stop["v"],
                )
            except Exception as e:  # noqa
                logger.exception("Scan stream error")
                holder["error"] = str(e)[:300]
            finally:
                q.put(None)  # end signal

        threading.Thread(target=worker, daemon=True).start()

        try:
            # Immediate init + expected value so client can build progress
            # bar before the first (throttled) walk event comes.
            yield ": init\n\n"
            start = {
                "phase": "walk",
                "files": 0,
                "dirs_done": 0,
                "dirs_known": 1,
                "expected_files": expected,
            }
            yield f"data: {json.dumps(start)}\n\n"

            while True:
                try:
                    item = q.get(timeout=10)
                except _queue.Empty:
                    # Heartbeat as DATA event (not SSE comment): comments
                    # don't fire onmessage client-side → the client watchdog
                    # would falsely abort a long, healthy walk.
                    yield f"data: {json.dumps({'phase': 'heartbeat'})}\n\n"
                    continue
                if item is None:
                    break
                if item.get("phase") == "walk":
                    item["expected_files"] = expected
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            stop["v"] = True

        if "error" in holder:
            yield f"data: {json.dumps({'error': holder['error']})}\n\n"
            return
        result = holder.get("result")
        if result is None or result.error:
            err = result.error if result else "Scan failed"
            yield f"data: {json.dumps({'error': err})}\n\n"
            return
        # NO file list in complete event: the display has loaded paginated
        # via /api/files since the UI rework — the full list here would be
        # tens of MB of JSON for nothing on a large library.
        resp = {
            "complete": True,
            "directory": result.directory,
            "total": result.total,
            "done": result.done,
            "recursive": result.recursive,
            "ultra": result.ultra,
            "stats": result.stats,
            "index": result.index,
            "source": result.source,
            "orphans": result.orphans,
        }
        if result.transcript_dirs_found:
            resp["transcript_dirs_found"] = result.transcript_dirs_found
        yield f"data: {json.dumps(resp)}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Full indexing (FFprobe metadata → DB) ─────────────────────────────────


@main_bp.route("/api/index", methods=["GET"])
def index_media():
    """Fully index a directory: FFprobe metadata persisted to the DB.

    Query: recursive (bool), force (bool), path (optional subdirectory).
    SSE stream with progress; skips already indexed files.
    """
    if not ext.db:
        return jsonify({"error": "No database configured"}), 501

    recursive = request.args.get("recursive", "true").lower() in ("true", "1", "yes")
    force = request.args.get("force", "false").lower() in ("true", "1", "yes")
    sub = request.args.get("path", "").strip()

    vdir = config.media_dir
    if sub:
        # DEFERRED (#132): for the sub-index (?path=), vdir is set to the
        # subdirectory → index_directory writes rel_path RELATIVE TO THE
        # SUBDIRECTORY instead of to media_dir, which creates duplicates/cache
        # misses compared to the full scan. Clean fix: the indexer needs a
        # separate rel base (media_dir) while only the subdirectory is walked —
        # that is a signature change in scanner/indexer.py (a different file)
        # and therefore not feasible here. The stored abs_path remains the
        # stable key (the scan matches on it), which limits the damage.
        vdir = safe_path(config.media_dir, sub)
    if not vdir.exists():
        return jsonify({"error": "Directory not found"}), 404

    def stream():
        from whisper_tm.scanner.indexer import index_directory

        # #128: couple client disconnect to the indexer. If the browser
        # breaks the SSE connection, the generator is closed (GeneratorExit) →
        # we set the stop flag and close the indexer generator so its
        # ThreadPool finishes/discards the still-open FFprobe batch instead of
        # continuing to sample the entire collection.
        stop = {"v": False}
        gen = index_directory(
            vdir, recursive=recursive, db=ext.db, force=force, should_stop=lambda: stop["v"]
        )
        try:
            for prog in gen:
                yield f"data: {json.dumps(prog)}\n\n"
        except Exception as e:  # noqa
            logger.exception("Indexing error")
            yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
        finally:
            stop["v"] = True
            gen.close()

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@main_bp.route("/api/browse")
def browse_directory():
    """Browse filesystem directories."""
    raw_path = request.args.get("path", "").strip()

    # The directory browser stays within the allowed trees. Previously it
    # listed every directory on the host — a complete disclosure of the
    # filesystem structure, without login.
    from whisper_tm.core.security import allowed_roots, owner_request, root_allowed

    roots = allowed_roots()
    # The owner (this machine, or through the login) may look for the
    # library anywhere — see owner_request()
    anywhere = owner_request()
    if not roots and not anywhere:
        return jsonify({"error": "No allowed directories configured (security.allowed_roots)"}), 403

    if not raw_path:
        # Without a value, show the first allowed root, not "/".
        p = roots[0] if roots else Path.home()
    else:
        p = Path(raw_path)
        try:
            p = p.resolve()
        except OSError:
            pass

    attempts = 0
    while (not p.exists() or not p.is_dir()) and attempts < 50:
        parent = p.parent
        if parent == p:
            break
        p = parent
        attempts += 1

    if not anywhere and not root_allowed(p):
        return jsonify(
            {
                "error": "Directory is outside the allowed directories (security.allowed_roots)",
                "path": str(roots[0]),
                "dirs": [],
            }
        ), 403

    dirs_list = []
    # Windows: Drive letters
    import platform

    if platform.system() == "Windows" and str(p) in ("\\", "C:\\", "/"):
        import string

        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                dirs_list.append({"name": drive, "path": drive})
        return jsonify({"path": str(p), "dirs": dirs_list})

    # Only offer ".." as long as the parent folder itself is still allowed —
    # otherwise going up leads into a 403 instead of into a directory.
    if str(p) != "/" and p.parent != p and (anywhere or root_allowed(p.parent)):
        dirs_list.append({"name": "..", "path": str(p.parent)})

    try:
        with os.scandir(str(p)) as entries:
            subdirs = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=True):
                        subdirs.append(entry.name)
                except OSError:
                    continue

        subdirs.sort(key=str.lower)
        for name in subdirs:
            dirs_list.append({"name": name, "path": str(p / name)})
    except PermissionError:
        return jsonify({"path": str(p), "dirs": dirs_list, "error": "No permission"})

    return jsonify({"path": str(p), "dirs": dirs_list})


# ── Filesystem watcher API ────────────────────────────────────────────────


@main_bp.route("/api/watcher/status")
def watcher_status():
    """Filesystem watcher status."""
    from whisper_tm.extensions import file_watcher

    if file_watcher:
        return jsonify(file_watcher.status())
    return jsonify({"running": False, "enabled": False})


# ── Watching for new files, and transcribing them (core/autoscan.py) ───────


@main_bp.route("/api/autoscan")
def autoscan_status():
    """Continuous scanning: on or off, the folders, the last round, the
    files it found and what became of them."""
    from whisper_tm.core.autoscan import AUTOSCAN

    return jsonify(AUTOSCAN.status())


@main_bp.route("/api/autoscan", methods=["POST"])
def autoscan_switch():
    """Switch continuous scanning on or off, or choose what it watches —
    saved, as the settings page does. Body (any of):

    ``enabled`` (bool), ``transcribe`` (bool);
    ``mode``: ``off`` | ``library`` (every allowed folder tree) |
    ``folders`` (only the chosen folders);
    ``watch`` / ``unwatch``: one absolute folder to add to or take from the
    chosen folders (watching one switches to the folders mode).
    """
    from whisper_tm.core import paths as _paths
    from whisper_tm.core.autoscan import AUTOSCAN
    from whisper_tm.core.security import root_allowed

    d = request.get_json(silent=True) or {}
    if not any(k in d for k in ("enabled", "transcribe", "mode", "watch", "unwatch")):
        return jsonify(
            {"error": "nothing to change (enabled, transcribe, mode, watch, unwatch)"}
        ), 400
    folders = [
        str(x) for x in (config.get("autoscan", "folders", default=[]) or []) if str(x).strip()
    ]
    for key in ("watch", "unwatch"):
        if key in d:
            f = _paths.canon(str(d.get(key) or ""))
            if not f or not root_allowed(f):
                return jsonify({"error": f"{key}: not a folder of the library"}), 400
            canon = [_paths.canon(x) for x in folders]
            if key == "watch" and f not in canon:
                folders.append(f)
            if key == "unwatch":
                folders = [x for x, c in zip(folders, canon) if c != f]
            config.set("autoscan", "folders", folders)
            if key == "watch":
                config.set("autoscan", "enabled", True)
            elif not folders:
                # Nothing chosen left: off, not "everything"
                config.set("autoscan", "enabled", False)
    mode = d.get("mode")
    if mode not in (None, "off", "library", "folders"):
        return jsonify({"error": "mode: off, library or folders"}), 400
    if mode == "off":
        config.set("autoscan", "enabled", False)
    elif mode == "library":
        config.set("autoscan", "folders", [])
        config.set("autoscan", "enabled", True)
    elif mode == "folders":
        if not folders:
            return jsonify({"error": "no folder chosen yet — watch one first"}), 400
        config.set("autoscan", "enabled", True)
    for key in ("enabled", "transcribe"):
        if key in d:
            config.set("autoscan", key, bool(d[key]))
    try:
        config.save()
    except Exception as e:  # noqa: BLE001 — it still runs, just not after a restart
        logger.warning("autoscan: settings not saved: %s", e)
    AUTOSCAN.apply()
    return jsonify(AUTOSCAN.status())


@main_bp.route("/api/autoscan/scan", methods=["POST"])
def autoscan_scan():
    """A round now, not at the next interval."""
    from whisper_tm.core.autoscan import AUTOSCAN

    if not AUTOSCAN.running:
        return jsonify({"error": "continuous scanning is off"}), 409
    AUTOSCAN.scan_now()
    return jsonify({"ok": True})


# ── File navigation (for Search → File) ────────────────────────────────────


@main_bp.route("/api/navigate_to_file", methods=["POST"])
def navigate_to_file():
    """Finds a file, switches to the right directory, and returns the
    file entry. All server-side — no JS guessing needed.

    Body: {"filename": "video.mp4", "media_dir": "/optional/hint", "debug": true}
    Returns: {"found": true, "directory": "...", "file": {...}, "files": [...], "debug_log": [...]}
    """
    data = request.json or {}
    filename = data.get("filename", "").strip()
    hint_dir = data.get("media_dir", "").strip()
    debug = data.get("debug", False)

    if not filename:
        return jsonify({"error": "No filename"}), 400

    # A full path names the file itself — the voice map sends one for a
    # recording outside the folder it was loaded for. The steps below join
    # `filename` to a directory (paths.abs_from), which reads "/mnt/…" as
    # relative: step 1a found the file, step 3 looked for "<dir>/mnt/…",
    # switched the list to the wrong folder and answered "File not found"
    # for a file that was there. Its folder is the place to look.
    if paths.is_absolute(filename):
        whole = Path(paths.canon(filename))
        if not root_allowed(whole.parent):
            return jsonify(
                {
                    "found": False,
                    "error": "The file lies outside the allowed "
                    "directories (security.allowed_roots)",
                }
            ), 403
        hint_dir, filename = str(whole.parent), whole.name

    # From here on, basename is ONLY needed for the filesystem search (1d)
    # and the mapping in the scan result — no longer for DB lookups. There
    # the identity decides (docs/PATHS.md).
    basename = Path(filename).name
    dbg = []  # debug log

    def _dbg(msg):
        dbg.append(msg)
        logger.debug("[navigate] %s", msg)

    _dbg(f"START: filename='{filename}', hint_dir='{hint_dir}'")
    _dbg(f"Current media_dir: '{config.media_dir}'")

    # Nothing outside the allowed trees is looked at: not the hint, not a
    # path from the database, not a folder to search. The search used to
    # run first and the root check only at the end — a hint like
    # "/home/<user>/x" walked the home folder, and the debug log of the
    # refusal named the file it had found there.
    if hint_dir and not root_allowed(hint_dir):
        _dbg("0. Hint outside the allowed directories — ignored")
        hint_dir = ""

    # ── Step 1: find directory ─────────────────────────────────────
    target_dir = None

    # 1a. Hint from search result (media_dir from DB)
    # No more basename candidate (docs/PATHS.md): (hint_dir / "intro.mp4")
    # often exists even though a/intro.mp4 was meant — navigation then
    # landed on the wrong file.
    if hint_dir:
        p = Path(hint_dir) / filename
        _dbg(f"1a. Check hint: '{p}' → exists={p.exists()}")
        if p.exists():
            target_dir = hint_dir
            _dbg("1a. FOUND via hint_dir")

    # 1b. Check DB media_dir
    if not target_dir and ext.db:
        try:
            with ext.db.connection() as conn:
                row = ext.db.resolve_transcript(
                    conn, filename, str(config.media_dir), columns="abs_path, rel_path, media_dir"
                )
                _dbg(f"1b. DB resolution → {dict(row) if row else 'NOT FOUND'}")
                if row and row["abs_path"] and not root_allowed(row["abs_path"]):
                    _dbg("1b. DB path outside the allowed directories — ignored")
                    row = None
                if row:
                    # The stored absolute path is the identity: if the file
                    # is there, its media directory is the target.
                    if row["abs_path"] and Path(row["abs_path"]).exists():
                        target_dir = row["media_dir"] or str(Path(row["abs_path"]).parent)
                        _dbg(f"1b. FOUND via abs_path → '{target_dir}'")
                    elif row["media_dir"] and root_allowed(row["media_dir"]):
                        p = Path(row["media_dir"]) / row["rel_path"]
                        _dbg(f"1b. Check media_dir: '{p}' → exists={p.exists()}")
                        if p.exists():
                            target_dir = row["media_dir"]
        except Exception as e:
            _dbg(f"1b. DB error: {e}")

    # 1c. Check current media_dir
    if not target_dir:
        p = config.media_dir / filename
        _dbg(f"1c. Check current media_dir: '{p}' → exists={p.exists()}")
        if p.exists():
            target_dir = str(config.media_dir)
            _dbg("1c. FOUND in current media_dir")

    # 1c2. DB lookup in media_files (#118): resolves the file via the stored
    #      abs_path instead of immediately running the expensive rglob full
    #      walk (up to 50k files) in the request thread. Just a single DB lookup.
    if not target_dir and ext.db:
        try:
            with ext.db.connection() as conn:
                row = conn.execute(
                    "SELECT abs_path, rel_path FROM media_files "
                    "WHERE abs_path != '' AND (abs_path = ? OR rel_path = ?) "
                    "ORDER BY LENGTH(abs_path) LIMIT 1",
                    (paths.abs_from(filename, str(config.media_dir)), filename),
                ).fetchone()
                if row and row["abs_path"] and root_allowed(row["abs_path"]):
                    ap = Path(row["abs_path"])
                    _dbg(f"1c2. media_files abs_path='{ap}' → exists={ap.exists()}")
                    if ap.exists() and ap.is_file():
                        target_dir = str(ap.parent)
                        filename = ap.name
                        _dbg("1c2. FOUND via media_files DB lookup")
        except Exception as e:
            _dbg(f"1c2. DB error: {e}")

    # 1d. Recursive search
    if not target_dir:
        search_roots = []
        if config.media_dir.exists():
            search_roots.append(config.media_dir)
        if hint_dir:
            hint_path = Path(hint_dir)
            if hint_path.exists() and hint_path not in search_roots:
                search_roots.append(hint_path)
            if (
                hint_path.parent.exists()
                and hint_path.parent not in search_roots
                and root_allowed(hint_path.parent)
            ):
                search_roots.append(hint_path.parent)

        _dbg(f"1d. Recursive search in {len(search_roots)} roots: {[str(r) for r in search_roots]}")

        from whisper_tm.scanner.filesystem import iter_files_named

        for root in search_roots:
            try:
                # iter_files_named instead of rglob: rglob also descends into
                # hidden directories (.zfs/snapshot → the entire tree once
                # more per snapshot, plus SMB-lethal special files from
                # recovery remnants). iter_files_named respects
                # scanner.ignore_patterns and matches the name exactly (no
                # glob → no escaping needed, #119). Collect all name matches
                # and prefer the one whose relative path matches the
                # request. Taking the first hit was a coin-flip decision for
                # same-named files in multiple folders — and regularly
                # landed on the wrong one.
                candidates = [p for p in iter_files_named(root, basename) if p.is_file()]
                if candidates:
                    matching = [p for p in candidates if paths.rel_from(p, root) == filename]
                    hits = matching[0] if matching else candidates[0]
                    if not matching and len(candidates) > 1:
                        _dbg(
                            f"1d. AMBIGUOUS: {len(candidates)} files are named "
                            f"'{basename}' — using '{hits}'"
                        )
                    target_dir = str(root)
                    filename = paths.rel_from(hits, root) or basename
                    _dbg(f"1d. FOUND: '{hits}' → target_dir='{target_dir}', filename='{filename}'")
            except (OSError, PermissionError) as e:
                _dbg(f"1d. rglob error in '{root}': {e}")
            if target_dir:
                break

    if not target_dir:
        _dbg("FAILED: File not found on filesystem")
        resp = {
            "found": False,
            "error": f"File '{basename}' not found on the filesystem",
        }
        if debug:
            resp["debug_log"] = dbg
        return jsonify(resp)

    # Jump into the file's FOLDER: the search must not land in a huge
    # root, where the file is stuck somewhere in the list —
    # it should show its location (Explorer behavior "open file path").
    # Root = parent folder, entry = basename along with its neighbors.
    _abs = Path(paths.abs_from(filename, target_dir) or (Path(target_dir) / filename))
    if _abs.is_file():
        target_dir = str(_abs.parent)
        filename = _abs.name
        _dbg(f"1e. Folder jump: target_dir='{target_dir}', filename='{filename}'")

    _dbg(f"Target dir: '{target_dir}', filename: '{filename}'")

    # ── Step 2: switch directory ────────────────────────────────────
    # #116: serialize the mutation and record the set path as a snapshot so a
    # parallel writer doesn't bend the response toward a different directory.
    # (DEFERRED: cleaner would be a per-request target directory instead of
    # global.) target_dir comes indirectly from the request (hint_dir or DB
    # hit) and is about to become the global root — so check as with
    # /api/set_directory.
    if not root_allowed(target_dir):
        # No debug log here: it would name what lies outside
        logger.debug("[navigate] 2. REJECTED (root not allowed): %s", target_dir)
        return jsonify(
            {
                "found": False,
                "error": "Target directory lies outside the allowed "
                "directories (security.allowed_roots)",
            }
        ), 403
    try:
        with _media_dir_lock:
            config.media_dir = target_dir
            switched_dir = str(config.media_dir)
        _dbg(f"2. Directory switched to: '{switched_dir}'")
    except ValueError as e:
        _dbg(f"2. FAILED: {e}")
        resp = {"found": False, "error": str(e)}
        if debug:
            resp["debug_log"] = dbg
        return jsonify(resp)

    # ── Step 3: resolve ONE file (point query, NO walk) ──────────────
    # The absolute path is known and exists after step 1. Previously,
    # scan_media_directory(recursive=True) ran here — a complete NAS walk
    # (minutes on a large library) just to build the list entry for the ONE
    # file. Three quick clicks = three parallel full walks, and the file
    # only opened after the last one. Now: an
    # indexed DB lookup like /api/scan_entry; if the DB doesn't yet know the
    # file (freshly placed), register_single_file registers it
    # immediately — without a walk.
    found_file = None
    abs_path = paths.abs_from(filename, target_dir)
    if ext.db and abs_path:
        from whisper_tm.scanner import db_row_to_entry

        try:
            row = ext.db.get_media_file_at(abs_path)
            if row is None:
                from whisper_tm.scanner import register_single_file

                _dbg("3. Not in DB — registering single file")
                register_single_file(Path(abs_path))
                row = ext.db.get_media_file_at(abs_path)
            if row is not None:
                found_file = db_row_to_entry(row, target_dir)
                _dbg(f"3. DB point query: '{found_file.get('name') if found_file else None}'")
        except Exception as e:
            _dbg(f"3. DB error: {e}")

    if not found_file and abs_path:
        # Step 1 has already verified the file on disk — if DB lookup AND
        # registration both fail (transient CIFS error, DB locked),
        # navigation must still not report "not found". Build a minimal
        # entry directly from the filesystem; the
        # DB catches up on the next scan/index run.
        try:
            from whisper_tm.scanner import db_row_to_entry

            st = Path(abs_path).stat()
            found_file = db_row_to_entry(
                {
                    "abs_path": abs_path,
                    "file_size": st.st_size,
                    "modified_at": st.st_mtime,
                    "is_video": Path(abs_path).suffix.lower() in config.video_extensions,
                    "status": "pending",
                },
                target_dir,
            )
            _dbg("3. Fallback: entry built directly from the filesystem")
        except OSError as e:
            _dbg(f"3. Fallback stat failed: {e}")

    if not found_file:
        _dbg(f"3. NO MATCH: '{filename}' neither in DB nor registrable")

    # Response without the former full list (files/stats/total_files): both
    # consumers (search.js goToSearchResult, settings.js navigation test)
    # only read found/directory/file — the frontend loads the list paginated
    # itself afterward (reloadFileList).
    resp = {
        "found": found_file is not None,
        "directory": switched_dir,
        "file": found_file,
        "filename_used": filename,
        "basename": basename,
    }
    if debug:
        resp["debug_log"] = dbg
    return jsonify(resp)

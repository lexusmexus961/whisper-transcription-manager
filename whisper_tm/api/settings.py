"""
Settings API: read, write, validate, export configuration.
"""

import collections
import logging
import threading
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from whisper_tm.config import (
    CHECKED_DIRECTORIES,
    SECRET_PLACEHOLDER,
    coerce_value,
    config,
    is_placeholder,
    is_secret,
    unmask_embedded,
)
from whisper_tm.core.security import request_is_local, root_allowed
from whisper_tm.extensions import db, hw

logger = logging.getLogger(__name__)


settings_bp = Blueprint("settings", __name__)

_STATS_TTL = 30.0
# Two lock levels: the "state" lock protects fast cache access (no torn reads
# of cache/ts pairs), the "compute" lock enforces single-flight — on parallel
# cache misses only ONE request computes, others wait and use the fresh result
# (no thundering herd, #178/#179). The compute lock is NEVER taken with the
# state lock in reverse order → no deadlock.
#
# Here only the cache for /api/db/stats (settings panel) remains. Actual
# statistics are moved to api/stats.py and bring their own cache.
_db_stats_lock = threading.Lock()
_db_stats_compute_lock = threading.Lock()
_db_stats_cache: dict | None = None
_db_stats_cache_ts: float = 0.0


def invalidate_stats_cache() -> None:
    """Call after a transcription has finished."""
    global _db_stats_cache_ts
    _db_stats_cache_ts = 0.0
    try:  # an analysis plan counted before it is stale too
        from whisper_tm.core.analysis import clear_plan_cache

        clear_plan_cache()
    except Exception:  # noqa: BLE001 — a cache, never a failure
        pass


#: Settings the page cannot change — only config.yaml can. They decide who
#: may reach the app, what it may touch and which programs it starts; the
#: settings API needs no login while auth.enabled is off, and a request that
#: changes them could switch the app's own protection off (auth.public_paths
#: set to "/"), move it to another database, or put a command into the
#: llama.cpp start script.
_FILE_ONLY = (
    "security.",
    "auth.",
    "server.",
    "logging.",
    "llama.",
    "database.path",
    "embedding.local_extra_args",
    "mcp.token",
    # the reranker is the same llama-server start
    "rerank.extra_args",
    "rerank.local_model",
    "rerank.host",
    # a database of its own, written where it says
    "audio_analysis.path",
    # whether links into the local network are fetched
    "youtube.allow_private_hosts",
)


def _file_only(key: str) -> bool:
    """Is ``key`` — or a section that holds a file-only key — file-only?

    Not only the keys below a file-only prefix: the section itself, too.
    ``{"auth": false}`` flattens to the key "auth", which no "auth." prefix
    matches — and replaced the whole login section (password, token,
    enabled) with ``false``. Same for "database" (its path), "embedding"
    and "mcp" (their file-only keys inside).
    """
    if key.startswith(_FILE_ONLY) or key in _FILE_ONLY:
        return True
    return any(f.startswith(key + ".") for f in _FILE_ONLY)


#: Settings only a browser on this machine may change while there is no
#: login — unchanged values sent back by the page pass:
#: * whether the MCP network services need their token, and which address
#:   they listen on: from the LAN, one request opened the MCP tools
#:   (transcribing, downloading) to the whole network;
#: * where the MCP server reaches the app and with which token, and the
#:   address its player links point to: moved to a foreign server, the
#:   token went there and the tools talked to an impostor;
#: * the Python environment Images & sounds starts, and its folders: a
#:   program of anyone's choosing would run;
#: * the app's folders: uploads and clips written, a work folder emptied
#:   wherever a request says;
#: * the local embedding server's and the Images & sounds service's address,
#:   port and model: a llama-server or media service listening on the LAN
#:   without a key, any file loaded as a model, any repository fetched;
#: * where the vector store and the database backups are written.
_LOCAL_ONLY = (
    "mcp.host",
    "mcp.require_token",
    "mcp.base_url",
    "mcp.listen_base_url",
    "mcp.api_token",
    "media_search.environment",
    "media_search.path",
    "media_search.cache_dir",
    "media_search.model",
    "media_search.host",
    "media_search.port",
    "embedding.local_host",
    "embedding.local_port",
    "embedding.local_model",
    "rag.path",
    "database.backup_path",
    *(f"directories.{name}" for name in CHECKED_DIRECTORIES),
)

#: Endpoint and the key that is sent to it. A key is only ever sent to the
#: endpoint it was entered for: changing the endpoint without entering a key
#: again forgets the stored one — otherwise one request moving the endpoint
#: to a foreign server had the server send it the saved API key.
_ENDPOINT_KEYS = (("llm.endpoint", "llm.api_key"), ("embedding.endpoint", "embedding.api_key"))


def _may_change_protection() -> bool:
    """May this request change _LOCAL_ONLY settings / see the MCP token?"""
    from whisper_tm.core.auth import is_auth_enabled

    return is_auth_enabled() or request_is_local(request.remote_addr, request.headers)


#: yt-dlp options the settings page may add to every download. The value is
#: split like a shell line and handed to yt-dlp, and some of its options run
#: programs or write anywhere (--exec, --output, --ffmpeg-location,
#: --downloader, --netrc-cmd, --config-locations …). So the page takes the
#: options that change WHAT is fetched and HOW politely; anything else goes
#: into config.yaml by hand.
_YTDLP_OPTIONS = frozenset(
    """
    --limit-rate -r --throttled-rate --retries -R --fragment-retries
    --extractor-retries --file-access-retries --retry-sleep --socket-timeout
    --sleep-interval --min-sleep-interval --max-sleep-interval
    --sleep-requests --sleep-subtitles --concurrent-fragments -N
    --format -f --format-sort -S --format-sort-force --merge-output-format
    --audio-format --audio-quality --prefer-free-formats --no-prefer-free-formats
    --check-formats --no-check-formats
    --sub-langs --write-subs --no-write-subs --write-auto-subs
    --no-write-auto-subs --sub-format --embed-subs --no-embed-subs
    --embed-metadata --no-embed-metadata --embed-chapters --no-embed-chapters
    --embed-thumbnail --no-embed-thumbnail
    --no-playlist --yes-playlist --playlist-items -I --max-downloads
    --match-filters --no-match-filters --date --datebefore --dateafter
    --min-filesize --max-filesize --age-limit
    --force-ipv4 -4 --force-ipv6 -6 --geo-bypass --geo-bypass-country --xff
    --source-address --impersonate
    --extractor-args --add-header --referer --user-agent
    --live-from-start --no-live-from-start --wait-for-video --no-wait-for-video
    --sponsorblock-remove --sponsorblock-mark --no-sponsorblock
    --restrict-filenames --no-restrict-filenames --windows-filenames
    --trim-filenames --no-mtime --mtime --no-part --part
    --abort-on-error --no-abort-on-error --ignore-errors -i
    --mark-watched --no-mark-watched
""".split()
)

#: The options above that take no value — every other one takes the next
#: word (or its own "=value"). A word that is neither an option nor a value
#: would land before "--" in the command and become a download target.
_YTDLP_FLAGS = frozenset(
    """
    --prefer-free-formats --no-prefer-free-formats --check-formats --no-check-formats
    --format-sort-force --write-subs --no-write-subs --write-auto-subs --no-write-auto-subs
    --embed-subs --no-embed-subs --embed-metadata --no-embed-metadata --embed-chapters
    --no-embed-chapters --embed-thumbnail --no-embed-thumbnail --no-playlist --yes-playlist
    --no-match-filters --force-ipv4 -4 --force-ipv6 -6 --geo-bypass
    --live-from-start --no-live-from-start --no-wait-for-video --no-sponsorblock
    --restrict-filenames --no-restrict-filenames --windows-filenames
    --no-mtime --mtime --no-part --part --abort-on-error --no-abort-on-error
    --ignore-errors -i --mark-watched --no-mark-watched
""".split()
)

#: Lower bounds of values a job divides by or counts with — 0 there fails
#: every long file (the page itself also refuses them).
_AT_LEAST = {
    "hardware.asr_block_max_sec": 60,
    "hardware.sep_chunk_duration_sec": 60,
    "hardware.gpu_index": 0,
    "search.max_results": 1,
    "rag.chunk_max_chars": 100,
    "rag.batch_size": 1,
    "embedding.local_batch": 128,
}


def _check_ytdlp_args(value: Any) -> str:
    """Why ``value`` can't be saved as youtube.extra_args ("" = it can)."""
    from whisper_tm.core import commands

    try:
        words = commands.split_args(str(value or ""))
    except ValueError as e:
        return f"not readable as a command line ({e})"
    refused, bare, i = [], [], 0
    while i < len(words):
        word = words[i]
        name = word.split("=", 1)[0]
        if not word.startswith("-"):
            bare.append(word)
            i += 1
        elif name not in _YTDLP_OPTIONS:
            refused.append(name)
            i += 1
        elif name in _YTDLP_FLAGS or "=" in word:
            i += 1
        elif i + 1 < len(words):
            i += 2  # the option and its value
        else:
            return f"{name} needs a value"
    if refused:
        return (
            "not allowed from the settings page: "
            + ", ".join(sorted(set(refused)))
            + " — options that run programs, choose files or change how the "
            "connection is made (--proxy, --no-check-certificates) can only "
            "be set in config.yaml"
        )
    if bare:
        return (
            f"{bare[0]!r} is not an option — every word must be an option or "
            f"the value of the one before it"
        )
    return ""


@settings_bp.route("/api/settings")
def get_settings():
    """Complete configuration as JSON — secrets not in plain text.

    ``defaults`` is config.default.yaml: the page marks what differs from
    it and offers the way back per setting."""
    from whisper_tm.core import vram
    from whisper_tm.core.version import version_info

    data, defaults = config.all_masked(), config.defaults_masked()
    # hardware.low_vram "auto" as what it means on this machine: the page's
    # switch and the dialogs' preset are on/off, and a small card's answer
    # is on (core/vram.py)
    for shown in (data, defaults):
        hw = shown.get("hardware")
        if isinstance(hw, dict) and "low_vram" in hw:
            hw["low_vram"] = vram.resolve(hw["low_vram"])
    return jsonify(
        {
            "config": data,
            "defaults": defaults,
            "dirs": {k: str(v) for k, v in config.dirs.items()},
            "data_dir": str(config.data_dir),
            "db_path": str(config.db_path),
            "version": version_info(),
            "config_path": str(config.config_path) if config.config_path else "",
            "warnings": config.validate(),
        }
    )


@settings_bp.route("/api/settings", methods=["PUT"])
def update_settings():
    """Update configuration (partial update)."""
    updates = request.json
    if not updates or not isinstance(updates, dict):
        return jsonify({"error": "No settings"}), 400

    errors = []
    dirs_changed = False

    # LLM providers go through core/llm_provider.save(): only there
    # is a saved API key preserved if the form leaves the field empty
    # (UI never sends back keys). If the list were written directly to config
    # here, the key would be gone on next save.
    llm_updates = updates.get("llm")
    if isinstance(llm_updates, dict) and isinstance(llm_updates.get("providers"), list):
        try:
            from whisper_tm.core import llm_provider as _lp

            # in_file follows the request's ?save flag — providers should
            # behave like any other setting (the later
            # config.save() below picks them up, since they're already in the
            # running config).
            _lp.save([p for p in llm_updates["providers"] if isinstance(p, dict)], in_file=False)
        except Exception as e:
            errors.append(f"llm.providers: {e}")
        llm_updates = dict(llm_updates)
        llm_updates.pop("providers", None)
        updates = dict(updates)
        updates["llm"] = llm_updates

    flat = dict(_flatten_dict(updates))

    # Endpoint/key pairs whose endpoint this request really changes: their
    # stored key is forgotten unless the request brings a new one
    def _as_stored(key, value):
        if isinstance(value, str) and SECRET_PLACEHOLDER in value:
            kept = unmask_embedded(value, config.get(key))
            return kept if kept is not None else value
        return value

    moved_endpoints = [
        (ep, kk)
        for ep, kk in _ENDPOINT_KEYS
        if ep in flat
        and str(_as_stored(ep, flat[ep]) or "").strip()
        != str(config.get(ep, default="") or "").strip()
    ]

    for key, value in flat.items():
        try:
            # security.allowed_roots, the login and the server address must
            # not be changeable through the running app — otherwise a
            # single request could switch the protection off.
            if _file_only(key):
                errors.append(f"{key}: only changeable in the configuration file")
                continue
            # A whole section replaced by a scalar or a list ({"scanner": 0})
            # would take every setting in it along
            default = config.default(key)
            if isinstance(default, dict) and not isinstance(value, dict):
                errors.append(f"{key}: is a section, not a single value")
                continue
            # A folder is checked when it changes — a page that sends every
            # field back must not fail on a value set long ago by hand
            if (
                key.startswith("directories.")
                and key.split(".", 1)[1] in CHECKED_DIRECTORIES
                and str(value or "").strip() != str(config.get(key) or "").strip()
            ):
                why = config.check_directory(key.split(".", 1)[1], value)
                if why:
                    errors.append(f"{key}: {why}")
                    continue
            # The masking placeholder comes back from GET /api/settings when
            # the user hasn't touched the field. Writing it would overwrite
            # the real secret.
            if is_secret(key) and is_placeholder(value):
                continue
            # A password inside a URL or a header comes back masked from
            # GET: unchanged, the stored value stays; changed around it,
            # the credentials must be typed again
            if isinstance(value, str) and SECRET_PLACEHOLDER in value:
                kept = unmask_embedded(value, config.get(key))
                if kept is None:
                    errors.append(f"{key}: type the password in it again — it is hidden here")
                    continue
                value = kept
            # The type of the default: a <select> hands out strings, and
            # "false" is true in Python
            if default is not None:
                value = coerce_value(value, default)
            if key in _LOCAL_ONLY and value != config.get(key) and not _may_change_protection():
                errors.append(
                    f"{key}: only changeable from this machine's own "
                    f"browser, or with the login switched on"
                )
                continue
            if key in _AT_LEAST:
                try:
                    too_small = float(value) < _AT_LEAST[key]
                except (TypeError, ValueError):
                    too_small = True
                if too_small:
                    errors.append(f"{key}: must be a number of at least {_AT_LEAST[key]}")
                    continue
            if key == "youtube.extra_args":
                why = _check_ytdlp_args(value)
                if why:
                    errors.append(f"{key}: {why}")
                    continue
            if key == "transcription.hallucination_patterns":
                # A pattern that doesn't compile would be skipped at every
                # transcription — refuse it here, where it can be fixed
                from whisper_tm.core.helpers import compile_artifact_patterns

                _, bad = compile_artifact_patterns(value)
                if bad:
                    errors.append(f"{key}: not a valid regular expression — " + "; ".join(bad))
                    continue
            # ONLY directories.media is a MEDIA root and must lie within the
            # allowed trees: it is the basis of every safe_path check — via
            # {"directories":{"media":"/"}} every path check could otherwise
            # be disabled. It goes through the same setter as the directory
            # bar (/api/set_directory): existence check, the uploads folder
            # that follows it, and the change notice the file watcher needs.
            # The OTHER directories.* keys are app working directories
            # (work/logs/clips/…): they DELIBERATELY lie outside the media
            # root (work belongs on the local SSD) and only ever receive
            # app-generated or secure_filename-sanitized names.
            if key == "directories.media":
                path = str(value or "").strip()
                if not path:
                    errors.append(f"{key}: the media folder can't be empty")
                elif not root_allowed(path) and not _may_change_protection():
                    errors.append(
                        f"{key}: lies outside the allowed directories "
                        f"(security.allowed_roots) — a new library folder can be "
                        f"chosen on the machine the server runs on, or with the login on"
                    )
                else:
                    # The owner may move the library anywhere (core/security.py:
                    # owner_request) — the tree counts as allowed from here on
                    from whisper_tm.core.security import remember_root

                    new_tree = not root_allowed(path)
                    config.media_dir = path
                    if new_tree:
                        remember_root(path)
                continue
            config.set(key, value)
            if key.startswith("directories."):
                dirs_changed = True
        except Exception as e:
            errors.append(f"{key}: {e}")

    for ep, kk in moved_endpoints:
        given = flat.get(kk)
        if config.get(ep, default="") == flat[ep] and not (
            isinstance(given, str) and given.strip() and not is_placeholder(given)
        ):
            config.set(kk, "")

    # Activate directory changes: config.set() only changes raw data, resolved
    # paths need a re-resolve.
    if dirs_changed:
        try:
            config._resolve_directories()
            config._ensure_directories()
        except Exception as e:
            errors.append(f"Directories: {e}")

    warnings = config.validate()

    # Continuous scanning follows its settings right away (on/off, folders,
    # interval) — no restart
    if any(k.startswith("autoscan.") for k in flat):
        try:
            from whisper_tm.core.autoscan import AUTOSCAN

            AUTOSCAN.apply()
        except Exception as e:
            errors.append(f"autoscan: {e}")

    # Optional save
    save = request.args.get("save", "false").lower() in ("true", "1")
    saved_path = None
    if save:
        try:
            saved_path = str(config.save())
        except Exception as e:
            errors.append(f"Saving: {e}")

    # #177: on partial errors (individual keys, directories, or save failed)
    # must not blanketly report ok:true.
    return jsonify(
        {
            "ok": not errors,
            "warnings": warnings,
            "errors": errors,
            "saved": saved_path,
        }
    )


@settings_bp.route("/api/settings/artifact_test", methods=["POST"])
def artifact_test():
    """Try artifact patterns before saving them.

    Body: ``patterns`` (lines, as typed — not the saved ones) and ``text``
    (a segment text). Answers which line matches the text, and which lines
    are not valid regular expressions.
    """
    from whisper_tm.core.helpers import artifact_reason, compile_artifact_patterns

    data = request.get_json(silent=True) or {}
    compiled, errors = compile_artifact_patterns(data.get("patterns") or [])
    text = str(data.get("text") or "")
    why = artifact_reason(text, compiled) if text.strip() else None
    return jsonify({"dropped": why is not None, "reason": why, "errors": errors})


@settings_bp.route("/api/settings/export")
def export_settings():
    """Export configuration as YAML."""
    from flask import Response

    yaml_str = config.export_config()
    return Response(
        yaml_str,
        mimetype="text/yaml",
        headers={"Content-Disposition": "attachment; filename=config.yaml"},
    )


@settings_bp.route("/api/settings/directories")
def get_directories():
    """Resolved directories."""
    dirs = config.dirs
    info = {}
    for name, path in dirs.items():
        exists = path.exists()
        writable = exists and _is_writable(path)
        info[name] = {
            "path": str(path),
            "exists": exists,
            "writable": writable,
        }
        if exists:
            try:
                import shutil

                usage = shutil.disk_usage(str(path))
                info[name]["disk_total_gb"] = round(usage.total / (1024**3), 1)
                info[name]["disk_free_gb"] = round(usage.free / (1024**3), 1)
            except Exception as e:
                logger.debug("Unhandled exception: %s", e)
    return jsonify(info)


@settings_bp.route("/api/hardware")
def get_hardware():
    """Hardware profile as JSON."""
    if not hw:
        return jsonify({"error": "Hardware not detected"}), 500
    return jsonify(
        {
            "os": hw.os_name,
            "os_version": hw.os_version,
            "arch": hw.arch,
            "cpu": hw.cpu_name,
            "cpu_cores": hw.cpu_cores,
            "cpu_threads": hw.cpu_threads,
            "ram_total_mb": hw.ram_total_mb,
            "ram_available_mb": hw.ram_available_mb,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_mb": g.memory_total_mb,
                    "memory_free_mb": g.memory_free_mb,
                    "vendor": g.vendor,
                    "driver": g.driver_version,
                }
                for g in hw.gpus
            ],
            "device": hw.best_device,
            "cuda_version": hw.cuda_version,
            "rocm_version": hw.rocm_version,
            "has_ffmpeg": hw.has_ffmpeg,
            "ffmpeg_version": hw.ffmpeg_version,
            "has_nvdec": hw.has_nvdec,
            "has_audio_separator": hw.has_audio_separator,
            "has_ytdlp": hw.has_ytdlp,
            "has_fastmcp": hw.has_fastmcp,
            "fastmcp_version": hw.fastmcp_version,
            "recommendations": {
                "compute_type": hw.recommended_compute_type,
                "sep_chunk_duration": hw.recommended_sep_chunk_duration,
                "batch_size": hw.recommended_batch_size,
            },
            # What the app chose for this machine, in one sentence
            "machine_defaults": _machine_defaults_text(hw),
            "summary": hw.summary(),
        }
    )


def _machine_defaults_text(hw) -> str:
    from whisper_tm.hardware import machine_defaults, machine_defaults_text

    return machine_defaults_text(*machine_defaults(hw))


def _listening(host: str, port: int) -> bool:
    """Does something accept connections on host:port right now?"""
    import socket

    target = "127.0.0.1" if str(host) in ("", "0.0.0.0", "::", "[::]") else str(host)
    try:
        with socket.create_connection((target, int(port)), timeout=0.3):
            return True
    except (OSError, ValueError):
        return False


@settings_bp.route("/api/mcp/access")
def mcp_access():
    """The MCP network services: their access token, and whether they run.

    The token is shown here so a client can be set up with it — the page
    that shows it is as open as the rest of the app (auth.enabled).
    """
    import sys

    import whisper_tm
    from whisper_tm.mcp_server import access_token, token_file

    m = config.get("mcp", default={}) or {}
    own = str(m.get("token") or "").strip()
    host = str(m.get("host") or "127.0.0.1")
    services = [
        {
            "key": "http",
            "name": "MCP over HTTP",
            "path": "/mcp",
            "enabled": bool(m.get("autostart_http")),
            "port": int(m.get("port") or 5399),
        },
        {
            "key": "openapi",
            "name": "OpenAPI",
            "path": "",
            "enabled": bool(m.get("autostart_openapi")),
            "port": int(m.get("openapi_port") or 5400),
        },
    ]
    for s in services:
        s["listening"] = _listening(host, s["port"])
    # The token only goes to a browser on this machine, or behind the
    # login. Without a login, anyone on the LAN read it here — and with it
    # used the MCP tools that require_token was meant to keep closed.
    shown = _may_change_protection()
    token = ""
    if shown:
        try:
            token = access_token()
        except Exception as e:  # noqa: BLE001
            logger.warning("MCP token not available: %s", e)
    # This machine's paths (the token file, the app's Python, code and
    # config) go only where the token goes — the network learned the
    # folder layout from them
    local = (
        {
            "file": str(token_file()),
            # For the stdio set-up: the Python of this app, and where its code
            # and config are — a client starts the server from anywhere, and an
            # installed copy of the package may be an older one
            "python": sys.executable,
            "package_root": str(Path(whisper_tm.__file__).resolve().parent.parent),
            "config": str(config.config_path or ""),
        }
        if shown
        else {"file": "", "python": "", "package_root": "", "config": ""}
    )
    return jsonify(
        {
            "token": token,
            "token_hidden": not shown,
            "required": bool(m.get("require_token", True)),
            "source": "config" if own else "file",
            "host": host,
            "services": services,
            **local,
        }
    )


@settings_bp.route("/api/mcp/access/renew", methods=["POST"])
def mcp_access_renew():
    """A new access token — the old one stops working at once.

    A token of one's own (mcp.token in config.yaml) is replaced too: it is
    cleared, and the generated file token takes over.
    """
    from whisper_tm.mcp_server import access_token

    if not _may_change_protection():
        return jsonify(
            {
                "error": "A new MCP token can only be made from this "
                "machine's own browser, or with the login "
                "switched on."
            }
        ), 403
    if str(config.get("mcp", "token", default="") or "").strip():
        config.set("mcp", "token", "")
        config.save()
    return jsonify({"ok": True, "token": access_token(renew=True)})


@settings_bp.route("/api/db/stats")
def db_stats():
    """Database statistics (with 30s in-memory cache, single-flight)."""
    global _db_stats_cache, _db_stats_cache_ts
    if not db:
        return jsonify({"error": "No database"}), 500
    now = time.monotonic()
    with _db_stats_lock:
        if _db_stats_cache is not None and (now - _db_stats_cache_ts) < _STATS_TTL:
            return jsonify(_db_stats_cache)
    # Single-flight (#179): only ONE computation, parallel misses wait here.
    with _db_stats_compute_lock:
        # Double-check: has another request computed fresh in the meantime?
        now = time.monotonic()
        with _db_stats_lock:
            if _db_stats_cache is not None and (now - _db_stats_cache_ts) < _STATS_TTL:
                return jsonify(_db_stats_cache)
        try:
            stats = db.get_stats()
            stats["db_size_bytes"] = db.get_db_size()
            with _db_stats_lock:
                _db_stats_cache = stats
                _db_stats_cache_ts = time.monotonic()
            return jsonify(stats)
        except Exception as e:
            # Don't cache error result (transient errors must not stick).
            return jsonify(
                {
                    "files": {},
                    "transcripts": {},
                    "top_models": [],
                    "top_languages": [],
                    "db_size_bytes": 0,
                    "error": str(e),
                }
            )


def _transcript_folders(root: Path) -> list[Path]:
    """Every ``.transcripts`` folder below ``root`` — hidden folders are not
    entered (a ZFS share has the whole tree again under .zfs/snapshot)."""
    import os

    found: list[Path] = []
    for dirpath, dirnames, _files in os.walk(root, followlinks=False):
        if Path(dirpath).name == ".transcripts":
            found.append(Path(dirpath))
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".") or d == ".transcripts"]
    return found


@settings_bp.route("/api/db/sync", methods=["POST"])
def db_sync():
    """Import the JSON transcripts the database doesn't have. With the
    transcripts beside their recordings (alongside), every ``.transcripts``
    folder of the library — not only the one at its top: after losing the
    database this is the way back, and recordings in subfolders have their
    transcripts next to them."""
    if not db:
        return jsonify({"error": "No database"}), 500
    mode = config.get("directories", "transcript_mode", default="alongside")
    if mode == "alongside":
        try:
            folders = _transcript_folders(Path(config.media_dir))
        except OSError as e:
            return jsonify({"error": f"The library can't be read: {e}"}), 500
    else:
        folders = [config.transcript_dir]
    total = {"imported": 0, "skipped": 0, "errors": 0}
    for folder in folders:
        r = db.sync_from_json_dir(folder, media_dir=str(config.media_dir))
        for k in total:
            v = r.get(k, 0)
            total[k] += len(v) if isinstance(v, list) else int(v or 0)
    return jsonify({"ok": True, **total, "folders": len(folders)})


@settings_bp.route("/api/db/rebuild_index", methods=["POST"])
def rebuild_index():
    """Rebuild FTS5 index."""
    if not db:
        return jsonify({"error": "No database"}), 500
    db.rebuild_fts_index()
    return jsonify({"ok": True})


@settings_bp.route("/api/db/vacuum", methods=["POST"])
def vacuum_db():
    """Compress database (with auto-backup)."""
    if not db:
        return jsonify({"error": "No database"}), 500
    size_before = db.get_db_size()
    db.vacuum()
    size_after = db.get_db_size()
    saved = size_before - size_after
    return jsonify(
        {
            "ok": True,
            "size_bytes": size_after,
            "saved_bytes": saved,
        }
    )


@settings_bp.route("/api/db/backup", methods=["POST"])
def backup_db():
    """Create a DB backup now — where backups belong.

    Without a body it goes to ``db.backup_target()``: the configured backup
    folder (``database.backup_path``, e.g. a network share), and next to the
    database only while none is configured. It used to write a full copy into
    ``<db dir>/backups`` every time — on the same disk as the database, the
    weakest backup there is, and several GB each for a large library. A
    configured folder that can't be reached is reported, not swapped for such
    a copy.

    Body ``{"target": "external"}`` insists on ``database.backup_path``
    (400 without one). ``backups`` in the answer: how many backups the
    target holds now.
    """
    if not db:
        return jsonify({"error": "No database"}), 500

    # get_json(silent=True), not request.json: the "Create backup" button
    # posts no body, and request.json answers that with 415.
    data = request.get_json(silent=True)
    data = data if isinstance(data, dict) else {}
    configured = config.backup_dir
    if data.get("target") == "external" and not configured:
        return jsonify({"error": "No external backup destination configured"}), 400
    dest = db.backup_target()

    bp = db.backup(dest)
    if not bp:
        if configured:
            return jsonify({"error": f"Backup destination unreachable: {dest}"}), 500
        return jsonify({"error": "Backup failed"}), 500
    # A manual backup into the configured folder counts as a checkpoint —
    # otherwise the counter would trigger a second, identical one shortly after.
    if configured:
        db._reset_backup_counter()
    try:
        count = len(list(dest.glob(f"{Path(db._db_path).stem}_*.db")))
    except OSError:
        count = None
    return jsonify(
        {
            "ok": True,
            "backup": bp.name,
            "path": str(bp),
            "size_bytes": bp.stat().st_size,
            "backups": count,
        }
    )


@settings_bp.route("/api/db/backup_status")
def backup_status():
    """Status of automatic external backup (for settings)."""
    if not db:
        return jsonify({"error": "No database"}), 500

    path = config.backup_dir
    try:
        every = int(config.get("database", "backup_every_changes", default=0))
    except (TypeError, ValueError):
        every = 0

    out = {
        "path": str(path) if path else "",
        "every_changes": every,
        "enabled": bool(path) and every > 0,
        "counter": db.get_backup_counter(),
        "running": db._ext_backup_running,
        "reachable": None,
        "backups": 0,
        "last": None,
        "last_size_bytes": 0,
    }
    if path:
        p = path
        out["reachable"] = p.is_dir()
        if out["reachable"]:
            try:
                stem = Path(db._db_path).stem
                files = sorted(
                    p.glob(f"{stem}_*.db"), key=lambda f: f.stat().st_mtime, reverse=True
                )
                out["backups"] = len(files)
                if files:
                    out["last"] = files[0].name
                    out["last_ts"] = files[0].stat().st_mtime
                    out["last_size_bytes"] = files[0].stat().st_size
            except OSError as e:
                out["reachable"] = False
                out["error"] = str(e)
    return jsonify(out)


#: What a reset keeps (section, keys — None = the whole section).
#:
#: * WHERE things live: directories, the database and vector store files.
#:   Resetting those would make the next start open a different (empty)
#:   database; "transcripts are kept" must hold after a restart, too.
#: * WHO may reach the app: server address, login, allowed roots. The page
#:   can't set them back — they are config.yaml-only — so a reset would
#:   switch a login off for good.
#: * What the user WROTE: LLM providers and their task assignment, the
#:   summary templates, model shortcuts and vocabularies. Their defaults are
#:   a starting point, not a replacement for what someone made.
#: Credentials anywhere else (tokens, API keys) are kept as well, see below.
_RESET_KEEPS = (
    ("directories", None),
    ("server", None),
    ("auth", None),
    ("security", None),
    # config.yaml-only like the login: a reset would set them back for good
    ("logging", None),
    ("llama", None),
    ("embedding", ("local_extra_args",)),
    ("mcp", ("token",)),
    ("database", ("path", "backup_path")),
    ("rag", ("path",)),
    ("llm", ("providers", "tasks", "presets", "models", "endpoint", "api_key", "model")),
    ("transcription", ("vocab_presets",)),
)


def _reset_keeps() -> list[tuple[str, tuple | None]]:
    """_RESET_KEEPS plus every key the page can't set (file-only) or may set
    only from this machine (local-only): a reset that dropped them pointed
    Images & sounds and the audio analysis at new empty databases, and threw
    away values the page has no way to bring back."""
    whole: set[str] = set()
    keys: dict[str, set] = {}
    for section, ks in _RESET_KEEPS:
        if ks is None:
            whole.add(section)
        else:
            keys.setdefault(section, set()).update(ks)
    for entry in (*_FILE_ONLY, *_LOCAL_ONLY):
        section, _, key = entry.partition(".")
        if not key:  # "security." — the whole section
            whole.add(section)
        else:
            keys.setdefault(section, set()).add(key.split(".", 1)[0])
    return [(s, None) for s in sorted(whole)] + [
        (s, tuple(sorted(k))) for s, k in sorted(keys.items()) if s not in whole
    ]


def _keep_credentials(current: Any, target: dict) -> None:
    """Carry every set secret (token, API key, password) over into target."""
    if not isinstance(current, dict):
        return
    for key, value in current.items():
        if isinstance(value, dict):
            sub = target.get(key)
            fresh = not isinstance(sub, dict)
            if fresh:
                sub = {}
            _keep_credentials(value, sub)
            if fresh and sub and key not in target:
                target[key] = sub
        elif is_secret(key) and isinstance(value, str) and value.strip():
            target[key] = value


@settings_bp.route("/api/settings/reset", methods=["POST"])
def reset_settings():
    """Reset the settings to their defaults — see _RESET_KEEPS for what stays.

    Transcripts and database are preserved.
    """
    try:
        import copy

        defaults = config._load_defaults()
        # Read, merge and swap under one lock hold: a settings save between
        # the copy and the swap was silently thrown away
        with config._lock:
            current = copy.deepcopy(config._data)
            for section, keys in _reset_keeps():
                block = current.get(section)
                if not isinstance(block, dict):
                    continue
                keep = block if keys is None else {k: block[k] for k in keys if k in block}
                if keep:
                    defaults.setdefault(section, {}).update(copy.deepcopy(keep))
            _keep_credentials(current, defaults)
            config._data = defaults
        config._resolve_directories()
        # #192: consistent with update_settings/set_directory — after
        # re-resolve also create directories.
        config._ensure_directories()
        saved = config.save()
        return jsonify({"ok": True, "saved_to": str(saved)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Helpers ───────────────────────────────────────────────────────────────


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten dict for config.set().

    Stops at "opaque" dicts (presets, models) that should be stored as a
    whole — their keys can contain dots/colons that would otherwise be
    interpreted as path separators.
    """
    # Keys whose value dicts should NOT be further resolved: named lists
    # that are saved as a whole — a deleted entry must go, and a name may
    # contain a dot ("Dr. Grace", "qwen3.5")
    OPAQUE_KEYS = {"presets", "models", "vocab_presets", "speaker_names"}

    items = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and key not in OPAQUE_KEYS:
            items.extend(_flatten_dict(value, full_key))
        else:
            items.append((full_key, value))
    return items


def _is_writable(path) -> bool:
    import os

    return os.access(str(path), os.W_OK)


# ── DB Export & Backup ────────────────────────────────────────────────────


@settings_bp.route("/api/db/export")
def export_db():
    """Download the database — a consistent snapshot, not the live file.

    The database runs in WAL mode: the latest commits can still sit in the
    -wal file, and the main file changes while it is read. Sending it as it
    lies gave a copy that could miss the newest transcripts or mix pages
    from before and after a write — a backup that only turns out broken
    when someone restores it. So the SQLite backup API writes a snapshot
    first (as ``Database.backup`` does), and the download sends that.
    Nothing of it stays behind: its name goes as soon as the file is open
    for sending, the response's close removes it where that isn't possible
    (Windows), and what a server that died meanwhile left goes on the next
    call.
    """
    import os
    import shutil
    import sqlite3
    import tempfile

    from flask import send_file

    from whisper_tm.core.helpers import format_size

    if not db:
        return jsonify({"error": "No database"}), 500
    db_file = Path(db._db_path)
    if not db_file.exists():
        return jsonify({"error": "Database file not found"}), 404

    # Next to the database first: the same file system, where a copy of the
    # database had room once already. The system temp folder only as the
    # second choice — on many systems it is a tmpfs, the snapshot would
    # take RAM.
    folders = [db_file.parent]
    system_tmp = Path(tempfile.gettempdir())
    try:
        if system_tmp.resolve() != db_file.parent.resolve():
            folders.append(system_tmp)
    except OSError:
        pass
    prefix = f".{db_file.name}.export-"

    # Leftovers of an export whose server died before the close callback
    # below could run (a crash, a kill during the download). Untouched for
    # ten minutes means no request writes it anymore: a snapshot in progress
    # changes all the time, and one still being downloaded stays readable
    # after the unlink — the open file keeps it alive.
    stale_before = time.time() - 600
    for folder in folders:
        try:
            with os.scandir(folder) as entries:
                leftovers = [e for e in entries if e.name.startswith(prefix)]
        except OSError as e:
            logger.debug("Export leftovers in %s not checked: %s", folder, e)
            continue
        for entry in leftovers:
            try:
                if entry.stat(follow_symlinks=False).st_mtime < stale_before:
                    os.unlink(entry.path)
            except OSError as e:
                logger.debug("Export leftover %s not removed: %s", entry.path, e)

    try:
        with db.connection() as conn:
            size = (
                conn.execute("PRAGMA page_count").fetchone()[0]
                * conn.execute("PRAGMA page_size").fetchone()[0]
            )
    except sqlite3.Error as e:
        return jsonify({"error": f"Database not readable: {e}"}), 500

    # "Space" means the snapshot takes at most half of what is free. Filling
    # the database's disk would fail the live database's next write, filling
    # the tmpfs takes the RAM the models need.
    target, free = None, {}
    for folder in folders:
        try:
            free[folder] = shutil.disk_usage(folder).free
        except OSError:
            continue
        if free[folder] >= 2 * size:
            target = folder
            break
    if target is None:
        where = ", ".join(f"{f}: {format_size(n)} free" for f, n in free.items())
        return jsonify(
            {
                "error": f"Not enough free space for the export snapshot "
                f"({format_size(size)}, it may take at most half of the "
                f"free space) — {where or 'no folder usable'}",
            }
        ), 507

    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=target)
    os.close(fd)  # created with mode 0600: it holds every transcript
    snapshot = Path(name)

    def _remove() -> None:
        for p in (snapshot, Path(f"{snapshot}-journal")):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Export snapshot not removed (%s): %s", p, e)

    keep = False
    try:
        dst = sqlite3.connect(snapshot)
        try:
            with db.connection() as conn:
                conn.backup(dst)
        finally:
            dst.close()
        # conditional=False: no range requests. Every request makes a new
        # snapshot, so resuming a download would stitch two different
        # databases together.
        resp = send_file(
            snapshot,
            as_attachment=True,
            download_name="whisper-tm-backup.db",
            mimetype="application/x-sqlite3",
            conditional=False,
        )
        # send_file marks the response direct_passthrough: the server then
        # gets the bare file wrapper and never calls the response's close()
        # — a close callback would never run. Passed through the response,
        # the file is closed by that close(), which then runs the callback.
        resp.direct_passthrough = False
        resp.call_on_close(_remove)
        keep = True  # the close callback deletes it after the download
        # send_file has opened the file, and on POSIX an open file keeps its
        # data without a name: removed now, nothing is left even when the
        # server dies during the download. (Windows refuses; there the
        # callback does it.)
        try:
            snapshot.unlink()
        except OSError:
            pass
        return resp
    except Exception as e:
        logger.warning("DB export failed: %s", e)
        return jsonify({"error": f"Export failed: {e}"}), 500
    finally:
        if not keep:
            _remove()


@settings_bp.route("/api/db/delete", methods=["POST"])
def delete_db():
    """Delete database and reinitialize."""
    if not db:
        return jsonify({"error": "No database"}), 500
    db_file = config.db_path
    from whisper_tm.core import queue as _tx_queue
    from whisper_tm.core.llm_queue import llm_queue

    if _tx_queue.busy() or llm_queue.busy():
        # A running job writes into the database that is about to vanish
        return jsonify(
            {
                "error": "A job is running — wait for it or stop it in the queue, "
                "then delete the database."
            }
        ), 409
    # A copy first, like Compact and the cleanup — in "Database only" mode
    # the database is the only copy of every transcript
    target = db.backup_target()
    saved = db.backup(dest_dir=target)
    if saved is None:
        return jsonify(
            {
                "error": f"No backup could be made in {target} — nothing was deleted. "
                f"Check database.backup_path and the free space."
            }
        ), 500
    try:
        with llm_queue.held():
            # Close connection and reset initialized flag
            db.reset()
            # Delete all DB files
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(db_file) + suffix)
                if p.exists():
                    p.unlink()
            # Reinitialize fresh (recreates schema)
            db.initialize()
        return jsonify({"ok": True, "backup": str(saved)})
    except Exception as e:
        return jsonify({"error": str(e), "backup": str(saved)}), 500


# ── Log Explorer ──────────────────────────────────────────────────────────


@settings_bp.route("/api/logs")
def list_logs():
    """The newest 200 job logs, optionally filtered by name (``q``).

    The job log folder can hold well over a hundred thousand files, possibly
    on a network share. Asking the file system for type, size and date of
    EVERY one of them before keeping 200 didn't answer within 30 s. Job log
    names start with their timestamp
    (``jobs/__init__.py::_persist_log``: "20240301_142501_<file>.log"), so
    name order is time order: one directory read, a sort over the names,
    and a stat only for the files that are returned.
    """
    import os
    import stat

    limit = 200
    logs_dir = config.dirs["logs"]
    q = request.args.get("q", "").strip().lower()
    try:
        with os.scandir(logs_dir) as entries:
            names = [
                e.name
                for e in entries
                if e.name.endswith(".log") and (not q or q in e.name.lower())
            ]
    except FileNotFoundError:
        return jsonify({"logs": []})
    except OSError as e:
        return jsonify({"logs": [], "error": f"Log folder not readable: {e}"}), 500
    names.sort(reverse=True)
    logs = []
    for name in names:
        try:
            st = os.lstat(os.path.join(logs_dir, name))
        except OSError:
            continue  # deleted in the meantime
        if not stat.S_ISREG(st.st_mode):
            continue  # a folder or a link whose name ends in .log
        logs.append({"filename": name, "size": st.st_size, "modified": st.st_mtime})
        if len(logs) == limit:
            break
    return jsonify({"logs": logs})


def _log_file(filename: str) -> Path | None:
    """The job log ``filename`` in the logs folder — or None.

    Only ``*.log`` files, only regular files, and no link: the logs folder
    is a setting, and with it set to "/" this endpoint used to hand out (and
    delete) any file the server could reach — the config with its secrets,
    the database. A link inside the folder is not followed out of it.
    """
    import stat

    from whisper_tm.core.security import safe_path_or_none

    logs_dir = config.dirs["logs"]
    if not str(filename or "").endswith(".log"):
        return None
    fp = safe_path_or_none(logs_dir, filename)
    if fp is None:
        return None
    raw = Path(logs_dir) / filename
    try:
        st = raw.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return fp


@settings_bp.route("/api/logs/view/<path:filename>")
def view_log(filename):
    """Read log file content (only last block, no full read)."""
    fp = _log_file(filename)
    if fp is None:
        return jsonify({"error": "Log not found"}), 404
    q = request.args.get("q", "").strip().lower()
    # #168: do NOT pull entire file into RAM via read_text()
    # (OOM with GB-sized logs). Stream via file iterator and keep only
    # last 5000 (if filtered) lines via deque; total_lines counted without full read.
    tail: collections.deque = collections.deque(maxlen=5000)
    total_lines = 0
    with fp.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if q and q not in line.lower():
                continue
            total_lines += 1
            tail.append(line)
    return jsonify(
        {
            "filename": filename,
            "lines": list(tail),
            "total_lines": total_lines,
        }
    )


@settings_bp.route("/api/logs/delete/<path:filename>", methods=["POST"])
def delete_log(filename):
    """Delete log file."""
    fp = _log_file(filename)
    if fp is None:
        return jsonify({"error": "Log not found"}), 404
    fp.unlink()
    return jsonify({"ok": True})


@settings_bp.route("/api/logs/delete_all", methods=["POST"])
def delete_all_logs():
    """Delete all log files."""
    logs_dir = config.dirs["logs"]
    if not logs_dir.exists():
        return jsonify({"ok": True, "deleted": 0})
    deleted = 0
    failed = 0
    for f in logs_dir.iterdir():
        if f.suffix == ".log" and f.is_file() and not f.is_symlink():
            try:
                f.unlink()
                deleted += 1
            except OSError as e:
                failed += 1
                logger.debug("Log not deleted (%s): %s", f, e)
    return jsonify({"ok": not failed, "deleted": deleted, "failed": failed})


# ── DB Explorer ──────────────────────────────────────────────────────────


@settings_bp.route("/api/db/explore")
def db_explore():
    """Mini DB explorer: read table contents (read-only).

    Query parameters:
        table: transcripts | media_files | segments | skiplist | job_history
        limit: Max rows (default 50, max 500)
        offset: Offset for paging
        search: Search term (LIKE on all text columns)
        columns: Comma-separated column list (default: all)
        order_by: Column + direction (e.g. "rel_path ASC")
    """
    if not db:
        return jsonify({"error": "No database"}), 501

    table = request.args.get("table", "transcripts").strip()
    limit = min(request.args.get("limit", type=int, default=50) or 50, 500)
    offset = max(request.args.get("offset", type=int, default=0) or 0, 0)
    search = request.args.get("search", "").strip()
    order_by = request.args.get("order_by", "").strip()
    columns = request.args.get("columns", "").strip()

    # Only allowed tables
    allowed = {"transcripts", "media_files", "segments", "skiplist", "job_history", "user_settings"}
    if table not in allowed:
        return jsonify({"error": f"Table '{table}' not allowed"}), 400

    try:
        with db.connection() as conn:
            # Column info
            cols_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_names = [c["name"] for c in cols_info]
            col_types = {c["name"]: c["type"] for c in cols_info}

            # Column filter
            if columns:
                selected_cols = [c.strip() for c in columns.split(",") if c.strip() in col_names]
            else:
                # Hide certain columns for large tables
                skip = {"json_data", "step_timings", "warnings_json"}
                selected_cols = [c for c in col_names if c not in skip]

            if not selected_cols:
                selected_cols = col_names[:10]

            select = ", ".join(selected_cols)

            # Count
            count_sql = f"SELECT COUNT(*) as cnt FROM {table}"
            count_params = []

            # Search
            where = ""
            params = []
            if search:
                text_cols = [
                    c for c in selected_cols if col_types.get(c, "").upper() in ("TEXT", "")
                ]
                if text_cols:
                    clauses = [f"{c} LIKE ?" for c in text_cols]
                    where = " WHERE (" + " OR ".join(clauses) + ")"
                    params = [f"%{search}%"] * len(text_cols)
                    count_sql += where
                    count_params = list(params)

            total = conn.execute(count_sql, count_params).fetchone()["cnt"]

            # Order
            order = ""
            if order_by:
                # Sanitize: only column name + ASC/DESC
                parts = order_by.split()
                if parts[0] in col_names:
                    direction = "DESC" if len(parts) > 1 and parts[1].upper() == "DESC" else "ASC"
                    order = f" ORDER BY {parts[0]} {direction}"
            if not order:
                order = f" ORDER BY {col_names[0]} ASC"

            # Query
            sql = f"SELECT {select} FROM {table}{where}{order} LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(sql, params).fetchall()

            return jsonify(
                {
                    "table": table,
                    "columns": selected_cols,
                    "column_types": {c: col_types.get(c, "?") for c in selected_cols},
                    "rows": [dict(r) for r in rows],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "tables": sorted(allowed),
                }
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Custom Done Sound ─────────────────────────────────────────────────────

_SOUND_TYPES = (".mp3", ".wav", ".ogg", ".m4a")


def _sounds_dir() -> Path:
    """Where uploaded done sounds live: the data folder.

    They used to go into the work folder — which every server start
    empties (config.clean_workdir), so an uploaded sound was gone after the
    next restart while the database still named it, and the done sound
    fell silent.
    """
    return config.data_dir / "sounds"


@settings_bp.route("/api/sounds")
def list_sounds():
    """List available done sounds."""
    sounds = [{"name": "done.mp3", "builtin": True}]
    custom_dir = _sounds_dir()
    if custom_dir.is_dir():
        for f in sorted(custom_dir.iterdir()):
            if f.suffix.lower() in _SOUND_TYPES:
                sounds.append({"name": f"custom/{f.name}", "builtin": False})
    try:
        current = db.get_setting("done_sound", default="done.mp3") if db else "done.mp3"
    except Exception as _e:  # noqa
        current = "done.mp3"
    # A chosen sound whose file is gone plays nothing at all — the built-in
    # one is the better answer
    if current not in {x["name"] for x in sounds}:
        current = "done.mp3"
    return jsonify({"sounds": sounds, "current": current})


@settings_bp.route("/api/sounds/upload", methods=["POST"])
def upload_sound():
    """Upload custom done sound."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    from werkzeug.utils import secure_filename

    name = secure_filename(f.filename) or "custom.mp3"
    if not name.lower().endswith(_SOUND_TYPES):
        return jsonify({"error": "Only mp3/wav/ogg/m4a"}), 400
    custom_dir = _sounds_dir()
    custom_dir.mkdir(parents=True, exist_ok=True)
    dest = custom_dir / name
    f.save(str(dest))
    return jsonify({"ok": True, "name": f"custom/{name}"})


@settings_bp.route("/api/sounds/select", methods=["POST"])
def select_sound():
    """Select done sound."""
    data = request.json or {}
    name = data.get("name", "done.mp3")
    if db:
        try:
            db.set_setting("done_sound", name)
        except Exception as e:
            # #172: don't swallow errors — otherwise UI reports success
            # even though selection wasn't saved.
            logger.error("done_sound save failed: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "current": name})


@settings_bp.route("/api/sounds/play/<path:filename>")
def play_sound(filename):
    """Play/stream sound file."""
    if filename == "done.mp3":
        static = Path(__file__).parent.parent / "static"
        fp = static / "done.mp3"
    else:
        from whisper_tm.core.security import safe_path

        # strip "custom/" prefix
        name = filename.replace("custom/", "", 1)
        fp = safe_path(_sounds_dir(), name)
    if not fp or not fp.exists():
        return jsonify({"error": "Sound not found"}), 404
    from flask import send_file

    return send_file(str(fp))

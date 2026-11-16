"""
UI preferences kept on the server, so they are the same in every browser:
the rail's pins and fold, Home's open and closed groups, the last tab of
each window, export options and presets, and the recently opened
recordings.

One small JSON file in the data folder (``ui_prefs.json``), written in one
step. Every key is checked; unknown keys are dropped, a value that does not
fit is refused with the key named.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core.helpers import atomic_write_json
from whisper_tm.core.security import root_allowed

logger = logging.getLogger(__name__)

ui_prefs_bp = Blueprint("ui_prefs", __name__)

#: Most places or window tabs the rail takes as pins
MAX_PINS = 6
#: Recently opened recordings that are kept
MAX_RECENT = 20
#: A request body and the stored file stay below this
MAX_BYTES = 64 * 1024
#: Free-form option blocks (export options, one preset's options)
MAX_BLOCK_BYTES = 8 * 1024
MAX_PRESETS = 20

_lock = threading.Lock()
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")
# A pin is a place ("voices") or a window's tab ("voices:map")
_PIN = re.compile(r"^[a-z][a-z0-9_-]{0,29}(?::[a-z][a-z0-9_-]{0,29})?$")


class PrefsError(ValueError):
    """A value that does not fit its key."""


def _path() -> Path:
    return Path(config.data_dir) / "ui_prefs.json"


def _block(value, key: str):
    """A free-form JSON object, limited in size."""
    if not isinstance(value, dict):
        raise PrefsError(f"{key}: an object is needed")
    if len(json.dumps(value, ensure_ascii=False)) > MAX_BLOCK_BYTES:
        raise PrefsError(f"{key}: larger than {MAX_BLOCK_BYTES} bytes")
    return value


def _pins(value):
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        raise PrefsError("pins: a list of names is needed")
    seen: list[str] = []
    for p in value:
        if not _PIN.match(p):
            raise PrefsError(f"pins: {p[:40]!r} is not a place or a window tab")
        if p not in seen:
            seen.append(p)
    if len(seen) > MAX_PINS:
        raise PrefsError(f"pins: at most {MAX_PINS}")
    return seen


def _flags(value, key: str, limit: int = 40):
    """{id: true/false} — e.g. Home's groups, open or closed."""
    if not isinstance(value, dict) or len(value) > limit:
        raise PrefsError(f"{key}: an object with at most {limit} entries is needed")
    out = {}
    for k, v in value.items():
        if not isinstance(k, str) or not _ID.match(k) or not isinstance(v, bool):
            raise PrefsError(f"{key}: {str(k)[:40]!r} needs a short name and true or false")
        out[k] = v
    return out


def _tabs(value):
    """{window: tab} — the tab each window was left on."""
    if not isinstance(value, dict) or len(value) > 20:
        raise PrefsError("last_tab: an object with at most 20 entries is needed")
    out = {}
    for k, v in value.items():
        if not (isinstance(k, str) and _ID.match(k) and isinstance(v, str) and _ID.match(v)):
            raise PrefsError(f"last_tab: {str(k)[:40]!r} needs a short window and tab name")
        out[k] = v
    return out


def _presets(value):
    if not isinstance(value, list) or len(value) > MAX_PRESETS:
        raise PrefsError(f"export_presets: a list of at most {MAX_PRESETS} is needed")
    out = []
    for p in value:
        if (
            not isinstance(p, dict)
            or not isinstance(p.get("name"), str)
            or not p["name"].strip()
            or len(p["name"]) > 60
        ):
            raise PrefsError("export_presets: every preset needs a name (60 characters at most)")
        out.append(
            {"name": p["name"].strip(), "options": _block(p.get("options", {}), "export_presets")}
        )
    return out


def _recent(value):
    if not isinstance(value, list):
        raise PrefsError("recent: a list is needed")
    out = []
    for r in value[:MAX_RECENT]:
        if not isinstance(r, dict) or not _good_path(r.get("path")):
            raise PrefsError("recent: every entry needs an absolute path")
        at = r.get("at", 0)
        # A finite time only: an infinite one made the whole answer unreadable
        ok = isinstance(at, (int, float)) and not isinstance(at, bool) and math.isfinite(at)
        out.append({"path": r["path"], "at": float(at) if ok else 0.0})
    return out


def _bool(value, key: str) -> bool:
    if not isinstance(value, bool):
        raise PrefsError(f"{key}: true or false")
    return value


def _good_path(p) -> bool:
    return isinstance(p, str) and 1 < len(p) <= 4096 and p.startswith("/") and "\x00" not in p


#: key → checker. Anything else in a request is dropped.
_KEYS = {
    "pins": _pins,
    "rail_folded": lambda v: _bool(v, "rail_folded"),
    "home_groups": lambda v: _flags(v, "home_groups"),
    "last_tab": _tabs,
    "export_options": lambda v: _block(v, "export_options"),
    "export_presets": _presets,
    "recent": _recent,
}


def load() -> dict:
    """The stored preferences ({} when none are stored or the file is
    unreadable — a broken file never stops the page)."""
    p = _path()
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("UI preferences unreadable (%s) — starting from defaults", e)
        return {}
    if not isinstance(data, dict):
        return {}
    clean = {}
    for k, check in _KEYS.items():
        if k in data:
            try:
                clean[k] = check(data[k])
            except (PrefsError, TypeError, ValueError):
                pass  # one bad stored key costs only that key
    return clean


def _save(data: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, data, indent=None)


def update(patch: dict) -> dict:
    """Merge the known keys of ``patch`` into the stored preferences."""
    checked = {}
    for k, v in patch.items():
        if k in _KEYS:
            checked[k] = _KEYS[k](v)
    with _lock:
        data = load()
        data.update(checked)
        if len(json.dumps(data, ensure_ascii=False)) > MAX_BYTES:
            raise PrefsError(f"the preferences would be larger than {MAX_BYTES} bytes")
        _save(data)
        return data


def _existing(recent: list) -> list:
    """Recently opened recordings that are still there (and inside the
    library: the list never answers for files elsewhere)."""
    out = []
    for r in recent:
        try:
            if root_allowed(r["path"]) and os.path.isfile(r["path"]):
                out.append(r)
        except OSError:
            continue
    return out


@ui_prefs_bp.route("/api/ui/prefs", methods=["GET"])
def get_prefs():
    """The UI preferences: pins, rail_folded, home_groups, last_tab,
    export_options, export_presets and recent (recordings that no longer
    exist are left out)."""
    data = load()
    if "recent" in data:
        data["recent"] = _existing(data["recent"])
    return jsonify(data)


@ui_prefs_bp.route("/api/ui/prefs", methods=["PUT", "PATCH"])
def put_prefs():
    """Change some UI preferences: the keys sent replace their stored
    values, the others stay. Unknown keys are dropped; a value that does
    not fit is refused (400) with the key named."""
    if (request.content_length or 0) > MAX_BYTES:
        return jsonify({"error": f"larger than {MAX_BYTES} bytes"}), 413
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "a JSON object is needed"}), 400
    try:
        data = update(body)
    except PrefsError as e:
        return jsonify({"error": str(e)}), 400
    except OSError as e:
        logger.error("UI preferences not saved: %s", e)
        return jsonify({"error": "could not be saved"}), 500
    if "recent" in data:
        data["recent"] = _existing(data["recent"])
    return jsonify(data)


@ui_prefs_bp.route("/api/ui/recent", methods=["POST"])
def add_recent():
    """Note a recording as just opened. Body: {"path": absolute path}.
    It moves to the front of the recently opened list (at most 20)."""
    body = request.get_json(silent=True) or {}
    path = body.get("path") if isinstance(body, dict) else None
    if not _good_path(path):
        return jsonify({"error": "an absolute path is needed"}), 400
    path = os.path.normpath(path)
    if not root_allowed(path):
        return jsonify({"error": "not inside the library"}), 403
    try:
        with _lock:
            data = load()
            recent = [r for r in data.get("recent", []) if r["path"] != path]
            recent.insert(0, {"path": path, "at": time.time()})
            data["recent"] = recent[:MAX_RECENT]
            _save(data)
    except OSError as e:
        logger.error("Recently opened not saved: %s", e)
        return jsonify({"error": "could not be saved"}), 500
    return jsonify({"recent": _existing(data["recent"])})

"""
A recording's clip cuts, kept on the server: the parts chosen in its
Clips tab, how the clip is to be made, and what was made from it.

They survive a reload and another browser, and Export's "Only the clip
cuts" reads them. One row per recording (by its absolute path, the
identity — docs/PATHS.md) in the main database, in a table of its own
that is made on first use.
"""

from __future__ import annotations

import json
import math
import threading
import time

MAX_CUTS = 50
MAX_MADE = 20
FORMATS = ("orig", "vert", "square")
PAUSE_MIN = (0.5, 0.8, 1.5, 3.0)
SETTINGS_DEFAULTS = {
    "format": "orig",
    "crop": 0.5,
    "pauses": False,
    "pause_min": 0.8,
    "subs": False,
    "fade": True,
    "stitch": "one",
}

_lock = threading.Lock()


class CutsError(ValueError):
    """Cuts or settings that do not fit."""


def _db():
    from whisper_tm.extensions import db

    if db is None:
        raise CutsError("No database configured")
    return db


def _table(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS clip_cuts ("
        " abs_path TEXT PRIMARY KEY,"
        " data_json TEXT NOT NULL DEFAULT '{}',"
        " updated_at REAL DEFAULT (unixepoch()))"
    )


def clean_cuts(raw, length: float = 0.0) -> list[dict]:
    """[{start, end, ai?, title?}] — each checked, in the order given."""
    if not isinstance(raw, list):
        raise CutsError("cuts: a list is needed")
    if len(raw) > MAX_CUTS:
        raise CutsError(f"at most {MAX_CUTS} cuts")
    out = []
    for c in raw:
        if not isinstance(c, dict):
            raise CutsError("every cut needs a start and an end")
        try:
            a, b = float(c.get("start")), float(c.get("end"))
        except (TypeError, ValueError):
            raise CutsError("a cut's start and end are seconds") from None
        # Not a number, and not infinite either: 1e999 was stored and came
        # back as "Infinity", which is no JSON — the cuts then answered 500
        if not (math.isfinite(a) and math.isfinite(b)):
            raise CutsError("a cut's start and end are seconds")
        a, b = max(0.0, min(a, b)), max(a, b)
        if length > 0:
            b = min(b, length)
        if b - a < 0.2:
            raise CutsError("a cut is at least 0.2 seconds long")
        cut = {"start": round(a, 3), "end": round(b, 3)}
        if c.get("ai"):
            cut["ai"] = True
        if c.get("title"):
            cut["title"] = str(c["title"])[:120]
        out.append(cut)
    return out


def clean_settings(raw) -> dict:
    s = dict(SETTINGS_DEFAULTS)
    s.update({k: v for k, v in (raw or {}).items() if k in SETTINGS_DEFAULTS})
    if s["format"] not in FORMATS:
        raise CutsError("format: orig, vert or square")
    if s["stitch"] not in ("one", "each"):
        raise CutsError("stitch: one or each")
    try:
        s["crop"] = max(0.0, min(1.0, float(s["crop"])))
        s["pause_min"] = float(s["pause_min"])
    except (TypeError, ValueError):
        raise CutsError("crop and pause_min are numbers") from None
    if s["pause_min"] not in PAUSE_MIN:
        raise CutsError(f"pause_min: one of {', '.join(map(str, PAUSE_MIN))}")
    for k in ("pauses", "subs", "fade"):
        s[k] = bool(s[k])
    return s


def load(abs_path: str) -> dict:
    """{cuts, settings, made} of a recording (empty ones if none)."""
    db = _db()
    with db.connection() as conn:
        _table(conn)
        r = conn.execute(
            "SELECT data_json FROM clip_cuts WHERE abs_path = ?", (abs_path,)
        ).fetchone()
    try:
        d = json.loads(r["data_json"]) if r else {}
    except (TypeError, ValueError):
        d = {}
    try:
        cuts = clean_cuts(d.get("cuts") or [])
    except CutsError:
        cuts = []
    try:
        settings = clean_settings(d.get("settings") or {})
    except CutsError:
        settings = dict(SETTINGS_DEFAULTS)
    made = [m for m in (d.get("made") or []) if isinstance(m, dict)][:MAX_MADE]
    return {"cuts": cuts, "settings": settings, "made": made}


def get(abs_path: str) -> list[dict]:
    """The cuts alone (Export's "Only the clip cuts")."""
    try:
        return load(abs_path)["cuts"]
    except CutsError:
        return []


def _write(conn, abs_path: str, d: dict) -> None:
    conn.execute(
        "INSERT INTO clip_cuts (abs_path, data_json, updated_at) VALUES (?, ?, ?)"
        " ON CONFLICT(abs_path) DO UPDATE SET data_json = excluded.data_json,"
        " updated_at = excluded.updated_at",
        (abs_path, json.dumps(d, ensure_ascii=False), time.time()),
    )


def save(abs_path: str, cuts=None, settings=None, length: float = 0.0) -> dict:
    """Replace the cuts and/or the settings (what is not given stays)."""
    with _lock:
        d = load(abs_path)
        if cuts is not None:
            d["cuts"] = clean_cuts(cuts, length)
        if settings is not None:
            d["settings"] = clean_settings(settings)
        with _db().connection() as conn:
            _table(conn)
            _write(conn, abs_path, d)
        return d


def add(abs_path: str, cuts: list[dict], length: float = 0.0) -> dict:
    """Append cuts (Best moments) to what is there."""
    with _lock:
        d = load(abs_path)
        d["cuts"] = clean_cuts((d["cuts"] + list(cuts))[:MAX_CUTS], length)
        with _db().connection() as conn:
            _table(conn)
            _write(conn, abs_path, d)
        return d


def note_made(abs_path: str, entry: dict) -> None:
    """Remember a clip made from this recording (newest first)."""
    with _lock:
        d = load(abs_path)
        d["made"] = ([entry] + d["made"])[:MAX_MADE]
        with _db().connection() as conn:
            _table(conn)
            _write(conn, abs_path, d)

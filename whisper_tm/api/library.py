"""
The library as a folder tree: every folder under the library root with
its counts — files, transcribed, open, failed, waiting in the queue — each
including its subfolders.

One pass over ``media_files`` (and the queue's open entries) builds the
whole tree; it is kept for a short while, so expanding folders and
switching between them costs nothing. The endpoint hands out one level at a
time (a folder and its direct subfolders): a big library's tree is
thousands of folders, and nobody opens them all at once.

Counting goes over ``abs_path``, like every other count in the app.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import PurePosixPath

from flask import Blueprint, jsonify, request

from whisper_tm.core import paths

logger = logging.getLogger(__name__)

library_bp = Blueprint("library", __name__)

CACHE_TTL = 30.0
_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}

_KEYS = ("files", "done", "open", "failed", "queued")


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _speak_cache.clear()


def _blank() -> dict:
    return {k: 0 for k in _KEYS}


def build(database, root: str) -> dict:
    """{folder abs path: counts} for the root and every folder below it
    (each folder's counts include its subfolders), plus
    {folder: sorted child folder names}."""
    base = paths.canon(root).rstrip("/")
    lo, hi = base + "/", base + "0"  # '0' follows '/'
    counts: dict[str, dict] = {base: _blank()}
    children: dict[str, set] = {base: set()}

    def add(folder: str, key: str) -> None:
        # The folder and every folder above it, up to the root
        while True:
            c = counts.get(folder)
            if c is None:
                c = counts[folder] = _blank()
                children.setdefault(folder, set())
                parent = folder.rsplit("/", 1)[0]
                children.setdefault(parent, set()).add(folder.rsplit("/", 1)[1])
            c[key] += 1
            if folder == base or len(folder) <= len(base):
                return
            folder = folder.rsplit("/", 1)[0]

    with database.connection() as conn:
        rows = conn.execute(
            "SELECT abs_path, status FROM media_files WHERE abs_path >= ? AND abs_path < ?",
            (lo, hi),
        ).fetchall()
        # The queue: waiting entries, and each file's latest failed/cancelled outcome
        queue = conn.execute(
            "SELECT base_dir, filename, status FROM queue_items"
            " WHERE status IN ('pending', 'error', 'cancelled') ORDER BY id"
        ).fetchall()
    for r in rows:
        folder = r["abs_path"].rsplit("/", 1)[0]
        add(folder, "files")
        st = r["status"] or "pending"
        if st == "done":
            add(folder, "done")
        elif st not in ("skipped", "orphan"):
            add(folder, "open")
    from whisper_tm.core.queue import _item_abs_path

    latest: dict[str, str] = {}
    waiting: set[str] = set()
    for q in queue:
        a = _item_abs_path({"base_dir": q["base_dir"], "filename": q["filename"]})
        if not a.startswith(lo):
            continue
        if q["status"] == "pending":
            waiting.add(a)
        else:
            latest[a] = q["status"]
    for a in waiting:
        add(a.rsplit("/", 1)[0], "queued")
    for a, st in latest.items():
        if st == "error" and a not in waiting:
            add(a.rsplit("/", 1)[0], "failed")
    return {
        "root": base,
        "counts": counts,
        "children": {k: sorted(v, key=str.lower) for k, v in children.items()},
    }


def tree(database, root: str) -> dict:
    """The cached tree of a root."""
    key = paths.canon(root)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    t = build(database, root)
    with _lock:
        _cache[key] = (time.monotonic(), t)
    return t


def _watched(folder: str, watched: list[str]) -> str:
    """'self' when the folder is watched, 'parent' when a folder above it
    is, '' otherwise."""
    for w in watched:
        w = w.rstrip("/")
        if folder == w:
            return "self"
        if folder.startswith(w + "/"):
            return "parent"
    return ""


def _watched_folders() -> list[str]:
    """The folders continuous scanning watches: none while it is off, the
    chosen ones, or — none chosen — the whole library (the allowed trees)."""
    try:
        from whisper_tm.core import autoscan

        if not autoscan.setting("enabled"):
            return []
        return [paths.canon(str(f)) for f in autoscan.watched_folders()]
    except Exception:
        return []


@library_bp.route("/api/library/tree")
def library_tree():
    """One folder of the library and its direct subfolders, each with its
    counts (files, done, open, failed, queued — subfolders included) and
    whether it is watched by continuous scanning.

    Query: ``path`` — a folder below the library root (relative; empty for
    the root). The root is the first allowed folder tree.
    """
    from whisper_tm.core.security import allowed_roots
    from whisper_tm.extensions import db

    if db is None:
        return jsonify({"error": "No database configured"}), 501
    roots = allowed_roots()
    if not roots:
        return jsonify({"error": "No library root (security.allowed_roots)"}), 403
    root = paths.canon(str(roots[0])).rstrip("/")
    rel = (request.args.get("path") or "").strip().strip("/")
    if ".." in rel.split("/"):
        return jsonify({"error": "Path outside the library"}), 400
    t = tree(db, root)
    here = root + ("/" + rel if rel else "")
    if here not in t["counts"]:
        return jsonify({"error": "No such folder in the library", "path": rel}), 404
    watched = _watched_folders()

    def node(abs_folder: str) -> dict:
        r = abs_folder[len(root) + 1 :] if abs_folder != root else ""
        return {
            "name": PurePosixPath(abs_folder).name if r else PurePosixPath(root).name,
            "path": r,
            "counts": t["counts"][abs_folder],
            "has_children": bool(t["children"].get(abs_folder)),
            "watched": _watched(abs_folder, watched),
        }

    out = node(here)
    out["children"] = [node(here + "/" + name) for name in t["children"].get(here, [])]
    out["root"] = root
    out["crumbs"] = [{"name": PurePosixPath(root).name, "path": ""}] + [
        {"name": part, "path": "/".join(rel.split("/")[: i + 1])}
        for i, part in enumerate(rel.split("/"))
        if rel
    ]
    return jsonify(out)


# ── Who speaks the most ──────────────────────────────────────────────────
# Per named person: their speech in a folder, counted like a recording's
# Speakers tab — each voice (recording + diarization label) belongs to the
# person most of its voice windows are named as, and its speech is the
# length of its segments. Names come from the voice library when asked
# and are never stored here.

_speak_cache: dict[str, tuple[float, list]] = {}


def speakers_in(database, sdb, folder: str, limit: int = 12) -> list[dict]:
    base = paths.canon(folder).rstrip("/")
    lo, hi = base + "/", base + "0"
    owner: dict[tuple[str, str], tuple[str, int]] = {}
    with sdb.connection() as c:
        for r in c.execute(
            "SELECT v.abs_path AS a, v.label AS l, p.name AS n, COUNT(*) AS k"
            " FROM speaker_assignment z JOIN speaker_vectors v ON v.id = z.vector_id"
            " JOIN speaker_people p ON p.id = z.person_id"
            " WHERE v.abs_path >= ? AND v.abs_path < ? GROUP BY v.abs_path, v.label, p.name",
            (lo, hi),
        ):
            key = (r["a"], r["l"])
            if key not in owner or r["k"] > owner[key][1]:
                owner[key] = (r["n"], r["k"])
    if not owner:
        return []
    per: dict[str, dict] = {}
    with database.connection() as c:
        for r in c.execute(
            "SELECT t.abs_path AS a, s.speaker AS l, SUM(s.end_sec - s.start_sec) AS sec"
            " FROM segments s JOIN transcripts t ON t.id = s.transcript_id"
            " WHERE t.complete = 1 AND t.abs_path >= ? AND t.abs_path < ?"
            " GROUP BY t.abs_path, s.speaker",
            (lo, hi),
        ):
            hit = owner.get((r["a"], r["l"]))
            if not hit:
                continue
            p = per.setdefault(hit[0], {"name": hit[0], "speech_sec": 0.0, "recordings": set()})
            p["speech_sec"] += max(0.0, r["sec"] or 0.0)
            p["recordings"].add(r["a"])
    out = sorted(per.values(), key=lambda p: -p["speech_sec"])[:limit]
    return [
        {
            "name": p["name"],
            "speech_sec": round(p["speech_sec"], 1),
            "recordings": len(p["recordings"]),
        }
        for p in out
    ]


def warm_speakers(folder: str) -> None:
    """Who speaks the most in a folder, counted ahead into the cache."""
    from whisper_tm.database.speaker import speaker_db
    from whisper_tm.extensions import db

    if db is None:
        return
    key = paths.canon(folder)
    _speak_cache[key] = (time.monotonic(), speakers_in(db, speaker_db(), folder))


@library_bp.route("/api/library/speakers")
def library_speakers():
    """The named people who speak the most in a folder: their speech
    (seconds) and in how many recordings. Query: ``folder`` — an absolute
    folder inside the allowed trees (default: the library root)."""
    from whisper_tm.core.security import allowed_roots, root_allowed
    from whisper_tm.database.speaker import speaker_db
    from whisper_tm.extensions import db

    if db is None:
        return jsonify({"error": "No database configured"}), 501
    folder = (request.args.get("folder") or "").strip()
    if not folder:
        roots = allowed_roots()
        folder = str(roots[0]) if roots else ""
    if not folder or not root_allowed(folder):
        return jsonify({"error": "Folder outside the library"}), 403
    key = paths.canon(folder)
    now = time.monotonic()
    hit = _speak_cache.get(key)
    if hit and now - hit[0] < CACHE_TTL * 2:
        return jsonify({"people": hit[1]})
    try:
        people = speakers_in(db, speaker_db(), folder)
    except Exception as e:  # noqa — no voice library: nobody to name
        logger.warning("Who speaks the most: %s", e)
        people = []
    _speak_cache[key] = (now, people)
    # One entry per folder ever asked for — kept to the newest 64, like the
    # statistics cache
    while len(_speak_cache) > 64:
        _speak_cache.pop(min(_speak_cache, key=lambda k: _speak_cache[k][0]), None)
    return jsonify({"people": people})

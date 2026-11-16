"""Watching the library for new files — and transcribing them.

An option in the settings for continuous scanning and automatic
transcription of new files — off by default — with a button in the top bar
to switch it on and off.

**What counts as new.** A file that wasn't there when the folder was first
watched. The first round of a folder only takes note of what is there (the
baseline) and does nothing with it: switching this on over a large library,
many of its files never transcribed, must not queue them all.
The baseline lives in its own small database (``<data>/autoscan.db``), so a
restart doesn't forget it — a file copied in while the server was down is
new on the next round.

**Not while it is still being written.** A new file waits until its size
and time haven't changed for ``settle_min`` minutes — a 4-GB video copied
onto the NAS is seen long before it is complete.

**Cheap rounds.** A round reads only the folders whose modification time
changed since the last round — a file created, deleted or renamed in a
folder changes that folder's time — and takes the rest from memory. One
``stat`` per folder instead of reading every folder of the NAS. The first
round after a start, and one every ``full_every_h`` hours, read everything.

**What happens to a new file.** It is added to the library (like an upload)
and, with ``transcribe`` on, queued at the end of the queue with the
transcription defaults of the settings — a file whose transcript already
exists only joins the library.

**A copy that stood still.** A copy that pauses longer than ``settle_min``
gets the file queued half-written; the run finds it corrupt and puts it on
the skip list. When such a file changes afterwards it is new again: it
settles and is queued once more, and the queue lets the skip entry go
(``helpers.skip_lapsed``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from whisper_tm.core import paths as _paths

logger = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": False,
    "transcribe": True,
    "interval_min": 15,
    "folders": [],
    "settle_min": 2,
    "full_every_h": 6,
}


def _corrupt_skips() -> dict:
    """The skip list's "corrupt" entries, by identity."""
    from whisper_tm.core.helpers import load_skiplist

    return {
        key: entry
        for key, entry in load_skiplist().items()
        if isinstance(entry, dict) and entry.get("reason") == "corrupt"
    }


def setting(name: str):
    from whisper_tm.config import config

    return config.get("autoscan", name, default=DEFAULTS[name])


def watched_folders() -> list[Path]:
    """The folders watched: the configured ones, else the library — the
    allowed roots.

    A folder inside another watched one is dropped: the outer round sees
    its files anyway, and the two rounds would each hold them under their
    own folder — every round would then find the other's files "new"
    (allowed roots such as ``/mnt/media`` and a folder below it)."""
    from whisper_tm.core import paths
    from whisper_tm.core.security import allowed_roots

    raw = [str(x).strip() for x in (setting("folders") or []) if str(x).strip()]
    found: list[Path] = []
    if not raw:
        found = list(allowed_roots())
    for x in raw:
        try:
            found.append(Path(x).expanduser().resolve())
        except (OSError, ValueError):
            logger.warning("autoscan.folders: unusable path %s", x)
    out: list[Path] = []
    for f in sorted(found, key=lambda p: len(str(p))):
        if not any(paths.is_inside(str(f), str(k)) for k in out):
            out.append(f)
    return out


def _library_root(path: Path) -> Path | None:
    """The allowed root a file belongs to — the base its queue entry is
    relative to, as for everything queued from the library."""
    from whisper_tm.core import paths
    from whisper_tm.core.security import allowed_roots

    best = None
    for r in allowed_roots():
        if paths.is_inside(str(path), str(r)) and (best is None or len(str(r)) > len(str(best))):
            best = r
    return best


def _own_folders() -> set[str]:
    """Folders the app writes into itself — clips and exports, the work
    folder, job logs, a central transcript folder. Never walked: what
    appears there is the app's own output, not a new recording."""
    from whisper_tm.config import config

    out = set()
    try:
        d = config.dirs
    except Exception:  # noqa: BLE001
        return out
    for key in ("clips", "work", "logs", "transcripts"):
        try:
            if d.get(key):
                out.add(os.path.realpath(str(d[key])))
        except (OSError, TypeError, ValueError):
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════
#  The walk — only the folders that changed
# ══════════════════════════════════════════════════════════════════════════


class _Folder:
    __slots__ = ("mtime", "files", "dirs")

    def __init__(self, mtime: int, files: dict, dirs: list):
        self.mtime, self.files, self.dirs = mtime, files, dirs


def walk(root: Path, cache: dict, full: bool, workers: int = 8) -> dict:
    """All media files below ``root`` as ``{abs_path: (size, mtime)}``.

    A folder whose modification time is what ``cache`` holds is not read
    again — its files and subfolders come from the cache. Returns
    ``{"files", "read", "cached", "errors", "ok"}``; ``ok`` is false when
    the root itself can't be read (a share that is gone must not look like
    a share whose files were all deleted).
    """
    from whisper_tm.config import config
    from whisper_tm.scanner.filesystem import _should_ignore

    ignore = config.get("scanner", "ignore_patterns", default=[".*"]) or []
    exts = config.media_extensions
    stats = {"read": 0, "cached": 0, "errors": 0}
    lock = threading.Lock()
    own = _own_folders()

    def visit(d: str):
        try:
            mt = os.stat(d).st_mtime_ns
        except OSError:
            with lock:
                stats["errors"] += 1
            return None
        held = cache.get(d)
        if held is not None and not full and held.mtime == mt:
            with lock:
                stats["cached"] += 1
            return held
        files, dirs = {}, []
        try:
            with os.scandir(d) as it:
                for e in it:
                    name = e.name
                    if _should_ignore(name, ignore):
                        continue
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.path not in own:
                                dirs.append(e.path)
                        elif (
                            e.is_file(follow_symlinks=False)
                            and os.path.splitext(name)[1].lower() in exts
                        ):
                            st = e.stat(follow_symlinks=False)
                            files[name] = (st.st_size, st.st_mtime)
                    except OSError:
                        with lock:
                            stats["errors"] += 1
        except OSError:
            with lock:
                stats["errors"] += 1
            return None
        folder = _Folder(mt, files, dirs)
        cache[d] = folder
        with lock:
            stats["read"] += 1
        return folder

    root_s = str(root)
    if not os.path.isdir(root_s):
        return {"files": {}, "ok": False, **stats}
    out: dict[str, tuple[int, float]] = {}
    seen = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        frontier = [root_s]
        while frontier:
            results = list(pool.map(visit, frontier))
            nxt = []
            for d, f in zip(frontier, results):
                if f is None:
                    if d == root_s:
                        return {"files": {}, "ok": False, **stats}
                    continue
                for name, sm in f.files.items():
                    out[os.path.join(d, name)] = sm
                for sub in f.dirs:
                    if sub not in seen:
                        seen.add(sub)
                        nxt.append(sub)
            frontier = nxt
    # Folders that are gone leave the cache
    for d in [k for k in cache if k.startswith(root_s) and k != root_s and k not in seen]:
        cache.pop(d, None)
    return {"files": out, "ok": True, **stats}


# ══════════════════════════════════════════════════════════════════════════
#  What was seen — a database of its own
# ══════════════════════════════════════════════════════════════════════════


class _State:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS roots (
                    root TEXT PRIMARY KEY, baseline_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS files (
                    abs_path TEXT PRIMARY KEY, root TEXT NOT NULL,
                    size INTEGER, mtime REAL,
                    first_seen REAL NOT NULL, changed_at REAL NOT NULL,
                    state TEXT NOT NULL, done_at REAL);
                CREATE INDEX IF NOT EXISTS idx_files_root ON files(root, state);
            """)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def baseline_at(self, root: str) -> float | None:
        with self._conn() as c:
            r = c.execute("SELECT baseline_at FROM roots WHERE root = ?", (root,)).fetchone()
        return r[0] if r else None

    def set_baseline(self, root: str, files: dict) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute("DELETE FROM files WHERE root = ?", (root,))
            c.executemany(
                "INSERT OR REPLACE INTO files (abs_path, root, size, mtime, first_seen, "
                "changed_at, state) VALUES (?,?,?,?,?,?, 'known')",
                [(p, root, s, m, now, now) for p, (s, m) in files.items()],
            )
            c.execute("INSERT OR REPLACE INTO roots (root, baseline_at) VALUES (?, ?)", (root, now))

    def rows(self, root: str) -> dict[str, sqlite3.Row]:
        with self._conn() as c:
            return {
                r["abs_path"]: r for r in c.execute("SELECT * FROM files WHERE root = ?", (root,))
            }

    def write(self, sql: str, rows: list) -> None:
        if rows:
            with self._conn() as c:
                c.executemany(sql, rows)

    def counts(self) -> dict:
        with self._conn() as c:
            return dict(c.execute("SELECT state, COUNT(*) FROM files GROUP BY state").fetchall())

    def note_known(self, files: dict[str, str]) -> None:
        """``{abs_path: watched folder}`` — known from now on, untouched."""
        now = time.time()
        rows = []
        for p, root in files.items():
            try:
                st = os.stat(p)
            except OSError:
                continue
            rows.append((p, root, st.st_size, st.st_mtime, now, now))
        self.write(
            "INSERT OR REPLACE INTO files (abs_path, root, size, mtime, first_seen, "
            "changed_at, state) VALUES (?,?,?,?,?,?, 'known')",
            rows,
        )

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            return [
                dict(r)
                for r in c.execute(
                    "SELECT abs_path, state, first_seen, done_at FROM files "
                    "WHERE state != 'known' ORDER BY COALESCE(done_at, first_seen) DESC LIMIT ?",
                    (limit,),
                )
            ]


# ══════════════════════════════════════════════════════════════════════════
#  The service
# ══════════════════════════════════════════════════════════════════════════


class AutoScan:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # Each loop has its own stop signal: a shared one, cleared by the
        # next start(), never reached a loop still inside a long round —
        # off-and-on within a round left two loops scanning
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._tls = threading.local()  # the running loop's stop
        self._round_lock = threading.Lock()  # one round at a time
        self._cache: dict[str, dict] = {}  # root → {folder: _Folder}
        self._last_full: dict[str, float] = {}
        self._state: _State | None = None
        self.info: dict = {
            "scanning": False,
            "last_at": 0.0,
            "last_sec": 0.0,
            "next_at": 0.0,
            "rounds": 0,
            "error": "",
            "last_round": {},
            "log": deque(maxlen=30),
        }

    def state(self) -> _State:
        if self._state is None:
            from whisper_tm.config import config

            self._state = _State(Path(config.data_dir) / "autoscan.db")
        return self._state

    # ── on / off ─────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def apply(self) -> None:
        """Start or stop to match ``autoscan.enabled``."""
        if setting("enabled"):
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        with self._lock:
            if self.running:
                self._wake.set()  # settings changed: a round now
                return
            self._stop = threading.Event()
            self._wake.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(self._stop,), name="autoscan", daemon=True
            )
            self._thread.start()
        logger.info("Autoscan on: %s", ", ".join(str(f) for f in watched_folders()))

    def stop(self) -> None:
        with self._lock:
            t = self._thread
            if t is None:
                return
            self._stop.set()
            self._wake.set()
            self._thread = None
        logger.info("Autoscan off")

    def scan_now(self) -> None:
        if self.running:
            self._wake.set()

    def _loop(self, stop: threading.Event | None = None) -> None:
        stop = stop or self._stop
        self._tls.stop = stop
        while not stop.is_set():
            t0 = time.monotonic()
            with self._round_lock:
                # A loop switched off while it waited for the lock ends here
                if stop.is_set():
                    break
                try:
                    self.round()
                    self.info["error"] = ""
                except Exception as e:  # noqa: BLE001 — shown in the status
                    logger.exception("Autoscan round failed")
                    self.info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            if stop.is_set():
                break
            took = time.monotonic() - t0
            # Never back to back on a slow share: at least twice the round
            wait = max(float(setting("interval_min")) * 60, took * 2, 30)
            self.info["next_at"] = time.time() + wait
            self._wake.wait(wait)
            self._wake.clear()

    # ── one round ────────────────────────────────────────────────────────

    def round(self) -> dict:
        from whisper_tm.config import config

        self.info["scanning"] = True
        summary = {"folders": [], "new": 0, "queued": 0, "added": 0, "pending": 0}
        try:
            workers = int(config.get("scanner", "walk_workers", default=8) or 8)
            settle = float(setting("settle_min")) * 60
            full_every = float(setting("full_every_h")) * 3600
            stop = getattr(self._tls, "stop", None) or self._stop
            for folder in watched_folders():
                if stop.is_set():
                    break
                root = str(folder)
                if _library_root(folder) is None:
                    summary["folders"].append(
                        {
                            "folder": root,
                            "error": "outside the allowed folders (security.allowed_roots)",
                        }
                    )
                    continue
                cache = self._cache.setdefault(root, {})
                full = time.time() - self._last_full.get(root, 0) >= full_every
                w = walk(folder, cache, full, workers=min(workers, 32))
                if full and w["ok"]:
                    self._last_full[root] = time.time()
                one = {
                    "folder": root,
                    "files": len(w["files"]),
                    "read": w["read"],
                    "cached": w["cached"],
                    "errors": w["errors"],
                }
                if not w["ok"]:
                    one["error"] = "can't be read"
                    summary["folders"].append(one)
                    continue
                st = self.state()
                if st.baseline_at(root) is None:
                    st.set_baseline(root, w["files"])
                    one["baseline"] = True
                    self._log(f"{root}: {len(w['files']):,} files taken as the starting point")
                    summary["folders"].append(one)
                    continue
                one.update(self._compare(root, w, settle))
                for k in ("new", "queued", "added", "pending"):
                    summary[k] += one.get(k, 0)
                summary["folders"].append(one)
        finally:
            self.info["scanning"] = False
            self.info["last_at"] = time.time()
            self.info["rounds"] += 1
            self.info["last_round"] = summary
        return summary

    def _compare(self, root: str, w: dict, settle: float) -> dict:
        from whisper_tm.core.helpers import skip_lapsed

        st = self.state()
        now = time.time()
        rows = st.rows(root)
        files = w["files"]
        inserts, changed, ready, again = [], [], [], []
        skips = None
        for p, (size, mtime) in files.items():
            r = rows.get(p)
            if r is None:
                inserts.append((p, root, size, mtime, now, now))
            elif r["state"] == "pending":
                # Asked directly: a file still being written grows without
                # its folder's time changing, so the walk's cached size of
                # an unchanged folder would call it settled too early
                try:
                    fresh = os.stat(p)
                    size, mtime = fresh.st_size, fresh.st_mtime
                except OSError:
                    continue
                if r["size"] != size or abs((r["mtime"] or 0) - mtime) > 0.5:
                    changed.append((size, mtime, now, p))
                elif now - r["changed_at"] >= settle:
                    ready.append(p)
            elif r["state"] == "queued":
                if skips is None:
                    skips = _corrupt_skips()
                entry = skips.get(_paths.canon(p))
                if entry is None:
                    continue
                try:
                    fresh = os.stat(p)
                except OSError:
                    continue
                if skip_lapsed(p, entry, fresh.st_mtime):
                    again.append((fresh.st_size, fresh.st_mtime, now, p))
        st.write(
            "INSERT OR REPLACE INTO files (abs_path, root, size, mtime, first_seen, "
            "changed_at, state) VALUES (?,?,?,?,?,?, 'pending')",
            inserts,
        )
        st.write("UPDATE files SET size = ?, mtime = ?, changed_at = ? WHERE abs_path = ?", changed)
        st.write(
            "UPDATE files SET size = ?, mtime = ?, changed_at = ?, state = 'pending', "
            "done_at = NULL WHERE abs_path = ?",
            again,
        )
        for p in inserts:
            self._log(f"new: {p[0]}")
        for p in again:
            self._log(f"changed since it was found corrupt: {p[3]}")
        out = {
            "new": len(inserts),
            "queued": 0,
            "added": 0,
            "pending": len(inserts) + len(changed) + len(again),
        }
        done = []
        for p in ready:
            what = self._act(Path(p))
            done.append((what, time.time(), p))
            out[what] = out.get(what, 0) + 1
            self._log(f"{'queued' if what == 'queued' else what}: {p}")
        st.write("UPDATE files SET state = ?, done_at = ? WHERE abs_path = ?", done)
        # Files that are gone: only after a walk without errors — a folder
        # that couldn't be read is not a folder whose files were deleted
        if not w["errors"]:
            gone = [(p,) for p in rows if p not in files]
            st.write("DELETE FROM files WHERE abs_path = ?", gone)
        return out

    def _act(self, path: Path) -> str:
        """Add one settled new file to the library — and queue it."""
        from whisper_tm.core import paths
        from whisper_tm.core import queue as q
        from whisper_tm.core.helpers import is_transcript_complete
        from whisper_tm.scanner import register_single_file

        lib_root = _library_root(path)
        if lib_root is None:
            return "skipped"
        rel = register_single_file(str(path), root=str(lib_root)) or paths.rel_from(path, lib_root)
        if not rel:
            return "skipped"
        if not setting("transcribe"):
            return "added"
        try:
            if is_transcript_complete(rel, media_dir=lib_root):
                return "added"
        except Exception as e:  # noqa: BLE001 — then the queue decides
            logger.debug("autoscan transcript check %s: %s", rel, e)
        added = q.enqueue([rel], {}, str(lib_root))
        return "queued" if added else "added"

    def _log(self, text: str) -> None:
        logger.info("Autoscan: %s", text)
        self.info["log"].appendleft({"at": time.time(), "text": text})

    # ── for the page ─────────────────────────────────────────────────────

    def status(self) -> dict:
        counts = {}
        recent = []
        try:
            counts = self.state().counts()
            recent = self.state().recent()
        except Exception as e:  # noqa: BLE001
            logger.debug("autoscan state: %s", e)
        return {
            "enabled": bool(setting("enabled")),
            "running": self.running,
            "transcribe": bool(setting("transcribe")),
            "interval_min": setting("interval_min"),
            "settle_min": setting("settle_min"),
            "folders": [str(f) for f in watched_folders()],
            # What it watches: off, the whole library, or chosen folders
            "mode": (
                "off"
                if not setting("enabled")
                else "folders"
                if [x for x in (setting("folders") or []) if str(x).strip()]
                else "library"
            ),
            "chosen": [str(x) for x in (setting("folders") or []) if str(x).strip()],
            "scanning": self.info["scanning"],
            "last_at": self.info["last_at"],
            "next_at": self.info["next_at"] if self.running else 0,
            "rounds": self.info["rounds"],
            "error": self.info["error"],
            "last_round": self.info["last_round"],
            "watched_files": sum(counts.values()),
            "counts": counts,
            "recent": recent,
            "log": list(self.info["log"])[:15],
        }


AUTOSCAN = AutoScan()


def note_made(*items) -> None:
    """Files (or folders of files) the app itself just wrote into the
    library — an export into a folder of your choice, the separated tracks
    next to a recording. Without this a round would find them new and
    queue them: a TTS dataset of 3,000 clips transcribed clip by clip.
    Only for watched folders whose baseline exists; never raises."""
    try:
        roots = [str(f) for f in watched_folders()]
        st = AUTOSCAN.state()
        files: dict[str, str] = {}
        for item in items:
            if not item:
                continue
            p = Path(item)
            paths = [str(x) for x in p.rglob("*") if x.is_file()] if p.is_dir() else [str(p)]
            for f in paths:
                real = os.path.realpath(f)
                root = next((r for r in roots if _paths.is_inside(real, r)), None)
                if root and st.baseline_at(root) is not None:
                    files[real] = root
        if files:
            st.note_known(files)
    except Exception as e:  # noqa: BLE001 — a nicety for the scanner
        logger.debug("autoscan note_made: %s", e)

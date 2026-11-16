"""
Helper functions: formatting, transcript paths, skiplist management.
Cross-platform compatible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import paths

logger = logging.getLogger(__name__)


# ── Formatting ───────────────────────────────────────────────────────────


def import_clean(name: str):
    """Import ``name`` — the first time from a thread of its own, whose stack
    holds nothing.

    Some libraries keep the exception of an optional import that failed, to
    raise it once the feature is used: optuna's ``try_import`` keeps "No
    module named 'redis'" (whisperx.asr → pyannote → optuna). An exception
    keeps its traceback, and a traceback keeps every frame above it alive —
    with the locals of whoever happened to import first, for as long as the
    process runs. The first import of whisperx.asr ran inside
    ``_asr_pipeline``, below the first queue job: its frames held large-v3's
    weights, and "Unload VRAM" freed everything but those 4 GB.
    From a thread of its own, it is that thread's empty frames that stay.

    A module that fails to import fails again here, in the caller — with
    its own error, as a plain import would.
    """
    import importlib
    import sys

    if name not in sys.modules:

        def load():
            try:
                importlib.import_module(name)
            except BaseException:  # noqa: BLE001 — raised again below, in the caller
                pass

        t = threading.Thread(target=load, name=f"import {name}", daemon=True)
        t.start()
        t.join()
    return importlib.import_module(name)


def format_ts(seconds: float) -> str:
    """Seconds → H:MM:SS string."""
    # NaN/inf/None (e.g. from invalid duration) → 0 instead of crash
    if seconds is None or not math.isfinite(seconds):
        seconds = 0
    s = int(seconds)
    if s < 0:
        s = 0
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def format_size(b: int | float) -> str:
    """Bytes → human-readable size."""
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"


def format_duration(secs: float) -> str:
    """Seconds → compact time format."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}min"
    return f"{secs / 3600:.1f}h"


def parse_bool(value) -> bool:
    """Robust bool parsing: accepts real bools AND strings like
    'false'/'0'/'no'/'off' (which bool() would incorrectly convert to True)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in ("false", "0", "no", "off", "")
    return bool(value)


# ── Transcript Paths ─────────────────────────────────────────────────────


def transcript_path_beside_file(abs_path: str | Path) -> Path:
    """``…/folder/film.mp4`` → ``…/folder/.transcripts/film.mp4.json``.

    Storage for ``transcript_mode: alongside``: the transcript sits in the
    folder of the media file, not in a central collection folder. The path
    depends solely on the actual location of the file — ``config.media_dir``
    plays no role and may change at runtime.

    This also eliminates ambiguity of flat names: two files with the same name
    in different folders each get their own ``.transcripts/`` and can no longer
    overwrite each other.
    """
    p = Path(paths.canon(abs_path) or str(abs_path))
    return p.parent / ".transcripts" / (p.name + ".json")


def transcript_path_central(abs_path: str | Path) -> Path:
    """Flat name in central collection folder (``transcript_mode: central``).

    The **absolute** path is encoded, not the ``rel_path``. That's the
    difference between unambiguous and not: ``rel_path`` depends on the
    currently active root; two files with the same name from different folders
    would end up on the same JSON file and overwrite each other (docs/PATHS.md).
    This can't happen with the absolute path.
    """
    canonical = paths.canon(abs_path)
    safe = canonical.strip("/").replace("/", "__").replace(":", "_")
    fname = safe + ".json"
    if _flat_name_ambiguous(canonical) and len(fname.encode("utf-8")) <= 240:
        # The flat form could have come from another path as well
        # ("/a/b__c" and "/a__b/c" are both "a__b__c"): the second
        # transcript overwrote the first. A NEW transcript
        # of such a path gets a hash of the path — but one already written
        # under the flat name stays THE transcript of its recording while
        # it exists: a second file beside it made edits
        # skip the JSON, a delete leave the old one to be found again, and a
        # re-sync import two files for one recording.
        legacy = config.transcript_dir / fname
        if _legacy_is_ours(legacy, canonical):
            return legacy
        return _transcript_path_central_hashed(canonical)
    if len(fname.encode("utf-8")) > 240:
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        short = safe[-100:] if len(safe) > 100 else safe
        # Truncate on BYTES, not just characters: 100 CJK characters are ~300
        # UTF-8 bytes and exceed the 255-byte filesystem limit.
        while len(short.encode("utf-8")) > 200 and short:
            short = short[1:]
        fname = f"{h}_{short}.json"
    return config.transcript_dir / fname


def _flat_name_ambiguous(canonical: str) -> bool:
    """Could the flat central name of ``canonical`` also come from another
    path? "/" becomes "__" and ":" becomes "_", so a path holding "__", a
    ":", or an "_" beside a "/" can't be told apart from its twin."""
    inner = canonical.strip("/")
    return "__" in inner or ":" in inner or "_/" in inner or "/_" in inner


def _transcript_path_central_hashed(canonical: str) -> Path:
    """The hashed central name of an ambiguous path (new transcripts)."""
    safe = canonical.strip("/").replace("/", "__").replace(":", "_")
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    stem = safe
    while len(f"{stem}.{h}.json".encode()) > 240 and stem:
        stem = stem[1:]
    return config.transcript_dir / f"{stem}.{h}.json"


def _legacy_is_ours(legacy: Path, canonical: str) -> bool:
    """Is the flat-named file ``legacy`` the transcript of ``canonical``?

    It exists, and the ``filename`` it records fits this path: the
    recording's own path, or a relative path this one ends with. Its twin
    (``/m/a/b/c.mp4`` for ``/m/a__b/c.mp4``) recorded ``a/b/c.mp4`` — that
    file is not ours to read or overwrite. A bare name that fits both can't
    be told apart; it is taken, as it always was.
    """
    try:
        if not legacy.is_file():
            return False
        with open(legacy, encoding="utf-8") as f:
            name = (json.load(f) or {}).get("filename") or ""
    except (OSError, ValueError, AttributeError):
        return True  # unreadable: never hide an existing file
    name = str(name).strip()
    if not name:
        return True
    if paths.is_absolute(name):
        return paths.canon(name) == canonical
    return canonical == "/" + name.strip("/") or canonical.endswith("/" + name.strip("/"))


def transcript_files_for(abs_path: str | Path) -> list[Path]:
    """Every transcript file of a recording that exists — for deleting it.

    Normally one. Central mode can hold two for an ambiguous path: the
    flat-named one from an earlier version and a hashed one written beside it;
    a delete that left one would have it found again.
    """
    canonical = paths.canon(abs_path)
    if not canonical:
        return []
    if config.transcript_mode == "alongside":
        p = transcript_path_beside_file(canonical)
        return [p] if p.exists() else []
    out = []
    main = transcript_path_central(canonical)
    if main.exists():
        out.append(main)
    if _flat_name_ambiguous(canonical):
        legacy = config.transcript_dir / (
            canonical.strip("/").replace("/", "__").replace(":", "_") + ".json"
        )
        hashed = _transcript_path_central_hashed(canonical)
        if hashed not in out and hashed.exists():
            out.append(hashed)
        # The flat-named file only when it is this recording's, not its twin's
        if legacy not in out and _legacy_is_ours(legacy, canonical):
            out.append(legacy)
    return out


def transcript_identity(
    relpath: str, media_dir: str | Path | None = None, abs_path: str | Path | None = None
) -> str:
    """Actual location of the media file — the identity (docs/PATHS.md).

    ``abs_path`` is the direct path; otherwise it's derived from ``relpath`` and the
    root. Empty return value means: not derivable.
    """
    if abs_path:
        k = paths.canon(abs_path)
        if k:
            return k
    if paths.is_absolute(relpath):
        return paths.canon(relpath)
    root = str(media_dir) if media_dir else ""
    if not root:
        try:
            root = str(config.media_dir)
        except Exception:  # noqa
            root = ""
    return paths.abs_from(relpath, root) if root else ""


def transcript_path_for(
    relpath: str, media_dir: str | Path | None = None, abs_path: str | Path | None = None
) -> Path:
    """The location of a media file's transcript. Exactly one.

    The mode decides (``directories.transcript_mode``):

    * ``alongside`` — ``<file folder>/.transcripts/<name>.json``
    * ``central`` / ``database`` — flat-encoded **absolute** path in the
      central folder

    Both forms depend on the identity, so are unambiguous. There's no second
    location additionally searched.

    Raises:
        ValueError: if the identity cannot be derived. This is a programming
            error at the call site — falling back to a guessed location would
            move the transcript instead of showing the error.
    """
    target = transcript_identity(relpath, media_dir, abs_path)
    if not target:
        raise ValueError(
            f"No transcript path derivable for {relpath!r} "
            f"(media_dir={media_dir!r}, abs_path={abs_path!r}) — "
            f"the call site must know the actual location "
            f"(docs/PATHS.md)"
        )
    if config.transcript_mode == "alongside":
        return transcript_path_beside_file(target)
    return transcript_path_central(target)


def find_transcript(
    relpath: str, media_dir: Path | None = None, abs_path: str | Path | None = None
) -> Path | None:
    """Transcript JSON of a media file — or None.

    There's exactly ONE possible location (:func:`transcript_path_for`). No
    second is searched. (An ambiguous central name written by an earlier
    version IS that location while it exists — transcript_path_central.)
    """
    try:
        tp = transcript_path_for(relpath, media_dir, abs_path)
    except ValueError:
        return None
    try:
        return tp if tp.exists() else None
    except OSError:
        return None


def atomic_write_json(path: str | Path, data, indent: int | None = 2) -> None:
    """Write a JSON file in one step: a temporary file in the same folder,
    then ``os.replace`` — a crash, a full disk or a dropped share mid-write
    leaves the old file as it was, never a truncated one.

    THE way to write a transcript (they are precious, and they may live on a
    network share next to the recordings). The temporary name is short and derived
    from the target name, like the worker's: target names can already be
    near the 255-byte limit.
    """
    path = Path(path)
    tag = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    tmp = path.with_name(f".wtm-{tag}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ── Transcript Completion Cache ──────────────────────────────────────────
# Avoids repeated JSON parsing and DB queries during scan/batch.
# TTL-based: cache entries expire after _COMPLETE_CACHE_TTL seconds.

_complete_cache: dict[str, tuple[bool, float]] = {}  # relpath → (is_complete, timestamp)
_COMPLETE_CACHE_TTL = 30.0  # seconds
_COMPLETE_CACHE_MAX = 10000  # hard ceiling against unbounded growth
_complete_cache_lock = threading.Lock()  # protects _complete_cache (multiple scan threads)


def _prune_complete_cache() -> None:
    """Prune cache: expired entries + hard ceiling (LRU-like).

    Must be called under _complete_cache_lock. Runs only when the
    ceiling is exceeded (amortized O(1) per insert).
    """
    if len(_complete_cache) <= _COMPLETE_CACHE_MAX:
        return
    now = time.time()
    # First remove expired entries
    for k in [k for k, (_, ts) in _complete_cache.items() if now - ts >= _COMPLETE_CACHE_TTL]:
        _complete_cache.pop(k, None)
    # If still over ceiling: discard oldest (FIFO)
    while len(_complete_cache) > _COMPLETE_CACHE_MAX:
        try:
            _complete_cache.pop(next(iter(_complete_cache)), None)
        except StopIteration:
            break


def invalidate_complete_cache(
    relpath: str | None = None, media_dir: str | Path | None = None
) -> None:
    """Invalidate cache — call after transcription/deletion.

    Args:
        relpath: Invalidate specific entry, or None for all.
        media_dir: Root that ``relpath`` refers to.
    """
    with _complete_cache_lock:
        if relpath is None:
            _complete_cache.clear()
            return
        # Cache is keyed on identity (see is_transcript_complete).
        # A pop on bare relpath would hit nothing, and the file would keep
        # its old status until TTL expires.
        root = str(media_dir) if media_dir else ""
        if not root:
            try:
                root = str(config.media_dir)
            except Exception:  # noqa
                root = ""
        _complete_cache.pop(paths.abs_from(relpath, root) if root else "", None)
        _complete_cache.pop(f"{root}\x00{relpath}", None)
        _complete_cache.pop(relpath, None)


def is_transcript_complete(relpath: str, media_dir: Path | None = None) -> bool:
    """Check if transcript exists AND is marked as complete.

    Uses TTL cache to avoid repeated filesystem/DB accesses.
    """
    import time as _time

    # Key is IDENTITY, not display value. Using relpath alone, the cache
    # would report an untranscribed file as done once a same-named file
    # under the current root had one — then the queue worker would skip it
    # (docs/PATHS.md).
    root = str(media_dir) if media_dir else str(config.media_dir)
    key = paths.abs_from(relpath, root) or f"{root}\x00{relpath}"

    # Cache check (under lock — multiple scan threads access concurrently)
    now = _time.time()
    with _complete_cache_lock:
        cached = _complete_cache.get(key)
        if cached is not None:
            is_complete, ts = cached
            if now - ts < _COMPLETE_CACHE_TTL:
                return is_complete

    # Keep expensive check outside lock (does I/O: DB + filesystem)
    result = _check_transcript_complete(relpath, media_dir)

    with _complete_cache_lock:
        _complete_cache[key] = (result, _time.time())
        _prune_complete_cache()
    return result


def _check_transcript_complete(relpath: str, media_dir: Path | None = None) -> bool:
    """Uncached implementation of is_transcript_complete."""
    # Check DB first
    from whisper_tm.extensions import db

    if db:
        try:
            # Use the PASSED root, not blindly config.media_dir:
            # the caller knows what relpath refers to (queue entry, scan root),
            # and that may differ from the current media directory. The argument
            # was previously discarded.
            root = str(media_dir) if media_dir else str(config.media_dir)
            if db.is_complete(relpath, media_dir=root):
                return True
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)

    # Fallback: JSON (multiple locations)
    try:
        tp = find_transcript(relpath, media_dir)
        if not tp:
            return False
        with open(tp, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        is_complete = data.get("complete", False) is True

        # Heal DB: take JSON matches without DB entry into DB,
        # so DB becomes the primary source from next access on
        # (e.g. legacy files from before DB introduction).
        if is_complete and db:
            try:
                db.upsert_transcript(relpath, data, media_dir=str(media_dir or config.media_dir))
            except Exception as e:
                logger.debug("DB heal failed: %s", e)

        return is_complete
    except (OSError, ValueError) as e:
        # Expected errors (I/O or corrupted JSON/Unicode) → not complete,
        # but log instead of silently swallowing. Unexpected errors
        # propagate now instead of falsely being "not complete".
        logger.debug("Transcript check failed for %s: %s", relpath, e)
        return False


# ── Skiplist ─────────────────────────────────────────────────────────────


def _skiplist_path() -> Path:
    return config.transcript_dir / ".skiplist.json"


# Process-wide skiplist cache: avoids is_skipped reloading the entire skiplist
# at EVERY call (O(N) per file in scan). Updated in add_to_skiplist;
# TTL catches changes via other paths.
_skiplist_lock = threading.Lock()
_skiplist_cache: dict | None = None
_skiplist_cache_ts: float = 0.0
_SKIPLIST_CACHE_TTL = 30.0  # seconds


def load_skiplist() -> dict:
    """Load skiplist (DB or JSON)."""
    from whisper_tm.extensions import db

    if db:
        try:
            return db.get_skiplist()
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)
    # Fallback JSON
    sp = _skiplist_path()
    if sp.exists():
        try:
            with open(sp, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)
    return {}


def save_skiplist(data: dict) -> None:
    sp = _skiplist_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (temp + os.replace): a crash mid-write
    # must not corrupt the skiplist as a partial file.
    atomic_write_json(sp, data)


def skiplist_key(relpath: str, media_dir: str | Path | None = None) -> str:
    """Key of a file in the skiplist — the identity (docs/PATHS.md).

    Falls back to ``rel_path`` if no identity can be derived;
    this keeps legacy files findable.
    """
    root = str(media_dir) if media_dir else ""
    if not root:
        try:
            root = str(config.media_dir)
        except Exception:  # noqa
            root = ""
    if paths.is_absolute(relpath):
        return paths.canon(relpath) or relpath
    return (paths.abs_from(relpath, root) if root else "") or relpath


def invalidate_skiplist_cache() -> None:
    """Discard skiplist cache — call after every change.

    Without this, removing from the skiplist only takes effect after 30
    seconds (TTL expires), and a just-freed file continues to be skipped.
    """
    global _skiplist_cache, _skiplist_cache_ts
    with _skiplist_lock:
        _skiplist_cache = None
        _skiplist_cache_ts = 0.0


def add_to_skiplist(
    relpath: str,
    reason: str = "no_audio",
    media_dir: str | Path | None = None,
    abs_path: str | Path | None = None,
) -> None:
    """Add file to the skiplist.

    ``abs_path`` is the actual location. Whoever knows it should provide it —
    that's more reliable than a root that the caller often doesn't have
    (e.g. the transcription worker only knows ``source_path``).
    """
    from whisper_tm.extensions import db

    global _skiplist_cache, _skiplist_cache_ts
    # Atomic read-modify-write under lock: concurrent scan threads must not
    # overwrite each other's entries during shared skiplist updates.
    with _skiplist_lock:
        if db:
            try:
                # PASS abs_path. Production callers (core/transcription.py) only
                # know source_path and don't provide media_dir — without this path,
                # db.add_skip would derive identity from config.media_dir and lock
                # the same-named file under the current root instead of the actual
                # broken one (docs/PATHS.md).
                db.add_skip(
                    relpath,
                    reason,
                    media_dir=(str(media_dir) if media_dir else None),
                    abs_path=(str(abs_path) if abs_path else None),
                )
            except Exception as e:
                logger.debug("Unhandled exception: %s", e)
        sl = load_skiplist()
        # Store under identity, not display value — otherwise a skipped file
        # would lock every same-named file in other directories too.
        key = paths.canon(abs_path) if abs_path else skiplist_key(relpath, media_dir)
        sl[key or relpath] = {
            "reason": reason,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rel_path": relpath,
            "added_at": time.time(),
        }
        save_skiplist(sl)
        # Update cache (instead of just invalidating) — saves the next load.
        _skiplist_cache = sl
        _skiplist_cache_ts = time.time()


def remove_from_skiplist(relpath: str, media_dir: str | Path | None = None) -> None:
    """Take a file off the skip list — database and JSON — under the same
    lock :func:`add_to_skiplist` holds for its read-modify-write. Without it
    a lane's entry written in between was lost."""
    from whisper_tm.extensions import db

    global _skiplist_cache, _skiplist_cache_ts
    with _skiplist_lock:
        if db:
            db.remove_skip(relpath, media_dir=(str(media_dir) if media_dir else None))
        key = skiplist_key(relpath, media_dir)
        sl = load_skiplist()
        if key in sl or relpath in sl:
            sl.pop(key, None)
            sl.pop(relpath, None)
            save_skiplist(sl)
        _skiplist_cache = None
        _skiplist_cache_ts = 0.0


def is_skipped(relpath: str, media_dir: str | Path | None = None) -> bool:
    global _skiplist_cache, _skiplist_cache_ts
    with _skiplist_lock:
        now = time.time()
        if _skiplist_cache is None or now - _skiplist_cache_ts >= _SKIPLIST_CACHE_TTL:
            _skiplist_cache = load_skiplist()
            _skiplist_cache_ts = now
        sl = _skiplist_cache
    key = skiplist_key(relpath, media_dir)
    if key in sl:
        if not skip_lapsed(key, sl[key]):
            return True
        lift_skip(relpath, media_dir)
        return False
    # Fallback for legacy entries still under display value. Only then —
    # otherwise name equality would be a hit again.
    entry = sl.get(relpath)
    return entry is not None and not paths.is_absolute(relpath)


def skip_lapsed(key: str, entry, mtime: float | None = None) -> bool:
    """A file found corrupt that changed since: it was most likely still
    being copied (a video queued half-written) — it gets its chance again.

    Only "corrupt": a skip by hand or for having no sound stays whatever
    happens to the file. A modification time ahead of this machine's clock
    (a NAS whose clock runs fast) counts once the clock reaches it — a
    broken file then gets one more try, not one per look.
    """
    if not isinstance(entry, dict) or entry.get("reason") != "corrupt":
        return False
    try:
        added = float(entry.get("added_at") or 0)
        if mtime is None:
            mtime = Path(key).stat().st_mtime
    except (OSError, TypeError, ValueError):
        return False
    return added > 0 and added + 1 < mtime <= time.time() + 1


def lift_skip(relpath: str, media_dir: str | Path | None = None) -> None:
    """Take a lapsed skip entry off the list (:func:`skip_lapsed`)."""
    logger.info("%s changed since it was found corrupt — trying it again", relpath)
    try:
        remove_from_skiplist(relpath, media_dir)
    except Exception as e:  # noqa: BLE001 — the caller treats it as not skipped anyway
        logger.debug("Skip entry not removed: %s", e)


# ── Segment Helper Functions ──────────────────────────────────────────────

# ── Whisper artifacts ──────────────────────────────────────────────────────
#
# Text Whisper invents on music or silence — a TV station's subtitle credit
# above all. Previously a fixed set of exact phrases: a segment went
# only when it was one of them character for character, so a credit with a
# year behind it, or a period where the set had an exclamation mark, passed
# every time — a soundtrack came back full of them. Now regular expressions
# from the config (transcription.hallucination_patterns), matched against
# the WHOLE segment text: a pattern never cuts words out of a longer segment.

#: Around a segment's text, these don't count when matching
_EDGE = " \t\r\n.…,;:!?-–—\"'„“”«»‚‘’()[]♪"

_patterns_cache: tuple[tuple, list] | None = None  # (config value, compiled)


def compile_artifact_patterns(patterns) -> tuple[list, list[str]]:
    """Compile pattern lines: (the patterns, one message per bad line).

    Empty lines are skipped; case is ignored.
    """
    compiled, errors = [], []
    lines = patterns.splitlines() if isinstance(patterns, str) else list(patterns or [])
    for n, raw in enumerate(lines, 1):
        p = str(raw).strip()
        if not p:
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            errors.append(f"line {n} ({p}): {e}")
    return compiled, errors


def _artifact_patterns() -> list:
    """The configured patterns, compiled once per config value — not at
    every segment."""
    global _patterns_cache
    raw = config.get("transcription", "hallucination_patterns", default=[]) or []
    key = tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else (str(raw),)
    cache = _patterns_cache
    if cache is None or cache[0] != key:
        compiled, errors = compile_artifact_patterns(list(key))
        for e in errors:
            logger.warning("Artifact pattern skipped — not a valid regular expression: %s", e)
        cache = (key, compiled)
        _patterns_cache = cache
    return cache[1]


def artifact_pattern(text: str, patterns=None) -> str | None:
    """The pattern a segment text is a Whisper artifact of — None if none.

    The whole text has to match, without the spaces and punctuation around
    it, and case doesn't matter: ``Subtitles by.*`` takes "Subtitles by the
    community." — and leaves "The subtitles by the door were wrong."
    ``patterns``: compiled patterns to use instead of the configured ones
    (the settings page tries lines that are not saved yet).
    """
    core = (text or "").strip().strip(_EDGE)
    if not core:
        return None
    for rx in _artifact_patterns() if patterns is None else patterns:
        if rx.fullmatch(core):
            return rx.pattern
    return None


def artifact_reason(text: str, patterns=None) -> str | None:
    """Why the artifact filter drops a segment text — None if it keeps it:
    the pattern it matches, or that it holds no letter or digit at all
    ("…", "♪ ♪" — nothing that could be speech)."""
    cleaned = (text or "").strip().strip(_EDGE)
    if not re.search(r"\w", cleaned):
        return "no letter or digit"
    return artifact_pattern(cleaned, patterns)


def is_silent(seg: dict, min_chars: int = 0) -> bool:
    """Check if segment is silent/hallucination.

    Pattern matching depends on ``transcription.filter_hallucinations``.
    Length checking doesn't: it has its own switch with
    ``min_segment_chars`` and filters real silence, not hallucinations.
    """
    text = (seg.get("text") or "").strip()
    if len(text.strip(_EDGE)) < min_chars:
        return True
    if not config.get("transcription", "filter_hallucinations", default=False):
        return False
    return artifact_reason(text) is not None


# ── Quotation marks around a whole line ──────────────────────────────────
#
# Whisper wraps a line that sounds like dialogue in German guillemets —
# learned from books and subtitles: "»It is ready." … "Try it now.«". None of
# it is spoken. Many transcripts can carry them at a segment's edge,
# and a TTS dataset built from them takes them over. A mark at the edge
# goes: the opening one when the segment does not close it before its end,
# the closing one when the segment did not open it (the quote began a
# segment earlier). A quote inside a sentence — "the so-called »plan«",
# "She said: »Done.«" — stays.

_QUOTE_CLOSE = {"»": "«", "‹": "›"}
_QUOTE_OPEN = {close: opening for opening, close in _QUOTE_CLOSE.items()}


def edge_quotes(text: str) -> tuple[str, str, str]:
    """``text`` without the quotation marks Whisper put around the line:
    (the text, the opening mark taken or "", the closing mark taken or "").
    A text that would be left empty stays as it is."""
    t = (text or "").strip()
    lead = trail = ""
    if t[:1] in _QUOTE_CLOSE:
        close = _QUOTE_CLOSE[t[0]]
        inner = t[1:-1] if t.endswith(close) else t[1:]
        if close not in inner:
            lead, t = t[0], t[1:].lstrip()
    if t[-1:] in _QUOTE_OPEN and _QUOTE_OPEN[t[-1]] not in t[:-1]:
        trail, t = t[-1], t[:-1].rstrip()
    if not t:
        return (text or "").strip(), "", ""
    return t, lead, trail


def _trim_word(words: list, i: int, mark: str, front: bool) -> None:
    """The mark off the first (``front``) or last word — or the word itself
    when the mark was all of it."""
    w = words[i]
    if not isinstance(w, dict):
        return
    t = str(w.get("word") or "").strip()
    if front and t.startswith(mark):
        t = t[1:].lstrip()
    elif not front and t.endswith(mark):
        t = t[:-1].rstrip()
    else:
        return
    if t:
        w["word"] = t
    else:
        del words[i]


def fmt_seg(seg: dict, include_words: bool = False) -> dict:
    """Format segment to unified output format."""
    text, lead, trail = edge_quotes(seg.get("text") or "")
    result = {
        "speaker": seg.get("speaker", "—"),
        "start": round(seg["start"], 2),
        "end": round(seg["end"], 2),
        "start_fmt": format_ts(seg["start"]),
        "end_fmt": format_ts(seg["end"]),
        "text": text,
    }
    # Scores & markers from post-processing — only if
    # present, to keep legacy transcripts and tests unchanged.
    #
    # ⚠ The flip side: any field the pipeline loses along the way simply
    # goes missing here — no error, no warning, the DB column stays NULL.
    # That's exactly how segments can end up without confidence and without markers,
    # without anything failing. Whoever doesn't find a value here
    # should look further up, not in this function — first at phase 3 of
    # `word_boundary_segmentation`.
    if seg.get("avg_logprob") is not None:
        result["avg_logprob"] = round(float(seg["avg_logprob"]), 3)
    if seg.get("phantom_score"):
        result["phantom_score"] = seg["phantom_score"]
    if seg.get("overlapped"):
        result["overlapped"] = 1
    # Word-level timestamps (from WhisperX alignment)
    # WhisperX can provide start/end/score = None → round(None) crashes
    if include_words and "words" in seg:
        result["words"] = [
            {
                "word": w.get("word", ""),
                "start": round(w["start"], 3) if w.get("start") is not None else None,
                "end": round(w["end"], 3) if w.get("end") is not None else None,
                "score": round(w.get("score") or 0, 3),
            }
            for w in seg["words"]
            if (w.get("word") or "").strip()
        ]
        if lead and result["words"]:
            _trim_word(result["words"], 0, lead, front=True)
        if trail and result["words"]:
            _trim_word(result["words"], -1, trail, front=False)
    return result

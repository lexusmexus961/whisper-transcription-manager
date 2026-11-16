"""
Media indexer: reads FFprobe metadata (duration, resolution, codecs, …)
ONCE per file and stores it permanently in DB (media_files).

Goals:
- Reduce load times: duration/metadata come from DB instead of repeated
  FFprobe calls on each directory open.
- Resumable: already indexed, unchanged files are skipped
  (comparison via index_mtime == current file mtime).
- Transcript discovery: JSON transcripts are discovered on filesystem
  and imported to DB (DB-first, JSON stays in place).

The core is a generator (`index_directory`) that yields progress dicts —
so the SSE endpoint can stream it and a test can consume it directly.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import paths
from whisper_tm.core.audio_utils import get_media_info
from whisper_tm.scanner.filesystem import collect_media_files

logger = logging.getLogger(__name__)


class _RateEstimator:
    """Adaptive processing rate estimate (files/s) → stable AND responsive ETA.

    Why not naive total average ``processed / elapsed``?
      - It drags a cold start (NAS connection setup, cold FS cache) through
        the *entire* run and thus underestimates speed permanently.
      - It barely reacts to real changes: if NAS speeds up in second half,
        that barely shows in total average → ETA stays too pessimistic.

    Instead an exponentially smoothed average with *time constant*::

        alpha = 1 - exp(-dt / tau)
        rate  = alpha * instantaneous_rate + (1 - alpha) * rate

    The trick is time constant ``tau``: smoothing hangs on really elapsed
    time ``dt``, not on (varying) reporting frequency. This makes the
    estimator self-scaling:
      - slow NAS (few files/s, large dt): large alpha → reacts quickly
        to load changes;
      - fast local run (many files/s, tiny dt): small alpha → smooths over
        many files, no jitter.

    Only real, positive progress intervals flow in; outliers (one interval)
    fade out over the following ticks.
    """

    __slots__ = ("tau", "rate", "_last_t", "_last_n")

    # tau=8s: empirically balanced (simulation). Smaller reacts faster
    # to real load changes (NAS), but jitter stays below ~1.5% thanks to
    # 15-file reporting throttle; smaller brings barely more responsiveness,
    # larger becomes noticeably sluggish.
    def __init__(self, tau_sec: float = 8.0):
        self.tau = tau_sec
        self.rate: float | None = None  # smoothed rate, files/s
        self._last_t: float | None = None
        self._last_n = 0

    def update(self, processed: int, now: float) -> None:
        """Feed with current state (processed files, timestamp)."""
        if self._last_t is None:  # first call: baseline only
            self._last_t, self._last_n = now, processed
            return
        dt = now - self._last_t
        dn = processed - self._last_n
        if dt <= 0 or dn <= 0:  # no measurable progress
            return
        inst = dn / dt  # instantaneous rate of this interval
        if self.rate is None:
            self.rate = inst  # seed with the first real measurement
        else:
            alpha = 1.0 - math.exp(-dt / self.tau)
            self.rate = alpha * inst + (1.0 - alpha) * self.rate
        self._last_t, self._last_n = now, processed

    def eta_sec(self, remaining: int) -> float | None:
        """Estimated remaining duration in seconds for ``remaining`` open files."""
        if not self.rate or self.rate <= 0 or remaining <= 0:
            return None
        return remaining / self.rate

    def sec_per_file(self) -> float | None:
        """Current average processing time per file (seconds)."""
        if not self.rate or self.rate <= 0:
            return None
        return 1.0 / self.rate


def _plausible_duration(duration: float, file_size: int) -> float:
    """Catch broken duration metadata.

    Typical case: a 400-byte FLV with duration = 2147483.647 s (= INT32_MAX
    in milliseconds, overflowed container field) — 596 h, distorting stats
    and capacity planning. The plausibility anchor is file size: under 256
    bytes per second of claimed duration (≈ 2 kbit/s) there can be no real
    recording — even minimally compressed mono Opus needs three times that.
    The threshold is deliberately chosen VERY conservatively so no
    legitimate file gets tripped; the overflow garbage is orders of
    magnitude below it anyway (400 B / 2,147,483 s ≈ 0.0002 B/s). Broken
    values are set to 0 (= duration unknown), not guessed.
    """
    if duration and duration > 0 and file_size > 0 and file_size < 256 * duration:
        return 0.0
    return duration or 0


def _mtime(fp: Path) -> float | None:
    """mtime of a file, None if it can't be read (one round trip on CIFS)."""
    try:
        return fp.stat().st_mtime
    except OSError:
        return None


def _transcript_beside(rel: str, fp: Path) -> bool:
    """Does a transcript JSON exist at the file's transcript location?"""
    from whisper_tm.core.helpers import transcript_path_for

    try:
        return transcript_path_for(rel, abs_path=fp).exists()
    except (OSError, ValueError):
        return False


def _probe(rel_str: str, fp: Path) -> dict | None:
    """FFprobe for one file → metadata dict for bulk_set_media_metadata."""
    try:
        st = fp.stat()
        mtime = st.st_mtime
    except OSError:
        return None
    info = get_media_info(fp)
    return {
        "rel_path": rel_str,
        "abs_path": paths.canon(fp),
        "duration_sec": _plausible_duration(info.get("duration", 0), st.st_size),
        "has_audio": info.get("has_audio", False),
        "has_video": info.get("has_video", False),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "audio_codec": info.get("audio_codec", ""),
        "video_codec": info.get("video_codec", ""),
        "bitrate": info.get("bitrate", 0),
        "media_title": info.get("title", ""),
        "indexed_at": time.time(),
        "index_mtime": mtime,
        "_ok": info.get("ok", False),
    }


def index_directory(
    vdir: Path | None = None,
    *,
    recursive: bool = True,
    db=None,
    force: bool = False,
    discover_transcripts: bool = True,
    workers: int = 8,
    batch_size: int = 100,
    should_stop=None,
):
    """Index a directory. Generator that yields progress dicts.

    Args:
        vdir: directory (default: config.media_dir)
        recursive: include subdirectories
        db: Database instance (required for saving)
        force: True = re-sample already indexed files too
        discover_transcripts: import JSON transcripts into the DB along the way
        workers: parallel FFprobe processes
        batch_size: DB write batch size
        should_stop: optional callable → True cleanly aborts

    Yields:
        {"done", "total", "indexed", "skipped", "failed", "current",
         "eta_sec", "rate_sec", "sec_per_file"}
        and finally {"complete": True, "stats": {...}}

        eta_sec       estimated remaining duration (s), adaptively smoothed
        rate_sec      current processing rate (files/s)
        sec_per_file  average processing time per file (s)
    """
    if vdir is None:
        vdir = config.media_dir
    vdir = Path(vdir)

    # 1./2. Collect files and load index state.
    #
    # Both run in a worker thread, while the loop below regularly delivers
    # progress. The reason isn't cosmetic: previously the FIRST event of
    # this generator only arrived once walk AND DB loading were finished.
    # On a library with hundreds of thousands of files that's minutes of radio
    # silence — the watchdog in the frontend (static/js/index.js, 60 s)
    # considered the stream dead, closed it and reported "timeout". The
    # server kept indexing invisibly. From the user's perspective, indexing
    # was simply broken.
    state: dict = {
        "files": None,
        "state": {},
        "found": 0,
        "folder": 0,
        "phase": "collect",
        "error": None,
    }

    def _walk_progress(p: dict) -> None:
        state["found"] = p.get("files", 0)
        state["folder"] = p.get("dirs_done", 0)

    def _prepare() -> None:
        try:
            found: list[tuple[str, Path]] = []
            for fp in collect_media_files(
                vdir,
                recursive=recursive,
                on_progress=_walk_progress,
                should_stop=should_stop,
            ):
                rel = paths.rel_from(fp, vdir)
                if rel is not None:
                    found.append((rel, fp))
            state["files"] = found
            if db and not force:
                state["phase"] = "db"
                try:
                    state["state"] = db.get_index_state_batch()
                except Exception as e:
                    logger.debug("Index state load failed: %s", e)
        except Exception as e:  # noqa — nothing may be lost in the thread
            state["error"] = e
            state["files"] = []

    preparation = threading.Thread(target=_prepare, name="index-collect", daemon=True)
    preparation.start()
    while True:
        # 1 s: often enough that the user sees the found files counting up
        # instead of staring at an apparently frozen window, and well below
        # any watchdog.
        preparation.join(timeout=1.0)
        if not preparation.is_alive():
            break
        # Sign of life: keeps the browser's watchdog calm and shows that
        # something is happening. total=0 means "not yet known" to the frontend.
        yield {
            "done": 0,
            "total": 0,
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "phase": state["phase"],
            "current": (
                f"{state['found']:,} files in {state['folder']:,} folders…".replace(",", ".")
                if state["phase"] == "collect"
                else "Loading index state…"
            ),
            "eta_sec": None,
            "rate_sec": None,
            "sec_per_file": None,
        }
    if state["error"] is not None:
        raise state["error"]

    file_list: list[tuple[str, Path]] = state["files"] or []
    total = len(file_list)
    state = state["state"]

    todo: list[tuple[str, Path]] = []
    skipped = 0
    # Look up via absolute path OR rel_path — this way a file already
    # captured while indexing the parent isn't probed again in the child.
    prevs = [] if force else [state.get(paths.canon(fp)) or state.get(rel) for rel, fp in file_list]
    # The mtime check is one network round trip per file on CIFS — in
    # parallel, like the scan's per-file work. Serially, an index run with
    # nothing to do spent minutes on the stat() calls alone.
    need_stat = [
        fp for (rel, fp), prev in zip(file_list, prevs, strict=False) if prev and prev[0] > 0
    ]  # indexed_at > 0
    mtimes: dict[Path, float | None] = {}
    if need_stat:
        with ThreadPoolExecutor(max_workers=max(1, workers) * 4) as stat_pool:
            mtimes = dict(zip(need_stat, stat_pool.map(_mtime, need_stat), strict=True))
    for i, (rel, fp) in enumerate(file_list):
        if force:
            todo.append((rel, fp))
            continue
        prev = prevs[i]
        if prev and prev[0] > 0:
            m = mtimes.get(fp)
            if m is None or abs(prev[1] - m) < 1.0:  # unreadable or unchanged
                skipped += 1
                continue
        todo.append((rel, fp))

    indexed = 0
    failed = 0
    done = skipped
    t0 = time.time()
    batch: list[dict] = []
    db_error: str | None = None

    # Seed the rate estimator at the start of probing (processed == 0): this
    # way the file-collecting/DB-state phase above does NOT count into the
    # processing rate.
    est = _RateEstimator()
    est.update(0, t0)

    yield {
        "done": done,
        "total": total,
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "current": "",
        "eta_sec": None,
        "rate_sec": None,
        "sec_per_file": None,
    }

    # 3. FFprobe in parallel, results written to the DB in batches
    def _flush():
        nonlocal batch, db_error
        if batch and db:
            try:
                db.bulk_set_media_metadata(batch)
            except Exception as e:
                # Do NOT swallow this: otherwise the indexer falsely reports
                # success while the metadata never lands in the DB (and the
                # "not indexed" popup keeps recurring after the scan).
                db_error = str(e)[:200]
                logger.warning("bulk_set_media_metadata failed: %s", e)
        batch = []

    if todo:
        # Submit tasks in waves instead of all (up to 50k) at once: this way
        # only ever one wave of futures is outstanding. On should_stop we
        # abort and discard the not-yet-started tasks (cancel_futures) —
        # otherwise the executor would, on shutdown with wait=True, wait for
        # ALL already submitted FFprobe calls and the index would be
        # practically unabortable. A normal run still processes ALL files.
        wave_size = max(1, workers) * 8
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        stopped = False
        try:
            for start in range(0, len(todo), wave_size):
                if should_stop and should_stop():
                    stopped = True
                    break
                wave = todo[start : start + wave_size]
                futures = {pool.submit(_probe, rel, fp): (rel, fp) for rel, fp in wave}
                for fut in as_completed(futures):
                    if should_stop and should_stop():
                        stopped = True
                        break
                    rel, fp = futures[fut]
                    meta = fut.result()
                    done += 1
                    if meta is None:
                        failed += 1
                    elif meta.pop("_ok", False):
                        indexed += 1
                        batch.append(meta)
                        if len(batch) >= batch_size:
                            _flush()
                    else:
                        # Do NOT persist a failed FFprobe as indexed
                        # (no indexed_at) — otherwise the file would count as
                        # indexed permanently and never be probed again
                        # (duration would stay 0). This way a later run
                        # retries it.
                        failed += 1

                    # report progress, throttled
                    if done % 15 == 0 or done == total:
                        processed = done - skipped
                        est.update(processed, time.time())
                        remaining = len(todo) - processed
                        eta = est.eta_sec(remaining)
                        spf = est.sec_per_file()
                        yield {
                            "done": done,
                            "total": total,
                            "indexed": indexed,
                            "skipped": skipped,
                            "failed": failed,
                            "current": Path(rel).name,
                            "eta_sec": round(eta, 1) if eta is not None else None,
                            "rate_sec": round(est.rate, 2) if est.rate else None,
                            "sec_per_file": round(spf, 3) if spf is not None else None,
                        }
                if stopped:
                    break
            _flush()
        finally:
            # On abort, cancel the still-waiting tasks so the index stops
            # immediately instead of waiting for all open FFprobe calls.
            pool.shutdown(cancel_futures=stopped)

    # 4. Transcript discovery (JSON → DB), so the DB stays the primary source
    healed = 0
    if discover_transcripts and db:
        from whisper_tm.scanner import _load_json_meta

        try:
            in_db = db.get_completion_set()
        except Exception:
            in_db = set()
        # Compare via identity: same-named files in different folders
        # would otherwise report each other as "already in DB"
        # (docs/PATHS.md).
        unknown = [(rel, fp) for rel, fp in file_list if paths.canon(fp) not in in_db]
        # Whether a JSON exists is one network round trip per file on CIFS —
        # checked in parallel, wave by wave (not all futures at once), so
        # stopping and the sign of life keep working.
        wave = 2000
        with ThreadPoolExecutor(max_workers=max(1, workers) * 4) as disc_pool:
            for start in range(0, len(unknown), wave):
                if should_stop and should_stop():
                    break
                part = unknown[start : start + wave]
                found = list(disc_pool.map(lambda it: _transcript_beside(*it), part))
                for (rel, fp), has_json in zip(part, found, strict=True):
                    tdata = _load_json_meta(rel, fp)[1] if has_json else None
                    if tdata is None:
                        continue
                    try:
                        # The actual location is known — pass it along
                        # instead of letting upsert_transcript reconstruct it
                        # from rel_path + root (docs/PATHS.md §Writing).
                        db.upsert_transcript(
                            rel, tdata, media_dir=str(vdir), abs_path=paths.canon(fp)
                        )
                        healed += 1
                    except Exception as e:
                        logger.debug("Transcript heal %s failed: %s", rel, e)
                # This loop hits the filesystem per file and runs over the
                # ENTIRE collection. Without a sign of life the stream goes
                # silent here a second time (see above) — on a NAS with
                # hundreds of thousands of files, considerably longer than the watchdog waits.
                done_n = min(start + wave, len(unknown))
                yield {
                    "done": total,
                    "total": total,
                    "indexed": indexed,
                    "skipped": skipped,
                    "failed": failed,
                    "phase": "transcripts",
                    "current": f"Searching transcripts… {done_n:,}/{len(unknown):,}".replace(
                        ",", "."
                    ),
                    "eta_sec": None,
                    "rate_sec": None,
                    "sec_per_file": None,
                }

    stats = {}
    if db:
        try:
            stats = db.get_index_stats()
        except Exception as e:
            logger.debug("get_index_stats failed: %s", e)

    yield {
        "complete": True,
        "total": total,
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "healed_transcripts": healed,
        "db_error": db_error,
        "elapsed_sec": round(time.time() - t0, 1),
        "stats": stats,
    }

"""
Media routes: video streaming, clip extraction, uploads, downloads.
With path traversal protection on all endpoints.
"""

import logging
import os
import shutil
import threading
import time
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file
from werkzeug.utils import secure_filename

from whisper_tm.config import MEDIA_EXTENSIONS, config
from whisper_tm.core import media_formats, paths
from whisper_tm.core.audio_utils import run_cancellable
from whisper_tm.core.helpers import format_size, parse_bool
from whisper_tm.core.security import check_root, safe_path

logger = logging.getLogger(__name__)


media_bp = Blueprint("media", __name__)

#: Upper limit for links per download job. Not against abuse — the
#: dialog is local —, but against a slipped clipboard: whoever
#: accidentally pastes an entire text file should be told so and
#: not spend hours downloading something they never wanted. Playlists count
#: as ONE link; their videos are not affected by this.
MAX_YT_LINKS = 50

# Serializes chunk upload completion (assembly) across requests
_upload_lock = threading.Lock()
# Final upload names chosen but not yet in place (see _reserve_upload_name)
_upload_reserved: set[str] = set()


def upload_name(raw: str | None) -> str:
    """The name an upload is saved under (:func:`paths.file_name`)."""
    return paths.file_name(raw)


def _audio_or_video(path: Path) -> bool:
    """Is it a recording? (:func:`media_formats.is_recording`)"""
    return media_formats.is_recording(path)


def remove_stale_chunks(max_age_h: float = 24.0) -> int:
    """At start: the chunk folders of uploads that never finished."""
    folder = Path(config.dirs["chunks"])
    cutoff = time.time() - max_age_h * 3600
    gone = 0
    try:
        entries = list(folder.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                gone += 1
        except OSError:
            continue
    if gone:
        logger.info("Removed %d unfinished uploads' chunks", gone)
    return gone


def _reserve_upload_name(dest_dir: Path, fname: str) -> Path:
    """A free name for a finished upload, held until the file is in place.

    Chosen under the lock and remembered: two uploads of the same name
    finishing at once used to both pick ``talk.mp3`` and write into the
    same file — one upload lost, both reported complete.
    """
    stem, ext = Path(fname).stem, Path(fname).suffix
    with _upload_lock:
        candidate, n = dest_dir / fname, 1
        while candidate.exists() or str(candidate) in _upload_reserved:
            candidate = dest_dir / f"{stem}_{n}{ext}"
            n += 1
        _upload_reserved.add(str(candidate))
        return candidate


def _release_upload_name(path: Path) -> None:
    with _upload_lock:
        _upload_reserved.discard(str(path))


# Serializes identical playback transcodes (per cache key), so parallel
# requests don't run the same transcode twice (and collide)
_transcode_locks_guard = threading.Lock()
_transcode_locks = {}

# A transcode that failed (or ran out of time) is not tried again on every
# click: each range request of the player used to start the same doomed
# ffmpeg run over a multi-hour recording. Cache key → when.
_transcode_failed: dict[str, float] = {}
TRANSCODE_RETRY_SEC = 600


def _transcode_timeout(duration: float) -> int:
    """How long a playback transcode may take: two minutes, more for a long
    recording (libmp3lame manages roughly 50x real time; a 10-h recording
    needs more than 120 s). Bounded at three hours."""
    try:
        d = float(duration or 0)
    except (TypeError, ValueError):
        d = 0.0
    return int(min(3 * 3600, max(120, d / 20)))


# ── Path resolution, briefly cached ──────────────────────────────────────
#
# Clicking a point in the transcript could take very long to load.
# Measured with /api/debug/timings over many jumps in one recording, most
# of the time went into the file stat on the NAS, not into send_file.
#
# Every click in the transcript is a new range request, and each one paid
# for the same resolution again: `safe_path` calls `.resolve()` TWICE, and
# that goes over CIFS directory by directory. Plus an `exists()`.
# While the batch run reads from the same share, that's seconds.
#
# So the RESULT of the resolution is cached, briefly. From a security
# standpoint this is harmless: the key is the full input (root +
# requested path), same input means same verdict — the check
# is not skipped, only not repeated.
#
# 30 seconds, because a typical CIFS share is mounted with `actimeo=30`:
# caching longer than the mount itself would be its own kind of falsehood.
_path_cache: dict[tuple[str, str], tuple[Path | None, float]] = {}
_path_cache_lock = threading.Lock()
PATH_CACHE_TTL = 30.0
PATH_CACHE_MAX = 256


def _resolved(filename: str):
    """``(path, exists)`` for a playback request — refreshed at most every 30 s.

    Returns ``(None, False)`` if the file is not there; ``safe_path``
    still aborts on its own with 403 if a traversal attempt is made,
    and it does so BEFORE anything reaches memory.

    The negative result is cached too: it cost as much as the
    positive one, and a file that has just appeared can be
    picked up within 30 seconds at the latest.
    """
    key = (str(config.media_dir), filename)
    now = time.time()
    with _path_cache_lock:
        entry = _path_cache.get(key)
        if entry and now - entry[1] < PATH_CACHE_TTL:
            return entry[0], entry[0] is not None

    fp = safe_path(config.media_dir, filename)  # can trigger 403/400
    present = fp.exists()

    with _path_cache_lock:
        # Simple cap: on overflow, evict the oldest entry.
        # No LRU — the list is small, and whoever is currently watching is
        # the most recent entry anyway.
        if len(_path_cache) >= PATH_CACHE_MAX:
            oldest = min(_path_cache, key=lambda k: _path_cache[k][1])
            _path_cache.pop(oldest, None)
        _path_cache[key] = (fp if present else None, now)
    return (fp if present else None), present


def clear_path_cache() -> None:
    """Forget everything — for tests and after a root change."""
    with _path_cache_lock:
        _path_cache.clear()


def _prune_playback_cache(cache_dir, max_files=300, max_bytes=2 * 1024**3):
    """Caps the playback cache (LRU by mtime) against unbounded growth.

    Removes the oldest cache files once file count or total size
    exceeds the limit. Temp files (*.tmp) are left untouched.

    Covers BOTH kinds: transcoded .mp3 and repackaged .m4a. If the
    cut only ran over .mp3, the M4A repackagings would grow without bound.
    """
    entries = []
    total = 0
    try:
        for p in [*cache_dir.glob("*.mp3"), *cache_dir.glob("*.m4a")]:
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
    except OSError:
        return
    entries.sort(key=lambda e: e[0])  # oldest first (LRU)
    i = 0
    while i < len(entries) and (len(entries) - i > max_files or total > max_bytes):
        _, size, victim = entries[i]
        try:
            victim.unlink()
            total -= size
        except OSError:
            pass
        i += 1


# Container formats that are just a bare elementary stream: no index, no
# duration in the header. ffprobe decodes and gets the correct length, the
# browser ESTIMATES it from the bitrate and gets it wrong — a voice note
# of a few minutes (.aac, ADTS) can report over an hour in the player, and seeking
# was broken. Such files therefore go through the
# transcode cache even though their codec would be browser-compatible.
def _repack(fp, cache_dir, cache_key):
    """Repackage a raw ADTS stream losslessly into an M4A container.

    Do NOT re-encode. Repackaging is several times faster than an MP3
    transcode and yields a smaller file. But a third factor
    is decisive: libmp3lame shifts the audio back by ~69 ms
    (encoder delay). In a transcription app this makes subtitles
    and audio drift apart — repackaging has start_time 0 by construction.

    ``-bsf:a aac_adtstoasc`` builds the container's AudioSpecificConfig
    from the ADTS headers (ffmpeg-bitstream-filters:
    "required … when copying an AAC stream from a raw ADTS AAC … to MOV/MP4").
    ``+faststart`` moves the moov atom to the front, otherwise the browser
    needs a second range request to the end of the file before the first sound.

    Returns the path — or None if the result is not correct.
    """
    target = cache_dir / f"{cache_key}.m4a"
    if target.exists():
        return target
    tmp = cache_dir / f"{cache_key}.m4a.tmp"
    try:
        # "-f mp4" MUST be set: we write to *.m4a.tmp, and
        # ffmpeg otherwise infers the output format from the extension — ".tmp"
        # it doesn't recognize and aborts.
        run_cancellable(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                *media_formats.source(fp),
                "-c:a",
                "copy",
                "-bsf:a",
                "aac_adtstoasc",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(tmp),
            ],
            job=None,
            timeout=120,
        )
        if not tmp.exists() or tmp.stat().st_size < 512:
            raise RuntimeError("Repackaging produced no usable file")

        # Sanity check. With implicitly signaled SBR (HE-AAC), the
        # bitstream filter can lose the doubling — then the M4A reports
        # half the length. Better to transcode than to deliver a wrong timeline.
        from whisper_tm.core.audio_utils import duration_from_frames, get_duration

        expected = duration_from_frames(fp)
        is_it = get_duration(tmp)
        if expected and is_it and abs(expected - is_it) > max(1.0, expected * 0.01):
            raise RuntimeError(
                f"Duration after repackaging differs ({is_it:.1f}s instead of {expected:.1f}s)"
            )
        tmp.replace(target)
        return target
    except Exception as e:
        logger.warning(f"Repackaging failed ({fp.name}): {e} — falling back to MP3 transcode")
        tmp.unlink(missing_ok=True)
        return None


@media_bp.route("/api/video/<path:filename>")
def serve_video(filename):
    """Serve a media file for browser playback.

    Automatically detects browser-incompatible codecs (AMR, GSM, PCM etc.)
    and transcodes to MP3/AAC as needed. Transcodes are cached.
    """
    # Measurement points per step: audio playback could take very long to
    # load, and every step needs to be timed.
    # Otherwise only the total request duration would be shown, and that
    # doesn't say whether the NAS was slow, ffprobe hung, or repackaging happened.
    from whisper_tm.core import debug as _dbg

    mp = _dbg.MeasurePoints(f"Playback {Path(filename).name}")

    fp, present = _resolved(filename)
    # Own measurement point, because this one call goes over CIFS: with a
    # cold connection it's the most expensive step of the whole endpoint — and
    # when jumping around in the transcript, every click would pay for it again;
    # hence it is cached for 30 s (see `_resolved`).
    mp.point("File stat (NAS)")
    if not present:
        abort(404)

    # Browser-compatible codecs — everything else gets transcoded
    BROWSER_AUDIO_CODECS = {
        "mp3",
        "aac",
        "vorbis",
        "opus",
        "flac",
        "pcm_f32le",
    }

    is_video_file = fp.suffix.lower() in {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".ts",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
        ".ogv",
        ".m2ts",
    }

    # Serve video files directly (browser handles codec support)
    if is_video_file:
        answer = send_file(str(fp), conditional=True)
        mp.point("send_file")
        mp.done("Video, unchanged")
        return answer

    # Audio files: check codec AND container
    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name:format=format_name",
            "-of",
            "csv=p=0",
            *media_formats.probe(fp),
        ]
        # Here too `errors="replace"` instead of strict: the `except` below
        # does catch the codec error, but at the cost that codec detection
        # fails for exactly the files whose metadata isn't UTF-8 —
        # and nobody sees why.
        r = run_cancellable(probe_cmd, job=None, timeout=10, label="Codec sample")
        lines = (
            [z.strip().lower() for z in r.stdout.splitlines() if z.strip()]
            if r.returncode == 0
            else []
        )
        codec = lines[0] if lines else ""
        container = lines[1] if len(lines) > 1 else ""
    except Exception:
        codec = container = ""
    mp.point("ffprobe (codec)")

    from whisper_tm.core.audio_utils import is_raw_stream

    raw = is_raw_stream(container)

    # Known browser-compatible codec → serve directly, BUT only in
    # a container that provides duration info.
    # Plain PCM in a WAV is the browsers' oldest format — transcoding every
    # .wav cost time and, for long ones, never finished
    plain_wav = codec == "pcm_s16le" and container == "wav"
    if (codec in BROWSER_AUDIO_CODECS or plain_wav) and not raw:
        answer = send_file(str(fp), conditional=True)
        mp.point("send_file")
        mp.done(f"served directly ({codec})")
        return answer

    cache_dir = config.work_dir.parent / "playback_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key: filename + modification date + size
    import hashlib

    stat = fp.stat()
    cache_key = hashlib.md5(f"{fp}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()[:16]

    # Raw stream with browser-friendly codec: repackaging is enough, re-encoding
    # would be a loss and a time cost for nothing (see _repack).
    if raw and codec in BROWSER_AUDIO_CODECS:
        with _transcode_locks_guard:
            key_lock = _transcode_locks.setdefault(cache_key, threading.Lock())
        # Waiting for the lock gets its OWN measurement point. It's the
        # spot where minutes can invisibly disappear: a second
        # request for the same file stalls here while the first one repackages —
        # in the total duration it looks as if repackaging is slow.
        with key_lock:
            mp.point("waited on lock")
            m4a = _repack(fp, cache_dir, cache_key)
            mp.point("repackaging")
        if m4a:
            _prune_playback_cache(cache_dir)
            answer = send_file(str(m4a), mimetype="audio/mp4", conditional=True)
            mp.point("send_file")
            mp.done("repackaged (raw stream)")
            return answer
        # otherwise: transcode below

    # Unknown/incompatible codec → transcode to MP3 (cached)
    if codec:
        logger.info(f"Playback: {fp.name} has codec '{codec}' — transcoding to MP3")

    cached_mp3 = cache_dir / f"{cache_key}.mp3"

    failed_at = _transcode_failed.get(cache_key)
    if not cached_mp3.exists() and failed_at and time.time() - failed_at < TRANSCODE_RETRY_SEC:
        answer = send_file(str(fp), conditional=True)
        mp.point("send_file")
        mp.done("Transcode failed recently, original")
        return answer

    if not cached_mp3.exists():
        # Lock per cache key: prevents parallel requests from running the
        # same transcode twice (and colliding)
        with _transcode_locks_guard:
            key_lock = _transcode_locks.get(cache_key)
            if key_lock is None:
                key_lock = threading.Lock()
                _transcode_locks[cache_key] = key_lock
        with key_lock:
            # Double check: another request may already be done
            if not cached_mp3.exists():
                # Transcode into temp + atomic rename, so an
                # aborted or parallel transcode doesn't leave a truncated
                # cache file that gets treated as valid
                tmp_mp3 = cache_dir / f"{cache_key}.mp3.tmp"
                try:
                    from whisper_tm.core.audio_utils import get_duration_safe

                    r = run_cancellable(
                        [
                            "ffmpeg",
                            "-y",
                            *media_formats.source(fp),
                            "-vn",
                            "-codec:a",
                            "libmp3lame",
                            "-q:a",
                            "2",
                            "-ar",
                            "44100",
                            "-ac",
                            "2",
                            "-f",
                            "mp3",
                            str(tmp_mp3),
                        ],
                        job=None,
                        timeout=_transcode_timeout(get_duration_safe(fp)),
                        label="Playback transcode",
                    )
                    # The exit code decides, not the file: a run that broke
                    # off (full disk) leaves a partial MP3 that would be
                    # served, truncated, until the recording changes
                    if r.returncode != 0:
                        raise RuntimeError(
                            f"ffmpeg code {r.returncode}: {(r.stderr or '').strip()[-200:]}"
                        )
                    tmp_mp3.replace(cached_mp3)
                    _transcode_failed.pop(cache_key, None)
                    _prune_playback_cache(cache_dir)
                except Exception as e:
                    logger.warning(f"Playback transcode failed: {e}")
                    tmp_mp3.unlink(missing_ok=True)
                    _transcode_failed[cache_key] = time.time()
                    # Fallback: serve original directly
                    answer = send_file(str(fp), conditional=True)
                    mp.point("Transcode failed, original")
                    mp.done("Fallback to original")
                    return answer

    if cached_mp3.exists() and cached_mp3.stat().st_size > 1000:
        try:
            cached_mp3.touch()  # LRU: mark last use
        except OSError:
            pass
        answer = send_file(str(cached_mp3), mimetype="audio/mpeg", conditional=True)
        mp.point("send_file")
        mp.done("from playback cache")
        return answer

    # Fallback
    answer = send_file(str(fp), conditional=True)
    mp.point("send_file")
    mp.done("Fallback: original")
    return answer


@media_bp.route("/api/download/<path:filename>")
def download_original(filename):
    """Download a media file as-is (path relative to the open media folder)."""
    fp = safe_path(config.media_dir, filename)
    if not fp.exists():
        abort(404)
    return send_file(str(fp), as_attachment=True, download_name=Path(filename).name)


def upload_chunk_bytes() -> int:
    """One piece of a web upload: ``server.upload_chunk_mb``, between 1 MB
    and what a request may carry (``server.max_upload_mb``)."""

    def number(key, default):
        try:
            return float(config.get("server", key, default=default))
        except (TypeError, ValueError):
            return float(default)

    mb = min(max(number("upload_chunk_mb", 32), 1.0), max(number("max_upload_mb", 200) - 1, 1.0))
    return int(mb * 1024 * 1024)


@media_bp.route("/api/upload_limits")
def upload_limits():
    """What the upload page needs to know before it sends the first piece."""
    try:
        cap_gb = float(config.get("server", "max_upload_gb", default=100) or 0)
    except (TypeError, ValueError):
        cap_gb = 100.0
    return jsonify({"chunk_bytes": upload_chunk_bytes(), "max_file_gb": cap_gb})


@media_bp.route("/api/upload_chunk", methods=["POST"])
def upload_chunk():
    """Chunked upload: collect chunks, assemble on the last chunk.

    The final file is placed in media_dir/uploads/ (in the user's media
    directory), so the scanner finds it and it can be transcribed.
    """
    file_id = request.form.get("fileId")
    ci = request.form.get("chunkIndex", type=int)
    tc = request.form.get("totalChunks", type=int)
    if ci is None:
        ci = 0
    if tc is None:
        tc = 1
    fname = upload_name(request.form.get("fileName", "file.mp4"))
    # Audio and video only — the built-in list, not the settings' extra
    # extensions: the upload folder can be pointed anywhere the app may
    # write, and a .pth, .desktop or authorized_keys put there runs code
    if Path(fname).suffix.lower() not in MEDIA_EXTENSIONS:
        return jsonify(
            {
                "error": f"Only audio and video files can be uploaded, "
                f"not {Path(fname).suffix or 'files without an extension'}"
            }
        ), 400
    chunk = request.files.get("chunk")
    if not chunk or not file_id:
        return jsonify({"error": "Missing params"}), 400
    if ci < 0 or tc < 1 or ci >= tc:
        return jsonify({"error": "Invalid chunkIndex/totalChunks"}), 400

    # Sanitize file_id to prevent path tricks
    file_id = secure_filename(file_id)
    if not file_id:
        return jsonify({"error": "Invalid fileId"}), 400

    chunks_dir = config.dirs["chunks"]
    ud = chunks_dir / file_id
    data = chunk.read()
    # Each chunk is capped by the request size; the whole file was not — a
    # client could send chunks until the disk was full
    cap_gb = float(config.get("server", "max_upload_gb", default=100) or 0)
    try:
        so_far = sum(p.stat().st_size for p in ud.glob("chunk_*")) if ud.exists() else 0
        free = shutil.disk_usage(chunks_dir if chunks_dir.exists() else chunks_dir.parent).free
    except OSError:
        so_far, free = 0, None
    if cap_gb > 0 and so_far + len(data) > cap_gb * 1e9:
        shutil.rmtree(ud, ignore_errors=True)
        return jsonify(
            {
                "error": f"The upload is larger than {cap_gb:g} GB "
                f"(server.max_upload_gb in config.yaml)"
            }
        ), 413
    if free is not None and free < len(data) + 1e9:
        shutil.rmtree(ud, ignore_errors=True)
        return jsonify(
            {"error": f"The disk of {chunks_dir} is nearly full — the upload would fill it"}
        ), 507
    try:
        ud.mkdir(parents=True, exist_ok=True)
        with (ud / f"chunk_{ci:06d}").open("wb") as cf:
            cf.write(data)
    except OSError as e:
        return jsonify(
            {"error": f"The upload's work folder {chunks_dir} can't be written: {e.strerror or e}"}
        ), 507 if e.errno == 28 else 500

    # Completion: check EXACTLY whether all indices 0..tc-1 are present — a
    # count comparison (len(glob) >= tc) misses missing chunks when
    # indices are sent twice, and assembles twice on parallel final
    # chunks. Lock serializes assembly per file_id.
    with _upload_lock:
        have = {p.name for p in ud.glob("chunk_*")} if ud.exists() else set()
        complete = all(f"chunk_{i:06d}" in have for i in range(tc))
        received = len(have)
        if complete:
            # "Claim" the chunks immediately (rename the directory), so
            # a parallel request doesn't assemble as well
            claimed = ud.with_name(file_id + ".assembling")
            try:
                ud.rename(claimed)
                ud = claimed
            except OSError:
                # Another request has already claimed it
                return jsonify(
                    {"status": "uploading", "chunksReceived": received, "totalChunks": tc}
                )

    if complete:
        # THE configured upload directory (Settings → Folders),
        # never a self-built path. Default: uploads/ in the media directory.
        dest_dir = config.upload_dir
        failure, code = "", 500
        final_path = final_name = None
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Deduplication: if file exists, append a suffix — reserved, so a
            # parallel upload of the same name gets another one
            final_path = _reserve_upload_name(dest_dir, fname)
            final_name = final_path.name
        except OSError as e:
            failure = f"The upload folder {dest_dir} can't be written: {e.strerror or e}"
        if final_path is not None:
            # Assemble under a hidden name and rename once complete: a scan
            # must never see (and queue) a half-assembled recording
            part = final_path.with_name(f".{final_name}.part")
            try:
                with open(part, "wb") as out:
                    for i in range(tc):
                        cp = ud / f"chunk_{i:06d}"
                        if cp.exists():
                            out.write(cp.read_bytes())
                # The extension said audio or video; the content must too —
                # a key or a config renamed .mp4 would otherwise lie in the
                # library, which the app serves
                if not _audio_or_video(part):
                    failure, code = (
                        (
                            f"Only audio and video files can be uploaded — "
                            f"{final_name} holds neither"
                        ),
                        400,
                    )
                else:
                    # The name is reserved against our own uploads; a rename,
                    # not a hard link — the upload folder is usually a CIFS share
                    os.replace(part, final_path)
            except OSError as e:
                logger.warning("Upload assembly failed (%s): %s", final_name, e)
                failure = f"The upload could not be put together in {dest_dir}: {e.strerror or e}"
            finally:
                part.unlink(missing_ok=True)
                _release_upload_name(final_path)
        # The chunks go whatever happened — a failed upload left them behind
        shutil.rmtree(ud, ignore_errors=True)
        if failure:
            return jsonify({"error": failure}), code

        # Register in the DB immediately: the file shows up without a full
        # walk in the (paginated) list — with large libraries a complete
        # NAS scan used to run for a SINGLE new file otherwise. Returns the
        # rel_path relative to the library (None if the upload directory
        # is configured outside the media directory).
        rel_path = None
        try:
            from whisper_tm.scanner import register_single_file

            rel_path = register_single_file(final_path)
        except Exception as e:  # noqa — display convenience, not required
            logger.debug("DB registration: %s", e)

        return jsonify(
            {
                "status": "complete",
                "filename": rel_path,  # None = outside the library
                "display_name": final_name,
                "directory": str(dest_dir),
            }
        )

    return jsonify({"status": "uploading", "chunksReceived": received, "totalChunks": tc})


#: Containers a re-encoded clip (H.264 + AAC) can stay in — a .webm, .ts …
#: becomes .mp4
_CLIP_VIDEO_EXT = {".mp4", ".m4v", ".mov", ".mkv"}


def _clip_ts(sec: float) -> str:
    """``0-00-03.2`` — the clip's name says where it was cut, to the tenth of
    a second the editor shows (whole seconds made two different cuts look
    alike, and the second one replaced the first)."""
    tenths = int(round(sec * 10))
    h, rest = divmod(tenths, 36000)
    m, rest = divmod(rest, 600)
    s, t = divmod(rest, 10)
    return f"{h}-{m:02d}-{s:02d}.{t}"


@media_bp.route("/api/clip", methods=["POST"])
def extract_clip():
    """Cut a clip from a video file — exactly the range chosen.

    A stream copy can only begin at a keyframe: ffmpeg goes back to the one
    before the start — in a YouTube video they lie seconds apart — and the
    clip would begin up to that much earlier, not at the range selected. A video is
    therefore re-encoded: decoded from the keyframe before the start,
    written from the start itself, to the frame. Sound alone stays a copy —
    it is cut at the packet, a few hundredths of a second.
    """
    from whisper_tm.core.audio_utils import get_media_info

    data = request.json if isinstance(request.json, dict) else {}
    filename = data.get("filename")
    try:
        start = float(data.get("start") or 0)
        end = float(data.get("end") or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid parameters"}), 400

    if not filename or end <= start:
        return jsonify({"error": "Invalid parameters"}), 400

    source = safe_path(config.media_dir, filename)
    if not source.exists():
        return jsonify({"error": "File not found"}), 404

    duration = end - start
    if duration > 3600:
        return jsonify({"error": "Clip too long (max 1h)"}), 400

    clips_dir = config.dirs["clips"]
    try:
        clips_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return jsonify(
            {"error": f"The clips folder {clips_dir} can't be written: {e.strerror or e}"}
        ), 500
    video = get_media_info(source).get("has_video", False)
    stem = Path(filename).stem
    ext = Path(filename).suffix or ".mp4"
    if video and ext.lower() not in _CLIP_VIDEO_EXT:
        ext = ".mp4"
    out_name = f"{stem}_clip_{_clip_ts(start)}_to_{_clip_ts(end)}{ext}"
    out_path = clips_dir / out_name

    # Unique temp name + atomic rename: prevents parallel
    # identical requests with -y from writing the same target file at once
    import uuid

    tmp_path = clips_dir / f"{out_path.stem}.{uuid.uuid4().hex[:8]}.part{ext}"

    try:
        head = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            *media_formats.source(source),
            "-t",
            f"{duration:.3f}",
        ]
        if video:
            # The picture without cover art (V), the first sound track
            attempts = [
                (
                    head
                    + [
                        "-map",
                        "0:V:0",
                        "-map",
                        "0:a:0?",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                    ]
                    + (["-movflags", "+faststart"] if ext.lower() != ".mkv" else [])
                    + [str(tmp_path)],
                    "Clip",
                )
            ]
        else:
            attempts = [
                (head + ["-c", "copy", "-avoid_negative_ts", "make_zero", str(tmp_path)], "Clip"),
                # A copy the container doesn't take: encoded as it likes
                (head + ["-vn", str(tmp_path)], "Clip (re-encoded)"),
            ]
        # VIA run_cancellable, not via subprocess.run(text=True).
        #
        # Cutting a clip could fail with "'utf-8' codec can't decode
        # byte 0xff" although the cut itself was fine — `text=True`
        # decodes ffmpeg's stderr STRICTLY as UTF-8, and a file can carry
        # a byte in its metadata that doesn't fit there. So the
        # exception was raised while READING THE MESSAGE, and the user saw a
        # codec error where there wasn't one.
        #
        # `run_cancellable` decodes with `errors="replace"` — and according to
        # its own header, ALL external tools run through it anyway. The
        # clip cut was the last spot that bypassed that.
        r = None
        for cmd, label in attempts:
            # Encoding runs at many times real time — an hour is minutes
            r = run_cancellable(cmd, job=None, timeout=int(600 + duration * 2), label=label)
            if r.returncode == 0:
                break
        if r.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            return jsonify({"error": f"FFmpeg: {r.stderr[-300:]}"}), 500

        if not tmp_path.exists():
            return jsonify({"error": "Clip creation failed"}), 500

        # Atomically rename to the final name (result path for the client)
        tmp_path.replace(out_path)

        return jsonify(
            {
                "ok": True,
                "filename": out_name,
                "size": format_size(out_path.stat().st_size),
            }
        )

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


@media_bp.route("/api/clip/waveform")
def clip_waveform():
    """The loudness of a stretch, for the clip editor's timeline.

    Query: ``file`` (relative to the open folder, or a full path in the
    allowed roots), ``start`` and ``end`` in seconds, ``n`` buckets — the
    timeline's width in pixels. Never waits for a long recording: what isn't
    decoded yet comes back as null with ``pending`` (core/waveform.py).
    """
    from whisper_tm.core import waveform

    filename = request.args.get("file") or request.args.get("filename") or ""
    try:
        start = float(request.args.get("start") or 0)
        end = float(request.args.get("end") or 0)
        n = int(request.args.get("n") or 1000)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid parameters"}), 400
    if not filename or end <= start or start < 0:
        return jsonify({"error": "Invalid parameters"}), 400
    if paths.is_absolute(filename):
        source = check_root(filename, field_name="file")
    else:
        source = safe_path(config.media_dir, filename)
    if not source.exists():
        return jsonify({"error": "File not found"}), 404
    try:
        return jsonify(waveform.peaks(source, start, end, n))
    except waveform.WaveformError as e:
        return jsonify({"error": str(e)}), 422


#: Upper limit for the remote listening sample. Unlike cutting in the
#: editor (1 h), this is about a supporting excerpt that someone — or a
#: language model — listens to in passing. Ten minutes is generous for that,
#: and the limit protects against a typo in `end` triggering an
#: hour-long re-encode.
MAX_SAMPLE_SEC = 600
#: A sample is brought to speech loudness: quiet recordings can lie
#: 20–30 dB below it (a player that seems silent). Silence stays silence. −18 LUFS with
#: a true peak of −4 dB: the MP3 encoding overshoots by ~3 dB, and −16/−1.5
#: reached full scale after it.
SAMPLE_LOUDNESS = "loudnorm=I=-18:TP=-4:LRA=11"


def _only_recording_named(rel: str) -> str:
    """The full path of the one transcribed recording whose path relative to
    its root is ``rel`` — empty when none or several are."""
    from whisper_tm.extensions import db

    if not db or not rel:
        return ""
    try:
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT abs_path FROM transcripts WHERE rel_path = ? "
                "AND abs_path IS NOT NULL AND abs_path != '' LIMIT 2",
                (rel,),
            ).fetchall()
    except Exception as e:  # noqa: BLE001 — no player is worth a 500
        logger.debug("clip lookup %s: %s", rel, e)
        return ""
    return rows[0]["abs_path"] if len(rows) == 1 else ""


@media_bp.route("/api/clip/audio")
def clip_audio():
    """An audio excerpt as mp3 — via GET, so that an `<audio>` element plays it.

    **Why `/api/clip` doesn't handle this.** That one cuts via POST and
    returns a filename; an `<audio src=…>` in the browser can only do GET.
    That's exactly what the MCP path needs though: the language model writes a
    line in its response for each result, and Open WebUI turns that into a
    player (`HTMLToken.svelte` — the address is in the content of the
    tag there, not in `src`).

    **Cutting happens on demand, not in advance.** A result list quickly has
    thirty hits; pre-cutting all of them would mean thirty ffmpeg calls
    for excerpts of which the user listens to two. Only clicking the
    player triggers the cut. What has been cut once stays in the clip
    folder and is served directly on the second lists.

    **Audio only.** `-vn`, and as mp3 at 96 kbit/s: 30 seconds become
    360 kB instead of the 40 MB the same excerpt weighs as video.
    """
    filename = request.args.get("file") or request.args.get("filename") or ""
    try:
        start = float(request.args.get("start") or 0)
        end = float(request.args.get("end") or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid parameters"}), 400

    if not filename or end <= start:
        return jsonify({"error": "Invalid parameters"}), 400
    if end - start > MAX_SAMPLE_SEC:
        return jsonify({"error": f"Excerpt too long (max {MAX_SAMPLE_SEC // 60} min)"}), 400

    # A full path names a recording outside the open folder — a search hit
    # of another root, a passage of the chat; it must lie in the
    # allowed roots
    if paths.is_absolute(filename):
        source = check_root(filename, field_name="file")
    else:
        source = safe_path(config.media_dir, filename)
        if not source.exists():
            # A player written before such hits were named by their full
            # path, relative to another root: the recording the name
            # belongs to — when exactly one does
            elsewhere = _only_recording_named(filename)
            if elsewhere:
                source = check_root(elsewhere, field_name="file")
    if not source.exists():
        return jsonify({"error": "File not found"}), 404

    # The name carries the cut within itself: same file, same seconds →
    # same name → no ffmpeg the second time. The short hash keeps it
    # short and filesystem-friendly, even if the path contains characters
    # a filename doesn't like.
    import hashlib

    key = hashlib.sha1(f"{filename}|{start:.3f}|{end:.3f}|{SAMPLE_LOUDNESS}".encode()).hexdigest()[
        :12
    ]
    target = config.dirs["clips"] / f"sample_{Path(filename).stem[:40]}_{key}.mp3"

    if not target.exists():
        import uuid

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return jsonify(
                {"error": f"The clips folder {target.parent} can't be written: {e.strerror or e}"}
            ), 500
        tmp = target.with_suffix(f".{uuid.uuid4().hex[:8]}.part.mp3")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",  # BEFORE -i: container seek
            *media_formats.source(source),
            "-t",
            f"{end - start:.3f}",
            "-vn",
            "-af",
            SAMPLE_LOUDNESS,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "96k",
            "-f",
            "mp3",  # required: the extension is .part.mp3
            str(tmp),
        ]
        r = run_cancellable(cmd, job=None, timeout=300, label="Listening sample")
        if r.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return jsonify({"error": f"FFmpeg: {r.stderr[-300:]}"}), 500
        tmp.replace(target)
        trim_samples(target.parent)

    # `as_attachment=False`: an attachment ends up in the download folder, a
    # player should play it.
    return send_file(str(target), mimetype="audio/mpeg", as_attachment=False, conditional=True)


#: Listening samples (sample_*.mp3 in the clips folder) older than this go
SAMPLE_KEEP_DAYS = 7
#: … and the oldest go while they take more than this
SAMPLE_MAX_MB = 500


def trim_samples(folder: Path) -> int:
    """Remove old listening samples. They are cut for every new passage an
    MCP client or a page asks about and nothing else removed them — normal
    use grew the folder forever, and a loop over start values filled the
    disk. Only ``sample_*.mp3`` at the top of the folder: the clips you made
    are never touched. Returns how many went."""
    now = time.time()
    try:
        files = sorted(
            (
                (f.stat().st_mtime, f.stat().st_size, f)
                for f in Path(folder).glob("sample_*.mp3")
                if f.is_file()
            ),
            key=lambda t: t[0],
        )
    except OSError:
        return 0
    total = sum(size for _, size, _ in files)
    gone = 0
    for mtime, size, f in files:
        if now - mtime < SAMPLE_KEEP_DAYS * 86400 and total <= SAMPLE_MAX_MB * 1024 * 1024:
            break
        try:
            f.unlink()
            total -= size
            gone += 1
        except OSError:
            continue
    return gone


@media_bp.route("/api/clip/download/<path:filename>")
def download_clip(filename):
    """Download a cut clip (path relative to the clips directory)."""
    fp = safe_path(config.dirs["clips"], filename)
    if not fp.exists():
        abort(404)
    return send_file(str(fp), as_attachment=True, download_name=fp.name)


# ══════════════════════════════════════════════════════════════════════════
#  YouTube Download (yt-dlp)
# ══════════════════════════════════════════════════════════════════════════


@media_bp.route("/api/youtube/download", methods=["POST"])
def youtube_download():
    """Start a yt-dlp download: one link, several links, OR playlists.

    Runs as a background job (core/ytdl.py): the response contains only the
    job ID, progress comes over the existing SSE channel
    /api/status/<job_id>, cancellation via /api/cancel/<job_id>. All files
    end up in media_dir/uploads/ (that's where the scanner finds them).
    """
    from whisper_tm.extensions import hw

    if not (hw and hw.has_ytdlp):
        return jsonify({"error": "yt-dlp not installed. Install with: pip install yt-dlp"}), 400

    data = request.json or {}
    # One link or several. `url` remains as the singular form — it's used
    # in scripts and in the MCP server; `urls` is the plural from the dialog.
    raw = data.get("urls")
    if raw is None:
        raw = [data.get("url") or ""]
    elif not isinstance(raw, list):
        return jsonify({"error": "urls must be a list"}), 400
    urls = [str(u).strip() for u in raw if str(u).strip()]
    if not urls:
        return jsonify({"error": "No URL given"}), 400
    if len(urls) > MAX_YT_LINKS:
        return jsonify(
            {"error": f"Too many links ({len(urls)}). Maximum: {MAX_YT_LINKS} per job."}
        ), 400
    # Check each one individually and NAME the broken one. A collective
    # message ("Invalid URL") would make the user search through ten lines.
    from whisper_tm.core.ytdl import local_target

    for u in urls:
        if not u.startswith(("http://", "https://")):
            return jsonify({"error": f"That's not a link: {u[:80]}"}), 400
        why = local_target(u)
        if why:
            return jsonify({"error": why}), 400

    # "Audio only" comes as a toggle from the dialog; if missing, the
    # setting (youtube.audio_only) applies — hence None instead of False.
    audio_only = data.get("audio_only")
    if audio_only is not None:
        audio_only = parse_bool(audio_only)

    from whisper_tm.core.ytdl import start_download

    jid = start_download(urls, audio_only=audio_only)
    return jsonify({"jobId": jid, "links": len(urls)})


# ── YouTube cookies (settings) ────────────────────────────────────────────
# Content of a cookies.txt (Netscape format) — needed for age-restricted/
# private videos. Lives fixed in the data directory, no config key.


@media_bp.route("/api/youtube/cookies", methods=["GET"])
def youtube_cookies_status():
    """Is a cookies.txt for yt-dlp stored? Returns ``exists``, the number of
    cookie ``lines`` and the ``changed`` timestamp — never the content.
    """
    from whisper_tm.core.ytdl import cookies_path

    p = cookies_path()
    if not p.exists():
        return jsonify({"exists": False})
    try:
        # "#HttpOnly_" is not a comment: in the Netscape format it marks an
        # HttpOnly cookie (browser exports write them this way, YouTube's
        # login cookies among them). Skipping them undercounted the file.
        lines = sum(
            1
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and (not line.startswith("#") or line.startswith("#HttpOnly_"))
        )
    except OSError:
        lines = 0
    return jsonify({"exists": True, "lines": lines, "changed": int(p.stat().st_mtime)})


@media_bp.route("/api/youtube/cookies", methods=["POST"])
def youtube_cookies_save():
    """Store a cookies.txt (Netscape format) for yt-dlp downloads that need a
    login. Body: ``text`` — the file's content. Saved with mode 0600.
    """
    import os
    import tempfile

    from whisper_tm.core.ytdl import cookies_path

    data = request.json or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No cookie content"}), 400
    # Plausibility instead of strict validation: yt-dlp reports details itself.
    if "\t" not in text and "Netscape" not in text.splitlines()[0]:
        return jsonify(
            {
                "error": "This doesn't look like a cookies.txt "
                "(Netscape format) — please paste the file's CONTENT, "
                "not the URL"
            }
        ), 400
    p = cookies_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Cookies are credentials. Written first and restricted to 0600 only
    # afterwards, the file was readable for others in between (umask 022)
    # — and stayed so if the chmod failed. mkstemp creates it with mode 0600
    # from the start, and the rename replaces the old file in one step:
    # yt-dlp never reads a half-written one either.
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        os.replace(tmp, p)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return jsonify({"error": f"Cookies could not be saved: {e}"}), 500
    return jsonify({"ok": True})


@media_bp.route("/api/youtube/cookies", methods=["DELETE"])
def youtube_cookies_delete():
    """Delete the stored yt-dlp cookies.txt."""
    from whisper_tm.core.ytdl import cookies_path

    cookies_path().unlink(missing_ok=True)
    return jsonify({"ok": True})

"""
yt-dlp downloads as background job: single video OR entire playlist.

Provide a link or playlist link, everything is downloaded in sequence and
lands in the library's upload directory (media_dir/uploads/ — the scanner
finds the files there). Download runs
as a job in job_manager: progress via existing SSE channel
(/api/status/<job_id>), cancellable via /api/cancel/<job_id>.

Cookies: whoever wants to load age-restricted/private videos puts the
content of a cookies.txt in settings (Netscape format, e.g. exported from a
browser extension). The file is fixed under the app's data directory and
is passed to yt-dlp via --cookies as soon as it exists — no config key needed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from platformdirs import user_data_dir

# extensions singletons are set only in create_app() — therefore
# module access at RUNTIME (ext.job_manager), never bind at import.
import whisper_tm.extensions as ext
from whisper_tm.config import APP_NAME, config
from whisper_tm.core.procgroup import signal_group

logger = logging.getLogger(__name__)

# Format recipe like the previous single download: best MP4 with audio.
_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
_META_TIMEOUT = 60  # playlist enumeration
_DL_TIMEOUT = 3600  # per video


def cookies_path() -> Path:
    """Fixed location of cookies.txt (alongside DB/logs in data directory)."""
    return Path(user_data_dir(APP_NAME)) / "youtube_cookies.txt"


def ytdlp_cmd() -> list[str]:
    """Call prefix for yt-dlp.

    Prefers the PACKAGE in server venv (python -m yt_dlp): this works
    even if .venv/bin is not in PATH (server started directly via venv
    python, systemd, Docker). The bare binary is only the fallback for
    system-wide installations.
    """
    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass
    return [shutil.which("yt-dlp") or "yt-dlp"]


def available() -> bool:
    """Is yt-dlp usable (package in venv OR binary in PATH)?"""
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return shutil.which("yt-dlp") is not None


def _cookie_args() -> list[str]:
    p = cookies_path()
    return ["--cookies", str(p)] if p.exists() else []


#: What :func:`local_target` found for a host, for a short while
_resolved: dict[str, tuple[float, str]] = {}
_resolved_lock = threading.Lock()
_RESOLVED_FOR = 30.0  # seconds

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


def _as_link(value: str) -> str:
    """What goes to yt-dlp as a link: an http(s) link as it is, a bare
    YouTube video id (a flat playlist may give only that) as its watch link.

    Anything else is refused: as the last argument, a value beginning with
    "-" would be read by yt-dlp as an option — YouTube ids do
    begin with "-". So is a link to this machine or the local network
    (:func:`local_target`) — the initial one and every entry a playlist
    page lists."""
    value = str(value or "").strip()
    if value.startswith(("http://", "https://")):
        why = local_target(value)
        if why:
            raise RuntimeError(why)
        return value
    if _VIDEO_ID.match(value):
        return f"https://www.youtube.com/watch?v={value}"
    raise RuntimeError(f"Not a link: {value[:80]}")


def local_target(url: str) -> str:
    """Why ``url`` may not be fetched: its host is this machine or the local
    network ("" = it may).

    Links reach the downloader from the add dialog, the browser extension,
    the MCP tools and Ask's actions — and through the last two from text
    in transcripts or pages a language model read. Such a link to
    http://127.0.0.1:… or a router's address had the server fetch from
    inside the network. Checked after DNS resolution, every address the
    name has; a name that doesn't resolve is left to yt-dlp, which fails on
    it. ``youtube.allow_private_hosts`` (config.yaml) lets them through."""
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    if config.get("youtube", "allow_private_hosts", default=False):
        return ""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return f"Not a link: {url[:80]}"
    if not host:
        return f"No host in the link: {url[:80]}"
    # One lookup per host: fifty links of one job, or the entries of a
    # playlist, name the same host — and a name that doesn't resolve can take
    # seconds each time
    now = time.monotonic()
    with _resolved_lock:
        known = _resolved.get(host)
        if known and now - known[0] < _RESOLVED_FOR:
            return known[1]
    why = ""
    try:
        found = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        found = []
    for *_, sockaddr in found:
        try:
            ip = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
        except ValueError:
            continue
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if not ip.is_global or ip.is_multicast:
            why = (
                f"{host} is this machine or the local network ({ip}) — links are "
                f"fetched from the internet only (youtube.allow_private_hosts "
                f"in config.yaml allows them)"
            )
            break
    with _resolved_lock:
        if len(_resolved) >= 256:
            _resolved.clear()
        _resolved[host] = (now, why)
    return why


def _kill_group(proc, kill: bool = False) -> None:
    """Signal yt-dlp's whole process group — its ffmpeg merge child too.
    (No groups on Windows: the terminate()/kill() after it ends yt-dlp.)"""
    signal_group(proc.pid, kill=kill)


def _stop(proc, *, gentle: bool) -> None:
    """End yt-dlp and wait for it — a killed process nobody waits for stays
    a zombie, and its ffmpeg child kept writing."""
    if gentle:
        _kill_group(proc)
        proc.terminate()
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
    _kill_group(proc, kill=True)
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp (pid %s) did not end after SIGKILL", proc.pid)


def _remove_partial(tmp_dir: Path, tmp_id: str) -> None:
    for p in tmp_dir.glob(f"ytdl_{tmp_id}*"):
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("ytdl: partial file %s not removed: %s", p, e)


def enumerate_entries(url: str) -> list[dict]:
    """Resolve URL: single video → 1 entry, playlist → all entries.

    --flat-playlist downloads NOTHING and fetches only the metadata
    line per entry — a matter of seconds even for hundreds of videos.
    """
    cmd = [
        *ytdlp_cmd(),
        "--flat-playlist",
        "-J",
        "--no-warnings",
        *_cookie_args(),
        "--",
        _as_link(url),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=_META_TIMEOUT)
    if r.returncode != 0:
        err = (r.stderr or "").strip().split("\n")[-1] or "Unknown error"
        raise RuntimeError(f"yt-dlp: {err[:300]}")
    info = json.loads(r.stdout)
    if info.get("_type") == "playlist":
        entries = []
        for e in info.get("entries") or []:
            if not e:
                continue  # deleted/private video in the list
            link = e.get("url") or e.get("webpage_url") or ""
            if not str(link).startswith(("http://", "https://")):
                try:
                    link = _as_link(link or e.get("id") or "")
                except RuntimeError:
                    continue  # nothing that can be fetched
            entries.append(
                {
                    "url": link,
                    "title": e.get("title") or e.get("id") or "video",
                    "id": e.get("id") or "",
                }
            )
        return entries
    return [{"url": url, "title": info.get("title") or "video", "id": info.get("id") or ""}]


def title_stem(title: str) -> str:
    """The file name a title becomes, without an extension, at most 80
    characters (ingest finds earlier downloads by it).

    An ASCII title the way it always was (werkzeug's secure_filename:
    words joined by "_"), so earlier downloads keep being found by name.
    Any other script is kept — secure_filename dropped it, and a Japanese
    or Cyrillic title fell back to the video id — with the same "_" between
    words and only what a file system can't take removed."""
    raw = str(title or "").strip()
    if not raw:
        return ""
    if raw.isascii():
        from werkzeug.utils import secure_filename

        return secure_filename(raw)[:80]
    from whisper_tm.core.paths import file_name

    # A slash is part of a title ("AC/DC"), not a folder; the ".x" keeps a
    # dot inside the title from being read as an extension
    joined = re.sub(r"\s+", "_", raw.replace("/", "_").replace("\\", "_"))
    stem = file_name(joined + ".x")[:-2]
    if stem == "upload":
        return ""  # nothing of the title was left
    return stem[:80].rstrip("._ ")


def _target_path(dest_dir: Path, title: str, video_id: str, extension: str) -> Path:
    """Unique target name from title (sanitized, deduplicated).

    ``extension`` comes from the file actually downloaded — with
    youtube.audio_only, yt-dlp delivers e.g. .m4a/.opus instead of .mp4.
    """
    # In any script: secure_filename kept ASCII only, so a Japanese or
    # Cyrillic title fell back to the video id
    safe = title_stem(title) or title_stem(video_id) or "youtube"
    out = dest_dir / f"{safe}{extension}"
    n = 1
    while out.exists():
        out = dest_dir / f"{safe}_{n}{extension}"
        n += 1
    return out


#: Pause before the second attempt at a video. Short enough that a run over
#: twenty links doesn't take noticeably longer, long enough for a
#: throttling to settle down.
RESTART_PAUSE_SEC = 5.0


def _format_args(audio_only: bool | None = None) -> list[str]:
    """Format arguments (youtube.audio_only / extra_args).

    ``audio_only`` overrides the config for THIS download — the
    add dialog has its own toggle for it; ``None`` means
    "use the setting".
    """
    if audio_only is None:
        audio_only = config.get("youtube", "audio_only", default=False)
    if audio_only:
        # -x extracts the audio track; bestaudio[m4a] avoids re-encoding.
        args = ["-f", "bestaudio[ext=m4a]/bestaudio/best", "-x"]
    else:
        args = ["--merge-output-format", "mp4", "-f", _FORMAT]
    extra = (config.get("youtube", "extra_args", default="") or "").strip()
    if extra:
        from whisper_tm.core import commands

        try:
            args += commands.split_args(extra)
        except ValueError as e:
            raise RuntimeError(f"youtube.extra_args not parseable: {e}") from e
    return args


_PCT_RE = None


def _download_one(
    job, entry: dict, tmp_dir: Path, dest_dir: Path, report_it=None, audio_only: bool | None = None
) -> Path:
    """Download one video (cancellable) and move it to the destination.

    ``report_it(share)`` gets the download progress of THIS video
    (0.0–1.0) from the yt-dlp status lines — basis for the percentage
    display in the frontend chip.
    """
    import hashlib
    import re
    from collections import deque

    global _PCT_RE
    if _PCT_RE is None:
        _PCT_RE = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")

    tmp_id = hashlib.md5(f"{entry['url']}_{time.time()}".encode()).hexdigest()[:12]
    tmp_pattern = tmp_dir / f"ytdl_{tmp_id}.%(ext)s"
    # --newline: progress as individual lines instead of \r-overwriting —
    # only that way can it be read line by line.
    cmd = [
        *ytdlp_cmd(),
        "--no-playlist",
        *_format_args(audio_only),
        "--no-warnings",
        "--progress",
        "--newline",
        *_cookie_args(),
        "--output",
        str(tmp_pattern),
        "--paths",
        "temp:" + str(tmp_dir),
        "--",
        _as_link(entry["url"]),
    ]

    # Popen + poll loop instead of run(): the cancel button (job.cancel)
    # must be able to stop a RUNNING download, not just the next one.
    # stderr → stdout: ONE stream for progress AND error text (two pipes
    # would need two readers, otherwise the full buffer blocks the process).
    # Its own process group: a cancel ends the ffmpeg merge child too
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    tail: deque = deque(maxlen=15)  # last non-progress lines

    def _read():
        last_message = 0.0
        for line in proc.stdout:
            line = line.strip()
            m = _PCT_RE.search(line)
            if m:
                if report_it:
                    now = time.monotonic()
                    if now - last_message >= 0.5:  # throttle
                        last_message = now
                        try:
                            report_it(min(float(m.group(1)) / 100.0, 1.0))
                        except ValueError:
                            pass
            elif line:
                tail.append(line)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()

    start = time.time()
    while proc.poll() is None:
        # A cancel or a timeout leaves no partial file behind: otherwise it
        # would stay in the work folder until the next server start
        if job.cancel:
            _stop(proc, gentle=True)
            _remove_partial(tmp_dir, tmp_id)
            raise InterruptedError("Cancelled")
        if time.time() - start > _DL_TIMEOUT:
            _stop(proc, gentle=False)
            _remove_partial(tmp_dir, tmp_id)
            raise RuntimeError(f"Timeout (> {_DL_TIMEOUT // 60} min)")
        time.sleep(0.5)
    reader.join(timeout=5)

    candidates = [
        p
        for p in sorted(
            tmp_dir.glob(f"ytdl_{tmp_id}*"), key=lambda p: p.stat().st_size, reverse=True
        )
        if p.is_file() and p.stat().st_size > 1000
    ]
    if proc.returncode != 0 or not candidates:
        for p in tmp_dir.glob(f"ytdl_{tmp_id}*"):
            p.unlink(missing_ok=True)
        err = tail[-1] if tail else "No output file"
        raise RuntimeError(str(err)[:300])

    target = _target_path(
        dest_dir, entry["title"], entry["id"], candidates[0].suffix.lower() or ".mp4"
    )
    _to_target(candidates[0], target)
    for p in tmp_dir.glob(f"ytdl_{tmp_id}*"):
        p.unlink(missing_ok=True)
    return target


def _to_target(source: Path, target: Path) -> None:
    """Move the file to its destination — WITHOUT copying metadata.

    shutil.move uses copy2/copystat across filesystem boundaries, and
    the chmod/utime inside it fails on a CIFS mount with
    "[Errno 1] Operation not permitted" — the download would then be
    wasted. So: rename if possible, otherwise a plain content copy.
    """
    import os
    import shutil as _sh

    try:
        os.rename(source, target)
    except OSError:
        _sh.copyfile(source, target)
        source.unlink(missing_ok=True)


def _worker(job_id: str, urls: list[str], audio_only: bool | None = None) -> None:
    job = ext.job_manager.get(job_id)
    if not job:
        return
    job.update(status="processing", progress=1, message="Resolving URL (playlist detection)…")
    # THE configured upload directory (Settings → Folders),
    # never a self-built path. Default: uploads/ in the media directory.
    dest_dir = config.upload_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = config.dirs.get("chunks", config.work_dir / "ytdl_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Resolve each URL individually and merge the entries. ONE link with
    # ONE video, one link with a playlist, and ten individual links are
    # then treated the same — the loop below only knows entries.
    #
    # A link that can't be resolved (deleted, private, typo) must not
    # drag the others down with it: it becomes an error entry and
    # ends up in the summary. Only if NOT A SINGLE URL yielded anything
    # does the job fail — otherwise someone who throws in ten links
    # would get nothing at all because of one dead link.
    entries, error = [], []
    for u in urls:
        try:
            part = enumerate_entries(u)
        except Exception as e:
            error.append({"title": u, "error": f"URL resolution: {e}"[:200]})
            ext.job_manager.log(job, f"✕ {u} — resolution failed: {e}")
            continue
        if not part:
            error.append({"title": u, "error": "No (available) videos at this URL"})
            ext.job_manager.log(job, f"✕ {u} — no videos found")
            continue
        if len(urls) > 1 or len(part) > 1:
            ext.job_manager.log(job, f"↳ {u}: {len(part)} Video{'s' if len(part) != 1 else ''}")
        entries.extend(part)

    if not entries:
        ext.job_manager.fail(
            job, error[0]["error"] if error else "No (available) videos at these links"
        )
        return

    n = len(entries)
    source = f" from {len(urls)} links" if len(urls) > 1 else " (playlist)" if n > 1 else ""
    ext.job_manager.log(job, f"{n} Video{'s' if n != 1 else ''} found{source}")
    ok, rel_paths = [], []
    for i, entry in enumerate(entries):
        if job.cancel:
            break
        job.update(progress=2 + int(i / n * 96), message=f"⬇ {i + 1}/{n}: {entry['title'][:70]}")

        def _report(share, _i=i, _title=entry["title"][:60]):
            # Overall progress = finished videos + fraction of the current one;
            # the percentage in the text is that of the CURRENT video.
            job.update(
                progress=2 + int((_i + share) / n * 96),
                message=f"⬇ {_i + 1}/{n} · {int(share * 100)} % · {_title}",
            )

        # Start over ONCE, not just refetch.
        #
        # yt-dlp retries ten times on its own (--retries, default 10) —
        # but always against the SAME stream URL. If that's invalidated,
        # all ten attempts fail the same way, with "unable to download video
        # data: HTTP Error 403: Forbidden".
        #
        # In runs over many links, some drop out that way, while the
        # same call a few minutes later fetches them without issue —
        # the difference is the FRESH resolution, i.e. a new
        # yt-dlp process that gets new URLs handed to it. That's exactly
        # what this loop does; a retry inside yt-dlp couldn't achieve that
        # at all.
        target = None
        last = None
        for attempt in (1, 2):
            try:
                target = _download_one(
                    job, entry, tmp_dir, dest_dir, report_it=_report, audio_only=audio_only
                )
                break
            except InterruptedError as e:
                # A cancel is a cancel — don't retry. And do NOT
                # re-raise from here: the cancel path (`break` plus
                # "Cancelled" message) depends on the `try` below, which
                # a `raise` from here would bypass.
                last = e
                break
            except Exception as e:
                last = e
                if attempt == 1 and not job.cancel:
                    ext.job_manager.log(
                        job, f"↻ {i + 1}/{n}: {str(e)[:80]} — second attempt with fresh resolution"
                    )
                    time.sleep(RESTART_PAUSE_SEC)
        try:
            if target is None:
                raise last or RuntimeError("Download failed")
            # What came is what its bytes say, not its name: a manifest or a
            # playlist file would make every later read open what it lists
            from whisper_tm.core import media_formats

            if not media_formats.is_recording(target):
                with contextlib.suppress(OSError):
                    target.unlink()
                raise RuntimeError("what was downloaded is not an audio or video file — removed")
            ok.append(target.name)
            # Noted like the extension's downloads (core/ingest): without it
            # a video fetched here was never recognized on YouTube again
            try:
                from whisper_tm.core import ingest
                from whisper_tm.core.paths import canon

                ingest._remember(entry.get("id") or "", canon(target), entry.get("title") or "")
            except Exception as e:  # noqa — a nicety, the download stands
                logger.debug("ytdl: video id not noted: %s", e)
            # Immediately into the DB: the file then appears in the
            # (paginated) list without a full walk and can be enqueued directly.
            #
            # First relative to the CURRENT root, otherwise relative to the
            # upload directory itself. With a subfolder open as the root
            # while pasting the link, the file would be
            # "outside", `files` would stay empty, the UI wouldn't enqueue
            # anything and only report "lies outside the media
            # directory". Yet it jumps into the upload directory anyway
            # afterward (there the entries are named after their basename).
            try:
                from whisper_tm.scanner import register_single_file

                rel = register_single_file(target) or register_single_file(target, root=dest_dir)
                if rel:
                    rel_paths.append(rel)
            except Exception as e:  # noqa — display convenience, not required
                logger.debug("DB registration: %s", e)
            ext.job_manager.log(
                job, f"✓ {i + 1}/{n}: {target.name} ({target.stat().st_size // 1_000_000} MB)"
            )
        except InterruptedError:
            break
        except Exception as e:
            error.append({"title": entry["title"], "error": str(e)[:200]})
            ext.job_manager.log(job, f"✕ {i + 1}/{n}: {entry['title'][:60]} — {e}")

    if job.cancel:
        ext.job_manager.fail(
            job, f"Cancelled — {len(ok)}/{n} done (already downloaded files remain)"
        )
        return
    # "files" = rel_paths into the library (basis for enqueue/open in the
    # frontend); empty if the upload directory is outside it.
    summary = {
        "total": n,
        "ok": len(ok),
        "error": error,
        "files": rel_paths,
        "in_library": len(rel_paths) == len(ok),
        "directory": str(dest_dir),
    }
    if ok:
        ext.job_manager.complete(job, summary)
        job.update(
            progress=100,
            message=f"✅ {len(ok)}/{n} downloaded" + (f", {len(error)} errors" if error else ""),
        )
    else:
        ext.job_manager.fail(
            job, "All downloads failed — " + hint_for_error(error[0]["error"] if error else "?")
        )


#: Error texts that almost always mean the same thing: yt-dlp is too old for
#: whatever YouTube is currently doing. Typically a stable release gets
#: ``403 Forbidden`` while a current nightly build loads the same file in
#: seconds — the difference being a newer player client the stable one
#: doesn't know yet. The fallback attempts are typical too: ``web``/
#: ``ios``/``mweb`` report "Requested format is not available", ``tv``
#: "The page needs to be reloaded" — all the same cause.
_STALE = (
    "403: Forbidden",
    "Requested format is not available",
    "The page needs to be reloaded",
    "Sign in to confirm you're not a bot",
)


def js_runtime() -> str:
    """The JavaScript runtime yt-dlp can use for YouTube's challenges
    (Deno, or Node): its path, or ""."""
    for name in ("deno", "node"):
        found = shutil.which(name)
        if found:
            return found
    home_deno = Path.home() / ".deno" / "bin" / "deno"
    return str(home_deno) if home_deno.exists() else ""


def hint_for_error(text: str) -> str:
    """Append the next step to the bare yt-dlp error.

    An "HTTP Error 403: Forbidden" alone first sends everyone off
    searching for network, permission or block problems. The answer is
    almost always the same and isn't written anywhere: install the
    nightly build.
    """
    t = str(text or "")
    if any(m in t for m in _STALE) and not js_runtime():
        return (
            t + " · No JavaScript runtime found — YouTube needs one to unlock its "
            "formats. Install Deno (curl -fsSL https://deno.land/install.sh | sh) "
            "and restart the server; a systemd service needs ~/.deno/bin on its PATH."
        )
    if any(m in t for m in _STALE):
        return (
            t + " · This is almost always an outdated yt-dlp — YouTube "
            "changes its delivery more often than stable releases "
            "come out. Get the nightly build: "
            "uv pip install --python ./.venv/bin/python --pre --upgrade "
            "yt-dlp"
        )
    return t


def dedup_key(u: str) -> str:
    """What identifies two links as the SAME download.

    Text equality isn't enough. Example: a list of links holds the same
    video twice — once as ``…watch?v=X&t=345s``, once as ``…watch?v=X``.
    Both would be downloaded, the second file named ``…_1.m4a``, and it
    would be transcribed a second time, although the files are
    bit-identical.

    The time anchor (``t=``) is meaningless for a download, as are
    ``si=``, ``feature=`` and the like. The **playlist** (``list=``) is
    NOT: ``…watch?v=X&list=Y`` loads the whole list, not just X. Such
    links are therefore left untouched — merging them with a single
    link via the video ID would be the worse mistake, since then an
    entire playlist would silently disappear.
    """
    from urllib.parse import parse_qs, urlparse

    try:
        z = urlparse(u)
    except ValueError:
        return u
    host = (z.hostname or "").lower().removeprefix("www.")
    q = parse_qs(z.query)
    if "list" in q:
        return u
    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        label = (q.get("v") or [""])[0]
        if not label and z.path.startswith(("/shorts/", "/live/", "/embed/")):
            label = z.path.split("/")[2] if len(z.path.split("/")) > 2 else ""
        if label:
            return f"yt:{label}"
    elif host == "youtu.be" and len(z.path) > 1:
        return f"yt:{z.path.lstrip('/')}"
    return u


def start_download(url: str | list[str], audio_only: bool | None = None) -> str:
    """Create and start a download job. Returns the job ID.

    ``url`` is a single link or a list — multiple links result in
    ONE job. That's the point: someone who wants to fetch ten videos
    shouldn't have to start ten jobs, keep an eye on ten progress
    displays, and clean up ten times.

    ``audio_only=None`` uses the setting (youtube.audio_only); the
    toggle in the add dialog sends True/False explicitly.
    """
    import hashlib

    urls = [url] if isinstance(url, str) else list(url)
    # Remove duplicates, keep order: someone who throws in the same
    # link twice doesn't want it downloaded twice.
    seen: dict[str, str] = {}
    for u in urls:
        u = (u or "").strip()
        if u:
            seen.setdefault(dedup_key(u), u)
    urls = list(seen.values())
    if not urls:
        raise ValueError("No URL specified")

    jid = "ytdl_" + hashlib.sha256(f"{'|'.join(urls)}_{time.time()}".encode()).hexdigest()[:12]
    # The job's filename is the label in the UI. With multiple links,
    # the concatenated list would be unreadable.
    labelling = urls[0] if len(urls) == 1 else f"{len(urls)} Links"
    ext.job_manager.create(jid, filename=labelling, job_type="youtube")
    threading.Thread(target=_worker, args=(jid, urls, audio_only), daemon=True).start()
    return jid

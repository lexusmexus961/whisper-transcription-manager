"""
Links in — YouTube videos and playlists to transcripts, summaries, analyses.

YouTube videos on a page can be selected — one or several; the extension
recognizes them and sends them to be transcribed (saved in the upload
folder), or transcribed and summarized. This is the server half: a browser
extension (or anything else that has links) sends them with a mode, and
asks how far they are.

Modes:

* ``transcribe`` — each video into the upload folder, queued at the front;
* ``summarize`` — the same, each with its summary after the transcript (the
  queue's own option); a video transcribed before gets its summary in the
  LLM queue, unless it has one;
* ``analyze`` — the same as transcribe, then one analysis over all of them
  with the task; it waits for their transcripts (core/analysis.py).

**Nothing twice.** The video id of every download is noted
(``youtube_ids.json`` in the data folder). A video downloaded before that
existed is found by the file name its title gets (ytdl._target_path) in the
upload folder. Either way it is not downloaded again; its transcript and
summary are taken as they are.

**Read live.** A batch keeps what can't be looked up — the links, what they
resolved to, the file each became, errors. Whether a file is waiting in the
queue, transcribed or summarized is read from the queue, the transcripts
and the summaries every time it is asked, so it is never stale. Batches are
JSON under the data folder (``ingest/``), like the analyses.

**How, chosen once or each time.** Summary and transcription settings
can be set as defaults or chosen per send: videos from the browser have
their own defaults (``ingest.defaults``
— model, language, speakers, the summary's template and language), apart
from the general ones; a send may set any of them for itself.

**Read and asked about.** The extension's side panel reads a video's
transcript and summaries (:func:`transcript`) and asks the language model
about it (:func:`ask` — a chat conversation with the video attached).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from whisper_tm.config import config
from whisper_tm.core import paths

logger = logging.getLogger(__name__)

MODES = ("transcribe", "summarize", "analyze")
#: Links one batch takes — a playlist counts its videos.
MAX_VIDEOS = 200
MEDIA_EXT = {
    ".m4a",
    ".mp4",
    ".webm",
    ".opus",
    ".mp3",
    ".mkv",
    ".wav",
    ".aac",
    ".ogg",
    ".flac",
    ".m4v",
    ".mov",
}

#: One download at a time, across batches — parallel yt-dlp runs get
#: throttled, and the storage writes them one after another anyway.
_download_lock = threading.Lock()
_lock = threading.Lock()
_running: dict[str, threading.Event] = {}  # batch → its cancel signal
_live: dict[str, dict] = {}  # batch → the worker's copy, progress and all


class IngestError(RuntimeError):
    """A batch can't be made — no links, an unknown mode, no yt-dlp."""


# ══════════════════════════════════════════════════════════════════════════
#  How a video is transcribed and summarized
# ══════════════════════════════════════════════════════════════════════════

#: The transcription's options — named as the queue's settings (api/queue.py)
TX_KEYS = ("model", "language", "diarize", "numSpeakers", "vocal_isolation", "vocab_preset")
#: The summary's
SUMMARY_KEYS = ("summary_preset", "summary_model", "summary_language", "summary_prompt")
#: A summary's own instruction ("Custom…" in the side panel) — free text
PROMPT_MAX = 4000
#: The download's: only the sound, or the video
DOWNLOAD_KEYS = ("audio_only",)
OPTION_KEYS = TX_KEYS + SUMMARY_KEYS + DOWNLOAD_KEYS
#: On, off, or None — the general setting
_SWITCHES = ("diarize", "vocal_isolation", "audio_only")

#: The Whisper models — those of the app's own transcription dialog
#: (static/js/txoptions.js), with what tells them apart.
WHISPER_MODELS: dict[str, str] = {
    "turbo": "turbo — fast & very good",
    "large-v3": "large-v3 — maximum quality",
    "large-v2": "large-v2 — proven",
    "distil-large-v3": "distil-large-v3 — faster",
    "distil-large-v2": "distil-large-v2",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
    "medium.en": "medium.en — English only",
    "small.en": "small.en — English only",
    "base.en": "base.en — English only",
    "tiny.en": "tiny.en — English only",
    "distil-medium.en": "distil-medium.en — English only",
    "distil-small.en": "distil-small.en — English only",
}


def _names(section: str, key: str) -> list[str]:
    d = config.get(section, key, default={}) or {}
    return [str(k) for k in d] if isinstance(d, dict) else []


def _choices(key: str):
    """What a text option may be."""
    from whisper_tm.core.summarize import LANGUAGES

    if key == "model":
        return WHISPER_MODELS
    if key == "language":
        return {"auto", *LANGUAGES}
    if key == "summary_language":
        return LANGUAGES
    if key == "vocab_preset":
        return _names("transcription", "vocab_presets")
    return _names("llm", "presets" if key == "summary_preset" else "models")


def clean_options(d: dict | None) -> dict:
    """What a send or the defaults may set — anything else, and a value that
    is none of the choices, is dropped. A key that is there with "" (or
    None for a switch or the number of speakers) says "the general
    setting"; a key that is not there says nothing."""
    out: dict = {}
    for k, v in (d or {}).items():
        if k not in OPTION_KEYS:
            continue
        if k in _SWITCHES:
            out[k] = v if isinstance(v, bool) else None
        elif k == "summary_prompt":
            # Written by the user, not one of the choices: taken as it is
            out[k] = str(v or "").strip()[:PROMPT_MAX]
        elif k == "numSpeakers":
            try:
                n = int(v)
            except (TypeError, ValueError):
                n = None
            out[k] = n if n is not None and 0 <= n <= 20 else None
        else:
            text = str(v or "").strip()
            out[k] = text if text and text in _choices(k) else ""
    return out


def _blank() -> dict:
    return {k: None if k in _SWITCHES or k == "numSpeakers" else "" for k in OPTION_KEYS}


def defaults() -> dict:
    """The defaults for videos from the browser — every option, "" or None
    where the general setting applies."""
    stored = config.get("ingest", "defaults", default={}) or {}
    out = _blank()
    out.update(clean_options(stored if isinstance(stored, dict) else {}))
    return out


def save_defaults(d: dict) -> dict:
    """Change some of the defaults (the extension's ⚙) — the rest stay."""
    out = defaults()
    out.update(clean_options(d))
    config.set("ingest", "defaults", out)
    config.save()
    return out


def general() -> dict:
    """What "the general setting" is, option by option — the forms name it."""

    def flag(*keys, default):
        v = config.get(*keys, default=default)
        return v if isinstance(v, bool) else default

    try:
        speakers = int(config.get("transcription", "default_num_speakers", default=0) or 0)
    except (TypeError, ValueError):
        speakers = 0
    return {
        "model": config.get("transcription", "default_model", default="turbo") or "turbo",
        "language": config.get("transcription", "default_language", default="auto") or "auto",
        "diarize": flag("transcription", "default_diarize", default=True),
        "numSpeakers": speakers,
        "vocal_isolation": flag("vocal_isolation", "default_enabled", default=True),
        "vocab_preset": config.get("transcription", "default_vocabulary", default="") or "",
        "summary_preset": config.get("llm", "auto_summarize_preset", default="") or "",
        "summary_model": config.get("llm", "auto_summarize_model", default="") or "",
        "summary_language": "",
        "audio_only": flag("youtube", "audio_only", default=False),
    }


def options() -> dict:
    """Everything the extension's forms offer: the choices, the general
    settings, and the defaults for videos from the browser."""
    from whisper_tm.core.llm import prompts
    from whisper_tm.core.summarize import LANGUAGES

    return {
        "whisper_models": [{"id": k, "label": v} for k, v in WHISPER_MODELS.items()],
        "languages": [{"id": k, "label": v} for k, v in LANGUAGES.items()],
        "vocab_presets": _names("transcription", "vocab_presets"),
        "summary_presets": _names("llm", "presets"),
        "summary_presets_builtin": list(prompts.STANDARD_PRESETS),
        "summary_models": _names("llm", "models"),
        "general": general(),
        "defaults": defaults(),
    }


def merged_options(options: dict | None) -> dict:
    """A send's options over the defaults."""
    out = defaults()
    out.update(clean_options(options))
    return out


def _db():
    from whisper_tm.extensions import db

    return db


# ══════════════════════════════════════════════════════════════════════════
#  Which videos are there already
# ══════════════════════════════════════════════════════════════════════════


def _index_path() -> Path:
    return Path(config.data_dir) / "youtube_ids.json"


def _index() -> dict:
    p = _index_path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


def _remember(video_id: str, abs_path: str, title: str) -> None:
    """Note a download's video id — every download: the extension's batches,
    the web UI's YouTube dialog and the MCP server (core/ytdl)."""
    if not video_id or not abs_path:
        return
    with _lock:
        idx = _index()
        idx[video_id] = {"abs_path": abs_path, "title": title, "at": time.time()}
        p = _index_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


def video_id(url: str) -> str:
    """The YouTube id of a link — "" for a playlist or anything else."""
    from whisper_tm.core.ytdl import dedup_key

    key = dedup_key(url or "")
    return key[3:] if key.startswith("yt:") else ""


def _upload_files() -> dict[str, str]:
    """The upload folder's media files by name without extension."""
    out: dict[str, str] = {}
    try:
        with os.scandir(config.upload_dir) as it:
            for e in it:
                stem, ext = os.path.splitext(e.name)
                if ext.lower() in MEDIA_EXT and e.is_file():
                    out.setdefault(stem, paths.canon(e.path))
    except OSError:
        pass
    return out


def _safe_title(title: str) -> str:
    from whisper_tm.core.ytdl import title_stem

    return title_stem(title or "")


class _Lookup:
    """Which file a page's video became — the index and the upload folder
    read once per request, maps built from them, the disk asked only about
    a match (known() asks about up to 200 videos at a time)."""

    def __init__(self):
        self.index = _index()
        self.by_title: dict[str, list[str]] = {}
        self.by_stem: dict[str, list[str]] = {}
        for entry in sorted(self.index.values(), key=lambda e: e.get("at") or 0, reverse=True):
            p = entry.get("abs_path") or ""
            if not p:
                continue
            t = _safe_title(entry.get("title") or "")
            if t:
                self.by_title.setdefault(t, []).append(p)
            self.by_stem.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(p)
        self._files: dict[str, str] | None = None

    def files(self) -> dict[str, str]:
        if self._files is None:
            self._files = _upload_files()
        return self._files

    def find(self, vid: str, title: str = "") -> str:
        """By the noted video id; with a title, also by the noted downloads'
        titles and file names and the upload folder's file names (a video
        downloaded before the ids were noted). No title: the id alone — a
        page without one learns nothing about the upload folder (the
        extension sends titles only on YouTube pages)."""
        hit = self.index.get(vid) if vid else None
        if hit and os.path.exists(hit.get("abs_path") or ""):
            return hit["abs_path"]
        safe = _safe_title(title)
        if not safe:
            return ""
        for p in self.by_title.get(safe, []) + self.by_stem.get(safe, []):
            if os.path.exists(p):
                return p
        files = self.files()
        if safe in files:
            return files[safe]
        numbered = re.compile(re.escape(safe) + r"_\d+")
        return next((p for stem, p in sorted(files.items()) if numbered.fullmatch(stem)), "")


def find_known(vid: str, title: str = "") -> str:
    """The file a page's video became (see :class:`_Lookup`) — "" if none."""
    return _Lookup().find(vid, title)


def find_existing(vid: str, title: str, files: dict[str, str] | None = None) -> str:
    """The file a video became before — by its id, else by the name its
    title gets in the upload folder.

    For a batch's own videos, whose titles come from yt-dlp: it also knows
    the downloads from before the ids were noted. What a web page asks
    goes through :func:`find_known`."""
    hit = _index().get(vid) if vid else None
    if hit and os.path.exists(hit.get("abs_path") or ""):
        return hit["abs_path"]
    if not title:
        return ""
    safe = _safe_title(title)
    if not safe:
        return ""
    files = files if files is not None else _upload_files()
    if safe in files:
        return files[safe]
    numbered = re.compile(re.escape(safe) + r"_\d+")
    return next((p for stem, p in sorted(files.items()) if numbered.fullmatch(stem)), "")


# ══════════════════════════════════════════════════════════════════════════
#  What a file is up to
# ══════════════════════════════════════════════════════════════════════════


def _upload_root() -> str:
    return paths.canon(config.upload_dir)


def file_state(abs_path: str, conn=None) -> dict:
    """Transcribed, waiting or running in the queue, summarized — read now."""
    db = _db()
    out = {
        "transcribed": False,
        "queue": "",
        "queue_error": "",
        "summary": "",
        "summary_at": 0.0,
        "summarizing": False,
    }
    if not db or not abs_path:
        return out

    def ask(c):
        r = c.execute(
            "SELECT segment_count FROM transcripts WHERE abs_path = ? ORDER BY id DESC LIMIT 1",
            (abs_path,),
        ).fetchone()
        out["transcribed"] = bool(r and (r[0] or 0) > 0)
        base, name = os.path.split(abs_path)
        r = c.execute(
            "SELECT status, error FROM queue_items WHERE base_dir = ? AND filename = ? "
            "ORDER BY id DESC LIMIT 1",
            (base, name),
        ).fetchone()
        if r:
            out["queue"], out["queue_error"] = r[0] or "", r[1] or ""
        r = c.execute(
            "SELECT text, created_at FROM summaries WHERE abs_path = ? "
            "ORDER BY sum_index DESC, id DESC LIMIT 1",
            (abs_path,),
        ).fetchone()
        out["summary"] = (r[0] or "") if r else ""
        out["summary_at"] = float(r[1] or 0) if r else 0.0
        # Also with a summary there: one more may be on its way ("again")
        r = c.execute(
            "SELECT 1 FROM llm_queue WHERE kind = 'summary' AND filename = ? "
            "AND status IN ('pending', 'running') LIMIT 1",
            (name,),
        ).fetchone()
        out["summarizing"] = bool(r)

    if conn is not None:
        ask(conn)
    else:
        with db.connection() as c:
            ask(c)
    if not out["summary"] and not out["summarizing"]:
        try:
            from whisper_tm.core import queue as q

            out["summarizing"] = q.summary_pending(abs_path)
        except Exception:  # noqa: BLE001 — not summarizing then
            pass
    return out


def open_link(abs_path: str) -> str:
    """The page's address that opens this recording."""
    return "/?open=" + quote(abs_path, safe="/") if abs_path else ""


def _state(item: dict, mode: str, conn) -> dict:
    """One video as the extension shows it: where it stands, and what it
    became."""
    out = {k: item.get(k) for k in ("url", "video_id", "title", "abs_path", "known", "progress")}
    out["error"] = item.get("error") or ""
    step = item.get("step") or "waiting"
    if out["error"]:
        out["state"] = "error"
    elif step != "ready":
        out["state"] = step  # waiting · resolving · downloading
    else:
        f = file_state(item["abs_path"], conn)
        if f["transcribed"]:
            out["state"] = "transcribed"
            if mode == "summarize":
                # Asked for again: the one from before doesn't count
                asked = item.get("asked_at") or 0
                if f["summary"] and f["summary_at"] >= asked:
                    out["state"], out["summary"] = "summarized", f["summary"]
                elif f["summarizing"]:
                    out["state"] = "summarizing"
                elif asked:
                    out["state"] = "error"
                    out["error"] = "The new summary failed — the LLM queue says why"
        else:
            out["state"] = {
                "pending": "queued",
                "processing": "transcribing",
                "error": "error",
            }.get(f["queue"], "not queued")
            if f["queue"] == "error":
                out["error"] = f["queue_error"] or "Transcription failed"
    out["open"] = open_link(item.get("abs_path") or "")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Storage
# ══════════════════════════════════════════════════════════════════════════

_ID = re.compile(r"^[a-f0-9]{12}$")


def _folder() -> Path:
    d = Path(config.data_dir) / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(bid: str) -> Path:
    if not _ID.match(bid or ""):
        raise IngestError("unknown batch")
    return _folder() / f"{bid}.json"


def load(bid: str) -> dict:
    """A batch — while it downloads, the worker's own copy: a video's
    progress is only written there."""
    p = _path(bid)
    if bid in _live:
        return _live[bid]
    if not p.is_file():
        raise IngestError("unknown batch")
    return json.loads(p.read_text(encoding="utf-8"))


def save(b: dict) -> None:
    b["updated"] = time.time()
    p = _path(b["id"])
    with _lock:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


def public(b: dict) -> dict:
    """The batch with every video's state, read now."""
    db = _db()
    out = {
        k: b.get(k)
        for k in (
            "id",
            "mode",
            "task",
            "title",
            "created",
            "updated",
            "error",
            "analysis_id",
            "message",
        )
    }
    out["downloading"] = b.get("status") == "running" and b["id"] in _running
    if db:
        with db.connection() as conn:
            out["items"] = [
                _state(it, b.get("mode", "transcribe"), conn)
                for it in b.get("items") or []
                if not it.get("dropped")
            ]
    else:
        out["items"] = []
    states = [it["state"] for it in out["items"]]
    busy = {"waiting", "resolving", "downloading", "queued", "transcribing", "summarizing"}
    if b.get("status") == "cancelled":
        out["status"] = "cancelled"
    elif b.get("status") == "error":
        out["status"] = "error"
    elif out["downloading"] or b.get("status") == "running" or any(s in busy for s in states):
        out["status"] = "working"
    else:
        out["status"] = "done"
    if b.get("analysis_id"):
        try:
            from whisper_tm.core import analysis

            a = analysis.public(analysis.load(b["analysis_id"]))
            out["analysis"] = {
                k: a.get(k) for k in ("id", "title", "status", "progress", "message", "error")
            }
            out["analysis"]["open"] = "/?analysis=" + a["id"]
            if a.get("status") in ("waiting", "queued", "running"):
                out["status"] = "working"
        except Exception:  # noqa: BLE001 — deleted meanwhile
            out["analysis"] = None
    return out


def listing(limit: int = 30) -> list[dict]:
    out = []
    for p in sorted(_folder().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(public(load(p.stem)))
        except (OSError, ValueError, IngestError):
            continue
        if len(out) >= limit:
            break
    return out


# ══════════════════════════════════════════════════════════════════════════
#  A batch
# ══════════════════════════════════════════════════════════════════════════


def start(
    urls: list[str],
    mode: str = "transcribe",
    *,
    task: str = "",
    title: str = "",
    audio_only: bool | None = None,
    options: dict | None = None,
    again: bool = False,
) -> dict:
    """Take the links and start working on them — answers at once.

    ``options``: how they are transcribed and summarized (:data:`OPTION_KEYS`)
    — what is left out comes from the defaults. ``again``: a video summarized
    before gets one more summary (another template, another language)."""
    from whisper_tm.core import ytdl

    if mode not in MODES:
        raise IngestError(f"Unknown mode: {mode!r} ({', '.join(MODES)})")
    if mode == "analyze" and not (task or "").strip():
        raise IngestError("An analysis needs a task")
    if mode in ("summarize", "analyze"):
        from whisper_tm.core import llm_client

        if not llm_client.ready("summary"):
            # Taken without one, the videos were fetched and transcribed and
            # the summary failed where the extension never showed it
            what = "a summary" if mode == "summarize" else "an analysis"
            raise IngestError(
                "No language model is set up in Whisper Transcription Manager "
                "(Settings → "
                f"AI models) — {what} needs one. Transcribe works without."
            )
    clean, seen = [], set()
    for u in urls or []:
        u = str(u or "").strip()
        if not u.startswith(("http://", "https://")):
            continue
        key = ytdl.dedup_key(u)
        if key not in seen:
            seen.add(key)
            clean.append(u)
    if not clean:
        raise IngestError("No links")
    if len(clean) > MAX_VIDEOS:
        raise IngestError(f"{len(clean)} links are more than one batch takes ({MAX_VIDEOS})")
    if not ytdl.available():
        raise IngestError("yt-dlp is not installed")
    b = {
        "id": uuid.uuid4().hex[:12],
        "created": time.time(),
        "mode": mode,
        "task": (task or "").strip(),
        "title": (title or "").strip()[:120],
        "audio_only": audio_only,
        "status": "running",
        "message": "Starting …",
        "options": merged_options(options),
        "again": bool(again),
        "items": [
            {
                "url": u,
                "video_id": video_id(u),
                "title": "",
                "abs_path": "",
                "step": "waiting",
                "progress": 0,
                "known": False,
                "error": "",
            }
            for u in clean
        ],
        "analysis_id": "",
        "error": "",
    }
    save(b)
    stop = threading.Event()
    _running[b["id"]] = stop
    threading.Thread(
        target=_work, args=(b["id"], stop), name=f"ingest-{b['id']}", daemon=True
    ).start()
    return b


def drop(bid: str, urls: list[str] | None = None) -> dict:
    """Take videos out of a batch — ``urls`` left out: all of them. One not
    fetched yet is not fetched, one downloading stops, one this batch put
    in the queue leaves it while it waits there; what was transcribed or
    is being transcribed stays in the library. A dropped video is no
    longer listed with the batch."""
    from whisper_tm.core import queue as q

    b = load(bid)
    want = None if urls is None else {str(u) for u in urls}
    dropped, unqueued = 0, []
    for it in b.get("items") or []:
        if it.get("dropped") or (want is not None and it.get("url") not in want):
            continue
        it["dropped"] = True
        dropped += 1
        # Put in the queue by this batch, not found in the library before
        if it.get("step") == "ready" and it.get("abs_path") and not it.get("known"):
            unqueued.append(it["abs_path"])
    if not dropped:
        return {"dropped": 0, "unqueued": 0}
    n = q.remove_pending_for(unqueued) if unqueued else 0
    save(b)
    return {"dropped": dropped, "unqueued": n}


def cancel(bid: str) -> bool:
    """Stop the downloads; what is queued already stays queued."""
    b = load(bid)
    stop = _running.get(bid)
    if stop is not None:
        stop.set()
        return True
    if b.get("status") == "running":  # a worker that is gone (restart)
        b.update(status="cancelled", message="Cancelled")
        save(b)
        return True
    return False


class _Job:
    """What ytdl._download_one asks a job: whether it is cancelled — the
    batch, or this one video (:func:`drop`)."""

    def __init__(self, stop: threading.Event, item: dict | None = None):
        self._stop = stop
        self._item = item or {}

    @property
    def cancel(self) -> bool:
        return self._stop.is_set() or bool(self._item.get("dropped"))


def _settings(mode: str, opts: dict | None = None) -> dict:
    """The queue entry's settings: what the send or the defaults set — the
    rest is the general setting (Settings → Transcription) — and the
    summary right after the transcript where it is asked for."""
    opts = opts if opts is not None else defaults()
    out = {k: opts[k] for k in TX_KEYS if opts.get(k) not in (None, "")}
    if mode == "summarize":
        out["summarize"] = True
        out.update({k: v for k, v in _summary_params(opts).items() if v})
    return out


def _summary_params(opts: dict) -> dict:
    """The summary's template, model and language — named as the queue's
    settings. An empty template or model is the one of the automatic
    summaries (Settings → AI tasks)."""
    g = general()
    return {
        "summarize_preset": opts.get("summary_preset") or g["summary_preset"],
        "summarize_model": opts.get("summary_model") or g["summary_model"],
        "summarize_language": opts.get("summary_language") or "",
        # An own instruction instead of the template ("Custom…" in the
        # side panel)
        "summarize_prompt": opts.get("summary_prompt") or "",
    }


def _resolve(b: dict, stop: threading.Event) -> list[dict]:
    """Every link as the videos it holds — a playlist as all of them."""
    from whisper_tm.core import ytdl

    out: list[dict] = []
    seen: set[str] = set()
    for it in b["items"]:
        if stop.is_set():
            break
        if it.get("dropped"):
            out.append(it)
            continue
        it["step"] = "resolving"
        save(b)
        try:
            entries = ytdl.enumerate_entries(it["url"])
        except Exception as e:  # noqa: BLE001 — this link fails, the others go on
            it.update(step="ready", error=ytdl.hint_for_error(str(e))[:300])
            out.append(it)
            continue
        if not entries:
            it.update(step="ready", error="No (available) videos at this link")
            out.append(it)
            continue
        if it.get("dropped"):  # taken out while it was looked up
            out.append(it)
            continue
        for e in entries:
            vid = e.get("id") or ""
            if vid and vid in seen:
                continue
            seen.add(vid)
            url = e.get("url") or it["url"]
            if vid and not str(url).startswith("http"):
                url = f"https://www.youtube.com/watch?v={vid}"
            out.append(
                {
                    "url": url if len(entries) > 1 else it["url"],
                    "video_id": vid,
                    "title": e.get("title") or "",
                    "abs_path": "",
                    "step": "waiting",
                    "progress": 0,
                    "known": False,
                    "error": "",
                    "playlist": it["url"] if len(entries) > 1 else "",
                }
            )
    return out[:MAX_VIDEOS]


def _work(bid: str, stop: threading.Event) -> None:
    from whisper_tm.core import queue as q
    from whisper_tm.core import ytdl

    b = load(bid)
    _live[bid] = b
    try:
        b["message"] = "Looking up the links …"
        b["items"] = _resolve(b, stop)
        save(b)
        upload = Path(config.upload_dir)
        upload.mkdir(parents=True, exist_ok=True)
        tmp_dir = config.dirs.get("chunks", config.work_dir / "ytdl_tmp")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        root = _upload_root()
        files = _upload_files()
        opts = b.get("options") or defaults()
        settings = _settings(b["mode"], opts)
        # Asked for with the send itself, else the option (None: the setting)
        audio_only = b.get("audio_only")
        if audio_only is None:
            audio_only = opts.get("audio_only")
        for i, it in enumerate(b["items"]):
            if stop.is_set():
                break
            if it.get("error") or it.get("dropped"):
                continue
            existing = find_existing(it["video_id"], it["title"], files)
            if existing:
                it.update(abs_path=existing, known=True)
                _remember(it["video_id"], existing, it["title"])
            else:
                it["step"] = "downloading"
                b["message"] = f"Downloading {i + 1} of {len(b['items'])} …"
                save(b)

                def report(share, _it=it):
                    _it["progress"] = int(share * 100)

                try:
                    with _download_lock:
                        target = ytdl._download_one(
                            _Job(stop, it),
                            {
                                "url": it["url"],
                                "title": it["title"] or it["video_id"],
                                "id": it["video_id"],
                            },
                            Path(tmp_dir),
                            upload,
                            report_it=report,
                            audio_only=audio_only,
                        )
                except InterruptedError:
                    it["step"] = "waiting"
                    if it.get("dropped") and not stop.is_set():
                        save(b)
                        continue  # this one taken out, the others go on
                    break
                except Exception as e:  # noqa: BLE001 — this video fails, the others go on
                    it.update(step="ready", error=ytdl.hint_for_error(str(e))[:300])
                    save(b)
                    continue
                abs_path = paths.canon(target)
                try:
                    from whisper_tm.scanner import register_single_file

                    register_single_file(target, root=upload)
                except Exception as e:  # noqa: BLE001 — the queue registers it too
                    logger.debug("ingest: register %s: %s", target, e)
                it.update(abs_path=abs_path, progress=100)
                _remember(it["video_id"], abs_path, it["title"])
            it["step"] = "ready"
            if it.get("dropped"):
                save(b)  # taken out while it downloaded
                continue
            _hand_on(it, b["mode"], settings, root, q, opts, b.get("again", False))
            save(b)
        if stop.is_set():
            b.update(status="cancelled", message="Cancelled — what was queued stays queued")
            for it in b["items"]:
                if it.get("step") != "ready":
                    it.update(step="ready", error=it.get("error") or "Cancelled")
        else:
            b.update(status="downloaded", message="")
            if b["mode"] == "analyze":
                _analysis(b)
        save(b)
    except Exception as e:  # noqa: BLE001 — reported with the batch
        logger.exception("Ingest %s failed", bid)
        b.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        save(b)
    finally:
        _running.pop(bid, None)
        _live.pop(bid, None)


def _hand_on(
    it: dict, mode: str, settings: dict, root: str, q, opts: dict | None = None, again: bool = False
) -> None:
    """Into the queue at the front — unless transcribed already; then, for a
    summary, into the LLM queue unless it has one (or ``again``: one more)."""
    f = file_state(it["abs_path"])
    base, name = os.path.split(it["abs_path"])
    if not f["transcribed"]:
        # A waiting entry of the same file is pulled to the front; a running
        # one is left as it is (core/queue.enqueue). A paused queue runs
        # just this one — the rest stays paused (core/queue.run_now)
        added = q.enqueue([name], settings, base_dir=base, front=True)
        q.run_now([e.get("id") for e in added or []])
        return
    if mode != "summarize" or f["summarizing"] or (f["summary"] and not again):
        return
    from whisper_tm.core.llm_queue import llm_queue
    from whisper_tm.core.summarize import LANGUAGES

    sp = _summary_params(opts if opts is not None else defaults())
    params = {
        "media_dir": base,
        "auto": True,
        # The summary's name in the app's list
        "label": (
            ("Custom" if sp["summarize_prompt"] else sp["summarize_preset"] or "Summary")
            + (f" · {LANGUAGES[sp['summarize_language']]}" if sp["summarize_language"] else "")
            + " (browser extension)"
        ),
    }
    for k, to in (
        ("summarize_preset", "preset"),
        ("summarize_model", "model"),
        ("summarize_language", "language"),
    ):
        if sp[k]:
            params[to] = sp[k]
    if sp["summarize_prompt"]:
        params["system_prompt"] = sp["summarize_prompt"]  # instead of the template's
    llm_queue.enqueue("summary", name, params, front=True)
    if f["summary"]:
        it["asked_at"] = time.time()  # the one from before doesn't count


def _analysis(b: dict) -> None:
    """One analysis over the batch's videos — it waits for their transcripts."""
    from whisper_tm.core import analysis

    done = [
        it["abs_path"]
        for it in b["items"]
        if it.get("abs_path") and not it.get("error") and not it.get("dropped")
    ]
    if not done:
        b["error"] = "No video could be fetched — no analysis"
        return
    a = analysis.create(
        {
            "kind": "paths",
            "abs_paths": done,
            "label": f"{len(done)} YouTube video{'s' if len(done) != 1 else ''}",
        },
        b["task"],
        title=b.get("title") or "",
        wait=True,
    )
    b["analysis_id"] = a["id"]


#: Videos one known() call asks about — a page's marks, not an inventory
KNOWN_MAX = 200


def known(videos: list[dict], summaries: bool = False) -> list[dict]:
    """Which of these videos are in the library already — for the extension's
    marks on a page. ``videos``: ``[{url, title}]``. ``summaries``: with the
    summary itself, for the video a watch page plays (its button shows it)."""
    db = _db()
    out = []
    if not db:
        return out
    lookup = _Lookup()
    with db.connection() as conn:
        for v in videos[:KNOWN_MAX]:
            url = str((v or {}).get("url") or "")
            vid = video_id(url)
            hit = lookup.find(vid, str((v or {}).get("title") or ""))
            row = {"url": url, "video_id": vid, "abs_path": hit, "state": ""}
            if hit:
                f = file_state(hit, conn)
                row["state"] = (
                    "summarized"
                    if f["summary"]
                    else "transcribed"
                    if f["transcribed"]
                    else {"pending": "queued", "processing": "transcribing"}.get(
                        f["queue"], "downloaded"
                    )
                )
                row["open"] = open_link(hit)
                if summaries and f["summary"]:
                    row["summary"] = f["summary"]
            out.append(row)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Read and asked about — the extension's side panel
# ══════════════════════════════════════════════════════════════════════════

#: A video's transcript a conversation carries at most (characters, ~4 per
#: token): about two hours of talk. The model reads it with the first
#: question; the next ones find it in the server's prompt cache.
VIDEO_CHARS = 200_000
_chats_lock = threading.Lock()


def _find(url: str, title: str = "") -> str:
    """The file a video became — "" if none (only an earlier download)."""
    return find_known(video_id(url), title or "")


def _shown_title(url: str, title: str, abs_path: str) -> str:
    return (
        title
        or (_index().get(video_id(url)) or {}).get("title")
        or Path(abs_path).stem.replace("_", " ")
    )


def transcript(url: str, title: str = "") -> dict:
    """A video's transcript, segment by segment with its times and speakers,
    and its summaries, newest first."""
    from whisper_tm.core import summarize

    abs_path = _find(url, title)
    if not abs_path:
        raise IngestError("This video is not in the library")
    base, name = os.path.split(abs_path)
    segments = summarize.load_segments(name, media_dir=base) or []
    sums = summarize.load_summaries(name, media_dir=base) or []
    return {
        "abs_path": abs_path,
        "open": open_link(abs_path),
        "title": _shown_title(url, title, abs_path),
        "segments": [
            {
                "start": round(float(sg.get("start") or 0), 2),
                "end": round(float(sg.get("end") or 0), 2),
                "text": (sg.get("text") or "").strip(),
                "speaker": "" if sg.get("speaker") in (None, "", "—") else str(sg["speaker"]),
            }
            for sg in segments
            if (sg.get("text") or "").strip()
        ],
        "summaries": [
            {
                "text": x.get("text") or "",
                "name": x.get("prompt") or "",
                "at": x.get("timestamp") or 0,
                "model": x.get("model") or "",
            }
            for x in reversed(sums)
            if x.get("text")
        ],
    }


def _chats_path() -> Path:
    return Path(config.data_dir) / "ingest_chats.json"


def _chats() -> dict:
    """Which conversation is about which file."""
    try:
        d = json.loads(_chats_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def ask(url: str, title: str = "", new: bool = False) -> dict:
    """A conversation about a video: its transcript and newest summary go
    along with every question (core/chat.py — in the system message, where
    the history's trimming doesn't reach them). The same video asked about
    again gets its conversation back, unless ``new``."""
    from whisper_tm.core import chat, summarize

    abs_path = _find(url, title)
    if not abs_path:
        raise IngestError("This video is not in the library")
    base, name = os.path.split(abs_path)
    segments = summarize.load_segments(name, media_dir=base)
    if not segments:
        raise IngestError("This video is not transcribed yet")
    with _chats_lock:
        index = _chats()
        cid = index.get(abs_path)
        if cid and not new:
            try:
                chat.load(cid)
                return {"conversation": cid, "fresh": False}
            except chat.ChatError:
                pass  # deleted in the app
        shown = _shown_title(url, title, abs_path)
        text = summarize.format_transcript(segments, include_speakers=True, include_timestamps=True)
        sums = summarize.load_summaries(name, media_dir=base) or []
        summary = (sums[-1].get("text") or "") if sums else ""
        context = (
            f'A video is attached to this conversation — "{shown}" ({url}).\n'
            + (f"\nIts summary:\n\n{summary}\n" if summary else "")
            + "\nIts transcript — each line with the time it is said at [H:MM:SS] "
            "and, where known, who speaks:\n\n"
            + text[:VIDEO_CHARS]
            + (
                "\n\n[The transcript goes on — only its first part is attached.]"
                if len(text) > VIDEO_CHARS
                else ""
            )
        )
        conv = chat.new_conversation()
        conv["title"] = ("About: " + shown)[:80]
        conv["attached"] = {
            "kind": "video",
            "id": abs_path,
            "title": shown,
            "url": url,
            "text": context,
        }
        hello = (
            f"The video **{shown}** is attached — its transcript"
            + (" and its summary" if summary else "")
            + ". Ask about it; a time like [0:04:12] jumps to the moment."
        )
        conv["turns"] = [
            {"role": "assistant", "at": time.time(), "steps": [], "rounds": [hello], "text": hello}
        ]
        chat.save(conv)
        index[abs_path] = conv["id"]
        tmp = _chats_path().with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_chats_path())
    return {"conversation": conv["id"], "fresh": True}

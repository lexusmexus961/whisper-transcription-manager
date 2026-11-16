"""
What the LLM queue does for each task type.

Separate from core/llm_queue.py so the queue itself doesn't need to know
about transcripts, database tables, or prompts — it just calls
``run(entry, queue)``.

Each task reports progress and **partial results** to the queue: the
UI can expand the run and follow individual sections as they happen
while it's still running.
"""

from __future__ import annotations

import json
import logging
import time

from whisper_tm.config import config
from whisper_tm.core import llm_client
from whisper_tm.core.llm import correction as corr
from whisper_tm.core.llm import translation as trans

logger = logging.getLogger(__name__)


def _db():
    from whisper_tm.extensions import db

    return db


class _QueueJob:
    """Adapter: chunk runners expect a job object with ``update()``
    and ``cancel`` — here both point to the queue entry."""

    def __init__(self, queue, entry_id: int, basis: int = 0, span: int = 100):
        self._q = queue
        self._id = entry_id
        self._basis = basis
        self._span = span

    def update(self, **kw):
        if "message" in kw or "progress" in kw:
            p = kw.get("progress")
            percent = self._basis + int(p / 100 * self._span) if isinstance(p, int) else self._basis
            self._q.progress(self._id, percent, kw.get("message", "") or "")

    @property
    def cancel(self) -> bool:
        return self._q.abort_requested(self._id)


def _load_segments(filename: str, language: str = "", *, named: bool = False) -> list:
    """Segments of the desired language version (empty = original).
    ``named``: speakers by their names from the file view — for a summary,
    never for a correction or translation, which write the segments back."""
    from whisper_tm.api.transcription import _load_transcript_data
    from whisper_tm.core.summarize import named_speakers

    tdata = _load_transcript_data(filename)
    if not tdata:
        raise RuntimeError("Transcript not found")
    if not tdata.get("segments"):
        raise RuntimeError("The transcript is empty — the recording has no speech")
    segments = tdata["segments"]
    if named:
        segments = named_speakers(segments, tdata.get("speaker_names"))
    if not language:
        return segments
    db = _db()
    track = db.get_translation(filename, language, media_dir=str(config.media_dir)) if db else None
    if not track:
        raise RuntimeError(f"Translation '{language}' not found")
    voice_map = {t["id"]: t["text"] for t in track if isinstance(t, dict)}
    return [
        {**s, "text": voice_map.get(i, s.get("text", "")), "words": None}
        for i, s in enumerate(segments)
    ]


# ══════════════════════════════════════════════════════════════════════════
#  Summarization
# ══════════════════════════════════════════════════════════════════════════


def _summary(entry: dict, queue) -> dict:
    from whisper_tm.core import summarize as s

    p = entry["params"]
    filename = entry["filename"]
    # ``media_dir``: the file's own folder when it is not below the open one
    # — a video fetched by the browser extension lies in the upload folder
    media_dir = p.get("media_dir") or None
    if media_dir:
        segments = s.load_segments(filename, media_dir=media_dir, named=True)
        if segments is None:
            raise RuntimeError("Transcript not found")
        if not segments:
            raise RuntimeError("The transcript is empty — the recording has no speech")
    else:
        segments = _load_segments(filename, p.get("source") or "", named=True)
    system_prompt = p.get("system_prompt")
    if system_prompt is None:  # None = use preset/default
        system_prompt = s.preset_prompt(p.get("preset") or "")
    # The language it is wanted in, whatever the transcript speaks
    system_prompt += s.language_rule(p.get("language") or "")
    t0 = time.time()

    pieces: list[str] = []
    last_write = 0.0
    for ev in s.summary_stream(
        segments,
        system_prompt=system_prompt,
        user_prompt=p.get("prompt") or "",
        metadata=p.get("metadata") or {"filename": filename},
        model=p.get("model") or "",
        include_speakers=p.get("include_speakers", False),
        include_timestamps=p.get("include_timestamps", True),
        part_callback=lambda part: queue.partial_result(entry["id"], part),
        progress=lambda pr, msg: queue.progress(entry["id"], pr, msg),
        abort=lambda: queue.abort_requested(entry["id"]),
    ):
        if ev.get("text"):
            pieces.append(ev["text"])
            # Live write — at most ~1x/second so the UI sees the text
            # grow without flooding the database.
            now = time.monotonic()
            if now - last_write >= 1.0:
                last_write = now
                queue.partial_text(entry["id"], "".join(pieces))
                queue.progress(entry["id"], 90, "Writing result…")

    answer = "".join(pieces).strip()
    if not answer:
        raise RuntimeError("Empty response from LLM")
    queue.partial_text(entry["id"], answer)

    saved = False
    if p.get("save", True):
        entry = {
            "text": answer,
            "prompt": p.get("label") or p.get("preset") or "",
            "timestamp": time.time(),
            # No model = the task assignment chose one; record that one
            "model": p.get("model") or llm_client.task_default_model("summary"),
            "auto": bool(p.get("auto")),
        }
        # One writer at a time — a "Save" in the dialog finishing alongside
        # must not lose this one, nor this one it (summarize.change_summaries)
        saved, _ = s.change_summaries(filename, lambda lst: lst.append(entry), media_dir=media_dir)

    return {
        "text": answer,
        "chars": len(answer),
        "duration": round(time.time() - t0, 1),
        "saved": saved,
        "source": p.get("source") or "",
    }


# ══════════════════════════════════════════════════════════════════════════
#  Correction / Translation
# ══════════════════════════════════════════════════════════════════════════


def _correction(entry: dict, queue) -> dict:
    p = entry["params"]
    filename = entry["filename"]
    segments = _load_segments(filename)
    job = _QueueJob(queue, entry["id"])
    result = corr.build_suggestions(
        segments,
        model=p.get("model") or "",
        only_marked=p.get("only_marked", True),
        score_threshold=p.get("score_threshold"),
        logprob_threshold=p.get("logprob_threshold"),
        job=job,
        part_callback=lambda t: queue.partial_result(entry["id"], t),
    )
    model = p.get("model") or llm_client.task_default_model("correction")
    result["model"] = model

    run_id = 0
    db = _db()
    if db and (result["suggestions"] or result["discarded"] or result["checked"]):
        try:
            run_id = db.save_llm_edit_run(
                filename, result, model=model, status="pending", media_dir=str(config.media_dir)
            )
        except Exception as e:
            logger.warning("Correction run not savable: %s", e)
    return {
        "run_id": run_id,
        "suggestions": len(result["suggestions"]),
        "discarded": len(result["discarded"]),
        "checked": len(result["checked"]),
        "confirmed": result.get("confirmed", 0),
        "chunks": result["chunks"],
        "chunks_failed": result["chunks_failed"],
        "marked_segments": result["marked_segments"],
        "duration": result["duration"],
        "confidence_data": result.get("confidence_data", True),
        # Without confidences "only flagged" checks everything — worth
        # a note only when flagged ones were asked for
        "only_marked": bool(p.get("only_marked", True)),
    }


def _translation(entry: dict, queue) -> dict:
    p = entry["params"]
    filename = entry["filename"]
    language = (p.get("language") or "").strip()
    if not language:
        raise RuntimeError("No target language specified")
    segments = _load_segments(filename)
    job = _QueueJob(queue, entry["id"])
    t0 = time.time()
    result = trans.translate(
        segments,
        language,
        model=p.get("model") or "",
        job=job,
        part_callback=lambda t: queue.partial_result(entry["id"], t),
    )
    track = result["translations"]
    if not track:
        raise RuntimeError("Translation returned no segments")
    # Each entry carries its segment's times: what a later run with gaps
    # checks before it takes an entry over (_same_segment)
    track = [{**t, **_times(segments, int(t["id"]))} for t in track]

    model = p.get("model") or llm_client.task_default_model("translation")
    db = _db()
    missing = list(result["missing"])
    kept = 0
    if missing and db:
        # A run with gaps must not throw away a complete track: what it
        # missed keeps the language's earlier translation (otherwise a few
        # chunks lost to a server hiccup would replace a full track).
        try:
            before = db.get_translation(filename, language, media_dir=str(config.media_dir)) or []
        except Exception as e:
            logger.warning("Earlier translation not readable: %s", e)
            before = []
        earlier = {
            int(t["id"]): t for t in before if isinstance(t, dict) and t.get("id") is not None
        }
        # Only entries that still belong to the same segment: after a
        # re-transcription id 4 is another sentence
        fill = [
            earlier[i] for i in missing if i in earlier and _same_segment(earlier[i], segments, i)
        ]
        if fill:
            kept = len(fill)
            filled = {int(t["id"]) for t in fill}
            missing = [i for i in missing if i not in filled]
            track = sorted(list(track) + fill, key=lambda t: int(t["id"]))
            logger.info("Translation %s: %d gaps filled from the earlier track", language, kept)
    saved = False
    if db:
        try:
            saved = db.replace_translation(
                filename, language, track, model=model, media_dir=str(config.media_dir)
            )
        except Exception as e:
            logger.warning("Translation could not be saved to DB: %s", e)
    _back_up_translations_json(filename, language, track, model)
    return {
        "language": language,
        "segments": len(track),
        "total": len(segments),
        "missing": len(missing),
        "missing_ids": missing[:50],
        "kept_from_before": kept,
        "chunks": result["chunks"],
        "chunks_failed": result["chunks_failed"],
        "duration": round(time.time() - t0, 1),
        "db": saved,
    }


def _times(segments: list, i: int) -> dict:
    """``{"start", "end"}`` of segment ``i``, rounded — {} if it has none."""
    try:
        seg = segments[i]
        return {"start": round(float(seg["start"]), 2), "end": round(float(seg["end"]), 2)}
    except (IndexError, KeyError, TypeError, ValueError):
        return {}


def _same_segment(entry: dict, segments: list, i: int) -> bool:
    """Does a saved translation entry still belong to segment ``i``?

    Ids are segment indices, and a re-transcription renumbers the sentences:
    only an entry whose stored times match the segment's today counts. An
    entry saved before the times were stored has none — it is not taken.
    """
    now = _times(segments, i)
    if not now or entry.get("start") is None or entry.get("end") is None:
        return False
    try:
        return (
            abs(float(entry["start"]) - now["start"]) < 0.011
            and abs(float(entry["end"]) - now["end"]) < 0.011
        )
    except (TypeError, ValueError):
        return False


def _back_up_translations_json(filename, language, track, model) -> None:
    """JSON backup like with the summaries (DB primary, JSON heals)."""
    try:
        safe = filename.replace("/", "_").replace("\\", "_")
        fp = config.transcript_dir / f"{safe}_translations.json"
        all_items = {}
        if fp.exists():
            try:
                with open(fp, encoding="utf-8") as f:
                    all_items = json.load(f) or {}
            except (OSError, json.JSONDecodeError):
                all_items = {}
        all_items[language] = {"model": model, "created_at": time.time(), "segments": track}
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("Translation JSON not writable: %s", e)


# ══════════════════════════════════════════════════════════════════════════
#  Analysis: one task over many recordings
# ══════════════════════════════════════════════════════════════════════════


def _analysis(entry: dict, queue) -> dict:
    from whisper_tm.core import analysis
    from whisper_tm.core.llm_queue import TaskRefused

    aid = str((entry.get("params") or {}).get("analysis_id") or "")
    try:
        return analysis.run(aid, queue, entry["id"])
    except analysis.AnalysisError as e:
        raise TaskRefused(str(e)) from e


# ══════════════════════════════════════════════════════════════════════════
#  Deep search: every passage of a situation, checked one by one
# ══════════════════════════════════════════════════════════════════════════


def _deep_search(entry: dict, queue) -> dict:
    from whisper_tm.rag import deep_search

    did = str((entry.get("params") or {}).get("deep_id") or "")
    try:
        return deep_search.run(did, queue, entry["id"])
    except deep_search.DeepSearchError as e:
        raise RuntimeError(str(e)) from e


def _media_describe(entry, queue):
    from whisper_tm.media import describe

    return describe.run(entry, queue)


def _best_moments(entry, queue):
    from whisper_tm.core import best_moments

    return best_moments.run(entry, queue)


_HANDLER = {
    "summary": _summary,
    "correction": _correction,
    "translation": _translation,
    "analysis": _analysis,
    "deep_search": _deep_search,
    "media_describe": _media_describe,
    "best_moments": _best_moments,
}

#: Kinds that run without a language model too: a deep search then ranks
#: its passages without checking them, and says so.
_WITHOUT_LLM = {"deep_search"}


def execute(entry: dict, queue) -> dict:
    """Process one queue entry (blocking, in the worker thread)."""
    handler = _HANDLER.get(entry["kind"])
    if not handler:
        raise RuntimeError(f"Unknown task type: {entry['kind']}")
    if entry["kind"] not in _WITHOUT_LLM and not config.get("llm", "endpoint", default=""):
        raise RuntimeError("LLM not configured (Settings → AI models)")
    try:
        return handler(entry, queue)
    except llm_client.LlmError as e:
        raise RuntimeError(str(e)) from e

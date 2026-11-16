"""
Transcript summarization — the core, independent of the browser.

Originally everything lived in the streaming route ``POST /api/summarize``
(api/llm.py): load segments, build prompt, shape request, stream. This way
summarization was only possible as long as a browser tab was open.

For automatic summarization after transcription (option in the
add/batch dialog) the queue needs the same work WITHOUT streaming and
WITHOUT request context. That's why there is one shared implementation here:

* ``chat_request()``  — endpoint/header/payload (Ollama native OR
  OpenAI-compatible); the route uses it with ``stream=True``, the queue with
  ``stream=False``.
* ``make_summary()`` — complete summarization of a file, blocking,
  result is saved (DB + JSON backup).

The load/save/format helpers were moved here from api/llm.py; the
routes import them from here, so they exist exactly once.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import requests

from whisper_tm.config import config
from whisper_tm.core import llm_client

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Load segments + format transcript
# ══════════════════════════════════════════════════════════════════════════


def named_speakers(segments, names) -> list:
    """The segments with each speaker label replaced by the name given in the
    file view (``speaker_names``: SPEAKER_00 → "Alice"). A summary called
    people "SPEAKER_00" although the transcript showed their names."""
    if not isinstance(names, dict) or not names:
        return segments
    out = []
    for seg in segments:
        name = names.get(seg.get("speaker")) if isinstance(seg, dict) else None
        out.append({**seg, "speaker": str(name)} if name else seg)
    return out


def load_segments(filename: str, media_dir=None, *, named: bool = False):
    """Load segments from DB or JSON (None = no transcript found).

    ``media_dir`` is the root of THIS file — for entries from a
    different directory (dir_map/queue) it is not config.media_dir.
    ``named``: speakers by their names from the file view (named_speakers).
    """
    from whisper_tm.core.helpers import find_transcript
    from whisper_tm.extensions import db

    def segments_of(data):
        segs = data.get("segments") or []
        return named_speakers(segs, data.get("speaker_names")) if named else segs

    # DB first
    if db:
        try:
            data = db.get_transcript(filename, media_dir=str(media_dir or config.media_dir))
            if data and data.get("segments"):
                return segments_of(data)
        except Exception:
            pass

    # JSON fallback
    tp = find_transcript(filename)
    if tp:
        try:
            with open(tp, encoding="utf-8") as f:
                data = json.load(f)
            return segments_of(data)
        except (OSError, json.JSONDecodeError) as e:
            # Corrupted/unreadable file → clean error instead of 500
            logger.warning("Transcript JSON not readable (%s): %s", tp, e)
            return None

    return None


def fmt_time(seconds) -> str:
    """Seconds as H:MM:SS — the format of the segments' own ``start_fmt``,
    so every timestamp the model sees looks alike (templates ask it to copy
    them exactly)."""
    from whisper_tm.core.helpers import format_ts

    return format_ts(seconds)


def format_transcript(segments, include_speakers=False, include_timestamps=True) -> str:
    """Format segments as readable text."""
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        prefix_parts = []
        if include_timestamps:
            start = seg.get("start_fmt") or fmt_time(seg.get("start", 0))
            prefix_parts.append(start)
        if include_speakers:
            speaker = seg.get("speaker", "")
            if speaker and speaker != "—":
                prefix_parts.append(speaker)

        if prefix_parts:
            lines.append(f"[{' | '.join(prefix_parts)}] {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def build_messages(system_prompt, user_prompt, transcript_text, metadata=None):
    """Build message array for the LLM.

    The system message is the template (shown in the dialog as it is sent).
    The user message: metadata block (if any), the transcript, and the
    additional instruction LAST — after tens of thousands of transcript
    tokens an instruction in front of them was the part the model followed
    least, and at the end it leaves the transcript prefix unchanged, so a
    rerun with a different instruction can reuse the server's prompt cache.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    parts = []

    if metadata:
        meta_lines = []
        if metadata.get("filename"):
            meta_lines.append(f"File: {metadata['filename']}")
        if metadata.get("duration"):
            meta_lines.append(f"Duration: {metadata['duration']}")
        if metadata.get("language"):
            meta_lines.append(f"Language: {metadata['language']}")
        if metadata.get("model"):
            meta_lines.append(f"Whisper model: {metadata['model']}")
        if metadata.get("date"):
            meta_lines.append(f"Transcribed: {metadata['date']}")
        if metadata.get("speakers"):
            meta_lines.append(f"Speakers: {metadata['speakers']}")
        if meta_lines:
            parts.append("Metadata:\n" + "\n".join(meta_lines))

    parts.append(f"Transcript:\n\n{transcript_text}")

    if user_prompt:
        parts.append(f"Additional instruction for this run:\n{user_prompt}")

    messages.append({"role": "user", "content": "\n\n---\n\n".join(parts)})
    return messages


# ══════════════════════════════════════════════════════════════════════════
#  Chat request (one place for both endpoint dialects)
# ══════════════════════════════════════════════════════════════════════════


def chat_request(messages: list, *, model: str = "", stream: bool = True):
    """Build endpoint, header and payload for a chat call.

    Returns ``(api_url, headers, payload, ollama_native)``. This
    delegates to llm_client.build_chat_payload — the ONE place for
    both endpoint dialects (signature stays the same for existing callers).
    """
    return llm_client.build_chat_payload(messages, model=model, stream=stream, task="summary")


def _answer_text(data: dict, ollama_native: bool) -> str:
    """Extract text from a NON-streamed chat response."""
    if ollama_native:
        return (data.get("message") or {}).get("content", "") or ""
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# ══════════════════════════════════════════════════════════════════════════
#  Map-reduce for transcripts exceeding context budget
#  (docs/LLM-PIPELINE.md §5.1)
# ══════════════════════════════════════════════════════════════════════════


def _needs_map_reduce(
    transcript_text: str, system_prompt: str, user_prompt: str
) -> tuple[bool, int]:
    """Does transcript + prompts + response fit in ONE call?

    Returns:
        (map_reduce_needed, context) — context 0 means unknown, then
        budget.chunk_budget calculates with the conservative fallback.
    """
    from whisper_tm.core.llm.budget import CONTEXT_FALLBACK, estimate_tokens

    context = llm_client.effective_context("summary")
    effective = context or CONTEXT_FALLBACK
    max_tokens = config.get("llm", "max_tokens", default=4096)
    margin = float(config.get("llm", "chunking", "safety_margin", default=0.10))
    input_tokens = (
        estimate_tokens(transcript_text)
        + estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt)
        + 200
    )  # meta/scaffolding
    fits = input_tokens + min(max_tokens, effective // 2) <= int(effective * (1.0 - margin))
    return (not fits), context


def _map_chunks(segments, include_speakers: bool, include_timestamps: bool, task: str = ""):
    """Build chunk plan for the map step (``task``: the template it collects
    for — part of every map call, so part of the budget)."""
    from whisper_tm.core.llm import prompts
    from whisper_tm.core.llm.budget import chunk_budget, estimate_tokens
    from whisper_tm.core.llm.chunking import split_segments

    lines = [format_transcript([s], include_speakers, include_timestamps) or "…" for s in segments]
    context = llm_client.effective_context("summary")
    margin = float(config.get("llm", "chunking", "safety_margin", default=0.10))
    map_max = 1200  # dense partial summary, not a novel
    budget = chunk_budget(
        system_tokens=estimate_tokens(prompts.map_system(task)),
        instruction_tokens=96,
        answer_reserve=map_max,
        context=context,
        margin=margin,
    )
    overlap = int(config.get("llm", "chunking", "overlap_segments", default=3))
    return split_segments(segments, lines, budget, overlap_segments=overlap), map_max


def _map_part(chunk, segments, total: int, model: str, map_max: int, task: str = "") -> str:
    """Condense one excerpt into notes for ``task`` (map step)."""
    from whisper_tm.core.llm import prompts

    from_value = fmt_time(segments[chunk.from_seg].get("start", 0))
    to_value = fmt_time(segments[chunk.to_seg].get("end", 0))
    text = "\n".join(chunk.lines)
    if chunk.context_before:
        text = (
            "(Continuation from previous section:)\n"
            + "\n".join(line for _i, line in chunk.context_before)
            + "\n---\n"
            + text
        )
    user = prompts.MAP_USER_TEMPLATE.format(
        number=chunk.index + 1, total=total, from_value=from_value, to_value=to_value, text=text
    )
    answer, _usage = llm_client.chat(
        [
            {"role": "system", "content": prompts.map_system(task)},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=map_max,
        thinking=False,
        task="summary",
    )
    return f"Part {chunk.index + 1} ({from_value}–{to_value}):\n{answer.strip()}"


def _condense_parts(parts: list[str], model: str, map_max: int, task: str = "") -> list[str]:
    """Pair up partial summaries until they fit together into the reduce budget
    (recursive reduce for edge cases)."""
    from whisper_tm.core.llm import prompts
    from whisper_tm.core.llm.budget import CONTEXT_FALLBACK, estimate_tokens

    context = llm_client.effective_context("summary") or CONTEXT_FALLBACK
    max_tokens = config.get("llm", "max_tokens", default=4096)
    budget = int(context * 0.9) - min(max_tokens, context // 2) - 1000

    rounds = 0
    while sum(estimate_tokens(t) for t in parts) > budget and len(parts) > 1 and rounds < 4:
        rounds += 1
        new_value = []
        group_parts: list[str] = []
        group_tokens = 0
        # Room for at least two map answers: below that (an unknown
        # context's 8192 fallback gives 2048) no two parts ever fit one
        # group, and every "merge" sent a single part alone — a paid call
        # per part and round that shortened nothing
        target = max(2048, budget // 2, 2 * map_max + 256)
        for t in parts:
            tt = estimate_tokens(t)
            # A group is condensed only with two parts or more
            if len(group_parts) >= 2 and group_tokens + tt > target:
                together = "\n\n".join(group_parts)
                answer, _u = llm_client.chat(
                    [
                        {"role": "system", "content": prompts.map_system(task)},
                        {"role": "user", "content": prompts.CONDENSE_USER + together},
                    ],
                    task="summary",
                    model=model,
                    max_tokens=map_max,
                    thinking=False,
                )
                new_value.append(answer.strip())
                group_parts, group_tokens = [], 0
            group_parts.append(t)
            group_tokens += tt
        if group_parts:
            # Bundle remaining uncompressed — the next round will compress
            # it if the total still exceeds the budget.
            new_value.append("\n\n".join(group_parts))
        parts = new_value
    return parts


def summary_stream(
    segments,
    *,
    system_prompt: str,
    user_prompt: str,
    metadata: dict | None,
    model: str,
    include_speakers: bool,
    include_timestamps: bool,
    part_callback=None,
    progress=None,
    abort=None,
):
    """Generator over summarization events.

    Events: {"status": str}   Progress (map phase, chunk plan)
            {"text": str}     Response text (streamed)

    Optional callbacks for the LLM queue (UI expands progress and reads
    sections as the run progresses):
      part_callback(dict)      for each completed map section
      progress(percent, msg)
      abort() -> bool        True → InterruptedError
    Errors come up as llm_client.LlmError exceptions.
    """

    def _check_abort():
        if abort is not None and abort():
            raise InterruptedError("Cancelled")

    transcript_text = format_transcript(segments, include_speakers, include_timestamps)
    if not transcript_text.strip():
        raise llm_client.LlmAnswerError("Transcript is empty")

    needed, _context = _needs_map_reduce(transcript_text, system_prompt, user_prompt)
    _check_abort()
    if not needed:
        if progress:
            progress(10, "Creating summary…")
        messages = build_messages(system_prompt, user_prompt, transcript_text, metadata)
        for piece in llm_client.chat_stream(messages, model=model, task="summary"):
            _check_abort()
            yield {"text": piece}
        return

    # ── Map phase ─────────────────────────────────────────────────────
    import time as _time

    from whisper_tm.core.llm import prompts

    # What the notes are collected for: the template and this run's
    # additional instruction — both reach the reduce step as they are
    task = "\n\n".join(t for t in ((system_prompt or "").strip(), (user_prompt or "").strip()) if t)
    chunks, map_max = _map_chunks(segments, include_speakers, include_timestamps, task)
    yield {
        "status": f"Transcript too large for one call — "
        f"{len(chunks)} sections will be pre-compressed"
    }
    _t_map = _time.monotonic()
    parts = []
    for chunk in chunks:
        _check_abort()
        message = f"Summarizing section {chunk.index + 1}/{len(chunks)}…"
        yield {"status": message}
        if progress:
            # Map phase takes 5–80% of progress, reduce takes the remainder
            progress(5 + int(chunk.index / max(1, len(chunks)) * 75), message)
        _t_chunk = _time.monotonic()
        text = _map_part(chunk, segments, len(chunks), model, map_max, task)
        parts.append(text)
        duration = _time.monotonic() - _t_chunk
        logger.info(
            "⏱ Map section %d/%d: %.1fs (%d segments)",
            chunk.index + 1,
            len(chunks),
            duration,
            chunk.to_seg - chunk.from_seg + 1,
        )
        if part_callback:
            part_callback(
                {
                    "number": chunk.index + 1,
                    "total": len(chunks),
                    "from_seg": chunk.from_seg,
                    "to_seg": chunk.to_seg,
                    "from_value": fmt_time(segments[chunk.from_seg].get("start", 0)),
                    "to_value": fmt_time(segments[chunk.to_seg].get("end", 0)),
                    "tokens": chunk.tokens,
                    "duration": round(duration, 1),
                    "text": text,
                    # LLM prose — UI is allowed to render it as markdown.
                    # Correction/translation parts do NOT set this flag
                    # (their text is indented line log).
                    "markdown": True,
                }
            )
    logger.info("⏱ Map phase total: %d sections in %.1fs", len(chunks), _time.monotonic() - _t_map)

    parts = _condense_parts(parts, model, map_max, task)

    # ── Reduce phase: user prompt over partial summaries ────
    _check_abort()
    yield {"status": "Creating overall summary…"}
    if progress:
        progress(85, "Creating overall summary…")
    reduce_text = prompts.REDUCE_INTRO_TEMPLATE.format(count=len(parts), parts="\n\n".join(parts))
    messages = build_messages(system_prompt, user_prompt, reduce_text, metadata)
    # build_messages sets "Transcript:" prefix — in reduce this is still
    # correct semantically: the parts REPRESENT the transcript.
    for piece in llm_client.chat_stream(messages, model=model, task="summary"):
        _check_abort()
        yield {"text": piece}


# ══════════════════════════════════════════════════════════════════════════
#  Saved summaries (DB primary, JSON as backup/fallback)
# ══════════════════════════════════════════════════════════════════════════


def summaries_path(filename: str) -> Path:
    """Path to summaries JSON file.

    Uses the *complete* filename including extension — otherwise files like
    ``a.mp4`` and ``a.wav`` (same ``stem``) would collide in the same file.
    """
    safe = filename.replace("/", "_").replace("\\", "_")
    return config.transcript_dir / f"{safe}_summaries.json"


def load_summaries_json(filename: str) -> list:
    """Load summaries from JSON file (fallback)."""
    fp = summaries_path(filename)
    if not fp.exists():
        return []
    try:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_summaries(filename: str, media_dir=None, strict: bool = False) -> list:
    """Load saved summaries. DB-first, JSON as fallback.

    On JSON hit without DB entry the DB is backfilled
    (self-healing) — on next access the DB responds directly.

    ``strict``: a DB read error is raised instead of falling back to the
    JSON — for a caller that writes the list back (:func:`change_summaries`):
    the backup may be older, and saving it would replace the database's
    list with it.
    """
    from whisper_tm.extensions import db

    if db:
        try:
            rows = db.get_summaries(filename, media_dir=str(media_dir or config.media_dir))
            if rows is not None:
                return rows
        except Exception as e:
            if strict:
                raise
            logger.warning("DB get_summaries failed, reading the JSON backup: %s", e)

    data = load_summaries_json(filename)
    if data and db:
        try:
            db.replace_summaries(
                filename,
                data,  # DB heal
                media_dir=str(media_dir or config.media_dir),
            )
        except Exception as e:
            logger.debug("DB summaries heal failed: %s", e)
    return data


def save_summaries(filename: str, summaries: list, media_dir=None) -> bool:
    """Save summaries: DB (primary source) + JSON file (backup).

    Returns ``True`` once at least one source (DB or JSON)
    has persisted successfully. Only if *both* fail does the
    summary get lost → ``False``, so caller can report honestly.
    """
    from whisper_tm.extensions import db

    db_ok = False
    if db:
        try:
            db.replace_summaries(filename, summaries, media_dir=str(media_dir or config.media_dir))
            db_ok = True
        except Exception as e:
            logger.debug("DB replace_summaries failed: %s", e)
    json_ok = False
    try:
        fp = summaries_path(filename)
        fp.parent.mkdir(parents=True, exist_ok=True)
        # Through a temporary file: a crash in the middle of the write must
        # not leave half a backup behind
        tmp = fp.with_name(fp.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        os.replace(tmp, fp)
        json_ok = True
    except OSError as e:
        logger.warning("Summaries JSON not writable: %s", e)
    return db_ok or json_ok


#: One change to the saved summaries at a time. Every writer reads the list,
#: changes it and writes it back; two at once (the queue's summary and a
#: "Save" in the dialog) kept only one of them. Changes are
#: rare and short — one lock for all files is enough.
_change_lock = threading.Lock()


def change_summaries(filename: str, change, media_dir=None):
    """Read the saved summaries, let ``change(list)`` change them in place,
    save — one writer at a time, and never from a failed read.

    Returns ``(saved, result)``: whether DB or JSON took the list, and what
    ``change`` returned. ``change`` may raise to leave everything as it was.
    A DB read error is raised (``load_summaries(strict=True)``) instead of
    writing the JSON backup's list over the database.
    """
    with _change_lock:
        summaries = load_summaries(filename, media_dir=media_dir, strict=True)
        result = change(summaries)
        return save_summaries(filename, summaries, media_dir=media_dir), result


# ══════════════════════════════════════════════════════════════════════════
#  Automatic summarization (queue post-processing)
# ══════════════════════════════════════════════════════════════════════════


def preset_prompt(preset: str) -> str:
    """System prompt of a preset — empty/unknown name = config default.

    Presets are in config a flat dict ``{name: system-prompt}``.
    """
    presets = config.get("llm", "presets", default={}) or {}
    if preset and isinstance(presets, dict) and presets.get(preset):
        return presets[preset]
    return config.get("llm", "system_prompt", default="")


#: The languages a summary can be written in — and a transcription be told
#: to expect (the codes are Whisper's). The order of the app's own dialog
#: (static/js/txoptions.js): the common ones first.
LANGUAGES: dict[str, str] = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "tr": "Turkish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "hi": "Hindi",
}


def language_rule(language: str) -> str:
    """What a template is told when the summary is wanted in one language.

    Every template ends on "Write in the language of the transcript" — an
    English video then gets an English summary. A reader can ask for one
    fixed language (e.g. German), whatever the video speaks. Empty or
    unknown = the template's own rule.
    """
    name = LANGUAGES.get((language or "").strip().lower())
    return (
        f"\n\nWrite the summary in {name}, whatever the language of the transcript." if name else ""
    )


def make_summary(
    filename: str,
    *,
    preset: str = "",
    model: str = "",
    segments=None,
    metadata=None,
    save: bool = True,
    media_dir=None,
    language: str = "",
    prompt: str = "",
) -> str:
    """Summarize a file and (by default) save the result.

    ``prompt``: an own instruction instead of the preset's ("Custom…" in the
    browser extension's side panel); the summary is listed as "Custom".

    Blocking, no streaming — designed for the queue: the browser does not need
    to be open. Raises on any failure (caller decides whether this fails the
    queue entry — it does NOT, see core/queue.py: a failed summary does not
    break a successful transcription).

    Returns the summary text.
    """
    endpoint = config.get("llm", "endpoint", default="")
    if not endpoint:
        raise RuntimeError("LLM not configured (Settings → AI models)")

    if segments is None:
        segments = load_segments(filename, media_dir=media_dir)
    if segments is None:
        raise RuntimeError("Transcript not found")

    text = format_transcript(segments, include_speakers=False, include_timestamps=True)
    if not text.strip():
        raise RuntimeError("Transcript is empty")

    system_prompt = (prompt or preset_prompt(preset)) + language_rule(language)
    needed, _ctx = _needs_map_reduce(text, system_prompt, "")
    if needed:
        # Transcript exceeds context → map-reduce over stream
        # generator (no browser; events are simply collected).
        pieces = []
        for ev in summary_stream(
            segments,
            system_prompt=system_prompt,
            user_prompt="",
            metadata=metadata or {"filename": filename},
            model=model,
            include_speakers=False,
            include_timestamps=True,
        ):
            if ev.get("text"):
                pieces.append(ev["text"])
            elif ev.get("status"):
                logger.info("Auto-summary %s: %s", filename, ev["status"])
        answer = "".join(pieces).strip()
        if not answer:
            raise RuntimeError("Empty response from LLM (map-reduce)")
    else:
        messages = build_messages(system_prompt, "", text, metadata or {"filename": filename})
        api_url, headers, payload, ollama_native = chat_request(messages, model=model, stream=False)

        # Generous read timeout: a local model can take minutes for a
        # long transcript. Connection establishment stays short.
        timeout = config.get("llm", "timeout", default=1800)
        from whisper_tm.core import local_key

        r = requests.post(
            api_url,
            headers=local_key.headers_for(api_url, headers),
            json=payload,
            timeout=(10, timeout),
        )
        if r.status_code != 200:
            raise RuntimeError(f"LLM error {r.status_code}: {r.text[:200]}")

        answer = _answer_text(r.json(), ollama_native).strip()
        if not answer:
            raise RuntimeError("Empty response from LLM")

    if save:
        entry = {
            "text": answer,
            "prompt": "Custom" if prompt else preset or "Automatic",
            "timestamp": time.time(),
            # NOT payload.get("model"): in the map-reduce branch there is no
            # payload — the access would crash there after complete LLM run
            # with UnboundLocalError.
            # No model = the summary task's assignment chose one
            "model": model or llm_client.task_default_model("summary"),
            "auto": True,
        }
        saved, _ = change_summaries(filename, lambda lst: lst.append(entry), media_dir=media_dir)
        if not saved:
            raise RuntimeError("Summary not saveable (DB and JSON both failed)")
    return answer

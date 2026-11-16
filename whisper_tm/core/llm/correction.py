"""
Confidence-guided transcript correction (docs/LLM-PIPELINE.md §5.2).

Process:
  1. Select segments whose confidence stands out (word score below threshold
     OR avg_logprob below threshold).
  2. Chunk transcript; only chunks with at least one flagged segment go to
     the LLM (mode "only marked") — remaining lines run as non-editable context.
  3. Uncertain words are marked in prompt: ⟨word|0.21⟩.
  4. LLM returns ONLY edits (JSON-forced): [{id, text, reason, confidence}].
  5. Each edit is validated server-side (ID allowed, length ratio, not identical,
     minimum confidence); a chunk wanting to edit too much (overcorrection
     signal) is completely rejected.

Edits are then SUGGESTIONS — applied only after review (or explicitly "accept
all"), see apply_edits_to().
"""

from __future__ import annotations

import logging
import re
import time

from whisper_tm.config import config
from whisper_tm.core import llm_client
from whisper_tm.core.llm import prompts
from whisper_tm.core.llm.budget import chunk_budget, estimate_tokens
from whisper_tm.core.llm.chunking import split_segments
from whisper_tm.core.llm.runner import run_chunks

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"⟨([^|⟩]*)\|[0-9.]+⟩")


def _cfg(name: str, default):
    return config.get("llm", "correction", name, default=default)


# ══════════════════════════════════════════════════════════════════════════
#  Selection + Prompt formatting
# ══════════════════════════════════════════════════════════════════════════


def is_suspicious(seg: dict, score_threshold: float, logprob_threshold: float) -> bool:
    """Judge segment by confidence thresholds."""
    if seg.get("avg_logprob") is not None and float(seg["avg_logprob"]) < logprob_threshold:
        return True
    for w in seg.get("words") or []:
        score = w.get("score")
        if score is not None and float(score) < score_threshold:
            return True
    return False


def choose_segments(
    segments: list[dict], *, score_threshold: float, logprob_threshold: float
) -> set[int]:
    """Indices of all flagged segments."""
    return {
        i for i, s in enumerate(segments) if is_suspicious(s, score_threshold, logprob_threshold)
    }


# Above this fraction of uncertain words, the WHOLE segment is marked (⚠)
# instead of individual words — CJK alignment often yields single-character
# "words" with score 0.00; word markers would flood the prompt (thousands of
# markers, several times the tokens).
SEGMENT_MARKER_SHARE = 0.4


def _uncertain_spans(text: str, words: list, score_threshold: float):
    """(start, end, min_score) spans of uncertain words in text.

    Searches each uncertain word from a running cursor (CJK has no spaces,
    punctuation attaches to words — reconstruction from word list would be
    wrong). Adjacent/overlapping finds merge into ONE span. Unfound words
    remain unmarked.
    """
    spans: list[list] = []
    cursor = 0
    for w in words:
        word = (w.get("word") or "").strip()
        score = w.get("score")
        if not word or score is None or float(score) >= score_threshold:
            continue
        pos = text.find(word, cursor)
        if pos < 0:
            continue
        end = pos + len(word)
        cursor = end
        # Merge if directly adjacent (or only whitespace between)
        if spans and not text[spans[-1][1] : pos].strip() and pos - spans[-1][1] <= 1:
            spans[-1][1] = end
            spans[-1][2] = min(spans[-1][2], float(score))
        else:
            spans.append([pos, end, float(score)])
    return spans


def mark_uncertain_words(seg: dict, score_threshold: float) -> str:
    """Segment text with ⟨…|score⟩ markers for uncertain passages.

    Adjacent uncertain words share one marker (with the lowest score).
    If more than SEGMENT_MARKER_SHARE of words are uncertain, no word-level
    marking is done — the caller sets the segment flag (⚠), see format_rows.
    """
    text = (seg.get("text") or "").strip()
    words = seg.get("words") or []
    if not text or not words or segment_uncertain(seg, score_threshold):
        return text

    spans = _uncertain_spans(text, words, score_threshold)
    out = []
    cursor = 0
    for start, end, score in spans:
        out.append(text[cursor:start])
        out.append(f"⟨{text[start:end]}|{score:.2f}⟩")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def segment_uncertain(seg: dict, score_threshold: float) -> bool:
    """True if so many words are uncertain that word markers would be pointless
    — then the whole segment is considered uncertain (⚠ flag)."""
    words = [
        w
        for w in (seg.get("words") or [])
        if (w.get("word") or "").strip() and w.get("score") is not None
    ]
    if not words:
        return False
    uncertain = sum(1 for w in words if float(w["score"]) < score_threshold)
    return uncertain / len(words) > SEGMENT_MARKER_SHARE


def format_lines(segments: list[dict], marked: set[int], score_threshold: float) -> list[str]:
    """Prompt lines: editable ``[id]``, context-only ``(id)``.

    ``[id]⚠`` = the whole segment is uncertain (instead of word markers).
    """
    lines = []
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker") or "—"
        sp = f" {speaker}:" if speaker != "—" else ""
        if i in marked:
            flag = "⚠" if segment_uncertain(seg, score_threshold) else ""
            lines.append(f"[{i}]{flag}{sp} {mark_uncertain_words(seg, score_threshold)}")
        else:
            lines.append(f"({i}){sp} {(seg.get('text') or '').strip()}")
    return lines


# ══════════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════════


def validate_edits(
    edits: list,
    segments: list[dict],
    allowed_ids: set[int],
    *,
    min_confidence: float,
    max_edit_share: float,
    checked_ok: list | None = None,
):
    """Check edits of a chunk.

    Returns:
        (accepted, rejected, confirmed)
        accepted    — actual correction suggestions
        rejected    — [(edit, reason)] for review display
        confirmed   — [ids] that the LLM explicitly reported as OK
                      (action="okay" or unchanged text). This is information,
                      not an error.
        On overcorrection (> max_edit_share of allowed IDs edited),
        the WHOLE chunk is rejected.
    """
    assumed, rejected = [], []
    # Explicitly reported as "checked, okay" IDs (compact list from LLM
    # response) — plus everything that came back unchanged.
    confirmed: list[int] = [
        i for i in (checked_ok or []) if isinstance(i, int) and i in allowed_ids
    ]
    seen: set[int] = set(confirmed)
    for e in edits or []:
        if not isinstance(e, dict):
            rejected.append((e, "not an object"))
            continue
        try:
            sid = int(e.get("id"))
        except (TypeError, ValueError):
            rejected.append((e, "id missing/invalid"))
            continue
        if sid not in allowed_ids:
            rejected.append((e, f"id {sid} not editable"))
            continue
        if sid in seen:
            rejected.append((e, f"id {sid} duplicate"))
            continue
        new_value = _MARKER_RE.sub(r"\1", str(e.get("text") or "")).strip()
        old_value = (segments[sid].get("text") or "").strip()
        if not new_value:
            rejected.append((e, "empty text"))
            continue
        if new_value == old_value:
            # Returned unchanged = same as "checked, okay"
            seen.add(sid)
            confirmed.append(sid)
            continue
        try:
            confidence = float(e.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            rejected.append((e, f"confidence {confidence:.2f} < {min_confidence:.2f}"))
            continue
        # Length ratio: a "correction" that halves/triples is a rephrase —
        # that's exactly what should not happen.
        ratio = len(new_value) / max(1, len(old_value))
        if not (0.3 <= ratio <= 3.0) and abs(len(new_value) - len(old_value)) > 12:
            rejected.append((e, f"length ratio {ratio:.2f}"))
            continue
        seen.add(sid)
        assumed.append(
            {
                "id": sid,
                "old_value": old_value,
                "new_value": new_value,
                "reason": str(e.get("reason") or "")[:300],
                "confidence": round(confidence, 2),
            }
        )

    # Overcorrection guard only from 5 editable segments — with 1–4, "all
    # edited" is quite plausible (e.g., exactly ONE marked segment in chunk)
    # so the ratio is not a signal.
    if len(allowed_ids) >= 5 and len(assumed) / len(allowed_ids) > max_edit_share:
        logger.warning(
            "Correction chunk rejected: %d/%d segments edited (overcorrection signal)",
            len(assumed),
            len(allowed_ids),
        )
        rejected.extend((a, "Chunk rejected: overcorrection") for a in assumed)
        assumed = []
    return assumed, rejected, confirmed


# ══════════════════════════════════════════════════════════════════════════
#  Main run
# ══════════════════════════════════════════════════════════════════════════


def build_suggestions(
    segments: list[dict],
    *,
    model: str = "",
    only_marked: bool = True,
    score_threshold: float | None = None,
    logprob_threshold: float | None = None,
    job=None,
    part_callback=None,
) -> dict:
    """Generate correction suggestions for a transcript (blocking).

    Returns (everything the review UI needs for full transparency):
        {"suggestions": [{id, old, new, reason, confidence}],
         "discarded": [{id, old, new, llm_reason, confidence,
                       rejected_because}],
         "checked": [{id, text, avg_logprob, uncertain_words}],
         "chunks": int, "chunks_failed": int,
         "marked_segments": int, "duration": float,
         "thresholds": {score, logprob}, "confidence_data": bool}
    """
    t_start = time.time()
    if score_threshold is None:
        score_threshold = float(_cfg("score_threshold", 0.40))
    if logprob_threshold is None:
        logprob_threshold = float(_cfg("logprob_threshold", -0.80))
    min_confidence = float(_cfg("min_confidence", 0.5))
    max_edit_share = float(_cfg("max_edit_share", 0.5))
    temperature = float(_cfg("temperature", 0.2))

    marked = choose_segments(
        segments, score_threshold=score_threshold, logprob_threshold=logprob_threshold
    )
    # Confidence data means: there are REAL word scores. A words list without
    # score fields (failed/missing alignment) doesn't count, nor does avg_logprob
    # alone — that's practically always ONE global value across all segments
    # so it never marks anything.
    # Without this fallback, the run ended after 1 ms with all zeros and looked
    # broken.
    has_confidence = any(
        any(w.get("score") is not None for w in (s.get("words") or [])) for s in segments
    )
    if only_marked and not has_confidence:
        # Without scores, "only flagged" would always be empty. Instead of
        # silently doing nothing, we check the whole transcript — the LLM then
        # corrects purely from context.
        logger.info(
            "Correction: no word scores in transcript — checking all %d segments from context",
            len(segments),
        )
        editable = {i for i, s in enumerate(segments) if (s.get("text") or "").strip()}
    elif only_marked:
        editable = marked
    else:
        editable = set(range(len(segments)))
    if not editable:
        return {
            "suggestions": [],
            "discarded": [],
            "checked": [],
            "chunks": 0,
            "chunks_failed": 0,
            "marked_segments": 0,
            "confirmed": 0,
            "duration": 0.0,
            "thresholds": {"score": score_threshold, "logprob": logprob_threshold},
            "confidence_data": has_confidence,
        }

    lines = format_lines(segments, editable, score_threshold)

    # Rules part of system prompt is replaceable via settings; the format
    # contract always follows in correction_system().
    system = prompts.correction_system(str(_cfg("system_prompt", "") or ""))
    system_tokens = estimate_tokens(system)
    context = llm_client.effective_context("correction")
    margin = float(config.get("llm", "chunking", "safety_margin", default=0.10))
    overlap = int(config.get("llm", "chunking", "overlap_segments", default=3))
    # Response ≈ 40% of input in worst case → input gets 1/1.5 of available
    # budget (remainder remains for response).
    available = chunk_budget(
        system_tokens=system_tokens,
        instruction_tokens=64,
        answer_reserve=0,
        context=context,
        margin=margin,
    )
    # Response can reach ≈ input size with many edits (full new text + reason
    # per segment) → input gets 1/2.2 of budget.
    input_budget = max(512, int(available / 2.2))
    # Chunk tasks: short timeout instead of hours cascade (see translation.py).
    timeout = min(int(config.get("llm", "timeout", default=1800)), 900)

    chunks = split_segments(
        segments, lines, input_budget, overlap_segments=overlap, max_segments=80
    )
    relevant = [c for c in chunks if any(i in editable for i in c.seg_ids)]

    all_suggestions: list[dict] = []
    # Collect rejected ones structured — the UI shows WHAT the LLM wanted
    # and WHY it didn't go through.
    all_rejected: list[dict] = []
    # Segments explicitly confirmed by LLM ("checked, okay")
    all_confirmed: set[int] = set()

    def _rejected_entry(edit, reason: str) -> dict:
        try:
            sid = int(edit.get("id")) if isinstance(edit, dict) else None
        except (TypeError, ValueError):
            sid = None
        old_value = ""
        if sid is not None and 0 <= sid < len(segments):
            old_value = (segments[sid].get("text") or "").strip()
        if isinstance(edit, dict):
            new_value = _MARKER_RE.sub(
                r"\1", str(edit.get("text") or edit.get("new_value") or "")
            ).strip()
            llm_reason = str(edit.get("reason") or "")[:200]
            try:
                confidence = round(float(edit.get("confidence", 0)), 2)
            except (TypeError, ValueError):
                confidence = 0.0
        else:
            new_value, llm_reason, confidence = "", "", 0.0
        return {
            "id": sid,
            "old_value": old_value,
            "new_value": new_value,
            "llm_reason": llm_reason,
            "confidence": confidence,
            "rejected_because": reason,
        }

    def _correct_lines(ids: list[int], context_lines: list[str]):
        """Correct a range of lines; a truncated answer → split in half."""
        parts = []
        # The headings the format contract names (prompts.CORRECTION_FORMAT:
        # 'Lines under "CONTEXT"') — they must match it word for word
        if context_lines:
            parts.append("CONTEXT (not editable):")
            parts.extend(context_lines)
        parts.append("Segments ([id] = editable, (id) = context only):")
        parts.extend(lines[i] for i in ids)
        user = "\n".join(parts)

        answer_max = max(2048, estimate_tokens(user) + len(ids) * 8)
        try:
            data = llm_client.chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                schema=prompts.CORRECTION_SCHEMA,
                schema_name="correction_edits",
                model=model,
                temperature=temperature,
                max_tokens=answer_max,
                timeout=timeout,
                task="correction",
            )
            return (data.get("edits") or [], data.get("checked_ok") or [])
        except llm_client.LlmAnswerTruncated:
            if len(ids) <= 1:
                # Generation loop on single segment (Whisper hallucination
                # with phrase repetition) → no edits for this segment instead
                # of chunk abort.
                logger.warning("Correction: segment %d triggers generation loop — skipped", ids[0])
                return ([], [])
            middle = len(ids) // 2
            logger.info(
                "Correction: response truncated — range %d–%d will be halved", ids[0], ids[-1]
            )
            a_edits, a_ok = _correct_lines(ids[:middle], context_lines)
            b_edits, b_ok = _correct_lines(ids[middle:], [])
            return (a_edits + b_edits, a_ok + b_ok)

    def _processor(chunk):
        edits, checked_ok = _correct_lines(
            list(chunk.seg_ids), [z for _i, z in chunk.context_before]
        )
        chunk_ids = {i for i in chunk.seg_ids if i in editable}
        ok, bad, confirmed = validate_edits(
            edits,
            segments,
            chunk_ids,
            min_confidence=min_confidence,
            max_edit_share=max_edit_share,
            checked_ok=checked_ok,
        )
        all_rejected.extend(_rejected_entry(e, reason) for e, reason in bad)
        all_confirmed.update(confirmed)
        return ok

    def _part(t):
        """Chunk protocol for UI: what did this section produce?"""
        if not part_callback:
            return
        edits = t.get("result") or []
        lines = [
            f"[{e['id']}] {e['old_value'][:60]}\n    → {e['new_value'][:60]}" for e in edits[:20]
        ]
        part_callback(
            {
                "number": t["number"],
                "total": t["total"],
                "from_seg": t["from_seg"],
                "to_seg": t["to_seg"],
                "tokens": t["tokens"],
                "duration": t["duration"],
                "text": ("\n".join(lines) if lines else "No corrections in this section."),
                "metric": f"{len(edits)} suggestions",
            }
        )

    results, failed = run_chunks(
        relevant, _processor, job=job, description="Correction", part_callback=_part
    )
    for idx in sorted(results):
        all_suggestions.extend(results[idx])

    return {
        "suggestions": all_suggestions,
        "discarded": all_rejected,
        # Checked segments (for "✔ Checked" tab of UI): id + short
        # confidence context so view works without extra request.
        "checked": [
            {
                "id": i,
                "text": (segments[i].get("text") or "").strip(),
                "confirmed": i in all_confirmed,
                "avg_logprob": segments[i].get("avg_logprob"),
                "uncertain_words": sum(
                    1
                    for w in (segments[i].get("words") or [])
                    if w.get("score") is not None and float(w["score"]) < score_threshold
                ),
            }
            for i in sorted(editable)
        ],
        "chunks": len(relevant),
        "chunks_failed": len(failed),
        "marked_segments": len(marked),
        "confirmed": len(all_confirmed),
        "duration": round(time.time() - t_start, 1),
        "thresholds": {"score": score_threshold, "logprob": logprob_threshold},
        "confidence_data": has_confidence,
    }


def apply_edits_to(tdata: dict, edits: list[dict]) -> list[dict]:
    """Apply accepted edits to a transcript dict (in place).

    Expects entries {id, new, ...}. Segments get ``edited: 1`` — their word
    times are then considered approximate (text no longer 1:1 from aligner).
    Returns actually applied edits (incl. old text for undo history).
    """
    segments = tdata.get("segments") or []
    applied = []
    for e in edits:
        try:
            sid = int(e["id"])
            new_value = str(e["new_value"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= sid < len(segments)) or not new_value:
            continue
        old_value = (segments[sid].get("text") or "").strip()
        if new_value == old_value:
            continue
        segments[sid]["text"] = new_value
        segments[sid]["edited"] = 1
        applied.append({"id": sid, "old_value": old_value, "new_value": new_value})
    return applied

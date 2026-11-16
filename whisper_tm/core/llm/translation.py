"""
Chunk-wise transcript translation (docs/LLM-PIPELINE.md §5.3).

Each chunk goes to the LLM as a numbered segment list, and (JSON-enforced)
exactly one translation per ID comes back. Missing IDs are requested in a
targeted follow-up request only for the gaps.

Rolling context: the last N source→target pairs of the preceding chunk
travel along in the prompt, so proper names, address form, and register
stay consistent across chunk boundaries.

Timestamps and speakers stay local — only [{id, text}] per language is
stored; when rendering/exporting, the timings always come from the
original transcript.
"""

from __future__ import annotations

import logging

from whisper_tm.config import config
from whisper_tm.core import llm_client
from whisper_tm.core.llm import prompts
from whisper_tm.core.llm.budget import chunk_budget, estimate_tokens
from whisper_tm.core.llm.chunking import split_segments
from whisper_tm.core.llm.runner import run_chunks

logger = logging.getLogger(__name__)


def _cfg(name: str, default):
    return config.get("llm", "translation", name, default=default)


def _lines(segments: list[dict]) -> list[str]:
    return [f"[{i}] {(s.get('text') or '').strip()}" for i, s in enumerate(segments)]


def _request_more(
    missing_items: list[int],
    segments: list[dict],
    system: str,
    model: str,
    temperature: float,
    timeout: int | None = None,
) -> dict[int, str]:
    """Targeted second request only for the missing IDs of a chunk.

    ``timeout`` MUST be the caller's capped chunk timeout — without it
    the request would fall back to the full llm.timeout (30 min) and
    the timeout cascade that the cap prevents would sneak back in
    through the back door.
    """
    user = "These segments are still missing. Translate them:\n" + "\n".join(
        f"[{i}] {(segments[i].get('text') or '').strip()}" for i in missing_items
    )
    data = llm_client.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema=prompts.TRANSLATION_SCHEMA,
        schema_name="translations",
        model=model,
        temperature=temperature,
        max_tokens=max(1024, estimate_tokens(user) * 2),
        timeout=timeout,
        task="translation",
    )
    out = {}
    for u in data.get("translations") or []:
        try:
            sid = int(u.get("id"))
            text = str(u.get("text") or "").strip()
        except (TypeError, ValueError):
            continue
        if sid in missing_items and text:
            out[sid] = text
    return out


def translate(
    segments: list[dict], target_language: str, *, model: str = "", job=None, part_callback=None
) -> dict:
    """Translate the complete transcript (blocking, chunk-wise).

    Returns:
        {"translations": [{id, text}] (ascending, gap-free where
         successful), "missing": [ids], "chunks": n,
         "chunks_failed": n}
    """
    if not segments:
        return {"translations": [], "missing": [], "chunks": 0, "chunks_failed": 0}
    target_language = (target_language or "").strip()
    if not target_language:
        raise ValueError("No target language specified")

    temperature = float(_cfg("temperature", 0.3))
    context_pairs = int(_cfg("context_pairs", 5))
    # Chunk tasks don't need 30-minute timeouts — a chunk is small; if
    # the server hangs, the run should fail promptly instead of
    # stretching the retry cascade over hours.
    timeout = min(int(config.get("llm", "timeout", default=1800)), 900)
    system = prompts.TRANSLATION_SYSTEM_TEMPLATE.format(target_language=target_language)

    lines = _lines(segments)
    system_tokens = estimate_tokens(system)
    context = llm_client.effective_context("translation")
    margin = float(config.get("llm", "chunking", "safety_margin", default=0.10))
    available = chunk_budget(
        system_tokens=system_tokens,
        instruction_tokens=128,
        answer_reserve=0,
        context=context,
        margin=margin,
    )
    # Response ≈ 2× input (language expansion, e.g. ja→de) PLUS ~16 tokens
    # of JSON scaffolding per segment → input gets 1/3.5 of the budget, and
    # the segment count per chunk is capped (scaffolding cost dominates
    # with many short segments — otherwise the response gets capped).
    input_budget = max(512, int(available / 3.5))

    # Overlap 0: the reading context here doesn't come from source lines,
    # but from the already TRANSLATED pairs (rolling context).
    chunks = split_segments(segments, lines, input_budget, overlap_segments=0, max_segments=60)

    result: dict[int, str] = {}
    # Rolling context is built from `result` in the processor — the
    # runner works strictly sequentially, so the state of the preceding
    # chunk is always already there for the next chunk.

    def _translate_ids(ids: list[int]) -> dict[int, str]:
        """Translate a segment range; on a capped response the range is
        halved (recursively, down to a single segment)."""
        parts = []
        if context_pairs > 0 and ids[0] > 0:
            pairs = []
            for i in range(max(0, ids[0] - context_pairs), ids[0]):
                if i in result:
                    source = (segments[i].get("text") or "").strip()
                    # (id) form: clearly distinguishable from the [id]
                    # target format — otherwise the LLM translates
                    # context lines right along with it.
                    pairs.append(f"({i}) {source}\n  → {result[i]}")
            if pairs:
                parts.append("Already translated (context for consistency):")
                parts.extend(pairs)
        parts.append("Segments to translate:")
        parts.extend(lines[i] for i in ids)
        user = "\n".join(parts)

        input_tokens = estimate_tokens(user)
        answer_max = int(input_tokens * 2.0) + len(ids) * 16 + 256
        try:
            data = llm_client.chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                schema=prompts.TRANSLATION_SCHEMA,
                schema_name="translations",
                model=model,
                temperature=temperature,
                max_tokens=answer_max,
                timeout=timeout,
                task="translation",
            )
        except llm_client.LlmAnswerTruncated:
            if len(ids) <= 1:
                # A SINGLE segment whose translation still hits the limit:
                # this is practically always a Whisper hallucination with
                # an N-times-repeated phrase, which the LLM gets stuck
                # repeating in a generation loop.
                # Skip it (→ missing) instead of sacrificing the chunk.
                logger.warning(
                    "Translation: segment %d triggers a generation loop — skipped (source: %r)",
                    ids[0],
                    (segments[ids[0]].get("text") or "")[:80],
                )
                return {}
            middle = len(ids) // 2
            logger.info(
                "Translation: response capped — range %d–%d is being halved", ids[0], ids[-1]
            )
            out = _translate_ids(ids[:middle])
            result.update(out)
            out.update(_translate_ids(ids[middle:]))
            return out

        id_set = set(ids)
        out: dict[int, str] = {}
        for u in data.get("translations") or []:
            try:
                sid = int(u.get("id"))
                text = str(u.get("text") or "").strip()
            except (TypeError, ValueError):
                continue
            if sid in id_set and text:
                out[sid] = text
        return out

    def _processor(chunk):
        chunk_ids = list(chunk.seg_ids)
        out = _translate_ids(chunk_ids)

        missing_items = sorted(set(chunk_ids) - set(out))
        if missing_items:
            logger.info(
                "Translation chunk %d: %d IDs missing — requesting", chunk.index, len(missing_items)
            )
            try:
                out.update(
                    _request_more(
                        missing_items, segments, system, model, temperature, timeout=timeout
                    )
                )
            except llm_client.LlmAnswerTruncated:
                # Follow-up request contains a loop segment → the gaps
                # simply stay gaps; the remainder of the chunk is saved.
                logger.warning(
                    "Translation chunk %d: follow-up request capped — %d segments remain open",
                    chunk.index,
                    len(missing_items),
                )
        result.update(out)
        return out

    def _part(t):
        """Chunk log: the translated lines of this section."""
        if not part_callback:
            return
        out = t.get("result") or {}
        lines = [f"[{i}] {out[i][:90]}" for i in sorted(out)][:25]
        part_callback(
            {
                "number": t["number"],
                "total": t["total"],
                "from_seg": t["from_seg"],
                "to_seg": t["to_seg"],
                "tokens": t["tokens"],
                "duration": t["duration"],
                "text": "\n".join(lines) or "(nothing translated)",
                "metric": f"{len(out)} segments",
            }
        )

    _res, failed = run_chunks(
        chunks,
        _processor,
        job=job,
        description=f"Translation → {target_language}",
        part_callback=_part,
    )

    missing = [
        i
        for i in range(len(segments))
        if i not in result and (segments[i].get("text") or "").strip()
    ]
    return {
        "translations": [{"id": i, "text": result[i]} for i in sorted(result)],
        "missing": missing,
        "chunks": len(chunks),
        "chunks_failed": len(failed),
    }

"""
Token estimation and chunk budget.

The estimation heuristic lives here (not in api/llm.py) so that chunker,
routes, and budget calculation see the same numbers (api/llm.py delegates).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Conservative default if context size is unknown (server unreachable,
# exotic server): prefer small chunks to truncated ones.
CONTEXT_FALLBACK = 8192


def estimate_tokens(text: str) -> int:
    """Estimate token count (without tokenizer dependency).

    Heuristic based on OpenAI/LLaMA tokenizer statistics:
    - English/German: ~4 characters per token (with spaces/punctuation)
    - CJK (Japanese/Chinese/Korean): ~1.5 characters per token
    Accuracy: ±10-15% — sufficient for budget checks, the security
    margin (see chunk_budget) handles the remainder.
    """
    if not text:
        return 0
    cjk_count = 0
    special_count = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF  # CJK Unified
            or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
            or 0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0xAC00 <= cp <= 0xD7AF  # Hangul
            or 0xFF00 <= cp <= 0xFFEF
        ):  # Fullwidth
            cjk_count += 1
        elif cp > 0x2000:
            # Other non-ASCII special characters (markers ⟨⟩, arrows, emoji):
            # typically tokenize as multiple byte tokens — conservatively 1 token
            # per character (a marker flood was otherwise underestimated 3.5×).
            special_count += 1
    non_cjk = len(text) - cjk_count - special_count
    return max(1, int(cjk_count / 1.5 + special_count + non_cjk / 4.0))


def chunk_budget(
    *,
    system_tokens: int,
    instruction_tokens: int = 0,
    answer_reserve: int = 0,
    context: int = 0,
    margin: float = 0.10,
) -> int:
    """How many input tokens a chunk (segment lines) may occupy.

    Budget = context − system − instruction − answer reserve − margin·context.
    ``context=0`` means unknown → CONTEXT_FALLBACK.
    """
    if not context or context <= 0:
        context = CONTEXT_FALLBACK
    budget = int(context * (1.0 - margin)) - system_tokens - instruction_tokens - answer_reserve
    if budget < 512:
        # Pathological (giant system prompt or mini context): work with 512
        # instead of risking infinite loop in chunker.
        logger.warning(
            "Chunk budget calculated as %d — raised to 512 (context %d, system %d, reserve %d)",
            budget,
            context,
            system_tokens,
            answer_reserve,
        )
        budget = 512
    return budget

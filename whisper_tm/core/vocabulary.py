"""
Vocabulary presets for Whisper (initial_prompt).

Whisper guesses by probability — rare proper nouns lose to frequently
sounding alternatives (e.g. an unusual place name in a Japanese show
turns into a common-sounding phrase). The ``initial_prompt`` is the
counter-spell: a short word list fed to the decoder as context and
makes correct spellings massively more likely. whisperx applies it in
batch mode to EVERY decode window — exactly what we want.

Presets live in ``transcription.vocab_presets`` (name → word list,
editable in settings panel), the selection travels as preset NAME through
the queue (settings_json) and is resolved only in the worker — presets
can thus be changed without touching waiting entries.

Budget: Whisper only uses the last ~224 tokens of the prompt. The
estimate here (CJK ≈ 1.5 characters/token, Latin ≈ 4) is the same
heuristic as LLM token estimation — sufficient for a warning;
nothing is hard cut (Whisper takes the tail itself).
"""

from __future__ import annotations

import logging

from whisper_tm.config import config

logger = logging.getLogger(__name__)

# Whisper context for the prompt: the last max_length//2 - 1 = 223 tokens.
WHISPER_PROMPT_BUDGET = 224


def estimate_tokens(text: str) -> int:
    """Token estimate without tokenizer (CJK ~1.5, Latin ~4 characters/token)."""
    if not text:
        return 0
    cjk = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x3040 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7AF
            or 0xFF00 <= cp <= 0xFFEF
        ):
            cjk += 1
    return max(1, int(cjk / 1.5 + (len(text) - cjk) / 4.0))


def preset_prompt(name: str | None) -> str:
    """Word list of a preset — '' / unknown = no prompt.

    Warns (once per call) instead of truncating if the list exceeds the
    Whisper budget — then only its end counts, the beginning is wasted.
    """
    if not name:
        return ""
    presets = config.get("transcription", "vocab_presets", default={}) or {}
    text = (presets.get(name) or "").strip() if isinstance(presets, dict) else ""
    if not text:
        logger.warning("Vocabulary preset '%s' is empty/unknown — running without", name)
        return ""
    tokens = estimate_tokens(text)
    if tokens > WHISPER_PROMPT_BUDGET:
        logger.warning(
            "Vocabulary preset '%s': ~%d tokens > Whisper budget %d — "
            "only the END of the list works, place most important terms at "
            "end or shorten",
            name,
            tokens,
            WHISPER_PROMPT_BUDGET,
        )
    return text

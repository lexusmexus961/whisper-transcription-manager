"""
Vocabulary presets (Whisper initial_prompt) — the chain from dialog select
to prompt resolution in the worker.

Reason: Whisper mangles domain vocabulary such as rare names and terms.
Presets are in transcription.vocab_presets; selection travels as NAME
through the queue and only becomes text in the worker — presets thus stay
editable without touching waiting entries.
"""

from __future__ import annotations

from whisper_tm.core import vocabulary as v


def test_whitelist_and_normalize_pass_name_through():
    from whisper_tm.api.queue import _SETTING_KEYS, _settings_from
    from whisper_tm.core.queue import _normalize

    assert "vocab_preset" in _SETTING_KEYS
    assert _settings_from({"vocab_preset": "X"})["vocab_preset"] == "X"
    assert _normalize({"vocab_preset": " 日本語プリセット "})["vocab_preset"] == "日本語プリセット"
    assert _normalize({})["vocab_preset"] is None  # no preset
    assert _normalize({"vocab_preset": ""})["vocab_preset"] is None


def test_worker_knows_parameter():
    import inspect

    from whisper_tm.core.transcription import transcription_worker

    assert "vocab_preset" in inspect.signature(transcription_worker).parameters


def test_preset_resolution_and_default():
    from whisper_tm.config import config

    config.set("transcription", "vocab_presets", {"Series A": "  東京都、駅前  ", "Empty": "   "})
    try:
        assert v.preset_prompt("Series A") == "東京都、駅前"
        assert v.preset_prompt("Empty") == ""  # empty → no prompt
        assert v.preset_prompt("does-not-exist") == ""
        assert v.preset_prompt(None) == ""
        assert v.preset_prompt("") == ""
    finally:
        config.set("transcription", "vocab_presets", {})


def test_the_shipped_example_holds_the_budget():
    # One example of the form: a sentence with the names
    from pathlib import Path

    import yaml

    d = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(encoding="utf-8")
    )
    presets = d["transcription"]["vocab_presets"]
    assert list(presets) == ["Tech & gaming"]
    text = presets["Tech & gaming"]
    assert "Discord" in text and "NVIDIA" in text
    assert v.estimate_tokens(text) <= v.WHISPER_PROMPT_BUDGET


def test_token_estimation_cjk_vs_latin():
    # CJK counts heavier (~1.5 chars/token) than Latin (~4)
    assert v.estimate_tokens("東京都") == 2
    assert v.estimate_tokens("abcdefgh") == 2
    assert v.estimate_tokens("") == 0
    long_text = "東京都、" * 120  # ~320 tokens
    assert v.estimate_tokens(long_text) > v.WHISPER_PROMPT_BUDGET

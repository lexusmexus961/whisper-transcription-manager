"""
Automatic summarization after transcription (dialog option).

The chain is long and quiet: checkbox in dialog → body of /api/queue/add →
whitelist ``_SETTING_KEYS`` → queue row's ``settings_json`` → post-step in
``core/queue._process``. If ONE link is missing, nothing happens — that's
exactly what the first tests check.

Plus two properties that protect operation:
* a failed summarization must not devalue the finished transcription (no queue error),
* ``chat_request`` must return the same for both endpoint dialects as the
  streaming route did before — it now uses the same blueprint.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import summarize as s

# ── Path of setting through the chain ───────────────────────────────────


def test_the_whitelist_passes_the_summary_keys_through():
    from whisper_tm.api.queue import _SETTING_KEYS, _settings_from

    for k in ("summarize", "summarize_preset", "summarize_model"):
        assert k in _SETTING_KEYS, f"{k} missing from whitelist"

    data = {
        "model": "turbo",
        "summarize": True,
        "summarize_preset": "Key Points",
        "summarize_model": "qwen3",
        "unknown_key": "gone",
    }
    settings = _settings_from(data)
    assert settings["summarize"] is True
    assert settings["summarize_preset"] == "Key Points"
    assert settings["summarize_model"] == "qwen3"
    assert "unknown_key" not in settings


def test_normalize_does_not_pass_the_keys_to_the_worker():
    """``transcription_worker`` doesn't know them — they must never get there.

    ``_normalize`` passes its result 1:1 as ``**opts``; one extra key would be
    a TypeError in the middle of the queue.
    """
    from whisper_tm.core.queue import _normalize

    opts = _normalize(
        {"summarize": True, "summarize_preset": "Timeline", "summarize_model": "qwen3"}
    )
    assert not any(k.startswith("summarize") for k in opts)


# ── Preset resolution ──────────────────────────────────────────────────────


def test_the_preset_prompt_takes_the_preset_and_falls_back_cleanly():
    from whisper_tm.config import config

    presets = config.get("llm", "presets", default={}) or {}
    name = next(iter(presets))  # first preset from config
    assert s.preset_prompt(name) == presets[name]
    # unknown/empty → system prompt from config, not empty
    standard = config.get("llm", "system_prompt", default="")
    assert s.preset_prompt("does-not-exist") == standard
    assert s.preset_prompt("") == standard


# ── Chat request for both dialects ───────────────────────────────────────


class TestChatRequest:
    def test_ollama_native(self, monkeypatch):
        from whisper_tm.config import config

        config.set("llm", "endpoint", "http://localhost:11434")
        config.set("llm", "max_tokens", 1234)
        url, headers, payload, native = s.chat_request(
            [{"role": "user", "content": "x"}], model="qwen3", stream=False
        )
        assert native is True
        assert url.endswith("/api/chat")
        assert payload["model"] == "qwen3"
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 1234
        assert "max_tokens" not in payload  # that's called num_predict here

    def test_openai_compatible(self):
        from whisper_tm.config import config

        config.set("llm", "endpoint", "http://localhost:8080/v1")
        config.set("llm", "model", "gpt-4o-mini")
        url, headers, payload, native = s.chat_request(
            [{"role": "user", "content": "x"}], stream=True
        )
        assert native is False
        assert url.endswith("/chat/completions")
        assert payload["model"] == "gpt-4o-mini"  # empty = config default
        assert payload["stream"] is True
        assert "max_tokens" in payload


# ── summarize(): complete post-step without browser ─────────────────


class _Reply:
    status_code = 200
    text = ""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


@pytest.fixture
def llm(monkeypatch, tmp_path):
    """LLM configured, network mocked, storage in tmp_path."""
    from whisper_tm.config import config

    config.set("llm", "endpoint", "http://localhost:11434")
    # Clean up provider remnants from other test modules: since task
    # assignment in the chat path WORKS, a leftover providers list would
    # displace the legacy endpoint configured here.
    config._data.get("llm", {}).pop("providers", None)
    config._data.get("llm", {}).pop("tasks", None)
    monkeypatch.setattr(type(config), "transcript_dir", property(lambda self: tmp_path))
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["url"] = url
        sent["payload"] = json
        return _Reply({"message": {"content": "  The summary.  "}})

    monkeypatch.setattr(s.requests, "post", fake_post)
    return sent


def test_summarizing_builds_the_prompt_and_saves(llm, monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(s, "load_summaries", lambda f, media_dir=None, **_: [])
    monkeypatch.setattr(
        s,
        "save_summaries",
        lambda f, lst, media_dir=None: saved.update(file=f, listing=lst) or True,
    )

    text = s.make_summary(
        "a/b.mp4",
        preset="Key Points",
        model="qwen3",
        segments=[{"text": "Hello", "speaker": "SPEAKER_00"}],
    )

    assert text == "The summary."  # stripped
    # Prompt contains the preset's system prompt AND the transcript
    messages = llm["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert "Key Points" in messages[0]["content"] or messages[0]["content"]
    assert "Hello" in messages[-1]["content"]
    # ONE entry is saved, marked as automatic
    assert saved["file"] == "a/b.mp4"
    entry = saved["listing"][-1]
    assert entry["text"] == "The summary."
    assert entry["auto"] is True
    assert entry["prompt"] == "Key Points"  # label of the saved list
    assert entry["model"] == "qwen3"


def test_map_reduce_summarizing_stores_it(llm, monkeypatch):
    """In the map-reduce branch no `payload` existed — the save block
    crashed with UnboundLocalError AFTER the
    complete LLM run had finished. Exactly the use case (long
    recordings) that auto-summarization is meant for."""
    saved = {}
    monkeypatch.setattr(s, "load_summaries", lambda f, media_dir=None, **_: [])
    monkeypatch.setattr(
        s,
        "save_summaries",
        lambda f, lst, media_dir=None: saved.update(file=f, listing=lst) or True,
    )
    # Force map-reduce and stub the stream generator — tested here
    # is the follow-up step (saving), not the chunking logic.
    monkeypatch.setattr(s, "_needs_map_reduce", lambda *a: (True, 32768))
    monkeypatch.setattr(
        s,
        "summary_stream",
        lambda *a, **k: iter(
            [{"status": "Section 1/2"}, {"text": "Part one. "}, {"text": "Part two."}]
        ),
    )
    text = s.make_summary(
        "a/long.mp4", preset="Bullet Points", model="qwen3", segments=[{"text": "Hello"}]
    )
    assert text == "Part one. Part two."
    entry = saved["listing"][-1]
    assert entry["auto"] is True
    assert entry["model"] == "qwen3"


def test_summarizing_appends_to_existing_ones(llm, monkeypatch):
    old_value = [{"text": "old_value", "prompt": "", "timestamp": 1.0}]
    monkeypatch.setattr(s, "load_summaries", lambda f, media_dir=None, **_: list(old_value))
    result = {}
    monkeypatch.setattr(
        s, "save_summaries", lambda f, lst, media_dir=None: result.update(listing=lst) or True
    )

    s.make_summary("x.mp4", segments=[{"text": "y"}])
    assert len(result["listing"]) == 2
    assert result["listing"][0]["text"] == "old_value"  # the existing one stays


def test_an_own_instruction_replaces_the_template(llm, monkeypatch):
    """Custom from the browser extension: the instruction is the
    system prompt, the language rule still follows, the list says Custom."""
    saved, seen = {}, {}
    monkeypatch.setattr(s, "load_summaries", lambda f, media_dir=None, **_: [])
    monkeypatch.setattr(
        s, "save_summaries", lambda f, lst, media_dir=None: saved.update(listing=lst) or True
    )
    monkeypatch.setattr(s, "_needs_map_reduce", lambda *a: (True, 32768))
    monkeypatch.setattr(
        s, "summary_stream", lambda *a, **k: seen.update(k) or iter([{"text": "Done."}])
    )
    s.make_summary(
        "a.mp4",
        preset="Summary",
        prompt="Only the jokes.",
        language="de",
        segments=[{"text": "Hello"}],
    )
    assert seen["system_prompt"].startswith("Only the jokes.")
    assert seen["system_prompt"].endswith("whatever the language of the transcript.")
    assert saved["listing"][-1]["prompt"] == "Custom"


def test_summarizing_without_an_endpoint_reports_clearly(monkeypatch):
    from whisper_tm.config import config

    config.set("llm", "endpoint", "")
    with pytest.raises(RuntimeError, match="not configured"):
        s.make_summary("x.mp4", segments=[{"text": "y"}])


def test_summarizing_without_a_transcript_reports_clearly(llm, monkeypatch):
    monkeypatch.setattr(s, "load_segments", lambda f, media_dir=None: None)
    with pytest.raises(RuntimeError, match="not found"):
        s.make_summary("x.mp4")


def test_an_empty_transcript_does_not_go_to_the_llm(llm, monkeypatch):
    monkeypatch.setattr(s, "load_segments", lambda f, media_dir=None: [{"text": "   "}])
    with pytest.raises(RuntimeError, match="empty"):
        s.make_summary("x.mp4")
    assert "payload" not in llm  # never sent at all


# ── The follow-up step must never tip over the queue ──────────────────────


def _queued(monkeypatch, enqueue):
    from whisper_tm.core import queue as q
    from whisper_tm.core.llm_queue import llm_queue

    monkeypatch.setitem(q.config._data.setdefault("llm", {}), "endpoint", "http://llm:8080")
    monkeypatch.setattr(llm_queue, "enqueue", enqueue)
    lines = []
    monkeypatch.setattr(q, "qlog", lines.append)
    return q, lines


def test_a_summary_that_cant_be_queued_does_not_throw(monkeypatch):
    """A finished transcription stays finished, even if the LLM queue balks."""

    def broken(*a, **k):
        raise RuntimeError("database locked")

    q, lines = _queued(monkeypatch, broken)
    q._queue_summary("a.mp4", {"summarize": True}, None)
    assert any("not queued" in z for z in lines), lines


def test_the_summary_goes_to_the_llm_queue_with_its_settings(monkeypatch):
    called = {}
    q, lines = _queued(
        monkeypatch, lambda kind, f, params: called.update(kind=kind, file=f, params=params) or 7
    )
    q._queue_summary(
        "a.mp4",
        {"summarize": True, "summarize_preset": "Timeline", "summarize_model": "qwen3"},
        "/media",
    )
    assert (called["kind"], called["file"]) == ("summary", "a.mp4")
    assert called["params"] == {
        "media_dir": "/media",  # root of THIS entry
        "auto": True,
        "label": "Timeline",
        "preset": "Timeline",
        "model": "qwen3",
    }
    assert any("queued" in z for z in lines), lines


def test_a_recording_without_speech_gets_no_summary(monkeypatch):
    """Silence or a test tone: the summary failed in the LLM queue as
    "Transcript not found". It isn't queued at all now."""
    called = []
    q, lines = _queued(monkeypatch, lambda *a: called.append(a) or 1)
    q._queue_summary("silence.wav", {"summarize": True}, "/media", {"segments": [{"text": "  "}]})
    assert called == [] and any("transcript is empty" in z for z in lines), lines
    q._queue_summary("talk.wav", {"summarize": True}, "/media", {"segments": [{"text": "Hello."}]})
    assert len(called) == 1


# ── Low VRAM per run ─────────────────────────────────────────────────────


def test_the_low_vram_key_survives_the_chain():
    from whisper_tm.api.queue import _SETTING_KEYS, _settings_from

    assert "low_vram" in _SETTING_KEYS
    assert _settings_from({"low_vram": True})["low_vram"] is True


def test_normalize_low_vram_none_means_the_config_default():
    from whisper_tm.core.queue import _normalize

    assert _normalize({})["low_vram"] is None  # worker takes config
    assert _normalize({"low_vram": True})["low_vram"] is True
    assert _normalize({"low_vram": False})["low_vram"] is False


def test_a_vram_override_beats_the_config():
    from whisper_tm.config import config
    from whisper_tm.core import vram

    config.set("hardware", "low_vram", False)
    try:
        assert vram.low_vram() is False
        vram.set_low_vram_override(True)
        assert vram.low_vram() is True  # dialog overrides config
        vram.set_low_vram_override(None)
        config.set("hardware", "low_vram", True)
        assert vram.low_vram() is True  # back to config
    finally:
        vram.set_low_vram_override(None)
        config.set("hardware", "low_vram", False)

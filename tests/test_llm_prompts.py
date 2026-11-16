"""
Prompt contracts: prompts are part of the interface to the model —
broken placeholders or a prompt that promises different fields than the
schema enforces otherwise only surface during live runs.
"""

from pathlib import Path

import pytest
import yaml

from whisper_tm.core.llm import prompts

# ── Placeholders ───────────────────────────────────────────────────────────


def test_the_translation_template_is_formattable():
    t = prompts.TRANSLATION_SYSTEM_TEMPLATE.format(target_language="English")
    assert "English" in t
    # JSON examples in the prompt must not carry double braces after format()
    # (otherwise the model sees {{…}}).
    assert "{{" not in t and "}}" not in t
    assert '"translations"' in t


def test_map_and_reduce_templates_are_formattable():
    m = prompts.MAP_USER_TEMPLATE.format(
        number=2, total=5, from_value="1:00", to_value="2:00", text="Content"
    )
    assert "2/5" in m and "1:00–2:00" in m and "Content" in m
    r = prompts.REDUCE_INTRO_TEMPLATE.format(count=3, parts="A\n\nB")
    assert "3 material collections" in r and "A\n\nB" in r


@pytest.mark.parametrize(
    "name",
    [
        "SUMMARY_DEFAULT",
        "MAP_SYSTEM",
        "CORRECTION_SYSTEM",
    ],
)
def test_fixed_prompts_have_no_open_placeholders(name):
    text = getattr(prompts, name)
    assert text.strip()
    # A '{…}' in a prompt WITHOUT format() call would be a typo —
    # only JSON examples are allowed (which carry quotes).
    import re

    for hits in re.findall(r"\{(\w+)\}", text):
        pytest.fail(f"{name}: open placeholder {{{hits}}}")


# ── Prompt ↔ Schema ────────────────────────────────────────────────────────


def test_the_correction_prompt_names_every_schema_field():
    fields = set(prompts.CORRECTION_SCHEMA["properties"]["edits"]["items"]["required"])
    assert fields == {"id", "text", "reason", "confidence"}
    for f in ("reason", "confidence"):
        assert f'"{f}"' in prompts.CORRECTION_SYSTEM, f"{f} explained in prompt?"
    # Both top-level fields must be explained in the prompt
    assert set(prompts.CORRECTION_SCHEMA["required"]) == {"edits", "checked_ok"}
    assert '"checked_ok"' in prompts.CORRECTION_SYSTEM


def test_correction_prompt_offers_compact_confirmation():
    """The model wants to report what it checked (most returns are
    unchanged confirmations). It gets
    a pure ID list for this — a full text field let a single
    chunk run for many minutes because the model copied every segment."""
    t = prompts.CORRECTION_SYSTEM
    assert "checked_ok" in t
    assert "ONLY the numbers" in t  # explicitly no text
    assert "differ" in t  # corrected text MUST be different
    assert "Never invent" in t
    # The example in the prompt must match the schema
    assert '"checked_ok": [13]' in t


def test_the_translation_prompt_names_the_schema_field():
    fields = set(prompts.TRANSLATION_SCHEMA["properties"]["translations"]["items"]["required"])
    assert fields == {"id", "text"}


# ── Included presets ──────────────────────────────────────────────────────


def test_the_default_presets_are_usable():
    assert len(prompts.STANDARD_PRESETS) >= 8
    for name, text in prompts.STANDARD_PRESETS.items():
        assert name.strip() == name and name
        assert len(text.strip()) > 80, f"{name}: too meager"
        # Each preset should either brake hallucinations OR set the language
        assert "language of the transcript" in text or "Respond" in text, (
            f"{name}: no language rule"
        )


def test_config_default_mirrors_the_presets():
    """config.default.yaml is just a copy — it must not drift."""
    cfg = yaml.safe_load(Path("config.default.yaml").read_text(encoding="utf-8"))
    presets = cfg["llm"]["presets"]
    assert set(presets) == set(prompts.STANDARD_PRESETS)
    for name, text in prompts.STANDARD_PRESETS.items():
        assert presets[name].strip() == text.strip(), f"{name} differs"
    assert cfg["llm"]["system_prompt"].strip() == prompts.SUMMARY_DEFAULT.strip()


# ── Merge endpoint ─────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    return create_app().test_client(), config


def test_merge_adds_missing_ones_and_spares_your_own(client):
    c, config = client
    config.set("llm", "presets", {"My Preset": "custom text", "Short Summary": "my version"})
    d = c.get("/api/llm/standard_presets").get_json()
    status = {p["name"]: p["status"] for p in d["presets"]}
    assert status["Short Summary"] == "different"
    assert status["Timeline"] == "new"
    assert d["custom"] == ["My Preset"]

    r = c.post("/api/llm/presets/merge", json={}).get_json()
    assert "Timeline" in r["added"]
    assert r["replaced"] == []  # without flag, nothing replaced
    now = config.get("llm", "presets")
    assert now["My Preset"] == "custom text"  # untouched
    assert now["Short Summary"] == "my version"  # untouched


def test_merge_with_overwrite_resets(client):
    c, config = client
    config.set("llm", "presets", {"Short Summary": "my version", "My Preset": "stays"})
    r = c.post("/api/llm/presets/merge", json={"overwrite": True}).get_json()
    assert "Short Summary" in r["replaced"]
    now = config.get("llm", "presets")
    assert now["Short Summary"] == prompts.STANDARD_PRESETS["Short Summary"]
    assert now["My Preset"] == "stays"


def test_merge_only_the_selected_ones(client):
    c, config = client
    config.set("llm", "presets", {})
    r = c.post("/api/llm/presets/merge", json={"names": ["Timeline", "doesntexist"]}).get_json()
    assert r["added"] == ["Timeline"]
    assert set(config.get("llm", "presets")) == {"Timeline"}


# ── Correction rules replaceable via settings ──────────────────────────────


def test_the_correction_system_default_is_rules_plus_format():
    full = prompts.correction_system()
    assert full == prompts.CORRECTION_SYSTEM
    assert full.startswith(prompts.CORRECTION_RULES)
    assert full.endswith(prompts.CORRECTION_FORMAT)


def test_own_rules_replace_only_the_rules_part():
    full = prompts.correction_system("Only correct proper names.")
    assert full.startswith("Only correct proper names.")
    assert prompts.CORRECTION_RULES not in full
    # The format contract always hangs behind it — it carries marker syntax,
    # output fields and the example:
    assert prompts.CORRECTION_FORMAT in full
    for required in ("⟨", '"edits"', '"checked_ok"', "OUTPUT"):
        assert required in full


def test_empty_correction_rules_fall_back_to_the_default():
    assert prompts.correction_system("") == prompts.CORRECTION_SYSTEM
    assert prompts.correction_system("   \n") == prompts.CORRECTION_SYSTEM


def test_correction_uses_config_rules(monkeypatch):
    """build_suggestions must really read llm.correction.system_prompt
    and send it as system prompt — otherwise the setting would be orphaned."""
    from whisper_tm.config import config
    from whisper_tm.core.llm import correction as corr

    seen = {}

    def fake_chat_json(messages, **kw):
        seen["system"] = messages[0]["content"]
        return {"edits": [], "checked_ok": []}

    monkeypatch.setattr(corr.llm_client, "chat_json", fake_chat_json)
    monkeypatch.setattr(corr.llm_client, "effective_context", lambda *a, **kw: 8192)
    old_value = config.get("llm", "correction", "system_prompt", default="")
    config.set("llm", "correction", "system_prompt", "MY RULES: don't touch anything.")
    try:
        segs = [{"start": 0.0, "end": 1.0, "text": "Hello World", "avg_logprob": -2.0, "words": []}]
        corr.build_suggestions(segs, only_marked=True)
        assert seen["system"].startswith("MY RULES: don't touch anything.")
        assert prompts.CORRECTION_FORMAT in seen["system"]
    finally:
        config.set("llm", "correction", "system_prompt", old_value)

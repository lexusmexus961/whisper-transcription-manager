"""Summary templates and what the summarize dialog relies on in the backend.

Templates share one shape and the same closing rules, the
dialog saves templates itself, the additional instruction moved behind the
transcript, map-reduce collects for the template instead of neutrally, and
a saved summary is only deleted if it is still the one that was shown.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from whisper_tm.core import summarize as s
from whisper_tm.core.helpers import format_ts
from whisper_tm.core.llm import prompts

REPO = Path(__file__).resolve().parent.parent


# ── The templates ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(prompts.STANDARD_PRESETS))
def test_every_template_has_the_shared_shape(name):
    text = prompts.STANDARD_PRESETS[name]
    assert "\nRules:\n" in text, name
    # The two rules every template ends on: the source, the language
    assert text.rstrip().endswith("Write in the language of the transcript."), name
    assert re.search(r"transcript|literally said|specific passages", text), name
    # No preamble-style openers like "You are…" — the first line is the task
    assert not text.startswith("You "), name


def test_the_default_prompt_is_the_summary_template():
    assert prompts.STANDARD_PRESETS["Summary"] == prompts.SUMMARY_DEFAULT


def test_timestamps_in_templates_look_like_the_transcripts():
    # The model is told to copy timestamps exactly — so the examples must
    # have the format it sees ([H:MM:SS], core/helpers.format_ts)
    assert format_ts(754) == "0:12:34"
    for name, text in prompts.STANDARD_PRESETS.items():
        for stamp in re.findall(r"\[(\d+(?::\d\d)+)\]", text):
            assert re.fullmatch(r"\d:\d\d:\d\d", stamp), (name, stamp)
    assert "[0:00:00] Title" in prompts.STANDARD_PRESETS["Chapter Marks"]


def test_the_words_the_dialog_hints_look_for():
    # summarize.js _sumHints: "speaker label" / "timestamp" in the prompt
    assert "speaker labels" in prompts.STANDARD_PRESETS["Speaker Analysis"]
    for name in ("Chapter Marks", "Timeline", "Highlights"):
        assert "timestamp" in prompts.STANDARD_PRESETS[name], name


# ── Messages ────────────────────────────────────────────────────────────────


def test_the_instruction_comes_after_the_transcript():
    msgs = s.build_messages(
        "TEMPLATE", "only the decisions", "LINE 1\nLINE 2", {"filename": "a.mp4"}
    )
    assert msgs[0] == {"role": "system", "content": "TEMPLATE"}
    user = msgs[1]["content"]
    assert user.index("File: a.mp4") < user.index("LINE 1") < user.index("only the decisions")
    assert user.rstrip().endswith("only the decisions")


def test_without_an_instruction_the_transcript_ends_the_message():
    user = s.build_messages("", "", "LINE", None)
    assert user == [{"role": "user", "content": "Transcript:\n\nLINE"}]


def test_every_timestamp_the_model_sees_has_one_format():
    segs = [{"start": 3725, "text": "late"}]  # no start_fmt stored
    assert s.format_transcript(segs) == "[1:02:05] late"
    assert s.fmt_time(3725) == "1:02:05"


# ── Map-reduce collects for the template ───────────────────────────────────


def test_the_map_step_knows_the_task(monkeypatch):
    from whisper_tm.core import llm_client

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 900)
    systems, users = [], []

    def fake_chat(messages, **kw):
        systems.append(messages[0]["content"])
        users.append(messages[-1]["content"])
        return "notes [0:00:01] a joke", {}

    def fake_stream(messages, **kw):
        final = messages[-1]["content"]
        assert "Part 1 (0:00:00–" in final
        assert final.rstrip().endswith("keep it short")
        yield "FINAL"

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    monkeypatch.setattr(llm_client, "chat_stream", fake_stream)
    segs = [
        {"start": i * 5.0, "end": i * 5.0 + 4, "text": f"Sentence number {i} " * 6}
        for i in range(120)
    ]
    events = list(
        s.summary_stream(
            segs,
            system_prompt="Find the funny moments.",
            user_prompt="keep it short",
            metadata=None,
            model="",
            include_speakers=False,
            include_timestamps=True,
        )
    )
    assert [e["text"] for e in events if "text" in e] == ["FINAL"]
    assert len(systems) >= 2
    for sys_prompt in systems:
        assert sys_prompt.startswith(prompts.MAP_SYSTEM)
        assert "Find the funny moments." in sys_prompt
        assert "keep it short" in sys_prompt
    assert all(u.startswith("Transcript excerpt ") for u in users)


def test_without_a_task_the_map_prompt_is_the_plain_one():
    assert prompts.map_system("") == prompts.MAP_SYSTEM
    assert prompts.map_system("  ") == prompts.MAP_SYSTEM
    assert prompts.map_system("X").endswith("<task>\nX\n</task>")


# ── Endpoints ───────────────────────────────────────────────────────────────


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


def test_status_names_the_builtin_templates(client):
    c, _config = client
    d = c.get("/api/llm/status").get_json()
    assert d["preset_builtin"] == list(prompts.STANDARD_PRESETS)


def test_saving_a_template_from_the_dialog(client):
    c, config = client
    config.set("llm", "presets", {"Mine": "old text", "Other": "stays"})
    r = c.post("/api/llm/presets/save", json={"name": " New one ", "text": "Prompt"})
    d = r.get_json()
    assert r.status_code == 200 and d["ok"] and d["replaced"] is False
    assert d["presets"] == {"Mine": "old text", "Other": "stays", "New one": "Prompt"}
    r = c.post("/api/llm/presets/save", json={"name": "Mine", "text": "new text"})
    assert r.get_json()["replaced"] is True
    assert config.get("llm", "presets") == {
        "Mine": "new text",
        "Other": "stays",
        "New one": "Prompt",
    }
    # it is on disk, not only in memory
    import yaml

    saved = yaml.safe_load(Path(config._config_path).read_text(encoding="utf-8"))
    assert saved["llm"]["presets"]["Mine"] == "new text"


@pytest.mark.parametrize(
    "body",
    [
        {"name": "", "text": "x"},
        {"name": "x" * 81, "text": "x"},
        {"name": "Name", "text": "   "},
        {},
    ],
)
def test_a_template_needs_a_name_and_a_text(client, body):
    c, config = client
    config.set("llm", "presets", {"Mine": "kept"})
    assert c.post("/api/llm/presets/save", json=body).status_code == 400
    assert config.get("llm", "presets") == {"Mine": "kept"}


def test_a_summary_is_only_deleted_if_it_is_still_the_one_shown(client):
    c, _config = client
    s.save_summaries(
        "f.mp4", [{"text": "A", "timestamp": 100.0}, {"text": "B", "timestamp": 200.0}]
    )
    # Another tab deleted A meanwhile: index 0 is now B — refuse
    s.save_summaries("f.mp4", [{"text": "B", "timestamp": 200.0}])
    r = c.post("/api/summaries/delete", json={"filename": "f.mp4", "index": 0, "timestamp": 100.0})
    assert r.status_code == 409
    assert [x["text"] for x in s.load_summaries("f.mp4")] == ["B"]
    # The right one goes
    r = c.post("/api/summaries/delete", json={"filename": "f.mp4", "index": 0, "timestamp": 200.0})
    assert r.status_code == 200 and s.load_summaries("f.mp4") == []


def test_deleting_without_a_timestamp_still_works(client):
    # Older callers (the transcript's summary view) send only the index
    c, _config = client
    s.save_summaries("g.mp4", [{"text": "A", "timestamp": time.time()}])
    r = c.post("/api/summaries/delete", json={"filename": "g.mp4", "index": 0})
    assert r.status_code == 200 and s.load_summaries("g.mp4") == []


# ── Settings ────────────────────────────────────────────────────────────────


def test_the_settings_check_reads_the_statuses_the_api_sends():
    # A status rename once left settings.js waiting for 'new_value'
    # and 'deviating', which never came: "all up to date", always
    api = (REPO / "whisper_tm/api/llm.py").read_text(encoding="utf-8")
    route = api[api.index("def default_presets") : api.index("def presets_merge")]
    sent = set(re.findall(r'"(new|same|different)"', route))
    js = (REPO / "whisper_tm/static/js/settings_tools.js").read_text(encoding="utf-8")
    check = js[
        js.index("async function _checkDefaultPresets") : js.index(
            "async function _applyDefaultPresets"
        )
    ]
    used = set(re.findall(r"p\.status === '(\w+)'", check))
    assert used == {"new", "different"} and used <= sent

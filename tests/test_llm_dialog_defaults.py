"""What the LLM dialogs send when nobody changes anything.

Two settings never reached a run started from a dialog:

* The task → provider assignment (``llm.tasks``). The "summarize afterwards"
  option of the transcription dialogs (and the summarize dialog, see
  tests/test_summarize_dialog.py) put the legacy ``llm.model`` into their
  default entry by NAME, and a named model pulls its own provider along
  (``llm_client.build_chat_payload``): the provider assigned to summaries
  was never used from there. The default entry sends '' — as the correction
  and translation dialogs already did — and names the model the assignment
  picks instead of the legacy one.
* ``llm.correction.score_threshold``: the correction dialog's field held a
  fixed 0.40 in the markup, and every run sent it.

The logic runs in node against the real files with a small fake DOM
(tests/js/llm_dialogs_harness.js).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "llm_dialogs_harness.js"
JS_DIR = REPO / "whisper_tm" / "static" / "js"
INDEX = REPO / "whisper_tm" / "static" / "index.html"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

# /api/llm/status as the server answers it: the legacy llm.model, the quick
# models, the providers' default models, and what each task runs on
STATUS = {
    "configured": True,
    "model": "qwen3",
    "models": {"Qwen": "qwen3", "Nano": "gpt-4.1-nano", "Mini": "gpt-4o-mini"},
    "provider_models": [
        {"model": "qwen3", "providers": "Local (llama.cpp)"},
        {"model": "gpt-4.1-nano", "providers": "OpenAI"},
    ],
    "task_models": {"summary": "gpt-4.1-nano", "correction": "qwen3", "translation": "gpt-4o-mini"},
    "presets": {"Summary": "Summarize the transcript."},
    "correction_score_threshold": 0.55,
    "auto_summarize": True,
    "auto_summarize_preset": "",
    "auto_summarize_model": "",
}


def js(body: str, **setup) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt({json.dumps(setup)});\n"
        f"const STATUS = {json.dumps(STATUS)};\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "const $ = (id) => h.get(id);\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"JS_DIR": str(JS_DIR), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


def values(options: list) -> list:
    return [o["value"] for o in options]


# ── "Summarize afterwards" in the transcription dialogs (txoptions.js) ───

TXO = {"files": ["txoptions.js"], "selects": ["optSumPreset", "optSumModel"]}


@node
class TestSummarizeAfterwards:
    def test_the_default_leaves_the_model_to_the_task_setting(self):
        r = js(
            """
            h.run(`_txoLlm = ${JSON.stringify(STATUS)}`);
            h.run("_txoFillSumFields('opt')");
            h.run("_txoSumPreset('opt')");
            output({options: $('optSumModel').options,
                    sent: JSON.parse(JSON.stringify(h.run("_sumSettings('opt')")))});
        """,
            **TXO,
        )
        assert r["options"][0] == {
            "value": "",
            "textContent": "Default (task setting): gpt-4.1-nano",
        }
        # The legacy model stays selectable by name; the summary task's own
        # model isn't listed a second time — the default runs it
        assert values(r["options"]) == ["", "qwen3", "gpt-4o-mini"]
        assert r["sent"] == {
            "summarize": True,
            "summarize_preset": "Summary",
            "summarize_model": "",
        }

    def test_a_model_preset_in_the_settings_is_still_chosen(self):
        r = js(
            """
            const status = {...STATUS, auto_summarize_model: 'qwen3'};
            h.run(`_txoLlm = ${JSON.stringify(status)}`);
            h.run("_txoFillSumFields('opt')");
            h.run("_txoSumPreset('opt')");
            output(h.run("_sumSettings('opt')").summarize_model);
        """,
            **TXO,
        )
        assert r == "qwen3"

    def test_without_a_known_task_model_the_default_is_still_there(self):
        # Without it the first quick model would be the silent default —
        # by name, with the same effect as the old bug
        r = js(
            """
            const status = {...STATUS, task_models: {}, model: ''};
            h.run(`_txoLlm = ${JSON.stringify(status)}`);
            h.run("_txoFillSumFields('opt')");
            output({first: $('optSumModel').options[0], value: $('optSumModel').value});
        """,
            **TXO,
        )
        assert r == {"first": {"value": "", "textContent": "Default (task setting)"}, "value": ""}


# ── Correction, translation and folder dialogs (llmtools.js) ─────────────

TOOLS = {
    "files": ["llmtools.js"],
    "selects": ["corrModel", "transModel", "folderLlmModel"],
    "answers": {"/api/llm/status": [200, STATUS], "/api/llm/queue/add": [200, {"ids": [7]}]},
}

# The run itself is not what is tested: attaching to it only records the id
OPEN = """
    h.run("_corrTrack = (id) => { globalThis.tracked = id; }");
    h.run("openCorrectionModal('a.mp4')");
"""


@node
class TestCorrectionDialog:
    def test_the_threshold_comes_from_the_settings(self):
        r = js(
            OPEN
            + """
            const before = $('corrScore').value;       // the answer isn't there yet
            await h.settle();
            const filled = $('corrScore').value;
            $('corrOnlySelected').checked = true;
            await h.run('startCorrection()');
            output({before, filled, options: $('corrModel').options,
                    sent: h.requests.find(q => q.url === '/api/llm/queue/add').body});
        """,
            **TOOLS,
        )
        assert r["before"] == ""
        assert r["filled"] == "0.55"
        assert r["options"][0] == {"value": "", "textContent": "Default (task setting): qwen3"}
        assert r["sent"]["kind"] == "correction"
        assert r["sent"]["params"] == {"only_marked": True, "model": "", "score_threshold": 0.55}

    def test_a_value_typed_meanwhile_wins(self):
        r = js(
            OPEN
            + """
            $('corrScore').value = '0.7';
            await h.settle();
            await h.run('startCorrection()');
            output({field: $('corrScore').value,
                    sent: h.requests.find(q => q.url === '/api/llm/queue/add').body.params});
        """,
            **TOOLS,
        )
        assert r["field"] == "0.7"
        assert r["sent"]["score_threshold"] == 0.7

    def test_without_an_answer_the_server_applies_the_setting(self):
        # Empty sends no threshold at all; the correction then reads
        # llm.correction.score_threshold itself
        tools = {**TOOLS, "answers": {**TOOLS["answers"], "/api/llm/status": [500, {}]}}
        r = js(
            OPEN
            + """
            await h.settle();
            await h.run('startCorrection()');
            output({field: $('corrScore').value,
                    sent: h.requests.find(q => q.url === '/api/llm/queue/add').body.params});
        """,
            **tools,
        )
        assert r["field"] == ""
        assert "score_threshold" not in r["sent"]


def test_the_markup_holds_no_fixed_threshold():
    html = INDEX.read_text(encoding="utf-8")
    field = re.search(r'<input[^>]*id="corrScore"[^>]*>', html).group(0)
    # 0.40 is only the hint while the settings load, never a value
    assert "value=" not in field
    assert 'placeholder="0.40"' in field


@node
class TestOtherDialogs:
    def test_translation_names_its_task_model(self):
        r = js(
            """
            await h.run("_llmModels('transModel', 'translation')");
            output($('transModel').options);
        """,
            **TOOLS,
        )
        assert r[0] == {"value": "", "textContent": "Default (task setting): gpt-4o-mini"}
        assert values(r) == ["", "qwen3", "gpt-4.1-nano"]

    def test_the_folder_dialog_stays_generic(self):
        # Its task can change after the list is built: no model is named,
        # and every model is offered by name
        r = js(
            """
            await h.run("_llmModels('folderLlmModel')");
            output($('folderLlmModel').options);
        """,
            **TOOLS,
        )
        assert r[0] == {"value": "", "textContent": "Default (task setting)"}
        assert values(r) == ["", "qwen3", "gpt-4.1-nano", "gpt-4o-mini"]

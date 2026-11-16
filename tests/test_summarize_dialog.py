"""The summarize dialog: the template is the system prompt, always in view.

The dialog is large, always shows the system prompt, and offers a tidy
set of templates. Previously a preset replaced the system prompt behind
a closed toggle — what went to the LLM
could not be seen — and the preset list had grown German duplicates of
the English built-ins.

The logic under test (static/js/summarize.js, via tests/js/summarize_harness.js):
which template the dialog opens with, when the prompt counts as edited,
that switching away from an edited prompt can be undone, the hints when a
template needs speaker labels or timestamps that are switched off, and what
a run sends and is saved as.
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
HARNESS = REPO / "tests" / "js" / "summarize_harness.js"
JS = REPO / "whisper_tm" / "static" / "js" / "summarize.js"
INDEX = REPO / "whisper_tm" / "static" / "index.html"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

SUMMARY = "Summarize the transcript.\nWrite in the language of the transcript."
STATUS = {
    "system_prompt": SUMMARY,
    "preset_builtin": ["Summary", "Chapter Marks", "Speaker Analysis"],
    "presets": {
        "Summary": SUMMARY,
        "Chapter Marks": "Chapters with timestamps, copied exactly.",
        "Speaker Analysis": "Work from the speaker labels.",
        "Zebra notes": "My own template.",
        "Funny Moments": "Every moment with its timestamp.",
    },
    "model": "qwen3",
    "models": {},
}


def js(body: str, **setup) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt({json.dumps(setup)});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "const $ = (id) => h.get(id);\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"SUMMARIZE_JS": str(JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@node
class TestOpening:
    def test_the_default_template_is_chosen_and_its_text_shown(self):
        r = js(
            """
            h.run('_populatePresets()');
            output({value: $('sumPreset').value, text: $('sumSysPrompt').value,
                    options: $('sumPreset').options.map(o => o.value),
                    groups: $('sumPreset').innerHTML.match(/label="[^"]+"/g)});
        """,
            status=STATUS,
        )
        assert r["value"] == "Summary"
        assert r["text"] == SUMMARY
        # Built-ins first in their own order, then yours alphabetically;
        # no separate default entry when a template carries its text
        assert r["options"] == [
            "Summary",
            "Chapter Marks",
            "Speaker Analysis",
            "Funny Moments",
            "Zebra notes",
        ]
        assert r["groups"] == ['label="Built-in"', 'label="Your templates"']

    def test_a_default_prompt_that_is_no_template_gets_its_own_entry(self):
        status = {**STATUS, "system_prompt": "Something else entirely."}
        r = js(
            """
            h.run('_populatePresets()');
            output({value: $('sumPreset').value, text: $('sumSysPrompt').value,
                    first: $('sumPreset').options[0]});
        """,
            status=status,
        )
        assert r["value"] == "__default__"
        assert r["text"] == "Something else entirely."
        assert r["first"]["textContent"] == "Default prompt (Settings)"

    def test_the_template_used_last_comes_back(self):
        r = js(
            """
            h.run('_populatePresets()');
            output({value: $('sumPreset').value, text: $('sumSysPrompt').value});
        """,
            status=STATUS,
            prefs={"template": "Zebra notes"},
        )
        assert r == {"value": "Zebra notes", "text": "My own template."}

    def test_a_remembered_template_that_is_gone_falls_back(self):
        r = js(
            """
            h.run('_populatePresets()');
            output($('sumPreset').value);
        """,
            status=STATUS,
            prefs={"template": "Deleted meanwhile"},
        )
        assert r == "Summary"


@node
class TestEditing:
    def test_an_edit_is_marked_and_can_be_reset(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = SUMMARY_EDITED;
            h.run('_sumSysInput()');
            const edited = {badge: !$('sumEdited').hidden, reset: !$('btnSumReset').hidden,
                            count: $('sumSysCount').textContent};
            h.run('_sumResetPrompt()');
            output({edited, after: {badge: !$('sumEdited').hidden, text: $('sumSysPrompt').value}});
        """.replace("SUMMARY_EDITED", json.dumps(SUMMARY + "\nOnly the decisions.")),
            status=STATUS,
        )
        assert r["edited"]["badge"] and r["edited"]["reset"]
        assert re.match(r"\d+ characters · ~\d+ tokens", r["edited"]["count"])
        assert r["after"] == {"badge": False, "text": SUMMARY}

    def test_whitespace_alone_is_no_edit(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = $('sumSysPrompt').value + '\\n\\n';
            h.run('_sumSysInput()');
            output(!$('sumEdited').hidden);
        """,
            status=STATUS,
        )
        assert r is False

    def test_switching_away_from_an_edit_can_be_undone(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = 'My careful edit';
            h.run('_sumSysInput()');
            $('sumPreset').value = 'Chapter Marks';
            h.run('_onPresetChange()');
            const switched = {text: $('sumSysPrompt').value, undo: !$('btnSumUndo').hidden};
            h.run('_sumUndoSwitch()');
            output({switched, back: {value: $('sumPreset').value, text: $('sumSysPrompt').value,
                                     undo: !$('btnSumUndo').hidden, edited: !$('sumEdited').hidden}});
        """,
            status=STATUS,
        )
        assert r["switched"] == {"text": "Chapters with timestamps, copied exactly.", "undo": True}
        assert r["back"] == {
            "value": "Summary",
            "text": "My careful edit",
            "undo": False,
            "edited": True,
        }

    def test_switching_without_an_edit_offers_no_undo(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumPreset').value = 'Zebra notes';
            h.run('_onPresetChange()');
            output({undo: !$('btnSumUndo').hidden, saved: JSON.parse(h.store.wtm_sumPrefs).template});
        """,
            status=STATUS,
        )
        assert r == {"undo": False, "saved": "Zebra notes"}

    def test_the_edited_prompt_survives_closing_and_reopening(self):
        # Esc must never throw the work away: the draft outlives the dialog
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = 'Draft';
            h.run('_sumSysInput()');
            $('sumSysPrompt').value = '';          // as if the DOM were rebuilt
            h.run('_populatePresets()');
            output({value: $('sumPreset').value, text: $('sumSysPrompt').value,
                    edited: !$('sumEdited').hidden});
        """,
            status=STATUS,
        )
        assert r == {"value": "Summary", "text": "Draft", "edited": True}


@node
class TestHints:
    def test_speaker_labels_asked_for_but_switched_off(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumPreset').value = 'Speaker Analysis';
            h.run('_onPresetChange()');
            const off = $('sumHints').innerHTML;
            $('sumSpeakers').checked = true;
            h.run('_sumHints()');
            output({off, on: $('sumHints').innerHTML});
        """,
            status=STATUS,
        )
        assert "speaker labels" in r["off"] and "_sumTurnOn('sumSpeakers')" in r["off"]
        assert r["on"] == ""

    def test_timestamps_asked_for_but_switched_off(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumPreset').value = 'Funny Moments';
            h.run('_onPresetChange()');
            $('sumTimestamps').checked = false;
            h.run('_sumHints()');
            output($('sumHints').innerHTML);
        """,
            status=STATUS,
        )
        assert "_sumTurnOn('sumTimestamps')" in r


@node
class TestRun:
    def test_what_a_run_sends(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = 'Edited prompt';
            h.run('_sumSysInput()');
            $('sumPrompt').value = 'only the decisions';
            await h.run('doSummarize()');
            output(h.queued[0]);
        """,
            status=STATUS,
        )
        p = r["params"]
        assert r["kind"] == "summary"
        assert p["system_prompt"] == "Edited prompt"
        assert p["preset"] == "Summary"
        assert p["prompt"] == "only the decisions"
        # Saved as what was used — edited and with the instruction
        assert p["label"] == "Summary (edited) · + “only the decisions”"
        assert p["save"] is True
        assert p["metadata"] == {"filename": "a.mp4", "duration": ""}

    def test_named_speakers_go_by_their_names(self):
        # Named in the file view, a speaker was still "SPEAKER_00" in the
        # LLM input view and in the speaker list sent with a run
        r = js(
            """
            h.run(`activeTranscript = {filename: 'a.mp4', speakers: ['SPEAKER_00', 'SPEAKER_01'],
                segments: [{start: 0, speaker: 'SPEAKER_00', text: 'hi'},
                           {start: 1, speaker: 'SPEAKER_01', text: 'ho'}]};
                getSpeakerName = id => ({SPEAKER_00: 'Alice'})[id] || id;
                document.querySelector('#sumMeta .sum-toggle[data-meta="speakers"]')
                    .setAttribute('aria-pressed', 'true');`);
            h.run('_populatePresets()');
            await h.run('doSummarize()');
            output({posted: h.run('_sumPostSegments()').map(x => x.speaker),
                    meta: h.queued[0].params.metadata});
        """,
            status=STATUS,
        )
        assert r["posted"] == ["Alice", "SPEAKER_01"]
        assert r["meta"]["speakers"] == "Alice, SPEAKER_01"

    def test_an_empty_system_prompt_is_sent_as_empty(self):
        # '' deliberately means "no system prompt" — never the default
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = '';
            h.run('_sumSysInput()');
            await h.run('doSummarize()');
            output({sys: h.queued[0].params.system_prompt, count: $('sumSysCount').textContent});
        """,
            status=STATUS,
        )
        assert r["sys"] == ""
        assert "no system prompt" in r["count"]


@node
class TestModel:
    """The default entry sends '' so the provider assigned to summaries in the
    settings (llm.tasks) decides. Previously it sent the legacy llm.model
    by name, and a named model pulls its own provider along
    (llm_client.build_chat_payload): the summary assignment was never used
    from this dialog."""

    MODELS = {
        **STATUS,
        "model": "qwen3",
        "task_models": {"summary": "gpt-4.1-nano"},
        "models": {"Qwen": "qwen3", "Nano": "gpt-4.1-nano"},
        "provider_models": [
            {"model": "qwen3", "providers": "Local"},
            {"model": "gpt-4o-mini", "providers": "OpenAI"},
        ],
    }

    def test_the_default_leaves_the_model_to_the_task_setting(self):
        r = js(
            """
            h.run('_populatePresets()');
            h.run('_populateModels()');
            await h.run('doSummarize()');
            output({options: $('sumModel').options, sent: h.queued[0].params.model});
        """,
            status=self.MODELS,
        )
        assert r["options"][0] == {
            "value": "",
            "textContent": "Default (task setting): gpt-4.1-nano",
        }
        # The legacy model stays selectable by name; the task's own model
        # isn't listed a second time — the default runs it
        assert [o["value"] for o in r["options"]] == ["", "qwen3", "gpt-4o-mini"]
        assert r["sent"] == ""

    def test_a_named_model_is_sent_by_name(self):
        r = js(
            """
            h.run('_populatePresets()');
            h.run('_populateModels()');
            $('sumModel').value = 'qwen3';
            await h.run('doSummarize()');
            output(h.queued[0].params.model);
        """,
            status=self.MODELS,
        )
        assert r == "qwen3"

    def test_without_a_known_task_model_the_default_is_still_there(self):
        # Without it the first model would be the silent default — by name
        status = {**self.MODELS, "task_models": {}, "model": ""}
        r = js(
            """
            h.run('_populateModels()');
            output({first: $('sumModel').options[0], value: $('sumModel').value});
        """,
            status=status,
        )
        assert r == {"first": {"value": "", "textContent": "Default (task setting)"}, "value": ""}


@node
class TestSaveAsTemplate:
    def test_saving_a_new_template(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumSysPrompt').value = 'Brand new';
            h.run('_sumSysInput()');
            h.run('_sumSaveTplOpen()');
            const prefilled = $('sumTplName').value;
            $('sumTplName').value = 'Mine';
            h.run('_sumTplNameInput()');
            const button = $('btnSumTplSave').textContent;
            await h.run('_sumSaveTpl()');
            output({prefilled, button, request: h.requests.find(q => q.url === '/api/llm/presets/save'),
                    value: $('sumPreset').value, edited: !$('sumEdited').hidden,
                    toasts: h.toasts});
        """,
            status=STATUS,
            answers={
                "/api/llm/presets/save": [
                    200,
                    {
                        "ok": True,
                        "replaced": False,
                        "presets": {**STATUS["presets"], "Mine": "Brand new"},
                    },
                ]
            },
        )
        # A built-in stays as shipped: no name offered to overwrite it
        assert r["prefilled"] == ""
        assert r["button"] == "Save"
        assert r["request"]["body"] == {"name": "Mine", "text": "Brand new"}
        assert r["value"] == "Mine" and r["edited"] is False
        assert r["toasts"] == [["success", "Template “Mine” saved"]]

    def test_your_own_template_is_offered_for_update(self):
        r = js(
            """
            h.run('_populatePresets()');
            $('sumPreset').value = 'Zebra notes';
            h.run('_onPresetChange()');
            $('sumSysPrompt').value = 'Better notes';
            h.run('_sumSysInput()');
            h.run('_sumSaveTplOpen()');
            output({name: $('sumTplName').value, button: $('btnSumTplSave').textContent,
                    note: $('sumTplNote').textContent});
        """,
            status=STATUS,
        )
        assert r["name"] == "Zebra notes"
        assert r["button"] == "Replace"
        assert "replaces its text" in r["note"]


def test_the_dialog_markup_keeps_the_prompt_in_view():
    html = INDEX.read_text(encoding="utf-8")
    dialog = html[html.index('id="summarizeModal"') : html.index("<!-- Find & Replace Modal -->")]
    # The system prompt is a plain textarea in the dialog — no toggle hides it
    ta = re.search(r'<textarea id="sumSysPrompt"[^>]*>', dialog).group(0)
    assert "display:none" not in ta and "hidden" not in ta
    assert "_toggleSysPrompt" not in dialog
    # Result, exact input and saved summaries side by side with the prompt
    for tab in ("sumTabResult", "sumTabInput", "sumTabSaved"):
        assert f'id="{tab}"' in dialog
    # No bare ✕ besides the close button: Esc presses the first one it finds
    assert dialog.count(">✕<") == 1

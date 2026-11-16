"""The transcription dialogs and their defaults.

* "Make these my defaults" writes a dialog's choices back to the settings
  the dialog starts from — so every option that is preset from a setting
  must name that setting, or the button would silently skip it.
* The artifact filter is off by default: what it would take out is tried
  and applied deliberately (Settings → Transcription → Clean-up).
* The shipped vocabulary is an example of the form (a sentence), and the
  dialogs can start with a vocabulary (transcription.default_vocabulary).
* A file row with +Q/⤒ keeps its TX button clickable: the empty hover
  overlay lay over it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "whisper_tm" / "static"
DEFAULTS = yaml.safe_load((REPO / "config.default.yaml").read_text(encoding="utf-8"))


def _get(path):
    node = DEFAULTS
    for k in path.split("."):
        node = node[k]
    return node


def _rows():
    js = (STATIC / "js" / "txoptions.js").read_text(encoding="utf-8")
    table = js[js.index("const TX_OPTIONS = [") : js.index("];", js.index("const TX_OPTIONS = ["))]
    # One piece per row: from its "{key: '" to the next one
    return [(part.split("'", 1)[0], part) for part in table.split("{key: '")[1:]]


class TestMakeTheseMyDefaults:
    def test_every_preset_option_names_its_setting(self):
        rows = _rows()
        assert len(rows) >= 10
        for key, body in rows:
            if "before:" in body:
                assert "setting:" in body, (
                    f"{key}: preset from a setting, but no setting to write back"
                )

    def test_every_setting_exists(self):
        for key, body in _rows():
            m = re.search(r"setting: '([\w.]+)'", body)
            if m:
                _get(m.group(1))  # KeyError = a setting that doesn't exist

    def test_the_button_is_in_every_transcription_dialog(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for prefix in ("opt", "batch", "up"):
            assert f"txSaveDefaults('{prefix}', this)" in html, prefix

    def test_only_what_changed_is_sent(self):
        js = (STATIC / "js" / "txoptions.js").read_text(encoding="utf-8")
        fn = js[js.index("async function txSaveDefaults") :]
        fn = fn[: fn.index("\n}\n")]
        assert "JSON.stringify(saved(path) ?? null) === JSON.stringify(value)" in fn
        assert "These already are your defaults" in fn


class TestShippedDefaults:
    def test_the_artifact_filter_is_off(self):
        assert _get("transcription.filter_hallucinations") is False

    def test_the_vocabulary_is_one_sentence_example(self):
        presets = _get("transcription.vocab_presets")
        assert len(presets) == 1
        text = next(iter(presets.values()))
        # A sentence, not a bare comma list — a list comes back as a list
        assert text.startswith("The conversation is about") and text.endswith(".")
        assert _get("transcription.default_vocabulary") == ""

    def test_the_default_vocabulary_is_on_the_settings_page(self):
        pages = (STATIC / "js" / "settings_pages.js").read_text(encoding="utf-8")
        assert "key: 'transcription.default_vocabulary'" in pages


class TestTheFileRow:
    def test_an_empty_hover_overlay_is_not_drawn(self):
        # Rows with +Q/⤒ carry TX inline; the empty overlay still covered it
        css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
        assert re.search(r"\.fi-actions:empty\s*\{\s*display:\s*none;?\s*\}", css)
        js = (STATIC / "js" / "filelist.js").read_text(encoding="utf-8")
        assert '<div class="fi-actions">${overlayActions}</div>' in js

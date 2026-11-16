"""The translation dialogs' target language.

Both dialogs opened on the first entry of their list, German, for everybody
— a new user of an English-default app who clicked through got a German
translation. They open on the language chosen last, else on the browser's
when the list has it, else on English; and English leads the lists.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"

HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.env.JS, 'utf8');
function fn(name) {
    const at = src.indexOf('function ' + name + '(');
    return src.slice(at, src.indexOf('\n}\n', at) + 3);
}
const at = src.indexOf('const _LLM_TARGETS');
eval(src.slice(at, src.indexOf('\n', at)) + '\n' + fn('llmPresetTarget') + fn('llmRememberTarget')
     + '\nthis.preset = llmPresetTarget; this.remember = llmRememberTarget;');
const run = (browser, stored, withFree = true) => {
    const store = {};
    if (stored) store.llmTargetLanguage = stored;
    global.localStorage = {getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; }};
    // node has a navigator of its own, and it is read-only by assignment
    Object.defineProperty(globalThis, 'navigator', {value: {language: browser}, configurable: true});
    global.Event = function (t) { this.type = t; };
    const sel = {value: 'German', options: ['English', 'German', 'Japanese', '_free'].map(value => ({value})),
                 dispatchEvent: () => {}};
    const free = {value: ''};
    global.document = {getElementById: id => id === 'sel' ? sel : (withFree && id === 'free' ? free : null)};
    this.preset('sel', 'free');
    return [sel.value, free.value];
};
console.log(JSON.stringify([
    run('en-US', ''), run('de-DE', ''), run('ja', ''), run('pl-PL', ''), run('', ''),
    run('en-US', 'German'), run('de-DE', 'Polish'), run('de-DE', 'Polish', false),
]));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_the_dialog_opens_on_the_last_or_the_browsers_language():
    import os

    done = subprocess.run(
        ["node", "-e", HARNESS],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "JS": str(STATIC / "js" / "llmtools.js")},
    )
    assert done.returncode == 0, done.stderr[-600:]
    assert json.loads(done.stdout) == [
        ["English", ""],
        ["German", ""],
        ["Japanese", ""],  # the browser's language
        ["English", ""],
        ["English", ""],  # not in the list, or unknown: English
        ["German", ""],  # the one chosen last wins
        ["_free", "Polish"],  # a typed one comes back as typed
        ["German", ""],  # no field to type into: the browser's
    ]


def test_english_leads_the_language_lists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for sel in ("transLanguage", "folderLlmLanguage"):
        at = html.index(f'id="{sel}"')
        first = html[at:].split("<option", 2)[1]
        assert 'value="English"' in first, sel
    pages = (STATIC / "js" / "settings_pages.js").read_text(encoding="utf-8")
    assert "['auto', 'Detect automatically'], ['en', 'English'], ['de', 'German']" in pages
    assert "[['en', 'English'], ['de', 'German']]" in pages
    tx = (STATIC / "js" / "txoptions.js").read_text(encoding="utf-8")
    common = tx[tx.index('<optgroup label="Common">') :]
    assert common.index('value="en"') < common.index('value="de"')
    js = (STATIC / "js" / "llmtools.js").read_text(encoding="utf-8")
    assert js.count("llmPresetTarget('") == 2 and js.count("llmRememberTarget(") == 3

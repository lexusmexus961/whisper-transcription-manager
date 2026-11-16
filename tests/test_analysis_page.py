"""The Analyses tab of the search workspace (static/js/analysis.js).

What the page has to get right on its own: the passages an answer cites
become chips that play (unknown ones stay plain), the passages panel lists
them in the answer's order and then the notes', the count before a start
says what will be read — or why nothing can be — the start button only
goes when there is a set and a task, and the set is described to the
server as the file list's selection, a folder or the recordings of a search.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const src = n => fs.readFileSync(process.env.JS + '/' + n, 'utf8');
function fn(file, name) {
    const s = src(file);
    for (const head of ['async function ' + name + '(', 'function ' + name + '(']) {
        const at = s.indexOf(head);
        if (at >= 0) return s.slice(at, s.indexOf('\n}\n', at) + 3);
    }
    throw new Error('missing ' + name);
}
const store = {};
const el = id => store[id] || (store[id] = {id, innerHTML: '', textContent: '', value: '', hidden: false,
                                            checked: true, disabled: false, options: [],
                                            classList: {toggle() {}}, focus() {}, setSelectionRange() {}});
const els = new Proxy(store, {get: (t, id) => el(id)});    // made on first use
const ctx = {
    console, JSON, Promise, Number, String, Object, Set, Map, Math, encodeURIComponent,
    document: {getElementById: el, querySelectorAll: () => []},
    setTimeout: () => 0, clearTimeout: () => {},
    esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'),
    escJs: s => String(s ?? ''),
    // core.js's counted noun: "1 window", "3 windows"
    plural: (n, one, many) => {
        const k = Number(n) || 0;
        return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
    },
    _renderMarkdown: s => '<p>' + s + '</p>',
    svState: {tab: 'analysis', query: ''}, svHasFilter: () => false,
    Toast: {success() {}, error() {}, info() {}},
    selectedFiles: new Set(),
};
vm.createContext(ctx);
vm.runInContext([
    src('analysis.js'),
    fn('chat.js', 'chatWhen'), fn('chat.js', 'chatRefTitle'), fn('chat.js', 'chatSpeaker'),
    fn('chat.js', 'chatFoldRefs'),
    ...src('chat.js').split('\n').filter(l => l.startsWith('const CHAT_REF_RUN')),
    "const CHAT_REFS_RE = /\\[((?:S\\d+)(?:\\s*[,;]\\s*S\\d+)*)\\]/g;",
    fn('searchview.js', 'svShortName'), fn('searchview.js', 'svFmt'),
    'this.AN = AN;',
].join('\n'), ctx);
const out = o => console.log('§§' + JSON.stringify(o));
"""


def js(body: str) -> dict:
    done = subprocess.run(
        ["node", "-e", HARNESS + f"(async () => {{ {body} }})();"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"JS": str(JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, done.stdout + done.stderr
    return json.loads(lines[-1][2:])


REFS = {
    "S1": {
        "filename": "talks/a.mp4",
        "file": "talks/a.mp4",
        "start": 65,
        "text": "one",
        "speaker": "Alice",
        "file_dir": "/media",
    },
    "S2": {
        "filename": "talks/b.mp4",
        "file": "talks/b.mp4",
        "start": 3,
        "text": "two",
        "speaker": "—",
        "file_dir": "/media",
    },
}


def test_references_become_chips_that_play():
    d = js(f"""ctx.AN.cur = {{refs: {json.dumps(REFS)}}};
              out({{html: ctx.anMd('Both say so [S2, S1] — and [S9].')}});""")
    html = d["html"]
    assert "onclick=\"anRefPlay('S2')\"" in html and "onclick=\"anRefPlay('S1')\"" in html
    assert "S1 ▶ 1:05" in html and "S2 ▶ 0:03" in html
    assert "a.mp4 · 1:05 · Alice" in html  # the chip's title
    assert 'title="Not among the passages">S9</span>' in html
    short = js(f"""ctx.AN.cur = {{refs: {{S1: {{...{json.dumps(REFS["S1"])}, short: true}}}}}};
                   out({{html: ctx.anMd('A line [S1].')}});""")["html"]
    assert ">S1 ▶</button>" in short  # a whole short recording: no time


def test_a_long_run_of_references_folds_after_its_first_few():
    refs = {f"S{i}": {**REFS["S1"], "start": i} for i in range(1, 11)}
    d = js(f"""ctx.AN.cur = {{refs: {json.dumps(refs)}}};
              out({{many: ctx.anMd('Mud [S1][S2][S3][S4][S5][S6][S7][S8][S9][S10].'),
                    few: ctx.anMd('Two [S1][S2].')}});""")
    many = d["many"]
    assert many.count('class="sv-ref"') == 10  # every one still there
    assert '<span class="sv-ref-more">' in many and "+6 more" in many
    assert many.index("S4 ▶") < many.index("sv-ref-more") < many.index("S5 ▶")
    assert "sv-ref-run" not in d["few"]


def test_the_passages_in_the_answers_order_then_the_notes():
    d = js(f"""ctx.AN.cur = {{refs: {json.dumps(REFS)}, result: 'First [S2], again [S2].',
                              parts: [{{text: 'note [S1] [S7]'}}, {{text: ''}}]}};
              out({{cited: ctx.anCited()}});""")
    assert d["cited"] == ["S2", "S1"]


def test_the_count_before_a_start():
    d = js("""const n = (p) => ctx.anPlanHtml(p);
              out({
                empty: n({files: 0}),
                big: n({files: 30000, too_large: true, transcribed: 0, limit_files: 20000}),
                one: n({files: 3, transcribed: 3, words: 7500, single: true, estimate_sec: 95,
                        sample: ['a', 'b', 'c']}),
                parts: n({files: 5000, transcribed: 4990, missing: 2, empty: 8, words: 50000,
                          single: false, parts: 7, estimate_sec: 660, sample: ['x']}),
                many: n({files: 9000, transcribed: 9000, words: 9, single: false, parts: 500,
                         max_parts: 400, too_large: true}),
              });""")
    assert d["empty"] == '<span class="an-muted">No recordings there.</span>'
    assert "30,000 files — more than one analysis takes (20,000)" in d["big"]
    assert "<b>3</b> recordings with a transcript" in d["one"]
    assert "read in <b>one go</b> · about 2 min" in d["one"]
    assert "<b>4,990</b> recordings" in d["parts"] and "2 without one" in d["parts"]
    assert "8 without speech" in d["parts"] and "50,000 words" in d["parts"]
    assert "<b>7 parts</b>, then one answer from their notes · about 11 min" in d["parts"]
    assert "<span>…</span>" in d["parts"]  # more than the sample
    assert "500 parts are more than one analysis reads (400)" in d["many"]


def test_the_start_button_wants_a_set_and_a_task():
    d = js("""const st = () => (ctx.anStartSync(), [els.anStartBtn.disabled, els.anStartNote.textContent]);
              const r = {};
              r.nothing = st();
              ctx.AN.plan = {transcribed: 5};
              r.noTask = st();
              els.anTask.value = 'Find small talk.';
              r.ready = st();
              ctx.AN.plan = {transcribed: 5, too_large: true};
              r.large = st();
              out(r);""")
    assert d["nothing"] == [True, "Choose recordings that have a transcript."]
    assert d["noTask"] == [True, "Write the task — or start from a template."]
    assert d["ready"] == [False, "It runs in the LLM queue — you can close this."]
    assert d["large"] == [True, "The set is too large."]


def test_the_set_as_the_server_gets_it():
    d = js("""const r = {};
              els.anFolder.value = '/mnt/media/talks'; els.anRecursive.checked = false;
              ctx.AN.src = 'folder'; r.folder = ctx.anSourceSpec();
              els.anFolder.value = '  '; r.none = ctx.anSourceSpec();
              ctx.AN.src = 'selection'; ctx.AN.selection = {base_dir: '/mnt', filenames: ['a.wav', 'b/c.wav']};
              r.selection = ctx.anSourceSpec();
              ctx.AN.src = 'search'; ctx.AN.search = null; r.searching = ctx.anSourceSpec();
              ctx.AN.search = {label: '"Bob"', abs_paths: ['/mnt/a.wav'], more: false};
              r.search = ctx.anSourceSpec();
              out(r);""")
    assert d["folder"] == {"kind": "folder", "folder": "/mnt/media/talks", "recursive": False}
    assert d["none"] is None and d["searching"] is None
    assert d["selection"] == {
        "kind": "files",
        "base_dir": "/mnt",
        "filenames": ["a.wav", "b/c.wav"],
    }
    assert d["search"] == {"kind": "paths", "abs_paths": ["/mnt/a.wav"], "label": '"Bob"'}


def test_the_list_says_how_far_each_one_is():
    d = js("""ctx.AN.list = [
                {id: 'a1', title: 'Small talk', status: 'running', progress: 42, created: 1,
                 counts: {transcribed: 4990}},
                {id: 'b2', title: 'Three videos', status: 'done', created: 1, counts: {files: 3}},
                {id: 'c3', title: 'Gone', status: 'cancelled', created: 1}];
              ctx.anRenderList();
              out({html: els.anList.innerHTML, badge: els.svTabNAnalysis.textContent});""")
    html = d["html"]
    assert '<span class="an-dot run"></span>' in html and "4,990 recordings · 42 %" in html
    assert '<span class="an-dot ok"></span>' in html and "3 recordings · done" in html
    assert '<span class="an-dot off"></span>' in html and "cancelled" in html
    assert d["badge"] == "1"


def test_a_template_with_a_gap_selects_the_gap():
    d = js("""ctx.AN.presets = [{key: 'find', label: 'Find every place where …',
                                 task: 'Find every place where … — list each one.'}];
              let sel = null;
              els.anTask.setSelectionRange = (a, b) => { sel = [a, b]; };
              ctx.anPreset('find');
              out({task: els.anTask.value, sel, preset: ctx.AN.preset});""")
    assert d["task"].startswith("Find every place where …")
    assert d["sel"] == [23, 24] and d["preset"] == "find"


def test_a_link_from_outside_opens_the_recording_or_the_analysis():
    """The browser extension's "open": ?open=<full path>&t=<seconds> opens the
    recording there, ?analysis=<id> the analysis — and the address loses the
    parameters, so a reload doesn't jump again (static/js/app.js).

    Opening switches the file list's folder on the server, for everyone:
    a link into another folder than the open one asks first."""
    app = (JS / "app.js").read_text(encoding="utf-8")
    body = app[app.index("function _openFromLink()") :]
    run = r"""
    const vm = require('vm');
    const calls = [];
    const make = (search, dir, answer) => ({
        location: {search, pathname: '/'}, URLSearchParams,
        history: {replaceState: (a, b, u) => calls.push(['replace', u])},
        document: {getElementById: id => (id === 'dirInput' ? {value: dir} : null)},
        confirm: () => (calls.push(['asked']), answer),
        goToSearchResult: (f, t, d) => calls.push(['open', f, t, d]),
        openAnalysis: id => calls.push(['analysis', id]),
    });
    const link = '?open=%2Fmnt%2Fmedia%2Fuploads%2FTalk%20one.m4a&t=12.5';
    for (const [s, dir, answer] of [[link, '/mnt/media/', false],   // inside: no question
                                    [link, '/mnt/med', false],      // elsewhere, declined
                                    [link, '', true],               // elsewhere, allowed
                                    ['?analysis=abc123def456', '/mnt/media', false],
                                    ['', '/mnt/media', false]]) {
        const ctx = make(s, dir, answer);
        vm.createContext(ctx);
        vm.runInContext(process.env.FN + '\n_openFromLink();', ctx);
    }
    console.log('§§' + JSON.stringify(calls));
    """
    done = subprocess.run(
        ["node", "-e", run],
        capture_output=True,
        text=True,
        timeout=60,
        env={"FN": body, "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    calls = json.loads([z for z in done.stdout.splitlines() if z.startswith("§§")][-1][2:])
    opened = ["open", "Talk one.m4a", 12.5, "/mnt/media/uploads"]
    assert calls == [
        ["replace", "/"],
        opened,
        ["replace", "/"],
        ["asked"],
        ["replace", "/"],
        ["asked"],
        opened,
        ["replace", "/"],
        ["analysis", "abc123def456"],
    ]

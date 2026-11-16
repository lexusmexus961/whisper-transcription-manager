"""The transcript view waits for a summary the queue still writes
(static/js/llmtools.js: _txViewsLoadMeta, _txSumPoll, _txViewbarRender).

The transcript of a file queued with "summarize" shows as soon as it is
done; the summary follows about a minute later. /api/summaries says
``pending`` meanwhile — the Summaries tab shows ⏳ and the view asks again
until it is there, then says so (before, the summary was written but the
view never learned of it, so it looked as if none had been made).
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
const src = fs.readFileSync(process.env.JS + '/llmtools.js', 'utf8');
function fn(name) {
    for (const head of ['async function ' + name + '(', 'function ' + name + '(']) {
        const at = src.indexOf(head);
        if (at >= 0) return src.slice(at, src.indexOf('\n}\n', at) + 3);
    }
    throw new Error('missing ' + name);
}
const bar = {innerHTML: ''};
const timers = [], toasts = [], asked = [];
let answers = [];
const ctx = {
    console, JSON, Promise,
    document: {getElementById: id => id === 'txViewBar' ? bar : null},
    setTimeout: (f, ms) => { timers.push([f, ms]); return timers.length; },
    clearTimeout: id => { if (id && timers[id - 1]) timers[id - 1][0] = null; },
    fetch: async url => { asked.push(url); const a = url.includes('/translations/') ? {translations: []}
                                                    : (answers.shift() || {summaries: [], pending: false});
                          return {json: async () => a}; },
    Toast: {success: t => toasts.push(t), error: t => toasts.push('ERR ' + t)},
    esc: s => String(s ?? ''), escJs: s => String(s ?? ''),
    activeFile: 'uploads/talk.m4a', _corrFilename: '',
    switchTxView: () => {}, _txApplyView: () => {},
    _sumViewRender: () => { ctx.sumRendered = (ctx.sumRendered || 0) + 1; },
};
vm.createContext(ctx);
vm.runInContext([
    "let _txView = {mode: 'original', lang: '', tracks: [], map: null, showOriginal: false, sumCount: 0, sumPending: false};",
    'let _sumPollTimer = null;',
    fn('_txViewsLoadMeta'), fn('_txSumPoll'), fn('_txViewbarRender'),
    'this.v = {get: () => _txView, load: _txViewsLoadMeta};',
].join('\n'), ctx);
const runTimer = async () => {
    const t = timers.filter(x => x[0]).pop();
    if (!t) return false;
    const f = t[0]; t[0] = null; await f(); await new Promise(r => setImmediate(r));
    return true;
};
const h = {bar, timers, toasts, asked, ctx, runTimer, answer: a => answers.push(...a)};
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


SUM = {"text": "x", "prompt": "Summary", "timestamp": 1}


def test_the_tab_shows_it_coming_and_the_view_asks_until_it_is_there():
    d = js(f"""h.answer([{{summaries: [], pending: true}}, {{summaries: [], pending: true}},
                         {{summaries: [{json.dumps(SUM)}], pending: false}}]);
              await ctx.v.load();
              const first = {{bar: h.bar.innerHTML, pending: ctx.v.get().sumPending,
                             delay: h.timers.filter(t => t[0]).map(t => t[1])}};
              await h.runTimer();
              const second = {{bar: h.bar.innerHTML, toasts: h.toasts.length}};
              await h.runTimer();
              out({{first, second, last: {{bar: h.bar.innerHTML, count: ctx.v.get().sumCount,
                                         pending: ctx.v.get().sumPending}},
                   toasts: h.toasts, more: await h.runTimer(),
                   asked: h.asked.filter(u => u.includes('/summaries/')).length}});""")
    assert d["first"]["pending"] is True and d["first"]["delay"] == [4000]
    assert "0 · ⏳" in d["first"]["bar"] and "being written" in d["first"]["bar"]
    assert "0 · ⏳" in d["second"]["bar"] and d["second"]["toasts"] == 0
    assert ">1</span>" in d["last"]["bar"] and "⏳" not in d["last"]["bar"]
    assert d["last"]["count"] == 1 and d["last"]["pending"] is False
    assert d["toasts"] == ["🤖 The summary is ready"]
    assert d["more"] is False  # nothing asks any more
    assert d["asked"] == 3


def test_another_file_stops_the_asking():
    d = js("""h.answer([{summaries: [], pending: true}]);
              await ctx.v.load();
              ctx.activeFile = 'other.mp4';
              await h.runTimer();
              out({asked: h.asked.filter(u => u.includes('/summaries/')).length,
                   more: await h.runTimer(), toasts: h.toasts});""")
    assert d == {"asked": 1, "more": False, "toasts": []}


def test_nothing_pending_nothing_asked_again():
    d = js(f"""h.answer([{{summaries: [{json.dumps(SUM)}], pending: false}}]);
              await ctx.v.load();
              out({{bar: h.bar.innerHTML, timers: h.timers.filter(t => t[0]).length}});""")
    assert ">1</span>" in d["bar"] and "⏳" not in d["bar"] and d["timers"] == 0

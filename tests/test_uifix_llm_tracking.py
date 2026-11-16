"""LLM dialogs and the transcript's summary/translation view stay with
what the user is looking at (static/js/llmtools.js, summarize.js,
queue_editor.js).

- llmQueueTrack keeps following a task through a failed poll (network,
  a 502 page while the server restarts) instead of declaring it failed —
  that re-enabled "Summarize" and invited a duplicate run. Only a real
  end (done / error / cancelled) or a 404 for the entry ends it, and a
  stop() during a request in flight keeps its answer away from the dialog.
- The section boxes ("View intermediate results") open from their own
  header: the ids they were looked up by did not match (and repeated).
- The summary view and the translation switch drop answers for a file
  that is no longer open; a refresh of the summary view does not blank it.
- The estimate and the debug view POST segments without their words.
- The queue editor reloads on a generation change while the stream keeps
  ticking — throttled, not pushed back by every tick.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
JS = REPO / "whisper_tm" / "static" / "js"
QE_HARNESS = REPO / "tests" / "js" / "queue_editor_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

# Loads single functions out of a static file into a small sandbox.
PRELUDE = r"""
const fs = require('fs'), vm = require('vm');
function fns(file, names) {
    const src = fs.readFileSync(process.env.JS + '/' + file, 'utf8');
    return names.map(name => {
        for (const head of ['async function ' + name + '(', 'function ' + name + '(']) {
            const at = src.indexOf(head);
            if (at >= 0) return src.slice(at, src.indexOf('\n}\n', at) + 3);
        }
        throw new Error('missing ' + name);
    }).join('\n');
}
const timers = [];
const toasts = [];
const out = o => console.log('§§' + JSON.stringify(o));
const flush = async () => { for (let i = 0; i < 10; i++) await new Promise(r => setImmediate(r)); };
const runTimer = async () => {
    const t = timers.find(x => x.fn);
    if (!t) return false;
    const f = t.fn; t.fn = null; await f(); await flush();
    return true;
};
const base = {
    console, JSON, Promise, Math, Object, Number, String, Error,
    setTimeout: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
    clearTimeout: id => { if (id && timers[id - 1]) timers[id - 1].fn = null; },
    Toast: {success: t => toasts.push('OK ' + t), error: t => toasts.push('ERR ' + t),
            info: t => toasts.push('INFO ' + t)},
    esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                             .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
    escJs: s => String(s ?? ''),
    // core.js's counted noun: "1 window", "3 windows"
    plural: (n, one, many) => {
        const k = Number(n) || 0;
        return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
    },
};
"""


def run(body: str, env_js: Path = JS) -> dict:
    done = subprocess.run(
        [
            "node",
            "-e",
            PRELUDE + f"(async () => {{ {body} }})()"
            ".catch(e => { console.error(e); process.exit(1); });",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={"JS": str(env_js), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, done.stdout + done.stderr
    return json.loads(lines[-1][2:])


# ── llmQueueTrack ────────────────────────────────────────────────────────

TRACK = r"""
let answers = [];
const ctx = {...base, fetch: async () => {
    const a = answers.shift();
    if (a === 'down') throw new TypeError('Failed to fetch');
    const [status, body] = a;
    return {ok: status < 400, status,
            json: async () => { if (body === 'html') throw new SyntaxError('Unexpected token <'); return body; }};
}};
vm.createContext(ctx);
vm.runInContext('let _lqLastStatus = null;\n' + fns('llmtools.js', ['llmQueueTrack']), ctx);
const seen = {ticks: [], done: 0, errors: []};
const track = () => ctx.llmQueueTrack(5, e => seen.ticks.push(e.status), () => seen.done++,
                                       m => seen.errors.push(m));
"""


def test_a_failed_poll_is_retried_not_reported_as_a_failure():
    d = run(
        TRACK
        + r"""
        answers = ['down', [502, 'html'], [500, {error: 'boom'}],
                   [200, {status: 'running'}], [200, {status: 'done', result: 'x'}]];
        track(); await flush();
        const delays = [];
        while (true) {
            const t = timers.find(x => x.fn);
            if (!t) break;
            delays.push(t.ms);
            await runTimer();
        }
        out({seen, delays});
    """
    )
    assert d["seen"]["errors"] == []
    assert d["seen"]["ticks"] == ["running", "done"] and d["seen"]["done"] == 1
    # backoff grows on consecutive failures, resets once an answer arrives
    assert d["delays"] == [1800, 3600, 7200, 700]


def test_a_failed_task_and_a_gone_entry_end_the_tracking():
    d = run(
        TRACK
        + r"""
        answers = [[200, {status: 'error', error: 'LLM unreachable'}]];
        track(); await flush();
        const failed = {...seen, timers: timers.filter(t => t.fn).length};
        seen.errors = []; seen.ticks = [];
        answers = [[404, {error: 'Entry not found'}]];
        track(); await flush();
        out({failed, gone: {errors: seen.errors, ticks: seen.ticks},
             timers: timers.filter(t => t.fn).length});
    """
    )
    assert d["failed"]["errors"] == ["LLM unreachable"] and d["failed"]["timers"] == 0
    assert d["gone"] == {"errors": ["Entry not found"], "ticks": []}
    assert d["timers"] == 0


def test_stop_during_a_request_in_flight_keeps_its_answer_away():
    d = run(
        TRACK
        + r"""
        let release;
        ctx.fetch = () => new Promise(r => { release = () => r({ok: true, status: 200,
                                                   json: async () => ({status: 'done'})}); });
        const t = track();
        await flush();
        t.stop();
        release(); await flush();
        out({seen, timers: timers.filter(x => x.fn).length});
    """
    )
    assert d["seen"] == {"ticks": [], "done": 0, "errors": []} and d["timers"] == 0


# ── section boxes ────────────────────────────────────────────────────────


def test_a_section_opens_from_its_own_header():
    d = run(r"""
        const ctx = {...base};
        vm.createContext(ctx);
        const st = fs.readFileSync(process.env.JS + '/state.js', 'utf8');
        const at = st.indexOf('function escJs(');
        vm.runInContext(st.slice(at, st.indexOf('\n}\n', at) + 3) + '\n'
            + 'const _partsState = new Map();\n'
            + fns('llmtools.js', ['_partsOf', 'llmPartsPanelToggle', 'llmPartToggle',
                                  'llmResetParts', 'llmPartsHtml']), ctx);
        const html = ctx.llmPartsHtml([{text: 'a'}, {text: 'b'}], 'q7');
        const box = (cls) => { const c = new Set(); return {cls, classList: {
            toggle: (k, on) => on ? c.add(k) : c.delete(k), has: k => c.has(k)}}; };
        const part = box('llm-part'), panel = box('llm-parts');
        const head = {closest: sel => sel === '.llm-part' ? part : sel === '.llm-parts' ? panel : null};
        ctx.llmPartToggle(1, head, 'q7');
        ctx.llmPartsPanelToggle(head, 'q7');
        const again = ctx.llmPartsHtml([{text: 'a'}, {text: 'b'}], 'q7');
        // another task's box (the state used to be global)
        const other = ctx.llmPartsHtml([{text: 'a'}, {text: 'b'}], 'q8');
        ctx.llmResetParts();                  // a dialog opening: others stay
        const kept = ctx.llmPartsHtml([{text: 'a'}, {text: 'b'}], 'q7');
        out({html, part: part.classList.has('open'), panel: panel.classList.has('open'),
             again, other, kept});
    """)
    assert 'id="llm' not in d["html"]  # no ids: several boxes can be shown at once
    assert "llmPartToggle(1, this, 'q7')" in d["html"]
    assert "llmPartsPanelToggle(this, 'q7')" in d["html"]
    assert d["part"] is True and d["panel"] is True
    # the state survives the next re-render (polling rebuilds the box)
    assert 'llm-part open" data-part="1"' in d["again"]
    assert 'class="llm-parts open"' in d["again"]
    # … but belongs to its task only
    assert " open" not in d["other"].replace("llmPart", "")
    assert d["kept"] == d["again"]


def test_no_german_section_id_left():
    src = (JS / "llmtools.js").read_text(encoding="utf-8")
    assert "llmTeil" not in src and "llmPartsBox" not in src
    assert "'transkript'" not in src


# ── summary view / translation switch ────────────────────────────────────

SUMVIEW = r"""
const el = {innerHTML: '', sets: 0};
let html = '';
Object.defineProperty(el, 'innerHTML', {get: () => html, set: v => { html = v; el.sets++; }});
let pendingFetch = null;
const ctx = {...base, activeFile: 'a.mp4',
    document: {getElementById: id => id === 'sumView' ? el : null},
    fetch: url => new Promise(r => { pendingFetch = () => r({ok: true, status: 200,
        json: async () => ({summaries: [{text: 'S of ' + decodeURIComponent(url.split('/').pop()),
                                         timestamp: 1}], pending: false})}); }),
    _txViewbarRender: () => {}, _txSumPoll: () => {},
};
vm.createContext(ctx);
vm.runInContext("let _txView = {sumCount: 0, sumPending: false};\nlet _sumViewSeq = 0;\n"
    + fns('llmtools.js', ['_sumViewRender']) + '\nthis.view = () => _txView;', ctx);
"""


def test_the_summary_view_drops_an_answer_for_a_file_no_longer_open():
    d = run(
        SUMVIEW
        + r"""
        const p = ctx._sumViewRender();
        await flush();
        const loading = html;
        ctx.activeFile = 'b.mp4';
        pendingFetch(); await p; await flush();
        out({loading, html, count: ctx.view().sumCount});
    """
    )
    assert "Loading" in d["loading"]
    assert d["html"] == d["loading"] and d["count"] == 0


def test_a_refresh_neither_blanks_the_view_nor_rebuilds_it_unchanged():
    d = run(
        SUMVIEW
        + r"""
        let p = ctx._sumViewRender(); await flush(); pendingFetch(); await p;
        const first = {sets: el.sets, html};
        p = ctx._sumViewRender(); await flush();
        const during = html;
        pendingFetch(); await p;
        out({first, during, sets: el.sets, same: html === first.html});
    """
    )
    assert "S of a.mp4" in d["first"]["html"] and d["first"]["sets"] == 2  # Loading…, content
    assert d["during"] == d["first"]["html"]  # no "Loading…" on a refresh
    assert d["sets"] == 2 and d["same"] is True  # unchanged → DOM left alone


def test_the_translation_switch_drops_a_track_for_a_file_no_longer_open():
    d = run(r"""
        let release;
        const ctx = {...base, activeFile: 'a.mp4',
            _fetchTranslationTrack: () => new Promise(r => { release = () => r([{id: 0, text: 'Hello'}]); }),
            _txViewbarRender: () => { ctx.bar = (ctx.bar || 0) + 1; }, _txApplyView: () => {}};
        vm.createContext(ctx);
        const src = fs.readFileSync(process.env.JS + '/llmtools.js', 'utf8');
        vm.runInContext("let _txView = {mode: 'original', lang: '', tracks: [{language: 'en'}], map: null};\n"
            + 'let _txSwitchSeq = 0;\n' + fns('llmtools.js', ['switchTxView'])
            + '\nthis.view = () => _txView;', ctx);
        const p = ctx.switchTxView('translation');
        await flush();
        ctx.activeFile = 'b.mp4';
        release(); await p;
        out({mode: ctx.view().mode, map: ctx.view().map, bar: ctx.bar || 0});
    """)
    assert d == {"mode": "original", "map": None, "bar": 0}


# ── summarize.js ─────────────────────────────────────────────────────────


def test_estimate_and_debug_post_segments_without_words():
    d = run(r"""
        const ctx = {...base, activeTranscript: {segments: [
            {start: 0, end: 2, start_fmt: '00:00', end_fmt: '00:02', speaker: 'A', text: 'hi',
             words: [{word: 'hi', start: 0, end: 1}]}]}};
        vm.createContext(ctx);
        vm.runInContext('let _sumSourceSegs = null;\n'
            + fns('summarize.js', ['_sumSegments', '_sumSpeakerName', '_sumPostSegments']), ctx);
        out(ctx._sumPostSegments());
    """)
    assert d == [
        {
            "start": 0,
            "end": 2,
            "start_fmt": "00:00",
            "end_fmt": "00:02",
            "speaker": "A",
            "text": "hi",
        }
    ]
    src = (JS / "summarize.js").read_text(encoding="utf-8")
    assert src.count("const segments = _sumPostSegments();") == 2


def test_translation_tracks_of_a_closed_dialog_stay_out_of_the_source_list():
    d = run(r"""
        const opts = [];
        const sel = {set innerHTML(v) { opts.length = 0; opts.push('orig'); }, appendChild: o => opts.push(o.value)};
        const releases = [];
        const ctx = {...base, activeTranscript: {language: 'de'},
            document: {getElementById: id => id === 'sumSource' ? sel : null,
                       createElement: () => ({})},
            fetch: url => new Promise(r => releases.push(() => r({json: async () => ({translations: [
                {language: url.includes('a.mp4') ? 'EN-A' : 'EN-B', segments: 1}]})})))};
        vm.createContext(ctx);
        vm.runInContext("let _sumSources = []; let _sumFilename = 'a.mp4'; let _sumSourcesSeq = 0;\n"
            + fns('summarize.js', ['_loadSumSources']) + "\nthis.setFile = f => { _sumFilename = f; };", ctx);
        const a = ctx._loadSumSources('a.mp4');
        ctx.setFile('b.mp4');
        const b = ctx._loadSumSources('b.mp4');
        releases[1](); await b;
        releases[0](); await a;
        out(opts);
    """)
    assert d == ["orig", "EN-B"]


# ── control buttons ──────────────────────────────────────────────────────


def test_a_failed_cancel_says_so_and_does_not_claim_it_was_requested():
    d = run(r"""
        let answer;
        const ctx = {...base, fetch: async () => answer(), _lqRender: () => {}};
        vm.createContext(ctx);
        vm.runInContext(fns('llmtools.js', ['_lqPost', 'llmQueueCancel', '_lqCancel', '_lqPause']), ctx);
        answer = () => { throw new TypeError('Failed to fetch'); };
        const down = await ctx.llmQueueCancel(3);
        answer = () => ({ok: false, status: 500, json: async () => { throw new SyntaxError('x'); }});
        await ctx._lqCancel(3);
        answer = () => ({ok: true, status: 200, json: async () => ({ok: false})});
        const over = await ctx.llmQueueCancel(3);
        answer = () => ({ok: true, status: 200, json: async () => ({ok: true})});
        await ctx._lqCancel(3);
        answer = () => ({ok: false, status: 502, json: async () => ({})});
        await ctx._lqPause(true);
        out({down, over, toasts});
    """)
    assert d["down"] is False and d["over"] is False
    assert d["toasts"] == [
        "ERR Cancel failed: Failed to fetch",
        "ERR Cancel failed: HTTP 500",
        "INFO Nothing to cancel — the task has already ended",
        "INFO Cancel requested — a running section still finishes",
        "ERR Pausing the LLM queue failed: HTTP 502",
    ]


# ── queue editor: generation reload ──────────────────────────────────────


def test_ticking_stream_does_not_push_the_reload_back_or_double_it():
    source = (
        f"const {{buildIt}} = require({str(QE_HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n"
        + r"""
        h.setAnswer('/api/queue/editor/items', () => ({ok: true, total: 1, offset: 0, limit: 200,
            items: [{id: 1, filename: 'a.mkv', dir: '/m', settings: {}, status: 'pending', position: 1, error: ''}],
            roots: [{dir: '/m', count: 1}], processing: null, generation: 7,
            filter: {status: ['pending'], dir: null, prefix: '', recursive: true, q: '', setting: null}}));
        h.run("openQueueEditor()"); await h.runTimers();
        const before = h.requests.length;
        // the stream ticks every second with a new generation; each tick
        // used to restart a 1 s timer, so the reload could slip and slip
        h.run("qeGenerationSeen(8)");
        const armed = h.timer.findIndex(t => t.fn);
        h.run("qeGenerationSeen(9)"); h.run("qeGenerationSeen(10)");
        const stillOne = h.timer.findIndex(t => t.fn);
        const open = h.timer.filter(t => t.fn).length;
        await h.runTimers();
        output({before, armed, stillOne, open, after: h.requests.length});
    """
        + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"QUEUE_EDITOR_JS": str(JS / "queue_editor.js"), "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    d = json.loads([z for z in done.stdout.splitlines() if z.startswith("§§")][-1][2:])
    # the same timer is still armed — not cleared and set anew per tick
    assert d["armed"] >= 0 and d["stillOne"] == d["armed"] and d["open"] == 1
    assert d["after"] == d["before"] + 2  # one items + one folders load


def test_a_failing_reload_backs_off_instead_of_toasting_every_tick():
    """A failed load brings no generation, so every tick (≈1 s) used to
    reload again — one error toast per second. Now it backs off."""
    source = (
        f"const {{buildIt}} = require({str(QE_HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n"
        + r"""
        h.run("openQueueEditor()"); await h.runTimers();
        h.setStatus('/api/queue/editor/items', 500, {error: 'db locked'});
        let now = 1000000;
        h.sandbox.Date = {now: () => now};
        const tick = async gen => { h.run(`qeGenerationSeen(${gen})`); await h.runTimers(); };
        await tick(8);                                   // fails → backoff 5 s
        const first = h.toasts.error;
        for (let i = 0; i < 4; i++) { now += 1000; await tick(9 + i); }   // within 5 s
        const during = h.toasts.error;
        now += 2000; await tick(20);                     // past 5 s: tried again
        const later = h.toasts.error;
        output({first, during, later});
    """
        + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"QUEUE_EDITOR_JS": str(JS / "queue_editor.js"), "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    d = json.loads([z for z in done.stdout.splitlines() if z.startswith("§§")][-1][2:])
    assert d["first"] == 1
    assert d["during"] == 1  # no toast per tick while backing off
    assert d["later"] == 2

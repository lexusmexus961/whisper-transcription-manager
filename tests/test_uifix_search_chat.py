"""Regressions from the UI review of Ask and Analyses.

* Ask: after a server restart the run is unknown (404) — the chat stayed
  "Answering …" for good, with Ask, New and the list blocked.
* Ask: every poll of a streaming answer re-rendered the whole thread.
* Analyses: polling stopped when another tab was shown and never resumed.
* Analyses: "Files of a search" was collected once and kept for any later
  search; an older collection could overwrite a newer one.
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
const js = n => fs.readFileSync(process.env.JS + '/' + n, 'utf8');
function fn(src, name) {
    const at = src.indexOf('function ' + name + '(');
    return src.slice(at, src.indexOf('\n}\n', at) + 3);
}
const fetched = [];
let answer = () => ({ok: true, status: 200, body: {}});
const ctx = {
    console, URL, Date, Promise, setTimeout, clearTimeout,
    location: {origin: 'http://127.0.0.1:5077'},
    document: {getElementById: () => null, querySelectorAll: () => []},
    svState: {tab: 'analysis', query: ''},
    Toast: {error: () => {}, success: () => {}, info: () => {}},
    fetch: async url => {
        fetched.push(String(url));
        const a = answer(String(url));
        return {ok: a.ok, status: a.status, json: async () => a.body};
    },
};
vm.createContext(ctx);
const state = js('state.js'), sv = js('searchview.js');
vm.runInContext([fn(state, 'esc'), fn(state, 'escJs'), fn(js('summarize.js'), '_renderMarkdown'),
                 fn(sv, 'svFmt'), fn(sv, 'svShortName'), js('chat.js'), js('analysis.js'),
                 'this.CHAT = CHAT; this.AN = AN;'].join('\n'), ctx);
const out = o => console.log('§§' + JSON.stringify(o));
const tick = () => new Promise(r => setTimeout(r, 5));
"""


def js(body: str) -> dict:
    source = (
        HARNESS
        + "(async () => {\n"
        + body
        + "\n})().catch(e => { console.error(e); process.exit(1); });"
    )
    done = subprocess.run(
        ["node", "-e", source],
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


def test_an_unknown_run_ends_the_answer():
    d = js(r"""
    answer = url => url.includes('/api/chat/runs/')
        ? {ok: false, status: 404, body: {error: 'unknown run'}}
        : {ok: true, status: 200, body: {conversations: []}};
    ctx.CHAT.conv = {id: 'c1', turns: [{role: 'user', text: 'hi'}], refs: {}};
    ctx.CHAT.run = {id: 'r1', events: new Map(), done: false, after: 0};
    ctx.svState.tab = 'chat';                       // the answer is in view
    await ctx.chatPoll();
    await tick();
    const runs = fetched.filter(u => u.includes('/api/chat/runs/')).length;
    await new Promise(r => setTimeout(r, 900));     // no further polls
    const last = ctx.CHAT.conv.turns[ctx.CHAT.conv.turns.length - 1];
    out({done: ctx.CHAT.run.done, runs,
         later: fetched.filter(u => u.includes('/api/chat/runs/')).length,
         role: last.role, error: last.error || '', live: !!last.live});
    """)
    assert d["done"] is True
    assert d["runs"] == d["later"] == 1
    assert d["role"] == "assistant" and "unknown run" in d["error"] and not d["live"]


def test_a_streaming_answer_redraws_only_the_live_turn():
    d = js(r"""
    // A thread element: one child per top-level message, as the browser has.
    const kids = [];
    const node = () => { const n = {querySelectorAll: () => [],
                                    remove() { kids.splice(kids.indexOf(n), 1); }}; return n; };
    const count = html => (String(html).match(/<div class="sv-msg /g) || []).length;
    const el = {
        get children() { return kids.slice(); },
        set innerHTML(h) { kids.length = 0; for (let i = 0; i < count(h); i++) kids.push(node()); },
        insertAdjacentHTML(_w, h) { for (let i = 0; i < count(h); i++) kids.push(node()); },
        querySelectorAll: () => [], scrollHeight: 0, scrollTop: 0, clientHeight: 0,
    };
    ctx.document.getElementById = id => (id === 'chatThread' ? el : null);
    ctx.chatRenderSources = () => {};
    const turns = [{role: 'user', text: 'one'}, {role: 'assistant', text: 'answer **one**'},
                   {role: 'user', text: 'two'}];
    ctx.CHAT.conv = {id: 'c1', turns, refs: {}};
    ctx.CHAT.run = {id: 'r1', done: false, after: 1,
                    events: new Map([[1, {n: 1, type: 'text', round: 0, text: 'par'}]])};
    const real = ctx.chatTurnHtml;
    let drawn = 0;
    ctx.chatTurnHtml = t => { drawn++; return real(t); };
    ctx.chatRender();
    const first = drawn;
    ctx.CHAT.run.events.set(2, {n: 2, type: 'text', round: 0, text: 'tial'});
    drawn = 0;
    ctx.chatRender();
    out({first, second: drawn, children: kids.length});
    """)
    assert d["first"] == 4  # three finished turns + the live one
    assert d["second"] == 1  # only the live one
    assert d["children"] == 4


def test_coming_back_to_a_running_analysis_polls_again():
    d = js(r"""
    let polled = 0;
    ctx.anLoadList = () => {};
    ctx.anShowView = () => {};
    ctx.anShowForm = () => {};
    ctx.anEl = () => ({options: [1], hidden: false});
    ctx.anPoll = () => { polled++; };
    ctx.AN.cur = {id: 'a1', status: 'running'};
    ctx.anOpen();
    ctx.AN.cur = {id: 'a1', status: 'done'};
    ctx.anOpen();
    out({polled});
    """)
    assert d["polled"] == 1


def test_the_search_source_follows_the_search():
    d = js(r"""
    let params = 'q=foo';
    ctx.svParams = () => params;
    ctx.svAbsPath = it => it.abs;
    ctx.anSourceSync = () => {};
    ctx.anPlanSoon = () => {};
    answer = url => ({ok: true, status: 200, body: {results: url.includes('q=foo')
        ? [{abs: '/m/foo.mp4'}] : [{abs: '/m/bar.mp4'}]}});
    ctx.AN.src = 'search';
    // Two collections in flight: the older one must not win.
    const older = ctx.anCollectSearch();
    params = 'q=bar';
    const newer = ctx.anCollectSearch();
    await Promise.all([older, newer]);
    const after = ctx.AN.search.abs_paths.slice();
    // The search moved on: choosing the source again collects anew.
    let collected = 0;
    const real = ctx.anCollectSearch;
    ctx.anCollectSearch = () => { collected++; };
    ctx.anSource('search');            // same search — kept
    params = 'q=baz';
    ctx.anSource('search');            // another search — collected again
    out({after, collected});
    """)
    assert d["after"] == ["/m/bar.mp4"]
    assert d["collected"] == 1


def test_visiting_the_tab_does_not_collect_a_changed_search():
    """anOpen used to collect up to 10 × 5000 hits on every visit after the
    search changed. Now it only marks it stale; the form, the
    source button and Start collect — Start never runs on the old one."""
    d = js(r"""
    let params = 'q=foo', collected = 0, started = 0;
    ctx.svParams = () => params;
    ctx.anCollectSearch = () => { collected++; };
    ctx.anLoadList = () => {};
    ctx.anShowView = () => {};
    ctx.anSourceSync = () => {};
    ctx.anRenderSources = () => {};
    ctx.anPlanSoon = () => {};
    ctx.anPoll = () => {};
    ctx.anEl = () => ({options: [1], hidden: false, value: 'x', checked: false, disabled: false});
    ctx.AN.src = 'search';
    ctx.AN.search = {label: '"foo"', abs_paths: ['/m/foo.mp4'], more: false, key: 'q=foo'};
    ctx.AN.cur = {id: 'a1', status: 'done'};
    params = 'q=bar';
    ctx.anOpen();                       // a result is shown: nothing collected
    const onVisit = collected;
    ctx.fetch = async () => { started++; return {ok: true, json: async () => ({})}; };
    await ctx.anStart();                // stale: collects instead of starting
    const onStart = collected;
    ctx.AN.cur = null;
    ctx.AN.searchKey = null;
    ctx.anShowForm();                   // the form shown: collected
    out({onVisit, onStart, onForm: collected, started});
    """)
    assert d["onVisit"] == 0
    assert d["onStart"] == 1
    assert d["onForm"] == 2
    assert d["started"] == 0


def test_an_answer_out_of_view_is_not_asked_for():
    """Ask used to poll every 400 ms whether or not anyone looked."""
    d = js(r"""
    answer = url => ({ok: true, status: 200, body: {events: [], done: false}});
    ctx.CHAT.conv = {id: 'c1', turns: [{role: 'user', text: 'hi'}], refs: {}};
    ctx.CHAT.run = {id: 'r1', events: new Map(), done: false, after: 0};
    ctx.svState.tab = 'find';
    await ctx.chatPoll();
    await tick();
    const hidden = fetched.filter(u => u.includes('/api/chat/runs/')).length;
    ctx.svState.tab = 'chat';
    ctx.chatPollResume();
    await tick();
    out({hidden, shown: fetched.filter(u => u.includes('/api/chat/runs/')).length});
    ctx.CHAT.run.done = true;                       // no polling on after the test
    """)
    assert d["hidden"] == 0 and d["shown"] == 1


def test_a_question_is_sent_once():
    d = js(r"""
    let release;
    answer = url => url.includes('/api/chat/send')
        ? new Promise(r => { release = () => r({ok: true, status: 200, body: {conversation: 'c1', run: 'r1'}}); })
        : {ok: true, status: 200, body: {conversations: [], events: [], done: true}};
    ctx.CHAT.conv = null; ctx.CHAT.run = null;
    const a = ctx.chatSend('hello'), b = ctx.chatSend('hello');
    await tick();
    release && release();
    await a; await b;
    out({sends: fetched.filter(u => u.includes('/api/chat/send')).length});
    """)
    assert d["sends"] == 1

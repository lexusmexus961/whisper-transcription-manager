"""The deep search tab — "find everything" for one situation.

A deep search runs for minutes in the LLM queue, on the server; the page
only asks how far it is (static/js/deepsearch.js). What it has to get right
on its own:

* the button next to the question starts it — once: the same question
  already running is shown instead of started twice;
* it asks ONE request at a time, from the results it has (`since`), and an
  answer for a search no longer shown is dropped;
* after failures it waits longer (2, 4, 8 … 30 s), and it asks nothing at
  all while nobody looks — search closed, another tab, browser tab hidden —
  and goes on when shown again;
* the plan, the steps (candidates → ranking n/N → checking n/N), the hits
  so far, the unclear ones apart; a hit plays and opens like any search hit,
  its copies too;
* Stop and delete (two clicks, no browser dialog), the kept ones to reopen;
* everything the server sends is escaped.

The harness (tests/js/deepsearch_harness.js) runs the real file with fake
timers and a scripted fetch, so time moves only when a test says so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "deepsearch_harness.js"
JS = REPO / "whisper_tm" / "static" / "js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

PRELUDE = r"""
const LIST = /\/api\/rag\/deep$/;
const doc = (over) => Object.assign({
    id: 'abc', query: 'bob is insulted', status: 'running', stage: 'checking',
    message: 'Checking 120 of 1,500 · 1 hit so far',
    plan: {situation: 'Bob is insulted', criteria_yes: 'Someone insults Bob to his face.',
           criteria_no: 'Bob is only mentioned.', must_terms: ['bob'],
           keywords: ['insult', 'idiot'], source: 'llm'},
    progress: {candidates: 17000, groups: 6000, ranked: 6000, rank_total: 6000,
               verified: 120, verify_total: 1500},
    counts: {hits: 1, maybe: 1}, results: [], next: 0, total_results: 0,
    created: 1700000000, started: 1700000002, finished: null, elapsed_sec: 85.2,
    reranked: true, note: null, error: null}, over);
const hit = (n, over) => Object.assign({
    n, verdict: 'hit', transcript_key: '/media/v/a' + n + '.mp4', rel_path: 'v/a' + n + '.mp4',
    file_dir: '/media', media_path: '/media/v/a' + n + '.mp4', start_sec: 120 + n, end_sec: 160,
    speaker: 'Bob', snippet: 'you idiot ' + n, score: 0.93, relevance: 0.93, copies: []}, over);
// A search with these answers, shown: the list, then the search itself
async function opened(answers, list) {
    H.answer(LIST, {searches: list || [{id: 'abc', query: 'bob is insulted', status: 'running',
                                        counts: {hits: 0, maybe: 0}, created: 1700000000}]});
    H.answer('/api/rag/deep/abc?', answers);
    H.show();
    await H.flush();
    H.ctx.dsLoad('abc');
    await H.flush();
}
"""


def js(body: str) -> dict:
    source = (
        f"const H = require({str(HARNESS)!r});\n"
        "const out = (o) => console.log('§§' + JSON.stringify(o));\n"
        + PRELUDE
        + f"(async () => {{\n{body}\n}})().catch(e => {{ console.error(e); process.exit(1); }});\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"JS_DIR": str(JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


# ── Starting ─────────────────────────────────────────────────────────────


def test_find_everything_starts_the_job_and_shows_it():
    d = js("""
    H.answer(LIST, {searches: []});
    H.answer('/api/rag/deep', {ok: true, id: 'abc', status: 'queued', entry_id: 7}, {method: 'POST'});
    H.answer('/api/rag/deep/abc?', doc({status: 'queued', stage: 'plan', plan: null, progress: {},
                                         counts: {hits: 0, maybe: 0}, elapsed_sec: null, started: null}));
    H.el('svOverlay').classList.add('visible');
    H.el('svQuery').value = '  bob is insulted ';
    await H.ctx.dsStartFromQuery();
    await H.flush();
    const post = H.requests.find(r => r.method === 'POST');
    out({post: post && post.url, body: post && post.body, tab: H.svState.tab,
         polls: H.polls('abc').map(r => r.url), view: H.html('dsView'),
         viewHidden: H.el('dsView').hidden, formHidden: H.el('dsForm').hidden,
         pending: H.pending(), toasts: H.toasts});
    """)
    assert d["post"] == "/api/rag/deep" and d["body"] == {"query": "bob is insulted"}
    assert d["tab"] == "deep", "the deep-search tab must come to the front"
    assert d["polls"] == ["/api/rag/deep/abc?since=0"]
    assert "“bob is insulted”" in d["view"]
    assert "waiting in the LLM queue" in d["view"]
    assert "Waiting in the LLM queue — it starts when the tasks before it are done" in d["view"]
    assert d["viewHidden"] is False and d["formHidden"] is True
    assert 4000 in d["pending"], "while it waits in the queue it asks every 4 s"
    assert any(k == "success" and "LLM queue" in m for k, m in d["toasts"])


def test_a_too_short_question_starts_nothing():
    d = js("""
    H.el('svOverlay').classList.add('visible');
    H.el('svQuery').value = 'ab';
    await H.ctx.dsStartFromQuery();
    await H.flush();
    out({requests: H.requests.length, tab: H.svState.tab, toasts: H.toasts,
         focused: H.el('svQuery').focused});
    """)
    assert d["requests"] == 0
    assert d["tab"] == "text", "nothing to show — the tab stays where it was"
    assert d["toasts"] and d["toasts"][0][0] == "info"
    assert d["focused"] is True


def test_the_same_question_already_running_is_shown_not_started_twice():
    d = js("""
    H.answer(LIST, {searches: [
        {id: 'run1', query: 'Bob  is insulted', status: 'running', counts: {hits: 3}},
        {id: 'old', query: 'bob is insulted', status: 'done', counts: {hits: 9}}]});
    H.answer('/api/rag/deep/run1?', doc({id: 'run1'}));
    H.show();
    await H.flush();
    H.el('svQuery').value = 'bob is insulted';
    await H.ctx.dsStartFromQuery();
    await H.flush();
    out({posts: H.requests.filter(r => r.method === 'POST').length,
         cur: H.ctx.DS.cur && H.ctx.DS.cur.id, toasts: H.toasts});
    """)
    assert d["posts"] == 0, "a second run of the same question was started"
    assert d["cur"] == "run1", "the running one is shown — not the finished one with that question"
    assert d["toasts"][-1][0] == "info"


def test_a_refused_start_says_why():
    d = js("""
    H.answer(LIST, {searches: []});
    H.answer('/api/rag/deep', {error: 'Empty search query'}, {method: 'POST', status: 400});
    H.show();
    await H.flush();
    H.el('dsQuery').value = 'someone apologizes for being late';
    await H.ctx.dsFormStart();
    await H.flush();
    out({toasts: H.toasts, cur: H.ctx.DS.cur, starting: H.ctx.DS.starting,
         disabled: H.el('dsStartBtn').disabled});
    """)
    assert ["error", "Deep search not started: Empty search query"] in d["toasts"]
    assert d["cur"] is None
    assert d["starting"] is False and d["disabled"] is False, "the button must work again"


# ── Asking how far it is ─────────────────────────────────────────────────


def test_polling_is_incremental_and_appends():
    d = js("""
    await opened((req) => req.url.endsWith('since=0')
        ? doc({results: [hit(0), hit(1, {verdict: 'maybe', snippet: 'maybe one'})], next: 2,
               counts: {hits: 1, maybe: 1}})
        : doc({results: [hit(2)], next: 3, counts: {hits: 2, maybe: 1}}));
    const first = H.html('dsHits');
    await H.tick(1500);
    out({polls: H.polls('abc').map(r => r.url), first, hits: H.html('dsHits'),
         maybe: H.html('dsMaybe'), since: H.ctx.DS.since,
         nHits: H.el('dsNHits').textContent, nMaybe: H.el('dsNMaybe').textContent,
         rows: (H.html('dsView').match(/class="ds-hit[ "]/g) || []).length});
    """)
    assert d["polls"] == ["/api/rag/deep/abc?since=0", "/api/rag/deep/abc?since=2"]
    assert "you idiot 0" in d["first"] and "you idiot 2" not in d["first"]
    assert "you idiot 0" in d["hits"] and "you idiot 2" in d["hits"], "the new hit was not appended"
    assert d["hits"].index("you idiot 0") < d["hits"].index("you idiot 2")
    assert "maybe one" in d["maybe"] and "maybe one" not in d["hits"]
    assert d["since"] == 3
    assert d["nHits"] == "2" and d["nMaybe"] == "1"
    assert d["rows"] == 3


def test_one_request_at_a_time():
    d = js("""
    await opened(doc({results: [hit(0)], next: 1}));
    H.answer('/api/rag/deep/abc?', doc({results: [], next: 1}), {hold: true});
    await H.tick(1500);                    // the next poll goes out — and hangs
    H.ctx.dsPoll(); H.ctx.dsPoll(); H.ctx.dsResume();
    await H.tick(5000);
    const during = H.polls('abc').length;
    H.release();
    await H.flush();
    out({during, after: H.polls('abc').length, pending: H.pending()});
    """)
    assert d["during"] == 2, "a second request went out while one was on its way"
    assert 1500 in d["pending"], "the answer schedules the next one"


def test_a_late_answer_for_another_search_is_dropped():
    d = js("""
    H.answer(LIST, {searches: []});
    H.answer('/api/rag/deep/abc?', doc({results: [hit(0, {snippet: 'from the old one'})], next: 1}),
             {hold: true});
    H.answer('/api/rag/deep/xyz?', doc({id: 'xyz', query: 'someone lies', results: [], next: 0}));
    H.show();
    await H.flush();
    H.ctx.dsLoad('abc');
    await H.flush();
    H.ctx.dsLoad('xyz');                   // another one chosen while abc's answer is out
    await H.flush();
    const aborted = H.polls('abc')[0].aborted;
    H.release();                           // abc's answer arrives late
    await H.flush();
    out({aborted, cur: H.ctx.DS.cur.id, query: H.ctx.DS.cur.query, view: H.html('dsView'),
         hits: H.ctx.DS.hits.length});
    """)
    assert d["aborted"] is True, "the abandoned request was not cancelled"
    assert d["cur"] == "xyz" and d["query"] == "someone lies"
    assert "from the old one" not in d["view"] and d["hits"] == 0


def test_a_late_answer_is_dropped_even_if_it_was_not_cancelled():
    """The sequence guard on its own — a server that answers anyway."""
    d = js("""
    H.answer(LIST, {searches: []});
    H.show();
    await H.flush();
    let late = null;
    H.answer('/api/rag/deep/abc?', doc({results: [hit(0, {snippet: 'from the old one'})], next: 1}));
    // The poll's own request is replaced by one that never hears the abort
    const realFetch = H.ctx.fetch;
    H.ctx.fetch = (url, opts) => url.includes('/abc?')
        ? new Promise(r => { late = () => r({ok: true, status: 200, json: () => Promise.resolve(
              doc({results: [hit(0, {snippet: 'from the old one'})], next: 1}))}); })
        : realFetch(url, opts);
    H.answer('/api/rag/deep/xyz?', doc({id: 'xyz', query: 'someone lies', next: 0}));
    H.ctx.dsLoad('abc');
    await H.flush();
    H.ctx.dsLoad('xyz');
    await H.flush();
    late();
    await H.flush();
    out({cur: H.ctx.DS.cur.id, hits: H.ctx.DS.hits.length, since: H.ctx.DS.since,
         view: H.html('dsView')});
    """)
    assert d["cur"] == "xyz" and d["hits"] == 0 and d["since"] == 0
    assert "from the old one" not in d["view"]


def test_failures_back_off_and_a_success_resets():
    d = js("""
    H.answer(LIST, {searches: []});
    H.answer('/api/rag/deep/abc?', null, {fail: 'network down'});
    H.show();
    await H.flush();
    H.ctx.dsLoad('abc');
    await H.flush();
    const waits = [H.pending()];
    const line = H.html('dsProgress');
    for (const ms of [2000, 4000, 8000, 16000, 30000]) {
        await H.tick(ms);
        waits.push(H.pending());
    }
    H.answer('/api/rag/deep/abc?', doc({results: [], next: 0}));
    await H.tick(30000);
    out({waits, line, after: H.pending(), fails: H.ctx.DS.fails, polls: H.polls('abc').length});
    """)
    assert d["waits"] == [[2000], [4000], [8000], [16000], [30000], [30000]], d["waits"]
    assert d["polls"] == 7
    assert "asking again in 2 s" in d["line"]
    assert d["fails"] == 0 and d["after"] == [1500]


def test_nothing_is_asked_while_the_browser_tab_is_hidden():
    d = js("""
    await opened(doc({results: [], next: 0}));
    H.answer('/api/rag/deep/abc?', doc({results: [], next: 0}), {hold: true});
    await H.tick(1500);                    // a poll on its way …
    const inflight = H.polls('abc')[1];
    H.hideTab();                           // … when the tab is hidden
    await H.flush();
    const aborted = inflight.aborted;      // at once — not by the 20-s guard later
    const pendingAtOnce = H.pending();
    const asked = H.requests.length;
    await H.tick(120000);
    const meanwhile = H.requests.length - asked;
    const pending = H.pending();
    H.answer('/api/rag/deep/abc?', doc({results: [], next: 0}));
    H.showTab();
    await H.flush();
    out({aborted, pendingAtOnce, meanwhile, pending,
         resumed: H.polls('abc').slice(-1)[0].url, polls: H.polls('abc').length});
    """)
    assert d["aborted"] is True, "the request on its way was left running"
    assert d["pendingAtOnce"] == [], "hidden, it must not even keep a timer"
    assert d["meanwhile"] == 0 and d["pending"] == [], "a hidden tab kept asking"
    assert d["polls"] == 3 and d["resumed"].endswith("since=0"), "shown again it must ask at once"


def test_the_browser_tab_shown_again_does_not_wake_it_on_another_search_tab():
    """Back in the browser while the Hits tab is shown: the deep search
    stays quiet — it only asks while its own tab is in view."""
    d = js("""
    await opened(doc({results: [], next: 0}));
    H.ctx.svTab('text');
    H.hideTab();
    const asked = H.requests.length;
    H.showTab();
    await H.tick(60000);
    out({asked: H.requests.length - asked, pending: H.pending()});
    """)
    assert d["asked"] == 0 and d["pending"] == []


def test_the_browser_tab_shown_again_does_not_wake_it_while_the_search_is_closed():
    """Closed on the deep-search tab, the tab stays 'deep' — coming back to
    the browser must still not start asking."""
    d = js("""
    await opened(doc({results: [], next: 0}));
    H.el('svOverlay').classList.remove('visible');
    H.ctx.dsLeave();                       // what closeSearchView does
    H.hideTab();
    const asked = H.requests.length;
    H.showTab();
    await H.tick(60000);
    out({tab: H.svState.tab, asked: H.requests.length - asked, pending: H.pending()});
    """)
    assert d["tab"] == "deep"
    assert d["asked"] == 0 and d["pending"] == []


def test_leaving_the_tab_stops_and_coming_back_resumes_from_the_cursor():
    d = js("""
    await opened((req) => doc({results: req.url.endsWith('since=0') ? [hit(0)] : [], next: 1}));
    H.svState.tab = 'deep';
    H.ctx.svTab('text');                   // the user looks at the hits
    const asked = H.requests.length;
    await H.tick(60000);
    const meanwhile = H.requests.length - asked;
    H.ctx.svTab('deep');
    await H.flush();
    const back = H.polls('abc').slice(-1)[0].url;
    // Closed: closeSearchView calls dsLeave
    H.el('svOverlay').classList.remove('visible');
    H.ctx.dsLeave();
    const asked2 = H.requests.length;
    await H.tick(60000);
    out({meanwhile, back, closed: H.requests.length - asked2, hits: H.ctx.DS.hits.length});
    """)
    assert d["meanwhile"] == 0, "another tab shown, it kept asking"
    assert d["back"] == "/api/rag/deep/abc?since=1", "coming back must go on from the cursor"
    assert d["closed"] == 0
    assert d["hits"] == 1


def test_a_finished_search_stops_asking_and_says_so():
    d = js("""
    await opened((req) => req.url.endsWith('since=0')
        ? doc({results: [hit(0)], next: 1})
        : doc({status: 'done', stage: 'done', results: [], next: 1, finished: 1700000087,
               elapsed_sec: 85.2, counts: {hits: 1, maybe: 0}}));
    await H.tick(1500);
    out({pending: H.pending(), toasts: H.toasts, view: H.html('dsView'),
         steps: H.html('dsProgress')});
    """)
    assert d["pending"] == [], "a finished search is still being asked about"
    assert ["success", "🔎 Deep search done: 1 hit"] in d["toasts"]
    assert "took 1 min 25 s" in d["view"]
    assert 'id="dsStopBtn"' not in d["view"] and "↻ Search again" in d["view"]
    assert d["steps"].count('class="done"') == 4


def test_reopening_a_finished_one_does_not_announce_it_again():
    d = js("""
    await opened(doc({status: 'done', stage: 'done', results: [hit(0)], next: 1}),
                 [{id: 'abc', query: 'bob is insulted', status: 'done', counts: {hits: 1}}]);
    out({toasts: H.toasts, pending: H.pending()});
    """)
    assert not [t for t in d["toasts"] if t[0] == "success"]
    assert d["pending"] == []


def test_a_rebuilt_search_is_read_again_from_the_start():
    d = js("""
    let round = 0;
    await opened((req) => {
        round++;
        if (round === 1) return doc({results: [hit(0), hit(1)], next: 2});
        if (round === 2) return doc({results: [], next: 1});      // fewer than we hold
        return doc({results: [hit(5)], next: 1});
    });
    await H.tick(1500);
    await H.tick(0);
    out({polls: H.polls('abc').map(r => r.url), hits: H.ctx.DS.hits.map(h => h.text),
         since: H.ctx.DS.since});
    """)
    assert d["polls"][-1].endswith("since=0"), d["polls"]
    assert d["hits"] == ["you idiot 5"] and d["since"] == 1


def test_a_search_deleted_elsewhere_ends_the_polling():
    d = js("""
    await opened(doc({results: [], next: 0}), []);
    H.answer('/api/rag/deep/abc?', {error: 'not found'}, {status: 404});
    await H.tick(1500);
    out({cur: H.ctx.DS.cur, pending: H.pending(), toasts: H.toasts,
         form: H.el('dsForm').hidden, view: H.el('dsView').hidden});
    """)
    assert d["cur"] is None and d["pending"] == []
    assert d["toasts"][-1] == ["info", "This deep search no longer exists"]
    assert d["form"] is False and d["view"] is True


# ── What it shows ────────────────────────────────────────────────────────


def test_the_plan_and_the_steps():
    d = js("""
    await opened(doc({results: [hit(0), hit(1, {verdict: 'maybe'})], next: 2}));
    out({plan: H.html('dsPlan'), steps: H.html('dsProgress')});
    """)
    plan, steps = d["plan"], d["steps"]
    for part in (
        "Situation",
        "Bob is insulted",
        "Counts",
        "Someone insults Bob to his face.",
        "Does not count",
        "Bob is only mentioned.",
        "Must contain",
        "Keywords",
    ):
        assert part in plan, part
    assert '<span class="ds-chip must">bob</span>' in plan
    assert '<span class="ds-chip">insult</span><span class="ds-chip">idiot</span>' in plan
    assert "generic" not in plan
    assert '<li class="done">Plan</li>' in steps
    assert (
        '<li class="done" title="17,000 passages — 6,000 once copies are folded together">Candidates <b>6,000</b></li>'
        in steps
    )
    assert '<li class="done">Ranking <b>6,000 / 6,000</b></li>' in steps
    assert '<li class="run">Checking <b>120 / 1,500</b></li>' in steps
    assert "width:8.0%" in steps, "the bar shows how far the checking is"
    assert "<b>1</b> hit so far · 1 unclear · 1 min 25 s running" in steps


def test_without_a_language_model_the_plan_says_it_is_generic():
    d = js("""
    await opened(doc({plan: {situation: 'bob is insulted', criteria_yes: '', criteria_no: '',
                             must_terms: [], keywords: ['bob', 'insult*'], source: 'fallback'},
                      reranked: false, note: 'Reranker busy — candidates in fused order'}));
    out({plan: H.html('dsPlan'), steps: H.html('dsProgress')});
    """)
    assert "· generic" in d["plan"]
    assert "Counts" not in d["plan"], "empty criteria are left out"
    assert "Reranker busy — candidates in fused order" in d["steps"]


def test_while_the_plan_is_written():
    # The server's record of a new search: every count 0 (rag/deep_search.py)
    d = js("""
    const zero = {candidates: 0, groups: 0, ranked: 0, rank_total: 0, verified: 0, verify_total: 0};
    let round = 0;
    await opened(() => ++round === 1
        ? doc({stage: 'plan', plan: null, progress: zero, counts: {hits: 0, maybe: 0, no: 0}})
        : doc({stage: 'done', status: 'done', plan: null, progress: zero,
               counts: {hits: 0, maybe: 0, no: 0}}));
    const during = {plan: H.html('dsPlan'), steps: H.html('dsProgress'), hits: H.html('dsHits')};
    await H.tick(1500);
    out({during, after: H.html('dsProgress'), hits: H.html('dsHits')});
    """)
    during = d["during"]
    assert "they appear here" in during["plan"]
    assert '<li class="run">Plan</li>' in during["steps"]
    assert "<li>Candidates</li>" in during["steps"], "a count of 0 before the step is noise"
    assert "<li>Ranking</li>" in during["steps"] and "<li>Checking</li>" in during["steps"]
    assert "Confirmed passages appear here once the checking starts." in during["hits"]
    assert "Candidates <b>0</b>" in d["after"], "after the step, 0 candidates is a finding"
    assert "Nothing was confirmed." in d["hits"]


def test_the_unclear_ones_are_apart_collapsed_and_hidden_while_none():
    d = js("""
    await opened((req) => req.url.endsWith('since=0')
        ? doc({results: [hit(0)], next: 1})
        : doc({results: [hit(1, {verdict: 'maybe', snippet: 'unclear one'})], next: 2}));
    const before = H.el('dsUnclear').classList.contains('empty');
    await H.tick(1500);
    const tag = H.html('dsView').match(/<details[^>]*>/)[0];
    out({before, after: H.el('dsUnclear').classList.contains('empty'), tag,
         maybe: H.html('dsMaybe')});
    """)
    assert d["before"] is True and d["after"] is False
    assert " open" not in d["tag"], "the unclear ones are collapsed"
    assert "unclear one" in d["maybe"]


def test_a_hit_plays_and_opens_like_a_search_hit():
    d = js("""
    await opened(doc({results: [hit(0, {copies: [
        {transcript_key: '/media/backup/a0.mp4', rel_path: 'backup/a0.mp4', file_dir: '/media',
         media_path: '/media/backup/a0.mp4', start_sec: 121.5},
        {transcript_key: '/gone/a0.mp4', rel_path: null, file_dir: null,
         media_path: '/gone/a0.mp4', start_sec: 120}]})], next: 1}));
    const row = H.html('dsHits');
    H.ctx.dsGoto('hit', 0);
    H.ctx.dsPlay('hit', 0);
    const hot = H.el('dsR-hit-0').classList.contains('hot');
    const folded = H.html('dsHits');
    H.ctx.dsCopies('hit', 0);
    const open = H.html('dsHits');
    H.ctx.dsGoto('hit', 0, 0);
    H.ctx.dsPlay('hit', 0, 0);
    H.ctx.dsPlay('hit', 0, 1);             // the copy no longer in the library
    H.ctx.dsGoto('hit', 0, 1);
    out({row, calls: H.calls.filter(c => c[0] !== 'svTab'), hot, folded, open});
    """)
    assert d["calls"] == [
        ["svGoto", "v/a0.mp4", 120, "/media"],
        ["svPlay", "v/a0.mp4", 120, "/media"],
        ["svGoto", "backup/a0.mp4", 121.5, "/media"],
        ["svPlay", "backup/a0.mp4", 121.5, "/media"],
    ]
    assert d["hot"] is True
    # A click plays (the search stays open), a double-click or ↗ opens
    assert "onclick=\"dsPlay('hit', 0)\" ondblclick=\"dsGoto('hit', 0)\"" in d["row"]
    assert 'class="ds-open"' in d["row"] and "↗</button>" in d["row"]
    assert "dsPlay('hit', 0)" in d["row"] and "2:00" in d["row"]
    assert "+2 copies" in d["folded"] and "backup/a0.mp4" not in d["folded"]
    assert "backup/a0.mp4" in d["open"] and "not in the library" in d["open"]
    assert "dsGoto('hit', 0, 1)" not in d["open"], "a copy that is gone must not be clickable"


def test_a_result_no_longer_in_the_library_cannot_be_opened():
    d = js("""
    await opened(doc({results: [hit(0, {rel_path: null, file_dir: null})], next: 1}));
    H.ctx.dsGoto('hit', 0);
    H.ctx.dsPlay('hit', 0);
    out({row: H.html('dsHits'), calls: H.calls.filter(c => c[0] !== 'svTab')});
    """)
    assert 'class="ds-hit dead' in d["row"]
    assert "dsGoto(" not in d["row"] and "dsPlay(" not in d["row"]
    assert "/media/v/a0.mp4" in d["row"], "it still says where it was"
    assert d["calls"] == []


def test_everything_the_server_sends_is_escaped():
    d = js("""
    const evil = '<img src=x onerror=alert(1)>';
    H.answer(LIST, {searches: [{id: "e'1", query: evil, status: 'running', counts: {hits: 1}}]});
    H.answer("/api/rag/deep/e'1?", doc({id: "e'1", query: evil, message: evil, note: evil,
        error: evil, plan: {situation: evil, criteria_yes: evil, criteria_no: evil,
                            must_terms: [evil], keywords: [evil]},
        results: [hit(0, {snippet: evil, speaker: evil, rel_path: "v/" + evil + ".mp4",
                          copies: [{rel_path: "b/" + evil, file_dir: '/media', start_sec: 1}]}),
                  hit(1, {verdict: 'maybe', snippet: evil})], next: 2}));
    H.show();
    await H.flush();
    H.ctx.dsLoad("e'1");
    await H.flush();
    H.ctx.dsCopies('hit', 0);
    const all = H.html('dsView') + H.html('dsPlan') + H.html('dsList');
    out({all, raw: all.includes('<img'), list: H.html('dsList')});
    """)
    assert d["raw"] is False, "markup from the server got into the page"
    assert "&lt;img src=x onerror=alert(1)&gt;" in d["all"]
    assert "dsLoad('e\\x271')" in d["list"], "the id is escaped for its onclick"


# ── Stop, delete, the list ───────────────────────────────────────────────


def test_stop_asks_the_server_and_shows_it_stopped():
    d = js("""
    await opened(doc({results: [hit(0)], next: 1}));
    H.answer('/api/rag/deep/abc/stop', {ok: true}, {method: 'POST'});
    H.answer('/api/rag/deep/abc?', doc({status: 'stopped', results: [], next: 1}));
    const had = H.html('dsView').includes('id="dsStopBtn"');
    await H.ctx.dsStop();
    await H.flush();
    out({had, stop: H.requests.filter(r => r.method === 'POST').map(r => r.url),
         view: H.html('dsView'), pending: H.pending(), toasts: H.toasts});
    """)
    assert d["had"] is True
    assert d["stop"] == ["/api/rag/deep/abc/stop"]
    assert 'id="dsStopBtn"' not in d["view"] and ">stopped<" in d["view"]
    assert "you idiot 0" in d["view"], "what was found so far is kept"
    assert d["pending"] == []
    assert ["info", "Stopping — what was found so far is kept"] in d["toasts"]


def test_stop_on_one_that_ended_meanwhile():
    d = js("""
    await opened(doc({results: [], next: 0}));
    H.answer('/api/rag/deep/abc/stop', {ok: false}, {method: 'POST'});
    H.answer('/api/rag/deep/abc?', doc({status: 'done', stage: 'done', results: [], next: 0}));
    await H.ctx.dsStop();
    await H.flush();
    out({toasts: H.toasts, view: H.html('dsView')});
    """)
    assert ["info", "It is not running any more"] in d["toasts"]
    assert ">done<" in d["view"]


def test_unchecked_passages_are_marked():
    """Without a language model the best-ranked ones come back unread —
    as "unclear", and each says it was not checked."""
    d = js("""
    await opened(doc({plan: {situation: 'bob is insulted', keywords: ['bob'], source: 'fallback'},
                      results: [hit(0, {verdict: 'maybe', unchecked: true})], next: 1,
                      counts: {hits: 0, maybe: 1}}));
    out({maybe: H.html('dsMaybe'), hits: H.html('dsHits')});
    """)
    assert "not checked" in d["maybe"] and "you idiot 0" in d["maybe"]
    assert "you idiot 0" not in d["hits"]


def test_delete_takes_two_clicks_and_no_browser_dialog():
    d = js("""
    await opened(doc({status: 'done', stage: 'done', results: [], next: 0}));
    H.answer('/api/rag/deep/abc', {ok: true}, {method: 'DELETE'});
    H.ctx.confirm = () => { throw new Error('window.confirm was used'); };
    const deletes = () => H.requests.filter(r => r.method === 'DELETE').length;
    H.ctx.dsDeleteClick(H.el('dsDelBtn'), 'abc');
    const armed = H.el('dsDelBtn').textContent;
    const first = deletes();
    await H.tick(4000);                    // too long: disarmed again
    const disarmed = H.el('dsDelBtn').textContent;
    H.ctx.dsDeleteClick(H.el('dsDelBtn'), 'abc');
    const second = deletes();
    H.answer(LIST, {searches: []});
    await H.ctx.dsDeleteClick(H.el('dsDelBtn'), 'abc');
    await H.flush();
    out({armed, first, disarmed, second, third: deletes(), cur: H.ctx.DS.cur,
         form: H.el('dsForm').hidden, list: H.html('dsList')});
    """)
    assert d["armed"] == "Delete?" and d["first"] == 0
    assert d["disarmed"] == "Delete"
    assert d["second"] == 0 and d["third"] == 1
    assert d["cur"] is None and d["form"] is False
    assert "No deep searches yet" in d["list"]


def test_the_list_reopens_and_counts_the_running_ones():
    d = js("""
    H.answer(LIST, {searches: [
        {id: 'a1', query: 'bob is insulted', status: 'running', counts: {hits: 3, maybe: 1}, created: 1700000000},
        {id: 'a2', query: 'someone lies', status: 'queued', counts: {hits: 0}, created: 1700000000},
        {id: 'a3', query: 'a door slams', status: 'done', counts: {hits: 1}, created: 1600000000}]});
    H.answer('/api/rag/deep/a3?', doc({id: 'a3', query: 'a door slams', status: 'done', stage: 'done'}));
    H.show();
    await H.flush();
    const list = H.html('dsList');
    H.ctx.dsLoad('a3');
    await H.flush();
    out({list, badge: H.el('svTabNDeep').textContent, after: H.html('dsList'),
         cur: H.ctx.DS.cur.id});
    """)
    assert d["list"].count('class="sv-chat-item an-item ds-item') == 3
    assert "3 hits · 1 unclear · running" in d["list"]
    assert "0 hits · waiting in the LLM queue" in d["list"]
    assert "onclick=\"dsLoad('a3')\"" in d["list"]
    assert d["badge"] == "2"
    assert d["cur"] == "a3"
    assert 'ds-item active"' in d["after"] and d["after"].index("ds-item active") > d[
        "after"
    ].index("a2")


def test_the_list_is_asked_again_only_while_another_one_runs():
    d = js("""
    H.answer(LIST, {searches: [{id: 'a1', query: 'q', status: 'running'}]});
    H.show();
    await H.flush();
    const with_other = H.pending();
    await H.tick(8000);
    const lists = H.lists().length;
    H.answer(LIST, {searches: [{id: 'a1', query: 'q', status: 'done'}]});
    await H.tick(8000);
    out({with_other, lists, after: H.pending(), total: H.lists().length});
    """)
    assert d["with_other"] == [8000]
    assert d["lists"] == 2
    assert d["after"] == [] and d["total"] == 3


def test_the_list_too_is_one_request_at_a_time():
    """Asked for again while its request is on its way (a start, a delete,
    a status change), it asks once more AFTER it — never in parallel."""
    d = js("""
    H.answer(LIST, {searches: []}, {hold: true});
    H.show();
    await H.flush();
    H.ctx.dsLoadList(); H.ctx.dsLoadList(); H.ctx.dsResume();
    await H.flush();
    const during = H.lists().length;
    H.answer(LIST, {searches: [{id: 'n1', query: 'new one', status: 'queued'}]});
    H.release();                           // the first answer: the old list
    await H.flush();
    out({during, after: H.lists().length, list: H.html('dsList')});
    """)
    assert d["during"] == 1, "a second list request went out while one was on its way"
    assert d["after"] == 2, "the asks in between must lead to exactly one more"
    assert "new one" in d["list"], "the list must end up with the newer answer"


def test_the_llm_queue_row_opens_it():
    d = js("""
    H.answer(LIST, {searches: []});
    H.answer('/api/rag/deep/abc?', doc({results: [hit(0)], next: 1}));
    H.ctx.openDeepSearch('abc');
    await H.flush();
    out({visible: H.el('svOverlay').classList.contains('visible'), tab: H.svState.tab,
         cur: H.ctx.DS.cur.id, polls: H.polls('abc').length, hits: H.html('dsHits')});
    """)
    assert d["visible"] is True and d["tab"] == "deep"
    assert d["cur"] == "abc" and d["polls"] == 1
    assert "you idiot 0" in d["hits"]


def test_a_result_shows_the_speakers_name():
    """rag/speakers.py: the deep search sends the label's name as
    speaker_name — the hit's head shows the name, the label only without."""
    d = js("""
    await opened(doc({results: [
        hit(0, {speaker: 'SPEAKER_04', speaker_name: 'Bob', snippet: '[Bob] what is this about'}),
        hit(1, {speaker: 'SPEAKER_01', speaker_name: null})], next: 2, total_results: 2}));
    out({view: H.html('dsView')});
    """)
    assert '<span class="ds-hit-sp">Bob</span>' in d["view"]
    assert '<span class="ds-hit-sp">SPEAKER_01</span>' in d["view"]
    assert "[Bob] what is this about" in d["view"]

"""Queue · Jobs (static/js/queue_page.js): transcriptions and AI jobs in one
list, by section, with the actions each kind has."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
SRC = (STATIC / "js" / "queue_page.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")

_RUN = r"""
const vm = require('vm');
const els = {qjBar: {innerHTML: ''}, qjList: {innerHTML: ''}};
const answers = {
    pending: {total: 70, items: [{id: 5, filename: 'a/wait.mp3', settings: {model: 'tiny', language: 'en'}}]},
    'error,cancelled': {total: 1, items: [{id: 6, filename: 'a/bad.mp4', status: 'error', error: 'No audio track'}]},
    done: {total: 1, items: [{id: 7, filename: 'a/old.mp3', finished_at: 100}]},
};
const posted = [];
const sb = {
    console, setTimeout, clearTimeout, Promise, JSON, Date,
    document: {getElementById: id => els[id] || null, addEventListener() {}},
    fetch: (url, o) => {
        if (o && o.method === 'POST') { posted.push([url, o.body]); return Promise.resolve({ok: true, json: () => Promise.resolve({})}); }
        const st = decodeURIComponent(/status=([^&]+)/.exec(url)[1]);
        return Promise.resolve({ok: true, json: () => Promise.resolve(answers[st])});
    },
    esc: s => String(s).replace(/</g, '&lt;'), escJs: s => String(s),
    queueRunningItems: [{item_id: 4, filename: 'a/run.mp3', progress: 33}],
    queuePaused: false, queueEta: {sec: 600, pending: 70, paused: false}, queueRetryable: 1,
    llmQueueSnap: {paused: true, counts: {}, items: [
        {id: 1, kind: 'summary', status: 'running', filename: 'b/sum.mp3', progress: 50},
        {id: 2, kind: 'translation', status: 'pending', filename: 'b/tr.mp3', params: {language: 'German'}},
        {id: 3, kind: 'analysis', status: 'error', error: 'model gone', params: {analysis_id: 'x1'}},
        {id: 8, kind: 'summary', status: 'done', filename: 'b/new.mp3', finished_at: 200}]},
    LQ_KIND: {summary: 'Summary', translation: 'Translation', analysis: 'Analysis'},
    _dsLeft: s => Math.round(s / 60) + ' min', _finishText: () => 'today 12:00',
    Shell: {go() {}},
};
sb.window = sb; sb.globalThis = sb;
vm.createContext(sb);
vm.runInContext(SRC + '\nglobalThis.QueuePage = QueuePage;', sb);
(async () => {
    const Q = sb.QueuePage, out = {};
    Q.shown(true);
    await new Promise(r => setTimeout(r, 20));
    out.all = els.qjList.innerHTML; out.bar = els.qjBar.innerHTML;
    Q.setKind('tx'); out.tx = els.qjList.innerHTML;
    Q.setKind('llm'); out.llm = els.qjList.innerHTML; out.llmBar = els.qjBar.innerHTML;
    await Q.toFront(5); await Q.move(5, 1); await Q.remove(5); await Q.cancel(4); await Q.retry(6);
    out.posted = posted;
    console.log('§§' + JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    done = subprocess.run(
        ["node", "-e", _RUN.replace("SRC", json.dumps(SRC))],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    return json.loads(next(z for z in done.stdout.splitlines() if z.startswith("§§"))[2:])


def test_four_sections_with_both_kinds(run):
    html = run["all"]
    for title in ("Running", "Waiting", "Failed", "Done"):
        assert f"<h4>{title} " in html, title
    for name in (
        "run.mp3",
        "sum.mp3",
        "wait.mp3",
        "tr.mp3",
        "bad.mp4",
        "model gone",
        "old.mp3",
        "new.mp3",
    ):
        assert name in html, name
    # the waiting total is the queue's, not the page shown
    assert "and 69 more" in html


def test_the_newest_done_first(run):
    html = run["all"]
    assert html.index("new.mp3") < html.index("old.mp3")


def test_chips_filter_the_kinds(run):
    assert "run.mp3" in run["tx"] and "sum.mp3" not in run["tx"]
    assert "sum.mp3" in run["llm"] and "run.mp3" not in run["llm"]
    # each queue paused and resumed by its own button
    assert "<b>Transcriptions</b>" in run["bar"] and "<b>AI jobs</b>" in run["bar"]
    assert "<b>Transcriptions</b>" not in run["llmBar"] and "▶ Resume" in run["llmBar"]
    assert "10 min left" in run["bar"]


def test_the_actions_each_kind_has(run):
    tx = run["tx"]
    for act in (
        "QueuePage.toFront(5)",
        "QueuePage.move(5, -1)",
        "QueuePage.move(5, 1)",
        "QueuePage.settings(5)",
        "QueuePage.remove(5)",
        "QueuePage.cancel(4)",
        "QueuePage.retry(6)",
        "retryFailedQueue()",
    ):
        assert act in tx, act
    llm = run["llm"]
    for act in (
        "QueuePage.llmCancel(1)",
        "QueuePage.llmCancel(2)",
        "QueuePage.llmRetry(3)",
        "openAnalysis('x1')",
    ):
        assert act in llm, act


def test_the_actions_use_the_queue_api_unchanged(run):
    assert run["posted"] == [
        ["/api/queue/move", '{"id":5,"top":true}'],
        ["/api/queue/move", '{"id":5,"direction":1}'],
        ["/api/queue/remove", '{"id":5}'],
        ["/api/queue/cancel", '{"item_id":4}'],
        ["/api/queue/retry", '{"ids":[6]}'],
    ]


def test_the_queue_place_opens_on_jobs():
    sel = (STATIC / "js" / "library_select.js").read_text(encoding="utf-8")
    assert "qeShowTab('jobs')" in sel[sel.index("function openQueueEditorFromBar") :]
    assert 'id="qeTabJobs"' in HTML and 'id="qePaneJobs"' in HTML
    # the queue lives in the rail: the sidebar keeps only the hidden counters
    # its badge reads (shell.js::badges), no chips of its own
    assert 'id="tabQueueBtn"' not in HTML and 'id="tabLlmQueueBtn"' not in HTML
    assert (
        '<span hidden><span id="tabQueue">0</span><span id="llmQueueBadge">0</span></span>' in HTML
    )

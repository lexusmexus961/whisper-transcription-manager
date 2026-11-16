"""The queue editor's "Add folder" tab in node (static/js/queue_add.js).

The tab queues a whole folder for transcription, at the front or at the
end, even while something runs — the sidebar only queues single files.

What these tests pin:

* the tab starts on the folder open in the file list and counts its files
  without a transcript — audio, estimate, and how many are queued already;
* "at the front" is the default; the button and the notes say where;
* queuing sends that folder, the transcription options of the tab and the
  place, with the count the user saw;
* an empty folder field asks for one, a server error shows as one.
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
const els = {};
const el = id => (els[id] ||= {id, value: '', innerHTML: '', textContent: '', disabled: false,
    style: {}, classList: {s: new Set(), add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
                           contains(c) { return this.s.has(c); }},
    on: {}, addEventListener(t, f) { (this.on[t] ||= []).push(f); },
    fire(t) { (this.on[t] || []).forEach(f => f()); }});
let where = 'front';
const fetched = [], toasts = [];
let selectReply = {count: 3, audio_sec: 4500, estimate_sec: 5400, sample: [
    {abs_path: '/nas/podcasts/Season_01/a.wav', name: 'a.wav', duration_sec: 4}], offset: 0, limit: 8};
let queuedReply = {count: 1, audio_sec: 4, sample: []};
let allReply = {count: 0, audio_sec: 0, sample: []};
let pageReply = body => ({count: 3, audio_sec: 4500, offset: body.offset, limit: body.limit, sample: []});
const ctx = {
    console, setTimeout, clearTimeout, Promise, JSON,
    document: {getElementById: el,
               querySelector: sel => sel.includes('qaWhere') ? {value: where} : null},
    fetch: async (url, opts) => {
        const body = JSON.parse(opts.body);
        fetched.push({url, body});
        if (url.endsWith('/select')) return {json: async () => body.criteria.in_queue ? queuedReply
                                                             : body.offset ? pageReply(body)
                                                             : body.criteria.status ? selectReply : allReply};
        return {json: async () => ({ok: true, job: 'transcribe', matched: 3, added: 2, moved_up: 1,
                                    front: body.front, paused: false, already_queued: 0,
                                    outside_allowed_roots: 0})};
    },
    txCollect: p => ({model: 'large-v3', language: 'de', vocal_isolation: false, diarize: false,
                      word_timestamps: true}),
    Toast: {success: t => toasts.push(['ok', t]), error: t => toasts.push(['err', t])},
    refreshQueue: () => {}, queueRunningTotal: () => 3, queuePending: 100000,
};
vm.createContext(ctx);
const state = js('state.js'), ls = js('library_select.js');
vm.runInContext([fn(state, 'esc'), fn(ls, '_lsHours'), fn(ls, '_lsEl'), fn(ls, 'qeReloadTab'),
                 js('queue_add.js'),
                 'this.qa = {qaInit, qaUseOpenFolder, qaCount, qaWhereChanged, qaAskConfirm, qaEnqueue, qaState,'
                 + ' qeReloadTab, qaLoadMore, qaFolderChosen};'
                ].join('\n'), ctx);
el('dirInput').value = '/nas/podcasts';
el('qaLang').selectedOptions = [{textContent: 'German'}];
const h = {els, el, fetched, toasts, setWhere: w => { where = w; },
           setSelect: r => { selectReply = r; }, setQueued: r => { queuedReply = r; },
           setAll: r => { allReply = r; }, setPage: f => { pageReply = f; }};
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


def _text(html: str) -> str:
    import re

    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


class TestTheCount:
    def test_it_starts_on_the_open_folder_at_the_front(self):
        d = js("""ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  out({folder: h.el('qaFolder').value, summary: h.el('qaSummary').innerHTML,
                       button: h.el('qaBtnQueue').textContent, list: h.el('qaList').innerHTML,
                       asked: h.fetched.map(f => f.body)});""")
        assert d["folder"] == "/nas/podcasts"
        assert d["asked"][0] == {
            "criteria": {"folders": ["/nas/podcasts"], "status": ["open"]},
            "limit": 100,
            "estimate": True,
        }
        assert d["asked"][1]["criteria"]["in_queue"] is True
        s = _text(d["summary"])
        assert "3 files without a transcript in podcasts and its subfolders" in s
        assert "1 h audio" in s and "about 2 h to transcribe" in s
        assert "1 of them is queued already — it moves to the front too" in s
        assert d["button"] == "Queue 3 files at the front"
        assert "Season_01" in d["list"] and "… and 2 more" in d["list"]

    def test_at_the_end_says_so_and_counts_only_the_new_ones(self):
        # At the end the one already waiting stays where it is — the click
        # adds two, and says two.
        d = js("""ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  h.setWhere('back'); ctx.qa.qaWhereChanged();
                  ctx.qa.qaAskConfirm();
                  out({summary: h.el('qaSummary').innerHTML, button: h.el('qaBtnQueue').textContent,
                       confirm: h.el('qaConfirmText').innerHTML});""")
        assert d["button"] == "Queue 2 files at the end"
        assert "it stays where it is" in _text(d["summary"])
        c = _text(d["confirm"])
        assert "Queue 2 files at the end to transcribe" in c and "audio" not in c
        assert "They wait behind the 100,000 entries already waiting." in c

    def test_all_queued_already_at_the_end_is_nothing_to_click(self):
        # Found in the browser: right after queuing a folder at the end the
        # button still offered to queue the files at the end — for 0 new ones.
        d = js("""h.setSelect({count: 2, audio_sec: 9, offset: 0, limit: 8, sample: []});
                  h.setQueued({count: 2, audio_sec: 9, sample: []});
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  h.setWhere('back'); ctx.qa.qaWhereChanged();
                  const end = {text: h.el('qaBtnQueue').textContent, off: h.el('qaBtnQueue').disabled};
                  h.setWhere('front'); ctx.qa.qaWhereChanged();
                  out({end, front: {text: h.el('qaBtnQueue').textContent, off: h.el('qaBtnQueue').disabled}});""")
        assert d["end"] == {"text": "All of them are in the queue already", "off": True}
        assert d["front"] == {"text": "Queue 2 files at the front", "off": False}

    def test_an_empty_queue_says_they_are_next(self):
        d = js("""ctx.queueRunningTotal = () => 0; ctx.queuePending = 0;
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  ctx.qa.qaAskConfirm(); const front = h.el('qaConfirmText').innerHTML;
                  ctx.queueRunningTotal = () => 1;
                  h.setWhere('back'); ctx.qa.qaWhereChanged(); ctx.qa.qaAskConfirm();
                  const end = h.el('qaConfirmText').innerHTML;
                  ctx.queueRunningTotal = () => 0; ctx.qa.qaAskConfirm();
                  out({front, end, empty: h.el('qaConfirmText').innerHTML});""")
        assert "These are next." in _text(d["front"])
        assert "They follow the 1 running file." in _text(d["end"])
        assert "The queue is empty — they are next." in _text(d["empty"])

    def test_one_file_is_it(self):
        """ "Queue 1 file … The queue is empty — they are next. [Yes, queue them]" """
        d = js("""ctx.queueRunningTotal = () => 0; ctx.queuePending = 0;
                  h.setSelect({count: 1, audio_sec: 9, offset: 0, limit: 8, sample: []});
                  h.setQueued({count: 0, audio_sec: 0, sample: []});
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  ctx.qa.qaAskConfirm();
                  out({text: h.el('qaConfirmText').innerHTML, button: h.el('qaBtnConfirm').textContent});""")
        assert "This one is next." in _text(d["text"])
        assert d["button"] == "Yes, queue it"

    def test_an_empty_folder_field_and_an_error(self):
        d = js("""h.el('qaFolder').value = ''; await ctx.qa.qaCount();
                  const empty = h.el('qaSummary').innerHTML;
                  h.el('qaFolder').value = 'relative/path';
                  h.setSelect({error: 'folders: folders must be absolute paths'});
                  await ctx.qa.qaCount();
                  out({empty, error: h.el('qaSummary').innerHTML, button: h.el('qaBtnQueue').disabled});""")
        assert d["empty"] == "Choose a folder."
        assert "must be absolute" in d["error"] and d["button"] is True

    def test_a_folder_indexed_just_now_has_no_lengths_yet(self):
        # Found in the browser: a freshly scanned folder read "0 min audio"
        # and every file "0 min" — as if the files were empty.
        d = js("""h.setSelect({count: 2, audio_sec: 0, estimate_sec: 4, offset: 0, limit: 8,
                              sample: [{abs_path: '/nas/podcasts/a.wav', name: 'a.wav', duration_sec: 0},
                                       {abs_path: '/nas/podcasts/b.wav', name: 'b.wav', duration_sec: null}]});
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  ctx.qa.qaAskConfirm();
                  out({summary: h.el('qaSummary').innerHTML, list: h.el('qaList').innerHTML,
                       confirm: h.el('qaConfirmText').innerHTML});""")
        assert "length not known yet" in _text(d["summary"]) and "0 min" not in d["summary"]
        assert d["list"].count("<td>—</td>") == 2
        assert "Queue 2 files at the front to transcribe" in _text(d["confirm"])

    def test_a_done_folder_is_not_an_unknown_one(self):
        # Found in the browser: a folder whose files were all transcribed
        # said "open the folder once to index it" — as if it were unknown.
        d = js("""h.setSelect({count: 0, audio_sec: 0, offset: 0, limit: 8, sample: []});
                  h.setQueued({count: 0, audio_sec: 0, sample: []});
                  h.setAll({count: 7, audio_sec: 0, sample: []});
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  const done = h.el('qaSummary').innerHTML;
                  h.setAll({count: 0, audio_sec: 0, sample: []});
                  await ctx.qa.qaCount();
                  out({done, unknown: h.el('qaSummary').innerHTML,
                       asked: h.fetched.map(f => f.body.criteria).filter(c => !c.status)});""")
        assert _text(d["done"]) == (
            "No file without a transcript in podcasts — all 7 are transcribed or skipped."
        )
        assert "open the folder in the file list once" in _text(d["unknown"])
        assert d["asked"] == [{"folders": ["/nas/podcasts"]}] * 2

    def test_a_folder_of_one_file_says_so(self):
        """ "all 1 are transcribed or skipped" """
        d = js("""h.setSelect({count: 0, audio_sec: 0, offset: 0, limit: 8, sample: []});
                  h.setQueued({count: 0, audio_sec: 0, sample: []});
                  h.setAll({count: 1, audio_sec: 0, sample: []});
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  out({done: h.el('qaSummary').innerHTML});""")
        assert _text(d["done"]) == (
            "No file without a transcript in podcasts — its one file is transcribed or skipped."
        )


class TestTheConfirmation:
    def test_a_change_after_asking_closes_it(self):
        # The confirmation shows the place and the options of the moment it
        # was asked; the click sends the ones of the moment it is clicked.
        d = js("""ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  const open = () => { ctx.qa.qaAskConfirm(); return h.el('qaConfirm').style.display; };
                  const asked = open();
                  h.setWhere('back'); ctx.qa.qaWhereChanged();
                  const afterWhere = h.el('qaConfirm').style.display;
                  open(); h.el('qaTxOptions').fire('change');
                  const afterOption = h.el('qaConfirm').style.display;
                  open(); h.el('qaTxOptions').fire('input');
                  out({asked, afterWhere, afterOption, afterTyping: h.el('qaConfirm').style.display});""")
        assert d == {
            "asked": "",
            "afterWhere": "none",
            "afterOption": "none",
            "afterTyping": "none",
        }


class TestQueuing:
    def test_it_sends_the_folder_the_options_and_the_place(self):
        d = js("""ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  ctx.qa.qaAskConfirm();
                  const confirm = h.el('qaConfirmText').innerHTML;
                  await ctx.qa.qaEnqueue();
                  out({confirm, sent: h.fetched.filter(f => f.url.endsWith('/enqueue')).map(f => f.body),
                       toasts: h.toasts});""")
        c = _text(d["confirm"])
        assert "Queue 3 files (1 h audio) at the front to transcribe" in c
        assert "large-v3, German, no vocal isolation, no speakers, word timestamps" in c
        assert (
            "The 3 running files finish first; then these are next, before the 100,000 waiting entries"
            in c
        )
        assert d["sent"] == [
            {
                "criteria": {"folders": ["/nas/podcasts"], "status": ["open"]},
                "job": "transcribe",
                "front": True,
                "expected_count": 3,
                "settings": {
                    "model": "large-v3",
                    "language": "de",
                    "vocal_isolation": False,
                    "diarize": False,
                    "word_timestamps": True,
                },
            }
        ]
        assert d["toasts"] == [
            ["ok", "2 files queued at the front · 1 already waiting moved to the front."]
        ]


class TestReload:
    def test_the_reload_button_reloads_the_open_tab(self):
        # Found in the browser: after the folder ran through, the tab still
        # offered to queue its files and ↻ only reloaded the hidden queue list.
        d = js("""const calls = [];
                  ctx.lsCount = () => calls.push('library'); ctx.qeLoad = () => calls.push('queue');
                  const show = pane => ['qePaneQueue', 'qePaneLibrary', 'qePaneAdd'].forEach(
                      p => { h.el(p).style.display = p === pane ? '' : 'none'; });
                  h.el('qaFolder').value = '/nas/podcasts';
                  show('qePaneAdd'); ctx.qa.qeReloadTab(); await new Promise(r => setTimeout(r, 20));
                  const selects = h.fetched.filter(f => f.url.endsWith('/select')).length;
                  show('qePaneLibrary'); ctx.qa.qeReloadTab();
                  show('qePaneQueue'); ctx.qa.qeReloadTab();
                  out({selects, calls});""")
        assert d["selects"] == 2
        assert d["calls"] == ["library", "queue"]


def _rows(n, start=0):
    return [
        {
            "abs_path": f"/nas/podcasts/Season_02/f{i:04}.wav",
            "name": f"f{i:04}.wav",
            "duration_sec": 3,
        }
        for i in range(start, start + n)
    ]


class TestMoreFiles:
    """The list shows more than a few files: 100 at first, then 200 at a
    time — by the button or by scrolling near the end of the list."""

    def test_the_list_grows_page_by_page_until_all_are_there(self):
        d = js(f"""h.setSelect({{count: 450, audio_sec: 1350, estimate_sec: 900, offset: 0, limit: 100,
                                sample: {json.dumps(_rows(100))}}});
                  const all = {json.dumps(_rows(450))};
                  h.setPage(b => ({{count: 450, audio_sec: 1350, sample: all.slice(b.offset, b.offset + b.limit)}}));
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  const first = h.el('qaList').innerHTML;
                  await ctx.qa.qaLoadMore();
                  const second = h.el('qaList').innerHTML;
                  await ctx.qa.qaLoadMore(); await ctx.qa.qaLoadMore();
                  out({{first: [first.split('<tr ').length - 1, first.includes('Show 200 more')],
                       second: [second.split('<tr ').length - 1, second.includes('… and 150 more'),
                                second.includes('Show 150 more')],
                       last: h.el('qaList').innerHTML.split('<tr ').length - 1,
                       more: h.el('qaList').innerHTML.includes('more'),
                       pages: h.fetched.filter(f => f.body.offset).map(f => [f.body.offset, f.body.limit,
                                                                             f.body.estimate])}});""")
        assert d["first"] == [100, True]
        assert d["second"] == [300, True, True]
        assert d["last"] == 450 and d["more"] is False
        # The third call found everything there and asked nothing
        assert d["pages"] == [[100, 200, False], [300, 200, False]]

    def test_a_folder_that_shrank_ends_the_list_and_keeps_the_count(self):
        # The count guards the queuing (expected_count) — it stays what the
        # user saw; only the list stops asking for more.
        d = js(f"""h.setSelect({{count: 150, audio_sec: 450, offset: 0, limit: 100, sample: {json.dumps(_rows(100))}}});
                  h.setPage(b => ({{count: 120, audio_sec: 360, sample: {json.dumps(_rows(20, 100))}}}));
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  await ctx.qa.qaLoadMore(); await ctx.qa.qaLoadMore();
                  out({{rows: h.el('qaList').innerHTML.split('<tr ').length - 1,
                       more: h.el('qaList').innerHTML.includes('more'),
                       count: ctx.qa.qaState.last.count,
                       pages: h.fetched.filter(f => f.body.offset).length}});""")
        assert d == {"rows": 120, "more": False, "count": 150, "pages": 1}

    def test_a_new_count_drops_a_page_still_on_its_way(self):
        d = js(f"""h.setSelect({{count: 300, audio_sec: 900, offset: 0, limit: 100, sample: {json.dumps(_rows(100))}}});
                  let release;
                  h.setPage(b => new Promise(r => {{ release = () => r({{count: 300, sample: {json.dumps(_rows(200, 100))}}}); }}));
                  ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  const pending = ctx.qa.qaLoadMore(); await new Promise(r => setTimeout(r, 5));
                  h.setSelect({{count: 1, audio_sec: 3, offset: 0, limit: 100, sample: {json.dumps(_rows(1, 900))}}});
                  await ctx.qa.qaCount();
                  release(); await pending;
                  out({{rows: h.el('qaList').innerHTML.split('<tr ').length - 1,
                       loading: ctx.qa.qaState.loadingMore}});""")
        assert d == {"rows": 1, "loading": False}

    def test_the_tab_counts_the_folder_the_picker_chose(self):
        d = js("""ctx.qa.qaInit(); await new Promise(r => setTimeout(r, 20));
                  h.el('qaFolder').value = '/nas/other';
                  ctx.qa.qaFolderChosen(); await new Promise(r => setTimeout(r, 20));
                  out({asked: h.fetched.map(f => f.body.criteria.folders[0])});""")
        assert d["asked"][-2:] == ["/nas/other", "/nas/other"]

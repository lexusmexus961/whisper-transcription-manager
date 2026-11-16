"""Queue lanes in the browser: several files run at once — no browser, via
node harnesses.

The stream (/api/queue/stream) sends every running entry as
``running_items``, oldest first; ``current_*`` stays the oldest one. What
used to follow that one file must now follow each of them: the ⟳ row marks,
one Queue-tab row per running file with its own ✕, the row reload of the
file that finished (which is not necessarily the oldest), the overlay that
shows the OPEN file's progress, "in queue" counts and the stop label. A
server without lanes sends no list — then its one current file is the list.
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
JS = REPO / "whisper_tm" / "static" / "js"
LANES = REPO / "tests" / "js" / "lanes_harness.js"
EDITOR = REPO / "tests" / "js" / "queue_editor_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

# Three running entries: A is the oldest, B has a phase, C's lane has not
# made its job yet (job_id '').
ENTRIES = r"""
const item = (id, job, name, abs, more = {}) => ({item_id: id, job_id: job, filename: name,
    abs, dir: abs.slice(0, 2), started_at: 100 + id, ...more});
const A = item(1, 'j1', 'a/one.mkv', '/m/a/one.mkv', {progress: 40, message: 'Transcribing…'});
const B = item(2, 'j2', 'two.mp3', '/n/two.mp3', {progress: 75, message: 'Finding speakers…',
    phase: {index: 5, total: 8, title: 'Diarization', start: 1000}});
const C = item(3, '', 'three.wav', '/m/three.wav');
const tickOf = (running, more = {}) => ({running: true, paused: false, abort: false,
    current_file: running.length ? running[0].filename : null,
    current_abs: running.length ? running[0].abs : null,
    current_dir: running.length ? running[0].dir : null,
    current_job_id: running.length ? (running[0].job_id || null) : null,
    running_items: running, running_count: running.length, lanes: 3,
    total: 10, processed: 4, pending: 3, done: 4, failed: 0, skipped: 0,
    cancelled: 0, retryable: 0, generation: 1,
    job_progress: running.length ? running[0].progress : undefined, ...more});
const load = (entries) => h.run('srvList.entries = ' + JSON.stringify(entries)
    + '; srvList.total = srvList.entries.length');
const marks = () => h.value('srvList.entries.map(f => !!f._processing)');
const runRows = () => h.value(`document.getElementById('fileList').children
    .filter(c => c.getAttribute('data-run') != null)
    .map(c => ({key: c.getAttribute('data-run'), html: c.innerHTML}))`);
"""


def _node(harness: Path, env: dict, body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(harness)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        + ENTRIES
        + "(async () => {\n"
        + body
        + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={**env, "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


def lanes(body: str) -> dict:
    return _node(LANES, {"STATIC_JS": str(JS)}, body)


def editor(body: str) -> dict:
    return _node(EDITOR, {"QUEUE_EDITOR_JS": str(JS / "queue_editor.js")}, body)


class TestRunningMarks:
    def test_every_running_entry_counts_as_running_by_its_identity(self):
        r = lanes("""
            h.run('queueRunningItems = ' + JSON.stringify([A, B]));
            output([
                {name: 'one.mkv', abs_path: '/m/a/one.mkv'},     // other root open, same file
                {name: 'two.mp3', abs_path: '/n/two.mp3'},       // not the oldest — still running
                {name: 'two.mp3', abs_path: '/x/two.mp3'},       // same name, another file
                {name: 'a/one.mkv'},                             // no identity: by name
                {name: 'x.mkv', abs_path: '/m/x.mkv'},
            ].map(f => h.sandbox.isRunningFile(f)));
        """)
        assert r == [True, True, False, True, False]

    def test_every_running_file_gets_its_row_mark(self):
        r = lanes("""
            load([{name: 'a/one.mkv', abs_path: '/m/a/one.mkv'}, {name: 'two.mp3', abs_path: '/n/two.mp3'},
                  {name: 'x.mkv', abs_path: '/m/x.mkv'}]);
            h.tick(tickOf([A, B]));
            const both = marks();
            h.tick(tickOf([B], {processed: 5}));
            output({both, after: marks()});
        """)
        assert r["both"] == [True, True, False]
        assert r["after"] == [False, True, False]

    def test_a_tick_with_the_list_but_no_current_file_yet_still_marks_it(self):
        """The marks follow the running list: a tick that already lists X but
        names no current file must not clear X's mark — the next ticks see
        the same list and would never set it again."""
        r = lanes("""
            load([{name: 'x.mkv', abs_path: '/m/x.mkv'}]);
            const X = item(7, '', 'x.mkv', '/m/x.mkv');
            h.tick(tickOf([]));
            h.tick({...tickOf([X]), current_file: null, current_abs: null, current_job_id: null});
            const racy = marks();
            h.tick(tickOf([{...X, job_id: 'j7', progress: 5}]));
            h.tick(tickOf([{...X, job_id: 'j7', progress: 9}]));
            output({racy, later: marks()});
        """)
        assert r == {"racy": [True], "later": [True]}


class TestTheFinishedFile:
    def test_the_entry_that_left_is_reloaded_even_when_it_is_not_the_oldest(self):
        """B finishes while the older A keeps running: the old code reloaded
        the row of A (the previous 'current file') and left B stale."""
        r = lanes("""
            h.tick(tickOf([A, B]));
            h.tick(tickOf([A, C], {processed: 5}));
            output(h.calls);
        """)
        assert r["refreshOne"] == [["two.mp3", "/n/two.mp3"]] and r["scans"] == 0

    def test_more_finished_than_left_falls_back_to_a_full_scan(self):
        """A quickly-failing file ran through between two ticks: unknown
        which — the whole list is reloaded instead."""
        r = lanes("""
            h.tick(tickOf([A, B]));
            h.tick(tickOf([B], {processed: 6}));
            output(h.calls);
        """)
        assert r["scans"] == 1 and r["refreshOne"] == []

    def test_the_first_tick_reloads_nothing(self):
        r = lanes("""
            h.tick(tickOf([A, B], {processed: 9}));
            output(h.calls);
        """)
        assert r["refreshOne"] == [] and r["scans"] == 0

    def test_a_stop_reloads_the_rows_it_put_back_without_a_scan(self):
        r = lanes("""
            h.tick(tickOf([A, B]));
            h.tick(tickOf([], {paused: true}));
            output(h.calls);
        """)
        assert sorted(r["refreshOne"]) == [["a/one.mkv", "/m/a/one.mkv"], ["two.mp3", "/n/two.mp3"]]
        assert r["scans"] == 0


class TestTheOverlay:
    def test_it_follows_the_open_running_file_not_the_oldest(self):
        r = lanes("""
            h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3', status: 'pending'}");
            h.tick(tickOf([A, B]));
            output({streams: h.streams.map(s => s.url), watched: h.value('watchedJobId'),
                    current: h.value('currentJobId'), bar: h.el('progBar').style.width,
                    msg: h.el('progMsg').textContent,
                    visible: h.el('progressOverlay').classList.contains('visible')});
        """)
        assert "/api/status/j2" in r["streams"] and "/api/status/j1" not in r["streams"]
        assert r["watched"] == "j2" and r["current"] == "j1"
        assert (r["bar"], r["msg"], r["visible"]) == ("75%", "Finding speakers…", True)

    def test_without_an_open_running_file_it_is_the_oldest_until_one_is_opened(self):
        r = lanes("""
            h.tick(tickOf([A, B]));
            const first = h.value('watchedJobId');
            // selectFile() calls this right away — no tick in between
            h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3'}; syncWatchedJob()");
            const live = h.streams.filter(s => s.url.startsWith('/api/status/'));
            output({first, now: h.value('watchedJobId'), urls: live.map(s => s.url),
                    closed: live.map(s => s.closed), bar: h.el('progBar').style.width});
        """)
        assert r["first"] == "j1" and r["now"] == "j2"
        assert r["urls"] == ["/api/status/j1", "/api/status/j2"] and r["closed"] == [True, False]
        assert r["bar"] == "75%", "the switch shows the file's last known progress at once"

    def test_its_cancel_stops_the_watched_job(self):
        r = lanes("""
            h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3'}");
            h.tick(tickOf([A, B]));
            h.run('cancelCurrentJob()'); await h.idle();
            output(h.fetched.filter(f => f.method === 'POST').map(f => f.url));
        """)
        assert r == ["/api/cancel/j2"]

    def test_the_menu_item_names_the_file_it_cancels(self):
        r = lanes("""
            h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3'}");
            h.tick(tickOf([A, B]));
            const two = h.el('btnCancelInline').textContent;
            h.tick(tickOf([A], {processed: 5}));
            output({two, one: h.el('btnCancelInline').textContent});
        """)
        assert r == {"two": "✕ Cancel two.mp3", "one": "✕ Cancel the current file"}


# The open file C runs, but its lane is still checking it on the NAS: no job
# yet. Before, the overlay kept the oldest file's progress and stream, and
# its ✕ (and the ⋯ "Cancel three.wav") cancelled THAT file (currentJobId).
OPEN_C = """
    h.tick(tickOf([A, C]));                      // nothing open: follows A
    h.run("activeFile = 'three.wav'; activeFileObj = {name: 'three.wav', abs_path: '/m/three.wav', status: 'pending'}");
    h.run("document.getElementById('progressOverlay').classList.add('visible'); syncWatchedJob()");
    h.tick(tickOf([A, C]));
"""


class TestAWatchedFileWithoutAJobYet:
    def test_the_overlay_says_starting_for_it_not_another_files_progress(self):
        r = lanes(
            OPEN_C
            + """
            output({msg: h.el('progMsg').textContent, bar: h.el('progBar').style.width,
                    watched: h.value('watchedJobId'), current: h.value('currentJobId'),
                    open: h.streams.filter(s => s.url.startsWith('/api/status/') && !s.closed).map(s => s.url)});
        """
        )
        assert (r["msg"], r["bar"], r["open"]) == ("Starting…", "0%", [])
        assert r["watched"] is None and r["current"] == "j1"

    def test_its_cancel_cancels_its_queue_entry_not_the_oldest_job(self):
        r = lanes(
            OPEN_C
            + """
            const menu = h.el('btnCancelInline').textContent;
            await h.run('cancelCurrentJob()');
            output({menu, posts: h.fetched.filter(f => f.method === 'POST')});
        """
        )
        assert r["menu"] == "✕ Cancel three.wav"
        assert r["posts"] == [
            {"url": "/api/queue/cancel", "method": "POST", "body": {"item_id": 3}}
        ]

    def test_once_its_job_exists_the_overlay_follows_it(self):
        r = lanes(
            OPEN_C
            + """
            h.tick(tickOf([A, {...C, job_id: 'j3', progress: 2, message: 'Checking file…'}]));
            output({watched: h.value('watchedJobId'), msg: h.el('progMsg').textContent,
                    open: h.streams.filter(s => s.url.startsWith('/api/status/') && !s.closed).map(s => s.url)});
        """
        )
        assert r == {"watched": "j3", "msg": "Checking file…", "open": ["/api/status/j3"]}

    def test_a_page_without_a_running_list_cancels_the_current_job(self):
        r = lanes("""
            h.run("currentJobId = 'jz'");
            await h.run('cancelCurrentJob()');
            output(h.fetched.filter(f => f.method === 'POST').map(f => f.url));
        """)
        assert r == ["/api/cancel/jz"]


# The open file X (two.mp3) runs beside the older A; the user cancels X and
# its stream reports the end — the overlay over X shows "❌ Cancelled" with
# "✕ Close". Before, the next tick re-armed it as "✕ Cancel" for A, and the
# click meant to close cancelled A.
X_ENDS = """
    h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3', status: 'pending'}");
    h.tick(tickOf([A, B]));
    h.streams.find(s => s.url === '/api/status/j2')
        .onmessage({data: JSON.stringify({status: 'error', error: 'Cancelled', message: 'Cancelled'})});
    h.tick(tickOf([A], {processed: 5}));
    const overlay = () => ({msg: h.el('progMsg').textContent, btn: h.el('btnCancelJob').textContent,
        closes: h.value("document.getElementById('btnCancelJob').onclick === _closeProgressOverlay"),
        disabled: h.el('btnCancelJob').disabled,
        visible: h.el('progressOverlay').classList.contains('visible')});
    const posts = () => h.fetched.filter(f => f.method === 'POST').map(f => f.url);
"""


class TestTheOverlayAfterTheOpenFileEnds:
    def test_its_outcome_stays_and_the_click_closes_it(self):
        r = lanes(
            X_ENDS
            + """
            const after = overlay();
            h.run("document.getElementById('btnCancelJob').onclick()"); await h.idle();
            output({after, posts: posts(), closed: overlay().visible});
        """
        )
        assert r["after"] == {
            "msg": "❌ Cancelled",
            "btn": "✕ Close",
            "closes": True,
            "disabled": False,
            "visible": True,
        }
        assert r["posts"] == [] and r["closed"] is False

    def test_once_closed_the_live_stream_follows_the_oldest_unseen(self):
        r = lanes(
            X_ENDS
            + """
            h.run("document.getElementById('btnCancelJob').onclick()");
            h.tick(tickOf([A], {processed: 5}));
            output({watched: h.value('watchedJobId'), visible: overlay().visible,
                    open: h.streams.filter(s => s.url.startsWith('/api/status/') && !s.closed).map(s => s.url)});
        """
        )
        assert r == {"watched": "j1", "visible": False, "open": ["/api/status/j1"]}

    def test_the_menu_cancel_leaves_that_outcome_alone(self):
        """The ⋯ item names and cancels the oldest — the overlay over X keeps
        its "✕ Close" instead of turning into a disabled "Stopping…"."""
        r = lanes(
            X_ENDS
            + """
            await h.run('cancelCurrentJob()');
            output({posts: posts(), after: overlay()});
        """
        )
        assert r["posts"] == ["/api/cancel/j1"]
        assert (r["after"]["btn"], r["after"]["disabled"], r["after"]["closes"]) == (
            "✕ Close",
            False,
            True,
        )

    def test_while_its_entry_unwinds_the_outcome_stays(self):
        """The job has ended but its lane still lists the entry (unwinding,
        summary): the same run — nothing to re-attach."""
        r = lanes("""
            h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3', status: 'pending'}");
            h.tick(tickOf([A, B]));
            h.streams.find(s => s.url === '/api/status/j2')
                .onmessage({data: JSON.stringify({status: 'error', error: 'Cancelled', message: 'Cancelled'})});
            h.tick(tickOf([A, B]));
            output({msg: h.el('progMsg').textContent, btn: h.el('btnCancelJob').textContent,
                    streams: h.streams.filter(s => s.url === '/api/status/j2').length});
        """)
        assert r == {"msg": "❌ Cancelled", "btn": "✕ Close", "streams": 1}

    def test_a_rerun_of_that_file_is_followed_again(self):
        """A retry gives the file the same job id; its new run (a new start)
        still replaces the old outcome and gets its own live stream."""
        r = lanes(
            X_ENDS
            + """
            h.tick(tickOf([A, {...B, started_at: 500, progress: 3, message: 'Again'}], {processed: 5}));
            output({open: h.streams.filter(s => s.url === '/api/status/j2' && !s.closed).length,
                    now: overlay()});
        """
        )
        assert r["open"] == 1
        assert (r["now"]["msg"], r["now"]["btn"], r["now"]["closes"]) == (
            "Again",
            "✕ Cancel",
            False,
        )

    def test_with_no_file_open_it_follows_the_queue_but_the_rearm_is_inert(self):
        """No file open: the overlay follows the queue. When A ends, the next
        tick shows B — but the button that just said "✕ Close" is inert for a
        moment and names the file it would cancel: a click meant to close A's
        outcome must not cancel B."""
        r = lanes("""
            const overlay = () => ({msg: h.el('progMsg').textContent, btn: h.el('btnCancelJob').textContent,
                disabled: h.el('btnCancelJob').disabled,
                visible: h.el('progressOverlay').classList.contains('visible')});
            const posts = () => h.fetched.filter(f => f.method === 'POST').map(f => f.url);
            const click = () => h.run("document.getElementById('btnCancelJob').onclick()");
            h.tick(tickOf([A, B]));
            h.streams.find(s => s.url === '/api/status/j1')
                .onmessage({data: JSON.stringify({status: 'done', result: null})});
            const outcome = overlay();
            h.tick(tickOf([B], {processed: 5}));                  // the next tick follows B
            const next = overlay();
            const live = h.streams.filter(s => !s.closed && s.url.startsWith('/api/status/')).map(s => s.url);
            const timers = h.pendingTimers();
            await click(); await h.idle();                        // inside the inert moment
            const sentInside = posts();
            await h.runTimers();                                  // 1.5 s later
            const later = overlay();
            await click(); await h.idle();
            output({outcome, next, live, timers, sentInside, later, sentAfter: posts()});
        """)
        assert (r["outcome"]["msg"], r["outcome"]["btn"]) == ("✅ Done!", "✕ Close")
        assert r["next"] == {
            "msg": "Finding speakers…",
            "btn": "✕ Cancel two.mp3",
            "disabled": True,
            "visible": True,
        }
        assert r["live"] == ["/api/status/j2"] and r["timers"] == [1500]
        assert r["sentInside"] == [], "a click in the inert moment must do nothing"
        assert (r["later"]["btn"], r["later"]["disabled"]) == ("✕ Cancel two.mp3", False)
        assert r["sentAfter"] == ["/api/cancel/j2"]

    def test_an_overlay_left_without_an_outcome_goes(self):
        """X left the running list before its stream said how it ended: there
        is no outcome to keep, and its ✕ must not cancel A."""
        r = lanes("""
            h.run("activeFile = 'two.mp3'; activeFileObj = {name: 'two.mp3', abs_path: '/n/two.mp3', status: 'pending'}");
            h.tick(tickOf([A, B]));
            const before = h.el('progressOverlay').classList.contains('visible');
            h.tick(tickOf([A], {processed: 5}));
            output({before, after: h.el('progressOverlay').classList.contains('visible'),
                    watched: h.value('watchedJobId')});
        """)
        assert r == {"before": True, "after": False, "watched": "j1"}


class TestTheQueueTab:
    def test_one_row_per_running_entry_each_with_its_own_cancel(self):
        r = lanes("""
            h.run("activeTab = 'queue'");
            h.tick(tickOf([A, B, C]));
            output(runRows());
        """)
        assert [row["key"] for row in r] == ["1", "2", "3"]
        assert "cancelRunningItem('2', this)" in r[1]["html"]
        assert "⟳ 75% · Diarization (5/8)" in r[1]["html"]
        assert "⟳ 40%" in r[0]["html"] and "⟳ Processing…" in r[2]["html"]

    def test_the_cancel_hits_that_rows_job_only(self):
        r = lanes("""
            h.tick(tickOf([A, B, C]));
            await h.run("cancelRunningItem('2')");
            await h.run("cancelRunningItem('3')");      // its lane has no job yet
            await h.run("cancelRunningItem('9')");      // finished meanwhile
            output({posts: h.fetched.filter(f => f.method === 'POST').map(f => f.url), toasts: h.toasts});
        """)
        # Without a job yet the queue entry itself is cancelled
        assert r["posts"] == ["/api/cancel/j2", "/api/queue/cancel"]
        assert [t[1] for t in r["toasts"]] == ["This file is no longer running"]

    def test_progress_moves_in_place_without_a_rebuild(self):
        """A rebuild every second would swallow a click on ✕ mid-hover."""
        r = lanes("""
            h.run("activeTab = 'queue'");
            h.tick(tickOf([A, B, C]));
            h.run("document.getElementById('fileList').children.forEach(c => c._mark = 1)");
            h.tick(tickOf([A, {...B, progress: 80}, C]));
            output(h.value(`document.getElementById('fileList').children
                .filter(c => c.getAttribute('data-run') === '2')
                .map(c => ({mark: c._mark, text: c._sRun.textContent}))`));
        """)
        assert r == [{"mark": 1, "text": "⟳ 80% · Diarization (5/8)"}]


class TestCountsAndLabels:
    def test_in_queue_is_pending_plus_every_running_entry(self):
        r = lanes("""
            h.tick(tickOf([A, B, C]));
            output(h.value('queueRunningTotal()'));
        """)
        assert r == 3

    def test_the_add_button_subtracts_only_the_waiting_entries(self):
        """The server's open count already leaves out every file with a
        running entry — subtracting the running ones again hid the button
        (3 open, 3 running elsewhere) or undercounted it."""
        r = lanes("""
            const add = () => ({shown: h.el('btnBatchAdd').style.display !== 'none',
                                text: h.el('btnBatchAdd').textContent});
            h.tick(tickOf([A, B, C]));
            h.run("activeTab = 'pending'; srvList.total = 10; renderBatchButtons()");
            const ten = add();
            h.tick(tickOf([A, B, C], {pending: 0}));
            h.run("srvList.total = 3; renderBatchButtons()");
            output({ten, three: add()});
        """)
        # 10 open in the filter − 3 waiting = 7 not yet queued
        assert r["ten"] == {"shown": True, "text": "＋ Queue 7 open files"}
        assert r["three"] == {"shown": True, "text": "＋ Queue 3 open files"}

    def test_the_stop_label_counts_the_files_still_winding_down(self):
        """A stop sets the abort flag: the running files are being cancelled
        and put back."""
        r = lanes("""
            const label = () => h.el('batchProgLabel').textContent;
            const stop = {paused: true, abort: true};
            h.tick(tickOf([A, B], stop)); const two = label();
            const menu = h.el('btnCancelInline').style.display;
            h.tick(tickOf([A], stop)); const one = label();
            h.tick(tickOf([], stop)); const none = label();
            output({two, one, none, menu});
        """)
        assert r == {"two": "Stopping 2…", "one": "Stopping…", "none": "Paused", "menu": "none"}

    def test_a_queue_that_paused_itself_lets_its_files_finish(self):
        """After a repeated VRAM failure the queue pauses itself — without the
        abort flag: the other lanes are not cancelled but run to the end, for
        hours maybe. That is no "Stopping…", and each stays cancellable."""
        r = lanes("""
            const label = () => h.el('batchProgLabel').textContent;
            const oom = {paused: true, abort: false};
            h.tick(tickOf([A, B], oom));
            const two = {label: label(), menu: h.el('btnCancelInline').style.display,
                         button: h.el('btnBatch').textContent};
            h.tick(tickOf([B], {...oom, processed: 5})); const one = label();
            h.tick(tickOf([], {...oom, processed: 6})); const none = label();
            output({two, one, none});
        """)
        assert r["two"] == {
            "label": "Paused — 2 finishing",
            "menu": "",
            "button": "▶ Resume (3 waiting)",
        }
        assert (r["one"], r["none"]) == ("Paused — 1 finishing", "Paused")

    def test_the_abort_question_names_every_running_file(self):
        r = lanes("""
            h.tick(tickOf([A, B]));
            await h.run('abortQueue()');
            output({asked: h.confirms, posts: h.fetched.filter(f => f.method === 'POST').length});
        """)
        assert "the 2 running files (a/one.mkv, two.mp3) are aborted" in r["asked"][0].lower()
        assert r["posts"] == 0, "declined — nothing was sent"

    def test_the_sidebar_chip_and_the_health_card_count_them_too(self):
        stats = (JS / "stats.js").read_text(encoding="utf-8")
        assert "setCount('tabQueue', queuePending + queueRunningTotal());" in stats
        health = (JS / "health.js").read_text(encoding="utf-8")
        assert "q.running_count > 1 ? ` +${q.running_count - 1} more`" in health


class TestAServerWithoutLanes:
    def test_its_current_file_is_the_list(self):
        r = lanes("""
            load([{name: 'a/one.mkv', abs_path: '/m/a/one.mkv'}, {name: 'two.mp3', abs_path: '/n/two.mp3'}]);
            const old = (cur, job, more = {}) => ({running: true, paused: false, current_file: cur.filename,
                current_abs: cur.abs, current_dir: cur.dir, current_job_id: job, job_progress: 33,
                job_message: 'Old server', total: 10, processed: 4, pending: 3, ...more});
            h.tick(old(A, 'j1'));
            const first = {marks: marks(), total: h.value('queueRunningTotal()'),
                           streams: h.streams.map(s => s.url), bar: h.el('progBar').style.width};
            h.tick(old(B, 'j2', {processed: 5}));
            output({first, marks: marks(), refreshed: h.calls.refreshOne});
        """)
        assert r["first"]["marks"] == [True, False] and r["first"]["total"] == 1
        assert "/api/status/j1" in r["first"]["streams"] and r["first"]["bar"] == "33%"
        assert r["marks"] == [False, True]
        assert r["refreshed"] == [["a/one.mkv", "/m/a/one.mkv"]]

    def test_an_empty_list_beside_a_current_file_is_that_file(self):
        """A core without lanes behind this API sends an empty list."""
        r = lanes("""
            h.tick({...tickOf([A]), running_items: [], running_count: 0});
            output({items: h.value('queueRunningItems.map(r => r.filename)'), total: h.value('queueRunningTotal()')});
        """)
        assert r == {"items": ["a/one.mkv"], "total": 1}


ROWS = """
    h.setAnswer('/api/queue/editor/items', {ok: true, total: 1, offset: 0, limit: 200,
        items: [{id: 1, filename: 'e1.mkv', dir: '/m', settings: {}, status: 'pending', position: 1, error: ''}],
        roots: [], generation: 7, filter: {status: ['pending']},
        processing: {id: 99, filename: 'one.mkv', dir: '/m', settings: {}, status: 'processing'},
        processing_items: [{id: 99, filename: 'one.mkv', dir: '/m', settings: {}, status: 'processing'},
                           {id: 98, filename: 'two.mkv', dir: '/m', settings: {}, status: 'processing'}]});
"""


class TestTheQueueEditor:
    def test_every_running_entry_is_listed_and_none_can_be_selected(self):
        r = editor(
            ROWS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(98, {}); qeRowClick(99, {})");
            output({n: h.value("qeSelectedCount()"), html: h.elements.qeList.innerHTML});
        """
        )
        assert r["n"] == 0
        assert 'data-id="99"' in r["html"] and 'data-id="98"' in r["html"]
        assert r["html"].count("running — not editable") == 2

    def test_the_chip_says_how_many_run(self):
        r = editor("""
            const chip = () => ({text: h.elements.qeChipRunning.textContent,
                                 shown: h.elements.qeChipRunning.style.display !== 'none'});
            h.run("qeRenderHead()"); const idle = chip();
            h.run("queueCurrentFile = 'one.mkv'; qeRenderHead()"); const one = chip();
            h.run("queueRunningTotal = () => 3; qeRenderHead()");
            output({idle, one, three: chip()});
        """)
        assert r["idle"]["shown"] is False
        assert r["one"] == {"text": "⟳ running", "shown": True}
        assert r["three"] == {"text": "⟳ 3 running", "shown": True}


def test_the_overlays_spinner_stays_round():
    """A narrow window squeezed the turning circle beside the progress message
    into an oval: it never shrinks."""
    css = (REPO / "whisper_tm" / "static" / "css" / "app.css").read_text(encoding="utf-8")
    rule = re.search(r"\.prog-info \.spin \{([^}]*)\}", css).group(1)
    assert "flex:none" in rule.replace(" ", "")

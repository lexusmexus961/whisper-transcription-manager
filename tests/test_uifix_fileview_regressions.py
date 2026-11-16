"""File view, upload and editor: regressions from the UI review.

Each test drives the real static/js file in node (tests/js/
uifix_fileview_harness.js, or the existing filelist/speaker harnesses) and
pins one defect that was found and fixed:

* the upload dialog's Go button stayed disabled after a link download or an
  empty click (upload.js doUpload);
* a "separate only" link download called a function that did not exist
  (upload.js, link download completion);
* a slow voice-match answer for file A overwrote file B's (speaker.js);
* "Apply names" never reached the names that "Save names" saves (speaker.js);
* a failing list page was asked for again in a tight loop (filelist.js);
* the playing segment is found by binary search (fileview.js);
* a transcript-supplied start_fmt reached innerHTML unescaped (editor.js);
* a segment edit saved after a file switch overwrote the new file's segment
  (editor.js);
* a late log lookup replaced the log of the file opened since (transcribe.js).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "uifix_fileview_harness.js"
FILELIST_HARNESS = REPO / "tests" / "js" / "filelist_harness.js"
STATIC = REPO / "whisper_tm" / "static" / "js"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str, harness: Path = HARNESS, env: dict | None = None) -> dict:
    source = (
        f"const {{buildIt}} = require({str(harness)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "const sleep = ms => new Promise(r => setTimeout(r, ms));\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", ""), **(env or {})},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@node
class TestUploadButton:
    def test_an_empty_upload_click_unlocks_the_button(self):
        r = js("""
            const h = buildIt({files: ['upload.js']});
            await h.run("_upSource = 'file'; doUpload()");
            output({disabled: h.byId('btnUploadGo').disabled,
                    warned: h.toasts.map(t => t.msg)});
        """)
        assert r["disabled"] is False
        assert r["warned"] == ["Please select a file"]

    def test_a_link_click_without_url_unlocks_the_button(self):
        r = js("""
            const h = buildIt({files: ['upload.js']});
            await h.run("_upSource = 'link'; doUpload()");
            output({disabled: h.byId('btnUploadGo').disabled});
        """)
        assert r["disabled"] is False

    def test_a_started_link_download_unlocks_the_button_and_separate_only_enqueues(self):
        """Model "none" + vocal isolation: the finished download is handed to
        the queue (which separates), instead of calling a missing function."""
        r = js("""
            const h = buildIt({files: ['upload.js'], fetch: async url =>
                url.includes('/api/youtube/download') ? {jobId: 'j1'} : {}});
            const calls = [];
            h.run("activeTab = 'all'; reloadFileList = () => {}; setFileTab = () => {};"
                  + "txCollect = () => ({model: 'none', vocal_isolation: true});");
            h.sandbox.enqueueFiles = async (files, settings, x, opts) => {
                calls.push({files, model: settings.model, vi: settings.vocal_isolation,
                            front: opts && opts.front});
                return files.length;
            };
            h.byId('ytUrl').value = 'https://example.org/v';
            await h.run("_upSource = 'link'; doUpload()");
            const disabled = h.byId('btnUploadGo').disabled;
            h.sources[0].onmessage({data: JSON.stringify({status: 'done', result:
                {files: ['uploads/a.mp4'], ok: 1, total: 1, in_library: true}})});
            await h.settle(); await h.settle();
            output({disabled, calls, errors: h.toasts.filter(t => t.type === 'error').map(t => t.msg)});
        """)
        assert r["disabled"] is False
        assert r["errors"] == []
        assert r["calls"] == [{"files": ["a.mp4"], "model": "none", "vi": True, "front": True}]

    def test_a_separate_only_playlist_is_not_pushed_to_the_front(self):
        """A playlist with "separate only" used to put every file
        at the FRONT. A single link may go first, a playlist goes to the end;
        transcription keeps its front placement, playlist included."""
        r = js("""
            const run = async (model, files) => {
                const h = buildIt({files: ['upload.js'], fetch: async url =>
                    url.includes('/api/youtube/download') ? {jobId: 'j1'} : {}});
                const calls = [];
                h.run("activeTab = 'all'; reloadFileList = () => {}; setFileTab = () => {};"
                      + `txCollect = () => ({model: '${model}', vocal_isolation: true});`);
                h.sandbox.enqueueFiles = async (fs, settings, x, opts) => {
                    calls.push(!!(opts && opts.front)); return fs.length; };
                h.byId('ytUrl').value = 'https://example.org/v';
                await h.run("_upSource = 'link'; doUpload()");
                h.sources[0].onmessage({data: JSON.stringify({status: 'done', result:
                    {files, ok: files.length, total: files.length, in_library: true}})});
                await h.settle(); await h.settle();
                return calls[0];
            };
            output({single: await run('none', ['uploads/a.mp4']),
                    playlist: await run('none', ['uploads/a.mp4', 'uploads/b.mp4']),
                    transcribe: await run('turbo', ['uploads/a.mp4', 'uploads/b.mp4'])});
        """)
        assert r == {"single": True, "playlist": False, "transcribe": True}

    def test_no_call_to_the_removed_separation_dialog(self):
        assert "_startUploadSeparation" not in (STATIC / "upload.js").read_text()


@node
class TestSpeakerMatch:
    def test_a_late_answer_for_the_previous_file_is_dropped(self):
        r = js("""
            const people = name => [{person: name, labels: ['SPEAKER_00'], windows: 1,
                of_windows: 1, best_value: 0.9, voices_share: 1, voices_total: 1,
                timestamps: [1]}];
            const h = buildIt({files: ['speaker.js'], fetch: async url => {
                const slow = url.includes('a.mkv');
                if (slow) await sleep(30);
                return {windows: 3, model: 'm', labels: [{label: 'SPEAKER_00'}],
                        named_people: 1, threshold: 0.75,
                        people: people(slow ? 'Alice' : 'Bob')};
            }});
            const first = h.run("loadSpeakerRecognition('a.mkv')");
            await h.run("loadSpeakerRecognition('b.mkv')");
            await first;
            output({hits: h.sandbox.window._speakerHits.map(p => p.person),
                    html: h.byId('speakerRecognition').innerHTML});
        """)
        assert r["hits"] == ["Bob"]
        assert "Bob" in r["html"] and "Alice" not in r["html"]

    def test_applied_names_go_through_the_rename_path(self):
        """Each field's name reaches speakerNames; the labels and the dirty
        Save are redrawn ONCE (per speaker it used to rewrite
        every segment label of a 24 h recording again and again)."""
        r = js("""
            const h = buildIt({files: ['speaker.js']});
            const inputs = ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'].map(id => {
                const input = h.element('in'); input.dataset.speakerId = id; return input; });
            h.sandbox.document.querySelectorAll = sel =>
                sel.includes('sp-stat-name') ? inputs : [];
            const renamed = [];
            let redrawn = 0;
            h.sandbox.speakerNameFromInput = inp => renamed.push(inp.value);
            h.sandbox.speakerNamesChanged = () => { redrawn++; };
            h.sandbox.window._speakerHits = [
                {person: 'Alice', labels: ['SPEAKER_00'], best_value: 0.9},
                {person: 'Bob', labels: ['SPEAKER_01'], best_value: 0.8},
                {person: 'Carol', labels: ['SPEAKER_02'], best_value: 0.7}];
            h.run('applySpeakerNames()');
            output({renamed, redrawn, value: inputs[0].value});
        """)
        assert r["renamed"] == ["Alice", "Bob", "Carol"]
        assert r["redrawn"] == 1
        assert r["value"] == "Alice"

    def test_rename_state_and_redraw_are_split(self):
        src = (STATIC / "fileview.js").read_text()
        i = src.index("function onSpeakerRename(input) {")
        body = src[i : src.index("\n}\n", i)]
        assert "speakerNameFromInput(input);" in body and "speakerNamesChanged();" in body
        j = src.index("function speakerNameFromInput(input) {")
        state = src[j : src.index("\n}\n", j)]
        assert "querySelectorAll" not in state  # no DOM sweep per field


@node
class TestFileListRetry:
    def test_a_failing_page_is_retried_with_a_backoff_not_in_a_loop(self):
        r = js(
            """
            const h = buildIt({total: 5000});
            const timers = [];
            h.sandbox.setTimeout = (fn, ms) => { timers.push({fn, ms}); return timers.length; };
            let calls = 0;
            h.sandbox.fetch = async () => { calls++; throw new Error('network down'); };
            h.run('reloadFileList()');
            for (let i = 0; i < 5; i++) await new Promise(r => setImmediate(r));
            const afterFirst = calls;
            const delays = timers.map(t => t.ms);
            timers[0].fn();
            for (let i = 0; i < 5; i++) await new Promise(r => setImmediate(r));
            output({afterFirst, afterRetry: calls, delays: timers.map(t => t.ms)});
        """,
            harness=FILELIST_HARNESS,
            env={"FILELIST_JS": str(STATIC / "filelist.js")},
        )
        assert r["afterFirst"] == 1
        assert r["afterRetry"] == 2
        assert r["delays"][:2] == [1000, 2000]


@node
def test_a_finished_row_carries_no_index_badge():
    """The 🧠 with the number of indexed passages sat on every finished row
    and could not be clicked — gone; the statistics show the coverage."""
    r = js(
        """
        const h = buildIt({total: 1});
        h.sandbox.plural = (n, w) => n + ' ' + w + (n === 1 ? '' : 's');
        const div = h.run("_buildFileRow({name: 'talk.m4a', status: 'done', size: 1000, "
                          + "duration: 61, tx_word_count: 65}, new Set())");
        output({html: div.innerHTML, loader: h.value("typeof loadEmbeddedKeys")});
    """,
        harness=FILELIST_HARNESS,
        env={"FILELIST_JS": str(STATIC / "filelist.js")},
    )
    assert "talk.m4a" in r["html"] and "65w" in r["html"]
    assert "🧠" not in r["html"] and "fi-emb-badge" not in r["html"]
    assert r["loader"] == "undefined"  # nor the library-wide lookup behind it


@node
class TestPlayingSegment:
    def test_the_segment_at_a_time_is_found_by_its_start(self):
        r = js("""
            const h = buildIt({files: ['fileview.js']});
            const sorted = {starts: [0, 5, 10], sorted: true};
            const unsorted = {starts: [0, 10, 5], sorted: false};
            output({
                at: [-1, 0, 4.9, 5, 7, 10, 99].map(t => h.run(`_playSegAt(${JSON.stringify(sorted)}, ${t})`)),
                unsorted: [7, 12].map(t => h.run(`_playSegAt(${JSON.stringify(unsorted)}, ${t})`)),
            });
        """)
        assert r["at"] == [-1, 0, 0, 1, 1, 2, 2]
        # Unsorted lists keep the old rule: start <= t < start of the NEXT element
        assert r["unsorted"] == [2, 2]


@node
class TestEditor:
    def test_the_preview_escapes_the_time_label(self):
        r = js("""
            const h = buildIt({files: ['editor.js']});
            h.run(`activeTranscript = {segments: [{text: 'hello world',
                   start_fmt: '<img src=x onerror=alert(1)>'}]}`);
            h.byId('frFind').value = 'hello';
            h.byId('frReplace').value = 'bye';
            h.run('_frPreview()');
            output({html: h.byId('frPreview').innerHTML});
        """)
        assert "<img" not in r["html"]
        assert "&lt;img" in r["html"]

    def test_a_save_landing_after_a_file_switch_leaves_the_new_file_alone(self):
        r = js("""
            let release;
            const h = buildIt({files: ['editor.js'], fetch: async () => {
                await new Promise(r => { release = r; }); return {ok: true}; }});
            const ta = h.element('ta'); ta.value = 'edited';
            const box = h.byId('seg-txt-0');
            box.querySelector = () => ta;
            box.closest = () => null;
            h.run(`activeFile = 'a.mp4'; activeTranscript = {segments: [{text: 'old A'}]}`);
            const saving = h.run('_saveSegEdit(0)');
            await h.settle();
            h.run(`activeFile = 'b.mp4'; activeTranscript = {segments: [{text: 'B text'}]}`);
            release();
            await saving;
            output({b: h.run('activeTranscript.segments[0].text'),
                    sent: h.requests[0].body.filename});
        """)
        assert r["b"] == "B text"
        assert r["sent"] == "a.mp4"

    def test_a_save_brings_the_word_count_along(self):
        """The header's "Words" kept the old count until the file was opened
        again."""
        r = js("""
            const h = buildIt({files: ['editor.js'], fetch: async () => ({ok: true})});
            const ta = h.element('ta'); ta.value = 'one two three';
            const box = h.byId('seg-txt-0');
            box.querySelector = () => ta;
            box.closest = () => null;
            const count = {nodeType: 3, textContent: '5'};
            const chip = {querySelector: () => ({textContent: 'Words'}), lastChild: count};
            h.sandbox.document.querySelectorAll = sel => sel === '.fi-chip' ? [chip] : [];
            const file = {name: 'a.mp4', tx_word_count: 5};
            h.sandbox.fileByName = () => file;
            h.run(`activeFile = 'a.mp4'; activeTranscript = {segments: [{text: 'a b c d'}, {text: 'e'}]}`);
            await h.run('_saveSegEdit(0)');
            output({chip: count.textContent, file: file.tx_word_count});
        """)
        assert r == {"chip": "4", "file": 4}


@node
class TestLogModal:
    def test_a_late_lookup_does_not_replace_the_newer_log(self):
        r = js("""
            const h = buildIt({files: ['transcribe.js'], fetch: async url => {
                if (url.includes('a.mp4/logFile')) { await sleep(30); return {log_file: 'a.log'}; }
                if (url.includes('b.mp4/logFile')) return {log_file: 'b.log'};
                return {text: url.includes('a.log') ? 'LOG A' : 'LOG B'};
            }});
            const first = h.run("showLogModal('', 'a.mp4')");
            await h.run("showLogModal('', 'b.mp4')");
            await first;
            const afterBoth = h.byId('logModalContent').innerHTML;
            // Closed while a lookup runs: the closed modal stays untouched
            h.byId('logModalContent').innerHTML = '';
            const third = h.run("showLogModal('', 'a.mp4')");
            h.run('closeLogModal()');
            await third;
            output({afterBoth, afterClose: h.byId('logModalContent').innerHTML});
        """)
        assert r["afterBoth"] == "LOG B"
        assert r["afterClose"] == ""


@node
class TestTranscriptReload:
    """A failed RELOAD of the open file used to wipe its good view
    and null activeTranscript. Now only a load for a newly shown file
    takes the area over; a same-file reload failure keeps it and toasts."""

    def test_a_failed_reload_keeps_the_view(self):
        r = js("""
            const h = buildIt({files: ['fileview.js'], fetch: async () => ({__status: 500})});
            h.run(`activeFile = 'a.mp4'; activeTranscript = {filename: 'a.mp4', segments: [{text: 'good'}]}`);
            h.byId('transcriptArea').innerHTML = 'GOOD VIEW';
            await h.run("loadTranscript('a.mp4')");
            output({area: h.byId('transcriptArea').innerHTML,
                    kept: h.run('activeTranscript && activeTranscript.segments[0].text'),
                    toasts: h.toasts.filter(t => t.type === 'error').length});
        """)
        assert r["area"] == "GOOD VIEW"
        assert r["kept"] == "good"
        assert r["toasts"] == 1

    def test_a_failed_first_load_shows_the_error(self):
        r = js("""
            const h = buildIt({files: ['fileview.js'], fetch: async () => { throw new Error('down'); }});
            h.run(`activeFile = 'b.mp4'; activeTranscript = null`);
            await h.run("loadTranscript('b.mp4')");
            output({area: h.byId('transcriptArea').innerHTML,
                    active: h.run('activeTranscript')});
        """)
        assert "Failed to load the transcript: down" in r["area"]
        assert r["active"] is None


def test_an_empty_transcript_says_so():
    """A recording without speech showed a blank view that read as still
    loading."""
    src = (STATIC / "fileview.js").read_text(encoding="utf-8")
    assert "if (!(d.segments || []).some(s => String(s.text || '').trim()))" in src
    assert "No speech found — the transcript is empty" in src


def test_find_and_replace_grows_downward():
    """Centred, the dialog moved its fields up while its preview grew: a
    click meant for "Replace with" missed."""
    css = (REPO / "whisper_tm" / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert "#findReplaceModal.modal-bg { align-items:flex-start; }" in css

"""Queue Editor, UI: selection, target body, dry run → strips → live run,
mode bars, 409 recount, keyboard — no browser, via node harness.

The one sentence secured here: **the click hits what the
user saw.** qeTarget() builds exactly one body from filter + selection;
each action runs first as a dry run; removal from 200 only after
typing the number; a 409 leads to a NEW count with strips, never
to a second write attempt. Also covered: retry runs the same path, the root
dropdown doesn't clear itself, late responses expire, selections do NOT
survive closing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "queue_editor_harness.js"
JS = REPO / "whisper_tm" / "static" / "js" / "queue_editor.js"
INDEX = REPO / "whisper_tm" / "static" / "index.html"
CSS = REPO / "whisper_tm" / "static" / "css" / "app.css"
FILELIST = REPO / "whisper_tm" / "static" / "js" / "filelist.js"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"QUEUE_EDITOR_JS": str(JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


ITEMS = """
    h.setAnswer('/api/queue/editor/items', (url) => ({ok: true, total: 5, offset: 0, limit: 200,
        items: [1,2,3,4,5].map(i => ({id: i, filename: 'videos/shows/e' + i + '.mkv', dir: '/m', settings: {}, status: 'pending', position: i, error: ''})),
        roots: [{dir: '/m', count: 5}], processing: {id: 99, filename: 'running.mkv', dir: '/m', settings: {}, status: 'processing'},
        generation: 7, filter: {status: ['pending'], dir: null, prefix: '', recursive: true, q: '', setting: null}}));
"""


@node
class TestTargetAndSelection:
    def test_the_default_filter_without_a_selection_is_all(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            output({target: h.value("qeTarget()"), count: h.value("qeCount()"), gets: h.requests.map(a => a.url)});
        """
        )
        assert r["target"] == {"all": True} and r["count"] == 5
        assert any(
            "/api/queue/editor/items?status=pending&offset=0&limit=200" in u for u in r["gets"]
        )
        assert any("/api/queue/editor/folders?status=pending" in u for u in r["gets"])

    def test_selecting_individually_yields_ids_and_the_scope_jumps(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(2, {}); qeRowClick(4, {ctrlKey: true})");
            const a = {target: h.value("qeTarget()"), scope: h.value("qeScope"), selected: h.value("qeSelectedCount()")};
            h.run("qeRowClick(2, {})");                           // deselect again
            const b = {target: h.value("qeTarget()"), scope: h.value("qeScope")};
            h.run("qeRowClick(4, {})");                           // last one gone → scope back to all
            output({a, b, c: {scope: h.value("qeScope"), target: h.value("qeTarget()")}});
        """
        )
        assert r["a"] == {"target": {"ids": [2, 4]}, "scope": "selected", "selected": 2}
        assert r["b"] == {"target": {"ids": [4]}, "scope": "selected"}
        assert r["c"] == {"scope": "all_items", "target": {"all": True}}

    def test_shift_click_selects_a_range(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(2, {}); qeRowClick(5, {shiftKey: true})");
            output({target: h.value("qeTarget()")});
        """
        )
        assert r["target"] == {"ids": [2, 3, 4, 5]}

    def test_a_processing_row_cannot_be_selected(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(99, {})");
            output({n: h.value("qeSelectedCount()"), html: h.elements.qeList.innerHTML});
        """
        )
        assert r["n"] == 0
        assert "running — not editable" in r["html"] and 'data-id="99"' in r["html"]

    def test_all_hits_with_a_deselection_becomes_except_ids(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeSelectAll(); qeRowClick(3, {})");
            output({target: h.value("qeTarget()"), n: h.value("qeSelectedCount()")});
        """
        )
        assert r["target"] == {"all": True, "except_ids": [3]} and r["n"] == 4

    def test_a_filter_change_keeps_the_selection_but_not_all(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qePickFolder('videos/shows/', '/m')"); await h.runTimers();
            const a = {target: h.value("qeTarget()"), filter: h.value("({...qeFilter})"), toasts: h.toasts.info};
            h.run("qeClearSelection(); qeSelectAll(); qeCrumb('')"); await h.runTimers();
            output({a, b: {all_items: h.value("qeAll"), scope: h.value("qeScope")}});
        """
        )
        assert r["a"]["target"] == {"ids": [1]} and r["a"]["filter"]["prefix"] == "videos/shows/"
        assert r["a"]["filter"]["dir"] == "/m" and r["a"]["toasts"] == 1
        assert r["b"] == {"all_items": False, "scope": "all_items"}

    def test_the_selection_does_not_survive_closing_or_a_status_change(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qeRowClick(2, {ctrlKey: true}); closeQueueEditor()");
            h.run("openQueueEditor()"); await h.runTimers();
            const a = {n: h.value("qeSelection.size"), scope: h.value("qeScope"), anchor: h.value("qeAnchor")};
            h.run("qeRowClick(1, {}); qeStatusChanged('error')"); await h.runTimers();
            const b = {n: h.value("qeSelection.size"), scope: h.value("qeScope"), toasts: h.toasts.info};
            output({a, b});
        """
        )
        assert r["a"] == {"n": 0, "scope": "all_items", "anchor": None}
        assert r["b"] == {"n": 0, "scope": "all_items", "toasts": 0}

    def test_a_filter_target_without_a_selection(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeSettingFilterChanged('low_vram=false')"); await h.runTimers();
            const url = h.requests.filter(a => a.url.includes('/items')).slice(-1)[0].url;
            output({target: h.value("qeTarget()"), url});
        """
        )
        assert r["target"]["filter"]["setting"] == {"key": "low_vram", "value": False}
        assert "setting_key=low_vram&setting_value=false" in r["url"]


FOLDERS = """
    h.setAnswer('/api/queue/editor/folders', (url) => ({ok: true, generation: 1, total: 9, media_dir: '/media',
        files_here: url.includes('prefix=') ? 2 : 0,
        folders: url.includes('prefix=') ? [] : [{name: 'videos', prefix: 'videos/', dir: '/nas', count: 6, has_children: true},
                                               {name: 'clips', prefix: 'clips/', dir: '', count: 3, has_children: false}],
        roots: [{dir: '/nas', count: 6}, {dir: '', count: 3}]}));
"""


@node
class TestFoldersAndRoot:
    def test_media_root_and_all_are_different_values(self):
        r = js(
            ITEMS
            + FOLDERS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            const html = h.elements.qeRoot.innerHTML;
            const a = {visible: h.elements.qeRoot.style.display, value: h.elements.qeRoot.value,
                       empty: (html.match(/value=""/g) || []).length, all_items: html.includes('value="*"'), html};
            h.run("qeRootChanged('')"); await h.runTimers();      // select media root
            const b = {dir: h.value("qeFilter.dir"), value: h.elements.qeRoot.value, visible: h.elements.qeRoot.style.display};
            h.run("qeRootChanged('*')"); await h.runTimers();
            output({a, b, c: h.value("qeFilter.dir")});
        """
        )
        assert (
            r["a"]["visible"] == ""
            and r["a"]["value"] == "*"
            and r["a"]["empty"] == 1
            and r["a"]["all_items"]
        )
        assert "/media (media root)" in r["a"]["html"]
        assert r["b"] == {"dir": "", "value": "", "visible": ""}
        assert r["c"] is None

    def test_the_dropdown_stays_after_a_root_choice_and_the_breadcrumb_restores_it(self):
        r = js(
            ITEMS
            + FOLDERS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRootChanged('/nas')"); await h.runTimers();
            const a = {dir: h.value("qeFilter.dir"), visible: h.elements.qeRoot.style.display, value: h.elements.qeRoot.value};
            // Enter a folder row from a DIFFERENT root → dir follows the row
            h.run("qePickFolder('clips/', '')"); await h.runTimers();
            const b = {dir: h.value("qeFilter.dir"), prefix: h.value("qeFilter.prefix"), value: h.elements.qeRoot.value};
            // ⌂ all → back to the root chosen in the dropdown, not to "all"
            h.run("qeCrumb('')"); await h.runTimers();
            const c = {dir: h.value("qeFilter.dir"), prefix: h.value("qeFilter.prefix"), value: h.elements.qeRoot.value};
            h.run("qeRootChanged('*'); qePickFolder('videos/', '/nas')"); await h.runTimers();
            h.run("qeCrumb('')"); await h.runTimers();
            output({a, b, c, d: h.value("qeFilter.dir")});
        """
        )
        assert r["a"] == {"dir": "/nas", "visible": "", "value": "/nas"}
        assert r["b"] == {"dir": "", "prefix": "clips/", "value": ""}
        assert r["c"] == {"dir": "/nas", "prefix": "", "value": "/nas"}
        assert r["d"] is None

    def test_one_root_means_no_dropdown_but_a_foreign_root_stays_selectable(self):
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/folders', {ok: true, generation: 1, total: 3, files_here: 0, folders: [],
                roots: [{dir: '/nas', count: 3}]});
            h.run("openQueueEditor()"); await h.runTimers();
            const a = h.elements.qeRoot.style.display;
            // Folder row sets dir to the ONLY root → nothing to choose, no dropdown flicker
            h.run("qePickFolder('videos/', '/nas')"); await h.runTimers();
            const b = h.elements.qeRoot.style.display;
            // dir set to a root that has nothing in this status → selectable as "(0)", so the filter stays visible
            h.run("qeRootChoice = '/old_value'; qeSetFilter({dir: '/old_value', prefix: ''})"); await h.runTimers();
            const c = {visible: h.elements.qeRoot.style.display, value: h.elements.qeRoot.value,
                       null: h.elements.qeRoot.innerHTML.includes('value="/old_value">/old_value (0)')};
            output({a, b, c});
        """
        )
        assert r["a"] == "none" and r["b"] == "none"
        assert r["c"] == {"visible": "", "value": "/old_value", "null": True}

    def test_a_status_change_closes_folders_and_keeps_the_dropdown_root(self):
        r = js(
            ITEMS
            + FOLDERS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRootChanged('/nas'); qePickFolder('clips/', '')"); await h.runTimers();
            h.run("qeStatusChanged('error')"); await h.runTimers();
            output({f: h.value("({...qeFilter})")});
        """
        )
        assert r["f"]["status"] == ["error"] and r["f"]["prefix"] == "" and r["f"]["dir"] == "/nas"

    def test_here_only_turns_subfolders_off_and_on_again(self):
        r = js(
            ITEMS
            + FOLDERS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qePickFolder('videos/', '/nas')"); await h.runTimers();
            const a = {html: h.elements.qeFolder.innerHTML, tick_mark: h.elements.qeRecursive.checked};
            h.run("qeOnlyHere()"); await h.runTimers();
            const url = h.requests.filter(x => x.url.includes('/items')).slice(-1)[0].url;
            const b = {recursive: h.value("qeFilter.recursive"), tick_mark: h.elements.qeRecursive.checked, url,
                       active: h.elements.qeFolder.innerHTML.includes('qe-folder-here active')};
            h.run("qeOnlyHere()"); await h.runTimers();
            output({a, b, c: h.value("qeFilter.recursive")});
        """
        )
        assert (
            "qeOnlyHere()" in r["a"]["html"]
            and "directly here" in r["a"]["html"]
            and r["a"]["tick_mark"] is True
        )
        assert (
            r["b"]["recursive"] is False
            and r["b"]["tick_mark"] is False
            and "recursive=0" in r["b"]["url"]
            and r["b"]["active"]
        )
        assert r["c"] is True

    def test_the_arrow_closes_the_editor_before_jumping(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            output({html: h.elements.qeList.innerHTML});
        """
        )
        assert "closeQueueEditor(); openQueueFile(" in r["html"]

    def test_the_file_name_narrows_with_the_path(self):
        """The queue editor narrows by file name as well as by path."""
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.requests.length = 0;
            h.elements.qeSearch.value = ' videos ';
            h.elements.qeName.value = ' intro* ';
            h.run("qeSearchChanged()"); await h.runTimers();
            output({gets: h.requests.map(a => a.url), target: h.value("qeTarget()"),
                    reopened: (h.run("closeQueueEditor(); openQueueEditor()"), h.elements.qeName.value)});
        """
        )
        assert any("q=videos&name=intro*" in u.replace("%2A", "*") for u in r["gets"]), r["gets"]
        assert r["target"]["filter"]["name"] == "intro*" and r["target"]["filter"]["q"] == "videos"
        assert r["reopened"] == ""  # a new opening starts unfiltered

    def test_a_late_answer_does_not_overwrite_a_newer_one(self):
        """Broad search (slow) → narrower search (fast): the slow response is discarded."""
        r = js("""
            const orig = h.sandbox.fetch;
            let hold = null;
            h.sandbox.fetch = async (url, opts) => {
                h.requests.push({url, method: 'GET', body: null});
                const items = (n) => ({ok: true, status: 200, json: async () => ({ok: true, total: n, offset: 0, limit: 200,
                    items: [{id: 1, filename: 'a.mkv', dir: '', settings: {}, status: 'pending'}], roots: [], processing: null, generation: 1})});
                if (url.includes('/folders')) return {ok: true, status: 200, json: async () => ({ok: true, folders: [], roots: [], files_here: 0, total: 0, generation: 1})};
                if (url.includes('q=vid&')) return new Promise(r => { hold = () => r(items(25000)); });
                return items(url.includes('q=videos') ? 8000 : 3);
            };
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeSetFilter({q: 'vid'})");                 // hangs
            h.run("qeSetFilter({q: 'videos'})"); await h.runTimers();
            const a = h.value("qePage.total");
            hold(); await h.runTimers();                    // the old response only arrives now
            output({a, b: h.value("qePage.total"), q: h.value("qeFilter.q"), hits: h.elements.qeHits.textContent});
        """)
        assert r["a"] == 8000 and r["b"] == 8000 and r["q"] == "videos"
        assert r["hits"] == "8.000" or r["hits"] == "8,000"


@node
class TestActions:
    def test_low_vram_for_all_dry_run_strip_real_run(self):
        """User flow A: two clicks plus confirmation with a number."""
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/settings', (url, body) => body.dry_run
                ? {ok: true, dry_run: true, affected: 11234, skipped_processing: 1, roots: [{dir: '/m', count: 11234}],
                   sample: ['a.mkv', 'b.mkv'], patch: body.settings, generation: 7}
                : {ok: true, dry_run: false, affected: 11234, skipped_processing: 1, generation: 8, patch: body.settings});
            h.run("openQueueEditor()"); await h.runTimers();
            await h.value("qeAction('settings', {settings: {low_vram: true}})");
            const stripe = {visible: h.elements.qeConfirmation.style.display, text: h.elements.qeConfirmText.textContent,
                              button: h.elements.qeBtnConfirm.textContent, locked: h.elements.qeBtnConfirm.disabled,
                              number: h.elements.qeCountInput.style.display};
            const posts1 = h.requests.filter(a => a.method === 'POST').map(a => a.body);
            await h.value("qeConfirm()");
            const posts2 = h.requests.filter(a => a.method === 'POST').map(a => a.body);
            output({stripe, posts1, posts2, toasts: h.toasts, counter: h.counter,
                     stripAfter: h.elements.qeConfirmation.style.display});
        """
        )
        assert r["stripe"]["visible"] == ""
        assert (
            "Changing 11,234 pending entries" in r["stripe"]["text"]
            or "Changing 11,234" in r["stripe"]["text"]
        )
        assert "1 currently running and left untouched" in r["stripe"]["text"]
        assert "Affects ALL pending entries" in r["stripe"]["text"]
        assert "Spare VRAM → on" in r["stripe"]["text"]
        assert r["stripe"]["locked"] is False and r["stripe"]["number"] == "none"
        assert r["posts1"] == [{"all": True, "settings": {"low_vram": True}, "dry_run": True}]
        assert r["posts2"][-1] == {
            "all": True,
            "settings": {"low_vram": True},
            "dry_run": False,
            "expected": 11234,
        }
        assert (
            r["toasts"]["success"] == 1
            and r["counter"]["refreshQueue"] == 1
            and r["counter"]["renderFileList"] == 1
        )
        assert r["stripAfter"] == "none"

    def test_a_small_change_runs_immediately(self):
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/move', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 3,
                skipped_processing: 0, roots: [], sample: [], to: 'top', generation: 9}));
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qeRowClick(3, {ctrlKey: true}); qeRowClick(5, {ctrlKey: true})");
            await h.value("qeAction('move', {to: 'top'})");
            output({posts: h.requests.filter(a => a.method === 'POST').map(a => a.body), stripe: h.elements.qeConfirmation.style.display,
                     toasts: h.toasts.success});
        """
        )
        assert r["posts"] == [
            {"ids": [1, 3, 5], "to": "top", "dry_run": True},
            {"ids": [1, 3, 5], "to": "top", "dry_run": False, "expected": 3},
        ]
        assert r["stripe"] == "none" and r["toasts"] == 1

    def test_removing_always_shows_the_strip_and_the_count_from_200_on(self):
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/remove', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 5000,
                skipped_processing: 0, roots: [{dir: '/m', count: 5000}], sample: ['x'], generation: 9}));
            h.run("openQueueEditor()"); await h.runTimers();
            await h.value("qeAction('remove', {})");
            const before = {locked: h.elements.qeBtnConfirm.disabled, number: h.elements.qeCountInput.style.display,
                            danger: h.elements.qeConfirmation.classList.contains('danger')};
            h.elements.qeCountInput.value = '4999'; h.run("qeCountChecked()");
            const wrong = h.elements.qeBtnConfirm.disabled;
            h.elements.qeCountInput.value = '5.000'; h.run("qeCountChecked()");
            const right = h.elements.qeBtnConfirm.disabled;
            const posts = h.requests.filter(a => a.method === 'POST').length;
            output({before, wrong, right, posts});
        """
        )
        assert r["before"] == {"locked": True, "number": "", "danger": True}
        assert r["wrong"] is True and r["right"] is False
        assert r["posts"] == 1  # only the dry run

    def test_a_small_removal_needs_the_strip_but_no_count(self):
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/remove', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 3,
                skipped_processing: 0, roots: [], sample: [], generation: 9}));
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qeRowClick(3, {shiftKey: true})");
            await h.value("qeAction('remove', {})");
            output({stripe: h.elements.qeConfirmation.style.display, locked: h.elements.qeBtnConfirm.disabled,
                     number: h.elements.qeCountInput.style.display, posts: h.requests.filter(a => a.method === 'POST').length});
        """
        )
        assert r == {"stripe": "", "locked": False, "number": "none", "posts": 1}

    def test_409_leads_to_a_recount_not_to_a_second_write_attempt(self):
        r = js(
            ITEMS
            + """
            let real = 0;
            h.setAnswer('/api/queue/editor/remove', (url, body) => {
                if (body.dry_run) return {ok: true, dry_run: true, affected: 30, skipped_processing: 0, roots: [], sample: [], generation: 1};
                real++; return {error: 'Match count has changed: expected 30, now 29', affected: 29};
            });
            h.setStatus('/api/queue/editor/remove', 409, null);       // overwritten below
            h.setAnswer('/api/queue/editor/remove', (url, body) => {
                if (body.dry_run) return {ok: true, dry_run: true, affected: 30, skipped_processing: 0, roots: [], sample: [], generation: 1};
                real++; return {error: 'Match count has changed: expected 30, now 29', affected: 29};
            });
            // Status per response: dry 200, real 409
            const orig = h.sandbox.fetch;
            h.sandbox.fetch = async (url, opts) => {
                const r = await orig(url, opts);
                const body = opts && opts.body ? JSON.parse(opts.body) : {};
                if (url.includes('/remove') && !body.dry_run) return {ok: false, status: 409, json: r.json};
                return r;
            };
            h.run("openQueueEditor()"); await h.runTimers();
            await h.value("qeAction('remove', {})");
            h.elements.qeCountInput.value = '30'; h.run("qeCountChecked()");
            await h.value("qeConfirm()");
            const posts = h.requests.filter(a => a.method === 'POST').map(a => a.body.dry_run);
            output({posts, real, text: h.elements.qeConfirmText.textContent, pending: h.value("qePending && qePending.dry.affected")});
        """
        )  # noqa: E501
        assert r["posts"] == [True, False, True]  # dry → real(409) → NEW dry, no second write
        assert r["real"] == 1
        assert r["pending"] == 30  # new dry run awaits confirmation
        assert r["text"].startswith("The queue has changed")

    def test_the_real_run_sends_the_frozen_target_and_a_filter_change_clears_the_strip(self):
        """Strip open for folder A → user clicks folder B → "Yes" must NOT hit B."""
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/remove', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 25,
                skipped_processing: 0, roots: [], sample: [], generation: 9}));
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qePickFolder('A/', '/m')"); await h.runTimers();
            await h.value("qeAction('remove', {})");
            const a = {stripe: h.elements.qeConfirmation.style.display, target: h.value("qePending.target.filter.prefix")};
            h.run("qePickFolder('B/', '/m')"); await h.runTimers();
            const b = {stripe: h.elements.qeConfirmation.style.display, pending: h.value("qePending")};
            await h.value("qeConfirm()");                       // dead click: nothing pending
            const posts1 = h.requests.filter(x => x.method === 'POST').length;
            // And once the strip is up, EXACTLY the frozen target goes out — even if qeFilter was mutated
            await h.value("qeAction('remove', {})");
            h.run("qeFilter.prefix = 'C/'");                      // without the setter, so without clearing it
            await h.value("qeConfirm()");
            const real = h.requests.filter(x => x.method === 'POST' && x.body.dry_run === false).map(x => x.body.filter.prefix);
            output({a, b, posts1, real});
        """
        )
        assert r["a"] == {"stripe": "", "target": "A/"}
        assert r["b"] == {"stripe": "none", "pending": None}
        assert r["posts1"] == 1  # only the first dry run
        assert r["real"] == ["B/"]  # frozen at dry-run time, not 'C/'

    def test_409_then_zero_hits_leaves_no_strip_behind(self):
        r = js(
            ITEMS
            + """
            let rounds = 0;
            const orig = h.sandbox.fetch;
            h.sandbox.fetch = async (url, opts) => {
                const body = opts && opts.body ? JSON.parse(opts.body) : {};
                if (!url.includes('/move')) return orig(url, opts);
                h.requests.push({url, method: 'POST', body});
                if (body.dry_run) { rounds++; return {ok: true, status: 200, json: async () => ({ok: true, dry_run: true, affected: rounds === 1 ? 50 : 0, skipped_processing: 0, roots: [], sample: [], generation: 1})}; }
                return {ok: false, status: 409, json: async () => ({error: 'Match count has changed: expected 50, now 0', affected: 0})};
            };
            h.run("openQueueEditor()"); await h.runTimers();
            await h.value("qeAction('move', {to: 'top'})");         // 50 → strip
            const a = h.elements.qeConfirmation.style.display;
            await h.value("qeConfirm()");                        // 409 → new dry run: 0
            output({a, stripe: h.elements.qeConfirmation.style.display, pending: h.value("qePending"), warn: h.toasts.warn});
        """
        )
        assert r == {"a": "", "stripe": "none", "pending": None, "warn": 1}

    def test_ghosts_among_the_selected_force_the_strip(self):
        """15 marked, 5 still pending, 1 running → no instant path; strip names 9 ghosts (not 10)."""
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/settings', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 5,
                requested: 15, skipped_processing: 1, roots: [], sample: [], patch: {low_vram: true}, generation: 2}));
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("for (let i = 1; i <= 15; i++) qeSelection.add(i); qeScope = 'selected'");
            await h.value("qeAction('settings', {settings: {low_vram: true}})");
            const a = {stripe: h.elements.qeConfirmation.style.display, text: h.elements.qeConfirmText.textContent,
                       posts: h.requests.filter(x => x.method === 'POST').length};
            await h.value("qeConfirm()");
            output({a, selected: h.value("qeSelection.size"), success: h.toasts.success});
        """
        )
        assert r["a"]["stripe"] == "" and r["a"]["posts"] == 1
        assert "9 of the marked entries are no longer in the target" in r["a"]["text"]
        assert "1 currently running" in r["a"]["text"]
        assert r["selected"] == 0 and r["success"] == 1  # ghost marks gone after the run

    def test_the_retry_strip_names_the_status_and_no_running_entry(self):
        r = js("""
            h.setAnswer('/api/queue/editor/items', {ok: true, total: 30, offset: 0, limit: 200, roots: [], processing: null, generation: 1,
                items: [{id: 7, filename: 'x.mkv', dir: '', settings: {}, status: 'cancelled', error: ''}]});
            h.setAnswer('/api/queue/editor/retry', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 30,
                skipped_processing: 0, roots: [{dir: '', count: 30}], sample: ['x.mkv'], generation: 3}));
            h.run("openQueueEditor({status: ['cancelled']})"); await h.runTimers();
            await h.value("qeRetry()");
            output({text: h.elements.qeConfirmText.textContent, button: h.elements.qeBtnConfirm.textContent,
                     retryVisible: h.elements.qeBtnRetry.style.display});
        """)
        assert r["text"].startswith("Re-queueing: 30 cancelled entries")
        assert "currently running" not in r["text"] and r["button"] == "Yes, re-queue (30)"
        assert r["retryVisible"] == ""

    def test_a_late_dry_run_answer_after_closing_does_nothing(self):
        r = js(
            ITEMS
            + """
            let free = null;
            const orig = h.sandbox.fetch;
            h.sandbox.fetch = async (url, opts) => {
                const body = opts && opts.body ? JSON.parse(opts.body) : {};
                if (!url.includes('/settings')) return orig(url, opts);
                h.requests.push({url, method: 'POST', body});
                if (body.dry_run) return new Promise(r => { free = () => r({ok: true, status: 200, json: async () => ({ok: true, dry_run: true, affected: 3, skipped_processing: 0, roots: [], sample: [], generation: 1})}); });
                return {ok: true, status: 200, json: async () => ({ok: true, affected: 3, generation: 2})};
            };
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qeRowClick(3, {shiftKey: true})");
            const p = h.value("qeAction('settings', {settings: {low_vram: true}})");
            h.run("closeQueueEditor(); openQueueEditor()"); await h.runTimers();
            free(); await p; await h.runTimers();
            output({posts: h.requests.filter(x => x.method === 'POST').map(x => x.body.dry_run), running_flag: h.value("qeRunning"),
                     pending: h.value("qePending")});
        """
        )
        assert r == {"posts": [True], "running_flag": False, "pending": None}

    def test_409_below_20_hits_shows_the_strip_instead_of_ping_pong(self):
        """Instant path (≤20) + 409 → do NOT write instantly again: strip, user decides."""
        r = js(
            ITEMS
            + """
            let n = 15;
            const orig = h.sandbox.fetch;
            h.sandbox.fetch = async (url, opts) => {
                const body = opts && opts.body ? JSON.parse(opts.body) : {};
                if (!url.includes('/move')) return orig(url, opts);
                h.requests.push({url, method: 'POST', body});
                if (body.dry_run) return {ok: true, status: 200, json: async () => ({ok: true, dry_run: true, affected: n, skipped_processing: 0, roots: [], sample: [], generation: 1})};
                n -= 1;
                return {ok: false, status: 409, json: async () => ({error: 'Match count has changed: expected ' + (n + 1) + ', now ' + n, affected: n})};
            };
            h.run("openQueueEditor()"); await h.runTimers();
            await h.value("qeAction('move', {to: 'top'})");
            output({posts: h.requests.filter(a => a.method === 'POST').map(a => a.body.dry_run),
                     stripe: h.elements.qeConfirmation.style.display, text: h.elements.qeConfirmText.textContent,
                     button: h.elements.qeBtnConfirm.textContent, pending: h.value("qePending && qePending.dry.affected")});
        """
        )
        assert r["posts"] == [True, False, True]  # exactly ONE 409 cycle, then strip
        assert r["stripe"] == "" and "has changed" in r["text"] and "(14)" in r["button"]
        assert r["pending"] == 14

    def test_an_old_server_gives_404(self):
        r = js("""
            // app.py responds to a 404 with {"error": "Not found"} — the text is
            // therefore never "HTTP 404"; it's recognized by the status that api() attaches to the error.
            h.setStatus('/api/queue/editor/items', 404, {error: 'Not found'});
            h.setStatus('/api/queue/editor/folders', 404, {error: 'Not found'});
            h.run("openQueueEditor()"); await h.runTimers();
            output({hint: h.elements.qeHint.textContent, visible: h.elements.qeHint.style.display,
                     locked: h.elements.qeBtnLowVram.disabled, missing: h.value("qeServerMissing"), errorToasts: h.toasts.error});
        """)
        assert "restart needed" in r["hint"] and r["visible"] == "" and r["locked"] is True
        assert r["missing"] is True and r["errorToasts"] == 0

    def test_the_error_view_shows_retry_instead_of_mutations(self):
        r = js("""
            h.setAnswer('/api/queue/editor/items', {ok: true, total: 2, offset: 0, limit: 200, roots: [], processing: null, generation: 1,
                items: [{id: 7, filename: 'x.mkv', dir: '', settings: {}, status: 'error', error: 'Separation failed'},
                        {id: 8, filename: 'y.mkv', dir: '', settings: {}, status: 'error', error: ''}]});
            h.setAnswer('/api/queue/editor/retry', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: body.ids ? 1 : 2,
                skipped_processing: 0, roots: [], sample: [], generation: 3}));
            h.run("openQueueEditor({status: ['error']})"); await h.runTimers();
            const a = {retry: h.elements.qeBtnRetry.style.display, remove: h.elements.qeBtnRemove.style.display,
                       retryLocked: h.elements.qeBtnRetry.disabled, html: h.elements.qeList.innerHTML,
                       text: h.elements.qeBtnRetry.textContent};
            h.run("qeRowClick(7, {})");
            const b = {retryLocked: h.elements.qeBtnRetry.disabled, text: h.elements.qeBtnRetry.textContent};
            await h.value("qeRetry()");
            const c = {selected: h.value("qeSelection.size"), scope: h.value("qeScope"), toasts: h.toasts.success};
            output({a, b, c, posts: h.requests.filter(x => x.method === 'POST').map(x => [x.url, x.body])});
        """)
        assert r["a"]["retry"] == "" and r["a"]["remove"] == "none"
        assert r["a"]["retryLocked"] is False and r["a"]["text"] == "↩ Re-queue (2)"
        assert "Separation failed" in r["a"]["html"]
        assert r["b"]["retryLocked"] is False and r["b"]["text"] == "↩ Re-queue (1)"
        # Dry run → real run with expected — the same path as every action, no /api/queue/retry
        assert r["posts"] == [
            ["/api/queue/editor/retry", {"ids": [7], "dry_run": True}],
            ["/api/queue/editor/retry", {"ids": [7], "dry_run": False, "expected": 1}],
        ]
        assert r["c"] == {"selected": 0, "scope": "all_items", "toasts": 1}

    def test_retry_all_sends_the_displayed_status_not_an_empty_one(self):
        """Error view shows 12 → the body carries status ['error'], never {} (= error AND cancelled)."""
        r = js("""
            h.setAnswer('/api/queue/editor/items', {ok: true, total: 12, offset: 0, limit: 200, roots: [], processing: null, generation: 1,
                items: [{id: 7, filename: 'x.mkv', dir: '', settings: {}, status: 'error', error: ''}]});
            h.setAnswer('/api/queue/editor/retry', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 12,
                skipped_processing: 0, roots: [], sample: [], generation: 3}));
            h.run("openQueueEditor({status: ['error']})"); await h.runTimers();
            await h.value("qeRetry()");
            const posts = h.requests.filter(x => x.method === 'POST').map(x => x.body);
            output({posts, stripe: h.elements.qeConfirmation.style.display});
        """)
        assert len(r["posts"]) == 2 and all(p["filter"]["status"] == ["error"] for p in r["posts"])
        assert "all" not in r["posts"][0] and r["stripe"] == "none"

    def test_the_generation_triggers_a_debounced_reload(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            const before = h.requests.length;
            h.run("qeGenerationSeen(7)");                 // same generation → nothing
            h.run("qeGenerationSeen(8); qeGenerationSeen(9)");   // two changes → ONE reload
            const midway = h.requests.length;
            await h.runTimers();
            output({before, midway, after: h.requests.length});
        """
        )
        assert r["midway"] == r["before"] and r["after"] == r["before"] + 2


@node
class TestSettingsDialog:
    def test_only_groups_that_were_set_go_into_the_patch(self):
        r = js("""
            h.domLoaded();                       // install mode bars
            h.sandbox.collectValues = {model: 'turbo', language: 'de', low_vram: true, diarize: true, numSpeakers: null,
                                        vocal_isolation: true, vocal_model: 'bs', vocab_preset: ''};
            const sel = (g) => ({value: '', dataset: {group: g}, closest: () => h.elements['field:' + ({low_vram: 'qeLowVram', language: 'qeLang', speaker: 'qeDiarize', model: 'qeModel'})[g]]});
            let s = sel('low_vram'); s.value = 'setTo'; h.sandbox.__s = s; h.run("qeModeChanged(__s)");
            s = sel('language'); s.value = 'standard'; h.sandbox.__s = s; h.run("qeModeChanged(__s)");
            const a = h.value("qeCollect()");
            s = sel('speaker'); s.value = 'setTo'; h.sandbox.__s = s; h.run("qeModeChanged(__s)");
            const b = h.value("qeCollect()");
            output({a, b, summary: h.elements.qsSummary.textContent,
                     inserted: (h.elements['field:qeLowVram'].inserted || []).length,
                     modelHull: h.elements['field:qeModel'].classList.contains('qe-group'),
                     modelInactive: h.elements['field:qeModel'].classList.contains('qe-inactive')});
        """)
        assert r["a"] == {"low_vram": True, "language": None}
        assert r["b"] == {
            "low_vram": True,
            "language": None,
            "diarize": True,
            "numSpeakers": "auto",
        }
        assert "4 changes" in r["summary"] and "Language → Default" in r["summary"]
        assert r["inserted"] == 1 and r["modelHull"] and r["modelInactive"]

    def test_opening_resets_every_mode_and_counts(self):
        r = js(
            ITEMS
            + """
            h.domLoaded();
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeModeState.low_vram = 'setTo'; qeRowClick(1, {}); qeRowClick(2, {ctrlKey: true})");
            h.run("qeSettingsOpen()");
            const two = h.elements.qsCountNoun.textContent;
            h.run("qeRowClick(1, {})"); h.run("qeSettingsOpen()");
            output({count: h.elements.qsCount.textContent, mode: h.value("qeModeState.low_vram"),
                     button: h.elements.qsBtnApply.disabled, open_count: h.elements.qeSettingsModal.classList.contains('visible'),
                     nouns: [two, h.elements.qsCountNoun.textContent]});
        """
        )
        # "Change settings for 1 pending entries"
        assert r == {
            "count": "1",
            "mode": "keep",
            "button": True,
            "open_count": True,
            "nouns": ["entries", "entry"],
        }

    def test_the_dialog_stays_open_on_400_and_closes_on_success(self):
        r = js(
            ITEMS
            + """
            h.domLoaded();
            h.sandbox.collectValues = {low_vram: true};
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeSettingsOpen(); qeModeState.low_vram = 'setTo'");
            h.setStatus('/api/queue/editor/settings', 400, {error: 'Values must be scalars'});
            await h.value("qeSettingsApply()");
            const a = {open_count: h.elements.qeSettingsModal.classList.contains('visible'), mode: h.value("qeModeState.low_vram"),
                       error: h.toasts.error};
            h.setAnswer('/api/queue/editor/settings', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 5,
                skipped_processing: 0, roots: [], sample: [], patch: {low_vram: true}, generation: 2,
                hint: 'low_vram=false cannot disable a globally enabled VRAM-saving mode', ignored_keys: ['foo']}));
            await h.value("qeSettingsApply()");
            const b = {open_count: h.elements.qeSettingsModal.classList.contains('visible'), success: h.toasts.success,
                       texts: h.toasts.texts.filter(t => /Hint|Ignored/.test(t))};
            output({a, b});
        """
        )
        assert r["a"] == {"open_count": True, "mode": "setTo", "error": 1}
        assert r["b"]["open_count"] is False and r["b"]["success"] == 1
        # Instant path (5 ≤ 20): hint and ignored_keys still reach the user
        assert any("Hint: low_vram=false" in t for t in r["b"]["texts"])
        assert any("Ignored: foo" in t for t in r["b"]["texts"])

    def test_the_dialog_stays_open_when_the_real_run_fails(self):
        r = js(
            ITEMS
            + """
            h.domLoaded();
            h.sandbox.collectValues = {low_vram: true};
            const orig = h.sandbox.fetch;
            h.sandbox.fetch = async (url, opts) => {
                const body = opts && opts.body ? JSON.parse(opts.body) : {};
                if (!url.includes('/settings')) return orig(url, opts);
                h.requests.push({url, method: 'POST', body});
                if (body.dry_run) return {ok: true, status: 200, json: async () => ({ok: true, dry_run: true, affected: 3, skipped_processing: 0, roots: [], sample: [], generation: 1})};
                return {ok: false, status: 500, json: async () => ({error: 'database is locked'})};
            };
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qeRowClick(3, {shiftKey: true}); qeSettingsOpen(); qeModeState.low_vram = 'setTo'");
            await h.value("qeSettingsApply()");
            output({open_count: h.elements.qeSettingsModal.classList.contains('visible'), mode: h.value("qeModeState.low_vram"),
                     error: h.toasts.error, pending: h.value("qePending"), running_flag: h.value("qeRunning")});
        """
        )
        assert r == {
            "open_count": True,
            "mode": "setTo",
            "error": 1,
            "pending": None,
            "running_flag": False,
        }

    def test_a_missing_mode_bar_speaks_up(self):
        r = js("""
            h.elements.qeWordTimestamps = null;             // anchor gone (markup changed)
            const warnings = [];
            h.sandbox.console = {warn: (...a) => warnings.push(a.join(' ')), log() {}, error() {}};
            const missing = h.value("qeBuildModeBars()");
            output({missing, warnings});
        """)
        assert r["missing"] == ["word_timestamps"]
        assert len(r["warnings"]) == 1 and "word_timestamps" in r["warnings"][0]

    def test_a_split_wrapper_speaks_up_too(self):
        r = js("""
            // Voice-vectors checkbox slips into the speaker .field (markup drift)
            const sv = h.elements.qeSpeakerVectors;
            h.elements['field:qeSpeakerVectors'].children = [];
            h.elements['field:qeDiarize'].appendChild(sv);
            const warnings = [];
            h.sandbox.console = {warn: (...a) => warnings.push(a.join(' ')), log() {}, error() {}};
            const missing = h.value("qeBuildModeBars()");
            output({missing, warnings: warnings.length});
        """)
        assert r["missing"] == ["speaker_vectors (shell belongs to speaker)"] and r["warnings"] == 1

    def test_zero_hits_writes_nothing(self):
        """Marked items that have since been processed: warning instead of "changed — 0 entries"."""
        r = js(
            ITEMS
            + """
            h.setAnswer('/api/queue/editor/settings', (url, body) => ({ok: true, dry_run: !!body.dry_run, affected: 0,
                skipped_processing: 0, roots: [], sample: [], requested: 3, generation: 2}));
            h.run("openQueueEditor()"); await h.runTimers();
            h.run("qeRowClick(1, {}); qeRowClick(3, {shiftKey: true})");
            await h.value("qeAction('settings', {settings: {low_vram: true}})");
            output({posts: h.requests.filter(a => a.method === 'POST').map(a => a.body.dry_run), warn: h.toasts.warn,
                     success: h.toasts.success, selected: h.value("qeSelection.size")});
        """
        )
        assert r == {"posts": [True], "warn": 1, "success": 0, "selected": 0}

    def test_applying_without_a_change_warns(self):
        r = js("""
            h.domLoaded();
            await h.value("qeSettingsApply()");
            output({warn: h.toasts.warn, posts: h.requests.length});
        """)
        assert r == {"warn": 1, "posts": 0}


@node
class TestKeyboard:
    def test_esc_closes_from_the_inside_out_and_spares_the_sidebar(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            h.elements.qeSettingsModal.classList.add('visible');
            h.elements.qeConfirmation.style.display = '';
            const ev = () => ({key: 'Escape', n: 0, stopImmediatePropagation() { this.n++; }});
            const e1 = ev(); h.press(e1);
            const a = {settings: h.elements.qeSettingsModal.classList.contains('visible'), stripe: h.elements.qeConfirmation.style.display, stop: e1.n};
            const e2 = ev(); h.press(e2);
            const b = {stripe: h.elements.qeConfirmation.style.display, editor: h.elements.queueEditorModal.classList.contains('visible')};
            const e3 = ev(); h.press(e3);
            output({a, b, c: {editor: h.elements.queueEditorModal.classList.contains('visible')}, clearSelection: h.counter.clearSelection});
        """
        )
        assert r["a"] == {"settings": False, "stripe": "", "stop": 1}
        assert r["b"] == {"stripe": "none", "editor": True}
        assert r["c"]["editor"] is False and r["clearSelection"] == 0

    def test_ctrl_a_selects_all_hits_but_not_inside_an_input(self):
        r = js(
            ITEMS
            + """
            h.run("openQueueEditor()"); await h.runTimers();
            const ev = {key: 'a', ctrlKey: true, p: 0, s: 0, preventDefault() { this.p++; }, stopImmediatePropagation() { this.s++; }};
            h.press(ev);
            const a = {all_items: h.value("qeAll"), n: h.value("qeSelectedCount()"), p: ev.p, s: ev.s};
            h.run("qeClearSelection()");
            h.sandbox.document.activeElement = {tagName: 'INPUT'};
            h.press({key: 'a', ctrlKey: true, preventDefault() {}, stopImmediatePropagation() {}});
            output({a, b: h.value("qeAll")});
        """
        )
        assert r["a"] == {"all_items": True, "n": 5, "p": 1, "s": 1} and r["b"] is False

    def test_closed_editor_ignores_keys(self):
        r = js("""
            const ev = {key: 'Escape', n: 0, stopImmediatePropagation() { this.n++; }};
            h.press(ev);
            output({n: ev.n});
        """)
        assert r["n"] == 0


class TestWiringUp:
    """Source contracts: markup, script order, prefix, hooks."""

    def test_markup_and_script(self):
        html = INDEX.read_text(encoding="utf-8")
        for i in (
            "queueEditorModal",
            "qeSettingsModal",
            "qeTxOptions",
            "qeBtnLowVram",
            "qeCountInput",
            "qeFolder",
        ):
            assert f'id="{i}"' in html, i
        tag = '<script src="/static/js/queue_editor.js?v={{ v }}"></script>'
        assert html.index(tag) > html.index('<script src="/static/js/batch.js?v={{ v }}"></script>')
        assert html.index(tag) < html.index('<script src="/static/js/app.js?v={{ v }}"></script>')
        assert html.index('id="queueEditorModal"') < html.index('id="qeSettingsModal"')

    def test_prefix_qe_registered(self):
        tx = (REPO / "whisper_tm/static/js/txoptions.js").read_text(encoding="utf-8")
        assert "['opt', 'up', 'batch', 'qe', 'qa']" in tx and "renderTxOptions('qe')" in tx
        # 'qa': the "Add folder" tab's options of the new entries
        assert "renderTxOptions('qa')" in tx
        app = (REPO / "whisper_tm/static/js/app.js").read_text(encoding="utf-8")
        assert "TX_OPTION_PREFIXES.map(p => p + 'VocalModel')" in app
        assert (
            "for (const prefix of TX_OPTION_PREFIXES) {\n            txPreset(prefix, cfg);" in app
        )
        css = CSS.read_text(encoding="utf-8")
        assert ".txo-raster > #qeTxOnly" in css and ".qe-modal" in css

    def test_hooks_in_filelist_and_batch(self):
        fl = FILELIST.read_text(encoding="utf-8")
        # The Edit button in the queue header (built by lhBtn, id passed in)
        assert "'btnQueueEditor'" in fl and "openQueueEditor()" in fl
        assert "function queueBadges(" in fl and "queueBadges(item.settings || {})" in fl
        assert "more.onclick = () => openQueueEditor()" in fl
        b = (REPO / "whisper_tm/static/js/batch.js").read_text(encoding="utf-8")
        assert "qeGenerationSeen(d.generation)" in b
        # app.js' dependency "voice vectors need speaker" does not apply in the editor:
        # there the mode bar controls disabled, syncSpeakerVectors('qe') must bail out.
        a = (REPO / "whisper_tm/static/js/app.js").read_text(encoding="utf-8")
        sync = a[a.index("function syncSpeakerVectors(") :]
        sync = sync[: sync.index("box.disabled = !diarize")]
        assert "box.closest('.qe-group')) return;" in sync
        # core.js attaches the HTTP status to the error — that's how the editor recognizes a 404
        c = (REPO / "whisper_tm/static/js/core.js").read_text(encoding="utf-8")
        assert "err.status = resp.status;" in c
        # filelist.js: the JSDoc again sits above _queueRows, not above queueBadges
        assert "folder small. */\nfunction _queueRows(" in fl

    def test_no_native_confirm_and_a_clean_file(self):
        q = JS.read_text(encoding="utf-8")
        assert "confirm(" not in q.replace("qeConfirm", "")
        assert q.count("`") % 2 == 0
        assert " " not in q and " " not in q
        # qeTarget is the only place that builds {ids:/all:/filter:
        body = q[q.index("function qeTarget()") :]
        body = body[: body.index("\n}\n") + 3]
        remainder = q.replace(body, "")
        assert "{all: true}" not in remainder and "{ids: [" not in remainder
        assert "/api/queue/retry'" not in q  # retry goes through /editor/retry

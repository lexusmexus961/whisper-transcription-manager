"""Stale answers and stuck polls in the smaller views.

Each view below fetched without a guard: an older, slower answer could land
after a newer one and overwrite it (performance range, voice history
filter, extract page, settings folder browser, path picker), a superseded
statistics load reported "Timed out", the health view stacked a request
every 5 s behind a hanging one, and one failed extract poll froze an
export's progress for good. The checks run the real static files in node
(tests/js/uifix_misc_harness.js) and hand the answers out in reverse order.
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
HARNESS = REPO / "tests" / "js" / "uifix_misc_harness.js"
SETTINGS_HARNESS = REPO / "tests" / "js" / "settings_harness.js"
PATHPICKER_HARNESS = REPO / "tests" / "js" / "pathpicker_harness.js"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def run_node(source: str, env: dict | None = None) -> dict:
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


def js(files: list[str], body: str) -> dict:
    paths = json.dumps([str(JS / f) for f in files])
    return run_node(
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt({paths});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )


@node
class TestPerformanceRange:
    def test_a_slow_older_range_does_not_overwrite_the_newer_one(self):
        r = js(
            ["perfview.js"],
            """
            h.run('pvRender = () => { globalThis.drawn = (globalThis.drawn || []).concat([pv.data.tag]); }');
            h.run('pv.range = 604800; pvLoad(); pv.range = 3600; pvLoad();');
            await h.answer(1, {tag: '1h'});
            await h.answer(0, {tag: '7d'});
            output({data: h.run('pv.data.tag'), drawn: h.run('globalThis.drawn'),
                    loading: h.get('perfBody').classList.contains('loading')});
        """,
        )
        assert r == {"data": "1h", "drawn": ["1h"], "loading": False}


@node
class TestHealthRefresh:
    def test_no_second_request_while_one_is_in_flight(self):
        r = js(
            ["health.js"],
            """
            h.get('healthOverlay').classList.add('visible');
            h.run('_hcRefresh(); _hcRefresh(); _hcRefresh();');
            const during = h.requests.length;
            await h.fail(0);
            h.run('_hcRefresh()');
            output({during, after: h.requests.length,
                    shown: h.get('healthBody').innerHTML.includes('unreachable')});
        """,
        )
        assert r == {"during": 1, "after": 2, "shown": True}

    def test_the_request_has_a_timeout(self):
        src = (JS / "health.js").read_text(encoding="utf-8")
        assert "AbortSignal.timeout" in src

    def test_a_cut_line_can_be_read_and_sentences_wrap(self):
        """The cards cut every line at their edge: why the language model
        server won't start and how to start it could not be read."""
        r = js(
            ["health.js"],
            """
            h.sandbox.lines = ['8 threads · <b>x</b>', '<span class="hc-red">⚠ No model &quot;a&quot;</span>'];
            output({html: h.run('_hcCard("warn", "LLM", "stopped", lines)')});
        """,
        )
        assert '<div title="8 threads · x">' in r["html"]
        assert '<div title="⚠ No model &quot;a&quot;">' in r["html"]
        css = (REPO / "whisper_tm" / "static" / "css" / "app.css").read_text(encoding="utf-8")
        assert ".hc-card-lines > div:has(> .hc-red, > .hc-hint) { white-space:normal; }" in css
        assert 'class="hc-hint"' in (JS / "health.js").read_text(encoding="utf-8")


@node
class TestStatisticsSupersede:
    def test_a_superseded_load_says_nothing_and_keeps_the_dim(self):
        r = js(
            ["statsview.js"],
            """
            h.run('stvHead = () => {}; stvRender = d => "R:" + d.tag; stvDrawHistory = () => {};'
                  + 'stvWireHistory = () => {}; stvWire = () => {}; stvTextHtml = () => "";');
            h.run('stvLoad("/a"); stvLoad("/b");');
            await h.settle(); await h.settle();
            const body = h.get('statsBody');
            const afterSupersede = {timedOut: body.innerHTML.includes('Timed out'),
                                    loading: body.classList.contains('loading')};
            await h.answer(3, {tag: 'b'});
            output({afterSupersede, html: body.innerHTML, loading: body.classList.contains('loading')});
        """,
        )
        assert r["afterSupersede"] == {"timedOut": False, "loading": True}
        assert r["html"] == "R:b"
        assert r["loading"] is False

    def test_its_own_timeout_still_reports(self):
        r = js(
            ["statsview.js"],
            """
            h.run('stvHead = () => {}; stvWire = () => {};');
            h.run('stvLoad("/a")');
            h.timers.find(t => t.ms === h.run('STV_TIMEOUT')).fn();   // the load's own clock
            await h.settle(); await h.settle();
            output({timedOut: h.get('statsBody').innerHTML.includes('Timed out')});
        """,
        )
        assert r == {"timedOut": True}


@node
class TestVoiceHistoryFilter:
    def test_a_newer_filter_replaces_the_running_load(self):
        r = js(
            ["voices_history.js"],
            """
            h.run('historyDraw = () => {}');
            h.run('HISTORY.q = "Al"; historyLoad(true); HISTORY.q = "Alice"; historyLoad(true);');
            const asked = h.requests.map(x => x.url);
            await h.answer(1, {entries: [{id: 2, name: 'Alice'}]});
            await h.answer(0, {entries: [{id: 1, name: 'Al'}]});
            output({asked, entries: h.run('HISTORY.entries.map(e => e.name)'),
                    loading: h.run('HISTORY.loading')});
        """,
        )
        assert len(r["asked"]) == 2 and r["asked"][1].endswith("q=Alice")
        assert r["entries"] == ["Alice"]
        assert r["loading"] is False

    def test_load_more_still_waits_for_the_running_load(self):
        r = js(
            ["voices_history.js"],
            """
            h.run('historyDraw = () => {}');
            h.run('HISTORY.entries = [{id: 9}]; historyLoad(false); historyLoad(false);');
            output({n: h.requests.length});
        """,
        )
        assert r == {"n": 1}


@node
class TestExtract:
    def test_an_older_page_does_not_overwrite_the_newer_one(self):
        r = js(
            ["voices_extract.js"],
            """
            h.run('extractDrawPage = () => {}; extractButtons = () => {}; EXTRACT.run = {run_id: "r"};');
            h.run('EXTRACT.offset = 0; extractLoadPage(); EXTRACT.offset = 50; extractLoadPage();');
            await h.answer(1, {clips: ['page2'], total: 9});
            await h.answer(0, {clips: ['page1'], total: 9});
            output({page: h.run('EXTRACT.page')});
        """,
        )
        assert r == {"page": ["page2"]}

    def test_a_failed_poll_tries_again_while_the_export_runs(self):
        r = js(
            ["voices_extract.js"],
            """
            h.get('mapOverlay').classList.add('visible');
            h.run('extractDrawJob = () => {}; extractButtons = () => {};'
                  + 'EXTRACT.status = {running: true, run_id: "r"};');
            h.run('extractPoll()');
            await h.fail(0);
            output({retries: h.timers.map(t => t.ms)});
        """,
        )
        assert len(r["retries"]) == 1 and r["retries"][0] > 1000

    def test_a_failed_poll_of_a_finished_export_stays_quiet(self):
        r = js(
            ["voices_extract.js"],
            """
            h.get('mapOverlay').classList.add('visible');
            h.run('EXTRACT.status = {running: false};');
            h.run('extractPoll()');
            await h.fail(0);
            output({retries: h.timers.length});
        """,
        )
        assert r == {"retries": 0}


@node
class TestSettingsFolderBrowser:
    def test_the_last_clicked_folder_wins(self):
        source = (
            f"const {{buildIt}} = require({str(SETTINGS_HARNESS)!r});\n"
            "const h = buildIt();\n"
            "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
            "const pending = [];\n"
            "h.sandbox.fetch = (url) => new Promise(res => pending.push({url, res}));\n"
            "const settle = () => new Promise(r => setImmediate(r));\n"
            "(async () => {\n"
            "  const box = {innerHTML: ''}; const input = {value: ''};\n"
            "  h.sandbox.box = box; h.sandbox.input = input;\n"
            "  h.run(\"_sBrowseTo(box, input, '/m/a'); _sBrowseTo(box, input, '/m/b');\");\n"
            "  const answer = (i, path) => pending[i].res({ok: true, status: 200,\n"
            "      json: async () => ({path, dirs: []})});\n"
            "  answer(1, '/m/b'); await settle(); await settle();\n"
            "  answer(0, '/m/a'); await settle(); await settle();\n"
            "  output({path: box._path, shown: box.innerHTML.includes('/m/b')});\n"
            "})().catch(e => { console.error(e); process.exit(1); });\n"
        )
        r = run_node(source, {"SETTINGS_JS": str(JS / "settings.js")})
        assert r == {"path": "/m/b", "shown": True}


@node
class TestPathPickerLateAnswer:
    def test_a_navigation_answered_after_close_leaves_the_field_alone(self):
        source = (
            f"const {{buildIt}} = require({str(PATHPICKER_HARNESS)!r});\n"
            "const h = buildIt({answers: {}});\n"
            "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
            "(async () => {\n"
            "  h.elements.dirInput.value = '/mnt/a';\n"
            "  h.run(\"openPathPicker('main')\");\n"
            "  const nav = h.run(\"ppNavigate('/mnt/a')\");\n"
            "  h.elements.dirInput.value = '/mnt/a/b';\n"
            "  h.run('closePathPicker()');\n"
            "  await nav;\n"
            "  output({field: h.elements.dirInput.value});\n"
            "})().catch(e => { console.error(e); process.exit(1); });\n"
        )
        r = run_node(source, {"PATHPICKER_JS": str(JS / "pathpicker.js")})
        assert r == {"field": "/mnt/a/b"}


class TestOnclickStrings:
    """Values inside a JS string in an onclick go through escJs, not esc:
    esc's &#39; turns back into ' when the attribute is decoded."""

    @pytest.mark.parametrize(
        "name, call",
        [
            ("settings_tools.js", "_artApply"),
            ("settings_tools.js", "_artUndo"),
            ("settings.js", "_settingsJump"),
        ],
    )
    def test_escjs_in_the_handler(self, name, call):
        src = (JS / name).read_text(encoding="utf-8")
        uses = re.findall(re.escape(call) + r"\('\$\{(\w+)\(", src)
        assert uses and set(uses) == {"escJs"}, (call, uses)


@node
class TestStatisticsCalendar:
    def test_the_partial_first_month_gives_way_to_the_next(self):
        """The calendar starts on a Monday late in a month: that month's
        label sat two weeks before the next one and pushed it out — "Apr"
        stood where "May" belonged."""
        r = js(
            ["statsview.js"],
            """
            const RealDate = Date;
            h.sandbox.Date = class extends RealDate {
                constructor(...a) { super(...(a.length ? a : ['2026-10-22T10:00:00'])); }
            };
            h.run('fmtNumber = n => String(n); fmtDuration = s => Math.round(s) + "s"');
            h.run("stvState.data = {history: [{tag: '2026-10-22', cnt: 1, duration: 25, words: 17}]}");
            const box = h.get('stvCalendar');
            box.querySelector = () => ({scrollWidth: 0});
            h.run('stvDrawCalendar()');
            output({months: [...box.innerHTML.matchAll(/<text class="stv-axis" x="[0-9.]+" y="11">([^<]+)</g)]
                            .map(m => m[1])});
        """,
        )
        assert r["months"] == ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]

    def test_no_rate_below_a_minute_of_audio(self):
        """The server sends no speaking rate (0) for less than a minute of
        audio — the card said "0 per minute of speech" under 17 words."""
        src = (JS / "statsview.js").read_text(encoding="utf-8")
        assert "t.wpm ? `${fmtNumber(t.wpm)} per minute of speech` : ''" in src
        src = (JS / "stats.js").read_text(encoding="utf-8")
        assert "wpm ? `${wpm.toFixed(0)} per minute of speech` : ''" in src

    def test_the_hour_axis_and_the_month_labels(self):
        """Audio hours: the middle tick of one hour read "1 h" (rounded), and
        over a year the first day's "Oct 23" ran into "Nov" right after it."""
        r = js(
            ["statsview.js"],
            """
            const RealDate = Date;
            h.sandbox.Date = class extends RealDate {
                constructor(...a) { super(...(a.length ? a : ['2026-10-22T10:00:00'])); }
            };
            h.run('fmtNumber = n => String(n); fmtDuration = s => Math.round(s) + "s"');
            h.run("stvState.data = {history: [{tag: '2025-10-01', cnt: 0, duration: 0, words: 0},"
                  + "{tag: '2026-10-22', cnt: 1, duration: 25, words: 17}]}");
            h.run("stvState.metric = 'duration'; stvState.range = 365");
            h.run('stvDrawHistChart()');
            const svg = h.get('stvHistChart').innerHTML;
            output({axis: [...svg.matchAll(/text-anchor="end">([^<]+)</g)].map(m => m[1]),
                    months: [...svg.matchAll(/<text class="stv-axis" x="[0-9.]+" y="184">([^<]+)</g)].map(m => m[1]),
                    tip: svg.includes('25s of audio')});
        """,
        )
        assert r["axis"] == ["0 h", "0.5 h", "1 h"]
        assert r["months"][:2] == ["Nov", "Dec"] and r["months"][-1] == "Oct"
        assert r["tip"]


@node
class TestLibraryCriteriaCounts:
    def test_a_groups_number_follows_a_change(self):
        """A group's number was drawn with the criteria only: after a preset
        "Voices 3" stayed while the voices criteria were set back, and a
        path typed in showed no number on "Where"."""
        r = js(
            ["library_select.js"],
            """
            const group = t => String(h.run(`LS_GROUPS.findIndex(g => g.title === '${t}')`));
            const set = (t, el) => {
                const summary = {innerHTML: 'stale'};
                el.closest = () => ({dataset: {group: group(t)}, querySelector: () => summary});
                h.sandbox.el = el;
                h.run('lsSet(el)');
                return summary.innerHTML;
            };
            h.run('lsState.crit = {voices: false, voices_run: false}');
            const voices = set('Voices', {dataset: {key: 'voices'}, tagName: 'SELECT', value: 'undefined'});
            const where = set('Where', {dataset: {key: 'name'}, tagName: 'INPUT', type: 'text', value: 'tiny'});
            output({voices, where});
        """,
        )
        assert r == {
            "voices": 'Voices <span class="llm-chip">1</span>',
            "where": 'Where <span class="llm-chip">1</span>',
        }

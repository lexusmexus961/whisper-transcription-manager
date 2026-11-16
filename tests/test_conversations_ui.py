"""The search view's tab "Conversations" (conversations.js) and what hands
over to it — Ask's card, Find, Voices → People, the Conversation card of a
transcript's 📊 — run in node against fake answers of the HTTP API.

What these tests pin:

* rows show the server's numbers and escape every name and path; the
  handlers carry a row's number, never a name;
* an answer to filters changed since never lands (out of order); "Show
  more" pages on from where the server's last page ended, never asking for
  the same page twice;
* cvShow takes the API's parameters, drops what it doesn't know and opens
  the tab; cvQuery repeats the people and leaves "any" out;
* the person card, the typed-name check, the ⚙ panel and its chips, the
  half second a typed filter waits, Esc closing the panel first;
* switched off, nothing found, an empty library, the background count in
  the info line;
* the hand-offs: Ask's button, Find's status, Voices → People, the 📊 card
  — and that the tab asks the API what Ask's tool and Find's plan asked
  (mcp_server.conversation_query, find/plan.conversation_filters);
* the markup and the search view's hooks.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html import unescape
from pathlib import Path
from urllib.parse import parse_qsl

import pytest

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "whisper_tm" / "static"
JS = STATIC / "js"
HARNESS = Path(__file__).resolve().parent / "js" / "conversations_harness.js"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def ui(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch((e) => { console.error(e); process.exit(1); });\n"
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


EVIL = "<img src=x onerror=alert(1)>"


def row(**kw) -> dict:
    """One conversation as GET /api/conversations hands it out."""
    base = {
        "abs_path": "/lib/talks/evening.mkv",
        "filename": "talks/evening.mkv",
        "file_dir": "/lib",
        "start_sec": 4500.0,
        "end_sec": 8280.0,
        "duration_sec": 3780.0,
        "whole": False,
        "recording_sec": 21600.0,
        "language": "en",
        "date": "2024-03-01",
        "date_source": "name",
        "speech_sec": 3400.0,
        "words": 4100,
        "sentences": 360,
        "questions": 41,
        "wps": 11.4,
        "wps_pct": 78,
        "turn_rate": 3.1,
        "rate_pct": 30,
        "mean_turn_sec": 19.4,
        "turn_pct": 64,
        "questions_share": 0.11,
        "longest_turn_sec": 140.0,
        "longest_turn_start": 5100.0,
        "longest_turn_name": "Alice",
        "speakers": 3,
        "unnamed": 1,
        "people": [
            {"name": "Alice", "talk_sec": 1200.0, "share": 0.38, "wps": 12.1},
            {"name": "Bob", "talk_sec": 900.0, "share": 0.29, "wps": 10.2},
        ],
        "kind": "detailed",
        "group": True,
        "interview": False,
        "reasons": [
            "Sentences of 11.4 words — longer than 78 % of conversations in this language."
        ],
        "copies": 1,
    }
    base.update(kw)
    return base


def answer(results, total=None, **kw) -> dict:
    d = {
        "total": len(results) if total is None else total,
        "offset": 0,
        "limit": 50,
        "truncated": False,
        "media_dir": "/lib",
        "results": results,
        "baseline": {
            "language": "en",
            "source": "language",
            "conversations": 120,
            "wps": {"p40": 8.1, "p50": 9.0, "p60": 9.9},
        },
        "people_unknown": [],
        "ignored": [],
        "status": {"enabled": True, "counted": 40, "stale": 0, "running": False},
    }
    d.update(kw)
    return d


IDLE = {
    "enabled": True,
    "settings": {"enabled": True, "auto_run": True, "min_share": 0.05},
    "counts": {"transcripts": 40, "counted": 40, "stale": 0},
    "run": {"running": False},
}


def setup(find: dict | None = None, status: dict | None = None, people=None) -> str:
    """The answers of an idle library — overridden per test."""
    return (
        f"h.answers['/api/conversations'] = {json.dumps(find or answer([row()]))};\n"
        f"h.answers['/api/conversations/status'] = {json.dumps(status or IDLE)};\n"
        f"h.answers['/api/conversations/people'] = {json.dumps({'people': people or []})};\n"
        "h.answers['/api/conversations/person'] = {name: 'Alice', known: true, recordings: 3, conversations: 2,\n"
        "    talk_sec: 4000, wps: 9.5, questions_share: 0.08, partners: [], kinds: {}};\n"
    )


def handlers(html: str) -> list[str]:
    return re.findall(r'on(?:click|change|input|keydown|focus)="([^"]*)"', html)


def _clicked(html: str) -> str:
    """What the browser runs on a click of the first button: its onclick
    attribute, HTML-decoded."""
    return unescape(re.search(r'onclick="([^"]*)"', html)[1])


# The API's defaults: one side may send them, the other leave them out
_DEFAULTS = {("sort", "longest"), ("copies", "fold")}


def _value(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def _params(query: str) -> dict:
    """A query string as ``{name: [values]}``, numbers as numbers."""
    out: dict = {}
    for k, v in parse_qsl(query, keep_blank_values=True):
        if (k, v) not in _DEFAULTS:
            out.setdefault(k, []).append(_value(v))
    return out


def _params_of(filters: dict) -> dict:
    """API parameters handed over as a dict (cvShow's shape) in the shape of
    :func:`_params` — a 0 is "any", which nobody sends."""
    out: dict = {}
    for k, v in filters.items():
        for x in v if isinstance(v, list) else [v]:
            if (k, str(x)) not in _DEFAULTS and x != 0:
                out.setdefault(k, []).append(_value(str(x)))
    return out


@node
class TestTheList:
    def test_rows_are_escaped_and_open_where_they_are(self):
        evil = row(
            abs_path=f"/lib/{EVIL}/{EVIL}.mkv",
            filename=f"{EVIL}/{EVIL}.mkv",
            longest_turn_name=EVIL,
            people=[{"name": EVIL, "talk_sec": 60, "share": 0.5, "wps": 9}],
            reasons=[EVIL],
            kind=EVIL,
            date=EVIL,
        )
        d = ui(
            setup(answer([evil]))
            + """
        await h.run('cvOpen()'); await h.settle();
        const html = h.elements.cvList.innerHTML;
        h.run('cvOpenRow(0)'); h.run('cvPlayRow(0)'); h.run('cvMonologue(0)');
        output({html, goto: h.calls.goto, play: h.calls.play});
        """
        )
        html = d["html"]
        assert EVIL not in html and "&lt;img src=x onerror=alert(1)&gt;" in html
        # No name or path in a handler: the row's number only
        for code in handlers(html):
            assert re.fullmatch(r"(event\.stopPropagation\(\); )?cv[A-Za-z]+\(\d*\)", code), code
        assert d["goto"] == [
            [f"{EVIL}/{EVIL}.mkv", 4500, "/lib"],
            [f"{EVIL}/{EVIL}.mkv", 5100, "/lib"],
        ]
        assert d["play"] == [[f"{EVIL}/{EVIL}.mkv", 4500, "/lib"]]

    def test_a_row_shows_the_servers_numbers(self):
        rows = [
            row(),
            row(
                abs_path="/lib/b.mp3",
                filename="b.mp3",
                whole=True,
                recording_sec=1500,
                duration_sec=1500,
                wps=None,
                wps_pct=None,
                turn_rate=None,
                rate_pct=None,
                copies=3,
                people=[],
                unnamed=2,
                group=False,
                kind="monologue",
                longest_turn_sec=None,
                longest_turn_start=None,
                questions_share=None,
                date_source="file",
                date="2023-12-24",
            ),
        ]
        d = ui(
            setup(answer(rows))
            + """
        await h.run('cvOpen()'); await h.settle();
        output({html: h.elements.cvList.innerHTML, info: h.elements.cvInfo.innerHTML,
                urls: h.calls.api.map(c => c.url)});
        """
        )
        html = d["html"]
        first, second = html.split('class="cv-row" role="button"')[1:3]
        assert "1:15:00–2:18:00" in first and "of 6:00:00" in first and ">1h 03m<" in first
        assert "<b>11.4</b> words" in first and "longer than 78 %" in first
        assert "<b>3.1</b> /min" in first and "fewer than 70 %" in first and ">11 %<" in first
        assert (
            ">2m 20s</button>" in first and "Alice · 1:25:00" in first and ">2024-03-01<" in first
        )
        assert "Alice <b>38 %</b>" in first and "Bob <b>29 %</b>" in first and "+1 unnamed" in first
        assert "width:38.0%" in first and "width:29.0%" in first  # the stacked share bar
        assert ">Detailed<" in first and ">Group<" in first
        assert ">evening.mkv<" in first and ">/lib/talks<" in first  # the name, its own folder
        # A whole recording, unpunctuated, nobody named, kept in three files
        assert "the whole recording" in second and ">25:00<" in second and "+2 copies" in second
        assert "+2 unnamed" in second and "has no sentence punctuation" in second
        assert ">Monologue<" in second and "The file’s date" in second
        # Nothing counted for it is a dash, not a zero — and without sentence
        # punctuation no question can be told apart either
        assert ">0 %<" not in second
        assert re.search(
            r'cv-q"><small class="cv-lab">Questions</small><span class="cv-dim" data-tip="No share of '
            r'questions\n[^"]*">—<',
            second,
        )
        assert (
            d["urls"][0]
            == "/api/conversations?speakers_min=2&sort=longest&copies=fold&offset=0&limit=50"
        )
        assert "<b>2</b> conversations · 1.5 h" in d["info"]  # all loaded: added up
        assert "Compared with 120 conversations in EN" in d["info"]

    def test_an_answer_to_older_filters_never_lands(self):
        d = ui(
            setup()
            + f"""
        let release;
        const gate = new Promise(r => {{ release = r; }});
        h.answers['/api/conversations'] = async (url) => url.includes('kind=lively')
            ? {json.dumps(answer([row(filename="new.mkv", abs_path="/lib/new.mkv")]))}
            : (await gate, {json.dumps(answer([row(filename="old.mkv", abs_path="/lib/old.mkv")], total=9))});
        h.run('cvOpen()'); await h.settle();                        // the first answer is held back
        h.run("cvSetKind('lively')"); await h.settle();             // the second one comes at once
        release(); await h.settle();                                // the first one, late
        output({{html: h.elements.cvList.innerHTML, total: h.value('_cv.list.total'), info: h.elements.cvInfo.innerHTML}});
        """
        )
        assert "new.mkv" in d["html"] and "old.mkv" not in d["html"]
        assert d["total"] == 1 and "<b>1</b> conversation" in d["info"]

    def test_show_more_adds_the_next_page(self):
        page = [row(abs_path=f"/lib/{i}.mkv", filename=f"{i}.mkv") for i in range(50)]
        d = ui(
            setup()
            + f"""
        const page = {json.dumps(page)};
        // The order moved between the pages (the background count): the
        // second one starts with a row the first one had
        h.answers['/api/conversations'] = (url) => url.includes('offset=0')
            ? {{...{json.dumps(answer([], total=120))}, results: page}}
            : {json.dumps(answer([page[49], row(abs_path="/lib/99.mkv", filename="99.mkv")], total=120))};
        await h.run('cvOpen()'); await h.settle();
        const before = h.elements.cvList.innerHTML;
        h.run('cvMore()'); await h.settle();
        output({{before, n: h.value('_cv.list.results.length'), urls: h.calls.api.map(c => c.url),
                 info: h.elements.cvInfo.innerHTML, after: h.elements.cvList.innerHTML}});
        """
        )
        assert "Show more (50 of 120)" in d["before"] and "cvMore()" in d["before"]
        assert (
            "/api/conversations?speakers_min=2&sort=longest&copies=fold&offset=50&limit=50"
            in d["urls"]
        )
        assert d["n"] == 51  # the repeated one left out
        assert "Show more (51 of 120)" in d["after"]
        # Not all loaded: no hours (a page's sum is no total)
        assert '<b>120</b> conversations · <span class="cv-base"' in d["info"]

    def test_show_more_never_asks_for_the_same_page_twice(self):
        # Every row of the next page was shown already (the order moved): the
        # one after it is asked for, not the same one again — and a list read
        # to its end has no button
        page = [row(abs_path=f"/lib/{i}.mkv", filename=f"{i}.mkv") for i in range(50)]
        d = ui(
            setup()
            + f"""
        const page = {json.dumps(page)};
        h.answers['/api/conversations'] = (url) => url.includes('offset=0') || url.includes('offset=50')
            ? {{...{json.dumps(answer([], total=120))}, results: page}}
            : {json.dumps(answer([row(abs_path="/lib/last.mkv", filename="last.mkv")], total=120))};
        await h.run('cvOpen()'); await h.settle();
        h.run('cvMore()'); await h.settle();
        const stuck = h.elements.cvList.innerHTML;
        h.run('cvMore()'); await h.settle();
        output({{stuck, end: h.elements.cvList.innerHTML, info: h.elements.cvInfo.innerHTML,
                 offsets: h.calls.api.filter(c => c.url.startsWith('/api/conversations?'))
                     .map(c => c.url.match(/offset=(\\d+)/)[1])}});
        """
        )
        assert d["offsets"] == ["0", "50", "100"]
        assert "Show more (50 of 120)" in d["stuck"]
        assert "Show more" not in d["end"] and "last.mkv" in d["end"]
        # Every page read: what was shown is added up
        assert re.search(r"<b>120</b> conversations · \d+(\.\d)? h", d["info"])

    def test_the_servers_total_length(self):
        # total_sec covers every result, not the page shown: a paged list
        # says its hours too
        d = ui(
            setup(answer([row()], total=120, total_sec=72000))
            + """
        await h.run('cvOpen()'); await h.settle();
        output(h.elements.cvInfo.innerHTML);
        """
        )
        assert "<b>120</b> conversations · 20.0 h" in d

    def test_the_info_line_drops_the_old_count_while_loading(self):
        d = ui(
            setup(answer([row()], total=37))
            + f"""
        let release;
        const gate = new Promise(r => {{ release = r; }});
        await h.run('cvOpen()'); await h.settle();
        const before = h.elements.cvInfo.innerHTML;
        h.answers['/api/conversations'] = async () => (await gate, {json.dumps(answer([], total=0))});
        h.run("cvSetKind('lively')"); await h.settle();
        const loading = h.elements.cvInfo.innerHTML;
        release(); await h.settle();
        output({{before, loading, after: h.elements.cvInfo.innerHTML}});
        """
        )
        assert "<b>37</b> conversations" in d["before"]
        assert "<b>37</b>" not in d["loading"] and "Looking for conversations" in d["loading"]
        assert "<b>0</b> conversations" in d["after"]

    def test_nothing_found(self):
        d = ui(
            setup(answer([]))
            + """
        h.run("cvShow({people: ['Alice', 'Bob'], min_sec: 2700, max_sec: 4500})");
        await h.run('cvOpen()'); await h.settle();
        output(h.elements.cvList.innerHTML);
        """
        )
        assert "No conversation fits these filters" in d
        assert (
            "More turn up with “Any of them” instead of “All together”, fewer people or another length."
            in d
        )
        assert "cvResetAll()" in d

    def test_switched_off(self):
        d = ui(
            setup()
            + """
        const off = new Error('Conversation statistics are switched off (conversations.enabled).');
        off.status = 503;
        h.answers['/api/conversations'] = off;
        h.answers['/api/conversations/status'] = {enabled: false, counts: {}, run: {running: false}};
        await h.run('cvOpen()'); await h.settle();
        output({html: h.elements.cvList.innerHTML, info: h.elements.cvInfo.innerHTML});
        """
        )
        assert "Conversation statistics are switched off" in d["html"]
        assert "openSettings('search#conversations')" in d["html"]
        assert "switched off" in d["info"] and "Count everything again" not in d["info"]

    def test_a_busy_database_is_not_switched_off(self):
        d = ui(
            setup()
            + """
        const busy = new Error('The database is busy — try again in a moment.');
        busy.status = 503;
        h.answers['/api/conversations'] = busy;
        await h.run('cvOpen()'); await h.settle();
        output({html: h.elements.cvList.innerHTML, off: h.value('_cv.off')});
        """
        )
        assert d["off"] is False and "switched off" not in d["html"]
        assert "The database is busy" in d["html"]

    def test_garbage_answers_never_break_it(self):
        # Answers of another shape (an older or broken server, a proxy's
        # page): drawn without a throw and without "undefined" or "NaN"
        bad_row = {
            "abs_path": 7,
            "filename": None,
            "people": "Alice",
            "reasons": "x",
            "kind": {"a": 1},
            "start_sec": "soon",
            "duration_sec": None,
            "wps": "many",
            "turn_rate": [],
            "copies": "2",
            "longest_turn_sec": "long",
            "longest_turn_start": {},
            "date": 5,
            "unnamed": "two",
        }
        d = ui(f"""
        h.answers['/api/conversations'] = {{total: 'all', results: [null, 3, {json.dumps(bad_row)}],
                                            baseline: 'x', people_unknown: 'Alice', ignored: 4}};
        h.answers['/api/conversations/status'] = {{enabled: true, counts: 'x', run: 'y'}};
        h.answers['/api/conversations/people'] = {{people: 'nobody'}};
        h.answers['/api/conversations/person'] = {{name: 5, known: true, partners: 'x', kinds: [3], talk_sec: 'a'}};
        h.answers['/api/conversations/file'] = {{counted: true, summary: 'x', speakers: 'x', people: [null, 1],
                                                 parts: [null, {{people: 'x'}}], reasons: 'x'}};
        h.run("cvShow({{people: ['Alice']}})");
        await h.run('cvOpen()'); await h.settle();
        await h.run("txsConvLoad('/lib/a.mkv')");
        output({{list: h.elements.cvList.innerHTML, info: h.elements.cvInfo.innerHTML,
                 card: h.elements.cvPerson.innerHTML, stats: h.elements.txsConv.innerHTML}});
        """)
        for part, html in d.items():
            assert html and "undefined" not in html and "NaN" not in html, (part, html[:400])


@node
class TestTheFilters:
    def test_cvshow_takes_the_api_parameters_and_opens_the_tab(self):
        d = ui(
            setup()
            + """
        h.elements.svQuery.value = 'police';
        h.run(`cvShow({people: ['Alice', 'Bob', 'alice'], kind: 'detailed,group,nonsense', min_sec: 2700,
                       sort: 'sentences', wps_min: 'x', rubbish: 1, match: 'any', language: 'DE'})`);
        const hidden = {f: h.value('_cv.f'), tab: h.calls.tab.slice(), open: h.calls.open.slice(),
                        routed: h.sandbox.svState.routed};
        h.elements.svOverlay.classList.add('visible');
        h.sandbox.svState.tab = 'text';
        h.run("cvShow({sort: 'bogus'})");
        output({hidden, f: h.value('_cv.f'), tab: h.calls.tab});
        """
        )
        f = d["hidden"]["f"]
        assert (
            f["people"] == ["Alice", "Bob"]
            and f["kind"] == "detailed,group"
            and f["min_sec"] == 2700
        )
        assert (
            f["sort"] == "sentences"
            and f["wps_min"] == 0
            and f["match"] == "any"
            and "rubbish" not in f
        )
        assert f["language"] == "de"  # as transcripts store it
        assert f["speakers_min"] == 2 and f["copies"] == "fold"  # what was not named starts new
        assert d["hidden"]["tab"] == ["conversations"] and d["hidden"]["open"] == ["police"]
        assert d["hidden"]["routed"] == "police"  # the query isn't routed elsewhere
        assert d["f"]["sort"] == "longest" and d["f"]["people"] == []
        assert d["tab"][-1] == "svTab:conversations"  # open: the tab switch opens it

    def test_cvquery_repeats_the_people_and_leaves_any_out(self):
        d = ui(
            setup()
            + """
        h.run(`cvShow({people: ['Alice', 'Bob'], min_sec: 2700, max_sec: 4500, kind: 'detailed', wps_min: '',
                       language: '', q: 'two words', questions_min: 0.2})`);
        const both = h.value('cvQuery()');
        h.run("cvShow({sort: 'together'})");
        output({both, none: h.value('cvQuery()'), page: h.value('cvQuery({offset: 0, limit: 50})')});
        """
        )
        assert d["both"] == (
            "people=Alice&people=Bob&match=all&min_sec=2700&max_sec=4500&kind=detailed"
            "&speakers_min=2&questions_min=0.2&q=two+words&sort=longest&copies=fold"
        )
        # No people: no match, and "Most together" means nothing — longest
        assert d["none"] == "speakers_min=2&sort=longest&copies=fold"
        assert d["page"].endswith("&offset=0&limit=50")

    def test_a_monologue_lowers_the_speakers_minimum(self):
        d = ui(
            setup()
            + """
        const out = [];
        h.run("cvShow({})");
        h.run("cvSetKind('monologue')"); out.push(h.value('cvQuery()'));
        h.run("cvSetKind('detailed')"); out.push(h.value('cvQuery()'));
        h.run("cvShow({kind: 'monologue'})"); out.push(h.value('_cv.f.speakers_min'));
        h.run("cvShow({kind: 'monologue', speakers_min: 3})"); out.push(h.value('_cv.f.speakers_min'));
        h.run("cvShow({kind: 'detailed,group'})"); h.run("cvSetKind('detailed')"); out.push(h.value('_cv.f.kind'));
        h.run("cvSetKind('group')"); out.push(h.value('_cv.f.kind'));
        output(out);
        """
        )
        assert d[0] == "kind=monologue&sort=longest&copies=fold"  # one voice is a monologue
        assert d[1] == "kind=detailed&speakers_min=2&sort=longest&copies=fold"
        assert d[2] == 0 and d[3] == 3
        assert d[4] == "group" and d[5] == ""  # a click on the chosen kind takes it off

    def test_the_bar(self):
        d = ui(
            setup()
            + f"""
        h.run(`cvShow({{people: ['Alice', {json.dumps(EVIL)}], min_sec: 2700, max_sec: 4500, kind: 'lively,group',
                       sentences: 'longer'}})`);
        h.run('cvRenderFilters()');
        output(h.elements.cvFilters.innerHTML);
        """
        )
        assert EVIL not in d and "&lt;img" in d
        assert "data-names" in d and "cvPersonKey(event)" in d and "cvPersonDrop(1)" in d
        assert (
            "All together</button>" in d and "Any of them</button>" in d
        )  # two people: the choice
        assert re.search(r'class="ms-chip on" data-len="4"[^>]*>About 1 h<', d)
        assert re.search(r"class=\"ms-chip on\" onclick=\"cvSetKind\('lively'\)\"", d)
        assert re.search(r"class=\"ms-chip on\" onclick=\"cvSetKind\('group'\)\"", d)
        assert re.search(r'vx-seg-btn active"\s+onclick="cvSetSentences\(\'longer\'\)"', d)
        assert '<option value="together"' in d  # with people only
        for code in handlers(d):
            assert EVIL not in code and "img" not in code, code

    def test_the_panel_and_its_chips(self):
        d = ui(
            setup()
            + """
        await h.run('cvOpen()'); await h.settle();
        h.run('cvPanel(true)');
        const panel = h.elements.cvFilters.innerHTML;
        for (const [k, v] of [['wps_min', '12'], ['rate_max', '4.5'], ['speakers_min', '3'], ['questions_pct', '20'],
                              ['folder', '/media/talks'], ['min_min', '12'], ['max_min', '40']]) h.run(`cvField('${k}', '${v}')`);
        h.run("cvField('language', 'de', true)"); await h.settle();
        h.run("cvSetCopies(false)"); await h.settle();
        h.run('cvPanel(false)');
        const chips = h.elements.cvFilters.innerHTML;
        const url = h.calls.api.filter(c => c.url.startsWith('/api/conversations?')).pop().url;
        h.run("cvClear('wps')"); await h.settle();
        const after = h.elements.cvFilters.innerHTML;
        h.run('cvResetPanel()'); await h.settle();
        output({panel, chips, url, after, last: h.calls.api.filter(c => c.url.startsWith('/api/conversations?')).pop().url});
        """
        )
        for field in (
            "cvField('min_min'",
            "cvField('wps_min'",
            "cvField('rate_min'",
            "cvField('speakers_min'",
            "cvField('questions_pct'",
            "cvField('language'",
            "cvField('q'",
            "cvField('folder'",
            "cvField('date_from'",
            "cvSetCopies(this.checked)",
            "cvResetPanel()",
            "cvPanel(false)",
        ):
            assert field in d["panel"], field
        assert d["url"] == (
            "/api/conversations?min_sec=720&max_sec=2400&wps_min=12&rate_max=4.5&speakers_min=3"
            "&questions_min=0.2&language=de&folder=%2Fmedia%2Ftalks&sort=longest&copies=all"
            "&offset=0&limit=50"
        )
        chips = d["chips"]
        for chip in (
            "12–40 min",
            "≥ 12 words per sentence",
            "≤ 4.5 speaker changes per min",
            "≥ 3 speakers",
            "≥ 20 % questions",
            "Language DE",
            "In media/talks",
            "Every copy",
        ):
            assert chip in chips, chip
        assert "⚙ Filters <b>8</b>" in chips and "cvClear('wps')" in chips
        assert "≥ 12 words per sentence" not in d["after"] and "⚙ Filters <b>7</b>" in d["after"]
        assert (
            d["last"]
            == "/api/conversations?speakers_min=2&sort=longest&copies=fold&offset=0&limit=50"
        )

    def test_the_speakers_chip_says_what_it_keeps(self):
        d = ui(
            setup()
            + """
        const chip = (f) => { h.run(`cvShow(${JSON.stringify(f)})`); return h.value('cvPanelChips()')
            .filter(([k]) => k === 'speakers').map(([, label]) => label)[0] || ''; };
        output([chip({speakers_min: 0}), chip({speakers_min: 1}), chip({speakers_min: 0, speakers_max: 2}),
                chip({speakers_min: 3, speakers_max: 5}), chip({})]);
        """
        )
        # One speaker at least leaves out the recordings nobody diarized
        assert d == ["Any number of speakers", "≥ 1 speaker", "≤ 2 speakers", "3–5 speakers", ""]

    def test_a_typed_filter_waits_half_a_second(self):
        d = ui(
            setup()
            + """
        await h.run('cvOpen()'); await h.settle();
        const before = h.calls.api.length;
        h.run("cvField('q', 'int')"); h.run("cvField('q', 'interview')");
        const waiting = h.calls.api.length - before, timers = h.timers();
        h.runTimers(); await h.settle();
        const asked = h.calls.api.slice(before).map(c => c.url).filter(u => u.startsWith('/api/conversations?'));
        h.run("cvFieldKey({key: 'Enter', preventDefault() {}, target: {value: 'talk'}}, 'q')"); await h.settle();
        output({waiting, timers, asked, last: h.calls.api.pop().url});
        """
        )
        assert d["waiting"] == 0 and d["timers"] == 1  # the second key replaced the first wait
        assert d["asked"] == [
            "/api/conversations?speakers_min=2&q=interview&sort=longest&copies=fold&offset=0&limit=50"
        ]
        assert "q=talk" in d["last"]

    def test_a_typed_name_is_checked(self):
        d = ui(
            setup(people=[{"name": "Alice", "recordings": 3, "conversations": 2}])
            + """
        await h.run('cvOpen()'); await h.settle();
        // The names are asked for when the field gets the focus, not with
        // every visit of the tab (the heaviest answer of the tab)
        const opened = h.calls.api.filter(c => c.url.startsWith('/api/conversations/people')).length;
        h.run(h.elements.cvFilters.innerHTML.match(/onfocus="([^"]*)"/)[1]); await h.settle();
        h.run("cvAddPerson('alice')"); await h.settle();
        const known = h.value('_cv.f.people.slice()');
        h.run("cvAddPerson('Grace')");
        const hint = {people: h.value('_cv.f.people.slice()'), bar: h.elements.cvFilters.innerHTML};
        h.run("cvAddPerson('grace')"); await h.settle();
        output({opened, known, hint, people: h.value('_cv.f.people.slice()'),
                asked: h.calls.api.filter(c => c.url.startsWith('/api/conversations/people')).length});
        """
        )
        assert d["opened"] == 0 and d["asked"] == 1  # once, on the focus — then kept
        assert d["known"] == ["Alice"]  # the name as it is written
        assert (
            d["hint"]["people"] == ["Alice"]
            and "Nobody named “Grace” takes part" in d["hint"]["bar"]
        )
        assert d["people"] == ["Alice", "grace"]  # Enter again: looked for anyway

    def test_names_the_server_does_not_know(self):
        d = ui(
            setup(answer([], people_unknown=["Grace"]))
            + """
        h.run("cvShow({people: ['Grace', 'Alice']})");
        await h.run('cvOpen()'); await h.settle();
        h.run('cvRenderFilters()');
        output({info: h.elements.cvInfo.innerHTML, bar: h.elements.cvFilters.innerHTML});
        """
        )
        assert "Grace: no name of the voice library, nor typed in a transcript" in d["info"]
        assert 'class="cv-person unknown" data-name="Grace"' in d["bar"]
        assert 'class="cv-person" data-name="Alice"' in d["bar"]

    def test_a_typed_filter_still_waiting_is_dropped_by_a_hand_over(self):
        # Typed into the panel, then Ask's button: the half second runs out
        # after the hand-over, and must not ask for the typed filters again
        d = ui(
            setup()
            + """
        await h.run('cvOpen()'); await h.settle();
        h.run("cvField('q', 'evening')");
        h.run("cvShow({people: ['Alice']})");
        output({timers: h.timers(), q: h.value('_cv.f.q')});
        """
        )
        assert d["timers"] == 0 and d["q"] == ""

    def test_esc_closes_the_panel_first(self):
        d = ui(
            setup()
            + """
        const press = () => { const e = {key: 'Escape', stopped: false, preventDefault() {},
                                          stopImmediatePropagation() { this.stopped = true; }};
                              h.run('cvEscKey')(e); return e.stopped; };
        h.elements.svOverlay.classList.add('visible'); h.elements.svPaneConversations.classList.add('active');
        h.run('cvPanel(true)');
        h.sandbox.__above = true; const settingsAbove = press(); const stillOpen = h.value('_cv.panel');
        h.sandbox.__above = false;
        h.run('var VX_SUGGEST = {el: {hidden: false}, items: [1]}'); const suggesting = press();
        h.run('VX_SUGGEST.items = []');
        const first = press(), closed = !h.value('_cv.panel'), second = press();
        output({settingsAbove, stillOpen, suggesting, first, closed, second});
        """
        )
        assert d["settingsAbove"] is False and d["stillOpen"] is True  # the settings' own Esc
        assert d["suggesting"] is False  # the name list closes first
        assert d["first"] is True and d["closed"] is True  # the panel, not the workspace
        assert d["second"] is False  # then the workspace's turn


@node
class TestThePersonCard:
    def test_one_person_gets_a_card(self):
        person = {
            "name": "Alice",
            "known": True,
            "recordings": 12,
            "conversations": 9,
            "talk_sec": 18000,
            "wps": 9.5,
            "questions_share": 0.08,
            "partners": [
                {"name": EVIL, "recordings": 4, "together_sec": 7200},
                {"name": "Bob", "recordings": 3, "together_sec": 3600},
            ],
            "kinds": {"detailed": 4, "lively": 2, "monologue": 0},
            "first_date": "2021-05-01",
            "last_date": "2024-02-11",
            "longest": [],
        }
        d = ui(
            setup()
            + f"""
        h.answers['/api/conversations/person'] = {json.dumps(person)};
        h.run("cvShow({{people: ['Alice']}})");
        await h.run('cvOpen()'); await h.settle();
        const card = h.elements.cvPerson.innerHTML;
        const url = h.calls.api.find(c => c.url.startsWith('/api/conversations/person')).url;
        h.run('cvPartner(1)'); await h.settle();
        output({{card, url, people: h.value('_cv.f.people.slice()'), after: h.elements.cvPerson.innerHTML,
                 find: h.calls.api.filter(c => c.url.startsWith('/api/conversations?')).pop().url}});
        """
        )
        card = d["card"]
        assert d["url"] == "/api/conversations/person?person=Alice"
        assert EVIL not in card and "&lt;img" in card
        assert (
            "<b>5h 00m</b><span>of talk</span>" in card
            and "<b>12</b><span>recordings</span>" in card
        )
        assert (
            "<b>9</b><span>conversations</span>" in card
            and "<b>9.5</b><span>words per sentence</span>" in card
        )
        assert "cvPartner(0)" in card and "cvPartner(1)" in card and "Talks most with" in card
        assert (
            ">Detailed</span>" in card
            and ">Back-and-forth</span>" in card
            and ">Monologue<" not in card
        )
        assert (
            "width:100.0%" in card and "width:50.0%" in card and "2021-05-01 – 2024-02-11" in card
        )
        assert d["people"] == ["Alice", "Bob"] and d["after"] == ""  # two people: no card
        assert d["find"].startswith("/api/conversations?people=Alice&people=Bob&match=all")

    def test_look_again_asks_for_the_card_too(self):
        # Another filter keeps the card (it is of the person, not of the
        # filters); ↻ asks again — the background count adds to it
        d = ui(
            setup()
            + """
        h.run("cvShow({people: ['Alice']})");
        await h.run('cvOpen()'); await h.settle();
        const cards = () => h.calls.api.filter(c => c.url.startsWith('/api/conversations/person')).length;
        const first = cards();
        h.run("cvSetSentences('longer')"); await h.settle();
        const kept = cards();
        h.run('cvRefresh()'); await h.settle();
        output({first, kept, again: cards(), card: h.elements.cvPerson.innerHTML});
        """
        )
        assert d["first"] == 1 and d["kept"] == 1 and d["again"] == 2
        assert "cv-pc-tiles" in d["card"]

    def test_an_unknown_person(self):
        d = ui(
            setup()
            + """
        h.answers['/api/conversations/person'] = {name: 'Grace', known: false, recordings: 0};
        h.run("cvShow({people: ['Grace']})");
        await h.run('cvOpen()'); await h.settle();
        output(h.elements.cvPerson.innerHTML);
        """
        )
        assert "is no name of the voice library" in d and "cv-pc-tiles" not in d


@node
class TestTheBackgroundCount:
    def test_the_info_line_follows_it(self):
        running = {**IDLE, "run": {"running": True, "done": 45, "total": 100}}
        stale = {**IDLE, "counts": {"transcripts": 52, "counted": 40, "stale": 12}}
        d = ui(
            setup(status=running)
            + f"""
        await h.run('cvOpen()'); await h.settle();
        const counting = h.elements.cvInfo.innerHTML;
        h.answers['/api/conversations/status'] = {json.dumps(IDLE)};
        await h.run('cvLoadStatus()');
        const finished = h.elements.cvInfo.innerHTML;
        h.answers['/api/conversations/status'] = {json.dumps(stale)};
        await h.run('cvLoadStatus()');
        const left = h.elements.cvInfo.innerHTML;
        h.answers['/api/conversations/run/start'] = {{started: true, run: {{running: true}}}};
        await h.run('cvRunStart(true)');
        h.sandbox.__confirm = false;
        await h.run('cvRunStart(true)');
        const starts = h.calls.api.filter(c => c.url === '/api/conversations/run/start');
        output({{counting, finished, left, starts, confirms: h.calls.confirm.length}});
        """
        )
        assert "Counting statistics: 45 % — results are not complete yet" in d["counting"]
        assert "cvRunStop()" in d["counting"] and "Count everything again" not in d["counting"]
        assert "Counting finished" in d["finished"] and "cvRefresh()" in d["finished"]
        assert "12 transcripts not counted yet" in d["left"] and "cvRunStart(false)" in d["left"]
        assert "cvRunStart(true)" in d["left"]
        # Counting everything again asks first — and does nothing on "no"
        assert d["confirms"] == 2 and len(d["starts"]) == 1
        assert d["starts"][0]["opts"] == {"method": "POST", "body": {"recount": True}}

    def test_a_poll_that_changes_nothing_redraws_nothing(self):
        # The status comes every two seconds while the count goes: a button
        # replaced under the pointer loses its click (Stop, Count now)
        running = {
            **IDLE,
            "counts": {"transcripts": 5, "counted": 0, "stale": 5},
            "run": {"running": True, "done": 1, "total": 5},
        }
        stopped = {**running, "run": {"running": False}}
        d = ui(
            setup(answer([]), status=running)
            + f"""
        await h.run('cvOpen()'); await h.settle();
        const marks = () => {{ h.elements.cvInfo.innerHTML += '<!--seen-->'; h.elements.cvList.innerHTML += '<!--seen-->'; }};
        marks();
        await h.run('cvLoadStatus()');
        const same = {{info: h.elements.cvInfo.innerHTML, list: h.elements.cvList.innerHTML}};
        h.answers['/api/conversations/status'] = {json.dumps(stopped)};
        await h.run('cvLoadStatus()');
        output({{same, info: h.elements.cvInfo.innerHTML, list: h.elements.cvList.innerHTML}});
        """
        )
        assert d["same"]["info"].endswith("<!--seen-->") and d["same"]["list"].endswith(
            "<!--seen-->"
        )
        # The count stopped with nothing counted: now the list says so
        assert "<!--seen-->" not in d["list"] and "Nothing is counted yet" in d["list"]
        assert "<!--seen-->" not in d["info"]

    def test_counting_again_stopping_and_a_missed_answer(self):
        again = {**IDLE, "run": {"running": True, "recount": True, "done": 30, "total": 40}}
        stopping = {**IDLE, "run": {"running": True, "stopping": True, "done": 31, "total": 40}}
        d = ui(
            setup(status=again)
            + f"""
        await h.run('cvOpen()'); await h.settle();
        const recount = h.elements.cvInfo.innerHTML;
        // An answer missed while it counts: the progress stays, and the
        // count that ends after it is still seen ending
        h.answers['/api/conversations/status'] = new Error('Not reachable');
        await h.run('cvLoadStatus()');
        const missed = h.elements.cvInfo.innerHTML;
        h.answers['/api/conversations/status'] = {json.dumps(stopping)};
        await h.run('cvLoadStatus()');
        const stop = h.elements.cvInfo.innerHTML;
        h.answers['/api/conversations/status'] = {json.dumps(IDLE)};
        await h.run('cvLoadStatus()');
        output({{recount, missed, stop, done: h.elements.cvInfo.innerHTML}});
        """
        )
        # The old numbers stay readable while they are counted again
        assert (
            "Counting everything again: 75 % — some numbers are still the old ones" in d["recount"]
        )
        assert "results are not complete" not in d["recount"] and "cvRunStop()" in d["recount"]
        assert "Counting everything again: 75 %" in d["missed"]
        assert "Stopping the count" in d["stop"] and "cvRunStop()" not in d["stop"]
        assert "Counting finished" in d["done"]

    def test_nothing_counted_yet(self):
        # The list answers before the status does (it is asked for first):
        # the empty list learns from the late counts why it is empty
        d = ui(
            setup(
                answer([]), status={**IDLE, "counts": {"transcripts": 5, "counted": 0, "stale": 5}}
            )
            + """
        await h.run('cvOpen()'); await h.settle();
        const order = h.calls.api.map(c => c.url.split('?')[0]);
        output({html: h.elements.cvList.innerHTML, order});
        """
        )
        assert d["order"].index("/api/conversations") < d["order"].index(
            "/api/conversations/status"
        )
        assert "Nothing is counted yet" in d["html"] and "cvRunStart(false)" in d["html"]

    def test_an_empty_library_and_a_first_count(self):
        empty = {**IDLE, "counts": {"transcripts": 0, "counted": 0, "stale": 0}}
        first = {
            **IDLE,
            "counts": {"transcripts": 5, "counted": 0, "stale": 5},
            "run": {"running": True, "done": 0, "total": 5},
        }
        out = []
        for status in (empty, first):
            out.append(
                ui(
                    setup(answer([]), status=status)
                    + """
            await h.run('cvOpen()'); await h.settle();
            output(h.elements.cvList.innerHTML);
            """
                )
            )
        # No transcripts: nothing to count — no button that counts nothing
        assert "No transcripts yet" in out[0] and "cvRunStart" not in out[0]
        # Counting already: the info line follows it, no second "Count now"
        assert (
            "Nothing is counted yet" not in out[1]
            and "No conversation fits these filters" in out[1]
        )


@node
class TestHandOffs:
    def test_asks_button(self):
        step = {
            "conversations": {
                "filters": {
                    "people": [EVIL + "'", "Bob"],
                    "match": "all",
                    "kind": "detailed",
                    "min_sec": 2700,
                    "sort": "sentences",
                    "language": "en",
                    "speakers_min": 2,
                },
                "total": 4,
            }
        }
        d = ui(f"""
        const html = h.sandbox.chatConvsHtml({json.dumps(step)});
        const none = h.sandbox.chatConvsHtml({{conversations: {{filters: {{}}}}, error: true}});
        // A step that names no speakers minimum: the tab's own (two) — what
        // Ask counted with too (mcp_server.CONVERSATION_SPEAKERS)
        const bare = h.sandbox.chatConvsHtml({{conversations: {{filters: {{kind: 'lively'}}, total: 1}}}});
        output({{html, none, bare, shown: h.calls.shown, tab: h.calls.tab}});
        """)
        assert "<img" not in d["html"] and d["none"] == ""
        assert "cvShow({people: [" in d["html"] and "speakers_min: 2" in d["html"]
        assert "speakers_min" not in d["bare"]
        for html, want in (
            (d["html"], step["conversations"]["filters"]),
            (d["bare"], {"kind": "lively"}),
        ):
            got = ui(f"""
            h.run({json.dumps(_clicked(html))});
            output({{shown: h.calls.shown, tab: h.calls.tab, speakers: h.value('_cv.f.speakers_min')}});
            """)
            assert got["shown"] == [want] and got["tab"] == ["conversations"]
            assert got["speakers"] == 2

    # (args of find_conversations as a model writes them)
    TOOL_CALLS = [
        {},
        {
            "people": ["Alice", "Bob"],
            "min_minutes": 45,
            "max_minutes": 75,
            "kind": "detailed",
            "sentences": "longer",
            "sort": "sentence_length",
        },
        {"people": ["Carol"], "together": False, "kind": "monologue", "language": "German"},
        {
            "people": ["Alice", "Dave"],
            "kind": "group",
            "min_speakers": 3,
            "max_speakers": 5,
            "min_words_per_sentence": 9.5,
            "folder": "/media/talks",
            "sort": "together",
        },
        {"min_speakers": 0, "folder": "talks", "sort": "speaker_changes"},
        {"sort": "together"},
    ]

    def test_ask_and_the_tab_ask_the_api_the_same(self):
        """The button under Ask's step opens the tab with the filters the
        tool searched with: the tab's request carries exactly the tool's
        filters (the API's defaults aside), so it finds what Ask said it
        found."""
        from whisper_tm import mcp_server

        steps, calls = [], []
        for args in self.TOOL_CALLS:
            q = mcp_server.conversation_query(args)
            steps.append({"conversations": {"filters": q, "total": 1}})
            # what find_conversations sends: a 0 ("no minimum") stays off the wire
            calls.append({k: v for k, v in q.items() if v != 0})
        buttons = ui(f"output({json.dumps(steps)}.map(s => h.sandbox.chatConvsHtml(s)));")
        d = ui(f"""
        const out = [];
        for (const code of {json.dumps([_clicked(b) for b in buttons])}) {{
            h.run(code);
            out.push(h.value('cvQuery()'));
        }}
        output(out);
        """)
        for args, sent, query in zip(self.TOOL_CALLS, calls, d, strict=True):
            assert _params(query) == _params_of(sent), (args, query, sent)

    def test_find_and_the_tab_ask_the_api_the_same(self):
        """Find's run hands its plan to the tab as API parameters
        (find/plan.conversation_filters); the tab asks with them."""
        from whisper_tm.find import plan

        plans = [
            {
                "people": ["Alice", "Bob"],
                "conversation": {
                    "min_minutes": 45,
                    "max_minutes": 75,
                    "kind": "detailed",
                    "sentences": "longer",
                },
            },
            {
                "people": ["Carol"],
                "file_hints": ["kitchen"],
                "conversation": {
                    "min_minutes": 0,
                    "max_minutes": 0,
                    "kind": "monologue",
                    "sentences": "any",
                },
            },
            {
                "people": [],
                "conversation": {
                    "min_minutes": 120,
                    "max_minutes": 0,
                    "kind": "group",
                    "sentences": "shorter",
                },
            },
        ]
        filters = [plan.conversation_filters(p) for p in plans]
        d = ui(f"""
        const out = [];
        for (const f of {json.dumps(filters)}) {{
            h.run('fdApply')({{status: 'conversations', rev: out.length, conversation_filters: f}});
            h.run('_fd.run = null');
            out.push(h.value('cvQuery()'));
        }}
        output(out);
        """)
        for f, query in zip(filters, d, strict=True):
            want = _params_of(f)
            # a conversation is two voices or more — unless the plan said
            # otherwise (a monologue: no minimum)
            if "speakers_min" not in f:
                want["speakers_min"] = [2.0]
            assert _params(query) == want, (f, query)

    def test_find_hands_over_once(self):
        filters = {
            "people": ["Alice", "Bob"],
            "match": "all",
            "min_sec": 1800,
            "kind": "detailed",
            "sentences": "longer",
        }
        d = ui(f"""
        const st = {{status: 'conversations', rev: 1, conversation_filters: {json.dumps(filters)}}};
        h.run('fdApply')(st);
        h.run('fdApply')({{...st, rev: 2}});
        const once = h.calls.shown.length;
        h.run('_fd.run = null; __findVisible = false');
        h.run('fdApply')({{...st, rev: 3}});
        const hidden = h.calls.shown.length;
        h.run('fdShowConversations()');
        output({{once, hidden, shown: h.calls.shown, f: h.value('_cv.f'), status: h.value('FD_STATUS.conversations')}});
        """)
        assert d["once"] == 1 and d["hidden"] == 1 and len(d["shown"]) == 2
        assert d["shown"][0] == filters
        assert d["f"]["people"] == ["Alice", "Bob"] and d["f"]["sentences"] == "longer"
        assert d["f"]["speakers_min"] == 2  # conversations: two voices or more
        assert "Conversations shows them" in d["status"]

    def test_voices_people_opens_conversations_in_the_same_window(self):
        # The Voices window has a Conversations tab of its own (Find's view,
        # one state): a person's button stays in the window
        d = ui("""
        h.run("peopleConversations('Alice')");
        output({order: h.calls.order, shown: h.calls.shown, people: h.value("_cv.f.people")});
        """)
        assert d["order"] == ["cvShowInVoices", "vxOpen:conversations"]
        assert d["shown"] == [{"people": ["Alice"]}] and d["people"] == ["Alice"]


FILE = {
    "abs_path": "/lib/talks/evening.mkv",
    "filename": "talks/evening.mkv",
    "file_dir": "/lib",
    "counted": True,
    "summary": {
        "duration_sec": 3600,
        "speech_sec": 3000,
        "wps": 11.4,
        "wps_pct": 78,
        "turn_rate": 3.1,
        "rate_pct": 30,
        "mean_turn_sec": 19.4,
        "turn_pct": 64,
        "questions": 41,
        "sentences": 360,
        "questions_share": 0.11,
        "longest_turn_sec": 140,
        "longest_turn_start": 1100,
        "longest_turn_label": "SPEAKER_01",
        "punctuated": 1,
    },
    "kind": "detailed",
    "group": True,
    "reasons": ["Sentences of 11.4 words — longer than 78 % of conversations.", EVIL],
    "speakers": [
        {"label": "SPEAKER_00", "name": "Alice", "share": 0.5},
        {"label": "SPEAKER_01", "name": EVIL, "share": 0.4},
        {"label": "SPEAKER_02", "name": None, "share": 0.1},
    ],
    "people": [
        {"name": "Alice", "labels": ["SPEAKER_00"], "talk_sec": 1500, "share": 0.5},
        {"name": EVIL, "labels": ["SPEAKER_01"], "talk_sec": 1200, "share": 0.4},
    ],
    "parts": [
        {"start_sec": 0, "end_sec": 900, "people": ["Alice"], "unnamed": 0},
        {"start_sec": 900, "end_sec": 3600, "people": ["Alice", EVIL], "unnamed": 1},
    ],
    "baseline": {"language": "en", "source": "language", "conversations": 120},
}


@node
class TestToTheExtractor:
    def test_extract_hands_the_filters_and_the_first_person_over(self):
        d = ui(
            setup()
            + """
        h.run("var __x = []; function extractFromConversations(f, p) { __x.push([f, p]); }");
        h.run("cvShow({people: ['Alice', 'Bob'], kind: 'detailed,group', sentences: 'longer', " +
              "min_sec: 2700, max_sec: 4500, sort: 'rate', copies: 'all'})");
        await h.settle();
        const tools = h.value('cvToolsHtml()');
        h.run('cvExtract()');
        output({x: h.value('__x'), tools, words: h.value("cvDescribe({people: ['Alice', 'Bob'], " +
            "kind: 'detailed,group', sentences: 'longer', min_sec: 2700, max_sec: 4500, language: 'DE'})"),
            alone: h.value("cvDescribe({sentences: 'shorter', min_sec: 600, speakers_min: 0})")});
        """
        )
        [[filters, person]] = d["x"]
        assert person == "Alice"
        # The order and the copies stay behind: the extractor keeps copies apart
        assert filters == {
            "people": ["Alice", "Bob"],
            "match": "all",
            "min_sec": 2700,
            "max_sec": 4500,
            "kind": "detailed,group",
            "sentences": "longer",
            "speakers_min": 2,
        }
        assert "cvExtract()" in d["tools"]
        assert d["words"] == [
            "Alice and Bob together",
            "45 to 75 minutes",
            "Detailed + Group",
            "Longer sentences than usual for them",
            "Language DE",
        ]
        assert d["alone"] == ["Shorter sentences than usual", "≥ 10 min", "Any number of speakers"]

    def test_what_the_sentences_were_measured_against(self):
        theirs = {
            "sentences": "longer",
            "source": "people",
            "people": [EVIL, "Bob"],
            "conversations": 114,
            "longer_from": 5.2,
            "shorter_to": 4.8,
        }
        few = {
            "sentences": "shorter",
            "source": "language",
            "language": "de",
            "conversations": 900,
            "longer_from": 8.3,
            "shorter_to": 6.8,
            "people_conversations": 7,
        }
        d = ui(
            setup(answer([row()], sentences_baseline=theirs))
            + f"""
        await h.run('cvOpen()'); await h.settle();
        output({{info: h.elements.cvInfo.innerHTML,
                 few: h.value('cvSentBaseHtml({json.dumps(few)})'),
                 none: h.value('cvSentBaseHtml(null)')}});
        """
        )
        assert "Longer than usual for them: from 5.2 words per sentence" in d["info"]
        assert "the 114 conversations of" in d["info"] and EVIL not in d["info"]
        assert "Shorter than usual in the library: up to 6.8 words per sentence" in d["few"]
        assert "they have only 7 conversations, too few" in d["few"]
        assert d["none"] == ""

    def test_rows_say_where_they_lie_for_them(self):
        """Measured against the people, a row's "shorter than 77 %" (of the
        library) under "longer than usual" read as a contradiction: the
        row says where it lies for them, the library goes into the tip."""
        theirs = {
            "sentences": "longer",
            "source": "people",
            "people": ["Alice", "Bob"],
            "conversations": 40,
            "longer_from": 5.2,
            "shorter_to": 4.8,
        }
        rows = [
            row(wps=5.6, wps_pct=23),
            row(abs_path="/lib/b.mkv", filename="b.mkv", wps=5.0, wps_pct=12),
        ]
        d = ui(
            setup(answer(rows, sentences_baseline=theirs))
            + """
        await h.run('cvOpen()'); await h.settle();
        const theirs = h.elements.cvList.innerHTML;
        h.answers['/api/conversations'] = """
            + json.dumps(answer(rows))
            + """;
        h.run("cvShow({people: ['Alice', 'Bob']})"); await h.run('cvOpen()'); await h.settle();
        output({theirs, library: h.elements.cvList.innerHTML});
        """
        )
        first, second = d["theirs"].split('class="cv-row" role="button"')[1:3]
        assert ">longer for them<" in first
        assert "Against the library’s conversations: shorter than 77 %" in first
        assert ">usual for them<" in second
        assert "shorter than 77 %</small>" in d["library"] and "for them" not in d["library"]

    def test_the_sentence_buttons_say_what_usual_is(self):
        d = ui(
            setup()
            + """
        const library = h.value('cvSentencesHtml()');
        h.run("cvShow({people: ['Alice']})"); await h.settle();
        output({library, theirs: h.value('cvSentencesHtml()')});
        """
        )
        assert "most conversations of the library" in d["library"]
        assert "usual for them" in d["theirs"] and "of the library" not in d["theirs"]


@node
class TestTheStatisticsCard:
    def test_the_card(self):
        d = ui(f"""
        h.answers['/api/conversations/file'] = {json.dumps(FILE)};
        await h.run("txsConvLoad('/lib/talks/evening.mkv')");
        const html = h.elements.txsConv.innerHTML;
        h.answers['/api/conversations/file'] = {{...{json.dumps(FILE)}, stale: true}};
        await h.run("txsConvLoad('/lib/talks/evening.mkv')");
        const stale = h.elements.txsConv.innerHTML;
        h.run('txsConvMore()');
        output({{html, stale, url: h.calls.api[0].url, order: h.calls.order, shown: h.calls.shown,
                 card: h.value('txsConvCard()')}});
        """)
        html = d["html"]
        assert d["url"] == "/api/conversations/file?file=%2Flib%2Ftalks%2Fevening.mkv"
        assert 'id="txsConv"' in d["card"] and "Conversation" in d["card"]
        assert EVIL not in html and "&lt;img" in html
        assert ">Detailed<" in html and ">Group<" in html and "Sentences of 11.4 words" in html
        assert "11.4" in html and "longer than 78% of conversations" in html
        assert "3.1<small> /min</small>" in html and "fewer than 70% of conversations" in html
        assert "41 of 360 sentences" in html and ">2m 20s<" in html and 'data-seek="1100"' in html
        assert (
            "&lt;img src=x onerror=alert(1)&gt; · at 18:20" in html
        )  # the longest monologue's speaker
        assert (
            'data-seek="900"' in html
            and "+ 1 unnamed" in html
            and "+ 1 voice without a name" in html
        )
        assert "txsConvMore()" in html and "More conversations with these people" in html
        for code in handlers(html):
            assert code == "txsConvMore()", code
        assert d["order"] == ["closeTxStats", "cvShow"] and d["shown"] == [
            {"people": ["Alice", EVIL]}
        ]
        assert (
            "Counted before" not in html
            and "Counted before the transcript last changed" in d["stale"]
        )

    def test_unpunctuated_and_undiarized(self):
        # Without sentence punctuation every segment counts as a sentence:
        # no question can be told apart. Without diarization nobody has a
        # voice to be named by.
        plain = {
            **FILE,
            "summary": {
                **FILE["summary"],
                "punctuated": False,
                "wps": None,
                "wps_pct": None,
                "questions": 0,
                "sentences": 800,
                "questions_share": 0.0,
            },
        }
        blind = {
            **FILE,
            "kind": "unknown",
            "group": False,
            "speakers": [],
            "people": [],
            "parts": [],
            "reasons": ["Its speakers were not told apart: who spoke when is not known."],
        }
        d = ui(f"""
        const out = [];
        for (const f of [{json.dumps(plain)}, {json.dumps(blind)}]) {{
            h.answers['/api/conversations/file'] = f;
            await h.run("txsConvLoad('/lib/talks/evening.mkv')"); out.push(h.elements.txsConv.innerHTML);
        }}
        output(out);
        """)
        tile = re.search(
            r'stv-kpi-label">Questions</div>\s*<div class="stv-kpi-value">([^<]*)</div>\s*'
            r'<div class="stv-kpi-sub">([^<]*)</div>',
            d[0],
        )
        assert tile and tile.groups() == ("—", "no sentence punctuation")
        assert "800 sentences" not in d[0]
        assert "were not told apart — there is nobody to name" in d[1]
        assert "Nobody in it has a name yet" not in d[1]

    def test_not_counted_missing_and_off(self):
        d = ui("""
        const out = [];
        h.answers['/api/conversations/file'] = {counted: false};
        await h.run("txsConvLoad('/lib/a.mkv')"); out.push(h.elements.txsConv.innerHTML);
        for (const [status, text] of [[404, 'No transcript for this file'],
                                      [503, 'Conversation statistics are switched off (conversations.enabled).'],
                                      [503, 'The database is busy — try again in a moment.']]) {
            const e = new Error(text); e.status = status;
            h.answers['/api/conversations/file'] = e;
            await h.run("txsConvLoad('/lib/a.mkv')"); out.push(h.elements.txsConv.innerHTML);
        }
        output(out);
        """)
        assert "Not counted yet" in d[0]
        assert "no transcript of this file" in d[1]
        assert "switched off" in d[2]
        assert "The database is busy" in d[3] and "switched off" not in d[3]

    def test_a_file_switch_drops_the_late_answer(self):
        d = ui(f"""
        let release;
        const gate = new Promise(r => {{ release = r; }});
        h.answers['/api/conversations/file'] = async (url) => url.includes('first')
            ? (await gate, {{...{json.dumps(FILE)}, kind: 'lively'}}) : {json.dumps(FILE)};
        const first = h.run("txsConvLoad('/lib/first.mkv')");
        await h.run("txsConvLoad('/lib/second.mkv')");
        release(); await first; await h.settle();
        output(h.elements.txsConv.innerHTML);
        """)
        assert ">Detailed<" in d and "Back-and-forth" not in d


class TestTheWiring:
    HTML = (STATIC / "index.html").read_text(encoding="utf-8")
    SV = (JS / "searchview.js").read_text(encoding="utf-8")

    def test_the_tab_sits_after_same_audio(self):
        ov = self.HTML[self.HTML.index('id="svOverlay"') :]
        nav = ov[ov.index('class="vx-tabs"') : ov.index("</nav>")]
        audio, conv = nav.index('id="svTabBtnAudio"'), nav.index('id="svTabBtnConversations"')
        files = nav.index('id="svTabBtnFiles"')
        assert audio < conv < files and "vx-tab-sep" not in nav[audio:conv]
        button = nav[conv : nav.index("</button>", conv)]
        assert (
            "onclick=\"svTab('conversations')\"" in button
            and '<use href="#i-conversation"/>' in button
        )
        assert '<symbol id="i-conversation"' in self.HTML

    def test_the_pane_holds_containers_only(self):
        start = self.HTML.index('id="svPaneConversations"')
        pane = self.HTML[start : self.HTML.index("<!-- ── Deep search", start)]
        for i in ("cvInfo", "cvFilters", "cvResults", "cvPerson", "cvList"):
            assert f'id="{i}"' in pane, i
        assert not re.search(r"\son[a-z]+=", pane)  # every control is drawn by the script
        assert 'class="sv-pane ms-pane cv-pane"' in self.HTML

    def test_the_script_is_loaded_after_the_duplicates(self):
        html = self.HTML
        tag = '<script src="/static/js/conversations.js?v={{ v }}"></script>'
        assert tag in html
        assert html.index("audiodups.js") < html.index(tag) < html.index("/static/js/app.js")

    def test_the_search_views_hooks(self):
        sv = self.SV
        assert "'audio', 'conversations'];" in sv
        chrome = sv[sv.index("function svTabChrome(") :]
        assert "|| name === 'conversations';" in chrome[: chrome.index("\n}\n")]
        show = sv[sv.index("function svShowTab(") :]
        assert (
            "if (leaving === 'conversations' && name !== 'conversations' && typeof cvLeave === 'function') cvLeave();"
            in show[: show.index("\n}\n")]
        )
        tab = sv[sv.index("function svTab(") :]
        assert (
            "if (name === 'conversations' && typeof cvOpen === 'function') cvOpen();"
            in tab[: tab.index("\n}\n")]
        )
        run = sv[sv.index("function svRun() {") :]
        assert (
            "if (svState.tab === 'conversations') { if (typeof cvOpen === 'function') cvOpen(); return; }"
            in run[: run.index("svState.query =")]
        )
        close = sv[sv.index("function closeSearchView() {") :]
        assert "if (typeof cvLeave === 'function') cvLeave();" in close[: close.index("\n}\n")]

    def test_the_other_places(self):
        people = (JS / "voices_people.js").read_text(encoding="utf-8")
        row = people[people.index("function peopleRowHtml(") :]
        assert "peopleConversations('${key}')" in row[: row.index("\n}\n")]
        chat = (JS / "chat.js").read_text(encoding="utf-8")
        assert "${chatDupsHtml(s)}${chatConvsHtml(s)}" in chat
        for tool in ("find_conversations:", "conversation_stats:", "person_stats:"):
            assert tool in chat[chat.index("const CHAT_TOOL_LABELS") :], tool
        stats = (JS / "stats.js").read_text(encoding="utf-8")
        # the cards are drawn by txsRenderInto (the Statistics tab and the dialog)
        modal = stats[stats.index("function txsRenderInto(") :]
        modal = modal[: modal.index("\n}\n")]
        assert (
            modal.index("${txsHero(d, fm, s)}")
            < modal.index("${txsConvCard()}")
            < modal.index("${txsTimelineCard(s)}")
        )
        assert "txsConvLoad(txsConvPath());" in modal
        pages = (JS / "settings_pages.js").read_text(encoding="utf-8")
        search = pages[
            pages.index("id: 'search', group: 'Search & AI'") : pages.index("id: 'media', group:")
        ]
        for key in ("conversations.enabled", "conversations.auto_run", "conversations.min_share"):
            assert f"key: '{key}'" in search, key
        assert "_sSection({id: 'conversations'" in search

"""The Find tab's page (find.js), run in node: the suspected duplicates on a
hit and under it, the preview playing the moment, and the preview holding
on to its hit while the list reorders.

The spot should be playable right there, and hits that look like
duplicates are grouped as suspected duplicates — files with the same name
are sometimes different things.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js"
HARNESS = Path(__file__).resolve().parent / "js" / "find_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def find(body: str) -> dict:
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
        env={
            "FIND_JS": str(JS / "find.js"),
            "STATE_JS": str(JS / "state.js"),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


RUN = """
const dup = (key, why, t = 51) => ({key, name: key.split('/').pop(), t, video: true, why, size: 50000000, length: 480.0,
                                    thumb: null, rel: null, dir: null});
const m1 = {key: '/a/old house.mp4', name: 'old house.mp4', t: 50, video: true, score: 6.2, verdict: null,
            reason: null, engines: {words: {rank: 0, t: 50, text: 'house'}}, boosts: [], thumb: '/api/find/frame?t=50',
            dups: [dup('/b/old house.mkv', 'the same length and words'),
                   dup('/c/<img src=x onerror=alert(1)>.webm', 'the same length and picture')]};
const m2 = {key: '/v/clip.mp4', name: 'clip.mp4', t: 100, end: 150, video: true, score: 5, verdict: null, reason: null,
            engines: {meaning: {rank: 0, t: 100, text: 'x'}}, boosts: [], thumb: null, dups: []};
// `_fd` is a const of the page script: reached through an expression, not the global object
const fd = h.value('_fd');
fd.run = {status: 'judging', judged: 0, judge_total: 2, engines: {}, best: [m1, m2]};
"""


class TestDuplicatesOnTheHits:
    def test_a_badge_and_the_duplicates_below_a_best_match(self):
        out = find(
            RUN
            + """
            const closed = h.value("fdBestRow(_fd.run.best[0], 0)");
            h.run('fdToggleDups(0)');
            const open = h.value("fdBestRow(_fd.run.best[0], 0)");
            output({closed, open, tiles: h.calls.tiles});
        """
        )
        assert (
            "⧉ 2" in out["closed"]
            and "fd-dups-badge" in out["closed"]
            and "fd-dup-name" not in out["closed"]
        )
        # a row is a focusable option; its badges are real buttons (no button inside a button)
        assert (
            'role="option" tabindex="0"' in out["closed"]
            and '<button class="fd-dups-badge' in out["closed"]
        )
        assert '<button class="fd-best' not in out["closed"]
        assert (
            out["open"].count('class="fd-dup') >= 2
            and "fdPick('best', 0, '/b/old house.mkv')" in out["open"]
        )
        assert "the same length and words" in out["open"] and out["tiles"] >= 1
        # a file name is text, never markup
        assert (
            "<img src=x" not in out["open"] and "&lt;img src=x onerror=alert(1)&gt;" in out["open"]
        )

    def test_a_hit_without_duplicates_has_no_badge(self):
        out = find(RUN + "output({row: h.value('fdBestRow(_fd.run.best[1], 1)')});")
        assert "fd-dups-badge" not in out["row"]


class TestThePreview:
    def test_it_holds_its_hit_while_the_list_reorders(self):
        out = find(
            RUN
            + """
            h.run("fdPick('best', 1)");
            // the language model's verdicts reorder the list: new objects, another order
            const fresh = JSON.parse(JSON.stringify(fd.run.best)).reverse();
            fresh[0].verdict = 8; fresh[0].reason = 'it is';
            fd.run.best = fresh;
            const got = h.value('fdPicked()');
            output({key: got.it.key, verdict: got.it.verdict, parent: got.parent});
        """
        )
        assert out == {"key": "/v/clip.mp4", "verdict": 8, "parent": None}

    def test_it_plays_the_moment_and_a_verdict_does_not_restart_it(self):
        out = find(
            RUN
            + """
            h.run("fdPick('best', 1)");
            const box = h.elements.fdPvMedia;
            const first = box.innerHTML, writes = box.writes;
            const el = box.media;
            el.currentTime = 10; el.fire('timeupdate');
            const clock = h.elements.fdPvClock.innerHTML;
            fd.run.best[1].verdict = 7; fd.run.best[1].reason = 'close';
            h.run('fdRenderPreview()');
            const info = h.elements.fdPvInfo.innerHTML;
            el.fire('play');
            h.run('fdGo(false)');
            output({first, writes, after: box.writes, clock, info, play: h.calls.play, paused: h.calls.paused,
                    stopped: el.paused});
        """
        )
        assert (
            '<video class="fd-pv-video" src="/api/find/clip?t=100.0&amp;key=%2Fv%2Fclip.mp4&amp;end=150.0"'
            in out["first"]
        )
        assert out["after"] == out["writes"]  # the verdict came; the clip played on
        assert "1:46" in out["clock"] and "in the recording" in out["clock"]  # 96 s + 10 s
        assert "close" in out["info"] and "Open in the player" in out["info"]
        assert out["paused"] >= 1  # the main player paused for the clip
        assert (
            out["play"] == [["clip.mp4", 106, "/v"]] and out["stopped"] is True
        )  # on from where it was watched

    def test_a_duplicate_plays_itself_and_says_whose_it_is(self):
        out = find(
            RUN
            + """
            h.run("fdPick('best', 0)");
            h.run("fdPickDup('/b/old house.mkv')");
            const got = h.value('fdPicked()');
            output({key: got.it.key, parent: got.parent.key, media: h.elements.fdPvMedia.innerHTML,
                    info: h.elements.fdPvInfo.innerHTML, open: [...fd.open]});
        """
        )
        assert out["key"] == "/b/old house.mkv" and out["parent"] == "/a/old house.mp4"
        assert "key=%2Fb%2Fold%20house.mkv" in out["media"]
        assert "Suspected duplicate of" in out["info"] and "Suspected duplicates · 2" in out["info"]
        assert (
            "&lt;img src=x onerror=alert(1)&gt;" in out["info"] and "<img src=x" not in out["info"]
        )
        assert out["open"] == ["/a/old house.mp4|50"]  # its group unfolds in the list

    def test_an_audio_moment_gets_a_sound_player(self):
        out = find(
            RUN
            + """
            fd.run.best.push({key: '/w/PTT-20200101-WA0001.opus', name: 'PTT-20200101-WA0001.opus', t: 2,
                                         engines: {}, boosts: [], dups: []});
            h.run("fdPick('best', 2)");
            output({media: h.elements.fdPvMedia.innerHTML, from: h.value("fdClipFrom({t: 2})")});
        """
        )
        assert "<audio" in out["media"] and "<video" not in out["media"] and out["from"] == 0


class TestOneLinePerRecording:
    """Several moments of one recording aren't duplicates, but in a list they
    look like them (the same video at 3:10 and at 1:50): the best one
    stands, the others are "+n more moments" under it."""

    BODY = (
        RUN
        + """
        const later = {key: '/a/old house.mp4', name: 'old house.mp4', t: 110, video: true, score: 2, verdict: 1,
                       reason: 'a wooden house model', engines: {seen: {rank: 5, t: 110, text: ''}}, boosts: [], dups: [],
                       rec: '/a/old house.mp4'};
        m1.rec = '/a/old house.mp4';
        fd.run.best = [m1, m2, later];
    """
    )

    def test_the_other_moments_fold_under_the_best(self):
        out = find(
            self.BODY
            + """
            const body = h.value("fdTileBody('best', fdTileState('best'))");
            h.run('fdToggleMore(0)');
            const open = h.value("fdTileBody('best', fdTileState('best'))");
            output({body, open, count: h.value("fdTileState('best').count")});
        """
        )
        assert out["count"] == 2 and out["body"].count('<div class="fd-best') == 2
        assert "+1 more moment" in out["body"] and "a wooden house model" not in out["body"]
        assert "a wooden house model" in out["open"] and "fdPick('best', 2)" in out["open"]

    def test_picking_a_folded_moment_unfolds_it(self):
        out = find(
            self.BODY
            + """
            h.run("fdPick('best', 2)");
            output({more: [...fd.more], picked: h.value('fdPicked()').it.t});
        """
        )
        assert out == {"more": ["/a/old house.mp4|50"], "picked": 110}


class TestTheReviewsPoints:
    """Independent review: the top bar's query was dropped when the
    field held text, polling went on after the server forgot the run,
    unchanged lists were sent again and again, "Captain's" was never
    marked, ↗ on the file names searched everything."""

    def test_a_description_from_the_top_bar_always_runs(self):
        out = find(
            RUN
            + """
            h.answers['/api/find'] = (url, opts) => ({id: 'r2', state: {rev: 1, erev: 0, status: 'planning', engines: {},
                                                                         best: [], log: [], started: 0}});
            h.elements.fdQuery.value = 'an old description still in the field';
            h.elements.fdGrid.children = [1];
            h.run("fdIncoming('Alice shouts at a party and climbs onto a table')");
            h.run('fdOpen()');
            await new Promise(r => setTimeout(r, 0));
            output({field: h.elements.fdQuery.value, posted: h.calls.api.filter(c => c.url === '/api/find').map(c => c.opts.body)});
        """
        )
        assert out["field"] == "Alice shouts at a party and climbs onto a table"
        assert out["posted"] == [{"query": "Alice shouts at a party and climbs onto a table"}]

    def test_a_forgotten_run_stops_the_polling(self):
        out = find(
            RUN
            + """
            h.answers['/api/find/gone'] = () => { const e = new Error('No such run'); e.status = 404; throw e; };
            fd.id = 'gone'; fd.seq = 7;
            await h.value('fdPoll(7)');
            output({polling: fd.polling, status: fd.run.status, agent: h.elements.fdAgent.innerHTML});
        """
        )
        assert (
            out["polling"] is False
            and out["status"] == "gone"
            and "server was restarted" in out["agent"]
        )

    def test_unchanged_lists_stay_on_the_page(self):
        out = find(
            RUN
            + """
            fd.run.engines = {words: {status: 'done', items: [{key: '/a.mp4', t: 1}]}};
            h.run("fdApply({rev: 9, erev: 4, engines_same: true, status: 'judging', best: [], log: [], started: 0," +
                  " engines: {words: {status: 'done', count: 1}}})");
            output({items: fd.run.engines.words.items.length, erev: fd.erev});
        """
        )
        assert out == {"items": 1, "erev": 4}

    def test_the_words_are_marked_on_the_plain_text(self):
        out = find(
            RUN
            + """
            const text = 'Captain' + String.fromCharCode(39) + 's Crimson <b>Syndicate</b> & co';
            output({a: h.sandbox.fdMark(text, 'captain* | crimson | syndicat*')});
        """
        )
        assert (
            out["a"]
            == "<mark>Captain</mark>&#39;s <mark>Crimson</mark> <mark>Syndicat</mark>e &amp; co"
        )

    def test_skipped_searches_are_a_line_not_tiles(self):
        out = find(
            RUN
            + """
            fd.run.plan = {said_meaning: 'x', file_hints: [], people: []};
            fd.run.engines = {words: {status: 'skipped', note: 'no words to look for'}, meaning: {status: 'done', items: []},
                              seen: {status: 'running'}, heard: {status: 'skipped', note: 'no sound described'},
                              files: {status: 'skipped', note: 'no file name remembered'}, people: {status: 'skipped', note: 'nobody named'}};
            fd.run.log = [];
            h.run('fdRenderAgent()');
            output({tiles: h.value('fdShownTiles()'), agent: h.elements.fdAgent.innerHTML});
        """
        )
        assert out["tiles"] == ["meaning", "seen"]
        assert "Not searched:" in out["agent"] and "no sound described" in out["agent"]
        assert "Ask about these results" in out["agent"]

    def test_the_file_names_tile_opens_the_files_tab_with_the_name(self):
        out = find(
            RUN
            + """
            fd.run.plan = {file_hints: ['-WA'], said_words: [], people: []};
            h.elements.svQuery = h.elements.svQuery || {value: ''};
            let tabs = [], runs = 0;
            h.sandbox.svTab = (t) => tabs.push(t);
            h.sandbox.svRun = () => { runs += 1; };
            h.sandbox.svState = {tab: 'find', mode: 'both'};
            h.run("fdOpenTab('files')");
            output({query: h.elements.svQuery.value, tabs, runs});
        """
        )
        assert out == {"query": "-WA", "tabs": ["files"], "runs": 0}  # the tab runs it — once


class TestAskingForTheDuplicates:
    """ "Find duplicate audio files" is no moment: Same audio shows them —
    at once when the words say it, after the plan when only the language
    model recognizes it (find/plan.py, asks_for)."""

    STUBS = """
        h.run("var __shown = []; amDupIntent = (q) => /duplicate/i.test(q) ? {kind: 'same', q: ''} : null;"
              + "amShowDuplicates = (o) => __shown.push(o);");
        h.elements.svPaneFind.classList.add('active');
    """

    def test_the_words_go_there_without_a_run(self):
        out = find(
            RUN
            + self.STUBS
            + """
            h.elements.fdQuery.value = 'find duplicate audio files';
            await h.value('fdRun()');
            output({shown: h.value('__shown'), posted: h.calls.api.filter(c => c.url === '/api/find').length});
        """
        )
        assert out["shown"] == [{"kind": "same", "q": ""}] and out["posted"] == 0

    def test_the_plan_sends_them_there_once(self):
        out = find(
            RUN
            + self.STUBS
            + """
            const st = {rev: 2, erev: 0, status: 'duplicates', ended: 3, engines: {}, best: [], log: [],
                        plan: {asks_for: 'contained', file_hints: ['Archive'], summary: 'compilations'}};
            h.run('fdApply(' + JSON.stringify(st) + ')');
            h.run('fdApply(' + JSON.stringify({...st, rev: 3}) + ')');
            output({shown: h.value('__shown'), agent: h.elements.fdAgent.innerHTML});
        """
        )
        assert out["shown"] == [{"kind": "contains", "q": "Archive"}]
        assert "Same audio shows them" in out["agent"]

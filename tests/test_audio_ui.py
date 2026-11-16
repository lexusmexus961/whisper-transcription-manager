"""The search view's tab "Same audio" (audiomatch.js) and Settings → Audio
analysis (settings_audio.js), run in node."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
JS = REPO / "whisper_tm" / "static" / "js"
HARNESS = Path(__file__).resolve().parent / "js" / "audiomatch_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


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

HITS = json.dumps(
    {
        "hits": [
            {
                "id": 7,
                "name": EVIL + ".mp3",
                "path": "/lib/a/" + EVIL + ".mp3",
                "file_dir": "/lib/a",
                "filename": EVIL + ".mp3",
                "at": 47.0,
                "score": 43,
                "strength": "strong",
                "q_start": 0.2,
                "q_end": 5.8,
                "duration": 105,
                "lufs": -21.5,
                "lra": 7.3,
                "peak_db": -6.5,
                "silence": 0,
                "clipped": 0,
                "places": [{"at": 47.0, "score": 43}, {"at": 90.5, "score": 12}],
            },
            {
                "id": 8,
                "name": "copy.mp3",
                "path": "/lib/copy.mp3",
                "file_dir": "/lib",
                "filename": "copy.mp3",
                "at": 12.0,
                "score": 14,
                "strength": "match",
                "q_start": 0,
                "q_end": 6,
                "duration": 40,
                "lufs": None,
                "places": [{"at": 12.0, "score": 14}],
            },
        ],
        "hashes": 352,
        "seconds": 6,
        "ms": 17,
    }
)


class TestTheTab:
    def test_the_info_line(self):
        d = ui("""
        const say = [];
        for (const info of [{enabled: false},
                            {enabled: true, counts: {files: 0}, run: {running: false}},
                            {enabled: true, counts: {files: 1234, hours: 150.4, same: 5, contains: 2}, run: {running: false}},
                            {enabled: true, counts: {files: 10, hours: 1}, run: {running: true, message: '3 of 9 recordings analyzed', eta_sec: 7200}},
                            {error: 'down'}]) {
            h.run('_am.info = ' + JSON.stringify(info)); h.run('amRenderInfo()');
            say.push(h.elements.amInfo.innerHTML);
        }
        output(say);
        """)
        assert "switched off" in d[0]
        assert "Nothing analyzed yet" in d[1]
        assert "1,234" in d[2] and "150 h" in d[2] and "5 pairs the same, 2 contained" in d[2]
        assert (
            "3 of 9 recordings analyzed" in d[3]
            and "about 2 h left" in d[3]
            and "amRunStop()" in d[3]
        )
        assert "Not reachable: down" in d[4]

    def test_from_the_clip_editor(self):
        d = ui(f"""
        h.answers['/api/audio/search'] = {HITS};
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.run("clipFile = 'talk.mp4'; clipIn = 12; clipOut = 18;");
        h.elements.svQuery.value = 'police';
        h.run('amFromClip()');
        const before = {{tab: h.calls.tab.slice(), open: h.calls.open.slice(), routed: h.sandbox.svState.routed}};
        h.run('amOpen()');                                   // what openSearchView's svRun() does
        await h.settle();
        const search = h.calls.api.find(c => c.url === '/api/audio/search');
        output({{before, body: search.opts.body, source: h.elements.amSource.innerHTML}});
        """)
        assert d["before"]["tab"] == ["audio"] and d["before"]["open"] == ["police"]
        assert d["before"]["routed"] == "police"  # the text query isn't routed elsewhere
        assert d["body"] == {"file": "talk.mp4", "start": 12, "end": 18}
        assert (
            "0:12–0:18" in d["source"]
            and "talk.mp4" in d["source"]
            and "352 landmarks, 17 ms" in d["source"]
        )

    def test_too_short_or_too_long_is_refused_without_asking(self):
        d = ui("""
        h.run("clipFile = 'talk.mp4'; clipIn = 12; clipOut = 12.5; amFromClip()");
        h.run("clipIn = 0; clipOut = 400; amFromClip()");
        output({calls: h.calls.api.length, open: h.calls.open.length, toast: h.calls.toast});
        """)
        assert d["calls"] == 0 and d["open"] == 0
        assert d["toast"] == ["Choose at least one second", "At most five minutes at a time"]

    def test_hits_are_escaped_and_open_where_they_are(self):
        d = ui(f"""
        h.run('_am.info = {{enabled: true, counts: {{files: 9}}}}');
        h.run('_am.query = {{type: "range", file: "talk.mp4", start: 12, end: 18}}');
        h.run('_am.result = ' + JSON.stringify({HITS}));
        h.run('amRender()');
        const list = h.elements.amResults.innerHTML;
        h.run('amPick(0)');
        const preview = h.elements.amPreview.innerHTML;
        h.run('amOpenAt(0, 1, false)');
        h.run('amOpenAt(0, 0, true)');
        output({{list, preview, play: h.calls.play, goto: h.calls.goto}});
        """)
        assert EVIL not in d["list"] and "&lt;img" in d["list"]
        assert EVIL not in d["preview"]
        assert "at 0:47" in d["list"] and "2 places" in d["list"] and "-22 LUFS" in d["list"]
        assert "/api/clip/audio?file=%2Flib%2Fa%2F%3Cimg" in d["preview"]  # encoded into the URL
        assert "start=47.00&amp;end=53.00" in d["preview"]  # the snippet's length
        assert "-21.5 LUFS" in d["preview"] and "7.3 LU" in d["preview"]
        assert d["play"] == [[EVIL + ".mp3", 90.5, "/lib/a"]]
        assert d["goto"] == [[EVIL + ".mp3", 47, "/lib/a"]]

    def test_nothing_found(self):
        d = ui("""
        h.run('_am.info = {enabled: true, counts: {files: 42}}');
        h.run('_am.result = {hits: [], hashes: 10, seconds: 3, ms: 2}; _am.query = {type: "file", name: "x.mp3"}');
        h.run('amRender()');
        output(h.elements.amResults.innerHTML);
        """)
        assert "No recording holds this snippet" in d and "42 analyzed" in d

    def test_a_file_of_ones_own_is_sent_as_a_form(self):
        d = ui(f"""
        h.answers['/api/audio/search-file'] = {HITS};
        h.run("amUseFile({{name: 'mine.mp3'}})");
        await h.settle();
        const c = h.calls.api[0];
        output({{url: c.url, parts: c.opts.body.parts.map(p => [p[0], p[1].name]), source: h.elements.amSource.innerHTML}});
        """)
        assert d["url"] == "/api/audio/search-file" and d["parts"] == [["file", "mine.mp3"]]
        assert "mine.mp3" in d["source"]


def f(i, name, folder, extra=None):
    """A recording as the duplicates' routes hand it out."""
    return {
        "id": i,
        "name": name,
        "filename": name,
        "file_dir": folder,
        "path": folder + "/" + name,
        "duration": 40,
        "lufs": -20,
        "size": 2_000_000,
        "video": False,
        "pos": 0,
        "largest": False,
        **(extra or {}),
    }


OVERVIEW = f"""
h.answers['/api/audio/overview'] = {
    json.dumps(
        {
            "same": {"groups": 2, "files": 5, "extra": 3, "extra_bytes": 3_500_000_000},
            "contains": {"wholes": 1, "parts": 2, "pairs": 2},
            "folders": [
                {
                    "id": "p1",
                    "a": "/m/g/dlc2",
                    "b": "/m/g/dlc3",
                    "within": False,
                    "common": "/m/g",
                    "a_rest": "dlc2",
                    "b_rest": EVIL,
                    "groups": 3246,
                    "a_files": 3269,
                    "b_files": 4777,
                    "a_bytes": 1e9,
                    "b_bytes": 1.2e9,
                    "a_total": 5641,
                    "b_total": 4777,
                    "layout": 15,
                },
                {
                    "id": "p2",
                    "a": "/m/WA",
                    "b": "/m/WA",
                    "within": True,
                    "common": "/m/WA",
                    "a_rest": "",
                    "b_rest": "",
                    "groups": 12,
                    "a_files": 25,
                    "b_files": 25,
                    "a_bytes": 5e6,
                    "b_bytes": 5e6,
                    "a_total": 400,
                    "b_total": 0,
                    "layout": 0,
                },
            ],
            "folders_total": 340,
            "analyzed": 68161,
            "q": "",
            "min_sec": 0,
        }
    )
};
"""

GROUPS = json.dumps(
    {
        "kind": "same",
        "total": 2,
        "offset": 0,
        "sort": "space",
        "groups": [
            {
                "count": 3,
                "length": 5364,
                "extra": 2,
                "extra_bytes": 1_637_000_000,
                "found_at": 1,
                "files": [
                    f(
                        1,
                        "film.mp4",
                        "/m/a",
                        {"video": True, "size": 5_900_000_000, "largest": True, "duration": 5364},
                    ),
                    f(
                        2,
                        EVIL + ".mp4",
                        "/m/" + EVIL,
                        {"video": True, "size": 1_637_000_000, "pos": 0.5, "duration": 5363},
                    ),
                    f(3, "film.mp3", "/m/b", {"largest": True, "pos": 0.02, "duration": 5364}),
                ],
            },
            {
                "count": 2,
                "length": 2.4,
                "extra": 1,
                "extra_bytes": 30_000,
                "found_at": 2,
                "files": [f(4, "hit.wav", "/m/g/dlc2"), f(5, "hit.wav", "/m/g/dlc3")],
            },
        ],
    }
)

WHOLES = json.dumps(
    {
        "kind": "contains",
        "total": 1,
        "offset": 0,
        "sort": "parts",
        "wholes": [
            {
                "whole": f(10, "stream.mp4", "/m/vods", {"video": True, "duration": 3600}),
                "covered": 90,
                "parts": [
                    {
                        **f(11, "clip one.mp4", "/m/clips", {"duration": 30}),
                        "at": 100,
                        "matched": 30,
                        "score": 80,
                    },
                    {
                        **f(12, EVIL, "/m/clips", {"duration": 60}),
                        "at": 1800,
                        "matched": 60,
                        "score": 70,
                    },
                ],
            }
        ],
    }
)


class TestTheDuplicates:
    def test_the_tab_opens_on_them(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 68161, hours: 3785, same: 5842, contains: 1222}}, run: {{running: true}}}};
        h.answers['/api/audio/duplicates'] = {GROUPS};
        await h.run('amOpen()'); await h.settle();
        output({{html: h.elements.amResults.innerHTML, views: h.elements.amViews.innerHTML,
                 source: h.elements.amSource.innerHTML, urls: h.calls.api.map(c => c.url)}});
        """
        )
        html = d["html"]
        assert "/api/audio/overview?folders=12" in d["urls"]
        assert "/api/audio/duplicates?kind=same&sort=space&offset=0&limit=40" in d["urls"]
        # The tiles
        assert "<b>2</b><span>recordings exist more than once</span>" in html
        assert (
            "5 files — 3 of them extra copies" in html
            and "<b>3.5 GB</b><span>in the extra copies</span>" in html
        )
        assert (
            "amView('contains')" in html and "the library run goes on" in html and "68,161" in html
        )
        # The folders: two bars facing each other, a whole copy, one folder with itself
        assert (
            'style="width:58%"' in html
            and 'style="width:100%"' in html
            and "ad-side b whole" in html
        )
        assert "the same layout in 15 subfolders" in html and "Show all 340" in html
        assert "inside one folder" in html and "25 of its 400 files" in html
        # The groups: each copy's waveform, keyed — names never in a handler
        assert (
            html.count('class="ad-wave"') == 5
            and 'data-k="g0.1"' in html
            and 'data-span="5364.02"' in html
        )
        assert "onclick=\"adHear('g0.1', 0)\"" in html
        assert EVIL not in html and "&lt;img src=x" in html
        assert "1.6 GB in the extra copies" in html and html.count(">largest<") == 2
        assert "≋ The same recording <small>2</small>" in d["views"]
        assert "Where else is a snippet?" in d["source"]

    def test_hearing_a_copy_and_the_same_moment_in_the_others(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = {GROUPS};
        await h.run('amOpen()'); await h.settle();
        h.run("adHear('g0.0', 100)");
        const pv = h.elements.amPreview.innerHTML;
        h.run("adOpenIn('g0.1', true, 99.5)");
        h.run("adOpenIn('g0.2', false)");
        output({{pv, goto: h.calls.goto, play: h.calls.play}});
        """
        )
        pv = d["pv"]
        assert "/api/clip/audio?file=%2Fm%2Fa%2Ffilm.mp4&amp;start=100.00&amp;end=160.00" in pv
        assert "≋ Copy 1 of 3 · from 1:40" in pv and "The same moment in the other copies" in pv
        # The others where their sound is at that moment (their place on the shared timeline)
        assert "adHear('g0.1', 99.50)" in pv and "adHear('g0.2', 99.98)" in pv
        assert "<span>Size</span><b>5.9 GB</b><span>Format</span><b>MP4 (video)</b>" in pv
        assert EVIL not in pv
        assert d["goto"] == [[EVIL + ".mp4", 99.5, "/m/" + EVIL]] and d["play"] == [
            ["film.mp3", 0, "/m/b"]
        ]

    def test_recordings_that_hold_others(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = {WHOLES};
        h.run("amView('contains')"); await h.settle();
        const html = h.elements.amResults.innerHTML;
        h.run("adHearPart('w0.1')");
        output({{html, pv: h.elements.amPreview.innerHTML, urls: h.calls.api.map(c => c.url)}});
        """
        )
        html = d["html"]
        assert "/api/audio/duplicates?kind=contains&sort=parts&offset=0&limit=40" in d["urls"]
        assert (
            "<b>1</b><span>recording contain others</span>" in html
            or "<b>1</b><span>recording contains others</span>" in html
        )
        assert "holds <b>2</b> recordings · 1:30 of it" in html
        # Each part a block where it begins: 100 s and 1800 s of 3600
        assert "left:2.7777777777777777%" in html and "left:50%" in html
        assert "onclick=\"adHearPart('w0.0')\"" in html and 'class="ad-wave big"' in html
        assert EVIL not in html
        assert "/api/clip/audio?file=%2Fm%2Fvods%2Fstream.mp4&amp;start=1800.00" in d["pv"]
        assert "Playing here" in d["pv"]

    def test_filters_folders_and_order(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = {GROUPS};
        await h.run('amOpen()'); await h.settle();
        h.run("adSetFilter('duplicates in Archive')"); await h.settle();
        h.run('adSetLength(60)'); await h.settle();
        h.run("adSetSort('copies')"); await h.settle();
        h.run('adPickPair(0)'); await h.settle();
        const html = h.elements.amResults.innerHTML;
        h.run('adReset()'); await h.settle();
        output({{urls: h.calls.api.map(c => c.url), html}});
        """
        )
        urls = d["urls"]
        assert "/api/audio/overview?q=Archive&folders=12" in urls
        assert (
            "/api/audio/duplicates?q=Archive&min_sec=60&kind=same&sort=space&offset=0&limit=40"
            in urls
        )
        assert (
            "/api/audio/duplicates?q=Archive&min_sec=60&kind=same&sort=copies&offset=0&limit=40"
            in urls
        )
        assert (
            "/api/audio/duplicates?q=Archive&min_sec=60&kind=same&sort=copies&pair=p1&offset=0&limit=40"
            in urls
        )
        # The order changes no tiles: the overview is not asked again for it
        assert sum(u.startswith("/api/audio/overview?q=Archive&min_sec=60") for u in urls) == 1
        assert "Only dlc2 ⟷ " in d["html"] and EVIL not in d["html"]
        assert urls[-1] == "/api/audio/duplicates?kind=same&sort=copies&offset=0&limit=40"

    def test_looking_again(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = {GROUPS};
        await h.run('amOpen()'); await h.settle();
        h.run('adRefresh()'); await h.settle();
        output({{urls: h.calls.api.map(c => c.url), html: h.elements.amResults.innerHTML}});
        """
        )
        # The overview worked out anew first, then the list from it
        assert d["urls"][-2:] == [
            "/api/audio/overview?folders=12&fresh=1",
            "/api/audio/duplicates?kind=same&sort=space&offset=0&limit=40",
        ]
        assert "adRefresh()" in d["html"]

    def test_the_folder_shown_is_the_recordings_own(self):
        # A transcribed recording opens under its library root (file_dir):
        # the row names the folder it lies in, not that root
        d = ui(
            OVERVIEW
            + """
        h.answers['/api/audio/info'] = {enabled: true, counts: {files: 9}, run: {}};
        const row = (id, name) => ({id, name, filename: 'Archive/2019/' + name, file_dir: '/lib',
                                    path: '/lib/Archive/2019/' + name, duration: 40, size: 5, video: false, pos: 0});
        h.answers['/api/audio/duplicates'] = {kind: 'same', total: 1, groups: [
            {count: 2, length: 40, extra: 1, extra_bytes: 5, files: [row(1, 'a.mp3'), row(2, 'b.mp3')]}]};
        await h.run('amOpen()'); await h.settle();
        h.run("adHear('g0.0', 1)");
        h.run("adOpenIn('g0.1', false)");
        output({html: h.elements.amResults.innerHTML, pv: h.elements.amPreview.innerHTML, play: h.calls.play});
        """
        )
        assert '">/lib/Archive/2019</span>' in d["html"] and '">/lib</span>' not in d["html"]
        assert ">/lib/Archive/2019</div>" in d["pv"]
        assert d["play"] == [["Archive/2019/b.mp3", 0, "/lib"]]  # opened under its root

    def test_the_routed_words_can_be_searched_instead(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = {GROUPS};
        h.run("amIncoming('Duplikate')");
        await h.run('amOpen()'); await h.settle();
        const html = h.elements.amResults.innerHTML;
        h.run('adAsWords()');
        output({{html, tab: h.calls.tab, query: h.elements.svQuery.value, routed: h.value('svState.routed')}});
        """
        )
        assert "Opened for “Duplikate”" in d["html"] and "adAsWords()" in d["html"]
        assert d["tab"] == ["text"] and d["query"] == "Duplikate" and d["routed"] == "Duplikate"

    def test_a_late_answer_of_the_other_view_is_dropped(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        let release;
        const gate = new Promise(r => {{ release = r; }});
        h.answers['/api/audio/duplicates?kind=contains'] = async () => {{ await gate; return {WHOLES}; }};
        h.answers['/api/audio/duplicates?kind=same'] = {GROUPS};
        await h.run('amOpen()'); await h.settle();               // the same recording: loaded
        h.run("amView('contains')"); await h.settle();           // its answer still out
        h.run("amView('same')"); await h.settle();               // back, from what was loaded
        release(); await h.settle();
        output({{kind: h.value('_ad.list.kind'), html: h.elements.amResults.innerHTML}});
        """
        )
        assert d["kind"] == "same" and "copies</b>" in d["html"] and "ad-whole" not in d["html"]

    def test_a_new_list_resets_the_preview_and_more_shows_no_card_twice(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        const page = {GROUPS};
        h.answers['/api/audio/duplicates'] = (url) => url.includes('offset=0') ? page
            : {{...page, groups: [page.groups[1], {{...page.groups[1], files: [{{...page.groups[1].files[0], id: 77}}]}}]}};
        await h.run('amOpen()'); await h.settle();
        h.run("adHear('g0.0', 5)");
        const before = h.elements.amPreview.innerHTML;
        await h.run('adLoad(true)'); await h.settle();
        const cards = h.value('_ad.list.items.length');
        h.run("adSetSort('copies')"); await h.settle();
        output({{before, after: h.elements.amPreview.innerHTML, cards}});
        """
        )
        assert (
            "film.mp4" in d["before"]
            and "<audio" not in d["after"]
            and "Each copy is drawn" in d["after"]
        )
        assert d["cards"] == 3  # the repeated group left out, the new one added

    def test_all_folders_keep_the_list(self):
        d = ui(
            OVERVIEW
            + f"""
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = {GROUPS};
        await h.run('amOpen()'); await h.settle();
        h.run('adAllFolders()'); await h.settle();
        output({{urls: h.calls.api.map(c => c.url)}});
        """
        )
        assert d["urls"][-1] == "/api/audio/overview?folders=200"
        assert sum(u.startswith("/api/audio/duplicates") for u in d["urls"]) == 1

    def test_the_filter_panel(self):
        ov = json.loads(OVERVIEW.split("= ", 1)[1].rsplit(";", 1)[0])
        ov["facets"] = {
            "people": [["Alice", 3], [EVIL, 1]],
            "ext": [["mp4", 9], ["wav", 2]],
            "lang": [["de", 4]],
            "days": ["2019-01-01", "2024-05-01"],
        }
        d = ui(f"""
        h.answers['/api/audio/overview'] = {json.dumps(ov)};
        h.answers['/api/audio/info'] = {{enabled: true, counts: {{files: 9}}, run: {{}}}};
        h.answers['/api/audio/duplicates'] = (url) => url.includes('kind=contains') ? {WHOLES} : {GROUPS};
        await h.run('amOpen()'); await h.settle();
        h.run('adPanel()');
        const panel = h.elements.amResults.innerHTML;
        h.run("adSetF('person', 'Alice')"); await h.settle();
        h.run("adChoose('media', 1)"); await h.settle();
        h.run('adToggleExt(1)'); await h.settle();
        h.run("adSetF('since', '2020-01-01')"); await h.settle();
        h.run("adChoose('minCopies', 1)"); await h.settle();
        h.run("adChoose('spread', 1)"); await h.settle();
        const last = h.calls.api.filter(c => c.url.startsWith('/api/audio/duplicates')).pop().url;
        h.run('adPanel()');
        const chips = h.elements.amResults.innerHTML;
        h.run("adClearF('person')"); await h.settle();
        const after = h.calls.api.filter(c => c.url.startsWith('/api/audio/duplicates')).pop().url;
        // The clips of one folder where they are in the streams
        h.run("amView('contains')"); await h.settle();
        h.run('adResetFilters()'); await h.settle();
        h.run('adPanel()');
        const panel2 = h.elements.amResults.innerHTML;
        h.run("adSetF('qPart', 'clips')"); await h.settle();
        h.run("adSetF('qWhole', 'vods')"); await h.settle();
        const contains = h.calls.api.filter(c => c.url.startsWith('/api/audio/duplicates')).pop().url;
        output({{panel, last, chips, after, panel2, contains}});
        """)
        assert (
            "Alice (3)" in d["panel"] and "&lt;img src=x" in d["panel"] and EVIL not in d["panel"]
        )
        assert (
            "adToggleExt(1)" in d["panel"]
            and 'min="2019-01-01"' in d["panel"]
            and "Copies" in d["panel"]
        )
        assert d["last"] == (
            "/api/audio/duplicates?person=Alice&media=video&ext=wav&since=2020-01-01"
            "&min_copies=3&spread=across&kind=same&sort=space&offset=0&limit=40"
        )
        assert (
            "👤 Alice" in d["chips"]
            and "Across folders" in d["chips"]
            and "3+ copies" in d["chips"]
        )
        assert "⚙ Filters <b>6</b>" in d["chips"]
        assert "person=" not in d["after"] and "media=video" in d["after"]
        assert (
            "Holding one in" in d["panel2"]
            and "Parts in" in d["panel2"]
            and "Copies" not in d["panel2"]
        )
        assert d["contains"] == (
            "/api/audio/duplicates?q_whole=vods&q_part=clips&kind=contains&sort=parts"
            "&offset=0&limit=40"
        )

    def test_switched_off(self):
        d = ui("""
        h.answers['/api/audio/info'] = {enabled: false, run: {}};
        await h.run('amOpen()'); await h.settle();
        output({html: h.elements.amResults.innerHTML, urls: h.calls.api.map(c => c.url)});
        """)
        assert "Same audio is switched off" in d["html"] and d["urls"] == ["/api/audio/info"]


class TestAskedForInWords:
    PHRASES = {
        "find duplicate audio files": {"kind": "same", "q": ""},
        "duplicates in Game Sounds 2": {"kind": "same", "q": "Game Sounds 2"},
        "zeig mir doppelte Videos im Ordner Urlaub 2024": {"kind": "same", "q": "Urlaub 2024"},
        "Duplikate": {"kind": "same", "q": ""},
        "show me dupes": {"kind": "same", "q": ""},
        "recordings that exist twice": {"kind": "same", "q": ""},
        "identical recordings in WhatsApp?": {"kind": "same", "q": "WhatsApp"},
        "duplicates of interview.mp4": {"kind": "same", "q": "interview.mp4"},
        "find duplicate audio files in my library": {"kind": "same", "q": ""},
        "which videos contain other videos": {"kind": "contains", "q": ""},
        "welche Aufnahmen sind in anderen Aufnahmen enthalten": {"kind": "contains", "q": ""},
        "Videos, die andere Videos enthalten": {"kind": "contains", "q": ""},
        "zeig mir alle Duplikate im Ordner Archive": {"kind": "same", "q": "Archive"},
        'Duplikate in "Archive/2024"': {"kind": "same", "q": "Archive/2024"},
        # A description that only mentions a copy stays a description
        "he says it twice": None,
        '"duplicates"': None,
        "Alice sagt zweimal hallo": None,
        "das Video wo wir doppelt so schnell fahren": None,
        "copies of the report": None,
        "Rezept mit allen Zutaten enthalten": None,
        # A file named so is the file search's
        "duplicates.mp3": None,
        "dupes/2024": None,
        # Words that merely occur (an independent review's list)
        "Best Of Compilation 2023.mp4": None,
        "Gaming Compilation.mp4": None,
        "funny cat compilation": None,
        "Duplicates – Season 2": None,
        "the songs are twice as loud as the speech": None,
        "recordings that are twice as long": None,
        "die Dateien sind doppelt so groß": None,
        "the box contains other toys": None,
        "Die Suppe enthält andere Zutaten": None,
        "how to duplicate a file in Photoshop": None,
        "he talks about duplicates of the keys": None,
        "Compilations": None,
        "Zusammenschnitt Hochzeit": None,
    }

    def test_the_phrases(self):
        d = ui(f"output({json.dumps(list(self.PHRASES))}.map(q => h.sandbox.amDupIntent(q)));")
        assert dict(zip(self.PHRASES, d, strict=True)) == self.PHRASES

    def test_the_top_bar_sends_them_here(self):
        d = ui("""
        const qs = ['find duplicate audio files', 'Duplikate in Archive/2024', 'Erin Baker', 'vacation/2024',
                    'Alice walks home at night and finds a lost dog', '"duplicates"'];
        const routed = qs.map(q => h.sandbox.svRouteQuery(q));
        h.run("amIncoming('duplicates in Archive')");
        output({routed, kind: h.value('_ad.kind'), q: h.value('_ad.f.q'), view: h.value('_am.view')});
        """)
        assert d["routed"] == ["audio", "audio", "text", "files", "find", "text"]
        assert (d["kind"], d["q"], d["view"]) == ("same", "Archive", "same")

    def test_asks_button(self):
        d = ui(f"""
        const html = h.sandbox.chatDupsHtml({{duplicates: {{kind: 'contains', q: {json.dumps(EVIL + "'")}, min_sec: 60}}}});
        const none = h.sandbox.chatDupsHtml({{duplicates: {{kind: 'same'}}, error: true}});
        output({{html, none}});
        """)
        assert "amShowDuplicates({kind: 'contains', q: '" in d["html"] and "minSec: 60" in d["html"]
        assert "<img" not in d["html"] and "\\x27'" in d["html"] and d["none"] == ""


class TestTheFileView:
    def test_loudness_and_copies(self):
        d = ui("""
        const box = {dataset: {file: '/lib/a.mp4'}, innerHTML: '', closest: () => box};
        h.sandbox.document.querySelectorAll = () => [box];
        h.answers['/api/audio/file'] = {enabled: true, analyzed: true, copies: 2, contains: 0, contained_in: 1,
            file: {lufs: -16.4, lra: 6.2, peak_db: -0.3, silence: 0.12, clipped: 0}};
        await h.run("fiAudio('/lib/a.mp4')");
        const chips = box.innerHTML;
        h.answers['/api/audio/file'] = {enabled: true, analyzed: false};
        box.innerHTML = 'before';
        await h.run("fiAudio('/lib/a.mp4')");
        output({chips, unanalyzed: box.innerHTML, url: h.calls.api[0].url});
        """)
        assert d["url"] == "/api/audio/file?file=%2Flib%2Fa.mp4"
        assert (
            "-16 LUFS" in d["chips"]
            and "Loudness range 6.2 LU" in d["chips"]
            and "Near silence 12 %" in d["chips"]
        )
        assert "3 copies" in d["chips"] and "amShowFile(" in d["chips"]
        assert "The same recording 2× elsewhere, inside 1 longer one" in d["chips"]
        assert d["unanalyzed"] == "before"

    def test_copies_open_the_duplicates_of_that_recording(self):
        d = ui(
            OVERVIEW
            + """
        h.answers['/api/audio/duplicates'] = {kind: 'same', total: 0, groups: [], offset: 0};
        h.answers['/api/audio/info'] = {enabled: true, counts: {files: 3}, run: {}};
        h.run("amShowFile('/lib/a.mp4', 1)");
        await h.run('amOpen()'); await h.settle();
        const first = h.elements.amResults.innerHTML;
        const url = h.calls.api.find(c => c.url.startsWith('/api/audio/duplicates')).url;
        h.run('adClearFile()'); await h.settle();
        const urls = h.calls.api.filter(c => c.url.startsWith('/api/audio/duplicates')).map(c => c.url);
        output({first, url, last: urls[urls.length - 1], tab: h.calls.tab, open: h.calls.open});
        """
        )
        assert (
            d["url"]
            == "/api/audio/duplicates?kind=same&sort=space&file=%2Flib%2Fa.mp4&offset=0&limit=40"
        )
        assert "Only <b>a.mp4</b>" in d["first"] and "It exists nowhere else" in d["first"]
        assert "ad-tiles" not in d["first"]  # one recording's, not the library's
        assert d["last"] == "/api/audio/duplicates?kind=same&sort=space&offset=0&limit=40"
        assert d["tab"] == ["audio"] and d["open"] == [""]


class TestTheSettingsPage:
    def test_a_running_run(self):
        d = ui("""
        h.run(`_audioRender({enabled: true, counts: {files: 120, hours: 30.5, errors: 2, same: 3, contains: 1,
                disk_bytes: 25e6, entries: 1100000, segments: 2, pending: 5000},
                run: {running: true, files_done: 40, files_total: 100, skipped: 10, message: '40 of 100', speed: 812,
                      eta_sec: 5400, current: ['a.mp4'], last_error: 'b.mp4: no sound'}},
              {files: 60, hours: 20, bytes: 9e11})`);
        output({state: h.elements.amSetState.innerHTML, run: h.elements.amSetRun.innerHTML,
                est: h.elements.amSetEstimate.innerHTML, start: h.elements.amSetStart.hidden, stop: h.elements.amSetStop.hidden});
        """)
        assert "120</b> recordings · 30.5 h of sound" in d["state"] and "2 unreadable" in d["state"]
        assert (
            "3 pairs that are the same" in d["state"]
            and "2 segments, 5,000 still in memory" in d["state"]
        )
        assert (
            'style="width:50.0%"' in d["run"]
            and "812× real time" in d["run"]
            and "1.5 h left" in d["run"]
        )
        assert "Now: a.mp4" in d["run"] and "b.mp4: no sound" in d["run"]
        assert d["est"] == "" and d["start"] is True and d["stop"] is False

    def test_what_is_left_and_off(self):
        d = ui("""
        const out = [];
        h.run(`_audioRender({enabled: true, counts: {files: 0}, run: {running: false}}, {files: 60, hours: 0, bytes: 3.6e12})`);
        out.push(h.elements.amSetEstimate.innerHTML);
        h.run(`_audioRender({enabled: true, counts: {files: 0}, run: {running: false}}, {files: 9, hours: 31021.5, bytes: 1e9})`);
        out.push(h.elements.amSetEstimate.innerHTML);
        h.run(`_audioRender({enabled: true, counts: {files: 5}, run: {running: false, message: 'Done — 5 recordings analyzed'}}, {files: 0, hours: 0, bytes: 0})`);
        out.push(h.elements.amSetEstimate.innerHTML, h.elements.amSetRun.innerHTML);
        h.run(`_audioRender({enabled: false, run: {}}, null)`);
        out.push(h.elements.amSetState.innerHTML);
        output(out);
        """)
        assert "60</b> recordings" in d[0] and "3.6 TB to read" in d[0] and "about 10.0 h" in d[0]
        assert "of sound" not in d[0]  # lengths unknown: not said
        assert "31,022 h of sound" in d[1] and "days of sound" not in d[1]
        assert "Every recording the library scan found is analyzed" in d[2]
        assert "Done — 5 recordings analyzed" in d[3]
        assert "Switched off" in d[4]

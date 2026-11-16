"""The Images & sounds tab (mediasearch.js), its settings (settings_media.js)
and the transcript filter note (fileview.js), run in node.

Why the filter note exists: a long transcript with thousands of segments can
look like "only snippets" when the transcript view's search box still holds a
word from an earlier search and only a few segments match. Now the box says
how many it shows.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
JS = REPO / "whisper_tm" / "static" / "js"
MEDIA_HARNESS = Path(__file__).resolve().parent / "js" / "mediasearch_harness.js"
FILEVIEW_HARNESS = Path(__file__).resolve().parent / "js" / "fileview_harness.js"
SETTINGS_HARNESS = Path(__file__).resolve().parent / "js" / "settings_media_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _node(source: str, env: dict) -> dict:
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={**env, "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


def media(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(MEDIA_HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch((e) => { console.error(e); process.exit(1); });\n"
    )
    return _node(source, {"MEDIA_JS": str(JS / "mediasearch.js"), "STATE_JS": str(JS / "state.js")})


INFO = """{enabled: true, environment_ok: true, ready: true, dims: 512,
    counts: {kinds: {image: 10, video: 4, videoaudio: 0, audio: 20, speech: 3, description: 0}, files: {done: 7}},
    llm: {configured: true, vision: true, check_count: 2}, results_per_kind: 48, indexer: {running: false}}"""

HIT = """(kind, n, extra) => Object.assign({id: 100 + n, kind, moment: 500 + n, score: 0.7 - n / 100, t: 125 + n,
    start: 120 + n, dur: 10, name: 'stream <' + n + '>.mp4', path: '/lib/stream.mp4', file_dir: '/lib',
    filename: 'stream.mp4', video: kind !== 'audio', thumb: kind !== 'audio' ? '/api/media-search/thumb/' + (500 + n) : null,
    clip: '/api/media-search/clip/' + (500 + n), text: 'said <b>' + n + '</b>', tags: [['police', 0.7]]}, extra || {})"""


class TestTheTab:
    def test_states_before_it_is_ready(self):
        d = media("""
        const say = [];
        for (const info of [{enabled: false}, {enabled: true, environment_ok: false, environment_problems: ['not installed']},
                            {enabled: true, environment_ok: true, ready: false, counts: {kinds: {}, files: {}}}]) {
            h.run('_ms.info = ' + JSON.stringify(info)); h.run('msRenderInfo()');
            say.push([h.elements.msInfo.innerHTML, h.elements.msResults.innerHTML]);
        }
        output(say);
        """)
        assert "switched off" in d[0][0] and "Switch it on" in d[0][1]
        assert "not installed" in d[1][0] and "Install its environment" in d[1][1]
        assert "The index is empty" in d[2][0] and "Build the index" in d[2][1]

    def test_ready_line_and_controls(self):
        d = media(
            f"""
        h.run('_ms.info = ' + {json.dumps(INFO)}.replace(/\\n/g, ' '));
        h.run('msRenderInfo(); msRenderControls()');
        output({{info: h.elements.msInfo.innerHTML, controls: h.elements.msControls.innerHTML}});
        """.replace("{json.dumps(INFO)}", json.dumps(INFO))
        )
        assert "10 stills" in d["info"] and "from 7 recordings" in d["info"]
        # vector sizes up to what is stored, kinds without vectors marked empty
        assert '<option value="512" selected>512 (all)</option>' in d["controls"].replace("\n", "")
        assert 'value="768"' not in d["controls"]
        assert 'data-kind="videoaudio"' in d["controls"] and "ms-kind on empty" in d["controls"]

    def test_a_text_search_and_its_lists(self):
        d = media(f"""
        h.run('_ms.info = ' + {json.dumps(INFO)});
        const hit = {HIT};
        h.answers['/api/media-search?'] = {{results: {{image: [hit('image', 0), hit('image', 1)], audio: [hit('audio', 0)],
            video: [], videoaudio: [], speech: [], description: []}}, sound_label: 'dog barking', embed_ms: 5, rank_ms: 1, dims: 512}};
        h.elements.svQuery.value = 'dog';
        h.run('msRun()');
        await new Promise((r) => setImmediate(r)); await new Promise((r) => setImmediate(r));
        output({{url: h.calls.api[0].url, html: h.elements.msResults.innerHTML}});
        """)
        assert d["url"] == "/api/media-search?limit=48&q=dog"
        html = d["html"]
        assert "Sounds asked as “dog barking”" in html
        assert (
            html.count("Check with the language model") == 1
        )  # stills only — the others are empty
        assert "nothing found" in html and "none in the index" in html
        assert "stream &lt;0&gt;.mp4" in html and "<b>0</b>" not in html  # escaped
        assert "said &lt;b&gt;0&lt;/b&gt;" in html

    def test_a_check_sorts_by_the_verdict_and_dims_the_misses(self):
        d = media(f"""
        const hit = {HIT};
        h.run('_ms.info = ' + {json.dumps(INFO)});
        h.run('_ms.results = ' + JSON.stringify({{image: [hit('image', 0), hit('image', 1), hit('image', 2)]}}));
        h.run('_ms.checks.image = {{scores: {{"501": 8.5, "500": 2.0}}, running: false}}');
        h.run("_ms.last = 'fire'; _ms.source = {{type: 'text'}}");
        const order = h.value('msOrdered("image").map(x => x.moment)');
        h.run('_ms.on = new Set(["image"]); msRender()');
        output({{order, html: h.elements.msResults.innerHTML}});
        """)
        assert d["order"] == [501, 500, 502]  # checked by verdict, unchecked after
        assert "ms-verdict yes" in d["html"] and "ms-verdict no" in d["html"]
        assert "ms-tile dim" in d["html"] and "Checked — again" in d["html"]

    def test_what_a_check_asks_with(self):
        d = media("""
        const q = [];
        h.run("_ms.last = 'fire'; _ms.source = {type: 'text'}"); q.push(h.value('msCheckQuery()'));
        h.run("_ms.source = {type: 'file', text: 'at night'}"); q.push(h.value('msCheckQuery()'));
        h.run("_ms.source = {type: 'like', item: {}}"); q.push(h.value('msCheckQuery()'));
        output(q);
        """)
        assert d == ["fire", "at night", ""]

    def test_similar_and_a_file_ask_by_their_own_parameters(self):
        d = media(f"""
        const hit = {HIT};
        h.run('_ms.info = ' + {json.dumps(INFO)});
        h.answers['/api/media-search?'] = {{results: {{}}, embed_ms: 0, rank_ms: 0, dims: 256}};
        h.run('_ms.results = ' + JSON.stringify({{image: [hit('image', 3)]}}));
        h.run('_ms.dims = 256; msLike("image", 0)');
        await new Promise((r) => setImmediate(r));
        h.run("_ms.source = {{type: 'file', qid: 'q1', name: 'me.png', kind: 'image', text: '', url: 'blob:x'}}; msSearch()");
        await new Promise((r) => setImmediate(r));
        output({{urls: h.calls.api.map(c => c.url), banner: h.elements.msSource.innerHTML}});
        """)
        assert d["urls"] == [
            "/api/media-search?limit=48&dims=256&like=103",
            "/api/media-search?limit=48&dims=256&qid=q1",
        ]
        assert "Your file" in d["banner"] and "me.png" in d["banner"]

    def test_the_preview_and_where_it_opens(self):
        d = media(f"""
        const hit = {HIT};
        h.run('_ms.results = ' + JSON.stringify({{image: [hit('image', 0)], audio: [hit('audio', 1)]}}));
        h.run('msPlay("image", 0)');
        const pv = h.elements.msPreview.innerHTML;
        h.run('msOpenAt("image", 0, false); msOpenAt("image", 0, true); msOpenAt("audio", 0, false)');
        output({{pv, play: h.calls.play, goto: h.calls.goto}});
        """)
        assert "<video" in d["pv"] and "/api/media-search/clip/500" in d["pv"]
        assert (
            "In the player" in d["pv"]
            and "Transcript" in d["pv"]
            and "Similar" in d["pv"]
            and "Describe" in d["pv"]
        )
        assert d["play"] == [
            ["stream.mp4", 123, "/lib"],
            ["stream.mp4", 121, "/lib"],
        ]  # 2 s before a still
        assert d["goto"] == [["stream.mp4", 123, "/lib"]]

    def test_spoken_text_points_to_the_semantic_search(self):
        d = media(f"""
        const hit = {HIT};
        h.run('_ms.info = ' + {json.dumps(INFO)});
        h.run('_ms.results = ' + JSON.stringify({{speech: [hit('speech', 0)], image: [hit('image', 1)]}}));
        h.run("_ms.last = 'money'; _ms.source = {{type: 'text'}}; _ms.on = new Set(['image', 'speech']); msRender()");
        const html = h.elements.msResults.innerHTML;
        h.run('msSearchSaid()');
        output({{html, tabs: h.calls.tab, runs: h.calls.run}});
        """)
        assert d["html"].count("ms-said") == 1 and "Search what was said" in d["html"]
        assert d["html"].index("ms-said") > d["html"].index("Stills")  # under its own list only
        assert d["tabs"] == ["text"] and d["runs"] == 1

    def test_the_examples_are_english(self):
        d = media(
            "h.run('_ms.info = {ready: true, counts: {kinds: {}}}; msRenderControls()'); output(h.elements.msExamples.innerHTML);"
        )
        assert "outdoors at night" in d and "police" in d

    def test_clock(self):
        d = media("output(h.value('[msClock(5), msClock(65), msClock(3725), msClock(-3)]'));")
        assert d == ["0:05", "1:05", "1:02:05", "0:00"]


def fileview(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(FILEVIEW_HARNESS)!r});\n"
        "const h = buildIt('/lib');\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n" + body
    )
    return _node(source, {"FILEVIEW_JS": str(JS / "fileview.js")})


class TestTheTranscriptFilterNote:
    BODY = """
    const segs = [];
    for (let i = 0; i < 10; i++) segs.push({dataset: {text: i === 3 || i === 7 ? 'vending machine' : 'said', speaker: 'S'}, style: {}});
    h.sandbox.document.querySelectorAll = (sel) => sel === '.seg' ? segs : [];
    const input = {value: ''};
    const note = {hidden: true, innerHTML: ''};
    h.sandbox.document.getElementById = (id) => ({txSearch: input, txFilterNote: note})[id] || null;
    h.run('var activeSpeaker = null;');
    """

    def test_a_filter_says_how_many_it_shows(self):
        d = fileview(
            self.BODY
            + """
        input.value = 'Machine';
        h.run('filterSegments()');
        output({hidden: note.hidden, note: note.innerHTML, shown: segs.filter(s => s.style.display !== 'none').length});
        """
        )
        assert d["shown"] == 2 and d["hidden"] is False
        assert "2 of 10 segments" in d["note"] and "show all" in d["note"]

    def test_show_all_clears_it(self):
        d = fileview(
            self.BODY
            + """
        input.value = 'machine';
        h.run('filterSegments(); txClearFilter()');
        output({value: input.value, hidden: note.hidden, shown: segs.filter(s => s.style.display !== 'none').length});
        """
        )
        assert d == {"value": "", "hidden": True, "shown": 10}


# ── Settings → Images & sounds: the index card ─────────────────────────────


def settings(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(SETTINGS_HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n" + body
    )
    return _node(source, {"JS_DIR": str(JS)})


STATE = """{counts: {kinds: {image: 3000, video: 1250, videoaudio: 1200, audio: 4500, speech: 2900, description: 0},
    files: {done: 500, partial: 1}, moments: 4500, repeats: 150, dims: 768, format: 'int8', bytes: 20000000},
    settings: {folders: []}, ram_bytes: 1048576,
    estimate: {files: 100000, done: 500, todo: 99500, fills: 0, video_hours: 8000, audio_hours: 2000,
        hours_done: 225, share_done: 0.009, moments: 6000000, vectors: 18000000,
        vectors_by_kind: {image: 4000000, audio: 5000000}, gpu_hours: 600, bytes: 13000000000,
        thumb_bytes: 4000000000, thumbs_have: 65000000, thumbs_cap: 4194304000, free_gb: 50, min_free_gb: 20,
        clip_frames: 4, layout: {dims: 768, format: 'int8', wanted_dims: 768, differs: false}},
    run: {running: true, files_done: 200, moments_done: 850, vectors: 850, moments_per_hour: 12500, repeats: 12,
        errors: [{path: '/x.mp4', error: 'broken'}], missing: 1, message: 'live.mp4 · 150/4000 moments'}}"""


class TestTheIndexCard:
    def test_the_same_bar_as_the_semantic_search_by_hours(self):
        d = settings(
            f"h.run('_mediaRenderIndex(' + {json.dumps(STATE)} + ')'); "
            "output({cov: h.elements.msSetCoverage.innerHTML, run: h.elements.msSetRun.textContent, "
            "start: h.elements.msSetStart, stop: h.elements.msSetStop});"
        )
        cov = d["cov"]
        assert '<div class="set-bar"><span style="width:0.9%"></span></div>' in cov
        assert "500 / 100,000" in cov and "recordings" in cov
        assert "0.9 % of 10,000 h indexed — running" in cov
        assert "🖼 200 in this run" in cov and "⏳ 99,500 to go" in cov and "⚠ 1 failed" in cov
        assert "left at 12,500 moments an hour" in cov  # 6,000,000 / 12,500 = 20 days
        assert "20.0 days left" in cov
        assert d["run"] == "⟳ live.mp4 · 150/4000 moments"
        assert d["stop"]["hidden"] is False and d["start"]["disabled"] is True
        assert d["start"]["textContent"] == "🖼 Indexing …"

    def test_four_numbers_and_the_kinds(self):
        d = settings(
            f"h.run('_mediaRenderIndex(' + {json.dumps(STATE)} + ')'); output(h.elements.msSetIndex.innerHTML);"
        )
        assert d.count('class="set-kpi"') == 4
        assert "of 100,000 in scope" in d and "768 numbers, 8-bit" in d
        assert (
            "3,000 stills · 1,250 clips · 1,200 clips with sound · 4,500 sounds · 2,900 spoken text · 0 descriptions"
            in d
        )
        assert "In RAM" in d

    def test_what_is_left_and_whether_the_disk_holds_it(self):
        d = settings(f"""
        const st = {STATE};
        h.run('_mediaRenderIndex(' + JSON.stringify(st) + ')');
        const fits = h.elements.msSetEstimate.innerHTML;
        st.estimate.free_gb = 25;
        h.run('_mediaRenderIndex(' + JSON.stringify(st) + ')');
        output({{fits, full: h.elements.msSetEstimate.innerHTML}});
        """)
        assert "the whole library" in d["fits"] and "4 frames per clip" in d["fits"]
        assert "50 GB free; a run waits below 20 GB" in d["fits"] and "doesn't fit" not in d["fits"]
        assert "set-err-text" in d["full"] and "doesn't fit" in d["full"]

    def test_an_index_of_another_size_says_how_to_change_it(self):
        d = settings(f"""
        const st = {STATE};
        st.estimate.layout = {{dims: 512, format: 'int8', wanted_dims: 768, differs: true}};
        h.run('_mediaRenderIndex(' + JSON.stringify(st) + ')');
        output(h.elements.msSetEstimate.innerHTML);
        """)
        assert "512 numbers per vector; the settings say 768" in d and "Rebuild everything" in d

    def test_before_the_first_estimate_and_after_a_run(self):
        d = settings(f"""
        const st = {STATE};
        delete st.estimate; st.estimate_computing = true;
        st.run = {{running: false, ended: 1, message: 'Done', moments_done: 10, vectors: 30, errors: []}};
        h.run('_mediaRenderIndex(' + JSON.stringify(st) + ')');
        output({{cov: h.elements.msSetCoverage.innerHTML, run: h.elements.msSetRun.textContent,
                 start: h.elements.msSetStart.textContent, stop: h.elements.msSetStop.hidden}});
        """)
        assert "Counting what the scope holds" in d["cov"]
        assert d["run"] == "✓ Done — 10 moments, 30 vectors" and d["stop"] is True
        d2 = settings(
            "h.run('_mediaRenderIndex({counts: {kinds: {}}, run: {running: false, ended: 1, message: \\'Done\\', moments_done: 1, vectors: 1, errors: []}})'); output(h.elements.msSetRun.textContent);"
        )
        assert d2 == "✓ Done — 1 moment, 1 vector"
        assert d["start"] == "🖼 Index"

    def test_waiting_for_disk_space_is_shown_as_a_problem(self):
        d = settings(f"""
        const st = {STATE};
        st.run.waiting = 'disk'; st.run.message = '⏳ Waiting for disk space — 18.0 GB free, the index leaves 20 GB';
        h.run('_mediaRenderIndex(' + JSON.stringify(st) + ')');
        output({{cls: h.elements.msSetRun.className, text: h.elements.msSetRun.textContent}});
        """)
        assert "err" in d["cls"] and d["text"].startswith("⏳ Waiting for disk space")

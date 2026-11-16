"""The performance view's Live range in the browser (static/js/perfview.js).

The Live range shows the last 1 min 30 s of the system performance tab
and keeps scrolling. Checked in node
(tests/js/perfview_harness.js): Live asks for the last 90 s once and then
only for what is new, one request at a time; failed requests wait longer
and longer; the drawn charts move left with the clock on every animation
frame; leaving Live — another range, closing the view, a hidden tab —
aborts its request and stops its timers and frames; and the history
ranges draw as before.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PERFVIEW = REPO / "whisper_tm" / "static" / "js" / "perfview.js"
HARNESS = REPO / "tests" / "js" / "perfview_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

T = 1_790_000_000  # the harness clock's start, in seconds

# Shared helpers for every test: a live answer, the live requests, the
# pending poll.
PRELUDE = """
const T = h.clock.ms / 1000;
const reading = (t, more = {}) => ({t, cpu: 37, gpu: 80, ram_mb: 20480, ram_total_mb: 65536,
    vram_mb: 7168, vram_total_mb: 12288, vram_own_mb: 1024, vram_foreign_mb: 5900,
    procs: {'own:server': 1024, 'foreign:llama-server': 5900}, ...more});
const sources = {'own:server': {label: 'whisper-tm server', own: true},
                 'foreign:llama-server': {label: 'llama-server', own: false}};
const liveAnswer = (ts, now, step = 1) => ({now, step, span: 90, sources,
                                            readings: ts.map(t => reading(t))});
const live = () => h.requests.map((r, i) => [r, i]).filter(([r]) => r.url.startsWith('/api/system/live'));
const lastLive = () => live()[live().length - 1][1];
const polls = () => h.pending().filter(t => t.fn === h.run('pvLivePoll'));
"""


def run(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt([{str(PERFVIEW)!r}]);\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + PRELUDE + body + "\n})()"
        ".catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


class TestThePolling:
    def test_the_last_90_s_first_then_only_what_is_new(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            const first = lastLive();
            const waiting = polls().length;          // none while the request runs
            await h.answer(first, liveAnswer([T - 3, T - 2, T - 1], T));
            const next = polls();
            await h.fire(next[0]);
            const second = lastLive();
            // The answer repeats a reading the view has: it is kept once
            await h.answer(second, liveAnswer([T - 1, T], T + 1));
            output({first: h.requests[first].url, waiting, nextMs: next.map(t => t.ms),
                    second: h.requests[second].url,
                    have: h.run('pv.lv.readings.map(r => r.t)').map(t => t - T)});
        """)
        assert r["first"] == "/api/system/live"
        assert r["waiting"] == 0
        assert r["nextMs"] == [1000]
        assert r["second"] == f"/api/system/live?since={T - 1}"
        assert r["have"] == [-3, -2, -1, 0]

    def test_one_request_at_a_time(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            h.run('pvLivePoll(); pvLivePoll();');
            output({live: live().length});
        """)
        assert r == {"live": 1}

    def test_a_coarser_step_polls_less_often(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            await h.answer(lastLive(), liveAnswer([T - 2], T, 2));
            output({ms: polls().map(t => t.ms), status: h.get('pvLiveStatus').innerHTML});
        """)
        assert r["ms"] == [2000]
        assert "one reading every 2 s" in r["status"]

    def test_failures_wait_longer_and_longer(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            const waits = [];
            let status = '';
            for (let k = 0; k < 5; k++) {
                await h.fail(lastLive());
                const t = polls();
                waits.push(t.map(x => x.ms));
                if (k === 0) status = h.get('pvLiveStatus').innerHTML;
                await h.fire(t[0]);
            }
            await h.answer(lastLive(), liveAnswer([T - 1], T));
            waits.push(polls().map(x => x.ms));
            output({waits, status, back: h.get('pvLiveStatus').innerHTML});
        """)
        assert r["waits"] == [[2000], [4000], [8000], [15000], [15000], [1000]]
        assert "trying again in 2 s" in r["status"]
        assert "one reading per second" in r["back"]

    def test_a_request_without_an_answer_is_given_up(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            const req = lastLive();
            const limit = h.pending().find(t => t.ms === h.run('PV_LIVE_TIMEOUT'));
            await h.fire(limit);
            output({aborted: h.requests[req].aborted, retry: polls().map(t => t.ms),
                    status: h.get('pvLiveStatus').innerHTML});
        """)
        assert r["aborted"] is True
        assert r["retry"] == [2000]
        assert "no answer" in r["status"]


class TestLeaving:
    def test_another_range_stops_everything(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            await h.answer(lastLive(), liveAnswer([T - 2, T - 1], T));
            await h.fire(polls()[0]);
            const inFlight = lastLive();
            h.run('pvSetRange(3600)');
            await h.settle();
            const hist = h.requests[h.requests.length - 1].url;
            const q = new URLSearchParams(hist.split('?')[1]);
            const left = {aborted: h.requests[inFlight].aborted, polls: polls().length,
                          frames: h.activeFrames(), lv: h.run('pv.lv'), liveOn: h.run('pv.liveOn'),
                          span: Math.round(q.get('end') - q.get('start'))};
            h.tick(30);
            h.frame();
            output({...left, hist, liveRequests: live().length});
        """)
        assert r["aborted"] is True
        assert r["polls"] == 0 and r["frames"] == 0
        assert r["lv"] is None and r["liveOn"] is False
        assert r["hist"].startswith("/api/system/history?")
        assert r["span"] == 3600
        assert r["liveRequests"] == 2

    def test_closing_stops_it_and_opening_brings_it_back(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            await h.answer(lastLive(), liveAnswer([T - 1], T));
            h.run('closePerfView()');
            const closed = {polls: polls().length, frames: h.activeFrames(), lv: h.run('pv.lv')};
            h.run('openPerfView()');
            output({closed, reopened: h.requests[lastLive()].url,
                    history: h.requests.filter(x => x.url.startsWith('/api/system/history')).length});
        """)
        assert r["closed"] == {"polls": 0, "frames": 0, "lv": None}
        assert r["reopened"] == "/api/system/live"  # afresh: the last 90 s
        assert r["history"] == 1  # only the first opening's

    def test_a_hidden_tab_asks_for_nothing(self):
        r = run("""
            h.run('openPerfView()');
            // The process list's first answer: one request of it at a time
            await h.answer(h.requests.findIndex(x => x.url === '/api/system/vram'), {gpus: []});
            h.run("pvSetRange('live')");
            await h.answer(lastLive(), liveAnswer([T - 1], T));
            await h.fire(polls()[0]);
            const inFlight = lastLive();
            h.setVisibility('hidden');
            const hidden = {aborted: h.requests[inFlight].aborted, polls: polls().length,
                            frames: h.activeFrames(), procs: h.run('pv.procsTimer'),
                            intervals: h.pending().filter(t => t.fn === h.run('pvLoadProcs')).length};
            const before = h.requests.length;
            h.setVisibility('visible');
            output({hidden, shown: h.requests.slice(before).map(x => x.url)});
        """)
        assert r["hidden"] == {
            "aborted": True,
            "polls": 0,
            "frames": 0,
            "procs": None,
            "intervals": 0,
        }
        assert r["shown"] == ["/api/system/live", "/api/system/vram"]

    def test_a_history_answer_does_not_land_on_live(self):
        r = run("""
            h.run('openPerfView()');                 // asks for 24 h first
            h.run("pvSetRange('live')");
            await h.answer(0, {tag: '24h', points: [], bucket: 30});
            output({tag: h.run('pv.data && pv.data.tag') ?? null,
                    loading: h.get('perfBody').classList.contains('loading')});
        """)
        assert r == {"tag": None, "loading": False}


class TestScrolling:
    def test_every_frame_moves_the_charts_with_the_clock(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            await h.answer(lastLive(), liveAnswer([T - 3, T - 2, T - 1, T], T));
            const groups = h.run('pv.lv.groups');
            const svgs = h.run('pv.lv.svgs');
            h.tick(0.45);
            h.frame();
            const first = groups.map(g => g.attrs.transform);
            const end = svgs.map(s => +s.dataset.end - T);
            h.tick(0.45);
            h.frame();
            const second = groups[0].attrs.transform;
            // Nothing new for over 5 s: drawn again, the grid keeps up
            h.tick(5);
            h.frame();
            output({n: groups.length, first, end, second, again: h.run('pv.lv.groups')[0] !== groups[0],
                    drawn: h.run('pv.lv.drawn') - T, frames: h.activeFrames()});
        """)
        assert r["n"] == 5  # CPU, GPU, RAM, VRAM, this app's VRAM
        # 0.45 s of 90 s across the 840-px plot (900 - 46 - 14)
        assert set(r["first"]) == {"translate(-4.20,0)"}
        assert r["second"] == "translate(-8.40,0)"
        # The right edge trails the clock by 2 s
        assert all(abs(e - (0.45 - 2)) < 1e-6 for e in r["end"])
        assert r["again"] is True and abs(r["drawn"] - 3.9) < 1e-6
        assert r["frames"] == 1  # the loop goes on

    def test_the_axis_follows_the_servers_clock(self):
        """A browser whose clock is off by minutes still sees the last 90 s."""
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            await h.answer(lastLive(), liveAnswer([T + 299, T + 300], T + 300));
            output({end: h.run('pv.lv.end') - T});
        """)
        assert r["end"] == 298


class TestDrawing:
    def test_raw_readings_in_the_same_charts(self):
        r = run("""
            h.run('openPerfView()');
            h.run("pvSetRange('live')");
            // A reading a second from T - 95 to T - 1: those before the left
            // edge (T - 2 - 90) are kept to scroll in, but are not on screen
            await h.answer(lastLive(), liveAnswer(Array.from({length: 95}, (_, k) => T - 95 + k), T));
            const html = h.get('pvLiveCharts').innerHTML;
            output({html, body: h.get('perfBody').innerHTML, kept: h.run('pv.lv.readings.length'),
                    rows: h.get('pvLiveRows').innerHTML, sum: h.get('pvLiveSum').textContent});
        """)
        html = r["html"]
        for series in ("cpu", "gpu", "ram_mb", "vram_mb", "vram_own"):
            assert f'data-series="{series}"' in html
        assert "pv-band" not in html and "pv-swing" not in html  # no mean/peak
        assert html.count('class="pv-area"') == 3  # CPU, GPU, RAM
        assert html.count('class="pv-scroll"') == 5 and "clipPath" in html
        assert ">37%<" in html  # the newest value
        assert "llama-server" in html and "whisper-tm server" in html  # sources by name
        assert "line: each reading" in html
        # Seconds on the x axis (TZ=UTC: T is 14:13:20, ticks every 15 s)
        assert ">:45<" in html and ">:30<" in html
        assert 'class="pv-live-btn active"' in r["body"]
        assert "pv-legend" not in r["body"]  # no mean/peak key
        assert r["kept"] == 95 and r["sum"] == "Data table (92 readings)"  # the 90 s on screen
        assert r["rows"].index(":13:19") < r["rows"].index(":13:14")  # newest first

    def test_sources_grouped_like_the_history(self):
        r = run("""
            const procs = {'own:server': 3000, 'own:diarization': 900};
            for (let k = 1; k <= 8; k++) procs['foreign:p' + k] = k <= 2 ? k * 10 : k * 1000;
            const src = {};
            Object.keys(procs).forEach(k => { src[k] = {label: k.split(':')[1], own: k.startsWith('own:')}; });
            const d = h.run('pvLiveData')({readings: [{t: 1, procs}, {t: 2, procs: null}], sources: src});
            output({keys: d.vram_by.keys.map(k => k.key), other: d.vram_by.series['foreign:other'],
                    p8: d.vram_by.series['foreign:p8'], low: d.vram_by.low, bucket: d.bucket});
        """)
        assert r["keys"] == [
            "own:server",
            "own:diarization",
            "foreign:p8",
            "foreign:p7",
            "foreign:p6",
            "foreign:p5",
            "foreign:p4",
            "foreign:p3",
            "foreign:other",
        ]
        assert r["other"] == [30, None]  # p1 + p2; unknown where no split
        assert r["p8"] == [8000, None]
        assert r["low"] == {} and r["bucket"] == 2

    def test_the_history_ranges_draw_as_before(self):
        r = run("""
            h.run('openPerfView()');
            const pts = [0, 1, 2].map(i => ({t: T - 3600 + i * 600, cpu: 10, cpu_max: 50, gpu: 20,
                gpu_max: 90, ram_mb: 1024, ram_total_mb: 65536, vram_mb: 2048, vram_total_mb: 12288,
                vram_own_mb: null, vram_foreign_mb: null}));
            await h.answer(0, {start: T - 86400, end: T, bucket: 600, points: pts, first: T - 90000,
                               retention_days: 30, vram_by: {keys: [], series: {}, low: {}, high: {}}});
            const body = h.get('perfBody').innerHTML;
            output({body});
        """)
        body = r["body"]
        assert 'class="pv-live-btn"' in body  # offered, not active
        assert body.count('class="active"') == 1  # 24h
        assert "pv-band" in body and "pv-legend" in body  # mean/peak as ever
        assert "pv-scroll" not in body and "clipPath" not in body

"""The Ask page in node — players, references, the order of a turn.

Tool calls belong between the chat messages where the model actually makes
them, not on top; the model can present audio directly with <audio> blocks
(as in Open WebUI), and the page understands them — the passages sidebar
must not stay empty while the answer shows <audio> blocks with
``http://host:5077/…`` in them.

What these tests pin:

* an <audio> block becomes a player — around a clip address of any host,
  around a reference (<audio>S12</audio>); a clip the conversation knows
  as a passage plays that passage, by its recording's full path;
* an <audio> around anything that isn't one of our clips stays text;
* the passages sidebar lists what the answers quote AND play, in order;
* each tool call stands between the texts of the round that made it; an
  answer kept without rounds shows its calls first;
* while an answer is written, a player tag that isn't closed waits.
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
const ctx = {
    console, URL, location: {origin: 'http://127.0.0.1:5077'},
    document: {getElementById: () => null, querySelectorAll: () => []},
    svState: {},
};
vm.createContext(ctx);
const state = js('state.js'), sv = js('searchview.js');
vm.runInContext([fn(state, 'esc'), fn(state, 'escJs'), fn(js('summarize.js'), '_renderMarkdown'),
                 fn(sv, 'svFmt'), fn(sv, 'svShortName'), js('chat.js'),
                 'this.CHAT = CHAT;'].join('\n'), ctx);
ctx.CHAT.conv = {turns: [], refs: {
    S3: {file: 'videos/a.mp4', filename: 'videos/a.mp4', abs_path: '/media/videos/a.mp4',
         file_dir: '/media', start: 12.0, text: 'money talk', speaker: 'Alice'},
    S5: {file: 'b.mp3', filename: 'b.mp3', abs_path: '/media/b.mp3', file_dir: '/media',
         start: 61.0, text: 'other'},
}};
const out = o => console.log('§§' + JSON.stringify(o));
"""


def js(body: str) -> dict:
    done = subprocess.run(
        ["node", "-e", HARNESS + body],
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


class TestPlayers:
    def test_a_clip_of_any_host_plays_its_passage(self):
        """The model copied the tool description's example host; the clip
        is a known passage — it plays by the recording's full path."""
        d = js(r"""
        const html = ctx.chatMd('Here:\n\n<audio>\nhttp://host:5077/api/clip/audio?file=videos%2Fa.mp4&start=10.0&end=40.0\n</audio>\n\nDone.');
        out({html});
        """)
        html = d["html"]
        # A video's passage plays as video: Find's clip from its moment
        assert '<video class="sv-audio-video" controls preload="none"' in html
        assert 'src="/api/find/clip?key=%2Fmedia%2Fvideos%2Fa.mp4&amp;t=12.0"' in html
        assert 'poster="/api/find/frame?key=%2Fmedia%2Fvideos%2Fa.mp4&t=12"' in html
        assert 'data-ref="S3"' in html and "money talk" in html and "&lt;audio" not in html

    def test_a_reference_in_an_audio_tag(self):
        d = js(
            "out({html: ctx.chatMd('<audio>S5</audio>'), garbled: ctx.chatMd('<mudio>S5</audio>'),"
            " other: ctx.chatMd('<b>bold</b> text')});"
        )
        assert 'data-ref="S5"' in d["html"] and "file=%2Fmedia%2Fb.mp3&amp;start=59.0" in d["html"]
        assert 'data-ref="S5"' in d["garbled"] and "mudio" not in d["garbled"]
        assert "<audio" not in d["other"]

    def test_an_unknown_clip_plays_as_given_and_says_so(self):
        """No tool result named the moment — a model may write dozens of
        such "hits" itself. It plays, so it can be checked, and is marked."""
        d = js(r"""out({html: ctx.chatMd('<audio>http://192.0.2.5:5077/api/clip/audio?file=x%2Fy.mp3&start=3&end=9</audio>'),
                        known: ctx.chatMd('<audio>S3</audio>')});""")
        assert 'src="/api/clip/audio?file=x%2Fy.mp3&amp;start=3.0&amp;end=9.0"' in d["html"]
        assert "sv-audio unsure" in d["html"] and "not from a search" in d["html"]
        assert "unsure" not in d["known"]

    def test_anything_else_in_an_audio_tag_stays_text(self):
        d = js(r"""out({html: ctx.chatMd('<audio>https://example.com/a.mp3</audio>')});""")
        assert "<audio controls" not in d["html"] and "&lt;audio&gt;" in d["html"]

    def test_references_stay_chips(self):
        d = js("out({html: ctx.chatMd('Said here [S3][S5].')});")
        assert d["html"].count('class="sv-ref') == 2

    def test_an_unclosed_tag_waits_while_written(self):
        d = js(r"""out({live: ctx.chatMd('Listen:\n\n<audio>\nhttp://ho', true),
                        done: ctx.chatMd('Listen:\n\n<audio>\nhttp://ho', false)});""")
        assert "audio" not in d["live"] and "&lt;audio&gt;" in d["done"]


class TestTheSidebar:
    def test_what_is_quoted_and_played_in_order(self):
        d = js(r"""
        ctx.CHAT.conv.turns = [{role: 'user', text: 'q'},
            {role: 'assistant', text: 'First [S5], then:\n<audio>http://h/api/clip/audio?file=videos%2Fa.mp4&start=10&end=40</audio>\nand [S5] again, and <audio>http://h/api/clip/audio?file=c.mp4&start=1&end=5</audio>'}];
        out({shown: ctx.chatPresented()});
        """)
        assert d["shown"] == [
            {"ref": "S5"},
            {"ref": "S3"},
            {"file": "c.mp4", "start": 1, "end": 5, "key": "c.mp4@1"},
        ]


class TestTheOrderOfATurn:
    def test_each_call_between_the_texts_of_its_round(self):
        d = js(r"""
        const t = {role: 'assistant', at: 1, text: 'Let me look.\n\nIt was [S3].',
                   rounds: ['Let me look.', '', 'It was [S3].'],
                   steps: [{id: 'a', name: 'search', args: {text: 'x'}, round: 0, summary: '3 hits'},
                           {id: 'b', name: 'who_speaks', args: {file: 'a'}, round: 1, summary: '2 people'}]};
        out({html: ctx.chatTurnHtml(t)});
        """)
        h = d["html"]
        order = [h.index("Let me look."), h.index("3 hits"), h.index("2 people"), h.index("It was")]
        assert order == sorted(order), order

    def test_an_answer_kept_without_rounds_shows_its_calls_first(self):
        d = js(r"""
        out({html: ctx.chatTurnHtml({role: 'assistant', at: 1, text: 'Answer.',
             steps: [{id: 'a', name: 'search', args: {}, summary: '1 hits'}]})});
        """)
        assert d["html"].index("1 hits") < d["html"].index("Answer.")

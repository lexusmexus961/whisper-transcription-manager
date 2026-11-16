"""The gallery threshold setting reaches the file view.

``speaker.threshold_gallery`` is a setting, and the server applies it
whenever a request brings no threshold of its own. The file view's voice
match never let it: ``speaker.js`` started with ``let speakerThreshold =
0.75`` and sent ``?threshold=0.75`` with every request, so whatever was
configured, the view showed the result for 0.75 — and a slider saying 0.75.

Now the first request of a file view sends no threshold, and the slider
starts at the value the server reports back. Only a value picked with the
slider is sent, and only for the file it was picked on.

The logic under test (static/js/speaker.js, via tests/js/speaker_harness.js),
against a fake server that answers like ``/api/speaker/recording``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "speaker_harness.js"
JS = REPO / "whisper_tm" / "static" / "js" / "speaker.js"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str, **setup) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt({json.dumps(setup)});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"SPEAKER_JS": str(JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@node
class TestTheSettingApplies:
    def test_the_first_request_leaves_the_threshold_to_the_server(self):
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            output({requests: h.requests, slider: h.slider()});
        """,
            configured=0.81,
        )
        assert r["requests"] == ["/api/speaker/recording?file=a.mkv"]
        assert r["slider"]["value"] == "0.81"
        assert r["slider"]["label"] == "0.81"

    def test_nobody_reaching_it_is_told_the_threshold_in_effect(self):
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            output({html: h.box.innerHTML});
        """,
            configured=0.81,
        )
        assert "reach the threshold 0.81" in r["html"]

    def test_a_setting_outside_the_slider_range_is_still_where_it_starts(self):
        """The setting is a free number; a thumb pinned at 0.95 would show
        another threshold than the one in effect."""
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            output({slider: h.slider()});
        """,
            configured=0.97,
        )
        s = r["slider"]
        assert s["value"] == "0.97" and s["label"] == "0.97"
        assert float(s["min"]) <= 0.97 <= float(s["max"])


@node
class TestTheSlider:
    def test_a_moved_slider_is_sent(self):
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            h.run("setSpeakerThreshold('0.7')");
            await h.settle();
            output({requests: h.requests, slider: h.slider()});
        """,
            configured=0.81,
        )
        assert r["requests"][1] == "/api/speaker/recording?file=a.mkv&threshold=0.7"
        assert r["slider"]["value"] == "0.7"
        assert r["slider"]["label"] == "0.70"

    def test_the_same_file_keeps_the_picked_value(self):
        """The file view renders the box anew and asks again — the value
        picked for this file is still meant."""
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            h.run("setSpeakerThreshold('0.7')");
            await h.settle();
            await h.run("loadSpeakerRecognition('a.mkv')");
            output({requests: h.requests});
        """,
            configured=0.81,
        )
        assert r["requests"][-1] == "/api/speaker/recording?file=a.mkv&threshold=0.7"

    def test_the_next_file_starts_at_the_setting_again(self):
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            h.run("setSpeakerThreshold('0.7')");
            await h.settle();
            await h.run("loadSpeakerRecognition('b.mkv')");
            output({requests: h.requests, slider: h.slider()});
        """,
            configured=0.81,
        )
        assert r["requests"][-1] == "/api/speaker/recording?file=b.mkv"
        assert r["slider"]["value"] == "0.81"

    def test_an_unusable_value_falls_back_to_the_setting(self):
        """Not to a number of the view's own — that was the bug."""
        r = js(
            """
            await h.run("loadSpeakerRecognition('a.mkv')");
            h.run("setSpeakerThreshold('0.7')");
            await h.settle();
            h.run("setSpeakerThreshold('abc')");
            await h.settle();
            output({requests: h.requests, slider: h.slider()});
        """,
            configured=0.81,
        )
        assert r["requests"][-1] == "/api/speaker/recording?file=a.mkv"
        assert r["slider"]["value"] == "0.81"

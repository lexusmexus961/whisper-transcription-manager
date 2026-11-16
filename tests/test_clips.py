"""The Clips tab's server side: cuts kept per recording (core/cuts.py), the
clip render (core/clip_render.py) on generated media with real ffmpeg,
the clip job, Best moments (core/best_moments.py) and Export's "Only the
clip cuts"."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from flask import Flask

from whisper_tm.database import Database

ROOT = "/wtm-test-library/media"
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _probe(p: Path) -> dict:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height:format=duration",
            "-of",
            "json",
            str(p),
        ],
        capture_output=True,
        text=True,
    )
    d = json.loads(r.stdout)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    return {
        "duration": float(d["format"]["duration"]),
        "video": v,
        "size": (v["width"], v["height"]) if v else None,
    }


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """A 10 s video: a tone 0–3 s, silence 3–6.5 s, a tone 6.5–10 s; and the
    same as sound only."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("clipmedia")
    tone = "aevalsrc='if(between(t,3,6.5),0,0.4*sin(2*PI*440*t))':s=48000:d=10"
    v = d / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=25:duration=10",
            "-f",
            "lavfi",
            "-i",
            tone,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(v),
        ],
        check=True,
    )
    a = d / "memo.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", tone, "-c:a", "aac", str(a)], check=True
    )
    return d


# ── Cuts ──────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path, monkeypatch):
    from whisper_tm import extensions as ext

    d = Database(tmp_path / "test.db")
    d.initialize()
    monkeypatch.setattr(ext, "db", d)
    return d


def test_cuts_are_checked_and_kept(db):
    from whisper_tm.core import cuts

    a = ROOT + "/talk.mp4"
    assert cuts.load(a) == {"cuts": [], "settings": cuts.SETTINGS_DEFAULTS, "made": []}
    d = cuts.save(
        a,
        [{"start": 8, "end": 5, "ai": 1, "title": "x" * 300}, {"start": 1, "end": 2}],
        {"format": "vert", "crop": 3},
        length=7.5,
    )
    assert d["cuts"][0] == {"start": 5.0, "end": 7.5, "ai": True, "title": "x" * 120}, (
        "turned round, clipped to the length"
    )
    assert d["settings"]["format"] == "vert" and d["settings"]["crop"] == 1.0
    assert cuts.get(a) == d["cuts"], "kept for the next reader"
    for bad in (
        [{"start": 1, "end": 1.1}],
        [{"start": "x", "end": 2}],
        "nope",
        [{}] * 60,
        [{"start": 1, "end": 1e999}],
        [{"start": float("-inf"), "end": 5}],
    ):
        with pytest.raises(cuts.CutsError):
            cuts.save(a, bad)
    with pytest.raises(cuts.CutsError):
        cuts.save(a, None, {"format": "cinema"})
    cuts.add(a, [{"start": 20, "end": 30, "ai": True}])
    assert len(cuts.get(a)) == 3
    cuts.note_made(a, {"at": 1, "n": 1})
    assert cuts.load(a)["made"][0]["n"] == 1


# ── Render ────────────────────────────────────────────────────────────


def test_pauses_found_and_left_out():
    from whisper_tm.core import clip_render as cr

    assert cr.speech_parts(0, 10, [(3, 6.5)]) == [(0, 3.15), (6.35, 10)]
    assert cr.speech_parts(0, 10, []) == [(0, 10)]
    parts = cr.plan(
        Path("x"),
        [{"start": 0, "end": 10}, {"start": 12, "end": 14}],
        {"pauses": True, "pause_min": 0.8, "fade": True},
        find_silences=lambda s, a, b, m: [(3, 6.5)] if a == 0 else [],
    )
    assert [(p.start, p.end, p.fade_in, p.fade_out) for p in parts] == [
        (0, 3.15, True, False),
        (6.35, 10, False, True),
        (12, 14, True, True),
    ]


@needs_ffmpeg
def test_silence_detection_on_real_sound(media):
    from whisper_tm.core import clip_render as cr

    quiet = cr.silences(media / "memo.m4a", 0, 10, 0.8)
    assert len(quiet) == 1 and abs(quiet[0][0] - 3) < 0.15 and abs(quiet[0][1] - 6.5) < 0.15


@needs_ffmpeg
def test_two_cuts_one_clip_has_their_length(media, tmp_path):
    from whisper_tm.core import clip_render as cr

    s = {"format": "orig", "fade": True}
    parts = cr.plan(media / "talk.mp4", [{"start": 1, "end": 3}, {"start": 7, "end": 9.5}], s)
    out = cr.render(media / "talk.mp4", parts, tmp_path / "c", s, None)
    p = _probe(out)
    assert out.suffix == ".mp4" and abs(p["duration"] - 4.5) < 0.2 and p["size"] == (640, 360)
    assert [x.name for x in tmp_path.iterdir()] == ["c.mp4"], "no part files left"


@needs_ffmpeg
def test_vertical_with_subtitles_and_without_pauses(media, tmp_path):
    from whisper_tm.core import clip_render as cr

    s = {
        "format": "vert",
        "crop": 0.0,
        "pauses": True,
        "pause_min": 0.8,
        "subs": True,
        "fade": False,
    }
    parts = cr.plan(media / "talk.mp4", [{"start": 0, "end": 10}], s)
    lines = [
        {"start": 0.5, "end": 2.5, "text": "We start here.", "name": ""},
        {"start": 7, "end": 9, "text": "And end here.", "name": ""},
    ]
    out = cr.render(media / "talk.mp4", parts, tmp_path / "v", s, lines)
    p = _probe(out)
    assert p["size"] == (1080, 1920)
    assert abs(p["duration"] - (10 - 3.5 + 0.3)) < 0.35, "the 3.5 s pause is out, a little air kept"


@needs_ffmpeg
@pytest.mark.parametrize("name", ["Don't stop_clip", "talk.v2_clip1", "a=b,c;d_clip"])
def test_subtitles_whatever_the_clip_is_called(media, tmp_path, name):
    # The clip's parts lie in a folder named after it, and the subtitle
    # file's path goes into ffmpeg's filter graph: an apostrophe ended the
    # quote there. And a dot in the name was taken for its extension.
    from whisper_tm.core import clip_render as cr

    s = {"format": "orig", "subs": True, "fade": False}
    parts = cr.plan(media / "talk.mp4", [{"start": 0, "end": 3}], s)
    lines = [{"start": 0.5, "end": 2.5, "text": "We start here.", "name": ""}]
    out = cr.render(media / "talk.mp4", parts, tmp_path / name, s, lines)
    assert out.name == name + ".mp4" and _probe(out)["size"] == (640, 360)
    assert [x.name for x in tmp_path.iterdir()] == [name + ".mp4"]


@needs_ffmpeg
def test_one_clip_each_of_a_name_with_dots_stay_apart(media, tmp_path):
    from whisper_tm.core import clip_render as cr

    s = {"format": "orig", "fade": False}
    outs = [
        cr.render(
            media / "talk.mp4",
            cr.plan(media / "talk.mp4", [{"start": i, "end": i + 1}], s),
            tmp_path / f"talk.v2_clip{i + 1}",
            s,
            None,
        )
        for i in range(2)
    ]
    assert [o.name for o in outs] == ["talk.v2_clip1.mp4", "talk.v2_clip2.mp4"]
    assert all(o.exists() for o in outs)


@needs_ffmpeg
def test_sound_only_becomes_m4a(media, tmp_path):
    from whisper_tm.core import clip_render as cr

    s = {"format": "square", "fade": True}
    out = cr.render(
        media / "memo.m4a",
        cr.plan(media / "memo.m4a", [{"start": 0, "end": 2}], s),
        tmp_path / "m",
        s,
        None,
    )
    p = _probe(out)
    assert out.suffix == ".m4a" and p["video"] is None and abs(p["duration"] - 2) < 0.15


@needs_ffmpeg
def test_cancel_cleans_up(media, tmp_path):
    from whisper_tm.core import clip_render as cr

    s = {"format": "vert"}
    parts = cr.plan(media / "talk.mp4", [{"start": 0, "end": 4}, {"start": 5, "end": 9}], s)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(cr.ClipError):
        cr.render(media / "talk.mp4", parts, tmp_path / "x", s, None, should_stop=stop)
    assert list(tmp_path.iterdir()) == [], "no clip and no parts left"


# ── API and the job ───────────────────────────────────────────────────


@pytest.fixture
def client(db, media, monkeypatch, tmp_path):
    import whisper_tm.core.security as sec
    from whisper_tm.api import clips, export
    from whisper_tm.core import export_jobs

    monkeypatch.setattr(sec, "allowed_roots", lambda: [media])
    monkeypatch.setattr(
        export_jobs,
        "folder",
        lambda: (tmp_path / "exports").mkdir(exist_ok=True) or tmp_path / "exports",
    )
    db.upsert_transcript(
        "talk.mp4",
        {
            "complete": True,
            "language": "en",
            "total_duration_sec": 10,
            "segments": [
                {"start": 0.5, "end": 2.5, "speaker": "A", "text": "We start here."},
                {"start": 7, "end": 9, "speaker": "B", "text": "And end here."},
            ],
        },
        media_dir=str(media),
        abs_path=str(media / "talk.mp4"),
    )
    app = Flask(__name__)
    app.register_blueprint(clips.clips_bp)
    app.register_blueprint(export.export_bp)
    return app.test_client()


def _wait(jid, timeout=60):
    from whisper_tm.core import export_jobs

    end = time.time() + timeout
    while time.time() < end:
        j = export_jobs.get(jid)
        if j.status not in ("waiting", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_cuts_through_the_api_and_into_export(client, media):
    f = str(media / "talk.mp4")
    assert client.get("/api/clips?file=" + f).get_json()["cuts"] == []
    r = client.put(
        "/api/clips",
        json={"file": f, "cuts": [{"start": 6.8, "end": 9.5}], "settings": {"subs": True}},
    )
    assert r.status_code == 200 and r.get_json()["settings"]["subs"] is True
    assert client.get("/api/clips?file=" + f).get_json()["cuts"] == [{"start": 6.8, "end": 9.5}], (
        "survives a reload"
    )
    t = client.post(
        "/api/export", json={"files": [f], "format": "txt", "options": {"part": "cuts"}}
    ).data.decode()
    assert "And end here." in t and "We start here." not in t, (
        "Export's 'Only the clip cuts' uses them"
    )
    assert client.get("/api/clips?file=/etc/passwd").status_code == 400
    assert client.put("/api/clips", json={"file": f, "cuts": "x"}).status_code == 400


@needs_ffmpeg
def test_make_the_clip_one_each(client, media):
    f = str(media / "talk.mp4")
    client.put(
        "/api/clips",
        json={
            "file": f,
            "cuts": [{"start": 0, "end": 2}, {"start": 7, "end": 9}],
            "settings": {"stitch": "each", "format": "square", "subs": True},
        },
    )
    r = client.post("/api/clips/render", json={"file": f})
    assert r.status_code == 202
    job = r.get_json()["job"]
    assert job["kind"] == "clip" and "Clip · talk.mp4 · 0:04 · square · 2 clips" == job["label"]
    j = _wait(job["id"])
    assert j.status == "done" and j.result == "talk_clips.zip", j.error
    from whisper_tm.core import export_jobs

    names = sorted(zipfile.ZipFile(export_jobs.result_path(j.id)).namelist())
    assert names == ["talk_clip1.mp4", "talk_clip2.mp4"]
    made = client.get("/api/clips?file=" + f).get_json()["made"]
    assert made[0]["n"] == 2 and made[0]["job"] == j.id and made[0]["format"] == "square"


def test_no_cuts_no_clip(client, media):
    assert (
        client.post("/api/clips/render", json={"file": str(media / "talk.mp4")}).status_code == 400
    )


# ── Best moments ──────────────────────────────────────────────────────


def test_long_transcripts_are_joined_until_they_fit():
    from whisper_tm.core import best_moments as bm

    segs = [
        {"start": i * 5, "end": i * 5 + 4, "speaker": "A", "text": "word " * 20} for i in range(200)
    ]
    rows = bm.lines_of(segs, 6000)
    assert (
        sum(len(r["text"]) + 14 for r in rows) <= 6000
        and rows[0]["start"] == 0
        and rows[-1]["end"] == 199 * 5 + 4
    )


def test_the_answer_becomes_checked_cuts():
    from whisper_tm.core import best_moments as bm

    rows = [{"start": i * 10, "end": i * 10 + 9, "text": "x"} for i in range(30)]
    got = bm.moments_from(
        {
            "moments": [
                {"first": 2, "last": 4, "title": "A start"},
                {"first": 3, "last": 3, "title": "overlaps"},
                {"first": 10, "last": 29, "title": "far too long"},
                {"first": 99, "last": 120, "title": "not there"},
                {"first": "x", "last": 1, "title": "bad"},
            ]
        },
        rows,
    )
    assert got == [
        {"start": 20, "end": 49, "title": "A start", "ai": True},
        {"start": 100, "end": 190, "title": "far too long", "ai": True},
    ]
    one_line = bm.moments_from({"moments": [{"first": 5, "last": 5, "title": "short"}]}, rows)
    assert (one_line[0]["start"], one_line[0]["end"]) == (50, 69), "a single line takes the next in"
    assert bm.moments_from({}, rows) == [] and bm.moments_from({"moments": None}, rows) == []


class _Q:
    def __init__(self):
        self.msgs = []

    def progress(self, eid, pct, msg):
        self.msgs.append(msg)

    def abort_requested(self, eid):
        return False


def _seed(db):
    db.upsert_transcript(
        "a.mp4",
        {
            "complete": True,
            "segments": [
                {"start": i * 6, "end": i * 6 + 5, "speaker": "S", "text": f"Line {i}."}
                for i in range(20)
            ],
        },
        media_dir=ROOT,
        abs_path=ROOT + "/a.mp4",
    )


def test_best_moments_add_cuts(db, monkeypatch):
    from whisper_tm.core import best_moments as bm
    from whisper_tm.core import cuts, llm_client

    _seed(db)
    monkeypatch.setattr(
        llm_client,
        "chat_json",
        lambda *a, **k: {
            "moments": [
                {"first": 1, "last": 3, "title": "One"},
                {"first": 10, "last": 12, "title": "Two"},
            ]
        },
    )
    monkeypatch.setattr(llm_client, "effective_context", lambda task="": 8192)
    out = bm.run({"id": 1, "filename": "a.mp4", "params": {"abs": ROOT + "/a.mp4"}}, _Q())
    assert [m["title"] for m in out["moments"]] == ["One", "Two"]
    assert [(c["start"], c["end"], c.get("ai")) for c in cuts.get(ROOT + "/a.mp4")] == [
        (6, 23, True),
        (60, 77, True),
    ]


def test_an_empty_answer_says_so(db, monkeypatch):
    from whisper_tm.core import best_moments as bm
    from whisper_tm.core import cuts, llm_client

    _seed(db)
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: {"moments": []})
    with pytest.raises(RuntimeError, match="no usable moments"):
        bm.run({"id": 1, "filename": "a.mp4", "params": {"abs": ROOT + "/a.mp4"}}, _Q())
    assert cuts.get(ROOT + "/a.mp4") == []


# ── The tab ───────────────────────────────────────────────────────────


def test_the_clips_tab_is_the_editor_and_the_old_cut_button_is_gone():
    static = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    assert "clips.js" in html and "clips.css" in html
    assert "cutClip()" not in html and "Cut & download" not in html
    rec = (static / "js" / "recording.js").read_text(encoding="utf-8")
    assert "Clips.show()" in rec and "Clips.count()" in rec
    js = (static / "js" / "clips.js").read_text(encoding="utf-8")
    for call in ("/api/clips", "/api/clips/render", "/api/clips/best", "pickLines", "add30"):
        assert call in js, call

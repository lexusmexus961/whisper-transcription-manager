"""Links in — YouTube videos to transcripts, summaries, analyses (core/ingest.py).

Wherever there is a YouTube video it can be selected — not just one but
several — and sent to be transcribed (saved in the uploads folder), or
transcribed and summarized.

What these tests pin:

* a link's video id, whatever shape the link has; a playlist has none;
* nothing twice: a video downloaded before is found by its id, or — from
  before the ids were noted — by the name its title gets in the upload folder;
* a batch: every link resolved (a playlist into its videos), each video
  downloaded once and queued at the front — with the summary option for
  "summarize"; one transcribed before is not queued again, and gets its
  summary in the LLM queue unless it has one; one download failing doesn't
  stop the others; cancel;
* "analyze": one analysis over the videos, waiting for their transcripts and
  queued by itself once they are there;
* every video's state read live — queued, transcribing, transcribed,
  summarized (with the summary), failed — and the link that opens it;
* the routes, and which videos of a page are in the library already;
* how a video is done: videos from the browser have defaults of their own —
  the language detected, not the library's — a send sets what it wants over
  them, empty is the general setting; a template and a language reach the
  summary, also one made in the LLM queue; "again" makes one more, and the
  one from before doesn't count until the new one is there;
* audio only — asked for with the send, a default, or the setting; a video
  taken out of its batch is not fetched, stops downloading, or leaves the
  queue it was put in (never an entry the batch didn't make);
* the side panel's reading: a video's transcript and summaries, and a
  conversation about it with its transcript attached — the same one again.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import pytest

from whisper_tm.core import analysis, ingest, ytdl
from whisper_tm.core import queue as q


@pytest.fixture
def lib(tmp_path, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.config import config
    from whisper_tm.database import Database

    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    uploads = tmp_path / "media" / "uploads"
    uploads.mkdir(parents=True)
    monkeypatch.setattr(type(config), "upload_dir", property(lambda self: uploads))
    monkeypatch.setattr(analysis, "_roots", lambda: [str(tmp_path / "media")])
    monkeypatch.setattr(ytdl, "available", lambda: True)
    # An LLM queue worker another test's app started may still run — it reads
    # whatever database is current and would take up this test's entries
    from whisper_tm.core.llm_queue import llm_queue

    monkeypatch.setattr(llm_queue, "_paused", True)
    # … and the watcher for waiting analyses: the tests drive it themselves
    monkeypatch.setattr(analysis, "_watch_enabled", False)
    # A language model is set up (a summary or an analysis needs one)
    from whisper_tm.core import llm_client

    monkeypatch.setattr(llm_client, "ready", lambda task="": True)
    queued: list = []

    def enqueue(filenames, settings, base_dir=None, dir_map=None, front=False):
        queued.append((list(filenames), dict(settings), base_dir, front))
        out = []
        with db.connection() as c:
            for f in filenames:
                cur = c.execute(
                    "INSERT INTO queue_items (filename, base_dir, settings_json, status) "
                    "VALUES (?, ?, ?, 'pending')",
                    (f, base_dir, json.dumps(settings)),
                )
                out.append({"filename": f, "id": cur.lastrowid})
        return out

    monkeypatch.setattr(q, "enqueue", enqueue)
    # Not the real run_now either: the queue worker another test's app
    # started waits paused (tests/conftest.py) — run_now would wake it, and it
    # would take this test's entries from this test's database, until the
    # teardown closes that database under it (a segfault)
    run_now: list = []
    monkeypatch.setattr(q, "run_now", lambda ids: run_now.append(list(ids)) or len(ids))
    downloads: list = []

    def download(job, entry, tmp_dir, dest_dir, report_it=None, audio_only=None):
        if entry["id"] == "broken":
            raise RuntimeError("HTTP Error 403: Forbidden")
        downloads.append(entry["id"])
        if report_it:
            report_it(0.5)
        target = Path(dest_dir) / f"{entry['title'].replace(' ', '_')}.m4a"
        target.write_bytes(b"x" * 2000)
        return target

    monkeypatch.setattr(ytdl, "_download_one", download)
    monkeypatch.setattr(
        ytdl,
        "enumerate_entries",
        lambda url: [
            {"url": url, "title": "Video " + ingest.video_id(url), "id": ingest.video_id(url)}
        ],
    )
    analysis._deleted.clear()
    yield {
        "db": db,
        "uploads": uploads,
        "queued": queued,
        "downloads": downloads,
        "run_now": run_now,
        "tmp": tmp_path,
    }
    db.close_all()


def _run(b):
    """The worker, here and now instead of in its thread."""
    stop = threading.Event()
    ingest._running[b["id"]] = stop
    ingest._work(b["id"], stop)
    return ingest.load(b["id"])


def _start(monkeypatch, *args, **kw):
    """start() without its thread — the test runs the worker itself."""
    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", lambda self: None)
        return ingest.start(*args, **kw)


def _transcribed(db, abs_path, text="Hello there."):
    rel = Path(abs_path).name
    db.upsert_transcript(
        rel,
        {
            "complete": True,
            "segments": [{"start": 0.0, "end": 2.0, "text": text, "speaker": "SPEAKER_00"}],
        },
        media_dir=str(Path(abs_path).parent),
        abs_path=abs_path,
    )


YT = "https://www.youtube.com/watch?v="


@pytest.mark.parametrize(
    "url, vid",
    [
        (YT + "abc123XYZ_-", "abc123XYZ_-"),
        (YT + "abc&t=345s", "abc"),
        ("https://youtu.be/abc?si=x", "abc"),
        ("https://www.youtube.com/shorts/abc", "abc"),
        ("https://m.youtube.com/watch?v=abc", "abc"),
        (YT + "abc&list=PL1", ""),  # the whole list, not one video
        ("https://example.com/video", ""),
    ],
)
def test_a_links_video_id(url, vid):
    assert ingest.video_id(url) == vid


class TestNothingTwice:
    def test_by_its_id(self, lib):
        f = lib["uploads"] / "Some_title.m4a"
        f.write_bytes(b"x")
        ingest._remember("abc", str(f), "Some title")
        assert ingest.find_existing("abc", "") == str(f)
        f.unlink()  # gone: not taken
        assert ingest.find_existing("abc", "") == ""

    def test_from_before_by_the_name_its_title_gets(self, lib):
        (lib["uploads"] / "How_Fast_Is_It_Really.m4a").write_bytes(b"x")
        (lib["uploads"] / "Other_1.mp4").write_bytes(b"x")
        (lib["uploads"] / "notes.txt").write_bytes(b"x")
        found = ingest.find_existing("", "How Fast Is It Really?")
        assert found.endswith("/uploads/How_Fast_Is_It_Really.m4a")
        assert ingest.find_existing("", "Other").endswith("/uploads/Other_1.mp4")
        assert ingest.find_existing("", "notes") == ""  # not a recording
        assert ingest.find_existing("", "Never seen") == ""


class TestABatch:
    def test_refused(self, lib):
        with pytest.raises(ingest.IngestError, match="No links"):
            ingest.start(["not a link", ""])
        with pytest.raises(ingest.IngestError, match="Unknown mode"):
            ingest.start([YT + "a"], "burn")
        with pytest.raises(ingest.IngestError, match="needs a task"):
            ingest.start([YT + "a"], "analyze")

    def test_without_a_language_model_only_transcribe(self, lib, monkeypatch):
        """Taken without one, the videos were fetched and transcribed, and
        the summary failed where the extension never showed it."""
        from whisper_tm.core import llm_client

        monkeypatch.setattr(llm_client, "ready", lambda task="": False)
        for mode, task in (("summarize", ""), ("analyze", "compare them")):
            with pytest.raises(ingest.IngestError, match="No language model is set up"):
                ingest.start([YT + "a"], mode, task=task)
        assert _start(monkeypatch, [YT + "a"], "transcribe")["mode"] == "transcribe"

    def test_downloaded_once_and_queued_at_the_front(self, lib, monkeypatch):
        b = _start(monkeypatch, [YT + "aaa", YT + "aaa&t=5s", YT + "bbb"])
        assert [it["video_id"] for it in b["items"]] == ["aaa", "bbb"]  # the same video once
        b = _run(b)
        assert lib["downloads"] == ["aaa", "bbb"]
        up = str(lib["uploads"])
        # The language detected: a video from the browser can speak anything
        auto = {"language": "auto"}
        assert lib["queued"] == [
            (["Video_aaa.m4a"], auto, up, True),
            (["Video_bbb.m4a"], auto, up, True),
        ]
        assert b["status"] == "downloaded"
        # Noted by id: the same video again is not downloaded again
        b2 = _run(_start(monkeypatch, [YT + "bbb"]))
        assert lib["downloads"] == ["aaa", "bbb"] and b2["items"][0]["known"] is True
        # … and while it waits in the queue it is pulled forward, not added twice
        assert lib["queued"][-1] == (["Video_bbb.m4a"], auto, up, True)

    def test_a_paused_queue_runs_the_video_anyway(self, lib, monkeypatch):
        """When the queue is paused, only the new video runs."""
        _run(_start(monkeypatch, [YT + "aaa"]))
        with lib["db"].connection() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM queue_items")]
        assert lib["run_now"] == [ids] and len(ids) == 1

    def test_summarize_asks_the_queue_for_the_summary(self, lib, monkeypatch):
        _run(_start(monkeypatch, [YT + "aaa"], "summarize"))
        assert lib["queued"][0][1] == {"summarize": True, "language": "auto"}

    def test_an_own_instruction_goes_both_ways(self, lib, monkeypatch):
        """A custom instruction goes with the transcription in the queue, and
        to the LLM queue for a video transcribed before."""
        opts = {"summary_prompt": "Only the tips, as a list."}
        _run(_start(monkeypatch, [YT + "aaa"], "summarize", options=opts))
        assert lib["queued"][0][1]["summarize_prompt"] == "Only the tips, as a list."
        f = lib["uploads"] / "Video_ccc.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f))
        _run(_start(monkeypatch, [YT + "ccc"], "summarize", options=opts))
        with lib["db"].connection() as c:
            p = json.loads(c.execute("SELECT params_json FROM llm_queue").fetchone()[0])
        assert p["system_prompt"] == "Only the tips, as a list."
        assert p["label"] == "Custom (browser extension)"

    def test_one_transcribed_before_is_summarized_in_the_llm_queue(self, lib, monkeypatch):
        f = lib["uploads"] / "Video_ccc.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f))
        _run(_start(monkeypatch, [YT + "ccc"], "summarize"))
        assert lib["downloads"] == [] and lib["queued"] == []  # found by its title
        with lib["db"].connection() as c:
            r = c.execute("SELECT kind, filename, params_json FROM llm_queue").fetchone()
        assert r["kind"] == "summary" and r["filename"] == "Video_ccc.m4a"
        assert json.loads(r["params_json"])["media_dir"] == str(lib["uploads"])
        # Once more: a summary waiting is not asked for twice
        _run(_start(monkeypatch, [YT + "ccc"], "summarize"))
        with lib["db"].connection() as c:
            assert c.execute("SELECT COUNT(*) FROM llm_queue").fetchone()[0] == 1

    def test_transcribed_and_not_summarizing_is_left_alone(self, lib, monkeypatch):
        f = lib["uploads"] / "Video_ddd.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f))
        _run(_start(monkeypatch, [YT + "ddd"]))
        assert lib["queued"] == [] and lib["downloads"] == []

    def test_a_playlist_brings_its_videos(self, lib, monkeypatch):
        monkeypatch.setattr(
            ytdl,
            "enumerate_entries",
            lambda url: [
                {"url": "x1", "title": "One", "id": "x1"},
                {"url": "x2", "title": "Two", "id": "x2"},
            ],
        )
        b = _run(_start(monkeypatch, [YT + "aaa&list=PL1"]))
        assert [it["video_id"] for it in b["items"]] == ["x1", "x2"]
        assert (
            b["items"][0]["url"] == YT + "x1" and b["items"][0]["playlist"] == YT + "aaa&list=PL1"
        )

    def test_one_failing_doesnt_stop_the_others(self, lib, monkeypatch):
        b = _run(_start(monkeypatch, [YT + "broken", YT + "eee"]))
        assert "403" in b["items"][0]["error"] and "nightly" in b["items"][0]["error"]
        assert lib["downloads"] == ["eee"]
        states = [it["state"] for it in ingest.public(b)["items"]]
        assert states == ["error", "queued"]

    def test_cancel(self, lib, monkeypatch):
        b = _start(monkeypatch, [YT + "aaa", YT + "bbb"])
        stop = threading.Event()
        stop.set()
        ingest._running[b["id"]] = stop
        ingest._work(b["id"], stop)
        b = ingest.load(b["id"])
        assert b["status"] == "cancelled" and lib["downloads"] == []
        assert ingest.public(b)["status"] == "cancelled"


class TestLiveStates:
    def test_from_queue_to_summary(self, lib, monkeypatch):
        b = _run(_start(monkeypatch, [YT + "aaa"], "summarize"))
        it = ingest.public(b)["items"][0]
        assert it["state"] == "queued" and it["open"].startswith("/?open=/")
        assert ingest.public(b)["status"] == "working"
        db = lib["db"]
        with db.connection() as c:
            c.execute("UPDATE queue_items SET status='processing'")
        assert ingest.public(b)["items"][0]["state"] == "transcribing"
        abs_path = b["items"][0]["abs_path"]
        _transcribed(db, abs_path)
        with db.connection() as c:
            c.execute("UPDATE queue_items SET status='done'")
        assert ingest.public(b)["items"][0]["state"] == "transcribed"
        db.replace_summaries(
            Path(abs_path).name,
            [{"text": "It is about speed.", "prompt": "", "timestamp": time.time()}],
            media_dir=str(lib["uploads"]),
        )
        out = ingest.public(b)
        assert out["items"][0]["state"] == "summarized"
        assert out["items"][0]["summary"] == "It is about speed."
        assert out["status"] == "done"

    def test_a_failed_transcription(self, lib, monkeypatch):
        b = _run(_start(monkeypatch, [YT + "aaa"]))
        with lib["db"].connection() as c:
            c.execute("UPDATE queue_items SET status='error', error='No audio track'")
        it = ingest.public(b)["items"][0]
        assert it["state"] == "error" and it["error"] == "No audio track"


class TestAnalyze:
    def test_it_waits_for_the_transcripts_then_goes(self, lib, monkeypatch):
        b = _run(
            _start(
                monkeypatch,
                [YT + "aaa", YT + "bbb"],
                "analyze",
                task="Compare the two.",
                title="Two videos",
            )
        )
        a = analysis.load(b["analysis_id"])
        assert a["status"] == "waiting" and a["title"] == "Two videos"
        assert a["label"] == "2 YouTube videos" and len(analysis.files_of(a["id"])) == 2
        assert ingest.public(b)["analysis"]["status"] == "waiting"
        assert analysis.start_waiting() == 0  # still in the queue
        for it in b["items"]:
            _transcribed(lib["db"], it["abs_path"])
        with lib["db"].connection() as c:
            c.execute("UPDATE queue_items SET status='done'")
        assert analysis.start_waiting() == 1
        a = analysis.load(a["id"])
        assert a["status"] == "queued" and a["entry_id"]
        with lib["db"].connection() as c:
            assert (
                c.execute("SELECT kind FROM llm_queue WHERE id=?", (a["entry_id"],)).fetchone()[0]
                == "analysis"
            )

    def test_a_waiting_analysis_can_be_cancelled(self, lib, monkeypatch):
        b = _run(_start(monkeypatch, [YT + "aaa"], "analyze", task="Sum up."))
        assert analysis.cancel(b["analysis_id"]) is True
        assert analysis.load(b["analysis_id"])["status"] == "cancelled"
        with lib["db"].connection() as c:
            c.execute("UPDATE queue_items SET status='done'")
        assert analysis.start_waiting() == 0

    def test_nothing_waits_longer_than_a_day(self, lib, monkeypatch):
        b = _run(_start(monkeypatch, [YT + "aaa"], "analyze", task="Sum up."))
        a = analysis.load(b["analysis_id"])
        a["created"] -= analysis.WAIT_MAX_SEC + 1
        analysis.save(a)
        assert analysis.start_waiting() == 1


@pytest.fixture
def client(lib):
    from flask import Flask

    from whisper_tm.api.ingest import ingest_bp

    app = Flask(__name__)
    app.register_blueprint(ingest_bp)
    return app.test_client()


class TestRoutes:
    def test_start_get_list(self, client, lib, monkeypatch):
        monkeypatch.setattr(threading.Thread, "start", lambda self: None)
        r = client.post("/api/ingest", json={"urls": [YT + "aaa"], "mode": "summarize"})
        d = r.get_json()
        assert r.status_code == 200 and d["ok"]
        bid = d["batch"]["id"]
        assert d["batch"]["items"][0]["state"] == "waiting" and d["batch"]["mode"] == "summarize"
        assert client.get(f"/api/ingest/{bid}").get_json()["id"] == bid
        assert client.get("/api/ingest").get_json()["batches"][0]["id"] == bid
        assert client.post(f"/api/ingest/{bid}/cancel").get_json()["ok"] is True

    def test_refused(self, client):
        assert client.post("/api/ingest", json={"urls": "nope"}).status_code == 400
        assert client.post("/api/ingest", json={"urls": 5}).status_code == 400
        assert client.post("/api/ingest", json={"urls": [YT + "a"], "mode": "x"}).status_code == 400
        assert client.get("/api/ingest/zzz").status_code == 404
        assert client.get("/api/ingest/aaaaaaaaaaaa").status_code == 404
        assert client.post("/api/ingest/known", json={"videos": "x"}).status_code == 400

    def test_which_videos_are_there_already(self, client, lib):
        f = lib["uploads"] / "Talk_about_models.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f))
        g = lib["uploads"] / "Second.m4a"
        g.write_bytes(b"x")
        ingest._remember("sss", str(g), "Second")
        d = client.post(
            "/api/ingest/known",
            json={
                "videos": [
                    {"url": YT + "ttt", "title": "Talk about models"},
                    {"url": YT + "sss", "title": "whatever"},
                    {"url": YT + "new", "title": "New one"},
                ]
            },
        ).get_json()["videos"]
        assert [v["state"] for v in d] == ["transcribed", "downloaded", ""]
        assert "summary" not in d[0]
        # The playing video's button asks for the summary too
        f2 = lib["uploads"] / "Talk_about_models.m4a"
        lib["db"].replace_summaries(
            f2.name,
            [{"text": "It is about speed.", "prompt": "", "timestamp": time.time()}],
            media_dir=str(lib["uploads"]),
        )
        d2 = client.post(
            "/api/ingest/known",
            json={"summaries": True, "videos": [{"url": YT + "ttt", "title": "Talk about models"}]},
        ).get_json()["videos"]
        assert d2[0]["state"] == "summarized" and d2[0]["summary"] == "It is about speed."
        assert d[0]["open"].endswith("/uploads/Talk_about_models.m4a")
        assert d[1]["video_id"] == "sss" and "open" not in d[2]


def test_a_summary_for_a_file_in_its_own_folder(lib, monkeypatch):
    """The LLM queue's summary of a video in the upload folder — not below
    the folder open in the app — reads and saves it there."""
    from whisper_tm.core import llm_tasks
    from whisper_tm.core import summarize as s

    f = lib["uploads"] / "Video_x.m4a"
    f.write_bytes(b"x")
    _transcribed(lib["db"], str(f), "Models got faster.")
    monkeypatch.setattr(
        s,
        "summary_stream",
        lambda segments, **kw: iter([{"text": "About: " + segments[0]["text"]}]),
    )

    class Queue:
        def partial_text(self, *a):
            pass

        def progress(self, *a):
            pass

        def partial_result(self, *a):
            pass

        def abort_requested(self, *a):
            return False

    out = llm_tasks._summary(
        {"id": 1, "filename": "Video_x.m4a", "params": {"media_dir": str(lib["uploads"])}}, Queue()
    )
    assert out["text"] == "About: Models got faster." and out["saved"]
    with lib["db"].connection() as c:
        r = c.execute("SELECT abs_path, text FROM summaries").fetchone()
    assert r["abs_path"] == str(f) and r["text"] == "About: Models got faster."


# ══════════════════════════════════════════════════════════════════════════
#  How a video is done
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cfg(monkeypatch):
    """The defaults as they are — what a test saves is undone, and nothing
    is written."""
    from whisper_tm.config import config

    before = copy.deepcopy(config.get("ingest", "defaults", default={}))
    monkeypatch.setattr(config, "save", lambda *a, **k: None)
    yield config
    config.set("ingest", "defaults", before)


@pytest.fixture
def choices(monkeypatch):
    """The vocabularies, templates and models the settings have."""
    monkeypatch.setattr(
        ingest,
        "_names",
        lambda section, key: {
            "vocab_presets": ["Gaming"],
            "presets": ["Summary", "Bullet Points"],
            "models": ["fast"],
        }[key],
    )


class TestHowItIsDone:
    def test_videos_from_the_browser_detect_their_language(self, cfg):
        d = ingest.defaults()
        assert d["language"] == "auto"
        assert d["model"] == "" and d["diarize"] is None and d["numSpeakers"] is None
        assert d["summary_preset"] == "" and d["summary_language"] == ""

    def test_only_what_is_one_of_the_choices(self, choices):
        out = ingest.clean_options(
            {
                "model": "large-v3",
                "language": "xx",
                "diarize": "yes",
                "vocal_isolation": False,
                "numSpeakers": "3",
                "vocab_preset": "Gaming",
                "summary_preset": "Nope",
                "summary_language": "de",
                "summary_model": "fast",
                "burn": True,
            }
        )
        assert out == {
            "model": "large-v3",
            "language": "",
            "diarize": None,
            "vocal_isolation": False,
            "numSpeakers": 3,
            "vocab_preset": "Gaming",
            "summary_preset": "",
            "summary_language": "de",
            "summary_model": "fast",
        }
        assert ingest.clean_options({"numSpeakers": 99}) == {"numSpeakers": None}
        assert ingest.clean_options({"language": "auto"}) == {"language": "auto"}

    def test_an_own_instruction_is_free_text(self):
        """Custom: written by the user, not one of the choices."""
        assert ingest.clean_options({"summary_prompt": "  Only the tips.  "}) == {
            "summary_prompt": "Only the tips."
        }
        long = ingest.clean_options({"summary_prompt": "y" * 5000})["summary_prompt"]
        assert len(long) == ingest.PROMPT_MAX

    def test_the_defaults_are_kept(self, cfg, choices):
        ingest.save_defaults({"summary_preset": "Bullet Points", "summary_language": "de"})
        ingest.save_defaults({"model": "turbo"})  # the others stay
        d = ingest.defaults()
        assert (d["summary_preset"], d["summary_language"], d["model"]) == (
            "Bullet Points",
            "de",
            "turbo",
        )
        assert d["language"] == "auto"

    def test_a_send_over_the_defaults(self, lib, cfg, choices, monkeypatch):
        ingest.save_defaults({"summary_preset": "Bullet Points", "diarize": True})
        _run(
            _start(
                monkeypatch,
                [YT + "aaa"],
                "summarize",
                options={
                    "model": "large-v3",
                    "language": "en",
                    "numSpeakers": 2,
                    "summary_language": "de",
                },
            )
        )
        assert lib["queued"][0][1] == {
            "model": "large-v3",
            "language": "en",
            "diarize": True,
            "numSpeakers": 2,
            "summarize": True,
            "summarize_preset": "Bullet Points",
            "summarize_language": "de",
        }

    def test_empty_is_the_general_setting(self, lib, cfg, choices, monkeypatch):
        _run(_start(monkeypatch, [YT + "aaa"], options={"language": "", "diarize": None}))
        assert lib["queued"][0][1] == {}

    def test_one_transcribed_before_gets_its_summary_as_asked(self, lib, cfg, choices, monkeypatch):
        f = lib["uploads"] / "Video_ccc.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f))
        _run(
            _start(
                monkeypatch,
                [YT + "ccc"],
                "summarize",
                options={"summary_preset": "Bullet Points", "summary_language": "de"},
            )
        )
        with lib["db"].connection() as c:
            params = json.loads(c.execute("SELECT params_json FROM llm_queue").fetchone()[0])
        assert params["preset"] == "Bullet Points" and params["language"] == "de"
        assert params["label"] == "Bullet Points · German (browser extension)"

    def _summarized(self, lib):
        f = lib["uploads"] / "Video_ccc.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f))
        lib["db"].replace_summaries(
            f.name,
            [{"text": "Old.", "prompt": "", "timestamp": time.time() - 60}],
            media_dir=str(lib["uploads"]),
        )
        return f

    def test_once_more_when_asked_again(self, lib, cfg, choices, monkeypatch):
        f = self._summarized(lib)
        b = _run(_start(monkeypatch, [YT + "ccc"], "summarize"))  # not again: as it is
        assert ingest.public(b)["items"][0]["summary"] == "Old."
        with lib["db"].connection() as c:
            assert c.execute("SELECT COUNT(*) FROM llm_queue").fetchone()[0] == 0
        b = _run(
            _start(
                monkeypatch,
                [YT + "ccc"],
                "summarize",
                again=True,
                options={"summary_preset": "Bullet Points"},
            )
        )
        with lib["db"].connection() as c:
            assert c.execute("SELECT COUNT(*) FROM llm_queue").fetchone()[0] == 1
        # The one from before doesn't count …
        assert ingest.public(b)["items"][0]["state"] == "summarizing"
        with lib["db"].connection() as c:
            c.execute("UPDATE llm_queue SET status='done'")
        lib["db"].replace_summaries(
            f.name,
            [
                {"text": "Old.", "prompt": "", "timestamp": time.time() - 60},
                {"text": "New.", "prompt": "Bullet Points", "timestamp": time.time() + 1},
            ],
            media_dir=str(lib["uploads"]),
        )
        # … the new one does
        it = ingest.public(ingest.load(b["id"]))["items"][0]
        assert it["state"] == "summarized" and it["summary"] == "New."

    def test_a_new_summary_that_failed(self, lib, cfg, choices, monkeypatch):
        self._summarized(lib)
        b = _run(_start(monkeypatch, [YT + "ccc"], "summarize", again=True))
        with lib["db"].connection() as c:
            c.execute("UPDATE llm_queue SET status='error', error='LLM error 500'")
        it = ingest.public(ingest.load(b["id"]))["items"][0]
        assert it["state"] == "error" and "new summary failed" in it["error"]


def test_a_summary_in_the_language_asked_for(lib, monkeypatch):
    """However the template ends ("in the language of the transcript"), a
    summary asked for in German is told so — in the LLM queue, and right
    after a transcription."""
    from whisper_tm.core import llm_tasks
    from whisper_tm.core import summarize as s

    f = lib["uploads"] / "Video_x.m4a"
    f.write_bytes(b"x")
    _transcribed(lib["db"], str(f), "Models got faster.")
    seen = {}

    def stream(segments, **kw):
        seen.update(kw)
        return iter([{"text": "Models got faster."}])

    monkeypatch.setattr(s, "summary_stream", stream)

    class Queue:
        def partial_text(self, *a):
            pass

        def progress(self, *a):
            pass

        def partial_result(self, *a):
            pass

        def abort_requested(self, *a):
            return False

    llm_tasks._summary(
        {
            "id": 1,
            "filename": "Video_x.m4a",
            "params": {"media_dir": str(lib["uploads"]), "language": "de"},
        },
        Queue(),
    )
    assert seen["system_prompt"].endswith(
        "Write the summary in German, whatever the language of the transcript."
    )
    assert s.language_rule("") == s.language_rule("xx") == ""
    asked = {}
    from whisper_tm.core.llm_queue import llm_queue

    monkeypatch.setitem(q.config._data.setdefault("llm", {}), "endpoint", "http://llm:8080")
    monkeypatch.setattr(llm_queue, "enqueue", lambda kind, f, params: asked.update(params) or 1)
    q._queue_summary("Video_x.m4a", {"summarize_language": "de"}, str(lib["uploads"]))
    assert asked["language"] == "de"
    q._queue_summary("Video_x.m4a", {"summarize_prompt": "Only the jokes."}, str(lib["uploads"]))
    assert asked["system_prompt"] == "Only the jokes." and asked["label"] == "Custom"


# ══════════════════════════════════════════════════════════════════════════
#  Read and asked about
# ══════════════════════════════════════════════════════════════════════════


class TestReadAndAsked:
    def _talk(self, lib):
        f = lib["uploads"] / "Talk.m4a"
        f.write_bytes(b"x")
        _transcribed(lib["db"], str(f), "Models got faster.")
        lib["db"].replace_summaries(
            f.name,
            [
                {"text": "First.", "prompt": "Summary", "timestamp": 1.0},
                {"text": "It is about speed.", "prompt": "Bullet Points", "timestamp": 2.0},
            ],
            media_dir=str(lib["uploads"]),
        )
        ingest._remember("ttt", str(f), "A talk")
        return f

    def test_the_transcript_and_the_summaries(self, lib):
        self._talk(lib)
        d = ingest.transcript(YT + "ttt")
        assert d["title"] == "A talk" and d["open"].endswith("/uploads/Talk.m4a")
        assert d["segments"] == [
            {"start": 0.0, "end": 2.0, "text": "Models got faster.", "speaker": "SPEAKER_00"}
        ]
        assert [x["name"] for x in d["summaries"]] == ["Bullet Points", "Summary"]  # newest first
        with pytest.raises(ingest.IngestError, match="not in the library"):
            ingest.transcript(YT + "nope")

    def test_a_conversation_about_the_video(self, lib):
        from whisper_tm.core import chat

        self._talk(lib)
        d = ingest.ask(YT + "ttt")
        assert d["fresh"] is True
        conv = chat.load(d["conversation"])
        att = conv["attached"]
        assert att["kind"] == "video" and att["title"] == "A talk" and att["url"] == YT + "ttt"
        assert "[0:00:00 | SPEAKER_00] Models got faster." in att["text"]
        assert "Its summary:\n\nIt is about speed." in att["text"]
        assert conv["title"] == "About: A talk" and "is attached" in conv["turns"][0]["text"]
        # The same video again: the same conversation — or a new one
        assert ingest.ask(YT + "ttt") == {"conversation": d["conversation"], "fresh": False}
        assert ingest.ask(YT + "ttt", new=True)["conversation"] != d["conversation"]

    def test_not_before_it_is_transcribed(self, lib):
        f = lib["uploads"] / "Raw.m4a"
        f.write_bytes(b"x")
        ingest._remember("rrr", str(f), "Raw")
        with pytest.raises(ingest.IngestError, match="not transcribed yet"):
            ingest.ask(YT + "rrr")

    def test_the_side_panels_routes(self, client, lib, cfg, choices):
        d = client.get("/api/ingest/options").get_json()
        assert {"turbo", "large-v3"} <= {m["id"] for m in d["whisper_models"]}
        assert {"id": "de", "label": "German"} in d["languages"]
        assert d["summary_presets"] == ["Summary", "Bullet Points"]
        assert d["defaults"]["language"] == "auto" and "model" in d["general"]
        r = client.post("/api/ingest/defaults", json={"summary_language": "de"}).get_json()
        assert r["ok"] and r["defaults"]["summary_language"] == "de"
        self._talk(lib)
        t = client.post("/api/ingest/transcript", json={"url": YT + "ttt"}).get_json()
        assert t["segments"][0]["text"] == "Models got faster."
        a = client.post("/api/ingest/ask", json={"url": YT + "ttt"}).get_json()
        assert a["fresh"] is True and a["conversation"]
        assert client.post("/api/ingest/transcript", json={"url": YT + "zzz"}).status_code == 404
        assert client.post("/api/ingest/ask", json={"url": YT + "zzz"}).status_code == 404
        assert (
            client.post("/api/ingest", json={"urls": [YT + "a"], "options": "x"}).status_code == 400
        )


class TestAudioOnly:
    """Only the sound downloaded — asked for with a send, or a default."""

    @pytest.fixture
    def asked(self, lib, monkeypatch):
        seen: list = []
        real = ytdl._download_one

        def download(job, entry, tmp_dir, dest_dir, report_it=None, audio_only=None):
            seen.append(audio_only)
            return real(job, entry, tmp_dir, dest_dir, report_it=report_it, audio_only=audio_only)

        monkeypatch.setattr(ytdl, "_download_one", download)
        return seen

    def test_a_switch_like_the_others(self, cfg):
        assert ingest.clean_options({"audio_only": True}) == {"audio_only": True}
        assert ingest.clean_options({"audio_only": "yes"}) == {"audio_only": None}
        assert ingest.defaults()["audio_only"] is None
        assert ingest.general()["audio_only"] in (True, False)

    def test_the_send_the_default_the_setting(self, lib, cfg, asked, monkeypatch):
        _run(_start(monkeypatch, [YT + "aaa"], options={"audio_only": True}))
        ingest.save_defaults({"audio_only": False})
        _run(_start(monkeypatch, [YT + "bbb"]))
        ingest.save_defaults({"audio_only": None})
        _run(_start(monkeypatch, [YT + "ccc"]))
        # The route's own audio_only beats the option
        _run(_start(monkeypatch, [YT + "ddd"], audio_only=False, options={"audio_only": True}))
        assert asked == [True, False, None, False]
        # Not a setting of the transcription
        assert all("audio_only" not in s for _, s, _, _ in lib["queued"])


class TestTakenOut:
    """A video taken out of a batch (the side panel's ✕)."""

    def test_before_it_is_fetched(self, lib, monkeypatch):
        b = _start(monkeypatch, [YT + "aaa", YT + "bbb"])
        assert ingest.drop(b["id"], [YT + "aaa"]) == {"dropped": 1, "unqueued": 0}
        b = _run(b)
        assert lib["downloads"] == ["bbb"]
        assert [it["url"] for it in ingest.public(b)["items"]] == [YT + "bbb"]
        assert ingest.public(b)["status"] == "working"  # bbb waits in the queue
        # Taken out again, or one it never had: nothing happens
        assert ingest.drop(b["id"], [YT + "aaa", YT + "zzz"]) == {"dropped": 0, "unqueued": 0}

    def test_while_it_downloads(self, lib, monkeypatch):
        b = _start(monkeypatch, [YT + "aaa", YT + "bbb"])

        def download(job, entry, tmp_dir, dest_dir, report_it=None, audio_only=None):
            if entry["id"] == "aaa":
                ingest.drop(b["id"], [YT + "aaa"])
                assert job.cancel  # yt-dlp is stopped
                raise InterruptedError
            lib["downloads"].append(entry["id"])
            target = Path(dest_dir) / f"{entry['id']}.m4a"
            target.write_bytes(b"x")
            return target

        monkeypatch.setattr(ytdl, "_download_one", download)
        b = _run(b)
        assert lib["downloads"] == ["bbb"]  # the others go on
        assert b["status"] == "downloaded"
        assert [f for f, *_ in lib["queued"]] == [["bbb.m4a"]]

    def test_waiting_in_the_queue_it_leaves_it(self, lib, monkeypatch):
        b = _run(_start(monkeypatch, [YT + "aaa", YT + "bbb"]))
        assert ingest.drop(b["id"], [YT + "aaa"]) == {"dropped": 1, "unqueued": 1}
        with lib["db"].connection() as c:
            assert [r[0] for r in c.execute("SELECT filename FROM queue_items")] == [
                "Video_bbb.m4a"
            ]
        # All of them: the batch has nothing left to work on
        assert ingest.drop(b["id"]) == {"dropped": 1, "unqueued": 1}
        p = ingest.public(ingest.load(b["id"]))
        assert p["items"] == [] and p["status"] == "done"

    def test_found_in_the_library_its_queue_entry_stays(self, lib, monkeypatch):
        """Not put in the queue by this batch — the user's own entry stays."""
        _run(_start(monkeypatch, [YT + "aaa"]))
        b = _run(_start(monkeypatch, [YT + "aaa"]))
        assert b["items"][0]["known"] is True
        count = "SELECT COUNT(*) FROM queue_items WHERE status = 'pending'"
        with lib["db"].connection() as c:
            before = c.execute(count).fetchone()[0]
        assert ingest.drop(b["id"]) == {"dropped": 1, "unqueued": 0}
        with lib["db"].connection() as c:
            assert c.execute(count).fetchone()[0] == before > 0

    def test_the_route(self, client, lib, monkeypatch):
        b = _start(monkeypatch, [YT + "aaa", YT + "bbb"])
        r = client.post(f"/api/ingest/{b['id']}/drop", json={"urls": [YT + "bbb"]})
        assert r.get_json() == {"ok": True, "dropped": 1, "unqueued": 0}
        assert client.post(f"/api/ingest/{b['id']}/drop", json={"urls": "x"}).status_code == 400
        assert client.post(f"/api/ingest/{b['id']}/drop", json={}).get_json()["dropped"] == 1
        assert client.post("/api/ingest/aaaaaaaaaaaa/drop", json={}).status_code == 404

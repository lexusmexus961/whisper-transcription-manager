"""Multiple links, ONE job.

Until then the add dialog took exactly one URL. If someone wanted to fetch ten
videos, they had to open the dialog ten times, fill it ten times and then track
ten jobs — checking each one individually to see if it finished. Now one
field takes them all, and one job fetches them.

The work itself could already do it: ``_worker`` runs over a LIST of entries,
because that's what a playlist is. The only new thing is that this list can be
composed from multiple URLs.

The interesting question here is not the combining, but **what happens with one
bad link among ten good ones**. It must not drag the other nine down — otherwise
the batch function is worse than ten individual jobs, where at least nine work.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import ytdl

# ── Stubs ────────────────────────────────────────────────────────────


class FakeJob:
    def __init__(self):
        self.cancel = False
        self.lines: list[str] = []
        self.state_info: dict = {}
        self.status = ""
        self.result = None
        self.error = None

    def update(self, **kw):
        self.state_info.update(kw)


class FakeJobManager:
    def __init__(self, job):
        self.job = job

    def get(self, _jid):
        return self.job

    def create(self, jid, **kw):
        self.job.label = jid
        self.job.created = kw
        return self.job

    def log(self, job, text):
        job.lines.append(text)

    def fail(self, job, text):
        job.status, job.error = "error", text

    def complete(self, job, result):
        job.status, job.result = "done", result


@pytest.fixture
def job(monkeypatch, tmp_path):
    """A run without network and without disk: enumerate/download are stubbed."""
    j = FakeJob()
    monkeypatch.setattr(ytdl.ext, "job_manager", FakeJobManager(j))
    # `upload_dir` is a property without a setter — redirect the resolved
    # paths behind it, not the property itself.
    (tmp_path / "uploads").mkdir()
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(ytdl.config, "_resolved_uploads", tmp_path / "uploads", raising=False)
    monkeypatch.setattr(ytdl.config, "_resolved_work", tmp_path / "work", raising=False)
    monkeypatch.setattr(ytdl.config, "_resolved_chunks", tmp_path / "work", raising=False)

    def _load(job, entry, tmp, target_dir, report_it=None, audio_only=None):
        p = target_dir / f"{entry['title']}.m4a"
        p.write_bytes(b"x" * 1000)
        return p

    monkeypatch.setattr(ytdl, "_download_one", _load)
    return j


def _entries(*title):
    return [{"title": t, "url": f"https://x/{t}"} for t in title]


# ── Merging ────────────────────────────────────────────────────────


class TestSeveralLinks:
    def test_three_links_become_one_job(self, job, monkeypatch):
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries(u.rsplit("/", 1)[-1]))
        ytdl._worker("j1", ["https://y/a", "https://y/b", "https://y/c"])
        assert job.status == "done"
        assert job.result["total"] == 3 and job.result["ok"] == 3

    def test_playlists_and_single_videos_mix(self, job, monkeypatch):
        """A playlist is just a link that delivers multiple entries —
        after that, the loop no longer distinguishes the cases."""

        def _on(u):
            return _entries("p1", "p2", "p3") if "listing" in u else _entries("solo")

        monkeypatch.setattr(ytdl, "enumerate_entries", _on)
        ytdl._worker("j2", ["https://y/listing", "https://y/solo"])
        assert job.result["total"] == 4

    def test_the_progress_counts_across_everything(self, job, monkeypatch):
        """Not "1/1" per link, but 1/4, 2/4 … — otherwise the user sees
        four progress bars that all end at 100% and never knows how far
        the whole thing is."""
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("a", "b"))
        ytdl._worker("j3", ["https://y/1", "https://y/2"])
        assert "4/4" in " ".join(job.lines)


# ── The bad link among good ones ─────────────────────────────────────────


class TestOneBrokenLink:
    def test_it_does_not_take_the_others_down(self, job, monkeypatch):
        """The actual reason for this test: ten links, one deleted —
        the other nine must still arrive."""

        def _on(u):
            if "dead" in u:
                raise RuntimeError("Video unavailable")
            return _entries(u.rsplit("/", 1)[-1])

        monkeypatch.setattr(ytdl, "enumerate_entries", _on)
        ytdl._worker("j4", ["https://y/a", "https://y/dead", "https://y/b"])
        assert job.status == "done"
        assert job.result["ok"] == 2

    def test_and_it_is_named(self, job, monkeypatch):
        """A silent failure would be worse than the error itself — otherwise
        you have to search afterward for which of the ten videos is missing."""

        def _on(u):
            if "dead" in u:
                raise RuntimeError("Video unavailable")
            return _entries("a")

        monkeypatch.setattr(ytdl, "enumerate_entries", _on)
        ytdl._worker("j5", ["https://y/a", "https://y/dead"])
        f = job.result["error"]
        assert len(f) == 1 and "dead" in f[0]["title"]
        assert "unavailable" in f[0]["error"]

    def test_a_link_without_videos_counts_the_same(self, job, monkeypatch):
        monkeypatch.setattr(
            ytdl, "enumerate_entries", lambda u: [] if "empty" in u else _entries("a")
        )
        ytdl._worker("j6", ["https://y/empty", "https://y/good"])
        assert job.result["ok"] == 1
        assert job.result["error"][0]["title"].endswith("empty")

    def test_only_when_ALL_are_rotten_does_the_job_fail(self, job, monkeypatch):
        def _on(u):
            raise RuntimeError("Video unavailable")

        monkeypatch.setattr(ytdl, "enumerate_entries", _on)
        ytdl._worker("j7", ["https://y/a", "https://y/b"])
        assert job.status == "error"
        assert "unavailable" in job.error


# ── Job creation ───────────────────────────────────────────────────────


class TestStartDownload:
    @pytest.fixture(autouse=True)
    def _no_thread(self, monkeypatch, job):
        """Only check the creation, not the run."""
        started = {}

        class _T:
            def __init__(self, target=None, args=(), **kw):
                started["args"] = args

            def start(self):
                pass

        monkeypatch.setattr(ytdl.threading, "Thread", _T)
        self.started = started

    def test_a_single_string_still_works(self, job):
        """Scripts and the MCP server send a single URL."""
        ytdl.start_download("https://y/a")
        assert self.started["args"][1] == ["https://y/a"]

    def test_a_list_arrives_as_a_list(self, job):
        ytdl.start_download(["https://y/a", "https://y/b"])
        assert self.started["args"][1] == ["https://y/a", "https://y/b"]

    def test_duplicate_links_are_downloaded_once(self, job):
        """Someone who throws in the same link twice doesn't want it
        downloaded twice — and certainly not as a `video_1.mp4` alongside."""
        ytdl.start_download(["https://y/a", "https://y/b", "https://y/a"])
        assert self.started["args"][1] == ["https://y/a", "https://y/b"]

    def test_the_order_is_preserved(self, job):
        ytdl.start_download(["https://y/c", "https://y/a", "https://y/b"])
        assert self.started["args"][1] == ["https://y/c", "https://y/a", "https://y/b"]

    def test_empty_entries_are_thrown_out(self, job):
        ytdl.start_download(["https://y/a", "  ", ""])
        assert self.started["args"][1] == ["https://y/a"]

    def test_nothing_at_all_is_an_error(self, job):
        with pytest.raises(ValueError):
            ytdl.start_download(["  ", ""])

    def test_the_label_stays_readable(self, job):
        """The job's filename is the heading in the UI.
        Ten concatenated URLs would be unreadable there."""
        ytdl.start_download([f"https://y/{i}" for i in range(10)])
        assert job.created["filename"] == "10 Links"
        ytdl.start_download("https://y/only-one")
        assert job.created["filename"] == "https://y/only-one"


# ── The endpoint ─────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    import whisper_tm.extensions as ext
    from whisper_tm.app import create_app

    class _HW:
        has_ytdlp = True

    monkeypatch.setattr(ext, "hw", _HW())
    return create_app().test_client()


@pytest.fixture
def seen(monkeypatch):
    d = {}
    import whisper_tm.core.ytdl as y

    monkeypatch.setattr(
        y,
        "start_download",
        lambda url, audio_only=None: d.update(url=url, audio_only=audio_only) or "job1",
    )
    return d


class TestEndpoint:
    def test_urls_as_a_list(self, client, seen):
        r = client.post("/api/youtube/download", json={"urls": ["https://a/1", "https://a/2"]})
        assert r.status_code == 200
        assert r.get_json()["links"] == 2
        assert seen["url"] == ["https://a/1", "https://a/2"]

    def test_the_singular_form_stays(self, client, seen):
        """`url` is used by scripts and the MCP server."""
        r = client.post("/api/youtube/download", json={"url": "https://a/1"})
        assert r.status_code == 200
        assert seen["url"] == ["https://a/1"]

    def test_the_broken_link_is_NAMED(self, client, seen):
        """A collective message would make the user search through ten lines."""
        r = client.post(
            "/api/youtube/download", json={"urls": ["https://a/1", "youtube.com/watch"]}
        )
        assert r.status_code == 400
        assert "youtube.com/watch" in r.get_json()["error"]

    def test_an_empty_list(self, client, seen):
        r = client.post("/api/youtube/download", json={"urls": []})
        assert r.status_code == 400

    def test_no_link_is_silently_swallowed(self, client, seen):
        """Capping yes, silently dropping no — someone who sends 60 links
        and gets 50 downloaded assumes the other ten are done too."""
        from whisper_tm.api.media import MAX_YT_LINKS

        many = [f"https://a/{i}" for i in range(MAX_YT_LINKS + 10)]
        r = client.post("/api/youtube/download", json={"urls": many})
        assert r.status_code == 400
        assert str(MAX_YT_LINKS) in r.get_json()["error"]
        assert "url" not in seen, "something was started anyway"

    def test_the_limit_itself_still_works(self, client, seen):
        from whisper_tm.api.media import MAX_YT_LINKS

        many = [f"https://a/{i}" for i in range(MAX_YT_LINKS)]
        r = client.post("/api/youtube/download", json={"urls": many})
        assert r.status_code == 200

    def test_urls_must_be_a_list(self, client, seen):
        r = client.post("/api/youtube/download", json={"urls": "https://a/1"})
        assert r.status_code == 400


class TestDuplicates:
    """The same link, written differently.

    Typical case: in a longer list of links, the same video appears
    twice — once as `…watch?v=AbCdEfGhIj1&t=345s`, once as
    `…watch?v=AbCdEfGhIj1`. A text comparison sees two different
    strings, both get downloaded, the second is named `…_1.m4a`, and it
    is transcribed a second time. Bit-identical, so a complete
    transcription for nothing.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("https://www.youtube.com/watch?v=X&t=345s", "https://www.youtube.com/watch?v=X"),
            ("https://youtu.be/X", "https://www.youtube.com/watch?v=X"),
            ("https://youtu.be/X", "https://m.youtube.com/watch?v=X"),
            (
                "https://www.youtube.com/watch?v=X&feature=share",
                "https://www.youtube.com/watch?v=X&t=10",
            ),
        ],
    )
    def test_the_same_video_id_gives_the_same_key(self, a, b):
        assert ytdl.dedup_key(a) == ytdl.dedup_key(b)

    def test_different_videos_stay_different(self):
        assert ytdl.dedup_key("https://youtu.be/AAA") != ytdl.dedup_key("https://youtu.be/BBB")

    def test_a_PLAYLIST_is_NOT_merged_with_the_video(self):
        """`…watch?v=X&list=Y` loads the whole list, not just X. Merging
        both via the video ID would be the worse mistake — then an
        entire playlist would silently disappear."""
        one_by_one = "https://www.youtube.com/watch?v=X"
        listing = "https://www.youtube.com/watch?v=X&list=PL42"
        assert ytdl.dedup_key(one_by_one) != ytdl.dedup_key(listing)

    def test_foreign_services_stay_as_they_are(self):
        """The rule only knows YouTube. For Vimeo & co., the text counts —
        guessing would be worse than not recognizing it."""
        u = "https://vimeo.com/12345?x=1"
        assert ytdl.dedup_key(u) == u

    def test_broken_input_does_not_throw(self):
        for u in ("not even a url", "", "http://"):
            assert isinstance(ytdl.dedup_key(u), str)

    def test_the_real_case(self, job, monkeypatch):
        """Both spellings of one video in the same list."""

        class _T:
            def __init__(self, target=None, args=(), **kw):
                self.args = args
                TestDuplicates._last = args

            def start(self):
                pass

        monkeypatch.setattr(ytdl.threading, "Thread", _T)
        ytdl.start_download(
            [
                "https://www.youtube.com/watch?v=AbCdEfGhIj1&t=345s",
                "https://www.youtube.com/watch?v=ZyXwVuTsRq2",
                "https://www.youtube.com/watch?v=AbCdEfGhIj1",
            ]
        )
        urls = TestDuplicates._last[1]
        assert len(urls) == 2, urls
        # The FIRST-mentioned form is the one that's kept.
        assert urls[0].endswith("t=345s")


def test_frontend_and_backend_use_the_same_rule():
    """Two places, one rule: otherwise the list shows twenty entries, of
    which the server downloads nineteen — and afterward you go searching
    the result for the missing line."""
    from pathlib import Path

    js = (
        Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "upload.js"
    ).read_text(encoding="utf-8")
    assert "_ytKey" in js
    body = js[js.index("function _ytKey(") : js.index("function ytAddLink(")]
    # The same three decisions as in dedup_key()
    assert "list" in body, "the playlist exception is missing in the frontend"
    assert "youtu.be" in body
    assert "shorts" in body


class TestRestart:
    """A failed video gets ONE second attempt — with fresh resolution.

    yt-dlp retries ten times on its own (`--retries`, default 10), but
    always against the SAME stream URL. If that's invalidated, all ten
    fail the same way: "unable to download video data: HTTP Error 403:
    Forbidden".

    In a longer list, several links can drop out that way, while the
    same call a few minutes later fetches them without issue — the
    difference is the NEW yt-dlp process, which gets fresh URLs
    handed to it. A retry inside yt-dlp can't achieve that.
    """

    @pytest.fixture(autouse=True)
    def _no_pause(self, monkeypatch):
        monkeypatch.setattr(ytdl, "RESTART_PAUSE_SEC", 0)

    def test_the_second_attempt_saves_the_video(self, job, monkeypatch):
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("a"))
        attempts = {"n": 0}

        def _load(job_, entry, tmp, target_dir, report_it=None, audio_only=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("unable to download video data: HTTP Error 403: Forbidden")
            p = target_dir / "a.m4a"
            p.write_bytes(b"x" * 1000)
            return p

        monkeypatch.setattr(ytdl, "_download_one", _load)
        ytdl._worker("j10", ["https://y/a"])
        assert attempts["n"] == 2
        assert job.result["ok"] == 1 and not job.result["error"]

    def test_and_it_is_not_kept_quiet(self, job, monkeypatch):
        """A silent retry would look like an inexplicably slow download —
        and would hide the fact that YouTube is being difficult."""
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("a"))
        n = {"i": 0}

        def _load(job_, entry, tmp, target_dir, report_it=None, audio_only=None):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("HTTP Error 403: Forbidden")
            p = target_dir / "a.m4a"
            p.write_bytes(b"x")
            return p

        monkeypatch.setattr(ytdl, "_download_one", _load)
        ytdl._worker("j11", ["https://y/a"])
        assert any("↻" in z for z in job.lines), job.lines

    def test_exactly_once_not_endlessly(self, job, monkeypatch):
        """Otherwise a run over 20 links gets stuck on one dead video."""
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("a"))
        n = {"i": 0}

        def _load(*a, **kw):
            n["i"] += 1
            raise RuntimeError("403")

        monkeypatch.setattr(ytdl, "_download_one", _load)
        ytdl._worker("j12", ["https://y/a"])
        assert n["i"] == 2, f"{n['i']} attempts"
        assert job.status == "error"

    def test_a_CANCEL_is_not_retried(self, job, monkeypatch):
        """Someone who hits cancel WHILE the download is running doesn't
        want to see another attempt.

        (A `cancel` that's already set before the video doesn't even get
        this far — the loop breaks off before that. What's tested here is
        therefore the cancel in the middle of downloading, which looks
        like this: `_download_one` raises InterruptedError.)
        """
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("a"))
        n = {"i": 0}

        def _load(job_, *a, **kw):
            n["i"] += 1
            job_.cancel = True
            raise InterruptedError("Cancelled")

        monkeypatch.setattr(ytdl, "_download_one", _load)
        ytdl._worker("j13", ["https://y/a"])
        assert n["i"] == 1, f"the cancel was retried ({n['i']} attempts)"

    def test_a_cancel_does_not_bring_the_worker_down(self, job, monkeypatch):
        """The cancel path depends on the outer `try`. If the inner loop
        re-raised the InterruptedError itself, it would bypass that and
        the job would end without the "Cancelled" message."""
        monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("a", "b"))

        def _load(job_, entry, *a, **kw):
            job_.cancel = True
            raise InterruptedError("Cancelled")

        monkeypatch.setattr(ytdl, "_download_one", _load)
        ytdl._worker("j14", ["https://y/a"])  # must NOT propagate
        assert job.status == "error"
        assert "Cancelled" in (job.error or "")

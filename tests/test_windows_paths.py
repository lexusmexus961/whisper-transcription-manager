"""
What broke on Windows — checked here on Linux, with Windows-style values.

A review for Windows found two causes behind most of it: stored paths are
``C:/…`` (docs/PATHS.md) while a folder setting or ``config.media_dir`` is
``C:\\…``, and comparing the two with ``startswith`` never matched; and
output of the programs the app starts was read in the ANSI code page.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import PureWindowsPath

import pytest

from whisper_tm.core import commands

WIN_ROOT = r"C:\Users\alice\Videos"


# ── extra arguments: a path keeps its backslashes ───────────────────────────


@pytest.mark.parametrize(
    "text,words",
    [
        (r"--mmproj C:\models\x.gguf", ["--mmproj", r"C:\models\x.gguf"]),
        (r'--mmproj "C:\Program Files\x.gguf"', ["--mmproj", r"C:\Program Files\x.gguf"]),
        (r"--model \\srv\share\m.gguf -fa on", ["--model", r"\\srv\share\m.gguf", "-fa", "on"]),
        ("--opt 'two words'", ["--opt", "two words"]),
    ],
)
def test_extra_arguments_keep_a_windows_path(monkeypatch, text, words):
    monkeypatch.setattr(commands.sys, "platform", "win32")
    assert commands.split_args(text) == words


def test_extra_arguments_on_posix_are_split_as_before(monkeypatch):
    monkeypatch.setattr(commands.sys, "platform", "linux")
    assert commands.split_args("--flash-attn on --x 'a b'") == ["--flash-attn", "on", "--x", "a b"]
    with pytest.raises(ValueError):
        commands.split_args('"unclosed')


def test_the_ytdlp_check_sees_the_words_that_run(monkeypatch):
    """The settings page's allowlist and yt-dlp's command split alike — on
    Windows a quoted option must still be refused."""
    from whisper_tm.api.settings import _check_ytdlp_args

    monkeypatch.setattr(commands.sys, "platform", "win32")
    assert "--exec" in _check_ytdlp_args('"--exec" calc')
    assert _check_ytdlp_args("") == ""


# ── stored paths against a backslash root ──────────────────────────────────


class _Db:
    def __init__(self, rows):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE transcripts (rel_path TEXT, abs_path TEXT, complete INT, word_count INT)"
        )
        self.conn.executemany(
            "INSERT INTO transcripts VALUES (?, ?, 1, ?)",
            [r if len(r) == 3 else (*r, 5) for r in rows],
        )

    @contextmanager
    def connection(self):
        yield self.conn


def test_the_llm_queue_finds_a_folders_files_under_a_windows_root(monkeypatch):
    from whisper_tm.api import llm_queue
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: PureWindowsPath(WIN_ROOT)))
    db = _Db(
        [
            ("a.mp4", "C:/Users/alice/Videos/a.mp4"),
            ("sub/b.mp4", "C:/Users/alice/Videos/sub/b.mp4"),
            ("c.mp4", "D:/Other/c.mp4"),
        ]
    )
    assert llm_queue._files_with_transcript(db, "", True) == ["a.mp4", "sub/b.mp4"]


def test_mcp_cuts_a_windows_hit_to_the_root(monkeypatch):
    from whisper_tm import mcp_server as mcp

    monkeypatch.setattr(mcp, "_media_root", lambda: WIN_ROOT)
    assert mcp._rel_path("C:/Users/alice/Videos/sub/a.mp4") == "sub/a.mp4"
    assert mcp._rel_path("D:/elsewhere/a.mp4") == "D:/elsewhere/a.mp4"
    assert mcp._rel_path("sub/a.mp4") == "sub/a.mp4"
    monkeypatch.setattr(mcp, "_media_root", lambda: "/media")
    assert mcp._rel_path("/media/x/a.mp4") == "x/a.mp4"
    assert mcp._rel_path("/media2/a.mp4") == "/media2/a.mp4"


def test_images_and_sounds_folders_in_the_stored_form(monkeypatch):
    from whisper_tm.media import indexer

    monkeypatch.setattr(
        indexer,
        "_cfg",
        lambda name, default=None: (
            [WIN_ROOT + r"\clips", "relative/folder"] if name == "folders" else default
        ),
    )
    folders = indexer.settings()["folders"]
    assert folders[0] == "C:/Users/alice/Videos/clips"
    # not absolute: kept as written, so it still matches nothing
    assert folders[1] == "relative/folder"


def test_the_incident_list_takes_a_windows_path_as_a_path(monkeypatch):
    from flask import Flask

    from whisper_tm.api import incidents as api

    seen = {}

    class Db:
        def incidents(self, **kw):
            seen.update(kw)
            return {"items": [], "total": 0}

    import whisper_tm.extensions as ext

    # the endpoint takes the app's database first (_db) — an earlier test's
    # may still be set there
    monkeypatch.setattr(ext, "db", Db())
    monkeypatch.setattr(api, "db", None)
    app = Flask(__name__)
    app.register_blueprint(api.incidents_bp)
    app.test_client().get("/api/incidents?file=" + WIN_ROOT.replace("\\", "/") + "/sub/a.mp4")
    assert seen["abs_path"] == "C:/Users/alice/Videos/sub/a.mp4" and seen["filename"] is None
    app.test_client().get("/api/incidents?file=a.mp4")
    assert seen["abs_path"] is None and seen["filename"] == "a.mp4"


# ── names the app gives files ───────────────────────────────────────────────


def test_a_link_jobs_log_name_has_no_windows_forbidden_characters():
    from whisper_tm.jobs import log_name_part

    name = log_name_part("https://www.youtube.com/watch?v=abc")
    assert not set(name) & set('<>:"|?*/\\ ')


def test_the_log_search_finds_old_and_new_names(tmp_path, monkeypatch):
    """A file with ":" in its name (fine on Linux): logs written before keep
    it, new ones get "_" — the 📋 button finds either."""
    from whisper_tm.api import transcription as tx
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "dirs", property(lambda self: {"logs": tmp_path}))
    monkeypatch.setattr(tx, "_LOG_CACHE", {})
    (tmp_path / "20260101_000000_10:30_Meeting.mp4.log").write_text("old")
    assert tx._find_log_for_file("10:30 Meeting.mp4") == "20260101_000000_10:30_Meeting.mp4.log"
    time.sleep(0.02)
    (tmp_path / "20260102_000000_10_30_Meeting.mp4.log").write_text("new")
    monkeypatch.setattr(tx, "_LOG_CACHE", {})
    assert tx._find_log_for_file("10:30 Meeting.mp4") == "20260102_000000_10_30_Meeting.mp4.log"


def test_a_shortened_job_folder_is_still_known_as_the_apps():
    """The job folder takes the job ID's last 48 characters (Windows' 260
    character paths) — the cleanup must still recognize it."""
    import hashlib

    from whisper_tm.config import ConfigManager

    filename = "a/very/deeply/nested/library/folder/with/a/long/recording name.mp4"
    fhash = hashlib.sha256(filename.encode()).hexdigest()[:8]
    jid = filename.replace(" ", "_").replace(".", "_").replace("/", "_") + f"_{fhash}"
    assert len(jid) > 48
    assert ConfigManager._APP_WORK_RE.match(jid[-48:])


# ── processes ───────────────────────────────────────────────────────────────


def test_without_process_groups_the_children_go_too(monkeypatch):
    """yt-dlp's ffmpeg merge outlived it on Windows and kept the file locked."""
    import os

    import psutil

    from whisper_tm.core import procgroup

    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)",
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while not psutil.Process(parent.pid).children():
            assert time.monotonic() < deadline, "no grandchild"
            time.sleep(0.05)
        child = psutil.Process(parent.pid).children()[0]
        monkeypatch.delattr(os, "killpg", raising=False)
        assert procgroup.signal_group(parent.pid) is False
        # Ended: gone, or a zombie its sleeping parent hasn't collected yet
        deadline = time.monotonic() + 10
        while True:
            try:
                if child.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            assert time.monotonic() < deadline, "the grandchild still runs"
            time.sleep(0.05)
    finally:
        parent.kill()
        parent.wait(timeout=10)


def test_a_marker_older_than_its_process_is_a_reused_pid(tmp_path):
    """Windows hands out PIDs again quickly: a process started after the
    server wrote its marker is not that server, whatever its command line."""
    import json
    import os

    from whisper_tm.config import ConfigManager

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "whisper_tm"])
    try:
        cfg = ConfigManager()
        cfg._resolved_work = tmp_path  # type: ignore[attr-defined]
        marker = tmp_path / cfg.USAGE_FILE
        marker.write_text(
            json.dumps({"pid": child.pid, "cmdline": "python -m whisper_tm"}), encoding="utf-8"
        )
        past = time.time() - 3600
        os.utime(marker, (past, past))
        assert cfg._workdir_taken_by_other() == ""
        # A minute: within the slack for a clock set forward — still busy
        recent = time.time() - 60
        os.utime(marker, (recent, recent))
        assert cfg._workdir_taken_by_other() == f"PID {child.pid}"
    finally:
        child.kill()
        child.wait(timeout=10)


# ── the page: Ask's clip of a full Windows path ─────────────────────────────


def test_ask_plays_a_clip_of_a_full_windows_path():
    import json
    import shutil
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("node is needed for the page's code")
    src = (
        Path(__file__).resolve().parents[1] / "whisper_tm" / "static" / "js" / "chat.js"
    ).read_text(encoding="utf-8")
    start = src.index("function chatClipWhere(file) {")
    fn = src[start : src.index("\n}\n", start) + 2]
    cases = [
        "/mnt/v/sub/a.mp4",
        "C:/Users/x/Videos/a.mp4",
        "C:\\Users\\x\\a.mp4",
        "C:/a.mp4",
        "sub/a.mp4",
    ]
    script = fn + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(chatClipWhere)));"
    out = json.loads(
        subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    )
    assert out == [
        {"file": "a.mp4", "dir": "/mnt/v/sub"},
        {"file": "a.mp4", "dir": "C:/Users/x/Videos"},
        {"file": "a.mp4", "dir": "C:\\Users\\x"},
        {"file": "a.mp4", "dir": "C:/"},
        {"file": "sub/a.mp4", "dir": ""},
    ]


def test_the_llm_queue_names_a_folders_files_from_their_identity(monkeypatch):
    """The stored rel_path is relative to the folder open at transcription:
    "b.mp4" (transcribed with sub open) was missing from sub's batch."""
    from pathlib import PurePosixPath

    from whisper_tm.api import llm_queue
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: PurePosixPath("/lib")))
    db = _Db(
        [
            ("sub/a.mp4", "/lib/sub/a.mp4"),
            ("b.mp4", "/lib/sub/b.mp4"),
            ("c.mp4", "/lib/c.mp4"),
            ("x.mp4", "/elsewhere/x.mp4"),
        ]
    )
    assert llm_queue._files_with_transcript(db, "sub", False) == ["sub/a.mp4", "sub/b.mp4"]
    assert llm_queue._files_with_transcript(db, "", True) == ["c.mp4", "sub/a.mp4", "sub/b.mp4"]
    assert llm_queue._files_with_transcript(db, "", False) == ["c.mp4"]


def test_a_folder_batch_leaves_out_transcripts_without_speech(monkeypatch):
    """Every task on a transcript without a word fails: a folder batch over
    a folder with a silent recording queued it anyway."""
    from pathlib import PurePosixPath

    from whisper_tm.api import llm_queue
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: PurePosixPath("/lib")))
    db = _Db([("a.mp4", "/lib/a.mp4", 12), ("silence.wav", "/lib/silence.wav", 0)])
    empty: list = []
    assert llm_queue._files_with_transcript(db, "", True, empty=empty) == ["a.mp4"]
    assert empty == ["silence.wav"]

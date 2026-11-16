"""Opening a file must not depend on the log directory.

**The finding.** A server log showed:

    GET /api/video/…      → 206 in    1 ms   ← video arrives right away
    GET /api/transcript/… → 200 in   50 s   ← this one blocked

The cause was `_find_log_for_file`: it ran on EVERY transcript fetch,
read the entire log directory and did an `is_file()` per entry.
On a large installation (well over a hundred thousand log files on a
network share), just listing them took well over ten seconds.

The result only filled the target name of the "📋 Log" button.

These tests ensure both: that the hot path no longer touches the search,
and that it still finds the right thing on demand.
"""

from __future__ import annotations

import time

import pytest
from flask import Flask

from whisper_tm.api import transcription as tx


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    logs = tmp_path / "logs"
    logs.mkdir()
    # `dirs` is a property without a setter — so patch the type, not the
    # instance.
    monkeypatch.setattr(type(config), "dirs", property(lambda self: {"logs": logs}))
    monkeypatch.setattr(tx, "_LOG_CACHE", {})

    a = Flask(__name__)
    a.register_blueprint(tx.transcription_bp)
    a.logs = logs
    return a


class TestHotPath:
    def test_transcript_retrieval_does_not_search_log(self, app, monkeypatch):
        """The actual fix. The search must not occur here at all."""

        def bomb(*a, **k):
            raise AssertionError(
                "transcript retrieval searches log directory again — that's exactly "
                "what cost ~50 s per file call"
            )

        monkeypatch.setattr(tx, "_find_log_for_file", bomb)
        monkeypatch.setattr(
            tx,
            "db",
            type(
                "D",
                (),
                {
                    "get_transcript": staticmethod(
                        lambda *a, **k: {"segments": [], "language": "de"}
                    )
                },
            )(),
        )

        d = app.test_client().get("/api/transcript/anything.mkv").get_json()
        assert d["language"] == "de"

    def test_endpoint_on_demand_provides_name(self, app):
        (app.logs / "20260810_120000_film.mkv.log").write_text("old")
        d = app.test_client().get("/api/transcript/film.mkv/logfile").get_json()
        assert d["log_file"] == "20260810_120000_film.mkv.log"


class TestSearchItself:
    def test_finds_newest_hit(self, app):
        old = app.logs / "20260101_000000_film.mkv.log"
        new = app.logs / "20260810_120000_film.mkv.log"
        old.write_text("old")
        time.sleep(0.02)
        new.write_text("new")
        assert tx._find_log_for_file("film.mkv") == new.name

    def test_spaces_replaced_like_when_writing(self, app):
        (app.logs / "20260810_120000_my_film.mkv.log").write_text("x")
        assert tx._find_log_for_file("my film.mkv") == "20260810_120000_my_film.mkv.log"

    def test_no_hits_empty_not_error(self, app):
        (app.logs / "20260810_120000_other.mkv.log").write_text("x")
        assert tx._find_log_for_file("film.mkv") == ""

    def test_foreign_files_not_touched(self, app, monkeypatch):
        """Only the NAME decides — no stat per entry.

        That was the expensive part: over a hundred thousand entries × one
        ``is_file()`` over CIFS. Check that the search doesn't touch non-hits.
        """
        for i in range(50):
            (app.logs / f"20260810_1200{i:02d}_foreign{i}.mkv.log").write_text("x")
        (app.logs / "20260810_130000_film.mkv.log").write_text("x")

        import os as _os

        # More honest than a monkeypatch on DirEntry: we count
        # stat calls with their path.
        touched = []
        real_stat = _os.stat

        def counted(path, *a, **k):
            touched.append(str(path))
            return real_stat(path, *a, **k)

        monkeypatch.setattr(_os, "stat", counted)
        assert tx._find_log_for_file("film.mkv") == "20260810_130000_film.mkv.log"
        # Only check the FILE NAMES: the test's tmp path itself contains
        # the word "foreign" and would otherwise give a false alarm.
        foreign = [p for p in touched if _os.path.basename(p).startswith("20260810_1200")]
        assert not foreign, f"{len(foreign)} non-hits were touched anyway"

    def test_second_search_from_cache(self, app, monkeypatch):
        (app.logs / "20260810_120000_film.mkv.log").write_text("x")
        assert tx._find_log_for_file("film.mkv")

        import os as _os

        def bomb(*a, **k):
            raise AssertionError("directory was read again")

        # `os` is imported locally in _find_log_for_file — so patch
        # the module itself, not an attribute of `tx`.
        monkeypatch.setattr(_os, "scandir", bomb)
        assert tx._find_log_for_file("film.mkv") == "20260810_120000_film.mkv.log"

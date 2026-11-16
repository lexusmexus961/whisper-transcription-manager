"""Continuous scanning — new files found, settled, added and queued.

An option in the settings for continuous scanning and automatic
transcription when it finds new files — off by default.

What these tests pin:

* the first round of a folder only takes note — switching this on over a
  big library queues nothing that was already there;
* a file that appears later is new; it is queued only once its size and
  time stayed the same for the settle time, and a file still being written
  (growing inside a folder whose time didn't change) waits;
* a round reads only the folders that changed, the rest comes from memory;
* with ``transcribe`` off a new file only joins the library;
* a folder that can't be read deletes nothing; one outside the allowed
  roots is refused;
* the switch: off by default, on and off over the API, saved.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from whisper_tm.core import autoscan as au


@pytest.fixture
def lib(tmp_path, monkeypatch):
    from whisper_tm.config import config

    root = tmp_path / "library"
    (root / "a").mkdir(parents=True)
    (root / "a" / "old.mp4").write_bytes(b"x" * 10)
    (root / "b").mkdir()
    (root / "b" / "old.mp3").write_bytes(b"y" * 10)
    (root / "b" / "notes.txt").write_text("no media")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.mp4").write_bytes(b"z")
    monkeypatch.setitem(config._data.setdefault("security", {}), "allowed_roots", [str(root)])
    # The open library too: allowed_roots() always counts it, and the
    # session's default (~/Videos) is not there before a library is chosen
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: root))
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    a = config._data.setdefault("autoscan", {})
    for k, v in {
        "enabled": False,
        "transcribe": True,
        "interval_min": 15,
        "folders": [],
        "settle_min": 0,
        "full_every_h": 6,
    }.items():
        monkeypatch.setitem(a, k, v)
    queued, registered = [], []
    monkeypatch.setattr(
        "whisper_tm.core.queue.enqueue",
        lambda names, settings, base_dir=None, **k: (
            queued.append((list(names), settings, base_dir)) or [{"id": 1}]
        ),
    )
    monkeypatch.setattr(
        "whisper_tm.scanner.register_single_file",
        lambda p, db=None, root=None: registered.append(p) or os.path.relpath(p, root),
    )
    monkeypatch.setattr(
        "whisper_tm.core.helpers.is_transcript_complete",
        lambda rel, media_dir=None: rel.endswith("done.mp4"),
    )
    scan = au.AutoScan()
    return {"root": root, "scan": scan, "queued": queued, "registered": registered, "settings": a}


def _bump(path: Path):
    """A folder's time changes on create/delete — make it visibly newer
    (some filesystems keep a coarse clock)."""
    t = time.time() + 5
    os.utime(path, (t, t))


class TestTheWalk:
    def test_media_only_and_the_ignored_names(self, lib):
        w = au.walk(lib["root"], {}, full=True)
        assert w["ok"] and sorted(Path(p).name for p in w["files"]) == ["old.mp3", "old.mp4"]

    def test_an_unchanged_folder_is_not_read_again(self, lib):
        cache = {}
        au.walk(lib["root"], cache, full=True)
        w = au.walk(lib["root"], cache, full=False)
        assert w["read"] == 0 and w["cached"] == 3 and len(w["files"]) == 2
        (lib["root"] / "a" / "new.mkv").write_bytes(b"n")
        _bump(lib["root"] / "a")
        w = au.walk(lib["root"], cache, full=False)
        assert w["read"] == 1 and any(p.endswith("new.mkv") for p in w["files"])

    def test_a_root_that_is_gone(self, lib, tmp_path):
        w = au.walk(tmp_path / "missing", {}, full=True)
        assert not w["ok"] and w["files"] == {}


class TestRounds:
    def test_the_first_round_only_takes_note(self, lib):
        r = lib["scan"].round()
        assert r["folders"][0].get("baseline") and lib["queued"] == []
        assert lib["scan"].state().counts() == {"known": 2}

    def test_a_new_file_settles_then_is_queued(self, lib):
        s, root = lib["scan"], lib["root"]
        s.round()
        (root / "a" / "talk.mp4").write_bytes(b"t" * 100)
        _bump(root / "a")
        lib["settings"]["settle_min"] = 60  # an hour: not yet
        r = s.round()
        assert r["new"] == 1 and lib["queued"] == []
        lib["settings"]["settle_min"] = 0
        r = s.round()
        assert r["queued"] == 1
        assert lib["queued"] == [(["a/talk.mp4"], {}, str(root))]
        assert lib["registered"] == [str(root / "a" / "talk.mp4")]
        assert s.state().counts() == {"known": 2, "queued": 1}
        assert s.round()["queued"] == 0  # once only

    def test_a_growing_file_waits(self, lib):
        s, root = lib["scan"], lib["root"]
        s.round()
        f = root / "b" / "copying.mkv"
        f.write_bytes(b"1" * 10)
        _bump(root / "b")
        s.round()  # seen
        with open(f, "ab") as h:  # still being copied …
            h.write(b"2" * 10)  # … folder time unchanged
        assert s.round()["queued"] == 0  # grew: waits a round
        assert s.round()["queued"] == 1  # unchanged since: goes

    def test_without_transcribe_it_only_joins_the_library(self, lib):
        s, root = lib["scan"], lib["root"]
        s.round()
        lib["settings"]["transcribe"] = False
        (root / "a" / "clip.wav").write_bytes(b"w")
        _bump(root / "a")
        s.round()
        assert s.round()["added"] == 1 and lib["queued"] == []
        assert lib["registered"]

    def test_an_existing_transcript_is_not_queued_again(self, lib):
        s, root = lib["scan"], lib["root"]
        s.round()
        (root / "a" / "done.mp4").write_bytes(b"d")
        _bump(root / "a")
        s.round()
        assert s.round()["added"] == 1 and lib["queued"] == []

    def test_a_deleted_file_leaves_the_list(self, lib):
        s, root = lib["scan"], lib["root"]
        s.round()
        (root / "b" / "old.mp3").unlink()
        _bump(root / "b")
        s.round()
        assert s.state().counts() == {"known": 1}

    def test_a_folder_inside_another_is_watched_once(self, lib):
        """Overlapping folders made every round find the other's files new."""
        root = lib["root"]
        lib["settings"]["folders"] = [str(root / "a"), str(root), str(root / "a")]
        assert au.watched_folders() == [root.resolve()]
        s = lib["scan"]
        s.round()
        assert s.round()["new"] == 0 and lib["queued"] == []

    def test_what_the_app_writes_itself_is_not_new(self, lib):
        """An export into a folder of the library, separated tracks next to a
        recording: the app's own output, never queued."""
        s, root = lib["scan"], lib["root"]
        s.round()
        export = root / "a" / "tts-dataset"
        (export / "wavs").mkdir(parents=True)
        for i in range(3):
            (export / "wavs" / f"clip_{i}.wav").write_bytes(b"c")
        (root / "a" / "old_vocals.mp3").write_bytes(b"v")
        au.note_made(export, root / "a" / "old_vocals.mp3")
        _bump(root / "a")
        assert s.round()["new"] == 0 and s.round()["queued"] == 0
        assert lib["queued"] == []

    def test_the_apps_own_folders_are_not_walked(self, lib, monkeypatch):
        from whisper_tm.config import config

        clips = lib["root"] / "b" / "clips"
        clips.mkdir()
        (clips / "export.flac").write_bytes(b"f")
        real = type(config).dirs.fget
        monkeypatch.setattr(
            type(config), "dirs", property(lambda self: {**real(self), "clips": clips})
        )
        w = au.walk(lib["root"], {}, full=True)
        assert not any("clips" in p for p in w["files"])

    def test_outside_the_allowed_folders(self, lib, tmp_path):
        other = tmp_path / "elsewhere"
        other.mkdir()
        lib["settings"]["folders"] = [str(other)]
        r = lib["scan"].round()
        assert "outside the allowed folders" in r["folders"][0]["error"]


class TestTheSwitch:
    @pytest.fixture
    def client(self, lib, monkeypatch):
        from flask import Flask

        from whisper_tm.api.main import main_bp
        from whisper_tm.config import config

        monkeypatch.setattr(au, "AUTOSCAN", lib["scan"])
        monkeypatch.setattr(config, "save", lambda *a, **k: Path("/dev/null"))
        a = Flask(__name__)
        a.register_blueprint(main_bp)
        yield a.test_client()
        lib["scan"].stop()

    def test_off_by_default(self):
        import yaml

        d = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert d["autoscan"]["enabled"] is False and d["autoscan"]["transcribe"] is True

    def test_on_and_off(self, client, lib):
        assert client.get("/api/autoscan").get_json()["enabled"] is False
        d = client.post("/api/autoscan", json={"enabled": True}).get_json()
        assert d["enabled"] and d["running"]
        d = client.post("/api/autoscan", json={"enabled": False}).get_json()
        assert not d["enabled"]
        assert client.post("/api/autoscan", json={}).status_code == 400
        assert client.post("/api/autoscan/scan").status_code == 409

    def test_the_whole_library_chosen_folders_or_off(self, client, lib, tmp_path):
        root = str(tmp_path / "library")
        d = client.post("/api/autoscan", json={"mode": "library"}).get_json()
        assert d["mode"] == "library" and d["enabled"] and d["chosen"] == []
        # watching one folder: only that one (and its subfolders) from now on
        d = client.post("/api/autoscan", json={"watch": root + "/a"}).get_json()
        assert (
            d["mode"] == "folders"
            and d["chosen"] == [root + "/a"]
            and d["folders"] == [root + "/a"]
        )
        d = client.post("/api/autoscan", json={"watch": root + "/b"}).get_json()
        assert d["chosen"] == [root + "/a", root + "/b"]
        assert client.post("/api/autoscan", json={"watch": "/etc"}).status_code == 400
        d = client.post("/api/autoscan", json={"unwatch": root + "/a"}).get_json()
        assert d["chosen"] == [root + "/b"] and d["mode"] == "folders"
        # the last chosen folder taken out: off, not "everything"
        d = client.post("/api/autoscan", json={"unwatch": root + "/b"}).get_json()
        assert d["mode"] == "off" and not d["enabled"]
        assert client.post("/api/autoscan", json={"mode": "folders"}).status_code == 400
        d = client.post("/api/autoscan", json={"mode": "off"}).get_json()
        assert d["mode"] == "off"
        assert client.post("/api/autoscan", json={"mode": "sometimes"}).status_code == 400

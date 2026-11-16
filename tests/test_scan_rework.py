"""
Tests for the scan overhaul:
- collect_media_files: parallel filesystem walk (NAS acceleration)
- scan_media_directory_from_db: walk-free scan from DB only (auto-walk disabled)
"""

import time

import pytest

from whisper_tm.database import Database
from whisper_tm.scanner import scan_media_directory_from_db
from whisper_tm.scanner.filesystem import collect_media_files, iter_media_files

MEDIA_EXT = {".mp4", ".mkv", ".mp3"}


@pytest.fixture
def media_dir(tmp_path):
    """Temporary media directory with test files."""
    (tmp_path / "video1.mp4").write_text("fake video")
    (tmp_path / "video2.mkv").write_text("fake mkv")
    (tmp_path / "audio.mp3").write_text("fake audio")
    (tmp_path / "readme.txt").write_text("not media")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "deep.mp4").write_text("deep video")
    (tmp_path / "subdir" / "nested").mkdir()
    (tmp_path / "subdir" / "nested" / "deeper.mp3").write_text("deeper")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.mp4").write_text("hidden")
    return tmp_path


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close_all()


class TestCollectMediaFiles:
    def test_recursive_parallel(self, media_dir):
        files = collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
        )
        names = {f.name for f in files}
        assert names == {"video1.mp4", "video2.mkv", "audio.mp3", "deep.mp4", "deeper.mp3"}
        assert "secret.mp4" not in names  # .hidden ignored

    def test_matches_serial_iterator(self, media_dir):
        """Parallel and serial must deliver the same file set."""
        parallel = collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
        )
        serial = sorted(
            iter_media_files(
                media_dir,
                recursive=True,
                media_extensions=MEDIA_EXT,
                ignore_patterns=[".*"],
            )
        )
        assert parallel == serial

    def test_result_is_sorted(self, media_dir):
        files = collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
        )
        assert files == sorted(files)

    def test_non_recursive_falls_back(self, media_dir):
        files = collect_media_files(
            media_dir,
            recursive=False,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
        )
        names = {f.name for f in files}
        assert "deep.mp4" not in names
        assert "video1.mp4" in names

    def test_no_limit_whole_tree(self, media_dir):
        """Limits were removed — the parallel walk sees EVERYTHING."""
        files = collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
        )
        names = {f.name for f in files}
        assert "deep.mp4" in names
        assert "video1.mp4" in names

    def test_transcript_dir_callback(self, media_dir):
        (media_dir / ".transcripts").mkdir()
        found = []
        collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
            on_transcript_dir=lambda p: found.append(p),
        )
        assert len(found) == 1
        assert found[0].name == ".transcripts"

    def test_symlink_cycle_terminates(self, tmp_path):
        """Mutual symlinks must not hang the parallel walk and must not
        deliver any file twice."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "in_a.mp4").write_text("x")
        (b / "in_b.mp4").write_text("y")
        try:
            (a / "to_b").symlink_to(b, target_is_directory=True)
            (b / "to_a").symlink_to(a, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not supported")
        t0 = time.monotonic()
        files = collect_media_files(
            tmp_path,
            recursive=True,
            media_extensions={".mp4"},
            ignore_patterns=[".*"],
            workers=8,
        )
        assert time.monotonic() - t0 < 10
        # Every real file exactly ONCE (regardless of which path reached it)
        real = {f.resolve() for f in files}
        assert len(real) == 2
        assert len(files) == 2

    def test_progress_callback(self, media_dir):
        """on_progress delivers at least one completion event with the final state."""
        events = []
        files = collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
            on_progress=events.append,
        )
        assert events, "no progress event"
        last = events[-1]
        assert last["files"] == len(files)  # completion = final state
        assert last["dirs_done"] == last["dirs_known"]  # all directories done
        assert last["dirs_done"] >= 3  # root + subdir + nested
        assert last["errors"] == 0

    def test_transport_error_streak_aborts(self, tmp_path, monkeypatch):
        """Dead network drive + wait mode OFF → WalkAborted instead of
        grinding on for hours."""
        from whisper_tm.scanner import filesystem as fsmod

        root = tmp_path / "nas"
        root.mkdir()

        def fake_scan(path, ig, ext):
            if path == str(root):
                # Root delivers many subdirectories …
                subs = [(f"{path}/d{i}", f"d{i}", False) for i in range(400)]
                return [], subs, [], None
            # … which then ALL fail with a transport error (errno 112)
            return [], [], [], 112

        monkeypatch.setattr(fsmod, "_scan_one_dir", fake_scan)
        with pytest.raises(fsmod.WalkAborted, match="transport errors"):
            collect_media_files(
                root,
                recursive=True,
                media_extensions={".mp4"},
                ignore_patterns=[".*"],
                workers=4,
                wait_for_mount=False,
            )

    def test_wait_mode_recovers_and_retries(self, tmp_path, monkeypatch):
        """Wait mode ON: NAS outage → walk pauses, probes, catches up on the
        failed directories after it returns — NO abort."""
        from whisper_tm.scanner import filesystem as fsmod

        root = tmp_path / "nas"
        root.mkdir()
        n_subs = 8
        state = {"fails": 0, "down": True}

        def fake_scan(path, ig, ext):
            if path == str(root):
                subs = [(f"{path}/d{i}", f"d{i}", False) for i in range(n_subs)]
                return [], subs, [], None
            if state["down"]:
                state["fails"] += 1
                return [], [], [], 112  # Host is down
            return [f"{path}/clip.mp4"], [], [], None

        # Probe: "down" at first, from the 2nd attempt the NAS is back
        probes = {"n": 0}

        def fake_probe(path, timeout):
            probes["n"] += 1
            if probes["n"] >= 2:
                state["down"] = False
                return True
            return False

        monkeypatch.setattr(fsmod, "_scan_one_dir", fake_scan)
        monkeypatch.setattr(fsmod, "_probe_dir_alive", fake_probe)

        events = []
        files = collect_media_files(
            root,
            recursive=True,
            media_extensions={".mp4"},
            ignore_patterns=[".*"],
            workers=2,
            wait_for_mount=True,
            max_error_streak=4,
            probe_sec=0.01,
            on_progress=events.append,
        )
        # All directories were caught up after it returned
        assert len(files) == n_subs
        assert state["fails"] >= 4  # the outage was experienced
        assert any(e.get("waiting") for e in events)  # waiting state reported

    def test_wait_mode_timeout_aborts(self, tmp_path, monkeypatch):
        """Wait mode ON, but the NAS never comes back → WalkAborted after
        all once the maximum wait time is up."""
        from whisper_tm.scanner import filesystem as fsmod

        root = tmp_path / "nas"
        root.mkdir()

        def fake_scan(path, ig, ext):
            if path == str(root):
                subs = [(f"{path}/d{i}", f"d{i}", False) for i in range(10)]
                return [], subs, [], None
            return [], [], [], 112

        monkeypatch.setattr(fsmod, "_scan_one_dir", fake_scan)
        monkeypatch.setattr(fsmod, "_probe_dir_alive", lambda p, timeout: False)

        with pytest.raises(fsmod.WalkAborted, match="did not return"):
            collect_media_files(
                root,
                recursive=True,
                media_extensions={".mp4"},
                ignore_patterns=[".*"],
                workers=2,
                wait_for_mount=True,
                max_error_streak=4,
                probe_sec=0.01,
                wait_max_min=0.001,  # ~60ms
            )

    def test_permission_errors_do_not_abort(self, tmp_path, monkeypatch):
        """Ordinary errors (EACCES=13) are NOT transport errors — even
        many of them in a row don't abort the walk."""
        from whisper_tm.scanner import filesystem as fsmod

        root = tmp_path / "nas"
        root.mkdir()

        def fake_scan(path, ig, ext):
            if path == str(root):
                subs = [(f"{path}/d{i}", f"d{i}", False) for i in range(400)]
                return [f"{path}/ok.mp4"], subs, [], None
            return [], [], [], 13  # EACCES

        monkeypatch.setattr(fsmod, "_scan_one_dir", fake_scan)
        files = collect_media_files(
            root,
            recursive=True,
            media_extensions={".mp4"},
            ignore_patterns=[".*"],
            workers=4,
        )
        assert len(files) == 1

    def test_should_stop_aborts_early(self, media_dir):
        """should_stop → walk ends cleanly (partial result, no error)."""
        files = collect_media_files(
            media_dir,
            recursive=True,
            media_extensions=MEDIA_EXT,
            ignore_patterns=[".*"],
            workers=8,
            should_stop=lambda: True,
        )
        # Root directory was still processed, then stop — no raise
        assert len(files) <= 5

    def test_unreadable_subdir_does_not_abort(self, media_dir):
        """An unreadable subdirectory must not abort the walk."""
        import os

        locked = media_dir / "locked"
        locked.mkdir()
        (locked / "inside.mp4").write_text("x")
        os.chmod(locked, 0o000)
        try:
            files = collect_media_files(
                media_dir,
                recursive=True,
                media_extensions=MEDIA_EXT,
                ignore_patterns=[".*"],
                workers=8,
            )
            names = {f.name for f in files}
            assert "video1.mp4" in names  # the remainder was scanned anyway
        finally:
            os.chmod(locked, 0o755)


class TestScanProgressPhases:
    """scan_media_directory reports walk/db/process phases for the SSE stream."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        from whisper_tm.config import config

        if not config.is_loaded:
            config.load()

    def test_phases_emitted(self, media_dir, db):
        from whisper_tm.scanner import scan_media_directory

        events = []
        result = scan_media_directory(
            media_dir, recursive=True, ultra=True, db=db, on_progress=events.append
        )
        phases = {e["phase"] for e in events}
        assert {"walk", "db", "process"} <= phases
        proc = [e for e in events if e["phase"] == "process"]
        # Last process event = everything processed
        assert proc[-1]["done"] == proc[-1]["total"] == result.total


class TestScanFromDb:
    """Walk-free scan: file list exclusively from the DB (auto-walk off)."""

    def _seed(self, db, vdir):
        db.bulk_upsert_media(
            [
                {
                    "rel_path": "a.mp4",
                    "abs_path": f"{vdir}/a.mp4",
                    "display_name": "a.mp4",
                    "subfolder": "",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 100,
                    "modified_at": 1.0,
                    "status": "pending",
                },
                {
                    "rel_path": "sub/b.mp3",
                    "abs_path": f"{vdir}/sub/b.mp3",
                    "display_name": "b.mp3",
                    "subfolder": "sub",
                    "extension": ".mp3",
                    "is_video": False,
                    "file_size": 50,
                    "modified_at": 2.0,
                    "status": "pending",
                },
                # File OUTSIDE the directory → must not show up
                {
                    "rel_path": "elsewhere.mp4",
                    "abs_path": "/other/elsewhere.mp4",
                    "display_name": "elsewhere.mp4",
                    "subfolder": "",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 10,
                    "modified_at": 3.0,
                    "status": "pending",
                },
            ]
        )

    def test_scoped_to_directory(self, db, tmp_path):
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        result = scan_media_directory_from_db(vdir, db=db)
        assert result.error is None
        assert result.source == "db"
        names = {f["name"] for f in result.files}
        assert names == {"a.mp4", "sub/b.mp3"}

    def test_entry_shape(self, db, tmp_path):
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        result = scan_media_directory_from_db(vdir, db=db)
        by_name = {f["name"]: f for f in result.files}
        a = by_name["a.mp4"]
        assert a["display_name"] == "a.mp4"
        assert a["ext"] == ".mp4"
        assert a["is_video"] is True
        assert a["size"] == 100
        assert a["size_fmt"]
        assert a["status"] == "pending"
        assert a["indexed"] is False
        b = by_name["sub/b.mp3"]
        assert b["subfolder"] == "sub"

    def test_indexed_metadata_attached(self, db, tmp_path):
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        db.bulk_set_media_metadata(
            [
                {
                    "rel_path": "a.mp4",
                    "abs_path": f"{vdir}/a.mp4",
                    "duration_sec": 42.5,
                    "has_audio": True,
                    "has_video": True,
                    "width": 1920,
                    "height": 1080,
                    "audio_codec": "aac",
                    "video_codec": "h264",
                    "bitrate": 5000,
                    "indexed_at": time.time(),
                    "index_mtime": 1.0,
                }
            ]
        )
        result = scan_media_directory_from_db(vdir, db=db)
        a = {f["name"]: f for f in result.files}["a.mp4"]
        assert a["indexed"] is True
        assert a["media_duration_sec"] == 42.5
        assert a["media_width"] == 1920
        assert result.index["indexed"] == 1
        assert result.index["unindexed"] == 1

    def test_transcript_join_marks_done(self, db, tmp_path):
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        db.upsert_transcript(
            "sub/b.mp3",
            {
                "complete": True,
                "model": "large-v3",
                "language": "de",
                "segments": [{"start": 0, "end": 1, "text": "hello world"}],
                "total_duration_sec": 1.0,
            },
            media_dir=str(vdir),
        )
        result = scan_media_directory_from_db(vdir, db=db)
        b = {f["name"]: f for f in result.files}["sub/b.mp3"]
        assert b["status"] == "done"
        assert b["tx_model"] == "large-v3"
        assert result.done == 1

    def test_no_db_returns_error(self, tmp_path):
        result = scan_media_directory_from_db(tmp_path, db=None)
        assert result.error

    def test_count_media_files_under(self, db, tmp_path):
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        assert db.count_media_files_under(str(vdir)) == 2
        assert db.count_media_files_under("/other") == 1
        assert db.count_media_files_under("/nothing") == 0


class TestSingleEntryRefresh:
    """Incremental list update: reload ONE file instead of a full scan.

    Previously, after every finished transcription the frontend reloaded
    the entire library (several seconds for a large library, every few
    seconds). What matters is that the single lookup delivers the same
    entry as the full scan —
    otherwise the two paths drift apart.
    """

    _seed = TestScanFromDb._seed

    def _prefix(self, vdir):
        import os

        return str(vdir).rstrip("/\\") + os.sep

    def test_matches_full_scan_entry(self, db, tmp_path):
        """Core promise: single lookup == entry from the full scan."""
        from whisper_tm.scanner import db_row_to_entry

        vdir = tmp_path / "media"
        self._seed(db, vdir)
        db.bulk_set_media_metadata(
            [
                {
                    "rel_path": "a.mp4",
                    "abs_path": f"{vdir}/a.mp4",
                    "duration_sec": 42.5,
                    "has_audio": True,
                    "has_video": True,
                    "width": 1920,
                    "height": 1080,
                    "audio_codec": "aac",
                    "video_codec": "h264",
                    "bitrate": 5000,
                    "indexed_at": time.time(),
                    "index_mtime": 1.0,
                }
            ]
        )
        db.upsert_transcript(
            "sub/b.mp3",
            {
                "complete": True,
                "model": "large-v3",
                "language": "de",
                "segments": [{"start": 0, "end": 1, "text": "hello world"}],
                "total_duration_sec": 1.0,
            },
            media_dir=str(vdir),
        )

        full = {f["name"]: f for f in scan_media_directory_from_db(vdir, db=db).files}
        prefix = self._prefix(vdir)

        for name in ("a.mp4", "sub/b.mp3"):
            row = db.get_media_file_at(prefix + name)
            assert row is not None, name
            assert db_row_to_entry(row, prefix) == full[name], name

    def test_unknown_path_returns_none(self, db, tmp_path):
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        assert db.get_media_file_at(self._prefix(vdir) + "missing.mp4") is None

    def test_prefix_is_not_a_substring_match(self, db, tmp_path):
        """A dot query must not jump to a prefix neighbour."""
        vdir = tmp_path / "media"
        self._seed(db, vdir)
        # "a.mp" is a prefix of "a.mp4" — must NOT match
        assert db.get_media_file_at(self._prefix(vdir) + "a.mp") is None

    def test_transcript_row_wins_over_plain(self, db, tmp_path):
        """Duplicate (same abs_path): the row with the transcript wins —
        just like in the full scan."""
        from whisper_tm.scanner import db_row_to_entry

        vdir = tmp_path / "media"
        self._seed(db, vdir)
        # Second rel_path row on the same abs_path (media_dir change)
        db.bulk_upsert_media(
            [
                {
                    "rel_path": "old_value/a.mp4",
                    "abs_path": f"{vdir}/a.mp4",
                    "display_name": "a.mp4",
                    "subfolder": "old_value",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 100,
                    "modified_at": 1.0,
                    "status": "pending",
                },
            ]
        )
        # The transcript belongs to the SAME file — so the same abs_path.
        # The two rel_path values may differ (root change), but not the
        # identity (docs/PATHS.md).
        db.upsert_transcript(
            "old_value/a.mp4",
            {
                "complete": True,
                "model": "large-v3",
                "language": "de",
                "segments": [{"start": 0, "end": 1, "text": "hello"}],
                "total_duration_sec": 1.0,
            },
            media_dir=str(vdir),
            abs_path=f"{vdir}/a.mp4",
        )

        prefix = self._prefix(vdir)
        entry = db_row_to_entry(db.get_media_file_at(prefix + "a.mp4"), prefix)
        full = {f["name"]: f for f in scan_media_directory_from_db(vdir, db=db).files}
        # Both paths must pick the same row as the "best" one
        assert entry == full[entry["name"]]
        assert entry["status"] == "done"

    def test_like_wildcards_in_path_are_escaped(self, db, tmp_path):
        """_ and % in the directory name must not act as a LIKE wildcard."""
        vdir = tmp_path / "my_dir"
        # my_dir and myXdir: with an unescaped '_', myXdir would also match
        db.bulk_upsert_media(
            [
                {
                    "rel_path": "x.mp4",
                    "abs_path": f"{tmp_path}/my_dir/x.mp4",
                    "display_name": "x.mp4",
                    "subfolder": "",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 1,
                    "modified_at": 1.0,
                    "status": "pending",
                },
                {
                    "rel_path": "y.mp4",
                    "abs_path": f"{tmp_path}/myXdir/y.mp4",
                    "display_name": "y.mp4",
                    "subfolder": "",
                    "extension": ".mp4",
                    "is_video": True,
                    "file_size": 1,
                    "modified_at": 1.0,
                    "status": "pending",
                },
            ]
        )
        result = scan_media_directory_from_db(vdir, db=db)
        names = {f["name"] for f in result.files}
        assert names == {"x.mp4"}


# ── Names that are not valid UTF-8 ────────────────────────────────────────
#
# A Latin-1 name from an old archive reaches Python with stand-in characters
# that SQLite refuses. The scan wrote all its files in one go and logged the
# failure at debug level: one such name kept every new file of that scan out
# of the library.


def _odd_names(folder):
    """A file and a folder whose names hold a byte that is not UTF-8."""
    import os

    base = os.fsencode(str(folder))
    try:
        with open(os.path.join(base, b"caf\xe9 talk.mp4"), "wb") as f:
            f.write(b"x")
        os.mkdir(os.path.join(base, b"tapes\xff"))
        with open(os.path.join(base, b"tapes\xff", b"inside.mp4"), "wb") as f:
            f.write(b"x")
    except OSError:
        pytest.skip("this file system takes only UTF-8 names")


@pytest.mark.parametrize("walk", ["serial", "parallel"])
def test_a_name_that_is_not_utf8_is_skipped_and_named(media_dir, caplog, walk):
    from whisper_tm.scanner import filesystem

    _odd_names(media_dir)
    filesystem._bad_names_reported.clear()
    with caplog.at_level("WARNING", logger="whisper_tm.scanner.filesystem"):
        if walk == "serial":
            found = list(
                iter_media_files(
                    media_dir, recursive=True, media_extensions=MEDIA_EXT, ignore_patterns=[".*"]
                )
            )
        else:
            found = collect_media_files(
                media_dir,
                recursive=True,
                media_extensions=MEDIA_EXT,
                ignore_patterns=[".*"],
                workers=4,
            )
    names = {p.name for p in found}
    assert names == {"video1.mp4", "video2.mkv", "audio.mp3", "deep.mp4", "deeper.mp3"}
    for p in found:
        str(p).encode("utf-8")  # every path can be stored
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "caf\\xe9 talk.mp4" in said and "tapes\\xff" in said and "not valid UTF-8" in said


def test_the_other_files_of_such_a_scan_reach_the_library(media_dir, db, monkeypatch):
    from whisper_tm.config import config
    from whisper_tm.scanner import scan_media_directory

    _odd_names(media_dir)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media_dir))
    result = scan_media_directory(media_dir, recursive=True, db=db)
    assert result.error is None
    with db.connection() as conn:
        stored = {r[0].rsplit("/", 1)[-1] for r in conn.execute("SELECT abs_path FROM media_files")}
    assert {"video1.mp4", "video2.mkv", "audio.mp3", "deep.mp4", "deeper.mp3"} <= stored
    assert not any("caf" in n or "inside" in n for n in stored)

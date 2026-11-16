"""Root and filename overlap — existing files were considered gone.

**Finding from a running batch.** In the queue were hundreds of
entries like this:

    filename : videos/Archive/Videos/meeting_2024-03-01.mp4   (from media root)
    base_dir : /mnt/media/videos/Archive/Videos               (subdirectory)

The worker builds ``base_dir + filename`` and ends up at

    /mnt/media/videos/Archive/Videos/videos/Archive/Videos/meeting_2024-03-01.mp4

That doesn't exist — so "❌ Not found" for every one of them, while the
files lay untouched on the disk.

The combination comes from the frontend: `batch.js` sends as
``base_dir`` the currently DISPLAYED directory, but the filenames are
relative to the media root. If the two diverge, every entry
in this batch is affected — silently.

The worker repairs it now, but NARROWLY: only the media root is tried as
an alternative, it must be allowed, and the file must be there.
These tests hold both fast — that it's repaired, and that no guessing occurs.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def environment(tmp_path, monkeypatch):
    """Media root with a file in a subfolder."""
    from whisper_tm.config import config

    root = tmp_path / "nas"
    (root / "videos" / "Archive").mkdir(parents=True)
    file = root / "videos" / "Archive" / "film.mp4"
    file.write_bytes(b"x" * 2048)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: root))
    return root, file


class TestRepair:
    def test_an_overlapping_root_is_found(self, environment):
        """The case from operations."""
        from whisper_tm.core import paths

        root, file = environment
        name = "videos/Archive/film.mp4"
        subfolders = str(root / "videos" / "Archive")

        # This is how the worker computes it first — and misses:
        wrong = paths.abs_from(name, subfolders)
        assert wrong and not __import__("pathlib").Path(wrong).exists()

        # The fix uses the media root:
        correct = paths.abs_from(name, str(root))
        assert __import__("pathlib").Path(correct).exists()
        assert correct == str(file)

    def test_the_repair_does_not_leave_the_root(self, environment):
        """Nothing is guessed: abs_from still rejects traversal."""
        from whisper_tm.core import paths

        root, _ = environment
        assert paths.abs_from("../../etc/passwd", str(root)) == ""

    def test_without_an_overlap_everything_stays_as_it_was(self, environment):
        """The normal case must not change: filename relative to
        base_dir, and the file is there too."""
        from whisper_tm.core import paths

        root, file = environment
        subfolders = str(root / "videos" / "Archive")
        assert paths.abs_from("film.mp4", subfolders) == str(file)


class TestTheWorkerRepairs:
    """The fix must live IN the worker, not just as an idea."""

    def test_the_place_exists_and_is_narrowly_scoped(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "queue.py"
        ).read_text(encoding="utf-8")
        place = source[source.index("if not source.exists():") :]
        # Cut generously: the explanatory comment at this spot is
        # long, and too narrow a window made the test fail even though the
        # checked line was there.
        place = place[:4000]

        assert "_trim_overlap" in place, "the overlap is not being trimmed at all"
        assert "config.media_dir" in place, "the media root is not tried as a second option"
        assert "root_allowed(cand)" in place, (
            "the fix doesn't check the allowed roots — that would be a hole"
        )
        assert ".exists()" in place, "it's not being checked whether the file is really there"
        assert "Not found" in place, "without a match the entry must still end as an error"


class TestTrimmingTheOverlap:
    """The actual computation path — and why the first version wasn't enough.

    The first attempt was: "if the file isn't found under base_dir,
    use the media root". That missed: with the fix deployed and the
    server restarted, the same entries still failed.

    Reason: ``config.media_dir`` moves along while navigating. If the
    display was in the same subfolder, the "alternative" was identical to the
    wrong path — and the own condition ``replacement != target`` skipped
    the fix. That's why now the OVERLAP itself is trimmed, instead of
    hoping for a different root.
    """

    def test_the_production_case(self):
        from whisper_tm.core.queue import _trim_overlap as shorten

        assert (
            shorten(
                "videos/Archive/Videos/meeting_2024-03-01.mp4", "/mnt/media/videos/Archive/Videos"
            )
            == "meeting_2024-03-01.mp4"
        )

    def test_without_an_overlap_it_stays_empty(self):
        from whisper_tm.core.queue import _trim_overlap as shorten

        assert shorten("film.mp4", "/mnt/media/videos") == ""
        assert shorten("videos/Archive/film.mp4", "/mnt/media") == ""

    def test_a_name_always_remains(self):
        """Without this limit, the whole name could disappear."""
        from whisper_tm.core.queue import _trim_overlap as shorten

        assert shorten("videos", "/mnt/media/videos") == ""

    def test_the_longest_piece_wins(self):
        """For "a/a/file" under "…/a/a" not just one "a" may be dropped."""
        from whisper_tm.core.queue import _trim_overlap as shorten

        assert shorten("a/a/file.mp4", "/nas/a/a") == "file.mp4"

    def test_a_partial_match_does_not_count(self):
        """ "Video" is not "Videos" — a name part must match fully."""
        from whisper_tm.core.queue import _trim_overlap as shorten

        assert shorten("Videos/f.mp4", "/nas/Video") == ""

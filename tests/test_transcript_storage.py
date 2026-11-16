"""
Where transcripts land.

``transcript_mode: alongside`` meant until recently "a ``.transcripts/`` in
the media directory" — so ONE flat collection folder at the root of the
directory currently selected in the frontend, with ``__`` as encoded separator.

Two consequences:

* The media directory changes at runtime. If you went into a subfolder
  and transcribed there, you got a SECOND collection folder — transcripts
  scattered across as many folders as you opened directories.
* The flat name has been non-unique since schema v8: two same-named
  files from different folders pointed to the same JSON file.

Now the transcript lies in ``.transcripts/`` NEXT TO the media file, and the
location depends only on its actual path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper_tm.config import config
from whisper_tm.core.helpers import find_transcript, transcript_path_for


@pytest.fixture
def library(tmp_path, monkeypatch):
    media = tmp_path / "media"
    for sub in ("videos/series", "music"):
        (media / sub).mkdir(parents=True)
    (media / "videos" / "series" / "episode1.mp4").write_bytes(b"x")
    (media / "music" / "episode1.mp4").write_bytes(b"x")

    state = {"root": media}
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: state["root"]))
    monkeypatch.setattr(
        type(config), "transcript_dir", property(lambda self: state["root"] / ".transcripts")
    )
    monkeypatch.setattr(type(config), "transcript_mode", property(lambda self: "alongside"))
    return {"media": media, "state": state}


class TestAlongside:
    def test_lies_next_to_media_file(self, library):
        media = library["media"]
        tp = transcript_path_for("videos/series/episode1.mp4")
        assert tp == (media / "videos" / "series" / ".transcripts" / "episode1.mp4.json")

    def test_no_flat_collection_folder_anymore(self, library):
        tp = transcript_path_for("videos/series/episode1.mp4")
        assert "__" not in tp.name
        assert tp.parent != library["media"] / ".transcripts"

    def test_independent_from_selected_directory(self, library):
        """Same location regardless of which root the file is named from.

        This is the core: ``config.media_dir`` changes in the frontend, the
        storage location must not move with it.
        """
        media, state = library["media"], library["state"]
        a = transcript_path_for("videos/series/episode1.mp4")
        state["root"] = media / "videos"
        b = transcript_path_for("series/episode1.mp4")
        state["root"] = media / "videos" / "series"
        c = transcript_path_for("episode1.mp4")
        assert a == b == c

    def test_same_named_files_dont_collide(self, library):
        a = transcript_path_for("videos/series/episode1.mp4")
        b = transcript_path_for("music/episode1.mp4")
        assert a != b

    def test_abs_path_takes_precedence(self, library):
        """Whoever knows the actual location provides it — then nothing
        is reconstructed from rel_path + root (docs/PATHS.md)."""
        media = library["media"]
        real = media / "music" / "episode1.mp4"
        tp = transcript_path_for("videos/series/episode1.mp4", abs_path=real)
        assert tp == media / "music" / ".transcripts" / "episode1.mp4.json"


class TestExactlyOneLocation:
    """Search is done in ONE location — no fallback to a second storage.

    A permanent second path would be bridge code that no one can explain
    later.
    """

    def test_old_central_name_no_longer_found(self, library):
        media = library["media"]
        central = media / ".transcripts" / "videos__series__episode1.mp4.json"
        central.parent.mkdir(parents=True, exist_ok=True)
        central.write_text(json.dumps({"complete": True, "segments": []}), encoding="utf-8")

        assert find_transcript("videos/series/episode1.mp4") is None

    def test_finds_file_at_correct_location(self, library):
        media = library["media"]
        next_to = media / "videos" / "series" / ".transcripts" / "episode1.mp4.json"
        next_to.parent.mkdir(parents=True, exist_ok=True)
        next_to.write_text("{}", encoding="utf-8")

        assert find_transcript("videos/series/episode1.mp4") == next_to

    def test_reports_if_identity_cannot_be_derived(self, library, monkeypatch):
        """No fallback to a guessed location — prefer a clear error."""
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: Path("relative")))
        with pytest.raises(ValueError, match="No transcript path"):
            transcript_path_for("episode1.mp4")


class TestCentral:
    """The 'central' mode is explicitly ONE collection folder."""

    @pytest.fixture
    def central(self, tmp_path, monkeypatch):
        c = tmp_path / "central"
        monkeypatch.setattr(type(config), "transcript_mode", property(lambda self: "central"))
        monkeypatch.setattr(type(config), "transcript_dir", property(lambda self: c))
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: tmp_path / "media"))
        return c

    def test_encodes_absolute_path(self, central, tmp_path):
        """The ABSOLUTE path, not the rel_path.

        Via rel_path the name was not unique: two same-named files
        from different folders ended up on the same JSON file and
        overwrote each other (docs/PATHS.md).
        """
        tp = transcript_path_for("videos/series/episode1.mp4")
        expected = str(tmp_path / "media" / "videos" / "series" / "episode1.mp4")
        assert tp.name == expected.strip("/").replace("/", "__") + ".json"

    def test_same_named_files_dont_collide(self, central):
        a = transcript_path_for("videos/series/episode1.mp4")
        b = transcript_path_for("music/episode1.mp4")
        assert a != b

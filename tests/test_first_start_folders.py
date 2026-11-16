"""The first start leaves ~/Videos alone.

Before a library was chosen, the first start made uploads/, .transcripts/
and Whisper-TM-Clips in ~/Videos — the default library — even when the
library was going to be somewhere else. Folders inside the library now come
once one is chosen, the default clips folder with the first clip; the
app's own folders as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from whisper_tm.config import ConfigManager

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "Videos").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("VIDEO_DIR", raising=False)
    for key in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        monkeypatch.delenv(key, raising=False)
    return h


def _load(tmp_path: Path, directories: dict) -> ConfigManager:
    file = tmp_path / "cfg" / "config.yaml"
    file.parent.mkdir(exist_ok=True)
    file.write_text(yaml.safe_dump({"directories": directories}), encoding="utf-8")
    c = ConfigManager()
    c.load(file)
    return c


def test_before_a_library_is_chosen_videos_stays_untouched(home, tmp_path):
    c = _load(tmp_path, {"media": ""})
    assert c.media_dir == home / "Videos"
    assert list((home / "Videos").iterdir()) == []
    # the app's own folders are there
    assert c.work_dir.is_dir() and c.dirs["logs"].is_dir() and c.db_path.parent.is_dir()


def test_a_chosen_library_gets_its_folders(home, tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    c = _load(tmp_path, {"media": str(library)})
    assert (library / "uploads").is_dir() and (library / ".transcripts").is_dir()
    # the default clips folder only with the first clip
    assert c.dirs["clips"] == home / "Videos" / "Whisper-TM-Clips"
    assert not c.dirs["clips"].exists()


def test_folders_set_by_hand_are_made(home, tmp_path):
    clips, uploads = tmp_path / "my-clips", tmp_path / "incoming"
    _load(tmp_path, {"media": "", "clips": str(clips), "uploads": str(uploads)})
    assert clips.is_dir() and uploads.is_dir()
    assert list((home / "Videos").iterdir()) == []


def test_central_transcripts_are_the_apps_own(home, tmp_path):
    c = _load(tmp_path, {"media": "", "transcript_mode": "central"})
    assert c.transcript_dir.is_dir()
    assert list((home / "Videos").iterdir()) == []


def test_choosing_a_library_later_makes_its_transcript_folder(home, tmp_path):
    c = _load(tmp_path, {"media": ""})
    library = tmp_path / "library"
    library.mkdir()
    c.media_dir = library
    assert (library / ".transcripts").is_dir()


def test_the_first_clip_makes_the_clips_folder():
    """Cutting a clip and a listening sample write into it."""
    media = (REPO / "whisper_tm" / "api" / "media.py").read_text(encoding="utf-8")
    cut = media[media.index("def extract_clip(") :]
    cut = cut[: cut.index("\n@media_bp.route")]
    assert "clips_dir.mkdir(parents=True, exist_ok=True)" in cut
    sample = media[media.index('target = config.dirs["clips"]') :]
    sample = sample[: sample.index("\n@media_bp.route")]
    assert "target.parent.mkdir(parents=True, exist_ok=True)" in sample


def test_a_missing_default_library_is_a_choice_not_an_error(tmp_path, monkeypatch):
    """A first start without ~/Videos showed a red 'Directory not found'."""
    from whisper_tm.app import create_app
    from whisper_tm.config import config

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    config.load(str(cfg))
    c = create_app(str(cfg)).test_client()
    d = c.get("/api/scan").get_json()
    assert d["error"] and d["unchosen"] is True
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "gone")
    config._resolve_directories()
    assert c.get("/api/scan").get_json()["unchosen"] is False
    config.load(None)

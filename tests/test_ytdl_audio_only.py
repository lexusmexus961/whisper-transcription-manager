"""
"Audio only" is now a toggle in the add dialog, not just a setting. Key point:
the toggle overrides the config only for THIS download, and a missing toggle
(old clients, API use) must still use the setting — that's why None instead of
False as default.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import ytdl


@pytest.fixture(autouse=True)
def clean_config():
    from whisper_tm.config import config

    config.set("youtube", "audio_only", False)
    config.set("youtube", "extra_args", "")
    yield
    config.set("youtube", "audio_only", False)


def test_none_takes_the_setting():
    from whisper_tm.config import config

    assert "-x" not in ytdl._format_args(None)
    config.set("youtube", "audio_only", True)
    assert "-x" in ytdl._format_args(None)


def test_the_switch_beats_the_setting():
    from whisper_tm.config import config

    # setting off, toggle on
    assert "-x" in ytdl._format_args(True)
    # setting on, toggle off → video (was impossible before)
    config.set("youtube", "audio_only", True)
    args = ytdl._format_args(False)
    assert "-x" not in args
    assert "--merge-output-format" in args


def test_extra_args_stay_in_both_modes():
    from whisper_tm.config import config

    config.set("youtube", "extra_args", "--limit-rate 5M")
    assert ytdl._format_args(True)[-2:] == ["--limit-rate", "5M"]
    assert ytdl._format_args(False)[-2:] == ["--limit-rate", "5M"]


def test_the_route_passes_the_switch_through(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    app = create_app()
    client = app.test_client()

    import whisper_tm.extensions as ext

    class _HW:
        has_ytdlp = True

    monkeypatch.setattr(ext, "hw", _HW())
    seen = {}
    import whisper_tm.core.ytdl as y

    monkeypatch.setattr(
        y,
        "start_download",
        lambda url, audio_only=None: seen.update(url=url, audio_only=audio_only) or "job1",
    )

    r = client.post(
        "/api/youtube/download", json={"url": "https://example.com/v", "audio_only": True}
    )
    assert r.status_code == 200
    assert seen["audio_only"] is True

    client.post("/api/youtube/download", json={"url": "https://example.com/v", "audio_only": False})
    assert seen["audio_only"] is False

    # Without a field: None → the worker takes the setting
    client.post("/api/youtube/download", json={"url": "https://example.com/v"})
    assert seen["audio_only"] is None

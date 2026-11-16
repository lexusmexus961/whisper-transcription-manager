"""Spare VRAM switches itself on for a small card.

``hardware.low_vram`` was off unless set: an 8 GB card kept every model
resident and ran into the out-of-memory retry with every long file. Its
default is now "auto" — on for a card of up to 11 GB — and the settings page
and the dialogs show what that means on the machine they run on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from whisper_tm.core import debug, vram


@pytest.fixture
def card(monkeypatch):
    """The card's size in MB (None: no card), as NVML reports it."""
    monkeypatch.setattr(vram, "_card", None)
    sizes = {"mb": None}
    monkeypatch.setattr(
        debug,
        "gpu_static",
        lambda: {0: {"name": "card", "total_mb": sizes["mb"]}} if sizes["mb"] else {},
    )
    monkeypatch.setitem(sys.modules, "torch", None)  # NVML's answer only
    yield sizes
    vram._card = None


def _setting(monkeypatch, value):
    real = vram.config.get

    def fake(*path, default=None):
        if path == ("hardware", "low_vram"):
            return value
        return real(*path, default=default)

    monkeypatch.setattr(vram.config, "get", fake)


@pytest.mark.parametrize(
    "mb, on",
    [
        (4096, True),
        (6144, True),
        (8192, True),
        (10240, True),
        (11264, True),
        (12288, False),
        (16384, False),
        (24576, False),
    ],
)
def test_auto_follows_the_card(card, monkeypatch, mb, on):
    card["mb"] = mb
    _setting(monkeypatch, "auto")
    assert vram.default_low_vram() is on


def test_without_a_card_it_stays_off(card, monkeypatch):
    _setting(monkeypatch, "auto")
    assert vram.default_low_vram() is False


@pytest.mark.parametrize(
    "value, on",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("off", False),
        ("", True),
        (None, True),
        ("whatever", True),
    ],
)
def test_a_choice_of_ones_own_wins(card, monkeypatch, value, on):
    """On an 8 GB card: true and false decide, anything else is "auto"."""
    card["mb"] = 8192
    _setting(monkeypatch, value)
    assert vram.default_low_vram() is on


def test_the_card_the_jobs_run_on(card, monkeypatch):
    monkeypatch.setattr(
        debug, "gpu_static", lambda: {0: {"total_mb": 24576}, 1: {"total_mb": 8192}}
    )
    real = vram.config.get
    monkeypatch.setattr(
        vram.config,
        "get",
        lambda *path, default=None: (
            1 if path == ("hardware", "gpu_index") else real(*path, default=default)
        ),
    )
    assert vram.card_gb() == 8.0


def test_an_entrys_own_choice_still_wins(card, monkeypatch):
    card["mb"] = 8192
    _setting(monkeypatch, "auto")
    vram.set_low_vram_override(False)
    try:
        assert vram.low_vram() is False
    finally:
        vram.set_low_vram_override(None)
    assert vram.low_vram() is True


def test_the_settings_page_shows_what_auto_means(card, monkeypatch):
    from flask import Flask

    from whisper_tm.api.settings import settings_bp
    from whisper_tm.config import config

    card["mb"] = 8192
    monkeypatch.setitem(config._data.setdefault("hardware", {}), "low_vram", "auto")
    app = Flask(__name__)
    app.register_blueprint(settings_bp)
    data = app.test_client().get("/api/settings").get_json()
    assert data["config"]["hardware"]["low_vram"] is True
    assert data["defaults"]["hardware"]["low_vram"] is True  # no "changed" mark
    monkeypatch.setitem(config._data["hardware"], "low_vram", False)
    data = app.test_client().get("/api/settings").get_json()
    assert data["config"]["hardware"]["low_vram"] is False


def test_the_queue_asks_the_same_question():
    source = (
        Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "queue.py"
    ).read_text(encoding="utf-8")
    assert 'bool(config.get("hardware", "low_vram"' not in source
    assert "default_low_vram()" in source


def test_a_card_taken_to_be_smaller_is_spared_like_one(card, monkeypatch):
    # hardware.assume_vram_mb: for testing the smaller classes on a big card
    card["mb"] = 12288
    real = vram.config.get
    monkeypatch.setattr(
        vram.config,
        "get",
        lambda *path, default=None: (
            8192
            if path == ("hardware", "assume_vram_mb")
            else "auto"
            if path == ("hardware", "low_vram")
            else real(*path, default=default)
        ),
    )
    assert vram.card_gb() == 8.0 and vram.default_low_vram() is True

"""The defaults that follow the machine.

large-v3 with vocal isolation and three files at once suit a 12 GB card.
On a CPU they are hours per recording, and a 4 or 6 GB card cannot run them
at all (the separation model alone holds about 6.5 GB). Only setup.sh --gpu
cpu wrote other values, into config.yaml — Docker and an install whose GPU
was merely not found got the 12 GB values.

The app now chooses model, vocal isolation and files at once from the
hardware it finds, once at start, as *defaults*: a value the user set wins,
and nothing of it is written into config.yaml.
"""

from __future__ import annotations

import pytest
import yaml

from whisper_tm.config import ConfigManager
from whisper_tm.hardware import GPUInfo, HardwareProfile, machine_defaults, machine_defaults_text

CPU = {
    "transcription.default_model": "turbo",
    "vocal_isolation.default_enabled": False,
    "transcription.speaker_window_separation": False,
    "queue.lanes": 1,
}
#: A 6 GB card: the 30-second windows can be separated for the voice vectors
SIX = {**CPU, "transcription.speaker_window_separation": True}
MID = {
    "transcription.default_model": "large-v3",
    "vocal_isolation.default_enabled": True,
    "transcription.speaker_window_separation": True,
    "queue.lanes": 1,
}


def _machine(mb: int = 0, device: str = "cuda", assume: int = 0) -> HardwareProfile:
    p = HardwareProfile(best_device=device, assume_vram_mb=assume)
    if mb:
        p.gpus = [GPUInfo(index=0, name="card", vendor="nvidia", memory_total_mb=mb)]
    return p


@pytest.mark.parametrize(
    "mb, expected",
    [
        (2048, CPU),
        (4096, CPU),  # the separation model does not fit at all
        (6144, SIX),  # its 30-second windows do
        (8192, MID),
        (10240, MID),
        (11264, MID),  # large-v3, a file at a time
        (12288, {}),
        (16384, {}),
        (24576, {}),
        (49152, {}),  # the shipped values
        (12045, {}),
        (7900, MID),
        (5900, SIX),  # cards report a little less than their name
    ],
)
def test_each_class_of_card(mb, expected):
    values, what = machine_defaults(_machine(mb))
    assert values == expected, what


@pytest.mark.parametrize(
    "profile",
    [
        _machine(0, "cpu"),
        _machine(12288, "cpu"),  # no card, or a card that is not used
        _machine(16384, "mps"),  # WhisperX has no MPS backend
    ],
)
def test_without_a_gpu_that_computes_it_is_what_a_cpu_manages(profile):
    values, what = machine_defaults(profile)
    assert values == CPU and "CPU" in what


def test_a_card_can_be_taken_to_be_smaller():
    assert machine_defaults(_machine(12288, assume=6144))[0] == SIX
    assert machine_defaults(_machine(12288, assume=4096))[0] == CPU
    assert machine_defaults(_machine(12288, assume=8192))[0] == MID
    assert machine_defaults(_machine(12288))[0] == {}


def test_the_sentence_says_what_was_chosen():
    said = machine_defaults_text(*machine_defaults(_machine(0, "cpu")))
    assert "Whisper turbo" in said and "no vocal isolation" in said and "one file at a time" in said
    assert "no voice vectors" in said
    assert "no voice vectors" not in machine_defaults_text(*machine_defaults(_machine(6144)))
    assert "standard values" in machine_defaults_text(*machine_defaults(_machine(24576)))


# ── In the config: defaults, not settings ────────────────────────────────


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    def make(text: str = "{}\n") -> ConfigManager:
        f = tmp_path / "config.yaml"
        f.write_text(text, encoding="utf-8")
        c = ConfigManager()
        c.load(f)
        return c

    return make


def test_they_apply_where_nothing_was_set(cfg):
    c = cfg()
    assert c.get("transcription", "default_model") == "large-v3" and c.get("queue", "lanes") == 3
    applied = c.set_machine_defaults(CPU)
    assert applied == CPU
    assert c.get("transcription", "default_model") == "turbo"
    assert c.get("vocal_isolation", "default_enabled") is False and c.get("queue", "lanes") == 1
    assert c.default("transcription.default_model") == "turbo"  # what "reset" goes back to


def test_what_the_user_set_wins_even_when_it_is_the_shipped_value(cfg):
    # Somebody who wants large-v3 on a small card writes it into config.yaml
    c = cfg("transcription:\n  default_model: large-v3\nqueue:\n  lanes: 2\n")
    applied = c.set_machine_defaults(CPU)
    assert applied == {
        "vocal_isolation.default_enabled": False,
        "transcription.speaker_window_separation": False,
    }
    assert c.get("transcription", "default_model") == "large-v3" and c.get("queue", "lanes") == 2


def test_nothing_of_it_is_written_into_the_file(cfg, tmp_path):
    c = cfg("server:\n  port: 5123\n")
    c.set_machine_defaults(CPU)
    c.save()
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved == {"server": {"port": 5123}}


def test_a_value_chosen_in_the_settings_is_kept_and_saved(cfg, tmp_path):
    c = cfg()
    c.set_machine_defaults(CPU)
    c.set("transcription", "default_model", "large-v3")  # the user's choice on a small machine
    c.save()
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved == {"transcription": {"default_model": "large-v3"}}
    again = ConfigManager()
    again.load(tmp_path / "config.yaml")
    again.set_machine_defaults(CPU)
    assert again.get("transcription", "default_model") == "large-v3"


def test_the_next_load_starts_from_the_shipped_values_again(cfg, tmp_path):
    c = cfg()
    c.set_machine_defaults(CPU)
    c.load(tmp_path / "config.yaml")
    assert c.get("transcription", "default_model") == "large-v3" and c.machine_defaults == {}
    assert c.default("queue.lanes") == 3


def test_a_twelve_gb_card_changes_nothing(cfg):
    c = cfg()
    assert c.set_machine_defaults({}) == {} and c.get("queue", "lanes") == 3


# ── Where it is said ──────────────────────────────────────────────────────


def test_the_app_applies_them_at_start_and_says_so():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    app = (root / "whisper_tm" / "app.py").read_text(encoding="utf-8")
    assert "assume_vram_mb=" in app
    assert (
        app.index("ext.hw = detect_hardware(")
        < app.index("apply_machine_defaults(ext.hw)")
        < app.index("Database(")
    )
    prefetch = (root / "whisper_tm" / "core" / "prefetch.py").read_text(encoding="utf-8")
    assert "apply_machine_defaults()" in prefetch, (
        "--prefetch fetches the model this machine will use"
    )
    setup = (root / "setup.sh").read_text(encoding="utf-8")
    assert 'default_model: "turbo"' not in setup, "setup.sh writes no CPU profile any more"
    docker = yaml.safe_load((root / "config.docker.yaml").read_text(encoding="utf-8"))
    for key in ("default_model", "default_language", "default_diarize"):
        assert key not in docker["transcription"], "Docker has no defaults of its own"
    readme = (root / "docs" / "GUIDE.md").read_text(encoding="utf-8")
    assert "What the app chooses by itself" in readme
    pages = (root / "whisper_tm" / "static" / "js" / "settings_pages.js").read_text(
        encoding="utf-8"
    )
    assert "esc(hw.machine_defaults)" in pages, "Settings say what was chosen for the machine"
    settings = (root / "whisper_tm" / "api" / "settings.py").read_text(encoding="utf-8")
    assert '"machine_defaults": _machine_defaults_text(hw)' in settings


def test_language_detection_is_the_default():
    from pathlib import Path

    shipped = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(encoding="utf-8")
    )
    assert shipped["transcription"]["default_language"] == "auto"
    assert shipped["hardware"]["assume_vram_mb"] == 0


def test_applying_them_sets_the_config_and_returns_the_sentence(monkeypatch):
    from whisper_tm import hardware
    from whisper_tm.config import config

    config.load(None)
    try:
        profile, said = hardware.apply_machine_defaults(_machine(0, "cpu"))
        assert profile.best_device == "cpu" and "Whisper turbo" in said
        assert (
            config.get("transcription", "default_model") == "turbo"
            and config.get("queue", "lanes") == 1
        )
    finally:
        config.load(None)
    assert config.get("transcription", "default_model") == "large-v3"


# ── Batch and compute type by the size a card really reports ─────────────


@pytest.mark.parametrize(
    "mb, compute, batch",
    [
        (24250, "float16", 32),
        (16100, "float16", 16),
        (11931, "float16", 8),  # a 12 GB card, as torch reports it: what this project ran on
        (11100, "float16", 8),
        (10100, "float16", 8),
        (7950, "float16", 4),  # 8 GB: batch 8 ran out of memory with 7 GB free
        (5950, "float16", 4),
        (3950, "int8", 4),
        (1950, "int8", 2),
    ],
)
def test_batch_and_compute_type_follow_the_card(mb, compute, batch):
    from whisper_tm.hardware import _compute_recommendations

    p = _machine(mb)
    _compute_recommendations(p)
    assert (p.recommended_compute_type, p.recommended_batch_size) == (compute, batch)


def test_a_card_taken_to_be_smaller_gets_that_cards_batch():
    from whisper_tm.hardware import _compute_recommendations

    p = _machine(11931, assume=8192)
    _compute_recommendations(p)
    assert p.recommended_batch_size == 4


def test_the_wait_after_an_out_of_memory_goes_by_the_assumed_size(monkeypatch):
    from whisper_tm.core import vram

    real = vram.config.get
    monkeypatch.setattr(
        vram.config,
        "get",
        lambda *path, default=None: (
            8192 if path == ("hardware", "assume_vram_mb") else real(*path, default=default)
        ),
    )
    assert vram.total_mb() == 8192

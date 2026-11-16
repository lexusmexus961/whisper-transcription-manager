"""The PyTorch build follows the card's generation.

setup.sh installed the CUDA 12.8 build for every NVIDIA card. That build has
no code for cards older than compute capability 7.0 (GTX 10xx, GTX 9xx): the
card is found and Whisper runs, but alignment, diarization, separation and
the voice vectors stop with "no kernel image is available" in the first job.
The setup now takes the CUDA 12.6 build for such a card, and the app says at
start and in the system check when the build and the card don't fit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from whisper_tm.hardware import GPUInfo, HardwareProfile, build_misses_card, unsupported_card

ROOT = Path(__file__).resolve().parent.parent
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")
CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
CU126 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


@pytest.mark.parametrize(
    "cap, tag",
    [
        ("8.6", "cu128"),
        ("7.0", "cu128"),
        ("7.5", "cu128"),
        ("8.9", "cu128"),
        ("9.0", "cu128"),
        ("12.0", "cu128"),
        ("6.1", "cu126"),
        ("6.0", "cu126"),
        ("5.2", "cu126"),
        ("5.0", "cu126"),
        ("3.5", ""),
        ("3.0", ""),  # older than any build: the CPU
        ("", "cu128"),
        ("[N/A]", "cu128"),  # no answer: what PyPI brings
    ],
)
def test_the_setup_picks_the_build_for_the_cards_generation(tmp_path, cap, tag):
    at = SETUP.index("nvidia_torch_tag() {")
    fn = SETUP[at : SETUP.index("\n}\n", at) + 3]
    fake = tmp_path / "nvidia-smi"
    fake.write_text(f"#!/bin/sh\necho '{cap}'\n", encoding="utf-8")
    fake.chmod(0o755)
    done = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\nPATH={tmp_path}:$PATH\n{fn}\nnvidia_torch_tag"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == tag


def test_the_setup_uses_the_tag_for_index_and_swap():
    assert (
        'TORCH_TAG="${NVIDIA_TAG:-cu128}"; TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_TAG"'
        in SETUP
    )
    assert '"$TORCH_TAG" == "cu126" ]] && _want="cuda 12.6"' in SETUP
    assert 'GPU_TYPE="cpu"' in SETUP[SETUP.index('NVIDIA_TAG="$(nvidia_torch_tag)"') :][:400], (
        "a card older than any build is installed for the CPU"
    )


@pytest.mark.parametrize(
    "cap, archs, misses",
    [
        ("8.6", CU128, False),
        ("8.9", CU128, False),
        ("7.5", CU128, False),
        ("12.0", CU128, False),
        ("6.1", CU128, True),
        ("5.2", CU128, True),  # Pascal, Maxwell on the 12.8 build
        ("6.1", CU126, False),
        ("5.2", CU126, False),  # … and on the 12.6 build
        ("12.0", CU126, True),
        ("10.0", CU126, True),  # a new card on the old build
        ("", CU128, False),
        ("8.6", [], False),
        ("x.y", CU128, False),
    ],
)
def test_a_build_has_code_for_a_card_of_its_generations_only(cap, archs, misses):
    assert build_misses_card(cap, archs) is misses


def _machine(cap, archs, device="cuda"):
    return HardwareProfile(
        best_device=device,
        cuda_archs=archs,
        gpus=[
            GPUInfo(
                index=0,
                name="GeForce GTX 1080",
                vendor="nvidia",
                memory_total_mb=8100,
                compute_capability=cap,
            )
        ],
    )


def test_an_old_card_on_the_new_build_is_named_with_the_way_out():
    said = unsupported_card(_machine("6.1", CU128))
    assert "GTX 1080" in said and "6.1" in said and "older" in said
    assert "no kernel image" in said and "CUDA 12.6" in said


def test_a_new_card_on_an_old_build_is_named_too():
    said = unsupported_card(_machine("12.0", CU126))
    assert "newer" in said and "CUDA 12.6" not in said


@pytest.mark.parametrize(
    "profile",
    [
        _machine("8.6", CU128),
        _machine("6.1", CU126),
        _machine("6.1", CU128, device="cpu"),
        _machine("6.1", []),
        HardwareProfile(best_device="cuda", cuda_archs=CU128),
    ],
)
def test_nothing_is_said_when_the_card_fits_or_nothing_is_known(profile):
    assert unsupported_card(profile) == ""


def test_the_start_and_the_system_check_say_it(monkeypatch):
    import whisper_tm.extensions as ext
    from whisper_tm.core import diagnostics as dg

    monkeypatch.setattr(ext, "hw", _machine("6.1", CU128))
    checks = dg._machine_checks()
    bad = next(c for c in checks if c["title"] == "PyTorch and this card")
    assert bad["status"] == dg.FAIL and "CUDA 12.6" in bad["hint"]
    monkeypatch.setattr(ext, "hw", _machine("8.6", CU128))
    assert [c["title"] for c in dg._machine_checks()] == ["Defaults for this machine"]
    app = (ROOT / "whisper_tm" / "app.py").read_text(encoding="utf-8")
    assert "unsupported_card(ext.hw)" in app

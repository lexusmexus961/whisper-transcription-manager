"""Hints name the setup's and the start scripts' commands (core/commands.py)."""

from __future__ import annotations

import pytest

from whisper_tm.core import commands


@pytest.mark.parametrize(
    "option,expected",
    [
        ("", "./setup.sh"),
        ("--gpu nvidia", "./setup.sh --gpu nvidia"),
        ("--llama", "./setup.sh --llama"),
    ],
)
def test_the_setup(option, expected):
    assert commands.setup(option) == expected


def test_the_start_scripts():
    assert commands.start("--hf-token") == "./start.sh --hf-token"
    assert commands.start_llm() == "./start-llm.sh"


def test_every_setup_option_named_exists_in_setup_sh():
    """The options the hints name must be real options of setup.sh."""
    import re
    from pathlib import Path

    sh = (Path(__file__).resolve().parents[1] / "setup.sh").read_text(encoding="utf-8")
    for option in ("--gpu", "--llama", "--no-hf-token", "--dev"):
        assert re.search(rf"^\s+{option}\)", sh, re.M), option


# ── In the Docker image ───────────────────────────────────────────────────
#
# There are no setup or start scripts in the image, and no llama.cpp: a hint
# that says "./setup.sh --llama" sends a Docker user nowhere.


def test_in_docker_the_hints_name_compose(monkeypatch):
    monkeypatch.setenv("WHISPER_TM_DOCKER", "1")
    assert commands.in_docker()
    assert commands.setup("--gpu nvidia") == "docker compose up -d --build"
    assert commands.start() == "docker compose up -d"
    assert commands.start("--check") == "docker compose run --rm whisper-tm --check"
    assert "host.docker.internal" in commands.llama_missing()
    assert "setup.sh" not in commands.llama_missing()


def test_outside_docker_llama_comes_from_the_setup(monkeypatch):
    monkeypatch.delenv("WHISPER_TM_DOCKER", raising=False)
    assert not commands.in_docker()
    assert commands.llama_missing() == '"./setup.sh --llama" installs it'


def test_the_dockerfile_sets_the_flag():
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "WHISPER_TM_DOCKER=1" in dockerfile

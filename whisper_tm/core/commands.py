"""The setup's and the start scripts' commands, for hints.

Hints in the diagnostics, in error messages and in the setup guide name a
command to run — they take it from here, so they all say the same. In the
Docker image there are no such scripts: the container is rebuilt and run
with Compose, and llama.cpp is not part of it.
"""

from __future__ import annotations

import os
import sys

#: In Docker a language model (or embedding) server runs on the host and is
#: reached from the container at this address (docker-compose.yaml's
#: extra_hosts)
DOCKER_HOST_URL = "http://host.docker.internal:<port>"


def _windows() -> bool:
    return sys.platform == "win32"


def in_docker() -> bool:
    """In the project's Docker image? Its Dockerfile sets WHISPER_TM_DOCKER=1."""
    return os.environ.get("WHISPER_TM_DOCKER") == "1"


def setup(option: str = "") -> str:
    """``./setup.sh --gpu nvidia`` — in Docker, rebuilding the image."""
    if in_docker():
        return "docker compose up -d --build"
    return f"./setup.sh {option}".strip()


def start(option: str = "") -> str:
    """``./start.sh --check`` — in Docker, the server is a Compose service
    and a command runs in a second container of the image."""
    if in_docker():
        return f"docker compose run --rm whisper-tm {option}" if option else "docker compose up -d"
    return f"./start.sh {option}".strip()


def llama_missing() -> str:
    """How to get llama.cpp, as the end of "llama.cpp is not installed — …"."""
    if in_docker():
        return (
            "the Docker image has none: run its server on the host and enter "
            f"{DOCKER_HOST_URL} in the settings"
        )
    return f'"{setup("--llama")}" installs it'


def split_args(text: str) -> list[str]:
    """A settings value of extra arguments as a list, split like a shell.

    POSIX rules eat a path's backslashes (``C:\\models\\x.gguf`` became
    ``C:modelsx.gguf``): on Windows the words are split on spaces outside
    quotes, backslashes kept, a word's surrounding quotes taken off. The
    settings page's check of yt-dlp options uses this too — it must see the
    same words that are run. ValueError for an unclosed quote.
    """
    import shlex

    if not _windows():
        return shlex.split(text)
    out = []
    for word in shlex.split(text, posix=False):
        if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
            word = word[1:-1]
        out.append(word)
    return out


def start_llm() -> str:
    """The script that starts the local language model server."""
    return "./start-llm.sh"

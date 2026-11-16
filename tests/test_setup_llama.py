"""setup.sh: where llama-server comes from.

The "latest" release of ggml-org/llama.cpp had no binaries at all, so the
install found nothing; and an NVIDIA machine got the Vulkan build, which also
uses an AMD card in the same machine and ignores CUDA_VISIBLE_DEVICES — the
switch Whisper Transcription Manager uses to keep a CPU-only embedding server off the card.
"""

import subprocess
from pathlib import Path

SETUP = (Path(__file__).resolve().parent.parent / "setup.sh").read_text(encoding="utf-8")
PART = SETUP[SETUP.index("Pick the matching release package") : SETUP.index("# ── Model ──")]


def test_the_newest_release_with_the_package_not_latest():
    fetches = [line for line in PART.splitlines() if "curl" in line and "api.github.com" in line]
    assert fetches and all("releases?per_page=" in line for line in fetches)


def test_nvidia_gets_the_cuda_build_and_its_runtime():
    nvidia = PART[
        PART.index('elif [[ "$GPU_TYPE" == "nvidia" ]]') : PART.index(
            'elif [[ "$GPU_TYPE" == "amd" ]]'
        )
    ]
    assert "bin-ubuntu-cuda-" in nvidia and "vulkan" not in nvidia
    assert "cudart-llama-b" in PART


def test_the_url_choice_on_a_release_list():
    """The grep of the script, on a list whose newest entry has no files."""
    listing = (
        '[{"tag_name":"v0.5.0","assets":[]},'
        '{"assets":[{"browser_download_url":"https://x/b2/llama-b2-bin-ubuntu-cuda-13.4-x64.tar.gz"},'
        '{"browser_download_url":"https://x/b2/cudart-llama-b2-bin-ubuntu-cuda-13.4-x64.tar.gz"},'
        '{"browser_download_url":"https://x/b2/llama-b2-bin-ubuntu-vulkan-x64.tar.gz"}]},'
        '{"assets":[{"browser_download_url":"https://x/b1/llama-b1-bin-ubuntu-cuda-13.4-x64.tar.gz"}]}]'
    )
    line = next(x for x in PART.splitlines() if "${_pattern}" in x and "grep -oE" in x)
    grep = line[line.index("grep -oE") : line.index(" | head -1")]
    script = (
        '_releases="$1"; _pattern="bin-ubuntu-cuda-13\\.[0-9]+-x64\\.tar\\.gz"; '
        f'_url="$(printf \'%s\' "$_releases" | {grep} | head -1)"; '
        'echo "$_url"; echo "${_url/\\/llama-b/\\/cudart-llama-b}"'
    )
    out = subprocess.run(
        ["bash", "-c", script, "x", listing], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == [
        "https://x/b2/llama-b2-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "https://x/b2/cudart-llama-b2-bin-ubuntu-cuda-13.4-x64.tar.gz",
    ]

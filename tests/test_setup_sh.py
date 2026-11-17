"""
setup.sh — checked without running the installer.

* The llama step saved the config with ``config.save()``, which writes
  ~/.config/whisper-tm/config.yaml — from then on that file outranked the
  ./config.yaml the script tells the user to edit.
* Downloads went to fixed names in /tmp (symlink planting).
* start.sh / start-llm.sh hardcoded .venv and ignored ``--venv``.
* ``read`` at EOF (no terminal) ended the whole setup under ``set -e``.
* An option without its value died with "unbound variable".
* ``--llama-model`` lost to any GGUF already in models/gguf/ — the setup
  took that one as present, and the server the first file by name.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")


def _block(start: str, end: str) -> str:
    i = SETUP.index(start)
    return SETUP[i : SETUP.index(end, i)]


def test_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "setup.sh")], check=True)


def test_no_fixed_tmp_names():
    assert not re.search(r"/tmp/llamacpp", SETUP.replace("# names in /tmp (/tmp/llamacpp.tgz)", ""))
    assert "mktemp -d" in SETUP


def test_llama_step_saves_back_into_the_loaded_file():
    """That one line, into the file it was loaded from — its comments kept."""
    pycfg = _block("<< 'PYCFG'", "\nPYCFG")
    assert 'config.save_value("llm", "endpoint"' in pycfg and "config.save(" not in pycfg


def test_start_scripts_use_the_chosen_venv(tmp_path):
    block = _block("{\n    cat << 'STARTEOF'", "chmod +x start.sh")
    subprocess.run(["bash", "-c", f'VENV_DIR="/opt/my venv"\n{block}'], cwd=tmp_path, check=True)
    text = (tmp_path / "start.sh").read_text()
    assert "/opt/my\\ venv" in text and ".venv" not in text
    subprocess.run(["bash", "-n", str(tmp_path / "start.sh")], check=True)

    llm = _block("        {\n            cat << 'LLMEOF'", "chmod +x start-llm.sh")
    subprocess.run(["bash", "-c", f"VENV_DIR=.venv-x\n{llm}"], cwd=tmp_path, check=True)
    text = (tmp_path / "start-llm.sh").read_text()
    assert "VENV_DIR=.venv-x" in text and "--zeigen" not in text
    subprocess.run(["bash", "-n", str(tmp_path / "start-llm.sh")], check=True)


def test_read_at_eof_does_not_end_the_setup():
    assert 'answer || answer=""' in SETUP
    # Ctrl+D at any question of the setup: every read has its way out
    questions = [
        line
        for line in SETUP.splitlines()
        if re.match(r"\s*read -r", line) and "while read" not in line
    ]
    assert len(questions) >= 6 and all("||" in line for line in questions)


def test_the_model_asked_for_is_the_one_found(tmp_path):
    block = _block(
        '        if [[ -n "$LLAMA_MODEL" ]]; then\n            _present=',
        '        if [[ -n "$_present" ]]',
    )
    (tmp_path / "a-model-already-there.gguf").write_bytes(b"GGUF")

    def present(model: str) -> str:
        return subprocess.run(
            [
                "bash",
                "-c",
                f'LLAMA_MODEL_DIR="$1"; LLAMA_MODEL="$2"\n{block}\necho "$_present"',
                "x",
                str(tmp_path),
                model,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert present("") == str(tmp_path / "a-model-already-there.gguf")
    assert present("someone/Wanted-GGUF:wanted.gguf") == ""
    (tmp_path / "wanted.gguf").write_bytes(b"GGUF")
    assert present("someone/Wanted-GGUF:wanted.gguf") == str(tmp_path / "wanted.gguf")


def test_the_model_asked_for_goes_into_the_config():
    """Else the server takes the first GGUF by name — maybe not that one."""
    pycfg = _block("<< 'PYCFG'", "\nPYCFG")
    assert 'config.save_value("llama", "model", chosen)' in pycfg
    chosen = _block('        _chosen=""', "        WTM_LLAMA_MODEL=")
    out = subprocess.run(
        [
            "bash",
            "-c",
            'set -u; SCRIPT_DIR="/opt/my app"; LLAMA_MODEL="r:m.gguf"; '
            'LLAMA_MODEL_PATH="/opt/my app/models/gguf/m.gguf"\n' + chosen + 'echo "$_chosen"',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "models/gguf/m.gguf"
    out = subprocess.run(
        ["bash", "-c", 'set -u; SCRIPT_DIR=/a; LLAMA_MODEL=""\n' + chosen + 'echo "[$_chosen]"'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "[]"


def test_the_uv_installer_leaves_the_shell_profiles_alone():
    line = next(x for x in SETUP.splitlines() if "astral.sh/uv/install.sh" in x)
    assert "UV_NO_MODIFY_PATH=1 sh" in line


def test_an_option_without_value_is_reported():
    fn = _block("need_value() {", "\nwhile [[ $# -gt 0 ]]")
    r = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{fn}\nneed_value --gpu"], capture_output=True, text=True
    )
    assert r.returncode == 1 and "needs a value" in r.stderr
    r = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{fn}\nneed_value --gpu cpu"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0


def test_the_models_are_offered_before_the_first_job():
    """Downloaded while the machine is online and someone watches — not in
    the first job, or never should the machine be offline by then."""
    block = _block('step "Models"', "\n# ── llama.cpp")
    assert "-m whisper_tm --prefetch" in block and "if ! $PREFETCH" in block
    assert (
        "--no-prefetch) PREFETCH=false" in SETUP
        and "--no-prefetch" in SETUP[SETUP.index("--help|-h") :]
    )


@pytest.mark.parametrize("gpu", ["cpu", "nvidia"])
def test_the_setup_writes_no_values_for_the_machine(tmp_path, gpu):
    """large-v3, the vocal isolation and three lanes are hours per recording
    without a GPU. The setup used to write a CPU profile into config.yaml;
    the app now chooses for the machine at every start (hardware.machine_defaults)
    — in Docker and after a card was taken out too — so the file holds the
    library only, whatever the machine."""
    import yaml

    block = _block(
        'if [[ ! -f "config.yaml" ]]; then', '\nelse\n    ok "config.yaml already exists"'
    )
    script = (
        'set -euo pipefail\nok() { :; }; info() { echo "$1"; }\nSCRIPT_DIR=.; GPU_TYPE="$1"\n'
        'LIBRARY_DIR=""; ask_library() { :; }\n' + block + "\nfi\n"
    )
    done = subprocess.run(
        ["bash", "-c", script, "x", gpu], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert set(cfg) == {"directories"}
    assert ("Whisper turbo" in done.stdout) == (gpu == "cpu"), (
        "a CPU install is told what the app will start with"
    )


def test_the_library_root_becomes_the_media_folder_and_the_allowed_tree(tmp_path):
    """The library scales from ONE root: --library (or the question) writes it
    as directories.media and as the only allowed tree — quoted, so a folder
    name with a quote, an ampersand or a bar survives."""
    import yaml

    lib = tmp_path / "My Lib's & stuff|x"
    lib.mkdir()
    work = tmp_path / "app"
    work.mkdir()
    ask = _block("ask_library() {", '\nif [[ ! -f "config.yaml" ]]; then')
    block = _block(
        'if [[ ! -f "config.yaml" ]]; then', '\nelse\n    ok "config.yaml already exists"'
    )
    script = (
        'set -euo pipefail\nok() { :; }; info() { :; }; warn() { :; }; err() { echo "$*" >&2; }\n'
        'SCRIPT_DIR=.; GPU_TYPE=nvidia; LIBRARY_DIR="$1"\n' + ask + "\n" + block + "\nfi\n"
    )
    subprocess.run(
        ["bash", "-c", script, "x", str(lib)],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    cfg = yaml.safe_load((work / "config.yaml").read_text())
    real = str(lib.resolve())
    assert cfg["directories"]["media"] == real
    assert cfg["security"]["allowed_roots"] == [real]

    # Without a terminal and without --library nothing is asked or written
    (work / "config.yaml").unlink()
    subprocess.run(
        ["bash", "-c", script, "x", ""],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert set(yaml.safe_load((work / "config.yaml").read_text())) == {"directories"}

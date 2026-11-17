#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Whisper Transcription Manager — Setup Script (Linux)                   ║
# ║                                                                         ║
# ║  Builds a complete Python environment with:                             ║
# ║  - uv (fast package manager) or pip as fallback                         ║
# ║  - venv in the project directory (.venv/)                               ║
# ║  - GPU detection: NVIDIA CUDA / AMD ROCm / Apple MPS / CPU             ║
# ║  - All dependencies incl. WhisperX and audio-separator                 ║
# ║  - The Hugging Face token for speaker diarization (asked first)        ║
# ║                                                                         ║
# ║  Usage:                                                                 ║
# ║    ./setup.sh              # auto-detect                                ║
# ║    ./setup.sh --gpu nvidia  # force NVIDIA CUDA                         ║
# ║    ./setup.sh --gpu amd     # force AMD ROCm                            ║
# ║    ./setup.sh --gpu cpu     # CPU only                                  ║
# ║    ./setup.sh --no-uv       # use pip instead of uv                     ║
# ║    ./setup.sh --dev         # + developer tools (pytest, ruff)          ║
# ║    ./setup.sh --no-hf-token # without the Hugging Face token question   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }
step()  { echo -e "\n${CYAN}${BOLD}── $* ──${NC}"; }

# ── Parse arguments ───────────────────────────────────────────────────────
GPU_OVERRIDE=""
USE_UV=true
INSTALL_DEV=false
SETUP_LLAMA=""            # "" = ask, true/false = forced
LLAMA_MODEL=""           # HuggingFace repo:file, empty = default model
PYTHON_CMD=""
VENV_DIR=".venv"
ASK_HF_TOKEN=true         # false = --no-hf-token
PREFETCH=true             # false = --no-prefetch: the models download with the first job
CHECK_SPACE=true          # false = --skip-space-check
LIBRARY_DIR=""            # --library PATH: the library root, else asked

# An option that takes a value must get one — with `set -u` a missing value
# used to end the script with "unbound variable".
need_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Option $1 needs a value (see --help)" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       need_value "$@"; GPU_OVERRIDE="$2"; shift 2
                     case "$GPU_OVERRIDE" in
                         nvidia|amd|cpu|apple) ;;
                         *) echo "--gpu takes nvidia, amd, cpu or apple, not \"$GPU_OVERRIDE\"" >&2; exit 1 ;;
                     esac ;;
        --no-uv)     USE_UV=false; shift ;;
        --library)   need_value "$@"; LIBRARY_DIR="$2"; shift 2 ;;
        --dev)       INSTALL_DEV=true; shift ;;
        --llama)     SETUP_LLAMA=true; shift ;;
        --no-llama)  SETUP_LLAMA=false; shift ;;
        --llama-model) need_value "$@"; LLAMA_MODEL="$2"; shift 2 ;;
        --python)    need_value "$@"; PYTHON_CMD="$2"; shift 2 ;;
        --venv)      need_value "$@"; VENV_DIR="$2"; shift 2 ;;
        --no-hf-token) ASK_HF_TOKEN=false; shift ;;
        --no-prefetch) PREFETCH=false; shift ;;
        --skip-space-check) CHECK_SPACE=false; shift ;;
        --hf-token)
            # Never on the command line: other users' `ps` shows it, and the
            # shell history keeps it
            err "Paste the token when the setup asks for it (hidden), or later: ./start.sh --hf-token"
            exit 1 ;;
        --help|-h)
            echo "Usage: ./setup.sh [options]"
            echo ""
            echo "Options:"
            echo "  --gpu nvidia|amd|cpu   force GPU type (default: auto)"
            echo "  --no-uv               use pip instead of uv"
            echo "  --library PATH        the library root: the one folder all recordings live under"
            echo "  --dev                  install developer tools"
            echo "  --llama               install llama.cpp server (local LLM)"
            echo "  --no-llama            skip llama.cpp (no prompt)"
            echo "  --llama-model R:D     GGUF model (HF repo:file), else default"
            echo "  --python PATH         Python executable (default: python3)"
            echo "  --venv DIR            venv directory (default: .venv)"
            echo "  --no-hf-token         don't ask for the Hugging Face token"
            echo "  --no-prefetch         don't download the models now (the first job does)"
            echo "  --skip-space-check    install even if the free disk space looks too small"
            echo "  --help                this help"
            echo ""
            echo "Speaker diarization needs a Hugging Face token once. The setup asks"
            echo "for it first (hidden); later: ./start.sh --hf-token. A token in the"
            echo "environment (HF_TOKEN) is taken without the question — set it from a"
            echo "file, not typed into a command: the shell history keeps that."
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Project directory ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Downloads and unpacking go into a private folder of their own — fixed
# names in /tmp (/tmp/llamacpp.tgz) could be planted as a symlink by another
# user, and curl -o / tar wrote through it.
WORK_TMP="$(mktemp -d "${TMPDIR:-/tmp}/whisper-tm-setup.XXXXXXXX")"
trap 'rm -rf "$WORK_TMP"' EXIT

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Whisper Transcription Manager — Setup           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Find Python ───────────────────────────────────────────────────────────
# The app needs Python 3.11–3.13 (whisperx: <3.14); 3.12 is the one it
# is tested with. The database needs SQLite 3.38 or newer in that Python
# (unixepoch()) — Ubuntu 22.04's Python has 3.37. Without a fitting one, uv
# fetches its own 3.12 — so a distribution that ships only a newer (or an
# older) Python works too.
step "Checking Python"

# A Python 3.11–3.13 with SQLite 3.38+?
python_fits() {
    local ok
    ok="$("$1" -c 'import sqlite3, sys; print("y" if (3, 11) <= sys.version_info[:2] < (3, 14) and sqlite3.sqlite_version_info >= (3, 38) else "")' 2>/dev/null || true)"
    [[ -n "$ok" ]]
}

PYTHON=""
PYTHON_MANAGED=false      # true = uv downloads Python 3.12 for the venv
if [[ -n "$PYTHON_CMD" ]]; then
    if ! python_fits "$PYTHON_CMD"; then
        err "$PYTHON_CMD is not a Python 3.11–3.13 with SQLite 3.38+ (WhisperX needs one of those versions, the database that SQLite)"
        exit 1
    fi
    PYTHON="$PYTHON_CMD"
else
    for cmd in python3.12 python3.11 python3.13 python3; do
        if command -v "$cmd" &>/dev/null && python_fits "$cmd"; then
            PYTHON="$cmd"
            break
        fi
    done
fi
if [[ -n "$PYTHON" ]]; then
    ok "Found: $PYTHON ($("$PYTHON" --version 2>&1))"
elif $USE_UV; then
    PYTHON="3.12"
    PYTHON_MANAGED=true
    info "No Python 3.11–3.13 with SQLite 3.38+ here — uv will download Python 3.12 for the venv"
else
    err "No Python 3.11–3.13 with SQLite 3.38+ found (the app needs 3.11–3.13, the database SQLite 3.38)."
    err "Install Python 3.12 (https://www.python.org/downloads/), or run without"
    err "--no-uv: uv then downloads one by itself."
    exit 1
fi

# ── System info ───────────────────────────────────────────────────────────
step "Detecting system"

OS="$(uname -s)"
ARCH="$(uname -m)"
info "OS: $OS ($ARCH)"

# ── Build tools ───────────────────────────────────────────────────────────
# diffq, which audio-separator needs, has no ready-made package for Python
# 3.11 and newer: it is compiled during the install, which takes a C
# compiler and Python's header files. Checked here — not when they are
# found missing after gigabytes of downloads.
step "Checking build tools"

# $1: the Python's minor version, $2: "headers" when Python.h is missing too
build_tools_hint() {
    local distro="" dev=""
    [[ -r /etc/os-release ]] && distro="$(. /etc/os-release; echo "${ID:-} ${ID_LIKE:-}")"
    case "$OS $distro" in
        Darwin*)                  echo "    xcode-select --install" ;;
        *debian*|*ubuntu*)        [[ "${2:-}" == headers ]] && dev=" python3.$1-dev"
                                  echo "    sudo apt install build-essential$dev" ;;
        *fedora*|*rhel*|*centos*) [[ "${2:-}" == headers ]] && dev=" python3-devel"
                                  echo "    sudo dnf install gcc$dev" ;;
        *arch*)                   echo "    sudo pacman -S base-devel" ;;
        *suse*)                   [[ "${2:-}" == headers ]] && dev=" python3-devel"
                                  echo "    sudo zypper install gcc$dev" ;;
        *)                        echo "    gcc and your distribution's Python development package" ;;
    esac
}

if $PYTHON_MANAGED; then
    PY_MINOR=12    # uv's own Python brings its header files
else
    PY_MINOR="$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')"
fi
if (( PY_MINOR >= 11 )); then
    BUILD_TOOLS_OK=true
    MISSING_HEADERS=""
    # macOS has a /usr/bin/cc even without the Command Line Tools — a stub
    if { ! command -v cc && ! command -v gcc; } &>/dev/null \
            || { [[ "$OS" == "Darwin" ]] && ! xcode-select -p &>/dev/null; }; then
        err "No C compiler found — diffq (for audio-separator) is compiled during the install."
        BUILD_TOOLS_OK=false
    fi
    if ! $PYTHON_MANAGED && ! "$PYTHON" -c 'import os, sys, sysconfig
sys.exit(not os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' 2>/dev/null; then
        err "Python's header files (Python.h) are missing — diffq (for audio-separator) is compiled during the install."
        BUILD_TOOLS_OK=false
        MISSING_HEADERS=headers
    fi
    if ! $BUILD_TOOLS_OK; then
        echo "  Install them, then run the setup again:"
        build_tools_hint "$PY_MINOR" "$MISSING_HEADERS"
        exit 1
    fi
    ok "C compiler and Python header files found"
else
    ok "Python 3.$PY_MINOR — nothing needs compiling"
fi

# ── Detect GPU ────────────────────────────────────────────────────────────
step "Detecting GPU"

detect_gpu() {
    if [[ -n "$GPU_OVERRIDE" ]]; then
        echo "$GPU_OVERRIDE"
        return
    fi

    # NVIDIA
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null; then
            echo "nvidia"
            return
        fi
    fi

    # AMD ROCm
    if command -v rocm-smi &>/dev/null; then
        echo "amd"
        return
    fi
    if [[ -d "/opt/rocm" ]]; then
        echo "amd"
        return
    fi

    # Apple MPS
    if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
        echo "apple"
        return
    fi

    echo "cpu"
}

# PyTorch's build for an NVIDIA card, by the compute capability of the first
# card: "cu128", or "cu126" for a card older than 7.0 (GTX 10xx, GTX 9xx —
# the CUDA 12.8 build has no code for them: alignment, diarization and
# separation would stop with "no kernel image"), or nothing for a card older
# than any build covers (below 5.0). Without an answer from nvidia-smi: cu128.
nvidia_torch_tag() {
    local cap major
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')" || true
    major="${cap%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]]; then
        if (( major < 5 )); then return 0; fi
        if (( major < 7 )); then echo "cu126"; return 0; fi
    fi
    echo "cu128"
}

GPU_TYPE=$(detect_gpu)
NVIDIA_TAG=""

case "$GPU_TYPE" in
    nvidia)
        ok "NVIDIA GPU detected"
        if command -v nvidia-smi &>/dev/null; then
            # || true: a failing nvidia-smi (driver/library mismatch after a
            # driver update) must not end the setup without a word (pipefail)
            { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 | while read -r line; do
                info "  $line"
            done; } || true
        fi
        NVIDIA_TAG="$(nvidia_torch_tag)"
        if [[ -z "$NVIDIA_TAG" ]]; then
            warn "This card is older than any PyTorch build has code for (compute capability below 5.0) — installing for the CPU"
            GPU_TYPE="cpu"
        elif [[ "$NVIDIA_TAG" == "cu126" ]]; then
            info "  A card from before GTX 16xx / RTX 20xx: PyTorch's CUDA 12.6 build, the last one with code for it"
        fi
        ;;
    amd)    ok "AMD GPU detected (ROCm)" ;;
    apple)  ok "Apple Silicon detected (MPS)" ;;
    cpu)    if [[ -n "$GPU_OVERRIDE" ]]; then info "CPU mode (--gpu cpu)"; else warn "No GPU detected — CPU mode"; fi ;;
esac

# ── Free disk space ───────────────────────────────────────────────────────
# Checked before the long installs: a disk that fills up halfway leaves raw
# uv/pip/curl errors and no start.sh. What it takes, roughly: the
# environment (the PyTorch build is most of it) plus the installer's cache
# while it unpacks (on the venv's disk), and the models a first job needs
# (the user's cache). The language model's file comes on top if chosen.
step "Free disk space"
free_gb() {   # free GB on the file system of $1 (or its nearest existing parent)
    local p="$1"
    while [[ ! -e "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
    df -Pk "$p" 2>/dev/null | awk 'NR==2 {printf "%d", $4 / 1048576}'
}
fs_of() {
    local p="$1"
    while [[ ! -e "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
    df -Pk "$p" 2>/dev/null | awk 'NR==2 {print $1}'
}
case "$GPU_TYPE" in
    nvidia) NEED_ENV=14 ;;   # torch + CUDA libraries, the cache while unpacking
    amd)    NEED_ENV=20 ;;   # the ROCm build is larger still
    *)      NEED_ENV=5 ;;
esac
NEED_MODELS=0
$PREFETCH && NEED_MODELS=4
MODELS_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ "$(fs_of "$SCRIPT_DIR")" == "$(fs_of "$MODELS_DIR")" ]]; then
    NEEDS=("$SCRIPT_DIR:$((NEED_ENV + NEED_MODELS))")
else
    NEEDS=("$SCRIPT_DIR:$NEED_ENV" "$MODELS_DIR:$NEED_MODELS")
fi
SHORT=false
for entry in "${NEEDS[@]}"; do
    where="${entry%:*}"; need="${entry##*:}"
    [[ "$need" -le 0 ]] && continue
    have="$(free_gb "$where")"
    if [[ -z "$have" ]]; then
        info "Free space on $where unknown — not checked"
    elif [[ "$have" -lt "$need" ]]; then
        err "About $need GB are needed on the disk of $where, $have GB are free — $((need - have)) GB missing"
        SHORT=true
    else
        ok "$have GB free on the disk of $where (about $need GB needed)"
    fi
done
if $SHORT && $CHECK_SPACE; then
    err "Free some space, or put the environment elsewhere (--venv DIR) and the models"
    err "elsewhere (HF_HOME=…); --skip-space-check installs anyway."
    exit 1
fi

# ── Hugging Face token (speaker diarization) ──────────────────────────────
# Asked now, before the long installs: a wrong token or conditions not yet
# accepted show up while there is time to fix them. Saved at the end, when
# the app is installed (whisper-tm --hf-token checks it again, writes it
# into the config and downloads the model).
HF_REPO="pyannote/speaker-diarization-community-1"
HF_TOKEN_VALUE=""
HF_REJECTED=false         # a token was given and Hugging Face refused it

# The config file the app will load: the first of these that exists
# (whisper_tm/config.py::_find_config — the user config folder is
# platformdirs' user_config_dir)
app_config_file() {
    local home_cfg
    if [[ "$OS" == "Darwin" ]]; then
        home_cfg="$HOME/Library/Application Support/whisper-tm/config.yaml"
    else
        home_cfg="${XDG_CONFIG_HOME:-$HOME/.config}/whisper-tm/config.yaml"
    fi
    local f
    for f in "${WHISPER_TM_CONFIG:-}" "$home_cfg" "$SCRIPT_DIR/config.yaml"; do
        if [[ -n "$f" && -f "$f" ]]; then
            echo "$f"
            return
        fi
    done
}

hf_token_configured() {
    local f
    f="$(app_config_file)"
    [[ -n "$f" ]] && grep -qE '^[[:space:]]+hf_token:[[:space:]]*["'"'"']?hf_' "$f"
}

# ok | gated | invalid | offline | "error <code>" — the token goes to curl on
# stdin (-K -), never on its command line
hf_status() {
    command -v curl &>/dev/null || { echo "unchecked"; return; }
    local code
    code="$(printf 'header = "Authorization: Bearer %s"\n' "$1" \
        | curl -s -o /dev/null -w '%{http_code}' --max-time 20 -K - \
            "https://huggingface.co/api/models/$HF_REPO/auth-check" 2>/dev/null || true)"
    case "$code" in
        200) echo ok ;;
        403) echo gated ;;
        401) echo invalid ;;
        000|"") echo offline ;;
        *) echo "error $code" ;;
    esac
}

step "Hugging Face token (speaker diarization)"

if ! $ASK_HF_TOKEN; then
    info "Not asked (--no-hf-token) — later: ./start.sh --hf-token"
elif [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_VALUE="$HF_TOKEN"
    info "Using the token from HF_TOKEN"
elif hf_token_configured; then
    ok "A token is configured already — kept (change it: ./start.sh --hf-token)"
elif [[ -t 0 ]]; then
    echo ""
    echo "  Telling speakers apart (diarization) uses pyannote's free model"
    echo "  $HF_REPO. Hugging Face hands it out only to"
    echo "  signed-in accounts that accepted its conditions — once; after the first"
    echo "  download (~32 MB) it loads from the cache without a token."
    echo ""
    echo "    1. Sign in at https://huggingface.co (or create a free account:"
    echo "       https://huggingface.co/join)."
    echo "    2. Open https://huggingface.co/$HF_REPO"
    echo "       and accept the conditions — the form asks for a company or"
    echo "       university and a use case; access is granted at once."
    echo "    3. Create an access token of type \"Read\":"
    echo "       https://huggingface.co/settings/tokens"
    echo ""
    echo "  Without it everything else works — transcripts just have no speakers."
    echo ""
    read -rs -p "  Paste the token (hidden), or press Enter to skip: " HF_TOKEN_VALUE || HF_TOKEN_VALUE=""
    echo ""
else
    info "No terminal to ask in — skipped (later: ./start.sh --hf-token)"
fi
HF_TOKEN_VALUE="${HF_TOKEN_VALUE//[[:space:]]/}"

while [[ -n "$HF_TOKEN_VALUE" ]]; do
    case "$(hf_status "$HF_TOKEN_VALUE")" in
        ok)
            ok "The token works and may download the model"
            break ;;
        gated)
            warn "The token works, but its account has not accepted the model's conditions:"
            warn "  https://huggingface.co/$HF_REPO"
            warn "  (a fine-grained token also needs \"Read access to contents of all"
            warn "   public gated repos you can access\")"
            [[ -t 0 ]] || break
            read -r -p "  Accept them, then press Enter to check again (s = go on without): " _again || _again=s
            [[ "$_again" =~ ^[sS] ]] && break ;;
        invalid)
            warn "Hugging Face rejects this token."
            if [[ ! -t 0 ]]; then
                HF_TOKEN_VALUE=""
                HF_REJECTED=true
                break
            fi
            read -rs -p "  Paste it again (hidden), or press Enter to skip: " HF_TOKEN_VALUE || HF_TOKEN_VALUE=""
            echo ""
            HF_TOKEN_VALUE="${HF_TOKEN_VALUE//[[:space:]]/}"
            [[ -n "$HF_TOKEN_VALUE" ]] || HF_REJECTED=true ;;
        unchecked)
            info "curl is missing — the token is checked at the end"
            break ;;
        offline)
            warn "Hugging Face is not reachable now — the token is checked again at the end"
            break ;;
        *)
            warn "Unexpected answer from Hugging Face — the token is checked again at the end"
            break ;;
    esac
done

# ── Install uv ────────────────────────────────────────────────────────────
step "Package manager"

INSTALLER=""

if $USE_UV; then
    # Where its installer puts it — an earlier run's uv, not on PATH
    # because the installer leaves the shell profiles alone (below)
    export PATH="$PATH:$HOME/.local/bin:$HOME/.cargo/bin"
    if command -v uv &>/dev/null; then
        UV_VERSION=$(uv --version 2>&1)
        ok "uv already installed: $UV_VERSION"
        INSTALLER="uv"
    else
        info "Installing uv..."
        # UV_NO_MODIFY_PATH: the installer would add itself to the shell's
        # profile files — the setup finds it by itself, start.sh needs none
        if curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh 2>/dev/null; then
            if command -v uv &>/dev/null; then
                ok "uv installed: $(uv --version) ($(command -v uv); your shell profiles are unchanged)"
                INSTALLER="uv"
            else
                warn "uv installation failed, using pip"
                INSTALLER="pip"
            fi
        else
            warn "uv installation failed, using pip"
            INSTALLER="pip"
        fi
    fi
else
    info "Using pip (--no-uv)"
    INSTALLER="pip"
fi
if $PYTHON_MANAGED && [[ "$INSTALLER" != "uv" ]]; then
    err "No Python 3.11–3.13 here and uv could not be installed to fetch one."
    err "Install Python 3.12: https://www.python.org/downloads/"
    exit 1
fi

# ── Create venv ───────────────────────────────────────────────────────────
step "Virtual environment: $VENV_DIR"

if [[ -e "$VENV_DIR" && ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    # Never offer to delete what isn't a venv (a mistyped --venv)
    err "$VENV_DIR exists but is not a Python venv — choose another --venv,"
    err "or move it out of the way yourself."
    exit 1
fi
if [[ -d "$VENV_DIR" ]]; then
    warn "venv already exists: $VENV_DIR"
    # Without a terminal (stdin closed) read fails at EOF — with set -e that
    # ended the whole setup; no answer means "no"
    read -rp "  Delete and recreate? [y/N] " answer || answer=""
    if [[ "$answer" =~ ^[yY]$ ]]; then
        rm -rf "$VENV_DIR"
        info "Deleted."
    elif ! python_fits "$VENV_DIR/bin/python"; then
        err "The existing venv's Python is not a 3.11–3.13 with SQLite 3.38+ — WhisperX"
        err "can't be installed into it. Run the setup again and answer y to recreate it."
        exit 1
    else
        info "Reusing existing venv."
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    if [[ "$INSTALLER" == "uv" ]]; then
        uv venv "$VENV_DIR" --python "$PYTHON"
    else
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    ok "venv created: $VENV_DIR"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
ok "venv activated: $(python --version)"

# ── pip/uv wrapper ────────────────────────────────────────────────────────

# Every install goes through constraints.txt: the upper bounds the app is
# tested within (whisperx, pyannote, transformers, audio-separator)
CONSTRAINTS="$SCRIPT_DIR/constraints.txt"

pkg_install() {
    if [[ "$INSTALLER" == "uv" ]]; then
        uv pip install -c "$CONSTRAINTS" "$@"
    else
        pip install -c "$CONSTRAINTS" "$@"
    fi
}

# ── PyTorch's build for this machine ──────────────────────────────────────
# whisperx pins torch (torch~=2.8.0 for whisperx 3.8): the version comes
# from that pin, the PyTorch index only decides the build — CUDA 12.8 for
# NVIDIA (12.6 for a card older than GTX 16xx / RTX 20xx: nvidia_torch_tag),
# ROCm for AMD, the small CPU build without a GPU. Installing
# "torch" straight from an index would take its NEWEST build and break the
# pin. NVIDIA installs from PyPI, whose Linux torch is the CUDA build
# already (and whose torchcodec loads without NVIDIA's NPP library); a torch
# of another build is swapped below.
case "$GPU_TYPE" in
    nvidia) TORCH_TAG="${NVIDIA_TAG:-cu128}"; TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_TAG" ;;
    amd)    TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"; TORCH_TAG="rocm6.4" ;;
    cpu)    TORCH_INDEX="https://download.pytorch.org/whl/cpu";    TORCH_TAG="cpu" ;;
    *)      TORCH_INDEX="";                                        TORCH_TAG="" ;;
esac

# pkg_install with the PyTorch index next to PyPI — the best version of each
# package from either (uv would otherwise take everything the PyTorch index
# has from there alone)
pkg_install_torch_index() {
    if [[ -z "$TORCH_INDEX" ]]; then
        pkg_install "$@"
    elif [[ "$INSTALLER" == "uv" ]]; then
        uv pip install -c "$CONSTRAINTS" --index-strategy unsafe-best-match --extra-index-url "$TORCH_INDEX" "$@"
    else
        pip install -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX" "$@"
    fi
}

# The installed torch's build: "cuda 12.8", "hip 6.4", "cpu" or "none"
torch_build() {
    "$VENV_DIR/bin/python" -c "
import torch
v = torch.version
print('cuda ' + v.cuda if v.cuda else ('hip ' + v.hip if getattr(v, 'hip', None) else 'cpu'))
" 2>/dev/null || echo none
}

# torch, torchaudio and torchvision at their installed versions, as the
# $TORCH_TAG build
swap_torch() {
    local pins
    pins="$("$VENV_DIR/bin/python" - "$TORCH_TAG" << 'PYPINS'
import importlib.metadata as m
import sys
out = []
for name in ("torch", "torchaudio", "torchvision"):
    try:
        out.append(f"{name}=={m.version(name).split('+')[0]}+{sys.argv[1]}")
    except m.PackageNotFoundError:
        pass
print(" ".join(out))
PYPINS
)"
    [[ -n "$pins" ]] || return 1
    # shellcheck disable=SC2086 # one word per pin
    pkg_install_torch_index $pins
}

# ── Install WhisperX + audio-separator ────────────────────────────────────
step "Installing WhisperX"

if [[ "$GPU_TYPE" == "nvidia" ]]; then
    pkg_install whisperx
else
    pkg_install_torch_index whisperx
fi
ok "WhisperX installed"

step "Installing audio-separator"

# [cpu] on every machine: the Roformer models run on PyTorch (on the GPU);
# onnxruntime-gpu is built for CUDA 13 and fails to import without CUDA 13's
# libraries — pyannote imports it (through torchmetrics), so every
# transcription would fail
case "$GPU_TYPE" in
    nvidia) pkg_install "audio-separator[cpu]" ;;
    *)      pkg_install_torch_index "audio-separator[cpu]" ;;
esac
ok "audio-separator installed"

step "PyTorch build ($GPU_TYPE)"

TORCH_BUILD="$(torch_build)"
case "$GPU_TYPE" in
    nvidia|amd|cpu)
        _want="cuda"; [[ "$GPU_TYPE" == "amd" ]] && _want="hip"; [[ "$GPU_TYPE" == "cpu" ]] && _want="cpu"
        # An older card needs exactly the 12.6 build; what PyPI brings is 12.8
        [[ "$GPU_TYPE" == "nvidia" && "$TORCH_TAG" == "cu126" ]] && _want="cuda 12.6"
        if [[ "$TORCH_BUILD" != "$_want"* ]]; then
            info "PyTorch is the ${TORCH_BUILD} build — installing the $TORCH_TAG build of the same version…"
            swap_torch || warn "Swapping the PyTorch build failed"
            TORCH_BUILD="$(torch_build)"
        fi
        if [[ "$TORCH_BUILD" == "$_want"* ]]; then
            ok "PyTorch: $TORCH_BUILD"
        elif [[ "$GPU_TYPE" == "amd" ]]; then
            warn "PyTorch has no ROCm build ($TORCH_BUILD) — AMD is experimental and untested;"
            warn "the setup guide's AMD page in the app says what is known."
        else
            warn "PyTorch is the $TORCH_BUILD build, not $_want — transcription may run on the CPU."
            if [[ "$INSTALLER" == "uv" ]]; then
                # an environment uv made has no pip
                warn "By hand: uv pip install --python $VENV_DIR/bin/python --index-strategy unsafe-best-match --extra-index-url $TORCH_INDEX torch==<version>+$TORCH_TAG"
            else
                warn "By hand: $VENV_DIR/bin/python -m pip install --extra-index-url $TORCH_INDEX torch==<version>+$TORCH_TAG"
            fi
        fi
        ;;
    apple)
        ok "PyTorch: the default build, with MPS"
        ;;
esac

# ── Check FFmpeg ──────────────────────────────────────────────────────────
step "Checking FFmpeg"

if command -v ffmpeg &>/dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    ok "FFmpeg: $FFMPEG_VERSION"
else
    warn "FFmpeg not found!"
    echo "  Please install:"
    case "$OS" in
        Linux)  echo "    sudo apt install ffmpeg    # Debian/Ubuntu"
                echo "    sudo dnf install ffmpeg    # Fedora/RHEL"
                echo "    sudo pacman -S ffmpeg       # Arch" ;;
        Darwin) echo "    brew install ffmpeg" ;;
    esac
fi

# ── Project dependencies ──────────────────────────────────────────────────
# With what Ask (the chat) and the MCP server need (fastmcp, the [mcp]
# extra) and link downloads and the browser extension (yt-dlp). fastmcp and
# audio-separator disagree about beartype — fastmcp's token stores, unused
# here, want >=0.20, audio-separator <0.19 — and the app runs and is tested
# with 0.18.5: requirements-lock-overrides.txt settles it.
step "Installing Whisper Transcription Manager"

if [[ "$INSTALLER" == "uv" ]]; then
    uv pip install -c "$CONSTRAINTS" --override requirements-lock-overrides.txt -e ".[mcp]" yt-dlp
else
    pip install -c "$CONSTRAINTS" -e ".[mcp]" yt-dlp
    pip install -c "$CONSTRAINTS" -r requirements-lock-overrides.txt
fi
ok "whisper-tm installed"

# ── Developer tools ────────────────────────────────────────────────────────
if $INSTALL_DEV; then
    step "Developer tools"
    # ruff pinned like the CI's (.github/workflows/tests.yml): a new release
    # brings new rules
    pkg_install pytest pytest-cov ruff==0.15.21 mypy
    ok "Dev tools installed"
fi

# ── Configuration ──────────────────────────────────────────────────────────
step "Configuration"

# The library root: ONE folder — local or a mounted share — that every
# recording lives under, at any depth. The file list, queue, statistics and
# search scale from it, and it becomes the allowed tree
# (core/security.allowed_roots); a folder outside it is not part of the
# library. Asked once, for a new config.yaml.
ask_library() {
    [[ -n "$LIBRARY_DIR" ]] && return 0
    [[ -t 0 ]] || return 0
    echo ""
    echo "  Your library folder"
    echo "  Everything the app manages lives under ONE folder, the library root:"
    echo "  a local folder or a mounted NAS share. Subfolders of any depth are fine;"
    echo "  the file list, the queue, the statistics and the search all scale from"
    echo "  this root. Folders outside it are not part of the library, so choose the"
    echo "  top folder that holds ALL your recordings (e.g. /mnt/nas/media), not one"
    echo "  of its subfolders. Moving it later works (./start.sh --remap-path)."
    echo ""
    local _try
    for _try in 1 2 3; do
        read -r -p "  Library root (Enter = ~/Videos, choose later in the app): " LIBRARY_DIR || { echo; LIBRARY_DIR=""; }
        [[ -z "$LIBRARY_DIR" ]] && return 0
        LIBRARY_DIR="${LIBRARY_DIR/#\~/$HOME}"
        [[ -d "$LIBRARY_DIR" ]] && return 0
        warn "Not a folder here: $LIBRARY_DIR — is the share mounted?"
        LIBRARY_DIR=""
    done
}

if [[ ! -f "config.yaml" ]]; then
    ask_library
    if [[ -n "$LIBRARY_DIR" ]]; then
        LIBRARY_DIR="${LIBRARY_DIR/#\~/$HOME}"
        if [[ ! -d "$LIBRARY_DIR" ]]; then
            err "--library: not a folder here: $LIBRARY_DIR"
            exit 1
        fi
        LIBRARY_DIR="$(cd "$LIBRARY_DIR" && pwd -P)"
    fi
    # YAML single quotes: a quote inside is doubled
    _lib_yaml="'${LIBRARY_DIR//\'/\'\'}'"
    # Only what differs from config.default.yaml (which documents every
    # setting) belongs in it: a full copy froze every default at this
    # version's value. It will hold the Hugging Face token and API keys —
    # the owner's only, like every file the app saves (config.py::_write_private)
    (umask 077 && cat > config.yaml <<'CFG'
# Whisper Transcription Manager — your settings: only what differs from config.default.yaml,
# which documents every setting. The settings page saves into this file.

directories:
  # Your library root — every recording lives under it, at any depth.
  # Empty: $VIDEO_DIR or ~/Videos
  media: ""
CFG
)
    if [[ -n "$LIBRARY_DIR" ]]; then
        # The root is also the allowed tree: the library scales from it
        # The heredoc's last line is the empty media entry: replace it
        sed -i '/^  media: ""$/d' config.yaml
        printf '  media: %s\nsecurity:\n  allowed_roots:\n  - %s\n' "$_lib_yaml" "$_lib_yaml" >> config.yaml
    fi
    if [[ "$GPU_TYPE" == "cpu" ]]; then
        # Nothing is written for it: the app chooses what a CPU manages
        # by itself at every start (docs/GUIDE.md: Hardware Support), also in
        # Docker and on a machine whose card was taken out later
        info "No GPU: the app starts with Whisper turbo, no vocal isolation and one file at a time"
    fi
    chmod 600 config.yaml
    ok "config.yaml created (your settings; config.default.yaml documents them all)"
    if [[ -n "$LIBRARY_DIR" ]]; then
        ok "Library root: $LIBRARY_DIR"
    else
        info "Your library root: directories.media in $SCRIPT_DIR/config.yaml (or choose it in the app)"
    fi
else
    ok "config.yaml already exists"
fi
# The app reads the user config folder's config.yaml FIRST when it exists
# (whisper_tm/config.py::_find_config) — then that is the file to edit
APP_CONFIG="$(app_config_file)"
if [[ -n "$APP_CONFIG" && "$APP_CONFIG" != "$SCRIPT_DIR/config.yaml" ]]; then
    warn "$APP_CONFIG exists and takes priority over $SCRIPT_DIR/config.yaml — edit that one"
fi

# ── Speaker diarization: the token saved, the model downloaded ────────────
# whisper-tm --hf-token checks the token again, writes it into the config
# the app loads (its comments kept) and downloads the model (~32 MB). The
# token goes over as HF_TOKEN for this one command — never as an argument.
step "Speaker diarization"

HF_READY=false
if [[ -n "$HF_TOKEN_VALUE" ]] || hf_token_configured; then
    if HF_TOKEN="$HF_TOKEN_VALUE" "$VENV_DIR/bin/python" -m whisper_tm --hf-token --no-input; then
        HF_READY=true
    fi
elif "$VENV_DIR/bin/python" -c "from whisper_tm.core import hf_access as h; raise SystemExit(0 if h.model_cached() else 1)" 2>/dev/null; then
    ok "The diarization model is downloaded already"
    HF_READY=true
elif $HF_REJECTED; then
    warn "The Hugging Face token was rejected and not saved — set a valid one later:"
    warn "  ./start.sh --hf-token   (or the setup guide in the app)"
else
    info "No Hugging Face token — transcripts get no speakers until one is set:"
    info "  ./start.sh --hf-token   (or the setup guide in the app)"
fi

# ── Verification ───────────────────────────────────────────────────────────
step "Verifying installation"

"$VENV_DIR/bin/python" -c "
import sys
print(f'Python:          {sys.version.split()[0]}')

import torch
cuda = torch.cuda.is_available()
mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
device = 'cuda' if cuda else ('mps' if mps else 'cpu')
vram = ''
if cuda:
    props = torch.cuda.get_device_properties(0)
    vram = f' ({props.name}, {props.total_memory // 1024**2}MB)'
print(f'PyTorch:         {torch.__version__} [{device}]{vram}')

# The modules a transcription really loads: whisperx itself only defines
# lazy wrappers, so a broken faster-whisper or pyannote passed a bare import
import importlib
for label, module in (('Whisper (ASR)', 'whisperx.asr'),
                      ('Diarization', 'whisperx.diarize'),
                      ('Vocal isolation', 'audio_separator.separator'),
                      ('MCP server', 'fastmcp')):
    try:
        importlib.import_module(module)
        print(f'{label + \":\":<17}OK')
    except Exception as e:
        print(f'{label + \":\":<17}MISSING — {module}: {str(e)[:120]}')

try:
    from whisper_tm import __version__
    print(f'Whisper Transcription Manager:    v{__version__}')
except Exception as e: print(f'Whisper Transcription Manager:    ERROR ({e})')

try:
    from whisper_tm.hardware import detect_hardware
    hw = detect_hardware()
    print(f'Hardware:        {hw.best_device} ({len(hw.gpus)} GPU(s))')
except Exception as e: print(f'Hardware:        ERROR ({e})')

" || warn "The check above failed — ./start.sh --check shows what is missing."

# ── Models ─────────────────────────────────────────────────────────────────
# Whisper, the alignment model, NLTK's sentence splitter and the separation
# model download on first use — now, while the machine is online and someone
# is watching, instead of in the first job (or never, should it be offline
# by then). whisper-tm --prefetch says each one, or why it failed.
step "Models"
if ! $PREFETCH; then
    info "Not downloaded now (--no-prefetch) — the first job does; or later: ./start.sh --prefetch"
else
    _size="$("$VENV_DIR/bin/python" -c "from whisper_tm.config import config; config.load(None); from whisper_tm.core import prefetch; print(f'{prefetch.total_mb() / 1000:.1f} GB')" 2>/dev/null || true)"
    _fetch=true
    if [[ -t 0 ]]; then
        read -r -p "  Download the models the settings need now${_size:+ (~$_size)}? [Y/n] " _a || { echo; _a=n; }
        [[ "$_a" =~ ^[nN] ]] && _fetch=false
    fi
    if $_fetch; then
        "$VENV_DIR/bin/python" -m whisper_tm --prefetch \
            || warn "Not every model could be downloaded — the first job tries again; or later: ./start.sh --prefetch"
    else
        info "Skipped — the first job downloads them; or later: ./start.sh --prefetch"
    fi
fi


# ── llama.cpp server (local LLM for summarization/correction/translation) ──
# Optional: Whisper Transcription Manager needs some OpenAI-compatible endpoint for that.
# llama.cpp is the simplest local option — one binary,
# one GGUF model, done.
step "llama.cpp server (optional)"

LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/llama-server"
LLAMA_MODEL_DIR="$SCRIPT_DIR/models/gguf"
# Default model: Qwen3.5-9B at Q4_K_M — 5.7 GB, with its 32k context ~7 GB of
# VRAM (only every fourth layer keeps a classic KV cache): fits an 8 GB card.
# The section `llama` of config.default.yaml is tuned for it.
LLAMA_STD_REPO="unsloth/Qwen3.5-9B-GGUF"
LLAMA_DEFAULT_FILE="Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_DEFAULT_SIZE="5.7 GB"

if [[ -z "$SETUP_LLAMA" ]]; then
    if [[ -x "$LLAMA_BIN" ]]; then
        ok "llama.cpp already present ($LLAMA_BIN)"
        SETUP_LLAMA=false
    elif [[ -t 0 ]]; then
        echo ""
        echo "  For summarization, correction and translation, Whisper Transcription Manager"
        echo "  needs an LLM. Locally that works with llama.cpp"
        echo "  (alternative: OpenAI/Ollama later in the settings)."
        # Ctrl+D (end of input) fails read — with set -e the whole setup
        read -r -p "  Install the llama.cpp server now? [y/N] " _answer || { echo; _answer=""; }
        [[ "$_answer" =~ ^[yY] ]] && SETUP_LLAMA=true || SETUP_LLAMA=false
    else
        info "Non-interactive without --llama → skipped"
        SETUP_LLAMA=false
    fi
fi

if $SETUP_LLAMA; then
    if [[ -x "$LLAMA_BIN" ]]; then
        ok "llama-server already installed"
    else
        # Pick the matching release package.
        #
        # The "latest" release of ggml-org/llama.cpp can be a tag without any
        # binaries (some tagged releases ship none) — the builds are the
        # nightly bNNNNN releases. So the newest of the last 30 releases that
        # carries the wanted package is taken, not ".../releases/latest".
        #
        # NVIDIA gets the CUDA build again (ggml-org ships Linux CUDA builds
        # once more, plus a cudart package with the runtime): it sees only
        # the NVIDIA cards. The Vulkan build would also see an AMD or Intel
        # card in the same machine and split the model across them, and
        # Whisper Transcription Manager hides the card from a CPU-only start through
        # CUDA_VISIBLE_DEVICES, which Vulkan ignores. Vulkan stays the path
        # for AMD, and the fallback when no CUDA package is found.
        _os="$(uname -s)"
        _arch="$(uname -m)"
        _pattern=""
        _cudart=""
        case "$_os" in
            Linux)
                if [[ "$_arch" == "aarch64" || "$_arch" == "arm64" ]]; then
                    [[ "$GPU_TYPE" == "cpu" ]] && _pattern="bin-ubuntu-arm64.tar.gz" \
                                               || _pattern="bin-ubuntu-vulkan-arm64.tar.gz"
                elif [[ "$GPU_TYPE" == "nvidia" ]]; then
                    # The driver's CUDA major decides: 13.x drivers run the
                    # 13 build, older ones the 12 build
                    _cuda_major="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: *[0-9]+' \
                                   | grep -oE '[0-9]+$' | head -1 || true)"
                    if [[ "${_cuda_major:-0}" -ge 13 ]]; then
                        _pattern="bin-ubuntu-cuda-13\.[0-9]+-x64\.tar\.gz"
                    else
                        _pattern="bin-ubuntu-cuda-12\.[0-9]+-x64\.tar\.gz"
                    fi
                    _cudart=yes
                elif [[ "$GPU_TYPE" == "amd" ]]; then
                    _pattern="bin-ubuntu-vulkan-x64\.tar\.gz"
                else
                    _pattern="bin-ubuntu-x64\.tar\.gz"
                fi ;;
            Darwin)
                # macOS builds bundle Metal — no extra variant needed
                [[ "$_arch" == "arm64" ]] && _pattern="bin-macos-arm64\.tar\.gz" \
                                          || _pattern="bin-macos-x64\.tar\.gz" ;;
            *) warn "Unknown system ($_os) — please install llama.cpp manually" ;;
        esac

        if [[ -n "$_pattern" ]]; then
            info "Looking for the newest llama.cpp build ($_pattern)…"
            _releases="$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30" || true)"
            # Newest first; the "llama-b…" package, never the "cudart-…" one
            _url="$(printf '%s' "$_releases" | grep -oE "https://[^\"]*/llama-b[0-9]+-${_pattern}" | head -1 || true)"
            if [[ -z "$_url" && -n "$_cudart" ]]; then
                warn "No CUDA build found — falling back to the Vulkan build"
                warn "(it can also use other GPUs in this machine)"
                _cudart=""
                _url="$(printf '%s' "$_releases" | grep -oE "https://[^\"]*/llama-b[0-9]+-bin-ubuntu-vulkan-x64\.tar\.gz" | head -1 || true)"
            fi
            if [[ -z "$_url" ]]; then
                warn "No matching release found ($_pattern)"
                warn "Manual: https://github.com/ggml-org/llama.cpp/releases"
            else
                mkdir -p "$LLAMA_DIR"
                info "Downloading $(basename "$_url")…"
                if curl -fL --progress-bar "$_url" -o "$WORK_TMP/llamacpp.tgz"; then
                    # Copy the whole archive contents flat — WITH cp -a:
                    # some of the libraries are symlinks
                    # (libllama-common.so.0 → …so), and a `find -type f`
                    # leaves exactly those behind. The result would be a
                    # llama-server that trips over a missing .so on
                    # startup.
                    rm -rf "$WORK_TMP/llamacpp_x" && mkdir -p "$WORK_TMP/llamacpp_x"
                    # tar/cp guarded: the script runs with set -e — a
                    # broken archive must not abort the WHOLE setup,
                    # llama.cpp is optional.
                    if ! tar -xzf "$WORK_TMP/llamacpp.tgz" -C "$WORK_TMP/llamacpp_x" 2>/dev/null; then
                        warn "Archive broken — llama.cpp skipped"
                    fi
                    _source="$(dirname "$(find "$WORK_TMP/llamacpp_x" -name 'llama-server' -print -quit 2>/dev/null)" 2>/dev/null || true)"
                    if [[ -d "$_source" && "$_source" != "." ]]; then
                        cp -a "$_source"/. "$LLAMA_DIR"/ 2>/dev/null \
                            || warn "Copying to $LLAMA_DIR failed"
                    fi
                    # The CUDA runtime (cudart, cuBLAS) comes as a package of
                    # its own, from the same build — next to llama-server
                    if [[ -n "$_cudart" ]]; then
                        _rt_url="${_url/\/llama-b/\/cudart-llama-b}"
                        info "Downloading the CUDA runtime ($(basename "$_rt_url"))…"
                        rm -rf "$WORK_TMP/llamacpp_rt" && mkdir -p "$WORK_TMP/llamacpp_rt"
                        if curl -fL --progress-bar "$_rt_url" -o "$WORK_TMP/llamacpp_rt.tgz" \
                                && tar -xzf "$WORK_TMP/llamacpp_rt.tgz" -C "$WORK_TMP/llamacpp_rt" 2>/dev/null; then
                            find "$WORK_TMP/llamacpp_rt" \( -type f -o -type l \) -name '*.so*' \
                                -exec cp -a {} "$LLAMA_DIR"/ \; 2>/dev/null || true
                        else
                            warn "CUDA runtime not downloaded — llama-server needs the CUDA libraries of the system"
                        fi
                        rm -rf "$WORK_TMP/llamacpp_rt" "$WORK_TMP/llamacpp_rt.tgz"
                    fi
                    chmod +x "$LLAMA_DIR"/llama-* 2>/dev/null || true
                    rm -f "$WORK_TMP/llamacpp.tgz"; rm -rf "$WORK_TMP/llamacpp_x"
                    if [[ -x "$LLAMA_BIN" ]]; then
                        ok "llama-server installed: $LLAMA_BIN"
                        "$LLAMA_BIN" --version 2>&1 | head -2 || true
                    else
                        warn "llama-server not found in the archive"
                    fi
                else
                    warn "Download failed — llama.cpp skipped"
                fi
            fi
        fi
    fi

    # ── Model ──
    if [[ -x "$LLAMA_BIN" ]]; then
        mkdir -p "$LLAMA_MODEL_DIR"
        # A model asked for with --llama-model is that one, not any GGUF
        # that happens to lie there already
        if [[ -n "$LLAMA_MODEL" ]]; then
            _present="$(find "$LLAMA_MODEL_DIR" -name "${LLAMA_MODEL##*:}" -print -quit 2>/dev/null)"
        else
            _present="$(find "$LLAMA_MODEL_DIR" -name '*.gguf' -print -quit 2>/dev/null)"
        fi
        if [[ -n "$_present" ]]; then
            ok "GGUF model present: $(basename "$_present")"
            LLAMA_MODEL_PATH="$_present"
        else
            if [[ -n "$LLAMA_MODEL" ]]; then
                _repo="${LLAMA_MODEL%%:*}"; _file="${LLAMA_MODEL##*:}"
            else
                _repo="$LLAMA_STD_REPO"; _file="$LLAMA_DEFAULT_FILE"
            fi
            _target="$LLAMA_MODEL_DIR/$_file"
            echo ""
            if [[ -n "$LLAMA_MODEL" ]]; then
                echo "  Model: $_repo / $_file"
            else
                echo "  Model: $_repo / $_file  (~$LLAMA_DEFAULT_SIZE)"
            fi
            _download=true
            if [[ -t 0 && -z "$LLAMA_MODEL" ]]; then
                # Ctrl+D: no 5.7 GB download nobody said yes to
                read -r -p "  Download now? [Y/n] " _a || { echo; _a=n; }
                [[ "$_a" =~ ^[nN] ]] && _download=false
            fi
            if $_download; then
                info "Downloading model (this takes a while)…"
                if curl -fL --progress-bar \
                        "https://huggingface.co/$_repo/resolve/main/$_file" \
                        -o "$_target.part" && mv "$_target.part" "$_target"; then
                    ok "Model saved: $_target"
                    LLAMA_MODEL_PATH="$_target"
                else
                    rm -f "$_target.part"
                    warn "Model download failed — put it in "
                    warn "$LLAMA_MODEL_DIR/ manually later"
                fi
            else
                info "Skipped — put the GGUF file in $LLAMA_MODEL_DIR/ later"
            fi
        fi
    fi

    # ── Startup script for the LLM server ──
    # It reads the values at runtime from the config (section `llama`),
    # so tuning is possible without re-running setup. The defaults there
    # suit the default model on an 8 GB card; the health check reports
    # what doesn't fit on this machine.
    if [[ -x "$LLAMA_BIN" ]]; then
        {
            cat << 'LLMEOF'
#!/usr/bin/env bash
# llama.cpp server for Whisper Transcription Manager (generated by setup.sh).
# All parameters come from the config section `llama` — change them
# there, not here: this script gets overwritten by the next setup run.
# --show prints the command only, --check tests whether it fits.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LLMEOF
            printf 'VENV_DIR=%q\n' "$VENV_DIR"
            echo 'exec "$VENV_DIR/bin/python" -m whisper_tm.llm_server "$@"'
        } > start-llm.sh
        chmod +x start-llm.sh
        ok "start-llm.sh created (parameters from config.yaml → section llama)"

        # Set the endpoint if none is set yet — and the model asked for with
        # --llama-model: without a model set, the server takes the first
        # GGUF by name, which may be one already lying there
        _chosen=""
        if [[ -n "$LLAMA_MODEL" && -n "${LLAMA_MODEL_PATH:-}" ]]; then
            _chosen="${LLAMA_MODEL_PATH#"$SCRIPT_DIR"/}"     # relative to the app folder
        fi
        WTM_LLAMA_MODEL="$_chosen" "$VENV_DIR/bin/python" - << 'PYCFG' || true
import os
import sys
sys.path.insert(0, ".")
try:
    from whisper_tm.config import config
    config.load(None)
    if not config.get("llm", "endpoint", default=""):
        # Into the file it was loaded from, that one line only: a full
        # save would rewrite the file and drop its comments
        config.save_value("llm", "endpoint", "http://127.0.0.1:8080/v1")
        print("[OK]    LLM endpoint set to http://127.0.0.1:8080/v1")
    else:
        print("[INFO]  LLM endpoint already configured — unchanged")
    chosen = os.environ.get("WTM_LLAMA_MODEL", "")
    if chosen and config.get("llama", "model", default="") != chosen:
        config.save_value("llama", "model", chosen)
        print(f"[OK]    LLM server model set to {chosen}")
except Exception as e:
    print(f"[WARN]  Could not set the LLM settings automatically: {e}")
PYCFG
    fi
else
    info "llama.cpp skipped — set the LLM endpoint in the settings later"
fi

# ── Startup script ─────────────────────────────────────────────────────────
step "Creating startup script"

# The venv as chosen with --venv (relative paths are relative to this
# folder, where the script changes to first)
{
    cat << 'STARTEOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
STARTEOF
    printf 'VENV_DIR=%q\n' "$VENV_DIR"
    # The environment's own python, not "activate": that holds the folder's
    # absolute path and broke a moved folder (start-llm.sh does the same)
    echo 'exec "$VENV_DIR/bin/python" -m whisper_tm "$@"'
} > start.sh
chmod +x start.sh
ok "start.sh created"

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║  Setup complete!                                 ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Start:       ${CYAN}./start.sh${NC}  → http://localhost:9860"
echo -e "  Or:          ${CYAN}source $VENV_DIR/bin/activate && whisper-tm${NC}"
echo -e "  Hardware:    ${CYAN}./start.sh --check${NC}"
echo -e "  Config:      ${CYAN}${APP_CONFIG:-$SCRIPT_DIR/config.yaml}${NC} — your library: directories.media"
if ! command -v ffmpeg &>/dev/null; then
echo -e "  FFmpeg:      ${RED}missing${NC} — nothing transcribes without it (see \"Checking FFmpeg\" above)"
fi
if [[ "$GPU_TYPE" == "amd" ]]; then
echo -e "  AMD:         ${YELLOW}experimental${NC} — Whisper runs on the CPU until CTranslate2 is built with HIP"
echo -e "               (setup guide → AMD); alignment, speakers and vocal isolation use the card"
fi
if ! command -v deno &>/dev/null && ! [[ -x "$HOME/.deno/bin/deno" ]]; then
echo -e "  YouTube:     ${YELLOW}no Deno${NC} — only needed for YouTube links and the extension: ${CYAN}curl -fsSL https://deno.land/install.sh | sh${NC}"
fi
if $HF_READY; then
echo -e "  Speakers:    ${GREEN}ready${NC} (Hugging Face token / model in place)"
else
echo -e "  Speakers:    ${YELLOW}not set up${NC} — ${CYAN}./start.sh --hf-token${NC} (needs a free Hugging Face token)"
fi
if [[ -x "$SCRIPT_DIR/llama.cpp/llama-server" ]]; then
echo ""
echo -e "  ${BOLD}Local LLM server (llama.cpp):${NC}"
echo -e "   • Start: ${CYAN}./start-llm.sh${NC} in a SECOND terminal"
echo -e "     (runs in the foreground, Ctrl+C stops it)."
echo -e "   • It occupies all the VRAM — only start it when summarization/"
echo -e "     correction/translation is pending. Transcribing works without it."
echo -e "   • The defaults suit Qwen3.5-9B on 8 GB VRAM or more;"
echo -e "     tune in ${CYAN}config.yaml${NC} → section ${CYAN}llama${NC},"
echo -e "     does-it-fit check: ${CYAN}./start-llm.sh --check${NC}"
echo -e "   • Endpoint ${CYAN}http://127.0.0.1:8080/v1${NC} is set as the default —"
echo -e "     whether it's running is shown by the health check (🩺) in the app."
fi
if $INSTALL_DEV; then
echo -e "  Tests:       ${CYAN}source $VENV_DIR/bin/activate && pytest${NC}"
fi
echo ""

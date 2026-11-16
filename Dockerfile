# ╔══════════════════════════════════════════════════════════════════╗
# ║  Whisper Transcription Manager — Docker image (NVIDIA or CPU)    ║
# ║                                                                  ║
# ║  cp .env.example .env && docker compose up -d --build           ║
# ║  (docs/GUIDE.md: Docker)                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# PyTorch's wheels bring their own CUDA and cuDNN, so the base is plain
# Ubuntu: the NVIDIA Container Toolkit adds the driver when the container
# starts. The Python packages are the tested set of
# requirements-lock-nvidia.txt (`make setup-locked` installs the same).

# ── Stage 1: the Python packages ─────────────────────────────────────────
# A compiler is needed only here: diffq (audio-separator) has no wheel for
# Python 3.12 and is built from source
FROM ubuntu:24.04 AS packages

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv installs the locked set in a fraction of pip's time. Its 30-second
# timeout is too short for the CUDA packages (hundreds of MB each, many at
# once) on a slow line.
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 VIRTUAL_ENV=/opt/venv
RUN uv venv /opt/venv --python /usr/bin/python3

# torch's +cu128 builds come from the PyTorch index; torchcodec is the CPU
# build from PyPI (the cu128 one needs NVIDIA's NPP library to load), and
# requirements-lock-overrides.txt settles one conflict between two pins.
# The download cache stays with the builder: a rebuild fetches only what
# changed, not several GB again.
COPY requirements-lock-nvidia.txt requirements-lock-overrides.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /tmp && uv pip install --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --override requirements-lock-overrides.txt -r requirements-lock-nvidia.txt \
 && uv pip install --index-url https://pypi.org/simple --reinstall-package torchcodec \
        "torchcodec==$(sed -n 's/^torchcodec==//p' requirements-lock-nvidia.txt)"

# ── Stage 2: the image ───────────────────────────────────────────────────
FROM ubuntu:24.04

# libpython3.12t64: torchcodec (pyannote's audio decoding) links against
# the shared libpython
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libpython3.12t64 \
        ffmpeg \
        tini tzdata ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=packages /usr/local/bin/uv /usr/local/bin/uv
COPY --from=packages /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# ── The app ──────────────────────────────────────────────────────────────
COPY pyproject.toml LICENSE THIRD_PARTY_NOTICES.md config.default.yaml config.docker.yaml \
     docker-entrypoint.sh ./
COPY whisper_tm/ whisper_tm/
RUN uv pip install --no-cache --no-deps -e . \
 && python -m compileall -q whisper_tm \
 && chmod 755 docker-entrypoint.sh

# ── Where things live ────────────────────────────────────────────────────
# /media   the library, mounted read-only
# /config  config.yaml (written from config.docker.yaml on the first start),
#          the session key and the MCP token — a folder of the host
# /data    database, transcripts, work folder, uploads, clips, logs, chats
# /cache   downloaded models (Hugging Face, PyTorch, NLTK) and caches
ENV WHISPER_TM_CONFIG=/config/config.yaml \
    WHISPER_TM_DOCKER=1 \
    HOME=/home/whisper-tm \
    XDG_DATA_HOME=/data/share \
    XDG_CACHE_HOME=/cache \
    XDG_CONFIG_HOME=/cache/config \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    NLTK_DATA=/cache/nltk_data \
    NUMBA_CACHE_DIR=/cache/numba \
    MPLCONFIGDIR=/cache/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN mkdir -p /media /config /data /cache /home/whisper-tm

EXPOSE 9860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS -o /dev/null http://127.0.0.1:9860/api/health || exit 1

# tini: signals reach the server, and the processes it starts are reaped
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh", "python", "-m", "whisper_tm"]
CMD ["--host", "0.0.0.0", "--port", "9860"]

# Whisper Transcription Manager — the guide

Everything in detail: installing, every feature, the settings, the data, the
search and AI parts, the MCP server, the hardware and the limits. The short
overview is the [README](../README.md); updates, backups and running it as a
service are in [OPERATING.md](OPERATING.md).

**Contents**

- [Quick Setup](#quick-setup)
  - [Linux](#linux)
  - [Windows (Docker or WSL2)](#windows-docker-or-wsl2)
  - [Setup Options](#setup-options)
  - [Speaker diarization: the Hugging Face token](#speaker-diarization-the-hugging-face-token)
  - [Local LLM (llama.cpp)](#local-llm-llamacpp)
  - [Makefile](#makefile)
  - [Manual Installation](#manual-installation)
  - [Docker](#docker)
- [Features](#features)
  - [The window](#the-window)
  - [More](#more)
- [Configuration](#configuration)
  - [Network Access](#network-access)
  - [HTTP API](#http-api)
- [CLI](#cli)
- [Database](#database)
  - [Data Policy: DB-first, JSON as Backup](#data-policy-db-first-json-as-backup)
  - [Storage Location](#storage-location)
  - [Moving to Another Machine](#moving-to-another-machine)
  - [Tables](#tables)
  - [Media Indexing](#media-indexing)
  - [Three Transcript Modes](#three-transcript-modes)
  - [When is what saved?](#when-is-what-saved)
  - [Filesystem vs. DB](#filesystem-vs-db)
  - [Full-Text Search (FTS5)](#full-text-search-fts5)
  - [Configuration](#configuration)
- [RAG / Semantic Search](#rag--semantic-search)
  - [Finding situations](#finding-situations)
  - [Usage](#usage)
  - [Storage](#storage)
  - [Export (Settings → Database → Exports)](#export-settings--database--exports)
- [Images & Sounds](#images--sounds)
  - [Setup](#setup)
  - [Storage](#storage)
- [Same Audio](#same-audio)
  - [Storage](#storage)
- [Conversations](#conversations)
  - [Storage](#storage)
- [MCP Server (AI Integration)](#mcp-server-ai-integration)
  - [Setup](#setup)
  - [Access token](#access-token)
  - [Clients without MCP (Open WebUI): OpenAPI](#clients-without-mcp-open-webui-openapi)
  - [The Tools](#the-tools)
  - [Audio Clips: Listen to the Find, Don't Believe](#audio-clips-listen-to-the-find-dont-believe)
  - [Dividing Large Hit Sets](#dividing-large-hit-sets)
  - [Wait or Not](#wait-or-not)
  - [Settings (⚙ → MCP server)](#settings---mcp-server)
- [Architecture](#architecture)
- [Hardware Support](#hardware-support)
- [Known limitations](#known-limitations)
- [Prerequisites](#prerequisites)
- [Models and their licenses](#models-and-their-licenses)
- [Acknowledgements](#acknowledgements)
- [Contributing and support](#contributing-and-support)
- [License](#license)

## Quick Setup

### Linux

Needs git, curl, FFmpeg, a C compiler with Python's headers, about 20 GB of free
disk for an NVIDIA install (less without a GPU), and Deno for YouTube links —
see [Prerequisites](#prerequisites) for the one-line install per distribution.

```bash
git clone https://github.com/lexusmexus961/whisper-transcription-manager.git && cd whisper-transcription-manager

# Automatic setup (detects GPU, installs uv + venv + everything)
chmod +x setup.sh
./setup.sh

# Your library: choose it in the app (the path field at the top, on this
# machine or with the login on), or set directories.media in config.yaml
nano config.yaml

# Start — then open http://localhost:9860
./start.sh
```

The setup first asks for a **Hugging Face token** — only speaker diarization
needs it, and it takes a minute to get: see
[Speaker diarization: the Hugging Face token](#speaker-diarization-the-hugging-face-token).
Press Enter to skip it; everything else works without.

Two defaults to know: the transcription language is **detected for each
recording** — a library in one language is better off with it set under
Settings → Transcription → Language (detection guesses wrong more often on
noisy recordings); and **speaker diarization is off** — switch it on there
(Speaker diarization) or per job in the dialog that queues files. The model,
the vocal isolation and how many files run at once follow the machine
([Hardware Support](#hardware-support)).

Updating, uninstalling, where all data lives, running it as a systemd service
and the failures a first run most likely meets: [docs/OPERATING.md](OPERATING.md).

### Windows (Docker or WSL2)

There is no native Windows version — on Windows the app runs in Linux, one of
two ways:

- **Docker Desktop** (its WSL2 backend passes an NVIDIA card through): follow
  [Docker](#docker) below. In `.env` the media folder is a Windows path, such
  as `MEDIA_DIR=D:\Videos`.
- **WSL2** (Linux inside Windows) with the Linux setup. Once, in PowerShell as
  administrator:

  ```powershell
  wsl --install -d Ubuntu
  ```

  Then open *Ubuntu* from the Start menu (after a restart, if Windows asks for
  one), install these tools and follow the [Linux steps](#linux) above:

  ```bash
  sudo apt update && sudo apt install -y git curl ffmpeg build-essential python3-dev
  ```

For both:

- **NVIDIA**: the Windows driver brings CUDA into WSL2 — don't install an
  NVIDIA driver inside Ubuntu (`nvidia-smi` there shows the card).
- **Media**: Windows' drives are under `/mnt/c`, `/mnt/d`, … in WSL2 — point
  the media folder there in the settings. Reading them through `/mnt` (or
  Docker Desktop's mount of a Windows folder) is slower than WSL's own disk,
  where the data stays.
- **Memory**: WSL2 — Docker Desktop's backend too — gets half of the PC's RAM
  by default; give it more with `memory=` under `[wsl2]` in
  `%UserProfile%\.wslconfig`.
- **Network**: the browser on the same PC opens http://localhost:9860; other
  devices reach Docker Desktop's port once it is published to the network
  (`WHISPER_TM_PORT=9860`), but not a server in WSL2 without
  [extra setup](https://learn.microsoft.com/windows/wsl/networking).

Neither way is tested on a real Windows machine yet.

### Setup Options

```bash
# Force GPU type
./setup.sh --gpu nvidia     # NVIDIA CUDA
./setup.sh --gpu amd        # AMD ROCm (Linux, experimental)
./setup.sh --gpu cpu        # No GPU

# Extras
./setup.sh --dev            # + pytest, ruff, mypy
./setup.sh --llama          # + llama.cpp server (local LLM); asks only before the model download
./setup.sh --no-llama       # Skip llama.cpp
./setup.sh --no-uv          # Use pip instead of uv
./setup.sh --python python3.11  # Specific Python version
./setup.sh --no-prefetch    # Don't download the models now (~4.4 GB; the first job does)

# The Hugging Face token
./setup.sh --no-hf-token    # don't ask
./start.sh --hf-token       # later: asks for it (hidden), checks it, fetches the model
```

`setup.sh` installs the newest WhisperX and the PyTorch version it pins, as
the build for the machine (CUDA 12.8 for NVIDIA, ROCm for AMD, the small CPU
build otherwise). It needs Python 3.11–3.13 (WhisperX doesn't support 3.14 yet)
and prefers 3.12; with none of those installed, uv downloads Python 3.12 for
the venv. For exactly the environment the app is tested with (NVIDIA,
Python 3.12), `make setup-locked` builds `.venv` from
`requirements-lock-nvidia.txt` — every package pinned. It needs
[uv](https://docs.astral.sh/uv/) and only installs (no `config.yaml`, no
`start.sh`, no token question): then `make hf-token` and `make run`.

### Speaker diarization: the Hugging Face token

Telling speakers apart (who speaks when) uses pyannote's free model
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1).
It is **gated**: Hugging Face hands it out only to signed-in accounts that
accepted its conditions. Once:

1. Sign in at [huggingface.co](https://huggingface.co) — or
   [create a free account](https://huggingface.co/join).
2. Open [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   and accept the conditions. The form asks for a company or university and a
   use case; access is granted at once.
3. Create an access token of type **Read** at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A
   fine-grained token needs *Read access to contents of all public gated repos
   you can access*.
4. Hand it to the app — any one of:
   - the setup's first question (`setup.sh`, hidden input)
   - `./start.sh --hf-token` (or `make hf-token`), hidden input too
   - in the app: *Help & setup* → *Setup* → *Speaker diarization*,
     or Settings → Speakers
   - the environment variable `HF_TOKEN` (Docker: see below) — set from a file
     or a secret store, not typed into a command line, where the shell history
     keeps it

The setup, `--hf-token` and the setup guide check that the token works and may
fetch the model — a missing approval is named as such, with the link — save it
in the config file (`transcription.hf_token`, readable only by its owner) and
download the model right away (~32 MB). From then on it loads from the cache;
the token is only needed again for a new download. Without it, transcripts
have no speakers — and with it only once diarization is switched on: it is off
by default (Settings → Transcription → Speaker diarization, or per job).

No other model the app uses is gated: Whisper, the alignment models, the voice
vectors (ReDimNet), EmbeddingGemma 2 for Images & sounds and the GGUF language
models download without an account.

### Local LLM (llama.cpp)

Summarization, correction, and translation require a language model. The
setup asks if you want to install `llama.cpp` right away:

```bash
./setup.sh --llama                       # Binary + default model (~5.7 GB)
./setup.sh --llama --llama-model unsloth/Qwen3.5-27B-GGUF:Qwen3.5-27B-Q4_K_M.gguf
```

It downloads the appropriate release from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases)
(on Linux the CUDA build for NVIDIA with its runtime, Vulkan for AMD; the
CPU build without a GPU), puts it in `llama.cpp/`, fetches a
GGUF model to `models/gguf/`, and creates `start-llm.sh`. Then:

```bash
./start-llm.sh              # LLM server on port 8080, context 32768
./start.sh                  # Whisper Transcription Manager
```

`start-llm.sh` runs in the foreground — start it in a second terminal (or as a
service); Whisper Transcription Manager never starts the language model itself.

The default model, Qwen3.5-9B at Q4_K_M, takes ~7 GB of VRAM with its 32k
context, and the config's `llama` section is tuned for it on an 8 GB card. On
a smaller card llama.cpp runs the layers that don't fit on the CPU (slower);
another model may want other values — the comments there say which
(`n_cpu_moe` for a mixture of experts), and `./start-llm.sh --check` tells
what doesn't fit.

**One card for both.** The language model and transcription share the card's
memory. Transcribing with large-v3 needs 5.6–6.8 GB at its peak (turbo
less; with diarization and vocal isolation the steps follow each other, they
don't add up), the default language model ~7 GB. Roughly, per card:

| Card | Language model while the queue transcribes |
|---|---|
| 6 GB | No — stop `start-llm.sh` while the queue runs, or run the model on the CPU (`llama.n_gpu_layers: 0`); the app transcribes with turbo here |
| 8 GB | No — large-v3, which the app picks on this class, takes most of the card; the same choices as above |
| 12 GB | Only partly — a part of its layers on the CPU (`llama.n_gpu_layers`), or a smaller model |
| 24 GB | Yes |

When the card is short, a transcription waits for memory and the queue pauses
after a while with a message; summaries wait in their own queue until the
language model answers.

The endpoint `http://127.0.0.1:8080/v1` is automatically added to the config
if one doesn't exist yet.
The bundled servers (chat 8080, embeddings 8081, reranking 8082) require a key the
app makes at the first start (`llama_api_key` next to `config.yaml`, readable only by
you) and sends itself; another program that talks to them sends it as
`Authorization: Bearer <key>`. `llama.require_key: false` turns it off. Alternatively, you can add any providers (Ollama, OpenAI,
vLLM/LM Studio, custom) in settings under *AI models* and assign them to individual tasks.

### Makefile

```bash
make setup          # Complete setup
make setup-cpu      # CPU only
make setup-dev      # + dev tools
make hf-token       # Speaker diarization: check and save the token, fetch the model
make run            # Start server
make test           # Tests (after make setup-dev)
make test-cov       # Tests + coverage
make check          # Hardware diagnostics
make clean          # Clean up
make help           # Every target
```

### Manual Installation

With Python 3.11–3.13, a C compiler and Python's headers (see
[Prerequisites](#prerequisites)), in the app's folder (a git checkout): it runs
from there — `pip install -e` keeps it in place, while a wheel (`pip install .`)
carries neither the pages nor `config.default.yaml`, and refuses to start.

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# NVIDIA — on Linux PyPI's torch is the CUDA build already
pip install whisperx "audio-separator[cpu]"   # [cpu]: see requirements-nvidia.txt

# CPU only — the small CPU build of torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu whisperx "audio-separator[cpu]"

# Whisper Transcription Manager, with Ask/MCP (fastmcp) and link downloads (yt-dlp); then
# beartype as the app is tested with (fastmcp and audio-separator disagree)
pip install -e ".[mcp]" yt-dlp
pip install -r requirements-lock-overrides.txt
# Your settings: only what differs from config.default.yaml
printf 'directories:\n  media: "/path/to/your/library"\n' > config.yaml

# Speaker diarization (optional, see above), then start
whisper-tm --hf-token
whisper-tm
```

### Docker

Needs Docker with Compose 2.24 or newer and BuildKit (the `docker-buildx`
plugin; older distribution and NAS packages fail on the compose file's
`!reset` or the Dockerfile's cache mounts) and, for an NVIDIA card, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
— `docker run --rm --gpus all ubuntu nvidia-smi` must show the card.

```bash
git clone https://github.com/lexusmexus961/whisper-transcription-manager.git && cd whisper-transcription-manager
cp .env.example .env       # edit it: MEDIA_DIR (your library), HF_TOKEN
mkdir -p config            # your configuration, created on the first start
docker compose up -d --build
```

The first build takes about ten minutes. Measured on a fresh build (16 Nov
2026): the image is 14 GB and Docker's build cache keeps another 23 GB for
the next build (`docker builder prune` frees it) — so have about 40 GB free
before the first build. The models follow on first use (2–4 GB, by card). The web interface is then on http://localhost:9860.
`.env` holds what differs from one machine to the next, so
`docker-compose.yaml` stays as it is and an update is just
`git pull && docker compose build --pull && docker compose up -d` (`--pull`
brings the base image's security fixes).

- **Speaker diarization**: `HF_TOKEN` in `.env` — its account must have
  accepted the conditions of the diarization model first, see
  [Speaker diarization](#speaker-diarization-the-hugging-face-token).
- **Without an NVIDIA card**: remove the `#` in front of the
  `COMPOSE_PATH_SEPARATOR` and `COMPOSE_FILE` lines in `.env` — the server then
  transcribes on the CPU, much slower.
- **Where things live**: `./config/` holds `config.yaml` (written from
  `config.docker.yaml` on the first start; the settings page saves into it),
  the session key and the MCP token — keep it private. Database, transcripts,
  chats, analyses and logs are in the `whisper-tm-data` volume, downloaded
  models in `whisper-tm-cache`; both survive rebuilds and updates
  (`docker compose down -v` deletes them). The media are mounted read-only,
  so transcripts are kept in the data volume (`transcript_mode: central`),
  not next to the recordings.
- **User**: the server runs as the owner of `./config` (not as root), so
  what it writes there stays yours; `PUID`/`PGID` in `.env` choose another.
- **Other devices**: the port is published to this machine only
  (`127.0.0.1:9860`). `WHISPER_TM_PORT=9860` in `.env` publishes it to the
  network; then other devices open it by the machine's IP address
  (`http://192.0.2.10:9860`), and a name (`http://nas:9860`) works once it is
  in `security.trusted_origins` in `config/config.yaml` — the API answers only
  names it knows (against DNS rebinding). Anyone who reaches the port can use
  the app: see [Network Access](#network-access).
- **A language model** on the same machine (Ollama, a llama.cpp server) is
  `http://host.docker.internal:<port>` in Settings → AI models; it must listen
  on more than 127.0.0.1 (Ollama: `OLLAMA_HOST=0.0.0.0`). (Rootless Docker and
  Podman reach the host another way — see their network documentation.)
- **What the image lacks**: llama.cpp — so the search reranker can't start
  (searches keep their fused order: they work, a little less precisely) and
  neither can the local embedding server; and Deno — so YouTube links may fail
  or miss formats. Images & sounds' environment has not been tried in the
  image.
- **Commands** run in a second container: `docker compose run --rm
  whisper-tm --check` (the system check), `--remap-path` (see
  [Moving to Another Machine](#moving-to-another-machine)). Logs:
  `docker compose logs -f`.

## Features

### The window

- **The rail** on the left holds the places: *Add*, *Home*; *Library*, *Queue*;
  *Find*, *Ask*, *Analyses*, *Voices*; *Machine*, *Settings*, *Help & setup*.
  Pinned places and window tabs (up to six) sit in a group of their own, the
  same list as Home's pinned row. Badges show running and failed work; the dot
  at *Machine* turns yellow or red when something needs a look. The rail folds
  to icons. Alt+← / Alt+→ (or the mouse's back button) walk back and forward
  through places, tabs, recordings and steps inside a view; Esc undoes a step.
- **The top bar** keeps the search field (*/* focuses it: places, settings,
  recordings, what is said), the load meters, *Unload* and the notifications.
- **Home** — live cards (running, just finished, failed, recently opened),
  then every place and tab in groups that open and close.
- **Library** — the whole library as a folder tree with each folder's share
  transcribed, failures and whether it is watched, beside the chosen folder's
  statistics, *Needs attention* with what to do about it, and who speaks the
  most there. *Stats* in the folder bar opens it on the open folder.
- **Queue** — *Jobs*: transcriptions, AI jobs, exports and clips in one list
  (running, waiting, failed, done), each queue paused and resumed by its own
  button; *Edit in bulk*, *From the library* and *Add a folder* are its other
  tabs.
- **Machine** — *Health*, *Load history* and *Models & connections* (does each
  model and server answer, and where to set it).
- **Help & setup** — where things are, how to, keys, and the setup guide.
- **A recording** — its name and one quiet line, *Export* and ⋯ (voice map,
  log, download the original, delete); the player with who-speaks-when lanes;
  then the tabs *Transcript*, *Summary*, *Translation*, *Speakers* (who was
  there, each voice's numbers), *Clips*, *Statistics* (with the conversation),
  *Details* and *Analyses*. Tabs that don't fit go under *More*.
- **Export** — one page for every way out, opened from a recording's header,
  the folder bar, the file list's selection bar (⬇) and the Library. *This
  file*, or *Multiple files*: the library's folders beside the focused
  folder's recordings (a click picks one, Ctrl adds, Shift picks a range, *All
  of* takes the whole folder; a name filter, *added today / yesterday / in the
  last 7 or 30 days*, subfolders). Ten formats: text, subtitles (SRT, WebVTT),
  Word, PDF, Markdown, a table (CSV), data (JSON), and **video** (original,
  MP4, WebM, MKV; a smaller size, subtitles burned in or beside it, only some
  speakers, the pauses out) or **audio** (MP3, WAV, FLAC, M4A, Opus, the
  original track; quality, sample rate, mono, only some speakers, the pauses
  out). The options — the transcript, the translation, both or the summary;
  the summary on top; only some speakers; only the clip cuts; only lines with
  a word; times at every line, once a minute or none; names; paragraphs;
  unsure words marked; a header; subtitle rules; a ZIP with a file each or one
  document, with the folders kept — fold under one line that says what is
  chosen, and are remembered; presets (*YouTube subtitles*, *Notes for the
  team*, your own). A preview shows the first recording. Small text exports
  download at once; video, audio and big ones run as jobs in the Queue, the
  result kept a day (`core/exporter.py`, `core/media_export.py`,
  `core/export_jobs.py`, `/api/export*`). The PDF uses the standard font:
  characters outside Latin-1 are replaced — Word keeps them.
- **Clips** (a recording's tab) — a small video editor: several cuts on the
  waveform timeline (zoom, IN/OUT, the cuts as boxes on it), a list to edit
  their times, play, reorder or drop them; cuts picked from the transcript's
  lines (the first and the last line), 30 s from the playhead, or **Best
  moments (AI)** — the language model suggests three self-contained moments,
  each a cut marked *AI* to keep or drop. The frame: the original 16:9,
  vertical 9:16 or square 1:1 at a fixed position (no face tracking); pauses
  out (found as silence, a little air kept), subtitles burned in, soft fades,
  one clip or one clip each. *Make the clip* runs as a job in the Queue; what
  was made is listed under the cuts. The cuts are kept per recording on the
  server — Export's *only the clip cuts* uses them (`core/cuts.py`,
  `core/clip_render.py`, `core/best_moments.py`, `/api/clips*`).

### More

- **Platforms**: Linux, natively or in Docker; Windows through Docker Desktop or WSL2
  (see [Hardware Support](#hardware-support))
- **GPU**: NVIDIA CUDA; AMD ROCm on Linux (experimental, untested: Whisper itself runs on the
  CPU there unless CTranslate2 is built with HIP); without a GPU on the CPU
- **NAS-Scale**: SQLite + FTS5 for instant search across 100k+ transcripts
- **Independent Operation**: Runs as a service, no browser tab needed
- **Multi-Tab**: live updates via Server-Sent Events in every open browser tab
- **Full Config**: YAML-based, nearly every setting on the settings page and in the
  API (a few — allowed roots, trusted origins, the MCP token — only in the file)
- **Database-backed**: SQLite with a full-text search index, always on (JSON next
  to the media as a portable copy)
- **Batch Processing**: Server-side queue, survives browser disconnects; works on
  several files at once (*queue lanes*) with every model loaded once
- **Vocal Isolation**: MelBand Roformer (Kim), separated in fixed chunks, a pre-filter skips clean audio
- **Clips**: several cuts made into one clip or one each, vertical or square, without
  pauses, with subtitles (see [The window](#the-window)); *≋ Same audio* finds a
  stretch in every other recording
- **Audio analysis**: fingerprints, waveform and loudness of every recording, kept once —
  analyzed while transcribing, on the CPU
- **Speaker Diarization**: pyannote.audio with custom speaker naming
- **Settings**: 18 pages in four groups, every setting with a description,
  a search over all of them, unsaved and changed-from-default marks, per-setting
  reset
- **LLM + Embedding**: separately configurable (chat LLM and embedding model),
  OpenAI-compatible incl. **llama.cpp**, Ollama, vLLM, LM Studio; Machine →
  *Models & connections* shows live what's reachable
- **Tests**: config, database, security, jobs, queue, speakers, LLM/embedding, frontend wiring — `make test`
- **Search & Ask** (Enter in the search field, or ⤢ beside it) — a workspace in the
  system of the voices workspace, as big as the window, in tabs:
  - **Find**, the start tab: describe a moment in your own words — what is said, seen
    or heard, who is in it, what the file was called. A description typed in the top
    bar comes here (a few words or quoted words go to the text, a file name to the
    files). The description is searched by meaning at once; the language model plans
    the rest, checked against the voice library. The searches run at once in tiles of
    their own (word for word, by meaning, seen, heard — and all of them again inside the
    files whose name has what you remember and in what the named people say; each can
    be maximized), and *Best matches* brings them together per moment — each recording
    once, its other moments behind *+n more moments*, its copies (the same video as
    .mp4, .mkv and .webm, a backup) folded in as *suspected duplicates*, found by what
    the files hold (their sound's fingerprints, length, words, picture), never by their
    names. The language model
    checks the best twelve with their stills and the facts the program knows (does the
    name have the remembered piece, is it a video, does the person speak in it); what
    doesn't fit what you said caps its verdict. A click plays the moment right there: a
    clip cut from the recording, whatever its format; ↑ ↓ move through a list. *Ask
    about these results* continues in Ask — and Ask can run Find itself.
  - **Text**: one query bar (*Words*, *Words + meaning*, *Meaning*) and the filters as a
    toolbar (file, speaker label, person, *Filters & order* with date, duration, words,
    speed and sort); the hits as *Hits* (one dense line per hit, the passage in context
    beside it), *By recording* (each recording's hits across its runtime) or *Map* (the
    folder tree as a treemap of the word hits, the brighter the more).
  - **Images & sounds**, **Same audio** (every recording that holds a snippet, and the
    duplicates — see [Same Audio](#same-audio)), **Conversations** (who talked with whom,
    and how — see [Conversations](#conversations)), **Files** (file and folder names — each
    folder once, the files under it with their values in columns), **Ask**, **Deep
    search** and **Analyses**.
- **Ask** (the rail, or *Ask* in the search field's list, which takes the term and filters
  along) — a chat with the language model set under AI models (task *Chat*). It looks
  things up itself through the tools of the app's own MCP server, spoken to in memory
  (search, semantic search, transcripts, who speaks where, who talked with whom, the
  library) and shows each step between the texts where it made it. Every quote carries a
  reference (`S12`): a chip that plays the passage in the background player or opens its
  transcript. Players as in Open WebUI — `<audio>` around a tool result's clip address,
  or `<audio>S12</audio>` — show as a player with the quote. Beside the conversation, the
  passages the answers quote or play. The reference is resolved to the recording's real
  place, also when its file name exists under several roots. By default it only reads;
  Settings → Search → Ask (chat) → *Allow actions*, with Settings → MCP server →
  *Allow work*, lets it transcribe, download, summarize, translate, proofread and steer
  the queue — and then words in a transcript, or in a video the extension attached, can
  lead the model to do so. Ask's conversations are kept
  under the data folder (`chats/`); `core/chat.py`, `/api/chat/*`.
- **Analyses** (the rail; 🧠 in the file list's selection bar; a recording's *Analyses* tab
  lists the ones it is part of) — one
  task over many recordings: summarize three videos together, compare them, find every
  place where something happens in a whole folder. The set is the files selected in the
  file list, a folder (with or without its subfolders) or every recording of a search;
  before anything runs, the page counts it — recordings, words, how it will be read and
  roughly how long. The language model reads every transcript of the set, each line with
  a reference (`S12`): what fits goes in one call, a larger set in parts
  (`analysis.part_tokens`, 24,000 by default) whose notes keep the passages they rest on,
  then one answer from all notes; notes longer than an answer holds (a "find every …"
  over a large folder) stay as they are, with an overview above them — switched off,
  the notes alone are the answer. Three orders:
  folder by folder, by file name, and *dialogue lines together* for game files named
  like `DIA_Npc_Topic_15_00` (such games keep each voice in a folder of its own; in
  line order the lines of one dialogue stand together). It runs in the LLM queue on the
  server (the summary task's model, or another), shows each part's notes as they come,
  can be cancelled, and after a restart goes on with the parts already read. The answer
  cites every passage as a chip that plays it or opens its transcript; the notes of
  each part stay with it, it downloads as Markdown with its passages, and *Ask about
  it* opens a conversation in Ask with the analysis attached. Kept under the data folder
  (`analyses/`); `core/analysis.py`, `/api/analyses/*`.
- **Conversations** (tab of Find and of Voices; a recording's *Statistics* tab) —
  who talked with whom, and how. Every transcript's speech, words per sentence, speaker
  changes, talk shares, questions and longest monologue are counted from its segments
  with every write; the tab finds "an hour with Alice, Bob and Carol in long sentences"
  by the people (all together or any of them), the length, the kind (*Detailed*,
  *Back-and-forth*, *Interview*, *Group*, *Monologue*) and sentences longer or shorter
  than usual — for the people chosen, than usual for them; otherwise measured against
  the library's own conversations in the same language. In a recording of several
  hours, the stretch where they talk together. ✂ hands them to Voices → Extract: one
  voice's sentences from just these conversations, as a TTS dataset. Ask, Find and MCP
  clients search the same way (see [Conversations](#conversations));
  `conversations/`, `/api/conversations*`.
- **Browser extension** (`browser-extension/`, Chrome/Edge/Brave, loaded unpacked) — on
  any page, hover a YouTube video (a thumbnail, a link, an embedded player, a watch page)
  and click its badge; select as many as you like, then *Summarize*, *Transcribe* or
  *Analyze…*. The server downloads each video into the upload folder (a video
  in the library already — known by its id, or by the name its title gets — is not
  downloaded again), queues it at the front, summarizes it right after its transcript,
  or starts one analysis over all of them once every transcript is there. The bar on
  the page and the toolbar popup show each video's state and its summary, with a link
  that opens it in the app (`/?open=<path>`, `/?analysis=<id>`). Right-click a YouTube
  link: summarize or transcribe it. `core/ingest.py`, `/api/ingest/*`; how to install:
  `browser-extension/README.md`.
- **Continuous scanning** (*Auto* in the folder bar: off, the whole library, or chosen
  folders — *Watch this folder*; Settings → Library → Scanning) — the library (or the
  chosen folders) is checked every 15 minutes (`autoscan.interval_min`) for new files; a new file is
  added to the library and, unless switched off, queued for transcription with the
  defaults. The first round of a folder only notes what is there, so switching it on over
  a big library queues nothing old; a file still being copied waits until it stopped
  changing; rounds only read the folders whose time changed (one look per folder, the
  rest from memory). Off by default; the state lives in `<data>/autoscan.db`
  (`core/autoscan.py`, `/api/autoscan`).
- **Queue lanes** (`queue.lanes`, default 3) — several files in flight at once. Each
  model serves one file at a time (`core/gpu_gate.py`: Whisper alone, other steps side
  by side while the card has room); what overlaps is copying, decoding, writing and
  booking, so the GPU stops waiting — short voices jobs ran 2× faster, a mixed set with
  long files 1.3×. One file per name or path in flight, one long file at a time
  (`queue.long_file_sec`); stop, abort and ✕ reach every running file. `1` = one after
  the other.
- **Queue → Edit in bulk** — edit the queue instead of just viewing it: filter (folder, search,
  setting), select individually or "all matches", change settings for the selection
  (e.g. save VRAM for all waiting items), remove, move forward/backward, re-queue
  failed items — always with a dry run and match count before the click (`/api/queue/editor/*`).
- **Queue → From the library** — pick files from the whole library
  by what is known about them and queue them as one job: folder, name, extension,
  status, duration, size, transcription date, model, language, separation model, app
  commit, diarized or not, voices or not, voices without stored speech spans, enough
  speech for a voice, summaries,
  translations, correction runs, queue history, "same title already transcribed in
  another format". The count, audio hours and a rough GPU estimate follow every
  change; queuing — at the end or at the front — refuses if the library changed
  since it was counted (`/api/queue/library/*`, `core/library_select.py`).
- **Queue → Add a folder** — a folder's files without a
  transcript, subfolders included, into the running queue: at the front (next up, in
  path order; files of the folder already waiting move up with their own settings) or
  at the end, with the options of the transcription dialogs. It starts on the folder
  open in the file list and shows the count, the audio and the estimate before
  anything is queued; the folder field is the file list's path field (browse, filter,
  favorites), and the file list below it loads more while scrolling.
- **Job kinds** — *Transcribe*; *Voices only* (diarization + voice vectors for a
  transcribed file, through the same worker and the same code as a transcription —
  verified window-for-window identical — without touching the transcript; files that
  already have voices are skipped so their names stay, files without a transcript are
  skipped too). A transcript diarized by an earlier run keeps its speaker labels: the
  fresh diarization's labels are renamed after the transcript's speakers by time
  overlap, so a voice and its sentences share one label; a voice without a counterpart
  gets a free label instead of someone else's. Every run is recorded under
  `settings.voices` (windows, labels, reason when none) — the library criterion
  "Voices-only run done" keeps such files out of the next selection;
  *Renew voices* (a voices-only run that replaces a file's existing voices, e.g. the
  older ones without speech spans, and hands their names, trash and history to the
  new voices of the same people — matched by how alike the voices sound, not by
  label, since a label only means something within one diarization; if any named or
  discarded voice has no clear counterpart, the file keeps its voices and the run's
  log says which); *Re-transcribe* (warns: edits and voice names are lost).
- **Voices** (the rail) — one workspace with six tabs:
  - *Map*: every voice of the collection as tiles (or a graph), grouped by how alike
    they sound; the details of a voice or a group sit docked beside the map, and the
    largest groups show there while nothing is chosen. *Locate* from a transcript
    finds the recording by its voice windows (not by its file name, which many
    recordings share), shows them even in groups below the minimum size, and says
    when there is nothing to show (no voice vectors, or in the trash).
  - *Assign*: the guided assignment — the group that decides the most open speech
    first, its samples (up to 48, *Play all* one after another, the first N seconds
    each, Ctrl+P), name / trash / skip; beside them who the group sounds like (the
    nearest named people with similarity and votes — a click takes the name), when,
    where and how its speech is spread. The last decisions sit next to it (a click
    opens one to its samples, to hear what it was), and an undone decision comes back
    as the question it was. Every name field suggests names while typing (↓/↑, Tab).
  - *Help*: what each tab is for, the words the workspace uses, and its keys.
  - *People*: the named people with voices, speech, recordings and dates; rename, or
    merge one name into another. **Listen through** opens every voice of a person as a
    list with one player: autoplay, only the first N seconds of each if you like, and
    many filters (named directly or by spreading, date, folder, file, label, speech
    length, one voice per recording). The doubtful ones come first — voices nearer to
    another person, then the least fitting (best similarity to the person's direct
    voices in *other* recordings). Each can lose the name (N), go to the trash (T) or be
    named as someone else (A), one by one or many at once; every decision is one entry
    of the History, the last one undoable right there.
  - *History*: every change to a name or to the trash is one entry with the state of
    each voice before and after (`speaker_decisions`). **Undo** puts back exactly what
    that click changed — including a naming's spread — and only where no later
    decision changed the voice since; **Redo** reverses it. What was decided before
    the history existed is rebuilt once from the times of the names and the trash.
  - *Extract*: one person's voice as clips with their sentences — named voices, and
    optionally the ones recognized as the person; filters for date (from file names,
    also German day-first dates), folder, text, length, speaking rate, and how sure
    Whisper was of the sentence (in percent, from its mean token probability — stored
    per sentence, not per word; 75 % by default); only where the voice is alone (the
    stored speech spans; a voice without them is checked by the transcript's overlap
    marker instead, and a note offers to leave such voices out where a preview has any). The
    preview follows the filters and shows every clip with its certainty; with word
    timestamps every word is coloured by how sure its alignment is and lights up while
    the clip plays (also in the clip editor, and the words go into `metadata.jsonl`). **Clip editor**: ✎ opens a
    clip as a waveform with ±2.5 s around it — drag or nudge its bounds (←/→ start,
    Shift+←/→ end), *Cut at pauses* (S) snaps both to the nearest quiet spot, correct
    the text, keep (Enter) or drop (Del) and go on; the edits are kept on the server
    per person and sentence until it restarts, so a new preview has them too. On
    export, clips not trimmed by hand are cut at the nearest pause (half words at the
    edges), and *only the clips I kept* is an option. Written as a TTS dataset (WAV
    per clip, `metadata.csv` in the LJSpeech layout, `metadata.jsonl` with certainty
    and edits) or as one audio file with subtitles — into `<clips folder>/voice-exports`
    to download, or **into a folder of your choice** on the server (in the allowed
    folders; the dataset also unpacked, straight where a TTS trainer reads it).
    Use a person's voice only with their consent — such a dataset can train a
    model that speaks like them. An
    export never replaces anything: a name that is taken gets `-2`, `-3`
    (`/api/voices/*`, `core/voice_extract.py`). **Whole recordings** (or the button
    *All recordings with this voice*) exports every recording the person speaks in as
    it is, with `recordings.csv` — the same filters decide which. **From
    conversations**: the search view's Conversations tab hands its filters over (✂
    Extract there) — then only the sentences of those conversations count (a
    sentence whose middle lies in one of them; a stretch of a long recording, or a
    whole recording), e.g. Alice's voice from the hours she talks with Bob and Carol
    in long sentences.
- **Recorded settings** — every transcript stores how it was made
  (`transcripts.settings_json`: model, languages, device, batch, VRAM mode, separation,
  diarization asked for and done, voice vectors, app version and commit).

## Configuration

Whisper Transcription Manager uses a YAML config file — the first of these that exists:

1. `--config` CLI argument
2. `WHISPER_TM_CONFIG` environment variable
3. `config.yaml` in the user config folder: `~/.config/whisper-tm/`
4. `./config.yaml` (working directory)

The setup script creates a small `./config.yaml`. A config file holds only
what differs from `config.default.yaml`, which documents every setting — so the
defaults of a newer version still arrive. The settings page saves back into the
file that was loaded (without one, into the user config folder), with only what
differs from the defaults. Key settings:

```yaml
directories:
  media: "/mnt/nas/videos"        # Your media library
  work: "/mnt/ssd/whisper-tm-work"  # Fast local disk (SSD) — not /tmp, often RAM
  transcript_mode: "alongside"    # alongside | central | database

hardware:
  device: "auto"                  # auto | cuda | rocm | cpu
  low_vram: "auto"                # true = unload models after every file (auto: up to 11 GB)

database:
  fts_enabled: true               # Full-text search index
```

**Only in the file.** The settings page can't change what decides who may
reach the app, what it may touch and which programs it starts — it needs no
login while the login is off: the sections `security` (allowed roots,
trusted origins and extensions), `auth`, `server`, `logging` and `llama`, and
`database.path`, `embedding.local_extra_args`, `mcp.token`,
`rerank.extra_args`, `rerank.local_model`, `rerank.host`,
`audio_analysis.path` and `youtube.allow_private_hosts`. Edit them in the
config file and restart.

**Only from this machine** (a browser on the server itself, or anywhere with
the login on): where the MCP services listen and whether they need their
token (`mcp.host`, `mcp.require_token`), how the MCP server reaches the app
(`mcp.base_url`, `mcp.api_token`, `mcp.listen_base_url`), the app's folders
(`directories.uploads`, `work`, `clips`, `logs`, `transcripts`) and the
environment Images & sounds starts (`media_search.environment`, `path`,
`cache_dir`).

### Network Access

Out of the box the server listens on this machine only
(`server.host: "127.0.0.1"`; in Docker the port is published to
`127.0.0.1`) and asks for no password (`auth.enabled: false`). To use it from
other devices, open it to the network — `server.host: "0.0.0.0"` (or
`--host 0.0.0.0`; in Docker `WHISPER_TM_PORT=9860`). Then anyone who can reach
the port can read the transcripts, browse the media folder, start jobs,
change settings — and have the server contact addresses of their choosing
(testing an LLM or embedding endpoint does) —, and pages of other sites may
show the app in a frame (it isn't refused, so dashboards such as Home
Assistant can embed it). On a network you don't fully trust, set a password:

```yaml
auth:
  enabled: true
  password: "…"        # plaintext, or a bcrypt hash
  api_token: "…"       # for scripts: header X-API-Token
```

A bcrypt hash needs the `bcrypt` package in the app's environment:
`uv pip install --python .venv/bin/python bcrypt` (or `.venv/bin/pip install
bcrypt` after `./setup.sh --no-uv`).

Either way, pages from other sites can't use the server through your browser
(`security.trusted_origins` in `config.default.yaml`); the settings that start
programs — llama.cpp, the reranker, the server and login sections — can only
be changed in the config file; and without a login only a browser on this
machine may open the MCP services to the network, move the local embedding
server or the Images & sounds service, or choose where the vector store and
the backups are written. Failed logins and wrong API tokens are slowed down
per client after five attempts, and changing `auth.password` ends every
session made with the old one. To log every browser out without changing the
password, delete the `secret_key` file next to the config and restart (while
`server.secret_key` in `config.yaml` is empty).

**A machine several people use.** Everyone logged in to it, and every program
running there, counts as this machine: without the login they can read every
transcript, see the MCP token and change the settings above. On such a host,
switch the login on and keep the data folder readable only for yourself.

The web server is Werkzeug's, as the warning at startup says: fine on your own
network. To reach the app from the internet, put a reverse proxy with its own
login and TLS in front of it (and list its address in
`security.trusted_origins`).

**"This machine" is the address a request comes from.** A request carrying
`X-Forwarded-For`, `Forwarded` or `X-Real-IP` never counts as this machine —
so a reverse proxy on the same machine must send one (nginx doesn't by
default: `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`),
or every client it passes on may do what only this machine may: read the MCP
token, change the settings listed above. A tunnel (`ssh -L`, `socat`) sends
no such header: whoever uses it counts as this machine — or switch the login
on, which these rules then don't depend on.

### HTTP API

Everything the page does goes through a JSON API on the same port. Its
reference is generated from the routes: `/api/docs` (JSON), `/api/docs.md`
(Markdown; `?brief=1` for a short one, handy for a language model) and an
explorer under Help & setup → Setup → API reference. The query and body fields there
come from reading the code, best effort; endpoints marked **stream** answer
with Server-Sent Events.

From a script: with the login on, send `X-API-Token: <auth.api_token>`. A
script that reaches the server under a DNS name that isn't this machine's
gets 403 until the name is in `security.trusted_origins`; a request that
changes something and carries another site's `Origin` is refused the same way.

## CLI

`./start.sh` takes these options and passes them on — `./start.sh --check`,
`./start.sh --port 5001` — since it runs the environment's own Python. The
command `whisper-tm` itself works only in a shell where the environment is
activated (`source .venv/bin/activate`); in Docker it is
`docker compose run --rm whisper-tm <options>`.

```
whisper-tm [options]

  --config, -c PATH    Config file path
  --port, -p PORT      Server port (default: 9860)
  --host HOST          Server host (default: 127.0.0.1 — this machine only;
                       0.0.0.0 opens it to the network)
  --media, -m PATH     Media directory
  --workdir, -w PATH   Work directory (SSD recommended)
  --device, -d DEVICE  Compute device (auto/cuda/rocm/cpu)
  --check              Hardware and installation check (setup guide's checks)
  --gpu-test           With --check: also compute on the GPU (Whisper tiny)
  --hf-token           Set up speaker diarization: ask for, check and save the
                       Hugging Face token, download the model
  --no-input           With --hf-token: never ask (HF_TOKEN or the saved token)
  --prefetch           Download the models the settings need (Whisper, alignment,
                       NLTK's sentence splitter, separation; with diarization on,
                       its models), then exit — the setup offers it
  --debug              Debug mode
  --log-dir PATH       Log folder (default: logging.debug_dir, logs-debug in the
                       data folder)
  --trace-sql          Log every SQL statement (switches debug mode on)
  --trace-calls        Log every function call (slow)
  --version, -v        Show version

Moving to another machine (see Database → Moving to Another Machine):
  --show-roots               Where do the stored paths point?
  --remap-path OLD NEW       Translate stored paths (dry run)
  --apply                    …and write
  --drop-conflicting         …replacing rows a new server already created
  --no-backup                …without backup copies
```

## Database

Whisper Transcription Manager uses SQLite with FTS5 as the **primary source** for all transcript data.

### Data Policy: DB-first, JSON as Backup

All read access (transcripts, segments, status, metadata, search,
summaries) goes **first to the database**. JSON files on the filesystem are
preserved and serve as:

1. **Backup/Export** — human-readable and portable (subtitles for media
   players are a separate export: `transcription.export_alongside` writes
   `.srt`/`.vtt` next to each recording)
2. **Fallback** — if a transcript is found only as JSON (legacy data),
   it is read **and automatically imported into the DB** (self-healing)

Write access always goes to **both** destinations (DB + JSON), except in
`transcript_mode: database` mode (then DB only). New features (e.g., RAG indexes)
should be developed exclusively against the DB.

### Storage Location

```
~/.local/share/whisper-tm/library.db
```

Configurable in `config.yaml` — on a local disk: SQLite's write-ahead log
isn't safe on a network share. Backups may go there:
```yaml
database:
  path: "/srv/whisper-tm/library.db"        # a local disk
  backup_path: "/mnt/nas/whisper-tm-backups" # copies, e.g. on the NAS
```

### Moving to Another Machine

Every file is identified by its absolute path (`docs/PATHS.md`). If the
new server mounts the same media under a different name — `/mnt/media`
becomes `/srv/media` — the transcripts, summaries, speaker vectors and search
index are all still there, but point to paths that don't exist. The
command line translates them in the main database, `speaker.db`, `rag.db`,
`media.db` (Images & sounds) and `config.yaml` in one go:

```bash
# 1. Copy config.yaml (secret_key and mcp_token lie next to it) and the data
#    folder (~/.local/share/whisper-tm: the databases, chats, analyses) to
#    the new machine. Do NOT start the server yet.
# 2. Where do the stored paths point?
./start.sh --show-roots
# 3. Dry run — shows what would change, writes nothing
./start.sh --remap-path /mnt/media /srv/media
# 4. Translate (backs up every file first: *.before-path-remap-*)
./start.sh --remap-path /mnt/media /srv/media --apply
```

`--remap-path` can be given several times for several shares. Matching is
on path boundaries: `/mnt/nas` never touches `/mnt/nas2`. If the new path
is a symlink, the real path is stored — the server resolves its media
directory the same way. Running the command twice is harmless.

Started the server before translating? It then shows a red banner naming
the command, and the remap reports the files it indexed a second time
under the new path. `--drop-conflicting` deletes those new rows and keeps
the moved ones with their history; transcripts saved as JSON next to the
media are read in again when their folder is opened.

With Docker, run it in the container while the service is stopped:
`docker compose run --rm whisper-tm --remap-path /old /media`.

### Tables

| Table | Content |
|---------|--------|
| `media_files` | Filesystem cache + **complete media index**: duration, resolution, codecs, bitrate, has_audio/has_video, metadata title (cached once per FFprobe, `indexed_at`/`index_mtime` for resume) |
| `transcripts` | Transcript metadata: model, language, word/segment count, speakers, timings |
| `segments` | Individual segments with speaker, start/end time, and text |
| `segments_fts` | FTS5 full-text index on `segments.text` (automatic via trigger) |
| `summaries` | LLM summaries per file (DB primary source, JSON backup alongside) |
| `job_history` | History of last 10,000 transcription jobs with status, timings, errors |
| `incidents` | **Permanent memory of everything that went wrong**: every warning, error, OOM retry, system warning — with file identity (`abs_path`), step, and time. Never capped, survives reruns. Read: `GET /api/incidents?file=…&kind=…&days=…&q=…`, in file details under "Incidents", via MCP `incidents` |
| `skiplist` | Files to be skipped |
| `queue_items` | The transcription queue: each entry's file, settings, state and how often the server went down under it |
| `processing_steps` | Each transcript's step times (separation, Whisper, alignment, …), one row per step |
| `translations` | LLM translations per file and language |
| `llm_edits` | LLM correction runs: their suggestions, then the edits applied (the undo history) |
| `llm_queue` | The LLM queue: summaries, corrections, translations with their partial results |
| `conv_stats` | Conversation statistics per transcript: speech, words, sentences, questions, speaker changes, speakers, the longest monologue, the date in the file name — derived from the segments by every transcript write, counted again any time (see [Conversations](#conversations)) |
| `conv_speakers` | The same per transcript and diarization label: talk time, words, sentences, questions, turns, the longest turn |
| `conv_windows` | What each label said in every five minutes of a long diarized recording (JSON) — for the stretch where given people talk together, and a recording's *Who talked when* |
| `user_settings` | UI settings as key-value |
| `schema_info` | The schema version |

The other databases lie next to it: `speaker.db` (the voices), `rag.db`
(semantic search), `media.db` (Images & sounds), `audio.db` (Same audio),
`metrics.db` (load history) and `autoscan.db` (continuous scanning).

### Media Indexing

*📇 Index metadata* in the ⋯ folder menu (it shows how many files are missing)
starts it, with a progress chip in the path bar. Indexing reads the FFprobe metadata
(duration, resolution, codecs, etc.) **once** and stores it permanently in
`media_files` — afterwards duration & metadata are immediately available, without
recalculation on each directory open.

Features:

- **Resumable**: already indexed, unchanged files (checked via `index_mtime`) are skipped. An interrupted run resumes from there.
- **Transcript discovery**: JSON transcripts on the filesystem are discovered and imported into the DB (DB remains primary source).
- **Statistics**: index coverage, total media duration, and video count
  appear in the Library.

Endpoint: `GET /api/index` (SSE progress); the coverage comes with `/api/scan`
(its `index` field).
For RAG: `media_files` provides metadata per file (duration, codecs,
resolution) needed for filtering/"find entire file".

### Three Transcript Modes

| Mode | JSON File | DB | Description |
|-------|-----------|-----|-------------|
| `alongside` (default) | ✅ `.transcripts/` next to media | ✅ always | JSON next to media file; simple backup |
| `central` | ✅ Configurable path | ✅ always | Central JSON storage; good for read-only media |
| `database` | ❌ No file | ✅ always | DB only; fastest option for many files |

```yaml
directories:
  transcript_mode: "alongside"   # alongside | central | database
```

### When is what saved?

1. **App start**: If `database.sync_on_start = true` (default) → all JSON transcripts from `.transcripts/` are imported
2. **File scan** (`/api/scan`): Metadata of each media file is cached in `media_files`; the transcript status comes from the DB in one query, a JSON transcript is read (and imported) only where the DB has no row
3. **Transcription complete**: Write JSON file (except DB-only mode) + `upsert_transcript()` saves all metadata and segments in DB; FTS5 index updated via trigger
4. **Scan discovers `.transcripts/` folder**: auto-sync in background if `database.auto_add_dirs = true`

### Filesystem vs. DB

```
Scan: one DB query for the folder → a row?    ✅ use it
                                  → no row?   → a JSON transcript next to the
                                                media is read and imported
```

**The database is the primary source** (see the data policy above); the
JSON files are its portable copy and the way legacy transcripts come in.

### Full-Text Search (FTS5)

- Virtual table `segments_fts` on `segments.text`
- `unicode61` tokenizer with diacritic normalization (ä→a, é→e)
- Prefix matching: `word*` finds "wordsmith", "wording"
- Automatically maintained via SQL trigger (INSERT/UPDATE/DELETE)

### Configuration

```yaml
database:
  fts_enabled: true          # Enable full-text search
  sync_on_start: true        # Import JSON → DB on startup
  auto_add_dirs: true        # Auto-import discovered .transcripts/
  max_backups: 5             # Number of automatic backups
```

---

## RAG / Semantic Search

In addition to the SQLite full-text search, Whisper Transcription Manager can build a
**vector store** over the transcripts and offer **hybrid search** (semantic
similarity + full text, fused via Reciprocal Rank Fusion). The store is one
SQLite file (`rag.db`, next to `library.db`) — no extra server, nothing to
set up. The main database stays the source of truth; the vector store can be
deleted and rebuilt at any time.

There is deliberately no approximate vector index: one matrix product over
the whole store answers a query in about 100 ms at over half a million chunks,
which an index would first have to beat. The matrix stays in half precision in
memory (a few GB at that size); with `rag.enabled` on (it is off by default) it
is loaded in the background when the server starts, else by the first search.

### Finding situations

The search view's **Meaning** mode and the chat's `semantic_search` look for
*situations* — "Alice gets a compliment", "someone apologizes for being late" — not only
for similar wording:

1. **Candidates** by meaning (the closest passages), by wording (names and
   quoted words must appear) and, for a person of the voice library the query
   names, the passages that person speaks in.
2. **Copies** of a passage — the same recording in backup folders, or as
   `.mp4`/`.mkv`/`.webm` — fold into one hit.
3. **Speakers by name** where the transcript editor or the voice library
   knows them (`[Alice]` instead of `[SPEAKER_04]`) — looked up when the search
   runs, so a newly named voice shows at once.
4. **A reranker** (Qwen3-Reranker-4B through llama.cpp, started on demand on
   the GPU) reads the question and each of the best 120 passages together and
   orders them by relevance. Without a free GPU the list keeps its fused order
   and says so. It needs llama.cpp (`setup.sh --llama`) and the model file in
   `models/gguf/` (e.g. `Qwen3-Reranker-4B-Q4_K_M.gguf` from a GGUF repository
   on Hugging Face); without them the search runs without this step.

**🔎 Deep search** (next to the search field) goes through the whole
library: the language model plans the search (criteria, keywords), thousands
of candidates are ranked by the reranker, then the model checks the likely
passages one per request. Hits appear in the *Deep search* tab as they are
found; it runs in the LLM queue and goes on after a restart.

Measured on a real index against careful gold labels for two such queries:
the vector order alone had 4 of its best 10 right in both; with the reranker
10 and 6 of 10; of what the language model's check kept, 88–96 % was right.

```yaml
rerank:
  enabled: true
  local_model: models/gguf/Qwen3-Reranker-4B-Q4_K_M.gguf  # NVIDIA GPU, ~5 GB while it runs
  port: 8082
  idle_stop_sec: 300      # gives the card back after five idle minutes
  quick_candidates: 120   # passages the reranker orders per meaning search
```

### Usage

1. **Configure an embedding model** (Settings → AI models → Embedding model): a server of
   your own — llama.cpp with `--embedding`, Ollama (`nomic-embed-text`,
   `bge-m3`), OpenAI — or a llama.cpp server the app starts itself.
2. **Settings → Search → Semantic search**: "Embed new transcripts" builds the index
   (resumable, skips unchanged transcripts).
3. **Try a search** on the same page (*Hybrid (semantic + keyword)*, *Semantic only*,
   *Keyword only*).

```yaml
rag:
  enabled: true            # show the coverage in the file list and statistics
  path: ""                 # empty = rag.db next to library.db
  chunk_max_chars: 1200    # target chunk size
  rrf_k: 60                # hybrid fusion (Reciprocal Rank Fusion)
  top_k: 20
```

The embedding dimension is taken from the model on the first build.

### Storage

- **`chunks`** — chunk text, embedding (`float16` blob), speaker, start/end,
  source path
- **`chunks_fts`** — FTS5 full-text index over the chunk text
- **`indexed_transcripts`** — resume state per transcript
- **`rag_meta`** — model, dimension and fingerprint of the last build

### Export (Settings → Database → Exports)

- **App DB:** a consistent `.db` copy, made while it runs
- **Vector store:** `.db` download
- **Transcripts (portable):** JSONL (one transcript per line) or ZIP (each `.json`)

**The open folder's recordings:** *⬇ Export* in the folder bar opens the
export page on the open folder (see [The window](#the-window)) — "everything
downloaded into the uploads folder today" is: open the uploads folder, *⬇
Export*, *Added today*, a format, *Download*. The page counts them first.
(`POST /api/export/count`, `/api/export/preview`, `/api/export`; the
portable transcript export stays `GET /api/export/transcripts`.)

## Images & Sounds

The search view's tab **Images & sounds** finds moments by what is **seen or
heard** — "police at night", "a crowd", "gunshots", "rock music" — not by
what is said. Stills, short clips, clips with their sound, sound windows and
spoken text of the recordings lie in **one vector space**
([EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2),
740M parameters, Apache 2.0), so any of them asks all of them:

- **a text** in the search field — sounds are asked with the language model's
  short English label of it ("lots of people chatting in a crowded pub" →
  "busy pub crowd");
- **a hit** — 🔁 *Similar* searches with its own vector;
- **a file of your own** — 📎 an image, a sound or a video, optionally together
  with the text (one vector).

Each kind gets its own list (a query meets a still and a sound on different
scales); a hit plays its 10 s around the moment, opens in the player or in its
transcript, and shows the closest of 35 tags and what is said there.

**Not the semantic search.** Both are kept, for two questions:

| | Semantic search | Images & sounds |
|---|---|---|
| finds | what was **said** — the meaning of a passage | what is **seen or heard** at a moment, also where nobody talks |
| reads | the transcripts | the recordings: pictures and sound |
| a hit is | a passage of about a minute, with speaker names | a moment: a still every 20 s, 10 s of clip and of sound |
| model | the embedding model of AI models (text) | EmbeddingGemma 2: text, pictures, video, sound |
| used by | meaning hits, Ask, deep searches, Find, AI clients | the tab "Images & sounds", Find, `find_moments` |

*Spoken text* in Images & sounds is the words of a moment's 10 s, in the same
space as the pictures: it shows what is said where something is seen or heard,
and lets a picture or a sound find words. What was said, with its context, is
the semantic search's — the list says so and takes the query there. Weighing
the sound list by the spoken words made it worse (mean AP 0.251 → 0.243: 80 %
of the sound windows have speech, lyrics included), so the kinds stay apart.

**With a language model that sees pictures** (llama.cpp with a vision
projector, e.g. Qwen3.6): *🔎 Check with the language model* judges the best
stills one per request and sorts them by its verdict; *📝 Describe* writes a
sentence about a still, and a describing run (LLM queue) does it for all —
the descriptions become a list of their own.

Measured on a large archive (a 12 GB NVIDIA card, the queue running):

| | |
|---|---|
| Pictures, 12 queries × best 20, judged by eye — precision of the top 10 | by the picture 0.62 · by the description 0.79 · checked by the model 0.82 |
| nDCG@10 of the same | 0.67 · 0.91 · 0.96 |
| Sounds, 16 labeled kinds — mean AP | query in the library's language 0.115 · the model's English label 0.325 |
| The same by numbers per vector | 768: 0.251 · 512: 0.189 · 256: 0.201 · 128: 0.104 |
| Indexing speed on the GPU | stills 13/s · 4-frame clips 5.5/s · sounds 36/s |
| Frames per clip, cosine to 16 frames | 8: 0.99 · 4: 0.98 · 2: 0.965 |
| GPU time of a video moment, all kinds | clips with sound ~43 % · clips ~35 % · stills ~15 % · sounds ~5 % · spoken text ~1 % |

### Setup

The model needs newer libraries (transformers 5.19, sentence-transformers 6.1)
than the app's own environment, so it runs in **an environment of its own**
as a separate process on `127.0.0.1:8083`:

1. **Settings → Images & sounds → Install** — creates `media-env` in the data
   folder (with [uv](https://docs.astral.sh/uv/), packages linked from its
   cache: several GB with PyTorch's CUDA libraries) and downloads the model
   (1.5 GB).
2. **Choose the folders** and look at the estimate — GPU time and disk; a
   large archive takes weeks of GPU time. The whole library is the default
   scope — its transcribed recordings: a recording without a transcript is
   indexed once it has one (the spoken text is one of the kinds).
3. **🖼 Index** — newest recordings first, resumable, with a progress bar like
   the semantic search's. The run is a guest on the card the queue works on:
   it takes a stage of the GPU gate (never next to Whisper; next to other
   steps only with `vram_need_mb` free) and gives the card back when the queue
   waits. It waits while the disk has less than `min_free_gb` free. Moments
   that look like the one before (a webcam over hours) get no still and no
   clip of their own. A kind switched on later is only added to the moments
   that are there.

Queries run on the CPU (a text ~0.1 s, an image ~3.5 s, a video ~10 s); the
service stops after ten idle minutes.

```yaml
media_search:
  kinds: [image, video, videoaudio, audio, speech]   # all by default
  video_interval_sec: 20         # a moment every 20 s of a video
  clip_frames: 0                 # 0 = 8 up to 500 h of video in scope, else 4 (kept)
  dedupe_similarity: 0.97        # repeats get no still/clip of their own
  dims: 768                      # stored as 8-bit numbers: 768 bytes a vector
  folders: []                    # empty = the whole library
  index_while_queue_runs: true   # share the GPU through the gate
  min_free_gb: 20                # a run waits below this
  thumbs_max_mb: 4000            # beyond: stills are cut when shown
  llm_translate_sounds: true
  llm_check_count: 24
  llm_language: "English"        # of the descriptions
```

### Storage

- **`media.db`** next to `library.db` — `files` (progress, settings and the
  kinds each recording has), `moments` (time, sound window, thumbnail, what is
  said), `items` (a vector per moment and kind: all 768 numbers as 8-bit
  integers — rounding moves a cosine by ~0.001), `descriptions`
- **RAM** — every vector as stored (8-bit, packed for fbgemm, ~0.8 KB at
  768): a search scores all of them (26 ms for a million) and its best again
  in float32. The estimate on the settings page says what the whole scope
  will take. A first pass on fewer numbers was measured and dropped: the true
  top 10 sat as deep as 2–3 % of the collection when ranked by the first 128
  numbers, too deep for any shortlist at millions of vectors
- **`~/.cache/whisper-tm/media/thumbs`** (the cache folder;
  `media_search.cache_dir` moves it) — the stills (~22 KB each), a folder per
  recording, up to `thumbs_max_mb`
- **`~/.cache/whisper-tm/media/clips`** — preview clips, cut when played, kept up to
  `cache_max_mb`

---

## Same Audio

The search view's tab **Same audio** finds every recording that holds a
snippet — the same sound, also re-encoded, quieter or under noise — at the
exact second: an intro, a jingle, a clip someone cut out of a longer
recording. What only *sounds alike* (laughter, applause, a dog barking) is
[Images & sounds](#images--sounds)' job. The snippet is

- **a stretch of a recording** — set IN and OUT on the timeline of the
  recording's *Clips* tab (1 s to 5 min) and press **≋ Same audio**;
- **a file of your own** — 📎 a sound or a video (its first five minutes), or
  dropped on the tab.

Each hit says where the snippet begins in it — and every place, when a
recording holds it more than once —, plays that place, opens in the player or
in its transcript, and shows the recording's loudness. An analyzed recording
shows its loudness and *≋ n copies* above its transcript; a click lists them.

**Duplicates** come without a search: while a recording is analyzed, its
sound is looked up in what is analyzed already. The tab opens on them — or
say it: *find duplicate audio files* or *duplicates in Archive* (in English
or German) in the top bar or in Find goes there — Find's language model also
recognizes other phrasings, such as *which videos do I have twice?* — and so
does Ask (its tool `find_duplicates`, with a button that shows them).
Two views, drawn:

- **The same recording** — the same sound from start to end: another format,
  folder or bit rate. On top, how many recordings exist more than once and
  the space their extra copies take (every copy but the largest of each kind
  — a video and an mp3 of its sound both count as kept). Below, **where the
  copies are**: pairs of folders that hold the same recordings, each with a bar
  of how much of it is also in the other — a backup or a copied folder shows
  as two full bars, with how many subfolders share its layout. A click on a
  pair lists its recordings. Each group draws every copy as its waveform on
  one timeline: the same shapes one above the other are the proof; a line
  follows the pointer across all of them; a click plays that copy from there
  and offers the same moment in the others.
- **Contained in another** — a recording that holds others (a compilation, a
  stream the clips were cut from) as its waveform with a colored block for
  each part where it begins; a click on a block plays it there.

Both can be narrowed to a folder or a name and a shortest length (short sound
effects left out), and **⚙ Filters** narrow further: the person speaking (the
voice map's names), video or sound only, the format, the longest length, the
recording date (the date in the file name — a copy's file date is often only
when it was copied — else the file's own), the transcript's language,
transcribed or not, the number of copies, copies across folders or inside one.
For recordings holding others, the holding one and the parts are chosen each
by their folder: with *holding one in* `vods` and *parts in* `clips`, every
stream shows where in it each of the clips was. The lists are ordered by
space, copies, length or when they were found.

Find and the meaning search fold copies by the same relations first: the
same recording is folded as *the same sound*, a moment of a clip and the same
moment inside the longer recording as *the same passage (its sound)*.

**How it works.** Audio fingerprints as Shazam has them: the strongest
spectral peaks of every second, each paired with the next ones; a pair's two
frequencies and time step make a hash. A snippet's hashes found in a
recording at the same time offset mean it is there — chance spreads over all
offsets. Measured with 14 recordings that exist twice in other formats
(mkv, mp4, webm, opus, mp3), snippets of one copy searched in an index of
about 27 hours that holds the other — found in:

| The snippet | 3 s | 5 s | 10 s |
|---|---|---|---|
| another encode of the same recording | 100 % | 100 % | 100 % |
| … and then mp3 at 32 kbit/s | 100 % | 100 % | 100 % |
| … 20 dB quieter | 90 % | 93 % | 100 % |
| … under noise as loud as itself | 69 % | 90 % | 93 % |
| … with another recording mixed in as loud | 43 % | 81 % | 90 % |

A match needs 10 hashes in step; snippets played backwards — which can't be
in the index — scored 7 at most (one recording that plays its own beginning
backwards at its end was found there, rightly). A search takes ~15 ms. Not
found: copies played faster, slower or pitched (the hashes are frequencies
and time steps); a snippet re-recorded through a microphone is a matter of
luck.

**Analyzed once.** While a recording is transcribed, the analysis reads the
16 kHz sound the transcription has just decoded (~4 s of one core an hour of
sound, on a thread of its own) — the recording isn't read again. Recordings
transcribed before, or never, need one run: **Settings → Audio analysis → ≋
Analyze the library** reads every recording the library scan found, newest
first, at the lowest CPU priority, `parallel_files` at a time and at most
`max_read_mb_s`; it can be stopped and goes on where it stopped, also after a
restart. Reading the files is what takes the time — a large library on a
network share is read once, at 100 MB/s 360 GB an hour.

**Waveform and loudness.** The same pass keeps the waveform (the clip
editor's timeline then decodes nothing) and the loudness of every recording:
integrated loudness and loudness range (EBU R128 / ITU-R BS.1770-4), sample
peak, the share near silence and the share clipped.

```yaml
audio_analysis:
  enabled: true
  during_transcription: true   # from the WAV the transcription decodes anyway
  folders: []                  # the background run: empty = the whole library
  auto_run: false              # start it by itself (and when new recordings come)
  parallel_files: 2
  max_read_mb_s: 0             # 0 = no limit
  pause_for_queue: false
  min_free_gb: 5
```

### Storage

- **`audio.db`** next to `library.db` — `files` (one row per recording: the
  size and modification time it had, its loudness, how many hashes),
  `waveforms` (20 values a second), `relations` (the same / contains, with
  the offset)
- **`audio.db.fp/`** — the fingerprints, 48 bits an entry in segments
  written once, sorted by hash and read through mmap; new ones collect in
  memory (searched at once) and are written every ~110 hours of sound or ten
  minutes; segments of one size are merged in the background
- About **0.3 MB an hour of sound** in all; `--remap-path` moves its paths
  with the rest

---

## Conversations

The search view's tab **Conversations** finds conversations by who took part
and how they talked — an hour of Alice, Bob and Carol in long sentences, a
quick back-and-forth, an interview, the longest monologues. In a recording
of several hours it finds the stretch where the people talk together.

Every transcript is counted from its segments alone, with no model, by every
write of it (transcription, edits, speaker names, corrections, imports), so
the numbers are never older than the text:

- **Sentences** end at `.` `!` `?` `…` inside one speaker's turn, not where a
  segment ends: with word timestamps segments are subtitle cues of a few
  words, and Whisper's own segments end mid-sentence too. A transcript with
  fewer than one sentence end in 60 words gets no sentence length — some
  languages and models write no punctuation, and one endless "sentence" says
  nothing.
- **Words** are runs of letters of every script; languages written without
  spaces (Chinese, Japanese, Thai …) count letters.
- **Turns** are a speaker's consecutive segments (a pause of 20 s ends one);
  a **speaker change** is a turn by another speaker than the one before.
  Speaker changes per minute and the average turn (speech per change) need a
  minute of diarized speech.
- **Speakers** are the diarization labels with at least 5 % of the diarized
  speech — a label that said one word in an hour is diarization noise, not a
  participant.
- **Questions** are the sentences that end in a question mark. An
  **interview** has two speakers, one of them asking: 35 % or more of their
  sentences are questions, and they speak at most 35 % of the time.
- Phrases Whisper is known to invent over noise ("Thanks for watching" over
  a cough) are left out.

**Longer than usual.** Words per sentence, speaker changes per minute and the
average turn are compared with the library's own conversations in the same
language — punctuated recordings with two speakers or more, no voice holding
85 % of the speech, at least 20 sentences and two minutes of diarized speech.
Each of the three comes with its percentile: *Sentences of 11.4 words —
longer than 78 % of conversations in this language.* Sentences *longer than
usual* reach at least the 60th percentile, *shorter than usual* at most the
40th. A language with fewer than 30 such conversations is compared with the
whole library, a library with fewer with generic figures of spoken language.

With people chosen, *longer* and *shorter than usual* mean than usual for
**them**: the deciles of their own conversations (all together, or each one
of them takes part in), at least ten of them. People talk in their own way —
a round of friends who always talk quickly has sentences of five words where
a lecture has fifteen —, and measured against the library such a round
would never have a conversation in longer sentences: the hour where they
talked something through at length would not be found. The info line says
what was measured against (*Longer than usual for them: from 5.2 words per
sentence*), each row says where it lies for them, and the library's
percentile moves into its tip. With fewer than ten of their conversations
the library is the measure, and the line says so. The kinds stay measured
against the library — a kind says what a conversation is, whoever asks.

**Kinds.** Each conversation gets one — the first that fits — and says why
in a sentence or two:

| Kind | When |
|---|---|
| *No speech* | no words |
| *Speakers unknown* | not diarized: who spoke is unknown |
| *Monologue* | one speaker, or one voice speaks 85 % of the time or more |
| *Interview* | two speakers, one of them asking (above) |
| *Detailed* | long sentences and long turns: words per sentence at least the 60th percentile, the average turn at least the median |
| *Back-and-forth* | quick changes: speaker changes per minute at least the 60th percentile, sentences no longer than the median |
| *Conversation* | the rest |

Three speakers or more add the tag *Group*.

**People.** Names are never stored with the numbers: a label (`SPEAKER_00`)
means something only inside one recording, and who it is can change any time
in the voice library. They are looked up when a search runs, as in the other
searches — the names typed in the transcript first, then the voice map's
majority —, so a voice named a minute ago counts at once. A person takes
part in a recording with at least `conversations.min_share` of its diarized
speech (5 %) and 5 s; one person on several labels has their talk added
together.

**Stretches.** In a stream of several hours the people come and go, and the
recording as a whole says little about any hour of it. So a diarized
recording longer than ten minutes also keeps, for every five minutes, how
much each label said. Asked for people (*All together*, the default), the
tab finds the windows where all of them talk — each at least 15 s and
`min_share` of the window's speech. One window without them between two with
them doesn't end a stretch, and a stretch is at least two windows long (one
in a recording under a quarter of an hour). Every stretch is a result of its
own (up to five per recording, the longest first) with its own numbers, and
plays from its start; one that covers 90 % of the recording is the whole
recording. *Any of them* lists whole recordings where at least one of them
takes part. Speakers and speaker changes are counted by label when a
transcript is counted — before anyone has a name —, so where diarization
split one person into two labels, they count as two speakers, and a change
between the two as a speaker change.

Copies — one recording kept in several formats or folders — are listed once,
with *+n copies*: found by what they hold (length, speakers, words), never
by their names; the copy with more named people stays. The date is the one
in the file name, else the file's own (a copy's file date is often only when
it was copied).

**The tab.** People (the voice library's names are suggested; with two or
more, *All together* or *Any of them*), the length (*Under 5 min*,
*5–30 min*, *30–90 min*, *About 1 h* = 45–75 min, *Over 2 h*), the kind, the
sentences (*Longer than usual*, *Shorter than usual*) and **⚙ Filters**:
minutes, words per sentence, speaker changes per minute, speakers (at least
two by default), questions, language, text in the path, folder, dates,
copies. Each row shows the kind (its reasons on hover), the file, the
stretch (*1:15:00–2:18:00 of 6:00:00*) or the recording's length, the people
with their shares (and how many voices have no name), words per sentence
with its percentile, speaker changes per minute, questions, the longest
monologue (a click jumps there) and the date; ▶ plays it, *open ›* opens the
transcript there. **✂ Extract** opens Voices → Extract with these filters
(and the first person chosen): one voice's sentences from just these
conversations, as a dataset or one audio file. The list is ordered by length, sentence length, speaker
changes, questions, the longest monologue or the date — with people also by
how much they talk together. With one person in the filter, a card above the
list shows their talk time, recordings, conversations and words per
sentence, the people they talk with most (a click adds one) and the kinds of
their conversations.

The same numbers elsewhere:

- **A recording's Statistics tab** — the card *Conversation*: the recording's kind and
  why, words per sentence and speaker changes against the library, the
  average turn, questions, the longest monologue, the people with their
  shares, *Who talked when* (each part a jump) and *More conversations with
  these people*.
- **Voices → People** — *Conversations* next to a person shows theirs.
- **Voices → Extract** — *From conversations* shows the filters ✂ handed over
  in words; *Change …* goes back to the tab, *✕ Every recording* drops them.
- **Ask** and MCP clients — `find_conversations` (the tab's filters in
  words — with people, the kind *detailed* is read as their own measure,
  longer sentences than usual for them; Ask adds a button *Show them in
  Conversations*),
  `conversation_stats` (one recording: its kind, the people, who talked
  when) and `person_stats` (how a person talks, and with whom).
- **Find** hands such requests over: *a long talk with Alice and Bob* or
  *conversations where Carol speaks in long sentences*, typed in Find or the
  top bar's search field, open this tab with those filters instead of a search for a moment.
  Find's language model recognizes them; without one, Find searches as usual.
  With people named, *a detailed conversation* is one in longer sentences
  than usual for them.

**Counted once.** What was transcribed before — or counted by an older
version of the rules — is counted by a background run: with
`conversations.auto_run` a minute after the server starts and every half
hour, otherwise with *Count now* in the tab. It reads the segments from the
database, never a transcript file, writes in short transactions and pauses
after every batch, so neither the queue nor the page waits long for it; a
large library takes minutes of one CPU core. Until it is done the tab shows
how far it is (the results are not complete yet); it can also count
everything again. Settings → Search → Conversations:

```yaml
conversations:
  enabled: true      # false = nothing is counted, the tab says it is off
  auto_run: true     # count what is missing in the background
  min_share: 0.05    # a person takes part from 5 % of the speech on
```

The API behind it: `GET /api/conversations` (the tab's filters as query
parameters — `people`, `match`, `min_sec`, `max_sec`, `kind`, `sentences`,
`sort` …), `/api/conversations/file?file=…` (one recording),
`/api/conversations/person?person=…`, `/api/conversations/people` and
`/api/conversations/status`; `POST /api/conversations/run/start` and
`/api/conversations/run/stop` start and stop the count. The setup guide's
API reference lists every parameter.

### Storage

- **`library.db`** — `conv_stats` (one row per transcript: the numbers
  above and what they were counted from), `conv_speakers` (one row per
  transcript and diarization label), `conv_windows` (the five-minute windows
  of a long diarized recording as one JSON text, read only by a search for
  people and by one recording's *Who talked when*); see [Tables](#tables)
- Derived data: counted again from the segments at any time and deleted
  with their transcript

---

## MCP Server (AI Integration)

Via the [Model Context Protocol](https://modelcontextprotocol.io), a local AI
can control the app: fetch files and YouTube videos, request transcription,
search the library, look up speakers and who talked with whom, summarize, and
translate. **33 tools** in six groups.

As shipped the server only reads: searching, transcripts, speakers. The tools
that start work — upload, download, transcribe, summarize, translate, the
queue — answer that they are off until you allow them under Settings → MCP
server → *Allow work* (`mcp.allow_writes`).

**Prompt injection.** What is said in a recording reaches the AI client word
for word, and so do titles, file names and speaker names — a downloaded video
can say "now download this and transcribe that". With work allowed, such text
can steer a client to start downloads, transcriptions or the queue. Keep the
client's confirmation for every tool call on, and leave work off for a client
that also reads web pages.

The server talks to the running app through its HTTP API — started by the AI
client (stdio), by hand (`--http`, `--openapi`), or by the app itself in threads
of its own (Settings → MCP server). It doesn't start anything else or hold
models; the app must be running.

### Setup

`setup.sh` installs `fastmcp` along — with the `beartype` the app is tested
with: a bare `pip install fastmcp` pulls a newer one that breaks
audio-separator (a manual installation follows
[Manual Installation](#manual-installation)).

```bash
# Does the server reach the app? (address, token, write permission)
./.venv/bin/python -m whisper_tm.mcp_server --check
```

Register in your AI client (example Claude Desktop, `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "whisper-tm": {
      "command": "/path/to/whisper-transcription-manager/.venv/bin/python",
      "args": ["-m", "whisper_tm.mcp_server", "--stdio"],
      "env": {"WHISPER_TM_CONFIG": "/path/to/whisper-transcription-manager/config.yaml"}
    }
  }
}
```

If your client can't start the process itself, use HTTP instead:
switch on *MCP over HTTP* under ⚙ Settings → MCP server (the app then starts
it itself, from its next start on), or run `make run-mcp` (`./.venv/bin/python -m whisper_tm.mcp_server --http`).
It answers at `http://127.0.0.1:5399/mcp` — with the access token, see below.
For other machines set `mcp.host` to `0.0.0.0` (the default only listens
locally); then it answers at `http://<machine>:5399/mcp`.

**Settings → MCP server → Connect a client** shows these set-ups ready to
paste, with this machine's address, paths and token filled in: Claude Code,
a JSON configuration (Cursor, LM Studio, `.mcp.json`), Open WebUI and stdio.

### Access token

Both network services — MCP over HTTP and OpenAPI — require an access token
by default (`mcp.require_token: true`). Clients send it as
`Authorization: Bearer <token>` (or `X-API-Token`); without it every request
gets 401. The token is generated on the first start and kept in the file
`mcp_token` next to `config.yaml`, readable only by its owner — like the
session key, it doesn't belong in a config file that may be shared. The
settings page shows it, copies it and makes a new one (the old one stops
working at once; an MCP server started by hand keeps it until it restarts) —
in a browser on the server itself, or anywhere with the
login switched on; from another machine without a login it stays hidden.
`mcp.token` in `config.yaml` sets one of your own. The stdio path needs no
token: there the client starts the server itself.

### Clients without MCP (Open WebUI): OpenAPI

Open WebUI and some others integrate tools **only** as "OpenAPI Tool
Server" — they fetch a specification from `/openapi.json` and call the
endpoints described there. No `mcp` proxy needed:

```bash
make run-openapi        # ./.venv/bin/python -m whisper_tm.mcp_server --openapi
```

Then in Open WebUI under *Settings → Tools* add as server:

```
http://<machine>:5400
```

Same 33 tools, same limits (`allow_writes`,
`max_files_per_call`). The specification is generated at runtime from the
tool list — there's no separate list that could diverge.
Each tool is a `POST /<name>` with JSON body.

The service requires the access token like MCP over HTTP (see above) and
advertises it in the spec — enter it in Open WebUI as the key of the tool
server, authentication "Bearer".

**Time limits:** `transcribe` with `wait=true` keeps the HTTP request open until
the transcript is ready — for archive files, minutes to hours. Most clients time out before then. Via OpenAPI, better to use `wait=false` and then
call `job_status` afterwards.

### The Tools

Before 1.0 the tools and their arguments may change with a version: a client
should read the tool list again after an update (most do on every connect).

Tool names are **in English** — because a language model sits on the other
end, and models more reliably choose based on English names and
descriptions.

| Group | Tools |
|---|---|
| Ingest | `upload_file`, `download_youtube`, `transcribe`, `transcribe_many` |
| Monitor | `job_status`, `queue_status`, `control_queue`, `incidents` (what went wrong, when, for which file — permanent) |
| Read | `transcript` (text/srt/vtt/json/segments, with time window), `list_files`, `file_info`, `library_stats`, `clip` (audio excerpt to listen) |
| Search | `search`, `search_distribution` (where hits concentrate), `semantic_search`, `find_moments` (a moment described in words — every search at once, checked by the language model; about a minute), `find_duplicates` (recordings that exist more than once or contain another, by their sound), `rag_status`, `search_plan` (divide hits across helpers) |
| Speakers | `who_speaks`, `where_person_speaks`, `list_people`, `speaker_stats`, `find_conversations` (conversations — or the stretch of a long recording — by who takes part and how they talk), `conversation_stats` (one recording: its kind, the people, who talked when), `person_stats` (how a person talks, and with whom) |
| Post-process | `summarize`, `translate`, `proofread`, `llm_result`, `existing_summaries`, `existing_translation` |

Plus three resources: `whisperx://library/stats`, `whisperx://queue`,
`whisperx://transcript/{file*}`.

`upload_file` takes a path only for an audio or video file below the allowed
folders (`security.allowed_roots`) — over stdio too, since text in a
transcript can steer the assistant that calls it; any other file goes as
`content_base64`. Every upload must hold audio or video (ffprobe checks it),
whatever its name says.

### Audio Clips: Listen to the Find, Don't Believe

A quote from a transcript is a claim. `clip(file, start_sec,
end_sec)` cuts the location as mp3 (audio only, mono, 96 kbit/s — one
minute weighs around 700 kB) and returns **two things**:

* the audio as MCP content of type `audio` — clients that know this (Open
  WebUI since 0.6.31) attach a playable file to the result;
* a **three-line** snippet that becomes a player in chat:

  ```
  <audio>
  http://…/api/clip/audio?file=…&start=…&end=…
  </audio>
  ```

The snippet is the more important part, because it sits **in the middle of the answer,
right next to the quote** — thirteen matches get thirteen players.
The URL appearing in the tag's *content* rather than in `src` is no quirk: Open WebUI reads it exactly there (`HTMLToken.svelte`).

**Why three lines?** `audio` is not one of the block elements that
CommonMark recognizes. An HTML block is created only via rule 7: a
complete tag that stands ALONE on its line. Verified with `marked`,
the same parser Open WebUI uses:

| written as | Token | Player |
|---|---|---|
| `* <audio>URL</audio>` | list | no |
| `<audio>URL</audio>` (own paragraph) | paragraph | no |
| `<audio>` / URL / `</audio>` (three lines) | **html** | **yes** |

In the first two cases, the chat shows raw text with `&amp;` in it —
the `&amp;` is the proof, because the player converts it back, the
text output doesn't. That's why the rule appears in every tool description,
and why there's also `play_url`: the same URL bare, for a
regular link that works anywhere — even in a list.

`search`, `semantic_search`, and `where_person_speaks` know the
`with_audio` flag. It adds `player` and
`play_url` fields for each hit — nothing is cut yet. Only when someone clicks play does it trigger
the cut (`/api/clip/audio`, then it sits in the clip folder). A
hit list costs no ffmpeg runs for locations nobody listens to.

If players point to nothing, `mcp.listen_base_url` is wrong: the
MCP server reaches the app via `127.0.0.1`, but a chat on a different
machine doesn't. Left empty, the loopback address is automatically
swapped for this machine's network address.

### Dividing Large Hit Sets

If a search finds more than a model can read at once, `search_plan(text, per_batch=50)` divides it into batches — one per helper. For example:

```
search_plan("keyword")   →  3,000 hits in 1,500 files
                            60 batches, all 3,000 covered   (0.5 s)
```

Each batch comes with **ready-made calls**; a helper gets one batch
and sends them. The model doesn't need to count — it's easy to miscount.

**Division is by file, not by sequence number.** Two reasons:

* *Offset drifts.* Sorting is by `rel_path, start_sec`, and
  the queue keeps writing. If during the run a file arrives that comes earlier
  alphabetically, every later page shifts — helper 7 would see some of the same as helper 6.
* *Context beats even distribution.* "This file, these 47 hits" is assessable; "hits 300–350 across eight
  files" is confetti.

Offset is still available (`search(offset=…)`) — it's needed only WITHIN a file with more hits than one batch holds, and
it can't drift there. To prevent batch overlap, the planned calls set `exact_file: true`; otherwise `file` is a
substring filter.

Batches come **densest file first**. If the plan doesn't cover everything (facets go up to 5,000 files), the
`incomplete` field says how many hits are left out — a plan that fakes completeness is worse than none.

### Wait or Not

Long-running tools (`transcribe`, `download_youtube`, `summarize`, `translate`,
`proofread`) know `wait`. With `wait=true`, the server polls and returns the
finished result; if its time limit expires (`mcp.time_limit_min` for
`transcribe`, a `timeout_min` of their own for the others), it returns the job
ID with a note that the work **continues** — nothing is cancelled. Use
`job_status(job_id)` to check later.

### Settings (⚙ → MCP server)

| Key | Default | Effect |
|---|---|---|
| `mcp.autostart_http` / `mcp.autostart_openapi` | `false` | start the network services together with the app |
| `mcp.require_token` | `true` | the network services answer only with the access token (off only while `mcp.host` is `127.0.0.1`) |
| `mcp.token` | empty = generated | a token of your own instead of the generated one (config.yaml only) |
| `mcp.base_url` | empty = this app | only needed if MCP server runs on a different machine |
| `mcp.host` / `mcp.port` / `mcp.openapi_port` | `127.0.0.1` / `5399` / `5400` | address and ports of the network services |
| `mcp.allow_writes` | `false` | `true` = the tools that start work are on too: upload, download, transcribe, AI tasks, the queue (`--read-only` switches them off for one run) |
| `mcp.max_files_per_call` | `50` | ceiling for `transcribe_many` — more is **rejected**, not truncated |
| `mcp.time_limit_min` | `30` | how long `transcribe` waits, then returns the job ID |
| `mcp.api_token` | empty | only needed if `auth.enabled` is on |
| `mcp.listen_base_url` | empty = derived | address where a **browser** reaches the app — for players in responses |

**File references are paths relative to the media root** (`uploads/video.mp4`) — and
a recording outside it, which a search across every root finds, is named by its full
path; the tools take it back as it is. Full paths outside `security.allowed_roots` get
403 from the API (see `docs/PATHS.md`).

Intentionally **not** offered: directory changes, database maintenance, or
settings write access. An AI that changes the media root pulls the floor out from under all
other tools at once.

---

## Architecture

```
whisper-transcription-manager/
├── setup.sh                  # Setup (Linux, WSL2)
├── config.default.yaml       # Every setting, with its reasoning
├── Makefile                  # Dev shortcuts (`make help`)
├── browser-extension/        # Chrome/Edge extension (browser-extension/README.md)
├── tests/                    # pytest suite (`make test`), JS harnesses in tests/js/
├── tools/                    # Maintenance scripts (see tools/README.md)
├── docs/
│   ├── OPERATING.md          # Updating, data, a systemd unit, first-run failures
│   ├── PATHS.md              # Path model — read this before touching paths
│   └── LLM-PIPELINE.md       # Summary, correction, translation
└── whisper_tm/
    ├── __main__.py           # CLI entry point
    ├── app.py                # Flask factory
    ├── config.py             # YAML config manager
    ├── extensions.py         # Singleton registry (no circular imports)
    ├── mcp_server.py         # MCP server for a local AI (also behind Ask)
    ├── llm_server.py         # start-llm.sh: the local llama.cpp server
    ├── database/             # SQLite + FTS5 (library.db) and the voice store (speaker.db)
    ├── hardware/             # GPU/CPU detection
    ├── jobs/                 # Job lifecycle + TTL cleanup
    ├── scanner/              # Filesystem walk, indexing, watcher
    ├── rag/                  # Vector store (rag.db) + hybrid search, deep search
    ├── find/                 # Find: one description, every search at once
    ├── media/                # Images & sounds: index (media.db), search, the service
    ├── conversations/        # Conversations: statistics per transcript, the queries
    ├── core/
    │   ├── paths.py          # Path normalisation — the only place for it
    │   ├── transcription.py  # WhisperX pipeline
    │   ├── queue.py          # Persistent transcription queue
    │   ├── vocal_isolation.py
    │   ├── speaker*.py       # Voice vectors, the voice map (exact kNN, grouping), assignment
    │   ├── llm/              # Chunking, correction, translation, prompts
    │   ├── audio_utils.py    # FFmpeg (cross-platform)
    │   ├── hf_access.py      # The Hugging Face token for speaker diarization
    │   └── security.py       # Path traversal protection, the origin check
    ├── api/                  # Flask blueprints
    └── static/
        ├── index.html        # Template
        ├── css/              # app.css, settings.css, searchview.css, …
        └── js/               # Plain scripts, no build step
```

## Hardware Support

| Platform | GPU | Status |
|----------|-----|--------|
| Linux | NVIDIA (CUDA) | ✅ Full support — tested on 4, 8 and 12 GB cards (Turing, Ampere) |
| Linux | AMD (ROCm) | ⚠️ Experimental, untested — PyTorch's ROCm build; CTranslate2 must be built with HIP (the setup guide's AMD page) |
| Windows | NVIDIA (CUDA) | ⚠️ Through Docker Desktop or WSL2 — not yet tested on a real machine. No native Windows version |
| Any | CPU | ✅ Works, many times slower (the app then starts with Whisper turbo and without vocal isolation) |

**Several NVIDIA cards.** Start the app with `CUDA_VISIBLE_DEVICES=<n>
CUDA_DEVICE_ORDER=PCI_BUS_ID ./start.sh` to give it one card (the number as
`nvidia-smi` shows it); `hardware.gpu_index` moves Whisper alone, not the other
steps. Not tested on a machine with two NVIDIA cards.

Tested on x86_64. arm64 (ARM NAS boxes, Raspberry Pi)
is untested, natively and in Docker: the locked dependencies are resolved for
x86_64.

**What the app chooses by itself.** Three settings have no value that suits
every machine, so the app sets them once at start from the card it finds —
unless you have set them yourself (Settings, or config.yaml), which always
wins. The system check and Settings → About say what was chosen.

| Machine | Model | Vocal isolation | Voice vectors | Files at once | Models in memory |
|---------|-------|-----------------|---------------|---------------|------------------|
| No GPU | turbo | off | off | 1 | — |
| Card with up to 4 GB | turbo | off | off | 1 | loaded per step |
| 6 GB card | turbo | off | on | 1 | loaded per step |
| 8 GB card | large-v3 | on | on | 1 | loaded per step |
| 10 or 11 GB card | large-v3 | on | on | 1 | loaded per step |
| 12 GB and more | large-v3 | on | on | 3 | kept loaded |

*Voice vectors* are what finds a person again in other recordings. They are
made from separated sound — the whole recording's (vocal isolation) or only
its 30-second windows — and the separation model does not fit a 4 GB card and
takes many times a recording's length on a CPU. Diarization (who speaks when
within one recording) works on every machine. To have voice vectors there
all the same: Settings → Speakers → *Separate windows for voice vectors*.

**Card generations.** RTX 20xx to 50xx and GTX 16xx run PyTorch's CUDA 12.8
build; GTX 10xx and GTX 9xx its CUDA 12.6 build, which the setup installs by
itself for such a card (native install only — the Docker image holds the 12.8
build). The system check says when the installed build and the card don't
fit. Tested on Turing (an 8 GB card) and Ampere (a 4 GB laptop card, a 12 GB card);
a card older than Turing is not tested yet.

**Measured** (16 Nov 2026, fresh installs, an 11-minute recording with two
speakers and diarization, the first run including the one-time model
downloads):

| Machine | Profile | Time | Peak VRAM |
|---------|---------|------|-----------|
| 4 GB laptop card (Ampere) | turbo, no isolation | 2 min 42 s | 2.3 GB |
| 8 GB card (Turing) | large-v3, vocal isolation, voice vectors | 2 min 31 s | 5.5 GB |
| 8 GB card, in Docker | the same | 7 min 47 s (5.2 min of it the large-v3 download) | 5.5 GB |
| No GPU (22 threads) | turbo, no isolation | 12 min 39 s | — |

## Known limitations

A beta: these are known and open (the issue tracker has the details).

- **Platforms** — run on Linux with an NVIDIA card. Windows works only through
  Docker Desktop or WSL2 and is not tested on a real machine; AMD
  cards are experimental and untested; without a GPU everything is many times
  slower (see [Hardware Support](#hardware-support)).
- **One user** — the login is one password for everybody; there are no
  accounts or roles. Whoever reaches the page can do everything in it.
- **The root path only** — behind a reverse proxy the app must sit at `/`,
  not under a sub-path.
- **Video playback** — a video whose codecs the browser cannot decode (for
  example MPEG-4 Part 2 with AMR sound) does not play in the page. Its
  transcript, exports and clips are not affected.
- **The open folder is the library** — the folder open when settings are
  saved is where the app starts next time; a subfolder then stands for the
  library until you open the library again in the path field.
- **AI jobs after a change of library** — a correction or translation that
  still waits when another library folder is opened looks for its transcript
  there and ends with "transcript not found".
- **Out of memory on a 12 GB card** — when large-v3 loads beside the separation
  model the card can run full. No recording is lost, but that one starts
  again from its first step.
- **Translation and correction** show finished parts only, not the text as it
  is written (the summary does).
- **Keyword search in Chinese, Japanese and Thai** finds a clause only as a
  whole, not a word inside it; the search by meaning is not affected.
- **PDF export** uses a Latin-1 font: other scripts come out as question
  marks — Word and the text formats keep them.
- **No marks of your own** — a moment in a recording cannot be bookmarked or
  annotated yet (cuts in the Clips tab are the nearest thing).
- **Whisper's own repetition guard is off** — WhisperX's batched path has no
  temperature fallback, so repeated or invented lines are filtered after the
  fact (Settings → Transcription).

## Prerequisites

- **Python 3.11–3.13 with SQLite 3.38+** (3.12 recommended). The setup's uv
  fetches its own 3.12 when there is none — also on Ubuntu 22.04, whose Python
  has SQLite 3.37.
- **A C compiler and Python's header files** — diffq (for audio-separator) is
  compiled during the install on Python 3.11+; the setup checks first.
- **FFmpeg** (for audio extraction, clip editing)
- **For the GPU**: an NVIDIA card from the GTX 9xx generation on (compute
  capability 5.0+) with a current driver. The setup installs PyTorch's CUDA
  12.8 build, and for a card older than GTX 16xx / RTX 20xx (below 7.0, which
  that build has no code for) the CUDA 12.6 build. The Docker image holds the
  12.8 build only: older cards use the native install. No CUDA toolkit is
  needed — PyTorch brings its own.
- **Disk**: about 8 GB for the install (uv keeps the downloads in its cache as
  well — `uv cache clean` frees them), plus about 4 GB of models
  (`./start.sh --prefetch` fetches them ahead, the setup offers it)
- **[Deno](https://deno.com)** for downloads from YouTube (links, the browser
  extension): yt-dlp needs a JavaScript runtime to read YouTube's player, and
  Deno is the one it uses without settings. Node works too, with
  `--js-runtimes node` in `youtube.extra_args` (config.yaml). Without one,
  YouTube downloads fail or miss formats; nothing else needs it.
- **A Hugging Face token** (only for speaker diarization — see
  [Speaker diarization: the Hugging Face token](#speaker-diarization-the-hugging-face-token))

```bash
# Debian / Ubuntu / WSL2
sudo apt install git curl ffmpeg build-essential python3-dev
# Fedora (FFmpeg from RPM Fusion)
sudo dnf install git curl ffmpeg gcc python3-devel
# Arch
sudo pacman -S git curl ffmpeg base-devel
```

## Models and their licenses

The models are not part of this repository: they are downloaded from their
publishers on first use, and the publishers' terms apply. Some are
**non-commercial** or state no license at all — check them before any
commercial use. As stated by their publishers, as far as known:

| Used for | Model | License / terms |
|----------|-------|-----------------|
| Transcription | Whisper (`Systran/faster-whisper-*`, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`) | MIT |
| Word timing, English | torchaudio `WAV2VEC2_ASR_BASE_960H` | MIT |
| Word timing, German, French, Spanish, Italian | torchaudio `VOXPOPULI_ASR_BASE_10K_*` | **CC BY-NC 4.0 (non-commercial)** |
| Word timing, other languages | community models on Hugging Face (WhisperX's list) | mostly Apache-2.0 — per model |
| Speaker diarization | `pyannote/speaker-diarization-community-1` (gated) | CC BY 4.0 |
| Voice vectors | ReDimNet (`IDRnD/ReDimNet`: its code kept in `whisper_tm/vendor/redimnet`, the weights `M-vb2-ptn` downloaded once and checked against their SHA-256) | code MIT, two of its files Apache-2.0; the weights are trained on VoxBlink2, which its publishers release under **CC BY-NC-SA 4.0** and allow no commercial use of, models trained on it included |
| Vocal isolation | MelBand Roformer (Kim), BS-Roformer (viperx) from the UVR model repository | **no license stated** |
| Images & sounds | `google/embeddinggemma-2` | Apache-2.0, with Gemma's prohibited-use policy |
| Language model (`setup.sh --llama`) | Qwen3.5-9B (`unsloth/Qwen3.5-9B-GGUF`, Q4_K_M) | Apache-2.0 |
| (a package, not a model) | `diffq`, installed with audio-separator on every install path; only its Demucs models use it, this app doesn't | **CC BY-NC 4.0 (non-commercial)** |

Downloading from YouTube or other sites may be restricted by their terms of
service and by copyright: you are responsible for having the right to
download and transcribe what you process.

**Responsible use.** Whoever runs the server decides what is recorded,
transcribed and identified, and is responsible for it. Recognizing the same
person across recordings works with voice vectors — in the EU biometric data
(GDPR Art. 9), and elsewhere recording conversations or keeping voiceprints
needs consent too. Ask the people you record, delete what they want deleted,
and export a voice dataset only with its
speaker's consent. This is not legal advice.

## Acknowledgements

Whisper Transcription Manager is built on the work of others, above all:

- [WhisperX](https://github.com/m-bain/whisperX) — word-level alignment and the pipeline this app extends
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper on CTranslate2
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — vocal isolation with the UVR models
- [ReDimNet](https://github.com/IDRnD/ReDimNet) — the voice vectors behind speaker recognition
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — local language models
- [FFmpeg](https://ffmpeg.org) and [yt-dlp](https://github.com/yt-dlp/yt-dlp) — decoding, cutting and downloading
- [Plyr](https://github.com/sampotts/plyr) — the media player
- [Feather](https://github.com/feathericons/feather) — the icons

Whisper Transcription Manager is an independent project. It is not affiliated with or
endorsed by the authors of WhisperX, by OpenAI (Whisper) or by any of the
projects above.

## Contributing and support

- **Bugs**: open an issue with the bug template; it asks for the version
  (Settings → About, or `./start.sh --version`) and the check's output.
- **Questions and install help**: the question template.
- **Security problems**: privately, as [SECURITY.md](../SECURITY.md) describes —
  not as an issue.
- **Changes**: [CONTRIBUTING.md](../CONTRIBUTING.md); what changed per version:
  [CHANGELOG.md](../CHANGELOG.md).

This is a beta kept by one person in their spare time: answers can take days,
and not every feature request will be built.

## License

MIT — see [LICENSE](../LICENSE). Third-party material in this repository (the
Plyr player, Feather-style icons, ReDimNet's model code — MIT, with two files
under the Apache License 2.0): [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

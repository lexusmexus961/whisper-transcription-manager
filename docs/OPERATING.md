# Running Whisper Transcription Manager

Updating, where everything lives, uninstalling, running it as a service — and
the failures a first run most likely meets.

## Updating

A native install:

```bash
cd whisper-transcription-manager
git pull
./setup.sh            # installs what the new version needs; config.yaml and the data stay
# then restart the server (Ctrl+C and ./start.sh, or: systemctl --user restart whisper-tm)
```

Docker: `git pull && docker compose build --pull && docker compose up -d`.
`--pull` fetches the newest base image, so the system libraries and ffmpeg get
their security fixes; now and then add `--no-cache` to rebuild every layer.

The browser extension keeps running its old code until it is reloaded:
`chrome://extensions` → ↻ on Whisper Transcription Manager.

A version opens only the database layout it was made for. If an update needs
another one, the CHANGELOG says so under **Breaking**, with what to do; a
database of another version is refused with a message, never changed. The
start backs the database up once a day (`database.auto_backup`, into
`database.backup_path` or `backups/` next to the database), and the voice
database (`speaker.db`: the names, the trash, the history) once a week into
the same folder, keeping the newest three.

## Backups and restore

What the app backs up by itself: the library database once a day at start
(`library_<time>.db`, `database.max_backups` of them) and the voice database
once a week (`speaker_<time>.db`, the newest three), both into
`database.backup_path`, or `backups/` next to the database when that is empty.
Settings → Database → *Back up now* makes one on demand.

What else to keep, in the data folder unless noted:

- the config folder: `config.yaml`, `secret_key` (sessions), `mcp_token`;
- `chats/`, `analyses/`, `deep_searches/`, `ingest/`;
- the transcript files, if they live in a central folder (`directories.transcripts`);
- optionally `media.db`, `audio.db` with `audio.db.fp/` and `rag.db` — they can be
  built again from the recordings and transcripts, but that takes long.

Copy a database file only while the server is stopped (the `-wal` file next to
it is then empty), or with `sqlite3 library.db ".backup copy.db"` while it runs.

To restore: stop the server, delete `library.db-wal` and `library.db-shm`, put
the backup in place as `library.db` (or `speaker.db`), and start it again. A
backup made by another version is refused at start; restore one of this
version.

## Where everything lives

| What | Where |
|---|---|
| Your settings | the config file that was loaded — `./config.yaml` in the app folder, or `~/.config/whisper-tm/config.yaml`, `WHISPER_TM_CONFIG`, `--config` (GUIDE.md: Configuration). Next to it: `secret_key` (sessions) and `mcp_token` |
| Database (transcripts, the library, the queue) | `~/.local/share/whisper-tm/library.db` (`database.path`) |
| Next to the database — they move with `database.path` | `speaker.db` (voices: names, trash, history), `rag.db` (the meaning index; `rag.path`), `media.db` (Images & sounds; `media_search.path`), `audio.db` with `audio.db.fp/` (the audio analysis; `audio_analysis.path`), `metrics.db` (the load history), `backups/` (when `database.backup_path` is empty), the lock file `library.db.lock` |
| The data folder `~/.local/share/whisper-tm/` — stays there whatever `database.path` says | `chats/`, `analyses/`, `deep_searches/`, `exports/` (finished exports, a day), `ingest/`, `ingest_chats.json`, `youtube_ids.json`, `cleanups/` (the logs of transcript clean-ups), `sounds/` (your own done sounds), `derived/` (the reranker's prepared model, when its own folder is read-only), `ui_prefs.json`, `autoscan.db`, `logs/` (job logs; `directories.logs`), `transcripts/` (the central folder, when used) |
| A credential | `youtube_cookies.txt` in the data folder, if you gave one for age-restricted or private videos — keep it private |
| Images & sounds' Python environment | `media-env/` in the data folder (several GB; `media_search.environment`) |
| Transcript files | `.transcripts/` next to each recording (`transcript_mode: alongside`), or the central folder (`directories.transcripts`) |
| Work files | `~/.cache/whisper-tm/work/` (`directories.work`) — emptied at every start; next to it the separation models and cached waveforms |
| Debug logs | `logs-debug/` in the same data folder (`logging.debug_dir`) |
| Models | `~/.cache/huggingface/` (Whisper, the diarization model, embedding models), `~/.cache/torch/` (word alignment, the voice model's weights), `~/.cache/whisper-tm/separation_models/` (vocal isolation), `~/nltk_data/` (the sentence splitter), the language model's GGUF in `models/gguf/` of the app folder — `./start.sh --prefetch` downloads the ones the settings need |
| Docker | the volumes `whisper-tm-data` and `whisper-tm-cache`, and `./config` |

## Uninstalling

1. Stop the server (and its service: `systemctl --user disable --now whisper-tm`).
2. Delete the app folder — the environment (`.venv`) is inside it.
3. Your data, if it is to go too: `~/.local/share/whisper-tm/`,
   `~/.cache/whisper-tm/` and, if there is one, `~/.config/whisper-tm/`.
   The transcript files next to the recordings: every `.transcripts/` folder in
   the library (`find /your/library -type d -name .transcripts`).
4. What you pointed elsewhere: a `database.path`, `database.backup_path` or
   central transcript folder of your own; the clips folder
   (`~/Videos/Whisper-TM-Clips` unless `directories.clips` says otherwise); the
   uploads in the library (`uploads/`, `directories.uploads`); `.srt`/`.txt`
   files that "alongside" exports left next to recordings; the systemd unit
   file `~/.config/systemd/user/whisper-tm.service`.
5. The models: their folders under `~/.cache/huggingface/hub/` and
   `~/.cache/torch/`, and `~/nltk_data/` (other programs may share them).
6. If setup.sh installed uv: `~/.local/bin/uv` and `uvx`, its cache
   (`uv cache clean`) and its Pythons in `~/.local/share/uv/python/` — other
   tools may use uv too.

Docker: `docker compose down -v` (removes the volumes as well), then
`docker image rm whisper-tm:local`; the `./config` folder (your configuration,
the session key, the MCP token) and Docker's build cache
(`docker builder prune`) stay until you remove them.

## Running it as a service

A user unit starts it with the machine and restarts it after a crash:

```ini
# ~/.config/systemd/user/whisper-tm.service
[Unit]
Description=Whisper Transcription Manager

[Service]
# The folder you cloned into; with ./setup.sh --venv DIR, ExecStart names DIR
WorkingDirectory=%h/whisper-transcription-manager
ExecStart=%h/whisper-transcription-manager/.venv/bin/python -m whisper_tm
# yt-dlp needs Deno for YouTube; a user unit doesn't read your shell profile
Environment=PATH=%h/.deno/bin:/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=10
# SIGTERM ends it the way Ctrl+C does: the running files go back to the queue
TimeoutStopSec=120

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now whisper-tm
loginctl enable-linger "$USER"        # keeps it running without a login
journalctl --user -u whisper-tm -f  # what it says
```

A user unit can't wait for a NAS mount or the network (those are system units).
If the library's share isn't there yet when the app starts, the queue pauses
and says the library folder is gone; once the share is mounted, *Resume*
carries on.

## Behind a reverse proxy

For access from the internet, put a reverse proxy with TLS and its own login in
front of the app; the app's own login alone is not made for the open
internet. A starting point for nginx at the root of a host name (not yet
tried end to end with this version):

```nginx
server {
    server_name transcripts.example.org;     # and your TLS settings
    client_max_body_size 64m;                # above server.upload_chunk_mb (32)
    location / {
        proxy_pass http://127.0.0.1:9860;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;            # long exports and the live streams
        proxy_buffering off;                 # Server-Sent Events (queue, progress)
    }
}
```

Then put the public origin into the config, or every API request gets 403,
and with the login on keep its cookie to https:

```yaml
security:
  trusted_origins: ["https://transcripts.example.org"]
server:
  secure_cookie: true
```

The `X-Forwarded-For` line matters: a request without it counts as coming from
this machine (GUIDE.md: Network Access). Behind Cloudflare, a request may carry
at most 100 MB — the default upload piece (32 MB) fits; a stricter proxy needs
a smaller `server.upload_chunk_mb`. The app expects the root path of its host
name; a sub-path (`/transcripts/`) is not supported.

## The first failures

- **"Port 9860 can't be used" / "Another Whisper Transcription Manager server uses this
  database"** — a server is running already. Stop it, or give the second one
  its own `--port` and data folder (`database.path`, `directories.work`).
- **Other devices can't open it** — it listens on this machine only by
  default (`server.host: 127.0.0.1`): GUIDE.md: Network Access.
- **The transcript is in the wrong language** — the language is detected
  for each recording, and detection can miss on noisy ones: set the
  library's language under Settings → Transcription → Language, or the
  recording's own in the dialog that queues it.
- **No speakers in the transcripts** — speaker diarization is off by default
  (Settings → Transcription → Speaker diarization) and needs the Hugging Face
  token with the model's conditions accepted: `./start.sh --hf-token` checks
  both and names what is missing.
- **It runs on the CPU although there is a GPU** — `./start.sh --check` shows
  what was found and why (driver, CUDA, the PyTorch build); `--check --gpu-test`
  computes on the card.
- **"… could not be downloaded from …"** — a model downloads on first use,
  and the machine had no connection then (or a proxy or filter stopped it).
  `./start.sh --prefetch` fetches every model the settings need and says for
  each where it comes from; behind a proxy NLTK's sentence splitter needs
  `NLTK_ALLOW_PROXIED_URLOPEN=1`. The queue pauses on such an error — *Resume*
  once the model is there.
- **The queue paused by itself** — the system warning says why: the card ran
  out of memory and stayed full (another program on the GPU?), the disk of the
  work folder is (nearly) full, the library folder is gone (a share not
  mounted), or a model could not be downloaded (no internet on the first job).
  Fix it, then *Resume* in the queue; the entry is still there.
- **A job failed** — its log is in `logs/` in the data folder (the queue shows
  the error with the entry). The setup guide (*Help & setup* → *Setup*) checks the
  installation again.
- **"The server went down twice while this file ran"** — the process ended
  while the file was being worked on, twice (a crash, the out-of-memory
  killer): an entry the server went down under runs alone the next time, and
  the second time it isn't started again — a restart policy would otherwise
  run it into the same crash forever. What ended the process is in the
  server's output (`journalctl --user -u whisper-tm`, or the terminal);
  *Retry* in the queue runs it again.

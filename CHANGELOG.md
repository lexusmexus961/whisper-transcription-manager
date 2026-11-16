# Changelog

The notable changes of each version. Before 1.0 any version may change the
config, the database, the HTTP API, the MCP tools and what the browser
extension expects — such a change stands under **Breaking**, with what to do
about it. A version opens only the database layout it was made for: one of
another version is refused with a message, never changed.

## 0.9.0 — 2026-11-16 — first public release (beta)

The first published version. It ran in daily use on a large library before.
Fresh installs were tested on Linux with NVIDIA cards of 4, 8 and 12 GB
(Turing and Ampere), on a machine without a GPU, with Python 3.11 and 3.12,
and in Docker end to end; not tested on a card older than Turing, on an AMD
card or on a real Windows machine (Docker Desktop or WSL2). Hence a beta.

- **The window** — a rail of places on the left (Home, Library, Queue, Find,
  Ask, Analyses, Voices, Machine, Settings, Help & setup) with pins and
  badges; back and forward through places, tabs and recordings; Home with live
  cards; the Library as a folder tree with each folder's statistics; the Queue
  with transcriptions, AI jobs, exports and clips in one list; Machine and Help
  as windows with tabs.
- **A recording in tabs** — Transcript, Summary, Translation, Speakers, Clips,
  Statistics, Details, Analyses, under a quiet header and the player's
  who-speaks-when lanes.
- **Transcription** — WhisperX with word timing, speaker diarization
  (pyannote) and vocal isolation (MelBand Roformer), run by a queue on the
  server: it survives restarts and closed browsers, works on several files at
  once, takes an out-of-memory error up again with less memory, and pauses
  with the cause when the machine is at fault (a full disk, a share that is
  gone, no internet for a model).
- **Library** — a whole NAS indexed into SQLite with a full-text index;
  transcripts in the database and as JSON next to the recordings; continuous
  scanning of the whole library or of chosen folders; the Queue picks files
  from the whole library; the file list filters by type, length, when added,
  language, what a recording has, a subfolder and people.
- **Voices** — voice vectors (ReDimNet) for every speaker of a recording,
  people named once and found again across the library, a workspace to
  assign, review and extract voices, and the voice map.
- **Search & Ask** — *Find*: describe a moment in your own words, and every
  search runs at once (words, meaning, what is seen or heard, file names),
  checked by the language model. Besides it the text search, situations,
  deep search, *Images & sounds*, *Same audio* (every recording that holds a
  snippet, the duplicates, the recordings that hold others), *Conversations*
  (who talked with whom, and how: by the people, the length, the kind and
  sentences longer than usual — for the people chosen, than usual for them —,
  also the stretch of a long recording where given people talk together; ✂
  hands them to the voice extractor) and file names; *Ask*, a chat with the library
  through the app's own tools; *Analyses*, one task over many recordings.
- **Language model** — summaries, correction, translation and analyses
  through llama.cpp (the setup installs it with Qwen3.5-9B), Ollama, OpenAI,
  vLLM / LM Studio or any OpenAI-compatible server.
- **MCP server** — the app's tools for an AI client, over stdio or HTTP, and
  an OpenAPI description for clients without MCP. Besides the searches,
  transcripts and voices they cover conversations: who talked with whom, and
  how — across the library, in one recording, for one person.
- **Browser extension** — YouTube videos from any page summarized,
  transcribed or analyzed together, as video or audio only; videos taken
  back out of a batch in the side panel before they are fetched; with the
  server's login on, through its access token.
- **Export** — one page for a recording or any set of them (a folder, a
  selection, e.g. only the recordings added today): text, subtitles, Word,
  PDF, Markdown, a table, data, video and audio, with options that are
  remembered and presets; big and media exports run in the Queue.
- **Clips** — several cuts into one clip or one each, vertical or square,
  pauses out, subtitles burned in, soft fades; *Best moments* lets the
  language model suggest three.
- **Running it** — `setup.sh` or Docker; a health check, a setup guide and a
  performance view; every setting with its description
  ([docs/OPERATING.md](docs/OPERATING.md): updating, data, a systemd unit).

Defaults to know: the server listens on this machine only
(`server.host: 127.0.0.1`; docs/GUIDE.md: Network Access); the transcription
language is detected for each recording and speaker diarization is off; the
MCP server answers other machines only with its token; the model, the vocal
isolation and the number of files at once follow the machine (large-v3 with
vocal isolation from 8 GB, three files at once from 12 GB, Whisper turbo
without vocal isolation on smaller cards and on a CPU), and on a card of up
to 11 GB the models are loaded for each step and freed after it
(`hardware.low_vram: auto`).

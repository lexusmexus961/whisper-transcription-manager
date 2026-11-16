# Contributing

Thanks for taking a look. A few things that will save you time.

## Language

Everything in this repository is English: code, identifiers, comments,
docstrings, commit messages, documentation and user-facing text.

The one exception is German that is **data rather than language**: words the
code matches in German transcripts and queries. That is the German stopword
list, the filler-phrase and hallucination lists, Find's words for kinds of
recordings, the duplicates view's intent patterns, the analysis' markers for
"nothing was said", the search's word endings and its day-first dates, and
German sample text in test fixtures. Translating them would break the feature.

## Setup

```bash
./setup.sh --dev --no-hf-token --no-llama   # creates .venv and installs everything
./.venv/bin/python -m pytest -q
```

`setup.sh` detects the GPU and installs WhisperX with the PyTorch build for
it (CUDA, ROCm or CPU, at the version WhisperX pins), fastmcp (the chat and
the MCP server — their tests need it) and yt-dlp; `make setup-locked`
builds exactly the pinned environment of `requirements-lock-nvidia.txt`. The
tests need no Hugging Face token and no local language model —
`--no-hf-token` and `--no-llama` skip those questions.

Besides Python, the tests use **Node.js** (the frontend JavaScript runs in a
small Node harness) and **ffmpeg/ffprobe** (cutting, decoding, probing).
Without them those tests are skipped; in CI (`CI` set) a missing one stops
the run instead, so nothing is skipped silently there.

## Tests

The suite is the safety net for almost every change here — it covers the
API, the queue, the scanner, the speaker recognition and, through a small
Node harness, the frontend JavaScript.

```bash
./.venv/bin/python -m pytest -q          # all of it, about ten minutes
./.venv/bin/python -m pytest -q tests/test_queue.py
```

A pull request is expected to keep it green. New behaviour comes with a
test that fails without the change.

## Style

- Python: `./.venv/bin/python -m ruff check .` must be clean — with the
  ruff version the CI pins (0.15.21, which `setup.sh --dev` installs); a
  newer one brings new rules. The code is formatted with
  `./.venv/bin/python -m ruff format .` (the CI checks it). Type hints where
  they help.
- JavaScript: no framework, no build step. Globals are wired through the
  script order in `whisper_tm/static/index.html`; state lives in
  `static/js/state.js`.
- The database is the primary source, not the JSON files beside the media —
  `abs_path` is the stable key across a change of media directory.

## Architecture

`docs/` holds the longer write-ups: `PATHS.md` on path handling,
`LLM-PIPELINE.md` on correction and translation. What is known to be
unfinished: docs/GUIDE.md → Known limitations, and the issue tracker.

## Commits

One topic per commit. The message says what changed and why, not just
which files moved.

## License

By contributing you agree that your contribution is licensed under the
MIT License of this repository (`LICENSE`).

# tools/ — maintenance scripts

Scripts for looking after a library by hand; the app does not need them to
run.

## The tools

Every script runs with the environment's Python — `./.venv/bin/python
tools/<script>.py` from the app folder (or plain `python` in a shell with the
environment activated).

**`whisper_tm.mcp_server`** — not a script in `tools/`, but mentioned here
because you run it the same way: the MCP server through which a local AI can
operate the app (33 tools, stdio, HTTP or OpenAPI). It needs the **running
app** and talks to it via HTTP. Requires `fastmcp`, which `setup.sh` installs.
Setup in docs/GUIDE.md.

```bash
./.venv/bin/python -m whisper_tm.mcp_server --check      # can it reach the app?
./.venv/bin/python -m whisper_tm.mcp_server --stdio        # for AI client
./.venv/bin/python -m whisper_tm.mcp_server --http         # as service on port 5399
./.venv/bin/python -m whisper_tm.mcp_server --stdio --read-only   # don't trigger
./.venv/bin/python -m whisper_tm.mcp_server --openapi      # for non-MCP clients
```


**`migrate_paths.py`** — re-derives every `rel_path` against the current media
directory and consolidates duplicates that creates. Runs as a dry run by
default, only writes with `--apply` — with the server stopped: it rewrites
the rows a running server works with — creates a backup first and leaves a
marker in `schema_info`. For a library that moved to another path, use
`./start.sh --remap-path` instead (docs/GUIDE.md → Moving to Another Machine). The
path model is in `docs/PATHS.md`.

```bash
./.venv/bin/python tools/migrate_paths.py --root /new/root          # dry run
./.venv/bin/python tools/migrate_paths.py --root /new/root --apply  # apply (server stopped)
```

**`extension_zip.py`** — the browser extension as two zip files in `dist/`
(`make extension-zip`): one to load unpacked, with the fixed ID the server
trusts, and one for the Chrome Web Store without the manifest's `key`.

### Speaker tools

All of these work on `speaker.db` and need a GPU for anything that computes
vectors.

**`speaker_quality.py`** — how well the named voices recognize a person in a
recording they haven't seen: each recording is left out of the gallery in
turn and asked who speaks in it. Reports the hit rate (the right person above
the threshold), the accuracy (the best match is the right one) and the false
alarms (a wrong person above the threshold); `--threshold` tries another.

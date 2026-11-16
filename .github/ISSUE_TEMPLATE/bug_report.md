---
name: Bug report
about: Something behaves differently than it should
labels: bug
---

**What happened**

**What you expected**

**Steps to reproduce**

**Environment**
- Install method: setup.sh / make setup-locked / Docker / Docker Desktop or WSL2 / manual
- OS (on Windows: the WSL2 distribution):
- GPU and its memory, NVIDIA driver version (or CPU-only):
- Browser (for problems in the page or the extension):
- Version: output of `./start.sh --version` (Docker: `docker compose run --rm whisper-tm --version`)
- Output of `./start.sh --check` (Docker: `docker compose run --rm whisper-tm --check`), or the
  setup guide's System check

**Logs**
Relevant lines from the job log or the debug log folder (`logs-debug/` in the data folder).

Before you paste: logs and the check's output carry absolute paths (with your user
name), host names, file names, speaker names and transcript text — replace what you
don't want to publish. Don't paste your `config.yaml`: unlike the debug traces, it is
not redacted.

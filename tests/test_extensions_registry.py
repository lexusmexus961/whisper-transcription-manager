"""The registry must not be frozen at import time.

`whisper_tm/extensions.py` holds the singletons (DB, Hardware, JobManager)
and states in its own head: *"Only import this module for access to global
instances."* Several modules did this anyway

    from whisper_tm.extensions import db

and that is something different: It binds the VALUE at import time. `db` is
`None` at that point; `create_app()` sets `ext.db` only afterwards. Whoever
imports such a module BEFORE `create_app()` keeps the old `None` forever
— and gets a 503 from every database endpoint.

In the application this worked well, but only by chance of order: `app.py`
sets `ext.db` at line 100 and imports the blueprints only at line 264.
It came to light during a health check, via a roundabout way
that describes the issue well: A new test file imported `whisper_tm.api.main`
during collection — afterwards a completely different test failed (the traversal
protection reported 503 instead of 400), three files further away.

This test reproduces exactly this order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The modules that provide endpoints with database access. Whoever is listed
# here must read the registry at RUNTIME.
CHECKED = ["whisper_tm.api.main"]


def test_an_early_import_does_not_take_the_database_from_the_endpoint(tmp_path):
    """The core: import first, then `create_app()` — and it must work.

    Runs in its OWN process: The order of imports cannot be restored within
    a running interpreter once another test has already loaded the module.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    script = f"""
import os, sys
os.environ["WHISPER_TM_CONFIG"] = {str(cfg)!r}
os.environ["HOME"] = {str(tmp_path)!r}
sys.path.insert(0, {str(REPO)!r})

# FIRST the endpoint module — that is the case we are interested in.
import whisper_tm.api.main            # noqa: F401

from whisper_tm.config import config
config.load(None)
config._data.setdefault("directories", {{}})["media"] = {str(tmp_path / "m")!r}
config._data.setdefault("database", {{}})["path"] = {str(tmp_path / "t.db")!r}
config._resolve_directories()
config._ensure_directories()

from whisper_tm.app import create_app
app = create_app()

from whisper_tm import extensions as ext
assert ext.db is not None, "create_app() did not set a DB"

# And now the actual question: DOES the endpoint see it too?
c = app.test_client()
r = c.get("/api/scan_entry/does_not_exist.mp4")
print("STATUS", r.status_code)
"""
    run = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert run.returncode == 0, run.stderr[-3000:]
    line = [z for z in run.stdout.splitlines() if z.startswith("STATUS")]
    assert line, run.stdout[-2000:]
    status = int(line[-1].split()[1])
    assert status != 503, (
        "the endpoint sees no database — the module froze `db` at import time "
        "instead of reading the registry at runtime"
    )


def test_the_checked_file_does_not_bind_the_db_at_import():
    """The cheap guardian alongside.

    The test above starts a process and takes seconds; this one just reads
    the source code and fails immediately if someone reintroduces the pattern.
    """
    for module in CHECKED:
        path = REPO / (module.replace(".", "/") + ".py")
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            # Only at MODULE level, i.e., without indentation.
            #
            # The same import WITHIN a function is okay: It runs fresh on each
            # call and thus reads the current value. `watcher_status()` in
            # `api/main.py` does exactly this and must not be flagged.
            if line[:1].isspace() or line.lstrip().startswith("#"):
                continue
            assert not line.startswith("from whisper_tm.extensions import"), (
                f"{path.name}: `{line}` freezes the value at import time — "
                f"instead use `from whisper_tm import extensions as ext` "
                f"and `ext.db` at the point of use"
            )

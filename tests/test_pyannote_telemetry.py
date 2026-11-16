"""pyannote.audio's usage metrics are off unless the user turns them on.

pyannote.audio 4 sends each diarization's duration and speaker count to
pyannote.ai by default (its telemetry/config.yaml) — unless
PYANNOTE_METRICS_ENABLED says otherwise. A self-hosted server keeps that to
itself; whisper_tm sets the default before anything imports pyannote.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_after_import(**extra: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "PYANNOTE_METRICS_ENABLED"}
    env.update(extra)
    env["PYTHONPATH"] = str(ROOT)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, whisper_tm; print(os.environ.get('PYANNOTE_METRICS_ENABLED'))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_metrics_are_off_by_default():
    assert _env_after_import() == "false"


def test_the_users_choice_wins():
    assert _env_after_import(PYANNOTE_METRICS_ENABLED="true") == "true"

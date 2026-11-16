"""The test session never touches the real installation (tests/conftest.py).

A test that creates an app without moving HOME, next to a running server,
would import old transcripts from the real media folder into the real
database and empty the real work directory under running jobs.
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path


def test_no_path_of_the_app_lies_in_the_real_home():
    # Where HOME points is up to each test (several move it again) — but
    # never back to the account's own home
    from whisper_tm.config import config

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert Path(os.environ["HOME"]).resolve() != real_home
    for path in (config.db_path, config.work_dir, config.media_dir, config.transcript_dir):
        assert real_home not in Path(path).resolve().parents, path


def test_a_test_app_does_not_start_the_queue():
    from whisper_tm.config import config

    assert config.get("queue", "autostart") is False

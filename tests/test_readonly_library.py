"""A library folder that can't be written ("alongside" mode, a read-only
share) no longer costs the transcript at the very end.

The worker wrote the JSON next to the recording first and the database
second: a read-only share ran the whole GPU job and then failed on
.transcripts/, with nothing saved. Now the folder is tried right after the
audio is extracted (the job says at once that the transcript stays in the
database), the database is written first — the app reads from it — and a
file that can't be written beside the recording is a warning, not a lost
transcription.
"""

from __future__ import annotations

import inspect
import os
import re

import pytest

from whisper_tm.core import transcription as tr


def _flat(text: str) -> str:
    """Source text with the formatter's line breaks taken out: one space
    for every run of whitespace, none inside brackets, no trailing commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*([)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+([)\]}])", r"\1", text)


def test_a_writable_folder(tmp_path):
    target = tmp_path / "media" / ".transcripts"
    assert tr._folder_writable(target) and target.is_dir()
    assert list(target.iterdir()) == []  # the probe is gone


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes everywhere")
def test_a_read_only_folder(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    media.chmod(0o555)
    try:
        assert not tr._folder_writable(media / ".transcripts")
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        assert not tr._folder_writable(ro)
    finally:
        media.chmod(0o755)
        (tmp_path / "ro").chmod(0o755)


def test_the_database_first_then_the_file():
    src = _flat(inspect.getsource(tr.transcription_worker))
    save = src[src.index("transcript_mode = config.transcript_mode") :]
    assert save.index("db.upsert_transcript(filename, result,") < save.index(
        "atomic_write_json(tp, result)"
    )
    # A file that can't be written is a warning while the database has it
    handler = save[save.index("atomic_write_json(tp, result)") :]
    assert "except OSError as e:" in handler and "if not db or db_err is not None:" in handler
    assert "it is kept in the database" in handler


def test_the_folder_is_tried_before_the_gpu_work():
    src = inspect.getsource(tr.transcription_worker)
    check = src.index("_beside_ok = _folder_writable(")
    assert (
        src.index('step_done("Audio extraction")') < check < src.index('phase("Audio splitting")')
    )
    assert "the transcript is kept in the database only" in src

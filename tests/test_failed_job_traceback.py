"""A failed job's traceback goes into its log, not onto the console.

Every failed transcription printed its full traceback to the server's
console — one offline failure some 130 lines — while the queue already says
there in one line which file failed and why. The traceback now goes into
the job's log (kept as a file, where a bug report finds it) and the debug
log.
"""

from __future__ import annotations

import logging
from pathlib import Path

from whisper_tm.core import transcription as tx
from whisper_tm.jobs import JobManager

SOURCE = (
    Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "transcription.py"
).read_text(encoding="utf-8")


def test_the_traceback_lands_in_the_job_log(tmp_path, capsys, caplog):
    jm = JobManager(logs_dir=tmp_path / "logs")
    job = jm.create("job-1", filename="a.mp4")
    caplog.set_level(logging.DEBUG, logger=tx.logger.name)
    try:
        raise RuntimeError("model could not be downloaded")
    except RuntimeError:
        tx._log_traceback(jm, job, "a.mp4")
    jm.fail(job, "model could not be downloaded")
    text = (tmp_path / "logs" / job.log_file).read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text and "RuntimeError" in text
    out = capsys.readouterr()
    assert "Traceback" not in out.err and "Traceback" not in out.out
    assert any(r.exc_info for r in caplog.records)  # the debug log has it


def test_nothing_prints_it_any_more():
    assert "print_exc(" not in SOURCE

"""The final line was missing from EVERY job log.

Observed: the job log files end in the middle of post-processing —

    [12:00:00] ⏱ voice vectors: 30s
    [12:00:02] post-processing: 10 phantom suspects, 300 overlapped …

— and nothing more. No "✅ Done!", even though the queue shows the job as
completed.

The cause is one line too late in the source::

    job_manager.complete(job, result)                    # writes file
    job_manager.log(job, "✅ Done! …")                   # too late

``complete()`` and ``fail()`` call ``_persist_log()``, and that is an
``open(…, "w")`` over the lines collected so far. Everything after lands only
in the list in memory and dies with the job.

It only came to light when a newly added log line after it also did not appear
— a "Done!" that never existed, nobody misses.

Both are checked: the repaired callers AND the safeguard behind them.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from whisper_tm.jobs import JobManager

REPO = Path(__file__).resolve().parent.parent


def _flat(text: str) -> str:
    """Source text with the formatter's line breaks taken out: one space
    for every run of whitespace, none inside brackets, no trailing commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*([)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+([)\]}])", r"\1", text)


@pytest.fixture()
def jm(tmp_path):
    return JobManager(logs_dir=tmp_path / "logs")


def _job(jm):
    j = jm.create("job-1", filename="videos/archive/a recording.mp4")
    jm.log(j, "Job started")
    return j


class TestLateAddition:
    def test_a_line_after_complete_still_arrives(self, jm, tmp_path):
        j = _job(jm)
        jm.complete(j, {"ok": True})
        jm.log(j, "✅ Finished! 421 segments")
        text = (tmp_path / "logs" / j.log_file).read_text(encoding="utf-8")
        assert "✅ Finished! 421 segments" in text

    def test_a_line_after_fail_still_arrives(self, jm, tmp_path):
        j = _job(jm)
        jm.fail(j, "broken")
        jm.log(j, "❌ Cancelled")
        text = (tmp_path / "logs" / j.log_file).read_text(encoding="utf-8")
        assert "❌ Cancelled" in text

    def test_the_late_line_creates_no_second_file(self, jm, tmp_path):
        """The file name carries the timestamp. If it were regenerated, every
        addendum would create another file — with a large log directory
        that would never be noticed."""
        j = _job(jm)
        jm.complete(j, {"ok": True})
        for i in range(3):
            jm.log(j, f"Addendum {i}")
        files = list((tmp_path / "logs").glob("*.log"))
        assert len(files) == 1, [d.name for d in files]
        text = files[0].read_text(encoding="utf-8")
        assert all(f"Addendum {i}" in text for i in range(3))

    def test_before_completion_it_does_not_write_per_line(self, jm, tmp_path):
        """The safeguard must not turn into a bottleneck: during processing
        thousands of lines accumulate, and writing each one individually to
        a NAS would be the most expensive way to keep a log."""
        j = _job(jm)
        for i in range(50):
            jm.log(j, f"Step {i}")
        assert not list((tmp_path / "logs").glob("*.log"))


class TestCallers:
    """The safeguard is the second line of defense — the callers still need
    to be fixed. Otherwise the correct order would be left to chance."""

    def _source(self) -> str:
        from whisper_tm.core.transcription import transcription_worker

        return inspect.getsource(transcription_worker)

    def test_the_done_line_comes_before_complete(self):
        q = self._source()
        done = q.index('"✅ Done!')
        complete = q.index("job_manager.complete(job, result)")
        assert done < complete

    def test_the_save_line_comes_before_complete(self):
        q = self._source()
        memory = q.index("Server memory")
        complete = q.index("job_manager.complete(job, result)")
        assert memory < complete

    def test_the_cancel_line_comes_before_fail(self):
        q = self._source()
        line = q.index('"❌ Cancelled"')
        fail = q.index('job_manager.fail(job, "Cancelled")')
        assert line < fail

    def test_the_separation_only_run_too(self):
        """It has its own completion — and had the same bug."""
        q = _flat(self._source())
        line = q.index("Separated, no transcript")
        complete = q.index("job_manager.complete(job, {")
        assert line < complete

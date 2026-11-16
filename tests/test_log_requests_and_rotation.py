"""Outside debug mode the log keeps only failed request lines, a run's log
is rotated, and pruning takes the rotated parts along."""

import logging
import os
import re
import time

from whisper_tm.core import debug


def _flat(text: str) -> str:
    """Source text with the formatter's line breaks taken out: one space
    for every run of whitespace, none inside brackets, no trailing commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*([)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+([)\]}])", r"\1", text)


def _record(msg, *args, level=logging.INFO):
    return logging.LogRecord("werkzeug", level, __file__, 1, msg, args, None)


def test_only_failed_requests_are_kept():
    f = debug._RequestLineFilter()
    line = '127.0.0.1 - - [x] "%s" %s %s\n'
    assert not f.filter(_record(line, "GET /api/queue?file=/media/a.mp4 HTTP/1.1", "200", "-"))
    assert not f.filter(_record(line, "GET /static/x.js HTTP/1.1", "304", "-"))
    assert f.filter(_record(line, "GET /api/nothing HTTP/1.1", "404", "-"))
    assert f.filter(_record(line, "POST /api/x HTTP/1.1", "500", "-"))
    assert f.filter(_record(" * Running on http://127.0.0.1:9860"))
    assert f.filter(_record("something", level=logging.WARNING))


def test_pruning_takes_the_rotated_parts(tmp_path):
    for i, name in enumerate(["whisper_tm_1.log", "whisper_tm_2.log", "whisper_tm_3.log"]):
        p = tmp_path / name
        p.write_text("x")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    (tmp_path / "whisper_tm_1.log.1").write_text("x")
    (tmp_path / "whisper_tm_3.log.1").write_text("x")
    debug._prune_old_logs(tmp_path, keep=2)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["whisper_tm_2.log", "whisper_tm_3.log", "whisper_tm_3.log.1"]


def test_the_run_log_rotates():
    from pathlib import Path

    text = Path(debug.__file__).read_text(encoding="utf-8")
    assert "RotatingFileHandler(_log_file, maxBytes=LOG_FILE_BYTES" in _flat(text)


def test_werkzeugs_development_server_warning_is_dropped():
    import logging

    from whisper_tm.core.debug import _RequestLineFilter

    rec = logging.LogRecord(
        "werkzeug",
        logging.INFO,
        __file__,
        1,
        "\x1b[31mWARNING: This is a development server. Do not use it.\x1b[0m",
        None,
        None,
    )
    assert _RequestLineFilter().filter(rec) is False
    rec = logging.LogRecord(
        "werkzeug", logging.INFO, __file__, 1, " * Running on http://127.0.0.1:9860", None, None
    )
    assert _RequestLineFilter().filter(rec) is True

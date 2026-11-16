"""
Call tracing must not write secrets.

The trace hook logs every argument and every return value of the package's
functions. With it on, trace.log got the stored login password
(``← core.auth._get_password = '…'``), the typed one
(``_check_password(input_password='…')``), the HuggingFace token
(``config.hf_token``) and API keys passed on as plain ``value`` arguments.
"""

from __future__ import annotations

import logging

import pytest

from whisper_tm.config import config
from whisper_tm.core import debug


@pytest.fixture
def traced(caplog, monkeypatch):
    monkeypatch.setitem(
        config._data,
        "auth",
        {"enabled": False, "password": "pw-stored-1234", "api_token": "tok-stored-5678"},
    )
    monkeypatch.setitem(
        config._data.setdefault("transcription", {}), "hf_token", "hf_secret_abcdef"
    )
    caplog.set_level(logging.DEBUG, logger="whisper_tm.trace")
    debug.start_call_tracing(with_arguments=True)
    try:
        yield caplog
    finally:
        debug.stop_call_tracing(reason="test")


def _run_secret_code():
    from whisper_tm.config import coerce_value
    from whisper_tm.core import auth

    auth._check_password("pw-typed-9999")
    auth._check_token("tok-typed-0000")
    _ = config.hf_token
    # A secret handed on under a harmless parameter name
    coerce_value("tok-stored-5678", "")


def test_trace_writes_no_secret(traced):
    _run_secret_code()
    text = "\n".join(r.getMessage() for r in traced.records)
    assert "auth._check_password" in text  # the trace did run
    for secret in (
        "pw-stored-1234",
        "pw-typed-9999",
        "tok-stored-5678",
        "tok-typed-0000",
        "hf_secret_abcdef",
    ):
        assert secret not in text
    assert "<redacted>" in text


def test_ordinary_values_are_still_written(traced):
    from whisper_tm.config import coerce_value

    coerce_value("17", 3)
    text = "\n".join(r.getMessage() for r in traced.records)
    assert "value='17'" in text

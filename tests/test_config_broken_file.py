"""A config.yaml that can't be read says so in two lines.

The file was parsed without catching the parser's error: a stray tab or
quote printed a page of Python traceback at start, with ``--check`` and with
``--hf-token`` alike, and a file holding a list crashed further down.
"""

from __future__ import annotations

import sys

import pytest

from whisper_tm.config import ConfigError, _read_user_config


def test_a_typo_names_the_file_and_the_line(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("server:\n  port: 9860\n\thost: 127.0.0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        _read_user_config(f)
    said = str(e.value)
    assert str(f) in said and "line 3" in said and "defaults" in said
    assert "Traceback" not in said


def test_an_open_quote_names_its_line(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text('transcription:\n  default_language: "en\nqueue:\n  lanes: 2\n', encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        _read_user_config(f)
    assert "line" in str(e.value)


@pytest.mark.parametrize(
    "text, kind", [("- a\n- b\n", "list"), ("just words\n", "str"), ("42\n", "int")]
)
def test_a_file_that_holds_no_settings_says_what_it_needs(tmp_path, text, kind):
    f = tmp_path / "config.yaml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        _read_user_config(f)
    assert kind in str(e.value) and "key: value" in str(e.value)


def test_an_empty_file_is_the_defaults(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("# nothing set\n", encoding="utf-8")
    assert _read_user_config(f) == {}


def test_a_file_that_is_not_text_is_named(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_bytes(b"server:\n  host: \xff\xfe\n")
    with pytest.raises(ConfigError) as e:
        _read_user_config(f)
    assert "UTF-8" in str(e.value)


def test_the_command_prints_the_message_and_no_traceback(tmp_path, monkeypatch, capsys):
    from whisper_tm import __main__ as cli
    from whisper_tm.config import config

    f = tmp_path / "config.yaml"
    f.write_text("server:\n\tport: 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["whisper-tm", "--check", "--config", str(f)])
    try:
        with pytest.raises(SystemExit) as e:
            cli.run()
    finally:
        config.load(None)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert str(f) in err and "line 2" in err and "Traceback" not in err

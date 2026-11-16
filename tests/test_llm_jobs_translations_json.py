"""The translations' JSON backup (api/llm_jobs.py).

All tracks of a file live in one JSON next to the transcripts. It was read,
changed and written back with no lock and without a temporary file: two
translations of one file finishing together lost one language, and a crash
mid-write truncated the backup of every track.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from whisper_tm.api import llm_jobs


@pytest.fixture
def tdir(tmp_path, monkeypatch):
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "transcript_dir", property(lambda self: tmp_path))
    return tmp_path


def test_two_languages_at_once_both_stay(tdir, monkeypatch):
    real = llm_jobs._load_translations_json

    def slow_load(filename):
        data = real(filename)
        time.sleep(0.2)  # both read before either writes
        return data

    monkeypatch.setattr(llm_jobs, "_load_translations_json", slow_load)
    threads = [
        threading.Thread(
            target=llm_jobs._store_translation_json,
            args=("a.mp4", lang, {"model": "m", "segments": [lang]}),
        )
        for lang in ("English", "French")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert sorted(real("a.mp4")) == ["English", "French"]


def test_a_failed_write_keeps_the_old_backup(tdir, monkeypatch):
    llm_jobs._store_translation_json("a.mp4", "English", {"segments": ["x"]})

    real_dump = llm_jobs.json.dump

    def broken_dump(obj, f, **kw):
        f.write('{"Eng')
        raise OSError("disk full")

    monkeypatch.setattr(llm_jobs.json, "dump", broken_dump)
    llm_jobs._store_translation_json("a.mp4", "French", {"segments": ["y"]})
    monkeypatch.setattr(llm_jobs.json, "dump", real_dump)
    data = json.loads(llm_jobs._translations_json_path("a.mp4").read_text())
    assert list(data) == ["English"]
    assert [p.name for p in tdir.iterdir()] == ["a.mp4_translations.json"]


def test_removing_one_language(tdir):
    llm_jobs._store_translation_json("a.mp4", "English", {"segments": ["x"]})
    llm_jobs._store_translation_json("a.mp4", "French", {"segments": ["y"]})
    assert llm_jobs._drop_translation_json("a.mp4", "English") is True
    assert llm_jobs._drop_translation_json("a.mp4", "English") is False
    assert list(llm_jobs._load_translations_json("a.mp4")) == ["French"]

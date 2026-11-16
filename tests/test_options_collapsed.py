"""The path of a transcription option must be GAPLESS.

An option name existed literally in nine places in this application's
code, and three of the omissions are **silent**:

* if the key is missing in ``api/queue.py::_SETTING_KEYS``, the server
  silently discards it — no log, no error, the option simply never works;
* if the default in the frontend is missing, the toggle on the
  settings page is a fake: it dutifully gets saved, but the dialog
  overwrites the value every time it's opened;
* if the fourth rewrite in the upload path is missing, the option works
  everywhere except there.

Since the rewrite, every option is defined **once** in ``TX_OPTIONS``
(``static/js/txoptions.js``). These tests confirm that the table and
the server side match up — the table is only worth as much as it
actually is the single source.

The JS table is deliberately **parsed** here instead of rebuilt: a
rebuilt list in the test would be the tenth copy and would repeat exactly
the mistake the rewrite eliminated.
"""

import inspect
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
JS = REPO / "whisper_tm" / "static" / "js"


def _table() -> list[dict]:
    """Read ``TX_OPTIONS`` from txoptions.js (key, field_el, kind, has_before)."""
    text = (JS / "txoptions.js").read_text(encoding="utf-8")
    m = re.search(r"const TX_OPTIONS = \[(.*?)\n\];", text, re.S)
    assert m, "TX_OPTIONS not found — was the table renamed?"
    out = []
    for block in re.finditer(r"\{(.*?)\}(?=,\n|\n\];)", m.group(1), re.S):
        b = block.group(1)
        entry = {}
        for field_name in ("key", "field_el", "kind"):
            t = re.search(rf"{field_name}:\s*'([^']+)'", b)
            if t:
                entry[field_name] = t.group(1)
        entry["has_before"] = "before:" in b
        if "key" in entry:
            out.append(entry)
    return out


@pytest.fixture(scope="module")
def table():
    t = _table()
    assert len(t) >= 10, f"only {len(t)} options parsed — parser broken?"
    return t


# ── The table itself ───────────────────────────────────────────────────


def test_the_table_is_completely_described(table):
    for o in table:
        assert o.get("field_el"), f"{o['key']}: no element suffix"
        assert o.get("kind") in ("toggle", "value"), f"{o['key']}: kind missing"


def test_no_duplicate_keys(table):
    keys = [o["key"] for o in table]
    assert len(keys) == len(set(keys))


def test_no_duplicate_elements(table):
    fields = [o["field_el"] for o in table]
    assert len(fields) == len(set(fields))


# ── The path to the server ───────────────────────────────────────────────


def test_every_option_is_in_the_whitelist(table):
    """Otherwise ``_settings_from`` discards it silently."""
    from whisper_tm.api.queue import _SETTING_KEYS

    missing_ones = [o["key"] for o in table if o["key"] not in _SETTING_KEYS]
    assert not missing_ones, (
        f"These dialog options are missing from api/queue.py::_SETTING_KEYS "
        f"and are SILENTLY dropped by the server: {missing_ones}"
    )


def test_every_option_arrives_at_the_worker(table):
    """Either as a worker argument or as a deliberate post-step.

    Post-steps (``retranscribe``, ``summarize``) must NOT
    reach the worker — they are evaluated in ``core/queue.py`` itself.
    """
    from whisper_tm.core.queue import _normalize
    from whisper_tm.core.transcription import transcription_worker

    post_steps = {"retranscribe", "summarize", "summarize_preset", "summarize_model"}
    opts = _normalize({})
    parameter = set(inspect.signature(transcription_worker).parameters) | {
        "model_name",
        "min_speakers",
        "max_speakers",
        "separation_only",
    }
    for o in table:
        if o["key"] in post_steps:
            continue
        # The key may be renamed along the way (numSpeakers becomes
        # min/max_speakers, model becomes model_name) — then it must at least
        # have a counterpart in _normalize.
        assert o["key"] in opts or o["key"] in ("numSpeakers", "model"), (
            f"{o['key']} does not show up in _normalize() and therefore never reaches the worker"
        )
    assert set(opts) <= parameter, (
        f"_normalize returns keys that transcription_worker does not "
        f"accept — that would be a TypeError in the middle of the queue: "
        f"{set(opts) - parameter}"
    )


# ── The path from configuration ────────────────────────────────────────


def test_preset_options_have_a_config_default(table):
    """A default that points at nothing is an empty promise."""
    import yaml

    cfg = yaml.safe_load((REPO / "config.default.yaml").read_text(encoding="utf-8"))
    text = (JS / "txoptions.js").read_text(encoding="utf-8")
    for o in table:
        if not o["has_before"]:
            continue
        block = re.search(rf"key: '{re.escape(o['key'])}'.*?\}}", text, re.S)
        path_list = re.findall(r"c\.(\w+) \|\| \{\}\)\.(\w+)", block.group(0))
        for section, name in path_list:
            assert section in cfg, f"{o['key']}: section {section} missing"
            assert name in (cfg[section] or {}), (
                f"{o['key']}: config.default.yaml has no "
                f"{section}.{name} — the default would point at nothing"
            )


# ── The collectors are really gone ────────────────────────────────────────


@pytest.mark.parametrize(
    "file,the_function",
    [
        ("transcribe.js", "_gatherTxSettings"),
        ("batch.js", "_batchModalSettings"),
        ("upload.js", "_gatherUploadTxSettings"),
    ],
)
def test_collectors_no_longer_enumerate_anything_themselves(file, the_function):
    """Four enumerations of the same keys had already drifted apart:
    the single-file dialog sent the speaker count as the string ``'auto'``, the
    batch dialog as ``null``."""
    text = (JS / file).read_text(encoding="utf-8")
    m = re.search(rf"function {the_function}\(\) \{{(.*?)\n\}}", text, re.S)
    assert m, f"{the_function} not found"
    assert "txCollect(" in m.group(1), (
        f"{file}::{the_function} enumerates by hand again — the table is thus "
        f"no longer the single source"
    )
    assert "getElementById" not in m.group(1)


def test_upload_passes_through_instead_of_re_enumerating():
    """``_uploadQueueSettings`` was the fourth, most easily overlooked
    copy: if it was forgotten, an option worked everywhere except in the
    upload/link path — and silently so."""
    text = (JS / "upload.js").read_text(encoding="utf-8")
    m = re.search(r"function _uploadQueueSettings\(s\) \{(.*?)\n\}", text, re.S)
    assert m
    body = m.group(1)
    assert "...s" in body or "{...s}" in body
    # at most the post-step keys may still be named explicitly
    names = set(re.findall(r"\bs\.(\w+)", body))
    assert names <= {"summarize", "summarize_preset", "summarize_model"}, (
        f"enumerates options again: {names}"
    )


# ── The speaker option in particular ────────────────────────────────────


def test_speaker_vectors_is_wired_without_gaps(table):
    from whisper_tm.api.queue import _SETTING_KEYS
    from whisper_tm.core.queue import _normalize
    from whisper_tm.core.transcription import transcription_worker

    assert any(o["key"] == "speaker_vectors" for o in table)
    assert "speaker_vectors" in _SETTING_KEYS
    assert "speaker_vectors" in _normalize({})
    assert "speaker_vectors" in inspect.signature(transcription_worker).parameters


def test_speaker_vectors_follows_the_config(monkeypatch):
    from whisper_tm.config import config
    from whisper_tm.core.queue import _normalize

    monkeypatch.setitem(config._data.setdefault("transcription", {}), "speaker_vectors", False)
    assert _normalize({})["speaker_vectors"] is False
    # an explicit yes from the dialog beats the config
    assert _normalize({"speaker_vectors": True})["speaker_vectors"] is True


def test_an_explicit_no_is_not_overridden(monkeypatch):
    """``bool("false")`` is ``True`` — hence ``parse_bool``."""
    from whisper_tm.config import config
    from whisper_tm.core.queue import _normalize

    monkeypatch.setitem(config._data.setdefault("transcription", {}), "speaker_vectors", True)
    assert _normalize({"speaker_vectors": "false"})["speaker_vectors"] is False
    assert _normalize({"speaker_vectors": False})["speaker_vectors"] is False


def test_the_dialog_wiring_does_not_lag_behind_the_scan():
    """The listeners must be attached BEFORE ``await scanFiles()``.

    They used to come after it. The scan "can take minutes with a
    large library and a busy NAS" (per the comment in that spot) — and for
    that long, every checkbox in the transcription dialog was
    a dead control: Vocal Isolation didn't unfold the separation model,
    the warning next to the voice vectors didn't appear. None of it did
    anything; nothing happened at all.

    On a warm library it doesn't stand out — hence this
    test instead of a note.
    """
    app_js = (REPO / "whisper_tm" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    wiring = app_js.index("TX_OPTION_PREFIXES.forEach(")
    # The INSTRUCTION, not the mention: the comment in that spot
    # names `await scanFiles()` without a semicolon — matching that would
    # report an error that isn't one.
    scan = app_js.index("await scanFiles();")
    assert wiring < scan, (
        "The option toggles are only wired up after the directory scan "
        "— until then the dialog reacts to nothing"
    )

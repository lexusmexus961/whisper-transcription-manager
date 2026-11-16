"""Speaker names for the meaning search — looked up when a search runs.

The index holds what the chunker wrote: diarization labels (``[SPEAKER_04]
What is this?``). Who a label is lives in the voice library and in the
transcript editor, and it changes whenever a voice is named — so the names
are put in at search time instead of into the index (nothing is embedded
again).

A person named (voice map, transcript editor) in many recordings speaks in
many passages — and most of them do not contain their name. Those reached
"<person> talks about money" only when their wording happened to match, and
where they did, the reranker and the language model read ``[SPEAKER_04]``
and could not tell who was talking.

Two uses:

* Passages read with names — ``[Alice] What is this?`` — by the reranker, by
  the deep search's check, and on screen (:func:`named`).
* A candidate list per person the query names: the passages in which that
  person speaks (:func:`labels_of`, rag/retrieve.voice_ids).

Sources, per recording: the names typed in the transcript editor
(``transcripts.speaker_names``) first, then the voice map — the name most of
a label's named windows carry (SpeakerDB.map_names; the recognition part of
core/speaker.names_for_labels compares every recording with the gallery and
is left out here). Held for the process; built again when the voice library
changes (``speaker_db().generation``), when a name is typed in a transcript
(:func:`forget`), and after :data:`MAX_AGE_SEC` — 0.5 s per build.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

logger = logging.getLogger(__name__)

#: The held names are built again after this long at the latest (writes
#: from outside this process — a script — move no generation).
MAX_AGE_SEC = 600.0

#: A speaker label in a chunk's text, as the chunker writes it.
LABEL = re.compile(r"\[([^\]\n]{1,60})\]")
_WORD = re.compile(r"[^\W_]+")

_lock = threading.Lock()
#: ``epoch`` counts forget() calls: a build that started before one is not kept
_held: dict = {"at": -1e18, "gen": None, "index": None, "epoch": 0}


def _fold(name: str) -> str:
    from whisper_tm.database.speaker import fold_name

    return fold_name(name) or ""


def _clean(name) -> str:
    """A name as it may stand in brackets: one line, no brackets."""
    return " ".join(re.sub(r"[\[\]]", " ", str(name or "")).split())[:60]


def _typed(raw) -> dict[str, str]:
    """``transcripts.speaker_names`` — the editor's ``{label: name}`` JSON."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for label, name in data.items():
        name = _clean(name)
        if isinstance(label, str) and label and name and name != label:
            out[label] = name
    return out


def _generation():
    try:
        from whisper_tm.database.speaker import speaker_db

        db = speaker_db()
        return (db._db_path, db.generation)
    except Exception:  # noqa: BLE001 — no voice library: nothing to follow
        return None


def _build() -> dict:
    names: dict[str, dict[str, str]] = {}
    try:
        from whisper_tm.database.speaker import speaker_db

        for path, labels in speaker_db().map_names().items():
            names[path] = {label: _clean(n) for label, n in labels.items() if _clean(n)}
    except Exception as e:  # noqa: BLE001 — a search without names still works
        logger.info("Search: names of the voice map not readable: %s", e)
    typed = 0
    try:
        from whisper_tm.extensions import db

        if db:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT abs_path, rel_path, speaker_names FROM transcripts "
                    "WHERE speaker_names IS NOT NULL "
                    "AND speaker_names NOT IN ('', '{}')"
                ).fetchall()
            for abs_path, rel_path, raw in rows:
                given = _typed(raw)
                if not given:
                    continue
                typed += 1
                # The index knows a transcript by abs_path; rows from older
                # versions by rel_path (rag/retrieve.display_paths)
                for key in {abs_path, rel_path} - {"", None}:
                    names[key] = {**names.get(key, {}), **given}
    except Exception as e:  # noqa: BLE001
        logger.info("Search: names typed in transcripts not readable: %s", e)
    people: dict[str, dict[str, set[str]]] = {}
    shown: dict[str, str] = {}
    for key, labels in names.items():
        for label, name in labels.items():
            f = _fold(name)
            people.setdefault(f, {}).setdefault(key, set()).add(label)
            shown.setdefault(f, name)
    logger.info(
        "Search: speaker names for %d recordings (%d typed in the editor), %d people",
        len(names),
        typed,
        len(people),
    )
    return {"names": names, "people": people, "shown": shown}


def index() -> dict:
    """``{"names": {key: {label: name}}, "people": {folded name: {key:
    {labels}}}, "shown": {folded name: name}}`` — held, see the module."""
    gen = _generation()
    now = time.monotonic()
    with _lock:
        held = _held["index"]
        if held is not None and _held["gen"] == gen and now - _held["at"] < MAX_AGE_SEC:
            return held
        epoch = _held["epoch"]
    built = _build()
    with _lock:
        # A name typed while this was read is maybe not in it: not kept then
        if _held["epoch"] == epoch:
            _held.update(at=now, gen=gen, index=built)
    return built


def forget() -> None:
    """Build again on the next search — a name was typed in a transcript."""
    with _lock:
        _held.update(at=-1e18, gen=None, index=None, epoch=_held["epoch"] + 1)


def names_for(keys) -> dict[str, dict[str, str]]:
    """``{key: {label: name}}`` for the transcript keys that have names."""
    names = index()["names"]
    return {k: names[k] for k in dict.fromkeys(keys) if k in names}


def named(content: str, names: dict[str, str] | None) -> str:
    """``content`` with each known label replaced by its name:
    ``[SPEAKER_04] What is this?`` → ``[Alice] What is this?``."""
    if not names or not content:
        return content or ""
    return LABEL.sub(
        lambda m: f"[{names[m.group(1)]}]" if m.group(1) in names else m.group(0), content
    )


def name_rows(rows, *, key="transcript_key", fields=("content",)) -> None:
    """Name the speakers in ``rows`` (dicts) in place: each of ``fields``,
    and ``speaker_name`` for the row's dominant ``speaker``."""
    rows = [r for r in rows if r]
    if not rows:
        return
    try:
        names = names_for(r.get(key) for r in rows)
    except Exception as e:  # noqa: BLE001 — labels stay labels then
        logger.warning("Search: speaker names not available: %s", e)
        return
    for r in rows:
        given = names.get(r.get(key))
        if not given:
            continue
        for f in fields:
            if isinstance(r.get(f), str):
                r[f] = named(r[f], given)
        sp = r.get("speaker")
        if sp and sp in given:
            r["speaker_name"] = given[sp]


def people_named(tokens) -> list[str]:
    """The people (names as stored) whose name has one of ``tokens`` as a
    word — the query's name words (rag/retrieve.query_terms ``names``)."""
    want = {str(t).lower() for t in tokens or () if t}
    if not want:
        return []
    shown = index()["shown"]
    return sorted(
        (name for name in shown.values() if want & set(_WORD.findall(name.lower()))), key=str.lower
    )


def labels_of(person: str) -> dict[str, set[str]]:
    """``{transcript key: {labels}}`` — where ``person`` speaks."""
    return index()["people"].get(_fold(person), {})

"""Searching "Images & sounds": one query, every kind ranked on its own.

A query meets a still and a sound on different scales (cosines around 0.7
for a good still, 0.75 for a good sound, 0.8 for text on text), so one list
would favour one kind — each kind gets its own list.

The query is a text, a hit of the index ("Similar" — its own vector asks),
or a file someone drops in (media/service.py embeds it, optionally together
with a text). A text query asks the sounds with its English sound label
(media/llm.py) when the language model is there.

Every hit carries the closest labels of a fixed set (zero-shot tags), what
is said around it (speaker names put in at search time, as the meaning
search does), and where its recording lies (``file_dir`` + ``filename``,
the way the search view plays and opens hits).
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

LABELS = (
    "speech",
    "music",
    "singing",
    "laughter",
    "shouting",
    "applause",
    "gunshots",
    "explosion",
    "siren",
    "dog",
    "car",
    "engine",
    "rain",
    "wind",
    "birds",
    "crowd",
    "silence",
    "telephone",
    "video game",
    "cartoon",
    "outdoors",
    "indoors",
    "night",
    "webcam stream",
    "computer screen",
    "text on screen",
    "police",
    "food",
    "animals",
    "street",
    "nature",
    "tv show",
    "news",
    "sports",
    "party",
)
#: What a sound label replaces — the kinds that hear rather than see
HEARING = ("audio",)

_tags: dict = {"vectors": None}
_tags_lock = threading.Lock()
_files_cache: dict = {}


def embed_text(text: str, prompt: str = "SearchQuery") -> np.ndarray:
    from whisper_tm.core.media_server import SERVER
    from whisper_tm.media.service import unb64

    res = SERVER.call("/embed", {"texts": [text], "prompt": prompt}, timeout=60)
    return unb64(res["vectors"][0])


def _tag_vectors() -> np.ndarray | None:
    with _tags_lock:
        if _tags["vectors"] is not None:
            return _tags["vectors"]
    try:
        from whisper_tm.core.media_server import SERVER
        from whisper_tm.media.service import unb64

        res = SERVER.call("/embed", {"texts": list(LABELS), "prompt": "SearchQuery"}, timeout=120)
        vecs = np.stack([unb64(v) for v in res["vectors"]])
    except Exception as e:  # noqa: BLE001 — tags are an extra
        logger.info("Media search: no tag vectors: %s", e)
        return None
    with _tags_lock:
        _tags["vectors"] = vecs
    return vecs


def _tags_of(vec: np.ndarray, tags: np.ndarray | None) -> list:
    if tags is None:
        return []
    from whisper_tm.media.store import unit

    d = len(vec)
    s = unit(tags, d) @ unit(vec, d)
    return [[LABELS[k], round(float(s[k]), 3)] for k in np.argsort(-s)[:3]]


def _places(paths: set[str]) -> dict[str, tuple[str, str]]:
    """``{abs path: (file_dir, filename)}`` — the root a hit is relative
    to, from the transcripts (database._root_of, as the meaning search)."""
    from whisper_tm.database import _root_of
    from whisper_tm.extensions import db

    out = {}
    want = [p for p in paths if p not in _files_cache]
    if want and db:
        with db.connection() as conn:
            for k in range(0, len(want), 500):
                part = want[k : k + 500]
                for abs_path, rel, media_dir in conn.execute(
                    f"SELECT abs_path, rel_path, media_dir FROM transcripts WHERE abs_path IN "
                    f"({','.join('?' * len(part))})",
                    part,
                ):
                    root = _root_of(rel, abs_path, media_dir) or os.path.dirname(abs_path)
                    rel_path = (
                        rel if rel and not os.path.isabs(rel) else os.path.relpath(abs_path, root)
                    )
                    _files_cache[abs_path] = (root, rel_path)
    for p in paths:
        out[p] = _files_cache.get(p) or (os.path.dirname(p), os.path.basename(p))
    if len(_files_cache) > 50_000:
        _files_cache.clear()
    return out


def search(
    *,
    vector: np.ndarray | None = None,
    text: str = "",
    like: int | None = None,
    kinds: list[str] | None = None,
    dims: int | None = None,
    limit: int = 48,
    folders: list[str] | None = None,
    files: np.ndarray | None = None,
) -> dict:
    """Rank every kind for one query. Exactly one of ``vector`` (a file
    query), ``like`` (an item id) or ``text``. ``folders`` (path prefixes)
    and ``files`` (the store's file ids) narrow it to those recordings."""
    from whisper_tm.media import llm
    from whisper_tm.media.store import KINDS, get_store

    store = get_store()
    kinds = [k for k in (kinds or KINDS) if k in KINDS]
    t0 = time.monotonic()
    label = None
    exclude = None
    if like is not None:
        it = store.item(int(like))
        if not it:
            raise LookupError("No such item")
        q = it["vector"]
        exclude = it["id"]
        hear = q
    elif vector is not None:
        q = hear = np.asarray(vector, dtype=np.float32)
    else:
        q = embed_text(text)
        hear = q
        if set(kinds) & set(HEARING):
            label = llm.sound_label(text)
            if label and label.lower() != text.strip().lower():
                hear = embed_text(label)
    t1 = time.monotonic()
    if folders:
        from whisper_tm.core import paths

        under = store.file_ids_under([(paths.canon(f) or f).rstrip("/") + "/" for f in folders])
        files = under if files is None else np.intersect1d(under, np.asarray(files, dtype=np.int64))
    if files is not None and not len(files):
        return {
            "results": {k: [] for k in kinds},
            "sound_label": None,
            "embed_ms": round((time.monotonic() - t0) * 1000),
            "rank_ms": 0,
            "dims": min(dims or 768, store.dims() or 768),
        }
    results: dict[str, list] = {}
    tags = _tag_vectors()
    found: list[tuple[str, int, int, float, np.ndarray]] = []
    for kind in kinds:
        qq = hear if kind in HEARING else q
        for iid, mid, score, vec in store.rank(
            qq, kind, limit, dims=dims, files=files, exclude=exclude
        ):
            found.append((kind, iid, mid, score, vec))
    moments = {mid: store.moment(mid) for _, _, mid, _, _ in found}
    places = _places({m["path"] for m in moments.values() if m})
    names = _speaker_names({m["path"] for m in moments.values() if m})
    for kind, iid, mid, score, vec in found:
        m = moments.get(mid)
        if not m:
            continue
        root, rel = places[m["path"]]
        text_said = m["text"] or ""
        if names.get(m["path"]):
            from whisper_tm.rag import speakers

            text_said = speakers.named(text_said, names[m["path"]])
        results.setdefault(kind, []).append(
            {
                "id": iid,
                "kind": kind,
                "moment": mid,
                "score": round(score, 4),
                "t": m["t"],
                "start": m["start"],
                "dur": m["dur"],
                "repeat": bool(m["repeat"]),
                "name": os.path.basename(m["path"]),
                "path": m["path"],
                "file_dir": root,
                "filename": rel,
                "video": bool(m["video"]),
                "thumb": f"/api/media-search/thumb/{mid}" if m["thumb"] else None,
                "clip": f"/api/media-search/clip/{mid}",
                "text": text_said,
                "tags": _tags_of(vec, tags) if vec is not None else [],
            }
        )
    for kind in kinds:
        results.setdefault(kind, [])
    return {
        "results": results,
        "sound_label": label,
        "embed_ms": round((t1 - t0) * 1000),
        "rank_ms": round((time.monotonic() - t1) * 1000),
        "dims": min(dims or 768, store.dims() or 768),
    }


def _speaker_names(paths: set[str]) -> dict[str, dict]:
    try:
        from whisper_tm.rag import speakers

        return speakers.names_for(paths)
    except Exception as e:  # noqa: BLE001 — labels stay labels
        logger.debug("Media search: speaker names: %s", e)
        return {}

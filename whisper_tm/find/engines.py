"""Find's searches — each returns ``{"items", "count", "query", "note"}`` or
``{"skip": why}``; an item is one hit: a recording (``key``, its absolute
path) and a moment in it (``t``).

* ``words``    the full-text index, best bm25 first: each phrase as said, then
  its words alone — rare ones lead (a misheard name still shares its rare
  words with what was said)
* ``meaning``  the situation search (rag/retrieve.quick_search); the agent asks
  it with the description itself — before the plan is there — and with the
  plan's sentence
* ``seen``, ``heard``  Images & sounds
* ``files``    the searches above INSIDE the files whose name has a remembered
  piece ("-WA") — a hint finds moments of its own, not only lifts the global
  lists' (lifting a large set of matching files alone rarely puts one
  among the best)
* ``people``   the same inside what the named people say (the voice library)

The words are ranked by bm25, not in the order of the file paths (the text
search's own order), where "2025-…" would beat "VID-…-WA…" by name alone.
"""

from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

#: Hits per search kept and shown
PER_ENGINE = 60
#: Matches counted at most (a count, not a list)
COUNT_CAP = 10_000
#: Kinds of Images & sounds that see — and that hear
SEEN_KINDS = ("image", "description", "video", "videoaudio")
HEARD_KINDS = ("audio",)
RRF_K = 20


def _item(
    key,
    *,
    t=0.0,
    end=None,
    text="",
    score=None,
    rank=0,
    video=None,
    thumb=None,
    clip=None,
    rel=None,
    root=None,
    kind="",
    extra=None,
) -> dict:
    return {
        "key": key,
        "name": os.path.basename(key or ""),
        "rel": rel,
        "dir": root,
        "t": round(float(t or 0.0), 2),
        "end": round(float(end), 2) if end is not None else None,
        "text": text or "",
        "score": None if score is None else round(float(score), 4),
        "rank": rank,
        "video": video,
        "thumb": thumb,
        "clip": clip,
        "kind": kind,
        "extra": extra or {},
    }


class Ctx:
    """What a run's searches share: the library, and every text embedded
    once (the meaning search and its scoped twins ask with the same vector)."""

    def __init__(self, db):
        self.db = db
        self._lock = threading.Lock()
        self._qvec: dict[str, list] = {}
        self._mvec: dict[str, object] = {}
        self._rag = None
        self._embed = None

    def rag(self):
        with self._lock:
            if self._rag is None:
                from whisper_tm.rag.service import make_rag_db

                self._rag = make_rag_db()
            return self._rag

    def query_vectors(self, texts):
        """The meaning search's embedding (with its instruction prefix), each
        text once."""
        out = []
        for t in texts:
            with self._lock:
                got = self._qvec.get(t)
            if got is None:
                if self._embed is None:
                    from whisper_tm.rag.service import make_query_embed_fn

                    self._embed = make_query_embed_fn()
                got = self._embed([t])[0]
                with self._lock:
                    self._qvec[t] = got
            out.append(got)
        return out

    def media_vector(self, text: str):
        """Images & sounds' embedding of a text, once."""
        with self._lock:
            got = self._mvec.get(text)
        if got is None:
            from whisper_tm.media import search as msearch

            got = msearch.embed_text(text)
            with self._lock:
                self._mvec[text] = got
        return got


def places(keys, db) -> dict:
    """``{abs_path: (relative path, root)}`` — what the player and the
    transcript open (rag/retrieve.display_paths)."""
    from whisper_tm.rag import retrieve

    return {
        k: (p.get("rel_path"), p.get("file_dir"))
        for k, p in retrieve.display_paths(list(keys), db).items()
    }


def rrf(lists: list[list[dict]], kind: str, k: int = PER_ENGINE) -> list[dict]:
    """Several hit lists as one: reciprocal ranks per (recording, ~moment)."""
    merged: dict[tuple, dict] = {}
    for items in lists:
        for rank, it in enumerate(items):
            sig = (it["key"], round(float(it["t"] or 0) / 5))
            cur = merged.get(sig)
            if cur is None:
                cur = merged[sig] = {"it": dict(it, kind=kind), "s": 0.0, "via": []}
            cur["s"] += 1.0 / (RRF_K + rank)
            via = (it.get("extra") or {}).get("via") or it.get("kind")
            if via and via not in cur["via"]:
                cur["via"].append(via)
    out = []
    for n, x in enumerate(sorted(merged.values(), key=lambda x: -x["s"])[:k]):
        it = x["it"]
        it["rank"] = n
        it["extra"] = dict(it.get("extra") or {}, via=x["via"])
        out.append(it)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Where a search looks — everything, or inside a scope
# ══════════════════════════════════════════════════════════════════════════


def hint_sql(hint: str, columns) -> tuple[str, list]:
    """``(sql, args)`` matching a file-name piece in any of ``columns``. A
    piece with capitals is matched as written ("-WA" are WhatsApp files; "-wa"
    would be every "-walk" too: far more files than the WhatsApp ones)."""
    if hint != hint.lower():
        pattern = "*" + re.sub(r"([*?\[])", r"[\1]", hint) + "*"
        return "(" + " OR ".join(f"{c} GLOB ?" for c in columns) + ")", [pattern] * len(columns)
    like = "%" + hint.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    return (
        "(" + " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for c in columns) + ")",
        [like] * len(columns),
    )


class Scope:
    """Where a scoped search looks: the files whose path has a hint
    (``hint``), or the passages the named people speak (``voices``:
    ``[(abs_path, speaker label)]``, ``people``: their names)."""

    def __init__(self, label: str, *, hint: str | None = None, voices=None, people=None):
        self.label = label
        self.hint = hint
        self.voices = list(voices or [])
        self.people = list(people or [])
        self.paths = {p for p, _ in self.voices}

    def words_sql(self) -> tuple[str, list, list]:
        """``(condition, args, setup statements)`` for the segment search."""
        if self.hint:
            sql, args = hint_sql(self.hint, ("t.rel_path", "t.abs_path"))
            return sql, args, []
        return (
            "EXISTS (SELECT 1 FROM temp.find_voices v WHERE v.abs_path = t.abs_path "
            "AND v.label = s.speaker)",
            [],
            self.voices,
        )

    def chunk_ids(self, ctx: Ctx) -> list[int]:
        """The semantic index's passages inside the scope."""
        rag = ctx.rag()
        if self.hint:
            sql, args = hint_sql(self.hint, ("transcript_key",))
            with rag.connection() as c:
                return [r[0] for r in c.execute(f"SELECT id FROM {rag.TABLE} WHERE {sql}", args)]
        from whisper_tm.rag import retrieve

        ids: list[int] = []
        for got in retrieve.voice_ids(rag, self.people).values():
            ids.extend(got)
        return ids

    def media_files(self, hint_paths=None):
        """Images & sounds' file ids inside the scope."""
        from whisper_tm.media.store import get_store

        return get_store().file_ids_of(hint_paths if self.hint else self.paths)


# ══════════════════════════════════════════════════════════════════════════
#  Words
# ══════════════════════════════════════════════════════════════════════════


def fts_rows(db, expression: str, k: int, scope: Scope | None = None) -> tuple[list[dict], int]:
    """Segments matching an FTS5 expression, best bm25 first: ``(rows,
    matches — counted up to COUNT_CAP)``."""
    if not expression:
        return [], 0
    where = "WHERE segments_fts MATCH ? AND t.complete = 1"
    args: list = [expression]
    setup: list = []
    if scope is not None:
        cond, more, setup = scope.words_sql()
        where += " AND " + cond
        args += more
    base = (
        "FROM segments_fts JOIN segments s ON s.id = segments_fts.rowid "
        "JOIN transcripts t ON t.id = s.transcript_id " + where
    )
    import sqlite3

    with db.connection() as conn:
        if setup:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS find_voices (abs_path TEXT, label TEXT)")
            conn.execute("DELETE FROM temp.find_voices")
            conn.executemany("INSERT INTO temp.find_voices VALUES (?, ?)", setup)
        try:
            rows = conn.execute(
                "SELECT s.start_sec, s.end_sec, s.speaker, s.text, t.abs_path, t.rel_path, "
                "bm25(segments_fts) AS bm " + base + " ORDER BY bm LIMIT ?",
                args + [int(k)],
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 {base} LIMIT {COUNT_CAP})", args
            ).fetchone()[0]
        except sqlite3.OperationalError as e:
            logger.warning("Find: FTS expression rejected (%s): %s", expression[:200], e)
            return [], 0
        finally:
            if setup:
                conn.execute("DELETE FROM temp.find_voices")
    return [
        {
            "start": r[0],
            "end": r[1],
            "speaker": r[2],
            "text": r[3],
            "key": r[4],
            "rel": r[5],
            "bm": r[6],
        }
        for r in rows
    ], total


def words_queries(phrases: list[str]) -> list[tuple[str, str]]:
    """``[(label, FTS5 expression)]``: each phrase as said, then any of its
    words (stopwords out, long ones by their stem)."""
    from whisper_tm.rag import retrieve

    out = []
    for p in phrases:
        exact = retrieve.fts_term(p)
        if exact:
            out.append((p, exact))
        terms = retrieve.query_terms(p, known_names=())
        words = terms["must"] + terms["words"]
        if len(words) > 1:
            out.append((" | ".join(words), retrieve.fts_or(words)))
    return out


def word_items(rows_lists, db, kind: str) -> list[dict]:
    """FTS rows (several lists, best first each) as items — each moment once."""
    seen, items = set(), []
    for rows in rows_lists:
        for r in rows:
            sig = (r["key"], round(float(r["start"] or 0)))
            if sig in seen:
                continue
            seen.add(sig)
            items.append(
                _item(
                    r["key"],
                    t=r["start"],
                    end=r["end"],
                    text=r["text"] or "",
                    score=-float(r["bm"] or 0),
                    rel=r["rel"],
                    kind=kind,
                    extra={"phrase": r.get("phrase"), "speaker": r.get("speaker"), "via": "words"},
                )
            )
    pl = places({it["key"] for it in items}, db)
    for k, it in enumerate(items):
        it["rank"] = k
        rel, root = pl.get(it["key"], (it["rel"], None))
        it["rel"], it["dir"] = rel or it["rel"], root
    return items


def search_words(
    plan: dict, ctx: Ctx, scope: Scope | None = None, phrases: list[str] | None = None
) -> dict:
    phrases = plan["said_words"] if phrases is None else phrases
    if not phrases:
        return {"skip": "no words to look for"}
    lists, tried, total = [], [], 0
    for label, expr in words_queries(phrases):
        rows, n = fts_rows(ctx.db, expr, PER_ENGINE, scope)
        for r in rows:
            r["phrase"] = label
        lists.append(rows)
        tried.append(label)
        total = max(total, n)
    items = word_items(lists, ctx.db, "words")[:PER_ENGINE]
    return {
        "items": items,
        "count": total,
        "query": " · ".join(phrases),
        "note": "best match first (bm25) — the phrase, then its words alone",
    }


# ══════════════════════════════════════════════════════════════════════════
#  Meaning
# ══════════════════════════════════════════════════════════════════════════


def _meaning_items(results: list[dict], db, kind: str = "meaning") -> list[dict]:
    from whisper_tm.core import copies
    from whisper_tm.find.fusion import dup_of, is_video
    from whisper_tm.rag import retrieve

    retrieve.place_copies(results, db)
    facts = copies.facts(
        db, [c.get("transcript_key") for r in results for c in r.get("copies") or []]
    )
    items = []
    for k, r in enumerate(results):
        it = _item(
            r.get("transcript_key"),
            t=r.get("start_sec"),
            end=r.get("end_sec"),
            text=r.get("content") or "",
            score=r.get("score"),
            rank=k,
            rel=r.get("rel_path"),
            root=r.get("file_dir"),
            kind=kind,
            extra={
                "relevance": r.get("relevance"),
                "via": "meaning",
                "speaker": r.get("speaker_name") or r.get("speaker"),
            },
        )
        # the semantic search folds copies itself — they are this hit's duplicates
        it["dups"] = [
            dup_of(
                _item(
                    c["transcript_key"],
                    t=c.get("start_sec"),
                    end=c.get("end_sec"),
                    rel=c.get("rel_path"),
                    root=c.get("file_dir"),
                    video=is_video(c["transcript_key"]),
                ),
                "the same passage",
                facts,
            )
            for c in r.get("copies") or []
            if c.get("transcript_key")
        ]
        items.append(it)
    return items


def search_meaning(text: str, ctx: Ctx, rerank: bool = True) -> dict:
    """The situation search for ``text``. ``rerank=False`` for the first,
    quick look: the reranker shares the GPU with the queue and can take close
    to a minute under load — the description's hits come in seconds without
    it, the plan's sentence brings the reranked list."""
    if not text:
        return {"skip": "nothing said to look for by meaning"}
    from whisper_tm.rag import retrieve
    from whisper_tm.rag.service import embedding_ready

    ok, msg = embedding_ready()
    if not ok:
        return {"skip": f"semantic search not ready: {msg}"}
    out = retrieve.quick_search(
        text,
        k=PER_ENGINE,
        mode="meaning",
        rerank=rerank,
        group=True,
        db=ctx.rag(),
        embed_fn=ctx.query_vectors,
    )
    items = _meaning_items(out["results"], ctx.db)
    return {
        "items": items,
        "count": out.get("candidates") or len(items),
        "query": text,
        "note": out.get("note"),
    }


def scoped_meaning(text: str, ctx: Ctx, scope: Scope) -> list[dict]:
    """The passages inside ``scope`` closest in meaning to ``text``."""
    from whisper_tm.rag.service import embedding_ready

    if not text or not embedding_ready()[0]:
        return []
    ids = scope.chunk_ids(ctx)
    if not ids:
        return []
    qvec = ctx.query_vectors([text])[0]
    rows = ctx.rag().search_vector(qvec, k=PER_ENGINE, among=ids)
    for r in rows:
        r.setdefault("copies", [])
    return _meaning_items(rows, ctx.db, kind="meaning")


# ══════════════════════════════════════════════════════════════════════════
#  Seen and heard (Images & sounds)
# ══════════════════════════════════════════════════════════════════════════


def media_ready() -> str | None:
    """Why Images & sounds can't answer — None when it can."""
    try:
        from whisper_tm.core.media_server import enabled
        from whisper_tm.media import environment
        from whisper_tm.media.store import get_store

        if not enabled():
            return "Images & sounds is switched off"
        if not environment.check()["ok"]:
            return "Images & sounds isn't installed"
        if not sum(get_store().counts()["kinds"].values()):
            return "the Images & sounds index is empty"
    except Exception as e:  # noqa: BLE001
        return f"Images & sounds not available: {e}"
    return None


def _media_items(results: dict, kinds: tuple, kind: str) -> list[dict]:
    """Media hits of several kinds as one list: reciprocal ranks per moment."""
    by_moment: dict[int, dict] = {}
    for kd in kinds:
        for rank, h in enumerate(results.get(kd) or []):
            cur = by_moment.get(h["moment"])
            s = 1.0 / (RRF_K + rank)
            if cur is None:
                by_moment[h["moment"]] = {"h": h, "s": s, "kinds": [kd]}
            else:
                cur["s"] += s
                cur["kinds"].append(kd)
    ranked = sorted(by_moment.values(), key=lambda x: -x["s"])
    items = []
    for k, x in enumerate(ranked[:PER_ENGINE]):
        h = x["h"]
        items.append(
            _item(
                h["path"],
                t=h["t"] if h.get("video") else h["start"],
                end=(h["start"] + h["dur"]),
                text=h.get("text") or "",
                score=h.get("score"),
                rank=k,
                video=bool(h.get("video")),
                thumb=h.get("thumb"),
                clip=h.get("clip"),
                rel=h.get("filename"),
                root=h.get("file_dir"),
                kind=kind,
                extra={
                    "moment": h["moment"],
                    "kinds": x["kinds"],
                    "via": kind,
                    "tags": [t[0] for t in (h.get("tags") or [])][:3],
                },
            )
        )
    return items


def media_search(text: str, kinds: tuple, kind: str, ctx: Ctx, files=None) -> list[dict]:
    from whisper_tm.media import search as msearch

    out = msearch.search(
        vector=ctx.media_vector(text), kinds=list(kinds), limit=PER_ENGINE, files=files
    )
    return _media_items(out["results"], kinds, kind)


def search_seen(plan: dict, ctx: Ctx) -> dict:
    q = plan["seen"]
    if not q:
        return {"skip": "nothing visual described"}
    why = media_ready()
    if why:
        return {"skip": why}
    items = media_search(q, SEEN_KINDS, "seen", ctx)
    return {"items": items, "count": len(items), "query": q}


def search_heard(plan: dict, ctx: Ctx) -> dict:
    q = plan["heard"]
    if not q:
        return {"skip": "no sound described"}
    why = media_ready()
    if why:
        return {"skip": why}
    items = media_search(q, HEARD_KINDS, "heard", ctx)
    return {"items": items, "count": len(items), "query": q}


# ══════════════════════════════════════════════════════════════════════════
#  Inside a scope: the files a name piece points to, what people say
# ══════════════════════════════════════════════════════════════════════════


def file_hint_paths(db, hint: str) -> tuple[int, set]:
    """``(count, abs paths)`` of the files whose path or title has ``hint`` —
    every one of them (a boost needs the whole set)."""
    sql, args = hint_sql(hint, ("rel_path", "media_title"))
    with db.connection() as conn:
        rows = conn.execute(f"SELECT abs_path FROM media_files WHERE {sql}", args).fetchall()
    paths = {r[0] for r in rows if r[0]}
    return len(paths), paths


def scoped(plan: dict, ctx: Ctx, scope: Scope, kind: str, hint_paths=None) -> tuple[list, list]:
    """Every content search inside ``scope``, fused: ``(items, what was
    searched)``. A search failing leaves the others."""
    lists, asked = [], []
    phrases = list(plan["said_words"])
    if not phrases and plan.get("said_meaning"):
        phrases = [plan["said_meaning"]]  # inside a scope, the sentence's words too
    jobs = [
        ("words", lambda: search_words(plan, ctx, scope, phrases).get("items") or []),
        ("meaning", lambda: scoped_meaning(plan.get("said_meaning") or "", ctx, scope)),
    ]
    if (plan["seen"] or plan["heard"]) and not media_ready():
        files = scope.media_files(hint_paths)
        if len(files):
            if plan["seen"]:
                jobs.append(
                    ("seen", lambda: media_search(plan["seen"], SEEN_KINDS, "seen", ctx, files))
                )
            if plan["heard"]:
                jobs.append(
                    ("heard", lambda: media_search(plan["heard"], HEARD_KINDS, "heard", ctx, files))
                )
    for name, fn in jobs:
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001 — one failing leaves the others
            logger.info("Find: %s inside %s failed: %s", name, scope.label, e)
            continue
        if got:
            lists.append(got)
            asked.append(name)
    return rrf(lists, kind), asked


def search_files(plan: dict, ctx: Ctx) -> dict:
    hints = plan["file_hints"]
    if not hints:
        return {"skip": "no file name remembered"}
    paths: set = set()
    counts, items_lists, asked_all = [], [], set()
    for h in hints:
        n, ps = file_hint_paths(ctx.db, h)
        counts.append(f"“{h}”: {n:,} files")
        paths |= ps
        if n:
            items, asked = scoped(plan, ctx, Scope(h, hint=h), "files", hint_paths=ps)
            items_lists.append(items)
            asked_all.update(asked)
    if len(items_lists) > 1:
        items = rrf(items_lists, "files")
    else:
        items = items_lists[0] if items_lists else []
    note = " · ".join(counts) + (
        f" — searched inside them: {', '.join(sorted(asked_all))}" if asked_all else ""
    )
    return {
        "items": items,
        "count": len(paths),
        "query": " · ".join(hints),
        "note": note,
        "paths": paths,
    }


def search_people(plan: dict, ctx: Ctx) -> dict:
    names = plan["people"]
    if not names:
        return {"skip": "nobody named"}
    from whisper_tm.database.speaker import speaker_db

    sdb = speaker_db()
    voices, found, per_person = [], [], {}
    for name in names:
        spots = sdb.person_spots(name)
        if spots:
            voices.extend(spots)
            per_person[name] = {p for p, _ in spots}
            found.append(f"{name}: {len(per_person[name]):,} recordings")
    if not voices:
        return {"skip": "nobody of them in the voice library"}
    scope = Scope(", ".join(names), voices=voices, people=list(per_person))
    items, asked = scoped(plan, ctx, scope, "people")
    note = " · ".join(found) + (
        f" — searched in what they say: {', '.join(asked)}" if asked else ""
    )
    return {
        "items": items,
        "count": len(scope.paths),
        "query": " · ".join(names),
        "note": note,
        "paths": scope.paths,
        "per_person": per_person,
    }

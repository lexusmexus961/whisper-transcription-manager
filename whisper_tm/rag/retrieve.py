"""Situation search: candidates by meaning and by wording, copies collapsed,
the best ones reranked.

The goal: find everything, obscure situations too ("alice gets a compliment",
"someone apologizes for being late") — without junk in between. Measured on a
large real index (Qwen3-Embedding-4B) against careful gold labels:

* The vector search alone puts junk on top (P@10 0.40) and spreads the right
  passages far down its ranking — its best 100 hold only 24–33 % of them,
  its best 1,000 76–89 %.
* Full text with the name AND the action words is precise where a name is
  involved (P@10 0.80); fusing it with the vector list (RRF) keeps that.
* About a third of the candidates were copies of the same recording in other folders
  (backups) — five hits of one passage are one hit.
* The 4B reranker orders the fused candidates well (P@10 1.00 / 0.60);
  core/rerank_server.py runs it.

So a search here: the dense top k (the question with its instruction prefix,
rag/service.make_query_embed_fn) + full text "must terms AND any content
word" + full text "any content word" + for each person the query names, the
passages that person speaks in, ranked by the question's meaning (the voice
library knows their label — rag/speakers.py) → Reciprocal Rank Fusion
(k=60) → copies collapsed (same text, or vectors at cosine ≥ 0.97) → the
speakers' names put in (``[Alice]`` instead of ``[SPEAKER_04]``) → the best
groups reranked. Everything up to the reranker is pure and runs without a
model.

The deep search ("find everything", rag/deep_search.py) uses the same parts
with wider lists and a language model on top.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

#: Reciprocal Rank Fusion constant (as rag.rrf_k).
RRF_K = 60
#: Two candidates are copies at this cosine between their stored vectors.
COPY_COSINE = 0.97
#: ... or when their normalised text agrees in its first this many characters
COPY_TEXT_HEAD = 300
#: ... and its last this many (a shared intro does not make two passages one)
COPY_TEXT_TAIL = 100
#: List sizes of a quick search.
QUICK_DENSE = 300
QUICK_FTS = 300
QUICK_VOICE = 300
#: A person speaking in more passages than this gets no list of their own —
#: it would hold most of the library.
MAX_VOICE_PASSAGES = 400_000
#: Query words shorter than this are left out; a word this long or longer is
#: searched by its stem as a prefix (``insulted`` → ``insult*``).
MIN_WORD = 3
PREFIX_MIN = 5
#: A stem keeps at least this many letters.
STEM_MIN = 4
#: Inflection endings, longest first — one is taken off (German and English).
#: The German endings are data, matched against German transcripts.
_SUFFIXES = ("ing", "en", "er", "es", "em", "st", "et", "te", "ed", "e", "t", "n", "s")
MAX_MUST = 5
MAX_WORDS = 40

_WORD = re.compile(r"[^\W_]+")
_QUOTED = re.compile(r'"([^"]+)"|„([^“”"]+)[“”"]|“([^”]+)”|«([^»]+)»|»([^«]+)«')
_SENTENCE_END = re.compile(r"[.!?]")
#: Speaker labels the chunker puts into a chunk's text (``[Alice] …``)
_LABEL = re.compile(r"\[[^\]\n]{1,60}\]")


# ══════════════════════════════════════════════════════════════════════════
#  The query's words
# ══════════════════════════════════════════════════════════════════════════


def _stop() -> set[str]:
    from whisper_tm.config import config
    from whisper_tm.core.stopwords import stopwords

    return stopwords(config.get("search", "stopword_languages", default=["en"]))


def _dedupe(items) -> list:
    return list(dict.fromkeys(x for x in items if x))


def prefix_form(word: str) -> str:
    """How a content word is searched: a word of :data:`PREFIX_MIN` letters
    or more by its stem as a prefix (one inflection ending off, at least
    :data:`STEM_MIN` letters kept), a shorter one or a number as it is."""
    w = word.lower()
    if len(w) < PREFIX_MIN or any(ch.isdigit() for ch in w):
        return w
    for end in _SUFFIXES:
        if w.endswith(end) and len(w) - len(end) >= STEM_MIN:
            return w[: -len(end)] + "*"
    return w + "*"


def query_terms(q: str, *, stop: set[str] | None = None, known_names=None) -> dict:
    """The search terms of a query.

    ``must``: what a passage has to contain — words or phrases given in
    quotes, names of people in the voice library (``known_names``), and
    capitalised words that don't start a sentence. ``words``: the other
    content words (stopwords out — core/stopwords.py), long ones as a prefix
    of their stem. Also ``quoted``, ``names`` and ``capitalised`` — the
    deep search takes must terms only from the first two.
    """
    stop = _stop() if stop is None else stop
    names_known = {n.lower() for n in (known_names or ())}
    quoted: list[str] = []

    def take(m):
        text = next(g for g in m.groups() if g is not None)
        toks = [t.lower() for t in _WORD.findall(text)]
        if toks:
            quoted.append(" ".join(toks))
        return " "

    rest = _QUOTED.sub(take, q or "")
    names, capitalised, words = [], [], []
    prev_end = 0
    for i, m in enumerate(_WORD.finditer(rest)):
        tok = m.group(0)
        starts = i == 0 or bool(_SENTENCE_END.search(rest, prev_end, m.start()))
        prev_end = m.end()
        lw = tok.lower()
        if len(lw) < MIN_WORD or lw in stop:
            continue
        if lw in names_known:
            names.append(lw)
        elif tok[:1].isupper() and not starts:
            capitalised.append(lw)
        else:
            words.append(prefix_form(lw))
    must = _dedupe(quoted + names + capitalised)[:MAX_MUST]
    return {
        "must": must,
        "words": [w for w in _dedupe(words) if w not in must][:MAX_WORDS],
        "quoted": _dedupe(quoted),
        "names": _dedupe(names),
        "capitalised": _dedupe(capitalised),
    }


_names_lock = threading.Lock()
_names_cache: dict = {"at": -1e9, "names": frozenset()}
NAMES_TTL_SEC = 120.0


def known_names() -> frozenset:
    """The words of the names of people in the voice library (lowercase) — a
    query word that is one of them is a must term: *"alice gets a compliment"*
    means passages with Alice in them. Read at most every two minutes; empty
    when the library can't be read."""
    now = time.monotonic()
    with _names_lock:
        if now - _names_cache["at"] < NAMES_TTL_SEC:
            return _names_cache["names"]
    out: set[str] = set()
    try:
        from whisper_tm.database.speaker import speaker_db

        stop = _stop()
        for p in speaker_db().people():
            for tok in _WORD.findall(str(p.get("name") or "").lower()):
                if len(tok) >= MIN_WORD and not tok.isdigit() and tok not in stop:
                    out.add(tok)
    except Exception as e:  # noqa: BLE001 — a search without names still works
        logger.debug("Situation search: names of the voice library not readable: %s", e)
    names = frozenset(out)
    with _names_lock:
        _names_cache.update(at=now, names=names)
    return names


# ── FTS5 expressions — every token quoted ──────────────────────────────────


def fts_term(term: str) -> str | None:
    """One term as FTS5 syntax: its tokens quoted (a phrase when several), a
    trailing ``*`` kept as a prefix. ``None`` when nothing searchable is left."""
    term = (term or "").strip()
    prefix = term.endswith("*")
    toks = _WORD.findall((term[:-1] if prefix else term).lower())
    if not toks:
        return None
    return '"' + " ".join(toks) + '"' + ("*" if prefix else "")


def _join(terms, op: str) -> str:
    parts = _dedupe(fts_term(t) for t in terms)
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else "(" + f" {op} ".join(parts) + ")"


def fts_or(terms) -> str:
    """Any of the terms."""
    return _join(terms, "OR")


def fts_and(terms) -> str:
    """All of the terms."""
    return _join(terms, "AND")


def fts_must_any(must, words) -> str:
    """All must terms and at least one of the other words."""
    m, w = fts_and(must), fts_or(words)
    if m and w:
        return f"{m} AND {w}"
    return m or w


# ══════════════════════════════════════════════════════════════════════════
#  Candidates
# ══════════════════════════════════════════════════════════════════════════


def fuse(lists, rrf_k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion of ranked lists ``[(name, rows), …]``.

    ``fused`` = Σ 1/(rrf_k + rank) over the lists a row is in. A row keeps
    its rank and value in the dense list (``vrank``/``vscore``; a ``voice…``
    list gives its cosine too), its best full-text rank and bm25 value
    (``frank``/``fscore``), the lists it came from, and ``match``: vector,
    fts or hybrid. Best first; ties by id.
    """
    together: dict[int, dict] = {}
    for name, rows in lists:
        for rank, r in enumerate(rows, start=1):
            d = together.get(r["id"])
            if d is None:
                d = {k: v for k, v in r.items() if k not in ("score", "match")}
                d.update(fused=0.0, lists=[], vrank=None, vscore=None, frank=None, fscore=None)
                together[r["id"]] = d
            d["fused"] += 1.0 / (rrf_k + rank)
            d["lists"].append(name)
            if name == "dense":
                d["vrank"], d["vscore"] = rank, r.get("score")
            elif name.startswith("voice"):
                # A person's passages by meaning: the same cosine as dense
                if d["vscore"] is None:
                    d["vscore"] = r.get("score")
            else:
                if d["frank"] is None or rank < d["frank"]:
                    d["frank"] = rank
                sc = r.get("score")
                if sc is not None and (d["fscore"] is None or sc > d["fscore"]):
                    d["fscore"] = sc
    for d in together.values():
        dense = d["vrank"] is not None or any(n.startswith("voice") for n in d["lists"])
        text = d["frank"] is not None
        d["match"] = "hybrid" if dense and text else ("vector" if dense else "fts")
    return sorted(together.values(), key=lambda d: (-d["fused"], d["id"]))


#: A word found in more than this share of all excerpts says nothing about a
#: situation ("conversation", "sentence" — words a plan may offer): it only
#: fills the full-text lists with everyday talk.
MAX_TERM_SHARE = 0.05


def common_terms_out(db, words, *, max_share: float = MAX_TERM_SHARE):
    """``(kept, dropped)``: the words without those in more than ``max_share``
    of the excerpts. Counted with a LIMIT, so a common word costs no more than
    a rare one; when counting fails, every word is kept."""
    try:
        with db.connection() as c:
            total = c.execute(f"SELECT count(*) FROM {db.TABLE}").fetchone()[0]
            limit = int(total * max_share) + 1
            if total < 200:  # a small index: no statistics to speak of
                return list(words), []
            kept, dropped = [], []
            for w in words:
                expr = fts_term(w)
                if not expr:
                    continue
                n = c.execute(
                    "SELECT count(*) FROM (SELECT rowid FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? LIMIT ?)",
                    (expr, limit),
                ).fetchone()[0]
                (dropped if n >= limit else kept).append(w)
            return kept, dropped
    except Exception as e:  # noqa: BLE001 — a filter must never cost the search
        logger.debug("Situation search: word statistics not available: %s", e)
        return list(words), []


def voice_ids(db, people) -> dict[str, list[int]]:
    """``{person: [chunk ids]}`` — the passages each person speaks in: one
    of their labels (rag/speakers.labels_of) stands in the passage's text, or
    is its main speaker (an index built without labels in the text).
    Well under a second even for a speaker in thousands of recordings."""
    from whisper_tm.rag import speakers

    out: dict[str, list[int]] = {}
    with db.connection() as c:
        for person in people:
            where = speakers.labels_of(person)
            ids: list[int] = []
            keys = list(where)
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                for cid, key, content, main in c.execute(
                    f"SELECT id, transcript_key, content, speaker FROM {db.TABLE} "
                    f"WHERE transcript_key IN ({','.join('?' * len(part))})",
                    part,
                ):
                    labels = where.get(key, ())
                    if main in labels or any(f"[{label}]" in (content or "") for label in labels):
                        ids.append(cid)
            out[person] = ids
    return out


def candidates(
    query: str,
    *,
    k_dense: int,
    k_fts: int,
    extra_terms=None,
    must=None,
    k_must: int = 0,
    or_list: bool = True,
    terms=None,
    db=None,
    embed_fn=None,
    qvec=None,
    use_dense: bool = True,
    k_voice: int = 0,
    rrf_k: int = RRF_K,
) -> tuple[list[dict], dict]:
    """Fused candidates for ``query``: ``(rows, info)``.

    Lists: the dense top ``k_dense`` (the question embedded with its
    instruction prefix); full text "must AND any word" (``k_fts``); full text
    "any term" (``k_fts``, with ``or_list``); full text "must" alone
    (``k_must``, the deep search); for each person of the voice library the
    query names, the ``k_voice`` passages they speak in that are closest to
    the question (``voice:<name>``). ``must`` replaces the query's own must
    terms, ``extra_terms`` add to its words. A failing embedding leaves the
    dense and voice lists out (``info["dense_error"]``) — the full text
    still answers.
    """
    if db is None:
        from whisper_tm.rag.service import make_rag_db

        db = make_rag_db()
    terms = terms or query_terms(query, known_names=known_names())
    must_terms = _dedupe(terms["must"] if must is None else must)
    # A must term of the query's own that ``must`` replaced is still a word
    words = [
        w
        for w in _dedupe(list(terms["words"]) + list(terms["must"]) + list(extra_terms or []))
        if w not in must_terms
    ]
    dropped: list[str] = []
    if extra_terms:
        words, dropped = common_terms_out(db, words)
    lists: list[tuple[str, list[dict]]] = []
    info: dict = {"dense_error": None, "must": must_terms, "words": words, "too_common": dropped}
    want_voice = k_voice > 0 and bool(terms.get("names"))
    if use_dense and (k_dense > 0 or want_voice):
        try:
            if qvec is None:
                if embed_fn is None:
                    from whisper_tm.rag.service import make_query_embed_fn

                    embed_fn = make_query_embed_fn()
                got = embed_fn([query])
                if not got or not got[0]:
                    raise RuntimeError("the embedding returned no vector")
                qvec = got[0]
            if k_dense > 0:
                lists.append(("dense", db.search_vector(qvec, k=k_dense)))
        except Exception as e:  # noqa: BLE001 — full text alone still finds
            info["dense_error"] = str(e)[:200]
            logger.warning("Situation search: meaning list left out for %r: %s", query[:80], e)
            qvec = None
    elif not use_dense:
        qvec = None
    if want_voice and qvec is not None:
        try:
            from whisper_tm.rag import speakers

            people = speakers.people_named(terms["names"])
            voices: dict[str, int] = {}
            for person, ids in voice_ids(db, people).items():
                voices[person] = len(ids)
                if not ids or len(ids) > MAX_VOICE_PASSAGES:
                    continue
                lists.append((f"voice:{person}", db.search_vector(qvec, k=k_voice, among=ids)))
            info["voices"] = voices
        except Exception as e:  # noqa: BLE001 — the other lists still answer
            logger.warning("Situation search: speaker lists left out for %r: %s", query[:80], e)
    if k_fts > 0 and must_terms and words:
        lists.append(("must_any", db.search_match(fts_must_any(must_terms, words), k=k_fts)))
    if k_fts > 0 and or_list and (must_terms or words):
        lists.append(("any", db.search_match(fts_or(must_terms + words), k=k_fts)))
    if k_must > 0 and must_terms:
        lists.append(("must", db.search_match(fts_and(must_terms), k=k_must)))
    info["lists"] = {name: len(rows) for name, rows in lists}
    return fuse(lists, rrf_k), info


# ══════════════════════════════════════════════════════════════════════════
#  Copies
# ══════════════════════════════════════════════════════════════════════════


def text_key(content: str) -> str:
    """A passage's text for comparing copies: speaker labels out, lower case,
    everything but letters and digits out; its head and its tail."""
    t = re.sub(r"\W+", "", _LABEL.sub(" ", content or "").lower())
    if len(t) <= COPY_TEXT_HEAD + COPY_TEXT_TAIL:
        return t
    return t[:COPY_TEXT_HEAD] + "|" + t[-COPY_TEXT_TAIL:]


#: Two passages of same-named files count as copies when their time spans
#: overlap by this share of the shorter one … (separate transcriptions of one
#: video cut their ~60 s passages at other places: the same scene overlapped
#: by 45 % in the measurement; one transcription's neighbours share a few s)
COPY_TIME_OVERLAP = 0.35
#: … and their vectors reach this (or, without vectors, their words overlap by
#: this share). Measured on one video as .mp4/.mkv/.webm: passages of
#: the same moment 0.78–0.95, shifted neighbours 0.40–0.70.
COPY_STEM_COSINE = 0.7
COPY_STEM_WORDS = 0.35


def _time_overlap(a: dict, b: dict) -> float:
    """Shared time of two passages as a share of the shorter one (0 without times)."""
    try:
        s1, e1 = float(a.get("start_sec")), float(a.get("end_sec"))
        s2, e2 = float(b.get("start_sec")), float(b.get("end_sec"))
    except (TypeError, ValueError):
        return 0.0
    shorter = min(e1 - s1, e2 - s2)
    if shorter <= 0:
        return 0.0
    return max(0.0, min(e1, e2) - max(s1, s2)) / shorter


def _stem(path: str) -> str:
    """A file's name without folder and extension, lower case."""
    name = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name.strip().lower()


def _similar(a: dict, b: dict, V, ia: int, ib: int) -> bool:
    """Clearly the same talk: vectors at COPY_STEM_COSINE, or — without usable
    vectors — word sets overlapping by COPY_STEM_WORDS (Jaccard)."""
    if V is not None and np.isfinite(V[ia]).all() and np.isfinite(V[ib]).all():
        return float(V[ia] @ V[ib]) >= COPY_STEM_COSINE
    wa = set(_WORD.findall(_LABEL.sub(" ", a.get("content") or "").lower()))
    wb = set(_WORD.findall(_LABEL.sub(" ", b.get("content") or "").lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= COPY_STEM_WORDS


def recording_lengths(keys) -> dict | None:
    """``{abs_path: seconds}`` of these recordings from the library — None
    without it (the copies are then told by their text and names only)."""
    try:
        from whisper_tm.core import copies
        from whisper_tm.extensions import db

        if db is None:
            return None
        return {k: f[1] for k, f in copies.facts(db, keys).items() if f[1]}
    except Exception as e:  # noqa: BLE001 — grouped without the lengths then
        logger.debug("Situation search: lengths of the recordings not read: %s", e)
        return None


def copy_of(r: dict) -> dict:
    """What a group keeps of a copy."""
    return {
        "id": r.get("id"),
        "transcript_key": r.get("transcript_key"),
        "chunk_index": r.get("chunk_index"),
        "start_sec": r.get("start_sec"),
        "end_sec": r.get("end_sec"),
        "media_path": r.get("media_path"),
    }


def group_copies(
    rows: list[dict], vectors=None, *, threshold: float = COPY_COSINE, lengths: dict | None = None
) -> list[dict]:
    """Collapse copies of one passage — the same recording in another folder.

    ``rows`` in rank order; ``vectors`` (optional) their stored unit vectors,
    one row each (NaN rows are left out of the comparison). Two rows are
    copies when their :func:`text_key` agrees or their vectors reach
    ``threshold``; copies of copies are one group. ``lengths`` (optional,
    ``{transcript_key: seconds}``): recordings of the same length
    (core/copies) are copies too when their passages cover the same stretch
    of time and say clearly the same — whatever the files are called. Each
    group is its best-ranked row with ``copies`` (the others,
    :func:`copy_of`, in rank order); the groups keep the rank order.
    """
    n = len(rows)
    parent = list(range(n))
    # The files in each group: a group never takes two passages of ONE file.
    # Copies chain (a .mkv passage like a .webm one like the next .mkv
    # passage) — without this, neighbouring passages of one recording ended
    # up as "copies" of each other and one of them vanished from the list.
    files = [
        {(r.get("transcript_key") or r.get("media_path") or f"#{i}")} for i, r in enumerate(rows)
    ]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb and not (files[ra] & files[rb]):
            # The better rank stays the root: the representative
            keep, drop = min(ra, rb), max(ra, rb)
            parent[drop] = keep
            files[keep] |= files[drop]

    first: dict[str, int] = {}
    for i, r in enumerate(rows):
        key = text_key(r.get("content") or "")
        if not key:
            continue
        if key in first:
            union(first[key], i)
        else:
            first[key] = i
    V = None
    if vectors is not None and n > 1:
        V = np.asarray(vectors, dtype=np.float32)
        idx = np.flatnonzero(np.isfinite(V).all(axis=1))
        if len(idx) > 1:
            W = V[idx]
            for s in range(0, len(idx), 1024):
                S = W[s : s + 1024] @ W.T
                a, b = np.nonzero(S >= threshold)
                for x, y in zip(a + s, b):
                    if y > x:
                        union(int(idx[x]), int(idx[y]))
    # The same video in another container (.mp4/.mkv/.webm), transcribed on
    # its own (e.g. one passage three times in a top ten): the words
    # differ a little, the chunk borders too — so neither the text nor the
    # 0.97 vector rule catches it. Same file name, another file, mostly the
    # same stretch of time and clearly similar content is the same passage.
    by_stem: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        stem = _stem(r.get("transcript_key") or r.get("media_path") or "")
        if stem:
            by_stem.setdefault(stem, []).append(i)
    for members in by_stem.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = rows[members[x]], rows[members[y]]
                if (a.get("transcript_key") or "") == (b.get("transcript_key") or ""):
                    continue
                if _time_overlap(a, b) < COPY_TIME_OVERLAP:
                    continue
                if _similar(a, b, V, members[x], members[y]):
                    union(members[x], members[y])
    # The same video under another name (e.g. saved once under its video
    # ID and once under its title): the
    # rule above with the same length (core/copies — Find folds the same pairs)
    # in place of the same name.
    if lengths:
        from whisper_tm.core import copies

        known = sorted(
            (i for i, r in enumerate(rows) if lengths.get(r.get("transcript_key") or "")),
            key=lambda i: lengths[rows[i]["transcript_key"]],
        )
        for x, ix in enumerate(known):
            la = lengths[rows[ix]["transcript_key"]]
            for iy in known[x + 1 :]:
                lb = lengths[rows[iy]["transcript_key"]]
                if not copies.same_length(la, lb):
                    if lb - la > max(copies.LENGTH_SEC, copies.LENGTH_SHARE * lb):
                        break
                    continue
                a, b = rows[ix], rows[iy]
                if (
                    a.get("transcript_key") == b.get("transcript_key")
                    or _time_overlap(a, b) < COPY_TIME_OVERLAP
                ):
                    continue
                if _similar(a, b, V, ix, iy):
                    union(ix, iy)
    groups: dict[int, dict] = {}
    order: list[int] = []
    for i, r in enumerate(rows):
        root = find(i)
        if root not in groups:
            g = dict(r)
            g["copies"] = []
            groups[root] = g
            order.append(root)
        else:
            groups[root]["copies"].append(copy_of(r))
    return [groups[k] for k in order]


# ══════════════════════════════════════════════════════════════════════════
#  Places: from the index's key back to the file list
# ══════════════════════════════════════════════════════════════════════════


def display_paths(keys, db) -> dict:
    """abs_path keys → ``{"rel_path", "file_dir"}``, for jumping into the
    transcript (``svGoto(rel_path, start, root)``).

    The index knows a transcript by its ``abs_path`` (the identity,
    docs/PATHS.md); old rows by their ``rel_path`` — the second query finds
    those. A targeted query for the keys at hand, not
    ``get_rag_candidates()``: that loads the whole library. In
    blocks of 200, below SQLite's placeholder limit. A key no longer in the
    library is left out — its hit stays readable, just not jumpable.

    The root comes from the identity (database._root_of), not from the
    stored ``media_dir``: some transcripts carry their
    file's own folder there while ``rel_path`` is relative to a root above
    it — the hit's context said "Transcript not found", and playing and
    opening it pointed at a path that doesn't exist.
    """
    from whisper_tm.config import config
    from whisper_tm.database import _root_of

    keys = [k for k in dict.fromkeys(keys) if k]
    if not keys or not db:
        return {}
    hits: dict[str, dict] = {}
    try:
        with db.connection() as conn:
            for column_name in ("abs_path", "rel_path"):
                missing = [k for k in keys if k not in hits]
                if not missing:
                    break
                for i in range(0, len(missing), 200):
                    part = missing[i : i + 200]
                    sql = (
                        f"SELECT {column_name} AS k, rel_path, abs_path, media_dir "
                        f"FROM transcripts WHERE {column_name} IN "
                        f"({','.join('?' * len(part))})"
                    )
                    for r in conn.execute(sql, part).fetchall():
                        hits.setdefault(
                            r["k"],
                            {
                                "rel_path": r["rel_path"],
                                "file_dir": (
                                    _root_of(r["rel_path"], r["abs_path"], r["media_dir"])
                                    or str(config.media_dir)
                                ),
                            },
                        )
    except Exception as e:  # noqa: BLE001 — a missing jump is not a search error
        logger.debug("Display paths not resolvable: %s", e)
    return hits


def place_copies(items: list[dict], db) -> None:
    """Give every item and each of its copies ``rel_path`` and ``file_dir``."""
    keys = []
    for it in items:
        keys.append(it.get("transcript_key"))
        keys.extend(c.get("transcript_key") for c in it.get("copies") or [])
    places = display_paths(keys, db)
    for it in items:
        for x in [it, *(it.get("copies") or [])]:
            p = places.get(x.get("transcript_key") or "", {})
            x["rel_path"] = p.get("rel_path")
            x["file_dir"] = p.get("file_dir")


# ══════════════════════════════════════════════════════════════════════════
#  The quick search
# ══════════════════════════════════════════════════════════════════════════


def quick_search(
    query: str,
    *,
    k: int = 50,
    mode: str = "hybrid",
    rerank: bool = True,
    group: bool = True,
    db=None,
    embed_fn=None,
    rerank_fn=None,
    wait_sec: float | None = None,
    rrf_k: int = RRF_K,
) -> dict:
    """The situation search behind the meaning mode and the chat's
    ``semantic_search``.

    Candidates (dense :data:`QUICK_DENSE` + full text :data:`QUICK_FTS` +
    :data:`QUICK_VOICE` per named person; ``mode="fts"`` full text only) →
    copies collapsed (``group``) → speakers named → the best
    ``rerank.quick_candidates`` groups reranked with ``rerank.instruction``
    (``rerank``). Returns ``results`` (at most ``k``; each with
    ``relevance`` — ``None`` where not reranked — ``fused``, ``match``,
    ``copies``, ``speaker_name`` where known), ``reranked``, ``note`` (why
    not, when asked), ``min_score``, ``candidates`` (groups considered) and
    ``voices`` (passages per named person). Without the reranker the fused
    order stands.
    """
    from whisper_tm.core import rerank_server

    t0 = time.monotonic()
    if db is None:
        from whisper_tm.rag.service import make_rag_db

        db = make_rag_db()
    rows, info = candidates(
        query,
        k_dense=QUICK_DENSE,
        k_fts=QUICK_FTS,
        db=db,
        embed_fn=embed_fn,
        use_dense=(mode != "fts"),
        k_voice=QUICK_VOICE,
        rrf_k=rrf_k,
    )
    notes = []
    if info.get("dense_error"):
        notes.append(
            f"Meaning search failed ({info['dense_error'][:120]}) — full-text candidates only"
        )
    if group and len(rows) > 1:
        try:
            vectors = db.vectors_for([r["id"] for r in rows])
        except Exception as e:  # noqa: BLE001 — same text still collapses
            logger.warning("Situation search: stored vectors not readable: %s", e)
            vectors = None
        groups = group_copies(
            rows, vectors, lengths=recording_lengths(r.get("transcript_key") for r in rows)
        )
    else:
        groups = [dict(r, copies=[]) for r in rows]
    # Who speaks — for the reranker and on screen (rag/speakers.py)
    from whisper_tm.rag import speakers

    speakers.name_rows(groups)
    for g in groups:
        g["relevance"] = None
    reranked = False
    n_rerank = max(0, rerank_server.setting_int("quick_candidates"))
    if rerank and groups and n_rerank:
        top = groups[:n_rerank]
        try:
            scores = (rerank_fn or rerank_server.rerank)(
                query,
                [g.get("content") or "" for g in top],
                wait_sec=rerank_server.QUICK_WAIT_SEC if wait_sec is None else wait_sec,
            )
            for g, sc in zip(top, scores):
                g["relevance"] = float(sc)
            top.sort(key=lambda g: -g["relevance"])  # stable: ties keep fused order
            groups = top + groups[n_rerank:]
            reranked = True
        except rerank_server.RerankUnavailable as e:
            notes.append(e.note())
            logger.info("Situation search without the reranker (%s): %s", e.reason, e)
        except Exception as e:  # noqa: BLE001 — a broken reranker must not take the search
            logger.exception("Situation search %r: the reranker failed", query[:80])
            notes.append(rerank_server.RerankUnavailable("failed", str(e)).note())
    for g in groups:
        g["score"] = g["relevance"] if g["relevance"] is not None else g["fused"]
    took = time.monotonic() - t0
    logger.info(
        "Situation search %r: %d candidates, %d groups, %s in %.1f s",
        query[:80],
        len(rows),
        len(groups),
        "reranked" if reranked else "fused order",
        took,
    )
    return {
        "results": groups[: max(1, int(k))],
        "reranked": reranked,
        "grouped": bool(group),
        "note": " · ".join(notes) or None,
        "min_score": rerank_server.setting_float("quick_min_score"),
        "candidates": len(groups),
        "lists": info.get("lists") or {},
        "voices": info.get("voices") or {},
        "elapsed_ms": round(took * 1000),
    }

"""Find anything — one description, every search at once, ranked together.

The user describes a moment in their own words: what is said, seen or
heard, who is in it, what the file was called. A run:

1. The description itself is searched by meaning at once — the first hits
   come while the language model still plans (otherwise ~10 s of
   nothing before).
2. The plan (find/plan.py) splits the description, checked against the
   library: the voice library's names, no speech as a "sound", the kind of
   recording as a filter only when the description says it. A description
   that asks for no moment — recordings that exist twice, whole
   conversations with given people — ends here: the page shows them where
   they live (Same audio, Conversations).
3. Every search runs at once (find/engines.py): words, meaning, seen,
   heard — and the same INSIDE the files whose name has a remembered piece
   and in what the named people say. Each fills its tile as it finishes,
   its suspected duplicates folded.
4. The fusion (find/fusion.py) makes moments of them; copies of a
   recording are one recording.
5. The language model judges the best (find/judge.py) — one per recording
   first, with the facts the program knows and a still — and the list is
   ordered: may be it, not judged, rejects. Meanwhile the best ones' clips
   are cut, so the preview plays at once.

One run at a time: a new one stops the one before — the language model is
shared with the chat, the queue and the transcription. A run is a thread;
the page polls ``state`` (``rev`` grows with every change, ``erev`` with
every change of the searches' lists — unchanged lists aren't sent again).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from whisper_tm.find import engines, fusion, judge
from whisper_tm.find import plan as planning

logger = logging.getLogger(__name__)

ENGINES = ("words", "meaning", "seen", "heard", "files", "people")
#: The best fused moments shown
BEST_SHOWN = 40
KEEP_RUNS = 20
#: The best this many get their preview clip cut while the judge works
PRECUT = 5


# ══════════════════════════════════════════════════════════════════════════
#  A frame and a clip of a recording — the stills of the list, the preview
# ══════════════════════════════════════════════════════════════════════════

FRAMES_MAX_MB = 300
CLIPS_MAX_MB = 1000
#: Frames and clips cut at once
_frame_gate = threading.BoundedSemaphore(4)
_clip_gate = threading.BoundedSemaphore(2)
_cut_count = [0]

#: The preview plays the moment from a little before it, long enough to see
#: what happens — a passage of the meaning search all of it, up to the most
CLIP_BEFORE_SEC = 4.0
CLIP_SEC = 30.0
CLIP_MAX_SEC = 90.0


def frames_dir():
    from pathlib import Path

    from whisper_tm.config import config

    return Path(config.cache_dir) / "find" / "frames"


def clips_dir():
    return frames_dir().parent / "clips"


def _key_hash(key: str) -> str:
    import hashlib

    return hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def frame_url(key: str, t: float) -> str:
    from urllib.parse import quote

    return f"/api/find/frame?t={int(max(0, t))}&key={quote(key, safe='')}"


def frame_path(key: str, t: float):
    """Where the frame of ``key`` at ``t`` is kept (cut or not)."""
    return frames_dir() / _key_hash(key) / f"{int(max(0, t))}.jpg"


def recording(db, key: str) -> dict | None:
    """``{"video", "duration"}`` of a file of the library, None for any other path."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT is_video, has_video, duration_sec FROM media_files WHERE abs_path=? LIMIT 1",
            (key,),
        ).fetchone()
    if row is None:
        return None
    video = bool(row[1]) if row[1] is not None else bool(row[0])
    return {"video": video, "duration": float(row[2] or 0.0)}


def cut_frame(key: str, t: float, width: int = 480):
    """A JPEG of the recording at ``t`` (kept in the cache), or None."""
    from whisper_tm.media import cuts

    def make(out):
        with _frame_gate:
            return cuts.cut_still(key, int(max(0, t)), out, width=width)

    def after_cut():
        # Counted and trimmed after real cuts only, not on cache hits
        _cut_count[0] += 1
        if _cut_count[0] % 100 == 0:
            cuts.trim(frames_dir(), FRAMES_MAX_MB)

    return cuts.cached(frame_path(key, t), make, after_cut=after_cut)


def clip_window(t: float, end: float | None = None) -> tuple[float, float]:
    """``(start, length)`` of the preview of a moment at ``t`` (to ``end``):
    from ``CLIP_BEFORE_SEC`` before it — the page counts the recording's
    time from there — at least ``CLIP_SEC``, a passage to its end, at most
    ``CLIP_MAX_SEC``. At a recording's end the clip is just shorter."""
    t = max(0.0, float(t or 0.0))
    start = max(0.0, t - CLIP_BEFORE_SEC)
    want = CLIP_SEC
    if end is not None and float(end) > t:
        want = max(CLIP_SEC, float(end) - start + 2.0)
    return round(start, 1), round(min(CLIP_MAX_SEC, want), 1)


def cut_clip(key: str, start: float, length: float, video: bool):
    """The stretch ``start`` … ``start + length`` as a file the browser plays
    (media/cuts), cut once and kept (``CLIPS_MAX_MB``, the oldest go
    first); None when the recording can't be read."""
    from whisper_tm.media import cuts

    out = clips_dir() / f"{_key_hash(key)}-{start:.1f}-{length:.1f}.{'mp4' if video else 'm4a'}"

    def make(o):
        with _clip_gate:
            return cuts.cut_clip(key, start, length, video, o)

    return cuts.cached(
        out, make, after_cut=lambda: cuts.trim(clips_dir(), CLIPS_MAX_MB, suffixes=(".mp4", ".m4a"))
    )


#: No recording is longer (115 days): t=1e308 made a file name too long
SECONDS_MAX = 1e7


def finite_seconds(v) -> float | None:
    """A time from a page: a finite number 0 … :data:`SECONDS_MAX`, else None."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, x) if math.isfinite(x) and x <= SECONDS_MAX else None


# ══════════════════════════════════════════════════════════════════════════
#  A run
# ══════════════════════════════════════════════════════════════════════════

#: Who started a run: the Find tab, or Ask / an MCP client (find_moments)
OWNERS = ("tab", "chat")


class Run:
    def __init__(self, query: str, owner: str = "tab"):
        self.id = uuid.uuid4().hex[:12]
        self.query = query
        self.owner = owner if owner in OWNERS else "tab"
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.started = time.time()
        self.state = {
            "id": self.id,
            "query": query,
            "status": "planning",
            "plan": None,
            "plan_source": None,
            "engines": {n: {"status": "waiting"} for n in ENGINES},
            "best": [],
            "judged": 0,
            "judge_total": 0,
            "log": [],
            "rev": 0,
            "erev": 0,
            "started": self.started,
            "ended": None,
            "error": None,
            "owner": self.owner,
            "stopping": False,
            # A run that asks for conversations: the Conversations tab's
            # filters (find/plan.conversation_filters) — find.js opens it
            "conversation_filters": None,
        }
        self.people_paths: dict[str, set] = {}

    def set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)
            self.state["rev"] += 1

    def log(self, text: str) -> None:
        with self.lock:
            self.state["log"].append({"t": round(time.time() - self.started, 1), "text": text})
            self.state["rev"] += 1

    def engine(self, name: str, **kw) -> None:
        with self.lock:
            self.state["engines"][name].update(kw)
            self.state["rev"] += 1
            self.state["erev"] += 1

    def snapshot(self, erev: int | None = None) -> dict:
        """The state as the page gets it — the searches' lists left out when
        the page already has them (``erev`` unchanged)."""
        with self.lock:
            state = json.loads(json.dumps(self.state, default=list))
        if erev is not None and erev == state["erev"]:
            for e in state["engines"].values():
                e.pop("items", None)
            state["engines_same"] = True
        return state


_runs: dict[str, Run] = {}
_runs_lock = threading.Lock()


def start(query: str, owner: str = "tab") -> Run:
    """A run for ``query`` — the one of the same ``owner`` still going is
    stopped (one at a time each), so Ask's find_moments and the Find tab
    don't stop each other."""
    run = Run(" ".join((query or "").split())[:1000], owner)
    with _runs_lock:
        for old in _runs.values():
            if old.owner == run.owner and not old.state["ended"] and not old.stop_flag.is_set():
                old.stop_flag.set()
                old.set(stopping=True)
                old.log("Stopped — a new search started")
        _runs[run.id] = run
        for old in sorted(_runs.values(), key=lambda r: r.started)[:-KEEP_RUNS]:
            _runs.pop(old.id, None)
    threading.Thread(target=_go, args=(run,), name=f"find-{run.id}", daemon=True).start()
    return run


def get(run_id: str) -> Run | None:
    with _runs_lock:
        return _runs.get(run_id)


def stop(run_id: str) -> bool:
    run = get(run_id)
    if not run or run.state["ended"] or run.stop_flag.is_set():
        return False
    run.stop_flag.set()
    run.set(stopping=True)
    run.log("Stopping — what was found so far stays")
    return True


def _timed(fn):
    t0 = time.monotonic()
    out = fn()
    return out, round((time.monotonic() - t0) * 1000)


def _merge_meaning(raw: dict, planned: dict) -> dict:
    """The description's and the plan's meaning search as one list."""
    usable = [x for x in (raw, planned) if x and not x.get("skip")]
    if len(usable) < 2:
        return usable[0] if usable else planned
    lists = [x.get("items") or [] for x in usable]
    items = engines.rrf(lists, "meaning")
    # the duplicates the semantic search folded stay with their hit
    dups: dict = {}
    for lst in lists:
        for it in lst:
            dups.setdefault((it["key"], round(float(it["t"] or 0) / 5)), it.get("dups") or [])
    for it in items:
        it["dups"] = dups.get((it["key"], round(float(it["t"] or 0) / 5)), [])
    notes = [x.get("note") for x in usable if x.get("note")]
    return {
        "items": items,
        "count": max(x.get("count") or 0 for x in usable),
        "query": f"{planned.get('query')} · and the description itself",
        "note": " · ".join(dict.fromkeys(notes)) or None,
    }


def _go(run: Run) -> None:
    from whisper_tm.extensions import db

    ctx = engines.Ctx(db)
    pool = ThreadPoolExecutor(8, thread_name_prefix=f"find-{run.id}")
    results: dict[str, dict] = {}
    pending: dict = {}
    try:
        # the description itself, by meaning — its hits come while the
        # language model still plans
        run.engine("meaning", status="running")
        raw = pool.submit(_timed, lambda: engines.search_meaning(run.query, ctx, rerank=False))
        t0 = time.monotonic()
        planned = pool.submit(planning.make_plan, run.query)
        raw_shown = False
        while not planned.done():
            if run.stop_flag.is_set():
                _stopped(run, {"meaning": None})
                return
            if not raw_shown and raw.done():
                raw_shown = True
                _show_raw(run, raw, results, {"media": "any"}, db)
            time.sleep(0.1)
        plan, source = planned.result()
        if plan.get("asks_for") in ("duplicates", "contained"):
            # Not a moment: the recordings that exist twice or contain one
            # another are the audio analysis's (the page shows them there)
            for name in ENGINES:
                run.engine(name, status="skipped", note="asks for duplicates, not a moment")
            run.set(plan=plan, plan_source=source, status="duplicates", best=[], ended=time.time())
            run.log(
                "Asks for recordings that exist more than once, not for a moment — "
                "Same audio shows them"
            )
            return
        if plan.get("asks_for") == "conversations":
            # Not a moment either: whole conversations with given people, of
            # a length or a kind are counted, not searched (conversations/)
            # — the page opens the Conversations tab with these filters
            for name in ENGINES:
                run.engine(name, status="skipped", note="asks for conversations, not a moment")
            # The log first: the page stops asking once the run has ended,
            # and a note written after that (a name nobody has) would never
            # show
            for note in plan.get("notes") or []:
                run.log(note)
            run.log(
                "Asks for whole conversations, not for a moment — the Conversations tab shows them"
            )
            run.set(
                plan=plan,
                plan_source=source,
                status="conversations",
                best=[],
                conversation_filters=planning.conversation_filters(plan),
                ended=time.time(),
            )
            return
        run.set(plan=plan, plan_source=source, status="searching")
        run.log(
            f"Planned in {time.monotonic() - t0:.1f} s"
            + ("" if source == "llm" else f" ({source})")
        )
        for note in plan.get("notes") or []:
            run.log(note)

        def meaning():
            try:
                first, _ms = raw.result()
            except Exception as e:  # noqa: BLE001 — the plan's sentence alone then
                logger.info("Find %s: the description's meaning search failed: %s", run.id, e)
                first = {"skip": "the description's search failed"}
            # the plan's sentence — or, without one, the description again — reranked
            said = plan.get("said_meaning") or run.query
            try:
                planned = engines.search_meaning(said, ctx)
            except Exception as e:  # noqa: BLE001 — the description's hits stand
                if first.get("skip"):
                    raise
                logger.info("Find %s: the plan's meaning search failed: %s", run.id, e)
                return first
            return _merge_meaning(first, planned)

        jobs = {
            "words": lambda: engines.search_words(plan, ctx),
            "meaning": meaning,
            "seen": lambda: engines.search_seen(plan, ctx),
            "heard": lambda: engines.search_heard(plan, ctx),
            "files": lambda: engines.search_files(plan, ctx),
            "people": lambda: engines.search_people(plan, ctx),
        }
        for name, fn in jobs.items():
            run.engine(name, status="running")
            pending[name] = pool.submit(_timed, fn)
        while pending and not run.stop_flag.is_set():
            if not raw_shown and "meaning" in pending and raw.done():
                # the description's own hits, while the plan's search runs
                raw_shown = True
                _show_raw(run, raw, results, plan, db)
            for name, fut in list(pending.items()):
                if not fut.done():
                    continue
                del pending[name]
                try:
                    out, ms = fut.result()
                except Exception as e:  # noqa: BLE001 — one search failing leaves the others
                    logger.warning("Find %s: %s failed: %s", run.id, name, e)
                    run.engine(name, status="failed", error=str(e)[:300])
                    run.log(f"{name}: failed — {str(e)[:120]}")
                    continue
                if out.get("skip"):
                    results.pop(name, None)
                    run.engine(name, status="skipped", note=out["skip"], items=[])
                    continue
                results[name] = out
                if name == "people":
                    run.people_paths = out.get("per_person") or {}
                folded = _publish(run, name, out, db, status="done", ms=ms)
                run.log(
                    f"{name}: {out.get('count', 0):,} in {ms / 1000:.1f} s"
                    + (f" — {folded} suspected duplicates folded" if folded else "")
                )
                run.set(best=fusion.fuse(results, plan, db)[:BEST_SHOWN])
            if pending:
                time.sleep(0.15)
        if run.stop_flag.is_set():
            _stopped(run, pending)
            return
        best = fusion.fuse(results, plan, db)[:BEST_SHOWN]
        run.set(best=best, status="judging" if best else "done")
        if best:
            _precut(best[:PRECUT], db, run.stop_flag)
            _judge_all(run, plan, best, db)
        if run.stop_flag.is_set():
            _stopped(run, {})
        else:
            run.set(status="done", ended=time.time())
            run.log(f"Done in {time.time() - run.started:.0f} s")
    except Exception as e:  # noqa: BLE001 — shown on the page
        logger.exception("Find %s failed", run.id)
        run.set(status="failed", error=str(e)[:300], ended=time.time())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _show_raw(run: Run, raw, results: dict, plan: dict, db) -> None:
    """The description's own meaning hits in their tile and in Best matches
    — before the plan's searches are back."""
    try:
        first, _ms = raw.result()
    except Exception as e:  # noqa: BLE001 — the planned search still comes
        logger.info("Find %s: the description's meaning search failed: %s", run.id, e)
        return
    if first.get("skip"):
        return
    results["meaning"] = first
    _publish(run, "meaning", first, db, status="running")
    run.set(best=fusion.fuse(results, plan, db)[:BEST_SHOWN])
    run.log(
        f"meaning: the description itself — {len(first.get('items') or [])} hits while planning"
        if run.state.get("plan") is None
        else f"meaning: the description itself — {len(first.get('items') or [])} hits"
    )


def _stopped(run: Run, pending: dict) -> None:
    for name in pending:
        run.engine(name, status="stopped", note="stopped")
    run.set(status="stopped", ended=time.time())
    run.log(f"Stopped after {time.time() - run.started:.0f} s — what was found stays")


def _publish(run: Run, name: str, out: dict, db, *, status: str, ms: int | None = None) -> int:
    """A search's hits into its tile, its suspected duplicates folded;
    returns how many were folded."""
    shown, folded = out.get("items") or [], 0
    try:
        shown, folded = fusion.fold_engine(shown, db)
    except Exception as e:  # noqa: BLE001 — shown unfolded then
        logger.warning("Find %s: folding %s failed: %s", run.id, name, e)
    kw = {
        "status": status,
        "count": out.get("count", 0),
        "query": out.get("query"),
        "note": out.get("note"),
        "items": shown,
        "folded": folded,
    }
    if ms is not None:
        kw["ms"] = ms
    run.engine(name, **kw)
    return folded


def _precut(best: list[dict], db, stop_flag: threading.Event | None = None) -> None:
    """Cut the best moments' preview clips in the background — the click on
    one plays at once instead of after a cut of up to 15 s. A stopped run's
    cuts end (otherwise they go on, taking the cut slots of the next run)."""

    def work():
        for m in best:
            if stop_flag is not None and stop_flag.is_set():
                return
            try:
                rec = recording(db, m["key"]) if db is not None else None
                if rec:
                    start, length = clip_window(m["t"], m.get("end"))
                    cut_clip(m["key"], start, length, rec["video"])
            except Exception as e:  # noqa: BLE001 — cut when clicked then
                logger.debug("Find: clip of %s not cut ahead: %s", m["key"], e)

    threading.Thread(target=work, name="find-precut", daemon=True).start()


def _judge_all(run: Run, plan: dict, best: list[dict], db) -> None:
    from whisper_tm.core import copies

    try:
        from whisper_tm.media import llm as mllm

        vision = mllm.vision_available() is not False
    except Exception:  # noqa: BLE001
        vision = False
    total = min(judge.JUDGE_TOP, len(best))
    lengths = (
        {k: f[1] for k, f in copies.facts(db, [m["key"] for m in best]).items()}
        if db is not None
        else {}
    )
    run.set(judge_total=total)
    run.log(
        f"The language model checks the best {total} — a recording's next moment when "
        "its first is rejected" + (", with their stills" if vision else "")
    )
    fails = 0
    done: set = set()
    for k in range(total):
        if run.stop_flag.is_set():
            return
        m = judge.next_to_judge(best, done)
        if m is None:
            break
        done.add(id(m))
        facts, failed = judge.facts_for(m, plan, people_paths=run.people_paths, lengths=lengths)
        try:
            said = judge.said_around(db, m["key"], m["t"]) if db is not None else ""
            verdict, reason = judge.judge(run.query, plan, m, said, facts, vision)
        except Exception as e:  # noqa: BLE001 — three in a row end the checking
            fails += 1
            logger.info("Find %s: judging %s failed: %s", run.id, m["name"], e)
            if fails >= 3:
                run.log(f"The language model doesn't answer — stopped checking ({str(e)[:100]})")
                return
            continue
        fails = 0
        m["verdict"], m["reason"] = judge.capped(verdict, failed), reason
        m["failed"] = failed
        if failed and verdict is not None and verdict != m["verdict"]:
            m["verdict_said"] = verdict  # what the language model said before the cap
        logger.info(
            "Find %s: %s@%s → %s%s",
            run.id,
            m["name"],
            judge.clock(m["t"]),
            verdict,
            f" (capped: {'; '.join(failed)})" if failed else "",
        )
        run.set(best=judge.ordered(best), judged=k + 1)


# ══════════════════════════════════════════════════════════════════════════
#  Ask about it — a conversation with the run attached (core/chat.py)
# ══════════════════════════════════════════════════════════════════════════


def ask(run_id: str) -> str:
    """A conversation in Ask about this run: its best matches go along with
    every question, each a reference (S1 …) that plays."""
    from whisper_tm.core import chat

    run = get(run_id)
    if run is None:
        raise LookupError("No such run (they are kept until the server restarts)")
    s = run.snapshot()
    plan = s.get("plan") or {}
    refs, lines = {}, []
    for n, m in enumerate((s.get("best") or [])[:20], 1):
        ref = f"S{n}"
        said = next((e.get("text") for e in m["engines"].values() if e.get("text")), "") or ""
        refs[ref] = {
            "file": m.get("rel") or m["name"],
            "start": m["t"],
            "end": m.get("end"),
            "text": (m.get("reason") or said)[:300],
            "speaker": "",
            "abs_path": m["key"],
            "file_dir": m.get("dir") or "",
            "filename": m.get("rel") or m["name"],
        }
        found = ", ".join(judge.FOUND_BY.get(e, e) for e in m["engines"])
        # One decimal: 6.55 and 6.65 would both read "7/9"
        verdict = (
            f"{m['verdict']:.1f}/9 — {m.get('reason') or ''}"
            if m.get("verdict") is not None
            else "not checked"
        )
        lines.append(
            f"{ref}: {chat.tool_file(m['key'], m.get('rel') or m['name'])} at "
            f"{judge.clock(m['t'])} — {verdict} — "
            f"found {found}" + (f" — {len(m['dups'])} copies" if m.get("dups") else "")
        )
    searched = "; ".join(
        f"{k}: {v}"
        for k, v in (
            ("said word for word", ", ".join(plan.get("said_words") or [])),
            ("said, by meaning", plan.get("said_meaning")),
            ("seen", plan.get("seen")),
            ("heard", plan.get("heard")),
            ("people", ", ".join(plan.get("people") or [])),
            ("file name", ", ".join(plan.get("file_hints") or [])),
        )
        if v
    )
    context = (
        f"A Find run is attached to this conversation. The user described a moment: "
        f'"{s["query"]}"\nIt searched — {searched}.\n\n'
        "Its best matches (reference: file at time — the language model's verdict 0–9 "
        "and reason — what found it):\n" + "\n".join(lines)
    )
    conv = chat.new_conversation()
    conv["title"] = ("About: " + s["query"])[:80]
    conv["refs"] = refs
    conv["attached"] = {
        "kind": "find",
        "id": run_id,
        "title": s["query"][:120],
        "text": context[: chat.ATTACHED_CHARS],
    }
    hello = (
        f"The Find run **{s['query'][:120]}** is attached — its {len(refs)} best matches. "
        "Ask about them; for anything beyond them I search the library."
    )
    conv["turns"] = [
        {"role": "assistant", "at": time.time(), "steps": [], "rounds": [hello], "text": hello}
    ]
    chat.save(conv)
    return conv["id"]

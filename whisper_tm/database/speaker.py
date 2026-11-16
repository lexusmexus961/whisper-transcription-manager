"""Voice vectors — own database alongside ``library.db``.

Why its own file and not a table in ``library.db``?

* Vectors are **always computable**. They are an index over audio,
  not user content. You can delete and refill your own file without
  touching transcripts — and without main database schema migration
  depending on it.
* They are many and large: a sizeable library reaches **hundreds of
  thousands of rows × 2 × 192 float32 each**, easily **half a gigabyte**.
  ``library.db`` is copied before each migration and regularly
  backed up; half a gigabyte of numeric data has no place there.
* The large recap run writes for hours. A separate WAL keeps this write
  traffic away from the database the interface works with.

The binding model for paths applies here too: **``abs_path`` is the
identity of a file** (``docs/PATHS.md``). ``rel_path`` depends on
the currently chosen root and is not suitable for comparison —
statistics has already learned that.

Three tables:

``speaker_vectors``
    One window, one vector. ``label`` (``SPEAKER_00``) is explicitly only
    a **diarization cut mark**, not an identity — it only applies within
    this one file.
``speaker_people``
    The named people.
``speaker_assignment``
    Which vector belongs to which person. Assignment hangs on the
    **vector**, never on a group number: groups change with linking
    and threshold, an once-assigned name must not.
``speaker_decisions`` / ``speaker_decision_voices``
    The history: every change to a name or to the trash, with the state
    of each voice before and after — the ground an undo stands on (see
    "History of decisions" below).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import string
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Stored in ``PRAGMA user_version``. The vectors are disposable — a
# version bump may mean "recompute" if in doubt. The names, the trash and
# the history are not: they are the user's decisions.
#
# 4 is the first release's schema, created whole by SCHEMA_SQL; a version
# after it adds its step to initialize().
SCHEMA_VERSION = 4

#: What a decision changed about one voice — bits of
#: ``speaker_decision_voices.touched``.
NAME, TRASH = 1, 2

#: Entries that only record an undo or a redo. They are not decisions of
#: their own: "which decision changed this voice last?" skips them, and
#: they are neither undone nor listed.
BOOKKEEPING = ("undo", "redo")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS speaker_vectors (
    id          INTEGER PRIMARY KEY,
    abs_path    TEXT    NOT NULL,
    label     TEXT    NOT NULL,
    windows     INTEGER NOT NULL,
    start_sec   REAL    NOT NULL,
    speech_sec REAL    NOT NULL,
    segments    INTEGER,
    /* The actual speech spans of this window as JSON [[a,b],[a,b],…] in seconds.

       A window is 30 sec of NET speech, glued together from pieces scattered
       across the whole recording — sometimes spread across hours. Older
       versions stored only the START, and the sample took 30 seconds straight
       from there. That's almost never the material the vector was made from:
       the straight 30 seconds can hold none of the target voice at all, while
       the window's own pieces, spread over the whole span, hold it again and
       again.

       The listener heard other people and was to decide on a group
       that the model formed because of a completely different voice.
       The whole assignment path depends on the sample telling the truth. */
    parts       TEXT,
    v_raw       BLOB,
    v_separated  BLOB,
    model      TEXT    NOT NULL,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(abs_path, label, windows, model)
);
CREATE INDEX IF NOT EXISTS idx_sv_path  ON speaker_vectors(abs_path);
CREATE INDEX IF NOT EXISTS idx_sv_model ON speaker_vectors(model);

/* COLLATE NOCASE is not a cosmetic fix here but the core of the matter:
   This table IS the identity. Keeping "alice" and "Alice" as two people
   splits exactly the person whose recognition the whole effort is for —
   and nobody would notice because both names would look right in the list.

   Example: a name saved as "alice" while all other names are
   capitalized ("Bob", "Carol", "Dave"). Next time typing "Alice" would
   have created a second person.

   NOCASE only folds ASCII — "Élan"/"élan" stay different. This is a
   known SQLite limitation and still much better than no folding; for
   the common case (first letter capital or lowercase) it's fully sufficient. */
CREATE TABLE IF NOT EXISTS speaker_people (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE COLLATE NOCASE,
    color TEXT
);

CREATE TABLE IF NOT EXISTS speaker_assignment (
    vector_id INTEGER NOT NULL REFERENCES speaker_vectors(id) ON DELETE CASCADE,
    person_id INTEGER          REFERENCES speaker_people(id) ON DELETE CASCADE,
    source    TEXT,
    set_at TEXT DEFAULT (datetime('now')),
    UNIQUE(vector_id)
);
CREATE INDEX IF NOT EXISTS idx_sa_person ON speaker_assignment(person_id);

-- ── Grouping ─────────────────────────────────────────────────────────────
-- A RUN is a complete grouping of the whole stock at a given operating
-- point. Why runs are allowed to stand side by side and are not
-- overwritten: a run takes minutes, and the user's map is open during that
-- time. A run that pulls the groups out from under the live view turns
-- every click into a gamble.
CREATE TABLE IF NOT EXISTS speaker_runs (
    id           INTEGER PRIMARY KEY,
    model       TEXT,
    threshold     REAL,
    link_method   TEXT,
    k            INTEGER,
    nodes       INTEGER,
    groups      INTEGER,
    components  INTEGER,
    duration_sec    REAL,
    computed_at TEXT DEFAULT (datetime('now')),
    hint      TEXT
);

CREATE TABLE IF NOT EXISTS speaker_groups (
    run_id  INTEGER NOT NULL REFERENCES speaker_runs(id) ON DELETE CASCADE,
    number   INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    files  INTEGER,
    x        REAL,
    y        REAL,
    name     TEXT,
    PRIMARY KEY (run_id, number)
);

-- Membership hangs on the run's GROUP NUMBER, not its own row ID: one fewer
-- column at hundreds of thousands of rows, and the number is only valid within its own run
-- anyway.
CREATE TABLE IF NOT EXISTS speaker_membership (
    run_id   INTEGER NOT NULL,
    vector_id INTEGER NOT NULL REFERENCES speaker_vectors(id) ON DELETE CASCADE,
    group_nr    INTEGER NOT NULL,
    x         REAL,
    y         REAL,
    PRIMARY KEY (run_id, vector_id)
);
CREATE INDEX IF NOT EXISTS idx_sm_group
    ON speaker_membership(run_id, group_nr);

-- ── Trash bin ────────────────────────────────────────────────────────────
-- Voices nobody cares about: game sounds, a TV in the background, a radio
-- announcement. They are NOT deleted — the vectors cost money and compute
-- time, and a mistake would otherwise be final.
--
-- Stored per VOICE, even though discarding happens tile by tile. The reason
-- is the group number: it is created anew on every run and means something
-- different after the next grouping. A trash bin that held group numbers
-- would throw away the wrong voices after a threshold change.
CREATE TABLE IF NOT EXISTS speaker_trash (
    vector_id      INTEGER PRIMARY KEY
                   REFERENCES speaker_vectors(id) ON DELETE CASCADE,
    discarded_at TEXT DEFAULT (datetime('now')),
    reason          TEXT
);

-- ── History of decisions ─────────────────────────────────────────────────
-- One row per decision (a click: a group named, a group discarded, a voice
-- renamed, an undo), and per voice it changed the state before and after.
-- Written in the same transaction as the change itself, so the history
-- can't miss one. `touched` says which of the two it changed (bit 1: the
-- name, bit 2: the trash) — the other one is not part of this decision.
--
-- `at` in UTC, as SQLite writes datetime('now'). `undone_by` points at the
-- entry that undid this one, while it is undone. `reconstructed`: rebuilt
-- from the times of the assignments and the trash, for what was decided
-- before this history existed.
CREATE TABLE IF NOT EXISTS speaker_decisions (
    id            INTEGER PRIMARY KEY,
    at            TEXT NOT NULL DEFAULT (datetime('now')),
    kind          TEXT NOT NULL,
    origin        TEXT,
    name          TEXT,
    target        INTEGER,
    voices        INTEGER NOT NULL DEFAULT 0,
    spread        INTEGER NOT NULL DEFAULT 0,
    summary       TEXT,
    reconstructed INTEGER NOT NULL DEFAULT 0,
    undone_by     INTEGER,
    undone_at     TEXT
);
CREATE TABLE IF NOT EXISTS speaker_decision_voices (
    decision_id   INTEGER NOT NULL
                  REFERENCES speaker_decisions(id) ON DELETE CASCADE,
    vector_id     INTEGER NOT NULL
                  REFERENCES speaker_vectors(id) ON DELETE CASCADE,
    touched       INTEGER NOT NULL,
    name_before   TEXT,
    source_before TEXT,
    name_after    TEXT,
    source_after  TEXT,
    trash_before  INTEGER,
    reason_before TEXT,
    trash_after   INTEGER,
    reason_after  TEXT,
    -- 1 = the name came by spreading, not by the click itself
    spread        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (decision_id, vector_id)
) WITHOUT ROWID;
-- "Which later decision changed this voice?" — and the cascade when a
-- recording's vectors are deleted.
CREATE INDEX IF NOT EXISTS idx_sdv_vector
    ON speaker_decision_voices(vector_id, decision_id);
"""


def _parts_json(parts) -> str | None:
    """The speech spans as compact JSON — or nothing.

    Two decimal places: anything finer than a hundredth is not meaningfully
    cut by ffmpeg anyway, and at a hundred thousand windows with 20 pieces
    each that's the difference between roughly 40 and 90 MB.
    """
    if not parts:
        return None
    try:
        pairs = [[round(float(a), 2), round(float(b), 2)] for a, b in parts]
    except (TypeError, ValueError):
        return None
    return json.dumps(pairs, separators=(",", ":")) if pairs else None


def as_blob(v) -> bytes | None:
    """Bring a vector into its stored form: float32, contiguous, raw.

    Deliberately WITHOUT normalization. What is stored is what the model
    delivers; normalization happens at computation time. Whoever discards
    the length can never recover it — and the length is a hint about
    quality.
    """
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32).ravel()
    return a.tobytes()


def from_blob(b) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32)


#: How alike an old voice and a new voice of the same recording must be for
#: the new one to take over the old one's decisions — the cosine of the two
#: labels' centroids (:meth:`SpeakerDB.replace_keeping_decisions`).
CARRY_AT_LEAST = 0.80
#: ... and how far ahead of an old voice that carries something else.
CARRY_MARGIN = 0.10


def match_voices(
    old: dict[tuple, np.ndarray], new: list[np.ndarray | None], decided: set[tuple]
) -> dict:
    """Which old voice each new window is — window by window.

    ``old``: the old voices of a recording, keyed ``(label, person id,
    trashed)`` — the windows of one label that carry the same decision —
    with their centroids. ``decided``: the old voices that carry anything:
    a name, the trash or an entry of the history. ``new``: the new windows'
    vectors, unit length (``None``: no vector).

    For a new window, the candidates are its most similar old voice — if at
    least :data:`CARRY_AT_LEAST` alike — and every old voice within
    :data:`CARRY_MARGIN` of it. The window takes the decided candidates
    when they all carry the same name and trash. It takes nothing (which
    of them it is stays open) when they carry different things, or when an
    undecided candidate is a different voice than the decided ones —
    someone else sounds just as much like it. An undecided candidate that
    is the same voice as a decided one is no rival: one person, cut in two
    by the old diarization, of whom only one half was ever named (e.g.
    0.98 to the named half, both one voice).

    Window by window, not label by label: a new diarization groups
    differently. For example: one new label holds a dozen windows —
    two of the old "Alice", three that sound like Bob, two like Carol,
    the rest people never named. Its label as a whole can still be close
    to old Alice; carried by label, all of them would be named Alice.

    Returns ``takes`` ({window index: [old voice, …]}, the most similar
    first), ``best`` ({window index: (old voice, similarity)}) and
    ``nearest`` ({old voice: (window index, similarity)}).
    """
    olds = sorted(old, key=lambda k: (k[0], str(k[1]), k[2]))
    out: dict = {"takes": {}, "best": {}, "nearest": {}}
    rows = [j for j, v in enumerate(new) if v is not None]
    if not olds or not rows:
        return out
    centroids = np.stack([old[k] for k in olds])
    S = centroids @ np.stack([new[j] for j in rows]).T
    same_voice = centroids @ centroids.T >= CARRY_AT_LEAST
    for col, j in enumerate(rows):
        order = [int(i) for i in np.argsort(-S[:, col], kind="stable")]
        sim = float(S[order[0], col])
        out["best"][j] = (olds[order[0]], round(sim, 3))
        if sim < CARRY_AT_LEAST:
            continue
        cand = [i for i in order if S[i, col] > sim - CARRY_MARGIN]
        dec = [i for i in cand if olds[i] in decided]
        if not dec or len({olds[i][1:] for i in dec}) != 1:
            continue
        if any(not same_voice[u, dec].any() for u in cand if olds[u] not in decided):
            continue
        out["takes"][j] = [olds[i] for i in dec]
    for i, o in enumerate(olds):
        col = int(np.argmax(S[i]))
        out["nearest"][o] = (rows[col], round(float(S[i, col]), 3))
    return out


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def fold_name(name: str | None) -> str | None:
    """A name as ``COLLATE NOCASE`` compares it — ASCII letters folded,
    nothing else ("Alice" = "alice", "Élan" ≠ "élan"). Python's ``lower``
    folds more and would call two stored people the same one."""
    return None if name is None else name.translate(_ASCII_LOWER)


def _epoch(at: str) -> float:
    """``datetime('now')`` text (UTC) → seconds."""
    try:
        return datetime.strptime(at[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
    except (TypeError, ValueError):
        return 0.0


class _Decision:
    """The entry the writes of one click are collected in."""

    __slots__ = ("kind", "origin", "name", "target", "id", "fresh")

    def __init__(self, kind: str, origin: str, name: str | None, target: int | None):
        self.kind, self.origin, self.name, self.target = kind, origin, name, target
        self.id: int | None = None
        self.fresh = False


class SpeakerDB:
    """A very small database wrapper — one connection per thread.

    Deliberately not derived from ``Database``: half the application hangs
    off that (migrations, backups, statistics). Here a schema and a handful
    of queries are enough.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False
        # Counts every change to the vectors or names. A process cache of
        # the matrix (core/speaker_inventory.py) depends on this: it needs
        # to notice that a transcription has pushed in new vectors since
        # the last load. A timestamp would be blurrier — two changes in
        # the same second would look like one.
        self._generation = 0
        # A second marker that ONLY listens to the stock.
        #
        # A name does not change the grouping. The centroids come from
        # path, marker and vector — naming a group moves none of them.
        # Yet the assignment cache hung on `generation`, and that moves
        # on EVERY change: every click on "that's Alice" threw the
        # computation away and let the next suggestion be recomputed from
        # scratch — around 30 seconds at tens of thousands of centroids,
        # exactly in the working step done a hundred times in a row.
        #
        # `inventory_generation` only moves when the AMOUNT of vectors
        # changes: new recording, deleted recording, trash bin.
        self._inventory_generation = 0
        # And one for the trash alone: a held grouping only needs to look
        # at the trash again when it changed — not on every new recording
        # of a running queue (core/speaker_assignment.discarded).
        self._trash_generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def inventory_generation(self) -> int:
        """Marker for everything that depends on the VECTORS, not on names."""
        return self._inventory_generation

    @property
    def trash_generation(self) -> int:
        """Marker for the trash alone."""
        return self._trash_generation

    def _changed(self, inventory: bool = False, trash: bool = False) -> None:
        with self._lock:
            self._generation += 1
            if inventory or trash:
                self._inventory_generation += 1
            if trash:
                self._trash_generation += 1

    # ── Connection ────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(self._db_path, timeout=30.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            c.execute("PRAGMA foreign_keys = ON")
            self._local.conn = c
        return c

    @contextmanager
    def connection(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """One transaction. ``immediate`` takes the write lock before the
        first read: what was read is still true when it is written — no
        other writer can come in between."""
        c = self._conn()
        if immediate and not c.in_transaction:
            c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise

    def initialize(self) -> None:
        """Create the schema — or check that an existing file is one this
        version knows: the first release's (4). A file of a development
        build before it, or of a newer version, is refused, never changed:
        the names, the trash and the history in it are the user's."""
        from whisper_tm.database import DatabaseVersionError

        with self._lock:
            if self._initialized:
                return
            with self.connection() as c:
                stored = c.execute("PRAGMA user_version").fetchone()[0]
                filled = (
                    c.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1").fetchone()
                    is not None
                )
            if stored > SCHEMA_VERSION:
                raise DatabaseVersionError(
                    f"The voice database {self._db_path} is from a newer version of "
                    f"Whisper Transcription Manager (schema {stored}; this one knows up to "
                    f"{SCHEMA_VERSION}). Update the program, or restore a backup."
                )
            if filled and stored < SCHEMA_VERSION:
                raise DatabaseVersionError(
                    f"The voice database {self._db_path} is from a development build "
                    f"before the first release (schema {stored}); this version opens "
                    f"schema {SCHEMA_VERSION}. Move it aside to start with an empty one."
                )
            with self.connection() as c:
                c.executescript(SCHEMA_SQL)
                if stored < SCHEMA_VERSION:
                    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._initialized = True

    #: How many copies of the voice database a backup folder keeps
    BACKUPS_KEPT = 3
    #: A new copy once the newest is older than this
    BACKUP_EVERY_DAYS = 7.0

    def backup(self, dest_dir: str | Path) -> Path | None:
        """A copy of the voice database (names, trash, history) in
        ``dest_dir`` as ``speaker_<time>.db``; the newest BACKUPS_KEPT stay.

        Made next to the database first and copied over in one go: SQLite's
        backup API page by page onto a network share hardly moves. The final
        name exists only once every byte is there."""
        import shutil

        db_path = Path(self._db_path)
        if not db_path.exists():
            return None
        dest = Path(dest_dir)
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Voice backup destination unusable (%s): %s", dest, e)
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        final = dest / f"{db_path.stem}_{ts}.db"
        staging = db_path.parent / f".{db_path.stem}_{ts}.tmp"
        partial = dest / f".{final.name}.part"
        try:
            out = sqlite3.connect(str(staging))
            try:
                self._conn().backup(out)
            finally:
                out.close()
            try:
                os.replace(staging, final)
            except OSError:
                shutil.copyfile(staging, partial)
                os.replace(partial, final)
        except (OSError, sqlite3.Error) as e:
            logger.warning("Voice database backup failed (%s): %s", final, e)
            for leftover in (partial, final):
                with contextlib.suppress(OSError):
                    leftover.unlink(missing_ok=True)
            return None
        finally:
            with contextlib.suppress(OSError):
                staging.unlink(missing_ok=True)
        kept = sorted(
            dest.glob(f"{db_path.stem}_*.db"), key=lambda f: f.stat().st_mtime, reverse=True
        )
        for old in kept[self.BACKUPS_KEPT :]:
            with contextlib.suppress(OSError):
                old.unlink()
        return final

    def backup_due(self, dest_dir: str | Path) -> Path | None:
        """``backup`` when the newest copy in ``dest_dir`` is older than
        BACKUP_EVERY_DAYS (or there is none)."""
        stem = Path(self._db_path).stem
        try:
            newest = max(
                (f.stat().st_mtime for f in Path(dest_dir).glob(f"{stem}_*.db")), default=0.0
            )
        except OSError as e:
            logger.warning("Voice backup destination unreadable (%s): %s", dest_dir, e)
            return None
        if time.time() - newest < self.BACKUP_EVERY_DAYS * 86400:
            return None
        return self.backup(dest_dir)

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ── Vectors ───────────────────────────────────────────────────────

    def write(self, abs_path: str, model: str, lines: Iterable[dict]) -> int:
        """Store all windows of ONE recording — replacing existing ones.

        Delete first, then insert: a second run over the same file
        (different operating point, repaired diarization) must not leave
        behind a mix of old and new windows. The window numbers of a new
        run mean something different from those of the old one.

        Only rows of THIS model are discarded — a checkpoint change should
        leave the old vectors standing for comparison.
        """
        self.initialize()
        lines = list(lines)
        with self.connection() as c:
            c.execute(
                "DELETE FROM speaker_vectors WHERE abs_path = ? AND model = ?", (abs_path, model)
            )
            c.executemany(
                "INSERT INTO speaker_vectors "
                "(abs_path, label, windows, start_sec, speech_sec, "
                " segments, parts, v_raw, v_separated, model) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        abs_path,
                        z["label"],
                        int(z["windows"]),
                        float(z["start_sec"]),
                        float(z["speech_sec"]),
                        int(z.get("segments") or 0),
                        _parts_json(z.get("parts")),
                        as_blob(z.get("v_raw")),
                        as_blob(z.get("v_separated")),
                        model,
                    )
                    for z in lines
                ],
            )
        self._changed(inventory=True)
        return len(lines)

    def replace_keeping_decisions(self, abs_path: str, model: str, lines: Iterable[dict]) -> dict:
        """Replace one recording's voices by a new run — and hand each old
        voice's decisions to the new windows that are the same voice.

        Meant for re-running every recording without speech spans with
        voices only. Voices without stored
        spans: their sample is the window's first 30 s, and the extractor
        can't cut them cleanly. A new run stores the spans — but a plain
        :meth:`write` drops the names, the trash and the history, which hang
        on the old rows' ids.

        **By voice, not by label.** A label is only a cut mark of one
        diarization. The new run names its speakers after the transcript's,
        the old voices came from whichever diarization made them — a good
        share of the recordings with old voices can have a transcript
        without speakers or labels that disagree with it — and a new run
        also groups differently. So every new window is compared with the
        old voices (:func:`match_voices`) and takes a decision only from
        the old voice it clearly is.

        **All or nothing, per recording.** If any old voice that carries a
        decision (a name, the trash, an entry of the history) finds no new
        window that clearly is it, nothing changes: the recording keeps its
        old voices and the new run is dropped. Old voices without any
        decision need no counterpart — nothing hangs on them. Everything
        happens in one transaction that holds the write lock from the first
        read, so a click in the interface can't fall between reading and
        replacing.

        Otherwise, in that transaction: the old windows go, the new ones
        come in, every new window gets the name and trash of the old voice
        it is (with their original times), and the history follows in time —
        a new window takes the entries of the nearest old window of its old
        voice, and every old window hands its entries to the nearest new
        window that took its voice, so no entry loses its voices and undo
        and redo keep working. The entries' counts and samples are computed
        anew.

        Returns what happened, for the job's log: ``replaced``, what was
        carried to which new labels and the old voices that kept the
        recording as it was.
        """
        self.initialize()
        lines = list(lines)
        t0 = time.monotonic()
        with self.connection(immediate=True) as c:
            old = c.execute(
                "SELECT v.id, v.label, v.start_sec, v.v_separated, "
                "       z.person_id, p.name, z.source, z.set_at, "
                "       t.vector_id IS NOT NULL AS trashed, t.reason, t.discarded_at "
                "FROM speaker_vectors v "
                "LEFT JOIN speaker_assignment z ON z.vector_id = v.id "
                "LEFT JOIN speaker_people p ON p.id = z.person_id "
                "LEFT JOIN speaker_trash t ON t.vector_id = v.id "
                "WHERE v.abs_path = ? AND v.model = ? ORDER BY v.id",
                (abs_path, model),
            ).fetchall()
            journal: dict[int, list] = {}
            ids = [r["id"] for r in old]
            for a in range(0, len(ids), 900):
                chunk = ids[a : a + 900]
                for r in c.execute(
                    "SELECT * FROM speaker_decision_voices WHERE vector_id IN "
                    f"({','.join('?' * len(chunk))}) ORDER BY decision_id, vector_id",
                    chunk,
                ):
                    journal.setdefault(r["vector_id"], []).append(r)

            def unit(v) -> np.ndarray | None:
                if v is None:
                    return None
                v = np.asarray(v, np.float32).ravel()
                n = float(np.linalg.norm(v))
                return v / n if v.size and n > 0 else None

            # An old voice: the windows of one label that carry the same
            # decision — (label, person id, trashed)
            by_voice: dict[tuple, list] = {}
            for r in old:
                by_voice.setdefault((r["label"], r["person_id"], bool(r["trashed"])), []).append(r)
            centroids: dict[tuple, np.ndarray] = {}
            for k, rows in by_voice.items():
                vs = [u for u in (unit(from_blob(r["v_separated"])) for r in rows) if u is not None]
                if vs:
                    m = np.mean(vs, axis=0)
                    centroids[k] = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)
            decided = {
                k
                for k, rows in by_voice.items()
                if k[1] or k[2] or any(r["id"] in journal for r in rows)
            }
            m = match_voices(centroids, [unit(z.get("v_separated")) for z in lines], decided)
            taken = {k for ks in m["takes"].values() for k in ks}

            def who(k) -> str:
                r = by_voice[k][0]
                return r["name"] or ("trash" if r["trashed"] else "history")

            carried: dict[tuple, dict] = {}
            for j, ks in m["takes"].items():
                e = carried.setdefault(
                    (lines[j]["label"], who(ks[0])),
                    {
                        "new": lines[j]["label"],
                        "name": who(ks[0]),
                        "from": set(),
                        "windows": 0,
                        "sim": 1.0,
                    },
                )
                e["from"] |= {k[0] for k in ks}
                e["windows"] += 1
                e["sim"] = min(e["sim"], m["best"][j][1])
            out: dict = {
                "replaced": False,
                "windows": len(lines),
                "labels_old": len({r["label"] for r in old}),
                "labels_new": len({z["label"] for z in lines}),
                "carried": [{**e, "from": sorted(e["from"])} for _k, e in sorted(carried.items())],
                "unplaced": [
                    {
                        "old": k[0],
                        "name": who(k),
                        "best": (lines[m["nearest"][k][0]]["label"] if k in m["nearest"] else None),
                        "sim": m["nearest"][k][1] if k in m["nearest"] else None,
                    }
                    for k in sorted(decided - taken, key=lambda k: (k[0], str(k[1]), k[2]))
                ],
                "names": 0,
                "trash": 0,
                "history_entries": 0,
                "secs": 0.0,
            }
            if out["unplaced"]:
                out["secs"] = round(time.monotonic() - t0, 2)
                return out

            # ── Replace ─────────────────────────────────────────────────
            # The new windows come in first, under a passing model name:
            # their ids lie above every old one, so an id a page still
            # shows never comes back as another voice (SQLite hands out
            # the highest id + 1 — deleting first would hand the old ids
            # out again), and UNIQUE(abs_path, label, windows, model)
            # doesn't meet the old rows. Then the old go, and the new take
            # the model's name.
            passing = f"{model}#renewing"
            new_ids: list[int] = []
            for z in lines:
                new_ids.append(
                    int(
                        c.execute(
                            "INSERT INTO speaker_vectors "
                            "(abs_path, label, windows, start_sec, speech_sec, "
                            " segments, parts, v_raw, v_separated, model) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (
                                abs_path,
                                z["label"],
                                int(z["windows"]),
                                float(z["start_sec"]),
                                float(z["speech_sec"]),
                                int(z.get("segments") or 0),
                                _parts_json(z.get("parts")),
                                as_blob(z.get("v_raw")),
                                as_blob(z.get("v_separated")),
                                passing,
                            ),
                        ).lastrowid
                    )
                )
            for a in range(0, len(ids), 900):
                chunk = ids[a : a + 900]
                c.execute(
                    f"DELETE FROM speaker_vectors WHERE id IN ({','.join('?' * len(chunk))})", chunk
                )
            c.execute(
                "UPDATE speaker_vectors SET model = ? WHERE abs_path = ? AND model = ?",
                (model, abs_path, passing),
            )

            names, trash = [], []
            for j, ks in m["takes"].items():
                first = by_voice[ks[0]][0]
                if first["person_id"]:
                    names.append((new_ids[j], first["person_id"], first["source"], first["set_at"]))
                if first["trashed"]:
                    trash.append((new_ids[j], first["reason"], first["discarded_at"]))
            c.executemany(
                "INSERT INTO speaker_assignment (vector_id, person_id, source, set_at) "
                "VALUES (?, ?, ?, COALESCE(?, datetime('now')))",
                names,
            )
            c.executemany(
                "INSERT INTO speaker_trash (vector_id, reason, discarded_at) "
                "VALUES (?, ?, COALESCE(?, datetime('now')))",
                trash,
            )
            out["names"], out["trash"] = len(names), len(trash)

            # The history follows in time: each new window takes the entries
            # of the nearest old window of its old voice; each old window
            # hands its entries to the nearest new window that took its voice
            takers: dict[tuple, list[int]] = {}
            for j, ks in m["takes"].items():
                for k in ks:
                    takers.setdefault(k, []).append(j)
            links: dict[int, set[int]] = {j: set() for j in m["takes"]}
            for j, ks in m["takes"].items():
                at = float(lines[j]["start_sec"])
                rows = [r for k in ks for r in by_voice[k]]
                links[j].add(min(rows, key=lambda r: abs(float(r["start_sec"] or 0) - at))["id"])
            for k, js in takers.items():
                for r in by_voice[k]:
                    at = float(r["start_sec"] or 0)
                    links[min(js, key=lambda j: abs(float(lines[j]["start_sec"]) - at))].add(
                        r["id"]
                    )
            touched: set[int] = set()
            copies = []
            for j, from_ids in links.items():
                seen: set[int] = set()
                for o in sorted(from_ids):
                    for e in journal.get(o, []):
                        if e["decision_id"] in seen:
                            continue
                        seen.add(e["decision_id"])
                        touched.add(e["decision_id"])
                        copies.append(
                            (
                                e["decision_id"],
                                new_ids[j],
                                e["touched"],
                                e["name_before"],
                                e["source_before"],
                                e["name_after"],
                                e["source_after"],
                                e["trash_before"],
                                e["reason_before"],
                                e["trash_after"],
                                e["reason_after"],
                                e["spread"],
                            )
                        )
            c.executemany(
                "INSERT OR IGNORE INTO speaker_decision_voices "
                "(decision_id, vector_id, touched, name_before, source_before, "
                " name_after, source_after, trash_before, reason_before, "
                " trash_after, reason_after, spread) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                copies,
            )
            # Counts and samples of the entries that followed
            if touched:
                summaries = self._summarize(c, sorted(touched))
                for did in sorted(touched):
                    voices, spread = c.execute(
                        "SELECT COUNT(*), COALESCE(SUM(spread), 0) "
                        "FROM speaker_decision_voices WHERE decision_id = ?",
                        (did,),
                    ).fetchone()
                    c.execute(
                        "UPDATE speaker_decisions SET voices = ?, spread = ?, summary = ? "
                        "WHERE id = ?",
                        (voices, spread, json.dumps(summaries[did]), did),
                    )
            out["history_entries"] = len(touched)
            out["replaced"] = True
        # How long the write lock was held — clicks in the interface wait
        out["secs"] = round(time.monotonic() - t0, 2)
        self._changed(inventory=True, trash=True)
        return out

    def has_vectors(self, abs_path: str, model: str | None = None) -> bool:
        self.initialize()
        with self.connection() as c:
            if model:
                r = c.execute(
                    "SELECT 1 FROM speaker_vectors WHERE abs_path = ? AND model = ? LIMIT 1",
                    (abs_path, model),
                ).fetchone()
            else:
                r = c.execute(
                    "SELECT 1 FROM speaker_vectors WHERE abs_path = ? LIMIT 1", (abs_path,)
                ).fetchone()
        return r is not None

    def windows(self, abs_path: str, model: str | None = None) -> list[dict]:
        """All windows of a recording, sorted by time."""
        self.initialize()
        sql = (
            "SELECT v.*, p.name AS person, z.source "
            "FROM speaker_vectors v "
            "LEFT JOIN speaker_assignment z ON z.vector_id = v.id "
            "LEFT JOIN speaker_people  p ON p.id = z.person_id "
            "WHERE v.abs_path = ?"
        )
        args: list = [abs_path]
        if model:
            sql += " AND v.model = ?"
            args.append(model)
        sql += " ORDER BY v.start_sec, v.label"
        with self.connection() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def remove(self, abs_path: str) -> int:
        self.initialize()
        with self.connection() as c:
            cur = c.execute("DELETE FROM speaker_vectors WHERE abs_path = ?", (abs_path,))
        self._changed(inventory=True)
        return cur.rowcount or 0

    def all_vectors(
        self, model: str | None = None, preferred: str = "v_separated"
    ) -> tuple[np.ndarray, list[dict]]:
        """Load the whole stock as a matrix — the basis for kNN.

        ``preferred`` picks the version; where a row lacks it, it falls back
        to the other one. Rows with no vector at all are dropped.

        Return: ``(X, meta)`` with ``X.shape == (n, dim)``, NOT normalized.
        """
        self.initialize()
        sql = (
            "SELECT id, abs_path, label, windows, start_sec, speech_sec, "
            "segments, model, v_raw, v_separated FROM speaker_vectors"
        )
        args: list = []
        if model:
            sql += " WHERE model = ?"
            args.append(model)
        sql += " ORDER BY id"
        vs, meta = [], []
        with self.connection() as c:
            self._collect(c.execute(sql, args), preferred, vs, meta)
        if not vs:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.stack(vs).astype(np.float32), meta

    @staticmethod
    def _collect(rows, preferred: str, vs: list, meta: list) -> None:
        """Rows → vectors and their metadata, the one rule for both loaders."""
        other = "v_raw" if preferred == "v_separated" else "v_separated"
        for r in rows:
            v = from_blob(r[preferred])
            if v is None:
                v = from_blob(r[other])
            if v is None or v.size == 0:
                continue
            vs.append(v)
            meta.append(
                {
                    k: r[k]
                    for k in (
                        "id",
                        "abs_path",
                        "label",
                        "windows",
                        "start_sec",
                        "speech_sec",
                        "segments",
                        "model",
                    )
                }
            )

    def vector_ids(self, model: str | None = None) -> np.ndarray:
        """The ids of the stock, ascending — a fraction of a second for
        hundreds of thousands of rows, through
        the model index, without touching a vector."""
        self.initialize()
        sql, args = "SELECT id FROM speaker_vectors", []
        if model:
            sql += " WHERE model = ?"
            args.append(model)
        with self.connection() as c:
            ids = [r[0] for r in c.execute(sql + " ORDER BY id", args)]
        return np.array(ids, dtype=np.int64)

    def vectors_by_id(
        self, ids, model: str | None = None, preferred: str = "v_separated"
    ) -> tuple[np.ndarray, list[dict]]:
        """As :meth:`all_vectors`, for the given ids only (ascending) — what
        a held inventory needs when the queue added a few recordings."""
        self.initialize()
        ids = [int(i) for i in ids]
        vs, meta = [], []
        cond = " AND model = ?" if model else ""
        with self.connection() as c:
            for a in range(0, len(ids), 500):
                part = ids[a : a + 500]
                args: list = [*part] + ([model] if model else [])
                self._collect(
                    c.execute(
                        "SELECT id, abs_path, label, windows, start_sec, speech_sec, "
                        "segments, model, v_raw, v_separated FROM speaker_vectors "
                        f"WHERE id IN ({','.join('?' * len(part))}){cond} ORDER BY id",
                        args,
                    ),
                    preferred,
                    vs,
                    meta,
                )
        if not vs:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.stack(vs).astype(np.float32), meta

    # ── People ────────────────────────────────────────────────────────

    def people(self) -> list[dict]:
        self.initialize()
        with self.connection() as c:
            return [
                dict(r)
                for r in c.execute(
                    "SELECT p.id, p.name, p.color, "
                    "       (SELECT COUNT(*) FROM speaker_assignment z "
                    "         WHERE z.person_id = p.id) AS voices "
                    "FROM speaker_people p ORDER BY p.name"
                ).fetchall()
            ]

    def map_names(self, paths: list[str] | None = None) -> dict[str, dict[str, str]]:
        """The name each speaker label carries on the voice map, for every
        recording that has one (``paths``: of these only): ``{abs_path:
        {label: name}}``.

        The first rule of :func:`core.speaker.names_for_labels`: the name
        most of a label's named windows carry, if more than half of them do.
        One grouped query over the assignments for the meaning search
        (rag/speakers.py) — seconds at a million vectors; a few thousand
        ``paths`` go through the path index in well under a second.
        """
        from collections import Counter, defaultdict

        self.initialize()
        votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        head = (
            "SELECT v.abs_path, v.label, p.name, COUNT(*) "
            "FROM speaker_vectors v "
            "JOIN speaker_assignment z ON z.vector_id = v.id "
            "JOIN speaker_people p ON p.id = z.person_id "
        )
        with self.connection() as c:
            if paths is None:
                rows = c.execute(head + "GROUP BY v.abs_path, v.label, p.id").fetchall()
            else:
                rows = []
                want = list(dict.fromkeys(paths))
                for k in range(0, len(want), 500):
                    part = want[k : k + 500]
                    rows += c.execute(
                        head + f"WHERE v.abs_path IN ({','.join('?' * len(part))}) "
                        "GROUP BY v.abs_path, v.label, p.id",
                        part,
                    ).fetchall()
            for path, label, name, n in rows:
                votes[(path, label)][name] += n
        out: dict[str, dict[str, str]] = {}
        for (path, label), c in votes.items():
            ((name, n),) = c.most_common(1)
            if 2 * n > sum(c.values()):
                out.setdefault(path, {})[label] = name
        return out

    def people_stats(self) -> list[dict]:
        """Per person: named voices (and how many by spreading), speech,
        recordings, how many voices carry their speech spans, when they were
        named, and a few voices to listen to — the People tab."""
        from whisper_tm.core.voice_extract import date_from_name

        self.initialize()
        per: dict[str, dict] = {}
        with self.connection() as c:
            for name, vid, source, set_at, abs_path, label, speech, spans in c.execute(
                "SELECT p.name, v.id, z.source, z.set_at, v.abs_path, v.label, "
                "       v.speech_sec, v.parts IS NOT NULL "
                "FROM speaker_assignment z "
                "JOIN speaker_people p ON p.id = z.person_id "
                "JOIN speaker_vectors v ON v.id = z.vector_id"
            ):
                e = per.setdefault(
                    name,
                    {
                        "name": name,
                        "voices": 0,
                        "spread": 0,
                        "speech_sec": 0.0,
                        "files": set(),
                        "with_spans": 0,
                        "first_named": set_at,
                        "last_named": set_at,
                        "dates": [],
                        "speakers": {},
                    },
                )
                e["voices"] += 1
                e["spread"] += 1 if (source or "").endswith("-spread") else 0
                e["speech_sec"] += speech or 0.0
                e["files"].add(abs_path)
                e["with_spans"] += 1 if spans else 0
                if set_at and set_at < (e["first_named"] or set_at):
                    e["first_named"] = set_at
                if set_at and set_at > (e["last_named"] or ""):
                    e["last_named"] = set_at
                sp = e["speakers"].setdefault((abs_path, label), [0.0, vid, -1.0])
                sp[0] += speech or 0.0
                if (speech or 0.0) > sp[2]:
                    sp[1], sp[2] = vid, speech or 0.0
        out = []
        for e in per.values():
            dates = sorted(d for d in (date_from_name(f) for f in e["files"]) if d)
            best = sorted(e["speakers"].items(), key=lambda kv: -kv[1][0])[:6]
            out.append(
                {
                    "name": e["name"],
                    "voices": e["voices"],
                    "spread": e["spread"],
                    "speech_hours": round(e["speech_sec"] / 3600, 2),
                    "recordings": len(e["files"]),
                    "with_spans": e["with_spans"],
                    "first_named": e["first_named"],
                    "last_named": e["last_named"],
                    "first_date": dates[0] if dates else "",
                    "last_date": dates[-1] if dates else "",
                    "dated": len(dates),
                    "samples": [
                        {
                            "vector_id": v[1],
                            "abs_path": k[0],
                            "label": k[1],
                            "speech_sec": round(v[0], 1),
                        }
                        for k, v in best
                    ],
                }
            )
        out.sort(key=lambda e: -e["voices"])
        return out

    def person_id(self, name: str, create: bool = True, color: str | None = None) -> int | None:
        self.initialize()
        name = (name or "").strip()
        if not name:
            return None
        with self.connection() as c:
            r = c.execute("SELECT id FROM speaker_people WHERE name = ?", (name,)).fetchone()
            if r:
                return int(r["id"])
            if not create:
                return None
            cur = c.execute("INSERT INTO speaker_people (name, color) VALUES (?, ?)", (name, color))
            return int(cur.lastrowid)

    def assign(
        self, vector_id: int, person: str | None, source: str = "human", origin: str = "map"
    ) -> None:
        """Assign a vector to a person. ``person=None`` removes the
        assignment again. Either way it is one entry of the history."""
        person = (person or "").strip() or None
        self.write_names(
            [(int(vector_id), person, source if person else None)],
            kind="name" if person else "unname",
            origin=origin,
            name=person,
        )

    def person_vectors(
        self, preferred: str = "v_separated", model: str | None = None, ids_out: dict | None = None
    ) -> dict[str, np.ndarray]:
        """Per named person, their vectors as a matrix (not normalized).

        This is the gallery recognition votes with, so two kinds of named
        voices stay out:

        * **the trash** — discarding keeps the name, and a wrongly named
          voice thrown out in the review still said "this is X" in every
          recognition;
        * **other models** (``model``) — the matrix they are compared with
          is read for one model; a checkpoint change keeps the old rows,
          whose similarities mean nothing against the new ones.

        ``ids_out``: a dict that receives ``{person: vector ids}`` in the
        rows' order — so a caller holding the gallery can leave one
        recording's voices out without reading the vectors again.
        """
        self.initialize()
        other = "v_raw" if preferred == "v_separated" else "v_separated"
        out: dict[str, list[np.ndarray]] = {}
        sql = (
            "SELECT p.name, z.vector_id, v.v_raw, v.v_separated "
            "FROM speaker_assignment z "
            "JOIN speaker_people  p ON p.id = z.person_id "
            "JOIN speaker_vectors  v ON v.id = z.vector_id "
            "WHERE z.vector_id NOT IN (SELECT vector_id FROM speaker_trash)"
        )
        args: list = []
        if model:
            sql += " AND v.model = ?"
            args.append(model)
        ids: dict[str, list[int]] = {}
        with self.connection() as c:
            for r in c.execute(sql, args):
                v = from_blob(r[preferred])
                if v is None:
                    v = from_blob(r[other])
                if v is not None and v.size:
                    out.setdefault(r["name"], []).append(v)
                    ids.setdefault(r["name"], []).append(int(r["vector_id"]))
        if ids_out is not None:
            ids_out.update({n: np.asarray(v, dtype=np.int64) for n, v in ids.items()})
        return {n: np.stack(vs).astype(np.float32) for n, vs in out.items()}

    def person_spots(self, name: str) -> list[tuple[str, str]]:
        """Where this person speaks — as (recording, marker) pairs.

        This is the bridge from speaker recognition to full-text search:
        a text search limited to the recordings of a selected speaker
        (e.g. "search for x, but only where Alice speaks").

        Both levels come back together: the path for "only these
        recordings" and the diarization marker for "only their sentences".
        `speaker_vectors.label` and `segments.speaker` are the same
        value (`SPEAKER_00`), so they match without any conversion.
        """
        self.initialize()
        with self.connection() as c:
            return [
                (r["abs_path"], r["label"])
                for r in c.execute(
                    "SELECT DISTINCT v.abs_path, v.label "
                    "FROM speaker_assignment z "
                    "JOIN speaker_vectors v ON v.id = z.vector_id "
                    "JOIN speaker_people p ON p.id = z.person_id "
                    "WHERE p.name = ? COLLATE NOCASE",
                    (name,),
                )
            ]

    def names_per_vector(self) -> dict[int, str]:
        """``{vector_id: person name}`` for all named voices.

        The map needs this in one go: it draws up to thousands of points,
        and a query per point would be the classic N+1 trap.
        """
        self.initialize()
        with self.connection() as c:
            return {
                r["vector_id"]: r["name"]
                for r in c.execute(
                    "SELECT z.vector_id, p.name FROM speaker_assignment z "
                    "JOIN speaker_people p ON p.id = z.person_id"
                )
            }

    def assign_many(
        self, vector_ids, person: str, source: str = "human", origin: str | None = None
    ) -> int:
        """Assign many vectors to a person in ONE go.

        The assignment flow gives a name to a whole group and then spreads
        it — that's quickly thousands of rows. A `assign()` per row would
        mean a separate transaction and a separate `_changed()` per row;
        at 2,000 voices that throws away the card cache 2,000 times.
        """
        ids = [int(v) for v in vector_ids]
        if not ids:
            return 0
        person = (person or "").strip()
        if not person:
            raise ValueError("no name")
        self.write_names(
            [(v, person, source) for v in ids],
            kind="name",
            origin=origin or source,
            name=person,
            spread=source.endswith("-spread"),
        )
        return len(ids)

    # ── Grouping runs ─────────────────────────────────────────────────

    # ── Overview ──────────────────────────────────────────────────────

    def stats_data(self) -> dict:
        self.initialize()
        with self.connection() as c:
            row = c.execute(
                "SELECT COUNT(*) AS windows, COUNT(DISTINCT abs_path) AS files, "
                "       COALESCE(SUM(speech_sec), 0) AS speech_sec "
                "FROM speaker_vectors"
            ).fetchone()
            people = c.execute("SELECT COUNT(*) AS n FROM speaker_people").fetchone()["n"]
            named = c.execute(
                "SELECT COUNT(*) AS n FROM speaker_assignment WHERE person_id IS NOT NULL"
            ).fetchone()["n"]
            models = [
                dict(r)
                for r in c.execute(
                    "SELECT model, COUNT(*) AS n FROM speaker_vectors "
                    "GROUP BY model ORDER BY n DESC"
                ).fetchall()
            ]
            # Raw and separated vectors are NOT comparable. A mix in the
            # stock is the quietest imaginable error: both columns have
            # the same shape, both numbers look plausible, and it only
            # shows up when someone wonders why nothing matches. So count
            # and show it.
            revision = c.execute(
                "SELECT SUM(v_separated IS NOT NULL) AS separated, "
                "       SUM(v_separated IS NULL AND v_raw IS NOT NULL) AS raw "
                "FROM speaker_vectors"
            ).fetchone()
        p = Path(self._db_path)
        separated = int(revision["separated"] or 0)
        raw = int(revision["raw"] or 0)
        return {
            "windows": row["windows"],
            "files": row["files"],
            "speech_hours": round(row["speech_sec"] / 3600, 1),
            "people": people,
            "named_voices": named,
            "models": models,
            "revision_separated": separated,
            "revision_raw": raw,
            # Both at once means: the stock falls into two halves that
            # never find each other. Must be visible.
            "revisions_mixed": bool(separated and raw),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "path": str(p),
        }

    # ── Trash bin ─────────────────────────────────────────────────────

    def trash_ids(self) -> set[int]:
        """Which voices have been discarded?

        As a set, because the only caller (the card) filters with it —
        once per card build over tens of thousands of rows, and `in` on a list would be
        quadratic there.
        """
        c = self._conn()
        return {r[0] for r in c.execute("SELECT vector_id FROM speaker_trash")}

    def move_to_trash(self, vector_ids, reason: str = "", origin: str | None = None) -> int:
        """Discard voices. Discarding twice is not an error."""
        ids = [int(v) for v in vector_ids]
        if not ids:
            return 0
        self.write_trash(
            ids, True, reason=reason or None, kind="trash", origin=origin or reason or "map"
        )
        return len(ids)

    def restore_from_trash(self, vector_ids=None, origin: str | None = None) -> int:
        """Restore. Without an argument: everything."""
        if vector_ids is None:
            ids = sorted(self.trash_ids())
            n = len(ids)
        else:
            ids = [int(v) for v in vector_ids]
            if not ids:
                return 0
            n = len(ids)
        self.write_trash(ids, False, kind="restore", origin=origin or "map")
        return n

    # ── History of decisions ──────────────────────────────────────────
    #
    # After assigning lots of voices, one may notice a bit later that a
    # speaker should have been kept after all. Without a history the
    # only way back was the trash map: every discarded voice in one
    # picture, with nothing saying which click put which there.
    #
    # Every change to a name or to the trash now leaves an entry: which
    # voices, and each one's state before and after. That is what an undo
    # needs to put back exactly what the click changed — and ONLY that.
    #
    # **Not damaging the rest.** Decisions build on each other: a name
    # spreads, a later click discards the voices an earlier one freed. An
    # undo therefore reverts a voice only when this entry is the LAST
    # decision that changed it (on the same thing — name or trash) and the
    # value is still the one it left. Everything else is left alone and
    # counted, so a later decision is never overwritten by an older one.
    # Undo and redo entries are bookkeeping: they don't count as "the last
    # decision" — an undone entry and its undo cancel each other out.

    @contextmanager
    def decision(
        self, kind: str, origin: str = "", name: str | None = None, target: int | None = None
    ) -> Iterator[_Decision]:
        """Collect every change inside into ONE entry of the history.

        The guided assignment names a group and then spreads the name —
        two writes, one click, one entry: undoing it must take both back.
        A nested call joins the outer entry. Without an open entry, each
        write makes its own.
        """
        held = getattr(self._local, "decision", None)
        if held is not None:
            yield held
            return
        d = _Decision(kind, origin or "", name, target)
        self._local.decision = d
        try:
            yield d
        finally:
            self._local.decision = None
            self._local.last_decision = d.id
            if d.id is not None:
                self._finish_decision(d.id)

    def last_decision_id(self) -> int | None:
        """The entry this thread's last write made — for an answer that
        offers "undo" right away."""
        return getattr(self._local, "last_decision", None)

    def _decision_id(self, c: sqlite3.Connection) -> int:
        """The entry the current writes belong to — made on the first one,
        inside the same transaction as the change."""
        d = getattr(self._local, "decision", None)
        if d is None:
            raise RuntimeError("write outside a decision")
        if d.id is None:
            d.id = int(
                c.execute(
                    "INSERT INTO speaker_decisions (kind, origin, name, target) VALUES (?,?,?,?)",
                    (d.kind, d.origin, d.name, d.target),
                ).lastrowid
            )
            d.fresh = True
        return d.id

    def _ids_table(self, c: sqlite3.Connection, ids) -> str:
        """The ids as a temporary table to join against — a list of hundreds
        of thousands of ids is too long for ``IN (…)``."""
        c.execute("CREATE TEMP TABLE IF NOT EXISTS _decision_ids (id INTEGER PRIMARY KEY)")
        c.execute("DELETE FROM _decision_ids")
        c.executemany(
            "INSERT OR IGNORE INTO _decision_ids (id) VALUES (?)", ((int(v),) for v in ids)
        )
        return "_decision_ids"

    def _still_stored(self, c: sqlite3.Connection, ids) -> set[int]:
        """The ids among ``ids`` that are still stored. A recording's voices
        can be renewed while a page shows the old ones
        (:meth:`replace_keeping_decisions`); a click on such a voice skips
        it instead of failing on the missing row."""
        t = self._ids_table(c, ids)
        return {
            r[0] for r in c.execute(f"SELECT v.id FROM speaker_vectors v JOIN {t} i ON i.id = v.id")
        }

    def _names_of(self, c: sqlite3.Connection, ids) -> dict[int, tuple[str, str]]:
        """``{vector_id: (name, source)}`` for the named ones among ``ids``."""
        t = self._ids_table(c, ids)
        return {
            r[0]: (r[1], r[2])
            for r in c.execute(
                "SELECT z.vector_id, p.name, z.source FROM speaker_assignment z "
                f"JOIN {t} i ON i.id = z.vector_id "
                "JOIN speaker_people p ON p.id = z.person_id"
            )
        }

    def _trash_of(self, c: sqlite3.Connection, ids) -> dict[int, str | None]:
        """``{vector_id: reason}`` for the discarded ones among ``ids``."""
        t = self._ids_table(c, ids)
        return {
            r[0]: r[1]
            for r in c.execute(
                f"SELECT t.vector_id, t.reason FROM speaker_trash t "
                f"JOIN {t} i ON i.id = t.vector_id"
            )
        }

    _JOURNAL_SQL = (
        "INSERT INTO speaker_decision_voices (decision_id, vector_id, touched, "
        " name_before, source_before, name_after, source_after, "
        " trash_before, reason_before, trash_after, reason_after, spread) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
        # One click that changes a voice twice (restored, then named) is
        # still one row: the FIRST before and the LAST after per thing.
        "ON CONFLICT(decision_id, vector_id) DO UPDATE SET "
        " name_before = CASE WHEN touched & 1 THEN name_before "
        "               ELSE excluded.name_before END, "
        " source_before = CASE WHEN touched & 1 THEN source_before "
        "               ELSE excluded.source_before END, "
        " name_after = CASE WHEN excluded.touched & 1 THEN excluded.name_after "
        "               ELSE name_after END, "
        " source_after = CASE WHEN excluded.touched & 1 THEN excluded.source_after "
        "               ELSE source_after END, "
        " trash_before = CASE WHEN touched & 2 THEN trash_before "
        "               ELSE excluded.trash_before END, "
        " reason_before = CASE WHEN touched & 2 THEN reason_before "
        "               ELSE excluded.reason_before END, "
        " trash_after = CASE WHEN excluded.touched & 2 THEN excluded.trash_after "
        "               ELSE trash_after END, "
        " reason_after = CASE WHEN excluded.touched & 2 THEN excluded.reason_after "
        "               ELSE reason_after END, "
        " spread = CASE WHEN excluded.touched & 1 THEN excluded.spread "
        "               ELSE spread END, "
        " touched = touched | excluded.touched"
    )

    def _person_ids(self, c: sqlite3.Connection, names) -> dict[str, int]:
        """Person ids for these names, made where missing — on the open
        transaction (`person_id` would commit it halfway)."""
        out: dict[str, int] = {}
        for n in {x for x in names if x}:
            r = c.execute("SELECT id FROM speaker_people WHERE name = ?", (n,)).fetchone()
            out[n] = (
                int(r[0])
                if r
                else int(c.execute("INSERT INTO speaker_people (name) VALUES (?)", (n,)).lastrowid)
            )
        return out

    def _apply_names(self, c: sqlite3.Connection, changes, spread: bool = False) -> list[int]:
        """Set names inside the open transaction and write their history.

        ``changes``: ``(vector_id, name or None, source)``. A voice that
        already carries the name (in any case — the table folds ASCII case)
        is not changed and not written down: an undo must not touch what
        the click didn't. Returns the ids that changed.
        """
        changes = list(changes)
        if changes:
            here = self._still_stored(c, [v for v, _n, _s in changes])
            changes = [ch for ch in changes if ch[0] in here]
        if not changes:
            return []
        before = self._names_of(c, [v for v, _n, _s in changes])
        pids = self._person_ids(c, [n for _v, n, _s in changes])
        put, drop, journal = [], [], []
        done: list[int] = []
        for vid, new, source in changes:
            old = before.get(vid)
            if fold_name(old[0] if old else None) == fold_name(new):
                continue
            if new is None:
                drop.append((vid,))
            else:
                put.append((vid, pids[new], source))
            journal.append(
                [
                    vid,
                    NAME,
                    old[0] if old else None,
                    old[1] if old else None,
                    new,
                    source if new is not None else None,
                    None,
                    None,
                    None,
                    None,
                    1 if spread else 0,
                ]
            )
            done.append(vid)
        if not journal:
            return []
        did = self._decision_id(c)
        c.executemany("DELETE FROM speaker_assignment WHERE vector_id = ?", drop)
        c.executemany(
            "INSERT OR REPLACE INTO speaker_assignment "
            "(vector_id, person_id, source) VALUES (?,?,?)",
            put,
        )
        c.executemany(self._JOURNAL_SQL, [[did, *row] for row in journal])
        return done

    def _apply_trash(
        self,
        c: sqlite3.Connection,
        ids,
        to_trash: bool,
        reasons: dict[int, str | None] | None = None,
        reason: str | None = None,
    ) -> list[int]:
        """Move voices into or out of the trash inside the open transaction
        and write their history. ``reasons`` per voice wins over ``reason``
        (an undo puts each voice back with the reason it had). Returns the
        ids that changed."""
        ids = list(dict.fromkeys(int(v) for v in ids))
        if ids:
            here = self._still_stored(c, ids)
            ids = [v for v in ids if v in here]
        if not ids:
            return []
        before = self._trash_of(c, ids)
        reasons = reasons or {}
        if to_trash:
            moved = [v for v in ids if v not in before]
            rows = [(v, reasons.get(v, reason)) for v in moved]
            journal = [[v, TRASH, None, None, None, None, 0, None, 1, r, 0] for v, r in rows]
        else:
            moved = [v for v in ids if v in before]
            journal = [[v, TRASH, None, None, None, None, 1, before[v], 0, None, 0] for v in moved]
        if not moved:
            return []
        did = self._decision_id(c)
        if to_trash:
            c.executemany(
                "INSERT INTO speaker_trash (vector_id, reason) "
                "VALUES (?,?) ON CONFLICT(vector_id) DO NOTHING",
                rows,
            )
        else:
            c.executemany("DELETE FROM speaker_trash WHERE vector_id = ?", [(v,) for v in moved])
        c.executemany(self._JOURNAL_SQL, [[did, *row] for row in journal])
        return moved

    def _write(
        self, fn, *, kind: str, origin: str, name: str | None = None, target: int | None = None
    ):
        """Run ``fn(c)`` in one transaction, inside a decision. If the
        transaction fails, an entry it made is gone with it."""
        self.initialize()
        with self.decision(kind, origin, name, target) as d:
            fresh_before = d.id is None
            try:
                with self.connection(immediate=True) as c:
                    return fn(c)
            except Exception:
                if fresh_before:
                    d.id = None
                raise

    def write_names(
        self,
        changes,
        *,
        kind: str = "name",
        origin: str = "",
        name: str | None = None,
        spread: bool = False,
    ) -> list[int]:
        """Set (or remove) names — one entry of the history. Returns the ids
        that changed."""
        changes = [(int(v), (n or "").strip() or None, s) for v, n, s in changes]
        done = self._write(
            lambda c: self._apply_names(c, changes, spread), kind=kind, origin=origin, name=name
        )
        self._changed()
        return done

    def write_trash(
        self,
        ids,
        to_trash: bool,
        *,
        reason: str | None = None,
        kind: str = "trash",
        origin: str = "",
    ) -> list[int]:
        """Move voices into or out of the trash — one entry of the history."""
        done = self._write(
            lambda c: self._apply_trash(c, ids, to_trash, reason=reason), kind=kind, origin=origin
        )
        # The map hangs on this state via `inventory_generation`: without
        # the nudge, its held neighbourhood, runs and columns keep
        # delivering the old picture, in which the discarded voice is
        # still there.
        self._changed(trash=True)
        return done

    def rename_person(self, old: str, new: str, origin: str = "people") -> dict:
        """All voices of one person to another name — one entry.

        With an existing ``new`` this merges the two ("Alice Temp 1" into
        "Alice"). The empty old row is removed; an undo makes it again.
        """
        self.initialize()
        old, new = (old or "").strip(), (new or "").strip()
        if not old or not new:
            raise ValueError("a name is missing")
        with self.connection() as c:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT z.vector_id FROM speaker_assignment z "
                    "JOIN speaker_people p ON p.id = z.person_id "
                    "WHERE p.name = ?",
                    (old,),
                )
            ]
            same_person = c.execute(
                "SELECT COUNT(*) FROM speaker_people WHERE name = ? AND name = ?", (old, new)
            ).fetchone()[0]
        if not ids:
            raise ValueError(f'"{old}" has no voices')
        merged = bool(self.person_id(new, create=False)) and not same_person

        def run(c):
            if same_person:
                # Only the spelling changes ("alice" → "Alice"): the table folds
                # case, so the row itself is renamed; the voices stay.
                c.execute("UPDATE speaker_people SET name = ? WHERE name = ?", (new, old))
                return []
            sources = dict(
                c.execute(
                    "SELECT z.vector_id, z.source FROM speaker_assignment z "
                    "JOIN speaker_people p ON p.id = z.person_id WHERE p.name = ?",
                    (old,),
                ).fetchall()
            )
            done = self._apply_names(c, [(v, new, sources.get(v)) for v in ids])
            left = c.execute(
                "SELECT COUNT(*) FROM speaker_assignment z JOIN speaker_people p "
                "ON p.id = z.person_id WHERE p.name = ?",
                (old,),
            ).fetchone()[0]
            if not left:
                c.execute("DELETE FROM speaker_people WHERE name = ?", (old,))
            return done

        done = self._write(run, kind="rename", origin=origin, name=new)
        self._changed()
        return {
            "voices": len(ids),
            "changed": len(done),
            "merged": merged,
            "spelling_only": bool(same_person),
        }

    # ── Reading the history ───────────────────────────────────────────

    def _finish_decision(self, did: int) -> None:
        """Counts and a short summary for the list — what it was, how much,
        and a few voices to listen to again."""
        with self.connection() as c:
            voices, spread = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(spread), 0) "
                "FROM speaker_decision_voices WHERE decision_id = ?",
                (did,),
            ).fetchone()
            summary = self._summarize(c, [did])[did]
            c.execute(
                "UPDATE speaker_decisions SET voices = ?, spread = ?, summary = ? WHERE id = ?",
                (voices, spread, json.dumps(summary), did),
            )

    @staticmethod
    def _summarize(c: sqlite3.Connection, dids) -> dict[int, dict]:
        """Per entry: speakers (recording + label), recordings, speech, and
        up to eight samples — the voices with the most speech, the ones the
        click itself decided first (not the spread)."""
        out: dict[int, dict] = {
            d: {"speakers": 0, "recordings": 0, "speech_sec": 0.0, "samples": []} for d in dids
        }
        for part in range(0, len(dids), 400):
            chunk = list(dids)[part : part + 400]
            per: dict[int, dict] = {}
            for did, abs_path, label, speech, vid, spread in c.execute(
                "SELECT dv.decision_id, v.abs_path, v.label, v.speech_sec, "
                "       v.id, dv.spread "
                "FROM speaker_decision_voices dv "
                "JOIN speaker_vectors v ON v.id = dv.vector_id "
                f"WHERE dv.decision_id IN ({','.join('?' * len(chunk))})",
                chunk,
            ):
                sp = per.setdefault(did, {}).setdefault((abs_path, label), [0.0, vid, -1.0, 1])
                sp[0] += speech or 0.0
                if (speech or 0.0) > sp[2]:
                    sp[1], sp[2] = vid, speech or 0.0
                if not spread:
                    sp[3] = 0
            for did, speakers in per.items():
                ranked = sorted(speakers.items(), key=lambda kv: (kv[1][3], -kv[1][0]))
                out[did] = {
                    "speakers": len(speakers),
                    "recordings": len({k[0] for k in speakers}),
                    "speech_sec": round(sum(v[0] for v in speakers.values()), 1),
                    "samples": [
                        {
                            "vector_id": v[1],
                            "abs_path": k[0],
                            "label": k[1],
                            "speech_sec": round(v[0], 1),
                            "spread": bool(v[3]),
                        }
                        for k, v in ranked[:8]
                    ],
                }
        return out

    def decisions(
        self, limit: int = 40, before: int | None = None, kind: str = "all", query: str = ""
    ) -> tuple[list[dict], bool]:
        """The history, newest first — without the undo/redo bookkeeping.

        ``kind``: ``all``, ``name`` (namings and renames), ``trash``
        (discarded and restored), ``undone``. ``query`` finds a name.
        """
        self.initialize()
        where = [f"kind NOT IN ({','.join('?' * len(BOOKKEEPING))})"]
        args: list = list(BOOKKEEPING)
        if before:
            where.append("id < ?")
            args.append(int(before))
        if kind == "name":
            where.append("kind IN ('name', 'unname', 'rename')")
        elif kind == "trash":
            where.append("kind IN ('trash', 'restore')")
        elif kind == "undone":
            where.append("undone_by IS NOT NULL")
        if query.strip():
            where.append("name LIKE ?")
            args.append(f"%{query.strip()}%")
        with self.connection() as c:
            rows = [
                dict(r)
                for r in c.execute(
                    "SELECT * FROM speaker_decisions WHERE "
                    + " AND ".join(where)
                    + " ORDER BY id DESC LIMIT ?",
                    (*args, int(limit) + 1),
                )
            ]
        more = len(rows) > limit
        rows = rows[:limit]
        for r in rows:
            r["summary"] = json.loads(r["summary"]) if r.get("summary") else None
        return rows, more

    def decision_row(self, did: int) -> dict | None:
        self.initialize()
        with self.connection() as c:
            r = c.execute("SELECT * FROM speaker_decisions WHERE id = ?", (int(did),)).fetchone()
            if r is None:
                return None
            r = dict(r)
            if not r.get("summary"):
                # An entry whose summary wasn't written (the process ended
                # between the change and the count): make it now.
                r["summary"] = self._summarize(c, [r["id"]])[r["id"]]
            else:
                r["summary"] = json.loads(r["summary"])
        return r

    def decision_voice_ids(self, did: int) -> list[int] | None:
        """The voices an entry changed — ``None`` for an unknown entry."""
        self.initialize()
        with self.connection() as c:
            if not c.execute(
                "SELECT 1 FROM speaker_decisions WHERE id = ?", (int(did),)
            ).fetchone():
                return None
            return [
                r[0]
                for r in c.execute(
                    "SELECT vector_id FROM speaker_decision_voices "
                    "WHERE decision_id = ? ORDER BY vector_id",
                    (int(did),),
                )
            ]

    def history_counts(self) -> dict:
        self.initialize()
        with self.connection() as c:
            r = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(voices), 0), "
                "       SUM(undone_by IS NOT NULL) FROM speaker_decisions "
                f"WHERE kind NOT IN ({','.join('?' * len(BOOKKEEPING))})",
                BOOKKEEPING,
            ).fetchone()
        return {"decisions": r[0], "voices": r[1], "undone": r[2] or 0}

    # ── Undo and redo ─────────────────────────────────────────────────

    def _plan(self, c: sqlite3.Connection, did: int, redo: bool) -> dict:
        """What an undo (or redo) of this entry would change, voice by voice.

        A voice goes back only if (1) this entry is the last effective
        decision that changed it on that thing — no later click named,
        discarded or restored it — and (2) its value is still the one this
        entry left (for a redo: the one it found). The rest is counted by
        reason and left alone.
        """
        rows = c.execute(
            "SELECT vector_id, touched, name_before, source_before, name_after, "
            "       source_after, trash_before, reason_before, trash_after, "
            "       reason_after, spread "
            "FROM speaker_decision_voices WHERE decision_id = ?",
            (did,),
        ).fetchall()
        later: dict[int, int] = {}
        for vid, bits in c.execute(
            "SELECT mine.vector_id, later.touched "
            "FROM speaker_decision_voices mine "
            "JOIN speaker_decision_voices later "
            "  ON later.vector_id = mine.vector_id "
            " AND later.decision_id > mine.decision_id "
            "JOIN speaker_decisions d ON d.id = later.decision_id "
            "WHERE mine.decision_id = ? AND d.undone_by IS NULL "
            f"  AND d.kind NOT IN ({','.join('?' * len(BOOKKEEPING))})",
            (did, *BOOKKEEPING),
        ):
            later[vid] = later.get(vid, 0) | bits
        ids = [r[0] for r in rows]
        names = self._names_of(c, ids)
        trash = self._trash_of(c, ids)
        plan = {
            "names": [],
            "trash_in": {},
            "trash_out": [],
            "direct": [],
            "left": {},
            "rows": len(rows),
        }

        def leave(why):
            plan["left"][why] = plan["left"].get(why, 0) + 1

        for (
            vid,
            bits,
            n_before,
            s_before,
            n_after,
            s_after,
            t_before,
            r_before,
            t_after,
            r_after,
            spread,
        ) in rows:
            moved = False
            if bits & NAME:
                want_now, put = (
                    (n_before, (n_after, s_after)) if redo else (n_after, (n_before, s_before))
                )
                if later.get(vid, 0) & NAME:
                    leave("changed by a later decision")
                elif fold_name((names.get(vid) or (None,))[0]) != fold_name(want_now):
                    leave("name changed since")
                else:
                    plan["names"].append((vid, put[0], put[1] or "history"))
                    moved = True
            if bits & TRASH:
                want_now, put, why = (
                    (t_before, t_after, r_after) if redo else (t_after, t_before, r_before)
                )
                if later.get(vid, 0) & TRASH:
                    leave("changed by a later decision")
                elif int(vid in trash) != int(want_now or 0):
                    leave("trash changed since")
                elif put:
                    plan["trash_in"][vid] = why
                    moved = True
                else:
                    plan["trash_out"].append(vid)
                    moved = True
            if moved and not spread:
                plan["direct"].append(vid)
        return plan

    def undo_preview(self, did: int) -> dict:
        """What an undo would do — nothing is changed."""
        self.initialize()
        row = self.decision_row(did)
        if row is None:
            raise KeyError(did)
        with self.connection() as c:
            plan = self._plan(c, did, redo=bool(row["undone_by"]))
            later = self._later_same_name(c, row)
        moves = len(
            {v for v, _n, _s in plan["names"]} | set(plan["trash_in"]) | set(plan["trash_out"])
        )
        return {
            "entry": row,
            "redo": bool(row["undone_by"]),
            "moves": moves,
            "left": sorted(plan["left"].items(), key=lambda kv: -kv[1]),
            "later_same_name": later,
        }

    @staticmethod
    def _later_same_name(c: sqlite3.Connection, row: dict) -> list[dict]:
        """Later namings of the same person, still in effect. Their spread
        measured against a name centroid this entry was part of — if this
        naming was wrong, they are the ones to look at next."""
        if row["kind"] not in ("name", "rename") or not row.get("name"):
            return []
        return [
            dict(r)
            for r in c.execute(
                "SELECT id, at, voices, spread FROM speaker_decisions "
                "WHERE id > ? AND name = ? COLLATE NOCASE "
                "  AND kind IN ('name', 'rename') AND undone_by IS NULL "
                "ORDER BY id LIMIT 20",
                (row["id"], row["name"]),
            )
        ]

    def undo(self, did: int, origin: str = "history") -> dict:
        """Undo one entry — the voices it changed and nothing else."""
        return self._revert(did, redo=False, origin=origin)

    def redo(self, did: int, origin: str = "history") -> dict:
        """Put an undone entry back in force, as far as nothing changed since."""
        return self._revert(did, redo=True, origin=origin)

    def _revert(self, did: int, redo: bool, origin: str) -> dict:
        self.initialize()
        row = self.decision_row(did)
        if row is None:
            raise KeyError(did)
        if row["kind"] in BOOKKEEPING:
            raise ValueError("an undo is taken back with Redo on its entry")
        if redo and not row["undone_by"]:
            raise ValueError("this entry is in force — nothing to redo")
        if not redo and row["undone_by"]:
            raise ValueError("this entry is already undone")
        result: dict = {}

        def run(c):
            plan = self._plan(c, did, redo)
            named = self._apply_names(c, plan["names"])
            put_in = self._apply_trash(c, list(plan["trash_in"]), True, reasons=plan["trash_in"])
            taken_out = self._apply_trash(c, plan["trash_out"], False)
            changed = set(named) | set(put_in) | set(taken_out)
            # A name the undo took from its last voice goes too: an empty
            # person is a row in the People list with nothing behind it (a
            # rename undone left "Alice 2" standing with no voice). An undo
            # takes away the names the entry gave, a redo the ones it replaced.
            col = "name_before" if redo else "name_after"
            for (n,) in c.execute(
                f"SELECT DISTINCT {col} FROM speaker_decision_voices "
                f"WHERE decision_id = ? AND {col} IS NOT NULL",
                (did,),
            ).fetchall():
                c.execute(
                    "DELETE FROM speaker_people WHERE name = ? AND NOT EXISTS "
                    "(SELECT 1 FROM speaker_assignment z "
                    " WHERE z.person_id = speaker_people.id)",
                    (n,),
                )
            if changed:
                mine = self._local.decision.id
                if redo:
                    c.execute(
                        "UPDATE speaker_decisions SET undone_by = NULL, "
                        "undone_at = NULL WHERE id = ?",
                        (did,),
                    )
                else:
                    c.execute(
                        "UPDATE speaker_decisions SET undone_by = ?, "
                        "undone_at = datetime('now') WHERE id = ?",
                        (mine, did),
                    )
            result.update(
                changed=len(changed),
                rows=plan["rows"],
                left=sorted(plan["left"].items(), key=lambda kv: -kv[1]),
                direct=[v for v in plan["direct"] if v in changed],
                trash_changed=bool(put_in or taken_out),
            )

        self._write(
            run, kind="redo" if redo else "undo", origin=origin, name=row.get("name"), target=did
        )
        self._changed(trash=result.get("trash_changed", False))
        if not result["changed"]:
            raise ValueError(
                "nothing could be "
                + ("redone" if redo else "undone")
                + ": "
                + ", ".join(f"{n} {why}" for why, n in result["left"])
                if result["left"]
                else "nothing to change"
            )
        result["entry"] = self.decision_row(did)
        return result

    def open_voices(self, ids) -> list[int]:
        """The ids among these that carry no name and aren't discarded —
        what can be asked about again."""
        self.initialize()
        ids = [int(v) for v in ids]
        if not ids:
            return []
        with self.connection() as c:
            named = set(self._names_of(c, ids))
            trashed = set(self._trash_of(c, ids))
        return [v for v in ids if v not in named and v not in trashed]

    def speakers_of(self, ids) -> list[dict]:
        """The voices grouped by recording and label, most speech first —
        the shape of a question in the guided assignment."""
        self.initialize()
        ids = [int(v) for v in ids]
        if not ids:
            return []
        per: dict[tuple, dict] = {}
        with self.connection() as c:
            t = self._ids_table(c, ids)
            for vid, abs_path, label, speech in c.execute(
                "SELECT v.id, v.abs_path, v.label, v.speech_sec "
                f"FROM speaker_vectors v JOIN {t} i ON i.id = v.id "
                "ORDER BY v.id"
            ):
                s = per.setdefault(
                    (abs_path, label),
                    {
                        "abs_path": abs_path,
                        "label": label,
                        "speech_sec": 0.0,
                        "vector_ids": [],
                        "best": (-1.0, vid),
                    },
                )
                s["speech_sec"] += speech or 0.0
                s["vector_ids"].append(vid)
                if (speech or 0.0) > s["best"][0]:
                    s["best"] = (speech or 0.0, vid)
        out = sorted(per.values(), key=lambda s: -s["speech_sec"])
        for s in out:
            s["vector_id"] = s.pop("best")[1]
            s["speech_sec"] = round(s["speech_sec"], 1)
        return out


# ── Singleton ────────────────────────────────────────────────────────────

_db: SpeakerDB | None = None
_db_lock = threading.Lock()


def speaker_db() -> SpeakerDB:
    """The database alongside ``library.db``.

    The path is checked against the config on EVERY call: tests and the
    directory switch in the interface put the main database somewhere
    else, and a path captured once would have silently pointed at the
    wrong file.
    """
    global _db
    from whisper_tm.config import config

    target = str(Path(config.db_path).parent / "speaker.db")
    with _db_lock:
        if _db is None or _db._db_path != target:
            if _db is not None:
                try:
                    _db.close()
                except Exception:
                    pass
            _db = SpeakerDB(target)
    return _db

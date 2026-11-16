"""
Path remap — translate stored absolute paths after a move.

``abs_path`` is the identity of a file (docs/PATHS.md). That is right as long
as the media stays where it is, and it is exactly what breaks when the
installation moves to another machine where the same share is mounted under
a different name: ``/mnt/media/a.mp4`` becomes ``/srv/media/a.mp4``, and
every transcript, summary, speaker vector and search chunk points into the
void. Nothing is lost, but nothing is found either.

This module rewrites the old mount point to the new one in every place the
installation stores paths:

* the main database, ``speaker.db``, ``rag.db``, ``media.db`` (Images &
  sounds) and ``audio.db`` (audio analysis) — every text column of every
  table, not a hand-kept list: a
  column added next year is covered without anyone remembering this module;
* ``config.yaml`` — directories, allowed roots, backup target.

Matching is exact and on path boundaries: ``/mnt/nas`` matches ``/mnt/nas``
and ``/mnt/nas/…`` but not ``/mnt/nas2`` or ``/data/mnt/nas``. A path inside
free text (an error message, a JSON column, a ``schema_info`` key) is
translated as well. Several mappings are applied in one pass, longest prefix
first, so ``A → B`` and ``B → C`` never chain into ``A → C``.

Usage (the server must be stopped)::

    whisper-tm --show-roots                          # where do paths point?
    whisper-tm --remap-path /mnt/media /srv/media    # dry run: report only
    whisper-tm --remap-path /mnt/media /srv/media --apply

With ``--apply`` every file is backed up first (``*.before-path-remap-*``),
the databases are changed inside one transaction each, and nothing is
written at all if a translated path would collide with one that already
exists — that happens when the server was started on the new machine before
the remap and indexed the media a second time under the new name.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from whisper_tm.core import paths

__all__ = [
    "Remapper",
    "parse_mappings",
    "remap_database",
    "remap_config_text",
    "server_form",
    "stored_roots",
    "stale_roots",
    "run",
    "show_roots",
]

# A path starts after one of these (or at the start of the text) …
_BEFORE = r"""\s"'(\[{=:,;<>|"""
# … and the old root ends before one of these (or at the end of the text).
_AFTER = r"""/\s"'()\[\]{},;:<>|"""


# ── Mapping ──────────────────────────────────────────────────────────────


def parse_mappings(pairs: Iterable[Iterable[str]]) -> list[tuple[str, str]]:
    """Validate ``(old, new)`` pairs and bring them into canonical form.

    Raises ``ValueError`` with a message meant for the person at the
    terminal: a mapping that is silently corrected would rewrite a hundred
    thousand rows on a guess.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pair in pairs:
        old_raw, new_raw = list(pair)
        old, new = paths.canon(old_raw), paths.canon(new_raw)
        if not old:
            raise ValueError(f"old path is not absolute: {old_raw!r}")
        if not new:
            raise ValueError(f"new path is not absolute: {new_raw!r}")
        if old in ("/", "//") or re.fullmatch(r"[A-Za-z]:/", old):
            raise ValueError(f"refusing to remap the filesystem root {old!r}")
        if old == new:
            raise ValueError(f"old and new path are the same: {old!r}")
        if old in seen:
            raise ValueError(f"old path given twice: {old!r}")
        seen.add(old)
        out.append((old, new))
    if not out:
        raise ValueError("no mapping given")
    return sorted(out, key=lambda m: len(m[0]), reverse=True)


class Remapper:
    """Applies a set of prefix mappings to paths and to text containing paths."""

    def __init__(self, mappings: list[tuple[str, str]]):
        self.mappings = sorted(mappings, key=lambda m: len(m[0]), reverse=True)
        self._new = dict(self.mappings)
        alternatives = "|".join(self._alternative(old, new) for old, new in self.mappings)
        self._rx = re.compile(rf"(?<![^{_BEFORE}])({alternatives})(?=[{_AFTER}]|$)")

    @staticmethod
    def _alternative(old: str, new: str) -> str:
        """Regex for one old root.

        If the new root lies INSIDE the old one (``/mnt`` → ``/mnt/nas``),
        an already translated path still starts with the old root. Without
        the lookahead a second run would turn it into ``/mnt/nas/nas/…``;
        with it, running the remap twice is harmless.
        """
        rx = re.escape(old)
        if new.startswith(old + "/"):
            rx += rf"(?!{re.escape(new[len(old) :])}(?:[{_AFTER}]|$))"
        return rx

    @property
    def olds(self) -> list[str]:
        return [old for old, _ in self.mappings]

    def hits(self, text) -> bool:
        return isinstance(text, str) and self._rx.search(text) is not None

    def __call__(self, text):
        if not isinstance(text, str):
            return text
        return self._rx.sub(lambda m: self._new[m.group(1)], text)

    def path(self, p: str | os.PathLike) -> Path:
        """Translate one filesystem path (for the database locations)."""
        text = str(p)
        canonical = paths.canon(text)
        if canonical and self.hits(canonical):
            return Path(self(canonical))
        return Path(text)


# ── Databases ────────────────────────────────────────────────────────────


@dataclass
class ColumnHit:
    table: str
    column: str
    rows: int


@dataclass
class Conflict:
    table: str
    column: str
    rows: int
    examples: list[str]
    #: rowids of the rows that ALREADY carry the new path (the new server's)
    rowids: list[int] = field(default_factory=list)
    #: moving rows that would land on the SAME new value — no drop fixes that
    merging: int = 0


@dataclass
class DatabaseReport:
    path: Path
    hits: list[ColumnHit] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    dropped: int = 0
    applied: bool = False

    @property
    def rows(self) -> int:
        return sum(h.rows for h in self.hits)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _plain_tables(con: sqlite3.Connection) -> list[str]:
    """Ordinary tables — no virtual tables, no shadow tables.

    The FTS indexes take their content from a base table and are kept in
    sync by triggers; writing into them directly would corrupt them.
    """
    rows = con.execute("PRAGMA table_list").fetchall()
    return [
        r[1] for r in rows if r[0] == "main" and r[2] == "table" and not r[1].startswith("sqlite_")
    ]


def _text_columns(con: sqlite3.Connection, table: str) -> list[str]:
    cols = []
    for _cid, name, decl, *_ in con.execute(f"PRAGMA table_info({_q(table)})"):
        decl = (decl or "").upper()
        # Numeric and BLOB affinity never hold a path; everything else
        # (TEXT, VARCHAR, untyped) may.
        if any(t in decl for t in ("INT", "REAL", "FLOA", "DOUB", "BLOB")):
            continue
        cols.append(name)
    return cols


def _hit_filter(col: str, remapper: Remapper) -> tuple[str, list[str]]:
    """WHERE clause: cheap `instr` prefilter, exact regex check in Python."""
    ors = " OR ".join(f"instr({col}, ?) > 0" for _ in remapper.olds)
    return (f"typeof({col}) = 'text' AND ({ors}) AND wx_hit({col})", list(remapper.olds))


def _unique_indexes(con: sqlite3.Connection, table: str) -> list[list[str]]:
    out = []
    for row in con.execute(f"PRAGMA index_list({_q(table)})"):
        if not row[2]:  # not unique
            continue
        cols = [r[2] for r in con.execute(f"PRAGMA index_info({_q(row[1])})")]
        if cols and None not in cols:  # skip expression indexes
            out.append(cols)
    return out


def _conflicts(con, table, column, remapper) -> Conflict | None:
    """Rows whose translated value already exists in a UNIQUE index.

    Checked before anything is written, so a dry run reports it too.
    """
    rowids: set[int] = set()
    examples: list[str] = []
    merging = 0
    for cols in _unique_indexes(con, table):
        if column not in cols:
            continue
        where, args = _hit_filter("a." + _q(column), remapper)
        same = " AND ".join(
            f"b.{_q(c)} = wx_remap(a.{_q(c)})" if c == column else f"b.{_q(c)} IS a.{_q(c)}"
            for c in cols
        )
        # A row at the target that is itself moving on (A → B together
        # with B → C) frees its place in the same run: no collision, and
        # never a row to drop.
        sql = (
            f"SELECT b.rowid, a.{_q(column)} FROM {_q(table)} AS a "
            f"JOIN {_q(table)} AS b ON {same} AND b.rowid != a.rowid "
            f"WHERE {where} AND NOT wx_hit(b.{_q(column)})"
        )
        for rowid, value in con.execute(sql, args):
            if rowid not in rowids and len(examples) < 3:
                examples.append(f"{value} → {remapper(value)}")
            rowids.add(rowid)
        # Two moving rows with the same target (A → C and B → C): dropping
        # a row of the new server can't solve that — it blocks the run.
        where_m, args_m = _hit_filter(_q(column), remapper)
        group = ", ".join(f"wx_remap({_q(c)})" if c == column else _q(c) for c in cols)
        for value, n in con.execute(
            f"SELECT wx_remap({_q(column)}), COUNT(*) FROM {_q(table)} "
            f"WHERE {where_m} GROUP BY {group} HAVING COUNT(*) > 1",
            args_m,
        ):
            merging += n
            if len(examples) < 3:
                examples.append(f"{n} rows → {value}")
    if not rowids and not merging:
        return None
    return Conflict(table, column, len(rowids) + merging, examples, sorted(rowids), merging=merging)


#: Prefix of the interim value during the two-step update (control
#: characters: no stored path or text starts with them).
_INTERIM = "\x01wx-remap\x02"


def _connect(db_path: Path, remapper: Remapper) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), timeout=5, isolation_level=None)
    # Dropping a colliding row must take its dependants along (a media
    # row its transcripts, a transcript its segments) — exactly what the
    # server's own delete does. Has to be set outside a transaction.
    con.execute("PRAGMA foreign_keys = ON")
    con.create_function("wx_hit", 1, lambda v: 1 if remapper.hits(v) else 0, deterministic=True)
    con.create_function("wx_remap", 1, remapper, deterministic=True)
    return con


def remap_database(
    db_path: str | Path,
    remapper: Remapper,
    *,
    apply: bool = False,
    drop_conflicting: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DatabaseReport:
    """Count — and with ``apply`` rewrite — every stored path in one database.

    The whole database changes in ONE transaction: a crash halfway leaves
    it as it was, never half old, half new. On a conflict nothing is
    written, unless ``drop_conflicting``: then the rows that already carry
    the new path — the ones a server on the new machine created — are
    deleted first, with everything that hangs off them.
    """
    report = DatabaseReport(Path(db_path))
    con = _connect(Path(db_path), remapper)
    try:
        # IMMEDIATE takes the write lock up front: a running server that
        # holds it makes this fail fast instead of halfway through.
        con.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
        for table in _plain_tables(con):
            for column in _text_columns(con, table):
                col = _q(column)
                where, args = _hit_filter(col, remapper)
                n = con.execute(f"SELECT count(*) FROM {_q(table)} WHERE {where}", args).fetchone()[
                    0
                ]
                if not n:
                    continue
                report.hits.append(ColumnHit(table, column, n))
                conflict = _conflicts(con, table, column, remapper)
                if conflict:
                    report.conflicts.append(conflict)
        droppable = all(not c.merging for c in report.conflicts)
        if apply and ((drop_conflicting and droppable) or not report.conflicts):
            for conflict in report.conflicts:
                if progress:
                    progress(f"  dropping {conflict.rows:,} colliding row(s) in {conflict.table}")
                for i in range(0, len(conflict.rowids), 500):
                    chunk = conflict.rowids[i : i + 500]
                    cur = con.execute(
                        f"DELETE FROM {_q(conflict.table)} WHERE rowid IN "
                        f"({','.join('?' * len(chunk))})",
                        chunk,
                    )
                    report.dropped += cur.rowcount
            for hit in report.hits:
                if progress:
                    progress(f"  {hit.table}.{hit.column}: {hit.rows:,} rows")
                col = _q(hit.column)
                where, args = _hit_filter(col, remapper)
                # Two steps: a UNIQUE index is checked row by row during an
                # UPDATE. With A → B and B → C, the row moving to B/x can be
                # written before the one at B/x has left — a collision
                # that exists only halfway through. A marked interim value
                # can't collide with anything.
                con.execute(
                    f"UPDATE {_q(hit.table)} SET {col} = ? || wx_remap({col}) WHERE {where}",
                    [_INTERIM, *args],
                )
                con.execute(
                    f"UPDATE {_q(hit.table)} SET {col} = substr({col}, ?) "
                    f"WHERE substr({col}, 1, ?) = ?",
                    (len(_INTERIM) + 1, len(_INTERIM), _INTERIM),
                )
            con.execute("COMMIT")
            report.applied = True
        else:
            con.execute("ROLLBACK")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return report


# ── config.yaml ──────────────────────────────────────────────────────────


def _remap_tree(node, remapper: Remapper):
    if isinstance(node, dict):
        return {k: _remap_tree(v, remapper) for k, v in node.items()}
    if isinstance(node, list):
        return [_remap_tree(v, remapper) for v in node]
    return remapper(node)


def remap_config_text(text: str, remapper: Remapper) -> tuple[str, int]:
    """Translate paths in the raw YAML text; return ``(text, changed_lines)``.

    Rewriting the text instead of dumping the parsed data keeps comments,
    order and quoting as the person wrote them. Afterwards the result is
    parsed again and must equal the translated original — a new path that
    needs YAML quoting would otherwise change the meaning of the file.
    """
    old_lines = text.splitlines(keepends=True)
    new_lines = [remapper(line) for line in old_lines]
    changed = sum(a != b for a, b in zip(old_lines, new_lines, strict=True))
    new_text = "".join(new_lines)
    if changed:
        expected = _remap_tree(yaml.safe_load(text), remapper)
        if yaml.safe_load(new_text) != expected:
            raise ValueError(
                "config.yaml: the translated file would read differently — "
                "a new path probably needs quotes. Edit the paths in "
                "config.yaml by hand and run the remap again."
            )
    return new_text, changed


# ── Where the installation keeps its files ───────────────────────────────


@dataclass
class Installation:
    config_file: Path | None
    databases: list[Path]
    work_dir: Path | None
    media_dir: Path | None


def locate(config_path: str | Path | None = None, remapper: Remapper | None = None) -> Installation:
    """Find config and databases the way the server would — without side effects.

    ``ConfigManager.load`` creates every configured directory. On the new
    machine those are still the OLD paths before the remap, so directory
    creation is switched off for this instance.
    """
    from whisper_tm.config import ConfigManager

    cfg = ConfigManager()
    cfg._ensure_directories = lambda: None  # type: ignore[method-assign]
    cfg.load(config_path)

    def fix(p) -> Path:
        return remapper.path(p) if remapper else Path(p)

    main_db = fix(cfg.db_path)
    databases = [main_db]
    if (main_db.parent / "speaker.db").exists():
        databases.append(main_db.parent / "speaker.db")
    # Same rule as rag.service.vector_db_path().
    rag = cfg.path_setting("rag", "path")
    databases.append(fix(rag) if rag else main_db.parent / "rag.db")
    # Same rule as media.store.store_path(): Images & sounds keeps every
    # recording's full path
    media = cfg.path_setting("media_search", "path")
    databases.append(fix(media) if media else main_db.parent / "media.db")
    # Same rule as audio.store.store_path(): the audio analysis keeps every
    # recording's full path (its fingerprint segments hold only ids)
    audio = cfg.path_setting("audio_analysis", "path")
    databases.append(fix(audio) if audio else main_db.parent / "audio.db")
    return Installation(
        config_file=cfg.config_path,
        databases=[d for d in databases if d.exists()],
        work_dir=fix(cfg.work_dir),
        media_dir=fix(cfg.media_dir),
    )


def _server_running(work_dir: Path | None) -> str:
    """The same check the server uses before it clears a workdir."""
    if not work_dir:
        return ""
    from whisper_tm.config import ConfigManager

    probe = ConfigManager()
    probe._resolved_work = work_dir  # type: ignore[attr-defined]
    try:
        return probe._workdir_taken_by_other()
    except Exception:
        return ""


def _backup(src: Path, stamp: str) -> Path:
    target = src.with_name(f"{src.stem}.before-path-remap-{stamp}{src.suffix}")
    if src.suffix == ".db":
        with sqlite3.connect(str(src)) as s, sqlite3.connect(str(target)) as t:
            s.backup(t)
    else:
        shutil.copy2(src, target)
    return target


# ── Stored roots (what --show-roots prints, and the startup check) ───────

# (table, column) pairs that name the root a row was created under.
_ROOT_COLUMNS = (("transcripts", "media_dir"), ("queue_items", "base_dir"))


def stored_roots(db_path: str | Path) -> dict[str, int]:
    """Roots the stored rows were created under, with their row counts.

    Nested roots are counted towards the outermost one.

    Opened read-only: this also runs at server start, next to a worker
    that may be writing.
    """
    counts: dict[str, int] = {}
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, timeout=5)
    try:
        tables = set(_plain_tables(con))
        for table, column in _ROOT_COLUMNS:
            if table not in tables:
                continue
            if column not in _text_columns(con, table):
                continue
            for root, n in con.execute(
                f"SELECT {_q(column)}, count(*) FROM {_q(table)} "
                f"WHERE {_q(column)} != '' GROUP BY {_q(column)}"
            ):
                root = paths.canon(root)
                if root:
                    counts[root] = counts.get(root, 0) + n
    finally:
        con.close()
    # A folder that was opened as its own root lies inside the library
    # root; for a move only the outermost one matters.
    outer: dict[str, int] = {}
    for root in sorted(counts, key=len):
        parent = next((o for o in outer if paths.is_inside(root, o)), None)
        outer[parent or root] = outer.get(parent or root, 0) + counts[root]
    return dict(sorted(outer.items(), key=lambda kv: -kv[1]))


def server_form(p: str) -> str:
    """The path as the server will store it.

    The config resolves every configured directory through symlinks
    (``config._expand_path``), so a server whose media directory is
    ``/media`` → ``/mnt/nas`` stores ``/mnt/nas/…``. Translating to the
    symlink would give rows the server never looks up — transcripts still
    open through the relative fallback, speaker data does not.
    """
    if os.path.isdir(p):
        return paths.canon(os.path.realpath(p)) or p
    return p


def stale_roots(db_path: str | Path) -> dict[str, tuple[int, str]]:
    """Stored roots the server on this machine will not find.

    ``{root: (rows, target)}`` — ``target`` is empty if the root does not
    exist, or the real path if the root is a symlink.
    """
    out: dict[str, tuple[int, str]] = {}
    for root, n in stored_roots(db_path).items():
        if not os.path.isdir(root):
            out[root] = (n, "")
        elif (real := server_form(root)) != root:
            out[root] = (n, real)
    return out


# ── Command line ─────────────────────────────────────────────────────────


def show_roots(config_path: str | Path | None = None, out: Callable[[str], None] = print) -> int:
    inst = locate(config_path)
    if not inst.databases:
        out("No database found — nothing to show.")
        return 1
    main_db = inst.databases[0]
    out(f"Database: {main_db}")
    out(f"Media directory in config: {inst.media_dir}")
    out("")
    roots = stored_roots(main_db)
    if not roots:
        out("No stored roots.")
        return 0
    stale = stale_roots(main_db)
    for root, n in roots.items():
        target = stale.get(root, (0, None))[1]
        state = "ok" if target is None else ("SYMLINK" if target else "MISSING")
        suffix = f"  → {target}" if target else ""
        out(f"  {state:8} {n:>9,}  {root}{suffix}")
    if stale:
        root, (_n, target) = next(iter(stale.items()))
        out("")
        out(
            "Rows under a MISSING or SYMLINK root are not found by the server "
            "here. If the media moved, translate the paths, e.g.:"
        )
        out(f"  whisper-tm --remap-path {root} {target or inst.media_dir}")
    return 0


def run(
    pairs: Iterable[Iterable[str]],
    *,
    apply: bool = False,
    backup: bool = True,
    drop_conflicting: bool = False,
    config_path: str | Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Dry run or apply. Return code for the shell: 0 ok, 1 refused."""
    try:
        mappings = parse_mappings(pairs)
        resolved = [(old, server_form(new)) for old, new in mappings]
        for (_, new), (_, real) in zip(mappings, resolved, strict=True):
            if real != new:
                out(
                    f"Note: {new} is a symlink; the server stores the real path, so {real} is used."
                )
        remapper = Remapper(parse_mappings(resolved))
    except ValueError as e:
        out(f"Error: {e}")
        return 1

    inst = locate(config_path, remapper)
    for old, new in remapper.mappings:
        out(f"Mapping: {old}  →  {new}")
        if not os.path.isdir(new):
            out(f"  Warning: {new} does not exist on this machine.")
    out("")

    config_text = config_new = None
    config_changed = 0
    if inst.config_file:
        config_text = inst.config_file.read_text(encoding="utf-8")
        try:
            config_new, config_changed = remap_config_text(config_text, remapper)
        except ValueError as e:
            out(f"Error: {e}")
            return 1
        out(f"Config {inst.config_file}: {config_changed} line(s)")
    else:
        out("Config: none found (defaults) — nothing to translate there.")

    if not inst.databases:
        out("No database found.")
        return 1

    reports = [remap_database(db, remapper) for db in inst.databases]
    for rep in reports:
        out(f"Database {rep.path}: {rep.rows:,} value(s)")
        for h in rep.hits:
            out(f"  {h.table + '.' + h.column:<40} {h.rows:>10,}")
    conflicts = [(rep.path, c) for rep in reports for c in rep.conflicts]
    for db, c in conflicts:
        out("")
        out(
            f"CONFLICT in {db.name}: {c.table}.{c.column} — {c.rows:,} "
            f"translated path(s) already exist, e.g."
        )
        for ex in c.examples:
            out(f"    {ex}")
    if any(c.merging for _db, c in conflicts):
        out("")
        out(
            "Nothing was changed. Several old paths would land on the SAME "
            "new one (two mappings onto one target) — dropping rows can't "
            "solve that. Check the mappings."
        )
        return 1
    if conflicts and not drop_conflicting:
        out("")
        out(
            "Nothing was changed. The server has most likely already run on "
            "this machine and indexed the media under the new path. Either "
            "copy the databases from the old machine again and run the remap "
            "BEFORE the first start, or add --drop-conflicting: the rows the "
            "new server created for these files are deleted and the moved "
            "ones take their place. Transcripts kept as JSON next to the "
            "media are read in again when their folder is opened."
        )
        return 1
    if conflicts:
        out("")
        out(
            "--drop-conflicting: the rows listed above that already carry the "
            "new path will be deleted, then the moved rows take their place."
        )

    total = sum(r.rows for r in reports) + config_changed
    if not total:
        out("")
        out("Nothing to translate.")
        return 0
    if not apply:
        out("")
        out("Dry run — nothing was changed. Add --apply to write.")
        return 0

    busy = _server_running(inst.work_dir)
    if busy:
        out(f"Error: a server is running in this installation ({busy}). Stop it first.")
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if backup:
        out("")
        for rep in reports:
            if rep.rows:
                out(f"Backup: {_backup(rep.path, stamp)}")
        if config_changed and inst.config_file:
            out(f"Backup: {_backup(inst.config_file, stamp)}")

    out("")
    for rep in reports:
        if not rep.rows:
            continue
        out(f"Writing {rep.path.name} …")
        try:
            done = remap_database(
                rep.path, remapper, apply=True, drop_conflicting=drop_conflicting, progress=out
            )
        except sqlite3.OperationalError as e:
            out(f"Error: {rep.path.name}: {e} — is the server still running?")
            return 1
        except sqlite3.IntegrityError as e:
            # Two OLD paths that land on the same new one (two mappings
            # onto one target) — nothing in this database was written.
            out(f"Error: {rep.path.name}: {e}. Nothing was written to it.")
            return 1
        if done.dropped:
            out(f"  {done.dropped:,} colliding row(s) dropped")
        if not done.applied:
            out(
                f"Error: {rep.path.name} changed in the meantime; nothing "
                "written to it. Run the remap again."
            )
            return 1
    if config_changed and inst.config_file and config_new is not None:
        tmp = inst.config_file.with_name(inst.config_file.name + ".tmp")
        tmp.write_text(config_new, encoding="utf-8")
        os.replace(tmp, inst.config_file)
        out(f"Config written: {inst.config_file}")
    out("")
    out("Done.")
    return 0

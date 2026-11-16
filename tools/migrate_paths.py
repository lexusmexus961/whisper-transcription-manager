#!/usr/bin/env python3
"""Normalize paths in the database to the current media directory.

BACKGROUND
----------
`media_files.rel_path` and `transcripts.rel_path` are stored relative to the
media directory that was configured when the row was created. If the media
directory is changed later, older rows still point to the old root. Effects in
production:

  * Clicking a search result returned "no transcript found", while the video
    played normally — resolving goes through the disk, transcript lookup through
    the DB.
  * The same file was indexed or transcribed once under two roots: duplicate
    rows pointing to the same file.

This script makes the absolute path the source of truth, writes all rel_path
relative to the current media directory, and consolidates duplicates.

SAFETY
------
* Without ``--apply`` nothing happens — only reporting (dry run).
* With ``--apply`` it first backs up via the SQLite backup API.
* The app must not be running.
* The script is repeatable: a second run finds nothing to do.

ORDER (important)
-----------------
transcripts.media_id is an FK to media_files.id with ON DELETE CASCADE.
Deleting duplicate media_files rows would tear out their transcripts, so the
transcripts are FIRST reassigned to the row being kept.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
import time
from pathlib import Path

# When run as a file, sys.path[0] is the tools/ folder, NOT the repo.
# `import whisper_tm` would then reach a different copy of the package
# installed in the venv — which can deliver a wrong media directory, whereupon
# this script would mislabel half the database. The repo belongs at the
# front of the search path.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Priority order when merging duplicate file entries: an already-processed
# entry is worth more than an untouched one.
STATUS_RANK = {"done": 4, "skipped": 3, "error": 2, "pending": 1}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def back_up(db_path: Path) -> Path:
    """Back up via the SQLite backup API, not by copying the file.

    Copying a WAL database otherwise produces an incomplete snapshot.
    """
    target = db_path.with_name(f"{db_path.stem}_pre-migration_{time.strftime('%Y%m%d_%H%M%S')}.db")
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def norm(root: str, media_dir: str | None, rel: str) -> str | None:
    """rel_path relative to the current root — or None if outside it.

    Deliberately uses core/paths (docs/PATHS.md) instead of ad-hoc path
    arithmetic: ``startswith(root + os.sep)`` classified ``/a/bc`` as being
    under ``/a/b``, and ``os.path.relpath`` returns backslashes on Windows
    that no scanner lookup can find again.
    """
    from whisper_tm.core import paths

    return paths.rel_from(paths.abs_from(rel, media_dir or root), root)


def write_marker(conn, root: str, report: dict) -> None:
    """Record THAT and HOW the migration was run.

    Without a marker, there's no way to tell later whether this tool ever
    ran — exactly that question comes up again every time crooked paths
    are suspected.
    """
    import json as _json

    conn.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        (
            "migrate_paths",
            _json.dumps(
                {"timestamp": int(time.time()), "root": root, "report": report}, ensure_ascii=False
            ),
        ),
    )


def two_phase_update(conn, table: str, pairs: list[tuple[int, str]]) -> None:
    """Set rel_path without violating the UNIQUE constraint along the way.

    Historically necessary while rel_path had to be unique. Since schema v8
    it no longer is — the intermediate step remains anyway, because this
    tool also has to be able to run on databases that haven't been migrated
    yet.
    """
    conn.executemany(
        f"UPDATE {table} SET rel_path = '\x01' || id WHERE id = ?", [(i,) for i, _ in pairs]
    )
    conn.executemany(
        f"UPDATE {table} SET rel_path = ? WHERE id = ?", [(new_value, i) for i, new_value in pairs]
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", help="Path to the database (default: from config)")
    p.add_argument("--root", help="Current media directory (default: from config)")
    p.add_argument(
        "--apply", action="store_true", help="Actually write changes (otherwise dry run only)"
    )
    args = p.parse_args()

    db_path = Path(args.db) if args.db else None
    root = args.root
    if db_path is None or root is None:
        import whisper_tm
        from whisper_tm.config import config

        # Report which package copy was actually loaded — a stale copy
        # delivers the wrong root.
        log(f"Package           : {whisper_tm.__file__}")
        config.load()
        log(f"Config            : {config.config_path}")
        db_path = db_path or Path(config.db_path)
        root = root or str(config.media_dir)
    from whisper_tm.core import paths as _paths

    root = _paths.canon(root)

    log(f"Database          : {db_path}")
    log(f"Media directory   : {root}")
    log(f"Mode              : {'APPLY' if args.apply else 'Dry run (nothing will be written)'}")
    log()

    if not db_path.exists():
        log(f"ERROR: {db_path} does not exist.")
        return 1

    if args.apply:
        target = back_up(db_path)
        log(f"Backup created: {target} ({target.stat().st_size / 1e6:.0f} MB)")
        log()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    # Fallback root for file rows without abs_path: the media_dir of their
    # transcript. Without this, these rows would stay on the old rel_path
    # while their transcript gets normalized — the two would no longer find
    # each other afterwards. (Affects entries for files that have since been
    # deleted.)
    root_from_transcript = {
        r["rel_path"]: r["media_dir"]
        for r in conn.execute("SELECT rel_path, media_dir FROM transcripts")
        if r["media_dir"]
    }

    # ── media_files ───────────────────────────────────────────────────
    log("── media_files ──────────────────────────────────────────")
    groups: dict[str, list[sqlite3.Row]] = collections.defaultdict(list)
    without_abs = 0
    across_transcript = 0
    old_to_new: dict[str, str] = {}
    for r in conn.execute("SELECT id, rel_path, abs_path, status, updated_at FROM media_files"):
        # Via core/paths, not startswith/relpath: the former classified
        # /a/bc as being under /a/b, the latter returns backslashes on
        # Windows that no scanner lookup can find again (docs/PATHS.md).
        new_value = _paths.rel_from(r["abs_path"] or "", root)
        if new_value is None:
            replacement = root_from_transcript.get(r["rel_path"])
            new_value = norm(root, replacement, r["rel_path"]) if replacement else None
            if new_value is None:
                without_abs += 1
                continue
            across_transcript += 1
        groups[new_value].append(r)
        old_to_new[r["rel_path"]] = new_value

    mf_rename: list[tuple[int, str]] = []
    mf_delete: list[int] = []
    reparent: list[tuple[int, int]] = []
    for new_value, lines in groups.items():
        # Row to keep: furthest-along status, then most recent, then lowest id
        lines = sorted(
            lines,
            key=lambda x: (-STATUS_RANK.get(x["status"], 0), -(x["updated_at"] or 0), x["id"]),
        )
        keep, remainder = lines[0], lines[1:]
        if keep["rel_path"] != new_value:
            mf_rename.append((keep["id"], new_value))
        for z in remainder:
            mf_delete.append(z["id"])
            reparent.append((keep["id"], z["id"]))

    log(f"  Rows total                 : {sum(len(v) for v in groups.values()) + without_abs}")
    log(f"  no abs_path, matched via transcript: {across_transcript}")
    log(f"  no usable abs_path         : {without_abs} (left unchanged)")
    log(f"  rel_path to correct        : {len(mf_rename)}")
    log(f"  duplicates to merge        : {len(mf_delete)}")
    log()

    # ── transcripts ───────────────────────────────────────────────────
    log("── transcripts ──────────────────────────────────────────")
    t_groups: dict[str, list[sqlite3.Row]] = collections.defaultdict(list)
    t_outside = 0
    for r in conn.execute(
        "SELECT id, rel_path, media_dir, word_count, segment_count, updated_at FROM transcripts"
    ):
        new_value = norm(root, r["media_dir"], r["rel_path"])
        if new_value is None:
            t_outside += 1
            continue
        t_groups[new_value].append(r)

    t_rename: list[tuple[int, str]] = []
    t_delete: list[int] = []
    for new_value, lines in t_groups.items():
        # Row to keep: the richest-content match. In most duplicate groups
        # this agrees with the most recent one; where it differs, more
        # content is the better criterion than a newer timestamp.
        lines = sorted(
            lines,
            key=lambda x: (
                -(x["word_count"] or 0),
                -(x["segment_count"] or 0),
                -(x["updated_at"] or 0),
                x["id"],
            ),
        )
        keep, remainder = lines[0], lines[1:]
        if keep["rel_path"] != new_value:
            t_rename.append((keep["id"], new_value))
        t_delete.extend(z["id"] for z in remainder)

    log(f"  Rows total                 : {sum(len(v) for v in t_groups.values()) + t_outside}")
    log(f"  outside the root           : {t_outside} (left unchanged)")
    log(f"  rel_path to correct        : {len(t_rename)}")
    log(f"  duplicates to remove       : {len(t_delete)}")
    log()

    # Plausibility gate. If practically nothing falls under the assumed
    # root, the root is wrong — aborting is then the correct move, not
    # continuing. Without this check, a wrong media directory would run
    # through and relabel every row.
    total_t = sum(len(v) for v in t_groups.values()) + t_outside
    if total_t and t_outside / total_t > 0.5:
        log(f"ABORT: {t_outside} of {total_t} transcripts are not under {root}.")
        log("The media directory does not match this database.")
        log("Specify the correct root with --root.")
        return 2

    # ── skiplist / summaries ──────────────────────────────────────────
    log("── skiplist / summaries ─────────────────────────────────")
    sk_remap: list[tuple[str, str]] = []
    sk_delete: list[str] = []
    present = {r["rel_path"] for r in conn.execute("SELECT rel_path FROM skiplist")}
    targets: set[str] = set()
    for old_value in sorted(present):
        new_value = old_to_new.get(old_value)
        if new_value is None or new_value == old_value:
            continue
        if new_value in present or new_value in targets:
            sk_delete.append(old_value)  # target already exists → duplicate
        else:
            sk_remap.append((old_value, new_value))
            targets.add(new_value)
    remap_sums = [
        (r["rel_path"], old_to_new[r["rel_path"]])
        for r in conn.execute("SELECT DISTINCT rel_path FROM summaries")
        if r["rel_path"] in old_to_new and old_to_new[r["rel_path"]] != r["rel_path"]
    ]
    log(f"  skiplist  to rewrite       : {len(sk_remap)} (+{len(sk_delete)} duplicates)")
    log(f"  summaries to rewrite       : {len(remap_sums)}")
    log()

    if not args.apply:
        log("Dry run finished — nothing was changed.")
        log("Run again with  --apply  to actually execute.")
        return 0

    # ── Apply ──────────────────────────────────────────────────────────
    log("── Writing changes ──────────────────────────────────────")
    t0 = time.time()
    try:
        conn.execute("BEGIN")
        # 1. Reassign transcripts to the row being kept, BEFORE the
        #    duplicate media_files rows are deleted (ON DELETE CASCADE!).
        conn.executemany("UPDATE transcripts SET media_id = ? WHERE media_id = ?", reparent)
        log(f"  Transcripts reassigned     : {conn.total_changes}")

        before = conn.total_changes
        conn.executemany("DELETE FROM media_files WHERE id = ?", [(i,) for i in mf_delete])
        log(f"  media_files duplicates gone: {conn.total_changes - before}")

        two_phase_update(conn, "media_files", mf_rename)
        log(f"  media_files normalized     : {len(mf_rename)}")

        # 2. Transcript duplicates. Segments are attached via FK with
        #    CASCADE, the FTS triggers clean up the search index along with it.
        before = conn.total_changes
        conn.executemany("DELETE FROM transcripts WHERE id = ?", [(i,) for i in t_delete])
        log(
            f"  Transcript duplicates gone : {len(t_delete)} "
            f"(incl. segments: {conn.total_changes - before} rows)"
        )

        two_phase_update(conn, "transcripts", t_rename)
        # Set media_dir ONLY for rows whose rel_path is now actually
        # relative to this root. An unconditional UPDATE over the whole
        # table would otherwise claim a root that the paths don't match.
        normalized_ids = [i for group_nr in t_groups.values() for i in [z["id"] for z in group_nr]]
        conn.executemany(
            "UPDATE transcripts SET media_dir = ? WHERE id = ?", [(root, i) for i in normalized_ids]
        )
        log(
            f"  transcripts normalized     : {len(t_rename)} (media_dir set: {len(normalized_ids)})"
        )

        # 3. Apply skiplist / summaries
        conn.executemany("DELETE FROM skiplist WHERE rel_path = ?", [(a,) for a in sk_delete])
        conn.executemany(
            "UPDATE skiplist SET rel_path = ? WHERE rel_path = ?", [(n, a) for a, n in sk_remap]
        )
        conn.executemany(
            "UPDATE summaries SET rel_path = ? WHERE rel_path = ?", [(n, a) for a, n in remap_sums]
        )
        log(f"  skiplist/summaries         : {len(sk_remap)}/{len(remap_sums)}")

        conn.execute("COMMIT")
    except Exception as e:
        conn.execute("ROLLBACK")
        log(f"\nERROR — nothing changed (rollback): {e}")
        raise

    log(f"  Duration: {time.time() - t0:.1f}s")
    log()

    # ── Search index + verification ──────────────────────────────────
    log("── Post-processing ──────────────────────────────────────")
    conn.execute("INSERT INTO segments_fts(segments_fts) VALUES('rebuild')")
    write_marker(
        conn,
        root,
        {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("media_files", "transcripts", "skiplist", "summaries")
        },
    )
    conn.commit()
    log("  FTS index rebuilt")
    log("  Marker written to schema_info (key 'migrate_paths')")
    log(f"  integrity_check: {conn.execute('PRAGMA integrity_check').fetchone()[0]}")
    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    log(f"  foreign_key_check: {'ok' if not fk else f'{len(fk)} violations!'}")

    remainder = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE media_dir != ?", (root,)
    ).fetchone()[0]
    log(f"  Transcripts with foreign root: {remainder}")
    for t in ("media_files", "transcripts", "segments", "skiplist"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        log(f"  {t:<12}: {n}")
    conn.close()
    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

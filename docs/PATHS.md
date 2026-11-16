# Paths — binding model

This document is the **only** source of truth for how paths are stored,
compared, and composed in this project. Anyone writing code that identifies
a file follows this — there are no exceptions.

---

## The decision: the absolute path is the identity

**`abs_path` identifies a file. Always. Everywhere.**

The reason lies in a property of the application that is easy to overlook:

```python
# api/main.py, set_directory (changing directory)
config.media_dir = path
# api/main.py, jumping to a file from search
config.media_dir = target_dir
# api/settings.py, saving the folder settings
config.media_dir = path
```

`config.media_dir` **changes at runtime** as soon as someone switches to a
different folder in the frontend. A relative path is therefore not a stable
identifier: the same file gets a different `rel_path` depending on which
directory was active during the scan.

```
Scan with media_dir=/nas/video       →  rel_path = "series/episode1.mp4"
Scan with media_dir=/nas/video/series →  rel_path = "episode1.mp4"
                                          ^ same file, two identities
```

This was exactly the cause of the duplicate entries, the 404s when opening
transcripts, and the repair runs. `abs_path` and `realign_paths()` were a
patch on top of it. With this decision, the patch becomes the foundation.

### Division of roles

| Field | Role | Unique? |
|---|---|---|
| `abs_path` | **Identity.** Absolute, normalized, `/` as separator. | yes — partial UNIQUE index `WHERE abs_path != ''` |
| `rel_path` | **Display.** Relative to the root currently being viewed. | no |
| `media_dir` | **Origin.** Which root was active at write time. Diagnostics. | no |

`rel_path` is a *derived* value. It is allowed to change when the root
changes — that's not a bug, that's its job. It is never used for comparing
or linking.

The uniqueness is deliberately **partial**: rows for which no absolute
path could be derived keep an empty field and are retained.

Two indexes sit on `abs_path`, and both are needed: the partial one
enforces uniqueness, a second, non-partial one serves the lookups —
SQLite won't use a partial index for a plain `WHERE abs_path = ?`, because
at planning time it can't know the parameter isn't empty.

---

## The normal form

An `abs_path` in the database is **always**:

1. absolute (`/nas/video/a.mp4`, never `video/a.mp4`, never `~/video/a.mp4`)
2. normalized — no `.`, no `..`, no duplicate separators
3. using `/` as separator, even on Windows (`C:/video/a.mp4`)
4. without a trailing slash
5. **not** resolved through the filesystem — symlinks remain as-is

   One exception: the configured directories themselves. The config
   resolves `media`, `work` and the others once at load time
   (`config._expand_path`), so a media directory `/media` → `/mnt/nas`
   yields paths under `/mnt/nas`. Symlinks *inside* the library stay as
   they are.

Point 5 is intentional: `Path.resolve()` would query the filesystem. On a
NAS with hundreds of thousands of files that's a network round-trip per path, and with a
share that's occasionally unmounted, the result could differ from
yesterday's. Normalization is pure character arithmetic and therefore
predictable.

---

## The single API: `whisper_tm/core/paths.py`

Paths are **nowhere else** assembled or normalized.

```python
from whisper_tm.core import paths

paths.canon(p)              # any path  → canonical abs_path (str)
paths.rel_from(abs, root)   # abs_path + root → rel_path (str) | None if outside
paths.abs_from(rel, root)   # rel_path + root → canonical abs_path (str)
paths.is_inside(abs, root)  # is abs located under root?
paths.like_prefix(root)     # LIKE prefix with escaped wildcards (ESCAPE '\')
```

Forbidden, because this is exactly what produced the path bugs of the past:

```python
str(media_dir) + "/" + rel_path        # duplicate separators, no normpath
os.path.join(media_dir, rel_path)      # same, plus OS-native separator
abs_path[len(prefix):]                 # character arithmetic without checks
str(fp)                                # OS-native; backslashes on Windows
Path(x).name                           # basename as a stopgap → name collision
":" in p[:3]                           # "10:30 Meeting.mp4" is not a drive letter
"WHERE abs_path LIKE ? || '/%'"        # % and _ in the path act as wildcards
WHERE rel_path = ?                     # not unique
```

The last point is the one that went unnoticed the longest: `rel_path` lost
its uniqueness, but the queries relying on it stayed unchanged.
`delete_transcript` thus also deleted the summaries of a like-named file
from a different root.

---

## Lookup

There is **one** resolver, `Database.resolve_transcript()`. It searches
in this order:

1. `abs_path = ?` — the normal case. If the caller doesn't know a root,
   the currently configured one is used.
2. `rel_path = ?` — last resort for legacy rows without `abs_path` and for
   callers that don't know the root. Matches are counted; if **multiple**
   rows match, the request is ambiguous and this is logged, instead of
   silently taking the first one.

There is no longer a basename search. It was a stopgap for broken
`rel_path` values and invites a name collision: `a/intro.mp4` and
`b/intro.mp4` are two files, not one.

Callers should **always** pass the root along. It's not a mandatory
parameter — that would have broken every existing call without making
resolution any better — but without it you end up at stage 2, which can
be ambiguous.

---

## Writing

Every write path computes `abs_path` from what it reliably knows — the
actual location of the file on disk — and **not** from a reconstruction
made from `rel_path` plus a guessed root.

Two cases where exactly that went wrong:

* Nested `.transcripts/` folders: the reconstruction produced
  `/media/sub/sub/b.mp3`, overwrote the correct value, and propagated
  itself into `rel_path` via `realign_paths()`.
* The transcription worker: `filename` is relative to the directory the
  job was **enqueued** with. The queue is persistent — by the time the
  job runs, `config.media_dir` may long since point elsewhere. The worker
  therefore passes `abs_path=source_path`, the actual location.

## Lookup and cleanup

It's not just lookups that go through identity, but **every** query that
refers to a specific file: deletion, status changes, summaries, the
skiplist, linking a transcript to its media file, the scan join.

A `WHERE rel_path = ?` is a search for *some* file with that name. Where
it's unavoidable (rows without `abs_path`), it's restricted to
`abs_path = ''` and logged.

---

## The scan folder is allowed to differ from the root

The scanner doesn't necessarily run over the root the transcripts were
made under: browsing in the frontend changes `config.media_dir`, and a scan
runs over the folder that is open. The `rel_path` formed during that scan
then refers to the scan folder, while a later lookup might refer to a
different directory.

This used to be a contradiction that only the stopgaps papered over.
Under this model it's **no longer one**: `abs_path` is formed at write
time from the actual location and is independent of either root.
`rel_path` may freely diverge — it's for display, and `realign_paths()`
realigns it to the current root at startup.

## Why not relative paths as identity?

That would be the usual choice for a library that moves between machines
(NAS mount here `/mnt/nas`, in the container `/data`). It fails here
because of the mutating root: as long as the frontend can switch
`config.media_dir`, there is no single reference point against which a
relative path would stay stable.

The move case is handled explicitly instead: `./start.sh --remap-path
OLD NEW` (`database/path_remap.py`) rewrites the root in every stored path
— the main database, `speaker.db`, `rag.db` and `media.db`, every text
column, and `config.yaml` — in one
transaction per database. At startup the server compares the stored roots
with the filesystem and shows a banner if they are missing, so a move that
forgot the remap is visible instead of just looking like an empty library.
`realign_paths()` then re-derives `rel_path` as usual.

Should the mutating root ever disappear (a fixed `media_dir`, browsing
only within it), switching to relative identity would be a sensible
simplification. Until then, this document applies.

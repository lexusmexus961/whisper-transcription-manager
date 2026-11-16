"""
Paths — the only place where paths are normalized and composed.

The authoritative model is in docs/PATHS.md. Summary:

    abs_path is the identity of a file. Always. Everywhere.

The reason is an application property: ``config.media_dir`` changes at runtime
when someone switches the directory in the frontend. A relative path is thus
not a stable identifier — the same file would get a different ``rel_path``
depending on the active root. That's exactly where the duplicate entries and
404s when opening transcripts came from.

``rel_path`` remains as a *derived* display value and is never used for comparison.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
import unicodedata
from functools import lru_cache

__all__ = ["canon", "rel_from", "abs_from", "is_absolute", "is_inside", "like_prefix"]


def _is_windows_style(text: str) -> bool:
    """Does the path look like Windows (drive letter or UNC)?

    Needed so a path from a Windows installation is correctly normalized in DB
    even under Linux (tests, migration, database file copied from Windows).
    """
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return True
    return text.startswith("\\\\") or text.startswith("//?/")


def is_absolute(p: str | os.PathLike | None) -> bool:
    """Is the path absolute — POSIX, Windows drive, or UNC?

    Deliberately stricter than just ``":" in p[:3]``: on Linux, a colon in
    a filename is allowed and common in a media library. ``10:30 Meeting.mp4``
    would thus be seen as an absolute Windows path, and lookup by identity
    would silently fail.

    A drive letter only counts if a letter AND directly after it a separator
    follows (``C:/…``); ``C:foo`` is drive-relative and thus not absolute.
    """
    if p is None:
        return False
    text = str(p)
    if not text:
        return False
    if text[0] in "/\\":
        return True
    return len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] in "/\\"


@lru_cache(maxsize=8192)
def _canon_str(text: str) -> str:
    """Core of :func:`canon` for strings — buffered.

    In the file list, canon() is called twice per line, including once on
    the always-same root. With 200,000 files, that's 200,000 identical
    normalizations. The function is pure, so buffering is safe.
    """
    return _canon_raw(text)


def canon(p: str | os.PathLike | None) -> str:
    """Convert any path to canonical form.

    Canonical means: absolute, normalized (no ``.``/``..``, no duplicate
    separators), ``/`` as separator, no trailing slash.

    Deliberately **without** ``Path.resolve()``: that would query the filesystem.
    With 200,000 files on a NAS, that's a network roundtrip per path, and with
    an unmounted share, you'd get a different result than yesterday. Symlinks
    thus remain in place — normalization is pure string arithmetic and thus
    predictable.

    Returns **""** if input is empty OR not absolute. A relative path has no
    identity; silently making it into a plausible-looking value here would be
    guessing — and that value would end up as a supposed identity in the database.
    Whoever has a root uses :func:`abs_from`.
    """
    if p is None:
        return ""
    text = p if isinstance(p, str) else str(p)
    return _canon_str(text)


def _canon_raw(text: str) -> str:
    if not text.strip():
        return ""
    # Strip surrounding whitespace in BOTH branches. Previously only
    # the Windows branch did it: "/nas/video  " kept its spaces,
    # "C:\video  " didn't — same typo gave different abs_path by platform.
    # Only outside: spaces IN paths are valid and left untouched.
    text = text.strip()
    if not is_absolute(text):
        return ""

    if _is_windows_style(text):
        # Windows path: normalize with ntpath, then switch to /.
        out = ntpath.normpath(text).replace("\\", "/")
        # normpath turns "C:/" into "C:\" → after swap "C:/". Root slash must
        # stay, else "C:/" becomes "C:".
        if len(out) == 3 and out[1:] == ":/":
            return out
        return out.rstrip("/")

    # NO backslash replacement here: under POSIX, "\" is a valid filename
    # character. "/mnt/a\b.mp4" is ONE file, not two levels.
    out = posixpath.normpath(text)
    # The leading double slash REMAINS.
    #
    # Previously it was stripped here — with the reasoning that //host/x and
    # /host/x are the same file. The branch above leaves it in place though
    # (ntpath turns \\server\share into exactly //server/share), so the same
    # file had TWO identities depending on how it came in:
    #
    #     canon(r"\\server\share\a.mp4")  ->  //server/share/a.mp4
    #     canon("//server/share/a.mp4")   ->   /server/share/a.mp4
    #
    # Under the partial UNIQUE index, that's two rows for one file — exactly
    # the state this module should prevent.
    #
    # Of the two possible alignments, this one is lossless: a real UNC path is
    # broken without its double slash and can't be restored, while an accidental
    # double slash in a POSIX path is just an unusual spelling of the same file.
    # POSIX explicitly allows exactly two leading slashes as implementation-
    # defined; three or more are folded together by normpath anyway.
    return out if out in ("/", "//") else out.rstrip("/")


def _segments(canonical: str) -> list[str]:
    return [s for s in canonical.split("/") if s]


def is_inside(abs_path: str | os.PathLike, root: str | os.PathLike) -> bool:
    """Is ``abs_path`` below ``root`` (or is it the root itself)?

    Compares on segment boundary. A mere ``startswith`` would incorrectly
    classify ``/a/bc`` as below ``/a/b``.
    """
    a, r = canon(abs_path), canon(root)
    if not a or not r:
        return False
    if a == r:
        return True
    sa, sr = _segments(a), _segments(r)
    return len(sa) > len(sr) and sa[: len(sr)] == sr


def rel_from(abs_path: str | os.PathLike, root: str | os.PathLike) -> str | None:
    """Convert canonical ``abs_path`` to a ``rel_path`` below ``root``.

    Returns ``None`` if path is **not** below the root.
    Deliberately not ``os.path.relpath``: that would give a ``../..`` result,
    which doesn't belong in the database as rel_path (and on Windows also returns
    backslashes that scanner lookup won't find).
    """
    # Deliberately NOT via is_inside(): that would normalize both sides a second
    # time. In the file list, this function runs once per line — with 200,000 files,
    # four canon() calls instead of two is worth skipping.
    a, r = canon(abs_path), canon(root)
    if not a or not r:
        return None
    if a == r:
        return ""
    sa, sr = _segments(a), _segments(r)
    if len(sa) <= len(sr) or sa[: len(sr)] != sr:
        return None
    return "/".join(sa[len(sr) :])


def shown(abs_path: str, root: str) -> str:
    """A stored path as a page shows it: relative when it lies below
    ``root``, else whole.

    A plain string comparison — the stored paths are canonical already, and
    the voice map shortens one path per recording (tens of thousands). With
    the separator: ``/v/show extras/a.mp4`` does not lie below
    ``/v/show`` — a bare ``startswith`` shortened it to ``extras/a.mp4``,
    a path that leads nowhere.
    """
    p = abs_path or ""
    prefix = (root or "").rstrip("/") + "/"
    return p[len(prefix) :] if p.startswith(prefix) else p


def abs_from(rel_path: str | os.PathLike, root: str | os.PathLike) -> str:
    """``rel_path`` + root → canonical ``abs_path``.

    Counterpart to :func:`rel_from`. Goes through :func:`canon` so duplicate
    separators and ``..`` don't make it into the database — that's exactly
    what happened before with ``str(media_dir) + "/" + rel_path``.
    """
    r = canon(root)
    if not r:
        return ""
    rel = str(rel_path or "")
    # Whether a backslash is separator or name character is decided by the ROOT:
    # on Windows it separates, on POSIX it's a valid filename character. A
    # relative path alone doesn't reveal that.
    if _is_windows_style(r):
        rel = rel.replace("\\", "/")
    rel = rel.strip("/")
    if not rel:
        return r
    result = canon(r + "/" + rel)
    # A rel_path with ".." must not leave the root. Otherwise abs_from(
    # "../../etc/passwd", "/nas/video") becomes "/etc/passwd", which would go
    # into the database as an identity — input comes partly from requests.
    if not is_inside(result, r):
        return ""
    return result


def like_prefix(root: str | os.PathLike) -> str:
    """Canonical root → LIKE prefix with defused wildcards.

    ``%`` and ``_`` are allowed in filenames and folder names and must not act
    as LIKE placeholders. Without escaping, a query on ``/mnt/50%data`` also
    matches rows under ``/mnt/50-completely-different-data``.

    The matching ``ESCAPE '\\'`` belongs in the query; the returned value ends
    with ``/`` and is typically extended with ``|| '%'``.
    """
    prefix = canon(root) + "/"
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


#: Path separators of any system, control characters, and what a Windows or
#: SMB share refuses in a name (the upload folder is usually such a share)
_NAME_BAD_RE = re.compile(r'[\x00-\x1f\x7f/\\<>:"|?*]')
#: The longest name a file system takes is 255 bytes; the transcript beside
#: it adds ".json", the assembly a leading "." and ".part", a second
#: upload of the name "_12"
_NAME_MAX_BYTES = 240
_WINDOWS_DEVICES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def file_name(raw: str | None) -> str:
    """The name a file is saved under — an upload, a downloaded video's
    title: what it was called, in any script; only what a file system can't
    take, or what would leave the folder, is removed.

    werkzeug's ``secure_filename`` kept ASCII only: a Cyrillic or Japanese
    name shrank to its extension and was refused "without an extension",
    an accented one lost letters.
    """
    name = unicodedata.normalize("NFC", str(raw or ""))
    # The last part only — a browser may send a path (and "\\" is one on Windows)
    name = re.split(r"[/\\]", name)[-1]
    name = _NAME_BAD_RE.sub("_", name)
    # No hidden file, and nothing a share cuts off at the end
    name = name.lstrip(". ").rstrip(". ")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    ext = f".{ext}" if dot else ""
    if not stem.strip("_ "):
        stem = "upload"
    if stem.lower() in _WINDOWS_DEVICES:
        stem = f"_{stem}"
    room = _NAME_MAX_BYTES - len(ext.encode("utf-8"))
    while len(stem.encode("utf-8")) > room and stem:
        stem = stem[:-1]
    return f"{stem.rstrip('. ') or 'upload'}{ext}"

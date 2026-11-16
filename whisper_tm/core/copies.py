"""Copies of a recording — told by what the files hold, never by their names.

Many recordings of the library exist several times: as .mp4, .mkv and .webm
side by side, once more in a backup folder, under another name ("episode
12 (backup).mkv"). The same name is sometimes another recording, and
another name the same one, so only the content counts. Measured on 50
videos kept in three or four containers each:

* their lengths agree within 0.06 s;
* the words said within 20 s of the same moment overlap by 0.42–1.00
  (median 0.94 — separate transcriptions differ in a few words), those of
  different episodes of one series at the same moment by at most 0.27.

So two recordings are copies when their lengths agree and they say the same
around a moment they share (copies share their timeline). When neither says
anything there, the same size (not for formats whose size follows from
their length) or the same still decides. Words that differ overrule a size
or a picture that agree.

**The sound decides first** where the audio analysis has seen both
(whisper_tm/audio): its fingerprints found "the same recording" from start
to end, or one recording inside another and where — a re-encode, music
without words, a clip cut out of a stream. Where it knows nothing, the rest
above decides as before.

Used by Find (find/fusion.py) and the semantic search
(rag/retrieve.group_copies) — the same pair of files is one recording in
both.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Two recordings this close in length can be copies (seconds; the share
#: for long recordings)
LENGTH_SEC = 0.25
LENGTH_SHARE = 0.002
#: Moments of two copies this close are the same moment
NEAR_SEC = 15.0
#: Seconds on both sides of a moment whose words are compared
WINDOW_SEC = 20.0
#: The share of the words two copies have in common there (Jaccard)
WORDS = 0.35
#: Not the same length (a clip cut out of a stream, a re-upload with another
#: intro): the same passage only when the words agree this much
SAME_PASSAGE_WORDS = 0.6
#: Fewer different words than this around a moment say nothing about it
MIN_WORDS = 12
#: Stills of two copies this close (bits of a 64-bit difference hash) look alike
PICTURE_BITS = 6
#: Formats whose size follows from their length (uncompressed, fixed rate):
#: two different 30 s .wav files have the same size to the byte
SIZED_BY_LENGTH = (".wav", ".wave", ".aif", ".aiff", ".aifc", ".amr", ".au", ".snd", ".pcm")

_LABEL = re.compile(r"\[[^\]]{1,60}\]")
_WORD = re.compile(r"[^\W_]{2,}")


def same_length(a, b) -> bool:
    """Two lengths (seconds, or ``(size, length)``) close enough for copies."""
    la = a[1] if isinstance(a, (tuple, list)) else a
    lb = b[1] if isinstance(b, (tuple, list)) else b
    if not la or not lb:
        return False
    return abs(la - lb) <= max(LENGTH_SEC, LENGTH_SHARE * max(la, lb))


def sized_by_length(key: str) -> bool:
    return str(key or "").lower().endswith(SIZED_BY_LENGTH)


def word_set(text: str) -> frozenset:
    """The different words of a text — speaker labels out, lower case."""
    return frozenset(_WORD.findall(_LABEL.sub(" ", text or "").lower()))


def overlap(a: frozenset, b: frozenset) -> float | None:
    """Jaccard of two word sets — None when either says too little."""
    if len(a) < MIN_WORDS or len(b) < MIN_WORDS:
        return None
    return len(a & b) / len(a | b)


def picture_hash(path) -> int | None:
    """A 64-bit difference hash of a still — None for a flat one (a black
    frame looks like every black frame) or one that can't be read."""
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as im:
            g = im.convert("L")
            if ImageStat.Stat(g).stddev[0] < 8:
                return None
            px = g.resize((9, 8), Image.BILINEAR).tobytes()
    except Exception:  # noqa: BLE001 — compared without the picture then
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def facts(db, keys) -> dict:
    """``{abs_path: (size, length)}`` of the recordings in ``keys``."""
    keys = [k for k in dict.fromkeys(keys) if k]
    out: dict[str, tuple] = {}
    if not keys or db is None:
        return out
    with db.connection() as conn:
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            for r in conn.execute(
                f"SELECT abs_path, file_size, duration_sec FROM media_files "
                f"WHERE abs_path IN ({','.join('?' * len(part))})",
                part,
            ):
                out[r[0]] = (int(r[1] or 0), float(r[2] or 0.0))
    return out


def sound_relation(a: str, b: str) -> tuple[str, str, float] | None:
    """What the audio analysis knows of two recordings: ``("same", a, 0)``,
    or ``("contains", whole, where the other begins in it)`` — None when it
    knows nothing (or isn't there)."""
    try:
        from whisper_tm.audio.runner import enabled
        from whisper_tm.audio.store import get_store, store_path

        if not enabled() or not store_path().exists():
            return None
        with get_store().connect() as c:
            row = c.execute(
                "SELECT r.kind, fa.path, r.offset FROM relations r "
                "JOIN files fa ON fa.id = r.a JOIN files fb ON fb.id = r.b "
                "WHERE (fa.path = ? AND fb.path = ?) OR (fa.path = ? AND fb.path = ?)",
                (a, b, b, a),
            ).fetchone()
    except Exception as e:  # noqa: BLE001 — compared without the sound then
        logger.debug("Copies: no sound relation of %s and %s: %s", a, b, e)
        return None
    return (row[0], row[1], float(row[2] or 0.0)) if row else None


class Looks:
    """What comparing recordings needs, looked up once: their size and
    length, the words said around a moment, a still's hash.

    ``still(key, t, moment)``: a still of that moment on this disk, or None —
    the caller knows where its stills are (nothing is cut for a comparison).
    """

    def __init__(self, db, keys=(), still=None):
        self.db = db
        self.facts = facts(db, keys)
        self._still = still
        self._tid: dict[str, int | None] = {}
        self._words: dict[tuple, frozenset] = {}
        self._pics: dict[str, int | None] = {}
        self._sounds: dict[frozenset, tuple | None] = {}

    def sound(self, a: str, b: str) -> tuple[str, str, float] | None:
        """:func:`sound_relation` of two recordings, asked once."""
        pair = frozenset((a, b))
        if pair not in self._sounds:
            self._sounds[pair] = sound_relation(a, b)
        return self._sounds[pair]

    def add(self, keys) -> None:
        """Look up the size and length of more recordings."""
        missing = [k for k in keys if k and k not in self.facts]
        if missing:
            self.facts.update(facts(self.db, missing))

    def words(self, key: str, t: float) -> frozenset:
        """The words said within :data:`WINDOW_SEC` of ``t`` in ``key``."""
        t = round(float(t or 0.0))
        got = self._words.get((key, t))
        if got is not None:
            return got
        out = frozenset()
        if self.db is not None:
            try:
                with self.db.connection() as conn:
                    if key not in self._tid:
                        row = conn.execute(
                            "SELECT id FROM transcripts WHERE abs_path=? AND complete=1 "
                            "ORDER BY id DESC LIMIT 1",
                            (key,),
                        ).fetchone()
                        self._tid[key] = row[0] if row else None
                    tid = self._tid[key]
                    if tid is not None:
                        segs = conn.execute(
                            "SELECT text FROM segments WHERE transcript_id=? AND end_sec >= ? "
                            "AND start_sec <= ? ORDER BY start_sec LIMIT 80",
                            (tid, t - WINDOW_SEC, t + WINDOW_SEC),
                        ).fetchall()
                        out = word_set(" ".join(s[0] or "" for s in segs))
            except Exception as e:  # noqa: BLE001 — compared without the words then
                logger.debug("Copies: words around %s not read: %s", key, e)
        self._words[(key, t)] = out
        return out

    def picture(self, key: str, t: float, moment=None) -> int | None:
        if self._still is None:
            return None
        try:
            path = self._still(key, t, moment)
        except Exception:  # noqa: BLE001
            path = None
        if not path:
            return None
        if path not in self._pics:
            self._pics[path] = picture_hash(path)
        return self._pics[path]

    def copy_reason(self, a: str, b: str, probes, moments=None) -> str | None:
        """Why recordings ``a`` and ``b`` are copies of each other — or None.
        ``probes``: moments (seconds) to compare them at, best first;
        ``moments``: ``{(key, t): Images & sounds moment id}`` for stills."""
        if a == b:
            return None
        snd = self.sound(a, b)
        if snd and snd[0] == "same":
            return "the same sound"
        fa, fb = self.facts.get(a), self.facts.get(b)
        if not same_length(fa, fb):
            return None
        differ = False
        for t in probes:
            share = overlap(self.words(a, t), self.words(b, t))
            if share is None:
                continue
            if share >= WORDS:
                return "the same length and words"
            differ = True
        if differ:
            return None
        if fa[0] and fa[0] == fb[0] and not (sized_by_length(a) or sized_by_length(b)):
            return "the same size and length"
        moments = moments or {}
        for t in probes:
            ha = self.picture(a, t, moments.get((a, round(t))))
            hb = self.picture(b, t, moments.get((b, round(t))))
            if ha is not None and hb is not None and bin(ha ^ hb).count("1") <= PICTURE_BITS:
                return "the same length and picture"
        return None

    def moment_reason(self, a: dict, b: dict) -> tuple[str, bool] | None:
        """Why moment ``b`` looks like moment ``a`` in another file —
        ``(reason, the recordings are copies)`` — or None. A moment is a dict
        with ``key`` and ``t`` (and optionally ``rec``, its class of copies,
        and ``moment``, its Images & sounds moment)."""
        if a["key"] == b["key"] or (a.get("rec") and a.get("rec") == b.get("rec")):
            return None  # two moments of one recording are two moments
        ta, tb = float(a.get("t") or 0), float(b.get("t") or 0)
        snd = self.sound(a["key"], b["key"])
        if snd and snd[0] == "same" and abs(ta - tb) <= NEAR_SEC:
            return "the same sound", True
        if snd and snd[0] == "contains":
            # the moment in the part, carried over into the whole
            tw, tp = (ta, tb) if snd[1] == a["key"] else (tb, ta)
            if abs(tw - (tp + snd[2])) <= NEAR_SEC:
                return "the same passage (its sound)", False
        fa, fb = self.facts.get(a["key"]), self.facts.get(b["key"])
        copy = same_length(fa, fb) and abs(ta - tb) <= NEAR_SEC
        share = overlap(self.words(a["key"], ta), self.words(b["key"], tb))
        if share is not None and share < WORDS:
            copy = False
        if (
            copy
            and fa[0]
            and fa[0] == fb[0]
            and not (sized_by_length(a["key"]) or sized_by_length(b["key"]))
        ):
            return "the same size and length", True
        if share is not None:
            if copy and share >= WORDS:
                return "the same length and words", True
            if share >= SAME_PASSAGE_WORDS:
                return "the same words", False
        if copy:
            ha = self.picture(a["key"], ta, a.get("moment"))
            hb = self.picture(b["key"], tb, b.get("moment"))
            if ha is not None and hb is not None and bin(ha ^ hb).count("1") <= PICTURE_BITS:
                return "the same length and picture", True
        return None

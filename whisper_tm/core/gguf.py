"""Reading a GGUF file's metadata, and rewriting one string value in a copy.

What it is for: the reranker's prompt template (core/rerank_server.py). The
Qwen3-Reranker GGUF bakes its instruction into ``tokenizer.chat_template.rerank``
— *"Given a web search query, retrieve relevant passages that answer the
query"* — and llama.cpp only fills in ``{query}`` and ``{document}``.
``--override-kv`` cannot replace it: its strings are limited to 128 bytes. A
copy of the file whose template line reads ``<Instruct>: {query}`` lets every
request carry its own instruction (Qwen3-Reranker-4B ranks better with an
own instruction than with the web-search one).

The rewrite touches the header only. GGUF v2/v3, little-endian::

    "GGUF" | u32 version | u64 tensor_count | u64 kv_count
    kv_count × (string key | u32 type | value)
    tensor_count × (string name | u32 n_dims | n_dims × u64 | u32 type | u64 offset)
    zero padding up to ``general.alignment`` (default 32)
    tensor data — the offsets above count from here

A string is a u64 length and UTF-8 bytes; an array a u32 element type, a u64
count and the elements. The tensor offsets are relative to the data section,
so a longer or shorter header changes none of them: the new header is
written, padded to the alignment, and the data section is copied byte for
byte.

**Without a second copy of the weights.** The new template is shorter than
the old one. A string of spaces under a key of its own (:data:`PAD_KEY` —
llama.cpp ignores keys it doesn't know) fills the header back up to where
the data section starts, so the copy can be a reflink of the model (btrfs,
XFS: the 2.4 GB of weights shared, not copied) with only its header
rewritten. Where the file system can't, the data section is copied as
before.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

MAGIC = b"GGUF"
#: GGUF value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = range(8)
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR = {
    T_UINT8: "<B",
    T_INT8: "<b",
    T_UINT16: "<H",
    T_INT16: "<h",
    T_UINT32: "<I",
    T_INT32: "<i",
    T_FLOAT32: "<f",
    T_BOOL: "<?",
    T_UINT64: "<Q",
    T_INT64: "<q",
    T_FLOAT64: "<d",
}
_SIZE = {t: struct.calcsize(f) for t, f in _SCALAR.items()}

DEFAULT_ALIGNMENT = 32
#: The string that pads a shortened header back to the data section's place.
PAD_KEY = "whisper_tm.header_padding"
#: FICLONE (linux/fs.h): make a file share another one's data blocks.
FICLONE = 0x40049409
#: Arrays of strings up to this length are returned as lists; longer ones
#: (the tokenizer's 151,000 tokens) only as ``GgufArray`` — nobody here
#: needs them, and building the list costs memory for nothing.
MAX_LISTED = 64


class GgufError(ValueError):
    """Not a GGUF file this module understands, or a broken one."""


class GgufArray(tuple):
    """An array left out of the metadata: ``(element_type, count)``."""


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def take(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GgufError("unexpected end of file in the header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def raw_string(self) -> bytes:
        n = self.u64()
        if n > 1 << 30:
            raise GgufError(f"implausible string length {n}")
        return self.take(n)

    def skip(self, n: int) -> None:
        self.fh.seek(n, os.SEEK_CUR)


def _header(r: _Reader) -> tuple[int, int, int]:
    if r.take(4) != MAGIC:
        raise GgufError("not a GGUF file")
    version = r.u32()
    if version not in (2, 3):
        raise GgufError(f"GGUF version {version} is not supported")
    return version, r.u64(), r.u64()


def _value(r: _Reader, vtype: int, want: bool, listed: int = MAX_LISTED):
    """Read (or, with ``want`` False, skip) one value."""
    if vtype == T_STRING:
        b = r.raw_string()
        return b.decode("utf-8", "replace") if want else None
    if vtype == T_ARRAY:
        etype, count = r.u32(), r.u64()
        if etype == T_STRING:
            if want and count <= MAX_LISTED:
                return [r.raw_string().decode("utf-8", "replace") for _ in range(count)]
            for _ in range(count):
                r.skip(r.u64())
            return GgufArray((etype, count))
        if etype not in _SIZE:
            raise GgufError(f"unknown array element type {etype}")
        if want and count <= listed:
            fmt = "<" + _SCALAR[etype][1] * count
            return list(struct.unpack(fmt, r.take(_SIZE[etype] * count)))
        r.skip(_SIZE[etype] * count)
        return GgufArray((etype, count))
    if vtype not in _SCALAR:
        raise GgufError(f"unknown value type {vtype}")
    raw = r.take(_SIZE[vtype])
    return struct.unpack(_SCALAR[vtype], raw)[0] if want else None


def read_metadata(path: str | Path, keys=None, listed: int = MAX_LISTED) -> dict:
    """The key/value metadata of a GGUF file.

    ``keys``: only these (the others are skipped, not decoded). Long arrays
    come back as :class:`GgufArray` — numbers up to ``listed`` of them as a
    list all the same (a value per layer of a deep model).
    """
    wanted = set(keys) if keys is not None else None
    out: dict = {}
    with open(path, "rb") as fh:
        r = _Reader(fh)
        _version, _tensors, kv_count = _header(r)
        for _ in range(kv_count):
            key = r.raw_string().decode("utf-8", "replace")
            vtype = r.u32()
            want = wanted is None or key in wanted
            value = _value(r, vtype, want, listed)
            if want:
                out[key] = value
    return out


def _raw_value(r: _Reader, vtype: int) -> bytes:
    """One value exactly as it is stored, for writing it back unchanged."""
    if vtype == T_STRING:
        b = r.raw_string()
        return struct.pack("<Q", len(b)) + b
    if vtype == T_ARRAY:
        etype, count = r.u32(), r.u64()
        parts = [struct.pack("<IQ", etype, count)]
        if etype == T_STRING:
            for _ in range(count):
                b = r.raw_string()
                parts.append(struct.pack("<Q", len(b)))
                parts.append(b)
        elif etype in _SIZE:
            parts.append(r.take(_SIZE[etype] * count))
        else:
            raise GgufError(f"unknown array element type {etype}")
        return b"".join(parts)
    if vtype not in _SIZE:
        raise GgufError(f"unknown value type {vtype}")
    return r.take(_SIZE[vtype])


def _clone(src: Path, dst: Path) -> bool:
    """``dst`` as a reflink of ``src`` — the data blocks shared, no space
    used (btrfs, XFS). False where the file system can't."""
    try:
        import fcntl

        with open(src, "rb") as s, open(dst, "wb") as d:
            fcntl.ioctl(d.fileno(), FICLONE, s.fileno())
        return True
    except (OSError, ImportError) as e:
        logger.debug("GGUF: no reflink of %s (%s) — copying instead", src.name, e)
        dst.unlink(missing_ok=True)
        return False


def rewrite_string(
    src: str | Path, dst: str | Path, key: str, value: str, *, allow_copy: bool = True
) -> Path:
    """Write ``dst``: a copy of ``src`` with the string ``key`` set to ``value``.

    Everything else stays byte-identical: the other metadata, the tensor
    infos and the whole data section — except for :data:`PAD_KEY`, which
    fills a shorter header up to the data section's place so that the copy
    can share the weights (see the module). Written to ``dst.tmp`` first and
    moved into place, so a reader never sees half a file. ``allow_copy``
    False: only that way, no full copy (too little disk space). Raises
    GgufError when ``key`` is missing or not a string, or nothing may be
    written.
    """
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fh:
        r = _Reader(fh)
        version, tensor_count, kv_count = _header(r)
        kvs: list[bytes] = []
        alignment = DEFAULT_ALIGNMENT
        replaced = False
        for _ in range(kv_count):
            kraw = r.raw_string()
            vtype = r.u32()
            raw = _raw_value(r, vtype)
            name = kraw.decode("utf-8", "replace")
            if name == PAD_KEY:
                continue  # a padding of an earlier rewrite
            if name == "general.alignment" and vtype == T_UINT32:
                alignment = struct.unpack("<I", raw)[0] or DEFAULT_ALIGNMENT
            if name == key:
                if vtype != T_STRING:
                    raise GgufError(f"{key} is not a string value")
                b = value.encode("utf-8")
                raw = struct.pack("<Q", len(b)) + b
                replaced = True
            kvs.append(struct.pack("<Q", len(kraw)) + kraw + struct.pack("<I", vtype) + raw)
        if not replaced:
            raise GgufError(f"{key} not found")
        infos: list[bytes] = []
        for _ in range(tensor_count):
            name = r.raw_string()
            n_dims = r.u32()
            if n_dims > 8:
                raise GgufError(f"implausible tensor rank {n_dims}")
            rest = r.take(8 * n_dims + 4 + 8)  # dims, type, offset
            infos.append(struct.pack("<Q", len(name)) + name + struct.pack("<I", n_dims) + rest)
        header_end = fh.tell()
        data_start = -(-header_end // alignment) * alignment
        size = src.stat().st_size
        if data_start > size:
            raise GgufError("the data section starts behind the end of the file")
        body = b"".join(kvs)
        tensors = b"".join(infos)
        # The padded header: as long as the old one up to the data section
        pad_key = PAD_KEY.encode()
        gap = data_start - (24 + len(body) + len(tensors))
        spaces = gap - (8 + len(pad_key) + 4 + 8)
        if spaces >= 0:
            header = (
                MAGIC
                + struct.pack("<IQQ", version, tensor_count, len(kvs) + 1)
                + body
                + struct.pack("<Q", len(pad_key))
                + pad_key
                + struct.pack("<I", T_STRING)
                + struct.pack("<Q", spaces)
                + b" " * spaces
                + tensors
            )
            if len(header) == data_start and _clone(src, tmp):
                try:
                    with open(tmp, "r+b") as out:
                        out.write(header)
                    if tmp.stat().st_size != size:
                        raise GgufError("the reflinked copy has a different length")
                    tmp.replace(dst)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
                return dst
        if not allow_copy:
            raise GgufError("no room for a copy of the weights, and no reflink possible")
        try:
            with open(tmp, "wb") as out:
                out.write(MAGIC + struct.pack("<IQQ", version, tensor_count, len(kvs)))
                out.write(body)
                out.write(tensors)
                out.write(b"\0" * ((-out.tell()) % alignment))
                new_start = out.tell()
                fh.seek(data_start)
                shutil.copyfileobj(fh, out, 64 << 20)
                written = out.tell()
            if written - new_start != size - data_start:
                raise GgufError("the copied data section has a different length")
            tmp.replace(dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return dst

"""The reranker: its GGUF copy that takes an instruction, and its server.

What is pinned:

* core/gguf.py — the metadata reader, and the rewrite of one string value:
  the header changes, the tensor infos and the whole data section stay
  byte-identical (checked on a tiny GGUF built here, with alignment 32 and
  64);
* the instruction: the template line ``<Instruct>: …\\n<Query>: {query}``
  becomes ``<Instruct>: {query}``, the query then carries the instruction —
  other templates are left alone; the copy is made once and again when the
  source changes, in the second place when the first isn't writable;
* the start command (rank pooling, every layer on the GPU), and the
  lifecycle with a tiny Python script in place of llama-server: started
  once, the GPU gate's stage held for as long as it runs, the card given
  back on idle and when someone waits at the gate, busy/off/missing/no-GPU
  as RerankUnavailable instead of a hang, scores in the documents' order,
  batches, progress, cancel.

No model is loaded and no GPU is used: the gate is a test instance with
free memory dictated, the server a script.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import textwrap
import threading
import time

import pytest

from whisper_tm.config import config
from whisper_tm.core import gguf
from whisper_tm.core import rerank_server as rs
from whisper_tm.core.gpu_gate import GpuGate

QWEN_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant "
    "passages that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


# ══════════════════════════════════════════════════════════════════════════
#  A tiny GGUF
# ══════════════════════════════════════════════════════════════════════════


def _s(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def _build_gguf(path, *, template=QWEN_TEMPLATE, alignment=None, tensors=None):
    """A valid GGUF v3: a few kinds of metadata, tensors at aligned offsets."""
    align = alignment or 32
    kvs = [
        ("general.architecture", gguf.T_STRING, "qwen3"),
        ("qwen3.pooling_type", gguf.T_UINT32, 4),
        ("tokenizer.ggml.tokens", gguf.T_ARRAY, (gguf.T_STRING, ["a", "b", "<|im_end|>"])),
        ("tokenizer.ggml.token_type", gguf.T_ARRAY, (gguf.T_INT32, [1, 1, 3])),
        ("general.sampling.temp", gguf.T_FLOAT32, 0.5),
    ]
    if template is not None:
        kvs.append((rs.TEMPLATE_KEY, gguf.T_STRING, template))
    if alignment:
        kvs.append(("general.alignment", gguf.T_UINT32, alignment))
    tensors = tensors or [
        ("token_embd.weight", [4, 3], 0, bytes(range(48))),
        ("output_norm.weight", [4], 0, b"\x01\x02\x03\x04" * 4),
    ]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for key, t, v in kvs:
        out += _s(key.encode()) + struct.pack("<I", t)
        if t == gguf.T_STRING:
            out += _s(v.encode())
        elif t == gguf.T_UINT32:
            out += struct.pack("<I", v)
        elif t == gguf.T_FLOAT32:
            out += struct.pack("<f", v)
        elif t == gguf.T_ARRAY:
            etype, values = v
            out += struct.pack("<IQ", etype, len(values))
            for x in values:
                out += _s(x.encode()) if etype == gguf.T_STRING else struct.pack("<i", x)
    offsets, off = [], 0
    for _name, _dims, _t, data in tensors:
        off = -(-off // align) * align
        offsets.append(off)
        off += len(data)
    for (name, dims, ttype, _data), o in zip(tensors, offsets):
        out += _s(name.encode()) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ttype, o)
    out += b"\0" * ((-len(out)) % align)
    start = len(out)
    for (_n, _d, _t, data), o in zip(tensors, offsets):
        out += b"\0" * (start + o - len(out))
        out += data
    path.write_bytes(bytes(out))
    return path


def _sections(path, align=32):
    """(header up to the tensor infos' end, tensor infos, data section)."""
    raw = path.read_bytes()
    with open(path, "rb") as fh:
        r = gguf._Reader(fh)
        _v, n_tensors, n_kv = gguf._header(r)
        for _ in range(n_kv):
            r.raw_string()
            gguf._raw_value(r, r.u32())
        info_start = fh.tell()
        for _ in range(n_tensors):
            r.raw_string()
            nd = r.u32()
            r.take(8 * nd + 12)
        info_end = fh.tell()
    data_start = -(-info_end // align) * align
    return raw[:info_start], raw[info_start:info_end], raw[data_start:]


class TestTheGgufRewrite:
    @pytest.mark.parametrize("alignment", [None, 64])
    def test_one_string_changes_everything_else_stays(self, tmp_path, alignment):
        src = _build_gguf(tmp_path / "m.gguf", alignment=alignment)
        new = rs.derived_template(QWEN_TEMPLATE)
        dst = gguf.rewrite_string(src, tmp_path / "out.gguf", rs.TEMPLATE_KEY, new)
        before, after = gguf.read_metadata(src), gguf.read_metadata(dst)
        assert after[rs.TEMPLATE_KEY] == new
        assert {k: v for k, v in after.items() if k != rs.TEMPLATE_KEY} == {
            k: v for k, v in before.items() if k != rs.TEMPLATE_KEY
        }
        align = alignment or 32
        _h1, infos1, data1 = _sections(src, align)
        _h2, infos2, data2 = _sections(dst, align)
        assert infos1 == infos2, "a tensor info changed — its offset would point elsewhere"
        assert data1 == data2, "the data section is not byte-identical"
        assert len(data2) == len(data1)
        assert not (tmp_path / "out.gguf.tmp").exists()

    @pytest.mark.parametrize("alignment", [None, 64])
    def test_a_reflinked_copy_keeps_the_data_section_in_place(
        self, tmp_path, monkeypatch, alignment
    ):
        """btrfs/XFS: the copy shares the weights — so the shorter header is
        padded back to the original's data offset, and only the header is
        written into the clone."""
        cloned = []

        def clone(src, dst):  # tmpfs has no reflinks: a plain copy
            dst.write_bytes(src.read_bytes())
            cloned.append(dst)
            return True

        monkeypatch.setattr(gguf, "_clone", clone)
        tensors = [
            ("token_embd.weight", [4, 3], 0, bytes(range(48)) * 40),
            ("output_norm.weight", [4], 0, b"\x01\x02\x03\x04" * 4),
        ]
        src = _build_gguf(tmp_path / "m.gguf", alignment=alignment, tensors=tensors)
        new = rs.derived_template(QWEN_TEMPLATE)
        dst = gguf.rewrite_string(src, tmp_path / "out.gguf", rs.TEMPLATE_KEY, new)
        assert cloned and dst.stat().st_size == src.stat().st_size
        align = alignment or 32
        h1, infos1, data1 = _sections(src, align)
        h2, infos2, data2 = _sections(dst, align)
        assert infos1 == infos2 and data1 == data2
        # the data section starts where it did: the clone's blocks stay shared
        assert src.read_bytes()[-len(data1) :] == dst.read_bytes()[-len(data2) :]
        assert len(h2) + len(infos2) == len(src.read_bytes()) - len(data1)
        after = gguf.read_metadata(dst)
        assert after[rs.TEMPLATE_KEY] == new
        pad = after.pop(gguf.PAD_KEY)
        assert set(pad) <= {" "}
        before = gguf.read_metadata(src)
        assert {k: v for k, v in after.items() if k != rs.TEMPLATE_KEY} == {
            k: v for k, v in before.items() if k != rs.TEMPLATE_KEY
        }
        # deriving from the derived file again: one padding, not two
        again = gguf.rewrite_string(dst, tmp_path / "again.gguf", rs.TEMPLATE_KEY, new + " ")
        meta = gguf.read_metadata(again)
        assert meta[rs.TEMPLATE_KEY] == new + " " and gguf.PAD_KEY in meta
        with open(again, "rb") as fh:
            _v, _t, n_kv = gguf._header(gguf._Reader(fh))
        assert n_kv == len(before) + 1

    def test_a_longer_header_or_no_reflink_copies_the_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gguf, "_clone", lambda s, d: False)
        src = _build_gguf(tmp_path / "m.gguf")
        dst = gguf.rewrite_string(src, tmp_path / "out.gguf", rs.TEMPLATE_KEY, "x")
        assert gguf.PAD_KEY not in gguf.read_metadata(dst)
        assert _sections(src)[2] == _sections(dst)[2]
        with pytest.raises(gguf.GgufError, match="no reflink"):
            gguf.rewrite_string(src, tmp_path / "o2.gguf", rs.TEMPLATE_KEY, "x", allow_copy=False)
        assert not (tmp_path / "o2.gguf").exists()
        # a header that grows can't be padded: copied even where reflinks work
        monkeypatch.setattr(gguf, "_clone", lambda s, d: pytest.fail("cloned"))
        longer = gguf.rewrite_string(src, tmp_path / "o3.gguf", rs.TEMPLATE_KEY, QWEN_TEMPLATE * 3)
        assert gguf.read_metadata(longer)[rs.TEMPLATE_KEY] == QWEN_TEMPLATE * 3

    def test_the_reader_skips_what_it_is_not_asked_for(self, tmp_path):
        src = _build_gguf(tmp_path / "m.gguf")
        assert gguf.read_metadata(src, keys={rs.TEMPLATE_KEY}) == {rs.TEMPLATE_KEY: QWEN_TEMPLATE}
        meta = gguf.read_metadata(src)
        assert meta["qwen3.pooling_type"] == 4
        assert meta["tokenizer.ggml.tokens"] == ["a", "b", "<|im_end|>"]
        assert meta["tokenizer.ggml.token_type"] == [1, 1, 3]

    def test_long_arrays_are_summarised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gguf, "MAX_LISTED", 2)
        meta = gguf.read_metadata(_build_gguf(tmp_path / "m.gguf"))
        assert meta["tokenizer.ggml.tokens"] == (gguf.T_STRING, 3)
        assert isinstance(meta["tokenizer.ggml.tokens"], gguf.GgufArray)

    def test_refusals(self, tmp_path):
        src = _build_gguf(tmp_path / "m.gguf")
        with pytest.raises(gguf.GgufError, match="not found"):
            gguf.rewrite_string(src, tmp_path / "o.gguf", "no.such.key", "x")
        with pytest.raises(gguf.GgufError, match="not a string"):
            gguf.rewrite_string(src, tmp_path / "o.gguf", "qwen3.pooling_type", "x")
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"GGML" + b"\0" * 40)
        with pytest.raises(gguf.GgufError, match="not a GGUF"):
            gguf.read_metadata(bad)
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(src.read_bytes()[:60])
        with pytest.raises(gguf.GgufError):
            gguf.read_metadata(cut)
        assert not (tmp_path / "o.gguf").exists()


# ══════════════════════════════════════════════════════════════════════════
#  The instruction
# ══════════════════════════════════════════════════════════════════════════


class TestTheInstruction:
    def test_the_instruction_line_takes_the_query(self):
        new = rs.derived_template(QWEN_TEMPLATE)
        assert "<Instruct>: {query}\n<Document>: {document}" in new
        assert "web search" not in new
        assert new.replace("<Instruct>: {query}", "") == QWEN_TEMPLATE.replace(
            "<Instruct>: Given a web search query, retrieve relevant passages that answer "
            "the query\n<Query>: {query}",
            "",
        )

    @pytest.mark.parametrize(
        "template",
        [
            None,
            "",
            "<Query>: {query}\n<Document>: {document}",
            QWEN_TEMPLATE.replace("{document}", ""),
            QWEN_TEMPLATE + QWEN_TEMPLATE,  # two such lines
        ],
    )
    def test_other_templates_are_left_alone(self, template):
        assert rs.derived_template(template) is None

    def test_an_already_derived_template_is_not_derived_again(self):
        assert rs.derived_template(rs.derived_template(QWEN_TEMPLATE)) is None

    def test_the_query_carries_the_instruction_only_where_it_is_taken(self):
        assert rs.build_query("alice", "Judge it.", True) == "Judge it.\n<Query>: alice"
        assert rs.build_query("alice", "Judge it.", False) == "alice"
        assert rs.build_query("alice", "", True) == "alice"


class TestTheDerivedModel:
    def test_written_once_and_reused(self, tmp_path, monkeypatch):
        (tmp_path / "models").mkdir()
        src = _build_gguf(tmp_path / "models" / "Reranker.gguf")
        path, ok = rs.prepare_model(src)
        assert ok and path == tmp_path / "models" / "derived" / "Reranker.instruct.gguf"
        side = json.loads(rs._sidecar(path).read_text())
        assert side["size"] == src.stat().st_size and side["source"] == str(src)
        assert gguf.read_metadata(path, keys={rs.TEMPLATE_KEY})[
            rs.TEMPLATE_KEY
        ] == rs.derived_template(QWEN_TEMPLATE)

        def never(*a, **k):
            raise AssertionError("written again")

        monkeypatch.setattr(gguf, "rewrite_string", never)
        assert rs.prepare_model(src) == (path, True)

    def test_a_changed_source_is_derived_again(self, tmp_path):
        src = _build_gguf(tmp_path / "Reranker.gguf")
        path, _ = rs.prepare_model(src)
        first = path.stat().st_mtime_ns
        time.sleep(0.01)
        _build_gguf(src, tensors=[("t", [8], 0, bytes(range(32)))])
        path2, ok = rs.prepare_model(src)
        assert ok and path2 == path and path.stat().st_mtime_ns != first
        assert json.loads(rs._sidecar(path).read_text())["size"] == src.stat().st_size

    def test_the_second_place_when_the_first_is_not_writable(self, tmp_path, monkeypatch):
        src = _build_gguf(tmp_path / "Reranker.gguf")
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where a folder should be")
        other = tmp_path / "data" / "derived" / "Reranker.instruct.gguf"
        monkeypatch.setattr(
            rs,
            "_derived_targets",
            lambda s: [blocked / "derived" / "Reranker.instruct.gguf", other],
        )
        assert rs.prepare_model(src) == (other, True)

    def test_without_the_known_line_the_original_serves(self, tmp_path):
        src = _build_gguf(tmp_path / "Other.gguf", template="{query} {document}")
        assert rs.prepare_model(src) == (src, False)
        src2 = _build_gguf(tmp_path / "NoTemplate.gguf", template=None)
        assert rs.prepare_model(src2) == (src2, False)
        assert not (tmp_path / "derived").exists()

    def test_an_unreadable_model_serves_the_original(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"nothing")
        assert rs.prepare_model(bad) == (bad, False)


# ══════════════════════════════════════════════════════════════════════════
#  The start command and the configuration
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rerank_cfg():
    """Set rerank.* for a test; the section is restored afterwards."""
    before = config.get("rerank")
    saved = dict(before) if isinstance(before, dict) else None

    def put(**kw):
        for k, v in kw.items():
            config.set("rerank", k, v)

    yield put
    with config._lock:
        if saved is None:
            config._data.pop("rerank", None)
        else:
            config._data["rerank"] = saved


def test_the_start_command(tmp_path, monkeypatch, rerank_cfg):
    b = tmp_path / "llama-server"
    b.write_text("")
    monkeypatch.setattr(rs, "binary", lambda: b)
    rerank_cfg(port=8390, ctx=8192, parallel=4, batch=2048, extra_args="--threads 4")
    argv = rs._start_command(tmp_path / "m.instruct.gguf")
    assert argv[0] == str(b)
    assert argv[argv.index("--model") + 1] == str(tmp_path / "m.instruct.gguf")
    for flag in ("--embedding", "--reranking"):
        assert flag in argv
    assert argv[argv.index("--pooling") + 1] == "rank"
    assert argv[argv.index("--port") + 1] == "8390"
    assert argv[argv.index("--ctx-size") + 1] == "8192"
    assert argv[argv.index("--parallel") + 1] == "4"
    assert argv[argv.index("--ubatch-size") + 1] == "2048"
    # GPU only: every layer on the card
    assert argv[argv.index("--n-gpu-layers") + 1] == "999"
    assert argv[-2:] == ["--threads", "4"]


def test_the_defaults_are_those_of_the_default_config():
    import yaml

    from whisper_tm.core.embed_server import base_directory

    default = yaml.safe_load(
        (base_directory() / "config.default.yaml").read_text(encoding="utf-8")
    )["rerank"]
    assert default == rs.DEFAULTS


def test_available_says_what_is_missing(monkeypatch, tmp_path, rerank_cfg):
    rerank_cfg(enabled=False)
    assert rs.available() == (False, "switched off (rerank.enabled)")
    rerank_cfg(enabled=True, local_model="models/gguf/nothing-here.gguf")
    monkeypatch.setattr(rs, "binary", lambda: None)
    assert "llama-server not found" in rs.available()[1]
    monkeypatch.setattr(rs, "binary", lambda: tmp_path)
    assert rs.available() == (False, "model file not found: models/gguf/nothing-here.gguf")
    rerank_cfg(local_model="")
    assert "no reranker model set" in rs.available()[1]


def test_notes_say_why_the_order_is_by_similarity():
    assert (
        rs.RerankUnavailable("busy").note()
        == "Reranker busy (the GPU is in use) — sorted by meaning and wording"
    )
    assert rs.RerankUnavailable("off").note().startswith("Reranker switched off")
    assert "x.gguf" in rs.RerankUnavailable("missing", "model file not found: x.gguf").note()
    assert (
        rs.RerankUnavailable("failed", "boom").note()
        == "Reranker failed (boom) — sorted by meaning and wording"
    )


# ══════════════════════════════════════════════════════════════════════════
#  The server — a tiny script in place of llama-server
# ══════════════════════════════════════════════════════════════════════════

# /health → 200 after `--delay`; /v1/rerank → a score per document: 0.9 for
# one with "insult" in it, else 0.01 + 0.001 × its length; every query is
# appended to `--log`. `--crash` ends it at once.
MINI_RERANKER = textwrap.dedent("""
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    a = sys.argv
    if '--crash' in a:
        sys.stderr.write('deliberately failed\\n'); raise SystemExit(3)
    port = int(a[a.index('--port') + 1])
    log = a[a.index('--log') + 1]
    time.sleep(float(a[a.index('--delay') + 1]) if '--delay' in a else 0.0)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with open(log, 'a') as fh:
                fh.write(json.dumps({'query': body['query'], 'n': len(body['documents'])}) + '\\n')
            res = [{'index': i, 'relevance_score': 0.9 if 'insult' in d else 0.01 + 0.001 * len(d)}
                   for i, d in enumerate(body['documents'])]
            out = json.dumps({'results': list(reversed(res))}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out))); self.end_headers()
            self.wfile.write(out)
    HTTPServer(('127.0.0.1', port), H).serve_forever()
""")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def place(tmp_path, monkeypatch, rerank_cfg):
    """A RerankServer that starts the mini script, with a gate of its own and
    11 GB of free VRAM dictated."""
    script = tmp_path / "mini_reranker.py"
    script.write_text(MINI_RERANKER)
    log = tmp_path / "queries.jsonl"
    log.write_text("")
    (tmp_path / "models").mkdir()
    model = _build_gguf(tmp_path / "models" / "Reranker.gguf")
    port = _free_port()
    rerank_cfg(
        enabled=True,
        port=port,
        local_model=str(model),
        start_timeout_sec=20,
        idle_stop_sec=300,
        vram_need_mb=4800,
    )
    monkeypatch.setattr(rs, "binary", lambda: script)
    monkeypatch.setattr(rs, "base_directory", lambda: tmp_path)
    monkeypatch.setattr(rs, "_free_vram_mb", lambda: 11000)
    extra: list[str] = []

    def command(model_path):
        return [sys.executable, str(script), "--port", str(port), "--log", str(log), *extra]

    monkeypatch.setattr(rs, "_start_command", command)
    gate = GpuGate(free_mb=lambda: 11000)
    server = rs.RerankServer(gate=gate)
    yield {
        "server": server,
        "gate": gate,
        "log": log,
        "extra": extra,
        "port": port,
        "model": model,
        "tmp": tmp_path,
    }
    server.stop("test over", force=True)


def _queries(log):
    return [json.loads(x) for x in log.read_text().splitlines() if x.strip()]


def _until(cond, sec=10.0):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class TestTheServer:
    def test_scores_come_back_in_the_documents_order(self, place):
        s = place["server"]
        docs = ["a", "he got insulted", "abc"]
        scores = s.rerank("alice", docs, instruction="Judge it.")
        assert scores == pytest.approx([0.011, 0.9, 0.013])
        q = _queries(place["log"])
        # The copy of the model takes an instruction: it comes with the query
        assert q == [{"query": "Judge it.\n<Query>: alice", "n": 3}]
        st = s.state()
        assert st["running"] and st["instructable"] is True and st["pid"] > 0
        assert st["model"].endswith("Reranker.instruct.gguf")

    def test_batches_progress_and_one_start(self, place):
        s = place["server"]
        seen = []
        scores = s.rerank(
            "q",
            [f"doc {i}" for i in range(70)],
            batch=32,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert len(scores) == 70
        assert [x["n"] for x in _queries(place["log"])] == [32, 32, 6]
        assert seen == [(32, 70), (64, 70), (70, 70)]
        pid = s.pid()
        s.rerank("q", ["again"])
        assert s.pid() == pid, "started a second time"

    def test_the_default_instruction(self, place, rerank_cfg):
        rerank_cfg(instruction="Is it so?")
        place["server"].rerank("alice", ["x"])
        assert _queries(place["log"])[0]["query"] == "Is it so?\n<Query>: alice"

    def test_it_holds_the_gate_while_it_runs(self, place):
        s, gate = place["server"], place["gate"]
        s.rerank("q", ["x"])
        assert rs.STAGE in gate.state()["stages"], "running without the GPU gate"
        assert s.stop("test") is True
        assert _until(lambda: not gate.state()["stages"]), "the stage was not given back"
        assert s.pid() is None

    def test_without_process_groups_it_still_stops_and_gives_the_gate_back(
        self, place, monkeypatch
    ):
        """Windows has no os.killpg/os.getpgid: the AttributeError skipped the
        release of the gate, and the queue waited for a card nobody held."""
        import os

        s, gate = place["server"], place["gate"]
        s.rerank("q", ["x"])
        monkeypatch.delattr(os, "killpg")
        monkeypatch.delattr(os, "getpgid")
        assert s.stop("test") is True
        assert _until(lambda: not gate.state()["stages"]), "the stage was not given back"
        assert s.pid() is None

    def test_a_busy_card_is_unavailable_not_a_hang(self, place):
        s, gate = place["server"], place["gate"]
        held, release = threading.Event(), threading.Event()

        def whisper():
            with gate.exclusive("asr"):
                held.set()
                release.wait(10)

        t = threading.Thread(target=whisper, daemon=True)
        t.start()
        assert held.wait(5)
        try:
            t0 = time.monotonic()
            with pytest.raises(rs.RerankUnavailable) as e:
                s.rerank("q", ["x"], wait_sec=0.5)
            assert e.value.reason == "busy"
            assert time.monotonic() - t0 < 5
            assert s.pid() is None and _queries(place["log"]) == []
        finally:
            release.set()
            t.join(5)
        assert _until(lambda: not gate.state()["stages"] and gate.waiting() == 0)

    def test_too_little_vram_is_busy(self, place, monkeypatch):
        monkeypatch.setattr(rs, "_free_vram_mb", lambda: 2000)
        with pytest.raises(rs.RerankUnavailable, match="only 2000 MB of VRAM free") as e:
            place["server"].rerank("q", ["x"], wait_sec=0.5)
        assert e.value.reason == "busy"
        assert _until(lambda: not place["gate"].state()["stages"])

    def test_no_gpu_off_and_missing(self, place, monkeypatch, rerank_cfg):
        s = place["server"]
        monkeypatch.setattr(rs, "_free_vram_mb", lambda: None)
        with pytest.raises(rs.RerankUnavailable) as e:
            s.rerank("q", ["x"])
        assert e.value.reason == "no_gpu"
        monkeypatch.setattr(rs, "_free_vram_mb", lambda: 11000)
        rerank_cfg(enabled=False)
        with pytest.raises(rs.RerankUnavailable) as e:
            s.rerank("q", ["x"])
        assert e.value.reason == "off"
        rerank_cfg(enabled=True, local_model="models/gguf/gone.gguf")
        with pytest.raises(rs.RerankUnavailable) as e:
            s.rerank("q", ["x"])
        assert e.value.reason == "missing" and "gone.gguf" in str(e.value)
        assert s.pid() is None

    def test_switched_off_while_running_it_stops_at_once(self, place, rerank_cfg):
        s = place["server"]
        s.rerank("q", ["x"])
        assert s.pid()
        rerank_cfg(enabled=False)
        with pytest.raises(rs.RerankUnavailable) as e:
            s.rerank("q", ["x"])
        assert e.value.reason == "off" and s.pid() is None
        assert _until(lambda: not place["gate"].state()["stages"])

    def test_a_crash_at_startup_reports_and_gives_the_card_back(self, place):
        place["extra"].append("--crash")
        with pytest.raises(rs.RerankUnavailable, match="ended during its start") as e:
            place["server"].rerank("q", ["x"])
        assert e.value.reason == "failed"
        assert place["server"].pid() is None
        assert _until(lambda: not place["gate"].state()["stages"])

    def test_a_bad_start_command_gives_the_card_back(self, place, monkeypatch):
        # Regression: a ValueError from shlex (bad rerank.extra_args) kept the
        # 'rerank' stage held forever — only OSError released it
        import shlex

        monkeypatch.setattr(rs, "_start_command", lambda m: shlex.split("--foo 'unbalanced"))
        with pytest.raises(rs.RerankUnavailable) as e:
            place["server"].rerank("q", ["x"], wait_sec=2)
        assert e.value.reason == "failed"
        assert _until(lambda: not place["gate"].state()["stages"]), "the stage stays held"

    def test_others_wait_for_a_cold_start_instead_of_posting_early(self, place):
        # Regression: alive and the address were enough for the fast path — a
        # second caller posted before /health answered and failed
        place["extra"] += ["--delay", "1.5"]
        s = place["server"]
        out: dict = {}

        def call(name):
            try:
                out[name] = s.rerank("q", ["x"], wait_sec=10)
            except Exception as e:  # noqa: BLE001
                out[name] = e

        first = threading.Thread(target=call, args=("first",))
        first.start()
        assert _until(lambda: s.pid() is not None)
        second = threading.Thread(target=call, args=("second",))
        second.start()
        first.join(20)
        second.join(20)
        assert isinstance(out["first"], list) and isinstance(out["second"], list), out

    def test_cancel_ends_a_long_run_between_batches(self, place):
        calls = []

        def cancel():
            return len(calls) >= 1

        with pytest.raises(InterruptedError):
            place["server"].rerank(
                "q",
                [f"d{i}" for i in range(100)],
                batch=32,
                cancel=cancel,
                progress=lambda d, t: calls.append(d),
            )
        assert [x["n"] for x in _queries(place["log"])] == [32]

    def test_idle_stops_it(self, place, rerank_cfg, monkeypatch):
        monkeypatch.setattr(rs, "WATCH_TICK_SEC", 0.1)
        rerank_cfg(idle_stop_sec=0.5)
        s = place["server"]
        s.rerank("q", ["x"])
        assert _until(lambda: s.pid() is None, 10), "still running after the idle time"
        assert _until(lambda: not place["gate"].state()["stages"])

    def test_it_steps_aside_when_the_queue_waits(self, place, monkeypatch):
        monkeypatch.setattr(rs, "WATCH_TICK_SEC", 0.1)
        monkeypatch.setattr(rs, "YIELD_GRACE_SEC", 0.2)
        s, gate = place["server"], place["gate"]
        s.rerank("q", ["x"])
        assert s.pid()
        got = threading.Event()

        def whisper():
            with gate.exclusive("asr"):
                got.set()

        t = threading.Thread(target=whisper, daemon=True)
        t.start()
        assert got.wait(10), "Whisper never got the card"
        assert s.pid() is None
        t.join(5)

    def test_a_call_in_flight_is_not_stopped(self, place):
        s = place["server"]
        with s.session(wait_sec=5) as (ep, instructable):
            assert instructable is True
            assert s.stop("idle") is False
            assert s.pid()
        assert s.stop("idle") is True

    def test_a_long_run_gives_the_card_back_between_batches(self, place, monkeypatch):
        monkeypatch.setattr(rs, "YIELD_AFTER_SEC", 0.0)
        s, gate = place["server"], place["gate"]
        got, release = threading.Event(), threading.Event()

        def whisper():
            with gate.exclusive("asr"):
                got.set()
                release.wait(10)

        def progress(done, total):
            if done == 32:
                # Someone comes to the gate after the first batch
                threading.Thread(target=whisper, daemon=True).start()
                assert _until(lambda: gate.waiting() == 1)

        try:
            # The waiter keeps the card — the run's second batch waits for it
            # and gives up after its wait
            with pytest.raises(rs.RerankUnavailable) as e:
                s.rerank(
                    "q",
                    [f"d{i}" for i in range(64)],
                    batch=32,
                    wait_sec=0.5,
                    progress=progress,
                    yield_between=True,
                )
            assert e.value.reason == "busy"
            assert got.wait(5), "the waiting exclusive never got the card"
            assert s.pid() is None
            assert [x["n"] for x in _queries(place["log"])] == [32]
        finally:
            release.set()

    def test_an_answer_without_a_score_for_every_document_fails(self, monkeypatch):
        import requests

        class R:
            status_code = 200

            def json(self):
                return {"results": [{"index": 0, "relevance_score": 0.5}]}

        monkeypatch.setattr(requests, "post", lambda *a, **k: R())
        with pytest.raises(rs.RerankUnavailable, match="without a score"):
            rs._post_rerank("http://x", "q", ["a", "b"])

        class Bad:
            status_code = 500
            text = "boom"

        monkeypatch.setattr(requests, "post", lambda *a, **k: Bad())
        with pytest.raises(rs.RerankUnavailable, match="answered 500"):
            rs._post_rerank("http://x", "q", ["a"])

    def test_documents_are_cut_to_their_length(self, place):
        place["server"].rerank("q", ["x" * 5000])
        # 0.01 + 0.001 × length: the server saw 2,000 characters
        assert place["server"].rerank("q", ["y" * 5000]) == pytest.approx(
            [0.01 + 0.001 * rs.DOC_CHARS]
        )


def test_the_reranker_has_its_role_in_the_vram_list(monkeypatch):
    from whisper_tm.core import vram_procs

    monkeypatch.setattr(rs.SERVER, "pid", lambda: 424242)
    assert vram_procs._role_pids()[424242] == "reranker"
    assert vram_procs.ROLE_LABELS["reranker"] == "Reranker"

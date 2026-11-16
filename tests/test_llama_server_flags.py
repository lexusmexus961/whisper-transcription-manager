"""llama-server flags the build may not know, and the servers' logs.

Measured: llama-server keeps every finished prompt's state in RAM
(``--cache-ram``, 8 GiB by default). A rerank server can hold several GB for
it, and an embedding run copies tens of MB off the card for each of its
passages — neither ever comes back to a prompt. ``--cache-ram 0`` switches it
off, but an older build stops at the unknown argument, so the flag goes in
only where ``--help`` lists it. A long embedding run also leaves a log of
hundreds of MB.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from whisper_tm.core import embed_server, llama_server, rerank_server

HELP_NEW = """----- common params -----
-h,    --help, --usage                  print usage and exit
-cram, --cache-ram N                    set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 -
                                        disable)
--cache-reuse N                         min chunk size to attempt reusing from the cache
"""
HELP_OLD = """----- common params -----
-h,    --help, --usage                  print usage and exit
--cache-reuse N                         min chunk size to attempt reusing from the cache
"""


def _fake_server(tmp_path: Path, help_text: str, name: str = "llama-server") -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\ncat <<'EOF'\n" + help_text + "EOF\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_a_flag_the_help_lists_is_supported(tmp_path):
    new = _fake_server(tmp_path, HELP_NEW)
    assert llama_server.supports("--cache-ram", new)
    assert llama_server.no_prompt_cache_args(new) == ["--cache-ram", "0"]


def test_an_older_build_gets_no_unknown_flag(tmp_path):
    old = _fake_server(tmp_path, HELP_OLD)
    assert not llama_server.supports("--cache-ram", old)
    assert llama_server.no_prompt_cache_args(old) == []


def test_a_flag_is_not_found_inside_a_longer_one(tmp_path):
    new = _fake_server(tmp_path, HELP_NEW)
    assert not llama_server.supports("--cache", new)
    assert llama_server.supports("--cache-reuse", new)


def test_no_binary_supports_nothing(tmp_path):
    assert llama_server.help_text(tmp_path / "missing") == ""
    assert llama_server.no_prompt_cache_args(tmp_path / "missing") == []


def test_the_help_is_read_with_the_card_hidden(tmp_path):
    p = tmp_path / "llama-server"
    p.write_text('#!/bin/sh\necho "--cache-ram N cuda=[$CUDA_VISIBLE_DEVICES]"\n', encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    assert "cuda=[]" in llama_server.help_text(p)


@pytest.mark.parametrize("helptext, expected", [(HELP_NEW, True), (HELP_OLD, False)])
def test_the_rerank_server_starts_without_a_prompt_cache(tmp_path, monkeypatch, helptext, expected):
    fake = _fake_server(tmp_path, helptext)
    monkeypatch.setattr(rerank_server, "binary", lambda: fake)
    argv = rerank_server._start_command(tmp_path / "model.gguf")
    assert argv[0] == str(fake)
    has = "--cache-ram" in argv
    assert has is expected
    if has:
        assert argv[argv.index("--cache-ram") + 1] == "0"


@pytest.mark.parametrize("helptext, expected", [(HELP_NEW, True), (HELP_OLD, False)])
def test_the_embedding_server_starts_without_a_prompt_cache(
    tmp_path, monkeypatch, helptext, expected
):
    fake = _fake_server(tmp_path, helptext)
    model = tmp_path / "embed.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(embed_server, "binary", lambda: fake)
    monkeypatch.setattr(embed_server, "model_file", lambda: model)
    argv = embed_server._start_command(force_cpu=True)
    assert ("--cache-ram" in argv) is expected
    if expected:
        assert argv[argv.index("--cache-ram") + 1] == "0"


def test_a_long_log_is_cut_to_its_tail_at_a_line_start(tmp_path):
    log = tmp_path / "embed-server.log"
    lines = [f"line {i:07d} making room for prompt cache entry" for i in range(60000)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    size = log.stat().st_size
    llama_server.trim_log(log, max_bytes=size // 2, keep_bytes=4096)
    text = log.read_text(encoding="utf-8")
    assert text.startswith("=== (older output cut) ===\n")
    kept = text.splitlines()[1:]
    assert kept[-1] == lines[-1]
    assert kept[0] in lines  # a whole line, not a cut one
    assert len(text) < 4200


def test_a_short_log_stays_as_it_is(tmp_path):
    log = tmp_path / "rerank-server.log"
    log.write_text("one\ntwo\n", encoding="utf-8")
    llama_server.trim_log(log, max_bytes=1000)
    assert log.read_text(encoding="utf-8") == "one\ntwo\n"
    llama_server.trim_log(tmp_path / "missing.log")  # no error


def test_the_tail_of_a_big_log_is_read_from_its_end(tmp_path):
    log = tmp_path / "x.log"
    with open(log, "w", encoding="utf-8") as fh:
        for i in range(200000):
            fh.write(f"noise {i}\n")
        fh.write("error: unknown argument: --cache-ram\n\n")
    assert llama_server.log_tail(log, 2) == (
        "Last output: noise 199999 | error: unknown argument: --cache-ram"
    )
    assert llama_server.log_tail(None) == ""
    assert llama_server.log_tail(tmp_path / "missing.log") == ""


def test_the_servers_use_the_cheap_tail(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("a\nb\nc\n", encoding="utf-8")
    srv = rerank_server.RerankServer()
    srv._log = log
    assert srv._log_tail(2) == "Last output: b | c"
    emb = embed_server.EmbedServer()
    emb._log = log
    assert emb._log_tail(1) == "Last output: c"
    assert os.path.getsize(log) == 6


# ── Which GGUF the chat server loads ─────────────────────────────────────
# Without llama.model it took the first *.gguf by name — and the semantic
# search's "Qwen3-Embedding…" sorts before "qwen2.5…": the chat server
# loaded the embedding model.


@pytest.fixture
def gguf_folder(tmp_path, monkeypatch):
    folder = tmp_path / "models" / "gguf"
    folder.mkdir(parents=True)
    monkeypatch.setattr(llama_server, "base_directory", lambda: tmp_path)
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: default)
    return folder


def test_the_chat_model_is_not_an_embedding_reranker_or_projector(gguf_folder):
    for name in (
        "Qwen3-Embedding-4B-Q8_0.gguf",
        "Qwen3-Reranker-4B-Q4_K_M.gguf",
        "mmproj-F16.gguf",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
    ):
        (gguf_folder / name).write_bytes(b"GGUF")
    assert llama_server.model_file().name == "qwen2.5-7b-instruct-q4_k_m.gguf"


def test_a_split_model_is_loaded_from_its_first_part(gguf_folder):
    for name in ("chat-q4_k_m-00002-of-00002.gguf", "chat-q4_k_m-00001-of-00002.gguf"):
        (gguf_folder / name).write_bytes(b"GGUF")
    assert llama_server.model_file().name == "chat-q4_k_m-00001-of-00002.gguf"


def test_no_chat_model_is_none(gguf_folder):
    (gguf_folder / "Qwen3-Embedding-4B-Q8_0.gguf").write_bytes(b"GGUF")
    assert llama_server.model_file() is None


def test_a_relative_model_path_is_relative_to_the_project(gguf_folder, monkeypatch):
    (gguf_folder / "mine.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        llama_server,
        "_cfg",
        lambda key, default=None: "models/gguf/mine.gguf" if key == "model" else default,
    )
    assert llama_server.model_file() == gguf_folder / "mine.gguf"


@pytest.mark.parametrize(
    "layers,flash,expect_ngl,expect_fa",
    [
        (0, False, "0", "off"),  # 0 = the CPU only, false = off: said, not left to the server
        (99, True, "99", "on"),
    ],
)
def test_the_gpu_layers_and_flash_attention_are_said(
    monkeypatch, tmp_path, layers, flash, expect_ngl, expect_fa
):
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    values = {"n_gpu_layers": layers, "flash_attn": flash}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(llama_server, "binary", lambda: tmp_path / "llama-server")
    monkeypatch.setattr(llama_server, "model_file", lambda: model)
    argv = llama_server.start_command()
    assert argv[argv.index("--n-gpu-layers") + 1] == expect_ngl
    assert argv[argv.index("-fa") + 1] == expect_fa


#: The chat flags across llama.cpp's builds: -fa a switch before mid-2025,
#: --reasoning-budget before --reasoning, and early builds with neither
HELP_2026 = """----- common params -----
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: enabled)
-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ('on', 'off', or 'auto', default:
"""
HELP_MID_2025 = """----- common params -----
-fa,   --flash-attn                     enable Flash Attention (default: disabled)
--jinja                                 use jinja template for chat (default: disabled)
--reasoning-budget N                    controls the amount of thinking allowed; currently only one of: -1 for
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the CPU
"""
HELP_EARLY_2025 = """----- common params -----
-fa,   --flash-attn                     enable Flash Attention (default: disabled)
"""


@pytest.mark.parametrize(
    "help_, flash, expect",
    [
        (HELP_2026, True, ["--jinja", "--n-cpu-moe", "4", "-fa", "on", "--reasoning", "off"]),
        (HELP_MID_2025, True, ["--jinja", "--n-cpu-moe", "4", "-fa", "--reasoning-budget", "0"]),
        (HELP_MID_2025, False, ["--jinja", "--n-cpu-moe", "4", "--reasoning-budget", "0"]),
        (HELP_EARLY_2025, True, ["-fa"]),
    ],
)
def test_an_older_build_gets_only_what_it_knows(monkeypatch, tmp_path, help_, flash, expect):
    """An older llama-server stops at a flag it doesn't know — or at the "on"
    after its -fa switch — instead of starting."""
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    fake = _fake_server(tmp_path, help_)
    values = {"flash_attn": flash, "n_cpu_moe": 4}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(llama_server, "binary", lambda: fake)
    monkeypatch.setattr(llama_server, "model_file", lambda: model)
    argv = llama_server.start_command()
    assert argv[argv.index("--cache-type-v") + 2 :] == expect


# ── VRAM estimate of the health check ───────────────────────────────────────


def _gguf_header(path, arch: str, numbers: dict) -> Path:
    """A GGUF v3 with metadata only — what the estimate reads."""
    import struct

    from whisper_tm.core import gguf

    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b

    kvs = [("general.architecture", gguf.T_STRING, arch)]
    kvs += [
        (f"{arch}.{k}", gguf.T_ARRAY if isinstance(v, list) else gguf.T_UINT32, v)
        for k, v in numbers.items()
    ]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)))
    for key, t, v in kvs:
        out += s(key.encode()) + struct.pack("<I", t)
        if t == gguf.T_STRING:
            out += s(v.encode())
        elif t == gguf.T_ARRAY:
            # a value per layer: flags as bools, counts as numbers
            flags = isinstance(v[0], bool)
            out += struct.pack("<IQ", gguf.T_BOOL if flags else gguf.T_UINT32, len(v))
            out += struct.pack("<" + ("?" if flags else "I") * len(v), *v)
        else:
            out += struct.pack("<I", v)
    path.write_bytes(bytes(out))
    return path


#: Qwen2.5-7B's shape: 28 layers, 28 heads, 4 KV heads, 3584 wide (head 128)
DENSE = {
    "block_count": 28,
    "attention.head_count": 28,
    "attention.head_count_kv": 4,
    "embedding_length": 3584,
}


def test_a_dense_model_counts_whole_on_the_card(tmp_path, monkeypatch):
    """n_cpu_moe does nothing for a dense model; the estimate counted it as
    half a gigabyte on the card all the same — and never warned."""
    model = _gguf_header(tmp_path / "dense.gguf", "qwen2", DENSE)
    values = {"n_cpu_moe": 39, "ctx_size": 32768, "ubatch_size": 8192}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    # KV: 32768 tokens × 28 layers × 4 heads × (128 + 128) × 2 bytes = 1792 MiB
    assert llama_server.vram_estimate_mb(model) == 1792 + 700
    values["cache_type_k"] = values["cache_type_v"] = "q8_0"
    assert llama_server.vram_estimate_mb(model) == int(1792 * 34 / 32 / 2 + 700)


def test_a_mixture_of_experts_keeps_its_experts_in_ram(tmp_path, monkeypatch):
    model = _gguf_header(
        tmp_path / "moe.gguf", "qwen3moe", {**DENSE, "block_count": 40, "expert_count": 128}
    )
    values = {"n_cpu_moe": 40, "ctx_size": 32768, "ubatch_size": 8192}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    assert llama_server.vram_estimate_mb(model) == 512 + 700


def test_an_unreadable_header_keeps_the_old_rule(tmp_path, monkeypatch):
    model = tmp_path / "broken.gguf"
    model.write_bytes(b"not a gguf")
    values = {"n_cpu_moe": 39, "ubatch_size": 4096}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    assert llama_server.vram_estimate_mb(model) == 512 + 350


#: Qwen3.5-9B's shape (the setup's model): 32 layers, every fourth one with
#: attention — 4 KV heads of 256 — the others linear attention
HYBRID = {
    "block_count": 32,
    "attention.head_count": 16,
    "attention.head_count_kv": 4,
    "attention.key_length": 256,
    "attention.value_length": 256,
    "embedding_length": 4096,
}


@pytest.mark.parametrize(
    "arch,extra",
    [
        ("qwen35", {"full_attention_interval": 4}),
        ("qwen35", {}),  # llama.cpp's default for the architecture
        ("hybrid", {"attention.recurrent_layers": [True, True, True, False] * 8}),
        ("hybrid", {"attention.head_count_kv": [0, 0, 0, 4] * 8}),
    ],
)
def test_a_hybrid_counts_the_cache_of_its_attention_layers(tmp_path, monkeypatch, arch, extra):
    """Counted in all 32 layers, the setup's own model looked like 4 GB of
    cache at 32k instead of 1 — more than its weights."""
    model = _gguf_header(tmp_path / "hybrid.gguf", arch, {**HYBRID, **extra})
    values = {"ctx_size": 32768, "ubatch_size": 512}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    # KV: 32768 tokens × 8 layers × 4 heads × (256 + 256) × 2 bytes = 1024 MiB
    assert llama_server.vram_estimate_mb(model) == 1024 + 43


def test_a_model_without_hybrid_layers_counts_them_all(tmp_path, monkeypatch):
    model = _gguf_header(tmp_path / "dense.gguf", "qwen3", HYBRID)
    values = {"ctx_size": 32768, "ubatch_size": 512}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    assert llama_server.vram_estimate_mb(model) == 4096 + 43


def test_a_deep_hybrid_gives_its_layers_past_the_listed_length(tmp_path, monkeypatch):
    """A value per layer of 80 layers is longer than the arrays the GGUF
    reader lists by default — the count of entries is no head count."""
    model = _gguf_header(
        tmp_path / "deep.gguf",
        "hybrid",
        {**HYBRID, "block_count": 80, "attention.head_count_kv": [0, 0, 0, 4] * 20},
    )
    values = {"ctx_size": 32768, "ubatch_size": 512}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    assert llama_server.vram_estimate_mb(model) == 2560 + 43


def test_a_hybrid_mixture_of_experts_counts_its_small_cache(tmp_path, monkeypatch):
    model = _gguf_header(
        tmp_path / "moe.gguf",
        "qwen35moe",
        {**HYBRID, "block_count": 40, "attention.head_count_kv": 2, "expert_count": 256},
    )
    values = {"n_cpu_moe": 40, "ctx_size": 32768, "ubatch_size": 512}
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: values.get(key, default))
    # KV: 32768 × 10 layers × 2 heads × 512 × 2 bytes = 640 MiB
    assert llama_server.vram_estimate_mb(model) == 512 + 640 + 43


def test_the_defaults_leave_the_layers_to_llama_cpp(monkeypatch, tmp_path):
    """The setup's model on whatever card: no layer count and no experts
    on the CPU in the command — llama.cpp fits the layers to the free VRAM."""
    import yaml

    defaults = yaml.safe_load(
        (Path(__file__).parents[1] / "config.default.yaml").read_text(encoding="utf-8")
    )["llama"]
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(llama_server, "_cfg", lambda key, default=None: defaults.get(key, default))
    monkeypatch.setattr(llama_server, "binary", lambda: tmp_path / "llama-server")
    monkeypatch.setattr(llama_server, "model_file", lambda: model)
    argv = llama_server.start_command()
    assert "--n-gpu-layers" not in argv and "--n-cpu-moe" not in argv
    assert argv[argv.index("--ctx-size") + 1] == "32768"
    assert argv[argv.index("--ubatch-size") + 1] == "512"

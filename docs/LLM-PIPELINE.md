# LLM Pipeline — Architecture, Formats, Operations

Overview of the LLM integration (summary, **correction**,
**translation**) and its chunking framework.

## 1. What the design answers

A first version could only do one thing: stuff the whole transcript into
one prompt and summarize it. Its problems, which the design below answers:

1. **System prompt injection**: emptying the system prompt field in the
   summarize dialog meant the frontend didn't send the field at all
   (`if (sysPrompt)`), and the backend silently injected the config
   default. Empty never meant empty.
2. **Forced preset selection**: the dialog automatically preselected the
   alphabetically first preset — instead of the configured system
   prompt.
3. `/api/llm/status` didn't return `system_prompt` at all; the frontend
   read `undefined`.
4. **No chunking**: a transcript longer than the model's context (e.g.
   32k tokens) → truncated input or error.
5. **Settings without effect verification**: `max_tokens: 256000`
   (meaningless as a response limit), `context_size: 0`
   ("take whatever"), `thinking_budget` without server support.
6. No correction, no translation, no use of the existing word
   confidences (`words[].score`, `avg_logprob`).

## 2. Architecture

```
                       ┌────────────────────────────────────────┐
                       │  core/llm/                              │
   Transcript          │                                        │
   (segments with      │  budget.py    Token estimation +       │
    words/score) ────► │               chunk budget from n_ctx  │
                       │  chunking.py  Segment-boundary chunker │
                       │  runner.py    Chunk loop (retry,       │
                       │               progress, cancellation)  │
                       │  correction.py Confidence edits         │
                       │  translation.py Chunk translation     │
                       │  prompts.py   all default prompts      │
                       └───────┬────────────────────────────────┘
                               │ chat()/chat_json()
                       ┌───────▼───────────────┐
                       │ core/llm_client.py    │  Retries, error classes,
                       │                       │  response_format JSON,
                       └───────┬───────────────┘  context auto-detection
                               │
                 Ollama native │  OpenAI-compatible (llama.cpp, vLLM, …)
                       /api/chat  /v1/chat/completions
```

* `core/summarize.py` uses the package for map-reduce, but remains the
  home of the summarization logic. The queue's automatic summary after a
  transcript is a task in the LLM queue (core/queue._queue_summary).
* API: `api/llm.py` (status/estimate/summarize/summaries) +
  correction/translation endpoints with background jobs
  (`job_manager`, `job_type="llm"` — runs without an open browser).
* Persistence: edits via the canonical dual-write
  `_save_transcript_data` (DB v16 + JSON sidecar + FTS), translations in
  a new `translations` table (modeled on `summaries`), suggestions in
  `llm_edits`.

## 3. Basic rules (apply to ALL tasks)

1. **The LLM never sees timestamps/IDs as editable data.** Segments go
   out numbered (`[id] text`), only `id`-related results come back.
   Times, speakers, word lists stay local.
2. **Structured responses are enforced** wherever the server can:
   `response_format: json_schema` (llama.cpp: verified live,
   grammar sampling), Ollama native: `format`. Fallback: strict prompt +
   JSON extractor + one repair retry.
3. **Chunking at segment boundaries**, never mid-segment. Budget =
   `n_ctx − system − instruction − response reserve − 10% margin`.
   `n_ctx` preferably comes **from the server** (llama.cpp `/props` →
   `n_ctx=32768`; `/v1/models` `meta.n_ctx`; Ollama `/api/show`),
   `llm.context_size` is now only an override.
4. **Sequential chunk loop** (one llama.cpp slot), progress per chunk
   reported to the job, cancellation at any time, failed chunks are
   retried once at the end; partial results are never lost.
5. **Edit conservatively**: correction only changes what the confidence
   or context clearly supports; the server validates every response (ID
   exists, length ratio, no duplicates) and discards outliers.
6. **Nothing gets overwritten without being asked**: corrections are
   only suggestions first (review diff in the UI), translations are
   their own track. The original stays restorable (llm_edits history).

## 4. Data formats

### 4.1 Chunk (internal)

```python
Chunk(
  index=3, from_seg=120, to_seg=178,     # inclusive, segment indices
  context_before=[(118, "…"), (119, "…")],# read-only context (overlap)
  tokens=6200,
)
```

### 4.2 Correction — request to LLM (user content, per chunk)

```
Segments (⟨⟩ marks uncertain words with confidence 0–1):
[120] SPEAKER_00: We're meeting on ⟨Wednesday|0.21⟩ at ten.
[121] SPEAKER_01: The ⟨Bugdet|0.35⟩ is approved.
```

Response (enforced via JSON schema):

```json
{"edits": [
  {"id": 121, "text": "The budget is approved.",
   "reason": "typo per context", "confidence": 0.9}
 ],
 "checked_ok": [120]}
```

Only changed segments; `checked_ok` lists the ones checked and left as they
are (see 5a). Server-side validation: `id` within the chunk
range, `0.3 ≤ len(new)/len(old) ≤ 3`, text not identical, `confidence ≥
llm.correction.min_confidence`.

### 4.3 Translation — response format

```json
{"translations": [
  {"id": 120, "text": "We're meeting on Wednesday at ten."},
  {"id": 121, "text": "The budget is approved."}
]}
```

Completeness is checked; missing IDs are requested again in a follow-up
request (only the gaps). Rolling context: the last 5 source+target pairs
of the preceding chunk travel along as context (names, register).

### 4.4 Persistence

* **Correction suggestions**: table `llm_edits`
  (`abs_path, status: pending|applied|rejected, payload_json, model,
  created_at`). On applying: replace `segments[i].text`, the segment's
  `words` are kept but get an `edited: 1` marker (word times count as
  approximate), old text goes into `payload_json` → undo possible.
  Afterward `_save_transcript_data` + re-export SRT (if
  `export_alongside` is active).
* **Translations**: table `translations`
  (`abs_path, rel_path, language, segments_json, model, created_at`),
  JSON backup `{name}_translations.json` in the transcript directory
  (same healing pattern as summaries). `segments_json` =
  `[{id, text}]` — timings always come from the original when
  rendering/exporting.

## 5. Tasks in detail

### 5.1 Summarization

* Fits everything in one prompt → **one** call (streaming).
* Otherwise **map-reduce**: notes per chunk (headed `Part N [from–to]`,
  each point with its timestamp), then reduce with the actual template
  over the notes. Reduce too large → recursive. Map progress streams as
  status events, reduce streams text as usual.
* The **user message** is metadata, transcript, and the additional
  instruction *last*: after tens of thousands of transcript tokens that is
  where the model follows it best, and a rerun with a different
  instruction leaves the transcript prefix — and the server's prompt
  cache — untouched.

### 5.2 Correction

* Selection: only segments with `avg_logprob < threshold` OR at least
  one word `score < threshold` (default), optionally "whole transcript".
* Chunks contain the selection **plus context** (neighboring segments
  readable unshortened, but marked as "not editable").
* Temperature fixed low (`llm.correction.temperature`, default 0.2).
* A chunk's input gets 1/2.2 of the context left after the system prompt and
  the safety margin: with many edits the answer (new text plus reason per
  segment) reaches the input's size. At most 80 segments a chunk.

### 5.3 Translation

* Always chunk-wise (even small transcripts — one chunk is the special
  case, same code path).
* A chunk's input gets 1/3.5 of the context left after the system prompt
  and the safety margin: the answer can be twice as long (expansion, e.g.
  ja→de) plus ~16 tokens of JSON per segment. At most 60 segments a chunk —
  with many short segments the JSON scaffolding is what costs.
* Multiple languages per file storable; re-translating replaces the
  track of the same language.

## 5a. Prompt catalog (core/llm/prompts.py)

All default prompts of the pipeline live in **one** file:
`whisper_tm/core/llm/prompts.py`. `tests/test_llm_prompts.py` pins
down the format contracts (schemas, placeholders, marker syntax) —
whoever changes a prompt notices immediately if they break the contract
with the code.

**Two classes of prompts, deliberately separated:**

* **User data** — the default summarization prompt and the 10 built-in
  templates (config key `llm.presets`). Each template is the complete
  system prompt; the summarize dialog shows it as it is sent, marks it
  "edited" once changed and can save it as a template of its own.
  `config.default.yaml` only mirrors them for new installations, and
  Settings → AI tasks → "⬇ Built-in templates" pulls an existing config up to
  the state in prompts.py without touching templates the user wrote.
* **Task prompts** (correction, translation, map/reduce) — their format
  part is fixed: the response formats (JSON schema, ID rules) must match
  the validating code exactly, and an "improved" prompt that loosens the
  format would silently cause all results to be discarded. The
  correction's *rules* (what is corrected, what stays untouched) can be
  replaced — `llm.correction.system_prompt`, Settings → AI tasks →
  Correction rules; the format part is appended to them in any case.

### The 10 built-in templates

Summary (= default prompt), short summary, bullet points, meeting
minutes, chapter marks, timeline, facts & figures, speaker analysis,
questions & answers, highlights. All share one shape: a
one-line task, what to produce, `Format:`, `Rules:`, and the same two
closing rules. Shared construction principles:

* **Anti-hallucination in every preset**: "only what was literally
  said", "don't add anything", name unclear things as unclear instead of
  smoothing them over. Whisper transcripts contain errors — an LLM that
  smooths them over hides them.
* **Specifics before topic lists**: numbers, names, results belong in
  there (e.g. "price rises to €12" instead of "talks about prices").
* **Fallback instruction instead of invention**: chapter markers without
  timestamps in the transcript → say so and deliver a topic list; no
  facts → one sentence, not pseudo-facts.
* **"Write in the language of the transcript"** as the last line
  everywhere — otherwise the model answers in its favorite language.
* **Timestamps as the transcript has them**: lines reach the model as
  `[H:MM:SS]` (or `[H:MM:SS | speaker]`); templates show that format in
  their examples and ask for it to be copied, never estimated. Templates
  that need timestamps or speaker labels say so in those words — the
  dialog points out when the matching source option is switched off.

### Map/reduce (long transcripts)

The map step does NOT carry out the template: its format would
reinforce itself across two stages (e.g. "max 20 bullet points" × 8
chunks = 160 bullet points in the reduce input). But it knows the
template as the task it collects for (`prompts.map_system(task)`) — a
neutral collection dropped exactly what special templates
look for: a joke is not "substantively relevant". The map step keeps
names, numbers and technical terms verbatim, each point's timestamp and
short quotes; the reduce step gets the actual template and the
instruction to treat the parts as ONE transcript and not mention their
origin.

### Correction — "corrector, not proofreader"

* **Input markup**: `⟨text|score⟩` marks uncertain passages (0–1), `⚠`
  after the segment number means "whole segment uncertain" (set when
  > 40% of the words are uncertain — otherwise the prompt explodes with
  CJK single-character scores, live finding: 47k → 10.7k characters).
  Context lines with `(id)` are readable but not editable.
* **Narrow allow-list, broad deny-list**: only phonetically plausible
  meaning errors, garbled proper names with a correct model in context,
  and unambiguous homophones are corrected. Explicitly left alone:
  spoken language (filler words, false starts, dialect), punctuation/
  grammar, and anything plausible — "low confidence is a hint, not
  proof".
* **`checked_ok` as an ID list**: the model WANTS to report what it
  checked (live finding: 95% of returns were unchanged
  confirmations). Without its own field it copies out whole sentences
  for this — a single chunk ran over 13 minutes this way. With the ID
  field, the same statement costs one number: 6.3 seconds. The prompt
  actively steers this ("explicitly wanted", "if you're unsure, the ID
  belongs under checked_ok") — the field is at the same time the valve
  against invented corrections born of a sense of obligation to deliver.
* **Few-shot in the prompt**: one example with all three cases (edit,
  ⚠ segment with no change, homophone) shows the exact output format.

### Translation — "the segmentation is given"

* **Core rule first**: every numbered line is a subtitle with a fixed
  display time — never merge, split, reorder, or omit. The example shows
  exactly the critical case: a sentence spanning two lines, whose
  translation breaks at the same point. This keeps the original timings
  valid (timings never leave the app, see basic rule 1).
* **Subtitle reality instead of written language**: length roughly like
  the original (readability), preserve register, replace interjections
  idiomatically, keep repetitions (Whisper hallucination loops must not
  be optimized away — otherwise the line mapping breaks).
* **Consistency across chunks**: the rolling context (last 5
  source/target pairs) is tied to the prompt hint "orient yourself on
  the provided context" — names and forms of address stay stable across
  chunk boundaries.

### Schemas

`CORRECTION_SCHEMA` and `TRANSLATION_SCHEMA` (also in prompts.py) are as
tight as possible with `additionalProperties: false` and required
fields — llama.cpp enforces them via grammar sampling, the fallback path
validates against them. The schemas are part of the contract: code,
prompt and schema only change together.

## 6. Configuration

```yaml
llm:
  endpoint: ""
  model: ""
  max_tokens: 4096        # response limit; > context gets clamped + warning
  context_size: 0         # 0 = detect AUTO from server (shown in settings)
  temperature: 0.3        # summary/free only
  timeout: 1800
  retries: 3              # network/5xx retries with backoff
  chunking:
    overlap_segments: 3   # reading context between chunks
    safety_margin: 0.10
  correction:
    score_threshold: 0.40
    logprob_threshold: -0.80
    temperature: 0.2
    min_confidence: 0.5
    max_edit_share: 0.5  # >50% edited segments per chunk → discard chunk
    system_prompt: ""    # own correction rules; empty = the built-in ones
  translation:
    temperature: 0.3
    context_pairs: 5
```

Worth knowing:

* `thinking`/`thinking_budget`: native for Ollama; in the OpenAI path the
  `thinking` object is sent whenever `llm.thinking` is on (servers that
  don't know it ignore it, or answer 400 depending on the version — leave
  it off for those). Correction/translation force thinking=off
  (deterministic, fast).
* `max_tokens` clamp: never larger than the detected context / 2.

## 7. Error handling

| Error | Behavior |
|---|---|
| Endpoint unreachable | error class `LlmConnectionError`, retry ×3 backoff, then job error with plain text |
| HTTP 4xx | no retry (config error), message with server text |
| HTTP 5xx / timeout | retry with backoff |
| Response not valid JSON | 1 repair retry with parser error in the prompt, then mark chunk as failed |
| IDs missing (translation) | targeted follow-up request for only the gaps |
| Edits violate validation | discard edit individually (logged), keep the rest |
| Chunk fails permanently | job continues; result names the failed chunks |

## 8. Operations: providers, queue, setup

### Providers (core/llm_provider.py)

`llm.providers` is a list; each entry has `id`, `name`, `kind`
(llamacpp | ollama | openai | vllm | custom), `endpoint`, `api_key`,
`models`, `default_model`. `llm.tasks.<summary|correction|
translation|chat>` = `{provider, model}` assigns individual tasks
specifically; unset means "default provider" (the first active one). The
embedding model of the semantic search has a section of its own
(`embedding`).

A config with the old single fields (`llm.endpoint`/`llm.model`) is read
as if it were a provider called "Standard" — existing installations
don't need to change anything. Only saving in the settings creates the
new structure; the legacy fields stay mirrored to the first provider so
older code keeps working.

API keys NEVER go to the UI (`has_key: true` instead of the value), and
an empty key field on save does not overwrite an existing key.

### Queue (core/llm_queue.py, table llm_queue)

One worker, strictly sequential — the llama.cpp server has one slot.
Tasks run server-side: a closed dialog or browser doesn't cancel
anything, and a server restart resets `running` entries back to
`pending` and picks them up again. `parts_json` collects intermediate
results (per map section: time range, segment range, token count,
duration, text), which the UI can expand while the run is still going.
A reopened dialog finds its job via `GET /api/llm/queue/open?kind=…&filename=…`.

`POST /api/llm/queue/folder` enqueues a whole folder (only files with a
finished transcript, recursive optional, `dry_run_only` first delivers a
preview, `skip_existing` skips already completed ones).

### Running the local server (section `llama`)

`start-llm.sh` calls `python -m whisper_tm.llm_server`; the start
command is built in `core/llama_server.start_command()` from the config.
Tuning thus needs no new setup. `--show` prints the command, `--check`
the status report (identical to the health card).

The defaults suit the model the setup downloads, **Qwen3.5-9B at Q4_K_M**
(5.7 GB), on an 8 GB card: a hybrid whose KV cache lives in every fourth
layer only (32 KB a token, 1 GB at the default 32k context), so weights,
cache and buffers take ~7 GB. `n_gpu_layers` is left empty — llama.cpp fits
the layers to the free VRAM, and a smaller card runs the rest on the CPU —
and the batch sizes are llama.cpp's own (2048/512).

The levers, measured on a **GTX 1660 Ti (6 GB, Turing without Tensor
Cores)** with a mixture-of-experts model (Qwen3.6-35B-A3B, Q4_K_S: prompt
476 t/s, response 26.5 t/s with the values below) — what to set when a
model doesn't fit the card:

| Value | Effect | Why |
|---|---|---|
| `threads` (0 = auto) | ×6.6 in response speed | More threads than physical cores make llama.cpp wait at its spin-wait barriers for preempted threads. `$(nproc)` is actively harmful on oversubscribed VMs — hence: physical cores, one fewer in a VM |
| `ubatch_size` 8192 | +37% in prompt ingestion | Weights on the CPU cross the PCIe bus once per ubatch pass: the volume depends on the NUMBER of passes, not on the tokens. Only go larger with VRAM headroom: at 12288 only 129 MiB remained, and the server hard-crashes at startup if it doesn't fit. A model wholly on the card gains little from it |
| `n_cpu_moe` 39 (with `n_gpu_layers` 99) | Makes the model possible at all | For MoE models, the experts are ~93% of the weights; every layer on the GPU costs ~450 MB and only gains ~2.5%. The same VRAM put into a larger ubatch gains more. 0 for dense models |

Not effective (measured, therefore not offered as an option): smaller
quantization (Q3 was 25% slower than Q4_K_S, because dequantization on
the CPU is more expensive), quantized KV cache (with MoE the cache is
tiny), `--no-mmap` (2% — the model sits in the page cache), `--poll 0`.

The health check (`GET /api/health/detail` → `llama`) checks against the
*actual* machine: thread oversubscription, VRAM needs against card size,
missing model, and whether the running server has the same context as
the config. On other hardware the defaults are a starting point — the
card says where they don't fit.

### Installing llama.cpp alongside

`./setup.sh --llama` downloads the matching release from ggml-org: on
Linux the **CUDA** build for NVIDIA (with the cudart package of the same
build next to it), the **Vulkan** build for AMD, otherwise the CPU variant.
The archive contents are copied with `cp -a` —
the libraries are partly symlinks, a `find -type f` would leave them behind
and `llama-server` would die at startup due to a missing `.so`. Afterward
`start-llm.sh` is created (port 8080, `--ctx-size 32768`, `--jinja`), and
the endpoint lands in the config if none is set there yet.

## 9. Verified end to end

Run against a local llama.cpp server (a 35B mixture-of-experts model, 32k
context): context auto-detection (`/props`), `response_format` with a JSON
schema (the server enforces the grammar), an empty system prompt kept
empty, short and map/reduce summaries (a 350k-character transcript in four
map chunks and a reduce), automatic summaries as a queue post-step,
correction with review, apply and undo, and long translations. What those
runs changed in the code:

* **CJK alignment scores are a weak signal**: single characters score 0.00,
  and the first Japanese correction prompt carried 2,000 markers (51k
  tokens against an estimate of 14k). Markers are compacted, a segment more
  than 40 % uncertain is marked as a whole, and special characters count as
  one token each.
* **A response capped at `max_tokens`** (JSON scaffolding costs ~16 tokens a
  segment, and ja→de expands) made the repair retry mirror the giant answer
  into a timeout cascade: `finish_reason` is read, the range is halved
  automatically, the reserves include the scaffolding, at most 60/80
  segments go into a chunk, and a task times out after 15 minutes.
* **Whisper hallucination segments** (a phrase twenty times) drive the model
  into generation loops: they are detected, logged and skipped — the
  halving cascade ends at single segments.

## 10. Sources

* Confidence-guided ASR correction: arxiv.org/abs/2509.25048,
  arxiv.org/html/2407.21414 (threshold filtering against over-correction),
  arxiv.org/html/2409.09785 (GenSEC baselines)
* Map-reduce/hierarchical: arxiv.org/pdf/2502.00977 (context-aware
  merging), galileo.ai/blog/llm-summarization-strategies (~10% overlap)
* Subtitle translation: github.com/machinewrapped/llm-subtrans
  (timeline stays local), rockbenben/subtitle-translator
* Structured output: llama.cpp grammar sampling via
  response_format/json_schema (verified live on a local llama.cpp server)

"""
LLM pipeline package (docs/LLM-PIPELINE.md).

Modules:
  budget.py        Token estimation + chunk budget from context size
  chunking.py      Segment boundary chunker with read context (overlap)
  runner.py        Chunk loop: progress, cancellation, partial error tolerance
  prompts.py       all default prompts of the pipeline (ONE source)
  correction.py     confidence-guided transcript correction
  translation.py  chunk-wise transcript translation

Core rules (apply package-wide):
  * The LLM sees segments only numbered ("[id] text") — timestamps,
    speaker assignment, and word lists remain local.
  * Responses are JSON-forced (llm_client.chat_json).
  * Chunking never in the middle of a segment; budget comes from the
    actual loaded server context size (llm_client.effective_context).
"""

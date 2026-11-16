"""Offline without the voice model's weights, a transcription keeps its
transcript: the weights' download failing is its own error, which the
pipeline turns into a warning (a voices-only run still fails)."""

import urllib.error

import pytest


def test_a_failed_download_is_named(tmp_path, monkeypatch):
    import torch

    from whisper_tm.core import speaker

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    def refuse(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(torch.hub, "download_url_to_file", refuse)
    with pytest.raises(speaker.VoiceWeightsUnavailable, match="--prefetch"):
        speaker._redimnet_weights()


def test_the_pipeline_keeps_the_transcript_only_for_that_error():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "whisper_tm/core/transcription.py").read_text(
        encoding="utf-8"
    )
    block = src[src.index("from whisper_tm.core.speaker import VoiceWeightsUnavailable") :][:600]
    assert "if not isinstance(e, VoiceWeightsUnavailable) or voices_only:" in block
    assert 'raise RuntimeError(f"Voice vectors failed: ' in block
    assert 'job_manager.warn(job, f"Voice vectors skipped: {e}")' in block

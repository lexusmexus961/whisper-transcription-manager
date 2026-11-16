"""One copy of the Whisper weights per model and card (core/transcription.py).

An English upload ("auto") put at the front of a running queue used to get
a pipeline keyed by language, so it loaded a second large-v3 beside the
German one the other lanes used — enough to run a mid-range card out of
memory. A language, a vocabulary or a decoding option is a
pipeline (tokenizer, options, a small VAD) around the weights now; the
weights themselves load once.

Nothing here touches a GPU: the weights and the pipeline are stand-ins that
record how they were built.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import transcription


class Weights:
    built: list = []

    def __init__(self, name, **kw):
        self.name, self.kw = name, kw
        Weights.built.append(self)


@pytest.fixture
def asr(monkeypatch):
    import whisperx
    import whisperx.asr

    Weights.built = []
    pipelines = []

    def load_model(name, device, **kw):
        p = {"name": name, "device": device, **kw}
        pipelines.append(p)
        return p

    monkeypatch.setattr(whisperx.asr, "WhisperModel", Weights)
    monkeypatch.setattr(whisperx, "load_model", load_model)
    monkeypatch.setattr(transcription, "_models", {})
    monkeypatch.setattr("whisper_tm.core.vram.low_vram", lambda: False)

    def get(language="de", vocab="", extra=None, name="large-v3", compute="float16"):
        return transcription._asr_pipeline(name, "cuda", 0, compute, language, vocab, extra or {})

    get.pipelines = pipelines
    return get


class TestOneCopy:
    def test_another_language_is_another_pipeline_on_the_same_weights(self, asr):
        de, auto = asr("de"), asr("auto")
        assert len(Weights.built) == 1
        assert de is not auto
        assert de["model"] is auto["model"] is Weights.built[0]
        assert (de["language"], auto["language"]) == ("de", None)

    def test_a_vocabulary_and_decoding_options_share_them_too(self, asr):
        plain = asr("de")
        vocab = asr("de", vocab="Alice, Bob, Carol", extra={"repetition_penalty": 1.1})
        assert len(Weights.built) == 1 and plain["model"] is vocab["model"]
        assert vocab["asr_options"] == {
            "initial_prompt": "Alice, Bob, Carol",
            "repetition_penalty": 1.1,
        }
        assert plain["asr_options"] is None

    def test_the_same_choice_is_the_same_pipeline(self, asr):
        assert asr("de") is asr("de")
        assert len(asr.pipelines) == 1 and len(Weights.built) == 1

    def test_another_model_or_precision_has_its_own_weights(self, asr):
        asr(name="large-v3"), asr(name="medium"), asr(compute="int8")
        assert [(w.name, w.kw["compute_type"]) for w in Weights.built] == [
            ("large-v3", "float16"),
            ("medium", "float16"),
            ("large-v3", "int8"),
        ]

    def test_the_weights_are_built_as_whisperx_builds_them(self, asr):
        asr(compute="default")
        w = Weights.built[0]
        assert w.name == "large-v3"
        assert w.kw == {
            "device": "cuda",
            "device_index": 0,
            "compute_type": "float16",
            "cpu_threads": 4,
        }


class TestWithoutTheCache:
    def test_a_low_vram_run_keeps_nothing(self, asr, monkeypatch):
        # hardware.low_vram: load fresh, and the run's own `del` frees it —
        # the pipeline holds the only reference to its weights.
        monkeypatch.setattr("whisper_tm.core.vram.low_vram", lambda: True)
        a, b = asr("de"), asr("de")
        assert len(Weights.built) == 2 and a["model"] is not b["model"]
        assert transcription._models == {}

    def test_unloading_takes_the_weights_with_it(self, asr, monkeypatch):
        from whisper_tm.core import diar_process

        monkeypatch.setattr(diar_process, "terminate_all", lambda reason: None)
        asr("de"), asr("auto")
        assert len(transcription._models) == 3  # the weights and two pipelines
        transcription.release_models()
        assert transcription._models == {}
        asr("de")
        assert len(Weights.built) == 2  # loaded anew after unloading

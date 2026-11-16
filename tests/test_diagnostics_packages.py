"""--check imports what a transcription really loads; a missing module is
named, not hidden behind whisperx's lazy wrappers."""

from whisper_tm.core import diagnostics


def test_a_broken_module_is_named(monkeypatch):
    monkeypatch.setattr(diagnostics, "_package_result", None)
    monkeypatch.setattr(
        diagnostics,
        "PIPELINE_MODULES",
        (("Speaker diarization (pyannote)", "no_such_module_here"), ("JSON", "json")),
    )
    rows = diagnostics._package_checks()
    assert rows[0]["status"] == diagnostics.FAIL and "no_such_module_here" in rows[0]["detail"]
    assert rows[1]["status"] == diagnostics.OK
    assert diagnostics._package_checks() is rows  # once per run


def test_a_card_only_the_driver_reports_is_listed(monkeypatch, tmp_path):
    """WSL2: no DRM node for the card, but NVML knows it."""
    from whisper_tm.core import debug

    monkeypatch.setattr(
        debug, "gpu_static", lambda: {0: {"name": "NVIDIA RTX A", "total_mb": 8192}}
    )
    real = diagnostics.Path

    def fake_path(p, *a):
        return real(tmp_path / "nothing") if str(p) == "/sys/class/drm" else real(p, *a)

    monkeypatch.setattr(diagnostics, "Path", fake_path)
    gpus = diagnostics.list_gpus()
    assert gpus == [{"vendor": "nvidia", "slot": "nvml:0", "name": "NVIDIA RTX A", "vram_mb": 8192}]


def test_whisper_on_amd_uses_the_cpu_without_a_hip_ctranslate2(monkeypatch):
    from whisper_tm import hardware

    prof = hardware.HardwareProfile()
    prof.best_device = "rocm"
    monkeypatch.setattr(hardware, "_ct2_gpu_count", lambda: 0)
    assert hardware.get_whisperx_device(prof) == "cpu"
    monkeypatch.setattr(hardware, "_ct2_gpu_count", lambda: 1)
    assert hardware.get_whisperx_device(prof) == "cuda"
    prof.best_device = "cuda"
    assert hardware.get_whisperx_device(prof) == "cuda"


def test_card_memory_is_rounded_not_floored():
    """A 4 GB card reports 3792 MiB and was listed as '3GB'."""
    from whisper_tm import hardware

    prof = hardware.HardwareProfile()
    prof.gpus = [hardware.GPUInfo(index=0, name="Card", vendor="nvidia", memory_total_mb=3792)]
    assert "Card (4GB)" in prof.summary()

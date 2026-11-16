"""Every command that opens a source file names the containers it may be
(-format_whitelist): a manifest named like a recording would otherwise
make ffmpeg open files or URLs of its choosing."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from whisper_tm.core import media_formats

ROOT = Path(__file__).resolve().parent.parent / "whisper_tm"


def test_the_media_service_uses_the_same_list():
    src = (ROOT / "media/service.py").read_text(encoding="utf-8")
    block = src[src.index("SOURCE_FORMATS = (") : src.index(")\n", src.index("SOURCE_FORMATS = ("))]
    assert "".join(re.findall(r'"([^"]*)"', block)) == media_formats.WHITELIST


def test_no_bare_input_on_a_source_open():
    """A plain "-i" is left only where the input is the app's own work file:
    listed here by file, so a new source open has to pass through the helper."""
    allowed_bare = {
        "core/transcription.py",
        "core/vocal_isolation.py",
        "core/speaker.py",
        "core/diar_silence.py",
        "core/clip_render.py",
        "core/voice_extract.py",
        "api/media.py",
    }
    for f in ROOT.rglob("*.py"):
        rel = f.relative_to(ROOT).as_posix()
        text = f.read_text(encoding="utf-8")
        if '"ffmpeg"' not in text or rel in allowed_bare or rel == "media/service.py":
            continue
        bare = [m.start() for m in re.finditer(r'"-i",\s*(str\(|[a-z_]+[,\]\)])', text)]
        assert not bare, f"{rel}: a plain -i — use media_formats.source()"


def test_a_manifest_named_like_a_recording_is_not_opened(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    other = tmp_path / "secret.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", str(other)], check=True
    )
    fake = tmp_path / "talk.mp4"
    fake.write_text(f"ffconcat version 1.0\nfile '{other}'\n", encoding="utf-8")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", *media_formats.source(fake), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    real = tmp_path / "real.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", str(real)], check=True
    )
    ok = subprocess.run(
        ["ffmpeg", "-v", "error", *media_formats.source(real), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr

"""Clip cutting must not fail reading the error message.

Cutting a clip of a video failed with
❌ 'utf-8' codec can't decode byte 0xff in position N: invalid start
byte.

The cut itself was fine. `subprocess.run(..., text=True)` decodes ffmpeg's
stderr STRICTLY as UTF-8, and this file's metadata contains a byte that
doesn't fit there. The exception was thrown READING THE MESSAGE — the page
showed a codec error where there wasn't one.

`run_cancellable` decodes with ``errors="replace"``, and according to its own
header, ALL external tools go through it anyway. Clip cutting was the last
place that bypassed it.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEDIA = (REPO / "whisper_tm" / "api" / "media.py").read_text(encoding="utf-8")


def _without_comments(text: str) -> str:
    """Just the code — the comment about the error names it anyway."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return "\n".join(lines)


def test_clip_cut_goes_through_safe_path():
    body = MEDIA[MEDIA.index("def extract_clip(") :]
    body = _without_comments(body[: body.index("\n@media_bp.route")])
    assert "run_cancellable(" in body, "clip cut bypasses the wrapper"
    assert "subprocess.run(" not in body, (
        "strict decoding again — exactly the error described above"
    )


def test_no_strict_decoding_left_in_file():
    """A single place is enough to bring the error back."""
    assert "text=True" not in _without_comments(MEDIA), (
        "text=True decodes strictly as UTF-8 — files with foreign metadata make the call explode"
    )


def test_run_cancellable_replaces_broken_bytes():
    """The property everything depends on — reproduced here."""
    from whisper_tm.core.audio_utils import run_cancellable

    r = run_cancellable(
        ["python3", "-c", "import sys; sys.stderr.buffer.write(b'A\\xffB')"],
        job=None,
        timeout=30,
        label="Test",
    )
    assert "A" in r.stderr and "B" in r.stderr, r.stderr

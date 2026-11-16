"""Listening samples are trimmed by age and size; your own clips stay."""

import os
import time

from whisper_tm.api import media


def test_old_and_too_many_samples_go_own_clips_stay(tmp_path, monkeypatch):
    old = tmp_path / "sample_a_1.mp3"
    old.write_bytes(b"x" * 10)
    t = time.time() - (media.SAMPLE_KEEP_DAYS + 1) * 86400
    os.utime(old, (t, t))
    fresh = tmp_path / "sample_b_2.mp3"
    fresh.write_bytes(b"x" * 10)
    mine = tmp_path / "my clip.mp3"
    mine.write_bytes(b"x" * 10)
    os.utime(mine, (t, t))
    assert media.trim_samples(tmp_path) == 1
    assert not old.exists() and fresh.exists() and mine.exists()
    monkeypatch.setattr(media, "SAMPLE_MAX_MB", 0)
    assert media.trim_samples(tmp_path) == 1 and not fresh.exists() and mine.exists()

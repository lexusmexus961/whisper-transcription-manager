"""Error text must name the next step, not just the state.

A download failed and the job only showed the raw yt-dlp error.
"""

from __future__ import annotations


class TestHintOnAnOutdatedYtDlp:
    """A download error that points in the wrong direction.

    The job showed: ``All downloads failed — ERROR: unable to download
    video data: HTTP Error 403: Forbidden``. That sends anyone first looking
    for network, permission or blocking issues.

    Actually yt-dlp was too old. Measured: the LATEST stable version got
    403; a nightly build downloaded the same file without trouble. The difference was the player client
    (``visionos`` + m3u8). The fallback attempts also fit: ``web``/``ios``/
    ``mweb`` reported "Requested format is not available", ``tv`` reported
    "The page needs to be reloaded".

    YouTube changes this more often than stable versions come out. So the hint
    belongs in the error message, not in a file nobody reads when something
    goes wrong.
    """

    def test_the_403_is_told_the_next_step(self):
        from whisper_tm.core.ytdl import hint_for_error

        t = hint_for_error("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        assert "403" in t, "the original message must not disappear"
        assert "--pre" in t and "yt-dlp" in t, t

    def test_the_other_masks_of_the_same_cause_too(self):
        from whisper_tm.core.ytdl import hint_for_error

        for raw in (
            "ERROR: [youtube] x: Requested format is not available",
            "ERROR: [youtube] x: The page needs to be reloaded.",
            "ERROR: Sign in to confirm you're not a bot",
        ):
            assert "--pre" in hint_for_error(raw), raw

    def test_a_genuinely_different_error_stays_untouched(self):
        """Otherwise the hint sticks to every message and becomes noise — even
        on ones where it points in the wrong direction."""
        from whisper_tm.core.ytdl import hint_for_error

        for raw in (
            "ERROR: Video unavailable. This video is private",
            "ERROR: [generic] Unable to download webpage: <urlopen "
            "error [Errno -3] Temporary failure in name resolution>",
            "",
        ):
            assert hint_for_error(raw) == raw

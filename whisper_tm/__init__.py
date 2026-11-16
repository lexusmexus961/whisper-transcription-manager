"""
Whisper Transcription Manager — Professional Transcription Server.
Cross-platform, GPU-accelerated, NAS-scale.
"""

import os as _os

__version__ = "0.9.0"
__author__ = "lexusmexus961"

# pyannote.audio reports every diarization (its duration, the number of
# speakers) to pyannote.ai unless told otherwise — off for a self-hosted
# server. PYANNOTE_METRICS_ENABLED=true in the environment allows it; the
# worker processes inherit the setting.
_os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

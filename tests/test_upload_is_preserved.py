"""
Uploads are library files — the pipeline must NEVER delete them.

Incident: videos freshly loaded via yt-dlp disappeared from the media
storage after transcription; the player delivered 404, the transcript remained as
an orphan. The cause was a remnant in the ``finally`` of ``transcription_worker``:
when uploads/ was still a staging area in the data directory, the
source file was "cleaned up" on success. Since the restructuring (uploads/ is IN
the media directory, DB-registered, transcript alongside), this cleanup was
data destruction.

An end-to-end test would need the GPU pipeline; instead this
ensures at source code level that ``transcription_worker`` no longer
touches the source. Whoever rebuilds deletion must get past this test
and its history.
"""

from __future__ import annotations

import ast
import inspect

from whisper_tm.core import transcription


def _unlink_targets(function) -> list[str]:
    """All ``X.unlink(...)`` calls in the function, as source text of X."""
    tree = ast.parse(inspect.getsource(function))
    targets = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "unlink"
        ):
            targets.append(ast.unparse(node.func.value))
    return targets


def test_worker_does_not_delete_source_file():
    targets = _unlink_targets(transcription.transcription_worker)
    assert "source_path" not in targets, (
        "transcription_worker deletes the source file again — exactly what once "
        "destroyed downloaded videos after transcription from the media storage "
        "(uploads/ is library, not staging area)."
    )

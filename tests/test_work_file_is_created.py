"""Every source gets a work file — including an upload on the same disk.

Found while pushing a file through the MCP server (upload → transcribe):
the job failed with

    File not readable (ffprobe Code 1):
    …/work/uploads_probe_wav_5358beba/sample.wav: No such file or directory

The work file simply didn't exist. ``transcription_worker`` provides it in
two ways, and both had excluded each other:

* ``if same_fs and not <source is in uploads/>`` → Symlink
* ``if not same_fs and not work_file.exists()``     → One-pass (copy + WAV)

For an **upload on the same disk**, the first condition was wrong (due to the
uploads exception) and the second also (due to ``same_fs``). Neither branch
ran, ``work_file`` remained just a name without a file.

The uploads exception came from a time when the ``finally`` block deleted
the upload source after the run — a symlink would have pointed to nothing.
This deletion was later removed (see ``test_upload_is_preserved.py``),
but the exception remained.

A setup with the library on a network share and the work directory in
``~/.cache`` never hits this — never the same filesystem. It affected every
standard single-disk installation, i.e., the normal case.

What is tested here is the **completeness of the case distinction**: both
conditions are extracted from the source and evaluated across all four
combinations of ``same_fs`` × "source is an upload". At least one must
trigger each time. An end-to-end run would need the GPU pipeline; this form
catches exactly the gap that occurred here.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from whisper_tm.core import transcription


def _conditions() -> dict[str, str]:
    """Guards of both provisioning branches, as source code.

    They are identified by what they DO (place symlink or start one-pass)
    — not by their wording. Anyone rewriting the condition will still
    pass through here.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(transcription.transcription_worker)))
    found: dict[str, str] = {}
    for nodes in ast.walk(tree):
        if not isinstance(nodes, ast.If):
            continue
        body = "\n".join(ast.unparse(k) for k in nodes.body)
        if "work_file.symlink_to" in body:
            found["symlink"] = ast.unparse(nodes.test)
        elif "source_audio.mka" in body:
            found["one_pass"] = ast.unparse(nodes.test)
    return found


def _takes_effect(condition: str, *, same_fs: bool, is_upload: bool) -> bool:
    """Evaluates a condition with given circumstances.

    ``work_file.exists()`` is always false at this point — the work file
    is just being created; and ``config.dirs['uploads']`` determines whether
    the source counts as an upload.
    """

    class _Path(str):
        def exists(self):
            return False

    source = "/media/uploads/sample.wav" if is_upload else "/media/archive/a.mkv"
    space = {
        "same_fs": same_fs,
        "work_file": _Path("/work/sample.wav"),
        "source_path": _Path(source),
        "str": str,
        "config": type("C", (), {"dirs": {"uploads": _Path("/media/uploads")}})(),
    }
    return bool(eval(condition, {"__builtins__": {}}, space))  # noqa: S307


def test_both_branches_are_discoverable():
    """If this test breaks, the next one has nothing to check."""
    b = _conditions()
    assert set(b) == {"symlink", "one_pass"}, b


@pytest.mark.parametrize("same_fs", [True, False])
@pytest.mark.parametrize("is_upload", [True, False])
def test_work_file_is_always_created(same_fs, is_upload):
    b = _conditions()
    branches = [
        name
        for name, cond in b.items()
        if _takes_effect(cond, same_fs=same_fs, is_upload=is_upload)
    ]
    assert branches, (
        f"For same_fs={same_fs}, Upload={is_upload}, NO branch applies — "
        f"the work file is never created and the next step fails with "
        f"'No such file or directory'. Conditions: {b}"
    )


def test_upload_on_same_disk_is_symlinked():
    """The concrete case from the incident — and via the efficient route.

    One-pass wouldn't be wrong, but an unnecessary second copy: source and
    work directory are on the same disk.
    """
    b = _conditions()
    assert _takes_effect(b["symlink"], same_fs=True, is_upload=True)


def test_uploads_exception_is_gone():
    """It had a reason that has since disappeared; bringing it back
    breaks every single-disk installation."""
    b = _conditions()
    assert "uploads" not in b["symlink"], (
        "The uploads exception is back in the symlink branch. It used to protect "
        "against the source being deleted in finally — that no longer "
        "happens (test_upload_is_preserved.py). Bringing it back only brings "
        "the missing-work-file error back."
    )

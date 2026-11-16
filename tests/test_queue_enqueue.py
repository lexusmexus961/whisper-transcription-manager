"""
Enqueuing must not hang on the file system with large quantities.

Background: "transcribe all" over a six-figure number of files let
``POST /api/queue/add`` run for **minutes** — the interface showed
nothing, the call hung on the request.

Two causes, both measured:

* ``root_allowed(target)`` ran per file and calls ``Path.resolve()`` — on
  a NAS a network access per file — minutes for a six-figure number of files.
  The check was redundant: the base is checked beforehand, and
  ``paths.abs_from`` returns ``""`` as soon as the result leaves it.
* An ``INSERT`` per file instead of an ``executemany``.

Additionally the response contained all enqueued entries — with that many that would be
tens of MB of JSON, which the frontend doesn't evaluate at all.
"""

from __future__ import annotations

import time

import pytest

from whisper_tm.core import queue as q


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import whisper_tm.extensions as ext
    from whisper_tm.config import config
    from whisper_tm.database import Database

    media = tmp_path / "media"
    media.mkdir()
    db = Database(tmp_path / "q.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "media_on_start", property(lambda self: media))
    q._current = None
    q._abort = False
    q._paused = False
    q._solo.clear()
    return {"media": media, "db": db}


def test_enqueueing_does_not_touch_the_filesystem_per_file(environment, monkeypatch):
    """The actual regression protection: no resolve() per file.

    Instead of a time measurement (which would be green even with the error
    on a fast disk), count how many times ``Path.resolve`` is called.
    """
    from pathlib import Path

    counter = {"n": 0}
    real = Path.resolve

    def counted(self, *a, **k):
        counter["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", counted)

    files = [f"folder{i % 50}/file{i}.mp4" for i in range(2000)]
    added = q.enqueue(files, {}, str(environment["media"]))

    assert len(added) == 2000
    # A few calls for the root check are okay — but not
    # one per file. Before the fix there were over 4000.
    assert counter["n"] < 100, (
        f"{counter['n']} resolve() calls for 2000 files — the check is running per file again"
    )


def test_enqueueing_stays_fast(environment):
    """Rough time limit as a second safety net."""
    files = [f"folder{i % 50}/file{i}.mp4" for i in range(5000)]
    t = time.perf_counter()
    added = q.enqueue(files, {}, str(environment["media"]))
    duration = time.perf_counter() - t
    assert len(added) == 5000
    assert duration < 5.0, f"{duration:.1f}s for 5000 files"


def test_ids_are_consecutive_and_correct(environment):
    """executemany must not corrupt the returned IDs.

    ``/api/transcribe`` reads ``added[0]["id"]`` and returns it to the frontend
    as ``itemId``.
    """
    added = q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(environment["media"]))
    assert [e["filename"] for e in added] == ["a.mp4", "b.mp4", "c.mp4"]

    with environment["db"].connection() as conn:
        real = {
            r["id"]: r["filename"] for r in conn.execute("SELECT id, filename FROM queue_items")
        }
    for e in added:
        assert real[e["id"]] == e["filename"], "ID points to the wrong row"


def test_positions_are_consecutive(environment):
    added = q.enqueue([f"d{i}.mp4" for i in range(10)], {}, str(environment["media"]))
    assert [e["position"] for e in added] == list(range(1, 11))


def test_a_second_call_appends_at_the_end(environment):
    q.enqueue(["a.mp4"], {}, str(environment["media"]))
    second = q.enqueue(["b.mp4"], {}, str(environment["media"]))
    assert second[0]["position"] == 2
    with environment["db"].connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0]
    assert n == 2


def test_duplicates_are_still_skipped(environment):
    q.enqueue(["a.mp4", "b.mp4"], {}, str(environment["media"]))
    again = q.enqueue(["a.mp4", "c.mp4"], {}, str(environment["media"]))
    assert [e["filename"] for e in again] == ["c.mp4"]


def test_a_path_outside_the_root_is_refused(environment):
    """The security check must not be lost through the acceleration."""
    added = q.enqueue(
        ["../../etc/shadow", "/etc/shadow", "good.mp4"], {}, str(environment["media"])
    )
    assert [e["filename"] for e in added] == ["good.mp4"]


def test_a_foreign_root_is_refused(environment):
    added = q.enqueue(["shadow"], {}, "/etc")
    assert added == []


def test_enqueueing_at_the_front_comes_before_the_pending_ones(environment):
    """Upload/Link-Download enqueues at the FRONT (enqueue(front=True)).

    Behind a run over the whole library, a freshly uploaded file would otherwise
    get its turn in weeks. The worker pulls ORDER BY position, id — so
    the front entries must come before all pending positions
    (negative is allowed).
    """
    q.enqueue([f"old_value{i}.mp4" for i in range(5)], {}, str(environment["media"]))
    new_value = q.enqueue(["upload.mp4"], {}, str(environment["media"]), front=True)
    assert len(new_value) == 1

    with environment["db"].connection() as conn:
        order = [
            r["filename"]
            for r in conn.execute(
                "SELECT filename FROM queue_items WHERE status = 'pending' ORDER BY position, id"
            )
        ]
    assert order[0] == "upload.mp4"


def test_several_at_the_front_keep_their_order(environment):
    q.enqueue(["old_value.mp4"], {}, str(environment["media"]))
    q.enqueue(["p1.mp4", "p2.mp4", "p3.mp4"], {}, str(environment["media"]), front=True)
    with environment["db"].connection() as conn:
        order = [
            r["filename"]
            for r in conn.execute(
                "SELECT filename FROM queue_items WHERE status = 'pending' ORDER BY position, id"
            )
        ]
    assert order == ["p1.mp4", "p2.mp4", "p3.mp4", "old_value.mp4"]


# ── An upload runs next ─────────────────────────────────────────────
#
# The mechanism for this (`front=True`) already existed: Upload, Link-Download, and
# Re-transcription enqueue this way. But it had a hole — if the same
# file was already `pending` in the queue, the duplicate check
# silently rejected the call and the file stayed where it was. The promise thus depended
# on something the user cannot see: whether the file happened to be in
# a large run already.


def _pending(environment, n=3):
    with environment["db"].connection() as conn:
        return [
            r["filename"]
            for r in conn.execute(
                "SELECT filename FROM queue_items WHERE status='pending' "
                "ORDER BY position, id LIMIT ?",
                (n,),
            )
        ]


def test_an_upload_comes_next_while_a_run_is_going(environment):
    """The main case: big run started, then an upload comes in."""
    q.enqueue([f"large{i}.mp4" for i in range(50)], {}, str(environment["media"]))
    with environment["db"].connection() as conn:
        conn.execute("UPDATE queue_items SET status='processing' WHERE position=1")

    q.enqueue(["uploads/new_value.mp4"], {}, str(environment["media"]), front=True)

    assert _pending(environment)[0] == "uploads/new_value.mp4"


def test_a_pending_file_is_moved_up_instead_of_discarded(environment):
    """The actual hole: the file is already at the back of the queue."""
    q.enqueue([f"large{i}.mp4" for i in range(20)], {}, str(environment["media"]))
    q.enqueue(["uploads/already.mp4"], {}, str(environment["media"]))  # at the end

    added = q.enqueue(["uploads/already.mp4"], {}, str(environment["media"]), front=True)

    assert _pending(environment)[0] == "uploads/already.mp4"
    # It is reported, not silently "skipped" — otherwise
    # the interface reports "nothing enqueued", even though something happened.
    assert len(added) == 1 and added[0]["moved_up"] is True
    # And it must not appear twice in the queue.
    with environment["db"].connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM queue_items WHERE filename='uploads/already.mp4'"
        ).fetchone()["n"]
    assert n == 1


def test_the_running_file_is_left_untouched(environment):
    """An already RUNNING file stays running.

    Moving it to the front would mean canceling a model to get to a place
    we're already at — pure waste.
    """
    q.enqueue(["running.mp4", "b.mp4", "c.mp4"], {}, str(environment["media"]))
    with environment["db"].connection() as conn:
        conn.execute("UPDATE queue_items SET status='processing' WHERE filename='running.mp4'")

    added = q.enqueue(["running.mp4"], {}, str(environment["media"]), front=True)

    assert added == []
    with environment["db"].connection() as conn:
        r = conn.execute(
            "SELECT status, position FROM queue_items WHERE filename='running.mp4'"
        ).fetchone()
    assert r["status"] == "processing" and r["position"] == 1


def test_moving_up_and_a_new_file_in_the_same_call(environment):
    """Mixed: one is waiting, one is new. Both to the front, in the
    order of the request."""
    q.enqueue([f"large{i}.mp4" for i in range(10)], {}, str(environment["media"]))
    q.enqueue(["old_value.mp4"], {}, str(environment["media"]))

    q.enqueue(["old_value.mp4", "new_value.mp4"], {}, str(environment["media"]), front=True)

    assert _pending(environment, 2) == ["old_value.mp4", "new_value.mp4"]


def test_without_front_the_duplicate_stays_put(environment):
    """Counter-check: Normal enqueuing must NOT reorder.

    Otherwise an accidental second "transcribe all" would scramble the whole
    sequence.
    """
    q.enqueue([f"large{i}.mp4" for i in range(10)], {}, str(environment["media"]))
    before = _pending(environment, 10)

    added = q.enqueue(["large5.mp4"], {}, str(environment["media"]))

    assert added == []
    assert _pending(environment, 10) == before


def test_a_file_moved_up_keeps_its_id(environment):
    """The id must remain stable — /api/transcribe tracks the job
    through it, and a cancel button in the interface points to it."""
    q.enqueue(["a.mp4", "b.mp4"], {}, str(environment["media"]))
    with environment["db"].connection() as conn:
        before = conn.execute("SELECT id FROM queue_items WHERE filename='b.mp4'").fetchone()["id"]

    added = q.enqueue(["b.mp4"], {}, str(environment["media"]), front=True)

    assert added[0]["id"] == before

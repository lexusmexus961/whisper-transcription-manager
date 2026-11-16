"""One GPU, several queue lanes: which file may use which model, and when.

The queue runs up to ``queue.lanes`` files at once (core/queue.py). Every
model is still loaded once, and a model serves one file at a time — that is
a *stage*: separation, Whisper, alignment, diarization, voice vectors. Two
different stages may run together, so one file's separation can go while
another file waits for its diarization, or is still being copied from a
network share.

Measured on a 12 GB card: running two GPU stages together
gains nothing on the card itself — separation and diarization at the same
time each ran at half their own speed (sum of shares 1.00), and the
separator is just as fast at batch 1 as at batch 4. What lanes win is the
time the card stood idle while a file was copied, decoded, padded, written
and booked: a batch of files finished about a fifth sooner, a batch of short
ones almost twice as fast.

Rules:

* ``stage(name, need_mb)`` — one thread per name. A stage that would run
  next to another of ours only starts when NVML shows ``need_mb`` free;
  alone it always starts, as before lanes. A thread already inside a stage
  may enter another one (voice vectors separate their windows) and the
  same one again.
* ``exclusive(reason)`` — waits until no stage runs and holds every new one
  off (a waiting exclusive goes first). For what touches all models at
  once: the low-VRAM run, unloading, the cleanup after an out-of-memory.
  Its owner's own stages pass. Asking for it from inside a stage would wait
  for itself — that raises instead.
* ``try_exclusive()`` — the idle unloader: only when nothing runs right now.

Waiting at the gate is part of the run: a file whose job is cancelled — the
✕ on its row, a stop, a shutdown — leaves the wait with InterruptedError
instead of first taking its turn on a model it no longer needs (seen in
practice: a cancelled file waited for the separator and would then have
separated a 30-minute chunk). And its progress says what it waits for.

With one lane every stage finds the gate open, and nothing changes.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

logger = logging.getLogger(__name__)

#: What a stage needs on top of what is already loaded, in MB — checked
#: only while another stage of ours runs. Measured: the separator's peak
#: is 2.2 GB with its 0.87 GB of weights; the diarization worker keeps
#: ~2.3 GB and grows to its limit (hardware.diarization_vram_cap_mb) at
#: most; alignment and voice vectors are small models.
NEED_MB = {
    "separation": 1800,
    "alignment": 1500,
    "diarization": 1500,
    "vectors": 800,
}

#: Stages that never run next to another one: Whisper alone takes 5.6–6.8
#: GB at batch 16 (core/vram.py), more than a 12-GB card has left with the
#: other models resident.
EXCLUSIVE_STAGES = {"asr"}

#: What a waiting file's progress names.
STAGE_NAMES = {
    "separation": "the separator",
    "asr": "Whisper",
    "alignment": "the alignment model",
    "diarization": "the diarization",
    "vectors": "the voice-vector model",
}


def _job_hooks(job, name, cancel, on_wait):
    """The cancel check, the waiting note, and taking the note back.

    The note goes once the wait is over: it used to stay up while the
    file was long on the separator, and three running files all said they
    were waiting.
    """
    on_done = None
    if job is None:
        return cancel, on_wait, on_done
    if cancel is None:

        def cancel():
            return bool(getattr(job, "cancel", False))

    if on_wait is None:
        what = STAGE_NAMES.get(name)
        text = (
            f"⏳ Waiting for {what} — another file is using it"
            if what
            else "⏳ Waiting for the other files to leave the GPU"
        )
        before = []

        def on_wait():
            before.append(getattr(job, "message", ""))
            job.update(message=text)

        def on_done():
            if before and getattr(job, "message", None) == text:
                job.update(message=before[0])

    return cancel, on_wait, on_done


def _call_quietly(fn) -> None:
    if fn is None:
        return
    try:
        fn()
    except Exception as e:  # a progress note must never stop a run
        logger.debug("GPU gate wait note: %s", e)


class GpuGate:
    def __init__(self, free_mb=None):
        self._cv = threading.Condition()
        self._stages: dict[str, int] = {}  # stage name → thread ident
        self._held: dict[int, list[str]] = {}  # thread ident → its stages
        self._owner: int | None = None  # exclusive holder
        self._owner_depth = 0
        self._owner_reason = ""
        self._writers_waiting = 0
        # Threads waiting to enter a stage — with the exclusive waiters what
        # `waiting()` reports: the reranker's server holds a stage for as
        # long as it runs and gives the card back when someone waits for it
        # (core/rerank_server.py).
        self._stage_waiting = 0
        self._free_mb = free_mb  # injectable for tests
        self._local = threading.local()

    # ── Free memory ──────────────────────────────────────────────────

    def _free(self) -> int | None:
        if self._free_mb is not None:
            return self._free_mb()
        from whisper_tm.core import vram

        return vram.free_mb()

    # ── Stages ───────────────────────────────────────────────────────

    def _may_enter(self, me: int, name: str, need_mb: int) -> bool:
        if self._owner is not None and self._owner != me:
            return False
        holder = self._stages.get(name)
        if holder is not None and holder != me:
            return False
        # A waiting exclusive goes first — except for a thread that already
        # holds a stage: it must finish, or the exclusive waits forever.
        if self._writers_waiting and me not in self._held:
            return False
        others = any(t != me for t in self._stages.values())
        if others and need_mb > 0:
            free = self._free()
            if free is not None and free < need_mb:
                return False
        return True

    @contextlib.contextmanager
    def stage(self, name: str, need_mb: int | None = None, *, job=None, cancel=None, on_wait=None):
        """Use the model behind ``name`` for one file.

        ``job`` (or ``cancel``/``on_wait`` directly): a cancelled job leaves
        the wait with InterruptedError; ``on_wait`` runs once when the
        stage has to wait at all.
        """
        cancel, on_wait, on_done = _job_hooks(job, name, cancel, on_wait)
        if name in EXCLUSIVE_STAGES:
            with self.exclusive(name, cancel=cancel, on_wait=on_wait, on_done=on_done):
                yield
            return
        need = NEED_MB.get(name, 0) if need_mb is None else need_mb
        me = threading.get_ident()
        t0 = time.monotonic()
        with self._cv:
            passed = self._owner == me or self._stages.get(name) == me
            entered = passed or self._may_enter(me, name, need)
            if entered and not passed:
                self._enter(me, name)
        if not entered:
            # The note outside the lock: it takes the job's own lock.
            _call_quietly(on_wait)
            with self._cv:
                self._stage_waiting += 1
                try:
                    while not self._may_enter(me, name, need):
                        if cancel is not None and cancel():
                            raise InterruptedError("Cancelled")
                        self._cv.wait(timeout=1.0)
                finally:
                    self._stage_waiting -= 1
                self._enter(me, name)
            _call_quietly(on_done)
        self._add_wait(time.monotonic() - t0)
        try:
            yield
        finally:
            if not passed:
                with self._cv:
                    self._stages.pop(name, None)
                    mine = self._held.get(me, [])
                    if name in mine:
                        mine.remove(name)
                    if not mine:
                        self._held.pop(me, None)
                    self._cv.notify_all()

    def _enter(self, me: int, name: str) -> None:
        """Register a stage (caller holds _cv)."""
        self._stages[name] = me
        self._held.setdefault(me, []).append(name)

    # ── Exclusive ────────────────────────────────────────────────────

    @contextlib.contextmanager
    def exclusive(self, reason: str = "", *, job=None, cancel=None, on_wait=None, on_done=None):
        """All of the GPU for one thread: no stage runs while it is held.

        ``job``/``cancel``/``on_wait`` as for :meth:`stage`.
        """
        cancel, on_wait, done = _job_hooks(job, None, cancel, on_wait)
        on_done = on_done or done
        me = threading.get_ident()
        t0 = time.monotonic()
        with self._cv:
            if self._owner == me:
                self._owner_depth += 1
                nested = True
            else:
                nested = False
                if me in self._held:
                    raise RuntimeError(
                        f"exclusive GPU access ({reason}) requested inside the "
                        f"GPU stage {self._held[me]} — it would wait for itself"
                    )
                free = self._owner is None and not self._stages
                if free:
                    self._take(me, reason)
        if not nested and not free:
            _call_quietly(on_wait)
            with self._cv:
                self._writers_waiting += 1
                try:
                    while self._owner is not None or self._stages:
                        if cancel is not None and cancel():
                            raise InterruptedError("Cancelled")
                        self._cv.wait(timeout=1.0)
                finally:
                    self._writers_waiting -= 1
                    # Stages held off by this writer may go now
                    self._cv.notify_all()
                self._take(me, reason)
            _call_quietly(on_done)
        waited = time.monotonic() - t0
        if not nested:
            self._add_wait(waited)
            if waited > 1.0:
                logger.info("GPU exclusive for %s after %.1f s", reason or "?", waited)
        try:
            yield
        finally:
            with self._cv:
                self._owner_depth -= 1
                if self._owner_depth <= 0:
                    self._owner = None
                    self._owner_depth = 0
                    self._owner_reason = ""
                    self._cv.notify_all()

    def _take(self, me: int, reason: str) -> None:
        """Become the exclusive owner (caller holds _cv)."""
        self._owner = me
        self._owner_depth = 1
        self._owner_reason = reason

    @contextlib.contextmanager
    def try_exclusive(self, reason: str = ""):
        """Exclusive only if nothing runs right now — yields True or False."""
        me = threading.get_ident()
        with self._cv:
            got = self._owner is None and not self._stages and not self._writers_waiting
            if got:
                self._owner = me
                self._owner_depth = 1
                self._owner_reason = reason
        try:
            yield got
        finally:
            if got:
                with self._cv:
                    self._owner = None
                    self._owner_depth = 0
                    self._owner_reason = ""
                    self._cv.notify_all()

    # ── Looking at it ────────────────────────────────────────────────

    def holds_stage(self) -> bool:
        """Is the calling thread inside a stage?"""
        with self._cv:
            return threading.get_ident() in self._held

    def owns(self) -> bool:
        """Does the calling thread hold the GPU exclusively?"""
        with self._cv:
            return self._owner == threading.get_ident()

    def may_take_exclusive(self) -> bool:
        """Could the calling thread ask for exclusive without waiting for
        itself? (It owns it already, or holds no stage.)"""
        with self._cv:
            me = threading.get_ident()
            return self._owner == me or me not in self._held

    def waiting(self) -> int:
        """How many threads wait at the gate right now — for a stage or for
        the GPU to themselves."""
        with self._cv:
            return self._writers_waiting + self._stage_waiting

    def state(self) -> dict:
        with self._cv:
            return {
                "stages": dict(self._stages),
                "exclusive": self._owner_reason if self._owner else None,
                "waiting_exclusive": self._writers_waiting,
                "waiting_stages": self._stage_waiting,
            }

    def _add_wait(self, secs: float) -> None:
        self._local.waited = getattr(self._local, "waited", 0.0) + secs
        self._local.total = getattr(self._local, "total", 0.0) + secs

    def wait_total(self) -> float:
        """Seconds the calling thread ever waited at the gate — for timing
        a stretch without its waits (take the difference), without taking
        them from the step that books them."""
        return getattr(self._local, "total", 0.0)

    def take_wait(self) -> float:
        """Seconds the calling thread spent waiting at the gate since the
        last call — the pipeline books its steps without them."""
        secs = getattr(self._local, "waited", 0.0)
        self._local.waited = 0.0
        return secs


GATE = GpuGate()

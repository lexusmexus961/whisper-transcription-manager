"""The parent side: maintain a diarization worker, feed it, kill it if necessary.

The counterpart to :mod:`whisper_tm.core.diar_worker`. All the logic here
keeps the child process alive and — that is the purpose of the exercise —
reliably gets rid of it:

* **Cancellation:** ``job.cancel`` is checked every 0.5 s while waiting for the response.
  Set → ``kill()``, ``InterruptedError``.
* **Time limit:** exceeded → ``kill()``, ``TimeoutError``. This was exactly what was
  impossible before: pyannote ran without limit, and a hanging call in the
  same process cannot be saved.
* **Worker death:** is detected (EOF on stdout or ``poll()``), the
  next request starts a fresh one.

**Why no watchdog thread?** Because Python cannot kill a thread.
A watchdog could only *report* the hang, not end it — and the CUDA
context would remain occupied. The process is the only unit that the
operating system cleans up completely.

**Why long-lived?** Measured a fresh process costs up to the loaded
model 4.96 s. With 100,000 files that would be ~140 h additional. So: start
once, serve many requests. It is only terminated on cancellation,
timeout, in ``low_vram`` mode after each request — and by the
idle cleaner via :func:`whisper_tm.core.transcription.release_models`.

The result comes back as a list ``[(start, end, speaker), …]``. The
caller builds from it the DataFrame that whisperx expects (see
:func:`as_dataframe`).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

logger = logging.getLogger("whisper_tm.core.diar_process")

# How long can the worker take to start? It's loaded after ~5 s;
# the generous limit covers the first run, when the model still comes from
# the network. Here too `job.cancel` is checked — there's no time
# where you get stuck.
START_DEADLINE_S = 600.0


def _vram_cap_mb() -> int:
    """``hardware.diarization_vram_cap_mb`` — the worker's VRAM limit, 0 = none.

    Why a limit at all is explained in ``diar_worker._set_vram_cap``.
    """
    try:
        from whisper_tm.config import config

        return max(0, int(float(config.get("hardware", "diarization_vram_cap_mb", default=4096))))
    except (TypeError, ValueError):
        return 4096


def as_dataframe(segments):
    """Build from ``[(start, end, speaker), …]`` the DataFrame that whisperx wants.

    Three consumers access the result, and all three need
    exclusively ``start``/``end``/``speaker``:

    * ``whisperx.assign_word_speakers`` — iterates with ``iterrows`` exactly over
      these three columns;
    * ``postprocessing.overlapping_intervals`` — reads ``["start"]`` and
      ``["end"]``;
    * ``speaker.normalize_turns`` — handles ``iterrows``.

    The columns ``segment`` and ``label`` that pyannote additionally provides
    are not read by anyone — they are therefore not carried across the process boundary.
    """
    import pandas as pd

    return pd.DataFrame(
        [{"start": s, "end": e, "speaker": sp} for s, e, sp in segments],
        columns=["start", "end", "speaker"],
    )


class Worker:
    """A long-lived child process with loaded diarization model."""

    def __init__(
        self,
        device: str = "cpu",
        hf_token: str | None = None,
        offline: bool = False,
        command: list | None = None,
    ):
        self.device = device
        self.hf_token = hf_token or ""
        self.offline = offline
        # Injectible so tests can start a mock worker
        # (hanging, dying, chatty) — without GPU and without pyannote.
        self.command = command or [sys.executable, "-m", "whisper_tm.core.diar_worker", device]
        self._proc: subprocess.Popen | None = None
        self._answers: queue.Queue = queue.Queue()
        self._error_lines: deque = deque(maxlen=40)
        self._lock = threading.Lock()
        # One request at a time, start included. The answers carry no
        # request id: two lanes waiting on the same answer queue would take
        # each other's segments — and their voices would be stored under
        # the wrong file without any error. And a start's "ready" must not
        # be read by a request waiting next to it (seen with two
        # lanes: "Failed to load audio" as a start error).
        self._request_lock = threading.RLock()

    # ── Life and death ────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _reader(self, proc: subprocess.Popen, answers: queue.Queue | None = None) -> None:
        """Queue stdout line by line — including EOF marker.

        Without the EOF marker (``None``) the waiter would sit until the time limit,
        even though the worker is long dead.

        The queue is the one of THIS process, handed over at the start: a
        reader of a killed worker that looked ``self._answers`` up only
        when it got its last line could put that line — or its EOF — into
        the queue of the next worker.
        """
        answers = answers if answers is not None else self._answers
        try:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    answers.put(line)
        except Exception as e:
            logger.debug("Diar-Worker, reader ended: %s", e)
        finally:
            answers.put(None)

    def _error_reader(self, proc: subprocess.Popen) -> None:
        """Clean up stderr — otherwise the worker blocks on a full pipe buffer.

        Same trap as in ``run_cancellable``: if the OS buffer (~64 KB)
        is full, the writer hangs. pyannote and torch log abundantly, so
        this would not be a theoretical case.
        """
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    self._error_lines.append(line)
                    logger.debug("diar-worker: %s", line)
        except Exception:
            pass

    def _last_errors(self, n: int = 5) -> str:
        return " · ".join(list(self._error_lines)[-n:])

    def start_it(self, job=None) -> None:
        """Boot the process and wait until the model is ready."""
        with self._request_lock:
            self._start_it(job)

    def _start_it(self, job=None) -> None:
        with self._lock:
            if self.alive:
                return

            environment = dict(os.environ)
            # The token travels via the environment, not via the
            # command line — which appears in every process list.
            if self.hf_token:
                environment["HF_TOKEN"] = self.hf_token
            else:
                environment.pop("HF_TOKEN", None)
            if self.offline:
                # In the child process the env var is the right way:
                # huggingface_hub is only imported AFTER startup there and
                # then reads the constant from the environment. (In the running
                # server that's too late — there `_HfOfflineIfCached`
                # redirects the constant directly.)
                environment["HF_HUB_OFFLINE"] = "1"
            # Unbuffered stdout in the child: otherwise the response comes
            # only when the buffer is full or the process ends.
            environment["PYTHONUNBUFFERED"] = "1"
            # Applied by the worker before it loads the model; a changed
            # setting takes effect with the next worker start
            cap = _vram_cap_mb()
            if cap:
                environment["WTM_DIAR_VRAM_CAP_MB"] = str(cap)
            else:
                environment.pop("WTM_DIAR_VRAM_CAP_MB", None)

            t0 = time.time()
            self._answers = queue.Queue()
            self._error_lines.clear()
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                text=True,
                bufsize=1,
            )
            threading.Thread(
                target=self._reader,
                args=(self._proc, self._answers),
                daemon=True,
                name="diar-_reader",
            ).start()
            threading.Thread(
                target=self._error_reader,
                args=(self._proc,),
                daemon=True,
                name="diar-_error_reader",
            ).start()

        answer = self._wait(START_DEADLINE_S, job=job, what="Model startup")
        if answer.get("kind") != "ready":
            error = answer.get("error") or "unexpected response"
            self.terminate("Startup failed")
            raise RuntimeError(f"Diarization worker did not start: {error}")
        cap = answer.get("vram_cap_mb")
        logger.info(
            "Diarization worker ready (%s, %.1f s, PID %s%s)",
            self.device,
            time.time() - t0,
            self._proc.pid if self._proc else "?",
            f", VRAM limit {cap} MiB" if cap else "",
        )

    def terminate(self, reason: str = "") -> None:
        """Hard terminate. The only way to get rid of a hang.

        No ``terminate()`` with follow-up like in ``run_cancellable``: A
        hanging pyannote doesn't respond to SIGTERM, and politeness toward
        a process already considered lost just costs time — the
        CUDA context is cleaned up by the operating system anyway.
        """
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            logger.info(
                "Diarization worker terminating (PID %s%s)",
                proc.pid,
                f", {reason}" if reason else "",
            )
            try:
                proc.kill()
            except Exception as e:
                logger.debug("kill(): %s", e)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    # ── Wait with cancellation ───────────────────────────────────────────────

    def _wait(self, deadline_s: float, job=None, what: str = "Response") -> dict:
        """Wait for a protocol line — cancellable and with time limit."""
        end = time.time() + deadline_s
        while True:
            try:
                line = self._answers.get(timeout=0.5)
            except queue.Empty:
                # The model IS in use — even if this process
                # is just waiting. Without this message the
                # idle cleaner considers the worker idle and kills it;
                # its path there goes through `release_models`, which
                # includes the child process.
                #
                # Exactly so a long file can die — a recording of
                # many hours whose diarization takes longer than
                # the 10-minute limit:
                #
                #   Diarization worker terminating (Free models)
                #   ❌ Error: Diarization worker terminated unexpectedly
                #
                # The message costs nothing (a timestamp every half
                # second) and is the same one that `_resident_model` gives for
                # the models in its own process.
                try:
                    from whisper_tm.core.model_idle import report_usage

                    report_usage()
                except Exception:
                    pass
                if job is not None and getattr(job, "cancel", False):
                    self.terminate("Cancelled by user")
                    raise InterruptedError("Cancelled") from None
                if time.time() > end:
                    self.terminate("Timeout")
                    raise TimeoutError(
                        f"Diarization: {what} after {deadline_s:.0f}s without response "
                        f"— worker terminated. {self._last_errors()}"
                    ) from None
                continue
            if line is None:  # EOF: worker is dead
                self.terminate("Worker died")
                raise RuntimeError(
                    f"Diarization worker unexpectedly terminated. {self._last_errors()}"
                ) from None
            try:
                return json.loads(line)
            except Exception:
                # No JSON: not part of the protocol. Don't die on it —
                # log it and wait again.
                logger.debug("diar-worker (no protocol): %s", line[:200])
                continue

    def _rss_gb(self) -> float:
        """How much memory does the child process currently hold?"""
        if not self.alive:
            return 0.0
        try:
            # psutil, not /proc: Windows has none, and the limit was off there
            import psutil

            return psutil.Process(self._proc.pid).memory_info().rss / 2**30
        except Exception:  # noqa: BLE001 — gone in between, or not allowed
            return 0.0

    def _cap_memory(self) -> None:
        """Restart the worker if it holds too much memory.

        **Why at all.** The process is long-lived so the model doesn't have to be
        reloaded per file (4.96 s). But precisely because of this it accumulates:
        after several recordings of many hours it stood at **over 10 GiB**
        and stayed there while the machine had only a few GiB left.

        It's not the audio — numpy returns large arrays via ``mmap`` immediately
        to the operating system (measured: 4.78 → 0.03 GiB after ``del``).
        It's the cached tensors from pyannote and torch, and
        no library gives those up voluntarily; ``malloc_trim`` also
        didn't help.

        So the honest solution: **throw away the process as needed.** A
        restart costs 5 s, an OOM killer in the middle of a very long file
        costs an hour of computation time — and may hit the server instead of
        the worker.

        The limit is in ``hardware.diarization_worker_max_gb``
        (default 6.0). 0 turns the cap off.
        """
        try:
            from whisper_tm.config import config

            limit = float(config.get("hardware", "diarization_worker_max_gb", default=6.0))
        except (TypeError, ValueError):
            limit = 6.0
        if limit <= 0:
            return
        rss = self._rss_gb()
        if rss < limit:
            return
        logger.info(
            "Diarization worker holds %.2f GiB (limit %.1f) — "
            "will be restarted after this request; that costs "
            "one-time ~5 s",
            rss,
            limit,
        )
        self.terminate(f"Memory limit {limit:.1f} GiB")

    # ── The actual request ──────────────────────────────────────────

    def diarize_it(
        self, wav, min_speakers=None, max_speakers=None, job=None, deadline_s: float = 3600.0
    ) -> list:
        """Diarize a file. Return: ``[(start, end, speaker), …]``."""
        with self._request_lock:
            return self._diarize_it(wav, min_speakers, max_speakers, job=job, deadline_s=deadline_s)

    def _diarize_it(
        self, wav, min_speakers=None, max_speakers=None, job=None, deadline_s: float = 3600.0
    ) -> list:
        if not self.alive:
            self._start_it(job=job)

        request = {"kind": "diarize", "wav": str(wav)}
        if min_speakers:
            request["min_speakers"] = int(min_speakers)
        if max_speakers:
            request["max_speakers"] = int(max_speakers)

        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except Exception as e:
            self.terminate("Request not deliverable")
            raise RuntimeError(f"Diarization worker not reachable: {e}") from e

        answer = self._wait(deadline_s, job=job, what="Diarization")
        self._cap_memory()
        if not answer.get("ok"):
            # A technical error (corrupt file) does not terminate the worker
            # — the model remains loaded and the next request may
            # succeed.
            trace = answer.get("trace", "")
            if trace:
                logger.debug("Diarization failed:\n%s", trace)
            raise RuntimeError(f"Diarization failed: {answer.get('error', 'unknown')}")
        if answer.get("uncapped"):
            logger.info(
                "Diarization needed more than the worker's VRAM limit "
                "(hardware.diarization_vram_cap_mb) and ran once "
                "without it: %s",
                wav,
            )
        return [(float(s), float(e), str(sp)) for s, e, sp in answer.get("segments", [])]


# ── One worker per device ───────────────────────────────────────────
#
# Like `_models` in transcription.py: a process cache so the model
# stays loaded between files. The key contains device AND
# offline status — not the token (it only decides about the download).

_worker: dict = {}
_worker_lock = threading.Lock()


def take(
    device: str, hf_token: str | None = None, offline: bool = False, command: list | None = None
) -> Worker:
    key = (device, bool(offline))
    with _worker_lock:
        a = _worker.get(key)
        # Never started (`_proc is None`) is as good as "alive" —
        # the first request starts it. Only a DEAD one gets replaced.
        died = a is not None and a._proc is not None and not a.alive
        if a is None or died:
            a = Worker(device, hf_token=hf_token, offline=offline, command=command)
            _worker[key] = a
        return a


def terminate_all(reason: str = "") -> None:
    """Kill all workers — idle cleaner, tests, server end."""
    with _worker_lock:
        worker = list(_worker.values())
        _worker.clear()
    for a in worker:
        a.terminate(reason or "Free models")


def worker_pids() -> list[int]:
    """PIDs of the running workers — the performance view names them in the
    VRAM breakdown (core/vram_procs.py)."""
    with _worker_lock:
        return [a._proc.pid for a in _worker.values() if a._proc is not None and a.alive]


def something_running() -> bool:
    """Does a worker currently hold VRAM? (for the idle cleaner)"""
    with _worker_lock:
        return any(a.alive for a in _worker.values())

"""Model work off the control thread: one inference thread, a priority queue.

A local model call can take half an hour on a 16 GB Mac (a Skill written by
``qwen3.5:4b`` while ordinary apps are open).  Before this module the
worker's loop made those calls itself, so for that long it wrote no
``worker-status.json`` heartbeat, read no trigger (an approval in the app
waited for the whole call) and processed no events.  Now the worker has two
threads:

* the **control thread** (``main()``'s loop): triggers from the app and the
  CLI, the stale focus-session sweep, the event pipeline, the heartbeat and
  the Skills index, every tick.  It makes no model call; it only submits
  model work here and reads :meth:`InferenceRunner.status`;
* the **inference thread** (:class:`InferenceRunner`): every model call, one
  job at a time, highest priority first:

  - :data:`P0`: work the user waits for: a stopped focus recording (its
    keyframes, its Skill, the post-recording questions and the answers),
    a retry or merge the app asked for, and the model part of an approval
    (the procedure's embedding and style analysis, :func:`model_work`);
  - :data:`P1`: the focus recording's tail (the keyframes its Skill did
    not need, ``FocusProcessor.annotate_remaining``);
  - :data:`P2`: passive work (fallback VLM jobs, passive annotation and
    frame diffs, then passive discovery and the daily jobs).

  A job that is queued or running is not queued again (jobs are keyed), and
  a job that raised waits :func:`~agenthandover_worker.loop_guard.error_backoff_seconds`
  before it is taken again.  One thread means one model call at a time, as
  :mod:`~agenthandover_worker.model_scheduler` assumes; when the queue runs
  empty, ``on_idle`` (the scheduler's ``end_generation``) unloads the
  generation model.

The two threads share the worker's state (the database handle, the
knowledge base, the in-memory queues and counters).  :data:`STATE_LOCK`
keeps that state as consistent as it was with one thread: whichever thread
runs Python holds it, and the inference thread lets go of it only while it
waits for a model (:func:`model_wait`, around every Ollama request and the
post-recording questions subprocess).  So the control thread runs between
model calls, never in the middle of a read-modify-write, and code that
reads a record, calls a model and writes the record back re-reads it after
the call (the approval could have changed it meanwhile).
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

from agenthandover_worker.loop_guard import error_backoff_seconds

logger = logging.getLogger(__name__)

P0, P1, P2 = 0, 1, 2
PRIORITY_NAMES = {P0: "P0", P1: "P1", P2: "P2"}

# Distinct jobs that can wait at once.  Periodic jobs are keyed (at most one
# of each queued), so the queue fills only with deferred approvals, retries
# and merges; past this, a submission is refused and logged.
MAX_QUEUED = 64


class StateLock:
    """The lock both threads hold while they run Python on shared state.

    Reentrant for its holder, and :meth:`released` lets go of it (at any
    depth) for the duration of a model wait.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._depth = 0

    def acquire(self) -> None:
        me = threading.get_ident()
        if self._owner == me:
            self._depth += 1
            return
        self._lock.acquire()
        self._owner = me
        self._depth = 1

    def release(self) -> None:
        if self._owner != threading.get_ident():
            raise RuntimeError("StateLock released by a thread that does not hold it")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()

    def held(self) -> bool:
        """True when the calling thread holds the lock."""
        return self._owner == threading.get_ident()

    def __enter__(self) -> "StateLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    @contextmanager
    def released(self) -> Iterator[None]:
        """Let go of the lock for the block if the calling thread holds it."""
        if not self.held():
            yield
            return
        depth = self._depth
        self._depth = 0
        self._owner = None
        self._lock.release()
        try:
            yield
        finally:
            self._lock.acquire()
            self._owner = threading.get_ident()
            self._depth = depth


STATE_LOCK = StateLock()

# The runner that is running (one per worker process), for model_work and
# model_wait; None in tests, the CLI and the journey driver, where model
# work runs inline as before.
_ACTIVE: "InferenceRunner | None" = None


@contextmanager
def model_wait(what: str = "") -> Iterator[None]:
    """Wrap a blocking wait for a model: the state lock is free meanwhile.

    A wait on the control thread is a bug (the heartbeat stops for as long
    as the model takes): it is logged, and runs.
    """
    runner = _ACTIVE
    if runner is not None and runner.on_control_thread():
        logger.warning(
            "Model call on the control thread (%s): it holds up the heartbeat "
            "and the triggers; it belongs on the inference thread (model_work)",
            what or "?",
        )
    with STATE_LOCK.released():
        yield


def model_work(key: str, fn: Callable[[], None], priority: int = P0) -> bool:
    """Run *fn*, which calls a model, on the inference thread.

    On the inference thread, or with no runner (tests, the CLI), it runs
    now; elsewhere it is queued as job *key* and runs after the job in
    progress.  Returns False when it could not be queued (logged).
    """
    runner = _ACTIVE
    if runner is None or runner.on_inference_thread():
        fn()
        return True
    if runner.submit(key, priority, fn):
        return True
    logger.warning("Model work %s not queued (queue full or backing off)", key)
    return False


@dataclass(order=True)
class _Item:
    priority: int
    seq: int
    key: str = field(compare=False)
    fn: Callable[[], None] | None = field(compare=False)


class InferenceRunner:
    """One inference thread working through a bounded priority queue."""

    def __init__(
        self,
        *,
        on_idle: Callable[[], None] | None = None,
        max_queued: int = MAX_QUEUED,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_idle = on_idle
        self._clock = clock
        self._queue: queue.PriorityQueue[_Item] = queue.PriorityQueue(max_queued)
        self._seq = itertools.count()
        self._cond = threading.Condition()
        # Keys queued or running.
        self._keys: set[str] = set()
        self._running: tuple[str, int, float] | None = None
        self._failures: dict[str, int] = {}
        self._not_before: dict[str, float] = {}
        self._completed = 0
        self._failed = 0
        self._thread: threading.Thread | None = None
        self._control_ident: int | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> "InferenceRunner":
        """Start the inference thread; the calling thread is the control
        thread from now on."""
        global _ACTIVE
        self._control_ident = threading.get_ident()
        self._thread = threading.Thread(
            target=self._work, name="inference", daemon=True,
        )
        self._thread.start()
        _ACTIVE = self
        return self

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop taking jobs and wait up to *timeout* for the one in progress.

        A model call cannot be interrupted: after the timeout the thread is
        left to finish (it is a daemon thread, so it does not hold up the
        process's exit).  Returns True when the thread ended.
        """
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        if self._thread is None:
            return True
        # Waiting jobs are dropped (the next worker start finds their work
        # again); the stop marker then goes into the emptied queue.
        with self._cond:
            with self._queue.mutex:
                self._queue.queue.clear()
                self._queue.unfinished_tasks = 0
            self._keys = {self._running[0]} if self._running else set()
            self._cond.notify_all()
        self._queue.put(_Item(-1, next(self._seq), "", None))
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- control-thread side -----------------------------------------------

    def submit(self, key: str, priority: int, fn: Callable[[], None]) -> bool:
        """Queue *fn* as job *key* unless it is queued or running already,
        backing off after a failure, or the queue is full.  True when *key*
        is queued or running after the call."""
        with self._cond:
            if key in self._keys:
                return True
            if self._clock() < self._not_before.get(key, 0.0):
                return False
            try:
                self._queue.put_nowait(_Item(priority, next(self._seq), key, fn))
            except queue.Full:
                return False
            self._keys.add(key)
            return True

    def busy(self, key: str) -> bool:
        """True while job *key* is queued or running."""
        with self._cond:
            return key in self._keys

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running; True unless timed out."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._keys, timeout)

    def consecutive_failures(self) -> int:
        """The longest run of failures of any job still failing."""
        with self._cond:
            return max(self._failures.values(), default=0)

    def status(self) -> dict:
        """The ``inference`` block of ``worker-status.json``: the job running
        (and for how long), the jobs waiting in the order they will run,
        and how many ran and failed since the worker started."""
        now = self._clock()
        with self._queue.mutex:
            waiting = sorted(i for i in self._queue.queue if i.fn is not None)
        with self._cond:
            running = self._running
            status = {
                "running": None,
                "running_seconds": None,
                "queued": [f"{PRIORITY_NAMES.get(i.priority, i.priority)} {i.key}" for i in waiting],
                "depth": len(waiting) + (1 if running else 0),
                "completed": self._completed,
                "failed": self._failed,
                "retry_after_seconds": {
                    key: round(until - now, 1)
                    for key, until in self._not_before.items() if until > now
                },
            }
        if running is not None:
            key, priority, started = running
            status["running"] = f"{PRIORITY_NAMES.get(priority, priority)} {key}"
            status["running_seconds"] = round(now - started, 1)
        return status

    def on_control_thread(self) -> bool:
        return threading.get_ident() == self._control_ident

    def on_inference_thread(self) -> bool:
        return self._thread is not None and threading.get_ident() == self._thread.ident

    # -- inference-thread side ---------------------------------------------

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item.fn is None:
                return
            with self._cond:
                self._running = (item.key, item.priority, self._clock())
            try:
                with STATE_LOCK:
                    item.fn()
            except Exception:
                logger.exception("Inference job %s failed", item.key)
                with self._cond:
                    self._failed += 1
                    n = self._failures.get(item.key, 0) + 1
                    self._failures[item.key] = n
                    self._not_before[item.key] = self._clock() + error_backoff_seconds(n)
            else:
                with self._cond:
                    self._completed += 1
                    self._failures.pop(item.key, None)
                    self._not_before.pop(item.key, None)
            if self._queue.empty() and self._on_idle is not None:
                try:
                    with STATE_LOCK:
                        self._on_idle()
                except Exception:
                    logger.exception("Inference idle hook failed")
            with self._cond:
                self._running = None
                self._keys.discard(item.key)
                self._cond.notify_all()

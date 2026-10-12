"""The worker loop's stages and its tick.

One :func:`tick` is one poll iteration of ``main()``'s loop, on the control
thread, in this order:

1. under ``STATE_LOCK``: :func:`control.run` (app and CLI triggers, the
   stale focus-session sweep), :func:`ingest.run` (the event pipeline),
   :func:`control.submit_model_jobs` (the model work, handed to the
   inference thread), :func:`control.recognize` (live recognition,
   model-free) and :func:`export.run` (the budget's persisted state and the
   Skills index);
2. outside it, :func:`export.heartbeat` (``worker-status.json``).

The model jobs run on the inference thread (``inference_runner``), one at a
time, highest priority first: P0 :func:`synthesize.run` (a stopped focus
recording, or its answered questions), P1 :func:`annotate.run_focus_tail`,
P2 :func:`memory.run` (Memory's embeddings, while ``[memory] enabled``),
P2 :func:`annotate.run` (passive annotation, frame diffs) and P2 the
generation job (:func:`segment.run`, passive discovery, then
:func:`daily.run`, the once-per-day pass).  Control stages never call
a model; model stages run only as those jobs.

Every stage is ``run(ctx, tick)``: it reads and updates the
:class:`~agenthandover_worker.context.WorkerContext` and records what it
did on the :class:`Tick`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agenthandover_worker.inference_runner import STATE_LOCK
from agenthandover_worker.loop_guard import ERROR_THRESHOLD, error_backoff_seconds

logger = logging.getLogger("agenthandover_worker")


@dataclass
class Tick:
    """What one tick saw and did, and (once they ran) what its model jobs did."""

    triggers_consumed: int = 0
    unprocessed: int = 0
    # Events handed to run_pipeline and not yet marked processed; an error
    # while this is non-empty counts against them.
    inflight_event_ids: list[str] = field(default_factory=list)
    pipeline: dict | None = None
    # (key, priority, job) of the model jobs submitted, by priority.
    jobs: list[tuple[str, int, Callable[[], None]]] = field(default_factory=list)
    focus_sops: int = 0
    annotations: dict | None = None
    memory: dict | None = None
    diffs: dict | None = None
    passive_sops: int | None = None
    # The generation phase's plan and the local day it was made for.
    generation_plan: Any = None
    local_day: str = ""
    # The state lock was busy (the inference thread holds it): only the
    # heartbeat was written.
    skipped: bool = False
    # The exception that ended the control tick, and the backoff before
    # the next one.
    error: Exception | None = None
    backoff: float | None = None


def tick(ctx: Any, *, lock_timeout: float | None) -> Tick:
    """One control tick (see the module docstring).

    Waits at most *lock_timeout* seconds (None: as long as it takes) for
    the state the inference thread holds while it runs Python; after that
    the tick only writes the heartbeat.  An exception in the control tick
    is logged, counted and backed off from (``Tick.error``,
    ``Tick.backoff``); the worker stays up.
    """
    from agenthandover_worker.stages import control, export, ingest

    t = Tick()
    if not STATE_LOCK.acquire(timeout=lock_timeout):
        logger.debug("Control tick skipped: the inference thread holds the state")
        t.skipped = True
        export.heartbeat(ctx)
        return t
    try:
        control.run(ctx, t)
        ingest.run(ctx, t)
        control.submit_model_jobs(ctx, t)
        control.recognize(ctx, t)
        export.run(ctx, t)
        # Any tick that gets here without raising is healthy, whether or
        # not it had events to process.
        ctx.consecutive_errors = 0
    except Exception as exc:
        t.error = exc
        _count_error(ctx, t)
    finally:
        STATE_LOCK.release()
    export.heartbeat(ctx)
    return t


def _count_error(ctx: Any, t: Tick) -> None:
    ctx.consecutive_errors += 1
    logger.error(
        "Error in main loop (consecutive: %d)", ctx.consecutive_errors, exc_info=t.error,
    )
    to_quarantine = ctx.event_guard.failed(t.inflight_event_ids)
    if to_quarantine:
        ctx.db.quarantine_events(to_quarantine)
        logger.error(
            "Quarantined event %s: the pipeline failed on it alone repeatedly",
            ", ".join(to_quarantine),
        )
    if ctx.consecutive_errors == ERROR_THRESHOLD:
        logger.critical(
            "%d consecutive errors; backing off (up to 5 minutes between "
            "attempts) and staying up",
            ERROR_THRESHOLD,
        )
    t.backoff = error_backoff_seconds(ctx.consecutive_errors)

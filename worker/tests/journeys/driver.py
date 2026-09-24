"""Journey driver: the worker's own tick, one ``run_tick()`` per poll
iteration, on a controllable clock.

``JourneyWorker`` builds the worker as ``main()`` does
(:func:`agenthandover_worker.context.build_context` over one
``load_config()`` read, so ``config.toml`` in the test's ``HOME`` decides
models, hosts and feature flags, then :func:`~agenthandover_worker.context.start`)
and ``run_tick()`` runs the production tick,
:func:`agenthandover_worker.stages.tick`: triggers, the stale focus
sweep, the event pipeline, the model jobs, live recognition, the budget's
persisted state, the Skills index and the heartbeat (see the ``stages``
package docstring for the order).  Journeys therefore test the loop's own
code; what differs from ``main()`` is only what the driver passes in:

* the clock: the context's ``now`` is the journey :class:`Clock`, its
  local day is the UTC day, its ``monotonic`` timers (passive discovery
  every 2 hours, live recognition every 2 minutes, the idle log) and the
  model scheduler's clock run on it;
* the model scheduler sees memory to spare (unless a journey passes
  ``free_memory``), no swap in use and disk to spare, and the idle gate is
  on AC power and never inside the idle window: the host's real
  ``vm_stat``, swap, disk, power and idle time would make a replay depend
  on the machine running it;
* no service is started: no fallback VLM worker (its detection probes the
  local Ollama), no query API server, no startup embedding job;
* the once-per-day pass (``stages.daily``) runs only when the journey
  asks for it (``run_daily_jobs=True``, as ``main()`` always runs it):
  on the first tick of each day of the journey clock it processes the day
  before, and its model calls (session links, embeddings) must be in the
  journey's cassette;
* without :meth:`JourneyWorker.start_inference` there is no inference
  thread: ``run_tick()`` runs the tick's model jobs itself after the
  control tick, in the runner's priority order and under the state lock
  as the runner does, then the runner's idle hook
  (``ModelScheduler.end_generation``) and the heartbeat again.  After
  :meth:`JourneyWorker.start_inference` the jobs go to a real inference
  thread and ``run_tick()`` is the control tick alone.

Building a ``JourneyWorker`` resets the process-wide usage counters, as a
worker start does.  ``main()`` logs an exception inside a tick and carries
on with the next one (shrinking or quarantining the event batch); the
driver re-raises it so the journey fails at its cause.  A failing trigger
handler is logged and skipped by ``trigger_spool.drain`` itself, in both.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from agenthandover_worker import main as wm
from agenthandover_worker import ops_telemetry, stages
from agenthandover_worker.config import load_config
from agenthandover_worker.context import WorkerContext, build_context, start
from agenthandover_worker.inference_runner import STATE_LOCK, InferenceRunner
from agenthandover_worker.model_scheduler import GB
from agenthandover_worker.scheduler import GateResult
from agenthandover_worker.stages import export, segment

PASSIVE_DISCOVERY_INTERVAL = timedelta(seconds=segment.PASSIVE_DISCOVERY_INTERVAL_SECONDS)

# The adapters journeys drive.
ADAPTERS = ("openclaw", "claude-skill", "generic")


class Clock:
    """A ``now()`` that only moves when the test moves it (aware UTC)."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("Clock needs a timezone-aware start")
        self._now = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def timestamp(self) -> float:
        return self._now.timestamp()

    def advance(self, delta: timedelta | None = None, **kwargs: float) -> datetime:
        """Move forward by ``delta`` or ``timedelta(**kwargs)``; returns the new now."""
        step = delta if delta is not None else timedelta(**kwargs)
        if step < timedelta(0):
            raise ValueError("the clock does not go backwards")
        self._now += step
        return self._now


class OnACGate:
    """The idle gate of a journey: on AC power, so passive annotation runs,
    and never inside the idle window, so neither the idle shortcut to
    passive discovery nor the fallback VLM jobs run."""

    def check(self) -> GateResult:
        return GateResult(can_run=False, blockers=["journey_no_idle_window"], conditions=None)

    def passive_annotation_allowed(self) -> tuple[bool, str]:
        return True, "on_ac_power"


class JourneyWorker:
    """The worker's context, built as ``main()`` builds it, and its tick.

    The context's components and state read through as attributes
    (``journey.db``, ``journey.knowledge_base``, ``journey.v2_schema_ok``).
    """

    def __init__(
        self,
        *,
        db_path: Path,
        knowledge_dir: Path,
        sops_dir: Path,
        clock: Clock,
        adapter: str = "openclaw",
        free_memory: Callable[[], int | None] = lambda: 64 * GB,
        run_daily_jobs: bool = False,
    ) -> None:
        if adapter not in ADAPTERS:
            raise ValueError(f"adapter {adapter!r} is not driven by journeys")
        self.clock = clock
        # A worker is one process: its usage counters start at zero.
        ops_telemetry.MODEL_USAGE.reset()
        ops_telemetry.COMPUTE.reset()
        self.ctx: WorkerContext = build_context(
            load_config(),
            db_path=Path(db_path),
            knowledge_dir=Path(knowledge_dir),
            sops_dir=Path(sops_dir),
            adapter=adapter,
            now=clock.now,
            local_tz=timezone.utc,
            monotonic=clock.timestamp,
            idle_gate=OnACGate(),
            scheduler_probes={
                "clock": clock.timestamp,
                "free_memory": free_memory,
                "swap_used": lambda: 0,
                "free_disk": lambda: 100 * GB,
            },
        )
        self.ctx.run_daily_jobs = run_daily_jobs
        start(self.ctx)

    def __getattr__(self, name: str) -> Any:
        if name == "ctx":
            raise AttributeError(name)
        return getattr(self.ctx, name)

    @property
    def state_dir(self) -> Path:
        """The worker's data directory (``main._status_dir()``)."""
        return wm._status_dir()

    def close(self) -> None:
        if self.ctx.runner is not None:
            self.ctx.runner.stop(timeout=60)
            self.ctx.runner = None
        self.ctx.close()

    def __enter__(self) -> "JourneyWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start_inference(self) -> InferenceRunner:
        """Run model work on an inference thread, as ``main()`` does: from
        now on :meth:`run_tick` is the control thread's tick only (it
        submits the jobs and returns; the runner takes them in priority
        order).  :meth:`close` stops it."""
        self.ctx.runner = InferenceRunner(
            on_idle=self.ctx.model_scheduler.end_generation,
        ).start()
        return self.ctx.runner

    def run_tick(self) -> dict:
        """One poll iteration: ``stages.tick`` and, without an inference
        thread, the model jobs it submitted (see the module docstring).

        Returns what the iteration did, for assertions.  With an inference
        thread the model jobs are only submitted, so their entries stay at
        their defaults.
        """
        tick = stages.tick(self.ctx, lock_timeout=None)
        if tick.error is not None:
            raise tick.error
        if self.ctx.runner is None:
            # The runner's order (by priority, then submission), then its
            # idle hook: the end of the generation phase.
            for _key, _priority, job in sorted(tick.jobs, key=lambda j: j[1]):
                with STATE_LOCK:
                    job()
            with STATE_LOCK:
                self.ctx.model_scheduler.end_generation()
            export.heartbeat(self.ctx)
        return {
            "triggers_consumed": tick.triggers_consumed,
            "focus_sops": tick.focus_sops,
            "pipeline": tick.pipeline,
            "annotations": tick.annotations,
            "diffs": tick.diffs,
            "passive_sops": tick.passive_sops,
            "generation_plan": tick.generation_plan,
        }

    def status(self) -> dict:
        """The last ``worker-status.json`` :meth:`run_tick` wrote."""
        return json.loads((self.state_dir / "worker-status.json").read_text())

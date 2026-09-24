"""Segment stage (inference thread, P2): passive discovery.

The first half of the passive generation job: it plans the generation
phase for the jobs due (passive discovery and the once-per-day pass,
``ModelScheduler.plan_generation``) and, when the plan allows it, segments
the annotated frames into tasks and writes Skills from them
(``main._process_passive_discovery``), charged to the annotation budget.
:func:`agenthandover_worker.stages.daily.run` follows with the plan.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agenthandover_worker import main as wm
from agenthandover_worker.scheduler import charged_to

logger = logging.getLogger("agenthandover_worker")

# Passive discovery runs every 2 hours ...
PASSIVE_DISCOVERY_INTERVAL_SECONDS = 7200
# ... or, while the user is away and the idle gate allows it, 5 minutes
# after the last run.
PASSIVE_DISCOVERY_IDLE_SECONDS = 300


def daily_due(ctx: Any, local_day: str) -> bool:
    """The once-per-day pass has not run on *local_day* yet."""
    return ctx.run_daily_jobs and local_day != ctx.daily_jobs_date


def run(ctx: Any, tick: Any) -> int | None:
    """Plan the generation phase, then run passive discovery when it is
    due and allowed.  Returns the Skills it generated (None when it did not
    run); the plan and the local day it was made for are recorded on
    *tick*."""
    now_mono = ctx.monotonic()
    should_segment = _passive_discovery_due(ctx, now_mono)
    tick.local_day = ctx.local_now().date().isoformat()
    plan = ctx.model_scheduler.plan_generation(
        urgent=False,
        due={
            "passive_discovery": should_segment,
            "daily": daily_due(ctx, tick.local_day),
        },
    )
    tick.generation_plan = plan
    if not (should_segment and plan.allows("passive_discovery")):
        return None

    try:
        with charged_to(ctx.annotation_budget):
            pd_sops = wm._process_passive_discovery(
                ctx.db,
                knowledge_base=ctx.knowledge_base,
                segmenter=ctx.task_segmenter,
                sop_generator=ctx.sop_generator,
                sop_auto_approve=ctx.cfg.sop.auto_approve,
                continuity_tracker=ctx.continuity_tracker,
                variant_detector=ctx.variant_detector,
                evidence_normalizer=ctx.evidence_normalizer,
                now=ctx.now(),
                weekly_question_budget=ctx.cfg.knowledge.weekly_question_budget,
                **ctx.export_kwargs(),
            )
        tick.passive_sops = pd_sops
        if pd_sops > 0:
            ctx.total_passive_sops += pd_sops
            ctx.total_sops_generated += pd_sops
            logger.info("Passive discovery: generated %d SOP(s)!", pd_sops)
            wm._write_sops_index(ctx.db, ctx.knowledge_base)
    except Exception:
        logger.warning("Passive discovery failed", exc_info=True)
    ctx.last_passive_discovery = now_mono
    return tick.passive_sops


def _passive_discovery_due(ctx: Any, now_mono: float) -> bool:
    """Due every 2 hours, or 5 minutes after the last run while the user is
    away (no events, no VLM jobs) inside the idle window; never while a
    focus recording is pending (focus goes first)."""
    if ctx.task_segmenter is None or ctx.sop_generator is None or not ctx.v2_schema_ok:
        return False
    since = now_mono - ctx.last_passive_discovery
    due = since >= PASSIVE_DISCOVERY_INTERVAL_SECONDS
    if not due and ctx.control_idle:
        due = (
            ctx.idle_gate.check().can_run
            and since >= PASSIVE_DISCOVERY_IDLE_SECONDS
        )
    if Path(wm._status_dir()).joinpath("focus-session.json").exists():
        return False
    return due

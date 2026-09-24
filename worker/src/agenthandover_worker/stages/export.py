"""Export stage (control thread): what each tick publishes.

:func:`run`, under ``STATE_LOCK``: the persisted state of the passive
annotation budget and the reasoning budget (what model use spent survives
a restart) and ``sops-index.json`` for the app (throttled to every 5 s).
:func:`heartbeat`, outside it: ``worker-status.json``, written every tick
so a long model call never stops it (also after an error: the app and
``doctor`` see a worker that is alive but failing, not a dead one).
"""

from __future__ import annotations

import logging
from typing import Any

from agenthandover_worker import main as wm

logger = logging.getLogger("agenthandover_worker")


def run(ctx: Any, tick: Any) -> None:
    budget_json = ctx.annotation_budget.to_json()
    if budget_json != ctx.saved_budget_json:
        ctx.saved_budget_json = budget_json
        ctx.db.set_state(wm.ANNOTATION_BUDGET_STATE_KEY, budget_json)
    reasoning_json = ctx.reasoning_budget.to_json()
    if reasoning_json != ctx.saved_reasoning_budget_json:
        ctx.saved_reasoning_budget_json = reasoning_json
        ctx.db.set_state(wm.REASONING_BUDGET_STATE_KEY, reasoning_json)
    wm._write_sops_index(ctx.db, ctx.knowledge_base)


def heartbeat(ctx: Any) -> None:
    """``worker-status.json``; a failure to write it is only logged."""
    try:
        _write_heartbeat(ctx)
    except Exception:
        logger.debug("Failed to write worker status", exc_info=True)


def _write_heartbeat(ctx: Any) -> None:
    """The running totals, the passive annotation budget and why passive
    annotation is paused, the reasoning budget and whether it is spent,
    the model schedule and the inference thread's queue."""
    db = ctx.db
    runner = ctx.runner
    inference = None
    if runner is not None:
        inference = runner.status()
        if ctx.v2_schema_ok:
            try:
                backlog = db.count_passive_unannotated()
                inference["annotation_backlog"] = backlog
                # The daemon keeps fewer passive frames meanwhile.
                inference["lower_capture_cadence"] = (
                    ctx.annotation_budget.asks_lower_capture(backlog)
                )
            except Exception:
                logger.debug("Annotation backlog count failed", exc_info=True)
    vlm_cfg = ctx.cfg.vlm
    wm._write_worker_status(
        started_at=ctx.started_at,
        events_processed_today=ctx.total_events_processed,
        sops_generated=ctx.total_sops_generated,
        last_pipeline_duration_ms=ctx.last_pipeline_duration_ms,
        consecutive_errors=max(
            ctx.consecutive_errors,
            runner.consecutive_failures() if runner is not None else 0,
        ),
        vlm_available=ctx.vlm_available,
        vlm_mode=vlm_cfg.mode,
        vlm_provider=vlm_cfg.provider or None,
        v2_annotation_enabled=ctx.v2_annotation_enabled,
        v2_annotations_today=ctx.total_v2_annotations,
        v2_diffs_today=ctx.total_v2_diffs,
        annotation_paused_reason=ctx.annotation_paused_reason,
        annotation_budget=ctx.annotation_budget.status(),
        reasoning_budget=ctx.reasoning_budget.status(),
        reasoning_paused_reason=ctx.reasoning_budget.pause_reason(),
        model_schedule=ctx.model_scheduler.status(),
        inference=inference,
    )

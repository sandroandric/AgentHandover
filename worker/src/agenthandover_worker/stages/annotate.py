"""Annotate stage (inference thread): passive frames and frame diffs (P2,
:func:`run`), and a focus recording's remaining keyframes (P1,
:func:`run_focus_tail`).  Both run in the annotation phase
(``ModelScheduler.begin_annotation``).
"""

from __future__ import annotations

import logging
from typing import Any

from agenthandover_worker import main as wm
from agenthandover_worker.ops_telemetry import PipelineMetrics

logger = logging.getLogger("agenthandover_worker")


def run_focus_tail(ctx: Any, tick: Any) -> None:
    """P1: the last recording's keyframes its Skill did not need."""
    ctx.model_scheduler.begin_annotation()
    ctx.focus_processor.annotate_remaining(ctx.db)


def run(ctx: Any, tick: Any) -> dict | None:
    """P2: passive frames within the annotation budget, then frame diffs.
    Returns the annotation stats (None when the v2 pipeline is off), also
    recorded on *tick* with the diff stats."""
    db = ctx.db
    ctx.model_scheduler.begin_annotation()
    # The schema is checked before the first tick; this re-checks it in
    # case the daemon applies the migration after the worker started.
    if ctx.scene_annotator is not None and not ctx.v2_schema_ok:
        ctx.v2_schema_ok = wm._check_v2_schema(db)
        if ctx.v2_schema_ok:
            logger.info("v2 schema now available — scene annotation pipeline activated")

    if ctx.scene_annotator is not None and ctx.v2_schema_ok:
        tick.annotations = _annotate_passive_frames(ctx)

    if ctx.frame_differ is not None and ctx.v2_schema_ok:
        diff_stats = wm._process_diffs(db, ctx.frame_differ, budget=ctx.annotation_budget)
        tick.diffs = diff_stats
        ctx.total_v2_diffs += diff_stats["diffs"]
        if diff_stats["diffs"] > 0 or diff_stats["failed"] > 0:
            logger.info(
                "v2 frame diff: %d action diffs, %d edge cases, %d failed",
                diff_stats["diffs"],
                diff_stats["edge_cases"],
                diff_stats["failed"],
            )
    return tick.annotations


def _annotate_passive_frames(ctx: Any) -> dict:
    """Passive frames: the expired backlog is given up on, the rest are
    annotated on AC, with the screen locked or after two idle minutes,
    within the hourly and daily budget.  Focus recordings are held by
    neither (their keyframes and Skill run in the focus job)."""
    db = ctx.db
    budget = ctx.annotation_budget
    for expired in db.expire_annotation_backlog(
        max_age_hours=wm.ANNOTATION_BACKLOG_MAX_AGE_HOURS, now=ctx.now(),
    ):
        ctx.scene_annotator.discard_screenshot(expired, ctx.screenshots_dir)
    budget.roll(ctx.local_now())
    allowed, why = ctx.idle_gate.passive_annotation_allowed()
    ann_stats = {
        "annotated": 0, "skipped": 0, "failed": 0,
        "blocked": 0, "deferred": 0, "budget_exhausted": False,
    }
    if allowed:
        ann_stats = wm._process_annotations(
            db, ctx.scene_annotator, ctx.screenshots_dir,
            privacy_checker=ctx.privacy_checker,
            activity_classifier=ctx.activity_classifier,
            budget=budget,
        )
    if not allowed:
        reason = why
    elif ann_stats["budget_exhausted"]:
        reason = budget.pause_reason()
    else:
        reason = None
    if reason != ctx.annotation_paused_reason:
        logger.info(
            "Passive annotation %s", f"paused: {reason}" if reason else "resumed",
        )
    ctx.annotation_paused_reason = reason
    ctx.total_v2_annotations += ann_stats["annotated"]

    if ann_stats["annotated"] > 0 or ann_stats["blocked"] > 0:
        logger.info(
            "v2 annotation: %d annotated, %d skipped, %d failed, %d blocked",
            ann_stats["annotated"],
            ann_stats["skipped"],
            ann_stats["failed"],
            ann_stats["blocked"],
        )
    if ann_stats["annotated"] > 0:
        try:
            ctx.ops_telemetry.record_batch(PipelineMetrics(
                annotation_count=ann_stats["annotated"],
                classification_count=ann_stats["annotated"],
            ))
        except Exception:
            logger.debug("Ops telemetry record failed", exc_info=True)
        # Newly annotated events appear in search and recall.
        if ctx.activity_searcher is not None:
            try:
                added = ctx.activity_searcher.refresh_index()
                if added > 0:
                    logger.debug("FTS5 index refreshed: %d new entries", added)
            except Exception:
                logger.debug("FTS5 index refresh failed", exc_info=True)
    return ann_stats

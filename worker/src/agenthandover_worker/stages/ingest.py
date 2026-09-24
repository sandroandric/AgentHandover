"""Ingest stage: the event pipeline over the daemon's unprocessed events.

Runs on the control thread (model-free): episodes, clipboard links,
negative-demo pruning and translation (``main.run_pipeline``), then marks
the batch processed.  It
also sets the next poll interval and logs the poll.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agenthandover_worker import main as wm

logger = logging.getLogger("agenthandover_worker")


def run(ctx: Any, tick: Any) -> dict | None:
    """Poll the database and run the pipeline over one ``EventRetryGuard``
    batch.  Returns the pipeline's summary (None when there were no
    events), also recorded on *tick*."""
    db = ctx.db
    unprocessed = db.get_unprocessed_events(limit=ctx.event_guard.batch_limit)
    tick.unprocessed = len(unprocessed)
    _log_poll(ctx, tick)

    if not unprocessed:
        return None
    # Core pipeline ALWAYS runs: episodes, translation and export happen
    # immediately so users see SOPs as soon as patterns are detected.
    tick.inflight_event_ids = [ev["id"] for ev in unprocessed if "id" in ev]
    pipeline_start = time.monotonic()
    summary = wm.run_pipeline(
        unprocessed,
        episode_builder=ctx.episode_builder,
        clipboard_linker=ctx.clipboard_linker,
        pruner=ctx.pruner,
        translator=ctx.translator,
    )
    ctx.last_pipeline_duration_ms = int((time.monotonic() - pipeline_start) * 1000)
    tick.pipeline = summary
    logger.info("Pipeline summary: %s", summary)

    ctx.total_events_processed += summary["events_in"]
    logger.info(
        "Processed %d events into %d episodes.",
        summary["events_in"],
        summary["episodes"],
    )

    # Mark processed events so they are not re-read.
    if tick.inflight_event_ids:
        db.mark_events_processed(tick.inflight_event_ids)
    ctx.event_guard.succeeded(tick.inflight_event_ids)
    tick.inflight_event_ids = []
    return summary


def _log_poll(ctx: Any, tick: Any) -> None:
    """Log the poll, set the next poll interval, and every
    ``_IDLE_LOG_INTERVAL_SECONDS`` of idleness say the worker is watching."""
    if tick.unprocessed:
        logger.log(
            wm._poll_log_level(tick.unprocessed, ctx.last_logged_poll_count),
            "Poll: %d unprocessed events",
            tick.unprocessed,
        )
        ctx.last_logged_poll_count = tick.unprocessed
        ctx.current_interval = wm._next_poll_interval(
            ctx.current_interval, ctx.poll_interval,
            busy=True, triggers_consumed=tick.triggers_consumed,
        )
        ctx.last_idle_log_time = ctx.monotonic()
        return

    logger.debug("Poll: nothing to do")
    ctx.last_logged_poll_count = None
    ctx.current_interval = wm._next_poll_interval(
        ctx.current_interval, ctx.poll_interval,
        busy=False, triggers_consumed=tick.triggers_consumed,
    )
    now_mono = ctx.monotonic()
    if now_mono - ctx.last_idle_log_time >= wm._IDLE_LOG_INTERVAL_SECONDS:
        logger.info(
            "Watching for activity... (%d events today, %d SOPs generated)",
            ctx.total_events_processed,
            ctx.total_sops_generated,
        )
        ctx.last_idle_log_time = now_mono

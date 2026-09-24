"""Control stage: app and CLI requests, the stale focus sweep, the model
jobs handed to the inference thread, live recognition.

Runs on the control thread under ``STATE_LOCK`` and calls no model: a
trigger whose handling needs one (a retry, a merge, an approval's
embedding) queues it with ``model_work``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from agenthandover_worker import main as wm
from agenthandover_worker import trigger_spool, workflow_mining
from agenthandover_worker.inference_runner import P0, P1, P2, model_work

logger = logging.getLogger("agenthandover_worker")


def run(ctx: Any, tick: Any) -> int:
    """App and CLI requests first, oldest first, then close a focus
    recording that was never stopped.  Returns the triggers consumed."""
    # The vector KB's embedding budget is per tick.
    if ctx.vector_kb is not None:
        ctx.vector_kb.reset_cycle_budget()

    tick.triggers_consumed = trigger_spool.drain(wm._status_dir(), ctx.trigger_handlers)
    # A consumed trigger (merge, dismiss, promote, ...) may have changed
    # the curation queue; nothing else between daily curation passes does.
    if tick.triggers_consumed:
        wm._write_curation_queue(ctx.procedure_curator)

    # Close a recording that was never stopped, so its rows get processed
    # instead of staying pending forever.
    wm._sweep_stale_focus_session(wm._status_dir(), ctx.cfg.focus.max_session_hours)
    return tick.triggers_consumed


def submit_model_jobs(ctx: Any, tick: Any) -> list:
    """Hand this tick's model work to the inference thread, decided from
    what the control thread sees now: a stopped recording (P0), its tail
    once its Skill ran (P1), passive annotation and generation (P2).
    Returns the jobs, also recorded on *tick*."""
    from agenthandover_worker.stages import annotate, daily, segment, synthesize

    def generation() -> None:
        segment.run(ctx, tick)
        daily.run(ctx, tick)

    ctx.control_idle = not tick.unprocessed
    jobs: list[tuple[str, int, Callable[[], None]]] = []
    if wm._focus_job_due(wm._status_dir()):
        jobs.append(("focus", P0, lambda: synthesize.run(ctx, tick)))
    if (
        ctx.focus_processor is not None
        and ctx.v2_schema_ok
        and ctx.focus_processor.has_remaining
    ):
        jobs.append(("focus-tail", P1, lambda: annotate.run_focus_tail(ctx, tick)))
    jobs.append(("passive-annotation", P2, lambda: annotate.run(ctx, tick)))
    jobs.append(("passive-generation", P2, generation))
    tick.jobs = jobs
    if ctx.runner is not None:
        for key, priority, job in jobs:
            ctx.runner.submit(key, priority, job)
    return jobs


def recognize(ctx: Any, tick: Any) -> bool:
    """Live recognition (model-free), every
    ``workflow_mining.RECOGNIZE_INTERVAL_SECONDS``: tag the session in
    progress with the Skill it begins, record it at its end.  True when
    it ran."""
    if not ctx.v2_schema_ok or (
        ctx.monotonic() - ctx.last_recognition
        < workflow_mining.RECOGNIZE_INTERVAL_SECONDS
    ):
        return False
    try:
        workflow_mining.recognize(ctx.db, ctx.knowledge_base, now=ctx.now())
    except Exception:
        logger.warning("Live recognition failed", exc_info=True)
    ctx.last_recognition = ctx.monotonic()
    return True


def trigger_handlers(ctx: Any) -> dict[str, Callable[[dict], None]]:
    """One handler per trigger kind (``trigger_spool.KINDS``).

    A retry regenerates a Skill and a merge asks the model to settle
    conflicts: both run on the inference thread (P0, ``model_work``).  The
    other handlers run on the control thread; the model part of an
    approval (embedding, style analysis) and a review card's free-text
    step answer (its revision, ``questions.revise_step_answer``) are
    queued the same way.
    """
    db = ctx.db
    export_kwargs = ctx.export_kwargs()

    def merge_job(trigger: dict) -> None:
        wm._process_merge_trigger(trigger, procedure_curator=ctx.procedure_curator)
        # Written again: the tick that took the trigger wrote the curation
        # queue before the merge ran.
        wm._write_curation_queue(ctx.procedure_curator)

    return {
        "retry": lambda t: model_work(
            f"retry:{uuid.uuid4().hex}",
            lambda: wm._process_retry_triggers(
                t, db,
                focus_processor=ctx.focus_processor,
                sop_generator=ctx.sop_generator,
                screenshots_dir=ctx.screenshots_dir,
                sop_auto_approve=ctx.cfg.sop.auto_approve,
                **export_kwargs,
            ),
        ),
        "approve": lambda t: wm._process_approval_triggers(
            t, db, vector_kb=ctx.vector_kb, **export_kwargs,
        ),
        "failed-query": lambda t: wm._process_failed_query_trigger(t, db),
        "search-query": lambda t: wm._process_search_trigger(t, ctx.activity_searcher),
        "recall-query": lambda t: wm._process_recall_trigger(t, ctx.activity_searcher),
        "trust-accept": lambda t: wm._process_trust_accept_trigger(
            t, ctx.trust_advisor, ctx.constraint_manager,
        ),
        "trust-dismiss": lambda t: wm._process_trust_dismiss_trigger(t, ctx.trust_advisor),
        "staleness-reviewed": lambda t: wm._process_staleness_reviewed_trigger(
            t, ctx.staleness_detector, ctx.knowledge_base,
        ),
        "lifecycle-promote": lambda t: wm._process_lifecycle_promote_trigger(
            t,
            lifecycle_manager=ctx.lifecycle_manager,
            db=db,
            approve=lambda slug: wm._approve_sop(
                db, slug, vector_kb=ctx.vector_kb, **export_kwargs,
            ),
            knowledge_base=ctx.knowledge_base,
            # With --adapter claude-skill the primary writer is the Claude
            # Skill writer.
            claude_skill_writer=(
                ctx.claude_skill_writer
                or (ctx.sop_writer if ctx.adapter == "claude-skill" else None)
            ),
        ),
        "merge": lambda t: model_work(
            f"merge:{uuid.uuid4().hex}", lambda: merge_job(t),
        ),
        "drift-reviewed": lambda t: wm._process_drift_reviewed_trigger(
            t, procedure_curator=ctx.procedure_curator,
        ),
        "lifecycle-dismiss": lambda t: wm._process_lifecycle_dismiss_trigger(
            t, procedure_curator=ctx.procedure_curator,
        ),
        "export": lambda t: wm._check_export_trigger(
            t,
            openclaw_writer=ctx.sop_writer,
            sops_dir=ctx.sops_dir,
            procedure_writer=ctx.procedure_writer,
        ),
        "micro-answer": lambda t: wm._process_micro_answer_trigger(
            t, db, ctx.knowledge_base, now=ctx.now(), sop_generator=ctx.sop_generator,
        ),
    }

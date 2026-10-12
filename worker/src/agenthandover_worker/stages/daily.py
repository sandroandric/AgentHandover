"""Daily stage (inference thread, P2): the once-per-day pass.

The second half of the passive generation job, on the first tick of each
local day that the generation plan allows it (keyed on the date persisted
when it last ran, so a restart does not rerun it): yesterday's daily batch
(summary, profile, patterns, guardrails from corrections, session links,
trust suggestions, behavioural re-synthesis, evidence extraction, default
constraints, the day's questions, the digest), workflow observations, OCR
text preservation, staleness, lifecycle transitions and the bundle
compile, the curation queue, and the stores' maintenance.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from agenthandover_worker import main as wm
from agenthandover_worker import workflow_mining
from agenthandover_worker.focus_processor import recorded_observations
from agenthandover_worker.scheduler import charged_to
from agenthandover_worker.thumbnails import thumbnail_days

logger = logging.getLogger("agenthandover_worker")


def run(ctx: Any, tick: Any) -> bool:
    """The once-per-day pass, when the generation plan on *tick* allows it
    (``segment.run`` planned it).  True when it ran."""
    plan = tick.generation_plan
    if plan is None or not plan.allows("daily"):
        return False
    db = ctx.db
    today_local = tick.local_day

    # Process *yesterday* (the local calendar day): today is still in
    # progress and would produce an incomplete summary.
    yesterday_str = (ctx.local_now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        _daily_batch(ctx, yesterday_str)
    except Exception:
        logger.warning("Daily batch failed", exc_info=True)

    # Workflow observations: segment and observation TTLs, the promoted
    # workflows' recurrence and value.
    workflow_mining.run_daily(db, ctx.knowledge_base, now=ctx.now())
    _preserve_ocr_text(ctx)
    _check_staleness(ctx)

    if ctx.cfg.features.lifecycle_management:
        try:
            transitions = ctx.lifecycle_manager.apply_auto_transitions()
            if transitions:
                logger.info("Lifecycle: %d auto-transition(s)", len(transitions))
        except Exception:
            logger.debug("Lifecycle auto-transitions failed", exc_info=True)
        # Write what the gate now lets through (approved since the last
        # pass); drafts are never written.
        try:
            ctx.bundle_compiler.compile_all()
        except Exception:
            logger.warning("Bundle compile failed", exc_info=True)

    try:
        with charged_to(ctx.annotation_budget):
            curation_summary = ctx.procedure_curator.curate()
        if curation_summary.total_queue_items > 0:
            logger.info(
                "Curation: %d item(s) (%d merges, %d upgrades, %d stale, %d drift)",
                curation_summary.total_queue_items,
                curation_summary.merge_candidates,
                curation_summary.upgrade_candidates,
                curation_summary.stale_procedures,
                curation_summary.drift_reports,
            )
    except Exception:
        logger.debug("Curation batch failed", exc_info=True)
    # curation-queue.json for the app.
    wm._write_curation_queue(ctx.procedure_curator)

    wm._run_daily_maintenance(
        vector_kb=ctx.vector_kb,
        activity_searcher=ctx.activity_searcher,
        annotation_cache=ctx.annotation_cache,
        db=db,
        thumbnail_days=thumbnail_days(ctx.cfg.raw),
    )
    db.set_state(wm.DAILY_JOBS_STATE_KEY, today_local)
    ctx.daily_jobs_date = today_local
    return True


def _daily_batch(ctx: Any, day: str) -> None:
    """Aggregate *day*'s (complete) events; with three or more daily
    summaries, learn from them; then the default constraints, the day's
    questions and the digest."""
    kb = ctx.knowledge_base
    day_events = ctx.db.get_annotated_events_for_date(day)
    if not day_events:
        return
    summary = ctx.daily_processor.process_day(day, day_events)
    logger.info(
        "Daily batch: %d tasks, %.1f active hours",
        summary.task_count,
        summary.active_hours,
    )
    # The daily summary, for semantic search.
    wm._embed_in_vector_kb(
        ctx.vector_kb, "daily_summary", day,
        f"Daily summary for {day}: "
        f"{summary.task_count} tasks, "
        f"{summary.active_hours:.1f} active hours. "
        # ``DailySummary.tasks`` holds ``TaskBoundary`` records.
        + " ".join(t.intent for t in summary.tasks if t.intent)[:2000],
    )
    if len(kb.list_daily_summaries(limit=5)) >= 3:
        _learn_from_summaries(ctx)

    # Default constraints when the constraints file has none; on every
    # daily batch (outside the three-summaries gate) so they exist from
    # day one.  The guard writes them at most once.
    try:
        existing = kb.get_constraints()
        if not existing.get("global"):
            kb.update_constraints({
                "global": {
                    "default_trust_level": "observe",
                    "require_approval_for_destructive": True,
                    "max_autonomous_actions_per_hour": 10,
                },
                "per_procedure": existing.get("per_procedure", {}),
            })
            logger.info("Default global constraints seeded")
    except Exception:
        logger.debug("Constraint seeding failed", exc_info=True)
    # Ask what the day's evidence raised that no pass asked yet (the
    # questions agent runs queued), so the digest lists them with the
    # other open cards.
    wm._ask_micro_questions(
        ctx.db, kb, now=ctx.now(),
        weekly_budget=ctx.cfg.knowledge.weekly_question_budget,
    )
    try:
        from agenthandover_worker.daily_digest import DigestGenerator

        digest_generator = DigestGenerator(kb, db=ctx.db)
        digest_generator.save_digest(digest_generator.generate(day))
        logger.info("Daily digest generated for %s", day)
    except Exception:
        logger.debug("Digest generation failed", exc_info=True)


def _learn_from_summaries(ctx: Any) -> None:
    """Profile, patterns, guardrails from corrections, session links, trust
    suggestions, behavioural re-synthesis and evidence extraction."""
    kb = ctx.knowledge_base
    ctx.profile_builder.update_profile()
    try:
        profile = kb.get_profile()
        if profile:
            wm._embed_in_vector_kb(ctx.vector_kb, "profile", "user", _profile_text(profile))
    except Exception:
        logger.debug("Profile embedding failed", exc_info=True)
    # The activity classifier reads the updated profile.
    if ctx.activity_classifier is not None:
        try:
            ctx.activity_classifier._profile = kb.get_profile()
        except Exception:
            logger.debug("Failed to refresh classifier profile", exc_info=True)
    patterns = ctx.pattern_detector.detect_recurrence()
    if patterns:
        ctx.pattern_detector.update_triggers(patterns)
    chains = ctx.pattern_detector.detect_chains()
    if chains:
        ctx.pattern_detector.update_chains(chains)

    # Correction-to-guardrail learning.
    try:
        guardrail_count = 0
        for proc in kb.list_procedures():
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue
            guardrails = ctx.correction_detector.analyze_correction_patterns(slug)
            if guardrails:
                ctx.correction_detector.apply_guardrails_to_procedure(slug, guardrails)
                guardrail_count += len(guardrails)
        if guardrail_count:
            logger.info("Correction learning: %d guardrail(s) applied", guardrail_count)
    except Exception:
        logger.debug("Correction-to-guardrail learning failed", exc_info=True)

    try:
        ctx.session_linker.analyze_daily_summaries()
    except Exception:
        logger.debug("Session linking failed", exc_info=True)
    try:
        suggestions = ctx.trust_advisor.evaluate_all()
        if suggestions:
            logger.info("Trust advisor: %d new suggestion(s)", len(suggestions))
    except Exception:
        logger.debug("Trust evaluation failed", exc_info=True)

    try:
        _resynthesize(ctx)
    except Exception:
        logger.debug("Behavioral re-synthesis failed", exc_info=True)

    # Pre-expiry evidence extraction.
    try:
        extracted = ctx.evidence_extractor.extract_all_pending()
        if extracted:
            logger.info("Evidence extraction: %d procedure(s) processed", extracted)
    except Exception:
        logger.debug("Pre-expiry evidence extraction failed", exc_info=True)


def _profile_text(profile: dict) -> str:
    """The profile's content for its embedding: without ``updated_at``,
    the wall-clock time of the save, which says nothing to a search and
    made the text (and the embedding call) new on every daily pass."""
    content = {k: v for k, v in profile.items() if k != "updated_at"}
    return json.dumps(content, default=str)[:4000]


def _resynthesize(ctx: Any) -> None:
    """Behavioural re-synthesis of the procedures that are due for it."""
    kb = ctx.knowledge_base
    synthesizer = ctx.behavioral_synthesizer
    # Daily summaries, loaded once for all procedures.
    daily = [kb.get_daily_summary(d) for d in kb.list_daily_summaries(limit=7)]
    daily = [s for s in daily if s is not None]
    synth_count = 0
    for proc_info in kb.list_procedures():
        slug = proc_info.get("id", proc_info.get("slug", ""))
        if not slug:
            continue
        proc = kb.get_procedure(slug)
        if proc is None or proc.get("lifecycle_state") == "archived":
            continue
        if not synthesizer.should_synthesize(proc):
            continue
        # The recorded instances as rich per-frame dicts (verbatim text and
        # list picks), with the session ids the selection criteria cite
        # (focus_processor.recorded_observations).
        observations, instance_ids = recorded_observations(ctx.db, proc)
        insights = synthesizer.synthesize(
            slug, proc, observations,
            daily_summaries=daily,
            instance_ids=instance_ids,
        )
        if insights.strategy or insights.guardrails:
            # Into the procedure as saved now: an approval may have moved
            # it during the model call.
            proc = kb.get_procedure(slug) or proc
            kb.save_procedure(synthesizer.merge_insights_into_procedure(proc, insights))
            synth_count += 1
    if synth_count:
        logger.info("Behavioral synthesis: %d procedure(s) updated", synth_count)


def _preserve_ocr_text(ctx: Any) -> None:
    """Keep the OCR text of events about to expire, and embed it."""
    try:
        pcount, pitems = ctx.db.preserve_ocr_text(retention_days=14)
        if pcount:
            logger.info("Preserved %d texts before expiry", pcount)
        if pitems and ctx.vector_kb is not None:
            try:
                ctx.vector_kb.upsert_batch([
                    ("preserved_text", f"{eid}:{src}", txt[:2000])
                    for eid, src, txt in pitems
                ])
            except Exception:
                logger.debug("Preserved text embedding failed", exc_info=True)
    except Exception:
        logger.debug("OCR text preservation failed", exc_info=True)


def _check_staleness(ctx: Any) -> None:
    """Check every procedure's staleness and save it into the procedure
    (the app's review reads it there)."""
    kb = ctx.knowledge_base
    try:
        reports = ctx.staleness_detector.check_all()
        stale_count = sum(1 for r in reports if r.status != "current")
        if stale_count > 0:
            logger.info(
                "Staleness check: %d/%d procedures need attention",
                stale_count, len(reports),
            )
        for report in reports:
            try:
                proc = kb.get_procedure(report.slug)
                if proc is None:
                    continue
                staleness = proc.get("staleness", {})
                staleness["status"] = report.status
                staleness["signals"] = [
                    {"type": s.type, "detail": s.detail, "first_seen": s.first_seen}
                    for s in report.signals
                ]
                staleness["recommended_action"] = report.recommended_action
                proc["staleness"] = staleness
                kb.save_procedure(proc)
            except Exception:
                logger.debug(
                    "Failed to persist staleness for %s", report.slug, exc_info=True,
                )
    except Exception:
        logger.debug("Staleness check failed", exc_info=True)

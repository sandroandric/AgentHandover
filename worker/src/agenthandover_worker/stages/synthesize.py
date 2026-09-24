"""Synthesize stage (inference thread, P0): a stopped focus recording, or
the answers to its questions, becomes a Skill.

Never batched or held for memory.  The v2 pipeline when available
(semantic Skills from screen annotations: the recording's keyframes are
annotated before the Skill call switches Ollama to the generation model),
else v1 (episode builder, translator and one template).
"""

from __future__ import annotations

import logging
from typing import Any

from agenthandover_worker import main as wm

logger = logging.getLogger("agenthandover_worker")


def run(ctx: Any, tick: Any) -> int:
    """P0: the focus job.  Returns the Skills generated, also recorded on
    *tick*."""
    db = ctx.db
    scheduler = ctx.model_scheduler
    # Frames first: the recording's keyframes are annotated before its
    # Skill switches Ollama to the generation model.
    scheduler.begin_annotation()
    scheduler.plan_generation(
        urgent=wm._focus_work_ready(wm._status_dir(), db) is not None,
        due={}, take_batched=False,
    )
    if ctx.focus_processor is not None and ctx.v2_schema_ok:
        focus_sops = wm._process_focus_sessions_v2(
            db,
            focus_processor=ctx.focus_processor,
            screenshots_dir=ctx.screenshots_dir,
            sop_auto_approve=ctx.cfg.sop.auto_approve,
            behavioral_synthesizer=ctx.behavioral_synthesizer,
            knowledge_base=ctx.knowledge_base,
            vector_kb=ctx.vector_kb,
            weekly_question_budget=ctx.cfg.knowledge.weekly_question_budget,
            now=ctx.now(),
            **ctx.export_kwargs(),
        )
    else:
        focus_sops = wm._process_focus_sessions(
            db,
            episode_builder=ctx.episode_builder,
            clipboard_linker=ctx.clipboard_linker,
            translator=ctx.translator,
            scorer=ctx.scorer,
            sop_enhancer=ctx.sop_enhancer,
            sop_auto_approve=ctx.cfg.sop.auto_approve,
            **ctx.export_kwargs(),
        )
    tick.focus_sops = focus_sops
    if focus_sops > 0:
        ctx.total_sops_generated += focus_sops
        logger.info("Focus recording: generated %d SOP(s)!", focus_sops)
        wm._write_sops_index(db, ctx.knowledge_base)
    return focus_sops

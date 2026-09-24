"""Journey driver: one worker poll iteration per ``run_tick()``, on a
controllable clock.

``main()`` builds its components and runs its poll loop inline, so until
WP-S6 splits it, ``JourneyWorker`` builds the same components the same way
(one ``load_config()`` read, as in ``main()``, so ``config.toml`` in the
test's ``HOME`` decides models, hosts and feature flags) and ``run_tick()`` calls
the loop body's processors in the loop's order:

1. ``trigger_spool.drain`` over main()'s trigger handlers (retry, approve,
   failed query, search, recall, trust accept / dismiss, staleness
   reviewed, lifecycle promote, merge, drift reviewed, lifecycle dismiss,
   export) and, when a trigger was consumed, the curation queue refresh;
2. the stale focus-session sweep;
3. ``run_pipeline`` over one ``EventRetryGuard`` batch of unprocessed
   events, marking them processed and persisting VLM jobs;
   then, after the jobs below are decided, ``workflow_mining.recognize``
   every 2 minutes of *clock* time (model-free, so main() runs it in the
   control tick);
4. the model jobs main() hands to its inference thread
   (``inference_runner``), decided from what this tick sees and run in the
   runner's priority order:

   a. P0, when a recording stopped or its questions were answered: the
      focus job (``ModelScheduler.begin_annotation``, ``plan_generation``
      for itself, ``_process_focus_sessions_v2`` or the v1 path when the
      v2 pipeline is off);
   b. P1, when the last recording's Skill left keyframes (so from the tick
      after its focus job): ``annotate_remaining``;
   c. P2: the passive annotation job (``begin_annotation``,
      ``_process_annotations`` within main()'s hourly and daily
      ``AnnotationBudget``, rolled on the clock, and ``_process_diffs``,
      charged to it);
   d. P2: the passive generation job (``plan_generation``, then
      ``_process_passive_discovery`` every 2 hours of *clock* time, the
      timer starting when the worker is built as main() seeds it at loop
      entry, while no focus session is pending, charged to the budget);

   then the runner's idle hook, ``ModelScheduler.end_generation``.  After
   :meth:`JourneyWorker.start_inference` the jobs are submitted to a real
   inference thread instead, and ``run_tick()`` is the control tick alone;
5. the status heartbeat, ``_write_worker_status`` with the running totals,
   the annotation budget and pause reason, the compute counters, the
   model schedule and, with an inference thread, its queue (``status()``
   reads it back).  Building a ``JourneyWorker`` resets the process-wide
   usage counters, as a worker start does;
6. ``_write_sops_index``.

The scheduler runs on the journey clock and, unless a journey passes
``free_memory``, sees memory to spare, no swap in use and disk to spare
(the host's real ``vm_stat``, swap and disk would make a replay depend on
the machine running it).

Not driven here, because they only exist inline in ``main()``: the VLM
fallback job runner (needs ``VLMWorker``), the idle-gate shortcut for
passive discovery and passive annotation (both read real system idle
time and power state; the driver runs as if on AC), the expiry of the
passive annotation backlog (compares with SQLite's real ``now``), the
daily batch and the staleness / lifecycle / curation pass.  Journeys that
need those wait for WP-S6.

``main()`` logs an exception inside an iteration and carries on with the
next one (shrinking or quarantining the event batch); ``run_tick()`` lets it
propagate so the journey fails at its cause.  A failing trigger handler is
logged and skipped by ``trigger_spool.drain`` itself, in both.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from agenthandover_worker import main as wm
from agenthandover_worker import ops_telemetry
from agenthandover_worker import trigger_spool
from agenthandover_worker.annotation_cache import AnnotationCache
from agenthandover_worker.config import load_config
from agenthandover_worker.inference_runner import (
    P0, P1, P2, STATE_LOCK, InferenceRunner, model_work,
)
from agenthandover_worker.loop_guard import EventRetryGuard
from agenthandover_worker.model_scheduler import GB, ModelScheduler
from agenthandover_worker.ollama import OllamaClient, OllamaError
from agenthandover_worker.scheduler import AnnotationBudget, charged_to

PASSIVE_DISCOVERY_INTERVAL = timedelta(seconds=7200)  # main(): _PASSIVE_DISCOVERY_INTERVAL


class Clock:
    """A ``now()`` that only moves when the test moves it (aware UTC)."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("Clock needs a timezone-aware start")
        self._now = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta | None = None, **kwargs: float) -> datetime:
        """Move forward by ``delta`` or ``timedelta(**kwargs)``; returns the new now."""
        step = delta if delta is not None else timedelta(**kwargs)
        if step < timedelta(0):
            raise ValueError("the clock does not go backwards")
        self._now += step
        return self._now


class JourneyWorker:
    """The worker's components, built as ``main()`` builds them."""

    def __init__(
        self,
        *,
        db_path: Path,
        knowledge_dir: Path,
        sops_dir: Path,
        clock: Clock,
        adapter: str = "openclaw",
        free_memory: Callable[[], int | None] = lambda: 64 * GB,
    ) -> None:
        from agenthandover_worker.activity_classifier import ActivityClassifier
        from agenthandover_worker.activity_search import ActivitySearcher
        from agenthandover_worker.behavioral_synthesizer import (
            BehavioralSynthesizer,
            SynthesizerConfig,
        )
        from agenthandover_worker.constraint_manager import ConstraintManager
        from agenthandover_worker.continuity_tracker import ContinuityTracker
        from agenthandover_worker.evidence_normalizer import EvidenceNormalizer
        from agenthandover_worker.evidence_tracker import EvidenceTracker
        from agenthandover_worker.focus_questioner import FocusQuestioner
        from agenthandover_worker.knowledge_base import KnowledgeBase
        from agenthandover_worker.knowledge_export_adapter import KnowledgeBaseExportAdapter
        from agenthandover_worker.lifecycle_manager import LifecycleManager
        from agenthandover_worker.llm_reasoning import (
            LLMReasoner,
            ReasoningConfig,
            annotation_reasoner,
        )
        from agenthandover_worker.privacy_zones import PrivacyZoneChecker, PrivacyZoneConfig
        from agenthandover_worker.procedure_curator import ProcedureCurator
        from agenthandover_worker.procedure_matcher import ProcedureMatcher
        from agenthandover_worker.procedure_writer import ProcedureWriter
        from agenthandover_worker.quality_judge import QualityJudge
        from agenthandover_worker.staleness_detector import StalenessDetector
        from agenthandover_worker.trust_advisor import TrustAdvisor
        from agenthandover_worker.user_policy import UserPolicy
        from agenthandover_worker.variant_detector import VariantDetector
        from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig

        self.adapter = adapter
        self.db_path = Path(db_path)
        self.sops_dir = Path(sops_dir)
        self.clock = clock
        self.state_dir = wm._status_dir()
        self.screenshots_dir = self.state_dir / "screenshots"
        self._last_passive_discovery = clock.now()
        self._last_recognition = clock.now()
        # A worker is one process: its usage counters start at zero.
        ops_telemetry.MODEL_USAGE.reset()
        ops_telemetry.COMPUTE.reset()
        self._started_at = clock.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self._events_processed = 0
        self._sops_generated = 0
        self._annotations = 0
        self._diffs = 0
        self._paused_reason: str | None = None
        self.runner: InferenceRunner | None = None

        cfg = load_config()
        self.cfg = cfg
        vlm_cfg = cfg.vlm
        # Passive model use's budget, as main() builds it.
        self.annotation_budget = AnnotationBudget.from_config(vlm_cfg)
        features = cfg.features
        host = vlm_cfg.ollama_host

        kb = KnowledgeBase(root=Path(knowledge_dir))
        kb.ensure_structure()
        self.knowledge_base = kb

        self.vlm_queue = wm.VLMFallbackQueue()
        llm_reasoner = LLMReasoner(
            config=ReasoningConfig(model=vlm_cfg.reasoner_model, ollama_host=host),
            vlm_queue=self.vlm_queue,
        )
        # Vector KB with main()'s health check (unavailable -> None),
        # built before its consumers as main() builds it.
        self.vector_kb = None
        vector_kb = VectorKB(self.db_path, VectorKBConfig(
            embedding_model=cfg.embedding.model, ollama_host=host,
            retention_days=cfg.embedding.retention_days,
            max_rows=cfg.embedding.max_rows,
        ))
        try:
            embedding = vector_kb.compute_embeddings(["health check"])
        except (ConnectionError, OllamaError):
            embedding = None
        if embedding and embedding[0]:
            self.vector_kb = vector_kb
        else:
            vector_kb.close()

        self.lifecycle_manager = LifecycleManager(kb)
        self.procedure_writer = ProcedureWriter(
            kb=kb, evidence=EvidenceTracker(knowledge_base=kb),
            lifecycle_manager=self.lifecycle_manager, vector_kb=self.vector_kb,
            llm_reasoner=llm_reasoner,
        )
        self.kb_export_adapter = KnowledgeBaseExportAdapter(kb)
        pz_cfg = PrivacyZoneConfig()
        pz_raw = cfg.privacy_zones
        if pz_raw:
            pz_cfg = PrivacyZoneConfig(
                full_observation=pz_raw.get("full_observation", []),
                metadata_only=pz_raw.get("metadata_only", []),
                blocked=pz_raw.get("blocked", []),
                blocked_urls=pz_raw.get("blocked_urls", []),
                auto_pause=pz_raw.get("auto_pause", []),
            )
        self.privacy_checker = PrivacyZoneChecker(config=pz_cfg)
        self.staleness_detector = StalenessDetector(kb)
        self.activity_classifier = None
        if features.activity_classification:
            self.activity_classifier = ActivityClassifier(
                profile=kb.get_profile(), policy=UserPolicy(kb),
                llm_reasoner=annotation_reasoner(
                    vlm_cfg.annotation_model, host, self.vlm_queue,
                ),
            )
        self.constraint_manager = ConstraintManager(kb)

        self.trust_advisor = TrustAdvisor(kb)
        self.procedure_matcher = ProcedureMatcher(kb=kb, vector_kb=self.vector_kb)
        self.continuity_tracker = None
        if features.continuity_tracking:
            self.continuity_tracker = ContinuityTracker(
                kb=kb, matcher=self.procedure_matcher,
            )
        self.variant_detector = VariantDetector()
        self.evidence_normalizer = EvidenceNormalizer(variant_detector=self.variant_detector)
        self.focus_max_session_hours = cfg.focus.max_session_hours
        self.behavioral_synthesizer = BehavioralSynthesizer(
            config=SynthesizerConfig(model=vlm_cfg.sop_model, ollama_host=host),
            vlm_queue=self.vlm_queue, knowledge_base=kb,
        )
        self.focus_questioner = FocusQuestioner(llm_reasoner=llm_reasoner)
        self.procedure_curator = None
        if features.curation:
            self.procedure_curator = ProcedureCurator(
                kb=kb,
                staleness_detector=self.staleness_detector,
                trust_advisor=self.trust_advisor,
                lifecycle_manager=self.lifecycle_manager,
                evidence_normalizer=self.evidence_normalizer,
                quality_judge=QualityJudge(llm_reasoner),
                llm_reasoner=llm_reasoner,
            )

        self.episode_builder = wm.EpisodeBuilder()
        self.clipboard_linker = wm.ClipboardLinker()
        self.pruner = wm.NegativeDemoPruner()
        self.translator = wm.SemanticTranslator()
        self.scorer = wm.ConfidenceScorer()
        self.index_generator = wm.IndexGenerator()
        self.skill_md_writer = None
        self.claude_skill_writer = None
        if adapter == "generic":
            from agenthandover_worker.generic_writer import GenericWriter
            self.sop_writer = GenericWriter(output_dir=self.sops_dir)
        elif adapter == "claude-skill":
            from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
            self.sop_writer = ClaudeSkillWriter(skills_dir=cfg.export.skills_dir)
        elif adapter == "openclaw":
            self.sop_writer = wm.OpenClawWriter(workspace_dir=self.sops_dir.parent.parent)
        else:
            raise ValueError(f"adapter {adapter!r} is not driven by journeys")

        # SOP enhancement (the v1 focus fallback's), as main() sets it up.
        self.sop_enhancer = None
        llm_config = dataclasses.asdict(cfg.llm)
        if llm_config.get("enhance_sops", True):
            from agenthandover_worker.sop_enhancer import SOPEnhancer, create_llm_backend
            backend = create_llm_backend(llm_config, {
                "mode": vlm_cfg.mode,
                "provider": vlm_cfg.provider,
                "model": vlm_cfg.model,
                "api_key_env": vlm_cfg.api_key_env,
            })
            if backend is not None:
                self.sop_enhancer = SOPEnhancer(
                    backend=backend,
                    max_enhancements_per_day=int(llm_config.get("max_enhancements_per_day", 20)),
                )

        self.scene_annotator = None
        self.frame_differ = None
        self.focus_processor = None
        self.task_segmenter = None
        self.sop_generator = None
        self.annotation_cache = None
        if vlm_cfg.annotation_enabled:
            self.annotation_cache = AnnotationCache(self.db_path)
            self.scene_annotator = wm.SceneAnnotator(
                config=wm.AnnotationConfig(
                    model=vlm_cfg.annotation_model,
                    ollama_host=host,
                    sliding_window_max_age_sec=vlm_cfg.sliding_window_max_age_sec,
                    timeout_seconds=vlm_cfg.annotation_timeout_seconds,
                ),
                image_embedder=None,
                vector_kb=self.vector_kb,
                cache=self.annotation_cache,
            )
            self.scene_annotator.preload()
            self.frame_differ = wm.FrameDiffer(
                config=wm.DiffConfig(model=vlm_cfg.annotation_model, ollama_host=host),
            )
            self.sop_generator = wm.SOPGenerator(
                config=wm.SOPGeneratorConfig(model=vlm_cfg.sop_model, ollama_host=host),
            )
            self.focus_processor = wm.FocusProcessor(
                annotator=self.scene_annotator,
                differ=self.frame_differ,
                sop_generator=self.sop_generator,
            )
            self.task_segmenter = wm.TaskSegmenter(
                config=wm.SegmenterConfig(
                    embedding_model=cfg.embedding.model, ollama_host=host,
                ),
                llm_reasoner=llm_reasoner,
                cluster_label_store=kb.root / "observations" / "cluster_labels.json",
                vector_kb=self.vector_kb,
            )
        self.activity_searcher = ActivitySearcher(db_path=self.db_path)
        self.sop_auto_approve = cfg.sop.auto_approve
        self.model_scheduler = ModelScheduler(
            OllamaClient(host),
            annotation_model=vlm_cfg.annotation_model,
            sop_model=vlm_cfg.sop_model,
            clock=lambda: clock.now().timestamp(),
            free_memory=free_memory,
            swap_used=lambda: 0,
            free_disk=lambda: 100 * GB,
        )
        self.model_scheduler.register(
            self.sop_generator, self.behavioral_synthesizer, llm_reasoner,
        )

        self.db = wm.WorkerDB(self.db_path)
        self.v2_schema_ok = (
            self.scene_annotator is not None and wm._check_v2_schema(self.db)
        )
        self.event_guard = EventRetryGuard(max_batch=100)
        self.trigger_handlers = self._trigger_handlers()
        assert set(self.trigger_handlers) == set(trigger_spool.KINDS)
        trigger_spool.migrate_legacy(self.state_dir)
        wm._write_sops_index(self.db, kb, force=True)

    def _export_kwargs(self) -> dict:
        return {
            "openclaw_writer": self.sop_writer,
            "skill_md_writer": self.skill_md_writer,
            "claude_skill_writer": self.claude_skill_writer,
            "index_generator": self.index_generator,
            "procedure_writer": self.procedure_writer,
            "kb_export_adapter": self.kb_export_adapter,
        }

    def _trigger_handlers(self) -> dict:
        """main()'s ``trigger_handlers``, bound to these components."""
        db = self.db
        export_kwargs = self._export_kwargs()
        return {
            "retry": lambda t: model_work(
                f"retry:{id(t)}",
                lambda: wm._process_retry_triggers(
                    t, db,
                    focus_processor=self.focus_processor,
                    sop_generator=self.sop_generator,
                    screenshots_dir=self.screenshots_dir,
                    sop_auto_approve=self.sop_auto_approve,
                    **export_kwargs,
                ),
            ),
            "approve": lambda t: wm._process_approval_triggers(
                t, db, vector_kb=self.vector_kb, **export_kwargs,
            ),
            "failed-query": lambda t: wm._process_failed_query_trigger(t, db),
            "search-query": lambda t: wm._process_search_trigger(t, self.activity_searcher),
            "recall-query": lambda t: wm._process_recall_trigger(t, self.activity_searcher),
            "trust-accept": lambda t: wm._process_trust_accept_trigger(
                t, self.trust_advisor, self.constraint_manager,
            ),
            "trust-dismiss": lambda t: wm._process_trust_dismiss_trigger(t, self.trust_advisor),
            "staleness-reviewed": lambda t: wm._process_staleness_reviewed_trigger(
                t, self.staleness_detector, self.knowledge_base,
            ),
            "lifecycle-promote": lambda t: wm._process_lifecycle_promote_trigger(
                t,
                lifecycle_manager=self.lifecycle_manager,
                db=db,
                approve=lambda slug: wm._approve_sop(
                    db, slug, vector_kb=self.vector_kb, **export_kwargs,
                ),
                knowledge_base=self.knowledge_base,
                claude_skill_writer=(
                    self.claude_skill_writer
                    or (self.sop_writer if self.adapter == "claude-skill" else None)
                ),
            ),
            "merge": lambda t: model_work(f"merge:{id(t)}", lambda: self._merge_job(t)),
            "drift-reviewed": lambda t: wm._process_drift_reviewed_trigger(
                t, procedure_curator=self.procedure_curator,
            ),
            "lifecycle-dismiss": lambda t: wm._process_lifecycle_dismiss_trigger(
                t, procedure_curator=self.procedure_curator,
            ),
            "export": lambda t: wm._check_export_trigger(
                t,
                openclaw_writer=self.sop_writer,
                sops_dir=self.sops_dir,
                procedure_writer=self.procedure_writer,
            ),
        }

    def _merge_job(self, trigger: dict) -> None:
        wm._process_merge_trigger(trigger, procedure_curator=self.procedure_curator)
        wm._write_curation_queue(self.procedure_curator)

    def close(self) -> None:
        if self.runner is not None:
            self.runner.stop(timeout=60)
            self.runner = None
        self.db.close()
        if self.vector_kb is not None:
            self.vector_kb.close()
        self.activity_searcher.close()
        if self.annotation_cache is not None:
            self.annotation_cache.close()

    def __enter__(self) -> "JourneyWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start_inference(self) -> InferenceRunner:
        """Run model work on an inference thread, as ``main()`` does: from
        now on :meth:`run_tick` is the control thread's tick only (it
        submits the jobs and returns; the runner takes them in priority
        order).  :meth:`close` stops it."""
        self.runner = InferenceRunner(
            on_idle=self.model_scheduler.end_generation,
        ).start()
        return self.runner

    def run_tick(self) -> dict:
        """One poll iteration of main()'s loop (see module docstring).

        Returns what the iteration did, for assertions.  With an inference
        thread (:meth:`start_inference`) the model jobs are only submitted,
        so their entries stay at their defaults.
        """
        db = self.db
        result: dict = {
            "triggers_consumed": 0, "focus_sops": 0, "pipeline": None,
            "annotations": None, "diffs": None, "passive_sops": None,
            "generation_plan": None,
        }
        with STATE_LOCK:
            if self.vector_kb is not None:
                self.vector_kb.reset_cycle_budget()

            triggers_consumed = trigger_spool.drain(self.state_dir, self.trigger_handlers)
            result["triggers_consumed"] = triggers_consumed
            if triggers_consumed:
                wm._write_curation_queue(self.procedure_curator)

            wm._sweep_stale_focus_session(self.state_dir, self.focus_max_session_hours)
            unprocessed = db.get_unprocessed_events(limit=self.event_guard.batch_limit)
            if unprocessed:
                summary = wm.run_pipeline(
                    unprocessed,
                    episode_builder=self.episode_builder,
                    clipboard_linker=self.clipboard_linker,
                    pruner=self.pruner,
                    translator=self.translator,
                    scorer=self.scorer,
                    vlm_queue=self.vlm_queue,
                    db=db,
                )
                result["pipeline"] = summary
                event_ids = [ev["id"] for ev in unprocessed if "id" in ev]
                if event_ids:
                    db.mark_events_processed(event_ids)
                self.event_guard.succeeded(event_ids)
                if summary["vlm_enqueued"] > 0:
                    wm._persist_vlm_jobs(db, self.vlm_queue)
            self._events_processed += len(unprocessed)

            # main() decides the jobs from what the control thread sees now:
            # a recording's tail is due only once its focus job has run.
            jobs: list[tuple[str, int, Callable[[], None]]] = []
            if wm._focus_job_due(self.state_dir):
                jobs.append(("focus", P0, lambda: self._focus_job(result)))
            if (
                self.focus_processor is not None
                and self.v2_schema_ok
                and self.focus_processor.has_remaining
            ):
                jobs.append(("focus-tail", P1, self._focus_tail_job))
            jobs.append(("passive-annotation", P2, lambda: self._passive_annotation_job(result)))
            jobs.append(("passive-generation", P2, lambda: self._passive_generation_job(result)))
            if self.runner is not None:
                for key, priority, job in jobs:
                    self.runner.submit(key, priority, job)

            # As main(): live recognition every two minutes, model-free.
            if self.v2_schema_ok and self.clock.now() - self._last_recognition >= timedelta(
                seconds=wm.workflow_mining.RECOGNIZE_INTERVAL_SECONDS,
            ):
                wm.workflow_mining.recognize(db, self.knowledge_base, now=self.clock.now())
                self._last_recognition = self.clock.now()

        if self.runner is None:
            # The runner's order (the list is already by priority), then its
            # idle hook: the end of the generation phase.
            for _key, _priority, job in sorted(jobs, key=lambda j: j[1]):
                job()
            self.model_scheduler.end_generation()
        # As main(): the heartbeat and the index.
        self._write_status(result)
        with STATE_LOCK:
            wm._write_sops_index(db, self.knowledge_base)
        return result

    def _focus_job(self, result: dict) -> None:
        """main()'s P0 focus job."""
        db = self.db
        scheduler = self.model_scheduler
        scheduler.begin_annotation()
        scheduler.plan_generation(
            urgent=wm._focus_work_ready(self.state_dir, db) is not None,
            due={}, take_batched=False,
        )
        if self.focus_processor is not None and self.v2_schema_ok:
            focus_sops = wm._process_focus_sessions_v2(
                db,
                focus_processor=self.focus_processor,
                screenshots_dir=self.screenshots_dir,
                sop_auto_approve=self.sop_auto_approve,
                behavioral_synthesizer=self.behavioral_synthesizer,
                focus_questioner=self.focus_questioner,
                knowledge_base=self.knowledge_base,
                vector_kb=self.vector_kb,
                ollama_host=self.cfg.vlm.ollama_host,
                qna_model=scheduler.generation_model,
                **self._export_kwargs(),
            )
        else:
            focus_sops = wm._process_focus_sessions(
                db,
                episode_builder=self.episode_builder,
                clipboard_linker=self.clipboard_linker,
                translator=self.translator,
                scorer=self.scorer,
                vlm_queue=self.vlm_queue,
                sop_enhancer=self.sop_enhancer,
                sop_auto_approve=self.sop_auto_approve,
                **self._export_kwargs(),
            )
        result["focus_sops"] = focus_sops
        self._sops_generated += focus_sops
        if focus_sops > 0:
            wm._write_sops_index(db, self.knowledge_base)

    def _focus_tail_job(self) -> None:
        """main()'s P1 focus-tail job."""
        self.model_scheduler.begin_annotation()
        self.focus_processor.annotate_remaining(self.db)

    def _passive_annotation_job(self, result: dict) -> None:
        """main()'s P2 passive annotation job (no VLM fallback jobs, no
        backlog expiry: see the module docstring)."""
        db = self.db
        self.model_scheduler.begin_annotation()
        if self.scene_annotator is not None and not self.v2_schema_ok:
            self.v2_schema_ok = wm._check_v2_schema(db)
        if self.scene_annotator is not None and self.v2_schema_ok:
            self.annotation_budget.roll(self.clock.now())
            ann_stats = wm._process_annotations(
                db, self.scene_annotator, self.screenshots_dir,
                privacy_checker=self.privacy_checker,
                activity_classifier=self.activity_classifier,
                budget=self.annotation_budget,
            )
            result["annotations"] = ann_stats
            self._annotations += ann_stats["annotated"]
            # As main(): the budget names the cap that closed (hourly or daily).
            self._paused_reason = (
                self.annotation_budget.pause_reason()
                if ann_stats["budget_exhausted"] else None
            )
            if ann_stats["annotated"] > 0:
                self.activity_searcher.refresh_index()
        if self.frame_differ is not None and self.v2_schema_ok:
            diff_stats = wm._process_diffs(
                db, self.frame_differ, budget=self.annotation_budget,
            )
            result["diffs"] = diff_stats
            self._diffs += diff_stats["diffs"]

    def _passive_generation_job(self, result: dict) -> None:
        """main()'s P2 passive generation job (passive discovery; not the
        daily jobs: see the module docstring)."""
        passive_due = bool(
            self.task_segmenter is not None
            and self.sop_generator is not None
            and self.v2_schema_ok
            and self._passive_discovery_due()
            and not (self.state_dir / "focus-session.json").exists()
        )
        plan = self.model_scheduler.plan_generation(
            urgent=False, due={"passive_discovery": passive_due},
        )
        result["generation_plan"] = plan
        if passive_due and plan.allows("passive_discovery"):
            with charged_to(self.annotation_budget):
                passive_sops = wm._process_passive_discovery(
                    self.db,
                    knowledge_base=self.knowledge_base,
                    segmenter=self.task_segmenter,
                    sop_generator=self.sop_generator,
                    sop_auto_approve=self.sop_auto_approve,
                    continuity_tracker=self.continuity_tracker,
                    variant_detector=self.variant_detector,
                    evidence_normalizer=self.evidence_normalizer,
                    now=self.clock.now(),
                    **self._export_kwargs(),
                )
            result["passive_sops"] = passive_sops
            self._sops_generated += passive_sops
            self._last_passive_discovery = self.clock.now()
            if passive_sops > 0:
                wm._write_sops_index(self.db, self.knowledge_base)

    def _write_status(self, result: dict) -> None:
        """main()'s heartbeat: ``worker-status.json`` with the running
        totals, the passive annotation budget and pause reason (on AC, so
        only the budget pauses it), the compute counters, the model schedule
        and, with an inference thread, its queue."""
        inference = None
        if self.runner is not None:
            inference = self.runner.status()
            backlog = self.db.count_passive_unannotated()
            inference["annotation_backlog"] = backlog
            inference["lower_capture_cadence"] = self.annotation_budget.asks_lower_capture(backlog)
        wm._write_worker_status(
            started_at=self._started_at,
            events_processed_today=self._events_processed,
            sops_generated=self._sops_generated,
            last_pipeline_duration_ms=None,
            consecutive_errors=0,
            vlm_available=self.scene_annotator is not None,
            vlm_mode="local",
            v2_annotation_enabled=self.scene_annotator is not None,
            v2_annotations_today=self._annotations,
            v2_diffs_today=self._diffs,
            annotation_paused_reason=self._paused_reason,
            annotation_budget=self.annotation_budget.status(),
            model_schedule=self.model_scheduler.status(),
            inference=inference,
        )

    def status(self) -> dict:
        """The last ``worker-status.json`` :meth:`run_tick` wrote."""
        return json.loads((self.state_dir / "worker-status.json").read_text())

    def _passive_discovery_due(self) -> bool:
        # main() seeds its timer when the loop starts: due every 2 hours.
        return self.clock.now() - self._last_passive_discovery >= PASSIVE_DISCOVERY_INTERVAL

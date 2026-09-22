"""Journey driver: one worker poll iteration per ``run_tick()``, on a
controllable clock.

``main()`` builds its components and runs its poll loop inline, so until
WP-S6 splits it, ``JourneyWorker`` builds the same components the same way
(the config readers are ``main``'s own, so ``config.toml`` in the test's
``HOME`` decides models, hosts and feature flags) and ``run_tick()`` calls
the loop body's processors in the loop's order:

1. the trigger processors (retry, approval, failed query, search, recall,
   trust accept / dismiss, staleness reviewed, lifecycle promote, merge,
   drift reviewed, lifecycle dismiss) and the curation queue refresh;
2. the stale focus-session sweep, then ``_process_focus_sessions_v2``
   (or the v1 path when the v2 pipeline is off) and the export trigger;
3. ``run_pipeline`` over unprocessed events, marking them processed and
   persisting VLM jobs;
4. ``_process_annotations`` and ``_process_diffs``;
5. ``_process_passive_discovery`` every 2 hours of *clock* time while no
   focus session is pending;
6. ``_write_sops_index``.

Not driven here, because they only exist inline in ``main()``: the VLM
fallback job runner (needs ``VLMWorker``), the idle-gate shortcut for
passive discovery (reads real system idle time), the daily batch, the
staleness / lifecycle / curation pass and the status heartbeat.  Journeys
that need those wait for WP-S6.

``main()`` logs an exception inside an iteration and carries on with the
next one; ``run_tick()`` lets it propagate so the journey fails at its cause.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agenthandover_worker import main as wm

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
        from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig
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

        self.db_path = Path(db_path)
        self.sops_dir = Path(sops_dir)
        self.clock = clock
        self.state_dir = wm._status_dir()
        self.screenshots_dir = self.state_dir / "screenshots"
        self._last_passive_discovery: datetime | None = None

        flags = wm._read_feature_flags()
        v2_cfg = wm._read_vlm_v2_config()
        self.v2_cfg = v2_cfg
        host = v2_cfg["ollama_host"]

        kb = KnowledgeBase(root=Path(knowledge_dir))
        kb.ensure_structure()
        self.knowledge_base = kb
        self.lifecycle_manager = LifecycleManager(kb)
        self.procedure_writer = ProcedureWriter(
            kb=kb, evidence=EvidenceTracker(knowledge_base=kb),
            lifecycle_manager=self.lifecycle_manager, vector_kb=None, llm_reasoner=None,
        )
        self.kb_export_adapter = KnowledgeBaseExportAdapter(kb)
        self.privacy_checker = PrivacyZoneChecker(config=PrivacyZoneConfig())
        self.staleness_detector = StalenessDetector(kb)
        self.activity_classifier = None
        if flags["activity_classification"]:
            self.activity_classifier = ActivityClassifier(
                profile=kb.get_profile(), policy=UserPolicy(kb),
            )
        self.constraint_manager = ConstraintManager(kb)

        self.vlm_queue = wm.VLMFallbackQueue()
        llm_reasoner = LLMReasoner(
            config=ReasoningConfig(ollama_host=host), vlm_queue=self.vlm_queue,
        )
        if self.activity_classifier is not None:
            self.activity_classifier._llm_reasoner = llm_reasoner
        self.trust_advisor = TrustAdvisor(kb)
        self.continuity_tracker = None
        if flags["continuity_tracking"]:
            self.continuity_tracker = ContinuityTracker(
                kb=kb, matcher=ProcedureMatcher(kb=kb, vector_kb=None),
            )
        self.variant_detector = VariantDetector()
        self.evidence_normalizer = EvidenceNormalizer(variant_detector=self.variant_detector)
        self.focus_max_session_hours = wm._read_float_field("focus", "max_session_hours", 4.0)
        self.behavioral_synthesizer = BehavioralSynthesizer(
            config=SynthesizerConfig(model=v2_cfg["sop_model"], ollama_host=host),
            vlm_queue=self.vlm_queue, knowledge_base=kb,
        )
        self.focus_questioner = FocusQuestioner(llm_reasoner=llm_reasoner)
        self.procedure_curator = None
        if flags["curation"]:
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
            self.sop_writer = ClaudeSkillWriter(skills_dir=wm._skills_dir())
        elif adapter == "openclaw":
            self.sop_writer = wm.OpenClawWriter(workspace_dir=self.sops_dir.parent.parent)
        else:
            raise ValueError(f"adapter {adapter!r} is not driven by journeys")

        # SOP enhancement and the SOP inducer, as main() sets them up.
        self.sop_enhancer = None
        llm_config = wm._read_llm_config()
        if llm_config.get("enhance_sops", True):
            from agenthandover_worker.sop_enhancer import SOPEnhancer, create_llm_backend
            backend = create_llm_backend(llm_config, {
                "mode": wm._read_vlm_config_field("mode", "local"),
                "provider": wm._read_vlm_config_field("provider", ""),
                "model": wm._read_vlm_config_field("model", ""),
                "api_key_env": wm._read_vlm_config_field("api_key_env", ""),
            })
            if backend is not None:
                self.sop_enhancer = SOPEnhancer(
                    backend=backend,
                    max_enhancements_per_day=int(llm_config.get("max_enhancements_per_day", 20)),
                )
        from agenthandover_worker.sop_inducer import SOPInducer
        self.sop_inducer = SOPInducer(min_support=0.05, min_pattern_length=2, vlm_worker=None)

        self.scene_annotator = None
        self.frame_differ = None
        self.focus_processor = None
        self.task_segmenter = None
        self.sop_generator = None
        if v2_cfg["annotation_enabled"]:
            self.scene_annotator = wm.SceneAnnotator(
                config=wm.AnnotationConfig(
                    model=v2_cfg["annotation_model"],
                    ollama_host=host,
                    stale_skip_count=v2_cfg["stale_skip_count"],
                    sliding_window_max_age_sec=v2_cfg["sliding_window_max_age_sec"],
                    timeout_seconds=v2_cfg["annotation_timeout_seconds"],
                ),
                image_embedder=None,
                vector_kb=None,
            )
            self.scene_annotator.preload()
            self.frame_differ = wm.FrameDiffer(
                config=wm.DiffConfig(model=v2_cfg["annotation_model"], ollama_host=host),
            )
            self.sop_generator = wm.SOPGenerator(
                config=wm.SOPGeneratorConfig(model=v2_cfg["sop_model"], ollama_host=host),
            )
            self.focus_processor = wm.FocusProcessor(
                annotator=self.scene_annotator,
                differ=self.frame_differ,
                sop_generator=self.sop_generator,
            )
            self.task_segmenter = wm.TaskSegmenter(
                config=wm.SegmenterConfig(ollama_host=host),
                llm_reasoner=llm_reasoner,
                cluster_label_store=kb.root / "observations" / "cluster_labels.json",
                vector_kb=None,
            )
        self.activity_searcher = ActivitySearcher(db_path=self.db_path)
        self.sop_auto_approve = wm._read_sop_config()["auto_approve"]

        # Vector KB with main()'s health check: unavailable -> None.
        self.vector_kb = None
        vector_kb = VectorKB(self.db_path, VectorKBConfig(ollama_host=host))
        try:
            embedding = vector_kb.compute_embeddings(["health check"])
        except ConnectionError:
            embedding = None
        if embedding and embedding[0]:
            self.vector_kb = vector_kb
        else:
            vector_kb.close()

        self.db = wm.WorkerDB(self.db_path)
        self.v2_schema_ok = (
            self.scene_annotator is not None and wm._check_v2_schema(self.db)
        )
        wm._write_sops_index(self.db, kb, force=True)

    def close(self) -> None:
        self.db.close()
        if self.vector_kb is not None:
            self.vector_kb.close()
        self.activity_searcher.close()

    def __enter__(self) -> "JourneyWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run_tick(self) -> dict:
        """One poll iteration of main()'s loop body (see module docstring).

        Returns what the iteration did, for assertions.
        """
        db = self.db
        writers = {
            "openclaw_writer": self.sop_writer,
            "skill_md_writer": self.skill_md_writer,
            "claude_skill_writer": self.claude_skill_writer,
            "index_generator": self.index_generator,
            "procedure_writer": self.procedure_writer,
            "kb_export_adapter": self.kb_export_adapter,
        }
        if self.vector_kb is not None:
            self.vector_kb.reset_cycle_budget()

        wm._process_retry_triggers(
            db, focus_processor=self.focus_processor, sop_generator=self.sop_generator,
            screenshots_dir=self.screenshots_dir, sop_auto_approve=self.sop_auto_approve,
            **writers,
        )
        wm._process_approval_triggers(db, vector_kb=self.vector_kb, **writers)
        wm._process_failed_query_trigger(db)
        wm._process_search_trigger(self.activity_searcher)
        wm._process_recall_trigger(self.activity_searcher)
        wm._process_trust_accept_trigger(self.trust_advisor, self.constraint_manager)
        wm._process_trust_dismiss_trigger(self.trust_advisor)
        wm._process_staleness_reviewed_trigger(self.staleness_detector, self.knowledge_base)
        wm._process_lifecycle_promote_trigger(lifecycle_manager=self.lifecycle_manager)
        wm._process_merge_trigger(procedure_curator=self.procedure_curator)
        wm._process_drift_reviewed_trigger(procedure_curator=self.procedure_curator)
        wm._process_lifecycle_dismiss_trigger(procedure_curator=self.procedure_curator)
        wm._write_curation_queue(self.procedure_curator, force=True)

        wm._sweep_stale_focus_session(self.state_dir, self.focus_max_session_hours)
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
                ollama_host=self.v2_cfg["ollama_host"],
                qna_model=self.v2_cfg["sop_model"],
                **writers,
            )
        else:
            focus_sops = wm._process_focus_sessions(
                db,
                episode_builder=self.episode_builder,
                clipboard_linker=self.clipboard_linker,
                translator=self.translator,
                scorer=self.scorer,
                vlm_queue=self.vlm_queue,
                sop_inducer=self.sop_inducer,
                sop_enhancer=self.sop_enhancer,
                sop_auto_approve=self.sop_auto_approve,
                **writers,
            )
        if focus_sops > 0:
            wm._write_sops_index(db, self.knowledge_base)
        wm._check_export_trigger(
            openclaw_writer=self.sop_writer, sops_dir=self.sops_dir,
            procedure_writer=self.procedure_writer,
        )

        unprocessed = db.get_unprocessed_events(limit=100)
        summary = None
        if unprocessed:
            summary = wm.run_pipeline(
                unprocessed,
                episode_builder=self.episode_builder,
                clipboard_linker=self.clipboard_linker,
                pruner=self.pruner,
                translator=self.translator,
                scorer=self.scorer,
                vlm_queue=self.vlm_queue,
                sop_inducer=self.sop_inducer,
                sop_enhancer=self.sop_enhancer,
                db=db,
                **writers,
            )
            if summary["sops_exported"] > 0:
                wm._write_sops_index(db, self.knowledge_base)
            event_ids = [ev["id"] for ev in unprocessed if "id" in ev]
            if event_ids:
                db.mark_events_processed(event_ids)
            if summary["vlm_enqueued"] > 0:
                wm._persist_vlm_jobs(db, self.vlm_queue)

        ann_stats = diff_stats = None
        if self.scene_annotator is not None and not self.v2_schema_ok:
            self.v2_schema_ok = wm._check_v2_schema(db)
        if self.scene_annotator is not None and self.v2_schema_ok:
            ann_stats = wm._process_annotations(
                db, self.scene_annotator, self.screenshots_dir,
                privacy_checker=self.privacy_checker,
                activity_classifier=self.activity_classifier,
            )
            if ann_stats["annotated"] > 0:
                self.activity_searcher.refresh_index()
        if self.frame_differ is not None and self.v2_schema_ok:
            diff_stats = wm._process_diffs(db, self.frame_differ)

        passive_sops = None
        if (
            self.task_segmenter is not None
            and self.sop_generator is not None
            and self.v2_schema_ok
            and self._passive_discovery_due()
            and not (self.state_dir / "focus-session.json").exists()
        ):
            passive_sops = wm._process_passive_discovery(
                db,
                segmenter=self.task_segmenter,
                sop_generator=self.sop_generator,
                sop_auto_approve=self.sop_auto_approve,
                continuity_tracker=self.continuity_tracker,
                variant_detector=self.variant_detector,
                evidence_normalizer=self.evidence_normalizer,
                **writers,
            )
            self._last_passive_discovery = self.clock.now()
            if passive_sops > 0:
                wm._write_sops_index(db, self.knowledge_base)

        wm._write_sops_index(db, self.knowledge_base)
        return {
            "focus_sops": focus_sops,
            "pipeline": summary,
            "annotations": ann_stats,
            "diffs": diff_stats,
            "passive_sops": passive_sops,
        }

    def _passive_discovery_due(self) -> bool:
        # main() starts its timer at 0 on a monotonic clock, so the first
        # iteration after start-up is due; later ones every 2 hours.
        last = self._last_passive_discovery
        return last is None or self.clock.now() - last >= PASSIVE_DISCOVERY_INTERVAL

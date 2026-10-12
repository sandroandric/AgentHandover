"""The worker's context: every component and the loop's state, built once.

:func:`build_context` builds what ``main()`` used to build inline, from one
``load_config()`` read (``config.py``): the knowledge base, the model
clients, the pipelines and writers, the passive annotation budget, the
reasoning budget and the model scheduler, and opens the daemon's database.
:func:`start` then does the one-time work before the first tick (orphan
screenshots, the v2 schema check, the loop's timers and persisted state,
the trigger handlers, the Skills index).  The stages (``stages/``) read and update the context; the
loop's counters and timers live on it, so ``main()`` and the journey driver
run the same :func:`agenthandover_worker.stages.tick`.

Services a journey must not start (the VLM backend check, which probes the
local Ollama, and the query API server, which binds a port) are started by
``main()`` and set on the context afterwards.

Time comes from the context: :attr:`WorkerContext.now` (aware UTC),
:meth:`WorkerContext.local_now` (the user's wall time, for the local day)
and :attr:`WorkerContext.monotonic` (the loop's timers).  ``main()`` uses
the system clocks; the journey driver passes its virtual clock.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable

from agenthandover_worker import main as wm
from agenthandover_worker.annotation_cache import AnnotationCache
from agenthandover_worker.clipboard_linker import ClipboardLinker
from agenthandover_worker.confidence import ConfidenceScorer
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.exporter import IndexGenerator
from agenthandover_worker.focus_processor import FocusProcessor
from agenthandover_worker.frame_differ import DiffConfig, FrameDiffer
from agenthandover_worker.loop_guard import EventRetryGuard
from agenthandover_worker.model_scheduler import ModelScheduler
from agenthandover_worker.negative_demo import NegativeDemoPruner
from agenthandover_worker.ollama import OllamaClient, OllamaError
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.scene_annotator import AnnotationConfig, SceneAnnotator
from agenthandover_worker.scheduler import (
    AnnotationBudget,
    IdleJobGate,
    ReasoningBudget,
    SchedulerConfig,
)
from agenthandover_worker.sop_generator import SOPGenerator, SOPGeneratorConfig
from agenthandover_worker.task_segmenter import SegmenterConfig, TaskSegmenter
from agenthandover_worker.translator import SemanticTranslator

logger = logging.getLogger("agenthandover_worker")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class WorkerContext:
    """The worker's components and loop state (see the module docstring).

    Components are built by :func:`build_context`; the fields after
    ``trigger_handlers`` are the loop's state, which the stages update.
    """

    cfg: Any
    db_path: Path
    knowledge_dir: Path
    sops_dir: Path
    adapter: str
    poll_interval: float
    screenshots_dir: Path

    knowledge_base: Any
    ops_telemetry: Any
    reasoning_budget: Any
    llm_reasoner: Any
    vector_kb: Any
    lifecycle_manager: Any
    procedure_writer: Any
    kb_export_adapter: Any
    privacy_checker: Any
    staleness_detector: Any
    profile_builder: Any
    activity_classifier: Any
    pattern_detector: Any
    constraint_manager: Any
    runtime_validator: Any
    execution_monitor: Any
    correction_detector: Any
    trust_advisor: Any
    session_linker: Any
    procedure_matcher: Any
    daily_processor: Any
    continuity_tracker: Any
    variant_detector: Any
    evidence_normalizer: Any
    behavioral_synthesizer: Any
    procedure_curator: Any
    episode_builder: Any
    clipboard_linker: Any
    pruner: Any
    translator: Any
    scorer: Any
    index_generator: Any
    sop_writer: Any
    skill_md_writer: Any
    claude_skill_writer: Any
    sop_enhancer: Any
    annotation_cache: Any
    scene_annotator: Any
    frame_differ: Any
    sop_generator: Any
    focus_processor: Any
    task_segmenter: Any
    activity_searcher: Any
    procedure_verifier: Any
    bundle_compiler: Any
    idle_gate: Any
    annotation_budget: Any
    model_scheduler: Any
    db: Any
    # Memory's writer (``memory.MemoryWriter``); its job runs only while
    # ``[memory] enabled``, its daily purge always.
    memory_writer: Any = None

    now: Callable[[], datetime] = _utc_now
    # The user's time zone for the local day; None is the system's.
    local_tz: tzinfo | None = None
    monotonic: Callable[[], float] = time.monotonic

    # The once-per-day pass (stages/daily.py).  Only the journey driver
    # turns it off: no cassette holds the daily pass's model calls.
    run_daily_jobs: bool = True

    # Set by main() after the build (services a journey does not start).
    vlm_available: bool = False
    # The configured models Ollama does not serve (None: Ollama did not
    # answer); the app's menu offers their download.
    missing_models: list[str] | None = None
    query_api_server: Any = None
    # The inference thread; None in the journey driver's inline mode.
    runner: Any = None

    # Set by start().
    evidence_extractor: Any = None
    trigger_handlers: dict[str, Callable[[dict], None]] = field(default_factory=dict)

    # --- Loop state ---
    started_at: str = ""
    total_events_processed: int = 0
    total_sops_generated: int = 0
    total_passive_sops: int = 0
    last_pipeline_duration_ms: int | None = None
    total_v2_annotations: int = 0
    total_v2_diffs: int = 0
    v2_schema_ok: bool = False
    annotation_paused_reason: str | None = None
    current_interval: float = 0.0
    consecutive_errors: int = 0
    event_guard: Any = None
    last_idle_log_time: float = 0.0
    # The unprocessed-event count of the last poll logged at INFO.
    last_logged_poll_count: int | None = None
    # True when the control thread's last poll found no events (the user
    # is away): passive discovery may run early.
    control_idle: bool = False
    last_passive_discovery: float = 0.0
    last_recognition: float = 0.0
    daily_jobs_date: str = ""
    saved_budget_json: str = ""
    saved_reasoning_budget_json: str = ""

    @property
    def v2_annotation_enabled(self) -> bool:
        return self.cfg.vlm.annotation_enabled

    def local_now(self) -> datetime:
        """The user's wall time (aware), whose date is the local day."""
        return self.now().astimezone(self.local_tz)

    def export_kwargs(self) -> dict:
        """The writers every Skill-producing path takes."""
        return {
            "openclaw_writer": self.sop_writer,
            "skill_md_writer": self.skill_md_writer,
            "claude_skill_writer": self.claude_skill_writer,
            "index_generator": self.index_generator,
            "procedure_writer": self.procedure_writer,
            "kb_export_adapter": self.kb_export_adapter,
        }

    def close(self) -> None:
        """Close the database and the stores that hold connections."""
        self.db.close()
        if self.vector_kb is not None:
            self.vector_kb.close()
        if self.activity_searcher is not None:
            self.activity_searcher.close()
        if self.annotation_cache is not None:
            self.annotation_cache.close()
        if self.memory_writer is not None:
            self.memory_writer.close()


def build_context(
    cfg: Any,
    *,
    db_path: Path,
    knowledge_dir: Path,
    sops_dir: Path,
    adapter: str,
    poll_interval: float = wm.POLL_INTERVAL_SECONDS,
    json_export: bool = False,
    llm_overrides: dict | None = None,
    now: Callable[[], datetime] = _utc_now,
    local_tz: tzinfo | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    idle_gate: Any = None,
    scheduler_probes: dict | None = None,
) -> WorkerContext:
    """Build the worker's components from *cfg* and open the database.

    *llm_overrides* are the command line's ``[llm]`` overrides.
    *idle_gate* defaults to the system's ``IdleJobGate``;
    *scheduler_probes* are ``ModelScheduler`` keyword arguments (its clock
    and memory, swap and disk probes), the system's by default.
    """
    from agenthandover_worker.activity_classifier import ActivityClassifier
    from agenthandover_worker.behavioral_synthesizer import BehavioralSynthesizer, SynthesizerConfig
    from agenthandover_worker.bundle_compiler import BundleCompiler
    from agenthandover_worker.config_validator import ConfigValidator
    from agenthandover_worker.constraint_manager import ConstraintManager
    from agenthandover_worker.continuity_tracker import ContinuityTracker
    from agenthandover_worker.correction_detector import CorrectionDetector
    from agenthandover_worker.daily_processor import DailyBatchProcessor
    from agenthandover_worker.evidence_normalizer import EvidenceNormalizer
    from agenthandover_worker.evidence_tracker import EvidenceTracker
    from agenthandover_worker.execution_monitor import ExecutionMonitor
    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.knowledge_export_adapter import KnowledgeBaseExportAdapter
    from agenthandover_worker.lifecycle_manager import LifecycleManager
    from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig, annotation_reasoner
    from agenthandover_worker.ops_telemetry import OpsTelemetry
    from agenthandover_worker.pattern_detector import PatternDetector
    from agenthandover_worker.privacy_zones import PrivacyZoneChecker, PrivacyZoneConfig
    from agenthandover_worker.procedure_curator import ProcedureCurator
    from agenthandover_worker.procedure_matcher import ProcedureMatcher
    from agenthandover_worker.procedure_verifier import ProcedureVerifier
    from agenthandover_worker.procedure_writer import ProcedureWriter
    from agenthandover_worker.profile_builder import ProfileBuilder
    from agenthandover_worker.runtime_validator import RuntimeValidator
    from agenthandover_worker.session_linker import SessionLinker
    from agenthandover_worker.staleness_detector import StalenessDetector
    from agenthandover_worker.trust_advisor import TrustAdvisor
    from agenthandover_worker.user_policy import UserPolicy
    from agenthandover_worker.variant_detector import VariantDetector

    vlm_cfg = cfg.vlm
    host = vlm_cfg.ollama_host
    features = cfg.features

    knowledge_base = KnowledgeBase(root=knowledge_dir, db_path=db_path)
    knowledge_base.ensure_structure()
    # Procedures live in the database: a knowledge base from before
    # (v0.4) has its files imported once; the files stay as the export.
    knowledge_base.import_procedure_files()
    knowledge_base.export_procedures()

    try:
        for issue in ConfigValidator().validate(cfg.raw):
            log_fn = logger.error if issue.severity == "error" else logger.warning
            log_fn("Config [%s.%s]: %s", issue.section, issue.key, issue.message)
    except Exception:
        logger.debug("Config validation skipped", exc_info=True)
    logger.info(
        "Feature flags: %s",
        [k for k, v in vars(features).items() if not v] or "all enabled",
    )

    # The shared reasoner, before the modules that take it, and the daily
    # budget it shares with the behavioral synthesizer (on the local day).
    reasoning_budget = ReasoningBudget.from_config(
        vlm_cfg, clock=lambda: now().astimezone(local_tz),
    )
    llm_reasoner = LLMReasoner(
        config=ReasoningConfig(model=vlm_cfg.reasoner_model, ollama_host=host),
        budget=reasoning_budget,
    )
    vector_kb = _vector_kb(cfg, db_path)

    evidence_tracker = EvidenceTracker(knowledge_base=knowledge_base)
    lifecycle_manager = LifecycleManager(knowledge_base)
    staleness_detector = StalenessDetector(knowledge_base)
    activity_classifier = None
    if features.activity_classification:
        activity_classifier = ActivityClassifier(
            profile=knowledge_base.get_profile(),
            policy=UserPolicy(knowledge_base),
            # It classifies frames in the annotation phase: annotation model.
            llm_reasoner=annotation_reasoner(vlm_cfg.annotation_model, host),
        )
    execution_monitor = ExecutionMonitor(knowledge_base)
    trust_advisor = TrustAdvisor(knowledge_base, execution_monitor=execution_monitor)
    procedure_matcher = ProcedureMatcher(kb=knowledge_base, vector_kb=vector_kb)
    variant_detector = VariantDetector()
    evidence_normalizer = EvidenceNormalizer(variant_detector=variant_detector)
    behavioral_synthesizer = BehavioralSynthesizer(
        config=SynthesizerConfig(model=vlm_cfg.sop_model, ollama_host=host),
        budget=reasoning_budget,
        knowledge_base=knowledge_base,
    )
    procedure_curator = None
    if features.curation:
        from agenthandover_worker.quality_judge import QualityJudge

        procedure_curator = ProcedureCurator(
            kb=knowledge_base,
            staleness_detector=staleness_detector,
            trust_advisor=trust_advisor,
            lifecycle_manager=lifecycle_manager,
            evidence_normalizer=evidence_normalizer,
            quality_judge=QualityJudge(llm_reasoner),
            llm_reasoner=llm_reasoner,
        )
    logger.info("Knowledge base initialized: %s", knowledge_dir)

    sop_writer, skill_md_writer, claude_skill_writer = _writers(
        cfg, adapter, sops_dir, json_export,
    )
    # Before the v2 pipeline: both ask Ollama (the model list, then the
    # annotation model's preload), in this order.
    sop_enhancer = _sop_enhancer(cfg, llm_overrides or {})
    v2 = _v2_pipeline(cfg, db_path, knowledge_base, llm_reasoner, vector_kb)

    activity_searcher = None
    try:
        from agenthandover_worker.activity_search import ActivitySearcher

        activity_searcher = ActivitySearcher(db_path=db_path)
        logger.info("Activity searcher initialized")
    except Exception:
        logger.debug("Activity searcher not available", exc_info=True)

    runtime_validator = RuntimeValidator(knowledge_base) if features.runtime_validation else None
    procedure_verifier = ProcedureVerifier(knowledge_base, runtime_validator=runtime_validator)
    bundle_adapters = {
        name: writer
        for name, writer in (
            ("openclaw", sop_writer),
            ("skill_md", skill_md_writer),
            ("claude_skill", claude_skill_writer),
        )
        if writer is not None
    }
    logger.info("SOP auto_approve: %s", cfg.sop.auto_approve)

    model_scheduler = ModelScheduler(
        OllamaClient(host),
        annotation_model=vlm_cfg.annotation_model,
        sop_model=vlm_cfg.sop_model,
        **(scheduler_probes or {}),
    )
    model_scheduler.register(v2["sop_generator"], behavioral_synthesizer, llm_reasoner)
    privacy_checker = PrivacyZoneChecker(config=_privacy_zones(cfg, PrivacyZoneConfig))
    memory_writer = _memory_writer(
        cfg, db_path, privacy_checker, model_scheduler, v2["scene_annotator"], now, monotonic,
    )

    return WorkerContext(
        cfg=cfg,
        db_path=db_path,
        knowledge_dir=knowledge_dir,
        sops_dir=sops_dir,
        adapter=adapter,
        poll_interval=poll_interval,
        screenshots_dir=wm._status_dir() / "screenshots",
        knowledge_base=knowledge_base,
        ops_telemetry=OpsTelemetry(knowledge_base),
        reasoning_budget=reasoning_budget,
        llm_reasoner=llm_reasoner,
        vector_kb=vector_kb,
        lifecycle_manager=lifecycle_manager,
        procedure_writer=ProcedureWriter(
            kb=knowledge_base, evidence=evidence_tracker,
            lifecycle_manager=lifecycle_manager, vector_kb=vector_kb,
            llm_reasoner=llm_reasoner,
        ),
        kb_export_adapter=KnowledgeBaseExportAdapter(knowledge_base),
        privacy_checker=privacy_checker,
        staleness_detector=staleness_detector,
        profile_builder=ProfileBuilder(knowledge_base),
        activity_classifier=activity_classifier,
        pattern_detector=PatternDetector(knowledge_base),
        constraint_manager=ConstraintManager(knowledge_base),
        runtime_validator=runtime_validator,
        execution_monitor=execution_monitor,
        correction_detector=CorrectionDetector(knowledge_base, llm_reasoner=llm_reasoner),
        trust_advisor=trust_advisor,
        session_linker=SessionLinker(knowledge_base, llm_reasoner=llm_reasoner, vector_kb=vector_kb),
        procedure_matcher=procedure_matcher,
        daily_processor=DailyBatchProcessor(
            knowledge_base=knowledge_base, matcher=procedure_matcher,
        ),
        continuity_tracker=(
            ContinuityTracker(kb=knowledge_base, matcher=procedure_matcher)
            if features.continuity_tracking else None
        ),
        variant_detector=variant_detector,
        evidence_normalizer=evidence_normalizer,
        behavioral_synthesizer=behavioral_synthesizer,
        procedure_curator=procedure_curator,
        episode_builder=EpisodeBuilder(),
        clipboard_linker=ClipboardLinker(),
        pruner=NegativeDemoPruner(),
        translator=SemanticTranslator(),
        scorer=ConfidenceScorer(),
        index_generator=IndexGenerator(),
        sop_writer=sop_writer,
        skill_md_writer=skill_md_writer,
        claude_skill_writer=claude_skill_writer,
        sop_enhancer=sop_enhancer,
        annotation_cache=v2["annotation_cache"],
        scene_annotator=v2["scene_annotator"],
        frame_differ=v2["frame_differ"],
        sop_generator=v2["sop_generator"],
        focus_processor=v2["focus_processor"],
        task_segmenter=v2["task_segmenter"],
        activity_searcher=activity_searcher,
        procedure_verifier=procedure_verifier,
        bundle_compiler=BundleCompiler(
            kb=knowledge_base,
            lifecycle=lifecycle_manager,
            verifier=procedure_verifier,
            adapters=bundle_adapters,
        ),
        idle_gate=idle_gate or IdleJobGate(
            SchedulerConfig(**cfg.idle_jobs.scheduler_kwargs()),
        ),
        annotation_budget=AnnotationBudget.from_config(vlm_cfg),
        model_scheduler=model_scheduler,
        db=WorkerDB(db_path),
        memory_writer=memory_writer,
        now=now,
        local_tz=local_tz,
        monotonic=monotonic,
        current_interval=poll_interval,
        event_guard=EventRetryGuard(max_batch=100),
        started_at=now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def start(ctx: WorkerContext) -> None:
    """The one-time work before the first tick: orphan screenshots, the
    evidence extractor, the v2 schema check, the loop's timers and
    persisted state, the trigger handlers and the Skills index."""
    from agenthandover_worker import trigger_spool
    from agenthandover_worker.evidence_extractor import EvidenceExtractor
    from agenthandover_worker.stages import control

    db = ctx.db
    wm._sweep_orphan_screenshots(
        db, ctx.screenshots_dir,
        keep=ctx.memory_writer.held_screenshot_paths() if ctx.memory_writer else frozenset(),
    )
    ctx.evidence_extractor = EvidenceExtractor(kb=ctx.knowledge_base, db=db)

    # Checked once before the loop so a stopped focus recording takes the
    # v2 path on the very first tick (the passive annotation job re-checks
    # it in case the daemon migrates later).
    if ctx.scene_annotator is not None:
        ctx.v2_schema_ok = wm._check_v2_schema(db)
        if ctx.v2_schema_ok:
            logger.info("v2 schema detected — scene annotation pipeline active")
        else:
            logger.info(
                "v2 schema not found — scene annotation disabled "
                "(run daemon to apply migration)"
            )

    # Timers start now; once-per-day jobs are keyed on the local date they
    # last ran, persisted so a restart does not rerun them.
    ctx.last_idle_log_time = ctx.monotonic()
    ctx.last_passive_discovery = ctx.monotonic()
    ctx.last_recognition = ctx.monotonic()
    ctx.daily_jobs_date = db.get_state(wm.DAILY_JOBS_STATE_KEY) or ""
    ctx.annotation_budget.load(db.get_state(wm.ANNOTATION_BUDGET_STATE_KEY))
    ctx.saved_budget_json = ctx.annotation_budget.to_json()
    ctx.reasoning_budget.load(db.get_state(wm.REASONING_BUDGET_STATE_KEY))
    ctx.saved_reasoning_budget_json = ctx.reasoning_budget.to_json()

    ctx.trigger_handlers = control.trigger_handlers(ctx)
    assert set(ctx.trigger_handlers) == set(trigger_spool.KINDS)
    trigger_spool.migrate_legacy(wm._status_dir())

    # At startup the app sees the latest data (including new fields).
    wm._write_sops_index(db, ctx.knowledge_base, force=True)


def _vector_kb(cfg: Any, db_path: Path) -> Any:
    """The vector KB when its embedding model answers, else None."""
    try:
        from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig

        vkb_config = VectorKBConfig(
            embedding_model=cfg.embedding.model,
            ollama_host=cfg.vlm.ollama_host,
            retention_days=cfg.embedding.retention_days,
            max_rows=cfg.embedding.max_rows,
        )
        vector_kb = VectorKB(db_path, vkb_config)
        try:
            test_emb = vector_kb.compute_embeddings(["health check"])
        except (ConnectionError, OllamaError):
            logger.warning(
                "Vector KB model '%s' not available — "
                "run 'ollama pull %s' to enable semantic search",
                vkb_config.embedding_model, vkb_config.embedding_model,
            )
            vector_kb.close()
            return None
        if test_emb and test_emb[0]:
            logger.info(
                "Vector KB ready (model=%s, dim=%d, vectors=%d)",
                vkb_config.embedding_model, len(test_emb[0]), vector_kb.count(),
            )
            return vector_kb
        logger.warning(
            "Vector KB model '%s' returned empty embeddings — "
            "run 'ollama pull %s' to enable semantic search",
            vkb_config.embedding_model, vkb_config.embedding_model,
        )
        vector_kb.close()
    except Exception:
        logger.debug("Vector KB initialization failed — semantic search disabled", exc_info=True)
    return None


def _memory_writer(
    cfg: Any,
    db_path: Path,
    privacy_checker: Any,
    model_scheduler: Any,
    scene_annotator: Any,
    now: Callable[[], datetime],
    monotonic: Callable[[], float],
) -> Any:
    """Memory's writer, and the annotator asking it before a screenshot is
    deleted (``memory`` module docstring)."""
    from agenthandover_worker.memory import MemoryWriter, ollama_embedder

    memory = cfg.memory
    data_dir = wm._status_dir()
    writer = MemoryWriter(
        db_path,
        memory,
        embed=ollama_embedder(cfg.vlm.ollama_host, memory.model, memory.dimensions),
        screenshots_dir=data_dir / "screenshots",
        data_dir=data_dir,
        privacy_checker=privacy_checker,
        blocker=model_scheduler.embedding_blocker,
        now=now,
        monotonic=monotonic,
    )
    if scene_annotator is not None:
        scene_annotator.keep_screenshot = writer.holds
    if memory.enabled:
        logger.info(
            "Memory on (%s, %d dimensions, %d frames/hour)",
            memory.model, memory.dimensions, memory.max_frames_per_hour,
        )
    return writer


def _privacy_zones(cfg: Any, zone_config: type) -> Any:
    """``[privacy.zones]`` from config.toml, defaults when unset or invalid."""
    try:
        if cfg.privacy_zones:
            zones = zone_config.from_zones(cfg.privacy_zones)
            logger.info(
                "Privacy zones loaded: %d blocked, %d metadata_only, %d blocked_urls",
                len(zones.blocked), len(zones.metadata_only), len(zones.blocked_urls),
            )
            return zones
    except Exception:
        logger.debug("Privacy zone config parsing failed, using defaults", exc_info=True)
    return zone_config()


def _writers(cfg: Any, adapter: str, sops_dir: Path, json_export: bool) -> tuple:
    """(primary writer, SKILL.md writer, Claude Skill writer) for *adapter*."""
    workspace = sops_dir.parent.parent
    if adapter == "generic":
        from agenthandover_worker.generic_writer import GenericWriter

        return GenericWriter(output_dir=sops_dir, json_export=json_export), None, None
    if adapter == "skill-md":
        from agenthandover_worker.skill_md_writer import SkillMdWriter

        return SkillMdWriter(workspace_dir=workspace), None, None
    if adapter == "claude-skill":
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter

        return ClaudeSkillWriter(skills_dir=cfg.export.skills_dir), None, None
    if adapter == "all":
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
        from agenthandover_worker.skill_md_writer import SkillMdWriter

        return (
            OpenClawWriter(workspace_dir=workspace),
            SkillMdWriter(workspace_dir=workspace),
            ClaudeSkillWriter(skills_dir=cfg.export.skills_dir),
        )
    return OpenClawWriter(workspace_dir=workspace), None, None


def _sop_enhancer(cfg: Any, overrides: dict) -> Any:
    """The LLM SOP enhancer (the v1 focus path's) when enabled: the local
    SOP model through Ollama."""
    llm_config = dataclasses.asdict(cfg.llm)
    llm_config.update(overrides)
    if not llm_config.get("enhance_sops", True):
        logger.info("SOP enhancement disabled by config")
        return None
    from agenthandover_worker.sop_enhancer import SOPEnhancer, create_llm_backend

    llm_backend = create_llm_backend(
        llm_config, ollama_host=cfg.vlm.ollama_host, sop_model=cfg.vlm.sop_model,
    )
    if llm_backend is None:
        return None
    logger.info("SOP enhancement enabled (%s via Ollama)", llm_backend.model)
    return SOPEnhancer(
        backend=llm_backend,
        max_enhancements_per_day=int(llm_config.get("max_enhancements_per_day", 20)),
    )


def _v2_pipeline(
    cfg: Any, db_path: Path, knowledge_base: Any, llm_reasoner: Any, vector_kb: Any,
) -> dict:
    """The v2 scene annotation pipeline's components (all None when
    ``[vlm] annotation_enabled`` is off)."""
    vlm_cfg = cfg.vlm
    host = vlm_cfg.ollama_host
    parts: dict = dict.fromkeys((
        "annotation_cache", "scene_annotator", "frame_differ",
        "sop_generator", "focus_processor", "task_segmenter",
    ))
    if not vlm_cfg.annotation_enabled:
        logger.info("v2 scene annotation pipeline disabled by config")
        return parts

    parts["annotation_cache"] = AnnotationCache(db_path)
    scene_annotator = SceneAnnotator(
        config=AnnotationConfig(
            model=vlm_cfg.annotation_model,
            ollama_host=host,
            sliding_window_max_age_sec=vlm_cfg.sliding_window_max_age_sec,
            timeout_seconds=vlm_cfg.annotation_timeout_seconds,
        ),
        cache=parts["annotation_cache"],
    )
    # Load the annotation model now (with keep_alive) so the first real
    # frame meets a warm model.  A failure only logs and starts the
    # transport backoff; annotation retries after it.
    scene_annotator.preload()
    frame_differ = FrameDiffer(
        config=DiffConfig(model=vlm_cfg.annotation_model, ollama_host=host),
    )
    sop_generator = SOPGenerator(
        config=SOPGeneratorConfig(model=vlm_cfg.sop_model, ollama_host=host),
    )
    parts.update(
        scene_annotator=scene_annotator,
        frame_differ=frame_differ,
        sop_generator=sop_generator,
        # Focus recordings: annotate -> diff -> generate the Skill.
        focus_processor=FocusProcessor(
            annotator=scene_annotator, differ=frame_differ, sop_generator=sop_generator,
        ),
        # Passive discovery's segmenter.
        task_segmenter=TaskSegmenter(
            config=SegmenterConfig(embedding_model=cfg.embedding.model, ollama_host=host),
            llm_reasoner=llm_reasoner,
            cluster_label_store=knowledge_base.root / "observations" / "cluster_labels.json",
            vector_kb=vector_kb,
        ),
    )
    logger.info(
        "v2 scene annotation pipeline enabled (annotation=%s, sop=%s, window=%ds)",
        vlm_cfg.annotation_model, vlm_cfg.sop_model, vlm_cfg.sliding_window_max_age_sec,
    )
    return parts

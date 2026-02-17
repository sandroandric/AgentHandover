"""OpenMimic apprentice worker entry-point.

Provides the main loop that reads from the daemon's SQLite database
and drives episode building, semantic translation, confidence scoring,
VLM fallback enqueuing, SOP induction, formatting, and export.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from oc_apprentice_worker.clipboard_linker import ClipboardLinker
from oc_apprentice_worker.confidence import ConfidenceScorer, is_native_app_context
from oc_apprentice_worker.db import WorkerDB
from oc_apprentice_worker.deep_scan import DeepScanner
from oc_apprentice_worker.episode_builder import EpisodeBuilder
from oc_apprentice_worker.exporter import IndexGenerator
from oc_apprentice_worker.negative_demo import NegativeDemoPruner
from oc_apprentice_worker.openclaw_writer import OpenClawWriter
from oc_apprentice_worker.scheduler import IdleJobGate, SchedulerConfig
from oc_apprentice_worker.translator import SemanticTranslator
from oc_apprentice_worker.vlm_queue import VLMFallbackQueue, VLMJob

logger = logging.getLogger("oc_apprentice_worker")

# Default paths assume a single-user system. Use --db-path and --sops-dir
# CLI args to override for multi-user or containerized deployments.
# Must match the daemon's default — see crates/daemon/src/main.rs.
import platform as _platform

if _platform.system() == "Darwin":
    DEFAULT_DB_PATH = (
        Path.home() / "Library" / "Application Support" / "oc-apprentice" / "events.db"
    )
else:
    DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "oc-apprentice" / "events.db"
DEFAULT_SOPS_DIR = Path.home() / ".openclaw" / "workspace" / "memory" / "apprentice"
POLL_INTERVAL_SECONDS = 2.0
VLM_REJECT_THRESHOLD = 0.60


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oc-apprentice-worker",
        description="OpenMimic apprentice worker process",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=(
            "Path to the daemon's SQLite database "
            f"(default: {DEFAULT_DB_PATH})"
        ),
    )
    parser.add_argument(
        "--sops-dir",
        type=Path,
        default=DEFAULT_SOPS_DIR,
        help=(
            "Path to the SOP output directory "
            f"(default: {DEFAULT_SOPS_DIR})"
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_SECONDS,
        help=(
            "Seconds between poll cycles "
            f"(default: {POLL_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def run_pipeline(
    events: list[dict],
    *,
    episode_builder: EpisodeBuilder,
    clipboard_linker: ClipboardLinker,
    pruner: NegativeDemoPruner,
    translator: SemanticTranslator,
    scorer: ConfidenceScorer,
    vlm_queue: VLMFallbackQueue,
    openclaw_writer: OpenClawWriter,
    index_generator: IndexGenerator,
    sop_inducer: object | None = None,
) -> dict:
    """Run the full D->E->F pipeline on a batch of events.

    Returns a summary dict with counts and results for monitoring.
    """
    summary: dict = {
        "events_in": len(events),
        "episodes": 0,
        "positive_events": 0,
        "negative_events": 0,
        "translations": 0,
        "vlm_enqueued": 0,
        "sops_induced": 0,
        "sops_exported": 0,
    }

    if not events:
        return summary

    # Step D1: Build episodes from raw events
    episodes = episode_builder.process_events(events)
    summary["episodes"] = len(episodes)
    logger.info("Built %d episodes from %d events", len(episodes), len(events))

    # Step D2: Link clipboard copy-paste pairs across episode events
    all_episode_events = []
    for ep in episodes:
        all_episode_events.extend(ep.events)
    clipboard_links = clipboard_linker.find_links(all_episode_events)
    logger.debug("Found %d clipboard links", len(clipboard_links))

    # Build a set of paste event IDs that have clipboard provenance
    paste_ids_with_provenance: set[str] = set()
    for link in clipboard_links:
        paste_ids_with_provenance.add(link.paste_event_id)

    # Step D3: Prune negative demonstrations per episode
    positive_episodes_events: list[list[dict]] = []
    total_positive = 0
    total_negative = 0

    for ep in episodes:
        prune_result = pruner.prune(ep.events)
        total_positive += len(prune_result.positive_events)
        total_negative += len(prune_result.negative_events)
        if prune_result.positive_events:
            positive_episodes_events.append(prune_result.positive_events)

    summary["positive_events"] = total_positive
    summary["negative_events"] = total_negative
    logger.info(
        "Pruned: %d positive, %d negative events",
        total_positive,
        total_negative,
    )

    # Step E: Translate positive events into semantic steps
    all_translations = []
    episode_sop_steps: list[list[dict]] = []

    for ep_events in positive_episodes_events:
        translations = translator.translate_batch(ep_events)
        all_translations.extend(translations)

        # Score each translation and handle VLM fallback
        sop_steps_for_episode: list[dict] = []

        for idx, tr in enumerate(translations):
            # Build scoring context
            context: dict = {}
            if tr.pre_state.get("window_title"):
                context["expected_title"] = tr.pre_state["window_title"]
            if tr.pre_state.get("url"):
                context["expected_url"] = tr.pre_state["url"]
            if tr.pre_state.get("app_id"):
                context["expected_app"] = tr.pre_state["app_id"]

            # Check clipboard provenance
            raw_event_id = tr.raw_event_id
            if raw_event_id in paste_ids_with_provenance:
                context["clipboard_link"] = True

            # Check dwell snapshot provenance
            if tr.intent == "read":
                context["dwell_snapshot"] = True

            conf = scorer.score(tr, context)

            # Auto-enqueue VLM job for rejected translations
            if conf.decision == "reject" and conf.total < VLM_REJECT_THRESHOLD:
                priority = vlm_queue.compute_priority(
                    conf.total,
                    tr.intent,
                    datetime.now(timezone.utc),
                )
                # Boost priority for native app events (less context available)
                if conf.native_app:
                    priority = min(priority + ConfidenceScorer.NATIVE_APP_VLM_BOOST, 1.0)

                job = VLMJob(
                    job_id=str(uuid.uuid4()),
                    event_id=tr.raw_event_id,
                    episode_id="",
                    semantic_step_index=idx,
                    confidence_score=conf.total,
                    priority_score=priority,
                )
                if vlm_queue.enqueue(job):
                    summary["vlm_enqueued"] += 1

            # Build SOP step dict for high-confidence translations
            if conf.decision in ("accept", "accept_flagged"):
                target_desc = ""
                selector = None
                if tr.target:
                    target_desc = tr.target.selector
                    selector = tr.target.selector

                sop_step = {
                    "step": tr.intent,
                    "target": target_desc,
                    "selector": selector,
                    "parameters": tr.parameters,
                    "confidence": conf.total,
                    "pre_state": tr.pre_state,
                }
                sop_steps_for_episode.append(sop_step)

        if sop_steps_for_episode:
            episode_sop_steps.append(sop_steps_for_episode)

    summary["translations"] = len(all_translations)
    logger.info(
        "Translated %d events, enqueued %d VLM jobs",
        len(all_translations),
        summary["vlm_enqueued"],
    )

    # Step F: Induce SOPs from episode step sequences
    # Only attempt if we have the sop_inducer (requires prefixspan)
    sop_templates: list[dict] = []
    if sop_inducer is not None and episode_sop_steps:
        try:
            sop_templates = sop_inducer.induce(episode_sop_steps)
            summary["sops_induced"] = len(sop_templates)
            logger.info("Induced %d SOP templates", len(sop_templates))
        except Exception:
            logger.exception("SOP induction failed")

    # Step F2: Format, version, and export SOPs
    if sop_templates:
        try:
            paths = openclaw_writer.write_all_sops(sop_templates)
            summary["sops_exported"] = len(paths)

            # Update index with all SOP templates
            index_generator.update_index(
                openclaw_writer.sops_dir, sop_templates
            )
            logger.info("Exported %d SOPs", len(paths))
        except Exception:
            logger.exception("SOP export failed")

    return summary


def main(argv: list[str] | None = None) -> None:
    """Worker entry-point: connect to DB, poll for work, run until shutdown."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("Starting oc-apprentice-worker")
    logger.info("Database path: %s", args.db_path)
    logger.info("Poll interval: %.1fs", args.poll_interval)

    if not args.db_path.is_file():
        logger.error("Database file not found: %s", args.db_path)
        logger.error(
            "Is the daemon running? The daemon must create the database first."
        )
        sys.exit(1)

    shutdown_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — requesting shutdown", sig_name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Initialize pipeline components
    episode_builder = EpisodeBuilder()
    clipboard_linker = ClipboardLinker()
    pruner = NegativeDemoPruner()
    translator = SemanticTranslator()
    scorer = ConfidenceScorer()
    vlm_queue = VLMFallbackQueue()
    index_generator = IndexGenerator()
    openclaw_writer = OpenClawWriter(workspace_dir=args.sops_dir.parent.parent)

    # Check VLM availability and hint if not installed
    from oc_apprentice_worker.setup_vlm import check_vlm_available
    vlm_status = check_vlm_available()
    vlm_worker = None
    if any(vlm_status.values()):
        logger.info("VLM backend available — enhanced native app observation enabled")
        # Create VLM worker for variable classification
        # Priority: mlx_vlm > ollama > llama_cpp > openai_compat
        try:
            from oc_apprentice_worker.vlm_worker import VLMWorker, VLMConfig, VLMBackend
            if vlm_status["mlx_vlm"]:
                backend_type = VLMBackend.MLX_VLM
            elif vlm_status["ollama"]:
                backend_type = VLMBackend.OLLAMA
            elif vlm_status["llama_cpp"]:
                backend_type = VLMBackend.LLAMA_CPP
            else:
                backend_type = VLMBackend.OPENAI_COMPAT
            vlm_worker = VLMWorker(config=VLMConfig(backend=backend_type))
            logger.info(
                "VLM worker initialized (%s) for variable classification",
                backend_type.value,
            )
        except Exception:
            logger.warning("Failed to initialize VLM worker", exc_info=True)
    else:
        logger.info(
            "VLM not installed. For better native app observation, run: oc-setup-vlm"
        )

    # Try to import SOPInducer (requires prefixspan)
    sop_inducer = None
    try:
        from oc_apprentice_worker.sop_inducer import SOPInducer
        sop_inducer = SOPInducer(vlm_worker=vlm_worker)
        if vlm_worker is not None:
            logger.info("SOPInducer loaded with VLM-assisted variable classification")
        else:
            logger.info("SOPInducer loaded (prefixspan available)")
    except ImportError:
        logger.warning(
            "prefixspan not installed — SOP induction disabled. "
            "Install with: pip install prefixspan"
        )

    # Initialize scheduler gate (GAP 5) and deep scanner (GAP 6)
    idle_gate = IdleJobGate(SchedulerConfig())
    deep_scanner = DeepScanner()

    with WorkerDB(args.db_path) as db:
        logger.info("Connected to database, entering main loop")

        current_interval = args.poll_interval
        max_interval = max(60.0, args.poll_interval * 16)
        consecutive_errors = 0

        while not shutdown_requested:
            try:
                unprocessed = db.get_unprocessed_events(limit=100)
                pending_vlm = db.get_pending_vlm_jobs(limit=10)

                if unprocessed or pending_vlm:
                    logger.info(
                        "Poll: %d unprocessed events, %d pending VLM jobs",
                        len(unprocessed),
                        len(pending_vlm),
                    )
                    # Reset backoff when work is found
                    current_interval = args.poll_interval
                else:
                    logger.debug("Poll: nothing to do")
                    # Exponential backoff when idle
                    current_interval = min(current_interval * 2, max_interval)

                if unprocessed:
                    # Check scheduler gate before running heavy pipeline
                    gate_result = idle_gate.check()
                    if not gate_result.can_run:
                        logger.debug(
                            "Idle gate blocked: %s — deferring pipeline",
                            gate_result.blockers,
                        )
                        time.sleep(current_interval)
                        continue

                    summary = run_pipeline(
                        unprocessed,
                        episode_builder=episode_builder,
                        clipboard_linker=clipboard_linker,
                        pruner=pruner,
                        translator=translator,
                        scorer=scorer,
                        vlm_queue=vlm_queue,
                        openclaw_writer=openclaw_writer,
                        index_generator=index_generator,
                        sop_inducer=sop_inducer,
                    )
                    logger.info("Pipeline summary: %s", summary)

                    # Mark processed events so they are not re-read (GAP 4)
                    event_ids = [ev["id"] for ev in unprocessed if "id" in ev]
                    if event_ids:
                        db.mark_events_processed(event_ids)

                    # Run Tier 2 deep scan on any text artifacts from this batch
                    text_artifacts = []
                    for ev in unprocessed:
                        # Extract any text-like fields for deep scan
                        window_json = ev.get("window_json")
                        if window_json:
                            text_artifacts.append({
                                "id": ev.get("id", "unknown"),
                                "text": window_json,
                            })
                        metadata_json = ev.get("metadata_json")
                        if metadata_json:
                            text_artifacts.append({
                                "id": ev.get("id", "unknown"),
                                "text": metadata_json,
                            })

                    if text_artifacts:
                        scan_result = deep_scanner.scan_artifacts(text_artifacts)
                        if scan_result.has_pii:
                            logger.warning(
                                "Deep scan found %d PII match(es) in %d artifact(s)",
                                scan_result.total_pii,
                                scan_result.artifacts_scanned,
                            )

                    # Reset consecutive error counter after successful processing
                    consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                logger.error(
                    "Error in main loop (consecutive: %d)",
                    consecutive_errors,
                    exc_info=True,
                )
                if consecutive_errors >= 10:
                    logger.critical(
                        "10 consecutive errors without successful processing, shutting down"
                    )
                    break
                time.sleep(5)
                continue

            time.sleep(current_interval)

    logger.info("Worker shut down cleanly")


if __name__ == "__main__":
    main()

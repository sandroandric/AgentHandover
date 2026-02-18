"""OpenMimic apprentice worker entry-point.

Provides the main loop that reads from the daemon's SQLite database
and drives episode building, semantic translation, confidence scoring,
VLM fallback enqueuing, SOP induction, formatting, and export.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from oc_apprentice_worker.clipboard_linker import ClipboardLinker
from oc_apprentice_worker.confidence import ConfidenceScorer, is_native_app_context
from oc_apprentice_worker.db import WorkerDB
from oc_apprentice_worker.deep_scan import DeepScanner
from oc_apprentice_worker.episode_builder import EpisodeBuilder
from oc_apprentice_worker.exporter import IndexGenerator
from oc_apprentice_worker.negative_demo import NegativeDemoPruner
from oc_apprentice_worker.export_adapter import SOPExportAdapter
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

_WORKER_VERSION = "0.1.0"
_DB_RETRY_MAX_SECONDS = 120
_DB_RETRY_POLL_SECONDS = 5
_IDLE_LOG_INTERVAL_SECONDS = 300  # 5 minutes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oc-apprentice-worker",
        description="OpenMimic apprentice worker process",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_WORKER_VERSION}")
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
    parser.add_argument(
        "--adapter",
        choices=["openclaw", "generic"],
        default="openclaw",
        help="SOP export adapter (default: openclaw)",
    )
    parser.add_argument(
        "--json-export",
        action="store_true",
        default=False,
        help="Also export SOPs as JSON (used with generic adapter)",
    )
    return parser.parse_args(argv)


def _status_dir() -> Path:
    """Return the standard data directory for status/PID files."""
    if _platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "oc-apprentice"
    return Path.home() / ".local" / "share" / "oc-apprentice"


def _write_worker_status(
    *,
    started_at: str,
    events_processed_today: int,
    sops_generated: int,
    last_pipeline_duration_ms: int | None,
    consecutive_errors: int,
    vlm_available: bool,
    sop_inducer_available: bool,
) -> None:
    """Atomically write worker-status.json (tmp + fsync + rename)."""
    status = {
        "pid": os.getpid(),
        "version": _WORKER_VERSION,
        "started_at": started_at,
        "heartbeat": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "events_processed_today": events_processed_today,
        "sops_generated": sops_generated,
        "last_pipeline_duration_ms": last_pipeline_duration_ms,
        "consecutive_errors": consecutive_errors,
        "vlm_available": vlm_available,
        "sop_inducer_available": sop_inducer_available,
    }
    sdir = _status_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    target = sdir / "worker-status.json"
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(sdir), prefix=".worker-status.", suffix=".tmp"
        )
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(target))
    except Exception:
        logger.debug("Failed to write worker-status.json", exc_info=True)
        # Best-effort; don't crash the worker over a status file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _remove_worker_status() -> None:
    """Remove worker-status.json on clean shutdown."""
    try:
        (_status_dir() / "worker-status.json").unlink(missing_ok=True)
    except Exception:
        pass


def run_pipeline(
    events: list[dict],
    *,
    episode_builder: EpisodeBuilder,
    clipboard_linker: ClipboardLinker,
    pruner: NegativeDemoPruner,
    translator: SemanticTranslator,
    scorer: ConfidenceScorer,
    vlm_queue: VLMFallbackQueue,
    openclaw_writer: SOPExportAdapter,
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

    # Step E: Translate positive events into semantic steps (parallel per episode)
    all_translations = []
    episode_sop_steps: list[list[dict]] = []

    # Episodes are independent — translate in parallel
    if len(positive_episodes_events) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(positive_episodes_events))) as pool:
            translation_batches = list(pool.map(
                translator.translate_batch, positive_episodes_events
            ))
        for translations in translation_batches:
            all_translations.extend(translations)
    elif positive_episodes_events:
        translation_batches = [translator.translate_batch(positive_episodes_events[0])]
        all_translations.extend(translation_batches[0])
    else:
        translation_batches = []

    # Score each episode's translations and handle VLM fallback
    for translations in translation_batches:
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
                openclaw_writer.get_sops_dir(), sop_templates
            )
            logger.info("Exported %d SOPs", len(paths))
        except Exception:
            logger.exception("SOP export failed")

    return summary


def _run_deep_scan(scanner: DeepScanner, artifacts: list[dict]) -> None:
    """Background callback for Tier 2 deep scan."""
    try:
        scan_result = scanner.scan_artifacts(artifacts)
        if scan_result.has_pii:
            logger.warning(
                "Deep scan found %d PII match(es) in %d artifact(s)",
                scan_result.total_pii,
                scan_result.artifacts_scanned,
            )
    except Exception:
        logger.exception("Background deep scan failed")


def _wait_for_db(db_path: Path, shutdown_flag: list[bool]) -> bool:
    """Wait up to _DB_RETRY_MAX_SECONDS for the daemon to create the DB.

    Returns True if DB appeared, False if timed out or shutdown requested.
    """
    if db_path.is_file():
        return True

    logger.info(
        "Database not found at %s — waiting for daemon to create it "
        "(up to %ds)...",
        db_path,
        _DB_RETRY_MAX_SECONDS,
    )

    elapsed = 0
    while elapsed < _DB_RETRY_MAX_SECONDS:
        if shutdown_flag[0]:
            logger.info("Shutdown requested during DB wait — exiting")
            return False
        time.sleep(_DB_RETRY_POLL_SECONDS)
        elapsed += _DB_RETRY_POLL_SECONDS
        if db_path.is_file():
            logger.info("Database appeared after %ds", elapsed)
            return True
        logger.info(
            "Waiting for daemon to create database (%ds/%ds)...",
            elapsed,
            _DB_RETRY_MAX_SECONDS,
        )

    logger.error(
        "Database not created after %ds. Is the daemon running? "
        "Start it with: openmimic start",
        _DB_RETRY_MAX_SECONDS,
    )
    return False


def main(argv: list[str] | None = None) -> None:
    """Worker entry-point: connect to DB, poll for work, run until shutdown."""
    args = _parse_args(argv)

    from logging.handlers import RotatingFileHandler

    if _platform.system() == "Darwin":
        log_dir = Path.home() / "Library" / "Application Support" / "oc-apprentice" / "logs"
    else:
        log_dir = Path.home() / ".local" / "share" / "oc-apprentice" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt = "%Y-%m-%dT%H:%M:%S"

    # File handler with rotation (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        log_dir / "worker.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))

    # Also log to stderr for debugging when running manually
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=[file_handler, stderr_handler],
    )

    # Signal handling — set up BEFORE the DB retry loop so we can
    # exit cleanly if the user hits Ctrl-C while waiting.
    shutdown_flag: list[bool] = [False]

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — requesting shutdown", sig_name)
        shutdown_flag[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Write PID file
    pid_dir = _status_dir()
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "worker.pid"
    pid_file.write_text(str(os.getpid()))
    logger.info("PID file written: %s", pid_file)

    logger.info("Starting oc-apprentice-worker v%s", _WORKER_VERSION)
    logger.info("Database path: %s", args.db_path)
    logger.info("Poll interval: %.1fs", args.poll_interval)

    # Task 1: Retry loop — wait for the daemon to create the DB
    if not _wait_for_db(args.db_path, shutdown_flag):
        # Clean up PID file before exiting
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        sys.exit(1)

    if shutdown_flag[0]:
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # Initialize pipeline components
    episode_builder = EpisodeBuilder()
    clipboard_linker = ClipboardLinker()
    pruner = NegativeDemoPruner()
    translator = SemanticTranslator()
    scorer = ConfidenceScorer()
    vlm_queue = VLMFallbackQueue()
    index_generator = IndexGenerator()
    # Create export adapter based on config
    if args.adapter == "generic":
        from oc_apprentice_worker.generic_writer import GenericWriter
        sop_writer = GenericWriter(output_dir=args.sops_dir, json_export=args.json_export)
    else:
        sop_writer = OpenClawWriter(workspace_dir=args.sops_dir.parent.parent)

    # Check VLM availability and hint if not installed
    from oc_apprentice_worker.setup_vlm import check_vlm_available
    vlm_status = check_vlm_available()
    vlm_worker = None
    vlm_available = False
    if any(vlm_status.values()):
        vlm_available = True
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
    sop_inducer_available = False
    try:
        from oc_apprentice_worker.sop_inducer import SOPInducer
        sop_inducer = SOPInducer(vlm_worker=vlm_worker)
        sop_inducer_available = True
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

    # Single-thread pool for background deep scan (non-blocking privacy check)
    deep_scan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deep-scan")

    # Cumulative counters for status reporting
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_events_processed: int = 0
    total_sops_generated: int = 0
    last_pipeline_duration_ms: int | None = None

    # Task 5: Idle progress tracking
    last_idle_log_time = time.monotonic()

    # Write initial status file after DB connect
    _write_worker_status(
        started_at=started_at,
        events_processed_today=total_events_processed,
        sops_generated=total_sops_generated,
        last_pipeline_duration_ms=last_pipeline_duration_ms,
        consecutive_errors=0,
        vlm_available=vlm_available,
        sop_inducer_available=sop_inducer_available,
    )

    with WorkerDB(args.db_path) as db:
        logger.info("Connected to database, entering main loop")

        current_interval = args.poll_interval
        max_interval = max(60.0, args.poll_interval * 16)
        consecutive_errors = 0

        while not shutdown_flag[0]:
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
                    # Reset idle log timer
                    last_idle_log_time = time.monotonic()
                else:
                    logger.debug("Poll: nothing to do")
                    # Exponential backoff when idle
                    current_interval = min(current_interval * 2, max_interval)

                    # Task 5: Periodic idle progress message
                    now_mono = time.monotonic()
                    if now_mono - last_idle_log_time >= _IDLE_LOG_INTERVAL_SECONDS:
                        logger.info(
                            "Watching for activity... (%d events today, %d SOPs generated)",
                            total_events_processed,
                            total_sops_generated,
                        )
                        last_idle_log_time = now_mono

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

                    pipeline_start = time.monotonic()
                    summary = run_pipeline(
                        unprocessed,
                        episode_builder=episode_builder,
                        clipboard_linker=clipboard_linker,
                        pruner=pruner,
                        translator=translator,
                        scorer=scorer,
                        vlm_queue=vlm_queue,
                        openclaw_writer=sop_writer,
                        index_generator=index_generator,
                        sop_inducer=sop_inducer,
                    )
                    pipeline_elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
                    last_pipeline_duration_ms = pipeline_elapsed_ms
                    logger.info("Pipeline summary: %s", summary)

                    # Update cumulative counters
                    total_events_processed += summary["events_in"]
                    total_sops_generated += summary["sops_exported"]

                    # Task 5: User-facing progress messages
                    if summary["sops_exported"] > 0:
                        logger.info(
                            "Generated %d new SOP(s)!",
                            summary["sops_exported"],
                        )
                    else:
                        logger.info(
                            "Processed %d events into %d episodes. "
                            "SOPs generated after workflows repeated 2+ times.",
                            summary["events_in"],
                            summary["episodes"],
                        )

                    # Mark processed events so they are not re-read (GAP 4)
                    event_ids = [ev["id"] for ev in unprocessed if "id" in ev]
                    if event_ids:
                        db.mark_events_processed(event_ids)

                    # Run Tier 2 deep scan in background thread (non-blocking)
                    text_artifacts = []
                    for ev in unprocessed:
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
                        deep_scan_pool.submit(
                            _run_deep_scan, deep_scanner, text_artifacts
                        )

                    # Reset consecutive error counter after successful processing
                    consecutive_errors = 0

                # Write status file after each cycle (heartbeat)
                _write_worker_status(
                    started_at=started_at,
                    events_processed_today=total_events_processed,
                    sops_generated=total_sops_generated,
                    last_pipeline_duration_ms=last_pipeline_duration_ms,
                    consecutive_errors=consecutive_errors,
                    vlm_available=vlm_available,
                    sop_inducer_available=sop_inducer_available,
                )

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

    # Wait for any in-flight deep scan to finish before exit
    deep_scan_pool.shutdown(wait=True)

    # Remove PID file and status file on clean shutdown
    try:
        pid_file.unlink(missing_ok=True)
    except Exception:
        pass
    _remove_worker_status()

    logger.info("Worker shut down cleanly")


if __name__ == "__main__":
    main()

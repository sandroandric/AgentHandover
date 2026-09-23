"""AgentHandover apprentice worker entry-point.

Provides the main loop that reads from the daemon's SQLite database
and drives episode building, semantic translation, confidence scoring,
VLM fallback enqueuing, SOP induction, formatting, and export.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agenthandover_worker import trigger_spool
from agenthandover_worker import workflow_mining
from agenthandover_worker.clipboard_linker import ClipboardLinker
from agenthandover_worker.event_helpers import focused_field, parse_json_field
from agenthandover_worker.config import EXPORT_ADAPTERS, config_path, load_config
from agenthandover_worker.confidence import ConfidenceScorer, is_native_app_context
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.evidence_tracker import ObservationEvidence, append_observation
from agenthandover_worker.exporter import IndexGenerator
from agenthandover_worker.focus_processor import FocusProcessor
from agenthandover_worker.frame_differ import DiffConfig, FrameDiffer
from agenthandover_worker.negative_demo import NegativeDemoPruner
from agenthandover_worker.model_profiles import migrate_vlm_file
from agenthandover_worker.model_scheduler import ModelScheduler
from agenthandover_worker.ollama import OllamaClient, OllamaError
from agenthandover_worker.ops_telemetry import COMPUTE, MODEL_USAGE
from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.annotation_cache import AnnotationCache
from agenthandover_worker.scene_annotator import (
    AnnotationConfig,
    SceneAnnotator,
    as_compact,
    page_for_event,
    skip_record,
)
from agenthandover_worker.sop_generator import SOPGenerator, SOPGeneratorConfig, _generate_slug
from agenthandover_worker.sop_linter import lint_sop
from agenthandover_worker.task_segmenter import TaskSegmenter, SegmenterConfig
from agenthandover_worker.scheduler import (
    AnnotationBudget,
    IdleJobGate,
    SchedulerConfig,
    charged_to,
)
from agenthandover_worker.translator import SemanticTranslator
from agenthandover_worker.vlm_queue import VLMFallbackQueue, VLMJob
from agenthandover_worker.loop_guard import (
    ERROR_THRESHOLD,
    EventRetryGuard,
    error_backoff_seconds,
    interruptible_sleep,
)

logger = logging.getLogger("agenthandover_worker")

# Default paths assume a single-user system. Use --db-path and --sops-dir
# CLI args to override for multi-user or containerized deployments.
import platform as _platform

from agenthandover_worker.paths import DEFAULT_DB_PATH, KNOWLEDGE_BASE_DIR, data_dir

DEFAULT_SOPS_DIR = Path.home() / ".openclaw" / "workspace" / "memory" / "apprentice"
POLL_INTERVAL_SECONDS = 2.0
VLM_REJECT_THRESHOLD = 0.60

# Single source of truth: read from the package metadata (pyproject.toml
# → dist-info). See agenthandover_worker/__init__.py for the fallback
# behavior. This replaces a hardcoded "0.2.0" string that went stale
# across three releases and was caught by hikoae on v0.2.3 as
# "worker status/version still reports 0.2.0 in the runtime status file
# even though the installed package is 0.2.3".
from agenthandover_worker import __version__ as _WORKER_VERSION
_DB_RETRY_MAX_SECONDS = 120
_DB_RETRY_POLL_SECONDS = 5
_IDLE_LOG_INTERVAL_SECONDS = 300  # 5 minutes


def _lint_and_log(sop_template: dict, label: str) -> bool:
    """Lint an SOP template, log any issues, return True if valid.

    Errors are logged at ERROR level, warnings at WARNING level.
    ``label`` is also the lint source: VLM-generated labels (see
    ``sop_linter.VLM_GENERATED_SOURCES``) make a step without verify or
    description an error.
    """
    result = lint_sop(sop_template, source=label)
    ident = sop_template.get("slug") or sop_template.get("id") or "?"
    for issue in result.issues:
        if issue.severity == "error":
            logger.error(
                "SOP lint error [%s] %s [%s]: %s",
                label, issue.field, ident, issue.message,
            )
        else:
            logger.warning(
                "SOP lint warning [%s] %s [%s]: %s",
                label, issue.field, ident, issue.message,
            )
    if not result.valid:
        logger.error(
            "SOP '%s' failed validation (%d errors), skipping export",
            sop_template.get("title", "?"), len(result.errors),
        )
    return result.valid


def _generate_linted(generate, label: str, title: str) -> tuple:
    """Run ``generate()`` and lint its SOP; regenerate once if lint fails.

    ``generate`` returns a ``GeneratedSOP``.  Each call re-prompts the
    model (sampled, so a second answer can differ).  Returns
    ``(result, lint_error)``: ``lint_error`` is empty unless the SOP
    still fails lint after the one regeneration, in which case it lists
    the lint errors so the caller can record the failure for retry.
    """
    result = generate()
    if result.success and result.sop and not _lint_and_log(result.sop, label):
        logger.warning(
            "SOP for '%s' failed validation [%s], regenerating once",
            title, label,
        )
        result = generate()
        if result.success and result.sop and not _lint_and_log(result.sop, label):
            return result, _lint_error_text(result.sop, label)
    return result, ""


def _lint_error_text(sop_template: dict, label: str) -> str:
    """One-line summary of an SOP's lint errors, for failure records."""
    errors = lint_sop(sop_template, source=label).errors
    return "SOP failed validation: " + "; ".join(
        f"{e.field}: {e.message}" for e in errors
    )


def _export_via_adapter(
    adapter: "SOPExportAdapter",
    sop_templates: list[dict],
    procedure_writer: "ProcedureWriter | None",
) -> list[Path]:
    """Export, via *adapter*, the templates whose v3 procedure passes the
    hand-off gate (``ledger.handoff_gate``: approved for agents, no
    required slot unknown, trust above observe, fresh).

    A draft, or a template with no procedure in the knowledge base, is
    never written where agents read; it is written once a human approves
    it (the promote trigger and the lifecycle pass's
    ``BundleCompiler.compile_all``).  Returns the written file paths.
    """
    from agenthandover_worker.ledger import handoff_gate

    kb = procedure_writer._kb if procedure_writer is not None else None
    paths: list[Path] = []
    for template in sop_templates:
        slug = template.get("slug", "unknown")
        proc = kb.get_procedure(slug) if kb is not None else None
        if proc is None or not handoff_gate(proc).ok:
            logger.debug("Not exporting %s via %s: not approved for agents",
                         slug, type(adapter).__name__)
            continue
        try:
            paths.append(adapter.write_procedure(proc))
        except Exception:
            logger.warning(
                "Export failed for %s via %s",
                slug, type(adapter).__name__, exc_info=True,
            )
    return paths


def _read_keychain_api_key(provider: str) -> str:
    """Read an API key from macOS Keychain (set by SwiftUI onboarding).

    The onboarding stores keys with service 'com.agenthandover.app'
    and account 'agenthandover-{provider}-key'.  We retrieve via the
    ``security`` CLI to avoid a native dependency.

    Returns the key string, or "" if not found / not on macOS.
    """
    import subprocess

    if _platform.system() != "Darwin":
        return ""

    account = f"agenthandover-{provider}-key"
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s", "com.agenthandover.app",
                "-a", account,
                "-w",  # print password only
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _check_v2_schema(db: "WorkerDB") -> bool:
    """Check if the database has the v2 annotation columns.

    Returns True if the schema supports v2 scene annotation.
    """
    try:
        db._conn.execute(
            "SELECT annotation_status FROM events LIMIT 0"
        )
        return True
    except Exception:
        return False


def _process_annotations(
    db: "WorkerDB",
    annotator: "SceneAnnotator",
    screenshots_dir: Path,
    *,
    batch_size: int = 5,
    privacy_checker: "PrivacyZoneChecker | None" = None,
    activity_classifier: "ActivityClassifier | None" = None,
    budget: "AnnotationBudget | None" = None,
) -> dict:
    """Process a batch of unannotated screenshots through the scene annotator.

    Returns stats dict with counts of annotated/skipped/failed/blocked/
    deferred events, and ``budget_exhausted``.

    With a *budget*, each frame's model calls (annotation and activity
    classification) are charged to it under :func:`charged_to`, and each
    annotated frame counts against the hour (and, when the model was
    called, the day); once either cap is spent the batch stops, leaving
    the rows pending with their screenshots until the hour or the day
    allows more.

    Focus-session events (tagged with ``focus_session_id`` in metadata)
    are excluded in the query and left to the focus_processor. The scene
    annotator deletes the source JPEG after processing
    (``delete_screenshot_after_processing=True``), so if passive annotation
    ran on focus events first, the focus_processor would find
    ``missing_screenshot`` and fail to generate the SOP.  Excluding them in
    SQL (not after the LIMIT) also means focus rows that stay pending can
    never fill the batch and starve passive frames.

    When Ollama is unreachable (``annotator.ready()`` is False, or a call
    returns ``deferred``) the batch stops: the rows stay pending and their
    screenshots are kept for the next attempt after the backoff.

    The privacy zone check (the second line behind the daemon's, which
    keeps a blocked window from being captured at all) needs no model, so
    it runs first, Ollama up or not: a blocked or metadata-only frame's
    JPEG is deleted at once, never left waiting for the annotator.
    """
    stats = {
        "annotated": 0, "skipped": 0, "failed": 0, "blocked": 0, "deferred": 0,
        "budget_exhausted": False,
    }
    if budget is not None and budget.exhausted():
        stats["budget_exhausted"] = True
        return stats
    unannotated = db.get_unannotated_events(limit=batch_size, exclude_focus=True)
    if not unannotated:
        return stats
    ollama_ready: bool | None = None

    for event in unannotated:
        if budget is not None and budget.exhausted():
            stats["budget_exhausted"] = True
            break
        event_id = event.get("id", "unknown")
        timestamp = event.get("timestamp", "")

        # Privacy zone check — skip blocked events, metadata-only for restricted
        if privacy_checker is not None:
            from agenthandover_worker.privacy_zones import ObservationTier
            tier = privacy_checker.check_event(event)
            if tier == ObservationTier.BLOCKED:
                _discard_screenshot(event, screenshots_dir)
                db.save_annotation(event_id, "", status="privacy_blocked")
                stats["blocked"] += 1
                logger.debug("Event %s blocked by privacy zone", event_id[:8])
                continue
            if tier == ObservationTier.METADATA_ONLY:
                _discard_screenshot(event, screenshots_dir)
                # Save minimal metadata annotation (app + timestamp, no content)
                metadata_annotation = json.dumps({
                    "privacy_tier": "metadata_only",
                    "timestamp": timestamp,
                    "app": event.get("window_json", ""),
                })
                db.save_annotation(event_id, metadata_annotation, status="metadata_only")
                stats["skipped"] += 1
                logger.debug("Event %s metadata-only by privacy zone", event_id[:8])
                continue

        # Without Ollama the frame stays pending; later rows are still
        # scanned for privacy zones.
        if ollama_ready is None:
            ollama_ready = annotator.ready()
        if not ollama_ready:
            continue

        # Get sliding window context (last N annotations within time window)
        recent = db.get_recent_annotations(
            before_timestamp=timestamp,
            limit=annotator.config.sliding_window_size,
            max_age_seconds=annotator.config.sliding_window_max_age_sec,
        )

        # Run annotation
        with charged_to(budget):
            result = annotator.annotate_event(
                event,
                recent_annotations=recent,
                artifact_dir=str(screenshots_dir),
                page=page_for_event(db, event),
            )
        if budget is not None and result.status in ("completed", "failed"):
            budget.charge_frame(model_call=not result.cached)

        # Save result to DB
        if result.status == "completed" and result.annotation:
            # Enrich with activity classification (Phase 1)
            if activity_classifier is not None:
                try:
                    with charged_to(budget):
                        cls_result = activity_classifier.classify(
                            result.annotation, event_context=event,
                        )
                    tc = result.annotation.setdefault("task_context", {})
                    tc["activity_type"] = cls_result.activity_type.value
                    tc["learnability"] = cls_result.learnability.value
                    tc["classification_confidence"] = cls_result.confidence
                    tc["classification_source"] = cls_result.source
                    tc["classification_reasoning"] = cls_result.reasoning
                except Exception:
                    logger.debug("Activity classification failed", exc_info=True)

            # Attach visual text proxy for future vector embedding
            if result.visual_text_proxy:
                result.annotation["_visual_text_proxy"] = result.visual_text_proxy

            db.save_annotation(
                event_id,
                json.dumps(result.annotation),
                status="completed",
            )
            stats["annotated"] += 1
            what_doing = (
                result.annotation
                .get("task_context", {})
                .get("what_doing", "?")
            )
            logger.info(
                "Annotated event %s (%.1fs): %s",
                event_id[:8],
                result.inference_time_seconds,
                what_doing[:80],
            )
        elif result.status in ("skipped", "missing_screenshot"):
            db.save_annotation(event_id, skip_record(result), status=result.status)
            stats["skipped"] += 1
            logger.debug(
                "Annotation %s for %s: %s",
                result.status,
                event_id[:8],
                result.error or "",
            )
        elif result.status == "deferred":
            # Ollama unreachable: leave this and the remaining rows pending.
            stats["deferred"] += 1
            logger.warning(
                "Annotation deferred for %s: %s", event_id[:8], result.error,
            )
            break
        else:
            db.save_annotation(event_id, "", status="failed")
            stats["failed"] += 1
            logger.warning(
                "Annotation failed for %s: %s",
                event_id[:8],
                result.error,
            )

    return stats


# worker_state keys: the local date the once-per-day jobs last ran, and
# passive annotation's budget counters for the current local day.
DAILY_JOBS_STATE_KEY = "daily_jobs_date"
ANNOTATION_BUDGET_STATE_KEY = "annotation_budget"

# Passive frames still pending after this long are given up on.
ANNOTATION_BACKLOG_MAX_AGE_HOURS = 24.0


def _run_daily_maintenance(
    *,
    vector_kb: "VectorKB | None" = None,
    activity_searcher: "ActivitySearcher | None" = None,
    annotation_cache: "AnnotationCache | None" = None,
) -> None:
    """Prune what grows with time: vectors past their retention or from
    another embedding model (the row cap is enforced on every upsert),
    search-index entries of events the
    daemon deleted, and unused annotation-cache rows."""
    for name, job in (
        ("vector purge", vector_kb and vector_kb.purge_stale),
        ("search index prune", activity_searcher and activity_searcher.prune_index),
        ("annotation cache prune", annotation_cache and annotation_cache.prune),
    ):
        if not job:
            continue
        try:
            job()
        except Exception:
            logger.debug("Daily %s failed", name, exc_info=True)


def _discard_screenshot(event: dict, screenshots_dir: Path) -> None:
    """Delete an event's JPEG (only inside *screenshots_dir*)."""
    from agenthandover_worker.scene_annotator import find_screenshot

    path = find_screenshot(event, screenshots_dir)
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete screenshot %s", path, exc_info=True)


# A JPEG this young may belong to an event the daemon has not inserted yet
# (it writes the file, then sends the event to its storage writer).
ORPHAN_MIN_AGE_SECONDS = 300


def _sweep_orphan_screenshots(
    db: "WorkerDB",
    screenshots_dir: Path,
    *,
    min_age_seconds: float = ORPHAN_MIN_AGE_SECONDS,
) -> int:
    """Delete JPEGs in *screenshots_dir* that no pending event refers to.

    Run at worker start.  A frame's JPEG is needed only until its event is
    annotated (or blocked by a privacy zone); after that every path deletes
    it.  Anything else in the directory is an orphan: frames of blocked or
    metadata-only windows kept by an older worker, frames of events that
    were purged, files left by a crash.  Files younger than
    *min_age_seconds* are left alone.  Returns the number deleted.
    """
    if not screenshots_dir.is_dir():
        return 0
    referenced = {
        str(Path(p).resolve()) for p in db.pending_screenshot_paths()
    }
    cutoff = time.time() - min_age_seconds
    removed = 0
    for path in screenshots_dir.iterdir():
        if path.suffix.lower() not in (".jpg", ".jpeg") or not path.is_file():
            continue
        if str(path.resolve()) in referenced:
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            logger.debug("Could not remove orphan screenshot %s", path, exc_info=True)
    if removed:
        logger.info("Removed %d orphaned screenshot(s) from %s", removed, screenshots_dir)
    return removed


def _process_diffs(
    db: "WorkerDB",
    differ: "FrameDiffer",
    *,
    batch_size: int = 10,
    budget: "AnnotationBudget | None" = None,
) -> dict:
    """Process annotated events that need frame-to-frame diffs.

    For each event, finds the previous annotated event and computes
    the diff (either a code-only marker for edge cases or an LLM-based
    action diff).  A passive frame's diff call is charged to *budget*
    (a focus-session frame's is not), and while the budget's compute is
    spent a passive pair that needs the model stays pending.

    Returns stats dict with counts of diffs/edge_cases/failed/deferred.
    """
    needs_diff = db.get_events_needing_diff(limit=batch_size)
    stats = {"diffs": 0, "edge_cases": 0, "failed": 0, "deferred": 0}
    if not needs_diff:
        return stats

    for event in needs_diff:
        event_id = event.get("id", "unknown")
        timestamp = event.get("timestamp", "")

        # Get the previous annotated event
        prev_event = db.get_annotation_before(timestamp)
        if prev_event is None:
            # First annotated event — no predecessor to diff against
            db.save_frame_diff(
                event_id, json.dumps({"diff_type": "first_frame"})
            )
            stats["edge_cases"] += 1
            continue

        # Compute diff
        focus = parse_json_field(event, "metadata_json").get("focus_session_id")
        allow_model = bool(focus) or budget is None or not budget.compute_exhausted()
        with charged_to(None if focus else budget):
            result = differ.diff_pair(prev_event, event, allow_model=allow_model)
        if result.diff is None:
            stats["deferred"] += 1
            continue

        # Save
        db.save_frame_diff(event_id, json.dumps(result.diff))

        diff_type = result.diff.get("diff_type", "unknown")
        if diff_type == "action":
            stats["diffs"] += 1
            logger.debug(
                "Diff for %s (%.1fs): %s",
                event_id[:8],
                result.inference_time_seconds,
                result.diff.get("step_description", "?")[:80],
            )
        elif diff_type == "diff_failed":
            stats["failed"] += 1
            logger.debug(
                "Diff failed for %s: %s",
                event_id[:8],
                result.diff.get("error", "?"),
            )
        else:
            stats["edge_cases"] += 1

    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agenthandover-worker",
        description="AgentHandover apprentice worker process",
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
        choices=list(EXPORT_ADAPTERS),
        default=None,
        help="SOP export adapter (default: [export] adapter in config.toml, else openclaw)",
    )
    parser.add_argument(
        "--json-export",
        action="store_true",
        default=False,
        help="Also export SOPs as JSON (used with generic adapter)",
    )
    parser.add_argument(
        "--enhance-sops",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable LLM-enhanced SOP descriptions (default: from config)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM model for SOP enhancement",
    )
    parser.add_argument(
        "--max-enhancements-per-day",
        type=int,
        default=None,
        help="Max SOP enhancements per day (default: from config)",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=KNOWLEDGE_BASE_DIR,
        help=(
            "Path to the knowledge base directory "
            f"(default: {KNOWLEDGE_BASE_DIR})"
        ),
    )
    parser.add_argument(
        "--backfill-procedures",
        action="store_true",
        default=False,
        help=(
            "Repair saved procedures (fill empty descriptions and step "
            "fields lost by older versions), print the count, and exit"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help=(
            "Run one full poll iteration and exit; worker-status.json is "
            "kept as the record of that iteration"
        ),
    )
    return parser.parse_args(argv)


def _next_poll_interval(
    current: float, base: float, *, busy: bool, triggers_consumed: int,
) -> float:
    """Seconds to sleep before the next worker tick.

    Work found or a consumed trigger (someone is using the app or CLI)
    resets to *base*; an idle tick doubles, up to ``max(5, 4 * base)`` so
    a request written after a long idle stretch is picked up within
    seconds.
    """
    if busy or triggers_consumed:
        return base
    return min(current * 2, max(5.0, base * 4))


def _log_handlers(log_dir: Path, *, stderr_is_tty: bool) -> list[logging.Handler]:
    """The worker's log handlers: one rotating ``worker.log``, plus stderr
    only for a manual run in a terminal.

    Under launchd, stderr is ``worker.stderr.log``; a stderr handler there
    wrote every line twice (the two files were byte-identical).  That file
    now keeps only what bypasses logging: tracebacks of a crash and output
    of native libraries.
    """
    from logging.handlers import RotatingFileHandler

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_dir / "worker.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handlers: list[logging.Handler] = [file_handler]
    if stderr_is_tty:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(formatter)
    return handlers


def _poll_log_level(
    counts: tuple[int, int], last_logged: tuple[int, int] | None,
) -> int:
    """INFO for a poll whose (unprocessed events, pending VLM jobs) counts
    changed since the last one logged, DEBUG for a repeat.  A job the
    worker cannot finish kept the same line at INFO every 2 s (~43k
    lines a day)."""
    return logging.INFO if counts != last_logged else logging.DEBUG


def _status_dir() -> Path:
    """Return the standard data directory for status/PID files."""
    return data_dir()


def _migrate_vlm_config(path: Path | None = None) -> bool:
    """Move config.toml's ``[vlm]`` block off a tier's earlier default
    models (``model_profiles.migrate_vlm_file``: backup first, atomic
    write, only when it changes).  One INFO line when it changes
    something; a failure is a warning and never stops the worker.
    Returns whether the file changed."""
    target = config_path() if path is None else path
    if not target.is_file():
        return False
    try:
        changed, reason, backup = migrate_vlm_file(target)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Could not check the [vlm] models in %s against the tier table: %s", target, exc,
        )
        return False
    if changed:
        logger.info("Migrated [vlm] in %s: %s; backup: %s", target, reason, backup)
    return changed


def _atomic_write_result(path: Path, data: dict) -> None:
    """Atomically write a JSON result file (tmp + fsync + rename).

    Used for trigger result files that the CLI polls for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".result.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_worker_status(
    *,
    started_at: str,
    events_processed_today: int,
    sops_generated: int,
    last_pipeline_duration_ms: int | None,
    consecutive_errors: int,
    vlm_available: bool,
    vlm_queue_pending: int = 0,
    vlm_jobs_today: int = 0,
    vlm_dropped_today: int = 0,
    vlm_mode: str | None = None,
    vlm_provider: str | None = None,
    v2_annotations_today: int = 0,
    v2_diffs_today: int = 0,
    v2_annotation_enabled: bool = False,
    annotation_paused_reason: str | None = None,
    annotation_budget: dict | None = None,
    model_schedule: dict | None = None,
) -> None:
    """Atomically write worker-status.json (tmp + fsync + rename).

    ``annotation_paused_reason`` says why passive annotation is not running
    (``annotation_budget_exhausted`` for the day,
    ``annotation_hourly_budget_exhausted`` for the rolling hour,
    ``on_battery_user_active``), or is None while it runs.
    """
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
        "vlm_queue_pending": vlm_queue_pending,
        "vlm_jobs_today": vlm_jobs_today,
        "vlm_dropped_today": vlm_dropped_today,
    }
    if vlm_mode:
        status["vlm_mode"] = vlm_mode
    if vlm_provider:
        status["vlm_provider"] = vlm_provider
    if v2_annotation_enabled:
        status["v2_annotation_enabled"] = True
        status["v2_annotations_today"] = v2_annotations_today
        status["v2_diffs_today"] = v2_diffs_today
        status["annotation_paused_reason"] = annotation_paused_reason
        if annotation_budget is not None:
            status["annotation_budget"] = annotation_budget
    if model_schedule is not None:
        # Which model writes Skills, and why generation waits (memory, a
        # failed load), on a tier with two models.
        status["model_schedule"] = model_schedule
    # Tokens and Ollama timings per model since the worker started, and
    # model seconds per call type (since start and in the last hour).
    status["model_calls"] = MODEL_USAGE.snapshot()
    status["compute"] = COMPUTE.snapshot()
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
        logger.warning("Failed to write worker-status.json", exc_info=True)
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


# ------------------------------------------------------------------
# SOP index file — consumed by the SwiftUI menu bar app
# ------------------------------------------------------------------

_last_sops_index_write: float = 0.0


def _derive_tags(sop_tmpl: dict) -> list[str]:
    """Derive category tags from SOP template data (apps, title, steps)."""
    tags: list[str] = []
    apps = [a.lower() for a in sop_tmpl.get("apps_involved", [])]
    title_lower = sop_tmpl.get("title", "").lower()
    desc_lower = sop_tmpl.get("task_description", "").lower()
    combined = title_lower + " " + desc_lower

    # App-based tags
    browser_apps = {"chrome", "google chrome", "safari", "firefox", "brave", "edge"}
    dev_apps = {"visual studio code", "vs code", "terminal", "iterm", "xcode"}
    comm_apps = {"gmail", "google gmail", "slack", "mail", "messages", "outlook"}

    if any(a in browser_apps for a in apps) or "browse" in combined:
        tags.append("browsing")
    if any(a in dev_apps for a in apps) or any(
        w in combined for w in ("commit", "git", "code", "debug", "deploy")
    ):
        tags.append("development")
    if any(a in comm_apps for a in apps) or any(
        w in combined for w in ("email", "inbox", "message", "send")
    ):
        tags.append("communication")

    # Content-based tags
    if any(w in combined for w in ("spreadsheet", "docs", "document", "write", "edit doc")):
        tags.append("documentation")
    if any(w in combined for w in ("domain", "dns", "hosting", "deploy", "server")):
        tags.append("system")
    if any(w in combined for w in ("price", "cost", "invoice", "payment", "finance")):
        tags.append("finance")

    # Deduplicate and cap at 3
    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:3]


def _derive_short_title(title: str) -> str:
    """Derive a concise 3-6 word short title from a verbose title.

    Strips common noise prefixes like 'The user is...', 'User is...',
    then takes the core verb phrase.
    """
    import re as _re

    text = title.strip()
    # Strip common noise prefixes
    noise_prefixes = [
        r"^the user is\s+",
        r"^user is\s+",
        r"^the user\s+",
        r"^this (?:sop|workflow|procedure) (?:describes|documents|covers)\s+",
    ]
    for pattern in noise_prefixes:
        text = _re.sub(pattern, "", text, flags=_re.IGNORECASE)

    # Capitalise first letter
    if text:
        text = text[0].upper() + text[1:]

    # Take first 6 words, try to cut at a natural break
    words = text.split()
    if len(words) <= 6:
        return text.rstrip(".")
    short = " ".join(words[:6])
    # Try to avoid ending mid-phrase — cut at common break words
    for i in range(min(6, len(words)) - 1, 2, -1):
        if words[i].lower() in ("and", "or", "by", "for", "to", "in", "on",
                                  "with", "from", "the", "a", "an", "of"):
            short = " ".join(words[:i])
            break
    return short.rstrip(".,;:")


def _embed_in_vector_kb(
    vector_kb,
    source_type: str,
    source_id: str,
    text: str,
) -> None:
    """Embed a piece of content in the vector KB if available.

    Silent no-op if vector_kb is None or embedding fails.
    """
    if vector_kb is None or not text:
        return
    try:
        vector_kb.upsert(source_type, source_id, text[:4000])
    except Exception:
        logger.debug("Embedding failed for %s/%s", source_type, source_id, exc_info=True)


def _embed_procedure(vector_kb, procedure: dict) -> None:
    """Embed a procedure's title + description + steps into the vector KB."""
    if vector_kb is None:
        return
    slug = procedure.get("id", procedure.get("slug", ""))
    if not slug:
        return

    # Build a rich text representation for semantic search
    parts = []
    title = procedure.get("title", "")
    if title:
        parts.append(title)
    desc = procedure.get("description", "")
    if desc:
        parts.append(desc)
    strategy = procedure.get("strategy", "")
    if strategy:
        parts.append(f"Strategy: {strategy}")
    for step in procedure.get("steps", []):
        action = step.get("action", step.get("description", ""))
        if action:
            parts.append(action)

    text = "\n".join(parts)
    _embed_in_vector_kb(vector_kb, "procedure", slug, text)


def _refresh_claude_commands(
    commands_dir: Path | None = None,
    skills_dir: Path | None = None,
) -> None:
    """Make Claude Code see exactly the agent_ready procedures.

    Writes ``<skills>/<slug>/SKILL.md`` for every agent_ready procedure
    in the knowledge base, removes exported Skills whose procedure is no
    longer agent_ready, and rewrites the ``agenthandover-list`` command
    that points at them (``agent_connect.connect_claude_code``).  *commands_dir* and
    *skills_dir* override ``~/.claude/commands`` and the configured
    Skills root.  Never raises: a failed refresh must not undo the
    lifecycle change or approval that asked for it.
    """
    try:
        from agenthandover_worker.agent_connect import connect_claude_code
        connect_claude_code(
            commands_dir=commands_dir,
            skills_dir=skills_dir if skills_dir is not None else load_config().export.skills_dir,
        )
        logger.info("Refreshed Claude Code Skills")
    except Exception:
        logger.warning("Failed to refresh Claude Code commands", exc_info=True)


def _write_sops_index(
    db: WorkerDB,
    knowledge_base: "KnowledgeBase | None" = None,
    *,
    force: bool = False,
) -> None:
    """Atomically write sops-index.json for the SwiftUI app to read.

    Throttled to at most once every 5 seconds unless *force* is True.
    """
    global _last_sops_index_write
    now = time.time()
    if not force and (now - _last_sops_index_write) < 5.0:
        return

    try:
        all_sops_raw = db.get_generated_sops()
        failed = db.get_failed_generations()

        # Surface v2 pipeline SOPs and focus recordings.  ``v1_mined`` rows
        # were queued by the removed v1 PrefixSpan miner; they stay listed
        # so a user can still reject the drafts it left behind.
        _REVIEWABLE_SOURCES = {
            "v2_focus_recording", "v2_passive_discovery", "focus", "v1_mined",
        }
        all_sops = [
            s for s in all_sops_raw
            if s.get("source", "") in _REVIEWABLE_SOURCES
        ]

        # Also include KB procedures from focus sessions that may not
        # be in the generated_sops DB table yet.
        existing_slugs = {s.get("slug", "") for s in all_sops}
        try:
            kb_procs = knowledge_base.list_procedures()
        except Exception:
            kb_procs = []
        kb_by_slug = {proc.get("id", ""): proc for proc in kb_procs}
        try:
            for proc in kb_procs:
                slug = proc.get("id", "")
                source = proc.get("source", "")
                if slug and slug not in existing_slugs and source in _REVIEWABLE_SOURCES:
                    all_sops.append({
                        "slug": slug,
                        "title": proc.get("title", slug),
                        "source": source,
                        "status": "draft",
                        "confidence": proc.get("confidence_avg", 0),
                        "source_id": "",
                        "created_at": proc.get("generated_at", ""),
                    })
                    existing_slugs.add(slug)
        except Exception:
            pass

        # draft_count and approved_count computed after dedup below

        # Load full sop_json for each SOP to extract short_title/tags
        sop_templates: dict[str, dict] = {}
        for s in all_sops:
            sop_id = s.get("sop_id", "")
            if sop_id:
                full_record = db.get_generated_sop(sop_id)
                if full_record and full_record.get("sop_json"):
                    sop_json = full_record["sop_json"]
                    if isinstance(sop_json, str):
                        try:
                            sop_templates[sop_id] = json.loads(sop_json)
                        except (json.JSONDecodeError, TypeError):
                            sop_templates[sop_id] = {}
                    elif isinstance(sop_json, dict):
                        sop_templates[sop_id] = sop_json

        sop_entries = []
        for s in all_sops:
            sop_id = s.get("sop_id", "")
            sop_tmpl = sop_templates.get(sop_id, {})

            title = s.get("title", "Untitled")
            short_title = sop_tmpl.get("short_title", "")
            if not short_title:
                short_title = _derive_short_title(title)
            tags = sop_tmpl.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            # Derive tags from apps_involved if not set
            if not tags:
                tags = _derive_tags(sop_tmpl)

            # Look up lifecycle_state from KB procedure
            slug = s.get("slug", "")
            kb_proc = kb_by_slug.get(slug) if slug else None
            lifecycle_state = (
                kb_proc.get("lifecycle_state", "observed") if kb_proc else "observed"
            )

            sop_entries.append({
                "sop_id": s.get("sop_id", ""),
                "slug": slug,
                "title": title,
                "short_title": short_title,
                "tags": tags,
                "source": s.get("source", ""),
                "status": s.get("status", "draft"),
                "confidence": s.get("confidence", 0.0) or 0.0,
                "created_at": s.get("created_at", ""),
                "reviewed_at": s.get("reviewed_at"),
                "lifecycle_state": lifecycle_state,
            })

        # Deduplicate by slug — keep highest confidence entry per slug
        seen_slugs: dict[str, int] = {}
        deduped: list[dict] = []
        for entry in sop_entries:
            slug = entry.get("slug", "")
            if not slug:
                deduped.append(entry)
                continue
            conf = entry.get("confidence", 0.0)
            if slug in seen_slugs:
                existing_idx = seen_slugs[slug]
                if conf > deduped[existing_idx].get("confidence", 0.0):
                    deduped[existing_idx] = entry
            else:
                seen_slugs[slug] = len(deduped)
                deduped.append(entry)
        sop_entries = deduped

        draft_count = sum(1 for s in sop_entries if s.get("status") == "draft")
        approved_count = sum(1 for s in sop_entries if s.get("status") == "approved")

        index = {
            "updated_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "sops": sop_entries,
            "failed_count": len(failed),
            "draft_count": draft_count,
            "approved_count": approved_count,
        }

        sdir = _status_dir()
        sdir.mkdir(parents=True, exist_ok=True)
        target = sdir / "sops-index.json"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(sdir), prefix=".sops-index.", suffix=".tmp"
        )
        with os.fdopen(fd, "w") as f:
            json.dump(index, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(target))
        _last_sops_index_write = now
    except Exception:
        logger.debug("Failed to write sops-index.json", exc_info=True)
        try:
            os.unlink(tmp_path)  # type: ignore[possibly-undefined]
        except Exception:
            pass


def _write_curation_queue(procedure_curator: "ProcedureCurator | None" = None) -> bool:
    """Write curation-queue.json for the SwiftUI app to read.

    Contains merge candidates and other curation items that the app
    cannot derive from procedure JSON alone.  Called after a trigger was
    consumed and by the daily curation pass; the file is rewritten only
    when its merge candidates differ from what it holds.  Returns True
    when it wrote the file.
    """
    if procedure_curator is None:
        return False

    try:
        merges = procedure_curator.detect_merge_candidates()
        merge_list = [
            {
                "slug_a": m.slug_a,
                "slug_b": m.slug_b,
                "similarity": round(m.similarity, 4),
                "explanation": m.explanation,
                "shared_apps": m.shared_apps,
            }
            for m in merges
        ]

        sdir = _status_dir()
        target = sdir / "curation-queue.json"
        try:
            with open(target) as f:
                if json.load(f).get("merge_candidates") == merge_list:
                    return False
        except (OSError, ValueError, AttributeError):
            pass

        queue_data = {
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "merge_candidates": merge_list,
        }
        sdir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(sdir), prefix=".curation-queue.", suffix=".tmp"
        )
        with os.fdopen(fd, "w") as f:
            json.dump(queue_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(target))
        return True
    except Exception:
        logger.debug("Failed to write curation-queue.json", exc_info=True)
        try:
            os.unlink(tmp_path)  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return False


def run_pipeline(
    events: list[dict],
    *,
    episode_builder: EpisodeBuilder,
    clipboard_linker: ClipboardLinker,
    pruner: NegativeDemoPruner,
    translator: SemanticTranslator,
    scorer: ConfidenceScorer,
    vlm_queue: VLMFallbackQueue,
    db: "WorkerDB | None" = None,
) -> dict:
    """Build, prune, translate and score a batch of events (steps D->E).

    Translations the scorer rejects are queued for the VLM fallback.  No
    SOP comes out of this path: the v1 PrefixSpan miner that used to run
    here on every poll is gone (WP-F5), and passive recurrence belongs to
    ``workflow_mining``.

    Returns a summary dict with counts for monitoring.
    """
    summary: dict = {
        "events_in": len(events),
        "episodes": 0,
        "positive_events": 0,
        "negative_events": 0,
        "translations": 0,
        "vlm_enqueued": 0,
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
        for idx, tr in enumerate(translations):
            # Build scoring context from PREVIOUS step's post_state so the
            # context match score reflects an independent expectation rather
            # than trivially matching pre_state against itself.
            context: dict = {}
            if idx > 0:
                prev_tr = translations[idx - 1]
                prev_post = prev_tr.post_state if hasattr(prev_tr, "post_state") else {}
                if prev_post.get("window_title"):
                    context["expected_title"] = prev_post["window_title"]
                if prev_post.get("url"):
                    context["expected_url"] = prev_post["url"]
                if prev_post.get("app_id"):
                    context["expected_app"] = prev_post["app_id"]

            # Check clipboard provenance
            raw_event_id = tr.raw_event_id
            if raw_event_id in paste_ids_with_provenance:
                context["clipboard_link"] = True

            # Check dwell snapshot provenance
            if tr.intent == "read":
                context["dwell_snapshot"] = True

            # Apply VLM confidence boost if a completed VLM job exists
            # for this event (reconciliation after VLM processing)
            if db is not None:
                vlm_boost = db.get_completed_vlm_boost(tr.raw_event_id)
                if vlm_boost > 0.0:
                    context["vlm_boost"] = vlm_boost

            conf = scorer.score(tr, context)

            # Auto-enqueue VLM job for rejected translations —
            # but skip if there's already a completed VLM job for this event
            # (avoid infinite re-enqueue loops)
            has_vlm = db is not None and db.has_completed_vlm_job(tr.raw_event_id)
            if conf.decision == "reject" and conf.total < VLM_REJECT_THRESHOLD and not has_vlm:
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

    summary["translations"] = len(all_translations)
    logger.info(
        "Translated %d events, enqueued %d VLM jobs",
        len(all_translations),
        summary["vlm_enqueued"],
    )

    return summary


def _process_focus_sessions(
    db: "WorkerDB",
    *,
    episode_builder: EpisodeBuilder,
    clipboard_linker: ClipboardLinker,
    translator: SemanticTranslator,
    scorer: "ConfidenceScorer",
    vlm_queue: "VLMFallbackQueue",
    sop_enhancer: object | None,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
) -> int:
    """Process completed focus recording sessions.

    Queries for events tagged with focus_session_id, groups them by session,
    runs through the episode builder + translator, and calls
    ``sop_inducer.induce_from_focus_session()`` to produce a SOP from a
    single demonstration.

    Note: ``NegativeDemoPruner`` is deliberately skipped here because focus
    sessions are single-shot demonstrations — there are no negative counter-
    examples to learn from.

    Returns the number of SOPs exported from focus sessions.
    """
    state_dir = _status_dir()
    signal_path = state_dir / "focus-session.json"

    if not signal_path.is_file():
        return 0

    try:
        with open(signal_path) as f:
            signal = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read focus-session.json", exc_info=True)
        return 0

    if signal.get("status") != "stopped":
        return 0

    session_id = signal.get("session_id", "")
    title = signal.get("title", "Untitled focus session")

    if not session_id:
        return 0

    # Query events tagged with this focus session ID
    focus_events = db.get_focus_session_events(session_id)
    if not focus_events:
        import time as _time
        try:
            file_age = _time.time() - signal_path.stat().st_mtime
        except OSError:
            file_age = 0
        if file_age > 300:
            logger.warning(
                "Focus session '%s' (%s) has 0 events %.0fs after stop — cleaning up.",
                title, session_id, file_age,
            )
            _clear_focus_signal(signal_path)
            return 0

        logger.info(
            "Focus session '%s' (%s) has no events yet, will retry next cycle",
            title,
            session_id,
        )
        return 0

    logger.info(
        "Processing focus session '%s' (%s): %d events",
        title,
        session_id,
        len(focus_events),
    )

    # Run through episode builder
    episodes = episode_builder.process_events(focus_events)
    if not episodes:
        logger.warning("Focus session '%s' produced no episodes", title)
        # Clear the signal so we don't retry endlessly
        _clear_focus_signal(signal_path)
        return 0

    # Translate episodes into semantic steps
    episode_sop_steps: list[list[dict]] = []
    for ep in episodes:
        translations = translator.translate_batch(ep.events)
        sop_steps: list[dict] = []
        for idx, tr in enumerate(translations):
            # Context from previous step's post_state (see normal pipeline comment)
            context: dict = {}
            if idx > 0:
                prev_tr = translations[idx - 1]
                prev_post = prev_tr.post_state if hasattr(prev_tr, "post_state") else {}
                if prev_post.get("window_title"):
                    context["expected_title"] = prev_post["window_title"]
                if prev_post.get("app_id"):
                    context["expected_app"] = prev_post["app_id"]

            conf = scorer.score(tr, context)
            if conf.decision in ("accept", "accept_flagged"):
                target_desc = ""
                selector = None
                if tr.target:
                    target_desc = tr.target.selector
                    selector = tr.target.selector

                sop_steps.append({
                    "step": tr.intent,
                    "target": target_desc,
                    "selector": selector,
                    "parameters": tr.parameters,
                    "confidence": conf.total,
                    "pre_state": tr.pre_state,
                })
        if sop_steps:
            episode_sop_steps.append(sop_steps)

    if not episode_sop_steps:
        logger.warning("Focus session '%s' produced no SOP steps", title)
        _clear_focus_signal(signal_path)
        return 0

    # One SOP from the single demonstration
    from agenthandover_worker.sop_inducer import induce_from_focus_session

    sop_templates: list[dict] = []
    try:
        sop_templates = induce_from_focus_session(episode_sop_steps, title)
    except Exception:
        logger.exception("Focus session SOP induction failed for '%s'", title)

    # Enhance with LLM descriptions
    if sop_enhancer is not None and sop_templates:
        for template in sop_templates:
            try:
                enhanced = sop_enhancer.enhance_sop(template)
                if "task_description" in enhanced:
                    template.update(enhanced)
            except Exception:
                logger.debug(
                    "Focus SOP enhancement failed for '%s'",
                    template.get("slug", "unknown"),
                    exc_info=True,
                )

    # Lint / validate SOPs before export
    if sop_templates:
        valid_templates: list[dict] = []
        for template in sop_templates:
            if _lint_and_log(template, "focus"):
                valid_templates.append(template)
            else:
                # Save invalid SOPs as drafts for debugging, but skip export
                try:
                    db.save_generated_sop(
                        slug=template.get("slug", ""),
                        title=title,
                        source="focus",
                        sop_template=template,
                        confidence=0.0,
                        source_id=session_id,
                        auto_approve=False,
                    )
                except Exception:
                    logger.debug(
                        "Failed to save invalid SOP draft to DB", exc_info=True,
                    )
        sop_templates = valid_templates

    # Save generated SOPs to DB for review tracking
    exported = 0
    if sop_templates:
        for template in sop_templates:
            slug = template.get("slug", "")
            try:
                db.save_generated_sop(
                    slug=slug,
                    title=title,
                    source="focus",
                    sop_template=template,
                    confidence=template.get("confidence_avg", template.get("confidence", 0.0)),
                    source_id=session_id,
                    auto_approve=sop_auto_approve,
                )
            except Exception:
                logger.warning(
                    "Failed to save generated SOP to DB for '%s'",
                    title, exc_info=True,
                )

        # Only export if auto_approve is enabled
        if sop_auto_approve:
            # Deduplicate against known SOPs
            from agenthandover_worker.sop_dedup import deduplicate_templates
            sop_templates = deduplicate_templates(sop_templates, _status_dir())

            # Write v3 procedures to the knowledge base first so
            # all adapters (including OpenClaw) can use the richer v3 format.
            if procedure_writer is not None:
                for tpl in sop_templates:
                    try:
                        procedure_writer.write_procedure(
                            tpl,
                            source="sop_pipeline",
                            source_id=tpl.get("slug", "unknown"),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to write procedure for %s",
                            tpl.get("slug", "?"), exc_info=True,
                        )

            # Export via OpenClaw (uses v3 procedure from KB when available)
            try:
                paths = _export_via_adapter(
                    openclaw_writer, sop_templates, procedure_writer,
                )
                exported = len(paths)
                logger.info(
                    "Focus session '%s': exported %d SOP(s)", title, exported
                )
            except Exception:
                logger.exception("Focus session SOP export failed")

            # Also export as SKILL.md if writer is configured
            # (uses v3 procedure from KB when available)
            if skill_md_writer is not None:
                try:
                    _export_via_adapter(
                        skill_md_writer, sop_templates, procedure_writer,
                    )
                    logger.info(
                        "Focus session '%s': SKILL.md export complete", title
                    )
                except Exception:
                    logger.exception("Focus session SKILL.md export failed")

            # Also export as Claude Code skills if writer is configured
            # (uses v3 procedure from KB when available)
            if claude_skill_writer is not None:
                try:
                    _export_via_adapter(
                        claude_skill_writer, sop_templates, procedure_writer,
                    )
                    logger.info(
                        "Focus session '%s': Claude skill export complete", title
                    )
                except Exception:
                    logger.exception("Focus session Claude skill export failed")

            if kb_export_adapter is not None:
                try:
                    kb_export_adapter.write_all_sops(sop_templates)
                except Exception:
                    logger.debug("KB export adapter failed", exc_info=True)

            # Cache focus session SOPs for CLI export trigger
            _save_sop_cache(sop_templates)
        else:
            logger.info(
                "Focus session '%s': SOP(s) saved as draft (auto_approve=False)",
                title,
            )
    else:
        # SOP induction ran but produced no templates — record failure
        try:
            db.record_failed_generation(
                source="focus",
                source_id=session_id,
                error="SOP induction produced no templates",
                title=title,
                context={"event_count": len(focus_events)},
            )
        except Exception:
            logger.debug(
                "Failed to record generation failure for '%s'",
                title, exc_info=True,
            )

    # Clear the signal file so it's not reprocessed
    _clear_focus_signal(signal_path)

    return exported


def _process_focus_sessions_v2(
    db: "WorkerDB",
    *,
    focus_processor: "FocusProcessor",
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    screenshots_dir: str | Path = "",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
    behavioral_synthesizer: "BehavioralSynthesizer | None" = None,
    focus_questioner: "FocusQuestioner | None" = None,
    knowledge_base=None,
    vector_kb=None,
    ollama_host: str = "http://localhost:11434",
    qna_model: str = "qwen3.5:4b",
) -> int:
    """Process completed focus recording sessions via v2 VLM pipeline.

    Uses the scene annotator + frame differ + SOP generator instead of
    the v1 episode builder + translator path.  Produces
    semantic SOPs with exact screen-observed details.

    Enhanced flow:
      1. Check for pending focus Q&A answers from a previous cycle.
         If answered, merge answers into the saved procedure and export.
      2. Process new focus-session.json signal (SOP generation).
      3. Run behavioral synthesis with force=True (single demo).
      4. Generate 1-3 targeted questions via focus_questioner.
      5. If questions exist, pause export and write Q&A files for the CLI.
      6. If no questions (or questioner unavailable), export immediately.

    Returns the number of SOPs exported.
    """
    state_dir = _status_dir()

    # --- Check for pending focus Q&A answers from a previous cycle ---
    pending_export = _check_focus_answers(
        state_dir,
        db=db,
        focus_questioner=focus_questioner,
        openclaw_writer=openclaw_writer,
        skill_md_writer=skill_md_writer,
        claude_skill_writer=claude_skill_writer,
        index_generator=index_generator,
        sop_auto_approve=sop_auto_approve,
        procedure_writer=procedure_writer,
        kb_export_adapter=kb_export_adapter,
        knowledge_base=knowledge_base,
        vector_kb=vector_kb,
    )
    if pending_export is not None:
        return pending_export

    # --- Process new focus session signal ---
    signal_path = state_dir / "focus-session.json"

    if not signal_path.is_file():
        return 0

    try:
        with open(signal_path) as f:
            signal = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read focus-session.json", exc_info=True)
        return 0

    if signal.get("status") != "stopped":
        return 0

    session_id = signal.get("session_id", "")
    title = signal.get("title", "Untitled focus session")

    if not session_id:
        return 0

    # Query events tagged with this focus session ID
    focus_events = db.get_focus_session_events(session_id)
    if not focus_events:
        # Use the signal file's mtime (= when session was stopped/written)
        # not started_at, so long recordings don't get cleaned up prematurely.
        import time as _time
        try:
            file_age = _time.time() - signal_path.stat().st_mtime
        except OSError:
            file_age = 0
        if file_age > 300:
            logger.warning(
                "Focus v2 session '%s' (%s) has 0 events %.0fs after stop — "
                "daemon was likely not running. Cleaning up.",
                title, session_id, file_age,
            )
            _clear_focus_signal(signal_path)
            return 0

        logger.info(
            "Focus v2 session '%s' (%s) has no events yet, will retry",
            title, session_id,
        )
        return 0

    logger.info(
        "Focus v2: processing session '%s' (%s): %d events",
        title, session_id, len(focus_events),
    )

    # Run through v2 focus processor (lint included: an SOP that fails
    # validation is regenerated once, then recorded as a failure).  A
    # session whose SOP was generated before a restart resumes from its
    # saved draft instead of generating it again.
    saved = _read_focus_draft(state_dir, session_id)
    result, lint_error = _generate_linted(
        lambda: focus_processor.process_session(
            db, session_id, title, focus_events,
            screenshots_dir=screenshots_dir, saved=saved,
        ),
        "focus_v2",
        title,
    )
    if result.deferred:
        # Ollama unreachable: keep the signal and the pending rows; the
        # next tick retries the session.
        logger.warning(
            "Focus v2 session '%s' deferred: %s", title, result.error,
        )
        return 0

    exported = 0
    if result.success and result.sop and not lint_error:
        sop_templates = [result.sop]
        slug = result.sop.get("slug", "")

        # --- Convert to v3 procedure and set user's title ---
        from agenthandover_worker.procedure_schema import sop_to_procedure
        procedure_dict = sop_to_procedure(result.sop)
        procedure_dict["id"] = slug
        procedure_dict["slug"] = slug
        procedure_dict["title"] = title  # User's focus session title, not VLM's guess
        procedure_dict["source"] = "focus"
        procedure_dict["lifecycle_state"] = "draft"  # Focus recordings start as draft, not observed
        # A re-recording of an existing Skill is a new draft, but keeps
        # its evidence, trust, lifecycle history and agent-run record.
        if knowledge_base is not None:
            from agenthandover_worker.procedure_writer import merge_existing_procedure
            existing_proc = knowledge_base.get_procedure(slug)
            if existing_proc is not None:
                merge_existing_procedure(
                    procedure_dict, existing_proc, keep_lifecycle=False,
                )
        # This recording is one instance of the procedure: the ledger's
        # evidence for what was learned from it.
        append_observation(procedure_dict, ObservationEvidence(
            date=str(focus_events[0].get("timestamp") or "")[:10], type="focus",
            duration_minutes=0, session_id=session_id, event_count=len(focus_events),
        ), once_per_session=True)
        # Save the draft now: synthesis and Q&A only enrich it, and a
        # restart before they finish must not lose the generated SOP.
        _save_focus_draft(
            db, state_dir, session_id=session_id, title=title,
            result=result, procedure=procedure_dict,
        )
        if behavioral_synthesizer is not None:
            try:
                # Prefer the rich per-frame observations stashed by the
                # focus_processor (email_addresses, typed_text, urls,
                # visible_items and the frames' screen text) so the
                # synthesizer can quote verbatim text in goal/strategy.
                # Fall back to abstract SOP steps only if rich data is
                # unavailable.
                rich_obs = getattr(focus_processor, "_last_rich_observations", None)
                if rich_obs and rich_obs[0]:
                    obs_for_synthesis = rich_obs
                else:
                    obs_for_synthesis = [result.sop.get("steps", [])]
                insights = behavioral_synthesizer.synthesize(
                    slug, procedure_dict, obs_for_synthesis,
                    force=True,  # bypass min_observations for focus
                )
                if insights.goal or insights.strategy or insights.guardrails:
                    procedure_dict = behavioral_synthesizer.merge_insights_into_procedure(
                        procedure_dict, insights,
                    )
                    # Update the SOP template with synthesized insights
                    sop_templates = [procedure_dict]
                    logger.info(
                        "Focus v2 session '%s': behavioral synthesis complete "
                        "(goal=%s, strategy=%s, %d guardrails)",
                        title, bool(insights.goal), bool(insights.strategy),
                        len(insights.guardrails),
                    )
            except Exception:
                logger.debug(
                    "Focus v2 behavioral synthesis failed for '%s'",
                    title, exc_info=True,
                )

        # --- Generate targeted questions via subprocess (crash-proof) ---
        questions = []
        if focus_questioner is not None:
            logger.info("Focus Q&A: generating questions via subprocess...")
            try:
                import subprocess as _sp
                import tempfile as _tf

                # Serialize inputs to a temp file
                _qa_input = _tf.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False,
                )
                import json as _json
                _json.dump({
                    "procedure": procedure_dict,
                    "sop": result.sop,
                }, _qa_input)
                _qa_input.close()

                _qa_output = _qa_input.name + ".out"

                # Write Q&A script to temp file (avoids -c line length issues)
                _qa_script_file = _tf.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False,
                )
                _ollama_host = ollama_host

                _qa_script_file.write(
                    "import json, sys, os\n"
                    "from dataclasses import asdict\n"
                    "from agenthandover_worker.focus_questioner import FocusQuestioner\n"
                    "from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig\n"
                    "from agenthandover_worker.vlm_queue import VLMFallbackQueue\n"
                    "data = json.load(open(sys.argv[1]))\n"
                    "host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')\n"
                    # Read the Q&A model from the env the parent sets in
                    # _qa_env below (AH_QNA_MODEL). Defaulting to the
                    # ReasoningConfig class default (qwen3.5:4b) would
                    # silently fail for Gemma-4-only users — fixed in v0.2.3.
                    "model = os.environ.get('AH_QNA_MODEL', 'qwen3.5:4b')\n"
                    "reasoner = LLMReasoner(config=ReasoningConfig(model=model, ollama_host=host), vlm_queue=VLMFallbackQueue())\n"
                    "fq = FocusQuestioner(llm_reasoner=reasoner)\n"
                    "qs = fq.generate_questions(data['procedure'], data['sop'])\n"
                    "out = [asdict(q) if hasattr(q, '__dataclass_fields__') else q for q in qs]\n"
                    "json.dump(out, open(sys.argv[2], 'w'))\n"
                )
                _qa_script_file.close()

                # Run in separate process with 120s timeout, passing
                # Ollama host + Q&A model through env.
                _qa_env = dict(os.environ)
                _qa_env["OLLAMA_HOST"] = _ollama_host
                _qa_env["AH_QNA_MODEL"] = qna_model
                _qa_result = _sp.run(
                    [sys.executable, _qa_script_file.name, _qa_input.name, _qa_output],
                    timeout=1800,  # 30 min — 16GB machines can take 10+ min per Qwen call
                    capture_output=True,
                    env=_qa_env,
                )

                if _qa_result.returncode == 0:
                    try:
                        questions = _json.load(open(_qa_output))
                        logger.info(
                            "Focus Q&A subprocess completed: %d question(s)",
                            len(questions),
                        )
                    except FileNotFoundError:
                        logger.warning(
                            "Focus Q&A subprocess exited 0 but output file "
                            "not found: %s — stderr: %s",
                            _qa_output,
                            (_qa_result.stderr or b"")[:500].decode(
                                "utf-8", errors="replace"
                            ),
                        )
                        questions = []
                    except Exception as _qa_parse_err:
                        logger.warning(
                            "Focus Q&A subprocess exited 0 but output "
                            "unreadable (%s): %s — stderr: %s",
                            type(_qa_parse_err).__name__,
                            _qa_parse_err,
                            (_qa_result.stderr or b"")[:500].decode(
                                "utf-8", errors="replace"
                            ),
                        )
                        questions = []
                else:
                    logger.warning(
                        "Focus Q&A subprocess failed (exit %d): %s",
                        _qa_result.returncode,
                        (_qa_result.stderr or b"")[:500].decode(
                            "utf-8", errors="replace"
                        ),
                    )

                # Cleanup temp files
                os.unlink(_qa_input.name)
                os.unlink(_qa_script_file.name)
                try:
                    os.unlink(_qa_output)
                except FileNotFoundError:
                    pass

                if questions:
                    from agenthandover_worker.focus_questioner import (
                        write_focus_questions,
                        write_focus_pending,
                    )
                    write_focus_questions(
                        state_dir, session_id, slug, questions,
                    )
                    write_focus_pending(
                        state_dir, session_id, slug,
                        sop_template=result.sop,
                        procedure=procedure_dict,
                    )
                    logger.info(
                        "Focus v2 session '%s': %d question(s) generated, "
                        "pausing export for user answers",
                        title, len(questions),
                    )
                    # The draft stays a draft until the answers: it now
                    # carries the synthesis.
                    if not db.update_sop_json(slug, procedure_dict):
                        logger.warning(
                            "No draft record to update for '%s'", slug,
                        )
                    _clear_focus_signal(signal_path)
                    (state_dir / FOCUS_DRAFT_FILE).unlink(missing_ok=True)
                    return 0  # Don't export yet — wait for answers
            except Exception:
                logger.warning(
                    "Focus Q&A failed for '%s' — proceeding without questions",
                    title, exc_info=True,
                )

        # --- Standard export path (no questions or questioner unavailable) ---

        # Save procedure to KB so detail view can display steps/strategy
        try:
            knowledge_base.save_procedure(procedure_dict)
            logger.info("Saved procedure '%s' to knowledge base", slug)
            _embed_procedure(vector_kb, procedure_dict)
        except Exception:
            logger.debug("Failed to save procedure to KB", exc_info=True)

        # The draft record now carries the synthesis; auto-approve
        # approves it.
        if not db.update_sop_json(slug, procedure_dict):
            logger.warning("No draft record to update for '%s'", slug)
        elif sop_auto_approve:
            record = db.get_generated_sop_by_slug(slug)
            if record is not None:
                db.update_sop_status(record["sop_id"], "approved")

        # Only export if auto_approve is enabled
        if sop_auto_approve:
            exported = _focus_v2_export(
                sop_templates=sop_templates,
                title=title,
                inference_time=result.inference_time_seconds,
                openclaw_writer=openclaw_writer,
                skill_md_writer=skill_md_writer,
                claude_skill_writer=claude_skill_writer,
                index_generator=index_generator,
                procedure_writer=procedure_writer,
                kb_export_adapter=kb_export_adapter,
            )
            _refresh_claude_commands()
        else:
            logger.info(
                "Focus v2 session '%s': SOP saved as draft (auto_approve=False)",
                title,
            )
    else:
        # Record failure for retry tracking
        try:
            db.record_failed_generation(
                source="focus",
                source_id=session_id,
                error=lint_error or result.error or "Unknown error",
                title=title,
                context={"event_count": len(focus_events)},
            )
        except Exception:
            logger.debug(
                "Failed to record generation failure for '%s'",
                title, exc_info=True,
            )
        logger.warning(
            "Focus v2 session '%s' SOP generation failed: %s",
            title, lint_error or result.error,
        )

    # Clear the signal and the saved draft state.  The keyframes the SOP
    # did not need are annotated in the next tick's annotation phase
    # (``annotate_remaining``).
    _clear_focus_signal(signal_path)
    (state_dir / FOCUS_DRAFT_FILE).unlink(missing_ok=True)
    return exported


# The last focus session's generated SOP until the session is exported or
# waits for answers (``focus-pending.json`` then holds it).
FOCUS_DRAFT_FILE = "focus-draft.json"


def _read_focus_draft(state_dir: Path, session_id: str) -> "GeneratedSOP | None":
    """The SOP generated for *session_id* before a restart, from the draft
    state ``_save_focus_draft`` wrote, or None."""
    from agenthandover_worker.sop_generator import GeneratedSOP

    try:
        draft = json.loads((state_dir / FOCUS_DRAFT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(draft, dict) or draft.get("session_id") != session_id:
        return None
    sop = draft.get("sop_template")
    if not isinstance(sop, dict) or not sop:
        return None
    return GeneratedSOP(
        sop=sop, inference_time_seconds=float(draft.get("inference_time_seconds") or 0.0),
    )


def _save_focus_draft(
    db: "WorkerDB",
    state_dir: Path,
    *,
    session_id: str,
    title: str,
    result: "GeneratedSOP",
    procedure: dict,
) -> None:
    """Persist a focus session's SOP as soon as it is generated: a draft
    record in the review queue (updated, not duplicated, when the session
    resumes after a restart) and the state ``_read_focus_draft`` resumes
    from.  Synthesis, Q&A and approval update the record afterwards."""
    slug = procedure["slug"]
    try:
        _atomic_write_result(state_dir / FOCUS_DRAFT_FILE, {
            "session_id": session_id,
            "sop_template": result.sop,
            "inference_time_seconds": result.inference_time_seconds,
        })
    except OSError:
        logger.warning(
            "Could not save the draft state of '%s'; a restart before its "
            "export generates it again", slug, exc_info=True,
        )
    record = db.get_generated_sop_by_slug(slug)
    if record is not None and record.get("source_id") == session_id:
        db.update_sop_json(slug, procedure)
        return
    db.save_generated_sop(
        slug=slug,
        title=title,
        source="focus",
        sop_template=procedure,
        confidence=procedure.get(
            "confidence_avg", result.sop.get("confidence_avg", 0.5),
        ),
        source_id=session_id,
        auto_approve=False,
    )


def _focus_v2_export(
    *,
    sop_templates: list[dict],
    title: str,
    inference_time: float,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
) -> int:
    """Shared export logic for focus v2 sessions.

    Extracted from _process_focus_sessions_v2 to be reusable by both
    the immediate export path and the deferred Q&A answer path.

    Returns the number of SOPs exported.
    """
    # Deduplicate against known SOPs
    from agenthandover_worker.sop_dedup import deduplicate_templates
    sop_templates = deduplicate_templates(sop_templates, _status_dir())

    # Write v3 procedures to the knowledge base first so
    # all adapters (including OpenClaw) can use the richer v3 format.
    if procedure_writer is not None:
        for tpl in sop_templates:
            try:
                procedure_writer.write_procedure(
                    tpl,
                    source="sop_pipeline",
                    source_id=tpl.get("slug", "unknown"),
                )
            except Exception:
                logger.debug(
                    "Failed to write procedure for %s",
                    tpl.get("slug", "?"), exc_info=True,
                )

    exported = 0

    # Export via OpenClaw (uses v3 procedure from KB when available)
    try:
        paths = _export_via_adapter(
            openclaw_writer, sop_templates, procedure_writer,
        )
        exported = len(paths)
        logger.info(
            "Focus v2 session '%s': exported %d SOP(s) (%.1fs VLM time)",
            title, exported, inference_time,
        )
    except Exception:
        logger.exception("Focus v2 SOP export failed")

    # Also export as SKILL.md (uses v3 procedure when available)
    if skill_md_writer is not None:
        try:
            _export_via_adapter(
                skill_md_writer, sop_templates, procedure_writer,
            )
            logger.info("Focus v2 session '%s': SKILL.md export complete", title)
        except Exception:
            logger.exception("Focus v2 SKILL.md export failed")

    # Also export as Claude Code skills (uses v3 procedure when available)
    if claude_skill_writer is not None:
        try:
            _export_via_adapter(
                claude_skill_writer, sop_templates, procedure_writer,
            )
            logger.info("Focus v2 session '%s': Claude skill export complete", title)
        except Exception:
            logger.exception("Focus v2 Claude skill export failed")

    if kb_export_adapter is not None:
        try:
            kb_export_adapter.write_all_sops(sop_templates)
        except Exception:
            logger.debug("KB export adapter failed", exc_info=True)

    # Cache for CLI export trigger
    _save_sop_cache(sop_templates)

    return exported


def _focus_work_ready(state_dir: Path, db: "WorkerDB") -> str | None:
    """Focus work the user is waiting for, which the generation phase
    never batches or holds back for memory: ``"answers"`` when the
    post-recording questions were answered or skipped, ``"session"`` when
    a stopped recording has its events in the database, else None."""
    from agenthandover_worker.focus_questioner import read_focus_questions

    questions = read_focus_questions(state_dir)
    if questions is not None and questions.get("status") in ("answered", "skipped"):
        return "answers"
    try:
        signal = json.loads((state_dir / "focus-session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(signal, dict) or signal.get("status") != "stopped":
        return None
    session_id = signal.get("session_id")
    if session_id and db.get_focus_session_events(session_id):
        return "session"
    return None


def _check_focus_answers(
    state_dir: Path,
    *,
    db: "WorkerDB | None" = None,
    focus_questioner: "FocusQuestioner | None",
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
    knowledge_base=None,
    vector_kb=None,
) -> int | None:
    """Check for pending focus Q&A answers and complete export if ready.

    Returns:
        Number of SOPs exported if answers were processed.
        None if no pending Q&A was found (caller should proceed normally).
    """
    from agenthandover_worker.focus_questioner import (
        read_focus_questions,
        read_focus_pending,
        parse_qa_result_from_file,
        clear_focus_qa_files,
    )

    questions_data = read_focus_questions(state_dir)
    if questions_data is None:
        return None

    status = questions_data.get("status", "")
    if status not in ("answered", "skipped"):
        # Still pending — user hasn't answered yet
        return None

    # Load the saved pending SOP state
    pending = read_focus_pending(state_dir)
    if pending is None:
        logger.warning(
            "Focus Q&A answers found but no pending SOP state — clearing stale files",
        )
        clear_focus_qa_files(state_dir)
        return None

    slug = pending.get("slug", "")
    session_id = pending.get("session_id", "")
    sop_template = pending.get("sop_template", {})
    procedure_dict = pending.get("procedure", {})
    title = sop_template.get("title", procedure_dict.get("title", "Untitled"))

    # Parse and merge answers.  A skipped Q&A still merges: every
    # question is recorded as unanswered rather than dropped.
    qa_result = parse_qa_result_from_file(questions_data)
    if qa_result is not None and focus_questioner is not None and qa_result.questions:
        try:
            procedure_dict = focus_questioner.merge_answers(procedure_dict, qa_result)
            logger.info(
                "Focus v2 '%s': merged %d answer(s) into procedure (%d question(s))",
                title, sum(1 for a in qa_result.answers.values() if a.strip()),
                len(qa_result.questions),
            )
        except Exception:
            logger.debug(
                "Failed to merge focus Q&A answers for '%s'",
                title, exc_info=True,
            )

    sop_templates = [procedure_dict]

    # The review-queue row was saved before the answers; update it in
    # place so the app and a later approval read the merged procedure.
    if db is not None and not db.update_sop_json(slug, procedure_dict):
        logger.warning("No review record for '%s' to merge Q&A answers into", slug)

    # Save enriched procedure to KB (answers merged), keeping any agent
    # runs reported while the questions were open.
    if knowledge_base is not None:
        try:
            from agenthandover_worker.procedure_writer import carry_run_history
            existing_proc = knowledge_base.get_procedure(slug)
            if existing_proc is not None:
                carry_run_history(procedure_dict, existing_proc)
            knowledge_base.save_procedure(procedure_dict)
            logger.info("Saved Q&A-enriched procedure '%s' to KB", slug)
            _embed_procedure(vector_kb, procedure_dict)
        except Exception:
            logger.debug("Failed to save Q&A-enriched procedure", exc_info=True)

    # Export
    exported = 0
    if sop_auto_approve:
        exported = _focus_v2_export(
            sop_templates=sop_templates,
            title=title,
            inference_time=0.0,  # VLM time was in the original cycle
            openclaw_writer=openclaw_writer,
            skill_md_writer=skill_md_writer,
            claude_skill_writer=claude_skill_writer,
            index_generator=index_generator,
            procedure_writer=procedure_writer,
            kb_export_adapter=kb_export_adapter,
        )
        record = db.get_generated_sop_by_slug(slug) if db is not None else None
        if record is not None:
            db.update_sop_status(record["sop_id"], "approved")

    # Clean up Q&A files
    clear_focus_qa_files(state_dir)

    # Refresh the Claude Code Skills with the new/updated procedure
    _refresh_claude_commands()

    logger.info(
        "Focus v2 '%s': Q&A complete, exported %d SOP(s) (status=%s)",
        title, exported, status,
    )
    return exported


def _process_passive_discovery(
    db: "WorkerDB",
    *,
    knowledge_base: "KnowledgeBase",
    segmenter: "TaskSegmenter",
    sop_generator: "SOPGenerator",
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
    continuity_tracker: "ContinuityTracker | None" = None,
    variant_detector: "VariantDetector | None" = None,
    evidence_normalizer: "EvidenceNormalizer | None" = None,
    now: datetime | None = None,
) -> int:
    """Run passive discovery: segment annotations → generate SOPs.

    Called periodically (every 2 hours, on idle, or by CLI trigger).
    Returns the number of SOPs generated.  *now* ends the annotation
    window (default: the wall clock); a replay passes its virtual clock.

    Pipeline:
      1. Load recent annotated events from DB
      2. Run task segmenter (embeddings + clustering + noise filter)
      3. Persist segments (14-day TTL) and record every finished work
         session in ``workflow_observations`` (``workflow_mining``)
      4. For the workflows the review queue admits (observed at least
         three times, at most three per week, highest value first),
         generate passive SOPs
      5. Export SOPs
    """
    now = now or datetime.now(timezone.utc)
    # Step 1: Load annotated events
    events = db.get_annotated_events_in_window(
        hours=segmenter.config.default_window_hours,
        now=now,
    )
    if not events:
        logger.debug("Passive discovery: no annotated events in window")
        return 0

    logger.info(
        "Passive discovery: processing %d annotated events", len(events),
    )

    # Step 2: Run segmentation.  No segment spans a system sleep.
    sleep_boundaries = db.get_sleep_boundaries(since=events[0].get("timestamp"))
    seg_result = segmenter.segment(events, sleep_boundaries=sleep_boundaries)

    if not seg_result.segments:
        logger.info(
            "Passive discovery: no task segments found (%d noise frames)",
            seg_result.noise_frames_dropped,
        )
        return 0

    logger.info(
        "Passive discovery: %d segments in %d clusters "
        "(%.1fs embedding, %d noise dropped)",
        len(seg_result.segments),
        len(seg_result.clusters),
        seg_result.embedding_time_seconds,
        seg_result.noise_frames_dropped,
    )

    # Step 2b: Classify and merge interruptions between segments.
    # Collect all frames from every segment so classify_interruptions
    # can check what apps were active during gaps.
    all_frames = [f for seg in seg_result.segments for f in seg.frames]
    if all_frames:
        try:
            seg_result.segments = segmenter.classify_interruptions(
                seg_result.segments, all_frames, sleep_boundaries=sleep_boundaries,
            )
            # Rebuild cluster dict after merging
            seg_result.clusters = {}
            for seg in seg_result.segments:
                seg_result.clusters.setdefault(seg.cluster_id, []).append(seg)
            logger.debug(
                "Interruption classification: %d segments after merge",
                len(seg_result.segments),
            )
        except Exception:
            logger.debug(
                "Interruption classification failed, continuing with "
                "original segments",
                exc_info=True,
            )

    # Build continuity graph
    if continuity_tracker is not None:
        try:
            spans = continuity_tracker.build_graph(seg_result.segments)
            seg_result.spans = spans
            continuity_tracker.save_spans(spans)
        except Exception:
            logger.debug("Continuity tracking failed", exc_info=True)

    # Step 3: Persist segments, then record each finished work session as
    # a workflow observation (the grouping unit; it outlives raw frames).
    labels: dict[str, str] = {}
    for seg in seg_result.segments:
        event_ids = [f.event_id for f in seg.frames]
        labels.update(dict.fromkeys(event_ids, seg.task_label))
        db.save_task_segment(
            segment_id=seg.segment_id,
            cluster_id=seg.cluster_id,
            task_label=seg.task_label,
            event_ids=event_ids,
            apps=seg.apps_involved,
            start_time=seg.start_time,
            end_time=seg.end_time,
        )
    recorded = workflow_mining.record_observations(
        db, events, now=now, labels=labels, sleep_boundaries=sleep_boundaries,
    )
    logger.info("Passive discovery: %d work session(s) recorded", recorded)

    # Step 4: the workflows the review queue admits now.
    import json as _json

    candidates = workflow_mining.review_candidates(db, now=now)
    if not candidates:
        logger.info("Passive discovery: no workflow ready for review")
        return 0

    total_exported = 0

    for candidate in candidates:
        cluster_id = candidate.source_id
        task_label = candidate.task_label
        candidate_events = {
            eid for o in candidate.observations for eid in _json.loads(o["event_ids_json"])
        }

        # Reconstruct demonstrations (timelines) from DB event IDs; the
        # frames of observations older than the raw retention are gone.
        demonstrations: list[list[dict]] = []
        for obs in candidate.observations:
            event_ids = _json.loads(obs["event_ids_json"])
            events_for_seg = db.get_events_by_ids(event_ids)
            timeline: list[dict] = []
            for ev in events_for_seg:
                ann_raw = ev.get("scene_annotation_json")
                if not ann_raw:
                    continue
                try:
                    annotation = (
                        _json.loads(ann_raw) if isinstance(ann_raw, str)
                        else ann_raw
                    )
                except (_json.JSONDecodeError, TypeError):
                    continue
                diff = None
                diff_raw = ev.get("frame_diff_json")
                if diff_raw:
                    try:
                        diff = (
                            _json.loads(diff_raw) if isinstance(diff_raw, str)
                            else diff_raw
                        )
                    except (_json.JSONDecodeError, TypeError):
                        pass
                frame = {
                    "annotation": annotation,
                    "diff": diff,
                    "timestamp": ev.get("timestamp", ""),
                }
                # The focused field read through Accessibility (a native
                # app's cell or field, its label and, if captured, value).
                ev_field = focused_field(ev)
                if ev_field:
                    frame["focused_field"] = ev_field
                    frame["event_id"] = ev.get("id", "")
                timeline.append(frame)
            if timeline:
                demonstrations.append(timeline)

        if len(demonstrations) < segmenter.config.min_demonstrations:
            # Recorded as a failure for these observations, so the workflow
            # is tried again only when a new observation (or a retry) comes.
            db.record_failed_generation(
                source="passive",
                source_id=str(cluster_id),
                error=f"only {len(demonstrations)} observation(s) still have frames",
                title=task_label,
                context={"obs_ids": candidate.obs_ids},
            )
            continue

        # --- Semantic alignment + variant detection + parameter extraction ---
        _variant_results: dict = {}
        if variant_detector is not None and len(demonstrations) >= 2:
            try:
                from dataclasses import asdict as _asdict

                # Build step lists from demonstration annotations
                _demo_steps: list[list[dict]] = []
                for demo_tl in demonstrations:
                    steps = []
                    for frame in demo_tl:
                        ann = frame.get("annotation", {})
                        if not isinstance(ann, dict):
                            continue
                        ann = as_compact(ann)
                        # The target: the list entry the user picked, else
                        # the control they were on.
                        chosen = next(
                            (str(item.get("label") or "")
                             for item in ann.get("visible_items") or []
                             if isinstance(item, dict) and item.get("chosen")),
                            "",
                        )
                        steps.append({
                            "action": ann.get("task_context", {}).get("what_doing", ""),
                            "app": ann.get("app", ""),
                            "location": ann.get("location", ""),
                            "target": chosen or str(
                                (ann.get("ui_state") or {}).get("active_element") or ""
                            ),
                            "input": "",
                        })
                    if steps:
                        _demo_steps.append(steps)

                if len(_demo_steps) >= 2:
                    # Pairwise alignment against first demo
                    _alignments = []
                    for i in range(1, len(_demo_steps)):
                        aligned = variant_detector.semantic_align(
                            _demo_steps[0], _demo_steps[i],
                        )
                        _alignments.append(aligned)

                    _variants = variant_detector.detect_variants(
                        task_label or str(cluster_id), _demo_steps,
                    )
                    _parameters = variant_detector.extract_parameters(
                        _demo_steps, _alignments,
                    )
                    _normalized = variant_detector.normalize_workflow(
                        _demo_steps, _variants,
                    )

                    _variant_results = {
                        "alignments": _alignments,
                        "variants": _variants,
                        "parameters": _parameters,
                        "normalized": _normalized,
                        "demo_steps": _demo_steps,
                    }
                    logger.info(
                        "Variant detection for '%s': %d variants, %d params",
                        task_label[:60],
                        len(_variants),
                        len(_parameters),
                    )
            except Exception:
                logger.debug(
                    "Variant detection failed for '%s'",
                    task_label[:60], exc_info=True,
                )

        logger.info(
            "Passive discovery: generating SOP for '%s' (%d demos)",
            task_label[:60], len(demonstrations),
        )

        # Load evidence from prior procedure if it exists.  The generated
        # SOP is titled with task_label and slugged by _generate_slug, so
        # the same call names the procedure an earlier run wrote.
        _prior_evidence: dict | None = None
        try:
            _candidate_slug = _generate_slug(task_label) if task_label else ""
            if _candidate_slug:
                _prior_proc = knowledge_base.get_procedure(_candidate_slug)
                if _prior_proc is not None:
                    _prior_evidence = (
                        _prior_proc.get("evidence") or {}
                    ).get("extracted_evidence")
        except (OSError, KeyError, ValueError):
            logger.debug("Failed to load prior evidence for '%s'", task_label[:60], exc_info=True)

        # Use enriched generation if variant data available
        def _generate_passive():
            if _variant_results and hasattr(sop_generator, "generate_from_passive_enriched"):
                return sop_generator.generate_from_passive_enriched(
                    demonstrations,
                    task_label=task_label,
                    canonical_steps=_variant_results["normalized"].get("canonical_steps", []),
                    parameters=_variant_results["parameters"],
                    branches=_variant_results["normalized"].get("branches", []),
                    evidence_context=_prior_evidence,
                )
            return sop_generator.generate_from_passive(
                demonstrations, task_label=task_label,
            )

        # Lint included: an SOP that fails validation is regenerated once,
        # then recorded as a failure like any other generation failure.
        result, lint_error = _generate_linted(
            _generate_passive, "passive", task_label,
        )

        if result.deferred:
            # Ollama unreachable or the model failed to load: no failure
            # recorded, the segments stay pending for the next pass.
            logger.warning(
                "Passive discovery deferred at '%s': %s",
                task_label[:60], result.error,
            )
            break

        if lint_error or not result.success or not result.sop:
            # Record failure for retry tracking
            try:
                db.record_failed_generation(
                    source="passive",
                    source_id=str(cluster_id),
                    error=lint_error or result.error or "Unknown error",
                    title=task_label,
                    context={"obs_ids": candidate.obs_ids},
                )
            except Exception:
                logger.debug(
                    "Failed to record generation failure for '%s'",
                    task_label[:60], exc_info=True,
                )
            logger.warning(
                "Passive SOP generation failed for '%s': %s",
                task_label[:60], lint_error or result.error,
            )
            continue

        # The draft carries its observations: count, recurrence, value.
        result.sop.update(candidate.template_evidence())
        sop_templates = [result.sop]
        slug = result.sop.get("slug", "")

        # Attach evidence context to template for downstream consumers
        if _prior_evidence:
            for tpl in sop_templates:
                tpl["_extracted_evidence"] = _prior_evidence

        # Save generated SOP to DB for review tracking
        try:
            db.save_generated_sop(
                slug=slug,
                title=task_label,
                source="passive",
                sop_template=result.sop,
                confidence=result.sop.get("confidence_avg", result.sop.get("confidence", 0.0)),
                source_id=str(cluster_id),
                auto_approve=sop_auto_approve,
            )
        except Exception:
            logger.warning(
                "Failed to save generated SOP to DB for '%s'",
                task_label[:60], exc_info=True,
            )

        # Only export if auto_approve is enabled
        if sop_auto_approve:
            # Deduplicate against known SOPs (with evidence-based merge)
            from agenthandover_worker.sop_dedup import deduplicate_templates
            sop_templates = deduplicate_templates(
                sop_templates, _status_dir(),
                evidence_normalizer=evidence_normalizer,
            )

            # --- Enrich SOP templates with variant/evidence data ---
            if _variant_results:
                from dataclasses import asdict as _asdict
                for tpl in sop_templates:
                    try:
                        _vr_variants = _variant_results.get("variants", [])
                        _vr_params = _variant_results.get("parameters", [])
                        _vr_normalized = _variant_results.get("normalized", {})

                        if _vr_variants:
                            tpl["variants"] = [_asdict(v) for v in _vr_variants]
                        if _vr_params:
                            tpl["parameters_extracted"] = [
                                _asdict(p) for p in _vr_params
                            ]
                        if _vr_normalized and _vr_normalized.get("branches"):
                            tpl["branches"] = _vr_normalized["branches"]
                        if _vr_normalized and _vr_normalized.get("canonical_steps"):
                            tpl["_canonical_steps"] = _vr_normalized["canonical_steps"]
                    except Exception:
                        logger.debug(
                            "Failed to enrich template with variant data",
                            exc_info=True,
                        )

            # --- Enrich with continuity span metadata ---
            if seg_result.spans:
                for tpl in sop_templates:
                    try:
                        # Match spans holding a segment of this workflow.
                        # ContinuitySpan.segments is a list of objects with
                        # segment_id attribute, or dicts with "segment_id".
                        _matching_spans = []
                        _seg_id_set = {
                            s.segment_id for s in seg_result.segments
                            if any(f.event_id in candidate_events for f in s.frames)
                        }
                        for _span in seg_result.spans:
                            span_segs = getattr(_span, "segments", [])
                            for _sseg in span_segs:
                                _sid = getattr(_sseg, "segment_id", None)
                                if _sid is None and isinstance(_sseg, dict):
                                    _sid = _sseg.get("segment_id")
                                if _sid in _seg_id_set:
                                    _matching_spans.append(_span)
                                    break

                        if _matching_spans:
                            best_span = max(
                                _matching_spans,
                                key=lambda s: getattr(s, "total_duration_seconds", 0),
                            )
                            # Map span metadata onto procedure fields
                            tpl.setdefault("_span_metadata", {
                                "interruption_count": getattr(
                                    best_span, "interruption_count", 0
                                ),
                                "total_duration_seconds": getattr(
                                    best_span, "total_duration_seconds", 0
                                ),
                                "matched_procedure_candidates": getattr(
                                    best_span, "matched_procedure_candidates", []
                                ),
                            })
                    except Exception:
                        logger.debug(
                            "Failed to enrich template with span data",
                            exc_info=True,
                        )

            # Write v3 procedures to the knowledge base first so
            # all adapters (including OpenClaw) can use the richer v3 format.
            if procedure_writer is not None:
                for tpl in sop_templates:
                    try:
                        procedure_writer.write_procedure(
                            tpl,
                            source="sop_pipeline",
                            source_id=tpl.get("slug", "unknown"),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to write procedure for %s",
                            tpl.get("slug", "?"), exc_info=True,
                        )

            # Export via OpenClaw (uses v3 procedure from KB when available)
            try:
                paths = _export_via_adapter(
                    openclaw_writer, sop_templates, procedure_writer,
                )
                total_exported += len(paths)
                logger.info(
                    "Passive SOP '%s': exported %d SOP(s) (%.1fs VLM)",
                    task_label[:60], len(paths), result.inference_time_seconds,
                )
            except Exception:
                logger.exception("Passive SOP export failed for '%s'", task_label[:60])

            # Export via SKILL.md (uses v3 procedure when available)
            if skill_md_writer is not None:
                try:
                    _export_via_adapter(
                        skill_md_writer, sop_templates, procedure_writer,
                    )
                except Exception:
                    logger.exception("Passive SKILL.md export failed")

            # Export via Claude Code skills (uses v3 procedure when available)
            if claude_skill_writer is not None:
                try:
                    _export_via_adapter(
                        claude_skill_writer, sop_templates, procedure_writer,
                    )
                except Exception:
                    logger.exception("Passive Claude skill export failed")

            if kb_export_adapter is not None:
                try:
                    kb_export_adapter.write_all_sops(sop_templates)
                except Exception:
                    logger.debug("KB export adapter failed", exc_info=True)

            _save_sop_cache(sop_templates)
        else:
            logger.info(
                "Passive SOP '%s': saved as draft (auto_approve=False)",
                task_label[:60],
            )

        # The workflow is promoted: its observations belong to the Skill.
        db.set_observations_promoted(candidate.obs_ids, slug, promoted_at=now)

    return total_exported


def _record_retry_lint_failure(
    db: "WorkerDB",
    source: str,
    source_id: str,
    title: str,
    sop_template: dict,
    label: str,
    context: dict | None = None,
) -> None:
    """Record a retried SOP that still fails lint as a new failure, so it
    stays retryable (the retried record is marked retried afterwards)."""
    try:
        db.record_failed_generation(
            source=source,
            source_id=source_id,
            error=_lint_error_text(sop_template, label),
            title=title,
            context=context,
        )
    except Exception:
        logger.debug("Failed to record retry lint failure", exc_info=True)


def _process_retry_triggers(
    trigger: dict,
    db: "WorkerDB",
    *,
    focus_processor: "FocusProcessor | None" = None,
    sop_generator: "SOPGenerator | None" = None,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    screenshots_dir: str | Path = "",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
) -> None:
    """Handle a ``retry`` trigger.

    Re-runs SOP generation for a previously failed attempt, then marks
    the failure as retried in the DB.
    """
    failure_id = trigger.get("failure_id", "")
    if not failure_id:
        logger.warning("retry trigger missing failure_id")
        return

    logger.info("Processing retry trigger for failure_id=%s", failure_id)

    try:
        failure = db.get_failed_generation(failure_id)
    except Exception:
        logger.warning(
            "Failed to read failure record %s from DB",
            failure_id, exc_info=True,
        )
        return

    if failure is None:
        logger.warning("Retry trigger: failure_id=%s not found in DB", failure_id)
        return

    source = failure.get("source", "")
    source_id = failure.get("source_id", "")
    title = failure.get("title", "")

    try:
        if source == "focus" and focus_processor is not None:
            # Re-run focus session generation
            focus_events = db.get_focus_session_events(source_id)
            if focus_events:
                result = focus_processor.process_session(
                    db, source_id, title, focus_events,
                    screenshots_dir=screenshots_dir,
                )
                if result.success and result.sop:
                    slug = result.sop.get("slug", "")

                    # Lint before save+export
                    if not _lint_and_log(result.sop, "retry_focus"):
                        _record_retry_lint_failure(
                            db, "focus", source_id, title, result.sop,
                            "retry_focus",
                        )
                    else:
                        try:
                            db.save_generated_sop(
                                slug=slug,
                                title=title,
                                source="focus",
                                sop_template=result.sop,
                                confidence=result.sop.get("confidence_avg", result.sop.get("confidence", 0.0)),
                                source_id=source_id,
                                auto_approve=sop_auto_approve,
                            )
                        except Exception:
                            logger.warning(
                                "Retry: failed to save SOP to DB", exc_info=True,
                            )

                        if sop_auto_approve:
                            _export_sop_templates(
                                [result.sop],
                                openclaw_writer=openclaw_writer,
                                skill_md_writer=skill_md_writer,
                                claude_skill_writer=claude_skill_writer,
                                index_generator=index_generator,
                                procedure_writer=procedure_writer,
                                kb_export_adapter=kb_export_adapter,
                            )
                        logger.info(
                            "Retry succeeded for focus session '%s'", title,
                        )
                else:
                    logger.warning(
                        "Retry failed again for focus session '%s': %s",
                        title, result.error,
                    )
            else:
                logger.warning(
                    "Retry: no events found for focus session %s", source_id,
                )

        elif source == "passive" and sop_generator is not None:
            # Re-run passive generation for the workflow's observations
            context = failure.get("context_json")
            obs_ids = (context if isinstance(context, dict) else {}).get("obs_ids") or []
            pending_segs = db.get_workflow_observations(obs_ids=obs_ids) if obs_ids else []
            if pending_segs:
                import json as _json
                demonstrations: list[list[dict]] = []
                for seg_row in pending_segs:
                    event_ids = _json.loads(
                        seg_row.get("event_ids_json", "[]")
                    )
                    events_for_seg = db.get_events_by_ids(event_ids)
                    timeline: list[dict] = []
                    for ev in events_for_seg:
                        ann_raw = ev.get("scene_annotation_json")
                        if not ann_raw:
                            continue
                        try:
                            annotation = (
                                _json.loads(ann_raw)
                                if isinstance(ann_raw, str) else ann_raw
                            )
                        except (_json.JSONDecodeError, TypeError):
                            continue
                        diff = None
                        diff_raw = ev.get("frame_diff_json")
                        if diff_raw:
                            try:
                                diff = (
                                    _json.loads(diff_raw)
                                    if isinstance(diff_raw, str) else diff_raw
                                )
                            except (_json.JSONDecodeError, TypeError):
                                pass
                        timeline.append({
                            "annotation": annotation,
                            "diff": diff,
                            "timestamp": ev.get("timestamp", ""),
                        })
                    if timeline:
                        demonstrations.append(timeline)

                if demonstrations:
                    result = sop_generator.generate_from_passive(
                        demonstrations, task_label=title,
                    )
                    if result.success and result.sop:
                        slug = result.sop.get("slug", "")
                        result.sop.update(
                            episode_count=len(pending_segs),
                            observations=workflow_mining.observation_pointers(pending_segs),
                        )

                        # Lint before save+export
                        if not _lint_and_log(result.sop, "retry_passive"):
                            _record_retry_lint_failure(
                                db, "passive", source_id, title, result.sop,
                                "retry_passive", context={"obs_ids": obs_ids},
                            )
                        else:
                            try:
                                db.save_generated_sop(
                                    slug=slug,
                                    title=title,
                                    source="passive",
                                    sop_template=result.sop,
                                    confidence=result.sop.get("confidence_avg", result.sop.get("confidence", 0.0)),
                                    source_id=source_id,
                                    auto_approve=sop_auto_approve,
                                )
                            except Exception:
                                logger.warning(
                                    "Retry: failed to save SOP to DB",
                                    exc_info=True,
                                )

                            if sop_auto_approve:
                                _export_sop_templates(
                                    [result.sop],
                                    openclaw_writer=openclaw_writer,
                                    skill_md_writer=skill_md_writer,
                                    claude_skill_writer=claude_skill_writer,
                                    index_generator=index_generator,
                                    procedure_writer=procedure_writer,
                                    kb_export_adapter=kb_export_adapter,
                                )
                            db.set_observations_promoted(
                                obs_ids, slug, promoted_at=datetime.now(timezone.utc),
                            )
                            logger.info(
                                "Retry succeeded for passive workflow '%s'", title,
                            )
                    else:
                        logger.warning(
                            "Retry failed again for passive workflow '%s': %s",
                            title, result.error,
                        )
            else:
                logger.warning(
                    "Retry: no observations found for workflow %s", source_id,
                )
        else:
            logger.warning(
                "Retry trigger: unsupported source '%s' or processor unavailable",
                source,
            )
    except Exception:
        logger.warning("Retry trigger processing failed", exc_info=True)

    # Mark failure as retried regardless of outcome
    try:
        db.mark_failure_retried(failure_id)
    except Exception:
        logger.debug("Failed to mark failure as retried", exc_info=True)


def _approve_sop(
    db: "WorkerDB",
    sop_id_or_slug: str,
    *,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
    vector_kb=None,
) -> bool:
    """Lint, approve and export one generated SOP; return True on success.

    The stored record is linted before its status changes, so a record
    that fails lint stays a draft instead of becoming an approved SOP
    that was never exported.
    """
    # Resolve: try as UUID first, then as slug
    record = db.get_generated_sop(sop_id_or_slug)
    if record is None:
        record = db.get_generated_sop_by_slug(sop_id_or_slug)
    if record is None:
        logger.warning("Cannot approve SOP %s: not found", sop_id_or_slug)
        return False
    sop_id = record["sop_id"]
    sop_template = record.get("sop_json")
    if not isinstance(sop_template, dict) or not sop_template:
        logger.warning("Cannot approve SOP %s: no template stored", sop_id)
        return False
    if not _lint_and_log(sop_template, "approval"):
        logger.error("SOP %s failed lint validation, left as draft", sop_id)
        return False
    if not db.update_sop_status(sop_id, "approved"):
        logger.warning("Failed to approve SOP %s", sop_id)
        return False
    _export_sop_templates(
        [sop_template],
        openclaw_writer=openclaw_writer,
        skill_md_writer=skill_md_writer,
        claude_skill_writer=claude_skill_writer,
        index_generator=index_generator,
        procedure_writer=procedure_writer,
        kb_export_adapter=kb_export_adapter,
    )
    logger.info("Approved and exported SOP %s", sop_id)
    _embed_procedure(vector_kb, sop_template)
    _refresh_claude_commands()
    return True


def _process_approval_triggers(
    trigger: dict,
    db: "WorkerDB",
    **export_kwargs,
) -> None:
    """Handle an ``approve`` trigger: approve or reject a draft SOP.

    ``export_kwargs`` are the writers ``_approve_sop`` exports through.
    """
    sop_id_or_slug = trigger.get("sop_id", "")
    action = trigger.get("action", "")

    if not sop_id_or_slug or action not in ("approve", "reject"):
        logger.warning(
            "approve trigger invalid: sop_id=%s action=%s",
            sop_id_or_slug, action,
        )
        return

    logger.info(
        "Processing approval trigger: sop_id=%s action=%s",
        sop_id_or_slug, action,
    )
    if action == "approve":
        _approve_sop(db, sop_id_or_slug, **export_kwargs)
        return

    record = db.get_generated_sop(sop_id_or_slug)
    if record is None:
        record = db.get_generated_sop_by_slug(sop_id_or_slug)
    sop_id = record["sop_id"] if record else sop_id_or_slug
    if db.update_sop_status(sop_id, "rejected"):
        logger.info("Rejected SOP %s", sop_id)
    else:
        logger.warning("Failed to reject SOP %s (not found?)", sop_id)


def _process_failed_query_trigger(trigger: dict, db: "WorkerDB") -> None:
    """Handle a ``failed-query`` trigger.

    Reads failed generations from the DB and writes the result to
    failed-query-result.json for the CLI to pick up.
    """
    state_dir = _status_dir()
    logger.info("Processing failed generations query trigger")

    try:
        failures = db.get_failed_generations(include_retried=False)
        result = {
            "failures": [
                {
                    "id": f.get("failure_id", ""),
                    "sop_slug": f.get("title", "") or f.get("source_id", ""),
                    "source": f.get("source", ""),
                    "error": f.get("error", ""),
                    "created_at": f.get("created_at", ""),
                }
                for f in failures
            ]
        }
        result_path = state_dir / "failed-query-result.json"
        _atomic_write_result(result_path, result)
        logger.info("Wrote %d failures to result file", len(failures))
    except Exception:
        logger.warning("Failed to process failed query trigger", exc_info=True)


def _process_search_trigger(
    trigger: dict,
    activity_searcher: "ActivitySearcher | None",
) -> None:
    """Handle a ``search-query`` trigger (the CLI's fallback path).

    Runs the query through the ActivitySearcher and writes results
    to search-query-result.json for the CLI to pick up.
    """
    state_dir = _status_dir()
    logger.info("Processing search query trigger")

    # Remove any stale result file so the CLI cannot read old data
    result_path = state_dir / "search-query-result.json"
    result_path.unlink(missing_ok=True)

    try:
        query = trigger.get("query", "")
        limit = trigger.get("limit", 20)
        date = trigger.get("date")
        app = trigger.get("app")

        if activity_searcher is None:
            result = {"error": "Activity search not available (v2 schema required)"}
        elif not query:
            result = {"error": "Empty search query", "results": []}
        else:
            hits = activity_searcher.search(
                query, limit=limit, date=date, app=app,
            )
            result = {
                "results": [
                    {
                        "timestamp": h.timestamp,
                        "app": h.app,
                        "location": h.location,
                        "what_doing": h.what_doing,
                        "relevance_score": h.relevance_score,
                        "event_id": h.event_id,
                    }
                    for h in hits
                ]
            }

        result_path = state_dir / "search-query-result.json"
        _atomic_write_result(result_path, result)
        logger.info("Search: %d results written", len(result.get("results", [])))
    except Exception:
        logger.warning("Failed to process search trigger", exc_info=True)
        # Write error result so CLI doesn't hang
        try:
            result_path = state_dir / "search-query-result.json"
            _atomic_write_result(result_path, {"error": "Internal search error"})
        except Exception:
            pass


def _process_recall_trigger(
    trigger: dict,
    activity_searcher: "ActivitySearcher | None",
) -> None:
    """Handle a ``recall-query`` trigger (the CLI's fallback path).

    Runs session recall and writes results to recall-query-result.json.
    """
    state_dir = _status_dir()
    logger.info("Processing recall query trigger")

    # Remove any stale result file so the CLI cannot read old data
    result_path = state_dir / "recall-query-result.json"
    result_path.unlink(missing_ok=True)

    try:
        if activity_searcher is None:
            result = {"error": "Activity search not available (v2 schema required)"}
        else:
            from agenthandover_worker.activity_search import recall_payload
            result = recall_payload(activity_searcher, trigger)

        result_path = state_dir / "recall-query-result.json"
        _atomic_write_result(result_path, result)
        logger.info("Recall: %d entries written", len(result.get("entries", [])))
    except Exception:
        logger.warning("Failed to process recall trigger", exc_info=True)
        try:
            result_path = state_dir / "recall-query-result.json"
            _atomic_write_result(result_path, {"error": "Internal recall error"})
        except Exception:
            pass


# ------------------------------------------------------------------
# Trigger handlers: trust & staleness (MicroReviewView triggers)
# ------------------------------------------------------------------


def _process_trust_accept_trigger(
    trigger: dict,
    trust_advisor: "TrustAdvisor",
    constraint_manager: "ConstraintManager",
) -> None:
    """Handle a ``trust-accept`` trigger.

    The MicroReviewView writes this file when the user accepts a trust
    promotion suggestion.  We delegate to ``trust_advisor.accept_suggestion``
    which promotes the procedure's trust level in the knowledge base.
    """
    procedure_slug = trigger.get("procedure_slug", "")
    if not procedure_slug:
        logger.warning("trust-accept trigger missing procedure_slug")
        return

    logger.info("Processing trust accept trigger: slug=%s", procedure_slug)

    try:
        accepted = trust_advisor.accept_suggestion(procedure_slug)
        if accepted:
            logger.info(
                "Trust promotion accepted for '%s'", procedure_slug
            )
        else:
            logger.warning(
                "No pending trust suggestion found for '%s'", procedure_slug
            )
    except Exception:
        logger.warning(
            "Trust accept trigger processing failed for '%s'",
            procedure_slug,
            exc_info=True,
        )


def _process_trust_dismiss_trigger(
    trigger: dict,
    trust_advisor: "TrustAdvisor",
) -> None:
    """Handle a ``trust-dismiss`` trigger.

    The MicroReviewView writes this file when the user dismisses a trust
    promotion suggestion.  No state change is needed beyond marking the
    suggestion as dismissed — this simply acknowledges the user's choice.
    """
    procedure_slug = trigger.get("procedure_slug", "")
    if not procedure_slug:
        logger.warning("trust-dismiss trigger missing procedure_slug")
        return

    logger.info("Processing trust dismiss trigger: slug=%s", procedure_slug)

    try:
        dismissed = trust_advisor.dismiss_suggestion(procedure_slug)
        if dismissed:
            logger.info(
                "Trust suggestion dismissed for '%s'", procedure_slug
            )
        else:
            logger.info(
                "No pending trust suggestion to dismiss for '%s'",
                procedure_slug,
            )
    except Exception:
        logger.warning(
            "Trust dismiss trigger processing failed for '%s'",
            procedure_slug,
            exc_info=True,
        )


def _process_staleness_reviewed_trigger(
    trigger: dict,
    staleness_detector: "StalenessDetector",
    knowledge_base: "KnowledgeBase",
) -> None:
    """Handle a ``staleness-reviewed`` trigger.

    The MicroReviewView writes this file when the user acts on a staleness
    alert.  Two actions are supported:

    * ``"reviewed"`` — mark the procedure as freshly confirmed by updating
      ``staleness.last_confirmed`` to now.
    * ``"archive"`` — delete the procedure from the knowledge base.
    """
    procedure_slug = trigger.get("procedure_slug", "")
    action = trigger.get("action", "")

    if not procedure_slug or action not in ("reviewed", "archive"):
        logger.warning(
            "staleness-reviewed trigger invalid: slug=%s action=%s",
            procedure_slug,
            action,
        )
        return

    logger.info(
        "Processing staleness reviewed trigger: slug=%s action=%s",
        procedure_slug,
        action,
    )

    try:
        if action == "reviewed":
            proc = knowledge_base.get_procedure(procedure_slug)
            if proc is not None:
                proc.setdefault("staleness", {})[
                    "last_confirmed"
                ] = datetime.now(timezone.utc).isoformat()
                knowledge_base.save_procedure(proc)
                logger.info(
                    "Staleness last_confirmed updated for '%s'",
                    procedure_slug,
                )
            else:
                logger.warning(
                    "Procedure '%s' not found for staleness review",
                    procedure_slug,
                )
        elif action == "archive":
            deleted = knowledge_base.delete_procedure(procedure_slug)
            if deleted:
                logger.info(
                    "Procedure '%s' archived (deleted) per user request",
                    procedure_slug,
                )
            else:
                logger.warning(
                    "Could not delete procedure '%s' (not found?)",
                    procedure_slug,
                )
    except Exception:
        logger.warning(
            "Staleness reviewed trigger processing failed for '%s'",
            procedure_slug,
            exc_info=True,
        )


def _process_lifecycle_promote_trigger(
    trigger: dict,
    *,
    lifecycle_manager: "LifecycleManager",
    db: "WorkerDB | None" = None,
    approve: Callable[[str], bool] | None = None,
    knowledge_base: "KnowledgeBase | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
) -> None:
    """Handle a ``lifecycle-promote`` trigger from the app or the CLI.

    ``slug`` (app) or ``procedure_slug`` (CLI) names the procedure.  The
    target is reached by walking the promotion chain one valid transition
    at a time.  Promoting to ``agent_ready`` first lints a pending draft; a
    draft that fails lint is not promoted.  Once the procedure is
    ``agent_ready`` its review record is approved and exported through
    *approve* (every export passes the hand-off gate, which a draft never
    does, so the export has to follow the promotion).
    Claude Code commands are refreshed whenever the procedure enters or
    leaves ``agent_ready``; on entering it, the procedure is also written
    as a Claude Code Skill when *claude_skill_writer* is configured (the
    refresh removes Skills of procedures that are not agent_ready).
    """
    from agenthandover_worker.lifecycle_manager import ProcedureLifecycle

    slug = trigger.get("slug") or trigger.get("procedure_slug", "")
    to_state = trigger.get("to_state", "")
    actor = trigger.get("actor", "human")
    reason = trigger.get("reason", "")

    if not slug or not to_state:
        logger.warning("Lifecycle promote trigger missing slug or to_state")
        return
    try:
        target = ProcedureLifecycle(to_state)
    except ValueError:
        logger.warning("Invalid lifecycle state: %s", to_state)
        return

    logger.info("Processing lifecycle promote trigger: %s → %s", slug, to_state)

    record = None
    if target is ProcedureLifecycle.AGENT_READY and db is not None and approve is not None:
        record = db.get_generated_sop_by_slug(slug)
        template = record.get("sop_json") if record is not None else None
        if record is not None and record.get("status") == "draft" and not (
            isinstance(template, dict) and template and _lint_and_log(template, "approval")
        ):
            logger.warning("Not promoting %s: its draft could not be approved", slug)
            return

    before = lifecycle_manager.get_state(slug)
    try:
        lifecycle_manager.promote(
            slug, target, trigger="human_promote", actor=actor, reason=reason,
        )
        logger.info("Lifecycle promoted: %s → %s", slug, to_state)
    except Exception:
        logger.warning(
            "Lifecycle promote failed for %s → %s",
            slug, to_state, exc_info=True,
        )
    after = lifecycle_manager.get_state(slug)
    if before == after or ProcedureLifecycle.AGENT_READY not in (before, after):
        return
    if after is ProcedureLifecycle.AGENT_READY and record is not None:
        approve(slug)  # type: ignore[misc]  # set with record above
    if after is ProcedureLifecycle.AGENT_READY and claude_skill_writer is not None:
        proc = knowledge_base.get_procedure(slug) if knowledge_base is not None else None
        if proc is not None:
            try:
                claude_skill_writer.write_procedure(proc)
            except Exception:
                logger.warning("Claude Skill export failed for %s", slug, exc_info=True)
    _refresh_claude_commands()


def _process_merge_trigger(
    trigger: dict,
    *,
    procedure_curator: "ProcedureCurator | None" = None,
) -> None:
    """Handle a ``merge`` trigger written by the SwiftUI app."""
    # The SwiftUI app writes procedure_slug + merge_target; also accept
    # the slug_a/slug_b form for CLI callers.
    slug_a = trigger.get("slug_a") or trigger.get("procedure_slug", "")
    slug_b = trigger.get("slug_b") or trigger.get("merge_target", "")
    action = trigger.get("action", "merge")

    if not slug_a or not slug_b:
        logger.warning("Merge trigger missing slug_a/slug_b or procedure_slug/merge_target")
        return

    # Handle dismiss action
    if action == "dismiss":
        logger.info("Processing merge dismiss: %s + %s", slug_a, slug_b)
        if procedure_curator is not None:
            try:
                procedure_curator.dismiss_merge(slug_a, slug_b)
                logger.info("Merge dismissed: %s + %s", slug_a, slug_b)
            except Exception:
                logger.warning("Merge dismiss failed for %s + %s", slug_a, slug_b, exc_info=True)
        else:
            logger.warning("Merge dismiss received but procedure_curator is not available")
        return

    logger.info("Processing merge trigger: %s + %s", slug_a, slug_b)

    if procedure_curator is not None:
        try:
            result = procedure_curator.execute_merge(slug_a, slug_b, actor="human")
            logger.info("Merge result: %s", result)
        except Exception:
            logger.warning("Merge trigger failed for %s + %s", slug_a, slug_b, exc_info=True)
    else:
        logger.warning("Merge trigger received but procedure_curator is not available")


def _process_drift_reviewed_trigger(
    trigger: dict,
    *,
    procedure_curator: "ProcedureCurator | None" = None,
) -> None:
    """Handle a ``drift-reviewed`` trigger written by the SwiftUI app."""
    slug = trigger.get("procedure_slug", "")
    drift_type = trigger.get("drift_type", "")
    action = trigger.get("action", "dismiss")

    if not slug:
        logger.warning("Drift reviewed trigger missing procedure_slug")
        return

    logger.info("Processing drift reviewed trigger: %s type=%s action=%s", slug, drift_type, action)

    if procedure_curator is not None:
        try:
            if action == "dismiss" and drift_type:
                procedure_curator.dismiss_drift(slug, drift_type)
                logger.info("Drift signal dismissed: %s/%s", slug, drift_type)
            elif action in ("reviewed", "acknowledged"):
                # Mark the procedure's staleness as confirmed.
                # `datetime` and `timezone` are imported at module level
                # (line 20) — do NOT add a late `from datetime import ...`
                # here, Python would treat both as locals to this whole
                # function. See the FocusProcessor regression (issue #1).
                proc = procedure_curator._kb.get_procedure(slug)
                if proc is not None:
                    proc.setdefault("staleness", {})["last_confirmed"] = datetime.now(timezone.utc).isoformat()
                    procedure_curator._kb.save_procedure(proc)
                    logger.info("Drift reviewed and confirmed for %s", slug)
            elif action == "revert":
                # Revert is treated as a dismiss + record for now
                if drift_type:
                    procedure_curator.dismiss_drift(slug, drift_type)
                logger.info("Drift revert recorded for %s", slug)
        except Exception:
            logger.warning("Drift reviewed trigger failed for %s", slug, exc_info=True)
    else:
        logger.warning("Drift reviewed trigger received but procedure_curator is not available")


def _process_lifecycle_dismiss_trigger(
    trigger: dict,
    *,
    procedure_curator: "ProcedureCurator | None" = None,
) -> None:
    """Handle a ``lifecycle-dismiss`` trigger: record that user dismissed an upgrade suggestion."""
    slug = trigger.get("procedure_slug", "")
    if slug and procedure_curator is not None:
        try:
            # Record dismissal so upgrade card doesn't reappear
            decisions = procedure_curator._decisions
            if not isinstance(decisions, dict):
                # _decisions is a list of CurationDecision dicts; use _save_decisions
                # to persist.  We record via a special dismiss decision entry.
                procedure_curator._dismissed_drift.append({
                    "slug": slug,
                    "drift_type": "__lifecycle_upgrade_dismissed__",
                })
                procedure_curator._save_decisions()
            logger.info("Lifecycle upgrade dismissed for %s", slug)
        except Exception:
            logger.debug("Lifecycle dismiss failed", exc_info=True)


def _export_sop_templates(
    sop_templates: list[dict],
    *,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
) -> None:
    """Export SOP templates via all active adapters.

    Shared helper used by approval triggers and retry triggers to avoid
    duplicating the multi-adapter export logic.
    """
    from agenthandover_worker.sop_dedup import deduplicate_templates
    sop_templates = deduplicate_templates(sop_templates, _status_dir())

    # Write v3 procedures to the knowledge base first so all adapters
    # (including OpenClaw) can use the richer v3 format.
    if procedure_writer is not None:
        for tpl in sop_templates:
            try:
                procedure_writer.write_procedure(
                    tpl,
                    source="sop_pipeline",
                    source_id=tpl.get("slug", "unknown"),
                )
            except Exception:
                logger.debug(
                    "Failed to write procedure for %s",
                    tpl.get("slug", "?"), exc_info=True,
                )

    # Export via OpenClaw (uses v3 procedure from KB when available)
    try:
        paths = _export_via_adapter(
            openclaw_writer, sop_templates, procedure_writer,
        )
        logger.info("Exported %d SOP(s) via primary adapter", len(paths))
    except Exception:
        logger.exception("SOP export failed (primary adapter)")

    # Export via SKILL.md (uses v3 procedure when available)
    if skill_md_writer is not None:
        try:
            _export_via_adapter(
                skill_md_writer, sop_templates, procedure_writer,
            )
        except Exception:
            logger.exception("SKILL.md export failed")

    # Export via Claude Code skills (uses v3 procedure when available)
    if claude_skill_writer is not None:
        try:
            _export_via_adapter(
                claude_skill_writer, sop_templates, procedure_writer,
            )
        except Exception:
            logger.exception("Claude Code skill export failed")

    if kb_export_adapter is not None:
        try:
            kb_export_adapter.write_all_sops(sop_templates)
        except Exception:
            logger.debug("KB export adapter failed", exc_info=True)

    _save_sop_cache(sop_templates)


def _sweep_stale_focus_session(
    state_dir: Path, max_session_hours: float,
) -> bool:
    """Close a focus recording nobody stopped.

    When ``focus-session.json`` still says ``recording`` and was written
    more than *max_session_hours* ago (the app or CLI crashed, or the user
    forgot), rewrite it as ``stopped`` with ``stop_reason = "stale"``.  The
    daemon then stops tagging frames, and the normal focus path processes
    the session's rows on this or the next tick.  Returns True when a
    session was closed.
    """
    import time as _time

    signal_path = state_dir / "focus-session.json"
    try:
        age_seconds = _time.time() - signal_path.stat().st_mtime
        with open(signal_path) as f:
            signal = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(signal, dict) or signal.get("status") != "recording":
        return False
    if age_seconds < max_session_hours * 3600:
        return False

    signal["status"] = "stopped"
    signal["stop_reason"] = "stale"
    try:
        _atomic_write_result(signal_path, signal)
    except Exception:
        logger.warning("Failed to close stale focus session", exc_info=True)
        return False
    logger.warning(
        "Focus session '%s' (%s) was still recording after %.1f h "
        "(limit %.1f h); closed as stale",
        signal.get("title", ""), signal.get("session_id", ""),
        age_seconds / 3600, max_session_hours,
    )
    return True


def _clear_focus_signal(signal_path: Path) -> None:
    """Remove the focus session signal file."""
    try:
        signal_path.unlink(missing_ok=True)
    except Exception:
        logger.debug("Failed to remove focus-session.json", exc_info=True)


_SOP_CACHE_FILE = "sop-templates-cache.json"


def _save_sop_cache(sop_templates: list[dict]) -> None:
    """Persist SOP templates so the export trigger can re-read them later."""
    if not sop_templates:
        return
    cache_path = _status_dir() / _SOP_CACHE_FILE
    try:
        from agenthandover_worker.exporter import AtomicWriter
        AtomicWriter.write(cache_path, json.dumps(sop_templates, indent=2, default=str))
    except Exception:
        logger.debug("Failed to write SOP template cache", exc_info=True)


def _load_sop_cache() -> list[dict]:
    """Read cached SOP templates written by the pipeline."""
    cache_path = _status_dir() / _SOP_CACHE_FILE
    if not cache_path.is_file():
        return []
    try:
        with open(cache_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read SOP template cache", exc_info=True)
    return []


def _check_export_trigger(
    trigger: dict,
    *,
    openclaw_writer: "SOPExportAdapter",
    sops_dir: Path,
    procedure_writer: "ProcedureWriter | None" = None,
) -> None:
    """Handle an ``export`` trigger written by the CLI.

    The CLI ``agenthandover export`` writes a trigger file requesting re-export of
    existing SOPs in a specific format (skill-md, generic, openclaw).  The
    worker picks it up here, creates the appropriate writer, runs the re-export
    via ``_export_via_adapter()`` for v3-aware rendering, and removes the
    trigger.
    """
    fmt = trigger.get("format", "")
    sop_slug = trigger.get("sop_slug")
    output_dir = trigger.get("output_dir")
    logger.info(
        "Processing export trigger: format=%s slug=%s output=%s",
        fmt, sop_slug, output_dir,
    )

    # Load cached SOP templates from the last pipeline run
    sop_templates = _load_sop_cache()

    if not sop_templates:
        logger.warning(
            "Export trigger: no cached SOP templates found.  "
            "Run the pipeline first so SOPs are induced and cached."
        )
        # Still remove the trigger so it doesn't loop
        return

    if sop_slug:
        sop_templates = [
            t for t in sop_templates if t.get("slug") == sop_slug
        ]
        if not sop_templates:
            logger.warning("Export trigger: no SOP with slug '%s' found", sop_slug)
            return

    # Resolve output directory: trigger output_dir overrides default
    workspace_dir = Path(output_dir) if output_dir else sops_dir.parent.parent
    exported = 0

    if fmt in ("skill-md", "all"):
        try:
            from agenthandover_worker.skill_md_writer import SkillMdWriter
            writer = SkillMdWriter(workspace_dir=workspace_dir)
            paths = _export_via_adapter(writer, sop_templates, procedure_writer)
            exported += len(paths)
            logger.info("Export trigger: wrote %d SKILL.md file(s)", len(paths))
        except Exception:
            logger.exception("Export trigger: SKILL.md write failed")

    if fmt in ("openclaw", "all"):
        try:
            if output_dir:
                # OpenClawWriter is imported at module level (line 33) —
                # do NOT add a late import here, Python would treat it
                # as local to this entire function.
                oc_writer = OpenClawWriter(workspace_dir=Path(output_dir))
            else:
                oc_writer = openclaw_writer
            paths = _export_via_adapter(oc_writer, sop_templates, procedure_writer)
            exported += len(paths)
            logger.info("Export trigger: wrote %d OpenClaw file(s)", len(paths))
        except Exception:
            logger.exception("Export trigger: OpenClaw write failed")

    if fmt in ("generic", "all"):
        try:
            from agenthandover_worker.generic_writer import GenericWriter
            out = Path(output_dir) if output_dir else sops_dir
            writer = GenericWriter(output_dir=out, json_export=True)
            paths = _export_via_adapter(writer, sop_templates, procedure_writer)
            exported += len(paths)
            logger.info("Export trigger: wrote %d generic file(s)", len(paths))
        except Exception:
            logger.exception("Export trigger: generic write failed")

    if fmt in ("claude-skill", "all"):
        try:
            from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
            skills_dir = (
                Path(output_dir) / "skills" if output_dir
                else load_config().export.skills_dir
            )
            writer = ClaudeSkillWriter(skills_dir=skills_dir)
            paths = _export_via_adapter(writer, sop_templates, procedure_writer)
            exported += len(paths)
            logger.info("Export trigger: wrote %d Claude Code skill(s)", len(paths))
        except Exception:
            logger.exception("Export trigger: Claude Code skill write failed")

    if exported == 0 and fmt not in ("skill-md", "openclaw", "generic", "claude-skill", "all"):
        logger.warning("Export trigger: unsupported format '%s'", fmt)


def _persist_vlm_jobs(db: "WorkerDB", vlm_queue: VLMFallbackQueue) -> None:
    """Persist in-memory VLM queue jobs to the database."""
    from agenthandover_worker.vlm_queue import VLMJobStatus

    persisted = 0
    for job in vlm_queue._jobs:
        if job.status == VLMJobStatus.PENDING:
            ttl_str = (
                job.ttl_expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                if job.ttl_expires_at
                else ""
            )
            if db.enqueue_vlm_job(
                job_id=job.job_id,
                event_id=job.event_id,
                priority=job.priority_score,
                ttl_expires_at=ttl_str,
            ):
                persisted += 1
    if persisted:
        logger.info("Persisted %d VLM jobs to database", persisted)


def _process_vlm_jobs(
    db: "WorkerDB",
    pending_jobs: list[dict],
    vlm_worker: object,
    vlm_queue: "VLMFallbackQueue | None" = None,
) -> list[str]:
    """Process pending VLM jobs from the database using the VLM worker.

    For each job:
    1. Fetch the associated event for context
    2. Build a VLMRequest with event metadata
    3. Call vlm_worker.process_job() (not .infer() — that's the backend API)
    4. Store the result and mark as completed/failed
    5. Return event IDs for successful completions (for reconciliation)
    """
    from agenthandover_worker.vlm_worker import VLMRequest

    import json as _json

    completed_event_ids: list[str] = []

    for job_row in pending_jobs:
        job_id = job_row.get("id", "")
        event_id = job_row.get("event_id", "")

        try:
            # Fetch the event to get screenshot/DOM context
            event = db.get_event_by_id(event_id)
            if event is None:
                logger.warning("VLM job %s: event %s not found, marking failed", job_id, event_id)
                db.mark_vlm_job_failed(job_id)
                continue

            # Parse event metadata for VLMRequest fields
            kind_json = event.get("kind_json", "{}")
            window_json = event.get("window_json")

            kind_data = {}
            try:
                kind_data = _json.loads(kind_json) if kind_json else {}
            except (ValueError, TypeError):
                pass

            window_data = {}
            try:
                window_data = _json.loads(window_json) if window_json else {}
            except (ValueError, TypeError):
                pass

            # Extract event type from kind data
            event_type = "unknown"
            if isinstance(kind_data, dict):
                # kind_json is an enum variant like {"DwellSnapshot": {...}}
                event_type = next(iter(kind_data), "unknown")

            # Build VLMRequest with available context
            request = VLMRequest(
                job_id=job_id,
                dom_context=window_json or None,
                target_description=window_data.get("title", ""),
                event_type=event_type,
            )

            # process_job() handles prompt building, injection defense,
            # budget checks, and backend dispatch
            response = vlm_worker.process_job(request)  # type: ignore[union-attr]

            if response.success:
                result_dict = {
                    "target_description": response.target_description,
                    "suggested_selector": response.suggested_selector,
                    "confidence_boost": response.confidence_boost,
                    "reasoning": response.reasoning,
                    "inference_time_seconds": response.inference_time_seconds,
                }
                db.mark_vlm_job_completed(
                    job_id, result_json=_json.dumps(result_dict)
                )
                # Reconcile in-memory queue so backpressure stays accurate
                if vlm_queue is not None:
                    try:
                        compute_min = response.inference_time_seconds / 60.0
                        vlm_queue.record_completion(job_id, compute_min, result_dict)
                    except KeyError:
                        pass  # job may not exist in memory (DB-only)
                completed_event_ids.append(event_id)
                logger.info(
                    "VLM job %s completed (%.1fs, boost=%.2f)",
                    job_id,
                    response.inference_time_seconds,
                    response.confidence_boost,
                )
            else:
                # Budget exhaustion is transient — leave job as pending so it
                # gets retried after the daily budget resets.
                if response.error and "budget" in response.error.lower():
                    logger.info(
                        "VLM job %s deferred: %s (will retry after budget reset)",
                        job_id,
                        response.error,
                    )
                    # All remaining jobs are also over budget; stop this cycle.
                    break

                logger.warning(
                    "VLM job %s rejected: %s", job_id, response.error
                )
                db.mark_vlm_job_failed(job_id)
                # Reconcile in-memory queue for failed jobs too
                if vlm_queue is not None:
                    for mem_job in vlm_queue._jobs:
                        if mem_job.job_id == job_id:
                            from agenthandover_worker.vlm_queue import VLMJobStatus
                            mem_job.status = VLMJobStatus.FAILED
                            break

        except Exception:
            logger.warning("VLM job %s failed", job_id, exc_info=True)
            db.mark_vlm_job_failed(job_id)

    return completed_event_ids


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
        "Start it with: agenthandover start",
        _DB_RETRY_MAX_SECONDS,
    )
    return False


def main(argv: list[str] | None = None) -> None:
    """Worker entry-point: connect to DB, poll for work, run until shutdown."""
    args = _parse_args(argv)

    if args.backfill_procedures:
        # One-shot maintenance: no PID file, status file or DB, so it is
        # safe to run while the worker service is running.
        from agenthandover_worker.knowledge_base import KnowledgeBase

        logging.basicConfig(level=getattr(logging, args.log_level))
        count = KnowledgeBase(root=args.knowledge_dir).backfill_procedures()
        print(f"Backfilled {count} procedure(s) in {args.knowledge_dir}")
        return

    if _platform.system() == "Darwin":
        log_dir = Path.home() / "Library" / "Application Support" / "agenthandover" / "logs"
    else:
        log_dir = Path.home() / ".local" / "share" / "agenthandover" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=_log_handlers(log_dir, stderr_is_tty=sys.stderr.isatty()),
    )

    # Signal handling — set up BEFORE the DB retry loop so we can
    # exit cleanly if the user hits Ctrl-C while waiting.
    shutdown_flag: list[bool] = [False]

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — requesting shutdown", sig_name)
        shutdown_flag[0] = True

    previous_signal_handlers = {
        signum: signal.signal(signum, _handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def _restore_signal_handlers() -> None:
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)

    # Write PID file
    pid_dir = _status_dir()
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "worker.pid"
    pid_file.write_text(str(os.getpid()))
    logger.info("PID file written: %s", pid_file)

    logger.info("Starting agenthandover-worker v%s", _WORKER_VERSION)
    logger.info("Database path: %s", args.db_path)
    logger.info("Poll interval: %.1fs", args.poll_interval)

    # Write an early "starting" status file so the CLI and menu bar app
    # can see the worker is alive before heavy initialization finishes.
    # Previously the first status write happened ~700 lines later (after
    # DB connect, module imports, knowledge base init, vector KB, Ollama
    # health checks) — if anything failed before that point, no status
    # file was ever written and the CLI reported "not running" even though
    # the Python process was alive.  hikoae reported this exact symptom
    # on v0.2.5: launchctl showed a running process but no
    # worker-status.json appeared.
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_worker_status(
        started_at=started_at,
        events_processed_today=0,
        sops_generated=0,
        last_pipeline_duration_ms=None,
        consecutive_errors=0,
        vlm_available=False,
        vlm_mode="starting",
    )

    # Task 1: Retry loop — wait for the daemon to create the DB
    if not _wait_for_db(args.db_path, shutdown_flag):
        # Clean up PID file before exiting
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        _restore_signal_handlers()
        sys.exit(1)

    if shutdown_flag[0]:
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        _restore_signal_handlers()
        return

    # Initialize knowledge base and Phase 2+ modules
    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.evidence_tracker import EvidenceTracker
    from agenthandover_worker.procedure_writer import ProcedureWriter
    from agenthandover_worker.knowledge_export_adapter import KnowledgeBaseExportAdapter
    from agenthandover_worker.privacy_zones import PrivacyZoneChecker, PrivacyZoneConfig
    from agenthandover_worker.daily_processor import DailyBatchProcessor
    from agenthandover_worker.user_policy import UserPolicy
    from agenthandover_worker.activity_classifier import ActivityClassifier
    from agenthandover_worker.staleness_detector import StalenessDetector
    from agenthandover_worker.profile_builder import ProfileBuilder
    from agenthandover_worker.pattern_detector import PatternDetector
    from agenthandover_worker.constraint_manager import ConstraintManager
    from agenthandover_worker.execution_monitor import ExecutionMonitor
    from agenthandover_worker.correction_detector import CorrectionDetector
    from agenthandover_worker.trust_advisor import TrustAdvisor
    from agenthandover_worker.session_linker import SessionLinker
    from agenthandover_worker.daily_digest import DigestGenerator
    from agenthandover_worker.procedure_matcher import ProcedureMatcher
    from agenthandover_worker.continuity_tracker import ContinuityTracker
    from agenthandover_worker.lifecycle_manager import LifecycleManager
    from agenthandover_worker.procedure_curator import ProcedureCurator
    from agenthandover_worker.runtime_validator import RuntimeValidator
    from agenthandover_worker.config_validator import ConfigValidator
    from agenthandover_worker.ops_telemetry import OpsTelemetry, PipelineMetrics

    knowledge_base = KnowledgeBase(root=args.knowledge_dir)
    knowledge_base.ensure_structure()

    # A config still on a tier's earlier default models moves to the
    # tier's current ones (the installer does this too; this covers
    # upgrades by other means), then one read of config.toml for the run.
    _migrate_vlm_config()
    cfg = load_config()
    vlm_cfg = cfg.vlm
    if args.adapter is None:
        args.adapter = cfg.export.adapter
    logger.info("Export adapter: %s", args.adapter)

    # Config validation
    try:
        for issue in ConfigValidator().validate(cfg.raw):
            log_fn = logger.error if issue.severity == "error" else logger.warning
            log_fn("Config [%s.%s]: %s", issue.section, issue.key, issue.message)
    except Exception:
        logger.debug("Config validation skipped", exc_info=True)

    ops_telemetry = OpsTelemetry(knowledge_base)

    features = cfg.features
    logger.info(
        "Feature flags: %s",
        [k for k, v in vars(features).items() if not v] or "all enabled",
    )

    # Shared LLM reasoning utility — must be created before modules that use it.
    # VLMFallbackQueue is imported at module level (line 40) — do NOT re-import
    # it here, that shadows the module-level binding and makes Python treat
    # VLMFallbackQueue as local to this whole main() function (same class of
    # bug as the FocusProcessor regression caught by hikoae in issue #1).
    # LLMReasoner stays as a local import because it's not at module level —
    # the worker/llm_reasoning module is deliberately loaded lazily here.
    vlm_queue = VLMFallbackQueue()
    from agenthandover_worker.llm_reasoning import (
        LLMReasoner, ReasoningConfig, annotation_reasoner,
    )
    llm_reasoner = LLMReasoner(
        config=ReasoningConfig(
            model=vlm_cfg.reasoner_model, ollama_host=vlm_cfg.ollama_host,
        ),
        vlm_queue=vlm_queue,
    )

    # Initialize vector KB for semantic search and embedding storage
    vector_kb = None
    try:
        from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
        vkb_config = VectorKBConfig(
            embedding_model=cfg.embedding.model,
            ollama_host=vlm_cfg.ollama_host,
            retention_days=cfg.embedding.retention_days,
            max_rows=cfg.embedding.max_rows,
        )
        vector_kb = VectorKB(args.db_path, vkb_config)
        # Health check: verify embedding model is available
        try:
            test_emb = vector_kb.compute_embeddings(["health check"])
            if test_emb and test_emb[0]:
                logger.info(
                    "Vector KB ready (model=%s, dim=%d, vectors=%d)",
                    vkb_config.embedding_model, len(test_emb[0]),
                    vector_kb.count(),
                )
            else:
                logger.warning(
                    "Vector KB model '%s' returned empty embeddings — "
                    "run 'ollama pull %s' to enable semantic search",
                    vkb_config.embedding_model, vkb_config.embedding_model,
                )
                vector_kb.close()
                vector_kb = None
        except (ConnectionError, OllamaError):
            logger.warning(
                "Vector KB model '%s' not available — "
                "run 'ollama pull %s' to enable semantic search",
                vkb_config.embedding_model, vkb_config.embedding_model,
            )
            vector_kb.close()
            vector_kb = None
    except Exception:
        logger.debug("Vector KB initialization failed — semantic search disabled", exc_info=True)

    evidence_tracker = EvidenceTracker(knowledge_base=knowledge_base)
    lifecycle_manager = LifecycleManager(knowledge_base)
    procedure_writer = ProcedureWriter(kb=knowledge_base, evidence=evidence_tracker, lifecycle_manager=lifecycle_manager, vector_kb=vector_kb, llm_reasoner=llm_reasoner)
    kb_export_adapter = KnowledgeBaseExportAdapter(knowledge_base)
    # Build PrivacyZoneConfig from user's config.toml [privacy.zones]
    _pz_cfg = PrivacyZoneConfig()
    try:
        _pz_raw = cfg.privacy_zones
        if _pz_raw:
            _pz_cfg = PrivacyZoneConfig(
                full_observation=_pz_raw.get("full_observation", []),
                metadata_only=_pz_raw.get("metadata_only", []),
                blocked=_pz_raw.get("blocked", []),
                blocked_urls=_pz_raw.get("blocked_urls", []),
                auto_pause=_pz_raw.get("auto_pause", []),
            )
            logger.info(
                "Privacy zones loaded: %d blocked, %d metadata_only, %d blocked_urls",
                len(_pz_cfg.blocked), len(_pz_cfg.metadata_only), len(_pz_cfg.blocked_urls),
            )
    except Exception:
        logger.debug("Privacy zone config parsing failed, using defaults", exc_info=True)
    privacy_checker = PrivacyZoneChecker(config=_pz_cfg)
    staleness_detector = StalenessDetector(knowledge_base)
    profile_builder = ProfileBuilder(knowledge_base)
    activity_classifier = None
    if features.activity_classification:
        user_policy = UserPolicy(knowledge_base)
        activity_classifier = ActivityClassifier(
            profile=knowledge_base.get_profile(),
            policy=user_policy,
            # It classifies frames in the annotation phase: annotation model.
            llm_reasoner=annotation_reasoner(
                vlm_cfg.annotation_model, vlm_cfg.ollama_host, vlm_queue,
            ),
        )
    pattern_detector = PatternDetector(knowledge_base)
    constraint_manager = ConstraintManager(knowledge_base)

    # Phase 4: Execution monitoring, correction detection, trust advisor
    runtime_validator = None
    if features.runtime_validation:
        runtime_validator = RuntimeValidator(knowledge_base)
    execution_monitor = ExecutionMonitor(knowledge_base)
    correction_detector = CorrectionDetector(knowledge_base, llm_reasoner=llm_reasoner)
    trust_advisor = TrustAdvisor(knowledge_base, execution_monitor=execution_monitor)
    session_linker = SessionLinker(knowledge_base, llm_reasoner=llm_reasoner, vector_kb=vector_kb)
    procedure_matcher = ProcedureMatcher(kb=knowledge_base, vector_kb=vector_kb)
    daily_processor = DailyBatchProcessor(
        knowledge_base=knowledge_base, matcher=procedure_matcher,
    )
    continuity_tracker = None
    if features.continuity_tracking:
        continuity_tracker = ContinuityTracker(
            kb=knowledge_base,
            matcher=procedure_matcher,
        )

    # Phase 4: Variant detection and evidence normalization
    from agenthandover_worker.variant_detector import VariantDetector
    from agenthandover_worker.evidence_normalizer import EvidenceNormalizer

    variant_detector = VariantDetector()
    evidence_normalizer = EvidenceNormalizer(variant_detector=variant_detector)

    # A focus recording still "recording" after this long is closed as stale.
    focus_max_session_hours = cfg.focus.max_session_hours

    # Behavioral synthesis + evidence extraction
    from agenthandover_worker.behavioral_synthesizer import BehavioralSynthesizer, SynthesizerConfig
    _synth_model = vlm_cfg.sop_model
    _synth_host = vlm_cfg.ollama_host
    behavioral_synthesizer = BehavioralSynthesizer(
        config=SynthesizerConfig(model=_synth_model, ollama_host=_synth_host),
        vlm_queue=vlm_queue,
        knowledge_base=knowledge_base,
    )

    # Focus session questioner (gap analysis + targeted questions)
    from agenthandover_worker.focus_questioner import FocusQuestioner
    focus_questioner_inst = FocusQuestioner(llm_reasoner=llm_reasoner)

    evidence_extractor = None  # Initialized after DB is available

    # Phase 5: Curation orchestrator
    procedure_curator = None
    quality_judge = None
    if features.curation:
        from agenthandover_worker.quality_judge import QualityJudge
        quality_judge = QualityJudge(llm_reasoner)
        procedure_curator = ProcedureCurator(
            kb=knowledge_base,
            staleness_detector=staleness_detector,
            trust_advisor=trust_advisor,
            lifecycle_manager=lifecycle_manager,
            evidence_normalizer=evidence_normalizer,
            quality_judge=quality_judge,
            llm_reasoner=llm_reasoner,
        )

    logger.info("Knowledge base initialized: %s", args.knowledge_dir)

    # Check Ollama compatibility and log the active model profiles
    try:
        from agenthandover_worker.model_profiles import (
            MINIMUM_OLLAMA_VERSION_FOR_GEMMA4, get_profile, ollama_supports_gemma4,
        )

        # Warn if using Gemma 4 but Ollama is too old
        ann_model = vlm_cfg.annotation_model
        sop_model = vlm_cfg.sop_model
        uses_gemma = "gemma4" in ann_model or "gemma4" in sop_model
        if uses_gemma:
            ollama_host = vlm_cfg.ollama_host
            if not ollama_supports_gemma4(ollama_host):
                logger.warning(
                    "Gemma 4 QAT models require Ollama %s+. "
                    "Current version is too old (pulls will fail with HTTP 412). "
                    "Update the official app: https://ollama.com/download "
                    "(or: brew install --cask ollama-app). "
                    "Note: the Homebrew 'ollama' formula does not bundle the "
                    "GGUF runner — use the official app instead.",
                    MINIMUM_OLLAMA_VERSION_FOR_GEMMA4,
                )

        # Log active model profiles
        ann_profile = get_profile(ann_model)
        sop_profile = get_profile(sop_model)
        logger.info(
            "Model profiles: annotation=%s (think=%s), SOP=%s (think=%s)",
            ann_model, ann_profile.ann_think,
            sop_model, sop_profile.sop_think,
        )
    except Exception:
        logger.debug("Model profile detection failed", exc_info=True)

    # Initialize pipeline components
    episode_builder = EpisodeBuilder()
    clipboard_linker = ClipboardLinker()
    pruner = NegativeDemoPruner()
    translator = SemanticTranslator()
    scorer = ConfidenceScorer()
    # vlm_queue + llm_reasoner already created above
    index_generator = IndexGenerator()
    # Create export adapter based on config
    skill_md_writer = None
    claude_skill_writer = None
    if args.adapter == "generic":
        from agenthandover_worker.generic_writer import GenericWriter
        sop_writer = GenericWriter(output_dir=args.sops_dir, json_export=args.json_export)
    elif args.adapter == "skill-md":
        from agenthandover_worker.skill_md_writer import SkillMdWriter
        sop_writer = SkillMdWriter(workspace_dir=args.sops_dir.parent.parent)
    elif args.adapter == "claude-skill":
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
        sop_writer = ClaudeSkillWriter(skills_dir=cfg.export.skills_dir)
    elif args.adapter == "all":
        sop_writer = OpenClawWriter(workspace_dir=args.sops_dir.parent.parent)
        from agenthandover_worker.skill_md_writer import SkillMdWriter
        skill_md_writer = SkillMdWriter(workspace_dir=args.sops_dir.parent.parent)
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
        claude_skill_writer = ClaudeSkillWriter(skills_dir=cfg.export.skills_dir)
    else:
        sop_writer = OpenClawWriter(workspace_dir=args.sops_dir.parent.parent)

    # VLM initialization — supports local and remote modes
    from agenthandover_worker.vlm_worker import VLMWorker, VLMConfig, VLMBackend
    vlm_worker = None
    vlm_available = False
    vlm_mode_str = vlm_cfg.mode
    vlm_provider_str = vlm_cfg.provider

    if vlm_mode_str == "remote" and vlm_provider_str:
        # --- Remote mode: use cloud API ---
        logger.info(
            "VLM mode: remote (provider=%s). "
            "⚠️  Screenshots will be sent to %s cloud API for analysis.",
            vlm_provider_str,
            vlm_provider_str,
        )
        _provider_to_backend = {
            "openai": VLMBackend.OPENAI_COMPAT,
            "anthropic": VLMBackend.ANTHROPIC,
            "google": VLMBackend.GOOGLE_GENAI,
        }
        remote_backend = _provider_to_backend.get(vlm_provider_str)
        if remote_backend is None:
            logger.error("Unknown VLM provider: %s", vlm_provider_str)
        else:
            try:
                remote_model = vlm_cfg.model
                api_key_env = vlm_cfg.api_key_env
                # Resolve the actual API key:
                # 1. Check env var (standard path, works with shell profiles)
                # 2. Fall back to macOS Keychain (set by SwiftUI onboarding)
                api_key_value = os.environ.get(api_key_env, "") if api_key_env else ""
                if not api_key_value and vlm_provider_str:
                    api_key_value = _read_keychain_api_key(vlm_provider_str)
                    if api_key_value:
                        logger.info(
                            "Resolved API key from macOS Keychain "
                            "(set by onboarding) for provider=%s",
                            vlm_provider_str,
                        )
                vlm_config = VLMConfig(
                    backend=remote_backend,
                    mode="remote",
                    provider=vlm_provider_str,
                    remote_model=remote_model or None,
                    api_key=api_key_value or None,
                    api_key_env=api_key_env or None,
                )
                vlm_worker = VLMWorker(config=vlm_config)
                vlm_available = True
                model_display = remote_model or "(default)"
                logger.info(
                    "VLM worker initialized: remote/%s/%s",
                    vlm_provider_str,
                    model_display,
                )
            except Exception:
                logger.warning("Failed to initialize remote VLM worker", exc_info=True)
    else:
        # --- Local mode: existing detection chain ---
        from agenthandover_worker.setup_vlm import check_vlm_available
        vlm_status = check_vlm_available()
        if any(vlm_status.values()):
            logger.info("VLM backend(s) detected — attempting initialization")
            try:
                if vlm_status["ollama"]:
                    backend_type = VLMBackend.OLLAMA
                elif vlm_status.get("openai_compat"):
                    base_url = os.environ.get("AGENTHANDOVER_VLM_BASE_URL", "")
                    _local_prefixes = (
                        "http://localhost", "http://127.0.0.1",
                        "https://localhost", "https://127.0.0.1",
                        "http://[::1]",
                    )
                    if base_url and any(base_url.startswith(p) for p in _local_prefixes):
                        backend_type = VLMBackend.OPENAI_COMPAT
                    else:
                        logger.warning(
                            "OpenAI-compat backend skipped: deny_network_egress is "
                            "enforced by default. Set AGENTHANDOVER_VLM_BASE_URL to a "
                            "local server (e.g. http://localhost:8000) to use it."
                        )
                        backend_type = None  # type: ignore[assignment]
                else:
                    backend_type = None  # type: ignore[assignment]
                if backend_type is not None:
                    # [vlm] model, else the annotation model (the config
                    # loader fills that from the tier table when unset)
                    config_model = vlm_cfg.model or vlm_cfg.annotation_model
                    vlm_kwargs: dict = {"backend": backend_type}
                    if config_model:
                        vlm_kwargs["model_name"] = config_model
                    vlm_worker = VLMWorker(config=VLMConfig(**vlm_kwargs))
                    vlm_available = True
                    model_display = config_model or "(default)"
                    logger.info(
                        "VLM worker initialized (%s, model=%s) — enhanced native app observation enabled",
                        backend_type.value,
                        model_display,
                    )
                else:
                    logger.warning(
                        "VLM backend(s) detected but none usable (check config). "
                        "VLM features disabled."
                    )
            except Exception:
                logger.warning("Failed to initialize VLM worker", exc_info=True)
        else:
            logger.info(
                "VLM not installed. For better native app observation, run: agenthandover-setup-vlm"
            )

    # Initialize LLM-enhanced SOP descriptions
    sop_enhancer = None
    llm_config = dataclasses.asdict(cfg.llm)

    # CLI args override config
    if args.enhance_sops is not None:
        llm_config["enhance_sops"] = args.enhance_sops
    if args.llm_model is not None:
        llm_config["model"] = args.llm_model
    if args.max_enhancements_per_day is not None:
        llm_config["max_enhancements_per_day"] = args.max_enhancements_per_day

    if llm_config.get("enhance_sops", True):
        try:
            from agenthandover_worker.sop_enhancer import (
                SOPEnhancer,
                create_llm_backend,
            )

            # Build vlm config dict for create_llm_backend
            vlm_cfg_dict = {
                "mode": vlm_mode_str,
                "provider": vlm_provider_str,
                "model": vlm_cfg.model,
                "api_key_env": vlm_cfg.api_key_env,
            }
            llm_backend = create_llm_backend(llm_config, vlm_cfg_dict)
            if llm_backend is not None:
                sop_enhancer = SOPEnhancer(
                    backend=llm_backend,
                    max_enhancements_per_day=int(
                        llm_config.get("max_enhancements_per_day", 20)
                    ),
                )
                logger.info("SOP enhancement enabled (LLM backend ready)")
            else:
                logger.info("SOP enhancement disabled (no LLM backend available)")
        except Exception:
            logger.debug("SOP enhancement unavailable", exc_info=True)
    else:
        logger.info("SOP enhancement disabled by config")

    # Initialize v2 scene annotation pipeline
    scene_annotator: SceneAnnotator | None = None
    frame_differ: FrameDiffer | None = None
    v2_annotation_enabled = vlm_cfg.annotation_enabled
    screenshots_dir = _status_dir() / "screenshots"

    focus_processor: FocusProcessor | None = None
    task_segmenter: TaskSegmenter | None = None
    sop_generator: SOPGenerator | None = None

    annotation_cache: AnnotationCache | None = None
    if v2_annotation_enabled:
        annotation_cache = AnnotationCache(args.db_path)
        ann_config = AnnotationConfig(
            model=vlm_cfg.annotation_model,
            ollama_host=vlm_cfg.ollama_host,
            sliding_window_max_age_sec=vlm_cfg.sliding_window_max_age_sec,
            timeout_seconds=vlm_cfg.annotation_timeout_seconds,
        )
        # Initialize image embedder (SigLIP via mlx-embeddings, optional)
        # Only if user opted in via config.toml [embedding] image_embeddings = true
        _image_embedder = None
        if cfg.embedding.image_embeddings:
            try:
                from agenthandover_worker.vector_kb import ImageEmbedder
                _image_embedder = ImageEmbedder()
                if _image_embedder.available:
                    logger.info("Image embedder enabled (SigLIP %dd)", _image_embedder.DIM)
                else:
                    logger.info("Image embeddings enabled but mlx-embeddings not installed")
                    _image_embedder = None
            except Exception:
                logger.debug("Image embedder init failed", exc_info=True)
        else:
            logger.info("Image embeddings disabled (enable in config.toml [embedding] image_embeddings = true)")

        scene_annotator = SceneAnnotator(
            config=ann_config,
            image_embedder=_image_embedder,
            vector_kb=vector_kb,
            cache=annotation_cache,
        )
        # Load the annotation model now (with keep_alive) so the first
        # real frame meets a warm model.  A failure only logs and starts
        # the transport backoff; annotation retries after it.
        scene_annotator.preload()

        diff_config = DiffConfig(
            model=vlm_cfg.annotation_model,
            ollama_host=vlm_cfg.ollama_host,
        )
        frame_differ = FrameDiffer(config=diff_config)

        # SOP generator (4B thinking model) for focus session SOPs
        sop_gen_config = SOPGeneratorConfig(
            model=vlm_cfg.sop_model,
            ollama_host=vlm_cfg.ollama_host,
        )
        sop_generator = SOPGenerator(config=sop_gen_config)

        # Focus processor orchestrates: annotate → diff → generate SOP
        focus_processor = FocusProcessor(
            annotator=scene_annotator,
            differ=frame_differ,
            sop_generator=sop_generator,
        )

        # Task segmenter for passive discovery (CPU-only, no GPU)
        seg_config = SegmenterConfig(
            embedding_model=cfg.embedding.model,
            ollama_host=vlm_cfg.ollama_host,
        )
        task_segmenter = TaskSegmenter(
            config=seg_config,
            llm_reasoner=llm_reasoner,
            cluster_label_store=knowledge_base.root / "observations" / "cluster_labels.json",
            vector_kb=vector_kb,
        )

        logger.info(
            "v2 scene annotation pipeline enabled "
            "(annotation=%s, sop=%s, window=%ds)",
            vlm_cfg.annotation_model,
            vlm_cfg.sop_model,
            vlm_cfg.sliding_window_max_age_sec,
        )
    else:
        logger.info("v2 scene annotation pipeline disabled by config")

    # Initialize activity searcher for CLI search/recall commands
    activity_searcher = None
    try:
        from agenthandover_worker.activity_search import ActivitySearcher
        activity_searcher = ActivitySearcher(db_path=args.db_path)
        logger.info("Activity searcher initialized")
    except Exception:
        logger.debug("Activity searcher not available", exc_info=True)

    # Phase 3: Bundle compiler
    from agenthandover_worker.bundle_compiler import BundleCompiler
    bundle_adapters: dict[str, SOPExportAdapter] = {}
    if sop_writer is not None:
        bundle_adapters["openclaw"] = sop_writer
    if skill_md_writer is not None:
        bundle_adapters["skill_md"] = skill_md_writer
    if claude_skill_writer is not None:
        bundle_adapters["claude_skill"] = claude_skill_writer
    from agenthandover_worker.procedure_verifier import ProcedureVerifier
    procedure_verifier = ProcedureVerifier(knowledge_base, runtime_validator=runtime_validator)
    bundle_compiler = BundleCompiler(
        kb=knowledge_base,
        lifecycle=lifecycle_manager,
        verifier=procedure_verifier,
        adapters=bundle_adapters,
    )

    # Phase 4: Start query API server (if enabled in config)
    query_api_server = None
    try:
        if cfg.knowledge.query_api_enabled:
            from agenthandover_worker.query_api import TOKEN_FILE, QueryAPIServer
            _api_port = cfg.knowledge.query_api_port
            query_api_server = QueryAPIServer(
                knowledge_base=knowledge_base,
                port=_api_port,
                activity_searcher=activity_searcher,
                execution_monitor=execution_monitor,
                procedure_verifier=procedure_verifier,
                bundle_compiler=bundle_compiler,
                procedure_curator=procedure_curator,
                runtime_validator=runtime_validator,
                ops_telemetry=ops_telemetry,
                vector_kb=vector_kb,
                token_path=_status_dir() / TOKEN_FILE,
            )
            query_api_server.start()
            logger.info(
                "Query API server started on 127.0.0.1:%d (localhost-only; POST "
                "requires the X-AgentHandover-Token header from %s). "
                "Set knowledge.query_api_enabled = false in config.toml to disable.",
                _api_port, TOKEN_FILE,
            )
        else:
            logger.info("Query API server disabled (knowledge.query_api_enabled = false)")
    except Exception:
        logger.debug("Query API server not started", exc_info=True)

    # Read SOP review config
    sop_auto_approve = cfg.sop.auto_approve
    logger.info("SOP auto_approve: %s", sop_auto_approve)

    # Initialize scheduler gate (GAP 5)
    idle_gate = IdleJobGate(SchedulerConfig(**cfg.idle_jobs.scheduler_kwargs()))

    # Cumulative counters for status reporting
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_events_processed: int = 0
    total_sops_generated: int = 0
    last_pipeline_duration_ms: int | None = None
    total_v2_annotations: int = 0
    total_v2_diffs: int = 0
    v2_schema_ok: bool = False  # set after first DB check

    # Task 5: Idle progress tracking
    last_idle_log_time = time.monotonic()
    # (unprocessed, pending VLM) counts of the last poll logged at INFO.
    last_logged_poll_counts: tuple[int, int] | None = None

    # Passive discovery timer — every 2 hours (7200s), seeded when the
    # main loop starts.
    _PASSIVE_DISCOVERY_INTERVAL = 7200
    total_passive_sops: int = 0

    # Passive model use's hourly and daily budget, and why passive
    # annotation is paused (if it is).
    annotation_budget = AnnotationBudget.from_config(vlm_cfg)
    annotation_paused_reason: str | None = None
    # Annotation first, then Skills, one model loaded at a time.
    model_scheduler = ModelScheduler(
        OllamaClient(vlm_cfg.ollama_host),
        annotation_model=vlm_cfg.annotation_model,
        sop_model=vlm_cfg.sop_model,
    )
    model_scheduler.register(sop_generator, behavioral_synthesizer, llm_reasoner)

    # Write initial status file after DB connect
    _vlm_stats = vlm_queue.get_stats()
    _write_worker_status(
        started_at=started_at,
        events_processed_today=total_events_processed,
        sops_generated=total_sops_generated,
        last_pipeline_duration_ms=last_pipeline_duration_ms,
        consecutive_errors=0,
        vlm_available=vlm_available,
        vlm_queue_pending=_vlm_stats.pending_jobs,
        vlm_jobs_today=_vlm_stats.jobs_today,
        vlm_dropped_today=_vlm_stats.dropped_count,
        vlm_mode=vlm_mode_str,
        vlm_provider=vlm_provider_str or None,
        v2_annotation_enabled=v2_annotation_enabled,
        v2_annotations_today=total_v2_annotations,
        v2_diffs_today=total_v2_diffs,
    )

    with WorkerDB(args.db_path) as db:
        logger.info("Connected to database, entering main loop")
        _sweep_orphan_screenshots(db, screenshots_dir)

        # Initialize evidence extractor now that DB is available
        from agenthandover_worker.evidence_extractor import EvidenceExtractor
        evidence_extractor = EvidenceExtractor(kb=knowledge_base, db=db)

        # Check v2 schema ONCE before the main loop so focus session
        # dispatch can use the v2 path on the very first iteration.
        # Without this, v2_schema_ok starts False, focus dispatch falls
        # to v1 path, clears the signal, and v2 never processes it.
        if scene_annotator is not None:
            v2_schema_ok = _check_v2_schema(db)
            if v2_schema_ok:
                logger.info(
                    "v2 schema detected — scene annotation pipeline active"
                )
            else:
                logger.info(
                    "v2 schema not found — scene annotation disabled "
                    "(run daemon to apply migration)"
                )

        current_interval = args.poll_interval
        consecutive_errors = 0
        event_guard = EventRetryGuard(max_batch=100)

        # Timers start now; once-per-day jobs are keyed on the local date
        # they last ran, persisted so a restart does not rerun them.
        _last_passive_discovery = time.monotonic()
        daily_jobs_date = db.get_state(DAILY_JOBS_STATE_KEY) or ""
        annotation_budget.load(db.get_state(ANNOTATION_BUDGET_STATE_KEY))
        saved_budget_json = annotation_budget.to_json()

        # App and CLI requests arrive through the trigger spool; each kind
        # maps to one handler.
        export_kwargs = dict(
            openclaw_writer=sop_writer,
            skill_md_writer=skill_md_writer,
            claude_skill_writer=claude_skill_writer,
            index_generator=index_generator,
            procedure_writer=procedure_writer,
            kb_export_adapter=kb_export_adapter,
        )
        trigger_handlers: dict[str, Callable[[dict], None]] = {
            "retry": lambda t: _process_retry_triggers(
                t, db,
                focus_processor=focus_processor,
                sop_generator=sop_generator,
                screenshots_dir=screenshots_dir,
                sop_auto_approve=sop_auto_approve,
                **export_kwargs,
            ),
            "approve": lambda t: _process_approval_triggers(
                t, db, vector_kb=vector_kb, **export_kwargs,
            ),
            "failed-query": lambda t: _process_failed_query_trigger(t, db),
            "search-query": lambda t: _process_search_trigger(t, activity_searcher),
            "recall-query": lambda t: _process_recall_trigger(t, activity_searcher),
            "trust-accept": lambda t: _process_trust_accept_trigger(
                t, trust_advisor, constraint_manager,
            ),
            "trust-dismiss": lambda t: _process_trust_dismiss_trigger(t, trust_advisor),
            "staleness-reviewed": lambda t: _process_staleness_reviewed_trigger(
                t, staleness_detector, knowledge_base,
            ),
            "lifecycle-promote": lambda t: _process_lifecycle_promote_trigger(
                t,
                lifecycle_manager=lifecycle_manager,
                db=db,
                approve=lambda slug: _approve_sop(
                    db, slug, vector_kb=vector_kb, **export_kwargs,
                ),
                knowledge_base=knowledge_base,
                # With --adapter claude-skill the primary writer is the
                # Claude Skill writer.
                claude_skill_writer=(
                    claude_skill_writer
                    or (sop_writer if args.adapter == "claude-skill" else None)
                ),
            ),
            "merge": lambda t: _process_merge_trigger(
                t, procedure_curator=procedure_curator,
            ),
            "drift-reviewed": lambda t: _process_drift_reviewed_trigger(
                t, procedure_curator=procedure_curator,
            ),
            "lifecycle-dismiss": lambda t: _process_lifecycle_dismiss_trigger(
                t, procedure_curator=procedure_curator,
            ),
            "export": lambda t: _check_export_trigger(
                t,
                openclaw_writer=sop_writer,
                sops_dir=args.sops_dir,
                procedure_writer=procedure_writer,
            ),
        }
        assert set(trigger_handlers) == set(trigger_spool.KINDS)
        trigger_spool.migrate_legacy(_status_dir())

        # Force-write sops-index.json at startup so the SwiftUI app
        # immediately sees the latest data (including any new fields).
        _write_sops_index(db, knowledge_base, force=True)

        # Embed all existing KB procedures at startup (idempotent — skips unchanged)
        if vector_kb is not None:
            try:
                all_procs = knowledge_base.list_procedures()
                embedded = 0
                for proc in all_procs:
                    full = knowledge_base.get_procedure(proc.get("id", ""))
                    if full:
                        _embed_procedure(vector_kb, full)
                        embedded += 1
                if embedded:
                    logger.info("Embedded %d existing procedure(s) in vector KB", embedded)
            except Exception:
                logger.debug("Startup procedure embedding failed", exc_info=True)

        while not shutdown_flag[0]:
            # Reset vector KB embedding budget each cycle
            if vector_kb is not None:
                vector_kb.reset_cycle_budget()

            # Events handed to run_pipeline and not yet marked processed;
            # an error while this is non-empty counts against them.
            inflight_event_ids: list[str] = []
            try:
                # App and CLI requests first, oldest first
                triggers_consumed = trigger_spool.drain(
                    _status_dir(), trigger_handlers,
                )

                # A consumed trigger (merge, dismiss, promote, ...) may have
                # changed the curation queue; nothing else between daily
                # curation passes does.
                if triggers_consumed:
                    _write_curation_queue(procedure_curator)

                # Close a recording that was never stopped, so its rows get
                # processed instead of staying pending forever.
                _sweep_stale_focus_session(_status_dir(), focus_max_session_hours)

                unprocessed = db.get_unprocessed_events(
                    limit=event_guard.batch_limit
                )
                pending_vlm = db.get_pending_vlm_jobs(limit=10)

                if unprocessed or pending_vlm:
                    poll_counts = (len(unprocessed), len(pending_vlm))
                    logger.log(
                        _poll_log_level(poll_counts, last_logged_poll_counts),
                        "Poll: %d unprocessed events, %d pending VLM jobs",
                        *poll_counts,
                    )
                    last_logged_poll_counts = poll_counts
                    current_interval = _next_poll_interval(
                        current_interval, args.poll_interval,
                        busy=True, triggers_consumed=triggers_consumed,
                    )
                    # Reset idle log timer
                    last_idle_log_time = time.monotonic()
                else:
                    logger.debug("Poll: nothing to do")
                    last_logged_poll_counts = None
                    current_interval = _next_poll_interval(
                        current_interval, args.poll_interval,
                        busy=False, triggers_consumed=triggers_consumed,
                    )

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
                    # Core pipeline ALWAYS runs — episodes, translation,
                    # SOP mining, and export happen immediately so users
                    # see SOPs as soon as patterns are detected.
                    inflight_event_ids = [
                        ev["id"] for ev in unprocessed if "id" in ev
                    ]
                    pipeline_start = time.monotonic()
                    summary = run_pipeline(
                        unprocessed,
                        episode_builder=episode_builder,
                        clipboard_linker=clipboard_linker,
                        pruner=pruner,
                        translator=translator,
                        scorer=scorer,
                        vlm_queue=vlm_queue,
                        db=db,
                    )
                    pipeline_elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
                    last_pipeline_duration_ms = pipeline_elapsed_ms
                    logger.info("Pipeline summary: %s", summary)

                    total_events_processed += summary["events_in"]
                    logger.info(
                        "Processed %d events into %d episodes.",
                        summary["events_in"],
                        summary["episodes"],
                    )

                    # Mark processed events so they are not re-read (GAP 4)
                    if inflight_event_ids:
                        db.mark_events_processed(inflight_event_ids)
                    event_guard.succeeded(inflight_event_ids)
                    inflight_event_ids = []

                    # Persist in-memory VLM queue jobs to the DB so they survive restarts
                    if summary["vlm_enqueued"] > 0:
                        _persist_vlm_jobs(db, vlm_queue)

                # Process pending VLM jobs only in idle window
                if pending_vlm and vlm_worker is not None:
                    gate_result = idle_gate.check()
                    if gate_result.can_run:
                        with charged_to(annotation_budget):
                            vlm_completed_events = _process_vlm_jobs(
                                db, pending_vlm, vlm_worker, vlm_queue
                            )
                        # VLM reconciliation: reset events with completed VLM
                        # boosts so the pipeline re-evaluates them with higher
                        # confidence scores.
                        if vlm_completed_events:
                            db.mark_events_unprocessed(vlm_completed_events)
                            logger.info(
                                "VLM reconciliation: %d events reset for re-scoring with boost",
                                len(vlm_completed_events),
                            )

                # --- Annotation phase (model_scheduler.py) ---
                # Every frame this tick annotates, on the annotation model:
                # the last focus session's leftover keyframes, passive
                # frames and frame diffs.
                model_scheduler.begin_annotation()
                if focus_processor is not None and v2_schema_ok:
                    focus_processor.annotate_remaining(db)

                # --- v2 Scene Annotation Pipeline ---
                # Process unannotated screenshots and compute frame diffs.
                # Schema is checked once before the main loop; this fallback
                # re-checks if the daemon applies migration after worker start.
                if scene_annotator is not None and not v2_schema_ok:
                    v2_schema_ok = _check_v2_schema(db)
                    if v2_schema_ok:
                        logger.info(
                            "v2 schema now available — scene annotation pipeline activated"
                        )

                if scene_annotator is not None and v2_schema_ok:
                    # Passive frames older than a day are given up on.
                    for expired in db.expire_annotation_backlog(
                        max_age_hours=ANNOTATION_BACKLOG_MAX_AGE_HOURS,
                    ):
                        scene_annotator.discard_screenshot(expired, screenshots_dir)
                    # Passive annotation runs on AC, with the screen locked,
                    # or after two idle minutes, within the hourly and
                    # daily budget.  Focus recordings are held by neither
                    # (their keyframes above, their SOP in the generation
                    # phase).
                    annotation_budget.roll(datetime.now())
                    allowed, why = idle_gate.passive_annotation_allowed()
                    ann_stats = {
                        "annotated": 0, "skipped": 0, "failed": 0,
                        "blocked": 0, "deferred": 0, "budget_exhausted": False,
                    }
                    if allowed:
                        ann_stats = _process_annotations(
                            db, scene_annotator, screenshots_dir,
                            privacy_checker=privacy_checker,
                            activity_classifier=activity_classifier,
                            budget=annotation_budget,
                        )
                    if not allowed:
                        reason = why
                    elif ann_stats["budget_exhausted"]:
                        reason = annotation_budget.pause_reason()
                    else:
                        reason = None
                    if reason != annotation_paused_reason:
                        logger.info(
                            "Passive annotation %s",
                            f"paused: {reason}" if reason else "resumed",
                        )
                    annotation_paused_reason = reason
                    total_v2_annotations += ann_stats["annotated"]

                    if ann_stats["annotated"] > 0 or ann_stats["blocked"] > 0:
                        logger.info(
                            "v2 annotation: %d annotated, %d skipped, %d failed, %d blocked",
                            ann_stats["annotated"],
                            ann_stats["skipped"],
                            ann_stats["failed"],
                            ann_stats["blocked"],
                        )

                    if ops_telemetry is not None and ann_stats["annotated"] > 0:
                        try:
                            ops_telemetry.record_batch(PipelineMetrics(
                                annotation_count=ann_stats["annotated"],
                                classification_count=ann_stats["annotated"],
                            ))
                        except Exception:
                            pass

                    # Refresh FTS5 search index so newly annotated events
                    # appear in search/recall results.
                    if ann_stats["annotated"] > 0 and activity_searcher is not None:
                        try:
                            added = activity_searcher.refresh_index()
                            if added > 0:
                                logger.debug(
                                    "FTS5 index refreshed: %d new entries", added,
                                )
                        except Exception:
                            logger.debug(
                                "FTS5 index refresh failed", exc_info=True,
                            )

                if frame_differ is not None and v2_schema_ok:
                    diff_stats = _process_diffs(
                        db, frame_differ, budget=annotation_budget,
                    )
                    total_v2_diffs += diff_stats["diffs"]

                    if diff_stats["diffs"] > 0 or diff_stats["failed"] > 0:
                        logger.info(
                            "v2 frame diff: %d action diffs, %d edge cases, %d failed",
                            diff_stats["diffs"],
                            diff_stats["edge_cases"],
                            diff_stats["failed"],
                        )

                # --- Generation phase (model_scheduler.py) ---
                # SOP generation, synthesis, Q&A and the daily model jobs,
                # on the SOP model, once this tick's frames are annotated.
                # Passive discovery is due every 2 hours, or when the user
                # has been idle for 5+ minutes.
                should_segment = False
                if (
                    task_segmenter is not None
                    and sop_generator is not None
                    and v2_schema_ok
                ):
                    _now_mono_pd = time.monotonic()
                    should_segment = (
                        _now_mono_pd - _last_passive_discovery
                        >= _PASSIVE_DISCOVERY_INTERVAL
                    )
                    # Also run during extended idle (no unprocessed events
                    # for several cycles = user is away)
                    if not should_segment and not unprocessed and not pending_vlm:
                        # Check idle gate — passive discovery during idle window
                        pd_gate = idle_gate.check()
                        if pd_gate.can_run and (
                            _now_mono_pd - _last_passive_discovery >= 300
                        ):
                            should_segment = True

                    # Skip passive discovery if a focus session is pending
                    # (focus gets absolute priority)
                    _focus_pending = Path(
                        _status_dir()
                    ).joinpath("focus-session.json").exists()
                    if _focus_pending:
                        should_segment = False

                today_local = datetime.now().date().isoformat()
                focus_work = _focus_work_ready(_status_dir(), db)
                gen_plan = model_scheduler.plan_generation(
                    urgent=focus_work is not None,
                    due={
                        "passive_discovery": should_segment,
                        "daily": today_local != daily_jobs_date,
                    },
                )

                # Completed focus recordings, never batched or held for
                # memory: the v2 VLM pipeline when available (semantic SOPs
                # from screen annotations; its keyframes are annotated
                # before the SOP call switches to the generation model),
                # else v1 (episode builder + translator + one template).
                if focus_processor is not None and v2_schema_ok:
                    focus_sops = _process_focus_sessions_v2(
                        db,
                        focus_processor=focus_processor,
                        openclaw_writer=sop_writer,
                        skill_md_writer=skill_md_writer,
                        claude_skill_writer=claude_skill_writer,
                        index_generator=index_generator,
                        screenshots_dir=screenshots_dir,
                        sop_auto_approve=sop_auto_approve,
                        procedure_writer=procedure_writer,
                        kb_export_adapter=kb_export_adapter,
                        behavioral_synthesizer=behavioral_synthesizer,
                        focus_questioner=focus_questioner_inst,
                        knowledge_base=knowledge_base,
                        vector_kb=vector_kb,
                        ollama_host=vlm_cfg.ollama_host,
                        # Q&A questions on the generation phase's model
                        # (the configured SOP model): never a hardcoded
                        # default, which silently failed for users who
                        # never pulled it (fixed in v0.2.3).
                        qna_model=model_scheduler.generation_model,
                    )
                else:
                    focus_sops = _process_focus_sessions(
                        db,
                        episode_builder=episode_builder,
                        clipboard_linker=clipboard_linker,
                        translator=translator,
                        scorer=scorer,
                        vlm_queue=vlm_queue,
                        sop_enhancer=sop_enhancer,
                        openclaw_writer=sop_writer,
                        skill_md_writer=skill_md_writer,
                        claude_skill_writer=claude_skill_writer,
                        index_generator=index_generator,
                        sop_auto_approve=sop_auto_approve,
                        procedure_writer=procedure_writer,
                        kb_export_adapter=kb_export_adapter,
                    )
                if focus_sops > 0:
                    total_sops_generated += focus_sops
                    logger.info(
                        "Focus recording: generated %d SOP(s)!", focus_sops
                    )
                    _write_sops_index(db, knowledge_base)

                # --- v2 Passive Discovery (periodic) ---
                if should_segment and gen_plan.allows("passive_discovery"):
                    try:
                        with charged_to(annotation_budget):
                            pd_sops = _process_passive_discovery(
                                db,
                                knowledge_base=knowledge_base,
                                segmenter=task_segmenter,
                                sop_generator=sop_generator,
                                openclaw_writer=sop_writer,
                                skill_md_writer=skill_md_writer,
                                claude_skill_writer=claude_skill_writer,
                                index_generator=index_generator,
                                sop_auto_approve=sop_auto_approve,
                                procedure_writer=procedure_writer,
                                kb_export_adapter=kb_export_adapter,
                                continuity_tracker=continuity_tracker,
                                variant_detector=variant_detector,
                                evidence_normalizer=evidence_normalizer,
                            )
                        if pd_sops > 0:
                            total_passive_sops += pd_sops
                            total_sops_generated += pd_sops
                            logger.info(
                                "Passive discovery: generated %d SOP(s)!",
                                pd_sops,
                            )
                            _write_sops_index(db, knowledge_base)
                    except Exception:
                        logger.warning(
                            "Passive discovery failed",
                            exc_info=True,
                        )
                    _last_passive_discovery = _now_mono_pd

                # What passive model use spent survives a restart.
                if annotation_budget.to_json() != saved_budget_json:
                    saved_budget_json = annotation_budget.to_json()
                    db.set_state(ANNOTATION_BUDGET_STATE_KEY, saved_budget_json)

                # Write status file after each cycle (heartbeat).
                # Use authoritative DB count for pending VLM jobs (the
                # in-memory queue only tracks enqueue-side stats; DB-side
                # completions from _process_vlm_jobs are not reflected).
                _vlm_stats = vlm_queue.get_stats()
                try:
                    db_vlm_pending = db.count_pending_vlm_jobs()
                except Exception:
                    db_vlm_pending = _vlm_stats.pending_jobs
                _write_worker_status(
                    started_at=started_at,
                    events_processed_today=total_events_processed,
                    sops_generated=total_sops_generated,
                    last_pipeline_duration_ms=last_pipeline_duration_ms,
                    consecutive_errors=consecutive_errors,
                    vlm_available=vlm_available,
                    vlm_queue_pending=db_vlm_pending,
                    vlm_jobs_today=_vlm_stats.jobs_today,
                    vlm_dropped_today=_vlm_stats.dropped_count,
                    vlm_mode=vlm_mode_str,
                    vlm_provider=vlm_provider_str or None,
                    v2_annotation_enabled=v2_annotation_enabled,
                    v2_annotations_today=total_v2_annotations,
                    v2_diffs_today=total_v2_diffs,
                    annotation_paused_reason=annotation_paused_reason,
                    annotation_budget=annotation_budget.status(),
                    model_schedule=model_scheduler.status(),
                )

                # Refresh sops-index.json (throttled to every 5s)
                _write_sops_index(db, knowledge_base)

                # --- Once-per-day jobs: the first tick of each local day,
                # keyed on the date persisted when they last ran. ---
                run_daily = gen_plan.allows("daily")

                # The v1 miner's episode store is unused (WP-F5): drop it.
                if run_daily:
                    try:
                        db.drop_episode_store()
                    except Exception:
                        logger.debug("Episode store drop failed", exc_info=True)

                # --- Phase 2+: Daily batch + staleness (once per day) ---
                # Process *yesterday* (the local calendar day) — today is
                # still in progress and would produce an incomplete summary.
                yesterday_str = (
                    datetime.now().astimezone() - timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if run_daily:
                    try:
                        # Daily batch: aggregate yesterday's (complete) events
                        day_events = db.get_annotated_events_for_date(yesterday_str)
                        if day_events:
                            _daily_summary = daily_processor.process_day(
                                yesterday_str, day_events
                            )
                            logger.info(
                                "Daily batch: %d tasks, %.1f active hours",
                                _daily_summary.task_count,
                                _daily_summary.active_hours,
                            )
                            # Embed daily summary for semantic search
                            _embed_in_vector_kb(
                                vector_kb, "daily_summary", yesterday_str,
                                f"Daily summary for {yesterday_str}: "
                                f"{_daily_summary.task_count} tasks, "
                                f"{_daily_summary.active_hours:.1f} active hours. "
                                + " ".join(
                                    t.get("intent", "")
                                    for t in getattr(_daily_summary, "tasks", [])
                                    if t.get("intent")
                                )[:2000],
                            )
                            # Update profile + patterns periodically
                            summaries_count = len(
                                knowledge_base.list_daily_summaries(limit=5)
                            )
                            if summaries_count >= 3:
                                profile_builder.update_profile()
                                # Embed updated profile
                                try:
                                    profile = knowledge_base.get_profile()
                                    if profile:
                                        import json as _json
                                        _embed_in_vector_kb(
                                            vector_kb, "profile", "user",
                                            _json.dumps(profile, default=str)[:4000],
                                        )
                                except Exception:
                                    logger.debug("Profile embedding failed", exc_info=True)
                                # Refresh activity classifier with updated profile
                                if activity_classifier is not None:
                                    try:
                                        activity_classifier._profile = knowledge_base.get_profile()
                                    except Exception:
                                        logger.debug("Failed to refresh classifier profile", exc_info=True)
                                patterns = pattern_detector.detect_recurrence()
                                if patterns:
                                    pattern_detector.update_triggers(patterns)
                                chains = pattern_detector.detect_chains()
                                if chains:
                                    pattern_detector.update_chains(chains)
                                # Correction-to-guardrail learning
                                try:
                                    all_procs = knowledge_base.list_procedures()
                                    guardrail_count = 0
                                    for proc in all_procs:
                                        _cs = proc.get("id", proc.get("slug", ""))
                                        if not _cs:
                                            continue
                                        _guardrails = correction_detector.analyze_correction_patterns(_cs)
                                        if _guardrails:
                                            correction_detector.apply_guardrails_to_procedure(
                                                _cs, _guardrails,
                                            )
                                            guardrail_count += len(_guardrails)
                                    if guardrail_count:
                                        logger.info(
                                            "Correction learning: %d guardrail(s) applied",
                                            guardrail_count,
                                        )
                                except Exception:
                                    logger.debug(
                                        "Correction-to-guardrail learning failed",
                                        exc_info=True,
                                    )
                                # Phase 4: Session linking + trust evaluation + digest
                                try:
                                    session_linker.analyze_daily_summaries()
                                except Exception:
                                    logger.debug("Session linking failed", exc_info=True)
                                try:
                                    suggestions = trust_advisor.evaluate_all()
                                    if suggestions:
                                        logger.info(
                                            "Trust advisor: %d new suggestion(s)",
                                            len(suggestions),
                                        )
                                except Exception:
                                    logger.debug("Trust evaluation failed", exc_info=True)

                                # --- Behavioral re-synthesis trigger ---
                                try:
                                    all_procs_for_synth = knowledge_base.list_procedures()
                                    synth_count = 0
                                    # Pre-load daily summaries once for all procedures
                                    _synth_daily = [
                                        knowledge_base.get_daily_summary(d)
                                        for d in knowledge_base.list_daily_summaries(limit=7)
                                    ]
                                    _synth_daily = [s for s in _synth_daily if s is not None]

                                    for proc_info in all_procs_for_synth:
                                        p_slug = proc_info.get("id", proc_info.get("slug", ""))
                                        if not p_slug:
                                            continue
                                        proc = knowledge_base.get_procedure(p_slug)
                                        if proc is None:
                                            continue
                                        if proc.get("lifecycle_state") == "archived":
                                            continue
                                        if not behavioral_synthesizer.should_synthesize(proc):
                                            continue

                                        # Build observations for the synthesizer.
                                        #
                                        # For each observation record with a
                                        # session_id (focus-derived), look up
                                        # the actual events from the DB and
                                        # build rich per-frame dicts with
                                        # verbatim text (email_addresses,
                                        # urls, typed_text, screen_text,
                                        # active_element) via
                                        # FocusProcessor._build_pre_analysis_obs.
                                        # This surfaces the same verbatim
                                        # grounding to the synthesizer prompt's
                                        # TIMELINE EVIDENCE section that focus
                                        # recordings already get.
                                        #
                                        # For observations without a session_id
                                        # or where event lookup fails, fall
                                        # back to the abstracted SOP step list
                                        # so nothing gets worse.
                                        #
                                        # NOTE: FocusProcessor is imported at
                                        # module level (line 29) — DO NOT add a
                                        # late `from ... import FocusProcessor`
                                        # here. A redundant inner import shadows
                                        # the module-level binding and makes
                                        # Python treat FocusProcessor as local
                                        # to main() for the ENTIRE function,
                                        # which causes earlier uses at lines
                                        # 4842/4905 to raise UnboundLocalError.
                                        # Regression caught by hikoae in v0.2.2
                                        # (issue #1 follow-up comment).
                                        _synth_obs: list[list[dict]] = []
                                        evidence = proc.get("evidence", {})
                                        obs_records = evidence.get("observations", [])
                                        proc_steps = proc.get("steps", [])

                                        for _rec in obs_records:
                                            rich_frames: list[dict] = []
                                            _sid = _rec.get("session_id") if isinstance(_rec, dict) else None
                                            if _sid:
                                                try:
                                                    _events = db.get_focus_session_events(_sid)
                                                except Exception:
                                                    _events = []
                                                for _ev in _events:
                                                    _ann_raw = _ev.get("scene_annotation_json")
                                                    if not _ann_raw:
                                                        continue
                                                    try:
                                                        _ann = json.loads(_ann_raw) if isinstance(_ann_raw, str) else _ann_raw
                                                    except (json.JSONDecodeError, TypeError):
                                                        continue
                                                    rich_frames.append(
                                                        FocusProcessor._build_pre_analysis_obs({
                                                            "annotation": _ann,
                                                        })
                                                    )
                                            if rich_frames:
                                                _synth_obs.append(rich_frames)
                                            elif proc_steps:
                                                # Fallback: abstracted step dicts
                                                _synth_obs.append(proc_steps)

                                        insights = behavioral_synthesizer.synthesize(
                                            p_slug, proc, _synth_obs,
                                            daily_summaries=_synth_daily,
                                        )
                                        if insights.strategy or insights.guardrails:
                                            updated = behavioral_synthesizer.merge_insights_into_procedure(
                                                proc, insights,
                                            )
                                            knowledge_base.save_procedure(updated)
                                            synth_count += 1
                                    if synth_count:
                                        logger.info(
                                            "Behavioral synthesis: %d procedure(s) updated",
                                            synth_count,
                                        )
                                except Exception:
                                    logger.debug(
                                        "Behavioral re-synthesis failed", exc_info=True,
                                    )

                                # --- Pre-expiry evidence extraction ---
                                try:
                                    extracted = evidence_extractor.extract_all_pending()
                                    if extracted:
                                        logger.info(
                                            "Evidence extraction: %d procedure(s) processed",
                                            extracted,
                                        )
                                except Exception:
                                    logger.debug(
                                        "Pre-expiry evidence extraction failed",
                                        exc_info=True,
                                    )

                            # Seed default constraints if constraints file is empty.
                            # Runs on every daily batch (outside the summaries >= 3
                            # gate) so constraints are available from day one.  The
                            # inner guard ensures the write happens at most once.
                            try:
                                existing_constraints = knowledge_base.get_constraints()
                                if not existing_constraints.get("global"):
                                    knowledge_base.update_constraints({
                                        "global": {
                                            "default_trust_level": "observe",
                                            "require_approval_for_destructive": True,
                                            "max_autonomous_actions_per_hour": 10,
                                        },
                                        "per_procedure": existing_constraints.get(
                                            "per_procedure", {}
                                        ),
                                    })
                                    logger.info("Default global constraints seeded")
                            except Exception:
                                logger.debug(
                                    "Constraint seeding failed", exc_info=True
                                )
                            # Generate daily digest
                            try:
                                digest_generator = DigestGenerator(knowledge_base, db=db)
                                digest = digest_generator.generate(yesterday_str)
                                digest_generator.save_digest(digest)
                                logger.info("Daily digest generated for %s", yesterday_str)
                            except Exception:
                                logger.debug("Digest generation failed", exc_info=True)
                    except Exception:
                        logger.debug("Daily batch failed", exc_info=True)

                    # Workflow observations: segment and observation TTLs,
                    # the promoted workflows' recurrence and value.
                    workflow_mining.run_daily(
                        db, knowledge_base, now=datetime.now(timezone.utc),
                    )

                    # Preserve OCR text from soon-to-expire events
                    try:
                        pcount, pitems = db.preserve_ocr_text(retention_days=14)
                        if pcount:
                            logger.info("Preserved %d texts before expiry", pcount)
                        # Embed preserved texts in vector KB
                        if pitems and vector_kb is not None:
                            try:
                                vk_items = [
                                    ("preserved_text", f"{eid}:{src}", txt[:2000])
                                    for eid, src, txt in pitems
                                ]
                                vector_kb.upsert_batch(vk_items)
                            except Exception:
                                logger.debug("Preserved text embedding failed", exc_info=True)
                    except Exception:
                        logger.debug("OCR text preservation failed", exc_info=True)

                # Staleness check
                if run_daily:
                    try:
                        reports = staleness_detector.check_all()
                        stale_count = sum(
                            1 for r in reports if r.status != "current"
                        )
                        if stale_count > 0:
                            logger.info(
                                "Staleness check: %d/%d procedures need attention",
                                stale_count, len(reports),
                            )
                        # Persist staleness status back into procedure JSON
                        # so the SwiftUI MicroReviewView can read it.
                        for report in reports:
                            try:
                                proc = knowledge_base.get_procedure(report.slug)
                                if proc is not None:
                                    staleness = proc.get("staleness", {})
                                    staleness["status"] = report.status
                                    staleness["signals"] = [
                                        {
                                            "type": s.type,
                                            "detail": s.detail,
                                            "first_seen": s.first_seen,
                                        }
                                        for s in report.signals
                                    ]
                                    staleness["recommended_action"] = report.recommended_action
                                    proc["staleness"] = staleness
                                    knowledge_base.save_procedure(proc)
                            except Exception:
                                logger.debug(
                                    "Failed to persist staleness for %s",
                                    report.slug,
                                    exc_info=True,
                                )
                    except Exception:
                        logger.debug("Staleness check failed", exc_info=True)

                    # Lifecycle: auto-demote stale procedures
                    if features.lifecycle_management:
                        try:
                            transitions = lifecycle_manager.apply_auto_transitions()
                            if transitions:
                                logger.info("Lifecycle: %d auto-transition(s)", len(transitions))
                        except Exception:
                            logger.debug("Lifecycle auto-transitions failed", exc_info=True)
                        # Write what the gate now lets through (approved
                        # since the last pass); drafts are never written.
                        try:
                            bundle_compiler.compile_all()
                        except Exception:
                            logger.warning("Bundle compile failed", exc_info=True)

                    # Phase 5: Curation queue refresh
                    try:
                        with charged_to(annotation_budget):
                            curation_summary = procedure_curator.curate()
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

                    # Write curation-queue.json for SwiftUI app
                    _write_curation_queue(procedure_curator)

                    _run_daily_maintenance(
                        vector_kb=vector_kb,
                        activity_searcher=activity_searcher,
                        annotation_cache=annotation_cache,
                    )
                    db.set_state(DAILY_JOBS_STATE_KEY, today_local)
                    daily_jobs_date = today_local

                # End of the generation phase: unload the SOP model.
                model_scheduler.end_generation()

                # Any iteration that gets here without raising is healthy,
                # whether or not it had events to process.
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                logger.error(
                    "Error in main loop (consecutive: %d)",
                    consecutive_errors,
                    exc_info=True,
                )
                to_quarantine = event_guard.failed(inflight_event_ids)
                if to_quarantine:
                    db.quarantine_events(to_quarantine)
                    logger.error(
                        "Quarantined event %s: the pipeline failed on it "
                        "alone repeatedly",
                        ", ".join(to_quarantine),
                    )
                if consecutive_errors == ERROR_THRESHOLD:
                    logger.critical(
                        "%d consecutive errors; backing off (up to 5 minutes "
                        "between attempts) and staying up",
                        ERROR_THRESHOLD,
                    )
                # Keep the heartbeat fresh so the app and doctor see a
                # worker that is alive but failing, not a dead one.
                try:
                    _write_worker_status(
                        started_at=started_at,
                        events_processed_today=total_events_processed,
                        sops_generated=total_sops_generated,
                        last_pipeline_duration_ms=last_pipeline_duration_ms,
                        consecutive_errors=consecutive_errors,
                        vlm_available=vlm_available,
                        vlm_mode=vlm_mode_str,
                        vlm_provider=vlm_provider_str or None,
                    )
                except Exception:
                    logger.debug("Failed to write worker status", exc_info=True)
                if args.once:
                    break
                interruptible_sleep(
                    error_backoff_seconds(consecutive_errors),
                    lambda: shutdown_flag[0],
                )
                continue

            if args.once:
                break
            time.sleep(current_interval)

    # Stop query API server if running
    if query_api_server is not None:
        try:
            query_api_server.stop()
            logger.info("Query API server stopped")
        except Exception:
            pass

    # Remove PID file and status file on clean shutdown.  A --once run
    # keeps worker-status.json: it records that iteration, and readers
    # treat it as not running because the PID is gone.
    try:
        pid_file.unlink(missing_ok=True)
    except Exception:
        pass
    if not args.once:
        _remove_worker_status()
    _restore_signal_handlers()

    logger.info("Worker shut down cleanly")


if __name__ == "__main__":
    main()

"""AgentHandover apprentice worker entry-point.

Provides the main loop that reads from the daemon's SQLite database
and drives episode building, semantic translation, SOP induction,
formatting, and export.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker import run_signals
from agenthandover_worker import workflow_mining
from agenthandover_worker.clipboard_linker import ClipboardLinker
from agenthandover_worker.event_helpers import focused_field, parse_json_field
from agenthandover_worker.config import EXPORT_ADAPTERS, config_path, load_config
from agenthandover_worker.confidence import ConfidenceScorer
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.evidence_tracker import ObservationEvidence, append_observation
from agenthandover_worker.exporter import IndexGenerator
from agenthandover_worker.focus_processor import (
    FocusProcessor,
    recorded_observations,
    session_list_choices,
)
from agenthandover_worker.frame_differ import FrameDiffer
from agenthandover_worker.inference_runner import (
    P2, STATE_LOCK, InferenceRunner, model_wait, model_work,
)
from agenthandover_worker.negative_demo import NegativeDemoPruner
from agenthandover_worker.model_profiles import (
    migrate_embedding_file, migrate_vlm_file, served_models, upgrade_plan,
)
from agenthandover_worker.ops_telemetry import COMPUTE, MODEL_USAGE
from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.annotation_cache import AnnotationCache
from agenthandover_worker.scene_annotator import (
    SceneAnnotator,
    page_for_event,
    settle_typed_text,
    skip_record,
)
from agenthandover_worker.sop_generator import SOPGenerator, _generate_slug
from agenthandover_worker.sop_linter import lint_sop
from agenthandover_worker.task_segmenter import TaskSegmenter
from agenthandover_worker.scheduler import (
    AnnotationBudget,
    charged_to,
)
from agenthandover_worker.translator import SemanticTranslator
from agenthandover_worker.loop_guard import (
    interruptible_sleep,
)

logger = logging.getLogger("agenthandover_worker")

# Default paths assume a single-user system. Use --db-path and --sops-dir
# CLI args to override for multi-user or containerized deployments.
import platform as _platform

from agenthandover_worker.paths import DEFAULT_DB_PATH, KNOWLEDGE_BASE_DIR, data_dir

DEFAULT_SOPS_DIR = Path.home() / ".openclaw" / "workspace" / "memory" / "apprentice"
POLL_INTERVAL_SECONDS = 2.0

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
# A control tick waits this long for the state the inference thread holds
# while it runs Python; then it writes only the heartbeat (inference_runner).
CONTROL_LOCK_WAIT_SECONDS = 5.0


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
    memory: "MemoryWriter | None" = None,
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

    A completed annotation describes the frame's Memory moment
    (*memory*, ``MemoryWriter.annotated``).
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

            settle_typed_text(db, event, result.annotation)
            db.save_annotation(
                event_id,
                json.dumps(result.annotation),
                status="completed",
            )
            if memory is not None:
                try:
                    memory.annotated(event, result.annotation)
                except sqlite3.Error:
                    logger.warning(
                        "Memory description of %s not updated", event_id[:8], exc_info=True,
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
# the budget counters for the current local day (passive annotation's and
# the reasoner's).
DAILY_JOBS_STATE_KEY = "daily_jobs_date"
ANNOTATION_BUDGET_STATE_KEY = "annotation_budget"
REASONING_BUDGET_STATE_KEY = "reasoning_budget"

# Passive frames still pending after this long are given up on.
ANNOTATION_BACKLOG_MAX_AGE_HOURS = 24.0


def _run_daily_maintenance(
    *,
    vector_kb: "VectorKB | None" = None,
    activity_searcher: "ActivitySearcher | None" = None,
    annotation_cache: "AnnotationCache | None" = None,
    memory_writer: "MemoryWriter | None" = None,
    db: "WorkerDB | None" = None,
    thumbnail_days: int | None = None,
) -> None:
    """Prune what grows with time: vectors past their retention or from
    another embedding model (the row cap is enforced on every upsert),
    search-index entries of events the
    daemon deleted, unused annotation-cache rows, Memory thumbnails older
    than ``[memory] thumbnail_days`` (with *db* and *thumbnail_days*; they
    expire even while Memory is off), then Memory rows past ``[memory]
    retention_days``."""
    from agenthandover_worker.thumbnails import expire_thumbnails

    expire = (
        db is not None and thumbnail_days is not None
        and (lambda: expire_thumbnails(db, days=thumbnail_days))
    )
    for name, job in (
        ("vector purge", vector_kb and vector_kb.purge_stale),
        ("search index prune", activity_searcher and activity_searcher.prune_index),
        ("annotation cache prune", annotation_cache and annotation_cache.prune),
        ("thumbnail expiry", expire),
        ("memory purge", memory_writer and memory_writer.purge),
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
    keep: Collection[str] = frozenset(),
) -> int:
    """Delete JPEGs in *screenshots_dir* that no pending event refers to.

    Run at worker start.  A frame's JPEG is needed only until its event is
    annotated (or blocked by a privacy zone) and, with Memory on, embedded
    (*keep*: the resolved paths Memory still needs); after that every path
    deletes it.  Anything else in the directory is an orphan: frames of
    blocked or metadata-only windows kept by an older worker, frames of
    events that were purged, files left by a crash.  Files younger than
    *min_age_seconds* are left alone.  Returns the number deleted.
    """
    if not screenshots_dir.is_dir():
        return 0
    referenced = {
        str(Path(p).resolve()) for p in db.pending_screenshot_paths()
    } | set(keep)
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


def _poll_log_level(unprocessed: int, last_logged: int | None) -> int:
    """INFO for a poll whose unprocessed-event count changed since the last
    one logged, DEBUG for a repeat: the same line at INFO every 2 s is
    ~43k lines a day."""
    return logging.INFO if unprocessed != last_logged else logging.DEBUG


def _status_dir() -> Path:
    """Return the standard data directory for status/PID files."""
    return data_dir()


def _migrate_vlm_config(path: Path | None = None) -> bool:
    """Move config.toml's ``[vlm]`` block off earlier default models
    (``model_profiles.migrate_vlm_file``: backup first, atomic write, only
    when it changes), but only to models Ollama serves: when there is a
    switch to make, ``/api/tags`` at ``[vlm] ollama_host`` is asked, and a
    missing model keeps the current ones (one WARNING naming it; the app
    offers the download).  One INFO line when it changes something; a
    failure is a warning and never stops the worker.  Returns whether the
    file changed."""
    target = config_path() if path is None else path
    if not target.is_file():
        return False
    try:
        text = target.read_text(encoding="utf-8")
        plan, _ = upgrade_plan(text)
        served = served_models(text) if plan is not None else None
        changed, reason, backup = migrate_vlm_file(target, served=served)
        # The switch still waits when the file has a plan after the run.
        waits = plan is not None and not (
            changed and upgrade_plan(target.read_text(encoding="utf-8"))[0] is None
        )
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Could not check the [vlm] models in %s against the tier table: %s", target, exc,
        )
        return False
    level = logging.WARNING if waits else logging.INFO
    if changed:
        logger.log(level, "Migrated [vlm] in %s: %s; backup: %s", target, reason, backup)
    elif waits:
        logger.log(level, "Kept the [vlm] models in %s: %s", target, reason)
    return changed


def _migrate_embedding_config(
    path: Path | None = None, state_dir: Path | None = None,
) -> bool:
    """Remove ``[embedding] image_embeddings`` (the "Visual search" option
    removed in 2.0.0) from config.toml, recording in the status dir's
    ``memory-offer.json`` that it was on
    (``model_profiles.migrate_embedding_file``: backup first, atomic
    write, only when the key is there).  One INFO line when it changes
    something; a failure is a warning and never stops the worker.
    Returns whether the file changed."""
    target = config_path() if path is None else path
    if not target.is_file():
        return False
    try:
        changed, wanted, backup = migrate_embedding_file(
            target, _status_dir() if state_dir is None else state_dir,
        )
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not migrate [embedding] in %s: %s", target, exc)
        return False
    if changed:
        logger.info(
            "Removed [embedding] image_embeddings from %s (it was %s; visual "
            "search is now part of Memory); backup: %s",
            target, "on" if wanted else "off", backup,
        )
    return changed


def _migrate_config() -> None:
    """The upgrade's edits of config.toml, before the worker reads it:
    the model switch (once Ollama serves the models), then the removed
    ``[embedding] image_embeddings`` key."""
    _migrate_vlm_config()
    _migrate_embedding_config()


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
    v2_annotations_today: int = 0,
    v2_diffs_today: int = 0,
    v2_annotation_enabled: bool = False,
    annotation_paused_reason: str | None = None,
    annotation_budget: dict | None = None,
    reasoning_budget: dict | None = None,
    reasoning_paused_reason: str | None = None,
    model_schedule: dict | None = None,
    inference: dict | None = None,
    missing_models: list[str] | None = None,
    embedding_upgrade: dict | None = None,
    skills_enabled: bool = True,
) -> None:
    """Atomically write worker-status.json (tmp + fsync + rename).

    ``skills_enabled`` is ``[skills] enabled``: false is "Memory only",
    where no generative model is used and ``missing_models`` is empty.

    ``embedding_upgrade`` ("Upgrade everything"'s re-embed with
    EmbeddingGemma 2, ``EmbeddingUpgrade.status``) is ``{"state":
    "pending" | "running" | "failed" | "done", "done", "total", "dropped",
    "error"}``, written once the upgrade was asked for.

    ``missing_models`` (written when Ollama answered) lists the configured
    annotation and Skills models it does not serve; the app's menu offers
    their download while ``vlm_available`` is false.

    ``inference`` is the inference thread's queue
    (``InferenceRunner.status``) with ``annotation_backlog``, the passive
    frames waiting for annotation; the daemon lowers its passive capture
    cadence while that backlog is long.

    ``annotation_paused_reason`` says why passive annotation is not running
    (``annotation_budget_exhausted`` for the day,
    ``annotation_hourly_budget_exhausted`` for the rolling hour,
    ``on_battery_user_active``), or is None while it runs.

    ``reasoning_budget`` is the LLM reasoner's and behavioral synthesizer's
    daily budget (``ReasoningBudget.status``); ``reasoning_paused_reason``
    is ``reasoning_budget_exhausted`` while it is spent, None otherwise.
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
    }
    if missing_models is not None:
        status["missing_models"] = missing_models
    if v2_annotation_enabled:
        status["v2_annotation_enabled"] = True
        status["v2_annotations_today"] = v2_annotations_today
        status["v2_diffs_today"] = v2_diffs_today
        status["annotation_paused_reason"] = annotation_paused_reason
        if annotation_budget is not None:
            status["annotation_budget"] = annotation_budget
    if reasoning_budget is not None:
        status["reasoning_budget"] = reasoning_budget
        status["reasoning_paused_reason"] = reasoning_paused_reason
    if model_schedule is not None:
        # Which model writes Skills, and why generation waits (memory, a
        # failed load), on a tier with two models.
        status["model_schedule"] = model_schedule
    if inference is not None:
        status["inference"] = inference
    if embedding_upgrade is not None:
        status["embedding_upgrade"] = embedding_upgrade
    status["skills_enabled"] = skills_enabled
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
    _embed_in_vector_kb(vector_kb, "procedure", slug, _procedure_text(procedure))


def _procedure_text(procedure: dict) -> str:
    """A procedure's text for semantic search: title, description,
    strategy and step actions, one per line."""
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
    return "\n".join(parts)


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
) -> dict:
    """Build, prune and translate a batch of events (steps D->E).

    No SOP comes out of this path: the v1 PrefixSpan miner that used to run
    here on every poll is gone (WP-F5), and passive recurrence belongs to
    ``workflow_mining``.  Nothing is queued either: the VLM fallback jobs
    rejected translations used to queue fed only their own re-scoring.

    Returns a summary dict with counts for monitoring.
    """
    summary: dict = {
        "events_in": len(events),
        "episodes": 0,
        "positive_events": 0,
        "negative_events": 0,
        "translations": 0,
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
            for translations in pool.map(
                translator.translate_batch, positive_episodes_events
            ):
                all_translations.extend(translations)
    elif positive_episodes_events:
        all_translations.extend(translator.translate_batch(positive_episodes_events[0]))

    summary["translations"] = len(all_translations)
    logger.info("Translated %d events", len(all_translations))

    return summary


def _process_focus_sessions(
    db: "WorkerDB",
    *,
    episode_builder: EpisodeBuilder,
    clipboard_linker: ClipboardLinker,
    translator: SemanticTranslator,
    scorer: "ConfidenceScorer",
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
        _close_empty_focus_session(state_dir, signal_path, signal)
        return 0
    (state_dir / FOCUS_RESULT_FILE).unlink(missing_ok=True)

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
    knowledge_base=None,
    vector_kb=None,
    weekly_question_budget: int = 5,
    now: datetime | None = None,
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
      4. Ask about the slots the ledger does not know yet
         (``questions.ask_after_focus``: at most 3, within the weekly
         budget of *weekly_question_budget*, counted at *now*).
      5. If questions exist, pause export and write Q&A files for the app.
      6. If no questions, export immediately.

    Returns the number of SOPs exported.
    """
    state_dir = _status_dir()

    # --- Check for pending focus Q&A answers from a previous cycle ---
    now = now or datetime.now(timezone.utc)
    pending_export = _check_focus_answers(
        state_dir,
        db=db,
        now=now,
        openclaw_writer=openclaw_writer,
        skill_md_writer=skill_md_writer,
        claude_skill_writer=claude_skill_writer,
        index_generator=index_generator,
        sop_auto_approve=sop_auto_approve,
        procedure_writer=procedure_writer,
        kb_export_adapter=kb_export_adapter,
        knowledge_base=knowledge_base,
        vector_kb=vector_kb,
        sop_generator=getattr(focus_processor, "sop_generator", None),
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
        # Observation was off (the daemon was down): closed as nothing
        # recorded once the grace after the stop has passed.
        _close_empty_focus_session(state_dir, signal_path, signal)
        return 0
    # A session with events: the previous session's outcome is old news.
    (state_dir / FOCUS_RESULT_FILE).unlink(missing_ok=True)

    logger.info(
        "Focus v2: processing session '%s' (%s): %d events",
        title, session_id, len(focus_events),
    )

    # Run through v2 focus processor (lint included: an SOP that fails
    # validation is regenerated once, then recorded as a failure).  A
    # session whose SOP was generated before a restart resumes from its
    # saved draft instead of generating it again.
    #
    # A recording of a routine that already is a Skill (matched by its
    # workflow, ``workflow_mining.focus_skill``, once its keyframes are
    # annotated) is another run of that Skill, not a new one: no SOP is
    # generated for it, its steps stay as they are, whatever its lifecycle
    # (the run's own way goes to branch hypotheses, and the branches
    # agreeing runs make, below), and the run joins its evidence, what the
    # runs teach and its synthesis.
    def match_skill() -> dict | None:
        if knowledge_base is None:
            return None
        try:
            return workflow_mining.focus_skill(
                db, knowledge_base, db.get_focus_session_events(session_id),
                title, now=now,
            )
        except Exception:
            logger.warning("Focus session '%s' not matched to the Skills", title, exc_info=True)
            return None

    saved = _read_focus_draft(state_dir, session_id)
    result, lint_error = _generate_linted(
        lambda: focus_processor.process_session(
            db, session_id, title, focus_events,
            screenshots_dir=screenshots_dir, saved=saved, match=match_skill,
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
    skill = result.instance_of
    if result.success and (skill is not None or result.sop) and not lint_error:
        if skill is not None:
            procedure_dict = skill
            slug = str(skill.get("id") or skill.get("slug"))
            sop_templates = [procedure_dict]
            logger.info("Focus v2 session '%s' is another run of Skill '%s'", title, slug)
        else:
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
            # A recording the model named as an existing Skill is a new
            # draft of it, but keeps its evidence, trust, lifecycle
            # history and agent-run record.
            if knowledge_base is not None:
                from agenthandover_worker.procedure_writer import merge_existing_procedure
                existing_proc = knowledge_base.get_procedure(slug)
                if existing_proc is not None:
                    merge_existing_procedure(
                        procedure_dict, existing_proc, keep_lifecycle=False,
                    )
        # This recording is one instance of the procedure: the ledger's
        # evidence for what was learned from it, with what its surrounding
        # events showed (lead-in, commit, checks, corrections).
        append_observation(procedure_dict, ObservationEvidence(
            date=str(focus_events[0].get("timestamp") or "")[:10], type="focus",
            duration_minutes=0, session_id=session_id, event_count=len(focus_events),
            signals=run_signals.run_signals_from_db(db, focus_events),
        ), once_per_session=True)
        # Save the draft now: synthesis and Q&A only enrich it, and a
        # restart before they finish must not lose the generated SOP.
        _save_focus_draft(
            db, state_dir, session_id=session_id, title=title,
            result=result, procedure=procedure_dict, new_skill=skill is None,
        )
        # The recording (its frames as annotated now) is also an
        # observation of its Skill, so passive sessions of the same
        # workflow are attributed to it; another run of a Skill is
        # divergence-mined against its steps.
        try:
            workflow_mining.record_focus_observation(
                db, db.get_focus_session_events(session_id), slug, title,
                instance_of=skill, session_id=session_id,
            )
        except Exception:
            logger.warning("Focus observation for '%s' not recorded", slug, exc_info=True)
        if behavioral_synthesizer is not None:
            try:
                # Prefer the rich per-frame observations stashed by the
                # focus_processor (email_addresses, typed_text, urls,
                # visible_items and the frames' screen text) so the
                # synthesizer can quote verbatim text in goal/strategy.
                # Fall back to abstract SOP steps only if rich data is
                # unavailable.
                rich_obs = getattr(focus_processor, "_last_rich_observations", None)
                instance_ids: list[str] = []
                if rich_obs and rich_obs[0]:
                    # Earlier recordings of the same procedure join this
                    # one, so selection criteria contrast every run.
                    earlier = [
                        (obs, iid) for obs, iid in zip(*recorded_observations(db, procedure_dict))
                        if iid and iid != session_id
                    ]
                    obs_for_synthesis = [obs for obs, _ in earlier] + rich_obs
                    instance_ids = [iid for _, iid in earlier] + [session_id]
                else:
                    obs_for_synthesis = [(result.sop or procedure_dict).get("steps", [])]
                # Each run's clicks on list rows, from the browser's
                # reports: what the selection criteria contrast.
                instance_lists = {
                    iid: session_list_choices(db, db.get_focus_session_events(iid))
                    for iid in instance_ids
                }
                insights = behavioral_synthesizer.synthesize(
                    slug, procedure_dict, obs_for_synthesis,
                    force=True,  # bypass min_observations for focus
                    instance_ids=instance_ids,
                    instance_lists=instance_lists,
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

        # --- Ask about what the ledger does not know yet (no model) ---
        try:
            from agenthandover_worker.focus_questioner import ask_focus_questions
            from agenthandover_worker.questions import ask_after_focus

            questions = ask_after_focus(
                db, dict(procedure_dict, id=procedure_dict.get("id") or slug),
                now=now, weekly_budget=weekly_question_budget,
            )
            if questions:
                # Queued per session: an earlier session's questions still
                # in the app are shown first, these once those are done.
                shown = ask_focus_questions(
                    state_dir, session_id, slug, questions, title=title,
                    sop_template=result.sop or procedure_dict,
                    procedure=procedure_dict,
                )
                logger.info(
                    "Focus v2 session '%s': %d question(s) asked (%s), "
                    "pausing export for user answers",
                    title, len(questions),
                    "shown" if shown else "queued behind another session's",
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

        # --- Standard export path (no questions) ---

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
    # did not need are annotated by a later tick's focus-tail job
    # (``annotate_remaining``).
    _clear_focus_signal(signal_path)
    (state_dir / FOCUS_DRAFT_FILE).unlink(missing_ok=True)
    return exported


# The last focus session's generated SOP until the session is exported or
# waits for answers (its ``focus-pending/<session_id>.json`` then holds it).
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
    new_skill: bool = True,
) -> None:
    """Persist a focus session's SOP as soon as it is generated: a draft
    record in the review queue (updated, not duplicated, when the session
    resumes after a restart) and the state ``_read_focus_draft`` resumes
    from.  Synthesis, Q&A and approval update the record afterwards.  A
    session that is another run of an existing Skill (not *new_skill*)
    had no SOP generated: it adds no draft and no resume state (a restart
    matches it again), and the Skill's own record, if any, is updated."""
    slug = procedure.get("slug") or procedure["id"]
    if new_skill:
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
    if record is not None and (record.get("source_id") == session_id or not new_skill):
        db.update_sop_json(slug, procedure)
        return
    if not new_skill:
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
    if _focus_answers_ready(state_dir):
        return "answers"
    signal = _stopped_focus_signal(state_dir)
    session_id = signal.get("session_id") if signal else None
    if session_id and db.get_focus_session_events(session_id):
        return "session"
    return None


def _focus_answers_ready(state_dir: Path) -> bool:
    """True when the post-recording questions were answered or skipped."""
    from agenthandover_worker.focus_questioner import read_focus_questions

    questions = read_focus_questions(state_dir)
    return isinstance(questions, dict) and questions.get("status") in ("answered", "skipped")


def _restore_focus_queue(state_dir: Path) -> None:
    """Startup: move a 0.4.x/0.5.0 ``focus-pending.json`` into the
    per-session queue and show the oldest queued questions when the app
    has none (``focus_questioner.restore_focus_queue``).  A failure is
    logged: it never stops the worker."""
    from agenthandover_worker.focus_questioner import restore_focus_queue

    try:
        restore_focus_queue(state_dir)
    except Exception:
        logger.warning("Could not restore the focus Q&A queue", exc_info=True)


def _stopped_focus_signal(state_dir: Path) -> dict | None:
    """``focus-session.json`` when it says the recording stopped."""
    try:
        signal = json.loads((state_dir / "focus-session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(signal, dict) or signal.get("status") != "stopped":
        return None
    return signal


def _focus_job_due(state_dir: Path) -> bool:
    """True when the focus job has something to do: answered questions, or
    a stopped recording (with events to process, or none, which closes it
    as nothing recorded)."""
    return _focus_answers_ready(state_dir) or _stopped_focus_signal(state_dir) is not None


def _check_focus_answers(
    state_dir: Path,
    *,
    db: "WorkerDB | None" = None,
    now: datetime | None = None,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
    knowledge_base=None,
    vector_kb=None,
    sop_generator: "SOPGenerator | None" = None,
) -> int | None:
    """Check for pending focus Q&A answers and complete export if ready.

    The typed answers revise the steps they concern through one call on
    *sop_generator* (the SOP model, in the generation phase: focus answers
    make the phase urgent), see :func:`_revise_steps_from_answers`.

    Returns:
        Number of SOPs exported if answers were processed.
        None if no pending Q&A was found (caller should proceed normally).
    """
    from agenthandover_worker.focus_questioner import (
        clear_focus_qa_files,
        merge_answers,
        parse_qa_result_from_file,
        publish_next_focus_questions,
        read_focus_pending,
        read_focus_questions,
    )
    from agenthandover_worker.questions import close_focus_questions

    questions_data = read_focus_questions(state_dir)
    if not isinstance(questions_data, dict):
        return None

    status = questions_data.get("status", "")
    if status not in ("answered", "skipped"):
        # Still pending — user hasn't answered yet
        return None

    # The pending SOP state of the session these answers are for (each
    # session's is queued under its id).
    session_id = str(questions_data.get("session_id") or "")
    pending = read_focus_pending(state_dir, session_id)
    if pending is None:
        logger.warning(
            "Focus Q&A answers found but no pending SOP state for session %r "
            "— clearing the stale questions file",
            session_id,
        )
        clear_focus_qa_files(state_dir, session_id)
        publish_next_focus_questions(state_dir)
        return None

    slug = pending.get("slug", "")
    sop_template = pending.get("sop_template", {})
    procedure_dict = pending.get("procedure", {})
    title = sop_template.get("title", procedure_dict.get("title", "Untitled"))

    # Parse and merge answers.  A skipped Q&A still merges: every
    # question is recorded as unanswered rather than dropped.
    qa_result = parse_qa_result_from_file(questions_data)
    # What the user actually did, not the file's status: "answered" is
    # written by the app's Submit even when every field was cleared.
    qa_status = status
    if qa_result is not None and qa_result.questions:
        counts = qa_result.counts()
        qa_status = (
            f"{status if status == 'skipped' else qa_result.status()}: "
            f"{counts['typed']} typed, {counts['accepted_guess']} accepted guess(es), "
            f"{counts['unanswered']} unanswered"
        )
    if db is not None:
        # The micro-questions asked in the window are closed: answered or
        # skipped (a skipped one leaves its slot as it was).
        close_focus_questions(db, questions_data, now=now or datetime.now(timezone.utc))
    if qa_result is not None and qa_result.questions:
        try:
            procedure_dict = merge_answers(procedure_dict, qa_result)
            logger.info(
                "Focus v2 '%s': merged the answers into the procedure (%s)",
                title, qa_status,
            )
        except Exception:
            logger.debug(
                "Failed to merge focus Q&A answers for '%s'",
                title, exc_info=True,
            )
        if sop_generator is not None:
            _revise_steps_from_answers(procedure_dict, qa_result, sop_generator, title)

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

    # Clean up this session's Q&A files, then show the next queued
    # session's questions.
    clear_focus_qa_files(state_dir, session_id)
    publish_next_focus_questions(state_dir)

    # Refresh the Claude Code Skills with the new/updated procedure
    _refresh_claude_commands()

    logger.info(
        "Focus v2 '%s': Q&A complete, exported %d SOP(s) (status=%s)",
        title, exported, qa_status,
    )
    return exported


def _revise_steps_from_answers(
    procedure: dict, qa_result: "FocusQAResult", sop_generator: "SOPGenerator", title: str,
) -> list[dict]:
    """Apply the typed answers of a focus session to the steps they
    concern (in place): one ``SOPGenerator.revise`` call for all of them
    (none when no typed answer may change a step), validated and applied
    by ``focus_questioner.apply_answer_revisions``.  Accepted guesses
    change nothing.  When the model is unavailable or its answer is
    unusable, the steps stay as recorded: the answers are still in the
    Skill's clarifications.  Returns the applied revisions."""
    from agenthandover_worker.focus_questioner import (
        answer_revision_prompt,
        apply_answer_revisions,
    )

    prompt = answer_revision_prompt(procedure, qa_result)
    if prompt is None:
        return []
    try:
        response = sop_generator.revise(prompt)
    except ConnectionError as exc:
        logger.warning(
            "Focus v2 '%s': the typed answers could not revise the steps "
            "(model unavailable: %s); they stay in the clarifications", title, exc,
        )
        return []
    except (ValueError, TimeoutError, RuntimeError) as exc:
        logger.warning(
            "Focus v2 '%s': the answer revision was unusable (%s); steps unchanged",
            title, exc,
        )
        return []
    return apply_answer_revisions(procedure, qa_result, response)


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
    weekly_question_budget: int = 5,
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

    Right after the sessions are recorded, and again after new drafts,
    the micro-questions they raise are asked as review cards
    (:func:`_ask_micro_questions`, within *weekly_question_budget*).
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
    # The segmenter's frame vectors, reused by the variant alignment below.
    frame_embeddings: dict[str, list[float]] = {}
    for seg in seg_result.segments:
        event_ids = [f.event_id for f in seg.frames]
        labels.update(dict.fromkeys(event_ids, seg.task_label))
        frame_embeddings.update((f.event_id, f.embedding) for f in seg.frames if f.embedding)
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
    _ask_micro_questions(db, knowledge_base, now=now, weekly_budget=weekly_question_budget)

    # Step 4: the workflows the review queue admits now.
    import json as _json

    candidates = workflow_mining.review_candidates(db, now=now, kb=knowledge_base)
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
        demo_event_ids: list[list[str]] = []  # each timeline frame's event id
        for obs in candidate.observations:
            event_ids = _json.loads(obs["event_ids_json"])
            events_for_seg = db.get_events_by_ids(event_ids)
            timeline: list[dict] = []
            timeline_ids: list[str] = []
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
                timeline_ids.append(ev.get("id", ""))
            if timeline:
                demonstrations.append(timeline)
                demo_event_ids.append(timeline_ids)

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

        # --- Alignment (once per workflow), variants, parameters ---
        _variant_results: dict = {}
        if variant_detector is not None and len(demonstrations) >= 2:
            try:
                _variant_results = variant_detector.analyse(
                    task_label or str(cluster_id), demonstrations,
                    [[frame_embeddings.get(eid) for eid in ids] for ids in demo_event_ids],
                )
                if _variant_results:
                    logger.info(
                        "Variant detection for '%s': %d variants, %d params",
                        task_label[:60],
                        len(_variant_results["variants"]),
                        len(_variant_results["parameters"]),
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
        # Its steps' anchors: the elements its runs clicked (no model).
        try:
            workflow_mining.anchor_draft_steps(db, result.sop, candidate.observations)
        except Exception:
            logger.warning("Anchoring the passive steps of '%s' failed", task_label[:60], exc_info=True)
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

    if total_exported:
        _ask_micro_questions(db, knowledge_base, now=now, weekly_budget=weekly_question_budget)
    return total_exported


def _ask_micro_questions(
    db: "WorkerDB", knowledge_base: "KnowledgeBase", *, now: datetime, weekly_budget: int,
) -> None:
    """Ask the micro-questions the latest evidence raised, as review cards
    (``questions.refresh``); model-free, so a failure only loses the cards."""
    from agenthandover_worker import questions

    try:
        questions.refresh(db, knowledge_base, _status_dir(), now=now, weekly_budget=weekly_budget)
    except Exception:
        logger.warning("Micro-questions: refresh failed", exc_info=True)


def _process_micro_answer_trigger(
    trigger: dict, db: "WorkerDB", knowledge_base: "KnowledgeBase",
    *, now: datetime | None = None, sop_generator: "SOPGenerator | None" = None,
) -> None:
    """Handle a ``micro-answer`` trigger: a review card answered (or
    skipped) in the app (``questions.apply_answer``), at *now* (default:
    the wall clock).  A free-text answer to a step question is revised by
    *sop_generator* on the inference thread."""
    from agenthandover_worker import questions

    questions.apply_answer(
        trigger, db, knowledge_base, _status_dir(), now=now or datetime.now(timezone.utc),
        reviser=sop_generator,
    )


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
                        try:
                            workflow_mining.anchor_draft_steps(db, result.sop, pending_segs)
                        except Exception:
                            logger.warning(
                                "Anchoring the passive steps of '%s' failed", title, exc_info=True,
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
    # The embedding is a model call: queued for the inference thread when
    # the approval runs on the control thread.
    model_work(
        f"embed-procedure:{sop_id}",
        lambda: _embed_procedure(vector_kb, sop_template),
    )
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


# How a stopped focus session ended when it ended without a Skill the user
# can wait for, for the app's menu (``focus-result.json``, next to the
# signal): ``session_id``, ``title``, ``status``, ``message`` (what to
# show), ``stopped_at`` and ``closed_at`` (ISO 8601).  Removed when the
# next session with events is processed.
FOCUS_RESULT_FILE = "focus-result.json"
FOCUS_NOTHING_RECORDED = "nothing_recorded"
# A stopped session with no events yet is waited for this long after the
# stop (the daemon's last frames may still be on their way), then closed.
EMPTY_SESSION_GRACE_SECONDS = 120.0
# Sessions whose "no events yet" wait was logged (once each, not every tick).
_EMPTY_SESSIONS_LOGGED: set[str] = set()


def _focus_stop_time(signal: dict, signal_path: Path) -> float | None:
    """When the session was stopped (epoch seconds): the signal's
    ``stopped_at`` when it has one, else when the signal file was last
    written (the stop rewrote it)."""
    from agenthandover_worker.event_helpers import parse_timestamp

    stopped = parse_timestamp(signal.get("stopped_at"))
    if stopped is not None:
        return stopped.timestamp()
    try:
        return signal_path.stat().st_mtime
    except OSError:
        return None


def _close_empty_focus_session(
    state_dir: Path, signal_path: Path, signal: dict, *, now: float | None = None,
) -> bool:
    """A stopped focus session with no events: wait
    ``EMPTY_SESSION_GRACE_SECONDS`` after the stop, then close it: write
    ``focus-result.json`` (``nothing_recorded``, observation was off) and
    remove the signal, so the app stops showing "Analyzing" and says what
    happened.  The wait is logged once.  Returns True when closed."""
    session_id = str(signal.get("session_id") or "")
    title = str(signal.get("title") or "")
    now = time.time() if now is None else now
    stopped = _focus_stop_time(signal, signal_path)
    waited = now - stopped if stopped is not None else EMPTY_SESSION_GRACE_SECONDS
    if waited < EMPTY_SESSION_GRACE_SECONDS:
        if session_id not in _EMPTY_SESSIONS_LOGGED:
            _EMPTY_SESSIONS_LOGGED.add(session_id)
            logger.info(
                "Focus session '%s' (%s) has no events yet; waiting up to %.0f s "
                "after its stop", title, session_id, EMPTY_SESSION_GRACE_SECONDS,
            )
        return False
    _EMPTY_SESSIONS_LOGGED.discard(session_id)

    def iso(seconds: float) -> str:
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    try:
        _atomic_write_result(state_dir / FOCUS_RESULT_FILE, {
            "session_id": session_id,
            "title": title,
            "status": FOCUS_NOTHING_RECORDED,
            "message": "Nothing was recorded: observation was off during this session",
            "stopped_at": iso(stopped) if stopped is not None else None,
            "closed_at": iso(now),
        })
    except OSError:
        logger.warning("Could not write %s", FOCUS_RESULT_FILE, exc_info=True)
    logger.warning(
        "Focus session '%s' (%s) recorded no events (%.0f s after its stop): "
        "observation was off; closed as nothing recorded",
        title, session_id, waited,
    )
    _clear_focus_signal(signal_path)
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


# The CLI the pkg installs; a source checkout's on PATH otherwise.
_INSTALLED_CLI = "/usr/local/bin/agenthandover"


def _migrate_with_cli(db_path: Path, cli: str | None = None) -> None:
    """Bring *db_path* to the daemon's current schema with the daemon's own
    migrations: ``agenthandover db migrate --path`` (the worker never
    changes the schema itself).  Logs the outcome; never raises, as the
    schema check that follows says what is still wrong."""
    import shutil
    import subprocess

    cli = cli or (_INSTALLED_CLI if os.access(_INSTALLED_CLI, os.X_OK) else shutil.which("agenthandover"))
    if cli is None:
        logger.warning("The database is older than this worker and no agenthandover CLI was found to migrate it")
        return
    try:
        done = subprocess.run(
            [cli, "db", "migrate", "--path", str(db_path)],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("agenthandover db migrate could not run: %s", exc)
        return
    if done.returncode == 0:
        logger.info("%s", done.stdout.strip())
    else:
        logger.warning("agenthandover db migrate failed (%d): %s", done.returncode, done.stderr.strip())


def _schema_ready(db_path: Path, shutdown_flag: list[bool]) -> bool:
    """The daemon owns the schema: run only against the version this
    worker is written for.  Right after an upgrade, wait (as long as for
    the database itself) for the daemon to migrate; False, with the
    reason and the fix in the log and on stderr, when the database stays
    older or is newer."""
    from agenthandover_worker.db import SchemaVersionError, wait_for_schema_version

    try:
        wait_for_schema_version(
            db_path,
            timeout_s=_DB_RETRY_MAX_SECONDS,
            poll_s=_DB_RETRY_POLL_SECONDS,
            stop=lambda: shutdown_flag[0],
            migrate=lambda: _migrate_with_cli(db_path),
        )
    except SchemaVersionError as exc:
        logger.error("%s", exc)
        print(f"agenthandover-worker: {exc}", file=sys.stderr)
        return False
    return True


def _setup_logging(args: argparse.Namespace) -> None:
    """The rotating ``worker.log`` (plus stderr in a terminal)."""
    if _platform.system() == "Darwin":
        log_dir = Path.home() / "Library" / "Application Support" / "agenthandover" / "logs"
    else:
        log_dir = Path.home() / ".local" / "share" / "agenthandover" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=_log_handlers(log_dir, stderr_is_tty=sys.stderr.isatty()),
    )


def _install_signal_handlers(shutdown_flag: list[bool]) -> Callable[[], None]:
    """SIGINT and SIGTERM request a shutdown; returns the function that
    puts the previous handlers back."""

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received %s — requesting shutdown", signal.Signals(signum).name)
        shutdown_flag[0] = True

    previous = {
        signum: signal.signal(signum, _handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def restore() -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    return restore


def _llm_overrides(args: argparse.Namespace) -> dict:
    """The command line's ``[llm]`` overrides."""
    return {
        key: value
        for key, value in (
            ("enhance_sops", args.enhance_sops),
            ("model", args.llm_model),
            ("max_enhancements_per_day", args.max_enhancements_per_day),
        )
        if value is not None
    }


def _log_model_profiles(vlm_cfg) -> None:
    """Warn when a Gemma 4 model meets an Ollama too old for it, and log the
    active model profiles."""
    try:
        from agenthandover_worker.model_profiles import (
            MINIMUM_OLLAMA_VERSION_FOR_GEMMA4, get_profile, ollama_supports_gemma4,
        )

        ann_model = vlm_cfg.annotation_model
        sop_model = vlm_cfg.sop_model
        uses_gemma = "gemma4" in ann_model or "gemma4" in sop_model
        if uses_gemma and not ollama_supports_gemma4(vlm_cfg.ollama_host):
            logger.warning(
                "Gemma 4 QAT models require Ollama %s+. "
                "Current version is too old (pulls will fail with HTTP 412). "
                "Update the official app: https://ollama.com/download "
                "(or: brew install --cask ollama-app). "
                "Note: the Homebrew 'ollama' formula does not bundle the "
                "GGUF runner — use the official app instead.",
                MINIMUM_OLLAMA_VERSION_FOR_GEMMA4,
            )
        logger.info(
            "Model profiles: annotation=%s (think=%s), SOP=%s (think=%s)",
            ann_model, get_profile(ann_model).ann_think,
            sop_model, get_profile(sop_model).sop_think,
        )
    except Exception:
        logger.debug("Model profile detection failed", exc_info=True)


def _vlm_backend_available(vlm_cfg) -> bool:
    """``vlm_available`` in ``worker-status.json``: Ollama answers at
    ``[vlm] ollama_host`` and has the annotation model pulled."""
    from agenthandover_worker.ollama import OllamaClient

    available = OllamaClient(vlm_cfg.ollama_host).has_model(vlm_cfg.annotation_model)
    if available:
        logger.info("VLM available: %s at %s", vlm_cfg.annotation_model, vlm_cfg.ollama_host)
    else:
        logger.info(
            "VLM not available: %s is not served at %s (agenthandover setup --vlm pulls it)",
            vlm_cfg.annotation_model, vlm_cfg.ollama_host,
        )
    return available


def _generative_models(cfg) -> tuple[bool, list[str] | None]:
    """``(vlm_available, missing_models)`` for the worker status.  Memory
    only uses no generative model, so none is available or missing."""
    if not cfg.skills.enabled:
        return False, []
    return _vlm_backend_available(cfg.vlm), _missing_vlm_models(cfg.vlm)


def _missing_vlm_models(vlm_cfg) -> list[str] | None:
    """``missing_models`` in ``worker-status.json``: the annotation and
    Skills models Ollama does not serve at ``[vlm] ollama_host``; None
    when it does not answer."""
    from agenthandover_worker.model_profiles import missing_models
    from agenthandover_worker.ollama import OllamaClient

    try:
        served = OllamaClient(vlm_cfg.ollama_host).tags()
    except Exception:
        return None
    return missing_models(dict.fromkeys([vlm_cfg.annotation_model, vlm_cfg.sop_model]), served)


def _commonplace(ctx, memory_service) -> "object":
    """Commonplace over the worker's own stores, for ``POST
    /commonplace/search``: its knowledge base and vector store, Memory
    through *memory_service*'s index, and ``config.toml`` read on each
    search (the agents switch applies at once, as in the MCP server)."""
    from agenthandover_worker.commonplace import QUERY_EMBED_TIMEOUT_S, Commonplace

    vector_kb = ctx.vector_kb
    return Commonplace(
        knowledge_base=ctx.knowledge_base,
        vector_kb=lambda: (vector_kb, "") if vector_kb is not None
        else (None, "the worker has no vector store"),
        load_config=lambda: load_config(ctx.cfg.path),
        memory=lambda _cfg, query, *, keyword, **filters: memory_service.moments(
            query, keyword=keyword, **filters,
        ),
        db_path=lambda: Path(ctx.db_path),
        embed_timeout=QUERY_EMBED_TIMEOUT_S,
        local_tz=ctx.local_tz,
    )


def _start_query_api(ctx) -> "object | None":
    """The localhost query API server, when ``[knowledge]
    query_api_enabled``; None otherwise or when it could not start."""
    try:
        if not ctx.cfg.knowledge.query_api_enabled:
            logger.info("Query API server disabled (knowledge.query_api_enabled = false)")
            return None
        from agenthandover_worker.memory_api import MemoryService
        from agenthandover_worker.query_api import TOKEN_FILE, QueryAPIServer

        port = ctx.cfg.knowledge.query_api_port
        memory = MemoryService(
            ctx.cfg, db_path=ctx.db_path, data_dir=_status_dir(),
            knowledge_dir=ctx.knowledge_dir, writer=ctx.memory_writer,
            vector_kb=ctx.vector_kb, local_tz=ctx.local_tz, now=ctx.now,
        )
        commonplace = _commonplace(ctx, memory)
        server = QueryAPIServer(
            knowledge_base=ctx.knowledge_base,
            port=port,
            activity_searcher=ctx.activity_searcher,
            execution_monitor=ctx.execution_monitor,
            procedure_verifier=ctx.procedure_verifier,
            bundle_compiler=ctx.bundle_compiler,
            procedure_curator=ctx.procedure_curator,
            runtime_validator=ctx.runtime_validator,
            ops_telemetry=ctx.ops_telemetry,
            vector_kb=ctx.vector_kb,
            token_path=_status_dir() / TOKEN_FILE,
            memory=memory,
            commonplace=commonplace,
        )
        server.start()
        logger.info(
            "Query API server started on 127.0.0.1:%d (localhost-only; POST "
            "requires the X-AgentHandover-Token header from %s). "
            "Set knowledge.query_api_enabled = false in config.toml to disable.",
            port, TOKEN_FILE,
        )
        return server
    except Exception:
        logger.debug("Query API server not started", exc_info=True)
        return None


def _startup_embedding_job(knowledge_base, vector_kb) -> None:
    """P2: embed the knowledge base's procedures (idempotent: skips
    unchanged ones)."""
    try:
        embedded = 0
        for proc in knowledge_base.list_procedures():
            full = knowledge_base.get_procedure(proc.get("id", ""))
            if full:
                _embed_procedure(vector_kb, full)
                embedded += 1
        if embedded:
            logger.info("Embedded %d existing procedure(s) in vector KB", embedded)
    except Exception:
        logger.debug("Startup procedure embedding failed", exc_info=True)


def _shutdown(ctx, pid_file: Path, restore_signals: Callable[[], None], *, once: bool) -> None:
    """Stop the inference thread, the query API and the stores; remove the
    PID file and (after a loop, unless --once) the status file."""
    if ctx is not None:
        # A model call in progress cannot be interrupted: the inference
        # thread is a daemon thread and does not hold up the exit.
        if ctx.runner is not None and not ctx.runner.stop():
            logger.info("Inference job still running at shutdown; not waiting for it")
        if ctx.query_api_server is not None:
            try:
                ctx.query_api_server.stop()
                logger.info("Query API server stopped")
            except Exception:
                pass
        ctx.close()
    try:
        pid_file.unlink(missing_ok=True)
    except Exception:
        pass
    # A --once run keeps worker-status.json: it records that iteration,
    # and readers treat it as not running because the PID is gone.
    if ctx is not None and not once:
        _remove_worker_status()
    restore_signals()


def main(argv: list[str] | None = None) -> None:
    """Worker entry-point: wait for the daemon's database, build the
    context (``context.py``), then tick (``stages.tick``) until shutdown;
    model work runs on the inference thread."""
    args = _parse_args(argv)
    if args.backfill_procedures:
        # One-shot maintenance: no PID file or status file, so it is safe
        # to run while the worker service is running.
        from agenthandover_worker.knowledge_base import KnowledgeBase

        logging.basicConfig(level=getattr(logging, args.log_level))
        count = KnowledgeBase(
            root=args.knowledge_dir, db_path=args.db_path,
        ).backfill_procedures()
        print(f"Backfilled {count} procedure(s) in {args.db_path}")
        return

    _setup_logging(args)
    # Before the DB wait, so Ctrl-C while waiting exits cleanly.
    shutdown_flag: list[bool] = [False]
    restore_signals = _install_signal_handlers(shutdown_flag)
    pid_file = _status_dir() / "worker.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    logger.info("PID file written: %s", pid_file)
    logger.info("Starting agenthandover-worker v%s", _WORKER_VERSION)
    logger.info("Database path: %s", args.db_path)
    logger.info("Poll interval: %.1fs", args.poll_interval)
    # An early "starting" status, so the CLI and the app see the worker is
    # alive before the heavy initialization finishes (or fails).
    _write_worker_status(
        started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        events_processed_today=0, sops_generated=0, last_pipeline_duration_ms=None,
        consecutive_errors=0, vlm_available=False, skills_enabled=load_config().skills.enabled,
    )
    if not _wait_for_db(args.db_path, shutdown_flag) or not _schema_ready(
        args.db_path, shutdown_flag,
    ):
        _shutdown(None, pid_file, restore_signals, once=args.once)
        sys.exit(1)
    if shutdown_flag[0]:
        _shutdown(None, pid_file, restore_signals, once=args.once)
        return

    from agenthandover_worker import stages
    from agenthandover_worker.context import build_context, start
    from agenthandover_worker.stages import export

    # The upgrade's edits of config.toml, then one read.
    _migrate_config()
    cfg = load_config()
    if args.adapter is None:
        args.adapter = cfg.export.adapter
    logger.info("Export adapter: %s", args.adapter)
    ctx = build_context(
        cfg, db_path=args.db_path, knowledge_dir=args.knowledge_dir,
        sops_dir=args.sops_dir, adapter=args.adapter,
        poll_interval=args.poll_interval, json_export=args.json_export,
        llm_overrides=_llm_overrides(args),
    )
    try:
        _log_model_profiles(cfg.vlm)
        ctx.vlm_available, ctx.missing_models = _generative_models(cfg)
        ctx.query_api_server = _start_query_api(ctx)
        export.heartbeat(ctx)
        logger.info("Connected to database, entering main loop")
        start(ctx)
        ctx.runner = InferenceRunner(on_idle=ctx.model_scheduler.end_generation).start()
        # The runner's queue is in memory: review-card answers whose step
        # revision a stop cut short are queued again.
        from agenthandover_worker.questions import requeue_step_answers

        with STATE_LOCK:
            requeue_step_answers(ctx.db, ctx.knowledge_base, ctx.sop_generator)
        # Focus Q&As wait in a per-session queue; an upgrade from a single
        # focus-pending.json moves it there.
        _restore_focus_queue(_status_dir())
        if ctx.vector_kb is not None:
            ctx.runner.submit(
                "startup-embedding", P2,
                lambda: _startup_embedding_job(ctx.knowledge_base, ctx.vector_kb),
            )
        while not shutdown_flag[0]:
            tick = stages.tick(
                ctx, lock_timeout=None if args.once else CONTROL_LOCK_WAIT_SECONDS,
            )
            if args.once:
                # One iteration, including the model work it submitted.
                ctx.runner.wait_idle()
                export.heartbeat(ctx)
                break
            if tick.backoff is not None:
                interruptible_sleep(tick.backoff, lambda: shutdown_flag[0])
            else:
                time.sleep(ctx.current_interval)
    finally:
        _shutdown(ctx, pid_file, restore_signals, once=args.once)
    logger.info("Worker shut down cleanly")


if __name__ == "__main__":
    main()

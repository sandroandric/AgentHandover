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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from oc_apprentice_worker.clipboard_linker import ClipboardLinker
from oc_apprentice_worker.confidence import ConfidenceScorer, is_native_app_context
from oc_apprentice_worker.db import WorkerDB
from oc_apprentice_worker.deep_scan import DeepScanner
from oc_apprentice_worker.episode_builder import EpisodeBuilder
from oc_apprentice_worker.exporter import IndexGenerator
from oc_apprentice_worker.focus_processor import FocusProcessor
from oc_apprentice_worker.frame_differ import DiffConfig, FrameDiffer
from oc_apprentice_worker.negative_demo import NegativeDemoPruner
from oc_apprentice_worker.export_adapter import SOPExportAdapter
from oc_apprentice_worker.openclaw_writer import OpenClawWriter
from oc_apprentice_worker.scene_annotator import AnnotationConfig, SceneAnnotator
from oc_apprentice_worker.sop_generator import SOPGenerator, SOPGeneratorConfig
from oc_apprentice_worker.sop_linter import lint_sop
from oc_apprentice_worker.task_segmenter import TaskSegmenter, SegmenterConfig
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
KNOWLEDGE_BASE_DIR = Path.home() / ".openmimic" / "knowledge"
POLL_INTERVAL_SECONDS = 2.0
VLM_REJECT_THRESHOLD = 0.60

_WORKER_VERSION = "0.1.0"
_DB_RETRY_MAX_SECONDS = 120
_DB_RETRY_POLL_SECONDS = 5
_IDLE_LOG_INTERVAL_SECONDS = 300  # 5 minutes


def _lint_and_log(sop_template: dict, label: str) -> bool:
    """Lint an SOP template, log any issues, return True if valid.

    Errors are logged at ERROR level, warnings at WARNING level.
    """
    result = lint_sop(sop_template)
    for issue in result.issues:
        if issue.severity == "error":
            logger.error(
                "SOP lint error [%s] %s [%s]: %s",
                label, issue.field, sop_template.get("slug", "?"), issue.message,
            )
        else:
            logger.warning(
                "SOP lint warning [%s] %s [%s]: %s",
                label, issue.field, sop_template.get("slug", "?"), issue.message,
            )
    if not result.valid:
        logger.error(
            "SOP '%s' failed validation (%d errors), skipping export",
            sop_template.get("title", "?"), len(result.errors),
        )
    return result.valid


def _read_vlm_config_field(field: str, default: str = "") -> str:
    """Read a single field from the [vlm] section of config.toml.

    Falls back to *default* if the field or file is missing.
    """
    import tomllib

    if _platform.system() == "Darwin":
        config_path = (
            Path.home() / "Library" / "Application Support"
            / "oc-apprentice" / "config.toml"
        )
    else:
        config_path = Path.home() / ".config" / "oc-apprentice" / "config.toml"

    if not config_path.is_file():
        return default

    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        return str(cfg.get("vlm", {}).get(field, default))
    except Exception:
        return default


def _read_idle_jobs_config() -> dict:
    """Read [idle_jobs] from config.toml and return kwargs for SchedulerConfig.

    Returns an empty dict if the section is missing or unreadable.
    """
    import tomllib
    from datetime import time as dt_time

    if _platform.system() == "Darwin":
        config_path = (
            Path.home() / "Library" / "Application Support"
            / "oc-apprentice" / "config.toml"
        )
    else:
        config_path = Path.home() / ".config" / "oc-apprentice" / "config.toml"

    if not config_path.is_file():
        return {}

    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        section = cfg.get("idle_jobs", {})
        if not section:
            return {}

        result: dict = {}
        if "require_ac_power" in section:
            result["require_ac_power"] = bool(section["require_ac_power"])
        if "min_battery_percent" in section:
            result["min_battery_percent"] = int(section["min_battery_percent"])
        if "max_cpu_percent" in section:
            result["max_cpu_percent"] = int(section["max_cpu_percent"])
        if "max_temp_c" in section:
            result["max_temp_c"] = int(section["max_temp_c"])
        if "run_window_local_time" in section:
            # Format: "HH:MM-HH:MM"
            window = section["run_window_local_time"]
            parts = window.split("-")
            if len(parts) == 2:
                start_parts = parts[0].strip().split(":")
                end_parts = parts[1].strip().split(":")
                result["run_window_start"] = dt_time(
                    int(start_parts[0]), int(start_parts[1])
                )
                result["run_window_end"] = dt_time(
                    int(end_parts[0]), int(end_parts[1])
                )
        return result
    except Exception:
        logger.debug("Failed to read [idle_jobs] config", exc_info=True)
        return {}


def _read_keychain_api_key(provider: str) -> str:
    """Read an API key from macOS Keychain (set by SwiftUI onboarding).

    The onboarding stores keys with service 'com.openmimic.apprentice'
    and account 'openmimic-{provider}-key'.  We retrieve via the
    ``security`` CLI to avoid a native dependency.

    Returns the key string, or "" if not found / not on macOS.
    """
    import subprocess

    if _platform.system() != "Darwin":
        return ""

    account = f"openmimic-{provider}-key"
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s", "com.openmimic.apprentice",
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


def _read_llm_config() -> dict:
    """Read the entire [llm] section from config.toml.

    Returns a dict with defaults for missing fields.
    """
    import tomllib

    if _platform.system() == "Darwin":
        config_path = (
            Path.home() / "Library" / "Application Support"
            / "oc-apprentice" / "config.toml"
        )
    else:
        config_path = Path.home() / ".config" / "oc-apprentice" / "config.toml"

    defaults = {
        "enhance_sops": True,
        "max_enhancements_per_day": 20,
        "model": "",
        "timeout_seconds": 60,
        "temperature": 0.3,
        "max_tokens": 800,
    }

    if not config_path.is_file():
        return defaults

    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        llm = cfg.get("llm", {})
        for key, default_val in defaults.items():
            if key in llm:
                defaults[key] = llm[key]
        return defaults
    except Exception:
        return defaults


def _read_vlm_v2_config() -> dict:
    """Read v2 scene annotation pipeline config from the [vlm] section.

    Returns a dict with defaults for all v2-specific fields.
    """
    return {
        "annotation_enabled": _read_vlm_config_field(
            "annotation_enabled", "true"
        ).lower() in ("true", "1", "yes"),
        "annotation_model": _read_vlm_config_field(
            "annotation_model", "qwen3.5:2b"
        ),
        "sop_model": _read_vlm_config_field("sop_model", "qwen3.5:4b"),
        "stale_skip_count": int(
            _read_vlm_config_field("stale_skip_count", "3")
        ),
        "sliding_window_max_age_sec": int(
            _read_vlm_config_field("sliding_window_max_age_sec", "600")
        ),
        "ollama_host": _read_vlm_config_field(
            "ollama_host", "http://localhost:11434"
        ),
    }


def _read_sop_config() -> dict:
    """Read the [sop] section from config.toml.

    Returns a dict with defaults for missing fields.
    """
    import tomllib

    if _platform.system() == "Darwin":
        config_path = (
            Path.home() / "Library" / "Application Support"
            / "oc-apprentice" / "config.toml"
        )
    else:
        config_path = Path.home() / ".config" / "oc-apprentice" / "config.toml"

    defaults = {
        "auto_approve": True,
    }

    if not config_path.is_file():
        return defaults

    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        sop = cfg.get("sop", {})
        if "auto_approve" in sop:
            defaults["auto_approve"] = bool(sop["auto_approve"])
        return defaults
    except Exception:
        return defaults


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
) -> dict:
    """Process a batch of unannotated screenshots through the scene annotator.

    Returns stats dict with counts of annotated/skipped/failed events.
    """
    unannotated = db.get_unannotated_events(limit=batch_size)
    if not unannotated:
        return {"annotated": 0, "skipped": 0, "failed": 0, "blocked": 0}

    stats = {"annotated": 0, "skipped": 0, "failed": 0, "blocked": 0}

    for event in unannotated:
        event_id = event.get("id", "unknown")
        timestamp = event.get("timestamp", "")

        # Privacy zone check — skip blocked events, metadata-only for restricted
        if privacy_checker is not None:
            from oc_apprentice_worker.privacy_zones import ObservationTier
            tier = privacy_checker.check_event(event)
            if tier == ObservationTier.BLOCKED:
                db.save_annotation(event_id, "", status="privacy_blocked")
                stats["blocked"] += 1
                logger.debug("Event %s blocked by privacy zone", event_id[:8])
                continue
            if tier == ObservationTier.METADATA_ONLY:
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

        # Get sliding window context (last N annotations within time window)
        recent = db.get_recent_annotations(
            before_timestamp=timestamp,
            limit=annotator.config.sliding_window_size,
            max_age_seconds=annotator.config.sliding_window_max_age_sec,
        )

        # Run annotation
        result = annotator.annotate_event(
            event,
            recent_annotations=recent,
            artifact_dir=str(screenshots_dir),
        )

        # Save result to DB
        if result.status == "completed" and result.annotation:
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
            db.save_annotation(event_id, "", status=result.status)
            stats["skipped"] += 1
            logger.debug(
                "Annotation %s for %s: %s",
                result.status,
                event_id[:8],
                result.error or "",
            )
        else:
            db.save_annotation(event_id, "", status="failed")
            stats["failed"] += 1
            logger.warning(
                "Annotation failed for %s: %s",
                event_id[:8],
                result.error,
            )

    return stats


def _process_diffs(
    db: "WorkerDB",
    differ: "FrameDiffer",
    *,
    batch_size: int = 10,
) -> dict:
    """Process annotated events that need frame-to-frame diffs.

    For each event, finds the previous annotated event and computes
    the diff (either a code-only marker for edge cases or an LLM-based
    action diff).

    Returns stats dict with counts of diffs/edge_cases/failed.
    """
    needs_diff = db.get_events_needing_diff(limit=batch_size)
    if not needs_diff:
        return {"diffs": 0, "edge_cases": 0, "failed": 0}

    stats = {"diffs": 0, "edge_cases": 0, "failed": 0}

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
        result = differ.diff_pair(prev_event, event)

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
        choices=["openclaw", "generic", "skill-md", "claude-skill", "all"],
        default="openclaw",
        help="SOP export adapter (default: openclaw)",
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
    return parser.parse_args(argv)


def _status_dir() -> Path:
    """Return the standard data directory for status/PID files."""
    if _platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "oc-apprentice"
    return Path.home() / ".local" / "share" / "oc-apprentice"


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
    sop_inducer_available: bool,
    vlm_queue_pending: int = 0,
    vlm_jobs_today: int = 0,
    vlm_dropped_today: int = 0,
    vlm_mode: str | None = None,
    vlm_provider: str | None = None,
    v2_annotations_today: int = 0,
    v2_diffs_today: int = 0,
    v2_annotation_enabled: bool = False,
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


def _write_sops_index(db: WorkerDB, *, force: bool = False) -> None:
    """Atomically write sops-index.json for the SwiftUI app to read.

    Throttled to at most once every 5 seconds unless *force* is True.
    """
    global _last_sops_index_write
    now = time.time()
    if not force and (now - _last_sops_index_write) < 5.0:
        return

    try:
        all_sops = db.get_generated_sops()
        failed = db.get_failed_generations()

        draft_count = sum(1 for s in all_sops if s.get("status") == "draft")
        approved_count = sum(1 for s in all_sops if s.get("status") == "approved")

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

            sop_entries.append({
                "sop_id": s.get("sop_id", ""),
                "slug": s.get("slug", ""),
                "title": title,
                "short_title": short_title,
                "tags": tags,
                "source": s.get("source", ""),
                "status": s.get("status", "draft"),
                "confidence": s.get("confidence", 0.0) or 0.0,
                "created_at": s.get("created_at", ""),
                "reviewed_at": s.get("reviewed_at"),
            })

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
    sop_enhancer: object | None = None,
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    db: "WorkerDB | None" = None,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
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
        "skills_exported": 0,
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

    # Step E2: Persist translated episode steps to the episode store
    # so future pipeline cycles can see them during SOP induction.
    if db is not None and episode_sop_steps:
        for i, ep_steps in enumerate(episode_sop_steps):
            ep = episodes[i] if i < len(episodes) else None
            ep_id = ep.episode_id if ep else str(uuid.uuid4())
            thread_id = ep.thread_id if ep else "unknown"
            db.save_episode_steps(ep_id, thread_id, ep_steps)
        logger.debug(
            "Persisted %d episode(s) to episode store", len(episode_sop_steps)
        )

    # Step F: Induce SOPs from ALL historical episodes (not just current batch)
    # Load stored episodes + current batch for cross-cycle pattern mining.
    sop_templates: list[dict] = []
    if sop_inducer is not None:
        all_episodes_for_mining: list[list[dict]] = []
        if db is not None:
            all_episodes_for_mining = db.get_all_episode_steps(max_age_days=14)
            logger.info(
                "Loaded %d historical episode(s) from store for SOP mining",
                len(all_episodes_for_mining),
            )
        else:
            # Fallback: only use current batch (original behaviour)
            all_episodes_for_mining = episode_sop_steps

        if all_episodes_for_mining:
            try:
                sop_templates = sop_inducer.induce(all_episodes_for_mining)
                summary["sops_induced"] = len(sop_templates)
                logger.info("Induced %d SOP templates", len(sop_templates))
            except Exception:
                logger.exception("SOP induction failed")

    # Step F1.5: Enhance SOPs with LLM-generated descriptions
    if sop_enhancer is not None and sop_templates:
        enhanced_count = 0
        for template in sop_templates:
            try:
                enhanced = sop_enhancer.enhance_sop(template)
                if "task_description" in enhanced:
                    template.update(enhanced)
                    enhanced_count += 1
            except Exception:
                logger.debug(
                    "SOP enhancement failed for '%s'",
                    template.get("slug", "unknown"),
                    exc_info=True,
                )
        if enhanced_count:
            logger.info("Enhanced %d/%d SOPs with LLM descriptions",
                        enhanced_count, len(sop_templates))

    # Step F2: Format, version, and export SOPs
    if sop_templates:
        # Deduplicate against known SOPs before writing
        from oc_apprentice_worker.sop_dedup import deduplicate_templates
        sop_templates = deduplicate_templates(sop_templates, _status_dir())

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

        # Cache SOP templates for CLI export trigger re-export
        _save_sop_cache(sop_templates)

        # Step F3: Also export as SKILL.md if skill_md_writer is configured
        if skill_md_writer is not None:
            try:
                skill_paths = skill_md_writer.write_all_sops(sop_templates)
                summary["skills_exported"] = len(skill_paths)
                logger.info("Exported %d SKILL.md files", len(skill_paths))
            except Exception:
                logger.exception("SKILL.md export failed")

        # Step F4: Also export as Claude Code skills if configured
        if claude_skill_writer is not None:
            try:
                cs_paths = claude_skill_writer.write_all_sops(sop_templates)
                summary["claude_skills_exported"] = len(cs_paths)
                logger.info("Exported %d Claude Code skill(s)", len(cs_paths))
            except Exception:
                logger.exception("Claude Code skill export failed")

        # Step F5: Write v3 procedures to the knowledge base
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
        if kb_export_adapter is not None:
            try:
                kb_export_adapter.write_all_sops(sop_templates)
            except Exception:
                logger.debug("KB export adapter failed", exc_info=True)

    return summary


def _process_focus_sessions(
    db: "WorkerDB",
    *,
    episode_builder: EpisodeBuilder,
    clipboard_linker: ClipboardLinker,
    translator: SemanticTranslator,
    scorer: "ConfidenceScorer",
    vlm_queue: "VLMFallbackQueue",
    sop_inducer: object | None,
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
    ``induce_from_focus_session()`` (bypassing PrefixSpan multi-episode
    requirement) to produce a SOP from a single demonstration.

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

    # Induce SOP from focus session (bypasses PrefixSpan)
    sop_templates: list[dict] = []
    if sop_inducer is not None:
        try:
            sop_templates = sop_inducer.induce_from_focus_session(
                episode_sop_steps, title
            )
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
            from oc_apprentice_worker.sop_dedup import deduplicate_templates
            sop_templates = deduplicate_templates(sop_templates, _status_dir())

            try:
                paths = openclaw_writer.write_all_sops(sop_templates)
                exported = len(paths)
                index_generator.update_index(
                    openclaw_writer.get_sops_dir(), sop_templates
                )
                logger.info(
                    "Focus session '%s': exported %d SOP(s)", title, exported
                )
            except Exception:
                logger.exception("Focus session SOP export failed")

            # Also export as SKILL.md if writer is configured
            if skill_md_writer is not None:
                try:
                    skill_md_writer.write_all_sops(sop_templates)
                    logger.info(
                        "Focus session '%s': SKILL.md export complete", title
                    )
                except Exception:
                    logger.exception("Focus session SKILL.md export failed")

            # Also export as Claude Code skills if writer is configured
            if claude_skill_writer is not None:
                try:
                    claude_skill_writer.write_all_sops(sop_templates)
                    logger.info(
                        "Focus session '%s': Claude skill export complete", title
                    )
                except Exception:
                    logger.exception("Focus session Claude skill export failed")

            # Write v3 procedures to the knowledge base
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
    elif not sop_templates and sop_inducer is not None:
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
) -> int:
    """Process completed focus recording sessions via v2 VLM pipeline.

    Uses the scene annotator + frame differ + SOP generator instead of
    the v1 episode builder + translator + PrefixSpan path.  Produces
    semantic SOPs with exact screen-observed details.

    Returns the number of SOPs exported.
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
        logger.info(
            "Focus v2 session '%s' (%s) has no events yet, will retry",
            title, session_id,
        )
        return 0

    logger.info(
        "Focus v2: processing session '%s' (%s): %d events",
        title, session_id, len(focus_events),
    )

    # Run through v2 focus processor
    result = focus_processor.process_session(
        db, session_id, title, focus_events,
        screenshots_dir=screenshots_dir,
    )

    exported = 0
    if result.success and result.sop:
        sop_templates = [result.sop]
        slug = result.sop.get("slug", "")

        # Lint / validate before save+export
        if not _lint_and_log(result.sop, "focus_v2"):
            # Save invalid SOP as draft for debugging, skip export
            try:
                db.save_generated_sop(
                    slug=slug,
                    title=title,
                    source="focus",
                    sop_template=result.sop,
                    confidence=0.0,
                    source_id=session_id,
                    auto_approve=False,
                )
            except Exception:
                logger.debug(
                    "Failed to save invalid SOP draft to DB", exc_info=True,
                )
            _clear_focus_signal(signal_path)
            return 0

        # Save generated SOP to DB for review tracking
        try:
            db.save_generated_sop(
                slug=slug,
                title=title,
                source="focus",
                sop_template=result.sop,
                confidence=result.sop.get("confidence_avg", result.sop.get("confidence", 0.0)),
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
            from oc_apprentice_worker.sop_dedup import deduplicate_templates
            sop_templates = deduplicate_templates(sop_templates, _status_dir())

            # Export via primary writer
            try:
                paths = openclaw_writer.write_all_sops(sop_templates)
                exported = len(paths)
                index_generator.update_index(
                    openclaw_writer.get_sops_dir(), sop_templates
                )
                logger.info(
                    "Focus v2 session '%s': exported %d SOP(s) (%.1fs VLM time)",
                    title, exported, result.inference_time_seconds,
                )
            except Exception:
                logger.exception("Focus v2 SOP export failed")

            # Also export as SKILL.md
            if skill_md_writer is not None:
                try:
                    skill_md_writer.write_all_sops(sop_templates)
                    logger.info("Focus v2 session '%s': SKILL.md export complete", title)
                except Exception:
                    logger.exception("Focus v2 SKILL.md export failed")

            # Also export as Claude Code skills
            if claude_skill_writer is not None:
                try:
                    claude_skill_writer.write_all_sops(sop_templates)
                    logger.info("Focus v2 session '%s': Claude skill export complete", title)
                except Exception:
                    logger.exception("Focus v2 Claude skill export failed")

            # Write v3 procedures to the knowledge base
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
            if kb_export_adapter is not None:
                try:
                    kb_export_adapter.write_all_sops(sop_templates)
                except Exception:
                    logger.debug("KB export adapter failed", exc_info=True)

            # Cache for CLI export trigger
            _save_sop_cache(sop_templates)
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
                error=result.error or "Unknown error",
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
            title, result.error,
        )

    # Clear signal file
    _clear_focus_signal(signal_path)
    return exported


def _process_passive_discovery(
    db: "WorkerDB",
    *,
    segmenter: "TaskSegmenter",
    sop_generator: "SOPGenerator",
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    sop_auto_approve: bool = True,
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
) -> int:
    """Run passive discovery: segment annotations → generate SOPs.

    Called periodically (every 2 hours, on idle, or by CLI trigger).
    Returns the number of SOPs generated.

    Pipeline:
      1. Load recent annotated events from DB
      2. Run task segmenter (embeddings + clustering + noise filter)
      3. Persist segments to DB
      4. For clusters with ≥2 demonstrations, generate passive SOPs
      5. Export SOPs
    """
    # Step 1: Load annotated events
    events = db.get_annotated_events_in_window(
        hours=segmenter.config.default_window_hours,
    )
    if not events:
        logger.debug("Passive discovery: no annotated events in window")
        return 0

    logger.info(
        "Passive discovery: processing %d annotated events", len(events),
    )

    # Step 2: Run segmentation
    seg_result = segmenter.segment(events)

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
                seg_result.segments, all_frames,
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

    # Step 3: Persist segments to DB
    for seg in seg_result.segments:
        event_ids = [f.event_id for f in seg.frames]
        db.save_task_segment(
            segment_id=seg.segment_id,
            cluster_id=seg.cluster_id,
            task_label=seg.task_label,
            event_ids=event_ids,
            apps=seg.apps_involved,
            start_time=seg.start_time,
            end_time=seg.end_time,
        )

    # Step 4: Query DB for pending clusters (sop_generated = 0)
    # This uses persisted state instead of transient in-memory cluster IDs,
    # so re-segmentation with shuffled cluster IDs won't re-trigger SOPs.
    import json as _json

    pending_clusters = db.get_sop_pending_clusters()

    if not pending_clusters:
        logger.info(
            "Passive discovery: no clusters with ≥2 pending segments in DB",
        )
        return 0

    total_exported = 0

    for cluster_row in pending_clusters:
        cluster_id = cluster_row["cluster_id"]
        task_label = cluster_row.get("task_label", "")

        # Get all pending segments for this cluster from DB
        pending_segs = db.get_cluster_segments(cluster_id)
        if len(pending_segs) < segmenter.config.min_demonstrations:
            continue

        # Reconstruct demonstrations (timelines) from DB event IDs
        demonstrations: list[list[dict]] = []
        segment_ids: list[str] = []
        for seg_row in pending_segs:
            event_ids = _json.loads(seg_row.get("event_ids_json", "[]"))
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
                timeline.append({
                    "annotation": annotation,
                    "diff": diff,
                    "timestamp": ev.get("timestamp", ""),
                })
            if timeline:
                demonstrations.append(timeline)
                segment_ids.append(seg_row["segment_id"])

        if len(demonstrations) < segmenter.config.min_demonstrations:
            continue

        logger.info(
            "Passive discovery: generating SOP for '%s' (%d demos)",
            task_label[:60], len(demonstrations),
        )

        result = sop_generator.generate_from_passive(
            demonstrations, task_label=task_label,
        )

        if not result.success or not result.sop:
            # Record failure for retry tracking
            try:
                db.record_failed_generation(
                    source="passive",
                    source_id=str(cluster_id),
                    error=result.error or "Unknown error",
                    title=task_label,
                    context={"segment_count": len(pending_segs)},
                )
            except Exception:
                logger.debug(
                    "Failed to record generation failure for '%s'",
                    task_label[:60], exc_info=True,
                )
            logger.warning(
                "Passive SOP generation failed for '%s': %s",
                task_label[:60], result.error,
            )
            continue

        sop_templates = [result.sop]
        slug = result.sop.get("slug", "")

        # Lint / validate before save+export
        if not _lint_and_log(result.sop, "passive"):
            # Save invalid SOP as draft for debugging, skip export
            try:
                db.save_generated_sop(
                    slug=slug,
                    title=task_label,
                    source="passive",
                    sop_template=result.sop,
                    confidence=0.0,
                    source_id=str(cluster_id),
                    auto_approve=False,
                )
            except Exception:
                logger.debug(
                    "Failed to save invalid SOP draft to DB", exc_info=True,
                )
            continue

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
            # Deduplicate against known SOPs
            from oc_apprentice_worker.sop_dedup import deduplicate_templates
            sop_templates = deduplicate_templates(sop_templates, _status_dir())

            # Export
            try:
                paths = openclaw_writer.write_all_sops(sop_templates)
                total_exported += len(paths)
                index_generator.update_index(
                    openclaw_writer.get_sops_dir(), sop_templates,
                )
                logger.info(
                    "Passive SOP '%s': exported %d SOP(s) (%.1fs VLM)",
                    task_label[:60], len(paths), result.inference_time_seconds,
                )
            except Exception:
                logger.exception("Passive SOP export failed for '%s'", task_label[:60])

            if skill_md_writer is not None:
                try:
                    skill_md_writer.write_all_sops(sop_templates)
                except Exception:
                    logger.exception("Passive SKILL.md export failed")

            if claude_skill_writer is not None:
                try:
                    claude_skill_writer.write_all_sops(sop_templates)
                except Exception:
                    logger.exception("Passive Claude skill export failed")

            # Write v3 procedures to the knowledge base
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

        # Mark all pending segments in this cluster as SOP-generated
        for seg_id in segment_ids:
            db.mark_segment_sop_generated(seg_id)

    return total_exported


def _process_retry_triggers(
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
    """Check for and process a retry-trigger.json file.

    Re-runs SOP generation for a previously failed attempt, then marks
    the failure as retried in the DB.
    """
    state_dir = _status_dir()
    trigger_path = state_dir / "retry-trigger.json"

    if not trigger_path.is_file():
        return

    try:
        with open(trigger_path) as f:
            trigger = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read retry-trigger.json", exc_info=True)
        return

    failure_id = trigger.get("failure_id", "")
    if not failure_id:
        logger.warning("retry-trigger.json missing failure_id")
        _remove_trigger(trigger_path)
        return

    logger.info("Processing retry trigger for failure_id=%s", failure_id)

    try:
        failure = db.get_failed_generation(failure_id)
    except Exception:
        logger.warning(
            "Failed to read failure record %s from DB",
            failure_id, exc_info=True,
        )
        _remove_trigger(trigger_path)
        return

    if failure is None:
        logger.warning("Retry trigger: failure_id=%s not found in DB", failure_id)
        _remove_trigger(trigger_path)
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
                        try:
                            db.save_generated_sop(
                                slug=slug, title=title, source="focus",
                                sop_template=result.sop, confidence=0.0,
                                source_id=source_id, auto_approve=False,
                            )
                        except Exception:
                            logger.debug("Failed to save invalid retry SOP draft", exc_info=True)
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
            # Re-run passive generation for the cluster
            pending_segs = db.get_cluster_segments(int(source_id))
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

                        # Lint before save+export
                        if not _lint_and_log(result.sop, "retry_passive"):
                            try:
                                db.save_generated_sop(
                                    slug=slug, title=title, source="passive",
                                    sop_template=result.sop, confidence=0.0,
                                    source_id=source_id, auto_approve=False,
                                )
                            except Exception:
                                logger.debug("Failed to save invalid retry SOP draft", exc_info=True)
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
                            logger.info(
                                "Retry succeeded for passive cluster '%s'", title,
                            )
                    else:
                        logger.warning(
                            "Retry failed again for passive cluster '%s': %s",
                            title, result.error,
                        )
            else:
                logger.warning(
                    "Retry: no segments found for cluster %s", source_id,
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

    _remove_trigger(trigger_path)


def _process_approval_triggers(
    db: "WorkerDB",
    *,
    openclaw_writer: "SOPExportAdapter",
    skill_md_writer: "SOPExportAdapter | None" = None,
    claude_skill_writer: "SOPExportAdapter | None" = None,
    index_generator: "IndexGenerator",
    procedure_writer: "ProcedureWriter | None" = None,
    kb_export_adapter: "KnowledgeBaseExportAdapter | None" = None,
) -> None:
    """Check for and process an approve-trigger.json file.

    Approves or rejects a draft SOP.  On approval, exports the SOP via
    all active adapters.
    """
    state_dir = _status_dir()
    trigger_path = state_dir / "approve-trigger.json"

    if not trigger_path.is_file():
        return

    try:
        with open(trigger_path) as f:
            trigger = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read approve-trigger.json", exc_info=True)
        return

    sop_id_or_slug = trigger.get("sop_id", "")
    action = trigger.get("action", "")

    if not sop_id_or_slug or action not in ("approve", "reject"):
        logger.warning(
            "approve-trigger.json invalid: sop_id=%s action=%s",
            sop_id_or_slug, action,
        )
        _remove_trigger(trigger_path)
        return

    # Resolve: try as UUID first, then as slug
    sop_record = db.get_generated_sop(sop_id_or_slug)
    if sop_record is None:
        sop_record = db.get_generated_sop_by_slug(sop_id_or_slug)
    sop_id = sop_record["sop_id"] if sop_record else sop_id_or_slug

    logger.info("Processing approval trigger: sop_id=%s action=%s", sop_id, action)

    try:
        if action == "approve":
            if db.update_sop_status(sop_id, "approved"):
                sop_record = db.get_generated_sop(sop_id)
                if sop_record and sop_record.get("sop_json"):
                    sop_template = sop_record["sop_json"]
                    if not _lint_and_log(sop_template, "approval"):
                        logger.error(
                            "Approved SOP %s failed lint validation, export skipped",
                            sop_id,
                        )
                    else:
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
                else:
                    logger.warning(
                        "Approved SOP %s but could not load template for export",
                        sop_id,
                    )
            else:
                logger.warning("Failed to approve SOP %s (not found?)", sop_id)

        elif action == "reject":
            if db.update_sop_status(sop_id, "rejected"):
                logger.info("Rejected SOP %s", sop_id)
            else:
                logger.warning("Failed to reject SOP %s (not found?)", sop_id)
    except Exception:
        logger.warning(
            "Approval trigger processing failed for %s",
            sop_id, exc_info=True,
        )

    _remove_trigger(trigger_path)


def _process_failed_query_trigger(db: "WorkerDB") -> None:
    """Check for and process a failed-query-trigger.json file.

    Reads failed generations from the DB and writes the result to
    failed-query-result.json for the CLI to pick up.
    """
    state_dir = _status_dir()
    trigger_path = state_dir / "failed-query-trigger.json"

    if not trigger_path.is_file():
        return

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

    _remove_trigger(trigger_path)


def _process_search_trigger(
    activity_searcher: "ActivitySearcher | None",
) -> None:
    """Check for and process a search-query-trigger.json file.

    Runs the query through the ActivitySearcher and writes results
    to search-query-result.json for the CLI to pick up.
    """
    state_dir = _status_dir()
    trigger_path = state_dir / "search-query-trigger.json"

    if not trigger_path.is_file():
        return

    logger.info("Processing search query trigger")

    # Remove any stale result file so the CLI cannot read old data
    result_path = state_dir / "search-query-result.json"
    result_path.unlink(missing_ok=True)

    try:
        with open(trigger_path) as f:
            trigger = json.load(f)

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

    _remove_trigger(trigger_path)


def _process_recall_trigger(
    activity_searcher: "ActivitySearcher | None",
) -> None:
    """Check for and process a recall-query-trigger.json file.

    Runs session recall and writes results to recall-query-result.json.
    """
    state_dir = _status_dir()
    trigger_path = state_dir / "recall-query-trigger.json"

    if not trigger_path.is_file():
        return

    logger.info("Processing recall query trigger")

    # Remove any stale result file so the CLI cannot read old data
    result_path = state_dir / "recall-query-result.json"
    result_path.unlink(missing_ok=True)

    try:
        with open(trigger_path) as f:
            trigger = json.load(f)

        date = trigger.get("date")
        app = trigger.get("app")
        start_time = trigger.get("start_time")
        end_time = trigger.get("end_time")

        # Convert HH:MM shorthand to full ISO timestamps.
        # The searcher expects ISO format (e.g. "2026-03-12T09:30:00")
        # but the CLI / trigger file may use bare "HH:MM".
        import re as _re
        _hhmm_re = _re.compile(r"^\d{2}:\d{2}$")
        _date_prefix = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if start_time and _hhmm_re.match(start_time):
            start_time = f"{_date_prefix}T{start_time}:00"
        if end_time and _hhmm_re.match(end_time):
            end_time = f"{_date_prefix}T{end_time}:00"

        if activity_searcher is None:
            result = {"error": "Activity search not available (v2 schema required)"}
        else:
            timeline = activity_searcher.session_recall(
                date=date, app=app,
                start_time=start_time, end_time=end_time,
            )
            result = {
                "date": timeline.date,
                "total_active_minutes": timeline.total_active_minutes,
                "apps_used": timeline.apps_used,
                "entries": [
                    {
                        "timestamp": e.timestamp,
                        "app": e.app,
                        "location": e.location,
                        "what_doing": e.what_doing,
                        "event_id": e.event_id,
                    }
                    for e in timeline.entries
                ],
            }

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

    _remove_trigger(trigger_path)


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
    from oc_apprentice_worker.sop_dedup import deduplicate_templates
    sop_templates = deduplicate_templates(sop_templates, _status_dir())

    try:
        paths = openclaw_writer.write_all_sops(sop_templates)
        index_generator.update_index(
            openclaw_writer.get_sops_dir(), sop_templates,
        )
        logger.info("Exported %d SOP(s) via primary adapter", len(paths))
    except Exception:
        logger.exception("SOP export failed (primary adapter)")

    if skill_md_writer is not None:
        try:
            skill_md_writer.write_all_sops(sop_templates)
        except Exception:
            logger.exception("SKILL.md export failed")

    if claude_skill_writer is not None:
        try:
            claude_skill_writer.write_all_sops(sop_templates)
        except Exception:
            logger.exception("Claude Code skill export failed")

    # Write v3 procedures to the knowledge base so downstream consumers
    # (trust advisor, query API, staleness detector) can find them.
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

    if kb_export_adapter is not None:
        try:
            kb_export_adapter.write_all_sops(sop_templates)
        except Exception:
            logger.debug("KB export adapter failed", exc_info=True)

    _save_sop_cache(sop_templates)


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
        from oc_apprentice_worker.exporter import AtomicWriter
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
    *,
    openclaw_writer: "SOPExportAdapter",
    sops_dir: Path,
) -> None:
    """Check for and process an export-trigger.json file written by the CLI.

    The CLI ``openmimic export`` writes a trigger file requesting re-export of
    existing SOPs in a specific format (skill-md, generic, openclaw).  The
    worker picks it up here, creates the appropriate writer, runs the re-export,
    and removes the trigger.
    """
    state_dir = _status_dir()
    trigger_path = state_dir / "export-trigger.json"

    if not trigger_path.is_file():
        return

    try:
        with open(trigger_path) as f:
            trigger = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read export-trigger.json", exc_info=True)
        return

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
        _remove_trigger(trigger_path)
        return

    if sop_slug:
        sop_templates = [
            t for t in sop_templates if t.get("slug") == sop_slug
        ]
        if not sop_templates:
            logger.warning("Export trigger: no SOP with slug '%s' found", sop_slug)
            _remove_trigger(trigger_path)
            return

    # Resolve output directory: trigger output_dir overrides default
    workspace_dir = Path(output_dir) if output_dir else sops_dir.parent.parent
    exported = 0

    if fmt in ("skill-md", "all"):
        try:
            from oc_apprentice_worker.skill_md_writer import SkillMdWriter
            writer = SkillMdWriter(workspace_dir=workspace_dir)
            writer.write_all_sops(sop_templates)
            exported += len(sop_templates)
            logger.info("Export trigger: wrote %d SKILL.md file(s)", len(sop_templates))
        except Exception:
            logger.exception("Export trigger: SKILL.md write failed")

    if fmt in ("openclaw", "all"):
        try:
            if output_dir:
                # Respect output_dir: create a new writer pointing there
                from oc_apprentice_worker.openclaw_writer import OpenClawWriter
                oc_writer = OpenClawWriter(workspace_dir=Path(output_dir))
            else:
                oc_writer = openclaw_writer
            oc_writer.write_all_sops(sop_templates)
            exported += len(sop_templates)
            logger.info("Export trigger: wrote %d OpenClaw file(s)", len(sop_templates))
        except Exception:
            logger.exception("Export trigger: OpenClaw write failed")

    if fmt in ("generic", "all"):
        try:
            from oc_apprentice_worker.generic_writer import GenericWriter
            out = Path(output_dir) if output_dir else sops_dir
            writer = GenericWriter(output_dir=out, json_export=True)
            writer.write_all_sops(sop_templates)
            exported += len(sop_templates)
            logger.info("Export trigger: wrote %d generic file(s)", len(sop_templates))
        except Exception:
            logger.exception("Export trigger: generic write failed")

    if fmt in ("claude-skill", "all"):
        try:
            from oc_apprentice_worker.claude_skill_writer import ClaudeSkillWriter
            skills_dir = Path(output_dir) / "skills" if output_dir else None
            writer = ClaudeSkillWriter(skills_dir=skills_dir)
            writer.write_all_sops(sop_templates)
            exported += len(sop_templates)
            logger.info("Export trigger: wrote %d Claude Code skill(s)", len(sop_templates))
        except Exception:
            logger.exception("Export trigger: Claude Code skill write failed")

    if exported == 0 and fmt not in ("skill-md", "openclaw", "generic", "claude-skill", "all"):
        logger.warning("Export trigger: unsupported format '%s'", fmt)

    _remove_trigger(trigger_path)


def _remove_trigger(trigger_path: Path) -> None:
    """Remove the export trigger file."""
    try:
        trigger_path.unlink(missing_ok=True)
    except Exception:
        logger.debug("Failed to remove export-trigger.json", exc_info=True)


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


def _persist_vlm_jobs(db: "WorkerDB", vlm_queue: VLMFallbackQueue) -> None:
    """Persist in-memory VLM queue jobs to the database."""
    from oc_apprentice_worker.vlm_queue import VLMJobStatus

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
    from oc_apprentice_worker.vlm_worker import VLMRequest

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
                            from oc_apprentice_worker.vlm_queue import VLMJobStatus
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

    # Initialize knowledge base and Phase 2+ modules
    from oc_apprentice_worker.knowledge_base import KnowledgeBase
    from oc_apprentice_worker.evidence_tracker import EvidenceTracker
    from oc_apprentice_worker.procedure_writer import ProcedureWriter
    from oc_apprentice_worker.knowledge_export_adapter import KnowledgeBaseExportAdapter
    from oc_apprentice_worker.privacy_zones import PrivacyZoneChecker, PrivacyZoneConfig
    from oc_apprentice_worker.daily_processor import DailyBatchProcessor
    from oc_apprentice_worker.staleness_detector import StalenessDetector
    from oc_apprentice_worker.profile_builder import ProfileBuilder
    from oc_apprentice_worker.pattern_detector import PatternDetector
    from oc_apprentice_worker.constraint_manager import ConstraintManager
    from oc_apprentice_worker.execution_monitor import ExecutionMonitor
    from oc_apprentice_worker.correction_detector import CorrectionDetector
    from oc_apprentice_worker.trust_advisor import TrustAdvisor
    from oc_apprentice_worker.session_linker import SessionLinker
    from oc_apprentice_worker.daily_digest import DigestGenerator

    knowledge_base = KnowledgeBase(root=args.knowledge_dir)
    knowledge_base.ensure_structure()
    evidence_tracker = EvidenceTracker(knowledge_base=knowledge_base)
    procedure_writer = ProcedureWriter(kb=knowledge_base, evidence=evidence_tracker)
    kb_export_adapter = KnowledgeBaseExportAdapter(knowledge_base)
    privacy_checker = PrivacyZoneChecker()
    daily_processor = DailyBatchProcessor(knowledge_base=knowledge_base)
    staleness_detector = StalenessDetector(knowledge_base)
    profile_builder = ProfileBuilder(knowledge_base)
    pattern_detector = PatternDetector(knowledge_base)
    constraint_manager = ConstraintManager(knowledge_base)

    # Phase 4: Execution monitoring, correction detection, trust advisor
    execution_monitor = ExecutionMonitor(knowledge_base)
    correction_detector = CorrectionDetector(knowledge_base)
    trust_advisor = TrustAdvisor(knowledge_base)
    session_linker = SessionLinker(knowledge_base)
    digest_generator = DigestGenerator(knowledge_base)

    logger.info("Knowledge base initialized: %s", args.knowledge_dir)

    # Initialize pipeline components
    episode_builder = EpisodeBuilder()
    clipboard_linker = ClipboardLinker()
    pruner = NegativeDemoPruner()
    translator = SemanticTranslator()
    scorer = ConfidenceScorer()
    vlm_queue = VLMFallbackQueue()
    index_generator = IndexGenerator()
    # Create export adapter based on config
    skill_md_writer = None
    claude_skill_writer = None
    if args.adapter == "generic":
        from oc_apprentice_worker.generic_writer import GenericWriter
        sop_writer = GenericWriter(output_dir=args.sops_dir, json_export=args.json_export)
    elif args.adapter == "skill-md":
        from oc_apprentice_worker.skill_md_writer import SkillMdWriter
        sop_writer = SkillMdWriter(workspace_dir=args.sops_dir.parent.parent)
    elif args.adapter == "claude-skill":
        from oc_apprentice_worker.claude_skill_writer import ClaudeSkillWriter
        sop_writer = ClaudeSkillWriter()
    elif args.adapter == "all":
        sop_writer = OpenClawWriter(workspace_dir=args.sops_dir.parent.parent)
        from oc_apprentice_worker.skill_md_writer import SkillMdWriter
        skill_md_writer = SkillMdWriter(workspace_dir=args.sops_dir.parent.parent)
        from oc_apprentice_worker.claude_skill_writer import ClaudeSkillWriter
        claude_skill_writer = ClaudeSkillWriter()
    else:
        sop_writer = OpenClawWriter(workspace_dir=args.sops_dir.parent.parent)

    # VLM initialization — supports local and remote modes
    from oc_apprentice_worker.vlm_worker import VLMWorker, VLMConfig, VLMBackend
    vlm_worker = None
    vlm_available = False
    vlm_mode_str = _read_vlm_config_field("mode", "local")
    vlm_provider_str = _read_vlm_config_field("provider", "")

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
                remote_model = _read_vlm_config_field("model", "")
                api_key_env = _read_vlm_config_field("api_key_env", "")
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
        from oc_apprentice_worker.setup_vlm import check_vlm_available
        vlm_status = check_vlm_available()
        if any(vlm_status.values()):
            logger.info("VLM backend(s) detected — attempting initialization")
            try:
                # Check if the configured model name uses Ollama naming
                # convention (e.g. "qwen3.5:2b") vs HuggingFace repo IDs
                # (e.g. "mlx-community/llava-1.5-7b-4bit").  Ollama names
                # contain ":" which is invalid for HuggingFace repo IDs,
                # so we prefer the Ollama backend when that format is used.
                config_model_hint = _read_vlm_config_field("model", "")
                _model_is_ollama_style = ":" in config_model_hint

                if _model_is_ollama_style and vlm_status["ollama"]:
                    backend_type = VLMBackend.OLLAMA
                elif vlm_status["mlx_vlm"] and not _model_is_ollama_style:
                    backend_type = VLMBackend.MLX_VLM
                elif vlm_status["ollama"]:
                    backend_type = VLMBackend.OLLAMA
                elif vlm_status["mlx_vlm"]:
                    backend_type = VLMBackend.MLX_VLM
                elif vlm_status["llama_cpp"]:
                    backend_type = VLMBackend.LLAMA_CPP
                elif vlm_status.get("openai_compat"):
                    base_url = os.environ.get("OPENMIMIC_VLM_BASE_URL", "")
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
                            "enforced by default. Set OPENMIMIC_VLM_BASE_URL to a "
                            "local server (e.g. http://localhost:8000) to use it."
                        )
                        backend_type = None  # type: ignore[assignment]
                else:
                    backend_type = None  # type: ignore[assignment]
                if backend_type is not None:
                    # Read optional model override from config.toml [vlm] section
                    config_model = _read_vlm_config_field("model", "")
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
                "VLM not installed. For better native app observation, run: oc-setup-vlm"
            )

    # Initialize LLM-enhanced SOP descriptions
    sop_enhancer = None
    llm_config = _read_llm_config()

    # CLI args override config
    if args.enhance_sops is not None:
        llm_config["enhance_sops"] = args.enhance_sops
    if args.llm_model is not None:
        llm_config["model"] = args.llm_model
    if args.max_enhancements_per_day is not None:
        llm_config["max_enhancements_per_day"] = args.max_enhancements_per_day

    if llm_config.get("enhance_sops", True):
        try:
            from oc_apprentice_worker.sop_enhancer import (
                SOPEnhancer,
                create_llm_backend,
            )

            # Build vlm config dict for create_llm_backend
            vlm_cfg_dict = {
                "mode": vlm_mode_str,
                "provider": vlm_provider_str,
                "model": _read_vlm_config_field("model", ""),
                "api_key_env": _read_vlm_config_field("api_key_env", ""),
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

    # Try to import SOPInducer (requires prefixspan)
    sop_inducer = None
    sop_inducer_available = False
    try:
        from oc_apprentice_worker.sop_inducer import SOPInducer
        # Use low min_support (abs_support will be max(2, ...)) and
        # min_pattern_length=2 to enable cold-start pattern discovery.
        # The sliding-window mining inside the inducer creates enough
        # sequences for PrefixSpan even with few real episodes.
        sop_inducer = SOPInducer(
            min_support=0.05,
            min_pattern_length=2,
            vlm_worker=vlm_worker,
        )
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

    # Initialize v2 scene annotation pipeline
    v2_cfg = _read_vlm_v2_config()
    scene_annotator: SceneAnnotator | None = None
    frame_differ: FrameDiffer | None = None
    v2_annotation_enabled = v2_cfg["annotation_enabled"]
    screenshots_dir = _status_dir() / "screenshots"

    focus_processor: FocusProcessor | None = None
    task_segmenter: TaskSegmenter | None = None
    sop_generator: SOPGenerator | None = None

    if v2_annotation_enabled:
        ann_config = AnnotationConfig(
            model=v2_cfg["annotation_model"],
            ollama_host=v2_cfg["ollama_host"],
            stale_skip_count=v2_cfg["stale_skip_count"],
            sliding_window_max_age_sec=v2_cfg["sliding_window_max_age_sec"],
        )
        scene_annotator = SceneAnnotator(config=ann_config)

        diff_config = DiffConfig(
            model=v2_cfg["annotation_model"],
            ollama_host=v2_cfg["ollama_host"],
        )
        frame_differ = FrameDiffer(config=diff_config)

        # SOP generator (4B thinking model) for focus session SOPs
        sop_gen_config = SOPGeneratorConfig(
            model=v2_cfg["sop_model"],
            ollama_host=v2_cfg["ollama_host"],
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
            ollama_host=v2_cfg["ollama_host"],
        )
        task_segmenter = TaskSegmenter(config=seg_config)

        logger.info(
            "v2 scene annotation pipeline enabled "
            "(annotation=%s, sop=%s, stale_skip=%d, window=%ds)",
            v2_cfg["annotation_model"],
            v2_cfg["sop_model"],
            v2_cfg["stale_skip_count"],
            v2_cfg["sliding_window_max_age_sec"],
        )
    else:
        logger.info("v2 scene annotation pipeline disabled by config")

    # Initialize activity searcher for CLI search/recall commands
    activity_searcher = None
    try:
        from oc_apprentice_worker.activity_search import ActivitySearcher
        activity_searcher = ActivitySearcher(db_path=args.db_path)
        logger.info("Activity searcher initialized")
    except Exception:
        logger.debug("Activity searcher not available", exc_info=True)

    # Phase 4: Start query API server (if enabled in config)
    query_api_server = None
    try:
        import tomllib as _tomllib
        _cfg_path = (
            Path.home() / "Library" / "Application Support" / "oc-apprentice" / "config.toml"
            if _platform.system() == "Darwin"
            else Path.home() / ".config" / "oc-apprentice" / "config.toml"
        )
        _knowledge_cfg = {}
        if _cfg_path.is_file():
            with open(_cfg_path, "rb") as _cf:
                _knowledge_cfg = _tomllib.load(_cf).get("knowledge", {})
        if _knowledge_cfg.get("query_api_enabled", False):
            from oc_apprentice_worker.query_api import QueryAPIServer
            _api_port = _knowledge_cfg.get("query_api_port", 9477)
            query_api_server = QueryAPIServer(
                knowledge_base=knowledge_base,
                port=_api_port,
                activity_searcher=activity_searcher,
            )
            query_api_server.start()
            logger.info("Query API server started on port %d", _api_port)
    except Exception:
        logger.debug("Query API server not started", exc_info=True)

    # Read SOP review config
    sop_config = _read_sop_config()
    sop_auto_approve = sop_config["auto_approve"]
    logger.info("SOP auto_approve: %s", sop_auto_approve)

    # Initialize scheduler gate (GAP 5) and deep scanner (GAP 6)
    # Read idle_jobs config from config.toml if available
    _idle_cfg = _read_idle_jobs_config()
    idle_gate = IdleJobGate(SchedulerConfig(**_idle_cfg))
    deep_scanner = DeepScanner()

    # Single-thread pool for background deep scan (non-blocking privacy check)
    deep_scan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deep-scan")

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

    # Episode store cleanup — run once per day
    _last_episode_cleanup = 0.0

    # Daily batch processing — run once per day
    _last_daily_batch: str = ""  # YYYY-MM-DD of last batch
    _last_staleness_check = 0.0

    # Passive discovery timer — run every 2 hours (7200s)
    _PASSIVE_DISCOVERY_INTERVAL = 7200
    _last_passive_discovery = 0.0
    total_passive_sops: int = 0

    # Write initial status file after DB connect
    _vlm_stats = vlm_queue.get_stats()
    _write_worker_status(
        started_at=started_at,
        events_processed_today=total_events_processed,
        sops_generated=total_sops_generated,
        last_pipeline_duration_ms=last_pipeline_duration_ms,
        consecutive_errors=0,
        vlm_available=vlm_available,
        sop_inducer_available=sop_inducer_available,
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
        max_interval = max(60.0, args.poll_interval * 16)
        consecutive_errors = 0

        # Force-write sops-index.json at startup so the SwiftUI app
        # immediately sees the latest data (including any new fields).
        _write_sops_index(db, force=True)

        while not shutdown_flag[0]:
            try:
                # Process retry and approval triggers first
                _process_retry_triggers(
                    db,
                    focus_processor=focus_processor,
                    sop_generator=sop_generator,
                    openclaw_writer=sop_writer,
                    skill_md_writer=skill_md_writer,
                    claude_skill_writer=claude_skill_writer,
                    index_generator=index_generator,
                    screenshots_dir=screenshots_dir,
                    sop_auto_approve=sop_auto_approve,
                    procedure_writer=procedure_writer,
                    kb_export_adapter=kb_export_adapter,
                )
                _process_approval_triggers(
                    db,
                    openclaw_writer=sop_writer,
                    skill_md_writer=skill_md_writer,
                    claude_skill_writer=claude_skill_writer,
                    index_generator=index_generator,
                    procedure_writer=procedure_writer,
                    kb_export_adapter=kb_export_adapter,
                )
                _process_failed_query_trigger(db)
                _process_search_trigger(activity_searcher)
                _process_recall_trigger(activity_searcher)

                # Process any completed focus recording sessions first.
                # Use v2 VLM pipeline when available (semantic SOPs from
                # screen annotations), fall back to v1 (episode builder +
                # translator + PrefixSpan).
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
                    )
                else:
                    focus_sops = _process_focus_sessions(
                        db,
                        episode_builder=episode_builder,
                        clipboard_linker=clipboard_linker,
                        translator=translator,
                        scorer=scorer,
                        vlm_queue=vlm_queue,
                        sop_inducer=sop_inducer,
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
                    _write_sops_index(db)

                # Check for CLI-requested export triggers
                _check_export_trigger(
                    openclaw_writer=sop_writer,
                    sops_dir=args.sops_dir,
                )

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
                    # Core pipeline ALWAYS runs — episodes, translation,
                    # SOP mining, and export happen immediately so users
                    # see SOPs as soon as patterns are detected.
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
                        sop_enhancer=sop_enhancer,
                        skill_md_writer=skill_md_writer,
                        claude_skill_writer=claude_skill_writer,
                        db=db,
                        procedure_writer=procedure_writer,
                        kb_export_adapter=kb_export_adapter,
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
                        _write_sops_index(db)
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

                    # Persist in-memory VLM queue jobs to the DB so they survive restarts
                    if summary["vlm_enqueued"] > 0:
                        _persist_vlm_jobs(db, vlm_queue)

                    # Reset consecutive error counter after successful processing
                    consecutive_errors = 0

                    # --- Heavy jobs gated behind idle window ---
                    # VLM inference and deep privacy scans are CPU/GPU
                    # intensive and deferred to the idle window
                    # (default 01:00-05:00, AC power, low CPU).
                    gate_result = idle_gate.check()
                    if gate_result.can_run:
                        # Run Tier 2 deep scan in background (non-blocking)
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

                # Process pending VLM jobs only in idle window
                if pending_vlm and vlm_worker is not None:
                    gate_result = idle_gate.check()
                    if gate_result.can_run:
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
                    ann_stats = _process_annotations(
                        db, scene_annotator, screenshots_dir,
                        privacy_checker=privacy_checker,
                    )
                    total_v2_annotations += ann_stats["annotated"]

                    if ann_stats["annotated"] > 0 or ann_stats["blocked"] > 0:
                        logger.info(
                            "v2 annotation: %d annotated, %d skipped, %d failed, %d blocked",
                            ann_stats["annotated"],
                            ann_stats["skipped"],
                            ann_stats["failed"],
                            ann_stats["blocked"],
                        )

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
                    diff_stats = _process_diffs(db, frame_differ)
                    total_v2_diffs += diff_stats["diffs"]

                    if diff_stats["diffs"] > 0 or diff_stats["failed"] > 0:
                        logger.info(
                            "v2 frame diff: %d action diffs, %d edge cases, %d failed",
                            diff_stats["diffs"],
                            diff_stats["edge_cases"],
                            diff_stats["failed"],
                        )

                # --- v2 Passive Discovery (periodic) ---
                # Run task segmenter + passive SOP generation every 2 hours,
                # or when user has been idle for 5+ minutes.
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

                    if should_segment:
                        try:
                            pd_sops = _process_passive_discovery(
                                db,
                                segmenter=task_segmenter,
                                sop_generator=sop_generator,
                                openclaw_writer=sop_writer,
                                skill_md_writer=skill_md_writer,
                                claude_skill_writer=claude_skill_writer,
                                index_generator=index_generator,
                                sop_auto_approve=sop_auto_approve,
                                procedure_writer=procedure_writer,
                                kb_export_adapter=kb_export_adapter,
                            )
                            if pd_sops > 0:
                                total_passive_sops += pd_sops
                                total_sops_generated += pd_sops
                                logger.info(
                                    "Passive discovery: generated %d SOP(s)!",
                                    pd_sops,
                                )
                                _write_sops_index(db)
                        except Exception:
                            logger.warning(
                                "Passive discovery failed",
                                exc_info=True,
                            )
                        _last_passive_discovery = _now_mono_pd

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
                    sop_inducer_available=sop_inducer_available,
                    vlm_queue_pending=db_vlm_pending,
                    vlm_jobs_today=_vlm_stats.jobs_today,
                    vlm_dropped_today=_vlm_stats.dropped_count,
                    vlm_mode=vlm_mode_str,
                    vlm_provider=vlm_provider_str or None,
                    v2_annotation_enabled=v2_annotation_enabled,
                    v2_annotations_today=total_v2_annotations,
                    v2_diffs_today=total_v2_diffs,
                )

                # Refresh sops-index.json (throttled to every 5s)
                _write_sops_index(db)

                # Episode store cleanup — once per day (86400s)
                _now_mono = time.monotonic()
                if _now_mono - _last_episode_cleanup > 86400:
                    try:
                        db.cleanup_old_episodes(max_age_days=14)
                    except Exception:
                        logger.debug("Episode store cleanup failed", exc_info=True)
                    _last_episode_cleanup = _now_mono

                # --- Phase 2+: Daily batch + staleness (once per day) ---
                # Process *yesterday* — today is still in progress and would
                # produce an incomplete summary.  The guard uses today_str so
                # the batch runs at most once per calendar day.
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                yesterday_str = (
                    datetime.now(timezone.utc) - timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if today_str != _last_daily_batch:
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
                            # Update profile + patterns periodically
                            summaries_count = len(
                                knowledge_base.list_daily_summaries(limit=5)
                            )
                            if summaries_count >= 3:
                                profile_builder.update_profile()
                                patterns = pattern_detector.detect_recurrence()
                                if patterns:
                                    pattern_detector.update_triggers(patterns)
                                chains = pattern_detector.detect_chains()
                                if chains:
                                    pattern_detector.update_chains(chains)
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
                            # Generate daily digest
                            try:
                                digest = digest_generator.generate(yesterday_str)
                                digest_generator.save_digest(digest)
                                logger.info("Daily digest generated for %s", yesterday_str)
                            except Exception:
                                logger.debug("Digest generation failed", exc_info=True)
                    except Exception:
                        logger.debug("Daily batch failed", exc_info=True)
                    _last_daily_batch = today_str

                # Staleness check — once per day (86400s)
                if _now_mono - _last_staleness_check > 86400:
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
                    except Exception:
                        logger.debug("Staleness check failed", exc_info=True)
                    _last_staleness_check = _now_mono

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

    # Stop query API server if running
    if query_api_server is not None:
        try:
            query_api_server.stop()
            logger.info("Query API server stopped")
        except Exception:
            pass

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

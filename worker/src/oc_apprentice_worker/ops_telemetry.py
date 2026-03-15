"""Operational telemetry — local-first, privacy-safe pipeline metrics.

Records per-batch pipeline metrics (counts and timings only, no user data)
to daily JSON files for trend analysis and health monitoring.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Metrics for a single processing batch."""
    timestamp: str = ""
    annotation_count: int = 0
    annotation_time_seconds: float = 0.0
    segmentation_count: int = 0
    segmentation_time_seconds: float = 0.0
    sop_generation_count: int = 0
    sop_generation_time_seconds: float = 0.0
    classification_count: int = 0
    continuity_spans_active: int = 0
    curation_items_pending: int = 0
    false_ready_rejections: int = 0
    review_conversions: int = 0
    drift_signals_new: int = 0
    execution_success_count: int = 0
    execution_failure_count: int = 0

class OpsTelemetry:
    """Local-first telemetry that never leaves the machine."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        self._telemetry_dir = kb.root / "observations" / "telemetry"

    def record_batch(self, metrics: PipelineMetrics) -> None:
        """Append a batch of metrics to today's telemetry file."""
        if not metrics.timestamp:
            metrics.timestamp = datetime.now(timezone.utc).isoformat()

        self._telemetry_dir.mkdir(parents=True, exist_ok=True)
        date_str = metrics.timestamp[:10]  # YYYY-MM-DD
        path = self._telemetry_dir / f"{date_str}.json"

        # Load existing entries
        entries: list[dict] = []
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                entries = data.get("entries", [])
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append(asdict(metrics))

        self._kb.atomic_write_json(path, {
            "date": date_str,
            "entries": entries,
            "entry_count": len(entries),
        })

    def get_daily_summary(self, date: str) -> dict:
        """Aggregate metrics for a specific date."""
        path = self._telemetry_dir / f"{date}.json"
        if not path.exists():
            return {"date": date, "entries": 0}

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"date": date, "entries": 0}

        entries = data.get("entries", [])
        if not entries:
            return {"date": date, "entries": 0}

        # Sum all numeric fields
        summary = {"date": date, "entries": len(entries)}
        numeric_fields = [f_name for f_name in PipelineMetrics.__dataclass_fields__ if f_name != "timestamp"]
        for field_name in numeric_fields:
            values = [e.get(field_name, 0) for e in entries]
            summary[f"total_{field_name}"] = sum(values)
        return summary

    def get_trend(self, days: int = 7) -> list[dict]:
        """Get daily summaries for the last N days."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        trend = []
        for i in range(days):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            summary = self.get_daily_summary(date)
            trend.append(summary)
        return trend

    def get_health_snapshot(self) -> dict:
        """Return current system health overview."""
        # Count procedures by lifecycle state
        procedures = self._kb.list_procedures()
        lifecycle_counts: dict[str, int] = {}
        freshness_values: list[float] = []

        from oc_apprentice_worker.staleness_detector import procedure_freshness

        for proc in procedures:
            state = proc.get("lifecycle_state", "observed")
            lifecycle_counts[state] = lifecycle_counts.get(state, 0) + 1
            freshness_values.append(procedure_freshness(proc))

        avg_freshness = sum(freshness_values) / len(freshness_values) if freshness_values else 0.0

        return {
            "procedures_total": len(procedures),
            "lifecycle_distribution": lifecycle_counts,
            "avg_freshness": round(avg_freshness, 4),
            "procedures_stale": lifecycle_counts.get("stale", 0),
            "procedures_agent_ready": lifecycle_counts.get("agent_ready", 0),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

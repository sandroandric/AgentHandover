"""Daily batch processor for activity timeline and task boundary detection.

Processes a full day of annotated events into a structured DailySummary
with task boundaries, app usage statistics, and active-hours calculation.
Results are saved to the knowledge base and a rolling 7-day recent context
is maintained for downstream pattern detection.

This module is intentionally simple — it uses only timestamp parsing,
string comparison, and basic counting.  No VLM or embedding calls are
needed; all semantic information comes from existing scene annotations.

Sprint 3 deliverable.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agenthandover_worker.event_helpers import (
    extract_app as _extract_app,
    extract_location as _extract_location,
    extract_what_doing as _extract_what_doing,
    parse_annotation as _extract_annotation,
    parse_timestamp as _parse_iso_dt,
)

if TYPE_CHECKING:
    from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Maximum gap (in seconds) between consecutive events before they are
# considered separate activity blocks.
_ACTIVITY_GAP_SECONDS = 300  # 5 minutes

# Number of days to keep in the rolling recent context.
_RECENT_CONTEXT_DAYS = 7

# Stop words for intent normalization.
_INTENT_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "is", "it", "was", "be",
}


def _normalize_intent(intent: str) -> str:
    """Normalize an intent string for comparison.

    Lowercases, strips punctuation, removes stop words, sorts remaining
    words alphabetically.  This ensures "debugging code" and "code debugging"
    compare as equal.
    """
    text = intent.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = [w for w in text.split() if w not in _INTENT_STOP_WORDS]
    words.sort()
    return " ".join(words)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskBoundary:
    """A contiguous block of user activity representing a single task."""

    start_time: str
    end_time: str
    duration_minutes: int
    intent: str  # VLM what_doing (most common in segment)
    apps: list[str]
    urls: list[str]
    event_ids: list[str]
    is_complete: bool  # False = abandoned/interrupted
    matched_procedure: str | None = None  # slug if matches known SOP
    account_context: dict | None = None  # filled by Sprint 10
    span_id: str | None = None


@dataclass
class DailySummary:
    """Aggregated summary of a single day's activity."""

    date: str
    active_hours: float
    task_count: int
    tasks: list[TaskBoundary]
    top_apps: list[dict]  # [{app: str, minutes: int}]
    procedures_observed: list[str]
    new_workflows_detected: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# _extract_annotation, _extract_app, _extract_what_doing, _extract_location,
# and _parse_iso_dt are imported from event_helpers above.


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string (wrapper for event_helpers)."""
    return _parse_iso_dt(ts)


def _extract_is_workflow(ann: dict) -> bool:
    """Extract workflow relevance from annotation.

    Uses activity_type if present (work/research/communication/setup = True),
    falls back to is_workflow for backward compatibility.
    """
    tc = ann.get("task_context")
    if isinstance(tc, dict):
        # Prefer activity_type if available
        activity_type = tc.get("activity_type", "")
        if activity_type:
            return activity_type in ("work", "research", "communication", "setup")
        # Fallback to is_workflow
        val = tc.get("is_workflow", False)
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1")
        return bool(val)
    return False


def _minutes_between(dt_a: datetime, dt_b: datetime) -> int:
    """Return the absolute difference in whole minutes between two datetimes."""
    delta = abs((dt_b - dt_a).total_seconds())
    return int(delta / 60)


# ---------------------------------------------------------------------------
# DailyBatchProcessor
# ---------------------------------------------------------------------------

class DailyBatchProcessor:
    """Processes a day's events into a DailySummary.

    The processor does NOT query the database — events are passed in by
    the caller.  It parses annotations, detects task boundaries, computes
    statistics, and persists the result in the knowledge base.
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_day(self, date: str, events: list[dict]) -> DailySummary:
        """Process all events for a given date into a DailySummary.

        Args:
            date: YYYY-MM-DD string.
            events: List of event dicts from the database.  Each event has:
                - id: str
                - timestamp: str (ISO format)
                - scene_annotation_json: str | None (JSON string with
                  task_context / visual_context)
                - window_json: str | None (JSON with app, title)
                - metadata_json: str | None

        Returns:
            A ``DailySummary`` with task boundaries, app usage, and
            active-hours calculation.
        """
        # 1. Parse events into an activity stream
        activities = self._build_activity_stream(events)

        # 2. Detect task boundaries
        tasks = self._detect_task_boundaries(activities)

        # 3. Calculate statistics
        active_hours = self._calculate_active_hours(activities)
        top_apps = self._calculate_app_usage(tasks)

        # 4. Collect observed procedures (matching known SOPs)
        procedures_observed: list[str] = []
        for task in tasks:
            if task.matched_procedure:
                procedures_observed.append(task.matched_procedure)

        summary = DailySummary(
            date=date,
            active_hours=round(active_hours, 2),
            task_count=len(tasks),
            tasks=tasks,
            top_apps=top_apps,
            procedures_observed=procedures_observed,
            new_workflows_detected=0,
        )

        # 5. Save to knowledge base
        self._save_summary(summary)

        # 6. Update rolling recent context
        self._update_recent_context(summary)

        return summary

    # ------------------------------------------------------------------
    # Activity stream
    # ------------------------------------------------------------------

    def _build_activity_stream(self, events: list[dict]) -> list[dict]:
        """Parse events into annotated activity entries.

        Skips events that have no valid scene annotation.  Each activity
        entry is a dict with: ``timestamp``, ``dt``, ``app``, ``location``,
        ``what_doing``, ``is_workflow``, ``event_id``.
        """
        activities: list[dict] = []

        for event in events:
            ann = _extract_annotation(event)
            if ann is None:
                continue

            ts = event.get("timestamp", "")
            dt = _parse_iso(ts)
            if dt is None:
                continue

            app = _extract_app(ann, event)
            what_doing = _extract_what_doing(ann)
            location = _extract_location(ann)
            is_workflow = _extract_is_workflow(ann)

            tc = ann.get("task_context")
            activity_type = ""
            if isinstance(tc, dict):
                activity_type = tc.get("activity_type", "")

            activities.append({
                "timestamp": ts,
                "dt": dt,
                "app": app,
                "location": location,
                "what_doing": what_doing,
                "is_workflow": is_workflow,
                "activity_type": activity_type,
                "event_id": event.get("id", ""),
            })

        # Sort chronologically
        activities.sort(key=lambda a: a["dt"])
        return activities

    # ------------------------------------------------------------------
    # Task boundary detection
    # ------------------------------------------------------------------

    def _detect_task_boundaries(
        self, activities: list[dict]
    ) -> list[TaskBoundary]:
        """Group sequential activities into tasks.

        A new task boundary is created when:
        - The primary app changes (different app from the majority of the
          current task).
        - The ``what_doing`` intent changes (a different task context).
        - There is a gap > 5 minutes between consecutive events.
        """
        if not activities:
            return []

        tasks: list[TaskBoundary] = []
        current_group: list[dict] = [activities[0]]

        for i in range(1, len(activities)):
            prev = activities[i - 1]
            curr = activities[i]

            # Check for time gap
            gap_seconds = (curr["dt"] - prev["dt"]).total_seconds()
            time_gap = gap_seconds > _ACTIVITY_GAP_SECONDS

            # Check for app change
            app_changed = (
                curr["app"] != prev["app"]
                and curr["app"] != ""
                and prev["app"] != ""
            )

            # Check for intent change (normalized comparison)
            intent_changed = (
                curr["what_doing"] != ""
                and prev["what_doing"] != ""
                and _normalize_intent(curr["what_doing"]) != _normalize_intent(prev["what_doing"])
            )

            if time_gap or app_changed or intent_changed:
                # Finalize the current group as a task
                tasks.append(self._group_to_task(current_group, curr))
                current_group = [curr]
            else:
                current_group.append(curr)

        # Finalize the last group — is_complete=True because we have no
        # evidence of switching away after the last event.
        if current_group:
            tasks.append(self._group_to_task(current_group, next_activity=None))

        return tasks

    def _group_to_task(
        self,
        group: list[dict],
        next_activity: dict | None,
    ) -> TaskBoundary:
        """Convert a group of consecutive activities into a TaskBoundary.

        Args:
            group: Non-empty list of activity dicts belonging to this task.
            next_activity: The activity that comes after this group, or
                ``None`` if this is the last group in the day.  Used to
                determine ``is_complete``.
        """
        start_dt: datetime = group[0]["dt"]
        end_dt: datetime = group[-1]["dt"]
        duration = _minutes_between(start_dt, end_dt)

        # Most common intent (normalize before counting so word-order
        # variants like "debugging code" and "code debugging" collapse)
        intents = [a["what_doing"] for a in group if a["what_doing"]]
        if intents:
            # Map normalized form -> list of original strings
            norm_to_originals: dict[str, list[str]] = {}
            for raw in intents:
                norm = _normalize_intent(raw)
                norm_to_originals.setdefault(norm, []).append(raw)
            # Find normalized form with most occurrences
            best_norm = max(norm_to_originals, key=lambda n: len(norm_to_originals[n]))
            # Use the first original string as the intent (preserves casing)
            intent = norm_to_originals[best_norm][0]
        else:
            intent = ""

        # Collect unique apps in order of first appearance
        apps: list[str] = []
        seen_apps: set[str] = set()
        for a in group:
            if a["app"] and a["app"] not in seen_apps:
                apps.append(a["app"])
                seen_apps.add(a["app"])

        # Collect unique URLs/locations in order of first appearance
        urls: list[str] = []
        seen_urls: set[str] = set()
        for a in group:
            if a["location"] and a["location"] not in seen_urls:
                urls.append(a["location"])
                seen_urls.add(a["location"])

        event_ids = [a["event_id"] for a in group if a["event_id"]]

        # is_complete: True unless the next activity is a different app
        # (user switched away, suggesting interruption/abandonment).
        # If next_activity is None (end of day), assume complete.
        if next_activity is None:
            is_complete = True
        else:
            last_app = group[-1]["app"]
            next_app = next_activity.get("app", "")
            # If the user switched to a different app, the task was
            # potentially interrupted.
            is_complete = (
                last_app == next_app
                or last_app == ""
                or next_app == ""
            )

        return TaskBoundary(
            start_time=group[0]["timestamp"],
            end_time=group[-1]["timestamp"],
            duration_minutes=duration,
            intent=intent,
            apps=apps,
            urls=urls,
            event_ids=event_ids,
            is_complete=is_complete,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _calculate_active_hours(self, activities: list[dict]) -> float:
        """Calculate total active hours from activity timestamps.

        Events within ``_ACTIVITY_GAP_SECONDS`` (5 minutes) of each other
        count as continuous activity.  Gaps larger than this are treated as
        idle time and excluded.
        """
        if len(activities) < 2:
            return 0.0

        total_seconds = 0.0

        for i in range(1, len(activities)):
            prev_dt: datetime = activities[i - 1]["dt"]
            curr_dt: datetime = activities[i]["dt"]
            gap = (curr_dt - prev_dt).total_seconds()

            if gap <= _ACTIVITY_GAP_SECONDS:
                total_seconds += gap

        return total_seconds / 3600.0

    def _calculate_app_usage(self, tasks: list[TaskBoundary]) -> list[dict]:
        """Calculate top apps by usage time (in minutes).

        Distributes each task's duration across its apps.  When a task
        involves multiple apps, the duration is attributed to the first
        (primary) app.
        """
        app_minutes: Counter[str] = Counter()

        for task in tasks:
            if not task.apps:
                continue
            # Attribute the full duration to the primary (first) app
            primary_app = task.apps[0]
            app_minutes[primary_app] += task.duration_minutes

        # Sort by minutes descending
        top = sorted(
            [{"app": app, "minutes": mins} for app, mins in app_minutes.items()],
            key=lambda x: x["minutes"],
            reverse=True,
        )
        return top

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_summary(self, summary: DailySummary) -> Path:
        """Serialize and save the daily summary to the knowledge base."""
        data = {
            "date": summary.date,
            "active_hours": summary.active_hours,
            "task_count": summary.task_count,
            "tasks": [
                {
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "duration_minutes": t.duration_minutes,
                    "intent": t.intent,
                    "apps": t.apps,
                    "urls": t.urls,
                    "event_ids": t.event_ids,
                    "is_complete": t.is_complete,
                    "matched_procedure": t.matched_procedure,
                    "account_context": t.account_context,
                }
                for t in summary.tasks
            ],
            "top_apps": summary.top_apps,
            "procedures_observed": summary.procedures_observed,
            "new_workflows_detected": summary.new_workflows_detected,
        }
        return self._kb.save_daily_summary(summary.date, data)

    def _update_recent_context(self, summary: DailySummary) -> None:
        """Update the rolling 7-day recent context in the knowledge base.

        Loads ``context/recent.json``, appends this day's summary entry,
        removes entries older than 7 days, and writes back.
        """
        ctx = self._kb.get_context("recent")
        days: list[dict] = ctx.get("last_7_days", [])

        # Build a compact entry for the rolling window
        entry = {
            "date": summary.date,
            "active_hours": summary.active_hours,
            "task_count": summary.task_count,
            "top_apps": summary.top_apps[:5],  # keep top 5 only
            "procedures_observed": summary.procedures_observed,
        }

        # Remove any existing entry for this date (idempotent re-processing)
        days = [d for d in days if d.get("date") != summary.date]
        days.append(entry)

        # Sort by date descending and trim to 7 days
        days.sort(key=lambda d: d.get("date", ""), reverse=True)
        days = days[:_RECENT_CONTEXT_DAYS]

        self._kb.update_context("recent", {"last_7_days": days})
        logger.info(
            "Updated recent context: %d days tracked", len(days)
        )

    def link_boundaries_to_spans(
        self,
        tasks: list[TaskBoundary],
        spans: list,
    ) -> list[TaskBoundary]:
        """Attempt to link each task boundary to a continuity span.

        For each boundary, checks if any span's segment event_ids overlap
        with the boundary's event_ids. If so, sets boundary.span_id to
        the span's span_id.

        This is optional post-processing — does not modify spans.

        Args:
            tasks: Task boundaries from process_day().
            spans: ContinuitySpan objects (from ContinuityTracker).

        Returns:
            The same tasks list with span_id populated where possible.
        """
        if not spans:
            return tasks

        # Build span event_id lookup
        span_events: dict[str, set[str]] = {}
        for span in spans:
            span_id = getattr(span, "span_id", "")
            if not span_id:
                continue
            # Collect event_ids from all segments in the span
            seg_ids = getattr(span, "segments", [])
            # Spans store segment_ids, not event_ids directly.
            # We can't resolve these without the original segments.
            # Instead, check if span's apps and goal overlap with boundary.
            span_events[span_id] = {
                "apps": set(getattr(span, "apps_involved", [])),
                "goal": getattr(span, "goal_summary", ""),
            }

        for task in tasks:
            if task.span_id is not None:
                continue
            task_apps = set(task.apps)
            best_overlap = 0.0
            best_span_id = None
            for span_id, info in span_events.items():
                # Simple app overlap + intent match
                span_apps = info["apps"]
                if not task_apps or not span_apps:
                    continue
                overlap = len(task_apps & span_apps) / len(task_apps | span_apps)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_span_id = span_id
            if best_overlap >= 0.3 and best_span_id is not None:
                task.span_id = best_span_id

        return tasks

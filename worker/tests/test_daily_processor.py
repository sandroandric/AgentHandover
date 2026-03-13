"""Tests for the daily batch processor (Sprint 3).

Covers: activity stream parsing, task boundary detection, active-hours
calculation, app usage statistics, knowledge base persistence, and
rolling recent context updates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from oc_apprentice_worker.daily_processor import (
    DailyBatchProcessor,
    DailySummary,
    TaskBoundary,
    _extract_annotation,
    _extract_app,
    _extract_is_workflow,
    _extract_location,
    _extract_what_doing,
    _minutes_between,
    _parse_iso,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def processor(kb: KnowledgeBase) -> DailyBatchProcessor:
    """Create a DailyBatchProcessor backed by a temp knowledge base."""
    return DailyBatchProcessor(kb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    event_id: str,
    timestamp: str,
    app: str,
    what_doing: str,
    location: str = "",
    is_workflow: bool = True,
) -> dict:
    """Create a mock event dict with a valid scene annotation.

    Uses the v2 annotation format (visual_context / task_context).
    """
    annotation = {
        "task_context": {
            "what_doing": what_doing,
            "is_workflow": is_workflow,
        },
        "visual_context": {
            "active_app": app,
            "location": location,
        },
    }
    return {
        "id": event_id,
        "timestamp": timestamp,
        "scene_annotation_json": json.dumps(annotation),
        "window_json": json.dumps({"app": app, "title": f"{app} Window"}),
        "metadata_json": "{}",
    }


def make_event_v1(
    event_id: str,
    timestamp: str,
    app: str,
    what_doing: str,
    location: str = "",
    is_workflow: bool = True,
) -> dict:
    """Create a mock event with v1 annotation format (top-level app/location)."""
    annotation = {
        "app": app,
        "location": location,
        "task_context": {
            "what_doing": what_doing,
            "is_workflow": is_workflow,
        },
    }
    return {
        "id": event_id,
        "timestamp": timestamp,
        "scene_annotation_json": json.dumps(annotation),
        "window_json": json.dumps({"app": app, "title": f"{app} Window"}),
        "metadata_json": "{}",
    }


# ---------------------------------------------------------------------------
# Tests: Internal helpers
# ---------------------------------------------------------------------------

class TestParseISO:

    def test_basic_z_suffix(self) -> None:
        dt = _parse_iso("2026-03-10T09:00:00Z")
        assert dt is not None
        assert dt.hour == 9
        assert dt.tzinfo is not None

    def test_offset_format(self) -> None:
        dt = _parse_iso("2026-03-10T09:00:00+02:00")
        assert dt is not None

    def test_empty_string(self) -> None:
        assert _parse_iso("") is None

    def test_invalid_string(self) -> None:
        assert _parse_iso("not-a-date") is None

    def test_none_input(self) -> None:
        assert _parse_iso(None) is None  # type: ignore[arg-type]


class TestExtractAnnotation:

    def test_valid_json_string(self) -> None:
        event = {"scene_annotation_json": '{"task_context": {}}'}
        ann = _extract_annotation(event)
        assert ann is not None
        assert "task_context" in ann

    def test_already_parsed_dict(self) -> None:
        event = {"scene_annotation_json": {"task_context": {}}}
        ann = _extract_annotation(event)
        assert ann is not None

    def test_none(self) -> None:
        assert _extract_annotation({"scene_annotation_json": None}) is None

    def test_missing_key(self) -> None:
        assert _extract_annotation({}) is None

    def test_invalid_json(self) -> None:
        assert _extract_annotation({"scene_annotation_json": "{{bad"}) is None

    def test_non_dict_json(self) -> None:
        assert _extract_annotation({"scene_annotation_json": "[1,2]"}) is None


class TestExtractApp:

    def test_v1_top_level(self) -> None:
        ann = {"app": "Chrome", "visual_context": {"active_app": "Firefox"}}
        assert _extract_app(ann, {}) == "Chrome"

    def test_v2_visual_context(self) -> None:
        ann = {"visual_context": {"active_app": "Firefox"}}
        assert _extract_app(ann, {}) == "Firefox"

    def test_fallback_window_json(self) -> None:
        ann = {}
        event = {"window_json": json.dumps({"app": "Terminal"})}
        assert _extract_app(ann, event) == "Terminal"

    def test_no_app(self) -> None:
        assert _extract_app({}, {}) == ""


class TestExtractWhatDoing:

    def test_normal(self) -> None:
        ann = {"task_context": {"what_doing": "Writing code"}}
        assert _extract_what_doing(ann) == "Writing code"

    def test_missing_task_context(self) -> None:
        assert _extract_what_doing({}) == ""

    def test_task_context_not_dict(self) -> None:
        assert _extract_what_doing({"task_context": "nope"}) == ""


class TestExtractLocation:

    def test_v1_top_level(self) -> None:
        ann = {"location": "https://example.com"}
        assert _extract_location(ann) == "https://example.com"

    def test_v2_visual_context(self) -> None:
        ann = {"visual_context": {"location": "https://test.com"}}
        assert _extract_location(ann) == "https://test.com"

    def test_no_location(self) -> None:
        assert _extract_location({}) == ""


class TestExtractIsWorkflow:

    def test_true(self) -> None:
        ann = {"task_context": {"is_workflow": True}}
        assert _extract_is_workflow(ann) is True

    def test_false(self) -> None:
        ann = {"task_context": {"is_workflow": False}}
        assert _extract_is_workflow(ann) is False

    def test_string_true(self) -> None:
        ann = {"task_context": {"is_workflow": "true"}}
        assert _extract_is_workflow(ann) is True

    def test_missing(self) -> None:
        assert _extract_is_workflow({}) is False


class TestMinutesBetween:

    def test_same_time(self) -> None:
        dt = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc)
        assert _minutes_between(dt, dt) == 0

    def test_ten_minutes(self) -> None:
        a = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc)
        b = datetime(2026, 3, 10, 9, 10, tzinfo=timezone.utc)
        assert _minutes_between(a, b) == 10

    def test_order_independent(self) -> None:
        a = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc)
        b = datetime(2026, 3, 10, 9, 30, tzinfo=timezone.utc)
        assert _minutes_between(a, b) == _minutes_between(b, a)


# ---------------------------------------------------------------------------
# Tests: Empty / single event
# ---------------------------------------------------------------------------

class TestEmptyAndSingleEvent:

    def test_empty_events_returns_empty_summary(
        self, processor: DailyBatchProcessor
    ) -> None:
        summary = processor.process_day("2026-03-10", [])
        assert summary.date == "2026-03-10"
        assert summary.task_count == 0
        assert summary.tasks == []
        assert summary.active_hours == 0.0
        assert summary.top_apps == []
        assert summary.procedures_observed == []
        assert summary.new_workflows_detected == 0

    def test_single_event_produces_one_task(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Browsing docs"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 1
        assert len(summary.tasks) == 1
        task = summary.tasks[0]
        assert task.intent == "Browsing docs"
        assert task.apps == ["Chrome"]
        assert task.event_ids == ["e1"]
        assert task.is_complete is True


# ---------------------------------------------------------------------------
# Tests: Task boundary detection
# ---------------------------------------------------------------------------

class TestTaskBoundaryDetection:

    def test_same_app_same_intent_one_task(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Multiple events with the same app and intent form one task."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading email"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", "Reading email"),
            make_event("e3", "2026-03-10T09:02:00Z", "Chrome", "Reading email"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 1
        assert summary.tasks[0].event_ids == ["e1", "e2", "e3"]

    def test_app_change_creates_new_boundary(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Switching apps creates a new task boundary."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "VS Code", "Coding"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 2
        assert summary.tasks[0].apps == ["Chrome"]
        assert summary.tasks[1].apps == ["VS Code"]

    def test_time_gap_creates_new_boundary(
        self, processor: DailyBatchProcessor
    ) -> None:
        """A gap of more than 5 minutes creates a new task boundary."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:10:00Z", "Chrome", "Reading"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 2

    def test_time_gap_exactly_5min_stays_same_task(
        self, processor: DailyBatchProcessor
    ) -> None:
        """A gap of exactly 5 minutes (300 seconds) stays in the same task."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:05:00Z", "Chrome", "Reading"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 1

    def test_intent_change_creates_new_boundary(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Changing what_doing intent creates a new task boundary."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading email"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", "Writing report"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 2
        assert summary.tasks[0].intent == "Reading email"
        assert summary.tasks[1].intent == "Writing report"

    def test_empty_app_does_not_trigger_app_change(
        self, processor: DailyBatchProcessor
    ) -> None:
        """An event with empty app does not trigger an app-change boundary."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "", "Reading"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 1

    def test_empty_intent_does_not_trigger_intent_change(
        self, processor: DailyBatchProcessor
    ) -> None:
        """An event with empty what_doing does not trigger an intent boundary."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", ""),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 1


# ---------------------------------------------------------------------------
# Tests: Task boundary properties
# ---------------------------------------------------------------------------

class TestTaskBoundaryProperties:

    def test_duration_calculation(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
            make_event("e2", "2026-03-10T09:04:00Z", "Chrome", "Working"),
            make_event("e3", "2026-03-10T09:05:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].duration_minutes == 5

    def test_most_common_intent(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Intent is the most common what_doing value in the task."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", "Reading"),
            make_event("e3", "2026-03-10T09:02:00Z", "Chrome", "Reading"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].intent == "Reading"

    def test_is_complete_true_for_last_task(
        self, processor: DailyBatchProcessor
    ) -> None:
        """The last task of the day is marked is_complete=True."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "VS Code", "Coding"),
        ]
        summary = processor.process_day("2026-03-10", events)
        # Last task is always complete (end of day)
        assert summary.tasks[-1].is_complete is True

    def test_is_complete_false_when_switched_away(
        self, processor: DailyBatchProcessor
    ) -> None:
        """A task is incomplete if the user switched to a different app."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "VS Code", "Coding"),
        ]
        summary = processor.process_day("2026-03-10", events)
        # First task: Chrome → VS Code = potentially interrupted
        assert summary.tasks[0].is_complete is False

    def test_is_complete_true_same_app_transition(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Task is complete if the next task uses the same app."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading email"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", "Searching"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].is_complete is True

    def test_urls_collected(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event(
                "e1", "2026-03-10T09:00:00Z", "Chrome", "Browsing",
                location="https://example.com",
            ),
            make_event(
                "e2", "2026-03-10T09:01:00Z", "Chrome", "Browsing",
                location="https://test.com",
            ),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert "https://example.com" in summary.tasks[0].urls
        assert "https://test.com" in summary.tasks[0].urls

    def test_urls_deduplicated(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event(
                "e1", "2026-03-10T09:00:00Z", "Chrome", "Browsing",
                location="https://example.com",
            ),
            make_event(
                "e2", "2026-03-10T09:01:00Z", "Chrome", "Browsing",
                location="https://example.com",
            ),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].urls == ["https://example.com"]


# ---------------------------------------------------------------------------
# Tests: Active hours calculation
# ---------------------------------------------------------------------------

class TestActiveHours:

    def test_continuous_activity(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events within 5 min of each other count as continuous activity."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
            make_event("e2", "2026-03-10T09:05:00Z", "Chrome", "Working"),
            make_event("e3", "2026-03-10T10:00:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        # 9:00-9:05 = 5 min continuous, 9:05-10:00 = 55 min gap (excluded)
        expected = 5.0 / 60.0
        assert abs(summary.active_hours - expected) < 0.01

    def test_zero_active_hours_single_event(
        self, processor: DailyBatchProcessor
    ) -> None:
        """A single event has 0 active hours (no interval to measure)."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.active_hours == 0.0

    def test_one_hour_continuous(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events every minute for 60 minutes = 1 hour active."""
        events = []
        for i in range(61):
            hour = 9 + i // 60
            minute = i % 60
            events.append(
                make_event(
                    f"e{i}",
                    f"2026-03-10T{hour:02d}:{minute:02d}:00Z",
                    "Chrome",
                    "Working",
                )
            )
        summary = processor.process_day("2026-03-10", events)
        assert abs(summary.active_hours - 1.0) < 0.01

    def test_gap_excluded_from_active_hours(
        self, processor: DailyBatchProcessor
    ) -> None:
        """A 30-minute gap is excluded from active hours."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
            make_event("e2", "2026-03-10T09:02:00Z", "Chrome", "Working"),
            # 30-minute gap
            make_event("e3", "2026-03-10T09:32:00Z", "Chrome", "Working"),
            make_event("e4", "2026-03-10T09:34:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        # 2 min + 2 min = 4 min
        expected = 4.0 / 60.0
        assert abs(summary.active_hours - expected) < 0.01


# ---------------------------------------------------------------------------
# Tests: App usage
# ---------------------------------------------------------------------------

class TestAppUsage:

    def test_top_apps_sorted_by_minutes(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Browsing"),
            make_event("e2", "2026-03-10T09:04:00Z", "Chrome", "Browsing"),
            make_event("e3", "2026-03-10T09:05:00Z", "VS Code", "Coding"),
            make_event("e4", "2026-03-10T09:06:00Z", "VS Code", "Coding"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert len(summary.top_apps) >= 1
        # Chrome: 4 min, VS Code: 1 min
        chrome = next(a for a in summary.top_apps if a["app"] == "Chrome")
        vscode = next(a for a in summary.top_apps if a["app"] == "VS Code")
        assert chrome["minutes"] >= vscode["minutes"]

    def test_empty_tasks_empty_apps(
        self, processor: DailyBatchProcessor
    ) -> None:
        summary = processor.process_day("2026-03-10", [])
        assert summary.top_apps == []


# ---------------------------------------------------------------------------
# Tests: Events without/invalid annotations
# ---------------------------------------------------------------------------

class TestAnnotationEdgeCases:

    def test_events_without_annotations_skipped(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events with no scene_annotation_json are ignored."""
        events = [
            {
                "id": "e1",
                "timestamp": "2026-03-10T09:00:00Z",
                "scene_annotation_json": None,
                "window_json": "{}",
                "metadata_json": "{}",
            },
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 0

    def test_events_with_invalid_json_skipped(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events with malformed annotation JSON are ignored."""
        events = [
            {
                "id": "e1",
                "timestamp": "2026-03-10T09:00:00Z",
                "scene_annotation_json": "{{invalid json",
                "window_json": "{}",
                "metadata_json": "{}",
            },
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 0

    def test_mixed_valid_and_invalid(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Only valid events produce tasks; invalid ones are silently skipped."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            {
                "id": "e2",
                "timestamp": "2026-03-10T09:01:00Z",
                "scene_annotation_json": "not json",
                "window_json": "{}",
                "metadata_json": "{}",
            },
            make_event("e3", "2026-03-10T09:02:00Z", "Chrome", "Reading"),
        ]
        summary = processor.process_day("2026-03-10", events)
        # e1 and e3 are valid with same app/intent within 5 min
        assert summary.task_count == 1
        assert len(summary.tasks[0].event_ids) == 2

    def test_event_without_timestamp_skipped(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events with unparseable timestamps are skipped."""
        annotation = json.dumps({
            "task_context": {"what_doing": "Test", "is_workflow": True},
            "visual_context": {"active_app": "Chrome"},
        })
        events = [
            {
                "id": "e1",
                "timestamp": "not-a-timestamp",
                "scene_annotation_json": annotation,
                "window_json": "{}",
                "metadata_json": "{}",
            },
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 0


# ---------------------------------------------------------------------------
# Tests: Knowledge base persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_summary_saved_to_knowledge_base(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        processor.process_day("2026-03-10", events)
        loaded = kb.get_daily_summary("2026-03-10")
        assert loaded is not None
        assert loaded["date"] == "2026-03-10"
        assert loaded["task_count"] == 1
        assert len(loaded["tasks"]) == 1

    def test_summary_overwritten_on_reprocess(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        """Re-processing the same day overwrites the previous summary."""
        events1 = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        processor.process_day("2026-03-10", events1)

        events2 = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", "Working"),
        ]
        processor.process_day("2026-03-10", events2)

        loaded = kb.get_daily_summary("2026-03-10")
        assert loaded is not None
        assert loaded["task_count"] == 1  # one task with 2 events

    def test_saved_tasks_have_all_fields(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event(
                "e1", "2026-03-10T09:00:00Z", "Chrome", "Browsing",
                location="https://example.com",
            ),
        ]
        processor.process_day("2026-03-10", events)
        loaded = kb.get_daily_summary("2026-03-10")
        task = loaded["tasks"][0]
        assert "start_time" in task
        assert "end_time" in task
        assert "duration_minutes" in task
        assert "intent" in task
        assert "apps" in task
        assert "urls" in task
        assert "event_ids" in task
        assert "is_complete" in task
        assert "matched_procedure" in task
        assert "account_context" in task


# ---------------------------------------------------------------------------
# Tests: Recent context
# ---------------------------------------------------------------------------

class TestRecentContext:

    def test_recent_context_updated(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        processor.process_day("2026-03-10", events)
        ctx = kb.get_context("recent")
        assert "last_7_days" in ctx
        assert len(ctx["last_7_days"]) == 1
        assert ctx["last_7_days"][0]["date"] == "2026-03-10"

    def test_recent_context_rolling_7_days(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        """Only the most recent 7 days are kept in the rolling context."""
        for day in range(1, 10):  # 9 days
            events = [
                make_event(
                    f"e{day}",
                    f"2026-03-{day:02d}T09:00:00Z",
                    "Chrome",
                    "Working",
                ),
            ]
            processor.process_day(f"2026-03-{day:02d}", events)

        ctx = kb.get_context("recent")
        days = ctx["last_7_days"]
        assert len(days) == 7
        # Should have days 3-9 (newest 7)
        dates = [d["date"] for d in days]
        assert "2026-03-09" in dates
        assert "2026-03-03" in dates
        assert "2026-03-02" not in dates
        assert "2026-03-01" not in dates

    def test_recent_context_idempotent_reprocess(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        """Re-processing the same day does not create duplicate entries."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        processor.process_day("2026-03-10", events)
        processor.process_day("2026-03-10", events)
        ctx = kb.get_context("recent")
        dates = [d["date"] for d in ctx["last_7_days"]]
        assert dates.count("2026-03-10") == 1

    def test_recent_context_sorted_newest_first(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        for day in [5, 3, 7, 1]:
            events = [
                make_event(
                    f"e{day}",
                    f"2026-03-{day:02d}T09:00:00Z",
                    "Chrome",
                    "Working",
                ),
            ]
            processor.process_day(f"2026-03-{day:02d}", events)

        ctx = kb.get_context("recent")
        dates = [d["date"] for d in ctx["last_7_days"]]
        assert dates == sorted(dates, reverse=True)

    def test_recent_context_entry_has_summary_fields(
        self, kb: KnowledgeBase, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        processor.process_day("2026-03-10", events)
        ctx = kb.get_context("recent")
        entry = ctx["last_7_days"][0]
        assert "active_hours" in entry
        assert "task_count" in entry
        assert "top_apps" in entry
        assert "procedures_observed" in entry


# ---------------------------------------------------------------------------
# Tests: Large batch
# ---------------------------------------------------------------------------

class TestLargeBatch:

    def test_large_batch_50_events_3_apps(
        self, processor: DailyBatchProcessor
    ) -> None:
        """50 events across 3 apps should produce multiple tasks."""
        events = []
        apps = ["Chrome", "VS Code", "Terminal"]
        intents = [
            "Browsing docs",
            "Writing code",
            "Running tests",
        ]
        for i in range(50):
            app_idx = i // 17  # ~17 events per app
            if app_idx >= 3:
                app_idx = 2
            minute = i
            events.append(
                make_event(
                    f"e{i}",
                    f"2026-03-10T09:{minute:02d}:00Z",
                    apps[app_idx],
                    intents[app_idx],
                )
            )

        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count >= 2  # at least 2 app transitions
        assert summary.task_count <= 50  # at most one per event
        total_events = sum(len(t.event_ids) for t in summary.tasks)
        assert total_events == 50


# ---------------------------------------------------------------------------
# Tests: process_day return value
# ---------------------------------------------------------------------------

class TestProcessDayReturn:

    def test_returns_daily_summary(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        result = processor.process_day("2026-03-10", events)
        assert isinstance(result, DailySummary)

    def test_correct_task_count(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Reading"),
            make_event("e2", "2026-03-10T09:01:00Z", "VS Code", "Coding"),
            make_event("e3", "2026-03-10T09:02:00Z", "Terminal", "Testing"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 3
        assert summary.task_count == len(summary.tasks)

    def test_procedures_observed_initially_empty(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.procedures_observed == []

    def test_new_workflows_detected_zero(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.new_workflows_detected == 0


# ---------------------------------------------------------------------------
# Tests: v1 annotation format compatibility
# ---------------------------------------------------------------------------

class TestV1AnnotationFormat:

    def test_v1_format_works(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events with v1 annotation format (top-level app) are processed."""
        events = [
            make_event_v1("e1", "2026-03-10T09:00:00Z", "Chrome", "Browsing"),
            make_event_v1("e2", "2026-03-10T09:01:00Z", "Chrome", "Browsing"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 1
        assert summary.tasks[0].apps == ["Chrome"]


# ---------------------------------------------------------------------------
# Tests: Edge cases in task grouping
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_out_of_order_events_sorted(
        self, processor: DailyBatchProcessor
    ) -> None:
        """Events passed out of order are sorted chronologically."""
        events = [
            make_event("e3", "2026-03-10T09:02:00Z", "Chrome", "Working"),
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
            make_event("e2", "2026-03-10T09:01:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].event_ids == ["e1", "e2", "e3"]

    def test_multiple_apps_in_single_task_minor_switches(
        self, processor: DailyBatchProcessor
    ) -> None:
        """App changes always create boundaries; no 'minor switch' merging."""
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
            make_event("e2", "2026-03-10T09:01:00Z", "Finder", "Working"),
            make_event("e3", "2026-03-10T09:02:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        # Each app change creates a new boundary
        assert summary.task_count == 3

    def test_all_events_invalid_returns_empty(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            {
                "id": "e1",
                "timestamp": "2026-03-10T09:00:00Z",
                "scene_annotation_json": "nope",
                "window_json": "{}",
                "metadata_json": "{}",
            },
            {
                "id": "e2",
                "timestamp": "2026-03-10T09:01:00Z",
                "scene_annotation_json": None,
                "window_json": "{}",
                "metadata_json": "{}",
            },
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count == 0
        assert summary.active_hours == 0.0

    def test_matched_procedure_none_by_default(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].matched_procedure is None

    def test_account_context_none_by_default(
        self, processor: DailyBatchProcessor
    ) -> None:
        events = [
            make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Working"),
        ]
        summary = processor.process_day("2026-03-10", events)
        assert summary.tasks[0].account_context is None

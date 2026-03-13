"""Tests for the profile builder (Sprint 7).

Covers: tool inference, working hours, account detection, communication
style, empty summaries, multi-day aggregation, and app classification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.profile_builder import ProfileBuilder


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
def builder(kb: KnowledgeBase) -> ProfileBuilder:
    return ProfileBuilder(kb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_daily_summary(
    date: str,
    tasks: list[dict] | None = None,
    top_apps: list[dict] | None = None,
    active_hours: float = 6.0,
) -> dict:
    return {
        "date": date,
        "active_hours": active_hours,
        "task_count": len(tasks) if tasks else 0,
        "tasks": tasks or [],
        "top_apps": top_apps or [],
        "procedures_observed": [],
        "new_workflows_detected": 0,
    }


def make_task(
    intent: str,
    apps: list[str],
    urls: list[str] | None = None,
    start_time: str = "2026-03-10T09:00:00Z",
    end_time: str = "2026-03-10T09:10:00Z",
    duration_minutes: int = 10,
    matched_procedure: str | None = None,
) -> dict:
    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration_minutes,
        "intent": intent,
        "apps": apps,
        "urls": urls or [],
        "event_ids": [],
        "is_complete": True,
        "matched_procedure": matched_procedure,
    }


def _populate_summaries(kb: KnowledgeBase, summaries: list[dict]) -> None:
    """Helper to save multiple daily summaries to the KB."""
    for s in summaries:
        kb.save_daily_summary(s["date"], s)


# ---------------------------------------------------------------------------
# Empty / default state
# ---------------------------------------------------------------------------

class TestProfileBuilderEmpty:

    def test_no_summaries_returns_default_profile(
        self, builder: ProfileBuilder
    ) -> None:
        profile = builder.update_profile()
        assert profile["tools"] == {}
        assert profile["working_hours"] == {}
        assert profile["accounts"] == []
        assert profile["communication_style"] == {}

    def test_no_summaries_does_not_write_profile(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        builder.update_profile()
        # Profile file should not have been created (no updated_at)
        profile = kb.get_profile()
        assert profile["updated_at"] is None


# ---------------------------------------------------------------------------
# Tool inference
# ---------------------------------------------------------------------------

class TestToolInference:

    def test_browser_detection(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Google Chrome", "minutes": 120},
                {"app": "Finder", "minutes": 10},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["browser"] == "Google Chrome"

    def test_editor_detection(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "VS Code", "minutes": 200},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["editor"] == "VS Code"

    def test_terminal_detection(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "iTerm2", "minutes": 60},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["terminal"] == "iTerm2"

    def test_all_three_tools_detected(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Firefox", "minutes": 100},
                {"app": "Sublime Text", "minutes": 80},
                {"app": "Warp", "minutes": 40},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["browser"] == "Firefox"
        assert profile["tools"]["editor"] == "Sublime Text"
        assert profile["tools"]["terminal"] == "Warp"

    def test_primary_apps_ordered_by_usage(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Chrome", "minutes": 200},
                {"app": "Slack", "minutes": 50},
                {"app": "VS Code", "minutes": 150},
            ]),
        ])
        profile = builder.update_profile()
        apps = profile["tools"]["primary_apps"]
        assert apps[0]["app"] == "Chrome"
        assert apps[0]["total_minutes"] == 200
        assert apps[1]["app"] == "VS Code"
        assert apps[2]["app"] == "Slack"

    def test_days_used_counted_correctly(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Chrome", "minutes": 100},
            ]),
            make_daily_summary("2026-03-02", top_apps=[
                {"app": "Chrome", "minutes": 80},
            ]),
            make_daily_summary("2026-03-03", top_apps=[
                {"app": "Slack", "minutes": 30},
            ]),
        ])
        profile = builder.update_profile()
        apps_by_name = {
            a["app"]: a for a in profile["tools"]["primary_apps"]
        }
        assert apps_by_name["Chrome"]["days_used"] == 2
        assert apps_by_name["Chrome"]["total_minutes"] == 180
        assert apps_by_name["Slack"]["days_used"] == 1

    def test_no_browser_when_no_match(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Photoshop", "minutes": 200},
            ]),
        ])
        profile = builder.update_profile()
        assert "browser" not in profile["tools"]
        assert "editor" not in profile["tools"]
        assert "terminal" not in profile["tools"]

    def test_empty_app_name_skipped(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "", "minutes": 100},
                {"app": "Chrome", "minutes": 50},
            ]),
        ])
        profile = builder.update_profile()
        # Empty app should not appear
        app_names = [a["app"] for a in profile["tools"]["primary_apps"]]
        assert "" not in app_names

    def test_xcode_detected_as_editor(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Xcode", "minutes": 180},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["editor"] == "Xcode"

    def test_cursor_detected_as_editor(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Cursor", "minutes": 180},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["editor"] == "Cursor"

    def test_arc_detected_as_browser(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Arc", "minutes": 100},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["browser"] == "Arc"

    def test_brave_detected_as_browser(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Brave Browser", "minutes": 100},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["tools"]["browser"] == "Brave Browser"


# ---------------------------------------------------------------------------
# Working hours inference
# ---------------------------------------------------------------------------

class TestWorkingHours:

    def test_typical_hours_from_tasks(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        tasks = [
            make_task("coding", ["VS Code"],
                      start_time="2026-03-01T09:00:00Z",
                      end_time="2026-03-01T09:30:00Z"),
            make_task("email", ["Mail"],
                      start_time="2026-03-01T16:00:00Z",
                      end_time="2026-03-01T17:00:00Z"),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks, active_hours=8.0),
        ])
        profile = builder.update_profile()
        wh = profile["working_hours"]
        assert wh["typical_start"] == "09:00"
        assert wh["typical_end"] == "17:00"

    def test_avg_active_hours(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", active_hours=6.0),
            make_daily_summary("2026-03-02", active_hours=8.0),
            make_daily_summary("2026-03-03", active_hours=7.0),
        ])
        profile = builder.update_profile()
        assert profile["working_hours"]["avg_active_hours"] == 7.0

    def test_weekend_active_true(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        # 2026-03-07 is a Saturday
        tasks = [
            make_task("weekend work", ["Chrome"],
                      start_time="2026-03-07T10:00:00Z",
                      end_time="2026-03-07T12:00:00Z"),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-07", tasks=tasks, active_hours=2.0),
        ])
        profile = builder.update_profile()
        assert profile["working_hours"]["weekend_active"] is True

    def test_weekend_active_false(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        # 2026-03-09 is a Monday
        tasks = [
            make_task("work", ["Chrome"],
                      start_time="2026-03-09T09:00:00Z",
                      end_time="2026-03-09T17:00:00Z"),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-09", tasks=tasks, active_hours=8.0),
        ])
        profile = builder.update_profile()
        assert profile["working_hours"]["weekend_active"] is False

    def test_no_tasks_gives_partial_result(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", active_hours=5.0),
        ])
        profile = builder.update_profile()
        wh = profile["working_hours"]
        assert "typical_start" not in wh
        assert "typical_end" not in wh
        assert wh["avg_active_hours"] == 5.0

    def test_multiple_days_averaged(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for day, start_h, end_h in [
            ("2026-03-02", 8, 16),
            ("2026-03-03", 10, 18),
        ]:
            tasks = [
                make_task("work", ["Chrome"],
                          start_time=f"{day}T{start_h:02d}:00:00Z",
                          end_time=f"{day}T{end_h:02d}:00:00Z"),
            ]
            summaries.append(make_daily_summary(day, tasks=tasks))
        _populate_summaries(kb, summaries)
        profile = builder.update_profile()
        wh = profile["working_hours"]
        assert wh["typical_start"] == "09:00"  # (8+10)//2
        assert wh["typical_end"] == "17:00"  # (16+18)//2


# ---------------------------------------------------------------------------
# Account detection
# ---------------------------------------------------------------------------

class TestAccountDetection:

    def test_github_detected(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        tasks = [
            make_task("PR review", ["Chrome"],
                      urls=["https://github.com/org/repo/pull/42"]),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks),
        ])
        profile = builder.update_profile()
        services = [a["service"] for a in profile["accounts"]]
        assert "github" in services

    def test_gmail_detected(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        tasks = [
            make_task("email", ["Chrome"],
                      urls=["https://mail.google.com/mail/u/0/"]),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks),
        ])
        profile = builder.update_profile()
        services = [a["service"] for a in profile["accounts"]]
        assert "gmail" in services

    def test_slack_detected(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        tasks = [
            make_task("chat", ["Chrome"],
                      urls=["https://app.slack.com/client/T123/C456"]),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks),
        ])
        profile = builder.update_profile()
        services = [a["service"] for a in profile["accounts"]]
        assert "slack" in services

    def test_frequency_daily(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        # GitHub on 8 out of 10 days = daily (80%)
        summaries = []
        for i in range(1, 11):
            tasks = []
            if i <= 8:
                tasks.append(make_task("PR", ["Chrome"],
                                       urls=["https://github.com/foo"]))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)
        profile = builder.update_profile()
        gh = next(a for a in profile["accounts"] if a["service"] == "github")
        assert gh["frequency"] == "daily"

    def test_frequency_weekly(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        # Figma on 4 out of 10 days = weekly (40%)
        summaries = []
        for i in range(1, 11):
            tasks = []
            if i <= 4:
                tasks.append(make_task("design", ["Chrome"],
                                       urls=["https://figma.com/file/x"]))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)
        profile = builder.update_profile()
        fig = next(a for a in profile["accounts"] if a["service"] == "figma")
        assert fig["frequency"] == "weekly"

    def test_frequency_occasional(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        # Stripe on 2 out of 10 days = occasional (20%)
        summaries = []
        for i in range(1, 11):
            tasks = []
            if i <= 2:
                tasks.append(make_task("billing", ["Chrome"],
                                       urls=["https://dashboard.stripe.com/"]))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)
        profile = builder.update_profile()
        stripe = next(
            a for a in profile["accounts"] if a["service"] == "stripe"
        )
        assert stripe["frequency"] == "occasional"

    def test_no_urls_no_accounts(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        tasks = [make_task("coding", ["VS Code"])]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks),
        ])
        profile = builder.update_profile()
        assert profile["accounts"] == []

    def test_multiple_services_detected(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        tasks = [
            make_task("code", ["Chrome"],
                      urls=["https://github.com/repo"]),
            make_task("chat", ["Chrome"],
                      urls=["https://app.slack.com/client"]),
            make_task("docs", ["Chrome"],
                      urls=["https://notion.so/page"]),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks),
        ])
        profile = builder.update_profile()
        services = {a["service"] for a in profile["accounts"]}
        assert services == {"github", "slack", "notion"}

    def test_service_counted_once_per_day(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        # Multiple GitHub URLs on the same day should count as 1 day
        tasks = [
            make_task("PR 1", ["Chrome"],
                      urls=["https://github.com/repo/pull/1"]),
            make_task("PR 2", ["Chrome"],
                      urls=["https://github.com/repo/pull/2"]),
        ]
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=tasks),
        ])
        profile = builder.update_profile()
        gh = next(a for a in profile["accounts"] if a["service"] == "github")
        # With 1 day total and 1 day of github usage, ratio = 1.0 = daily
        assert gh["frequency"] == "daily"


# ---------------------------------------------------------------------------
# Communication style
# ---------------------------------------------------------------------------

class TestCommunicationStyle:

    def test_slack_primary_channel(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Slack", "minutes": 60},
                {"app": "Chrome", "minutes": 200},
            ]),
        ])
        profile = builder.update_profile()
        cs = profile["communication_style"]
        assert "Slack" in cs["primary_channels"]

    def test_multiple_comm_apps(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Slack", "minutes": 60},
                {"app": "Mail", "minutes": 30},
                {"app": "Zoom", "minutes": 45},
            ]),
        ])
        profile = builder.update_profile()
        cs = profile["communication_style"]
        assert len(cs["primary_channels"]) == 3
        # Slack should be first (most minutes)
        assert cs["primary_channels"][0] == "Slack"

    def test_avg_comm_minutes(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Slack", "minutes": 60},
            ]),
            make_daily_summary("2026-03-02", top_apps=[
                {"app": "Slack", "minutes": 40},
            ]),
        ])
        profile = builder.update_profile()
        cs = profile["communication_style"]
        assert cs["avg_comm_minutes_per_day"] == 50.0

    def test_no_comm_apps(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Chrome", "minutes": 200},
                {"app": "VS Code", "minutes": 180},
            ]),
        ])
        profile = builder.update_profile()
        cs = profile["communication_style"]
        assert cs["primary_channels"] == []
        assert cs["avg_comm_minutes_per_day"] == 0.0

    def test_max_three_channels(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Slack", "minutes": 60},
                {"app": "Mail", "minutes": 30},
                {"app": "Zoom", "minutes": 45},
                {"app": "Discord", "minutes": 20},
                {"app": "Telegram", "minutes": 10},
            ]),
        ])
        profile = builder.update_profile()
        cs = profile["communication_style"]
        assert len(cs["primary_channels"]) == 3


# ---------------------------------------------------------------------------
# Multi-day aggregation
# ---------------------------------------------------------------------------

class TestMultiDayAggregation:

    def test_profile_aggregates_across_days(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Chrome", "minutes": 100},
            ]),
            make_daily_summary("2026-03-02", top_apps=[
                {"app": "Chrome", "minutes": 50},
                {"app": "VS Code", "minutes": 120},
            ]),
        ])
        profile = builder.update_profile()
        apps_by_name = {
            a["app"]: a for a in profile["tools"]["primary_apps"]
        }
        assert apps_by_name["Chrome"]["total_minutes"] == 150
        assert apps_by_name["Chrome"]["days_used"] == 2
        assert apps_by_name["VS Code"]["days_used"] == 1

    def test_profile_updated_at_set(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Chrome", "minutes": 100},
            ]),
        ])
        profile = builder.update_profile()
        assert profile["updated_at"] is not None

    def test_profile_persisted_to_kb(
        self, builder: ProfileBuilder, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", top_apps=[
                {"app": "Safari", "minutes": 100},
            ]),
        ])
        builder.update_profile()
        # Read directly from KB
        profile = kb.get_profile()
        assert profile["tools"]["browser"] == "Safari"

    def test_load_summaries_respects_limit(
        self, kb: KnowledgeBase
    ) -> None:
        # Save 40 summaries but builder default is 30
        for i in range(1, 32):
            kb.save_daily_summary(
                f"2026-01-{i:02d}",
                make_daily_summary(f"2026-01-{i:02d}"),
            )
        builder = ProfileBuilder(kb)
        loaded = builder._load_summaries(limit=10)
        assert len(loaded) == 10

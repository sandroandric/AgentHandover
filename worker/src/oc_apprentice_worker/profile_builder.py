"""Profile builder — infers user profile from accumulated daily summaries.

Reads daily summaries from the knowledge base, aggregates tool usage,
working hours, accounts, and communication style into a persistent
user profile that agents can consume.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """Infer user profile from accumulated daily summaries."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def update_profile(self) -> dict:
        """Rebuild the profile from all available daily summaries.

        Reads daily summaries from the knowledge base, aggregates
        tool usage, working hours, accounts, and communication style.
        Returns the updated profile dict.
        """
        summaries = self._load_summaries()
        if not summaries:
            return self._kb.get_profile()

        profile = {
            "tools": self._infer_tools(summaries),
            "working_hours": self._infer_working_hours(summaries),
            "accounts": self._infer_accounts(summaries),
            "communication_style": self._infer_communication_style(summaries),
        }
        self._kb.update_profile(profile)
        return self._kb.get_profile()

    def _load_summaries(self, limit: int = 30) -> list[dict]:
        """Load recent daily summaries from KB."""
        return self._kb.load_daily_summaries(limit=limit)

    def _infer_tools(self, summaries: list[dict]) -> dict:
        """Infer primary tools from top_apps across all summaries.

        Returns dict like::

            {
                "browser": "Google Chrome",
                "editor": "VS Code",
                "terminal": "Terminal",
                "primary_apps": [
                    {"app": "Chrome", "total_minutes": 500, "days_used": 15},
                ],
            }
        """
        # Aggregate app usage across all days
        app_minutes: Counter = Counter()
        app_days: Counter = Counter()

        for summary in summaries:
            seen_apps: set[str] = set()
            for app_entry in summary.get("top_apps", []):
                app = app_entry.get("app", "")
                minutes = app_entry.get("minutes", 0)
                if app:
                    app_minutes[app] += minutes
                    if app not in seen_apps:
                        app_days[app] += 1
                        seen_apps.add(app)

        primary_apps = [
            {"app": app, "total_minutes": minutes, "days_used": app_days[app]}
            for app, minutes in app_minutes.most_common(20)
        ]

        # Try to classify primary tools
        tools: dict = {"primary_apps": primary_apps}
        browser_keywords = (
            "chrome", "firefox", "safari", "edge", "brave", "arc",
        )
        editor_keywords = (
            "code", "vim", "neovim", "sublime", "atom",
            "intellij", "pycharm", "xcode", "cursor",
        )
        terminal_keywords = (
            "terminal", "iterm", "warp", "alacritty", "kitty", "hyper",
        )

        for app, _ in app_minutes.most_common():
            lower = app.lower()
            if "browser" not in tools and any(
                kw in lower for kw in browser_keywords
            ):
                tools["browser"] = app
            if "editor" not in tools and any(
                kw in lower for kw in editor_keywords
            ):
                tools["editor"] = app
            if "terminal" not in tools and any(
                kw in lower for kw in terminal_keywords
            ):
                tools["terminal"] = app

        return tools

    def _infer_working_hours(self, summaries: list[dict]) -> dict:
        """Infer typical working hours from daily summary tasks.

        Returns dict like::

            {
                "typical_start": "09:00",
                "typical_end": "17:30",
                "avg_active_hours": 6.5,
                "weekend_active": False,
            }
        """
        start_hours: list[int] = []
        end_hours: list[int] = []
        active_hours_list: list[float] = []
        weekend_days = 0

        for summary in summaries:
            active = summary.get("active_hours", 0)
            active_hours_list.append(active)

            tasks = summary.get("tasks", [])
            if not tasks:
                continue

            # Parse the earliest and latest task times
            try:
                first_time = tasks[0].get("start_time", "")
                last_time = tasks[-1].get("end_time", "")
                if first_time:
                    dt = datetime.fromisoformat(
                        first_time.replace("Z", "+00:00")
                    )
                    start_hours.append(dt.hour)
                if last_time:
                    dt = datetime.fromisoformat(
                        last_time.replace("Z", "+00:00")
                    )
                    end_hours.append(dt.hour)
            except (ValueError, TypeError):
                continue

            # Check weekend
            date_str = summary.get("date", "")
            if date_str:
                try:
                    day = datetime.strptime(date_str, "%Y-%m-%d").weekday()
                    if day >= 5:
                        weekend_days += 1
                except ValueError:
                    pass

        result: dict = {}
        if start_hours:
            avg_start = sum(start_hours) // len(start_hours)
            result["typical_start"] = f"{avg_start:02d}:00"
        if end_hours:
            avg_end = sum(end_hours) // len(end_hours)
            result["typical_end"] = f"{avg_end:02d}:00"
        if active_hours_list:
            result["avg_active_hours"] = round(
                sum(active_hours_list) / len(active_hours_list), 1
            )
        result["weekend_active"] = weekend_days > 0

        return result

    def _infer_accounts(self, summaries: list[dict]) -> list[dict]:
        """Infer accounts from URL patterns in tasks.

        Returns list like::

            [{"service": "github", "frequency": "daily"}, ...]
        """
        service_days: Counter = Counter()
        total_days = len(summaries)

        service_patterns = {
            "github": ["github.com"],
            "gmail": ["mail.google.com", "gmail.com"],
            "slack": ["slack.com", "app.slack.com"],
            "jira": ["atlassian.net", "jira"],
            "notion": ["notion.so"],
            "figma": ["figma.com"],
            "stripe": ["stripe.com", "dashboard.stripe.com"],
            "vercel": ["vercel.com"],
            "aws": ["console.aws.amazon.com", "aws.amazon.com"],
            "google_drive": ["drive.google.com", "docs.google.com"],
        }

        for summary in summaries:
            seen_services: set[str] = set()
            for task in summary.get("tasks", []):
                for url in task.get("urls", []):
                    url_lower = url.lower()
                    for service, patterns in service_patterns.items():
                        if service not in seen_services:
                            if any(p in url_lower for p in patterns):
                                service_days[service] += 1
                                seen_services.add(service)

        accounts = []
        for service, days in service_days.most_common():
            if total_days > 0:
                ratio = days / total_days
                if ratio >= 0.7:
                    freq = "daily"
                elif ratio >= 0.3:
                    freq = "weekly"
                else:
                    freq = "occasional"
            else:
                freq = "occasional"
            accounts.append({"service": service, "frequency": freq})

        return accounts

    def _infer_communication_style(self, summaries: list[dict]) -> dict:
        """Infer communication patterns.

        Returns dict like::

            {
                "primary_channels": ["slack", "email"],
                "avg_comm_minutes_per_day": 45,
            }
        """
        comm_apps = {
            "Slack", "Microsoft Teams", "Discord", "Zoom",
            "Google Meet", "Mail", "Outlook", "Telegram", "Messages",
        }

        channel_minutes: Counter = Counter()

        for summary in summaries:
            for app_entry in summary.get("top_apps", []):
                app = app_entry.get("app", "")
                minutes = app_entry.get("minutes", 0)
                if app in comm_apps:
                    channel_minutes[app] += minutes

        total = sum(channel_minutes.values())
        avg_per_day = round(total / max(len(summaries), 1), 1)

        return {
            "primary_channels": [
                app for app, _ in channel_minutes.most_common(3)
            ],
            "avg_comm_minutes_per_day": avg_per_day,
        }

"""Outcome detection for task boundaries.

Analyzes events within a task boundary to determine what the task
accomplished (e.g. file created, data transferred, message sent).
Uses heuristic pattern matching on event metadata and VLM annotations
without requiring VLM access at detection time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from oc_apprentice_worker.event_helpers import (
    extract_app_from_event as _event_app,
    parse_annotation as _event_annotation,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectedOutcome:
    """A detected outcome from task execution."""

    type: str  # "file_created", "data_transfer", "communication_sent", etc.
    description: str
    verification: dict  # how to verify this outcome happened
    confidence: float


class OutcomeTracker:
    """Detect outcomes (what changed) after task execution.

    Analyzes events within a task boundary to determine what
    the task accomplished.
    """

    def __init__(self) -> None:
        pass

    def detect_outcomes(self, task_events: list[dict]) -> list[DetectedOutcome]:
        """Detect outcomes from a list of events in a task boundary.

        Args:
            task_events: List of event dicts with keys:
                - kind_json: str (e.g. '{"ClipboardChange":{}}')
                - window_json: str | None
                - metadata_json: str | None
                - scene_annotation_json: str | None

        Returns:
            List of detected outcomes.
        """
        outcomes: list[DetectedOutcome] = []

        clipboard = self._check_clipboard_transfer(task_events)
        if clipboard is not None:
            outcomes.append(clipboard)

        file_activity = self._check_file_activity(task_events)
        if file_activity is not None:
            outcomes.append(file_activity)

        communication = self._check_communication(task_events)
        if communication is not None:
            outcomes.append(communication)

        navigation = self._check_navigation_completion(task_events)
        if navigation is not None:
            outcomes.append(navigation)

        data_entry = self._check_data_entry(task_events)
        if data_entry is not None:
            outcomes.append(data_entry)

        return outcomes

    def _check_clipboard_transfer(
        self, events: list[dict]
    ) -> DetectedOutcome | None:
        """Check if clipboard was used to transfer data between apps."""
        clipboard_events = [
            e for e in events if _event_kind(e) == "ClipboardChange"
        ]
        if not clipboard_events:
            return None

        # Check if there were app switches around clipboard events
        apps_before: set[str] = set()
        apps_after: set[str] = set()

        for i, event in enumerate(events):
            if event in clipboard_events:
                # Get apps before and after this clipboard event
                for j in range(max(0, i - 3), i):
                    a = _event_app(events[j])
                    if a:
                        apps_before.add(a)
                for j in range(i + 1, min(len(events), i + 4)):
                    a = _event_app(events[j])
                    if a:
                        apps_after.add(a)

        if apps_before and apps_after and apps_before != apps_after:
            src = ", ".join(sorted(apps_before))
            dst = ", ".join(sorted(apps_after))
            return DetectedOutcome(
                type="data_transfer",
                description=f"Clipboard data transferred from {src} to {dst}",
                verification={
                    "check": "clipboard content matches destination"
                },
                confidence=0.7,
            )

        if clipboard_events:
            return DetectedOutcome(
                type="data_transfer",
                description="Clipboard was used during task",
                verification={"check": "clipboard content updated"},
                confidence=0.5,
            )

        return None

    def _check_file_activity(
        self, events: list[dict]
    ) -> DetectedOutcome | None:
        """Check for file creation/modification outcomes."""
        file_apps = {
            "Finder",
            "Preview",
            "TextEdit",
            "VS Code",
            "Visual Studio Code",
            "Xcode",
            "Sublime Text",
        }

        file_events = []
        for event in events:
            app = _event_app(event)
            if app in file_apps:
                file_events.append(event)

        if not file_events:
            # Check annotations for file-related activity
            for event in events:
                ann = _event_annotation(event)
                if ann:
                    what = (
                        ann.get("task_context", {})
                        .get("what_doing", "")
                        .lower()
                    )
                    if any(
                        kw in what
                        for kw in (
                            "save",
                            "create file",
                            "download",
                            "export",
                        )
                    ):
                        return DetectedOutcome(
                            type="file_created",
                            description=f"File operation detected: {what}",
                            verification={
                                "check": "file exists at expected location"
                            },
                            confidence=0.6,
                        )
            return None

        return DetectedOutcome(
            type="file_created",
            description=(
                f"File activity in "
                f"{_event_app(file_events[-1]) or 'unknown app'}"
            ),
            verification={"check": "file exists and was recently modified"},
            confidence=0.6,
        )

    def _check_communication(
        self, events: list[dict]
    ) -> DetectedOutcome | None:
        """Check if a message/email was sent."""
        comm_apps = {
            "Slack",
            "Microsoft Teams",
            "Discord",
            "Mail",
            "Outlook",
            "Messages",
            "Telegram",
        }

        for event in events:
            app = _event_app(event)
            if app in comm_apps:
                ann = _event_annotation(event)
                if ann:
                    what = (
                        ann.get("task_context", {})
                        .get("what_doing", "")
                        .lower()
                    )
                    if any(
                        kw in what
                        for kw in (
                            "send",
                            "reply",
                            "compose",
                            "message",
                            "chat",
                        )
                    ):
                        return DetectedOutcome(
                            type="communication_sent",
                            description=f"Communication via {app}",
                            verification={
                                "check": f"message sent in {app}"
                            },
                            confidence=0.75,
                        )

        return None

    def _check_navigation_completion(
        self, events: list[dict]
    ) -> DetectedOutcome | None:
        """Check if browser navigation completed a workflow."""
        urls: list[str] = []
        for event in events:
            ann = _event_annotation(event)
            if ann:
                loc = ann.get("visual_context", {}).get("location", "")
                if loc and loc.startswith("http"):
                    urls.append(loc)

        if len(urls) >= 2:
            return DetectedOutcome(
                type="navigation_completed",
                description=f"Navigated through {len(urls)} pages",
                verification={
                    "check": f"ended at {urls[-1]}",
                    "pages_visited": len(urls),
                },
                confidence=0.5,
            )

        return None

    def _check_data_entry(
        self, events: list[dict]
    ) -> DetectedOutcome | None:
        """Check if data was entered into a form or application."""
        for event in events:
            ann = _event_annotation(event)
            if ann:
                what = (
                    ann.get("task_context", {})
                    .get("what_doing", "")
                    .lower()
                )
                if any(
                    kw in what
                    for kw in ("fill", "enter", "type", "input", "form")
                ):
                    return DetectedOutcome(
                        type="data_entry",
                        description="Data entered into application",
                        verification={"check": "form/fields populated"},
                        confidence=0.6,
                    )
        return None


# ---------------------------------------------------------------------------
# Private helpers — event field extraction
# ---------------------------------------------------------------------------


def _event_kind(event: dict) -> str:
    """Extract event kind from kind_json."""
    kind_json = event.get("kind_json", "{}")
    try:
        kind = (
            json.loads(kind_json)
            if isinstance(kind_json, str)
            else kind_json
        )
        if isinstance(kind, dict):
            return next(iter(kind), "")
        return str(kind)
    except (json.JSONDecodeError, TypeError):
        return ""

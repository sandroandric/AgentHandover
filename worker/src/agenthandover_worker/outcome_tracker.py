"""Outcome detection for task boundaries.

Analyzes events within a task boundary to determine what the task
accomplished (e.g. file created, data transferred, message sent).
Uses heuristic pattern matching on event metadata and VLM annotations
without requiring VLM access at detection time.

When an ``LLMReasoner`` is provided, the LLM supplements heuristic
detection when heuristics find nothing or only low-confidence outcomes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agenthandover_worker.event_helpers import (
    extract_app_from_event as _event_app,
    parse_annotation as _event_annotation,
)

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner

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

    def __init__(
        self,
        llm_reasoner: "LLMReasoner | None" = None,
    ) -> None:
        self._llm_reasoner = llm_reasoner

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

        # LLM supplementation: when heuristics found nothing or only
        # low-confidence outcomes, ask the LLM for additional insight.
        if self._llm_reasoner is not None:
            all_low = all(o.confidence < 0.6 for o in outcomes)
            if not outcomes or all_low:
                llm_outcomes = self._detect_outcomes_with_llm(task_events)
                existing_types = {o.type for o in outcomes}
                for lo in llm_outcomes:
                    if lo.type not in existing_types:
                        outcomes.append(lo)

        return outcomes

    def _detect_outcomes_with_llm(
        self,
        task_events: list[dict],
    ) -> list[DetectedOutcome]:
        """Use LLM to detect outcomes when heuristics are insufficient.

        Takes the last 3-5 events, formats their annotations, and asks
        the LLM what the user accomplished.
        """
        if self._llm_reasoner is None:
            return []

        last_events = task_events[-5:] if len(task_events) >= 5 else task_events[-3:] if len(task_events) >= 3 else task_events
        if not last_events:
            return []

        # Build context from event annotations
        event_summaries: list[str] = []
        for event in last_events:
            ann = _event_annotation(event)
            if ann is None:
                continue
            app = ann.get("visual_context", {}).get("active_app", "")
            location = ann.get("visual_context", {}).get("location", "")
            what_doing = ann.get("task_context", {}).get("what_doing", "")
            parts = []
            if app:
                parts.append(f"app={app}")
            if location:
                parts.append(f"location={location}")
            if what_doing:
                parts.append(f"doing={what_doing}")
            if parts:
                event_summaries.append(", ".join(parts))

        if not event_summaries:
            return []

        context = "; ".join(event_summaries)
        prompt = (
            f"What did the user accomplish in this workflow? "
            f"What changed as a result? Events: {context}. "
            f'Respond with JSON: {{"outcomes": [{{"type": "...", '
            f'"description": "...", "verification": "..."}}]}}. '
            f"If insufficient context, respond with INSUFFICIENT_EVIDENCE."
        )

        try:
            result = self._llm_reasoner.reason_json(
                prompt=prompt,
                caller="outcome_tracker._detect_outcomes_with_llm",
            )
            if not result.success or result.abstained or not result.value:
                return []

            raw_outcomes = result.value.get("outcomes", [])
            detected: list[DetectedOutcome] = []
            for raw in raw_outcomes:
                if not isinstance(raw, dict):
                    continue
                outcome_type = raw.get("type", "")
                description = raw.get("description", "")
                verification = raw.get("verification", "")
                if not outcome_type or not description:
                    continue
                # Confidence: higher if verification is provided
                confidence = 0.6 if verification else 0.4
                detected.append(DetectedOutcome(
                    type=outcome_type,
                    description=description,
                    verification={"check": verification} if isinstance(verification, str) else verification,
                    confidence=confidence,
                ))
            return detected
        except Exception:
            logger.debug(
                "LLM outcome detection failed",
                exc_info=True,
            )
            return []

    def detect_outcomes_with_evidence(
        self, task_events_list: list[list[dict]],
    ) -> list:
        """Detect outcomes with evidence-based confidence.

        Runs detect_outcomes() on each observation separately,
        then computes confidence as observations_with_outcome / total.
        """
        if not task_events_list:
            return []

        # Run detection on each observation (cache results to avoid double LLM calls)
        all_detected: list[list[DetectedOutcome]] = []
        all_outcomes: dict[str, list[bool]] = {}
        for events in task_events_list:
            detected = self.detect_outcomes(events)
            all_detected.append(detected)
            for o in detected:
                all_outcomes.setdefault(o.type, []).append(True)

        # Compute evidence-weighted confidence
        total = len(task_events_list)
        results = []
        for outcome_type, occurrences in all_outcomes.items():
            confidence = len(occurrences) / total
            # Find a representative outcome from cached results
            found = False
            for detected in all_detected:
                for o in detected:
                    if o.type == outcome_type:
                        results.append(DetectedOutcome(
                            type=o.type,
                            description=o.description,
                            confidence=round(confidence, 4),
                            verification=o.verification,
                        ))
                        found = True
                        break
                if found:
                    break
        return results

    def detect_postconditions(self, task_events: list[dict]) -> list[dict]:
        """Examine last 2-3 frames to determine expected post-state."""
        if not task_events:
            return []
        last_frames = task_events[-3:] if len(task_events) >= 3 else task_events
        postconditions = []
        for event in last_frames:
            ann = event.get("scene_annotation_json", {})
            if isinstance(ann, str):
                try:
                    ann = json.loads(ann)
                except (json.JSONDecodeError, TypeError):
                    continue
            vc = ann.get("visual_context", {})
            location = vc.get("location", "")
            app = vc.get("active_app", "")
            if location:
                postconditions.append({"type": "url_state", "expected": location, "app": app})
            tc = ann.get("task_context", {})
            what_doing = tc.get("what_doing", "")
            if what_doing:
                postconditions.append({"type": "task_completed", "expected": what_doing, "app": app})
        return postconditions

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

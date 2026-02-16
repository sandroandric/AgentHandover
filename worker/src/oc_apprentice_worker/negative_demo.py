"""Negative Demonstration Pruning — detect undo/cancel/discard patterns.

Implements section 8.2 of the OpenMimic spec: identify events that represent
user mistakes or abandoned workflows, and separate them from positive
demonstrations so downstream SOP extraction only learns from successful
workflows.

Detection patterns:
1. **Undo actions** — keyboard shortcut Ctrl+Z / Cmd+Z
2. **Cancel workflows** — click on Cancel / Close / Discard / Don't Save
3. **Discard changes** — click on "Discard changes", "Don't save", "Revert"
4. **Back after error** — URL navigation backwards after an error page

When a negative marker is found the pruner looks backwards (up to 10 events
or 30 seconds within the same thread) and marks those preceding events as
negative too, since they are the "mistake" that was undone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Maximum number of events to look back when marking predecessors
LOOKBACK_MAX_EVENTS = 10

# Maximum seconds to look back when marking predecessors
LOOKBACK_MAX_SECONDS = 30.0


@dataclass
class PruneResult:
    """Result of negative demonstration analysis."""

    positive_events: list[dict] = field(default_factory=list)
    negative_events: list[dict] = field(default_factory=list)
    prune_reasons: dict[str, str] = field(default_factory=dict)  # event_id -> reason


class NegativeDemoPruner:
    """Analyse a stream of events and separate positive from negative demos."""

    # Keyboard shortcuts that indicate an undo action
    UNDO_PATTERNS = ["ctrl+z", "cmd+z", "command+z"]

    # Click-target text patterns that indicate cancel/discard workflows
    CANCEL_PATTERNS = [
        "cancel",
        "close",
        "discard",
        "don't save",
        "revert",
        "undo",
        "discard changes",
    ]

    def prune(self, events: list[dict]) -> PruneResult:
        """Analyse *events* and separate positive from negative demonstrations.

        Returns a ``PruneResult`` with three fields:
        - ``positive_events`` — events that represent successful workflows
        - ``negative_events`` — events flagged as mistakes / abandoned actions
        - ``prune_reasons`` — mapping of event id to human-readable reason
        """
        if not events:
            return PruneResult()

        # Set of event ids that are negative
        negative_ids: set[str] = set()
        prune_reasons: dict[str, str] = {}

        for idx, event in enumerate(events):
            eid = event.get("id", "")

            # Already marked negative (e.g. by lookback from a later trigger)
            if eid in negative_ids:
                continue

            reason = self._detect_negative(event, events[:idx])
            if reason is not None:
                # Mark the trigger event itself
                negative_ids.add(eid)
                prune_reasons[eid] = reason

                # Mark preceding events within the lookback window,
                # scoped to the same episode if episode_id is available
                episode_id = event.get("episode_id", "")
                self._mark_lookback(
                    events[:idx],
                    event,
                    negative_ids,
                    prune_reasons,
                    reason,
                    episode_id=episode_id,
                )

        # Split into positive / negative
        positive: list[dict] = []
        negative: list[dict] = []
        for event in events:
            eid = event.get("id", "")
            if eid in negative_ids:
                negative.append(event)
            else:
                positive.append(event)

        return PruneResult(
            positive_events=positive,
            negative_events=negative,
            prune_reasons=prune_reasons,
        )

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_negative(self, event: dict, prev_events: list[dict]) -> str | None:
        """Return a reason string if *event* is a negative marker, else None."""
        if self._is_undo_event(event):
            return "undo_shortcut"
        if self._is_cancel_event(event):
            return "cancel_click"
        if self._is_back_after_error(event, prev_events):
            return "back_after_error"
        return None

    def _is_undo_event(self, event: dict) -> bool:
        """True if event is a keyboard undo shortcut (Ctrl+Z / Cmd+Z)."""
        kind_json = event.get("kind_json", "")
        metadata_json = event.get("metadata_json", "")

        # The event kind must indicate a key event
        kind_str = kind_json if isinstance(kind_json, str) else json.dumps(kind_json)
        if "key" not in kind_str.lower():
            return False

        # Check metadata for the undo shortcut
        meta_str = self._get_metadata_string(metadata_json).lower()
        return any(pattern in meta_str for pattern in self.UNDO_PATTERNS)

    def _is_cancel_event(self, event: dict) -> bool:
        """True if event is a click on a cancel/discard/close element."""
        kind_json = event.get("kind_json", "")
        metadata_json = event.get("metadata_json", "")

        # The event kind should indicate a click
        kind_str = kind_json if isinstance(kind_json, str) else json.dumps(kind_json)
        if "click" not in kind_str.lower():
            return False

        # Check metadata for cancel-like button text
        meta_str = self._get_metadata_string(metadata_json).lower()
        return any(pattern in meta_str for pattern in self.CANCEL_PATTERNS)

    def _is_back_after_error(self, event: dict, prev_events: list[dict]) -> bool:
        """True if event is a back navigation following an error page."""
        kind_json = event.get("kind_json", "")
        metadata_json = event.get("metadata_json", "")

        # Check if current event is a navigation-back type
        kind_str = kind_json if isinstance(kind_json, str) else json.dumps(kind_json)
        meta_str = self._get_metadata_string(metadata_json).lower()

        is_back = "back" in meta_str or "back" in kind_str.lower()
        if not is_back:
            return False

        # Look at recent previous events for error indicators
        for prev in reversed(prev_events[-5:]):
            prev_meta = self._get_metadata_string(prev.get("metadata_json", "")).lower()
            if any(err in prev_meta for err in ("error", "404", "500", "503", "not found")):
                return True

        return False

    # ------------------------------------------------------------------
    # Lookback marking
    # ------------------------------------------------------------------

    def _mark_lookback(
        self,
        preceding: list[dict],
        trigger: dict,
        negative_ids: set[str],
        prune_reasons: dict[str, str],
        reason: str,
        episode_id: str = "",
    ) -> None:
        """Mark preceding events in the same thread as negative.

        Looks back at most ``LOOKBACK_MAX_EVENTS`` events or
        ``LOOKBACK_MAX_SECONDS`` seconds, whichever is more restrictive.
        Only marks events that share the same thread (app_id).
        When episode_id is provided, only marks events in the same episode.
        """
        trigger_ts = self._parse_timestamp(trigger)
        trigger_app = self._extract_app_id(trigger)

        count = 0
        for prev_event in reversed(preceding):
            if count >= LOOKBACK_MAX_EVENTS:
                break

            prev_eid = prev_event.get("id", "")
            if prev_eid in negative_ids:
                count += 1
                continue

            # Must be same thread
            prev_app = self._extract_app_id(prev_event)
            if prev_app != trigger_app:
                continue

            # Episode boundary check — do not span episodes
            if episode_id:
                prev_episode_id = prev_event.get("episode_id", "")
                if prev_episode_id and prev_episode_id != episode_id:
                    continue

            # Time window check
            if trigger_ts is not None:
                prev_ts = self._parse_timestamp(prev_event)
                if prev_ts is not None:
                    delta = (trigger_ts - prev_ts).total_seconds()
                    if delta > LOOKBACK_MAX_SECONDS:
                        break

            negative_ids.add(prev_eid)
            prune_reasons[prev_eid] = f"preceded_{reason}"
            count += 1

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _get_metadata_string(metadata_json: str | dict) -> str:
        """Return a flattened string representation of metadata for searching."""
        if not metadata_json:
            return ""
        if isinstance(metadata_json, dict):
            return json.dumps(metadata_json)
        return metadata_json

    @staticmethod
    def _parse_timestamp(event: dict) -> datetime | None:
        raw = event.get("timestamp")
        if not raw or not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_app_id(event: dict) -> str:
        window_json = event.get("window_json")
        if not window_json:
            return ""
        try:
            window = json.loads(window_json) if isinstance(window_json, str) else window_json
        except (json.JSONDecodeError, TypeError):
            return ""
        return window.get("app_id", "")

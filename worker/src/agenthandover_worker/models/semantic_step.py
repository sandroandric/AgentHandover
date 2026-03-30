"""Semantic Step Data Model — a single meaningful user action in an episode.

Implements the core data structure for Phase 2 of the AgentHandover spec.  Each
``SemanticStep`` represents one human-readable action (click, type, navigate,
etc.) grounded in evidence from the DOM, accessibility tree, and vision
subsystems.

The model supports:
- Full serialization round-trips via ``to_dict()`` / ``from_dict()``
- Simplified SOP-ready output via ``to_sop_step()``
- Confidence tracking with human-readable reasons
- Negative demonstration marking via ``is_negative``
- Evidence chain linking DOM anchors, AX paths, and vision bounding boxes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Evidence:
    """Evidence supporting a semantic step's interpretation.

    Each field anchors the step to a specific element or location in the
    captured context:

    - ``dom_anchor``: CSS selector or XPath to the target element
    - ``ax_path``: Accessibility tree path (e.g. "window > toolbar > button[Save]")
    - ``vision_bbox``: Bounding box from VLM analysis ``{x, y, width, height}``
    - ``screenshot_id``: Reference to the screenshot artifact used for grounding
    - ``url``: Page URL at the time of the action
    - ``window_title``: Window title at the time of the action
    """

    dom_anchor: str | None = None
    ax_path: str | None = None
    vision_bbox: dict | None = None  # {x, y, width, height}
    screenshot_id: str | None = None
    url: str | None = None
    window_title: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "dom_anchor": self.dom_anchor,
            "ax_path": self.ax_path,
            "vision_bbox": self.vision_bbox,
            "screenshot_id": self.screenshot_id,
            "url": self.url,
            "window_title": self.window_title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Evidence:
        """Deserialize from a dictionary."""
        return cls(
            dom_anchor=data.get("dom_anchor"),
            ax_path=data.get("ax_path"),
            vision_bbox=data.get("vision_bbox"),
            screenshot_id=data.get("screenshot_id"),
            url=data.get("url"),
            window_title=data.get("window_title"),
        )


@dataclass
class SemanticStep:
    """A single semantic step in an episode -- one meaningful user action.

    Core identity:
        ``step_id`` uniquely identifies this step.  ``episode_id`` and
        ``step_index`` locate it within its parent episode.

    Semantics:
        ``intent`` is the action verb (click, type, navigate, ...).
        ``target_description`` is human-readable ("Submit button in review form").
        ``target_selector`` is machine-usable ("[aria-label='Submit review']").

    Parameters:
        Action-specific key/value pairs.  E.g. ``{"text": "LGTM"}`` for a
        type action or ``{"url": "https://..."}`` for navigate.

    Confidence:
        ``confidence`` is [0.0, 1.0].  ``decision`` is the routing outcome:
        "accept", "accept_flagged", or "reject".

    Evidence:
        An ``Evidence`` instance linking the step to DOM, AX, and vision data.
    """

    step_id: str
    episode_id: str
    step_index: int

    # Core semantics
    intent: str  # "click", "type", "navigate", "select", "scroll", "copy", "paste", etc.
    target_description: str  # Human-readable: "Submit button in review form"
    target_selector: str | None = None  # Machine-usable: "[aria-label='Submit review']"

    # Parameters (action-specific)
    parameters: dict = field(default_factory=dict)
    # e.g. {"text": "LGTM", "url": "https://...", "file_name": "report.xlsx"}

    # State context
    pre_state: dict = field(default_factory=dict)
    # {"window_title": "PR #123", "url": "https://github.com/...", "app_id": "com.google.Chrome"}
    post_state: dict = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0
    confidence_reasons: list[str] = field(default_factory=list)
    decision: str = "reject"  # "accept", "accept_flagged", "reject"

    # Evidence chain
    evidence: Evidence = field(default_factory=Evidence)

    # Metadata
    raw_event_id: str = ""
    timestamp: datetime | None = None
    is_negative: bool = False  # Marked by negative demo pruner

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary.

        Datetimes are serialized as ISO 8601 strings (UTC).  Nested objects
        (Evidence, dicts, lists) are recursively converted.
        """
        return {
            "step_id": self.step_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "intent": self.intent,
            "target_description": self.target_description,
            "target_selector": self.target_selector,
            "parameters": self.parameters,
            "pre_state": self.pre_state,
            "post_state": self.post_state,
            "confidence": self.confidence,
            "confidence_reasons": list(self.confidence_reasons),
            "decision": self.decision,
            "evidence": self.evidence.to_dict(),
            "raw_event_id": self.raw_event_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_negative": self.is_negative,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SemanticStep:
        """Deserialize from a dictionary.

        Handles ISO 8601 timestamp strings and nested Evidence dicts.
        """
        # Parse timestamp
        ts_raw = data.get("timestamp")
        timestamp: datetime | None = None
        if ts_raw is not None and isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                timestamp = None
        elif isinstance(ts_raw, datetime):
            timestamp = ts_raw

        # Parse evidence
        evidence_data = data.get("evidence")
        if isinstance(evidence_data, dict):
            evidence = Evidence.from_dict(evidence_data)
        elif isinstance(evidence_data, Evidence):
            evidence = evidence_data
        else:
            evidence = Evidence()

        return cls(
            step_id=data["step_id"],
            episode_id=data["episode_id"],
            step_index=data["step_index"],
            intent=data["intent"],
            target_description=data["target_description"],
            target_selector=data.get("target_selector"),
            parameters=data.get("parameters", {}),
            pre_state=data.get("pre_state", {}),
            post_state=data.get("post_state", {}),
            confidence=data.get("confidence", 0.0),
            confidence_reasons=list(data.get("confidence_reasons", [])),
            decision=data.get("decision", "reject"),
            evidence=evidence,
            raw_event_id=data.get("raw_event_id", ""),
            timestamp=timestamp,
            is_negative=data.get("is_negative", False),
        )

    def to_sop_step(self) -> dict:
        """Convert to a simplified step for SOP inclusion.

        Returns a minimal dictionary with only the fields needed for
        SOP generation: the action verb, human-readable target, machine
        selector, parameters, and confidence score.
        """
        return {
            "step": self.intent,
            "target": self.target_description,
            "selector": self.target_selector,
            "parameters": self.parameters,
            "confidence": self.confidence,
        }

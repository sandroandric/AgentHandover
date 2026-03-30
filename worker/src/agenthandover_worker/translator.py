"""Semantic Translator — resolve UI actions via structured metadata (no VLM).

.. note:: v2 Pipeline Relationship
    In the v2 VLM-based pipeline, the semantic translator's primary
    translation role is replaced by ``scene_annotator.py`` (which reads
    the screen directly via VLM).  However, the translator's DOM
    extraction logic is still used to provide CSS selector hints for the
    DOM Hints appendix in SKILL.md files.

Implements section 9.1 of the AgentHandover spec: translate raw events with
DOM/ARIA/accessibility data into semantic steps.  VLM is used only as a
fallback when structured metadata is insufficient.

Structured Metadata Priority Order:
  1. ARIA-label / accessible name  (highest trust, confidence 0.40-0.45)
  2. Visible innerText (normalized) (confidence 0.25-0.35)
  3. Role + relative position to stable headings (confidence 0.15-0.25)
  4. data-testid (if stable)        (confidence 0.35-0.45)
  5. Vision bbox fallback           (lowest trust, confidence 0.10)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from agenthandover_worker.css_filter import CSSRotFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class UIAnchor:
    """Resolved UI element anchor from structured metadata."""

    method: str  # "aria_label", "inner_text", "role_position", "test_id", "vision_bbox"
    selector: str  # The resolved selector/description
    confidence_contribution: float  # 0.0-0.45 based on method quality
    raw_evidence: dict = field(default_factory=dict)


@dataclass
class TranslationResult:
    """Result of translating a raw event into a semantic description."""

    intent: str  # e.g., "click", "type", "navigate", "select", "scroll"
    target: UIAnchor | None
    parameters: dict  # e.g., {"text": "...", "url": "..."}
    pre_state: dict  # Window title, URL, app context before action
    post_state: dict  # Window title, URL after action
    raw_event_id: str


# ---------------------------------------------------------------------------
# Event kind -> intent mapping
# ---------------------------------------------------------------------------

_KIND_INTENT_MAP: dict[str, str] = {
    "ClickIntent": "click",
    "FocusChange": "navigate",
    "AppSwitch": "switch_app",
    "DwellSnapshot": "read",
    "ScrollReadSnapshot": "scroll_read",
    "ClipboardChange": "copy",
    "PasteDetected": "paste",
    "WindowTitleChange": "navigate",
    "KeyPress": "type",
    "SecureFieldFocus": "secure_focus",
}


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class SemanticTranslator:
    """Translate raw events into semantic descriptions using structured metadata.

    Attempts to ground each event's target element via accessibility metadata
    (ARIA labels, visible text, roles, test IDs) before falling back to
    vision bounding boxes.  Uses the ``CSSRotFilter`` to strip unstable
    selectors from all paths.
    """

    def __init__(self) -> None:
        self._css_filter = CSSRotFilter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_event(
        self,
        event: dict,
        context: dict | None = None,
    ) -> TranslationResult:
        """Translate a single raw event into a semantic description.

        Parameters
        ----------
        event:
            A raw event dict as stored in the daemon's SQLite database.
        context:
            Optional context dict carrying state from prior events
            (e.g. previous window title, URL, app).
        """
        context = context or {}

        intent = self._extract_intent(event)
        anchor = self._resolve_ui_anchor(event)
        parameters = self._extract_parameters(event, intent)
        pre_state = self._build_pre_state(event, context)
        post_state = self._build_post_state(event, context)
        raw_event_id = event.get("id", "")

        return TranslationResult(
            intent=intent,
            target=anchor,
            parameters=parameters,
            pre_state=pre_state,
            post_state=post_state,
            raw_event_id=raw_event_id,
        )

    def translate_batch(self, events: list[dict]) -> list[TranslationResult]:
        """Translate a sequence of events, building context between them.

        Context is accumulated across events so each translation benefits
        from knowledge of what came before (e.g. the window title from the
        previous event becomes the expected pre-state for the next).
        """
        results: list[TranslationResult] = []
        context: dict = {}

        for event in events:
            result = self.translate_event(event, context)
            results.append(result)

            # Update rolling context for the next event
            self._update_context(context, event, result)

        return results

    # ------------------------------------------------------------------
    # Intent extraction
    # ------------------------------------------------------------------

    def _extract_intent(self, event: dict) -> str:
        """Map event kind to semantic intent.

        Extracts the top-level key from ``kind_json`` and maps it through
        ``_KIND_INTENT_MAP``.  Falls back to ``"unknown"`` when the kind
        is not recognized.
        """
        kind_json = event.get("kind_json", "")
        if not kind_json:
            return "unknown"

        try:
            parsed = json.loads(kind_json) if isinstance(kind_json, str) else kind_json
        except (json.JSONDecodeError, TypeError):
            return "unknown"

        if isinstance(parsed, dict) and parsed:
            # Two serialisation formats:
            # 1. Flat struct:  {"type": "AppSwitch", "from_app": "...", ...}
            # 2. Rust enum:   {"AppSwitch": {"from_app": "...", ...}}
            if "type" in parsed:
                kind_name = parsed["type"]
            else:
                kind_name = next(iter(parsed))
            return _KIND_INTENT_MAP.get(kind_name, "unknown")

        return "unknown"

    # ------------------------------------------------------------------
    # UI anchor resolution (priority cascade)
    # ------------------------------------------------------------------

    def _resolve_ui_anchor(self, event: dict) -> UIAnchor | None:
        """Try each resolution method in priority order.

        Returns the first successful anchor, or ``None`` if no method
        can resolve the target element.

        Priority:
          1. ARIA-label (0.40-0.45)
          2. data-testid (0.35-0.45)  — tried before innerText because
             test IDs are more specific when present
          3. Visible innerText (0.25-0.35)
          4. Role + position (0.15-0.25)
          5. Vision bbox (0.10)
          6. App context — window/app name (0.10-0.15) — native app fallback
        """
        metadata = self._parse_metadata(event)

        if metadata:
            target = metadata.get("target", {})
            if not isinstance(target, dict):
                target = {}

            # Try DOM-level resolution methods in priority order
            resolvers = [
                self._try_aria_label,
                self._try_test_id,
                self._try_inner_text,
                self._try_role_position,
                self._try_vision_bbox,
            ]

            for resolver in resolvers:
                anchor = resolver(metadata)
                if anchor is not None:
                    return anchor

        # Fallback: use app/window context from kind_json and window_json
        # for native app events that lack DOM-level selectors
        return self._try_app_context(event)

    def _try_aria_label(self, metadata: dict) -> UIAnchor | None:
        """Try to resolve via ARIA-label. Confidence: 0.40-0.45.

        Looks for ``ariaLabel`` in the ``target`` sub-dict.  A non-empty
        label yields the highest confidence anchor.  Longer, more descriptive
        labels earn 0.45; short labels earn 0.40.
        """
        target = metadata.get("target", {})
        if not isinstance(target, dict):
            return None

        aria_label = target.get("ariaLabel", "")
        if not aria_label or not isinstance(aria_label, str):
            return None

        aria_label = aria_label.strip()
        if not aria_label:
            return None

        # More descriptive labels are slightly more trustworthy
        confidence = 0.45 if len(aria_label) > 3 else 0.40

        return UIAnchor(
            method="aria_label",
            selector=f"[aria-label='{aria_label}']",
            confidence_contribution=confidence,
            raw_evidence={"ariaLabel": aria_label},
        )

    def _try_test_id(self, metadata: dict) -> UIAnchor | None:
        """Try to resolve via data-testid. Confidence: 0.35-0.45.

        Test IDs that look randomized (containing hash-like segments) get
        lower confidence.  Stable, human-readable test IDs are treated
        nearly as trustworthy as ARIA labels.
        """
        target = metadata.get("target", {})
        if not isinstance(target, dict):
            return None

        test_id = target.get("testId", "")
        if not test_id or not isinstance(test_id, str):
            return None

        test_id = test_id.strip()
        if not test_id:
            return None

        # Check if the test ID looks randomized (hash-like suffix)
        if re.search(r"[a-f0-9]{8,}", test_id):
            # Looks like it might contain a hash — lower confidence
            confidence = 0.35
        else:
            confidence = 0.45

        return UIAnchor(
            method="test_id",
            selector=f"[data-testid='{test_id}']",
            confidence_contribution=confidence,
            raw_evidence={"testId": test_id},
        )

    def _try_inner_text(self, metadata: dict) -> UIAnchor | None:
        """Try to resolve via visible innerText. Confidence: 0.25-0.35.

        The inner text is normalized: trimmed, spaces collapsed, lowercased.
        Longer, more specific text earns higher confidence.  Very short text
        (single characters) or very long text (over 100 chars) get lower
        confidence due to ambiguity or fragility.
        """
        target = metadata.get("target", {})
        if not isinstance(target, dict):
            return None

        inner_text = target.get("innerText", "")
        if not inner_text or not isinstance(inner_text, str):
            return None

        # Normalize: trim, collapse whitespace, lowercase
        normalized = re.sub(r"\s+", " ", inner_text.strip()).lower()
        if not normalized:
            return None

        # Score based on text quality
        length = len(normalized)
        if length <= 1:
            confidence = 0.25
        elif length > 100:
            # Very long text is fragile — likely to change
            confidence = 0.25
        elif length >= 5:
            confidence = 0.35
        else:
            confidence = 0.30

        return UIAnchor(
            method="inner_text",
            selector=f"text='{normalized}'",
            confidence_contribution=confidence,
            raw_evidence={"innerText": inner_text, "normalized": normalized},
        )

    def _try_role_position(self, metadata: dict) -> UIAnchor | None:
        """Try to resolve via role + heading relationship. Confidence: 0.15-0.25.

        Uses the element's ARIA role and its position relative to headings
        or parent containers.  This is less stable than text-based anchors
        but more robust than coordinates.
        """
        target = metadata.get("target", {})
        if not isinstance(target, dict):
            return None

        role = target.get("role", "")
        if not role or not isinstance(role, str):
            return None

        role = role.strip()
        if not role:
            return None

        # Build a descriptor incorporating available structural info
        tag_name = target.get("tagName", "")
        composed_path = target.get("composedPath", [])

        # Clean composed path of CSS rot
        if composed_path and isinstance(composed_path, list):
            cleaned_path = []
            for segment in composed_path:
                if isinstance(segment, str):
                    cleaned_path.append(self._css_filter.clean_selector(segment))
                else:
                    cleaned_path.append(segment)
            composed_path = cleaned_path

        descriptor_parts = [f"role={role}"]
        if tag_name:
            descriptor_parts.append(f"tag={tag_name}")
        if composed_path:
            # Use first 3 levels of path for context
            path_str = " > ".join(str(s) for s in composed_path[:3])
            descriptor_parts.append(f"path={path_str}")

        selector = f"[{', '.join(descriptor_parts)}]"

        # More context = higher confidence
        confidence = 0.15
        if tag_name:
            confidence += 0.05
        if composed_path:
            confidence += 0.05

        return UIAnchor(
            method="role_position",
            selector=selector,
            confidence_contribution=min(confidence, 0.25),
            raw_evidence={"role": role, "tagName": tag_name, "composedPath": composed_path},
        )

    def _try_vision_bbox(self, metadata: dict) -> UIAnchor | None:
        """Fallback to vision bounding box. Confidence: 0.10.

        Uses raw x/y coordinates as a last resort.  This is the least
        stable anchor because coordinates change with window position,
        resolution, and layout changes.
        """
        x = metadata.get("x")
        y = metadata.get("y")

        if x is None or y is None:
            # Also check inside target dict
            target = metadata.get("target", {})
            if isinstance(target, dict):
                x = target.get("x", x)
                y = target.get("y", y)

        if x is None or y is None:
            return None

        try:
            x_val = float(x)
            y_val = float(y)
        except (TypeError, ValueError):
            return None

        return UIAnchor(
            method="vision_bbox",
            selector=f"bbox({x_val:.0f},{y_val:.0f})",
            confidence_contribution=0.10,
            raw_evidence={"x": x_val, "y": y_val},
        )

    def _try_app_context(self, event: dict) -> UIAnchor | None:
        """Fallback anchor from app/window context. Confidence: 0.10-0.15.

        For native macOS app events that lack DOM-level selectors, use
        the app name and window title from ``kind_json`` and ``window_json``
        as a low-confidence anchor.
        """
        # Extract app name from kind_json
        kind_json = event.get("kind_json", "")
        try:
            kind_data = json.loads(kind_json) if isinstance(kind_json, str) else kind_json
        except (json.JSONDecodeError, TypeError):
            kind_data = {}

        app_name = ""
        if isinstance(kind_data, dict):
            # AppSwitch has to_app; others may not
            app_name = kind_data.get("to_app", "")

        # Also try window_json
        window = self._parse_window(event)
        window_app = window.get("app_id", "")
        window_title = window.get("title", "")

        # Use the most informative identifier
        # Strip PID prefix if present: "pid:1234:AppName" → "AppName"
        identifier = app_name or window_app
        if identifier and ":" in identifier:
            parts = identifier.split(":")
            identifier = parts[-1] if len(parts) >= 3 else identifier

        if not identifier:
            return None

        # Window title adds specificity
        confidence = 0.15 if window_title else 0.10

        selector = identifier
        if window_title:
            selector = f"{identifier}:{window_title}"

        return UIAnchor(
            method="app_context",
            selector=selector,
            confidence_contribution=confidence,
            raw_evidence={
                "app_name": identifier,
                "window_title": window_title,
            },
        )

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def _extract_parameters(self, event: dict, intent: str) -> dict:
        """Extract intent-specific parameters from the event."""
        metadata = self._parse_metadata(event)
        params: dict = {}

        if intent == "click":
            target = metadata.get("target", {}) if metadata else {}
            if isinstance(target, dict):
                if target.get("innerText"):
                    params["button_text"] = target["innerText"]
                if target.get("tagName"):
                    params["element_type"] = target["tagName"]

        elif intent == "type":
            if metadata:
                if metadata.get("key"):
                    params["key"] = metadata["key"]
                if metadata.get("text"):
                    params["text"] = metadata["text"]
                if metadata.get("shortcut"):
                    params["shortcut"] = metadata["shortcut"]

        elif intent in ("navigate", "switch_app"):
            if metadata:
                if metadata.get("url"):
                    params["url"] = metadata["url"]
                if metadata.get("app_name"):
                    params["app_name"] = metadata["app_name"]
            # For AppSwitch events, extract from/to app from kind_json
            kind_json = event.get("kind_json", "")
            try:
                kind_data = json.loads(kind_json) if isinstance(kind_json, str) else kind_json
                if isinstance(kind_data, dict):
                    if kind_data.get("to_app"):
                        params["app_name"] = kind_data["to_app"]
                    if kind_data.get("from_app"):
                        params["from_app"] = kind_data["from_app"]
            except (json.JSONDecodeError, TypeError):
                pass

        elif intent == "copy":
            if metadata:
                if metadata.get("content_hash"):
                    params["content_hash"] = metadata["content_hash"]
                if metadata.get("content_types"):
                    params["content_types"] = metadata["content_types"]

        elif intent == "paste":
            if metadata:
                if metadata.get("content_hash"):
                    params["content_hash"] = metadata["content_hash"]
                if metadata.get("target_app"):
                    params["target_app"] = metadata["target_app"]

        elif intent in ("read", "scroll_read"):
            if metadata:
                if metadata.get("url"):
                    params["url"] = metadata["url"]

        return params

    # ------------------------------------------------------------------
    # State building
    # ------------------------------------------------------------------

    def _build_pre_state(self, event: dict, context: dict) -> dict:
        """Build the pre-action state from the event and accumulated context."""
        state: dict = {}

        # Window info
        window = self._parse_window(event)
        if window:
            if window.get("title"):
                state["window_title"] = window["title"]
            if window.get("app_id"):
                state["app_id"] = window["app_id"]

        # URL from metadata
        metadata = self._parse_metadata(event)
        if metadata and metadata.get("url"):
            state["url"] = metadata["url"]

        # Fill gaps from context (previous event's post-state)
        if not state.get("window_title") and context.get("last_window_title"):
            state["window_title"] = context["last_window_title"]
        if not state.get("url") and context.get("last_url"):
            state["url"] = context["last_url"]
        if not state.get("app_id") and context.get("last_app_id"):
            state["app_id"] = context["last_app_id"]

        return state

    def _build_post_state(self, event: dict, context: dict) -> dict:
        """Build the post-action state.

        For most events the post state is the same as the pre state.
        For navigation events we look for a destination URL/title.
        """
        state: dict = {}

        window = self._parse_window(event)
        if window:
            if window.get("title"):
                state["window_title"] = window["title"]
            if window.get("app_id"):
                state["app_id"] = window["app_id"]

        metadata = self._parse_metadata(event)
        if metadata:
            if metadata.get("url"):
                state["url"] = metadata["url"]
            # Navigation events may have a destination
            if metadata.get("destination_url"):
                state["url"] = metadata["destination_url"]
            if metadata.get("new_title"):
                state["window_title"] = metadata["new_title"]

        return state

    # ------------------------------------------------------------------
    # Context management (for batch translation)
    # ------------------------------------------------------------------

    def _update_context(
        self,
        context: dict,
        event: dict,
        result: TranslationResult,
    ) -> None:
        """Update rolling context after translating an event.

        The context carries forward state that subsequent events can
        reference as their pre-state.
        """
        # Carry forward the post-state as the next event's expected pre-state
        if result.post_state.get("window_title"):
            context["last_window_title"] = result.post_state["window_title"]
        if result.post_state.get("url"):
            context["last_url"] = result.post_state["url"]
        if result.post_state.get("app_id"):
            context["last_app_id"] = result.post_state["app_id"]

        # Track metadata for provenance
        metadata = self._parse_metadata(event)
        if metadata:
            if metadata.get("content_hash"):
                context["clipboard_link"] = True
            if result.intent == "read":
                context["dwell_snapshot"] = True

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_metadata(self, event: dict) -> dict:
        """Parse ``metadata_json`` from the event, returning a dict."""
        metadata_json = event.get("metadata_json", "")
        if not metadata_json:
            return {}

        try:
            parsed = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except (json.JSONDecodeError, TypeError):
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _parse_window(self, event: dict) -> dict:
        """Parse ``window_json`` from the event, returning a dict."""
        window_json = event.get("window_json", "")
        if not window_json:
            return {}

        try:
            parsed = json.loads(window_json) if isinstance(window_json, str) else window_json
        except (json.JSONDecodeError, TypeError):
            return {}

        return parsed if isinstance(parsed, dict) else {}

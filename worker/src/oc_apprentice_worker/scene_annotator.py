"""VLM scene annotation for captured screenshots.

Processes DwellSnapshot/ScrollReadSnapshot frames through a local VLM
(qwen3.5:2b with think=False) to produce structured semantic annotations.
Each annotation describes what the user is doing, what's visible on screen,
and whether the activity is part of a meaningful workflow.

Key features:
- 3-frame sliding window context (time-bounded to 10 minutes)
- Stale-frame skipping: 3+ consecutive same-app non-workflow → skip
- JSON validation with markdown-fence stripping and one retry
- Screenshot lifecycle: delete JPEG only after successful annotation
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

ANNOTATION_FIELDS = {
    "app",
    "location",
    "visible_content",
    "ui_state",
    "task_context",
}

TASK_CONTEXT_FIELDS = {"what_doing", "likely_next", "is_workflow"}


@dataclass
class AnnotationConfig:
    """Configuration for the scene annotator."""

    model: str = "qwen3.5:2b"
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 800
    stale_skip_count: int = 3
    sliding_window_size: int = 3
    sliding_window_max_age_sec: int = 600
    delete_screenshot_on_success: bool = True


@dataclass
class AnnotationResult:
    """Result of a single frame annotation."""

    event_id: str
    status: str  # completed, failed, skipped, missing_screenshot
    annotation: dict | None = None
    error: str | None = None
    inference_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a screen annotation assistant. You analyze screenshots and "
    "extract structured information about what the user is doing. "
    "Respond with ONLY valid JSON, no markdown fences, no commentary."
)

ANNOTATION_PROMPT_TEMPLATE = """\
Analyze this screenshot and extract the following information as JSON:

{{
  "app": "<application name visible on screen>",
  "location": "<URL, file path, or app-specific location>",
  "visible_content": {{
    "headings": ["<main headings or titles visible>"],
    "labels": ["<form labels, button text, menu items>"],
    "values": ["<filled form values, selected options, typed text>"]
  }},
  "ui_state": {{
    "active_element": "<what element has focus or was just interacted with>",
    "modals_or_popups": "<any overlays, dropdowns, or dialogs visible>",
    "scroll_position": "<top, middle, bottom, or specific>"
  }},
  "task_context": {{
    "what_doing": "<one sentence: what task is the user performing right now>",
    "likely_next": "<one sentence: what will the user probably do next>",
    "is_workflow": <true if this is a structured, repeatable task; false for browsing/chatting/reading>
  }}
}}

{context_section}Respond with ONLY the JSON object. No markdown, no explanation."""

CONTEXT_TEMPLATE = """\
PREVIOUS FRAMES (last {window_age_label}):
{context_lines}

Use the previous frames to understand what task the user is working on.
If the current screenshot continues a task from the previous frames, reflect that in task_context.

"""


def _build_context_section(recent_annotations: list[dict]) -> str:
    """Build the sliding-window context section for the prompt."""
    if not recent_annotations:
        return ""

    lines = []
    for ann in reversed(recent_annotations):  # oldest first
        ts = ann.get("timestamp", "?")
        # Truncate to HH:MM:SS for readability
        if len(ts) > 19:
            ts = ts[11:19]
        elif len(ts) > 8:
            ts = ts[-8:]

        ann_json = ann.get("scene_annotation_json")
        if not ann_json:
            continue

        try:
            parsed = json.loads(ann_json)
        except (json.JSONDecodeError, TypeError):
            continue

        app = parsed.get("app", "?")
        location = parsed.get("location", "?")
        what_doing = parsed.get("task_context", {}).get("what_doing", "?")
        lines.append(f"- [{ts}] {app} — {location} — {what_doing}")

    if not lines:
        return ""

    return CONTEXT_TEMPLATE.format(
        window_age_label="10 min",
        context_lines="\n".join(lines),
    )


def build_annotation_prompt(
    recent_annotations: list[dict] | None = None,
) -> str:
    """Build the full annotation prompt with optional sliding-window context."""
    context_section = ""
    if recent_annotations:
        context_section = _build_context_section(recent_annotations)

    return ANNOTATION_PROMPT_TEMPLATE.format(context_section=context_section)


# ---------------------------------------------------------------------------
# JSON validation / repair
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

_THINK_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and thinking tags from VLM output."""
    # Strip <think>...</think> blocks (Qwen thinking mode artefact)
    text = _THINK_RE.sub("", text).strip()
    # Extract content inside fences if present
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _validate_annotation(raw: str) -> dict | None:
    """Parse and validate an annotation JSON string.

    Returns the parsed dict if valid, None if malformed.
    Tolerant of missing optional fields but requires the top-level
    structure and task_context.what_doing at minimum.
    """
    cleaned = _strip_markdown_fences(raw)
    if not cleaned:
        return None

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Must have task_context with what_doing at minimum
    tc = data.get("task_context")
    if not isinstance(tc, dict) or "what_doing" not in tc:
        return None

    # Ensure is_workflow is boolean (VLM sometimes returns string)
    if "is_workflow" in tc:
        val = tc["is_workflow"]
        if isinstance(val, str):
            tc["is_workflow"] = val.lower() in ("true", "yes", "1")
        elif not isinstance(val, bool):
            tc["is_workflow"] = False

    return data


# ---------------------------------------------------------------------------
# Stale-frame detection
# ---------------------------------------------------------------------------

@dataclass
class _StaleTracker:
    """Track consecutive non-workflow frames for stale-skip logic."""

    consecutive_non_workflow: int = 0
    last_app: str = ""
    last_location: str = ""

    def update(self, annotation: dict) -> bool:
        """Update tracker and return True if this frame should be skipped.

        A frame is stale (skip-worthy) after *threshold* consecutive
        annotations where:
        - app + location are unchanged
        - is_workflow is False
        """
        app = annotation.get("app", "")
        location = annotation.get("location", "")
        is_workflow = annotation.get("task_context", {}).get("is_workflow", False)

        if is_workflow or app != self.last_app or location != self.last_location:
            self.consecutive_non_workflow = 0
            self.last_app = app
            self.last_location = location
            if not is_workflow and app == self.last_app:
                self.consecutive_non_workflow = 1
            return False

        # Same app + location, non-workflow
        self.consecutive_non_workflow += 1
        return False  # Don't skip this one yet — caller checks threshold

    def should_skip(self, threshold: int) -> bool:
        """Return True if we've exceeded the stale threshold."""
        return self.consecutive_non_workflow >= threshold

    def reset(self) -> None:
        """Reset tracker (e.g. on significant screen change or app switch)."""
        self.consecutive_non_workflow = 0
        self.last_app = ""
        self.last_location = ""


# ---------------------------------------------------------------------------
# Ollama VLM client
# ---------------------------------------------------------------------------

def _call_ollama_vlm(
    *,
    model: str,
    prompt: str,
    image_path: str | Path | None = None,
    image_base64: str | None = None,
    host: str = "http://localhost:11434",
    num_predict: int = 800,
    system: str = "",
    timeout: float = 60.0,
) -> tuple[str, float]:
    """Call Ollama's /api/generate with an optional image.

    Returns (response_text, inference_time_seconds).
    Raises on connection or HTTP errors.
    """
    import urllib.request
    import urllib.error
    import base64

    url = f"{host}/api/generate"

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            # think=False: prevent Qwen thinking mode from consuming tokens
            "think": False,
        },
    }

    if system:
        payload["system"] = system

    # Attach image
    if image_path and not image_base64:
        img_path = Path(image_path)
        if img_path.is_file():
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("ascii")

    if image_base64:
        payload["images"] = [image_base64]

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Ollama not reachable at {host}: {exc}") from exc

    elapsed = time.monotonic() - start
    return result.get("response", ""), elapsed


# ---------------------------------------------------------------------------
# SceneAnnotator
# ---------------------------------------------------------------------------

class SceneAnnotator:
    """Continuous scene annotation engine.

    Processes un-annotated screenshots from the DB, calls the local VLM,
    validates the response, and stores the structured annotation back.

    Designed to run in a dedicated thread within the worker process.
    """

    def __init__(self, config: AnnotationConfig | None = None) -> None:
        self.config = config or AnnotationConfig()
        self._stale = _StaleTracker()
        self._stats = {
            "annotated": 0,
            "skipped_stale": 0,
            "failed": 0,
            "missing_screenshot": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Core annotation
    # ------------------------------------------------------------------

    def annotate_event(
        self,
        event: dict,
        *,
        recent_annotations: list[dict] | None = None,
        artifact_dir: str | Path | None = None,
        skip_stale_check: bool = False,
    ) -> AnnotationResult:
        """Annotate a single event's screenshot.

        *event* is a row dict from the events table.
        *recent_annotations* is the sliding-window context (up to 3 recent).
        *artifact_dir* is where screenshot files are stored.
        *skip_stale_check* disables stale-frame skipping (used for focus
        sessions where every frame matters).

        Returns an AnnotationResult with status and annotation data.
        """
        event_id = event.get("id", "unknown")

        # --- Stale-frame check ---
        # If we've been skipping due to stale frames, check if this one
        # should also be skipped.
        if not skip_stale_check and self._stale.should_skip(self.config.stale_skip_count):
            # Check if the screen actually changed (app/title differ from recent)
            window_json = event.get("window_json")
            current_app = ""
            current_title = ""
            if window_json:
                try:
                    w = json.loads(window_json)
                    current_app = w.get("app_bundle_id", "") or w.get("app_name", "")
                    current_title = w.get("title", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            if (
                current_app == self._stale.last_app
                and current_title == self._stale.last_location
            ):
                # Still stale — skip this frame
                self._stats["skipped_stale"] += 1
                logger.debug(
                    "Stale-skip event %s (app=%s, %d consecutive)",
                    event_id,
                    current_app,
                    self._stale.consecutive_non_workflow,
                )
                return AnnotationResult(
                    event_id=event_id,
                    status="skipped",
                    error="stale_frame",
                )
            else:
                # Screen changed — reset stale tracker and proceed
                self._stale.reset()

        # --- Locate screenshot ---
        screenshot_path = self._find_screenshot(event, artifact_dir)
        if screenshot_path is None:
            self._stats["missing_screenshot"] += 1
            return AnnotationResult(
                event_id=event_id,
                status="missing_screenshot",
                error="no_screenshot_file",
            )

        # --- Build prompt ---
        prompt = build_annotation_prompt(recent_annotations)

        # --- Call VLM ---
        try:
            raw_response, inference_time = _call_ollama_vlm(
                model=self.config.model,
                prompt=prompt,
                image_path=screenshot_path,
                host=self.config.ollama_host,
                num_predict=self.config.num_predict,
                system=SYSTEM_PROMPT,
            )
        except ConnectionError as exc:
            self._stats["failed"] += 1
            return AnnotationResult(
                event_id=event_id,
                status="failed",
                error=f"ollama_connection: {exc}",
            )
        except Exception as exc:
            self._stats["failed"] += 1
            return AnnotationResult(
                event_id=event_id,
                status="failed",
                error=f"vlm_error: {exc}",
            )

        # --- Validate JSON ---
        annotation = _validate_annotation(raw_response)

        if annotation is None:
            # Retry once with explicit JSON-only instruction
            logger.debug(
                "Annotation JSON invalid for %s, retrying with JSON-only suffix",
                event_id,
            )
            retry_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. No text outside the JSON."
            try:
                raw_response, retry_time = _call_ollama_vlm(
                    model=self.config.model,
                    prompt=retry_prompt,
                    image_path=screenshot_path,
                    host=self.config.ollama_host,
                    num_predict=self.config.num_predict,
                    system=SYSTEM_PROMPT,
                )
                inference_time += retry_time
                annotation = _validate_annotation(raw_response)
            except Exception:
                pass

        if annotation is None:
            self._stats["failed"] += 1
            return AnnotationResult(
                event_id=event_id,
                status="failed",
                error="invalid_json_after_retry",
                inference_time_seconds=inference_time,
            )

        # --- Update stale tracker ---
        self._stale.update(annotation)

        # --- Delete screenshot on success ---
        if self.config.delete_screenshot_on_success and screenshot_path:
            try:
                Path(screenshot_path).unlink(missing_ok=True)
                logger.debug("Deleted screenshot after annotation: %s", screenshot_path)
            except OSError:
                logger.debug("Failed to delete screenshot %s", screenshot_path, exc_info=True)

        self._stats["annotated"] += 1
        return AnnotationResult(
            event_id=event_id,
            status="completed",
            annotation=annotation,
            inference_time_seconds=inference_time,
        )

    # ------------------------------------------------------------------
    # Screenshot location
    # ------------------------------------------------------------------

    def _find_screenshot(
        self,
        event: dict,
        artifact_dir: str | Path | None,
    ) -> str | None:
        """Resolve the screenshot file path for an event.

        Checks artifact_ids_json for screenshot references, then looks
        for the file on disk.
        """
        artifact_ids_raw = event.get("artifact_ids_json", "[]")
        try:
            artifact_ids = json.loads(artifact_ids_raw) if artifact_ids_raw else []
        except (json.JSONDecodeError, TypeError):
            artifact_ids = []

        if not artifact_ids:
            return None

        # The first artifact is typically the screenshot
        artifact_id = artifact_ids[0] if artifact_ids else None
        if not artifact_id:
            return None

        # Try common locations
        if artifact_dir:
            base = Path(artifact_dir)
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = base / f"{artifact_id}{ext}"
                if candidate.is_file():
                    return str(candidate)

        # Check metadata_json for explicit path
        metadata_raw = event.get("metadata_json", "{}")
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        screenshot_path = metadata.get("screenshot_path")
        if screenshot_path and Path(screenshot_path).is_file():
            return screenshot_path

        return None

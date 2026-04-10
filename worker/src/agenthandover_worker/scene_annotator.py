"""VLM scene annotation for captured screenshots.

Processes DwellSnapshot/ScrollReadSnapshot frames through a local VLM
(qwen3.5:2b with think=False) to produce structured semantic annotations.
Each annotation describes what the user is doing, what's visible on screen,
and whether the activity is part of a meaningful workflow.

Key features:
- 3-frame sliding window context (time-bounded to 10 minutes)
- Stale-frame skipping: 3+ consecutive same-app non-workflow → skip
- JSON validation with markdown-fence stripping and one retry
- Screenshot lifecycle: delete JPEG after VLM processes it (success or failure)
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
    num_predict: int = 1500
    stale_skip_count: int = 3
    sliding_window_size: int = 3
    sliding_window_max_age_sec: int = 600
    delete_screenshot_after_processing: bool = True


@dataclass
class AnnotationResult:
    """Result of a single frame annotation."""

    event_id: str
    status: str  # completed, failed, skipped, missing_screenshot
    annotation: dict | None = None
    error: str | None = None
    inference_time_seconds: float = 0.0
    visual_text_proxy: str | None = None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a screen annotation assistant. You analyze screenshots and "
    "extract structured information about what the user is doing. "
    "Respond with ONLY valid JSON, no markdown fences, no commentary."
)

ANNOTATION_PROMPT_TEMPLATE = """\
Analyze this screenshot and extract DETAILED information as JSON.
Be specific — include exact text, names, email addresses, URLs, numbers, and values visible on screen.

{{
  "app": "<exact application name visible on screen>",
  "location": "<full URL, file path, or app-specific location/page>",
  "visible_content": {{
    "headings": ["<main headings or titles visible>"],
    "labels": ["<form labels, button text, menu items, tab names>"],
    "values": ["<ALL filled form values: email addresses, names, typed text, selected options, prices, dates, counts>"]
  }},
  "ui_state": {{
    "active_element": "<what element has focus — include the exact text/value in it>",
    "modals_or_popups": "<any overlays, dropdowns, or dialogs visible — include their content>",
    "scroll_position": "<top, middle, bottom>"
  }},
  "key_text": {{
    "email_addresses": ["<any email addresses visible on screen>"],
    "urls": ["<any URLs or links visible>"],
    "names": ["<people names, company names, product names>"],
    "typed_text": "<exact text the user is typing or has typed in any input field>",
    "selected_text": "<any highlighted or selected text>"
  }},
  "compose": {{
    "is_compose_window_open": <true|false: is an email/message compose window visible AT ALL>,
    "recipient": "<exact text in the To/Recipient field, including chip names like 'Sandro Andric (incentive.ae)'. Empty string if no compose window.>",
    "subject": "<exact text in the Subject field as visible on screen. Empty string if no compose window or subject is empty.>",
    "body_first_line": "<the first line of the email body, exactly as visible on screen (e.g. 'Hi Sandro,' or 'Top news:'). Empty string if no body text visible.>",
    "send_button_state": "<'visible', 'focused', 'just_clicked', or 'not_visible'>"
  }},
  "task_context": {{
    "what_doing": "<one detailed sentence: what specific task is the user performing — include names, recipients, subjects>",
    "likely_next": "<one sentence: what will the user probably do next>",
    "is_workflow": <true if this is a structured, repeatable task; false for browsing/chatting/reading>
  }}
}}

CRITICAL FOR COMPOSE WINDOWS: When the screenshot shows an open email
compose window (Gmail, Outlook, Mail.app, Slack DM, etc.), you MUST fill
the ``compose`` block with VERBATIM text from the actual compose fields:
- ``recipient``: copy the EXACT text in the To field, including any
  contact-chip suffixes like "(incentive.ae)" or "<email@domain>"
- ``subject``: copy the EXACT subject line text as displayed
- ``body_first_line``: copy the FIRST visible line of the message body
Do NOT paraphrase or summarize.  Do NOT confuse it with text from the
inbox list panel that's still visible alongside the compose window.
If a compose window is NOT visible, set is_compose_window_open=false
and leave the other compose fields as empty strings.

{ocr_section}{context_section}IMPORTANT: Whenever an OCR TEXT section is provided above, treat it as \
ground truth for ANY text field (email_addresses, urls, typed_text, values, \
active_element, selected_text, names, subjects, counts).  The OCR was produced \
by the OS's text extraction at confidence 1.0 and is exact.  Visual reading \
of the screenshot is often blurry or downscaled — the OCR is the source of \
truth when the two disagree.

Respond with ONLY the JSON object. No markdown, no explanation."""

OCR_TEMPLATE = """\
OCR TEXT (high-confidence text extracted by the OS for this exact frame — \
treat as ground truth, not visual guess):
{ocr_text}

"""

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


def _extract_ocr_text_from_event(event: dict, max_chars: int = 4000) -> str:
    """Pull high-confidence OCR text out of the event's ``metadata_json``.

    The daemon writes OCR results as::

        metadata_json = {
            "ocr": {
                "elements": [
                    {"text": "...", "confidence": 1.0, "bbox_normalized": [...]},
                    ...
                ]
            }
        }

    We extract the ``text`` of each element, preserving daemon order (roughly
    top-to-bottom, left-to-right), and join with newlines.  This gets injected
    into the VLM prompt as ground truth so Gemma doesn't have to re-read
    small/downscaled text visually.

    Historically this was never passed to the VLM prompt — OCR was only used
    for the vector text proxy via ``ocr.full_text`` (a field the daemon never
    writes).  Bug caught 2026-04-10.
    """
    metadata_raw = event.get("metadata_json", "{}")
    try:
        metadata = (
            json.loads(metadata_raw)
            if isinstance(metadata_raw, str)
            else metadata_raw
        )
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(metadata, dict):
        return ""

    ocr_data = metadata.get("ocr")
    if not isinstance(ocr_data, dict):
        return ""

    # Prefer the structured elements list (what the daemon actually writes)
    elements = ocr_data.get("elements")
    if isinstance(elements, list):
        texts: list[str] = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            text = el.get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        combined = "\n".join(texts)
        if combined:
            return combined[:max_chars]

    # Fallback: some older formats may have stored a flat full_text string
    full_text = ocr_data.get("full_text", "")
    if isinstance(full_text, str) and full_text:
        return full_text[:max_chars]

    return ""


def build_annotation_prompt(
    recent_annotations: list[dict] | None = None,
    ocr_text: str = "",
) -> str:
    """Build the full annotation prompt with optional sliding-window context
    and optional OCR ground-truth text for the current frame."""
    context_section = ""
    if recent_annotations:
        context_section = _build_context_section(recent_annotations)

    ocr_section = ""
    if ocr_text:
        ocr_section = OCR_TEMPLATE.format(ocr_text=ocr_text)

    return ANNOTATION_PROMPT_TEMPLATE.format(
        context_section=context_section,
        ocr_section=ocr_section,
    )


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
    num_predict: int = 1500,
    system: str = "",
    timeout: float = 60.0,
    think: bool | str = False,
    extra_options: dict | None = None,
) -> tuple[str, float]:
    """Call Ollama's /api/generate with an optional image.

    Returns (response_text, inference_time_seconds).
    Raises on connection or HTTP errors.

    Args:
        think: False, True, "low", "medium", or "high". Controls reasoning
            depth for models that support it (Gemma 4, Qwen 3.5).
        extra_options: Additional Ollama options (top_k, presence_penalty, etc.)
            merged into the options dict. Use model_profiles.get_profile() to
            get optimal settings per model.
    """
    import urllib.request
    import urllib.error
    import base64

    url = f"{host}/api/generate"

    options = {
        "num_predict": num_predict,
        "num_ctx": 8192,
    }
    if extra_options:
        options.update(extra_options)

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # think must be a top-level parameter, NOT inside options.
        "think": think,
        "options": options,
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

    def __init__(
        self,
        config: AnnotationConfig | None = None,
        image_embedder=None,
        vector_kb=None,
    ) -> None:
        self.config = config or AnnotationConfig()
        self._image_embedder = image_embedder
        self._vector_kb = vector_kb
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
        # Extract daemon OCR text from this event and feed it to the VLM
        # as ground truth. Prevents Gemma from visually misreading small
        # text (email addresses, subject lines, counts) at half resolution.
        ocr_text = _extract_ocr_text_from_event(event)
        prompt = build_annotation_prompt(recent_annotations, ocr_text=ocr_text)

        # --- Call VLM ---
        try:
            from agenthandover_worker.model_profiles import get_profile
            profile = get_profile(self.config.model)
            raw_response, inference_time = _call_ollama_vlm(
                model=self.config.model,
                prompt=prompt,
                image_path=screenshot_path,
                host=self.config.ollama_host,
                num_predict=profile.ann_num_predict,
                system=profile.ann_system or SYSTEM_PROMPT,
                think=profile.ann_think,
                extra_options={
                    k: v for k, v in profile.ann_options().items()
                    if k not in ("num_predict", "num_ctx")
                },
            )
        except ConnectionError as exc:
            self._stats["failed"] += 1
            self._delete_screenshot(screenshot_path)
            return AnnotationResult(
                event_id=event_id,
                status="failed",
                error=f"ollama_connection: {exc}",
            )
        except Exception as exc:
            self._stats["failed"] += 1
            self._delete_screenshot(screenshot_path)
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
                    num_predict=profile.ann_num_predict,
                    system=profile.ann_system or SYSTEM_PROMPT,
                    think=profile.ann_think,
                    extra_options={
                        k: v for k, v in profile.ann_options().items()
                        if k not in ("num_predict", "num_ctx")
                    },
                )
                inference_time += retry_time
                annotation = _validate_annotation(raw_response)
            except Exception:
                pass

        # --- Embed screenshot before deletion (SigLIP image embedding) ---
        if self._image_embedder and self._vector_kb and screenshot_path:
            try:
                vec = self._image_embedder.embed_image(screenshot_path)
                if vec:
                    self._vector_kb.upsert(
                        "visual", event_id, f"image:{event_id}",
                        embedding=vec,
                    )
            except Exception:
                logger.debug("Image embedding failed for %s", event_id, exc_info=True)

        # --- Delete screenshot after VLM processing (success or failure) ---
        # The raw JPEG has no further value — only the structured annotation
        # matters. Leaving unencrypted JPEGs on disk after failed annotations
        # is a privacy risk that maintenance.rs cannot clean up.
        self._delete_screenshot(screenshot_path)

        if annotation is None:
            self._stats["failed"] += 1
            return AnnotationResult(
                event_id=event_id,
                status="failed",
                error="invalid_json_after_retry",
                inference_time_seconds=inference_time,
            )

        # --- Build visual text proxy (only for successful annotations) ---
        # Combines VLM annotation + OCR text into an embeddable text
        # representation.  Stored in annotation JSON so future vector-KB
        # pipelines can embed it even after the screenshot is gone.
        visual_proxy = self._build_visual_text_proxy(event, annotation)

        # --- Update stale tracker ---
        self._stale.update(annotation)

        self._stats["annotated"] += 1
        result = AnnotationResult(
            event_id=event_id,
            status="completed",
            annotation=annotation,
            inference_time_seconds=inference_time,
        )
        if visual_proxy:
            result.visual_text_proxy = visual_proxy
        return result

    # ------------------------------------------------------------------
    # Visual text proxy
    # ------------------------------------------------------------------

    @staticmethod
    def _build_visual_text_proxy(
        event: dict, annotation: dict | None,
    ) -> str | None:
        """Build an embeddable text representation of a visual frame.

        Combines VLM annotation fields (what_doing, app, location,
        ui_elements) with OCR text from event metadata.  The resulting
        string can be embedded via nomic-embed-text as a stand-in for
        the deleted screenshot, enabling future visual similarity search.
        """
        parts: list[str] = []

        # From VLM annotation
        if annotation:
            wd = annotation.get("what_doing", "")
            if wd:
                parts.append(f"Activity: {wd}")
            app = annotation.get("app", "")
            loc = annotation.get("location", "")
            if app:
                parts.append(f"App: {app}")
            if loc:
                parts.append(f"Location: {loc}")
            ui_els = annotation.get("ui_elements_visible", [])
            if ui_els:
                parts.append(f"UI: {', '.join(str(e) for e in ui_els[:10])}")

        # From OCR (stored in event metadata by daemon)
        metadata_raw = event.get("metadata_json", "{}")
        try:
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        ocr_data = metadata.get("ocr", {})
        if isinstance(ocr_data, dict):
            ocr_text = ocr_data.get("full_text", "")
            if ocr_text:
                # Cap at 2000 chars — enough for embedding, not wasteful
                parts.append(f"OCR: {ocr_text[:2000]}")

        if not parts:
            return None

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Screenshot cleanup
    # ------------------------------------------------------------------

    def _delete_screenshot(self, screenshot_path: str | None) -> None:
        """Delete a screenshot file if configured to do so.

        Called after the VLM has processed the image, regardless of whether
        the annotation succeeded or failed.  The annotation JSON is what
        matters — the raw JPEG has no further value.
        """
        if not self.config.delete_screenshot_after_processing or not screenshot_path:
            return
        try:
            Path(screenshot_path).unlink(missing_ok=True)
            logger.debug("Deleted screenshot after processing: %s", screenshot_path)
        except OSError:
            logger.debug("Failed to delete screenshot %s", screenshot_path, exc_info=True)

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

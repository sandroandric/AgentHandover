"""Semantic SOP generator from VLM annotations and frame diffs.

Replaces PrefixSpan-based SOP induction for the v2 pipeline.  Takes a
sequence of scene annotations + frame diffs (produced by scene_annotator
and frame_differ) and generates a structured, human-readable SOP using
qwen3.5:4b with thinking enabled.

Two modes:
  - **Focus** (single demonstration): generates SOP from one recorded
    walkthrough.  No minimum repeat count — the user explicitly said
    "this is a task".
  - **Passive** (multi-demonstration): generates SOP from 2+ observed
    task segments identified by the task segmenter.  Detects variables
    by comparing values across demonstrations.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from oc_apprentice_worker.confidence import compute_v2_confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SOPGeneratorConfig:
    """Configuration for the VLM-based SOP generator."""

    model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 8000
    timeout: float = 180.0  # 3 minutes for 4B thinking


@dataclass
class GeneratedSOP:
    """Result of SOP generation."""

    sop: dict  # SOP template dict compatible with SkillMdWriter
    inference_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SOP_SYSTEM_PROMPT = """\
You are a workflow documentation expert.  You analyze sequences of \
screen observations and produce clear, actionable Standard Operating \
Procedures (SOPs) that another person or AI agent can follow to \
reproduce the same task.

Write SOPs that are:
- Specific: include exact URLs, field names, button labels, values
- Actionable: each step is a concrete action (click, type, navigate)
- Verified: each step includes how to confirm it worked
- Universal: use application names and locations, not DOM selectors

Respond with ONLY valid JSON.  No markdown fences, no commentary."""

FOCUS_SOP_PROMPT = """\
The user performed a task called "{title}" and I recorded the following \
timeline of screen observations.  Generate a complete SOP from this \
single demonstration.

TIMELINE ({frame_count} frames):
{timeline_text}

Generate a JSON SOP with this exact structure:
{{
  "title": "<human-readable task name>",
  "description": "<1-2 sentence description of what this task accomplishes>",
  "when_to_use": "<when should someone perform this task>",
  "prerequisites": ["<what must be true before starting>"],
  "steps": [
    {{
      "step_number": 1,
      "action": "<verb phrase: what to do>",
      "app": "<application name>",
      "location": "<URL, file path, or screen location>",
      "input": "<text to type or value to enter, if any>",
      "verify": "<how to confirm this step succeeded>"
    }}
  ],
  "success_criteria": ["<how to confirm the entire task succeeded>"],
  "variables": [
    {{
      "name": "<variable_name>",
      "description": "<what this variable represents>",
      "example": "<example value from the demonstration>"
    }}
  ],
  "common_errors": ["<potential failure modes and how to recover>"],
  "apps_involved": ["<list of applications used>"]
}}

Rules:
- Include EVERY meaningful step (navigation, data entry, confirmation)
- Skip redundant frames (reading, scrolling without action)
- Extract variables: values that would change each time (dates, amounts, \
names) should be marked as {{{{variable_name}}}}
- Include verify steps so an agent can confirm success
- The "input" field should contain the EXACT text typed or selected

Respond with ONLY the JSON object."""

PASSIVE_SOP_PROMPT = """\
I observed the following task performed {demo_count} times.  Generate a \
SOP by comparing the demonstrations and identifying the common steps and \
variables.

{demonstrations_text}

Generate a JSON SOP with this exact structure:
{{
  "title": "<human-readable task name>",
  "description": "<1-2 sentence description>",
  "when_to_use": "<when to perform this task>",
  "prerequisites": ["<preconditions>"],
  "steps": [
    {{
      "step_number": 1,
      "action": "<verb phrase>",
      "app": "<application>",
      "location": "<URL or location>",
      "input": "<text/value to enter, use {{{{variable}}}} for parts that differ>",
      "verify": "<confirmation check>"
    }}
  ],
  "success_criteria": ["<overall success checks>"],
  "variables": [
    {{
      "name": "<variable_name>",
      "description": "<what varies>",
      "example": "<example from demonstration 1>"
    }}
  ],
  "common_errors": ["<failure modes>"],
  "apps_involved": ["<applications used>"]
}}

Rules:
- Steps that appear in ALL demonstrations are required steps
- Values that DIFFER between demonstrations become {{{{variables}}}}
- Values that are CONSTANT across demonstrations are hardcoded
- Order steps by the most common sequence
- Skip noise frames (reading, idle, unrelated browsing)

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Timeline formatting
# ---------------------------------------------------------------------------

def _format_timeline_entry(
    idx: int,
    annotation: dict,
    diff: dict | None,
    timestamp: str,
) -> str:
    """Format a single timeline entry for the SOP generation prompt."""
    lines = []

    # Time + app
    ts_short = timestamp[11:19] if len(timestamp) > 19 else timestamp
    app = annotation.get("app", "Unknown")
    location = annotation.get("location", "")
    what_doing = annotation.get("task_context", {}).get("what_doing", "")

    lines.append(f"[{ts_short}] Frame {idx + 1}: {app}")
    if location:
        lines.append(f"  Location: {location}")
    if what_doing:
        lines.append(f"  Activity: {what_doing}")

    # Visible content
    vc = annotation.get("visible_content", {})
    if isinstance(vc, dict):
        headings = vc.get("headings", [])
        if headings:
            lines.append(f"  Headings: {', '.join(str(h) for h in headings[:5])}")
        labels = vc.get("labels", [])
        if labels:
            lines.append(f"  Labels: {', '.join(str(l) for l in labels[:8])}")
        values = vc.get("values", [])
        if values:
            lines.append(f"  Values: {', '.join(str(v) for v in values[:8])}")

    # UI state
    ui = annotation.get("ui_state", {})
    if isinstance(ui, dict):
        active = ui.get("active_element", "")
        if active:
            lines.append(f"  Active: {active}")

    # Frame diff (what changed since last frame)
    if diff and isinstance(diff, dict):
        diff_type = diff.get("diff_type", "")
        if diff_type == "action":
            actions = diff.get("actions", [])
            if actions:
                lines.append("  Actions since previous frame:")
                for a in actions[:6]:
                    lines.append(f"    - {a}")
            inputs = diff.get("inputs", [])
            if inputs:
                lines.append("  Inputs:")
                for inp in inputs[:6]:
                    field_name = inp.get("field", "?")
                    value = inp.get("value", "?")
                    lines.append(f"    - {field_name}: {value}")
            step_desc = diff.get("step_description", "")
            if step_desc:
                lines.append(f"  Summary: {step_desc}")
        elif diff_type == "app_switch":
            from_app = diff.get("from_app", "?")
            to_app = diff.get("to_app", "?")
            lines.append(f"  [Switched from {from_app} to {to_app}]")
        elif diff_type == "no_change":
            lines.append("  [No visible change]")

    return "\n".join(lines)


def _build_focus_prompt(
    title: str,
    timeline: list[dict],
) -> str:
    """Build the SOP generation prompt for a focus recording session."""
    entries = []
    for i, frame in enumerate(timeline):
        annotation = frame.get("annotation", {})
        diff = frame.get("diff")
        timestamp = frame.get("timestamp", "")
        entries.append(_format_timeline_entry(i, annotation, diff, timestamp))

    timeline_text = "\n\n".join(entries)

    return FOCUS_SOP_PROMPT.format(
        title=title,
        frame_count=len(timeline),
        timeline_text=timeline_text,
    )


def _build_passive_prompt(
    demonstrations: list[list[dict]],
) -> str:
    """Build the SOP generation prompt for passive discovery (multi-demo)."""
    demo_texts = []
    for demo_idx, timeline in enumerate(demonstrations):
        entries = []
        for i, frame in enumerate(timeline):
            annotation = frame.get("annotation", {})
            diff = frame.get("diff")
            timestamp = frame.get("timestamp", "")
            entries.append(
                _format_timeline_entry(i, annotation, diff, timestamp)
            )
        demo_text = f"--- Demonstration {demo_idx + 1} ({len(timeline)} frames) ---\n"
        demo_text += "\n\n".join(entries)
        demo_texts.append(demo_text)

    return PASSIVE_SOP_PROMPT.format(
        demo_count=len(demonstrations),
        demonstrations_text="\n\n".join(demo_texts),
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_sop_response(raw: str) -> dict | None:
    """Parse the VLM's JSON response into a SOP dict.

    Handles markdown fences, thinking tags, and basic validation.
    """
    # Strip thinking tags
    text = _THINK_RE.sub("", raw).strip()

    # Try to extract from markdown fences
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()

    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    data = json.loads(text[i:])
                    break
                except json.JSONDecodeError:
                    continue
        else:
            return None

    if not isinstance(data, dict):
        return None

    # Minimal validation: must have title and steps
    if "title" not in data:
        return None
    if "steps" not in data or not isinstance(data["steps"], list):
        return None

    return data


def _vlm_sop_to_template(
    vlm_sop: dict,
    *,
    mode: str = "focus",
    title_override: str | None = None,
    confidence: float = 0.0,
) -> dict:
    """Convert the VLM's raw SOP JSON into the internal SOP template format.

    The template format is compatible with SkillMdWriter and OpenClawWriter.
    """
    title = title_override or vlm_sop.get("title", "Untitled")
    slug = _generate_slug(title)

    # Convert VLM steps to internal format
    steps = []
    for raw_step in vlm_sop.get("steps", []):
        step = {
            "step": raw_step.get("action", "action"),
            "target": raw_step.get("location", ""),
            "selector": None,  # v2 SOPs are semantic, not DOM-based
            "parameters": {},
            "confidence": confidence,
            "pre_state": {},
        }

        # Populate parameters from step fields
        if raw_step.get("app"):
            step["parameters"]["app"] = raw_step["app"]
        if raw_step.get("input"):
            step["parameters"]["input"] = raw_step["input"]
        if raw_step.get("verify"):
            step["parameters"]["verify"] = raw_step["verify"]
        if raw_step.get("location"):
            step["parameters"]["location"] = raw_step["location"]

        steps.append(step)

    # Convert VLM variables to internal format
    variables = []
    for raw_var in vlm_sop.get("variables", []):
        variables.append({
            "name": raw_var.get("name", "unknown"),
            "type": "string",
            "example": raw_var.get("example", ""),
            "default": "",
            "description": raw_var.get("description", ""),
        })

    # Build apps list
    apps_involved = vlm_sop.get("apps_involved", [])
    if not apps_involved:
        # Extract from steps
        seen: set[str] = set()
        for raw_step in vlm_sop.get("steps", []):
            app = raw_step.get("app", "")
            if app and app not in seen:
                apps_involved.append(app)
                seen.add(app)

    # Build preconditions
    preconditions = []
    for prereq in vlm_sop.get("prerequisites", []):
        if isinstance(prereq, str):
            preconditions.append(prereq)

    # Build task_description from VLM output
    task_description = vlm_sop.get("description", "")

    # Build execution_overview from VLM output
    execution_overview = {}
    if vlm_sop.get("when_to_use"):
        execution_overview["when_to_use"] = vlm_sop["when_to_use"]
    if vlm_sop.get("success_criteria"):
        sc = vlm_sop["success_criteria"]
        if isinstance(sc, list):
            execution_overview["success_criteria"] = "; ".join(str(s) for s in sc)
        else:
            execution_overview["success_criteria"] = str(sc)
    if vlm_sop.get("common_errors"):
        ce = vlm_sop["common_errors"]
        if isinstance(ce, list):
            execution_overview["common_errors"] = "; ".join(str(e) for e in ce)
        else:
            execution_overview["common_errors"] = str(ce)
    if preconditions:
        execution_overview["prerequisites"] = "; ".join(preconditions)

    template = {
        "slug": slug,
        "title": title,
        "steps": steps,
        "variables": variables,
        "confidence_avg": 0.0,  # Set by compute_v2_confidence below
        "episode_count": 1 if mode == "focus" else 0,
        "abs_support": 1 if mode == "focus" else 2,
        "apps_involved": apps_involved,
        "preconditions": preconditions,
        "task_description": task_description,
        "execution_overview": execution_overview,
        "source": "v2_focus_recording" if mode == "focus" else "v2_passive_discovery",
    }

    # Compute v2 confidence score
    breakdown = compute_v2_confidence(
        template,
        is_focus=(mode == "focus"),
    )
    template["confidence_avg"] = breakdown.total
    template["confidence_breakdown"] = {
        "demo_count": breakdown.demo_count_score,
        "step_consistency": breakdown.step_consistency_score,
        "annotation_quality": breakdown.annotation_quality_score,
        "variable_detection": breakdown.variable_detection_score,
        "focus_bonus": breakdown.focus_bonus,
        "reasons": breakdown.reasons,
    }

    return template


def _generate_slug(title: str) -> str:
    """Generate a filesystem-safe slug from a title."""
    import unicodedata

    slug = unicodedata.normalize("NFKD", title)
    slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:80] if slug else "untitled"


# ---------------------------------------------------------------------------
# Ollama client (reused from scene_annotator)
# ---------------------------------------------------------------------------

def _call_ollama(
    *,
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    num_predict: int = 8000,
    system: str = "",
    timeout: float = 180.0,
    think: bool = True,
) -> tuple[str, float]:
    """Call Ollama's /api/generate for text-only SOP generation.

    Uses think=True for 4B model to enable reasoning (produces better SOPs).
    Returns (response_text, inference_time_seconds).
    """
    import urllib.request
    import urllib.error

    url = f"{host}/api/generate"

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "think": think,
        },
    }

    if system:
        payload["system"] = system

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
# SOPGenerator
# ---------------------------------------------------------------------------

class SOPGenerator:
    """Generate semantic SOPs from VLM annotations and frame diffs.

    Uses qwen3.5:4b (think=True) for high-quality SOP generation.
    Designed to run as the final stage in the v2 pipeline.
    """

    def __init__(self, config: SOPGeneratorConfig | None = None) -> None:
        self.config = config or SOPGeneratorConfig()

    def generate_from_focus(
        self,
        timeline: list[dict],
        title: str,
    ) -> GeneratedSOP:
        """Generate a SOP from a focus recording session.

        Args:
            timeline: List of dicts, each with:
                - annotation: parsed scene annotation dict
                - diff: parsed frame diff dict (or None)
                - timestamp: ISO timestamp string
            title: User-provided task name.

        Returns:
            GeneratedSOP with the template dict.
        """
        if not timeline:
            return GeneratedSOP(
                sop={},
                success=False,
                error="Empty timeline",
            )

        # Filter out frames with no meaningful annotation
        meaningful = [
            f for f in timeline
            if f.get("annotation") and isinstance(f["annotation"], dict)
        ]
        if not meaningful:
            return GeneratedSOP(
                sop={},
                success=False,
                error="No annotated frames in timeline",
            )

        # Build prompt
        prompt = _build_focus_prompt(title, meaningful)

        # Call VLM
        try:
            raw_response, elapsed = _call_ollama(
                model=self.config.model,
                prompt=prompt,
                host=self.config.ollama_host,
                num_predict=self.config.num_predict,
                system=SOP_SYSTEM_PROMPT,
                timeout=self.config.timeout,
                think=True,
            )
        except ConnectionError as exc:
            return GeneratedSOP(
                sop={},
                success=False,
                error=f"VLM connection failed: {exc}",
            )
        except Exception as exc:
            return GeneratedSOP(
                sop={},
                success=False,
                error=f"VLM call failed: {exc}",
            )

        # Parse response
        vlm_sop = _parse_sop_response(raw_response)
        if vlm_sop is None:
            # Retry with explicit JSON-only instruction
            logger.warning(
                "SOP generation: invalid JSON on first attempt, retrying "
                "with JSON-only suffix"
            )
            try:
                raw_response2, elapsed2 = _call_ollama(
                    model=self.config.model,
                    prompt=prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON.",
                    host=self.config.ollama_host,
                    num_predict=self.config.num_predict,
                    system=SOP_SYSTEM_PROMPT,
                    timeout=self.config.timeout,
                    think=True,
                )
                elapsed += elapsed2
                vlm_sop = _parse_sop_response(raw_response2)
            except Exception:
                pass

            if vlm_sop is None:
                return GeneratedSOP(
                    sop={},
                    inference_time_seconds=elapsed,
                    success=False,
                    error="Failed to parse VLM response as valid SOP JSON",
                )

        # Convert to internal template format
        template = _vlm_sop_to_template(
            vlm_sop,
            mode="focus",
            title_override=title,
        )

        # Enrich confidence with annotation data from the timeline
        annotations = [
            f.get("annotation", {}) for f in meaningful
            if isinstance(f.get("annotation"), dict)
        ]
        if annotations:
            breakdown = compute_v2_confidence(
                template,
                demonstrations=[meaningful],
                annotations=annotations,
                is_focus=True,
            )
            template["confidence_avg"] = breakdown.total
            template["confidence_breakdown"] = {
                "demo_count": breakdown.demo_count_score,
                "step_consistency": breakdown.step_consistency_score,
                "annotation_quality": breakdown.annotation_quality_score,
                "variable_detection": breakdown.variable_detection_score,
                "focus_bonus": breakdown.focus_bonus,
                "reasons": breakdown.reasons,
            }

        logger.info(
            "Generated focus SOP '%s': %d steps, confidence=%.2f, %.1fs",
            title,
            len(template["steps"]),
            template["confidence_avg"],
            elapsed,
        )

        return GeneratedSOP(
            sop=template,
            inference_time_seconds=elapsed,
        )

    def generate_from_passive(
        self,
        demonstrations: list[list[dict]],
        task_label: str | None = None,
    ) -> GeneratedSOP:
        """Generate a SOP from multiple passive observations.

        Args:
            demonstrations: List of timelines (each a list of frame dicts).
            task_label: Optional cluster label for the task.

        Returns:
            GeneratedSOP with the template dict.
        """
        if len(demonstrations) < 2:
            return GeneratedSOP(
                sop={},
                success=False,
                error="Need at least 2 demonstrations for passive SOP",
            )

        # Build prompt
        prompt = _build_passive_prompt(demonstrations)

        # Call VLM
        try:
            raw_response, elapsed = _call_ollama(
                model=self.config.model,
                prompt=prompt,
                host=self.config.ollama_host,
                num_predict=self.config.num_predict,
                system=SOP_SYSTEM_PROMPT,
                timeout=self.config.timeout,
                think=True,
            )
        except Exception as exc:
            return GeneratedSOP(
                sop={},
                success=False,
                error=f"VLM call failed: {exc}",
            )

        # Parse response
        vlm_sop = _parse_sop_response(raw_response)
        if vlm_sop is None:
            return GeneratedSOP(
                sop={},
                inference_time_seconds=elapsed,
                success=False,
                error="Failed to parse VLM response as valid SOP JSON",
            )

        # Override title with cluster label if provided
        title = task_label or vlm_sop.get("title", "Untitled Workflow")

        template = _vlm_sop_to_template(
            vlm_sop,
            mode="passive",
            title_override=title,
        )
        template["episode_count"] = len(demonstrations)
        template["abs_support"] = len(demonstrations)

        # Collect all annotations across demonstrations for quality scoring
        all_annotations: list[dict] = []
        for demo in demonstrations:
            for frame in demo:
                ann = frame.get("annotation")
                if isinstance(ann, dict):
                    all_annotations.append(ann)

        # Compute v2 confidence with full demonstration data
        breakdown = compute_v2_confidence(
            template,
            demonstrations=demonstrations,
            annotations=all_annotations if all_annotations else None,
            is_focus=False,
        )
        template["confidence_avg"] = breakdown.total
        template["confidence_breakdown"] = {
            "demo_count": breakdown.demo_count_score,
            "step_consistency": breakdown.step_consistency_score,
            "annotation_quality": breakdown.annotation_quality_score,
            "variable_detection": breakdown.variable_detection_score,
            "focus_bonus": breakdown.focus_bonus,
            "reasons": breakdown.reasons,
        }

        logger.info(
            "Generated passive SOP '%s': %d steps from %d demos, "
            "confidence=%.2f, %.1fs",
            title,
            len(template["steps"]),
            len(demonstrations),
            template["confidence_avg"],
            elapsed,
        )

        return GeneratedSOP(
            sop=template,
            inference_time_seconds=elapsed,
        )

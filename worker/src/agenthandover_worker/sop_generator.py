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

from agenthandover_worker.confidence import compute_v2_confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SOPGeneratorConfig:
    """Configuration for the VLM-based SOP generator."""

    model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 12000
    timeout: float = 1800.0  # 30 min — 16GB machines need 10+ min per call
    max_timeline_frames: int = 20  # Cap frames sent to SOP gen to keep prompt manageable


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
  "short_title": "<concise 3-6 word label for the task, e.g. 'Check Gmail inbox', 'Deploy to production', 'Search expired domains'>",
  "tags": ["<1-3 category tags from: communication, development, browsing, documentation, system, finance, design, data, testing, deployment>"],
  "description": "<1-2 sentence description of what this task accomplishes>",
  "outcome": "<single sentence: what the user achieves when this workflow completes successfully>",
  "when_to_use": "<when should someone perform this task>",
  "prerequisites": ["<what the user needs before starting: accounts, tools, files, permissions, etc.>"],
  "steps": [
    {{
      "step_number": 1,
      "action": "<verb phrase: what to do>",
      "app": "<application name>",
      "location": "<URL, file path, or screen location>",
      "input": "<text to type or value to enter, if any>",
      "verify": "<concrete check to confirm this step succeeded — what should the user see or be able to confirm?>"
    }}
  ],
  "success_criteria": ["<how to confirm the entire task succeeded>"],
  "variables": [
    {{
      "name": "<variable_name>",
      "type": "<one of: text, email, url, number, date, filepath, password, selection>",
      "example": "<example value from the demonstration>",
      "description": "<what this variable represents>",
      "required": true,
      "sensitive": false,
      "validation": "<optional hint for valid values, e.g. 'Must be a valid email format'>"
    }}
  ],
  "common_errors": ["<plain text description of potential failure and how to recover — NOT a dict or object>"],
  "apps_involved": ["<list of applications used>"]
}}

Rules:
- Include EVERY meaningful step (navigation, data entry, confirmation)
- Skip redundant frames (reading, scrolling without action)
- Extract variables: values that would change each time (dates, amounts, \
names) should be marked as {{{{variable_name}}}}
- The "outcome" is a single sentence describing the end result of the whole workflow
- The "prerequisites" list things the user needs before starting (accounts, \
permissions, files, tools — not steps)
- Each step's "verify" must be a concrete, observable check (what appears on \
screen, what changes, what confirms success)
- The "input" field should contain the EXACT text typed or selected
- Variable types: text (default, free-form), email, url, number, date, \
filepath, password (always sensitive=true), selection (one of a set)
- Mark sensitive=true for passwords, API keys, tokens, SSNs, credit cards
- Use type "password" for credentials (these are always sensitive)
- Ignore any frames where the user is configuring the recording tool \
(e.g., AgentHandover settings, starting/stopping recording, checking daemon \
status). Focus only on the actual task being performed.
- IMPORTANT: Frames marked with ">>> COPY ACTION" indicate the user \
copied content to the clipboard. You MUST include these as explicit \
steps: one step for the Copy action (what was copied and from where) \
and a subsequent step for the Paste action (where it was pasted). \
Example: 'Copy the domain name from the registrar page' followed by \
'Paste the domain name into the DNS settings field'.

Respond with ONLY the JSON object."""

PASSIVE_SOP_PROMPT = """\
I observed the following task performed {demo_count} times.  Generate a \
SOP by comparing the demonstrations and identifying the common steps and \
variables.

{demonstrations_text}

Generate a JSON SOP with this exact structure:
{{
  "title": "<human-readable task name>",
  "short_title": "<concise 3-6 word label for the task, e.g. 'Check Gmail inbox', 'Deploy to production', 'Search expired domains'>",
  "tags": ["<1-3 category tags from: communication, development, browsing, documentation, system, finance, design, data, testing, deployment>"],
  "description": "<1-2 sentence description>",
  "outcome": "<single sentence: what the user achieves when this workflow completes successfully>",
  "when_to_use": "<when to perform this task>",
  "prerequisites": ["<what the user needs before starting: accounts, tools, files, permissions, etc.>"],
  "steps": [
    {{
      "step_number": 1,
      "action": "<verb phrase>",
      "app": "<application>",
      "location": "<URL or location>",
      "input": "<text/value to enter, use {{{{variable}}}} for parts that differ>",
      "verify": "<concrete check to confirm this step succeeded — what should the user see or confirm?>"
    }}
  ],
  "success_criteria": ["<overall success checks>"],
  "variables": [
    {{
      "name": "<variable_name>",
      "type": "<one of: text, email, url, number, date, filepath, password, selection>",
      "example": "<example from demonstration 1>",
      "description": "<what varies>",
      "required": true,
      "sensitive": false,
      "validation": "<optional hint for valid values>"
    }}
  ],
  "common_errors": ["<plain text description of potential failure and how to recover — NOT a dict or object>"],
  "apps_involved": ["<applications used>"]
}}

Rules:
- Steps that appear in ALL demonstrations are required steps
- Values that DIFFER between demonstrations become {{{{variables}}}}
- Values that are CONSTANT across demonstrations are hardcoded
- Order steps by the most common sequence
- Skip noise frames (reading, idle, unrelated browsing)
- Variable types: text (default, free-form), email, url, number, date, \
filepath, password (always sensitive=true), selection (one of a set)
- Mark sensitive=true for passwords, API keys, tokens, SSNs, credit cards
- Use type "password" for credentials (these are always sensitive)

Respond with ONLY the JSON object."""


ENRICHED_PASSIVE_PROMPT = """\
I observed the following task performed {demo_count} times.  Pre-analysis \
has already identified the canonical steps, extracted parameters, and \
detected branch points.

PRE-ANALYZED CANONICAL STEPS:
{canonical_steps_text}

EXTRACTED PARAMETERS (values that vary across demonstrations):
{parameters_text}

DETECTED BRANCH POINTS (where demonstrations diverge):
{branches_text}

RAW DEMONSTRATIONS (for context):
{demonstrations_text}

Given the pre-analyzed structure above, generate a JSON SOP that focuses \
on the STRATEGY and DECISION LOGIC behind these steps, not just the \
mechanics.

Generate a JSON SOP with this exact structure:
{{
  "title": "<human-readable task name>",
  "short_title": "<concise 3-6 word label>",
  "tags": ["<1-3 category tags from: communication, development, browsing, documentation, system, finance, design, data, testing, deployment>"],
  "description": "<1-2 sentence description of the STRATEGY, not just the mechanics>",
  "outcome": "<what the user achieves when this workflow completes>",
  "when_to_use": "<conditions that trigger this workflow>",
  "prerequisites": ["<what the user needs before starting>"],
  "steps": [
    {{
      "step_number": 1,
      "action": "<verb phrase — use the canonical action as baseline>",
      "app": "<application>",
      "location": "<URL or location>",
      "input": "<text/value, use {{{{variable}}}} for parameters>",
      "verify": "<how to confirm this step succeeded>"
    }}
  ],
  "success_criteria": ["<overall success checks>"],
  "variables": [
    {{
      "name": "<from extracted parameters above>",
      "type": "<text, email, url, number, date, filepath, password, selection>",
      "example": "<example from demonstrations>",
      "description": "<what this variable represents>",
      "required": true,
      "sensitive": false
    }}
  ],
  "common_errors": ["<potential failures and recovery>"],
  "apps_involved": ["<applications used>"]
}}

Rules:
- Use the PRE-ANALYZED CANONICAL STEPS as the foundation — do not reinvent them
- Use the EXTRACTED PARAMETERS as confirmed variables
- Where BRANCH POINTS exist, pick the most common path for the main steps
- Focus your analysis on WHY the user does each step and WHAT determines choices
- Values that are CONSTANT across demonstrations are hardcoded
- Variable types: text, email, url, number, date, filepath, password, selection
- Mark sensitive=true for passwords, API keys, tokens

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Timeline formatting
# ---------------------------------------------------------------------------

def _format_timeline_entry(
    idx: int,
    annotation: dict,
    diff: dict | None,
    timestamp: str,
    clipboard_context: dict | None = None,
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

    # Clipboard context (attached by FocusProcessor._attach_clipboard_context)
    # Rendered as a prominent top-level marker so Qwen treats it as an
    # explicit COPY action that must become a step in the SOP.
    if clipboard_context and isinstance(clipboard_context, dict):
        byte_size = clipboard_context.get("byte_size", 0)
        content_types = clipboard_context.get("content_types", [])
        preview = clipboard_context.get("content_preview", "")
        types_str = ", ".join(str(t) for t in content_types) if content_types else "unknown"
        lines.append(
            f"  >>> COPY ACTION: User copied content to clipboard "
            f"({byte_size} bytes, {types_str})"
        )
        if preview:
            lines.append(f'  >>> Copied text: "{preview[:200]}"')
        lines.append(
            "  >>> (Include this as a Copy step and a subsequent Paste step in the SOP)"
        )

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
        clipboard_ctx = frame.get("clipboard_context")
        entries.append(_format_timeline_entry(
            i, annotation, diff, timestamp,
            clipboard_context=clipboard_ctx,
        ))

    timeline_text = "\n\n".join(entries)

    return FOCUS_SOP_PROMPT.format(
        title=title,
        frame_count=len(timeline),
        timeline_text=timeline_text,
    )


MAX_FRAMES_PER_DEMO = 25


def _sample_demo_frames(timeline: list[dict], demo_idx: int) -> list[dict]:
    """Sample frames from a demonstration to stay within prompt budget.

    Strategy:
    - Skip frames where the diff is None or indicates "no_change" (noise).
    - If the remaining meaningful frames exceed MAX_FRAMES_PER_DEMO, sample
      evenly while always keeping the first and last frames.
    """
    # Filter out noise frames (no diff or no_change), but always keep first/last
    meaningful: list[tuple[int, dict]] = []
    for i, frame in enumerate(timeline):
        diff = frame.get("diff")
        is_first = i == 0
        is_last = i == len(timeline) - 1
        is_noise = (
            diff is None
            or (isinstance(diff, dict) and diff.get("diff_type") == "no_change")
        )
        if is_first or is_last or not is_noise:
            meaningful.append((i, frame))

    total = len(meaningful)
    cap = MAX_FRAMES_PER_DEMO

    if total <= cap:
        return [f for _, f in meaningful]

    # Evenly sample while keeping first and last
    logger.info(
        "Sampled %d of %d frames for demo %d",
        cap, len(timeline), demo_idx + 1,
    )
    if cap <= 2:
        return [meaningful[0][1], meaningful[-1][1]]

    indices = [0]
    step = (total - 1) / (cap - 1)
    for i in range(1, cap - 1):
        indices.append(round(i * step))
    indices.append(total - 1)

    # Deduplicate while preserving order
    seen: set[int] = set()
    sampled: list[dict] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            sampled.append(meaningful[idx][1])
    return sampled


def _build_passive_prompt(
    demonstrations: list[list[dict]],
) -> str:
    """Build the SOP generation prompt for passive discovery (multi-demo)."""
    demo_texts = []
    for demo_idx, timeline in enumerate(demonstrations):
        sampled = _sample_demo_frames(timeline, demo_idx)
        entries = []
        for i, frame in enumerate(sampled):
            annotation = frame.get("annotation", {})
            diff = frame.get("diff")
            timestamp = frame.get("timestamp", "")
            clipboard_ctx = frame.get("clipboard_context")
            entries.append(
                _format_timeline_entry(
                    i, annotation, diff, timestamp,
                    clipboard_context=clipboard_ctx,
                )
            )
        demo_text = f"--- Demonstration {demo_idx + 1} ({len(timeline)} frames) ---\n"
        demo_text += "\n\n".join(entries)
        demo_texts.append(demo_text)

    return PASSIVE_SOP_PROMPT.format(
        demo_count=len(demonstrations),
        demonstrations_text="\n\n".join(demo_texts),
    )


def _build_enriched_passive_prompt(
    demonstrations: list[list[dict]],
    canonical_steps: list[dict],
    parameters: list,
    branches: list[dict],
    evidence_context: dict | None = None,
) -> str:
    """Build the enriched SOP generation prompt with pre-analyzed data.

    Includes canonical steps, extracted parameters, and branch points
    so the VLM can focus on strategy and decision logic.
    """
    # Format canonical steps
    canon_lines = []
    for cs in canonical_steps:
        action = cs.get("action", "")
        conf = cs.get("confidence", 0.0)
        alts = cs.get("alternatives", [])
        line = f"  {cs.get('position', '?')}. {action} (confidence: {conf:.0%})"
        if alts:
            alt_strs = [f"{a.get('action', '?')} ({a.get('observation_count', 0)}x)" for a in alts]
            line += f"  [alternatives: {', '.join(alt_strs)}]"
        canon_lines.append(line)
    canonical_steps_text = "\n".join(canon_lines) if canon_lines else "(none detected)"

    # Format parameters
    from dataclasses import asdict
    param_lines = []
    for p in parameters:
        p_dict = asdict(p) if hasattr(p, "__dataclass_fields__") else p
        name = p_dict.get("name", "?")
        ptype = p_dict.get("type", "text")
        values = p_dict.get("values_seen", [])
        positions = p_dict.get("step_positions", [])
        param_lines.append(
            f"  - {name} (type: {ptype}, at steps: {positions}): "
            f"seen values: {values[:5]}"
        )
    parameters_text = "\n".join(param_lines) if param_lines else "(none detected)"

    # Format branches
    branch_lines = []
    for b in branches:
        pos = b.get("position", "?")
        canon = b.get("canonical_action", "?")
        alts = b.get("alternatives", [])
        alt_strs = [f"{a.get('action', '?')} ({a.get('observation_count', 0)}x)" for a in alts]
        branch_lines.append(
            f"  Step {pos}: canonical='{canon}', alternatives=[{', '.join(alt_strs)}]"
        )
    branches_text = "\n".join(branch_lines) if branch_lines else "(none detected)"

    # Evidence section (from prior observations)
    evidence_text = ""
    if evidence_context:
        evidence_parts = []
        timing = evidence_context.get("timing_patterns", {})
        total_dur = timing.get("total_duration_seconds", 0)
        if total_dur > 0:
            evidence_parts.append(f"- Duration: {total_dur / 60:.0f} minutes")
        selection = evidence_context.get("selection_signals", {})
        high_engagement = selection.get("high_engagement_locations", [])
        if high_engagement:
            evidence_parts.append(
                f"- High engagement at: {', '.join(str(loc) for loc in high_engagement[:5])}"
            )
        content = evidence_context.get("content_produced", {})
        content_count = content.get("count", 0)
        content_types = content.get("types", [])
        if content_count > 0:
            evidence_parts.append(
                f"- Content produced: {content_count} items ({', '.join(content_types[:5])})"
            )
        if evidence_parts:
            evidence_text = (
                "\n\nEVIDENCE FROM PRIOR OBSERVATIONS:\n"
                + "\n".join(evidence_parts)
            )

    # Build abbreviated demonstrations text (reuse passive prompt builder)
    demo_texts = []
    for demo_idx, timeline in enumerate(demonstrations):
        sampled = _sample_demo_frames(timeline, demo_idx)
        entries = []
        for i, frame in enumerate(sampled):
            annotation = frame.get("annotation", {})
            diff = frame.get("diff")
            timestamp = frame.get("timestamp", "")
            clipboard_ctx = frame.get("clipboard_context")
            entries.append(
                _format_timeline_entry(
                    i, annotation, diff, timestamp,
                    clipboard_context=clipboard_ctx,
                )
            )
        demo_text = f"--- Demonstration {demo_idx + 1} ({len(timeline)} frames) ---\n"
        demo_text += "\n\n".join(entries)
        demo_texts.append(demo_text)

    prompt = ENRICHED_PASSIVE_PROMPT.format(
        demo_count=len(demonstrations),
        canonical_steps_text=canonical_steps_text,
        parameters_text=parameters_text,
        branches_text=branches_text,
        demonstrations_text="\n\n".join(demo_texts),
    )

    if evidence_text:
        prompt += evidence_text

    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_sop_response(raw: str) -> dict | None:
    """Parse the VLM's JSON response into a SOP dict.

    Handles markdown fences, thinking tags, truncated JSON, and Qwen noise.
    """
    # Strip thinking tags
    text = _THINK_RE.sub("", raw).strip()

    # Strip Qwen end-of-text tokens and trailing noise
    for token in ("<|endoftext|>", "<|im_end|>", "<|end|>"):
        text = text.split(token)[0]
    text = text.strip()

    # Try to extract from markdown fences
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()

    if not text:
        return None

    # Find the first { to locate JSON start
    json_start = text.find("{")
    if json_start < 0:
        return None
    text = text[json_start:]

    # Try direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open brackets
        data = _try_repair_json(text)

    if not isinstance(data, dict):
        return None

    # Minimal validation: must have title and steps
    if "title" not in data:
        return None
    if "steps" not in data or not isinstance(data["steps"], list):
        return None

    return data


def _try_repair_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON from a model that hit its token limit.

    Strategy: progressively trim from the end and close open brackets.
    """
    # Count open brackets
    opens = 0
    open_sq = 0
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_sq += 1
        elif ch == "]":
            open_sq -= 1

    # Try closing open brackets
    if opens > 0 or open_sq > 0:
        # Trim any trailing partial value (incomplete string, number, etc.)
        # Find last complete value (ends with }, ], ", number, true, false, null)
        trimmed = text.rstrip()
        # Remove trailing comma if present
        if trimmed.endswith(","):
            trimmed = trimmed[:-1]

        # Close open brackets
        repair = trimmed + ("]" * max(0, open_sq)) + ("}" * max(0, opens))
        try:
            return json.loads(repair)
        except json.JSONDecodeError:
            pass

    # Try finding a valid JSON substring by trimming from the end
    for end in range(len(text), max(len(text) - 500, 0), -1):
        candidate = text[:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _vlm_sop_to_template(
    vlm_sop: dict,
    *,
    mode: str = "focus",
    title_override: str | None = None,
    confidence: float = 0.0,
    timeline: list[dict] | None = None,
) -> dict:
    """Convert the VLM's raw SOP JSON into the internal SOP template format.

    The template format is compatible with SkillMdWriter and OpenClawWriter.
    """
    title = title_override or vlm_sop.get("title", "Untitled")
    slug = _generate_slug(title)

    # Convert VLM steps to internal format
    steps = []
    for step_idx, raw_step in enumerate(vlm_sop.get("steps", [])):
        # Handle steps that are strings (Qwen sometimes returns plain text)
        if isinstance(raw_step, str):
            raw_step = {"action": raw_step}

        # Qwen returns step descriptions in various field names.
        # Reject literal "action" — it's a placeholder, not a real description.
        def _non_placeholder(val):
            return val and val.strip().lower() not in ("action", "step", "")

        params = raw_step.get("parameters", {}) if isinstance(raw_step.get("parameters"), dict) else {}
        step_text = (
            (raw_step.get("action") if _non_placeholder(raw_step.get("action")) else None)
            or (raw_step.get("description") if _non_placeholder(raw_step.get("description")) else None)
            or (raw_step.get("step") if _non_placeholder(raw_step.get("step")) else None)
            or raw_step.get("step_name")
            or raw_step.get("name")
            or raw_step.get("instruction")
            or params.get("description")
            or params.get("verify")
            or raw_step.get("verify")
            or raw_step.get("target")
            or str(raw_step)
        )
        step = {
            "step": step_text,
            "target": raw_step.get("location", ""),
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

        # Try to extract CSS selector from DOM nodes
        selector = _extract_selector_for_step(raw_step, timeline, step_idx)
        step["selector"] = selector

        steps.append(step)

    # Convert VLM variables to internal format
    # VLM may return variables as dicts or as plain strings
    variables = []
    for raw_var in vlm_sop.get("variables", []):
        if isinstance(raw_var, str):
            # VLM returned a plain string like "query" instead of a dict
            variables.append({
                "name": raw_var,
                "type": "text",
                "example": "",
                "default": "",
                "description": "",
                "required": True,
                "sensitive": False,
                "validation": "",
            })
        elif isinstance(raw_var, dict):
            var_type = raw_var.get("type", "text")
            # Normalise legacy "string" type to "text"
            if var_type == "string":
                var_type = "text"
            sensitive = raw_var.get("sensitive", False)
            # password type is always sensitive
            if var_type == "password":
                sensitive = True
            variables.append({
                "name": raw_var.get("name", "unknown"),
                "type": var_type,
                "example": raw_var.get("example", ""),
                "default": raw_var.get("default", ""),
                "description": raw_var.get("description", ""),
                "required": raw_var.get("required", True),
                "sensitive": sensitive,
                "validation": raw_var.get("validation", ""),
            })
        # Skip any other types silently

    # Build apps list
    apps_involved = vlm_sop.get("apps_involved", [])
    if not apps_involved:
        # Extract from steps
        seen: set[str] = set()
        for raw_step in vlm_sop.get("steps", []):
            if isinstance(raw_step, str):
                raw_step = {"action": raw_step}
            app = raw_step.get("app", "")
            if app and app not in seen:
                apps_involved.append(app)
                seen.add(app)

    # Build preconditions
    preconditions = []
    for prereq in vlm_sop.get("prerequisites", []):
        if isinstance(prereq, str):
            preconditions.append(prereq)

    # Build outcome from VLM output
    outcome = vlm_sop.get("outcome", "")

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

    # Extract short_title and tags from VLM output
    short_title = vlm_sop.get("short_title", "")
    if not short_title:
        # Fallback: derive from title — strip noise prefixes, take core phrase
        _t = title
        for pfx in ("The user is ", "User is ", "The user "):
            if _t.lower().startswith(pfx.lower()):
                _t = _t[len(pfx):]
                break
        if _t:
            _t = _t[0].upper() + _t[1:]
        words = _t.split()
        if len(words) <= 6:
            short_title = _t.rstrip(".")
        else:
            short_title = " ".join(words[:6]).rstrip(".,;:")

    tags = vlm_sop.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif not isinstance(tags, list):
        tags = []
    # Normalise: lowercase, deduplicate, cap at 3
    tags = list(dict.fromkeys(t.lower().strip() for t in tags if isinstance(t, str)))[:3]

    template = {
        "slug": slug,
        "title": title,
        "short_title": short_title,
        "tags": tags,
        "steps": steps,
        "variables": variables,
        "confidence_avg": 0.0,  # Set by compute_v2_confidence below
        "episode_count": 1 if mode == "focus" else 0,
        "abs_support": 1 if mode == "focus" else 2,
        "apps_involved": apps_involved,
        "preconditions": preconditions,
        "outcome": outcome,
        "task_description": task_description,
        "execution_overview": execution_overview,
        "source": "v2_focus_recording" if mode == "focus" else "v2_passive_discovery",
    }

    # Internal: used by skill_md_writer for DOM hints
    template["_timeline"] = timeline

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


def _extract_selector_for_step(
    raw_step: dict,
    timeline: list[dict] | None,
    step_index: int,
) -> str | None:
    """Match a VLM step to DOM nodes and extract the best CSS selector.

    Strategy:
    1. Map step_index to the nearest timeline entry
    2. If it has dom_nodes, search for elements matching the step's action target
    3. Priority: aria-label > data-testid > id > role+tag
    """
    if not timeline:
        return None

    # Map step index to timeline entry (steps may be fewer than frames)
    if step_index < len(timeline):
        entry = timeline[step_index]
    elif timeline:
        entry = timeline[-1]
    else:
        return None

    dom_nodes = entry.get("dom_nodes")
    if not dom_nodes or not isinstance(dom_nodes, list):
        return None

    # What are we looking for?
    if isinstance(raw_step, str):
        raw_step = {"action": raw_step}
    action = raw_step.get("action", "").lower()
    location = raw_step.get("location", "").lower()
    input_val = raw_step.get("input", "")

    # Search DOM nodes for the best match
    best_selector = None
    best_score = 0

    for node in dom_nodes:
        if not isinstance(node, dict):
            continue

        tag = node.get("tag", "").lower()
        text = node.get("text", node.get("innerText", "")).strip()
        aria = node.get("ariaLabel", node.get("aria-label", "")).strip()
        test_id = node.get("testId", node.get("data-testid", "")).strip()
        node_id = node.get("id", "").strip()
        role = node.get("role", "").strip()
        node_type = node.get("type", "").strip()

        # Score how well this node matches the step
        score = 0

        # Text match with step action or input
        if text and action:
            text_lower = text.lower()
            if any(word in text_lower for word in action.split() if len(word) > 2):
                score += 2
        if text and input_val and input_val.lower() in text.lower():
            score += 3

        # Interactive elements get a bonus
        if tag in ("button", "a", "input", "select", "textarea"):
            score += 1
        if role in ("button", "link", "textbox", "combobox", "menuitem", "tab"):
            score += 1

        if score <= 0:
            continue

        # Build the best selector based on available attributes
        selector = None
        selector_score = score

        if aria:
            selector = f"[aria-label='{aria}']"
            selector_score += 5
        elif test_id:
            selector = f"[data-testid='{test_id}']"
            selector_score += 4
        elif node_id:
            selector = f"#{node_id}"
            selector_score += 4
        elif text and len(text) < 50:
            # Use text content with tag
            if tag:
                selector = f"{tag}:has-text('{text[:40]}')"
            else:
                selector = f"*:has-text('{text[:40]}')"
            selector_score += 2
        elif role and tag:
            selector = f"{tag}[role='{role}']"
            selector_score += 1

        if selector and selector_score > best_score:
            best_selector = selector
            best_score = selector_score

    return best_selector


# ---------------------------------------------------------------------------
# Smart sampling for focus sessions
# ---------------------------------------------------------------------------

def _smart_sample_focus(frames: list[dict], max_frames: int) -> list[dict]:
    """Priority-based sampling that keeps the most informative frames.

    Scoring per frame:
    - First / last frame: +100  (always kept)
    - Has clipboard_context:    +10
    - diff.diff_type == "action": +5
    - diff.diff_type == "app_switch": +3
    - diff.diff_type == "no_change": -2
    - App name contains "agenthandover" or "agenthandover": -5
    - annotation.task_context.what_doing contains "agenthandover": -3

    Top N by score are selected, then re-sorted by their original
    temporal order so the SOP sees a coherent sequence.
    """
    if len(frames) <= max_frames:
        return frames

    scored: list[tuple[float, int, dict]] = []

    for i, frame in enumerate(frames):
        score: float = 0.0

        # First / last frame bonus
        if i == 0 or i == len(frames) - 1:
            score += 100.0

        # Clipboard context
        if frame.get("clipboard_context"):
            score += 10.0

        # Diff type scoring
        diff = frame.get("diff")
        if isinstance(diff, dict):
            diff_type = diff.get("diff_type", "")
            if diff_type == "action":
                score += 5.0
            elif diff_type == "app_switch":
                score += 3.0
            elif diff_type == "no_change":
                score -= 2.0

        # Penalise frames from AgentHandover's own UI
        annotation = frame.get("annotation", {})
        if isinstance(annotation, dict):
            app_name = str(annotation.get("app", "")).lower()
            if "agenthandover" in app_name or "agenthandover" in app_name:
                score -= 5.0

            task_ctx = annotation.get("task_context", {})
            if isinstance(task_ctx, dict):
                what_doing = str(task_ctx.get("what_doing", "")).lower()
                if "agenthandover" in what_doing:
                    score -= 3.0

        scored.append((score, i, frame))

    # Sort by score descending; on tie, prefer earlier frame (lower index)
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Take top N
    top = scored[:max_frames]

    # Re-sort by original temporal order
    top.sort(key=lambda x: x[1])

    logger.info(
        "Smart-sampled %d of %d focus frames (score range: %.1f to %.1f)",
        max_frames, len(frames), scored[-1][0], scored[0][0],
    )

    return [frame for _, _, frame in top]


# ---------------------------------------------------------------------------
# Ollama client (reused from scene_annotator)
# ---------------------------------------------------------------------------

def _call_ollama(
    *,
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    num_predict: int = 12000,
    system: str = "",
    timeout: float = 1800.0,
    think: bool | str = True,
    format_json: bool = False,
    extra_options: dict | None = None,
) -> tuple[str, float]:
    """Call Ollama's /api/generate for text-only SOP generation.

    Returns (response_text, inference_time_seconds).

    Args:
        think: False, True, "low", "medium", or "high". For Gemma 4 models,
            "high" enables deep reasoning which produces better SOPs.
        extra_options: Additional Ollama options merged into the options dict.
            Use model_profiles.get_profile().sop_options() for optimal settings.
    """
    import urllib.request
    import urllib.error

    url = f"{host}/api/generate"

    options: dict = {
        "num_predict": num_predict,
        "num_ctx": 24576,
        "temperature": 0.3,
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
    # format: "json" forces Ollama to use GBNF grammar for valid JSON.
    # Only safe when think=False (incompatible with thinking mode).
    if format_json and not think:
        payload["format"] = "json"

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
        behavioral_context: str = "",
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

        # Sample frames if timeline is too large for the VLM prompt.
        # Use priority-based smart sampling to keep the most informative frames.
        max_frames = self.config.max_timeline_frames
        if len(meaningful) > max_frames:
            meaningful = _smart_sample_focus(meaningful, max_frames)

        # Build prompt
        prompt = _build_focus_prompt(title, meaningful)

        # Inject behavioral context if available (pre-analysis of user intent)
        if behavioral_context:
            prompt += f"\n\nBEHAVIORAL CONTEXT (use this to understand the user's intent and filter noise like ads, notifications, or unrelated content):\n{behavioral_context}\n"

        # Call VLM with model-specific profile
        from agenthandover_worker.model_profiles import get_profile
        profile = get_profile(self.config.model)
        try:
            raw_response, elapsed = _call_ollama(
                model=self.config.model,
                prompt=prompt,
                host=self.config.ollama_host,
                num_predict=profile.sop_num_predict,
                system=profile.sop_system or SOP_SYSTEM_PROMPT,
                timeout=self.config.timeout,
                think=profile.sop_think,
                extra_options={
                    k: v for k, v in profile.sop_options().items()
                    if k not in ("num_predict",)
                },
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

        # Parse response — up to 3 attempts with increasingly strict prompting
        vlm_sop = _parse_sop_response(raw_response)
        if vlm_sop is None:
            logger.warning(
                "SOP generation: invalid JSON on attempt 1 (len=%d), retrying with JSON-only suffix",
                len(raw_response),
            )
            logger.debug("Raw response (attempt 1): %.500s", raw_response[:500])
            try:
                raw_response2, elapsed2 = _call_ollama(
                    model=self.config.model,
                    prompt=prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON. No commentary, no markdown.",
                    host=self.config.ollama_host,
                    num_predict=self.config.num_predict,
                    system=SOP_SYSTEM_PROMPT,
                    timeout=self.config.timeout,
                    think=False,
                    format_json=True,  # Force JSON grammar on retry
                )
                elapsed += elapsed2
                vlm_sop = _parse_sop_response(raw_response2)
            except Exception:
                pass

        if vlm_sop is None:
            # Third attempt: shorter prompt, no thinking, explicit JSON start
            logger.warning(
                "SOP generation: invalid JSON on attempt 2, trying minimal prompt"
            )
            try:
                # Simplified prompt that's less likely to confuse smaller models
                minimal_prompt = (
                    f'Generate a JSON SOP for the task "{title}" based on these observations:\n\n'
                )
                for i, frame in enumerate(meaningful[:10]):
                    ann = frame.get("annotation", {})
                    what = ann.get("task_context", {}).get("what_doing", "")
                    app = ann.get("app", "")
                    if what:
                        minimal_prompt += f"  {i+1}. [{app}] {what}\n"
                minimal_prompt += (
                    '\nRespond with ONLY this JSON structure: '
                    '{"title": "...", "steps": [{"action": "...", "app": "...", "verify": "..."}]}'
                )
                raw_response3, elapsed3 = _call_ollama(
                    model=self.config.model,
                    prompt=minimal_prompt,
                    host=self.config.ollama_host,
                    num_predict=4000,
                    system="You output ONLY valid JSON. No explanation.",
                    timeout=self.config.timeout,
                    think=False,
                    format_json=True,
                )
                elapsed += elapsed3
                vlm_sop = _parse_sop_response(raw_response3)
            except Exception:
                pass

            if vlm_sop is None:
                logger.error("SOP generation: all 3 attempts failed for '%s'", title)
                return GeneratedSOP(
                    sop={},
                    inference_time_seconds=elapsed,
                    success=False,
                    error="Failed to parse VLM response as valid SOP JSON after 3 attempts",
                )

        # Convert to internal template format
        template = _vlm_sop_to_template(
            vlm_sop,
            mode="focus",
            title_override=title,
            timeline=meaningful,
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
            # Retry with repair instruction
            logger.warning("SOP JSON parse failed, retrying with repair prompt")
            try:
                repair_prefix = (
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY a valid JSON object, "
                    "no markdown fences, no explanatory text.\n\n"
                )
                raw_response2, elapsed2 = _call_ollama(
                    model=self.config.model,
                    prompt=repair_prefix + prompt,
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

    def generate_from_passive_enriched(
        self,
        demonstrations: list[list[dict]],
        task_label: str | None = None,
        canonical_steps: list[dict] | None = None,
        parameters: list | None = None,
        branches: list[dict] | None = None,
        evidence_context: dict | None = None,
    ) -> GeneratedSOP:
        """Generate an enriched SOP using pre-computed variant analysis.

        Uses the ENRICHED_PASSIVE_PROMPT which provides the VLM with
        pre-analyzed canonical steps, extracted parameters, and variant
        branches.  The VLM's job shifts from "figure out everything from
        raw frames" to "explain the strategy and fill in decision logic."

        Args:
            demonstrations: List of timelines (each a list of frame dicts).
            task_label: Optional cluster label for the task.
            canonical_steps: Pre-computed canonical steps from variant_detector.
            parameters: Pre-computed parameters from variant_detector.
            branches: Pre-computed branch points from variant_detector.

        Returns:
            GeneratedSOP with enriched template dict.
        """
        if len(demonstrations) < 2:
            return GeneratedSOP(
                sop={}, success=False,
                error="Need at least 2 demonstrations for passive SOP",
            )

        # Build the enriched prompt
        prompt = _build_enriched_passive_prompt(
            demonstrations,
            canonical_steps=canonical_steps or [],
            parameters=parameters or [],
            branches=branches or [],
            evidence_context=evidence_context,
        )

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
                sop={}, success=False,
                error=f"VLM call failed: {exc}",
            )

        # Parse response
        vlm_sop = _parse_sop_response(raw_response)
        if vlm_sop is None:
            logger.warning("Enriched SOP JSON parse failed, retrying")
            try:
                repair_prefix = (
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY a valid JSON object.\n\n"
                )
                raw_response2, elapsed2 = _call_ollama(
                    model=self.config.model,
                    prompt=repair_prefix + prompt,
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
                # Fall back to standard passive generation
                logger.info("Enriched generation failed, falling back to standard")
                return self.generate_from_passive(demonstrations, task_label)

        title = task_label or vlm_sop.get("title", "Untitled Workflow")

        template = _vlm_sop_to_template(
            vlm_sop, mode="passive", title_override=title,
        )
        template["episode_count"] = len(demonstrations)
        template["abs_support"] = len(demonstrations)

        # Carry over pre-computed data so it flows to procedure_schema
        if canonical_steps:
            template["_canonical_steps"] = canonical_steps
        if branches:
            template["branches"] = branches

        # Collect all annotations for confidence scoring
        all_annotations: list[dict] = []
        for demo in demonstrations:
            for frame in demo:
                ann = frame.get("annotation")
                if isinstance(ann, dict):
                    all_annotations.append(ann)

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
            "Generated enriched passive SOP '%s': %d steps from %d demos, "
            "confidence=%.2f, %.1fs",
            title, len(template["steps"]), len(demonstrations),
            template["confidence_avg"], elapsed,
        )

        return GeneratedSOP(
            sop=template, inference_time_seconds=elapsed,
        )

"""Versioned JSON schema for SOP export.

Provides a structured JSON representation of SOPs that matches
the YAML frontmatter structure used in Markdown SOPs. This enables
machine-readable SOP consumption by any agent framework.

The JSON output mirrors every field from the YAML frontmatter
(see ``sop_format.SOPFormatter._build_frontmatter``) so that
round-tripping between formats is lossless.
"""

from __future__ import annotations

from datetime import datetime, timezone

SOP_SCHEMA_VERSION = "3.0.0"

# Accepted schema versions for backward compatibility
_ACCEPTED_VERSIONS = frozenset(("1.0.0", "1.1.0", "2.0.0", "3.0.0"))

_GENERATOR = "openmimic"
_GENERATOR_VERSION = "0.1.0"

# Required top-level fields for validation
_REQUIRED_FIELDS = ("schema_version", "slug", "title", "steps")

# Valid confidence summary labels (mirrors SOPFormatter._confidence_label)
_CONFIDENCE_LABELS = frozenset(("high", "medium", "low"))


def _confidence_label(score: float) -> str:
    """Map average confidence score to a summary label.

    Thresholds match ``SOPFormatter._confidence_label``:
    - >= 0.85: "high"
    - >= 0.60: "medium"
    - < 0.60: "low"
    """
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def sop_to_json(sop_template: dict) -> dict:
    """Convert a SOP template dict to a versioned JSON export format.

    The output mirrors the YAML frontmatter produced by
    ``SOPFormatter._build_frontmatter`` so that both export paths
    carry identical semantic content.

    Args:
        sop_template: Internal SOP template dict with keys like
            slug, title, steps, confidence_avg, apps_involved,
            preconditions, postconditions, exceptions_seen, tags, etc.

    Returns:
        Versioned JSON dict ready for serialization.
    """
    steps = sop_template.get("steps", [])
    variables = sop_template.get("variables", [])
    confidence_avg = sop_template.get("confidence_avg", 0.0)

    json_steps = []
    for i, step in enumerate(steps):
        json_step: dict = {
            "index": i,
            "action": step.get("step", step.get("action", "")),
            "target": step.get("target", ""),
            "selector": step.get("selector"),
            "parameters": step.get("parameters", {}),
            "confidence": step.get("confidence", 0.0),
        }
        if step.get("pre_state"):
            json_step["pre_state"] = step["pre_state"]
        json_steps.append(json_step)

    json_variables = []
    for var in variables:
        entry: dict = {
            "name": var.get("name", ""),
            "type": var.get("type", "string"),
        }
        if "description" in var:
            entry["description"] = var["description"]
        if "example" in var:
            entry["example"] = var["example"]
        if "min" in var:
            entry["min"] = var["min"]
        if "max" in var:
            entry["max"] = var["max"]
        if "choices" in var:
            entry["choices"] = var["choices"]
        json_variables.append(entry)

    result: dict = {
        "schema_version": SOP_SCHEMA_VERSION,
        "slug": sop_template.get("slug", "unknown"),
        "title": sop_template.get("title", "Untitled"),
        "description": sop_template.get("description", ""),
        "confidence_avg": round(confidence_avg, 4),
        "confidence_summary": _confidence_label(confidence_avg),
        "episode_count": sop_template.get("episode_count", 0),
        "evidence_window": sop_template.get("evidence_window", "last_30_days"),
        "apps_involved": sop_template.get("apps_involved", []),
        "steps": json_steps,
        "variables": json_variables,
        "preconditions": sop_template.get("preconditions", []),
        "postconditions": sop_template.get("postconditions", []),
        "exceptions_seen": sop_template.get("exceptions_seen", []),
        "tags": sop_template.get("tags", []),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "generator": _GENERATOR,
            "generator_version": _GENERATOR_VERSION,
        },
    }

    # Add LLM-enhanced fields when present (schema 1.1.0+)
    task_description = sop_template.get("task_description")
    if task_description:
        result["task_description"] = task_description

    execution_overview = sop_template.get("execution_overview")
    if isinstance(execution_overview, dict) and execution_overview:
        result["execution_overview"] = execution_overview

    # v2 fields (schema 2.0.0): source mode & confidence breakdown
    source = sop_template.get("source")
    if source:
        result["source"] = source

    confidence_breakdown = sop_template.get("confidence_breakdown")
    if isinstance(confidence_breakdown, dict) and confidence_breakdown:
        result["confidence_breakdown"] = confidence_breakdown

    return result


def validate_sop_json(data: dict) -> list[str]:
    """Validate a SOP JSON dict against the schema.

    Checks required fields, type constraints on sub-structures,
    and schema version compatibility.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    # Required top-level fields
    for field in _REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Schema version check — accept all versions in _ACCEPTED_VERSIONS
    if "schema_version" in data and data["schema_version"] not in _ACCEPTED_VERSIONS:
        errors.append(
            f"Unsupported schema version: {data['schema_version']} "
            f"(expected one of {sorted(_ACCEPTED_VERSIONS)})"
        )

    # Steps must be a list of dicts with required sub-fields
    if "steps" in data:
        if not isinstance(data["steps"], list):
            errors.append("Field 'steps' must be a list")
        else:
            for i, step in enumerate(data["steps"]):
                if not isinstance(step, dict):
                    errors.append(f"steps[{i}] must be a dict")
                    continue
                if "action" not in step:
                    errors.append(f"steps[{i}] missing required field: action")

    # Variables must be a list of dicts with name and type
    if "variables" in data:
        if not isinstance(data["variables"], list):
            errors.append("Field 'variables' must be a list")
        else:
            for i, var in enumerate(data["variables"]):
                if not isinstance(var, dict):
                    errors.append(f"variables[{i}] must be a dict")
                    continue
                if "name" not in var:
                    errors.append(f"variables[{i}] missing required field: name")
                if "type" not in var:
                    errors.append(f"variables[{i}] missing required field: type")

    # List fields must be lists when present
    for list_field in ("apps_involved", "preconditions", "postconditions",
                       "exceptions_seen", "tags"):
        if list_field in data and not isinstance(data[list_field], list):
            errors.append(f"Field '{list_field}' must be a list")

    # Confidence summary must be a known label when present
    if "confidence_summary" in data:
        if data["confidence_summary"] not in _CONFIDENCE_LABELS:
            errors.append(
                f"Invalid confidence_summary: {data['confidence_summary']!r} "
                f"(expected one of {sorted(_CONFIDENCE_LABELS)})"
            )

    # LLM-enhanced optional fields (schema 1.1.0)
    if "task_description" in data:
        if not isinstance(data["task_description"], str):
            errors.append("Field 'task_description' must be a string")

    if "execution_overview" in data:
        if not isinstance(data["execution_overview"], dict):
            errors.append("Field 'execution_overview' must be a dict")
        else:
            for key, val in data["execution_overview"].items():
                if not isinstance(val, str):
                    errors.append(
                        f"execution_overview['{key}'] must be a string"
                    )

    # v2 optional fields (schema 2.0.0)
    if "source" in data:
        if not isinstance(data["source"], str):
            errors.append("Field 'source' must be a string")

    if "confidence_breakdown" in data:
        if not isinstance(data["confidence_breakdown"], dict):
            errors.append("Field 'confidence_breakdown' must be a dict")

    return errors

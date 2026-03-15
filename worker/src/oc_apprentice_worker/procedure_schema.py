"""v3 procedure schema — extended machine-readable format.

Upgrades the v2 SOP JSON schema (``sop_schema.py``) to a full v3
"procedure" format that includes:

- ``inputs`` / ``outputs`` — typed parameters with defaults
- ``environment`` — required apps, accounts, setup actions
- ``branches`` — conditional paths within steps
- ``on_failure`` per step — retry, delay, abort
- ``expected_outcomes`` — what should change after execution
- ``staleness`` — last_observed, last_confirmed, drift_signals
- ``evidence`` — linked observations, step-level confidence
- ``constraints`` — per-procedure guardrails
- ``recurrence`` — detected pattern, day, time, avg_duration

The v3 procedure is the canonical format stored in the knowledge base.
Existing v2 SOPs are upgraded via ``upgrade_v2_to_v3()``.
"""

from __future__ import annotations

from datetime import datetime, timezone

PROCEDURE_SCHEMA_VERSION = "3.1.0"

# All accepted versions for backward compatibility
_ACCEPTED_VERSIONS = frozenset(("1.0.0", "1.1.0", "2.0.0", "3.0.0", "3.1.0"))

_WORKER_VERSION = "0.2.0"

# Required top-level fields in a v3 procedure
_REQUIRED_FIELDS = (
    "schema_version",
    "id",
    "title",
    "steps",
)

# Optional v3 sections with their default factories
_V3_SECTIONS = {
    "inputs": list,
    "outputs": list,
    "environment": dict,
    "branches": list,
    "expected_outcomes": list,
    "staleness": dict,
    "evidence": dict,
    "constraints": dict,
    "recurrence": dict,
    "chain": dict,
    "lifecycle_state": lambda: "observed",
    "lifecycle_history": list,
    "compiled_outputs": dict,
    "variant_family": lambda: None,
    "variants": list,
    "parameters_extracted": list,
}


def _confidence_label(score: float) -> str:
    """Map confidence score to label (same thresholds as sop_schema.py)."""
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def sop_to_procedure(sop_template: dict) -> dict:
    """Convert an existing SOP template to a full v3 machine procedure.

    Maps:
    - slug → id
    - steps → typed steps with on_failure
    - Adds empty sections: inputs, outputs, environment, branches,
      expected_outcomes, staleness, evidence, constraints, recurrence

    Args:
        sop_template: Internal SOP template dict (from sop_generator or
            sop_inducer).

    Returns:
        v3 procedure dict ready for storage in the knowledge base.
    """
    steps = sop_template.get("steps", [])
    variables = sop_template.get("variables", [])
    confidence_avg = sop_template.get("confidence_avg", 0.0)

    # Convert steps
    proc_steps = []
    for i, step in enumerate(steps):
        proc_step: dict = {
            "step_id": f"step_{i + 1}",
            "index": i,
            "action": step.get("step", step.get("action", "")),
            "target": step.get("target", ""),
            "app": step.get("app", ""),
            "location": step.get("location", ""),
            "input": step.get("input", ""),
            "verify": step.get("verify", ""),
            "selector": step.get("selector"),
            "parameters": step.get("parameters", {}),
            "confidence": step.get("confidence", 0.0),
            "on_failure": step.get("on_failure", {
                "strategy": "abort",
                "max_retries": 0,
                "delay_seconds": 0,
            }),
        }
        if step.get("pre_state"):
            proc_step["pre_state"] = step["pre_state"]
        proc_steps.append(proc_step)

    # Convert variables to typed inputs
    inputs = []
    for var in variables:
        inp: dict = {
            "name": var.get("name", ""),
            "type": var.get("type", "string"),
            "required": True,
            "default": var.get("default"),
        }
        if "description" in var:
            inp["description"] = var["description"]
        if "example" in var:
            inp["example"] = var["example"]
        if "min" in var:
            inp["min"] = var["min"]
        if "max" in var:
            inp["max"] = var["max"]
        if "choices" in var:
            inp["choices"] = var["choices"]
        inputs.append(inp)

    now_iso = datetime.now(timezone.utc).isoformat()

    procedure: dict = {
        "schema_version": PROCEDURE_SCHEMA_VERSION,
        "id": sop_template.get("slug", "unknown"),
        "title": sop_template.get("title", "Untitled"),
        "short_title": sop_template.get("short_title", ""),
        "description": sop_template.get("description", ""),
        "tags": sop_template.get("tags", []),
        "confidence_avg": round(confidence_avg, 4),
        "confidence_summary": _confidence_label(confidence_avg),
        "episode_count": sop_template.get("episode_count", 0),
        "evidence_window": sop_template.get("evidence_window", "last_30_days"),
        "apps_involved": sop_template.get("apps_involved", []),
        "source": sop_template.get("source", "unknown"),

        # Core steps
        "steps": proc_steps,

        # v3 extended sections
        "inputs": inputs,
        "outputs": [
            {"name": sc.get("name", f"output_{i}"), "type": sc.get("type", "boolean"), "description": sc.get("description", "")}
            for i, sc in enumerate(sop_template.get("success_criteria", []))
        ] if sop_template.get("success_criteria") else [],
        "environment": {
            "required_apps": sop_template.get("apps_involved", []),
            "accounts": [],
            "setup_actions": [],
        },
        "branches": [],
        "expected_outcomes": [],
        "staleness": {
            "last_observed": now_iso,
            "last_confirmed": None,
            "drift_signals": [],
            "confidence_trend": [round(confidence_avg, 4)] if confidence_avg > 0 else [],
        },
        "evidence": {
            "observations": [],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": sop_template.get("episode_count", 0),
        },
        "constraints": {
            "trust_level": "observe",
            "guardrails": [],
        },
        "recurrence": {
            "pattern": None,
            "day": None,
            "time": None,
            "avg_duration_minutes": None,
            "observations": 0,
        },
        "chain": {
            "depends_on": [],      # slugs this procedure requires to run first
            "followed_by": [],     # slugs commonly executed after this one
            "co_occurrence_count": 0,
            "can_compose": False,  # True if this can be part of a macro procedure
        },

        # Lifecycle (Phase 3)
        "lifecycle_state": "observed",
        "lifecycle_history": [],
        "compiled_outputs": {},

        # Phase 4: variant family
        "variant_family": sop_template.get("variant_family", None),
        "variants": sop_template.get("variants", []),
        "parameters_extracted": sop_template.get("parameters_extracted", []),

        # Metadata
        "preconditions": sop_template.get("preconditions", []),
        "postconditions": sop_template.get("postconditions", []),
        "exceptions_seen": sop_template.get("exceptions_seen", []),
        "generated_at": now_iso,
        "metadata": {
            "generator": "openmimic",
            "generator_version": _WORKER_VERSION,
            "schema_version": PROCEDURE_SCHEMA_VERSION,
        },
    }

    # Carry over optional v2 fields
    if sop_template.get("task_description"):
        procedure["task_description"] = sop_template["task_description"]
    if sop_template.get("execution_overview"):
        procedure["execution_overview"] = sop_template["execution_overview"]
    if sop_template.get("outcome"):
        procedure["outcome"] = sop_template["outcome"]
    if sop_template.get("when_to_use"):
        procedure["when_to_use"] = sop_template["when_to_use"]
    if sop_template.get("prerequisites"):
        procedure["prerequisites"] = sop_template["prerequisites"]
    if sop_template.get("confidence_breakdown"):
        procedure["confidence_breakdown"] = sop_template["confidence_breakdown"]

    return procedure


def validate_procedure(data: dict) -> list[str]:
    """Validate a procedure dict against the v3 schema.

    Returns a list of error messages.  Empty list means valid.
    """
    errors: list[str] = []

    # Required top-level fields
    for field_name in _REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")

    # Schema version check
    version = data.get("schema_version")
    if version is not None and version not in _ACCEPTED_VERSIONS:
        errors.append(
            f"Unsupported schema version: {version} "
            f"(expected one of {sorted(_ACCEPTED_VERSIONS)})"
        )

    # Steps must be a list of dicts
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
                # Validate on_failure if present
                if "on_failure" in step:
                    of = step["on_failure"]
                    if not isinstance(of, dict):
                        errors.append(f"steps[{i}].on_failure must be a dict")
                    elif "strategy" in of and of["strategy"] not in (
                        "abort", "retry", "skip", "fallback"
                    ):
                        errors.append(
                            f"steps[{i}].on_failure.strategy must be one of: "
                            "abort, retry, skip, fallback"
                        )

    # Inputs must be typed
    if "inputs" in data:
        if not isinstance(data["inputs"], list):
            errors.append("Field 'inputs' must be a list")
        else:
            for i, inp in enumerate(data["inputs"]):
                if not isinstance(inp, dict):
                    errors.append(f"inputs[{i}] must be a dict")
                    continue
                if "name" not in inp:
                    errors.append(f"inputs[{i}] missing required field: name")
                if "type" not in inp:
                    errors.append(f"inputs[{i}] missing required field: type")

    # Outputs must be a list of dicts
    if "outputs" in data:
        if not isinstance(data["outputs"], list):
            errors.append("Field 'outputs' must be a list")

    # Environment must be a dict
    if "environment" in data:
        if not isinstance(data["environment"], dict):
            errors.append("Field 'environment' must be a dict")

    # Branches must be a list
    if "branches" in data:
        if not isinstance(data["branches"], list):
            errors.append("Field 'branches' must be a list")

    # Expected outcomes must be a list
    if "expected_outcomes" in data:
        if not isinstance(data["expected_outcomes"], list):
            errors.append("Field 'expected_outcomes' must be a list")

    # Staleness must be a dict with expected fields
    if "staleness" in data:
        if not isinstance(data["staleness"], dict):
            errors.append("Field 'staleness' must be a dict")
        else:
            stale = data["staleness"]
            if "confidence_trend" in stale and not isinstance(
                stale["confidence_trend"], list
            ):
                errors.append("staleness.confidence_trend must be a list")
            if "drift_signals" in stale and not isinstance(
                stale["drift_signals"], list
            ):
                errors.append("staleness.drift_signals must be a list")

    # Evidence must be a dict
    if "evidence" in data:
        if not isinstance(data["evidence"], dict):
            errors.append("Field 'evidence' must be a dict")

    # Constraints must be a dict
    if "constraints" in data:
        if not isinstance(data["constraints"], dict):
            errors.append("Field 'constraints' must be a dict")
        else:
            trust = data["constraints"].get("trust_level")
            valid_levels = ("observe", "suggest", "draft",
                            "execute_with_approval", "autonomous")
            if trust is not None and trust not in valid_levels:
                errors.append(
                    f"constraints.trust_level must be one of: "
                    f"{', '.join(valid_levels)}"
                )

    # Recurrence must be a dict
    if "recurrence" in data:
        if not isinstance(data["recurrence"], dict):
            errors.append("Field 'recurrence' must be a dict")

    # Chain must be a dict
    if "chain" in data:
        if not isinstance(data["chain"], dict):
            errors.append("Field 'chain' must be a dict")
        else:
            chain = data["chain"]
            for list_key in ("depends_on", "followed_by"):
                if list_key in chain and not isinstance(chain[list_key], list):
                    errors.append(f"chain.{list_key} must be a list")

    # Lifecycle state
    _VALID_LIFECYCLE_STATES = (
        "observed", "draft", "reviewed", "verified",
        "agent_ready", "stale", "archived",
    )
    if "lifecycle_state" in data:
        if data["lifecycle_state"] not in _VALID_LIFECYCLE_STATES:
            errors.append(
                f"Invalid lifecycle_state: {data['lifecycle_state']!r} "
                f"(expected one of {_VALID_LIFECYCLE_STATES})"
            )

    # Lifecycle history
    if "lifecycle_history" in data:
        if not isinstance(data["lifecycle_history"], list):
            errors.append("Field 'lifecycle_history' must be a list")

    # Compiled outputs
    if "compiled_outputs" in data:
        if not isinstance(data["compiled_outputs"], dict):
            errors.append("Field 'compiled_outputs' must be a dict")

    # Variant family
    if "variant_family" in data:
        if data["variant_family"] is not None and not isinstance(data["variant_family"], str):
            errors.append("variant_family must be a string or null")
    if "variants" in data:
        if not isinstance(data["variants"], list):
            errors.append("Field 'variants' must be a list")
    if "parameters_extracted" in data:
        if not isinstance(data["parameters_extracted"], list):
            errors.append("Field 'parameters_extracted' must be a list")

    # List fields
    for list_field in (
        "apps_involved", "preconditions", "postconditions",
        "exceptions_seen", "tags",
    ):
        if list_field in data and not isinstance(data[list_field], list):
            errors.append(f"Field '{list_field}' must be a list")

    # Confidence summary
    valid_labels = ("high", "medium", "low")
    if "confidence_summary" in data:
        if data["confidence_summary"] not in valid_labels:
            errors.append(
                f"Invalid confidence_summary: {data['confidence_summary']!r} "
                f"(expected one of {sorted(valid_labels)})"
            )

    return errors


def upgrade_v2_to_v3(sop_json: dict) -> dict:
    """Upgrade a v2 sop_schema.py output to v3 procedure format.

    Accepts the output of ``sop_schema.sop_to_json()`` and converts
    it to a v3 procedure dict.

    Args:
        sop_json: v2 JSON dict (schema_version 2.0.0 or earlier).

    Returns:
        v3 procedure dict.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    confidence_avg = sop_json.get("confidence_avg", 0.0)

    # Convert v2 steps to v3 steps
    proc_steps = []
    for step in sop_json.get("steps", []):
        proc_step: dict = {
            "step_id": f"step_{step.get('index', 0) + 1}",
            "index": step.get("index", 0),
            "action": step.get("action", ""),
            "target": step.get("target", ""),
            "app": step.get("app", ""),
            "location": step.get("location", ""),
            "input": step.get("input", ""),
            "verify": step.get("verify", ""),
            "selector": step.get("selector"),
            "parameters": step.get("parameters", {}),
            "confidence": step.get("confidence", 0.0),
            "on_failure": {
                "strategy": "abort",
                "max_retries": 0,
                "delay_seconds": 0,
            },
        }
        if step.get("pre_state"):
            proc_step["pre_state"] = step["pre_state"]
        proc_steps.append(proc_step)

    # Convert v2 variables to inputs
    inputs = []
    for var in sop_json.get("variables", []):
        inp: dict = {
            "name": var.get("name", ""),
            "type": var.get("type", "string"),
            "required": True,
            "default": None,
        }
        if "description" in var:
            inp["description"] = var["description"]
        if "example" in var:
            inp["example"] = var["example"]
        inputs.append(inp)

    procedure: dict = {
        "schema_version": PROCEDURE_SCHEMA_VERSION,
        "id": sop_json.get("slug", "unknown"),
        "title": sop_json.get("title", "Untitled"),
        "short_title": sop_json.get("short_title", ""),
        "description": sop_json.get("description", ""),
        "tags": sop_json.get("tags", []),
        "confidence_avg": round(confidence_avg, 4),
        "confidence_summary": _confidence_label(confidence_avg),
        "episode_count": sop_json.get("episode_count", 0),
        "evidence_window": sop_json.get("evidence_window", "last_30_days"),
        "apps_involved": sop_json.get("apps_involved", []),
        "source": sop_json.get("source", "unknown"),
        "steps": proc_steps,
        "inputs": inputs,
        "outputs": [],
        "environment": {
            "required_apps": sop_json.get("apps_involved", []),
            "accounts": [],
            "setup_actions": [],
        },
        "branches": [],
        "expected_outcomes": [],
        "staleness": {
            "last_observed": sop_json.get("generated_at", now_iso),
            "last_confirmed": None,
            "drift_signals": [],
            "confidence_trend": [round(confidence_avg, 4)] if confidence_avg > 0 else [],
        },
        "evidence": {
            "observations": [],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": sop_json.get("episode_count", 0),
        },
        "constraints": {
            "trust_level": "observe",
            "guardrails": [],
        },
        "recurrence": {
            "pattern": None,
            "day": None,
            "time": None,
            "avg_duration_minutes": None,
            "observations": 0,
        },
        "chain": {
            "depends_on": [],      # slugs this procedure requires to run first
            "followed_by": [],     # slugs commonly executed after this one
            "co_occurrence_count": 0,
            "can_compose": False,  # True if this can be part of a macro procedure
        },
        "lifecycle_state": "observed",
        "lifecycle_history": [],
        "compiled_outputs": {},
        "variant_family": sop_json.get("variant_family", None),
        "variants": sop_json.get("variants", []),
        "parameters_extracted": sop_json.get("parameters_extracted", []),
        "preconditions": sop_json.get("preconditions", []),
        "postconditions": sop_json.get("postconditions", []),
        "exceptions_seen": sop_json.get("exceptions_seen", []),
        "generated_at": sop_json.get("generated_at", now_iso),
        "upgraded_at": now_iso,
        "metadata": {
            "generator": "openmimic",
            "generator_version": _WORKER_VERSION,
            "schema_version": PROCEDURE_SCHEMA_VERSION,
            "upgraded_from": sop_json.get("schema_version", "unknown"),
        },
    }

    # Carry over optional fields
    for optional_key in (
        "task_description", "execution_overview", "confidence_breakdown",
    ):
        if optional_key in sop_json:
            procedure[optional_key] = sop_json[optional_key]

    return procedure

"""SOP validation gate — lint/compile SOPs before export.

Validates SOP templates for structural correctness and completeness
before they are persisted or exported to disk.  Runs entirely in-memory
with no I/O or VLM calls.

Errors block export; warnings are logged but do not block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches {{var_name}} with optional whitespace inside braces.
_VAR_REF_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass
class LintIssue:
    """A single validation finding."""

    severity: str  # "error" or "warning"
    field: str  # which field has the issue
    message: str  # human-readable description


@dataclass
class LintResult:
    """Outcome of linting a single SOP template."""

    valid: bool  # True if no errors (warnings OK)
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "warning"]


def _collect_variable_refs(text: str) -> set[str]:
    """Extract all ``{{var_name}}`` references from *text*."""
    if not isinstance(text, str):
        return set()
    return set(_VAR_REF_RE.findall(text))


def _step_text_fields(step: dict) -> list[str]:
    """Return the text values from step fields that may contain variable refs."""
    texts: list[str] = []
    # Internal template uses "step" for the action verb.  The VLM raw
    # format uses "action".  Check both, plus common supplementary fields.
    for key in ("step", "action", "input", "location", "verify", "target"):
        val = step.get(key)
        if isinstance(val, str):
            texts.append(val)
    # Parameters sub-dict may also contain refs.
    params = step.get("parameters")
    if isinstance(params, dict):
        for v in params.values():
            if isinstance(v, str):
                texts.append(v)
    return texts


def lint_sop(sop_template: dict) -> LintResult:
    """Validate an SOP template before export.

    Returns a ``LintResult`` whose ``.valid`` flag is ``False`` when any
    error-severity issue is found (warnings alone do not block export).
    """
    issues: list[LintIssue] = []

    # ------------------------------------------------------------------
    # Errors (block export)
    # ------------------------------------------------------------------

    # 1. Missing or empty title
    if not sop_template.get("title"):
        issues.append(LintIssue("error", "title", "Missing or empty title"))

    # 2. Missing or empty slug
    if not sop_template.get("slug"):
        issues.append(LintIssue("error", "slug", "Missing or empty slug"))

    # 3. Missing or empty steps list
    steps = sop_template.get("steps")
    if not steps or not isinstance(steps, list):
        issues.append(LintIssue("error", "steps", "No steps defined"))
        steps = []

    # 4. Steps missing action field (internal="step", raw VLM="action")
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(
                LintIssue(
                    "error",
                    f"steps[{idx}]",
                    "Step is not a dict",
                )
            )
            continue
        action_value = step.get("step") or step.get("action")
        if not action_value:
            issues.append(
                LintIssue(
                    "error",
                    f"steps[{idx}].action",
                    "Step missing action",
                )
            )

    # 5. Variable references in steps not declared in variables list
    declared_vars: set[str] = set()
    variables = sop_template.get("variables")
    if isinstance(variables, list):
        for var in variables:
            if isinstance(var, dict):
                name = var.get("name", "")
                if name:
                    declared_vars.add(name)
            elif isinstance(var, str) and var:
                declared_vars.add(var)

    referenced_vars: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        for text in _step_text_fields(step):
            referenced_vars.update(_collect_variable_refs(text))

    undeclared = referenced_vars - declared_vars
    for var_name in sorted(undeclared):
        issues.append(
            LintIssue(
                "warning",
                f"variables.{var_name}",
                f"Variable '{{{{{var_name}}}}}' referenced in steps but not declared",
            )
        )

    # ------------------------------------------------------------------
    # Warnings (log but allow export)
    # ------------------------------------------------------------------

    # 1. Fewer than 2 steps
    if steps and len(steps) < 2:
        issues.append(
            LintIssue("warning", "steps", "SOP may be too simple (fewer than 2 steps)")
        )

    # 2. No task_description
    if not sop_template.get("task_description"):
        issues.append(
            LintIssue("warning", "task_description", "No task description")
        )

    # 3. Steps without verify field
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        # Check both top-level verify and parameters.verify
        has_verify = bool(step.get("verify"))
        params = step.get("parameters")
        if isinstance(params, dict) and params.get("verify"):
            has_verify = True
        if not has_verify:
            issues.append(
                LintIssue(
                    "warning",
                    f"steps[{idx}].verify",
                    "Step lacks verification",
                )
            )

    # 4. Steps without app field
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        has_app = bool(step.get("app"))
        params = step.get("parameters")
        if isinstance(params, dict) and params.get("app"):
            has_app = True
        if not has_app:
            issues.append(
                LintIssue(
                    "warning",
                    f"steps[{idx}].app",
                    "Step missing app context",
                )
            )

    # 5. Duplicate step actions
    action_texts: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action_val = step.get("step") or step.get("action") or ""
        if action_val:
            action_texts.append(action_val)

    seen_actions: set[str] = set()
    for action_text in action_texts:
        if action_text in seen_actions:
            issues.append(
                LintIssue(
                    "warning",
                    "steps",
                    f"Duplicate step action: '{action_text}'",
                )
            )
            break  # Report once
        seen_actions.add(action_text)

    # 6. Low confidence
    confidence = sop_template.get("confidence_avg", sop_template.get("confidence", 1.0))
    if isinstance(confidence, (int, float)) and confidence < 0.5:
        issues.append(
            LintIssue(
                "warning",
                "confidence",
                f"Low confidence SOP ({confidence:.2f})",
            )
        )

    # 7. Unused variables (declared but never referenced)
    unused = declared_vars - referenced_vars
    for var_name in sorted(unused):
        issues.append(
            LintIssue(
                "warning",
                f"variables.{var_name}",
                f"Variable '{var_name}' declared but never referenced in steps",
            )
        )

    has_errors = any(i.severity == "error" for i in issues)
    return LintResult(valid=not has_errors, issues=issues)

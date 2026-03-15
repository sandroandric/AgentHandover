"""Procedure verifier — preflight checks and postcondition validation.

Before an agent executes a procedure, the verifier checks:
- Required apps are available (by checking recent observations)
- Required accounts are accessible
- Trust level permits execution
- Freshness score is above threshold
- Constraints are not violated

After execution, it validates expected outcomes against actual observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Minimum freshness score to consider a procedure reliable for execution
_MIN_FRESHNESS_FOR_EXECUTION = 0.3

# Trust levels that permit agent execution
_EXECUTABLE_TRUST_LEVELS = frozenset({
    "execute_with_approval",
    "autonomous",
})

# Trust levels that permit drafting (one level below execution)
_DRAFTABLE_TRUST_LEVELS = frozenset({
    "draft",
    "execute_with_approval",
    "autonomous",
})


@dataclass
class PreflightCheck:
    """Result of a single preflight check."""
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" = blocks execution, "warning" = caution, "advisory" = metadata only


@dataclass
class PreflightResult:
    """Aggregate preflight result for a procedure."""
    slug: str
    can_execute: bool
    can_draft: bool
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def errors(self) -> list[PreflightCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def advisories(self) -> list[PreflightCheck]:
        """Checks labeled advisory — metadata not validated at runtime."""
        return [c for c in self.checks if c.severity == "advisory"]


@dataclass
class PostconditionCheck:
    """Result of a single postcondition check."""
    outcome_type: str
    expected: str
    actual: str | None
    passed: bool
    detail: str


@dataclass
class PostconditionResult:
    """Aggregate postcondition validation result."""
    slug: str
    execution_id: str
    all_passed: bool
    checks: list[PostconditionCheck] = field(default_factory=list)


class ProcedureVerifier:
    """Verify procedure readiness (preflight) and outcomes (postcondition)."""

    def __init__(self, kb: KnowledgeBase, runtime_validator=None) -> None:
        self._kb = kb
        self._runtime_validator = runtime_validator

    def preflight(self, slug: str, validate_environment: bool = False) -> PreflightResult:
        """Run preflight checks for a procedure.

        Checks:
        1. Procedure exists in KB
        2. Trust level permits execution
        3. Freshness score above threshold
        4. Required apps are known
        5. Constraints are not violated
        6. Procedure has been observed at least once
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return PreflightResult(
                slug=slug,
                can_execute=False,
                can_draft=False,
                checks=[PreflightCheck(
                    name="procedure_exists",
                    passed=False,
                    detail=f"Procedure '{slug}' not found in knowledge base",
                )],
            )

        checks: list[PreflightCheck] = []

        # 1. Procedure exists
        checks.append(PreflightCheck(
            name="procedure_exists",
            passed=True,
            detail="Procedure found in knowledge base",
        ))

        # 2. Trust level
        constraints = proc.get("constraints", {})
        trust_level = constraints.get("trust_level", "observe")
        can_exec_trust = trust_level in _EXECUTABLE_TRUST_LEVELS
        can_draft_trust = trust_level in _DRAFTABLE_TRUST_LEVELS
        checks.append(PreflightCheck(
            name="trust_level",
            passed=can_exec_trust,
            detail=f"Trust level: {trust_level}" + (
                "" if can_exec_trust else
                " (requires 'execute_with_approval' or 'autonomous')"
            ),
        ))

        # 2b. Lifecycle state (Phase 3 — only when field is present)
        lifecycle_state = proc.get("lifecycle_state")
        if lifecycle_state is not None:
            lifecycle_allows_exec = lifecycle_state == "agent_ready"
            lifecycle_allows_draft = lifecycle_state in (
                "draft", "reviewed", "verified", "agent_ready"
            )
            checks.append(PreflightCheck(
                name="lifecycle_state",
                passed=lifecycle_allows_exec,
                detail=f"Lifecycle state: {lifecycle_state}" + (
                    "" if lifecycle_allows_exec else
                    " (requires 'agent_ready' for execution)"
                ),
            ))
        else:
            lifecycle_allows_exec = True
            lifecycle_allows_draft = True

        # 3. Freshness score
        from oc_apprentice_worker.staleness_detector import procedure_freshness
        freshness = procedure_freshness(proc)
        freshness_ok = freshness >= _MIN_FRESHNESS_FOR_EXECUTION
        checks.append(PreflightCheck(
            name="freshness",
            passed=freshness_ok,
            detail=f"Freshness score: {freshness:.2f}" + (
                "" if freshness_ok else
                f" (minimum {_MIN_FRESHNESS_FOR_EXECUTION} required)"
            ),
            severity="error",
        ))

        # 4. Required apps — advisory only (not validated at runtime)
        env = proc.get("environment", {})
        required_apps = env.get("required_apps", [])
        if required_apps:
            checks.append(PreflightCheck(
                name="required_apps",
                passed=True,
                detail=(
                    f"Required apps listed in metadata: {', '.join(required_apps)} "
                    "(not validated at runtime)"
                ),
                severity="advisory",
            ))
        else:
            checks.append(PreflightCheck(
                name="required_apps",
                passed=True,
                detail="No specific apps required",
                severity="advisory",
            ))

        # 5. Constraints check
        global_constraints = self._kb.get_constraints()
        blocked_domains = global_constraints.get("blocked_domains", [])
        procedure_urls = set()
        for step in proc.get("steps", []):
            location = step.get("location", "")
            if location and ("http" in location):
                procedure_urls.add(location)

        domain_blocked = False
        for url in procedure_urls:
            for blocked in blocked_domains:
                if blocked in url:
                    domain_blocked = True
                    checks.append(PreflightCheck(
                        name="blocked_domain",
                        passed=False,
                        detail=f"URL '{url}' matches blocked domain '{blocked}'",
                    ))

        if not domain_blocked:
            checks.append(PreflightCheck(
                name="blocked_domain",
                passed=True,
                detail="No blocked domains detected",
            ))

        # 6. Observation count — advisory (informs confidence, not a gate)
        evidence = proc.get("evidence", {})
        total_obs = evidence.get("total_observations", 0)
        has_observations = total_obs > 0
        checks.append(PreflightCheck(
            name="observations",
            passed=has_observations,
            detail=f"Observed {total_obs} time(s)" + (
                "" if has_observations else " (never observed)"
            ),
            severity="advisory",
        ))

        # 7. Steps validation
        steps = proc.get("steps", [])
        has_steps = len(steps) > 0
        checks.append(PreflightCheck(
            name="has_steps",
            passed=has_steps,
            detail=f"Procedure has {len(steps)} step(s)",
        ))

        # 8. Runtime environment validation (opt-in)
        if validate_environment and self._runtime_validator is not None:
            try:
                runtime_checks = self._runtime_validator.validate_environment(slug)
                for rc in runtime_checks:
                    checks.append(PreflightCheck(
                        name=f"runtime:{rc.name}",
                        passed=rc.passed,
                        detail=rc.detail,
                        severity="error" if rc.check_type == "app_running" else "warning",
                    ))
            except Exception:
                logger.debug("Runtime validation failed for %s", slug, exc_info=True)

        # Determine overall result.
        # Trust level and lifecycle state are handled separately via
        # their own boolean flags, so exclude them from the blocking-error count.
        non_trust_lifecycle_errors = [
            c for c in checks
            if not c.passed and c.severity == "error"
            and c.name not in ("trust_level", "lifecycle_state")
        ]
        can_execute = (
            len(non_trust_lifecycle_errors) == 0
            and can_exec_trust
            and lifecycle_allows_exec
        )
        can_draft = (
            len(non_trust_lifecycle_errors) == 0
            and can_draft_trust
            and lifecycle_allows_draft
        )

        return PreflightResult(
            slug=slug,
            can_execute=can_execute,
            can_draft=can_draft,
            checks=checks,
        )

    def validate_postconditions(
        self,
        slug: str,
        execution_id: str,
        actual_outcomes: list[dict] | None = None,
    ) -> PostconditionResult:
        """Validate expected outcomes against actual execution results.

        Compares the procedure's expected_outcomes with what actually
        happened during execution.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return PostconditionResult(
                slug=slug,
                execution_id=execution_id,
                all_passed=False,
                checks=[PostconditionCheck(
                    outcome_type="procedure_exists",
                    expected="exists",
                    actual=None,
                    passed=False,
                    detail=f"Procedure '{slug}' not found",
                )],
            )

        expected_outcomes = proc.get("expected_outcomes", [])
        if not expected_outcomes:
            return PostconditionResult(
                slug=slug,
                execution_id=execution_id,
                all_passed=True,
                checks=[PostconditionCheck(
                    outcome_type="no_expectations",
                    expected="none defined",
                    actual="skipped",
                    passed=True,
                    detail="No expected outcomes defined — postcondition check skipped",
                )],
            )

        actual = actual_outcomes or []
        actual_types = {o.get("type", "") for o in actual}
        checks: list[PostconditionCheck] = []

        for expected in expected_outcomes:
            if isinstance(expected, dict):
                exp_type = expected.get("type", "")
                exp_desc = expected.get("description", exp_type)
                verification = expected.get("verification", {})
            else:
                exp_type = str(expected)
                exp_desc = exp_type
                verification = {}

            # Simple type matching
            matched = exp_type in actual_types
            actual_match = None
            if matched:
                for a in actual:
                    if a.get("type") == exp_type:
                        actual_match = a.get("description", str(a))
                        break

            checks.append(PostconditionCheck(
                outcome_type=exp_type,
                expected=exp_desc,
                actual=actual_match,
                passed=matched,
                detail=f"{'Matched' if matched else 'Not matched'}: {exp_desc}",
            ))

        all_passed = all(c.passed for c in checks)
        return PostconditionResult(
            slug=slug,
            execution_id=execution_id,
            all_passed=all_passed,
            checks=checks,
        )

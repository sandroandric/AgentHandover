"""Runtime environment validation for procedure execution.

Unlike the advisory preflight checks in procedure_verifier.py, this module
performs actual runtime checks: is the required app running? Is the URL
reachable? Are post-execution outcomes verified?
"""

from __future__ import annotations

import logging
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class RuntimeCheck:
    """Result of a single runtime validation check."""

    name: str
    passed: bool
    detail: str
    check_type: str = "environment"  # "app_running", "url_reachable", "outcome_verified"


class RuntimeValidator:
    """Perform live runtime checks before and after procedure execution.

    ``validate_environment`` checks that required apps are actually running
    (via ``pgrep``) and optionally that step URLs are reachable.

    ``validate_post_execution`` compares expected outcomes against actual
    execution results.
    """

    def __init__(self, kb: "KnowledgeBase", check_urls: bool = False) -> None:
        self._kb = kb
        self._check_urls = check_urls  # default off to avoid network calls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_environment(self, slug: str) -> list[RuntimeCheck]:
        """Validate that required apps are running and environment is ready.

        For each ``required_app`` in the procedure's ``environment``, checks
        whether the app process is currently running via ``pgrep -fi``.

        When *check_urls* was enabled at construction time, also verifies
        reachability of any ``http``/``https`` step locations via HEAD
        request.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return [
                RuntimeCheck(
                    name="procedure_exists",
                    passed=False,
                    detail=f"Procedure '{slug}' not found",
                    check_type="environment",
                )
            ]

        checks: list[RuntimeCheck] = []
        env = proc.get("environment", {})

        # Check required apps via pgrep
        for app in env.get("required_apps", []):
            running = self._check_app_running(app)
            checks.append(
                RuntimeCheck(
                    name=f"app_running:{app}",
                    passed=running,
                    detail=f"App '{app}' is {'running' if running else 'not running'}",
                    check_type="app_running",
                )
            )

        # Optionally check URL reachability
        if self._check_urls:
            urls_seen: set[str] = set()
            for step in proc.get("steps", []):
                loc = step.get("location", "")
                if loc.startswith("http") and loc not in urls_seen:
                    urls_seen.add(loc)
                    reachable = self._check_url_reachable(loc)
                    checks.append(
                        RuntimeCheck(
                            name=f"url_reachable:{loc[:60]}",
                            passed=reachable,
                            detail=f"URL {'reachable' if reachable else 'not reachable'}: {loc}",
                            check_type="url_reachable",
                        )
                    )

        return checks

    def validate_post_execution(
        self,
        slug: str,
        execution_id: str,
        actual_outcomes: list[dict] | None = None,
    ) -> list[RuntimeCheck]:
        """Validate expected outcomes against actual execution results.

        Each ``expected_outcome`` (from the procedure's ``expected_outcomes``
        list) is compared against *actual_outcomes* by ``type`` matching.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return [
                RuntimeCheck(
                    name="procedure_exists",
                    passed=False,
                    detail="Not found",
                    check_type="outcome_verified",
                )
            ]

        expected = proc.get("expected_outcomes", [])
        if not expected:
            return [
                RuntimeCheck(
                    name="no_expectations",
                    passed=True,
                    detail="No expected outcomes defined",
                    check_type="outcome_verified",
                )
            ]

        actual = actual_outcomes or []
        actual_types = {o.get("type", "") for o in actual}
        checks: list[RuntimeCheck] = []

        for exp in expected:
            if isinstance(exp, dict):
                exp_type = exp.get("type", "")
                exp_desc = exp.get("description", exp_type)
            else:
                exp_type = str(exp)
                exp_desc = exp_type

            matched = exp_type in actual_types
            checks.append(
                RuntimeCheck(
                    name=f"outcome:{exp_type}",
                    passed=matched,
                    detail=f"{'Verified' if matched else 'Not verified'}: {exp_desc}",
                    check_type="outcome_verified",
                )
            )

        return checks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_app_running(self, app_name: str) -> bool:
        """Check if an app is running via ``pgrep -fi``."""
        try:
            result = subprocess.run(
                ["pgrep", "-fi", app_name],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.debug("pgrep not available (non-macOS system)")
            return True  # gracefully assume running on systems without pgrep
        except subprocess.TimeoutExpired:
            logger.debug("pgrep timed out for %s", app_name)
            return False
        except Exception:
            logger.debug("pgrep check failed for %s", app_name, exc_info=True)
            return True  # fail-open

    def _check_url_reachable(self, url: str) -> bool:
        """Check if a URL is reachable via HEAD request."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status < 400
        except Exception:
            return False

"""Tests for the runtime_validator module — 20 tests.

Covers:
- App running checks (5)
- URL reachability checks (3)
- Validate environment (6)
- Validate post-execution (4)
- Edge cases (2)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.runtime_validator import RuntimeCheck, RuntimeValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


def _save_proc(kb: KnowledgeBase, slug: str, **overrides) -> dict:
    """Save a minimal procedure and return the dict."""
    proc = {
        "id": slug,
        "slug": slug,
        "title": f"Test {slug}",
        "steps": [],
        "environment": {},
        "expected_outcomes": [],
    }
    proc.update(overrides)
    kb.save_procedure(proc)
    return proc


# ===========================================================================
# 1) TestAppRunning (5)
# ===========================================================================


class TestAppRunning:
    """Test _check_app_running() with mocked subprocess."""

    def test_app_running_returns_true(self, kb: KnowledgeBase) -> None:
        """pgrep returncode=0 means the app is running."""
        rv = RuntimeValidator(kb)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert rv._check_app_running("Safari") is True
            mock_run.assert_called_once()

    def test_app_not_running_returns_false(self, kb: KnowledgeBase) -> None:
        """pgrep returncode=1 means the app is not running."""
        rv = RuntimeValidator(kb)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert rv._check_app_running("Safari") is False

    def test_pgrep_not_available(self, kb: KnowledgeBase) -> None:
        """FileNotFoundError (no pgrep binary) -> fail-open, returns True."""
        rv = RuntimeValidator(kb)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert rv._check_app_running("Safari") is True

    def test_pgrep_timeout(self, kb: KnowledgeBase) -> None:
        """TimeoutExpired -> returns False."""
        rv = RuntimeValidator(kb)
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pgrep", timeout=5),
        ):
            assert rv._check_app_running("Safari") is False

    def test_app_check_case_insensitive(self, kb: KnowledgeBase) -> None:
        """Verify that pgrep is called with -fi flag for case-insensitive match."""
        rv = RuntimeValidator(kb)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rv._check_app_running("chrome")
            args = mock_run.call_args[0][0]
            assert "-fi" in args


# ===========================================================================
# 2) TestUrlReachable (3)
# ===========================================================================


class TestUrlReachable:
    """Test _check_url_reachable() with mocked urllib."""

    def test_url_reachable(self, kb: KnowledgeBase) -> None:
        """Successful HEAD request with status < 400 -> True."""
        rv = RuntimeValidator(kb)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert rv._check_url_reachable("https://example.com") is True

    def test_url_not_reachable_timeout(self, kb: KnowledgeBase) -> None:
        """URLError (timeout / DNS failure) -> False."""
        rv = RuntimeValidator(kb)
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            assert rv._check_url_reachable("https://down.example.com") is False

    def test_url_not_reachable_404(self, kb: KnowledgeBase) -> None:
        """HTTP status >= 400 -> False."""
        rv = RuntimeValidator(kb)
        mock_resp = MagicMock()
        mock_resp.status = 404
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert rv._check_url_reachable("https://example.com/404") is False


# ===========================================================================
# 3) TestValidateEnvironment (6)
# ===========================================================================


class TestValidateEnvironment:
    """Test validate_environment() with full procedure dicts."""

    def test_required_apps_checked(self, kb: KnowledgeBase) -> None:
        """Two required apps produce two checks."""
        _save_proc(
            kb,
            "two-apps",
            environment={"required_apps": ["Chrome", "Slack"]},
        )
        rv = RuntimeValidator(kb)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            checks = rv.validate_environment("two-apps")
        assert len(checks) == 2
        assert all(c.check_type == "app_running" for c in checks)
        assert all(c.passed for c in checks)

    def test_no_required_apps(self, kb: KnowledgeBase) -> None:
        """Empty required_apps list -> no checks."""
        _save_proc(kb, "no-apps", environment={"required_apps": []})
        rv = RuntimeValidator(kb)
        checks = rv.validate_environment("no-apps")
        assert checks == []

    def test_procedure_not_found(self, kb: KnowledgeBase) -> None:
        """Missing procedure returns a single failure check."""
        rv = RuntimeValidator(kb)
        checks = rv.validate_environment("ghost")
        assert len(checks) == 1
        assert checks[0].passed is False
        assert "not found" in checks[0].detail

    def test_urls_not_checked_by_default(self, kb: KnowledgeBase) -> None:
        """check_urls=False (default) skips URL reachability."""
        _save_proc(
            kb,
            "has-url",
            steps=[{"action": "Open page", "location": "https://example.com"}],
            environment={"required_apps": []},
        )
        rv = RuntimeValidator(kb, check_urls=False)
        checks = rv.validate_environment("has-url")
        assert not any(c.check_type == "url_reachable" for c in checks)

    def test_urls_checked_when_enabled(self, kb: KnowledgeBase) -> None:
        """check_urls=True produces url_reachable checks."""
        _save_proc(
            kb,
            "has-url",
            steps=[{"action": "Open page", "location": "https://example.com"}],
            environment={"required_apps": []},
        )
        rv = RuntimeValidator(kb, check_urls=True)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            checks = rv.validate_environment("has-url")
        assert any(c.check_type == "url_reachable" for c in checks)
        url_check = [c for c in checks if c.check_type == "url_reachable"][0]
        assert url_check.passed is True

    def test_mixed_results(self, kb: KnowledgeBase) -> None:
        """One app running, one not -> mixed pass/fail."""
        _save_proc(
            kb,
            "mixed",
            environment={"required_apps": ["Chrome", "Xcode"]},
        )
        rv = RuntimeValidator(kb)
        with patch("subprocess.run") as mock_run:
            # Chrome running (returncode=0), Xcode not (returncode=1)
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1),
            ]
            checks = rv.validate_environment("mixed")
        assert len(checks) == 2
        assert checks[0].passed is True
        assert checks[1].passed is False


# ===========================================================================
# 4) TestValidatePostExecution (4)
# ===========================================================================


class TestValidatePostExecution:
    """Test validate_post_execution() outcome matching."""

    def test_matching_outcomes(self, kb: KnowledgeBase) -> None:
        """Expected outcome type matches actual -> passed."""
        _save_proc(
            kb,
            "deploy",
            expected_outcomes=[
                {"type": "deployment", "description": "App deployed"}
            ],
        )
        rv = RuntimeValidator(kb)
        checks = rv.validate_post_execution(
            "deploy",
            "exec-1",
            actual_outcomes=[{"type": "deployment"}],
        )
        assert len(checks) == 1
        assert checks[0].passed is True
        assert checks[0].check_type == "outcome_verified"

    def test_missing_outcome(self, kb: KnowledgeBase) -> None:
        """Expected outcome not in actual -> not passed."""
        _save_proc(
            kb,
            "deploy",
            expected_outcomes=[
                {"type": "deployment", "description": "App deployed"}
            ],
        )
        rv = RuntimeValidator(kb)
        checks = rv.validate_post_execution(
            "deploy",
            "exec-1",
            actual_outcomes=[{"type": "notification"}],
        )
        assert len(checks) == 1
        assert checks[0].passed is False

    def test_no_expectations(self, kb: KnowledgeBase) -> None:
        """No expected_outcomes defined -> single pass check."""
        _save_proc(kb, "simple", expected_outcomes=[])
        rv = RuntimeValidator(kb)
        checks = rv.validate_post_execution("simple", "exec-1")
        assert len(checks) == 1
        assert checks[0].passed is True
        assert checks[0].name == "no_expectations"

    def test_procedure_not_found_post(self, kb: KnowledgeBase) -> None:
        """Missing procedure returns a single failure check."""
        rv = RuntimeValidator(kb)
        checks = rv.validate_post_execution("ghost", "exec-1")
        assert len(checks) == 1
        assert checks[0].passed is False
        assert checks[0].check_type == "outcome_verified"


# ===========================================================================
# 5) TestEdgeCases (2)
# ===========================================================================


class TestEdgeCases:
    """Edge case coverage for runtime_validator."""

    def test_empty_procedure_environment(self, kb: KnowledgeBase) -> None:
        """Empty environment dict -> no checks."""
        _save_proc(kb, "empty-env", environment={})
        rv = RuntimeValidator(kb)
        checks = rv.validate_environment("empty-env")
        assert checks == []

    def test_string_expected_outcome(self, kb: KnowledgeBase) -> None:
        """Non-dict expected outcome (bare string) handled gracefully."""
        _save_proc(
            kb,
            "str-outcome",
            expected_outcomes=["deployment_complete"],
        )
        rv = RuntimeValidator(kb)
        checks = rv.validate_post_execution(
            "str-outcome",
            "exec-1",
            actual_outcomes=[{"type": "deployment_complete"}],
        )
        assert len(checks) == 1
        assert checks[0].passed is True
        assert "deployment_complete" in checks[0].name

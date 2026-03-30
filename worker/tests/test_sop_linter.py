"""Tests for the SOP linter / validation gate."""

from __future__ import annotations

import pytest

from agenthandover_worker.sop_linter import LintIssue, LintResult, lint_sop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_sop(**overrides) -> dict:
    """Return a minimal valid SOP template."""
    sop = {
        "slug": "deploy-to-production",
        "title": "Deploy to Production",
        "task_description": "Deploys the latest build to the production server.",
        "steps": [
            {
                "step": "Open terminal",
                "app": "Terminal",
                "verify": "Terminal window is visible",
                "parameters": {},
            },
            {
                "step": "Run deploy command",
                "app": "Terminal",
                "verify": "Deploy output shows success",
                "parameters": {"input": "make deploy"},
            },
        ],
        "variables": [],
        "confidence_avg": 0.85,
    }
    sop.update(overrides)
    return sop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidSOP:
    def test_valid_sop_passes(self):
        result = lint_sop(_valid_sop())
        assert result.valid is True
        assert result.errors == []

    def test_valid_sop_with_warnings_still_valid(self):
        """Warnings don't block export."""
        sop = _valid_sop(task_description="")
        result = lint_sop(sop)
        assert result.valid is True
        assert len(result.warnings) > 0
        assert len(result.errors) == 0


class TestErrors:
    def test_missing_title_error(self):
        result = lint_sop(_valid_sop(title=""))
        assert result.valid is False
        errs = [i for i in result.errors if i.field == "title"]
        assert len(errs) == 1
        assert "title" in errs[0].message.lower()

    def test_missing_title_none(self):
        sop = _valid_sop()
        del sop["title"]
        result = lint_sop(sop)
        assert result.valid is False

    def test_missing_slug_error(self):
        result = lint_sop(_valid_sop(slug=""))
        assert result.valid is False
        errs = [i for i in result.errors if i.field == "slug"]
        assert len(errs) == 1

    def test_empty_steps_error(self):
        result = lint_sop(_valid_sop(steps=[]))
        assert result.valid is False
        errs = [i for i in result.errors if i.field == "steps"]
        assert any("no steps" in e.message.lower() for e in errs)

    def test_missing_steps_key(self):
        sop = _valid_sop()
        del sop["steps"]
        result = lint_sop(sop)
        assert result.valid is False

    def test_step_missing_action_error(self):
        sop = _valid_sop(steps=[
            {"app": "Chrome", "verify": "Page loaded"},
            {"step": "Click submit", "app": "Chrome", "verify": "Submitted"},
        ])
        result = lint_sop(sop)
        assert result.valid is False
        errs = [i for i in result.errors if "action" in i.field]
        assert len(errs) == 1

    def test_step_with_action_key_is_valid(self):
        """VLM raw format uses 'action' instead of 'step'."""
        sop = _valid_sop(steps=[
            {"action": "Open browser", "app": "Chrome", "verify": "ok"},
            {"action": "Navigate to URL", "app": "Chrome", "verify": "ok"},
        ])
        result = lint_sop(sop)
        assert result.valid is True

    def test_undeclared_variable_warning(self):
        """Undeclared variables are warnings, not errors (VLM output may be imprecise)."""
        sop = _valid_sop(steps=[
            {
                "step": "Navigate to {{base_url}}/dashboard",
                "app": "Chrome",
                "verify": "Dashboard visible",
            },
            {
                "step": "Enter {{ user_name }}",
                "app": "Chrome",
                "verify": "Name field filled",
            },
        ])
        sop["variables"] = [{"name": "base_url", "type": "string", "example": "https://example.com"}]
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "user_name" in i.message]
        assert len(warns) == 1

    def test_undeclared_variable_in_parameters(self):
        """Variables referenced inside step.parameters should also be checked."""
        sop = _valid_sop(steps=[
            {
                "step": "Type credentials",
                "app": "Chrome",
                "verify": "ok",
                "parameters": {"input": "{{password}}"},
            },
            {
                "step": "Submit",
                "app": "Chrome",
                "verify": "ok",
            },
        ])
        sop["variables"] = []
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "password" in i.message]
        assert len(warns) == 1

    def test_variable_ref_with_spaces(self):
        """Both {{var}} and {{ var }} should be detected."""
        sop = _valid_sop(steps=[
            {"step": "Go to {{ url }}", "app": "Chrome", "verify": "ok"},
            {"step": "Click save", "app": "Chrome", "verify": "ok"},
        ])
        sop["variables"] = [{"name": "url", "type": "string", "example": "https://x.com"}]
        result = lint_sop(sop)
        # url is declared, so no errors
        assert result.valid is True


class TestWarnings:
    def test_few_steps_warning(self):
        sop = _valid_sop(steps=[
            {"step": "Do the thing", "app": "App", "verify": "Done"},
        ])
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "too simple" in i.message.lower()]
        assert len(warns) == 1

    def test_missing_task_description_warning(self):
        sop = _valid_sop(task_description="")
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "task description" in i.message.lower()]
        assert len(warns) == 1

    def test_step_without_verify_warning(self):
        sop = _valid_sop(steps=[
            {"step": "Open app", "app": "Terminal"},
            {"step": "Run command", "app": "Terminal", "verify": "Output shown"},
        ])
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "verification" in i.message.lower()]
        assert len(warns) == 1
        assert "steps[0]" in warns[0].field

    def test_step_with_verify_in_parameters(self):
        """verify inside parameters dict should count."""
        sop = _valid_sop(steps=[
            {
                "step": "Open app",
                "app": "Terminal",
                "parameters": {"verify": "Terminal visible"},
            },
            {"step": "Run command", "app": "Terminal", "verify": "Done"},
        ])
        result = lint_sop(sop)
        warns = [i for i in result.warnings if "verification" in i.message.lower()]
        assert len(warns) == 0

    def test_step_without_app_warning(self):
        sop = _valid_sop(steps=[
            {"step": "Open file", "verify": "File is open"},
            {"step": "Edit file", "app": "VSCode", "verify": "Edited"},
        ])
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "app context" in i.message.lower()]
        assert len(warns) == 1

    def test_step_with_app_in_parameters(self):
        """app inside parameters dict should count."""
        sop = _valid_sop(steps=[
            {
                "step": "Open file",
                "verify": "ok",
                "parameters": {"app": "Finder"},
            },
            {"step": "Edit file", "app": "VSCode", "verify": "ok"},
        ])
        result = lint_sop(sop)
        warns = [i for i in result.warnings if "app context" in i.message.lower()]
        assert len(warns) == 0

    def test_duplicate_actions_warning(self):
        sop = _valid_sop(steps=[
            {"step": "Click submit", "app": "Chrome", "verify": "ok"},
            {"step": "Click submit", "app": "Chrome", "verify": "ok"},
        ])
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "duplicate" in i.message.lower()]
        assert len(warns) == 1

    def test_low_confidence_warning(self):
        sop = _valid_sop(confidence_avg=0.3)
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "low confidence" in i.message.lower()]
        assert len(warns) == 1

    def test_high_confidence_no_warning(self):
        sop = _valid_sop(confidence_avg=0.9)
        result = lint_sop(sop)
        warns = [i for i in result.warnings if "confidence" in i.message.lower()]
        assert len(warns) == 0

    def test_unused_variable_warning(self):
        sop = _valid_sop()
        sop["variables"] = [
            {"name": "env", "type": "string", "example": "prod"},
        ]
        result = lint_sop(sop)
        assert result.valid is True
        warns = [i for i in result.warnings if "never referenced" in i.message.lower()]
        assert len(warns) == 1
        assert "env" in warns[0].message


class TestLintResultProperties:
    def test_lint_result_properties(self):
        result = LintResult(
            valid=False,
            issues=[
                LintIssue("error", "title", "Missing title"),
                LintIssue("warning", "steps", "Too simple"),
                LintIssue("error", "slug", "Missing slug"),
            ],
        )
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.valid is False

    def test_empty_result_is_valid(self):
        result = LintResult(valid=True, issues=[])
        assert result.errors == []
        assert result.warnings == []

    def test_warnings_only_is_valid(self):
        result = LintResult(
            valid=True,
            issues=[LintIssue("warning", "x", "minor")],
        )
        assert result.valid is True
        assert len(result.warnings) == 1
        assert len(result.errors) == 0


class TestStringVariableDeclaration:
    """Variables may be declared as plain strings (not dicts)."""

    def test_string_variable_declaration(self):
        sop = _valid_sop(steps=[
            {"step": "Go to {{url}}", "app": "Chrome", "verify": "ok"},
            {"step": "Click save", "app": "Chrome", "verify": "ok"},
        ])
        sop["variables"] = ["url"]
        result = lint_sop(sop)
        assert result.valid is True
        # No undeclared variable error
        errs = [i for i in result.errors if "url" in i.message]
        assert len(errs) == 0

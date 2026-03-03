"""Tests for SKILL.md Export Adapter.

Tests cover:
- write_sop() produces correct markdown format
- File naming: SKILL.<slug>.md
- Index file generation
- Steps formatting with selectors
- Input variables section
- Enhanced SOPs include task_description + execution_overview
- list_sops() returns correct inventory
- Round-trip: write then list
- Edge cases: empty steps, missing fields, special characters in title
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from oc_apprentice_worker.skill_md_writer import SkillMdWriter


def _make_sop_template(**overrides) -> dict:
    """Create a standard SOP template dict for testing."""
    template = {
        "slug": "file-expense-report",
        "title": "File Expense Report",
        "steps": [
            {
                "step": "click",
                "target": "New Report button",
                "selector": "#new-report-btn",
                "parameters": {"app_id": "com.chrome.Chrome"},
                "confidence": 0.92,
                "pre_state": {"app_id": "com.chrome.Chrome", "url": "https://expenses.example.com"},
            },
            {
                "step": "type",
                "target": "Amount field",
                "selector": "input[name=amount]",
                "parameters": {"value": "42.50"},
                "confidence": 0.88,
                "pre_state": {},
            },
            {
                "step": "click",
                "target": "Submit button",
                "selector": "#submit-btn",
                "parameters": {},
                "confidence": 0.95,
                "pre_state": {},
            },
        ],
        "variables": [
            {"name": "amount", "type": "number", "example": "42.50"},
            {"name": "category", "type": "enum", "example": "Travel", "choices": ["Travel", "Office", "Food"]},
        ],
        "confidence_avg": 0.9167,
        "episode_count": 5,
        "apps_involved": ["com.chrome.Chrome"],
        "preconditions": ["app_open:com.chrome.Chrome", "url_open:https://expenses.example.com"],
        "postconditions": ["final_action:click"],
        "exceptions_seen": [],
    }
    template.update(overrides)
    return template


class TestSkillMdWriter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = SkillMdWriter(workspace_dir=self.tmpdir)

    def test_write_sop_creates_file(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        assert path.exists()
        assert path.name == "SKILL.file-expense-report.md"

    def test_write_sop_content_has_title(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert content.startswith("# File Expense Report")

    def test_write_sop_content_has_steps(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Steps" in content
        assert "**Click New Report button**" in content
        assert "**Type Amount field**" in content
        assert "**Click Submit button**" in content

    def test_write_sop_content_has_selectors(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "`#new-report-btn`" in content
        assert "`input[name=amount]`" in content

    def test_write_sop_content_has_variables(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Input Variables" in content
        assert "{{amount}}" in content
        assert "{{category}}" in content

    def test_write_sop_content_has_metadata(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Metadata" in content
        assert "Confidence: 0.92" in content
        assert "Observed: 5 time(s)" in content
        assert "Schema: 2.0.0" in content

    def test_write_sop_content_has_when_to_use(self):
        template = _make_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## When to Use" in content
        assert "com.chrome.Chrome" in content
        assert "https://expenses.example.com" in content

    def test_write_all_sops_creates_index(self):
        templates = [
            _make_sop_template(),
            _make_sop_template(slug="submit-form", title="Submit Form", episode_count=3),
        ]
        paths = self.writer.write_all_sops(templates)
        assert len(paths) == 2

        index_path = self.writer.skills_dir / "SKILLS-INDEX.md"
        assert index_path.exists()
        content = index_path.read_text()
        assert "# Skills Index" in content
        assert "file-expense-report" in content
        assert "submit-form" in content

    def test_list_sops_returns_correct_inventory(self):
        template = _make_sop_template()
        self.writer.write_sop(template)

        sops = self.writer.list_sops()
        assert len(sops) == 1
        assert sops[0]["slug"] == "file-expense-report"
        assert sops[0]["title"] == "File Expense Report"
        assert sops[0]["size_bytes"] > 0

    def test_list_sops_empty_dir(self):
        assert self.writer.list_sops() == []

    def test_roundtrip_write_then_list(self):
        templates = [
            _make_sop_template(),
            _make_sop_template(slug="login-workflow", title="Login Workflow"),
        ]
        self.writer.write_all_sops(templates)

        sops = self.writer.list_sops()
        slugs = {s["slug"] for s in sops}
        assert "file-expense-report" in slugs
        assert "login-workflow" in slugs

    def test_enhanced_sop_includes_task_description(self):
        template = _make_sop_template(
            task_description="This workflow files an expense report in the company's internal system."
        )
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "This workflow files an expense report" in content

    def test_enhanced_sop_includes_execution_overview(self):
        template = _make_sop_template(
            execution_overview={
                "goal": "Submit a new expense report",
                "prerequisites": "Must be logged into the expense portal",
                "success_criteria": "Report appears in pending queue",
            }
        )
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Execution Overview" in content
        assert "Submit a new expense report" in content
        assert "Must be logged into the expense portal" in content

    def test_empty_steps_sop(self):
        template = _make_sop_template(steps=[])
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Steps" in content
        # Should still have metadata
        assert "## Metadata" in content

    def test_missing_fields_handled(self):
        """Minimal SOP template with only required fields."""
        template = {"slug": "minimal", "title": "Minimal SOP"}
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "# Minimal SOP" in content
        assert "## Steps" in content

    def test_special_characters_in_title(self):
        template = _make_sop_template(
            slug="upload-file-csv",
            title="Upload File (*.csv) & Submit"
        )
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "# Upload File (*.csv) & Submit" in content

    def test_focus_recording_source_metadata(self):
        template = _make_sop_template(
            source="focus_recording",
            episode_count=1,
        )
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "Focus recording" in content

    def test_file_naming_slugification(self):
        template = _make_sop_template(slug="My Complex_Slug with SPACES!")
        path = self.writer.write_sop(template)
        # Should be lowercased and hyphenated
        assert "SKILL." in path.name
        assert path.name.endswith(".md")
        # No spaces or uppercase in the slug portion
        slug_part = path.stem[6:]  # Remove "SKILL."
        assert " " not in slug_part

    def test_get_sops_dir(self):
        expected = Path(self.tmpdir).resolve() / "skills"
        assert self.writer.get_sops_dir() == expected

    def test_write_all_sops_empty_removes_stale_index(self):
        """When called with empty list, stale SKILLS-INDEX.md should be removed."""
        # First, create an index by writing some SOPs
        templates = [
            _make_sop_template(),
            _make_sop_template(slug="second-sop", title="Second SOP"),
        ]
        self.writer.write_all_sops(templates)
        index_path = self.writer.skills_dir / "SKILLS-INDEX.md"
        assert index_path.exists()

        # Now call with empty list — index should be cleaned up
        self.writer.write_all_sops([])
        assert not index_path.exists()

    def test_write_all_sops_empty_no_crash_without_index(self):
        """Empty call when no index exists should not crash."""
        result = self.writer.write_all_sops([])
        assert result == []

    def test_write_metadata(self):
        path = self.writer.write_metadata("test_meta", {"key": "value"})
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["key"] == "value"
        assert "generated_at" in data


def _make_v2_sop_template(**overrides) -> dict:
    """Create a v2 SOP template (from VLM pipeline) for testing."""
    template = {
        "slug": "deploy-feature-staging",
        "title": "Deploy Feature to Staging",
        "source": "v2_focus_recording",
        "steps": [
            {
                "step": "Review code changes",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "VS Code",
                    "location": "~/openmimic/worker/main.py",
                    "verify": "git status shows expected files changed",
                },
                "confidence": 0.85,
                "pre_state": {},
            },
            {
                "step": "Run unit tests",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "VS Code → Terminal",
                    "input": "pytest tests/ -v",
                    "verify": "Output shows all tests passed",
                },
                "confidence": 0.90,
                "pre_state": {},
            },
            {
                "step": "Commit and push",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "VS Code → Terminal",
                    "input": "git add -A && git commit -m '{{commit_message}}'",
                    "verify": "Push completes without errors",
                },
                "confidence": 0.88,
                "pre_state": {},
            },
        ],
        "variables": [
            {
                "name": "commit_message",
                "type": "string",
                "example": "feat: add scene annotator",
                "description": "Descriptive commit message",
            },
            {
                "name": "staging_url",
                "type": "string",
                "example": "staging-api.openmimic.dev",
                "description": "Staging base URL",
            },
        ],
        "confidence_avg": 0.82,
        "episode_count": 1,
        "apps_involved": ["VS Code", "Chrome"],
        "preconditions": ["Repository access with push permissions", "CI/CD pipeline configured"],
        "task_description": "This workflow deploys a new feature from local development to staging.",
        "execution_overview": {
            "when_to_use": "New feature branch is ready for staging",
            "success_criteria": "All CI jobs pass; staging health endpoint returns healthy",
            "common_errors": "CI fails on test job; Staging health returns old version",
        },
        "confidence_breakdown": {
            "demo_count": 0.15,
            "step_consistency": 0.30,
            "annotation_quality": 0.17,
            "variable_detection": 0.10,
            "focus_bonus": 0.10,
            "reasons": ["demos=1", "focus_recording_bonus"],
        },
    }
    template.update(overrides)
    return template


class TestSkillMdWriterV2:
    """Tests for v2 (VLM pipeline) SOP rendering."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = SkillMdWriter(workspace_dir=self.tmpdir)

    def test_v2_detected_by_source(self):
        template = _make_v2_sop_template()
        assert self.writer._is_v2_sop(template)

    def test_v1_not_detected_as_v2(self):
        template = _make_sop_template()
        assert not self.writer._is_v2_sop(template)

    def test_v2_has_description_section(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Description" in content
        assert "deploys a new feature" in content

    def test_v2_has_when_to_use(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## When to Use" in content
        assert "New feature branch is ready" in content

    def test_v2_has_prerequisites(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Prerequisites" in content
        assert "Repository access" in content
        assert "CI/CD pipeline" in content

    def test_v2_steps_have_semantic_fields(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        # Step headings
        assert "### Step 1:" in content
        assert "### Step 2:" in content
        assert "### Step 3:" in content
        # Semantic fields
        assert "- **Action**:" in content
        assert "- **App**: VS Code" in content
        assert "- **Verify**:" in content

    def test_v2_step_input_rendered(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "- **Input**: `pytest tests/ -v`" in content

    def test_v2_step_location_rendered(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "- **Location**: `~/openmimic/worker/main.py`" in content

    def test_v2_success_criteria(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Success Criteria" in content
        assert "CI jobs pass" in content

    def test_v2_variables_table(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Variables" in content
        assert "| Variable |" in content
        assert "{{commit_message}}" in content
        assert "feat: add scene annotator" in content

    def test_v2_common_errors(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Common Errors" in content
        assert "CI fails on test job" in content

    def test_v2_metadata_focus_mode(self):
        template = _make_v2_sop_template(source="v2_focus_recording")
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Metadata" in content
        assert "Mode: Focus Recording" in content

    def test_v2_metadata_passive_mode(self):
        template = _make_v2_sop_template(source="v2_passive_discovery")
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "Mode: Passive Discovery" in content

    def test_v2_metadata_demonstrations(self):
        template = _make_v2_sop_template(episode_count=3)
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "3 demonstration(s)" in content

    def test_v2_metadata_apps_listed(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "Apps: VS Code, Chrome" in content

    def test_v2_no_dom_hints_when_no_selectors(self):
        template = _make_v2_sop_template()
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## DOM Hints" not in content

    def test_v2_dom_hints_when_selectors_present(self):
        template = _make_v2_sop_template()
        template["steps"][0]["selector"] = "#review-btn"
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## DOM Hints" in content
        assert "<details>" in content
        assert "`#review-btn`" in content

    def test_v2_empty_task_description_no_section(self):
        template = _make_v2_sop_template(task_description="")
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Description" not in content

    def test_v2_no_prerequisites_no_section(self):
        template = _make_v2_sop_template(preconditions=[])
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Prerequisites" not in content

    def test_v2_no_variables_no_section(self):
        template = _make_v2_sop_template()
        template["variables"] = []
        path = self.writer.write_sop(template)
        content = path.read_text()
        assert "## Variables" not in content

    def test_v2_index_includes_v2_sops(self):
        templates = [_make_v2_sop_template()]
        self.writer.write_all_sops(templates)
        index_path = self.writer.skills_dir / "SKILLS-INDEX.md"
        assert index_path.exists()
        content = index_path.read_text()
        assert "deploy-feature-staging" in content

    def test_v2_mixed_v1_v2_export(self):
        """Both v1 and v2 SOPs export correctly side by side."""
        templates = [
            _make_sop_template(),
            _make_v2_sop_template(),
        ]
        paths = self.writer.write_all_sops(templates)
        assert len(paths) == 2

        v1_content = paths[0].read_text()
        v2_content = paths[1].read_text()

        # v1 uses numbered list steps
        assert "1. **Click" in v1_content
        # v2 uses ### Step N: headings
        assert "### Step 1:" in v2_content

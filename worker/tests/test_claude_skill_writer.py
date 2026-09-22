"""Tests for Claude Code Skill Export Adapter.

Rendering is ``skill_render``'s job (see ``test_skill_render.py``); these
tests cover the writer's file layout:
- File operations (write_sop, write_all_sops, list_sops, write_metadata, index)
- Edge cases (v1 legacy, empty steps, special chars in slug)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_v2_focus_sop(**overrides) -> dict:
    """Create a v2 focus-recording SOP template for testing."""
    template = {
        "slug": "search-product-amazon",
        "title": "Search for a Product on Amazon",
        "source": "v2_focus_recording",
        "steps": [
            {
                "step": "Open Amazon homepage",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "Chrome",
                    "location": "https://www.amazon.com",
                    "verify": "Amazon homepage loads",
                },
                "confidence": 0.90,
                "pre_state": {},
            },
            {
                "step": "Enter search query",
                "target": "",
                "selector": "#twotabsearchtextbox",
                "parameters": {
                    "app": "Chrome",
                    "input": "{{search_query}}",
                    "verify": "Search results page shows items matching the query",
                },
                "confidence": 0.88,
                "pre_state": {},
            },
            {
                "step": "Select category filter",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "Chrome",
                    "input": "{{category}}",
                    "location": "left sidebar",
                    "verify": "Results are filtered to the selected category",
                },
                "confidence": 0.80,
                "pre_state": {},
            },
        ],
        "variables": [
            {
                "name": "search_query",
                "type": "string",
                "example": "wireless earbuds",
                "description": "Product search term",
            },
            {
                "name": "category",
                "type": "string",
                "example": "Electronics",
                "description": "Product category filter",
            },
        ],
        "confidence_avg": 0.85,
        "episode_count": 2,
        "apps_involved": ["Chrome"],
        "preconditions": ["Amazon.com is accessible", "Browser is open"],
        "task_description": "Search for a product on Amazon by entering a query and filtering by category.",
        "execution_overview": {
            "when_to_use": "When you need to find a product on Amazon",
            "prerequisites": "Browser must be open; Amazon.com must be accessible",
            "success_criteria": "Search results page displays products matching the query in the selected category",
            "common_errors": "No results found: Try a broader search query; Category not available: Skip the filter step",
        },
    }
    template.update(overrides)
    return template


def _make_v2_passive_sop(**overrides) -> dict:
    """Create a v2 passive-discovery SOP template for testing."""
    template = {
        "slug": "deploy-feature-staging",
        "title": "Deploy Feature to Staging",
        "source": "v2_passive_discovery",
        "steps": [
            {
                "step": "Review code changes",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "VS Code",
                    "location": "~/project/src/main.py",
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
                    "app": "VS Code \u2192 Terminal",
                    "input": "pytest tests/ -v",
                    "verify": "All tests passed",
                },
                "confidence": 0.90,
                "pre_state": {},
            },
            {
                "step": "Commit and push",
                "target": "",
                "selector": None,
                "parameters": {
                    "app": "VS Code \u2192 Terminal",
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
                "example": "feat: add new endpoint",
                "description": "Descriptive commit message",
            },
        ],
        "confidence_avg": 0.82,
        "episode_count": 3,
        "apps_involved": ["VS Code", "Chrome"],
        "preconditions": ["Repository access with push permissions"],
        "task_description": "Deploy a new feature from local development to staging.",
        "execution_overview": {
            "when_to_use": "New feature branch is ready for staging",
            "success_criteria": "CI passes; staging endpoint returns healthy",
            "common_errors": "CI fails: Check test logs; Push rejected: Pull first",
        },
    }
    template.update(overrides)
    return template


def _make_v1_sop(**overrides) -> dict:
    """Create a legacy v1 SOP template (no v2_ source prefix)."""
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
                "pre_state": {},
            },
            {
                "step": "type",
                "target": "Amount field",
                "selector": "input[name=amount]",
                "parameters": {"value": "42.50"},
                "confidence": 0.88,
                "pre_state": {},
            },
        ],
        "variables": [],
        "confidence_avg": 0.90,
        "episode_count": 5,
        "apps_involved": ["com.chrome.Chrome"],
        "preconditions": [],
    }
    template.update(overrides)
    return template


# ------------------------------------------------------------------
# Tests: Frontmatter generation
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Tests: Step rendering
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Tests: Variable mapping
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Tests: Full rendering
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Tests: File operations
# ------------------------------------------------------------------


class TestFileOperations:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ClaudeSkillWriter(skills_dir=self.tmpdir)

    def test_write_sop_creates_directory(self):
        """write_sop creates ~/.claude/skills/<slug>/SKILL.md."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        assert path.exists()
        assert path.name == "SKILL.md"
        assert path.parent.name == "search-product-amazon"

    def test_write_all_sops(self):
        """Multiple skills are written correctly."""
        sops = [_make_v2_focus_sop(), _make_v2_passive_sop()]
        paths = self.writer.write_all_sops(sops)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert paths[0].parent.name == "search-product-amazon"
        assert paths[1].parent.name == "deploy-feature-staging"

    def test_list_sops(self):
        """list_sops returns correct inventory after writing."""
        sops = [_make_v2_focus_sop(), _make_v2_passive_sop()]
        self.writer.write_all_sops(sops)

        listed = self.writer.list_sops()
        assert len(listed) == 2
        slugs = {s["slug"] for s in listed}
        assert "search-product-amazon" in slugs
        assert "deploy-feature-staging" in slugs

        # Each entry should have required fields
        for entry in listed:
            assert "title" in entry
            assert "path" in entry
            assert "size_bytes" in entry
            assert entry["size_bytes"] > 0

    def test_list_sops_empty_dir(self):
        """list_sops returns empty list when no skills exist."""
        assert self.writer.list_sops() == []

    def test_write_metadata(self):
        """write_metadata creates a JSON file."""
        path = self.writer.write_metadata("export_info", {"format": "claude-skill", "count": 5})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["format"] == "claude-skill"
        assert data["count"] == 5
        assert "generated_at" in data
        assert data["metadata_type"] == "export_info"

    def test_index_generation(self):
        """AGENTHANDOVER-INDEX.md is created on write_all_sops."""
        sops = [_make_v2_focus_sop(), _make_v2_passive_sop()]
        self.writer.write_all_sops(sops)

        index_path = Path(self.tmpdir) / "AGENTHANDOVER-INDEX.md"
        assert index_path.exists()

        content = index_path.read_text()
        assert "# AgentHandover Exported Skills" in content
        assert "search-product-amazon" in content
        assert "deploy-feature-staging" in content
        assert "| Skill |" in content  # Table header

    def test_index_removed_on_empty(self):
        """Index is removed when called with empty SOP list."""
        # First create index
        self.writer.write_all_sops([_make_v2_focus_sop()])
        index_path = Path(self.tmpdir) / "AGENTHANDOVER-INDEX.md"
        assert index_path.exists()

        # Now call with empty
        self.writer.write_all_sops([])
        assert not index_path.exists()

    def test_get_sops_dir(self):
        """get_sops_dir returns the skills root."""
        assert self.writer.get_sops_dir() == Path(self.tmpdir).resolve()


# ------------------------------------------------------------------
# Tests: Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ClaudeSkillWriter(skills_dir=self.tmpdir)

    def test_v1_sop_template(self):
        """Legacy v1 SOP template is handled gracefully (no crash)."""
        sop = _make_v1_sop()
        path = self.writer.write_sop(sop)
        assert path.exists()
        content = path.read_text()
        # Should still produce valid frontmatter
        assert content.startswith("---\n")
        assert "name: file-expense-report" in content
        # Steps should render (even if simpler)
        assert "## Steps" in content

    def test_empty_steps(self):
        """SOP with no steps produces valid skill file."""
        sop = _make_v2_focus_sop(steps=[])
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert content.startswith("---\n")
        assert "## Steps" in content
        # Footer should still be present
        assert "Generated by AgentHandover" in content

    def test_special_chars_in_slug(self):
        """Special characters in slug produce filesystem-safe names."""
        sop = _make_v2_focus_sop(slug="My Complex_Slug with SPACES!")
        path = self.writer.write_sop(sop)
        assert path.exists()
        # Directory name should be slugified
        dir_name = path.parent.name
        assert " " not in dir_name
        assert dir_name == dir_name.lower()
        assert "!" not in dir_name

    def test_missing_execution_overview(self):
        """SOP without execution_overview still renders."""
        sop = _make_v2_focus_sop()
        del sop["execution_overview"]
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert content.startswith("---\n")
        # Should not crash, sections just absent
        assert "## Common errors" not in content
        assert "## Success criteria" not in content

    def test_write_sop_overwrite(self):
        """Writing same slug twice overwrites cleanly."""
        sop = _make_v2_focus_sop()
        path1 = self.writer.write_sop(sop)
        content1 = path1.read_text()

        sop["task_description"] = "Updated description for the skill."
        path2 = self.writer.write_sop(sop)
        content2 = path2.read_text()

        assert path1 == path2
        assert "Updated description" in content2
        assert "Updated description" not in content1

    def test_frontmatter_parseable(self):
        """Generated frontmatter can be parsed back."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()

        fm = ClaudeSkillWriter._parse_frontmatter_from_text(content)
        assert fm["name"] == "search-product-amazon"
        assert "description" in fm
        assert "allowed-tools" in fm

"""Tests for Claude Code Skill Export Adapter.

Tests cover:
- Frontmatter generation (name, description, argument-hint, allowed-tools)
- Step rendering (natural language, variable substitution, verify notes)
- Variable mapping (ordered by appearance, empty case)
- Full rendering (v2 focus SOPs, passive SOPs, prerequisites, errors, DOM hints)
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


class TestFrontmatter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ClaudeSkillWriter(skills_dir=self.tmpdir)

    def test_frontmatter_basic(self):
        """Frontmatter contains name and description."""
        sop = _make_v2_focus_sop()
        fm = self.writer._build_frontmatter(sop)
        assert "name: search-product-amazon" in fm
        assert "description:" in fm
        assert fm.startswith("---\n")
        assert fm.strip().endswith("---")

    def test_frontmatter_with_variables(self):
        """argument-hint is present when variables exist."""
        sop = _make_v2_focus_sop()
        fm = self.writer._build_frontmatter(sop)
        assert "argument-hint: [search_query] [category]" in fm

    def test_frontmatter_no_variables(self):
        """argument-hint is absent when no variables."""
        sop = _make_v2_focus_sop(variables=[])
        fm = self.writer._build_frontmatter(sop)
        assert "argument-hint" not in fm

    def test_frontmatter_allowed_tools_browser(self):
        """Chrome maps to Bash in allowed-tools."""
        sop = _make_v2_focus_sop(apps_involved=["Chrome"])
        fm = self.writer._build_frontmatter(sop)
        assert "allowed-tools:" in fm
        assert "Bash" in fm

    def test_frontmatter_allowed_tools_terminal(self):
        """Terminal/VS Code maps to Bash."""
        sop = _make_v2_focus_sop(apps_involved=["VS Code"])
        fm = self.writer._build_frontmatter(sop)
        assert "Bash" in fm

    def test_frontmatter_allowed_tools_default(self):
        """No apps -> Bash, Read as defaults."""
        sop = _make_v2_focus_sop(apps_involved=[])
        fm = self.writer._build_frontmatter(sop)
        assert "Bash" in fm
        assert "Read" in fm


# ------------------------------------------------------------------
# Tests: Step rendering
# ------------------------------------------------------------------


class TestStepRendering:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ClaudeSkillWriter(skills_dir=self.tmpdir)

    def test_step_with_all_fields(self):
        """Step with app, location, input, verify all present."""
        step = {
            "step": "Enter search query",
            "parameters": {
                "app": "Chrome",
                "location": "https://www.amazon.com",
                "input": "wireless earbuds",
                "verify": "Results page loads",
            },
        }
        rendered = self.writer._render_step(step, {}, step_num=1)
        assert "1." in rendered
        assert "**Chrome**" in rendered
        assert "`https://www.amazon.com`" in rendered
        assert "`wireless earbuds`" in rendered
        assert "_Verify: Results page loads_" in rendered

    def test_step_minimal(self):
        """Step with just action, no params."""
        step = {"step": "Click submit button", "parameters": {}}
        rendered = self.writer._render_step(step, {}, step_num=3)
        assert "3." in rendered
        assert "Click submit button" in rendered

    def test_step_variable_substitution(self):
        """{{email}} is replaced with $0."""
        step = {
            "step": "Enter email address",
            "parameters": {"input": "{{email}}"},
        }
        var_map = {"email": 0}
        rendered = self.writer._render_step(step, var_map, step_num=1)
        assert "$0" in rendered
        assert "{{email}}" not in rendered

    def test_step_multiple_variables(self):
        """Multiple variables get correct index mapping."""
        step = {
            "step": "Fill form",
            "parameters": {"input": "{{first_name}} {{last_name}}"},
        }
        var_map = {"first_name": 0, "last_name": 1}
        rendered = self.writer._render_step(step, var_map, step_num=1)
        assert "$0" in rendered
        assert "$1" in rendered
        assert "{{first_name}}" not in rendered
        assert "{{last_name}}" not in rendered

    def test_step_with_verify(self):
        """Verify is rendered as italicized note."""
        step = {
            "step": "Click Submit",
            "parameters": {"verify": "Form is submitted"},
        }
        rendered = self.writer._render_step(step, {}, step_num=1)
        assert "_Verify: Form is submitted_" in rendered


# ------------------------------------------------------------------
# Tests: Variable mapping
# ------------------------------------------------------------------


class TestVariableMapping:
    def setup_method(self):
        self.writer = ClaudeSkillWriter(skills_dir=tempfile.mkdtemp())

    def test_variable_map_basic(self):
        """Variables are indexed in order of appearance."""
        variables = [
            {"name": "email", "type": "string"},
            {"name": "password", "type": "string"},
        ]
        var_map = self.writer._build_variable_map(variables)
        assert var_map == {"email": 0, "password": 1}

    def test_variable_map_empty(self):
        """No variables -> empty map."""
        var_map = self.writer._build_variable_map([])
        assert var_map == {}


# ------------------------------------------------------------------
# Tests: Full rendering
# ------------------------------------------------------------------


class TestFullRendering:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ClaudeSkillWriter(skills_dir=self.tmpdir)

    def test_render_v2_focus_sop(self):
        """Complete v2 focus recording SOP renders correctly."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()

        # Frontmatter
        assert content.startswith("---\n")
        assert "name: search-product-amazon" in content
        assert "argument-hint: [search_query] [category]" in content

        # Body
        assert "## Steps" in content
        assert "1." in content
        assert "2." in content
        assert "3." in content
        assert "$0" in content  # search_query
        assert "$1" in content  # category
        assert "_Verify:" in content

        # Footer
        assert "Generated by AgentHandover" in content
        assert "confidence: 0.85" in content

    def test_render_v2_passive_sop(self):
        """Multi-demo passive SOP renders correctly."""
        sop = _make_v2_passive_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()

        assert "name: deploy-feature-staging" in content
        assert "argument-hint: [commit_message]" in content
        assert "$0" in content  # commit_message
        assert "3 demonstration(s)" in content

    def test_render_with_prerequisites(self):
        """Prerequisites section is rendered."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "## Prerequisites" in content
        assert "Amazon.com is accessible" in content
        assert "Browser is open" in content

    def test_render_with_common_errors(self):
        """Common errors are rendered as warning notes."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "## Common Errors" in content
        assert "No results found" in content

    def test_render_with_dom_hints(self):
        """DOM selectors from steps appear in browser automation notes."""
        sop = _make_v2_focus_sop()
        # Step 2 has selector="#twotabsearchtextbox"
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "## Browser Automation Notes" in content
        assert "#twotabsearchtextbox" in content

    def test_render_with_success_criteria(self):
        """Success criteria section renders from execution_overview."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "## Success Criteria" in content
        assert "Search results page" in content

    def test_render_arguments_reference(self):
        """Arguments reference line is present when variables exist."""
        sop = _make_v2_focus_sop()
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "**Arguments:**" in content
        assert "`$0` = search_query" in content
        assert "`$1` = category" in content

    def test_render_no_arguments_when_no_variables(self):
        """No arguments line when no variables."""
        sop = _make_v2_focus_sop(variables=[])
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "**Arguments:**" not in content

    def test_render_with_timeline_dom_hints(self):
        """Timeline DOM nodes are rendered in browser automation notes."""
        sop = _make_v2_focus_sop()
        sop["_timeline"] = [{
            "dom_nodes": [
                {"tag": "button", "text": "Add to Cart", "id": "add-to-cart-btn"},
                {"tag": "input", "type": "text", "ariaLabel": "Search", "role": "textbox"},
            ],
        }]
        path = self.writer.write_sop(sop)
        content = path.read_text()
        assert "## Browser Automation Notes" in content
        assert "add-to-cart-btn" in content


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
        assert "## Common Errors" not in content
        assert "## Success Criteria" not in content

    def test_missing_task_description(self):
        """SOP without task_description falls back to title."""
        sop = _make_v2_focus_sop()
        del sop["task_description"]
        path = self.writer.write_sop(sop)
        content = path.read_text()
        # Body (after closing frontmatter ---) should start with title
        # Split on "---\n" gives [empty, frontmatter, body...]
        parts = content.split("---\n", 2)
        body = parts[2].strip() if len(parts) >= 3 else ""
        assert body.startswith("Search for a Product on Amazon.")

    def test_description_derivation_from_task_desc(self):
        """Description comes from first sentence of task_description."""
        sop = _make_v2_focus_sop(
            task_description="Search for a product on Amazon. This is detailed."
        )
        desc = ClaudeSkillWriter._derive_description(sop)
        assert desc == "Search for a product on Amazon."

    def test_description_derivation_from_when_to_use(self):
        """Description falls back to when_to_use if no task_description."""
        sop = _make_v2_focus_sop(task_description="")
        desc = ClaudeSkillWriter._derive_description(sop)
        assert "find a product" in desc.lower() or "when" in desc.lower()

    def test_description_derivation_from_title(self):
        """Description falls back to title if nothing else."""
        sop = {
            "slug": "test",
            "title": "My Test Skill",
        }
        desc = ClaudeSkillWriter._derive_description(sop)
        assert desc == "My Test Skill."

    def test_allowed_tools_filesystem_apps(self):
        """Finder maps to Read, Write."""
        tools = self.writer._derive_allowed_tools(["Finder"])
        assert "Read" in tools
        assert "Write" in tools

    def test_allowed_tools_unknown_apps(self):
        """Unknown apps get broad tool access."""
        tools = self.writer._derive_allowed_tools(["SomeObscureApp"])
        assert "Bash" in tools
        assert "Read" in tools
        assert "Write" in tools
        assert "Grep" in tools

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

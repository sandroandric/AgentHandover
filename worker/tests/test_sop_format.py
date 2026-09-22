"""Tests for manual edit detection of rendered SOP files."""

from __future__ import annotations

from pathlib import Path

from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.skill_render import render_packet
from agenthandover_worker.sop_format import SOPFormatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_sop_template(
    slug: str = "test_workflow",
    title: str = "Test Workflow",
    confidence_avg: float = 0.88,
    steps: list[dict] | None = None,
    variables: list[dict] | None = None,
    apps: list[str] | None = None,
) -> dict:
    if steps is None:
        steps = [
            {
                "step": "click",
                "target": "Submit button",
                "selector": "[aria-label='Submit']",
                "parameters": {"text": "Hello"},
                "confidence": 0.9,
            },
            {
                "step": "type",
                "target": "Email field",
                "selector": "#email",
                "parameters": {"text": "user@example.com"},
                "confidence": 0.85,
            },
        ]
    if variables is None:
        variables = []
    if apps is None:
        apps = ["Chrome"]

    return {
        "slug": slug,
        "title": title,
        "steps": steps,
        "variables": variables,
        "confidence_avg": confidence_avg,
        "episode_count": 5,
        "apps_involved": apps,
    }


def _rendered(template: dict | None = None) -> str:
    return render_packet(
        sop_to_procedure(template or _sample_sop_template()), target="openclaw",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBodyHashComputation:
    """SHA-256 hash of body content."""

    def test_hash_is_sha256_prefixed(self):
        formatter = SOPFormatter()
        body = "# Test\n\nSome content"
        result = formatter._compute_body_hash(body)

        assert result.startswith("sha256:")
        hex_part = result[7:]
        assert len(hex_part) == 64  # SHA-256 hex digest length

    def test_hash_is_deterministic(self):
        formatter = SOPFormatter()
        body = "# Test\n\nSome content"
        h1 = formatter._compute_body_hash(body)
        h2 = formatter._compute_body_hash(body)
        assert h1 == h2

    def test_hash_changes_with_content(self):
        formatter = SOPFormatter()
        h1 = formatter._compute_body_hash("Content A")
        h2 = formatter._compute_body_hash("Content B")
        assert h1 != h2

    def test_hash_normalizes_line_endings(self):
        formatter = SOPFormatter()
        h1 = formatter._compute_body_hash("line1\nline2\n")
        h2 = formatter._compute_body_hash("line1\r\nline2\n")
        assert h1 == h2


class TestManualEditDetectionNoEdit:
    """Hash matches -> no edit detected."""

    def test_no_edit_detected(self, tmp_path: Path):
        formatter = SOPFormatter()
        content = _rendered()

        filepath = tmp_path / "test_sop.md"
        filepath.write_text(content, encoding="utf-8")

        was_edited, reason = formatter.detect_manual_edit(str(filepath))
        assert was_edited is False
        assert reason == "hash_matches"


class TestManualEditDetectionEdited:
    """Hash mismatch -> edit detected."""

    def test_edit_detected(self, tmp_path: Path):
        formatter = SOPFormatter()
        content = _rendered()

        filepath = tmp_path / "test_sop.md"
        filepath.write_text(content, encoding="utf-8")

        # Simulate manual edit by appending to body
        modified = content + "\n\n<!-- Human note: reviewed and approved -->"
        filepath.write_text(modified, encoding="utf-8")

        was_edited, reason = formatter.detect_manual_edit(str(filepath))
        assert was_edited is True
        assert reason == "body_hash_mismatch"


class TestFrontmatterExtraction:
    """Parse existing SOP file into frontmatter and body."""

    def test_extracts_frontmatter_and_body(self):
        formatter = SOPFormatter()
        content = "---\nsop_version: 1\nsop_slug: test\n---\n\n# Title\n\nBody here"
        fm, body = formatter._extract_frontmatter_and_body(content)

        assert fm["sop_version"] == 1
        assert fm["sop_slug"] == "test"
        assert "Title" in body
        assert "Body here" in body

    def test_no_frontmatter(self):
        formatter = SOPFormatter()
        content = "# Just a markdown file\n\nNo frontmatter here."
        fm, body = formatter._extract_frontmatter_and_body(content)

        assert fm == {}
        assert "Just a markdown file" in body


class TestMissingFileNotEdited:
    """New file -> not edited."""

    def test_missing_file_returns_false(self, tmp_path: Path):
        formatter = SOPFormatter()
        was_edited, reason = formatter.detect_manual_edit(
            str(tmp_path / "nonexistent.md")
        )
        assert was_edited is False
        assert reason == "file_not_found"

    def test_file_without_hash_not_edited(self, tmp_path: Path):
        formatter = SOPFormatter()
        filepath = tmp_path / "no_hash.md"
        filepath.write_text("---\nname: x\n---\n\n# Content\n", encoding="utf-8")

        was_edited, reason = formatter.detect_manual_edit(str(filepath))
        assert was_edited is False
        assert reason == "no_hash_in_frontmatter"

    def test_legacy_file_is_replaceable(self, tmp_path: Path):
        """Pre-renderer files hashed only part of their body; never 'edited'."""
        formatter = SOPFormatter()
        filepath = tmp_path / "legacy.md"
        filepath.write_text(
            "---\nsop_version: 1\ngenerated_body_hash: sha256:abc\n---\n\n# X\n",
            encoding="utf-8",
        )
        assert formatter.detect_manual_edit(str(filepath)) == (False, "legacy_format")

    def test_triple_dash_inside_a_value_does_not_end_frontmatter(self):
        content = render_packet(
            sop_to_procedure(_sample_sop_template(title="A --- B: C")),
            target="openclaw",
        )
        fm, body = SOPFormatter._extract_frontmatter_and_body(content)
        assert fm["title"] == "A --- B: C"
        assert body.startswith("# A --- B: C\n")
        assert fm["generated_body_hash"] == SOPFormatter._compute_body_hash(body)

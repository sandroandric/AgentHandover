"""Tests for the SOP Versioner — drift detection, archiving, draft creation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from agenthandover_worker.sop_format import SOPFormatter
from agenthandover_worker.sop_versioner import SOPVersioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sop_content(formatter: SOPFormatter, slug: str = "test_sop") -> str:
    """Generate a valid SOP file content with proper frontmatter and hash."""
    template = {
        "slug": slug,
        "title": "Test SOP",
        "steps": [
            {"step": "click", "target": "Button A", "selector": None,
             "parameters": {}, "confidence": 0.9},
            {"step": "type", "target": "Field B", "selector": None,
             "parameters": {"text": "hello"}, "confidence": 0.85},
        ],
        "variables": [],
        "confidence_avg": 0.87,
        "episode_count": 3,
        "apps_involved": ["Chrome"],
    }
    return formatter.format_sop(template)


def _make_sop_content_v2(formatter: SOPFormatter, slug: str = "test_sop") -> str:
    """Generate a second version of SOP content (different body)."""
    template = {
        "slug": slug,
        "title": "Test SOP V2",
        "steps": [
            {"step": "click", "target": "Button A", "selector": None,
             "parameters": {}, "confidence": 0.92},
            {"step": "type", "target": "Field B", "selector": None,
             "parameters": {"text": "hello world"}, "confidence": 0.88},
            {"step": "click", "target": "Save button", "selector": None,
             "parameters": {}, "confidence": 0.95},
        ],
        "variables": [],
        "confidence_avg": 0.92,
        "episode_count": 5,
        "apps_involved": ["Chrome", "Slack"],
    }
    return formatter.format_sop(template)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNewSOPWrittenDirectly:
    """No existing file -> direct write."""

    def test_new_sop_written(self, tmp_path: Path):
        formatter = SOPFormatter()
        sops_dir = tmp_path / "sops"
        versioner = SOPVersioner(sops_dir)

        content = _make_sop_content(formatter)
        result = versioner.write_sop("test_sop", content, formatter)

        assert result.exists()
        assert result.name == "sop.test_sop.md"
        assert result.read_text(encoding="utf-8") == content

    def test_sops_dir_created(self, tmp_path: Path):
        formatter = SOPFormatter()
        sops_dir = tmp_path / "new_dir" / "sops"
        versioner = SOPVersioner(sops_dir)

        content = _make_sop_content(formatter)
        versioner.write_sop("test_sop", content, formatter)

        assert sops_dir.exists()


class TestExistingNotEditedArchived:
    """Old version archived, new canonical written."""

    def test_archives_and_writes_new(self, tmp_path: Path):
        formatter = SOPFormatter()
        sops_dir = tmp_path / "sops"
        versioner = SOPVersioner(sops_dir)

        # Write initial version
        content_v1 = _make_sop_content(formatter)
        versioner.write_sop("test_sop", content_v1, formatter)

        # Write updated version
        content_v2 = _make_sop_content_v2(formatter)
        result = versioner.write_sop("test_sop", content_v2, formatter)

        # New canonical should have v2 content
        assert result.name == "sop.test_sop.md"
        assert result.read_text(encoding="utf-8") == content_v2

        # Archive should have v1
        archive_files = list((sops_dir / "archive").iterdir())
        assert len(archive_files) == 1
        assert archive_files[0].read_text(encoding="utf-8") == content_v1


class TestExistingEditedCreatesDraft:
    """Manual edit detected -> v2_draft created."""

    def test_creates_draft(self, tmp_path: Path):
        formatter = SOPFormatter()
        sops_dir = tmp_path / "sops"
        versioner = SOPVersioner(sops_dir)

        # Write initial version
        content_v1 = _make_sop_content(formatter)
        versioner.write_sop("test_sop", content_v1, formatter)

        # Simulate manual edit (modify body, breaking hash)
        canonical = versioner.get_canonical_path("test_sop")
        modified = canonical.read_text(encoding="utf-8")
        modified += "\n\n<!-- Human review: approved -->\n"
        canonical.write_text(modified, encoding="utf-8")

        # Write v2 — should become draft
        content_v2 = _make_sop_content_v2(formatter)
        result = versioner.write_sop("test_sop", content_v2, formatter)

        assert result.name == "sop.test_sop.v2_draft.md"
        assert result.read_text(encoding="utf-8") == content_v2

        # Original should still have the manual edit
        original_content = canonical.read_text(encoding="utf-8")
        assert "Human review" in original_content


class TestArchivePathFormat:
    """Archive files use timestamp.hash format."""

    def test_archive_naming(self, tmp_path: Path):
        formatter = SOPFormatter()
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir(parents=True)
        versioner = SOPVersioner(sops_dir)

        # Create a file to archive
        filepath = sops_dir / "sop.test_sop.md"
        content = "---\nsop_version: 1\n---\n\n# Test\n"
        filepath.write_text(content, encoding="utf-8")

        archived = versioner.archive_sop(filepath)

        # Check format: sop.test_sop.<timestamp>.<hash>.md
        name = archived.name
        assert name.startswith("sop.test_sop.")
        assert name.endswith(".md")

        # Extract parts: sop.test_sop.YYYYMMDDTHHMMSS.hash8.md
        parts = name.split(".")
        # sop, test_sop, timestamp, hash, md
        assert len(parts) == 5
        # Timestamp part should be like 20260216T123456
        assert re.match(r"\d{8}T\d{6}", parts[2])
        # Hash part should be 8 hex chars
        assert re.match(r"[a-f0-9]{8}", parts[3])

    def test_archived_file_has_original_content(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir(parents=True)
        versioner = SOPVersioner(sops_dir)

        filepath = sops_dir / "sop.test_sop.md"
        content = "Original content here"
        filepath.write_text(content, encoding="utf-8")

        archived = versioner.archive_sop(filepath)
        assert archived.read_text(encoding="utf-8") == content
        assert not filepath.exists()  # Original moved


class TestListVersions:
    """Multiple archived versions found."""

    def test_lists_versions(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        archive_dir = sops_dir / "archive"
        archive_dir.mkdir(parents=True)
        versioner = SOPVersioner(sops_dir)

        # Create fake archive files
        (archive_dir / "sop.test_sop.20260101T000000.aabbccdd.md").write_text("v1")
        (archive_dir / "sop.test_sop.20260102T000000.eeff0011.md").write_text("v2")
        (archive_dir / "sop.other.20260101T000000.00112233.md").write_text("other")

        versions = versioner.list_versions("test_sop")

        assert len(versions) == 2
        # Should be sorted chronologically
        assert "20260101" in versions[0].name
        assert "20260102" in versions[1].name

    def test_empty_archive(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        versioner = SOPVersioner(sops_dir)

        versions = versioner.list_versions("test_sop")
        assert versions == []


class TestCanonicalPath:
    """Correct path construction."""

    def test_canonical_path(self, tmp_path: Path):
        versioner = SOPVersioner(tmp_path / "sops")
        path = versioner.get_canonical_path("my_workflow")
        assert path == tmp_path / "sops" / "sop.my_workflow.md"


class TestDraftPath:
    """Correct draft path construction."""

    def test_draft_path(self, tmp_path: Path):
        versioner = SOPVersioner(tmp_path / "sops")
        path = versioner.get_draft_path("my_workflow")
        assert path == tmp_path / "sops" / "sop.my_workflow.v2_draft.md"

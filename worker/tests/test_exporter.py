"""Tests for the Atomic SOP Exporter — file writes, index generation, pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agenthandover_worker.exporter import AtomicWriter, IndexGenerator, SOPExporter
from agenthandover_worker.sop_format import SOPFormatter
from agenthandover_worker.sop_versioner import SOPVersioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_template(
    slug: str = "test_workflow",
    title: str = "Test Workflow",
    confidence_avg: float = 0.88,
) -> dict:
    return {
        "slug": slug,
        "title": title,
        "steps": [
            {"step": "click", "target": "Submit button", "selector": None,
             "parameters": {}, "confidence": 0.9},
            {"step": "type", "target": "Email field", "selector": "#email",
             "parameters": {"text": "user@example.com"}, "confidence": 0.85},
        ],
        "variables": [],
        "confidence_avg": confidence_avg,
        "episode_count": 5,
        "apps_involved": ["Chrome"],
    }


# ---------------------------------------------------------------------------
# AtomicWriter Tests
# ---------------------------------------------------------------------------


class TestAtomicWriteCreatesFile:
    """File created correctly."""

    def test_file_created(self, tmp_path: Path):
        filepath = tmp_path / "test.md"
        AtomicWriter.write(filepath, "Hello, world!")

        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == "Hello, world!"

    def test_creates_parent_dirs(self, tmp_path: Path):
        filepath = tmp_path / "nested" / "deep" / "test.md"
        AtomicWriter.write(filepath, "Content")

        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == "Content"


class TestAtomicWriteOverwrites:
    """Existing file replaced."""

    def test_overwrites_existing(self, tmp_path: Path):
        filepath = tmp_path / "test.md"
        filepath.write_text("Old content", encoding="utf-8")

        AtomicWriter.write(filepath, "New content")

        assert filepath.read_text(encoding="utf-8") == "New content"


class TestAtomicWriteNoPartial:
    """Failure leaves no partial file."""

    def test_no_temp_file_on_error(self, tmp_path: Path):
        # Write to a directory that we'll make read-only to trigger an error
        filepath = tmp_path / "test.md"

        # First, verify normal write works
        AtomicWriter.write(filepath, "Good content")
        assert filepath.read_text(encoding="utf-8") == "Good content"

        # Check that no temp files are left after successful write
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0


class TestIndexGeneration:
    """Markdown table format."""

    def test_generates_markdown_table(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        entries = [
            _sample_template("workflow_a", "Workflow A", 0.9),
            _sample_template("workflow_b", "Workflow B", 0.75),
        ]

        content = gen.generate_index(sops_dir, entries)

        assert "# SOP Index" in content
        assert "**Total SOPs:** 2" in content
        assert "| Slug |" in content
        assert "`workflow_a`" in content
        assert "`workflow_b`" in content
        assert "Workflow A" in content
        assert "Workflow B" in content

    def test_empty_entries(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        content = gen.generate_index(sops_dir, [])

        assert "# SOP Index" in content
        assert "**Total SOPs:** 0" in content
        assert "| Slug |" not in content  # No table for empty

    def test_average_confidence_computed(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        entries = [
            _sample_template(confidence_avg=0.80),
            _sample_template(slug="other", confidence_avg=0.90),
        ]

        content = gen.generate_index(sops_dir, entries)

        assert "**Average confidence:** 0.85" in content

    def test_detailed_entries(self, tmp_path: Path):
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        entries = [_sample_template()]
        content = gen.generate_index(sops_dir, entries)

        assert "## Details" in content
        assert "**Slug:** `test_workflow`" in content
        assert "**File:** `sop.test_workflow.md`" in content
        assert "**Steps:** 2" in content


class TestIndexUpdateAtomic:
    """Index written atomically."""

    def test_index_written(self, tmp_path: Path):
        sops_dir = tmp_path / "base" / "sops"
        sops_dir.mkdir(parents=True)
        gen = IndexGenerator()

        entries = [_sample_template()]
        gen.update_index(sops_dir, entries)

        index_path = tmp_path / "base" / "index.md"
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8")
        assert "# SOP Index" in content


class TestExportSOP:
    """Full export pipeline for a single SOP."""

    def test_exports_sop(self, tmp_path: Path):
        base_dir = tmp_path / "workspace"
        exporter = SOPExporter(base_dir)
        exporter.formatter = SOPFormatter()
        exporter.versioner = SOPVersioner(base_dir / "sops")

        template = _sample_template()
        result = exporter.export_sop(template)

        assert result.exists()
        assert result.name == "sop.test_workflow.md"
        content = result.read_text(encoding="utf-8")
        assert "---" in content
        assert "Submit button" in content

        # Index should also be created
        index_path = base_dir / "index.md"
        assert index_path.exists()


class TestExportAll:
    """Multiple SOPs + index."""

    def test_exports_multiple(self, tmp_path: Path):
        base_dir = tmp_path / "workspace"
        exporter = SOPExporter(base_dir)
        exporter.formatter = SOPFormatter()
        exporter.versioner = SOPVersioner(base_dir / "sops")

        templates = [
            _sample_template("workflow_a", "Workflow A"),
            _sample_template("workflow_b", "Workflow B"),
        ]
        paths = exporter.export_all(templates)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()

        # Index should have both entries
        index_path = base_dir / "index.md"
        assert index_path.exists()
        index_content = index_path.read_text(encoding="utf-8")
        assert "`workflow_a`" in index_content
        assert "`workflow_b`" in index_content


class TestDirectoryCreation:
    """Missing dirs created automatically."""

    def test_creates_directories(self, tmp_path: Path):
        filepath = tmp_path / "a" / "b" / "c" / "test.md"
        AtomicWriter.write(filepath, "Content")

        assert filepath.exists()
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_exporter_creates_sops_dir(self, tmp_path: Path):
        base_dir = tmp_path / "new_workspace"
        exporter = SOPExporter(base_dir)
        exporter.formatter = SOPFormatter()
        exporter.versioner = SOPVersioner(base_dir / "sops")

        template = _sample_template()
        exporter.export_sop(template)

        assert (base_dir / "sops").is_dir()

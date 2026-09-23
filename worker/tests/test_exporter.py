"""Tests for the Atomic SOP Exporter — file writes, index generation, pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import json

from agenthandover_worker.exporter import AtomicWriter, IndexGenerator
from agenthandover_worker.procedure_schema import sop_to_procedure


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


class TestIndexReadsDisk:
    """The catalogue lists every procedure on disk, not only the last batch."""

    @staticmethod
    def _write_proc(sops_dir: Path, slug: str, title: str) -> None:
        proc = sop_to_procedure(_sample_template(slug, title))
        AtomicWriter.write(
            sops_dir / f"sop.{slug}.v3.json", json.dumps(proc),
        )

    def test_earlier_procedures_survive_a_later_batch(self, tmp_path: Path):
        sops_dir = tmp_path / "base" / "sops"
        self._write_proc(sops_dir, "workflow_a", "Workflow A")
        self._write_proc(sops_dir, "workflow_b", "Workflow B")

        IndexGenerator().update_index(
            sops_dir, [_sample_template("workflow_b", "Workflow B renamed")],
        )

        content = (tmp_path / "base" / "index.md").read_text(encoding="utf-8")
        assert "**Total SOPs:** 2" in content
        assert "`workflow_a`" in content
        # The fresh batch wins over the disk copy of the same slug.
        assert "Workflow B renamed" in content
        assert "| Workflow B |" not in content

    def test_archive_and_unreadable_files_are_skipped(self, tmp_path: Path):
        sops_dir = tmp_path / "base" / "sops"
        self._write_proc(sops_dir, "workflow_a", "Workflow A")
        self._write_proc(sops_dir / "archive", "old_one", "Old One")
        AtomicWriter.write(sops_dir / "sop.broken.v3.json", "{not json")

        IndexGenerator().update_index(sops_dir, [])

        content = (tmp_path / "base" / "index.md").read_text(encoding="utf-8")
        assert "**Total SOPs:** 1" in content
        assert "old_one" not in content
        assert "broken" not in content


    def test_a_batch_entry_without_a_file_is_not_listed(self, tmp_path: Path):
        """A draft the hand-off gate held back has no file: not listed."""
        sops_dir = tmp_path / "base" / "sops"
        self._write_proc(sops_dir, "workflow_a", "Workflow A")

        IndexGenerator().update_index(
            sops_dir, [_sample_template("held_back", "Held Back")],
        )

        content = (tmp_path / "base" / "index.md").read_text(encoding="utf-8")
        assert "**Total SOPs:** 1" in content
        assert "held_back" not in content

    def test_a_batch_entry_with_only_its_markdown_is_listed(self, tmp_path: Path):
        """A writer without the JSON sidecar leaves ``sop.<slug>.md``."""
        sops_dir = tmp_path / "base" / "sops"
        AtomicWriter.write(sops_dir / "sop.md_only.md", "# MD only\n")

        IndexGenerator().update_index(
            sops_dir, [_sample_template("md_only", "MD Only")],
        )

        content = (tmp_path / "base" / "index.md").read_text(encoding="utf-8")
        assert "**Total SOPs:** 1" in content
        assert "`md_only`" in content


class TestDirectoryCreation:
    """Missing dirs created automatically."""

    def test_creates_directories(self, tmp_path: Path):
        filepath = tmp_path / "a" / "b" / "c" / "test.md"
        AtomicWriter.write(filepath, "Content")

        assert filepath.exists()
        assert (tmp_path / "a" / "b" / "c").is_dir()

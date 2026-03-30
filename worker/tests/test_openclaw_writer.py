"""Tests for the OpenClaw Integration Writer — workspace directory management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenthandover_worker.openclaw_writer import OpenClawWriter


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
# Tests
# ---------------------------------------------------------------------------


class TestEnsureDirectoryStructure:
    """All required directories created."""

    def test_creates_all_dirs(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        writer.ensure_directory_structure()

        assert writer.sops_dir.exists()
        assert writer.metadata_dir.exists()
        assert (writer.sops_dir / "archive").exists()

    def test_idempotent(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        writer.ensure_directory_structure()
        writer.ensure_directory_structure()  # Should not raise

        assert writer.sops_dir.exists()


class TestWriteSOP:
    """SOP written to correct path."""

    def test_writes_sop_file(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        template = _sample_template()

        result = writer.write_sop(template)

        assert result.exists()
        assert result.name == "sop.test_workflow.md"
        content = result.read_text(encoding="utf-8")
        assert "---" in content
        assert "Submit button" in content

    def test_sop_in_correct_directory(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        template = _sample_template()

        result = writer.write_sop(template)

        expected_parent = (
            tmp_path / "workspace" / "memory" / "apprentice" / "sops"
        )
        assert result.parent == expected_parent


class TestWriteAllSOPs:
    """Multiple SOPs + index written."""

    def test_writes_multiple_sops(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")

        templates = [
            _sample_template("workflow_a", "Workflow A"),
            _sample_template("workflow_b", "Workflow B"),
        ]
        paths = writer.write_all_sops(templates)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()

        # Index should exist
        index_path = (
            tmp_path / "workspace" / "memory" / "apprentice" / "index.md"
        )
        assert index_path.exists()
        index_content = index_path.read_text(encoding="utf-8")
        assert "`workflow_a`" in index_content
        assert "`workflow_b`" in index_content


class TestWriteMetadata:
    """Metadata file written."""

    def test_writes_metadata_json(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")

        data = {
            "total_episodes": 42,
            "patterns_found": 7,
            "avg_confidence": 0.87,
        }
        result = writer.write_metadata("confidence_log", data)

        assert result.exists()
        assert result.name == "confidence_log.json"

        content = json.loads(result.read_text(encoding="utf-8"))
        assert content["metadata_type"] == "confidence_log"
        assert content["total_episodes"] == 42
        assert "generated_at" in content

    def test_metadata_in_correct_directory(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")

        result = writer.write_metadata("episode_stats", {"count": 10})

        expected_parent = (
            tmp_path / "workspace" / "memory" / "apprentice" / "metadata"
        )
        assert result.parent == expected_parent


class TestCustomWorkspace:
    """Non-default workspace path."""

    def test_custom_path(self, tmp_path: Path):
        custom = tmp_path / "custom" / "workspace"
        writer = OpenClawWriter(workspace_dir=custom)

        assert writer.workspace == custom
        assert writer.sops_dir == custom / "memory" / "apprentice" / "sops"

        template = _sample_template()
        result = writer.write_sop(template)

        assert result.exists()
        assert str(custom) in str(result)


class TestLearningOnlyPaths:
    """Files only in apprentice/ subtree."""

    def test_all_files_under_apprentice(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")

        # Write a SOP and metadata
        writer.write_sop(_sample_template())
        writer.write_metadata("test_meta", {"key": "value"})

        # Recursively find all files
        apprentice = tmp_path / "workspace" / "memory" / "apprentice"
        all_files = list(apprentice.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]

        assert len(all_files) >= 2  # At least SOP + metadata

        # Every file must be under the apprentice directory
        for f in all_files:
            assert str(f).startswith(str(apprentice)), (
                f"File {f} is outside apprentice subtree"
            )

    def test_no_files_outside_memory(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        writer.write_sop(_sample_template())

        # Check that no files exist directly in workspace (only under memory/)
        workspace = tmp_path / "workspace"
        direct_files = [
            f for f in workspace.iterdir()
            if f.is_file()
        ]
        assert len(direct_files) == 0


class TestArchiveSubdirectory:
    """Archive dir created within sops/."""

    def test_archive_dir_exists(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        writer.ensure_directory_structure()

        archive = writer.sops_dir / "archive"
        assert archive.exists()
        assert archive.is_dir()

    def test_archive_under_sops(self, tmp_path: Path):
        writer = OpenClawWriter(workspace_dir=tmp_path / "workspace")
        writer.ensure_directory_structure()

        # Verify the path structure
        expected = (
            tmp_path
            / "workspace"
            / "memory"
            / "apprentice"
            / "sops"
            / "archive"
        )
        assert expected.exists()

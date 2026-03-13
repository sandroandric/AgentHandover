"""Tests for the KBSync module."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.kb_sync import (
    KBSync,
    SyncDiff,
    SyncManifest,
    SyncResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def remote_kb(tmp_path: Path) -> KnowledgeBase:
    """Create a separate KB for remote simulation."""
    kb = KnowledgeBase(root=tmp_path / "remote_knowledge")
    kb.ensure_structure()
    return kb


def _write_file(kb: KnowledgeBase, rel_path: str, content: str) -> Path:
    """Helper: write a file at a relative path under the KB root."""
    path = kb.root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Tests: build_manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_empty_kb_manifest(self, kb: KnowledgeBase) -> None:
        sync = KBSync(kb)
        manifest = sync.build_manifest()
        assert isinstance(manifest.files, dict)
        assert manifest.machine_id == "local"
        assert manifest.generated_at is not None

    def test_manifest_includes_files(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        manifest = sync.build_manifest()
        assert "procedures/deploy-api.json" in manifest.files

    def test_manifest_checksums_are_sha256(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "test", "steps": []})
        sync = KBSync(kb)
        manifest = sync.build_manifest()
        for rel, info in manifest.files.items():
            assert "checksum" in info
            assert len(info["checksum"]) == 64  # SHA-256 hex length
            assert "size" in info
            assert "modified" in info

    def test_manifest_custom_machine_id(self, kb: KnowledgeBase) -> None:
        sync = KBSync(kb, machine_id="my-laptop")
        manifest = sync.build_manifest()
        assert manifest.machine_id == "my-laptop"

    def test_manifest_skips_temp_files(self, kb: KnowledgeBase) -> None:
        # Write a temp file that should be skipped
        temp_file = kb.root / ".kb_something.tmp"
        temp_file.write_text("temp data")
        sync = KBSync(kb)
        manifest = sync.build_manifest()
        assert ".kb_something.tmp" not in manifest.files


# ---------------------------------------------------------------------------
# Tests: compute_diff
# ---------------------------------------------------------------------------


class TestComputeDiff:
    def test_diff_with_identical_kbs(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        local_manifest = sync.build_manifest()
        # Same manifest = no diff
        diff = sync.compute_diff(local_manifest)
        assert diff.added == []
        assert diff.modified == []
        assert diff.deleted == []
        assert len(diff.unchanged) > 0

    def test_diff_with_added_file(self, kb: KnowledgeBase) -> None:
        sync = KBSync(kb)
        # Remote has a file we don't have
        remote_manifest = SyncManifest(
            files={
                "procedures/new-proc.json": {
                    "checksum": "abc123",
                    "size": 100,
                    "modified": "2025-01-01T00:00:00+00:00",
                },
            },
            generated_at="2025-01-01T00:00:00Z",
            machine_id="remote",
        )
        diff = sync.compute_diff(remote_manifest)
        assert "procedures/new-proc.json" in diff.added

    def test_diff_with_deleted_file(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        # Remote is empty
        remote_manifest = SyncManifest(
            files={},
            generated_at="2025-01-01T00:00:00Z",
            machine_id="remote",
        )
        diff = sync.compute_diff(remote_manifest)
        assert "procedures/deploy-api.json" in diff.deleted

    def test_diff_with_modified_file(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        # Remote has the same file but different checksum
        remote_manifest = SyncManifest(
            files={
                "procedures/deploy-api.json": {
                    "checksum": "different_checksum_here",
                    "size": 200,
                    "modified": "2025-01-01T00:00:00+00:00",
                },
            },
            generated_at="2025-01-01T00:00:00Z",
            machine_id="remote",
        )
        diff = sync.compute_diff(remote_manifest)
        assert "procedures/deploy-api.json" in diff.modified

    def test_diff_unchanged_files(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        local_manifest = sync.build_manifest()
        diff = sync.compute_diff(local_manifest)
        assert "procedures/deploy-api.json" in diff.unchanged


# ---------------------------------------------------------------------------
# Tests: export_bundle
# ---------------------------------------------------------------------------


class TestExportBundle:
    def test_export_creates_zip(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        output = tmp_path / "export.zip"
        result = sync.export_bundle(output)
        assert result.exists()
        assert result.suffix == ".zip"

    def test_export_contains_files(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "deploy-api", "steps": []})
        sync = KBSync(kb)
        output = tmp_path / "export.zip"
        result = sync.export_bundle(output)
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert "procedures/deploy-api.json" in names
            assert "_manifest.json" in names

    def test_export_manifest_valid(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "test", "steps": []})
        sync = KBSync(kb)
        output = tmp_path / "export.zip"
        result = sync.export_bundle(output)
        with zipfile.ZipFile(result, "r") as zf:
            manifest_data = json.loads(zf.read("_manifest.json"))
            assert "files" in manifest_data
            assert "generated_at" in manifest_data

    def test_export_to_directory_generates_name(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "test", "steps": []})
        sync = KBSync(kb)
        out_dir = tmp_path / "exports"
        out_dir.mkdir()
        result = sync.export_bundle(out_dir)
        assert result.suffix == ".zip"
        assert result.exists()

    def test_export_adds_zip_extension(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        sync = KBSync(kb)
        output = tmp_path / "bundle"
        result = sync.export_bundle(output)
        assert result.suffix == ".zip"


# ---------------------------------------------------------------------------
# Tests: import_bundle
# ---------------------------------------------------------------------------


class TestImportBundle:
    def test_import_merge(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        # Create an export from a different KB
        other_kb = KnowledgeBase(root=tmp_path / "other")
        other_kb.ensure_structure()
        other_kb.save_procedure({"id": "imported-proc", "steps": ["step1"]})
        other_sync = KBSync(other_kb)
        bundle = other_sync.export_bundle(tmp_path / "bundle.zip")

        # Import into our KB
        sync = KBSync(kb)
        result = sync.import_bundle(bundle, strategy="merge")
        assert result.status == "success"
        assert result.files_pulled > 0

        # Verify the procedure was imported
        proc = kb.get_procedure("imported-proc")
        assert proc is not None

    def test_import_replace(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        # Pre-populate local KB
        kb.save_procedure({"id": "local-proc", "steps": []})

        # Create bundle from other KB
        other_kb = KnowledgeBase(root=tmp_path / "other")
        other_kb.ensure_structure()
        other_kb.save_procedure({"id": "remote-proc", "steps": ["step1"]})
        other_sync = KBSync(other_kb)
        bundle = other_sync.export_bundle(tmp_path / "bundle.zip")

        # Import with replace
        sync = KBSync(kb)
        result = sync.import_bundle(bundle, strategy="replace")
        assert result.status == "success"

        # Remote proc should exist
        assert kb.get_procedure("remote-proc") is not None
        # Local-only proc should be gone (files were deleted before import)
        assert kb.get_procedure("local-proc") is None

    def test_import_nonexistent_bundle(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        sync = KBSync(kb)
        result = sync.import_bundle(tmp_path / "nonexistent.zip")
        assert result.status == "error"
        assert len(result.errors) > 0

    def test_import_saves_sync_state(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        other_kb = KnowledgeBase(root=tmp_path / "other")
        other_kb.ensure_structure()
        other_sync = KBSync(other_kb)
        bundle = other_sync.export_bundle(tmp_path / "bundle.zip")

        sync = KBSync(kb)
        sync.import_bundle(bundle)

        state_path = kb.root / ".sync_state.json"
        assert state_path.is_file()
        state = json.loads(state_path.read_text())
        assert state["last_operation"] == "import"


# ---------------------------------------------------------------------------
# Tests: sync_to_directory
# ---------------------------------------------------------------------------


class TestSyncToDirectory:
    def test_push_local_to_remote(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "local-proc", "steps": []})
        remote_dir = tmp_path / "remote"
        sync = KBSync(kb)
        result = sync.sync_to_directory(remote_dir, strategy="push")
        assert result.files_pushed > 0

        # Verify file exists in remote
        assert (remote_dir / "procedures" / "local-proc.json").is_file()

    def test_pull_remote_to_local(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        remote_dir = tmp_path / "remote"
        remote_kb = KnowledgeBase(root=remote_dir)
        remote_kb.ensure_structure()
        remote_kb.save_procedure({"id": "remote-proc", "steps": []})

        sync = KBSync(kb)
        result = sync.sync_to_directory(remote_dir, strategy="pull")
        assert result.files_pulled > 0

        # Verify file exists locally
        assert kb.get_procedure("remote-proc") is not None

    def test_merge_bidirectional(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "local-only", "steps": []})

        remote_dir = tmp_path / "remote"
        remote_kb = KnowledgeBase(root=remote_dir)
        remote_kb.ensure_structure()
        remote_kb.save_procedure({"id": "remote-only", "steps": []})

        sync = KBSync(kb)
        result = sync.sync_to_directory(remote_dir, strategy="merge")

        # Local-only should have been pushed
        assert (remote_dir / "procedures" / "local-only.json").is_file()
        # Remote-only should have been pulled
        assert kb.get_procedure("remote-only") is not None

    def test_sync_creates_remote_dir(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        remote_dir = tmp_path / "new_remote"
        assert not remote_dir.exists()
        sync = KBSync(kb)
        result = sync.sync_to_directory(remote_dir)
        assert remote_dir.exists()
        assert result.status == "success"

    def test_sync_saves_sync_state(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        remote_dir = tmp_path / "remote"
        sync = KBSync(kb)
        sync.sync_to_directory(remote_dir)
        state_path = kb.root / ".sync_state.json"
        assert state_path.is_file()


# ---------------------------------------------------------------------------
# Tests: file checksum
# ---------------------------------------------------------------------------


class TestFileChecksum:
    def test_checksum_consistency(self, kb: KnowledgeBase) -> None:
        _write_file(kb, "test.txt", "hello world")
        path = kb.root / "test.txt"
        sync = KBSync(kb)
        c1 = sync._file_checksum(path)
        c2 = sync._file_checksum(path)
        assert c1 == c2

    def test_different_content_different_checksum(self, kb: KnowledgeBase) -> None:
        p1 = _write_file(kb, "a.txt", "content A")
        p2 = _write_file(kb, "b.txt", "content B")
        sync = KBSync(kb)
        assert sync._file_checksum(p1) != sync._file_checksum(p2)

    def test_checksum_is_64_hex_chars(self, kb: KnowledgeBase) -> None:
        p = _write_file(kb, "c.txt", "test")
        sync = KBSync(kb)
        checksum = sync._file_checksum(p)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)


# ---------------------------------------------------------------------------
# Tests: round-trip export/import
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_export_import_preserves_procedures(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_procedure({"id": "proc-a", "steps": ["step1", "step2"]})
        kb.save_procedure({"id": "proc-b", "steps": ["step3"]})

        sync = KBSync(kb)
        bundle = sync.export_bundle(tmp_path / "roundtrip.zip")

        # Import into a fresh KB
        new_kb = KnowledgeBase(root=tmp_path / "fresh")
        new_kb.ensure_structure()
        new_sync = KBSync(new_kb)
        result = new_sync.import_bundle(bundle)
        assert result.status == "success"

        assert new_kb.get_procedure("proc-a") is not None
        assert new_kb.get_procedure("proc-b") is not None

    def test_export_import_preserves_daily_summaries(
        self, kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        kb.save_daily_summary("2025-03-01", {"tasks": [{"intent": "test"}]})

        sync = KBSync(kb)
        bundle = sync.export_bundle(tmp_path / "daily.zip")

        new_kb = KnowledgeBase(root=tmp_path / "fresh")
        new_kb.ensure_structure()
        new_sync = KBSync(new_kb)
        new_sync.import_bundle(bundle)

        summary = new_kb.get_daily_summary("2025-03-01")
        assert summary is not None
        assert summary["tasks"] == [{"intent": "test"}]

    def test_manifest_matches_after_import(
        self, kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        kb.save_procedure({"id": "test", "steps": []})

        sync = KBSync(kb)
        bundle = sync.export_bundle(tmp_path / "manifest.zip")

        new_kb = KnowledgeBase(root=tmp_path / "fresh")
        new_kb.ensure_structure()
        new_sync = KBSync(new_kb)
        new_sync.import_bundle(bundle)

        # Build manifests and compare file sets
        original_manifest = sync.build_manifest()
        new_manifest = new_sync.build_manifest()

        # The fresh KB may have extra dirs from ensure_structure,
        # but all original files should be present
        for rel in original_manifest.files:
            assert rel in new_manifest.files


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_kb_export_import(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        sync = KBSync(kb)
        bundle = sync.export_bundle(tmp_path / "empty.zip")
        assert bundle.exists()

        new_kb = KnowledgeBase(root=tmp_path / "fresh")
        new_kb.ensure_structure()
        new_sync = KBSync(new_kb)
        result = new_sync.import_bundle(bundle)
        assert result.status == "success"

    def test_sync_empty_to_empty(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        remote_dir = tmp_path / "remote"
        sync = KBSync(kb)
        result = sync.sync_to_directory(remote_dir)
        assert result.status == "success"
        assert result.files_pushed == 0
        assert result.files_pulled == 0

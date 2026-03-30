"""Tests for the SOP export adapter system."""

import json
import pytest
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.generic_writer import GenericWriter
from agenthandover_worker.sop_schema import sop_to_json, validate_sop_json, SOP_SCHEMA_VERSION


class TestSOPExportAdapterInterface:
    """Verify that all adapters implement the ABC correctly."""

    def test_openclaw_writer_is_adapter(self):
        assert issubclass(OpenClawWriter, SOPExportAdapter)

    def test_generic_writer_is_adapter(self):
        assert issubclass(GenericWriter, SOPExportAdapter)

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SOPExportAdapter()


class TestGenericWriter:
    """Test the generic filesystem adapter."""

    def test_write_sop_creates_md_and_json(self, tmp_path):
        writer = GenericWriter(output_dir=tmp_path, json_export=True)
        sop = {
            "slug": "test-login",
            "title": "Test Login",
            "steps": [{"step": "click", "target": "button"}],
            "confidence_avg": 0.85,
            "episode_count": 3,
            "apps_involved": ["Chrome"],
        }
        path = writer.write_sop(sop)
        assert path.exists()
        assert path.name == "sop.test-login.md"

        json_path = tmp_path / "sops" / "sop.test-login.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["schema_version"] == SOP_SCHEMA_VERSION
        assert data["slug"] == "test-login"

    def test_write_sop_without_json(self, tmp_path):
        writer = GenericWriter(output_dir=tmp_path, json_export=False)
        sop = {"slug": "no-json", "title": "No JSON", "steps": []}
        writer.write_sop(sop)
        assert not (tmp_path / "sops" / "sop.no-json.json").exists()

    def test_list_sops(self, tmp_path):
        writer = GenericWriter(output_dir=tmp_path)
        writer.write_sop({"slug": "a-first", "title": "First SOP", "steps": []})
        writer.write_sop({"slug": "b-second", "title": "Second SOP", "steps": []})
        sops = writer.list_sops()
        assert len(sops) == 2
        assert sops[0]["slug"] == "a-first"
        assert sops[1]["slug"] == "b-second"

    def test_get_sops_dir(self, tmp_path):
        writer = GenericWriter(output_dir=tmp_path)
        assert writer.get_sops_dir() == tmp_path / "sops"

    def test_write_metadata(self, tmp_path):
        writer = GenericWriter(output_dir=tmp_path)
        path = writer.write_metadata("test_report", {"total": 42})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total"] == 42
        assert data["metadata_type"] == "test_report"


class TestSOPSchema:
    """Test the JSON schema module."""

    def test_sop_to_json_basic(self):
        sop = {
            "slug": "my-sop",
            "title": "My SOP",
            "steps": [{"step": "click", "target": "btn", "confidence": 0.9}],
            "confidence_avg": 0.9,
            "episode_count": 5,
            "apps_involved": ["Chrome", "Slack"],
            "variables": [{"name": "url", "type": "string", "description": "Target URL"}],
        }
        result = sop_to_json(sop)
        assert result["schema_version"] == SOP_SCHEMA_VERSION
        assert result["slug"] == "my-sop"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["index"] == 0
        assert len(result["variables"]) == 1

    def test_validate_valid(self):
        data = sop_to_json({"slug": "test", "title": "Test", "steps": []})
        errors = validate_sop_json(data)
        assert errors == []

    def test_validate_missing_fields(self):
        errors = validate_sop_json({})
        assert len(errors) >= 3  # slug, title, steps at minimum

    def test_validate_wrong_version(self):
        data = sop_to_json({"slug": "x", "title": "X", "steps": []})
        data["schema_version"] = "99.0.0"
        errors = validate_sop_json(data)
        assert any("Unsupported schema" in e for e in errors)

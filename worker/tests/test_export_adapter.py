"""Tests for the SOP export adapter system."""

import json
import pytest
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.generic_writer import GenericWriter
from agenthandover_worker.procedure_schema import PROCEDURE_SCHEMA_VERSION


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

        json_path = tmp_path / "sops" / "sop.test-login.v3.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["schema_version"] == PROCEDURE_SCHEMA_VERSION
        assert data["id"] == "test-login"

    def test_write_sop_without_json(self, tmp_path):
        writer = GenericWriter(output_dir=tmp_path, json_export=False)
        sop = {"slug": "no-json", "title": "No JSON", "steps": []}
        writer.write_sop(sop)
        assert not list((tmp_path / "sops").glob("sop.no-json*.json"))

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

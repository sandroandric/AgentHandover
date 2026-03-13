"""Tests for the knowledge base export adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.knowledge_export_adapter import KnowledgeBaseExportAdapter


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def adapter(kb: KnowledgeBase) -> KnowledgeBaseExportAdapter:
    return KnowledgeBaseExportAdapter(kb)


@pytest.fixture()
def sample_sop() -> dict:
    return {
        "slug": "test-sop",
        "title": "Test SOP",
        "steps": [{"action": "Do thing", "confidence": 0.9}],
        "confidence_avg": 0.85,
        "apps_involved": ["Chrome"],
        "tags": ["testing"],
    }


class TestWriteSOP:

    def test_write_sop_creates_file(
        self, adapter: KnowledgeBaseExportAdapter, sample_sop: dict
    ) -> None:
        path = adapter.write_sop(sample_sop)
        assert path.is_file()
        assert "test-sop" in path.name

    def test_write_sop_stores_v3(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        adapter.write_sop(sample_sop)
        proc = kb.get_procedure("test-sop")
        assert proc is not None
        assert proc["schema_version"] == "3.0.0"


class TestWriteAllSOPs:

    def test_write_multiple(
        self, adapter: KnowledgeBaseExportAdapter
    ) -> None:
        sops = [
            {"slug": f"sop-{i}", "title": f"SOP {i}",
             "steps": [{"action": "step"}]}
            for i in range(3)
        ]
        paths = adapter.write_all_sops(sops)
        assert len(paths) == 3
        for p in paths:
            assert p.is_file()


class TestWriteMetadata:

    def test_write_profile(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase
    ) -> None:
        path = adapter.write_metadata("profile", {"tools": {"editor": "vim"}})
        assert path.name == "profile.json"
        profile = kb.get_profile()
        assert profile["tools"]["editor"] == "vim"

    def test_write_triggers(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase
    ) -> None:
        path = adapter.write_metadata("triggers", {"recurrence": [], "chains": []})
        assert path.name == "triggers.json"

    def test_write_constraints(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase
    ) -> None:
        path = adapter.write_metadata("constraints", {
            "global": {"max_spend": 100},
            "per_procedure": {},
        })
        assert path.name == "constraints.json"

    def test_write_decisions(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase
    ) -> None:
        path = adapter.write_metadata("decisions", {"decision_sets": []})
        assert path.name == "decisions.json"

    def test_write_custom_context(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase
    ) -> None:
        path = adapter.write_metadata("recent", {"last_7_days": []})
        assert "recent.json" in path.name


class TestGetSOPsDir:

    def test_returns_procedures_dir(
        self, adapter: KnowledgeBaseExportAdapter, kb: KnowledgeBase
    ) -> None:
        assert adapter.get_sops_dir() == kb.root / "procedures"


class TestListSOPs:

    def test_empty(self, adapter: KnowledgeBaseExportAdapter) -> None:
        assert adapter.list_sops() == []

    def test_list_after_write(
        self, adapter: KnowledgeBaseExportAdapter, sample_sop: dict
    ) -> None:
        adapter.write_sop(sample_sop)
        sops = adapter.list_sops()
        assert len(sops) == 1
        assert sops[0]["slug"] == "test-sop"
        assert sops[0]["title"] == "Test SOP"
        assert sops[0]["confidence"] == 0.85

    def test_list_multiple(self, adapter: KnowledgeBaseExportAdapter) -> None:
        for i in range(3):
            adapter.write_sop({
                "slug": f"sop-{i}",
                "title": f"SOP {i}",
                "steps": [{"action": "step"}],
                "confidence_avg": 0.8 + i * 0.05,
            })
        sops = adapter.list_sops()
        assert len(sops) == 3

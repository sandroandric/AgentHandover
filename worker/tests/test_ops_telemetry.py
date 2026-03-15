from oc_apprentice_worker.ops_telemetry import OpsTelemetry, PipelineMetrics
from oc_apprentice_worker.knowledge_base import KnowledgeBase
import pytest

@pytest.fixture
def kb(tmp_path):
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb

class TestRecordBatch:
    def test_creates_daily_file(self, kb):
        t = OpsTelemetry(kb)
        t.record_batch(PipelineMetrics(timestamp="2026-03-14T10:00:00Z", annotation_count=5))
        path = kb.root / "observations" / "telemetry" / "2026-03-14.json"
        assert path.exists()

    def test_appends_to_existing(self, kb):
        t = OpsTelemetry(kb)
        t.record_batch(PipelineMetrics(timestamp="2026-03-14T10:00:00Z", annotation_count=5))
        t.record_batch(PipelineMetrics(timestamp="2026-03-14T11:00:00Z", annotation_count=3))
        import json
        with open(kb.root / "observations" / "telemetry" / "2026-03-14.json") as f:
            data = json.load(f)
        assert data["entry_count"] == 2

    def test_auto_timestamp(self, kb):
        t = OpsTelemetry(kb)
        m = PipelineMetrics(annotation_count=1)
        t.record_batch(m)
        assert m.timestamp  # should be set

class TestDailySummary:
    def test_aggregates_entries(self, kb):
        t = OpsTelemetry(kb)
        t.record_batch(PipelineMetrics(timestamp="2026-03-14T10:00:00Z", annotation_count=5))
        t.record_batch(PipelineMetrics(timestamp="2026-03-14T11:00:00Z", annotation_count=3))
        summary = t.get_daily_summary("2026-03-14")
        assert summary["total_annotation_count"] == 8

    def test_missing_date(self, kb):
        summary = OpsTelemetry(kb).get_daily_summary("2099-01-01")
        assert summary["entries"] == 0

class TestTrend:
    def test_returns_7_days(self, kb):
        trend = OpsTelemetry(kb).get_trend(7)
        assert len(trend) == 7

    def test_includes_populated_days(self, kb):
        t = OpsTelemetry(kb)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        t.record_batch(PipelineMetrics(timestamp=f"{today}T10:00:00Z", annotation_count=5))
        trend = t.get_trend(1)
        assert trend[0]["entries"] > 0

class TestHealthSnapshot:
    def test_returns_procedure_counts(self, kb):
        # Save a procedure
        from oc_apprentice_worker.procedure_schema import sop_to_procedure
        proc = sop_to_procedure({"slug": "test", "title": "Test", "steps": [{"step": "Do", "app": "Chrome", "confidence": 0.9}], "confidence_avg": 0.9, "apps_involved": ["Chrome"], "source": "test"})
        kb.save_procedure(proc)
        snapshot = OpsTelemetry(kb).get_health_snapshot()
        assert snapshot["procedures_total"] == 1

    def test_empty_kb(self, kb):
        snapshot = OpsTelemetry(kb).get_health_snapshot()
        assert snapshot["procedures_total"] == 0

class TestEdgeCases:
    def test_corrupted_file_handled(self, kb):
        t = OpsTelemetry(kb)
        path = kb.root / "observations" / "telemetry"
        path.mkdir(parents=True, exist_ok=True)
        (path / "2026-03-14.json").write_text("not json")
        t.record_batch(PipelineMetrics(timestamp="2026-03-14T10:00:00Z", annotation_count=5))
        # Should not crash, overwrites corrupted file

    def test_first_run_no_telemetry_dir(self, kb):
        t = OpsTelemetry(kb)
        t.record_batch(PipelineMetrics(annotation_count=1))
        # Should create dir

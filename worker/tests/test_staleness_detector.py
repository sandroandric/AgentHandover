"""Tests for the staleness detector."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.staleness_detector import StalenessDetector


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def detector(kb: KnowledgeBase) -> StalenessDetector:
    return StalenessDetector(kb)


def _make_procedure(slug: str, days_old: int = 0, **overrides: object) -> dict:
    """Create a procedure with configurable staleness."""
    now = datetime.now(timezone.utc)
    observed = (now - timedelta(days=days_old)).isoformat()
    proc = sop_to_procedure({
        "slug": slug,
        "title": f"Procedure {slug}",
        "steps": [{"action": "Do something"}],
        "confidence_avg": 0.85,
        "source": "passive",
    })
    proc["staleness"]["last_observed"] = observed
    for key, val in overrides.items():
        if key in proc:
            proc[key] = val
        elif "." in key:
            parts = key.split(".")
            d = proc
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
    return proc


# ---------------------------------------------------------------------------
# Individual procedure checks
# ---------------------------------------------------------------------------

class TestCheckProcedure:

    def test_current_procedure(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("current", days_old=5)
        kb.save_procedure(proc)
        report = detector.check_procedure("current")
        assert report.status == "current"
        assert report.recommended_action == "none"

    def test_needs_review_30_days(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("old", days_old=35)
        kb.save_procedure(proc)
        report = detector.check_procedure("old")
        assert report.status == "needs_review"
        assert report.recommended_action == "review"

    def test_stale_60_days(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("stale", days_old=65)
        kb.save_procedure(proc)
        report = detector.check_procedure("stale")
        assert report.status == "stale"
        assert report.recommended_action == "archive"

    def test_not_found(self, detector: StalenessDetector) -> None:
        report = detector.check_procedure("nonexistent")
        assert report.status == "stale"
        assert report.recommended_action == "archive"

    def test_no_last_observed(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("no-date", days_old=0)
        proc["staleness"]["last_observed"] = None
        kb.save_procedure(proc)
        report = detector.check_procedure("no-date")
        assert len(report.signals) > 0

    def test_confidence_drift(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("drift", days_old=5)
        proc["staleness"]["confidence_trend"] = [0.90, 0.85, 0.70]
        kb.save_procedure(proc)
        report = detector.check_procedure("drift")
        assert any(s.type == "confidence_drift" for s in report.signals)
        assert report.status == "needs_review"

    def test_no_confidence_drift_stable(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("stable", days_old=5)
        proc["staleness"]["confidence_trend"] = [0.85, 0.86, 0.87]
        kb.save_procedure(proc)
        report = detector.check_procedure("stable")
        assert not any(s.type == "confidence_drift" for s in report.signals)

    def test_contradictions_signal(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("contradicted", days_old=5)
        proc["evidence"]["contradictions"] = [
            {"step_id": "step_1", "expected": "A", "actual": "B"}
        ]
        kb.save_procedure(proc)
        report = detector.check_procedure("contradicted")
        assert any(s.type == "step_failure" for s in report.signals)

    def test_drift_signals_forwarded(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("drift-sig", days_old=5)
        proc["staleness"]["drift_signals"] = [
            {"type": "url_changed", "detail": "URL moved", "first_seen": "2026-03-01"}
        ]
        kb.save_procedure(proc)
        report = detector.check_procedure("drift-sig")
        assert any(s.type == "url_changed" for s in report.signals)

    def test_confidence_trend_in_report(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("trend", days_old=5)
        proc["staleness"]["confidence_trend"] = [0.85, 0.86, 0.87]
        kb.save_procedure(proc)
        report = detector.check_procedure("trend")
        assert report.confidence_trend == [0.85, 0.86, 0.87]


# ---------------------------------------------------------------------------
# check_all
# ---------------------------------------------------------------------------

class TestCheckAll:

    def test_check_all_empty(self, detector: StalenessDetector) -> None:
        reports = detector.check_all()
        assert reports == []

    def test_check_all_multiple(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        kb.save_procedure(_make_procedure("a", days_old=5))
        kb.save_procedure(_make_procedure("b", days_old=40))
        kb.save_procedure(_make_procedure("c", days_old=70))
        reports = detector.check_all()
        assert len(reports) == 3
        statuses = {r.slug: r.status for r in reports}
        assert statuses["a"] == "current"
        assert statuses["b"] == "needs_review"
        assert statuses["c"] == "stale"

    def test_check_all_staleness_with_contradictions_and_drift(
        self, kb: KnowledgeBase, detector: StalenessDetector
    ) -> None:
        proc = _make_procedure("bad", days_old=35)
        proc["staleness"]["confidence_trend"] = [0.90, 0.80, 0.60]
        proc["evidence"]["contradictions"] = [{"step_id": "step_1"}]
        kb.save_procedure(proc)
        reports = detector.check_all()
        assert len(reports) == 1
        assert reports[0].status == "stale"
        assert reports[0].recommended_action == "re-observe"

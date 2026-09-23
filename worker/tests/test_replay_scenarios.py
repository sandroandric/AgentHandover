"""Replay scenario evaluation tests.

Run with: ``pytest -m eval worker/tests/test_replay_scenarios.py``

Each test loads a fixture, runs a pipeline stage through the production
code with its production defaults, and asserts metrics above trivial
thresholds.  Ground truth is converted by the scorer's own ``*_truth``
functions, the same ones ``Scorer.score_scenario`` uses.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agenthandover_worker.staleness_detector import procedure_freshness
from eval.replay_runner import ReplayScenario, ReplayRunner
from eval.scorer import (
    Scorer,
    boundary_truth,
    classification_truth,
    dedup_truth,
    readiness_truth,
    segmentation_truth,
)
from journeys.cassette import open_cassette

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "replay"

# Use pytest.mark.eval for all tests in this module
pytestmark = pytest.mark.eval


@pytest.fixture()
def ollama_host():
    """Real nomic-embed-text vectors for the segmenter, from the recorded
    ``eval_replay`` cassette (``AH_RECORD=1`` re-records)."""
    with open_cassette("eval_replay") as tape:
        yield tape.host


def _load(name: str) -> ReplayScenario:
    """Load a scenario fixture by name."""
    return ReplayScenario.load(FIXTURES_DIR / f"{name}.json")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    """Classification pipeline produces non-degenerate results."""

    @pytest.mark.parametrize("name", [
        "morning_expense_interruptions",
        "noise_heavy_browsing_session",
        "entertainment_detour",
    ])
    def test_classification_f1_above_zero(self, name: str, tmp_path: Path) -> None:
        scenario = _load(name)
        runner = ReplayRunner(scenario)
        scorer = Scorer()
        predictions = runner.run_classification()

        gt_raw = scenario.ground_truth.get("classification", [])
        if not gt_raw:
            pytest.skip("No classification ground truth")

        gt = classification_truth(gt_raw)
        metrics = scorer.score_classification(predictions, gt)
        assert metrics["f1"] > 0.0, f"Classification F1 is 0 for {name}"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class TestSegmentation:
    """Segmentation clusters frames correctly."""

    @pytest.mark.parametrize("name", [
        "morning_expense_interruptions",
        "cross_app_research_workflow",
        "long_gap_resumption",
    ])
    def test_segmentation_purity_above_half(self, name: str, tmp_path: Path, ollama_host) -> None:
        scenario = _load(name)
        runner = ReplayRunner(scenario, ollama_host=ollama_host)
        scorer = Scorer()
        result = runner.run_segmentation()

        gt_segments = scenario.ground_truth.get("segments", [])
        if not gt_segments:
            pytest.skip("No segmentation ground truth")

        gt = segmentation_truth(gt_segments, scenario.events)

        metrics = scorer.score_segmentation(result, gt)
        assert metrics["purity"] >= 0.5, (
            f"Segmentation purity below 0.5 for {name}: {metrics['purity']}"
        )


# ---------------------------------------------------------------------------
# Task boundaries
# ---------------------------------------------------------------------------


class TestTaskBoundaries:
    """Task boundary detection is non-degenerate."""

    @pytest.mark.parametrize("name", [
        "morning_expense_interruptions",
        "cross_app_research_workflow",
        "multi_day_recurring_standup",
    ])
    def test_boundary_jaccard_above_zero(self, name: str, tmp_path: Path) -> None:
        scenario = _load(name)
        runner = ReplayRunner(scenario)
        scorer = Scorer()

        predicted = runner.run_task_boundaries(tmp_path)

        gt = scenario.ground_truth.get("task_boundaries", [])
        if not gt:
            pytest.skip("No boundary ground truth")

        metrics = scorer.score_task_boundaries(predicted, boundary_truth(gt, scenario.events))
        assert metrics["jaccard"] > 0.0, (
            f"Boundary Jaccard is 0 for {name}: {metrics}"
        )


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    """Dedup correctly separates distinct SOPs."""

    def test_no_false_merge_on_different_tasks(self, tmp_path: Path) -> None:
        scenario = _load("similar_but_different_sops")
        runner = ReplayRunner(scenario)
        scorer = Scorer()

        decisions = runner.run_dedup(tmp_path)

        gt_raw = scenario.ground_truth.get("dedup_families", [])
        if not gt_raw:
            pytest.skip("No dedup ground truth")

        gt = dedup_truth(gt_raw)
        metrics = scorer.score_dedup(decisions, gt)
        assert metrics["false_merge_rate"] == 0.0, (
            f"False merge detected! Rate={metrics['false_merge_rate']}"
        )


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class TestReadiness:
    """Readiness checks match expected values."""

    def test_zero_false_ready(self, tmp_path: Path) -> None:
        scenario = _load("stale_and_draft_procedures")
        runner = ReplayRunner(scenario)
        kb = runner.setup_kb(tmp_path)
        scorer = Scorer()

        results = runner.run_readiness(kb)

        gt_raw = scenario.ground_truth.get("procedure_readiness", [])
        if not gt_raw:
            pytest.skip("No readiness ground truth")

        gt = readiness_truth(gt_raw)
        metrics = scorer.score_readiness(results, gt)
        assert metrics["false_ready_rate"] == 0.0, (
            f"False-ready detected! Rate={metrics['false_ready_rate']}"
        )

    def test_fresh_autonomous_not_blocked(self, tmp_path: Path) -> None:
        scenario = _load("stale_and_draft_procedures")
        runner = ReplayRunner(scenario)
        kb = runner.setup_kb(tmp_path)
        scorer = Scorer()

        results = runner.run_readiness(kb)

        gt_raw = scenario.ground_truth.get("procedure_readiness", [])
        if not gt_raw:
            pytest.skip("No readiness ground truth")

        gt = readiness_truth(gt_raw)
        metrics = scorer.score_readiness(results, gt)
        assert metrics["false_block_rate"] == 0.0, (
            f"False-block detected! Rate={metrics['false_block_rate']}"
        )


# ---------------------------------------------------------------------------
# Export parity
# ---------------------------------------------------------------------------


class TestExportParity:
    """All adapters produce consistent output."""

    def test_export_parity_above_threshold(self, tmp_path: Path) -> None:
        scenario = _load("stale_and_draft_procedures")
        runner = ReplayRunner(scenario)
        kb = runner.setup_kb(tmp_path)
        scorer = Scorer()

        parity = runner.run_export_parity(kb, tmp_path / "export")
        metrics = scorer.score_export_parity(parity)
        assert metrics["field_coverage_parity"] >= 0.95, (
            f"Export parity below 0.95: {metrics['field_coverage_parity']}"
        )


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------


class TestRecurrence:
    """Recurrence detection finds daily patterns."""

    @pytest.mark.xfail(strict=True, reason=(
        "measured on the production path (WP-M4): the daily batch splits each "
        "standup into a Zoom task and a Jira task, and ProcedureMatcher scores "
        "neither >= 0.50 against daily-standup (the verb map leaves 'opening' / "
        "'reviewing' unnormalised), so no day observes the procedure; the "
        "recurrence came only from hand-written matched_procedure summaries"
    ))
    def test_daily_pattern_detected(self, tmp_path: Path) -> None:
        scenario = _load("multi_day_recurring_standup")
        runner = ReplayRunner(scenario)
        kb = runner.setup_kb(tmp_path)
        scorer = Scorer()

        detected = runner.run_recurrence(kb)

        gt = scenario.ground_truth.get("recurrence", [])
        if not gt:
            pytest.skip("No recurrence ground truth")

        metrics = scorer.score_recurrence(detected, gt)
        assert metrics["recall"] > 0.0, (
            f"No recurrence patterns detected: {metrics}"
        )


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------


class TestContinuity:
    """Continuity tracking links interrupted sessions."""

    def test_continuity_produces_spans(self, tmp_path: Path, ollama_host) -> None:
        scenario = _load("interruption_resume_workflow")
        runner = ReplayRunner(scenario, ollama_host=ollama_host)
        scorer = Scorer()

        spans = runner.run_continuity(tmp_path / "cont")

        gt = scenario.ground_truth.get("continuity", {})
        expected = gt.get("expected_spans", [])
        if not expected:
            pytest.skip("No continuity ground truth")

        metrics = scorer.score_continuity(spans, expected)
        # The tracker should produce at least some spans
        assert len(spans) > 0, "No continuity spans produced"
        # False merge rate should be low
        assert metrics["false_merge_rate"] <= 0.5, (
            f"High false merge rate: {metrics['false_merge_rate']}"
        )


# ---------------------------------------------------------------------------
# Fixture reference date (time seam)
# ---------------------------------------------------------------------------


class TestReferenceDateShift:
    """Fixtures with ``reference_date`` are replayed relative to *now*."""

    REFERENCE = datetime(2026, 3, 13, 12, tzinfo=timezone.utc)

    def _procs(self, now: datetime) -> dict[str, dict]:
        scenario = ReplayScenario.load(
            FIXTURES_DIR / "stale_and_draft_procedures.json", now=now,
        )
        return {p["id"]: p for p in scenario.procedures}

    def test_timestamps_move_with_now(self) -> None:
        now = datetime(2031, 7, 1, 8, 30, tzinfo=timezone.utc)
        procs = self._procs(now)
        # The fixture's "today" lands on *now* ...
        assert procs["proc-fresh-autonomous"]["staleness"]["last_observed"] == (
            "2031-07-01T08:30:00Z"
        )
        # ... and the stale procedure keeps its distance from "today".
        stale = datetime.fromisoformat(
            procs["proc-stale-autonomous"]["staleness"]["last_observed"].replace("Z", "+00:00")
        )
        original = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert now - stale == self.REFERENCE - original

    @pytest.mark.parametrize("days_later", [0, 190, 5 * 365])
    def test_freshness_is_independent_of_run_date(self, days_later: int) -> None:
        now = self.REFERENCE + timedelta(days=days_later)
        procs = self._procs(now)
        fresh = procedure_freshness(procs["proc-fresh-autonomous"], now=now)
        stale = procedure_freshness(procs["proc-stale-autonomous"], now=now)
        assert fresh == procedure_freshness(
            self._procs(self.REFERENCE)["proc-fresh-autonomous"], now=self.REFERENCE,
        )
        assert fresh >= 0.9
        assert stale < 0.1

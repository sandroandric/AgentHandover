"""Replay scenario evaluation tests.

Run with: ``pytest -m eval worker/tests/test_replay_scenarios.py``

Each test loads a fixture, runs a pipeline stage, and asserts metrics
above trivial thresholds.  Ground-truth data is converted from the
fixture format to the format expected by ``Scorer`` methods.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from eval.replay_runner import ReplayScenario, ReplayRunner
from eval.scorer import Scorer

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "replay"

# Use pytest.mark.eval for all tests in this module
pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Ground-truth conversion helpers
# ---------------------------------------------------------------------------

def _classification_gt_to_scorer_format(gt: list[dict]) -> list[dict]:
    """Convert fixture classification GT (label: work/noise) to scorer
    format (is_workflow: bool).

    Fixture format:  ``{"event_id": str, "label": "work" | "noise"}``
    Scorer expects:  ``{"event_id": str, "is_workflow": bool}``
    """
    return [
        {
            "event_id": item["event_id"],
            "is_workflow": item.get("label", "noise") == "work",
        }
        for item in gt
    ]


def _segments_gt_to_scorer_format(
    segments: list[dict],
    all_event_ids: list[str],
) -> list[dict]:
    """Convert fixture segmentation GT to per-event scorer format.

    Fixture format::

        [{"segment_id": str, "event_ids": [...], "cluster_family": str}, ...]

    Scorer expects::

        [{"event_id": str, "family": int, "is_noise": bool}, ...]

    Events not mentioned in any segment are treated as noise.
    """
    # Assign a numeric family ID to each cluster_family string.
    family_map: dict[str, int] = {}
    event_to_family: dict[str, int] = {}
    segmented_ids: set[str] = set()

    for seg in segments:
        family_str = seg.get("cluster_family", seg.get("segment_id", "unknown"))
        if family_str not in family_map:
            family_map[family_str] = len(family_map)
        fam_id = family_map[family_str]
        for eid in seg.get("event_ids", []):
            event_to_family[eid] = fam_id
            segmented_ids.add(eid)

    result: list[dict] = []
    for eid in all_event_ids:
        if eid in segmented_ids:
            result.append({
                "event_id": eid,
                "family": event_to_family[eid],
                "is_noise": False,
            })
        else:
            # Assign noise events a unique family so they don't cluster.
            result.append({
                "event_id": eid,
                "family": -1,
                "is_noise": True,
            })
    return result


def _dedup_families_to_scorer_format(gt: list[dict]) -> list[dict]:
    """Convert fixture dedup_families GT to per-pair scorer format.

    Fixture format::

        [{"member_slugs": [s1, s2], "should_merge": bool}, ...]

    Scorer expects::

        [{"slug_a": str, "slug_b": str, "should_merge": bool}, ...]
    """
    result: list[dict] = []
    for item in gt:
        slugs = item.get("member_slugs", [])
        should_merge = item.get("should_merge", False)
        if len(slugs) >= 2:
            # Generate all pairs for families with >2 members.
            from itertools import combinations
            for a, b in combinations(slugs, 2):
                result.append({
                    "slug_a": a,
                    "slug_b": b,
                    "should_merge": should_merge,
                })
    return result


def _readiness_gt_to_scorer_format(gt: list[dict]) -> list[dict]:
    """Convert fixture procedure_readiness GT to scorer format.

    Fixture format::

        [{"procedure_id": str, "expected_can_execute": bool,
          "expected_can_draft": bool}, ...]

    Scorer expects::

        [{"slug": str, "can_execute": bool, "can_draft": bool}, ...]
    """
    return [
        {
            "slug": item.get("procedure_id", item.get("slug", "")),
            "can_execute": item.get("expected_can_execute", False),
            "can_draft": item.get("expected_can_draft", False),
        }
        for item in gt
    ]


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

        gt = _classification_gt_to_scorer_format(gt_raw)
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
    def test_segmentation_purity_above_half(self, name: str, tmp_path: Path) -> None:
        scenario = _load(name)
        runner = ReplayRunner(scenario)
        scorer = Scorer()
        result = runner.run_segmentation()

        gt_segments = scenario.ground_truth.get("segments", [])
        if not gt_segments:
            pytest.skip("No segmentation ground truth")

        # Collect all event IDs in fixture order.
        all_event_ids = [ev.get("id", "") for ev in scenario.events]
        gt = _segments_gt_to_scorer_format(gt_segments, all_event_ids)

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

        # Task boundaries in fixtures use {after_event_id, before_event_id}
        # rather than {event_ids: [...]} sets.  Convert: each boundary pair
        # becomes a set of its two event IDs (excluding None values).
        gt_converted: list[dict] = []
        for item in gt:
            eids = []
            if item.get("after_event_id"):
                eids.append(item["after_event_id"])
            if item.get("before_event_id"):
                eids.append(item["before_event_id"])
            if eids:
                gt_converted.append({"event_ids": eids})

        if not gt_converted:
            pytest.skip("No usable boundary ground truth after conversion")

        metrics = scorer.score_task_boundaries(predicted, gt_converted)
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

        gt = _dedup_families_to_scorer_format(gt_raw)
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

        gt = _readiness_gt_to_scorer_format(gt_raw)
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

        gt = _readiness_gt_to_scorer_format(gt_raw)
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

    def test_continuity_produces_spans(self, tmp_path: Path) -> None:
        scenario = _load("interruption_resume_workflow")
        runner = ReplayRunner(scenario)
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

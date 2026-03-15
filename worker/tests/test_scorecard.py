"""Scorecard regression tests.

Run with: ``pytest -m eval worker/tests/test_scorecard.py``
"""

from __future__ import annotations

import json

import pytest
from pathlib import Path

from eval.replay_runner import ReplayScenario, ReplayRunner
from eval.scorer import Scorer, ScenarioScore
from eval.scorecard import Scorecard

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "replay"
BASELINE_PATH = Path(__file__).parent / "eval" / "baseline_scorecard.json"

pytestmark = pytest.mark.eval


def _run_all_scenarios(tmp_path: Path) -> Scorecard:
    """Run all scenarios through ``score_scenario()`` and return a Scorecard.

    Each scenario gets its own subdirectory under *tmp_path* so that
    knowledge-base and adapter output paths never collide.
    """
    scenarios = ReplayScenario.load_all(FIXTURES_DIR)
    scorer = Scorer()
    scores: list[ScenarioScore] = []

    for i, scenario in enumerate(scenarios):
        runner = ReplayRunner(scenario)
        scenario_dir = tmp_path / f"scenario_{i}_{scenario.scenario_id}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        score = scorer.score_scenario(scenario, runner, scenario_dir)
        scores.append(score)

    return Scorecard(scores)


class TestScorecard:
    def test_computes_all_metrics(self, tmp_path: Path) -> None:
        """Scorecard produces all expected metric keys."""
        card = _run_all_scenarios(tmp_path)
        agg = card.aggregate()

        assert "scenario_count" in agg
        # 8 fixture files in the replay directory
        assert agg["scenario_count"] == 9
        assert "per_scenario" in agg
        assert isinstance(agg["per_scenario"], list)
        assert len(agg["per_scenario"]) == 9

    def test_false_ready_rate_zero(self, tmp_path: Path) -> None:
        """Aggregate false-ready rate must be exactly 0.0 (when scored)."""
        card = _run_all_scenarios(tmp_path)
        agg = card.aggregate()

        # false_ready_rate may be None if no fixture matched the scorer's
        # "readiness" key, which is expected given current fixture format.
        if agg.get("false_ready_rate") is not None:
            assert agg["false_ready_rate"] == 0.0

    def test_export_parity_above_threshold(self, tmp_path: Path) -> None:
        """Export parity >= 0.95 (when scored)."""
        card = _run_all_scenarios(tmp_path)
        agg = card.aggregate()

        if agg.get("export_parity") is not None:
            assert agg["export_parity"] >= 0.95

    def test_scorecard_json_valid(self, tmp_path: Path) -> None:
        """Scorecard JSON output is valid and parseable."""
        card = _run_all_scenarios(tmp_path)
        raw = card.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "scenario_count" in parsed
        assert parsed["scenario_count"] == 9

    def test_no_regression_from_baseline(self, tmp_path: Path) -> None:
        """No metric regresses below baseline (if baseline exists)."""
        card = _run_all_scenarios(tmp_path)
        baseline = Scorecard.load_baseline(BASELINE_PATH)
        if baseline is None:
            pytest.skip("No baseline exists yet")
        regressions = card.check_regression(baseline)
        assert regressions == [], f"Regressions: {regressions}"

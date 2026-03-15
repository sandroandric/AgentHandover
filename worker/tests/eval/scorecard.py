"""Scorecard — aggregate scenario scores into a regression-trackable baseline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from eval.scorer import ScenarioScore


# Metric dimensions where *lower* is better.  For these, regression means
# the current value is *higher* than the baseline + tolerance (not lower).
_LOWER_IS_BETTER = frozenset({
    "false_merge_rate",
    "missed_merge_rate",
    "false_ready_rate",
    "false_block_rate",
    "continuity_false_merge_rate",
})

# All metric keys that the scorecard attempts to aggregate.
_METRIC_MAP: dict[str, tuple[str, str]] = {
    # key_in_aggregate: (ScenarioScore field, sub-key within that dict)
    "classification_f1":      ("classification", "f1"),
    "classification_precision": ("classification", "precision"),
    "classification_recall":  ("classification", "recall"),
    "activity_type_f1":       ("classification", "activity_type_f1"),
    "segmentation_ari":       ("segmentation", "ari"),
    "segmentation_purity":    ("segmentation", "purity"),
    "boundary_f1":            ("task_boundaries", "f1"),
    "boundary_jaccard":       ("task_boundaries", "jaccard"),
    "false_merge_rate":       ("dedup", "false_merge_rate"),
    "missed_merge_rate":      ("dedup", "missed_merge_rate"),
    "false_ready_rate":       ("readiness", "false_ready_rate"),
    "false_block_rate":       ("readiness", "false_block_rate"),
    "readiness_accuracy":     ("readiness", "accuracy"),
    "export_parity":          ("export_parity", "field_coverage_parity"),
    "recurrence_precision":   ("recurrence", "precision"),
    "recurrence_recall":      ("recurrence", "recall"),
    "continuity_span_f1":     ("continuity", "span_f1"),
    "continuity_false_merge_rate": ("continuity", "false_merge_rate"),
}


class Scorecard:
    """Aggregates scores across all evaluated scenarios."""

    def __init__(self, scores: list[ScenarioScore]):
        self._scores = scores

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> dict:
        """Compute macro-averaged metrics across all scenarios.

        For each metric dimension (classification, segmentation, etc.),
        average the per-scenario values across scenarios that have ground truth
        for that dimension.

        Returns dict with keys:
        - classification_f1, classification_precision, classification_recall
        - segmentation_ari, segmentation_purity
        - boundary_f1, boundary_jaccard
        - false_merge_rate, missed_merge_rate
        - false_ready_rate, false_block_rate, readiness_accuracy
        - export_parity
        - recurrence_precision, recurrence_recall
        - scenario_count
        - per_scenario: list of individual ScenarioScore dicts
        """
        agg: dict[str, float | None] = {}

        for agg_key, (field_name, sub_key) in _METRIC_MAP.items():
            values: list[float] = []
            for score in self._scores:
                dim_dict = getattr(score, field_name, None)
                if dim_dict is not None and isinstance(dim_dict, dict):
                    val = dim_dict.get(sub_key)
                    if val is not None:
                        values.append(float(val))
            if values:
                agg[agg_key] = round(sum(values) / len(values), 4)
            else:
                agg[agg_key] = None

        agg["scenario_count"] = len(self._scores)
        agg["per_scenario"] = [asdict(s) for s in self._scores]
        return agg

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize aggregated scorecard to indented JSON."""
        return json.dumps(self.aggregate(), indent=2, default=str)

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    @classmethod
    def load_baseline(cls, path: Path) -> dict | None:
        """Load baseline scorecard from JSON, or None if file doesn't exist."""
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def check_regression(self, baseline: dict, tolerance: float = 0.02) -> list[str]:
        """Compare current aggregate to baseline.

        A metric regresses if current < baseline - tolerance.
        Returns list of regression message strings (empty = no regression).

        Special case: false_merge_rate, missed_merge_rate, false_ready_rate,
        and false_block_rate should NOT increase (lower is better), so
        regression means current > baseline + tolerance.
        """
        current = self.aggregate()
        regressions: list[str] = []

        for key in _METRIC_MAP:
            cur_val = current.get(key)
            base_val = baseline.get(key)

            # Skip metrics that are absent in either scorecard.
            if cur_val is None or base_val is None:
                continue

            if key in _LOWER_IS_BETTER:
                # For lower-is-better metrics, regression = value increased.
                if cur_val > base_val + tolerance:
                    regressions.append(
                        f"{key}: {cur_val:.4f} > baseline {base_val:.4f} "
                        f"+ tolerance {tolerance} (lower is better)"
                    )
            else:
                # For higher-is-better metrics, regression = value decreased.
                if cur_val < base_val - tolerance:
                    regressions.append(
                        f"{key}: {cur_val:.4f} < baseline {base_val:.4f} "
                        f"- tolerance {tolerance}"
                    )

        return regressions

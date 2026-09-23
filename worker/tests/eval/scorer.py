"""Scorer — compute evaluation metrics from runner outputs vs ground truth.

Provides ``ScenarioScore`` (a dataclass holding per-dimension metric dicts)
and ``Scorer`` (orchestrates metric computation by comparing runner outputs
against ground truth).  All heavy numeric work is delegated to
``eval.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.metrics import (
    adjusted_rand_index,
    boundary_overlap_jaccard,
    classification_metrics,
    cluster_purity,
    family_precision_recall,
    multiclass_macro_f1,
    noise_drop_accuracy,
    precision_recall_f1,
)


# ---------------------------------------------------------------------------
# ScenarioScore — per-scenario metric container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioScore:
    """Aggregated evaluation metrics for a single scenario.

    Each field is ``None`` when the corresponding ground-truth section
    does not exist in the scenario fixture, indicating that dimension
    was not evaluated.
    """

    scenario_id: str
    classification: dict | None = None
    """``{"precision", "recall", "f1", "accuracy"}``"""

    segmentation: dict | None = None
    """``{"ari", "purity", "noise_drop_accuracy"}``"""

    task_boundaries: dict | None = None
    """``{"precision", "recall", "f1", "jaccard"}``"""

    dedup: dict | None = None
    """``{"false_merge_rate", "missed_merge_rate"}``"""

    readiness: dict | None = None
    """``{"false_ready_rate", "false_block_rate", "accuracy"}``"""

    export_parity: dict | None = None
    """``{"field_coverage_parity"}``"""

    recurrence: dict | None = None
    """``{"precision", "recall"}``"""

    continuity: dict | None = None
    """``{"span_precision", "span_recall", "span_f1", "false_merge_rate"}``"""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    """Compute evaluation metrics by comparing runner outputs to ground truth.

    Each ``score_*`` method handles one evaluation dimension.  The
    ``score_scenario()`` method orchestrates everything: it checks which
    ground-truth sections exist, runs the corresponding runner methods
    and scoring methods, and returns a ``ScenarioScore``.
    """

    # ------------------------------------------------------------------
    # 1. Classification
    # ------------------------------------------------------------------

    def score_classification(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
    ) -> dict:
        """Score binary ``is_workflow`` classification.

        Matches predictions to ground truth by ``event_id`` and delegates
        to ``classification_metrics()``.

        Args:
            predictions:  ``[{"event_id": str, "is_workflow": bool}, ...]``
            ground_truth: ``[{"event_id": str, "is_workflow": bool}, ...]``

        Returns:
            Dict with ``precision``, ``recall``, ``f1``, ``accuracy``.
        """
        # Index ground truth by event_id
        gt_by_id = {g["event_id"]: g["is_workflow"] for g in ground_truth}

        pred_labels: list[bool] = []
        actual_labels: list[bool] = []

        for pred in predictions:
            eid = pred["event_id"]
            if eid in gt_by_id:
                pred_labels.append(bool(pred["is_workflow"]))
                actual_labels.append(bool(gt_by_id[eid]))

        if not pred_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        raw = classification_metrics(pred_labels, actual_labels)
        return {
            "precision": raw["precision"],
            "recall": raw["recall"],
            "f1": raw["f1"],
            "accuracy": raw["accuracy"],
        }

    # ------------------------------------------------------------------
    # 1b. Activity-type classification (multiclass)
    # ------------------------------------------------------------------

    def score_activity_type_classification(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
    ) -> dict:
        """Score 8-class ``activity_type`` classification.

        Matches predictions to ground truth by ``event_id`` and computes
        macro-averaged F1 across all activity types present in the ground
        truth.

        Args:
            predictions:  ``[{"event_id": str, "activity_type": str, ...}, ...]``
            ground_truth: ``[{"event_id": str, "activity_type": str, ...}, ...]``

        Returns:
            Dict with ``activity_type_f1`` (float) and ``per_class``
            (dict of per-class precision/recall/F1).
        """
        gt_by_id = {g["event_id"]: g["activity_type"] for g in ground_truth}

        pred_labels: list[str] = []
        actual_labels: list[str] = []

        for pred in predictions:
            eid = pred["event_id"]
            if eid in gt_by_id:
                pred_labels.append(str(pred.get("activity_type", "")))
                actual_labels.append(str(gt_by_id[eid]))

        if not pred_labels:
            return {"activity_type_f1": 0.0, "per_class": {}}

        raw = multiclass_macro_f1(pred_labels, actual_labels)
        return {
            "activity_type_f1": raw["macro_f1"],
            "per_class": raw["per_class"],
        }

    # ------------------------------------------------------------------
    # 2. Segmentation
    # ------------------------------------------------------------------

    def score_segmentation(
        self,
        result: Any,
        ground_truth: list[dict],
    ) -> dict:
        """Score segmentation quality (ARI, purity, noise drop accuracy).

        Maps each event to a predicted cluster ID (from the
        ``SegmentationResult``) and a truth cluster family (from
        ground truth).

        Args:
            result:       ``SegmentationResult`` from the task segmenter.
            ground_truth: ``[{"event_id": str, "family": int,
                             "is_noise": bool}, ...]``

        Returns:
            Dict with ``ari``, ``purity``, ``noise_drop_accuracy``.
        """
        # Build predicted cluster map: event_id -> cluster_id
        pred_cluster: dict[str, int] = {}
        for seg in result.segments:
            for frame in seg.frames:
                pred_cluster[frame.event_id] = seg.cluster_id

        # Build ground truth map
        gt_by_id = {g["event_id"]: g for g in ground_truth}

        # Collect aligned labels for ARI / purity (excluding noise events)
        predicted_labels: list[int] = []
        truth_labels: list[int] = []

        # Collect IDs for noise-drop accuracy
        all_ids: set[str] = set()
        truth_noise_ids: set[str] = set()
        predicted_dropped: set[str] = set()

        for gt_item in ground_truth:
            eid = gt_item["event_id"]
            all_ids.add(eid)

            is_noise = gt_item.get("is_noise", False)
            if is_noise:
                truth_noise_ids.add(eid)

            if eid not in pred_cluster:
                # The segmenter dropped this event (treated as noise)
                predicted_dropped.add(eid)
            else:
                if not is_noise:
                    # Include in clustering metrics
                    predicted_labels.append(pred_cluster[eid])
                    truth_labels.append(gt_item["family"])

        # ARI and purity
        if len(predicted_labels) >= 2:
            ari = adjusted_rand_index(predicted_labels, truth_labels)
            pur = cluster_purity(predicted_labels, truth_labels)
        else:
            ari = 0.0
            pur = 0.0

        # Noise drop accuracy
        nda = noise_drop_accuracy(predicted_dropped, truth_noise_ids, all_ids)

        return {
            "ari": round(ari, 4),
            "purity": round(pur, 4),
            "noise_drop_accuracy": round(nda.get("accuracy", 0.0), 4),
        }

    # ------------------------------------------------------------------
    # 3. Task boundaries
    # ------------------------------------------------------------------

    def score_task_boundaries(
        self,
        predicted: list,
        ground_truth: list[dict],
    ) -> dict:
        """Score task-boundary detection quality.

        Converts both predicted ``TaskBoundary`` objects and ground-truth
        dicts into sets of event IDs, then computes Jaccard overlap and
        boundary-level precision/recall.

        A predicted boundary "matches" a truth boundary when their
        Jaccard overlap exceeds 0.3.

        Args:
            predicted:    List of ``TaskBoundary`` objects.
            ground_truth: ``[{"event_ids": [...], "intent": str}, ...]``

        Returns:
            Dict with ``precision``, ``recall``, ``f1``, ``jaccard``.
        """
        pred_sets: list[set[str]] = []
        for tb in predicted:
            eids = getattr(tb, "event_ids", [])
            if eids:
                pred_sets.append(set(eids))

        truth_sets: list[set[str]] = []
        for gt_item in ground_truth:
            eids = gt_item.get("event_ids", [])
            if eids:
                truth_sets.append(set(eids))

        # Average best-match Jaccard
        jaccard = boundary_overlap_jaccard(pred_sets, truth_sets)

        # Precision/recall at match threshold 0.3
        match_threshold = 0.3
        matched_truth: set[int] = set()
        matched_pred: set[int] = set()

        for pi, ps in enumerate(pred_sets):
            best_j = 0.0
            best_ti = -1
            for ti, ts in enumerate(truth_sets):
                inter = len(ps & ts)
                union = len(ps | ts)
                j = inter / union if union > 0 else 0.0
                if j > best_j:
                    best_j = j
                    best_ti = ti
            if best_j >= match_threshold and best_ti >= 0:
                matched_pred.add(pi)
                matched_truth.add(best_ti)

        tp = len(matched_truth)
        fp = len(pred_sets) - len(matched_pred)
        fn = len(truth_sets) - len(matched_truth)

        prf = precision_recall_f1(tp, fp, fn)

        return {
            "precision": round(prf["precision"], 4),
            "recall": round(prf["recall"], 4),
            "f1": round(prf["f1"], 4),
            "jaccard": round(jaccard, 4),
        }

    # ------------------------------------------------------------------
    # 4. Dedup
    # ------------------------------------------------------------------

    def score_dedup(
        self,
        merge_decisions: list[tuple],
        ground_truth: list[dict],
    ) -> dict:
        """Score dedup merge decisions against ground-truth families.

        For each ground-truth entry, checks whether pairs that
        ``should_merge`` have similarity above the threshold (0.70)
        and pairs that should not merge stay below it.

        Args:
            merge_decisions: ``[(slug_a, slug_b, similarity), ...]``
            ground_truth:    ``[{"slug_a": str, "slug_b": str,
                                "should_merge": bool}, ...]``

        Returns:
            Dict with ``false_merge_rate`` and ``missed_merge_rate``.
        """
        threshold = 0.70

        # Index similarity scores by pair
        sim_map: dict[frozenset[str], float] = {}
        for slug_a, slug_b, sim in merge_decisions:
            sim_map[frozenset((slug_a, slug_b))] = sim

        false_merges = 0
        total_no_merge = 0
        missed_merges = 0
        total_should_merge = 0

        for gt_item in ground_truth:
            pair = frozenset((gt_item["slug_a"], gt_item["slug_b"]))
            sim = sim_map.get(pair, 0.0)
            should_merge = gt_item.get("should_merge", False)

            if should_merge:
                total_should_merge += 1
                if sim < threshold:
                    missed_merges += 1
            else:
                total_no_merge += 1
                if sim >= threshold:
                    false_merges += 1

        false_merge_rate = (
            false_merges / total_no_merge if total_no_merge > 0 else 0.0
        )
        missed_merge_rate = (
            missed_merges / total_should_merge
            if total_should_merge > 0
            else 0.0
        )

        return {
            "false_merge_rate": round(false_merge_rate, 4),
            "missed_merge_rate": round(missed_merge_rate, 4),
        }

    # ------------------------------------------------------------------
    # 5. Readiness
    # ------------------------------------------------------------------

    def score_readiness(
        self,
        preflight_results: list[dict],
        ground_truth: list[dict],
    ) -> dict:
        """Score preflight readiness predictions against expected values.

        Computes false-ready rate (predicted can_execute but should not),
        false-block rate (predicted cannot execute but should), and
        overall accuracy.

        Args:
            preflight_results: ``[{"slug": str, "can_execute": bool,
                                  "can_draft": bool}, ...]``
            ground_truth:      ``[{"slug": str, "can_execute": bool,
                                  "can_draft": bool}, ...]``

        Returns:
            Dict with ``false_ready_rate``, ``false_block_rate``,
            ``accuracy``.
        """
        gt_by_slug = {g["slug"]: g for g in ground_truth}

        false_ready = 0
        false_block = 0
        correct = 0
        total = 0

        for pf in preflight_results:
            slug = pf["slug"]
            gt = gt_by_slug.get(slug)
            if gt is None:
                continue

            total += 1
            pred_exec = pf.get("can_execute", False)
            gt_exec = gt.get("can_execute", False)

            if pred_exec == gt_exec:
                correct += 1
            elif pred_exec and not gt_exec:
                false_ready += 1
            elif not pred_exec and gt_exec:
                false_block += 1

        accuracy = correct / total if total > 0 else 0.0
        false_ready_rate = false_ready / total if total > 0 else 0.0
        false_block_rate = false_block / total if total > 0 else 0.0

        return {
            "false_ready_rate": round(false_ready_rate, 4),
            "false_block_rate": round(false_block_rate, 4),
            "accuracy": round(accuracy, 4),
        }

    # ------------------------------------------------------------------
    # 6. Export parity
    # ------------------------------------------------------------------

    def score_export_parity(self, parity_result: dict) -> dict:
        """Pass through the field-coverage parity score from the runner.

        Args:
            parity_result: Dict returned by ``ReplayRunner.run_export_parity()``.

        Returns:
            Dict with ``field_coverage_parity``.
        """
        return {
            "field_coverage_parity": parity_result.get(
                "field_coverage_parity", 0.0
            ),
        }

    # ------------------------------------------------------------------
    # 7. Recurrence
    # ------------------------------------------------------------------

    def score_recurrence(
        self,
        detected: list[dict],
        ground_truth: list[dict],
    ) -> dict:
        """Score recurrence pattern detection against expected patterns.

        A detected pattern "matches" an expected one when both the
        ``procedure_slug`` and ``pattern`` type agree.

        Args:
            detected:     ``[{"procedure_slug": str, "pattern": str, ...}, ...]``
            ground_truth: ``[{"procedure_slug": str, "pattern": str}, ...]``

        Returns:
            Dict with ``precision`` and ``recall``.
        """
        detected_keys: set[tuple[str, str]] = set()
        for d in detected:
            detected_keys.add(
                (d.get("procedure_slug", ""), d.get("pattern", ""))
            )

        truth_keys: set[tuple[str, str]] = set()
        for g in ground_truth:
            truth_keys.add(
                (g.get("procedure_slug", ""), g.get("pattern", ""))
            )

        tp = len(detected_keys & truth_keys)
        fp = len(detected_keys - truth_keys)
        fn = len(truth_keys - detected_keys)

        prf = precision_recall_f1(tp, fp, fn)

        return {
            "precision": round(prf["precision"], 4),
            "recall": round(prf["recall"], 4),
        }

    # ------------------------------------------------------------------
    # 8. Continuity
    # ------------------------------------------------------------------

    def score_continuity(
        self,
        predicted_spans: list,
        ground_truth: list[dict],
    ) -> dict:
        """Score continuity quality.

        A predicted span "matches" a ground-truth span when the Jaccard
        overlap of their event_id sets exceeds 0.5.

        Args:
            predicted_spans: List of dicts with ``'span_id'`` and
                ``'event_ids'`` keys (resolved by the runner).
            ground_truth: ``[{"goal": str, "event_ids": [...], ...}]``

        Returns:
            Dict with ``span_precision``, ``span_recall``, ``span_f1``,
            ``false_merge_rate``.
        """
        # Build event_id sets for predicted spans
        pred_sets: list[set[str]] = []
        for span in predicted_spans:
            eids = span.get("event_ids", [])
            pred_sets.append(set(eids))

        # Build event_id sets for ground-truth spans
        gt_sets: list[set[str]] = []
        for gt_span in ground_truth:
            eids = gt_span.get("event_ids", [])
            gt_sets.append(set(eids))

        if not gt_sets and not pred_sets:
            return {
                "span_precision": 1.0,
                "span_recall": 1.0,
                "span_f1": 1.0,
                "false_merge_rate": 0.0,
            }

        # Match threshold for Jaccard overlap
        match_threshold = 0.5

        # For each predicted span, find best matching GT span
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        for pi, ps in enumerate(pred_sets):
            if not ps:
                continue
            best_j = 0.0
            best_gi = -1
            for gi, gs in enumerate(gt_sets):
                if not gs:
                    continue
                inter = len(ps & gs)
                union = len(ps | gs)
                j = inter / union if union > 0 else 0.0
                if j > best_j:
                    best_j = j
                    best_gi = gi
            if best_j >= match_threshold and best_gi >= 0:
                matched_pred.add(pi)
                matched_gt.add(best_gi)

        tp = len(matched_gt)
        fp = len(pred_sets) - len(matched_pred)
        fn = len(gt_sets) - len(matched_gt)

        prf = precision_recall_f1(tp, fp, fn)

        # False merge rate: fraction of predicted spans whose event_ids
        # overlap with 2+ different GT spans (wrongful merging).
        false_merge_count = 0
        for ps in pred_sets:
            if not ps:
                continue
            overlapping_gt = 0
            for gs in gt_sets:
                if not gs:
                    continue
                if ps & gs:
                    overlapping_gt += 1
            if overlapping_gt >= 2:
                false_merge_count += 1

        false_merge_rate = (
            false_merge_count / len(pred_sets) if pred_sets else 0.0
        )

        return {
            "span_precision": round(prf["precision"], 4),
            "span_recall": round(prf["recall"], 4),
            "span_f1": round(prf["f1"], 4),
            "false_merge_rate": round(false_merge_rate, 4),
        }

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def score_scenario(
        self,
        scenario: Any,
        runner: Any,
        tmp_path: Path,
    ) -> ScenarioScore:
        """Run all applicable evaluation dimensions for a scenario.

        Checks which ground-truth sections exist in the scenario, converts
        each from the fixture format (the ``*_truth`` functions below),
        executes the corresponding runner method and scores it.  A runner
        or scorer error propagates: a dimension that cannot be measured
        fails the scenario instead of vanishing from the scorecard.

        Args:
            scenario: A ``ReplayScenario``.
            runner:   A ``ReplayRunner`` for that scenario.
            tmp_path: Temporary directory for KB and adapter output.

        Returns:
            A ``ScenarioScore`` with metrics for each evaluated dimension.
        """
        gt = scenario.ground_truth
        score = ScenarioScore(scenario_id=scenario.scenario_id)

        # Set up KB once (shared by readiness, export parity, recurrence)
        kb = runner.setup_kb(tmp_path / "kb")

        if "classification" in gt:
            score.classification = self.score_classification(
                runner.run_classification(), classification_truth(gt["classification"]),
            )
            if any(item.get("activity_type") for item in gt["classification"]):
                score.classification.update(self.score_activity_type_classification(
                    runner.run_activity_classification(), gt["classification"],
                ))

        if "segments" in gt:
            score.segmentation = self.score_segmentation(
                runner.run_segmentation(),
                segmentation_truth(gt["segments"], scenario.events),
            )

        if "task_boundaries" in gt:
            score.task_boundaries = self.score_task_boundaries(
                runner.run_task_boundaries(tmp_path / "boundaries"),
                boundary_truth(gt["task_boundaries"], scenario.events),
            )

        if "dedup_families" in gt:
            score.dedup = self.score_dedup(
                runner.run_dedup(tmp_path / "dedup"), dedup_truth(gt["dedup_families"]),
            )

        if "procedure_readiness" in gt:
            score.readiness = self.score_readiness(
                runner.run_readiness(kb), readiness_truth(gt["procedure_readiness"]),
            )

        # Export parity runs whenever the scenario has procedures.
        if scenario.procedures:
            score.export_parity = self.score_export_parity(
                runner.run_export_parity(kb, tmp_path / "export_parity"),
            )

        if "recurrence" in gt:
            score.recurrence = self.score_recurrence(
                runner.run_recurrence(kb), gt["recurrence"],
            )

        if "continuity" in gt:
            score.continuity = self.score_continuity(
                runner.run_continuity(tmp_path / "continuity"),
                gt["continuity"].get("expected_spans", []),
            )

        return score


# ---------------------------------------------------------------------------
# Ground truth: fixture format -> scorer format
# ---------------------------------------------------------------------------


def classification_truth(items: list[dict]) -> list[dict]:
    """``[{"event_id", "is_workflow": bool}]`` from fixture items, which say
    ``label`` (``"work"`` / ``"noise"``), ``is_workflow``, or only an
    ``activity_type``, in that order of precedence."""
    out = []
    for item in items:
        if "label" in item:
            is_wf = item["label"] == "work"
        elif item.get("is_workflow") is not None:
            is_wf = bool(item["is_workflow"])
        else:
            is_wf = item.get("activity_type", "") in ("work", "research", "communication", "setup")
        out.append({"event_id": item["event_id"], "is_workflow": is_wf})
    return out


def segmentation_truth(segments: list[dict], events: list[dict]) -> list[dict]:
    """``[{"event_id", "family": int, "is_noise": bool}]`` for every event of
    the scenario: an event in no ground-truth segment is noise."""
    family_ids: dict[str, int] = {}
    family_of: dict[str, int] = {}
    noise: set[str] = set()
    for seg in segments:
        family = seg.get("cluster_family", seg.get("segment_id", "unknown"))
        fam_id = family_ids.setdefault(family, len(family_ids))
        for eid in seg.get("event_ids", []):
            family_of[eid] = fam_id
            if seg.get("is_noise"):
                noise.add(eid)
    return [
        {
            "event_id": ev["id"],
            "family": family_of.get(ev["id"], -1),
            "is_noise": ev["id"] in noise or ev["id"] not in family_of,
        }
        for ev in events
    ]


def boundary_truth(boundaries: list[dict], events: list[dict]) -> list[dict]:
    """``[{"event_ids"}]``, one per true task.  The fixture marks each
    boundary as a cut between ``after_event_id`` and the next task; the
    tasks are the runs of time-ordered events between cuts."""
    cuts = {b["after_event_id"] for b in boundaries if b.get("after_event_id")}
    tasks: list[dict] = []
    current: list[str] = []
    for ev in sorted(events, key=lambda e: e.get("timestamp", "")):
        current.append(ev["id"])
        if ev["id"] in cuts:
            tasks.append({"event_ids": current})
            current = []
    if current:
        tasks.append({"event_ids": current})
    return tasks


def dedup_truth(families: list[dict]) -> list[dict]:
    """``[{"slug_a", "slug_b", "should_merge"}]``: every pair in a family."""
    from itertools import combinations
    return [
        {"slug_a": a, "slug_b": b, "should_merge": fam.get("should_merge", False)}
        for fam in families
        for a, b in combinations(fam.get("member_slugs", []), 2)
    ]


def readiness_truth(items: list[dict]) -> list[dict]:
    """``[{"slug", "can_execute", "can_draft"}]``."""
    return [
        {
            "slug": item.get("slug", item.get("procedure_id", "")),
            "can_execute": item.get("expected_can_execute", False),
            "can_draft": item.get("expected_can_draft", False),
        }
        for item in items
    ]

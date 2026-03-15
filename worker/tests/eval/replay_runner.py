"""Replay runner — loads fixture JSON files and exercises pipeline stages without VLM.

Provides ``ReplayScenario`` (parsed fixture data) and ``ReplayRunner``
(executes pipeline stages against fixture events and procedures).
Together they allow deterministic, offline evaluation of the OpenMimic
pipeline using pre-recorded ground-truth data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any
from unittest.mock import patch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReplayScenario — parsed fixture data
# ---------------------------------------------------------------------------

@dataclass
class ReplayScenario:
    """A single evaluation scenario loaded from a JSON fixture file.

    Attributes:
        scenario_id:     Unique identifier for this scenario.
        description:     Human-readable description of what the scenario tests.
        tags:            Categorical tags (e.g. ``["segmentation", "dedup"]``).
        events:          Raw event dicts as they would appear in the database.
        procedures:      v3 procedure dicts to pre-load into the knowledge base.
        daily_summaries: Daily summary dicts to pre-load into the knowledge base.
        ground_truth:    Expected outputs keyed by evaluation dimension
                         (``classification``, ``segmentation``, etc.).
    """

    scenario_id: str
    description: str
    tags: list[str]
    events: list[dict]
    procedures: list[dict]
    daily_summaries: list[dict]
    ground_truth: dict

    @classmethod
    def load(cls, path: Path) -> ReplayScenario:
        """Load a single scenario from a JSON fixture file.

        Args:
            path: Path to the JSON file.

        Returns:
            A fully parsed ``ReplayScenario``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If required top-level keys are missing.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # scenario_id: prefer explicit field, fall back to filename stem
        scenario_id = data.get("scenario_id", path.stem)

        # daily_summaries may be nested under ground_truth in some fixtures
        daily_summaries = data.get("daily_summaries", [])
        if not daily_summaries:
            daily_summaries = data.get("ground_truth", {}).get("daily_summaries", [])

        return cls(
            scenario_id=scenario_id,
            description=data.get("description", ""),
            tags=data.get("tags", []),
            events=data.get("events", []),
            procedures=data.get("procedures", []),
            daily_summaries=daily_summaries,
            ground_truth=data.get("ground_truth", {}),
        )

    @classmethod
    def load_all(cls, fixtures_dir: Path) -> list[ReplayScenario]:
        """Load all ``.json`` scenario files from a directory.

        Files are sorted by name so that evaluation order is deterministic.

        Args:
            fixtures_dir: Directory containing JSON fixture files.

        Returns:
            List of parsed scenarios.
        """
        scenarios: list[ReplayScenario] = []
        if not fixtures_dir.is_dir():
            logger.warning("Fixtures directory does not exist: %s", fixtures_dir)
            return scenarios

        for p in sorted(fixtures_dir.glob("*.json")):
            try:
                scenarios.append(cls.load(p))
            except Exception as exc:
                logger.warning("Failed to load fixture %s: %s", p, exc)
        return scenarios


# ---------------------------------------------------------------------------
# ReplayRunner — exercises pipeline stages against a scenario
# ---------------------------------------------------------------------------

class ReplayRunner:
    """Execute pipeline stages against a ``ReplayScenario`` without VLM calls.

    Each ``run_*`` method exercises one evaluation dimension (classification,
    segmentation, task boundaries, dedup, readiness, export parity,
    recurrence) using pre-stored fixture data and mock embeddings.
    """

    def __init__(self, scenario: ReplayScenario) -> None:
        self._scenario = scenario

    # ------------------------------------------------------------------
    # Knowledge-base setup
    # ------------------------------------------------------------------

    def setup_kb(self, tmp_path: Path) -> Any:
        """Create a temporary knowledge base and populate it from the scenario.

        Loads procedures via ``kb.save_procedure()``, sets constraints
        where ``_constraints_to_set`` is present, and saves daily
        summaries.

        Args:
            tmp_path: Temporary directory that will serve as the KB root.

        Returns:
            A ``KnowledgeBase`` instance rooted at *tmp_path*.
        """
        from oc_apprentice_worker.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(root=tmp_path)
        kb.ensure_structure()

        # Load procedures
        for proc in self._scenario.procedures:
            kb.save_procedure(proc)

            # Apply any scenario-specified constraints
            constraints_to_set = proc.get("_constraints_to_set")
            if constraints_to_set and isinstance(constraints_to_set, dict):
                current = kb.get_constraints()
                blocked = constraints_to_set.get("blocked_domains", [])
                if blocked:
                    existing_blocked = current.get("blocked_domains", [])
                    current["blocked_domains"] = list(
                        set(existing_blocked) | set(blocked)
                    )
                current.update(
                    {k: v for k, v in constraints_to_set.items()
                     if k != "blocked_domains"}
                )
                kb.update_constraints(current)

        # Load daily summaries
        for summary in self._scenario.daily_summaries:
            date = summary.get("date", "unknown")
            kb.save_daily_summary(date, summary)

        return kb

    # ------------------------------------------------------------------
    # 1. Classification
    # ------------------------------------------------------------------

    def run_classification(self) -> list[dict]:
        """Extract ``is_workflow`` labels from each event's annotation.

        Parses ``scene_annotation_json.task_context.is_workflow`` from
        every event in the scenario.

        Returns:
            List of ``{"event_id": str, "is_workflow": bool}`` dicts,
            one per event that has a valid annotation.
        """
        results: list[dict] = []
        for event in self._scenario.events:
            ann = event.get("scene_annotation_json")
            if ann is None:
                continue

            # Handle both dict and string forms
            if isinstance(ann, str):
                try:
                    ann = json.loads(ann)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(ann, dict):
                continue

            tc = ann.get("task_context", {})
            if not isinstance(tc, dict):
                continue

            is_wf = tc.get("is_workflow", False)
            if isinstance(is_wf, str):
                is_wf = is_wf.lower() in ("true", "yes", "1")

            results.append({
                "event_id": event.get("id", ""),
                "is_workflow": bool(is_wf),
            })
        return results

    # ------------------------------------------------------------------
    # 1b. Activity-type classification
    # ------------------------------------------------------------------

    def run_activity_classification(self) -> list[dict]:
        """Run ``ActivityClassifier`` on each event's annotation.

        Uses keyword/URL heuristics only (no profile, no policy) to
        classify each event into the 8-class activity taxonomy.

        Returns:
            List of ``{"event_id": str, "activity_type": str,
            "learnability": str, "confidence": float, "source": str}``
            dicts, one per event with a valid annotation.
        """
        from oc_apprentice_worker.activity_classifier import ActivityClassifier

        classifier = ActivityClassifier()
        results: list[dict] = []

        for event in self._scenario.events:
            ann = event.get("scene_annotation_json")
            if ann is None:
                continue

            # Handle both dict and string forms
            if isinstance(ann, str):
                try:
                    ann = json.loads(ann)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(ann, dict):
                continue

            result = classifier.classify(ann)
            results.append({
                "event_id": event.get("id", ""),
                "activity_type": result.activity_type.value,
                "learnability": result.learnability.value,
                "confidence": round(result.confidence, 4),
                "source": result.source,
            })

        return results

    # ------------------------------------------------------------------
    # 2. Segmentation
    # ------------------------------------------------------------------

    def run_segmentation(self) -> Any:
        """Run ``TaskSegmenter.segment()`` with mock embeddings.

        Events in fixtures store ``scene_annotation_json`` as a dict;
        ``AnnotatedFrame.from_event()`` accepts both dict and JSON string
        forms.  To be safe we JSON-encode dict-valued annotation fields
        before passing them to the segmenter.

        The embedding mock intercepts ``_compute_embeddings()`` and
        returns the ``embedding`` vectors pre-stored in each event.

        Returns:
            A ``SegmentationResult`` from the task segmenter.
        """
        from oc_apprentice_worker.task_segmenter import (
            TaskSegmenter,
            SegmentationResult,
        )

        # Build events with JSON-string fields (as the DB would store them)
        prepared_events = self._prepare_events_for_segmenter()

        # Collect pre-stored embeddings in event order (only for events
        # that have annotations, since those are the ones that produce
        # AnnotatedFrames and therefore request embeddings).
        ordered_embeddings = self._collect_ordered_embeddings()

        def _mock_compute_embeddings(
            texts: list[str], *, model: str = "", host: str = "",
            timeout: float = 30.0,
        ) -> list[list[float]]:
            """Return pre-stored embeddings aligned to *texts*.

            The segmenter requests embeddings in the same order as it
            creates AnnotatedFrames (i.e. event order, skipping events
            without valid annotations).  We return the stored embeddings
            1:1 by position.
            """
            result: list[list[float]] = []
            for i in range(len(texts)):
                if i < len(ordered_embeddings):
                    result.append(ordered_embeddings[i])
                else:
                    result.append([])
            return result

        segmenter = TaskSegmenter()

        with patch(
            "oc_apprentice_worker.task_segmenter._compute_embeddings",
            side_effect=_mock_compute_embeddings,
        ):
            return segmenter.segment(prepared_events)

    # ------------------------------------------------------------------
    # 3. Task boundaries
    # ------------------------------------------------------------------

    def run_task_boundaries(self, tmp_path: Path) -> list:
        """Detect task boundaries using ``DailyBatchProcessor``.

        Builds an activity stream from the scenario events and runs
        ``process_day()`` to detect boundaries.

        Args:
            tmp_path: Temporary directory for the knowledge base.

        Returns:
            List of ``TaskBoundary`` objects.
        """
        from oc_apprentice_worker.daily_processor import DailyBatchProcessor

        kb = self.setup_kb(tmp_path)
        processor = DailyBatchProcessor(knowledge_base=kb)

        # Prepare events: ensure JSON string fields
        prepared = self._prepare_events_for_daily_processor()

        # Derive date from the first event's timestamp, or use a default
        date = "2025-01-15"
        for ev in self._scenario.events:
            ts = ev.get("timestamp", "")
            if ts and len(ts) >= 10:
                date = ts[:10]
                break

        summary = processor.process_day(date, prepared)
        return summary.tasks

    # ------------------------------------------------------------------
    # 4. Dedup
    # ------------------------------------------------------------------

    def run_dedup(self, tmp_path: Path) -> list[tuple[str, str, float]]:
        """Compute pairwise fingerprint similarity for all procedure pairs.

        Converts each procedure to an SOP template, computes structural
        fingerprints via ``sop_dedup.compute_fingerprint()``, then
        measures ``sop_dedup.fingerprint_similarity()`` for every pair.

        Args:
            tmp_path: Temporary directory (unused but kept for API
                symmetry with other ``run_*`` methods).

        Returns:
            List of ``(slug_a, slug_b, similarity)`` tuples.
        """
        from oc_apprentice_worker.export_adapter import procedure_to_sop_template
        from oc_apprentice_worker.sop_dedup import (
            compute_fingerprint,
            fingerprint_similarity,
        )

        procedures = self._scenario.procedures
        if len(procedures) < 2:
            return []

        # Pre-compute SOP templates and fingerprints
        templates: list[tuple[str, dict]] = []
        for proc in procedures:
            slug = proc.get("id", proc.get("slug", "unknown"))
            sop = procedure_to_sop_template(proc)
            fp = compute_fingerprint(sop)
            templates.append((slug, fp))

        results: list[tuple[str, str, float]] = []
        for (slug_a, fp_a), (slug_b, fp_b) in combinations(templates, 2):
            sim = fingerprint_similarity(fp_a, fp_b)
            results.append((slug_a, slug_b, sim))

        return results

    # ------------------------------------------------------------------
    # 5. Readiness
    # ------------------------------------------------------------------

    def run_readiness(self, kb: Any) -> list[dict]:
        """Run preflight checks for every procedure in the scenario.

        Args:
            kb: A ``KnowledgeBase`` instance (from ``setup_kb()``).

        Returns:
            List of ``{"slug": str, "can_execute": bool, "can_draft": bool}``
            dicts, one per procedure.
        """
        from oc_apprentice_worker.procedure_verifier import ProcedureVerifier

        verifier = ProcedureVerifier(kb)
        results: list[dict] = []

        for proc in self._scenario.procedures:
            slug = proc.get("id", proc.get("slug", "unknown"))
            pf = verifier.preflight(slug)
            results.append({
                "slug": pf.slug,
                "can_execute": pf.can_execute,
                "can_draft": pf.can_draft,
            })

        return results

    # ------------------------------------------------------------------
    # 6. Export parity
    # ------------------------------------------------------------------

    def run_export_parity(self, kb: Any, tmp_path: Path) -> dict:
        """Verify that all four export adapters produce output with required fields.

        For each procedure, calls ``write_procedure()`` on every adapter
        and checks that the output file exists and contains the required
        fields (title, steps).

        Args:
            kb: A ``KnowledgeBase`` instance.
            tmp_path: Temporary directory for adapter output.

        Returns:
            Dict with ``field_coverage_parity`` (float 0-1),
            ``adapters_checked`` (int), and ``procedures_checked`` (int).
        """
        from oc_apprentice_worker.openclaw_writer import OpenClawWriter
        from oc_apprentice_worker.skill_md_writer import SkillMdWriter
        from oc_apprentice_worker.claude_skill_writer import ClaudeSkillWriter
        from oc_apprentice_worker.generic_writer import GenericWriter

        # Each adapter gets its own subdirectory
        adapters = [
            OpenClawWriter(workspace_dir=tmp_path / "openclaw"),
            SkillMdWriter(workspace_dir=tmp_path / "skillmd"),
            ClaudeSkillWriter(skills_dir=tmp_path / "claude"),
            GenericWriter(output_dir=tmp_path / "generic"),
        ]

        procedures = self._scenario.procedures
        if not procedures:
            return {
                "field_coverage_parity": 1.0,
                "adapters_checked": len(adapters),
                "procedures_checked": 0,
            }

        total_checks = 0
        passed_checks = 0

        for proc in procedures:
            for adapter in adapters:
                total_checks += 1
                try:
                    out_path = adapter.write_procedure(proc)
                    if out_path.exists() and out_path.stat().st_size > 0:
                        content = out_path.read_text(encoding="utf-8")
                        # Check for required fields: title and steps
                        title = proc.get("title", "")
                        has_title = title and (
                            title in content or title.lower() in content.lower()
                        )
                        has_steps = (
                            "step" in content.lower()
                            or "steps" in content.lower()
                        )
                        if has_title and has_steps:
                            passed_checks += 1
                        else:
                            logger.debug(
                                "Export parity: %s missing title/steps for %s",
                                type(adapter).__name__,
                                proc.get("id", "?"),
                            )
                    else:
                        logger.debug(
                            "Export parity: %s produced empty file for %s",
                            type(adapter).__name__,
                            proc.get("id", "?"),
                        )
                except Exception as exc:
                    logger.warning(
                        "Export parity: %s failed for %s: %s",
                        type(adapter).__name__,
                        proc.get("id", "?"),
                        exc,
                    )

        parity = passed_checks / total_checks if total_checks > 0 else 0.0

        return {
            "field_coverage_parity": round(parity, 4),
            "adapters_checked": len(adapters),
            "procedures_checked": len(procedures),
        }

    # ------------------------------------------------------------------
    # 7. Recurrence
    # ------------------------------------------------------------------

    def run_recurrence(self, kb: Any) -> list[dict]:
        """Detect recurrence patterns from daily summaries stored in the KB.

        If the scenario includes daily summaries, saves them to the KB
        (if not already saved by ``setup_kb()``) and runs
        ``PatternDetector.detect_recurrence()``.

        Args:
            kb: A ``KnowledgeBase`` instance with daily summaries loaded.

        Returns:
            List of pattern dicts (serialized ``RecurrencePattern`` objects).
        """
        from oc_apprentice_worker.pattern_detector import PatternDetector

        if not self._scenario.daily_summaries:
            return []

        # Ensure summaries are in the KB (idempotent)
        for summary in self._scenario.daily_summaries:
            date = summary.get("date", "unknown")
            kb.save_daily_summary(date, summary)

        detector = PatternDetector(kb, min_observations=2)
        patterns = detector.detect_recurrence()

        return [
            {
                "procedure_slug": p.procedure_slug,
                "pattern": p.pattern,
                "confidence": round(p.confidence, 4),
                "day": p.day,
                "time": p.time,
                "avg_duration_minutes": p.avg_duration_minutes,
                "observations": p.observations,
            }
            for p in patterns
        ]

    # ------------------------------------------------------------------
    # 8. Continuity
    # ------------------------------------------------------------------

    def run_continuity(self, tmp_path: Path) -> list[dict]:
        """Run segmentation + continuity tracking with mock embeddings.

        Calls ``run_segmentation()`` to produce segments, then builds a
        continuity graph via ``ContinuityTracker.build_graph()``.  For
        each resulting span, resolves ``segment_ids`` back to segment
        objects and collects the constituent ``event_ids``.

        Args:
            tmp_path: Temporary directory for the continuity KB.

        Returns:
            List of span dicts with ``span_id``, ``event_ids`` (resolved),
            ``goal_summary``, and ``state``.
        """
        from oc_apprentice_worker.continuity_tracker import ContinuityTracker
        from oc_apprentice_worker.knowledge_base import KnowledgeBase

        # 1. Run segmentation to get segments
        seg_result = self.run_segmentation()

        # 2. Create a temp KB for the tracker
        kb = KnowledgeBase(root=tmp_path)
        kb.ensure_structure()

        # 3. Build the continuity graph
        tracker = ContinuityTracker(kb)
        spans = tracker.build_graph(seg_result.segments, existing_spans=[])

        # 4. Build a lookup from segment_id -> segment object
        seg_by_id: dict[str, Any] = {}
        for seg in seg_result.segments:
            seg_by_id[seg.segment_id] = seg

        # 5. For each span, resolve segment_ids -> event_ids
        result: list[dict] = []
        for span in spans:
            event_ids: list[str] = []
            for sid in span.segments:
                seg = seg_by_id.get(sid)
                if seg is not None:
                    for frame in seg.frames:
                        if frame.event_id and frame.event_id not in event_ids:
                            event_ids.append(frame.event_id)
            result.append({
                "span_id": span.span_id,
                "event_ids": event_ids,
                "goal_summary": span.goal_summary,
                "state": span.state,
            })

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_events_for_segmenter(self) -> list[dict]:
        """Convert scenario events so that JSON fields are strings.

        ``AnnotatedFrame.from_event()`` handles both dict and string
        forms, but for faithful simulation of the real DB-to-worker
        path we JSON-encode any dict-valued fields.
        """
        prepared: list[dict] = []
        for event in self._scenario.events:
            ev = dict(event)
            for json_field in (
                "scene_annotation_json",
                "frame_diff_json",
                "window_json",
                "kind_json",
                "metadata_json",
            ):
                val = ev.get(json_field)
                if isinstance(val, dict):
                    ev[json_field] = json.dumps(val)
            prepared.append(ev)
        return prepared

    def _prepare_events_for_daily_processor(self) -> list[dict]:
        """Convert scenario events for ``DailyBatchProcessor``.

        The daily processor's ``_build_activity_stream()`` passes events
        through ``parse_annotation()`` from ``event_helpers``, which
        handles both dict and JSON-string ``scene_annotation_json``.
        We JSON-encode dict-valued fields for consistency.
        """
        return self._prepare_events_for_segmenter()

    def _collect_ordered_embeddings(self) -> list[list[float]]:
        """Collect pre-stored embeddings from events that have annotations.

        The segmenter creates ``AnnotatedFrame`` objects only for events
        with a valid ``scene_annotation_json``.  Embeddings are requested
        in that same order, so we collect them by iterating events and
        skipping those without annotations.
        """
        embeddings: list[list[float]] = []
        for event in self._scenario.events:
            ann = event.get("scene_annotation_json")
            if ann is None:
                continue

            # Verify the annotation is parseable
            if isinstance(ann, str):
                try:
                    parsed = json.loads(ann)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(parsed, dict):
                    continue
            elif not isinstance(ann, dict):
                continue

            emb = event.get("embedding", [])
            embeddings.append(emb if isinstance(emb, list) else [])

        return embeddings

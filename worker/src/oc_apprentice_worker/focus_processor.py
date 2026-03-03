"""Focus session processor for the v2 VLM annotation pipeline.

Orchestrates the end-to-end processing of a completed focus recording
session:  ensures all events are annotated → ensures frame diffs exist →
collects the annotated timeline → generates a semantic SOP.

This replaces the v1 focus processing path (episode_builder → translator
→ scorer → sop_inducer.induce_from_focus_session) when v2 annotation is
enabled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from oc_apprentice_worker.scene_annotator import SceneAnnotator
from oc_apprentice_worker.frame_differ import FrameDiffer
from oc_apprentice_worker.sop_generator import SOPGenerator, GeneratedSOP

logger = logging.getLogger(__name__)


class FocusProcessor:
    """Process completed focus recording sessions through the v2 pipeline.

    Usage::

        processor = FocusProcessor(annotator, differ, sop_generator)
        result = processor.process_session(db, session_id, title, events,
                                           screenshots_dir=...)
    """

    def __init__(
        self,
        annotator: SceneAnnotator,
        differ: FrameDiffer,
        sop_generator: SOPGenerator,
    ) -> None:
        self.annotator = annotator
        self.differ = differ
        self.sop_generator = sop_generator

    def process_session(
        self,
        db: object,
        session_id: str,
        title: str,
        events: list[dict],
        *,
        screenshots_dir: str | Path = "",
    ) -> GeneratedSOP:
        """Process a completed focus session end-to-end.

        Steps:
          1. Ensure all events have scene annotations (run VLM if needed)
          2. Ensure all annotated events have frame diffs
          3. Collect the full annotated timeline
          4. Generate a semantic SOP via the SOP generator

        Args:
            db: WorkerDB instance.
            session_id: Focus session UUID.
            title: User-provided task name.
            events: Raw event dicts from the DB (already filtered by session_id).
            screenshots_dir: Directory containing screenshot JPEGs.

        Returns:
            GeneratedSOP with the template dict (or error).
        """
        if not events:
            return GeneratedSOP(
                sop={}, success=False, error="No events in focus session"
            )

        logger.info(
            "Focus v2: processing session '%s' (%s) — %d events",
            title, session_id, len(events),
        )

        # Step 1: Ensure all events are annotated
        ann_stats = self._ensure_annotations(db, events, screenshots_dir)
        logger.info(
            "Focus v2 annotation: %d annotated, %d skipped, %d failed",
            ann_stats["annotated"],
            ann_stats["skipped"],
            ann_stats["failed"],
        )

        # Step 2: Ensure frame diffs exist
        diff_stats = self._ensure_diffs(db, events)
        logger.info(
            "Focus v2 diffs: %d action, %d edge, %d failed",
            diff_stats["diffs"],
            diff_stats["edge_cases"],
            diff_stats["failed"],
        )

        # Step 3: Collect the annotated timeline
        timeline = self._collect_timeline(db, events)
        if not timeline:
            return GeneratedSOP(
                sop={},
                success=False,
                error="No annotated frames produced for focus session",
            )

        logger.info(
            "Focus v2 timeline: %d annotated frames (of %d total events)",
            len(timeline), len(events),
        )

        # Step 4: Generate SOP
        result = self.sop_generator.generate_from_focus(timeline, title)
        return result

    # ------------------------------------------------------------------
    # Step 1: Annotation
    # ------------------------------------------------------------------

    def _ensure_annotations(
        self,
        db: object,
        events: list[dict],
        screenshots_dir: str | Path,
    ) -> dict:
        """Annotate any un-annotated events in the focus session.

        Focus events are processed with priority — we annotate them
        immediately rather than waiting for the background loop.
        """
        stats = {"annotated": 0, "skipped": 0, "failed": 0, "already_done": 0}

        for event in events:
            event_id = event.get("id", "unknown")
            status = event.get("annotation_status", "pending")

            # Already annotated
            if status in ("completed", "skipped", "missing_screenshot"):
                stats["already_done"] += 1
                continue

            # Get sliding window context from recent annotations
            timestamp = event.get("timestamp", "")
            recent = db.get_recent_annotations(  # type: ignore[union-attr]
                before_timestamp=timestamp,
                limit=self.annotator.config.sliding_window_size,
                max_age_seconds=self.annotator.config.sliding_window_max_age_sec,
            )

            # For focus sessions, disable stale-frame skipping —
            # every frame matters in a single-demonstration recording.
            result = self.annotator.annotate_event(
                event,
                recent_annotations=recent,
                artifact_dir=str(screenshots_dir),
                skip_stale_check=True,
            )

            if result.status == "completed" and result.annotation:
                db.save_annotation(  # type: ignore[union-attr]
                    event_id,
                    json.dumps(result.annotation),
                    status="completed",
                )
                stats["annotated"] += 1
            elif result.status in ("skipped", "missing_screenshot"):
                db.save_annotation(  # type: ignore[union-attr]
                    event_id, "", status=result.status
                )
                stats["skipped"] += 1
            else:
                db.save_annotation(  # type: ignore[union-attr]
                    event_id, "", status="failed"
                )
                stats["failed"] += 1

        return stats

    # ------------------------------------------------------------------
    # Step 2: Frame diffs
    # ------------------------------------------------------------------

    def _ensure_diffs(self, db: object, events: list[dict]) -> dict:
        """Compute frame diffs for annotated events that lack them."""
        stats = {"diffs": 0, "edge_cases": 0, "failed": 0, "already_done": 0}

        # Re-read events to get fresh annotation data
        annotated_events = []
        for event in events:
            event_id = event.get("id", "unknown")
            # Refresh from DB to get annotation
            fresh = db.get_event_by_id(event_id)  # type: ignore[union-attr]
            if fresh and fresh.get("annotation_status") == "completed":
                annotated_events.append(fresh)

        for i, event in enumerate(annotated_events):
            event_id = event.get("id", "unknown")

            # Already has diff
            if event.get("frame_diff_json"):
                stats["already_done"] += 1
                continue

            timestamp = event.get("timestamp", "")

            # Get previous annotated event
            if i == 0:
                # First frame — use DB predecessor (may be outside focus session)
                prev = db.get_annotation_before(timestamp)  # type: ignore[union-attr]
            else:
                prev = annotated_events[i - 1]

            if prev is None:
                db.save_frame_diff(  # type: ignore[union-attr]
                    event_id, json.dumps({"diff_type": "first_frame"})
                )
                stats["edge_cases"] += 1
                continue

            result = self.differ.diff_pair(prev, event)
            db.save_frame_diff(  # type: ignore[union-attr]
                event_id, json.dumps(result.diff)
            )

            diff_type = result.diff.get("diff_type", "unknown")
            if diff_type == "action":
                stats["diffs"] += 1
            elif diff_type == "diff_failed":
                stats["failed"] += 1
            else:
                stats["edge_cases"] += 1

        return stats

    # ------------------------------------------------------------------
    # Step 3: Collect timeline
    # ------------------------------------------------------------------

    def _collect_timeline(
        self,
        db: object,
        events: list[dict],
    ) -> list[dict]:
        """Collect the annotated timeline for SOP generation.

        Returns a list of dicts, each containing:
        - annotation: parsed scene annotation dict
        - diff: parsed frame diff dict (or None)
        - timestamp: ISO timestamp string
        - app: application name
        """
        timeline = []

        for event in events:
            event_id = event.get("id", "unknown")
            # Re-read to get latest annotation + diff data
            fresh = db.get_event_by_id(event_id)  # type: ignore[union-attr]
            if not fresh or fresh.get("annotation_status") != "completed":
                continue

            ann_json = fresh.get("scene_annotation_json", "")
            if not ann_json:
                continue

            try:
                annotation = json.loads(ann_json)
            except (json.JSONDecodeError, TypeError):
                continue

            diff = None
            diff_json = fresh.get("frame_diff_json", "")
            if diff_json:
                try:
                    diff = json.loads(diff_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            timeline.append({
                "annotation": annotation,
                "diff": diff,
                "timestamp": fresh.get("timestamp", ""),
                "app": annotation.get("app", ""),
                "event_id": event_id,
            })

        return timeline

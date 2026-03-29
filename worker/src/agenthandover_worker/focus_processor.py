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

from agenthandover_worker.scene_annotator import SceneAnnotator
from agenthandover_worker.frame_differ import FrameDiffer
from agenthandover_worker.sop_generator import SOPGenerator, GeneratedSOP

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
        behavioral_synthesizer: "BehavioralSynthesizer | None" = None,
    ) -> None:
        self.annotator = annotator
        self.differ = differ
        self.sop_generator = sop_generator
        self.behavioral_synthesizer = behavioral_synthesizer

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

        # Step 0a: Filter out AgentHandover's own events
        events = self._filter_own_events(events)
        if not events:
            return GeneratedSOP(
                sop={}, success=False, error="All events were self-referential (AgentHandover's own UI)"
            )

        # Step 0b: Attach clipboard context to adjacent screenshots
        events = self._attach_clipboard_context(events)

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

        # Step 4: Behavioral pre-analysis (optional, informs SOP generation)
        behavioral_context = ""
        if self.behavioral_synthesizer is not None:
            try:
                from agenthandover_worker.sop_generator import _generate_slug
                pre_procedure = {"title": title, "steps": [], "source": "focus"}
                timeline_obs = [
                    [{"action": f.get("annotation", {}).get("task_context", {}).get("what_doing", "")}
                     for f in timeline if f.get("annotation")]
                ]
                insights = self.behavioral_synthesizer.synthesize(
                    _generate_slug(title), pre_procedure, timeline_obs,
                    force=True,
                )
                if insights.strategy:
                    behavioral_context = f"User intent: {insights.strategy}\n"
                    if insights.selection_criteria:
                        behavioral_context += (
                            "Selection criteria: "
                            + ", ".join(c.criterion for c in insights.selection_criteria)
                            + "\n"
                        )
                    logger.info(
                        "Focus v2 session '%s': behavioral pre-analysis complete",
                        title,
                    )
            except Exception:
                logger.debug(
                    "Behavioral pre-analysis failed for '%s'", title, exc_info=True,
                )

        # Step 5: Generate SOP (with behavioral context if available)
        result = self.sop_generator.generate_from_focus(
            timeline, title,
            behavioral_context=behavioral_context,
        )
        return result


    # ------------------------------------------------------------------
    # Step 0a: Filter own events
    # ------------------------------------------------------------------

    _OWN_APP_PATTERNS = ("agenthandover", "agenthandoverapp", "agenthandover_worker")
    _OWN_TITLE_PATTERNS = ("agenthandover",)

    def _filter_own_events(self, events: list[dict]) -> list[dict]:
        """Remove events generated by AgentHandover's own UI.

        Filters events where the app name, bundle ID, or window title
        references AgentHandover or its sub-processes, so the SOP only
        captures the user's actual task.
        """
        original_count = len(events)
        filtered: list[dict] = []

        for event in events:
            if self._is_own_event(event):
                continue
            filtered.append(event)

        removed = original_count - len(filtered)
        if removed:
            logger.info(
                "Focus v2: filtered %d self-referential event(s) "
                "(%d remaining of %d)",
                removed, len(filtered), original_count,
            )

        return filtered

    def _is_own_event(self, event: dict) -> bool:
        """Check whether an event originates from AgentHandover's own UI."""
        window_json_raw = event.get("window_json", "")
        if not window_json_raw:
            return False

        try:
            window = (
                json.loads(window_json_raw)
                if isinstance(window_json_raw, str)
                else window_json_raw
            )
        except (json.JSONDecodeError, TypeError):
            return False

        if not isinstance(window, dict):
            return False

        app_name = str(window.get("app_name", "")).lower()
        bundle_id = str(window.get("app_bundle_id", "")).lower()
        title = str(window.get("title", "")).lower()

        # Check app name / bundle ID
        for pattern in self._OWN_APP_PATTERNS:
            if pattern in app_name or pattern in bundle_id:
                return True

        # Check window title
        for pattern in self._OWN_TITLE_PATTERNS:
            if pattern in title:
                return True

        return False

    # ------------------------------------------------------------------
    # Step 0b: Attach clipboard context
    # ------------------------------------------------------------------

    def _attach_clipboard_context(self, events: list[dict]) -> list[dict]:
        """Attach clipboard metadata to the nearest screenshot event.

        ClipboardChange events have no screenshots and cannot be annotated
        by the VLM.  This method:
        1. Identifies ClipboardChange events via ``kind_json``.
        2. For each, finds the nearest DwellSnapshot event (preferring the
           one immediately AFTER; if none exists, uses the one immediately
           BEFORE).
        3. Stores clipboard info (content_types, byte_size) keyed by the
           target event's ID in ``self._clipboard_map``.
        4. Removes ClipboardChange events from the list.

        Returns the cleaned event list (DwellSnapshots only).
        """
        self._clipboard_map: dict[str, dict] = {}

        # Separate clipboard events from the rest
        clipboard_events: list[tuple[int, dict]] = []
        non_clipboard: list[tuple[int, dict]] = []

        for i, event in enumerate(events):
            kind = self._extract_event_kind(event)
            if kind == "ClipboardChange":
                clipboard_events.append((i, event))
            else:
                non_clipboard.append((i, event))

        if not clipboard_events:
            return events  # nothing to do

        # Build index of DwellSnapshot positions for quick lookup
        dwell_positions: list[tuple[int, str]] = []
        for orig_idx, evt in non_clipboard:
            kind = self._extract_event_kind(evt)
            if kind == "DwellSnapshot":
                dwell_positions.append((orig_idx, evt.get("id", "")))

        for clip_idx, clip_event in clipboard_events:
            clip_meta = self._extract_clipboard_meta(clip_event)
            if not clip_meta:
                continue

            # Find nearest DwellSnapshot: prefer immediately after, else before
            best_id: str | None = None
            best_distance = float("inf")
            prefer_after = True

            for dwell_orig_idx, dwell_id in dwell_positions:
                distance = dwell_orig_idx - clip_idx
                abs_distance = abs(distance)

                if abs_distance >= best_distance:
                    # If current best is "after" and this is "before" with
                    # same distance, keep "after"
                    if abs_distance == best_distance and not prefer_after and distance > 0:
                        best_id = dwell_id
                        best_distance = abs_distance
                        prefer_after = True
                    continue

                # Prefer "after" (positive distance) over "before"
                if distance > 0:
                    best_id = dwell_id
                    best_distance = abs_distance
                    prefer_after = True
                elif best_id is None or (not prefer_after and abs_distance < best_distance):
                    best_id = dwell_id
                    best_distance = abs_distance
                    prefer_after = False

            if best_id:
                self._clipboard_map[best_id] = clip_meta

        logger.info(
            "Focus v2: attached %d clipboard context(s) to snapshots, "
            "removed %d ClipboardChange event(s)",
            len(self._clipboard_map), len(clipboard_events),
        )

        # Return only non-clipboard events (preserving order)
        return [evt for _, evt in non_clipboard]

    @staticmethod
    def _extract_event_kind(event: dict) -> str:
        """Extract the event kind name from ``kind_json``."""
        kind_json = event.get("kind_json", "")
        if not kind_json:
            return ""
        try:
            parsed = (
                json.loads(kind_json)
                if isinstance(kind_json, str)
                else kind_json
            )
        except (json.JSONDecodeError, TypeError):
            return ""
        if isinstance(parsed, dict) and parsed:
            return next(iter(parsed))
        return ""

    @staticmethod
    def _extract_clipboard_meta(event: dict) -> dict | None:
        """Extract clipboard metadata (content_types, byte_size) from an event."""
        metadata_json = event.get("metadata_json", "")
        if not metadata_json:
            return None
        try:
            metadata = (
                json.loads(metadata_json)
                if isinstance(metadata_json, str)
                else metadata_json
            )
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(metadata, dict):
            return None
        content_types = metadata.get("content_types", [])
        byte_size = metadata.get("byte_size", 0)
        if not content_types and not byte_size:
            return None
        return {"content_types": content_types, "byte_size": byte_size}

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
                # First frame in focus session — always use first_frame marker.
                # Do NOT look up a DB predecessor: it may come from unrelated
                # pre-session activity and would pollute the first step's diff.
                prev = None
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

            # Try to find a DOM snapshot near this event
            dom_nodes = None
            location = annotation.get("location", "")
            if location and location.startswith("http"):
                dom_snaps = db.get_dom_snapshots_near_timestamp(  # type: ignore[union-attr]
                    fresh.get("timestamp", ""),
                    location,
                    tolerance_sec=5.0,
                )
                if dom_snaps:
                    dom_nodes = dom_snaps[0].get("nodes")

            # Look up clipboard context attached in Step 0b
            clipboard_ctx = (
                self._clipboard_map.get(event_id)
                if hasattr(self, "_clipboard_map")
                else None
            )

            timeline.append({
                "annotation": annotation,
                "diff": diff,
                "dom_nodes": dom_nodes,
                "timestamp": fresh.get("timestamp", ""),
                "app": annotation.get("app", ""),
                "event_id": event_id,
                "clipboard_context": clipboard_ctx,
            })

        # Safety net: the first frame in a focus session must never carry a
        # diff — it is the clean starting point of the SOP.  Any diff stored
        # for it (e.g. a "first_frame" marker or a stale pre-session diff)
        # would pollute the first step with irrelevant context.
        if timeline:
            timeline[0]["diff"] = None

        return timeline

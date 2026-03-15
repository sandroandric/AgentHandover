"""Continuity graph tracker for Phase 2 cross-segment linking.

Tracks how task segments relate to each other over time by building a
graph of :class:`ContinuityEdge` relationships between segments, grouped
into :class:`ContinuitySpan` objects that represent a single logical
activity thread potentially spanning many segments and time gaps.

Persistence: ``{kb_root}/observations/continuity_spans.json``.

Decision matrix for classifying relationships (first match wins):

    CONTINUE  — gap < 60s AND emb_sim >= 0.70 AND app_overlap >= 0.5
    RESUME    — gap < 30min AND emb_sim >= 0.60
    RESUME    — gap < 4h AND emb_sim >= 0.75 AND app_overlap >= 0.5
    BRANCH    — emb_sim in [0.40, 0.60) AND app_overlap >= 0.3
    RESTART   — gap > 24h AND emb_sim >= 0.70
    NEW_TASK  — default (confidence = 1.0 - max(emb_sim, app_overlap))
"""

from __future__ import annotations

import collections
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.task_segmenter import _cosine_similarity, _timestamp_to_epoch

if TYPE_CHECKING:
    from oc_apprentice_worker.task_segmenter import TaskSegment

logger = logging.getLogger(__name__)

# Persistence file relative to KB observations directory
_CONTINUITY_FILE = "continuity_spans.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ContinuityType(str, Enum):
    """Type of continuity relationship between two segments."""

    CONTINUE = "continue"   # Same task, no gap
    RESUME = "resume"       # Same task after gap (<30min or <4h with high sim)
    BRANCH = "branch"       # Related but diverged
    RESTART = "restart"     # Same goal, fresh start (>24h)
    NEW_TASK = "new_task"   # Completely new


@dataclass
class ContinuityEdge:
    """A directed relationship between two segments within a span."""

    from_segment_id: str
    to_segment_id: str
    continuity_type: ContinuityType
    confidence: float
    reasoning: str
    gap_seconds: int
    embedding_similarity: float
    app_overlap: float


@dataclass
class ContinuitySpan:
    """A logical activity thread grouping related segments over time.

    A span may contain segments separated by minutes, hours, or even days,
    as long as the continuity tracker determines they belong to the same
    logical line of work.
    """

    span_id: str
    goal_summary: str
    continuity_confidence: float
    segments: list[str] = field(default_factory=list)
    edges: list[ContinuityEdge] = field(default_factory=list)
    state: str = "active"
    activity_type: str = ""
    matched_procedure_candidates: list = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    total_duration_seconds: int = 0
    interruption_count: int = 0
    apps_involved: list[str] = field(default_factory=list)
    representative_embedding: list[float] = field(default_factory=list)
    parent_span_id: str | None = None


# ---------------------------------------------------------------------------
# Time gap thresholds (seconds)
# ---------------------------------------------------------------------------

_GAP_CONTINUE = 60         # < 60s
_GAP_RESUME_SHORT = 1800   # < 30 min
_GAP_RESUME_LONG = 14400   # < 4 h
_GAP_RESTART = 86400       # > 24 h
_GAP_PAUSED = 14400        # > 4 h  -> state = "paused"
_GAP_COMPLETED = 86400     # > 24 h -> state = "completed"


# ---------------------------------------------------------------------------
# ContinuityTracker
# ---------------------------------------------------------------------------

class ContinuityTracker:
    """Builds and maintains a continuity graph across task segments.

    The graph connects :class:`TaskSegment` objects (produced by
    :class:`~oc_apprentice_worker.task_segmenter.TaskSegmenter`) into
    :class:`ContinuitySpan` threads that track the same logical activity
    across interruptions, pauses, and session boundaries.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        matcher: "ProcedureMatcher | None" = None,
        similarity_threshold: float = 0.60,
        uncertain_threshold: float = 0.50,
    ) -> None:
        self._kb = kb
        self._matcher = matcher
        self._similarity_threshold = similarity_threshold
        self._uncertain_threshold = uncertain_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(
        self,
        segments: list[TaskSegment],
        existing_spans: list[ContinuitySpan] | None = None,
    ) -> list[ContinuitySpan]:
        """Build or extend a continuity graph from new segments.

        For each segment (processed in chronological order by start_time):

        1. Compute a representative embedding (average of frame embeddings).
        2. Compare against all active/paused/uncertain spans.
        3. Pick the best-matching span (highest confidence).
        4. If best confidence >= ``uncertain_threshold`` AND type != NEW_TASK:
           - CONTINUE / RESUME / RESTART: extend the existing span.
           - BRANCH: create a new span with ``parent_span_id`` pointing
             to the matched span.
        5. Otherwise: create a new span (state ``"uncertain"`` if there
           was a close-but-insufficient candidate).

        After all segments are processed the method updates span states
        based on elapsed time and optionally matches procedures.

        Args:
            segments: New :class:`TaskSegment` objects to integrate.
            existing_spans: Previously persisted spans to extend.
                If ``None``, loads from disk via :meth:`load_spans`.

        Returns:
            The full list of spans (both existing and newly created).
        """
        if existing_spans is None:
            spans = self.load_spans()
        else:
            spans = list(existing_spans)

        # Process segments in chronological order
        sorted_segments = sorted(
            segments, key=lambda s: _timestamp_to_epoch(s.start_time)
        )

        for segment in sorted_segments:
            seg_embedding = self._compute_representative_embedding(segment)

            # Find best matching span among eligible states
            best_edge: ContinuityEdge | None = None
            best_span: ContinuitySpan | None = None

            for span in spans:
                if span.state not in ("active", "paused", "uncertain"):
                    continue

                # Temporarily attach the segment embedding for classification
                edge = self.classify_relationship(segment, span)

                if best_edge is None or edge.confidence > best_edge.confidence:
                    best_edge = edge
                    best_span = span

            if (
                best_edge is not None
                and best_span is not None
                and best_edge.confidence >= self._uncertain_threshold
                and best_edge.continuity_type != ContinuityType.NEW_TASK
            ):
                if best_edge.continuity_type == ContinuityType.BRANCH:
                    # Create a child span linked to the parent
                    new_span = self._create_span_from_segment(
                        segment, parent_span_id=best_span.span_id,
                    )
                    # Override embedding with the computed one
                    if seg_embedding:
                        new_span.representative_embedding = list(seg_embedding)
                    spans.append(new_span)
                else:
                    # CONTINUE, RESUME, or RESTART: extend the existing span
                    self._update_span_from_segment(best_span, segment, best_edge)
            else:
                # No good match — create a new span
                new_span = self._create_span_from_segment(segment)
                if seg_embedding:
                    new_span.representative_embedding = list(seg_embedding)

                # Mark as uncertain if there was a close candidate
                if (
                    best_edge is not None
                    and best_edge.confidence > 0.0
                    and best_edge.continuity_type != ContinuityType.NEW_TASK
                ):
                    new_span.state = "uncertain"

                spans.append(new_span)

        # Post-processing: update states based on current time
        now_iso = datetime.now(timezone.utc).isoformat()
        self._update_span_states(spans, now_iso)

        # Match procedures if a matcher is available
        if self._matcher is not None:
            for span in spans:
                if not span.matched_procedure_candidates:
                    try:
                        candidates = self._matcher.match(span.goal_summary)
                        span.matched_procedure_candidates = candidates
                    except Exception:
                        logger.debug(
                            "Procedure matching failed for span %s",
                            span.span_id,
                            exc_info=True,
                        )

        return spans

    def classify_relationship(
        self,
        segment: TaskSegment,
        span: ContinuitySpan,
    ) -> ContinuityEdge:
        """Classify the continuity relationship between a segment and a span.

        Uses embedding cosine similarity, app overlap (Jaccard), and time
        gap to determine the relationship type.

        Args:
            segment: The new segment to classify.
            span: The existing span to compare against.

        Returns:
            A :class:`ContinuityEdge` describing the relationship.
        """
        seg_embedding = self._compute_representative_embedding(segment)
        emb_sim = _cosine_similarity(seg_embedding, span.representative_embedding)
        app_overlap = self._compute_app_overlap(
            segment.apps_involved, span.apps_involved,
        )

        # Compute gap in seconds
        seg_start_epoch = _timestamp_to_epoch(segment.start_time)
        span_last_epoch = _timestamp_to_epoch(span.last_seen)
        gap = int(seg_start_epoch - span_last_epoch) if seg_start_epoch and span_last_epoch else 0
        if gap < 0:
            gap = 0

        # Decision matrix (first match wins)
        continuity_type: ContinuityType
        confidence: float
        reasoning: str

        if gap < _GAP_CONTINUE and emb_sim >= 0.70 and app_overlap >= 0.5:
            continuity_type = ContinuityType.CONTINUE
            confidence = min(emb_sim, app_overlap)
            reasoning = (
                f"Small gap ({gap}s), high embedding similarity ({emb_sim:.2f}), "
                f"strong app overlap ({app_overlap:.2f})"
            )
        elif gap < _GAP_RESUME_SHORT and emb_sim >= 0.60:
            continuity_type = ContinuityType.RESUME
            confidence = emb_sim
            reasoning = (
                f"Short gap ({gap}s < 30min), sufficient embedding similarity ({emb_sim:.2f})"
            )
        elif gap < _GAP_RESUME_LONG and emb_sim >= 0.75 and app_overlap >= 0.5:
            continuity_type = ContinuityType.RESUME
            confidence = min(emb_sim, app_overlap)
            reasoning = (
                f"Medium gap ({gap}s < 4h), high embedding similarity ({emb_sim:.2f}), "
                f"strong app overlap ({app_overlap:.2f})"
            )
        elif 0.40 <= emb_sim < 0.60 and app_overlap >= 0.3:
            continuity_type = ContinuityType.BRANCH
            confidence = (emb_sim + app_overlap) / 2.0
            reasoning = (
                f"Moderate embedding similarity ({emb_sim:.2f}), "
                f"some app overlap ({app_overlap:.2f}) — likely branched"
            )
        elif gap > _GAP_RESTART and emb_sim >= 0.70:
            continuity_type = ContinuityType.RESTART
            confidence = emb_sim
            reasoning = (
                f"Large gap ({gap}s > 24h) but high embedding similarity ({emb_sim:.2f}) "
                f"— same goal, fresh start"
            )
        else:
            continuity_type = ContinuityType.NEW_TASK
            confidence = 1.0 - max(emb_sim, app_overlap)
            reasoning = (
                f"No match: emb_sim={emb_sim:.2f}, app_overlap={app_overlap:.2f}, "
                f"gap={gap}s"
            )

        return ContinuityEdge(
            from_segment_id=span.segments[-1] if span.segments else "",
            to_segment_id=segment.segment_id,
            continuity_type=continuity_type,
            confidence=confidence,
            reasoning=reasoning,
            gap_seconds=gap,
            embedding_similarity=emb_sim,
            app_overlap=app_overlap,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _compute_representative_embedding(
        self, segment: TaskSegment,
    ) -> list[float]:
        """Compute the average embedding across all frames in a segment.

        Returns an empty list if no frames carry embeddings.
        """
        valid_embeddings = [
            f.embedding for f in segment.frames if f.embedding
        ]
        if not valid_embeddings:
            return []

        dim = len(valid_embeddings[0])
        avg = [0.0] * dim
        for emb in valid_embeddings:
            if len(emb) != dim:
                continue
            for i in range(dim):
                avg[i] += emb[i]

        n = len(valid_embeddings)
        return [v / n for v in avg]

    def _compute_app_overlap(
        self, apps_a: list[str], apps_b: list[str],
    ) -> float:
        """Jaccard similarity between two app lists (case-insensitive).

        Returns 0.0 if both lists are empty.
        """
        set_a = {a.lower() for a in apps_a}
        set_b = {b.lower() for b in apps_b}
        if not set_a and not set_b:
            return 0.0
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Span mutation helpers
    # ------------------------------------------------------------------

    def _update_span_from_segment(
        self,
        span: ContinuitySpan,
        segment: TaskSegment,
        edge: ContinuityEdge,
    ) -> None:
        """Extend a span with a new segment.

        Updates: segments list, edges list, last_seen, total_duration,
        interruption_count, apps_involved (union), representative_embedding
        (running average), and continuity_confidence (running minimum).
        """
        n_existing = len(span.segments)

        # Add segment and edge
        span.segments.append(segment.segment_id)
        span.edges.append(edge)

        # Update last_seen
        if segment.end_time:
            span.last_seen = segment.end_time

        # Update total duration (add this segment's internal duration)
        seg_duration = self._segment_duration_seconds(segment)
        span.total_duration_seconds += seg_duration

        # Update interruption count (any non-CONTINUE edge is an interruption)
        if edge.continuity_type != ContinuityType.CONTINUE:
            span.interruption_count += 1

        # Update apps_involved (union, preserving order)
        existing_apps_lower = {a.lower() for a in span.apps_involved}
        for app in segment.apps_involved:
            if app.lower() not in existing_apps_lower:
                span.apps_involved.append(app)
                existing_apps_lower.add(app.lower())

        # Update representative embedding (running average)
        seg_embedding = self._compute_representative_embedding(segment)
        if seg_embedding and span.representative_embedding:
            dim = len(span.representative_embedding)
            if len(seg_embedding) == dim and n_existing > 0:
                span.representative_embedding = [
                    (span.representative_embedding[i] * n_existing + seg_embedding[i])
                    / (n_existing + 1)
                    for i in range(dim)
                ]
        elif seg_embedding and not span.representative_embedding:
            span.representative_embedding = list(seg_embedding)

        # Update continuity confidence (running minimum)
        span.continuity_confidence = min(
            span.continuity_confidence, edge.confidence,
        )

        # Update activity type (most common across all segment frames)
        self._update_activity_type(span, segment)

        # If the span was paused or uncertain, reactivate it
        if span.state in ("paused", "uncertain"):
            span.state = "active"

    def _create_span_from_segment(
        self,
        segment: TaskSegment,
        parent_span_id: str | None = None,
    ) -> ContinuitySpan:
        """Create a new :class:`ContinuitySpan` from a single segment."""
        seg_embedding = self._compute_representative_embedding(segment)
        seg_duration = self._segment_duration_seconds(segment)

        # Determine activity type from frame annotations
        activity_type = self._most_common_activity_type(segment)

        return ContinuitySpan(
            span_id=str(uuid.uuid4()),
            goal_summary=segment.task_label,
            continuity_confidence=1.0,
            segments=[segment.segment_id],
            edges=[],
            state="active",
            activity_type=activity_type,
            matched_procedure_candidates=[],
            first_seen=segment.start_time,
            last_seen=segment.end_time,
            total_duration_seconds=seg_duration,
            interruption_count=0,
            apps_involved=list(segment.apps_involved),
            representative_embedding=list(seg_embedding) if seg_embedding else [],
            parent_span_id=parent_span_id,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _update_span_states(
        self, spans: list[ContinuitySpan], current_time_iso: str,
    ) -> None:
        """Update span states based on time elapsed since last activity.

        Rules:
        - gap > 4h (14400s) and currently "active" -> "paused"
        - gap > 24h (86400s) and "active" or "paused" -> "completed"
        - "uncertain" state is never changed automatically.
        """
        current_epoch = _timestamp_to_epoch(current_time_iso)
        if not current_epoch:
            return

        for span in spans:
            if span.state == "uncertain":
                continue

            last_epoch = _timestamp_to_epoch(span.last_seen)
            if not last_epoch:
                continue

            gap = current_epoch - last_epoch
            if gap < 0:
                continue

            if gap > _GAP_COMPLETED and span.state in ("active", "paused"):
                span.state = "completed"
            elif gap > _GAP_PAUSED and span.state == "active":
                span.state = "paused"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_spans(self) -> list[ContinuitySpan]:
        """Load continuity spans from the knowledge base.

        Returns an empty list if the file does not exist or is malformed.
        """
        path = self._kb.root / "observations" / _CONTINUITY_FILE
        if not path.is_file():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load continuity spans: %s", exc)
            return []

        spans: list[ContinuitySpan] = []
        for item in data.get("spans", []):
            edges = []
            for edge_data in item.get("edges", []):
                edges.append(ContinuityEdge(
                    from_segment_id=edge_data["from_segment_id"],
                    to_segment_id=edge_data["to_segment_id"],
                    continuity_type=ContinuityType(edge_data["continuity_type"]),
                    confidence=edge_data["confidence"],
                    reasoning=edge_data["reasoning"],
                    gap_seconds=edge_data["gap_seconds"],
                    embedding_similarity=edge_data["embedding_similarity"],
                    app_overlap=edge_data["app_overlap"],
                ))

            spans.append(ContinuitySpan(
                span_id=item["span_id"],
                goal_summary=item.get("goal_summary", ""),
                continuity_confidence=item.get("continuity_confidence", 1.0),
                segments=item.get("segments", []),
                edges=edges,
                state=item.get("state", "active"),
                activity_type=item.get("activity_type", ""),
                matched_procedure_candidates=item.get(
                    "matched_procedure_candidates", [],
                ),
                first_seen=item.get("first_seen", ""),
                last_seen=item.get("last_seen", ""),
                total_duration_seconds=item.get("total_duration_seconds", 0),
                interruption_count=item.get("interruption_count", 0),
                apps_involved=item.get("apps_involved", []),
                representative_embedding=item.get("representative_embedding", []),
                parent_span_id=item.get("parent_span_id"),
            ))

        logger.debug("Loaded %d continuity spans from %s", len(spans), path)
        return spans

    def save_spans(self, spans: list[ContinuitySpan]) -> None:
        """Persist continuity spans to disk using atomic write.

        Serializes all spans (including edges with enum values) to
        ``{kb_root}/observations/continuity_spans.json``.
        """
        serialized_spans = []
        for span in spans:
            span_dict = asdict(span)
            # Ensure ContinuityType enums are serialized as their string values.
            # asdict() on a str-Enum already produces the string value, but
            # we normalise defensively in case of subclass oddities.
            for edge_dict in span_dict.get("edges", []):
                ct = edge_dict.get("continuity_type")
                if isinstance(ct, ContinuityType):
                    edge_dict["continuity_type"] = ct.value
            serialized_spans.append(span_dict)

        data = {
            "spans": serialized_spans,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / "observations" / _CONTINUITY_FILE
        self._kb.atomic_write_json(path, data)
        logger.info(
            "Saved %d continuity spans to %s", len(spans), path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_duration_seconds(segment: TaskSegment) -> int:
        """Compute the internal duration of a segment in seconds.

        Duration = last_frame.timestamp - first_frame.timestamp.
        Returns 0 if the segment has fewer than 2 frames or timestamps
        cannot be parsed.
        """
        if len(segment.frames) < 2:
            return 0
        first_epoch = _timestamp_to_epoch(segment.frames[0].timestamp)
        last_epoch = _timestamp_to_epoch(segment.frames[-1].timestamp)
        if not first_epoch or not last_epoch:
            return 0
        diff = int(last_epoch - first_epoch)
        return max(diff, 0)

    @staticmethod
    def _most_common_activity_type(segment: TaskSegment) -> str:
        """Return the most common activity_type across all frames in a segment."""
        counter: collections.Counter[str] = collections.Counter()
        for frame in segment.frames:
            if frame.activity_type:
                counter[frame.activity_type] += 1
        if not counter:
            return ""
        return counter.most_common(1)[0][0]

    def _update_activity_type(
        self, span: ContinuitySpan, new_segment: TaskSegment,
    ) -> None:
        """Re-evaluate activity_type for a span after adding a new segment.

        Uses the new segment's most common activity type.  If the new
        segment contributes a non-empty type, it may override the span's
        current type (we keep the most common across all segments, but
        since we don't store per-segment counters we use a simple heuristic:
        only replace if the span's current type is empty).
        """
        seg_type = self._most_common_activity_type(new_segment)
        if seg_type and not span.activity_type:
            span.activity_type = seg_type

"""Task segmenter for passive workflow discovery (v2 pipeline).

Replaces the heuristic episode_builder.py.  Uses VLM scene annotations
and text embeddings to identify, cluster, and stitch task segments from
the continuous annotation stream.

Pipeline:
  1. Load completed annotations from a time window
  2. Extract ``task_context.what_doing`` from each annotation
  3. Compute text embeddings via Ollama ``/api/embed`` (all-minilm:l6-v2)
  4. Cluster by cosine similarity ≥ threshold (default 0.75)
  5. Filter noise: drop clusters where ALL frames have is_workflow=False
  6. Stitch interrupted workflows: merge same-cluster segments separated
     by noise within a configurable gap (default 30 min)
  7. Output: ``TaskSegment`` objects with ordered event lists

Cross-session linking deferred to Phase 3 enhancement — this module
provides the core segmentation and stitching logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmenterConfig:
    """Configuration for the task segmenter."""

    # Embedding model (CPU-only, ~45MB, ~1ms per embedding)
    embedding_model: str = "all-minilm:l6-v2"
    ollama_host: str = "http://localhost:11434"

    # Clustering
    similarity_threshold: float = 0.75  # cosine similarity for same-task

    # Noise filtering
    min_workflow_ratio: float = 0.0  # ratio of is_workflow=True frames
    # 0.0 means keep clusters with at least 1 workflow frame

    # Interrupted workflow stitching
    stitch_max_gap_seconds: int = 1800  # 30 minutes

    # Minimum demonstrations for SOP generation
    min_demonstrations: int = 2

    # Time window for annotation loading
    default_window_hours: int = 4

    # Interruption parameters
    brief_interrupt_max_seconds: int = 60      # absorb interrupts shorter than this
    pause_max_minutes: int = 30                # merge pauses up to this duration
    related_interrupt_max_seconds: int = 300   # link related interrupts up to 5 min


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TaskState(Enum):
    """State of a task segment in the interruption model."""

    ACTIVE = "active"
    PAUSED = "paused"
    RESUMED = "resumed"
    ABANDONED = "abandoned"
    RELATED = "related"


@dataclass
class InterruptionEvent:
    """Represents an interruption within a task segment."""

    start_time: str
    end_time: str
    duration_seconds: int
    interrupting_app: str
    classification: TaskState  # PAUSED or RELATED


@dataclass
class AnnotatedFrame:
    """A single annotated frame with parsed metadata."""

    event_id: str
    timestamp: str
    annotation: dict
    diff: dict | None = None
    what_doing: str = ""
    is_workflow: bool = False
    activity_type: str = ""      # 8-class taxonomy (empty = legacy event)
    learnability: str = ""       # learning relevance (empty = legacy event)
    app: str = ""
    location: str = ""
    embedding: list[float] = field(default_factory=list)

    @classmethod
    def from_event(cls, event: dict) -> AnnotatedFrame | None:
        """Parse an event dict into an AnnotatedFrame.

        Returns None if the event has no valid annotation.
        """
        ann_json = event.get("scene_annotation_json")
        if not ann_json:
            return None

        try:
            annotation = json.loads(ann_json) if isinstance(ann_json, str) else ann_json
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(annotation, dict):
            return None

        tc = annotation.get("task_context", {})
        what_doing = tc.get("what_doing", "") if isinstance(tc, dict) else ""
        is_workflow = tc.get("is_workflow", False) if isinstance(tc, dict) else False
        if isinstance(is_workflow, str):
            is_workflow = is_workflow.lower() in ("true", "yes", "1")

        activity_type = tc.get("activity_type", "") if isinstance(tc, dict) else ""
        learnability = tc.get("learnability", "") if isinstance(tc, dict) else ""

        # Parse frame diff
        diff = None
        diff_json = event.get("frame_diff_json")
        if diff_json:
            try:
                diff = json.loads(diff_json) if isinstance(diff_json, str) else diff_json
            except (json.JSONDecodeError, TypeError):
                pass

        return cls(
            event_id=event.get("id", ""),
            timestamp=event.get("timestamp", ""),
            annotation=annotation,
            diff=diff,
            what_doing=what_doing,
            is_workflow=bool(is_workflow),
            activity_type=activity_type,
            learnability=learnability,
            app=annotation.get("app", ""),
            location=annotation.get("location", ""),
        )


@dataclass
class TaskSegment:
    """A contiguous segment of frames belonging to the same task.

    Multiple segments from the same cluster form demonstrations
    of the same workflow.
    """

    segment_id: str
    cluster_id: int
    frames: list[AnnotatedFrame] = field(default_factory=list)
    task_label: str = ""  # representative what_doing for this cluster
    apps_involved: list[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    interruptions: list[InterruptionEvent] = field(default_factory=list)
    state: TaskState = TaskState.ACTIVE

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def workflow_ratio(self) -> float:
        """Ratio of frames with is_workflow=True."""
        if not self.frames:
            return 0.0
        wf = sum(1 for f in self.frames if f.is_workflow)
        return wf / len(self.frames)

    def to_timeline(self) -> list[dict]:
        """Convert to timeline format expected by SOPGenerator."""
        timeline = []
        for frame in self.frames:
            timeline.append({
                "annotation": frame.annotation,
                "diff": frame.diff,
                "timestamp": frame.timestamp,
            })
        return timeline


@dataclass
class SegmentationResult:
    """Result of a segmentation pass."""

    segments: list[TaskSegment] = field(default_factory=list)
    clusters: dict[int, list[TaskSegment]] = field(default_factory=dict)
    noise_frames_dropped: int = 0
    total_frames_processed: int = 0
    embedding_time_seconds: float = 0.0
    spans: list | None = None  # ContinuitySpan list, populated by ContinuityTracker


# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

def _compute_embeddings(
    texts: list[str],
    *,
    model: str = "all-minilm:l6-v2",
    host: str = "http://localhost:11434",
    timeout: float = 30.0,
) -> list[list[float]]:
    """Compute text embeddings via Ollama's /api/embed endpoint.

    Returns a list of embedding vectors (one per input text).
    Raises ConnectionError if Ollama is not reachable.
    """
    import urllib.request
    import urllib.error

    if not texts:
        return []

    url = f"{host}/api/embed"
    payload = {
        "model": model,
        "input": texts,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ollama not reachable at {host}: {exc}"
        ) from exc

    embeddings = result.get("embeddings", [])
    if len(embeddings) != len(texts):
        logger.warning(
            "Embedding count mismatch: expected %d, got %d",
            len(texts), len(embeddings),
        )
        # Pad with empty vectors if needed
        while len(embeddings) < len(texts):
            embeddings.append([])

    return embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 for empty or zero-length vectors.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Noise classification
# ---------------------------------------------------------------------------

def _is_noise_frame(frame: AnnotatedFrame) -> bool:
    """Check if a frame is noise.

    Uses activity_type/learnability when available (Phase 1+),
    falls back to is_workflow for legacy events.
    """
    if frame.activity_type:
        return (
            frame.activity_type in ("entertainment", "dead_time")
            or frame.learnability == "ignore"
        )
    return not frame.is_workflow


# ---------------------------------------------------------------------------
# Greedy clustering
# ---------------------------------------------------------------------------

def _cluster_frames(
    frames: list[AnnotatedFrame],
    threshold: float = 0.75,
) -> dict[int, list[int]]:
    """Cluster frames by embedding cosine similarity.

    Uses greedy single-linkage: each frame joins the first cluster
    whose centroid has similarity >= threshold, or starts a new cluster.

    Returns a dict of cluster_id -> list of frame indices.
    """
    if not frames:
        return {}

    clusters: dict[int, list[int]] = {}
    centroids: dict[int, list[float]] = {}
    next_id = 0

    for i, frame in enumerate(frames):
        if not frame.embedding:
            # No embedding — put in its own cluster
            clusters[next_id] = [i]
            next_id += 1
            continue

        # Find best matching cluster
        best_cluster = -1
        best_sim = 0.0

        for cid, centroid in centroids.items():
            sim = _cosine_similarity(frame.embedding, centroid)
            if sim >= threshold and sim > best_sim:
                best_cluster = cid
                best_sim = sim

        if best_cluster >= 0:
            clusters[best_cluster].append(i)
            # Update centroid (running average)
            _update_centroid(centroids, best_cluster, frame.embedding, len(clusters[best_cluster]))
        else:
            clusters[next_id] = [i]
            centroids[next_id] = list(frame.embedding)
            next_id += 1

    return clusters


def _update_centroid(
    centroids: dict[int, list[float]],
    cluster_id: int,
    new_vec: list[float],
    cluster_size: int,
) -> None:
    """Update the centroid of a cluster with a new vector (running average)."""
    old = centroids[cluster_id]
    if not old or not new_vec:
        return

    # Running average: new_centroid = old * (n-1)/n + new_vec/n
    n = cluster_size
    centroids[cluster_id] = [
        (old[j] * (n - 1) + new_vec[j]) / n
        for j in range(len(old))
    ]


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _timestamp_to_epoch(ts: str) -> float:
    """Parse an ISO timestamp to epoch seconds. Returns 0 on failure."""
    if not ts:
        return 0.0
    try:
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _split_into_contiguous_segments(
    frames: list[AnnotatedFrame],
    cluster_id: int,
    max_gap_seconds: int = 1800,
) -> list[TaskSegment]:
    """Split a cluster's frames into contiguous time segments.

    Two consecutive frames with > max_gap_seconds between them
    form separate segments (indicating an interruption).
    """
    if not frames:
        return []

    # Sort by timestamp
    sorted_frames = sorted(frames, key=lambda f: f.timestamp)

    segments: list[TaskSegment] = []
    current_frames: list[AnnotatedFrame] = [sorted_frames[0]]

    for i in range(1, len(sorted_frames)):
        prev_epoch = _timestamp_to_epoch(sorted_frames[i - 1].timestamp)
        curr_epoch = _timestamp_to_epoch(sorted_frames[i].timestamp)

        gap = curr_epoch - prev_epoch if prev_epoch and curr_epoch else 0

        if gap > max_gap_seconds:
            # Gap too large — start new segment
            seg = _make_segment(current_frames, cluster_id, len(segments))
            segments.append(seg)
            current_frames = [sorted_frames[i]]
        else:
            current_frames.append(sorted_frames[i])

    # Final segment
    if current_frames:
        seg = _make_segment(current_frames, cluster_id, len(segments))
        segments.append(seg)

    return segments


def _content_segment_id(frames: list[AnnotatedFrame]) -> str:
    """Derive a stable segment ID from the event IDs in the segment.

    Uses a SHA-256 hash of sorted event IDs so the identity is independent
    of transient cluster numbering.  This prevents re-clustering from creating
    duplicate pending rows in the DB.
    """
    event_ids = sorted(f.event_id for f in frames if f.event_id)
    digest = hashlib.sha256("|".join(event_ids).encode()).hexdigest()[:12]
    return f"seg-{digest}"


def _make_segment(
    frames: list[AnnotatedFrame],
    cluster_id: int,
    seq: int,
) -> TaskSegment:
    """Create a TaskSegment from a list of frames."""
    # Find representative task label (most common what_doing)
    labels: dict[str, int] = {}
    for f in frames:
        if f.what_doing:
            labels[f.what_doing] = labels.get(f.what_doing, 0) + 1
    task_label = max(labels, key=labels.get) if labels else ""

    # Collect unique apps
    apps: list[str] = []
    seen_apps: set[str] = set()
    for f in frames:
        if f.app and f.app not in seen_apps:
            apps.append(f.app)
            seen_apps.add(f.app)

    return TaskSegment(
        segment_id=_content_segment_id(frames),
        cluster_id=cluster_id,
        frames=frames,
        task_label=task_label,
        apps_involved=apps,
        start_time=frames[0].timestamp if frames else "",
        end_time=frames[-1].timestamp if frames else "",
    )


def _stitch_interrupted_workflows(
    segments: list[TaskSegment],
    noise_segments: list[TaskSegment],
    max_gap_seconds: int = 1800,
) -> list[TaskSegment]:
    """Stitch interrupted workflows: merge segments of the same cluster
    separated by noise within max_gap_seconds.

    If cluster A appears at time T, then noise at T+5min, then cluster A
    again at T+10min, merge the two A segments into one.
    """
    if len(segments) < 2:
        return segments

    # Sort all segments by start_time
    sorted_segs = sorted(segments, key=lambda s: s.start_time)

    # Group by cluster_id
    by_cluster: dict[int, list[TaskSegment]] = {}
    for seg in sorted_segs:
        by_cluster.setdefault(seg.cluster_id, []).append(seg)

    merged: list[TaskSegment] = []
    for cluster_id, cluster_segs in by_cluster.items():
        if len(cluster_segs) <= 1:
            merged.extend(cluster_segs)
            continue

        # Try to merge consecutive segments within gap
        current = cluster_segs[0]
        for i in range(1, len(cluster_segs)):
            next_seg = cluster_segs[i]

            curr_end_epoch = _timestamp_to_epoch(current.end_time)
            next_start_epoch = _timestamp_to_epoch(next_seg.start_time)
            gap = next_start_epoch - curr_end_epoch if curr_end_epoch and next_start_epoch else float("inf")

            if gap <= max_gap_seconds:
                # Merge: combine frames
                merged_frames = current.frames + next_seg.frames
                current = TaskSegment(
                    segment_id=_content_segment_id(merged_frames),
                    cluster_id=cluster_id,
                    frames=merged_frames,
                    task_label=current.task_label,
                    apps_involved=list(set(current.apps_involved + next_seg.apps_involved)),
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                )
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)

    return merged


# ---------------------------------------------------------------------------
# TaskSegmenter
# ---------------------------------------------------------------------------

class TaskSegmenter:
    """Segment VLM annotations into task clusters for passive SOP discovery.

    Designed to run periodically (every 2 hours, on idle, or on-demand)
    as Thread 3 in the worker process.  CPU-only (no GPU access needed).
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()

    def segment(
        self,
        events: list[dict],
    ) -> SegmentationResult:
        """Run the full segmentation pipeline on a batch of annotated events.

        Args:
            events: List of event dicts from the DB with completed
                annotations (scene_annotation_json, frame_diff_json).

        Returns:
            SegmentationResult with task segments grouped by cluster.
        """
        result = SegmentationResult(total_frames_processed=len(events))

        if not events:
            return result

        # Step 1: Parse events into AnnotatedFrames
        frames: list[AnnotatedFrame] = []
        for event in events:
            frame = AnnotatedFrame.from_event(event)
            if frame is not None:
                frames.append(frame)

        if not frames:
            return result

        result.total_frames_processed = len(frames)

        # Step 2: Compute embeddings for all what_doing strings
        texts = [f.what_doing if f.what_doing else f.app for f in frames]
        try:
            embed_start = time.monotonic()
            embeddings = _compute_embeddings(
                texts,
                model=self.config.embedding_model,
                host=self.config.ollama_host,
            )
            result.embedding_time_seconds = time.monotonic() - embed_start

            for i, emb in enumerate(embeddings):
                if i < len(frames):
                    frames[i].embedding = emb

            logger.info(
                "Computed %d embeddings in %.1fs",
                len(embeddings), result.embedding_time_seconds,
            )
        except ConnectionError as exc:
            logger.warning("Embedding failed (Ollama not reachable): %s", exc)
            # Fall back to app-based clustering (no embeddings)
            return self._fallback_app_clustering(frames, result)

        # Step 3: Cluster by embedding similarity
        clusters = _cluster_frames(frames, self.config.similarity_threshold)

        # Step 4: Split clusters into contiguous segments + noise filtering
        all_segments: list[TaskSegment] = []
        noise_segments: list[TaskSegment] = []

        for cluster_id, frame_indices in clusters.items():
            cluster_frames = [frames[i] for i in frame_indices]

            # Check noise: all non-workflow?
            noise_count = sum(1 for f in cluster_frames if _is_noise_frame(f))
            if noise_count == len(cluster_frames):
                # Pure noise cluster
                result.noise_frames_dropped += len(cluster_frames)
                # Still create segments for stitching reference
                segs = _split_into_contiguous_segments(
                    cluster_frames, cluster_id,
                    self.config.stitch_max_gap_seconds,
                )
                noise_segments.extend(segs)
                continue

            # Split into contiguous segments
            segs = _split_into_contiguous_segments(
                cluster_frames, cluster_id,
                self.config.stitch_max_gap_seconds,
            )
            all_segments.extend(segs)

        # Step 5: Stitch interrupted workflows
        stitched = _stitch_interrupted_workflows(
            all_segments, noise_segments,
            self.config.stitch_max_gap_seconds,
        )

        # Step 6: Group segments by cluster for SOP generation
        result.segments = stitched
        for seg in stitched:
            result.clusters.setdefault(seg.cluster_id, []).append(seg)

        logger.info(
            "Segmentation: %d frames → %d clusters, %d segments, "
            "%d noise frames dropped",
            len(frames), len(result.clusters),
            len(result.segments), result.noise_frames_dropped,
        )

        return result

    def get_sop_ready_clusters(
        self, result: SegmentationResult,
    ) -> list[tuple[str, list[list[dict]]]]:
        """Extract clusters with ≥ min_demonstrations for SOP generation.

        Returns a list of (task_label, demonstrations) tuples where each
        demonstration is a timeline (list of frame dicts) suitable for
        ``SOPGenerator.generate_from_passive()``.
        """
        ready: list[tuple[str, list[list[dict]]]] = []

        for cluster_id, segments in result.clusters.items():
            if len(segments) < self.config.min_demonstrations:
                continue

            # Find the best task label from the cluster
            task_label = ""
            for seg in segments:
                if seg.task_label:
                    task_label = seg.task_label
                    break

            # Convert segments to timelines
            demonstrations = [seg.to_timeline() for seg in segments]

            ready.append((task_label, demonstrations))

        return ready

    def classify_interruptions(
        self,
        segments: list[TaskSegment],
        all_frames: list[AnnotatedFrame],
    ) -> list[TaskSegment]:
        """Post-process segments to classify and merge interruptions.

        Rules:
        - Brief interrupt (<60s): Absorb into parent segment. A 30s Slack check
          mid-research doesn't split the task.
        - Pause (<30min): Mark as paused, merge if same cluster resumes.
          The parent segment's events span the full duration.
        - Related interrupt (<5min): Link as related if the interrupting app
          is semantically related to the parent task.
        - Abandon (>30min or no return): Mark parent as abandoned.

        This method operates on the already-clustered segments from
        segment_tasks() and merges/annotates them.
        """
        if not segments:
            return []

        # Sort segments by start_time for chronological processing
        sorted_segs = sorted(segments, key=lambda s: _timestamp_to_epoch(s.start_time))

        brief_max = self.config.brief_interrupt_max_seconds
        pause_max = self.config.pause_max_minutes * 60  # convert to seconds
        related_max = self.config.related_interrupt_max_seconds

        # Build a lookup of all frames by timestamp range for finding
        # what apps were active during gaps
        all_frames_sorted = sorted(all_frames, key=lambda f: _timestamp_to_epoch(f.timestamp))

        # Phase 1: Identify which segments are "interruptions" between
        # same-cluster parent segments.
        # We iterate over pairs of same-cluster segments and check for
        # intervening segments from other clusters.

        # Group by cluster_id, preserving chronological order
        by_cluster: dict[int, list[int]] = {}  # cluster_id -> indices in sorted_segs
        for idx, seg in enumerate(sorted_segs):
            by_cluster.setdefault(seg.cluster_id, []).append(idx)

        # Track which segment indices should be merged into others
        # merge_into[i] = j means segment i should be merged into segment j
        merge_into: dict[int, int] = {}
        # Track which segments are "interrupting" segments (absorbed/linked)
        interruption_segs: set[int] = set()

        for cluster_id, indices in by_cluster.items():
            if len(indices) < 2:
                continue

            # Check consecutive pairs within the same cluster
            i = 0
            while i < len(indices) - 1:
                idx_a = indices[i]
                idx_b = indices[i + 1]
                seg_a = sorted_segs[idx_a]
                seg_b = sorted_segs[idx_b]

                end_a = _timestamp_to_epoch(seg_a.end_time)
                start_b = _timestamp_to_epoch(seg_b.start_time)

                if not end_a or not start_b:
                    i += 1
                    continue

                gap_seconds = start_b - end_a

                if gap_seconds <= 0:
                    # Overlapping or adjacent — just merge
                    merge_into[idx_b] = idx_a
                    # Update indices: idx_b is consumed, subsequent indices
                    # in this cluster now point to idx_a's merged result
                    indices[i + 1] = idx_a
                    i += 1
                    continue

                # Find intervening segments (from OTHER clusters) in the gap.
                # Only consider segments whose cluster has a single occurrence
                # (true interruptions); segments belonging to multi-segment
                # clusters are kept for their own cluster's merging.
                intervening = []
                for other_idx, other_seg in enumerate(sorted_segs):
                    if other_idx == idx_a or other_idx == idx_b:
                        continue
                    if other_idx in merge_into:
                        continue
                    # Skip segments from clusters that have their own
                    # multi-segment merging to do
                    other_cluster = other_seg.cluster_id
                    if len(by_cluster.get(other_cluster, [])) > 1:
                        continue
                    other_start = _timestamp_to_epoch(other_seg.start_time)
                    other_end = _timestamp_to_epoch(other_seg.end_time)
                    if other_start and other_end:
                        # Segment falls within the gap
                        if other_start >= end_a and other_end <= start_b:
                            intervening.append(other_idx)

                # Find apps active during the gap from all_frames
                gap_apps = set()
                for frame in all_frames_sorted:
                    ft = _timestamp_to_epoch(frame.timestamp)
                    if ft and end_a <= ft <= start_b:
                        if frame.app:
                            gap_apps.add(frame.app)

                if gap_seconds <= brief_max:
                    # Brief interrupt: absorb gap entirely, merge segments
                    merge_into[idx_b] = idx_a
                    # Mark any intervening segments as absorbed interruptions
                    for inter_idx in intervening:
                        interruption_segs.add(inter_idx)
                        merge_into[inter_idx] = idx_a
                    # Update the chain for subsequent pairs
                    indices[i + 1] = idx_a
                elif gap_seconds <= pause_max:
                    # Pause: merge segments and annotate with InterruptionEvent
                    merge_into[idx_b] = idx_a
                    for inter_idx in intervening:
                        interruption_segs.add(inter_idx)
                        merge_into[inter_idx] = idx_a
                    indices[i + 1] = idx_a
                    # The InterruptionEvent will be created during merge
                else:
                    # Gap exceeds pause_max — check for related interrupt
                    # Related: gap < related_max AND interrupting app is
                    # semantically related (same app set overlap)
                    if gap_seconds <= related_max and gap_apps & set(seg_a.apps_involved):
                        # Related interrupt — link but don't merge
                        # Just annotate, don't merge the segments
                        pass
                    # Otherwise: abandon (handled in Phase 2)

                i += 1

        # Phase 2: Build the merged result
        # Resolve merge chains: if A merges into B and B merges into C,
        # then A merges into C
        def resolve_target(idx: int) -> int:
            visited = set()
            while idx in merge_into:
                if idx in visited:
                    break
                visited.add(idx)
                idx = merge_into[idx]
            return idx

        # Collect merged groups: target_idx -> list of source indices
        merge_groups: dict[int, list[int]] = {}
        for idx in range(len(sorted_segs)):
            target = resolve_target(idx)
            merge_groups.setdefault(target, []).append(idx)

        result: list[TaskSegment] = []
        processed: set[int] = set()

        for target_idx, group_indices in sorted(merge_groups.items()):
            if target_idx in processed:
                continue

            group_indices_sorted = sorted(group_indices)
            base_seg = sorted_segs[group_indices_sorted[0]]

            if len(group_indices_sorted) == 1:
                # No merging needed for this segment
                result.append(base_seg)
                processed.add(target_idx)
                continue

            # Merge all segments in the group
            all_merged_frames: list[AnnotatedFrame] = []
            merged_apps: list[str] = []
            seen_apps: set[str] = set()
            interruptions: list[InterruptionEvent] = []
            earliest_start = base_seg.start_time
            latest_end = base_seg.end_time

            for seg_idx in group_indices_sorted:
                seg = sorted_segs[seg_idx]

                # Collect frames from non-interruption segments
                if seg_idx not in interruption_segs:
                    all_merged_frames.extend(seg.frames)

                # Track apps
                for app in seg.apps_involved:
                    if app not in seen_apps:
                        merged_apps.append(app)
                        seen_apps.add(app)

                # Track time bounds
                seg_start_epoch = _timestamp_to_epoch(seg.start_time)
                seg_end_epoch = _timestamp_to_epoch(seg.end_time)
                earliest_epoch = _timestamp_to_epoch(earliest_start)
                latest_epoch = _timestamp_to_epoch(latest_end)

                if seg_start_epoch and (not earliest_epoch or seg_start_epoch < earliest_epoch):
                    earliest_start = seg.start_time
                if seg_end_epoch and (not latest_epoch or seg_end_epoch > latest_epoch):
                    latest_end = seg.end_time

            # Create InterruptionEvents for gaps between consecutive
            # same-cluster segments that were merged
            same_cluster_indices = [
                i for i in group_indices_sorted
                if sorted_segs[i].cluster_id == base_seg.cluster_id
                and i not in interruption_segs
            ]
            same_cluster_indices.sort(key=lambda i: _timestamp_to_epoch(sorted_segs[i].start_time))

            for k in range(len(same_cluster_indices) - 1):
                prev_seg = sorted_segs[same_cluster_indices[k]]
                next_seg = sorted_segs[same_cluster_indices[k + 1]]
                prev_end = _timestamp_to_epoch(prev_seg.end_time)
                next_start = _timestamp_to_epoch(next_seg.start_time)
                if prev_end and next_start and next_start > prev_end:
                    gap = int(next_start - prev_end)

                    # Find the interrupting app during this gap
                    inter_app = ""
                    for inter_idx in group_indices_sorted:
                        if inter_idx in interruption_segs:
                            inter_seg = sorted_segs[inter_idx]
                            inter_start = _timestamp_to_epoch(inter_seg.start_time)
                            inter_end = _timestamp_to_epoch(inter_seg.end_time)
                            if inter_start and inter_end:
                                if inter_start >= prev_end and inter_end <= next_start:
                                    inter_app = (
                                        inter_seg.apps_involved[0]
                                        if inter_seg.apps_involved
                                        else ""
                                    )
                                    break

                    if not inter_app:
                        # Check all_frames for apps in the gap
                        for frame in all_frames_sorted:
                            ft = _timestamp_to_epoch(frame.timestamp)
                            if ft and prev_end <= ft <= next_start and frame.app:
                                inter_app = frame.app
                                break

                    classification = (
                        TaskState.PAUSED
                        if gap > brief_max
                        else TaskState.PAUSED
                    )

                    interruptions.append(InterruptionEvent(
                        start_time=prev_seg.end_time,
                        end_time=next_seg.start_time,
                        duration_seconds=gap,
                        interrupting_app=inter_app,
                        classification=classification,
                    ))

            # Sort merged frames by timestamp
            all_merged_frames.sort(key=lambda f: _timestamp_to_epoch(f.timestamp))

            merged_segment = TaskSegment(
                segment_id=_content_segment_id(all_merged_frames),
                cluster_id=base_seg.cluster_id,
                frames=all_merged_frames,
                task_label=base_seg.task_label,
                apps_involved=merged_apps,
                start_time=earliest_start,
                end_time=latest_end,
                interruptions=interruptions,
                state=TaskState.ACTIVE,
            )
            result.append(merged_segment)
            processed.update(group_indices_sorted)

        # Phase 3: Mark abandoned segments
        # A segment is abandoned if:
        # - It's not the last segment chronologically for its cluster, AND
        # - The gap to the next same-cluster segment exceeds pause_max, AND
        # - There's no subsequent return to the same cluster
        # OR
        # - The segment's cluster doesn't appear again within pause_max
        # after this segment ends

        # Rebuild cluster groups from the result
        result_by_cluster: dict[int, list[int]] = {}
        for idx, seg in enumerate(result):
            result_by_cluster.setdefault(seg.cluster_id, []).append(idx)

        for cluster_id, res_indices in result_by_cluster.items():
            if len(res_indices) < 2:
                continue
            res_indices_sorted = sorted(
                res_indices,
                key=lambda i: _timestamp_to_epoch(result[i].start_time),
            )
            for k in range(len(res_indices_sorted) - 1):
                seg = result[res_indices_sorted[k]]
                next_seg = result[res_indices_sorted[k + 1]]
                end_epoch = _timestamp_to_epoch(seg.end_time)
                next_start_epoch = _timestamp_to_epoch(next_seg.start_time)
                if end_epoch and next_start_epoch:
                    gap = next_start_epoch - end_epoch
                    if gap > pause_max:
                        seg.state = TaskState.ABANDONED

        # Phase 4: Annotate related interruptions
        # Check consecutive segments (in chronological order in the result)
        # where a gap < related_max exists and the interrupting app
        # is in the parent segment's app set
        result_sorted = sorted(result, key=lambda s: _timestamp_to_epoch(s.start_time))
        for k in range(len(result_sorted) - 1):
            seg_a = result_sorted[k]
            seg_b = result_sorted[k + 1]

            if seg_a.cluster_id == seg_b.cluster_id:
                continue  # Same cluster, already handled

            end_a = _timestamp_to_epoch(seg_a.end_time)
            start_b = _timestamp_to_epoch(seg_b.start_time)
            if not end_a or not start_b:
                continue

            gap = start_b - end_a
            if gap < 0:
                gap = 0

            # Check if seg_b is a brief related interruption before
            # seg_a's cluster resumes
            seg_b_duration = (
                _timestamp_to_epoch(seg_b.end_time) - start_b
                if _timestamp_to_epoch(seg_b.end_time) and start_b
                else 0
            )

            if seg_b_duration <= related_max:
                # Check if seg_a's cluster resumes after seg_b
                end_b = _timestamp_to_epoch(seg_b.end_time)
                if end_b:
                    for seg_c in result_sorted[k + 2:]:
                        if seg_c.cluster_id == seg_a.cluster_id:
                            start_c = _timestamp_to_epoch(seg_c.start_time)
                            if start_c and (start_c - end_b) <= related_max:
                                # seg_b is related to seg_a
                                # Check if apps overlap (semantic relation)
                                if set(seg_b.apps_involved) & set(seg_a.apps_involved):
                                    seg_a.interruptions.append(InterruptionEvent(
                                        start_time=seg_b.start_time,
                                        end_time=seg_b.end_time,
                                        duration_seconds=int(seg_b_duration),
                                        interrupting_app=(
                                            seg_b.apps_involved[0]
                                            if seg_b.apps_involved
                                            else ""
                                        ),
                                        classification=TaskState.RELATED,
                                    ))
                            break

        return result_sorted

    def _fallback_app_clustering(
        self,
        frames: list[AnnotatedFrame],
        result: SegmentationResult,
    ) -> SegmentationResult:
        """Fallback clustering by app name when embeddings are unavailable.

        Groups frames by app name, splits into contiguous segments.
        Less accurate than embedding-based clustering but still useful.
        """
        by_app: dict[str, list[AnnotatedFrame]] = {}
        for f in frames:
            key = f.app or "unknown"
            by_app.setdefault(key, []).append(f)

        cluster_id = 0
        for app, app_frames in by_app.items():
            noise_count = sum(1 for f in app_frames if _is_noise_frame(f))
            if noise_count == len(app_frames):
                result.noise_frames_dropped += len(app_frames)
                continue

            segs = _split_into_contiguous_segments(
                app_frames, cluster_id,
                self.config.stitch_max_gap_seconds,
            )
            result.segments.extend(segs)
            result.clusters.setdefault(cluster_id, []).extend(segs)
            cluster_id += 1

        return result

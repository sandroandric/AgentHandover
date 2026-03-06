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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedFrame:
    """A single annotated frame with parsed metadata."""

    event_id: str
    timestamp: str
    annotation: dict
    diff: dict | None = None
    what_doing: str = ""
    is_workflow: bool = False
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
            wf_count = sum(1 for f in cluster_frames if f.is_workflow)
            if wf_count == 0:
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
            wf_count = sum(1 for f in app_frames if f.is_workflow)
            if wf_count == 0:
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

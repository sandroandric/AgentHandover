"""Task segmenter for passive workflow discovery (v2 pipeline).

Replaces the heuristic episode_builder.py.  Uses VLM scene annotations
and text embeddings to identify and cluster task segments from the
continuous annotation stream.

Pipeline:
  1. Load completed annotations from a time window
  2. Extract ``task_context.what_doing`` from each annotation
  3. Compute text embeddings via Ollama ``/api/embed`` (nomic-embed-text, ``clustering:`` prefix)
  4. Cluster by cosine similarity ≥ threshold (default 0.83)
  5. Filter noise: drop clusters where ALL frames have is_workflow=False
  6. Split each cluster into contiguous segments wherever two consecutive
     frames are more than ``stitch_max_gap_seconds`` apart (default 30 min),
     so a same-cluster return within that gap stays one segment
  7. Output: ``TaskSegment`` objects with ordered event lists

Cross-segment linking lives in ``continuity_tracker``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agenthandover_worker.model_profiles import EMBEDDING_MODEL, embedding_profile
from agenthandover_worker.ollama import OllamaClient, OllamaError

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner
    from agenthandover_worker.vector_kb import VectorKB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmenterConfig:
    """Configuration for the task segmenter."""

    # ``[embedding] model``; its profile gives the clustering prefix, the
    # stored size and the threshold below.
    embedding_model: str = EMBEDDING_MODEL
    ollama_host: str = "http://localhost:11434"

    # Clustering: cosine similarity to a cluster centroid for "same task".
    # None: the embedding profile's ``segmenter`` threshold.  For
    # nomic-embed-text that is 0.83, calibrated for its ``clustering:``-
    # prefixed vectors (which sit ~0.09 higher than unprefixed ones): the
    # operating point the old unprefixed 0.75 had (no two tasks merged) on
    # the labelled set in tests/journeys/test_segmenter_threshold.py.
    similarity_threshold: float | None = None

    # Noise filtering
    min_workflow_ratio: float = 0.0  # ratio of is_workflow=True frames
    # 0.0 means keep clusters with at least 1 workflow frame

    # A larger gap between consecutive same-cluster frames starts a new segment
    stitch_max_gap_seconds: int = 1800  # 30 minutes

    # Minimum demonstrations for SOP generation
    min_demonstrations: int = 2

    # Time window for annotation loading (24h gives full-day visibility)
    default_window_hours: int = 24


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


def frame_text(frame: AnnotatedFrame) -> str:
    """The text a frame is embedded from (with the clustering prefix):
    what the user was doing, the app, the location and, when the
    annotation has one, the start of its visual text proxy."""
    parts = [frame.what_doing or frame.app]
    if frame.app and frame.what_doing:
        parts.append(frame.app)
    if frame.location:
        parts.append(frame.location)
    proxy = frame.annotation.get("_visual_text_proxy", "") if frame.annotation else ""
    if proxy and len(proxy) > 20:
        parts.append(proxy[:500])
    return " | ".join(parts)


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
    spans: list | None = None  # ContinuitySpan list, populated by ContinuityTracker


# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

# The embedding models are trained with task prefixes; grouping frames by
# task is their ``clustering`` task.  The similarity threshold is
# calibrated for vectors computed with that prefix.


def _compute_embeddings(
    texts: list[str],
    *,
    model: str,
    host: str = "http://localhost:11434",
    timeout: float = 30.0,
) -> list[list[float]]:
    """Embed *texts* with *model*'s clustering task prefix, at its
    profile's size, via /api/embed.

    Returns a list of embedding vectors (one per input text).
    Raises ConnectionError if Ollama is not reachable.
    """
    profile = embedding_profile(model)
    return OllamaClient(host).embed(
        model, texts, prefix=profile.cluster_prefix,
        dimensions=profile.dimensions, timeout=timeout,
    )


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
    if frame.learnability == "ignore":
        return True
    if frame.activity_type and frame.activity_type != "unknown":
        return frame.activity_type in ("entertainment", "dead_time")
    return not frame.is_workflow


# ---------------------------------------------------------------------------
# Greedy clustering
# ---------------------------------------------------------------------------

def _cluster_frames(
    frames: list[AnnotatedFrame],
    threshold: float = 0.83,
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


def _sleep_starts(sleep_boundaries: list[tuple[str, str]] | None) -> list[float]:
    """Epoch seconds at which each ``(slept_at, woke_at)`` sleep began."""
    return sorted(
        epoch for slept_at, _ in sleep_boundaries or []
        if (epoch := _timestamp_to_epoch(slept_at))
    )


def _sleep_between(start: float, end: float, sleep_starts: list[float]) -> bool:
    """True when the Mac went to sleep at or after *start* and before *end*."""
    return any(start <= s < end for s in sleep_starts)


def _split_into_contiguous_segments(
    frames: list[AnnotatedFrame],
    cluster_id: int,
    max_gap_seconds: int = 1800,
    sleep_starts: list[float] | None = None,
) -> list[TaskSegment]:
    """Split a cluster's frames into contiguous time segments.

    Two consecutive frames with > max_gap_seconds between them, or with a
    system sleep between them (*sleep_starts*), form separate segments.
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
        slept = bool(prev_epoch and curr_epoch) and _sleep_between(
            prev_epoch, curr_epoch, sleep_starts or [],
        )

        if gap > max_gap_seconds or slept:
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


# ---------------------------------------------------------------------------
# TaskSegmenter
# ---------------------------------------------------------------------------

class TaskSegmenter:
    """Segment VLM annotations into task clusters for passive SOP discovery.

    Designed to run periodically (every 2 hours, on idle, or on-demand)
    as Thread 3 in the worker process.  CPU-only (no GPU access needed).
    """

    def __init__(
        self,
        config: SegmenterConfig | None = None,
        llm_reasoner: "LLMReasoner | None" = None,
        cluster_label_store: Path | None = None,
        vector_kb: "VectorKB | None" = None,
    ) -> None:
        self.config = config or SegmenterConfig()
        self._profile = embedding_profile(self.config.embedding_model)
        self._llm_reasoner = llm_reasoner
        self._cluster_label_store = cluster_label_store
        self._vector_kb = vector_kb
        self._label_cache: dict = {}
        self._load_label_cache()

    def _label_cluster_with_llm(self, frames: list[AnnotatedFrame]) -> str | None:
        """Use LLM to generate a concise workflow name for a cluster.

        Collects unique what_doing strings, apps, and locations from
        the cluster's frames and asks the LLM for a 3-8 word label.

        Returns the label string, or None on failure/budget/abstention.
        """
        if self._llm_reasoner is None:
            return None

        what_doings = sorted({f.what_doing for f in frames if f.what_doing})
        apps = sorted({f.app for f in frames if f.app})
        locations = sorted({f.location for f in frames if f.location})

        if not what_doings:
            return None

        prompt = (
            "Given these observations from a user's workflow session, "
            "provide a concise workflow name (3-8 words).\n\n"
            f"Activities observed: {', '.join(what_doings)}\n"
            f"Applications used: {', '.join(apps) if apps else 'unknown'}\n"
            f"Locations: {', '.join(locations) if locations else 'unknown'}\n\n"
            "Respond with ONLY the workflow name, nothing else."
        )

        try:
            result = self._llm_reasoner.reason_text(
                prompt, caller="task_segmenter.label_cluster",
            )
            if result.success and not result.abstained and result.value:
                label = str(result.value).strip()
                if label:
                    return label
        except Exception:
            logger.debug("LLM cluster labeling failed", exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Stable cluster label cache
    # ------------------------------------------------------------------

    def _compute_centroid_hash(self, frames: list[AnnotatedFrame]) -> str:
        """Compute a stable hash from the average embedding of frames."""
        embedded = [f for f in frames if f.embedding]
        if not embedded:
            return ""
        dim = len(embedded[0].embedding)
        avg = [0.0] * dim
        for f in embedded:
            for j in range(min(dim, len(f.embedding))):
                avg[j] += f.embedding[j]
        n = len(embedded)
        avg = [round(v / n, 4) for v in avg]
        digest = hashlib.sha256(str(avg).encode()).hexdigest()[:16]
        return digest

    def _get_centroid_vector(self, frames: list[AnnotatedFrame]) -> list[float]:
        """Compute average embedding vector from frames."""
        embedded = [f for f in frames if f.embedding]
        if not embedded:
            return []
        dim = len(embedded[0].embedding)
        avg = [0.0] * dim
        for f in embedded:
            for j in range(min(dim, len(f.embedding))):
                avg[j] += f.embedding[j]
        n = len(embedded)
        return [round(v / n, 4) for v in avg]

    def _get_or_create_cluster_label(
        self, frames: list[AnnotatedFrame],
    ) -> str | None:
        """Get a cached label or create a new one for a cluster.

        1. Compute centroid hash, check exact match in cache.
        2. If no exact match, check cosine similarity > 0.9 against cached centroids.
        3. If no match, call _label_cluster_with_llm(), cache the result.
        """
        centroid_hash = self._compute_centroid_hash(frames)
        if not centroid_hash:
            return self._label_cluster_with_llm(frames)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Exact hash match
        if centroid_hash in self._label_cache:
            entry = self._label_cache[centroid_hash]
            entry["last_used"] = today
            entry["use_count"] = entry.get("use_count", 0) + 1
            self._save_label_cache()
            return entry.get("label")

        # Cosine similarity check against cached centroids
        centroid_vec = self._get_centroid_vector(frames)
        if centroid_vec:
            for cached_hash, entry in self._label_cache.items():
                cached_centroid = entry.get("centroid")
                if not cached_centroid:
                    continue
                sim = _cosine_similarity(centroid_vec, cached_centroid)
                if sim > 0.9:
                    entry["last_used"] = today
                    entry["use_count"] = entry.get("use_count", 0) + 1
                    self._save_label_cache()
                    return entry.get("label")

        # No match — generate new label
        label = self._label_cluster_with_llm(frames)
        if label is not None:
            self._label_cache[centroid_hash] = {
                "label": label,
                "centroid": centroid_vec,
                "first_seen": today,
                "last_used": today,
                "use_count": 1,
            }
            self._save_label_cache()

        return label

    def _load_label_cache(self) -> None:
        """Load the cluster label cache from disk."""
        if self._cluster_label_store is None or not self._cluster_label_store.is_file():
            self._label_cache = {}
            return
        try:
            with open(self._cluster_label_store) as f:
                self._label_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._label_cache = {}

    def _save_label_cache(self) -> None:
        """Save the cluster label cache to disk with atomic write and 30-day pruning."""
        if self._cluster_label_store is None:
            return

        # Prune entries not used in 30+ days
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pruned: dict = {}
        for k, v in self._label_cache.items():
            last_used = v.get("last_used", today)
            try:
                last_dt = datetime.strptime(last_used, "%Y-%m-%d")
                today_dt = datetime.strptime(today, "%Y-%m-%d")
                if (today_dt - last_dt).days <= 30:
                    pruned[k] = v
            except ValueError:
                pruned[k] = v
        self._label_cache = pruned

        # Atomic write: write to tmp then rename
        self._cluster_label_store.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cluster_label_store.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._label_cache, f, indent=2)
            tmp_path.replace(self._cluster_label_store)
        except OSError:
            logger.debug("Failed to save cluster label cache", exc_info=True)

    def _similarity_threshold(self) -> float:
        """The configured clustering threshold, else the profile's.
        Raises ValueError for a model not calibrated yet."""
        if self.config.similarity_threshold is not None:
            return self.config.similarity_threshold
        if self._profile.thresholds is None:
            raise ValueError(
                f"{self._profile.model} has no calibrated segmenter threshold"
            )
        return self._profile.thresholds.segmenter

    def use_embedding_model(self, model: str, vector_kb: "VectorKB | None") -> None:
        """Embed with *model* from now on (its clustering prefix and
        threshold), sharing frame vectors with *vector_kb*: the switch of
        ``embedding_upgrade``, after which the frame vectors it re-embedded
        are this segmenter's cache."""
        self.config.embedding_model = model
        self._profile = embedding_profile(model)
        self._vector_kb = vector_kb

    def _shared_kb(self) -> "VectorKB | None":
        """The vector KB when it embeds with the segmenter's model (the
        frame vectors it holds are then the segmenter's own), else None."""
        vkb = self._vector_kb
        if vkb is None or vkb.profile.model != self._profile.model:
            return None
        return vkb

    def _cached_embeddings(
        self, frames: list[AnnotatedFrame], texts: list[str],
    ) -> list[list[float] | None]:
        """Each frame's stored vector for exactly this text and model, or None."""
        vkb = self._shared_kb()
        if vkb is None:
            return [None] * len(frames)
        cached: list[list[float] | None] = []
        for frame, text in zip(frames, texts):
            try:
                cached.append(
                    vkb.get_current(
                        "annotation", frame.event_id, text,
                        prefix=self._profile.cluster_prefix,
                    )
                    if frame.event_id else None
                )
            except sqlite3.Error:
                logger.debug("Vector KB lookup failed", exc_info=True)
                cached.append(None)
        return cached

    def segment(
        self,
        events: list[dict],
        sleep_boundaries: list[tuple[str, str]] | None = None,
    ) -> SegmentationResult:
        """Run the full segmentation pipeline on a batch of annotated events.

        Args:
            events: List of event dicts from the DB with completed
                annotations (scene_annotation_json, frame_diff_json).
            sleep_boundaries: ``(slept_at, woke_at)`` of each system sleep
                (``WorkerDB.get_sleep_boundaries``); no segment spans one.

        Returns:
            SegmentationResult with task segments grouped by cluster.
        """
        sleep_starts = _sleep_starts(sleep_boundaries)
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

        # Step 2: Compute embeddings — include app, location, and visual
        # proxy when available for richer semantic representation
        texts = [frame_text(f) for f in frames]

        # A frame embedded by an earlier pass keeps its vector in the vector
        # KB (keyed by event id and text); only the others go to Ollama, so
        # the 2-hourly pass over a 24-hour window never re-embeds a frame.
        cached = self._cached_embeddings(frames, texts)
        missing = [i for i, emb in enumerate(cached) if emb is None]
        try:
            embed_start = time.monotonic()
            embeddings = _compute_embeddings(
                [texts[i] for i in missing],
                model=self.config.embedding_model,
                host=self.config.ollama_host,
            ) if missing else []
            result.embedding_time_seconds = time.monotonic() - embed_start

            for i, emb in enumerate(cached):
                if emb is not None:
                    frames[i].embedding = emb
            for i, emb in zip(missing, embeddings):
                frames[i].embedding = emb

            logger.info(
                "Computed %d embeddings in %.1fs (%d reused)",
                len(embeddings), result.embedding_time_seconds,
                len(frames) - len(missing),
            )

            # Store annotation embeddings in vector KB (piggybacks on
            # the embed call — zero extra Ollama cost).  Only into a KB of
            # the same model: the rows are labelled with the KB's model.
            vkb = self._shared_kb()
            if vkb is not None and embeddings:
                try:
                    items = [
                        ("annotation", frames[i].event_id, texts[i])
                        for i, emb in zip(missing, embeddings) if emb
                    ]
                    valid_embs = [emb for emb in embeddings if emb]
                    stored = vkb.upsert_batch(
                        items, embeddings=valid_embs,
                        prefix=self._profile.cluster_prefix,
                    )
                    logger.info(
                        "Vector KB: %d/%d annotation embeddings stored",
                        stored, len(items),
                    )
                except Exception:
                    logger.warning("Vector KB storage failed", exc_info=True)

        except (ConnectionError, OllamaError) as exc:
            logger.warning("Embedding failed (Ollama not reachable): %s", exc)
            # Fall back to app-based clustering (no embeddings)
            return self._fallback_app_clustering(frames, result, sleep_starts)

        # Step 3: Cluster by embedding similarity
        clusters = _cluster_frames(frames, self._similarity_threshold())

        # Step 4: Split clusters into contiguous segments + noise filtering.
        # A cluster's frames split wherever two consecutive frames are more
        # than ``stitch_max_gap_seconds`` apart, so same-cluster segments are
        # never within that gap of each other (nothing left to stitch).
        for cluster_id, frame_indices in clusters.items():
            cluster_frames = [frames[i] for i in frame_indices]

            # Check noise: all non-workflow?
            noise_count = sum(1 for f in cluster_frames if _is_noise_frame(f))
            if noise_count == len(cluster_frames):
                # Pure noise cluster
                result.noise_frames_dropped += len(cluster_frames)
                continue

            segs = _split_into_contiguous_segments(
                cluster_frames, cluster_id,
                self.config.stitch_max_gap_seconds,
                sleep_starts,
            )
            # Step 5: Group segments by cluster for SOP generation
            result.segments.extend(segs)
            result.clusters.setdefault(cluster_id, []).extend(segs)

        # Step 6: Cluster labeling (with persistent cache when available)
        if self._llm_reasoner is not None:
            for cluster_id, cluster_segs in result.clusters.items():
                # Collect all frames from this cluster
                all_cluster_frames: list[AnnotatedFrame] = []
                for seg in cluster_segs:
                    all_cluster_frames.extend(seg.frames)

                if self._cluster_label_store is not None:
                    label = self._get_or_create_cluster_label(all_cluster_frames)
                else:
                    label = self._label_cluster_with_llm(all_cluster_frames)
                if label is not None:
                    for seg in cluster_segs:
                        seg.task_label = label

        logger.info(
            "Segmentation: %d frames → %d clusters, %d segments, "
            "%d noise frames dropped",
            len(frames), len(result.clusters),
            len(result.segments), result.noise_frames_dropped,
        )

        return result

    def _fallback_app_clustering(
        self,
        frames: list[AnnotatedFrame],
        result: SegmentationResult,
        sleep_starts: list[float] | None = None,
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
                sleep_starts,
            )
            result.segments.extend(segs)
            result.clusters.setdefault(cluster_id, []).extend(segs)
            cluster_id += 1

        return result

"""Tests for the v2 task segmenter (passive discovery pipeline)."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from oc_apprentice_worker.task_segmenter import (
    AnnotatedFrame,
    SegmentationResult,
    SegmenterConfig,
    TaskSegment,
    TaskSegmenter,
    _cluster_frames,
    _compute_embeddings,
    _cosine_similarity,
    _make_segment,
    _split_into_contiguous_segments,
    _stitch_interrupted_workflows,
    _timestamp_to_epoch,
    _update_centroid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_id: str = "e1",
    timestamp: str = "2026-03-04T10:00:00Z",
    app: str = "Google Chrome",
    location: str = "https://example.com",
    what_doing: str = "Filing expense report",
    is_workflow: bool = True,
    diff_type: str | None = None,
) -> dict:
    """Create a mock event dict with annotation."""
    annotation = {
        "app": app,
        "location": location,
        "visible_content": {"headings": ["Test"], "labels": [], "values": []},
        "ui_state": {"active_element": "field", "modals_or_popups": "none"},
        "task_context": {
            "what_doing": what_doing,
            "likely_next": "submit",
            "is_workflow": is_workflow,
        },
    }

    diff = None
    if diff_type:
        diff = {"diff_type": diff_type}

    event = {
        "id": event_id,
        "timestamp": timestamp,
        "scene_annotation_json": json.dumps(annotation),
        "annotation_status": "completed",
    }
    if diff:
        event["frame_diff_json"] = json.dumps(diff)

    return event


def _make_frame(
    event_id: str = "e1",
    timestamp: str = "2026-03-04T10:00:00Z",
    what_doing: str = "Filing expense report",
    is_workflow: bool = True,
    app: str = "Chrome",
    embedding: list[float] | None = None,
) -> AnnotatedFrame:
    """Create an AnnotatedFrame with optional embedding."""
    return AnnotatedFrame(
        event_id=event_id,
        timestamp=timestamp,
        annotation={"app": app, "task_context": {"what_doing": what_doing, "is_workflow": is_workflow}},
        what_doing=what_doing,
        is_workflow=is_workflow,
        app=app,
        embedding=embedding or [],
    )


# ---------------------------------------------------------------------------
# AnnotatedFrame.from_event
# ---------------------------------------------------------------------------

class TestAnnotatedFrameFromEvent:
    def test_valid_event(self):
        event = _make_event()
        frame = AnnotatedFrame.from_event(event)
        assert frame is not None
        assert frame.event_id == "e1"
        assert frame.what_doing == "Filing expense report"
        assert frame.is_workflow is True
        assert frame.app == "Google Chrome"

    def test_no_annotation(self):
        event = {"id": "e1", "timestamp": "2026-03-04T10:00:00Z"}
        assert AnnotatedFrame.from_event(event) is None

    def test_empty_annotation(self):
        event = {"id": "e1", "scene_annotation_json": "{}"}
        frame = AnnotatedFrame.from_event(event)
        assert frame is not None
        assert frame.what_doing == ""
        assert frame.is_workflow is False

    def test_invalid_json(self):
        event = {"id": "e1", "scene_annotation_json": "not json"}
        assert AnnotatedFrame.from_event(event) is None

    def test_is_workflow_string_true(self):
        event = _make_event(is_workflow=True)
        # Manually set is_workflow as string
        ann = json.loads(event["scene_annotation_json"])
        ann["task_context"]["is_workflow"] = "true"
        event["scene_annotation_json"] = json.dumps(ann)
        frame = AnnotatedFrame.from_event(event)
        assert frame.is_workflow is True

    def test_is_workflow_string_false(self):
        event = _make_event(is_workflow=False)
        ann = json.loads(event["scene_annotation_json"])
        ann["task_context"]["is_workflow"] = "false"
        event["scene_annotation_json"] = json.dumps(ann)
        frame = AnnotatedFrame.from_event(event)
        assert frame.is_workflow is False

    def test_frame_diff_parsed(self):
        event = _make_event(diff_type="action")
        frame = AnnotatedFrame.from_event(event)
        assert frame.diff is not None
        assert frame.diff["diff_type"] == "action"

    def test_no_diff(self):
        event = _make_event()
        frame = AnnotatedFrame.from_event(event)
        assert frame.diff is None


# ---------------------------------------------------------------------------
# TaskSegment
# ---------------------------------------------------------------------------

class TestTaskSegment:
    def test_frame_count(self):
        seg = TaskSegment(segment_id="s1", cluster_id=0, frames=[
            _make_frame("e1"), _make_frame("e2"), _make_frame("e3"),
        ])
        assert seg.frame_count == 3

    def test_workflow_ratio_all_workflow(self):
        seg = TaskSegment(segment_id="s1", cluster_id=0, frames=[
            _make_frame("e1", is_workflow=True),
            _make_frame("e2", is_workflow=True),
        ])
        assert seg.workflow_ratio == 1.0

    def test_workflow_ratio_mixed(self):
        seg = TaskSegment(segment_id="s1", cluster_id=0, frames=[
            _make_frame("e1", is_workflow=True),
            _make_frame("e2", is_workflow=False),
            _make_frame("e3", is_workflow=True),
            _make_frame("e4", is_workflow=False),
        ])
        assert seg.workflow_ratio == 0.5

    def test_workflow_ratio_empty(self):
        seg = TaskSegment(segment_id="s1", cluster_id=0)
        assert seg.workflow_ratio == 0.0

    def test_to_timeline(self):
        frames = [
            _make_frame("e1", timestamp="2026-03-04T10:00:00Z"),
            _make_frame("e2", timestamp="2026-03-04T10:01:00Z"),
        ]
        frames[0].diff = {"diff_type": "action"}
        seg = TaskSegment(segment_id="s1", cluster_id=0, frames=frames)
        timeline = seg.to_timeline()
        assert len(timeline) == 2
        assert "annotation" in timeline[0]
        assert "diff" in timeline[0]
        assert "timestamp" in timeline[0]
        assert timeline[0]["diff"]["diff_type"] == "action"
        assert timeline[1]["diff"] is None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_similar_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        sim = _cosine_similarity(a, b)
        assert sim > 0.99  # Very similar


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class TestClusterFrames:
    def test_empty(self):
        assert _cluster_frames([]) == {}

    def test_single_frame(self):
        frame = _make_frame("e1", embedding=[1.0, 0.0, 0.0])
        clusters = _cluster_frames([frame], threshold=0.75)
        assert len(clusters) == 1

    def test_two_similar_frames(self):
        """Two nearly identical embeddings should cluster together."""
        f1 = _make_frame("e1", embedding=[1.0, 0.0, 0.0])
        f2 = _make_frame("e2", embedding=[0.95, 0.05, 0.0])
        clusters = _cluster_frames([f1, f2], threshold=0.75)
        assert len(clusters) == 1

    def test_two_different_frames(self):
        """Two orthogonal embeddings should be separate clusters."""
        f1 = _make_frame("e1", embedding=[1.0, 0.0, 0.0])
        f2 = _make_frame("e2", embedding=[0.0, 1.0, 0.0])
        clusters = _cluster_frames([f1, f2], threshold=0.75)
        assert len(clusters) == 2

    def test_three_frames_two_clusters(self):
        """Two similar + one different = 2 clusters."""
        f1 = _make_frame("e1", embedding=[1.0, 0.0, 0.0])
        f2 = _make_frame("e2", embedding=[0.99, 0.01, 0.0])
        f3 = _make_frame("e3", embedding=[0.0, 1.0, 0.0])
        clusters = _cluster_frames([f1, f2, f3], threshold=0.75)
        assert len(clusters) == 2

    def test_frame_without_embedding(self):
        """Frame without embedding goes to its own cluster."""
        f1 = _make_frame("e1", embedding=[1.0, 0.0])
        f2 = _make_frame("e2", embedding=[])
        clusters = _cluster_frames([f1, f2], threshold=0.75)
        assert len(clusters) == 2

    def test_low_threshold_merges_more(self):
        """Lower threshold = more frames in same cluster."""
        f1 = _make_frame("e1", embedding=[1.0, 0.0])
        f2 = _make_frame("e2", embedding=[0.7, 0.7])
        # Cosine similarity ~ 0.7, so threshold 0.5 merges, 0.9 doesn't
        clusters_loose = _cluster_frames([f1, f2], threshold=0.5)
        clusters_strict = _cluster_frames([f1, f2], threshold=0.9)
        assert len(clusters_loose) <= len(clusters_strict)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestTimestampToEpoch:
    def test_valid_z_suffix(self):
        epoch = _timestamp_to_epoch("2026-03-04T10:00:00Z")
        assert epoch > 0

    def test_valid_offset(self):
        epoch = _timestamp_to_epoch("2026-03-04T10:00:00+00:00")
        assert epoch > 0

    def test_empty(self):
        assert _timestamp_to_epoch("") == 0.0

    def test_invalid(self):
        assert _timestamp_to_epoch("not a timestamp") == 0.0


# ---------------------------------------------------------------------------
# Contiguous segment splitting
# ---------------------------------------------------------------------------

class TestSplitIntoContiguousSegments:
    def test_single_frame(self):
        frames = [_make_frame("e1", timestamp="2026-03-04T10:00:00Z")]
        segs = _split_into_contiguous_segments(frames, cluster_id=0)
        assert len(segs) == 1
        assert segs[0].frame_count == 1

    def test_continuous_frames(self):
        """Frames within gap = one segment."""
        frames = [
            _make_frame("e1", timestamp="2026-03-04T10:00:00Z"),
            _make_frame("e2", timestamp="2026-03-04T10:01:00Z"),
            _make_frame("e3", timestamp="2026-03-04T10:02:00Z"),
        ]
        segs = _split_into_contiguous_segments(frames, cluster_id=0, max_gap_seconds=300)
        assert len(segs) == 1
        assert segs[0].frame_count == 3

    def test_gap_splits_segments(self):
        """Large gap between frames = two segments."""
        frames = [
            _make_frame("e1", timestamp="2026-03-04T10:00:00Z"),
            _make_frame("e2", timestamp="2026-03-04T10:01:00Z"),
            _make_frame("e3", timestamp="2026-03-04T11:00:00Z"),  # 1 hour gap
        ]
        segs = _split_into_contiguous_segments(frames, cluster_id=0, max_gap_seconds=1800)
        assert len(segs) == 2
        assert segs[0].frame_count == 2
        assert segs[1].frame_count == 1

    def test_multiple_gaps(self):
        """Multiple large gaps = multiple segments."""
        frames = [
            _make_frame("e1", timestamp="2026-03-04T10:00:00Z"),
            _make_frame("e2", timestamp="2026-03-04T11:00:00Z"),
            _make_frame("e3", timestamp="2026-03-04T12:00:00Z"),
        ]
        segs = _split_into_contiguous_segments(frames, cluster_id=0, max_gap_seconds=1800)
        assert len(segs) == 3

    def test_empty_frames(self):
        segs = _split_into_contiguous_segments([], cluster_id=0)
        assert len(segs) == 0


# ---------------------------------------------------------------------------
# Make segment
# ---------------------------------------------------------------------------

class TestMakeSegment:
    def test_task_label_most_common(self):
        frames = [
            _make_frame("e1", what_doing="Filing expense report"),
            _make_frame("e2", what_doing="Filing expense report"),
            _make_frame("e3", what_doing="Submitting expenses"),
        ]
        seg = _make_segment(frames, cluster_id=0, seq=0)
        assert seg.task_label == "Filing expense report"

    def test_apps_collected(self):
        frames = [
            _make_frame("e1", app="Chrome"),
            _make_frame("e2", app="Finder"),
            _make_frame("e3", app="Chrome"),
        ]
        seg = _make_segment(frames, cluster_id=0, seq=0)
        assert seg.apps_involved == ["Chrome", "Finder"]

    def test_segment_times(self):
        frames = [
            _make_frame("e1", timestamp="2026-03-04T10:00:00Z"),
            _make_frame("e2", timestamp="2026-03-04T10:05:00Z"),
        ]
        seg = _make_segment(frames, cluster_id=0, seq=0)
        assert seg.start_time == "2026-03-04T10:00:00Z"
        assert seg.end_time == "2026-03-04T10:05:00Z"


# ---------------------------------------------------------------------------
# Interrupted workflow stitching
# ---------------------------------------------------------------------------

class TestStitchInterruptedWorkflows:
    def test_no_stitching_needed(self):
        """Single segment per cluster = no stitching."""
        seg = TaskSegment(
            segment_id="s1", cluster_id=0,
            frames=[_make_frame("e1")],
            start_time="2026-03-04T10:00:00Z",
            end_time="2026-03-04T10:05:00Z",
        )
        result = _stitch_interrupted_workflows([seg], [])
        assert len(result) == 1

    def test_merges_close_segments(self):
        """Two segments of same cluster within gap = merge."""
        seg1 = TaskSegment(
            segment_id="s1", cluster_id=0,
            frames=[_make_frame("e1")],
            task_label="expense report",
            apps_involved=["Chrome"],
            start_time="2026-03-04T10:00:00Z",
            end_time="2026-03-04T10:05:00Z",
        )
        seg2 = TaskSegment(
            segment_id="s2", cluster_id=0,
            frames=[_make_frame("e2")],
            task_label="expense report",
            apps_involved=["Chrome"],
            start_time="2026-03-04T10:10:00Z",
            end_time="2026-03-04T10:15:00Z",
        )
        result = _stitch_interrupted_workflows(
            [seg1, seg2], [], max_gap_seconds=1800,
        )
        assert len(result) == 1
        assert result[0].frame_count == 2

    def test_no_merge_across_large_gap(self):
        """Two segments of same cluster > gap = stay separate."""
        seg1 = TaskSegment(
            segment_id="s1", cluster_id=0,
            frames=[_make_frame("e1")],
            start_time="2026-03-04T10:00:00Z",
            end_time="2026-03-04T10:05:00Z",
        )
        seg2 = TaskSegment(
            segment_id="s2", cluster_id=0,
            frames=[_make_frame("e2")],
            start_time="2026-03-04T12:00:00Z",
            end_time="2026-03-04T12:05:00Z",
        )
        result = _stitch_interrupted_workflows(
            [seg1, seg2], [], max_gap_seconds=1800,
        )
        assert len(result) == 2

    def test_different_clusters_not_merged(self):
        """Segments from different clusters are never merged."""
        seg1 = TaskSegment(
            segment_id="s1", cluster_id=0,
            frames=[_make_frame("e1")],
            start_time="2026-03-04T10:00:00Z",
            end_time="2026-03-04T10:05:00Z",
        )
        seg2 = TaskSegment(
            segment_id="s2", cluster_id=1,
            frames=[_make_frame("e2")],
            start_time="2026-03-04T10:10:00Z",
            end_time="2026-03-04T10:15:00Z",
        )
        result = _stitch_interrupted_workflows(
            [seg1, seg2], [], max_gap_seconds=1800,
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TaskSegmenter.segment
# ---------------------------------------------------------------------------

def _mock_embeddings(texts, *, model, host):
    """Generate deterministic mock embeddings based on text content."""
    embeddings = []
    for text in texts:
        # Simple hash-to-vector: similar texts get similar vectors
        if "expense" in text.lower():
            embeddings.append([0.9, 0.1, 0.0, 0.0])
        elif "browsing" in text.lower() or "reading" in text.lower():
            embeddings.append([0.0, 0.9, 0.1, 0.0])
        elif "deploy" in text.lower():
            embeddings.append([0.0, 0.0, 0.9, 0.1])
        elif "chat" in text.lower():
            embeddings.append([0.0, 0.0, 0.1, 0.9])
        else:
            # Default: unique-ish
            import hashlib
            h = hashlib.md5(text.encode()).digest()
            embeddings.append([b / 255.0 for b in h[:4]])
    return embeddings


class TestTaskSegmenterSegment:
    def test_empty_events(self):
        seg = TaskSegmenter()
        result = seg.segment([])
        assert result.total_frames_processed == 0
        assert len(result.segments) == 0

    def test_events_without_annotations(self):
        events = [{"id": "e1", "timestamp": "2026-03-04T10:00:00Z"}]
        seg = TaskSegmenter()
        result = seg.segment(events)
        assert len(result.segments) == 0

    @patch("oc_apprentice_worker.task_segmenter._compute_embeddings", side_effect=_mock_embeddings)
    def test_two_expense_events_cluster(self, mock_embed):
        events = [
            _make_event("e1", "2026-03-04T10:00:00Z", what_doing="Filing expense report"),
            _make_event("e2", "2026-03-04T10:01:00Z", what_doing="Submitting expense form"),
        ]
        seg = TaskSegmenter()
        result = seg.segment(events)
        # Both expense-related → should cluster together
        assert len(result.clusters) == 1
        assert result.segments[0].frame_count == 2

    @patch("oc_apprentice_worker.task_segmenter._compute_embeddings", side_effect=_mock_embeddings)
    def test_mixed_workflow_and_noise(self, mock_embed):
        events = [
            _make_event("e1", "2026-03-04T10:00:00Z",
                        what_doing="Filing expense report", is_workflow=True),
            _make_event("e2", "2026-03-04T10:01:00Z",
                        what_doing="Reading Reddit", is_workflow=False),
            _make_event("e3", "2026-03-04T10:02:00Z",
                        what_doing="Browsing news", is_workflow=False),
        ]
        seg = TaskSegmenter()
        result = seg.segment(events)
        # Expense = workflow, reading/browsing = noise
        assert result.noise_frames_dropped >= 1
        # At least 1 segment for expense
        assert any(s.frame_count > 0 for s in result.segments)

    @patch("oc_apprentice_worker.task_segmenter._compute_embeddings", side_effect=_mock_embeddings)
    def test_noise_only_events(self, mock_embed):
        """All non-workflow events = all noise = no segments."""
        events = [
            _make_event("e1", "2026-03-04T10:00:00Z",
                        what_doing="Reading documentation", is_workflow=False),
            _make_event("e2", "2026-03-04T10:01:00Z",
                        what_doing="Browsing Reddit", is_workflow=False),
        ]
        seg = TaskSegmenter()
        result = seg.segment(events)
        assert len(result.segments) == 0
        assert result.noise_frames_dropped == 2

    @patch("oc_apprentice_worker.task_segmenter._compute_embeddings",
           side_effect=ConnectionError("No Ollama"))
    def test_fallback_on_embedding_failure(self, mock_embed):
        """When embeddings fail, falls back to app-based clustering."""
        events = [
            _make_event("e1", "2026-03-04T10:00:00Z",
                        app="Chrome", what_doing="Expense form", is_workflow=True),
            _make_event("e2", "2026-03-04T10:01:00Z",
                        app="Chrome", what_doing="Submit expense", is_workflow=True),
        ]
        seg = TaskSegmenter()
        result = seg.segment(events)
        # Fallback clusters by app name
        assert len(result.segments) >= 1


# ---------------------------------------------------------------------------
# TaskSegmenter.get_sop_ready_clusters
# ---------------------------------------------------------------------------

class TestGetSopReadyClusters:
    def test_not_enough_demonstrations(self):
        """Single segment = not enough for passive SOP."""
        seg = TaskSegmenter()
        result = SegmentationResult(
            clusters={0: [
                TaskSegment(segment_id="s1", cluster_id=0,
                            frames=[_make_frame("e1")], task_label="expense"),
            ]},
        )
        ready = seg.get_sop_ready_clusters(result)
        assert len(ready) == 0

    def test_two_demonstrations_ready(self):
        """Two segments in same cluster = ready for SOP."""
        seg = TaskSegmenter()
        result = SegmentationResult(
            clusters={0: [
                TaskSegment(segment_id="s1", cluster_id=0,
                            frames=[_make_frame("e1")], task_label="expense"),
                TaskSegment(segment_id="s2", cluster_id=0,
                            frames=[_make_frame("e2")], task_label="expense"),
            ]},
        )
        ready = seg.get_sop_ready_clusters(result)
        assert len(ready) == 1
        task_label, demos = ready[0]
        assert task_label == "expense"
        assert len(demos) == 2

    def test_custom_min_demonstrations(self):
        """Custom min_demonstrations respected."""
        config = SegmenterConfig(min_demonstrations=3)
        seg = TaskSegmenter(config=config)
        result = SegmentationResult(
            clusters={0: [
                TaskSegment(segment_id="s1", cluster_id=0,
                            frames=[_make_frame("e1")], task_label="expense"),
                TaskSegment(segment_id="s2", cluster_id=0,
                            frames=[_make_frame("e2")], task_label="expense"),
            ]},
        )
        ready = seg.get_sop_ready_clusters(result)
        assert len(ready) == 0


# ---------------------------------------------------------------------------
# Update centroid
# ---------------------------------------------------------------------------

class TestUpdateCentroid:
    def test_running_average(self):
        centroids = {0: [1.0, 0.0]}
        _update_centroid(centroids, 0, [0.0, 1.0], cluster_size=2)
        # Average of [1,0] and [0,1] = [0.5, 0.5]
        assert centroids[0] == pytest.approx([0.5, 0.5])

    def test_three_vectors(self):
        centroids = {0: [0.5, 0.5]}  # After 2 vectors
        _update_centroid(centroids, 0, [1.0, 0.0], cluster_size=3)
        # (0.5*2 + 1.0) / 3 = 0.667, (0.5*2 + 0.0) / 3 = 0.333
        assert centroids[0][0] == pytest.approx(2.0 / 3)
        assert centroids[0][1] == pytest.approx(1.0 / 3)


# ---------------------------------------------------------------------------
# DB methods
# ---------------------------------------------------------------------------

def _make_test_db(with_v2_schema: bool = True) -> tuple[sqlite3.Connection, Path]:
    """Create an in-memory SQLite DB with events table."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=wal")
    conn.execute(
        "CREATE TABLE events ("
        "  id TEXT PRIMARY KEY,"
        "  timestamp TEXT NOT NULL,"
        "  kind_json TEXT DEFAULT '{}',"
        "  window_json TEXT DEFAULT '{}',"
        "  metadata_json TEXT DEFAULT '{}',"
        "  processed INTEGER DEFAULT 0,"
        "  artifact_ids_json TEXT DEFAULT '[]'"
        ")"
    )
    if with_v2_schema:
        conn.execute("ALTER TABLE events ADD COLUMN scene_annotation_json TEXT")
        conn.execute("ALTER TABLE events ADD COLUMN annotation_status TEXT DEFAULT 'pending'")
        conn.execute("ALTER TABLE events ADD COLUMN frame_diff_json TEXT")
    conn.commit()
    return conn, Path(tmp.name)


class TestDBPassiveDiscoveryMethods:
    def test_get_annotated_events_in_window(self):
        conn, db_path = _make_test_db()
        try:
            from oc_apprentice_worker.db import WorkerDB
            # Insert annotated events
            ann = json.dumps({"task_context": {"what_doing": "test", "is_workflow": True}})
            conn.execute(
                "INSERT INTO events (id, timestamp, annotation_status, scene_annotation_json) "
                "VALUES (?, datetime('now'), 'completed', ?)",
                ("e1", ann),
            )
            conn.execute(
                "INSERT INTO events (id, timestamp, annotation_status) "
                "VALUES (?, datetime('now'), 'pending')",
                ("e2",),
            )
            conn.commit()

            db = WorkerDB(db_path)
            events = db.get_annotated_events_in_window(hours=1)
            assert len(events) == 1
            assert events[0]["id"] == "e1"
            db.close()
        finally:
            conn.close()
            db_path.unlink(missing_ok=True)

    def test_save_and_get_task_segment(self):
        conn, db_path = _make_test_db()
        try:
            from oc_apprentice_worker.db import WorkerDB
            db = WorkerDB(db_path)

            ok = db.save_task_segment(
                segment_id="seg-0-0",
                cluster_id=0,
                task_label="expense report",
                event_ids=["e1", "e2"],
                apps=["Chrome"],
                start_time="2026-03-04T10:00:00Z",
                end_time="2026-03-04T10:05:00Z",
            )
            assert ok is True

            segments = db.get_cluster_segments(0)
            assert len(segments) == 1
            assert segments[0]["segment_id"] == "seg-0-0"
            assert segments[0]["task_label"] == "expense report"
            db.close()
        finally:
            conn.close()
            db_path.unlink(missing_ok=True)

    def test_mark_segment_sop_generated(self):
        conn, db_path = _make_test_db()
        try:
            from oc_apprentice_worker.db import WorkerDB
            db = WorkerDB(db_path)

            db.save_task_segment(
                segment_id="seg-0-0", cluster_id=0, task_label="test",
                event_ids=["e1"], apps=[], start_time="", end_time="",
            )
            db.save_task_segment(
                segment_id="seg-0-1", cluster_id=0, task_label="test",
                event_ids=["e2"], apps=[], start_time="", end_time="",
            )

            # Before marking: both visible
            assert len(db.get_cluster_segments(0)) == 2

            db.mark_segment_sop_generated("seg-0-0")

            # After marking: only one visible (sop_generated=0 filter)
            segs = db.get_cluster_segments(0)
            assert len(segs) == 1
            assert segs[0]["segment_id"] == "seg-0-1"
            db.close()
        finally:
            conn.close()
            db_path.unlink(missing_ok=True)

    def test_get_sop_pending_clusters(self):
        conn, db_path = _make_test_db()
        try:
            from oc_apprentice_worker.db import WorkerDB
            db = WorkerDB(db_path)

            # Create cluster with 2 segments
            db.save_task_segment(
                segment_id="seg-0-0", cluster_id=0, task_label="expense",
                event_ids=["e1"], apps=[], start_time="", end_time="",
            )
            db.save_task_segment(
                segment_id="seg-0-1", cluster_id=0, task_label="expense",
                event_ids=["e2"], apps=[], start_time="", end_time="",
            )
            # Cluster with only 1 segment
            db.save_task_segment(
                segment_id="seg-1-0", cluster_id=1, task_label="deploy",
                event_ids=["e3"], apps=[], start_time="", end_time="",
            )

            pending = db.get_sop_pending_clusters()
            assert len(pending) == 1
            assert pending[0]["cluster_id"] == 0
            assert pending[0]["seg_count"] == 2
            db.close()
        finally:
            conn.close()
            db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration: _process_passive_discovery
# ---------------------------------------------------------------------------

class TestProcessPassiveDiscovery:
    @patch("oc_apprentice_worker.task_segmenter._compute_embeddings", side_effect=_mock_embeddings)
    def test_full_pipeline_generates_sop(self, mock_embed):
        """End-to-end: events → segment → generate → export."""
        conn, db_path = _make_test_db()
        try:
            from oc_apprentice_worker.db import WorkerDB
            from oc_apprentice_worker.main import _process_passive_discovery

            # Insert multiple expense-related annotated events
            for i in range(4):
                ann = json.dumps({
                    "app": "Chrome",
                    "location": "https://expensify.com",
                    "task_context": {
                        "what_doing": "Filing expense report",
                        "is_workflow": True,
                    },
                })
                minute = f"{i:02d}"
                # Two separate time segments (demonstrations)
                if i < 2:
                    ts = f"2026-03-04T10:{minute}:00Z"
                else:
                    ts = f"2026-03-04T14:{minute}:00Z"  # 4 hours later
                conn.execute(
                    "INSERT INTO events "
                    "(id, timestamp, annotation_status, scene_annotation_json) "
                    "VALUES (?, ?, 'completed', ?)",
                    (f"e{i}", ts, ann),
                )
            conn.commit()

            db = WorkerDB(db_path)

            # Mock SOP generator
            from oc_apprentice_worker.sop_generator import SOPGenerator, GeneratedSOP
            mock_sop_gen = MagicMock(spec=SOPGenerator)
            mock_sop_gen.generate_from_passive.return_value = GeneratedSOP(
                sop={"slug": "expense-report", "title": "Expense Report",
                     "steps": [{"step": "open form"}], "variables": []},
                inference_time_seconds=10.0,
            )

            segmenter = TaskSegmenter(config=SegmenterConfig(
                stitch_max_gap_seconds=1800,
                default_window_hours=24,
            ))

            mock_writer = MagicMock()
            mock_writer.write_all_sops.return_value = ["/fake/path.md"]
            mock_writer.get_sops_dir.return_value = Path("/fake")
            mock_index = MagicMock()

            sops = _process_passive_discovery(
                db,
                segmenter=segmenter,
                sop_generator=mock_sop_gen,
                openclaw_writer=mock_writer,
                index_generator=mock_index,
            )

            assert sops >= 1
            mock_sop_gen.generate_from_passive.assert_called_once()
            mock_writer.write_all_sops.assert_called()
            db.close()
        finally:
            conn.close()
            db_path.unlink(missing_ok=True)

    @patch("oc_apprentice_worker.task_segmenter._compute_embeddings", side_effect=_mock_embeddings)
    def test_no_sop_with_single_demonstration(self, mock_embed):
        """Only 1 demonstration = no SOP generated."""
        conn, db_path = _make_test_db()
        try:
            from oc_apprentice_worker.db import WorkerDB
            from oc_apprentice_worker.main import _process_passive_discovery

            # Insert 2 close events (same demonstration)
            for i in range(2):
                ann = json.dumps({
                    "app": "Chrome",
                    "location": "https://expensify.com",
                    "task_context": {
                        "what_doing": "Filing expense report",
                        "is_workflow": True,
                    },
                })
                conn.execute(
                    "INSERT INTO events "
                    "(id, timestamp, annotation_status, scene_annotation_json) "
                    "VALUES (?, ?, 'completed', ?)",
                    (f"e{i}", f"2026-03-04T10:0{i}:00Z", ann),
                )
            conn.commit()

            db = WorkerDB(db_path)
            mock_sop_gen = MagicMock()

            segmenter = TaskSegmenter(config=SegmenterConfig(
                default_window_hours=24,
            ))

            mock_writer = MagicMock()
            mock_writer.get_sops_dir.return_value = Path("/fake")
            mock_index = MagicMock()

            sops = _process_passive_discovery(
                db,
                segmenter=segmenter,
                sop_generator=mock_sop_gen,
                openclaw_writer=mock_writer,
                index_generator=mock_index,
            )

            assert sops == 0
            mock_sop_gen.generate_from_passive.assert_not_called()
            db.close()
        finally:
            conn.close()
            db_path.unlink(missing_ok=True)

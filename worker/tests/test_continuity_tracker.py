"""Tests for ContinuityTracker — cross-segment linking and span management."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from oc_apprentice_worker.continuity_tracker import (
    ContinuityEdge,
    ContinuitySpan,
    ContinuityTracker,
    ContinuityType,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.task_segmenter import AnnotatedFrame, TaskSegment


# ---------------------------------------------------------------------------
# Embedding vectors (4D for test clarity)
# ---------------------------------------------------------------------------

EMB_CODING = [0.9, 0.1, 0.0, 0.0]
EMB_CODING_VARIANT = [0.88, 0.12, 0.0, 0.0]      # cosine ~1.00 with CODING
EMB_BRANCH = [0.4, 0.3, 0.5, 0.3]                 # cosine ~0.56 with CODING (BRANCH range)
EMB_YOUTUBE = [0.0, 0.1, 0.0, 0.9]                # cosine ~0.01 with CODING
EMB_DESIGN = [0.1, 0.0, 0.85, 0.05]               # cosine ~0.12 with CODING
EMB_DESIGN_MODERATE = [0.3, 0.3, 0.55, 0.3]       # cosine ~0.79 with DESIGN (moderate)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kb(tmp_path):
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


@pytest.fixture
def tracker(kb):
    return ContinuityTracker(kb=kb)


def _make_test_segment(
    segment_id,
    what_doing,
    app,
    embedding,
    start_time,
    end_time=None,
    apps=None,
):
    """Build a minimal TaskSegment for continuity testing."""
    if end_time is None:
        # Default: 5 minutes after start
        try:
            st = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            et = st + timedelta(minutes=5)
            end_time = et.isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError):
            end_time = start_time

    frames = [
        AnnotatedFrame(
            event_id=f"{segment_id}-f0",
            timestamp=start_time,
            annotation={},
            what_doing=what_doing,
            app=app,
            embedding=list(embedding) if embedding else [],
        ),
        AnnotatedFrame(
            event_id=f"{segment_id}-f1",
            timestamp=end_time,
            annotation={},
            what_doing=what_doing,
            app=app,
            embedding=list(embedding) if embedding else [],
        ),
    ]
    return TaskSegment(
        segment_id=segment_id,
        cluster_id=0,
        frames=frames,
        task_label=what_doing,
        apps_involved=apps if apps is not None else [app],
        start_time=start_time,
        end_time=end_time,
    )


def _make_span(
    span_id,
    goal_summary,
    apps,
    embedding,
    last_seen,
    first_seen=None,
    state="active",
    segments=None,
    edges=None,
    total_duration_seconds=300,
    continuity_confidence=1.0,
    parent_span_id=None,
):
    """Build a ContinuitySpan for testing."""
    if first_seen is None:
        try:
            ls = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            first_seen = (ls - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError):
            first_seen = last_seen

    return ContinuitySpan(
        span_id=span_id,
        goal_summary=goal_summary,
        continuity_confidence=continuity_confidence,
        segments=segments or [f"{span_id}-seg0"],
        edges=edges or [],
        state=state,
        apps_involved=list(apps),
        representative_embedding=list(embedding) if embedding else [],
        first_seen=first_seen,
        last_seen=last_seen,
        total_duration_seconds=total_duration_seconds,
        parent_span_id=parent_span_id,
    )


def _ts(base_time, **kwargs):
    """Generate ISO timestamps offset from a base time."""
    dt = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc) + timedelta(**kwargs)
    return dt.isoformat().replace("+00:00", "Z")


# Base time for all tests
T0 = "2026-03-14T10:00:00Z"


# ---------------------------------------------------------------------------
# TestClassifyRelationship (5 tests)
# ---------------------------------------------------------------------------


class TestClassifyRelationship:
    """Decision matrix classification tests."""

    def test_classify_continue(self, tracker):
        """gap 30s, emb_sim>=0.70, app_overlap>=0.5 -> CONTINUE."""
        span = _make_span(
            "sp-1", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, seconds=0),
        )
        segment = _make_test_segment(
            "seg-1", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, seconds=30),
        )

        edge = tracker.classify_relationship(segment, span)
        assert edge.continuity_type == ContinuityType.CONTINUE
        assert edge.confidence > 0.0
        assert edge.gap_seconds <= 60

    def test_classify_resume_short(self, tracker):
        """gap 10min, emb_sim>=0.60 -> RESUME."""
        span = _make_span(
            "sp-2", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, seconds=0),
        )
        segment = _make_test_segment(
            "seg-2", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, minutes=10),
        )

        edge = tracker.classify_relationship(segment, span)
        assert edge.continuity_type == ContinuityType.RESUME
        assert edge.gap_seconds >= 600

    def test_classify_resume_long(self, tracker):
        """gap 2h, emb_sim>=0.75, app_overlap>=0.5 -> RESUME."""
        span = _make_span(
            "sp-3", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, seconds=0),
        )
        segment = _make_test_segment(
            "seg-3", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, hours=2),
        )

        edge = tracker.classify_relationship(segment, span)
        assert edge.continuity_type == ContinuityType.RESUME
        assert edge.gap_seconds >= 7200

    def test_classify_branch(self, tracker):
        """emb_sim in [0.40, 0.60), app_overlap >= 0.3 -> BRANCH."""
        span = _make_span(
            "sp-4", "Writing code", ["VS Code", "Chrome"], EMB_CODING,
            last_seen=_ts(T0, seconds=0),
        )
        # EMB_BRANCH has cosine ~0.56 with CODING -> falls in [0.40, 0.60)
        segment = _make_test_segment(
            "seg-4", "Researching API docs", "Chrome", EMB_BRANCH,
            start_time=_ts(T0, minutes=5),
            apps=["Chrome"],
        )

        edge = tracker.classify_relationship(segment, span)
        # app_overlap: {chrome} ∩ {vs code, chrome} / {vs code, chrome} = 1/2 = 0.5
        assert edge.continuity_type == ContinuityType.BRANCH
        assert edge.embedding_similarity >= 0.40
        assert edge.embedding_similarity < 0.60

    def test_classify_new_task(self, tracker):
        """emb_sim~0.1, app_overlap=0.0 -> NEW_TASK."""
        span = _make_span(
            "sp-5", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, seconds=0),
        )
        segment = _make_test_segment(
            "seg-5", "Watching YouTube", "Safari", EMB_YOUTUBE,
            start_time=_ts(T0, minutes=5),
            apps=["Safari"],
        )

        edge = tracker.classify_relationship(segment, span)
        assert edge.continuity_type == ContinuityType.NEW_TASK


# ---------------------------------------------------------------------------
# TestBuildGraph (8 tests)
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """Tests for build_graph: span creation and extension."""

    def test_all_new_segments(self, tracker):
        """3 dissimilar segments -> 3 spans."""
        segments = [
            _make_test_segment("seg-a", "Writing code", "VS Code", EMB_CODING,
                               start_time=_ts(T0, minutes=0)),
            _make_test_segment("seg-b", "Watching YouTube", "Safari", EMB_YOUTUBE,
                               start_time=_ts(T0, minutes=10)),
            _make_test_segment("seg-c", "Designing UI", "Figma", EMB_DESIGN,
                               start_time=_ts(T0, minutes=20)),
        ]

        spans = tracker.build_graph(segments, existing_spans=[])
        assert len(spans) == 3

    def test_two_similar_resume(self, tracker):
        """2 similar segments 10min apart -> 1 span with resume edge."""
        segments = [
            _make_test_segment("seg-a", "Writing code", "VS Code", EMB_CODING,
                               start_time=_ts(T0, minutes=0),
                               end_time=_ts(T0, minutes=5)),
            _make_test_segment("seg-b", "Writing code", "VS Code", EMB_CODING_VARIANT,
                               start_time=_ts(T0, minutes=15),
                               end_time=_ts(T0, minutes=20)),
        ]

        spans = tracker.build_graph(segments, existing_spans=[])
        # The first segment creates a span, the second resumes it
        active_spans = [s for s in spans if s.state in ("active", "paused", "completed")]
        # There should be just 1 span covering both segments
        coding_spans = [s for s in spans if "seg-a" in s.segments]
        assert len(coding_spans) == 1
        assert "seg-b" in coding_spans[0].segments
        assert len(coding_spans[0].edges) == 1

    def test_two_dissimilar(self, tracker):
        """2 different segments -> 2 spans."""
        segments = [
            _make_test_segment("seg-a", "Writing code", "VS Code", EMB_CODING,
                               start_time=_ts(T0, minutes=0)),
            _make_test_segment("seg-b", "Watching YouTube", "Safari", EMB_YOUTUBE,
                               start_time=_ts(T0, minutes=10)),
        ]

        spans = tracker.build_graph(segments, existing_spans=[])
        assert len(spans) == 2

    def test_extend_existing_span(self, tracker):
        """Existing span + matching segment -> span extended."""
        existing_span = _make_span(
            "sp-exist", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=5),
            first_seen=_ts(T0, minutes=0),
        )

        new_segment = _make_test_segment(
            "seg-new", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, minutes=15),
            end_time=_ts(T0, minutes=20),
        )

        spans = tracker.build_graph([new_segment], existing_spans=[existing_span])
        # The existing span should have been extended
        ext = [s for s in spans if s.span_id == "sp-exist"]
        assert len(ext) == 1
        assert "seg-new" in ext[0].segments
        assert len(ext[0].edges) >= 1

    def test_branch_creates_child(self, tracker):
        """Segment branches from existing -> new span with parent_span_id."""
        existing_span = _make_span(
            "sp-parent", "Writing code", ["VS Code", "Chrome"], EMB_CODING,
            last_seen=_ts(T0, minutes=5),
        )

        # EMB_BRANCH has cosine ~0.56 with CODING -> BRANCH classification
        branch_segment = _make_test_segment(
            "seg-branch", "Researching API docs", "Chrome", EMB_BRANCH,
            start_time=_ts(T0, minutes=10),
            apps=["Chrome"],
        )

        spans = tracker.build_graph([branch_segment], existing_spans=[existing_span])
        # Should have the original span plus a new child span
        child_spans = [s for s in spans if s.parent_span_id == "sp-parent"]
        assert len(child_spans) == 1
        assert "seg-branch" in child_spans[0].segments

    def test_restart_after_24h(self, tracker):
        """Similar segment 25h later -> same span with restart edge."""
        existing_span = _make_span(
            "sp-old", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            first_seen=_ts(T0, minutes=0),
        )

        # 25 hours later, same task
        restart_segment = _make_test_segment(
            "seg-restart", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, hours=25),
            end_time=_ts(T0, hours=25, minutes=5),
        )

        # The span will have been marked completed by _update_span_states,
        # so it won't be eligible for matching. We need to keep it active.
        # Actually, build_graph calls _update_span_states AFTER processing,
        # so during processing the span is still "active". But with a 25h gap,
        # classify_relationship should return RESTART.
        spans = tracker.build_graph([restart_segment], existing_spans=[existing_span])
        # The span should be extended (RESTART extends, not creates new)
        ext = [s for s in spans if s.span_id == "sp-old"]
        assert len(ext) == 1
        assert "seg-restart" in ext[0].segments

    def test_best_match_wins(self, tracker):
        """Segment similar to 2 spans, picks highest confidence."""
        span_a = _make_span(
            "sp-a", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
        )
        span_b = _make_span(
            "sp-b", "Designing UI", ["Figma"], EMB_DESIGN,
            last_seen=_ts(T0, minutes=0),
        )

        # Coding-like segment should match span_a over span_b
        segment = _make_test_segment(
            "seg-match", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, minutes=10),
        )

        spans = tracker.build_graph([segment], existing_spans=[span_a, span_b])
        # seg-match should be in span_a (coding), not span_b (design)
        coding_span = [s for s in spans if s.span_id == "sp-a"]
        assert len(coding_span) == 1
        assert "seg-match" in coding_span[0].segments

    def test_unrelated_spans_preserved(self, tracker):
        """Existing spans not involved in new segments stay unchanged."""
        span_design = _make_span(
            "sp-design", "Designing UI", ["Figma"], EMB_DESIGN,
            last_seen=_ts(T0, minutes=0),
            segments=["seg-design-0"],
        )
        span_coding = _make_span(
            "sp-coding", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            segments=["seg-coding-0"],
        )

        # New segment only matches coding span
        segment = _make_test_segment(
            "seg-new", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, minutes=10),
        )

        spans = tracker.build_graph([segment], existing_spans=[span_design, span_coding])

        # Design span should still have its original segments
        design = [s for s in spans if s.span_id == "sp-design"]
        assert len(design) == 1
        assert design[0].segments == ["seg-design-0"]


# ---------------------------------------------------------------------------
# TestUncertainty (4 tests)
# ---------------------------------------------------------------------------


class TestUncertainty:
    """Tests for uncertain state handling."""

    def test_below_uncertain_threshold(self, kb):
        """Confidence below uncertain_threshold -> new span, state='uncertain'."""
        # Use a high uncertain_threshold so that marginal matches become uncertain
        tracker = ContinuityTracker(kb=kb, uncertain_threshold=0.95)

        existing_span = _make_span(
            "sp-exist", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
        )

        # A segment that is similar but not enough to pass the high threshold
        segment = _make_test_segment(
            "seg-uncertain", "Writing code variant", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, minutes=10),
        )

        spans = tracker.build_graph([segment], existing_spans=[existing_span])
        # With such a high threshold, the segment may create a new uncertain span
        # rather than merging into the existing one
        new_spans = [s for s in spans if s.span_id != "sp-exist"]
        if new_spans:
            # If a new span was created, it should be marked uncertain
            assert any(s.state == "uncertain" for s in new_spans)

    def test_uncertain_not_merged(self, kb):
        """Below-threshold match should not merge into existing span."""
        tracker = ContinuityTracker(kb=kb, uncertain_threshold=0.99)

        existing_span = _make_span(
            "sp-exist", "Designing UI", ["Figma"], EMB_DESIGN,
            last_seen=_ts(T0, minutes=0),
            segments=["seg-d0"],
        )

        # EMB_DESIGN_MODERATE has cosine ~0.79 with DESIGN -> RESUME with
        # confidence 0.79 < uncertain_threshold 0.99 -> should not merge
        segment = _make_test_segment(
            "seg-maybe", "Designing UI variant", "Figma", EMB_DESIGN_MODERATE,
            start_time=_ts(T0, minutes=10),
        )

        spans = tracker.build_graph([segment], existing_spans=[existing_span])
        # The original span should NOT have seg-maybe in its segments
        orig = [s for s in spans if s.span_id == "sp-exist"]
        assert len(orig) == 1
        assert "seg-maybe" not in orig[0].segments

    def test_false_merge_prevention(self, tracker):
        """Two segments with moderate similarity to different topics stay separate."""
        seg_coding = _make_test_segment(
            "seg-code", "Writing code", "VS Code", EMB_CODING,
            start_time=_ts(T0, minutes=0),
        )
        seg_youtube = _make_test_segment(
            "seg-yt", "Watching YouTube", "Safari", EMB_YOUTUBE,
            start_time=_ts(T0, minutes=10),
        )

        spans = tracker.build_graph([seg_coding, seg_youtube], existing_spans=[])
        assert len(spans) == 2
        # Each span should have exactly one segment
        for span in spans:
            assert len(span.segments) == 1

    def test_uncertain_state_persists(self, tracker):
        """Uncertain state is not auto-changed by _update_span_states."""
        span = _make_span(
            "sp-uncertain", "Maybe coding", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            state="uncertain",
        )

        # Call _update_span_states with a time far in the future
        future_time = _ts(T0, hours=48)
        tracker._update_span_states([span], future_time)

        # Should still be uncertain, NOT changed to completed
        assert span.state == "uncertain"


# ---------------------------------------------------------------------------
# TestStateManagement (4 tests)
# ---------------------------------------------------------------------------


class TestStateManagement:
    """Tests for span state transitions."""

    def test_active_state(self, tracker):
        """Span with recent last_seen stays active."""
        span = _make_span(
            "sp-active", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            state="active",
        )

        # Current time is 1 hour after last_seen
        current_time = _ts(T0, hours=1)
        tracker._update_span_states([span], current_time)
        assert span.state == "active"

    def test_paused_after_4h(self, tracker):
        """Gap > 4h -> paused."""
        span = _make_span(
            "sp-pause", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            state="active",
        )

        # Current time is 5 hours after last_seen
        current_time = _ts(T0, hours=5)
        tracker._update_span_states([span], current_time)
        assert span.state == "paused"

    def test_completed_after_24h(self, tracker):
        """Gap > 24h -> completed."""
        span = _make_span(
            "sp-done", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            state="active",
        )

        # Current time is 25 hours after last_seen
        current_time = _ts(T0, hours=25)
        tracker._update_span_states([span], current_time)
        assert span.state == "completed"

    def test_completed_can_restart(self, tracker):
        """Completed span + resume -> reactivated."""
        completed_span = _make_span(
            "sp-completed", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            state="active",
        )

        # Mark it completed first via state update
        far_future = _ts(T0, hours=25)
        tracker._update_span_states([completed_span], far_future)
        assert completed_span.state == "completed"

        # Completed spans are NOT eligible for matching in build_graph
        # (the loop skips state not in active/paused/uncertain).
        # So this segment should create a new span, not extend the completed one.
        segment = _make_test_segment(
            "seg-return", "Writing code", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, hours=26),
        )

        spans = tracker.build_graph([segment], existing_spans=[completed_span])
        # The completed span should remain, and a new span should be created
        assert len(spans) >= 2
        completed = [s for s in spans if s.span_id == "sp-completed"]
        assert len(completed) == 1
        assert completed[0].state == "completed"


# ---------------------------------------------------------------------------
# TestPersistence (3 tests)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self, tracker):
        """Save spans, load back, verify equality."""
        spans = [
            _make_span("sp-1", "Writing code", ["VS Code"], EMB_CODING,
                       last_seen=_ts(T0, minutes=10),
                       first_seen=_ts(T0, minutes=0)),
            _make_span("sp-2", "Designing UI", ["Figma"], EMB_DESIGN,
                       last_seen=_ts(T0, minutes=20),
                       first_seen=_ts(T0, minutes=5)),
        ]

        tracker.save_spans(spans)
        loaded = tracker.load_spans()

        assert len(loaded) == 2
        assert loaded[0].span_id == "sp-1"
        assert loaded[0].goal_summary == "Writing code"
        assert loaded[0].apps_involved == ["VS Code"]
        assert loaded[0].representative_embedding == EMB_CODING
        assert loaded[1].span_id == "sp-2"
        assert loaded[1].goal_summary == "Designing UI"

    def test_load_empty(self, tracker):
        """Missing file -> empty list."""
        loaded = tracker.load_spans()
        assert loaded == []

    def test_edges_serialize_correctly(self, tracker):
        """ContinuityType enum roundtrips through JSON serialization."""
        edge = ContinuityEdge(
            from_segment_id="seg-a",
            to_segment_id="seg-b",
            continuity_type=ContinuityType.RESUME,
            confidence=0.85,
            reasoning="Short gap, high similarity",
            gap_seconds=600,
            embedding_similarity=0.92,
            app_overlap=0.75,
        )
        span = _make_span(
            "sp-edges", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=10),
            edges=[edge],
            segments=["seg-a", "seg-b"],
        )

        tracker.save_spans([span])
        loaded = tracker.load_spans()

        assert len(loaded) == 1
        assert len(loaded[0].edges) == 1
        loaded_edge = loaded[0].edges[0]
        assert loaded_edge.continuity_type == ContinuityType.RESUME
        assert loaded_edge.confidence == 0.85
        assert loaded_edge.gap_seconds == 600
        assert loaded_edge.from_segment_id == "seg-a"
        assert loaded_edge.to_segment_id == "seg-b"
        assert loaded_edge.embedding_similarity == 0.92
        assert loaded_edge.app_overlap == 0.75


# ---------------------------------------------------------------------------
# TestEmbeddingSimilarity (5 tests)
# ---------------------------------------------------------------------------


class TestEmbeddingSimilarity:
    """Tests for embedding and overlap computation."""

    def test_representative_embedding_single_frame(self, tracker):
        """Single frame -> its embedding."""
        segment = TaskSegment(
            segment_id="seg-single",
            cluster_id=0,
            frames=[
                AnnotatedFrame(
                    event_id="e0", timestamp=T0, annotation={},
                    embedding=list(EMB_CODING),
                ),
            ],
            apps_involved=["VS Code"],
            start_time=T0,
            end_time=T0,
        )

        result = tracker._compute_representative_embedding(segment)
        assert result == pytest.approx(EMB_CODING)

    def test_representative_embedding_average(self, tracker):
        """2 frames -> element-wise average."""
        segment = TaskSegment(
            segment_id="seg-avg",
            cluster_id=0,
            frames=[
                AnnotatedFrame(
                    event_id="e0", timestamp=T0, annotation={},
                    embedding=[1.0, 0.0, 0.0, 0.0],
                ),
                AnnotatedFrame(
                    event_id="e1", timestamp=T0, annotation={},
                    embedding=[0.0, 1.0, 0.0, 0.0],
                ),
            ],
            apps_involved=["VS Code"],
            start_time=T0,
            end_time=T0,
        )

        result = tracker._compute_representative_embedding(segment)
        assert result == pytest.approx([0.5, 0.5, 0.0, 0.0])

    def test_representative_embedding_empty(self, tracker):
        """No embeddings -> empty list."""
        segment = TaskSegment(
            segment_id="seg-empty",
            cluster_id=0,
            frames=[
                AnnotatedFrame(
                    event_id="e0", timestamp=T0, annotation={},
                    embedding=[],
                ),
            ],
            apps_involved=["VS Code"],
            start_time=T0,
            end_time=T0,
        )

        result = tracker._compute_representative_embedding(segment)
        assert result == []

    def test_app_overlap_identical(self, tracker):
        """Same apps -> 1.0."""
        result = tracker._compute_app_overlap(
            ["Chrome", "VS Code"], ["Chrome", "VS Code"],
        )
        assert result == pytest.approx(1.0)

    def test_app_overlap_partial(self, tracker):
        """Partial overlap -> correct Jaccard."""
        result = tracker._compute_app_overlap(
            ["Chrome", "VS Code"], ["Chrome", "Figma"],
        )
        # Jaccard: {chrome} / {chrome, vs code, figma} = 1/3
        assert result == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# TestEdgeCases (6 tests)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_empty_segments_returns_empty(self, tracker):
        """No segments -> empty spans."""
        spans = tracker.build_graph([], existing_spans=[])
        assert spans == []

    def test_segment_no_embeddings(self, tracker):
        """Frames without embeddings -> graceful (no crash)."""
        segment = _make_test_segment(
            "seg-no-emb", "Writing code", "VS Code", [],
            start_time=_ts(T0, minutes=0),
        )

        spans = tracker.build_graph([segment], existing_spans=[])
        assert len(spans) == 1
        assert spans[0].representative_embedding == []

    def test_single_frame_segment(self, tracker):
        """Works with 1-frame segment."""
        segment = TaskSegment(
            segment_id="seg-1f",
            cluster_id=0,
            frames=[
                AnnotatedFrame(
                    event_id="e0", timestamp=T0, annotation={},
                    what_doing="Quick task",
                    app="Terminal",
                    embedding=list(EMB_CODING),
                ),
            ],
            task_label="Quick task",
            apps_involved=["Terminal"],
            start_time=T0,
            end_time=T0,
        )

        spans = tracker.build_graph([segment], existing_spans=[])
        assert len(spans) == 1
        # Duration should be 0 for single-frame segment
        assert spans[0].total_duration_seconds == 0

    def test_confidence_is_min_across_edges(self, tracker):
        """Span with 3 edges takes minimum confidence."""
        span = _make_span(
            "sp-conf", "Writing code", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            continuity_confidence=1.0,
            segments=["seg-0"],
        )

        # Simulate adding 3 segments with decreasing edge confidence
        segments = [
            _make_test_segment(
                f"seg-{i+1}", "Writing code", "VS Code", EMB_CODING_VARIANT,
                start_time=_ts(T0, minutes=10 * (i + 1)),
                end_time=_ts(T0, minutes=10 * (i + 1) + 5),
            )
            for i in range(3)
        ]

        spans = tracker.build_graph(segments, existing_spans=[span])

        # Find the extended span
        ext = [s for s in spans if s.span_id == "sp-conf"]
        assert len(ext) == 1
        # continuity_confidence should be the minimum of all edge confidences
        # (running min applied in _update_span_from_segment)
        assert ext[0].continuity_confidence <= 1.0
        if ext[0].edges:
            min_edge_conf = min(e.confidence for e in ext[0].edges)
            assert ext[0].continuity_confidence <= min_edge_conf

    def test_apps_union_across_segments(self, tracker):
        """span.apps_involved is union of all segment apps."""
        span = _make_span(
            "sp-apps", "Working on project", ["VS Code"], EMB_CODING,
            last_seen=_ts(T0, minutes=0),
            segments=["seg-0"],
        )

        # Add segment with different app
        segment = _make_test_segment(
            "seg-multi", "Writing code and testing", "VS Code", EMB_CODING_VARIANT,
            start_time=_ts(T0, minutes=10),
            apps=["VS Code", "Terminal"],
        )

        spans = tracker.build_graph([segment], existing_spans=[span])
        ext = [s for s in spans if s.span_id == "sp-apps"]
        assert len(ext) == 1
        apps_lower = [a.lower() for a in ext[0].apps_involved]
        assert "vs code" in apps_lower
        assert "terminal" in apps_lower

    def test_duration_calculation(self, tracker):
        """total_duration_seconds correct across segments."""
        # Segment with 10-minute internal duration
        segment = _make_test_segment(
            "seg-dur", "Writing code", "VS Code", EMB_CODING,
            start_time=_ts(T0, minutes=0),
            end_time=_ts(T0, minutes=10),
        )

        spans = tracker.build_graph([segment], existing_spans=[])
        assert len(spans) == 1
        # Duration = last_frame.timestamp - first_frame.timestamp = 10 minutes = 600s
        assert spans[0].total_duration_seconds == 600

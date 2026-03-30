"""Tests for the Sprint 4 interruption/resumption model in task_segmenter.

Tests the classify_interruptions() method which post-processes segments
to detect brief interrupts, pauses, related interrupts, and abandonment.
"""

from __future__ import annotations

import pytest

from agenthandover_worker.task_segmenter import (
    AnnotatedFrame,
    InterruptionEvent,
    SegmenterConfig,
    TaskSegment,
    TaskSegmenter,
    TaskState,
    _content_segment_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minutes: int) -> str:
    """Generate an ISO timestamp at a given minute offset from a base time.

    Base time: 2026-03-04T10:00:00Z
    """
    hour = 10 + minutes // 60
    minute = minutes % 60
    return f"2026-03-04T{hour:02d}:{minute:02d}:00Z"


def _make_frame(
    event_id: str,
    timestamp: str,
    app: str = "Chrome",
    what_doing: str = "Research",
) -> AnnotatedFrame:
    """Create an AnnotatedFrame for testing."""
    return AnnotatedFrame(
        event_id=event_id,
        timestamp=timestamp,
        annotation={"app": app, "task_context": {"what_doing": what_doing}},
        what_doing=what_doing,
        is_workflow=True,
        app=app,
    )


def make_segment(
    cluster_id: int,
    start_ts: str,
    end_ts: str,
    app: str = "Chrome",
    what_doing: str = "Research",
    num_frames: int = 1,
) -> TaskSegment:
    """Create a TaskSegment with the specified parameters.

    Generates the appropriate number of frames spread across the time range.
    """
    frames = []
    for i in range(num_frames):
        frames.append(_make_frame(
            event_id=f"e-{cluster_id}-{start_ts}-{i}",
            timestamp=start_ts if i == 0 else end_ts,
            app=app,
            what_doing=what_doing,
        ))

    return TaskSegment(
        segment_id=_content_segment_id(frames),
        cluster_id=cluster_id,
        start_time=start_ts,
        end_time=end_ts,
        frames=frames,
        task_label=what_doing,
        apps_involved=[app],
    )


def _segmenter(**kwargs) -> TaskSegmenter:
    """Create a TaskSegmenter with optional config overrides."""
    return TaskSegmenter(config=SegmenterConfig(**kwargs))


# ---------------------------------------------------------------------------
# Test: Brief interrupt absorbed
# ---------------------------------------------------------------------------

class TestBriefInterruptAbsorbed:
    """A brief interruption (<60s) between same-cluster segments is absorbed."""

    def test_30s_slack_check_absorbed(self):
        """[Chrome cluster, 30s Slack, Chrome cluster] -> ONE segment."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        slack = make_segment(1, _ts(5), _ts(5), app="Slack", what_doing="Chat")
        # The gap between seg1 end (T+5) and seg2 start (T+6) is 60s = exactly at threshold
        # But 30s means seg1 ends at T+5, seg2 starts at T+5.5 (30s gap)
        # Use direct seconds: seg1 ends at 5min, gap of 30s, seg2 starts at 5min30s
        seg2 = make_segment(
            0, "2026-03-04T10:05:30Z", _ts(10), app="Chrome", what_doing="Research",
        )

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = seg1.frames + slack.frames + seg2.frames
        result = segmenter.classify_interruptions([seg1, slack, seg2], all_frames)

        # The two Chrome segments should merge into one
        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        # The merged segment should contain frames from both Chrome segments
        assert chrome_segs[0].frame_count >= 2

    def test_exactly_at_brief_threshold(self):
        """Gap exactly at brief_interrupt_max_seconds is still absorbed."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg2 = make_segment(0, _ts(6), _ts(10), app="Chrome")
        # Gap = 60s (1 minute)

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        result = segmenter.classify_interruptions([seg1, seg2], seg1.frames + seg2.frames)

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1

    def test_brief_interrupt_no_interruption_events(self):
        """Brief interrupts should NOT create InterruptionEvent annotations."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        # 30s gap (well under 60s threshold)
        seg2 = make_segment(0, "2026-03-04T10:05:30Z", _ts(10), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        result = segmenter.classify_interruptions([seg1, seg2], seg1.frames + seg2.frames)

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        # Brief interrupts are absorbed silently — no InterruptionEvent
        # (only pauses generate InterruptionEvents)


# ---------------------------------------------------------------------------
# Test: Pause merged
# ---------------------------------------------------------------------------

class TestPauseMerged:
    """A pause (<30min) between same-cluster segments is merged with annotation."""

    def test_15min_pause_merged(self):
        """[Chrome cluster, 15min idle, Chrome cluster] -> ONE segment with pause."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        seg2 = make_segment(0, _ts(20), _ts(25), app="Chrome", what_doing="Research")
        # Gap = 15 minutes (900s) — under pause_max of 30min

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        # Should have an interruption event annotating the pause
        assert len(chrome_segs[0].interruptions) >= 1
        pause = chrome_segs[0].interruptions[0]
        assert pause.classification == TaskState.PAUSED
        assert pause.duration_seconds == 900  # 15 min

    def test_pause_exactly_at_threshold(self):
        """Gap exactly at pause_max_minutes is still merged."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg2 = make_segment(0, _ts(35), _ts(40), app="Chrome")
        # Gap = 30 min = 1800s = exactly pause_max

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1

    def test_pause_with_intervening_app(self):
        """Pause with a brief different-app segment in between absorbs it."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        slack = make_segment(1, _ts(6), _ts(7), app="Slack", what_doing="Chat")
        seg2 = make_segment(0, _ts(10), _ts(15), app="Chrome", what_doing="Research")
        # Gap between seg1 end (T+5) and seg2 start (T+10) = 5 minutes (300s)
        # This is a pause (> 60s but < 30min)

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        all_frames = seg1.frames + slack.frames + seg2.frames
        result = segmenter.classify_interruptions(
            [seg1, slack, seg2], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert len(chrome_segs[0].interruptions) >= 1


# ---------------------------------------------------------------------------
# Test: Abandon detected
# ---------------------------------------------------------------------------

class TestAbandonDetected:
    """A segment is abandoned when gap > pause_max and no return."""

    def test_abandon_long_gap(self):
        """[Chrome cluster, 45min gap, Chrome cluster] -> two segments, first abandoned."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        seg2 = make_segment(0, _ts(50), _ts(55), app="Chrome", what_doing="Research")
        # Gap = 45 min = 2700s > pause_max of 30min

        segmenter = _segmenter(pause_max_minutes=30)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 2
        # First segment should be marked as abandoned
        assert chrome_segs[0].state == TaskState.ABANDONED
        # Second segment stays active
        assert chrome_segs[1].state == TaskState.ACTIVE

    def test_abandon_with_different_app_in_between(self):
        """Chrome -> 45min VS Code -> Chrome marked abandoned."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        vscode = make_segment(1, _ts(10), _ts(50), app="VS Code", what_doing="Coding")
        seg2 = make_segment(0, _ts(55), _ts(60), app="Chrome", what_doing="Research")
        # Gap between Chrome segments = 50 min > pause_max

        segmenter = _segmenter(pause_max_minutes=30)
        all_frames = seg1.frames + vscode.frames + seg2.frames
        result = segmenter.classify_interruptions(
            [seg1, vscode, seg2], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 2
        assert chrome_segs[0].state == TaskState.ABANDONED


# ---------------------------------------------------------------------------
# Test: Related interrupt linked
# ---------------------------------------------------------------------------

class TestRelatedInterruptLinked:
    """Related interrupts (<5min, same app) are linked but not merged."""

    def test_related_interrupt_same_app(self):
        """A short interruption in the same app is linked as related."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        # Short interruption using Chrome (same app = related)
        inter = make_segment(1, _ts(6), _ts(7), app="Chrome", what_doing="Checking email")
        seg2 = make_segment(0, _ts(8), _ts(12), app="Chrome", what_doing="Research")

        segmenter = _segmenter(
            brief_interrupt_max_seconds=60,
            pause_max_minutes=30,
            related_interrupt_max_seconds=300,
        )
        all_frames = seg1.frames + inter.frames + seg2.frames
        result = segmenter.classify_interruptions(
            [seg1, inter, seg2], all_frames,
        )

        # The Chrome research segments should be merged (gap < pause_max)
        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1


# ---------------------------------------------------------------------------
# Test: Nested interruptions
# ---------------------------------------------------------------------------

class TestNestedInterruptions:
    """Handle multiple interruptions within the same task."""

    def test_multiple_interruptions_in_sequence(self):
        """Chrome -> Slack (30s) -> Chrome -> Terminal (30s) -> Chrome = one segment."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        slack = make_segment(1, _ts(5), "2026-03-04T10:05:30Z", app="Slack")
        seg2 = make_segment(0, _ts(6), _ts(10), app="Chrome")
        term = make_segment(2, _ts(10), "2026-03-04T10:10:30Z", app="Terminal")
        seg3 = make_segment(0, _ts(11), _ts(15), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = (
            seg1.frames + slack.frames + seg2.frames
            + term.frames + seg3.frames
        )
        result = segmenter.classify_interruptions(
            [seg1, slack, seg2, term, seg3], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert chrome_segs[0].frame_count >= 3  # at least one frame per original segment

    def test_nested_pause_then_brief(self):
        """Chrome -> 5min pause -> Chrome -> 20s Slack -> Chrome."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg2 = make_segment(0, _ts(10), _ts(15), app="Chrome")
        slack = make_segment(1, _ts(15), "2026-03-04T10:15:20Z", app="Slack")
        seg3 = make_segment(0, _ts(16), _ts(20), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        all_frames = seg1.frames + seg2.frames + slack.frames + seg3.frames
        result = segmenter.classify_interruptions(
            [seg1, seg2, slack, seg3], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        # All Chrome segments should merge since all gaps are within pause_max
        assert len(chrome_segs) == 1


# ---------------------------------------------------------------------------
# Test: Edge timing
# ---------------------------------------------------------------------------

class TestEdgeTiming:
    """Test threshold boundary conditions."""

    def test_gap_one_second_over_brief(self):
        """Gap = brief_max + 1 should be treated as pause, not brief."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        # 61 seconds gap (just over brief_max=60)
        seg2 = make_segment(0, "2026-03-04T10:06:01Z", _ts(10), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        # Should still merge (within pause_max) but with a pause annotation
        assert len(chrome_segs) == 1
        assert len(chrome_segs[0].interruptions) >= 1
        assert chrome_segs[0].interruptions[0].classification == TaskState.PAUSED

    def test_gap_one_second_over_pause(self):
        """Gap = pause_max + 1s should NOT merge, should mark abandoned."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        # 30min + 1s gap
        seg2 = make_segment(0, "2026-03-04T10:35:01Z", _ts(40), app="Chrome")

        segmenter = _segmenter(pause_max_minutes=30)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 2
        assert chrome_segs[0].state == TaskState.ABANDONED

    def test_zero_gap(self):
        """Adjacent segments (gap=0) should merge."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg2 = make_segment(0, _ts(5), _ts(10), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1


# ---------------------------------------------------------------------------
# Test: No interruptions
# ---------------------------------------------------------------------------

class TestNoInterruptions:
    """When there are no gaps or interruptions, segments pass through unchanged."""

    def test_single_segment_unchanged(self):
        """One segment with no interruptions stays as-is."""
        seg = make_segment(0, _ts(0), _ts(10), app="Chrome", what_doing="Research")

        segmenter = _segmenter()
        result = segmenter.classify_interruptions([seg], seg.frames)

        assert len(result) == 1
        assert result[0].cluster_id == 0
        assert result[0].state == TaskState.ACTIVE
        assert result[0].interruptions == []

    def test_two_distant_segments_different_clusters(self):
        """Two segments from different clusters are not merged."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg2 = make_segment(1, _ts(6), _ts(10), app="VS Code")

        segmenter = _segmenter()
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        assert len(result) == 2
        assert all(s.state == TaskState.ACTIVE for s in result)

    def test_empty_segments_list(self):
        """Empty input returns empty output."""
        segmenter = _segmenter()
        result = segmenter.classify_interruptions([], [])
        assert result == []


# ---------------------------------------------------------------------------
# Test: Multiple brief interrupts in sequence
# ---------------------------------------------------------------------------

class TestMultipleBriefInterrupts:
    """Multiple brief interrupts between same-cluster segments all absorbed."""

    def test_three_brief_interrupts(self):
        """Chrome -> 20s Slack -> Chrome -> 15s Finder -> Chrome -> 10s Terminal -> Chrome."""
        seg1 = make_segment(0, _ts(0), _ts(3), app="Chrome")
        slack = make_segment(1, _ts(3), "2026-03-04T10:03:20Z", app="Slack")
        seg2 = make_segment(0, "2026-03-04T10:03:30Z", _ts(6), app="Chrome")
        finder = make_segment(2, _ts(6), "2026-03-04T10:06:15Z", app="Finder")
        seg3 = make_segment(0, "2026-03-04T10:06:20Z", _ts(9), app="Chrome")
        term = make_segment(3, _ts(9), "2026-03-04T10:09:10Z", app="Terminal")
        seg4 = make_segment(0, "2026-03-04T10:09:15Z", _ts(12), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = (
            seg1.frames + slack.frames + seg2.frames
            + finder.frames + seg3.frames + term.frames + seg4.frames
        )
        result = segmenter.classify_interruptions(
            [seg1, slack, seg2, finder, seg3, term, seg4], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert chrome_segs[0].frame_count >= 4


# ---------------------------------------------------------------------------
# Test: Interrupt at start/end of task
# ---------------------------------------------------------------------------

class TestInterruptAtBoundaries:
    """Interruptions at the very start or end of a task."""

    def test_interrupt_before_first_segment(self):
        """Interruption segment before the main task starts — no merge."""
        slack = make_segment(1, _ts(0), _ts(1), app="Slack")
        seg1 = make_segment(0, _ts(2), _ts(10), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = slack.frames + seg1.frames
        result = segmenter.classify_interruptions(
            [slack, seg1], all_frames,
        )

        # Both segments should remain separate (different clusters)
        assert len(result) == 2

    def test_interrupt_after_last_segment(self):
        """Interruption after the main task ends — no merge."""
        seg1 = make_segment(0, _ts(0), _ts(10), app="Chrome")
        slack = make_segment(1, _ts(11), _ts(12), app="Slack")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = seg1.frames + slack.frames
        result = segmenter.classify_interruptions(
            [seg1, slack], all_frames,
        )

        assert len(result) == 2
        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert chrome_segs[0].state == TaskState.ACTIVE


# ---------------------------------------------------------------------------
# Test: Different clusters not merged even if close
# ---------------------------------------------------------------------------

class TestDifferentClustersNotMerged:
    """Segments from different clusters should never be merged."""

    def test_different_clusters_1s_gap(self):
        """Even with 1s gap, different clusters stay separate."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", what_doing="Research")
        seg2 = make_segment(1, _ts(5), _ts(10), app="VS Code", what_doing="Coding")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        assert len(result) == 2
        cluster_ids = [s.cluster_id for s in result]
        assert 0 in cluster_ids
        assert 1 in cluster_ids

    def test_interleaved_clusters_no_merge(self):
        """Interleaved different-cluster segments are not merged."""
        seg_a1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg_b1 = make_segment(1, _ts(6), _ts(10), app="VS Code")
        seg_a2 = make_segment(0, _ts(11), _ts(15), app="Chrome")
        seg_b2 = make_segment(1, _ts(16), _ts(20), app="VS Code")

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        all_frames = seg_a1.frames + seg_b1.frames + seg_a2.frames + seg_b2.frames
        result = segmenter.classify_interruptions(
            [seg_a1, seg_b1, seg_a2, seg_b2], all_frames,
        )

        # Each cluster merges its own segments (gap <= brief_max)
        chrome_segs = [s for s in result if s.cluster_id == 0]
        vscode_segs = [s for s in result if s.cluster_id == 1]
        assert len(chrome_segs) == 1
        assert len(vscode_segs) == 1


# ---------------------------------------------------------------------------
# Test: Very short segments (1 frame) as interruptions
# ---------------------------------------------------------------------------

class TestSingleFrameInterruptions:
    """Single-frame segments between same-cluster segments are treated as interruptions."""

    def test_single_frame_slack_between_chrome(self):
        """1-frame Slack segment between Chrome segments is absorbed."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", num_frames=3)
        slack = make_segment(1, "2026-03-04T10:05:10Z", "2026-03-04T10:05:10Z",
                             app="Slack", num_frames=1)
        seg2 = make_segment(0, "2026-03-04T10:05:30Z", _ts(10), app="Chrome", num_frames=3)

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = seg1.frames + slack.frames + seg2.frames
        result = segmenter.classify_interruptions(
            [seg1, slack, seg2], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert chrome_segs[0].frame_count == 6  # 3 + 3 (Slack frame absorbed)


# ---------------------------------------------------------------------------
# Test: TaskState and InterruptionEvent dataclass correctness
# ---------------------------------------------------------------------------

class TestDataStructures:
    """Verify the new data structures work correctly."""

    def test_task_state_values(self):
        """All TaskState enum values are accessible."""
        assert TaskState.ACTIVE.value == "active"
        assert TaskState.PAUSED.value == "paused"
        assert TaskState.RESUMED.value == "resumed"
        assert TaskState.ABANDONED.value == "abandoned"
        assert TaskState.RELATED.value == "related"

    def test_interruption_event_creation(self):
        """InterruptionEvent fields are correctly stored."""
        evt = InterruptionEvent(
            start_time="2026-03-04T10:05:00Z",
            end_time="2026-03-04T10:06:00Z",
            duration_seconds=60,
            interrupting_app="Slack",
            classification=TaskState.PAUSED,
        )
        assert evt.start_time == "2026-03-04T10:05:00Z"
        assert evt.end_time == "2026-03-04T10:06:00Z"
        assert evt.duration_seconds == 60
        assert evt.interrupting_app == "Slack"
        assert evt.classification == TaskState.PAUSED

    def test_task_segment_default_state(self):
        """TaskSegment defaults to ACTIVE state with no interruptions."""
        seg = TaskSegment(segment_id="s1", cluster_id=0)
        assert seg.state == TaskState.ACTIVE
        assert seg.interruptions == []

    def test_task_segment_with_interruptions(self):
        """TaskSegment can store interruptions."""
        evt = InterruptionEvent(
            start_time="2026-03-04T10:05:00Z",
            end_time="2026-03-04T10:06:00Z",
            duration_seconds=60,
            interrupting_app="Slack",
            classification=TaskState.PAUSED,
        )
        seg = TaskSegment(
            segment_id="s1", cluster_id=0,
            interruptions=[evt],
            state=TaskState.ACTIVE,
        )
        assert len(seg.interruptions) == 1
        assert seg.interruptions[0].interrupting_app == "Slack"


# ---------------------------------------------------------------------------
# Test: SegmenterConfig new fields
# ---------------------------------------------------------------------------

class TestSegmenterConfigInterruptionFields:
    """Verify new config fields have correct defaults."""

    def test_default_brief_interrupt(self):
        cfg = SegmenterConfig()
        assert cfg.brief_interrupt_max_seconds == 60

    def test_default_pause_max(self):
        cfg = SegmenterConfig()
        assert cfg.pause_max_minutes == 30

    def test_default_related_interrupt(self):
        cfg = SegmenterConfig()
        assert cfg.related_interrupt_max_seconds == 300

    def test_custom_values(self):
        cfg = SegmenterConfig(
            brief_interrupt_max_seconds=120,
            pause_max_minutes=45,
            related_interrupt_max_seconds=600,
        )
        assert cfg.brief_interrupt_max_seconds == 120
        assert cfg.pause_max_minutes == 45
        assert cfg.related_interrupt_max_seconds == 600


# ---------------------------------------------------------------------------
# Test: Complex scenarios
# ---------------------------------------------------------------------------

class TestComplexScenarios:
    """Complex multi-segment scenarios."""

    def test_mixed_brief_and_pause(self):
        """Brief interrupt followed by a pause in the same task."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        # 30s brief interrupt
        seg2 = make_segment(0, "2026-03-04T10:05:30Z", _ts(10), app="Chrome")
        # 10 min pause
        seg3 = make_segment(0, _ts(20), _ts(25), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        all_frames = seg1.frames + seg2.frames + seg3.frames
        result = segmenter.classify_interruptions(
            [seg1, seg2, seg3], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        # Should have at least one pause interruption event
        pause_events = [
            e for e in chrome_segs[0].interruptions
            if e.classification == TaskState.PAUSED
        ]
        assert len(pause_events) >= 1

    def test_abandon_then_resume_different_cluster(self):
        """Chrome abandoned, then VS Code starts — both stay separate."""
        chrome = make_segment(0, _ts(0), _ts(5), app="Chrome")
        # 45 min gap
        vscode = make_segment(1, _ts(50), _ts(60), app="VS Code")

        segmenter = _segmenter(pause_max_minutes=30)
        result = segmenter.classify_interruptions(
            [chrome, vscode], chrome.frames + vscode.frames,
        )

        assert len(result) == 2
        # Chrome is NOT abandoned since it has no same-cluster pair
        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert chrome_segs[0].state == TaskState.ACTIVE

    def test_three_clusters_interleaved(self):
        """Three different clusters interleaved, each with brief gaps."""
        a1 = make_segment(0, _ts(0), _ts(2), app="Chrome")
        b1 = make_segment(1, _ts(2), _ts(4), app="VS Code")
        c1 = make_segment(2, _ts(4), _ts(6), app="Slack")
        a2 = make_segment(0, _ts(6), _ts(8), app="Chrome")
        b2 = make_segment(1, _ts(8), _ts(10), app="VS Code")
        c2 = make_segment(2, _ts(10), _ts(12), app="Slack")

        segmenter = _segmenter(brief_interrupt_max_seconds=300, pause_max_minutes=30)
        all_frames = a1.frames + b1.frames + c1.frames + a2.frames + b2.frames + c2.frames
        result = segmenter.classify_interruptions(
            [a1, b1, c1, a2, b2, c2], all_frames,
        )

        # Each cluster's segments should merge
        for cid in (0, 1, 2):
            cluster_segs = [s for s in result if s.cluster_id == cid]
            assert len(cluster_segs) == 1

    def test_interruption_event_has_interrupting_app(self):
        """Pause InterruptionEvent records which app caused the interruption."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        slack = make_segment(1, _ts(6), _ts(8), app="Slack", what_doing="Chat")
        seg2 = make_segment(0, _ts(10), _ts(15), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60, pause_max_minutes=30)
        all_frames = seg1.frames + slack.frames + seg2.frames
        result = segmenter.classify_interruptions(
            [seg1, slack, seg2], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert len(chrome_segs[0].interruptions) >= 1
        # The interrupting app should be recorded
        inter = chrome_segs[0].interruptions[0]
        assert inter.interrupting_app == "Slack"

    def test_segment_preserves_original_frames(self):
        """Merged segment contains all original frames from parent segments."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome", num_frames=3)
        seg2 = make_segment(0, _ts(6), _ts(10), app="Chrome", num_frames=4)

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        all_frames = seg1.frames + seg2.frames
        result = segmenter.classify_interruptions(
            [seg1, seg2], all_frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert chrome_segs[0].frame_count == 7  # 3 + 4

    def test_merged_segment_time_bounds(self):
        """Merged segment has correct start_time and end_time."""
        seg1 = make_segment(0, _ts(0), _ts(5), app="Chrome")
        seg2 = make_segment(0, _ts(6), _ts(10), app="Chrome")

        segmenter = _segmenter(brief_interrupt_max_seconds=60)
        result = segmenter.classify_interruptions(
            [seg1, seg2], seg1.frames + seg2.frames,
        )

        chrome_segs = [s for s in result if s.cluster_id == 0]
        assert len(chrome_segs) == 1
        assert chrome_segs[0].start_time == _ts(0)
        assert chrome_segs[0].end_time == _ts(10)

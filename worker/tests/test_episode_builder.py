"""Tests for agenthandover_worker.episode_builder.

Covers thread multiplexing, soft/hard cap splitting, segment linking,
and edge cases (empty input, interleaved apps).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from agenthandover_worker.episode_builder import Episode, EpisodeBuilder


def _ts(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the daemon produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_event(
    *,
    app_id: str = "com.example.App",
    url: str | None = None,
    timestamp: str | None = None,
    event_id: str | None = None,
    kind: str = "FocusChange",
) -> dict:
    """Build a minimal event dict that the EpisodeBuilder can consume."""
    eid = event_id or str(uuid.uuid4())
    window = {"app_id": app_id, "title": "Test Window"}
    metadata: dict = {}
    if url:
        metadata["url"] = url

    return {
        "id": eid,
        "timestamp": timestamp or _ts(datetime.now(timezone.utc)),
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps(window),
        "metadata_json": json.dumps(metadata),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "processed": 0,
    }


# ------------------------------------------------------------------
# 1. Single app → single episode
# ------------------------------------------------------------------


class TestSingleAppSingleEpisode:
    def test_single_app_single_episode(self) -> None:
        """All events from the same app within caps → one episode."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=i)))
            for i in range(5)
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.event_count == 5
        assert ep.thread_id == "com.apple.Safari"
        assert ep.segment_id == 0
        assert ep.prev_segment_id is None


# ------------------------------------------------------------------
# 2. Different apps → different episodes
# ------------------------------------------------------------------


class TestDifferentAppsDifferentEpisodes:
    def test_different_apps_different_episodes(self) -> None:
        """Events from two distinct apps → two episodes."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=0))),
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=1))),
            _make_event(app_id="com.microsoft.VSCode", timestamp=_ts(base + timedelta(seconds=2))),
            _make_event(app_id="com.microsoft.VSCode", timestamp=_ts(base + timedelta(seconds=3))),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 2
        thread_ids = {ep.thread_id for ep in episodes}
        assert "com.apple.Safari" in thread_ids
        assert "com.microsoft.VSCode" in thread_ids

        for ep in episodes:
            assert ep.event_count == 2


# ------------------------------------------------------------------
# 3. Soft cap splits episode
# ------------------------------------------------------------------


class TestSoftCapSplitsEpisode:
    def test_soft_cap_splits_episode(self) -> None:
        """Events spanning >15 min → 2 segments with linking."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = []

        # 10 events in first 14 minutes
        for i in range(10):
            events.append(
                _make_event(
                    app_id="com.apple.Safari",
                    timestamp=_ts(base + timedelta(minutes=i * 1.4)),
                )
            )

        # 5 events after 15 minutes
        for i in range(5):
            events.append(
                _make_event(
                    app_id="com.apple.Safari",
                    timestamp=_ts(base + timedelta(minutes=15 + i)),
                )
            )

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        # Should be split into at least 2 segments
        safari_episodes = [ep for ep in episodes if ep.thread_id == "com.apple.Safari"]
        assert len(safari_episodes) >= 2

        # Total events preserved
        total_events = sum(ep.event_count for ep in safari_episodes)
        assert total_events == 15

        # All share the same episode_id
        episode_ids = {ep.episode_id for ep in safari_episodes}
        assert len(episode_ids) == 1

        # Segments are numbered correctly
        segments = sorted(safari_episodes, key=lambda e: e.segment_id)
        assert segments[0].segment_id == 0
        assert segments[1].segment_id == 1
        assert segments[1].prev_segment_id == 0


# ------------------------------------------------------------------
# 4. Hard cap splits episode
# ------------------------------------------------------------------


class TestHardCapSplitsEpisode:
    def test_hard_cap_splits_episode(self) -> None:
        """250 events → 2 segments (200 + 50)."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Notes",
                timestamp=_ts(base + timedelta(seconds=i)),
            )
            for i in range(250)
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        notes_episodes = [ep for ep in episodes if ep.thread_id == "com.apple.Notes"]
        assert len(notes_episodes) == 2

        segments = sorted(notes_episodes, key=lambda e: e.segment_id)
        assert segments[0].event_count == 200
        assert segments[1].event_count == 50

        # Same episode_id
        assert segments[0].episode_id == segments[1].episode_id


# ------------------------------------------------------------------
# 5. Empty events
# ------------------------------------------------------------------


class TestEmptyEvents:
    def test_empty_events(self) -> None:
        """Empty list → empty result."""
        builder = EpisodeBuilder()
        assert builder.process_events([]) == []


# ------------------------------------------------------------------
# 6. Segment linking
# ------------------------------------------------------------------


class TestSegmentLinking:
    def test_segment_linking(self) -> None:
        """Verify prev_segment_id chain is correct across 3+ segments."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = []

        # Create events spanning 3 segments via soft cap (>30 min total)
        for i in range(45):
            events.append(
                _make_event(
                    app_id="com.app.Editor",
                    timestamp=_ts(base + timedelta(minutes=i)),
                )
            )

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        editor_episodes = [ep for ep in episodes if ep.thread_id == "com.app.Editor"]
        assert len(editor_episodes) >= 3

        segments = sorted(editor_episodes, key=lambda e: e.segment_id)

        # First segment has no predecessor
        assert segments[0].prev_segment_id is None
        assert segments[0].segment_id == 0

        # Each subsequent segment links to the previous one
        for i in range(1, len(segments)):
            assert segments[i].prev_segment_id == segments[i - 1].segment_id
            assert segments[i].segment_id == i

        # All share the same episode_id
        assert len({s.episode_id for s in segments}) == 1


# ------------------------------------------------------------------
# 7. Thread multiplexing (interleaved apps)
# ------------------------------------------------------------------


class TestThreadMultiplexing:
    def test_thread_multiplexing(self) -> None:
        """Interleaved app events → separate threads (episodes)."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=0))),
            _make_event(app_id="com.microsoft.VSCode", timestamp=_ts(base + timedelta(seconds=1))),
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=2))),
            _make_event(app_id="com.microsoft.VSCode", timestamp=_ts(base + timedelta(seconds=3))),
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=4))),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 2

        safari = [ep for ep in episodes if ep.thread_id == "com.apple.Safari"]
        vscode = [ep for ep in episodes if ep.thread_id == "com.microsoft.VSCode"]

        assert len(safari) == 1
        assert safari[0].event_count == 3

        assert len(vscode) == 1
        assert vscode[0].event_count == 2

    def test_url_domain_differentiates_threads(self) -> None:
        """Same app but different URL domains → separate threads."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                url="https://github.com/repo",
                timestamp=_ts(base + timedelta(seconds=0)),
            ),
            _make_event(
                app_id="com.apple.Safari",
                url="https://stackoverflow.com/q/123",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
            _make_event(
                app_id="com.apple.Safari",
                url="https://github.com/other",
                timestamp=_ts(base + timedelta(seconds=2)),
            ),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 2
        thread_ids = {ep.thread_id for ep in episodes}
        assert "com.apple.Safari:github.com" in thread_ids
        assert "com.apple.Safari:stackoverflow.com" in thread_ids

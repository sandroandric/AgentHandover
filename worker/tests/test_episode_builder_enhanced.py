"""Tests for enhanced EpisodeBuilder features.

Covers entity-based thread clustering (ticket IDs, filenames),
clipboard linker integration, and continuation_of metadata.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from agenthandover_worker.episode_builder import Episode, EpisodeBuilder


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_event(
    *,
    app_id: str = "com.example.App",
    url: str | None = None,
    timestamp: str | None = None,
    event_id: str | None = None,
    kind: str = "FocusChange",
    title: str = "Test Window",
    metadata_extra: dict | None = None,
) -> dict:
    eid = event_id or str(uuid.uuid4())
    window = {"app_id": app_id, "title": title}
    metadata: dict = {}
    if url:
        metadata["url"] = url
    if metadata_extra:
        metadata.update(metadata_extra)

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
# 1. Ticket ID extraction (JIRA-123 pattern)
# ------------------------------------------------------------------


class TestTicketIDExtraction:
    def test_jira_ticket_clusters_events(self) -> None:
        """Events with JIRA-123 in title go to same thread."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                title="PROJ-123 - Pull Request",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Safari",
                title="PROJ-123 - Review",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
            _make_event(
                app_id="com.apple.Safari",
                title="PROJ-456 - Other PR",
                timestamp=_ts(base + timedelta(seconds=2)),
            ),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 2
        thread_ids = {ep.thread_id for ep in episodes}
        assert any("PROJ-123" in tid for tid in thread_ids)
        assert any("PROJ-456" in tid for tid in thread_ids)

    def test_ticket_in_url_clusters(self) -> None:
        """Ticket ID in URL also clusters events."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                url="https://jira.example.com/browse/FEAT-789",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Safari",
                url="https://jira.example.com/browse/FEAT-789",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 1
        assert "FEAT-789" in episodes[0].thread_id


# ------------------------------------------------------------------
# 2. Issue number extraction (#456 pattern)
# ------------------------------------------------------------------


class TestIssueNumberExtraction:
    def test_github_issue_clusters(self) -> None:
        """Events with #123 in URL cluster together."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                url="https://github.com/repo/issues/123",
                title="Issue #123 - Bug",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Safari",
                url="https://github.com/repo/issues/123",
                title="Issue #123 - Discussion",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 1
        # Thread should contain the issue number
        assert "#123" in episodes[0].thread_id or "123" in episodes[0].thread_id


# ------------------------------------------------------------------
# 3. Filename extraction from window title
# ------------------------------------------------------------------


class TestFilenameExtraction:
    def test_filename_in_title_clusters(self) -> None:
        """Events with filename in title like 'report.pdf - Preview' cluster."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Preview",
                title="report.pdf - Preview",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Preview",
                title="report.pdf - Preview",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
            _make_event(
                app_id="com.apple.Preview",
                title="slides.pptx - Preview",
                timestamp=_ts(base + timedelta(seconds=2)),
            ),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 2
        thread_ids = {ep.thread_id for ep in episodes}
        assert any("report.pdf" in tid for tid in thread_ids)
        assert any("slides.pptx" in tid for tid in thread_ids)


# ------------------------------------------------------------------
# 4. No entity falls back to app_id:domain
# ------------------------------------------------------------------


class TestNoEntityFallback:
    def test_no_entity_falls_back(self) -> None:
        """Events without entities cluster by app_id:domain as before."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                url="https://example.com/page",
                title="Example Site",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Safari",
                url="https://example.com/other",
                title="Another Page",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        assert len(episodes) == 1
        assert episodes[0].thread_id == "com.apple.Safari:example.com"


# ------------------------------------------------------------------
# 5. Continuation_of metadata on split
# ------------------------------------------------------------------


class TestContinuationOfMetadata:
    def test_split_adds_continuation_of(self) -> None:
        """When splitting, metadata['continuation_of'] is set."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.app.Editor",
                timestamp=_ts(base + timedelta(minutes=i)),
            )
            for i in range(20)
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        editor_eps = [ep for ep in episodes if "com.app.Editor" in ep.thread_id]
        if len(editor_eps) >= 2:
            segments = sorted(editor_eps, key=lambda e: e.segment_id)
            # First segment has no continuation_of
            assert "continuation_of" not in segments[0].metadata
            # Second segment has continuation_of pointing to first
            assert "continuation_of" in segments[1].metadata
            expected_id = f"{segments[0].episode_id}:seg{segments[0].segment_id}"
            assert segments[1].metadata["continuation_of"] == expected_id

    def test_hard_cap_split_adds_continuation_of(self) -> None:
        """Hard cap (200 events) split also adds continuation_of."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Notes",
                timestamp=_ts(base + timedelta(seconds=i)),
            )
            for i in range(210)
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        notes_eps = [ep for ep in episodes if "com.apple.Notes" in ep.thread_id]
        assert len(notes_eps) == 2

        segments = sorted(notes_eps, key=lambda e: e.segment_id)
        assert segments[1].metadata.get("continuation_of") is not None


# ------------------------------------------------------------------
# 6. Clipboard linker integration
# ------------------------------------------------------------------


class TestClipboardLinkerIntegration:
    def test_clipboard_links_annotated_on_episodes(self) -> None:
        """Copy-paste pairs are annotated as clipboard_links on episodes."""
        import hashlib
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        content_hash = hashlib.sha256(b"test content").hexdigest()

        # Clipboard events use the real daemon format: kind_json carries
        # {"type": "<kind>", "content_hash": ..., "byte_size": ...}.
        # metadata_json holds unrelated fields and is {} for clipboard
        # events in production.  (Historical bug caught 2026-04-10: this
        # fixture previously wrote to metadata_json, which matched the
        # broken production code, so both drifted from reality together.)
        events = [
            {
                "id": "copy-1",
                "timestamp": _ts(base),
                "kind_json": json.dumps({
                    "type": "ClipboardChange",
                    "content_hash": content_hash,
                    "content_types": ["public.utf8-plain-text"],
                    "byte_size": 12,
                    "high_entropy": False,
                }),
                "window_json": json.dumps({"app_id": "com.apple.Terminal", "title": "Terminal"}),
                "metadata_json": "{}",
                "display_topology_json": "[]",
                "primary_display_id": "main",
                "processed": 0,
            },
            {
                "id": "paste-1",
                "timestamp": _ts(base + timedelta(minutes=2)),
                "kind_json": json.dumps({
                    "type": "PasteDetected",
                    "content_hash": content_hash,
                    "target_app": "com.apple.TextEdit",
                    "byte_size": 12,
                }),
                "window_json": json.dumps({"app_id": "com.apple.Terminal", "title": "Terminal"}),
                "metadata_json": "{}",
                "display_topology_json": "[]",
                "primary_display_id": "main",
                "processed": 0,
            },
        ]

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        # Find the episode containing the paste
        found_link = False
        for ep in episodes:
            if ep.clipboard_links:
                found_link = True
                link = ep.clipboard_links[0]
                assert link.copy_event_id == "copy-1"
                assert link.paste_event_id == "paste-1"
                assert link.content_hash == content_hash

        assert found_link


# ------------------------------------------------------------------
# 7. Episode metadata field exists
# ------------------------------------------------------------------


class TestEpisodeMetadata:
    def test_episode_has_metadata_field(self) -> None:
        ep = Episode(episode_id="test-123")
        assert isinstance(ep.metadata, dict)
        assert ep.metadata == {}

    def test_episode_has_clipboard_links_field(self) -> None:
        ep = Episode(episode_id="test-123")
        assert isinstance(ep.clipboard_links, list)
        assert ep.clipboard_links == []

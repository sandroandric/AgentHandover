"""Tests for evidence_extractor.py — relevance filtering and extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenthandover_worker.evidence_extractor import EvidenceExtractor


def _make_event(
    event_id: str,
    app: str = "Chrome",
    location: str = "",
    what_doing: str = "",
    timestamp: str = "2026-03-10T10:00:00Z",
    event_type: str = "DwellSnapshot",
    diff_inputs: list | None = None,
) -> dict:
    """Create a synthetic event dict."""
    annotation = {
        "app": app,
        "location": location,
        "task_context": {"what_doing": what_doing},
    }
    ev = {
        "event_id": event_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "scene_annotation_json": json.dumps(annotation),
        "annotation_status": "completed",
    }
    if diff_inputs:
        ev["frame_diff_json"] = json.dumps({
            "diff_type": "action",
            "inputs": diff_inputs,
        })
    return ev


def _make_procedure(
    slug: str = "test-proc",
    apps: list[str] | None = None,
    steps: list[dict] | None = None,
    title: str = "Test Workflow",
    observations: list[dict] | None = None,
) -> dict:
    """Create a synthetic procedure dict."""
    return {
        "id": slug,
        "title": title,
        "description": "",
        "apps_involved": apps or ["Chrome"],
        "steps": steps or [
            {"action": "Browse Reddit startups", "location": "https://reddit.com/r/startups"},
            {"action": "Write marketing reply", "location": "https://reddit.com/r/startups"},
        ],
        "evidence": {
            "observations": observations or [{"date": "2026-03-10"}],
        },
        "lifecycle_state": "draft",
    }


def _make_kb_and_db(tmp_path, procedure, events):
    """Create mock KB and DB for testing."""
    kb = MagicMock()
    kb.get_procedure.return_value = procedure
    kb.save_procedure.return_value = None

    db = MagicMock()
    db.get_events_for_procedure_window.return_value = events

    return kb, db


# ---------------------------------------------------------------------------
# Relevance filtering
# ---------------------------------------------------------------------------


class TestRelevanceFiltering:
    """Tests for the app/domain/keyword filtering logic."""

    def test_filters_out_unrelated_app_events(self):
        """Events from apps not in the procedure's apps_involved are excluded."""
        proc = _make_procedure(apps=["Chrome"], title="Reddit Marketing")
        events = [
            _make_event("e1", app="Chrome", location="https://reddit.com/r/startups",
                        what_doing="browsing reddit marketing posts"),
            _make_event("e2", app="Slack", location="",
                        what_doing="chatting with team about lunch"),
            _make_event("e3", app="Calendar", location="",
                        what_doing="checking meetings schedule"),
        ]
        kb, db = _make_kb_and_db(None, proc, events)
        extractor = EvidenceExtractor(kb=kb, db=db)
        result = extractor.extract_for_procedure("test-proc")

        # Only the Chrome/Reddit event should be included
        assert result["event_count"] == 1

    def test_filters_same_app_without_keyword_overlap(self):
        """Chrome events without keyword overlap with procedure steps are excluded."""
        proc = _make_procedure(
            apps=["Chrome"],
            steps=[{"action": "Browse Reddit startups", "location": "https://reddit.com/r/startups"}],
            title="Reddit Marketing Workflow",
        )
        events = [
            _make_event("e1", app="Chrome", location="https://reddit.com/r/startups",
                        what_doing="browsing reddit marketing posts"),
            _make_event("e2", app="Chrome", location="https://news.ycombinator.com",
                        what_doing="reading technology news articles"),
            _make_event("e3", app="Chrome", location="https://youtube.com/watch",
                        what_doing="watching funny cat videos entertainment"),
        ]
        kb, db = _make_kb_and_db(None, proc, events)
        extractor = EvidenceExtractor(kb=kb, db=db)
        result = extractor.extract_for_procedure("test-proc")

        # Only e1 should match (Chrome + "reddit"/"marketing" keywords)
        # e2 and e3 are Chrome but no keyword overlap with procedure
        assert result["event_count"] == 1

    def test_same_domain_without_keyword_excluded(self):
        """Same domain (reddit.com) but different subreddit/activity excluded."""
        proc = _make_procedure(
            apps=["Chrome"],
            steps=[{"action": "Browse startup posts", "location": "https://reddit.com/r/startups"}],
            title="Reddit Startup Marketing",
        )
        events = [
            _make_event("e1", app="Chrome", location="https://reddit.com/r/startups",
                        what_doing="browsing startup marketing posts"),
            _make_event("e2", app="Chrome", location="https://reddit.com/r/funny",
                        what_doing="scrolling through funny memes entertainment"),
        ]
        kb, db = _make_kb_and_db(None, proc, events)
        extractor = EvidenceExtractor(kb=kb, db=db)
        result = extractor.extract_for_procedure("test-proc")

        # e1: domain reddit.com + keyword "startup"/"marketing" → included
        # e2: domain reddit.com but keywords "funny"/"memes"/"entertainment" → no overlap → excluded
        assert result["event_count"] == 1

    def test_no_filter_data_includes_all(self):
        """Procedure with no apps/steps/title includes all events."""
        proc = {
            "id": "test-proc", "title": "", "description": "",
            "apps_involved": [], "steps": [],
            "evidence": {"observations": [{"date": "2026-03-10"}]},
            "lifecycle_state": "draft",
        }
        events = [
            _make_event("e1", app="Chrome", what_doing="anything"),
            _make_event("e2", app="Slack", what_doing="chatting"),
        ]
        kb = MagicMock()
        kb.get_procedure.return_value = proc
        kb.save_procedure.return_value = None
        db = MagicMock()
        db.get_events_for_procedure_window.return_value = events

        extractor = EvidenceExtractor(kb=kb, db=db)
        result = extractor.extract_for_procedure("test-proc")

        assert result.get("event_count") == 2

    def test_deduplicates_events_across_windows(self):
        """Same event appearing in overlapping observation windows is counted once."""
        proc = _make_procedure(
            apps=["Chrome"],
            title="Reddit Marketing",
            observations=[
                {"date": "2026-03-10"},
                {"date": "2026-03-10"},  # Same day, two observations
            ],
        )
        events = [
            _make_event("e1", app="Chrome", what_doing="reddit marketing browsing"),
        ]
        kb, db = _make_kb_and_db(None, proc, events)
        extractor = EvidenceExtractor(kb=kb, db=db)
        result = extractor.extract_for_procedure("test-proc")

        # Even though two observations query the same day, event is counted once
        assert result["event_count"] == 1


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


class TestContentExtraction:

    def test_extracts_clipboard_events(self):
        events = [
            {
                "event_id": "c1",
                "event_type": "ClipboardChange",
                "timestamp": "2026-03-10T10:00:00Z",
                "metadata_json": json.dumps({
                    "byte_size": 256,
                    "content_types": ["text/plain"],
                }),
                "scene_annotation_json": None,
            },
        ]
        extractor = EvidenceExtractor(kb=MagicMock(), db=MagicMock())
        content = extractor.extract_content_produced(events)
        assert len(content) == 1
        assert content[0]["type"] == "clipboard"
        assert content[0]["byte_size"] == 256

    def test_extracts_text_inputs(self):
        events = [
            {
                "event_id": "t1",
                "event_type": "DwellSnapshot",
                "timestamp": "2026-03-10T10:00:00Z",
                "frame_diff_json": json.dumps({
                    "diff_type": "action",
                    "inputs": [{"field": "reply", "value": "Great marketing insight here"}],
                }),
                "scene_annotation_json": None,
                "metadata_json": None,
            },
        ]
        extractor = EvidenceExtractor(kb=MagicMock(), db=MagicMock())
        content = extractor.extract_content_produced(events)
        assert len(content) == 1
        assert content[0]["type"] == "text_input"
        assert "marketing" in content[0]["value_preview"]


# ---------------------------------------------------------------------------
# Selection signals
# ---------------------------------------------------------------------------


class TestSelectionSignals:

    def test_classifies_engagement_levels(self):
        # Dwell time = time spent at a location before navigating away.
        # post1: 45s (high), post2: 3s (low), post3: 15s (medium)
        events = [
            _make_event("e1", location="https://reddit.com/post1",
                        timestamp="2026-03-10T10:00:00Z"),
            _make_event("e2", location="https://reddit.com/post2",
                        timestamp="2026-03-10T10:00:45Z"),  # post1 dwell = 45s
            _make_event("e3", location="https://reddit.com/post3",
                        timestamp="2026-03-10T10:00:48Z"),  # post2 dwell = 3s
            _make_event("e4", location="https://reddit.com/feed",
                        timestamp="2026-03-10T10:01:03Z"),  # post3 dwell = 15s
        ]
        extractor = EvidenceExtractor(kb=MagicMock(), db=MagicMock())
        signals = extractor.extract_selection_signals(events)

        # Should have dwell signals for post1, post2, post3
        assert len(signals) >= 2
        by_loc = {s["location"]: s for s in signals}
        assert by_loc["https://reddit.com/post1"]["engagement"] == "high"
        assert by_loc["https://reddit.com/post2"]["engagement"] == "low"

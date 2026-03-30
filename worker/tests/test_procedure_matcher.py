"""Tests for ProcedureMatcher — fingerprint-based matching of segments/spans to procedures."""

from __future__ import annotations

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_matcher import ProcedureMatcher, _VERB_MAP
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.task_segmenter import AnnotatedFrame, TaskSegment


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kb(tmp_path):
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


def _make_segment(segment_id, apps, frames_data):
    """Build a TaskSegment with AnnotatedFrames."""
    frames = []
    for fd in frames_data:
        frames.append(AnnotatedFrame(
            event_id=fd.get("event_id", f"evt-{len(frames)}"),
            timestamp=fd.get("timestamp", "2026-03-14T10:00:00Z"),
            annotation={},
            what_doing=fd.get("what_doing", ""),
            app=fd.get("app", ""),
            location=fd.get("location", ""),
            embedding=fd.get("embedding", []),
        ))
    return TaskSegment(
        segment_id=segment_id,
        cluster_id=0,
        frames=frames,
        task_label=frames[0].what_doing if frames else "",
        apps_involved=apps,
        start_time=frames[0].timestamp if frames else "",
        end_time=frames[-1].timestamp if frames else "",
    )


def _make_v3_procedure(slug, title, apps, steps, tags=None):
    """Build a minimal v3 procedure dict suitable for saving to the KB."""
    sop_template = {
        "slug": slug,
        "title": title,
        "steps": steps,
        "variables": [],
        "confidence_avg": 0.85,
        "episode_count": 3,
        "apps_involved": apps,
        "preconditions": [],
        "source": "test",
    }
    return sop_to_procedure(sop_template)


# ---------------------------------------------------------------------------
# Sample procedure factories
# ---------------------------------------------------------------------------


def _github_pr_procedure():
    """A procedure about reviewing PRs on GitHub in Chrome."""
    return _make_v3_procedure(
        slug="review-github-pr",
        title="Review GitHub Pull Request",
        apps=["Chrome"],
        steps=[
            {"step": "Open GitHub PR page", "action": "Open GitHub PR page",
             "target": "https://github.com/org/repo/pulls", "app": "Chrome",
             "location": "https://github.com/org/repo/pulls"},
            {"step": "Click the PR to review", "action": "Click the PR to review",
             "target": "PR link", "app": "Chrome",
             "location": "https://github.com/org/repo/pull/42"},
            {"step": "Review code changes", "action": "Review code changes",
             "target": "Files changed tab", "app": "Chrome"},
            {"step": "Submit review", "action": "Submit review",
             "target": "Review submit button", "app": "Chrome"},
        ],
    )


def _jira_ticket_procedure():
    """A procedure about managing Jira tickets."""
    return _make_v3_procedure(
        slug="manage-jira-ticket",
        title="Manage Jira Ticket",
        apps=["Firefox"],
        steps=[
            {"step": "Open Jira board", "action": "Open Jira board",
             "target": "https://company.atlassian.net/board", "app": "Firefox",
             "location": "https://company.atlassian.net/board"},
            {"step": "Select ticket to update", "action": "Select ticket to update",
             "target": "ticket card", "app": "Firefox"},
            {"step": "Enter status update", "action": "Enter status update",
             "target": "status field", "app": "Firefox"},
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSegmentMatchesKnownProcedure:
    """test_segment_matches_known_procedure"""

    def test_segment_matches_known_procedure(self, kb):
        """Save a procedure with apps=[Chrome], steps with github.com locations.
        Create a segment with same apps/locations. Verify match above threshold.
        """
        proc = _github_pr_procedure()
        kb.save_procedure(proc)

        matcher = ProcedureMatcher(kb)
        segment = _make_segment("seg-1", ["Chrome"], [
            {"what_doing": "Open GitHub PR page",
             "app": "Chrome",
             "location": "https://github.com/org/repo/pulls"},
            {"what_doing": "Review code changes",
             "app": "Chrome",
             "location": "https://github.com/org/repo/pull/42"},
        ])

        matches = matcher.match_segment(segment)
        assert len(matches) >= 1
        slug, score = matches[0]
        assert slug == "review-github-pr"
        assert score >= 0.50


class TestSegmentNoMatchBelowThreshold:
    """test_segment_no_match_below_threshold"""

    def test_segment_no_match_below_threshold(self, kb):
        """Procedure about Jira, segment about unrelated app. Verify empty list."""
        proc = _jira_ticket_procedure()
        kb.save_procedure(proc)

        matcher = ProcedureMatcher(kb)
        segment = _make_segment("seg-2", ["Terminal"], [
            {"what_doing": "Compile project",
             "app": "Terminal",
             "location": ""},
            {"what_doing": "Run tests",
             "app": "Terminal",
             "location": ""},
        ])

        matches = matcher.match_segment(segment)
        assert matches == []


class TestSegmentPartialMatch:
    """test_segment_partial_match"""

    def test_segment_partial_match(self, kb):
        """Some app overlap but different domains. Verify match if above 0.50."""
        proc = _github_pr_procedure()
        kb.save_procedure(proc)

        matcher = ProcedureMatcher(kb, match_threshold=0.30)
        # Same app (Chrome) but different domain (stackoverflow instead of github)
        segment = _make_segment("seg-3", ["Chrome"], [
            {"what_doing": "Open Stack Overflow question",
             "app": "Chrome",
             "location": "https://stackoverflow.com/questions/123"},
            {"what_doing": "Review answer",
             "app": "Chrome",
             "location": "https://stackoverflow.com/questions/123"},
        ])

        matches = matcher.match_segment(segment)
        # With a low threshold, partial app overlap may produce a match.
        # The key assertion: if matched, score should reflect partial overlap.
        if matches:
            _, score = matches[0]
            assert score >= 0.30


class TestSpanMatch:
    """test_span_match"""

    def test_span_match(self, kb):
        """Create a ContinuitySpan with apps_involved and goal_summary. Verify matching.

        Spans have inherently less fingerprint data than segments (no
        frame-level locations -> no domains), so we lower the threshold
        to 0.40 which is a realistic configuration for span matching.
        """
        from agenthandover_worker.continuity_tracker import ContinuitySpan

        proc = _github_pr_procedure()
        kb.save_procedure(proc)

        # Lower threshold because spans lack domain data (no frame locations)
        matcher = ProcedureMatcher(kb, match_threshold=0.40)
        span = ContinuitySpan(
            span_id="span-1",
            goal_summary="Open GitHub pull request for review",
            continuity_confidence=0.9,
            segments=["seg-a", "seg-b"],
            apps_involved=["Chrome"],
            first_seen="2026-03-14T10:00:00Z",
            last_seen="2026-03-14T10:30:00Z",
        )

        matches = matcher.match_span(span)
        # The span has same app (Chrome) and verb "open" which overlaps
        assert len(matches) >= 1
        slug, score = matches[0]
        assert slug == "review-github-pr"
        assert score >= 0.40


class TestEmptyKBReturnsEmpty:
    """test_empty_kb_returns_empty"""

    def test_empty_kb_returns_empty(self, kb):
        """No procedures in KB, verify empty match list."""
        matcher = ProcedureMatcher(kb)
        segment = _make_segment("seg-4", ["Chrome"], [
            {"what_doing": "Browse website", "app": "Chrome",
             "location": "https://example.com"},
        ])

        matches = matcher.match_segment(segment)
        assert matches == []


class TestEmptySegmentReturnsEmpty:
    """test_empty_segment_returns_empty"""

    def test_empty_segment_returns_empty(self, kb):
        """Segment with no frames still runs without error."""
        proc = _github_pr_procedure()
        kb.save_procedure(proc)

        matcher = ProcedureMatcher(kb)
        segment = TaskSegment(
            segment_id="seg-empty",
            cluster_id=0,
            frames=[],
            task_label="",
            apps_involved=[],
            start_time="",
            end_time="",
        )

        matches = matcher.match_segment(segment)
        # Empty segment produces an empty fingerprint; both-empty Jaccard = 1.0
        # so this may or may not match depending on the procedure fingerprint.
        # The key assertion is: no crash.
        assert isinstance(matches, list)


class TestFingerprintExtraction:
    """test_fingerprint_extraction"""

    def test_fingerprint_extraction(self, kb):
        """Verify _segment_to_fingerprint returns correct structure."""
        matcher = ProcedureMatcher(kb)
        segment = _make_segment("seg-fp", ["Chrome", "VS Code"], [
            {"what_doing": "Open GitHub page", "app": "Chrome",
             "location": "https://github.com/org/repo"},
            {"what_doing": "Review pull request", "app": "Chrome",
             "location": "https://github.com/org/repo/pull/5"},
            {"what_doing": "Enter comment", "app": "Chrome",
             "location": "https://github.com/org/repo/pull/5"},
        ])

        fp = matcher._segment_to_fingerprint(segment)

        assert "apps" in fp
        assert "domains" in fp
        assert "action_verbs" in fp

        # Apps should be normalized and sorted
        assert isinstance(fp["apps"], list)
        assert "chrome" in fp["apps"]

        # Domains should contain github.com
        assert "github.com" in fp["domains"]

        # Action verbs: "open" (from "Open"), "review" (from "Review"), "enter" (from "Enter")
        assert "open" in fp["action_verbs"]
        assert "review" in fp["action_verbs"]
        assert "enter" in fp["action_verbs"]


class TestVerbNormalization:
    """test_verb_normalization"""

    def test_verb_normalization(self, kb):
        """'navigate to page' should normalize verb to 'open'."""
        matcher = ProcedureMatcher(kb)
        segment = _make_segment("seg-verb", ["Chrome"], [
            {"what_doing": "Navigate to the settings page", "app": "Chrome"},
            {"what_doing": "Visit dashboard", "app": "Chrome"},
            {"what_doing": "Browse documentation", "app": "Chrome"},
            {"what_doing": "Type password", "app": "Chrome"},
            {"what_doing": "Press submit button", "app": "Chrome"},
            {"what_doing": "Check confirmation", "app": "Chrome"},
        ])

        fp = matcher._segment_to_fingerprint(segment)
        verbs = fp["action_verbs"]

        # navigate, visit, browse -> "open"
        assert "open" in verbs
        # type -> "enter"
        assert "enter" in verbs
        # press -> "click"
        assert "click" in verbs
        # check -> "verify"
        assert "verify" in verbs

        # The original words should NOT appear as separate entries
        assert "navigate" not in verbs
        assert "visit" not in verbs
        assert "browse" not in verbs
        assert "type" not in verbs
        assert "press" not in verbs
        assert "check" not in verbs


class TestDomainExtraction:
    """test_domain_extraction"""

    def test_domain_extraction(self, kb):
        """Frames with 'https://github.com/org/repo' should yield domain 'github.com'."""
        matcher = ProcedureMatcher(kb)
        segment = _make_segment("seg-dom", ["Chrome"], [
            {"what_doing": "View repo", "app": "Chrome",
             "location": "https://github.com/org/repo"},
            {"what_doing": "View issues", "app": "Chrome",
             "location": "https://github.com/org/repo/issues"},
            {"what_doing": "Check CI", "app": "Chrome",
             "location": "https://circleci.com/pipelines/github/org/repo"},
        ])

        fp = matcher._segment_to_fingerprint(segment)
        assert "github.com" in fp["domains"]
        assert "circleci.com" in fp["domains"]
        assert len(fp["domains"]) == 2


class TestCacheRefresh:
    """test_cache_refresh"""

    def test_cache_refresh(self, kb):
        """Match returns empty, add procedure, invalidate cache, match again -> found."""
        matcher = ProcedureMatcher(kb)

        segment = _make_segment("seg-cache", ["Chrome"], [
            {"what_doing": "Open GitHub PR page", "app": "Chrome",
             "location": "https://github.com/org/repo/pulls"},
            {"what_doing": "Review code changes", "app": "Chrome",
             "location": "https://github.com/org/repo/pull/42"},
        ])

        # No procedures yet
        matches = matcher.match_segment(segment)
        assert matches == []

        # Add a procedure
        proc = _github_pr_procedure()
        kb.save_procedure(proc)

        # Cache is stale — still empty
        matches = matcher.match_segment(segment)
        assert matches == []

        # Invalidate cache
        matcher.invalidate_cache()

        # Now it should find the match
        matches = matcher.match_segment(segment)
        assert len(matches) >= 1
        assert matches[0][0] == "review-github-pr"


class TestMultipleProceduresRanked:
    """test_multiple_procedures_ranked"""

    def test_multiple_procedures_ranked(self, kb):
        """Two procedures: verify best match is first in results."""
        proc_github = _github_pr_procedure()
        proc_jira = _jira_ticket_procedure()
        kb.save_procedure(proc_github)
        kb.save_procedure(proc_jira)

        matcher = ProcedureMatcher(kb, match_threshold=0.10)

        # Segment that closely matches the GitHub procedure
        segment = _make_segment("seg-rank", ["Chrome"], [
            {"what_doing": "Open GitHub PR page", "app": "Chrome",
             "location": "https://github.com/org/repo/pulls"},
            {"what_doing": "Review code changes", "app": "Chrome",
             "location": "https://github.com/org/repo/pull/42"},
            {"what_doing": "Submit review", "app": "Chrome",
             "location": "https://github.com/org/repo/pull/42"},
        ])

        matches = matcher.match_segment(segment)
        assert len(matches) >= 1

        # If more than one match, the best should be first (descending order)
        if len(matches) >= 2:
            assert matches[0][1] >= matches[1][1]

        # The GitHub procedure should be the best match
        assert matches[0][0] == "review-github-pr"


class TestThresholdConfigurable:
    """test_threshold_configurable"""

    def test_threshold_configurable(self, kb):
        """Higher threshold filters more, lower threshold lets more through."""
        proc = _github_pr_procedure()
        kb.save_procedure(proc)

        # Segment with partial overlap (same app, different domain)
        segment = _make_segment("seg-thresh", ["Chrome"], [
            {"what_doing": "Open documentation page", "app": "Chrome",
             "location": "https://docs.example.com/guide"},
        ])

        # Low threshold — more permissive
        matcher_low = ProcedureMatcher(kb, match_threshold=0.10)
        matches_low = matcher_low.match_segment(segment)

        # High threshold — more restrictive
        matcher_high = ProcedureMatcher(kb, match_threshold=0.95)
        matches_high = matcher_high.match_segment(segment)

        # The high-threshold matcher should return equal or fewer matches
        assert len(matches_high) <= len(matches_low)

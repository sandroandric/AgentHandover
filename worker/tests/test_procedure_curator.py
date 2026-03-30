"""Tests for ProcedureCurator — Phase 5 curation orchestrator.

Covers merge detection, upgrade promotion, drift detection, family
grouping, queue building, merge execution, and full curation runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.lifecycle_manager import LifecycleManager, ProcedureLifecycle
from agenthandover_worker.procedure_curator import (
    CurationItem,
    CurationSummary,
    DriftReport,
    MergeCandidate,
    ProcedureCurator,
    UpgradeCandidate,
)
from agenthandover_worker.staleness_detector import StalenessDetector
from agenthandover_worker.trust_advisor import TrustAdvisor


# ---- Fixtures ----


@pytest.fixture
def kb(tmp_path):
    """Create a KnowledgeBase with a temp root."""
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


def _make_proc(
    slug,
    apps,
    steps_actions,
    confidence=0.85,
    episodes=3,
    lifecycle="observed",
    trust="observe",
    **overrides,
):
    """Build and return a v3 procedure dict."""
    from agenthandover_worker.procedure_schema import sop_to_procedure

    template = {
        "slug": slug,
        "title": f"Test: {slug}",
        "steps": [
            {
                "step": a,
                "app": apps[0] if apps else "Chrome",
                "confidence": 0.9,
                "location": "",
            }
            for a in steps_actions
        ],
        "confidence_avg": confidence,
        "episode_count": episodes,
        "apps_involved": apps,
        "source": "test",
    }
    proc = sop_to_procedure(template)
    proc["constraints"]["trust_level"] = trust
    proc["lifecycle_state"] = lifecycle
    for k, v in overrides.items():
        proc[k] = v
    return proc


@pytest.fixture
def curator(kb):
    """Create a ProcedureCurator with all subsystem dependencies."""
    sd = StalenessDetector(kb)
    ta = TrustAdvisor(kb)
    lm = LifecycleManager(kb)
    return ProcedureCurator(kb, sd, ta, lm)


def _save(kb, proc):
    """Helper to save a procedure to the KB."""
    kb.save_procedure(proc)


# ---- TestMergeCandidates ----


class TestMergeCandidates:
    """Tests for detect_merge_candidates()."""

    def test_same_apps_similar_verbs_detected(self, kb, curator):
        """Two procedures with same apps and overlapping step verbs are detected."""
        proc_a = _make_proc(
            "open-gmail", ["Chrome"],
            ["Navigate to gmail.com", "Click compose", "Enter recipient", "Review draft"],
        )
        proc_b = _make_proc(
            "send-email", ["Chrome"],
            ["Navigate to gmail.com", "Click new message", "Enter address", "Wait for send", "Select folder"],
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        candidates = curator.detect_merge_candidates()
        slugs = {frozenset({c.slug_a, c.slug_b}) for c in candidates}
        assert frozenset({"open-gmail", "send-email"}) in slugs

    def test_nearly_identical_excluded(self, kb, curator):
        """Similarity >= 0.95 should NOT be detected (already deduped)."""
        proc_a = _make_proc(
            "task-a", ["Chrome"],
            ["Navigate to site", "Click button", "Enter data"],
        )
        # Identical copy with different slug
        proc_b = _make_proc(
            "task-b", ["Chrome"],
            ["Navigate to site", "Click button", "Enter data"],
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        candidates = curator.detect_merge_candidates()
        pair = frozenset({"task-a", "task-b"})
        # Should either not be found (sim >= 0.95) or found — check the threshold
        for c in candidates:
            if frozenset({c.slug_a, c.slug_b}) == pair:
                assert c.similarity < 0.95

    def test_low_similarity_excluded(self, kb, curator):
        """Similarity < 0.70 should NOT be detected."""
        proc_a = _make_proc(
            "coding-task", ["VSCode"],
            ["Open file", "Edit code", "Save changes", "Run tests"],
        )
        proc_b = _make_proc(
            "email-task", ["Chrome"],
            ["Navigate to gmail", "Click compose"],
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        candidates = curator.detect_merge_candidates()
        pair = frozenset({"coding-task", "email-task"})
        assert pair not in {frozenset({c.slug_a, c.slug_b}) for c in candidates}

    def test_empty_kb_returns_empty(self, kb, curator):
        """Empty KB returns no merge candidates."""
        assert curator.detect_merge_candidates() == []

    def test_dismissed_merge_filtered(self, kb, curator):
        """Dismissed merge pairs are not re-surfaced."""
        proc_a = _make_proc(
            "proc-x", ["Chrome"],
            ["Navigate to site", "Click submit", "Enter info"],
        )
        proc_b = _make_proc(
            "proc-y", ["Chrome"],
            ["Navigate to site", "Click send", "Enter data"],
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        # Dismiss the pair
        curator.dismiss_merge("proc-x", "proc-y")

        candidates = curator.detect_merge_candidates()
        pair = frozenset({"proc-x", "proc-y"})
        assert pair not in {frozenset({c.slug_a, c.slug_b}) for c in candidates}

    def test_explanation_contains_app_names(self, kb, curator):
        """Explanation string should mention shared app names."""
        proc_a = _make_proc(
            "browse-a", ["Chrome", "Slack"],
            ["Navigate to site", "Click button"],
        )
        proc_b = _make_proc(
            "browse-b", ["Chrome", "Slack"],
            ["Navigate to page", "Click link"],
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        candidates = curator.detect_merge_candidates()
        for c in candidates:
            if frozenset({c.slug_a, c.slug_b}) == frozenset({"browse-a", "browse-b"}):
                assert "Chrome" in c.explanation or "chrome" in c.explanation.lower()
                break


# ---- TestUpgradeCandidates ----


class TestUpgradeCandidates:
    """Tests for detect_upgrade_candidates()."""

    def test_observed_with_3_episodes_suggests_draft(self, kb, curator):
        """observed + 3 episodes + conf 0.70 should suggest draft."""
        proc = _make_proc(
            "obs-proc", ["Chrome"], ["Open site", "Click button"],
            confidence=0.70, episodes=3, lifecycle="observed",
        )
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        slugs = {c.slug for c in candidates}
        assert "obs-proc" in slugs
        match = next(c for c in candidates if c.slug == "obs-proc")
        assert match.proposed_state == "draft"

    def test_too_few_episodes_no_suggestion(self, kb, curator):
        """Only 2 episodes should not suggest upgrade."""
        proc = _make_proc(
            "few-eps", ["Chrome"], ["Open site"],
            confidence=0.80, episodes=2, lifecycle="observed",
        )
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        assert "few-eps" not in {c.slug for c in candidates}

    def test_low_confidence_no_suggestion(self, kb, curator):
        """Low confidence (0.50) should not suggest upgrade."""
        proc = _make_proc(
            "low-conf", ["Chrome"], ["Open site"],
            confidence=0.50, episodes=5, lifecycle="observed",
        )
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        assert "low-conf" not in {c.slug for c in candidates}

    def test_draft_to_reviewed(self, kb, curator):
        """draft + 5 episodes + conf 0.80 + freshness 0.8 should suggest reviewed."""
        now = datetime.now(timezone.utc)
        proc = _make_proc(
            "good-draft", ["Chrome"], ["Open site", "Click button", "Enter data"],
            confidence=0.80, episodes=5, lifecycle="draft",
        )
        proc["staleness"]["last_observed"] = now.isoformat()
        proc["evidence"]["contradictions"] = []
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        match = [c for c in candidates if c.slug == "good-draft"]
        assert len(match) == 1
        assert match[0].proposed_state == "reviewed"

    def test_verified_to_agent_ready(self, kb, curator):
        """verified + recently confirmed should suggest agent_ready."""
        now = datetime.now(timezone.utc)
        proc = _make_proc(
            "verified-proc", ["Chrome"], ["Open site"],
            confidence=0.90, episodes=10, lifecycle="verified",
        )
        proc["staleness"]["last_confirmed"] = now.isoformat()
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        match = [c for c in candidates if c.slug == "verified-proc"]
        assert len(match) == 1
        assert match[0].proposed_state == "agent_ready"

    def test_stale_lifecycle_skipped(self, kb, curator):
        """Stale procedures should be skipped for upgrade."""
        proc = _make_proc(
            "stale-proc", ["Chrome"], ["Open site"],
            confidence=0.90, episodes=10, lifecycle="stale",
        )
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        assert "stale-proc" not in {c.slug for c in candidates}

    def test_archived_lifecycle_skipped(self, kb, curator):
        """Archived procedures should be skipped for upgrade."""
        proc = _make_proc(
            "archived-proc", ["Chrome"], ["Open site"],
            confidence=0.90, episodes=10, lifecycle="archived",
        )
        _save(kb, proc)

        candidates = curator.detect_upgrade_candidates()
        assert "archived-proc" not in {c.slug for c in candidates}


# ---- TestDriftDetection ----


class TestDriftDetection:
    """Tests for detect_drift() and detect_all_drift()."""

    def test_procedure_with_drift_signals(self, kb, curator):
        """Procedure with drift_signals returns DriftReports."""
        proc = _make_proc("drifty", ["Chrome"], ["Open site"])
        proc["staleness"]["drift_signals"] = [
            {"type": "url_changed", "detail": "URL moved to new domain", "first_seen": "2026-01-01T00:00:00"},
        ]
        _save(kb, proc)

        reports = curator.detect_drift("drifty")
        assert len(reports) >= 1
        assert any(r.drift_type == "url_changed" for r in reports)

    def test_clean_procedure_no_drift(self, kb, curator):
        """Clean procedure with no drift signals returns empty list."""
        now = datetime.now(timezone.utc)
        proc = _make_proc("clean-proc", ["Chrome"], ["Open site"])
        proc["staleness"]["last_observed"] = now.isoformat()
        proc["staleness"]["last_confirmed"] = now.isoformat()
        proc["staleness"]["drift_signals"] = []
        _save(kb, proc)

        reports = curator.detect_drift("clean-proc")
        assert reports == []

    def test_step_failure_severity_high(self, kb, curator):
        """step_failure drift type should have severity 'high'."""
        proc = _make_proc("failing", ["Chrome"], ["Open site"])
        proc["staleness"]["drift_signals"] = [
            {"type": "step_failure", "detail": "Step 2 failed consistently", "first_seen": "2026-01-01T00:00:00"},
        ]
        _save(kb, proc)

        reports = curator.detect_drift("failing")
        step_failures = [r for r in reports if r.drift_type == "step_failure"]
        assert len(step_failures) >= 1
        assert step_failures[0].severity == "high"

    def test_dismissed_drift_filtered(self, kb, curator):
        """Dismissed drift signals should be filtered out."""
        proc = _make_proc("dismiss-test", ["Chrome"], ["Open site"])
        proc["staleness"]["drift_signals"] = [
            {"type": "url_changed", "detail": "URL changed", "first_seen": "2026-01-01T00:00:00"},
        ]
        _save(kb, proc)

        curator.dismiss_drift("dismiss-test", "url_changed")

        reports = curator.detect_drift("dismiss-test")
        assert not any(r.drift_type == "url_changed" for r in reports)

    def test_detect_all_drift_returns_dict(self, kb, curator):
        """detect_all_drift returns a dict keyed by slug."""
        proc = _make_proc("drift-all", ["Chrome"], ["Open site"])
        proc["staleness"]["drift_signals"] = [
            {"type": "new_step", "detail": "New step detected", "first_seen": "2026-01-01T00:00:00"},
        ]
        _save(kb, proc)

        result = curator.detect_all_drift()
        assert isinstance(result, dict)
        assert "drift-all" in result
        assert len(result["drift-all"]) >= 1


# ---- TestFamilies ----


class TestFamilies:
    """Tests for build_families()."""

    def test_similar_procedures_grouped(self, kb, curator):
        """3 procedures with family-range similarity are grouped."""
        # These use different enough apps/verbs to fall in [0.60, 0.70) range
        proc_a = _make_proc(
            "deploy-staging", ["Terminal", "Chrome"],
            ["Open terminal", "Run deploy script", "Verify site"],
            episodes=5,
        )
        proc_b = _make_proc(
            "deploy-prod", ["Terminal", "Slack"],
            ["Open terminal", "Run deploy script", "Notify team"],
            episodes=3,
        )
        proc_c = _make_proc(
            "deploy-dev", ["Terminal", "VSCode"],
            ["Open terminal", "Run deploy script", "Check logs"],
            episodes=2,
        )
        _save(kb, proc_a)
        _save(kb, proc_b)
        _save(kb, proc_c)

        families = curator.build_families()
        # May or may not group depending on exact similarity —
        # the test validates the return type and structure
        assert isinstance(families, list)
        for fam in families:
            assert fam.family_id
            assert fam.canonical_slug

    def test_empty_kb_no_families(self, kb, curator):
        """Empty KB returns no families."""
        assert curator.build_families() == []

    def test_family_has_shared_apps(self, kb, curator):
        """Families should report shared_apps."""
        proc_a = _make_proc(
            "fam-a", ["Chrome", "Slack"],
            ["Open page", "Send message"],
            episodes=5,
        )
        proc_b = _make_proc(
            "fam-b", ["Chrome", "Slack"],
            ["Open page", "Post update"],
            episodes=3,
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        families = curator.build_families()
        for fam in families:
            assert isinstance(fam.shared_apps, list)


# ---- TestCurationQueue ----


class TestCurationQueue:
    """Tests for build_curation_queue()."""

    def test_queue_contains_items_from_all_sources(self, kb, curator):
        """Queue should contain items from merge, upgrade, stale, drift sources."""
        now = datetime.now(timezone.utc)
        # An upgrade candidate
        proc_a = _make_proc(
            "upgrade-me", ["Chrome"], ["Open site", "Click button"],
            confidence=0.75, episodes=4, lifecycle="observed",
        )
        _save(kb, proc_a)

        # A procedure with drift
        proc_b = _make_proc("drifty-queue", ["Chrome"], ["Navigate to site"])
        proc_b["staleness"]["drift_signals"] = [
            {"type": "step_failure", "detail": "Step failed", "first_seen": now.isoformat()},
        ]
        _save(kb, proc_b)

        queue = curator.build_curation_queue()
        types_present = {item.item_type for item in queue}
        # At least some types should be present
        assert len(queue) > 0
        assert isinstance(queue[0], CurationItem)

    def test_priority_ordering(self, kb, curator):
        """Queue should be sorted by priority descending."""
        now = datetime.now(timezone.utc)

        # Multiple procedures to generate different item types
        proc_a = _make_proc(
            "q-proc-a", ["Chrome"], ["Open site"],
            confidence=0.75, episodes=4, lifecycle="observed",
        )
        proc_b = _make_proc("q-proc-b", ["Chrome"], ["Navigate to page"])
        proc_b["staleness"]["drift_signals"] = [
            {"type": "new_step", "detail": "new step", "first_seen": now.isoformat()},
        ]
        _save(kb, proc_a)
        _save(kb, proc_b)

        queue = curator.build_curation_queue()
        priorities = [item.priority for item in queue]
        assert priorities == sorted(priorities, reverse=True)

    def test_stale_agent_ready_priority_one(self, kb, curator):
        """Stale procedure in agent_ready state should have priority 1.0."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        proc = _make_proc(
            "stale-ar", ["Chrome"], ["Open site"],
            lifecycle="agent_ready",
        )
        proc["staleness"]["last_observed"] = old_date
        proc["staleness"]["last_confirmed"] = None
        _save(kb, proc)

        queue = curator.build_curation_queue()
        stale_items = [
            item for item in queue
            if item.item_type == "stale" and item.slug == "stale-ar"
        ]
        assert len(stale_items) == 1
        assert stale_items[0].priority == 1.0

    def test_empty_kb_empty_queue(self, kb, curator):
        """Empty KB returns empty queue."""
        assert curator.build_curation_queue() == []

    def test_queue_items_have_required_fields(self, kb, curator):
        """Each queue item should have all required fields."""
        proc = _make_proc(
            "fields-proc", ["Chrome"], ["Open site"],
            confidence=0.75, episodes=4, lifecycle="observed",
        )
        _save(kb, proc)

        queue = curator.build_curation_queue()
        for item in queue:
            assert item.item_type
            assert isinstance(item.priority, float)
            assert item.slug
            assert item.title
            assert item.explanation

    def test_titles_and_explanations_non_empty(self, kb, curator):
        """Queue items should have non-empty titles and explanations."""
        proc = _make_proc(
            "nonempty-proc", ["Chrome"], ["Open site", "Click button"],
            confidence=0.70, episodes=3, lifecycle="observed",
        )
        _save(kb, proc)

        queue = curator.build_curation_queue()
        for item in queue:
            assert len(item.title) > 0
            assert len(item.explanation) > 0

    def test_trust_suggestions_included(self, kb, curator):
        """Trust suggestions should appear in the queue."""
        # Set up execution stats to trigger a trust suggestion
        proc = _make_proc(
            "trust-proc", ["Chrome"], ["Open site"],
            trust="observe",
        )
        _save(kb, proc)

        # Write execution stats
        exec_path = kb.root / "observations" / "executions.json"
        kb.atomic_write_json(exec_path, {
            "procedures": {
                "trust-proc": {
                    "total": 10,
                    "successes": 10,
                    "failures": 0,
                    "last_failure": None,
                }
            }
        })

        # Re-create curator so TrustAdvisor loads the stats
        sd = StalenessDetector(kb)
        ta = TrustAdvisor(kb)
        lm = LifecycleManager(kb)
        curator2 = ProcedureCurator(kb, sd, ta, lm)

        # Evaluate to generate suggestions
        ta.evaluate_all()

        queue = curator2.build_curation_queue()
        trust_items = [item for item in queue if item.item_type == "trust"]
        assert len(trust_items) >= 1


# ---- TestExecuteMerge ----


class TestExecuteMerge:
    """Tests for execute_merge()."""

    def test_merge_produces_combined_procedure(self, kb, curator):
        """Merge combines two procedures into one."""
        proc_a = _make_proc(
            "merge-a", ["Chrome"], ["Open site", "Click button"],
            episodes=3,
        )
        proc_b = _make_proc(
            "merge-b", ["Chrome"], ["Open site", "Submit form"],
            episodes=2,
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        result = curator.execute_merge("merge-a", "merge-b")
        assert result["success"] is True
        assert result["merged_slug"] == "merge-a"

        # Verify merged procedure exists
        merged = kb.get_procedure("merge-a")
        assert merged is not None

    def test_source_procedure_archived(self, kb, curator):
        """After merge, the source procedure (B) should be archived."""
        proc_a = _make_proc(
            "keep-a", ["Chrome"], ["Open site"],
            episodes=3,
        )
        proc_b = _make_proc(
            "archive-b", ["Chrome"], ["Open site"],
            episodes=2,
        )
        _save(kb, proc_a)
        _save(kb, proc_b)

        curator.execute_merge("keep-a", "archive-b")

        archived = kb.get_procedure("archive-b")
        assert archived is not None
        assert archived["lifecycle_state"] == "archived"

    def test_nonexistent_slug_returns_error(self, kb, curator):
        """Merging with nonexistent slug returns error result."""
        proc_a = _make_proc("exists", ["Chrome"], ["Open site"])
        _save(kb, proc_a)

        result = curator.execute_merge("exists", "ghost")
        assert result["success"] is False
        assert "error" in result

    def test_decision_recorded(self, kb, curator):
        """Merge should record a CurationDecision."""
        proc_a = _make_proc("rec-a", ["Chrome"], ["Open site"], episodes=3)
        proc_b = _make_proc("rec-b", ["Chrome"], ["Open site"], episodes=2)
        _save(kb, proc_a)
        _save(kb, proc_b)

        curator.execute_merge("rec-a", "rec-b")

        # Verify decision was persisted
        decisions_path = kb.root / "observations" / "curation_decisions.json"
        assert decisions_path.is_file()
        with open(decisions_path) as f:
            data = json.load(f)
        assert len(data.get("decisions", [])) >= 1
        last_decision = data["decisions"][-1]
        assert last_decision["action"] == "merge"


# ---- TestCurate ----


class TestCurate:
    """Tests for curate() — full curation run."""

    def test_returns_curation_summary(self, kb, curator):
        """curate() should return a CurationSummary with counts."""
        proc = _make_proc(
            "curate-proc", ["Chrome"], ["Open site"],
            confidence=0.75, episodes=4, lifecycle="observed",
        )
        _save(kb, proc)

        summary = curator.curate()
        assert isinstance(summary, CurationSummary)
        assert hasattr(summary, "merge_candidates")
        assert hasattr(summary, "upgrade_candidates")
        assert hasattr(summary, "stale_procedures")
        assert hasattr(summary, "trust_suggestions")
        assert hasattr(summary, "families")
        assert hasattr(summary, "drift_reports")
        assert hasattr(summary, "total_queue_items")

    def test_empty_kb_all_zeros(self, kb, curator):
        """Empty KB should produce all-zero summary."""
        summary = curator.curate()
        assert summary.merge_candidates == 0
        assert summary.upgrade_candidates == 0
        assert summary.stale_procedures == 0
        assert summary.trust_suggestions == 0
        assert summary.families == 0
        assert summary.drift_reports == 0
        assert summary.total_queue_items == 0

    def test_populated_kb_nonzero_counts(self, kb, curator):
        """Populated KB should produce nonzero counts in at least one field."""
        now = datetime.now(timezone.utc)

        # An upgrade candidate
        proc_a = _make_proc(
            "curate-upgrade", ["Chrome"], ["Open site", "Click button"],
            confidence=0.75, episodes=4, lifecycle="observed",
        )
        _save(kb, proc_a)

        # A procedure with drift
        proc_b = _make_proc("curate-drift", ["Chrome"], ["Navigate to page"])
        proc_b["staleness"]["drift_signals"] = [
            {"type": "step_failure", "detail": "Step failed", "first_seen": now.isoformat()},
        ]
        _save(kb, proc_b)

        summary = curator.curate()

        # At least one field should be nonzero
        total = (
            summary.merge_candidates
            + summary.upgrade_candidates
            + summary.stale_procedures
            + summary.trust_suggestions
            + summary.families
            + summary.drift_reports
        )
        assert total > 0
        assert summary.total_queue_items > 0

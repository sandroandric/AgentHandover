"""Tests for the escalation_handler module — 15 tests.

Covers:
- Handle failure (5)
- Handle deviation (3)
- Recent failures counting (3)
- Demotion (3)
- Edge cases (1)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.escalation_handler import (
    EscalationDecision,
    EscalationHandler,
    EscalationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


def _save_proc(kb: KnowledgeBase, slug: str, **overrides) -> dict:
    """Save a minimal procedure and return the dict."""
    proc = {
        "id": slug,
        "slug": slug,
        "title": f"Test {slug}",
        "steps": [],
        "environment": {},
        "expected_outcomes": [],
    }
    proc.update(overrides)
    kb.save_procedure(proc)
    return proc


def _write_executions(kb: KnowledgeBase, records: list[dict]) -> None:
    """Write execution history directly to the KB observations dir."""
    path = kb.root / "observations" / "executions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    kb.atomic_write_json(
        path,
        {
            "records": records,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _make_failure_record(
    slug: str,
    status: str = "failed",
    days_ago: int = 0,
) -> dict:
    """Create a minimal execution record for testing."""
    ts = (
        datetime.now(timezone.utc) - timedelta(days=days_ago)
    ).isoformat()
    return {
        "execution_id": f"exec-{slug}-{days_ago}-{status}",
        "procedure_slug": slug,
        "agent_id": "test",
        "status": status,
        "started_at": ts,
        "completed_at": ts,
    }


# ===========================================================================
# 1) TestHandleFailure (5)
# ===========================================================================


class TestHandleFailure:
    """Test handle_failure() escalation logic."""

    def test_first_failure_with_retry(self, kb: KnowledgeBase) -> None:
        """Procedure with retry strategy returns RETRY."""
        _save_proc(
            kb,
            "retry-proc",
            steps=[
                {
                    "action": "Do thing",
                    "on_failure": {"strategy": "retry", "max_retries": 3},
                }
            ],
        )
        handler = EscalationHandler(kb)
        result = handler.handle_failure("retry-proc", "exec-1", "Connection refused")
        assert result.decision == EscalationDecision.RETRY
        assert result.max_retries == 3

    def test_no_retry_strategy(self, kb: KnowledgeBase) -> None:
        """No retry strategy -> ABORT_NOTIFY."""
        _save_proc(
            kb,
            "no-retry",
            steps=[
                {
                    "action": "Do thing",
                    "on_failure": {"strategy": "abort"},
                }
            ],
        )
        handler = EscalationHandler(kb)
        result = handler.handle_failure("no-retry", "exec-1", "Timeout")
        assert result.decision == EscalationDecision.ABORT_NOTIFY
        assert "Timeout" in result.reason

    def test_exhausted_retries(self, kb: KnowledgeBase) -> None:
        """No retry strategy in steps -> ABORT_NOTIFY (caller manages counts)."""
        _save_proc(
            kb,
            "exhausted",
            steps=[{"action": "Do thing"}],
        )
        handler = EscalationHandler(kb)
        result = handler.handle_failure("exhausted", "exec-1", "Still failing")
        assert result.decision == EscalationDecision.ABORT_NOTIFY

    def test_demotion_threshold_reached(self, kb: KnowledgeBase) -> None:
        """3+ recent failures -> DEMOTE."""
        _save_proc(kb, "flaky")
        _write_executions(
            kb,
            [
                _make_failure_record("flaky", "failed", 0),
                _make_failure_record("flaky", "failed", 1),
                _make_failure_record("flaky", "deviated", 2),
            ],
        )
        handler = EscalationHandler(kb)
        result = handler.handle_failure("flaky", "exec-4", "Another failure")
        assert result.decision == EscalationDecision.DEMOTE
        assert "3" in result.reason

    def test_procedure_not_found(self, kb: KnowledgeBase) -> None:
        """Missing procedure -> ABORT_NOTIFY."""
        handler = EscalationHandler(kb)
        result = handler.handle_failure("ghost", "exec-1", "error")
        assert result.decision == EscalationDecision.ABORT_NOTIFY
        assert "not found" in result.reason


# ===========================================================================
# 2) TestHandleDeviation (3)
# ===========================================================================


class TestHandleDeviation:
    """Test handle_deviation() escalation logic."""

    def test_deviation_below_threshold(self, kb: KnowledgeBase) -> None:
        """Deviation with no prior failures -> ABORT_NOTIFY."""
        _save_proc(kb, "dev-proc")
        handler = EscalationHandler(kb)
        result = handler.handle_deviation("dev-proc", "exec-1", "Went off script")
        assert result.decision == EscalationDecision.ABORT_NOTIFY

    def test_deviation_hits_threshold(self, kb: KnowledgeBase) -> None:
        """3+ recent failures/deviations -> DEMOTE."""
        _save_proc(kb, "dev-proc")
        _write_executions(
            kb,
            [
                _make_failure_record("dev-proc", "deviated", 0),
                _make_failure_record("dev-proc", "deviated", 1),
                _make_failure_record("dev-proc", "failed", 2),
            ],
        )
        handler = EscalationHandler(kb)
        result = handler.handle_deviation("dev-proc", "exec-4", "Went off again")
        assert result.decision == EscalationDecision.DEMOTE

    def test_deviation_detail_in_reason(self, kb: KnowledgeBase) -> None:
        """Deviation detail text appears in the reason field."""
        _save_proc(kb, "detail-proc")
        handler = EscalationHandler(kb)
        result = handler.handle_deviation(
            "detail-proc", "exec-1", "Used different browser"
        )
        assert "Used different browser" in result.reason


# ===========================================================================
# 3) TestRecentFailures (3)
# ===========================================================================


class TestRecentFailures:
    """Test get_recent_failures() counting logic."""

    def test_count_failures_in_window(self, kb: KnowledgeBase) -> None:
        """Two recent failures are counted."""
        _save_proc(kb, "count-proc")
        _write_executions(
            kb,
            [
                _make_failure_record("count-proc", "failed", 0),
                _make_failure_record("count-proc", "deviated", 3),
            ],
        )
        handler = EscalationHandler(kb)
        assert handler.get_recent_failures("count-proc") == 2

    def test_old_failures_excluded(self, kb: KnowledgeBase) -> None:
        """Failures older than 7 days are not counted."""
        _save_proc(kb, "old-proc")
        _write_executions(
            kb,
            [
                _make_failure_record("old-proc", "failed", 0),
                _make_failure_record("old-proc", "failed", 10),  # outside window
            ],
        )
        handler = EscalationHandler(kb)
        assert handler.get_recent_failures("old-proc") == 1

    def test_no_history_returns_zero(self, kb: KnowledgeBase) -> None:
        """No executions.json file -> 0 failures."""
        handler = EscalationHandler(kb)
        assert handler.get_recent_failures("any-slug") == 0


# ===========================================================================
# 4) TestDemotion (3)
# ===========================================================================


class TestDemotion:
    """Test _apply_demotion() lifecycle integration."""

    def test_demotion_transitions_lifecycle(self, kb: KnowledgeBase) -> None:
        """agent_ready -> stale via lifecycle manager."""
        from agenthandover_worker.lifecycle_manager import (
            LifecycleManager,
            ProcedureLifecycle,
        )

        _save_proc(kb, "demote-me")
        # Set lifecycle_state to agent_ready
        proc = kb.get_procedure("demote-me")
        proc["lifecycle_state"] = "agent_ready"
        kb.save_procedure(proc)

        lm = LifecycleManager(kb)
        handler = EscalationHandler(kb, lifecycle_manager=lm)

        _write_executions(
            kb,
            [
                _make_failure_record("demote-me", "failed", 0),
                _make_failure_record("demote-me", "failed", 1),
                _make_failure_record("demote-me", "failed", 2),
            ],
        )

        result = handler.handle_failure("demote-me", "exec-4", "Another boom")
        assert result.decision == EscalationDecision.DEMOTE
        assert result.demoted is True
        assert lm.get_state("demote-me") == ProcedureLifecycle.STALE

    def test_demotion_without_lifecycle_manager(
        self, kb: KnowledgeBase
    ) -> None:
        """No lifecycle manager -> demoted=False, no crash."""
        handler = EscalationHandler(kb, lifecycle_manager=None)
        result = handler._apply_demotion("any-slug")
        assert result is False

    def test_demotion_from_draft(self, kb: KnowledgeBase) -> None:
        """Demotion is applied when current state is draft."""
        from agenthandover_worker.lifecycle_manager import (
            LifecycleManager,
            ProcedureLifecycle,
        )

        _save_proc(kb, "draft-proc")
        proc = kb.get_procedure("draft-proc")
        proc["lifecycle_state"] = "draft"
        kb.save_procedure(proc)

        lm = LifecycleManager(kb)
        handler = EscalationHandler(kb, lifecycle_manager=lm)

        # _apply_demotion covers both agent_ready and draft
        result = handler._apply_demotion("draft-proc")
        assert result is True
        assert lm.get_state("draft-proc") == ProcedureLifecycle.STALE

    def test_demotion_not_from_observed(self, kb: KnowledgeBase) -> None:
        """Demotion is not applied when current state is observed."""
        from agenthandover_worker.lifecycle_manager import (
            LifecycleManager,
            ProcedureLifecycle,
        )

        _save_proc(kb, "observed-proc")
        proc = kb.get_procedure("observed-proc")
        proc["lifecycle_state"] = "observed"
        kb.save_procedure(proc)

        lm = LifecycleManager(kb)
        handler = EscalationHandler(kb, lifecycle_manager=lm)

        # _apply_demotion should not demote from observed
        result = handler._apply_demotion("observed-proc")
        assert result is False
        assert lm.get_state("observed-proc") == ProcedureLifecycle.OBSERVED


# ===========================================================================
# 5) TestEdgeCases (1)
# ===========================================================================


class TestEdgeCases:
    """Edge case coverage."""

    def test_check_demotion_threshold(self, kb: KnowledgeBase) -> None:
        """check_demotion_threshold returns bool correctly."""
        _save_proc(kb, "threshold-proc")
        handler = EscalationHandler(kb)

        # No failures -> False
        assert handler.check_demotion_threshold("threshold-proc") is False

        # Write 3 failures -> True
        _write_executions(
            kb,
            [
                _make_failure_record("threshold-proc", "failed", 0),
                _make_failure_record("threshold-proc", "failed", 1),
                _make_failure_record("threshold-proc", "deviated", 2),
            ],
        )
        assert handler.check_demotion_threshold("threshold-proc") is True

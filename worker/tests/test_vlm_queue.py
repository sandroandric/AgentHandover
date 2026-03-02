"""Tests for the VLM Fallback Queue.

Covers:
1.  test_priority_calculation — verify formula with known values
2.  test_enqueue_within_budget — job enqueued successfully
3.  test_enqueue_exceeds_daily_limit — returns False after 50 jobs
4.  test_dequeue_highest_priority — priority ordering
5.  test_expire_stale_jobs — jobs past TTL marked expired
6.  test_backpressure_drops_lowest — queue > max drops low priority
7.  test_can_dispatch_budget_check — daily dispatch limit
8.  test_record_completion — stats updated
9.  test_compute_minutes_budget — stops after budget exhausted
10. test_daily_counter_reset — counters reset at midnight
11. test_dequeue_returns_none_when_empty
12. test_record_completion_unknown_job_raises
13. test_ttl_auto_set_on_enqueue
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from oc_apprentice_worker.vlm_queue import (
    DEFAULT_RISK_WEIGHT,
    RISK_WEIGHTS,
    QueueBudget,
    QueueStats,
    VLMFallbackQueue,
    VLMJob,
    VLMJobStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    priority: float = 0.5,
    confidence: float = 0.3,
    job_id: str | None = None,
    created_at: datetime | None = None,
    ttl_expires_at: datetime | None = None,
    status: VLMJobStatus = VLMJobStatus.PENDING,
) -> VLMJob:
    """Create a VLM job with sensible defaults."""
    now = datetime.now(timezone.utc)
    return VLMJob(
        job_id=job_id or str(uuid.uuid4()),
        event_id=f"evt-{uuid.uuid4().hex[:8]}",
        episode_id=f"ep-{uuid.uuid4().hex[:8]}",
        semantic_step_index=0,
        confidence_score=confidence,
        priority_score=priority,
        query="What is this element?",
        created_at=created_at or now,
        ttl_expires_at=ttl_expires_at,
        status=status,
    )


# ---------------------------------------------------------------------------
# Test 1: Priority calculation
# ---------------------------------------------------------------------------


class TestPriorityCalculation:
    def test_known_values(self) -> None:
        """Verify the formula with known inputs.

        priority = (1 - confidence) * risk_weight * recency_weight

        With confidence=0.2, intent="send_email" (risk=1.0), and
        created_at=now (recency~=1.0), priority should be ~0.8.
        """
        queue = VLMFallbackQueue()
        now = datetime.now(timezone.utc)
        priority = queue.compute_priority(
            confidence=0.2,
            intent="send_email",
            created_at=now,
        )

        # (1 - 0.2) * 1.0 * exp(~0 / 24) ≈ 0.8 * 1.0 * 1.0 = 0.8
        assert 0.75 <= priority <= 0.85

    def test_high_confidence_low_priority(self) -> None:
        """High confidence should yield low priority."""
        queue = VLMFallbackQueue()
        now = datetime.now(timezone.utc)
        priority = queue.compute_priority(
            confidence=0.95,
            intent="click",
            created_at=now,
        )
        # (1 - 0.95) * 0.6 * ~1.0 = 0.05 * 0.6 = 0.03
        assert priority < 0.05

    def test_old_event_decayed_priority(self) -> None:
        """An event from 48 hours ago should have heavily decayed priority."""
        queue = VLMFallbackQueue()
        old = datetime.now(timezone.utc) - timedelta(hours=48)
        priority = queue.compute_priority(
            confidence=0.0,
            intent="send_email",
            created_at=old,
        )

        # recency_weight = exp(-48/24) = exp(-2) ≈ 0.135
        # (1 - 0) * 1.0 * 0.135 ≈ 0.135
        expected_recency = math.exp(-48.0 / 24.0)
        assert abs(priority - expected_recency) < 0.02

    def test_unknown_intent_uses_default_weight(self) -> None:
        queue = VLMFallbackQueue()
        now = datetime.now(timezone.utc)
        priority = queue.compute_priority(
            confidence=0.0,
            intent="unknown_action_type",
            created_at=now,
        )
        # (1 - 0) * 0.5 * ~1.0 = 0.5
        assert 0.45 <= priority <= 0.55

    def test_risk_weights_table(self) -> None:
        """Verify that critical actions have higher risk weights."""
        assert RISK_WEIGHTS["send_email"] == 1.0
        assert RISK_WEIGHTS["delete"] == 0.9
        assert RISK_WEIGHTS["scroll"] == 0.5
        assert RISK_WEIGHTS["read"] == 0.4
        assert DEFAULT_RISK_WEIGHT == 0.5


# ---------------------------------------------------------------------------
# Test 2: Enqueue within budget
# ---------------------------------------------------------------------------


class TestEnqueueWithinBudget:
    def test_job_enqueued_successfully(self) -> None:
        queue = VLMFallbackQueue()
        job = _make_job(priority=0.7)
        result = queue.enqueue(job)

        assert result is True
        assert job.status == VLMJobStatus.PENDING
        stats = queue.get_stats()
        assert stats.total_jobs == 1
        assert stats.pending_jobs == 1
        assert stats.jobs_today == 1

    def test_multiple_jobs_ordered_by_priority(self) -> None:
        queue = VLMFallbackQueue()
        low = _make_job(priority=0.1)
        mid = _make_job(priority=0.5)
        high = _make_job(priority=0.9)

        queue.enqueue(low)
        queue.enqueue(high)
        queue.enqueue(mid)

        # Dequeue should return highest priority first
        first = queue.dequeue()
        assert first is not None
        assert first.job_id == high.job_id


# ---------------------------------------------------------------------------
# Test 3: Enqueue exceeds daily limit
# ---------------------------------------------------------------------------


class TestEnqueueExceedsDailyLimit:
    def test_returns_false_after_limit(self) -> None:
        budget = QueueBudget(max_jobs_per_day=5, max_queue_size=1000)
        queue = VLMFallbackQueue(budget=budget)

        # Enqueue 5 jobs (within budget)
        for _ in range(5):
            assert queue.enqueue(_make_job()) is True

        # 6th job should be rejected
        assert queue.enqueue(_make_job()) is False

        stats = queue.get_stats()
        assert stats.jobs_today == 5
        assert stats.total_jobs == 5

    def test_returns_false_at_50_default_limit(self) -> None:
        queue = VLMFallbackQueue()

        for _ in range(50):
            assert queue.enqueue(_make_job()) is True

        assert queue.enqueue(_make_job()) is False


# ---------------------------------------------------------------------------
# Test 4: Dequeue highest priority
# ---------------------------------------------------------------------------


class TestDequeueHighestPriority:
    def test_returns_highest_priority(self) -> None:
        queue = VLMFallbackQueue()

        jobs = [
            _make_job(priority=0.3, job_id="low"),
            _make_job(priority=0.9, job_id="high"),
            _make_job(priority=0.6, job_id="mid"),
        ]
        for j in jobs:
            queue.enqueue(j)

        first = queue.dequeue()
        assert first is not None
        assert first.job_id == "high"
        assert first.status == VLMJobStatus.PROCESSING

    def test_skips_processing_jobs(self) -> None:
        queue = VLMFallbackQueue()

        high = _make_job(priority=0.9, job_id="high")
        low = _make_job(priority=0.3, job_id="low")
        queue.enqueue(high)
        queue.enqueue(low)

        # Dequeue the high-priority job (now PROCESSING)
        first = queue.dequeue()
        assert first is not None
        assert first.job_id == "high"

        # Next dequeue should get the low-priority one
        second = queue.dequeue()
        assert second is not None
        assert second.job_id == "low"

    def test_returns_none_when_empty(self) -> None:
        queue = VLMFallbackQueue()
        assert queue.dequeue() is None

    def test_returns_none_when_all_processing(self) -> None:
        queue = VLMFallbackQueue()
        queue.enqueue(_make_job(priority=0.5))
        queue.dequeue()  # now processing
        assert queue.dequeue() is None


# ---------------------------------------------------------------------------
# Test 5: Expire stale jobs
# ---------------------------------------------------------------------------


class TestExpireStaleJobs:
    def test_jobs_past_ttl_marked_expired(self) -> None:
        queue = VLMFallbackQueue()

        # Job that expired 1 hour ago
        past = datetime.now(timezone.utc) - timedelta(days=8)
        ttl = datetime.now(timezone.utc) - timedelta(hours=1)
        expired_job = _make_job(priority=0.5, job_id="expired", created_at=past, ttl_expires_at=ttl)
        queue.enqueue(expired_job)

        # Job with future TTL
        future_ttl = datetime.now(timezone.utc) + timedelta(days=7)
        fresh_job = _make_job(priority=0.5, job_id="fresh", ttl_expires_at=future_ttl)
        queue.enqueue(fresh_job)

        expired_count = queue.expire_stale_jobs()
        assert expired_count == 1

        # Verify the expired job has correct status
        for j in queue._jobs:
            if j.job_id == "expired":
                assert j.status == VLMJobStatus.EXPIRED
            elif j.job_id == "fresh":
                assert j.status == VLMJobStatus.PENDING

    def test_does_not_expire_processing_jobs(self) -> None:
        """Processing jobs should not be expired even if past TTL."""
        queue = VLMFallbackQueue()

        past_ttl = datetime.now(timezone.utc) - timedelta(hours=1)
        job = _make_job(priority=0.5, ttl_expires_at=past_ttl)
        queue.enqueue(job)

        # Move to processing
        queue.dequeue()

        expired_count = queue.expire_stale_jobs()
        assert expired_count == 0

    def test_no_jobs_to_expire(self) -> None:
        queue = VLMFallbackQueue()
        assert queue.expire_stale_jobs() == 0


# ---------------------------------------------------------------------------
# Test 6: Backpressure drops lowest priority
# ---------------------------------------------------------------------------


class TestBackpressureDropsLowest:
    def test_drops_lowest_when_over_max_size(self) -> None:
        budget = QueueBudget(max_queue_size=5, max_jobs_per_day=1000)
        queue = VLMFallbackQueue(budget=budget)

        # Enqueue 5 jobs with ascending priority
        for i in range(5):
            queue.enqueue(_make_job(priority=0.1 * (i + 1)))

        stats = queue.get_stats()
        assert stats.pending_jobs == 5

        # Enqueue a 6th job with high priority — should trigger backpressure
        high = _make_job(priority=0.9, job_id="high-priority")
        queue.enqueue(high)

        # One lowest-priority job should have been dropped
        stats = queue.get_stats()
        assert stats.pending_jobs == 5  # max_queue_size
        assert stats.dropped_count == 1

        # The dropped job should be the lowest priority one
        dropped = [j for j in queue._jobs if j.status == VLMJobStatus.DROPPED]
        assert len(dropped) == 1
        assert dropped[0].priority_score == pytest.approx(0.1, abs=0.01)

    def test_no_drop_when_within_budget(self) -> None:
        budget = QueueBudget(max_queue_size=10, max_jobs_per_day=100)
        queue = VLMFallbackQueue(budget=budget)

        for i in range(5):
            queue.enqueue(_make_job(priority=0.1 * (i + 1)))

        dropped = queue.enforce_backpressure()
        assert dropped == 0
        assert queue._dropped_count == 0


# ---------------------------------------------------------------------------
# Test 7: can_dispatch budget check
# ---------------------------------------------------------------------------


class TestCanDispatchBudgetCheck:
    def test_can_dispatch_within_budget(self) -> None:
        queue = VLMFallbackQueue()
        assert queue.can_dispatch() is True

    def test_cannot_dispatch_after_daily_limit(self) -> None:
        budget = QueueBudget(max_jobs_per_day=3)
        queue = VLMFallbackQueue(budget=budget)

        for _ in range(3):
            queue.enqueue(_make_job())

        assert queue.can_dispatch() is False

    def test_cannot_dispatch_after_compute_budget(self) -> None:
        budget = QueueBudget(max_compute_minutes_per_day=5.0, max_jobs_per_day=100)
        queue = VLMFallbackQueue(budget=budget)

        job = _make_job(job_id="j1")
        queue.enqueue(job)
        queue.dequeue()
        queue.record_completion("j1", compute_minutes=5.0, result={"ok": True})

        assert queue.can_dispatch() is False


# ---------------------------------------------------------------------------
# Test 8: Record completion
# ---------------------------------------------------------------------------


class TestRecordCompletion:
    def test_stats_updated_after_completion(self) -> None:
        queue = VLMFallbackQueue()

        job = _make_job(priority=0.7, job_id="complete-me")
        queue.enqueue(job)
        queue.dequeue()

        queue.record_completion("complete-me", compute_minutes=2.5, result={"intent": "click"})

        # Verify job status
        for j in queue._jobs:
            if j.job_id == "complete-me":
                assert j.status == VLMJobStatus.COMPLETED
                assert j.result == {"intent": "click"}

        stats = queue.get_stats()
        assert stats.compute_minutes_today == pytest.approx(2.5)
        assert stats.pending_jobs == 0

    def test_record_completion_unknown_job_raises(self) -> None:
        queue = VLMFallbackQueue()
        with pytest.raises(KeyError, match="VLM job not found"):
            queue.record_completion("nonexistent", 1.0, {})


# ---------------------------------------------------------------------------
# Test 9: Compute minutes budget
# ---------------------------------------------------------------------------


class TestComputeMinutesBudget:
    def test_stops_enqueue_after_compute_budget_exhausted(self) -> None:
        budget = QueueBudget(max_compute_minutes_per_day=10.0, max_jobs_per_day=100)
        queue = VLMFallbackQueue(budget=budget)

        # Complete jobs consuming 10 minutes total
        for i in range(5):
            job = _make_job(job_id=f"j{i}")
            queue.enqueue(job)
            queue.dequeue()
            queue.record_completion(f"j{i}", compute_minutes=2.0, result={})

        stats = queue.get_stats()
        assert stats.compute_minutes_today == pytest.approx(10.0)

        # Next enqueue should be rejected
        assert queue.enqueue(_make_job()) is False
        assert queue.can_dispatch() is False

    def test_cumulative_compute_tracking(self) -> None:
        queue = VLMFallbackQueue()

        for i in range(3):
            job = _make_job(job_id=f"j{i}")
            queue.enqueue(job)
            queue.dequeue()
            queue.record_completion(f"j{i}", compute_minutes=1.5, result={})

        stats = queue.get_stats()
        assert stats.compute_minutes_today == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Test 10: Daily counter reset
# ---------------------------------------------------------------------------


class TestDailyCounterReset:
    def test_counters_reset_at_midnight(self) -> None:
        queue = VLMFallbackQueue()

        # Enqueue some jobs
        for _ in range(3):
            queue.enqueue(_make_job())

        # Record some compute time
        job = _make_job(job_id="j-compute")
        queue.enqueue(job)
        queue.dequeue()
        queue.record_completion("j-compute", compute_minutes=5.0, result={})

        assert queue._jobs_dispatched_today == 4
        assert queue._compute_minutes_today == pytest.approx(5.0)

        # Simulate date change by setting _today to yesterday
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        queue._today = yesterday

        # Any public method should trigger the reset
        stats = queue.get_stats()
        assert stats.jobs_today == 0
        assert stats.compute_minutes_today == pytest.approx(0.0)

    def test_dropped_count_resets_at_midnight(self) -> None:
        budget = QueueBudget(max_queue_size=1)
        queue = VLMFallbackQueue(budget=budget)

        # Fill queue, then cause a drop
        queue.enqueue(_make_job(priority=0.9))
        queue.enqueue(_make_job(priority=0.1))  # drops the lower-priority one
        assert queue._dropped_count == 1

        # Simulate day change
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        queue._today = yesterday

        stats = queue.get_stats()
        assert stats.dropped_count == 0, "dropped_count should reset daily"

    def test_can_dispatch_after_reset(self) -> None:
        budget = QueueBudget(max_jobs_per_day=2)
        queue = VLMFallbackQueue(budget=budget)

        queue.enqueue(_make_job())
        queue.enqueue(_make_job())
        assert queue.can_dispatch() is False

        # Simulate day change
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        queue._today = yesterday

        assert queue.can_dispatch() is True


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestTTLAutoSet:
    def test_ttl_auto_set_on_enqueue(self) -> None:
        budget = QueueBudget(job_ttl_days=7)
        queue = VLMFallbackQueue(budget=budget)

        job = _make_job(ttl_expires_at=None)
        queue.enqueue(job)

        assert job.ttl_expires_at is not None
        expected = job.created_at + timedelta(days=7)
        # Allow 1 second tolerance
        delta = abs((job.ttl_expires_at - expected).total_seconds())
        assert delta < 1.0

    def test_explicit_ttl_preserved(self) -> None:
        queue = VLMFallbackQueue()
        custom_ttl = datetime(2099, 1, 1, tzinfo=timezone.utc)
        job = _make_job(ttl_expires_at=custom_ttl)
        queue.enqueue(job)
        assert job.ttl_expires_at == custom_ttl


class TestQueueStatsSnapshot:
    def test_stats_reflect_current_state(self) -> None:
        queue = VLMFallbackQueue()

        queue.enqueue(_make_job(priority=0.8))
        queue.enqueue(_make_job(priority=0.3))

        stats = queue.get_stats()
        assert stats.total_jobs == 2
        assert stats.pending_jobs == 2
        assert stats.jobs_today == 2
        assert stats.compute_minutes_today == 0.0
        assert stats.dropped_count == 0

    def test_stats_after_dequeue_and_complete(self) -> None:
        queue = VLMFallbackQueue()

        job = _make_job(priority=0.5, job_id="stat-job")
        queue.enqueue(job)
        queue.dequeue()
        queue.record_completion("stat-job", compute_minutes=3.0, result={"done": True})

        stats = queue.get_stats()
        assert stats.total_jobs == 1
        assert stats.pending_jobs == 0
        assert stats.compute_minutes_today == pytest.approx(3.0)


class TestVLMJobStatus:
    def test_status_enum_values(self) -> None:
        assert VLMJobStatus.PENDING.value == "pending"
        assert VLMJobStatus.PROCESSING.value == "processing"
        assert VLMJobStatus.COMPLETED.value == "completed"
        assert VLMJobStatus.FAILED.value == "failed"
        assert VLMJobStatus.EXPIRED.value == "expired"
        assert VLMJobStatus.DROPPED.value == "dropped"

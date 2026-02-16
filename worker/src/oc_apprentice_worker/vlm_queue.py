"""VLM Fallback Queue — priority queue with budgets, backpressure, TTL expiry.

Implements section 9.4 of the OpenMimic spec.  When the local semantic
translator has low confidence in a step interpretation, it enqueues a
VLM (Vision Language Model) job for richer analysis.

Priority scoring:
    priority = (1 - confidence_score) * risk_weight * recency_weight
    where:
      confidence_score = current confidence [0.0-1.0]
      risk_weight = action importance (0.5-1.0), from RISK_WEIGHTS table
      recency_weight = exp(-hours_since_capture / 24)  -- decays over 24h

Budget constraints:
    - max_jobs_per_day: 50
    - max_queue_size: 500
    - job_ttl_days: 7
    - max_compute_minutes_per_day: 20

Backpressure:
    When the queue exceeds ``max_queue_size``, the lowest priority pending
    jobs are dropped until the queue is within budget.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class VLMJobStatus(Enum):
    """Lifecycle states for a VLM analysis job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DROPPED = "dropped"


@dataclass
class VLMJob:
    """A single VLM analysis request queued for processing.

    Fields
    ------
    job_id:
        Unique identifier for this job.
    event_id:
        The raw event that triggered low-confidence interpretation.
    episode_id:
        The episode this event belongs to.
    semantic_step_index:
        Index of the step within the episode that needs VLM help.
    confidence_score:
        The confidence score that was too low, triggering VLM fallback.
    priority_score:
        Computed priority for queue ordering (higher = more urgent).
    screenshot_id:
        Optional reference to a screenshot artifact for VLM input.
    dom_snapshot_id:
        Optional reference to a DOM snapshot artifact.
    query:
        The question/prompt to send to the VLM.
    created_at:
        When this job was created.
    ttl_expires_at:
        When this job expires if not processed.
    status:
        Current lifecycle state.
    result:
        VLM analysis result (populated on completion).
    """

    job_id: str
    event_id: str
    episode_id: str
    semantic_step_index: int
    confidence_score: float
    priority_score: float
    screenshot_id: str | None = None
    dom_snapshot_id: str | None = None
    query: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_expires_at: datetime | None = None
    status: VLMJobStatus = VLMJobStatus.PENDING
    result: dict | None = None


# ---------------------------------------------------------------------------
# Risk weights for common action types
# ---------------------------------------------------------------------------

RISK_WEIGHTS: dict[str, float] = {
    "send_email": 1.0,
    "send_message": 0.9,
    "delete": 0.9,
    "submit_form": 0.8,
    "save": 0.7,
    "open_file": 0.7,
    "navigate": 0.6,
    "click": 0.6,
    "select": 0.5,
    "scroll": 0.5,
    "read": 0.4,
    "copy": 0.4,
    "paste": 0.5,
}

# Default risk weight for unknown action types
DEFAULT_RISK_WEIGHT: float = 0.5


@dataclass
class QueueBudget:
    """Budget constraints for the VLM fallback queue.

    All limits are per-day (UTC midnight reset) unless otherwise noted.
    """

    max_jobs_per_day: int = 50
    max_queue_size: int = 500
    job_ttl_days: int = 7
    max_compute_minutes_per_day: float = 20.0


@dataclass
class QueueStats:
    """Snapshot of current queue state for monitoring."""

    total_jobs: int
    pending_jobs: int
    jobs_today: int
    compute_minutes_today: float
    dropped_count: int


class VLMFallbackQueue:
    """Priority queue for VLM analysis jobs with budget enforcement.

    Jobs are ordered by priority score (highest first).  The queue enforces
    daily job limits, compute-time budgets, TTL expiry, and backpressure
    (dropping lowest-priority jobs when the queue is full).

    Parameters
    ----------
    budget:
        Budget constraints.  Defaults to ``QueueBudget()`` with standard limits.
    """

    def __init__(self, budget: QueueBudget | None = None) -> None:
        self.budget = budget or QueueBudget()
        self._jobs: list[VLMJob] = []
        self._jobs_dispatched_today: int = 0
        self._compute_minutes_today: float = 0.0
        self._dropped_count: int = 0
        self._today: datetime = datetime.now(timezone.utc).date()
        self._daily_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Priority calculation
    # ------------------------------------------------------------------

    def compute_priority(
        self,
        confidence: float,
        intent: str,
        created_at: datetime,
    ) -> float:
        """Calculate priority score for a VLM job.

        Formula:
            priority = (1 - confidence) * risk_weight * recency_weight

        Parameters
        ----------
        confidence:
            Current confidence score [0.0, 1.0].
        intent:
            The action type (e.g. "click", "send_email") used to look up
            the risk weight.
        created_at:
            When the original event was captured, used for recency decay.

        Returns
        -------
        float
            Priority score.  Higher values are more urgent.
        """
        risk_weight = RISK_WEIGHTS.get(intent, DEFAULT_RISK_WEIGHT)
        # Clamp hours_elapsed to [0, 720] (30 days) to limit impact of clock skew
        hours_elapsed = min(
            max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0),
            720.0,
        )
        recency_weight = math.exp(-hours_elapsed / 24.0)
        return (1.0 - confidence) * risk_weight * recency_weight

    # ------------------------------------------------------------------
    # Core queue operations
    # ------------------------------------------------------------------

    def enqueue(self, job: VLMJob) -> bool:
        """Add a job to the queue.

        Sets the job's TTL if not already set.  Returns ``False`` if the
        daily dispatch limit has been reached or the queue is at capacity
        after backpressure (should not happen in normal operation since
        backpressure drops lower-priority jobs).

        Parameters
        ----------
        job:
            The VLM job to enqueue.

        Returns
        -------
        bool
            True if the job was successfully enqueued, False if budget
            constraints prevented it.
        """
        self._reset_daily_counters_if_needed()

        # Check daily job dispatch limit
        if self._jobs_dispatched_today >= self.budget.max_jobs_per_day:
            logger.warning(
                "VLM queue daily limit reached (%d/%d), rejecting job %s",
                self._jobs_dispatched_today,
                self.budget.max_jobs_per_day,
                job.job_id,
            )
            return False

        # Check compute minutes budget
        if self._compute_minutes_today >= self.budget.max_compute_minutes_per_day:
            logger.warning(
                "VLM queue compute budget exhausted (%.1f/%.1f min), rejecting job %s",
                self._compute_minutes_today,
                self.budget.max_compute_minutes_per_day,
                job.job_id,
            )
            return False

        # Set TTL if not already set
        if job.ttl_expires_at is None:
            job.ttl_expires_at = job.created_at + timedelta(days=self.budget.job_ttl_days)

        # Ensure the job is pending
        job.status = VLMJobStatus.PENDING

        # Insert in priority order (highest first)
        inserted = False
        for i, existing in enumerate(self._jobs):
            if job.priority_score > existing.priority_score:
                self._jobs.insert(i, job)
                inserted = True
                break
        if not inserted:
            self._jobs.append(job)

        self._jobs_dispatched_today += 1

        # Apply backpressure if needed
        self.enforce_backpressure()

        logger.debug(
            "Enqueued VLM job %s (priority=%.4f, queue_size=%d)",
            job.job_id,
            job.priority_score,
            len(self._jobs),
        )
        return True

    def dequeue(self) -> VLMJob | None:
        """Get the highest priority pending job.

        The job is marked as ``PROCESSING`` and remains in the queue until
        ``record_completion()`` is called.

        Returns
        -------
        VLMJob or None
            The highest-priority pending job, or None if no pending jobs.
        """
        self._reset_daily_counters_if_needed()

        for job in self._jobs:
            if job.status == VLMJobStatus.PENDING:
                job.status = VLMJobStatus.PROCESSING
                return job

        return None

    def expire_stale_jobs(self) -> int:
        """Expire jobs past their TTL.

        Returns
        -------
        int
            Number of jobs expired.
        """
        now = datetime.now(timezone.utc)
        expired_count = 0

        for job in self._jobs:
            if job.status != VLMJobStatus.PENDING:
                continue
            if job.ttl_expires_at is not None and now >= job.ttl_expires_at:
                job.status = VLMJobStatus.EXPIRED
                expired_count += 1
                logger.debug("Expired VLM job %s (TTL reached)", job.job_id)

        return expired_count

    def enforce_backpressure(self) -> int:
        """Drop lowest-priority pending jobs if queue exceeds max size.

        Only pending jobs are eligible for dropping.  Jobs that are
        processing, completed, failed, expired, or already dropped are
        not affected.

        Returns
        -------
        int
            Number of jobs dropped.
        """
        # Count all non-terminal jobs (pending + processing)
        active_count = sum(
            1
            for j in self._jobs
            if j.status in (VLMJobStatus.PENDING, VLMJobStatus.PROCESSING)
        )

        if active_count <= self.budget.max_queue_size:
            return 0

        excess = active_count - self.budget.max_queue_size
        dropped = 0

        # Drop from the end (lowest priority, since list is sorted by
        # descending priority) — only drop PENDING jobs
        for job in reversed(self._jobs):
            if dropped >= excess:
                break
            if job.status == VLMJobStatus.PENDING:
                job.status = VLMJobStatus.DROPPED
                self._dropped_count += 1
                dropped += 1
                logger.debug(
                    "Dropped VLM job %s (backpressure, priority=%.4f)",
                    job.job_id,
                    job.priority_score,
                )

        return dropped

    def can_dispatch(self) -> bool:
        """Check if daily budget allows another job dispatch.

        Returns True if both the daily job count and compute minutes
        are within budget.
        """
        self._reset_daily_counters_if_needed()
        if self._jobs_dispatched_today >= self.budget.max_jobs_per_day:
            return False
        if self._compute_minutes_today >= self.budget.max_compute_minutes_per_day:
            return False
        return True

    def record_completion(
        self,
        job_id: str,
        compute_minutes: float,
        result: dict,
    ) -> None:
        """Record that a VLM job completed successfully.

        Updates the job's status to COMPLETED, stores the result, and
        adds the compute time to today's running total.

        Parameters
        ----------
        job_id:
            The job to mark as completed.
        compute_minutes:
            Wall-clock minutes consumed by VLM processing.
        result:
            The VLM analysis result dictionary.

        Raises
        ------
        KeyError
            If no job with the given ``job_id`` exists in the queue.
        """
        self._reset_daily_counters_if_needed()

        for job in self._jobs:
            if job.job_id == job_id:
                job.status = VLMJobStatus.COMPLETED
                job.result = result
                self._compute_minutes_today += compute_minutes
                logger.debug(
                    "VLM job %s completed (%.2f min, total today: %.2f min)",
                    job_id,
                    compute_minutes,
                    self._compute_minutes_today,
                )
                return

        raise KeyError(f"VLM job not found: {job_id}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> QueueStats:
        """Get current queue statistics.

        Returns
        -------
        QueueStats
            Snapshot of queue state.
        """
        self._reset_daily_counters_if_needed()

        total = len(self._jobs)
        pending = sum(1 for j in self._jobs if j.status == VLMJobStatus.PENDING)

        return QueueStats(
            total_jobs=total,
            pending_jobs=pending,
            jobs_today=self._jobs_dispatched_today,
            compute_minutes_today=self._compute_minutes_today,
            dropped_count=self._dropped_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_daily_counters_if_needed(self) -> None:
        """Reset counters at midnight UTC.

        Called at the start of every public method to ensure counters
        reflect the current day.  Thread-safe via _daily_lock.
        """
        today = datetime.now(timezone.utc).date()
        if today != self._today:
            with self._daily_lock:
                # Double-check after acquiring lock
                if today != self._today:
                    logger.info(
                        "VLM queue daily reset: %s -> %s (dispatched=%d, compute=%.1f min)",
                        self._today,
                        today,
                        self._jobs_dispatched_today,
                        self._compute_minutes_today,
                    )
                    self._jobs_dispatched_today = 0
                    self._compute_minutes_today = 0.0
                    self._today = today

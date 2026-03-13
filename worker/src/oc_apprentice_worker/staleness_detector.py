"""Staleness detector — flags procedures that may be outdated.

Checks procedures for signs of staleness:
- Not observed in a long time
- Step failures in recent observations
- Significant confidence drift
- URL or app changes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Thresholds
_NEEDS_REVIEW_DAYS = 30
_STALE_DAYS = 60
_CONFIDENCE_DRIFT_THRESHOLD = 0.15


@dataclass
class StalenessSignal:
    """A single signal indicating potential staleness."""

    type: str  # "last_observed_old", "step_failure", "new_step", "url_changed", "confidence_drift"
    detail: str
    first_seen: str


@dataclass
class StalenessReport:
    """Staleness assessment for a single procedure."""

    slug: str
    status: str  # "current", "needs_review", "stale"
    signals: list[StalenessSignal] = field(default_factory=list)
    confidence_trend: list[float] = field(default_factory=list)
    recommended_action: str = "none"  # "none", "review", "re-observe", "archive"


class StalenessDetector:
    """Detect stale procedures in the knowledge base."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def check_all(self) -> list[StalenessReport]:
        """Check all procedures for staleness.

        Returns a list of StalenessReports, one per procedure.
        """
        procedures = self._kb.list_procedures()
        reports = []
        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if slug:
                reports.append(self.check_procedure(slug, proc))
        return reports

    def check_procedure(
        self, slug: str, proc: dict | None = None
    ) -> StalenessReport:
        """Check a single procedure for staleness.

        Args:
            slug: Procedure slug.
            proc: Pre-loaded procedure dict, or None to load from KB.
        """
        if proc is None:
            proc = self._kb.get_procedure(slug)
        if proc is None:
            return StalenessReport(
                slug=slug,
                status="stale",
                signals=[StalenessSignal(
                    type="not_found",
                    detail="Procedure not found in knowledge base",
                    first_seen=datetime.now(timezone.utc).isoformat(),
                )],
                recommended_action="archive",
            )

        signals: list[StalenessSignal] = []
        now = datetime.now(timezone.utc)

        # Check last_observed age
        staleness = proc.get("staleness", {})
        last_observed = staleness.get("last_observed")
        if last_observed:
            try:
                lo_dt = datetime.fromisoformat(
                    last_observed.replace("Z", "+00:00")
                )
                days_old = (now - lo_dt).days
                if days_old >= _STALE_DAYS:
                    signals.append(StalenessSignal(
                        type="last_observed_old",
                        detail=f"Last observed {days_old} days ago",
                        first_seen=now.isoformat(),
                    ))
                elif days_old >= _NEEDS_REVIEW_DAYS:
                    signals.append(StalenessSignal(
                        type="last_observed_old",
                        detail=f"Last observed {days_old} days ago",
                        first_seen=now.isoformat(),
                    ))
            except (ValueError, TypeError):
                pass
        else:
            signals.append(StalenessSignal(
                type="last_observed_old",
                detail="No last_observed timestamp recorded",
                first_seen=now.isoformat(),
            ))

        # Check confidence drift
        confidence_trend = staleness.get("confidence_trend", [])
        if len(confidence_trend) >= 3:
            recent = confidence_trend[-3:]
            if recent[0] - recent[-1] >= _CONFIDENCE_DRIFT_THRESHOLD:
                signals.append(StalenessSignal(
                    type="confidence_drift",
                    detail=(
                        f"Confidence dropped from {recent[0]:.2f} to "
                        f"{recent[-1]:.2f} over last {len(recent)} observations"
                    ),
                    first_seen=now.isoformat(),
                ))

        # Check drift_signals from staleness
        drift_signals = staleness.get("drift_signals", [])
        for ds in drift_signals:
            if isinstance(ds, dict):
                signals.append(StalenessSignal(
                    type=ds.get("type", "drift_signal"),
                    detail=ds.get("detail", "Unknown drift signal"),
                    first_seen=ds.get("first_seen", now.isoformat()),
                ))

        # Check evidence for contradictions
        evidence = proc.get("evidence", {})
        contradictions = evidence.get("contradictions", [])
        if contradictions:
            signals.append(StalenessSignal(
                type="step_failure",
                detail=f"{len(contradictions)} step contradiction(s) detected",
                first_seen=now.isoformat(),
            ))

        # Determine status and recommended action
        status, action = self._assess(signals, last_observed, now)

        return StalenessReport(
            slug=slug,
            status=status,
            signals=signals,
            confidence_trend=confidence_trend,
            recommended_action=action,
        )

    def _assess(
        self,
        signals: list[StalenessSignal],
        last_observed: str | None,
        now: datetime,
    ) -> tuple[str, str]:
        """Determine status and recommended action from signals."""
        if not signals:
            return "current", "none"

        has_old = any(s.type == "last_observed_old" for s in signals)
        has_drift = any(s.type == "confidence_drift" for s in signals)
        has_failure = any(s.type == "step_failure" for s in signals)

        # Check age
        days_old = 0
        if last_observed:
            try:
                lo_dt = datetime.fromisoformat(
                    last_observed.replace("Z", "+00:00")
                )
                days_old = (now - lo_dt).days
            except (ValueError, TypeError):
                pass

        if days_old >= _STALE_DAYS:
            return "stale", "archive"

        if days_old >= _NEEDS_REVIEW_DAYS or has_drift or has_failure:
            if has_failure and has_drift:
                return "stale", "re-observe"
            return "needs_review", "review"

        return "current", "none"

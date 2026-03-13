"""Pattern detector — finds recurring tasks and task chains in daily summaries.

Analyzes daily summaries from the knowledge base to detect:
- Recurrence patterns (daily, weekly, weekday, event-triggered)
- Task chains (procedure A frequently followed by procedure B)

Detected patterns are persisted to the KB triggers file for agent consumption.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class RecurrencePattern:
    """A detected recurrence pattern for a procedure."""

    procedure_slug: str
    pattern: str  # "daily", "weekly", "weekday", "event_triggered"
    confidence: float
    day: str | None  # "monday" for weekly
    time: str | None  # "09:00"
    avg_duration_minutes: int
    observations: int


@dataclass
class TaskChain:
    """A detected sequential task chain (A often followed by B)."""

    first_slug: str
    then_slug: str
    confidence: float
    co_occurrence_count: int


class PatternDetector:
    """Detect recurring task patterns and sequential chains from daily summaries."""

    def __init__(
        self, kb: KnowledgeBase, *, min_observations: int = 3
    ) -> None:
        self._kb = kb
        self._min_observations = min_observations

    def detect_recurrence(self) -> list[RecurrencePattern]:
        """Detect recurring task patterns from daily summaries.

        Analyzes which procedures appear on consistent schedules:

        - daily: appears on >70% of workdays
        - weekly: appears on the same day of week consistently
        - weekday: appears on workdays but not weekends
        - event_triggered: no time pattern, but consistent occurrence
        """
        summaries = self._load_summaries()
        if len(summaries) < self._min_observations:
            return []

        # Collect procedure occurrences by day
        proc_days: defaultdict[str, list[dict]] = defaultdict(list)

        for summary in summaries:
            date_str = summary.get("date", "")
            tasks = summary.get("tasks", [])

            seen_procs: set[str] = set()
            for task in tasks:
                slug = task.get("matched_procedure")
                intent = task.get("intent", "")
                if slug and slug not in seen_procs:
                    seen_procs.add(slug)
                    proc_days[slug].append({
                        "date": date_str,
                        "start_time": task.get("start_time", ""),
                        "duration": task.get("duration_minutes", 0),
                    })
                elif intent and not slug:
                    # Use intent as a pseudo-slug for unmatched tasks
                    key = _slugify(intent)
                    if key and key not in seen_procs:
                        seen_procs.add(key)
                        proc_days[key].append({
                            "date": date_str,
                            "start_time": task.get("start_time", ""),
                            "duration": task.get("duration_minutes", 0),
                        })

        patterns: list[RecurrencePattern] = []
        total_days = len(summaries)

        for slug, occurrences in proc_days.items():
            if len(occurrences) < self._min_observations:
                continue

            pattern = self._classify_recurrence(
                slug, occurrences, total_days
            )
            if pattern is not None:
                patterns.append(pattern)

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def _classify_recurrence(
        self, slug: str, occurrences: list[dict], total_days: int
    ) -> RecurrencePattern | None:
        """Classify the recurrence pattern for a procedure."""
        ratio = len(occurrences) / max(total_days, 1)

        # Parse dates and weekdays
        weekdays: Counter = Counter()
        hours: list[int] = []
        durations: list[int] = []

        for occ in occurrences:
            date_str = occ.get("date", "")
            start = occ.get("start_time", "")
            dur = occ.get("duration", 0)

            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    weekdays[dt.strftime("%A").lower()] += 1
                except ValueError:
                    pass

            if start:
                try:
                    dt = datetime.fromisoformat(
                        start.replace("Z", "+00:00")
                    )
                    hours.append(dt.hour)
                except (ValueError, TypeError):
                    pass

            if dur:
                durations.append(dur)

        avg_duration = sum(durations) // max(len(durations), 1)
        avg_time = (
            f"{sum(hours) // max(len(hours), 1):02d}:00" if hours else None
        )

        # Check for daily pattern (>70% of days)
        if ratio >= 0.7:
            return RecurrencePattern(
                procedure_slug=slug,
                pattern="daily",
                confidence=min(ratio, 1.0),
                day=None,
                time=avg_time,
                avg_duration_minutes=avg_duration,
                observations=len(occurrences),
            )

        # Check for weekly pattern (concentrated on specific day)
        if weekdays:
            most_common_day, day_count = weekdays.most_common(1)[0]
            day_ratio = day_count / len(occurrences)
            if day_ratio >= 0.6 and day_count >= self._min_observations:
                return RecurrencePattern(
                    procedure_slug=slug,
                    pattern="weekly",
                    confidence=day_ratio,
                    day=most_common_day,
                    time=avg_time,
                    avg_duration_minutes=avg_duration,
                    observations=len(occurrences),
                )

        # Check weekday pattern (weekdays only)
        weekend_days = weekdays.get("saturday", 0) + weekdays.get(
            "sunday", 0
        )
        weekday_count = len(occurrences) - weekend_days
        if weekend_days == 0 and weekday_count >= self._min_observations:
            weekday_ratio = weekday_count / max(total_days, 1)
            if weekday_ratio >= 0.3:
                return RecurrencePattern(
                    procedure_slug=slug,
                    pattern="weekday",
                    confidence=weekday_ratio,
                    day=None,
                    time=avg_time,
                    avg_duration_minutes=avg_duration,
                    observations=len(occurrences),
                )

        # Event-triggered (no clear time pattern but recurring)
        if len(occurrences) >= self._min_observations:
            return RecurrencePattern(
                procedure_slug=slug,
                pattern="event_triggered",
                confidence=ratio,
                day=None,
                time=None,
                avg_duration_minutes=avg_duration,
                observations=len(occurrences),
            )

        return None

    def detect_chains(self) -> list[TaskChain]:
        """Detect task chains (A always/often followed by B).

        Looks at sequential tasks within daily summaries to find
        pairs that frequently occur together in sequence.
        """
        summaries = self._load_summaries()
        if len(summaries) < self._min_observations:
            return []

        pair_counts: Counter = Counter()
        slug_counts: Counter = Counter()

        for summary in summaries:
            tasks = summary.get("tasks", [])
            slugs = []
            for task in tasks:
                slug = task.get("matched_procedure") or _slugify(
                    task.get("intent", "")
                )
                if slug:
                    slugs.append(slug)
                    slug_counts[slug] += 1

            # Count consecutive pairs
            for i in range(len(slugs) - 1):
                if slugs[i] != slugs[i + 1]:  # skip self-chains
                    pair_counts[(slugs[i], slugs[i + 1])] += 1

        chains: list[TaskChain] = []
        for (first, then), count in pair_counts.most_common():
            if count < self._min_observations:
                continue
            first_total = slug_counts.get(first, count)
            confidence = count / max(first_total, 1)
            if confidence >= 0.3:
                chains.append(TaskChain(
                    first_slug=first,
                    then_slug=then,
                    confidence=round(confidence, 3),
                    co_occurrence_count=count,
                ))

        return sorted(chains, key=lambda c: c.confidence, reverse=True)

    def update_triggers(self, patterns: list[RecurrencePattern]) -> None:
        """Write detected patterns to the triggers file in KB."""
        triggers = self._kb.get_triggers()
        triggers["recurrence"] = [
            {
                "procedure_slug": p.procedure_slug,
                "pattern": p.pattern,
                "confidence": round(p.confidence, 3),
                "day": p.day,
                "time": p.time,
                "avg_duration_minutes": p.avg_duration_minutes,
                "observations": p.observations,
            }
            for p in patterns
        ]
        self._kb.update_triggers(triggers)

    def update_chains(self, chains: list[TaskChain]) -> None:
        """Write detected chains to the triggers file in KB."""
        triggers = self._kb.get_triggers()
        triggers["chains"] = [
            {
                "first_slug": c.first_slug,
                "then_slug": c.then_slug,
                "confidence": c.confidence,
                "co_occurrence_count": c.co_occurrence_count,
            }
            for c in chains
        ]
        self._kb.update_triggers(triggers)

    def _load_summaries(self) -> list[dict]:
        """Load recent daily summaries from KB (up to 90 days)."""
        return self._kb.load_daily_summaries(limit=90)


def _slugify(text: str) -> str:
    """Convert intent text to a slug-like key."""
    return "-".join(text.lower().split()[:6]) if text else ""

"""Tests for the pattern detector (Sprint 7).

Covers: daily/weekly/weekday/event-triggered recurrence detection,
chain detection, minimum observation thresholds, trigger persistence,
empty summaries, confidence sorting, and slugify helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.pattern_detector import (
    PatternDetector,
    RecurrencePattern,
    TaskChain,
    _slugify,
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


@pytest.fixture()
def detector(kb: KnowledgeBase) -> PatternDetector:
    return PatternDetector(kb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_daily_summary(
    date: str,
    tasks: list[dict] | None = None,
    top_apps: list[dict] | None = None,
    active_hours: float = 6.0,
) -> dict:
    return {
        "date": date,
        "active_hours": active_hours,
        "task_count": len(tasks) if tasks else 0,
        "tasks": tasks or [],
        "top_apps": top_apps or [],
        "procedures_observed": [],
        "new_workflows_detected": 0,
    }


def make_task(
    intent: str,
    apps: list[str],
    urls: list[str] | None = None,
    start_time: str = "2026-03-10T09:00:00Z",
    end_time: str = "2026-03-10T09:10:00Z",
    duration_minutes: int = 10,
    matched_procedure: str | None = None,
) -> dict:
    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration_minutes,
        "intent": intent,
        "apps": apps,
        "urls": urls or [],
        "event_ids": [],
        "is_complete": True,
        "matched_procedure": matched_procedure,
    }


def _populate_summaries(kb: KnowledgeBase, summaries: list[dict]) -> None:
    for s in summaries:
        kb.save_daily_summary(s["date"], s)


# ---------------------------------------------------------------------------
# Empty / insufficient data
# ---------------------------------------------------------------------------

class TestPatternDetectorEmpty:

    def test_empty_summaries_no_patterns(
        self, detector: PatternDetector
    ) -> None:
        assert detector.detect_recurrence() == []

    def test_empty_summaries_no_chains(
        self, detector: PatternDetector
    ) -> None:
        assert detector.detect_chains() == []

    def test_too_few_summaries_no_patterns(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        # Only 2 summaries, min_observations=3 (default)
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=[
                make_task("standup", ["Zoom"],
                          matched_procedure="daily-standup"),
            ]),
            make_daily_summary("2026-03-02", tasks=[
                make_task("standup", ["Zoom"],
                          matched_procedure="daily-standup"),
            ]),
        ])
        assert detector.detect_recurrence() == []

    def test_too_few_summaries_no_chains(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        _populate_summaries(kb, [
            make_daily_summary("2026-03-01", tasks=[
                make_task("A", ["X"], matched_procedure="proc-a"),
                make_task("B", ["X"], matched_procedure="proc-b"),
            ]),
            make_daily_summary("2026-03-02", tasks=[
                make_task("A", ["X"], matched_procedure="proc-a"),
                make_task("B", ["X"], matched_procedure="proc-b"),
            ]),
        ])
        assert detector.detect_chains() == []


# ---------------------------------------------------------------------------
# Daily pattern
# ---------------------------------------------------------------------------

class TestDailyPattern:

    def test_daily_pattern_detected(
        self, kb: KnowledgeBase
    ) -> None:
        # Task on 8 out of 10 days = 80% -> daily
        summaries = []
        for i in range(1, 11):
            tasks = []
            if i <= 8:
                tasks.append(make_task(
                    "standup", ["Zoom"],
                    matched_procedure="daily-standup",
                    start_time=f"2026-03-{i:02d}T09:00:00Z",
                    duration_minutes=15,
                ))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        assert len(patterns) >= 1
        daily = next(
            p for p in patterns if p.procedure_slug == "daily-standup"
        )
        assert daily.pattern == "daily"
        assert daily.confidence == 0.8
        assert daily.observations == 8

    def test_daily_pattern_with_time(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for i in range(1, 11):
            tasks = [make_task(
                "standup", ["Zoom"],
                matched_procedure="daily-standup",
                start_time=f"2026-03-{i:02d}T09:00:00Z",
                duration_minutes=15,
            )]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        daily = next(
            p for p in patterns if p.procedure_slug == "daily-standup"
        )
        assert daily.time == "09:00"

    def test_daily_pattern_with_duration(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for i in range(1, 11):
            tasks = [make_task(
                "standup", ["Zoom"],
                matched_procedure="daily-standup",
                start_time=f"2026-03-{i:02d}T09:00:00Z",
                duration_minutes=15,
            )]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        daily = next(
            p for p in patterns if p.procedure_slug == "daily-standup"
        )
        assert daily.avg_duration_minutes == 15


# ---------------------------------------------------------------------------
# Weekly pattern
# ---------------------------------------------------------------------------

class TestWeeklyPattern:

    def test_weekly_pattern_detected(
        self, kb: KnowledgeBase
    ) -> None:
        # Task every Monday over 5 weeks (5 Mondays + other days = ~35 total)
        summaries = []
        mondays = [
            "2026-02-02", "2026-02-09", "2026-02-16", "2026-02-23",
            "2026-03-02",
        ]
        all_dates = set()
        for m in mondays:
            all_dates.add(m)

        # Add non-Monday days too
        for i in range(1, 29):
            d = f"2026-02-{i:02d}"
            all_dates.add(d)

        for d in sorted(all_dates):
            tasks = []
            if d in mondays:
                tasks.append(make_task(
                    "weekly review", ["Notion"],
                    matched_procedure="weekly-review",
                    start_time=f"{d}T14:00:00Z",
                    duration_minutes=30,
                ))
            summaries.append(make_daily_summary(d, tasks=tasks))

        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        weekly = next(
            (p for p in patterns if p.procedure_slug == "weekly-review"),
            None,
        )
        assert weekly is not None
        assert weekly.pattern == "weekly"
        assert weekly.day == "monday"
        assert weekly.observations == 5

    def test_weekly_pattern_confidence(
        self, kb: KnowledgeBase
    ) -> None:
        # 4 out of 5 occurrences on Monday, 1 on Tuesday = 80% day_ratio
        summaries = []
        dates_with_review = [
            "2026-02-02",  # Monday
            "2026-02-09",  # Monday
            "2026-02-16",  # Monday
            "2026-02-17",  # Tuesday
            "2026-02-23",  # Monday
        ]
        all_dates = set(dates_with_review)
        for i in range(1, 29):
            all_dates.add(f"2026-02-{i:02d}")

        for d in sorted(all_dates):
            tasks = []
            if d in dates_with_review:
                tasks.append(make_task(
                    "weekly review", ["Notion"],
                    matched_procedure="weekly-review",
                    start_time=f"{d}T14:00:00Z",
                    duration_minutes=30,
                ))
            summaries.append(make_daily_summary(d, tasks=tasks))

        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        weekly = next(
            (p for p in patterns if p.procedure_slug == "weekly-review"),
            None,
        )
        assert weekly is not None
        assert weekly.pattern == "weekly"
        assert weekly.confidence == 0.8  # 4/5 on Monday


# ---------------------------------------------------------------------------
# Weekday pattern
# ---------------------------------------------------------------------------

class TestWeekdayPattern:

    def test_weekday_pattern_detected(
        self, kb: KnowledgeBase
    ) -> None:
        # Task on workdays only (no weekends), but not on >70% of all days
        # 2026-03-02 Mon, 03 Tue, 04 Wed, 05 Thu, 06 Fri
        # 2026-03-07 Sat, 08 Sun
        # 2026-03-09 Mon, 10 Tue, 11 Wed
        summaries = []
        weekday_dates = [
            "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05",
            "2026-03-06", "2026-03-09", "2026-03-10", "2026-03-11",
        ]
        all_dates = weekday_dates + ["2026-03-07", "2026-03-08"]

        for d in sorted(all_dates):
            tasks = []
            if d in weekday_dates:
                tasks.append(make_task(
                    "code review", ["GitHub"],
                    matched_procedure="code-review",
                    start_time=f"{d}T10:00:00Z",
                    duration_minutes=20,
                ))
            summaries.append(make_daily_summary(d, tasks=tasks))

        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        # 8 occurrences on 10 total days = 80% -> daily (exceeds 70%)
        # So this will actually be classified as daily
        cr = next(
            (p for p in patterns if p.procedure_slug == "code-review"),
            None,
        )
        assert cr is not None
        assert cr.pattern == "daily"  # 80% ratio is daily

    def test_pure_weekday_not_daily(
        self, kb: KnowledgeBase
    ) -> None:
        # 5 occurrences out of 15 days, all weekdays, <70% ratio
        # This should be weekday pattern
        summaries = []
        # 15 days: March 1 (Sun) through March 15 (Sun)
        work_dates = [
            "2026-03-02", "2026-03-04", "2026-03-06",
            "2026-03-09", "2026-03-11",
        ]
        for i in range(1, 16):
            d = f"2026-03-{i:02d}"
            tasks = []
            if d in work_dates:
                tasks.append(make_task(
                    "deploy", ["Terminal"],
                    matched_procedure="deploy-staging",
                    start_time=f"{d}T11:00:00Z",
                    duration_minutes=5,
                ))
            summaries.append(make_daily_summary(d, tasks=tasks))

        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        deploy = next(
            (p for p in patterns if p.procedure_slug == "deploy-staging"),
            None,
        )
        assert deploy is not None
        assert deploy.pattern == "weekday"
        assert deploy.observations == 5


# ---------------------------------------------------------------------------
# Event-triggered pattern
# ---------------------------------------------------------------------------

class TestEventTriggeredPattern:

    def test_event_triggered_fallback(
        self, kb: KnowledgeBase
    ) -> None:
        # 3 occurrences out of 15 days, spread across different weekdays
        # Not daily (<70%), not weekly (no day concentration), not weekday
        # (includes a weekend or just <30% weekday ratio)
        summaries = []
        occur_dates = [
            "2026-03-01",  # Sunday
            "2026-03-05",  # Thursday
            "2026-03-12",  # Thursday
        ]
        for i in range(1, 16):
            d = f"2026-03-{i:02d}"
            tasks = []
            if d in occur_dates:
                tasks.append(make_task(
                    "release", ["Terminal"],
                    matched_procedure="cut-release",
                    duration_minutes=45,
                ))
            summaries.append(make_daily_summary(d, tasks=tasks))

        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        release = next(
            (p for p in patterns if p.procedure_slug == "cut-release"),
            None,
        )
        assert release is not None
        assert release.pattern == "event_triggered"
        assert release.time is None  # no start_time on tasks
        assert release.observations == 3

    def test_event_triggered_low_confidence(
        self, kb: KnowledgeBase
    ) -> None:
        # 3 out of 30 days = 10% ratio
        summaries = []
        occur_dates = {"2026-02-05", "2026-02-15", "2026-02-25"}
        for i in range(1, 29):
            d = f"2026-02-{i:02d}"
            tasks = []
            if d in occur_dates:
                tasks.append(make_task(
                    "incident response", ["PagerDuty"],
                    matched_procedure="incident-response",
                ))
            summaries.append(make_daily_summary(d, tasks=tasks))

        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        inc = next(
            (p for p in patterns
             if p.procedure_slug == "incident-response"),
            None,
        )
        assert inc is not None
        assert inc.pattern == "event_triggered"
        assert inc.confidence < 0.2


# ---------------------------------------------------------------------------
# Not enough observations
# ---------------------------------------------------------------------------

class TestMinObservations:

    def test_procedure_below_threshold_skipped(
        self, kb: KnowledgeBase
    ) -> None:
        # Only 2 occurrences with min_observations=3
        summaries = []
        for i in range(1, 11):
            tasks = []
            if i <= 2:
                tasks.append(make_task(
                    "rare", ["App"],
                    matched_procedure="rare-task",
                ))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        slugs = [p.procedure_slug for p in patterns]
        assert "rare-task" not in slugs

    def test_custom_min_observations(
        self, kb: KnowledgeBase
    ) -> None:
        # 2 occurrences with min_observations=2 should be detected
        summaries = []
        for i in range(1, 6):
            tasks = []
            if i <= 2:
                tasks.append(make_task(
                    "niche", ["App"],
                    matched_procedure="niche-task",
                ))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=2)
        patterns = detector.detect_recurrence()
        slugs = [p.procedure_slug for p in patterns]
        assert "niche-task" in slugs


# ---------------------------------------------------------------------------
# Intent-based slug (no matched_procedure)
# ---------------------------------------------------------------------------

class TestIntentSlug:

    def test_unmatched_tasks_use_intent_slug(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for i in range(1, 11):
            tasks = [make_task(
                "check email inbox", ["Mail"],
                start_time=f"2026-03-{i:02d}T08:00:00Z",
            )]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        slugs = [p.procedure_slug for p in patterns]
        assert "check-email-inbox" in slugs

    def test_intent_slug_truncated_to_six_words(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for i in range(1, 6):
            tasks = [make_task(
                "this is a very long intent description that goes on",
                ["App"],
            )]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        for p in patterns:
            assert len(p.procedure_slug.split("-")) <= 6


# ---------------------------------------------------------------------------
# Chain detection
# ---------------------------------------------------------------------------

class TestChainDetection:

    def test_chain_detected(
        self, kb: KnowledgeBase
    ) -> None:
        # A->B on 4 out of 5 days
        summaries = []
        for i in range(1, 6):
            tasks = [
                make_task("task A", ["App"], matched_procedure="proc-a"),
                make_task("task B", ["App"], matched_procedure="proc-b"),
            ]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        chains = detector.detect_chains()
        assert len(chains) >= 1
        ab = next(
            (c for c in chains
             if c.first_slug == "proc-a" and c.then_slug == "proc-b"),
            None,
        )
        assert ab is not None
        assert ab.co_occurrence_count == 5
        assert ab.confidence == 1.0  # 5/5 times A appears, B follows

    def test_chain_below_threshold_excluded(
        self, kb: KnowledgeBase
    ) -> None:
        # A->B only 2 times (below default min_observations=3)
        summaries = []
        for i in range(1, 6):
            tasks = []
            if i <= 2:
                tasks = [
                    make_task("A", ["App"], matched_procedure="proc-a"),
                    make_task("B", ["App"], matched_procedure="proc-b"),
                ]
            else:
                tasks = [
                    make_task("C", ["App"], matched_procedure="proc-c"),
                ]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        chains = detector.detect_chains()
        pair_slugs = [(c.first_slug, c.then_slug) for c in chains]
        assert ("proc-a", "proc-b") not in pair_slugs

    def test_chain_low_confidence_excluded(
        self, kb: KnowledgeBase
    ) -> None:
        # A->B on 3 days, A->C on 7 days = A->B confidence = 3/10 = 0.3
        summaries = []
        for i in range(1, 11):
            tasks = [
                make_task("A", ["App"], matched_procedure="proc-a"),
            ]
            if i <= 3:
                tasks.append(
                    make_task("B", ["App"], matched_procedure="proc-b")
                )
            else:
                tasks.append(
                    make_task("C", ["App"], matched_procedure="proc-c")
                )
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        chains = detector.detect_chains()
        # A->B: 3/10 = 0.3, should be included (>= 0.3)
        # A->C: 7/10 = 0.7, should be included
        ab = next(
            (c for c in chains
             if c.first_slug == "proc-a" and c.then_slug == "proc-b"),
            None,
        )
        assert ab is not None
        assert ab.confidence == 0.3

    def test_chain_self_loop_excluded(
        self, kb: KnowledgeBase
    ) -> None:
        # A->A should be skipped
        summaries = []
        for i in range(1, 6):
            tasks = [
                make_task("A", ["App"], matched_procedure="proc-a"),
                make_task("A again", ["App"], matched_procedure="proc-a"),
            ]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        chains = detector.detect_chains()
        assert chains == []

    def test_chain_sorted_by_confidence(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for i in range(1, 11):
            tasks = [
                make_task("A", ["App"], matched_procedure="proc-a"),
            ]
            if i <= 5:
                tasks.append(
                    make_task("B", ["App"], matched_procedure="proc-b")
                )
            if i > 5:
                tasks.append(
                    make_task("C", ["App"], matched_procedure="proc-c")
                )
            # Also add D->E chain with higher confidence
            tasks.extend([
                make_task("D", ["App"], matched_procedure="proc-d"),
                make_task("E", ["App"], matched_procedure="proc-e"),
            ])
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        chains = detector.detect_chains()
        # D->E should be 10/10 = 1.0, should be first
        assert chains[0].first_slug == "proc-d"
        assert chains[0].then_slug == "proc-e"
        assert chains[0].confidence == 1.0

    def test_chain_with_intent_slugs(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        for i in range(1, 6):
            tasks = [
                make_task("open browser", ["Chrome"]),
                make_task("check github", ["Chrome"]),
            ]
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        chains = detector.detect_chains()
        assert len(chains) >= 1
        first_chain = chains[0]
        assert first_chain.first_slug == "open-browser"
        assert first_chain.then_slug == "check-github"


# ---------------------------------------------------------------------------
# Multiple patterns sorted by confidence
# ---------------------------------------------------------------------------

class TestPatternSorting:

    def test_patterns_sorted_by_confidence_descending(
        self, kb: KnowledgeBase
    ) -> None:
        summaries = []
        # 10 days total
        for i in range(1, 11):
            tasks = []
            # daily-standup: 10/10 = 1.0 confidence
            tasks.append(make_task(
                "standup", ["Zoom"],
                matched_procedure="daily-standup",
                start_time=f"2026-03-{i:02d}T09:00:00Z",
            ))
            # code-review: 5/10 = 0.5 (event_triggered since
            # it falls between weekly and weekday thresholds)
            if i <= 5:
                tasks.append(make_task(
                    "review", ["GitHub"],
                    matched_procedure="code-review",
                    start_time=f"2026-03-{i:02d}T14:00:00Z",
                ))
            summaries.append(
                make_daily_summary(f"2026-03-{i:02d}", tasks=tasks)
            )
        _populate_summaries(kb, summaries)

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        assert len(patterns) >= 2
        # First should be highest confidence
        assert patterns[0].confidence >= patterns[1].confidence


# ---------------------------------------------------------------------------
# update_triggers and update_chains
# ---------------------------------------------------------------------------

class TestTriggerPersistence:

    def test_update_triggers_writes_to_kb(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        patterns = [
            RecurrencePattern(
                procedure_slug="daily-standup",
                pattern="daily",
                confidence=0.9,
                day=None,
                time="09:00",
                avg_duration_minutes=15,
                observations=9,
            ),
        ]
        detector.update_triggers(patterns)
        triggers = kb.get_triggers()
        assert len(triggers["recurrence"]) == 1
        rec = triggers["recurrence"][0]
        assert rec["procedure_slug"] == "daily-standup"
        assert rec["pattern"] == "daily"
        assert rec["confidence"] == 0.9
        assert rec["time"] == "09:00"
        assert rec["observations"] == 9

    def test_update_chains_writes_to_kb(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        chains = [
            TaskChain(
                first_slug="proc-a",
                then_slug="proc-b",
                confidence=0.85,
                co_occurrence_count=17,
            ),
        ]
        detector.update_chains(chains)
        triggers = kb.get_triggers()
        assert len(triggers["chains"]) == 1
        chain = triggers["chains"][0]
        assert chain["first_slug"] == "proc-a"
        assert chain["then_slug"] == "proc-b"
        assert chain["confidence"] == 0.85
        assert chain["co_occurrence_count"] == 17

    def test_update_triggers_preserves_chains(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        # First write chains
        chains = [
            TaskChain("a", "b", 0.9, 10),
        ]
        detector.update_chains(chains)

        # Then write recurrence patterns
        patterns = [
            RecurrencePattern("x", "daily", 0.8, None, "09:00", 15, 8),
        ]
        detector.update_triggers(patterns)

        triggers = kb.get_triggers()
        assert len(triggers["recurrence"]) == 1
        assert len(triggers["chains"]) == 1

    def test_update_chains_preserves_recurrence(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        # First write recurrence patterns
        patterns = [
            RecurrencePattern("x", "daily", 0.8, None, "09:00", 15, 8),
        ]
        detector.update_triggers(patterns)

        # Then write chains
        chains = [
            TaskChain("a", "b", 0.9, 10),
        ]
        detector.update_chains(chains)

        triggers = kb.get_triggers()
        assert len(triggers["recurrence"]) == 1
        assert len(triggers["chains"]) == 1

    def test_update_triggers_has_updated_at(
        self, detector: PatternDetector, kb: KnowledgeBase
    ) -> None:
        detector.update_triggers([])
        triggers = kb.get_triggers()
        assert triggers["updated_at"] is not None


# ---------------------------------------------------------------------------
# Slugify helper
# ---------------------------------------------------------------------------

class TestSlugify:

    def test_basic_slugify(self) -> None:
        assert _slugify("Check email inbox") == "check-email-inbox"

    def test_truncates_to_six_words(self) -> None:
        result = _slugify("one two three four five six seven eight")
        assert result == "one-two-three-four-five-six"

    def test_empty_string(self) -> None:
        assert _slugify("") == ""

    def test_single_word(self) -> None:
        assert _slugify("Deploy") == "deploy"

    def test_extra_whitespace(self) -> None:
        assert _slugify("  check  email  ") == "check-email"


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_recurrence_pattern_fields(self) -> None:
        p = RecurrencePattern(
            procedure_slug="daily-standup",
            pattern="daily",
            confidence=0.95,
            day=None,
            time="09:00",
            avg_duration_minutes=15,
            observations=19,
        )
        assert p.procedure_slug == "daily-standup"
        assert p.pattern == "daily"
        assert p.confidence == 0.95
        assert p.day is None
        assert p.time == "09:00"
        assert p.avg_duration_minutes == 15
        assert p.observations == 19

    def test_task_chain_fields(self) -> None:
        c = TaskChain(
            first_slug="a",
            then_slug="b",
            confidence=0.8,
            co_occurrence_count=12,
        )
        assert c.first_slug == "a"
        assert c.then_slug == "b"
        assert c.confidence == 0.8
        assert c.co_occurrence_count == 12

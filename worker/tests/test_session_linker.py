"""Tests for the SessionLinker module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.session_linker import (
    STOP_WORDS,
    LinkedTask,
    SessionLinker,
    _SIMILARITY_THRESHOLD,
    _STALE_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


def _save_daily(
    kb: KnowledgeBase,
    date: str,
    tasks: list[dict],
) -> None:
    """Helper: save a daily summary with given tasks."""
    kb.save_daily_summary(date, {"tasks": tasks})


# ---------------------------------------------------------------------------
# Tests: intent normalization
# ---------------------------------------------------------------------------


class TestNormalizeIntent:
    def test_basic_normalization(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("Deploy the API server")
        assert "deploy" in result
        assert "api" in result
        assert "server" in result
        # Stop words removed
        assert "the" not in result

    def test_punctuation_removed(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("Deploy! the API, server.")
        assert "deploy" in result
        assert "!" not in result
        assert "," not in result
        assert "." not in result

    def test_case_insensitive(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        r1 = linker._normalize_intent("Deploy API")
        r2 = linker._normalize_intent("deploy api")
        assert r1 == r2

    def test_stop_words_removed(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("a test for the project")
        # Only "test" and "project" should remain (single-char tokens removed)
        assert "test" in result
        assert "project" in result
        assert "a" not in result.split()
        assert "the" not in result.split()
        assert "for" not in result.split()

    def test_single_char_tokens_removed(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("I do X on Y")
        # Single char tokens 'I', 'X', 'Y' removed; stop word 'on' removed;
        # 'do' is kept (2 chars, not a stop word)
        assert result == "do"


# ---------------------------------------------------------------------------
# Tests: intent similarity
# ---------------------------------------------------------------------------


class TestIntentSimilarity:
    def test_identical_intents(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim = linker._intent_similarity("Deploy API server", "Deploy API server")
        assert sim == 1.0

    def test_completely_different(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim = linker._intent_similarity("Deploy API server", "Cook dinner tonight")
        assert sim == 0.0

    def test_partial_overlap(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim = linker._intent_similarity("Deploy API server", "Deploy API client")
        # {"deploy", "api", "server"} vs {"deploy", "api", "client"}
        # intersection=2, union=4 -> 0.5
        assert sim == pytest.approx(0.5, abs=0.01)

    def test_above_threshold(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim = linker._intent_similarity(
            "Update project documentation", "Update documentation project"
        )
        assert sim >= _SIMILARITY_THRESHOLD

    def test_both_empty_returns_one(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim = linker._intent_similarity("the", "a")  # All stop words
        assert sim == 1.0

    def test_one_empty_returns_zero(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim = linker._intent_similarity("the", "deploy server")
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Tests: link creation from daily summaries
# ---------------------------------------------------------------------------


class TestLinkCreation:
    def test_same_intent_across_days_creates_link(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].intent == "Deploy API server"
        assert links[0].total_duration_minutes == 50
        assert links[0].span_days >= 2
        assert len(links[0].sessions) == 2

    def test_similar_intent_creates_link(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server to production", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server staging", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        # "deploy api server production" vs "deploy api server staging"
        # intersection=3, union=5 -> 0.6 >= 0.4
        assert len(links) == 1

    def test_unrelated_tasks_no_link(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Cook dinner tonight", "duration_minutes": 60, "apps": ["Safari"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 0

    def test_same_day_tasks_not_linked(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
            {"intent": "Deploy API server again", "duration_minutes": 15, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        # Same day — should not create a cross-day link
        assert len(links) == 0

    def test_matched_procedure_links_tasks(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "something unique alpha", "duration_minutes": 10, "apps": ["Xcode"],
             "matched_procedure": "deploy-api"},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "totally different beta", "duration_minutes": 20, "apps": ["Xcode"],
             "matched_procedure": "deploy-api"},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].matched_procedure == "deploy-api"

    def test_multiple_distinct_links(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 45, "apps": ["VS Code"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 30, "apps": ["VS Code"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 2


# ---------------------------------------------------------------------------
# Tests: active vs stale status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_recent_link_is_active(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].status == "active"

    def test_old_link_becomes_stale(self, kb: KnowledgeBase) -> None:
        old_date1 = (datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS + 5)).strftime("%Y-%m-%d")
        old_date2 = (datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS + 3)).strftime("%Y-%m-%d")
        _save_daily(kb, old_date1, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, old_date2, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].status == "stale"


# ---------------------------------------------------------------------------
# Tests: mark completed
# ---------------------------------------------------------------------------


class TestMarkCompleted:
    def test_mark_existing_completed(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1

        result = linker.mark_completed(links[0].link_id)
        assert result is True
        assert links[0].status == "completed"
        assert len(linker.get_active_links()) == 0

    def test_mark_nonexistent_returns_false(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        assert linker.mark_completed("nonexistent-id") is False


# ---------------------------------------------------------------------------
# Tests: get_active_links
# ---------------------------------------------------------------------------


class TestGetActiveLinks:
    def test_only_active_returned(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 45, "apps": ["VS Code"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 30, "apps": ["VS Code"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 2

        linker.mark_completed(links[0].link_id)
        active = linker.get_active_links()
        assert len(active) == 1


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_links_survive_reload(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        linker1 = SessionLinker(kb)
        linker1.analyze_daily_summaries()
        assert len(linker1._links) == 1

        linker2 = SessionLinker(kb)
        assert len(linker2._links) == 1
        assert linker2._links[0].intent == "Deploy API server"

    def test_completed_state_persists(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        linker1 = SessionLinker(kb)
        links = linker1.analyze_daily_summaries()
        linker1.mark_completed(links[0].link_id)

        linker2 = SessionLinker(kb)
        assert linker2._links[0].status == "completed"

    def test_empty_kb_loads_no_links(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        assert linker._links == []
        assert linker.get_active_links() == []


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_intent_skipped(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 0

    def test_no_daily_summaries(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert links == []

    def test_description_fallback_used(self, kb: KnowledgeBase) -> None:
        """When 'intent' is missing, fall back to 'description'."""
        _save_daily(kb, "2025-03-01", [
            {"description": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"description": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1

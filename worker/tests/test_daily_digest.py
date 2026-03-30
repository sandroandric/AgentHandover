"""Tests for the DigestGenerator module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.daily_digest import (
    DailyDigest,
    DigestGenerator,
    DigestHighlight,
    DigestSection,
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


def _save_daily(kb: KnowledgeBase, date: str, data: dict) -> None:
    """Helper: save a daily summary."""
    kb.save_daily_summary(date, data)


def _save_trust_suggestions(kb: KnowledgeBase, suggestions: list[dict]) -> None:
    """Helper: save trust suggestions file."""
    path = kb.root / "observations" / "trust_suggestions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"suggestions": suggestions}, f)


# ---------------------------------------------------------------------------
# Tests: generate with daily summary
# ---------------------------------------------------------------------------


class TestGenerateWithSummary:
    def test_basic_digest(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 60, "apps": ["Terminal"]},
                {"intent": "Write docs", "duration_minutes": 30, "apps": ["VS Code"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.date == "2025-03-01"
        assert digest.tasks_completed == 2
        assert digest.active_hours == pytest.approx(1.5, abs=0.01)

    def test_procedures_counted(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 60, "apps": ["Terminal"],
                 "matched_procedure": "deploy-api"},
                {"intent": "Deploy API again", "duration_minutes": 30, "apps": ["Terminal"],
                 "matched_procedure": "deploy-api"},
                {"intent": "Write docs", "duration_minutes": 45, "apps": ["VS Code"],
                 "matched_procedure": "write-docs"},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.procedures_observed == 2

    def test_procedures_from_summary_level_field(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [],
            "procedures_observed": ["deploy-api", "write-docs"],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.procedures_observed == 2

    def test_procedures_from_dict_format(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [],
            "procedures_observed": [{"slug": "deploy-api"}, {"slug": "send-email"}],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.procedures_observed == 2


# ---------------------------------------------------------------------------
# Tests: generate without daily summary
# ---------------------------------------------------------------------------


class TestGenerateWithoutSummary:
    def test_no_summary_returns_zero_stats(self, kb: KnowledgeBase) -> None:
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.tasks_completed == 0
        assert digest.active_hours == 0.0
        assert digest.procedures_observed == 0
        assert digest.highlights == []
        assert digest.date == "2025-03-01"

    def test_default_date_is_today(self, kb: KnowledgeBase) -> None:
        gen = DigestGenerator(kb)
        digest = gen.generate()
        expected = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert digest.date == expected


# ---------------------------------------------------------------------------
# Tests: summary text generation
# ---------------------------------------------------------------------------


class TestSummaryText:
    def test_basic_summary_text(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 120, "apps": ["Terminal"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        text = gen.generate_summary_text(digest)
        assert "2.0 hours" in text
        assert "1 tasks" in text

    def test_summary_text_with_procedures(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy", "duration_minutes": 60, "apps": ["Terminal"],
                 "matched_procedure": "deploy-api"},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        text = gen.generate_summary_text(digest)
        assert "1 procedure observed" in text

    def test_summary_text_plural_procedures(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy", "duration_minutes": 60, "apps": ["Terminal"],
                 "matched_procedure": "deploy-api"},
                {"intent": "Email", "duration_minutes": 30, "apps": ["Mail"],
                 "matched_procedure": "send-email"},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        text = gen.generate_summary_text(digest)
        assert "2 procedures observed" in text

    def test_summary_no_procedures(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Browse web", "duration_minutes": 30, "apps": ["Safari"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        text = gen.generate_summary_text(digest)
        assert "procedure" not in text


# ---------------------------------------------------------------------------
# Tests: save / load digest
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 60, "apps": ["Terminal"],
                 "matched_procedure": "deploy-api"},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        gen.save_digest(digest)

        loaded = gen.get_digest("2025-03-01")
        assert loaded is not None
        assert loaded.date == digest.date
        assert loaded.tasks_completed == digest.tasks_completed
        assert loaded.active_hours == digest.active_hours
        assert loaded.procedures_observed == digest.procedures_observed
        assert loaded.summary == digest.summary

    def test_load_nonexistent_returns_none(self, kb: KnowledgeBase) -> None:
        gen = DigestGenerator(kb)
        assert gen.get_digest("2099-01-01") is None

    def test_highlights_roundtrip(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [],
            "new_procedures": ["deploy-api"],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        gen.save_digest(digest)

        loaded = gen.get_digest("2025-03-01")
        assert loaded is not None
        assert len(loaded.highlights) == len(digest.highlights)
        if loaded.highlights:
            assert loaded.highlights[0].type == digest.highlights[0].type
            assert loaded.highlights[0].title == digest.highlights[0].title

    def test_sections_roundtrip(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 60, "apps": ["Terminal"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        gen.save_digest(digest)

        loaded = gen.get_digest("2025-03-01")
        assert loaded is not None
        assert len(loaded.sections) == len(digest.sections)

    def test_stats_roundtrip(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 60, "apps": ["Terminal", "VS Code"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        gen.save_digest(digest)

        loaded = gen.get_digest("2025-03-01")
        assert loaded is not None
        assert loaded.stats["total_tasks"] == 1
        assert loaded.stats["apps_used"] == 2


# ---------------------------------------------------------------------------
# Tests: stats calculation
# ---------------------------------------------------------------------------


class TestStats:
    def test_app_list_and_count(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy", "duration_minutes": 60, "apps": ["Terminal", "VS Code"]},
                {"intent": "Browse", "duration_minutes": 30, "apps": ["Safari", "Terminal"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.stats["apps_used"] == 3
        assert set(digest.stats["app_list"]) == {"Terminal", "VS Code", "Safari"}

    def test_zero_tasks_stats(self, kb: KnowledgeBase) -> None:
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.stats["total_tasks"] == 0
        assert digest.stats["apps_used"] == 0
        assert digest.stats["active_hours"] == 0.0


# ---------------------------------------------------------------------------
# Tests: highlights extraction
# ---------------------------------------------------------------------------


class TestHighlights:
    def test_new_procedure_highlight(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [],
            "new_procedures": ["deploy-api"],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        new_proc_highlights = [h for h in digest.highlights if h.type == "new_procedure"]
        assert len(new_proc_highlights) == 1
        assert "deploy-api" in new_proc_highlights[0].title
        assert new_proc_highlights[0].priority == 1

    def test_trust_suggestion_highlight(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {"tasks": []})
        _save_trust_suggestions(kb, [
            {
                "procedure_slug": "deploy-api",
                "current_level": "observe",
                "suggested_level": "suggest",
                "reason": "Good track record",
                "dismissed": False,
                "accepted": False,
            },
        ])
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        trust_highlights = [h for h in digest.highlights if h.type == "trust_suggestion"]
        assert len(trust_highlights) == 1
        assert trust_highlights[0].priority == 2

    def test_dismissed_suggestion_not_highlighted(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {"tasks": []})
        _save_trust_suggestions(kb, [
            {
                "procedure_slug": "deploy-api",
                "current_level": "observe",
                "suggested_level": "suggest",
                "reason": "Good track record",
                "dismissed": True,
                "accepted": False,
            },
        ])
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        trust_highlights = [h for h in digest.highlights if h.type == "trust_suggestion"]
        assert len(trust_highlights) == 0

    def test_pattern_highlight(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [],
            "patterns_detected": ["morning-deploy"],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        pattern_highlights = [h for h in digest.highlights if h.type == "pattern_detected"]
        assert len(pattern_highlights) == 1

    def test_milestone_highlight_many_tasks(self, kb: KnowledgeBase) -> None:
        tasks = [
            {"intent": f"Task {i}", "duration_minutes": 10, "apps": ["Terminal"]}
            for i in range(10)
        ]
        _save_daily(kb, "2025-03-01", {"tasks": tasks})
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        milestones = [h for h in digest.highlights if h.type == "milestone"]
        assert len(milestones) == 1
        assert milestones[0].priority == 3

    def test_no_milestone_for_few_tasks(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy", "duration_minutes": 60, "apps": ["Terminal"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        milestones = [h for h in digest.highlights if h.type == "milestone"]
        assert len(milestones) == 0


# ---------------------------------------------------------------------------
# Tests: sections
# ---------------------------------------------------------------------------


class TestSections:
    def test_tasks_section_created(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy API", "duration_minutes": 60, "apps": ["Terminal"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        task_sections = [s for s in digest.sections if s.title == "Tasks"]
        assert len(task_sections) == 1
        assert len(task_sections[0].items) == 1
        assert task_sections[0].items[0]["intent"] == "Deploy API"

    def test_apps_section_created(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", {
            "tasks": [
                {"intent": "Deploy", "duration_minutes": 60, "apps": ["Terminal", "VS Code"]},
            ],
        })
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        app_sections = [s for s in digest.sections if s.title == "Applications Used"]
        assert len(app_sections) == 1
        assert len(app_sections[0].items) == 2

    def test_no_sections_for_empty_day(self, kb: KnowledgeBase) -> None:
        gen = DigestGenerator(kb)
        digest = gen.generate("2025-03-01")
        assert digest.sections == []

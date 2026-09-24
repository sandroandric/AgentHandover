"""Tests for the DigestGenerator module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from daemon_db import kb_db
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
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
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


class TestWorkHighlights:
    """Every day with recorded work (a task of a minute or more) says what
    the work was, from the day's own summary, before any Skill or
    recurring workflow exists; a day without work stays empty."""

    def _highlights(self, kb: KnowledgeBase, tasks: list[dict], **summary) -> list[DigestHighlight]:
        _save_daily(kb, "2026-08-31", {"tasks": tasks, **summary})
        return DigestGenerator(kb).generate("2026-08-31").highlights

    def test_a_first_day_names_its_top_tasks_by_time(self, kb: KnowledgeBase) -> None:
        highlights = self._highlights(kb, [
            {"intent": "Reply to email", "duration_minutes": 4, "apps": ["Mail"]},
            {"intent": "Reconcile the Stripe payout", "duration_minutes": 12,
             "apps": ["Google Chrome"]},
            {"intent": "Fill in the payout sheet", "duration_minutes": 9,
             "apps": ["Google Chrome"]},
            {"intent": "Post in Slack", "duration_minutes": 7, "apps": ["Slack"]},
            # A single frame: counted among the tasks, never a top task.
            {"intent": "Glance at Mail", "duration_minutes": 0, "apps": ["Mail"]},
        ], top_apps=[
            {"app": "Google Chrome", "minutes": 21}, {"app": "Slack", "minutes": 7},
            {"app": "Mail", "minutes": 4},
        ])
        top = [(h.title, h.detail) for h in highlights if h.type == "top_task"]
        assert top == [
            ("Top task: Reconcile the Stripe payout", "12 min in Google Chrome."),
            ("Top task: Fill in the payout sheet", "9 min in Google Chrome."),
            ("Top task: Post in Slack", "7 min in Slack."),
        ]
        (overview,) = [h for h in highlights if h.type == "day_overview"]
        assert overview.title == "32 min of work across 5 tasks"
        assert overview.detail == (
            "Most time in Google Chrome (21 min), Slack (7 min), Mail (4 min)."
        )
        assert {h.priority for h in highlights} == {3}

    def test_the_same_task_split_by_an_app_switch_adds_up(self, kb: KnowledgeBase) -> None:
        highlights = self._highlights(kb, [
            {"intent": "Reconcile the payout", "duration_minutes": 5, "apps": ["Google Chrome"]},
            {"intent": "Chat", "duration_minutes": 6, "apps": ["Slack"]},
            {"intent": "Reconcile the payout", "duration_minutes": 3, "apps": ["Numbers"]},
        ])
        top = [(h.title, h.detail) for h in highlights if h.type == "top_task"]
        assert top == [
            ("Top task: Reconcile the payout", "8 min in Google Chrome, Numbers."),
            ("Top task: Chat", "6 min in Slack."),
        ]

    def test_a_two_minute_day_has_a_highlight(self, kb: KnowledgeBase) -> None:
        """No cut-off beyond the daily processor's: the digest's summary
        and Tasks section already report the two minutes."""
        highlights = self._highlights(kb, [
            {"intent": "Check the payout", "duration_minutes": 2, "apps": ["Google Chrome"]},
        ])
        assert [(h.type, h.title) for h in highlights] == [
            ("top_task", "Top task: Check the payout"),
            ("day_overview", "2 min of work across 1 task"),
        ]

    def test_a_task_without_an_intent_is_named_by_its_apps(self, kb: KnowledgeBase) -> None:
        highlights = self._highlights(kb, [
            {"intent": "", "duration_minutes": 3, "apps": ["Terminal"]},
            {"intent": "", "duration_minutes": 1, "apps": []},
        ])
        assert [h.title for h in highlights if h.type == "top_task"] == [
            "Top task: Work in Terminal", "Top task: Unlabelled task",
        ]

    def test_a_day_without_recorded_work_has_none(self, kb: KnowledgeBase) -> None:
        assert self._highlights(kb, []) == []
        # Single frames: tasks of zero minutes are no recorded work.
        assert self._highlights(kb, [
            {"intent": "Glance at Slack", "duration_minutes": 0, "apps": ["Slack"]},
        ]) == []

    def test_other_highlights_come_first(self, kb: KnowledgeBase) -> None:
        highlights = self._highlights(
            kb, [{"intent": "Deploy", "duration_minutes": 20, "apps": ["Terminal"]}],
            new_procedures=["deploy-api"],
        )
        assert [h.type for h in highlights] == ["new_procedure", "top_task", "day_overview"]

    def test_the_overview_counts_the_days_work_sessions(
        self, kb: KnowledgeBase, tmp_db_path: Path,
    ) -> None:
        from agenthandover_worker.db import WorkerDB

        with WorkerDB(tmp_db_path) as db:
            for n, (date, minute) in enumerate([
                ("2026-08-31", 540), ("2026-08-31", 600), ("2026-09-01", 540),
            ]):
                assert db.save_workflow_observation({
                    "obs_id": f"o{n}", "workflow_key": f"k{n}",
                    "started_utc": f"{date}T{minute // 60:02d}:00:00.000Z",
                    "started_local_date": date, "started_local_weekday": 0,
                    "started_local_minute": minute, "duration_s": 600.0,
                    "apps_json": json.dumps(["Slack"]), "hosts_json": "[]",
                    "step_sig_json": "[]", "evidence_json": "{}", "source": "passive",
                    "task_label": "Chat", "event_ids_json": json.dumps([f"o{n}-e1"]),
                })
            _save_daily(kb, "2026-08-31", {
                "tasks": [{"intent": "Chat", "duration_minutes": 20, "apps": ["Slack"]}],
                "top_apps": [{"app": "Slack", "minutes": 20}],
            })
            digest = DigestGenerator(kb, db=db).generate("2026-08-31")
        (overview,) = [h for h in digest.highlights if h.type == "day_overview"]
        assert overview.detail == "2 work sessions; most time in Slack (20 min)."


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


class TestRunQuestions:
    """A question an agent run raised (``skill_improver.feed_ledger``) is
    the Skill's decisions question: once asked as a review card
    (``questions.refresh``, which the daily batch runs before the digest),
    it stays in every digest until the user answers it."""

    NOW = datetime(2025, 3, 2, 8, 0, tzinfo=timezone.utc)
    RUN_ID = "run:export-report:run-1:step_1"

    @pytest.fixture()
    def db(self, tmp_db_path: Path):
        from agenthandover_worker.db import WorkerDB

        with WorkerDB(tmp_db_path) as worker_db:
            yield worker_db

    def _digest(self, kb: KnowledgeBase, db, tmp_path: Path) -> tuple[DailyDigest, list[dict]]:
        from agenthandover_worker import questions

        questions.refresh(db, kb, tmp_path, now=self.NOW)
        digest = DigestGenerator(kb, db=db).generate("2025-03-02")
        return digest, [
            item for s in digest.sections if s.title == "Questions for you"
            for item in s.items if item["question_id"].startswith("run:")
        ]

    def _proc(self, kb: KnowledgeBase, answer: str = "") -> None:
        kb.save_procedure({
            "id": "export-report", "title": "Export the report",
            "steps": [{"step_id": "step_1", "action": "Open the dashboard"}],
            "agent_clarifications": [
                # A post-recording question the user skipped: not a run question.
                {"category": "goal", "question": "Why?", "answer": "", "status": "unanswered"},
                {
                    "category": "decision",
                    "question": "An agent running this Skill did not follow step step_1.",
                    "answer": answer, "status": "answered" if answer else "unanswered",
                    "origin": {"kind": "agent_run", "id": "run-1", "step_id": "step_1"},
                    "asked_at": "2025-03-01T10:00:00+00:00",
                },
            ],
        })

    def test_an_open_run_question_is_listed(self, kb: KnowledgeBase, db, tmp_path: Path) -> None:
        self._proc(kb)
        digest, items = self._digest(kb, db, tmp_path)
        assert items == [{
            "title": "An agent running this Skill did not follow step step_1.",
            "detail": "Export the report: asked after an agent run",
            "question_id": self.RUN_ID, "choices": "",
        }]
        # One questions section; the general decisions question is not in it.
        [section] = [s for s in digest.sections if "question" in s.title.lower()]
        assert "slot:export-report:decisions" not in [i["question_id"] for i in section.items]
        # And it survives the round trip the app reads.
        gen = DigestGenerator(kb, db=db)
        gen.save_digest(digest)
        loaded = gen.get_digest("2025-03-02")
        [loaded_section] = [s for s in loaded.sections if s.title == "Questions for you"]
        assert loaded_section.items == section.items

    def test_an_answered_question_is_not_listed(
        self, kb: KnowledgeBase, db, tmp_path: Path,
    ) -> None:
        self._proc(kb, answer="Retry once, then stop")
        _digest, items = self._digest(kb, db, tmp_path)
        assert items == []

    def test_without_the_worker_db_no_questions_are_listed(self, kb: KnowledgeBase) -> None:
        self._proc(kb)
        digest = DigestGenerator(kb).generate("2025-03-02")
        assert [s for s in digest.sections if s.title == "Questions for you"] == []

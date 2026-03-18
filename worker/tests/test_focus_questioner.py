"""Tests for focus_questioner.py — gap analysis and targeted questions.

Tests cover:
- Question generation with mocked LLM responses
- Question parsing and validation
- Answer merging into procedures by category
- File IPC helpers (write/read questions, pending state)
- Graceful degradation when LLM fails
- Edge cases (empty procedures, no gaps, max questions cap)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.focus_questioner import (
    FocusQuestion,
    FocusQAResult,
    FocusQuestioner,
    clear_focus_qa_files,
    parse_qa_result_from_file,
    read_focus_pending,
    read_focus_questions,
    write_focus_pending,
    write_focus_questions,
    FOCUS_QUESTIONS_FILE,
    FOCUS_PENDING_FILE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_procedure(**overrides) -> dict:
    """Create a sample procedure dict for testing."""
    proc = {
        "title": "Reddit Community Engagement",
        "description": "Browse Reddit and engage with startup community posts",
        "slug": "reddit-community-engagement",
        "strategy": "",
        "steps": [
            {
                "step_id": "step_1",
                "action": "Open browser to Reddit",
                "app": "Chrome",
                "url": "https://reddit.com/r/startups",
            },
            {
                "step_id": "step_2",
                "action": "Browse post list",
                "app": "Chrome",
            },
            {
                "step_id": "step_3",
                "action": "Open interesting post",
                "app": "Chrome",
            },
            {
                "step_id": "step_4",
                "action": "Write comment",
                "app": "Chrome",
            },
        ],
        "apps_involved": ["com.google.Chrome"],
        "variables": [],
        "inputs": [],
    }
    proc.update(overrides)
    return proc


def _make_reasoner(response_value=None, success=True, abstained=False):
    """Create a mock LLMReasoner."""
    from agenthandover_worker.llm_reasoning import ReasoningResult

    reasoner = MagicMock()
    result = ReasoningResult(
        value=response_value,
        success=success,
        abstained=abstained,
        model="qwen3.5:4b",
        prompt_hash="test123",
        elapsed_seconds=1.5,
        generated_at="2026-03-18T10:00:00Z",
    )
    reasoner.reason_json.return_value = result
    return reasoner


# ---------------------------------------------------------------------------
# Tests: Question generation
# ---------------------------------------------------------------------------


class TestGenerateQuestions:

    def test_generates_questions_from_llm_response(self):
        llm_response = {
            "questions": [
                {
                    "question": "Does this workflow require logging into Reddit?",
                    "category": "credentials",
                    "context": "Browser URLs detected at reddit.com but no login step recorded",
                    "default": "No login required",
                },
                {
                    "question": "What is your goal when browsing r/startups?",
                    "category": "strategy",
                    "context": "Strategy field is empty — agent needs to know the intent",
                    "default": "Browse and engage with relevant posts",
                },
            ]
        }
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())

        assert len(questions) == 2
        assert questions[0].category == "credentials"
        assert "Reddit" in questions[0].question
        assert questions[1].category == "strategy"

    def test_empty_questions_when_no_gaps(self):
        llm_response = {"questions": []}
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure(strategy="Clear strategy"))
        assert questions == []

    def test_max_three_questions(self):
        llm_response = {
            "questions": [
                {"question": f"Q{i}?", "category": "strategy", "context": "ctx", "default": "def"}
                for i in range(5)
            ]
        }
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())
        assert len(questions) == 3

    def test_graceful_degradation_on_llm_failure(self):
        reasoner = _make_reasoner(success=False)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())
        assert questions == []

    def test_graceful_degradation_on_abstention(self):
        reasoner = _make_reasoner(abstained=True)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())
        assert questions == []

    def test_open_category_preserved(self):
        """Non-standard categories are kept as-is (not forced to a fixed set)."""
        llm_response = {
            "questions": [
                {
                    "question": "What happens next?",
                    "category": "error_handling",
                    "context": "test",
                    "default": "default",
                }
            ]
        }
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())
        assert len(questions) == 1
        assert questions[0].category == "error_handling"

    def test_empty_question_text_filtered(self):
        llm_response = {
            "questions": [
                {"question": "", "category": "strategy", "context": "ctx", "default": "def"},
                {"question": "Valid question?", "category": "strategy", "context": "ctx", "default": "def"},
            ]
        }
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())
        assert len(questions) == 1
        assert questions[0].question == "Valid question?"

    def test_empty_default_gets_placeholder(self):
        llm_response = {
            "questions": [
                {"question": "What?", "category": "strategy", "context": "ctx", "default": ""},
            ]
        }
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questions = questioner.generate_questions(_make_procedure())
        assert questions[0].default == "Not specified"

    def test_llm_called_with_correct_caller(self):
        llm_response = {"questions": []}
        reasoner = _make_reasoner(response_value=llm_response)
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questioner.generate_questions(_make_procedure())

        call_kwargs = reasoner.reason_json.call_args
        assert call_kwargs.kwargs.get("caller") == "focus_questioner"


# ---------------------------------------------------------------------------
# Tests: Answer merging
# ---------------------------------------------------------------------------


class TestMergeAnswers:

    def _make_qa(self, questions, answers):
        return FocusQAResult(questions=questions, answers=answers)

    def test_merge_credentials_answer(self):
        questions = [
            FocusQuestion(
                question="Requires login?",
                category="credentials",
                context="ctx",
                default="No login required",
            )
        ]
        qa = self._make_qa(questions, {0: "Reddit account (personal)"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)

        assert "environment" in result
        assert len(result["environment"]["accounts"]) == 1
        assert "Reddit" in result["environment"]["accounts"][0]["service"]
        assert len(result["inputs"]) == 1
        assert result["inputs"][0]["credential"] is True

    def test_merge_credentials_no_login(self):
        questions = [
            FocusQuestion(
                question="Requires login?",
                category="credentials",
                context="ctx",
                default="No login required",
            )
        ]
        qa = self._make_qa(questions, {0: "No login required"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)

        assert result.get("environment", {}).get("accounts", []) == []

    def test_merge_strategy_answer(self):
        questions = [
            FocusQuestion(
                question="What's the goal?",
                category="strategy",
                context="ctx",
                default="Browse posts",
            )
        ]
        qa = self._make_qa(questions, {0: "Find and engage with SaaS founders"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert "Find and engage with SaaS founders" in result["strategy"]

    def test_merge_strategy_appends_to_existing(self):
        proc = _make_procedure(strategy="Existing strategy.")
        questions = [
            FocusQuestion(
                question="More detail?",
                category="strategy",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "Focus on B2B posts"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(proc, qa)
        assert "Existing strategy." in result["strategy"]
        assert "Focus on B2B posts" in result["strategy"]

    def test_merge_decision_answer(self):
        questions = [
            FocusQuestion(
                question="How do you decide?",
                category="decision",
                context="Branch detected",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "Skip posts older than 24h"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert len(result["branches"]) == 1
        assert result["branches"][0]["condition"] == "Skip posts older than 24h"
        assert result["branches"][0]["source"] == "user_clarification"

    def test_merge_verification_answer(self):
        questions = [
            FocusQuestion(
                question="What does done look like?",
                category="verification",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "At least 3 comments posted"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert len(result["expected_outcomes"]) == 1
        assert "3 comments" in result["expected_outcomes"][0]["description"]

    def test_merge_scope_recurring(self):
        questions = [
            FocusQuestion(
                question="Schedule?",
                category="scope",
                context="ctx",
                default="One-off",
            )
        ]
        qa = self._make_qa(questions, {0: "Daily at 9am"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert result["recurrence"] == "Daily at 9am"

    def test_merge_scope_one_off(self):
        questions = [
            FocusQuestion(
                question="Schedule?",
                category="scope",
                context="ctx",
                default="One-off",
            )
        ]
        qa = self._make_qa(questions, {0: "one-off"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert result["recurrence"] is None

    def test_merge_does_not_mutate_input(self):
        proc = _make_procedure()
        original_strategy = proc.get("strategy", "")
        questions = [
            FocusQuestion(
                question="Goal?",
                category="strategy",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "New strategy"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        questioner.merge_answers(proc, qa)
        assert proc.get("strategy", "") == original_strategy

    def test_merge_empty_answer_uses_default(self):
        questions = [
            FocusQuestion(
                question="Goal?",
                category="strategy",
                context="ctx",
                default="Browse and engage",
            )
        ]
        qa = self._make_qa(questions, {0: ""})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert "Browse and engage" in result["strategy"]

    def test_merge_out_of_range_index_ignored(self):
        questions = [
            FocusQuestion(
                question="Q?",
                category="strategy",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {5: "bogus", -1: "also bogus"})
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        # Should not crash and strategy should remain empty
        assert result.get("strategy", "") == ""

    def test_merge_multiple_answers(self):
        questions = [
            FocusQuestion("Login?", "credentials", "ctx", "No"),
            FocusQuestion("Goal?", "strategy", "ctx", "Browse"),
            FocusQuestion("Schedule?", "scope", "ctx", "One-off"),
        ]
        qa = self._make_qa(questions, {
            0: "Reddit personal account",
            1: "Find SaaS founders",
            2: "Weekly on Monday",
        })
        reasoner = _make_reasoner()
        questioner = FocusQuestioner(llm_reasoner=reasoner)

        result = questioner.merge_answers(_make_procedure(), qa)
        assert len(result["environment"]["accounts"]) == 1
        assert "SaaS" in result["strategy"]
        assert result["recurrence"] == "Weekly on Monday"


# ---------------------------------------------------------------------------
# Tests: File IPC helpers
# ---------------------------------------------------------------------------


class TestFileIPC:

    def test_write_and_read_questions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            questions = [
                FocusQuestion("Q1?", "credentials", "ctx1", "def1"),
                FocusQuestion("Q2?", "strategy", "ctx2", "def2"),
            ]

            path = write_focus_questions(state_dir, "sess-123", "my-slug", questions)
            assert path.exists()

            data = read_focus_questions(state_dir)
            assert data is not None
            assert data["session_id"] == "sess-123"
            assert data["slug"] == "my-slug"
            assert data["status"] == "pending"
            assert len(data["questions"]) == 2
            assert data["questions"][0]["question"] == "Q1?"
            assert data["questions"][1]["category"] == "strategy"

    def test_write_and_read_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            sop = {"title": "Test", "slug": "test"}
            proc = {"title": "Test", "strategy": "strat"}

            path = write_focus_pending(state_dir, "sess-123", "test", sop, proc)
            assert path.exists()

            data = read_focus_pending(state_dir)
            assert data is not None
            assert data["session_id"] == "sess-123"
            assert data["sop_template"]["title"] == "Test"
            assert data["procedure"]["strategy"] == "strat"

    def test_read_missing_questions_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert read_focus_questions(Path(tmpdir)) is None

    def test_read_missing_pending_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert read_focus_pending(Path(tmpdir)) is None

    def test_clear_qa_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            (state_dir / FOCUS_QUESTIONS_FILE).write_text("{}")
            (state_dir / FOCUS_PENDING_FILE).write_text("{}")

            clear_focus_qa_files(state_dir)

            assert not (state_dir / FOCUS_QUESTIONS_FILE).exists()
            assert not (state_dir / FOCUS_PENDING_FILE).exists()

    def test_clear_nonexistent_files_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clear_focus_qa_files(Path(tmpdir))  # Should not raise

    def test_read_invalid_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            (state_dir / FOCUS_QUESTIONS_FILE).write_text("not valid json{{{")
            assert read_focus_questions(state_dir) is None


class TestParseQAResult:

    def test_parse_answered(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [
                {"question": "Q1?", "category": "strategy", "context": "c", "default": "d"},
                {"question": "Q2?", "category": "credentials", "context": "c", "default": "d"},
            ],
            "status": "answered",
            "answers": {"0": "Answer one", "1": "Answer two"},
        }

        result = parse_qa_result_from_file(data)
        assert result is not None
        assert len(result.questions) == 2
        assert result.answers[0] == "Answer one"
        assert result.answers[1] == "Answer two"

    def test_parse_skipped_uses_defaults(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [
                {"question": "Q1?", "category": "strategy", "context": "c", "default": "Default 1"},
            ],
            "status": "skipped",
        }

        result = parse_qa_result_from_file(data)
        assert result is not None
        assert result.answers[0] == "Default 1"

    def test_parse_pending_returns_none(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [],
            "status": "pending",
        }
        assert parse_qa_result_from_file(data) is None

    def test_parse_invalid_status_returns_none(self):
        data = {
            "status": "unknown",
            "questions": [],
        }
        assert parse_qa_result_from_file(data) is None

    def test_parse_invalid_answer_keys_ignored(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [
                {"question": "Q?", "category": "strategy", "context": "c", "default": "d"},
            ],
            "status": "answered",
            "answers": {"0": "Good", "not_a_number": "Bad"},
        }

        result = parse_qa_result_from_file(data)
        assert result is not None
        assert 0 in result.answers
        assert len(result.answers) == 1

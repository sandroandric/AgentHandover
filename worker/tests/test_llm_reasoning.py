"""Tests for the shared LLM reasoning utility."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.llm_reasoning import (
    LLMReasoner,
    ReasoningConfig,
    ReasoningResult,
    TruncatedAnswer,
)
from agenthandover_worker.scheduler import ReasoningBudget


def _make_reasoner(budget=None):
    return LLMReasoner(
        config=ReasoningConfig(model="test-model", ollama_host="http://test:11434"),
        budget=budget,
    )


def _budget(max_calls: int = 100, max_minutes: float = 60.0) -> ReasoningBudget:
    day = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
    return ReasoningBudget(max_calls, max_minutes * 60.0, clock=lambda: day)


def _spent_budget() -> ReasoningBudget:
    budget = _budget(max_calls=1)
    budget.charge(1.0)
    return budget


def _mock_ollama(return_value):
    """Patch _call_ollama on the LLMReasoner class."""
    return patch.object(
        LLMReasoner, "_call_ollama", return_value=return_value,
    )


# ---------------------------------------------------------------------------
# reason_json
# ---------------------------------------------------------------------------


class TestReasonJSON:

    def test_parses_valid_response(self):
        data = {"strategy": "daily marketing", "confidence": 0.9}
        with _mock_ollama((json.dumps(data), 1.5)):
            r = _make_reasoner()
            result = r.reason_json("test prompt")
        assert result.success
        assert result.value == data
        assert not result.abstained

    def test_handles_fences_and_think_tags(self):
        raw = '<think>reasoning here</think>\n```json\n{"key": "val"}\n```'
        # _call_ollama strips think tags, so we simulate post-strip
        stripped = '```json\n{"key": "val"}\n```'
        with _mock_ollama((stripped, 2.0)):
            result = _make_reasoner().reason_json("prompt")
        assert result.success
        assert result.value == {"key": "val"}

    def test_handles_json_with_preamble(self):
        raw = 'Here is the result:\n{"answer": 42}'
        with _mock_ollama((raw, 1.0)):
            result = _make_reasoner().reason_json("prompt")
        assert result.success
        assert result.value == {"answer": 42}

    def test_returns_failure_when_over_budget(self):
        with patch.object(LLMReasoner, "_call_ollama") as call:
            result = _make_reasoner(budget=_spent_budget()).reason_json("prompt")
        assert not result.success
        assert result.error == "Over the daily reasoning budget"
        call.assert_not_called()

    def test_returns_failure_on_connection_error(self):
        with patch.object(
            LLMReasoner, "_call_ollama",
            side_effect=ConnectionError("refused"),
        ):
            result = _make_reasoner().reason_json("prompt")
        assert not result.success
        assert "connection" in result.error.lower()

    def test_detects_abstention(self):
        with _mock_ollama(("INSUFFICIENT_EVIDENCE: not enough data", 1.0)):
            result = _make_reasoner().reason_json("prompt")
        assert result.success  # call succeeded
        assert result.abstained
        assert result.value is None

    def test_returns_failure_on_invalid_json(self):
        with _mock_ollama(("this is not json at all", 1.0)):
            result = _make_reasoner().reason_json("prompt")
        assert not result.success
        assert "parse" in result.error.lower()

    def test_prompt_hash_deterministic(self):
        with _mock_ollama(('{"a": 1}', 1.0)):
            r1 = _make_reasoner().reason_json("same prompt")
        with _mock_ollama(('{"a": 1}', 1.0)):
            r2 = _make_reasoner().reason_json("same prompt")
        assert r1.prompt_hash == r2.prompt_hash
        assert len(r1.prompt_hash) == 12

    def test_different_prompts_different_hashes(self):
        with _mock_ollama(('{"a": 1}', 1.0)):
            r1 = _make_reasoner().reason_json("prompt A")
        with _mock_ollama(('{"a": 1}', 1.0)):
            r2 = _make_reasoner().reason_json("prompt B")
        assert r1.prompt_hash != r2.prompt_hash


# ---------------------------------------------------------------------------
# reason_text
# ---------------------------------------------------------------------------


class TestReasonText:

    def test_returns_plain_string(self):
        with _mock_ollama(("Daily Reddit Marketing", 1.0)):
            result = _make_reasoner().reason_text("prompt")
        assert result.success
        assert result.value == "Daily Reddit Marketing"

    def test_strips_quotes(self):
        with _mock_ollama(('"Weekly Expense Filing"', 0.5)):
            result = _make_reasoner().reason_text("prompt")
        assert result.value == "Weekly Expense Filing"

    def test_preserves_multiline(self):
        with _mock_ollama(("Good Label\nExtra explanation here", 0.5)):
            result = _make_reasoner().reason_text("prompt")
        assert result.value == "Good Label\nExtra explanation here"

    def test_detects_abstention(self):
        with _mock_ollama(("INSUFFICIENT_EVIDENCE", 0.5)):
            result = _make_reasoner().reason_text("prompt")
        assert result.abstained
        assert result.value is None

    def test_over_budget(self):
        result = _make_reasoner(budget=_spent_budget()).reason_text("prompt")
        assert not result.success


# ---------------------------------------------------------------------------
# reason_yesno
# ---------------------------------------------------------------------------


class TestReasonYesNo:

    def test_yes(self):
        with _mock_ollama(("YES, these are the same workflow.", 0.5)):
            result = _make_reasoner().reason_yesno("prompt")
        assert result.success
        assert result.value is True

    def test_no(self):
        with _mock_ollama(("No, they are different tasks.", 0.5)):
            result = _make_reasoner().reason_yesno("prompt")
        assert result.success
        assert result.value is False

    def test_ambiguous_returns_none(self):
        with _mock_ollama(("It depends on the context.", 0.5)):
            result = _make_reasoner().reason_yesno("prompt")
        assert result.success
        assert result.value is None

    def test_detects_abstention(self):
        with _mock_ollama(("INSUFFICIENT_EVIDENCE", 0.5)):
            result = _make_reasoner().reason_yesno("prompt")
        assert result.abstained

    def test_true_variants(self):
        for word in ("True", "correct", "Affirmative answer"):
            with _mock_ollama((word, 0.5)):
                result = _make_reasoner().reason_yesno("prompt")
            assert result.value is True, f"Expected True for '{word}'"

    def test_false_variants(self):
        for word in ("False", "incorrect", "Negative"):
            with _mock_ollama((word, 0.5)):
                result = _make_reasoner().reason_yesno("prompt")
            assert result.value is False, f"Expected False for '{word}'"


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------


class TestBudgetTracking:

    def test_each_call_is_charged_to_the_budget(self):
        budget = _budget()
        with _mock_ollama(('{"ok": true}', 3.5)):
            _make_reasoner(budget=budget).reason_json("prompt")
            _make_reasoner(budget=budget).reason_json("prompt")
        assert (budget.calls, budget.compute_seconds) == (2, 7.0)

    def test_a_truncated_answer_is_charged_too(self):
        budget = _budget()
        with patch.object(
            LLMReasoner, "_call_ollama", side_effect=TruncatedAnswer("cut", 12.0),
        ):
            result = _make_reasoner(budget=budget).reason_text("prompt")
        assert not result.success
        assert (budget.calls, budget.compute_seconds) == (1, 12.0)

    def test_a_refused_call_does_not_switch_models(self):
        """Over budget, the reasoner asks the model scheduler for nothing."""
        reasoner = _make_reasoner(budget=_spent_budget())
        reasoner.before_call = MagicMock(return_value=True)
        reasoner.reason_text("prompt")
        reasoner.before_call.assert_not_called()

    def test_no_budget_no_error(self):
        with _mock_ollama(('{"ok": true}', 1.0)):
            result = _make_reasoner(budget=None).reason_json("prompt")
        assert result.success


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:

    def test_make_provenance(self):
        with _mock_ollama(('{"data": 1}', 2.0)):
            reasoner = _make_reasoner()
            result = reasoner.reason_json("prompt", caller="test_caller")
            prov = reasoner.make_provenance(
                result, caller="test_caller", evidence_refs=["obs-1"],
            )
        assert prov["source"] == "llm_reasoning"
        assert prov["model"] == "test-model"
        assert prov["caller"] == "test_caller"
        assert prov["evidence_refs"] == ["obs-1"]
        assert len(prov["prompt_hash"]) == 12

    def test_generated_at_is_iso(self):
        with _mock_ollama(('{"data": 1}', 1.0)):
            result = _make_reasoner().reason_json("prompt")
        assert "T" in result.generated_at


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_response(self):
        with _mock_ollama(("", 0.1)):
            result = _make_reasoner().reason_json("prompt")
        assert not result.success

    def test_generic_exception_handled(self):
        with patch.object(
            LLMReasoner, "_call_ollama",
            side_effect=RuntimeError("unexpected"),
        ):
            result = _make_reasoner().reason_json("prompt")
        assert not result.success
        assert "unexpected" in result.error


# ---------------------------------------------------------------------------
# Truncated answers (done_reason "length") and thinking (WP-C9)
# ---------------------------------------------------------------------------


class TestTruncatedAnswer:
    """An answer Ollama cut off at num_predict is never parsed as complete."""

    CUT_OFF = '{"questions": [{"question": "Which Slack workspace'
    COMPLETE = '{"questions": []}'

    @staticmethod
    def _fake_generate(replies, calls):
        from agenthandover_worker.ollama import Response

        def generate(self, model, prompt, **kwargs):
            calls.append(kwargs)
            text, done_reason = replies.pop(0)
            return Response(text=text, done_reason=done_reason, elapsed=2.0)
        return generate

    def _run(self, replies, **reason_kwargs):
        from agenthandover_worker.ollama import OllamaClient

        calls: list[dict] = []
        with patch.object(OllamaClient, "generate", self._fake_generate(replies, calls)):
            result = _make_reasoner().reason_json("p", **reason_kwargs)
        return result, calls

    def test_reasoning_calls_do_not_think_by_default(self):
        result, calls = self._run([(self.COMPLETE, "stop")])
        assert result.success and result.value == {"questions": []}
        assert [c["think"] for c in calls] == [False]

    def test_a_truncated_answer_without_thinking_fails(self):
        result, calls = self._run([(self.CUT_OFF, "length")])
        assert not result.success
        assert result.value is None
        assert "Truncated answer" in result.error
        assert len(calls) == 1

    def test_a_truncated_answer_with_thinking_is_retried_once_without(self):
        result, calls = self._run(
            [(self.CUT_OFF, "length"), (self.COMPLETE, "stop")], think=True,
        )
        assert [c["think"] for c in calls] == [True, False]
        assert result.success and result.value == {"questions": []}
        assert result.elapsed_seconds == 4.0

    def test_a_retry_that_is_truncated_again_fails(self):
        result, calls = self._run(
            [(self.CUT_OFF, "length"), (self.CUT_OFF, "length")], think=True,
        )
        assert len(calls) == 2
        assert not result.success and "Truncated answer" in result.error

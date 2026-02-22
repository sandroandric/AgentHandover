"""Tests for sop_enhancer — LLM-enhanced SOP descriptions."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from oc_apprentice_worker.sop_enhancer import (
    SOPEnhancer,
    _OVERVIEW_KEYS,
    create_llm_backend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(
    response: dict | None = None,
    available: bool = True,
) -> MagicMock:
    """Create a mock VLM backend for testing."""
    backend = MagicMock()
    backend.is_available.return_value = available
    if response is not None:
        backend.infer.return_value = response
    return backend


def _sample_response() -> dict:
    return {
        "task_description": "This workflow submits a contact form to request support.",
        "execution_overview": {
            "goal": "Submit the contact form successfully",
            "prerequisites": "Browser open with the support page loaded",
            "key_inputs": "Name, email, message text",
            "decision_points": "Choice of support category",
            "success_criteria": "Confirmation page is displayed",
            "typical_duration": "1-2 minutes",
        },
    }


def _sample_template(
    slug: str = "submit_form",
    title: str = "Submit Contact Form",
    steps: list[dict] | None = None,
) -> dict:
    if steps is None:
        steps = [
            {"step": "click", "target": "Name field", "confidence": 0.9},
            {"step": "type", "target": "Name field", "parameters": {"text": "Alice"}, "confidence": 0.85},
            {"step": "click", "target": "Submit button", "confidence": 0.88},
        ]
    return {
        "slug": slug,
        "title": title,
        "steps": steps,
        "variables": [],
        "confidence_avg": 0.88,
        "apps_involved": ["Chrome"],
        "preconditions": [],
    }


# ---------------------------------------------------------------------------
# Basic enhancement
# ---------------------------------------------------------------------------


class TestEnhanceSOP:
    """Core enhancement flow."""

    def test_enhance_adds_task_description(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        result = enhancer.enhance_sop(template)
        assert "task_description" in result
        assert "contact form" in result["task_description"]

    def test_enhance_adds_execution_overview(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        result = enhancer.enhance_sop(template)
        assert "execution_overview" in result
        assert "goal" in result["execution_overview"]

    def test_enhance_returns_template_unchanged_on_failure(self):
        backend = _mock_backend(available=True)
        backend.infer.side_effect = RuntimeError("LLM error")
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        result = enhancer.enhance_sop(template)
        assert "task_description" not in result
        assert "execution_overview" not in result

    def test_enhance_retries_once_on_failure(self):
        backend = _mock_backend(available=True)
        # First call fails, second succeeds
        backend.infer.side_effect = [
            RuntimeError("Parse error"),
            _sample_response(),
        ]
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        result = enhancer.enhance_sop(template)
        assert "task_description" in result
        assert backend.infer.call_count == 2

    def test_enhance_second_call_has_json_suffix(self):
        backend = _mock_backend(available=True)
        backend.infer.side_effect = [
            RuntimeError("Parse error"),
            _sample_response(),
        ]
        enhancer = SOPEnhancer(backend)
        enhancer.enhance_sop(_sample_template())
        # Second call should have JSON-only suffix
        second_call_prompt = backend.infer.call_args_list[1][0][0]
        assert "JSON only" in second_call_prompt


# ---------------------------------------------------------------------------
# Budget and caching
# ---------------------------------------------------------------------------


class TestBudget:
    """Daily budget enforcement."""

    def test_respects_daily_budget(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend, max_enhancements_per_day=2)
        for i in range(3):
            enhancer.enhance_sop(_sample_template(slug=f"sop_{i}"))
        # Third call should not invoke backend
        assert backend.infer.call_count == 2

    def test_budget_resets_on_new_day(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend, max_enhancements_per_day=1)
        enhancer.enhance_sop(_sample_template(slug="sop_1"))
        assert backend.infer.call_count == 1

        # Simulate next day
        enhancer._last_reset_date = "1970-01-01"
        enhancer.enhance_sop(_sample_template(slug="sop_2"))
        assert backend.infer.call_count == 2

    def test_get_stats_budget(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend, max_enhancements_per_day=5)
        enhancer.enhance_sop(_sample_template())
        stats = enhancer.get_stats()
        assert stats["enhancements_today"] == 1
        assert stats["budget_remaining"] == 4


class TestCache:
    """Hash-based caching to skip unchanged SOPs."""

    def test_skips_unchanged_sop(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        enhancer.enhance_sop(template)
        assert backend.infer.call_count == 1

        # Same slug, same steps — should skip
        template2 = _sample_template()
        enhancer.enhance_sop(template2)
        assert backend.infer.call_count == 1

    def test_re_enhances_changed_sop(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend)
        enhancer.enhance_sop(_sample_template())
        assert backend.infer.call_count == 1

        # Different steps — should re-enhance
        changed = _sample_template(steps=[
            {"step": "navigate", "target": "Homepage", "confidence": 0.9},
        ])
        enhancer.enhance_sop(changed)
        assert backend.infer.call_count == 2

    def test_cache_stats(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend)
        enhancer.enhance_sop(_sample_template(slug="sop_a"))
        enhancer.enhance_sop(_sample_template(slug="sop_b"))
        stats = enhancer.get_stats()
        assert stats["cached_sops"] == 2


# ---------------------------------------------------------------------------
# Should-enhance logic
# ---------------------------------------------------------------------------


class TestShouldEnhance:
    """_should_enhance checks budget, availability, and cache."""

    def test_skips_when_backend_unavailable(self):
        backend = _mock_backend(available=False)
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        result = enhancer.enhance_sop(template)
        assert "task_description" not in result
        assert backend.infer.call_count == 0

    def test_skips_when_budget_exhausted(self):
        backend = _mock_backend(_sample_response())
        enhancer = SOPEnhancer(backend, max_enhancements_per_day=0)
        template = _sample_template()
        result = enhancer.enhance_sop(template)
        assert "task_description" not in result
        assert backend.infer.call_count == 0


# ---------------------------------------------------------------------------
# Steps hash
# ---------------------------------------------------------------------------


class TestComputeStepsHash:
    """Hash computation for change detection."""

    def test_deterministic(self):
        steps = [{"step": "click", "target": "A"}, {"step": "type", "target": "B"}]
        h1 = SOPEnhancer._compute_steps_hash(steps)
        h2 = SOPEnhancer._compute_steps_hash(steps)
        assert h1 == h2

    def test_different_steps_different_hash(self):
        steps_a = [{"step": "click", "target": "A"}]
        steps_b = [{"step": "click", "target": "B"}]
        assert SOPEnhancer._compute_steps_hash(steps_a) != SOPEnhancer._compute_steps_hash(steps_b)

    def test_uses_action_key_fallback(self):
        steps = [{"action": "navigate", "target": "Home"}]
        h = SOPEnhancer._compute_steps_hash(steps)
        assert len(h) == 64  # SHA-256 hex digest

    def test_empty_steps(self):
        h = SOPEnhancer._compute_steps_hash([])
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Prompt construction for LLM."""

    def test_system_prompt_has_json_spec(self):
        backend = _mock_backend()
        enhancer = SOPEnhancer(backend)
        system, user = enhancer._build_prompt(_sample_template())
        assert "task_description" in system
        assert "execution_overview" in system
        assert "JSON" in system

    def test_user_prompt_has_title_and_steps(self):
        backend = _mock_backend()
        enhancer = SOPEnhancer(backend)
        system, user = enhancer._build_prompt(_sample_template())
        assert "Submit Contact Form" in user
        assert "click" in user
        assert "Name field" in user

    def test_user_prompt_marks_data_untrusted(self):
        backend = _mock_backend()
        enhancer = SOPEnhancer(backend)
        _, user = enhancer._build_prompt(_sample_template())
        assert "untrusted" in user.lower()

    def test_user_prompt_includes_variables(self):
        backend = _mock_backend()
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        template["variables"] = [{"name": "customer_name"}, {"name": "order_id"}]
        _, user = enhancer._build_prompt(template)
        assert "customer_name" in user
        assert "order_id" in user

    def test_user_prompt_includes_preconditions(self):
        backend = _mock_backend()
        enhancer = SOPEnhancer(backend)
        template = _sample_template()
        template["preconditions"] = ["app_open:Chrome"]
        _, user = enhancer._build_prompt(template)
        assert "app_open:Chrome" in user


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Validate and extract fields from LLM response."""

    def test_valid_response(self):
        task_desc, overview = SOPEnhancer._parse_response(_sample_response())
        assert "contact form" in task_desc
        assert overview["goal"] == "Submit the contact form successfully"

    def test_missing_task_description(self):
        with pytest.raises(ValueError, match="task_description"):
            SOPEnhancer._parse_response({"execution_overview": {}})

    def test_empty_task_description(self):
        with pytest.raises(ValueError, match="task_description"):
            SOPEnhancer._parse_response({
                "task_description": "   ",
                "execution_overview": {},
            })

    def test_missing_execution_overview(self):
        with pytest.raises(ValueError, match="execution_overview"):
            SOPEnhancer._parse_response({
                "task_description": "Valid desc",
            })

    def test_non_dict_execution_overview(self):
        with pytest.raises(ValueError, match="execution_overview"):
            SOPEnhancer._parse_response({
                "task_description": "Valid desc",
                "execution_overview": "not a dict",
            })

    def test_non_string_overview_values_coerced(self):
        response = _sample_response()
        response["execution_overview"]["goal"] = 42
        task_desc, overview = SOPEnhancer._parse_response(response)
        assert overview["goal"] == "42"

    def test_task_description_stripped(self):
        response = _sample_response()
        response["task_description"] = "  padded text  "
        task_desc, _ = SOPEnhancer._parse_response(response)
        assert task_desc == "padded text"


# ---------------------------------------------------------------------------
# create_llm_backend factory
# ---------------------------------------------------------------------------


class TestCreateLlmBackend:
    """Factory function for LLM backend."""

    def test_returns_none_when_ollama_unavailable(self):
        with patch.dict("sys.modules", {"ollama": None}):
            result = create_llm_backend({"model": ""}, {"mode": "local"})
        assert result is None

    @patch("oc_apprentice_worker.vlm_worker.VLMWorker")
    def test_remote_provider_mapping(self, mock_worker_cls):
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = True
        mock_worker = MagicMock()
        mock_worker._backend = mock_backend
        mock_worker_cls.return_value = mock_worker

        result = create_llm_backend(
            {"model": "gpt-4o-mini"},
            {"mode": "remote", "provider": "openai", "api_key_env": "OPENAI_API_KEY"},
        )
        assert result is mock_backend

    @patch("oc_apprentice_worker.vlm_worker.VLMWorker")
    def test_remote_unknown_provider_returns_none(self, mock_worker_cls):
        result = create_llm_backend(
            {"model": ""},
            {"mode": "remote", "provider": "unknown_provider"},
        )
        assert result is None

    @patch("oc_apprentice_worker.vlm_worker.VLMWorker")
    def test_remote_backend_not_available_returns_none(self, mock_worker_cls):
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = False
        mock_worker = MagicMock()
        mock_worker._backend = mock_backend
        mock_worker_cls.return_value = mock_worker

        result = create_llm_backend(
            {"model": ""},
            {"mode": "remote", "provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
        )
        assert result is None

    def test_none_vlm_config_defaults_to_local(self):
        # Should attempt local mode (Ollama) — which will fail in test env
        result = create_llm_backend({"model": ""}, None)
        assert result is None  # Ollama not available in test env

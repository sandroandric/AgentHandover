"""Tests for the VLM backend module: config defaults, the mock backend and
the backend factory (``create_backend``)."""

from __future__ import annotations

import pytest

from agenthandover_worker.vlm_worker import (
    MockVLMBackend,
    VLMBackend,
    VLMConfig,
    create_backend,
)


class TestVLMConfigDefaults:
    def test_default_backend(self) -> None:
        cfg = VLMConfig()
        assert cfg.backend == VLMBackend.MOCK

    def test_default_model_name(self) -> None:
        cfg = VLMConfig()
        assert cfg.model_name == "mlx-community/llava-1.5-7b-4bit"

    def test_default_max_tokens(self) -> None:
        cfg = VLMConfig()
        assert cfg.max_tokens == 512

    def test_default_temperature(self) -> None:
        cfg = VLMConfig()
        assert cfg.temperature == 0.1

    def test_default_timeout(self) -> None:
        cfg = VLMConfig()
        assert cfg.timeout_seconds == 120.0


# ---------------------------------------------------------------------------
# Mock Backend
# ---------------------------------------------------------------------------


class TestMockBackend:
    def test_returns_default_response(self) -> None:
        backend = MockVLMBackend()
        result = backend.infer("test prompt")
        assert result["target_description"] == "Button element"
        assert result["suggested_selector"] == "[role='button']"
        assert result["confidence_boost"] == 0.15
        assert "reasoning" in result

    def test_returns_configured_responses(self) -> None:
        custom = [
            {"target_description": "Link element", "confidence_boost": 0.20},
            {"target_description": "Input field", "confidence_boost": 0.10},
        ]
        backend = MockVLMBackend(responses=custom)

        first = backend.infer("prompt 1")
        assert first["target_description"] == "Link element"
        assert first["confidence_boost"] == 0.20

        second = backend.infer("prompt 2")
        assert second["target_description"] == "Input field"

    def test_falls_back_to_default_after_configured_exhausted(self) -> None:
        custom = [{"target_description": "Custom", "confidence_boost": 0.25}]
        backend = MockVLMBackend(responses=custom)

        backend.infer("first")
        result = backend.infer("second")
        assert result["target_description"] == "Button element"

    def test_call_count_increments(self) -> None:
        backend = MockVLMBackend()
        assert backend.call_count == 0
        backend.infer("a")
        assert backend.call_count == 1
        backend.infer("b")
        backend.infer("c")
        assert backend.call_count == 3

    def test_is_available(self) -> None:
        backend = MockVLMBackend()
        assert backend.is_available() is True


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class TestBackendFactory:
    def test_creates_mock_backend_by_default(self) -> None:
        assert isinstance(create_backend(VLMConfig()), MockVLMBackend)

    def test_creates_ollama_backend(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend
        backend = create_backend(VLMConfig(backend=VLMBackend.OLLAMA))
        assert isinstance(backend, OllamaBackend)

    def test_creates_openai_compat_backend(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        backend = create_backend(VLMConfig(backend=VLMBackend.OPENAI_COMPAT))
        assert isinstance(backend, OpenAICompatBackend)

    def test_creates_anthropic_backend(self) -> None:
        from agenthandover_worker.backends.anthropic import AnthropicBackend
        backend = create_backend(VLMConfig(backend=VLMBackend.ANTHROPIC))
        assert isinstance(backend, AnthropicBackend)

    def test_creates_google_backend(self) -> None:
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        backend = create_backend(VLMConfig(backend=VLMBackend.GOOGLE_GENAI))
        assert isinstance(backend, GoogleGenAIBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend(VLMConfig(backend="nope"))  # type: ignore[arg-type]

    def test_enum_has_5_members(self) -> None:
        assert len(VLMBackend) == 5
        assert VLMBackend.OLLAMA.value == "ollama"
        assert VLMBackend.OPENAI_COMPAT.value == "openai-compat"
        assert VLMBackend.ANTHROPIC.value == "anthropic"
        assert VLMBackend.GOOGLE_GENAI.value == "google-genai"
        assert VLMBackend.MOCK.value == "mock"

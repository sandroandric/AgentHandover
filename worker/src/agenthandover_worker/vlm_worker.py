"""VLM backends: their configuration, the inference interface and the factory.

The one model user is ``sop_enhancer.create_llm_backend`` (the ``[llm]``
SOP text enhancer of the older focus path), local through Ollama or remote
through a cloud provider (``[vlm] mode = "remote"``); the worker's start-up
also builds the configured backend once for ``vlm_available`` in
``worker-status.json``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any


class VLMBackend(str, Enum):
    """Supported VLM inference backends."""
    OLLAMA = "ollama"              # Local Ollama server
    OPENAI_COMPAT = "openai-compat"  # OpenAI-compatible API
    ANTHROPIC = "anthropic"        # Anthropic Claude API
    GOOGLE_GENAI = "google-genai"  # Google Generative AI API
    MOCK = "mock"                  # For testing


@dataclass
class VLMConfig:
    """VLM backend configuration."""
    backend: VLMBackend = VLMBackend.MOCK
    model_name: str = "mlx-community/llava-1.5-7b-4bit"
    max_tokens: int = 512
    temperature: float = 0.1
    timeout_seconds: float = 120.0
    # openai-compat specific
    api_key: str | None = None             # API key (also reads env vars)
    base_url: str | None = None            # API base URL (None = standard OpenAI)
    # ollama specific
    ollama_host: str | None = None         # Ollama server URL (None = localhost:11434)
    # ollama think parameter — controls Qwen thinking mode.
    # False: Qwen3.5 models run without thinking (faster, no token waste).
    # True: enables reasoning step before response (used by sop_generator).
    # None: let Ollama/model decide (defaults to thinking for Qwen3.5).
    think: bool | None = False
    # remote mode fields
    mode: str = "local"                    # "local" or "remote"
    provider: str | None = None            # "openai" | "anthropic" | "google"
    remote_model: str | None = None        # provider-specific model override
    api_key_env: str | None = None         # env var name holding the key


class VLMInferenceBackend(abc.ABC):
    """Abstract base class for VLM inference backends."""

    @abc.abstractmethod
    def infer(
        self,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run inference and return parsed result.

        Args:
            prompt: The user-turn prompt (data section).
            image_base64: Optional base64-encoded image.
            system_prompt: Optional system-role instructions. Backends that
                support system messages (Ollama, OpenAI) should send this
                as a separate system message rather than embedding it in
                the user turn.
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        ...


class MockVLMBackend(VLMInferenceBackend):
    """Mock backend for testing."""

    def __init__(self, responses: list[dict] | None = None):
        self._responses = responses or []
        self._call_count = 0

    def infer(
        self,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if self._call_count < len(self._responses):
            result = self._responses[self._call_count]
        else:
            result = {
                "target_description": "Button element",
                "suggested_selector": "[role='button']",
                "confidence_boost": 0.15,
                "reasoning": "Identified button via visual analysis",
            }
        self._call_count += 1
        return result

    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count


def create_backend(config: VLMConfig) -> VLMInferenceBackend:
    """The backend *config* names (imported lazily: each pulls in its
    provider's SDK)."""
    if config.backend == VLMBackend.MOCK:
        return MockVLMBackend()
    if config.backend == VLMBackend.OLLAMA:
        from agenthandover_worker.backends.ollama import OllamaBackend
        return OllamaBackend(config)
    if config.backend == VLMBackend.OPENAI_COMPAT:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(config)
    if config.backend == VLMBackend.ANTHROPIC:
        from agenthandover_worker.backends.anthropic import AnthropicBackend
        return AnthropicBackend(config)
    if config.backend == VLMBackend.GOOGLE_GENAI:
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        return GoogleGenAIBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")

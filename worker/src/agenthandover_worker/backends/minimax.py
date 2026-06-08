"""MiniMax M-series Claude-compatible backend.

Uses the ``anthropic`` Python SDK pointed at MiniMax's Anthropic-compatible
endpoint for cloud-hosted VLM/LLM inference. Mirrors the structure of
``anthropic.py`` so MiniMax appears as a first-class peer of the Anthropic,
Google, and OpenAI backends.

The MiniMax Anthropic-compatible API does not accept several optional
parameters supported by Anthropic (``top_k``, ``stop_sequences``,
``service_tier``, ``mcp_servers``, ``context_management``, ``container``);
those are stripped client-side. Temperature is clamped into ``(0.0, 1.0]``
since MiniMax rejects ``0.0``.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING, Any

from agenthandover_worker.backends._json_parser import extract_json
from agenthandover_worker.backends._timeout import run_with_timeout
from agenthandover_worker.vlm_worker import VLMInferenceBackend

if TYPE_CHECKING:
    from agenthandover_worker.vlm_worker import VLMConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "MiniMax-M2.7"
_DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
# Parameters accepted by Anthropic SDK but rejected by the MiniMax
# Anthropic-compatible endpoint. Stripped before sending.
_UNSUPPORTED_PARAMS = frozenset((
    "top_k",
    "stop_sequences",
    "service_tier",
    "mcp_servers",
    "context_management",
    "container",
))


def _detect_media_type(image_base64: str) -> str:
    """Detect image media type from base64-encoded data via magic bytes.

    Returns 'image/png' for PNG, 'image/jpeg' otherwise.
    """
    try:
        header = base64.b64decode(image_base64[:32], validate=True)
        if header[:4] == b"\x89PNG":
            return "image/png"
    except Exception:
        pass
    return "image/jpeg"


def _clamp_temperature(value: float) -> float:
    """MiniMax accepts temperature in (0.0, 1.0]; clamp accordingly."""
    if value <= 0.0:
        return 0.01
    if value > 1.0:
        return 1.0
    return value


class MiniMaxBackend(VLMInferenceBackend):
    """VLM/LLM backend for the MiniMax Anthropic-compatible API."""

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _resolve_api_key(self) -> str | None:
        """Resolve API key: config.api_key -> config.api_key_env -> MINIMAX_API_KEY."""
        if self._config.api_key:
            return self._config.api_key
        if self._config.api_key_env:
            key = os.environ.get(self._config.api_key_env)
            if key:
                return key
        return os.environ.get("MINIMAX_API_KEY")

    def _resolve_base_url(self) -> str:
        """Resolve base URL: config.base_url -> MINIMAX_BASE_URL -> default."""
        if self._config.base_url:
            return self._config.base_url
        env_url = os.environ.get("MINIMAX_BASE_URL")
        if env_url:
            return env_url
        return _DEFAULT_BASE_URL

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "MiniMax API key not found. Set MINIMAX_API_KEY env var "
                "or configure api_key_env in config.toml [vlm] section."
            )

        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=self._resolve_base_url(),
            timeout=self._config.timeout_seconds,
        )
        return self._client

    @property
    def _model_name(self) -> str:
        if self._config.remote_model:
            return self._config.remote_model
        return _DEFAULT_MODEL

    def infer(
        self,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()

        # Build content blocks for the user message
        content: list[dict[str, Any]] = []

        if image_base64:
            media_type = _detect_media_type(image_base64)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            })

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": _clamp_temperature(self._config.temperature),
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Strip parameters that the MiniMax endpoint rejects.
        for key in _UNSUPPORTED_PARAMS:
            kwargs.pop(key, None)

        def _call() -> Any:
            return client.messages.create(**kwargs)

        response = run_with_timeout(_call, self._config.timeout_seconds)
        # Extract text from the first text content block
        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text = block.text
                break

        return extract_json(raw_text)

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

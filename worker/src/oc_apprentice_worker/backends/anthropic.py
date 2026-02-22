"""Anthropic Claude API backend.

Uses the ``anthropic`` Python SDK for cloud-hosted VLM inference.
Image content uses Anthropic's native base64 image blocks (not OpenAI
image_url format). System prompt is passed via the top-level ``system=``
kwarg, not as a message.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING, Any

from oc_apprentice_worker.backends._json_parser import extract_json
from oc_apprentice_worker.backends._timeout import run_with_timeout
from oc_apprentice_worker.vlm_worker import VLMInferenceBackend

if TYPE_CHECKING:
    from oc_apprentice_worker.vlm_worker import VLMConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-20250514"


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


class AnthropicBackend(VLMInferenceBackend):
    """VLM backend for the Anthropic Claude API."""

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _resolve_api_key(self) -> str | None:
        """Resolve API key: config.api_key → config.api_key_env → ANTHROPIC_API_KEY."""
        if self._config.api_key:
            return self._config.api_key
        if self._config.api_key_env:
            key = os.environ.get(self._config.api_key_env)
            if key:
                return key
        return os.environ.get("ANTHROPIC_API_KEY")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var "
                "or configure api_key_env in config.toml [vlm] section."
            )

        self._client = anthropic.Anthropic(
            api_key=api_key,
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
            "temperature": self._config.temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

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

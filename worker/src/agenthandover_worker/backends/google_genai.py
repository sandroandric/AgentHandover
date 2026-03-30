"""Google Generative AI (Gemini) backend.

Uses the ``google-genai`` SDK for cloud-hosted VLM inference.
Image content is passed via ``types.Part.from_bytes``. System
instructions use ``GenerateContentConfig.system_instruction``.
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

_DEFAULT_MODEL = "gemini-2.0-flash"


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect image MIME type from raw bytes via magic bytes."""
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    return "image/jpeg"


class GoogleGenAIBackend(VLMInferenceBackend):
    """VLM backend for the Google Generative AI (Gemini) API."""

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _resolve_api_key(self) -> str | None:
        """Resolve API key: config.api_key → config.api_key_env → GOOGLE_API_KEY."""
        if self._config.api_key:
            return self._config.api_key
        if self._config.api_key_env:
            key = os.environ.get(self._config.api_key_env)
            if key:
                return key
        return os.environ.get("GOOGLE_API_KEY")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google import genai

        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY env var "
                "or configure api_key_env in config.toml [vlm] section."
            )

        self._client = genai.Client(api_key=api_key)
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
        from google.genai import types

        client = self._get_client()

        # Build content parts
        contents: list[Any] = []

        if image_base64:
            decoded_bytes = base64.b64decode(image_base64)
            mime_type = _detect_mime_type(decoded_bytes)
            contents.append(
                types.Part.from_bytes(data=decoded_bytes, mime_type=mime_type)
            )

        contents.append(prompt)

        # Build generation config
        config_kwargs: dict[str, Any] = {
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_tokens,
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        gen_config = types.GenerateContentConfig(**config_kwargs)

        def _call() -> Any:
            return client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=gen_config,
            )

        response = run_with_timeout(_call, self._config.timeout_seconds)
        raw_text = response.text or ""
        return extract_json(raw_text)

    def is_available(self) -> bool:
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

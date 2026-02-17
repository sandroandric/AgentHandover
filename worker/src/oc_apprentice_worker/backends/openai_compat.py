"""OpenAI-compatible API backend.

Works with any API that implements the OpenAI chat completions interface:
standard OpenAI, Azure OpenAI, local servers (vLLM, text-generation-webui,
LM Studio, etc.).
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from oc_apprentice_worker.backends._json_parser import extract_json
from oc_apprentice_worker.backends._timeout import run_with_timeout
from oc_apprentice_worker.vlm_worker import VLMInferenceBackend

if TYPE_CHECKING:
    from oc_apprentice_worker.vlm_worker import VLMConfig

logger = logging.getLogger(__name__)

_DEFAULT_MLX_MODEL = "mlx-community/llava-1.5-7b-4bit"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatBackend(VLMInferenceBackend):
    """VLM backend for OpenAI-compatible APIs."""

    _AVAILABILITY_TTL = 60.0  # Cache is_available() result for 60 seconds

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Any = None
        self._available_cache: bool | None = None
        self._available_cache_time: float = 0.0

    def _resolve_api_key(self) -> str | None:
        if self._config.api_key:
            return self._config.api_key
        key = os.environ.get("OPENMIMIC_API_KEY")
        if key:
            return key
        return os.environ.get("OPENAI_API_KEY")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        api_key = self._resolve_api_key()
        # For local providers with custom base_url: key may not be needed
        if not api_key and self._config.base_url:
            api_key = "not-needed"

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
        )
        return self._client

    @property
    def _model_name(self) -> str:
        name = self._config.model_name
        if name == _DEFAULT_MLX_MODEL:
            return _OPENAI_DEFAULT_MODEL
        return name

    def infer(self, prompt: str, image_base64: str | None = None) -> dict[str, Any]:
        client = self._get_client()

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            })

        def _call() -> Any:
            return client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

        response = run_with_timeout(_call, self._config.timeout_seconds)
        raw_text = response.choices[0].message.content
        return extract_json(raw_text)

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False

        # Remote provider (no custom base_url): just check API key exists
        if not self._config.base_url:
            return self._resolve_api_key() is not None

        # Local provider (custom base_url): try connecting (cached)
        now = time.monotonic()
        if (
            self._available_cache is not None
            and (now - self._available_cache_time) < self._AVAILABILITY_TTL
        ):
            return self._available_cache
        try:
            client = self._get_client()
            client.models.list()
            self._available_cache = True
        except Exception:
            self._available_cache = False
        self._available_cache_time = now
        return self._available_cache

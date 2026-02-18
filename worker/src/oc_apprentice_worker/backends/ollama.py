"""Ollama backend for local server inference.

Uses the ``ollama`` Python package to talk to a locally-running Ollama
server. The server must be started separately (e.g. ``ollama serve``).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from oc_apprentice_worker.backends._json_parser import extract_json
from oc_apprentice_worker.backends._timeout import run_with_timeout
from oc_apprentice_worker.vlm_worker import VLMInferenceBackend

if TYPE_CHECKING:
    from oc_apprentice_worker.vlm_worker import VLMConfig

logger = logging.getLogger(__name__)

# Default MLX model name — when this is the configured model, substitute
# a sensible Ollama default since MLX models aren't available in Ollama.
_DEFAULT_MLX_MODEL = "mlx-community/llava-1.5-7b-4bit"
_OLLAMA_DEFAULT_MODEL = "llava:7b"


class OllamaBackend(VLMInferenceBackend):
    """VLM backend using a local Ollama server."""

    _AVAILABILITY_TTL = 60.0  # Cache is_available() result for 60 seconds

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Any = None
        self._available_cache: bool | None = None
        self._available_cache_time: float = 0.0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import ollama

        if self._config.ollama_host:
            self._client = ollama.Client(host=self._config.ollama_host)
        else:
            self._client = ollama.Client()
        return self._client

    @property
    def _model_name(self) -> str:
        name = self._config.model_name
        if name == _DEFAULT_MLX_MODEL:
            return _OLLAMA_DEFAULT_MODEL
        return name

    def infer(
        self,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()

        images = [image_base64] if image_base64 else None

        # Build messages with system-role separation when provided
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)

        def _generate() -> dict:
            return client.chat(
                model=self._model_name,
                messages=messages,
                options={
                    "temperature": self._config.temperature,
                    "num_predict": self._config.max_tokens,
                },
            )

        response = run_with_timeout(_generate, self._config.timeout_seconds)
        raw_text = response["message"]["content"]
        return extract_json(raw_text)

    def is_available(self) -> bool:
        now = time.monotonic()
        if (
            self._available_cache is not None
            and (now - self._available_cache_time) < self._AVAILABILITY_TTL
        ):
            return self._available_cache
        try:
            client = self._get_client()
            client.list()
            self._available_cache = True
        except Exception:
            self._available_cache = False
        self._available_cache_time = now
        return self._available_cache

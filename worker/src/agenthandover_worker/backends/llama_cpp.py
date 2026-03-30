"""Llama.cpp backend for cross-platform GGUF model inference.

Uses the ``llama-cpp-python`` package. Requires a local ``.gguf`` model file
specified via ``config.model_path``. Optionally accepts a CLIP model for
multimodal (image + text) inference via ``config.clip_model_path``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenthandover_worker.backends._json_parser import extract_json
from agenthandover_worker.backends._timeout import run_with_timeout
from agenthandover_worker.vlm_worker import VLMInferenceBackend

if TYPE_CHECKING:
    from agenthandover_worker.vlm_worker import VLMConfig

logger = logging.getLogger(__name__)


class LlamaCppBackend(VLMInferenceBackend):
    """Cross-platform VLM backend using llama-cpp-python."""

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._model: Any = None
        self._lock = threading.Lock()

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            if not self._config.model_path:
                raise ValueError(
                    "LlamaCppBackend requires config.model_path "
                    "(path to a .gguf model file)"
                )

            model_file = Path(self._config.model_path)
            if not model_file.is_file():
                raise FileNotFoundError(
                    f"GGUF model file not found: {model_file}"
                )

            from llama_cpp import Llama

            chat_handler = None
            if self._config.clip_model_path:
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                chat_handler = Llava15ChatHandler(
                    clip_model_path=self._config.clip_model_path
                )

            logger.info("Loading llama.cpp model: %s", model_file)
            self._model = Llama(
                model_path=str(model_file),
                chat_handler=chat_handler,
                n_ctx=self._config.n_ctx,
                logits_all=True,
                verbose=False,
            )
            logger.info("llama.cpp model loaded successfully")

    def infer(
        self,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        self._lazy_load()

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            })

        # llama-cpp-python supports system messages via chat completion API
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        def _generate() -> dict:
            return self._model.create_chat_completion(
                messages=messages,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

        response = run_with_timeout(_generate, self._config.timeout_seconds)
        raw_text = response["choices"][0]["message"]["content"]
        return extract_json(raw_text)

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except (ImportError, OSError):
            return False
        if self._config.model_path:
            return Path(self._config.model_path).is_file()
        return False

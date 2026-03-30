"""MLX-VLM backend for Apple Silicon Macs.

Uses the ``mlx-vlm`` package for fast local inference on Apple Silicon.
Model is loaded lazily on first inference call.
"""

from __future__ import annotations

import base64
import logging
import threading
from io import BytesIO
from typing import TYPE_CHECKING, Any

from agenthandover_worker.backends._json_parser import extract_json
from agenthandover_worker.backends._timeout import run_with_timeout
from agenthandover_worker.vlm_worker import VLMInferenceBackend

if TYPE_CHECKING:
    from agenthandover_worker.vlm_worker import VLMConfig

logger = logging.getLogger(__name__)


class MLXVLMBackend(VLMInferenceBackend):
    """Apple Silicon VLM backend using mlx-vlm."""

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._model: Any = None
        self._processor: Any = None
        self._model_config: Any = None
        self._lock = threading.Lock()

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import mlx_vlm

            logger.info("Loading MLX-VLM model: %s", self._config.model_name)
            self._model, self._processor = mlx_vlm.load(self._config.model_name)
            from mlx_vlm.utils import load_config
            self._model_config = load_config(self._config.model_name)
            logger.info("MLX-VLM model loaded successfully")

    def infer(
        self,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        self._lazy_load()
        import mlx_vlm
        from mlx_vlm.prompt_utils import apply_chat_template

        # MLX-VLM doesn't have native system-role support via chat template,
        # so prepend system prompt to user prompt when provided.
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        image = None
        if image_base64:
            from PIL import Image

            img_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(img_bytes))

        num_images = 1 if image is not None else 0
        formatted_prompt = apply_chat_template(
            self._processor, self._model_config, full_prompt, num_images=num_images
        )

        def _generate() -> str:
            return mlx_vlm.generate(
                self._model,
                self._processor,
                formatted_prompt,
                image=image,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

        raw_output = run_with_timeout(_generate, self._config.timeout_seconds)
        return extract_json(raw_output)

    def is_available(self) -> bool:
        try:
            import mlx_vlm  # noqa: F401
            return True
        except ImportError:
            return False

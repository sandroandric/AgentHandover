"""Tests for the 4 VLM inference backends.

All tests mock at the library level — no real inference is performed.
Each backend gets 8+ tests covering:
- is_available with package missing/present
- infer with text-only and image input
- JSON extraction from code blocks
- unparseable output raises ValueError
- lazy loading (model not loaded at __init__)
- timeout handling
Plus backend-specific edge cases.
"""

from __future__ import annotations

import base64
import types
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.vlm_worker import VLMConfig, VLMBackend


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_config(**overrides) -> VLMConfig:
    defaults = {
        "backend": VLMBackend.MOCK,
        "model_name": "test-model",
        "max_tokens": 64,
        "temperature": 0.1,
        "timeout_seconds": 10.0,
    }
    defaults.update(overrides)
    return VLMConfig(**defaults)


_VALID_JSON = '{"target_description": "button", "confidence_boost": 0.2}'
_CODE_BLOCK_JSON = '```json\n{"target_description": "link"}\n```'
_GARBAGE = "I cannot process this image properly. No JSON here."


# =========================================================================
# MLX-VLM Backend Tests
# =========================================================================

class TestMLXVLMBackend:
    def test_is_available_package_missing(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        def _fail_import(name, *args, **kwargs):
            if name == "mlx_vlm":
                raise ImportError("No module named 'mlx_vlm'")
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", _fail_import)
        backend = MLXVLMBackend(_make_config())
        assert backend.is_available() is False

    def test_is_available_package_present(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        # Create a fake mlx_vlm module
        fake_module = types.ModuleType("mlx_vlm")
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_module)
        backend = MLXVLMBackend(_make_config())
        assert backend.is_available() is True

    def test_infer_text_only(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        backend = MLXVLMBackend(_make_config())
        # Pre-set model to skip lazy loading
        backend._model = MagicMock()
        backend._processor = MagicMock()
        backend._model_config = MagicMock()

        fake_mlx = types.ModuleType("mlx_vlm")
        fake_mlx.generate = MagicMock(return_value=_VALID_JSON)
        fake_mlx.load = MagicMock()
        fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        fake_prompt_utils.apply_chat_template = MagicMock(return_value="formatted")
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_mlx)
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

        result = backend.infer("test prompt")
        assert result["target_description"] == "button"

    def test_infer_with_image(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        backend = MLXVLMBackend(_make_config())
        backend._model = MagicMock()
        backend._processor = MagicMock()
        backend._model_config = MagicMock()

        fake_mlx = types.ModuleType("mlx_vlm")
        fake_mlx.generate = MagicMock(return_value=_VALID_JSON)
        fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        fake_prompt_utils.apply_chat_template = MagicMock(return_value="formatted")
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_mlx)
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

        # Mock PIL.Image
        fake_pil = types.ModuleType("PIL")
        fake_image_mod = types.ModuleType("PIL.Image")
        mock_img = MagicMock()
        fake_image_mod.open = MagicMock(return_value=mock_img)
        fake_pil.Image = fake_image_mod
        monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
        monkeypatch.setitem(__import__("sys").modules, "PIL.Image", fake_image_mod)

        img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
        result = backend.infer("test prompt", image_base64=img_b64)
        assert result["target_description"] == "button"
        # Verify generate was called with image kwarg
        call_kwargs = fake_mlx.generate.call_args
        assert call_kwargs.kwargs.get("image") is mock_img or call_kwargs[1].get("image") is mock_img

    def test_infer_json_in_code_block(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        backend = MLXVLMBackend(_make_config())
        backend._model = MagicMock()
        backend._processor = MagicMock()
        backend._model_config = MagicMock()

        fake_mlx = types.ModuleType("mlx_vlm")
        fake_mlx.generate = MagicMock(return_value=_CODE_BLOCK_JSON)
        fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        fake_prompt_utils.apply_chat_template = MagicMock(return_value="formatted")
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_mlx)
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

        result = backend.infer("test")
        assert result["target_description"] == "link"

    def test_infer_unparseable_output(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        backend = MLXVLMBackend(_make_config())
        backend._model = MagicMock()
        backend._processor = MagicMock()
        backend._model_config = MagicMock()

        fake_mlx = types.ModuleType("mlx_vlm")
        fake_mlx.generate = MagicMock(return_value=_GARBAGE)
        fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        fake_prompt_utils.apply_chat_template = MagicMock(return_value="formatted")
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_mlx)
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

        with pytest.raises(ValueError, match="Could not extract"):
            backend.infer("test")

    def test_lazy_loading(self) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend

        backend = MLXVLMBackend(_make_config())
        assert backend._model is None
        assert backend._processor is None

    def test_timeout_exceeded(self, monkeypatch) -> None:
        from agenthandover_worker.backends.mlx_vlm import MLXVLMBackend
        import time

        backend = MLXVLMBackend(_make_config(timeout_seconds=0.1))
        backend._model = MagicMock()
        backend._processor = MagicMock()
        backend._model_config = MagicMock()

        def slow_generate(*args, **kwargs):
            time.sleep(5)
            return _VALID_JSON

        fake_mlx = types.ModuleType("mlx_vlm")
        fake_mlx.generate = slow_generate
        fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        fake_prompt_utils.apply_chat_template = MagicMock(return_value="formatted")
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_mlx)
        monkeypatch.setitem(__import__("sys").modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer("test")


# =========================================================================
# LlamaCpp Backend Tests
# =========================================================================

class TestLlamaCppBackend:
    def test_is_available_package_missing(self, monkeypatch) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        def _fail_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("No module named 'llama_cpp'")
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", _fail_import)
        backend = LlamaCppBackend(_make_config())
        assert backend.is_available() is False

    def test_is_available_package_present_no_model(self) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        # No model_path set — should be unavailable
        backend = LlamaCppBackend(_make_config())
        assert backend.is_available() is False

    def test_is_available_model_file_missing(self, tmp_path) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(
            _make_config(model_path=str(tmp_path / "nonexistent.gguf"))
        )
        assert backend.is_available() is False

    def test_model_path_validation(self) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(_make_config())
        with pytest.raises(ValueError, match="model_path"):
            backend._lazy_load()

    def test_infer_text_only(self, monkeypatch) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(_make_config(model_path="/fake/model.gguf"))
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": _VALID_JSON}}]
        }
        backend._model = mock_model

        result = backend.infer("test prompt")
        assert result["target_description"] == "button"

    def test_infer_with_image(self, monkeypatch) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(_make_config(model_path="/fake/model.gguf"))
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": _VALID_JSON}}]
        }
        backend._model = mock_model

        img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
        result = backend.infer("test prompt", image_base64=img_b64)
        assert result["target_description"] == "button"
        # Verify image_url was included in the message
        call_args = mock_model.create_chat_completion.call_args
        messages = call_args[1].get("messages") or call_args[0][0] if call_args[0] else call_args[1]["messages"]
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image_url"

    def test_infer_json_in_code_block(self) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(_make_config(model_path="/fake/model.gguf"))
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": _CODE_BLOCK_JSON}}]
        }
        backend._model = mock_model

        result = backend.infer("test")
        assert result["target_description"] == "link"

    def test_infer_unparseable_output(self) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(_make_config(model_path="/fake/model.gguf"))
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": _GARBAGE}}]
        }
        backend._model = mock_model

        with pytest.raises(ValueError, match="Could not extract"):
            backend.infer("test")

    def test_lazy_loading(self) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(_make_config())
        assert backend._model is None

    def test_timeout_exceeded(self) -> None:
        from agenthandover_worker.backends.llama_cpp import LlamaCppBackend
        import time

        backend = LlamaCppBackend(_make_config(
            model_path="/fake/model.gguf",
            timeout_seconds=0.1,
        ))

        def slow_completion(*args, **kwargs):
            time.sleep(5)
            return {"choices": [{"message": {"content": _VALID_JSON}}]}

        mock_model = MagicMock()
        mock_model.create_chat_completion = slow_completion
        backend._model = mock_model

        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer("test")


# =========================================================================
# Ollama Backend Tests
# =========================================================================

class TestOllamaBackend:
    def test_is_available_package_missing(self, monkeypatch) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        def _fail_import(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("No module named 'ollama'")
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", _fail_import)
        backend = OllamaBackend(_make_config())
        # Reset client so _get_client tries to import
        backend._client = None
        assert backend.is_available() is False

    def test_is_available_server_reachable(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        backend._client = mock_client
        assert backend.is_available() is True

    def test_is_available_server_down(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        mock_client = MagicMock()
        mock_client.list.side_effect = ConnectionError("refused")
        backend._client = mock_client
        assert backend.is_available() is False

    def test_infer_text_only(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": _VALID_JSON}}
        backend._client = mock_client

        result = backend.infer("test prompt")
        assert result["target_description"] == "button"

    def test_infer_with_image(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": _VALID_JSON}}
        backend._client = mock_client

        img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
        result = backend.infer("test prompt", image_base64=img_b64)
        assert result["target_description"] == "button"
        # Verify images were passed
        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["images"] == [img_b64]

    def test_infer_json_in_code_block(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": _CODE_BLOCK_JSON}}
        backend._client = mock_client

        result = backend.infer("test")
        assert result["target_description"] == "link"

    def test_infer_unparseable_output(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": _GARBAGE}}
        backend._client = mock_client

        with pytest.raises(ValueError, match="Could not extract"):
            backend.infer("test")

    def test_lazy_loading(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        backend = OllamaBackend(_make_config())
        assert backend._client is None

    def test_model_name_fallback(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        # When using the default MLX model name, should substitute llava:7b
        config = _make_config(model_name="mlx-community/llava-1.5-7b-4bit")
        backend = OllamaBackend(config)
        assert backend._model_name == "llava:7b"

    def test_model_name_custom_preserved(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend

        config = _make_config(model_name="llava:13b")
        backend = OllamaBackend(config)
        assert backend._model_name == "llava:13b"

    def test_timeout_exceeded(self) -> None:
        from agenthandover_worker.backends.ollama import OllamaBackend
        import time

        backend = OllamaBackend(_make_config(timeout_seconds=0.1))

        def slow_chat(*args, **kwargs):
            time.sleep(5)
            return {"message": {"content": _VALID_JSON}}

        mock_client = MagicMock()
        mock_client.chat = slow_chat
        backend._client = mock_client

        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer("test")


# =========================================================================
# OpenAI-Compatible Backend Tests
# =========================================================================

class TestOpenAICompatBackend:
    def test_is_available_package_missing(self, monkeypatch) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        def _fail_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", _fail_import)
        backend = OpenAICompatBackend(_make_config())
        assert backend.is_available() is False

    def test_is_available_with_api_key(self, monkeypatch) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        import sys

        # Ensure openai module exists (mock it)
        fake_openai = types.ModuleType("openai")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        config = _make_config(api_key="sk-test-key")
        backend = OpenAICompatBackend(config)
        assert backend.is_available() is True

    def test_is_available_key_from_env(self, monkeypatch) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        import sys

        fake_openai = types.ModuleType("openai")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        backend = OpenAICompatBackend(_make_config())
        assert backend.is_available() is True

    def test_is_available_no_key(self, monkeypatch) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        import sys

        fake_openai = types.ModuleType("openai")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AGENTHANDOVER_API_KEY", raising=False)

        backend = OpenAICompatBackend(_make_config())
        assert backend.is_available() is False

    def test_is_available_local_provider(self, monkeypatch) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        import sys

        fake_openai = types.ModuleType("openai")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        config = _make_config(base_url="http://localhost:8080/v1")
        backend = OpenAICompatBackend(config)
        # Mock client to simulate local server
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        backend._client = mock_client
        assert backend.is_available() is True

    def test_infer_text_only(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        config = _make_config(api_key="sk-test")
        backend = OpenAICompatBackend(config)

        mock_choice = MagicMock()
        mock_choice.message.content = _VALID_JSON
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        result = backend.infer("test prompt")
        assert result["target_description"] == "button"

    def test_infer_with_image(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        config = _make_config(api_key="sk-test")
        backend = OpenAICompatBackend(config)

        mock_choice = MagicMock()
        mock_choice.message.content = _VALID_JSON
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
        result = backend.infer("test prompt", image_base64=img_b64)
        assert result["target_description"] == "button"
        # Verify content has image_url
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image_url"

    def test_infer_json_in_code_block(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        config = _make_config(api_key="sk-test")
        backend = OpenAICompatBackend(config)

        mock_choice = MagicMock()
        mock_choice.message.content = _CODE_BLOCK_JSON
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        result = backend.infer("test")
        assert result["target_description"] == "link"

    def test_infer_unparseable_output(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        config = _make_config(api_key="sk-test")
        backend = OpenAICompatBackend(config)

        mock_choice = MagicMock()
        mock_choice.message.content = _GARBAGE
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        with pytest.raises(ValueError, match="Could not extract"):
            backend.infer("test")

    def test_lazy_loading(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        backend = OpenAICompatBackend(_make_config())
        assert backend._client is None

    def test_model_name_fallback(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        config = _make_config(model_name="mlx-community/llava-1.5-7b-4bit")
        backend = OpenAICompatBackend(config)
        assert backend._model_name == "gpt-4o-mini"

    def test_model_name_custom_preserved(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        config = _make_config(model_name="gpt-4o")
        backend = OpenAICompatBackend(config)
        assert backend._model_name == "gpt-4o"

    def test_timeout_exceeded(self) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend
        import time

        config = _make_config(api_key="sk-test", timeout_seconds=0.1)
        backend = OpenAICompatBackend(config)

        def slow_create(*args, **kwargs):
            time.sleep(5)
            mock_choice = MagicMock()
            mock_choice.message.content = _VALID_JSON
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create = slow_create
        backend._client = mock_client

        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer("test")

    def test_local_provider_dummy_key(self, monkeypatch) -> None:
        from agenthandover_worker.backends.openai_compat import OpenAICompatBackend

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AGENTHANDOVER_API_KEY", raising=False)

        config = _make_config(base_url="http://localhost:8080/v1")
        backend = OpenAICompatBackend(config)
        assert backend._resolve_api_key() is None
        # _get_client should use "not-needed" for local providers

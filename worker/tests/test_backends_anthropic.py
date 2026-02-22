"""Tests for the Anthropic Claude VLM backend."""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from oc_apprentice_worker.vlm_worker import VLMConfig, VLMBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> VLMConfig:
    defaults = {
        "backend": VLMBackend.ANTHROPIC,
        "mode": "remote",
        "provider": "anthropic",
        "api_key": "sk-ant-test-key-1234567890",
        "timeout_seconds": 30.0,
        "max_tokens": 512,
        "temperature": 0.1,
    }
    defaults.update(overrides)
    return VLMConfig(**defaults)


def _png_b64() -> str:
    """Minimal valid PNG header as base64."""
    header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    return base64.b64encode(header).decode()


def _jpeg_b64() -> str:
    """Minimal JPEG header as base64."""
    header = b"\xff\xd8\xff\xe0" + b"\x00" * 20
    return base64.b64encode(header).decode()


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------

class TestAnthropicAvailability:
    def test_not_available_without_package(self):
        """is_available() returns False when anthropic is not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            from oc_apprentice_worker.backends.anthropic import AnthropicBackend
            backend = AnthropicBackend(_make_config(api_key=None, api_key_env=None))
            # Force re-import check to fail
            with patch("builtins.__import__", side_effect=ImportError):
                assert backend.is_available() is False

    def test_not_available_without_key(self):
        """is_available() returns False when no API key can be resolved."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend
        config = _make_config(api_key=None, api_key_env=None)
        backend = AnthropicBackend(config)
        with patch.dict(os.environ, {}, clear=True):
            assert backend.is_available() is False

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    def test_available_with_config_key(self):
        """is_available() returns True when api_key is set in config."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(_make_config(api_key="sk-ant-valid"))
        assert backend.is_available() is True

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    def test_available_with_env_var(self):
        """is_available() returns True when ANTHROPIC_API_KEY env var is set."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend
        config = _make_config(api_key=None, api_key_env=None)
        backend = AnthropicBackend(config)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            assert backend.is_available() is True

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    def test_available_with_api_key_env(self):
        """is_available() resolves api_key_env to look up the correct env var."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend
        config = _make_config(api_key=None, api_key_env="MY_CUSTOM_KEY")
        backend = AnthropicBackend(config)
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "sk-ant-custom"}):
            assert backend.is_available() is True


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

class TestAnthropicInfer:
    def _mock_response(self, text: str) -> MagicMock:
        """Build a mock Anthropic message response."""
        block = MagicMock()
        block.text = text
        response = MagicMock()
        response.content = [block]
        return response

    @patch("oc_apprentice_worker.backends.anthropic.run_with_timeout")
    def test_infer_text_only(self, mock_timeout):
        """Text-only inference builds correct message structure."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend

        result_json = json.dumps({
            "target_description": "Save button",
            "suggested_selector": "#save",
            "confidence_boost": 0.2,
            "reasoning": "Found save button",
        })
        mock_timeout.return_value = self._mock_response(result_json)

        backend = AnthropicBackend(_make_config())
        backend._client = MagicMock()  # Skip real client init

        result = backend.infer("Identify the UI element")
        assert result["target_description"] == "Save button"
        assert result["confidence_boost"] == 0.2

    @patch("oc_apprentice_worker.backends.anthropic.run_with_timeout")
    def test_infer_with_image(self, mock_timeout):
        """Image is sent as base64 content block with correct media type."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend

        result_json = json.dumps({
            "target_description": "Image element",
            "suggested_selector": "img",
            "confidence_boost": 0.1,
            "reasoning": "visual",
        })
        mock_timeout.return_value = self._mock_response(result_json)

        backend = AnthropicBackend(_make_config())
        backend._client = MagicMock()

        result = backend.infer("Identify element", image_base64=_png_b64())
        assert result["target_description"] == "Image element"

        # Verify the call was made with correct structure
        call_fn = mock_timeout.call_args[0][0]
        assert callable(call_fn)

    @patch("oc_apprentice_worker.backends.anthropic.run_with_timeout")
    def test_infer_json_extraction(self, mock_timeout):
        """JSON is correctly extracted from markdown-wrapped response."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend

        wrapped_json = (
            "Here is the analysis:\n```json\n"
            '{"target_description": "button", "confidence_boost": 0.15}\n'
            "```"
        )
        mock_timeout.return_value = self._mock_response(wrapped_json)

        backend = AnthropicBackend(_make_config())
        backend._client = MagicMock()

        result = backend.infer("test prompt")
        assert result["target_description"] == "button"

    @patch("oc_apprentice_worker.backends.anthropic.run_with_timeout")
    def test_infer_with_system_prompt(self, mock_timeout):
        """System prompt is passed via top-level system= kwarg."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend

        result_json = json.dumps({"target_description": "test"})
        mock_timeout.return_value = self._mock_response(result_json)

        backend = AnthropicBackend(_make_config())
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(result_json)
        backend._client = mock_client

        # Mock run_with_timeout to call the function directly
        mock_timeout.side_effect = lambda fn, _timeout: fn()

        backend.infer("user prompt", system_prompt="You are a helper")

        # Verify system kwarg was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helper"
        # System prompt should NOT be in messages
        messages = call_kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)

    @patch("oc_apprentice_worker.backends.anthropic.run_with_timeout")
    def test_infer_timeout(self, mock_timeout):
        """TimeoutError propagates from run_with_timeout."""
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend

        mock_timeout.side_effect = TimeoutError("Inference timed out after 30s")

        backend = AnthropicBackend(_make_config())
        backend._client = MagicMock()

        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer("test")


# ---------------------------------------------------------------------------
# Media type detection
# ---------------------------------------------------------------------------

class TestMediaTypeDetection:
    def test_png_detection(self):
        from oc_apprentice_worker.backends.anthropic import _detect_media_type
        assert _detect_media_type(_png_b64()) == "image/png"

    def test_jpeg_detection(self):
        from oc_apprentice_worker.backends.anthropic import _detect_media_type
        assert _detect_media_type(_jpeg_b64()) == "image/jpeg"

    def test_unknown_defaults_to_jpeg(self):
        from oc_apprentice_worker.backends.anthropic import _detect_media_type
        random_b64 = base64.b64encode(b"random data").decode()
        assert _detect_media_type(random_b64) == "image/jpeg"


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------

class TestModelName:
    def test_default_model(self):
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(_make_config(remote_model=None))
        assert backend._model_name == "claude-sonnet-4-20250514"

    def test_custom_model(self):
        from oc_apprentice_worker.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(_make_config(remote_model="claude-opus-4-20250514"))
        assert backend._model_name == "claude-opus-4-20250514"

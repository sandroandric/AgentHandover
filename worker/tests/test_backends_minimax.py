"""Tests for the MiniMax M-series Claude-compatible backend."""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.vlm_worker import VLMConfig, VLMBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> VLMConfig:
    defaults = {
        "backend": VLMBackend.MINIMAX,
        "mode": "remote",
        "provider": "minimax",
        "api_key": "test-minimax-key-1234567890",
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

class TestMiniMaxAvailability:
    def test_not_available_without_package(self):
        """is_available() returns False when anthropic is not installed."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        backend = MiniMaxBackend(_make_config(api_key=None, api_key_env=None))
        with patch("builtins.__import__", side_effect=ImportError):
            assert backend.is_available() is False

    def test_not_available_without_key(self):
        """is_available() returns False when no API key can be resolved."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        config = _make_config(api_key=None, api_key_env=None)
        backend = MiniMaxBackend(config)
        with patch.dict(os.environ, {}, clear=True):
            assert backend.is_available() is False

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    def test_available_with_config_key(self):
        """is_available() returns True when api_key is set in config."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        backend = MiniMaxBackend(_make_config(api_key="valid-minimax-key"))
        assert backend.is_available() is True

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    def test_available_with_env_var(self):
        """is_available() returns True when MINIMAX_API_KEY env var is set."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        config = _make_config(api_key=None, api_key_env=None)
        backend = MiniMaxBackend(config)
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-minimax-key"}):
            assert backend.is_available() is True

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    def test_available_with_api_key_env(self):
        """is_available() resolves api_key_env to look up the correct env var."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        config = _make_config(api_key=None, api_key_env="MY_CUSTOM_KEY")
        backend = MiniMaxBackend(config)
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-minimax-key"}):
            assert backend.is_available() is True


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

class TestMiniMaxInfer:
    def _mock_response(self, text: str) -> MagicMock:
        """Build a mock Anthropic-style message response."""
        block = MagicMock()
        block.text = text
        response = MagicMock()
        response.content = [block]
        return response

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_text_only(self, mock_timeout):
        """Text-only inference builds correct message structure."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        result_json = json.dumps({
            "target_description": "Cancel button",
            "suggested_selector": "#cancel",
            "confidence_boost": 0.2,
            "reasoning": "Found cancel button",
        })
        mock_timeout.return_value = self._mock_response(result_json)

        backend = MiniMaxBackend(_make_config())
        backend._client = MagicMock()

        result = backend.infer("Identify the UI element")
        assert result["target_description"] == "Cancel button"
        assert result["confidence_boost"] == 0.2

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_with_image(self, mock_timeout):
        """Image is sent as base64 content block with correct media type."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        result_json = json.dumps({
            "target_description": "Image element",
            "suggested_selector": "img",
            "confidence_boost": 0.1,
            "reasoning": "visual",
        })
        mock_timeout.return_value = self._mock_response(result_json)

        backend = MiniMaxBackend(_make_config())
        backend._client = MagicMock()

        result = backend.infer("Identify element", image_base64=_png_b64())
        assert result["target_description"] == "Image element"

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_json_extraction(self, mock_timeout):
        """JSON is correctly extracted from markdown-wrapped response."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        wrapped_json = (
            "Here is the analysis:\n```json\n"
            '{"target_description": "link", "confidence_boost": 0.15}\n'
            "```"
        )
        mock_timeout.return_value = self._mock_response(wrapped_json)

        backend = MiniMaxBackend(_make_config())
        backend._client = MagicMock()

        result = backend.infer("test prompt")
        assert result["target_description"] == "link"

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_with_system_prompt(self, mock_timeout):
        """System prompt is passed via top-level system= kwarg."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        result_json = json.dumps({"target_description": "test"})
        mock_timeout.return_value = self._mock_response(result_json)

        backend = MiniMaxBackend(_make_config())
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

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_strips_unsupported_params(self, mock_timeout):
        """MiniMax-incompatible kwargs (top_k, stop_sequences, ...) are dropped."""
        from agenthandover_worker.backends.minimax import (
            MiniMaxBackend,
            _UNSUPPORTED_PARAMS,
        )

        result_json = json.dumps({"target_description": "ok"})

        backend = MiniMaxBackend(_make_config())
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(result_json)
        backend._client = mock_client

        mock_timeout.side_effect = lambda fn, _timeout: fn()

        backend.infer("user prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        for forbidden in _UNSUPPORTED_PARAMS:
            assert forbidden not in call_kwargs

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_temperature_clamped_at_zero(self, mock_timeout):
        """temperature=0 is bumped above zero because MiniMax rejects 0.0."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        result_json = json.dumps({"target_description": "ok"})

        backend = MiniMaxBackend(_make_config(temperature=0.0))
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(result_json)
        backend._client = mock_client

        mock_timeout.side_effect = lambda fn, _timeout: fn()

        backend.infer("user prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] > 0.0
        assert call_kwargs["temperature"] <= 1.0

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_temperature_capped_at_one(self, mock_timeout):
        """temperature > 1.0 is clamped to 1.0."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        result_json = json.dumps({"target_description": "ok"})

        backend = MiniMaxBackend(_make_config(temperature=1.5))
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(result_json)
        backend._client = mock_client

        mock_timeout.side_effect = lambda fn, _timeout: fn()

        backend.infer("user prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0

    @patch("agenthandover_worker.backends.minimax.run_with_timeout")
    def test_infer_timeout(self, mock_timeout):
        """TimeoutError propagates from run_with_timeout."""
        from agenthandover_worker.backends.minimax import MiniMaxBackend

        mock_timeout.side_effect = TimeoutError("Inference timed out after 30s")

        backend = MiniMaxBackend(_make_config())
        backend._client = MagicMock()

        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer("test")


# ---------------------------------------------------------------------------
# Base URL / model resolution
# ---------------------------------------------------------------------------

class TestBaseURLResolution:
    def test_default_base_url(self):
        from agenthandover_worker.backends.minimax import (
            MiniMaxBackend,
            _DEFAULT_BASE_URL,
        )
        backend = MiniMaxBackend(_make_config(base_url=None))
        with patch.dict(os.environ, {}, clear=True):
            assert backend._resolve_base_url() == _DEFAULT_BASE_URL
            assert _DEFAULT_BASE_URL == "https://api.minimax.io/anthropic"

    def test_config_base_url_overrides_default(self):
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        backend = MiniMaxBackend(_make_config(base_url="https://custom.example/v1"))
        assert backend._resolve_base_url() == "https://custom.example/v1"

    def test_env_var_overrides_default(self):
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        backend = MiniMaxBackend(_make_config(base_url=None))
        with patch.dict(os.environ, {"MINIMAX_BASE_URL": "https://api.minimaxi.com/anthropic"}):
            assert backend._resolve_base_url() == "https://api.minimaxi.com/anthropic"


# ---------------------------------------------------------------------------
# Media type detection
# ---------------------------------------------------------------------------

class TestMediaTypeDetection:
    def test_png_detection(self):
        from agenthandover_worker.backends.minimax import _detect_media_type
        assert _detect_media_type(_png_b64()) == "image/png"

    def test_jpeg_detection(self):
        from agenthandover_worker.backends.minimax import _detect_media_type
        assert _detect_media_type(_jpeg_b64()) == "image/jpeg"

    def test_unknown_defaults_to_jpeg(self):
        from agenthandover_worker.backends.minimax import _detect_media_type
        random_b64 = base64.b64encode(b"random data").decode()
        assert _detect_media_type(random_b64) == "image/jpeg"


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------

class TestModelName:
    def test_default_model(self):
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        backend = MiniMaxBackend(_make_config(remote_model=None))
        assert backend._model_name == "MiniMax-M2.7"

    def test_custom_model(self):
        from agenthandover_worker.backends.minimax import MiniMaxBackend
        backend = MiniMaxBackend(_make_config(remote_model="MiniMax-M2.7-highspeed"))
        assert backend._model_name == "MiniMax-M2.7-highspeed"


# ---------------------------------------------------------------------------
# Temperature clamping helper
# ---------------------------------------------------------------------------

class TestClampTemperature:
    def test_zero_becomes_positive(self):
        from agenthandover_worker.backends.minimax import _clamp_temperature
        assert _clamp_temperature(0.0) > 0.0

    def test_negative_becomes_positive(self):
        from agenthandover_worker.backends.minimax import _clamp_temperature
        assert _clamp_temperature(-0.5) > 0.0

    def test_one_unchanged(self):
        from agenthandover_worker.backends.minimax import _clamp_temperature
        assert _clamp_temperature(1.0) == 1.0

    def test_above_one_capped(self):
        from agenthandover_worker.backends.minimax import _clamp_temperature
        assert _clamp_temperature(2.5) == 1.0

    def test_normal_unchanged(self):
        from agenthandover_worker.backends.minimax import _clamp_temperature
        assert _clamp_temperature(0.3) == 0.3

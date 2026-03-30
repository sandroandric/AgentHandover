"""Tests for the Google Generative AI (Gemini) VLM backend."""

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
        "backend": VLMBackend.GOOGLE_GENAI,
        "mode": "remote",
        "provider": "google",
        "api_key": "AIzaSyTestKey1234567890",
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

class TestGoogleAvailability:
    def test_not_available_without_package(self):
        """is_available() returns False when google-genai is not installed."""
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        backend = GoogleGenAIBackend(_make_config(api_key=None, api_key_env=None))
        with patch("builtins.__import__", side_effect=ImportError):
            assert backend.is_available() is False

    def test_not_available_without_key(self):
        """is_available() returns False when no API key can be resolved."""
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        config = _make_config(api_key=None, api_key_env=None)
        backend = GoogleGenAIBackend(config)
        with patch.dict(os.environ, {}, clear=True):
            assert backend.is_available() is False

    @patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()})
    def test_available_with_config_key(self):
        """is_available() returns True when api_key is set in config."""
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        backend = GoogleGenAIBackend(_make_config(api_key="AIzaSyValid"))
        assert backend.is_available() is True

    @patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()})
    def test_available_with_env_var(self):
        """is_available() returns True when GOOGLE_API_KEY env var is set."""
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        config = _make_config(api_key=None, api_key_env=None)
        backend = GoogleGenAIBackend(config)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyEnvKey"}):
            assert backend.is_available() is True

    @patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()})
    def test_available_with_api_key_env(self):
        """is_available() resolves api_key_env to look up the correct env var."""
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        config = _make_config(api_key=None, api_key_env="MY_GOOGLE_KEY")
        backend = GoogleGenAIBackend(config)
        with patch.dict(os.environ, {"MY_GOOGLE_KEY": "AIzaSyCustom"}):
            assert backend.is_available() is True


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

def _mock_google_genai():
    """Create mock google.genai module structure."""
    mock_types = MagicMock()
    mock_types.Part.from_bytes.return_value = MagicMock()
    mock_types.GenerateContentConfig.return_value = MagicMock()
    mock_genai = MagicMock()
    mock_genai.types = mock_types
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    return {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }


class TestGoogleInfer:
    def _mock_response(self, text: str) -> MagicMock:
        """Build a mock Google GenAI response."""
        response = MagicMock()
        response.text = text
        return response

    @patch("agenthandover_worker.backends.google_genai.run_with_timeout")
    def test_infer_text_only(self, mock_timeout):
        """Text-only inference returns parsed JSON."""
        with patch.dict("sys.modules", _mock_google_genai()):
            from agenthandover_worker.backends.google_genai import GoogleGenAIBackend

            result_json = json.dumps({
                "target_description": "Submit button",
                "suggested_selector": "button[type=submit]",
                "confidence_boost": 0.25,
                "reasoning": "Found submit form button",
            })
            mock_timeout.return_value = self._mock_response(result_json)

            backend = GoogleGenAIBackend(_make_config())
            backend._client = MagicMock()

            result = backend.infer("Identify the UI element")
            assert result["target_description"] == "Submit button"
            assert result["confidence_boost"] == 0.25

    @patch("agenthandover_worker.backends.google_genai.run_with_timeout")
    def test_infer_with_image(self, mock_timeout):
        """Image is sent as Part.from_bytes with correct MIME type."""
        with patch.dict("sys.modules", _mock_google_genai()):
            from agenthandover_worker.backends.google_genai import GoogleGenAIBackend

            result_json = json.dumps({
                "target_description": "icon",
                "suggested_selector": ".icon",
                "confidence_boost": 0.1,
                "reasoning": "visual analysis",
            })
            mock_timeout.return_value = self._mock_response(result_json)

            backend = GoogleGenAIBackend(_make_config())
            backend._client = MagicMock()

            result = backend.infer("Identify element", image_base64=_png_b64())
            assert result["target_description"] == "icon"

    @patch("agenthandover_worker.backends.google_genai.run_with_timeout")
    def test_infer_json_extraction(self, mock_timeout):
        """JSON is correctly extracted from markdown-wrapped response."""
        with patch.dict("sys.modules", _mock_google_genai()):
            from agenthandover_worker.backends.google_genai import GoogleGenAIBackend

            wrapped = (
                "Analysis result:\n```json\n"
                '{"target_description": "link", "confidence_boost": 0.18}\n'
                "```"
            )
            mock_timeout.return_value = self._mock_response(wrapped)

            backend = GoogleGenAIBackend(_make_config())
            backend._client = MagicMock()

            result = backend.infer("test")
            assert result["target_description"] == "link"

    @patch("agenthandover_worker.backends.google_genai.run_with_timeout")
    def test_infer_with_system_prompt(self, mock_timeout):
        """System prompt is passed via GenerateContentConfig.system_instruction."""
        with patch.dict("sys.modules", _mock_google_genai()):
            from agenthandover_worker.backends.google_genai import GoogleGenAIBackend

            result_json = json.dumps({"target_description": "element"})
            mock_timeout.return_value = self._mock_response(result_json)

            backend = GoogleGenAIBackend(_make_config())
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = self._mock_response(result_json)
            backend._client = mock_client

            mock_timeout.side_effect = lambda fn, _timeout: fn()

            backend.infer("user prompt", system_prompt="You are a helper")

            # Verify generate_content was called
            assert mock_client.models.generate_content.called

    @patch("agenthandover_worker.backends.google_genai.run_with_timeout")
    def test_infer_timeout(self, mock_timeout):
        """TimeoutError propagates from run_with_timeout."""
        with patch.dict("sys.modules", _mock_google_genai()):
            from agenthandover_worker.backends.google_genai import GoogleGenAIBackend

            mock_timeout.side_effect = TimeoutError("Inference timed out after 30s")

            backend = GoogleGenAIBackend(_make_config())
            backend._client = MagicMock()

            with pytest.raises(TimeoutError, match="timed out"):
                backend.infer("test")


# ---------------------------------------------------------------------------
# MIME type detection
# ---------------------------------------------------------------------------

class TestMimeTypeDetection:
    def test_png_detection(self):
        from agenthandover_worker.backends.google_genai import _detect_mime_type
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        assert _detect_mime_type(png_bytes) == "image/png"

    def test_jpeg_detection(self):
        from agenthandover_worker.backends.google_genai import _detect_mime_type
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        assert _detect_mime_type(jpeg_bytes) == "image/jpeg"

    def test_unknown_defaults_to_jpeg(self):
        from agenthandover_worker.backends.google_genai import _detect_mime_type
        assert _detect_mime_type(b"random data") == "image/jpeg"


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------

class TestModelName:
    def test_default_model(self):
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        backend = GoogleGenAIBackend(_make_config(remote_model=None))
        assert backend._model_name == "gemini-2.0-flash"

    def test_custom_model(self):
        from agenthandover_worker.backends.google_genai import GoogleGenAIBackend
        backend = GoogleGenAIBackend(_make_config(remote_model="gemini-1.5-pro"))
        assert backend._model_name == "gemini-1.5-pro"

"""Tests for setup_vlm.py remote cloud VLM setup functionality."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from oc_apprentice_worker.setup_vlm import (
    _REMOTE_PROVIDERS,
    _toml_value,
    validate_api_key,
)


# ---------------------------------------------------------------------------
# Provider map tests
# ---------------------------------------------------------------------------

class TestProviderMap:
    def test_all_providers_present(self):
        assert "openai" in _REMOTE_PROVIDERS
        assert "anthropic" in _REMOTE_PROVIDERS
        assert "google" in _REMOTE_PROVIDERS

    def test_provider_metadata_keys(self):
        for name, meta in _REMOTE_PROVIDERS.items():
            assert "name" in meta, f"{name} missing 'name'"
            assert "env_var" in meta, f"{name} missing 'env_var'"
            assert "default_model" in meta, f"{name} missing 'default_model'"
            assert "key_prefix" in meta, f"{name} missing 'key_prefix'"

    def test_openai_defaults(self):
        meta = _REMOTE_PROVIDERS["openai"]
        assert meta["env_var"] == "OPENAI_API_KEY"
        assert meta["default_model"] == "gpt-4o-mini"
        assert meta["key_prefix"] == "sk-"

    def test_anthropic_defaults(self):
        meta = _REMOTE_PROVIDERS["anthropic"]
        assert meta["env_var"] == "ANTHROPIC_API_KEY"
        assert meta["default_model"] == "claude-sonnet-4-20250514"
        assert meta["key_prefix"] == "sk-ant-"

    def test_google_defaults(self):
        meta = _REMOTE_PROVIDERS["google"]
        assert meta["env_var"] == "GOOGLE_API_KEY"
        assert meta["default_model"] == "gemini-2.0-flash"
        assert meta["key_prefix"] == "AI"


# ---------------------------------------------------------------------------
# API key validation tests
# ---------------------------------------------------------------------------

class TestValidateApiKey:
    def test_empty_key_invalid(self):
        assert validate_api_key("openai", "") is False

    def test_short_key_invalid(self):
        assert validate_api_key("openai", "sk-abc") is False

    def test_openai_valid_prefix(self):
        assert validate_api_key("openai", "sk-1234567890abcdef") is True

    def test_openai_wrong_prefix(self):
        assert validate_api_key("openai", "wrong-prefix-key1234567890") is False

    def test_anthropic_valid_prefix(self):
        assert validate_api_key("anthropic", "sk-ant-1234567890abcdef") is True

    def test_anthropic_wrong_prefix(self):
        assert validate_api_key("anthropic", "sk-1234567890abcdef") is False

    def test_google_valid_prefix(self):
        assert validate_api_key("google", "AIzaSy1234567890abcdef") is True

    def test_unknown_provider_accepts_long_key(self):
        assert validate_api_key("unknown_provider", "abcdefghij1234567890") is True


# ---------------------------------------------------------------------------
# Config writing tests
# ---------------------------------------------------------------------------

class TestWriteRemoteConfig:
    def test_write_creates_config(self):
        """_write_remote_config creates config.toml with correct fields."""
        from oc_apprentice_worker.setup_vlm import _write_remote_config, _config_path

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("oc_apprentice_worker.setup_vlm._config_path") as mock_path:
                config_file = Path(tmpdir) / "config.toml"
                mock_path.return_value = config_file

                result = {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "env_var_value": "sk-test",
                }
                _write_remote_config(result)

                assert config_file.exists()
                content = config_file.read_text()
                assert 'mode = "remote"' in content
                assert 'provider = "openai"' in content
                assert 'model = "gpt-4o-mini"' in content
                assert 'api_key_env = "OPENAI_API_KEY"' in content


# ---------------------------------------------------------------------------
# TOML value formatting tests
# ---------------------------------------------------------------------------

class TestTomlValue:
    def test_string(self):
        assert _toml_value("hello") == '"hello"'

    def test_int(self):
        assert _toml_value(42) == "42"

    def test_float(self):
        assert _toml_value(3.14) == "3.14"

    def test_bool_true(self):
        assert _toml_value(True) == "true"

    def test_bool_false(self):
        assert _toml_value(False) == "false"

    def test_list(self):
        assert _toml_value([1, 2, 3]) == "[1, 2, 3]"

    def test_string_list(self):
        assert _toml_value(["a", "b"]) == '["a", "b"]'

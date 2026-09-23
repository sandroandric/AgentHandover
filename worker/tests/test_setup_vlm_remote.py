"""Tests for setup_vlm.py remote cloud VLM setup functionality."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agenthandover_worker.setup_vlm import (
    _REMOTE_PROVIDERS,
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
        from agenthandover_worker.setup_vlm import _write_remote_config, _config_path

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agenthandover_worker.setup_vlm._config_path") as mock_path:
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
# In-place [vlm] edit (#138): comments and [privacy.zones] survive
# ---------------------------------------------------------------------------

_USER_CONFIG = """# My AgentHandover config -- hand edited
[observer]
capture_screenshots = true   # keep frames

[vlm]
# which backend
mode = "local"
model = "qwen3.5:4b"
ollama_host = "http://localhost:11434"

[privacy]
enable_inline_secret_redaction = true

[privacy.zones]
# never look at the password manager
blocked = ["1password*"]
blocked_urls = ["*vault.example.com*"]
"""


class TestRemoteConfigInPlace:
    def _run(self, tmp_path: Path, text: str | None) -> str:
        from agenthandover_worker.setup_vlm import _write_remote_config

        config_file = tmp_path / "config.toml"
        if text is not None:
            config_file.write_text(text, encoding="utf-8")
        with patch("agenthandover_worker.setup_vlm._config_path", return_value=config_file):
            _write_remote_config({
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "env_var_value": "sk-test",
            })
        return config_file.read_text(encoding="utf-8")

    def test_other_sections_and_comments_are_byte_identical(self, tmp_path):
        import tomllib

        after = self._run(tmp_path, _USER_CONFIG)
        before_vlm, _, _ = _USER_CONFIG.partition("[vlm]")
        assert after.startswith(before_vlm), "everything above [vlm] is untouched"
        tail = _USER_CONFIG[_USER_CONFIG.index("[privacy]"):]
        assert after.endswith(tail), "[privacy] and [privacy.zones] with comments are untouched"

        cfg = tomllib.loads(after)
        assert cfg["vlm"] == {
            "mode": "remote",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "ollama_host": "http://localhost:11434",
        }
        assert cfg["privacy"]["zones"]["blocked"] == ["1password*"]
        assert "# which backend" in after

    def test_running_twice_does_not_duplicate_keys(self, tmp_path):
        import tomllib

        once = self._run(tmp_path, _USER_CONFIG)
        twice = self._run(tmp_path, once)
        assert twice == once
        tomllib.loads(twice)

    def test_keys_outside_vlm_are_not_touched(self, tmp_path):
        text = '[llm]\nmodel = "other"\nmode = "x"\n\n[vlm]\nmode = "local"\n'
        after = self._run(tmp_path, text)
        assert after.startswith('[llm]\nmodel = "other"\nmode = "x"\n\n[vlm]\n')

    def test_missing_vlm_section_is_appended(self, tmp_path):
        import tomllib

        text = "[privacy.zones]\nblocked = [\"1password*\"]"
        after = self._run(tmp_path, text)
        assert after.startswith(text + "\n")
        assert tomllib.loads(after)["vlm"]["mode"] == "remote"

    def test_missing_file_is_created(self, tmp_path):
        import tomllib

        assert tomllib.loads(self._run(tmp_path, None))["vlm"]["provider"] == "openai"

"""``main._skills_dir``: the one resolver for ClaudeSkillWriter's root."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agenthandover_worker.main import _skills_dir


def test_default_is_claude_skills_under_home():
    with patch("agenthandover_worker.main._config_path", return_value=None):
        assert _skills_dir() == Path.home() / ".claude" / "skills"


def test_reads_export_claude_skills_dir(tmp_path: Path):
    config = tmp_path / "config.toml"
    target = tmp_path / "skills-out"
    config.write_text(f'[export]\nclaude_skills_dir = "{target}"\n')
    with patch("agenthandover_worker.main._config_path", return_value=str(config)):
        assert _skills_dir() == target


def test_expands_tilde(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('[export]\nclaude_skills_dir = "~/agent-skills"\n')
    with patch("agenthandover_worker.main._config_path", return_value=str(config)):
        assert _skills_dir() == Path.home() / "agent-skills"


def test_missing_key_or_bad_toml_falls_back_to_default(tmp_path: Path):
    default = Path.home() / ".claude" / "skills"
    config = tmp_path / "config.toml"
    config.write_text('[export]\nadapter = "openclaw"\n')
    with patch("agenthandover_worker.main._config_path", return_value=str(config)):
        assert _skills_dir() == default
    config.write_text("[export\nnot toml")
    with patch("agenthandover_worker.main._config_path", return_value=str(config)):
        assert _skills_dir() == default

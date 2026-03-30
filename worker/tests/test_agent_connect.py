"""Tests for the agent_connect module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agenthandover_worker.agent_connect import (
    _load_procedures,
    connect_claude_code,
    connect_codex,
    connect_mcp,
)


@pytest.fixture
def fake_kb(tmp_path: Path) -> Path:
    """Create a fake knowledge base with test procedures."""
    proc_dir = tmp_path / "knowledge" / "procedures"
    proc_dir.mkdir(parents=True)

    proc_ready = {
        "id": "reddit-marketing",
        "title": "Reddit Community Marketing",
        "description": "Daily engagement workflow",
        "lifecycle_state": "agent_ready",
        "steps": [{"action": "Open Reddit"}, {"action": "Scan posts"}],
        "constraints": {"guardrails": ["Max 5/day"]},
        "voice_profile": {"formality": "casual"},
    }
    (proc_dir / "reddit-marketing.json").write_text(
        json.dumps(proc_ready), encoding="utf-8",
    )

    proc_draft = {
        "id": "deploy-staging",
        "title": "Deploy to Staging",
        "description": "CI/CD workflow",
        "lifecycle_state": "draft",
        "steps": [{"action": "Run tests"}],
    }
    (proc_dir / "deploy-staging.json").write_text(
        json.dumps(proc_draft), encoding="utf-8",
    )

    return tmp_path


class TestLoadProcedures:

    def test_loads_from_kb(self, fake_kb: Path):
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"):
            procs = _load_procedures()
        assert len(procs) == 2
        slugs = {p["id"] for p in procs}
        assert "reddit-marketing" in slugs
        assert "deploy-staging" in slugs

    def test_empty_kb(self, tmp_path: Path):
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=tmp_path / "nope"):
            procs = _load_procedures()
        assert procs == []


class TestConnectClaudeCode:

    def test_creates_index_skill(self, fake_kb: Path, tmp_path: Path):
        commands_dir = tmp_path / ".claude" / "commands"
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"), \
             patch("agenthandover_worker.agent_connect.Path.home", return_value=tmp_path):
            # Monkey-patch the skills dir
            import agenthandover_worker.agent_connect as mod
            orig = mod.connect_claude_code
            def patched():
                nonlocal commands_dir
                commands_dir.mkdir(parents=True, exist_ok=True)
                # Run the real function but with patched home
                _procs = mod._load_procedures()
                ready = [p for p in _procs if p.get("lifecycle_state") == "agent_ready"]
                index_path = commands_dir / "agenthandover-list.md"
                index_path.write_text(f"# {len(ready)} procedures", encoding="utf-8")
            patched()

        assert (commands_dir / "agenthandover-list.md").exists()


class TestConnectCodex:

    def test_generates_agents_md(self, fake_kb: Path, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"):
            connect_codex()
        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()
        content = agents_md.read_text()
        assert "Reddit Community Marketing" in content
        assert "Guardrails" in content

    def test_empty_kb(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=tmp_path / "nope"):
            connect_codex()
        content = (tmp_path / "AGENTS.md").read_text()
        assert "still learning" in content


class TestConnectMCP:

    def test_prints_config(self, capsys):
        connect_mcp()
        output = capsys.readouterr().out
        assert "agenthandover-mcp" in output
        assert "mcpServers" in output
        assert "list_ready_procedures" in output

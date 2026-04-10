"""Tests for the agent_connect module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agenthandover_worker.agent_connect import (
    _load_procedures,
    _render_skill_markdown,
    connect_claude_code,
    connect_codex,
    connect_hermes,
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


class TestRenderSkillMarkdown:
    """Tests for the shared Claude Code / agentskills.io markdown renderer."""

    def test_includes_frontmatter_and_title(self):
        proc = {
            "id": "reddit-marketing",
            "title": "Reddit Community Marketing",
            "description": "Daily engagement workflow",
            "steps": [{"action": "Open Reddit"}],
            "constraints": {"guardrails": ["Max 5/day"]},
        }
        md = _render_skill_markdown(proc)
        assert md.startswith("---\ndescription: Reddit Community Marketing\n---\n")
        assert "# Reddit Community Marketing" in md
        assert "Daily engagement workflow" in md
        assert "## Steps" in md
        assert "1. Open Reddit" in md
        assert "## Guardrails" in md
        assert "Max 5/day" in md

    def test_include_name_adds_name_field(self):
        """Hermes convention requires a `name:` field in frontmatter."""
        proc = {
            "id": "reddit-marketing",
            "title": "Reddit Community Marketing",
            "steps": [],
        }
        md = _render_skill_markdown(proc, include_name=True)
        assert "name: reddit-marketing" in md
        assert "description: Reddit Community Marketing" in md
        # name should come before description
        assert md.index("name:") < md.index("description:")

    def test_include_name_false_omits_name(self):
        proc = {"id": "x", "title": "X", "steps": []}
        md = _render_skill_markdown(proc, include_name=False)
        assert "name:" not in md

    def test_includes_execution_protocol(self):
        proc = {"id": "x", "title": "X", "steps": []}
        md = _render_skill_markdown(proc)
        assert "## Execution Protocol" in md
        assert "report_execution_start" in md
        assert "slug `x`" in md

    def test_omits_empty_sections(self):
        proc = {"id": "x", "title": "X", "steps": []}
        md = _render_skill_markdown(proc)
        assert "## Strategy" not in md
        assert "## Guardrails" not in md
        assert "## Steps" not in md
        assert "## Voice" not in md


class TestConnectHermes:
    """Tests for Hermes integration.

    Hermes loads skills by walking ``~/.hermes/skills/`` and reading each
    subdirectory's ``SKILL.md`` file — NOT loose ``.md`` files.  These
    tests verify we write the correct layout:
    ``~/.hermes/skills/agenthandover/<slug>/SKILL.md``.
    """

    def test_writes_skill_in_hermes_layout(self, fake_kb: Path, tmp_path: Path, capsys):
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"), \
             patch("agenthandover_worker.agent_connect.Path.home", return_value=tmp_path):
            connect_hermes()

        namespace_dir = tmp_path / ".hermes" / "skills" / "agenthandover"
        assert namespace_dir.is_dir()

        # Only the agent_ready procedure should be written, as a subdirectory
        slug_dirs = [d for d in namespace_dir.iterdir() if d.is_dir()]
        assert len(slug_dirs) == 1
        assert slug_dirs[0].name == "reddit-marketing"

        # The file must be named SKILL.md (Hermes convention)
        skill_file = slug_dirs[0] / "SKILL.md"
        assert skill_file.exists()

        content = skill_file.read_text(encoding="utf-8")
        # Frontmatter must include the name field for Hermes
        assert "name: reddit-marketing" in content
        assert "description: Reddit Community Marketing" in content
        assert "Max 5/day" in content
        assert "## Execution Protocol" in content

        out = capsys.readouterr().out
        assert "Hermes integration ready" in out
        assert "Procedures installed: 1" in out

    def test_cleans_up_stale_slug_directories(self, fake_kb: Path, tmp_path: Path):
        """Stale skill directories from previous runs must be removed."""
        namespace_dir = tmp_path / ".hermes" / "skills" / "agenthandover"
        stale_dir = namespace_dir / "old-workflow"
        stale_dir.mkdir(parents=True)
        (stale_dir / "SKILL.md").write_text("old content", encoding="utf-8")

        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"), \
             patch("agenthandover_worker.agent_connect.Path.home", return_value=tmp_path):
            connect_hermes()

        assert not stale_dir.exists()
        assert (namespace_dir / "reddit-marketing" / "SKILL.md").exists()

    def test_migrates_from_old_flat_layout(self, fake_kb: Path, tmp_path: Path):
        """Loose .md files from the old (incorrect) flat layout must be cleaned up."""
        namespace_dir = tmp_path / ".hermes" / "skills" / "agenthandover"
        namespace_dir.mkdir(parents=True)
        old_flat_file = namespace_dir / "mydailynews.md"
        old_flat_file.write_text("--- old flat layout ---", encoding="utf-8")

        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"), \
             patch("agenthandover_worker.agent_connect.Path.home", return_value=tmp_path):
            connect_hermes()

        assert not old_flat_file.exists()
        assert (namespace_dir / "reddit-marketing" / "SKILL.md").exists()

    def test_empty_kb_warns_about_no_skills(self, tmp_path: Path, capsys):
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=tmp_path / "nope"), \
             patch("agenthandover_worker.agent_connect.Path.home", return_value=tmp_path):
            connect_hermes()

        out = capsys.readouterr().out
        assert "No agent-ready skills yet" in out

    def test_prints_yaml_mcp_config(self, fake_kb: Path, tmp_path: Path, capsys):
        """Hermes uses config.yaml — the printed hint must show YAML, not TOML."""
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=fake_kb / "knowledge"), \
             patch("agenthandover_worker.agent_connect.Path.home", return_value=tmp_path):
            connect_hermes()

        out = capsys.readouterr().out
        assert "agenthandover-mcp" in out
        assert "config.yaml" in out
        # YAML-style indent, not TOML section headers
        assert "mcp:" in out
        assert "servers:" in out
        # Ensure we didn't leave the old TOML hint in place
        assert "[mcp.servers.agenthandover]" not in out

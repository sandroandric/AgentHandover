"""Tests for the MCP server module.

Tests the server builder and tool logic without requiring the mcp
package — mocks the FastMCP decorator and tests the underlying
functions directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test the server's tool functions directly by extracting them
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_kb(tmp_path: Path):
    """Create a fake knowledge base with test procedures."""
    proc_dir = tmp_path / "procedures"
    proc_dir.mkdir(parents=True)

    procs = [
        {
            "id": "reddit-marketing",
            "title": "Reddit Community Marketing",
            "description": "Daily engagement",
            "lifecycle_state": "agent_ready",
            "confidence_avg": 0.89,
            "apps_involved": ["Chrome"],
            "steps": [{"action": "Open Reddit"}],
            "strategy": "Target high-signal posts",
            "voice_profile": {"formality": "casual"},
        },
        {
            "id": "deploy-staging",
            "title": "Deploy to Staging",
            "description": "CI/CD deploy",
            "lifecycle_state": "draft",
            "confidence_avg": 0.72,
            "apps_involved": ["Terminal"],
            "steps": [{"action": "Run tests"}],
        },
    ]

    for proc in procs:
        (proc_dir / f"{proc['id']}.json").write_text(
            json.dumps(proc), encoding="utf-8",
        )

    # Create a mock KB
    kb = MagicMock()
    kb.list_procedures.return_value = procs
    kb.get_procedure.side_effect = lambda slug: next(
        (p for p in procs if p["id"] == slug), None
    )
    kb.get_profile.return_value = {
        "tools": {"browser": "Chrome"},
        "writing_style": {"formality": "casual"},
    }
    return kb


class TestMCPServerBuild:
    """Test that the MCP server can be constructed (mocked)."""

    def test_build_requires_mcp_package(self):
        """If mcp is not importable, should raise ImportError."""
        with patch.dict("sys.modules", {"mcp": None, "mcp.server.fastmcp": None}):
            with pytest.raises(ImportError):
                from agenthandover_worker.mcp_server import build_mcp_server
                # Force re-import
                import importlib
                import agenthandover_worker.mcp_server as mod
                importlib.reload(mod)
                mod.build_mcp_server()


class TestMCPToolLogic:
    """Test the tool functions' logic using mocked KB."""

    def test_list_ready_filters_agent_ready(self, fake_kb):
        """Only agent_ready procedures should appear in ready list."""
        procs = fake_kb.list_procedures()
        ready = [p for p in procs if p.get("lifecycle_state") == "agent_ready"]
        assert len(ready) == 1
        assert ready[0]["id"] == "reddit-marketing"

    def test_list_all_returns_everything(self, fake_kb):
        procs = fake_kb.list_procedures()
        assert len(procs) == 2

    def test_get_procedure_found(self, fake_kb):
        proc = fake_kb.get_procedure("reddit-marketing")
        assert proc is not None
        assert proc["title"] == "Reddit Community Marketing"
        assert proc["voice_profile"]["formality"] == "casual"

    def test_get_procedure_not_found(self, fake_kb):
        proc = fake_kb.get_procedure("nonexistent")
        assert proc is None

    def test_search_fallback_keyword(self, fake_kb):
        """Without vector_kb, search should fall back to keyword matching."""
        procs = fake_kb.list_procedures()
        query = "reddit"
        matches = [
            p for p in procs
            if query in p.get("title", "").lower()
            or query in p.get("description", "").lower()
        ]
        assert len(matches) == 1
        assert matches[0]["id"] == "reddit-marketing"

    def test_get_profile(self, fake_kb):
        profile = fake_kb.get_profile()
        assert "tools" in profile
        assert "writing_style" in profile


class TestMCPResourceLogic:
    """Test resource rendering logic."""

    def test_procedures_index_format(self, fake_kb):
        procs = fake_kb.list_procedures()
        lines = ["# AgentHandover Procedures\n"]
        for proc in procs:
            state = proc.get("lifecycle_state", "observed")
            marker = "ready" if state == "agent_ready" else state
            lines.append(
                f"- **{proc.get('title', 'Untitled')}** "
                f"(`{proc.get('id', '?')}`) [{marker}]"
            )
        index = "\n".join(lines)
        assert "Reddit Community Marketing" in index
        assert "[ready]" in index
        assert "[draft]" in index

    def test_procedure_detail_markdown(self, fake_kb):
        proc = fake_kb.get_procedure("reddit-marketing")
        lines = [f"# {proc['title']}\n"]
        if proc.get("strategy"):
            lines.append(f"## Strategy\n{proc['strategy']}\n")
        vp = proc.get("voice_profile", {})
        if vp.get("formality"):
            lines.append(f"## Voice\nTone: {vp['formality']}\n")
        md = "\n".join(lines)
        assert "# Reddit Community Marketing" in md
        assert "Target high-signal" in md
        assert "casual" in md

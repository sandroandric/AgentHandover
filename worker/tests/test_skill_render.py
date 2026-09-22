"""Tests for ``skill_render`` — the one renderer every agent integration uses.

- Golden file per target: the approved focus-session procedure rendered
  for each target is diffed against a reviewed file under
  ``fixtures/skill_packets/``.  Regenerate after an intended change with
  ``AH_UPDATE_GOLDENS=1`` and review the diff.
- Same packet: the Claude skill export (focus), the Claude command from
  ``connect claude-code`` (approve) and the MCP ``get_skill`` answer carry
  the same body.
- Frontmatter: YAML-escaped, per-target keys, ``allowed-tools`` from apps.
- Version: footer, ``/health`` and the installed package agree.
- Catalogues list every Skill on disk after a second export tick.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agenthandover_worker import __version__
from agenthandover_worker.procedure_schema import (
    PROCEDURE_SCHEMA_VERSION,
    sop_to_procedure,
)
from agenthandover_worker.skill_render import (
    TARGETS,
    allowed_tools,
    body_sections,
    description,
    frontmatter,
    render_body,
    render_packet,
    serialize_frontmatter,
)
from fixtures.vlm_answers import generated_sop_template

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "skill_packets"


def focus_procedure() -> dict:
    """The approved procedure from a focus session on the stored answer.

    Steps come from the generator run on a stored model answer; the
    fields a focus session adds afterwards (synthesis, Q&A answers,
    evidence, approval) are set the way the pipeline stores them.
    """
    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    proc["inputs"].append({
        "name": "namecheap_password", "type": "password", "required": False,
        "default": None, "credential": True,
    })
    proc.update({
        "lifecycle_state": "agent_ready",
        "source": "focus",
        "goal": "Find out whether a domain is free to register and what it costs.",
        "strategy": "Check the exact match first; look at alternatives only when it is taken.",
        "when_not_to_use": ["The domain is already registered to the user"],
        "branches": [{
            "condition": "the exact match is taken",
            "action": "note the Make offer price and stop",
        }],
        "agent_clarifications": [{
            "category": "guardrails",
            "question": "May the agent buy the domain?",
            "answer": "No, only report availability and price.",
            "context": "",
        }],
        "constraints": {
            "trust_level": "observe",
            "guardrails": ["Never add a domain to the cart or pay for anything"],
        },
        "voice_profile": {
            "formality": "casual", "style_confidence": "high", "sample_count": 4,
        },
        "evidence": {
            "observations": [{
                "date": "2026-09-01", "type": "focus", "duration_minutes": 3,
                "session_id": "focus-7f3a", "event_count": 42,
            }],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": 1,
        },
    })
    return proc


def _normalise(text: str) -> str:
    """Mask the release-dependent parts: the version and the body hash over it."""
    text = re.sub(r"generated_body_hash: sha256:[0-9a-f]{64}", "generated_body_hash: <HASH>", text)
    return text.replace(__version__, "<VERSION>")


def _body_of(packet: str) -> str:
    """The text after the frontmatter block."""
    assert packet.startswith("---\n")
    end = packet.index("\n---\n", 3)
    return packet[end + len("\n---\n"):].lstrip("\n")


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_golden_packet(target: str):
    rendered = _normalise(render_packet(focus_procedure(), target=target))
    golden = GOLDEN_DIR / f"{target}.md"
    if os.environ.get("AH_UPDATE_GOLDENS") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(rendered, encoding="utf-8")
    assert golden.exists(), f"missing golden {golden}; run with AH_UPDATE_GOLDENS=1"
    assert rendered == golden.read_text(encoding="utf-8")


def test_rendering_is_pure():
    proc = focus_procedure()
    snapshot = copy.deepcopy(proc)
    first = render_packet(proc, target="openclaw")
    assert proc == snapshot
    assert render_packet(proc, target="openclaw") == first


# ---------------------------------------------------------------------------
# One body for every target
# ---------------------------------------------------------------------------


def test_every_target_shares_the_body():
    proc = focus_procedure()
    body = render_body(proc)
    for target in TARGETS:
        packet = render_packet(proc, target=target)
        if target == "codex":
            assert packet.startswith("### Check Domain Availability\n")
            assert "\n#### Steps\n" in packet
            assert packet.replace("\n#### ", "\n## ").replace("### ", "# ", 1) == body
        else:
            assert _body_of(packet) == body, target


def test_focus_approve_and_mcp_read_the_same_packet(tmp_path: Path):
    """J-focus / J-approve / J-mcp at unit level: one body, three routes."""
    from agenthandover_worker.agent_connect import connect_claude_code
    from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.mcp_server import build_mcp_server

    proc = focus_procedure()
    kb_root = tmp_path / "knowledge"
    kb = KnowledgeBase(root=kb_root)
    kb.ensure_structure()
    kb.save_procedure(proc)

    # Focus: the pipeline exports the Claude Code skill.
    skill = ClaudeSkillWriter(skills_dir=tmp_path / "skills").write_procedure(proc)
    focus_body = _body_of(skill.read_text(encoding="utf-8"))

    # Approve: connect claude-code writes the Claude Code Skill.
    approve_skills = tmp_path / "approve-skills"
    with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=kb_root):
        connect_claude_code(commands_dir=tmp_path / "commands", skills_dir=approve_skills)
    approve_body = _body_of(
        (approve_skills / proc["id"] / "SKILL.md").read_text(encoding="utf-8")
    )

    # MCP: get_skill returns the packet.
    with patch("agenthandover_worker.mcp_server._get_kb_root", return_value=kb_root):
        server = build_mcp_server()
        answer = json.loads(server._tool_manager.get_tool("get_skill").fn(proc["id"]))
    mcp_body = _body_of(answer["packet"])

    assert focus_body == approve_body == mcp_body
    assert "1. [step_1] Open the Namecheap homepage" in focus_body
    assert "report_execution_start" in mcp_body


def test_codex_agents_md_uses_the_codex_packet(tmp_path: Path, monkeypatch):
    from agenthandover_worker.agent_connect import connect_codex
    from agenthandover_worker.knowledge_base import KnowledgeBase

    proc = focus_procedure()
    kb_root = tmp_path / "knowledge"
    kb = KnowledgeBase(root=kb_root)
    kb.ensure_structure()
    kb.save_procedure(proc)
    monkeypatch.chdir(tmp_path)
    with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=kb_root):
        connect_codex()
    agents_md = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert render_packet(kb.get_procedure(proc["id"]), target="codex") in agents_md


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def test_frontmatter_keys_per_target():
    proc = focus_procedure()
    assert list(frontmatter(proc, "claude-skill")) == [
        "name", "description", "argument-hint", "allowed-tools",
    ]
    assert list(frontmatter(proc, "claude-command")) == [
        "description", "argument-hint", "allowed-tools",
    ]
    assert list(frontmatter(proc, "hermes")) == ["name", "description"]
    assert list(frontmatter(proc, "mcp")) == ["name", "description"]
    assert frontmatter(proc, "codex") == {}
    # Credentials are named in the body, never taken as command arguments.
    assert frontmatter(proc, "claude-command")["argument-hint"] == "[domain_name]"
    fm = yaml.safe_load(render_packet(proc, target="openclaw").split("\n---\n")[0][4:])
    assert fm["generated_body_hash"].startswith("sha256:")
    with pytest.raises(ValueError):
        frontmatter(proc, "cursor")  # type: ignore[arg-type]


def test_titles_and_descriptions_are_yaml_escaped():
    proc = {
        "id": "Weird: Slug", "title": 'Report: "Q3" #1 - yes',
        "description": "Send: the 'weekly' report\nto #ops. Then relax.",
        "steps": [],
    }
    for target in ("claude-skill", "openclaw", "hermes"):
        packet = render_packet(proc, target=target)
        fm = yaml.safe_load(packet[4:packet.index("\n---\n", 3)])
        assert fm["description"] == "Send: the 'weekly' report to #ops."
        assert fm["name"] == "weird-slug"
    assert serialize_frontmatter({}) == ""


def test_description_priority():
    base = {"title": "Title here", "steps": []}
    assert description(base) == "Title here."
    assert description({**base, "when_to_use": "Every Monday. Also Fridays."}) == "Every Monday."
    assert description({**base, "task_description": "Do the thing. More."}) == "Do the thing."
    assert description({**base, "task_description": "T.", "description": "D."}) == "D."
    assert description({**base, "description": "D.", "goal": "G at example.com today."}) == (
        "G at example.com today."
    )


def test_allowed_tools_never_blanket_bash():
    browser = {"steps": [{"app": "Google Chrome"}], "apps_involved": []}
    assert allowed_tools(browser) == ["Read", "WebFetch"]
    terminal = {"steps": [{"parameters": {"app": "iTerm2"}}]}
    assert allowed_tools(terminal) == ["Bash", "Read"]
    unknown = {"steps": [{"app": "Slack"}], "apps_involved": ["Notion"]}
    assert allowed_tools(unknown) == ["Read"]
    editor = {"apps_involved": ["Visual Studio Code"], "steps": []}
    assert allowed_tools(editor) == ["Read", "Edit", "Grep", "Glob"]
    finder = {"apps_involved": ["Finder"], "steps": []}
    assert allowed_tools(finder) == ["Read", "Write", "Glob"]


# ---------------------------------------------------------------------------
# Body sections
# ---------------------------------------------------------------------------


def test_credentials_are_named_never_valued():
    proc = focus_procedure()
    proc["inputs"][-1]["example"] = "hunter2"
    text = "\n".join(body_sections(proc))
    assert "- `namecheap_password` (credential): take it from your credential store" in text
    assert "hunter2" not in text


def test_step_lines_route_on_shape_not_source():
    template_shape = {"id": "a", "title": "A", "source": "v2_focus_recording", "steps": [{
        "step": "Open the site", "parameters": {"app": "Safari", "location": "https://a.b",
                                                "verify": "Page loads"},
        "selector": "#go",
    }]}
    proc_shape = {"id": "a", "title": "A", "source": "anything", "steps": [{
        "action": "Open the site", "app": "Safari", "location": "https://a.b",
        "verify": "Page loads", "selector": "#go",
    }]}
    expected = (
        "1. [step_1] Open the site\n   App: Safari\n   Target: https://a.b\n"
        "   Verify: Page loads\n   Anchor: `#go`"
    )
    assert expected in "\n".join(body_sections(template_shape))
    assert expected in "\n".join(body_sections(proc_shape))


def test_empty_procedure_renders():
    text = render_packet({"id": "x", "title": "X"}, target="claude-skill")
    assert "## Steps\n\nNo steps were recorded." in text
    assert "## Guardrails" not in text
    assert "## Voice" not in text


# ---------------------------------------------------------------------------
# One version string
# ---------------------------------------------------------------------------


def test_footer_health_and_package_version_agree(tmp_path: Path):
    import urllib.request

    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.query_api import QueryAPIServer

    installed = importlib.metadata.version("agenthandover-worker")
    assert __version__ == installed

    packet = render_packet(focus_procedure(), target="claude-skill")
    assert packet.rstrip().endswith(
        f"*Generated by AgentHandover v{installed} "
        f"(procedure schema {PROCEDURE_SCHEMA_VERSION})*"
    )

    kb = KnowledgeBase(root=tmp_path / "kb")
    kb.ensure_structure()
    server = QueryAPIServer(kb, port=0)
    server.start()
    try:
        port = server._httpd.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            health = json.loads(resp.read())
    finally:
        server.stop()
    assert health["version"] == installed


# ---------------------------------------------------------------------------
# Catalogues list every Skill on disk (coverage audit #127)
# ---------------------------------------------------------------------------


def _second_procedure() -> dict:
    proc = focus_procedure()
    proc.update({"id": "check-domain-price", "title": "Check Domain Price"})
    return proc


def test_two_ticks_keep_both_skills_in_every_catalogue(tmp_path: Path):
    """J-approve at unit level: one Skill exported per tick, both listed."""
    from agenthandover_worker.agent_connect import connect_claude_code
    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.openclaw_writer import OpenClawWriter

    kb_root = tmp_path / "knowledge"
    kb = KnowledgeBase(root=kb_root)
    kb.ensure_structure()
    openclaw = OpenClawWriter(workspace_dir=tmp_path / "openclaw")
    commands = tmp_path / "commands"

    for proc in (focus_procedure(), _second_procedure()):  # one per tick
        kb.save_procedure(proc)
        openclaw.write_procedure(proc)
        with patch("agenthandover_worker.agent_connect._get_kb_root", return_value=kb_root):
            connect_claude_code(commands_dir=commands, skills_dir=tmp_path / "skills")

    catalogue = (openclaw.apprentice_dir / "index.md").read_text(encoding="utf-8")
    assert "**Total SOPs:** 2" in catalogue
    assert "`check-domain-availability`" in catalogue
    assert "`check-domain-price`" in catalogue

    listing = (commands / "agenthandover-list.md").read_text(encoding="utf-8")
    assert "`/check-domain-availability`" in listing
    assert "`/check-domain-price`" in listing
    assert sorted(p.parent.name for p in (tmp_path / "skills").glob("*/SKILL.md")) == [
        "check-domain-availability", "check-domain-price",
    ]


def test_openclaw_keeps_a_hand_edited_file(tmp_path: Path):
    from agenthandover_worker.openclaw_writer import OpenClawWriter

    writer = OpenClawWriter(workspace_dir=tmp_path)
    proc = focus_procedure()
    path = writer.write_procedure(proc)
    path.write_text(path.read_text(encoding="utf-8") + "\nMy own note.\n", encoding="utf-8")

    proc["strategy"] = "A newer strategy."
    draft = writer.write_procedure(proc)

    assert draft.name == f"sop.{proc['id']}.v2_draft.md"
    assert "My own note." in path.read_text(encoding="utf-8")
    assert "A newer strategy." in draft.read_text(encoding="utf-8")

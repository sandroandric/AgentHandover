"""Agent connection helper — sets up integration between AgentHandover and execution agents.

Usage::

    agenthandover connect claude-code   # Register skills as /slash-commands
    agenthandover connect codex         # Generate AGENTS.md for Codex
    agenthandover connect openclaw      # Verify OpenClaw path and sync
    agenthandover connect hermes        # Install skills into ~/.hermes/skills/
    agenthandover connect mcp           # Show MCP config for any agent
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agenthandover_worker.skill_render import description as skill_description
from agenthandover_worker.skill_render import render_packet, version_line


def _get_kb_root() -> Path:
    """Resolve knowledge base root.

    Must match ``knowledge_base.DEFAULT_KNOWLEDGE_DIR`` so we read the same
    directory that the worker writes to.
    """
    return Path.home() / ".agenthandover" / "knowledge"


def _load_procedures() -> list[dict]:
    """Load all procedures from the knowledge base."""
    kb_root = _get_kb_root()
    proc_dir = kb_root / "procedures"
    if not proc_dir.is_dir():
        return []

    procedures = []
    for f in sorted(proc_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            procedures.append(data)
        except (json.JSONDecodeError, OSError):
            pass
    return procedures


def connect_claude_code(commands_dir: Path | None = None) -> None:
    """Set up Claude Code integration.

    Creates:
    1. Individual /slash-commands per agent_ready procedure with full
       execution instructions (steps, strategy, guardrails, reporting).
    2. A meta-index /agenthandover-list command that lists all skills.
    3. Removes stale commands for procedures that are no longer agent_ready.

    Args:
        commands_dir: Where the command files go.  Defaults to
            ``~/.claude/commands``.
    """
    skills_dir = commands_dir if commands_dir is not None else Path.home() / ".claude" / "commands"
    skills_dir.mkdir(parents=True, exist_ok=True)

    procedures = _load_procedures()
    ready = [p for p in procedures if p.get("lifecycle_state") == "agent_ready"]

    # Track which command files we write so we can clean up stale ones
    written_files: set[str] = {"agenthandover-list.md"}

    # Write individual slash commands per procedure
    for proc in ready:
        filename = f"ah-{proc.get('id', '?')}.md"
        written_files.add(filename)
        (skills_dir / filename).write_text(
            render_packet(proc, target="claude-command"), encoding="utf-8",
        )

    # Generate index command
    index_path = skills_dir / "agenthandover-list.md"
    index_lines = [
        "---",
        "description: List all AgentHandover skills available for execution",
        "---",
        "",
        "# Available AgentHandover Skills",
        "",
    ]

    if ready:
        for proc in ready:
            slug = proc.get("id", "?")
            title = proc.get("title", "Untitled")
            desc = skill_description(proc)
            index_lines.append(f"- **{title}** (`/ah-{slug}`): {desc}")
    else:
        index_lines.append("No agent-ready skills yet. AgentHandover is still learning your workflows.")
        index_lines.append("")
        index_lines.append("Record a Focus Session to teach it a workflow, then approve the draft.")

    index_lines.append("")
    index_lines.append("---")
    index_lines.append(
        f"*{version_line()}. Run `agenthandover connect claude-code` to refresh.*"
    )

    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    # Clean up stale command files from previous runs
    for f in skills_dir.glob("ah-*.md"):
        if f.name not in written_files:
            f.unlink(missing_ok=True)

    print("Claude Code integration ready.")
    print(f"  Skills directory: {skills_dir}")
    print(f"  Index: {index_path}")
    print(f"  Procedures available: {len(ready)}")
    print()
    if ready:
        print("Your skills are available as /slash-commands:")
        for proc in ready:
            print(f"  /ah-{proc.get('id', '?')} — {proc.get('title', '')}")
    print()
    print("For richer integration, add the MCP server:")
    print("  agenthandover connect mcp")


def connect_codex() -> None:
    """Generate AGENTS.md for Codex in the current directory."""
    procedures = _load_procedures()
    ready = [p for p in procedures if p.get("lifecycle_state") == "agent_ready"]

    lines = [
        "# Agent Instructions (AgentHandover)",
        "",
        "The following procedures were learned by observing the user's workflows.",
        "Each procedure includes steps, strategy, guardrails, and voice guidance.",
        "",
        "## Available Procedures",
        "",
    ]

    for proc in ready:
        lines.append(render_packet(proc, target="codex"))

    if not ready:
        lines.append("No agent-ready procedures yet. AgentHandover is still learning.")
        lines.append("")

    lines.append("*For real-time access, use the AgentHandover REST API at localhost:9477*")
    lines.append("*or the MCP server: `agenthandover serve-mcp`*")

    output = Path("AGENTS.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Codex integration ready.")
    print(f"  Generated: {output.resolve()}")
    print(f"  Procedures: {len(ready)}")


def connect_openclaw() -> None:
    """Verify and configure OpenClaw integration."""
    import platform
    import tomllib

    openclaw_default = Path.home() / ".openclaw" / "workspace"

    # Read config to check custom path
    if platform.system() == "Darwin":
        config_path = Path.home() / "Library" / "Application Support" / "agenthandover" / "config.toml"
    else:
        config_path = Path.home() / ".config" / "agenthandover" / "config.toml"

    configured_path = openclaw_default
    if config_path.is_file():
        try:
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            wp = cfg.get("openclaw", {}).get("workspace_path", "")
            if wp:
                configured_path = Path(wp).expanduser()
        except Exception:
            pass

    sops_dir = configured_path / "memory" / "apprentice" / "sops"

    if sops_dir.is_dir():
        sop_count = len(list(sops_dir.glob("sop.*.md")))
        print(f"OpenClaw integration active.")
        print(f"  Workspace: {configured_path}")
        print(f"  SOPs directory: {sops_dir}")
        print(f"  Procedures synced: {sop_count}")
    elif configured_path.is_dir():
        print(f"OpenClaw workspace found at {configured_path}")
        print(f"  SOPs directory will be created when first procedure compiles.")
    else:
        print(f"OpenClaw workspace not found at {configured_path}")
        print()
        print("If OpenClaw is installed elsewhere, set the path in config.toml:")
        print()
        print("  [openclaw]")
        print('  workspace_path = "/path/to/openclaw/workspace"')


def connect_hermes() -> None:
    """Install AgentHandover skills into the Hermes skill library.

    Hermes (https://github.com/NousResearch/hermes-agent) loads skills
    by walking ``~/.hermes/skills/`` and reading each subdirectory's
    ``SKILL.md`` file (agentskills.io / anthropic-skills convention).
    The skill identity is taken from the ``name:`` frontmatter field
    (or the parent directory name as a fallback).

    Writes each agent-ready procedure to::

        ~/.hermes/skills/agenthandover/<slug>/SKILL.md

    The ``agenthandover/`` namespace groups our skills alongside
    Hermes's other source-namespaced skills (e.g. ``openclaw-imports/``).
    Stale ``<slug>/`` directories from previous runs are removed on each
    refresh.  Also prints the MCP config Hermes users can add for live
    semantic search and the execution-feedback loop.
    """
    import shutil

    namespace_dir = Path.home() / ".hermes" / "skills" / "agenthandover"
    namespace_dir.mkdir(parents=True, exist_ok=True)

    procedures = _load_procedures()
    ready = [p for p in procedures if p.get("lifecycle_state") == "agent_ready"]

    written_slugs: set[str] = set()
    for proc in ready:
        slug = proc.get("id", "?")
        slug_dir = namespace_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        (slug_dir / "SKILL.md").write_text(
            render_packet(proc, target="hermes"),
            encoding="utf-8",
        )
        written_slugs.add(slug)

    # Clean up stale skill directories from previous runs
    for entry in namespace_dir.iterdir():
        if entry.is_dir() and entry.name not in written_slugs:
            shutil.rmtree(entry, ignore_errors=True)
        # Also remove any loose .md files from the old (incorrect) flat layout
        elif entry.is_file() and entry.suffix == ".md":
            entry.unlink(missing_ok=True)

    hermes_root = Path.home() / ".hermes"
    hermes_exists = hermes_root.is_dir()

    print("Hermes integration ready.")
    print(f"  Skills namespace: {namespace_dir}")
    print(f"  Procedures installed: {len(ready)}")
    print()
    if not hermes_exists:
        print("Note: ~/.hermes not found — Hermes may not be installed yet.")
        print("  Install Hermes: https://github.com/NousResearch/hermes-agent")
        print("  Skills have been staged — they'll be picked up once you install.")
        print()
    if ready:
        print("In Hermes, your skills are available via:")
        print("  /skills                    — list all skills")
        for proc in ready:
            print(f"  /{proc.get('id', '?')}   — {proc.get('title', '')}")
    else:
        print("No agent-ready skills yet. AgentHandover is still learning your workflows.")
        print("Record a Focus Session and approve the draft to make it available.")
    print()
    print("For live semantic search + execution feedback, also add the MCP server")
    print("to Hermes. Edit ~/.hermes/config.yaml and add:")
    print()
    print("  mcp:")
    print("    servers:")
    print("      agenthandover:")
    print('        command: "agenthandover-mcp"')
    print()
    print("Or run `agenthandover connect mcp` to see the JSON-format config.")


def connect_mcp() -> None:
    """Print MCP configuration for any MCP-compatible agent."""
    config = {
        "mcpServers": {
            "agenthandover": {
                "command": "agenthandover-mcp",
            }
        }
    }

    print("MCP Server configuration")
    print("========================")
    print()
    print("Add this to your agent's MCP settings:")
    print()
    print("For Claude Code (~/.claude/settings.json):")
    print(json.dumps(config, indent=2))
    print()
    print("For Cursor/Windsurf (settings.json > mcpServers):")
    print(json.dumps(config["mcpServers"]["agenthandover"], indent=2))
    print()
    print("Available MCP tools:")
    print("  - list_ready_procedures  — executable procedures only")
    print("  - list_all_procedures    — all procedures with lifecycle state")
    print("  - get_procedure(slug)    — full procedure with steps + strategy + voice")
    print("  - search_procedures(q)   — semantic search by meaning")
    print("  - get_user_profile       — user tools, hours, writing style")
    print()
    print("Start the server manually:")
    print("  agenthandover-mcp        # stdio transport (for agents)")
    print("  agenthandover-mcp --sse  # SSE transport (for HTTP clients)")


def main():
    """Entry point for `agenthandover connect` command."""
    if len(sys.argv) < 2:
        print("Usage: agenthandover connect <agent>")
        print()
        print("Agents:")
        print("  claude-code   Register skills as /slash-commands")
        print("  codex         Generate AGENTS.md for Codex")
        print("  openclaw      Verify OpenClaw path and sync status")
        print("  hermes        Install skills into ~/.hermes/skills/")
        print("  mcp           Show MCP config for any MCP-compatible agent")
        sys.exit(1)

    agent = sys.argv[1].lower().replace("_", "-")

    handlers = {
        "claude-code": connect_claude_code,
        "claude": connect_claude_code,
        "codex": connect_codex,
        "openclaw": connect_openclaw,
        "claw": connect_openclaw,
        "hermes": connect_hermes,
        "mcp": connect_mcp,
    }

    handler = handlers.get(agent)
    if handler is None:
        print(f"Unknown agent: {agent}")
        print(f"Available: {', '.join(handlers.keys())}")
        sys.exit(1)

    handler()


if __name__ == "__main__":
    main()

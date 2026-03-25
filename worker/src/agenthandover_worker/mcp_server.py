"""MCP server — exposes AgentHandover procedures to any MCP-compatible agent.

Wraps the knowledge base and vector KB as MCP tools + resources so agents
like Claude Code, Cursor, Windsurf, or any MCP client can discover and
consume procedures without custom integration.

Usage::

    agenthandover serve-mcp          # stdio transport (default)
    agenthandover serve-mcp --sse    # SSE transport for HTTP clients

Configure in Claude Code settings.json::

    {
      "mcpServers": {
        "agenthandover": {
          "command": "agenthandover-mcp"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_kb_root() -> Path:
    """Resolve the knowledge base root directory."""
    import platform
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "agenthandover" / "knowledge"
    return Path.home() / ".agenthandover" / "knowledge"


def _get_db_path() -> Path:
    """Resolve the daemon database path."""
    import platform
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "agenthandover" / "agenthandover.db"
    return Path.home() / ".agenthandover" / "agenthandover.db"


def build_mcp_server():
    """Build and return a FastMCP server instance.

    Lazily loads knowledge base and vector KB on first tool call.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "MCP package not installed. Install with: pip install mcp"
        )

    mcp = FastMCP(
        "AgentHandover",
        description="Access your learned workflows, procedures, and handoff bundles. "
        "AgentHandover watches how you work and turns your workflows into "
        "step-by-step procedures that AI agents can follow.",
    )

    # Lazy-loaded singletons
    _state: dict = {}

    def _kb():
        if "kb" not in _state:
            from agenthandover_worker.knowledge_base import KnowledgeBase
            _state["kb"] = KnowledgeBase(root=_get_kb_root())
        return _state["kb"]

    def _vector_kb():
        if "vkb" not in _state:
            try:
                from agenthandover_worker.vector_kb import VectorKB
                db_path = _get_db_path()
                if db_path.exists():
                    _state["vkb"] = VectorKB(db_path)
                else:
                    _state["vkb"] = None
            except Exception:
                logger.warning("VectorKB init failed; semantic search disabled", exc_info=True)
                _state["vkb"] = None
        return _state["vkb"]

    # ------------------------------------------------------------------
    # Tools — agents call these to get data
    # ------------------------------------------------------------------

    @mcp.tool()
    def list_ready_skills() -> str:
        """List all Skills ready for agent execution.

        Returns only procedures that have passed all readiness gates
        (lifecycle=agent_ready, trust=execute, freshness check).
        If a procedure is in this list, an agent can execute it.
        """
        kb = _kb()
        procedures = kb.list_procedures()
        ready = []
        for proc in procedures:
            if proc.get("lifecycle_state") == "agent_ready":
                ready.append({
                    "slug": proc.get("id", ""),
                    "title": proc.get("title", ""),
                    "description": proc.get("description", ""),
                    "apps": proc.get("apps_involved", []),
                    "confidence": proc.get("confidence_avg", 0),
                })
        return json.dumps(ready, indent=2)

    @mcp.tool()
    def list_all_skills() -> str:
        """List ALL Skills with their lifecycle state.

        Includes drafts, observed, reviewed, and agent_ready procedures.
        Use this for discovery — not all of these are ready for execution.
        """
        kb = _kb()
        procedures = kb.list_procedures()
        result = []
        for proc in procedures:
            result.append({
                "slug": proc.get("id", ""),
                "title": proc.get("title", ""),
                "lifecycle_state": proc.get("lifecycle_state", "observed"),
                "confidence": proc.get("confidence_avg", 0),
                "apps": proc.get("apps_involved", []),
            })
        return json.dumps(result, indent=2)

    @mcp.tool()
    def get_skill(slug: str) -> str:
        """Get a full Skill by slug.

        Returns the complete procedure including steps, strategy,
        selection criteria, guardrails, voice profile, and content templates.
        This is everything an agent needs to execute the workflow.
        """
        kb = _kb()
        proc = kb.get_procedure(slug)
        if proc is None:
            return json.dumps({"error": f"Procedure '{slug}' not found"})
        # Append execution protocol so agents know how to report back
        proc["execution_protocol"] = {
            "instructions": (
                "When executing this Skill, report your progress using the "
                "AgentHandover MCP tools: call report_execution_start before "
                "beginning, report_step_result after each step, and "
                "report_execution_complete when finished. This improves "
                "the Skill for future runs."
            ),
            "tools": [
                "report_execution_start",
                "report_step_result",
                "report_execution_complete",
            ],
        }
        return json.dumps(proc, indent=2, default=str)

    @mcp.tool()
    def search_skills(query: str, limit: int = 5) -> str:
        """Semantic search — find Skills by meaning, not just keywords.

        Use this when you know what KIND of task you need but not the exact
        procedure name. For example: "deploy to production" will find
        procedures about deployment even if they're titled differently.
        """
        vkb = _vector_kb()
        if vkb is None:
            # Fallback to title/description search
            kb = _kb()
            procedures = kb.list_procedures()
            query_lower = query.lower()
            matches = []
            for proc in procedures:
                title = proc.get("title", "").lower()
                desc = proc.get("description", "").lower()
                if query_lower in title or query_lower in desc:
                    matches.append({
                        "slug": proc.get("id", ""),
                        "title": proc.get("title", ""),
                        "score": 1.0 if query_lower in title else 0.7,
                    })
            matches.sort(key=lambda m: m["score"], reverse=True)
            return json.dumps(matches[:limit], indent=2)

        results = vkb.search(
            query, top_k=limit, source_types=["procedure"], min_score=0.3,
        )
        return json.dumps(
            [{"slug": r.source_id, "score": r.score} for r in results],
            indent=2,
        )

    @mcp.tool()
    def get_user_profile() -> str:
        """Get the user's profile — tools, working hours, writing style.

        Use this to understand the user's preferences and adapt your
        behavior accordingly.
        """
        kb = _kb()
        return json.dumps(kb.get_profile(), indent=2, default=str)

    # ------------------------------------------------------------------
    # Execution reporting — agents call these during Skill execution
    # ------------------------------------------------------------------

    def _monitor():
        if "monitor" not in _state:
            from agenthandover_worker.execution_monitor import ExecutionMonitor
            _state["monitor"] = ExecutionMonitor(_kb())
        return _state["monitor"]

    @mcp.tool()
    def report_execution_start(slug: str, agent_id: str = "unknown") -> str:
        """Report that you are starting to execute a Skill.

        Call this BEFORE beginning execution. Returns an execution_id
        to use in subsequent step reports.
        """
        monitor = _monitor()
        execution_id = monitor.start_execution(slug, agent_id)
        kb = _kb()
        proc = kb.get_procedure(slug)
        expected_steps = []
        if proc:
            for s in proc.get("steps", []):
                expected_steps.append(s.get("step_id", s.get("action", "")[:40]))
        return json.dumps({
            "execution_id": execution_id,
            "slug": slug,
            "expected_steps": expected_steps,
        })

    @mcp.tool()
    def report_step_result(
        execution_id: str,
        step_id: str,
        status: str = "completed",
        actual_action: str = "",
        notes: str = "",
    ) -> str:
        """Report the result of a single step during Skill execution.

        Call after each step. Set status to 'completed' or 'deviated'.
        If deviated, describe what you actually did in actual_action.
        """
        monitor = _monitor()
        monitor.record_step(execution_id, step_id, actual_action or step_id)
        if status == "deviated" and notes:
            record = monitor._active.get(execution_id)
            if record:
                record.deviations.append({
                    "step_id": step_id,
                    "detail": notes,
                })
        return json.dumps({"recorded": True, "step_id": step_id})

    @mcp.tool()
    def report_execution_complete(
        execution_id: str,
        status: str = "completed",
        notes: str = "",
    ) -> str:
        """Report that Skill execution is finished.

        Call when done. Status: 'completed', 'failed', or 'aborted'.
        If failed, include the error in notes.
        """
        monitor = _monitor()
        if status == "failed":
            record = monitor.fail_execution(execution_id, notes or "unknown error")
        elif status == "aborted":
            record = monitor.abort_execution(execution_id)
        else:
            record = monitor.complete_execution(execution_id)

        # Trigger Skill improvement
        improvements = []
        try:
            from agenthandover_worker.skill_improver import SkillImprover
            improver = SkillImprover(_kb())
            improvements = improver.process_execution(record)
        except Exception:
            logger.debug("Skill improvement failed", exc_info=True)

        return json.dumps({
            "execution_id": execution_id,
            "final_status": record.status.value if hasattr(record.status, 'value') else str(record.status),
            "improvements": improvements,
        })

    # ------------------------------------------------------------------
    # Resources — agents can read these for context
    # ------------------------------------------------------------------

    @mcp.resource("agenthandover://procedures")
    def procedures_index() -> str:
        """Index of all learned procedures with status."""
        kb = _kb()
        procedures = kb.list_procedures()
        lines = ["# AgentHandover Procedures\n"]
        for proc in procedures:
            state = proc.get("lifecycle_state", "observed")
            marker = "ready" if state == "agent_ready" else state
            lines.append(
                f"- **{proc.get('title', 'Untitled')}** "
                f"(`{proc.get('id', '?')}`) [{marker}]"
            )
        return "\n".join(lines)

    @mcp.resource("agenthandover://procedures/{slug}")
    def procedure_detail(slug: str) -> str:
        """Full Skill as readable markdown."""
        kb = _kb()
        proc = kb.get_procedure(slug)
        if proc is None:
            return f"Procedure '{slug}' not found."

        lines = [f"# {proc.get('title', 'Untitled')}\n"]
        desc = proc.get("description", "")
        if desc:
            lines.append(f"{desc}\n")

        strategy = proc.get("strategy")
        if strategy:
            lines.append(f"## Strategy\n{strategy}\n")

        steps = proc.get("steps", [])
        if steps:
            lines.append("## Steps\n")
            for s in steps:
                action = s.get("action", s.get("description", ""))
                lines.append(f"1. {action}")
            lines.append("")

        vp = proc.get("voice_profile", {})
        if vp and vp.get("formality"):
            lines.append(f"## Voice\nTone: {vp['formality']}\n")

        return "\n".join(lines)

    @mcp.resource("agenthandover://profile")
    def user_profile_resource() -> str:
        """User profile as readable text."""
        kb = _kb()
        profile = kb.get_profile()
        return json.dumps(profile, indent=2, default=str)

    return mcp


def main():
    """Entry point for `agenthandover-mcp` command."""
    import sys

    logging.basicConfig(level=logging.WARNING)

    server = build_mcp_server()

    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"

    server.run(transport=transport)


if __name__ == "__main__":
    main()

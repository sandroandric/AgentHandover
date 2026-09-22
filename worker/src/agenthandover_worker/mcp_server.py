"""MCP server — exposes AgentHandover procedures to any MCP-compatible agent.

Wraps the knowledge base and vector KB as MCP tools + resources so agents
like Claude Code, Cursor, Windsurf, or any MCP client can discover and
consume procedures without custom integration.

Usage::

    agenthandover-mcp                # stdio transport (default)
    agenthandover-mcp --sse          # SSE transport for HTTP clients

Pair Claude Code with ``agenthandover connect claude-code`` (it runs
``claude mcp add --scope user --transport stdio agenthandover -- <path>``);
``agenthandover connect mcp`` prints the config for other agents.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker import paths

logger = logging.getLogger(__name__)


def _get_kb_root() -> Path:
    """The knowledge base the worker writes (``paths.knowledge_base_dir()``)."""
    return paths.knowledge_base_dir()


def _get_db_path() -> Path:
    """The worker's database, which holds the vector store (``paths.db_path()``)."""
    return paths.db_path()


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
        instructions="Access your learned workflows, procedures, and handoff bundles. "
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
        """``(VectorKB | None, reason)``; the reason says why it is unavailable."""
        if "vkb" not in _state:
            db_path = _get_db_path()
            if not db_path.exists():
                _state["vkb"] = (None, f"no worker database at {db_path}")
            else:
                try:
                    from agenthandover_worker.config import load_config
                    from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig

                    _state["vkb"] = (VectorKB(db_path, VectorKBConfig(
                        ollama_host=load_config().vlm.ollama_host,
                    )), "")
                except Exception as exc:
                    logger.warning("VectorKB init failed", exc_info=True)
                    _state["vkb"] = (None, f"vector store failed to open: {exc}")
            if _state["vkb"][0] is None:
                logger.warning(
                    "Semantic search unavailable (%s); search_skills uses keyword matching",
                    _state["vkb"][1],
                )
        return _state["vkb"]

    def _keyword_search(query: str, limit: int) -> list[dict]:
        query_lower = query.lower()
        matches = []
        for proc in _kb().list_procedures():
            title = proc.get("title", "").lower()
            desc = proc.get("description", "").lower()
            if query_lower in title or query_lower in desc:
                matches.append({
                    "slug": proc.get("id", ""),
                    "title": proc.get("title", ""),
                    "score": 1.0 if query_lower in title else 0.7,
                })
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches[:limit]

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

        Returns the procedure (steps, strategy, selection criteria,
        guardrails, voice profile, content templates) plus ``packet``:
        the same Skill text every other agent integration receives,
        including the execution protocol for reporting progress back.
        """
        from agenthandover_worker.skill_render import render_packet

        kb = _kb()
        proc = kb.get_procedure(slug)
        if proc is None:
            return json.dumps({"error": f"Procedure '{slug}' not found"})
        proc["packet"] = render_packet(proc, target="mcp")
        return json.dumps(proc, indent=2, default=str)

    @mcp.tool()
    def search_skills(query: str, limit: int = 5) -> str:
        """Semantic search — find Skills by meaning, not just keywords.

        Use this when you know what KIND of task you need but not the exact
        procedure name. For example: "deploy to production" will find
        procedures about deployment even if they're titled differently.

        Returns ``{"mode", "reason", "results"}``: ``mode`` is
        ``"semantic"`` or ``"keyword_fallback"`` (a substring match on
        titles and descriptions), and ``reason`` says why semantic search
        was not used.
        """
        from agenthandover_worker.ollama import OllamaError

        vkb, reason = _vector_kb()
        if vkb is not None:
            if vkb.count("procedure") == 0:
                reason = "no Skill has been embedded yet"
            else:
                try:
                    vectors = vkb.compute_embeddings([query])
                except (ConnectionError, OllamaError) as exc:
                    reason = f"embedding model unavailable: {exc}"
                    logger.warning("search_skills: %s", reason)
                else:
                    if vectors and vectors[0]:
                        hits = vkb.search_by_vector(
                            vectors[0], top_k=limit,
                            source_types=["procedure"], min_score=0.3,
                        )
                        titles = {
                            p.get("id", ""): p.get("title", "")
                            for p in _kb().list_procedures()
                        }
                        return json.dumps({
                            "mode": "semantic",
                            "reason": "",
                            "results": [
                                {
                                    "slug": r.source_id,
                                    "title": titles.get(r.source_id, ""),
                                    "score": round(r.score, 4),
                                }
                                for r in hits
                            ],
                        }, indent=2)
                    reason = "embedding model returned no vector"
        return json.dumps({
            "mode": "keyword_fallback",
            "reason": reason,
            "results": _keyword_search(query, limit),
        }, indent=2)

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
        to use in subsequent step reports, and the step ids to report
        (the ids the Skill shows in brackets).
        """
        from agenthandover_worker.procedure_schema import procedure_steps, step_key

        monitor = _monitor()
        execution_id = monitor.start_execution(slug, agent_id)
        proc = _kb().get_procedure(slug) or {}
        return json.dumps({
            "execution_id": execution_id,
            "slug": slug,
            "expected_steps": [
                step_key(s, i) for i, s in enumerate(procedure_steps(proc))
            ],
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

        Call after each step with its id (``step_2``) or its number (``2``).
        Set status to 'completed' or 'deviated'. If you did something
        other than the step says, describe what you did in actual_action.
        """
        step = _monitor().record_step(
            execution_id,
            step_id,
            actual_action,
            deviated=status == "deviated",
            notes=notes,
        )
        if step is None:
            return json.dumps({
                "recorded": False,
                "error": f"No active execution '{execution_id}'",
            })
        return json.dumps({
            "recorded": True,
            "step_id": step.step_id,
            "deviated": step.status.value == "deviated",
        })

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
        try:
            if status == "failed":
                record = monitor.fail_execution(execution_id, notes or "unknown error")
            elif status == "aborted":
                record = monitor.abort_execution(execution_id)
            else:
                record = monitor.complete_execution(execution_id)
        except KeyError:
            return json.dumps({
                "execution_id": execution_id,
                "error": f"No active execution '{execution_id}'",
            })

        from agenthandover_worker.skill_improver import SkillImprover

        improvements = SkillImprover(_kb(), monitor).process_execution(record)
        return json.dumps({
            "execution_id": execution_id,
            "final_status": record.status.value,
            "deviations": len(record.deviations),
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

        from agenthandover_worker.procedure_schema import procedure_steps, step_key

        steps = procedure_steps(proc)
        if steps:
            lines.append("## Steps\n")
            for i, s in enumerate(steps):
                action = s.get("action") or s.get("description", "")
                lines.append(f"{i + 1}. [{step_key(s, i)}] {action}")
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

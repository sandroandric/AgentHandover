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
from pathlib import Path

from agenthandover_worker import paths
from agenthandover_worker.commonplace import (  # noqa: F401  (the timeouts: re-exported)
    EMBED_RETRY_AFTER_S,
    MCP_INSTRUCTIONS,
    QUERY_EMBED_TIMEOUT_S,
    Commonplace,
)

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

    mcp = FastMCP("AgentHandover", instructions=MCP_INSTRUCTIONS)

    # Lazy-loaded singletons
    _state: dict = {}

    def _kb():
        if "kb" not in _state:
            from agenthandover_worker.knowledge_base import KnowledgeBase
            _state["kb"] = KnowledgeBase(root=_get_kb_root(), db_path=_get_db_path())
        return _state["kb"]

    def _vector_kb():
        """``(VectorKB | None, reason)``; the reason says why it is unavailable.

        Re-opened when ``[embedding] model`` changed since it was opened:
        "Upgrade everything" switches the worker's vectors while an MCP
        session may be running, and a search with the old model would find
        nothing (search filters rows by model)."""
        try:
            from agenthandover_worker.config import load_config

            model = load_config().embedding.model
        except Exception:
            model = _state.get("vkb_model")
        if "vkb" in _state and model != _state.get("vkb_model"):
            old = _state.pop("vkb")[0]
            if old is not None:
                try:
                    old.close()
                except Exception:
                    logger.debug("Closing the old VectorKB failed", exc_info=True)
        if "vkb" not in _state:
            _state["vkb_model"] = model
            db_path = _get_db_path()
            if not db_path.exists():
                _state["vkb"] = (None, f"no worker database at {db_path}")
            else:
                try:
                    from agenthandover_worker.config import load_config
                    from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig

                    cfg = load_config()
                    _state["vkb"] = (VectorKB(db_path, VectorKBConfig(
                        embedding_model=cfg.embedding.model,
                        ollama_host=cfg.vlm.ollama_host,
                        timeout_seconds=QUERY_EMBED_TIMEOUT_S,
                    )), "")
                except Exception as exc:
                    logger.warning("VectorKB init failed", exc_info=True)
                    _state["vkb"] = (None, f"vector store failed to open: {exc}")
            if _state["vkb"][0] is None:
                logger.warning(
                    "Semantic search unavailable (%s); Skill and day search use keyword matching",
                    _state["vkb"][1],
                )
        return _state["vkb"]

    def _memory_index(cfg):
        """The Memory index for ``[memory]``'s model (one per model)."""
        from agenthandover_worker.memory import MemoryIndex, query_embedder

        key = (cfg.memory.model, cfg.memory.dimensions, cfg.vlm.ollama_host)
        if _state.get("memory_key") != key:
            if "memory" in _state:
                _state.pop("memory").close()
            _state["memory"] = MemoryIndex(
                _get_db_path(), model=cfg.memory.model, dim=cfg.memory.dimensions,
                embed_query=query_embedder(
                    cfg.vlm.ollama_host, cfg.memory.model, cfg.memory.dimensions,
                    timeout=QUERY_EMBED_TIMEOUT_S,
                ),
            )
            _state["memory_key"] = key
        return _state["memory"]

    def _memory_search(cfg, query: str, *, keyword: bool, **filters) -> list[dict]:
        from agenthandover_worker.memory import keyword_search

        index = _memory_index(cfg)
        if keyword:
            return keyword_search(index, query, **filters)
        return index.search(query, **filters)

    def _load_config():
        from agenthandover_worker.config import load_config

        return load_config()  # read on every call: the agents switch applies at once

    commonplace = Commonplace(
        knowledge_base=_kb,
        vector_kb=_vector_kb,
        load_config=_load_config,
        memory=_memory_search,
        db_path=lambda: _get_db_path(),
        embed_timeout=QUERY_EMBED_TIMEOUT_S,
    )

    # ------------------------------------------------------------------
    # Tools — agents call these to get data
    # ------------------------------------------------------------------

    @mcp.tool()
    def search_commonplace(
        query: str,
        kinds: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search Commonplace: everything AgentHandover knows about the user's work.

        Use this before asking the user about their own work, tools, habits
        or something they saw.  *kinds* picks what to search, any of
        ``skills`` (workflows learned from the user), ``memory`` (what was
        on their screen; only when the user lets agents search Memory),
        ``days`` (daily summaries) and ``profile`` (tools, working hours,
        writing style; returned whole); default skills, memory and days.
        *since* / *until* (ISO-8601) keep Memory moments and days in that
        range; *limit* is per kind (1-50).

        Returns ``{"query", <kind>: {"mode", "reason", "results"}}``, one
        group per kind, each ranked by its own scorer (scores of different
        kinds are not comparable).  ``mode`` is ``"semantic"``,
        ``"keyword_fallback"`` (``reason`` says why), ``"not_allowed"``
        (``reason`` says how the user can allow it) or ``"whole"`` (the
        profile).  A Skill carries ``handoff``: ``ready`` (you may run
        it), ``held_back`` (with ``reasons``) or ``not_approved``.  Text
        only: never images, vectors or file paths.
        """
        try:
            found = commonplace.search(query, kinds, since, until, limit)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(found, indent=2, default=str)

    @mcp.tool()
    def list_ready_skills() -> str:
        """List the Skills an agent may run, and the approved ones held back.

        Returns ``{"ready": [...], "held_back": [...]}``.  A Skill is in
        ``ready`` only when all four checks of ``ledger.handoff_gate`` pass:
        a human approved it (lifecycle ``agent_ready``); none of its required
        ledger slots (goal, steps, done criteria) is unknown; its trust level
        is above ``observe``; its freshness is at least the floor.  An
        approved Skill failing any check is listed in ``held_back`` with the
        failing ``reasons``.  Slots still unknown are listed per Skill in
        ``not_yet_known``: ask the user about those, never assume them.
        """
        from agenthandover_worker.ledger import handoff_gate, unknown_slots

        ready, held_back = [], []
        for proc in _kb().list_procedures():
            if proc.get("lifecycle_state") != "agent_ready":
                continue
            gate = handoff_gate(proc)
            if not gate.ok:
                held_back.append({
                    "slug": proc.get("id", ""),
                    "title": proc.get("title", ""),
                    "reasons": gate.reasons,
                })
                continue
            ready.append({
                "slug": proc.get("id", ""),
                "title": proc.get("title", ""),
                "description": proc.get("description", ""),
                "apps": proc.get("apps_involved", []),
                "confidence": proc.get("confidence_avg", 0),
                "not_yet_known": unknown_slots(proc),
            })
        return json.dumps({"ready": ready, "held_back": held_back}, indent=2)

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
        ``"semantic"`` or ``"keyword_fallback"`` (query words matched in
        titles, descriptions, tags and steps), and ``reason`` says why
        semantic search was not used.
        """
        mode, reason, hits = commonplace.skills(query, limit)
        return json.dumps({
            "mode": mode,
            "reason": reason,
            "results": [
                {"slug": proc.get("id", ""), "title": proc.get("title", ""), "score": score}
                for proc, score in hits
            ],
        }, indent=2)

    @mcp.tool()
    def search_memory(
        query: str,
        since: str | None = None,
        until: str | None = None,
        app: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search the user's Memory: what was on their screen, by meaning.

        Use this to find something the user saw or worked on ("the invoice
        from Acme last week", "the error in the terminal this morning").
        *since* / *until* are ISO-8601 times; *app* keeps one app.

        Returns ``{"mode", "reason", "results"}``: each result has ``app``,
        ``window_title``, ``url_host``, ``description``, ``first_ts``,
        ``last_ts`` and ``score`` (never images).  ``mode`` is
        ``"semantic"``, ``"keyword_fallback"`` (words matched in the
        descriptions; ``reason`` says why), or ``"not_allowed"`` when the
        user has not let agents search Memory.
        """
        from agenthandover_worker.memory_forget import parse_time

        try:
            since_at = parse_time(since, "since") if since else None
            until_at = parse_time(until, "until") if until else None
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        found = commonplace.memory(
            query, since=since_at, until=until_at, app=app,
            limit=max(1, min(int(limit), 50)),
        )
        return json.dumps(found, indent=2)

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


def main(argv: list[str] | None = None):
    """Entry point for `agenthandover-mcp` command."""
    import argparse

    from agenthandover_worker import __version__

    parser = argparse.ArgumentParser(
        prog="agenthandover-mcp",
        description=(
            "Serve the Skills AgentHandover learned to an MCP client. "
            "Speaks MCP over stdio by default; an MCP client (Claude Code, "
            "Cursor, ...) starts this command itself."
        ),
    )
    parser.add_argument(
        "--sse", action="store_true",
        help="serve over HTTP server-sent events instead of stdio",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    server = build_mcp_server()
    server.run(transport="sse" if args.sse else "stdio")


if __name__ == "__main__":
    main()

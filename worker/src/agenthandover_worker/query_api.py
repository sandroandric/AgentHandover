"""Agent Query API — local HTTP server for read-only knowledge base access.

AI agents query this server to read procedures, profile, decisions,
triggers, constraints, context, and daily summaries from the knowledge
base.  An optional ``ActivitySearcher`` can be provided for full-text
search over VLM annotations.

The server binds to ``127.0.0.1`` only (no external access) and runs
in a daemon thread so it can be started alongside the worker process.

Usage::

    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.query_api import QueryAPIServer

    kb = KnowledgeBase()
    server = QueryAPIServer(kb, port=9477)
    server.start()
    # ... server is now accepting requests ...
    server.stop()
"""

from __future__ import annotations

import http.server
import json
import logging
import re
import threading
from dataclasses import asdict
from typing import Any

from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"

# URL patterns
_PROCEDURE_SLUG_RE = re.compile(r"^/procedures/([a-zA-Z0-9_-]+)$")
_CONTEXT_NAME_RE = re.compile(r"^/context/([a-zA-Z0-9_-]+)$")
_DAILY_DATE_RE = re.compile(r"^/daily/(\d{4}-\d{2}-\d{2})$")
_BUNDLE_SLUG_RE = re.compile(r"^/bundle/([a-zA-Z0-9_-]+)$")


def _compute_freshness(proc: dict) -> float:
    """Compute freshness score, importing lazily to avoid circular imports."""
    try:
        from agenthandover_worker.staleness_detector import procedure_freshness
        return procedure_freshness(proc)
    except Exception:
        return 1.0


class QueryAPIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the knowledge base query API.

    Each handler method reads from ``self.server.knowledge_base``.
    ``POST /search`` reads from ``self.server.activity_searcher`` if
    available.
    """

    # Suppress default stderr logging — route through Python logger instead
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("QueryAPI %s %s", self.client_address[0], format % args)

    # ------------------------------------------------------------------
    # GET dispatcher
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        """Route GET requests to the appropriate handler."""
        path = self.path.split("?")[0]  # strip query string

        try:
            if path == "/health":
                self._handle_health()
            elif path == "/procedures":
                self._handle_procedures_list()
            elif _PROCEDURE_SLUG_RE.match(path):
                slug = _PROCEDURE_SLUG_RE.match(path).group(1)  # type: ignore[union-attr]
                self._handle_procedure_detail(slug)
            elif path == "/profile":
                self._handle_profile()
            elif path == "/decisions":
                self._handle_decisions()
            elif path == "/triggers":
                self._handle_triggers()
            elif path == "/constraints":
                self._handle_constraints()
            elif _CONTEXT_NAME_RE.match(path):
                name = _CONTEXT_NAME_RE.match(path).group(1)  # type: ignore[union-attr]
                self._handle_context(name)
            elif _DAILY_DATE_RE.match(path):
                date = _DAILY_DATE_RE.match(path).group(1)  # type: ignore[union-attr]
                self._handle_daily(date)
            elif _BUNDLE_SLUG_RE.match(path):
                slug = _BUNDLE_SLUG_RE.match(path).group(1)  # type: ignore[union-attr]
                self._handle_bundle(slug)
            elif path == "/ready":
                self._handle_ready()
            elif path == "/available":
                self._handle_available()
            elif path == "/curation/queue":
                self._handle_curation_queue()
            elif path == "/curation/merges":
                self._handle_curation_merges()
            elif path == "/curation/upgrades":
                self._handle_curation_upgrades()
            elif path == "/curation/families":
                self._handle_curation_families()
            elif path == "/curation/summary":
                self._handle_curation_summary()
            elif path.startswith("/validate/"):
                slug = path[len("/validate/"):]
                if slug:
                    self._handle_validate(slug)
                else:
                    self._send_error(400, "Missing slug")
            elif path.startswith("/curation/drift/"):
                slug = path[len("/curation/drift/"):]
                if slug:
                    self._handle_curation_drift(slug)
                else:
                    self._send_error(400, "Missing slug")
            elif path == "/health/detailed":
                self._handle_health_detailed()
            elif path == "/telemetry/trend":
                self._handle_telemetry_trend()
            elif path == "/version":
                self._handle_version()
            else:
                self._send_error(404, f"Not found: {path}")
        except Exception:
            logger.exception("Internal error handling GET %s", path)
            self._send_error(500, "Internal server error")

    # ------------------------------------------------------------------
    # POST dispatcher
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        """Route POST requests to the appropriate handler."""
        path = self.path.split("?")[0]

        try:
            if path == "/search":
                self._handle_search()
            elif path == "/curation/merge":
                self._handle_curation_merge_action()
            elif path == "/curation/promote":
                self._handle_curation_promote()
            elif path == "/curation/demote":
                self._handle_curation_demote()
            elif path == "/curation/archive":
                self._handle_curation_archive()
            elif path == "/curation/dismiss-merge":
                self._handle_curation_dismiss_merge()
            elif path == "/curation/dismiss-drift":
                self._handle_curation_dismiss_drift()
            elif path == "/search/semantic":
                self._handle_semantic_search()
            else:
                self._send_error(404, f"Not found: {path}")
        except Exception:
            logger.exception("Internal error handling POST %s", path)
            self._send_error(500, "Internal server error")

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        self._send_json({"status": "ok", "version": _VERSION})

    def _handle_procedures_list(self) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        procedures = kb.list_procedures()
        summaries = []
        for proc in procedures:
            summaries.append({
                "id": proc.get("id", proc.get("slug", "")),
                "title": proc.get("title", ""),
                "confidence": proc.get("confidence", proc.get("confidence_avg")),
                "last_observed": proc.get("staleness", {}).get("last_observed", proc.get("generated_at")),
                "trust_level": proc.get("constraints", {}).get("trust_level", "observe"),
                "freshness_score": _compute_freshness(proc),
                "lifecycle_state": proc.get("lifecycle_state", "observed"),
            })
        self._send_json({"procedures": summaries, "count": len(summaries)})

    def _handle_procedure_detail(self, slug: str) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        procedure = kb.get_procedure(slug)
        if procedure is None:
            self._send_error(404, f"Procedure not found: {slug}")
            return
        self._send_json(procedure)

    def _handle_profile(self) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        self._send_json(kb.get_profile())

    def _handle_decisions(self) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        self._send_json(kb.get_decisions())

    def _handle_triggers(self) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        self._send_json(kb.get_triggers())

    def _handle_constraints(self) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        self._send_json(kb.get_constraints())

    def _handle_context(self, name: str) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        self._send_json(kb.get_context(name))

    def _handle_daily(self, date: str) -> None:
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        summary = kb.get_daily_summary(date)
        if summary is None:
            self._send_error(404, f"No daily summary for: {date}")
            return
        self._send_json(summary)

    def _handle_search(self) -> None:
        searcher = getattr(self.server, "activity_searcher", None)
        if searcher is None:
            self._send_error(
                501,
                "Search is not available — no ActivitySearcher configured",
            )
            return

        # Read and parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Request body is required")
            return

        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        query = body.get("query", "")
        if not query:
            self._send_error(400, "Missing 'query' field")
            return

        limit = body.get("limit", 20)
        date = body.get("date")
        app = body.get("app")

        results = searcher.search(query, limit=limit, date=date, app=app)

        # Convert SearchResult dataclass instances to dicts
        serialized = []
        for r in results:
            if hasattr(r, "__dataclass_fields__"):
                serialized.append(asdict(r))
            elif isinstance(r, dict):
                serialized.append(r)
            else:
                serialized.append(str(r))

        self._send_json({
            "query": query,
            "results": serialized,
            "count": len(serialized),
        })

    def _handle_semantic_search(self) -> None:
        """POST /search/semantic — vector similarity search across the KB."""
        vector_kb = getattr(self.server, "vector_kb", None)
        if vector_kb is None:
            self._send_error(
                501,
                "Semantic search not available — no vector KB configured",
            )
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Request body is required")
            return

        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        query = body.get("query", "")
        if not query:
            self._send_error(400, "Missing 'query' field")
            return

        top_k = body.get("limit", 10)
        source_types = body.get("source_types")
        min_score = body.get("min_score", 0.3)

        results = vector_kb.search(
            query,
            top_k=top_k,
            source_types=source_types,
            min_score=min_score,
        )

        self._send_json({
            "query": query,
            "results": [
                {
                    "source_type": r.source_type,
                    "source_id": r.source_id,
                    "score": r.score,
                    "model": r.model,
                }
                for r in results
            ],
            "count": len(results),
        })

    def _handle_bundle(self, slug: str) -> None:
        """Return a complete agent handoff bundle for a procedure."""
        # Try BundleCompiler first, fall back to manual assembly
        bundle_compiler = getattr(self.server, "bundle_compiler", None)
        if bundle_compiler is not None:
            try:
                from dataclasses import asdict as _asdict
                bundle = bundle_compiler.compile(slug)
                if bundle is None:
                    self._send_error(404, f"Procedure not found: {slug}")
                    return

                # Load the full procedure for the response body
                kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
                procedure = kb.get_procedure(slug) or {}

                # Run preflight if verifier is available
                preflight = None
                verifier = getattr(self.server, "procedure_verifier", None)
                if verifier is not None:
                    try:
                        pf_result = verifier.preflight(slug)
                        preflight = {
                            "can_execute": pf_result.can_execute,
                            "can_draft": pf_result.can_draft,
                            "errors": [{"name": c.name, "detail": c.detail} for c in pf_result.errors],
                            "warnings": [{"name": c.name, "detail": c.detail} for c in pf_result.warnings],
                        }
                    except Exception:
                        pass

                # Execution stats
                execution_stats = None
                exec_monitor = getattr(self.server, "execution_monitor", None)
                if exec_monitor is not None:
                    try:
                        execution_stats = exec_monitor.get_success_rate(slug)
                    except Exception:
                        pass

                self._send_json({
                    "slug": bundle.slug,
                    "executable": bundle.readiness.can_execute,
                    "procedure": procedure,
                    "lifecycle_state": bundle.readiness.lifecycle_state,
                    "trust_level": bundle.readiness.trust_level,
                    "freshness_score": bundle.readiness.freshness,
                    "readiness": _asdict(bundle.readiness),
                    "compiled_outputs": [_asdict(co) for co in bundle.compiled_outputs],
                    "preflight": preflight,
                    "execution_stats": execution_stats,
                    "chain": procedure.get("chain", {}),
                    "recurrence": procedure.get("recurrence", {}),
                })
                return
            except Exception:
                logger.debug("BundleCompiler failed, falling back", exc_info=True)

        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        procedure = kb.get_procedure(slug)
        if procedure is None:
            self._send_error(404, f"Procedure not found: {slug}")
            return

        # Compute freshness score
        from agenthandover_worker.staleness_detector import procedure_freshness
        freshness = procedure_freshness(procedure)

        # Get trust level and constraints
        constraints = procedure.get("constraints", {})
        trust_level = constraints.get("trust_level", "observe")

        # Get global constraints
        global_constraints = kb.get_constraints()

        # Build staleness info
        staleness = procedure.get("staleness", {})

        # Run preflight check if verifier is available
        preflight = None
        verifier = getattr(self.server, "procedure_verifier", None)
        if verifier is not None:
            try:
                result = verifier.preflight(slug)
                preflight = {
                    "can_execute": result.can_execute,
                    "can_draft": result.can_draft,
                    "errors": [{"name": c.name, "detail": c.detail} for c in result.errors],
                    "warnings": [{"name": c.name, "detail": c.detail} for c in result.warnings],
                }
            except Exception:
                pass

        # Build export paths
        export_paths = {}
        # Check known export locations
        from pathlib import Path
        kb_path = kb.root / "procedures" / f"{slug}.json"
        if kb_path.is_file():
            export_paths["knowledge_base"] = str(kb_path)

        openclaw_path = Path.home() / ".openclaw" / "workspace" / "memory" / "apprentice" / "sops" / f"sop.{slug}.md"
        if openclaw_path.is_file():
            export_paths["openclaw"] = str(openclaw_path)

        skill_md_path = Path.home() / ".openclaw" / "workspace" / "memory" / "apprentice" / "skills" / f"SKILL.{slug}.md"
        if skill_md_path.is_file():
            export_paths["skill_md"] = str(skill_md_path)

        claude_skill_path = Path.home() / ".claude" / "skills" / slug / "SKILL.md"
        if claude_skill_path.is_file():
            export_paths["claude_skill"] = str(claude_skill_path)

        # Execution history
        exec_monitor = getattr(self.server, "execution_monitor", None)
        execution_stats = None
        if exec_monitor is not None:
            try:
                execution_stats = exec_monitor.get_success_rate(slug)
            except Exception:
                pass

        # Determine executability for the fallback path
        _executable = False
        if preflight is not None and preflight.get("can_execute"):
            _lifecycle = procedure.get("lifecycle_state", "observed")
            if _lifecycle == "agent_ready" and trust_level in ("execute_with_approval", "autonomous"):
                _executable = True

        bundle = {
            "slug": slug,
            "executable": _executable,
            "procedure": procedure,
            "trust_level": trust_level,
            "freshness_score": freshness,
            "staleness": {
                "status": staleness.get("status", "unknown"),
                "last_observed": staleness.get("last_observed"),
                "last_confirmed": staleness.get("last_confirmed"),
            },
            "constraints": {
                "procedure": constraints,
                "global": global_constraints,
            },
            "preflight": preflight,
            "export_paths": export_paths,
            "execution_stats": execution_stats,
            "chain": procedure.get("chain", {}),
            "recurrence": procedure.get("recurrence", {}),
        }

        self._send_json(bundle)

    def _handle_validate(self, slug: str) -> None:
        """Run runtime validation for a procedure."""
        runtime_validator = getattr(self.server, "runtime_validator", None)
        if runtime_validator is None:
            self._send_error(501, "Runtime validation not configured")
            return
        from dataclasses import asdict as _asdict
        checks = runtime_validator.validate_environment(slug)
        all_passed = all(c.passed for c in checks)
        self._send_json({
            "slug": slug,
            "all_passed": all_passed,
            "checks": [_asdict(c) for c in checks],
            "count": len(checks),
        })

    # ------------------------------------------------------------------
    # Curation GET handlers
    # ------------------------------------------------------------------

    def _handle_curation_queue(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        items = curator.build_curation_queue()
        self._send_json({"items": [asdict(i) for i in items], "count": len(items)})

    def _handle_curation_merges(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        candidates = curator.detect_merge_candidates()
        self._send_json({"merge_candidates": [asdict(c) for c in candidates], "count": len(candidates)})

    def _handle_curation_upgrades(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        candidates = curator.detect_upgrade_candidates()
        self._send_json({"upgrade_candidates": [asdict(c) for c in candidates], "count": len(candidates)})

    def _handle_curation_drift(self, slug: str) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        reports = curator.detect_drift(slug)
        self._send_json({"slug": slug, "drift_reports": [asdict(r) for r in reports], "count": len(reports)})

    def _handle_curation_families(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        families = curator.build_families()
        self._send_json({"families": [asdict(f) for f in families], "count": len(families)})

    def _handle_curation_summary(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        summary = curator.curate()
        self._send_json(asdict(summary))

    # ------------------------------------------------------------------
    # Curation POST handlers
    # ------------------------------------------------------------------

    def _handle_curation_merge_action(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        body = self._read_json_body()
        if body is None:
            return
        slug_a = body.get("slug_a", "")
        slug_b = body.get("slug_b", "")
        if not slug_a or not slug_b:
            self._send_error(400, "Missing slug_a or slug_b")
            return
        result = curator.execute_merge(slug_a, slug_b, actor="human")
        self._send_json(result)

    def _handle_curation_promote(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        body = self._read_json_body()
        if body is None:
            return
        slug = body.get("slug", "")
        to_state = body.get("to_state", "")
        reason = body.get("reason", "")
        if not slug or not to_state:
            self._send_error(400, "Missing slug or to_state")
            return
        result = curator.execute_promote(slug, to_state, actor="human", reason=reason)
        self._send_json(result)

    def _handle_curation_demote(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        body = self._read_json_body()
        if body is None:
            return
        slug = body.get("slug", "")
        to_state = body.get("to_state", "")
        reason = body.get("reason", "")
        if not slug or not to_state:
            self._send_error(400, "Missing slug or to_state")
            return
        result = curator.execute_demote(slug, to_state, actor="human", reason=reason)
        self._send_json(result)

    def _handle_curation_archive(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        body = self._read_json_body()
        if body is None:
            return
        slug = body.get("slug", "")
        reason = body.get("reason", "")
        if not slug:
            self._send_error(400, "Missing slug")
            return
        result = curator.execute_archive(slug, actor="human", reason=reason)
        self._send_json(result)

    def _handle_curation_dismiss_merge(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        body = self._read_json_body()
        if body is None:
            return
        slug_a = body.get("slug_a", "")
        slug_b = body.get("slug_b", "")
        if not slug_a or not slug_b:
            self._send_error(400, "Missing slug_a or slug_b")
            return
        curator.dismiss_merge(slug_a, slug_b)
        self._send_json({"success": True, "dismissed": [slug_a, slug_b]})

    def _handle_curation_dismiss_drift(self) -> None:
        curator = getattr(self.server, "procedure_curator", None)
        if curator is None:
            self._send_error(501, "Curation not configured")
            return
        body = self._read_json_body()
        if body is None:
            return
        slug = body.get("slug", "")
        drift_type = body.get("drift_type", "")
        if not slug or not drift_type:
            self._send_error(400, "Missing slug or drift_type")
            return
        curator.dismiss_drift(slug, drift_type)
        self._send_json({"success": True, "slug": slug, "drift_type": drift_type})

    def _handle_ready(self) -> None:
        """Return ONLY procedures that are ready for agent execution.

        Strict endpoint: every procedure in the response has passed all
        readiness gates (lifecycle=agent_ready, trust=execute+, freshness,
        preflight).  If it's in this response, an agent can execute it.

        For discovery of all procedures including drafts, use ``/available``.
        """
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        from agenthandover_worker.staleness_detector import procedure_freshness
        from agenthandover_worker.bundle_compiler import BundleCompiler
        from agenthandover_worker.lifecycle_manager import ProcedureLifecycle

        verifier = getattr(self.server, "procedure_verifier", None)
        procedures = kb.list_procedures()
        ready = []

        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue

            # Use the canonical readiness computation
            lifecycle_str = proc.get("lifecycle_state", "observed")
            try:
                lifecycle = ProcedureLifecycle(lifecycle_str)
            except ValueError:
                continue

            constraints = proc.get("constraints", {})
            trust_level = constraints.get("trust_level", "observe")
            freshness = procedure_freshness(proc)

            # Run preflight if verifier available
            preflight = None
            if verifier is not None:
                try:
                    preflight = verifier.preflight(slug)
                except Exception:
                    pass

            readiness = BundleCompiler.compute_readiness(
                lifecycle_state=lifecycle,
                trust_level=trust_level,
                freshness=freshness,
                preflight=preflight,
            )

            # STRICT: only include if can_execute is True
            if not readiness.can_execute:
                continue

            ready.append({
                "id": slug,
                "title": proc.get("title", ""),
                "trust_level": trust_level,
                "executable": True,
                "freshness_score": freshness,
                "confidence": proc.get("confidence_avg", 0.0),
                "last_observed": proc.get("staleness", {}).get("last_observed"),
                "apps": proc.get("apps_involved", []),
                "chain": proc.get("chain", {}),
                "lifecycle_state": lifecycle_str,
            })

        self._send_json({
            "ready_procedures": ready,
            "count": len(ready),
        })

    def _handle_available(self) -> None:
        """Return ALL procedures with readiness info for discovery.

        Unlike ``/ready`` which only returns executable procedures,
        this endpoint returns every procedure with its full readiness
        assessment, including ``blocked_by`` reasons.  Use this for
        browsing, dashboards, and agent discovery of draft work.
        """
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        from agenthandover_worker.staleness_detector import procedure_freshness
        from agenthandover_worker.bundle_compiler import BundleCompiler
        from agenthandover_worker.lifecycle_manager import ProcedureLifecycle

        procedures = kb.list_procedures()
        available = []

        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue

            lifecycle_str = proc.get("lifecycle_state", "observed")
            try:
                lifecycle = ProcedureLifecycle(lifecycle_str)
            except ValueError:
                lifecycle = ProcedureLifecycle.OBSERVED

            constraints = proc.get("constraints", {})
            trust_level = constraints.get("trust_level", "observe")
            freshness = procedure_freshness(proc)

            readiness = BundleCompiler.compute_readiness(
                lifecycle_state=lifecycle,
                trust_level=trust_level,
                freshness=freshness,
            )

            available.append({
                "id": slug,
                "title": proc.get("title", ""),
                "trust_level": trust_level,
                "can_execute": readiness.can_execute,
                "can_draft": readiness.can_draft,
                "freshness_score": freshness,
                "confidence": proc.get("confidence_avg", 0.0),
                "last_observed": proc.get("staleness", {}).get("last_observed"),
                "apps": proc.get("apps_involved", []),
                "lifecycle_state": lifecycle_str,
                "blocked_by": readiness.reasons if readiness.reasons else [],
            })

        self._send_json({
            "available_procedures": available,
            "count": len(available),
        })

    # ------------------------------------------------------------------
    # Telemetry / version handlers
    # ------------------------------------------------------------------

    def _handle_health_detailed(self) -> None:
        telemetry = getattr(self.server, "ops_telemetry", None)
        if telemetry is None:
            self._send_error(501, "Telemetry not configured")
            return
        self._send_json(telemetry.get_health_snapshot())

    def _handle_telemetry_trend(self) -> None:
        telemetry = getattr(self.server, "ops_telemetry", None)
        if telemetry is None:
            self._send_error(501, "Telemetry not configured")
            return
        self._send_json({"trend": telemetry.get_trend(7)})

    def _handle_version(self) -> None:
        from agenthandover_worker.procedure_schema import PROCEDURE_SCHEMA_VERSION
        self._send_json({
            "worker_version": "0.2.0",
            "schema_version": PROCEDURE_SCHEMA_VERSION,
        })

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _read_json_body(self) -> dict | None:
        """Read and parse JSON request body. Returns None on error (sends 400)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return None
            body = self.rfile.read(content_length)
            return json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return None

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_json(self, data: dict | list, status: int = 200) -> None:
        """Send a JSON response with proper headers."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        """Send a JSON error response."""
        self._send_json({"error": message}, status=status)


class QueryAPIServer:
    """Wraps HTTPServer, runs in a daemon thread.

    Parameters
    ----------
    knowledge_base:
        The KnowledgeBase instance to serve read-only data from.
    port:
        TCP port to bind to (default 9477).
    activity_searcher:
        Optional ActivitySearcher for ``POST /search``.
    execution_monitor:
        Optional execution monitor for success-rate stats in bundles.
    procedure_verifier:
        Optional procedure verifier for preflight checks in bundles.
    bundle_compiler:
        Optional BundleCompiler for the ``/bundle`` endpoint.
    procedure_curator:
        Optional ProcedureCurator for ``/curation/*`` endpoints.
    runtime_validator:
        Optional RuntimeValidator for ``/validate/{slug}`` endpoint.
    ops_telemetry:
        Optional OpsTelemetry for ``/health/detailed`` and ``/telemetry/trend`` endpoints.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        port: int = 9477,
        activity_searcher: Any = None,
        execution_monitor: Any = None,
        procedure_verifier: Any = None,
        bundle_compiler: Any = None,
        procedure_curator: Any = None,
        runtime_validator: Any = None,
        ops_telemetry: Any = None,
        vector_kb: Any = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._port = port
        self._activity_searcher = activity_searcher
        self._execution_monitor = execution_monitor
        self._procedure_verifier = procedure_verifier
        self._bundle_compiler = bundle_compiler
        self._procedure_curator = procedure_curator
        self._runtime_validator = runtime_validator
        self._ops_telemetry = ops_telemetry
        self._vector_kb = vector_kb
        self._httpd: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Launch the HTTP server in a daemon thread."""
        if self._running:
            logger.warning("QueryAPIServer is already running")
            return

        self._httpd = http.server.HTTPServer(
            ("127.0.0.1", self._port),
            QueryAPIHandler,
        )
        # Attach knowledge base and searcher to the server instance
        # so the handler can access them via self.server
        self._httpd.knowledge_base = self._knowledge_base  # type: ignore[attr-defined]
        self._httpd.activity_searcher = self._activity_searcher  # type: ignore[attr-defined]
        self._httpd.execution_monitor = self._execution_monitor  # type: ignore[attr-defined]
        self._httpd.procedure_verifier = self._procedure_verifier  # type: ignore[attr-defined]
        self._httpd.bundle_compiler = self._bundle_compiler  # type: ignore[attr-defined]
        self._httpd.procedure_curator = self._procedure_curator  # type: ignore[attr-defined]
        self._httpd.runtime_validator = self._runtime_validator  # type: ignore[attr-defined]
        self._httpd.ops_telemetry = self._ops_telemetry  # type: ignore[attr-defined]
        self._httpd.vector_kb = self._vector_kb  # type: ignore[attr-defined]

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="query-api",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info("QueryAPIServer started on 127.0.0.1:%d", self._port)

    def stop(self) -> None:
        """Shutdown the server and wait for the thread to finish."""
        if not self._running or self._httpd is None:
            return
        self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._httpd.server_close()
        self._running = False
        self._httpd = None
        self._thread = None
        logger.info("QueryAPIServer stopped")

    @property
    def is_running(self) -> bool:
        """Return True if the server is currently running."""
        return self._running

    @property
    def port(self) -> int:
        """Return the port the server is bound to."""
        return self._port

"""Agent Query API — local HTTP server for read-only knowledge base access.

AI agents query this server to read procedures, profile, decisions,
triggers, constraints, context, and daily summaries from the knowledge
base.  An optional ``ActivitySearcher`` can be provided for full-text
search over VLM annotations.

The server binds to ``127.0.0.1`` only (no external access) and runs
in a daemon thread so it can be started alongside the worker process.

Binding to loopback does not keep a web page out: a browser will send a
cross-site POST to ``127.0.0.1`` (CSRF), and a DNS-rebound hostname lets
a page read GET responses.  So every request must carry the loopback
``Host`` this server is bound to and no ``Origin`` header (browsers set
one on cross-site and on every POST request), and every POST must carry
``X-AgentHandover-Token``: a random token kept in ``query-api-token``
(mode 0600) in the data directory, which only the user's own processes
can read.

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

import hmac
import http.server
import json
import logging
import os
import re
import secrets
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agenthandover_worker import __version__
from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


def procedure_export_paths(kb: KnowledgeBase, slug: str) -> dict[str, str]:
    """Where *slug* has been exported, by target; only files that exist.

    The Claude Code path is the one ``ClaudeSkillWriter`` and
    ``agent_connect.connect_claude_code`` write:
    ``<[export] claude_skills_dir>/<slugify(slug)>/SKILL.md``.
    """
    from pathlib import Path

    from agenthandover_worker.config import load_config
    from agenthandover_worker.skill_render import slugify

    apprentice = Path.home() / ".openclaw" / "workspace" / "memory" / "apprentice"
    candidates = {
        "knowledge_base": kb.root / "procedures" / f"{slug}.json",
        "openclaw": apprentice / "sops" / f"sop.{slug}.md",
        "skill_md": apprentice / "skills" / f"SKILL.{slug}.md",
        "claude_skill": load_config().export.skills_dir / slugify(slug) / "SKILL.md",
    }
    return {name: str(path) for name, path in candidates.items() if path.is_file()}


# URL patterns
_PROCEDURE_SLUG_RE = re.compile(r"^/procedures/([a-zA-Z0-9_-]+)$")
_CONTEXT_NAME_RE = re.compile(r"^/context/([a-zA-Z0-9_-]+)$")
_DAILY_DATE_RE = re.compile(r"^/daily/(\d{4}-\d{2}-\d{2})$")
_BUNDLE_SLUG_RE = re.compile(r"^/bundle/([a-zA-Z0-9_-]+)$")

TOKEN_HEADER = "X-AgentHandover-Token"
TOKEN_FILE = "query-api-token"


def load_or_create_token(path: Path) -> str:
    """Return the API token stored at ``path``, creating it (mode 0600) if absent.

    The token is created atomically and exclusively (written to a private
    temp file, then hard-linked into place), so two processes starting
    together agree on one token.  An existing file readable by anyone but
    its owner is tightened to 0600; an empty or unreadable one is replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            fd = -1
        if fd >= 0:
            with os.fdopen(fd, "r", encoding="utf-8") as fh:
                if os.fstat(fh.fileno()).st_mode & 0o077:
                    os.fchmod(fh.fileno(), 0o600)
                token = fh.read().strip()
            if token:
                return token
            path.unlink(missing_ok=True)
        token = secrets.token_urlsafe(32)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
        tfd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(tfd, "w", encoding="utf-8") as fh:
                fh.write(token + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.link(tmp, path)
            return token
        except FileExistsError:
            continue  # another process won the race: read its token
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not create or read the API token at {path}")


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

    def _reject_untrusted(self, *, require_token: bool) -> bool:
        """Send 403 and return True unless the request is from a local client.

        A request is refused when its ``Host`` is not the loopback address
        this server is bound to (DNS rebinding), when it carries an
        ``Origin`` (a browser page), or, for POST, when it lacks the token.
        """
        port = self.server.server_address[1]
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._send_error(403, "Forbidden: Host must be 127.0.0.1:%d" % port)
            return True
        if self.headers.get("Origin") is not None:
            self._send_error(403, "Forbidden: browser requests are not accepted")
            return True
        if require_token:
            sent = self.headers.get(TOKEN_HEADER) or ""
            expected: str = self.server.api_token  # type: ignore[attr-defined]
            if not hmac.compare_digest(sent.encode(), expected.encode()):
                self._send_error(
                    403,
                    f"Forbidden: missing or wrong {TOKEN_HEADER} header "
                    f"(the token is in {TOKEN_FILE} in the data directory)",
                )
                return True
        return False

    def do_GET(self) -> None:  # noqa: N802
        """Route GET requests to the appropriate handler."""
        path = self.path.split("?")[0]  # strip query string
        if self._reject_untrusted(require_token=False):
            return

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
            elif path == "/memory/status":
                self._handle_memory("status")
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
        if self._reject_untrusted(require_token=True):
            return

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
            elif path == "/recall":
                self._handle_recall()
            elif path in ("/memory/search", "/memory/pause", "/memory/resume", "/memory/forget"):
                self._handle_memory(path.rsplit("/", 1)[1])
            elif path == "/commonplace/search":
                self._handle_commonplace_search()
            elif _BUNDLE_SLUG_RE.match(path):
                slug = _BUNDLE_SLUG_RE.match(path).group(1)  # type: ignore[union-attr]
                self._handle_bundle_compile(slug)
            else:
                self._send_error(404, f"Not found: {path}")
        except Exception:
            logger.exception("Internal error handling POST %s", path)
            self._send_error(500, "Internal server error")

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        self._send_json({"status": "ok", "version": __version__})

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

    def _handle_recall(self) -> None:
        """POST /recall — session recall (what the CLI's ``recall`` shows)."""
        searcher = getattr(self.server, "activity_searcher", None)
        if searcher is None:
            self._send_error(
                501,
                "Recall is not available — no ActivitySearcher configured",
            )
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return
        if not isinstance(body, dict):
            self._send_error(400, "Request body must be a JSON object")
            return

        from agenthandover_worker.activity_search import recall_payload
        self._send_json(recall_payload(searcher, body))

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

        if not isinstance(body, dict):
            self._send_error(400, "Request body must be a JSON object")
            return
        query = body.get("query", "")
        if not isinstance(query, str) or not query:
            self._send_error(400, "Missing 'query' field")
            return

        top_k = body.get("limit", 10)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            self._send_error(400, "'limit' must be a positive integer")
            return
        source_types = body.get("source_types")
        if source_types is not None and not (
            isinstance(source_types, list)
            and all(isinstance(t, str) for t in source_types)
        ):
            self._send_error(400, "'source_types' must be a list of strings")
            return
        # Default floor: the one measured on the configured embedding model.
        min_score = body.get("min_score")
        if min_score is None:
            min_score = vector_kb.thresholds.min_score
        elif isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            self._send_error(400, "'min_score' must be a number")
            return

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

    def _handle_memory(self, action: str) -> None:
        """``/memory/*`` (``memory_api.MemoryService``): the body (empty is
        ``{}``) goes to the service, whose answer is sent as is."""
        service = getattr(self.server, "memory", None)
        if service is None:
            self._send_error(501, "Memory is not available in this worker")
            return
        if action == "status":
            status, reply = service.status()
            self._send_json(reply, status=status)
            return
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(content_length) if content_length else b""
        try:
            body = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return
        if action == "resume":
            status, reply = service.resume()
        else:
            status, reply = getattr(service, action)(body)
        self._send_json(reply, status=status)

    def _handle_commonplace_search(self) -> None:
        """``POST /commonplace/search {"query", "kinds"?, "since"?,
        "until"?, "limit"?}``: ``commonplace.Commonplace.search``'s answer,
        one group per kind (Memory only while agents may search it);
        400 with ``error`` for a bad request."""
        commonplace = getattr(self.server, "commonplace", None)
        if commonplace is None:
            self._send_error(501, "Commonplace search is not available in this worker")
            return
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(content_length) if content_length else b""
        try:
            body = json.loads(raw) if raw.strip() else None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return
        if not isinstance(body, dict):
            self._send_error(400, "Request body must be a JSON object")
            return
        unknown = set(body) - {"query", "kinds", "since", "until", "limit"}
        if unknown:
            self._send_error(400, f"unknown field(s): {', '.join(sorted(unknown))}")
            return
        try:
            answer = commonplace.search(
                body.get("query"), body.get("kinds"), body.get("since"),
                body.get("until"), body.get("limit", 5),
            )
        except ValueError as exc:
            self._send_error(400, str(exc))
            return
        self._send_json(answer)

    def _bundle_extras(self, slug: str) -> tuple[dict | None, dict | None]:
        """Preflight summary and execution stats for the bundle responses."""
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
                logger.debug("Preflight failed for %s", slug, exc_info=True)
        execution_stats = None
        exec_monitor = getattr(self.server, "execution_monitor", None)
        if exec_monitor is not None:
            try:
                execution_stats = exec_monitor.get_success_rate(slug)
            except Exception:
                logger.debug("Execution stats failed for %s", slug, exc_info=True)
        return preflight, execution_stats

    def _send_bundle(self, slug: str, procedure: dict, readiness, compiled_outputs: list[dict]) -> None:
        from dataclasses import asdict as _asdict

        preflight, execution_stats = self._bundle_extras(slug)
        self._send_json({
            "slug": slug,
            "executable": readiness.can_execute,
            "procedure": procedure,
            "lifecycle_state": readiness.lifecycle_state,
            "trust_level": readiness.trust_level,
            "freshness_score": readiness.freshness,
            "readiness": _asdict(readiness),
            "compiled_outputs": compiled_outputs,
            "preflight": preflight,
            "execution_stats": execution_stats,
            "chain": procedure.get("chain", {}),
            "recurrence": procedure.get("recurrence", {}),
        })

    def _handle_bundle_compile(self, slug: str) -> None:
        """POST /bundle/<slug>: compile the procedure now (writes every
        adapter target when it passes the hand-off gate, stores
        ``compiled_outputs``) and return the bundle."""
        bundle_compiler = getattr(self.server, "bundle_compiler", None)
        if bundle_compiler is None:
            self._send_error(503, "Bundle compiler not available")
            return
        from dataclasses import asdict as _asdict

        bundle = bundle_compiler.compile(slug)
        if bundle is None:
            self._send_error(404, f"Procedure not found: {slug}")
            return
        kb: KnowledgeBase = self.server.knowledge_base  # type: ignore[attr-defined]
        self._send_bundle(
            slug, kb.get_procedure(slug) or {}, bundle.readiness,
            [_asdict(co) for co in bundle.compiled_outputs],
        )

    def _handle_bundle(self, slug: str) -> None:
        """GET /bundle/<slug>: the handoff bundle, without side effects.

        Readiness is computed now (lifecycle, trust, freshness, preflight
        and the ledger's required slots); ``compiled_outputs`` are the ones
        stored by the last compile.  Nothing is written: compiling is
        ``POST /bundle/<slug>``.
        """
        bundle_compiler = getattr(self.server, "bundle_compiler", None)
        if bundle_compiler is not None:
            assessed = bundle_compiler.assess(slug)
            if assessed is None:
                self._send_error(404, f"Procedure not found: {slug}")
                return
            procedure, readiness = assessed
            stored = procedure.get("compiled_outputs") or {}
            self._send_bundle(slug, procedure, readiness, [
                {"adapter_name": name, **info}
                for name, info in stored.items() if isinstance(info, dict)
            ])
            return

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

        preflight, execution_stats = self._bundle_extras(slug)
        export_paths = procedure_export_paths(kb, slug)

        # Determine executability for the fallback path
        from agenthandover_worker.ledger import readiness as ledger_readiness, unknown_slots
        _executable = False
        if preflight is not None and preflight.get("can_execute"):
            _lifecycle = procedure.get("lifecycle_state", "observed")
            if (
                _lifecycle == "agent_ready"
                and trust_level in ("execute_with_approval", "autonomous")
                and ledger_readiness(procedure).ready
            ):
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
            "not_yet_known": unknown_slots(procedure),
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
        from agenthandover_worker.ledger import readiness as ledger_readiness

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
                unknown_required=ledger_readiness(proc).unknown_required,
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
        from agenthandover_worker.ledger import readiness as ledger_readiness

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
                unknown_required=ledger_readiness(proc).unknown_required,
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
        # Worker version read from package metadata (dist-info), not a
        # hardcoded string. Previously frozen at "0.2.0" through three
        # releases — the REST /version endpoint was lying about which
        # worker the agent was talking to.
        from agenthandover_worker.procedure_schema import PROCEDURE_SCHEMA_VERSION
        self._send_json({
            "worker_version": __version__,
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
    memory:
        Optional ``memory_api.MemoryService`` for the ``/memory/*`` endpoints.
    commonplace:
        Optional ``commonplace.Commonplace`` for ``POST /commonplace/search``.
    token_path:
        File holding the POST token (created 0600 if absent).  ``None``
        keeps a random token in memory only (tests read ``token``).
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
        token_path: Path | None = None,
        memory: Any = None,
        commonplace: Any = None,
    ) -> None:
        self._token_path = token_path
        self._token: str | None = None
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
        self._memory = memory
        self._commonplace = commonplace
        self._httpd: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Launch the HTTP server in a daemon thread."""
        if self._running:
            logger.warning("QueryAPIServer is already running")
            return

        self._token = (
            load_or_create_token(self._token_path)
            if self._token_path is not None
            else secrets.token_urlsafe(32)
        )
        self._httpd = http.server.HTTPServer(
            ("127.0.0.1", self._port),
            QueryAPIHandler,
        )
        self._port = self._httpd.server_address[1]  # resolves port=0
        self._httpd.api_token = self._token  # type: ignore[attr-defined]
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
        self._httpd.memory = self._memory  # type: ignore[attr-defined]
        self._httpd.commonplace = self._commonplace  # type: ignore[attr-defined]

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
    def token(self) -> str | None:
        """The token POST requests must send (set once the server started)."""
        return self._token

    @property
    def port(self) -> int:
        """Return the port the server is bound to."""
        return self._port

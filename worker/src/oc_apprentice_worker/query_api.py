"""Agent Query API — local HTTP server for read-only knowledge base access.

AI agents query this server to read procedures, profile, decisions,
triggers, constraints, context, and daily summaries from the knowledge
base.  An optional ``ActivitySearcher`` can be provided for full-text
search over VLM annotations.

The server binds to ``127.0.0.1`` only (no external access) and runs
in a daemon thread so it can be started alongside the worker process.

Usage::

    from oc_apprentice_worker.knowledge_base import KnowledgeBase
    from oc_apprentice_worker.query_api import QueryAPIServer

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

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"

# URL patterns
_PROCEDURE_SLUG_RE = re.compile(r"^/procedures/([a-zA-Z0-9_-]+)$")
_CONTEXT_NAME_RE = re.compile(r"^/context/([a-zA-Z0-9_-]+)$")
_DAILY_DATE_RE = re.compile(r"^/daily/(\d{4}-\d{2}-\d{2})$")


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
                "last_observed": proc.get("last_observed", proc.get("updated_at")),
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
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        port: int = 9477,
        activity_searcher: Any = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._port = port
        self._activity_searcher = activity_searcher
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

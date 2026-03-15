"""Tests for the Agent Query API module."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.query_api import QueryAPIServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> tuple[int, dict]:
    """Send a GET request and return (status_code, json_body)."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return exc.code, body


def _post(port: int, path: str, data: dict) -> tuple[int, dict]:
    """Send a POST request with JSON body and return (status_code, json_body)."""
    url = f"http://127.0.0.1:{port}{path}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return exc.code, body


def _get_raw(port: int, path: str) -> urllib.request.Request:
    """Send a GET and return the full response for header inspection."""
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url) as resp:
        return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def sample_procedure() -> dict:
    """A minimal procedure for testing."""
    return {
        "id": "check-expired-domains",
        "slug": "check-expired-domains",
        "title": "Check Expired Domains",
        "confidence": 0.87,
        "confidence_avg": 0.87,
        "generated_at": "2026-03-10T12:00:00Z",
        "staleness": {
            "last_observed": "2026-03-10T12:00:00Z",
            "last_confirmed": None,
            "drift_signals": [],
            "confidence_trend": [0.87],
        },
        "constraints": {"trust_level": "observe", "guardrails": []},
        "steps": [
            {"step": "Open browser", "action": "launch", "target": "Chrome"},
        ],
    }


@pytest.fixture()
def populated_kb(kb: KnowledgeBase, sample_procedure: dict) -> KnowledgeBase:
    """A KnowledgeBase pre-populated with test data."""
    # Procedure
    kb.save_procedure(sample_procedure)
    kb.save_procedure({
        "id": "deploy-app",
        "slug": "deploy-app",
        "title": "Deploy Application",
        "confidence": 0.72,
        "last_observed": "2026-03-09T08:00:00Z",
        "steps": [],
    })

    # Profile
    kb.update_profile({
        "tools": {"editor": "VSCode", "terminal": "iTerm2"},
        "working_hours": {"start": "09:00", "end": "17:00"},
    })

    # Decisions
    kb.update_decisions({
        "decision_sets": [
            {"name": "deploy-gating", "rules": ["all tests pass"]},
        ],
    })

    # Triggers
    kb.update_triggers({
        "recurrence": [{"pattern": "daily", "procedure": "check-expired-domains"}],
        "chains": [],
    })

    # Constraints
    kb.update_constraints({
        "global": {"max_concurrent": 3},
        "per_procedure": {"deploy-app": {"requires_approval": True}},
    })

    # Context
    kb.update_context("recent", {
        "last_app": "Chrome",
        "last_url": "https://example.com",
    })

    # Daily summary
    kb.save_daily_summary("2026-03-10", {
        "events": 142,
        "procedures_observed": ["check-expired-domains"],
    })

    return kb


@pytest.fixture()
def server(populated_kb: KnowledgeBase) -> QueryAPIServer:
    """Start a QueryAPIServer on a random port, yield it, then stop."""
    port = _free_port()
    srv = QueryAPIServer(populated_kb, port=port)
    srv.start()
    # Brief pause to ensure the server thread is ready
    time.sleep(0.1)
    yield srv
    srv.stop()


@pytest.fixture()
def port(server: QueryAPIServer) -> int:
    """Return the port of the running server."""
    return server.port


# ---------------------------------------------------------------------------
# Server lifecycle tests
# ---------------------------------------------------------------------------

class TestServerLifecycle:
    """Test server start/stop behavior."""

    def test_start_sets_running(self, populated_kb: KnowledgeBase) -> None:
        port = _free_port()
        srv = QueryAPIServer(populated_kb, port=port)
        assert not srv.is_running
        srv.start()
        try:
            assert srv.is_running
        finally:
            srv.stop()

    def test_stop_clears_running(self, populated_kb: KnowledgeBase) -> None:
        port = _free_port()
        srv = QueryAPIServer(populated_kb, port=port)
        srv.start()
        time.sleep(0.05)
        srv.stop()
        assert not srv.is_running

    def test_double_start_is_safe(self, populated_kb: KnowledgeBase) -> None:
        port = _free_port()
        srv = QueryAPIServer(populated_kb, port=port)
        srv.start()
        try:
            srv.start()  # should not raise
            assert srv.is_running
        finally:
            srv.stop()

    def test_double_stop_is_safe(self, populated_kb: KnowledgeBase) -> None:
        port = _free_port()
        srv = QueryAPIServer(populated_kb, port=port)
        srv.start()
        time.sleep(0.05)
        srv.stop()
        srv.stop()  # should not raise
        assert not srv.is_running

    def test_port_property(self, populated_kb: KnowledgeBase) -> None:
        port = _free_port()
        srv = QueryAPIServer(populated_kb, port=port)
        assert srv.port == port


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Test GET /health."""

    def test_health_returns_ok(self, port: int) -> None:
        status, body = _get(port, "/health")
        assert status == 200
        assert body["status"] == "ok"

    def test_health_returns_version(self, port: int) -> None:
        status, body = _get(port, "/health")
        assert body["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# Procedures endpoints
# ---------------------------------------------------------------------------

class TestProceduresEndpoint:
    """Test GET /procedures and GET /procedures/{slug}."""

    def test_list_procedures(self, port: int) -> None:
        status, body = _get(port, "/procedures")
        assert status == 200
        assert body["count"] == 2
        ids = [p["id"] for p in body["procedures"]]
        assert "check-expired-domains" in ids
        assert "deploy-app" in ids

    def test_list_procedures_summary_fields(self, port: int) -> None:
        status, body = _get(port, "/procedures")
        proc = next(
            p for p in body["procedures"]
            if p["id"] == "check-expired-domains"
        )
        assert proc["title"] == "Check Expired Domains"
        assert proc["confidence"] == 0.87
        assert proc["last_observed"] == "2026-03-10T12:00:00Z"
        assert proc["trust_level"] == "observe"
        assert "freshness_score" in proc

    def test_get_procedure_by_slug(self, port: int) -> None:
        status, body = _get(port, "/procedures/check-expired-domains")
        assert status == 200
        assert body["id"] == "check-expired-domains"
        assert body["title"] == "Check Expired Domains"
        assert len(body["steps"]) == 1

    def test_procedure_not_found(self, port: int) -> None:
        status, body = _get(port, "/procedures/nonexistent-sop")
        assert status == 404
        assert "error" in body

    def test_empty_procedures_list(self, kb: KnowledgeBase) -> None:
        """A knowledge base with no procedures returns empty list."""
        port = _free_port()
        srv = QueryAPIServer(kb, port=port)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/procedures")
            assert status == 200
            assert body["count"] == 0
            assert body["procedures"] == []
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Profile endpoint
# ---------------------------------------------------------------------------

class TestProfileEndpoint:
    """Test GET /profile."""

    def test_get_profile(self, port: int) -> None:
        status, body = _get(port, "/profile")
        assert status == 200
        assert body["tools"]["editor"] == "VSCode"
        assert body["working_hours"]["start"] == "09:00"

    def test_empty_profile(self, kb: KnowledgeBase) -> None:
        """An empty knowledge base returns default profile."""
        port = _free_port()
        srv = QueryAPIServer(kb, port=port)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/profile")
            assert status == 200
            assert "tools" in body
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Decisions endpoint
# ---------------------------------------------------------------------------

class TestDecisionsEndpoint:
    """Test GET /decisions."""

    def test_get_decisions(self, port: int) -> None:
        status, body = _get(port, "/decisions")
        assert status == 200
        assert len(body["decision_sets"]) == 1
        assert body["decision_sets"][0]["name"] == "deploy-gating"


# ---------------------------------------------------------------------------
# Triggers endpoint
# ---------------------------------------------------------------------------

class TestTriggersEndpoint:
    """Test GET /triggers."""

    def test_get_triggers(self, port: int) -> None:
        status, body = _get(port, "/triggers")
        assert status == 200
        assert len(body["recurrence"]) == 1
        assert body["recurrence"][0]["pattern"] == "daily"
        assert body["chains"] == []


# ---------------------------------------------------------------------------
# Constraints endpoint
# ---------------------------------------------------------------------------

class TestConstraintsEndpoint:
    """Test GET /constraints."""

    def test_get_constraints(self, port: int) -> None:
        status, body = _get(port, "/constraints")
        assert status == 200
        assert body["global"]["max_concurrent"] == 3
        assert body["per_procedure"]["deploy-app"]["requires_approval"] is True


# ---------------------------------------------------------------------------
# Context endpoint
# ---------------------------------------------------------------------------

class TestContextEndpoint:
    """Test GET /context/{name}."""

    def test_get_existing_context(self, port: int) -> None:
        status, body = _get(port, "/context/recent")
        assert status == 200
        assert body["last_app"] == "Chrome"

    def test_get_missing_context(self, port: int) -> None:
        status, body = _get(port, "/context/nonexistent")
        assert status == 200
        assert body == {}


# ---------------------------------------------------------------------------
# Daily endpoint
# ---------------------------------------------------------------------------

class TestDailyEndpoint:
    """Test GET /daily/{date}."""

    def test_get_existing_daily(self, port: int) -> None:
        status, body = _get(port, "/daily/2026-03-10")
        assert status == 200
        assert body["events"] == 142
        assert "check-expired-domains" in body["procedures_observed"]

    def test_daily_not_found(self, port: int) -> None:
        status, body = _get(port, "/daily/1999-01-01")
        assert status == 404
        assert "error" in body


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    """Test POST /search."""

    def test_search_without_searcher_returns_501(self, port: int) -> None:
        status, body = _post(port, "/search", {"query": "domains"})
        assert status == 501
        assert "error" in body

    def test_search_with_mock_searcher(
        self, populated_kb: KnowledgeBase,
    ) -> None:
        """POST /search with a mock ActivitySearcher returns results."""

        @dataclass
        class FakeResult:
            timestamp: str
            app: str
            location: str
            what_doing: str
            relevance_score: float
            event_id: str
            screenshot_id: str | None = None

        class FakeSearcher:
            def search(
                self,
                query: str,
                *,
                limit: int = 20,
                date: str | None = None,
                app: str | None = None,
            ) -> list:
                return [
                    FakeResult(
                        timestamp="2026-03-10T12:00:00Z",
                        app="Chrome",
                        location="https://example.com",
                        what_doing="Browsing domains",
                        relevance_score=0.95,
                        event_id="evt-001",
                    ),
                ]

        p = _free_port()
        srv = QueryAPIServer(
            populated_kb,
            port=p,
            activity_searcher=FakeSearcher(),
        )
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _post(p, "/search", {
                "query": "domains",
                "limit": 10,
                "date": "2026-03-10",
            })
            assert status == 200
            assert body["count"] == 1
            assert body["query"] == "domains"
            assert body["results"][0]["app"] == "Chrome"
            assert body["results"][0]["relevance_score"] == 0.95
        finally:
            srv.stop()

    def test_search_empty_body_returns_400(
        self, populated_kb: KnowledgeBase,
    ) -> None:
        """POST /search with empty body returns 400."""

        class FakeSearcher:
            def search(self, query: str, **kwargs: Any) -> list:
                return []

        p = _free_port()
        srv = QueryAPIServer(
            populated_kb,
            port=p,
            activity_searcher=FakeSearcher(),
        )
        srv.start()
        time.sleep(0.1)
        try:
            url = f"http://127.0.0.1:{p}/search"
            req = urllib.request.Request(
                url,
                data=b"",
                headers={"Content-Type": "application/json", "Content-Length": "0"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    pytest.fail("Expected HTTPError")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
        finally:
            srv.stop()

    def test_search_missing_query_returns_400(
        self, populated_kb: KnowledgeBase,
    ) -> None:
        """POST /search with missing query field returns 400."""

        class FakeSearcher:
            def search(self, query: str, **kwargs: Any) -> list:
                return []

        p = _free_port()
        srv = QueryAPIServer(
            populated_kb,
            port=p,
            activity_searcher=FakeSearcher(),
        )
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _post(p, "/search", {"limit": 10})
            assert status == 400
            assert "query" in body["error"].lower()
        finally:
            srv.stop()

    def test_search_invalid_json_returns_400(
        self, populated_kb: KnowledgeBase,
    ) -> None:
        """POST /search with invalid JSON returns 400."""

        class FakeSearcher:
            def search(self, query: str, **kwargs: Any) -> list:
                return []

        p = _free_port()
        srv = QueryAPIServer(
            populated_kb,
            port=p,
            activity_searcher=FakeSearcher(),
        )
        srv.start()
        time.sleep(0.1)
        try:
            url = f"http://127.0.0.1:{p}/search"
            req = urllib.request.Request(
                url,
                data=b"not valid json{{{",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    pytest.fail("Expected HTTPError")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Error handling & routing
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error responses and unknown routes."""

    def test_unknown_get_route_returns_404(self, port: int) -> None:
        status, body = _get(port, "/nonexistent")
        assert status == 404
        assert "error" in body

    def test_unknown_post_route_returns_404(self, port: int) -> None:
        status, body = _post(port, "/nonexistent", {})
        assert status == 404
        assert "error" in body

    def test_deep_unknown_path_returns_404(self, port: int) -> None:
        status, body = _get(port, "/a/b/c/d")
        assert status == 404


# ---------------------------------------------------------------------------
# JSON content type
# ---------------------------------------------------------------------------

class TestContentType:
    """Verify JSON Content-Type headers."""

    def test_health_content_type(self, port: int) -> None:
        resp = _get_raw(port, "/health")
        assert resp.headers["Content-Type"] == "application/json"

    def test_procedures_content_type(self, port: int) -> None:
        resp = _get_raw(port, "/procedures")
        assert resp.headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Sequential requests
# ---------------------------------------------------------------------------

class TestSequentialRequests:
    """Verify server handles multiple sequential requests."""

    def test_multiple_requests(self, port: int) -> None:
        """Send several sequential requests and verify all succeed."""
        for _ in range(5):
            status, body = _get(port, "/health")
            assert status == 200
            assert body["status"] == "ok"

    def test_mixed_endpoint_requests(self, port: int) -> None:
        """Hit multiple different endpoints sequentially."""
        endpoints = [
            "/health",
            "/procedures",
            "/profile",
            "/decisions",
            "/triggers",
            "/constraints",
            "/context/recent",
            "/daily/2026-03-10",
        ]
        for endpoint in endpoints:
            status, _ = _get(port, endpoint)
            assert status == 200, f"Failed on {endpoint}"

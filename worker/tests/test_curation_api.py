"""Tests for the curation API endpoints in query_api and main integration."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.lifecycle_manager import LifecycleManager
from agenthandover_worker.procedure_curator import ProcedureCurator
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.query_api import QueryAPIServer
from agenthandover_worker.staleness_detector import StalenessDetector
from agenthandover_worker.trust_advisor import TrustAdvisor


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


def _save_proc(kb: KnowledgeBase, slug: str, apps: list[str], actions: list[str],
               lifecycle: str = "observed", trust: str = "observe", **kw) -> dict:
    """Create and save a minimal procedure to the knowledge base."""
    template = {
        "slug": slug,
        "title": f"Test: {slug}",
        "steps": [
            {"step": a, "app": apps[0], "confidence": 0.9}
            for a in actions
        ],
        "confidence_avg": 0.85,
        "episode_count": 3,
        "apps_involved": apps,
        "source": "test",
    }
    proc = sop_to_procedure(template)
    proc["constraints"]["trust_level"] = trust
    proc["lifecycle_state"] = lifecycle
    for k, v in kw.items():
        proc[k] = v
    kb.save_procedure(proc)
    return proc


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
def curator(kb: KnowledgeBase) -> ProcedureCurator:
    """Create a ProcedureCurator with real subsystems."""
    return ProcedureCurator(
        kb=kb,
        staleness_detector=StalenessDetector(kb),
        trust_advisor=TrustAdvisor(kb),
        lifecycle_manager=LifecycleManager(kb),
    )


# ---------------------------------------------------------------------------
# TestCurationAPIIntegration — GET endpoints
# ---------------------------------------------------------------------------

class TestCurationAPIIntegration:
    """Test GET curation endpoints via live HTTP server."""

    def test_queue_returns_items(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """GET /curation/queue returns structured queue."""
        # Save a procedure eligible for upgrade (observed -> draft)
        _save_proc(kb, "task-a", ["Chrome"], ["open browser", "click link", "submit form"],
                   lifecycle="observed", episode_count=5, confidence_avg=0.80)

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/curation/queue")
            assert status == 200
            assert "items" in body
            assert "count" in body
            assert isinstance(body["items"], list)
            assert body["count"] == len(body["items"])
        finally:
            srv.stop()

    def test_merges_returns_candidates(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """GET /curation/merges returns merge candidates list."""
        # Two similar procedures (same apps, same actions)
        _save_proc(kb, "deploy-v1", ["Terminal", "Chrome"],
                   ["open terminal", "run deploy script", "verify in browser"])
        _save_proc(kb, "deploy-v2", ["Terminal", "Chrome"],
                   ["open terminal", "run deploy script", "check in browser"])

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/curation/merges")
            assert status == 200
            assert "merge_candidates" in body
            assert "count" in body
            assert isinstance(body["merge_candidates"], list)
        finally:
            srv.stop()

    def test_upgrades_returns_candidates(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """GET /curation/upgrades returns upgrade candidates."""
        _save_proc(kb, "eligible-sop", ["Chrome"], ["open page", "fill form", "submit"],
                   lifecycle="observed", episode_count=5, confidence_avg=0.80)

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/curation/upgrades")
            assert status == 200
            assert "upgrade_candidates" in body
            assert "count" in body
            assert isinstance(body["upgrade_candidates"], list)
            # Should detect the eligible procedure
            if body["count"] > 0:
                assert body["upgrade_candidates"][0]["slug"] == "eligible-sop"
                assert body["upgrade_candidates"][0]["proposed_state"] == "draft"
        finally:
            srv.stop()

    def test_drift_returns_reports(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """GET /curation/drift/{slug} returns drift reports for a procedure."""
        proc = _save_proc(kb, "drifty-sop", ["Chrome"],
                          ["open page", "click button"])
        # Add drift signals
        proc["staleness"] = {
            "last_observed": "2026-03-10T12:00:00Z",
            "drift_signals": [
                {"type": "url_changed", "detail": "URL changed from /old to /new", "first_seen": "2026-03-12"},
            ],
            "confidence_trend": [0.9, 0.85],
        }
        kb.save_procedure(proc)

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/curation/drift/drifty-sop")
            assert status == 200
            assert body["slug"] == "drifty-sop"
            assert "drift_reports" in body
            assert "count" in body
            # Should detect the url_changed drift signal
            if body["count"] > 0:
                assert body["drift_reports"][0]["drift_type"] == "url_changed"
        finally:
            srv.stop()

    def test_families_returns_list(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """GET /curation/families returns procedure family list."""
        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/curation/families")
            assert status == 200
            assert "families" in body
            assert "count" in body
            assert isinstance(body["families"], list)
        finally:
            srv.stop()

    def test_summary_returns_counts(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """GET /curation/summary returns CurationSummary fields."""
        _save_proc(kb, "some-sop", ["Chrome"], ["open page"])

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _get(port, "/curation/summary")
            assert status == 200
            # CurationSummary fields
            assert "merge_candidates" in body
            assert "upgrade_candidates" in body
            assert "stale_procedures" in body
            assert "drift_reports" in body
            assert "total_queue_items" in body
            assert "families" in body
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# TestCurationActions — POST endpoints
# ---------------------------------------------------------------------------

class TestCurationActions:
    """Test POST curation action endpoints via live HTTP server."""

    def test_merge_executes(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """POST /curation/merge archives source procedure."""
        _save_proc(kb, "merge-target", ["Chrome"], ["open page", "fill form"])
        _save_proc(kb, "merge-source", ["Chrome"], ["open page", "fill form"])

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _post(port, "/curation/merge", {
                "slug_a": "merge-target",
                "slug_b": "merge-source",
            })
            assert status == 200
            assert body["success"] is True
            assert body["merged_slug"] == "merge-target"
            assert body["archived_slug"] == "merge-source"

            # Verify source is archived
            source_proc = kb.get_procedure("merge-source")
            assert source_proc["lifecycle_state"] == "archived"
        finally:
            srv.stop()

    def test_promote_transitions(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """POST /curation/promote changes lifecycle state."""
        _save_proc(kb, "promo-sop", ["Chrome"], ["open page"],
                   lifecycle="observed")

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _post(port, "/curation/promote", {
                "slug": "promo-sop",
                "to_state": "draft",
                "reason": "Manual promotion for testing",
            })
            assert status == 200
            assert body["success"] is True
            assert body["new_state"] == "draft"

            # Verify procedure state changed
            proc = kb.get_procedure("promo-sop")
            assert proc["lifecycle_state"] == "draft"
        finally:
            srv.stop()

    def test_archive_transitions(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """POST /curation/archive transitions to ARCHIVED."""
        _save_proc(kb, "archive-me", ["Chrome"], ["open page"],
                   lifecycle="observed")

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            status, body = _post(port, "/curation/archive", {
                "slug": "archive-me",
                "reason": "No longer relevant",
            })
            assert status == 200
            assert body["success"] is True
            assert body["new_state"] == "archived"

            proc = kb.get_procedure("archive-me")
            assert proc["lifecycle_state"] == "archived"
        finally:
            srv.stop()

    def test_dismiss_merge_prevents_resurface(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """POST /curation/dismiss-merge prevents the pair from resurfacing."""
        _save_proc(kb, "proc-x", ["Chrome"], ["open page", "fill form", "submit"])
        _save_proc(kb, "proc-y", ["Chrome"], ["open page", "fill form", "submit"])

        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            # Dismiss the merge
            status, body = _post(port, "/curation/dismiss-merge", {
                "slug_a": "proc-x",
                "slug_b": "proc-y",
            })
            assert status == 200
            assert body["success"] is True

            # Check merges — the dismissed pair should not appear
            status, body = _get(port, "/curation/merges")
            assert status == 200
            for candidate in body["merge_candidates"]:
                pair = {candidate["slug_a"], candidate["slug_b"]}
                assert pair != {"proc-x", "proc-y"}, "Dismissed pair should not resurface"
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# TestQueryAPIServerAcceptsCurator — constructor tests
# ---------------------------------------------------------------------------

class TestQueryAPIServerAcceptsCurator:
    """Test that QueryAPIServer properly accepts the procedure_curator param."""

    def test_server_accepts_curator_param(self, kb: KnowledgeBase, curator: ProcedureCurator) -> None:
        """QueryAPIServer(kb, procedure_curator=curator) does not raise."""
        port = _free_port()
        srv = QueryAPIServer(kb, port=port, procedure_curator=curator)
        srv.start()
        time.sleep(0.1)
        try:
            assert srv.is_running
            # Curation endpoint should work
            status, body = _get(port, "/curation/summary")
            assert status == 200
        finally:
            srv.stop()

    def test_server_without_curator(self, kb: KnowledgeBase) -> None:
        """QueryAPIServer(kb) works and returns 501 for curation endpoints."""
        port = _free_port()
        srv = QueryAPIServer(kb, port=port)
        srv.start()
        time.sleep(0.1)
        try:
            assert srv.is_running
            # Curation endpoint should return 501
            status, body = _get(port, "/curation/summary")
            assert status == 501
            assert "Curation not configured" in body["error"]

            # Non-curation endpoint should still work
            status, body = _get(port, "/health")
            assert status == 200
        finally:
            srv.stop()

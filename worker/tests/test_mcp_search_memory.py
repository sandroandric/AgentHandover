"""MCP ``search_memory``: behind the agents switch, descriptions and times
only, keyword fallback when the query cannot be embedded."""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import patch

import pytest

from agenthandover_worker import memory
from agenthandover_worker.ollama import OllamaUnavailable
from memory_world import NOW, World, axis

pytest.importorskip("mcp")

FIELDS = {"app", "window_title", "url_host", "description", "first_ts", "last_ts", "score"}


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.frame("inv", NOW - timedelta(minutes=30), "Google Chrome", "Invoice INV-2041",
                "Paying the Acme invoice", memory=False, url="https://billing.acme.com/inv")
    world.moment("inv", NOW - timedelta(minutes=30), NOW - timedelta(minutes=25),
                 "Google Chrome", "Invoice INV-2041", "Paying the Acme invoice",
                 thumb_id="inv", vector=axis(1))
    world.moment("chat", NOW - timedelta(minutes=10), NOW - timedelta(minutes=10),
                 "Slack", "#design", "Discussing the hero banner", vector=axis(2))
    config = tmp_path / "config.toml"
    monkeypatch.setattr("agenthandover_worker.config.config_path", lambda: config)
    calls: list[str] = []
    state = {"down": False}

    def fake_query_embedder(host, model, dim, *, timeout):
        assert timeout == 8.0  # an agent is waiting: the short MCP timeout

        def embed(query: str) -> list[float]:
            calls.append(query)
            if state["down"]:
                raise OllamaUnavailable("Ollama not reachable (fake)")
            return axis(1) if "invoice" in query else axis(2)

        return embed

    monkeypatch.setattr(memory, "query_embedder", fake_query_embedder)
    from agenthandover_worker.mcp_server import build_mcp_server

    with patch("agenthandover_worker.mcp_server._get_kb_root", return_value=world.kb.root), \
         patch("agenthandover_worker.mcp_server._get_db_path", return_value=world.db):
        srv = build_mcp_server()
        tool = srv._tool_manager.get_tool("search_memory").fn

        def search(query: str, **kwargs) -> dict:
            return json.loads(tool(query, **kwargs))

        def switch(*, agents: bool, enabled: bool = True) -> None:
            config.write_text(
                f"[memory]\nenabled = {str(enabled).lower()}\n"
                f"agents_access = {str(agents).lower()}\n"
            )

        yield search, switch, calls, state
    world.close()


def test_switched_off_it_explains_how_to_turn_it_on_and_searches_nothing(mcp):
    search, switch, calls, _ = mcp
    result = search("acme invoice")  # no config.toml: the default is off
    assert result["mode"] == "not_allowed"
    assert result["results"] == []
    assert "Memory Settings" in result["reason"]
    # Agents are never handed the command that would turn it on.
    assert "agenthandover memory agents on" not in result["reason"]
    assert "Settings" in result["reason"]
    switch(agents=False)
    assert search("acme invoice")["mode"] == "not_allowed"
    assert calls == []


def test_switched_on_it_answers_at_once_without_a_restart(mcp):
    search, switch, calls, _ = mcp
    assert search("acme invoice")["mode"] == "not_allowed"
    switch(agents=True)  # re-read on the next call
    result = search("acme invoice")
    assert result["mode"] == "semantic"
    assert result["results"][0]["description"] == "Paying the Acme invoice"
    assert result["results"][0]["url_host"] == ""
    assert calls == ["acme invoice"]
    switch(agents=False)
    assert search("acme invoice")["mode"] == "not_allowed"


def test_results_never_carry_images_thumbnail_ids_or_paths(mcp):
    search, switch, _, _ = mcp
    switch(agents=True)
    result = search("acme invoice", limit=10)
    assert result["results"]
    for moment in result["results"]:
        assert set(moment) == FIELDS
    text = json.dumps(result)
    assert "thumb" not in text and ".ahthumb" not in text and ".jpg" not in text
    assert "/screenshots" not in text and '"inv"' not in text


def test_filters_reach_the_search(mcp):
    search, switch, _, _ = mcp
    switch(agents=True)
    result = search("acme invoice", app="slack")
    assert [m["app"] for m in result["results"]] == ["Slack"]
    result = search("acme invoice", since="2026-10-12T11:45:00Z")
    assert [m["app"] for m in result["results"]] == ["Slack"]
    assert "error" in search("acme invoice", since="yesterday-ish")


def test_memory_off_is_explained(mcp):
    search, switch, calls, _ = mcp
    switch(agents=True, enabled=False)
    result = search("acme invoice")
    assert result["mode"] == "not_allowed" and "Memory is off" in result["reason"]
    assert calls == []


def test_without_the_model_it_falls_back_to_keywords_and_says_so(mcp):
    search, switch, calls, state = mcp
    switch(agents=True)
    state["down"] = True
    result = search("the hero banner discussion")
    assert result["mode"] == "keyword_fallback"
    assert "Memory model is unavailable" in result["reason"]
    assert [m["description"] for m in result["results"]] == ["Discussing the hero banner"]
    assert set(result["results"][0]) == FIELDS
    # The next search does not wait on the model again (backoff).
    result = search("acme invoice")
    assert result["mode"] == "keyword_fallback" and "retrying after" in result["reason"]
    assert calls == ["the hero banner discussion"]

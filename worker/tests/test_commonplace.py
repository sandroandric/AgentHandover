"""Commonplace: one search over Skills, Memory, daily summaries and the
profile, each kind ranked on its own; Memory only behind the agents switch;
text only.  Through ``Commonplace.search`` and ``POST /commonplace/search``
(the MCP tool's own tests are in ``test_mcp_server.py``; J-commonplace runs
all of it through the MCP server and the worker)."""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from agenthandover_worker import commonplace as commonplace_mod
from agenthandover_worker.commonplace import MEMORY_FIELDS, Commonplace
from agenthandover_worker.config import Config, MemoryConfig
from agenthandover_worker.memory import MemoryIndex, keyword_search
from agenthandover_worker.model_profiles import MEMORY_MODEL
from agenthandover_worker.ollama import OllamaUnavailable
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.query_api import TOKEN_HEADER, QueryAPIServer
from fixtures.vlm_answers import generated_sop_template
from memory_world import DIM, NOW, World, axis

# Words a fake embedding model maps to an axis: a query and a document
# about the same thing point the same way.
TOPICS = (("domain", 1), ("invoice", 2), ("banner", 3))


def topic_vector(text: str) -> list[float]:
    return axis(next((i for word, i in TOPICS if word in text.lower()), 9))


def _skill(slug: str, title: str, **changes) -> dict:
    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    proc.update({
        "id": slug, "slug": slug, "title": title,
        "description": f"{title}, as the user does it",
        "goal": "Find out whether a domain is free to register and what it costs.",
        "lifecycle_state": "agent_ready",
        "constraints": {"trust_level": "suggest", "guardrails": []},
        "staleness": {"last_observed": datetime.now(timezone.utc).isoformat()},
    })
    proc.update(changes)
    return proc


class Hub:
    """A worker database and knowledge base holding three Skills, two
    Memory moments, two daily summaries and a profile, and a
    :class:`Commonplace` over them with fake embedding models."""

    def __init__(self, tmp_path, monkeypatch) -> None:
        self.world = w = World(tmp_path)
        kb = w.kb
        kb.save_procedure(_skill("check-domain", "Check Domain Availability"))
        kb.save_procedure(_skill(
            "old-domain-report", "Old domain report",
            constraints={"trust_level": "observe", "guardrails": []},
        ))
        kb.save_procedure(_skill(
            "deploy-staging", "Deploy to staging", lifecycle_state="draft",
        ))
        w.vkb.upsert("procedure", "check-domain", "domain", embedding=axis(1))
        mixed = 0.8 * np.asarray(axis(1)) + 0.6 * np.asarray(axis(4))
        w.vkb.upsert("procedure", "old-domain-report", "domain", embedding=[float(x) for x in mixed])
        w.vkb.upsert("procedure", "deploy-staging", "deploy", embedding=axis(6))

        w.moment("inv", NOW - timedelta(minutes=30), NOW - timedelta(minutes=25),
                 "Google Chrome", "Invoice INV-2041", "Paying the Acme invoice",
                 thumb_id="inv", vector=axis(2))
        w.moment("dom", NOW - timedelta(days=2), NOW - timedelta(days=2),
                 "Safari", "Namecheap", "Checking whether example.com the domain is free",
                 vector=axis(1))
        self.day("2026-10-10", ["Checked domain prices on Namecheap"], axis(1))
        self.day("2026-10-12", ["Paid the Acme invoice", "Reviewed the banner"], axis(2))
        kb.update_profile({"tools": {"browser": "Safari"}, "working_hours": {"start": "09:00"}})

        self.text_calls: list[tuple[str, float | None]] = []
        self.memory_calls: list[str] = []
        self.text_down = False
        self.memory_down = False

        def text_embed(texts, *, prefix=None, timeout=None):
            self.text_calls.append((texts[0], timeout))
            if self.text_down:
                raise OllamaUnavailable("Ollama not reachable (fake)")
            return [topic_vector(t) for t in texts]

        def memory_embed(query: str) -> list[float]:
            self.memory_calls.append(query)
            if self.memory_down:
                raise OllamaUnavailable("Ollama not reachable (fake)")
            return topic_vector(query)

        monkeypatch.setattr(w.vkb, "compute_embeddings", text_embed)
        self.index = MemoryIndex(w.db, model=MEMORY_MODEL, dim=DIM, embed_query=memory_embed)
        self.config = Config(memory=MemoryConfig(enabled=True, agents_access=False))

        def memory(cfg, query, *, keyword, **filters):
            if keyword:
                return keyword_search(self.index, query, **filters)
            return self.index.search(query, **filters)

        self.hub = Commonplace(
            knowledge_base=kb,
            vector_kb=lambda: (w.vkb, ""),
            load_config=lambda: self.config,
            memory=memory,
            db_path=lambda: w.db,
            embed_timeout=8.0,
            local_tz=timezone.utc,
        )

    def day(self, date: str, intents: list[str], vector: list[float]) -> None:
        self.world.kb.save_daily_summary(date, {
            "active_hours": 2.5, "task_count": len(intents),
            "tasks": [{"intent": i, "event_ids": ["ev-secret"], "urls": ["https://x.test/a"]}
                      for i in intents],
            "top_apps": [{"app": "Safari", "minutes": 90}, {"app": "Slack", "minutes": 20}],
        })
        self.world.vkb.upsert("daily_summary", date, " ".join(intents), embedding=vector)

    def agents(self, on: bool, *, enabled: bool = True) -> None:
        self.config = dataclasses.replace(
            self.config, memory=dataclasses.replace(
                self.config.memory, agents_access=on, enabled=enabled,
            ),
        )

    def search(self, query: str, **kwargs) -> dict:
        return self.hub.search(query, **kwargs)

    def close(self) -> None:
        self.index.close()
        self.world.close()


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    h = Hub(tmp_path, monkeypatch)
    yield h
    h.close()


# -- kinds --------------------------------------------------------------


def test_default_kinds_are_skills_memory_and_days_each_in_its_own_group(hub):
    result = hub.search("domain availability")
    assert set(result) == {"query", "skills", "memory", "days"}
    assert result["query"] == "domain availability"
    for kind in ("skills", "memory", "days"):
        assert set(result[kind]) == {"mode", "reason", "results"}


def test_skills_rank_by_meaning_and_carry_the_handoff_gate(hub):
    skills = hub.search("domain", kinds=["skills"])["skills"]
    assert skills["mode"] == "semantic" and skills["reason"] == ""
    first, second = skills["results"]
    assert first == {
        "slug": "check-domain", "title": "Check Domain Availability",
        "description": "Check Domain Availability, as the user does it",
        "score": 1.0, "handoff": "ready",
    }
    assert second["slug"] == "old-domain-report" and second["score"] == 0.8
    assert second["handoff"] == "held_back"
    assert any("trust level is 'observe'" in r for r in second["reasons"])
    # The draft points elsewhere: below the model's floor, not listed.
    assert "deploy-staging" not in json.dumps(skills)


def test_a_draft_skill_found_by_words_says_it_is_not_approved(hub):
    hub.text_down = True
    skills = hub.search("staging", kinds=["skills"])["skills"]
    assert skills["mode"] == "keyword_fallback"
    assert [(s["slug"], s["handoff"]) for s in skills["results"]] == [("deploy-staging", "not_approved")]


def test_days_rank_by_meaning_with_an_excerpt_and_no_ids_or_urls(hub):
    days = hub.search("the invoice", kinds=["days"])["days"]
    assert days["mode"] == "semantic"
    assert days["results"] == [{
        "date": "2026-10-12", "active_hours": 2.5, "task_count": 2,
        "apps": ["Safari", "Slack"],
        "excerpt": "Paid the Acme invoice; Reviewed the banner", "score": 1.0,
    }]


def test_profile_is_returned_whole_when_asked_for(hub):
    result = hub.search("anything", kinds=["profile"])
    assert set(result) == {"query", "profile"}
    assert result["profile"]["mode"] == "whole"
    (profile,) = result["profile"]["results"]
    assert profile["tools"] == {"browser": "Safari"}
    assert "updated_at" not in profile
    assert hub.text_calls == [] and hub.memory_calls == []


def test_the_text_query_is_embedded_once_for_skills_and_days(hub):
    hub.search("domain", kinds=["skills", "days"], limit=3)
    assert hub.text_calls == [("domain", 8.0)]


# -- Memory and the agents switch ---------------------------------------


def test_memory_waits_for_the_users_switch_and_follows_it_at_once(hub):
    off = hub.search("acme invoice")["memory"]
    assert off["mode"] == "not_allowed" and off["results"] == []
    assert "agenthandover memory agents on" in off["reason"]
    assert hub.memory_calls == []

    hub.agents(True)
    on = hub.search("acme invoice")["memory"]
    assert on["mode"] == "semantic"
    assert on["results"][0]["description"] == "Paying the Acme invoice"
    assert set(on["results"][0]) == set(MEMORY_FIELDS)

    hub.agents(False)
    assert hub.search("acme invoice")["memory"]["mode"] == "not_allowed"
    hub.agents(True, enabled=False)
    off = hub.search("acme invoice")["memory"]
    assert off["mode"] == "not_allowed" and "Memory is off" in off["reason"]
    assert hub.memory_calls == ["acme invoice"]


def test_memory_falls_back_to_words_and_backs_off(hub):
    hub.agents(True)
    hub.memory_down = True
    first = hub.search("acme invoice", kinds=["memory"])["memory"]
    assert first["mode"] == "keyword_fallback"
    assert "Memory model is unavailable" in first["reason"]
    assert [m["description"] for m in first["results"]] == ["Paying the Acme invoice"]
    second = hub.search("acme invoice", kinds=["memory"])["memory"]
    assert "retrying after" in second["reason"]
    assert hub.memory_calls == ["acme invoice"]


def test_without_a_database_memory_is_not_searched(hub, tmp_path):
    hub.agents(True)
    hub.hub._db_path = lambda: tmp_path / "none.db"
    memory = hub.search("acme invoice", kinds=["memory"])["memory"]
    assert memory["mode"] == "not_allowed" and "no worker database" in memory["reason"]


# -- filters -------------------------------------------------------------


def test_since_and_until_keep_moments_and_days_in_range_and_leave_skills_alone(hub):
    hub.agents(True)
    recent = hub.search("domain", since="2026-10-11T00:00:00Z")
    assert [s["slug"] for s in recent["skills"]["results"]][0] == "check-domain"
    assert all(m["first_ts"] >= "2026-10-11" for m in recent["memory"]["results"])
    assert "Checking whether example.com the domain is free" not in json.dumps(recent["memory"])
    assert recent["days"]["results"] == []  # the domain day is the 10th

    older = hub.search("domain", until="2026-10-10T23:59:59Z")
    assert [d["date"] for d in older["days"]["results"]] == ["2026-10-10"]
    assert [m["app"] for m in older["memory"]["results"]] == ["Safari"]


def test_limit_is_per_kind(hub):
    hub.agents(True)
    result = hub.search("domain", limit=1)
    assert len(result["skills"]["results"]) == 1
    assert len(result["memory"]["results"]) == 1


def test_days_by_words_when_the_text_model_is_down_with_the_range(hub):
    hub.text_down = True
    days = hub.search("banner review", kinds=["days"])["days"]
    assert days["mode"] == "keyword_fallback"
    assert "embedding model unavailable" in days["reason"]
    assert [d["date"] for d in days["results"]] == ["2026-10-12"]
    assert days["results"][0]["score"] == 1.0
    days = hub.search("banner", kinds=["days"], until="2026-10-11T00:00:00Z")["days"]
    assert days["results"] == []


def test_days_not_embedded_yet_are_searched_by_words(hub):
    hub.world.vkb.delete_by_source("daily_summary", "2026-10-10")
    hub.world.vkb.delete_by_source("daily_summary", "2026-10-12")
    days = hub.search("namecheap prices", kinds=["days"])["days"]
    assert days == {
        "mode": "keyword_fallback", "reason": "no daily summary has been embedded yet",
        "results": [days["results"][0]],
    }
    assert days["results"][0]["date"] == "2026-10-10"


def test_the_text_model_down_falls_back_and_is_not_retried_every_search(hub, monkeypatch):
    hub.text_down = True
    first = hub.search("domain", kinds=["skills", "days"])
    assert first["skills"]["mode"] == first["days"]["mode"] == "keyword_fallback"
    assert [s["slug"] for s in first["skills"]["results"]][:1] == ["check-domain"]
    second = hub.search("domain", kinds=["skills"])
    assert "retrying after" in second["skills"]["reason"]
    assert len(hub.text_calls) == 1
    hub.text_down = False
    later = commonplace_mod.time.monotonic() + commonplace_mod.EMBED_RETRY_AFTER_S + 1
    monkeypatch.setattr(commonplace_mod.time, "monotonic", lambda: later)
    assert hub.search("domain", kinds=["skills"])["skills"]["mode"] == "semantic"


# -- text only ------------------------------------------------------------


def test_results_are_text_never_vectors_images_ids_or_paths(hub):
    hub.agents(True)
    text = json.dumps(hub.search("domain invoice banner", kinds=list(commonplace_mod.KINDS), limit=50))
    for leaked in ("thumb", ".ahthumb", ".jpg", "embedding", "event_ids", "ev-secret",
                   "x.test", str(hub.world.data_dir), '"inv"', '"dom"', '"rows"'):
        assert leaked not in text, leaked


# -- bad requests ---------------------------------------------------------


@pytest.mark.parametrize(("kwargs", "error"), [
    ({"query": ""}, "'query' must be a non-empty string"),
    ({"query": "x", "kinds": []}, "'kinds' must be a non-empty list"),
    ({"query": "x", "kinds": "skills"}, "'kinds' must be a non-empty list"),
    ({"query": "x", "kinds": ["skills", "files"]}, "unknown kind(s): files"),
    ({"query": "x", "limit": 0}, "'limit' must be an integer from 1 to 50"),
    ({"query": "x", "limit": True}, "'limit' must be an integer from 1 to 50"),
    ({"query": "x", "since": "yesterday-ish"}, "'since' must be an ISO-8601 time"),
    ({"query": "x", "since": "2026-10-12T00:00:00Z", "until": "2026-10-11T00:00:00Z"},
     "'since' is after 'until'"),
])
def test_bad_requests_say_what_is_wrong(hub, kwargs, error):
    with pytest.raises(ValueError) as exc:
        hub.hub.search(**kwargs)
    assert error in str(exc.value)


# -- POST /commonplace/search --------------------------------------------


def _post(srv, body, *, raw: bytes | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}/commonplace/search",
        data=raw if raw is not None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", TOKEN_HEADER: srv.token}, method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture()
def api(hub):
    srv = QueryAPIServer(hub.world.kb, port=0, commonplace=hub.hub)
    srv.start()
    yield srv
    srv.stop()


def test_api_answers_the_same_groups(hub, api):
    status, body = _post(api, {"query": "domain", "kinds": ["skills", "days"], "limit": 2})
    assert status == 200
    assert body == hub.search("domain", kinds=["skills", "days"], limit=2)


def test_api_keeps_memory_behind_the_switch(hub, api):
    status, body = _post(api, {"query": "acme invoice", "kinds": ["memory"]})
    assert status == 200 and body["memory"]["mode"] == "not_allowed"
    hub.agents(True)
    status, body = _post(api, {"query": "acme invoice", "kinds": ["memory"],
                               "since": "2026-10-12T00:00:00Z"})
    assert body["memory"]["results"][0]["app"] == "Google Chrome"


def test_api_bad_requests_are_400(api):
    assert _post(api, {"query": "x", "kinds": ["nope"]}) == (
        400, {"error": "unknown kind(s): nope (known: skills, memory, days, profile)"},
    )
    assert _post(api, {"query": "x", "app": "Slack"}) == (400, {"error": "unknown field(s): app"})
    assert _post(api, {"limit": 3})[0] == 400
    assert _post(api, None, raw=b"[1]") == (400, {"error": "Request body must be a JSON object"})
    assert _post(api, None, raw=b"{")[0] == 400
    assert _post(api, None, raw=b"")[0] == 400


def test_api_without_commonplace_is_501(hub):
    srv = QueryAPIServer(hub.world.kb, port=0)
    srv.start()
    try:
        assert _post(srv, {"query": "x"})[0] == 501
    finally:
        srv.stop()


def test_a_day_excerpt_names_each_task_once():
    """A day often repeats one task; the excerpt says it once, in order."""
    from agenthandover_worker.commonplace import day_item

    item = day_item("2026-09-25", {"tasks": [
        {"intent": "editing a document in TextEdit"},
        {"intent": "reviewing the models"},
        {"intent": "editing a document in TextEdit"},
    ]}, 0.6)
    assert item["excerpt"] == "editing a document in TextEdit; reviewing the models"

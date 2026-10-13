"""Commonplace: everything AgentHandover knows locally, as one searchable hub.

Commonplace holds four kinds of knowledge, each searched by its own scorer
and returned in its own group (scores of different kinds are never mixed):

* ``skills`` — the Skills learned from the user's workflows, by meaning
  (``[embedding] model`` vectors) or, when the model cannot answer, by the
  query's words in titles, descriptions, tags and steps; each says whether
  the hand-off gate lets an agent run it.
* ``memory`` — moments of what was on the user's screen
  (:class:`memory.MemoryIndex`), only while the user lets agents search
  Memory (``[memory] agents_access``, read on every search); otherwise the
  group says how the user can allow it.
* ``days`` — the daily summaries, by meaning (their ``daily_summary``
  vectors) or by words, with their date and a short excerpt.
* ``profile`` — the user's profile (tools, working hours, writing style);
  not ranked, returned whole when asked for.

Results are text: never vectors, images, thumbnail ids or file paths.
The MCP server (``search_commonplace``) and the query API
(``POST /commonplace/search``, which ``agenthandover search`` calls) both
answer through :meth:`Commonplace.search`.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

KINDS = ("skills", "memory", "days", "profile")
DEFAULT_KINDS = ("skills", "memory", "days")
DEFAULT_LIMIT = 5
MAX_LIMIT = 50
# After the embedding model failed, searches skip it for this long instead
# of each waiting on it again (an agent is waiting).
EMBED_RETRY_AFTER_S = 60.0
# Excerpt of a daily summary: its tasks' intents, this long at most.
EXCERPT_CHARS = 400

MEMORY_FIELDS = ("app", "window_title", "url_host", "description", "first_ts", "last_ts", "score")

MEMORY_NOT_ALLOWED = (
    "The user has not let agents search their Memory. They can turn it on "
    "in AgentHandover's Memory Settings (\"Let agents search my Memory\") or "
    "with `agenthandover memory agents on`."
)
MEMORY_OFF = "Memory is off. The user can turn it on in AgentHandover's Settings."

_STOPWORDS = frozenset(
    "a an and at by for from how i in into is it my of on or the to with".split()
)

# What every connected agent is told (the MCP server's instructions).
MCP_INSTRUCTIONS = (
    "AgentHandover watches how the user works, on their Mac, and keeps what it "
    "learns in Commonplace: Skills (step-by-step procedures learned from the "
    "user's own workflows), Memory (what was on the user's screen, searchable "
    "by meaning; only when the user has let agents search it), daily summaries "
    "of their work, and their profile (tools, working hours, writing style). "
    "Before asking the user about their own work, tools, habits or something "
    "they saw, call search_commonplace: results come grouped by kind, as text "
    "(never images). To run a Skill, use list_ready_skills and get_skill and "
    "report progress with the report_execution_* tools."
)

# What `agenthandover connect` writes into the agent's own instructions.
AGENT_NOTE = (
    "## Commonplace\n"
    "\n"
    "AgentHandover keeps what it learns about the user's work in Commonplace, on "
    "their Mac: Skills (workflows learned from watching them), Memory (what was "
    "on their screen, searchable by meaning, only if the user lets agents search "
    "it), daily summaries and their profile. Before asking the user about their "
    "own work, tools or habits, search it: the MCP tool "
    "`search_commonplace(query, kinds?, since?, until?, limit?)` of the "
    "`agenthandover` server, or `agenthandover search \"<query>\"` in a terminal "
    "(`--json` for machine-readable output). Results are grouped by kind "
    "(skills, memory, days; ask for `profile` by kind) and are text only.\n"
)


def _model_errors() -> tuple[type[BaseException], ...]:
    """What a query embedding raises when its model cannot answer.  (Imported
    here: this module stays light for the MCP server's import.)"""
    from agenthandover_worker.ollama import OllamaError

    return (ConnectionError, TimeoutError, OllamaError)


def _words(query: str) -> list[str]:
    """The query's words that carry meaning (no stopwords, 2+ letters);
    the whole query when none does."""
    return [
        w for w in re.findall(r"\w+", query.lower())
        if w not in _STOPWORDS and len(w) > 1
    ] or [query.lower().strip()]


def keyword_skills(procedures: list[dict], query: str, limit: int) -> list[tuple[dict, float]]:
    """Score each Skill by the share of query words it contains: a word in
    the title counts 1, one only in the description, tags or steps 0.5.
    ``(procedure, score)`` best first, scores above 0 only."""
    from agenthandover_worker.procedure_schema import procedure_steps

    words = _words(query)
    matches = []
    for proc in procedures:
        title = f"{proc.get('title', '')} {proc.get('short_title', '')}".lower()
        body = " ".join([
            proc.get("description", ""),
            " ".join(proc.get("tags", [])),
            *(s.get("action") or s.get("description", "") for s in procedure_steps(proc)),
        ]).lower()
        score = sum(
            1.0 if w in title else 0.5 if w in body else 0.0 for w in words
        ) / len(words)
        if score > 0:
            matches.append((proc, round(score, 2)))
    matches.sort(key=lambda m: m[1], reverse=True)
    return matches[:limit]


def _summary_apps(summary: dict) -> list[str]:
    """The day's top apps, by name (``top_apps`` holds ``{app, minutes}``)."""
    apps = []
    for entry in summary.get("top_apps") or []:
        name = entry.get("app") if isinstance(entry, dict) else entry
        if isinstance(name, str) and name:
            apps.append(name)
    return apps[:5]


def _summary_intents(summary: dict) -> list[str]:
    intents = []
    for task in summary.get("tasks") or []:
        intent = task.get("intent") if isinstance(task, dict) else task
        if isinstance(intent, str) and intent.strip():
            intents.append(intent.strip())
    return intents


def day_item(date: str, summary: dict, score: float) -> dict:
    """A daily summary as a ``days`` result: date, size, apps and an
    excerpt of its tasks (no event ids, URLs or paths)."""
    excerpt = "; ".join(_summary_intents(summary))
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[:EXCERPT_CHARS - 1].rstrip() + "…"
    return {
        "date": date,
        "active_hours": summary.get("active_hours"),
        "task_count": summary.get("task_count"),
        "apps": _summary_apps(summary),
        "excerpt": excerpt,
        "score": score,
    }


def skill_item(proc: dict, score: float) -> dict:
    """A Skill as a ``skills`` result, with what the hand-off gate says:
    ``handoff`` is ``ready`` (an agent may run it), ``held_back`` (approved,
    but a check fails: ``reasons``) or ``not_approved``."""
    from agenthandover_worker.ledger import handoff_gate

    item = {
        "slug": proc.get("id", ""),
        "title": proc.get("title", ""),
        "description": proc.get("description", ""),
        "score": score,
    }
    if proc.get("lifecycle_state") != "agent_ready":
        item["handoff"] = "not_approved"
    else:
        gate = handoff_gate(proc)
        item["handoff"] = "ready" if gate.ok else "held_back"
        if not gate.ok:
            item["reasons"] = gate.reasons
    return item


def group(mode: str, reason: str, results: list) -> dict:
    return {"mode": mode, "reason": reason, "results": results}


class Commonplace:
    """Search over Skills, Memory, daily summaries and the profile.

    *knowledge_base* is a ``KnowledgeBase`` or a callable returning one;
    *vector_kb* returns ``(VectorKB | None, reason)`` (the reason says why
    there is none); *load_config* returns the current ``Config`` (read on
    each search, so the agents switch applies at once); *memory* runs a
    Memory search: ``memory(config, query, keyword=bool, since=, until=,
    app=, limit=)`` → moments, raising ``OllamaError``/``ConnectionError``/
    ``TimeoutError``/``ValueError`` when the query cannot be embedded (None:
    Memory is not available here); *db_path* returns the worker database
    (Memory is searched only when it exists).  *embed_timeout* bounds the
    query's embedding: someone is waiting.
    """

    def __init__(
        self,
        *,
        knowledge_base: Any,
        vector_kb: Callable[[], tuple[Any, str]],
        load_config: Callable[[], Any],
        memory: Callable[..., list[dict]] | None,
        db_path: Callable[[], Path],
        embed_timeout: float,
        local_tz: tzinfo | None = None,
    ) -> None:
        self._kb = knowledge_base if callable(knowledge_base) else (lambda: knowledge_base)
        self._vector_kb = vector_kb
        self._load_config = load_config
        self._memory = memory
        self._db_path = db_path
        self.embed_timeout = embed_timeout
        self.local_tz = local_tz
        # (monotonic time until which a model is skipped, why)
        self._embed_down: tuple[float, str] = (0.0, "")
        self._memory_down: tuple[float, str] = (0.0, "")

    # -- the request ----------------------------------------------------

    def search(
        self,
        query: str,
        kinds: list[str] | tuple[str, ...] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> dict:
        """``{"query", <kind>: {"mode", "reason", "results"}, ...}`` for each
        of *kinds* (default ``skills``, ``memory``, ``days``), each kind's
        results ranked by its own scorer, at most *limit* each.  *since* /
        *until* (ISO-8601) keep Memory moments and days that overlap the
        range; Skills and the profile are not tied to a time.

        Raises ValueError (saying what is wrong) for a bad request.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        if kinds is None:
            kinds = DEFAULT_KINDS
        if (
            isinstance(kinds, str) or not isinstance(kinds, (list, tuple)) or not kinds
            or not all(isinstance(k, str) for k in kinds)
        ):
            raise ValueError(f"'kinds' must be a non-empty list of: {', '.join(KINDS)}")
        unknown = [k for k in kinds if k not in KINDS]
        if unknown:
            raise ValueError(
                f"unknown kind(s): {', '.join(unknown)} (known: {', '.join(KINDS)})"
            )
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"'limit' must be an integer from 1 to {MAX_LIMIT}")
        from agenthandover_worker.memory_forget import parse_time

        since_at = parse_time(since, "since") if since is not None else None
        until_at = parse_time(until, "until") if until is not None else None
        if since_at is not None and until_at is not None and since_at > until_at:
            raise ValueError("'since' is after 'until'")

        vector: dict[str, Any] = {}  # the text query's vector, embedded once
        answer: dict[str, Any] = {"query": query}
        for kind in dict.fromkeys(kinds):
            if kind == "skills":
                answer[kind] = self._skills_group(query, limit, vector)
            elif kind == "memory":
                answer[kind] = self.memory(query, since=since_at, until=until_at, limit=limit)
            elif kind == "days":
                answer[kind] = self._days_group(query, since_at, until_at, limit, vector)
            else:
                answer[kind] = group("whole", "", [self._profile()])
        return answer

    # -- the text model (Skills and days) -------------------------------

    def _query_vector(self, vkb: Any, query: str, cache: dict) -> tuple[list[float] | None, str]:
        """The query embedded with the configured text model (once per
        search), or ``(None, why not)``."""
        if "vector" in cache:
            return cache["vector"]
        down_until, down_reason = self._embed_down
        if time.monotonic() < down_until:
            cache["vector"] = (None, down_reason)
            return cache["vector"]
        try:
            vectors = vkb.compute_embeddings(
                [query], prefix=vkb.profile.query_prefix, timeout=self.embed_timeout,
            )
        except _model_errors() as exc:
            reason = f"embedding model unavailable: {exc}"
            logger.warning("Commonplace search: %s", reason)
            self._embed_down = (
                time.monotonic() + EMBED_RETRY_AFTER_S,
                f"{reason} (retrying after {EMBED_RETRY_AFTER_S:.0f}s)",
            )
            cache["vector"] = (None, reason)
            return cache["vector"]
        if not vectors or not vectors[0]:
            cache["vector"] = (None, "embedding model returned no vector")
        else:
            cache["vector"] = (vectors[0], "")
        return cache["vector"]

    # -- skills ---------------------------------------------------------

    def skills(
        self, query: str, limit: int, cache: dict | None = None,
    ) -> tuple[str, str, list[tuple[dict, float]]]:
        """``(mode, reason, [(procedure, score)])``: by meaning when the
        Skills are embedded and the model answers, else by words
        (``keyword_fallback``, *reason* says why)."""
        cache = {} if cache is None else cache
        vkb, reason = self._vector_kb()
        procedures = {p.get("id", ""): p for p in self._kb().list_procedures()}
        if vkb is not None:
            if vkb.count("procedure") == 0:
                reason = "no Skill has been embedded yet"
            else:
                vector, reason = self._query_vector(vkb, query, cache)
                if vector is not None:
                    hits = vkb.search_by_vector(
                        vector, top_k=limit, source_types=["procedure"],
                        min_score=vkb.thresholds.min_score,
                    )
                    return "semantic", "", [
                        (procedures.get(h.source_id) or {"id": h.source_id}, h.score)
                        for h in hits
                    ]
        return "keyword_fallback", reason, keyword_skills(list(procedures.values()), query, limit)

    def _skills_group(self, query: str, limit: int, cache: dict) -> dict:
        mode, reason, hits = self.skills(query, limit, cache)
        return group(mode, reason, [skill_item(proc, score) for proc, score in hits])

    # -- memory ---------------------------------------------------------

    def memory(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        app: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> dict:
        """The ``memory`` group: moments as text (:data:`MEMORY_FIELDS`), or
        ``not_allowed`` with how the user can allow it."""
        cfg = self._load_config()
        if not cfg.memory.agents_access:
            return group("not_allowed", MEMORY_NOT_ALLOWED, [])
        if not cfg.memory.enabled:
            return group("not_allowed", MEMORY_OFF, [])
        if self._memory is None:
            return group("not_allowed", "Memory is not available in this process.", [])
        db_path = self._db_path()
        if not db_path.exists():
            return group("not_allowed", f"no worker database at {db_path}", [])
        filters = {"since": since, "until": until, "app": app or None, "limit": limit}

        def moments(found: list[dict]) -> list[dict]:
            return [{k: m[k] for k in MEMORY_FIELDS} for m in found]

        down_until, reason = self._memory_down
        if time.monotonic() >= down_until:
            try:
                return group("semantic", "", moments(
                    self._memory(cfg, query, keyword=False, **filters)
                ))
            except (*_model_errors(), ValueError) as exc:
                reason = f"the Memory model is unavailable: {exc}"
                logger.warning("Memory search: %s", reason)
                self._memory_down = (
                    time.monotonic() + EMBED_RETRY_AFTER_S,
                    f"{reason} (retrying after {EMBED_RETRY_AFTER_S:.0f}s)",
                )
        return group("keyword_fallback", reason, moments(
            self._memory(cfg, query, keyword=True, **filters)
        ))

    # -- days -----------------------------------------------------------

    def _local_day(self, moment: datetime) -> str:
        local = moment.astimezone(self.local_tz) if self.local_tz else moment.astimezone()
        return local.date().isoformat()

    def _days_group(
        self,
        query: str,
        since: datetime | None,
        until: datetime | None,
        limit: int,
        cache: dict,
    ) -> dict:
        """Days whose summary matches *query*, by meaning (their
        ``daily_summary`` vectors) or by words; *since*/*until* keep the
        local days that overlap the range."""
        first = self._local_day(since) if since is not None else None
        last = self._local_day(until) if until is not None else None

        def in_range(day: str) -> bool:
            return (first is None or day >= first) and (last is None or day <= last)

        kb = self._kb()
        vkb, reason = self._vector_kb()
        if vkb is not None:
            embedded = vkb.count("daily_summary")
            if embedded == 0:
                reason = "no daily summary has been embedded yet"
            else:
                vector, reason = self._query_vector(vkb, query, cache)
                if vector is not None:
                    found = []
                    for hit in vkb.search_by_vector(
                        vector, top_k=embedded, source_types=["daily_summary"],
                        min_score=vkb.thresholds.min_score,
                    ):
                        if not in_range(hit.source_id):
                            continue
                        summary = kb.get_daily_summary(hit.source_id)
                        if summary is not None:
                            found.append(day_item(hit.source_id, summary, hit.score))
                        if len(found) == limit:
                            break
                    return group("semantic", "", found)
        words = _words(query)
        found = []
        for day in kb.list_daily_summaries(limit=100_000):
            if not in_range(day):
                continue
            summary = kb.get_daily_summary(day)
            if summary is None:
                continue
            text = " ".join(_summary_intents(summary) + _summary_apps(summary)).lower()
            score = round(sum(w in text for w in words) / len(words), 2)
            if score > 0:
                found.append(day_item(day, summary, score))
        # Best first, then newest (the list is newest first; sort is stable).
        found.sort(key=lambda d: d["score"], reverse=True)
        return group("keyword_fallback", reason, found[:limit])

    # -- profile --------------------------------------------------------

    def _profile(self) -> dict:
        profile = dict(self._kb().get_profile())
        profile.pop("updated_at", None)
        return profile

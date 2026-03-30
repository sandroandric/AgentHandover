"""Cross-day session linker for AgentHandover.

Links tasks across daily summaries by detecting recurring intents using
Jaccard similarity on normalized tokens.  When the same kind of task
appears on multiple days, it creates a :class:`LinkedTask` that tracks
total duration, span, and status.

Linked tasks are persisted at ``{kb_root}/observations/session_links.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

from agenthandover_worker.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner
    from agenthandover_worker.vector_kb import VectorKB

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "is", "was", "are", "were", "this", "that",
}

# Similarity threshold for linking tasks by intent
_SIMILARITY_THRESHOLD = 0.4

# Days of inactivity before a link is marked stale
_STALE_DAYS = 14


@dataclass
class LinkedTask:
    """A task that spans multiple sessions across different days."""

    link_id: str
    intent: str
    sessions: list[dict]  # [{date, task_index, duration_minutes, apps}]
    total_duration_minutes: int
    first_seen: str
    last_seen: str
    span_days: int
    status: str  # "active", "completed", "stale"
    matched_procedure: str | None = None
    span_id: str | None = None


class SessionLinker:
    """Links related tasks across daily summaries."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_reasoner: "LLMReasoner | None" = None,
        vector_kb: "VectorKB | None" = None,
    ) -> None:
        self._kb = knowledge_base
        self._llm_reasoner = llm_reasoner
        self._vector_kb = vector_kb
        self._links: list[LinkedTask] = []
        self._load_links()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_daily_summaries(
        self, lookback_days: int = 30
    ) -> list[LinkedTask]:
        """Scan recent daily summaries and link recurring tasks.

        Returns all links (including newly created ones).
        """
        dates = self._kb.list_daily_summaries(limit=lookback_days)
        if not dates:
            return list(self._links)

        # Collect all tasks from daily summaries
        all_tasks: list[dict] = []
        for date_str in dates:
            summary = self._kb.get_daily_summary(date_str)
            if summary is None:
                continue
            tasks = summary.get("tasks", [])
            for idx, task in enumerate(tasks):
                intent = task.get("intent", task.get("description", ""))
                if not intent:
                    continue
                all_tasks.append({
                    "date": date_str,
                    "task_index": idx,
                    "intent": intent,
                    "duration_minutes": task.get("duration_minutes", 0),
                    "apps": task.get("apps", []),
                    "matched_procedure": task.get("matched_procedure", None),
                })

        # Group tasks by intent similarity
        used_indices: set[int] = set()
        for i, task_a in enumerate(all_tasks):
            if i in used_indices:
                continue

            # Check if this task matches an existing link
            matched_link = self._find_matching_link(task_a)
            if matched_link is not None:
                # Add session if not already present
                session_key = (task_a["date"], task_a["task_index"])
                existing_keys = {
                    (s["date"], s["task_index"])
                    for s in matched_link.sessions
                }
                if session_key not in existing_keys:
                    matched_link.sessions.append({
                        "date": task_a["date"],
                        "task_index": task_a["task_index"],
                        "duration_minutes": task_a["duration_minutes"],
                        "apps": task_a["apps"],
                    })
                    self._update_link_stats(matched_link)
                used_indices.add(i)
                continue

            # Find similar tasks to group together
            group = [task_a]
            used_indices.add(i)

            for j, task_b in enumerate(all_tasks):
                if j in used_indices:
                    continue
                if task_a["date"] == task_b["date"]:
                    continue  # Only link across different days

                similarity = self._intent_similarity(
                    task_a["intent"], task_b["intent"]
                )
                # Also consider matching procedure as a link signal
                proc_match = (
                    task_a["matched_procedure"] is not None
                    and task_a["matched_procedure"] == task_b["matched_procedure"]
                )

                # Ambiguous range: use LLM semantic check to break ties.
                # Wider range (0.15-0.6) catches both "below threshold but
                # might be same" AND "above threshold but might be different".
                is_match = similarity >= _SIMILARITY_THRESHOLD
                if (
                    not proc_match
                    and self._llm_reasoner is not None
                    and 0.15 <= similarity <= 0.6
                ):
                    llm_verdict = self._semantic_check(
                        task_a["intent"], task_b["intent"],
                    )
                    if llm_verdict is True:
                        is_match = True

                if is_match or proc_match:
                    group.append(task_b)
                    used_indices.add(j)

            # Only create a link if the task spans multiple days
            unique_dates = {t["date"] for t in group}
            if len(unique_dates) >= 2:
                sessions = [
                    {
                        "date": t["date"],
                        "task_index": t["task_index"],
                        "duration_minutes": t["duration_minutes"],
                        "apps": t["apps"],
                    }
                    for t in group
                ]
                sorted_dates = sorted(unique_dates)
                first = sorted_dates[0]
                last = sorted_dates[-1]
                first_dt = datetime.strptime(first, "%Y-%m-%d")
                last_dt = datetime.strptime(last, "%Y-%m-%d")
                span = (last_dt - first_dt).days + 1

                total_dur = sum(t["duration_minutes"] for t in group)

                # Determine procedure (use the most common one)
                procs = [
                    t["matched_procedure"]
                    for t in group
                    if t["matched_procedure"] is not None
                ]
                matched_proc = max(set(procs), key=procs.count) if procs else None

                link = LinkedTask(
                    link_id=str(uuid.uuid4()),
                    intent=task_a["intent"],
                    sessions=sessions,
                    total_duration_minutes=total_dur,
                    first_seen=first,
                    last_seen=last,
                    span_days=span,
                    status="active",
                    matched_procedure=matched_proc,
                )
                self._links.append(link)

        # Update stale status
        self._update_stale_status()
        self._save_links()
        return list(self._links)

    def get_active_links(self) -> list[LinkedTask]:
        """Return only links with status 'active'."""
        return [link for link in self._links if link.status == "active"]

    def mark_completed(self, link_id: str) -> bool:
        """Mark a linked task as completed.

        Returns ``True`` if the link was found and marked.
        """
        for link in self._links:
            if link.link_id == link_id:
                link.status = "completed"
                self._save_links()
                return True
        return False

    # ------------------------------------------------------------------
    # Intent similarity
    # ------------------------------------------------------------------

    def _normalize_intent(self, intent: str) -> str:
        """Normalize an intent string: lowercase, strip punctuation, remove stop words."""
        text = intent.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return " ".join(tokens)

    def _intent_similarity(self, a: str, b: str) -> float:
        """Semantic similarity via vector KB, with Jaccard fallback."""
        if self._vector_kb is not None:
            try:
                from agenthandover_worker.vector_kb import VectorKB
                embs = self._vector_kb.compute_embeddings([a, b])
                if len(embs) == 2 and embs[0] and embs[1]:
                    return VectorKB.cosine_similarity(embs[0], embs[1])
            except Exception:
                logger.debug("Vector similarity failed, falling back to Jaccard")
        return self._jaccard_similarity(a, b)

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Jaccard similarity on normalized tokens (fallback)."""
        tokens_a = set(self._normalize_intent(a).split())
        tokens_b = set(self._normalize_intent(b).split())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _semantic_check(self, intent_a: str, intent_b: str) -> bool | None:
        """Use LLM to determine if two intents describe the same workflow.

        Returns True if the LLM says YES, False if NO, None on
        failure/ambiguity/abstention.
        """
        if self._llm_reasoner is None:
            return None

        prompt = (
            "Are these two task descriptions the same recurring workflow? "
            f"Task A: '{intent_a}' Task B: '{intent_b}'. "
            "Answer YES or NO."
        )

        try:
            result = self._llm_reasoner.reason_yesno(
                prompt, caller="session_linker.semantic_check",
            )
            if result.success and not result.abstained:
                return result.value  # True, False, or None (ambiguous)
        except Exception:
            logger.debug("LLM semantic check failed", exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_matching_link(self, task: dict) -> LinkedTask | None:
        """Find an existing link that matches this task by intent, procedure, or span_id."""
        # Prefer span_id match first (continuity graph linkage)
        task_span_id = task.get("span_id")
        if task_span_id:
            for link in self._links:
                if link.status == "completed":
                    continue
                if link.span_id == task_span_id:
                    return link

        for link in self._links:
            if link.status == "completed":
                continue

            # Check procedure match
            if (
                task.get("matched_procedure") is not None
                and link.matched_procedure == task["matched_procedure"]
            ):
                return link

            # Check intent similarity (with LLM for ambiguous range)
            similarity = self._intent_similarity(link.intent, task.get("intent", ""))
            if similarity >= _SIMILARITY_THRESHOLD:
                return link

            # LLM semantic check for ambiguous similarity — same logic
            # as the new-group path so existing links don't fragment
            if (
                self._llm_reasoner is not None
                and 0.15 <= similarity < _SIMILARITY_THRESHOLD
            ):
                is_same = self._semantic_check(link.intent, task.get("intent", ""))
                if is_same is True:
                    return link

        return None

    def _update_link_stats(self, link: LinkedTask) -> None:
        """Recalculate stats for a link after adding sessions."""
        link.total_duration_minutes = sum(
            s["duration_minutes"] for s in link.sessions
        )
        dates = sorted({s["date"] for s in link.sessions})
        if dates:
            link.first_seen = dates[0]
            link.last_seen = dates[-1]
            first_dt = datetime.strptime(dates[0], "%Y-%m-%d")
            last_dt = datetime.strptime(dates[-1], "%Y-%m-%d")
            link.span_days = (last_dt - first_dt).days + 1

    def _update_stale_status(self) -> None:
        """Mark active links as stale if no session in the last N days."""
        now = datetime.now(timezone.utc).date()
        for link in self._links:
            if link.status != "active":
                continue
            try:
                last = datetime.strptime(link.last_seen, "%Y-%m-%d").date()
                if (now - last).days > _STALE_DAYS:
                    link.status = "stale"
            except ValueError:
                pass

    def _load_links(self) -> None:
        """Load links from persistent storage."""
        path = self._kb.root / "observations" / "session_links.json"
        if not path.is_file():
            self._links = []
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._links = []
            return

        self._links = []
        for item in data.get("links", []):
            self._links.append(
                LinkedTask(
                    link_id=item["link_id"],
                    intent=item["intent"],
                    sessions=item["sessions"],
                    total_duration_minutes=item["total_duration_minutes"],
                    first_seen=item["first_seen"],
                    last_seen=item["last_seen"],
                    span_days=item["span_days"],
                    status=item.get("status", "active"),
                    matched_procedure=item.get("matched_procedure"),
                    span_id=item.get("span_id"),
                )
            )

    def _save_links(self) -> None:
        """Persist links to session_links.json using atomic write."""
        data = {
            "links": [
                {
                    "link_id": link.link_id,
                    "intent": link.intent,
                    "sessions": link.sessions,
                    "total_duration_minutes": link.total_duration_minutes,
                    "first_seen": link.first_seen,
                    "last_seen": link.last_seen,
                    "span_days": link.span_days,
                    "status": link.status,
                    "matched_procedure": link.matched_procedure,
                    "span_id": link.span_id,
                }
                for link in self._links
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / "observations" / "session_links.json"
        self._kb.atomic_write_json(path, data)

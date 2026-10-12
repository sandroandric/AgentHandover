"""Cross-day session linker for AgentHandover.

Links tasks across daily summaries by detecting recurring intents using
Jaccard similarity on normalized tokens.  When the same kind of task
appears on multiple days, it creates a :class:`LinkedTask` that tracks
total duration, span, and status.

Linked tasks are persisted at ``{kb_root}/observations/session_links.json``,
together with the model's YES/NO judgements of intent pairs, so a pair is
asked about once rather than on every daily pass.

Cost per daily pass: the intents not yet embedded are embedded in one
batched call, every pair is compared in memory, tasks already in a link are
not compared again, and at most :data:`_MAX_LLM_CHECKS_PER_PASS` new pairs
go to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
import math
import operator
import re
import uuid
from typing import TYPE_CHECKING

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.redaction import redact

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner
    from agenthandover_worker.vector_kb import VectorKB

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "is", "was", "are", "were", "this", "that",
}

# Linking thresholds, one per similarity metric: Jaccard over normalised
# intent tokens, and cosine of clustering-prefixed vectors of the vector
# KB's embedding model.  A score from one metric is never compared with the
# other's threshold.  The cosine threshold and its LLM band are the
# embedding profile's (``linker_cos``, ``linker_band``); nomic-embed-text's
# are measured on intent texts in tests/journeys/test_linker_threshold.py:
# same-task pairs score 0.76-0.96, and the closest different-task pair (a
# Stripe payout reconciliation vs. failed payments in the Stripe dashboard)
# 0.88, so the segmenter's 0.83 would link different tasks.  0.89 links no
# two tasks; the pairs in between go to the LLM band.
_JACCARD_THRESHOLD = 0.4
# Just below its metric's threshold, an LLM YES links two intents.  Only
# below: at or above the threshold the intents link without a model call,
# so a verdict there would change nothing.
_JACCARD_LLM_BAND = 0.25
# New pairs in that band sent to the model per daily pass.  A pair once
# answered is remembered (``judgements`` in session_links.json), so the
# pairs over the cap are asked on a later day.
_MAX_LLM_CHECKS_PER_PASS = 10

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
        # Intent text -> its embedding, so each intent is embedded once.
        self._embeddings: dict[str, list[float]] = {}
        # Intents the model gave no vector for in this pass: their pairs
        # use Jaccard instead of retrying the model for every pair.
        self._not_embedded: set[str] = set()
        # Pair key -> the model's answer (True, False, or None for an
        # answer that was neither), persisted with the links.
        self._judgements: dict[str, bool | None] = {}
        self._consulted: set[str] = set()
        self._llm_checks_left = _MAX_LLM_CHECKS_PER_PASS
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

        self._begin_pass(
            [t["intent"] for t in all_tasks]
            + [link.intent for link in self._links if link.status != "completed"]
        )

        # A task already in a link was placed on an earlier pass: it is
        # neither matched nor grouped again.
        linked = {
            (s["date"], s["task_index"])
            for link in self._links
            for s in link.sessions
        }
        used_indices: set[int] = {
            i for i, t in enumerate(all_tasks)
            if (t["date"], t["task_index"]) in linked
        }
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

                # The same matched procedure links them outright.
                proc_match = (
                    task_a["matched_procedure"] is not None
                    and task_a["matched_procedure"] == task_b["matched_procedure"]
                )
                if proc_match or self._intents_match(task_a["intent"], task_b["intent"]):
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
        # Keep the judgements this pass consulted or made; the others are
        # of pairs no longer compared (linked, or out of the lookback).
        self._judgements = {
            k: v for k, v in self._judgements.items() if k in self._consulted
        }
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

    def _begin_pass(self, intents: list[str]) -> None:
        """Reset the per-pass model budget and embed, in one batched call,
        the pass's intents that have no embedding yet."""
        self._llm_checks_left = _MAX_LLM_CHECKS_PER_PASS
        self._consulted = set()
        self._not_embedded = set()
        texts = list(dict.fromkeys(t for t in intents if t))
        # Only the pass's intents are kept, so the cache does not grow.
        keep = set(texts)
        self._embeddings = {t: e for t, e in self._embeddings.items() if t in keep}
        self._embed(texts)

    def _embed(self, texts) -> None:
        """Embed the *texts* not embedded yet, in one call.  A text the
        model gives no vector for (or every text, when the call fails) is
        not tried again until the next pass: its pairs use Jaccard."""
        missing = [
            t for t in dict.fromkeys(texts)
            if t not in self._embeddings and t not in self._not_embedded
        ]
        if self._vector_kb is None or not missing:
            return
        try:
            embs = self._vector_kb.compute_embeddings(
                missing, prefix=self._vector_kb.profile.cluster_prefix,
            )
        except Exception:
            logger.debug("Intent embedding failed, linking by Jaccard", exc_info=True)
            embs = []
        for text, emb in zip(missing, embs):
            norm = math.sqrt(sum(x * x for x in emb)) if emb else 0.0
            if norm > 0.0:
                # Stored unit-length: a cosine is then a dot product.
                self._embeddings[text] = [x / norm for x in emb]
        self._not_embedded.update(t for t in missing if t not in self._embeddings)

    def _normalize_intent(self, intent: str) -> str:
        """Normalize an intent string: lowercase, strip punctuation, remove stop words."""
        text = intent.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return " ".join(tokens)

    def _intent_similarity(self, a: str, b: str) -> tuple[float, str]:
        """``(score, metric)``: the cosine of the intents' embeddings when
        the vector KB can embed them, else token Jaccard."""
        self._embed((a, b))
        ea, eb = self._embeddings.get(a), self._embeddings.get(b)
        if ea is not None and eb is not None and len(ea) == len(eb):
            return sum(map(operator.mul, ea, eb)), "cosine"
        return self._jaccard_similarity(a, b), "jaccard"

    def _intents_match(self, a: str, b: str) -> bool:
        """Two intents are the same recurring task: the score reaches its
        metric's threshold, or it falls in the band just below and the LLM
        says YES (remembered from an earlier pass, or asked while the
        pass's budget lasts)."""
        score, metric = self._intent_similarity(a, b)
        if metric == "cosine":
            thresholds = self._vector_kb.thresholds
            threshold, band = thresholds.linker_cos, thresholds.linker_band
        else:
            threshold, band = _JACCARD_THRESHOLD, _JACCARD_LLM_BAND
        if score >= threshold:
            return True
        if self._llm_reasoner is None or score < threshold - band:
            return False
        key = _pair_key(a, b)
        self._consulted.add(key)
        if key in self._judgements:
            return self._judgements[key] is True
        if self._llm_checks_left <= 0:
            return False
        self._llm_checks_left -= 1
        answered, verdict = self._semantic_check(a, b)
        if answered:
            self._judgements[key] = verdict
        return verdict is True

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

    def _semantic_check(
        self, intent_a: str, intent_b: str,
    ) -> tuple[bool, bool | None]:
        """Use LLM to determine if two intents describe the same workflow.

        Returns ``(answered, verdict)``: *verdict* is True for YES, False
        for NO, None for an answer that is neither; *answered* is False
        when the model gave no answer (failure, abstention, over budget),
        which is not remembered.
        """
        if self._llm_reasoner is None:
            return False, None

        # Intents are model summaries of the screen: redacted like every
        # other screen-derived text before it reaches a prompt.
        prompt = (
            "Are these two task descriptions the same recurring workflow? "
            f"Task A: '{redact(intent_a)}' Task B: '{redact(intent_b)}'. "
            "Answer YES or NO."
        )

        try:
            result = self._llm_reasoner.reason_yesno(
                prompt, caller="session_linker.semantic_check",
            )
            if result.success and not result.abstained:
                return True, result.value  # True, False, or None (ambiguous)
        except Exception:
            logger.debug("LLM semantic check failed", exc_info=True)

        return False, None

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

            # Intent similarity (with the LLM just below the threshold),
            # the same test as the new-group path so links don't fragment.
            if self._intents_match(link.intent, task.get("intent", "")):
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

        judgements = data.get("judgements", {})
        if isinstance(judgements, dict):
            self._judgements = {
                str(k): v for k, v in judgements.items() if v in (True, False, None)
            }

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
            "judgements": self._judgements,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / "observations" / "session_links.json"
        self._kb.atomic_write_json(path, data)


def _pair_key(a: str, b: str) -> str:
    """Order-insensitive key of an intent pair (a hash: the file does not
    hold a second copy of the intents)."""
    first, second = sorted((a, b))
    return hashlib.sha256(f"{first}\x00{second}".encode("utf-8")).hexdigest()

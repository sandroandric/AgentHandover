"""Daily digest generator for AgentHandover.

Produces a structured summary of the user's day — tasks completed,
procedures observed, trust suggestions, and notable patterns.

Digests are stored at ``{kb_root}/observations/digests/{date}.json``.

Summary text is generated programmatically (no LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging

from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class DigestHighlight:
    """A notable event from the day."""

    type: str  # "new_procedure", "trust_suggestion", "stale_alert", "pattern_detected", "milestone"
    title: str
    detail: str
    priority: int  # 1 (high) to 3 (low)


@dataclass
class DigestSection:
    """A section of the daily digest."""

    title: str
    items: list[dict]


@dataclass
class DailyDigest:
    """The complete daily digest."""

    date: str
    generated_at: str
    summary: str
    active_hours: float
    tasks_completed: int
    procedures_observed: int
    highlights: list[DigestHighlight]
    sections: list[DigestSection]
    stats: dict


class DigestGenerator:
    """Generates daily digests from knowledge base data."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, date: str | None = None) -> DailyDigest:
        """Generate a digest for *date* (YYYY-MM-DD).

        If *date* is ``None``, uses today's date.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        summary_data = self._kb.get_daily_summary(date)
        tasks = self._extract_tasks(summary_data)
        active_hours = self._compute_active_hours(tasks)
        tasks_completed = len(tasks)
        procedures_observed = self._count_procedures(summary_data)
        highlights = self._extract_highlights(date, summary_data)
        sections = self._build_sections(tasks, summary_data)
        stats = self._compute_stats(summary_data, tasks, active_hours)

        summary_text = self._build_summary_text(
            active_hours, tasks_completed, procedures_observed, date
        )

        digest = DailyDigest(
            date=date,
            generated_at=datetime.now(timezone.utc).isoformat(),
            summary=summary_text,
            active_hours=active_hours,
            tasks_completed=tasks_completed,
            procedures_observed=procedures_observed,
            highlights=highlights,
            sections=sections,
            stats=stats,
        )
        return digest

    def generate_summary_text(self, digest: DailyDigest) -> str:
        """Generate a human-readable summary from a digest."""
        return self._build_summary_text(
            digest.active_hours,
            digest.tasks_completed,
            digest.procedures_observed,
            digest.date,
        )

    def save_digest(self, digest: DailyDigest) -> None:
        """Persist a digest to the knowledge base."""
        path = (
            self._kb.root / "observations" / "digests" / f"{digest.date}.json"
        )
        data = {
            "date": digest.date,
            "generated_at": digest.generated_at,
            "summary": digest.summary,
            "active_hours": digest.active_hours,
            "tasks_completed": digest.tasks_completed,
            "procedures_observed": digest.procedures_observed,
            "highlights": [
                {
                    "type": h.type,
                    "title": h.title,
                    "detail": h.detail,
                    "priority": h.priority,
                }
                for h in digest.highlights
            ],
            "sections": [
                {"title": s.title, "items": s.items}
                for s in digest.sections
            ],
            "stats": digest.stats,
        }
        self._kb.atomic_write_json(path, data)
        logger.info("Digest saved for %s", digest.date)

    def get_digest(self, date: str) -> DailyDigest | None:
        """Load a previously saved digest by date string (YYYY-MM-DD)."""
        path = (
            self._kb.root / "observations" / "digests" / f"{date}.json"
        )
        if not path.is_file():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        highlights = [
            DigestHighlight(
                type=h["type"],
                title=h["title"],
                detail=h["detail"],
                priority=h["priority"],
            )
            for h in data.get("highlights", [])
        ]
        sections = [
            DigestSection(title=s["title"], items=s["items"])
            for s in data.get("sections", [])
        ]
        return DailyDigest(
            date=data["date"],
            generated_at=data["generated_at"],
            summary=data["summary"],
            active_hours=data["active_hours"],
            tasks_completed=data["tasks_completed"],
            procedures_observed=data["procedures_observed"],
            highlights=highlights,
            sections=sections,
            stats=data.get("stats", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_tasks(self, summary_data: dict | None) -> list[dict]:
        """Extract the tasks list from a daily summary."""
        if summary_data is None:
            return []
        return summary_data.get("tasks", [])

    def _compute_active_hours(self, tasks: list[dict]) -> float:
        """Sum up task durations into total active hours."""
        total_minutes = sum(
            t.get("duration_minutes", 0) for t in tasks
        )
        return round(total_minutes / 60.0, 2)

    def _count_procedures(self, summary_data: dict | None) -> int:
        """Count distinct procedures referenced in the daily summary."""
        if summary_data is None:
            return 0
        tasks = summary_data.get("tasks", [])
        procedures = set()
        for task in tasks:
            proc = task.get("matched_procedure")
            if proc:
                procedures.add(proc)
        # Also count from the summary-level field if present
        summary_procs = summary_data.get("procedures_observed", [])
        if isinstance(summary_procs, list):
            for p in summary_procs:
                if isinstance(p, str):
                    procedures.add(p)
                elif isinstance(p, dict) and "slug" in p:
                    procedures.add(p["slug"])
        return len(procedures)

    def _extract_highlights(
        self, date: str, summary_data: dict | None
    ) -> list[DigestHighlight]:
        """Extract highlights from the day's data."""
        highlights: list[DigestHighlight] = []

        if summary_data is None:
            return highlights

        # Check for new procedures
        new_procs = summary_data.get("new_procedures", [])
        for proc in new_procs:
            name = proc if isinstance(proc, str) else proc.get("slug", str(proc))
            highlights.append(
                DigestHighlight(
                    type="new_procedure",
                    title=f"New procedure: {name}",
                    detail=f"A new procedure '{name}' was observed on {date}.",
                    priority=1,
                )
            )

        # Check for trust suggestions
        trust_path = self._kb.root / "observations" / "trust_suggestions.json"
        if trust_path.is_file():
            try:
                with open(trust_path) as f:
                    trust_data = json.load(f)
                for s in trust_data.get("suggestions", []):
                    if not s.get("dismissed") and not s.get("accepted"):
                        highlights.append(
                            DigestHighlight(
                                type="trust_suggestion",
                                title=f"Trust promotion available: {s['procedure_slug']}",
                                detail=s.get("reason", ""),
                                priority=2,
                            )
                        )
            except (json.JSONDecodeError, OSError):
                pass

        # Check for patterns
        patterns = summary_data.get("patterns_detected", [])
        for pattern in patterns:
            name = pattern if isinstance(pattern, str) else pattern.get("name", str(pattern))
            highlights.append(
                DigestHighlight(
                    type="pattern_detected",
                    title=f"Pattern detected: {name}",
                    detail=f"A recurring pattern '{name}' was detected.",
                    priority=2,
                )
            )

        # Milestones
        tasks = summary_data.get("tasks", [])
        if len(tasks) >= 10:
            highlights.append(
                DigestHighlight(
                    type="milestone",
                    title="Productive day!",
                    detail=f"You completed {len(tasks)} tasks today.",
                    priority=3,
                )
            )

        return highlights

    def _build_sections(
        self, tasks: list[dict], summary_data: dict | None
    ) -> list[DigestSection]:
        """Build sections of the digest."""
        sections: list[DigestSection] = []

        # Tasks section
        if tasks:
            task_items = []
            for t in tasks:
                task_items.append({
                    "intent": t.get("intent", t.get("description", "Unknown")),
                    "duration_minutes": t.get("duration_minutes", 0),
                    "apps": t.get("apps", []),
                    "matched_procedure": t.get("matched_procedure"),
                })
            sections.append(DigestSection(title="Tasks", items=task_items))

        # Applications section
        if tasks:
            app_counts: dict[str, int] = {}
            for t in tasks:
                for app in t.get("apps", []):
                    app_counts[app] = app_counts.get(app, 0) + 1
            if app_counts:
                app_items = [
                    {"app": app, "task_count": count}
                    for app, count in sorted(
                        app_counts.items(), key=lambda x: x[1], reverse=True
                    )
                ]
                sections.append(
                    DigestSection(title="Applications Used", items=app_items)
                )

        return sections

    def _compute_stats(
        self,
        summary_data: dict | None,
        tasks: list[dict],
        active_hours: float,
    ) -> dict:
        """Compute aggregate statistics."""
        apps_used: set[str] = set()
        for t in tasks:
            for app in t.get("apps", []):
                apps_used.add(app)

        return {
            "total_tasks": len(tasks),
            "active_hours": active_hours,
            "apps_used": len(apps_used),
            "app_list": sorted(apps_used),
        }

    def _build_summary_text(
        self,
        active_hours: float,
        tasks_completed: int,
        procedures_observed: int,
        date: str,
    ) -> str:
        """Build a programmatic summary string (no LLM)."""
        parts = []
        parts.append(
            f"You worked {active_hours:.1f} hours across {tasks_completed} tasks"
        )
        if procedures_observed > 0:
            parts.append(
                f"{procedures_observed} {'procedure' if procedures_observed == 1 else 'procedures'} observed"
            )
        return ". ".join(parts) + "."

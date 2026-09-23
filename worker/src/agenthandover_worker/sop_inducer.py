"""SOP template from a single focus recording (the v1 focus fallback).

When the v2 annotation pipeline is unavailable, a completed focus
recording goes episode builder → translator → scorer →
:func:`induce_from_focus_session`, which turns the recording's accepted
semantic steps into one SOP template.

The v1 PrefixSpan miner that used to live here (it mined passive
episodes every poll) is gone: passive recurrence belongs to
``workflow_mining``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter

from agenthandover_worker.confidence import compute_v2_confidence

logger = logging.getLogger(__name__)


def induce_from_focus_session(episodes: list[list[dict]], title: str) -> list[dict]:
    """Produce SOP templates from a single focus recording session.

    Args:
        episodes: Episodes (lists of semantic step dicts from
            ``SemanticStep.to_sop_step()``) built from the session's events.
        title: User-provided title for the workflow.

    Returns:
        A list with one SOP template (``source: "focus_recording"``), or an
        empty list when the session has no steps.
    """
    all_steps = [step for ep in episodes for step in ep]
    if not all_steps:
        return []

    apps: set[str] = set()
    for step in all_steps:
        for key in ("parameters", "pre_state"):
            fields = step.get(key, {})
            if isinstance(fields, dict):
                app = fields.get("app_id") or fields.get("app")
                if app:
                    apps.add(app)

    template = {
        "slug": slug_from_title(title),
        "title": title,
        "steps": [s.copy() for s in all_steps],
        "variables": [],  # single demo — no variable abstraction possible
        "step_confidence_mean": round(_mean_step_confidence(all_steps), 4),
        "episode_count": 1,
        "abs_support": 1,
        "apps_involved": sorted(apps),
        "preconditions": detect_preconditions([all_steps]),
        "postconditions": detect_postconditions([all_steps]),
        "exceptions_seen": [],
        "source": "focus_recording",
        "focus_title": title,
    }
    # The v1 path has no scene annotations, so annotation quality takes the
    # scorer's conservative default.
    template["confidence_avg"] = compute_v2_confidence(template, is_focus=True).total

    logger.info(
        "Focus session SOP: '%s' with %d steps (confidence=%.2f)",
        title,
        len(all_steps),
        template["confidence_avg"],
    )
    return [template]


def slug_from_title(title: str) -> str:
    """URL-safe slug from a user-provided title, with a short hash suffix."""
    slug = unicodedata.normalize("NFKD", title)
    slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug[:60]
    hash_suffix = hashlib.sha256(title.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{hash_suffix}"


def _mean_step_confidence(steps: list[dict]) -> float:
    values = [
        s.get("confidence", 0.0) for s in steps
        if isinstance(s.get("confidence", 0.0), (int, float))
    ]
    return sum(values) / len(values) if values else 0.0


def detect_preconditions(instances: list[list[dict]]) -> list[str]:
    """What apps/URLs must be open before the SOP starts.

    Examines the first step of each instance; an app or URL present in at
    least 80% of instances is a precondition.
    """
    preconditions: list[str] = []
    app_counts: Counter[str] = Counter()
    url_counts: Counter[str] = Counter()

    for instance in instances:
        if not instance:
            continue
        pre_state = instance[0].get("pre_state", {})
        if isinstance(pre_state, dict):
            app = pre_state.get("app_id") or pre_state.get("app")
            if app:
                app_counts[app] += 1
            url = pre_state.get("url")
            if url:
                url_counts[url] += 1

    threshold = max(1, len(instances) * 0.8)
    for app, count in app_counts.most_common():
        if count >= threshold:
            preconditions.append(f"app_open:{app}")
    for url, count in url_counts.most_common():
        if count >= threshold:
            preconditions.append(f"url_open:{url}")
    return preconditions


def detect_postconditions(instances: list[list[dict]]) -> list[str]:
    """The final state after the SOP completes.

    Examines the last step of each instance; an action or target present in
    at least half of the instances is a postcondition.
    """
    postconditions: list[str] = []
    final_intents: Counter[str] = Counter()
    final_targets: Counter[str] = Counter()

    for instance in instances:
        if not instance:
            continue
        last_step = instance[-1]
        intent = last_step.get("step", "")
        target = last_step.get("target", "")
        if intent:
            final_intents[intent] += 1
        if target:
            final_targets[target] += 1

    threshold = max(1, len(instances) * 0.5)
    for intent, count in final_intents.most_common():
        if count >= threshold:
            postconditions.append(f"final_action:{intent}")
    for target, count in final_targets.most_common(3):
        if count >= threshold:
            postconditions.append(f"final_target:{target}")
    return postconditions

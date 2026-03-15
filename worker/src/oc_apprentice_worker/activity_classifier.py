"""8-class activity taxonomy classifier for OpenMimic.

Classifies annotated screen events into one of 8 activity types using a
three-stage pipeline: keyword/URL heuristics, profile-based prior blending,
and optional policy overrides.  No VLM call — everything is derived from
existing annotation data produced by the scene annotator.

Activity types:  WORK, RESEARCH, COMMUNICATION, SETUP, PERSONAL_ADMIN,
                 ENTERTAINMENT, DEAD_TIME, CONTEXT_SWITCH.

Each classification also carries a *learnability* tag that tells the SOP
pipeline whether to record, ignore, or prioritise the activity.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oc_apprentice_worker.user_policy import UserPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActivityType(str, Enum):
    """The 8-class activity taxonomy."""

    WORK = "work"
    RESEARCH = "research"
    COMMUNICATION = "communication"
    SETUP = "setup"
    PERSONAL_ADMIN = "personal_admin"
    ENTERTAINMENT = "entertainment"
    DEAD_TIME = "dead_time"
    CONTEXT_SWITCH = "context_switch"


class Learnability(str, Enum):
    """How the SOP pipeline should treat this activity."""

    IGNORE = "ignore"
    CONTEXT_ONLY = "context_only"
    CANDIDATE_WORKFLOW = "candidate_workflow"
    EXECUTION_RELEVANT = "execution_relevant"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Output of a single activity classification."""

    activity_type: ActivityType
    learnability: Learnability
    confidence: float
    source: str    # "heuristic", "prior", "policy"
    reasoning: str


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_KEYWORD_TABLE: dict[ActivityType, list[str]] = {
    ActivityType.WORK: [
        "coding", "debugging", "reviewing", "deploying", "testing code",
        "writing code", "designing", "editing file", "building project",
        "committing", "pushing", "pull request", "merging", "refactoring",
        "fixing bug", "implementing", "developing", "programming",
        "compiling",
    ],
    ActivityType.RESEARCH: [
        "researching", "reading paper", "reading documentation",
        "reading docs", "searching for", "exploring", "learning about",
        "studying", "looking up", "investigating",
    ],
    ActivityType.COMMUNICATION: [
        "messaging", "emailing", "chatting", "calling", "meeting",
        "standup", "sending message", "replying", "slack", "discussing",
        "video call",
    ],
    ActivityType.SETUP: [
        "installing", "configuring", "setting up", "updating software",
        "downloading", "upgrading", "initializing", "bootstrapping",
    ],
    ActivityType.PERSONAL_ADMIN: [
        "paying bill", "booking", "scheduling personal", "shopping",
        "ordering", "banking", "filing tax", "expense report", "insurance",
    ],
    ActivityType.ENTERTAINMENT: [
        "watching", "streaming", "gaming", "browsing social", "scrolling",
        "playing music", "watching video", "reading news for fun", "memes",
    ],
    ActivityType.DEAD_TIME: [
        "idle", "doing nothing", "staring", "waiting", "loading screen",
    ],
    ActivityType.CONTEXT_SWITCH: [
        "switching between", "transitioning", "alt-tabbing", "moving to",
    ],
}

_URL_HINTS: list[tuple[str, ActivityType]] = [
    # Entertainment
    ("*youtube.com*", ActivityType.ENTERTAINMENT),
    ("*netflix.com*", ActivityType.ENTERTAINMENT),
    ("*twitch.tv*", ActivityType.ENTERTAINMENT),
    ("*reddit.com*", ActivityType.ENTERTAINMENT),
    ("*instagram.com*", ActivityType.ENTERTAINMENT),
    ("*tiktok.com*", ActivityType.ENTERTAINMENT),
    ("*twitter.com*", ActivityType.ENTERTAINMENT),
    ("*x.com*", ActivityType.ENTERTAINMENT),
    # Communication
    ("*facebook.com*", ActivityType.COMMUNICATION),
    ("*mail.google.com*", ActivityType.COMMUNICATION),
    ("*outlook.*", ActivityType.COMMUNICATION),
    ("*slack.com*", ActivityType.COMMUNICATION),
    ("*teams.microsoft*", ActivityType.COMMUNICATION),
    ("*discord.com*", ActivityType.COMMUNICATION),
    # Work
    ("*github.com*", ActivityType.WORK),
    ("*gitlab.com*", ActivityType.WORK),
    ("*bitbucket.org*", ActivityType.WORK),
    ("*jira.*", ActivityType.WORK),
    ("*linear.app*", ActivityType.WORK),
    ("*notion.so*", ActivityType.WORK),
    # Research
    ("*stackoverflow.com*", ActivityType.RESEARCH),
    ("*docs.*", ActivityType.RESEARCH),
    ("*developer.*", ActivityType.RESEARCH),
    ("*arxiv.org*", ActivityType.RESEARCH),
    ("*scholar.google*", ActivityType.RESEARCH),
    # Personal admin
    ("*amazon.com*", ActivityType.PERSONAL_ADMIN),
    ("*ebay.com*", ActivityType.PERSONAL_ADMIN),
]

_APP_HINTS: dict[str, ActivityType] = {
    # Communication
    "Zoom": ActivityType.COMMUNICATION,
    "FaceTime": ActivityType.COMMUNICATION,
    "Teams": ActivityType.COMMUNICATION,
    "Slack": ActivityType.COMMUNICATION,
    # Work — IDEs and editors
    "Xcode": ActivityType.WORK,
    "VS Code": ActivityType.WORK,
    "IntelliJ": ActivityType.WORK,
    "PyCharm": ActivityType.WORK,
    "WebStorm": ActivityType.WORK,
    "Terminal": ActivityType.WORK,
    "iTerm": ActivityType.WORK,
    # Work — design
    "Figma": ActivityType.WORK,
    "Sketch": ActivityType.WORK,
}

_LEARNABILITY_MAP: dict[ActivityType, Learnability] = {
    ActivityType.WORK: Learnability.EXECUTION_RELEVANT,
    ActivityType.RESEARCH: Learnability.EXECUTION_RELEVANT,
    ActivityType.SETUP: Learnability.EXECUTION_RELEVANT,
    ActivityType.COMMUNICATION: Learnability.CANDIDATE_WORKFLOW,
    ActivityType.PERSONAL_ADMIN: Learnability.CONTEXT_ONLY,
    ActivityType.CONTEXT_SWITCH: Learnability.CONTEXT_ONLY,
    ActivityType.ENTERTAINMENT: Learnability.IGNORE,
    ActivityType.DEAD_TIME: Learnability.IGNORE,
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ActivityClassifier:
    """Classify annotated screen events into the 8-class activity taxonomy.

    Three-stage pipeline:
        1. Keyword + URL heuristics (fast, deterministic)
        2. Prior blending from user profile (adjusts ambiguous cases)
        3. Policy override (user-defined rules, highest authority)

    Usage::

        classifier = ActivityClassifier(profile=user_profile)
        result = classifier.classify(annotation)
    """

    def __init__(
        self,
        profile: dict | None = None,
        policy: UserPolicy | None = None,
    ) -> None:
        self._profile = profile
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        annotation: dict,
        event_context: dict | None = None,
    ) -> ClassificationResult:
        """Classify a single annotated event.

        Parameters
        ----------
        annotation:
            The scene annotation dict produced by ``SceneAnnotator``.
            Expected keys: ``task_context.what_doing``, ``app``,
            ``location`` (or ``visual_context.location``).
        event_context:
            Optional dict with extra event metadata, e.g.
            ``{"timestamp": "2026-03-14T09:30:00Z"}``.

        Returns
        -------
        ClassificationResult
            The activity type, learnability, confidence, source, and
            reasoning for the classification.
        """
        # Extract fields from annotation
        what_doing = (
            annotation.get("task_context", {}).get("what_doing", "") or ""
        )
        location = (
            annotation.get("visual_context", {}).get("location", "")
            or annotation.get("location", "")
            or ""
        )
        app = annotation.get("app", "") or ""
        is_workflow = annotation.get("task_context", {}).get(
            "is_workflow", False,
        )

        # Stage 1 — keyword + URL heuristics
        result = self._stage_heuristic(what_doing, location, app, is_workflow)

        # Stage 2 — prior blending (only when profile exists and
        # confidence is below 0.9)
        if self._profile is not None and result.confidence < 0.9:
            result = self._stage_prior(result, app, event_context)

        # Infer learnability from activity type (before policy override)
        result.learnability = _LEARNABILITY_MAP.get(
            result.activity_type, Learnability.CONTEXT_ONLY,
        )

        # Stage 3 — policy override
        if self._policy is not None:
            result = self._stage_policy(result, app, location)

        return result

    def classify_from_is_workflow(
        self,
        is_workflow: bool,
    ) -> ClassificationResult:
        """Quick classification from the legacy ``is_workflow`` flag.

        Useful when no full annotation is available — e.g. during
        migration of older events.

        Parameters
        ----------
        is_workflow:
            The boolean flag from ``task_context.is_workflow``.

        Returns
        -------
        ClassificationResult
        """
        if is_workflow:
            return ClassificationResult(
                activity_type=ActivityType.WORK,
                learnability=Learnability.EXECUTION_RELEVANT,
                confidence=0.5,
                source="heuristic",
                reasoning="Derived from legacy is_workflow=True",
            )
        return ClassificationResult(
            activity_type=ActivityType.ENTERTAINMENT,
            learnability=Learnability.IGNORE,
            confidence=0.5,
            source="heuristic",
            reasoning="Derived from legacy is_workflow=False",
        )

    # ------------------------------------------------------------------
    # Stage 1 — keyword + URL heuristics
    # ------------------------------------------------------------------

    def _stage_heuristic(
        self,
        what_doing: str,
        location: str,
        app: str,
        is_workflow: bool,
    ) -> ClassificationResult:
        """Deterministic classification from keywords, URLs, and app name."""
        what_lower = what_doing.lower()
        location_lower = location.lower()

        # 1. Try keyword match on what_doing
        for activity_type, keywords in _KEYWORD_TABLE.items():
            for kw in keywords:
                if kw in what_lower:
                    return ClassificationResult(
                        activity_type=activity_type,
                        learnability=Learnability.CONTEXT_ONLY,  # placeholder
                        confidence=0.8,
                        source="heuristic",
                        reasoning=f"Keyword match: '{kw}' in what_doing",
                    )

        # 2. Try URL match on location
        for pattern, activity_type in _URL_HINTS:
            if fnmatch.fnmatch(location_lower, pattern.lower()):
                return ClassificationResult(
                    activity_type=activity_type,
                    learnability=Learnability.CONTEXT_ONLY,
                    confidence=0.8,
                    source="heuristic",
                    reasoning=f"URL match: '{pattern}' on location",
                )

        # 3. Try app name match
        for app_hint, activity_type in _APP_HINTS.items():
            if app_hint.lower() in app.lower():
                return ClassificationResult(
                    activity_type=activity_type,
                    learnability=Learnability.CONTEXT_ONLY,
                    confidence=0.8,
                    source="heuristic",
                    reasoning=f"App match: '{app_hint}' in app name",
                )

        # 4. Fall back to is_workflow
        if is_workflow:
            return ClassificationResult(
                activity_type=ActivityType.WORK,
                learnability=Learnability.CONTEXT_ONLY,
                confidence=0.5,
                source="heuristic",
                reasoning="Fallback: is_workflow=True",
            )
        return ClassificationResult(
            activity_type=ActivityType.ENTERTAINMENT,
            learnability=Learnability.CONTEXT_ONLY,
            confidence=0.5,
            source="heuristic",
            reasoning="Fallback: is_workflow=False, no heuristic match",
        )

    # ------------------------------------------------------------------
    # Stage 2 — prior blending
    # ------------------------------------------------------------------

    def _stage_prior(
        self,
        current: ClassificationResult,
        app: str,
        event_context: dict | None,
    ) -> ClassificationResult:
        """Adjust classification using the user profile."""
        assert self._profile is not None
        changed = False

        # 2a. Check if app is a primary work app
        primary_apps = self._profile.get("tools", {}).get("primary_apps", [])
        primary_app_names = {
            entry.get("app", "").lower() for entry in primary_apps
        }

        app_lower = app.lower()
        is_primary = any(
            name and name in app_lower
            for name in primary_app_names
        )

        if is_primary and current.activity_type in (
            ActivityType.ENTERTAINMENT,
            ActivityType.PERSONAL_ADMIN,
        ) and current.confidence <= 0.8:
            current = ClassificationResult(
                activity_type=ActivityType.WORK,
                learnability=current.learnability,
                confidence=0.7,
                source="prior",
                reasoning=(
                    f"Profile override: '{app}' is a primary work app, "
                    f"was {current.activity_type.value}"
                ),
            )
            changed = True

        # 2b. Check working hours
        if event_context and not changed:
            timestamp_str = event_context.get("timestamp", "")
            working_hours = self._profile.get("working_hours", {})
            typical_start = working_hours.get("typical_start", "")
            typical_end = working_hours.get("typical_end", "")

            if timestamp_str and typical_start and typical_end:
                try:
                    ts = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    event_time = ts.strftime("%H:%M")

                    if event_time < typical_start or event_time > typical_end:
                        # Outside working hours — ambiguous WORK gets
                        # lower confidence
                        if current.activity_type == ActivityType.WORK:
                            new_confidence = max(
                                0.0, current.confidence - 0.1,
                            )
                            current = ClassificationResult(
                                activity_type=current.activity_type,
                                learnability=current.learnability,
                                confidence=new_confidence,
                                source="prior",
                                reasoning=(
                                    f"Outside working hours "
                                    f"({typical_start}-{typical_end}), "
                                    f"confidence reduced"
                                ),
                            )
                            changed = True
                except (ValueError, TypeError):
                    logger.debug(
                        "Could not parse timestamp for prior blending: %s",
                        timestamp_str,
                    )

        if changed:
            current.source = "prior"

        return current

    # ------------------------------------------------------------------
    # Stage 3 — policy override
    # ------------------------------------------------------------------

    def _stage_policy(
        self,
        current: ClassificationResult,
        app: str,
        location: str,
    ) -> ClassificationResult:
        """Apply user-defined policy rules."""
        assert self._policy is not None

        rule = self._policy.check(app=app, url=location)
        if rule is None:
            return current

        action = (rule.action.value if hasattr(rule.action, "value") else str(rule.action)).upper()
        pattern = rule.pattern

        if action in ("IGNORE", "NEVER_LEARN"):
            current.learnability = Learnability.IGNORE
            current.confidence = 1.0
            current.source = "policy"
            current.reasoning = (
                f"Policy {action}: pattern '{pattern}' matched"
            )

        elif action == "CLASSIFY_AS":
            raw_value = rule.value
            try:
                new_type = ActivityType(raw_value)
            except ValueError:
                logger.warning(
                    "Policy CLASSIFY_AS has invalid value '%s', ignoring",
                    raw_value,
                )
                return current

            current.activity_type = new_type
            current.learnability = _LEARNABILITY_MAP.get(
                new_type, Learnability.CONTEXT_ONLY,
            )
            current.confidence = 1.0
            current.source = "policy"
            current.reasoning = (
                f"Policy CLASSIFY_AS: pattern '{pattern}' → "
                f"{new_type.value}"
            )

        elif action == "ALWAYS_INCLUDE":
            current.learnability = Learnability.EXECUTION_RELEVANT
            current.confidence = 1.0
            current.source = "policy"
            current.reasoning = (
                f"Policy ALWAYS_INCLUDE: pattern '{pattern}' matched"
            )

        return current

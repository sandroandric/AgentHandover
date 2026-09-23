"""8-class activity taxonomy classifier for AgentHandover.

Classifies annotated screen events into one of 8 activity types using a
three-stage pipeline: keyword/URL heuristics, profile-based prior blending,
and optional policy overrides.  No VLM call — everything is derived from
existing annotation data produced by the scene annotator.

Activity types:  WORK, RESEARCH, COMMUNICATION, SETUP, PERSONAL_ADMIN,
                 ENTERTAINMENT, DEAD_TIME, CONTEXT_SWITCH, plus UNKNOWN
                 when no heuristic matches (consumers then fall back to
                 the annotator's is_workflow).

Each classification also carries a *learnability* tag that tells the SOP
pipeline whether to record, ignore, or prioritise the activity.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner
    from agenthandover_worker.user_policy import UserPolicy

logger = logging.getLogger(__name__)

# How many distinct classifier prompts keep their LLM answer.
_LLM_ANSWERS_MAX = 512


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
    # No heuristic matched: the classifier has no opinion, and consumers
    # fall back to the annotator's own is_workflow verdict.
    UNKNOWN = "unknown"


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

# Hosts, matched against the location's parsed host name.  Each host also
# covers its subdomains (``foo.com`` and ``*.foo.com``), never a longer name
# that merely contains it: ``x.com`` must not match ``dropbox.com``.  The
# first match wins, so a specific subdomain precedes its parent's entry.
_HOST_HINTS: list[tuple[str, ActivityType]] = [
    # Entertainment
    ("youtube.com", ActivityType.ENTERTAINMENT),
    ("netflix.com", ActivityType.ENTERTAINMENT),
    ("twitch.tv", ActivityType.ENTERTAINMENT),
    ("reddit.com", ActivityType.ENTERTAINMENT),
    ("instagram.com", ActivityType.ENTERTAINMENT),
    ("tiktok.com", ActivityType.ENTERTAINMENT),
    ("twitter.com", ActivityType.ENTERTAINMENT),
    ("x.com", ActivityType.ENTERTAINMENT),
    # Communication
    ("facebook.com", ActivityType.COMMUNICATION),
    ("mail.google.com", ActivityType.COMMUNICATION),
    ("outlook.com", ActivityType.COMMUNICATION),
    ("outlook.live.com", ActivityType.COMMUNICATION),
    ("outlook.office.com", ActivityType.COMMUNICATION),
    ("outlook.office365.com", ActivityType.COMMUNICATION),
    ("slack.com", ActivityType.COMMUNICATION),
    ("teams.microsoft.com", ActivityType.COMMUNICATION),
    ("discord.com", ActivityType.COMMUNICATION),
    # Research (developer documentation hosts precede their parent sites)
    ("docs.github.com", ActivityType.RESEARCH),
    ("stackoverflow.com", ActivityType.RESEARCH),
    ("docs.python.org", ActivityType.RESEARCH),
    ("developer.mozilla.org", ActivityType.RESEARCH),
    ("developer.apple.com", ActivityType.RESEARCH),
    ("developer.android.com", ActivityType.RESEARCH),
    ("learn.microsoft.com", ActivityType.RESEARCH),
    ("readthedocs.io", ActivityType.RESEARCH),
    ("docs.rs", ActivityType.RESEARCH),
    ("arxiv.org", ActivityType.RESEARCH),
    ("scholar.google.com", ActivityType.RESEARCH),
    # Work
    ("github.com", ActivityType.WORK),
    ("gitlab.com", ActivityType.WORK),
    ("bitbucket.org", ActivityType.WORK),
    ("atlassian.net", ActivityType.WORK),
    ("linear.app", ActivityType.WORK),
    ("notion.so", ActivityType.WORK),
    # Personal admin
    ("amazon.com", ActivityType.PERSONAL_ADMIN),
    ("ebay.com", ActivityType.PERSONAL_ADMIN),
]

# The host glob pairs, in hint order.
_URL_HINTS: list[tuple[str, ActivityType]] = [
    (pattern, activity_type)
    for host, activity_type in _HOST_HINTS
    for pattern in (host, f"*.{host}")
]


def _location_host(location: str) -> str:
    """The lowercased host name of a URL-like location, or "".

    Accepts full URLs and scheme-less ones (``dropbox.com/home``).  File
    paths, app-internal locations and free text have no host.
    """
    loc = location.strip()
    if not loc:
        return ""
    try:
        host = urlsplit(loc).hostname
        if host is None and not loc.startswith(("/", "~", ".")):
            first = loc.split("/", 1)[0]
            if first and not any(ch.isspace() for ch in first):
                host = urlsplit("//" + loc).hostname
    except ValueError:
        return ""
    if not host or "." not in host:
        return ""
    return host.rstrip(".").lower()


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
    ActivityType.UNKNOWN: Learnability.CONTEXT_ONLY,
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

    # Default working hours assumption for day-1 classification
    _DEFAULT_WORKING_HOURS = {"typical_start": "08:00", "typical_end": "19:00"}

    def __init__(
        self,
        profile: dict | None = None,
        policy: UserPolicy | None = None,
        llm_reasoner: "LLMReasoner | None" = None,
    ) -> None:
        self._profile = profile
        self._policy = policy
        self._llm_reasoner = llm_reasoner
        # The LLM's answer per prompt. The prompt is only what_doing, app
        # and location, which a cached annotation repeats exactly, so a
        # screen seen again costs no second call.
        self._llm_answers: OrderedDict[str, str] = OrderedDict()
        # Session-level app frequency tracker — learns from current session
        # even before the multi-day profile exists
        self._session_app_counts: dict[str, int] = {}
        self._session_total: int = 0

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
        task_context = annotation.get("task_context", {})
        is_workflow = task_context.get("is_workflow", False)
        # The annotator saw the frame and said whether it is a workflow:
        # an app-level prior (this app is usually work) must not overrule
        # that verdict, or YouTube in a work browser becomes work.
        vlm_judged = "is_workflow" in task_context

        # App tracking happens after Stage 1 (below) so we only count
        # apps seen in work-like contexts, not noise apps.

        # Stage 1 — keyword + URL heuristics
        result = self._stage_heuristic(what_doing, location, app, is_workflow)

        # Track app usage AFTER heuristic — only count apps that appear
        # in work-like contexts, not noise apps like YouTube
        if app:
            app_key = app.lower()
            self._session_total += 1
            if result.activity_type in (
                ActivityType.WORK, ActivityType.RESEARCH,
                ActivityType.SETUP, ActivityType.COMMUNICATION,
            ):
                self._session_app_counts[app_key] = (
                    self._session_app_counts.get(app_key, 0) + 1
                )
            # Rolling window: keep only the last 200 observations
            if self._session_total > 200:
                # Decay all counts by half to keep the window fresh
                self._session_app_counts = {
                    k: v // 2 for k, v in self._session_app_counts.items()
                    if v // 2 > 0
                }
                self._session_total = sum(self._session_app_counts.values())

        # Stage 2 — prior blending
        if result.confidence < 0.9:
            if self._profile is not None:
                # Full profile available (after 3+ daily summaries)
                result = self._stage_prior(
                    result, app, event_context, app_override=not vlm_judged,
                )
            elif self._session_total >= 10:
                # No profile yet — use session-learned priors from day 1
                result = self._stage_session_prior(
                    result, app, event_context, app_override=not vlm_judged,
                )

        # Stage 2.5 — LLM refinement for ambiguous classifications
        if (
            self._llm_reasoner is not None
            and result.confidence < 0.6
            and result.source != "policy"
        ):
            llm_type = self._classify_with_llm(annotation, result)
            if llm_type is not None:
                result = ClassificationResult(
                    activity_type=ActivityType(llm_type),
                    learnability=Learnability.CONTEXT_ONLY,  # placeholder
                    confidence=0.75,
                    source="llm",
                    reasoning=f"LLM classified as '{llm_type}' (heuristic was "
                              f"'{result.activity_type.value}' at {result.confidence:.2f})",
                )

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
            activity_type=ActivityType.UNKNOWN,
            learnability=Learnability.CONTEXT_ONLY,
            confidence=0.3,
            source="heuristic",
            reasoning="Derived from legacy is_workflow=False",
        )

    # ------------------------------------------------------------------
    # LLM refinement (between Stage 2 and Stage 3)
    # ------------------------------------------------------------------

    def _classify_with_llm(
        self,
        annotation: dict,
        heuristic_result: ClassificationResult,
    ) -> str | None:
        """Use LLM to classify an ambiguous activity.

        Returns the ActivityType value string, or None on failure/budget/abstention.
        """
        if self._llm_reasoner is None:
            return None

        what_doing = (
            annotation.get("task_context", {}).get("what_doing", "") or ""
        )
        app = annotation.get("app", "") or ""
        location = (
            annotation.get("visual_context", {}).get("location", "")
            or annotation.get("location", "")
            or ""
        )

        prompt = (
            "Classify this activity into exactly one category: "
            "work, research, communication, setup, personal_admin, "
            "entertainment, dead_time, context_switch. "
            f"Activity: {what_doing} App: {app} Location: {location}. "
            "Respond with ONLY the category name."
        )

        if prompt in self._llm_answers:
            self._llm_answers.move_to_end(prompt)
            return self._llm_answers[prompt]

        try:
            result = self._llm_reasoner.reason_text(
                prompt, caller="activity_classifier.classify_with_llm",
                think=False,
            )
            if not result.success or result.abstained or not result.value:
                return None

            raw = str(result.value).strip().lower().replace(" ", "_")
            # Match against ActivityType enum values
            valid_values = {at.value for at in ActivityType}
            if raw in valid_values:
                self._llm_answers[prompt] = raw
                if len(self._llm_answers) > _LLM_ANSWERS_MAX:
                    self._llm_answers.popitem(last=False)
                return raw
        except Exception:
            logger.debug("LLM classification failed", exc_info=True)

        return None

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

        # 2. Try a host match on location.  Only a location without a
        # parseable host (a file path, an app-internal location) falls back
        # to searching the raw string for a hint host.
        host = _location_host(location)
        for pattern, activity_type in _URL_HINTS:
            if host:
                matched = fnmatch.fnmatchcase(host, pattern)
            else:
                # The hint host as a whole name, not a tail of a longer one.
                matched = not pattern.startswith("*.") and re.search(
                    rf"(?<![a-z0-9-]){re.escape(pattern)}(?![a-z0-9-])",
                    location_lower,
                ) is not None
            if matched:
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
            activity_type=ActivityType.UNKNOWN,
            learnability=Learnability.CONTEXT_ONLY,
            confidence=0.3,
            source="heuristic",
            reasoning="Fallback: is_workflow=False, no heuristic match",
        )

    # ------------------------------------------------------------------
    # Stage 2a — session-level prior (day 1, before profile exists)
    # ------------------------------------------------------------------

    def _stage_session_prior(
        self,
        current: ClassificationResult,
        app: str,
        event_context: dict | None,
        *,
        app_override: bool = True,
    ) -> ClassificationResult:
        """Adjust classification using session-learned app frequency.

        After ~10 work-classified observations, apps that account for a
        significant share of work activity are treated as primary work apps.
        Only apps already seen in work/research/setup/communication contexts
        are tracked — noise apps (YouTube, Reddit) are excluded from counts.
        """
        work_total = sum(self._session_app_counts.values())
        if not app or work_total < 10:
            return current

        app_key = app.lower()
        app_count = self._session_app_counts.get(app_key, 0)
        app_ratio = app_count / work_total if work_total > 0 else 0.0

        # If this app accounts for >10% of work-classified observations,
        # it's a known work app — override ambiguous classifications
        is_frequent = app_ratio >= 0.10

        if app_override and is_frequent and current.activity_type in (
            ActivityType.ENTERTAINMENT,
            ActivityType.PERSONAL_ADMIN,
            ActivityType.UNKNOWN,
        ) and current.confidence <= 0.8:
            return ClassificationResult(
                activity_type=ActivityType.WORK,
                learnability=current.learnability,
                confidence=0.7,
                source="prior",
                reasoning=(
                    f"Session prior: '{app}' is {app_ratio:.0%} of observations, "
                    f"likely a work app (was {current.activity_type.value})"
                ),
            )

        # Apply default working hours assumption
        if event_context and current.activity_type == ActivityType.WORK:
            timestamp_str = event_context.get("timestamp", "")
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    event_time = ts.strftime("%H:%M")
                    start = self._DEFAULT_WORKING_HOURS["typical_start"]
                    end = self._DEFAULT_WORKING_HOURS["typical_end"]
                    if event_time < start or event_time > end:
                        return ClassificationResult(
                            activity_type=current.activity_type,
                            learnability=current.learnability,
                            confidence=max(current.confidence - 0.1, 0.0),
                            source="prior",
                            reasoning=(
                                f"Outside default working hours ({start}-{end}), "
                                f"confidence reduced"
                            ),
                        )
                except (ValueError, TypeError):
                    pass

        return current

    # ------------------------------------------------------------------
    # Stage 2b — profile-based prior (after 3+ daily summaries)
    # ------------------------------------------------------------------

    def _stage_prior(
        self,
        current: ClassificationResult,
        app: str,
        event_context: dict | None,
        *,
        app_override: bool = True,
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

        if app_override and is_primary and current.activity_type in (
            ActivityType.ENTERTAINMENT,
            ActivityType.PERSONAL_ADMIN,
            ActivityType.UNKNOWN,
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

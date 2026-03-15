"""Tests for the 8-class activity taxonomy classifier.

Covers all three stages of the classification pipeline:
  1. Keyword + URL + app heuristics
  2. Profile-based prior blending
  3. User-policy overrides

Plus backward compatibility, learnability inference, confidence values,
edge cases, and full activity-type reachability.
"""

from __future__ import annotations

import pytest

from oc_apprentice_worker.activity_classifier import (
    ActivityClassifier,
    ActivityType,
    ClassificationResult,
    Learnability,
    _APP_HINTS,
    _KEYWORD_TABLE,
    _LEARNABILITY_MAP,
    _URL_HINTS,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.user_policy import PolicyAction, PolicyRule, UserPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ann(
    what_doing: str = "",
    is_workflow: bool = False,
    active_app: str = "",
    location: str = "",
) -> dict:
    """Build an annotation dict matching the format produced by SceneAnnotator."""
    return {
        "task_context": {
            "what_doing": what_doing,
            "is_workflow": is_workflow,
        },
        "visual_context": {
            "active_app": active_app,
            "location": location,
        },
        "app": active_app,
    }


@pytest.fixture()
def classifier() -> ActivityClassifier:
    """Bare classifier with no profile and no policy."""
    return ActivityClassifier()


@pytest.fixture()
def kb(tmp_path):
    """Temporary knowledge base for policy tests."""
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


# ===================================================================
# TestKeywordMapping — 16 tests
# ===================================================================

class TestKeywordMapping:
    """Stage 1: verify keyword matches in what_doing."""

    def test_work_keyword_coding(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="I am coding a feature"))
        assert result.activity_type == ActivityType.WORK

    def test_research_keyword_studying(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="studying the new API"))
        assert result.activity_type == ActivityType.RESEARCH

    def test_communication_keyword_emailing(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="emailing the client"))
        assert result.activity_type == ActivityType.COMMUNICATION

    def test_setup_keyword_installing(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="installing homebrew packages"))
        assert result.activity_type == ActivityType.SETUP

    def test_personal_admin_keyword_paying_bill(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="paying bill for electricity"))
        assert result.activity_type == ActivityType.PERSONAL_ADMIN

    def test_entertainment_keyword_watching(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="watching a tutorial video"))
        assert result.activity_type == ActivityType.ENTERTAINMENT

    def test_dead_time_keyword_idle(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="idle at login screen"))
        assert result.activity_type == ActivityType.DEAD_TIME

    def test_context_switch_keyword_switching(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="switching between windows"))
        assert result.activity_type == ActivityType.CONTEXT_SWITCH

    # --- No false positives ---

    def test_reading_a_book_not_research(self, classifier: ActivityClassifier):
        """'reading a book' should not match RESEARCH keywords like
        'reading paper' or 'reading documentation'."""
        result = classifier.classify(_ann(what_doing="reading a book"))
        assert result.activity_type != ActivityType.RESEARCH

    def test_playing_guitar_no_keyword_match(self, classifier: ActivityClassifier):
        """'playing guitar' should not match ENTERTAINMENT keyword 'playing music'
        (substring match fails because 'playing music' != 'playing guitar').
        Falls back to is_workflow=False default (ENTERTAINMENT) — but via
        fallback, not keyword match."""
        result = classifier.classify(_ann(what_doing="playing guitar"))
        # No keyword matched, so source is heuristic fallback at 0.5 confidence
        assert result.confidence == 0.5
        assert result.source == "heuristic"

    def test_running_errands_not_work(self, classifier: ActivityClassifier):
        """'running errands' should not match any WORK keyword."""
        result = classifier.classify(_ann(what_doing="running errands"))
        assert result.activity_type != ActivityType.WORK

    def test_plain_nothing_not_dead_time(self, classifier: ActivityClassifier):
        """'nothing to do here' should not match DEAD_TIME 'doing nothing'
        unless substring match occurs — 'doing nothing' is the keyword."""
        result = classifier.classify(_ann(what_doing="nothing to do here"))
        assert result.activity_type != ActivityType.DEAD_TIME

    def test_scheduling_meeting_not_setup(self, classifier: ActivityClassifier):
        """'scheduling a meeting' should not match SETUP keywords."""
        # SETUP has 'setting up', not 'scheduling'. COMMUNICATION has 'meeting'.
        result = classifier.classify(_ann(what_doing="scheduling a meeting"))
        # Should match COMMUNICATION's 'meeting' keyword, not SETUP.
        assert result.activity_type != ActivityType.SETUP

    def test_ordering_food_not_setup(self, classifier: ActivityClassifier):
        """'ordering food' should match PERSONAL_ADMIN, not SETUP."""
        result = classifier.classify(_ann(what_doing="ordering food online"))
        assert result.activity_type == ActivityType.PERSONAL_ADMIN
        assert result.activity_type != ActivityType.SETUP

    def test_exploring_menu_is_research(self, classifier: ActivityClassifier):
        """'exploring' is a RESEARCH keyword."""
        result = classifier.classify(_ann(what_doing="exploring the settings"))
        assert result.activity_type == ActivityType.RESEARCH

    def test_moving_to_is_context_switch(self, classifier: ActivityClassifier):
        """'moving to' is a CONTEXT_SWITCH keyword."""
        result = classifier.classify(_ann(what_doing="moving to next task"))
        assert result.activity_type == ActivityType.CONTEXT_SWITCH


# ===================================================================
# TestURLHeuristics — 8 tests
# ===================================================================

class TestURLHeuristics:
    """Stage 1: URL pattern matching via _URL_HINTS."""

    def test_youtube_entertainment(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://youtube.com/watch?v=abc"))
        assert result.activity_type == ActivityType.ENTERTAINMENT

    def test_github_work(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://github.com/user/repo"))
        assert result.activity_type == ActivityType.WORK

    def test_stackoverflow_research(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://stackoverflow.com/questions/123"))
        assert result.activity_type == ActivityType.RESEARCH

    def test_slack_communication(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://app.slack.com/client/T01/C01"))
        assert result.activity_type == ActivityType.COMMUNICATION

    def test_gmail_communication(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://mail.google.com/mail/u/0/"))
        assert result.activity_type == ActivityType.COMMUNICATION

    def test_amazon_personal_admin(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://amazon.com/dp/B01234"))
        assert result.activity_type == ActivityType.PERSONAL_ADMIN

    def test_arxiv_research(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://arxiv.org/abs/2301.12345"))
        assert result.activity_type == ActivityType.RESEARCH

    def test_netflix_entertainment(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(location="https://netflix.com/title/80123456"))
        assert result.activity_type == ActivityType.ENTERTAINMENT


# ===================================================================
# TestAppHints — 4 tests
# ===================================================================

class TestAppHints:
    """Stage 1: app-name matching via _APP_HINTS."""

    def test_vscode_work(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(active_app="VS Code"))
        assert result.activity_type == ActivityType.WORK

    def test_zoom_communication(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(active_app="Zoom"))
        assert result.activity_type == ActivityType.COMMUNICATION

    def test_terminal_work(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(active_app="Terminal"))
        assert result.activity_type == ActivityType.WORK

    def test_unknown_app_fallback(self, classifier: ActivityClassifier):
        """An unknown app with no keyword signals falls back to is_workflow=False."""
        result = classifier.classify(_ann(active_app="SomeRandomApp"))
        assert result.activity_type == ActivityType.ENTERTAINMENT
        assert result.confidence == 0.5


# ===================================================================
# TestPriorBlending — 6 tests
# ===================================================================

class TestPriorBlending:
    """Stage 2: profile-based prior adjustments."""

    def _profile_with_apps(self, apps: list[str]) -> dict:
        return {
            "tools": {
                "primary_apps": [{"app": a} for a in apps],
            },
            "working_hours": {
                "typical_start": "09:00",
                "typical_end": "17:00",
            },
        }

    def test_known_work_app_overrides_entertainment(self):
        """Primary work app should override an ambiguous ENTERTAINMENT
        classification back to WORK."""
        profile = self._profile_with_apps(["Safari"])
        c = ActivityClassifier(profile=profile)
        # Safari is not in _APP_HINTS, so with no keyword it falls to
        # is_workflow=False → ENTERTAINMENT at 0.5 confidence, which
        # the prior can override because app is primary and confidence <= 0.8.
        result = c.classify(_ann(active_app="Safari"))
        assert result.activity_type == ActivityType.WORK
        assert result.source == "prior"

    def test_outside_working_hours_reduces_confidence(self):
        """An event outside working hours should get lower confidence for WORK."""
        profile = self._profile_with_apps([])
        c = ActivityClassifier(profile=profile)
        ctx = {"timestamp": "2026-03-14T22:30:00Z"}
        # With is_workflow=True, heuristic produces WORK at 0.5 confidence.
        result = c.classify(_ann(is_workflow=True), event_context=ctx)
        assert result.activity_type == ActivityType.WORK
        assert result.confidence < 0.5  # reduced by 0.1

    def test_empty_profile_degrades_gracefully(self):
        """Empty profile dict should not cause errors."""
        c = ActivityClassifier(profile={})
        result = c.classify(_ann(what_doing="coding a feature"))
        assert result.activity_type == ActivityType.WORK
        assert result.confidence == 0.8  # keyword match, not modified

    def test_profile_no_working_hours(self):
        """Profile with primary_apps but no working_hours still works."""
        profile = {"tools": {"primary_apps": [{"app": "VS Code"}]}}
        c = ActivityClassifier(profile=profile)
        result = c.classify(_ann(what_doing="coding a feature"))
        assert result.activity_type == ActivityType.WORK

    def test_profile_with_apps_but_no_hours(self):
        """Profile with primary_apps but empty working_hours object."""
        profile = {
            "tools": {"primary_apps": [{"app": "Finder"}]},
            "working_hours": {},
        }
        c = ActivityClassifier(profile=profile)
        result = c.classify(_ann(active_app="Finder"))
        # Finder is not in _APP_HINTS → fallback ENTERTAINMENT @ 0.5.
        # Finder is in primary_apps → prior overrides to WORK.
        assert result.activity_type == ActivityType.WORK

    def test_prior_does_not_override_high_confidence_keyword(self):
        """Keyword match at 0.8 confidence should not be overridden by prior
        because the prior only fires when confidence < 0.9, but the app
        override only fires when type is ENTERTAINMENT or PERSONAL_ADMIN."""
        profile = self._profile_with_apps(["Terminal"])
        c = ActivityClassifier(profile=profile)
        # 'coding' → WORK at 0.8. Already WORK, so the primary-app check
        # doesn't change it (it only fires for ENTERTAINMENT/PERSONAL_ADMIN).
        result = c.classify(_ann(what_doing="coding", active_app="Terminal"))
        assert result.activity_type == ActivityType.WORK
        assert result.confidence == 0.8
        assert result.source == "heuristic"


# ===================================================================
# TestPolicyOverride — 8 tests
# ===================================================================

class TestPolicyOverride:
    """Stage 3: user policy rule overrides.

    NOTE: The current activity_classifier._stage_policy implementation
    calls rule.get("action", ...) on a PolicyRule dataclass, which will
    raise AttributeError at runtime.  These tests document the intended
    behaviour; they will fail until that bug is fixed (rule should use
    rule.action / rule.pattern instead of dict-style .get()).
    """

    def test_ignore_action_sets_learnability_ignore(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="*YouTube*",
            action=PolicyAction.IGNORE,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(active_app="YouTube Music"))
        assert result.learnability == Learnability.IGNORE

    def test_never_learn_sets_learnability_ignore(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="url", pattern="*reddit.com*",
            action=PolicyAction.NEVER_LEARN,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(location="https://reddit.com/r/python"))
        assert result.learnability == Learnability.IGNORE

    def test_classify_as_overrides_activity_type(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="url", pattern="*stackoverflow*",
            action=PolicyAction.CLASSIFY_AS,
            value="work",
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(location="https://stackoverflow.com/q/123"))
        assert result.activity_type == ActivityType.WORK

    def test_always_include_sets_execution_relevant(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Terminal",
            action=PolicyAction.ALWAYS_INCLUDE,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(active_app="Terminal"))
        assert result.learnability == Learnability.EXECUTION_RELEVANT

    def test_policy_overrides_keyword_result(self, kb):
        """Policy should override even a strong keyword match."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="*",
            action=PolicyAction.CLASSIFY_AS,
            value="entertainment",
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(
            what_doing="coding a feature", active_app="VS Code",
        ))
        assert result.activity_type == ActivityType.ENTERTAINMENT

    def test_policy_sets_confidence_and_source(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="*",
            action=PolicyAction.ALWAYS_INCLUDE,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(active_app="Anything"))
        assert result.confidence == 1.0
        assert result.source == "policy"

    def test_first_matching_rule_wins(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Chrome",
            action=PolicyAction.IGNORE,
        ))
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Chrome",
            action=PolicyAction.ALWAYS_INCLUDE,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(active_app="Chrome"))
        assert result.learnability == Learnability.IGNORE

    def test_no_matching_rule_preserves_original(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Firefox",
            action=PolicyAction.IGNORE,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(
            what_doing="coding in VS Code", active_app="VS Code",
        ))
        assert result.activity_type == ActivityType.WORK
        assert result.source == "heuristic"


# ===================================================================
# TestLearnabilityInference — 8 tests
# ===================================================================

class TestLearnabilityInference:
    """Verify _LEARNABILITY_MAP assignment for each ActivityType."""

    @pytest.mark.parametrize(
        "activity_type, expected_learnability",
        [
            (ActivityType.WORK, Learnability.EXECUTION_RELEVANT),
            (ActivityType.RESEARCH, Learnability.EXECUTION_RELEVANT),
            (ActivityType.SETUP, Learnability.EXECUTION_RELEVANT),
            (ActivityType.COMMUNICATION, Learnability.CANDIDATE_WORKFLOW),
            (ActivityType.PERSONAL_ADMIN, Learnability.CONTEXT_ONLY),
            (ActivityType.CONTEXT_SWITCH, Learnability.CONTEXT_ONLY),
            (ActivityType.ENTERTAINMENT, Learnability.IGNORE),
            (ActivityType.DEAD_TIME, Learnability.IGNORE),
        ],
    )
    def test_learnability_mapping(
        self,
        classifier: ActivityClassifier,
        activity_type: ActivityType,
        expected_learnability: Learnability,
    ):
        assert _LEARNABILITY_MAP[activity_type] == expected_learnability


# ===================================================================
# TestConfidenceValues — 4 tests
# ===================================================================

class TestConfidenceValues:
    """Verify confidence levels for each classification source."""

    def test_keyword_match_confidence(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann(what_doing="debugging a test"))
        assert result.confidence == pytest.approx(0.8)

    def test_prior_blend_confidence(self):
        profile = {
            "tools": {"primary_apps": [{"app": "Safari"}]},
            "working_hours": {"typical_start": "09:00", "typical_end": "17:00"},
        }
        c = ActivityClassifier(profile=profile)
        # Safari not in _APP_HINTS → ENTERTAINMENT @ 0.5, prior overrides to WORK @ 0.7
        result = c.classify(_ann(active_app="Safari"))
        assert result.confidence == pytest.approx(0.7)

    def test_policy_override_confidence(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="*",
            action=PolicyAction.ALWAYS_INCLUDE,
        ))
        c = ActivityClassifier(policy=policy)
        result = c.classify(_ann(active_app="SomeApp"))
        assert result.confidence == pytest.approx(1.0)

    def test_fallback_confidence(self, classifier: ActivityClassifier):
        result = classifier.classify(_ann())
        assert result.confidence == pytest.approx(0.5)


# ===================================================================
# TestBackwardCompat — 4 tests
# ===================================================================

class TestBackwardCompat:
    """Tests for classify_from_is_workflow backward-compat bridge."""

    def test_is_workflow_true_returns_work(self, classifier: ActivityClassifier):
        result = classifier.classify_from_is_workflow(True)
        assert result.activity_type == ActivityType.WORK
        assert result.learnability == Learnability.EXECUTION_RELEVANT

    def test_is_workflow_false_returns_entertainment(self, classifier: ActivityClassifier):
        result = classifier.classify_from_is_workflow(False)
        assert result.activity_type == ActivityType.ENTERTAINMENT
        assert result.learnability == Learnability.IGNORE

    def test_annotation_is_workflow_true_no_what_doing(self, classifier: ActivityClassifier):
        """is_workflow=True with empty what_doing falls back to WORK."""
        result = classifier.classify(_ann(is_workflow=True))
        assert result.activity_type == ActivityType.WORK

    def test_annotation_is_workflow_false_no_what_doing(self, classifier: ActivityClassifier):
        """is_workflow=False with empty what_doing falls back to ENTERTAINMENT."""
        result = classifier.classify(_ann(is_workflow=False))
        assert result.activity_type == ActivityType.ENTERTAINMENT


# ===================================================================
# TestAllClassesReachable — 1 test
# ===================================================================

class TestAllClassesReachable:
    """Confirm every ActivityType is reachable through classify()."""

    def test_all_8_activity_types_reachable(self, classifier: ActivityClassifier):
        annotations = [
            _ann(what_doing="coding a feature"),           # WORK
            _ann(what_doing="studying a new framework"),   # RESEARCH
            _ann(what_doing="emailing the team"),          # COMMUNICATION
            _ann(what_doing="installing dependencies"),    # SETUP
            _ann(what_doing="paying bill for internet"),   # PERSONAL_ADMIN
            _ann(what_doing="watching a movie"),           # ENTERTAINMENT
            _ann(what_doing="idle at the screen"),         # DEAD_TIME
            _ann(what_doing="switching between tabs"),     # CONTEXT_SWITCH
        ]
        types_seen = {
            classifier.classify(ann).activity_type for ann in annotations
        }
        assert types_seen == set(ActivityType)


# ===================================================================
# TestEdgeCases — 5 tests
# ===================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_empty_what_doing(self, classifier: ActivityClassifier):
        """Empty what_doing should not crash and should use fallback."""
        result = classifier.classify(_ann(what_doing=""))
        assert isinstance(result, ClassificationResult)
        assert result.confidence == 0.5

    def test_none_annotation_fields(self, classifier: ActivityClassifier):
        """Annotation with None values instead of strings should not crash."""
        ann = {
            "task_context": {"what_doing": None, "is_workflow": False},
            "visual_context": {"active_app": None, "location": None},
            "app": None,
        }
        result = classifier.classify(ann)
        assert isinstance(result, ClassificationResult)

    def test_missing_task_context_key(self, classifier: ActivityClassifier):
        """Annotation missing task_context entirely should not crash."""
        result = classifier.classify({"visual_context": {"location": ""}})
        assert isinstance(result, ClassificationResult)

    def test_keyword_wins_over_url_when_both_match(self, classifier: ActivityClassifier):
        """When both keyword and URL match, keyword takes priority (checked first)."""
        # 'coding' → WORK via keyword. youtube.com → ENTERTAINMENT via URL.
        # Keyword is checked first, so WORK should win.
        result = classifier.classify(_ann(
            what_doing="coding while on youtube.com",
            location="https://youtube.com/watch",
        ))
        assert result.activity_type == ActivityType.WORK

    def test_first_keyword_table_match_wins(self, classifier: ActivityClassifier):
        """When what_doing matches keywords from multiple tables, the first
        table in iteration order wins (WORK is first in _KEYWORD_TABLE)."""
        # 'debugging' is WORK, 'researching' is RESEARCH.
        # The iteration order of _KEYWORD_TABLE determines which wins.
        result = classifier.classify(_ann(what_doing="debugging and researching"))
        # WORK table is iterated first, so 'debugging' matches first.
        assert result.activity_type == ActivityType.WORK

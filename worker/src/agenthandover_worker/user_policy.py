"""User policy controls for AgentHandover activity classification.

Provides per-user rules that override or refine how activities are
classified and whether they are learnable.  Rules are stored as an
ordered list in ``~/.agenthandover/knowledge/policy.json`` and evaluated
first-match-wins (like firewall rules).

Supported rule types:
  - ``app``             — match against application name
  - ``url``             — match against page URL
  - ``browser_profile`` — match against browser profile name
  - ``source``          — match against URL (reserved for future source context)

All pattern matching uses :func:`fnmatch.fnmatch` with case-insensitive
comparison (both sides lowercased).
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_POLICY_FILE = "policy.json"


class PolicyAction(str, Enum):
    """Action to take when a policy rule matches an activity."""

    IGNORE = "ignore"
    NEVER_LEARN = "never_learn"
    CLASSIFY_AS = "classify_as"
    ALWAYS_INCLUDE = "always_include"


@dataclass
class PolicyRule:
    """A single user policy rule.

    Attributes:
        rule_type: The dimension to match against — one of ``"app"``,
            ``"url"``, ``"browser_profile"``, or ``"source"``.
        pattern: An fnmatch-style glob pattern (case-insensitive).
        action: What to do when the rule matches.
        value: Only used with :attr:`PolicyAction.CLASSIFY_AS` — the
            activity_type string to assign.
        comment: Optional human-readable note explaining the rule.
    """

    rule_type: str
    pattern: str
    action: PolicyAction
    value: str = ""
    comment: str = ""


class UserPolicy:
    """Manages user policy rules for activity classification.

    Rules are persisted in the knowledge base as ``policy.json`` and
    evaluated in order (first match wins).
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        self._rules: list[PolicyRule] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load rules from KB policy.json.

        If the file does not exist or contains invalid JSON, initialises
        with an empty rule list.  Malformed individual rules are skipped
        with a warning.
        """
        path = self._kb.root / _POLICY_FILE
        if not path.is_file():
            self._rules = []
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            self._rules = []
            return

        if not isinstance(data, dict):
            logger.warning(
                "Expected dict in %s, got %s", path, type(data).__name__
            )
            self._rules = []
            return

        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            logger.warning(
                "Expected list for 'rules' in %s, got %s",
                path,
                type(raw_rules).__name__,
            )
            self._rules = []
            return

        rules: list[PolicyRule] = []
        for i, entry in enumerate(raw_rules):
            if not isinstance(entry, dict):
                logger.warning(
                    "Skipping non-dict rule at index %d in %s", i, path
                )
                continue
            try:
                rule = PolicyRule(
                    rule_type=entry["rule_type"],
                    pattern=entry["pattern"],
                    action=PolicyAction(entry["action"]),
                    value=entry.get("value", ""),
                    comment=entry.get("comment", ""),
                )
                rules.append(rule)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid rule at index %d in %s: %s", i, path, exc
                )
                continue

        self._rules = rules

    def load_rules(self) -> list[PolicyRule]:
        """Return current rules list."""
        return list(self._rules)

    def save_rules(self) -> None:
        """Persist current rules to KB using atomic write."""
        payload = {
            "rules": [asdict(r) for r in self._rules],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / _POLICY_FILE
        self._kb.atomic_write_json(path, payload)
        logger.info("Policy rules saved (%d rules)", len(self._rules))

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: PolicyRule) -> None:
        """Append a rule and persist immediately."""
        self._rules.append(rule)
        self.save_rules()

    def remove_rule(self, index: int) -> None:
        """Remove the rule at *index* and persist.

        Raises:
            IndexError: If *index* is out of range.
        """
        if index < 0 or index >= len(self._rules):
            raise IndexError(
                f"Rule index {index} out of range (0..{len(self._rules) - 1})"
            )
        del self._rules[index]
        self.save_rules()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def check(
        self,
        app: str = "",
        url: str = "",
        browser_profile: str = "",
    ) -> PolicyRule | None:
        """Check an activity against all rules in order.

        First matching rule wins.  Returns ``None`` if no rule matches.

        Args:
            app: Application name (e.g. ``"VS Code"``).
            url: Page URL (e.g. ``"https://youtube.com/watch?v=..."``).
            browser_profile: Browser profile name (e.g. ``"Default"``).

        Returns:
            The first matching :class:`PolicyRule`, or ``None``.
        """
        for rule in self._rules:
            if self._matches(rule, app=app, url=url, browser_profile=browser_profile):
                return rule
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(
        rule: PolicyRule,
        *,
        app: str,
        url: str,
        browser_profile: str,
    ) -> bool:
        """Return True if *rule* matches the given context fields."""
        if rule.rule_type == "app":
            return bool(app) and fnmatch.fnmatch(app.lower(), rule.pattern.lower())
        if rule.rule_type == "url":
            return bool(url) and fnmatch.fnmatch(url.lower(), rule.pattern.lower())
        if rule.rule_type == "browser_profile":
            return bool(browser_profile) and fnmatch.fnmatch(
                browser_profile.lower(), rule.pattern.lower()
            )
        if rule.rule_type == "source":
            return bool(url) and fnmatch.fnmatch(url.lower(), rule.pattern.lower())
        logger.warning("Unknown rule_type %r — skipping", rule.rule_type)
        return False

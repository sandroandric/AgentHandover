"""Privacy zone enforcement for observation events.

Implements privacy zoning from section 12 of the AgentHandoverBible:
four observation tiers (full, metadata-only, blocked, pause) with
default blocked lists for sensitive apps and URLs, configurable
per-user overrides, and time-based auto-pause windows.

The daemon evaluates the same zones before it captures a frame
(``crates/daemon/src/observer/privacy_zones.rs``): a blocked window never
gets a screenshot.  This checker is the second line, on stored events: it
catches frames written before a zone was configured, by an older daemon,
or while the browser URL was not yet known, and the caller deletes their
JPEG.

Defence layers:
1. Auto-pause time windows (highest priority — blocks everything)
2. URL pattern matching (fnmatch glob)
3. App name / bundle ID / window title pattern matching (fnmatch glob)
4. Default blocked lists for password managers, banking, auth sites

Every match is case-insensitive: patterns are folded once, at construction,
and each candidate once per check.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum
from typing import ClassVar

from agenthandover_worker.event_helpers import display_name, parse_json_field

logger = logging.getLogger(__name__)


class ObservationTier(Enum):
    """Observation tier controlling what data is captured.

    Ordered from least to most restrictive:
      FULL          — capture everything (screenshots, DOM, events)
      METADATA_ONLY — capture app/timestamp but no content
      BLOCKED       — drop the event entirely
    """
    FULL = "full"
    METADATA_ONLY = "metadata_only"
    BLOCKED = "blocked"


# Tier restrictiveness ranking (higher = more restrictive)
_TIER_RANK: dict[ObservationTier, int] = {
    ObservationTier.FULL: 0,
    ObservationTier.METADATA_ONLY: 1,
    ObservationTier.BLOCKED: 2,
}


@dataclass
class PrivacyZoneConfig:
    """User-supplied privacy zone configuration.

    All list fields accept fnmatch-style glob patterns.

    Attributes:
        full_observation: App name/bundle ID patterns that are always
            fully observed (override defaults).
        metadata_only: App patterns observed at metadata level only.
        blocked: App name/bundle ID patterns that are blocked.
        blocked_urls: URL patterns that are blocked (case-insensitive).
        auto_pause: Time windows in "HH:MM-HH:MM" format during which
            all observation is paused.  Supports overnight windows
            (e.g. "22:00-06:00").
    """
    full_observation: list[str] = field(default_factory=list)
    metadata_only: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    blocked_urls: list[str] = field(default_factory=list)
    auto_pause: list[str] = field(default_factory=list)


class PrivacyZoneChecker:
    """Checks events against privacy zones and returns the applicable tier.

    Merges a default blocked list (password managers, banking apps) with
    user-supplied configuration.  The most restrictive tier always wins
    when multiple rules match.

    The default lists are mirrored in the daemon
    (``DEFAULT_BLOCKED_APPS`` / ``DEFAULT_BLOCKED_URLS`` in
    ``crates/daemon/src/observer/privacy_zones.rs``); a test keeps them equal.
    """

    # Default blocked apps — password managers, keychains, security tools
    _DEFAULT_BLOCKED: ClassVar[set[str]] = {
        "1Password",
        "1Password 7",
        "1Password 8",
        "Bitwarden",
        "LastPass",
        "Dashlane",
        "KeePassXC",
        "Keychain Access",
    }

    # Default blocked URL patterns — banking, gov, payroll, auth pages
    _DEFAULT_BLOCKED_URLS: ClassVar[list[str]] = [
        "*bank*",
        "*banking*",
        "*.gov/*",
        "*payroll*",
        "*paycheck*",
        "*salary*",
        "*benefits*",
        "*1password.com*",
        "*bitwarden.com*",
        "*lastpass.com*",
        "*dashlane.com*",
        "*accounts.google.com/signin*",
        "*login.*",
        "*signin.*",
    ]

    def __init__(self, config: PrivacyZoneConfig | None = None) -> None:
        cfg = config or PrivacyZoneConfig()

        # Every pattern is case-folded once, here.
        def fold(patterns) -> list[str]:
            folded: list[str] = []
            for pattern in patterns:
                p = str(pattern).casefold()
                if p not in folded:
                    folded.append(p)
            return folded

        # User-blocked apps and URLs are merged with the defaults.
        self._blocked_apps: list[str] = fold([*self._DEFAULT_BLOCKED, *cfg.blocked])
        self._blocked_urls: list[str] = fold(
            [*self._DEFAULT_BLOCKED_URLS, *cfg.blocked_urls]
        )
        # Explicit full-observation overrides
        self._full_observation: list[str] = fold(cfg.full_observation)
        # Metadata-only patterns
        self._metadata_only: list[str] = fold(cfg.metadata_only)

        # Parse auto-pause time windows into (start, end) tuples
        self._auto_pause_windows: list[tuple[dt_time, dt_time]] = []
        for window_str in cfg.auto_pause:
            parsed = self._parse_time_window(window_str)
            if parsed is not None:
                self._auto_pause_windows.append(parsed)

        logger.debug(
            "PrivacyZoneChecker initialised: %d blocked apps, "
            "%d blocked URL patterns, %d auto-pause windows",
            len(self._blocked_apps),
            len(self._blocked_urls),
            len(self._auto_pause_windows),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_event(self, event: dict) -> ObservationTier:
        """Determine the observation tier for a raw event dict.

        Checks (in order of priority):
          1. Auto-pause time windows → BLOCKED
          2. App and title from window_json → tier per check_app()
          3. URL from metadata_json → tier per check_url()

        The most restrictive result across all checks wins.

        Args:
            event: Raw event dict as stored in the daemon DB.  Expected
                keys: ``window_json`` (JSON with ``app_id`` —
                ``pid:<pid>:<name>`` for native windows, a bundle id for
                browser events — and ``title``), ``metadata_json`` (JSON,
                may contain ``url``), ``kind_json``.

        Returns:
            The applicable ObservationTier.
        """
        # 1. Auto-pause overrides everything.  The window is checked at the
        # time the event happened, not when the worker reaches it.
        if self.is_auto_paused(self._event_local_time(event)):
            return ObservationTier.BLOCKED

        most_restrictive = ObservationTier.FULL

        # 2. Extract app info from window_json
        app_name, bundle_id, title = self._extract_app_info(event)
        if app_name or bundle_id or title:
            app_tier = self.check_app(bundle_id, app_name, title)
            most_restrictive = self._most_restrictive(most_restrictive, app_tier)

        # 3. Extract URL from metadata_json
        url = self._extract_url(event)
        if url:
            url_tier = self.check_url(url)
            most_restrictive = self._most_restrictive(most_restrictive, url_tier)

        return most_restrictive

    def check_app(
        self, bundle_id: str, app_name: str, title: str = "",
    ) -> ObservationTier:
        """Determine the observation tier for an app (and its window title).

        Checks in order:
          1. Explicit full_observation patterns → FULL
          2. Blocked patterns (defaults + user) → BLOCKED
          3. Metadata-only patterns → METADATA_ONLY
          4. Default → FULL

        Note: blocked takes priority over metadata_only, but
        full_observation overrides blocked (allows user to unblock
        a default-blocked app).

        Args:
            bundle_id: macOS bundle identifier (e.g. "com.1password.1password").
            app_name: Human-readable app name (e.g. "1Password").
            title: The window title.

        Returns:
            The applicable ObservationTier.
        """
        candidates = [
            c.casefold() for c in (app_name, bundle_id, title) if c
        ]

        def matches(patterns: list[str]) -> bool:
            return any(
                fnmatch.fnmatchcase(candidate, pattern)
                for pattern in patterns
                for candidate in candidates
            )

        # Explicit full-observation override first, then blocked, then
        # metadata-only.
        if matches(self._full_observation):
            return ObservationTier.FULL
        if matches(self._blocked_apps):
            return ObservationTier.BLOCKED
        if matches(self._metadata_only):
            return ObservationTier.METADATA_ONLY
        return ObservationTier.FULL

    def check_url(self, url: str) -> ObservationTier:
        """Determine the observation tier for a URL.

        Uses case-insensitive fnmatch glob matching against blocked
        URL patterns (defaults + user-supplied, folded at construction).

        Args:
            url: The full URL string to check.

        Returns:
            BLOCKED if any pattern matches, otherwise FULL.
        """
        if not url:
            return ObservationTier.FULL

        folded = url.casefold()
        for pattern in self._blocked_urls:
            if fnmatch.fnmatchcase(folded, pattern):
                return ObservationTier.BLOCKED

        return ObservationTier.FULL

    def is_auto_paused(self, at: dt_time | None = None) -> bool:
        """Check if local time *at* (default: now) falls within any
        auto-pause window.

        Supports overnight windows (e.g. "22:00-06:00" means paused
        from 10pm to 6am).

        Returns:
            True if observation is paused at that time.
        """
        if not self._auto_pause_windows:
            return False

        now = at if at is not None else datetime.now().time()

        for start, end in self._auto_pause_windows:
            if start <= end:
                # Same-day window: e.g. 09:00-17:00
                if start <= now <= end:
                    return True
            else:
                # Overnight window: e.g. 22:00-06:00
                # Active from start->midnight OR midnight->end
                if now >= start or now <= end:
                    return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time_window(window_str: str) -> tuple[dt_time, dt_time] | None:
        """Parse a "HH:MM-HH:MM" string into a (start, end) time tuple.

        Returns None if the format is invalid.
        """
        parts = window_str.strip().split("-")
        if len(parts) != 2:
            logger.warning("Invalid auto_pause window format: %r", window_str)
            return None

        try:
            start_parts = parts[0].strip().split(":")
            end_parts = parts[1].strip().split(":")

            if len(start_parts) != 2 or len(end_parts) != 2:
                raise ValueError("Expected HH:MM format")

            start = dt_time(int(start_parts[0]), int(start_parts[1]))
            end = dt_time(int(end_parts[0]), int(end_parts[1]))
            return (start, end)
        except (ValueError, IndexError) as exc:
            logger.warning(
                "Failed to parse auto_pause window %r: %s", window_str, exc
            )
            return None

    @staticmethod
    def _extract_app_info(event: dict) -> tuple[str, str, str]:
        """Extract (app_name, bundle_id, title) from an event's window_json.

        The daemon writes ``app_id``: ``pid:<pid>:<name>`` for native
        windows (the ``pid:<pid>:`` prefix is stripped: the name is the app
        name) and a bundle id for browser extension events.  Legacy
        ``app`` / ``app_name`` / ``app_bundle_id`` keys are honoured too.

        Returns ("", "", "") if window_json is missing or unparseable.
        """
        window = parse_json_field(event, "window_json")
        if not window:
            return ("", "", "")

        app_id = str(window.get("app_id") or "")
        app_name = str(window.get("app") or window.get("app_name") or "")
        if not app_name and app_id:
            app_name = display_name(app_id)
        bundle_id = str(window.get("app_bundle_id") or "")
        if not bundle_id and app_id and not app_id.startswith("pid:"):
            bundle_id = app_id
        title = str(window.get("title") or "")
        return (app_name, bundle_id, title)

    @staticmethod
    def _event_local_time(event: dict) -> dt_time | None:
        """The event's timestamp as local wall-clock time, or None."""
        raw = event.get("timestamp")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if stamp.tzinfo is None:
            return None
        return stamp.astimezone().time()

    @staticmethod
    def _extract_url(event: dict) -> str:
        """Extract a URL from an event's metadata_json.

        Returns "" if no URL is found.
        """
        metadata_json_raw = event.get("metadata_json", "")
        if not metadata_json_raw:
            return ""

        try:
            metadata = (
                json.loads(metadata_json_raw)
                if isinstance(metadata_json_raw, str)
                else metadata_json_raw
            )
        except (json.JSONDecodeError, TypeError):
            return ""

        if not isinstance(metadata, dict):
            return ""

        return str(metadata.get("url", ""))

    @staticmethod
    def _most_restrictive(
        a: ObservationTier, b: ObservationTier
    ) -> ObservationTier:
        """Return the more restrictive of two observation tiers."""
        if _TIER_RANK[a] >= _TIER_RANK[b]:
            return a
        return b

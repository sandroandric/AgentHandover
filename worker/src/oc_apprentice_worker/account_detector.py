"""Account/environment detector for context enrichment.

Detects which service account and environment (prod/staging/test) the
user is working in based on URLs, window titles, and VLM annotations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AccountContext:
    """Detected account/environment context."""

    service: str  # "stripe", "github", "gmail"
    identity: str  # "personal", "work", "client-co"
    environment: str  # "production", "staging", "test"
    url_pattern: str | None = None


# URL → service mapping
_SERVICE_PATTERNS: list[tuple[str, str]] = [
    (r"github\.com", "github"),
    (r"gitlab\.com", "gitlab"),
    (r"bitbucket\.org", "bitbucket"),
    (r"mail\.google\.com|gmail\.com", "gmail"),
    (r"calendar\.google\.com", "google_calendar"),
    (r"drive\.google\.com|docs\.google\.com|sheets\.google\.com", "google_drive"),
    (r"slack\.com|app\.slack\.com", "slack"),
    (r"notion\.so", "notion"),
    (r"figma\.com", "figma"),
    (r"stripe\.com|dashboard\.stripe\.com", "stripe"),
    (r"console\.aws\.amazon\.com", "aws"),
    (r"portal\.azure\.com", "azure"),
    (r"console\.cloud\.google\.com", "gcp"),
    (r"vercel\.com", "vercel"),
    (r"netlify\.com|app\.netlify\.com", "netlify"),
    (r"heroku\.com|dashboard\.heroku\.com", "heroku"),
    (r"linear\.app", "linear"),
    (r"atlassian\.net|jira", "jira"),
    (r"trello\.com", "trello"),
    (r"airtable\.com", "airtable"),
    (r"twitter\.com|x\.com", "twitter"),
    (r"linkedin\.com", "linkedin"),
    (r"openai\.com|platform\.openai\.com", "openai"),
    (r"anthropic\.com|console\.anthropic\.com", "anthropic"),
]

# Environment detection patterns
_ENV_PATTERNS: list[tuple[str, str]] = [
    (r"localhost|127\.0\.0\.1|0\.0\.0\.0", "development"),
    (r"staging\.|stage\.|stg\.", "staging"),
    (r"test\.|testing\.", "test"),
    (r"/test/|/test$", "test"),
    (r"preview\.|preview/", "preview"),
    (r"dev\.|\.dev/|development\.", "development"),
    (r"sandbox\.", "sandbox"),
    (r"prod\.|production\.", "production"),
]

# Title-based service detection (for native apps)
_TITLE_SERVICE_MAP: dict[str, str] = {
    "slack": "slack",
    "discord": "discord",
    "teams": "microsoft_teams",
    "zoom": "zoom",
    "figma": "figma",
    "notion": "notion",
    "linear": "linear",
    "xcode": "xcode",
    "vs code": "vscode",
    "visual studio code": "vscode",
}


class AccountDetector:
    """Detect service, identity, and environment from context."""

    def __init__(self) -> None:
        # Compile regexes once
        self._service_re = [
            (re.compile(pattern, re.IGNORECASE), service)
            for pattern, service in _SERVICE_PATTERNS
        ]
        self._env_re = [
            (re.compile(pattern, re.IGNORECASE), env)
            for pattern, env in _ENV_PATTERNS
        ]

    def detect_from_url(self, url: str) -> AccountContext | None:
        """Detect account context from a URL.

        Returns None if no known service matches.
        """
        if not url:
            return None

        service = self._match_service(url)
        if service is None:
            return None

        environment = self._detect_environment(url)
        identity = self._detect_identity(url)

        return AccountContext(
            service=service,
            identity=identity,
            environment=environment,
            url_pattern=url,
        )

    def detect_from_title(self, title: str) -> AccountContext | None:
        """Detect account context from a window title.

        Returns None if no known service matches.
        """
        if not title:
            return None

        title_lower = title.lower()
        for keyword, service in _TITLE_SERVICE_MAP.items():
            if keyword in title_lower:
                return AccountContext(
                    service=service,
                    identity="unknown",
                    environment="production",
                )

        return None

    def detect_from_annotation(self, annotation: dict) -> AccountContext | None:
        """Detect account context from a VLM annotation dict.

        Tries URL first (from visual_context.location), then falls
        back to app name / title matching.
        """
        if not isinstance(annotation, dict):
            return None

        vc = annotation.get("visual_context", {})
        if not isinstance(vc, dict):
            return None

        # Try URL first
        location = vc.get("location", "")
        if location and location.startswith("http"):
            result = self.detect_from_url(location)
            if result is not None:
                return result

        # Try app name
        app = vc.get("active_app", "")
        if app:
            result = self.detect_from_title(app)
            if result is not None:
                return result

        return None

    def _match_service(self, url: str) -> str | None:
        """Match URL against known service patterns."""
        for pattern, service in self._service_re:
            if pattern.search(url):
                return service
        return None

    def _detect_environment(self, url: str) -> str:
        """Detect environment from URL patterns."""
        for pattern, env in self._env_re:
            if pattern.search(url):
                return env
        return "production"

    def _detect_identity(self, url: str) -> str:
        """Try to detect identity (personal vs work) from URL."""
        # Simple heuristic: org-specific subdomains
        if "personal" in url.lower():
            return "personal"
        # Default to unknown — will be enriched by profile builder
        return "unknown"

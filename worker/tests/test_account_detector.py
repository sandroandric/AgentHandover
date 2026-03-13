"""Tests for the account detector."""

from __future__ import annotations

import pytest

from oc_apprentice_worker.account_detector import AccountContext, AccountDetector


@pytest.fixture()
def detector() -> AccountDetector:
    return AccountDetector()


# ---------------------------------------------------------------------------
# detect_from_url
# ---------------------------------------------------------------------------

class TestDetectFromURL:

    def test_github(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://github.com/user/repo")
        assert result is not None
        assert result.service == "github"

    def test_gmail(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://mail.google.com/mail/u/0/#inbox")
        assert result is not None
        assert result.service == "gmail"

    def test_stripe(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://dashboard.stripe.com/test/payments")
        assert result is not None
        assert result.service == "stripe"
        assert result.environment == "test"

    def test_stripe_production(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://dashboard.stripe.com/payments")
        assert result is not None
        assert result.service == "stripe"
        assert result.environment == "production"

    def test_slack(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://app.slack.com/client/T123/C456")
        assert result is not None
        assert result.service == "slack"

    def test_aws_console(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://console.aws.amazon.com/s3/buckets")
        assert result is not None
        assert result.service == "aws"

    def test_vercel(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://vercel.com/dashboard")
        assert result is not None
        assert result.service == "vercel"

    def test_unknown_url(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://example.com/page")
        assert result is None

    def test_empty_url(self, detector: AccountDetector) -> None:
        assert detector.detect_from_url("") is None

    def test_localhost_development(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("http://localhost:3000")
        assert result is None  # localhost isn't a known service

    def test_staging_environment(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://staging.stripe.com/dashboard")
        assert result is not None
        assert result.environment == "staging"

    def test_google_drive(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://docs.google.com/spreadsheets/d/123")
        assert result is not None
        assert result.service == "google_drive"

    def test_linear(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://linear.app/team/issue/123")
        assert result is not None
        assert result.service == "linear"

    def test_notion(self, detector: AccountDetector) -> None:
        result = detector.detect_from_url("https://www.notion.so/workspace/page-123")
        assert result is not None
        assert result.service == "notion"


# ---------------------------------------------------------------------------
# detect_from_title
# ---------------------------------------------------------------------------

class TestDetectFromTitle:

    def test_slack_app(self, detector: AccountDetector) -> None:
        result = detector.detect_from_title("Slack - #general")
        assert result is not None
        assert result.service == "slack"

    def test_vscode(self, detector: AccountDetector) -> None:
        result = detector.detect_from_title("main.py - VS Code")
        assert result is not None
        assert result.service == "vscode"

    def test_discord(self, detector: AccountDetector) -> None:
        result = detector.detect_from_title("Discord - Server")
        assert result is not None
        assert result.service == "discord"

    def test_unknown_title(self, detector: AccountDetector) -> None:
        assert detector.detect_from_title("My Custom App") is None

    def test_empty_title(self, detector: AccountDetector) -> None:
        assert detector.detect_from_title("") is None


# ---------------------------------------------------------------------------
# detect_from_annotation
# ---------------------------------------------------------------------------

class TestDetectFromAnnotation:

    def test_annotation_with_url(self, detector: AccountDetector) -> None:
        ann = {
            "visual_context": {
                "active_app": "Google Chrome",
                "location": "https://github.com/user/repo",
            }
        }
        result = detector.detect_from_annotation(ann)
        assert result is not None
        assert result.service == "github"

    def test_annotation_with_app_only(self, detector: AccountDetector) -> None:
        ann = {
            "visual_context": {
                "active_app": "Slack",
                "location": "",
            }
        }
        result = detector.detect_from_annotation(ann)
        assert result is not None
        assert result.service == "slack"

    def test_annotation_url_takes_priority(self, detector: AccountDetector) -> None:
        ann = {
            "visual_context": {
                "active_app": "Google Chrome",
                "location": "https://app.slack.com/client",
            }
        }
        result = detector.detect_from_annotation(ann)
        assert result is not None
        assert result.service == "slack"

    def test_annotation_no_match(self, detector: AccountDetector) -> None:
        ann = {
            "visual_context": {
                "active_app": "Finder",
                "location": "/Users/test",
            }
        }
        result = detector.detect_from_annotation(ann)
        assert result is None

    def test_annotation_empty(self, detector: AccountDetector) -> None:
        assert detector.detect_from_annotation({}) is None

    def test_annotation_not_dict(self, detector: AccountDetector) -> None:
        assert detector.detect_from_annotation("not a dict") is None  # type: ignore[arg-type]

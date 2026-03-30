"""Tests for the Privacy Zones module.

Comprehensive tests for observation tier enforcement per spec section 12.

Covers:
1.  TestDefaultBlockedApps — password managers / keychains blocked by default
2.  TestDefaultBlockedURLs — banking, .gov, payroll URLs blocked by default
3.  TestCustomBlockedApps — user-specified blocked app patterns
4.  TestCustomBlockedURLs — user-specified blocked URL patterns
5.  TestFullObservationApps — explicitly allowed apps override defaults
6.  TestMetadataOnlyApps — metadata-only tier apps
7.  TestCheckEventBlockedApp — check_event with blocked app → BLOCKED
8.  TestCheckEventBlockedURL — check_event with blocked URL → BLOCKED
9.  TestCheckEventNormalApp — check_event with normal app → FULL
10. TestCheckAppGlob — glob pattern matching for app names
11. TestCheckURLGlob — glob pattern matching for URLs (case insensitive)
12. TestAutoPauseDuringWindow — auto-pause within window → BLOCKED
13. TestAutoPauseOutsideWindow — auto-pause outside window → FULL
14. TestAutoPauseOvernight — overnight window (22:00-06:00)
15. TestMultipleAutoPauseWindows — multiple time windows
16. TestEmptyConfig — defaults apply with empty config
17. TestNoneConfig — defaults apply with None config
18. TestCombinedAppAllowedURLBlocked — most restrictive wins
19. TestPartialAppMatches — partial matches with glob patterns
20. TestEventMissingWindowJSON — missing window_json → FULL (default)
21. TestEventMissingMetadataJSON — missing metadata_json → check app only
22. TestObservationTierEnum — enum values are correct strings
23. TestPrivacyZoneConfigDefaults — config defaults are empty lists
24. TestBundleIDMatching — bundle IDs checked against blocked patterns
25. TestURLCaseInsensitivity — URL matching is case insensitive
26. TestTimeWindowParsing — valid and invalid time window formats
27. TestAutoPauseBoundaryTimes — edge cases at window boundaries
28. TestMostRestrictiveTier — tier comparison logic
29. TestEventWithDictWindowJSON — window_json as dict (not string)
30. TestEventWithDictMetadataJSON — metadata_json as dict (not string)
"""

from __future__ import annotations

import json
from datetime import datetime, time as dt_time
from unittest.mock import patch

import pytest

from agenthandover_worker.privacy_zones import (
    ObservationTier,
    PrivacyZoneChecker,
    PrivacyZoneConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    app: str = "",
    bundle_id: str = "",
    title: str = "",
    url: str = "",
    kind: str = "DwellSnapshot",
) -> dict:
    """Build a minimal event dict for testing."""
    event: dict = {
        "id": "test-event-1",
        "timestamp": "2026-03-11T10:00:00.000Z",
        "kind_json": json.dumps({kind: {}}),
    }
    if app or bundle_id or title:
        window: dict = {}
        if app:
            window["app"] = app
        if bundle_id:
            window["app_bundle_id"] = bundle_id
        if title:
            window["title"] = title
        event["window_json"] = json.dumps(window)
    if url:
        event["metadata_json"] = json.dumps({"url": url})
    return event


# ---------------------------------------------------------------------------
# Test 1: Default Blocked Apps
# ---------------------------------------------------------------------------


class TestDefaultBlockedApps:
    def test_1password_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "1Password") == ObservationTier.BLOCKED

    def test_1password_7_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "1Password 7") == ObservationTier.BLOCKED

    def test_1password_8_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "1Password 8") == ObservationTier.BLOCKED

    def test_bitwarden_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "Bitwarden") == ObservationTier.BLOCKED

    def test_lastpass_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "LastPass") == ObservationTier.BLOCKED

    def test_dashlane_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "Dashlane") == ObservationTier.BLOCKED

    def test_keepassxc_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "KeePassXC") == ObservationTier.BLOCKED

    def test_keychain_access_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "Keychain Access") == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 2: Default Blocked URLs
# ---------------------------------------------------------------------------


class TestDefaultBlockedURLs:
    def test_bank_url_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://www.chasebank.com/accounts") == ObservationTier.BLOCKED

    def test_banking_url_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://onlinebanking.example.com") == ObservationTier.BLOCKED

    def test_gov_url_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://www.irs.gov/refunds") == ObservationTier.BLOCKED

    def test_payroll_url_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://payroll.gusto.com/dashboard") == ObservationTier.BLOCKED

    def test_salary_url_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://hr.example.com/salary") == ObservationTier.BLOCKED

    def test_1password_com_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://my.1password.com/vaults") == ObservationTier.BLOCKED

    def test_google_signin_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://accounts.google.com/signin/v2") == ObservationTier.BLOCKED

    def test_login_page_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://login.example.com/auth") == ObservationTier.BLOCKED

    def test_signin_page_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://signin.microsoft.com") == ObservationTier.BLOCKED

    def test_normal_url_allowed(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://github.com/pulls") == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 3: Custom Blocked Apps
# ---------------------------------------------------------------------------


class TestCustomBlockedApps:
    def test_custom_app_blocked(self) -> None:
        config = PrivacyZoneConfig(blocked=["Slack"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "Slack") == ObservationTier.BLOCKED

    def test_custom_glob_pattern_blocked(self) -> None:
        config = PrivacyZoneConfig(blocked=["*VPN*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "NordVPN") == ObservationTier.BLOCKED

    def test_defaults_still_apply_with_custom(self) -> None:
        config = PrivacyZoneConfig(blocked=["CustomApp"])
        checker = PrivacyZoneChecker(config)
        # Default blocked should still work
        assert checker.check_app("", "1Password") == ObservationTier.BLOCKED
        # Custom should also work
        assert checker.check_app("", "CustomApp") == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 4: Custom Blocked URLs
# ---------------------------------------------------------------------------


class TestCustomBlockedURLs:
    def test_custom_url_blocked(self) -> None:
        config = PrivacyZoneConfig(blocked_urls=["*secret-internal.corp.com*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_url("https://secret-internal.corp.com/docs") == ObservationTier.BLOCKED

    def test_custom_url_glob(self) -> None:
        config = PrivacyZoneConfig(blocked_urls=["*medical*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_url("https://portal.medical-records.com") == ObservationTier.BLOCKED

    def test_defaults_still_apply_with_custom_urls(self) -> None:
        config = PrivacyZoneConfig(blocked_urls=["*custom-blocked.com*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_url("https://www.chasebank.com/login") == ObservationTier.BLOCKED
        assert checker.check_url("https://custom-blocked.com/page") == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 5: Full Observation Apps (explicit overrides)
# ---------------------------------------------------------------------------


class TestFullObservationApps:
    def test_full_observation_override_blocks_default(self) -> None:
        """Explicitly allowing an app that is default-blocked should FULL it."""
        config = PrivacyZoneConfig(full_observation=["1Password"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "1Password") == ObservationTier.FULL

    def test_full_observation_glob_pattern(self) -> None:
        config = PrivacyZoneConfig(
            full_observation=["My*"],
            blocked=["MySecretApp"],
        )
        checker = PrivacyZoneChecker(config)
        # full_observation wins over blocked
        assert checker.check_app("", "MySecretApp") == ObservationTier.FULL

    def test_full_observation_normal_app(self) -> None:
        config = PrivacyZoneConfig(full_observation=["Safari"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "Safari") == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 6: Metadata Only Apps
# ---------------------------------------------------------------------------


class TestMetadataOnlyApps:
    def test_metadata_only_app(self) -> None:
        config = PrivacyZoneConfig(metadata_only=["Messages"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "Messages") == ObservationTier.METADATA_ONLY

    def test_metadata_only_glob(self) -> None:
        config = PrivacyZoneConfig(metadata_only=["*Chat*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "Google Chat") == ObservationTier.METADATA_ONLY

    def test_blocked_overrides_metadata_only(self) -> None:
        """Blocked is more restrictive than metadata_only."""
        config = PrivacyZoneConfig(
            metadata_only=["1Password"],
            # 1Password is in default blocked
        )
        checker = PrivacyZoneChecker(config)
        # Blocked comes before metadata_only in check_app logic
        assert checker.check_app("", "1Password") == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 7: check_event with Blocked App
# ---------------------------------------------------------------------------


class TestCheckEventBlockedApp:
    def test_event_with_blocked_app(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="1Password")
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_event_with_blocked_bundle_id(self) -> None:
        config = PrivacyZoneConfig(blocked=["com.1password.*"])
        checker = PrivacyZoneChecker(config)
        event = _make_event(bundle_id="com.1password.1password7")
        assert checker.check_event(event) == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 8: check_event with Blocked URL
# ---------------------------------------------------------------------------


class TestCheckEventBlockedURL:
    def test_event_with_blocked_url(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="Safari", url="https://login.example.com/auth")
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_event_with_banking_url(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="Chrome", url="https://onlinebanking.chase.com")
        assert checker.check_event(event) == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 9: check_event with Normal App
# ---------------------------------------------------------------------------


class TestCheckEventNormalApp:
    def test_normal_app_full(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="Visual Studio Code")
        assert checker.check_event(event) == ObservationTier.FULL

    def test_normal_app_with_normal_url(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="Safari", url="https://docs.python.org/3/")
        assert checker.check_event(event) == ObservationTier.FULL

    def test_empty_event(self) -> None:
        checker = PrivacyZoneChecker()
        event = {"id": "test", "timestamp": "2026-03-11T10:00:00Z"}
        assert checker.check_event(event) == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 10: check_app with Glob Patterns
# ---------------------------------------------------------------------------


class TestCheckAppGlob:
    def test_wildcard_prefix(self) -> None:
        config = PrivacyZoneConfig(blocked=["*Password*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "SuperPassword Manager") == ObservationTier.BLOCKED

    def test_wildcard_suffix(self) -> None:
        config = PrivacyZoneConfig(blocked=["Vault*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "VaultManager") == ObservationTier.BLOCKED

    def test_question_mark_glob(self) -> None:
        config = PrivacyZoneConfig(blocked=["App?"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "App1") == ObservationTier.BLOCKED
        assert checker.check_app("", "AppXY") == ObservationTier.FULL

    def test_no_match(self) -> None:
        config = PrivacyZoneConfig(blocked=["SpecificApp"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "OtherApp") == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 11: check_url with Glob Patterns (case insensitive)
# ---------------------------------------------------------------------------


class TestCheckURLGlob:
    def test_case_insensitive_match(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://BANKING.Example.COM") == ObservationTier.BLOCKED

    def test_mixed_case_login(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://Login.MyApp.com/page") == ObservationTier.BLOCKED

    def test_custom_url_case_insensitive(self) -> None:
        config = PrivacyZoneConfig(blocked_urls=["*PRIVATE*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_url("https://example.com/private/docs") == ObservationTier.BLOCKED

    def test_empty_url(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("") == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 12: Auto-pause During Window
# ---------------------------------------------------------------------------


class TestAutoPauseDuringWindow:
    def test_paused_during_window(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        # Mock current time to be within the window
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(12, 0)
            assert checker.is_auto_paused() is True

    def test_event_blocked_during_auto_pause(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(12, 0)
            event = _make_event(app="Visual Studio Code")
            assert checker.check_event(event) == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 13: Auto-pause Outside Window
# ---------------------------------------------------------------------------


class TestAutoPauseOutsideWindow:
    def test_not_paused_outside_window(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(20, 0)
            assert checker.is_auto_paused() is False

    def test_event_allowed_outside_auto_pause(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(20, 0)
            event = _make_event(app="Visual Studio Code")
            assert checker.check_event(event) == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 14: Auto-pause Overnight Window
# ---------------------------------------------------------------------------


class TestAutoPauseOvernight:
    def test_paused_late_night(self) -> None:
        """22:00-06:00 — 23:00 should be paused."""
        config = PrivacyZoneConfig(auto_pause=["22:00-06:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(23, 0)
            assert checker.is_auto_paused() is True

    def test_paused_early_morning(self) -> None:
        """22:00-06:00 — 03:00 should be paused."""
        config = PrivacyZoneConfig(auto_pause=["22:00-06:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(3, 0)
            assert checker.is_auto_paused() is True

    def test_not_paused_midday(self) -> None:
        """22:00-06:00 — 12:00 should NOT be paused."""
        config = PrivacyZoneConfig(auto_pause=["22:00-06:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(12, 0)
            assert checker.is_auto_paused() is False

    def test_paused_at_midnight(self) -> None:
        """22:00-06:00 — 00:00 should be paused."""
        config = PrivacyZoneConfig(auto_pause=["22:00-06:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(0, 0)
            assert checker.is_auto_paused() is True


# ---------------------------------------------------------------------------
# Test 15: Multiple Auto-pause Windows
# ---------------------------------------------------------------------------


class TestMultipleAutoPauseWindows:
    def test_first_window_matches(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["12:00-13:00", "18:00-19:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(12, 30)
            assert checker.is_auto_paused() is True

    def test_second_window_matches(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["12:00-13:00", "18:00-19:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(18, 30)
            assert checker.is_auto_paused() is True

    def test_between_windows_not_paused(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["12:00-13:00", "18:00-19:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(15, 0)
            assert checker.is_auto_paused() is False


# ---------------------------------------------------------------------------
# Test 16: Empty Config
# ---------------------------------------------------------------------------


class TestEmptyConfig:
    def test_defaults_apply_with_empty_config(self) -> None:
        config = PrivacyZoneConfig()
        checker = PrivacyZoneChecker(config)
        # Default blocked apps
        assert checker.check_app("", "1Password") == ObservationTier.BLOCKED
        # Default blocked URLs
        assert checker.check_url("https://mybank.com/accounts") == ObservationTier.BLOCKED
        # Normal apps allowed
        assert checker.check_app("", "Safari") == ObservationTier.FULL

    def test_no_auto_pause_with_empty_config(self) -> None:
        config = PrivacyZoneConfig()
        checker = PrivacyZoneChecker(config)
        assert checker.is_auto_paused() is False


# ---------------------------------------------------------------------------
# Test 17: None Config
# ---------------------------------------------------------------------------


class TestNoneConfig:
    def test_defaults_apply_with_none_config(self) -> None:
        checker = PrivacyZoneChecker(None)
        assert checker.check_app("", "Bitwarden") == ObservationTier.BLOCKED
        assert checker.check_url("https://banking.example.com") == ObservationTier.BLOCKED
        assert checker.check_app("", "Finder") == ObservationTier.FULL

    def test_no_auto_pause_with_none_config(self) -> None:
        checker = PrivacyZoneChecker(None)
        assert checker.is_auto_paused() is False


# ---------------------------------------------------------------------------
# Test 18: Combined — App Allowed but URL Blocked
# ---------------------------------------------------------------------------


class TestCombinedAppAllowedURLBlocked:
    def test_most_restrictive_wins(self) -> None:
        """Safari is FULL but a banking URL should result in BLOCKED."""
        checker = PrivacyZoneChecker()
        event = _make_event(
            app="Safari",
            url="https://banking.wellsfargo.com/dashboard",
        )
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_app_metadata_only_url_blocked(self) -> None:
        """App is metadata_only but URL is blocked → BLOCKED wins."""
        config = PrivacyZoneConfig(metadata_only=["Chrome"])
        checker = PrivacyZoneChecker(config)
        event = _make_event(
            app="Chrome",
            url="https://login.example.com/auth",
        )
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_app_blocked_url_normal(self) -> None:
        """Blocked app with a normal URL → still BLOCKED."""
        checker = PrivacyZoneChecker()
        event = _make_event(
            app="1Password",
            url="https://docs.python.org",
        )
        assert checker.check_event(event) == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 19: Partial App Matches
# ---------------------------------------------------------------------------


class TestPartialAppMatches:
    def test_exact_match_required_without_glob(self) -> None:
        """Default blocked list uses exact names, not substrings."""
        checker = PrivacyZoneChecker()
        # "1Password Helper" does NOT exactly match "1Password"
        assert checker.check_app("", "1Password Helper") == ObservationTier.FULL

    def test_glob_wildcard_matches_substring(self) -> None:
        config = PrivacyZoneConfig(blocked=["*Password*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("", "1Password Helper") == ObservationTier.BLOCKED
        assert checker.check_app("", "MyPasswordManager") == ObservationTier.BLOCKED

    def test_default_blocked_exact_only(self) -> None:
        """'Bitwarden' should block but 'Bitwarden Desktop Helper' should not."""
        checker = PrivacyZoneChecker()
        assert checker.check_app("", "Bitwarden") == ObservationTier.BLOCKED
        assert checker.check_app("", "Bitwarden Desktop Helper") == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 20: Event with Missing window_json
# ---------------------------------------------------------------------------


class TestEventMissingWindowJSON:
    def test_no_window_json_returns_full(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "kind_json": '{"DwellSnapshot":{}}',
        }
        assert checker.check_event(event) == ObservationTier.FULL

    def test_empty_window_json_returns_full(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "kind_json": '{"DwellSnapshot":{}}',
            "window_json": "",
        }
        assert checker.check_event(event) == ObservationTier.FULL

    def test_invalid_window_json_returns_full(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "kind_json": '{"DwellSnapshot":{}}',
            "window_json": "not-valid-json",
        }
        assert checker.check_event(event) == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 21: Event with Missing metadata_json
# ---------------------------------------------------------------------------


class TestEventMissingMetadataJSON:
    def test_no_metadata_checks_app_only(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="1Password")
        # No metadata_json, but app is blocked
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_no_metadata_normal_app(self) -> None:
        checker = PrivacyZoneChecker()
        event = _make_event(app="Finder")
        assert checker.check_event(event) == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 22: ObservationTier Enum
# ---------------------------------------------------------------------------


class TestObservationTierEnum:
    def test_full_value(self) -> None:
        assert ObservationTier.FULL.value == "full"

    def test_metadata_only_value(self) -> None:
        assert ObservationTier.METADATA_ONLY.value == "metadata_only"

    def test_blocked_value(self) -> None:
        assert ObservationTier.BLOCKED.value == "blocked"

    def test_all_members(self) -> None:
        assert len(ObservationTier) == 3


# ---------------------------------------------------------------------------
# Test 23: PrivacyZoneConfig Defaults
# ---------------------------------------------------------------------------


class TestPrivacyZoneConfigDefaults:
    def test_all_lists_empty_by_default(self) -> None:
        config = PrivacyZoneConfig()
        assert config.full_observation == []
        assert config.metadata_only == []
        assert config.blocked == []
        assert config.blocked_urls == []
        assert config.auto_pause == []


# ---------------------------------------------------------------------------
# Test 24: Bundle ID Matching
# ---------------------------------------------------------------------------


class TestBundleIDMatching:
    def test_bundle_id_exact_blocked(self) -> None:
        config = PrivacyZoneConfig(blocked=["com.agilebits.onepassword7"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("com.agilebits.onepassword7", "") == ObservationTier.BLOCKED

    def test_bundle_id_glob_blocked(self) -> None:
        config = PrivacyZoneConfig(blocked=["com.agilebits.*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("com.agilebits.onepassword7", "") == ObservationTier.BLOCKED

    def test_bundle_id_in_event(self) -> None:
        config = PrivacyZoneConfig(blocked=["com.secret.*"])
        checker = PrivacyZoneChecker(config)
        event = _make_event(bundle_id="com.secret.app")
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_bundle_id_not_matching(self) -> None:
        config = PrivacyZoneConfig(blocked=["com.secret.*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_app("com.apple.safari", "") == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 25: URL Case Insensitivity
# ---------------------------------------------------------------------------


class TestURLCaseInsensitivity:
    def test_uppercase_bank_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("HTTPS://BANK.EXAMPLE.COM") == ObservationTier.BLOCKED

    def test_mixed_case_payroll_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        assert checker.check_url("https://Payroll.Company.com") == ObservationTier.BLOCKED

    def test_custom_pattern_case_insensitive(self) -> None:
        config = PrivacyZoneConfig(blocked_urls=["*SENSITIVE*"])
        checker = PrivacyZoneChecker(config)
        assert checker.check_url("https://example.com/sensitive/data") == ObservationTier.BLOCKED
        assert checker.check_url("https://example.com/SENSITIVE/DATA") == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 26: Time Window Parsing
# ---------------------------------------------------------------------------


class TestTimeWindowParsing:
    def test_valid_window_parsed(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        assert len(checker._auto_pause_windows) == 1
        assert checker._auto_pause_windows[0] == (dt_time(9, 0), dt_time(17, 0))

    def test_midnight_window_parsed(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["00:00-23:59"])
        checker = PrivacyZoneChecker(config)
        assert len(checker._auto_pause_windows) == 1
        assert checker._auto_pause_windows[0] == (dt_time(0, 0), dt_time(23, 59))

    def test_invalid_window_skipped(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["invalid", "09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        # Only the valid one should be parsed
        assert len(checker._auto_pause_windows) == 1

    def test_empty_string_skipped(self) -> None:
        config = PrivacyZoneConfig(auto_pause=[""])
        checker = PrivacyZoneChecker(config)
        assert len(checker._auto_pause_windows) == 0

    def test_bad_hour_skipped(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["25:00-06:00"])
        checker = PrivacyZoneChecker(config)
        assert len(checker._auto_pause_windows) == 0


# ---------------------------------------------------------------------------
# Test 27: Auto-pause Boundary Times
# ---------------------------------------------------------------------------


class TestAutoPauseBoundaryTimes:
    def test_exactly_at_start(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(9, 0)
            assert checker.is_auto_paused() is True

    def test_exactly_at_end(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(17, 0)
            assert checker.is_auto_paused() is True

    def test_one_minute_before_start(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(8, 59)
            assert checker.is_auto_paused() is False

    def test_one_minute_after_end(self) -> None:
        config = PrivacyZoneConfig(auto_pause=["09:00-17:00"])
        checker = PrivacyZoneChecker(config)
        with patch("agenthandover_worker.privacy_zones.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(17, 1)
            assert checker.is_auto_paused() is False


# ---------------------------------------------------------------------------
# Test 28: Most Restrictive Tier
# ---------------------------------------------------------------------------


class TestMostRestrictiveTier:
    def test_blocked_beats_full(self) -> None:
        result = PrivacyZoneChecker._most_restrictive(
            ObservationTier.FULL, ObservationTier.BLOCKED
        )
        assert result == ObservationTier.BLOCKED

    def test_blocked_beats_metadata_only(self) -> None:
        result = PrivacyZoneChecker._most_restrictive(
            ObservationTier.METADATA_ONLY, ObservationTier.BLOCKED
        )
        assert result == ObservationTier.BLOCKED

    def test_metadata_only_beats_full(self) -> None:
        result = PrivacyZoneChecker._most_restrictive(
            ObservationTier.FULL, ObservationTier.METADATA_ONLY
        )
        assert result == ObservationTier.METADATA_ONLY

    def test_same_tier_returns_same(self) -> None:
        result = PrivacyZoneChecker._most_restrictive(
            ObservationTier.FULL, ObservationTier.FULL
        )
        assert result == ObservationTier.FULL

    def test_order_does_not_matter(self) -> None:
        r1 = PrivacyZoneChecker._most_restrictive(
            ObservationTier.FULL, ObservationTier.BLOCKED
        )
        r2 = PrivacyZoneChecker._most_restrictive(
            ObservationTier.BLOCKED, ObservationTier.FULL
        )
        assert r1 == r2 == ObservationTier.BLOCKED


# ---------------------------------------------------------------------------
# Test 29: Event with dict window_json (not string)
# ---------------------------------------------------------------------------


class TestEventWithDictWindowJSON:
    def test_dict_window_json_parsed(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "window_json": {"app": "1Password", "title": "Vault"},
        }
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_dict_window_json_normal_app(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "window_json": {"app": "Finder", "title": "Documents"},
        }
        assert checker.check_event(event) == ObservationTier.FULL


# ---------------------------------------------------------------------------
# Test 30: Event with dict metadata_json (not string)
# ---------------------------------------------------------------------------


class TestEventWithDictMetadataJSON:
    def test_dict_metadata_json_url_blocked(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "window_json": json.dumps({"app": "Safari"}),
            "metadata_json": {"url": "https://login.example.com"},
        }
        assert checker.check_event(event) == ObservationTier.BLOCKED

    def test_dict_metadata_json_url_allowed(self) -> None:
        checker = PrivacyZoneChecker()
        event = {
            "id": "test",
            "timestamp": "2026-03-11T10:00:00Z",
            "window_json": json.dumps({"app": "Safari"}),
            "metadata_json": {"url": "https://github.com"},
        }
        assert checker.check_event(event) == ObservationTier.FULL

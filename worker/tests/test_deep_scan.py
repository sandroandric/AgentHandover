"""Tests for the Tier 2 deep scan module.

Covers PII pattern detection (email, phone, SSN, credit card, ZIP, IP),
Luhn validation, artifact batch scanning, and edge cases.
"""

from __future__ import annotations

import re

from agenthandover_worker.deep_scan import (
    DeepScanner,
    DeepScanResult,
    PIIMatch,
    _luhn_check,
)


# ------------------------------------------------------------------
# 1. Email detection
# ------------------------------------------------------------------


class TestEmailDetection:
    def test_detects_email(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("Contact us at user@example.com for info.")
        emails = [m for m in matches if m.pattern_name == "email"]
        assert len(emails) == 1
        assert emails[0].matched_text == "user@example.com"

    def test_no_false_positive_on_non_email(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("This is a regular sentence.")
        emails = [m for m in matches if m.pattern_name == "email"]
        assert len(emails) == 0


# ------------------------------------------------------------------
# 2. Phone number detection
# ------------------------------------------------------------------


class TestPhoneDetection:
    def test_detects_us_phone(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("Call (555) 123-4567 for support.")
        phones = [m for m in matches if m.pattern_name == "phone_us"]
        assert len(phones) >= 1

    def test_detects_dashed_phone(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("Phone: 555-123-4567")
        phones = [m for m in matches if m.pattern_name == "phone_us"]
        assert len(phones) >= 1


# ------------------------------------------------------------------
# 3. SSN detection
# ------------------------------------------------------------------


class TestSSNDetection:
    def test_detects_ssn(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("SSN: 123-45-6789")
        ssns = [m for m in matches if m.pattern_name == "ssn"]
        assert len(ssns) >= 1

    def test_filters_invalid_ssn_area(self) -> None:
        scanner = DeepScanner()
        # 000 prefix is invalid for SSN
        matches = scanner.scan_text("Number: 000-12-3456")
        ssns = [m for m in matches if m.pattern_name == "ssn"]
        assert len(ssns) == 0

    def test_filters_666_ssn_area(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("Number: 666-12-3456")
        ssns = [m for m in matches if m.pattern_name == "ssn"]
        assert len(ssns) == 0


# ------------------------------------------------------------------
# 4. Credit card detection with Luhn
# ------------------------------------------------------------------


class TestCreditCardDetection:
    def test_detects_valid_visa(self) -> None:
        scanner = DeepScanner()
        # Valid Visa test number
        matches = scanner.scan_text("Card: 4111 1111 1111 1111")
        cards = [m for m in matches if m.pattern_name == "credit_card"]
        assert len(cards) >= 1

    def test_rejects_invalid_luhn(self) -> None:
        scanner = DeepScanner()
        # Invalid Luhn number
        matches = scanner.scan_text("Card: 1234 5678 9012 3456")
        cards = [m for m in matches if m.pattern_name == "credit_card"]
        assert len(cards) == 0


# ------------------------------------------------------------------
# 5. Luhn check function
# ------------------------------------------------------------------


class TestLuhnCheck:
    def test_valid_visa(self) -> None:
        assert _luhn_check("4111111111111111") is True

    def test_valid_mastercard(self) -> None:
        assert _luhn_check("5500000000000004") is True

    def test_invalid_number(self) -> None:
        assert _luhn_check("1234567890123456") is False

    def test_too_short(self) -> None:
        assert _luhn_check("123456") is False


# ------------------------------------------------------------------
# 6. IPv4 detection
# ------------------------------------------------------------------


class TestIPv4Detection:
    def test_detects_ipv4(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("Server at 192.168.1.100 is down.")
        ips = [m for m in matches if m.pattern_name == "ipv4"]
        assert len(ips) >= 1
        assert ips[0].matched_text == "192.168.1.100"


# ------------------------------------------------------------------
# 7. Batch artifact scanning
# ------------------------------------------------------------------


class TestBatchScanning:
    def test_scans_multiple_artifacts(self) -> None:
        scanner = DeepScanner()
        artifacts = [
            {"id": "art-1", "text": "Email: user@test.com"},
            {"id": "art-2", "text": "No PII here at all."},
            {"id": "art-3", "text": "IP: 10.0.0.1"},
        ]

        result = scanner.scan_artifacts(artifacts)

        assert result.artifacts_scanned == 3
        assert result.has_pii is True
        assert result.total_pii >= 2

    def test_empty_artifacts(self) -> None:
        scanner = DeepScanner()
        result = scanner.scan_artifacts([])

        assert result.artifacts_scanned == 0
        assert result.has_pii is False
        assert result.total_pii == 0


# ------------------------------------------------------------------
# 8. DeepScanResult properties
# ------------------------------------------------------------------


class TestDeepScanResult:
    def test_result_has_pii(self) -> None:
        result = DeepScanResult(
            artifacts_scanned=1,
            pii_found=[PIIMatch(
                pattern_name="email",
                matched_text="x@y.com",
                start_pos=0,
                end_pos=7,
            )],
        )
        assert result.has_pii is True
        assert result.total_pii == 1

    def test_result_no_pii(self) -> None:
        result = DeepScanResult(artifacts_scanned=5)
        assert result.has_pii is False
        assert result.total_pii == 0


# ------------------------------------------------------------------
# 9. Custom patterns
# ------------------------------------------------------------------


class TestCustomPatterns:
    def test_custom_pattern_detected(self) -> None:
        custom = [("api_key", re.compile(r"sk-[a-zA-Z0-9]{20,}"))]
        scanner = DeepScanner(extra_patterns=custom)
        matches = scanner.scan_text("Key: sk-abcdefghijklmnopqrstuvwxyz")
        api_keys = [m for m in matches if m.pattern_name == "api_key"]
        assert len(api_keys) == 1


# ------------------------------------------------------------------
# 10. Empty text
# ------------------------------------------------------------------


class TestEmptyText:
    def test_empty_text_returns_no_matches(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("")
        assert len(matches) == 0

    def test_none_artifact_text_skipped(self) -> None:
        scanner = DeepScanner()
        result = scanner.scan_artifacts([{"id": "x", "text": ""}])
        assert result.artifacts_scanned == 0


# ------------------------------------------------------------------
# 11. Artifact ID propagation
# ------------------------------------------------------------------


class TestArtifactIdPropagation:
    def test_artifact_id_on_matches(self) -> None:
        scanner = DeepScanner()
        matches = scanner.scan_text("user@test.com", artifact_id="art-42")
        assert len(matches) >= 1
        assert matches[0].artifact_id == "art-42"

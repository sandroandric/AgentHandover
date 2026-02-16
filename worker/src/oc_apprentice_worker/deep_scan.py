"""Tier 2 Idle Deep Scan — scan stored artifacts for missed PII patterns.

During system idle time, this module performs a secondary pass over stored
DOM and accessibility text artifacts looking for PII patterns that the
real-time pipeline may have missed.  This provides defense-in-depth for
privacy protection.

Detected patterns:
- Email addresses
- Phone numbers (US formats)
- Social Security Numbers (SSN)
- Credit card numbers (Luhn-validated)
- Mailing addresses (US ZIP code patterns)
- IP addresses (IPv4)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """A single PII detection in scanned text."""

    pattern_name: str
    matched_text: str
    start_pos: int
    end_pos: int
    artifact_id: str = ""


@dataclass
class DeepScanResult:
    """Result of a deep scan over a batch of artifacts."""

    artifacts_scanned: int = 0
    pii_found: list[PIIMatch] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_pii(self) -> int:
        return len(self.pii_found)

    @property
    def has_pii(self) -> bool:
        return self.total_pii > 0


# PII detection patterns
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )),
    ("phone_us", re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )),
    ("ssn", re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    )),
    ("credit_card", re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    )),
    ("us_zip", re.compile(
        r"\b\d{5}(?:-\d{4})?\b"
    )),
    ("ipv4", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    )),
]


def _luhn_check(number_str: str) -> bool:
    """Validate a number string using the Luhn algorithm."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class DeepScanner:
    """Scan stored text artifacts for missed PII patterns.

    Designed to run during system idle time as a Tier 2 privacy check.
    The scanner does not modify artifacts; it reports findings for
    upstream remediation.

    Parameters
    ----------
    extra_patterns:
        Additional (name, compiled_regex) pairs to scan for.
    """

    def __init__(
        self,
        extra_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    ) -> None:
        self._patterns = list(_PII_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def scan_text(self, text: str, artifact_id: str = "") -> list[PIIMatch]:
        """Scan a single text string for PII patterns.

        Parameters
        ----------
        text:
            The text content to scan.
        artifact_id:
            Optional artifact identifier for tracking.

        Returns
        -------
        list[PIIMatch]
            All PII matches found in the text.
        """
        if not text:
            return []

        matches: list[PIIMatch] = []

        for pattern_name, regex in self._patterns:
            for m in regex.finditer(text):
                matched_text = m.group()

                # For credit card matches, validate with Luhn
                if pattern_name == "credit_card":
                    digits_only = re.sub(r"[-\s]", "", matched_text)
                    if not _luhn_check(digits_only):
                        continue

                # For SSN, filter out obvious non-SSN patterns
                if pattern_name == "ssn":
                    digits_only = re.sub(r"[-\s]", "", matched_text)
                    # SSNs don't start with 000, 666, or 900-999
                    area = int(digits_only[:3])
                    if area == 0 or area == 666 or area >= 900:
                        continue
                    # Group (digits 4-5) cannot be 00
                    group = int(digits_only[3:5])
                    if group == 0:
                        continue
                    # Serial (digits 6-9) cannot be 0000
                    serial = int(digits_only[5:9])
                    if serial == 0:
                        continue

                matches.append(PIIMatch(
                    pattern_name=pattern_name,
                    matched_text=matched_text,
                    start_pos=m.start(),
                    end_pos=m.end(),
                    artifact_id=artifact_id,
                ))

        return matches

    def scan_artifacts(
        self,
        artifacts: list[dict],
    ) -> DeepScanResult:
        """Scan a batch of text artifacts for PII.

        Each artifact dict should contain:
        - ``id``: artifact identifier
        - ``text``: the text content to scan

        Parameters
        ----------
        artifacts:
            List of artifact dicts to scan.

        Returns
        -------
        DeepScanResult
            Summary of all PII found across all artifacts.
        """
        result = DeepScanResult()

        for artifact in artifacts:
            artifact_id = artifact.get("id", "unknown")
            text = artifact.get("text", "")

            if not text:
                continue

            result.artifacts_scanned += 1

            try:
                matches = self.scan_text(text, artifact_id=artifact_id)
                result.pii_found.extend(matches)
            except Exception as exc:
                error_msg = f"Error scanning artifact {artifact_id}: {exc}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        if result.has_pii:
            logger.warning(
                "Deep scan found %d PII match(es) across %d artifact(s)",
                result.total_pii,
                result.artifacts_scanned,
            )
        else:
            logger.debug(
                "Deep scan clean: %d artifact(s) scanned, no PII found",
                result.artifacts_scanned,
            )

        return result

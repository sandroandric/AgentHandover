"""Secret and personal-data redaction for text the worker stores.

Mirrors the daemon's ``Redactor`` (``crates/common/src/redaction.rs``):
the same patterns, replacement tokens, Luhn-checked card numbers and
long hex strings, applied in the same order, plus email addresses and
phone numbers.  The worker runs it over every model-written annotation
before the annotation is stored (``scene_annotation_json``).
"""

from __future__ import annotations

import re

# (pattern, replacement), in the daemon's order.  As there, the whole
# match is replaced (the labelled-password rule keeps its label).
_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?i)(AKIA[0-9A-Z]{16})"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)(?:aws_secret_access_key|secret_key)\s*[=:]\s*([A-Za-z0-9/+=]{30,})"),
     "[REDACTED_SECRET]"),
    (re.compile(
        r"(?i)(?:api[_-]?key|api[_-]?token|auth[_-]?token|bearer)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?"
    ), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(\d{3}-\d{2}-\d{4})\b"), "[REDACTED_SSN]"),
    (re.compile(
        r"(?s)(-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE KEY-----.*?"
        r"-----END\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE KEY-----)"
    ), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"(gh[pousr]_[A-Za-z0-9]{36,})"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(github_pat_[A-Za-z0-9_]{22,})"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(xox[bpors]-[A-Za-z0-9\-]{10,})"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"(sk-ant-[A-Za-z0-9_\-]{20,})"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"(sk-proj-[A-Za-z0-9_\-]{20,})"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\b([sr]k_(?:live|test)_[A-Za-z0-9]{16,})"), "[REDACTED_STRIPE_KEY]"),
    (re.compile(r"(AIza[0-9A-Za-z_\-]{35})"), "[REDACTED_GOOGLE_API_KEY]"),
    (re.compile(r"(glpat-[A-Za-z0-9_\-]{20,})"), "[REDACTED_GITLAB_TOKEN]"),
    (re.compile(r"\b(hf_[A-Za-z0-9]{30,})"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"\b(eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})"),
     "[REDACTED_JWT]"),
    (re.compile(r"(?i)\b((?:password|passwd|passphrase|pwd)[ \t]*[=:][ \t]*)['\"]?[^\s'\"&,;]+['\"]?"),
     r"\1[REDACTED_PASSWORD]"),
)

_CC = re.compile(r"\b([3-6]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4})\b")
_HIGH_ENTROPY = re.compile(r"\b([a-f0-9]{80,})\b")

# Worker additions (the daemon has none for these).
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# A phone number: optional +country code and (area code), then two to
# five groups of 2-4 digits split by a space, dot or dash.  A match is
# redacted only with 9-15 digits (E.164), and never next to ``:`` or
# ``/``, so dates, times, counts, versions and URL paths survive.
_PHONE = re.compile(
    r"(?<![\w+:/])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,4}){1,4}(?![\w:/])"
)
_PHONE_SEPARATOR = re.compile(r"[\s.()-]")


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _phone(match: re.Match) -> str:
    text = match.group(0)
    digits = sum(c.isdigit() for c in text)
    # A phone number has separators or a +; a bare digit run is an id or
    # a count.  ISO dates (2026-09-22) have 8 digits and stay.
    if 9 <= digits <= 15 and (text.startswith("+") or _PHONE_SEPARATOR.search(text)):
        return "[REDACTED_PHONE]"
    return text


def redact(text: str) -> str:
    """*text* with secrets, card numbers, email addresses and phone numbers
    replaced by ``[REDACTED_*]`` tokens."""
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _CC.sub(lambda m: "[REDACTED_CC]" if _luhn(m.group(0)) else m.group(0), text)
    text = _HIGH_ENTROPY.sub("[REDACTED_HIGH_ENTROPY]", text)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    return _PHONE.sub(_phone, text)


def redact_value(value):
    """*value* (a JSON-like structure) with every string redacted; dict
    keys are left as they are."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value

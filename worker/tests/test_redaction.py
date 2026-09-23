"""Regression pins for the worker's redaction (coverage audit #8): the
daemon's Redactor patterns (crates/common/src/redaction.rs) plus email
addresses and phone numbers."""

from __future__ import annotations

import pytest

from agenthandover_worker.redaction import redact, redact_value


@pytest.mark.parametrize(("text", "token"), [
    # The daemon's patterns, same tokens.
    ("key AKIAIOSFODNN7EXAMPLE here", "[REDACTED_AWS_KEY]"),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "[REDACTED_SECRET]"),
    ("api_key=abcdefghijklmnopqrstuvwxyz123", "[REDACTED_API_KEY]"),
    ("Authorization: Bearer: abcdefghijklmnopqrstuvwx", "[REDACTED_API_KEY]"),
    ("SSN 123-45-6789", "[REDACTED_SSN]"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    ("ghp_" + "a" * 36, "[REDACTED_GITHUB_TOKEN]"),
    ("github_pat_" + "B" * 22, "[REDACTED_GITHUB_TOKEN]"),
    ("xoxb-1234567890-abcdef", "[REDACTED_SLACK_TOKEN]"),
    ("sk-ant-" + "c" * 24, "[REDACTED_ANTHROPIC_KEY]"),
    ("sk-proj-" + "d" * 24, "[REDACTED_OPENAI_KEY]"),
    ("sk_live_" + "e" * 24, "[REDACTED_STRIPE_KEY]"),
    ("AIza" + "f" * 35, "[REDACTED_GOOGLE_API_KEY]"),
    ("glpat-" + "g" * 20, "[REDACTED_GITLAB_TOKEN]"),
    ("hf_" + "h" * 30, "[REDACTED_HF_TOKEN]"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
     "[REDACTED_JWT]"),
    ("card 4111 1111 1111 1111", "[REDACTED_CC]"),
    ("blob " + "ab12" * 21, "[REDACTED_HIGH_ENTROPY]"),
    # Worker additions.
    ("mail jane.doe@example.com today", "[REDACTED_EMAIL]"),
    ("Call +1 415 555 0100", "[REDACTED_PHONE]"),
    ("(415) 555-0100", "[REDACTED_PHONE]"),
    ("+44 20 7946 0958", "[REDACTED_PHONE]"),
])
def test_sensitive_text_is_replaced(text, token):
    out = redact(text)
    assert token in out
    assert out != text


def test_labelled_password_keeps_its_label():
    assert redact("password: hunter2 ok") == "password: [REDACTED_PASSWORD] ok"


@pytest.mark.parametrize("text", [
    "530 points | 47 comments",
    "2026-09-15 08:30 meeting",
    "v0.34.2 at 192.168.1.100",
    "order 12345678901",
    "card 4111 1111 1111 1112",  # fails the Luhn check
    "https://news.ycombinator.com/item?id=41234567",
    "Daily news - 15 Sep",
    "sha256 " + "a" * 64,
])
def test_ordinary_text_is_left_alone(text):
    assert redact(text) == text


def test_redact_value_walks_nested_structures_and_keeps_keys():
    value = {
        "key_text": {"emails": ["jane@example.com"], "typed_text": "api_key=" + "x" * 24},
        "count": 3,
        "flag": True,
        "jane@example.com": "kept key",
    }
    out = redact_value(value)
    assert out["key_text"]["emails"] == ["[REDACTED_EMAIL]"]
    assert out["key_text"]["typed_text"] == "[REDACTED_API_KEY]"
    assert out["count"] == 3 and out["flag"] is True
    assert "jane@example.com" in out

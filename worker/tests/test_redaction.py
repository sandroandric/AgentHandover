"""Regression pins for the worker's redaction (coverage audit #8): the
daemon's Redactor patterns (crates/common/src/redaction.rs), the password
phrasing rules, URL reduction, email addresses and phone numbers."""

from __future__ import annotations

import pytest

from agenthandover_worker.redaction import redact, redact_value, strip_url, strip_urls


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


# --- Passwords (WP-F1) ------------------------------------------------------

@pytest.mark.parametrize(("text", "expected"), [
    ("my password is hunter22", "my password is [REDACTED_PASSWORD]"),
    ("The staging GitHub password is Tr0ub4dor&3-horse-battery",
     "The staging GitHub password is [REDACTED_PASSWORD]"),
    ("pw: abc", "pw: [REDACTED_PASSWORD]"),
    ("PW=abc", "PW=[REDACTED_PASSWORD]"),
    ("passcode 482913", "passcode [REDACTED_PIN]"),
    ("passcode Zq9-x", "passcode [REDACTED_PASSWORD]"),
    ("my password's S3cret!x", "my password's [REDACTED_PASSWORD]"),
    ("my password’s S3cret", "my password’s [REDACTED_PASSWORD]"),
    ("PIN 1234", "PIN [REDACTED_PIN]"),
    ("Your pin code is 90210.", "Your pin code is [REDACTED_PIN]."),
    ("passcode: 4829", "passcode: [REDACTED_PASSWORD]"),  # labelled
    ("the wifi password was CorrectHorse, ok", "the wifi password was [REDACTED_PASSWORD], ok"),
    ('password is "hunter22".', 'password is "[REDACTED_PASSWORD]".'),
    # A password manager's two-column layout.
    ("password  Tr0ub4dor&3-horse-battery", "password  [REDACTED_PASSWORD]"),
    # ``:`` keeps an ``&`` inside the value; ``=`` stops there (a query).
    ("password: Tr0ub4dor&3 ok", "password: [REDACTED_PASSWORD] ok"),
    ("password=hunter2&x=1", "password=[REDACTED_PASSWORD]&x=1"),
])
def test_password_phrasing_is_redacted(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize("text", [
    "reset your password",
    "Reset your password",
    "Forgot password?",
    "Password is required.",
    "Your password is incorrect",
    "New password\nConfirm new password",
    "Password (optional)",
    "Change your passcode settings",
    "password reset link sent",
    "Enter the PIN you received",  # a PIN only as digits after the word
    "pinned 1234 items",
    "Passwords",
])
def test_prose_about_passwords_is_left_alone(text):
    assert redact(text) == text


def test_redaction_is_idempotent():
    text = (
        "my password is hunter22, PIN 1234, see "
        "https://x.com/reset?token=abc123&q=cats#top and jane@example.com"
    )
    once = redact(text)
    assert "hunter22" not in once and "1234" not in once and "abc123" not in once
    assert redact(once) == once


# --- URLs (WP-F1) -------------------------------------------------------------

@pytest.mark.parametrize(("url", "expected"), [
    ("https://x.com/reset?token=abc123", "https://x.com/reset"),
    ("https://x.com/a?q=cats&token=1&page=2#frag", "https://x.com/a?q=cats&page=2"),
    ("https://x.com/list?sort=new&view=grid&tab=all&id=7",
     "https://x.com/list?sort=new&view=grid&tab=all&id=7"),
    ("https://x.com/cb?code=1&state=2&sig=3&key=4&auth_token=5&authuser=0",
     "https://x.com/cb"),
    ("https://user:pw@host.example:8443/p?x=1", "https://host.example:8443/p"),
    ("accounts.example.com/reset?token=8f3a9c2e71b04d55a0c4", "accounts.example.com/reset"),
    ("mail.google.com/mail/u/0/#inbox", "mail.google.com/mail/u/0/"),
    ("https://example.com", "https://example.com"),
    ("https://example.com/path/only", "https://example.com/path/only"),
])
def test_strip_url(url, expected):
    assert strip_url(url) == expected


def test_urls_inside_text_are_stripped_and_prose_kept():
    text = (
        "Open accounts.example.com/reset?token=8f3a9c2e71b04d55a0c4 now "
        "(or https://x.com/a?code=9). Did you edit config.toml? "
        "See https://news.ycombinator.com/item?id=41234567."
    )
    assert strip_urls(text) == (
        "Open accounts.example.com/reset now "
        "(or https://x.com/a). Did you edit config.toml? "
        "See https://news.ycombinator.com/item?id=41234567."
    )


def test_redact_value_strips_stored_urls():
    ann = {
        "location": "https://x.com/reset?token=abc123",
        "key_text": {"urls": ["x.com/reset?token=abc123", "https://a.io/?q=1#f"]},
        "_screen_text": "x.com/reset?token=abc123\nReset your password",
    }
    out = redact_value(ann)
    assert "abc123" not in str(out)
    assert out["location"] == "https://x.com/reset"
    assert out["key_text"]["urls"] == ["x.com/reset", "https://a.io/?q=1"]
    assert out["_screen_text"] == "x.com/reset\nReset your password"

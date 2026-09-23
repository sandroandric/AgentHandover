"""Secret and personal-data redaction for text the worker stores or puts
in a prompt.

Mirrors the daemon's ``Redactor`` (``crates/common/src/redaction.rs``):
the same patterns, replacement tokens, Luhn-checked card numbers and
long hex strings, applied in the same order (the password rules
included), plus the worker's additions: URLs reduced by
:func:`strip_url`, email addresses and phone numbers.

The worker runs :func:`redact` over the screen text before it enters a
model prompt (OCR, page elements, the window header, the evidence blocks
of the diff, SOP, synthesis and focus Q&A prompts) and :func:`redact_value`
over every model-written annotation before it is stored.
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
)

# Passwords.  A value is dropped, its label kept, when it follows
#   - a label and ``=`` (``pwd=x``, stopping at ``&`` as in a query) or
#     ``:`` (``password: x``, ``pw: x``);
#   - a PIN label (``PIN 1234``, ``passcode: 482913``): 4-8 digits;
#   - ``password is x``, ``my password's x``, or the bare ``passcode x``
#     (a password manager's ``password  x`` column): only when the value
#     looks like a secret (a digit, a symbol inside it, or an inner capital
#     letter), so prose that mentions passwords ("reset your password",
#     "password is required") is left alone.
_PW_LABEL = r"(?:pass(?:word|wd|phrase|code)|pwd|pw)"
_PW_TOKEN = "[REDACTED_PASSWORD]"
_PW_EQUALS = re.compile(rf"(?i)\b({_PW_LABEL}[ \t]*=[ \t]*)['\"]?[^\s'\"&,;]+['\"]?")
_PW_COLON = re.compile(rf"(?i)\b({_PW_LABEL}[ \t]*:[ \t]*)['\"]?[^\s'\",;]+['\"]?")
_PIN = re.compile(
    r"(?i)\b((?:pin|passcode)(?:[ \t]+(?:code|number))?(?:[ \t]+is)?[ \t]*[:=#]?[ \t]*)\d{4,8}\b"
)
_PW_PHRASE = re.compile(
    rf"(?i)\b({_PW_LABEL}(?:['\u2019]s|[ \t]+(?:is|was))?:?[ \t]+)(\S+)"
)
_WRAP_OPEN = "([{<'\"\u201c\u2018"
_WRAP_CLOSE = ".,;:)]}>'\"\u201d\u2019"
_SECRET_LIKE = re.compile(r"\d|[^\w]|[a-z][A-Z]")


def _pw_phrase(match: re.Match) -> str:
    value = match.group(2)
    core = value.lstrip(_WRAP_OPEN)
    lead = value[: len(value) - len(core)]
    stripped = core.rstrip(_WRAP_CLOSE)
    trail = core[len(stripped):]
    if (
        len(stripped) < 4
        or stripped.startswith("[REDACTED")
        or not _SECRET_LIKE.search(stripped)
    ):
        return match.group(0)
    return f"{match.group(1)}{lead}{_PW_TOKEN}{trail}"


def _redact_passwords(text: str) -> str:
    text = _PW_EQUALS.sub(rf"\1{_PW_TOKEN}", text)
    text = _PW_COLON.sub(rf"\1{_PW_TOKEN}", text)
    text = _PIN.sub(r"\1[REDACTED_PIN]", text)
    return _PW_PHRASE.sub(_pw_phrase, text)


# URLs.  Query keys that carry a page's harmless state; every other key
# (token, code, key, sig, auth*, session, ...) is dropped with its value.
URL_QUERY_KEEP = frozenset({"q", "page", "tab", "view", "id", "sort"})
# A URL with a scheme, or a bare host with a path/query/fragment (the
# address bar shows ``example.com/reset?token=...`` without ``https://``).
_URL = re.compile(
    r"(?i)(?<![\w@.-])(?:[a-z][a-z0-9+.-]*://[^\s<>\"'`]+"
    r"|(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?[/?#][^\s<>\"'`]*)"
)
_URL_TRAILING = ".,;:!?)]}'\u201d\u2019"

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


def strip_url(url: str) -> str:
    """*url* reduced to scheme, host, port and path, plus the query keys in
    :data:`URL_QUERY_KEEP`.  The fragment and any ``user:password@`` are
    dropped.  Works on bare ``host/path`` too.

    The worker's twin of the daemon's ``strip_url_to_origin_and_path``
    (which drops the whole query), used for every URL the worker stores or
    puts in a prompt.
    """
    base, _, _fragment = url.partition("#")
    base, has_query, query = base.partition("?")
    scheme, sep, rest = base.partition("://")
    if sep:
        authority, slash, path = rest.partition("/")
        base = f"{scheme}://{authority.rpartition('@')[2]}{slash}{path}"
    if not has_query:
        return base
    kept = [
        pair for pair in query.split("&")
        if pair.partition("=")[0].lower() in URL_QUERY_KEEP
    ]
    return f"{base}?{'&'.join(kept)}" if kept else base


def _url(match: re.Match) -> str:
    text = match.group(0)
    core = text.rstrip(_URL_TRAILING)
    if "?" not in core and "#" not in core and "@" not in core:
        return text
    return strip_url(core) + text[len(core):]


def strip_urls(text: str) -> str:
    """*text* with every URL-looking string in it reduced by
    :func:`strip_url`."""
    return _URL.sub(_url, text) if text else text


def redact(text: str) -> str:
    """*text* with URLs reduced (:func:`strip_url`) and secrets, passwords,
    card numbers, email addresses and phone numbers replaced by
    ``[REDACTED_*]`` tokens."""
    if not text:
        return text
    text = strip_urls(text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _redact_passwords(text)
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

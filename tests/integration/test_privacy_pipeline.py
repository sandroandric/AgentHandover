"""Privacy Pipeline Integration Tests.

Verifies that secrets are properly redacted through the event pipeline.
Seeds mock events with known secret values and asserts that none of the
raw secrets appear in the output after passing through the Redactor.

The Redactor patterns here mirror the Rust implementation at
crates/common/src/redaction.rs — same regex patterns, same replacement
tokens. This ensures the Python pipeline can independently verify that
all secret categories are scrubbed.

Test categories:
  - AWS access keys
  - AWS secret keys
  - GitHub tokens
  - Credit card numbers (with dashes and without)
  - Social Security Numbers
  - PEM private keys
  - High-entropy hex strings
  - Bearer tokens
  - Slack tokens
  - Generic API keys in key=value format
  - Multiple secrets in a single event
  - Secrets in metadata_json values
  - Secrets across different event fields
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Add worker src to Python path
# ---------------------------------------------------------------------------

WORKER_SRC = Path(__file__).resolve().parent.parent.parent / "worker" / "src"
sys.path.insert(0, str(WORKER_SRC))

from agenthandover_worker.db import WorkerDB

# ---------------------------------------------------------------------------
# Python Redactor — mirrors crates/common/src/redaction.rs
# ---------------------------------------------------------------------------


class Redactor:
    """Python port of the Rust Redactor for integration testing.

    Patterns match the Rust implementation exactly so that test results
    are consistent with the daemon's runtime behaviour.
    """

    def __init__(self) -> None:
        self._patterns: list[tuple[re.Pattern[str], str]] = [
            # AWS Access Key ID (starts with AKIA)
            (re.compile(r"(?i)(AKIA[0-9A-Z]{16})"), "[REDACTED_AWS_KEY]"),
            # AWS Secret Access Key (40 char base64-ish after = or :)
            (
                re.compile(
                    r"(?i)(?:aws_secret_access_key|secret_key)\s*[=:]\s*"
                    r"([A-Za-z0-9/+=]{30,})"
                ),
                "[REDACTED_SECRET]",
            ),
            # Generic API keys/tokens (long alphanumeric after common key words)
            (
                re.compile(
                    r"(?i)(?:api[_-]?key|api[_-]?token|auth[_-]?token|bearer)"
                    r"""\s*[=:]\s*['"]?([A-Za-z0-9_\-]{20,})['"]?"""
                ),
                "[REDACTED_API_KEY]",
            ),
            # Credit card numbers (Visa, MC, Amex, Discover with optional dashes/spaces)
            (
                re.compile(r"\b([3-6]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4})\b"),
                "[REDACTED_CC]",
            ),
            # SSN
            (re.compile(r"\b(\d{3}-\d{2}-\d{4})\b"), "[REDACTED_SSN]"),
            # Private keys (PEM format)
            (
                re.compile(
                    r"(?s)(-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE KEY-----"
                    r".*?"
                    r"-----END\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE KEY-----)"
                ),
                "[REDACTED_PRIVATE_KEY]",
            ),
            # GitHub tokens
            (re.compile(r"(ghp_[A-Za-z0-9]{36,})"), "[REDACTED_GITHUB_TOKEN]"),
            (re.compile(r"(gho_[A-Za-z0-9]{36,})"), "[REDACTED_GITHUB_TOKEN]"),
            # Slack tokens
            (
                re.compile(r"(xox[bpors]-[A-Za-z0-9\-]{10,})"),
                "[REDACTED_SLACK_TOKEN]",
            ),
        ]
        self._high_entropy = re.compile(r"\b([a-f0-9]{48,})\b")

    def redact(self, text: str) -> str:
        output = text
        for pattern, replacement in self._patterns:
            output = pattern.sub(replacement, output)
        output = self._high_entropy.sub("[REDACTED_HIGH_ENTROPY]", output)
        return output


# ---------------------------------------------------------------------------
# Schema — mirrors crates/storage/src/migrations/v001_initial.sql
# ---------------------------------------------------------------------------

DAEMON_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    kind_json TEXT NOT NULL,
    window_json TEXT,
    display_topology_json TEXT NOT NULL,
    primary_display_id TEXT NOT NULL,
    cursor_x INTEGER,
    cursor_y INTEGER,
    ui_scale REAL,
    artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    processed INTEGER NOT NULL DEFAULT 0,
    episode_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);
CREATE INDEX IF NOT EXISTS idx_events_episode_id ON events(episode_id);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str | None = None,
    timestamp: str | None = None,
    kind_json: str = '{"FocusChange":{}}',
    window_json: str | None = None,
    metadata_json: str = "{}",
) -> str:
    eid = event_id or _new_uuid()
    ts = timestamp or _now_iso()
    conn.execute(
        "INSERT INTO events "
        "(id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, metadata_json, processed) "
        "VALUES (?, ?, ?, ?, '[]', 'main', ?, 0)",
        (eid, ts, kind_json, window_json, metadata_json),
    )
    conn.commit()
    return eid


def _redact_event_row(event: dict, redactor: Redactor) -> dict:
    """Apply redaction to all text fields of an event row (dict from SQLite)."""
    redacted = dict(event)

    # Redact window_json title
    if redacted.get("window_json"):
        window = json.loads(redacted["window_json"])
        if "title" in window:
            window["title"] = redactor.redact(window["title"])
        redacted["window_json"] = json.dumps(window)

    # Redact metadata_json string values
    if redacted.get("metadata_json"):
        meta = json.loads(redacted["metadata_json"])
        if isinstance(meta, dict):
            for key in meta:
                if isinstance(meta[key], str):
                    meta[key] = redactor.redact(meta[key])
        redacted["metadata_json"] = json.dumps(meta)

    # Redact kind_json string values (e.g. target_description in ClickIntent)
    if redacted.get("kind_json"):
        kind = json.loads(redacted["kind_json"])
        if isinstance(kind, dict):
            for k, v in kind.items():
                if isinstance(v, dict):
                    for inner_k, inner_v in v.items():
                        if isinstance(inner_v, str):
                            v[inner_k] = redactor.redact(inner_v)
                elif isinstance(v, str):
                    kind[k] = redactor.redact(v)
        redacted["kind_json"] = json.dumps(kind)

    return redacted


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    db_file = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=wal;")
    conn.executescript(DAEMON_SCHEMA)
    conn.close()
    return db_file


@pytest.fixture()
def write_conn(tmp_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_db_path))
    conn.row_factory = sqlite3.Row
    yield conn  # type: ignore[misc]
    conn.close()


@pytest.fixture()
def redactor() -> Redactor:
    return Redactor()


# ===================================================================
# Test 1: AWS Access Key redaction in window title
# ===================================================================


class TestAWSAccessKeyRedaction:
    def test_aws_access_key_redacted_from_title(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """AWS access key (AKIA...) in window title is redacted."""
        secret = "AKIAIOSFODNN7EXAMPLE"
        _insert_event(
            write_conn,
            event_id="ev-aws-key",
            window_json=json.dumps({
                "app_id": "com.apple.Terminal",
                "title": f"export AWS_ACCESS_KEY_ID={secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 1
        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_AWS_KEY]" in window["title"]


# ===================================================================
# Test 2: GitHub token redaction in window title
# ===================================================================


class TestGitHubTokenRedaction:
    def test_github_personal_token_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """GitHub personal access token (ghp_...) is redacted."""
        secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        _insert_event(
            write_conn,
            event_id="ev-ghp",
            window_json=json.dumps({
                "app_id": "com.apple.Terminal",
                "title": f"GITHUB_TOKEN={secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_GITHUB_TOKEN]" in window["title"]


# ===================================================================
# Test 3: Credit card number redaction (with dashes)
# ===================================================================


class TestCreditCardDashesRedaction:
    def test_visa_with_dashes_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Visa credit card number with dashes is redacted."""
        secret = "4111-1111-1111-1111"
        _insert_event(
            write_conn,
            event_id="ev-cc-dashes",
            window_json=json.dumps({
                "app_id": "com.google.Chrome",
                "title": f"Payment: {secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_CC]" in window["title"]


# ===================================================================
# Test 4: Credit card number redaction (without dashes)
# ===================================================================


class TestCreditCardNoDashesRedaction:
    def test_visa_without_dashes_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Visa credit card number without dashes is redacted."""
        secret = "4111111111111111"
        _insert_event(
            write_conn,
            event_id="ev-cc-nodash",
            window_json=json.dumps({
                "app_id": "com.google.Chrome",
                "title": f"Card: {secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_CC]" in window["title"]


# ===================================================================
# Test 5: SSN redaction
# ===================================================================


class TestSSNRedaction:
    def test_ssn_redacted_from_title(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Social Security Number (XXX-XX-XXXX) is redacted."""
        secret = "123-45-6789"
        _insert_event(
            write_conn,
            event_id="ev-ssn",
            window_json=json.dumps({
                "app_id": "com.google.Chrome",
                "title": f"SSN: {secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_SSN]" in window["title"]


# ===================================================================
# Test 6: PEM private key redaction in metadata
# ===================================================================


class TestPEMPrivateKeyRedaction:
    def test_rsa_private_key_redacted_from_metadata(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """PEM RSA private key in metadata_json is redacted."""
        secret = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEAz3bZ...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        _insert_event(
            write_conn,
            event_id="ev-pem",
            metadata_json=json.dumps({"key_file": secret}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        meta = json.loads(redacted["metadata_json"])

        assert "MIIEowIBAAKCAQEAz3bZ" not in meta["key_file"]
        assert "[REDACTED_PRIVATE_KEY]" in meta["key_file"]


# ===================================================================
# Test 7: High-entropy hex string redaction
# ===================================================================


class TestHighEntropyHexRedaction:
    def test_64_char_hex_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """64+ character hex string (potential secret) is redacted."""
        secret = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
        _insert_event(
            write_conn,
            event_id="ev-hex",
            window_json=json.dumps({
                "app_id": "com.apple.Terminal",
                "title": f"hash: {secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_HIGH_ENTROPY]" in window["title"]


# ===================================================================
# Test 8: Bearer token redaction
# ===================================================================


class TestBearerTokenRedaction:
    def test_bearer_token_in_key_value_format_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Bearer token in key=value format is redacted."""
        secret_value = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9_long_token_value"
        full_text = f"bearer = {secret_value}"
        _insert_event(
            write_conn,
            event_id="ev-bearer",
            metadata_json=json.dumps({"auth_header": full_text}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        meta = json.loads(redacted["metadata_json"])

        assert secret_value not in meta["auth_header"]
        assert "[REDACTED_API_KEY]" in meta["auth_header"]


# ===================================================================
# Test 9: Slack token redaction
# ===================================================================


class TestSlackTokenRedaction:
    def test_slack_bot_token_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Slack bot token (xoxb-...) is redacted."""
        secret = "xoxb-1234567890-1234567890123-ABCDEFghijklMNOP"
        _insert_event(
            write_conn,
            event_id="ev-slack",
            window_json=json.dumps({
                "app_id": "com.apple.Terminal",
                "title": f"SLACK_TOKEN={secret}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])

        assert secret not in window["title"]
        assert "[REDACTED_SLACK_TOKEN]" in window["title"]


# ===================================================================
# Test 10: AWS secret access key redaction in metadata
# ===================================================================


class TestAWSSecretKeyRedaction:
    def test_aws_secret_access_key_redacted_from_metadata(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """AWS secret access key in metadata_json is redacted."""
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        full_text = f"aws_secret_access_key = {secret}"
        _insert_event(
            write_conn,
            event_id="ev-aws-secret",
            metadata_json=json.dumps({"env_var": full_text}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        meta = json.loads(redacted["metadata_json"])

        assert secret not in meta["env_var"]
        assert "[REDACTED_SECRET]" in meta["env_var"]


# ===================================================================
# Test 11: Generic API key in key=value format
# ===================================================================


class TestGenericAPIKeyRedaction:
    def test_api_key_in_key_value_format_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Generic API key (api_key=...) is redacted."""
        secret_value = "sk_live_1234567890abcdefghij"
        full_text = f"api_key = {secret_value}"
        _insert_event(
            write_conn,
            event_id="ev-apikey",
            metadata_json=json.dumps({"config_line": full_text}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        meta = json.loads(redacted["metadata_json"])

        assert secret_value not in meta["config_line"]
        assert "[REDACTED_API_KEY]" in meta["config_line"]


# ===================================================================
# Test 12: Multiple secrets in a single event
# ===================================================================


class TestMultipleSecretsInOneEvent:
    def test_multiple_secrets_all_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Multiple different secret types in one event are all redacted."""
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        cc_number = "4111-1111-1111-1111"
        ssn = "123-45-6789"
        _insert_event(
            write_conn,
            event_id="ev-multi",
            window_json=json.dumps({
                "app_id": "com.google.Chrome",
                "title": f"key={aws_key} cc={cc_number} ssn={ssn}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])
        title = window["title"]

        assert aws_key not in title
        assert cc_number not in title
        assert ssn not in title
        assert "[REDACTED_AWS_KEY]" in title
        assert "[REDACTED_CC]" in title
        assert "[REDACTED_SSN]" in title


# ===================================================================
# Test 13: Secrets in both title and metadata
# ===================================================================


class TestSecretsAcrossFields:
    def test_secrets_redacted_from_both_title_and_metadata(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Secrets in both window title and metadata_json are redacted."""
        github_token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        slack_token = "xoxb-1234567890-1234567890123-ABCDEFghijklMNOP"
        _insert_event(
            write_conn,
            event_id="ev-cross-field",
            window_json=json.dumps({
                "app_id": "com.apple.Terminal",
                "title": f"Token: {github_token}",
            }),
            metadata_json=json.dumps({
                "env": f"SLACK_TOKEN={slack_token}",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])
        meta = json.loads(redacted["metadata_json"])

        assert github_token not in window["title"]
        assert "[REDACTED_GITHUB_TOKEN]" in window["title"]
        assert slack_token not in meta["env"]
        assert "[REDACTED_SLACK_TOKEN]" in meta["env"]


# ===================================================================
# Test 14: Normal text passes through unchanged
# ===================================================================


class TestNormalTextPassthrough:
    def test_normal_text_not_modified(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Normal text without secrets passes through redaction unchanged."""
        title = "Google Chrome - Search Results for Python tutorials"
        meta_value = "User browsed documentation for 15 minutes"
        _insert_event(
            write_conn,
            event_id="ev-normal",
            window_json=json.dumps({
                "app_id": "com.google.Chrome",
                "title": title,
            }),
            metadata_json=json.dumps({"summary": meta_value}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        window = json.loads(redacted["window_json"])
        meta = json.loads(redacted["metadata_json"])

        assert window["title"] == title
        assert meta["summary"] == meta_value


# ===================================================================
# Test 15: Full pipeline round-trip with redaction
# ===================================================================


class TestFullPipelineRoundTrip:
    def test_redacted_events_through_full_db_roundtrip(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Events with secrets are redacted, stored, and re-read cleanly."""
        secrets = {
            "aws_key": "AKIAIOSFODNN7EXAMPLE",
            "github_token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
            "cc_number": "4111-1111-1111-1111",
            "ssn": "123-45-6789",
            "slack_token": "xoxb-1234567890-1234567890123-ABCDEFghijklMNOP",
        }

        # Insert events with various secrets
        _insert_event(
            write_conn,
            event_id="ev-rt-1",
            window_json=json.dumps({
                "app_id": "com.apple.Terminal",
                "title": f"AWS={secrets['aws_key']}",
            }),
        )
        _insert_event(
            write_conn,
            event_id="ev-rt-2",
            metadata_json=json.dumps({
                "token": secrets["github_token"],
                "card": secrets["cc_number"],
            }),
        )
        _insert_event(
            write_conn,
            event_id="ev-rt-3",
            window_json=json.dumps({
                "app_id": "com.google.Chrome",
                "title": f"SSN: {secrets['ssn']} Slack: {secrets['slack_token']}",
            }),
        )

        # Read, redact, verify
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 3

        for event in events:
            redacted = _redact_event_row(dict(event), redactor)

            # Serialize the entire event to a single string for scanning
            full_text = json.dumps(redacted)

            # Verify NONE of the raw secrets appear anywhere in the output
            for secret_name, secret_value in secrets.items():
                assert secret_value not in full_text, (
                    f"Secret '{secret_name}' ({secret_value}) leaked in event "
                    f"{redacted.get('id', 'unknown')}"
                )


# ===================================================================
# Test 16: Secrets in ClickIntent target_description (kind_json)
# ===================================================================


class TestSecretsInKindJson:
    def test_secret_in_click_target_description_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """Secret in kind_json target_description field is redacted."""
        secret = "AKIAIOSFODNN7EXAMPLE"
        _insert_event(
            write_conn,
            event_id="ev-kind",
            kind_json=json.dumps({
                "ClickIntent": {"target_description": f"Copy key {secret}"},
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        kind = json.loads(redacted["kind_json"])

        assert secret not in kind["ClickIntent"]["target_description"]
        assert "[REDACTED_AWS_KEY]" in kind["ClickIntent"]["target_description"]


# ===================================================================
# Test 17: EC Private Key redaction
# ===================================================================


class TestECPrivateKeyRedaction:
    def test_ec_private_key_redacted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection, redactor: Redactor
    ) -> None:
        """EC private key (-----BEGIN EC PRIVATE KEY-----) is redacted."""
        secret = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "MHQCAQEEIBkg...\n"
            "-----END EC PRIVATE KEY-----"
        )
        _insert_event(
            write_conn,
            event_id="ev-ec-pem",
            metadata_json=json.dumps({"key_data": secret}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        redacted = _redact_event_row(dict(events[0]), redactor)
        meta = json.loads(redacted["metadata_json"])

        assert "MHQCAQEEIBkg" not in meta["key_data"]
        assert "[REDACTED_PRIVATE_KEY]" in meta["key_data"]

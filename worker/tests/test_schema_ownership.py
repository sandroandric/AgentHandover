"""One schema owner: the daemon creates and migrates every table
(``crates/storage/src/migrations``); the worker only checks the version.

The upgrade of a v0.4.0 database is pinned in
``crates/storage/tests/migration_test.rs``; the worker's start against a
mismatched version and the import of a v0.4.0 knowledge base in
``test_main_once.py``.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from agenthandover_worker import db as worker_db
from agenthandover_worker.db import (
    SCHEMA_VERSION,
    SchemaVersionError,
    check_schema_version,
    wait_for_schema_version,
)
from daemon_db import schema_version

WORKER_SRC = Path(worker_db.__file__).resolve().parent


def _set_version(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {version}")
    conn.close()


def test_the_worker_expects_the_daemons_current_schema() -> None:
    assert SCHEMA_VERSION == schema_version()


def test_no_worker_module_changes_the_schema() -> None:
    """DDL lives in the daemon's migrations only."""
    ddl = re.compile(r"\b(CREATE|ALTER|DROP)\s+(VIRTUAL\s+)?(TABLE|INDEX)\b")
    offenders = [
        f"{path.relative_to(WORKER_SRC)}:{n}"
        for path in sorted(WORKER_SRC.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if ddl.search(line)
    ]
    assert offenders == []


def test_a_database_the_daemon_created_passes(tmp_db_path: Path) -> None:
    check_schema_version(tmp_db_path)
    wait_for_schema_version(tmp_db_path, timeout_s=0, poll_s=0)


def test_an_older_database_fails_with_the_fix(tmp_db_path: Path) -> None:
    _set_version(tmp_db_path, SCHEMA_VERSION - 2)
    with pytest.raises(SchemaVersionError) as info:
        wait_for_schema_version(tmp_db_path, timeout_s=0.1, poll_s=0.05)
    message = str(info.value)
    assert f"schema version {SCHEMA_VERSION - 2}" in message
    assert "agenthandover restart" in message


def test_a_newer_database_fails_at_once(tmp_db_path: Path) -> None:
    _set_version(tmp_db_path, SCHEMA_VERSION + 1)
    started = time.monotonic()
    with pytest.raises(SchemaVersionError, match="newer than the version"):
        wait_for_schema_version(tmp_db_path, timeout_s=60, poll_s=1)
    assert time.monotonic() - started < 1


def test_the_worker_waits_for_the_daemon_to_migrate(tmp_db_path: Path) -> None:
    """Right after an upgrade the worker can start before the daemon has
    migrated: it waits, then runs."""
    _set_version(tmp_db_path, SCHEMA_VERSION - 1)
    daemon = threading.Timer(0.2, _set_version, (tmp_db_path, SCHEMA_VERSION))
    daemon.start()
    try:
        wait_for_schema_version(tmp_db_path, timeout_s=5, poll_s=0.05)
    finally:
        daemon.cancel()
    check_schema_version(tmp_db_path)


def test_a_shutdown_ends_the_wait(tmp_db_path: Path) -> None:
    _set_version(tmp_db_path, SCHEMA_VERSION - 1)
    wait_for_schema_version(tmp_db_path, timeout_s=60, poll_s=1, stop=lambda: True)


# -- after an upgrade with capture paused (the daemon never starts) --------


def test_an_older_database_is_migrated_once_without_waiting_for_the_daemon(tmp_db_path: Path) -> None:
    """A 0.4.0 database upgraded while capture was paused stayed old for
    good: the worker now asks for the daemon's migrations itself."""
    _set_version(tmp_db_path, 3)
    calls: list[int] = []

    def migrate() -> None:
        calls.append(1)
        _set_version(tmp_db_path, SCHEMA_VERSION)

    started = time.monotonic()
    wait_for_schema_version(tmp_db_path, timeout_s=60, poll_s=1, migrate=migrate)
    assert calls == [1]
    assert time.monotonic() - started < 1, "no wait for a daemon"
    check_schema_version(tmp_db_path)


def test_a_failed_migration_is_not_retried_and_the_wait_ends_with_the_fix(tmp_db_path: Path) -> None:
    _set_version(tmp_db_path, 3)
    calls: list[int] = []
    with pytest.raises(SchemaVersionError, match="agenthandover db migrate"):
        wait_for_schema_version(
            tmp_db_path, timeout_s=0.1, poll_s=0.05, migrate=lambda: calls.append(1),
        )
    assert calls == [1]


def test_a_newer_or_current_database_is_never_migrated(tmp_db_path: Path) -> None:
    calls: list[int] = []
    wait_for_schema_version(tmp_db_path, timeout_s=0, poll_s=0, migrate=lambda: calls.append(1))
    _set_version(tmp_db_path, SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionError):
        wait_for_schema_version(tmp_db_path, timeout_s=60, poll_s=1, migrate=lambda: calls.append(1))
    assert calls == []


def _fake_cli(tmp_path: Path, script: str) -> str:
    cli = tmp_path / "agenthandover"
    cli.write_text("#!/bin/sh\n" + script, encoding="utf-8")
    cli.chmod(0o755)
    return str(cli)


def test_the_worker_runs_the_clis_migration_on_its_own_database(tmp_path: Path, caplog) -> None:
    from agenthandover_worker import main as worker_main

    args = tmp_path / "args"
    cli = _fake_cli(tmp_path, f'echo "$@" > {args}\necho "Database migrated from schema 3 to {SCHEMA_VERSION}"\n')
    db = tmp_path / "events.db"
    with caplog.at_level("INFO"):
        worker_main._migrate_with_cli(db, cli=cli)
    assert args.read_text().split() == ["db", "migrate", "--path", str(db)]
    assert f"Database migrated from schema 3 to {SCHEMA_VERSION}" in caplog.text


def test_a_failing_or_missing_cli_is_logged_not_raised(tmp_path: Path, caplog, monkeypatch) -> None:
    from agenthandover_worker import main as worker_main

    cli = _fake_cli(tmp_path, 'echo "error: unrecognized subcommand db" >&2\nexit 2\n')
    with caplog.at_level("WARNING"):
        worker_main._migrate_with_cli(tmp_path / "events.db", cli=cli)
        worker_main._migrate_with_cli(tmp_path / "events.db", cli=str(tmp_path / "missing"))
    assert "agenthandover db migrate failed (2): error: unrecognized subcommand db" in caplog.text
    assert "could not run" in caplog.text

    monkeypatch.setattr(worker_main, "_INSTALLED_CLI", str(tmp_path / "nowhere"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with caplog.at_level("WARNING"):
        worker_main._migrate_with_cli(tmp_path / "events.db")
    assert "no agenthandover CLI was found" in caplog.text

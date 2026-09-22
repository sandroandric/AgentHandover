"""Shared fixtures for worker tests.

Creates temporary SQLite databases by applying the Rust daemon's own
migrations (see ``daemon_db.py``), and sandboxes ``HOME`` for the whole
session (see ``home_sandbox.py``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from daemon_db import create_daemon_db
# Sandboxed HOME: registered before any agenthandover_worker import.
from home_sandbox import (  # noqa: F401  (pytest discovers these by name)
    pytest_configure,
    pytest_unconfigure,
    real_home_untouched,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    """Create an empty database with the daemon schema, return its path."""
    return create_daemon_db(tmp_path / "events.db")


@pytest.fixture()
def write_conn(tmp_db_path: Path) -> sqlite3.Connection:
    """Return a read-write connection to the temp database (for inserting
    test data).  Callers must NOT close this; the fixture handles cleanup.
    """
    conn = sqlite3.connect(str(tmp_db_path))
    conn.row_factory = sqlite3.Row
    yield conn  # type: ignore[misc]
    conn.close()

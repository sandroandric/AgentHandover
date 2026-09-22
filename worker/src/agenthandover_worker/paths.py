"""Where the worker, the MCP server and the agent connectors find their data.

One definition, used by ``main``, ``mcp_server``, ``agent_connect`` and
``knowledge_base``, so the MCP server reads the same database and
knowledge base the worker writes.  The database path must match the
daemon's default (``crates/daemon/src/main.rs``).

The functions resolve ``HOME`` when called; ``DEFAULT_DB_PATH`` and
``KNOWLEDGE_BASE_DIR`` are their values at import, the worker's
``--db-path`` and ``--knowledge-dir`` defaults.  Default paths assume a
single-user system.
"""

from __future__ import annotations

import platform
from pathlib import Path


def data_dir() -> Path:
    """The daemon/worker data directory (database, status files, config)."""
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "agenthandover"
    return Path.home() / ".local" / "share" / "agenthandover"


def db_path() -> Path:
    """The daemon's database, which the worker also writes (vector store included)."""
    return data_dir() / "events.db"


def knowledge_base_dir() -> Path:
    """The knowledge base root the worker writes procedures to."""
    return Path.home() / ".agenthandover" / "knowledge"


DEFAULT_DB_PATH = db_path()
KNOWLEDGE_BASE_DIR = knowledge_base_dir()

"""Where the worker, the MCP server and the agent connectors find their data.

One definition, imported by ``main``, ``mcp_server``, ``agent_connect``
and ``knowledge_base``, so the MCP server reads the same database and
knowledge base the worker writes.  ``DEFAULT_DB_PATH`` must match the
daemon's default (``crates/daemon/src/main.rs``).

Default paths assume a single-user system; the worker's ``--db-path``
and ``--knowledge-dir`` override them.
"""

from __future__ import annotations

import platform
from pathlib import Path


def data_dir() -> Path:
    """The daemon/worker data directory (database, status files, config)."""
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "agenthandover"
    return Path.home() / ".local" / "share" / "agenthandover"


DEFAULT_DB_PATH = data_dir() / "events.db"
KNOWLEDGE_BASE_DIR = Path.home() / ".agenthandover" / "knowledge"

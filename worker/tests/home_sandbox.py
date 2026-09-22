"""Sandboxed ``HOME`` for every Python test run.

The worker resolves its data, knowledge-base, config and Claude Code
paths from ``Path.home()``.  Without a sandbox the suite writes into the
developer's real ``~/.claude/skills`` and
``~/Library/Application Support/agenthandover``.

``install()`` points ``HOME`` at a fresh temporary directory and records
every write the test process makes under the real home directory (via a
``sys`` audit hook), so the session-end check can fail the run if any
code path bypasses ``HOME``.  Writes inside the repository checkout and
inside the sandbox are allowed; everything else under the real home is
a violation.

Both the repository-root ``conftest.py`` and ``worker/tests/conftest.py``
call ``install()`` from ``pytest_configure``, so the sandbox is active
before any test module (and so any ``agenthandover_worker`` module) is
imported, whichever directory pytest runs from.
"""

from __future__ import annotations

import os
import pwd
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()

_WRITE_MODE_CHARS = set("wax+")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# Audit events whose path arguments are all written/created/removed.
_MUTATING_EVENTS = {
    "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.symlink",
    "os.link", "os.truncate", "os.chmod", "os.chown", "os.utime",
    "shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.rmtree",
    "sqlite3.connect",
}

_sandbox: Path | None = None
_violations: list[str] = []


def _as_path(arg: object) -> str | None:
    if isinstance(arg, os.PathLike):
        arg = os.fspath(arg)
    if isinstance(arg, bytes):
        arg = os.fsdecode(arg)
    if isinstance(arg, str) and arg and arg != ":memory:" and not arg.startswith("file:"):
        return os.path.abspath(arg)
    return None


def _is_violation(path: str) -> bool:
    p = Path(os.path.realpath(path))
    if not p.is_relative_to(REAL_HOME):
        return False
    if p.is_relative_to(REPO_ROOT):
        return False
    return _sandbox is None or not p.is_relative_to(_sandbox)


def _audit(event: str, args: tuple) -> None:
    if event == "open":
        path, mode, flags = args[0], args[1], args[2]
        if isinstance(mode, str):
            writes = bool(_WRITE_MODE_CHARS & set(mode))
        else:
            writes = bool((flags or 0) & _WRITE_FLAGS)
        candidates = [path] if writes else []
    elif event in _MUTATING_EVENTS:
        # copy/link events are (src, dst, ...): only dst is written.
        if event in ("shutil.copyfile", "shutil.copytree", "os.symlink", "os.link"):
            candidates = [args[1]]
        else:
            candidates = [a for a in args[:2]]
    else:
        return
    for arg in candidates:
        path = _as_path(arg)
        if path is not None and _is_violation(path):
            _violations.append(f"{event}: {path}")


def install() -> Path:
    """Point ``HOME`` at a temporary directory (idempotent); return it."""
    global _sandbox
    if _sandbox is not None:
        return _sandbox
    _sandbox = Path(tempfile.mkdtemp(prefix="agenthandover-test-home-")).resolve()
    os.environ["HOME"] = str(_sandbox)
    # Tools that honour XDG before HOME must not escape the sandbox either.
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        os.environ.pop(var, None)
    # ``agenthandover connect claude-code`` runs the Claude Code CLI, which
    # writes the developer's real Claude Code config.  Tests pair against a
    # stub (they set AH_CLAUDE_BIN); anything else finds no CLI here.
    os.environ["AH_CLAUDE_BIN"] = str(_sandbox / "claude-cli-disabled-in-tests")
    sys.addaudithook(_audit)
    return _sandbox


def violations() -> list[str]:
    """Writes recorded under the real home outside the repo and sandbox."""
    return list(_violations)


def sandbox_home() -> Path | None:
    return _sandbox


def remove() -> None:
    """Delete the sandbox directory (the audit hook stays; it only records)."""
    if _sandbox is not None:
        shutil.rmtree(_sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# pytest hooks and fixture, imported by both conftest.py files
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    install()


def pytest_unconfigure(config: pytest.Config) -> None:
    remove()


@pytest.fixture(scope="session", autouse=True)
def real_home_untouched():
    """Fail the session if any test wrote under the real home directory."""
    assert Path.home().resolve() == sandbox_home(), (
        "HOME is not sandboxed; the conftest hook did not run"
    )
    yield
    found = violations()
    assert not found, (
        "Tests wrote under the real home directory "
        f"({REAL_HOME}):\n  " + "\n  ".join(sorted(set(found)))
    )

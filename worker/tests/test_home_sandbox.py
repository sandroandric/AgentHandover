"""Pins for the sandboxed HOME every test session runs under."""

from __future__ import annotations

import os
from pathlib import Path

import home_sandbox


def test_home_points_at_sandbox():
    sandbox = home_sandbox.sandbox_home()
    assert sandbox is not None
    assert Path.home().resolve() == sandbox
    assert Path(os.path.expanduser("~")).resolve() == sandbox
    assert sandbox != home_sandbox.REAL_HOME


def test_real_home_paths_are_violations():
    assert home_sandbox._is_violation(str(home_sandbox.REAL_HOME / ".claude" / "skills" / "x" / "SKILL.md"))
    assert home_sandbox._is_violation(
        str(home_sandbox.REAL_HOME / "Library" / "Application Support" / "agenthandover" / "a.json")
    )


def test_repo_and_sandbox_paths_are_allowed(tmp_path):
    assert not home_sandbox._is_violation(str(home_sandbox.REPO_ROOT / "worker" / "x.pyc"))
    assert not home_sandbox._is_violation(str(home_sandbox.sandbox_home() / ".claude" / "x"))
    assert not home_sandbox._is_violation(str(tmp_path / "x"))


def test_write_detection_does_not_record_reads(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(home_sandbox, "_violations", recorded)
    target = str(home_sandbox.REAL_HOME / "not-written")
    home_sandbox._audit("open", (target, "r", 0))
    home_sandbox._audit("open", (target, None, os.O_RDONLY))
    assert recorded == []
    home_sandbox._audit("open", (target, "w", 0))
    home_sandbox._audit("open", (target, None, os.O_WRONLY | os.O_CREAT))
    home_sandbox._audit("os.mkdir", (target, 0o777, -1))
    home_sandbox._audit("shutil.copyfile", ("/etc/hosts", target))
    home_sandbox._audit("sqlite3.connect", (target,))
    assert len(recorded) == 5

"""J-install: the real pre/postinstall into temporary prefixes.

The journey itself is ``tests/install/test_install.sh`` (WP-I1): fresh and
upgrade installs, a broken venv rebuilt, an old-Python-only prefix refused,
the build-pkg staging checks.  It builds the Rust binaries, needs network
access for pip and is meant for the test Mac, so it runs only with
``AH_INSTALL_JOURNEY=1``.

The console-script check below needs no install: it runs the
``agenthandover-mcp`` this environment installed from the same entry point.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_JOURNEY = REPO_ROOT / "tests" / "install" / "test_install.sh"


@pytest.mark.skipif(
    os.environ.get("AH_INSTALL_JOURNEY") != "1",
    reason="install journey runs on the test Mac: set AH_INSTALL_JOURNEY=1",
)
def test_install_journey() -> None:
    result = subprocess.run(
        ["bash", str(INSTALL_JOURNEY)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    output = result.stdout + result.stderr
    summary = re.search(r"Install journey: (\d+) passed, (\d+) failed, (\d+) skipped", output)
    assert summary, output[-4000:]
    failures = [line for line in output.splitlines() if line.startswith("  FAIL")]
    assert result.returncode == 0 and summary.group(2) == "0", "\n".join(failures) or output[-4000:]


@pytest.mark.xfail(strict=True, reason=(
    "WP-I1/WP-I2: agenthandover-mcp has no --help: mcp_server.main "
    "(mcp_server.py:394-406) ignores every argument but --sse and starts the "
    "stdio server, so '--help' prints nothing and exits 0 only because stdin "
    "is at EOF (from a terminal it waits for an MCP client)"
))
def test_mcp_console_script_answers_help(tmp_path: Path) -> None:
    script = Path(sys.executable).parent / "agenthandover-mcp"
    if not script.is_file():
        pytest.skip("agenthandover-mcp is not installed in this environment")
    result = subprocess.run(
        [str(script), "--help"], stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=60, env={**os.environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower(), (result.stdout, result.stderr)

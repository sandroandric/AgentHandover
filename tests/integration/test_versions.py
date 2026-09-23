"""Every component carries one version (scripts/version.py).

Real-life finding (test Mac, 2026-09-23): a package built with
VERSION=0.5.0-dev installed a worker that reported 0.4.0, because
build-pkg.sh named the package after $VERSION while the worker, CLI and
app kept the versions in their own manifests.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "version.py"

_spec = importlib.util.spec_from_file_location("ah_version_script", SCRIPT)
version_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(version_script)


def _copy_tree(dst: Path) -> Path:
    for rel, _pattern, _count in version_script.FIELDS:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst / rel)
    return dst


def test_the_repo_carries_one_version():
    version = version_script.check()
    assert version_script.VERSION_RE.match(version), version


def test_the_installed_worker_reports_the_repo_version():
    from agenthandover_worker import __version__

    assert __version__ == version_script.check()


def test_cli_check_prints_the_version():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "check"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == version_script.check()


def test_stamp_writes_every_field_and_nothing_else(tmp_path):
    root = _copy_tree(tmp_path)
    old = version_script.check(root)
    version_script.stamp("12.34.56", root)
    assert version_script.check(root) == "12.34.56"
    assert all(v == ["12.34.56"] * len(v) for v in version_script.read_versions(root).values())
    for rel, _pattern, count in version_script.FIELDS:
        before = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
        after = (root / rel).read_text(encoding="utf-8").splitlines()
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(before) == len(after), rel
        assert len(changed) == count, (rel, changed)
        assert all(a.replace(old, "12.34.56") == b for a, b in changed), (rel, changed)


def test_a_disagreeing_file_is_named(tmp_path):
    root = _copy_tree(tmp_path)
    plist = root / "app/AgentHandoverApp/Sources/AgentHandoverApp/Info.plist"
    old = version_script.check(root)
    plist.write_text(
        plist.read_text(encoding="utf-8").replace(f"<string>{old}</string>", "<string>9.9.9</string>"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        version_script.check(root)
    assert "Info.plist: 9.9.9, 9.9.9" in str(exc.value)
    assert f"worker/pyproject.toml: {old}" in str(exc.value)


@pytest.mark.parametrize("bad", ["0.5.0-dev", "0.5", "v0.5.0", "0.5.0.dev0"])
def test_stamp_refuses_versions_some_component_cannot_carry(tmp_path, bad):
    root = _copy_tree(tmp_path)
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        version_script.stamp(bad, root)
    assert version_script.check(root) == version_script.check()


def test_a_file_whose_layout_changed_is_refused_before_any_write(tmp_path):
    root = _copy_tree(tmp_path)
    (root / "extension/manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extension/manifest.json: expected 1"):
        version_script.stamp("1.2.3", root)
    assert (root / "Cargo.toml").read_text(encoding="utf-8") == (REPO_ROOT / "Cargo.toml").read_text(
        encoding="utf-8"
    )

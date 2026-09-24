"""The docs say what the code does (WP-S3).

- README's MCP tool table and model tier table are generated from the code
  (the MCP server's tool list, ``model_profiles.TIERS``); a change to either
  fails here until the README is rewritten with ``AH_UPDATE_DOCS=1`` and
  the diff reviewed.  The CLI table is pinned the same way on the Rust side
  (``crates/cli/src/docs_tests.rs``), which also checks every
  ``agenthandover <subcommand>`` the README and the in-app FAQ mention.
- ``config.example.toml`` lists only keys some code reads, and documents
  every key the worker's config loader has.
- The changelog's newest entry is the repo's version.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import tomllib
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CONFIG_EXAMPLE = REPO_ROOT / "config.example.toml"
RUST_CONFIG = REPO_ROOT / "crates" / "common" / "src" / "config.rs"


# ---------------------------------------------------------------------------
# Generated README sections
# ---------------------------------------------------------------------------


def _section(text: str, name: str) -> tuple[int, int]:
    start = f"<!-- {name}:start -->\n"
    end = f"<!-- {name}:end -->"
    i, j = text.find(start), text.find(end)
    assert i >= 0 and j > i, f"README has no {name} markers"
    return i + len(start), j


def _check_generated(name: str, generated: str) -> None:
    text = README.read_text(encoding="utf-8")
    i, j = _section(text, name)
    if os.environ.get("AH_UPDATE_DOCS") == "1":
        README.write_text(text[:i] + generated + text[j:], encoding="utf-8")
        return
    assert text[i:j] == generated, (
        f"README's {name} section is not what the code generates; rewrite it "
        "with AH_UPDATE_DOCS=1 and review the diff"
    )


def mcp_tools_table() -> str:
    """The MCP server's tools and resources, with the first line of each
    docstring (what an agent reads)."""
    pytest.importorskip("mcp")
    from agenthandover_worker.mcp_server import build_mcp_server

    server = build_mcp_server()
    tools = asyncio.run(server.list_tools())
    templates = asyncio.run(server.list_resource_templates())
    resources = asyncio.run(server.list_resources())
    lines = ["| Tool | What it does |", "|------|-------------|"]
    for tool in tools:
        required = tool.inputSchema.get("required") or []
        summary = (tool.description or "").strip().splitlines()[0]
        lines.append(f"| `{tool.name}({', '.join(required)})` | {summary} |")
    lines += ["", "| Resource | What it is |", "|----------|-----------|"]
    for res in [*resources, *templates]:
        uri = getattr(res, "uriTemplate", None) or str(res.uri)
        summary = (res.description or "").strip().splitlines()[0]
        lines.append(f"| `{uri}` | {summary} |")
    return "\n".join(lines) + "\n"


def tier_table() -> str:
    """``model_profiles.TIERS`` with each tier's download (the models it
    pulls, ``MODEL_SIZES_GB``)."""
    from agenthandover_worker.model_profiles import MODEL_SIZES_GB, TIERS

    lines = [
        "| RAM | Tier | Screen annotation | Skill writing | Download |",
        "|-----|------|-------------------|---------------|----------|",
    ]
    for tier in TIERS:
        ram = f"{tier.min_ram_gb} GB+" if tier.min_ram_gb else "8 GB"
        size = sum(MODEL_SIZES_GB[m] for m in tier.models)
        lines.append(
            f"| {ram} | {tier.label} | `{tier.annotation_model}` | "
            f"`{tier.sop_model}` | ~{size:.1f} GB |"
        )
    return "\n".join(lines) + "\n"


def test_readme_mcp_tools_are_the_servers() -> None:
    _check_generated("mcp-tools", mcp_tools_table())


def test_readme_tier_table_is_the_tier_table() -> None:
    _check_generated("tiers", tier_table())


def test_the_readme_quotes_the_ollama_minimum_for_gemma() -> None:
    from agenthandover_worker.model_profiles import MINIMUM_OLLAMA_VERSION_FOR_GEMMA4

    text = README.read_text(encoding="utf-8")
    assert f"Ollama {MINIMUM_OLLAMA_VERSION_FOR_GEMMA4}" in text
    assert "0.30.6" not in text.split("## Changelog")[0]


# ---------------------------------------------------------------------------
# config.example.toml
# ---------------------------------------------------------------------------


def _example_keys() -> list[tuple[str, str]]:
    """(section, key) for every key in config.example.toml, commented out
    or not."""
    keys, section = [], None
    for line in CONFIG_EXAMPLE.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^\s*#?\s*\[([a-z_.]+)\]\s*$", line)
        if header:
            section = header.group(1)
            continue
        key = re.match(r"^#?\s?([a-z_]+)\s*=", line)
        if key and section:
            keys.append((section, key.group(1)))
    return keys


def _worker_fields() -> dict[str, set[str]]:
    """The worker loader's keys per section (``config.Config``)."""
    from agenthandover_worker import config
    from agenthandover_worker.privacy_zones import PrivacyZoneConfig

    out: dict[str, set[str]] = {}
    defaults = config.Config()
    for f in fields(config.Config):
        value = getattr(defaults, f.name)
        if is_dataclass(value):
            out[f.name] = {g.name for g in fields(value)}
    # Read as one string and split into the two times.
    out["idle_jobs"] = (out["idle_jobs"] - {"run_window_start", "run_window_end"}) | {
        "run_window_local_time"
    }
    out["privacy.zones"] = {g.name for g in fields(PrivacyZoneConfig)}
    return out


def _daemon_reads(section: str, key: str) -> bool:
    """A field of the daemon's config struct that Rust code outside the
    config module reads (``app_config.<section>.<key>``,
    ``<section>_config.<key>``)."""
    rust = RUST_CONFIG.read_text(encoding="utf-8")
    if not re.search(rf"\bpub {key}:", rust):
        return False
    pattern = re.compile(rf"\b{section}(?:_config)?\s*\.\s*{key}\b")
    return any(
        pattern.search(path.read_text(encoding="utf-8"))
        for path in (REPO_ROOT / "crates").glob("*/src/**/*.rs")
        if path != RUST_CONFIG
    )


def test_config_example_lists_only_keys_some_code_reads() -> None:
    worker = _worker_fields()
    unread = [
        f"[{section}] {key}" for section, key in _example_keys()
        if key not in worker.get(section, set()) and not _daemon_reads(section, key)
    ]
    assert unread == []


def test_config_example_documents_every_worker_key() -> None:
    documented = set(_example_keys())
    missing = [
        f"[{section}] {key}" for section, keys in _worker_fields().items()
        for key in sorted(keys) if (section, key) not in documented
    ]
    assert missing == []


def test_config_example_is_valid_toml_the_worker_loads(tmp_path: Path) -> None:
    from agenthandover_worker.config import load_config

    tomllib.loads(CONFIG_EXAMPLE.read_text(encoding="utf-8"))
    cfg = load_config(CONFIG_EXAMPLE)
    assert cfg.sop.auto_approve is False
    assert cfg.knowledge.weekly_question_budget == 5


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------


def test_the_newest_changelog_entry_is_the_repo_version() -> None:
    spec = importlib.util.spec_from_file_location(
        "ah_version_script_docs", REPO_ROOT / "scripts" / "version.py",
    )
    version_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_script)
    version = version_script.check()
    for doc, heading in ((CHANGELOG, r"^## \[(\d+\.\d+\.\d+)\]"), (README, r"^### v(\d+\.\d+\.\d+)")):
        [first] = re.findall(heading, doc.read_text(encoding="utf-8"), re.M)[:1]
        assert first == version, doc.name

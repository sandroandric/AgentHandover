"""The export manifest: every path AgentHandover writes into a Claude Code
skills root is recorded, so ``agenthandover uninstall --purge-data``
removes exactly those and never a Skill the user wrote
(``crates/cli/src/commands/uninstall.rs``, #60)."""

from __future__ import annotations

import json
from pathlib import Path

from agenthandover_worker.claude_skill_writer import (
    EXPORT_MANIFEST,
    ClaudeSkillWriter,
    forget_export,
    record_export,
)
from agenthandover_worker.paths import data_dir


def _manifest() -> list[str]:
    path = data_dir() / EXPORT_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))["paths"] if path.exists() else []


def _procedure(slug: str) -> dict:
    return {
        "id": slug,
        "title": "File expenses",
        "steps": [{"step": "Open the expenses page", "action": "open"}],
    }


def test_exported_skills_and_index_are_recorded_and_forgotten(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    mine = root / "mine" / "SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("my own skill\n", encoding="utf-8")

    writer = ClaudeSkillWriter(skills_dir=root)
    skill = writer.write_procedure(_procedure("file-expenses"))
    meta = writer.write_metadata("export-summary", {"count": 1})

    recorded = _manifest()
    assert str(skill.parent) in recorded
    assert str(meta) in recorded
    assert str(mine.parent) not in recorded, "a hand-written Skill is never recorded"

    writer.write_all_sops([])  # nothing to export: the index is removed
    assert str(root / "AGENTHANDOVER-INDEX.md") not in _manifest()

    forget_export(skill.parent)
    assert str(skill.parent) not in _manifest()


def test_record_is_idempotent_and_survives_a_corrupt_manifest(tmp_path: Path) -> None:
    (data_dir()).mkdir(parents=True, exist_ok=True)
    (data_dir() / EXPORT_MANIFEST).write_text("{not json", encoding="utf-8")
    record_export(tmp_path / "a")
    record_export(tmp_path / "a")
    assert _manifest() == [str(tmp_path / "a")]

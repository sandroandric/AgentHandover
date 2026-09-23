"""Connector journey: `connect codex` and `connect hermes` never destroy the
user's own files.

The user keeps prose in ``./AGENTS.md`` and a hand-written Skill inside
AgentHandover's Hermes namespace.  Both connectors run twice through the
real CLI entry point; the prose and the hand-written file survive, and the
generated block appears exactly once.  No model is called.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agenthandover_worker.agent_connect import CODEX_BEGIN, CODEX_END, HERMES_MANIFEST
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import sop_to_procedure
from fixtures.vlm_answers import generated_sop_template

SLUG = "check-domain-availability"
PROSE = "# House rules\n\nNever push to main without review.\n"


def _connect(env: dict, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.agent_connect", *args],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=120,
    )


def test_codex_and_hermes_keep_the_users_files(temp_home, kb_root, tmp_path: Path) -> None:
    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    # Approved for agents (the approval grants trust "suggest") with a goal,
    # so the hand-off gate lets it through.
    proc.update({
        "id": SLUG, "slug": SLUG, "title": "Check Domain Availability",
        "goal": "Find out whether a domain is free to register and what it costs.",
        "lifecycle_state": "agent_ready",
        "constraints": {"trust_level": "suggest", "guardrails": []},
    })
    kb = KnowledgeBase(root=kb_root)
    kb.ensure_structure()
    kb.save_procedure(proc)

    project = tmp_path / "project"
    project.mkdir()
    agents_md = project / "AGENTS.md"
    agents_md.write_text(PROSE, encoding="utf-8")

    namespace = temp_home.path / ".hermes" / "skills" / "agenthandover"
    (namespace / "my-own-skill").mkdir(parents=True)
    handmade = namespace / "my-own-skill" / "SKILL.md"
    handmade.write_text("---\nname: my-own-skill\n---\nWritten by hand.\n", encoding="utf-8")

    env = {**os.environ, "HOME": str(temp_home.path)}

    # Codex: an AGENTS.md without markers is not touched without consent.
    refused = _connect(env, project, "codex")
    assert refused.returncode == 1
    assert str(agents_md.resolve()) in refused.stdout
    assert agents_md.read_text(encoding="utf-8") == PROSE

    first = _connect(env, project, "codex", "--append")
    assert first.returncode == 0, first.stdout + first.stderr
    after_first = agents_md.read_text(encoding="utf-8")
    second = _connect(env, project, "codex")
    assert second.returncode == 0, second.stdout + second.stderr
    content = agents_md.read_text(encoding="utf-8")

    assert content == after_first  # the second run replaced the block in place
    assert content.startswith(PROSE)
    assert content.count(CODEX_BEGIN) == 1 and content.count(CODEX_END) == 1
    assert content.count("### Check Domain Availability") == 1

    # Hermes, twice.
    for _ in range(2):
        run = _connect(env, project, "hermes")
        assert run.returncode == 0, run.stdout + run.stderr

    assert handmade.read_text(encoding="utf-8").endswith("Written by hand.\n")
    generated = namespace / SLUG / "SKILL.md"
    assert generated.read_text(encoding="utf-8").startswith(f"---\nname: {SLUG}\n")
    manifest = json.loads((namespace / HERMES_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["files"] == [f"{SLUG}/SKILL.md"]
    assert sorted(p.name for p in namespace.iterdir()) == sorted(
        [HERMES_MANIFEST, SLUG, "my-own-skill"]
    )

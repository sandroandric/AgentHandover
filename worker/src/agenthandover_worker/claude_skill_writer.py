"""Claude Code Skill Export Adapter — native Claude Code skill files.

Each procedure is written to ``~/.claude/skills/<slug>/SKILL.md`` (Claude
Code's personal skills directory convention), rendered by
``skill_render.render_packet(proc, target="claude-skill")``.  SOP
templates are converted to procedures first, so both entry points
produce the same packet.

Implements the ``SOPExportAdapter`` ABC.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agenthandover_worker import __version__
from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.exporter import AtomicWriter
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.skill_render import description as skill_description, render_packet


class ClaudeSkillWriter(SOPExportAdapter):
    """Write SOPs as Claude Code-compatible skill files.

    Each SOP becomes ``~/.claude/skills/<slug>/SKILL.md`` with YAML
    frontmatter (name, description, argument-hint, allowed-tools) and
    numbered natural language instructions.

    Args:
        skills_dir: Override for the skills root directory.
            Defaults to ``~/.claude/skills``.
    """

    def __init__(self, skills_dir: str | Path | None = None):
        if skills_dir is None:
            self._skills_root = Path("~/.claude/skills").expanduser().resolve()
        else:
            self._skills_root = Path(skills_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # SOPExportAdapter interface
    # ------------------------------------------------------------------

    def write_sop(self, sop_template: dict) -> Path:
        """Write a single Claude Code skill file from an SOP template."""
        return self.write_procedure(sop_to_procedure(sop_template))

    def write_procedure(self, procedure: dict) -> Path:
        """Write a v3 procedure as ``<skills_root>/<slug>/SKILL.md``."""
        slug = self._slugify(procedure.get("id") or procedure.get("slug") or "unknown")
        skill_dir = self._skills_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        AtomicWriter.write(path, render_packet(procedure, target="claude-skill"))
        return path

    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Write all skill files + AGENTHANDOVER-INDEX.md manifest."""
        paths = [self.write_sop(t) for t in sop_templates]

        if sop_templates:
            self._write_index(sop_templates)
        else:
            index_path = self._skills_root / "AGENTHANDOVER-INDEX.md"
            if index_path.exists():
                index_path.unlink()

        return paths

    def write_metadata(self, metadata_type: str, data: dict) -> Path:
        """Write a metadata JSON file in the skills root."""
        self._skills_root.mkdir(parents=True, exist_ok=True)
        filepath = self._skills_root / f"{metadata_type}.json"
        enriched = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata_type": metadata_type,
            **data,
        }
        AtomicWriter.write(filepath, json.dumps(enriched, indent=2, default=str))
        return filepath

    def get_sops_dir(self) -> Path:
        """Return the skills root directory."""
        return self._skills_root

    def list_sops(self) -> list[dict]:
        """List all exported Claude Code skills with summary info."""
        sops: list[dict] = []
        if not self._skills_root.exists():
            return sops

        for skill_dir in sorted(self._skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            slug = skill_dir.name
            title = ""
            description = ""

            try:
                head = skill_file.read_text(encoding="utf-8")[:4096]
                # Extract name from frontmatter
                fm = self._parse_frontmatter_from_text(head)
                title = fm.get("name", "")
                description = fm.get("description", "")
            except OSError:
                pass

            if not title:
                title = slug.replace("-", " ").title()

            sops.append({
                "slug": slug,
                "title": title,
                "description": description,
                "path": str(skill_file),
                "size_bytes": skill_file.stat().st_size if skill_file.exists() else 0,
            })

        return sops

    # ------------------------------------------------------------------
    # Index generation
    # ------------------------------------------------------------------

    def _write_index(self, sop_templates: list[dict]) -> Path:
        """Write AGENTHANDOVER-INDEX.md listing all exported Claude Code skills."""
        lines: list[str] = []
        lines.append("# AgentHandover Exported Skills")
        lines.append("")
        lines.append(
            f"*Generated by AgentHandover v{__version__} on "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        )
        lines.append("")
        lines.append("Skills exported as Claude Code personal skills.")
        lines.append("Use `/skill-name` in Claude Code to invoke.")
        lines.append("")
        lines.append("| Skill | Description | Confidence | Demos |")
        lines.append("|-------|-------------|------------|-------|")

        for sop in sop_templates:
            slug = self._slugify(sop.get("slug", "unknown"))
            desc = skill_description(sop)
            # Truncate description for table
            if len(desc) > 60:
                desc = desc[:57] + "..."
            confidence = sop.get("confidence_avg", 0.0)
            episodes = sop.get("episode_count", 0)
            link = f"[{slug}]({slug}/SKILL.md)"
            lines.append(f"| {link} | {desc} | {confidence:.2f} | {episodes} |")

        lines.append("")

        index_path = self._skills_root / "AGENTHANDOVER-INDEX.md"
        self._skills_root.mkdir(parents=True, exist_ok=True)
        AtomicWriter.write(index_path, "\n".join(lines))
        return index_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        """Ensure slug is filesystem-safe: lowercase, hyphens, no specials."""
        slug = unicodedata.normalize("NFKD", text)
        slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug[:80] if slug else "unknown"

    @staticmethod
    def _parse_frontmatter_from_text(text: str) -> dict:
        """Parse the YAML frontmatter block at the top of *text*."""
        if not text.startswith("---\n"):
            return {}
        end = text.find("\n---\n", 3)
        if end == -1:
            return {}
        try:
            data = yaml.safe_load(text[4:end])
        except yaml.YAMLError:
            return {}
        return data if isinstance(data, dict) else {}

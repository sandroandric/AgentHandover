"""SKILL.md Export Adapter — clean, agent-readable markdown skill files.

Produces ``SKILL.<slug>.md`` files in a ``skills/`` directory, plus a
``SKILLS-INDEX.md`` manifest table.  The format is intentionally simple
(no YAML frontmatter) so that any AI agent framework can consume it:
Claude skills, OpenClaw, custom agents, etc.

Implements the ``SOPExportAdapter`` ABC.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from oc_apprentice_worker.export_adapter import SOPExportAdapter
from oc_apprentice_worker.exporter import AtomicWriter


_SKILL_SCHEMA_VERSION = "1.1.0"
_OPENMIMIC_VERSION = "0.1.0"


class SkillMdWriter(SOPExportAdapter):
    """Write SOPs as clean SKILL.<slug>.md files for AI agent consumption.

    Args:
        workspace_dir: Root directory under which ``skills/`` is created.
    """

    def __init__(self, workspace_dir: str | Path):
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.skills_dir = self.workspace_dir / "skills"

    def _ensure_dirs(self) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def write_sop(self, sop_template: dict) -> Path:
        """Write a single SKILL.<slug>.md file."""
        self._ensure_dirs()
        slug = self._slugify(sop_template.get("slug", "unknown"))
        content = self._render_skill_md(sop_template)
        path = self.skills_dir / f"SKILL.{slug}.md"
        AtomicWriter.write(path, content)
        return path

    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Write all skill files + SKILLS-INDEX.md manifest."""
        paths = [self.write_sop(t) for t in sop_templates]

        if sop_templates:
            self._write_index(sop_templates)
        else:
            # Remove stale index when no SOPs to export
            index_path = self.skills_dir / "SKILLS-INDEX.md"
            if index_path.exists():
                index_path.unlink()

        return paths

    def write_metadata(self, metadata_type: str, data: dict) -> Path:
        """Write a metadata file (delegates to JSON in skills dir)."""
        import json

        self._ensure_dirs()
        filepath = self.skills_dir / f"{metadata_type}.json"
        enriched = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata_type": metadata_type,
            **data,
        }
        AtomicWriter.write(filepath, json.dumps(enriched, indent=2, default=str))
        return filepath

    def get_sops_dir(self) -> Path:
        """Return the skills directory path."""
        return self.skills_dir

    def list_sops(self) -> list[dict]:
        """List existing skill files with summary info."""
        sops = []
        if not self.skills_dir.exists():
            return sops

        for skill_file in sorted(self.skills_dir.glob("SKILL.*.md")):
            name = skill_file.stem  # e.g. "SKILL.file-expense-report"
            # Extract slug: remove "SKILL." prefix
            slug = name[6:] if name.startswith("SKILL.") else name

            title = ""
            confidence = ""
            last_updated = ""

            try:
                with skill_file.open(encoding="utf-8") as f:
                    head = f.read(2048)
                for line in head.splitlines():
                    if line.startswith("# ") and not title:
                        title = line[2:].strip()
                    if "Confidence:" in line:
                        confidence = line.split("Confidence:")[-1].strip()
                    if "Last updated:" in line:
                        last_updated = line.split("Last updated:")[-1].strip()
            except OSError:
                pass

            # Fallback: derive title from slug
            if not title:
                title = slug.replace("-", " ").title()

            sops.append({
                "slug": slug,
                "title": title,
                "path": str(skill_file),
                "confidence": confidence,
                "last_updated": last_updated,
                "size_bytes": skill_file.stat().st_size if skill_file.exists() else 0,
            })

        return sops

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_skill_md(self, sop: dict) -> str:
        """Render a SOP template as a clean SKILL.md file."""
        lines: list[str] = []
        title = sop.get("title", "Untitled Skill")
        steps = sop.get("steps", [])
        variables = sop.get("variables", [])
        confidence_avg = sop.get("confidence_avg", 0.0)
        episode_count = sop.get("episode_count", 0)
        apps_involved = sop.get("apps_involved", [])
        preconditions = sop.get("preconditions", [])
        task_description = sop.get("task_description")
        execution_overview = sop.get("execution_overview")

        # Title
        lines.append(f"# {title}")
        lines.append("")

        # Task description (from LLM enhancer)
        if task_description:
            lines.append(task_description)
            lines.append("")

        # When to Use
        lines.append("## When to Use")
        if apps_involved:
            for app in apps_involved:
                lines.append(f"- Application: {app}")

        # Extract URL preconditions
        for pre in preconditions:
            if pre.startswith("url_open:"):
                url = pre[len("url_open:"):]
                lines.append(f"- URL pattern: {url}")
            elif pre.startswith("app_open:"):
                app = pre[len("app_open:"):]
                lines.append(f"- Trigger: {app} must be open")

        if not apps_involved and not preconditions:
            lines.append("- General workflow")
        lines.append("")

        # Steps
        lines.append("## Steps")
        lines.append("")
        for i, step in enumerate(steps, 1):
            intent = step.get("step", "action")
            target = step.get("target", "")
            selector = step.get("selector")
            params = step.get("parameters", {})

            # Step header: **Action Target** — description
            action_verb = intent.capitalize()
            if target:
                step_line = f"{i}. **{action_verb} {target}**"
            else:
                step_line = f"{i}. **{action_verb}**"

            # Add parameter summary as description
            if isinstance(params, dict) and params:
                param_desc = ", ".join(
                    f"{k}={v}" for k, v in params.items()
                    if k not in ("app_id", "app")
                )
                if param_desc:
                    step_line += f" — {param_desc}"

            lines.append(step_line)

            # Selector details
            if selector:
                lines.append(f"   - Selector: `{selector}`")
            # Add parameter details
            if isinstance(params, dict):
                for key, value in params.items():
                    if key not in ("app_id", "app"):
                        lines.append(f"   - {key}: `{value}`")
            lines.append("")

        # Input Variables
        if variables:
            lines.append("## Input Variables")
            for var in variables:
                var_name = var.get("name", "unknown")
                var_type = var.get("type", "string")
                example = var.get("example", "")
                default = var.get("default", "")

                line = f"- `{{{{{var_name}}}}}`: {var_type}"
                if example:
                    line += f" (example: {example})"
                if default:
                    line += f" (default: {default})"
                lines.append(line)
            lines.append("")

        # Execution Overview (from LLM enhancer)
        if isinstance(execution_overview, dict) and execution_overview:
            lines.append("## Execution Overview")
            for key, value in execution_overview.items():
                label = key.replace("_", " ").capitalize()
                lines.append(f"- **{label}**: {value}")
            lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append(f"- Source: OpenMimic v{_OPENMIMIC_VERSION}")
        lines.append(f"- Confidence: {confidence_avg:.2f}")
        lines.append(f"- Observed: {episode_count} time(s)")
        lines.append(
            f"- Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )
        lines.append(f"- Schema: {_SKILL_SCHEMA_VERSION}")

        # Add focus recording source if applicable
        source = sop.get("source")
        if source == "focus_recording":
            lines.append("- Mode: Focus recording (single demonstration)")
        lines.append("")

        return "\n".join(lines)

    def _write_index(self, sop_templates: list[dict]) -> Path:
        """Write SKILLS-INDEX.md — a table of all skills."""
        lines: list[str] = []
        lines.append("# Skills Index")
        lines.append("")
        lines.append(
            f"*Generated by OpenMimic v{_OPENMIMIC_VERSION} on "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        )
        lines.append("")
        lines.append("| Slug | Title | Confidence | Episodes | Last Updated |")
        lines.append("|------|-------|------------|----------|--------------|")

        for sop in sop_templates:
            slug = self._slugify(sop.get("slug", "unknown"))
            title = sop.get("title", "Untitled")
            confidence = sop.get("confidence_avg", 0.0)
            episodes = sop.get("episode_count", 0)
            updated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            link = f"[{slug}](SKILL.{slug}.md)"
            lines.append(
                f"| {link} | {title} | {confidence:.2f} | {episodes} | {updated} |"
            )

        lines.append("")

        index_path = self.skills_dir / "SKILLS-INDEX.md"
        AtomicWriter.write(index_path, "\n".join(lines))
        return index_path

    @staticmethod
    def _slugify(text: str) -> str:
        """Ensure slug is filesystem-safe: lowercase, hyphens, no specials."""
        slug = unicodedata.normalize("NFKD", text)
        slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug[:80] if slug else "unknown"

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


_SKILL_SCHEMA_VERSION = "2.0.0"
_OPENMIMIC_VERSION = "0.2.0"


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

    def write_procedure(self, procedure: dict) -> Path:
        """Write a v3 procedure with enriched SKILL.md content.

        Includes v3 fields: environment, constraints, expected_outcomes,
        on_failure per step.
        """
        self._ensure_dirs()
        slug = self._slugify(procedure.get("id", procedure.get("slug", "unknown")))

        # Use the base SOP rendering first
        from oc_apprentice_worker.export_adapter import procedure_to_sop_template
        sop_template = procedure_to_sop_template(procedure)
        content = self._render_skill_md(sop_template)

        # Append v3-only sections
        extra_lines = []

        # Environment section
        env = procedure.get("environment", {})
        if env.get("required_apps") or env.get("accounts") or env.get("setup_actions"):
            extra_lines.append("## Environment")
            for app in env.get("required_apps", []):
                extra_lines.append(f"- Required app: {app}")
            for acct in env.get("accounts", []):
                svc = acct.get("service", "unknown")
                identity = acct.get("identity", "")
                extra_lines.append(f"- Account: {svc}" + (f" ({identity})" if identity else ""))
            for action in env.get("setup_actions", []):
                extra_lines.append(f"- Setup: {action}")
            extra_lines.append("")

        # Constraints section
        constraints = procedure.get("constraints", {})
        trust_level = constraints.get("trust_level", "")
        guardrails = constraints.get("guardrails", [])
        if trust_level or guardrails:
            extra_lines.append("## Constraints")
            if trust_level:
                extra_lines.append(f"- Trust level: {trust_level}")
            for g in guardrails:
                extra_lines.append(f"- {g}")
            extra_lines.append("")

        # Expected Outcomes section
        outcomes = procedure.get("expected_outcomes", [])
        if outcomes:
            extra_lines.append("## Expected Outcomes")
            for o in outcomes:
                if isinstance(o, dict):
                    desc = o.get("description", o.get("type", ""))
                    extra_lines.append(f"- {desc}")
                else:
                    extra_lines.append(f"- {o}")
            extra_lines.append("")

        if extra_lines:
            # Insert before the Metadata section
            meta_marker = "## Metadata"
            if meta_marker in content:
                content = content.replace(meta_marker, "\n".join(extra_lines) + meta_marker)
            else:
                content += "\n" + "\n".join(extra_lines)

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

    def _is_v2_sop(self, sop: dict) -> bool:
        """Detect whether an SOP template was generated by the v2 pipeline.

        v2 SOPs have ``source`` starting with ``v2_`` and steps with
        semantic parameter keys (app, input, verify, location) instead
        of DOM selectors.
        """
        source = sop.get("source", "")
        return str(source).startswith("v2_")

    def _render_skill_md(self, sop: dict) -> str:
        """Render a SOP template as a clean SKILL.md file.

        Dispatches to v2 rendering for VLM-generated SOPs and v1 format
        for legacy PrefixSpan-based SOPs.
        """
        if self._is_v2_sop(sop):
            return self._render_v2_skill_md(sop)
        return self._render_v1_skill_md(sop)

    def _render_v1_skill_md(self, sop: dict) -> str:
        """Render a v1 SOP template (PrefixSpan / DOM-based)."""
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
                if isinstance(var, str):
                    # Legacy format: plain string variable name
                    lines.append(f"- `{{{{{var}}}}}`: text")
                    continue
                var_name = var.get("name", "unknown")
                var_type = var.get("type", "text")
                # Normalise legacy "string" type for display
                if var_type == "string":
                    var_type = "text"
                example = var.get("example", "")
                default = var.get("default", "")

                line = f"- `{{{{{var_name}}}}}`: {var_type}"
                if example:
                    line += f" (example: {example})"
                if default:
                    line += f" (default: {default})"
                if var.get("sensitive", False):
                    line += " — Sensitive — do not log or display"
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

    def _render_v2_skill_md(self, sop: dict) -> str:
        """Render a v2 semantic SOP (from VLM pipeline).

        Produces the rich SKILL.md format with Description, When to Use,
        Prerequisites, semantic Steps (Action/App/Location/Input/Verify),
        Success Criteria, Variables table, Common Errors, and collapsible
        DOM Hints appendix.
        """
        lines: list[str] = []
        title = sop.get("title", "Untitled Skill")
        steps = sop.get("steps", [])
        variables = sop.get("variables", [])
        confidence_avg = sop.get("confidence_avg", 0.0)
        episode_count = sop.get("episode_count", 0)
        apps_involved = sop.get("apps_involved", [])
        preconditions = sop.get("preconditions", [])
        outcome = sop.get("outcome", "")
        task_description = sop.get("task_description", "")
        execution_overview = sop.get("execution_overview", {})
        source = sop.get("source", "")

        # Title
        lines.append(f"# {title}")
        lines.append("")

        # Outcome
        if outcome:
            lines.append("## Outcome")
            lines.append(outcome)
            lines.append("")

        # Description
        if task_description:
            lines.append("## Description")
            lines.append(task_description)
            lines.append("")

        # Before You Start (prerequisites)
        if preconditions:
            lines.append("## Before You Start")
            for pre in preconditions:
                lines.append(f"- {pre}")
            lines.append("")

        # When to Use
        lines.append("## When to Use")
        when_to_use = ""
        if isinstance(execution_overview, dict):
            when_to_use = execution_overview.get("when_to_use", "")
        if when_to_use:
            lines.append(f"- {when_to_use}")
        if apps_involved:
            app_list = ", ".join(apps_involved)
            lines.append(f"- Application: {app_list}")
        if not when_to_use and not apps_involved:
            lines.append("- General workflow")
        lines.append("")

        # Steps — rich v2 format
        lines.append("## Steps")
        lines.append("")
        for i, step in enumerate(steps, 1):
            action = step.get("step", step.get("action", "action"))
            target = step.get("target", "")
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                params = {}

            # Step heading
            step_title = action
            if target:
                step_title = f"{action}"
            lines.append(f"### Step {i}: {step_title}")

            # Action line
            lines.append(f"- **Action**: {action}")

            # App
            app = params.get("app", "")
            if app:
                lines.append(f"- **App**: {app}")

            # Location
            location = params.get("location", "") or target
            if location:
                lines.append(f"- **Location**: `{location}`")

            # Input
            input_val = params.get("input", "")
            if input_val:
                lines.append(f"- **Input**: `{input_val}`")

            # Verify — italicized on its own line for easy scanning
            verify = params.get("verify", "")
            if verify:
                lines.append(f"  _Verify: {verify}_")

            lines.append("")

        # Success Criteria
        success_criteria = ""
        if isinstance(execution_overview, dict):
            success_criteria = execution_overview.get("success_criteria", "")
        if success_criteria:
            lines.append("## Success Criteria")
            # May be semicolon-separated
            for criterion in success_criteria.split(";"):
                criterion = criterion.strip()
                if criterion:
                    lines.append(f"- {criterion}")
            lines.append("")

        # Variables — rich markdown table with type, required, sensitivity
        if variables:
            lines.append("## Variables")
            lines.append("")
            lines.append("| Variable | Type | Required | Description | Example |")
            lines.append("|----------|------|----------|-------------|---------|")
            sensitive_vars: list[str] = []
            for var in variables:
                if isinstance(var, str):
                    # Legacy format: plain string variable name
                    lines.append(
                        f"| `{var}` | text | Yes | — | — |"
                    )
                    continue
                var_name = var.get("name", "unknown")
                var_type = var.get("type", "text")
                # Normalise legacy "string" type for display
                if var_type == "string":
                    var_type = "text"
                var_required = "Yes" if var.get("required", True) else "No"
                var_desc = var.get("description", "") or "—"
                var_example = var.get("example", "") or "—"
                lines.append(
                    f"| `{var_name}` | {var_type} | {var_required} | {var_desc} | {var_example} |"
                )
                if var.get("sensitive", False):
                    sensitive_vars.append(var_name)
            lines.append("")
            # Sensitivity warnings
            for svar in sensitive_vars:
                lines.append(
                    f"> **{svar}**: Sensitive — do not log or display"
                )
            if sensitive_vars:
                lines.append("")

        # Common Errors
        common_errors = ""
        if isinstance(execution_overview, dict):
            common_errors = execution_overview.get("common_errors", "")
        if common_errors:
            lines.append("## Common Errors")
            for err in common_errors.split(";"):
                err = err.strip()
                if err:
                    lines.append(f"- {err}")
            lines.append("")

        # DOM Hints (collapsible appendix)
        timeline = sop.get("_timeline")
        dom_hints = self._collect_dom_hints(steps, timeline=timeline)
        if dom_hints:
            lines.append("## DOM Hints (Optional)")
            lines.append("<details>")
            lines.append("<summary>Browser automation selectors</summary>")
            lines.append("")
            for hint in dom_hints:
                lines.append(f"- {hint}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append(f"- Source: OpenMimic v{_OPENMIMIC_VERSION}")

        # Mode
        if source == "v2_focus_recording":
            lines.append("- Mode: Focus Recording")
        elif source == "v2_passive_discovery":
            lines.append("- Mode: Passive Discovery")

        lines.append(f"- Confidence: {confidence_avg:.2f}")
        lines.append(f"- Observed: {episode_count} demonstration(s)")

        if apps_involved:
            lines.append(f"- Apps: {', '.join(apps_involved)}")

        lines.append(
            f"- Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )
        lines.append(f"- Schema: {_SKILL_SCHEMA_VERSION}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _collect_dom_hints(
        steps: list[dict],
        timeline: list[dict] | None = None,
    ) -> list[str]:
        """Extract DOM selector hints from step selectors AND timeline DOM nodes."""
        hints: list[str] = []

        # Per-step selectors (from sop_generator)
        for i, step in enumerate(steps, 1):
            selector = step.get("selector")
            if selector:
                action = step.get("step", step.get("action", "action"))
                hints.append(f"Step {i} ({action}): `{selector}`")

        # Full-page interactive elements from timeline dom_nodes
        if timeline:
            page_elements = _extract_page_interactive_elements(timeline)
            if page_elements:
                if hints:
                    hints.append("")
                hints.append("**Interactive elements on page:**")
                for elem in page_elements[:20]:  # Cap at 20 elements
                    hints.append(f"`{elem}`")

        return hints

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


def _extract_page_interactive_elements(timeline: list[dict]) -> list[str]:
    """Extract a compact list of interactive elements from timeline DOM nodes.

    Collects unique selectors for buttons, links, inputs, and other
    interactive elements visible during the recording.
    """
    seen: set[str] = set()
    elements: list[str] = []

    for entry in timeline:
        dom_nodes = entry.get("dom_nodes")
        if not dom_nodes or not isinstance(dom_nodes, list):
            continue

        for node in dom_nodes:
            if not isinstance(node, dict):
                continue

            tag = node.get("tag", "").lower()
            role = node.get("role", "").strip()
            aria = node.get("ariaLabel", node.get("aria-label", "")).strip()
            test_id = node.get("testId", node.get("data-testid", "")).strip()
            node_id = node.get("id", "").strip()
            text = node.get("text", node.get("innerText", "")).strip()
            node_type = node.get("type", "").strip()

            # Only interactive elements
            interactive_tags = {"button", "a", "input", "select", "textarea", "label"}
            interactive_roles = {"button", "link", "textbox", "combobox", "menuitem",
                                 "tab", "checkbox", "radio", "switch", "searchbox"}

            if tag not in interactive_tags and role not in interactive_roles:
                continue

            # Build a descriptive selector
            if aria:
                selector = f"[aria-label='{aria}']"
            elif test_id:
                selector = f"[data-testid='{test_id}']"
            elif node_id:
                selector = f"#{node_id}"
            elif text and len(text) < 50:
                selector = f"{tag or '*'}:has-text('{text[:40]}')"
            elif tag and node_type:
                selector = f"{tag}[type='{node_type}']"
            elif tag and role:
                selector = f"{tag}[role='{role}']"
            else:
                continue

            if selector not in seen:
                seen.add(selector)
                elements.append(selector)

    return elements

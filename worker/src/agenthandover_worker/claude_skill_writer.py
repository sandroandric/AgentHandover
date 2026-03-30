"""Claude Code Skill Export Adapter — native Claude Code skill files.

Produces Claude Code-compatible ``SKILL.md`` files with YAML frontmatter
and natural language instructions.  Each skill is written to its own
subdirectory under ``~/.claude/skills/<slug>/SKILL.md``, matching Claude
Code's personal skills directory convention.

This is a **template-based format conversion** (no LLM pass):

- SOP ``slug`` -> ``name`` in YAML frontmatter
- SOP ``task_description`` -> ``description`` in frontmatter
- SOP ``variables`` -> ``argument-hint`` + ``$0``/``$1`` in body
- SOP ``apps_involved`` -> ``allowed-tools`` mapping
- SOP ``steps`` -> numbered natural language instructions
- SOP ``execution_overview`` -> Prerequisites / Success Criteria / Common Errors

Implements the ``SOPExportAdapter`` ABC.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.exporter import AtomicWriter


_AGENTHANDOVER_VERSION = "0.2.0"

# ---- Tool derivation maps ------------------------------------------------

# Apps whose presence implies browser automation (-> Bash for CLI-driven control)
_BROWSER_APPS = frozenset({
    "chrome", "google chrome", "firefox", "safari", "brave", "edge",
    "microsoft edge", "arc", "opera", "vivaldi",
})

# Apps whose presence implies terminal / IDE usage (-> Bash)
_TERMINAL_APPS = frozenset({
    "terminal", "iterm", "iterm2", "warp", "alacritty", "kitty",
    "hyper", "vs code", "vscode", "visual studio code", "intellij",
    "pycharm", "webstorm", "neovim", "vim", "emacs", "sublime text",
    "xcode", "android studio", "cursor",
})

# Apps whose presence implies filesystem operations (-> Read, Write)
_FILESYSTEM_APPS = frozenset({
    "finder", "file explorer", "nautilus", "files",
})


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
        """Write a single Claude Code skill file."""
        slug = self._slugify(sop_template.get("slug", "unknown"))
        content = self._render_skill_md(sop_template)

        skill_dir = self._skills_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        AtomicWriter.write(path, content)
        return path

    def write_procedure(self, procedure: dict) -> Path:
        """Write a v3 procedure as a Claude Code skill with v3 enrichment."""
        slug = self._slugify(procedure.get("id", procedure.get("slug", "unknown")))

        # Use the base SOP rendering first
        from agenthandover_worker.export_adapter import procedure_to_sop_template
        sop_template = procedure_to_sop_template(procedure)
        content = self._render_skill_md(sop_template)

        # Append v3-only sections before the footer
        extra_lines = []

        # Voice & style guidance
        from agenthandover_worker.export_adapter import render_voice_style_section
        extra_lines.extend(render_voice_style_section(procedure))

        # Strategy section (behavioral synthesis)
        strategy = procedure.get("strategy")
        if strategy:
            extra_lines.append("## Strategy")
            extra_lines.append(strategy)
            extra_lines.append("")

        # Evidence summary (pre-synthesis fallback)
        evidence_summary = procedure.get("evidence_summary")
        if evidence_summary and not strategy:
            extra_lines.append("## Observed Patterns")
            extra_lines.append(evidence_summary)
            extra_lines.append("")

        # Selection criteria (behavioral synthesis)
        selection = procedure.get("selection_criteria", [])
        if selection:
            extra_lines.append("## Selection Criteria")
            for sc in selection:
                criterion = sc.get("criterion", "")
                if criterion:
                    conf = sc.get("confidence", 0.0)
                    extra_lines.append(f"- {criterion} (confidence: {conf:.0%})")
                    examples = sc.get("examples", [])
                    for ex in examples[:2]:
                        extra_lines.append(f"  - Example: {ex}")
            extra_lines.append("")

        # Output templates (behavioral synthesis)
        templates = procedure.get("content_templates", [])
        if templates:
            extra_lines.append("## Output Templates")
            for ct in templates:
                template = ct.get("template", "")
                if template:
                    extra_lines.append(f"- Template: {template}")
                    variables = ct.get("variables", [])
                    if variables:
                        extra_lines.append(f"  Variables: {', '.join(variables)}")
            extra_lines.append("")

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

        # Constraints / Do NOT section (guardrails from behavioral synthesis)
        constraints = procedure.get("constraints", {})
        trust_level = constraints.get("trust_level", "")
        guardrails = constraints.get("guardrails", [])
        if guardrails:
            extra_lines.append("## Do NOT")
            if trust_level:
                extra_lines.append(f"- Trust level: {trust_level}")
            for g in guardrails:
                extra_lines.append(f"- {g}")
            extra_lines.append("")
        elif trust_level:
            extra_lines.append("## Constraints")
            extra_lines.append(f"- Trust level: {trust_level}")
            extra_lines.append("")

        # Workflow rhythm (behavioral synthesis)
        rhythm = procedure.get("workflow_rhythm", {})
        if rhythm and rhythm.get("phases"):
            extra_lines.append("## Workflow Rhythm")
            avg_dur = rhythm.get("avg_duration_minutes")
            if avg_dur:
                extra_lines.append(f"- Typical duration: {avg_dur:.0f} minutes")
            for phase in rhythm.get("phases", []):
                name = phase.get("name", "")
                dur = phase.get("typical_duration_minutes", 0)
                desc = phase.get("description", "")
                extra_lines.append(f"- **{name}** (~{dur} min): {desc}")
            extra_lines.append("")

        # MCP browser tools hint for web-based workflows
        apps = procedure.get("apps_involved", [])
        has_browser = any(
            a.lower() in _BROWSER_APPS or "browser" in a.lower() or "chrome" in a.lower()
            for a in apps
        )
        if has_browser:
            extra_lines.append("## Browser Automation")
            extra_lines.append("This workflow involves web browser interaction.")
            extra_lines.append("Consider using MCP browser tools for automation.")
            extra_lines.append("")

        if extra_lines:
            # Insert before the footer (---)
            footer_marker = "\n---\n"
            if footer_marker in content:
                content = content.replace(footer_marker, "\n" + "\n".join(extra_lines) + footer_marker, 1)
            else:
                content += "\n" + "\n".join(extra_lines)

        skill_dir = self._skills_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        AtomicWriter.write(path, content)
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
                head = skill_file.read_text(encoding="utf-8")[:2048]
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
    # Rendering
    # ------------------------------------------------------------------

    def _render_skill_md(self, sop: dict) -> str:
        """Render a complete SKILL.md with YAML frontmatter + body."""
        frontmatter = self._build_frontmatter(sop)
        body = self._build_body(sop)
        return frontmatter + body

    def _build_frontmatter(self, sop: dict) -> str:
        """Build YAML frontmatter block."""
        slug = self._slugify(sop.get("slug", "unknown"))
        description = self._derive_description(sop)
        apps = sop.get("apps_involved", [])
        variables = sop.get("variables", [])
        allowed_tools = self._derive_allowed_tools(apps)

        lines = ["---"]
        lines.append(f"name: {slug}")
        lines.append(f"description: {description}")

        if variables:
            hint_parts = [f"[{v.get('name', 'arg')}]" for v in variables]
            lines.append(f"argument-hint: {' '.join(hint_parts)}")

        lines.append(f"allowed-tools: {', '.join(allowed_tools)}")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _build_body(self, sop: dict) -> str:
        """Build the instruction body (everything after frontmatter)."""
        lines: list[str] = []
        title = sop.get("title", "Untitled Skill")
        steps = sop.get("steps", [])
        variables = sop.get("variables", [])
        preconditions = sop.get("preconditions", [])
        execution_overview = sop.get("execution_overview", {})
        if not isinstance(execution_overview, dict):
            execution_overview = {}
        confidence_avg = sop.get("confidence_avg", 0.0)
        episode_count = sop.get("episode_count", 0)

        # Build variable index map: {{name}} -> $N
        var_map = self._build_variable_map(variables)

        # Opening line — short task summary
        task_desc = sop.get("task_description", "")
        if task_desc:
            lines.append(task_desc)
        else:
            lines.append(title + ".")
        lines.append("")

        # Arguments reference
        if var_map:
            parts = [f"`${i}` = {name}" for name, i in var_map.items()]
            lines.append(f"**Arguments:** {', '.join(parts)}")
            lines.append("")

        # Prerequisites
        prereqs = list(preconditions)
        eo_prereqs = execution_overview.get("prerequisites", "")
        if eo_prereqs:
            for p in eo_prereqs.split(";"):
                p = p.strip()
                if p and p not in prereqs:
                    prereqs.append(p)

        if prereqs:
            lines.append("## Prerequisites")
            for pre in prereqs:
                lines.append(f"- {pre}")
            lines.append("")

        # Steps
        lines.append("## Steps")
        lines.append("")
        for i, step in enumerate(steps, 1):
            rendered = self._render_step(step, var_map, step_num=i)
            lines.append(rendered)
            lines.append("")

        # Success Criteria
        success = execution_overview.get("success_criteria", "")
        if success:
            lines.append("## Success Criteria")
            for criterion in success.split(";"):
                criterion = criterion.strip()
                if criterion:
                    # Substitute variables in criteria text
                    criterion = self._substitute_variables(criterion, var_map)
                    lines.append(f"- {criterion}")
            lines.append("")

        # Common Errors
        errors = execution_overview.get("common_errors", "")
        if errors:
            lines.append("## Common Errors")
            for err in errors.split(";"):
                err = err.strip()
                if err:
                    # Try to split on colon for "Error: description" pattern
                    if ": " in err:
                        label, desc = err.split(": ", 1)
                        lines.append(f"- **{label}**: {desc}")
                    else:
                        lines.append(f"- {err}")
            lines.append("")

        # DOM Hints (selectors from steps or timeline)
        dom_hints = self._collect_dom_hints(steps, sop.get("_timeline"))
        if dom_hints:
            lines.append("## Browser Automation Notes")
            lines.append("<details>")
            lines.append("<summary>DOM selectors observed during recording</summary>")
            lines.append("")
            for hint in dom_hints:
                lines.append(f"- {hint}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            f"*Generated by AgentHandover v{_AGENTHANDOVER_VERSION} "
            f"-- observed {episode_count} demonstration(s), "
            f"confidence: {confidence_avg:.2f}*"
        )
        lines.append("")

        return "\n".join(lines)

    def _render_step(
        self,
        step: dict,
        var_map: dict[str, int],
        *,
        step_num: int = 1,
    ) -> str:
        """Render a single step as a numbered natural language instruction.

        Template logic (no LLM):
        - If step has ``app`` param -> "In **{app}**, ..."
        - If step has ``location`` (URL/path) -> "go to `{location}`" or "open `{location}`"
        - Action + input form the main instruction line
        - If step has ``verify`` -> appended as italicized note
        - ``{{var}}`` placeholders are replaced with ``$N``
        """
        action = step.get("step", step.get("action", "Perform action"))
        params = step.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        app = params.get("app", "")
        location = params.get("location", "") or step.get("target", "")
        input_val = params.get("input", "")
        verify = params.get("verify", "")

        # Build the instruction line
        parts: list[str] = []

        # App prefix
        if app:
            parts.append(f"In **{app}**,")

        # Location
        if location:
            # Substitute variables
            location = self._substitute_variables(location, var_map)
            if location.startswith("http://") or location.startswith("https://"):
                parts.append(f"navigate to `{location}`.")
            else:
                parts.append(f"open `{location}`.")

        # Core action
        action_text = self._substitute_variables(action, var_map)
        if not location:
            # Action is the primary instruction
            parts.append(action_text + ".")
        else:
            # Action supplements the location instruction
            if action_text.lower() not in ("navigate", "open", "go to"):
                parts.append(action_text + ".")

        # Input
        if input_val:
            input_text = self._substitute_variables(input_val, var_map)
            parts.append(f"Enter `{input_text}`.")

        main_line = f"{step_num}. " + " ".join(parts)

        # Verification note
        if verify:
            verify_text = self._substitute_variables(verify, var_map)
            main_line += f"\n   _Verify: {verify_text}_"

        return main_line

    # ------------------------------------------------------------------
    # Tool derivation
    # ------------------------------------------------------------------

    def _derive_allowed_tools(self, apps: list[str]) -> list[str]:
        """Map apps_involved to Claude Code tool names.

        Logic:
        - Browser apps -> Bash (for CLI browser automation)
        - Terminal / IDE apps -> Bash
        - Filesystem apps -> Read, Write
        - Mixed / unknown -> Bash, Read, Write, Grep
        - Default (no apps): Bash, Read
        """
        if not apps:
            return ["Bash", "Read"]

        tools: set[str] = set()
        has_known = False

        for app in apps:
            app_lower = app.lower().strip()

            if app_lower in _BROWSER_APPS:
                tools.add("Bash")
                has_known = True
            elif app_lower in _TERMINAL_APPS:
                tools.add("Bash")
                has_known = True
            elif app_lower in _FILESYSTEM_APPS:
                tools.update({"Read", "Write"})
                has_known = True
            else:
                # Check partial matches for common patterns
                if any(b in app_lower for b in ("terminal", "shell", "console", "code", "ide")):
                    tools.add("Bash")
                    has_known = True
                elif any(b in app_lower for b in ("browser", "chrome", "firefox", "safari")):
                    tools.add("Bash")
                    has_known = True

        if not has_known:
            # Unknown apps: provide broad tool access
            return ["Bash", "Read", "Write", "Grep"]

        # Ensure at least Bash + Read for any known combo
        tools.add("Bash")
        if not tools.intersection({"Read", "Write"}):
            tools.add("Read")

        # Stable ordering
        order = ["Bash", "Read", "Write", "Grep", "Glob", "Edit"]
        return [t for t in order if t in tools]

    # ------------------------------------------------------------------
    # Variable handling
    # ------------------------------------------------------------------

    def _build_variable_map(self, variables: list[dict]) -> dict[str, int]:
        """Build a mapping from variable name to positional index.

        Returns:
            dict mapping variable name -> integer index (0-based)
        """
        var_map: dict[str, int] = {}
        for i, var in enumerate(variables):
            name = var.get("name", f"arg{i}")
            var_map[name] = i
        return var_map

    def _substitute_variables(self, text: str, var_map: dict[str, int]) -> str:
        """Replace ``{{variable_name}}`` with ``$N`` based on var_map."""
        if not var_map or "{{" not in text:
            return text

        for name, idx in var_map.items():
            text = text.replace(f"{{{{{name}}}}}", f"${idx}")

        return text

    # ------------------------------------------------------------------
    # Description derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_description(sop: dict) -> str:
        """Derive a concise description for the frontmatter.

        Priority:
        1. task_description (first sentence)
        2. execution_overview.when_to_use (first sentence)
        3. title
        """
        task_desc = sop.get("task_description", "")
        if task_desc:
            # Take first sentence
            first = task_desc.split(".")[0].strip()
            if first:
                return first + "."

        eo = sop.get("execution_overview", {})
        if isinstance(eo, dict):
            when = eo.get("when_to_use", "")
            if when:
                first = when.split(".")[0].strip()
                if first:
                    return first + "."

        title = sop.get("title", "Untitled Skill")
        return title + "."

    # ------------------------------------------------------------------
    # DOM hints collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_dom_hints(
        steps: list[dict],
        timeline: list[dict] | None = None,
    ) -> list[str]:
        """Extract DOM selector hints from step selectors and timeline."""
        hints: list[str] = []

        for i, step in enumerate(steps, 1):
            selector = step.get("selector")
            if selector:
                action = step.get("step", step.get("action", "action"))
                hints.append(f"Step {i} ({action}): `{selector}`")

        if timeline and isinstance(timeline, list):
            page_elements = _extract_interactive_elements(timeline)
            if page_elements:
                if hints:
                    hints.append("")
                hints.append("**Interactive elements on page:**")
                for elem in page_elements[:20]:
                    hints.append(f"`{elem}`")

        return hints

    # ------------------------------------------------------------------
    # Index generation
    # ------------------------------------------------------------------

    def _write_index(self, sop_templates: list[dict]) -> Path:
        """Write AGENTHANDOVER-INDEX.md listing all exported Claude Code skills."""
        lines: list[str] = []
        lines.append("# AgentHandover Exported Skills")
        lines.append("")
        lines.append(
            f"*Generated by AgentHandover v{_AGENTHANDOVER_VERSION} on "
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
            desc = self._derive_description(sop)
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
    def _parse_frontmatter_from_text(text: str) -> dict[str, str]:
        """Parse simple YAML frontmatter from text.

        Returns a flat dict of key: value pairs.
        Only handles single-line values (sufficient for our format).
        """
        result: dict[str, str] = {}
        in_frontmatter = False

        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    break  # End of frontmatter
            if in_frontmatter and ": " in stripped:
                key, _, value = stripped.partition(": ")
                result[key.strip()] = value.strip()

        return result


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_interactive_elements(timeline: list[dict]) -> list[str]:
    """Extract a compact list of interactive elements from timeline DOM nodes.

    Mirrors the logic in skill_md_writer but kept here to avoid coupling.
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

            interactive_tags = {"button", "a", "input", "select", "textarea", "label"}
            interactive_roles = {
                "button", "link", "textbox", "combobox", "menuitem",
                "tab", "checkbox", "radio", "switch", "searchbox",
            }

            if tag not in interactive_tags and role not in interactive_roles:
                continue

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

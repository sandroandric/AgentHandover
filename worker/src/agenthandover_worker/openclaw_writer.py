"""OpenClaw Integration Writer — write SOPs to the OpenClaw workspace.

Implements section 11 of the AgentHandover spec.  Writes learned SOPs to
``~/.openclaw/workspace/memory/apprentice/sops/`` where OpenClaw agents
can discover and execute them.

Learning-only policy: this module only writes to the ``memory/apprentice/``
subtree.  It never registers action tools or executes commands.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.exporter import AtomicWriter, IndexGenerator, SOPExporter
from agenthandover_worker.sop_format import SOPFormatter
from agenthandover_worker.sop_versioner import SOPVersioner

OPENCLAW_WORKSPACE = Path.home() / ".openclaw" / "workspace"
APPRENTICE_DIR = OPENCLAW_WORKSPACE / "memory" / "apprentice"
SOPS_DIR = APPRENTICE_DIR / "sops"
METADATA_DIR = APPRENTICE_DIR / "metadata"


class OpenClawWriter(SOPExportAdapter):
    """Write SOPs to the OpenClaw workspace.

    Learning-only policy: only writes to ``memory/apprentice/`` subtree.
    Never registers action tools or executes commands.

    The writer manages the full directory structure:
    - ``sops/`` — canonical SOP files
    - ``sops/archive/`` — archived old versions
    - ``metadata/`` — confidence logs, episode stats, etc.
    """

    def __init__(self, workspace_dir: str | Path | None = None):
        if workspace_dir:
            self.workspace = Path(workspace_dir)
        else:
            self.workspace = OPENCLAW_WORKSPACE
        self.apprentice_dir = self.workspace / "memory" / "apprentice"
        self.sops_dir = self.apprentice_dir / "sops"
        self.metadata_dir = self.apprentice_dir / "metadata"

        # Build internal pipeline components
        self.formatter = SOPFormatter()
        self.versioner = SOPVersioner(
            sops_dir=self.sops_dir,
            archive_dir=self.sops_dir / "archive",
        )
        self.exporter = SOPExporter(self.apprentice_dir)
        self.exporter.formatter = self.formatter
        self.exporter.versioner = self.versioner

    def ensure_directory_structure(self) -> None:
        """Create the OpenClaw workspace directory structure.

        Creates:
        - memory/apprentice/sops/
        - memory/apprentice/sops/archive/
        - memory/apprentice/metadata/
        """
        self.sops_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        (self.sops_dir / "archive").mkdir(exist_ok=True)

    def write_sop(self, sop_template: dict) -> Path:
        """Write a single SOP to the OpenClaw workspace.

        Ensures directory structure exists, then uses the full export
        pipeline (format, version, atomic write, index update).

        Returns:
            Path to the written SOP file.
        """
        self.ensure_directory_structure()
        return self.exporter.export_sop(sop_template)

    def write_procedure(self, procedure: dict) -> Path:
        """Write a v3 procedure with enriched OpenClaw content.

        Includes v3 fields: environment, constraints, expected_outcomes,
        staleness, evidence summary, chain metadata.
        """
        from agenthandover_worker.export_adapter import procedure_to_sop_template

        self.ensure_directory_structure()

        # Render base SOP markdown
        sop_template = procedure_to_sop_template(procedure)
        md_content = self.formatter.format_sop(sop_template)

        # Append v3-only sections
        extra_lines: list[str] = []

        # Voice & style guidance (tells agent HOW to write)
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

        # Selection criteria
        selection = procedure.get("selection_criteria", [])
        if selection:
            extra_lines.append("## Selection Criteria")
            for sc in selection:
                criterion = sc.get("criterion", "")
                if criterion:
                    extra_lines.append(f"- {criterion}")
            extra_lines.append("")

        # Content templates
        templates = procedure.get("content_templates", [])
        if templates:
            extra_lines.append("## Content Templates")
            for ct in templates:
                template = ct.get("template", "")
                if template:
                    extra_lines.append(f"- {template}")
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

        # Browser automation hints for web-based steps
        apps = procedure.get("apps_involved", [])
        has_browser = any(
            a.lower() in ("chrome", "google chrome", "firefox", "safari",
                          "brave", "edge", "arc")
            for a in apps
        )
        if has_browser:
            extra_lines.append("## Execution Hints")
            extra_lines.append("- This workflow involves web browser interaction")
            extra_lines.append("- Consider native browser automation for web steps")
            # Check for API alternatives
            urls = set()
            for step in procedure.get("steps", []):
                loc = step.get("location", "")
                if "api." in loc or "/api/" in loc:
                    urls.add(loc)
            if urls:
                extra_lines.append("- API endpoints detected (may be faster than browser):")
                for url in sorted(urls)[:5]:
                    extra_lines.append(f"  - {url}")
            extra_lines.append("")

        # Constraints section with guardrails
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

        # Staleness section
        staleness = procedure.get("staleness", {})
        last_observed = staleness.get("last_observed")
        last_confirmed = staleness.get("last_confirmed")
        if last_observed or last_confirmed:
            extra_lines.append("## Staleness")
            if last_observed:
                extra_lines.append(f"- Last observed: {last_observed}")
            if last_confirmed:
                extra_lines.append(f"- Last confirmed: {last_confirmed}")
            drift = staleness.get("drift_signals", [])
            if drift:
                extra_lines.append(f"- Drift signals: {len(drift)}")
            extra_lines.append("")

        # Evidence summary
        evidence = procedure.get("evidence", {})
        total_obs = evidence.get("total_observations", 0)
        contradictions = evidence.get("contradictions", [])
        if total_obs or contradictions:
            extra_lines.append("## Evidence")
            extra_lines.append(f"- Total observations: {total_obs}")
            if contradictions:
                extra_lines.append(f"- Contradictions: {len(contradictions)}")
            extra_lines.append("")

        # Chain metadata
        chain = procedure.get("chain", {})
        depends_on = chain.get("depends_on", [])
        followed_by = chain.get("followed_by", [])
        if depends_on or followed_by:
            extra_lines.append("## Chain")
            if depends_on:
                extra_lines.append(f"- Depends on: {', '.join(depends_on)}")
            if followed_by:
                extra_lines.append(f"- Followed by: {', '.join(followed_by)}")
            extra_lines.append("")

        # Credential references for authenticated workflows
        inputs = procedure.get("inputs", [])
        cred_inputs = [i for i in inputs if i.get("credential")]
        if cred_inputs:
            extra_lines.append("## Credentials")
            for ci in cred_inputs:
                extra_lines.append(
                    f"- {ci.get('name', '?')}: requires credential "
                    f"(type: {ci.get('type', 'text')})"
                )
            extra_lines.append("")

        if extra_lines:
            md_content += "\n" + "\n".join(extra_lines)

        # Write using atomic writer + versioner
        slug = sop_template.get("slug", "unknown")
        filepath = self.sops_dir / f"sop.{slug}.md"

        # Archive previous version if it exists
        if filepath.exists():
            self.versioner.archive_sop(filepath)

        AtomicWriter.write(filepath, md_content)

        # Also write v3 JSON sidecar for machine consumption
        json_path = self.sops_dir / f"sop.{slug}.v3.json"
        AtomicWriter.write(
            json_path,
            json.dumps(procedure, indent=2, default=str),
        )

        return filepath

    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Write multiple SOPs and update the index.

        Returns:
            List of paths to all written SOP files.
        """
        self.ensure_directory_structure()
        return self.exporter.export_all(sop_templates)

    def write_metadata(self, metadata_type: str, data: dict) -> Path:
        """Write a metadata file (confidence_log, episode_stats, etc.).

        Metadata files are written atomically as JSON to the metadata
        directory with the naming convention ``<type>.json``.

        Args:
            metadata_type: Name for the metadata file (e.g. "confidence_log",
                "episode_stats", "induction_report").
            data: Dictionary to serialize as JSON.

        Returns:
            Path to the written metadata file.
        """
        self.ensure_directory_structure()

        # Add timestamp to data
        enriched = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata_type": metadata_type,
            **data,
        }

        filepath = self.metadata_dir / f"{metadata_type}.json"
        content = json.dumps(enriched, indent=2, default=str)
        AtomicWriter.write(filepath, content)
        return filepath

    def get_sops_dir(self) -> Path:
        """Return the SOPs directory path."""
        return self.sops_dir

    def list_sops(self) -> list[dict]:
        """List all SOPs in the workspace with summary info.

        Scans the sops directory for .md files matching the SOP naming
        convention (sop.*.md) and extracts frontmatter metadata.
        """
        sops = []
        if not self.sops_dir.exists():
            return sops

        for sop_file in sorted(self.sops_dir.glob("sop.*.md")):
            # Extract slug from filename: sop.<slug>.md
            name = sop_file.stem  # "sop.<slug>"
            parts = name.split(".", 1)
            slug = parts[1] if len(parts) > 1 else name

            # Read file to extract title from first heading
            title = slug.replace("-", " ").title()
            try:
                with sop_file.open(encoding="utf-8") as f:
                    head = f.read(1024)
                for line in head.splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            except OSError:
                pass

            sops.append({
                "slug": slug,
                "title": title,
                "path": str(sop_file),
                "size_bytes": sop_file.stat().st_size if sop_file.exists() else 0,
            })

        return sops

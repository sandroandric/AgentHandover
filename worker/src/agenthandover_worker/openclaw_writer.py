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
from agenthandover_worker.exporter import AtomicWriter, IndexGenerator
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.skill_render import render_packet
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

        # Manual-edit detection + archiving of replaced versions
        self.formatter = SOPFormatter()
        self.versioner = SOPVersioner(
            sops_dir=self.sops_dir,
            archive_dir=self.sops_dir / "archive",
        )
        self.index_generator = IndexGenerator()

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
        """Write a single SOP template (converted to a procedure first)."""
        path = self._write(sop_to_procedure(sop_template))
        self.index_generator.update_index(self.sops_dir, [])
        return path

    def write_procedure(self, procedure: dict) -> Path:
        """Write a v3 procedure and refresh the catalogue.

        The markdown is ``render_packet(procedure, target="openclaw")``;
        a ``sop.<slug>.v3.json`` sidecar carries the procedure itself.
        A file the user edited by hand is kept and the new version is
        written next to it as ``sop.<slug>.v2_draft.md``.
        """
        path = self._write(procedure)
        self.index_generator.update_index(self.sops_dir, [])
        return path

    def _write(self, procedure: dict) -> Path:
        self.ensure_directory_structure()
        slug = str(procedure.get("id") or procedure.get("slug") or "unknown")
        path = self.versioner.write_sop(
            slug, render_packet(procedure, target="openclaw"), self.formatter,
        )
        AtomicWriter.write(
            self.sops_dir / f"sop.{slug}.v3.json",
            json.dumps(procedure, indent=2, default=str),
        )
        return path

    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Write multiple SOP templates and refresh the catalogue once."""
        paths = [self._write(sop_to_procedure(t)) for t in sop_templates]
        self.index_generator.update_index(self.sops_dir, [])
        return paths

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

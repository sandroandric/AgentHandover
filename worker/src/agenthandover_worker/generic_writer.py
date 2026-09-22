"""Generic filesystem SOP export adapter.

Writes SOPs as both Markdown (.md) and JSON (.json) files to any
specified directory. Does not assume any specific agent workspace
structure -- suitable for standalone or custom integrations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.exporter import AtomicWriter
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.skill_render import render_packet


class GenericWriter(SOPExportAdapter):
    """Write SOPs as .md + .json to a configurable directory.

    Args:
        output_dir: Directory where SOPs will be written.
        json_export: If True, also write a .json version of each SOP.
    """

    def __init__(self, output_dir: str | Path, json_export: bool = True):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.sops_dir_path = self.output_dir / "sops"
        self.sops_dir = self.sops_dir_path
        self.metadata_dir = self.output_dir / "metadata"
        self.json_export = json_export

    def _ensure_dirs(self) -> None:
        self.sops_dir_path.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def write_sop(self, sop_template: dict) -> Path:
        """Write a single SOP template (converted to a procedure first)."""
        return self.write_procedure(sop_to_procedure(sop_template))

    def write_procedure(self, procedure: dict) -> Path:
        """Write ``sop.<slug>.md`` (and ``sop.<slug>.v3.json`` when enabled).

        The markdown is ``render_packet(procedure, target="generic")``;
        the JSON is the procedure itself, for machine consumption.
        """
        self._ensure_dirs()
        slug = str(procedure.get("id") or procedure.get("slug") or "unknown")

        md_path = self.sops_dir_path / f"sop.{slug}.md"
        AtomicWriter.write(md_path, render_packet(procedure, target="generic"))

        if self.json_export:
            AtomicWriter.write(
                self.sops_dir_path / f"sop.{slug}.v3.json",
                json.dumps(procedure, indent=2, default=str),
            )

        return md_path

    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Write multiple SOPs and return paths to all written files."""
        return [self.write_sop(t) for t in sop_templates]

    def write_metadata(self, metadata_type: str, data: dict) -> Path:
        """Write a metadata JSON file."""
        self._ensure_dirs()
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
        return self.sops_dir_path

    def list_sops(self) -> list[dict]:
        """List all SOPs with summary info."""
        sops = []
        if not self.sops_dir_path.exists():
            return sops

        for sop_file in sorted(self.sops_dir_path.glob("sop.*.md")):
            name = sop_file.stem
            parts = name.split(".", 1)
            slug = parts[1] if len(parts) > 1 else name

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

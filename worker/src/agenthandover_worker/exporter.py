"""Atomic SOP Exporter — safe file writes and index catalog generation.

Implements sections 10.4 and 10.5 of the AgentHandover spec.  Provides:
- ``AtomicWriter``: crash-safe file writes via temp + fsync + rename
- ``IndexGenerator``: generates ``index.md`` catalog of all SOPs
- ``SOPExporter``: orchestrates the full export pipeline
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agenthandover_worker.sop_format import SOPFormatter
    from agenthandover_worker.sop_versioner import SOPVersioner


class AtomicWriter:
    """Atomically write content to a file path.

    Uses the classic temp-file + fsync + rename pattern to ensure that
    readers never see a partially written file.  If the write fails at
    any point, the temp file is cleaned up and the original file (if any)
    remains untouched.
    """

    @staticmethod
    def write(filepath: str | Path, content: str) -> None:
        """Atomically write *content* to *filepath*.

        1. Write to a temp file in the same directory
        2. Flush + fsync the temp file
        3. Atomic rename to the target path

        If the parent directory does not exist it will be created.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            dir=str(filepath.parent),
            prefix=".tmp_",
            suffix=".md",
        )
        renamed = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.rename(temp_path, str(filepath))
            renamed = True
        except Exception:
            raise
        finally:
            if not renamed:
                try:
                    os.unlink(temp_path)
                except OSError as cleanup_err:
                    logger.warning(
                        "Failed to clean up temp file %s: %s",
                        temp_path,
                        cleanup_err,
                    )


class IndexGenerator:
    """Generate an ``index.md`` catalog of all SOP files.

    The index contains a header with summary statistics, a markdown table
    listing all SOPs, and detailed per-SOP entries.
    """

    def generate_index(self, sops_dir: Path, sop_entries: list[dict]) -> str:
        """Generate index.md catalog content.

        Args:
            sops_dir: directory containing SOP files
            sop_entries: list of SOP template dicts with slug, title,
                confidence_avg, apps_involved, episode_count keys

        Returns:
            Markdown string for the index file.
        """
        now = datetime.now(timezone.utc).isoformat()
        total = len(sop_entries)

        if total > 0:
            avg_confidence = sum(
                e.get("confidence_avg", 0.0) for e in sop_entries
            ) / total
        else:
            avg_confidence = 0.0

        lines: list[str] = []
        lines.append("# SOP Index")
        lines.append("")
        lines.append(f"**Last updated:** {now}")
        lines.append(f"**Total SOPs:** {total}")
        lines.append(f"**Average confidence:** {avg_confidence:.2f}")
        lines.append("")

        if sop_entries:
            # Summary table
            lines.append("## Summary")
            lines.append("")
            lines.append("| Slug | Title | Confidence | Episodes | Apps |")
            lines.append("|------|-------|------------|----------|------|")
            for entry in sorted(sop_entries, key=lambda e: e.get("slug", "")):
                slug = entry.get("slug", "unknown")
                title = entry.get("title", "Untitled")
                conf = entry.get("confidence_avg", 0.0)
                ep_count = entry.get("episode_count", 0)
                apps = ", ".join(entry.get("apps_involved", []))
                lines.append(
                    f"| `{slug}` | {title} | {conf:.2f} | {ep_count} | {apps} |"
                )
            lines.append("")

            # Detailed entries
            lines.append("## Details")
            lines.append("")
            for entry in sorted(sop_entries, key=lambda e: e.get("slug", "")):
                slug = entry.get("slug", "unknown")
                title = entry.get("title", "Untitled")
                conf = entry.get("confidence_avg", 0.0)
                ep_count = entry.get("episode_count", 0)
                apps = entry.get("apps_involved", [])
                steps = entry.get("steps", [])
                variables = entry.get("variables", [])

                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"- **Slug:** `{slug}`")
                lines.append(f"- **File:** `sop.{slug}.md`")
                lines.append(f"- **Confidence:** {conf:.2f}")
                lines.append(f"- **Episodes observed:** {ep_count}")
                if apps:
                    lines.append(f"- **Apps:** {', '.join(apps)}")
                lines.append(f"- **Steps:** {len(steps)}")
                lines.append(f"- **Last learned:** {now}")

                # Required inputs from variables
                if variables:
                    inputs_parts = []
                    for var in variables:
                        var_name = var.get("name", "unknown")
                        var_type = var.get("type", "string")
                        inputs_parts.append(f"`{var_name}` ({var_type})")
                    lines.append(f"- **Required inputs:** {', '.join(inputs_parts)}")

                lines.append("")

        return "\n".join(lines)

    def update_index(self, sops_dir: Path, sop_entries: list[dict]) -> None:
        """Atomically write the index.md file.

        The index file is placed in the parent of ``sops_dir``
        (i.e. alongside the ``sops/`` directory).
        """
        content = self.generate_index(sops_dir, sop_entries)
        AtomicWriter.write(sops_dir.parent / "index.md", content)


class SOPExporter:
    """Orchestrate SOP export with atomic writes and index maintenance.

    Combines formatting, versioning, atomic writing, and index generation
    into a single export pipeline.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.sops_dir = self.base_dir / "sops"
        self.writer = AtomicWriter()
        self.index_gen = IndexGenerator()
        self.versioner: SOPVersioner | None = None  # Set externally
        self.formatter: SOPFormatter | None = None  # Set externally

    def export_sop(self, sop_template: dict) -> Path:
        """Export a single SOP: format, version, atomic write, update index.

        Returns:
            Path to the written SOP file.
        """
        if self.formatter is None:
            raise RuntimeError("SOPExporter.formatter must be set before export")
        if self.versioner is None:
            raise RuntimeError("SOPExporter.versioner must be set before export")

        # Format the SOP content
        content = self.formatter.format_sop(sop_template)
        slug = sop_template.get("slug", "unknown")

        # Write with versioning (handles archiving / draft detection)
        result_path = self.versioner.write_sop(slug, content, self.formatter)

        # Update index with this single SOP
        self.index_gen.update_index(self.sops_dir, [sop_template])

        return result_path

    def export_all(self, sop_templates: list[dict]) -> list[Path]:
        """Export multiple SOPs and update the index.

        Returns:
            List of paths to all written SOP files.
        """
        if self.formatter is None:
            raise RuntimeError("SOPExporter.formatter must be set before export")
        if self.versioner is None:
            raise RuntimeError("SOPExporter.versioner must be set before export")

        paths: list[Path] = []
        for template in sop_templates:
            content = self.formatter.format_sop(template)
            slug = template.get("slug", "unknown")
            result_path = self.versioner.write_sop(slug, content, self.formatter)
            paths.append(result_path)

        # Update index with all SOPs
        self.index_gen.update_index(self.sops_dir, sop_templates)

        return paths

"""SOP Versioner — drift detection, archiving, and canonical SOP management.

Implements section 10.2 of the OpenMimic spec.  Maintains a single canonical
SOP file per slug, archives old versions when updating, and creates draft files
when manual edits are detected (to avoid overwriting human improvements).
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from oc_apprentice_worker.exporter import AtomicWriter

if TYPE_CHECKING:
    from oc_apprentice_worker.sop_format import SOPFormatter


class SOPVersioner:
    """Manage SOP versioning with drift detection and archival.

    Canonical SOP files live at ``sops/sop.<slug>.md``.  When a new version
    is generated and the old file has not been manually edited, the old
    file is archived and the new one replaces it.  If the old file was
    manually edited, the new version is written as ``sop.<slug>.v2_draft.md``
    so the human can merge changes.
    """

    def __init__(
        self,
        sops_dir: str | Path,
        archive_dir: str | Path | None = None,
    ):
        self.sops_dir = Path(sops_dir)
        self.archive_dir = (
            Path(archive_dir) if archive_dir else self.sops_dir / "archive"
        )

    def write_sop(self, slug: str, content: str, formatter: SOPFormatter) -> Path:
        """Write a SOP file with versioning.

        If no existing file: write directly.
        If existing file not manually edited: archive old, write new.
        If existing file manually edited: write as v2_draft, don't overwrite.

        Returns:
            Path to the written file.
        """
        self.sops_dir.mkdir(parents=True, exist_ok=True)

        canonical = self.get_canonical_path(slug)

        if not canonical.exists():
            # No existing file — write atomically
            AtomicWriter.write(canonical, content)
            return canonical

        # Existing file found — check for manual edits
        was_edited, reason = formatter.detect_manual_edit(str(canonical))

        if was_edited:
            # Manual edit detected — write as draft, don't overwrite
            draft = self.get_draft_path(slug)
            AtomicWriter.write(draft, content)
            return draft

        # Not manually edited — archive old, write new atomically
        self.archive_sop(canonical)
        AtomicWriter.write(canonical, content)
        return canonical

    def archive_sop(self, filepath: Path) -> Path:
        """Move old SOP to archive with timestamp and hash.

        Archive format: ``archive/sop.<slug>.<timestamp>.<hash_short>.md``

        The hash is computed from the file content to ensure unique archive
        names even when archiving multiple versions in quick succession.

        Returns:
            Path to the archived file.
        """
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        content = filepath.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        # Extract stem: sop.my_slug.md -> sop.my_slug
        stem = filepath.stem  # sop.my_slug

        archive_name = f"{stem}.{timestamp}.{content_hash}.md"
        archive_path = self.archive_dir / archive_name

        shutil.move(str(filepath), str(archive_path))
        return archive_path

    def get_canonical_path(self, slug: str) -> Path:
        """Get the canonical path for a SOP: sops/sop.<slug>.md"""
        return self.sops_dir / f"sop.{slug}.md"

    def get_draft_path(self, slug: str) -> Path:
        """Get the draft path: sops/sop.<slug>.v2_draft.md"""
        return self.sops_dir / f"sop.{slug}.v2_draft.md"

    def list_versions(self, slug: str) -> list[Path]:
        """List all archived versions of a SOP, sorted by name (chronological).

        Returns paths matching ``archive/sop.<slug>.*`` pattern.
        """
        if not self.archive_dir.exists():
            return []

        pattern = f"sop.{slug}.*.md"
        versions = sorted(self.archive_dir.glob(pattern))
        return versions

"""Knowledge base synchronization for OpenMimic.

Provides export/import of the knowledge base as zip bundles, and
directory-to-directory sync with conflict detection.

Export bundles are standard zip files containing all KB JSON files.
Sync state is tracked at ``{kb_root}/.sync_state.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import logging
import shutil
import zipfile

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class SyncManifest:
    """Manifest listing all files in a KB with checksums."""

    files: dict[str, dict]  # rel_path -> {"checksum": sha256, "size": bytes, "modified": iso}
    generated_at: str
    machine_id: str


@dataclass
class SyncDiff:
    """Differences between a local and remote manifest."""

    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged: list[str]


@dataclass
class SyncResult:
    """Result of a sync operation."""

    status: str  # "success", "partial", "error"
    files_pulled: int
    files_pushed: int
    conflicts: list[str]
    errors: list[str]


class KBSync:
    """Export, import, and sync the knowledge base."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        machine_id: str | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._machine_id = machine_id or "local"

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def build_manifest(self) -> SyncManifest:
        """Build a manifest of all files in the knowledge base.

        Scans the KB root recursively and records SHA-256 checksum,
        size, and modification time for each file.
        """
        files: dict[str, dict] = {}
        root = self._kb.root

        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                # Skip hidden temp files and sync state
                if path.name.startswith(".kb_") and path.name.endswith(".tmp"):
                    continue
                rel = str(path.relative_to(root))
                checksum = self._file_checksum(path)
                stat = path.stat()
                files[rel] = {
                    "checksum": checksum,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }

        return SyncManifest(
            files=files,
            generated_at=datetime.now(timezone.utc).isoformat(),
            machine_id=self._machine_id,
        )

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def compute_diff(self, remote_manifest: SyncManifest) -> SyncDiff:
        """Compute what changed between local and a remote manifest.

        - **added**: files in remote but not local.
        - **modified**: files in both but with different checksums.
        - **deleted**: files in local but not remote.
        - **unchanged**: files with matching checksums.
        """
        local_manifest = self.build_manifest()

        local_files = set(local_manifest.files.keys())
        remote_files = set(remote_manifest.files.keys())

        added = sorted(remote_files - local_files)
        deleted = sorted(local_files - remote_files)
        unchanged: list[str] = []
        modified: list[str] = []

        for rel in sorted(local_files & remote_files):
            if (
                local_manifest.files[rel]["checksum"]
                == remote_manifest.files[rel]["checksum"]
            ):
                unchanged.append(rel)
            else:
                modified.append(rel)

        return SyncDiff(
            added=added,
            modified=modified,
            deleted=deleted,
            unchanged=unchanged,
        )

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_bundle(self, output_path: Path) -> Path:
        """Export the entire KB as a zip bundle.

        Returns the path to the created zip file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If output_path is a directory, generate a filename
        if output_path.is_dir():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"kb_export_{ts}.zip"

        # Ensure .zip extension
        if output_path.suffix != ".zip":
            output_path = output_path.with_suffix(".zip")

        root = self._kb.root

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if root.is_dir():
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    if path.name.startswith(".kb_") and path.name.endswith(".tmp"):
                        continue
                    rel = str(path.relative_to(root))
                    zf.write(path, rel)

            # Include manifest
            manifest = self.build_manifest()
            manifest_json = json.dumps(
                {
                    "files": manifest.files,
                    "generated_at": manifest.generated_at,
                    "machine_id": manifest.machine_id,
                },
                indent=2,
            )
            zf.writestr("_manifest.json", manifest_json)

        logger.info("KB exported to %s", output_path)
        return output_path

    def import_bundle(
        self, bundle_path: Path, strategy: str = "merge"
    ) -> SyncResult:
        """Import a KB zip bundle.

        Strategies:
        - ``"merge"``: add new files, update modified files, keep local-only files.
        - ``"replace"``: replace the entire KB with the bundle contents.

        Returns a :class:`SyncResult`.
        """
        bundle_path = Path(bundle_path)
        if not bundle_path.is_file():
            return SyncResult(
                status="error",
                files_pulled=0,
                files_pushed=0,
                conflicts=[],
                errors=[f"Bundle not found: {bundle_path}"],
            )

        root = self._kb.root
        root.mkdir(parents=True, exist_ok=True)

        files_pulled = 0
        conflicts: list[str] = []
        errors: list[str] = []

        try:
            with zipfile.ZipFile(bundle_path, "r") as zf:
                # Read manifest if present
                remote_manifest = None
                if "_manifest.json" in zf.namelist():
                    manifest_data = json.loads(zf.read("_manifest.json"))
                    remote_manifest = SyncManifest(
                        files=manifest_data.get("files", {}),
                        generated_at=manifest_data.get("generated_at", ""),
                        machine_id=manifest_data.get("machine_id", "unknown"),
                    )

                if strategy == "replace":
                    # Remove all existing files first
                    if root.is_dir():
                        for existing in root.rglob("*"):
                            if existing.is_file():
                                existing.unlink()

                for name in zf.namelist():
                    if name == "_manifest.json":
                        continue

                    # Security: prevent zip path traversal (e.g. ../../etc/passwd)
                    # Resolve the target and verify it stays within the KB root.
                    target = (root / name).resolve()
                    if not str(target).startswith(str(root.resolve())):
                        errors.append(f"Skipped path-traversal entry: {name}")
                        logger.warning(
                            "Zip entry %r escapes knowledge base root, skipped",
                            name,
                        )
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)

                    if strategy == "merge" and target.is_file():
                        # Check for conflict: both modified
                        local_checksum = self._file_checksum(target)
                        if remote_manifest and name in remote_manifest.files:
                            remote_checksum = remote_manifest.files[name]["checksum"]
                            if local_checksum != remote_checksum:
                                conflicts.append(name)
                                # In merge mode, remote wins for conflicts
                        # else: no conflict info, just overwrite

                    data = zf.read(name)
                    target.write_bytes(data)
                    files_pulled += 1

        except (zipfile.BadZipFile, OSError) as exc:
            errors.append(f"Failed to read bundle: {exc}")
            return SyncResult(
                status="error",
                files_pulled=files_pulled,
                files_pushed=0,
                conflicts=conflicts,
                errors=errors,
            )

        self._save_sync_state("import", bundle_path)

        status = "success"
        if conflicts:
            status = "partial"
        if errors:
            status = "error"

        return SyncResult(
            status=status,
            files_pulled=files_pulled,
            files_pushed=0,
            conflicts=conflicts,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Directory sync
    # ------------------------------------------------------------------

    def sync_to_directory(
        self, remote_dir: Path, strategy: str = "merge"
    ) -> SyncResult:
        """Sync the local KB with a remote directory.

        Creates a temporary :class:`KBSync` for the remote directory,
        computes the diff, and copies files in both directions.

        Strategies:
        - ``"merge"``: bidirectional merge (newest wins for conflicts).
        - ``"push"``: local -> remote only.
        - ``"pull"``: remote -> local only.
        """
        remote_dir = Path(remote_dir)
        remote_dir.mkdir(parents=True, exist_ok=True)

        # Build a temporary KB for the remote
        remote_kb = KnowledgeBase(root=remote_dir)
        remote_kb.ensure_structure()
        remote_sync = KBSync(remote_kb, machine_id="remote")

        local_manifest = self.build_manifest()
        remote_manifest = remote_sync.build_manifest()

        local_files = set(local_manifest.files.keys())
        remote_files = set(remote_manifest.files.keys())

        files_pushed = 0
        files_pulled = 0
        conflicts: list[str] = []
        errors: list[str] = []

        root = self._kb.root

        # Files only in local -> push to remote
        if strategy in ("merge", "push"):
            for rel in sorted(local_files - remote_files):
                src = root / rel
                dst = remote_dir / rel
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                    files_pushed += 1
                except OSError as exc:
                    errors.append(f"Failed to push {rel}: {exc}")

        # Files only in remote -> pull to local
        if strategy in ("merge", "pull"):
            for rel in sorted(remote_files - local_files):
                src = remote_dir / rel
                dst = root / rel
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                    files_pulled += 1
                except OSError as exc:
                    errors.append(f"Failed to pull {rel}: {exc}")

        # Files in both with different checksums
        for rel in sorted(local_files & remote_files):
            local_cksum = local_manifest.files[rel]["checksum"]
            remote_cksum = remote_manifest.files[rel]["checksum"]
            if local_cksum == remote_cksum:
                continue

            if strategy == "push":
                src = root / rel
                dst = remote_dir / rel
                try:
                    shutil.copy2(str(src), str(dst))
                    files_pushed += 1
                except OSError as exc:
                    errors.append(f"Failed to push {rel}: {exc}")
            elif strategy == "pull":
                src = remote_dir / rel
                dst = root / rel
                try:
                    shutil.copy2(str(src), str(dst))
                    files_pulled += 1
                except OSError as exc:
                    errors.append(f"Failed to pull {rel}: {exc}")
            else:
                # merge: newest file wins
                local_mod = local_manifest.files[rel]["modified"]
                remote_mod = remote_manifest.files[rel]["modified"]
                if remote_mod > local_mod:
                    src = remote_dir / rel
                    dst = root / rel
                    try:
                        shutil.copy2(str(src), str(dst))
                        files_pulled += 1
                    except OSError as exc:
                        errors.append(f"Failed to pull {rel}: {exc}")
                else:
                    src = root / rel
                    dst = remote_dir / rel
                    try:
                        shutil.copy2(str(src), str(dst))
                        files_pushed += 1
                    except OSError as exc:
                        errors.append(f"Failed to push {rel}: {exc}")
                conflicts.append(rel)

        self._save_sync_state("sync", remote_dir)

        status = "success"
        if conflicts:
            status = "partial"
        if errors:
            status = "error"

        return SyncResult(
            status=status,
            files_pulled=files_pulled,
            files_pushed=files_pushed,
            conflicts=conflicts,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _file_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()

    def _save_sync_state(self, operation: str, target: Path) -> None:
        """Save sync state to .sync_state.json."""
        state = {
            "last_operation": operation,
            "target": str(target),
            "machine_id": self._machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / ".sync_state.json"
        self._kb.atomic_write_json(path, state)

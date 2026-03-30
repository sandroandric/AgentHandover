"""Team Knowledge Sharing — export and import anonymized procedures.

Enables teams to share learned procedures safely by stripping PII
(emails, user paths, IPs, auth tokens) before export and applying
trust-level constraints on import.

Export format::

    {
        "agenthandover_shared_procedures": "1.0",
        "exported_at": "...",
        "exported_by": "anonymous",
        "procedures": [SharedProcedure as dict, ...]
    }
"""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_HOME_PATH_RE = re.compile(r"/(?:Users|home)/[a-zA-Z0-9._-]+/")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_AUTH_PARAM_RE = re.compile(
    r"[?&](token|session|key|auth|secret|password|api_key)=[^&\s]+",
    re.IGNORECASE,
)

# Export format version
_FORMAT_VERSION = "1.0"

# Sections that are removed entirely during anonymization because they
# contain machine-specific observation data or PII-rich evidence.
_REMOVED_SECTIONS = frozenset({"evidence", "staleness"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SharedProcedure:
    """A single procedure packaged for team sharing."""

    share_id: str
    original_slug: str
    title: str
    procedure: dict
    shared_by: str
    shared_at: str
    tags: list[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class ImportResult:
    """Summary of an import operation."""

    imported: int
    skipped: int
    conflicts: list[str]
    errors: list[str]


# ---------------------------------------------------------------------------
# TeamSharing
# ---------------------------------------------------------------------------


class TeamSharing:
    """Export and import anonymized procedures for team sharing."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        machine_alias: str = "anonymous",
    ) -> None:
        self._kb = knowledge_base
        self._alias = machine_alias

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_procedures(
        self,
        slugs: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[SharedProcedure]:
        """Export procedures from the KB for sharing (anonymized).

        Args:
            slugs: If provided, only export these procedure slugs.
                   If *None*, export all procedures in the KB.
            tags:  If provided, only export procedures whose ``tags``
                   field contains at least one of these tags.  Applied
                   after *slugs* filtering.

        Returns:
            List of :class:`SharedProcedure` ready for serialization.
        """
        all_procs = self._kb.list_procedures()

        # Filter by slugs
        if slugs is not None:
            slug_set = set(slugs)
            all_procs = [
                p for p in all_procs
                if p.get("id", p.get("slug")) in slug_set
            ]

        # Filter by tags
        if tags is not None:
            tag_set = set(tags)
            all_procs = [
                p for p in all_procs
                if tag_set & set(p.get("tags", []))
            ]

        now_iso = datetime.now(timezone.utc).isoformat()
        shared: list[SharedProcedure] = []

        for proc in all_procs:
            slug = proc.get("id", proc.get("slug", "unknown"))
            anonymized = self.anonymize_procedure(proc)
            sp = SharedProcedure(
                share_id=str(uuid.uuid4()),
                original_slug=slug,
                title=proc.get("title", "Untitled"),
                procedure=anonymized,
                shared_by=self._alias,
                shared_at=now_iso,
                tags=list(proc.get("tags", [])),
                category=proc.get("category", "general"),
            )
            shared.append(sp)

        logger.info("Exported %d procedures for sharing", len(shared))
        return shared

    def export_to_file(
        self,
        output_path: Path,
        slugs: list[str] | None = None,
    ) -> Path:
        """Export procedures to a JSON file.

        Args:
            output_path: Destination file path.
            slugs: Optional list of slugs to export (default: all).

        Returns:
            The *output_path* for convenience.
        """
        shared = self.export_procedures(slugs=slugs)
        now_iso = datetime.now(timezone.utc).isoformat()

        envelope = {
            "agenthandover_shared_procedures": _FORMAT_VERSION,
            "exported_at": now_iso,
            "exported_by": self._alias,
            "procedures": [asdict(sp) for sp in shared],
        }

        output_path = Path(output_path)
        self._kb.atomic_write_json(output_path, envelope)

        logger.info("Exported %d procedures to %s", len(shared), output_path)
        return output_path

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_from_file(
        self,
        input_path: Path,
        trust_level: str = "observe",
    ) -> ImportResult:
        """Import shared procedures from a JSON file.

        Args:
            input_path:   Path to a JSON file in the export format.
            trust_level:  Trust level to assign to all imported
                          procedures (default ``"observe"``).

        Returns:
            :class:`ImportResult` summarizing what happened.
        """
        input_path = Path(input_path)
        try:
            with open(input_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            return ImportResult(
                imported=0,
                skipped=0,
                conflicts=[],
                errors=[f"Failed to read {input_path}: {exc}"],
            )

        if not isinstance(data, dict):
            return ImportResult(
                imported=0,
                skipped=0,
                conflicts=[],
                errors=["File does not contain a valid JSON object"],
            )

        raw_procs = data.get("procedures", [])
        shared: list[SharedProcedure] = []
        errors: list[str] = []

        for i, item in enumerate(raw_procs):
            try:
                sp = SharedProcedure(
                    share_id=item.get("share_id", str(uuid.uuid4())),
                    original_slug=item.get("original_slug", "unknown"),
                    title=item.get("title", "Untitled"),
                    procedure=item.get("procedure", {}),
                    shared_by=item.get("shared_by", "unknown"),
                    shared_at=item.get("shared_at", ""),
                    tags=item.get("tags", []),
                    category=item.get("category", "general"),
                )
                shared.append(sp)
            except Exception as exc:
                errors.append(f"procedures[{i}]: {exc}")

        result = self.import_procedures(shared, trust_level=trust_level)
        result.errors.extend(errors)
        return result

    def import_procedures(
        self,
        shared: list[SharedProcedure],
        trust_level: str = "observe",
    ) -> ImportResult:
        """Import shared procedures into the knowledge base.

        - Skips any procedure whose slug already exists in the KB
          (conflict — no overwrite).
        - Assigns the given *trust_level* via the procedure's
          ``constraints`` section.
        - Generates a new ``id`` for each imported procedure.

        Args:
            shared:       List of :class:`SharedProcedure` to import.
            trust_level:  Trust level string (default ``"observe"``).

        Returns:
            :class:`ImportResult` summarizing what happened.
        """
        imported = 0
        skipped = 0
        conflicts: list[str] = []
        errors: list[str] = []

        for sp in shared:
            slug = sp.original_slug

            # Conflict check — skip if slug already present
            existing = self._kb.get_procedure(slug)
            if existing is not None:
                conflicts.append(slug)
                skipped += 1
                continue

            try:
                proc = copy.deepcopy(sp.procedure)

                # Ensure required fields
                proc.setdefault("schema_version", "3.0.0")
                proc["id"] = slug
                proc.setdefault("title", sp.title)
                proc.setdefault("steps", [])

                # Set trust level for imported procedure
                proc.setdefault("constraints", {})
                proc["constraints"]["trust_level"] = trust_level

                # Tag as imported
                proc.setdefault("metadata", {})
                proc["metadata"]["imported"] = True
                proc["metadata"]["imported_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                proc["metadata"]["shared_by"] = sp.shared_by
                proc["metadata"]["share_id"] = sp.share_id

                self._kb.save_procedure(proc)
                imported += 1
            except Exception as exc:
                errors.append(f"Failed to import '{slug}': {exc}")

        logger.info(
            "Import complete: %d imported, %d skipped, %d conflicts, %d errors",
            imported,
            skipped,
            len(conflicts),
            len(errors),
        )
        return ImportResult(
            imported=imported,
            skipped=skipped,
            conflicts=conflicts,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Anonymization
    # ------------------------------------------------------------------

    def anonymize_procedure(self, procedure: dict) -> dict:
        """Strip PII from a procedure for safe sharing.

        Removes:
            - Emails (replaced with ``<email>``)
            - User home paths (``/Users/name/`` → ``~/``)
            - IP addresses (replaced with ``<ip>``)
            - Auth query parameters (``?token=...``, ``&key=...``)
            - ``evidence`` section entirely
            - ``staleness`` section entirely

        Preserves:
            - Step structure (actions, targets, apps)
            - App names and generic URLs
            - Decision rules
            - Expected outcomes
            - Variable definitions
            - Tags, title, description
        """
        proc = copy.deepcopy(procedure)

        # Remove sections that carry observation/machine data
        for section in _REMOVED_SECTIONS:
            proc.pop(section, None)

        # Recursively strip PII from all remaining values
        proc = self._strip_pii_from_value(proc)

        return proc

    def _strip_pii_from_value(self, value: object) -> object:
        """Recursively strip PII from any value (str, dict, list, etc.)."""
        if isinstance(value, str):
            return self._strip_pii_from_string(value)
        if isinstance(value, dict):
            return {
                self._strip_pii_from_string(k) if isinstance(k, str) else k:
                self._strip_pii_from_value(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._strip_pii_from_value(item) for item in value]
        # int, float, bool, None — pass through unchanged
        return value

    def _strip_pii_from_string(self, text: str) -> str:
        """Apply all PII regex patterns to a string."""
        # Order matters: auth params before general URL cleaning
        text = _AUTH_PARAM_RE.sub("", text)
        text = _EMAIL_RE.sub("<email>", text)
        text = _HOME_PATH_RE.sub("~/", text)
        text = _IP_RE.sub("<ip>", text)
        return text

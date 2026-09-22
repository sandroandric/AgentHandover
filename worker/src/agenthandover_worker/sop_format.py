"""Manual-edit detection for exported SOP files.

Rendering lives in ``skill_render``; OpenClaw packets carry a
``generated_body_hash`` in their frontmatter.  This module compares that
hash with the body on disk so the versioner never overwrites a file the
user edited by hand.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agenthandover_worker.skill_render import body_hash


class SOPFormatter:
    """Detect manual edits of SOP files written with a body hash."""

    @staticmethod
    def _compute_body_hash(body: str) -> str:
        return body_hash(body)

    def detect_manual_edit(self, filepath: str) -> tuple[bool, str]:
        """Check if an existing SOP file was manually edited.

        Compares the stored ``generated_body_hash`` with the hash of the
        current body.

        Returns:
            (was_edited, reason) — if the file does not exist,
            returns (False, "file_not_found").
        """
        path = Path(filepath)
        if not path.exists():
            return False, "file_not_found"

        content = path.read_text(encoding="utf-8")
        frontmatter, body = self._extract_frontmatter_and_body(content)

        # Files written before the single renderer (``sop_version: 1``)
        # hashed only part of their body, so their hash cannot tell an
        # edit apart from the sections appended after hashing.  They are
        # archived (not deleted) when replaced, so no edit is lost.
        if "sop_version" in frontmatter:
            return False, "legacy_format"

        stored_hash = frontmatter.get("generated_body_hash", "")
        if not stored_hash:
            return False, "no_hash_in_frontmatter"

        if stored_hash == self._compute_body_hash(body):
            return False, "hash_matches"

        return True, "body_hash_mismatch"

    @staticmethod
    def _extract_frontmatter_and_body(content: str) -> tuple[dict, str]:
        """Split a file into its YAML frontmatter dict and body string.

        Expects ``---\\n<yaml>\\n---\\n<body>``.  The closing fence is the
        first line that is exactly ``---``, so ``---`` inside a YAML value
        does not end the frontmatter.
        """
        if not content.startswith("---\n"):
            return {}, content

        end_idx = content.find("\n---\n", 3)
        if end_idx == -1:
            return {}, content
        fm_text = content[4:end_idx]
        body = content[end_idx + len("\n---\n"):].lstrip("\n")

        try:
            frontmatter = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            return {}, content
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        return frontmatter, body

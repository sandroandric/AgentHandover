"""SOP Formatter — generate SOP markdown files with YAML frontmatter.

Implements sections 10.2-10.3 and 13.3 of the OpenMimic spec.  Each SOP is
a markdown file with a YAML frontmatter block containing metadata, variables,
and a SHA-256 hash of the body for manual edit detection.
"""

from __future__ import annotations

import hashlib
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml


class SOPFormatter:
    """Format SOP templates into markdown files with YAML frontmatter.

    The generated files follow the spec's SOP format:
    - YAML frontmatter with metadata, variables, confidence summary
    - Markdown body with numbered steps
    - SHA-256 body hash in frontmatter for manual edit detection
    """

    VERSION = "0.1.0"

    def format_sop(self, sop_template: dict) -> str:
        """Generate a complete SOP markdown file with YAML frontmatter.

        Frontmatter fields (section 10.3):
        - sop_version: 1
        - sop_slug, sop_title
        - generated_by: "oc-apprentice v0.1.0"
        - generated_at: ISO 8601
        - evidence_window: "last_30_days"
        - confidence_summary: "high" | "medium" | "low"
        - confidence_score_avg: float
        - generated_body_hash: "sha256:..."
        - apps_involved: list
        - input_variables: list of {name, type, example}
        - preconditions, postconditions: lists
        - exceptions_seen: list
        - tags: list
        """
        body = self._format_body(sop_template)
        body_hash = self._compute_body_hash(body)
        frontmatter = self._build_frontmatter(sop_template, body_hash)
        fm_yaml = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        return f"---\n{fm_yaml}---\n\n{body}"

    def _format_body(self, sop: dict) -> str:
        """Format the markdown body from SOP template steps.

        Produces a titled section with numbered steps, each describing
        the action, target, and any parameters.
        """
        title = sop.get("title", "Untitled SOP")
        steps = sop.get("steps", [])
        variables = sop.get("variables", [])

        lines: list[str] = []
        lines.append(f"# {title}")
        lines.append("")

        # Task Description section (LLM-generated) if present
        task_description = sop.get("task_description")
        if task_description:
            lines.append("## Task Description")
            lines.append("")
            lines.append(task_description)
            lines.append("")

        # Execution Overview section (LLM-generated) if present
        execution_overview = sop.get("execution_overview")
        if isinstance(execution_overview, dict) and execution_overview:
            lines.append("## Execution Overview")
            lines.append("")
            for key, value in execution_overview.items():
                label = key.replace("_", " ").title()
                lines.append(f"- **{label}**: {value}")
            lines.append("")

        # Variables section if present
        if variables:
            lines.append("## Input Variables")
            lines.append("")
            for var in variables:
                var_name = var.get("name", "unknown")
                var_type = var.get("type", "string")
                var_example = var.get("example", "")
                line = f"- **${{{var_name}}}** ({var_type})"
                if var_example:
                    line += f": e.g. `{var_example}`"
                lines.append(line)
            lines.append("")

        # Steps section
        lines.append("## Steps")
        lines.append("")

        for i, step in enumerate(steps, 1):
            intent = step.get("step", "action")
            target = step.get("target", "")
            selector = step.get("selector")
            params = step.get("parameters", {})
            confidence = step.get("confidence", 0.0)

            # Build step line
            step_line = f"{i}. **{intent.capitalize()}** "
            if target:
                step_line += f"on _{target}_"

            lines.append(step_line)

            # Add parameter details as sub-items
            if isinstance(params, dict) and params:
                for key, value in params.items():
                    lines.append(f"   - {key}: `{value}`")

            if selector:
                lines.append(f"   - selector: `{selector}`")

            lines.append(f"   - confidence: {confidence:.2f}")
            lines.append("")

        return "\n".join(lines)

    def _compute_body_hash(self, body: str) -> str:
        """Compute SHA-256 hash of the body content.

        Normalizes Unicode to NFC and line endings for consistent
        hashing across platforms regardless of NFC/NFD encoding.
        """
        normalized = unicodedata.normalize("NFC", body.strip("\n").replace("\r\n", "\n"))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def _build_frontmatter(self, sop: dict, body_hash: str) -> dict:
        """Build the YAML frontmatter dictionary."""
        confidence_avg = sop.get("confidence_avg", 0.0)
        variables = sop.get("variables", [])

        # Build input_variables list for frontmatter
        input_variables: list[dict] = []
        for var in variables:
            entry: dict = {
                "name": var.get("name", "unknown"),
                "type": var.get("type", "string"),
            }
            if "example" in var:
                entry["example"] = var["example"]
            input_variables.append(entry)

        frontmatter: dict = {
            "sop_version": 1,
            "sop_slug": sop.get("slug", "unknown"),
            "sop_title": sop.get("title", "Untitled SOP"),
            "generated_by": f"oc-apprentice v{self.VERSION}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evidence_window": "last_30_days",
            "confidence_summary": self._confidence_label(confidence_avg),
            "confidence_score_avg": round(confidence_avg, 4),
            "generated_body_hash": body_hash,
            "apps_involved": sop.get("apps_involved", []),
            "input_variables": input_variables,
            "preconditions": sop.get("preconditions", []),
            "postconditions": sop.get("postconditions", []),
            "exceptions_seen": sop.get("exceptions_seen", []),
            "tags": sop.get("tags", []),
        }

        # Add LLM-enhanced fields when present
        task_description = sop.get("task_description")
        if task_description:
            frontmatter["task_description"] = task_description

        execution_overview = sop.get("execution_overview")
        if isinstance(execution_overview, dict) and execution_overview:
            frontmatter["execution_overview"] = execution_overview

        return frontmatter

    def _confidence_label(self, score: float) -> str:
        """Map average confidence score to a label.

        - >= 0.85: "high"
        - >= 0.60: "medium"
        - < 0.60: "low"
        """
        if score >= 0.85:
            return "high"
        if score >= 0.60:
            return "medium"
        return "low"

    def detect_manual_edit(self, filepath: str) -> tuple[bool, str]:
        """Check if an existing SOP file was manually edited.

        Compares stored generated_body_hash with current body hash.

        Returns:
            (was_edited, reason) — if the file does not exist,
            returns (False, "file_not_found").
        """
        path = Path(filepath)
        if not path.exists():
            return False, "file_not_found"

        content = path.read_text(encoding="utf-8")
        frontmatter, body = self._extract_frontmatter_and_body(content)

        stored_hash = frontmatter.get("generated_body_hash", "")
        if not stored_hash:
            return False, "no_hash_in_frontmatter"

        current_hash = self._compute_body_hash(body)
        if stored_hash == current_hash:
            return False, "hash_matches"

        return True, "body_hash_mismatch"

    def _extract_frontmatter_and_body(self, content: str) -> tuple[dict, str]:
        """Parse a SOP file into frontmatter dict and body string.

        Expects the format:
        ---
        key: value
        ---

        Body content here...
        """
        if not content.startswith("---"):
            return {}, content

        # Find the closing ---
        end_idx = content.index("---", 3)
        fm_text = content[3:end_idx].strip()
        body = content[end_idx + 3:].strip("\n")

        # Remove leading blank lines from body
        if body.startswith("\n"):
            body = body.lstrip("\n")

        frontmatter = yaml.safe_load(fm_text) or {}
        return frontmatter, body

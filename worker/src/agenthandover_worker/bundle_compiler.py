"""Bundle compiler — produces agent-ready procedure bundles.

Compiles a procedure into a ProcedureBundle that agents consume.
The bundle includes readiness assessment, compiled outputs for each
export adapter, and checksums for change detection.

Key design:
- compute_readiness() is a STATIC method — pure function from
  (lifecycle, trust, freshness, preflight)
- compile() loads procedure, computes readiness, compiles all targets,
  stores compiled_outputs, returns ProcedureBundle
- compile_target() calls adapter.write_procedure(), computes SHA-256
  of output file, returns CompiledOutput
- needs_recompile() compares SHA-256 of canonical procedure JSON
  (excluding compiled_outputs and lifecycle_history) against stored
  checksum
- _procedure_checksum() helper: computes SHA-256 of procedure dict
  excluding transient fields
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.lifecycle_manager import LifecycleManager, ProcedureLifecycle
from agenthandover_worker.procedure_verifier import ProcedureVerifier, PreflightResult
from agenthandover_worker.staleness_detector import procedure_freshness

logger = logging.getLogger(__name__)

# Transient fields excluded from the canonical checksum.
# These change frequently but do not affect the procedure's semantic content.
_TRANSIENT_FIELDS = frozenset({
    "compiled_outputs",
    "lifecycle_history",
    "_procedure_checksum",
})

# Trust levels that permit execution
_EXECUTABLE_TRUST_LEVELS = frozenset({
    "execute_with_approval",
    "autonomous",
})

# Trust levels that permit drafting
_DRAFTABLE_TRUST_LEVELS = frozenset({
    "draft",
    "execute_with_approval",
    "autonomous",
})

# Lifecycle states that permit drafting
_DRAFTABLE_LIFECYCLE_STATES = frozenset({
    ProcedureLifecycle.DRAFT,
    ProcedureLifecycle.REVIEWED,
    ProcedureLifecycle.VERIFIED,
    ProcedureLifecycle.AGENT_READY,
})

_MIN_FRESHNESS = 0.3


@dataclass
class ReadinessResult:
    """Agent-facing readiness assessment for a procedure."""

    can_execute: bool
    can_draft: bool
    is_ready: bool
    reasons: list[str] = field(default_factory=list)

    lifecycle_state: str = "observed"
    trust_level: str = "observe"
    freshness: float = 0.0
    has_preflight_errors: bool = False


@dataclass
class CompiledOutput:
    """A single compiled export target."""

    adapter_name: str
    output_path: str
    sha256: str
    compiled_at: str
    size_bytes: int = 0


@dataclass
class ProcedureBundle:
    """Agent-facing package for a procedure.

    Agents should consume ProcedureBundle, not raw procedures.
    """

    slug: str
    title: str
    readiness: ReadinessResult
    compiled_outputs: list[CompiledOutput] = field(default_factory=list)
    procedure_checksum: str = ""
    compiled_at: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "slug": self.slug,
            "title": self.title,
            "readiness": {
                "can_execute": self.readiness.can_execute,
                "can_draft": self.readiness.can_draft,
                "is_ready": self.readiness.is_ready,
                "reasons": self.readiness.reasons,
                "lifecycle_state": self.readiness.lifecycle_state,
                "trust_level": self.readiness.trust_level,
                "freshness": self.readiness.freshness,
                "has_preflight_errors": self.readiness.has_preflight_errors,
            },
            "compiled_outputs": [
                {
                    "adapter_name": co.adapter_name,
                    "output_path": co.output_path,
                    "sha256": co.sha256,
                    "compiled_at": co.compiled_at,
                    "size_bytes": co.size_bytes,
                }
                for co in self.compiled_outputs
            ],
            "procedure_checksum": self.procedure_checksum,
            "compiled_at": self.compiled_at,
        }


def _procedure_checksum(proc: dict) -> str:
    """Compute SHA-256 of procedure dict excluding transient fields.

    Creates a canonical JSON representation (sorted keys, no indent)
    of the procedure with transient fields removed, then returns
    the hex digest.
    """
    canonical = {
        k: v for k, v in proc.items()
        if k not in _TRANSIENT_FIELDS
    }
    raw = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class BundleCompiler:
    """Compiles procedures into agent-ready bundles."""

    def __init__(
        self,
        kb: KnowledgeBase,
        lifecycle: LifecycleManager,
        verifier: ProcedureVerifier,
        adapters: dict[str, SOPExportAdapter] | None = None,
    ) -> None:
        self._kb = kb
        self._lifecycle = lifecycle
        self._verifier = verifier
        self._adapters = adapters or {}

    @staticmethod
    def compute_readiness(
        lifecycle_state: ProcedureLifecycle,
        trust_level: str,
        freshness: float,
        preflight: PreflightResult | None = None,
    ) -> ReadinessResult:
        """Compute readiness from inputs. Pure function, no side effects.

        Readiness rules:
        - can_execute requires ALL of:
            lifecycle == "agent_ready"
            trust in {execute_with_approval, autonomous}
            freshness >= 0.3
            no preflight blocking errors (or preflight not available)
        - can_draft requires ALL of:
            lifecycle in {draft, reviewed, verified, agent_ready}
            trust in {draft, execute_with_approval, autonomous}
            freshness >= 0.3
            no preflight blocking errors (or preflight not available)
        - is_ready = can_execute
        - reasons: list of human-readable strings explaining WHY not ready
        """
        reasons: list[str] = []

        # Check for preflight blocking errors (excluding trust_level and
        # lifecycle_state checks which we handle ourselves)
        has_preflight_errors = False
        if preflight is not None:
            blocking = [
                c for c in preflight.checks
                if not c.passed and c.severity == "error"
                and c.name not in ("trust_level", "lifecycle_state")
            ]
            if blocking:
                has_preflight_errors = True
                for check in blocking:
                    reasons.append(f"Preflight check '{check.name}' failed: {check.detail}")

        # Lifecycle check for execution
        lifecycle_allows_exec = lifecycle_state == ProcedureLifecycle.AGENT_READY
        if not lifecycle_allows_exec:
            reasons.append(
                f"Lifecycle state is '{lifecycle_state.value}', "
                f"requires 'agent_ready' for execution"
            )

        # Lifecycle check for drafting
        lifecycle_allows_draft = lifecycle_state in _DRAFTABLE_LIFECYCLE_STATES
        if not lifecycle_allows_draft:
            reasons.append(
                f"Lifecycle state is '{lifecycle_state.value}', "
                f"requires one of {sorted(s.value for s in _DRAFTABLE_LIFECYCLE_STATES)} for drafting"
            )

        # Trust check for execution
        trust_allows_exec = trust_level in _EXECUTABLE_TRUST_LEVELS
        if not trust_allows_exec:
            reasons.append(
                f"Trust level is '{trust_level}', "
                f"requires 'execute_with_approval' or 'autonomous' for execution"
            )

        # Trust check for drafting
        trust_allows_draft = trust_level in _DRAFTABLE_TRUST_LEVELS
        if not trust_allows_draft:
            reasons.append(
                f"Trust level is '{trust_level}', "
                f"requires 'draft' or higher for drafting"
            )

        # Freshness check
        freshness_ok = freshness >= _MIN_FRESHNESS
        if not freshness_ok:
            reasons.append(
                f"Freshness score is {freshness:.2f}, "
                f"requires >= {_MIN_FRESHNESS}"
            )

        can_execute = (
            lifecycle_allows_exec
            and trust_allows_exec
            and freshness_ok
            and not has_preflight_errors
        )
        can_draft = (
            lifecycle_allows_draft
            and trust_allows_draft
            and freshness_ok
            and not has_preflight_errors
        )

        return ReadinessResult(
            can_execute=can_execute,
            can_draft=can_draft,
            is_ready=can_execute,
            reasons=reasons,
            lifecycle_state=lifecycle_state.value,
            trust_level=trust_level,
            freshness=freshness,
            has_preflight_errors=has_preflight_errors,
        )

    def compile(self, slug: str) -> ProcedureBundle | None:
        """Compile a procedure into an agent-ready bundle.

        Loads the procedure, computes readiness, compiles all registered
        adapter targets, stores compiled_outputs on the procedure, and
        returns a ProcedureBundle.

        Returns None if the procedure does not exist.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            logger.warning("Cannot compile nonexistent procedure: %s", slug)
            return None

        # Gather inputs for readiness
        lifecycle_state = self._lifecycle.get_state(slug)
        constraints = proc.get("constraints", {})
        trust_level = constraints.get("trust_level", "observe")
        freshness = procedure_freshness(proc)
        preflight = self._verifier.preflight(slug)

        # Compute readiness
        readiness = self.compute_readiness(
            lifecycle_state=lifecycle_state,
            trust_level=trust_level,
            freshness=freshness,
            preflight=preflight,
        )

        # Only compile adapter targets (write to agent-visible directories)
        # when the procedure is fully executable.  Draft/reviewed/verified
        # procedures must NOT be written to live agent workspaces — only
        # agent_ready procedures with sufficient trust pass this gate.
        compiled_outputs: list[CompiledOutput] = []
        if readiness.can_execute:
            for adapter_name, adapter in self._adapters.items():
                try:
                    output = self.compile_target(proc, adapter_name, adapter)
                    compiled_outputs.append(output)
                except Exception:
                    logger.warning(
                        "Failed to compile target '%s' for procedure '%s'",
                        adapter_name, slug, exc_info=True,
                    )
        else:
            logger.debug(
                "Skipping adapter compilation for '%s': not ready "
                "(can_execute=%s, can_draft=%s)",
                slug, readiness.can_execute, readiness.can_draft,
            )

        # Compute procedure checksum
        checksum = _procedure_checksum(proc)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Store compiled_outputs back on the procedure
        proc["compiled_outputs"] = {
            co.adapter_name: {
                "output_path": co.output_path,
                "sha256": co.sha256,
                "compiled_at": co.compiled_at,
                "size_bytes": co.size_bytes,
            }
            for co in compiled_outputs
        }
        proc["_procedure_checksum"] = checksum
        self._kb.save_procedure(proc)

        bundle = ProcedureBundle(
            slug=slug,
            title=proc.get("title", "Untitled"),
            readiness=readiness,
            compiled_outputs=compiled_outputs,
            procedure_checksum=checksum,
            compiled_at=now_iso,
        )

        logger.info(
            "Compiled bundle for '%s': ready=%s, targets=%d",
            slug, readiness.is_ready, len(compiled_outputs),
        )
        return bundle

    def compile_target(
        self,
        proc: dict,
        adapter_name: str,
        adapter: SOPExportAdapter,
    ) -> CompiledOutput:
        """Compile a single export target for a procedure.

        Calls adapter.write_procedure(), computes SHA-256 of the output
        file, and returns a CompiledOutput.
        """
        output_path = adapter.write_procedure(proc)
        sha256 = _file_sha256(output_path)
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            size_bytes = output_path.stat().st_size
        except OSError:
            size_bytes = 0

        return CompiledOutput(
            adapter_name=adapter_name,
            output_path=str(output_path),
            sha256=sha256,
            compiled_at=now_iso,
            size_bytes=size_bytes,
        )

    def needs_recompile(self, slug: str) -> bool:
        """Check if a procedure needs recompilation.

        Compares SHA-256 of the canonical procedure JSON (excluding
        compiled_outputs and lifecycle_history) against the stored
        checksum. Returns True if they differ or if no checksum exists.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return False

        stored_checksum = proc.get("_procedure_checksum", "")
        if not stored_checksum:
            return True

        current_checksum = _procedure_checksum(proc)
        return current_checksum != stored_checksum

    def compile_all(self, *, force: bool = False) -> list[ProcedureBundle]:
        """Compile all procedures that need recompilation.

        Args:
            force: If True, recompile even if checksums match.

        Returns:
            List of compiled ProcedureBundles.
        """
        bundles: list[ProcedureBundle] = []
        for proc in self._kb.list_procedures():
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue
            if not force and not self.needs_recompile(slug):
                logger.debug("Skipping '%s' — checksum unchanged", slug)
                continue
            bundle = self.compile(slug)
            if bundle is not None:
                bundles.append(bundle)
        return bundles

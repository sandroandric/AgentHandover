"""Tests for the BundleCompiler — agent-ready procedure bundle compilation.

Covers:
- Compile bundle (2)
- Compile targets with adapters (4)
- needs_recompile (3)
- Cache / checksum (2)
- Readiness computation (5)
- compiled_outputs stored (2)
- Checksum correctness (2)
- Edge cases (5)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oc_apprentice_worker.bundle_compiler import (
    BundleCompiler,
    CompiledOutput,
    ProcedureBundle,
    ReadinessResult,
    _procedure_checksum,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.lifecycle_manager import LifecycleManager, ProcedureLifecycle
from oc_apprentice_worker.procedure_schema import sop_to_procedure
from oc_apprentice_worker.procedure_verifier import (
    PreflightCheck,
    PreflightResult,
    ProcedureVerifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kb(tmp_path):
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


@pytest.fixture
def lm(kb):
    return LifecycleManager(kb)


@pytest.fixture
def verifier(kb):
    return ProcedureVerifier(kb)


@pytest.fixture
def compiler(kb, lm, verifier):
    """BundleCompiler with no adapters."""
    return BundleCompiler(kb=kb, lifecycle=lm, verifier=verifier)


@pytest.fixture
def compiler_with_adapters(kb, lm, verifier, tmp_path):
    """BundleCompiler with GenericWriter adapters for 4 targets."""
    from oc_apprentice_worker.generic_writer import GenericWriter

    adapters = {}
    for name in ("target_a", "target_b", "target_c", "target_d"):
        out_dir = tmp_path / name
        out_dir.mkdir()
        adapters[name] = GenericWriter(output_dir=out_dir, json_export=True)
    return BundleCompiler(kb=kb, lifecycle=lm, verifier=verifier, adapters=adapters)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_procedure(kb, slug, lifecycle_state="observed", trust_level="observe", **overrides):
    """Build a minimal v3 procedure, save it, return the dict."""
    template = {
        "slug": slug,
        "title": f"Test {slug}",
        "steps": [{"step": "Do thing", "app": "Chrome", "confidence": 0.9}],
        "confidence_avg": 0.9,
        "apps_involved": ["Chrome"],
        "source": "test",
    }
    proc = sop_to_procedure(template)
    proc["constraints"]["trust_level"] = trust_level
    proc["lifecycle_state"] = lifecycle_state
    for k, v in overrides.items():
        proc[k] = v
    kb.save_procedure(proc)
    return proc


# ===================================================================
# 1) Compile bundle (2)
# ===================================================================

class TestCompileBundle:
    """Basic compile() tests."""

    def test_compile_returns_bundle(self, kb, compiler):
        _save_procedure(kb, "comp1", lifecycle_state="agent_ready", trust_level="autonomous")
        bundle = compiler.compile("comp1")
        assert bundle is not None
        assert isinstance(bundle, ProcedureBundle)
        assert bundle.slug == "comp1"
        assert bundle.title == "Test comp1"
        assert bundle.procedure_checksum  # non-empty

    def test_compile_nonexistent_returns_none(self, kb, compiler):
        result = compiler.compile("no-such-slug")
        assert result is None


# ===================================================================
# 2) Compile targets with adapters (4)
# ===================================================================

class TestCompileTargets:
    """Compile with registered adapters (GenericWriter)."""

    def test_compile_produces_outputs_for_all_adapters(self, kb, compiler_with_adapters):
        _save_procedure(kb, "tgt1", lifecycle_state="agent_ready", trust_level="autonomous")
        bundle = compiler_with_adapters.compile("tgt1")
        assert bundle is not None
        assert len(bundle.compiled_outputs) == 4

    def test_compiled_output_has_sha256(self, kb, compiler_with_adapters):
        _save_procedure(kb, "tgt2", lifecycle_state="draft", trust_level="draft")
        bundle = compiler_with_adapters.compile("tgt2")
        assert bundle is not None
        for co in bundle.compiled_outputs:
            assert len(co.sha256) == 64  # SHA-256 hex length

    def test_compiled_output_has_path(self, kb, compiler_with_adapters):
        _save_procedure(kb, "tgt3", lifecycle_state="draft", trust_level="draft")
        bundle = compiler_with_adapters.compile("tgt3")
        assert bundle is not None
        for co in bundle.compiled_outputs:
            assert Path(co.output_path).exists()

    def test_compiled_output_has_size_bytes(self, kb, compiler_with_adapters):
        _save_procedure(kb, "tgt4", lifecycle_state="draft", trust_level="draft")
        bundle = compiler_with_adapters.compile("tgt4")
        assert bundle is not None
        for co in bundle.compiled_outputs:
            assert co.size_bytes > 0


# ===================================================================
# 3) needs_recompile (3)
# ===================================================================

class TestNeedsRecompile:
    """Change detection via procedure checksum."""

    def test_needs_recompile_true_when_no_checksum(self, kb, compiler):
        _save_procedure(kb, "recomp1")
        assert compiler.needs_recompile("recomp1") is True

    def test_needs_recompile_false_after_compile(self, kb, compiler):
        _save_procedure(kb, "recomp2", lifecycle_state="draft", trust_level="draft")
        compiler.compile("recomp2")
        assert compiler.needs_recompile("recomp2") is False

    def test_needs_recompile_true_after_procedure_change(self, kb, compiler):
        _save_procedure(kb, "recomp3", lifecycle_state="draft", trust_level="draft")
        compiler.compile("recomp3")
        # Mutate the procedure
        proc = kb.get_procedure("recomp3")
        proc["title"] = "Changed Title"
        kb.save_procedure(proc)
        assert compiler.needs_recompile("recomp3") is True


# ===================================================================
# 4) Cache / checksum (2)
# ===================================================================

class TestCache:
    """Checksum caching behavior."""

    def test_compile_all_skips_unchanged(self, kb, compiler):
        _save_procedure(kb, "cache1", lifecycle_state="draft", trust_level="draft")
        # First compile
        bundles1 = compiler.compile_all()
        assert len(bundles1) == 1
        # Second compile — checksum should match, so skip
        bundles2 = compiler.compile_all()
        assert len(bundles2) == 0

    def test_compile_all_force_recompiles(self, kb, compiler):
        _save_procedure(kb, "cache2", lifecycle_state="draft", trust_level="draft")
        compiler.compile_all()
        # Force recompile
        bundles = compiler.compile_all(force=True)
        assert len(bundles) == 1


# ===================================================================
# 5) Readiness computation (5)
# ===================================================================

class TestComputeReadiness:
    """Test compute_readiness() as a pure static method."""

    def test_agent_ready_autonomous_fresh(self):
        result = BundleCompiler.compute_readiness(
            lifecycle_state=ProcedureLifecycle.AGENT_READY,
            trust_level="autonomous",
            freshness=0.9,
        )
        assert result.can_execute is True
        assert result.can_draft is True
        assert result.is_ready is True
        assert len(result.reasons) == 0

    def test_draft_trust_not_executable(self):
        result = BundleCompiler.compute_readiness(
            lifecycle_state=ProcedureLifecycle.AGENT_READY,
            trust_level="draft",
            freshness=0.9,
        )
        assert result.can_execute is False
        assert result.can_draft is True

    def test_low_freshness_blocks_everything(self):
        result = BundleCompiler.compute_readiness(
            lifecycle_state=ProcedureLifecycle.AGENT_READY,
            trust_level="autonomous",
            freshness=0.1,
        )
        assert result.can_execute is False
        assert result.can_draft is False
        assert any("Freshness" in r for r in result.reasons)

    def test_observed_lifecycle_blocks_drafting(self):
        result = BundleCompiler.compute_readiness(
            lifecycle_state=ProcedureLifecycle.OBSERVED,
            trust_level="draft",
            freshness=0.9,
        )
        assert result.can_execute is False
        assert result.can_draft is False
        assert any("observed" in r for r in result.reasons)

    def test_preflight_errors_block_execution(self):
        preflight = PreflightResult(
            slug="test",
            can_execute=False,
            can_draft=False,
            checks=[
                PreflightCheck(
                    name="has_steps",
                    passed=False,
                    detail="Procedure has 0 steps",
                    severity="error",
                ),
            ],
        )
        result = BundleCompiler.compute_readiness(
            lifecycle_state=ProcedureLifecycle.AGENT_READY,
            trust_level="autonomous",
            freshness=0.9,
            preflight=preflight,
        )
        assert result.can_execute is False
        assert result.has_preflight_errors is True


# ===================================================================
# 6) compiled_outputs stored (2)
# ===================================================================

class TestCompiledOutputsStored:
    """Compiled outputs should be persisted back onto the procedure."""

    def test_compiled_outputs_written_to_procedure(self, kb, compiler_with_adapters):
        _save_procedure(kb, "stored1", lifecycle_state="draft", trust_level="draft")
        compiler_with_adapters.compile("stored1")
        proc = kb.get_procedure("stored1")
        assert "compiled_outputs" in proc
        assert isinstance(proc["compiled_outputs"], dict)
        assert len(proc["compiled_outputs"]) == 4

    def test_compiled_output_has_sha256_in_procedure(self, kb, compiler_with_adapters):
        _save_procedure(kb, "stored2", lifecycle_state="draft", trust_level="draft")
        compiler_with_adapters.compile("stored2")
        proc = kb.get_procedure("stored2")
        for name, info in proc["compiled_outputs"].items():
            assert "sha256" in info
            assert len(info["sha256"]) == 64


# ===================================================================
# 7) Checksum correctness (2)
# ===================================================================

class TestChecksumCorrectness:
    """Procedure checksum should be deterministic and exclude transient fields."""

    def test_checksum_is_deterministic(self, kb):
        proc = _save_procedure(kb, "cksum1")
        proc_loaded = kb.get_procedure("cksum1")
        c1 = _procedure_checksum(proc_loaded)
        c2 = _procedure_checksum(proc_loaded)
        assert c1 == c2

    def test_checksum_excludes_transient_fields(self, kb):
        proc = _save_procedure(kb, "cksum2")
        proc_loaded = kb.get_procedure("cksum2")
        c1 = _procedure_checksum(proc_loaded)
        # Add transient fields
        proc_loaded["compiled_outputs"] = {"test": {"sha256": "abc"}}
        proc_loaded["lifecycle_history"] = [{"from_state": "x", "to_state": "y"}]
        c2 = _procedure_checksum(proc_loaded)
        assert c1 == c2


# ===================================================================
# 8) Edge cases (5)
# ===================================================================

class TestEdgeCases:
    """Miscellaneous edge-case coverage."""

    def test_needs_recompile_nonexistent_returns_false(self, kb, compiler):
        assert compiler.needs_recompile("ghost") is False

    def test_compile_all_empty_kb(self, kb, compiler):
        bundles = compiler.compile_all()
        assert bundles == []

    def test_bundle_to_dict(self, kb, compiler):
        _save_procedure(kb, "todict1", lifecycle_state="agent_ready", trust_level="autonomous")
        bundle = compiler.compile("todict1")
        d = bundle.to_dict()
        assert d["slug"] == "todict1"
        assert "readiness" in d
        assert "compiled_outputs" in d
        assert d["readiness"]["can_execute"] is True

    def test_readiness_result_fields(self):
        result = BundleCompiler.compute_readiness(
            lifecycle_state=ProcedureLifecycle.DRAFT,
            trust_level="draft",
            freshness=0.5,
        )
        assert result.lifecycle_state == "draft"
        assert result.trust_level == "draft"
        assert result.freshness == 0.5
        assert result.has_preflight_errors is False

    def test_compile_all_force_with_multiple_procedures(self, kb, compiler):
        _save_procedure(kb, "multi1", lifecycle_state="draft", trust_level="draft")
        _save_procedure(kb, "multi2", lifecycle_state="verified", trust_level="autonomous")
        bundles = compiler.compile_all(force=True)
        assert len(bundles) == 2
        slugs = {b.slug for b in bundles}
        assert slugs == {"multi1", "multi2"}

"""Tests for the team knowledge sharing module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.team_sharing import (
    ImportResult,
    SharedProcedure,
    TeamSharing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def ts(kb: KnowledgeBase) -> TeamSharing:
    """Create a TeamSharing instance with the test KB."""
    return TeamSharing(kb, machine_alias="test-machine")


def _make_procedure(
    slug: str = "test-proc",
    title: str = "Test Procedure",
    *,
    tags: list[str] | None = None,
    steps: list[dict] | None = None,
    evidence: dict | None = None,
    staleness: dict | None = None,
) -> dict:
    """Helper: build a minimal v3 procedure dict."""
    proc: dict = {
        "schema_version": "3.0.0",
        "id": slug,
        "title": title,
        "tags": tags or [],
        "steps": steps or [
            {
                "step_id": "step_1",
                "index": 0,
                "action": "Open browser",
                "target": "Chrome",
                "app": "Google Chrome",
                "confidence": 0.9,
            },
            {
                "step_id": "step_2",
                "index": 1,
                "action": "Navigate to site",
                "target": "address bar",
                "app": "Google Chrome",
                "input": "https://example.com",
                "confidence": 0.85,
            },
        ],
        "inputs": [{"name": "url", "type": "string", "required": True}],
        "constraints": {"trust_level": "observe", "guardrails": []},
        "evidence": evidence or {
            "observations": ["obs-001", "obs-002"],
            "step_evidence": [{"step_id": "step_1", "count": 5}],
            "total_observations": 5,
        },
        "staleness": staleness or {
            "last_observed": "2026-03-10T12:00:00+00:00",
            "drift_signals": [],
            "confidence_trend": [0.85, 0.87, 0.9],
        },
        "description": "A test procedure.",
        "apps_involved": ["Google Chrome"],
    }
    return proc


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportSingle:
    """Test exporting a single procedure."""

    def test_export_single_by_slug(self, kb: KnowledgeBase, ts: TeamSharing) -> None:
        proc = _make_procedure("deploy-app", "Deploy Application")
        kb.save_procedure(proc)

        shared = ts.export_procedures(slugs=["deploy-app"])
        assert len(shared) == 1
        assert shared[0].original_slug == "deploy-app"
        assert shared[0].title == "Deploy Application"
        assert shared[0].shared_by == "test-machine"

    def test_export_single_has_share_id(self, kb: KnowledgeBase, ts: TeamSharing) -> None:
        kb.save_procedure(_make_procedure("proc-a"))
        shared = ts.export_procedures(slugs=["proc-a"])
        assert shared[0].share_id  # non-empty UUID string

    def test_export_nonexistent_slug_returns_empty(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        shared = ts.export_procedures(slugs=["does-not-exist"])
        assert shared == []


class TestExportAll:
    """Test exporting all procedures."""

    def test_export_all_procedures(self, kb: KnowledgeBase, ts: TeamSharing) -> None:
        kb.save_procedure(_make_procedure("proc-a", "Proc A"))
        kb.save_procedure(_make_procedure("proc-b", "Proc B"))
        kb.save_procedure(_make_procedure("proc-c", "Proc C"))

        shared = ts.export_procedures()
        assert len(shared) == 3
        slugs = {sp.original_slug for sp in shared}
        assert slugs == {"proc-a", "proc-b", "proc-c"}

    def test_export_empty_kb(self, ts: TeamSharing) -> None:
        shared = ts.export_procedures()
        assert shared == []

    def test_export_filter_by_tags(self, kb: KnowledgeBase, ts: TeamSharing) -> None:
        kb.save_procedure(_make_procedure("web-proc", tags=["browsing", "web"]))
        kb.save_procedure(_make_procedure("dev-proc", tags=["development"]))

        shared = ts.export_procedures(tags=["browsing"])
        assert len(shared) == 1
        assert shared[0].original_slug == "web-proc"


# ---------------------------------------------------------------------------
# Anonymization tests
# ---------------------------------------------------------------------------


class TestAnonymizeEmails:
    """Test email PII stripping."""

    def test_email_in_step_input(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["input"] = "Send to user@company.com"
        anon = ts.anonymize_procedure(proc)
        assert "user@company.com" not in json.dumps(anon)
        assert "<email>" in anon["steps"][0]["input"]

    def test_email_in_description(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["description"] = "Contact admin@example.org for access"
        anon = ts.anonymize_procedure(proc)
        assert "<email>" in anon["description"]
        assert "admin@example.org" not in anon["description"]

    def test_multiple_emails(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["description"] = "CC alice@foo.com and bob@bar.com"
        anon = ts.anonymize_procedure(proc)
        assert anon["description"].count("<email>") == 2


class TestAnonymizeHomePaths:
    """Test home path normalization."""

    def test_macos_user_path(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["input"] = "Open /Users/johndoe/Documents/report.pdf"
        anon = ts.anonymize_procedure(proc)
        assert "/Users/johndoe/" not in anon["steps"][0]["input"]
        assert "~/Documents/report.pdf" in anon["steps"][0]["input"]

    def test_linux_home_path(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["input"] = "Edit /home/alice/projects/main.py"
        anon = ts.anonymize_procedure(proc)
        assert "/home/alice/" not in anon["steps"][0]["input"]
        assert "~/projects/main.py" in anon["steps"][0]["input"]


class TestAnonymizeAuthParams:
    """Test auth parameter stripping."""

    def test_token_param(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][1]["input"] = "https://api.example.com/data?token=abc123secret"
        anon = ts.anonymize_procedure(proc)
        assert "abc123secret" not in anon["steps"][1]["input"]
        assert "token=" not in anon["steps"][1]["input"]

    def test_api_key_param(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][1]["input"] = "https://site.com/page?api_key=xyz789&page=2"
        anon = ts.anonymize_procedure(proc)
        assert "xyz789" not in anon["steps"][1]["input"]
        assert "page=2" in anon["steps"][1]["input"]

    def test_multiple_auth_params(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][1]["input"] = (
            "https://site.com?session=s1&password=pw&name=test"
        )
        anon = ts.anonymize_procedure(proc)
        assert "session=" not in anon["steps"][1]["input"]
        assert "password=" not in anon["steps"][1]["input"]
        assert "name=test" in anon["steps"][1]["input"]


class TestAnonymizeIPs:
    """Test IP address stripping."""

    def test_ip_in_step(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["input"] = "Connect to 192.168.1.100"
        anon = ts.anonymize_procedure(proc)
        assert "192.168.1.100" not in anon["steps"][0]["input"]
        assert "<ip>" in anon["steps"][0]["input"]

    def test_ip_in_description(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["description"] = "Server at 10.0.0.1 and 172.16.0.5"
        anon = ts.anonymize_procedure(proc)
        assert anon["description"].count("<ip>") == 2


class TestAnonymizeEvidenceRemoved:
    """Test that evidence and staleness sections are removed."""

    def test_evidence_section_removed(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        assert "evidence" in proc
        anon = ts.anonymize_procedure(proc)
        assert "evidence" not in anon

    def test_staleness_section_removed(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        assert "staleness" in proc
        anon = ts.anonymize_procedure(proc)
        assert "staleness" not in anon


class TestAnonymizeStepStructure:
    """Test that step structure is preserved after anonymization."""

    def test_step_count_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        anon = ts.anonymize_procedure(proc)
        assert len(anon["steps"]) == len(proc["steps"])

    def test_step_actions_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        anon = ts.anonymize_procedure(proc)
        assert anon["steps"][0]["action"] == "Open browser"
        assert anon["steps"][1]["action"] == "Navigate to site"

    def test_app_names_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        anon = ts.anonymize_procedure(proc)
        assert anon["steps"][0]["app"] == "Google Chrome"

    def test_inputs_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        anon = ts.anonymize_procedure(proc)
        assert len(anon["inputs"]) == 1
        assert anon["inputs"][0]["name"] == "url"

    def test_title_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        anon = ts.anonymize_procedure(proc)
        assert anon["title"] == "Test Procedure"

    def test_original_not_mutated(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["input"] = "user@test.com"
        original_input = proc["steps"][0]["input"]
        ts.anonymize_procedure(proc)
        assert proc["steps"][0]["input"] == original_input


# ---------------------------------------------------------------------------
# File export/import tests
# ---------------------------------------------------------------------------


class TestExportToFile:
    """Test exporting to a JSON file."""

    def test_export_creates_valid_json(
        self, kb: KnowledgeBase, ts: TeamSharing, tmp_path: Path
    ) -> None:
        kb.save_procedure(_make_procedure("proc-a"))
        output = tmp_path / "shared.json"
        result_path = ts.export_to_file(output)

        assert result_path == output
        assert output.is_file()

        with open(output) as f:
            data = json.load(f)

        assert data["agenthandover_shared_procedures"] == "1.0"
        assert data["exported_by"] == "test-machine"
        assert isinstance(data["procedures"], list)
        assert len(data["procedures"]) == 1
        assert data["procedures"][0]["original_slug"] == "proc-a"

    def test_export_file_has_exported_at(
        self, kb: KnowledgeBase, ts: TeamSharing, tmp_path: Path
    ) -> None:
        kb.save_procedure(_make_procedure("proc-a"))
        ts.export_to_file(tmp_path / "out.json")

        with open(tmp_path / "out.json") as f:
            data = json.load(f)

        assert "exported_at" in data
        assert data["exported_at"]  # non-empty

    def test_export_empty_kb_to_file(
        self, ts: TeamSharing, tmp_path: Path
    ) -> None:
        ts.export_to_file(tmp_path / "empty.json")

        with open(tmp_path / "empty.json") as f:
            data = json.load(f)

        assert data["procedures"] == []


class TestImportFromFile:
    """Test importing from a JSON file."""

    def test_import_from_exported_file(
        self, kb: KnowledgeBase, ts: TeamSharing, tmp_path: Path
    ) -> None:
        kb.save_procedure(_make_procedure("proc-a", "Proc A"))
        ts.export_to_file(tmp_path / "share.json")

        # Delete original and reimport
        kb.delete_procedure("proc-a")
        assert kb.get_procedure("proc-a") is None

        result = ts.import_from_file(tmp_path / "share.json")
        assert result.imported == 1
        assert result.skipped == 0
        assert result.conflicts == []

        loaded = kb.get_procedure("proc-a")
        assert loaded is not None
        assert loaded["title"] == "Proc A"

    def test_import_nonexistent_file(self, ts: TeamSharing, tmp_path: Path) -> None:
        result = ts.import_from_file(tmp_path / "nope.json")
        assert result.imported == 0
        assert len(result.errors) >= 1

    def test_import_invalid_json(self, ts: TeamSharing, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        result = ts.import_from_file(bad)
        assert result.imported == 0
        assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# Import logic tests
# ---------------------------------------------------------------------------


class TestImportConflicts:
    """Test conflict detection during import."""

    def test_conflict_on_existing_slug(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        kb.save_procedure(_make_procedure("deploy-app"))

        sp = SharedProcedure(
            share_id="share-1",
            original_slug="deploy-app",
            title="Deploy App (team version)",
            procedure=_make_procedure("deploy-app")["steps"][0],
            shared_by="other-machine",
            shared_at="2026-03-11T00:00:00Z",
        )
        # Provide a proper procedure dict
        sp.procedure = {"steps": [{"action": "test"}]}

        result = ts.import_procedures([sp])
        assert result.skipped == 1
        assert "deploy-app" in result.conflicts
        assert result.imported == 0

    def test_no_conflict_on_new_slug(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        sp = SharedProcedure(
            share_id="share-2",
            original_slug="brand-new-proc",
            title="Brand New",
            procedure={"steps": [{"action": "do something"}]},
            shared_by="teammate",
            shared_at="2026-03-11T00:00:00Z",
        )

        result = ts.import_procedures([sp])
        assert result.imported == 1
        assert result.conflicts == []


class TestImportTrustLevel:
    """Test that import sets the correct trust level."""

    def test_default_trust_level_observe(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        sp = SharedProcedure(
            share_id="s1",
            original_slug="imported-proc",
            title="Imported",
            procedure={"steps": []},
            shared_by="team",
            shared_at="2026-03-11T00:00:00Z",
        )
        ts.import_procedures([sp])

        loaded = kb.get_procedure("imported-proc")
        assert loaded["constraints"]["trust_level"] == "observe"

    def test_custom_trust_level(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        sp = SharedProcedure(
            share_id="s2",
            original_slug="trusted-proc",
            title="Trusted",
            procedure={"steps": []},
            shared_by="team",
            shared_at="2026-03-11T00:00:00Z",
        )
        ts.import_procedures([sp], trust_level="suggest")

        loaded = kb.get_procedure("trusted-proc")
        assert loaded["constraints"]["trust_level"] == "suggest"


class TestImportSkipsDuplicates:
    """Test that importing the same procedure twice skips the second."""

    def test_double_import_skips(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        sp = SharedProcedure(
            share_id="s1",
            original_slug="dup-proc",
            title="Duplicate",
            procedure={"steps": []},
            shared_by="team",
            shared_at="2026-03-11T00:00:00Z",
        )

        r1 = ts.import_procedures([sp])
        assert r1.imported == 1

        r2 = ts.import_procedures([sp])
        assert r2.imported == 0
        assert r2.skipped == 1
        assert "dup-proc" in r2.conflicts


class TestImportMetadata:
    """Test that imported procedures carry import metadata."""

    def test_imported_flag_set(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        sp = SharedProcedure(
            share_id="s-meta",
            original_slug="meta-proc",
            title="With Metadata",
            procedure={"steps": []},
            shared_by="alice-machine",
            shared_at="2026-03-11T00:00:00Z",
        )
        ts.import_procedures([sp])

        loaded = kb.get_procedure("meta-proc")
        assert loaded["metadata"]["imported"] is True
        assert loaded["metadata"]["shared_by"] == "alice-machine"
        assert loaded["metadata"]["share_id"] == "s-meta"


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test export-then-import round trip."""

    def test_round_trip_preserves_steps(
        self, kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        ts_export = TeamSharing(kb, machine_alias="exporter")
        proc = _make_procedure("round-trip", "Round Trip Proc")
        kb.save_procedure(proc)

        export_path = tmp_path / "round_trip.json"
        ts_export.export_to_file(export_path)

        # Import into a fresh KB
        kb2 = KnowledgeBase(root=tmp_path / "kb2")
        kb2.ensure_structure()
        ts_import = TeamSharing(kb2, machine_alias="importer")

        result = ts_import.import_from_file(export_path)
        assert result.imported == 1

        loaded = kb2.get_procedure("round-trip")
        assert loaded is not None
        assert loaded["title"] == "Round Trip Proc"
        assert len(loaded["steps"]) == 2
        assert loaded["steps"][0]["action"] == "Open browser"

    def test_round_trip_strips_evidence(
        self, kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        ts = TeamSharing(kb, machine_alias="test")
        proc = _make_procedure("ev-proc")
        proc["evidence"] = {"observations": ["secret-obs-1"]}
        kb.save_procedure(proc)

        path = tmp_path / "ev_export.json"
        ts.export_to_file(path)

        with open(path) as f:
            data = json.load(f)

        exported_proc = data["procedures"][0]["procedure"]
        assert "evidence" not in exported_proc


# ---------------------------------------------------------------------------
# Nested PII stripping tests
# ---------------------------------------------------------------------------


class TestNestedPIIStripping:
    """Test PII stripping in deeply nested structures."""

    def test_nested_dict(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["metadata"] = {
            "contact": {"email": "dev@internal.io", "team": "backend"},
            "server": "10.0.0.42",
        }
        anon = ts.anonymize_procedure(proc)
        assert anon["metadata"]["contact"]["email"] == "<email>"
        assert anon["metadata"]["server"] == "<ip>"

    def test_nested_list(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["notes"] = [
            "Contact admin@corp.com",
            "Server at 192.168.0.1",
            ["nested@email.com"],
        ]
        anon = ts.anonymize_procedure(proc)
        assert "<email>" in anon["notes"][0]
        assert "<ip>" in anon["notes"][1]
        assert "<email>" in anon["notes"][2][0]

    def test_mixed_types_in_list(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["misc"] = [42, True, None, "user@test.com", 3.14]
        anon = ts.anonymize_procedure(proc)
        assert anon["misc"] == [42, True, None, "<email>", 3.14]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_procedure(self, ts: TeamSharing) -> None:
        proc: dict = {
            "schema_version": "3.0.0",
            "id": "empty",
            "title": "Empty",
            "steps": [],
        }
        anon = ts.anonymize_procedure(proc)
        assert anon["steps"] == []
        assert anon["title"] == "Empty"

    def test_no_pii_present(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        # Procedure with no PII — should pass through unchanged (minus evidence/staleness)
        anon = ts.anonymize_procedure(proc)
        assert anon["steps"][0]["action"] == "Open browser"
        assert anon["title"] == "Test Procedure"

    def test_none_values_in_procedure(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["selector"] = None
        proc["optional_field"] = None
        anon = ts.anonymize_procedure(proc)
        assert anon["steps"][0]["selector"] is None
        assert anon["optional_field"] is None

    def test_numeric_values_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["steps"][0]["confidence"] = 0.95
        proc["episode_count"] = 10
        anon = ts.anonymize_procedure(proc)
        assert anon["steps"][0]["confidence"] == 0.95
        assert anon["episode_count"] == 10

    def test_boolean_values_preserved(self, ts: TeamSharing) -> None:
        proc = _make_procedure()
        proc["inputs"][0]["required"] = True
        anon = ts.anonymize_procedure(proc)
        assert anon["inputs"][0]["required"] is True

    def test_shared_procedure_default_category(self, ts: TeamSharing) -> None:
        sp = SharedProcedure(
            share_id="x",
            original_slug="s",
            title="T",
            procedure={},
            shared_by="me",
            shared_at="now",
        )
        assert sp.category == "general"
        assert sp.tags == []

    def test_import_result_dataclass(self) -> None:
        r = ImportResult(imported=3, skipped=1, conflicts=["a"], errors=[])
        assert r.imported == 3
        assert r.skipped == 1

    def test_import_file_not_dict(self, ts: TeamSharing, tmp_path: Path) -> None:
        bad = tmp_path / "array.json"
        bad.write_text("[1, 2, 3]")
        result = ts.import_from_file(bad)
        assert result.imported == 0
        assert any("valid JSON object" in e for e in result.errors)

    def test_export_with_slugs_and_tags_combined(
        self, kb: KnowledgeBase, ts: TeamSharing
    ) -> None:
        """Slugs filter first, then tags filter the result."""
        kb.save_procedure(_make_procedure("a", tags=["web"]))
        kb.save_procedure(_make_procedure("b", tags=["dev"]))
        kb.save_procedure(_make_procedure("c", tags=["web"]))

        # Request slugs a and b, but only tag "web" — only a matches
        shared = ts.export_procedures(slugs=["a", "b"], tags=["web"])
        assert len(shared) == 1
        assert shared[0].original_slug == "a"

    def test_anonymize_url_preserves_base(self, ts: TeamSharing) -> None:
        """Auth params are stripped but the base URL survives."""
        proc = _make_procedure()
        proc["steps"][0]["input"] = "https://app.example.com/page?auth=secret123&view=list"
        anon = ts.anonymize_procedure(proc)
        result = anon["steps"][0]["input"]
        assert "https://app.example.com/page" in result
        assert "auth=" not in result
        assert "view=list" in result

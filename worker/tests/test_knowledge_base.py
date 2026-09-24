"""Tests for the knowledge base manager."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from daemon_db import kb_db
from agenthandover_worker.procedure_schema import (
    PROCEDURE_SCHEMA_VERSION,
    sop_to_procedure,
    validate_procedure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
    kb.ensure_structure()
    return kb


@pytest.fixture()
def sample_sop_template() -> dict:
    """A realistic SOP template as produced by the SOP generator."""
    return {
        "slug": "check-expired-domains",
        "title": "Check Expired Domains on GoDaddy Auctions",
        "short_title": "Check expired domains",
        "description": "Search for recently expired domains on GoDaddy Auctions.",
        "outcome": "User finds and evaluates expired domains for purchase.",
        "when_to_use": "When looking for domain investment opportunities.",
        "prerequisites": ["GoDaddy account", "Web browser"],
        "tags": ["browsing", "finance"],
        "confidence_avg": 0.87,
        "episode_count": 3,
        "evidence_window": "last_30_days",
        "apps_involved": ["Google Chrome"],
        "source": "passive",
        "variables": [
            {
                "name": "search_query",
                "type": "string",
                "description": "The domain search term",
                "example": "ai",
            },
        ],
        "steps": [
            {
                "step": "Navigate to GoDaddy Auctions",
                "action": "Navigate to GoDaddy Auctions",
                "target": "address bar",
                "app": "Google Chrome",
                "location": "https://auctions.godaddy.com",
                "input": "https://auctions.godaddy.com",
                "verify": "GoDaddy Auctions page loads",
                "selector": None,
                "parameters": {},
                "confidence": 0.92,
            },
            {
                "step": "Search for domains",
                "action": "Type search query",
                "target": "search box",
                "app": "Google Chrome",
                "location": "https://auctions.godaddy.com",
                "input": "{search_query}",
                "verify": "Search results appear",
                "selector": None,
                "parameters": {},
                "confidence": 0.85,
            },
        ],
        "preconditions": ["Logged into GoDaddy"],
        "postconditions": ["Domain list reviewed"],
        "exceptions_seen": ["Network timeout"],
    }


@pytest.fixture()
def sample_procedure(sample_sop_template: dict) -> dict:
    """A v3 procedure converted from the sample SOP template."""
    return sop_to_procedure(sample_sop_template)


# ---------------------------------------------------------------------------
# Knowledge Base — directory structure
# ---------------------------------------------------------------------------

class TestKnowledgeBaseStructure:

    def test_ensure_structure_creates_dirs(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(root=tmp_path / "kb", db_path=kb_db(tmp_path / "kb"))
        kb.ensure_structure()
        assert (tmp_path / "kb").is_dir()
        assert (tmp_path / "kb" / "procedures").is_dir()
        assert (tmp_path / "kb" / "observations" / "daily").is_dir()
        assert (tmp_path / "kb" / "observations" / "patterns").is_dir()
        assert (tmp_path / "kb" / "context").is_dir()

    def test_ensure_structure_idempotent(self, kb: KnowledgeBase) -> None:
        kb.ensure_structure()
        kb.ensure_structure()
        assert kb.root.is_dir()

    def test_root_property(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(root=tmp_path / "myroot", db_path=kb_db(tmp_path / "myroot"))
        assert kb.root == tmp_path / "myroot"


# ---------------------------------------------------------------------------
# Knowledge Base — procedures CRUD
# ---------------------------------------------------------------------------

class TestKnowledgeBaseProcedures:

    def test_save_and_get_procedure(
        self, kb: KnowledgeBase, sample_procedure: dict
    ) -> None:
        path = kb.save_procedure(sample_procedure)
        assert path.is_file()
        loaded = kb.get_procedure("check-expired-domains")
        assert loaded is not None
        assert loaded["id"] == "check-expired-domains"
        assert loaded["title"] == "Check Expired Domains on GoDaddy Auctions"

    def test_get_nonexistent_procedure(self, kb: KnowledgeBase) -> None:
        assert kb.get_procedure("nonexistent") is None

    def test_list_procedures_empty(self, kb: KnowledgeBase) -> None:
        assert kb.list_procedures() == []

    def test_list_procedures(
        self, kb: KnowledgeBase, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        procs = kb.list_procedures()
        assert len(procs) == 1
        assert procs[0]["id"] == "check-expired-domains"

    def test_list_procedures_multiple(self, kb: KnowledgeBase) -> None:
        for i in range(3):
            kb.save_procedure({
                "id": f"proc-{i}",
                "title": f"Procedure {i}",
                "schema_version": "3.0.0",
                "steps": [],
            })
        procs = kb.list_procedures()
        assert len(procs) == 3

    def test_save_overwrites(
        self, kb: KnowledgeBase, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        sample_procedure["title"] = "Updated Title"
        kb.save_procedure(sample_procedure)
        loaded = kb.get_procedure("check-expired-domains")
        assert loaded is not None
        assert loaded["title"] == "Updated Title"

    def test_delete_procedure(
        self, kb: KnowledgeBase, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        assert kb.delete_procedure("check-expired-domains") is True
        assert kb.get_procedure("check-expired-domains") is None

    def test_delete_nonexistent(self, kb: KnowledgeBase) -> None:
        assert kb.delete_procedure("nonexistent") is False


# ---------------------------------------------------------------------------
# Knowledge Base — procedures live in the database, the files are the export
# ---------------------------------------------------------------------------


def _rows(kb: KnowledgeBase) -> dict[str, tuple[str, str, str]]:
    conn = sqlite3.connect(kb.db_path)
    try:
        return {
            slug: (title, state, text)
            for slug, title, state, text in conn.execute(
                "SELECT slug, title, lifecycle_state, procedure_json FROM procedures"
            )
        }
    finally:
        conn.close()


def _v040_file(kb: KnowledgeBase, name: str, data: object) -> bytes:
    """A procedure file as v0.4.0's KnowledgeBase wrote it."""
    path = kb.root / "procedures" / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path.read_bytes()


class TestProcedureStore:

    def test_save_writes_the_row_and_its_export_byte_for_byte(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        sample_procedure["lifecycle_state"] = "draft"
        path = kb.save_procedure(sample_procedure)
        title, state, text = _rows(kb)["check-expired-domains"]
        assert (title, state) == ("Check Expired Domains on GoDaddy Auctions", "draft")
        assert path == kb.root / "procedures" / "check-expired-domains.json"
        # The export is what the file readers (app, agents) always got:
        # the procedure as json.dump(indent=2, default=str) writes it.
        assert path.read_text(encoding="utf-8") == text
        assert text == json.dumps(sample_procedure, indent=2, default=str)

    def test_reads_come_from_the_table_not_the_files(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        path = kb.save_procedure(sample_procedure)
        path.unlink()
        _v040_file(kb, "stray.json", {"id": "stray", "title": "Not saved"})
        assert kb.get_procedure("check-expired-domains")["id"] == "check-expired-domains"
        assert [p["id"] for p in kb.list_procedures()] == ["check-expired-domains"]
        assert kb.get_procedure("stray") is None

    def test_delete_removes_the_row_and_the_export(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        path = kb.save_procedure(sample_procedure)
        assert kb.delete_procedure("check-expired-domains") is True
        assert _rows(kb) == {}
        assert not path.exists()

    def test_list_keeps_the_file_name_order(self, kb: KnowledgeBase) -> None:
        # "a-b.json" sorts before "a.json" ('-' < '.'), though "a" < "a-b".
        for slug in ("a", "a-b", "b"):
            kb.save_procedure({"id": slug, "title": slug, "steps": []})
        assert [p["id"] for p in kb.list_procedures()] == ["a-b", "a", "b"]

    def test_list_parses_only_what_changed(self, kb: KnowledgeBase) -> None:
        kb.save_procedure({"id": "one", "title": "One", "steps": []})
        kb.save_procedure({"id": "two", "title": "Two", "steps": []})
        first = {p["id"]: p for p in kb.list_procedures()}
        kb.save_procedure({"id": "two", "title": "Two again", "steps": []})
        second = {p["id"]: p for p in kb.list_procedures()}
        assert second["one"] is first["one"]
        assert second["two"]["title"] == "Two again"

    def test_a_second_instance_sees_the_same_store(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        kb.save_procedure(sample_procedure)
        other = KnowledgeBase(root=kb.root, db_path=kb.db_path)
        assert other.get_procedure("check-expired-domains") is not None

    def test_without_the_daemon_database_nothing_is_read_or_saved(
        self, tmp_path: Path, sample_procedure: dict,
    ) -> None:
        kb = KnowledgeBase(root=tmp_path / "kb", db_path=tmp_path / "missing.db")
        assert kb.list_procedures() == []
        assert kb.get_procedure("check-expired-domains") is None
        with pytest.raises(FileNotFoundError, match="daemon"):
            kb.save_procedure(sample_procedure)
        assert not (tmp_path / "missing.db").exists()

    def test_export_rewrites_missing_and_diverged_files(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        path = kb.save_procedure(sample_procedure)
        kb.save_procedure({"id": "other", "title": "Other", "steps": []})
        text = path.read_text(encoding="utf-8")
        path.write_text("{}", encoding="utf-8")
        (kb.root / "procedures" / "other.json").unlink()
        stray = _v040_file(kb, "stray.json", {"id": "stray"})

        assert kb.export_procedures() == 2
        assert path.read_text(encoding="utf-8") == text
        assert (kb.root / "procedures" / "other.json").is_file()
        assert (kb.root / "procedures" / "stray.json").read_bytes() == stray
        assert kb.export_procedures() == 0


class TestImportProcedureFiles:
    """The upgrade from v0.4.0, whose knowledge base kept procedures only
    as ``procedures/<slug>.json``."""

    def test_imports_each_file_verbatim_and_leaves_it_as_the_export(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        sample_procedure["lifecycle_state"] = "agent_ready"
        a = _v040_file(kb, "check-expired-domains.json", sample_procedure)
        b = _v040_file(kb, "b.json", {"id": "b", "title": "B"})
        _v040_file(kb, "list.json", [1, 2])
        (kb.root / "procedures" / "broken.json").write_text("{nope", encoding="utf-8")
        _v040_file(kb, ".hidden.json", {"id": "hidden"})

        assert kb.import_procedure_files() == 2

        rows = _rows(kb)
        assert set(rows) == {"check-expired-domains", "b"}
        assert rows["check-expired-domains"] == (
            "Check Expired Domains on GoDaddy Auctions", "agent_ready", a.decode(),
        )
        assert rows["b"][2] == b.decode()
        assert (kb.root / "procedures" / "check-expired-domains.json").read_bytes() == a
        assert (kb.root / "procedures" / "broken.json").is_file()
        assert kb.get_procedure("check-expired-domains") == json.loads(a)

    def test_runs_once(self, kb: KnowledgeBase, sample_procedure: dict) -> None:
        _v040_file(kb, "check-expired-domains.json", sample_procedure)
        assert kb.import_procedure_files() == 1
        assert kb.import_procedure_files() == 0

        # A procedure deleted after the import is not brought back by a
        # file that reappears (or was left behind).
        kb.delete_procedure("check-expired-domains")
        _v040_file(kb, "check-expired-domains.json", sample_procedure)
        assert kb.import_procedure_files() == 0
        assert kb.list_procedures() == []

    def test_a_row_already_saved_is_kept(
        self, kb: KnowledgeBase, sample_procedure: dict,
    ) -> None:
        kb.save_procedure(dict(sample_procedure, title="Newer"))
        stored = _rows(kb)["check-expired-domains"]
        _v040_file(kb, "check-expired-domains.json", sample_procedure)
        assert kb.import_procedure_files() == 0
        assert _rows(kb)["check-expired-domains"] == stored

    def test_an_empty_knowledge_base_imports_nothing_and_is_marked(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "fresh"
        kb = KnowledgeBase(root=root, db_path=kb_db(root))
        assert kb.import_procedure_files() == 0
        (root / "procedures").mkdir(parents=True)
        _v040_file(kb, "late.json", {"id": "late"})
        assert kb.import_procedure_files() == 0


# ---------------------------------------------------------------------------
# Knowledge Base — profile
# ---------------------------------------------------------------------------

class TestKnowledgeBaseProfile:

    def test_get_default_profile(self, kb: KnowledgeBase) -> None:
        profile = kb.get_profile()
        assert "tools" in profile
        assert "working_hours" in profile
        assert profile["updated_at"] is None

    def test_update_profile(self, kb: KnowledgeBase) -> None:
        kb.update_profile({"tools": {"browser": "Chrome", "editor": "VS Code"}})
        profile = kb.get_profile()
        assert profile["tools"]["browser"] == "Chrome"
        assert profile["updated_at"] is not None

    def test_update_profile_merges(self, kb: KnowledgeBase) -> None:
        kb.update_profile({"tools": {"browser": "Chrome"}})
        kb.update_profile({"working_hours": {"start": "09:00", "end": "17:00"}})
        profile = kb.get_profile()
        assert profile["tools"]["browser"] == "Chrome"
        assert profile["working_hours"]["start"] == "09:00"


# ---------------------------------------------------------------------------
# Knowledge Base — decisions, triggers, constraints
# ---------------------------------------------------------------------------

class TestKnowledgeBaseDecisions:

    def test_get_default_decisions(self, kb: KnowledgeBase) -> None:
        decisions = kb.get_decisions()
        assert decisions["decision_sets"] == []

    def test_update_decisions(self, kb: KnowledgeBase) -> None:
        kb.update_decisions({
            "decision_sets": [{"slug": "test", "rules": []}]
        })
        decisions = kb.get_decisions()
        assert len(decisions["decision_sets"]) == 1


class TestKnowledgeBaseTriggers:

    def test_get_default_triggers(self, kb: KnowledgeBase) -> None:
        triggers = kb.get_triggers()
        assert triggers["recurrence"] == []
        assert triggers["chains"] == []

    def test_update_triggers(self, kb: KnowledgeBase) -> None:
        kb.update_triggers({
            "recurrence": [{"slug": "daily-standup", "pattern": "daily"}],
            "chains": [],
        })
        triggers = kb.get_triggers()
        assert len(triggers["recurrence"]) == 1


class TestKnowledgeBaseConstraints:

    def test_get_default_constraints(self, kb: KnowledgeBase) -> None:
        constraints = kb.get_constraints()
        assert constraints["global"] == {}
        assert constraints["per_procedure"] == {}

    def test_update_constraints(self, kb: KnowledgeBase) -> None:
        kb.update_constraints({
            "global": {"max_spend_usd_without_approval": 100},
            "per_procedure": {},
        })
        constraints = kb.get_constraints()
        assert constraints["global"]["max_spend_usd_without_approval"] == 100


# ---------------------------------------------------------------------------
# Knowledge Base — context
# ---------------------------------------------------------------------------

class TestKnowledgeBaseContext:

    def test_get_empty_context(self, kb: KnowledgeBase) -> None:
        ctx = kb.get_context("recent")
        assert ctx == {}

    def test_update_and_get_context(self, kb: KnowledgeBase) -> None:
        kb.update_context("recent", {"last_7_days": [{"date": "2026-03-10"}]})
        ctx = kb.get_context("recent")
        assert "last_7_days" in ctx
        assert ctx["updated_at"] is not None


# ---------------------------------------------------------------------------
# Knowledge Base — daily summaries
# ---------------------------------------------------------------------------

class TestKnowledgeBaseDailySummaries:

    def test_save_and_get_daily_summary(self, kb: KnowledgeBase) -> None:
        summary = {"active_hours": 6.5, "task_count": 12}
        path = kb.save_daily_summary("2026-03-10", summary)
        assert path.is_file()
        loaded = kb.get_daily_summary("2026-03-10")
        assert loaded is not None
        assert loaded["active_hours"] == 6.5
        assert loaded["date"] == "2026-03-10"

    def test_get_nonexistent_summary(self, kb: KnowledgeBase) -> None:
        assert kb.get_daily_summary("1999-01-01") is None

    def test_list_daily_summaries(self, kb: KnowledgeBase) -> None:
        for day in range(1, 6):
            kb.save_daily_summary(f"2026-03-{day:02d}", {"task_count": day})
        dates = kb.list_daily_summaries(limit=3)
        assert len(dates) == 3
        assert dates[0] == "2026-03-05"  # newest first

    def test_list_daily_summaries_empty(self, kb: KnowledgeBase) -> None:
        assert kb.list_daily_summaries() == []


# ---------------------------------------------------------------------------
# Knowledge Base — patterns
# ---------------------------------------------------------------------------

class TestKnowledgeBasePatterns:

    def test_save_and_get_pattern(self, kb: KnowledgeBase) -> None:
        data = {"chains": [{"first": "a", "then": "b"}]}
        path = kb.save_pattern("chains", data)
        assert path.is_file()
        loaded = kb.get_pattern("chains")
        assert loaded is not None
        assert loaded["pattern_type"] == "chains"

    def test_get_nonexistent_pattern(self, kb: KnowledgeBase) -> None:
        assert kb.get_pattern("nonexistent") is None


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_atomic_write_round_trip(self, kb: KnowledgeBase) -> None:
        data = {"key": "value", "nested": {"a": 1}}
        path = kb.root / "test.json"
        kb.atomic_write_json(path, data)
        loaded = kb._read_json(path)
        assert loaded == data

    def test_read_invalid_json(self, kb: KnowledgeBase) -> None:
        path = kb.root / "invalid.json"
        path.write_text("not json{{{")
        assert kb._read_json(path) is None

    def test_read_non_dict_json(self, kb: KnowledgeBase) -> None:
        path = kb.root / "array.json"
        path.write_text("[1, 2, 3]")
        assert kb._read_json(path) is None

    def test_concurrent_reads_safe(self, kb: KnowledgeBase) -> None:
        """Multiple threads reading the same file should not crash."""
        data = {"key": f"value_{i}" for i in range(100)}
        path = kb.root / "concurrent.json"
        kb.atomic_write_json(path, data)

        results: list[dict | None] = [None] * 10
        errors: list[Exception | None] = [None] * 10

        def read_file(idx: int) -> None:
            try:
                results[idx] = kb._read_json(path)
            except Exception as exc:
                errors[idx] = exc

        threads = [threading.Thread(target=read_file, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(10):
            assert errors[i] is None, f"Thread {i} raised: {errors[i]}"
            assert results[i] == data


# ---------------------------------------------------------------------------
# Procedure schema — sop_to_procedure conversion
# ---------------------------------------------------------------------------

class TestSOPToProcedure:

    def test_basic_conversion(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        assert proc["schema_version"] == PROCEDURE_SCHEMA_VERSION
        assert proc["id"] == "check-expired-domains"
        assert proc["title"] == "Check Expired Domains on GoDaddy Auctions"
        assert len(proc["steps"]) == 2
        assert proc["steps"][0]["step_id"] == "step_1"
        assert proc["steps"][0]["action"] == "Navigate to GoDaddy Auctions"

    def test_steps_have_on_failure(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        for step in proc["steps"]:
            assert "on_failure" in step
            assert step["on_failure"]["strategy"] == "abort"

    def test_variables_become_inputs(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        assert len(proc["inputs"]) == 1
        assert proc["inputs"][0]["name"] == "search_query"
        assert proc["inputs"][0]["type"] == "string"
        assert proc["inputs"][0]["required"] is True

    def test_v3_sections_present(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        assert isinstance(proc["inputs"], list)
        assert isinstance(proc["outputs"], list)
        assert isinstance(proc["environment"], dict)
        assert isinstance(proc["branches"], list)
        assert isinstance(proc["expected_outcomes"], list)
        assert isinstance(proc["staleness"], dict)
        assert isinstance(proc["evidence"], dict)
        assert isinstance(proc["constraints"], dict)
        assert isinstance(proc["recurrence"], dict)

    def test_environment_inherits_apps(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        assert "Google Chrome" in proc["environment"]["required_apps"]

    def test_confidence_label(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        assert proc["confidence_summary"] == "high"

    def test_confidence_label_medium(self) -> None:
        proc = sop_to_procedure({"slug": "t", "title": "T", "steps": [],
                                  "confidence_avg": 0.72})
        assert proc["confidence_summary"] == "medium"

    def test_confidence_label_low(self) -> None:
        proc = sop_to_procedure({"slug": "t", "title": "T", "steps": [],
                                  "confidence_avg": 0.30})
        assert proc["confidence_summary"] == "low"

    def test_validates_clean(self, sample_sop_template: dict) -> None:
        proc = sop_to_procedure(sample_sop_template)
        errors = validate_procedure(proc)
        assert errors == [], f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# Procedure schema — validation
# ---------------------------------------------------------------------------

class TestValidateProcedure:

    def test_valid_procedure(self, sample_procedure: dict) -> None:
        assert validate_procedure(sample_procedure) == []

    def test_missing_required_fields(self) -> None:
        errors = validate_procedure({})
        assert any("id" in e for e in errors)
        assert any("title" in e for e in errors)
        assert any("steps" in e for e in errors)

    def test_bad_schema_version(self) -> None:
        errors = validate_procedure({
            "schema_version": "99.0.0",
            "id": "x",
            "title": "X",
            "steps": [],
        })
        assert any("Unsupported schema version" in e for e in errors)

    def test_steps_not_list(self) -> None:
        errors = validate_procedure({
            "schema_version": "3.0.0",
            "id": "x",
            "title": "X",
            "steps": "not a list",
        })
        assert any("steps" in e and "list" in e for e in errors)

    def test_step_missing_action(self) -> None:
        errors = validate_procedure({
            "schema_version": "3.0.0",
            "id": "x",
            "title": "X",
            "steps": [{"target": "y"}],
        })
        assert any("action" in e for e in errors)

    def test_bad_on_failure_strategy(self) -> None:
        errors = validate_procedure({
            "schema_version": "3.0.0",
            "id": "x",
            "title": "X",
            "steps": [{"action": "y", "on_failure": {"strategy": "explode"}}],
        })
        assert any("on_failure.strategy" in e for e in errors)

    def test_bad_trust_level(self) -> None:
        errors = validate_procedure({
            "schema_version": "3.0.0",
            "id": "x",
            "title": "X",
            "steps": [],
            "constraints": {"trust_level": "superuser"},
        })
        assert any("trust_level" in e for e in errors)

    def test_inputs_validation(self) -> None:
        errors = validate_procedure({
            "schema_version": "3.0.0",
            "id": "x",
            "title": "X",
            "steps": [],
            "inputs": [{"name": "q"}],  # missing type
        })
        assert any("type" in e for e in errors)

    def test_staleness_confidence_trend_not_list(self) -> None:
        errors = validate_procedure({
            "schema_version": "3.0.0",
            "id": "x",
            "title": "X",
            "steps": [],
            "staleness": {"confidence_trend": "nope"},
        })
        assert any("confidence_trend" in e for e in errors)


# ---------------------------------------------------------------------------
# Integration: full round-trip through knowledge base
# ---------------------------------------------------------------------------

class TestKBRoundTrip:

    def test_sop_to_procedure_to_kb(
        self, kb: KnowledgeBase, sample_sop_template: dict
    ) -> None:
        proc = sop_to_procedure(sample_sop_template)
        path = kb.save_procedure(proc)
        loaded = kb.get_procedure("check-expired-domains")
        assert loaded is not None
        assert loaded["id"] == proc["id"]
        assert loaded["steps"][0]["action"] == proc["steps"][0]["action"]
        errors = validate_procedure(loaded)
        assert errors == []


# ---------------------------------------------------------------------------
# Backfill of procedures saved by older versions (WP-P1)
# ---------------------------------------------------------------------------

def _legacy_procedure() -> dict:
    """A procedure as v0.4.0's sop_to_procedure saved generator output:
    empty description, step fields present only under ``parameters``."""
    return {
        "schema_version": "3.2.0",
        "id": "legacy-flow",
        "title": "Legacy Flow",
        "description": "",
        "task_description": "Do the legacy flow end to end.",
        "steps": [{
            "step_id": "step_1",
            "index": 0,
            "action": "Open the page",
            "description": "Start here.",
            "target": "",
            "app": "",
            "location": "",
            "input": "",
            "verify": "",
            "parameters": {
                "app": "Safari",
                "location": "https://example.com",
                "verify": "The page loads.",
            },
        }],
    }


class TestBackfillProcedures:

    def test_backfill_fills_description_and_step_fields(
        self, kb: KnowledgeBase,
    ) -> None:
        path = kb.save_procedure(_legacy_procedure())
        kb.save_procedure({
            "id": "complete", "title": "C", "description": "Has one.",
            "steps": [],
        })
        assert kb.backfill_procedures() == 1
        data = kb.get_procedure("legacy-flow")
        # The export follows the row.
        assert json.loads(path.read_text()) == data
        assert data["description"] == "Do the legacy flow end to end."
        step = data["steps"][0]
        assert step["app"] == "Safari"
        assert step["location"] == "https://example.com"
        assert step["target"] == "https://example.com"
        assert step["verify"] == "The page loads."
        # Idempotent: a second run changes nothing
        assert kb.backfill_procedures() == 0

    def test_backfill_empty_kb(self, tmp_path: Path) -> None:
        assert KnowledgeBase(root=tmp_path / "none", db_path=kb_db(tmp_path / "none")).backfill_procedures() == 0

    def test_cli_flag_backfills_and_exits(
        self, kb: KnowledgeBase, capsys: pytest.CaptureFixture[str],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agenthandover_worker import main as worker_main

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        kb.save_procedure(_legacy_procedure())
        worker_main.main([
            "--backfill-procedures", "--knowledge-dir", str(kb.root),
            "--db-path", str(kb.db_path),
        ])
        assert "Backfilled 1 procedure(s)" in capsys.readouterr().out
        assert kb.get_procedure("legacy-flow")["description"]
        # One-shot: no PID/status files written for the service
        assert not (tmp_path / "home").exists()

"""``main(["--once", ...])`` runs one full poll iteration and returns.

Smoke test for the worker entry point against an empty daemon database
(built from the daemon's migrations) with ``HOME`` pointed at a temp
directory.  Every Ollama host the worker reads (config and the ``ollama``
client's ``OLLAMA_HOST``) points at a closed local port and the query API
is disabled, so the run touches no service on this machine.
"""

from __future__ import annotations

import json
import signal
import socket
from pathlib import Path

import pytest

from agenthandover_worker import main as worker_main


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def once_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    host = f"http://127.0.0.1:{_closed_port()}"
    monkeypatch.setenv("OLLAMA_HOST", host)
    config_dir = worker_main._status_dir()
    if worker_main._platform.system() != "Darwin":
        config_dir = home / ".config" / "agenthandover"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        f'[vlm]\nollama_host = "{host}"\n\n'
        "[knowledge]\nquery_api_enabled = false\n",
        encoding="utf-8",
    )
    return home


def test_once_runs_one_iteration_and_cleans_up(
    once_home: Path, tmp_db_path: Path, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    status_dir = worker_main._status_dir()
    assert status_dir.is_relative_to(tmp_path)

    handlers_before = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    worker_main.main([
        "--once",
        "--db-path", str(tmp_db_path),
        "--sops-dir", str(tmp_path / "sops"),
        "--knowledge-dir", str(tmp_path / "knowledge"),
        "--adapter", "generic",
        "--log-level", "ERROR",
    ])

    loop_errors = [r for r in caplog.records if "Error in main loop" in r.getMessage()]
    assert loop_errors == []
    # Written only inside the poll loop body.
    assert (status_dir / "curation-queue.json").is_file()

    status_file = status_dir / "worker-status.json"
    assert status_file.is_file()
    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status["consecutive_errors"] == 0
    assert "annotation_paused_reason" in status
    assert status["annotation_budget"]["max_frames"] == 1500
    # The model work ran on the inference thread and --once waited for it:
    # nothing queued or running, nothing failed; the thread is stopped.
    inference = status["inference"]
    assert inference["running"] is None and inference["depth"] == 0, inference
    assert inference["failed"] == 0 and inference["completed"] >= 2, inference
    assert inference["annotation_backlog"] == 0
    assert inference["lower_capture_cadence"] is False
    from agenthandover_worker import inference_runner
    assert inference_runner._ACTIVE is None
    # The once-per-day jobs ran and recorded today's local date.
    from datetime import date

    from agenthandover_worker.db import WorkerDB
    with WorkerDB(tmp_db_path) as db:
        assert db.get_state(worker_main.DAILY_JOBS_STATE_KEY) == date.today().isoformat()
    assert not (status_dir / "worker.pid").exists()
    assert {
        signum: signal.getsignal(signum) for signum in handlers_before
    } == handlers_before


def test_vector_kb_and_matcher_reach_their_consumers(
    once_home: Path, tmp_db_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() builds the vector KB and the reasoner before the modules that
    take them, and builds one ProcedureMatcher whatever the feature flags:
    every consumer receives the live objects, not a forward-declared None."""
    from agenthandover_worker.daily_processor import DailyBatchProcessor
    from agenthandover_worker.procedure_matcher import ProcedureMatcher
    from agenthandover_worker.procedure_writer import ProcedureWriter
    from agenthandover_worker.session_linker import SessionLinker
    from agenthandover_worker.task_segmenter import TaskSegmenter
    from agenthandover_worker.vector_kb import VectorKB

    # The embedding health check passes without a model server.
    monkeypatch.setattr(
        VectorKB, "compute_embeddings",
        lambda self, texts, *, model=None: [[0.1, 0.2, 0.3] for _ in texts],
    )
    seen: dict[str, dict] = {}

    def record(cls):
        original = cls.__init__

        def init(self, *args, **kwargs):
            seen[cls.__name__] = kwargs
            original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", init)

    for cls in (DailyBatchProcessor, ProcedureMatcher, ProcedureWriter,
                SessionLinker, TaskSegmenter):
        record(cls)

    config = worker_main._status_dir() / "config.toml"
    if worker_main._platform.system() != "Darwin":
        config = once_home / ".config" / "agenthandover" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\n[features]\ncontinuity_tracking = false\n",
        encoding="utf-8",
    )

    worker_main.main([
        "--once",
        "--db-path", str(tmp_db_path),
        "--sops-dir", str(tmp_path / "sops"),
        "--knowledge-dir", str(tmp_path / "knowledge"),
        "--adapter", "generic",
        "--log-level", "ERROR",
    ])

    vector_kb = seen["ProcedureWriter"]["vector_kb"]
    assert isinstance(vector_kb, VectorKB)
    assert seen["ProcedureWriter"]["llm_reasoner"] is not None
    assert seen["SessionLinker"]["vector_kb"] is vector_kb
    assert seen["SessionLinker"]["llm_reasoner"] is not None
    assert seen["TaskSegmenter"]["vector_kb"] is vector_kb
    assert seen["ProcedureMatcher"]["vector_kb"] is vector_kb
    assert isinstance(seen["DailyBatchProcessor"]["matcher"], ProcedureMatcher)


def test_startup_migrates_a_config_on_an_earlier_tier_default(
    once_home: Path, tmp_db_path: Path, tmp_path: Path,
) -> None:
    """main() moves a [vlm] block still on the 16 GB tier's pre-WP-F6
    models to its current ones before it reads the config, once, with a
    backup beside it."""
    config = worker_main._status_dir() / "config.toml"
    if worker_main._platform.system() != "Darwin":
        config = once_home / ".config" / "agenthandover" / "config.toml"
    host_line = config.read_text(encoding="utf-8").splitlines()[1]
    old = (
        '[vlm]\nmode = "local"\ntier = "recommended"\n'
        'annotation_model = "qwen3.5:4b"\nsop_model = "gemma4:12b-it-qat"\n'
        f"{host_line}\n\n[knowledge]\nquery_api_enabled = false\n"
    )
    config.write_text(old, encoding="utf-8")

    for _ in range(2):
        worker_main.main([
            "--once",
            "--db-path", str(tmp_db_path),
            "--sops-dir", str(tmp_path / "sops"),
            "--knowledge-dir", str(tmp_path / "knowledge"),
            "--adapter", "generic",
            "--log-level", "ERROR",
        ])

    assert config.read_text(encoding="utf-8") == old.replace(
        'sop_model = "gemma4:12b-it-qat"', 'sop_model = "qwen3.5:4b"',
    )
    backups = list(config.parent.glob("config.toml.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == old


def _run_once(db_path: Path, tmp_path: Path) -> None:
    worker_main.main([
        "--once",
        "--db-path", str(db_path),
        "--sops-dir", str(tmp_path / "sops"),
        "--knowledge-dir", str(tmp_path / "knowledge"),
        "--adapter", "generic",
        "--log-level", "ERROR",
    ])


@pytest.mark.parametrize("found", [5, 99])
def test_a_database_at_another_schema_version_stops_the_worker_loudly(
    once_home: Path, tmp_db_path: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, found: int,
) -> None:
    """The daemon owns the schema: a database it has not migrated (an
    older daemon) or migrated past this worker (a newer daemon) stops the
    worker with the reason and the fix, before anything is written."""
    import sqlite3

    from agenthandover_worker.db import SCHEMA_VERSION

    monkeypatch.setattr(worker_main, "_DB_RETRY_MAX_SECONDS", 0)
    conn = sqlite3.connect(tmp_db_path)
    conn.execute(f"PRAGMA user_version = {found}")
    conn.close()

    with pytest.raises(SystemExit) as exit_info:
        _run_once(tmp_db_path, tmp_path)

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert f"schema version {found}" in err
    assert f"version {SCHEMA_VERSION}" in err
    assert ("agenthandover restart" if found < SCHEMA_VERSION else "reinstall") in err
    assert not (tmp_path / "knowledge").exists()
    assert not (worker_main._status_dir() / "worker.pid").exists()


def test_first_start_imports_a_v040_knowledge_base_once(
    once_home: Path, tmp_db_path: Path, tmp_path: Path,
) -> None:
    """Upgrading from v0.4.0: the procedure files the old worker wrote go
    into the procedures table on the first start, the files stay byte for
    byte (they are the export agents read), and later starts import
    nothing (a procedure deleted since stays deleted)."""
    import sqlite3

    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.procedure_schema import sop_to_procedure

    proc_dir = tmp_path / "knowledge" / "procedures"
    proc_dir.mkdir(parents=True)
    files = {}
    for slug, state in (("file-expenses", "agent_ready"), ("weekly-report", "draft")):
        proc = sop_to_procedure({
            "slug": slug, "title": slug.replace("-", " ").title(),
            "steps": [{"step": "Open the app", "confidence": 0.9}],
            "confidence_avg": 0.9,
        })
        proc["lifecycle_state"] = state
        path = proc_dir / f"{slug}.json"
        # v0.4.0's KnowledgeBase.atomic_write_json format.
        with open(path, "w") as f:
            json.dump(proc, f, indent=2, default=str)
        files[slug] = path.read_bytes()
    (proc_dir / "broken.json").write_text("{not json", encoding="utf-8")
    mtimes = {p.name: p.stat().st_mtime_ns for p in proc_dir.iterdir()}

    _run_once(tmp_db_path, tmp_path)

    conn = sqlite3.connect(tmp_db_path)
    rows = dict(conn.execute("SELECT slug, procedure_json FROM procedures"))
    states = dict(conn.execute("SELECT slug, lifecycle_state FROM procedures"))
    conn.close()
    assert rows == {slug: data.decode("utf-8") for slug, data in files.items()}
    assert states == {"file-expenses": "agent_ready", "weekly-report": "draft"}
    # The files are untouched: same bytes, never rewritten.
    assert {p.name: p.stat().st_mtime_ns for p in proc_dir.iterdir()} == mtimes
    for slug, data in files.items():
        assert (proc_dir / f"{slug}.json").read_bytes() == data

    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=tmp_db_path)
    assert kb.delete_procedure("weekly-report")
    (proc_dir / "weekly-report.json").write_bytes(files["weekly-report"])

    _run_once(tmp_db_path, tmp_path)

    assert [p["id"] for p in kb.list_procedures()] == ["file-expenses"]

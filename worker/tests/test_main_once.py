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

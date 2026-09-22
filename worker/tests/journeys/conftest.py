"""Journey fixtures: a temp ``HOME``, the knowledge base root, session
fixtures loaded through the daemon's migrations, the Ollama cassette, a
controllable clock and the worker driver.

Until a journey points it at a cassette, every Ollama host the worker reads
(``[vlm] ollama_host`` in ``config.toml`` and ``OLLAMA_HOST`` for the
``ollama`` client) is a closed local port, and the query API is off, so a
journey never reaches a service running on the machine.
"""

from __future__ import annotations

import socket
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from agenthandover_worker import bundle_compiler, staleness_detector
from agenthandover_worker import main as worker_main
from journeys.cassette import Cassette, open_cassette
from journeys.driver import Clock, JourneyWorker
from journeys.fixture_loader import LoadedFixture, load_fixture

# Where every journey's clock starts unless the journey moves it.
CLOCK_START = datetime(2026, 9, 15, 9, 0, 0, tzinfo=timezone.utc)


def _closed_port_host() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


@dataclass
class TempHome:
    path: Path
    monkeypatch: pytest.MonkeyPatch

    @property
    def data_dir(self) -> Path:
        """The daemon/worker data directory (``main._status_dir()``)."""
        return worker_main._status_dir()

    @property
    def config_file(self) -> Path:
        if worker_main._platform.system() == "Darwin":
            return self.data_dir / "config.toml"
        return self.path / ".config" / "agenthandover" / "config.toml"

    def configure(self, *, ollama_host: str, extra: str = "") -> None:
        """Write ``config.toml``; ``extra`` is appended verbatim (more TOML sections)."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(
            f'[vlm]\nollama_host = "{ollama_host}"\n\n'
            "[knowledge]\nquery_api_enabled = false\n"
            + (f"\n{extra}\n" if extra else ""),
            encoding="utf-8",
        )
        self.monkeypatch.setenv("OLLAMA_HOST", ollama_host)


@pytest.fixture()
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TempHome:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    th = TempHome(path=home, monkeypatch=monkeypatch)
    th.data_dir.mkdir(parents=True, exist_ok=True)
    th.configure(ollama_host=_closed_port_host())
    return th


@pytest.fixture()
def kb_root(temp_home: TempHome) -> Path:
    """The knowledge base root the installed worker uses (``~/.agenthandover/knowledge``)."""
    return temp_home.path / ".agenthandover" / "knowledge"


@pytest.fixture()
def events_db(temp_home: TempHome) -> Callable[..., LoadedFixture]:
    """``events_db(name, rebase_to=None)``: load a fixture into the data dir's ``events.db``."""

    def load(name: str, *, rebase_to: datetime | None = None) -> LoadedFixture:
        return load_fixture(name, temp_home.data_dir, rebase_to=rebase_to)

    return load


@pytest.fixture()
def cassette(temp_home: TempHome) -> Callable[..., Cassette]:
    """``cassette(journey, record=None)``: start the journey's cassette and point
    the worker's config at it.  A replay miss fails the test at teardown."""
    with ExitStack() as stack:

        def start(journey: str, *, record: bool | None = None) -> Cassette:
            tape = stack.enter_context(open_cassette(journey, record=record))
            temp_home.configure(ollama_host=tape.host)
            return tape

        yield start


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """A clock at ``CLOCK_START``, also the default ``now`` of
    ``staleness_detector.procedure_freshness`` (the WP-T1 time seam)."""
    journey_clock = Clock(CLOCK_START)
    original = staleness_detector.procedure_freshness

    def freshness_on_clock(proc: dict, now: datetime | None = None) -> float:
        return original(proc, now if now is not None else journey_clock.now())

    monkeypatch.setattr(staleness_detector, "procedure_freshness", freshness_on_clock)
    # bundle_compiler binds the name at import; the other callers import it per call.
    monkeypatch.setattr(bundle_compiler, "procedure_freshness", freshness_on_clock)
    return journey_clock


@pytest.fixture()
def worker(temp_home: TempHome, kb_root: Path, clock: Clock) -> Callable[..., JourneyWorker]:
    """``worker(fixture_or_db_path, adapter="openclaw")``: the driver over that database."""
    with ExitStack() as stack:

        def build(db: LoadedFixture | Path, *, adapter: str = "openclaw") -> JourneyWorker:
            db_path = db.db_path if isinstance(db, LoadedFixture) else Path(db)
            return stack.enter_context(JourneyWorker(
                db_path=db_path,
                knowledge_dir=kb_root,
                sops_dir=temp_home.path / ".openclaw" / "workspace" / "memory" / "apprentice",
                clock=clock,
                adapter=adapter,
            ))

        yield build

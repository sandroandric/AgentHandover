"""The worker writes each log line once, and an idle poll stays quiet.

Real-life finding (test Mac, 2026-09-23): worker.log and the launchd
stderr file worker.stderr.log were byte-identical, and a stuck VLM job
logged "Poll: 0 unprocessed events, 1 pending VLM jobs" at INFO every 2 s.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from agenthandover_worker.main import _log_handlers, _poll_log_level


def _emit(handlers: list[logging.Handler], message: str) -> None:
    log = logging.getLogger("test_worker_logging")
    log.propagate = False
    log.setLevel(logging.INFO)
    for handler in handlers:
        log.addHandler(handler)
    try:
        log.info(message)
    finally:
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()


def test_under_launchd_lines_go_to_worker_log_only(tmp_path, capsys):
    handlers = _log_handlers(tmp_path, stderr_is_tty=False)
    assert [type(h) for h in handlers] == [RotatingFileHandler]

    _emit(handlers, "one line")

    assert "one line" in (tmp_path / "worker.log").read_text()
    assert capsys.readouterr().err == ""


def test_a_manual_run_in_a_terminal_also_logs_to_stderr(tmp_path, capsys):
    handlers = _log_handlers(tmp_path, stderr_is_tty=True)
    assert len(handlers) == 2

    _emit(handlers, "visible")

    assert "visible" in (tmp_path / "worker.log").read_text()
    assert "[INFO] test_worker_logging: visible" in capsys.readouterr().err


def test_repeated_poll_counts_drop_to_debug():
    assert _poll_log_level(1, None) == logging.INFO
    assert _poll_log_level(1, 1) == logging.DEBUG
    assert _poll_log_level(3, 1) == logging.INFO
    assert _poll_log_level(3, 3) == logging.DEBUG

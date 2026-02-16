"""OpenMimic apprentice worker entry-point.

Provides the main loop that reads from the daemon's SQLite database
and will (in later tasks) drive episode building, semantic translation,
and SOP induction.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from oc_apprentice_worker.db import WorkerDB

logger = logging.getLogger("oc_apprentice_worker")

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "oc-apprentice" / "events.db"
POLL_INTERVAL_SECONDS = 2.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oc-apprentice-worker",
        description="OpenMimic apprentice worker process",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=(
            "Path to the daemon's SQLite database "
            f"(default: {DEFAULT_DB_PATH})"
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_SECONDS,
        help=(
            "Seconds between poll cycles "
            f"(default: {POLL_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Worker entry-point: connect to DB, poll for work, run until shutdown."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("Starting oc-apprentice-worker")
    logger.info("Database path: %s", args.db_path)
    logger.info("Poll interval: %.1fs", args.poll_interval)

    if not args.db_path.is_file():
        logger.error("Database file not found: %s", args.db_path)
        logger.error(
            "Is the daemon running? The daemon must create the database first."
        )
        sys.exit(1)

    shutdown_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — requesting shutdown", sig_name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    with WorkerDB(args.db_path) as db:
        logger.info("Connected to database, entering main loop")

        while not shutdown_requested:
            unprocessed = db.get_unprocessed_events(limit=100)
            pending_vlm = db.get_pending_vlm_jobs(limit=10)

            if unprocessed or pending_vlm:
                logger.info(
                    "Poll: %d unprocessed events, %d pending VLM jobs",
                    len(unprocessed),
                    len(pending_vlm),
                )
            else:
                logger.debug("Poll: nothing to do")

            # Future tasks will add processing logic here:
            #   - Episode boundary detection
            #   - VLM screenshot description
            #   - Semantic SOP extraction

            time.sleep(args.poll_interval)

    logger.info("Worker shut down cleanly")


if __name__ == "__main__":
    main()

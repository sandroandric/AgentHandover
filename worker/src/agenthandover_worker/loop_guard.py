"""Keep the worker's main loop alive through errors and poison events.

Before this module the loop exited after 10 consecutive errors, and
launchd (``KeepAlive`` restarts only on error exits) left it dead, so one
event that crashed the pipeline on every attempt stopped all processing.
Now the loop backs off instead of exiting, and an event that keeps
failing is isolated and quarantined:

* A failed batch of several events is retried at half the size, so the
  failing event ends up alone within a few ticks while the healthy events
  around it get processed.
* An event that fails alone ``MAX_SOLO_FAILURES`` times is quarantined
  (taken out of the unprocessed queue) instead of blocking the queue.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

ERROR_THRESHOLD = 10
MAX_SOLO_FAILURES = 3
_BASE_BACKOFF_SECONDS = 5.0
_MAX_BACKOFF_SECONDS = 300.0


def error_backoff_seconds(consecutive_errors: int) -> float:
    """Seconds to wait after the *consecutive_errors*-th error in a row.

    5 s below the threshold; from the threshold on, doubling from 5 s and
    capped at 5 minutes.
    """
    if consecutive_errors < ERROR_THRESHOLD:
        return _BASE_BACKOFF_SECONDS
    exponent = min(consecutive_errors - ERROR_THRESHOLD, 6)
    return min(_MAX_BACKOFF_SECONDS, _BASE_BACKOFF_SECONDS * 2**exponent)


def interruptible_sleep(
    seconds: float,
    should_stop: Callable[[], bool],
    *,
    step: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Sleep up to *seconds*, returning early once *should_stop()* is true,
    so a long backoff does not delay shutdown (launchd sends SIGKILL 20 s
    after SIGTERM)."""
    remaining = seconds
    while remaining > 0 and not should_stop():
        chunk = min(step, remaining)
        sleep(chunk)
        remaining -= chunk


class EventRetryGuard:
    """Batch sizing and quarantine decisions for the unprocessed-event queue."""

    def __init__(self, max_batch: int = 100) -> None:
        self.max_batch = max_batch
        self.batch_limit = max_batch
        self._solo_failures: dict[str, int] = {}

    def succeeded(self, event_ids: Sequence[str]) -> None:
        """The pipeline processed *event_ids*: restore the full batch size."""
        self.batch_limit = self.max_batch
        for eid in event_ids:
            self._solo_failures.pop(eid, None)

    def failed(self, event_ids: Sequence[str]) -> list[str]:
        """Record a failed pipeline attempt on *event_ids*.

        Returns the ids to quarantine now (empty unless a single event has
        failed alone ``MAX_SOLO_FAILURES`` times). A failure with no events
        in flight (an error outside event processing) changes nothing.
        """
        if not event_ids:
            return []
        if len(event_ids) > 1:
            self.batch_limit = max(1, len(event_ids) // 2)
            return []
        eid = event_ids[0]
        self.batch_limit = 1
        failures = self._solo_failures.get(eid, 0) + 1
        if failures >= MAX_SOLO_FAILURES:
            self._solo_failures.pop(eid, None)
            self.batch_limit = self.max_batch
            return [eid]
        self._solo_failures[eid] = failures
        return []

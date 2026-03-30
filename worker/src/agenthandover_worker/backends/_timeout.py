"""Timeout utility for VLM inference backends.

Wraps a callable in a ThreadPoolExecutor to enforce wall-clock timeouts
on blocking inference calls that may not respect signals.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_timeout(fn: Callable[[], T], timeout_seconds: float) -> T:
    """Run *fn* with a wall-clock timeout.

    Uses a single-thread ``ThreadPoolExecutor`` so the calling thread
    is not blocked beyond *timeout_seconds*.

    Note: On timeout, the background thread may continue running (Python
    threads cannot be forcibly killed). The pool is shut down without
    waiting so the caller returns promptly.

    Raises:
        TimeoutError: If *fn* does not complete within the deadline.
        Exception: Any exception raised by *fn* is re-raised.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        raise TimeoutError(
            f"Inference timed out after {timeout_seconds:.0f}s"
        ) from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

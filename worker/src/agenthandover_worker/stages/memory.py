"""Memory stage (inference thread, P2): embed the kept frames waiting
(``memory.MemoryWriter.run``), before the screenshots are deleted.

Submitted ahead of passive annotation each tick while ``[memory] enabled``;
it runs without the idle gate and the annotation budget (frames that are
never annotated still enter Memory), within ``[memory]
max_frames_per_hour`` and the model scheduler's memory guard.
"""

from __future__ import annotations

from typing import Any


def run(ctx: Any, tick: Any) -> dict:
    """Embed up to one job's worth of frames; the counts are also recorded
    on *tick*."""
    tick.memory = ctx.memory_writer.run()
    return tick.memory

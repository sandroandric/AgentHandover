"""Embedding upgrade stage (inference thread, P2): "Upgrade everything"'s
background re-embed and, when it is complete, the switch of the worker's
text paths to the new model (``embedding_upgrade``).

Submitted each tick while the upgrade is wanted or its switch is half
done; :func:`finish` also runs once before the first tick
(``context.start``), so a restart between the switch's transaction and its
config write completes it before anything embeds with the old model.
"""

from __future__ import annotations

from typing import Any


def due(ctx: Any) -> bool:
    """True when the job has work: the upgrade is wanted, or one that
    started is not done (its switch to complete, or its staged rows to
    drop when the offer's answer changed)."""
    upgrade = ctx.embedding_upgrade
    if upgrade is None:
        return False
    status = upgrade.status()
    return upgrade.wanted() or (status is not None and status["state"] != "done")


def run(ctx: Any, tick: Any) -> dict | None:
    """Re-embed one batch, or switch; the upgrade's status is also
    recorded on *tick*."""
    upgrade = ctx.embedding_upgrade
    if upgrade.step():
        finish(ctx)
    tick.embedding_upgrade = upgrade.status()
    return tick.embedding_upgrade


def finish(ctx: Any) -> bool:
    """Complete a switch whose rows are replaced: the worker's objects
    embed and search with the new model, then config.toml says so.  True
    when there was a switch to complete and it completed."""
    from agenthandover_worker.context import use_embedding_model
    from agenthandover_worker.embedding_upgrade import TARGET_MODEL

    upgrade = ctx.embedding_upgrade
    if upgrade is None or (upgrade.saved_state() or {}).get("state") != "switched":
        return False
    if ctx.cfg.embedding.model != TARGET_MODEL:
        use_embedding_model(ctx, TARGET_MODEL)
    return upgrade.complete()

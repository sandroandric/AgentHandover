"""The session linker's cosine threshold against real nomic-embed-text vectors.

``SessionLinker`` links daily-summary tasks across days by intent.  With a
live vector KB it scores a pair by the cosine of the intents'
``clustering:``-prefixed embeddings; without one, by token Jaccard.  The two
scores live on different scales, so each has its own threshold
(``_JACCARD_THRESHOLD``, and the embedding profile's ``linker_cos``); this
file pins nomic-embed-text's cosine one on the model's real output (answered by the recorded cassette).

The set: intents of five recurring tasks, each phrased the way three
different days' summaries phrase it, plus hard negatives that share a site,
an app or a verb with one of them.  The pin is the one that matters for
mining: no pair from two different tasks reaches the threshold (a false
link merges two tasks' history).  Same-task pairs that fall below it are
left to the LLM band just under the threshold, or stay unlinked (a missed
link costs one day of history).  Recorded: same-task cosines 0.76-0.96,
the closest different-task pair (Stripe payout vs. failed Stripe payments)
0.88, hence 0.89.

Record: ``AH_RECORD=1 AH_RECORD_UPSTREAM=http://<ollama> pytest <this file>``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.model_profiles import EMBEDDING_PROFILES
from agenthandover_worker.session_linker import SessionLinker
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
from journeys.cassette import open_cassette
from daemon_db import create_daemon_db, kb_db

JOURNEY = "linker_threshold"
# The linker's cosine threshold and LLM band: nomic-embed-text's profile.
_THRESHOLDS = EMBEDDING_PROFILES["nomic-embed-text"].thresholds
_COSINE_THRESHOLD = _THRESHOLDS.linker_cos
_COSINE_BAND = _THRESHOLDS.linker_band

# (task, intent as a daily summary words it)
INTENTS = [
    ("stripe", "Reconciling yesterday's Stripe payout in the reconciliation sheet"),
    ("stripe", "Entering the Stripe payout amount into the payout reconciliation Google Sheet"),
    ("stripe", "Checking the latest Stripe payout and marking it reconciled"),
    ("expense", "Filing the weekly expense report in Expensify"),
    ("expense", "Adding last week's receipts to an Expensify expense report"),
    ("expense", "Submitting the weekly expense report with receipts for approval"),
    ("standup", "Writing the daily standup update in Slack"),
    ("standup", "Posting yesterday's progress and today's plan in the standup channel"),
    ("standup", "Sharing the daily standup notes with the team on Slack"),
    ("pr", "Reviewing a pull request on GitHub"),
    ("pr", "Leaving review comments on a GitHub pull request"),
    ("pr", "Approving a teammate's pull request after reviewing the diff"),
    ("invoice", "Sending the monthly invoice to a client from QuickBooks"),
    ("invoice", "Creating and emailing this month's client invoice in QuickBooks"),
    ("invoice", "Preparing the client invoice for the month in QuickBooks"),
    # Hard negatives: a shared site, app or verb, a different task.
    ("stripe_failed", "Looking into failed payments in the Stripe dashboard"),
    ("budget", "Reviewing the Q3 budget in a Google Sheet"),
    ("bug", "Filing a bug report in GitHub issues"),
    ("slack_support", "Answering a customer question in the Slack support channel"),
    ("expense_policy", "Reading the company travel expense policy"),
]


def _scores(tmp_path: Path) -> list[tuple[str, str, float]]:
    """``(task_a, task_b, cosine)`` for every pair, in a fixed order."""
    with open_cassette(JOURNEY) as tape:
        linker = SessionLinker(
            KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge")),
            vector_kb=VectorKB(create_daemon_db(tmp_path / "vectors.db"), VectorKBConfig(ollama_host=tape.host)),
        )
        out = []
        for (ta, a), (tb, b) in combinations(INTENTS, 2):
            score, metric = linker._intent_similarity(a, b)
            assert metric == "cosine"
            out.append((ta, tb, score))
        return out


def test_no_two_tasks_reach_the_cosine_threshold(tmp_path: Path) -> None:
    scores = _scores(tmp_path)
    cross = [(ta, tb, round(s, 3)) for ta, tb, s in scores if ta != tb and s >= _COSINE_THRESHOLD]
    assert cross == []


def test_most_same_task_pairs_link_or_reach_the_llm_band(tmp_path: Path) -> None:
    same = [s for ta, tb, s in _scores(tmp_path) if ta == tb]
    linked = sum(s >= _COSINE_THRESHOLD for s in same)
    in_band = sum(_COSINE_THRESHOLD - _COSINE_BAND <= s < _COSINE_THRESHOLD for s in same)
    assert linked >= len(same) // 2, (linked, len(same))
    assert linked + in_band >= 0.8 * len(same), (linked, in_band, len(same))


def test_the_jaccard_threshold_would_link_different_tasks(tmp_path: Path) -> None:
    """Why the cosine needs its own threshold: at the Jaccard-calibrated
    0.4, real cosines link pairs from different tasks."""
    cross = [s for ta, tb, s in _scores(tmp_path) if ta != tb and s >= 0.4]
    assert cross, "every cross-task cosine is under 0.4; re-check the calibration"


def test_the_segmenter_threshold_would_link_different_tasks(tmp_path: Path) -> None:
    """Why the linker does not reuse the segmenter's 0.83: on intent texts
    it links the Stripe payout routine with a failed-payments lookup."""
    cross = {(ta, tb) for ta, tb, s in _scores(tmp_path) if ta != tb and s >= 0.83}
    assert ("stripe", "stripe_failed") in cross

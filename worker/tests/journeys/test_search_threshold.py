"""Skill search's score floor and the procedure matcher's threshold against
real embedding vectors, for every embedding model the worker can use.

Eight procedures are embedded the way the worker embeds an approved Skill
(``main._embed_procedure``: title, description, strategy, steps) into a
real ``VectorKB``.  Three kinds of text are then searched against them
through ``VectorKB.search`` with Ollama answered by the recorded cassette:

* agent questions (what ``search_skills`` / ``/search/semantic`` receive),
  two per procedure, phrased without the procedure's own words where a
  person would;
* task labels (what ``ProcedureMatcher.match_segment`` sends: a segment's
  label, the way the segmenter's annotations word a task);
* unrelated questions, which no procedure answers.

The pins, per model:

* ``min_score`` (the search floor): every agent question finds its
  procedure in the top two and above the floor, and at least 15 of 16
  first; the floor only drops noise, ranking does the rest.  Recorded:
  nomic-embed-text ranks all 16 first; EmbeddingGemma 2 ranks "get my
  receipts reimbursed" second (the Stripe payout procedure 0.674, the
  expense report 0.618).  Lowest question-procedure score: nomic 0.417
  (floor 0.3), EmbeddingGemma 0.488 (floor 0.4).
* ``matcher``: no task label reaches the threshold against another
  procedure (a false match adds evidence to the wrong Skill), most labels
  reach their own, and no unrelated question reaches any procedure.
* ``dedup``: a new Skill drafted from another demonstration of the same
  task (``find_matching_sop``'s query, ``title | description``, worded
  differently) reaches its procedure, and none reaches another one.

Neither the matcher's nor dedup's semantic path has a production caller
today (``ProcedureMatcher.match_span`` and ``deduplicate_templates`` use
fingerprints); the numbers are measured so that the paths are right when
one is wired.

Record: ``AH_RECORD=1 AH_RECORD_UPSTREAM=http://<ollama> pytest <this file>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agenthandover_worker.main import _embed_procedure
from agenthandover_worker.model_profiles import EMBEDDING_MODEL, EMBEDDING_PROFILES, MEMORY_MODEL
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
from journeys.cassette import open_cassette
from daemon_db import create_daemon_db

JOURNEY = "search_threshold"
MODELS = [EMBEDDING_MODEL, MEMORY_MODEL]

PROCEDURES = [
    {
        "id": "reconcile-stripe-payout",
        "title": "Reconcile the Stripe payout",
        "description": "Match yesterday's Stripe payout against the bank deposit and mark it reconciled in the finance sheet.",
        "steps": [
            {"action": "Open the Stripe dashboard and go to Payouts"},
            {"action": "Copy the latest payout amount and arrival date"},
            {"action": "Find the matching deposit in the bank statement"},
            {"action": "Enter the amount in the Payout Reconciliation Google Sheet and mark it reconciled"},
        ],
    },
    {
        "id": "submit-weekly-expenses",
        "title": "Submit the weekly expense report",
        "description": "Collect the week's receipts and submit an expense report in Expensify for approval.",
        "steps": [
            {"action": "Open Expensify and create a new report"},
            {"action": "Upload each receipt and set its category"},
            {"action": "Submit the report to the manager for approval"},
        ],
    },
    {
        "id": "post-daily-standup",
        "title": "Post the daily standup",
        "description": "Write yesterday's progress, today's plan and blockers in the team's standup channel on Slack.",
        "steps": [
            {"action": "Open the #standup channel in Slack"},
            {"action": "Write what was done yesterday, today's plan and any blockers"},
            {"action": "Post the message"},
        ],
    },
    {
        "id": "review-pull-request",
        "title": "Review a pull request",
        "description": "Read a teammate's GitHub pull request, comment on the changed lines and approve or request changes.",
        "steps": [
            {"action": "Open the pull request on GitHub and read the description"},
            {"action": "Go through the Files changed tab and leave comments"},
            {"action": "Approve or request changes"},
        ],
    },
    {
        "id": "send-client-invoice",
        "title": "Send the monthly client invoice",
        "description": "Create this month's invoice for the client in QuickBooks and email it.",
        "steps": [
            {"action": "Open QuickBooks and create an invoice for the client"},
            {"action": "Add the month's billable hours and check the total"},
            {"action": "Email the invoice to the client's billing address"},
        ],
    },
    {
        "id": "deploy-staging",
        "title": "Deploy to staging",
        "description": "Run the deploy script for the staging environment and check the logs and health page.",
        "strategy": "Deploy from the main branch only after CI is green.",
        "steps": [
            {"action": "Pull the latest main branch in the terminal"},
            {"action": "Run ./deploy.sh staging"},
            {"action": "Watch the deploy logs for errors and open the staging health page"},
        ],
    },
    {
        "id": "book-meeting-room",
        "title": "Book a meeting room",
        "description": "Find a free meeting room for a team meeting in Google Calendar and add it to the invite.",
        "steps": [
            {"action": "Open the event in Google Calendar"},
            {"action": "Click Rooms and pick a free room with enough seats"},
            {"action": "Save the event and notify the guests"},
        ],
    },
    {
        "id": "update-crm-lead",
        "title": "Update a lead in the CRM",
        "description": "Log a sales call with a lead in HubSpot, update the deal stage and set a follow-up task.",
        "steps": [
            {"action": "Open the contact in HubSpot"},
            {"action": "Log the call notes"},
            {"action": "Move the deal to the next stage and create a follow-up task"},
        ],
    },
]

# (procedure id, agent question)
QUESTIONS = [
    ("reconcile-stripe-payout", "how do I reconcile a Stripe payout"),
    ("reconcile-stripe-payout", "check that the money from Stripe arrived in the bank and record it"),
    ("submit-weekly-expenses", "file my expenses for this week"),
    ("submit-weekly-expenses", "get my receipts reimbursed"),
    ("post-daily-standup", "write the standup update"),
    ("post-daily-standup", "tell the team what I did yesterday and what I'm doing today"),
    ("review-pull-request", "review a PR on GitHub"),
    ("review-pull-request", "look over a colleague's code changes and approve them"),
    ("send-client-invoice", "bill the client for this month"),
    ("send-client-invoice", "create and email an invoice in QuickBooks"),
    ("deploy-staging", "deploy to the staging server"),
    ("deploy-staging", "push the latest build to the test environment"),
    ("book-meeting-room", "reserve a conference room for a meeting"),
    ("book-meeting-room", "find a free room for the team sync"),
    ("update-crm-lead", "log a sales call in HubSpot"),
    ("update-crm-lead", "move a deal forward after talking to a prospect"),
]

# (procedure id, task label as the segmenter's annotations word it)
LABELS = [
    ("reconcile-stripe-payout", "Reconciling Stripe payout in Google Sheets"),
    ("submit-weekly-expenses", "Filing expense report in Expensify"),
    ("post-daily-standup", "Writing standup update in Slack"),
    ("review-pull-request", "Reviewing pull request diff on GitHub"),
    ("send-client-invoice", "Creating client invoice in QuickBooks"),
    ("deploy-staging", "Running deploy script for staging"),
    ("book-meeting-room", "Booking a room in Google Calendar"),
    ("update-crm-lead", "Updating deal stage in HubSpot"),
]

# (procedure id, "title | description" of a new draft of the same task)
REDRAFTS = [
    ("reconcile-stripe-payout", "Stripe payout reconciliation | Check the Stripe payout against the bank account and record it in the reconciliation spreadsheet."),
    ("submit-weekly-expenses", "Weekly Expensify report | Add this week's receipts to a report and send it off for manager approval."),
    ("post-daily-standup", "Daily standup message | Share what got done yesterday and the plan for today in the Slack standup channel."),
    ("review-pull-request", "GitHub code review | Go through a teammate's PR, comment on the diff and approve it."),
    ("send-client-invoice", "Monthly invoicing | Prepare the client's invoice for the month in QuickBooks and send it by email."),
    ("deploy-staging", "Staging deployment | Pull main, run the staging deploy script and verify the health page."),
    ("book-meeting-room", "Reserve a room for a meeting | Add a free conference room to a Google Calendar event."),
    ("update-crm-lead", "Log a call with a prospect | Record call notes in HubSpot and advance the deal stage."),
]

UNRELATED = [
    "book a flight to Lisbon",
    "what is the weather tomorrow",
    "convert a video to mp4",
    "order groceries for the week",
    "change my desktop wallpaper",
    "learn to play the guitar",
]


def _scores(tmp_path: Path, model: str, texts: list[str]) -> list[dict[str, float]]:
    """For each text, ``{procedure id: score}`` over all procedures."""
    with open_cassette(JOURNEY) as tape:
        vkb = VectorKB(
            create_daemon_db(tmp_path / "vectors.db"),
            VectorKBConfig(ollama_host=tape.host, embedding_model=model),
        )
        for procedure in PROCEDURES:
            _embed_procedure(vkb, procedure)
        out = []
        for text in texts:
            results = vkb.search(text, top_k=len(PROCEDURES), source_types=["procedure"])
            assert len(results) == len(PROCEDURES), text
            out.append({r.source_id: r.score for r in results})
        return out


def _thresholds(model: str):
    return EMBEDDING_PROFILES[model].thresholds


@pytest.mark.parametrize("model", MODELS)
def test_every_question_finds_its_procedure_near_the_top_and_above_the_floor(
    tmp_path: Path, model: str,
) -> None:
    floor = _thresholds(model).min_score
    scores = _scores(tmp_path, model, [q for _, q in QUESTIONS])
    ranks = []
    for (want, question), by_id in zip(QUESTIONS, scores):
        ranked = sorted(by_id, key=by_id.get, reverse=True)
        ranks.append((question, ranked.index(want) + 1, round(by_id[want], 3)))
    assert [r for r in ranks if r[1] > 2 or r[2] < floor] == []
    assert sum(rank == 1 for _, rank, _ in ranks) >= len(QUESTIONS) - 1, ranks
    every_score = min(score for by_id in scores for score in by_id.values())
    assert every_score >= floor, "the floor should only drop noise"


@pytest.mark.parametrize("model", MODELS)
def test_no_task_label_matches_another_procedure(tmp_path: Path, model: str) -> None:
    threshold = _thresholds(model).matcher
    scores = _scores(tmp_path, model, [label for _, label in LABELS])
    wrong = [
        (label, other, round(score, 3))
        for (want, label), by_id in zip(LABELS, scores)
        for other, score in by_id.items()
        if other != want and score >= threshold
    ]
    assert wrong == []


@pytest.mark.parametrize("model", MODELS)
def test_most_task_labels_match_their_procedure(tmp_path: Path, model: str) -> None:
    threshold = _thresholds(model).matcher
    scores = _scores(tmp_path, model, [label for _, label in LABELS])
    matched = [by_id[want] >= threshold for (want, _), by_id in zip(LABELS, scores)]
    assert sum(matched) >= 0.75 * len(LABELS), [
        (label, round(by_id[want], 3)) for (want, label), by_id in zip(LABELS, scores)
    ]


@pytest.mark.parametrize("model", MODELS)
def test_no_unrelated_question_reaches_the_matcher_threshold(
    tmp_path: Path, model: str,
) -> None:
    threshold = _thresholds(model).matcher
    scores = _scores(tmp_path, model, UNRELATED)
    reached = [
        (text, max(by_id, key=by_id.get), round(max(by_id.values()), 3))
        for text, by_id in zip(UNRELATED, scores)
        if max(by_id.values()) >= threshold
    ]
    assert reached == []


@pytest.mark.parametrize("model", MODELS)
def test_a_redraft_reaches_its_procedure_and_no_other(tmp_path: Path, model: str) -> None:
    threshold = _thresholds(model).dedup
    scores = _scores(tmp_path, model, [text for _, text in REDRAFTS])
    missed = [
        (want, round(by_id[want], 3))
        for (want, _), by_id in zip(REDRAFTS, scores)
        if by_id[want] < threshold
    ]
    wrong = [
        (want, other, round(score, 3))
        for (want, _), by_id in zip(REDRAFTS, scores)
        for other, score in by_id.items()
        if other != want and score >= threshold
    ]
    assert (missed, wrong) == ([], [])

"""Micro-questions: what AgentHandover asks the user, when, and how often.

Every question comes from what the ledger (:mod:`ledger`) or the
observation log (:mod:`workflow_mining`) does not know yet; no model writes
them.  Two kinds:

- a **slot question** per ``unknown`` or ``hypothesis`` slot of a
  procedure (``What starts this task?``), its choices drawn from what the
  runs showed: the recurring lead-in or start time, the commit that ended
  the runs, the checks after it, and for decisions the reasons the
  entries the user skipped were skipped (from the chosen-versus-visible
  criteria of :mod:`contrast`).  The answer lands in the procedure's
  ``agent_clarifications`` with the slot as its category, so the ledger
  makes the slot ``known`` with an ``answer`` evidence pointer.  Where
  an agent run deviated at a decision (:mod:`skill_improver` queues an
  unanswered ``decision`` entry in ``agent_clarifications``), that entry
  is the decisions question, asked as a review card; its answer fills
  that entry;
- a **labelling card** per newly discovered workflow (a group of at least
  :data:`MIN_LABEL_OBSERVATIONS` work sessions no one labelled yet): "Was
  this a routine you would hand off?" yes / no / not sure, and a name.
  The answer is written on the workflow's observations
  (``workflow_observations.user_label`` / ``user_is_routine``); a ``no``
  rejects the workflow's draft and keeps it out of the review queue for
  :data:`workflow_mining.SUPPRESS_DAYS` days.  These answers are the
  ground truth ``replay --labels answers`` scores against.

When they are asked: a focus recording's questions at once, in the Q&A
window (at most :data:`FOCUS_MAX_QUESTIONS`); the rest as review cards
(``micro-questions.json``, which the app watches every few seconds) right
after the evidence that raised them was recorded (a passive discovery
pass), and every card still open is listed in the day's digest.

How often: at most the weekly budget (``[knowledge]
weekly_question_budget``, default :data:`WEEKLY_BUDGET`) questions are
asked in any seven days, the most valuable first (:func:`next_questions`),
and each question once.  A question skipped or left open changes
nothing: its slot stays what it was.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from agenthandover_worker.ledger import REQUIRED_SLOTS, derive_knowledge
from agenthandover_worker.skill_improver import RUN_QUESTION

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB
    from agenthandover_worker.focus_questioner import FocusQuestion
    from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

WEEKLY_BUDGET = 5
FOCUS_MAX_QUESTIONS = 3
#: A workflow seen this often gets a labelling card.
MIN_LABEL_OBSERVATIONS = 2
MICRO_QUESTIONS_FILE = "micro-questions.json"

KIND_SLOT = "slot"
KIND_LABEL = "label"
#: Asked in the focus Q&A window, or as a review card (and in the digest).
CHANNEL_FOCUS = "focus"
CHANNEL_CARD = "card"

#: The context of a run question whose deviation carried no detail, and
#: the digest's line under it.
RUN_CONTEXT = "Asked after an agent run"

LABEL_YES = "Yes"
LABEL_NO = "No"
LABEL_NOT_SURE = "Not sure"
_ROUTINE = {LABEL_YES.lower(): True, LABEL_NO.lower(): False}

#: The slots a question can fill, most valuable first (``steps`` comes
#: from the frames; ``voice`` and ``context`` are optional).
ASKED_SLOTS: tuple[str, ...] = (
    "goal", "done_criteria", "trigger", "decisions", "verification",
    "inputs", "exceptions", "guardrails",
)


@dataclass
class Question:
    """One question, as the app shows it and the worker records it."""

    id: str
    kind: str  # KIND_SLOT or KIND_LABEL
    subject: str  # the procedure's slug, or the workflow's key
    title: str  # what the subject is called
    question: str
    context: str
    choices: list[str] = field(default_factory=list)
    slot: str | None = None
    # A slot's hypothesis value: the Q&A window pre-fills it as a guess.
    default: str = ""
    # Labelling cards: the name field, and every key of the workflow.
    asks_name: bool = False
    workflow_keys: list[str] = field(default_factory=list)
    obs_ids: list[str] = field(default_factory=list)
    value: float = 0.0
    # A question an agent run queued (``skill_improver``): the ``origin``
    # of its entry in ``agent_clarifications``, which the answer fills.
    origin: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Question":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# ---------------------------------------------------------------------------
# Slot questions
# ---------------------------------------------------------------------------


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _skip_reason(criterion: dict) -> str | None:
    """Why an entry the user did not pick was skipped, from a
    chosen-versus-visible criterion (``contrast.criteria_hypotheses``)."""
    phrase = _text(criterion.get("criterion"))
    if not phrase:
        return None
    attribute, op = criterion.get("attribute"), criterion.get("op")
    reason = {
        ("age_minutes", "<="): "Too old",
        ("age_minutes", ">="): "Too new",
        ("price", "<="): "Too expensive",
        ("price", ">="): "Too cheap",
        ("position", "<="): "Too far down the list",
        ("position", ">="): "Too high in the list",
        ("keyword", "contains"): "Off-topic",
    }.get((attribute, op))
    if reason is None and isinstance(attribute, str) and op in (">=", "<="):
        reason = f"Too {'few' if op == '>=' else 'many'} {attribute}s"
    return f"{reason} (the ones you pick {phrase})" if reason else None


def _learned(proc: dict, name: str) -> dict:
    learned = proc.get("learned") if isinstance(proc.get("learned"), dict) else {}
    item = learned.get(name)
    return item if isinstance(item, dict) else {}


def _slot_text(proc: dict, slot: str) -> tuple[str, str, list[str], str]:
    """(question, context, choices, guess) for *slot* of *proc*."""
    if slot == "goal":
        goal = _text(proc.get("goal") or proc.get("task_description"))
        return (
            "What is this task for?",
            "The goal was inferred from what the recording showed.",
            [goal] if goal else [], goal,
        )
    if slot == "done_criteria":
        done = _text(_learned(proc, "done").get("description"))
        return (
            "How do you know this task is finished?",
            "An agent needs to know when to stop.",
            [done] if done else [], done,
        )
    if slot == "trigger":
        trigger = _learned(proc, "trigger")
        choices = []
        if trigger.get("calendar_event"):
            choices.append(f"The calendar event \"{_text(trigger['calendar_event'])}\"")
        if _text(trigger.get("description")):
            choices.append(_text(trigger["description"]))
        choices.append("Nothing fixed: I start it when I decide to")
        return (
            "What starts this task?",
            "An agent needs to know when to run it.",
            choices, choices[0] if len(choices) > 1 else "",
        )
    if slot == "decisions":
        reasons = [
            r for c in proc.get("selection_criteria") or [] if isinstance(c, dict)
            if (r := _skip_reason(c))
        ]
        if reasons:
            return (
                "Why do you skip the entries you don't pick?",
                "The runs show you pick some entries of a list and skip others.",
                reasons, "",
            )
        return (
            "How do you choose between the options in this task?",
            "An agent will meet the same choices.", [], "",
        )
    if slot == "verification":
        checks = [
            _text(c.get("description"))
            for c in _learned(proc, "verification").get("checks") or []
            if isinstance(c, dict) and _text(c.get("description"))
        ]
        return (
            "How do you check that this task worked?",
            "An agent should check its work the way you do.",
            [*checks, "I don't check it"], checks[0] if checks else "",
        )
    if slot == "inputs":
        names = [
            str(v.get("name")) for v in proc.get("inputs") or []
            if isinstance(v, dict) and v.get("name")
        ]
        return (
            "Which values change from one run to the next?",
            "An agent needs to be given these each time.",
            [], ", ".join(names),
        )
    if slot == "exceptions":
        return (
            "What goes wrong in this task, and what do you do then?",
            "An agent should handle it the way you do.",
            ["Nothing usually goes wrong"], "",
        )
    if slot == "guardrails":
        return (
            "Is there anything an agent must never do in this task?",
            "Guardrails are what the agent is told never to do.",
            ["No, nothing special"], "",
        )
    raise KeyError(slot)


def _run_questions(proc: dict) -> list[dict]:
    """The questions agent runs queued on *proc* that are still open
    (``skill_improver.feed_ledger``: unanswered ``decision`` entries of
    ``agent_clarifications`` whose ``origin`` is the run)."""
    clarifications = proc.get("agent_clarifications")
    return [
        item for item in (clarifications if isinstance(clarifications, list) else [])
        if isinstance(item, dict) and item.get("status") == "unanswered"
        and isinstance(item.get("origin"), dict)
        and item["origin"].get("kind") == RUN_QUESTION
        and not _text(item.get("answer"))
    ]


def slot_questions(proc: dict) -> list[Question]:
    """One question per ``unknown`` or ``hypothesis`` slot of *proc* that
    no answer settled yet (a slot the user already answered, even with an
    accepted guess, is not asked again).  Where agent runs queued
    questions about the decisions, those are its questions, one per
    deviated step, instead of the general one."""
    slug = str(proc.get("id") or proc.get("slug") or "")
    if not slug:
        return []
    ledger = derive_knowledge(proc)
    value = float((proc.get("value") or {}).get("value_score") or 0.0)
    title = _text(proc.get("title")) or slug
    out = []
    for slot in ASKED_SLOTS:
        entry = ledger[slot]
        if entry["status"] == "known" or entry.get("source") == "user":
            continue
        # Unknown required slots first, then unknown, then hypotheses.
        tier = 0 if entry["status"] == "unknown" and slot in REQUIRED_SLOTS else (
            1 if entry["status"] == "unknown" else 2
        )
        rank = _rank_value(tier, ASKED_SLOTS.index(slot), value)
        runs = _run_questions(proc) if slot == "decisions" else []
        if runs:
            out.extend(
                Question(
                    id=f"run:{slug}:{item['origin'].get('id', '')}:{item['origin'].get('step_id', '')}",
                    kind=KIND_SLOT, subject=slug, title=title,
                    question=_text(item.get("question")),
                    context=_text(item.get("context")) or RUN_CONTEXT,
                    slot=slot, origin=dict(item["origin"]), value=rank,
                )
                for item in runs
            )
            continue
        question, context, choices, guess = _slot_text(proc, slot)
        out.append(Question(
            id=f"slot:{slug}:{slot}", kind=KIND_SLOT, subject=slug,
            title=title, question=question, context=context,
            choices=list(dict.fromkeys(choices)), slot=slot,
            default=guess if entry["status"] == "hypothesis" else "",
            value=rank,
        ))
    return out


def _rank_value(tier: int, order: int, value_score: float) -> float:
    """Higher is asked first: labelling cards (tier -1), then slots by
    tier, slot order and the procedure's value (hours a month)."""
    return (3 - tier) * 1_000_000 - order * 10_000 + min(value_score / 3600, 9_999)


# ---------------------------------------------------------------------------
# Labelling cards
# ---------------------------------------------------------------------------


def label_questions(db: "WorkerDB", *, now: datetime) -> list[Question]:
    """One labelling card per workflow group of the observation window
    seen at least :data:`MIN_LABEL_OBSERVATIONS` times, never labelled and
    not recorded in a focus session."""
    from agenthandover_worker.workflow_mining import about_hours, load_window

    win = load_window(db, now)
    out = []
    for group in win.groups:
        observations = win.window_observations(group)
        if len(observations) < MIN_LABEL_OBSERVATIONS:
            continue
        if any(o.get("labelled_at") for o in group.observations):
            continue
        # A workflow the user recorded on purpose is one they hand off.
        if any(o.get("source") == "focus" for o in group.observations):
            continue
        value = win.value(group)
        apps = list(dict.fromkeys(
            a for o in observations for a in json.loads(o.get("apps_json") or "[]")
        ))
        hosts = list(dict.fromkeys(
            h for o in observations for h in json.loads(o.get("hosts_json") or "[]")
        ))
        where = ", ".join(apps + hosts) or "the same apps"
        name = group.task_label or group.promoted_slug or ""
        out.append(Question(
            id=f"label:{group.source_id}", kind=KIND_LABEL, subject=group.source_id,
            title=name or f"A workflow in {where}",
            question="Was this a routine you would hand off?",
            context=(
                f"Seen {len(observations)} times, {about_hours(value)}, in {where}."
            ),
            choices=[LABEL_YES, LABEL_NO, LABEL_NOT_SURE],
            default=name, asks_name=True,
            workflow_keys=sorted({o["workflow_key"] for o in group.observations}),
            obs_ids=[o["obs_id"] for o in group.observations],
            value=4 * 1_000_000 + min(value.get("value_score", 0.0) / 3600, 9_999),
        ))
    return out


# ---------------------------------------------------------------------------
# Choosing and recording
# ---------------------------------------------------------------------------


def next_questions(
    procs: list[dict],
    budget: int,
    *,
    workflows: list[Question] = (),
    asked: set[str] = frozenset(),
) -> list[Question]:
    """The *budget* most valuable questions not asked yet: labelling cards
    (*workflows*, :func:`label_questions`) first, highest value first, then
    one question per unknown or hypothesis slot of *procs*: unknown
    required slots, other unknown slots, hypotheses, each in
    :data:`ASKED_SLOTS` order and the more valuable procedure first."""
    candidates = [
        q for q in [*workflows, *(q for p in procs for q in slot_questions(p))]
        if q.id not in asked
    ]
    candidates.sort(key=lambda q: (-q.value, q.id))
    return candidates[:max(budget, 0)]


def budget_left(db: "WorkerDB", *, now: datetime, weekly_budget: int = WEEKLY_BUDGET) -> int:
    """Questions that may still be asked in the seven days before *now*."""
    return max(weekly_budget - db.count_micro_questions_since(now - timedelta(days=7)), 0)


def _asked(db: "WorkerDB") -> tuple[set[str], set[str]]:
    """(ids of every question asked, workflow keys a card was asked about)."""
    ids, keys = set(), set()
    for row in db.get_micro_questions():
        ids.add(row["question_id"])
        if row["kind"] == KIND_LABEL:
            keys.update(json.loads(row["question_json"]).get("workflow_keys") or [])
    return ids, keys


def record_asked(
    db: "WorkerDB", questions: list[Question], *, channel: str, now: datetime,
) -> None:
    from agenthandover_worker.db import _sqlite_utc

    db.save_micro_questions([
        {
            "question_id": q.id, "kind": q.kind, "subject": q.subject, "slot": q.slot,
            "channel": channel, "question_json": json.dumps(q.to_json()),
            "asked_at": _sqlite_utc(now),
        }
        for q in questions
    ])


# ---------------------------------------------------------------------------
# Focus recordings: asked in the Q&A window
# ---------------------------------------------------------------------------


def ask_after_focus(
    db: "WorkerDB", proc: dict, *, now: datetime, weekly_budget: int = WEEKLY_BUDGET,
) -> list["FocusQuestion"]:
    """The questions for a focus recording's procedure, recorded as asked:
    at most :data:`FOCUS_MAX_QUESTIONS`, within the weekly budget."""
    from agenthandover_worker.focus_questioner import FocusQuestion

    budget = min(FOCUS_MAX_QUESTIONS, budget_left(db, now=now, weekly_budget=weekly_budget))
    ids, _ = _asked(db)
    # A run question is about an agent's run, not this recording: it is
    # asked as a review card, whose answer fills the run's own entry.
    runs = {q.id for q in slot_questions(proc) if q.origin}
    chosen = next_questions([proc], budget, asked=ids | runs)
    record_asked(db, chosen, channel=CHANNEL_FOCUS, now=now)
    if chosen or budget == 0:
        logger.info(
            "Focus questions: %d asked (%d left in this week's budget of %d)",
            len(chosen), budget - len(chosen), weekly_budget,
        )
    return [
        FocusQuestion(
            question=q.question, category=q.slot or "", context=q.context,
            default=q.default, id=q.id, choices=q.choices,
        )
        for q in chosen
    ]


def close_focus_questions(db: "WorkerDB", questions_data: dict, *, now: datetime) -> None:
    """Close the focus questions the app answered or skipped: a question
    with a non-blank answer is ``answered``, every other one ``skipped``."""
    answered = (
        questions_data.get("answers") or {} if questions_data.get("status") == "answered" else {}
    )
    for raw in questions_data.get("questions") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        answer = str(answered.get(str(raw.get("index"))) or "").strip()
        db.close_micro_question(
            raw["id"], status="answered" if answer else "skipped", answer=answer, at=now,
        )


# ---------------------------------------------------------------------------
# Review cards: asked after the evidence, answered through the spool
# ---------------------------------------------------------------------------


def refresh(
    db: "WorkerDB",
    kb: "KnowledgeBase",
    state_dir: Path,
    *,
    now: datetime,
    weekly_budget: int = WEEKLY_BUDGET,
) -> list[Question]:
    """Ask what the budget allows (labelling cards for new workflows,
    slot questions for the knowledge base's procedures) and write every
    open card to ``micro-questions.json``.  Returns the new questions."""
    new: list[Question] = []
    budget = budget_left(db, now=now, weekly_budget=weekly_budget)
    if budget:
        ids, keys = _asked(db)
        workflows = [
            q for q in label_questions(db, now=now) if not keys & set(q.workflow_keys)
        ]
        procs = [
            p for p in kb.list_procedures() if p.get("lifecycle_state") != "archived"
        ]
        new = next_questions(procs, budget, workflows=workflows, asked=ids)
        record_asked(db, new, channel=CHANNEL_CARD, now=now)
        if new:
            logger.info(
                "Micro-questions: %d asked as review cards (%s)",
                len(new), ", ".join(q.id for q in new),
            )
    write_cards(db, state_dir)
    return new


def open_cards(db: "WorkerDB") -> list[dict]:
    """The open review cards, oldest first, as the app reads them."""
    return [
        {**json.loads(row["question_json"]), "asked_at": row["asked_at"]}
        for row in db.get_micro_questions(status="open", channel=CHANNEL_CARD)
    ]


def write_cards(db: "WorkerDB", state_dir: Path) -> Path:
    from agenthandover_worker.focus_questioner import write_json_atomic

    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / MICRO_QUESTIONS_FILE
    write_json_atomic(path, {"questions": open_cards(db)})
    return path


def apply_answer(
    trigger: dict,
    db: "WorkerDB",
    kb: "KnowledgeBase",
    state_dir: Path,
    *,
    now: datetime,
) -> bool:
    """A ``micro-answer`` trigger from a review card: ``{"question_id",
    "answer", "name"}`` (a blank answer skips the question).  Returns True
    when an open question was closed."""
    qid = str(trigger.get("question_id") or "")
    row = next((r for r in db.get_micro_questions(status="open") if r["question_id"] == qid), None)
    if row is None:
        logger.warning("micro-answer for %r: no such open question", qid)
        return False
    question = Question.from_json(json.loads(row["question_json"]))
    answer = _text(trigger.get("answer"))
    if not answer:
        db.close_micro_question(qid, status="skipped", answer="", at=now)
        write_cards(db, state_dir)
        return True
    if question.kind == KIND_LABEL:
        _apply_label(db, question, answer, _text(trigger.get("name")), now=now)
    elif not _apply_slot(kb, question, answer):
        return False
    db.close_micro_question(qid, status="answered", answer=answer, at=now)
    write_cards(db, state_dir)
    return True


def _apply_label(
    db: "WorkerDB", question: Question, answer: str, name: str, *, now: datetime,
) -> None:
    is_routine = _ROUTINE.get(answer.lower())
    db.label_workflow(
        question.workflow_keys, label=name, is_routine=is_routine, labelled_at=now,
    )
    logger.info(
        "Workflow %s labelled %r (routine: %s)",
        question.subject[:12], name, "not sure" if is_routine is None else is_routine,
    )
    if is_routine is not False:
        return
    # Not a routine: its draft leaves the review queue.
    slugs = {
        o["promoted_slug"] for o in db.get_workflow_observations(obs_ids=question.obs_ids)
        if o.get("promoted_slug")
    }
    for record in db.get_generated_sops(status="draft"):
        if record.get("slug") in slugs or (
            record.get("source") == "passive"
            and record.get("source_id") in question.workflow_keys
        ):
            db.update_sop_status(record["sop_id"], "rejected")
            logger.info("Rejected draft %s: the user said it is not a routine", record["slug"])


def _apply_slot(kb: "KnowledgeBase", question: Question, answer: str) -> bool:
    """The answer as a typed clarification of the procedure: the ledger
    makes its slot known, with the answer as evidence.  A run question's
    answer fills the entry the run queued; any other is a new entry."""
    proc = kb.get_procedure(question.subject)
    if proc is None:
        logger.warning("micro-answer for %s: no procedure %r", question.id, question.subject)
        return False
    queued = next(
        (c for c in _run_questions(proc) if c["origin"] == question.origin), None,
    ) if question.origin else None
    if queued is not None:
        queued.update(answer=answer, status="answered", source="typed", question_id=question.id)
    else:
        proc.setdefault("agent_clarifications", []).append({
            "category": question.slot,
            "question": question.question,
            "answer": answer,
            "status": "answered",
            "source": "typed",
            "context": question.context,
            "question_id": question.id,
        })
    kb.save_procedure(proc)
    return True


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def digest_items(db: "WorkerDB", date: str) -> tuple[list[dict], list[dict]]:
    """(the open cards, the day's work sessions to tag) for the digest of
    the local *date*.  Every value is text, as the app shows it: each item
    has the ``title`` and ``detail`` the digest window prints."""
    questions = []
    for card in open_cards(db):
        about = card.get("title") or ""
        choices = " / ".join(card.get("choices") or [])
        detail = RUN_CONTEXT.lower() if card.get("origin") else choices
        questions.append({
            "title": card["question"],
            "detail": f"{about}: {detail}" if about and detail else about or detail,
            "question_id": card["id"],
            "choices": choices,
        })
    sessions = []
    for obs in db.get_workflow_observations():
        if obs.get("started_local_date") != date:
            continue
        routine = obs.get("user_is_routine")
        minute = int(obs.get("started_local_minute") or 0)
        session = obs.get("task_label") or ", ".join(json.loads(obs.get("apps_json") or "[]"))
        started = f"{minute // 60:02d}:{minute % 60:02d}"
        minutes = f"{float(obs.get('duration_s') or 0) / 60:.0f}"
        name = obs.get("user_label") or ""
        tagged = (
            "not tagged" if not obs.get("labelled_at")
            else "not sure" if routine is None else ("yes" if routine else "no")
        )
        sessions.append({
            "title": name or session,
            "detail": f"{started}, {minutes} min, routine: {tagged}",
            "session": session, "started": started, "minutes": minutes,
            "name": name, "routine": tagged,
        })
    return questions, sessions

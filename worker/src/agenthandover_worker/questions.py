"""Micro-questions: what AgentHandover asks the user, when, and how often.

Every question comes from what the ledger (:mod:`ledger`) or the
observation log (:mod:`workflow_mining`) does not know yet; no model writes
them.  Three kinds:

- a **step question** per step that step grounding flagged
  (:mod:`step_grounding`, :mod:`clipboard_grounding`) and no answer
  settled yet: a step the recording did not show (dropped), a clause cut
  from a step, a copy or paste the clipboard did not record, and a typed
  line the recording showed that the model had left out or cut short.
  The question quotes what the model wrote and what the recording showed,
  with concrete choices ("Step 3 says 'Paste the copied text', but nothing
  was copied before it in the recording.  What did you do?" [Typed it]
  [Pasted it from somewhere else]).  A choice decides the step in code
  (:func:`apply_step_answer`: restore what the model wrote, or keep the
  grounded step); any other answer is left to the focus answers'
  one revision call (``focus_questioner.answer_revision_prompt``); on a
  review card the same revision runs as a P0 job on the inference thread
  (:func:`revise_step_answer`), and the answer is kept on its step
  whatever the model does.  Either way the step cites the answer
  (``evidence.user_answer``), so the ledger's ``steps`` slot carries an
  ``answer`` pointer next to its frames;
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
and each question once.  A dropped step or a cut clause (the Skill may miss
something the user did) and an unconfirmed copy or paste rank below an
unknown required slot and above every other slot; a typed line the model
had missed ranks above the hypotheses.  A question skipped or left open changes
nothing: its slot stays what it was.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from agenthandover_worker.clipboard_grounding import PASTE
from agenthandover_worker.focus_questioner import USER_ANSWER
from agenthandover_worker.ledger import REQUIRED_SLOTS, derive_knowledge
from agenthandover_worker.skill_improver import RUN_QUESTION
from agenthandover_worker.step_grounding import (
    ADDED,
    INPUT_ADDED,
    REWRITTEN,
    clause_question,
)

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB
    from agenthandover_worker.focus_questioner import FocusQAResult, FocusQuestion
    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.sop_generator import SOPGenerator

logger = logging.getLogger(__name__)

WEEKLY_BUDGET = 5
FOCUS_MAX_QUESTIONS = 3
#: A workflow seen this often gets a labelling card.
MIN_LABEL_OBSERVATIONS = 2
MICRO_QUESTIONS_FILE = "micro-questions.json"

KIND_SLOT = "slot"
KIND_LABEL = "label"
KIND_STEP = "step"
#: The ledger slot a step question informs, and its answers' category.
STEP_SLOT = "steps"
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
    kind: str  # KIND_SLOT, KIND_STEP or KIND_LABEL
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
    # Step questions: the (0-based) steps the question is about when it
    # was asked; a free-text answer may revise these and their neighbours.
    step_indexes: list[int] = field(default_factory=list)

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
# Step questions
# ---------------------------------------------------------------------------

# What step grounding flagged, by what the question is about.
FLAG_DROPPED = "dropped"  # a step no frame showed (evidence.dropped_steps)
FLAG_CUT = "cut"  # a clause cut from a step (evidence.grounding, rewritten)
FLAG_CLIPBOARD = "clipboard"  # a copy/paste rewritten (evidence.clipboard)
FLAG_ADDED = "added"  # a typed line the model left out (evidence.grounding)
FLAG_INPUT = "input"  # a step's input set from the frames (evidence.grounding)

#: Rank tiers for :func:`_rank_value`: every step question sits between
#: an unknown required slot (0) and an unknown optional slot (1).  A
#: concrete step question fixes what the agent will execute; a generic
#: optional slot does not.  Among them: what the Skill may be missing or
#: have wrong first, then an unconfirmed copy or paste, then a typed line
#: or an input the recording showed.
_STEP_TIERS: dict[str, float] = {
    FLAG_DROPPED: 0.5, FLAG_CUT: 0.5, FLAG_CLIPBOARD: 0.6,
    FLAG_ADDED: 0.7, FLAG_INPUT: 0.7,
}

YES_I_DID = "Yes, I did"
NO_I_DID_NOT = "No, I didn't"
TYPED_IT = "Typed it"
PASTED_IT = "Pasted it from somewhere else"
READ_IT = "No, I only read it"
COPIED_IT = "Yes, I copied it"
TYPED_THAT = "Yes, I typed it"
LEAVE_IT_OUT = "No, leave it out"
WHOLE_LINE = "The whole line"
YES = "Yes"
NO = "No"
#: A quoted step text is cut at this many characters.
_QUOTE_CHARS = 120


@dataclass
class _Flag:
    """One thing step grounding flagged on a procedure."""

    kind: str  # FLAG_*
    record: dict  # the flag's record: a dropped entry, or a step's evidence entry
    step: dict | None  # the flagged step (None: a dropped step)
    index: int  # the step's position (a dropped step: where it would go)
    identity: str  # what names the flag across renumbering


def _quote(text: object) -> str:
    text = _text(text)
    return text if len(text) <= _QUOTE_CHARS else text[:_QUOTE_CHARS - 3].rstrip() + "..."


def _steps(proc: dict) -> list[dict]:
    return [s for s in proc.get("steps") or [] if isinstance(s, dict)]


def _frames(item: dict | None) -> list[str]:
    evidence = (item or {}).get("evidence")
    return [str(f) for f in (evidence.get("frames") or [])] if isinstance(evidence, dict) else []


def _input(step: dict) -> str:
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    return str(step.get("input") or params.get("input") or "")


def _dropped(proc: dict) -> list[dict]:
    evidence = proc.get("evidence") if isinstance(proc.get("evidence"), dict) else {}
    dropped = evidence.get("dropped_steps")
    return [d for d in dropped if isinstance(d, dict)] if isinstance(dropped, list) else []


def _dropped_position(steps: list[dict], entry: dict) -> int:
    """Where a dropped step would stand: after the last step citing a frame
    of the step it followed (``after_frames``), at the start when it
    followed none, at the end when that step is gone."""
    after = set(entry.get("after_frames") or [])
    if not after:
        return 0 if "after_frames" in entry else len(steps)
    return max(
        (i + 1 for i, s in enumerate(steps) if after & set(_frames(s))), default=len(steps),
    )


def _flags(proc: dict, *, answered: bool = False) -> list[_Flag]:
    """What step grounding flagged on *proc*'s steps and dropped steps, in
    step order; with *answered*, the flags an answer settled too."""
    steps = _steps(proc)
    out: list[_Flag] = []
    for i, step in enumerate(steps):
        evidence = step.get("evidence") if isinstance(step.get("evidence"), dict) else {}
        frames = "|".join(_frames(step))
        grounding = evidence.get("grounding")
        if isinstance(grounding, dict):
            original = grounding.get("original")
            kind = {REWRITTEN: FLAG_CUT, ADDED: FLAG_ADDED, INPUT_ADDED: FLAG_INPUT}.get(
                grounding.get("status"),
            )
            if kind is not None:
                text = original.get("step") if isinstance(original, dict) else original
                out.append(_Flag(kind, grounding, step, i, f"{frames}\n{text or ''}"))
        clipboard = evidence.get("clipboard")
        if isinstance(clipboard, dict) and clipboard.get("claimed"):
            original = clipboard.get("original") if isinstance(clipboard.get("original"), dict) else {}
            text = json.dumps(original, sort_keys=True)
            out.append(_Flag(FLAG_CLIPBOARD, clipboard, step, i, f"{frames}\n{text}"))
    for entry in _dropped(proc):
        position = _dropped_position(steps, entry)
        identity = "|".join(str(f) for f in entry.get("frames") or []) + f"\n{entry.get('action')}"
        out.append(_Flag(FLAG_DROPPED, entry, None, position, identity))
    out.sort(key=lambda f: f.index)
    return [f for f in out if answered or not isinstance(f.record.get("user_answer"), dict)]


def _flag_id(slug: str, flag: _Flag) -> str:
    digest = hashlib.sha256(f"{flag.kind}\n{flag.identity}".encode()).hexdigest()[:12]
    return f"step:{slug}:{flag.kind}:{digest}"


def _flag_text(flag: _Flag) -> tuple[str, str, list[str]]:
    """(question, context, choices) for *flag*."""
    n = flag.index + 1
    record = flag.record
    step = flag.step or {}
    action = _quote(step.get("action") or step.get("step"))
    if flag.kind == FLAG_DROPPED:
        return (
            f"The model thought you also did \"{_quote(record.get('action'))}\", but the "
            "recording did not show it. Did you?",
            f"Left out of the Skill: {_text(record.get('reason'))}.",
            [NO_I_DID_NOT, YES_I_DID],
        )
    if flag.kind == FLAG_CUT:
        cut = [c for c in record.get("cut") or [] if isinstance(c, dict) and c.get("clause")]
        asks = [clause_question(str(c["clause"])) for c in cut]
        original = _quote((record.get("original") or {}).get("step"))
        if len(asks) == 1:
            shown, ask = asks[0]
        else:
            shown = " or ".join(
                f"'{_quote(c['clause'])}'" for c in cut
            ) or "part of it"
            ask = "Did you do that?"
        return (
            f"The recording did not show {shown} at step {n} (\"{action}\"). {ask}",
            f"The model wrote: \"{original}\".",
            [NO_I_DID_NOT, YES_I_DID],
        )
    if flag.kind == FLAG_CLIPBOARD:
        original = record.get("original") if isinstance(record.get("original"), dict) else {}
        said = _quote(original.get("step") or original.get("action")
                      or next(iter(original.values()), action))
        if PASTE in (record.get("claimed") or []):
            return (
                f"Step {n} says \"{said}\", but nothing was copied before it in the "
                "recording. What did you do?",
                f"The Skill has it as: \"{action}\".",
                [TYPED_IT, PASTED_IT],
            )
        return (
            f"Step {n} says \"{said}\", but the recording shows no copy. Did you copy it?",
            f"The Skill has it as: \"{action}\".",
            [READ_IT, COPIED_IT],
        )
    typed = _quote(_input(step))
    if flag.kind == FLAG_ADDED:
        return (
            f"The recording shows you typed \"{typed}\" (step {n}), which the model had "
            "left out. Did you type it?",
            "Every line the recording shows typed is a step.",
            [TYPED_THAT, LEAVE_IT_OUT],
        )
    partial = _quote(record.get("original"))
    if partial:
        return (
            f"At step {n} the model had you type \"{partial}\", but the recording shows "
            f"the finished line \"{typed}\". Which did you type?",
            "The frames may catch a line half typed.",
            [WHOLE_LINE, f"Only \"{partial}\""],
        )
    return (
        f"The recording shows you typed \"{typed}\" at step {n} (\"{action}\"). "
        "Is that what this step types?",
        "The model had left the text out.",
        [YES, NO],
    )


def step_questions(proc: dict) -> list[Question]:
    """One question per step grounding flagged on *proc* that no answer
    settled yet, ranked by :data:`_STEP_TIERS` and step order."""
    slug = str(proc.get("id") or proc.get("slug") or "")
    if not slug:
        return []
    value = float((proc.get("value") or {}).get("value_score") or 0.0)
    title = _text(proc.get("title")) or slug
    steps = _steps(proc)
    out = []
    for flag in _flags(proc):
        question, context, choices = _flag_text(flag)
        out.append(Question(
            id=_flag_id(slug, flag), kind=KIND_STEP, subject=slug, title=title,
            question=question, context=context, choices=choices, slot=STEP_SLOT,
            value=_rank_value(_STEP_TIERS[flag.kind], flag.index / 100, value),
            step_indexes=_flag_steps(flag, len(steps)),
        ))
    return out


def _flag_steps(flag: _Flag, count: int) -> list[int]:
    """The steps a flag's question is about: its step, or for a dropped
    step the ones either side of where it stood."""
    around = [flag.index] if flag.step is not None else [flag.index - 1, flag.index]
    return [i for i in around if 0 <= i < count]


def choice_index(choices: list[str], answer: str) -> int | None:
    """Which of *choices* *answer* is (spacing and case aside), else None."""
    wanted = _text(answer).lower()
    return next((i for i, c in enumerate(choices) if _text(c).lower() == wanted), None)


def _find(proc: dict, question_id: str, *, answered: bool) -> _Flag | None:
    slug = str(proc.get("id") or proc.get("slug") or "")
    return next(
        (f for f in _flags(proc, answered=answered) if _flag_id(slug, f) == question_id), None,
    )


def step_question_indexes(proc: dict, question_id: str) -> list[int]:
    """The steps step question *question_id* is about in *proc* as it is
    now (answered or not; empty when the flag is gone)."""
    flag = _find(proc, question_id, answered=True)
    return _flag_steps(flag, len(_steps(proc))) if flag is not None else []


def _set_input(step: dict, text: str) -> None:
    step["input"] = text
    if isinstance(step.get("parameters"), dict) and "input" in step["parameters"]:
        step["parameters"]["input"] = text


def _restored_step(entry: dict, origin: dict) -> dict:
    """A dropped step as a procedure step, restored by the user's answer."""
    params = dict(entry.get("parameters") or {})
    return {
        "action": str(entry.get("action") or ""),
        "description": str(entry.get("description") or ""),
        "target": str(params.get("target") or params.get("location") or ""),
        "app": str(params.get("app") or ""),
        "location": str(params.get("location") or ""),
        "input": str(params.get("input") or ""),
        "verify": str(params.get("verify") or ""),
        "selector": None,
        "anchor": None,
        "parameters": params,
        "confidence": 0.0,
        "on_failure": {"strategy": "abort", "max_retries": 0, "delay_seconds": 0},
        "evidence": {
            "frames": list(entry.get("frames") or []),
            "dom_snapshot_ids": [],
            "source": USER_ANSWER,
            "user_answer": origin,
        },
    }


def apply_step_answer(
    proc: dict, question_id: str, answer: str, *, clarification: int, question: str,
) -> bool:
    """Apply the answer to step question *question_id* to *proc* (in
    place); returns True when one of the question's choices decided the
    step, False for any other answer (left to the typed-answer revision)
    or a flag that is no longer on the procedure.

    Either way the flag records the answer (``user_answer``: the
    *clarification* it was recorded as, the *question*, the answer) and so
    does the step's evidence, which gives the ledger's ``steps`` slot an
    ``answer`` pointer.  A choice then does what it says: "Yes, I did"
    restores the dropped step or the cut clauses, "Pasted it" / "Yes, I
    copied it" the model's clipboard wording, "No, leave it out" removes
    an added typed line, "Only ..." / "No" put back the model's input; the
    other choices keep the grounded step as it is.  A restored or removed
    step renumbers the steps, and the branch hypotheses and divergence
    branches stay on the steps they named (``branch_mining.shift_steps``:
    a Skill another run was matched to carries them); any other choice
    leaves the step ids as they are."""
    flag = _find(proc, question_id, answered=False)
    if flag is None:
        logger.info("Step question %s: the step changed since it was asked", question_id)
        return False
    origin = {
        "clarification": clarification, "question": question, "answer": answer,
        "question_id": question_id,
    }
    flag.record["user_answer"] = dict(origin)
    if flag.step is not None:
        flag.step.setdefault("evidence", {})["user_answer"] = dict(origin)
    _question, _context, choices = _flag_text(flag)
    choice = choice_index(choices, answer)
    if choice is None:
        return False
    from agenthandover_worker.branch_mining import shift_steps

    steps = _steps(proc)
    step = flag.step
    restore = choice == 1
    moved = False
    if flag.kind == FLAG_DROPPED and restore:
        proc["evidence"]["dropped_steps"] = [
            d for d in proc["evidence"]["dropped_steps"] if d is not flag.record
        ]
        steps.insert(flag.index, _restored_step(flag.record, origin))
        shift_steps(proc, flag.index, +1, steps)
        moved = True
    elif flag.kind == FLAG_CUT and restore:
        original = flag.record.get("original") or {}
        step["action"] = str(original.get("step") or step.get("action") or "")
        if original.get("description"):
            step["description"] = str(original["description"])
        step["evidence"]["source"] = USER_ANSWER
    elif flag.kind == FLAG_CLIPBOARD and restore:
        for key, text in (flag.record.get("original") or {}).items():
            name = key.split(".", 1)[-1]
            name = "action" if name == "step" else name
            if key.startswith("parameters.") and isinstance(step.get("parameters"), dict):
                step["parameters"][name] = text
            if not key.startswith("parameters.") or name in step:
                step[name] = text
        step["evidence"]["source"] = USER_ANSWER
    elif flag.kind == FLAG_ADDED and restore:
        # Not typed after all: it leaves the steps, and the evidence keeps
        # it with the answer.
        at = next(i for i, s in enumerate(steps) if s is step)
        del steps[at]
        shift_steps(proc, at, -1, steps)
        moved = True
        proc.setdefault("evidence", {}).setdefault("dropped_steps", []).append({
            "action": str(step.get("action") or ""),
            "frames": _frames(step),
            "reason": "the user said they did not type it",
            "user_answer": dict(origin),
        })
    elif flag.kind == FLAG_INPUT and restore:
        _set_input(step, str(flag.record.get("original") or ""))
        step["evidence"]["source"] = USER_ANSWER
    if moved:
        for n, s in enumerate(steps):
            s["index"] = n
            s["step_id"] = f"step_{n + 1}"
        proc["steps"] = steps
    logger.info(
        "Step question %s answered %r: %s", question_id, answer,
        "restored what the model wrote" if restore else "kept the grounded step",
    )
    return True


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
    the questions of *procs*: unknown required slots, then every step
    question (dropped steps and cut clauses, unconfirmed copies and
    pastes, typed lines and inputs the recording showed), then unknown
    optional slots, then hypotheses; slots in :data:`ASKED_SLOTS` order,
    steps in step order, and the more valuable procedure first."""
    candidates = [
        q for q in [
            *workflows,
            *(q for p in procs for q in [*slot_questions(p), *step_questions(p)]),
        ]
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
            step_indexes=list(q.step_indexes),
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
    reviser: "SOPGenerator | None" = None,
) -> bool:
    """A ``micro-answer`` trigger from a review card: ``{"question_id",
    "answer", "name"}`` (a blank answer skips the question).  Returns True
    when an open question was closed.

    A step question answered with one of its choices is applied in code
    (:func:`apply_step_answer`).  Any other answer is recorded on its step,
    and it revises the step through *reviser* (``SOPGenerator.revise``),
    as the focus window's typed answers do: :func:`revise_step_answer`,
    queued as a P0 job on the inference thread (``model_work``), never
    called here, because the trigger runs on the control thread.  With no
    *reviser* (the v2 pipeline is off), the answer is only recorded."""
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
    if (
        question.kind == KIND_STEP and reviser is not None
        and choice_index(question.choices, answer) is None
    ):
        from agenthandover_worker.inference_runner import P0, model_work

        model_work(
            f"step-answer-revision:{qid}",
            lambda: revise_step_answer(kb, question, answer, reviser),
            P0,
        )
    return True


def _step_answer_qa(proc: dict, question: Question, answer: str) -> "FocusQAResult | None":
    """The card's answer as the focus window's typed answer to one step
    question, about its steps as they now stand; None when the flag is no
    longer on the procedure (its step changed since it was asked)."""
    from agenthandover_worker.focus_questioner import (
        ANSWER_TYPED,
        FocusQAResult,
        FocusQuestion,
    )

    indexes = step_question_indexes(proc, question.id)
    if not indexes:
        return None
    return FocusQAResult(
        questions=[FocusQuestion(
            question=question.question, category=question.slot or STEP_SLOT,
            context=question.context, default="", step_indexes=indexes,
            id=question.id, choices=list(question.choices),
        )],
        answers={0: answer},
        sources={0: ANSWER_TYPED},
    )


def revise_step_answer(
    kb: "KnowledgeBase", question: Question, answer: str, reviser: "SOPGenerator",
) -> list[dict]:
    """Revise the Skill's steps with a review card's free-text answer to
    step question *question*: the focus window's one revision call
    (``focus_questioner.answer_revision_prompt``, ``SOPGenerator.revise``)
    and its rules (``apply_answer_revisions``: the question's steps and
    their neighbours, typed text kept, duplicated neighbours dropped).
    Runs on the inference thread.  Returns the applied revisions.

    The answer was recorded before this ran (:func:`_apply_slot`).  When
    the model is unavailable, its answer is unusable, or the steps changed
    while it was answering, the steps stay as they are and the answer stays
    recorded."""
    from agenthandover_worker.focus_questioner import (
        answer_revision_prompt,
        apply_answer_revisions,
    )

    proc = kb.get_procedure(question.subject)
    qa = _step_answer_qa(proc, question, answer) if proc is not None else None
    prompt = answer_revision_prompt(proc, qa) if qa is not None else None
    if prompt is None:
        logger.info("Step question %s: no step left to revise; the answer stays recorded",
                    question.id)
        return []
    try:
        response = reviser.revise(prompt)
    except ConnectionError as exc:
        logger.warning(
            "Step question %s: the answer could not revise the steps (model "
            "unavailable: %s); it stays recorded", question.id, exc,
        )
        return []
    except (ValueError, TimeoutError, RuntimeError) as exc:
        logger.warning(
            "Step question %s: the answer revision was unusable (%s); steps unchanged",
            question.id, exc,
        )
        return []
    # The model call let go of the state lock: the procedure is read again,
    # and a revision numbered on steps that changed meanwhile is not applied.
    fresh = kb.get_procedure(question.subject)
    if fresh is None or fresh.get("steps") != proc.get("steps"):
        logger.warning(
            "Step question %s: the steps changed during the revision; steps "
            "unchanged, the answer stays recorded", question.id,
        )
        return []
    applied = apply_answer_revisions(fresh, qa, response)
    if applied:
        kb.save_procedure(fresh)
        logger.info("Step question %s: the card's answer revised the steps", question.id)
    return applied


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
        clarifications = proc.setdefault("agent_clarifications", [])
        clarifications.append({
            "category": question.slot,
            "question": question.question,
            "answer": answer,
            "status": "answered",
            "source": "typed",
            "context": question.context,
            "question_id": question.id,
        })
        if question.kind == KIND_STEP:
            # A choice decides the step here.  Any other answer is recorded
            # on the step it is about; ``apply_answer`` then queues its
            # revision for the inference thread (``revise_step_answer``).
            apply_step_answer(
                proc, question.id, answer,
                clarification=len(clarifications) - 1, question=question.question,
            )
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

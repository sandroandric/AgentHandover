"""The focus recording's questions: the Q&A files, merging the answers,
and the typed answers' revision of the steps.

The questions themselves are the ledger's (:mod:`questions`): after a
focus recording's SOP is generated, the worker asks about the slots it
does not know yet, writes them to ``focus-questions.json`` in the status
directory (with the choices the runs suggest) and pauses the export; the
app's Q&A window shows them and writes the answers back, and on the next
poll the worker merges them into the procedure (:func:`merge_answers`,
then :func:`answer_revision_prompt` / :func:`apply_answer_revisions` for
the typed answers that change steps).

Each session's pending state waits in its own
``focus-pending/<session_id>.json``; the app sees one session's questions
at a time, oldest first (:func:`ask_focus_questions`).
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenthandover_worker.redaction import redact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FocusQuestion:
    """A single targeted question about a focus recording."""

    question: str
    category: str  # "credentials", "strategy", "decision", "verification", "scope"
    context: str   # why this question matters
    # The model's guess.  The app pre-fills the answer field with it:
    # submitted unchanged it is recorded as an accepted guess (a
    # hypothesis), never as an answer the user typed.
    default: str
    # Step indexes (0-based) the question relates to.  When the user's answer
    # corrects a value (e.g. "the subreddit is r/ClaudeAI, not r/midclaw"),
    # merge_answers() rewrites these specific steps' text.  Empty list means
    # the answer applies to the procedure as a whole, not a particular step.
    step_indexes: list[int] = field(default_factory=list)
    # The micro-question's id (``questions``): its row in ``micro_questions``.
    id: str = ""
    # Answers the runs suggest: the app offers them next to the text field.
    choices: list[str] = field(default_factory=list)


@dataclass
class FocusQAResult:
    """Questions and their answers."""

    questions: list[FocusQuestion]
    # question_index -> answer text.  A question with no entry (or an
    # empty one) was not answered.
    answers: dict[int, str]
    # question_index -> ANSWER_TYPED or ANSWER_ACCEPTED_GUESS, for each
    # entry of ``answers``; an answer without one was typed.
    sources: dict[int, str] = field(default_factory=dict)

    def source(self, idx: int) -> str:
        return self.sources.get(idx, ANSWER_TYPED)

    def counts(self) -> dict[str, int]:
        """How many questions were answered by typing, by accepting the
        model's guess, and not at all."""
        counts = {ANSWER_TYPED: 0, ANSWER_ACCEPTED_GUESS: 0, "unanswered": 0}
        for idx in range(len(self.questions)):
            if str(self.answers.get(idx) or "").strip():
                counts[self.source(idx)] += 1
            else:
                counts["unanswered"] += 1
        return counts

    def status(self) -> str:
        """``answered`` (every question), ``partial`` or ``unanswered``."""
        counts = self.counts()
        if not counts["unanswered"]:
            return "answered"
        return "unanswered" if counts["unanswered"] == len(self.questions) else "partial"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOCUS_QUESTIONS_FILE = "focus-questions.json"

# Where an answer came from (``answer_sources`` in the answered file, and
# ``source`` on the clarification): typed by the user, or the model's
# pre-filled guess submitted unchanged.
ANSWER_TYPED = "typed"
ANSWER_ACCEPTED_GUESS = "accepted_guess"

# Each session waiting for its answers has its pending state here,
# ``<session_id>.json``.  The app sees one session's questions at a time
# in ``focus-questions.json``, and the next queued session's once those
# are processed.
FOCUS_PENDING_DIR = "focus-pending"
# The single pending file of 0.4.x/0.5.0, moved into the queue at startup.
FOCUS_PENDING_FILE = "focus-pending.json"


# ---------------------------------------------------------------------------
# Merging the answers
# ---------------------------------------------------------------------------


def merge_answers(
    procedure: dict,
    qa_result: FocusQAResult,
) -> dict:
    """Merge user answers into the procedure dict.

    Does NOT mutate the input procedure.  Returns a new dict.  The step
    questions of *qa_result* get the ``step_indexes`` of their steps in
    the returned procedure (a choice may have restored or removed a step).

    A step question (``questions.step_questions``) answered with one of
    its choices changes its step in code (``questions.apply_step_answer``);
    answered otherwise, it is left to the revision call.

    Q&A answers always land in ``agent_clarifications[]``.  They do NOT
    auto-create ``environment.accounts[]`` or ``branches[]`` entries from
    free text — earlier behavior corrupted those structured fields with
    narrative answers like "Assume credentials are provided via a logged
    in browser" appearing as ``accounts[0].service`` and "Select the top
    3 trending topics" appearing as ``branches[0].condition``.  Both made
    downstream Skills harder to execute, not easier.

    Where the answer is a concrete value the user is correcting (e.g.
    "the subreddit is r/ClaudeAI") AND the question has ``step_indexes``
    pointing at the affected steps, ``_rewrite_steps_for_clarifications``
    applies a targeted in-place text substitution so the step body stays
    self-consistent with the clarification.

    Two narrow categories still update procedure-level fields directly
    because they are non-corrupting and the field shape matches free text:

    - ``strategy``: only writes if the procedure has no synthesised
      strategy yet (avoid clobbering behavioral_synthesizer output).
    - ``scope`` (recurrence): writes the recurrence string verbatim.
    """
    proc = json.loads(json.dumps(procedure))  # deep copy
    clarifications = proc.setdefault("agent_clarifications", [])

    for idx, q in enumerate(qa_result.questions):
        answer = str(qa_result.answers.get(idx) or "").strip()
        if not answer:
            # Skipped, or the field was cleared: the question stays
            # open.
            clarifications.append({
                "category": q.category,
                "question": q.question,
                "answer": "",
                "status": "unanswered",
                "context": q.context,
                **({"question_id": q.id} if q.id else {}),
            })
            continue

        cat = q.category.lower().replace(" ", "_")

        # Every answer is recorded in agent_clarifications. This is the
        # canonical home for Q&A — UI reads from here, audit reads from
        # here, downstream consumers read from here.
        # An accepted guess is recorded as such: the ledger keeps it a
        # hypothesis, where a typed answer is known.
        source = qa_result.source(idx)
        clarifications.append({
            "category": q.category,
            "question": q.question,
            "answer": answer,
            "status": "answered",
            "source": source,
            "context": q.context,
            **({"question_id": q.id} if q.id else {}),
        })
        if source == ANSWER_ACCEPTED_GUESS:
            # The model's own guess confirms nothing: it fills no
            # procedure field and rewrites no step.
            continue

        if is_step_question(q):
            # A step grounding flagged: a choice decides it in code; any
            # other answer goes to the revision call (``_revisable``).
            from agenthandover_worker.questions import apply_step_answer

            apply_step_answer(
                proc, q.id, answer, clarification=len(clarifications) - 1,
                question=q.question,
            )
            continue

        # Strategy: free text, but only fill if we don't already have one
        # from behavioral synthesis (which is structurally richer).
        if cat in ("strategy", "goal", "purpose", "approach"):
            _merge_strategy(proc, answer)
        # Scope/recurrence: the answer maps directly to a single field.
        elif cat in ("scope", "scheduling", "recurrence", "frequency"):
            _merge_scope(proc, answer)
        # All other categories (credentials, decision, verification, etc.)
        # are recorded in agent_clarifications only.  No structural-field
        # auto-population from free text.

    # After all answers are recorded, sweep the steps for in-place
    # text substitutions when the user's answer corrects a specific
    # value referenced by a step (e.g. "the subreddit is r/ClaudeAI").
    _rewrite_steps_for_clarifications(proc, qa_result)

    # A choice may have restored or removed a step: a step question left to
    # the revision call names its steps as they now stand.
    from agenthandover_worker.questions import step_question_indexes

    for q in qa_result.questions:
        if is_step_question(q):
            q.step_indexes = step_question_indexes(proc, q.id)

    return proc


def is_step_question(question: FocusQuestion) -> bool:
    """A question about a step grounding flagged (``questions.step_questions``)."""
    return question.id.startswith("step:")

def _merge_strategy(proc: dict, answer: str) -> None:
    """Merge strategy answer into procedure.

    Only fills ``strategy`` when behavioral_synthesizer hasn't already
    produced one — Q&A clarifications are coarser than synthesised
    strategy and shouldn't clobber it.
    """
    existing = (proc.get("strategy") or "").strip()
    if not existing:
        proc["strategy"] = answer

def _merge_scope(proc: dict, answer: str) -> None:
    """Merge scope/recurrence answer into procedure."""
    lower = answer.lower()
    if lower in ("one-off", "once", "no", "n/a"):
        proc["recurrence"] = None
    else:
        proc["recurrence"] = answer

def _rewrite_steps_for_clarifications(
    proc: dict,
    qa_result: FocusQAResult,
) -> None:
    """In-place rewrite of step text based on user clarifications.

    When a question carries ``step_indexes`` and the answer is a concrete
    replacement value (e.g. "r/ClaudeAI" correcting an observed
    "r/midclaw"), substitute the answer into the relevant step's
    ``action``, ``target``, ``location``, and ``parameters`` fields.

    Strategy is conservative — substitute only when:
    1. The question explicitly names step indexes
    2. The answer is short enough to plausibly be a value (≤ 80 chars)
    3. The category is one where corrections are expected (decision,
       data, verification, scope) — not for free-text strategy answers

    Falls back to no-op when none of the above hold.  This avoids
    false-positive substitutions on long narrative answers.
    """
    steps = proc.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return

    rewrite_categories = {
        "decision", "decision_logic", "branching", "condition",
        "data", "data_source", "input", "parameter",
        "verification", "success", "done", "outcome", "output_format",
        "scope", "target", "destination",
    }

    for idx, answer in qa_result.answers.items():
        if idx < 0 or idx >= len(qa_result.questions):
            continue
        if qa_result.source(idx) == ANSWER_ACCEPTED_GUESS:
            continue
        q = qa_result.questions[idx]
        ans = str(answer or "").strip()
        if not ans or len(ans) > 80:
            continue
        cat = q.category.lower().replace(" ", "_")
        if cat not in rewrite_categories:
            continue
        if not q.step_indexes:
            continue

        for step_idx in q.step_indexes:
            if not isinstance(step_idx, int) or step_idx < 0 or step_idx >= len(steps):
                continue
            step = steps[step_idx]
            if not isinstance(step, dict):
                continue
            # Replace the most-specific reference field first.
            # If the answer looks like a URL fragment (starts with /,
            # http, or r/) we treat it as a target/location override.
            if ans.startswith(("r/", "/r/", "http://", "https://", "/")):
                if "target" in step:
                    step["target"] = ans
                if isinstance(step.get("parameters"), dict):
                    step["parameters"]["target"] = ans
                    if "location" in step["parameters"]:
                        step["parameters"]["location"] = ans
            # Otherwise leave the answer in agent_clarifications only —
            # we don't have enough information to know what specific
            # field the answer should overwrite.


# ---------------------------------------------------------------------------
# Typed answers revise the steps
# ---------------------------------------------------------------------------

_REVISION_PROMPT = """\
A user recorded this workflow, then answered questions about it. Revise \
the steps so that they do what the user's answers say.

STEPS:
{steps_text}

ANSWERS:
{answers_text}

Return a JSON object:
{{
  "revisions": [
    {{
      "answer": <the answer's number>,
      "replaces": [<the numbers of the steps the answer changes>],
      "steps": [
        {{"action": "<what to do: short, imperative>", "input": "<exact text to type, or empty>", "verify": "<what shows it worked>"}}
      ]
    }}
  ]
}}

Rules:
- Change only what an answer says. An answer that changes no step gets no revision.
- "steps" take the place of the steps in "replaces", in order; there may be more or fewer of them.
- Text the user says to type goes in "input", in the user's words.
- If no answer changes a step, return {{"revisions": []}}

Respond with ONLY the JSON object."""

# Categories whose answer fills a procedure field (``merge_answers``), not
# a step.
_FIELD_CATEGORIES = frozenset({
    "strategy", "goal", "purpose", "approach",
    "scope", "scheduling", "recurrence", "frequency",
})
# A revision replaces at most this many steps more than it names.
_MAX_ADDED_STEPS = 2
USER_ANSWER = "user_answer"


def _revisable(qa_result: FocusQAResult) -> list[int]:
    """The indexes of the typed answers that may change steps: not an
    accepted guess (it confirms nothing), not a field's answer (strategy,
    scope), not a URL the deterministic rewrite already put in its steps,
    not a step question's choice (``merge_answers`` applied it in code)."""
    from agenthandover_worker.questions import choice_index

    indexes = []
    for idx, q in enumerate(qa_result.questions):
        answer = str(qa_result.answers.get(idx) or "").strip()
        if not answer or qa_result.source(idx) != ANSWER_TYPED:
            continue
        if q.category.lower().replace(" ", "_") in _FIELD_CATEGORIES:
            continue
        if is_step_question(q) and choice_index(q.choices, answer) is not None:
            continue
        if q.step_indexes and answer.startswith(("r/", "/r/", "http://", "https://", "/")):
            continue
        indexes.append(idx)
    return indexes


def answer_revision_prompt(proc: dict, qa_result: FocusQAResult) -> str | None:
    """The prompt of the one revision call for a session's typed answers
    (``SOPGenerator.revise``), or None when no typed answer may change a
    step (no call).  Steps and answers are numbered from 1; each answer
    says which steps its question was about."""
    indexes = _revisable(qa_result)
    steps = [s for s in proc.get("steps") or [] if isinstance(s, dict)]
    if not indexes or not steps:
        return None
    step_lines = []
    for n, s in enumerate(steps, 1):
        line = f"{n}. {s.get('action') or s.get('step') or ''}"
        app = s.get("app") or (s.get("parameters") or {}).get("app")
        if app:
            line += f" (in {app})"
        if s.get("input"):
            line += f" [input: {s['input']}]"
        step_lines.append(line)
    answer_lines = []
    for n, idx in enumerate(indexes, 1):
        q = qa_result.questions[idx]
        about = ", ".join(str(i + 1) for i in q.step_indexes if 0 <= i < len(steps))
        answer_lines.append(
            f"{n}. Question{f' (about steps {about})' if about else ''}: {q.question}\n"
            f"   Answer: {str(qa_result.answers[idx]).strip()}"
        )
    return _REVISION_PROMPT.format(
        steps_text=redact("\n".join(step_lines)),
        answers_text=redact("\n".join(answer_lines)),
    )


def apply_answer_revisions(
    proc: dict, qa_result: FocusQAResult, response: dict,
) -> list[dict]:
    """Apply the revision call's *response* to *proc*'s steps (in place);
    returns one record per applied revision.

    A revision is kept only when it answers a typed answer, replaces steps
    that exist (the question's own steps when it named any), does not
    touch a step another revision replaced, and brings 1 to (replaced + 2)
    steps with an action.  Each new step keeps the app, location and
    frames of the steps it replaces, and its evidence says it came from
    the answer: ``source: "user_answer"`` and ``user_answer`` (the
    clarification it answers, the question, the answer and the replaced
    actions).  The steps are renumbered."""
    indexes = _revisable(qa_result)
    steps = [s for s in proc.get("steps") or [] if isinstance(s, dict)]
    revisions = response.get("revisions") if isinstance(response, dict) else None
    if not isinstance(revisions, list) or not steps:
        return []
    clarifications = proc.get("agent_clarifications") or []
    taken: set[int] = set()
    planned: list[tuple[list[int], list[dict], int]] = []
    for revision in revisions:
        if not isinstance(revision, dict):
            continue
        try:
            number = int(revision.get("answer"))
            if number < 1:
                continue
            idx = indexes[number - 1]
            replaced = sorted({int(n) - 1 for n in revision.get("replaces") or []})
        except (TypeError, ValueError, IndexError):
            continue
        new_steps = [
            s for s in revision.get("steps") or []
            if isinstance(s, dict) and str(s.get("action") or "").strip()
        ]
        # The question's own steps and their immediate neighbours (an answer
        # about steps 4-7 may rightly also rewrite step 8: seen live, where
        # the answer named the line after the ones asked about), else any.
        named = set(qa_result.questions[idx].step_indexes)
        allowed = (
            named | {min(named) - 1, max(named) + 1} if named else set(range(len(steps)))
        )
        if (
            not replaced or not set(replaced) <= allowed & set(range(len(steps)))
            or taken & set(replaced)
            or not 1 <= len(new_steps) <= len(replaced) + _MAX_ADDED_STEPS
        ):
            logger.info("Focus Q&A: revision for answer %d not applied: %r", idx, revision)
            continue
        taken.update(replaced)
        planned.append((replaced, new_steps, idx))

    applied: list[dict] = []
    # From the last position back, so earlier positions stay valid.
    for replaced, new_steps, idx in sorted(planned, key=lambda p: -p[0][0]):
        q = qa_result.questions[idx]
        answer = str(qa_result.answers[idx]).strip()
        old = [steps[i] for i in replaced]
        clarification = max(
            (i for i, c in enumerate(clarifications)
             if isinstance(c, dict) and c.get("question") == q.question),
            default=None,
        )
        origin = {
            "clarification": clarification, "question": q.question, "answer": answer,
            "replaced": [str(s.get("action") or "") for s in old],
        }
        base = old[0]
        built = []
        for new in new_steps:
            action = str(new["action"]).strip()
            built.append({
                **{k: base.get(k) for k in ("app", "location", "target", "on_failure")
                   if k in base},
                "action": action,
                "description": f"As the user answered: {answer}",
                "input": str(new.get("input") or "").strip(),
                "verify": str(new.get("verify") or "").strip() or str(base.get("verify") or ""),
                "anchor": None,
                "selector": None,
                "parameters": {},
                "confidence": base.get("confidence", 0.0),
                "evidence": {
                    "frames": list(dict.fromkeys(
                        f for s in old for f in (s.get("evidence") or {}).get("frames") or []
                    )),
                    "dom_snapshot_ids": list(dict.fromkeys(
                        d for s in old
                        for d in (s.get("evidence") or {}).get("dom_snapshot_ids") or []
                    )),
                    "source": USER_ANSWER,
                    "user_answer": origin,
                },
            })
        _keep_typed_text(built, old, proc)
        steps[replaced[0]:replaced[-1] + 1] = [
            *built, *(s for i, s in enumerate(steps[replaced[0]:replaced[-1] + 1],
                                              replaced[0]) if i not in replaced),
        ]
        _drop_duplicated_neighbours(steps, replaced[0], len(built), proc)
        applied.append({**origin, "steps": [s["action"] for s in built]})
    for n, step in enumerate(steps):
        step["index"] = n
        step["step_id"] = f"step_{n + 1}"
    proc["steps"] = steps
    if applied:
        logger.info(
            "Focus Q&A: %d typed answer(s) revised the steps: %s", len(applied),
            "; ".join(f"{a['replaced']} -> {a['steps']}" for a in applied),
        )
    return list(reversed(applied))


def _typed(step: dict, proc: dict) -> str:
    """The step's input with each ``{{name}}`` read as its example value,
    normalised (lowercase words, one space apart)."""
    examples = {
        str(v.get("name")): str(v.get("example") or "")
        for v in proc.get("variables") or [] if isinstance(v, dict) and v.get("name")
    }
    text = re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: examples.get(m.group(1), m.group(0)),
                  str(step.get("input") or ""))
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _keep_typed_text(built: list[dict], old: list[dict], proc: dict) -> None:
    """A revised step may not lose typed text its replaced steps carried:
    an input that is only the start of an old step's input gets the old
    one (seen live: the revision wrote "Reserved for documentation:
    example.com" where step grounding had put the whole typed line)."""
    for new in built:
        text = _typed(new, proc)
        if not text:
            continue
        longer = max(
            (o for o in old if len(_typed(o, proc)) > len(text) and text in _typed(o, proc)),
            key=lambda o: len(_typed(o, proc)), default=None,
        )
        if longer is not None:
            new["input"] = str(longer.get("input") or "")


def _drop_duplicated_neighbours(steps: list[dict], start: int, count: int, proc: dict) -> None:
    """Remove an untouched step right before or after the revised ones
    that types what a revised step now types (seen live: the answer's new
    Source step, and the old one kept next to it)."""
    revised = {_typed(s, proc) for s in steps[start:start + count]} - {""}
    for i in (start + count, start - 1):
        if 0 <= i < len(steps) and not start <= i < start + count and _typed(steps[i], proc) in revised:
            del steps[i]


# ---------------------------------------------------------------------------
# File IPC helpers — used by main.py to save/load pending Q&A state
# ---------------------------------------------------------------------------


def _questions_payload(questions: list[FocusQuestion]) -> list[dict]:
    """The questions as ``focus-questions.json`` lists them."""
    return [
        {
            "index": i,
            "id": q.id,
            "question": q.question,
            "category": q.category,
            "context": q.context,
            "default": q.default,
            "step_indexes": list(q.step_indexes),
            "choices": list(q.choices),
        }
        for i, q in enumerate(questions)
    ]


def _write_questions_file(
    state_dir: Path, session_id: str, slug: str, title: str, questions: list[dict],
) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FOCUS_QUESTIONS_FILE
    write_json_atomic(path, {
        "session_id": session_id,
        "slug": slug,
        "title": title,
        "questions": questions,
        "status": "pending",
    })
    logger.info(
        "Wrote %d focus question(s) to %s (session=%s, slug=%s)",
        len(questions), path, session_id, slug,
    )
    return path


def write_focus_questions(
    state_dir: Path,
    session_id: str,
    slug: str,
    questions: list[FocusQuestion],
    *,
    title: str = "",
) -> Path:
    """Write focus questions to the status directory for the app.

    *title* is the session's title as the user typed it: the app shows it
    as is (the slug loses its casing, "IANA" became "Iana").  Returns the
    path to the written file.  The worker asks through
    :func:`ask_focus_questions`, which never replaces another session's
    questions.
    """
    return _write_questions_file(
        state_dir, session_id, slug, title, _questions_payload(questions),
    )


def _pending_path(state_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)
    return state_dir / FOCUS_PENDING_DIR / f"{safe}.json"


def _pending_entries(state_dir: Path) -> list[dict]:
    """The queued sessions, oldest first.  An unreadable entry, or one
    without a session id, is logged and skipped."""
    queue_dir = state_dir / FOCUS_PENDING_DIR
    if not queue_dir.is_dir():
        return []
    entries: list[dict] = []
    for path in sorted(queue_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("session_id"), str) \
                or not data["session_id"]:
            logger.warning("Skipping unreadable queued focus Q&A entry %s", path)
            continue
        entries.append(data)
    entries.sort(key=lambda e: e["seq"] if isinstance(e.get("seq"), int) else 0)
    return entries


def write_focus_pending(
    state_dir: Path,
    session_id: str,
    slug: str,
    sop_template: dict,
    procedure: dict,
    *,
    title: str = "",
    questions: list[dict] | None = None,
) -> Path:
    """Queue a session's pending SOP state, with the questions the app is
    to show for it, so the export resumes after its Q&A.

    Each session has its own entry, ``focus-pending/<session_id>.json``:
    a second session never replaces the first's state.  The queue is
    oldest first (``seq``); a session queued again keeps its place.
    Returns the path to the written file.
    """
    path = _pending_path(state_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _pending_entries(state_dir)
    seq = next((e.get("seq") for e in entries if e["session_id"] == session_id), None)
    if not isinstance(seq, int):
        seq = 1 + max(
            (e["seq"] for e in entries if isinstance(e.get("seq"), int)), default=0,
        )
    write_json_atomic(path, {
        "session_id": session_id,
        "slug": slug,
        "title": title,
        "seq": seq,
        "sop_template": sop_template,
        "procedure": procedure,
        "questions": list(questions or []),
    })
    logger.debug("Saved focus pending state to %s", path)
    return path


def read_focus_questions(state_dir: Path) -> dict | None:
    """Read focus questions file.  Returns None if missing or unreadable."""
    path = state_dir / FOCUS_QUESTIONS_FILE
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read %s", path, exc_info=True)
        return None


def read_focus_pending(state_dir: Path, session_id: str | None = None) -> dict | None:
    """The pending state queued for *session_id* (without one: the oldest
    queued session's).  Returns None if there is none or it is unreadable."""
    if session_id is None:
        entries = _pending_entries(state_dir)
        return entries[0] if entries else None
    if not session_id:
        return None
    path = _pending_path(state_dir, session_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not read queued focus Q&A entry %s", path)
        return None
    if not isinstance(data, dict) or data.get("session_id") != session_id:
        logger.warning("Queued focus Q&A entry %s is not session %s", path, session_id)
        return None
    return data


def clear_focus_qa_files(state_dir: Path, session_id: str) -> None:
    """Remove the questions file and *session_id*'s queued state once its
    answers are processed.  The other queued sessions stay."""
    paths = [state_dir / FOCUS_QUESTIONS_FILE]
    if session_id:
        paths.append(_pending_path(state_dir, session_id))
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to remove %s", path, exc_info=True)


def publish_next_focus_questions(state_dir: Path) -> str | None:
    """Show the oldest queued session's questions when the app has none.

    ``focus-questions.json`` is written only when it is absent (or
    unreadable): another session's pending or answered file is never
    replaced.  An entry without questions cannot be shown and is passed
    over.  Returns the session id now shown, or None when nothing was
    written.
    """
    path = state_dir / FOCUS_QUESTIONS_FILE
    if path.exists():
        if isinstance(read_focus_questions(state_dir), dict):
            return None
        logger.warning("%s is unreadable; showing the next queued questions instead", path)
    for entry in _pending_entries(state_dir):
        questions = entry.get("questions")
        if not isinstance(questions, list) or not questions:
            continue
        _write_questions_file(
            state_dir, entry["session_id"], str(entry.get("slug") or ""),
            str(entry.get("title") or ""), questions,
        )
        return entry["session_id"]
    return None


def ask_focus_questions(
    state_dir: Path,
    session_id: str,
    slug: str,
    questions: list[FocusQuestion],
    *,
    title: str,
    sop_template: dict,
    procedure: dict,
) -> bool:
    """Queue a session's questions with its pending state, and show them
    unless another session's questions are still in the app (then they
    are shown once those are processed).  Returns True when the app now
    shows this session's questions."""
    payload = _questions_payload(questions)
    write_focus_pending(
        state_dir, session_id, slug, sop_template, procedure,
        title=title, questions=payload,
    )
    shown = read_focus_questions(state_dir)
    if isinstance(shown, dict) and shown.get("session_id") == session_id \
            and shown.get("status") == "pending":
        # The same session asked again: its unanswered questions are
        # replaced by the new ones.
        _write_questions_file(state_dir, session_id, slug, title, payload)
        return True
    return publish_next_focus_questions(state_dir) == session_id


def restore_focus_queue(state_dir: Path) -> str | None:
    """At startup: move a 0.4.x/0.5.0 ``focus-pending.json`` into the
    queue under its session id (idempotent: an entry already queued is
    kept), then show the oldest queued questions if the app has none.
    Returns the session id newly shown, or None."""
    legacy = state_dir / FOCUS_PENDING_FILE
    if legacy.is_file():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        session_id = data.get("session_id") if isinstance(data, dict) else None
        if not isinstance(session_id, str) or not session_id:
            logger.warning("Leaving unreadable %s in place (no session id)", legacy)
        else:
            if read_focus_pending(state_dir, session_id) is None:
                shown = read_focus_questions(state_dir)
                if not (isinstance(shown, dict) and shown.get("session_id") == session_id):
                    logger.warning(
                        "%s (session %s) has no questions file: queued without questions",
                        legacy, session_id,
                    )
                    shown = {}
                write_focus_pending(
                    state_dir, session_id, str(data.get("slug") or ""),
                    data.get("sop_template") or {}, data.get("procedure") or {},
                    title=str(shown.get("title") or ""),
                    questions=list(shown.get("questions") or []),
                )
            legacy.unlink(missing_ok=True)
            logger.info("Moved %s into the focus Q&A queue (session %s)", legacy, session_id)
    return publish_next_focus_questions(state_dir)


def parse_qa_result_from_file(
    questions_data: dict,
) -> FocusQAResult | None:
    """Parse a focus-questions.json with status=answered into a FocusQAResult.

    The app writes ``answers`` (question index -> text; a cleared field is
    left out) and ``answer_sources`` (index -> ``typed`` or
    ``accepted_guess``).  A file without sources (an older app) had only
    typed answers.  Returns None if the data is invalid or not in answered
    state.
    """
    if questions_data.get("status") not in ("answered", "skipped"):
        return None

    raw_questions = questions_data.get("questions", [])
    questions: list[FocusQuestion] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            continue
        questions.append(FocusQuestion(
            question=raw.get("question", ""),
            category=raw.get("category", "strategy"),
            context=raw.get("context", ""),
            default=raw.get("default", ""),
            step_indexes=[i for i in raw.get("step_indexes") or [] if isinstance(i, int)],
            id=str(raw.get("id") or ""),
            choices=[str(c) for c in raw.get("choices") or []],
        ))

    answers: dict[int, str] = {}
    sources: dict[int, str] = {}
    if questions_data.get("status") == "answered":
        raw_answers = questions_data.get("answers") or {}
        raw_sources = questions_data.get("answer_sources") or {}
        for key, value in raw_answers.items():
            try:
                idx = int(key)
            except (ValueError, TypeError):
                continue
            answers[idx] = str(value)
            if raw_sources.get(key) == ANSWER_ACCEPTED_GUESS:
                sources[idx] = ANSWER_ACCEPTED_GUESS
    # "skipped" leaves ``answers`` empty: merge_answers records every
    # question as unanswered.

    return FocusQAResult(questions=questions, answers=answers, sources=sources)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write JSON (tmp + fsync + rename)."""
    import os
    import tempfile

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".focus-q.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

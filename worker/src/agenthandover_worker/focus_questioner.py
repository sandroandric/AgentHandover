"""Focus session questioner — gap analysis and targeted questions.

After a focus recording generates a SOP, this module analyzes the
procedure + behavioral insights for gaps and generates 1-3 targeted
questions.  User answers are merged back into the procedure before
final export.

Question categories:
- ``credentials``: workflow requires login but no auth step recorded
- ``strategy``: the overall goal/approach is unclear
- ``decision``: branch conditions detected but logic unclear
- ``verification``: no success criteria defined
- ``scope``: timing/recurrence patterns suggest scheduling

The question delivery uses JSON file IPC: the worker writes
``focus-questions.json`` to the status directory; the CLI reads it,
presents questions to the user, writes answers back, and the worker
merges answers on the next poll cycle.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agenthandover_worker.clipboard_grounding import COPIED, COPY, PASTE, claims, rewrite
from agenthandover_worker.redaction import redact

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner

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
FOCUS_PENDING_FILE = "focus-pending.json"

_LEGACY_MERGE_CATEGORIES = frozenset({
    "credentials", "strategy", "decision", "verification", "scope",
})

_MAX_QUESTIONS = 3

_NO_CLIPBOARD_NOTE = (
    "\n\nCLIPBOARD: no copy, cut or paste was recorded during this session: "
    "the user typed the text. Do not ask about copying or pasting."
)


# ---------------------------------------------------------------------------
# LLM prompt for question generation
# ---------------------------------------------------------------------------

_QUESTIONER_SYSTEM = """\
You are an AI agent that has been given a workflow to execute autonomously. \
Before you can execute it, you need to identify what's missing or unclear. \
Think practically: what would YOU need to know to do this task reliably \
on your own, without the human watching? \
Respond with ONLY valid JSON. No markdown fences, no commentary."""


_QUESTIONER_PROMPT = """\
You are an AI agent about to execute this workflow autonomously. \
A human just recorded themselves doing it. Here is what was captured:

TITLE: {title}
DESCRIPTION: {description}
STRATEGY: {strategy}

STEPS:
{steps_text}

APPS INVOLVED: {apps}
URLS DETECTED: {urls}
INPUTS/VARIABLES: {variables}{clipboard_note}

Put yourself in the agent's position. Think step by step:
- Could you actually execute each step with the information given?
- Do you know WHERE to do things (which app, which URL, which page)?
- Do you know HOW to make decisions the human made implicitly?
- Do you know WHEN you're done and whether you succeeded?
- Do you have ACCESS to everything you'd need (logins, permissions, data)?

Identify 1-3 questions that would be MOST practically useful for you \
as the executing agent. Do not ask generic questions. Ask specific, \
practical questions grounded in what you see in the steps above.

Return a JSON object:
{{
  "questions": [
    {{
      "question": "<specific, practical question you need answered>",
      "category": "<short label: e.g. access, data_source, decision_logic, output_format, error_handling, verification, scheduling, permissions, or any other relevant label>",
      "context": "<why you need this to execute reliably>",
      "default": "<your best guess if the human doesn't answer>",
      "step_numbers": [<list of 1-based step numbers this question is about>]
    }}
  ]
}}

Rules:
- Ask 1-3 questions MAXIMUM. Only what truly blocks reliable execution.
- If you could execute this workflow as-is, return {{"questions": []}}
- Every question must reference specific steps or details from above.
- Defaults should be your most conservative practical assumption.
- step_numbers is REQUIRED when the question is about a specific step or
  steps in the list above. Use [] when the question is general (e.g. about
  scheduling or overall strategy) and not tied to a single step.

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# FocusQuestioner
# ---------------------------------------------------------------------------


class FocusQuestioner:
    """Analyze a generated procedure for gaps and produce targeted questions.

    Uses ``LLMReasoner.reason_json()`` for question generation and
    deterministic logic for answer merging.
    """

    def __init__(self, llm_reasoner: "LLMReasoner") -> None:
        self._reasoner = llm_reasoner

    def generate_questions(
        self,
        procedure: dict,
        sop_template: dict | None = None,
    ) -> list[FocusQuestion]:
        """Analyze procedure for gaps and generate 1-3 targeted questions.

        Args:
            procedure: The v3 procedure dict from focus processing.
            sop_template: Optional SOP template with additional context.

        Returns:
            List of 0-3 FocusQuestion objects.  Empty if no gaps found
            or if the LLM call fails (graceful degradation).
        """
        # Build the prompt from procedure data
        title = procedure.get("title", "Untitled")
        description = procedure.get("description", "")
        strategy = procedure.get("strategy", "")

        steps = procedure.get("steps", [])
        steps_lines = []
        for s in steps:
            action = s.get("action", s.get("step", ""))
            app = s.get("app", s.get("parameters", {}).get("app", ""))
            line = f"  - {action}"
            if app:
                line += f" (in {app})"
            steps_lines.append(line)
        steps_text = "\n".join(steps_lines) if steps_lines else "(no steps)"

        apps = ", ".join(procedure.get("apps_involved", [])) or "(none)"

        # Extract URLs from steps and SOP template
        urls_set: set[str] = set()
        for s in steps:
            for key in ("location", "target", "url"):
                val = s.get(key, "") or s.get("parameters", {}).get(key, "")
                if val and ("http://" in val or "https://" in val):
                    urls_set.add(val)
        if sop_template:
            for s in sop_template.get("steps", []):
                for key in ("location", "target", "url"):
                    val = s.get(key, "") or s.get("parameters", {}).get(key, "")
                    if val and ("http://" in val or "https://" in val):
                        urls_set.add(val)
        urls = ", ".join(sorted(urls_set)) or "(none)"

        # Extract variables / inputs
        variables_list = procedure.get("variables", procedure.get("inputs", []))
        if isinstance(variables_list, list):
            variables = ", ".join(
                v.get("name", str(v)) if isinstance(v, dict) else str(v)
                for v in variables_list
            ) or "(none)"
        else:
            variables = "(none)"

        # What the clipboard recorded in the session (None: no record).  A
        # session that copied nothing is told so, as the SOP prompt is, and
        # the synthesized text's copy wording is not handed on as a fact.
        # The steps are the Skill's, already grounded (clipboard_grounding,
        # step_grounding).
        recorded = _session_clipboard(sop_template)
        note = ""
        if recorded == set():
            note = _NO_CLIPBOARD_NOTE
            description, strategy = rewrite(description or ""), rewrite(strategy or "")

        # The procedure's text quotes the screen: redacted before the
        # model sees it.
        prompt = _QUESTIONER_PROMPT.format(
            title=redact(title),
            description=redact(description),
            strategy=redact(strategy) or "(not yet determined)",
            steps_text=redact(steps_text),
            apps=redact(apps),
            urls=redact(urls),
            variables=redact(variables),
            clipboard_note=note,
        )

        result = self._reasoner.reason_json(
            prompt=prompt,
            system=_QUESTIONER_SYSTEM,
            caller="focus_questioner",
        )

        if not result.success or result.abstained or not isinstance(result.value, dict):
            logger.debug(
                "Focus questioner LLM call failed or abstained: %s",
                result.error or "abstained",
            )
            return []

        return drop_unrecorded_clipboard_questions(
            self._parse_questions(result.value), recorded,
        )

    @staticmethod
    def _parse_questions(data: dict) -> list[FocusQuestion]:
        """Parse LLM response into validated FocusQuestion objects."""
        raw_questions = data.get("questions", [])
        if not isinstance(raw_questions, list):
            return []

        questions: list[FocusQuestion] = []
        for raw in raw_questions:
            if not isinstance(raw, dict):
                continue

            question_text = raw.get("question", "").strip()
            category = raw.get("category", "").strip().lower()
            context = raw.get("context", "").strip()
            default = raw.get("default", "").strip()

            if not question_text:
                continue
            if not category:
                category = "general"

            # step_indexes lets the LLM tag a question to specific steps so
            # merge_answers can rewrite those steps in place.  Accept both
            # 1-based "step_numbers" (LLM-friendly) and 0-based indexes;
            # normalise to 0-based here.
            raw_step_idx = raw.get("step_indexes") or raw.get("step_numbers") or []
            step_indexes: list[int] = []
            if isinstance(raw_step_idx, list):
                for v in raw_step_idx:
                    try:
                        i = int(v)
                    except (TypeError, ValueError):
                        continue
                    # If the LLM gave 1-based numbers (>= 1), shift to 0-based.
                    # 0 stays as 0 (already 0-based).  We can't perfectly tell
                    # the difference; assume LLM uses 1-based when any value
                    # >= 1 appears.
                    step_indexes.append(i - 1 if i >= 1 else i)

            questions.append(FocusQuestion(
                question=question_text,
                category=category,
                context=context,
                default=default,
                step_indexes=step_indexes,
            ))

            if len(questions) >= _MAX_QUESTIONS:
                break

        return questions

    def merge_answers(
        self,
        procedure: dict,
        qa_result: FocusQAResult,
    ) -> dict:
        """Merge user answers into the procedure dict.

        Does NOT mutate the input.  Returns a new dict.

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
            })
            if source == ANSWER_ACCEPTED_GUESS:
                # The model's own guess confirms nothing: it fills no
                # procedure field and rewrites no step.
                continue

            # Strategy: free text, but only fill if we don't already have one
            # from behavioral synthesis (which is structurally richer).
            if cat in ("strategy", "goal", "purpose", "approach"):
                self._merge_strategy(proc, answer)
            # Scope/recurrence: the answer maps directly to a single field.
            elif cat in ("scope", "scheduling", "recurrence", "frequency"):
                self._merge_scope(proc, answer)
            # All other categories (credentials, decision, verification, etc.)
            # are recorded in agent_clarifications only.  No structural-field
            # auto-population from free text.

        # After all answers are recorded, sweep the steps for in-place
        # text substitutions when the user's answer corrects a specific
        # value referenced by a step (e.g. "the subreddit is r/ClaudeAI").
        self._rewrite_steps_for_clarifications(proc, qa_result)

        return proc

    @staticmethod
    def _merge_strategy(proc: dict, answer: str) -> None:
        """Merge strategy answer into procedure.

        Only fills ``strategy`` when behavioral_synthesizer hasn't already
        produced one — Q&A clarifications are coarser than synthesised
        strategy and shouldn't clobber it.
        """
        existing = (proc.get("strategy") or "").strip()
        if not existing:
            proc["strategy"] = answer

    @staticmethod
    def _merge_scope(proc: dict, answer: str) -> None:
        """Merge scope/recurrence answer into procedure."""
        lower = answer.lower()
        if lower in ("one-off", "once", "no", "n/a"):
            proc["recurrence"] = None
        else:
            proc["recurrence"] = answer

    @staticmethod
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


def _session_clipboard(sop_template: dict | None) -> set[str] | None:
    """The clipboard kinds the session recorded (``copy``, ``paste``), from
    the SOP template's ``session_clipboard``; None without a record."""
    kinds = (sop_template or {}).get("session_clipboard")
    return None if kinds is None else {str(k) for k in kinds}


def drop_unrecorded_clipboard_questions(
    questions: list[FocusQuestion], recorded: set[str] | None,
) -> list[FocusQuestion]:
    """*questions* without those that assume a copy or paste the clipboard
    did not record (*recorded*: the session's clipboard kinds; None keeps
    every question).  The live run asked "Should I copy the exact text
    ..." of a session that only typed.  A copy needs a recorded copy, a
    paste a recorded paste or copy (the rule of ``clipboard_grounding``)."""
    if recorded is None:
        return questions
    supported = {COPY: COPY in recorded, COPIED: COPY in recorded,
                 PASTE: PASTE in recorded or COPY in recorded}
    kept = []
    for q in questions:
        claimed = claims([q.question, q.context, q.default])
        if any(not supported[c] for c in claimed):
            logger.info(
                "Focus questioner: dropped a question that assumes an unrecorded "
                "copy or paste: %r", q.question,
            )
            continue
        kept.append(q)
    return kept


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
    scope), not a URL the deterministic rewrite already put in its steps."""
    indexes = []
    for idx, q in enumerate(qa_result.questions):
        answer = str(qa_result.answers.get(idx) or "").strip()
        if not answer or qa_result.source(idx) != ANSWER_TYPED:
            continue
        if q.category.lower().replace(" ", "_") in _FIELD_CATEGORIES:
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
    path to the written file.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FOCUS_QUESTIONS_FILE

    payload = {
        "session_id": session_id,
        "slug": slug,
        "title": title,
        "questions": [
            {
                "index": i,
                "question": q.question if hasattr(q, "question") else q.get("question", ""),
                "category": q.category if hasattr(q, "category") else q.get("category", ""),
                "context": q.context if hasattr(q, "context") else q.get("context", ""),
                "default": q.default if hasattr(q, "default") else q.get("default", ""),
                "step_indexes": list(
                    q.step_indexes if hasattr(q, "step_indexes") else q.get("step_indexes") or []
                ),
            }
            for i, q in enumerate(questions)
        ],
        "status": "pending",
    }

    _atomic_write_json(path, payload)
    logger.info(
        "Wrote %d focus question(s) to %s (session=%s, slug=%s)",
        len(questions), path, session_id, slug,
    )
    return path


def write_focus_pending(
    state_dir: Path,
    session_id: str,
    slug: str,
    sop_template: dict,
    procedure: dict,
) -> Path:
    """Save the pending SOP state so it can be resumed after Q&A.

    Returns the path to the written file.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FOCUS_PENDING_FILE

    payload = {
        "session_id": session_id,
        "slug": slug,
        "sop_template": sop_template,
        "procedure": procedure,
    }

    _atomic_write_json(path, payload)
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


def read_focus_pending(state_dir: Path) -> dict | None:
    """Read focus pending state file.  Returns None if missing."""
    path = state_dir / FOCUS_PENDING_FILE
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read %s", path, exc_info=True)
        return None


def clear_focus_qa_files(state_dir: Path) -> None:
    """Remove both focus Q&A files after export completes."""
    for filename in (FOCUS_QUESTIONS_FILE, FOCUS_PENDING_FILE):
        path = state_dir / filename
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to remove %s", path, exc_info=True)


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


def _atomic_write_json(path: Path, data: Any) -> None:
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

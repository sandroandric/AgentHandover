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
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
    default: str   # suggested default if user skips
    # Step indexes (0-based) the question relates to.  When the user's answer
    # corrects a value (e.g. "the subreddit is r/ClaudeAI, not r/midclaw"),
    # merge_answers() rewrites these specific steps' text.  Empty list means
    # the answer applies to the procedure as a whole, not a particular step.
    step_indexes: list[int] = field(default_factory=list)


@dataclass
class FocusQAResult:
    """Questions and their answers."""

    questions: list[FocusQuestion]
    answers: dict[int, str]  # question_index -> answer text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOCUS_QUESTIONS_FILE = "focus-questions.json"
FOCUS_PENDING_FILE = "focus-pending.json"

_LEGACY_MERGE_CATEGORIES = frozenset({
    "credentials", "strategy", "decision", "verification", "scope",
})

_MAX_QUESTIONS = 3


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
INPUTS/VARIABLES: {variables}

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

        prompt = _QUESTIONER_PROMPT.format(
            title=title,
            description=description,
            strategy=strategy or "(not yet determined)",
            steps_text=steps_text,
            apps=apps,
            urls=urls,
            variables=variables,
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

        return self._parse_questions(result.value)

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
            if not default:
                default = "Not specified"

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

        for idx, answer in qa_result.answers.items():
            if idx < 0 or idx >= len(qa_result.questions):
                continue

            q = qa_result.questions[idx]
            answer = answer.strip()
            if not answer:
                answer = q.default

            cat = q.category.lower().replace(" ", "_")

            # Every answer is recorded in agent_clarifications. This is the
            # canonical home for Q&A — UI reads from here, audit reads from
            # here, downstream consumers read from here.
            clarifications = proc.setdefault("agent_clarifications", [])
            clarifications.append({
                "category": q.category,
                "question": q.question,
                "answer": answer,
                "context": q.context,
            })

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
            q = qa_result.questions[idx]
            ans = answer.strip() or q.default
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
# File IPC helpers — used by main.py to save/load pending Q&A state
# ---------------------------------------------------------------------------


def write_focus_questions(
    state_dir: Path,
    session_id: str,
    slug: str,
    questions: list[FocusQuestion],
) -> Path:
    """Write focus questions to the status directory for CLI pickup.

    Returns the path to the written file.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FOCUS_QUESTIONS_FILE

    payload = {
        "session_id": session_id,
        "slug": slug,
        "questions": [
            {
                "index": i,
                "question": q.question if hasattr(q, "question") else q.get("question", ""),
                "category": q.category if hasattr(q, "category") else q.get("category", ""),
                "context": q.context if hasattr(q, "context") else q.get("context", ""),
                "default": q.default if hasattr(q, "default") else q.get("default", ""),
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

    Returns None if the data is invalid or not in answered state.
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
            default=raw.get("default", "Not specified"),
        ))

    answers: dict[int, str] = {}
    if questions_data.get("status") == "answered":
        raw_answers = questions_data.get("answers", {})
        for key, value in raw_answers.items():
            try:
                answers[int(key)] = str(value)
            except (ValueError, TypeError):
                continue
    elif questions_data.get("status") == "skipped":
        # Use defaults for all questions
        for i, q in enumerate(questions):
            answers[i] = q.default

    return FocusQAResult(questions=questions, answers=answers)


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

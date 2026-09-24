"""The learning ledger: what AgentHandover knows about a procedure, slot by slot.

A Skill is only as good as what was actually learned.  The ledger makes
that explicit: for each *slot* of a procedure it records whether the value
is **known** (grounded in evidence), a **hypothesis** (a value inferred
from what was seen, not yet confirmed), or **unknown** (nothing learned,
so an agent must ask and never assume).

Stored on the procedure as ``proc["knowledge"]``::

    {
      "<slot>": {
        "status":     "known" | "hypothesis" | "unknown",
        "value_ref":  "<dotted path into the procedure>" | None,
        "evidence":   [{"kind": "frame" | "instance" | "answer" | "run",
                        "id": "<event id | session or observation id"
                              " | clarification path | run id>"}],
        "source":     "observation" | "contrast" | "divergence" | "user"
                      | "agent" | None,
        "updated_at": "<ISO-8601 UTC>",
      },
      ...
    }

Slots (:data:`SLOTS`): the nine tracked slots ``goal``, ``done_criteria``,
``trigger``, ``inputs``, ``steps``, ``decisions``, ``exceptions``,
``guardrails``, ``verification``, plus the optional ``voice`` and
``context``.  :data:`REQUIRED_SLOTS` are the ones an agent cannot act
without (what to produce, how, and when it is done); :func:`readiness`
fails while any of them is ``unknown``.  The other slots are reported as
"Not yet known" in the packet so the agent asks instead of guessing.

Who writes what:

- ``source="observation"`` entries are *derived* from the procedure's own
  fields by :func:`derive_knowledge` every time the procedure is saved
  (``KnowledgeBase.save_procedure``).  They are recomputed, never edited.
- ``source="user"`` entries are derived from answered
  ``agent_clarifications`` (the post-recording questions): an answer the
  user typed makes its slot ``known`` and wins over everything else; the
  model's guess submitted unchanged (``source: "accepted_guess"``) makes
  it a ``hypothesis`` unless the slot is already known or agent runs
  confirmed it.
- ``trigger``, ``done_criteria`` and ``verification`` are also derived
  from what the runs' own events showed: :func:`refresh_knowledge` first
  recomputes ``proc["learned"]`` (``run_signals.learn``: a recurring
  lead-in or start time, the commit that ends the runs, the checks after
  it and the fields the user corrected), and each learned item becomes a
  ``hypothesis`` (``source="observation"``) whose evidence is the
  ``instance`` pointers of the runs it was seen in.
- ``contrast`` / ``divergence`` / ``agent`` entries are written by the
  learners that produce them (chosen-versus-visible contrast, divergence
  analysis, agent runs: ``skill_improver.feed_ledger``) through
  :func:`set_slot`.  The derivation keeps them as they are until their
  writer changes them.

Rules the derivation follows (see :func:`_observed`):

- ``known`` needs at least one evidence pointer; a value without any
  evidence is at most a ``hypothesis``.
- Recorded steps are ``known`` when they cite the frames they were
  written from; a typed answer that revised steps is part of the
  evidence (``answer`` pointers next to the frames); every other observed value is inferred by a model from
  the recording(s) and stays a ``hypothesis`` until a user answer (or a
  later learner) grounds it.  ``inputs`` becomes ``known`` once two or
  more instances were observed (the values were seen to vary), and
  ``context`` (the apps used) is ``known`` because the apps are read from
  the frames, not inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agenthandover_worker.run_signals import learn

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: What each slot holds, in the order the app and the packet show them.
SLOTS: dict[str, str] = {
    "goal": "what the user is producing",
    "done_criteria": "how to tell the task is finished",
    "trigger": "what starts the task: a time, an event or a message",
    "inputs": "the values that change from run to run",
    "steps": "the actions, in order",
    "decisions": "how the user chooses between options",
    "exceptions": "what goes wrong and what the user does then",
    "guardrails": "what the user never does",
    "verification": "how the user checks the result",
    "voice": "how the user writes",
    "context": "the apps and accounts the task uses",
}
OPTIONAL_SLOTS: tuple[str, ...] = ("voice", "context")
TRACKED_SLOTS: tuple[str, ...] = tuple(s for s in SLOTS if s not in OPTIONAL_SLOTS)
#: A procedure with any of these ``unknown`` is not ready for an agent.
REQUIRED_SLOTS: tuple[str, ...] = ("goal", "steps", "done_criteria")

STATUSES: tuple[str, ...] = ("known", "hypothesis", "unknown")
SOURCES: tuple[str, ...] = ("observation", "contrast", "divergence", "user", "agent")
EVIDENCE_KINDS: tuple[str, ...] = ("frame", "instance", "answer", "run")

# Which slot a question's category informs: a micro-question's category is
# its slot (``questions``); the free labels of the questions of older
# versions map as below, and unmapped ones inform no slot.
_ANSWER_SLOTS: dict[str, str] = {
    **{slot: slot for slot in SLOTS},
    **dict.fromkeys(("goal", "purpose", "strategy", "approach"), "goal"),
    **dict.fromkeys(
        ("success", "done", "outcome", "output_format", "success_criteria"),
        "done_criteria",
    ),
    **dict.fromkeys(
        ("scope", "scheduling", "recurrence", "frequency", "trigger", "timing"),
        "trigger",
    ),
    **dict.fromkeys(
        ("data", "data_source", "input", "inputs", "parameter", "credentials",
         "access", "permissions"),
        "inputs",
    ),
    **dict.fromkeys(
        ("decision", "decision_logic", "branching", "condition", "selection"),
        "decisions",
    ),
    **dict.fromkeys(("error_handling", "exception", "exceptions", "errors"), "exceptions"),
    **dict.fromkeys(("guardrail", "guardrails", "constraint", "safety"), "guardrails"),
    **dict.fromkeys(("verification", "verify", "check"), "verification"),
    **dict.fromkeys(("voice", "tone", "style"), "voice"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry(status: str, value_ref: str | None, evidence: list[dict], source: str | None) -> dict:
    return {
        "status": status,
        "value_ref": value_ref,
        "evidence": evidence,
        "source": source,
    }


def _unknown() -> dict:
    return _entry("unknown", None, [], None)


# ---------------------------------------------------------------------------
# Derivation from the procedure's own fields
# ---------------------------------------------------------------------------


def _dedup(items: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in items:
        key = (item["kind"], item["id"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _instances(proc: dict) -> list[dict]:
    """One ``instance`` pointer per observed session of this procedure: a
    focus recording's ``session_id``, or a passively mined work session's
    ``obs_id`` (its row in ``workflow_observations``)."""
    observations = (proc.get("evidence") or {}).get("observations") or []
    return _dedup([
        {"kind": "instance", "id": str(obs.get("session_id") or obs["obs_id"])}
        for obs in observations
        if isinstance(obs, dict) and (obs.get("session_id") or obs.get("obs_id"))
    ])


def _step_frames(proc: dict) -> list[dict]:
    """The frames the steps cite and the answers about steps: a step's
    evidence ``user_answer`` (an answer that revised or confirmed it) and a
    dropped step's (``evidence.dropped_steps``: the user said they did not
    do it) point at their clarification."""
    frames: list[dict] = []
    answers: list[dict] = []
    evidence = proc.get("evidence") if isinstance(proc.get("evidence"), dict) else {}
    dropped = evidence.get("dropped_steps") if isinstance(evidence.get("dropped_steps"), list) else []
    for step in proc.get("steps") or []:
        ev = step.get("evidence") if isinstance(step, dict) else None
        for frame in (ev.get("frames") or []) if isinstance(ev, dict) else []:
            frames.append({"kind": "frame", "id": str(frame)})
    for ev in [
        *(s.get("evidence") for s in proc.get("steps") or [] if isinstance(s, dict)),
        *dropped,
    ]:
        origin = ev.get("user_answer") if isinstance(ev, dict) else None
        if isinstance(origin, dict) and isinstance(origin.get("clarification"), int):
            answers.append(
                {"kind": "answer", "id": f"agent_clarifications.{origin['clarification']}"}
            )
    return _dedup(frames + answers)


def _first_filled(proc: dict, paths: tuple[str, ...]) -> str | None:
    """The first dotted path whose value is non-empty."""
    for path in paths:
        value: object = proc
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, str):
            value = value.strip()
        if value:
            return path
    return None


def _observed(proc: dict, slot: str) -> dict:
    """The ``observation`` entry for *slot*, from the procedure's fields."""
    instances = _instances(proc)

    def hypothesis(ref: str | None) -> dict:
        return _entry("hypothesis", ref, instances, "observation") if ref else _unknown()

    def known(ref: str, evidence: list[dict]) -> dict:
        if not evidence:  # known needs evidence; without it, a hypothesis
            return hypothesis(ref)
        return _entry("known", ref, evidence, "observation")

    if slot == "steps":
        if not any(isinstance(s, dict) for s in proc.get("steps") or []):
            return _unknown()
        return known("steps", instances + _step_frames(proc))
    if slot == "goal":
        # The synthesiser writes "Unclear — ..." when it could not tell.
        goal = proc.get("goal")
        if isinstance(goal, str) and goal.strip().lower().startswith("unclear"):
            return _unknown()
        return hypothesis(_first_filled(proc, ("goal", "task_description")))
    # What the runs' own events showed (``run_signals.learn``): read, not
    # inferred, but seen in some runs only, so a hypothesis until the user
    # confirms it.  It rests on the runs it was seen in.
    learned = proc.get("learned") if isinstance(proc.get("learned"), dict) else {}

    def from_runs(name: str, ref: str, frames: list[str]) -> dict | None:
        item = learned.get(name)
        if not isinstance(item, dict) or not item.get("instances"):
            return None
        evidence = [{"kind": "instance", "id": str(i)} for i in item["instances"]]
        evidence += [{"kind": "frame", "id": str(f)} for f in frames if f]
        return _entry("hypothesis", ref, _dedup(evidence), "observation")

    if slot == "done_criteria":
        done = learned.get("done") if isinstance(learned.get("done"), dict) else {}
        return from_runs("done", "learned.done", done.get("event_ids") or []) or hypothesis(
            _first_filled(proc, (
                "outcome", "expected_outcomes", "execution_overview.success_criteria",
                "postconditions", "outputs",
            ))
        )
    if slot == "trigger":
        trigger = learned.get("trigger") if isinstance(learned.get("trigger"), dict) else {}
        # A calendar event seen before the runs is the trigger's value.
        ref = "learned.trigger.calendar_event" if trigger.get("calendar_event") else "learned.trigger"
        return from_runs("trigger", ref, []) or hypothesis(
            _first_filled(proc, ("recurrence.pattern",))
        )
    if slot == "inputs":
        ref = _first_filled(proc, ("inputs",))
        if ref and len(instances) >= 2:
            return known(ref, instances)
        return hypothesis(ref)
    if slot == "decisions":
        return hypothesis(_first_filled(proc, ("selection_criteria", "branches")))
    if slot == "exceptions":
        return hypothesis(_first_filled(proc, (
            "exceptions_seen", "execution_overview.common_errors",
        )))
    if slot == "guardrails":
        return hypothesis(_first_filled(proc, ("constraints.guardrails",)))
    if slot == "verification":
        checks = learned.get("verification") if isinstance(learned.get("verification"), dict) else {}
        corrected = [
            f for c in checks.get("corrections") or [] if isinstance(c, dict)
            for f in c.get("event_ids") or []
        ]
        entry = from_runs("verification", "learned.verification", corrected)
        if entry is not None:
            return entry
        outcomes = proc.get("expected_outcomes") or []
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, dict) and outcome.get("verification"):
                return hypothesis(f"expected_outcomes.{i}.verification")
        return _unknown()
    if slot == "voice":
        return hypothesis(_first_filled(proc, ("voice_profile", "content_templates")))
    if slot == "context":
        ref = _first_filled(proc, ("environment.required_apps", "apps_involved"))
        return known(ref, instances) if ref else _unknown()
    raise KeyError(slot)


def _answered(proc: dict) -> dict[str, dict]:
    """``user`` entries from answered post-recording questions, by slot.

    An answer the user typed makes its slot ``known``.  The model's guess
    submitted unchanged (``source: "accepted_guess"``) is a ``hypothesis``:
    the user let it stand but said nothing of their own.  A typed answer
    for a slot wins over accepted guesses for it.
    """
    typed: dict[str, list[dict]] = {}
    guessed: dict[str, list[dict]] = {}
    clarifications = proc.get("agent_clarifications")
    if not isinstance(clarifications, list):
        return {}
    for i, item in enumerate(clarifications):
        if not isinstance(item, dict) or item.get("status") == "unanswered":
            continue
        if not str(item.get("answer") or "").strip():
            continue
        category = str(item.get("category") or "").strip().lower().replace(" ", "_")
        slot = _ANSWER_SLOTS.get(category)
        # An answer about a step is cited by that step (``_step_frames``):
        # it joins the steps' frames instead of replacing them.
        if slot is not None and slot != "steps":
            by_slot = guessed if item.get("source") == "accepted_guess" else typed
            by_slot.setdefault(slot, []).append(
                {"kind": "answer", "id": f"agent_clarifications.{i}"}
            )
    entries = {
        slot: _entry("hypothesis", evidence[0]["id"], evidence, "user")
        for slot, evidence in guessed.items()
    }
    entries.update({
        slot: _entry("known", evidence[0]["id"], evidence, "user")
        for slot, evidence in typed.items()
    })
    return entries


def derive_knowledge(proc: dict) -> dict[str, dict]:
    """The ledger for *proc*: derived entries merged with the stored ones.

    Priority per slot: a typed answer (an accepted guess only over a slot
    not already known), then a stored entry written by a
    learner other than the observation derivation (``contrast``,
    ``divergence``, ``agent``, or a ``user`` entry set directly), then the
    observation derivation.  An entry equal to the stored one keeps its
    ``updated_at``.  Pure: does not modify *proc*.
    """
    stored = proc.get("knowledge") if isinstance(proc.get("knowledge"), dict) else {}
    answered = _answered(proc)
    now = _now()
    ledger: dict[str, dict] = {}
    for slot in SLOTS:
        previous = stored.get(slot) if isinstance(stored.get(slot), dict) else None
        if previous and previous.get("source") not in (None, "observation") and not (
            previous.get("source") == "user"
            and str(previous.get("value_ref") or "").startswith("agent_clarifications.")
        ):
            entry = {k: previous.get(k) for k in ("status", "value_ref", "evidence", "source")}
        else:
            entry = _observed(proc, slot)
        # A typed answer wins; an accepted guess (a hypothesis) never
        # replaces what is already known, nor a hypothesis agent runs
        # have confirmed (that entry already cites the guess).
        answer = answered.get(slot)
        if answer is not None and (
            answer["status"] == "known" or entry["status"] == "unknown"
            or (entry["status"] == "hypothesis" and entry["source"] != "agent")
        ):
            entry = answer
        same = previous is not None and all(
            previous.get(k) == entry[k] for k in ("status", "value_ref", "evidence", "source")
        )
        entry["updated_at"] = previous.get("updated_at") if same and previous.get("updated_at") else now
        ledger[slot] = entry
    return ledger


def refresh_knowledge(proc: dict) -> dict[str, dict]:
    """Recompute what the runs taught (``proc["learned"]``,
    :func:`run_signals.learn`) and ``proc["knowledge"]`` in place, and
    return the ledger."""
    learn(proc)
    proc["knowledge"] = derive_knowledge(proc)
    return proc["knowledge"]


def set_slot(
    proc: dict,
    slot: str,
    *,
    status: str,
    source: str,
    evidence: list[dict],
    value_ref: str | None = None,
) -> dict:
    """Record what a learner found for *slot* (validated), in place.

    The entry survives later derivations until a user's answer or the same
    learner changes it.  Returns the entry.
    """
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r} (one of {list(SLOTS)})")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if source not in SOURCES or source == "observation":
        raise ValueError("observation entries are derived; set_slot takes "
                         f"one of {[s for s in SOURCES if s != 'observation']}")
    for item in evidence:
        if item.get("kind") not in EVIDENCE_KINDS or not item.get("id"):
            raise ValueError(f"evidence must be {{kind in {EVIDENCE_KINDS}, id}}: {item!r}")
    if status == "known" and not evidence:
        raise ValueError("a known slot needs at least one evidence pointer")
    ledger = proc.get("knowledge") if isinstance(proc.get("knowledge"), dict) else {}
    entry = _entry(status, value_ref, [dict(e) for e in evidence], source)
    entry["updated_at"] = _now()
    ledger[slot] = entry
    proc["knowledge"] = ledger
    return entry


def validate_knowledge(knowledge: object) -> list[str]:
    """Schema errors in a stored ledger (empty when valid)."""
    if not isinstance(knowledge, dict):
        return ["Field 'knowledge' must be a dict"]
    errors: list[str] = []
    for slot, entry in knowledge.items():
        if slot not in SLOTS:
            errors.append(f"knowledge.{slot}: unknown slot")
            continue
        if not isinstance(entry, dict):
            errors.append(f"knowledge.{slot} must be a dict")
            continue
        if entry.get("status") not in STATUSES:
            errors.append(f"knowledge.{slot}.status must be one of {STATUSES}")
        if entry.get("source") is not None and entry["source"] not in SOURCES:
            errors.append(f"knowledge.{slot}.source must be one of {SOURCES}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(e, dict) and e.get("kind") in EVIDENCE_KINDS and e.get("id")
            for e in evidence
        ):
            errors.append(f"knowledge.{slot}.evidence must be a list of {{kind, id}}")
        elif entry.get("status") == "known" and not evidence:
            errors.append(f"knowledge.{slot}: known without evidence")
    return errors


# ---------------------------------------------------------------------------
# Readiness, confidence, and the one hand-off gate
# ---------------------------------------------------------------------------


def unknown_slots(proc: dict, slots: tuple[str, ...] = TRACKED_SLOTS) -> list[str]:
    """The *slots* of *proc* whose status is ``unknown``, in slot order."""
    ledger = derive_knowledge(proc)
    return [s for s in slots if ledger[s]["status"] == "unknown"]


@dataclass
class Readiness:
    """Whether the ledger holds enough for an agent to act."""

    ready: bool
    unknown_required: list[str] = field(default_factory=list)
    confidence: float = 0.0


def readiness(proc: dict) -> Readiness:
    """Ready when no :data:`REQUIRED_SLOTS` slot is ``unknown``."""
    ledger = derive_knowledge(proc)
    missing = [s for s in REQUIRED_SLOTS if ledger[s]["status"] == "unknown"]
    return Readiness(ready=not missing, unknown_required=missing, confidence=confidence(ledger))


def confidence(ledger: dict[str, dict]) -> float:
    """The grounded fraction: the share of :data:`TRACKED_SLOTS` that are ``known``."""
    known = sum(1 for s in TRACKED_SLOTS if (ledger.get(s) or {}).get("status") == "known")
    return round(known / len(TRACKED_SLOTS), 2)


@dataclass
class Gate:
    """The hand-off decision for one procedure, with every failing reason."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


def handoff_gate(proc: dict, *, now: datetime | None = None) -> Gate:
    """The one check every agent-facing surface applies before handing a
    procedure over (the MCP server's ``list_ready_skills``, the Claude
    Code / Codex / OpenClaw connectors, the pipeline's exports and the
    bundle compiler).  All must hold:

    1. lifecycle is ``agent_ready`` (a human approved it);
    2. :func:`readiness`: no required ledger slot is ``unknown``;
    3. ``constraints.trust_level`` is above ``observe``;
    4. ``procedure_freshness(proc)`` is at least
       :data:`staleness_detector.MIN_FRESHNESS`.
    """
    from agenthandover_worker.constraint_manager import TrustLevel
    from agenthandover_worker.staleness_detector import MIN_FRESHNESS, procedure_freshness

    reasons: list[str] = []
    state = proc.get("lifecycle_state") or "observed"
    if state != "agent_ready":
        reasons.append(f"lifecycle is '{state}', not 'agent_ready' (not approved for agents)")
    ready = readiness(proc)
    if not ready.ready:
        reasons.append("not yet known: " + ", ".join(ready.unknown_required))
    trust = str((proc.get("constraints") or {}).get("trust_level") or "observe")
    if TrustLevel.from_string(trust) is TrustLevel.OBSERVE:
        reasons.append(f"trust level is '{trust}' (must be above 'observe')")
    freshness = procedure_freshness(proc, now=now)
    if freshness < MIN_FRESHNESS:
        reasons.append(f"freshness {freshness:.2f} is below {MIN_FRESHNESS}")
    return Gate(ok=not reasons, reasons=reasons)

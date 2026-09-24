"""Divergence branches: where a new instance of a Skill leaves its steps.

Every observation attached to a Skill (``workflow_mining.attach_observations``)
is aligned to the Skill's steps.  Both sides are compared by *place*, the
``(app, host_or_doc)`` the workflow key is built from: an instance's frames
collapse into the places it visited, a procedure step's place is its
``app`` and ``location`` (steps with neither are left out).  The aligner is
``VariantDetector.semantic_align``, the same one workflow grouping uses.

The instance *diverges* where it

- visits places the Skill never visits (another page of a step's own site,
  another story or record, is the step, not a divergence), or
- never visits the place of a canonical step (a skip).

Returning to a place the Skill does visit elsewhere is not a divergence.
One detour is one divergence: the aligned columns that diverge one after
the other, with the steps they replace, and a return to the detour's
places right after a step outside it continues the detour
(:func:`find_divergences`).

Each divergence is stored as a *branch hypothesis* on the procedure::

    proc["branch_hypotheses"] = [{
        "kind": "detour" | "skip",   # a skip does nothing instead
        "step_index": <first canonical step the divergence replaces or
                       precedes; len(steps) when it comes after the last one>,
        "rejoin_index": <the canonical step the instance is back on;
                         step_index when nothing is replaced>,
        "condition_state": {"ocr_snippet", "url", "visible_items"},
        "alternative_steps": [{"action", "app", "location"}],  # [] = skipped
        "evidence": [{"kind": "instance", "id": <obs_id>},
                     {"kind": "frame", "id": <event id>}, ...],
        "observed_at": <ISO-8601 UTC>,
    }]

``condition_state`` is the screen the user decided on: the frame before
the divergence (its salient OCR lines, URL and list entries).  Two
hypotheses from different instances *agree* when they replace the same
steps with the same places and their screens share an entry or an OCR line;
the second agreeing instance promotes them to a branch in
``proc["branches"]`` (``source: "divergence"``, every agreeing hypothesis's
evidence), whose ``condition`` names what the screens shared.  Later
agreeing instances add their evidence to the branch.  A procedure keeps
its :data:`MAX_HYPOTHESES` most recent hypotheses.  No model is called.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agenthandover_worker.dom import host_or_doc, site_of
from agenthandover_worker.scene_annotator import as_compact
from agenthandover_worker.sop_dedup import _normalize_app
from agenthandover_worker.task_segmenter import AnnotatedFrame
from agenthandover_worker.variant_detector import VariantDetector

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB
    from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_ALIGNER = VariantDetector()
# What a condition state keeps of its frame.
OCR_SNIPPET_CHARS = 300
MAX_VISIBLE_ITEMS = 10
# What a promoted branch's condition quotes.
MAX_CONDITION_TERMS = 3
# A detour the user steps out of for at most this many aligned steps and
# returns to is one detour (``find_divergences``).
DETOUR_RETURN_COLUMNS = 2
# Hypotheses a procedure keeps (the oldest go first; a promoted branch
# keeps its own evidence).
MAX_HYPOTHESES = 50


def _place(app: str, location: str) -> tuple[str, str]:
    return (_normalize_app(app or ""), host_or_doc(location or ""))


def _as_step(place: tuple[str, str]) -> dict:
    # Compared on app and place, as workflow_mining.align_score does.
    return {"action": place[1], "app": place[0], "location": place[1]}


def condition_state(frame: AnnotatedFrame) -> dict:
    """The screen of *frame*: its salient OCR lines, URL and list entries."""
    ann = as_compact(frame.annotation)
    items = [
        str(i.get("label") or "").strip()
        for i in ann.get("visible_items") or [] if isinstance(i, dict)
    ]
    location = frame.location or ""
    return {
        "ocr_snippet": str(ann.get("_screen_text") or "")[:OCR_SNIPPET_CHARS],
        "url": location if location.startswith(("http://", "https://")) else "",
        "visible_items": [i for i in items if i][:MAX_VISIBLE_ITEMS],
    }


def _site(place: tuple[str, str]) -> tuple[str, str]:
    return (place[0], site_of(place[1]))


def find_divergences(proc: dict, frames: list[AnnotatedFrame]) -> list[dict]:
    """The divergences of the instance *frames* (oldest first) from
    *proc*'s steps, as ``{step_index, rejoin_index, condition_frame,
    alternative}``: the instance leaves the Skill before (or instead of)
    canonical step ``step_index`` and is back on it at ``rejoin_index``
    (the steps in between are replaced; equal indexes mean the new places
    come before the step), and *alternative* lists the frames of what it
    did instead (empty for a skip).

    The alignment's columns are read in order.  A column *agrees* when
    the instance performs the canonical step (the same place, or another
    page of the same site: another story, another record), returns to a
    place it already performed a step on or that the Skill visits
    elsewhere, or leaves out a step whose place it visits elsewhere.  Every
    other column diverges, and a run of contiguous diverging columns is
    one divergence (one detour).  A detour that only goes back to the
    places of the detour just before it, after at most
    :data:`DETOUR_RETURN_COLUMNS` agreeing columns, continues that detour:
    the two and the columns between them are one divergence (the user
    stepped out of the detour, to write a note, and returned to it)."""
    steps = proc.get("steps") or []
    canon: list[tuple[int, tuple[str, str]]] = [
        (i, _place(str(s.get("app") or ""), str(s.get("location") or "")))
        for i, s in enumerate(steps)
        if isinstance(s, dict) and (s.get("app") or s.get("location"))
    ]
    if not canon or not frames:
        return []
    # The instance's places, run-length collapsed; each keeps its first frame.
    runs: list[tuple[tuple[str, str], int]] = []
    for idx, frame in enumerate(frames):
        place = _place(frame.app, frame.location)
        if not runs or runs[-1][0] != place:
            runs.append((place, idx))
    visited = {p for p, _ in runs}
    aligned = _ALIGNER.semantic_align(
        [_as_step(p) for _, p in canon], [_as_step(p) for p, _ in runs],
    )
    # Places the instance performed a canonical step on.
    performed = {
        runs[pos.index_b][0] for pos in aligned
        if pos.index_a is not None and pos.index_b is not None
        and _site(runs[pos.index_b][0]) == _site(canon[pos.index_a][1])
    }
    known = {p for _, p in canon} | performed

    # Each column as (canonical step or None, instance run or None,
    # diverges); a column pairing a step with a place of another site is
    # a skip of the step and a visit to the place, and its run is left out
    # when only the skip diverges.  A step left out here whose place the
    # instance visits elsewhere neither agrees nor diverges (None).
    columns: list[tuple[int | None, int | None, bool | None]] = []
    for pos in aligned:
        a, b = pos.index_a, pos.index_b
        step = canon[a][0] if a is not None else None
        if b is None:
            columns.append((step, None, canon[a][1] not in visited or None))
            continue
        if a is not None and _site(runs[b][0]) == _site(canon[a][1]):
            columns.append((step, b, False))
            continue
        skipped = a is not None and canon[a][1] not in visited
        new_place = runs[b][0] not in known
        columns.append((step, b if new_place or not skipped else None, skipped or new_place))

    # Runs of diverging columns, contiguous but for undecided ones, as
    # [first, last] column.
    spans: list[list[int]] = []
    for c, (_, _, diverges) in enumerate(columns):
        if not diverges:
            continue
        if spans and all(d is None for _, _, d in columns[spans[-1][1] + 1:c]):
            spans[-1][1] = c
        else:
            spans.append([c, c])

    def places(span: list[int]) -> set[tuple[str, str]]:
        return {runs[b][0] for _, b, _ in columns[span[0]:span[1] + 1] if b is not None}

    merged: list[list[int]] = []
    for span in spans:
        if (merged and places(span) and places(span) <= places(merged[-1])
                and span[0] - merged[-1][1] - 1 <= DETOUR_RETURN_COLUMNS):
            merged[-1][1] = span[1]  # back to the detour just before
        else:
            merged.append(span)

    divergences: list[dict] = []
    for first, last in merged:
        inside = columns[first:last + 1]
        alternative = [b for _, b, _ in inside if b is not None]
        replaced = [step for step, _, _ in inside if step is not None]
        after = next((step for step, _, _ in columns[last + 1:] if step is not None), len(steps))
        if alternative:
            condition_at = max(runs[alternative[0]][1] - 1, 0)
        else:
            before = [b for _, b, _ in columns[:first] if b is not None]
            condition_at = runs[before[-1]][1] if before else 0
        divergences.append({
            "step_index": replaced[0] if replaced else after,
            "rejoin_index": replaced[-1] + 1 if replaced else after,
            "condition_frame": frames[condition_at],
            "alternative": [frames[runs[b][1]] for b in alternative],
        })
    return divergences


def _hypothesis(divergence: dict, obs_id: str, now: str) -> dict:
    alt = divergence["alternative"]
    evidence = [{"kind": "instance", "id": obs_id}]
    frame_ids = [divergence["condition_frame"].event_id] + [f.event_id for f in alt]
    evidence += [{"kind": "frame", "id": eid} for eid in dict.fromkeys(frame_ids) if eid]
    return {
        "kind": "detour" if alt else "skip",
        "step_index": divergence["step_index"],
        "rejoin_index": divergence["rejoin_index"],
        "condition_state": condition_state(divergence["condition_frame"]),
        "alternative_steps": [
            {"action": f.what_doing, "app": f.app, "location": f.location} for f in alt
        ],
        "evidence": evidence,
        "observed_at": now,
    }


def _key(hyp: dict) -> tuple:
    """Hypotheses with equal keys leave the same steps for the same places."""
    return (hyp["step_index"], hyp.get("rejoin_index"), tuple(
        _place(str(s.get("app") or ""), str(s.get("location") or ""))
        for s in hyp.get("alternative_steps") or []
    ))


def _instance(hyp: dict) -> str:
    return next((e["id"] for e in hyp.get("evidence") or [] if e.get("kind") == "instance"), "")


def _lines(state: dict) -> set[str]:
    return {ln.strip() for ln in str(state.get("ocr_snippet") or "").splitlines() if ln.strip()}


def shared_state(states: list[dict]) -> dict:
    """What every one of *states* shows: list entries, OCR lines, URL."""
    items = set.intersection(*(set(s.get("visible_items") or []) for s in states))
    lines = set.intersection(*(_lines(s) for s in states))
    urls = {s.get("url") or "" for s in states}
    first = states[0]
    return {
        "visible_items": [i for i in first.get("visible_items") or [] if i in items],
        "ocr_lines": [ln.strip() for ln in str(first.get("ocr_snippet") or "").splitlines()
                      if ln.strip() in lines],
        "url": urls.pop() if len(urls) == 1 else "",
    }


def _branch(hyps: list[dict], steps: list) -> dict | None:
    """The branch the agreeing *hyps* support, or None when their screens
    share nothing to name the condition by."""
    shared = shared_state([h["condition_state"] for h in hyps])
    terms = list(dict.fromkeys(shared["visible_items"] + shared["ocr_lines"]))[:MAX_CONDITION_TERMS]
    if not terms:
        return None
    condition = "the screen shows " + ", ".join(f'"{t}"' for t in terms)
    if shared["url"]:
        condition += f" ({shared['url']})"
    k, rejoin = hyps[0]["step_index"], hyps[0]["rejoin_index"]
    alt = hyps[0]["alternative_steps"]
    replaced = f"step {k + 1}" if rejoin == k + 1 else f"steps {k + 1}-{rejoin}"
    if not alt:
        skipped = [
            str(s.get("action") or "") for s in steps[k:rejoin] if isinstance(s, dict)
        ]
        action = f"skip {replaced}" + (f" ({'; '.join(a for a in skipped if a)})" if any(skipped) else "")
    else:
        if rejoin > k:
            where = f"instead of {replaced}"
        else:
            where = f"before step {k + 1}" if k < len(steps) else "after the last step"
        action = f"{where}: " + "; then ".join(
            s.get("action") or f"go to {s.get('location') or s.get('app')}" for s in alt
        )
    evidence: list[dict] = []
    for h in hyps:
        evidence += [e for e in h["evidence"] if e not in evidence]
    return {
        "kind": "detour" if alt else "skip",
        "step_index": k,
        "rejoin_index": rejoin,
        "condition": condition,
        "action": action,
        "alternative_steps": alt,
        "condition_state": {
            "ocr_snippet": "\n".join(shared["ocr_lines"]),
            "url": shared["url"],
            "visible_items": shared["visible_items"],
        },
        "source": "divergence",
        "evidence": evidence,
    }


def record_divergences(proc: dict, obs_id: str, frames: list[AnnotatedFrame]) -> bool:
    """Add the instance's divergences from *proc*'s steps as hypotheses
    and promote the ones two instances agree on.  An instance already
    recorded adds nothing.  Returns whether *proc* changed."""
    hyps: list[dict] = proc.setdefault("branch_hypotheses", [])
    if any(_instance(h) == obs_id for h in hyps):
        return False
    now = datetime.now(timezone.utc).isoformat()
    new = [_hypothesis(d, obs_id, now) for d in find_divergences(proc, frames)]
    if not new:
        return False
    hyps.extend(new)
    branches: list = proc.setdefault("branches", [])
    for hyp in new:
        agreeing = [h for h in hyps if _key(h) == _key(hyp)]
        if len({_instance(h) for h in agreeing}) < 2:
            continue
        branch = _branch(agreeing, proc.get("steps") or [])
        if branch is None:
            continue
        existing = next(
            (b for b in branches if isinstance(b, dict) and b.get("source") == "divergence"
             and _key(b) == _key(hyp)),
            None,
        )
        if existing is None:
            branches.append(branch)
        else:
            existing.update(branch)
    del hyps[:-MAX_HYPOTHESES]
    return True


def mine_observations(
    db: "WorkerDB", kb: "KnowledgeBase", slug: str, observations: list[dict],
) -> int:
    """Record the divergences of *observations* (attached to the Skill
    *slug*) whose frames are still stored.  Returns how many instances
    changed the procedure."""
    proc = kb.get_procedure(slug)
    if proc is None:
        return 0
    changed = 0
    for obs in observations:
        ids = json.loads(obs.get("event_ids_json") or "[]")
        frames = [f for f in map(AnnotatedFrame.from_event, db.get_events_by_ids(ids)) if f]
        frames.sort(key=lambda f: f.timestamp)
        if frames and record_divergences(proc, obs["obs_id"], frames):
            changed += 1
    if changed:
        kb.save_procedure(proc)
    return changed

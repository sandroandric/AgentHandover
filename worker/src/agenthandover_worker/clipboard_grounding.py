"""Copy and paste only where the clipboard shows them.

A step that says the user copied, cut or pasted must have a clipboard
event in its time window: a ``ClipboardChange`` (the daemon's pasteboard
monitor, one per copy or cut) or a ``PasteDetected`` (the browser
extension's paste, with ``matched_copy_hash`` when the pasted content is a
recorded copy).  A model reading screenshots of text appearing in a
document says "pasted" as readily as "typed"; without an event, the claim
is not what was recorded.

The focus timeline carries each frame's ``clipboard_events`` (the events
since the previous frame, :func:`attach_clipboard_events`).  After the SOP
is generated, :func:`ground_clipboard_claims` checks every step against
them and rewrites an unsupported claim as reading or typing text, keeping
the model's words and the reason on the step's evidence.  The check is in
code: the prompt's clipboard note (:func:`clipboard_note`) helps the model,
it does not decide.
"""

from __future__ import annotations

import logging
import re

from agenthandover_worker.event_helpers import parse_json_field, parse_timestamp

logger = logging.getLogger(__name__)

COPY = "copy"
COPIED = "copied"
PASTE = "paste"

# The daemon's event kinds, by what they record.
EVENT_KINDS: dict[str, str] = {"ClipboardChange": COPY, "PasteDetected": PASTE}

_COPY_RE = re.compile(r"\b(copy|copies|copying)\b", re.IGNORECASE)
# "the copied text": a copy made before this step, not by it.
_COPIED_RE = re.compile(r"\bcopied\b", re.IGNORECASE)
_CUT_RE = re.compile(r"^\s*cut\b", re.IGNORECASE)  # "cut" as the step's verb only
_PASTE_RE = re.compile(r"\b(paste|pastes|pasted|pasting)\b", re.IGNORECASE)


def _word(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Rewrites per unsupported claim, in order: the phrases first, then the
# verbs.  Either claim also drops "to/from the clipboard".
_CLIPBOARD_PHRASE = (_word(r"\s+(to|from|into|onto|via|using|in|on)\s+the\s+clipboard\b"), "")
_REWRITES: dict[str, tuple[tuple[re.Pattern, str], ...]] = {
    COPY: (
        # "the copied text" / "the previously copied information" -> "the text"
        (_word(r"\b(the|this|that|these|those)\s+(previously\s+)?copied\s+"), r"\1 "),
        (_word(r"^(\s*)cut\b"), r"\1delete"),
        (_word(r"\bcopy\b"), "read"),
        (_word(r"\bcopies\b"), "reads"),
        (_word(r"\bcopied\b"), "read"),
        (_word(r"\bcopying\b"), "reading"),
    ),
    PASTE: (
        (_word(r"\bpaste\b"), "type"),
        (_word(r"\bpastes\b"), "types"),
        (_word(r"\bpasted\b"), "typed"),
        (_word(r"\bpasting\b"), "typing"),
    ),
}

# The step fields an agent reads as what to do and how to check it.
_STEP_FIELDS = ("step", "action", "description")
_PARAM_FIELDS = ("verify",)


def event_kind(event: dict) -> str | None:
    """``copy`` / ``paste`` for a clipboard event row, else None."""
    return EVENT_KINDS.get(str(parse_json_field(event, "kind_json").get("type") or ""))


def attach_clipboard_events(timeline: list[dict], events: list[dict]) -> None:
    """Give every timeline frame ``clipboard_events``: the clipboard events
    (``{"kind", "timestamp", "event_id"}``) since the previous frame.  An
    event belongs to the first frame at or after it (the screen the copy
    or paste led to), an event after the last frame to the last frame, so
    none is lost.  In place."""
    for frame in timeline:
        frame["clipboard_events"] = []
    timed = [
        (when, frame) for frame in timeline
        if (when := parse_timestamp(frame.get("timestamp"))) is not None
    ]
    if not timed:
        return
    for event in events:
        when = parse_timestamp(event.get("timestamp"))
        if when is None:
            continue
        owner = next((frame for t, frame in timed if t >= when), timed[-1][1])
        owner["clipboard_events"].append(event)


def session_clipboard_kinds(timeline: list[dict]) -> set[str] | None:
    """The kinds of clipboard event the session recorded, or None when the
    timeline carries no clipboard record (a passive demonstration)."""
    if not timeline or not all("clipboard_events" in f for f in timeline):
        return None
    return {e.get("kind") for f in timeline for e in f["clipboard_events"]}


def clipboard_note(timeline: list[dict]) -> str:
    """The SOP prompt's note when the session recorded no clipboard event
    at all ("" otherwise, so a session with copies gets the prompt it
    always had)."""
    if session_clipboard_kinds(timeline) != set():
        return ""
    return (
        "\n\nCLIPBOARD: no copy, cut or paste was recorded during this "
        "session. Text that appears in a document or field was typed by the "
        "user: write those steps as typing, never as Copy or Paste steps, "
        "even where a frame's description says copied or pasted."
    )


def claims(texts: list[str]) -> set[str]:
    """The clipboard actions *texts* claim: ``copy`` (copy or cut, done in
    the step), ``copied`` (content copied before it) and/or ``paste``."""
    found: set[str] = set()
    for text in texts:
        if _COPY_RE.search(text) or _CUT_RE.search(text):
            found.add(COPY)
        if _COPIED_RE.search(text):
            found.add(COPIED)
        if _PASTE_RE.search(text):
            found.add(PASTE)
    return found


def rewrite(text: str, kinds: set[str] | frozenset[str] = frozenset((COPY, PASTE))) -> str:
    """*text* with the wording of the *kinds* of claim rewritten: a copy or
    cut as reading, a paste as typing ("Paste the copied text" -> "Type
    the text")."""
    out = text
    for pattern, repl in (_CLIPBOARD_PHRASE, *(r for k in (COPY, PASTE) if k in kinds
                                               for r in _REWRITES[k])):
        out = pattern.sub(repl, out)
    if out and text[:1].isupper():
        out = out[0].upper() + out[1:]
    return out


def _step_texts(step: dict) -> dict[str, str]:
    """The step's text fields a claim is read from, keyed ``field`` or
    ``parameters.field``."""
    texts = {k: step[k] for k in _STEP_FIELDS if isinstance(step.get(k), str) and step[k]}
    params = step.get("parameters")
    if isinstance(params, dict):
        texts.update({
            f"parameters.{k}": params[k] for k in _PARAM_FIELDS
            if isinstance(params.get(k), str) and params[k]
        })
    return texts


def _window(step: dict, timeline: list[dict]) -> tuple[list[dict], list[dict]]:
    """The clipboard events in the step's time window, and every event up
    to its end.  The window runs from the step's first cited frame to the
    frame after its last (a copy made on a frame is recorded before the
    next one); a step that cites no frame is checked against the whole
    session."""
    ids = set((step.get("evidence") or {}).get("frames") or [])
    cited = [i for i, frame in enumerate(timeline) if frame.get("event_id") in ids]
    if not cited:
        every = [e for f in timeline for e in f["clipboard_events"]]
        return every, every
    end = min(max(cited) + 2, len(timeline))
    in_window = [e for f in timeline[min(cited):end] for e in f["clipboard_events"]]
    before_end = [e for f in timeline[:end] for e in f["clipboard_events"]]
    return in_window, before_end


def ground_clipboard_claims(steps: list[dict], timeline: list[dict]) -> list[dict]:
    """Rewrite the copy/paste claims of *steps* that no clipboard event
    supports; returns one record per rewritten step.

    A copy or cut is supported by a copy event in the step's window; "the
    copied text" by any copy up to the window's end; a paste by a paste
    event in the window, or by any copy up to its end (pastes outside the
    browser are not recorded, but they paste a recorded copy).  An unsupported step has its wording rewritten as
    reading or typing, and ``evidence.clipboard`` records what it claimed,
    the model's original words and why.  In place; a timeline without a
    clipboard record (passive) is left alone.
    """
    if session_clipboard_kinds(timeline) is None:
        return []
    rewritten: list[dict] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        texts = _step_texts(step)
        claimed = claims(list(texts.values()))
        if not claimed:
            continue
        in_window, before_end = _window(step, timeline)
        window_kinds = {e.get("kind") for e in in_window}
        copied_before = any(e.get("kind") == COPY for e in before_end)
        supported = {
            COPY: COPY in window_kinds,
            COPIED: copied_before,
            PASTE: PASTE in window_kinds or copied_before,
        }
        unsupported = sorted({
            COPY if c == COPIED else c for c in claimed if not supported[c]
        })
        if not unsupported:
            continue
        original: dict[str, str] = {}
        for key, text in texts.items():
            new = rewrite(text, set(unsupported))
            if new != text:
                original[key] = text
                if key.startswith("parameters."):
                    step["parameters"][key.split(".", 1)[1]] = new
                else:
                    step[key] = new
        reason = (
            f"no {' or '.join(unsupported)} was recorded on the clipboard "
            "while this step happened; the step is written as the text the "
            "user read or typed"
        )
        record = {"claimed": unsupported, "recorded": False, "reason": reason,
                  "original": original}
        step.setdefault("evidence", {})["clipboard"] = record
        rewritten.append({"step_index": index, **record})
    if rewritten:
        logger.info(
            "Clipboard grounding: rewrote %d step(s) that claimed a copy or paste "
            "the clipboard did not record: %s", len(rewritten),
            "; ".join(
                f"step {r['step_index'] + 1} ({'/'.join(r['claimed'])}): "
                f"{next(iter(r['original'].values()), '')!r}"
                for r in rewritten
            ),
        )
    return rewritten

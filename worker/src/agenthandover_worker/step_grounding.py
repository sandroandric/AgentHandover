"""Steps only where the frames show them, in the order they happened.

A focus Skill's steps are written by a model reading the session's
timeline.  The live run of 2026-09-24 (session 46b767ef, qwen3.5:4b; the
user had Safari on iana.org/domains/reserved, opened TextEdit, pressed
Cmd-N and typed two lines) got back five steps: the new TextEdit document
came first although it was made after Safari; "click the 'Example Domain'
link" named a tab title seen only in Safari's tab strip; "Select the
typed text" happened in no frame; and the second typed line was in no
step.

:func:`ground_steps` checks the generated steps against the timeline in
code, after generation (the SOP prompt asks each step for its frames; the
checks do not depend on the model following it):

1. Every step cites the frames it comes from (``evidence.frames``).  A
   step citing none is dropped, unless no step cites any (then nothing
   can be checked, and the steps are kept as written).
2. An action that clicks a named control (a link, button or tab) needs
   that name in its frames' content: the document's text without the
   window chrome (``frame_differ.content_lines``), the page's DOM, the
   recorded clicks, the location and listed items.  A selection needs a
   frame that shows one (a copy, a click, or a diff that says so), and a
   scroll a diff that says so.  An unsupported part of an action is cut
   ("Scroll down and click the 'Example Domain' link" keeps nothing: the
   step is dropped); what was cut and why is kept on the step's
   ``evidence.grounding`` or, for a dropped step, in the report.
3. Steps are ordered by where their longest run of consecutive cited
   frames starts (the earliest on a tie; stable): one stray citation
   cannot move a step (seen live: a navigation citing frames 11, 13, 14,
   15, where 11 was the paste before it).
4. Every line the user typed that the frames show (a typed text or input
   value of a frame, visible in its document content and not there on the
   previous frame of the same app) appears in some step: as the empty
   input of a typing step on that frame, or as a new "Type" step at its
   place in the order.
"""

from __future__ import annotations

import logging
import re

from agenthandover_worker.clipboard_grounding import COPY
from agenthandover_worker.dom import accessible_name, flatten_dom, implicit_role
from agenthandover_worker.frame_differ import content_lines

logger = logging.getLogger(__name__)

# Quoted names: '...', "...", ‘...’, “...”.  A quote opens after a space or
# bracket and closes before a space, punctuation or the end, so the
# apostrophe in 'SQLite's WAL mode' does not close it.
_QUOTED_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s(\[]))(?:'(.+?)'|\"(.+?)\"|‘(.+?)’|“(.+?)”)"
    r"(?=$|[\s.,;:!?)\]])"
)
# "Press" names a control only with its kind ("press the Send button"):
# "press 'Return'" is a key.
_CLICK_RE = re.compile(r"\b(click|clicks|clicked|clicking|tap|taps|tapped|tapping)\b", re.I)
_ELEMENTS = r"(link|button|tab|menu item|menu|checkbox|check box|icon|toggle|radio button)"
_ELEMENT_RE = re.compile(rf"\b{_ELEMENTS}s?\b", re.IGNORECASE)
# "Click the Send button" -> "Send"
_NAMED_ELEMENT_RE = re.compile(
    r"\b(?:click|clicks|clicked|clicking|tap|taps|tapped|press|presses|pressed)\s+"
    rf"(?:on\s+)?(?:the\s+|a\s+)?(.+?)\s+{_ELEMENTS}s?\b",
    re.IGNORECASE,
)
_SELECT_RE = re.compile(r"^\s*(select|selects|selecting|highlight|highlights|highlighting)\b", re.I)
_SCROLL_RE = re.compile(r"^\s*(scroll|scrolls|scrolling)\b", re.IGNORECASE)
_SELECTION_EVIDENCE_RE = re.compile(r"\b(select|selects|selected|selecting|highlight\w*)\b", re.I)
# A save or rename the step claims: a save sheet or a document that got a
# real name must show it (seen live: "Save the document and rename it to
# 'Untitled 4'" for a new, unsaved TextEdit window).
_SAVE_RE = re.compile(r"^\s*(save|saves|saving|rename|renames|renaming)\b", re.IGNORECASE)
_UNTITLED_RE = re.compile(r"^\s*untitled(\s+\d+)?(\s*[-—].*)?$", re.IGNORECASE)
_TYPING_RE = re.compile(r"\b(type|types|typed|typing|enter|enters|write|writes)\b", re.I)
# "Open Safari and navigate to ..." / "Scroll down, then click ..."
_CLAUSE_RE = re.compile(r",?\s+(?:and then|then|and)\s+|;\s*", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s+")
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\s*\}\}")
# A typed line counts from this many characters and two words on: OCR
# fragments and single keys are not lines the user typed.
_MIN_TYPED_CHARS = 8
# The text-entry roles whose own label or placeholder is not typed text.
_TEXT_ROLES = frozenset({"textbox", "searchbox", "combobox"})

REWRITTEN = "rewritten"
ADDED = "added"
INPUT_ADDED = "input_added"


def _norm(text: object) -> str:
    """Lowercase words and numbers, one space apart: punctuation, quotes
    and the prompt's `` | `` separators do not count."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _contains(haystack: str, needle: str) -> bool:
    """*needle* as whole words of *haystack* (both normalised)."""
    return bool(needle) and f" {needle} " in f" {haystack} "


# ---------------------------------------------------------------------------
# What a frame shows
# ---------------------------------------------------------------------------


def frame_content(frame: dict) -> list[str]:
    """The frame's document lines without the window chrome: the
    ``content_text`` the focus processor attaches (``frame_differ.
    content_lines``, with the frame's geometry), else the annotation's
    screen text without menu, font and format lines and toolbar numbers
    (``content_lines`` without the geometry)."""
    if isinstance(frame.get("content_text"), list):
        return [str(line) for line in frame["content_text"]]
    annotation = frame.get("annotation") if isinstance(frame.get("annotation"), dict) else {}
    return content_lines(None, annotation)[0]


def _click_names(frame: dict) -> list[str]:
    names = []
    for click in frame.get("click_targets") or []:
        target = click.get("target") if isinstance(click, dict) else None
        if isinstance(target, dict):
            names += [str(target.get(k) or "") for k in ("innerText", "ariaLabel", "text", "name")]
    return [n for n in names if n.strip()]


def _frame_corpus(frame: dict) -> str:
    """Everything the frame shows that a step may name: its document
    lines, location, URLs, listed items, DOM names and clicked controls."""
    annotation = frame.get("annotation") if isinstance(frame.get("annotation"), dict) else {}
    key_text = annotation.get("key_text") if isinstance(annotation.get("key_text"), dict) else {}
    parts = [
        *frame_content(frame),
        str(annotation.get("location") or ""),
        *(str(u) for u in key_text.get("urls") or []),
        *(str(i.get("label") or "") for i in annotation.get("visible_items") or []
          if isinstance(i, dict)),
        *(accessible_name(n) for n in flatten_dom(frame.get("dom_nodes"))),
        *_click_names(frame),
    ]
    return _norm("\n".join(parts))


def _diff_text(frame: dict) -> str:
    diff = frame.get("diff") if isinstance(frame.get("diff"), dict) else {}
    return " ".join(
        [str(a) for a in diff.get("actions") or []]
        + [str(diff.get("step_description") or ""), str(diff.get("navigation") or "")]
    )


def _control_names(timeline: list[dict]) -> set[str]:
    """The labels and placeholders of the session's text fields: the
    clicked fields' and the DOM's text-entry elements' names, and the
    focused fields' labels."""
    names: set[str] = set()
    for frame in timeline:
        for click in frame.get("click_targets") or []:
            target = click.get("target") if isinstance(click, dict) else None
            if isinstance(target, dict) and implicit_role(target) in _TEXT_ROLES:
                names.update(_norm(target.get(k)) for k in ("ariaLabel", "name", "placeholder"))
        names.update(
            _norm(accessible_name(n)) for n in flatten_dom(frame.get("dom_nodes"))
            if implicit_role(n) in _TEXT_ROLES
        )
        field = frame.get("focused_field")
        if isinstance(field, dict) and field.get("label"):
            names.add(_norm(field["label"]))
    names.discard("")
    return names


# ---------------------------------------------------------------------------
# Actions against their frames
# ---------------------------------------------------------------------------


def _targets(clause: str) -> list[str]:
    """The control names a clause clicks, when it clicks a named control
    (quoted, or "the X button/link/tab")."""
    if not (_CLICK_RE.search(clause) or _ELEMENT_RE.search(clause)):
        return []
    quoted = [next(g for g in m.groups() if g) for m in _QUOTED_RE.finditer(clause)]
    if quoted:
        return quoted
    named = _NAMED_ELEMENT_RE.search(clause)
    return [named.group(1)] if named else []


def _app(frame: dict) -> str:
    annotation = frame.get("annotation") if isinstance(frame.get("annotation"), dict) else {}
    return str(annotation.get("app") or frame.get("app") or "")


def _location(frame: dict) -> str:
    annotation = frame.get("annotation") if isinstance(frame.get("annotation"), dict) else {}
    return str(annotation.get("location") or "")


def _scrolled(timeline: list[dict], index: int) -> bool:
    """Whether frame *index* shows the page scrolled since the previous
    frame of the same app and place: lines left while others arrived, and
    some stayed.  A page that loads only gains lines (seen live: the diff
    model called a loading page "scrolled down", twice)."""
    frame = timeline[index]
    if frame.get("kind") == "ScrollReadSnapshot":
        return True
    before = next((
        timeline[j] for j in range(index - 1, -1, -1) if _app(timeline[j]) == _app(frame)
    ), None)
    if before is None or _location(before) != _location(frame):
        return False
    old = {_norm(line) for line in frame_content(before)} - {""}
    new = {_norm(line) for line in frame_content(frame)} - {""}
    return bool(old - new) and bool(new - old) and bool(old & new)


def _saved(timeline: list[dict], indexes: list[int]) -> bool:
    """Whether the frames around *indexes* show a save: a save sheet, or
    the document going from an untitled name to a real one."""
    window = range(max(indexes[0] - 1, 0), min(indexes[-1] + 2, len(timeline)))
    names = [_location(timeline[i]) for i in window]
    if any("save as" in _norm("\n".join(frame_content(timeline[i]))) for i in window):
        return True
    return any(
        _UNTITLED_RE.match(a) and b and not _UNTITLED_RE.match(b)
        for a, b in zip(names, names[1:])
    )


def _unsupported(
    clause: str, frames: list[dict], window_kinds: set[str],
    timeline: list[dict] | None = None, indexes: list[int] | None = None,
) -> str | None:
    """Why *clause* is not shown by *frames* (the step's cited frames, at
    *indexes* of *timeline*), or None when it is.  Scrolls and saves need
    what the recording shows, not the diff model's own words."""
    timeline = timeline or frames
    indexes = indexes if indexes is not None else list(range(len(frames)))
    for target in _targets(clause):
        if not any(_contains(_frame_corpus(f), _norm(target)) for f in frames):
            return f"'{target.strip()}' is not in the frames this step cites"
    if _SELECT_RE.search(clause):
        shown = COPY in window_kinds or any(
            f.get("click_targets") or _SELECTION_EVIDENCE_RE.search(_diff_text(f))
            for f in frames
        )
        if not shown:
            return "no frame this step cites shows a selection"
    if _SCROLL_RE.search(clause):
        if not any(_scrolled(timeline, i) for i in indexes):
            return "no frame this step cites shows a scroll"
    if _SAVE_RE.search(clause) and indexes and not _saved(timeline, indexes):
        return "no frame shows a save (a save sheet, or the document getting a name)"
    return None


def clause_question(clause: str) -> tuple[str, str]:
    """(what the recording did not show, the question to ask) for a clause
    :func:`_unsupported` cut: "a scroll" / "Did you scroll?"."""
    if _SCROLL_RE.search(clause):
        return "a scroll", "Did you scroll?"
    if _SELECT_RE.search(clause):
        return "a selection", "Did you select it?"
    if _SAVE_RE.search(clause):
        return "a save", "Did you save it?"
    targets = _targets(clause)
    if targets:
        names = " or ".join(f"'{t.strip()}'" for t in targets)
        return names, f"Did you click {names}?"
    return f"'{clause.strip()}'", "Did you do that?"


def _clauses(action: str) -> list[str]:
    """*action* split at "and", "then" and semicolons, never inside a
    quoted name ('Terms and Conditions')."""
    quoted = [m.span() for m in _QUOTED_RE.finditer(action)]
    clauses, start = [], 0
    for sep in _CLAUSE_RE.finditer(action):
        if any(a <= sep.start() < b for a, b in quoted):
            continue
        clauses.append(action[start:sep.start()])
        start = sep.end()
    clauses.append(action[start:])
    return [c.strip() for c in clauses if c.strip()]


def _sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].upper() + text[1:] if text else text


def _step_texts(step: dict, variables: dict[str, str]) -> str:
    """What the step does (its action and the input it types), with each
    ``{{name}}`` read as its example value.  Not its description, target
    or verify: a line that only a check mentions ("The full text 'Reserved
    for documentation: {{domain_list}} ...' is visible", seen live) is
    typed by no step."""
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    text = "\n".join(str(v or "") for v in (
        step.get("step"), step.get("action"), params.get("input"),
    ))
    return _VAR_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), text)


# ---------------------------------------------------------------------------
# Typed lines
# ---------------------------------------------------------------------------


def typed_lines(timeline: list[dict]) -> list[tuple[int, str]]:
    """``(frame index, line)`` for each distinct line the user typed that
    the frames show, at the first frame that shows it.

    A candidate is a line of a frame's typed text or of a diff input's
    value.  It counts when it is in that frame's document content, was
    not there on the previous frame of the same app (it was added, not
    read), and is not a text field's own label or placeholder (the Slack
    composer's "Message #daily-news").  A line caught while it was being
    typed stands for its finished form: inside a longer counted line it
    is not counted; inside a longer content line of a later frame of the
    same app it becomes that line (the frames of a typing session may
    only ever catch the line half typed, seen live)."""
    controls = _control_names(timeline)
    found: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    last_by_app: dict[str, int] = {}
    for i, frame in enumerate(timeline):
        annotation = frame.get("annotation") if isinstance(frame.get("annotation"), dict) else {}
        app = str(annotation.get("app") or frame.get("app") or "")
        content = _norm("\n".join(frame_content(frame)))
        before = last_by_app.get(app)
        previous = _norm("\n".join(frame_content(timeline[before]))) if before is not None else ""
        last_by_app[app] = i
        key_text = annotation.get("key_text") if isinstance(annotation.get("key_text"), dict) else {}
        raw = str(key_text.get("typed_text") or "").splitlines()
        diff = frame.get("diff") if isinstance(frame.get("diff"), dict) else {}
        for item in diff.get("inputs") or []:
            if isinstance(item, dict):
                raw += re.split(r"\n| \| ", str(item.get("value") or ""))
        for line in raw:
            line = _LIST_MARKER_RE.sub("", line).strip()
            norm = _norm(line)
            if (
                len(norm) < _MIN_TYPED_CHARS or " " not in norm or norm in seen
                or norm in controls or not _contains(content, norm)
                or _contains(previous, norm)
            ):
                continue
            seen.add(norm)
            found.append((i, line, norm, app))

    def finished(i: int, line: str, norm: str, app: str) -> tuple[str, str]:
        """The longest later content line of *app* that extends *line*
        (by characters: typing stops mid-word, "example.or"), else it."""
        best = (line, norm)
        for j in range(i + 1, len(timeline)):
            frame = timeline[j]
            annotation = frame.get("annotation") if isinstance(frame.get("annotation"), dict) else {}
            if str(annotation.get("app") or frame.get("app") or "") != app:
                continue
            for text in frame_content(frame):
                other = _norm(text)
                if len(other) > len(best[1]) and norm in other:
                    best = (_LIST_MARKER_RE.sub("", text).strip(), other)
        return best

    out: list[tuple[int, str]] = []
    done: set[str] = set()
    for i, line, norm, app in found:
        if any(
            norm != other and (other.startswith(norm) or _contains(other, norm))
            for _j, _l, other, _a in found
        ):
            continue
        line, norm = finished(i, line, norm, app)
        if norm not in done:
            done.add(norm)
            out.append((i, line))
    return out


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def ground_steps(
    steps: list[dict], timeline: list[dict], variables: list[dict] | None = None,
) -> dict:
    """Check *steps* (the SOP template's, in place) against *timeline* (the
    session's whole annotated timeline); returns what was changed:
    ``{"dropped": [...], "rewritten": [...], "added": [...],
    "reordered": bool}`` (``dropped`` entries carry the step's
    ``action``, cited ``frames`` and ``reason``).

    Rewritten and added steps record ``evidence.grounding`` (``status``,
    ``reason`` and, when rewritten, the model's ``original`` words and the
    ``cut`` clauses with their reasons).  A dropped entry also keeps the
    step's ``description`` and ``parameters`` and where it stood
    (``after_frames``), so a user's answer can restore it
    (:mod:`questions`).  A timeline or steps without frame citations are
    left alone."""
    report: dict = {"dropped": [], "rewritten": [], "added": [], "reordered": False}
    position = {f.get("event_id"): i for i, f in enumerate(timeline) if f.get("event_id")}

    def cited(step: dict) -> list[int]:
        ids = (step.get("evidence") or {}).get("frames") or []
        return sorted({position[e] for e in ids if e in position})

    if not timeline or not any(isinstance(s, dict) and cited(s) for s in steps):
        if steps:
            logger.warning(
                "Step grounding: no step cites a frame of the timeline; %d step(s) "
                "kept unchecked", len(steps),
            )
        return report

    # Each dropped step's record and where it stood: its cited frames, or
    # (citing none) the kept step before it in the model's order.
    placed: list[tuple[dict, list[int], dict | None]] = []

    def drop(step: dict, reason: str, indexes: list[int]) -> None:
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        record = {
            "action": str(step.get("step") or step.get("action") or ""),
            "frames": list((step.get("evidence") or {}).get("frames") or []),
            "reason": reason,
            # What the user's "Yes, I did" restores (``questions``).
            "description": str(step.get("description") or ""),
            "parameters": dict(params),
        }
        report["dropped"].append(record)
        placed.append((record, indexes, kept[-1] if kept else None))

    kept: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        indexes = cited(step)
        if not indexes:
            drop(step, "the step cites no recorded frame", indexes)
            continue
        frames = [timeline[i] for i in indexes]
        # A copy in the step's window (its frames to the frame after the
        # last): a copy needs a selection.
        window = timeline[indexes[0]:min(indexes[-1] + 2, len(timeline))]
        window_kinds = {e.get("kind") for f in window for e in f.get("clipboard_events") or []}
        action = str(step.get("step") or "")
        clauses = _clauses(action)
        reasons = {
            c: r for c in clauses
            if (r := _unsupported(c, frames, window_kinds, timeline, indexes))
        }
        if not reasons:
            kept.append(step)
            continue
        reason = "; ".join(dict.fromkeys(reasons.values()))
        remaining = [c for c in clauses if c not in reasons]
        if not remaining:
            drop(step, reason, indexes)
            continue
        original = {"step": action, "description": str(step.get("description") or "")}
        step["step"] = _sentence(" and ".join(remaining))
        cut = [_norm(t) for c in reasons for t in _targets(c)] + [_norm(c) for c in reasons]
        if any(_contains(_norm(original["description"]), t) for t in cut if t):
            step["description"] = f"{step['step']}."
        record = {"status": REWRITTEN, "reason": f"cut from the step: {reason}",
                  "original": original,
                  "cut": [{"clause": c, "reason": r} for c, r in reasons.items()]}
        step.setdefault("evidence", {})["grounding"] = record
        report["rewritten"].append({"action": step["step"], **record})
        kept.append(step)

    # 3. The order the frames were recorded in (stable for ties), by where
    # each step's longest run of consecutive cited frames starts, so one
    # stray citation cannot move it.
    def anchor(frames: list[int]) -> int:
        runs: list[list[int]] = []
        for f in frames:
            if runs and f == runs[-1][-1] + 1:
                runs[-1].append(f)
            else:
                runs.append([f])
        return max(runs, key=len)[0]  # max keeps the earliest of equal runs

    order = sorted(range(len(kept)), key=lambda k: (anchor(cited(kept[k])), k))
    report["reordered"] = order != list(range(len(kept)))
    kept = [kept[k] for k in order]

    # 4. Every typed line the frames show is in some step.
    examples = {
        str(v.get("name")): str(v.get("example") or "")
        for v in variables or [] if isinstance(v, dict) and v.get("name")
    }
    for index, line in typed_lines(timeline):
        norm = _norm(line)
        if any(_contains(_norm(_step_texts(s, examples)), norm) for s in kept):
            continue
        frame = timeline[index]
        app = str((frame.get("annotation") or {}).get("app") or frame.get("app") or "")
        reason = "the frames show this line typed, and no step had it"
        # A typing step holding only the start of the line (caught half
        # typed) gets the finished line.
        partial = next((
            s for s in kept
            if _TYPING_RE.search(str(s.get("step") or ""))
            and (text := _norm((s.get("parameters") or {}).get("input")))
            and len(text) < len(norm) and text in norm
        ), None)
        if partial is not None:
            params = partial.setdefault("parameters", {})
            record = {"status": INPUT_ADDED, "reason": "the frames show the finished line",
                      "original": str(params.get("input") or "")}
            params["input"] = line
            partial.setdefault("evidence", {}).setdefault("grounding", record)
            report["added"].append({"action": partial["step"], "input": line, **record})
            continue
        typing = next((
            s for s in kept
            if _TYPING_RE.search(str(s.get("step") or ""))
            and not str((s.get("parameters") or {}).get("input") or "").strip()
            and cited(s)[0] <= index <= cited(s)[-1] + 1
        ), None)
        if typing is not None:
            typing.setdefault("parameters", {})["input"] = line
            record = {"status": INPUT_ADDED, "reason": reason}
            typing.setdefault("evidence", {}).setdefault("grounding", record)
            report["added"].append({"action": typing["step"], "input": line, **record})
            continue
        at = max((k + 1 for k, s in enumerate(kept) if cited(s)[0] <= index), default=0)
        location = str((frame.get("annotation") or {}).get("location") or "")
        new = {
            "step": f'Type "{line}"',
            "description": "Type this line exactly as the user did in the recording.",
            "target": location,
            "parameters": {
                **({"app": app} if app else {}),
                **({"location": location} if location else {}),
                "input": line,
                "verify": f'The text "{line}" appears in the document.',
            },
            "confidence": 0.0,
            "pre_state": {},
            "evidence": {
                "frames": [frame["event_id"]] if frame.get("event_id") else [],
                "dom_snapshot_ids": [],
                "grounding": {"status": ADDED, "reason": reason},
            },
            "anchor": None,
        }
        kept.insert(at, new)
        report["added"].append({"action": new["step"], "input": line, "status": ADDED,
                                "reason": reason})

    # Where each dropped step stood among the final steps: after the last
    # one whose frames start at or before its own (citing none, after the
    # step before it in the model's order).  That step is named by its
    # frames (``after_frames``; empty: at the start), which later answers
    # keep while they renumber the steps.
    for record, indexes, before in placed:
        if indexes:
            start = anchor(indexes)
            earlier = [s for s in kept if anchor(cited(s)) <= start]
            before = earlier[-1] if earlier else None
        record["after_frames"] = list(((before or {}).get("evidence") or {}).get("frames") or [])

    steps[:] = kept
    if report["dropped"] or report["rewritten"] or report["added"] or report["reordered"]:
        logger.info(
            "Step grounding: %d dropped, %d rewritten, %d typed line(s) added%s: %s",
            len(report["dropped"]), len(report["rewritten"]), len(report["added"]),
            ", steps reordered by their frames" if report["reordered"] else "",
            "; ".join(
                [f"dropped {d['action']!r} ({d['reason']})" for d in report["dropped"]]
                + [f"rewrote {r['original']['step']!r} ({r['reason']})" for r in report["rewritten"]]
                + [f"typed {a['input']!r}" for a in report["added"]]
            ),
        )
    return report

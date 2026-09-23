"""Frame-to-frame diff engine for consecutive scene annotations.

Compares pairs of consecutive annotations to produce action descriptions:
what the user DID between two frames (typed text, clicked buttons,
navigated pages, etc.).

Produces four types of markers:
- "action": real diff via text-only LLM call (~3.6s)
- "app_switch": app changed (code-only, free)
- "session_gap": time gap > threshold (code-only, free)
- "no_change": identical content (code-only, free)
- "stale_skip": previous frame was stale-skipped (code-only, free)
- "diff_failed": LLM diff failed (fallback marker)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from agenthandover_worker.annotation_cache import normalise_screen_text
from agenthandover_worker.event_helpers import display_name, parse_json_field
from agenthandover_worker.json_extract import extract_json
from agenthandover_worker.redaction import redact
from agenthandover_worker.scene_annotator import as_compact, primary_display_bounds
from agenthandover_worker.model_profiles import STANDARD_TIER

logger = logging.getLogger(__name__)

# Time gap threshold for session_gap marker (seconds)
SESSION_GAP_THRESHOLD = 600  # 10 minutes


@dataclass
class DiffConfig:
    """Configuration for the frame differ."""

    model: str = STANDARD_TIER.annotation_model  # main() passes the configured model
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 400
    session_gap_seconds: int = SESSION_GAP_THRESHOLD


@dataclass
class DiffResult:
    """Result of comparing two consecutive annotations."""

    event_id: str
    # None: the pair needs the model and the caller did not allow it.
    diff: dict | None
    inference_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Code-only markers (no LLM call needed)
# ---------------------------------------------------------------------------

def _make_app_switch_marker(from_app: str, to_app: str) -> dict:
    return {
        "diff_type": "app_switch",
        "from_app": from_app,
        "to_app": to_app,
    }


def _make_session_gap_marker(gap_seconds: float, reason: str = "time_gap") -> dict:
    return {
        "diff_type": "session_gap",
        "gap_seconds": int(gap_seconds),
        "reason": reason,
    }


def _make_no_change_marker(what_doing: str, duration_seconds: float) -> dict:
    return {
        "diff_type": "no_change",
        "what_doing": what_doing,
        "duration_seconds": int(duration_seconds),
    }


def _make_stale_skip_marker(reason: str = "stale_frame") -> dict:
    return {
        "diff_type": "stale_skip",
        "reason": reason,
    }


def _make_diff_failed_marker(error: str) -> dict:
    return {
        "diff_type": "diff_failed",
        "error": error,
    }


# ---------------------------------------------------------------------------
# Window chrome: text that is not the document
# ---------------------------------------------------------------------------

# The title bar and toolbar of a window (a TextEdit window's title bar and
# format bar end about 58 points down): a short label entirely inside this
# band is a control, not content.
TOOLBAR_BAND_POINTS = 60
# A toolbar label has at most this many words; a longer line in the band
# is content (a web page's own header, a document's first line).
_TOOLBAR_LABEL_MAX_WORDS = 4

# Words that make up menus, font and format controls: a line made only of
# them (a font popup's "Helvetica Regular", a menu bar's "File Edit
# Format") is chrome wherever it is.  OCR misreads them ("Halvetica
# Ragular"), so words of four letters or more match at 75% similarity.
_CHROME_WORDS = frozenset("""
    helvetica neue arial times new roman courier menlo monaco georgia verdana
    futura avenir next gill sans palatino optima baskerville didot geneva
    lucida grande sf pro text display mono san francisco calibri cambria
    tahoma trebuchet charter hoefler rockwell
    regular bold italic oblique light medium semibold condensed black heavy
    thin ultralight book
    styles style spacing lists list paragraph body title heading subheading
    caption default align
    file edit view insert format tools window help history bookmarks develop
    go arrange share table
    b i u
""".split())
_WORD_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.IGNORECASE)
_CHROME_MATCH = 0.75


def _is_chrome_word(word: str) -> bool:
    if word in _CHROME_WORDS:
        return True
    if len(word) < 4:
        return False
    return any(
        len(known) >= 4 and SequenceMatcher(None, word, known).ratio() >= _CHROME_MATCH
        for known in _CHROME_WORDS
    )


def is_chrome_line(line: str, app: str = "") -> bool:
    """True when every word of *line* is a menu, font or format control
    word.  The frontmost *app*'s name counts too (its menu bar starts with
    it), and so do numbers (a point size), but only beside a control word:
    "Notes" alone is a window or sidebar title, "12" alone may be content."""
    words = [w.lower() for w in _WORD_RE.findall(line)]
    app_words = {w.lower() for w in _WORD_RE.findall(app)}
    controls = [w for w in words if not w[0].isdigit() and w not in app_words]
    return bool(controls) and all(_is_chrome_word(w) for w in controls)


def _ocr_boxes(event: dict) -> dict[str, list]:
    """OCR line text -> its ``bbox_normalized`` (first occurrence)."""
    ocr = parse_json_field(event, "metadata_json").get("ocr")
    boxes: dict[str, list] = {}
    for el in (ocr.get("elements") if isinstance(ocr, dict) else None) or []:
        if not isinstance(el, dict):
            continue
        text, bbox = str(el.get("text") or "").strip(), el.get("bbox_normalized")
        if text and isinstance(bbox, list) and len(bbox) == 4 and all(
            isinstance(v, (int, float)) for v in bbox
        ):
            boxes.setdefault(text, bbox)
    return boxes


def _window_frame(event: dict) -> tuple[float, float, float, float, float] | None:
    """The frontmost window in the screenshot's 0..1 coordinates
    ``(x, y, w, h, toolbar band height)``, or None without the geometry."""
    bounds = parse_json_field(event, "window_json").get("bounds_global_px")
    display = primary_display_bounds(event)
    if display is None or not (
        isinstance(bounds, list) and len(bounds) == 4
        and all(isinstance(v, (int, float)) for v in bounds) and bounds[2] and bounds[3]
    ):
        return None
    dx, dy, dw, dh = display
    x, y, w, h = bounds
    return (x - dx) / dw, (y - dy) / dh, w / dw, h / dh, TOOLBAR_BAND_POINTS / dh


def content_lines(
    event: dict | None, annotation: dict, *, geometry: bool = True,
) -> tuple[list[str], list[str]]:
    """The annotation's screen text split into the document's lines and
    the window chrome's (``(content, chrome)``).

    A line made only of menu, font and format words is chrome
    (:func:`is_chrome_line`).  With *geometry* and the frame's geometry
    (the OCR boxes and the window's bounds on the primary display), so is
    a line outside the frontmost window (the menu bar, another window, the
    Dock) and a short label entirely inside the window's title and toolbar
    band (tabs, the address bar, font popups).

    The diff prompt drops only the chrome words (``geometry=False``): the
    address bar and tab titles are how the model sees a typed URL or a
    switched tab.  The claimed edits are checked against the document's
    text with the geometry (:func:`ground_text_changes`)."""
    lines = [
        line.strip() for line in str(annotation.get("_screen_text") or "").splitlines()
        if line.strip()
    ]
    app = str(annotation.get("app") or "")
    if event is not None:
        app = app or display_name(parse_json_field(event, "window_json").get("app_id"))
    boxes = _ocr_boxes(event) if event is not None and geometry else {}
    window = _window_frame(event) if event is not None and boxes else None
    content: list[str] = []
    chrome: list[str] = []
    for line in lines:
        bbox = boxes.get(line)
        outside = in_band = False
        if window is not None and bbox is not None:
            wx, wy, ww, wh, band = window
            bx, by, bw, bh = bbox
            cx, cy = bx + bw / 2, by + bh / 2
            outside = not (wx <= cx <= wx + ww and wy <= cy <= wy + wh)
            in_band = (
                by + bh <= wy + band
                and len(line.split()) <= _TOOLBAR_LABEL_MAX_WORDS
            )
        (chrome if outside or in_band or is_chrome_line(line, app) else content).append(line)
    return content, chrome


# ---------------------------------------------------------------------------
# Edge case detection
# ---------------------------------------------------------------------------

def _parse_annotation(event: dict) -> dict | None:
    """Extract the parsed annotation from an event row, in the compact
    schema (:func:`scene_annotator.as_compact`)."""
    raw = event.get("scene_annotation_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return as_compact(parsed) if isinstance(parsed, dict) else None


def _parse_timestamp(ts_str: str) -> float:
    """Parse ISO timestamp to Unix epoch seconds."""
    from datetime import datetime, timezone

    # Handle various ISO formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f+00:00",
        "%Y-%m-%dT%H:%M:%S+00:00",
    ):
        try:
            dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    # Fallback: try fromisoformat
    try:
        from datetime import datetime as _dt
        dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _detect_edge_case(
    prev_event: dict,
    curr_event: dict,
    prev_annotation: dict,
    curr_annotation: dict,
    config: DiffConfig,
) -> dict | None:
    """Check for edge cases that produce code-only markers (no LLM needed).

    Returns a marker dict if an edge case is detected, None otherwise.
    """
    prev_ts = prev_event.get("timestamp", "")
    curr_ts = curr_event.get("timestamp", "")

    # Time gap check
    if prev_ts and curr_ts:
        gap = _parse_timestamp(curr_ts) - _parse_timestamp(prev_ts)
        if gap > config.session_gap_seconds:
            return _make_session_gap_marker(gap, "time_gap")

    # App switch: don't short-circuit — fall through to LLM diff.
    # App switches are often the most meaningful transitions (e.g.,
    # "copied product info from PH, switched to Gmail, pasted into email").
    # The LLM diff will capture what happened during the switch.
    prev_app = prev_annotation.get("app", "")
    curr_app = curr_annotation.get("app", "")

    # No-change check: same app + location + same screen text and typing
    prev_loc = prev_annotation.get("location", "")
    curr_loc = curr_annotation.get("location", "")
    prev_what = prev_annotation.get("task_context", {}).get("what_doing", "")
    curr_what = curr_annotation.get("task_context", {}).get("what_doing", "")

    # Screen text compared with its volatile parts (clocks, "5s ago",
    # counters) folded, as the annotation cache compares it: a screen the
    # cache calls unchanged is not sent to the model to learn the same.
    # Only the document's text counts: a toolbar that flickers between
    # OCR readings is not a change (:func:`content_lines`).
    def _content(event: dict, annotation: dict) -> tuple:
        kt = annotation.get("key_text") or {}
        screen = normalise_screen_text(
            "\n".join(content_lines(event, annotation, geometry=False)[0])
        )
        return (screen, kt.get("typed_text", ""))

    if (
        prev_app == curr_app
        and prev_loc == curr_loc
        and _content(prev_event, prev_annotation) == _content(curr_event, curr_annotation)
        and prev_what == curr_what
    ):
        gap = 0.0
        if prev_ts and curr_ts:
            gap = _parse_timestamp(curr_ts) - _parse_timestamp(prev_ts)
        return _make_no_change_marker(curr_what, gap)

    # No edge case — needs LLM diff
    return None


# ---------------------------------------------------------------------------
# LLM diff prompt
# ---------------------------------------------------------------------------

DIFF_SYSTEM_PROMPT = (
    "You compare two consecutive screen annotations and identify what "
    "actions the user took between them. "
    "Respond with ONLY valid JSON, no markdown fences, no commentary."
)

DIFF_PROMPT_TEMPLATE = """\
Compare these two consecutive screen annotations and identify what the user did between Frame A and Frame B.

FRAME A (before):
{frame_a}

FRAME B (after):
{frame_b}

Respond with this JSON structure:
{{
  "diff_type": "action",
  "actions": ["<list each distinct user action as a sentence>"],
  "inputs": [
    {{"field": "<field name>", "value": "<what was typed/selected>"}}
  ],
  "navigation": "<describe any page/URL/screen changes, or 'none'>",
  "step_description": "<one sentence summarizing what the user did>"
}}

Respond with ONLY the JSON object."""


# The screen text per frame in the diff prompt (two frames per call).
_DIFF_SCREEN_TEXT_MAX_CHARS = 600


def _format_annotation_for_diff(annotation: dict, content: list[str] | None = None) -> str:
    """Format an annotation dict into a compact text representation for the
    diff prompt, redacted (rows stored before the current redaction rules
    carry what they carried).  *content* is the frame's document text
    (:func:`content_lines`), the screen text when not given: the window's
    chrome is never offered as something the user edited."""
    lines = []
    app = annotation.get("app", "?")
    location = annotation.get("location", "?")
    lines.append(f"App: {app}")
    lines.append(f"Location: {location}")

    if content is None:
        content = content_lines(None, annotation, geometry=False)[0]
    screen = " | ".join(content)
    if screen:
        lines.append(f"Screen text: {screen[:_DIFF_SCREEN_TEXT_MAX_CHARS]}")
    kt = annotation.get("key_text") or {}
    if kt.get("typed_text"):
        lines.append(f"Typed: {kt['typed_text']}")
    items = [
        f"{item.get('label')}{' (chosen)' if item.get('chosen') else ''}"
        for item in annotation.get("visible_items") or []
        if isinstance(item, dict) and item.get("label")
    ]
    if items:
        lines.append(f"Items: {'; '.join(items)}")

    ui = annotation.get("ui_state", {})
    if ui.get("active_element"):
        lines.append(f"Active element: {ui['active_element']}")
    if ui.get("modal"):
        lines.append(f"Modal/popup: {ui['modal']}")

    tc = annotation.get("task_context", {})
    if tc.get("what_doing"):
        lines.append(f"Doing: {tc['what_doing']}")

    return redact("\n".join(lines))


# ---------------------------------------------------------------------------
# JSON validation for diff output
# ---------------------------------------------------------------------------

def _validate_diff(raw: str) -> dict | None:
    """Parse and validate a diff JSON response."""
    try:
        data = extract_json(raw)
    except ValueError:
        return None

    # Must have diff_type and at least step_description or actions
    if "diff_type" not in data:
        data["diff_type"] = "action"

    if "actions" not in data and "step_description" not in data:
        return None

    return data


# A text change the diff claims ("changed 'X' to 'Y'", "typed 'Z'") must be
# in the document: quoted text of an edit that is on neither frame's content
# (the live run's "changing 'Halvetica Ragular' to 'IANA'", read off
# TextEdit's font popup) drops that action.
_EDIT_VERB_RE = re.compile(
    r"\b((re)?typ(e|es|ed|ing)|enter(s|ed|ing)?|chang(e|es|ed|ing)|correct(s|ed|ing)?|"
    r"replac(e|es|ed|ing)|edit(s|ed|ing)?|writ(e|es|ing)|wrote|written|"
    r"delet(e|es|ed|ing)|renam(e|es|ed|ing)|updat(e|es|ed|ing)|fix(es|ed|ing)?|"
    r"modif(y|ies|ied|ying)|insert(s|ed|ing)?|add(s|ed|ing)?|remov(e|es|ed|ing))\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"\u201c\u2018']([^\"\u201d\u2019']{2,})[\"\u201d\u2019']")


def _squash(text: str) -> str:
    """Lowercase, one space between words; `` | `` (how the prompt joins a
    frame's lines, and the model copies it) counts as a space."""
    return " ".join(str(text).lower().replace(" | ", " ").split())


def ground_text_changes(diff: dict, corpus: str, chrome: str = "", app: str = "") -> dict:
    """*diff* without the text changes that are not the user's.

    *corpus* is what the user can have edited (both frames' document text,
    typed text and locations), *chrome* the windows' controls
    (:func:`content_lines`).  An edit action ("changed 'X' to 'Y'",
    "typed 'Z'") whose quoted text is not in *corpus* is dropped, and the
    summary with it when it quotes such text; an input whose value is a
    control's text (in *chrome* and not in *corpus*, or made of control
    words) is dropped.  What was dropped is kept under ``ungrounded``."""
    haystack = _squash(corpus)
    controls = _squash(chrome)

    def grounded(text: str) -> bool:
        return all(
            _squash(q) in haystack for q in _QUOTED_RE.findall(text)
        ) if _EDIT_VERB_RE.search(text) else True

    def control_text(value: str) -> bool:
        squashed = _squash(value)
        return bool(squashed) and (
            (squashed in controls and squashed not in haystack)
            or is_chrome_line(value, app)
        )

    dropped: list[str] = []
    actions = []
    for action in diff.get("actions") or []:
        (actions if grounded(str(action)) else dropped).append(action)
    inputs = []
    for item in diff.get("inputs") or []:
        value = str(item.get("value") or "") if isinstance(item, dict) else ""
        if control_text(value):
            dropped.append(f"input {item.get('field', '?')}: {value}")
        else:
            inputs.append(item)
    if not dropped:
        return diff
    out = {**diff, "actions": actions, "inputs": inputs, "ungrounded": dropped}
    summary = str(diff.get("step_description") or "")
    if summary and not grounded(summary):
        out["step_description"] = actions[0] if actions else ""
        dropped.append(summary)
    logger.info("Frame diff: dropped %d edit claim(s) not in the documents: %s",
                len(dropped), "; ".join(dropped))
    return out


def _corpus(annotation: dict, content: list[str]) -> str:
    kt = annotation.get("key_text") or {}
    return "\n".join([
        *content, str(kt.get("typed_text") or ""), str(annotation.get("location") or ""),
        *(str(i.get("label") or "") for i in annotation.get("visible_items") or []
          if isinstance(i, dict)),
    ])


# ---------------------------------------------------------------------------
# FrameDiffer
# ---------------------------------------------------------------------------

class FrameDiffer:
    """Compares consecutive annotation pairs to produce action diffs.

    Designed to run in a dedicated thread within the worker process.
    """

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig()
        self._stats = {
            "diffs_computed": 0,
            "edge_cases": 0,
            "failed": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def diff_pair(
        self,
        prev_event: dict,
        curr_event: dict,
        *,
        allow_model: bool = True,
    ) -> DiffResult:
        """Compute the diff between two consecutive annotated events.

        Returns a DiffResult with the diff dict (may be an edge-case
        marker or an LLM-generated action diff).  With *allow_model*
        False, a pair that needs the model gets ``diff=None`` (the caller
        leaves it pending) instead of a model call.
        """
        curr_id = curr_event.get("id", "unknown")

        # Parse annotations
        prev_ann = _parse_annotation(prev_event)
        curr_ann = _parse_annotation(curr_event)

        if prev_ann is None or curr_ann is None:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker("missing_annotation"),
            )

        # Check for stale-skipped previous frame
        prev_status = prev_event.get("annotation_status", "")
        if prev_status == "skipped":
            self._stats["edge_cases"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_stale_skip_marker("previous_frame_skipped"),
            )

        # Check edge cases (app switch, time gap, no change)
        edge_case = _detect_edge_case(
            prev_event, curr_event, prev_ann, curr_ann, self.config
        )
        if edge_case is not None:
            self._stats["edge_cases"] += 1
            return DiffResult(event_id=curr_id, diff=edge_case)

        if not allow_model:
            return DiffResult(event_id=curr_id, diff=None)

        # --- LLM diff ---
        # Without the font, format and menu controls' words: they are not
        # the user's (``content_lines``).
        frame_a = _format_annotation_for_diff(
            prev_ann, content_lines(prev_event, prev_ann, geometry=False)[0],
        )
        frame_b = _format_annotation_for_diff(
            curr_ann, content_lines(curr_event, curr_ann, geometry=False)[0],
        )

        prompt = DIFF_PROMPT_TEMPLATE.format(frame_a=frame_a, frame_b=frame_b)

        try:
            from agenthandover_worker.scene_annotator import (
                ANNOTATION_KEEP_ALIVE, _call_ollama_vlm,
            )
            from agenthandover_worker.model_profiles import get_profile
            profile = get_profile(self.config.model)

            # The differ runs on the annotation model: the same num_ctx and
            # keep_alive keep Ollama from reloading or unloading it early.
            raw_response, inference_time = _call_ollama_vlm(
                model=self.config.model,
                prompt=prompt,
                host=self.config.ollama_host,
                num_predict=profile.diff_num_predict,
                num_ctx=profile.ann_num_ctx,
                keep_alive=ANNOTATION_KEEP_ALIVE,
                system=profile.diff_system or DIFF_SYSTEM_PROMPT,
                think=profile.diff_think,
                extra_options={
                    k: v for k, v in profile.diff_options().items()
                    if k not in ("num_predict",)
                },
                call_type="diff",
            )
        except ConnectionError as exc:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker(f"ollama_connection: {exc}"),
            )
        except Exception as exc:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker(f"llm_error: {exc}"),
            )

        # Validate
        diff = _validate_diff(raw_response)
        if diff is None:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker("invalid_json"),
                inference_time_seconds=inference_time,
            )

        # A claimed edit must be in the documents (the windows' text
        # without their chrome, located by the frames' geometry).
        content_a, chrome_a = content_lines(prev_event, prev_ann)
        content_b, chrome_b = content_lines(curr_event, curr_ann)
        diff = ground_text_changes(
            diff,
            _corpus(prev_ann, content_a) + "\n" + _corpus(curr_ann, content_b),
            chrome="\n".join(chrome_a + chrome_b),
            app=str(curr_ann.get("app") or ""),
        )
        self._stats["diffs_computed"] += 1
        return DiffResult(
            event_id=curr_id,
            diff=diff,
            inference_time_seconds=inference_time,
        )

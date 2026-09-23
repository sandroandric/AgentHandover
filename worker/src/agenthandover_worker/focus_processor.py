"""Focus session processor for the v2 VLM annotation pipeline.

Orchestrates the end-to-end processing of a completed focus recording
session:  picks keyframes → annotates the SOP's keyframes → ensures frame
diffs exist → collects the annotated timeline → generates a semantic SOP.
The session's other keyframes are annotated after its Skill is written
(:meth:`FocusProcessor.annotate_remaining`); frames that are not keyframes
are never sent to the model.

This replaces the v1 focus processing path (episode_builder → translator
→ scorer → sop_inducer.induce_from_focus_session) when v2 annotation is
enabled.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Callable

from agenthandover_worker.dom import url_host
from agenthandover_worker.event_helpers import (
    display_name,
    focused_field,
    parse_json_field,
    parse_timestamp,
)
from agenthandover_worker.scene_annotator import SceneAnnotator, as_compact, page_for_event
from agenthandover_worker.frame_differ import FrameDiffer
from agenthandover_worker.sop_generator import SOPGenerator, GeneratedSOP

logger = logging.getLogger(__name__)

# Annotation statuses the focus path does not re-annotate.
_ANNOTATION_DONE = ("completed", "skipped", "missing_screenshot")

# How far (seconds) a DOM snapshot or click may lie from a frame to join it.
_DOM_TOLERANCE_SEC = 5.0

# Frames of one session that are ever annotated, and the subset annotated
# before the SOP is generated (the SOP prompt's frame cap,
# ``SOPGeneratorConfig.max_timeline_frames``).
FOCUS_KEYFRAMES = 60
SOP_KEYFRAMES = 20

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _frame_place(event: dict) -> tuple[str, str, str]:
    """(app, window title, URL): a dwell is a run of frames sharing it."""
    window = parse_json_field(event, "window_json")
    metadata = parse_json_field(event, "metadata_json")
    return (
        str(window.get("app_id") or ""),
        str(window.get("title") or ""),
        str(metadata.get("url") or ""),
    )


def _ocr_words(event: dict) -> frozenset[str]:
    """The lowercased words of the daemon's OCR for a frame."""
    ocr = parse_json_field(event, "metadata_json").get("ocr")
    if not isinstance(ocr, dict):
        return frozenset()
    text = ocr.get("full_text")
    if not isinstance(text, str) or not text:
        elements = ocr.get("elements")
        text = " ".join(
            str(el.get("text", "")) for el in elements if isinstance(el, dict)
        ) if isinstance(elements, list) else ""
    return frozenset(_WORD_RE.findall(text.lower()))


def select_keyframes(
    frames: list[dict],
    k: int = FOCUS_KEYFRAMES,
    *,
    click_times: list | tuple = (),
    clipboard_ids: frozenset[str] | set[str] = frozenset(),
) -> list[dict]:
    """The *k* most informative frames of a session, in time order.

    Needs no model: every signal comes from the rows.  The first and last
    frame of every dwell (a run of frames with the same app, title and URL,
    so every title or URL change) rank first, the session's first and last
    frame above all; then frames next to a copy (*clipboard_ids*), frames
    within ``_DOM_TOLERANCE_SEC`` of a click (*click_times*, datetimes),
    frames whose OCR text changed most from the previous frame, and frames
    after a long pause.  When there are more dwell boundaries than *k*,
    those other signals decide among them.  Rows without a screenshot
    (app switches, title changes) cannot be annotated and rank last.
    """
    if len(frames) <= k:
        return list(frames)
    n = len(frames)
    places = [_frame_place(f) for f in frames]
    words = [_ocr_words(f) for f in frames]
    times = [parse_timestamp(f.get("timestamp")) for f in frames]
    slack = timedelta(seconds=_DOM_TOLERANCE_SEC)

    scores: list[float] = []
    for i, frame in enumerate(frames):
        score = 1000.0 if i in (0, n - 1) else 0.0
        if not parse_json_field(frame, "metadata_json").get("screenshot_path"):
            score -= 10_000.0
        if (
            i in (0, n - 1)
            or places[i - 1] != places[i]
            or places[i + 1] != places[i]
        ):
            score += 100.0
        if frame.get("id") in clipboard_ids:
            score += 10.0
        when = times[i]
        if when is not None and any(abs(c - when) <= slack for c in click_times):
            score += 8.0
        if i > 0:
            union = words[i - 1] | words[i]
            if union:
                score += 5.0 * (1.0 - len(words[i - 1] & words[i]) / len(union))
            if when is not None and times[i - 1] is not None:
                gap = (when - times[i - 1]).total_seconds()
                score += 3.0 * min(max(gap, 0.0) / 30.0, 1.0)
        scores.append(score)

    top = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    return [frames[i] for i in sorted(top)]


class FocusProcessor:
    """Process completed focus recording sessions through the v2 pipeline.

    Usage::

        processor = FocusProcessor(annotator, differ, sop_generator)
        result = processor.process_session(db, session_id, title, events,
                                           screenshots_dir=...)
    """

    def __init__(
        self,
        annotator: SceneAnnotator,
        differ: FrameDiffer,
        sop_generator: SOPGenerator,
    ) -> None:
        self.annotator = annotator
        self.differ = differ
        self.sop_generator = sop_generator
        self._last_rich_observations: list[list[dict]] = []
        # Keyframes of the last session that wait for annotate_remaining().
        self._remaining: list[dict] = []
        self._remaining_dir: str | Path = ""
        # Called once the keyframes are annotated, before the SOP call: the
        # worker's switch to the generation model
        # (``ModelScheduler.enter_generation``).  False defers the session.
        self.before_generation: Callable[[], bool] | None = None

    def process_session(
        self,
        db: object,
        session_id: str,
        title: str,
        events: list[dict],
        *,
        screenshots_dir: str | Path = "",
    ) -> GeneratedSOP:
        """Process a completed focus session end-to-end.

        Steps:
          1. Pick the session's keyframes (:func:`select_keyframes`), mark
             the other frames skipped, and annotate the SOP's keyframes
          2. Ensure the annotated SOP keyframes have frame diffs
          3. Collect the full annotated timeline
          4. Generate a semantic SOP via the SOP generator

        Args:
            db: WorkerDB instance.
            session_id: Focus session UUID.
            title: User-provided task name.
            events: Raw event dicts from the DB (already filtered by session_id).
            screenshots_dir: Directory containing screenshot JPEGs.

        Returns:
            GeneratedSOP with the template dict (or error).
        """
        if not events:
            return GeneratedSOP(
                sop={}, success=False, error="No events in focus session"
            )

        logger.info(
            "Focus v2: processing session '%s' (%s) — %d events",
            title, session_id, len(events),
        )

        # Step 0a: Filter out AgentHandover's own events
        events = self._filter_own_events(events)
        if not events:
            return GeneratedSOP(
                sop={}, success=False, error="All events were self-referential (AgentHandover's own UI)"
            )

        # Step 0b: Attach clipboard context to adjacent screenshots
        events = self._attach_clipboard_context(events)

        # Step 0c: Keyframes.  Only these are ever annotated, the SOP's
        # own first; the rest wait until the Skill is written.
        click_times = self._click_times(db, events)
        clipboard_ids = frozenset(self._clipboard_map)
        keyframes = select_keyframes(
            events, FOCUS_KEYFRAMES,
            click_times=click_times, clipboard_ids=clipboard_ids,
        )
        all_frames = events
        events = select_keyframes(
            keyframes, SOP_KEYFRAMES,
            click_times=click_times, clipboard_ids=clipboard_ids,
        )
        sop_ids = {e.get("id") for e in events}
        self._remaining = [e for e in keyframes if e.get("id") not in sop_ids]
        self._remaining_dir = screenshots_dir

        # Step 1: Ensure the SOP keyframes are annotated.  During the transport
        # backoff, don't hit Ollama at all; the session waits untouched.
        needs_annotation = any(
            e.get("annotation_status", "pending") not in _ANNOTATION_DONE
            for e in events
        )
        if needs_annotation and not self.annotator.ready():
            self._remaining = []
            return GeneratedSOP(
                sop={}, success=False, deferred=True,
                error="Annotation deferred: Ollama unavailable",
            )
        self._drop_frames(db, all_frames, keyframes, screenshots_dir)
        ann_stats = self._ensure_annotations(db, events, screenshots_dir)
        logger.info(
            "Focus v2 annotation: %d annotated, %d skipped, %d failed, %d deferred",
            ann_stats["annotated"],
            ann_stats["skipped"],
            ann_stats["failed"],
            ann_stats["deferred"],
        )
        if ann_stats["deferred"]:
            # Ollama is unreachable: keep the session (rows pending,
            # screenshots kept) and retry on a later tick.
            return GeneratedSOP(
                sop={}, success=False, deferred=True,
                error="Annotation deferred: Ollama unavailable",
            )

        # Step 2: Ensure frame diffs exist
        diff_stats = self._ensure_diffs(db, events)
        logger.info(
            "Focus v2 diffs: %d action, %d edge, %d failed",
            diff_stats["diffs"],
            diff_stats["edge_cases"],
            diff_stats["failed"],
        )

        # Step 3: Collect the annotated timeline
        timeline = self._collect_timeline(db, events)
        if not timeline:
            return GeneratedSOP(
                sop={},
                success=False,
                error="No annotated frames produced for focus session",
            )

        logger.info(
            "Focus v2 timeline: %d annotated frames (of %d total events)",
            len(timeline), len(events),
        )

        # Rich per-frame observations for the single behavioral synthesis
        # call that main.py runs after SOP generation.
        self._last_rich_observations = [
            [
                self._build_pre_analysis_obs(f)
                for f in timeline if f.get("annotation")
            ]
        ]

        # Step 4: Generate SOP, on the generation model
        if self.before_generation is not None and not self.before_generation():
            return GeneratedSOP(
                sop={}, success=False, deferred=True,
                error="Generation deferred: the SOP model could not be loaded",
            )
        return self.sop_generator.generate_from_focus(timeline, title)

    @staticmethod
    def _build_pre_analysis_obs(frame: dict) -> dict:
        """Build one observation dict from a timeline frame for the
        behavioral synthesizer.

        Carries every field of the compact annotation plus the frame's
        screen text, so the synthesizer can quote verbatim text (emails,
        URLs, typed content, the focused control, the picked list entry,
        subject lines and message bodies on screen) rather than just
        'user is doing X'.
        """
        ann = as_compact(frame.get("annotation"))
        tc = ann.get("task_context") or {}
        kt = ann.get("key_text") or {}
        ui = ann.get("ui_state") or {}

        obs: dict = {
            "action": tc.get("what_doing", ""),
            "app": ann.get("app", ""),
            "location": ann.get("location", ""),
            "target": ann.get("location", ""),
        }

        # Verbatim grounding — helps the synthesizer write concrete goals
        emails = kt.get("emails") or []
        if emails:
            obs["email_addresses"] = emails
        urls = kt.get("urls") or []
        if urls:
            obs["urls"] = urls
        typed = kt.get("typed_text") or ""
        if typed:
            obs["typed_text"] = typed[:400]

        # UI state — "active_element=Send button" tells the synthesizer
        # the user is about to send something, even when typed_text is
        # empty.
        active_element = ui.get("active_element") or ""
        if active_element:
            obs["active_element"] = active_element[:200]
        modal = ui.get("modal") or ""
        if modal:
            obs["modal"] = modal[:200]

        items = [
            f"{item.get('label')}{' (chosen)' if item.get('chosen') else ''}"
            for item in ann.get("visible_items") or []
            if isinstance(item, dict) and item.get("label")
        ]
        if items:
            obs["visible_items"] = items

        # The screen's own text (the salient OCR lines the annotation
        # prompt carried), where subject lines, message bodies and
        # recipient chips live.
        screen_text = ann.get("_screen_text") or ""
        if screen_text:
            obs["screen_text"] = screen_text[:600]

        return obs

    # ------------------------------------------------------------------
    # Step 0a: Filter own events
    # ------------------------------------------------------------------

    _OWN_APP_PATTERNS = ("agenthandover", "agenthandoverapp", "agenthandover_worker")
    _OWN_TITLE_PATTERNS = ("agenthandover",)

    def _filter_own_events(self, events: list[dict]) -> list[dict]:
        """Remove events generated by AgentHandover's own UI.

        Filters events where the app name, bundle ID, or window title
        references AgentHandover or its sub-processes, so the SOP only
        captures the user's actual task.
        """
        original_count = len(events)
        filtered: list[dict] = []

        for event in events:
            if self._is_own_event(event):
                continue
            filtered.append(event)

        removed = original_count - len(filtered)
        if removed:
            logger.info(
                "Focus v2: filtered %d self-referential event(s) "
                "(%d remaining of %d)",
                removed, len(filtered), original_count,
            )

        return filtered

    def _is_own_event(self, event: dict) -> bool:
        """Check whether an event originates from AgentHandover's own UI."""
        window = parse_json_field(event, "window_json")
        if not window:
            return False

        # ``app_id`` is ``pid:<pid>:<owner name>`` or a bundle id.
        app_id = str(window.get("app_id", "")).lower()
        title = str(window.get("title", "")).lower()

        # Check app name / bundle ID
        for pattern in self._OWN_APP_PATTERNS:
            if pattern in app_id:
                return True

        # Check window title
        for pattern in self._OWN_TITLE_PATTERNS:
            if pattern in title:
                return True

        return False

    # ------------------------------------------------------------------
    # Step 0b: Attach clipboard context
    # ------------------------------------------------------------------

    def _attach_clipboard_context(self, events: list[dict]) -> list[dict]:
        """Attach clipboard metadata to the nearest screenshot event.

        ClipboardChange events have no screenshots and cannot be annotated
        by the VLM.  This method:
        1. Identifies ClipboardChange events via ``kind_json``.
        2. For each, finds the snapshot immediately BEFORE the clipboard
           event — this is the frame the user was looking at when they
           pressed Cmd+C, so it's the SOURCE of the copy.  Falls back to
           the snapshot AFTER if no BEFORE exists (very rare).
        3. Stores clipboard info keyed by the target event's ID in
           ``self._clipboard_map``.
        4. Removes ClipboardChange events from the list.

        Historical bug (caught 2026-04-10): the matcher preferred the
        snapshot AFTER the clipboard event.  When a user copied from HN
        and switched to Gmail before the next snapshot fired, the COPY
        marker landed on a Gmail frame and the SOP generator wrote
        ``Copy from Gmail`` instead of ``Copy from Hacker News``.

        Returns the cleaned event list (DwellSnapshots only).
        """
        self._clipboard_map: dict[str, dict] = {}

        # Separate clipboard events from the rest
        clipboard_events: list[tuple[int, dict]] = []
        non_clipboard: list[tuple[int, dict]] = []

        for i, event in enumerate(events):
            kind = self._extract_event_kind(event)
            if kind == "ClipboardChange":
                clipboard_events.append((i, event))
            else:
                non_clipboard.append((i, event))

        if not clipboard_events:
            return events  # nothing to do

        # Build index of DwellSnapshot positions for quick lookup
        dwell_positions: list[tuple[int, str]] = []
        for orig_idx, evt in non_clipboard:
            kind = self._extract_event_kind(evt)
            if kind == "DwellSnapshot":
                dwell_positions.append((orig_idx, evt.get("id", "")))

        for clip_idx, clip_event in clipboard_events:
            clip_meta = self._extract_clipboard_meta(clip_event)
            if not clip_meta:
                continue

            # Find the closest DwellSnapshot BEFORE the clipboard event —
            # that's where the user actually pressed Cmd+C.
            best_before: tuple[int, str] | None = None  # (idx, dwell_id)
            best_after: tuple[int, str] | None = None
            for dwell_orig_idx, dwell_id in dwell_positions:
                if dwell_orig_idx < clip_idx:
                    if best_before is None or dwell_orig_idx > best_before[0]:
                        best_before = (dwell_orig_idx, dwell_id)
                elif dwell_orig_idx > clip_idx:
                    if best_after is None or dwell_orig_idx < best_after[0]:
                        best_after = (dwell_orig_idx, dwell_id)

            # Prefer BEFORE (the actual copy source); fall back to AFTER
            target = best_before or best_after
            if target is None:
                continue
            target_id = target[1]

            # Look up the source frame's app/location so we can include
            # them in the COPY marker — gives the SOP generator the
            # context it needs to attribute the action correctly.
            source_app = ""
            source_location = ""
            for orig_idx, evt in non_clipboard:
                if evt.get("id", "") == target_id:
                    win = parse_json_field(evt, "window_json")
                    source_app = display_name(win.get("app_id"))
                    source_location = str(win.get("title") or "")
                    break
            if source_app:
                clip_meta["source_app"] = source_app
            if source_location:
                clip_meta["source_location"] = source_location

            self._clipboard_map[target_id] = clip_meta

        logger.info(
            "Focus v2: attached %d clipboard context(s) to snapshots "
            "(prefer BEFORE = source of copy), "
            "removed %d ClipboardChange event(s)",
            len(self._clipboard_map), len(clipboard_events),
        )

        # Return only non-clipboard events (preserving order)
        return [evt for _, evt in non_clipboard]

    @staticmethod
    def _extract_event_kind(event: dict) -> str:
        """Extract the event kind name from ``kind_json``.

        kind_json is serialized as ``{"type": "DwellSnapshot", ...}`` or
        ``{"type": "ClipboardChange", ...}``. We need the VALUE of ``type``,
        not the first key.
        """
        kind_json = event.get("kind_json", "")
        if not kind_json:
            return ""
        try:
            parsed = (
                json.loads(kind_json)
                if isinstance(kind_json, str)
                else kind_json
            )
        except (json.JSONDecodeError, TypeError):
            return ""
        if isinstance(parsed, dict) and "type" in parsed:
            return str(parsed.get("type", ""))
        return ""

    @staticmethod
    def _extract_clipboard_meta(event: dict) -> dict | None:
        """Extract clipboard metadata (content_types, byte_size, preview) from an event.

        ClipboardChange events serialize the payload inside ``kind_json``,
        not ``metadata_json``.  The daemon writes it as::

            kind_json = {
                "type": "ClipboardChange",
                "content_types": [...],
                "byte_size": N,
                "high_entropy": bool,
                "content_hash": "...",
                "content_preview": "...",   # optional
            }

        This was historically read from ``metadata_json``, which is always
        ``{}`` for clipboard events — so clipboard context was never
        attached to focus SOPs.
        """
        kind_json = event.get("kind_json", "")
        if not kind_json:
            return None
        try:
            parsed = (
                json.loads(kind_json)
                if isinstance(kind_json, str)
                else kind_json
            )
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        content_types = parsed.get("content_types", [])
        byte_size = parsed.get("byte_size", 0)
        if not content_types and not byte_size:
            return None
        meta: dict = {"content_types": content_types, "byte_size": byte_size}
        # Include extra fields that the SOP generator may use for context
        if parsed.get("high_entropy") is not None:
            meta["high_entropy"] = parsed["high_entropy"]
        if parsed.get("content_hash"):
            meta["content_hash"] = parsed["content_hash"]
        # The copied text itself (present when the daemon's
        # ``enable_clipboard_preview`` is on): what the Copy/Paste steps
        # must name.
        if parsed.get("content_preview"):
            meta["content_preview"] = str(parsed["content_preview"])
        return meta

    # ------------------------------------------------------------------
    # Step 0c: Keyframes
    # ------------------------------------------------------------------

    @staticmethod
    def _click_times(db: object, events: list[dict]) -> list:
        """When the session's Chrome clicks happened (datetimes)."""
        times = [
            when for e in events
            if (when := parse_timestamp(e.get("timestamp"))) is not None
        ]
        if not times:
            return []
        slack = timedelta(seconds=_DOM_TOLERANCE_SEC)
        clicks = db.get_click_intents(  # type: ignore[union-attr]
            (min(times) - slack).isoformat(), (max(times) + slack).isoformat(),
        )
        return [
            when for c in clicks
            if (when := parse_timestamp(c.get("timestamp"))) is not None
        ]

    def _drop_frames(
        self,
        db: object,
        events: list[dict],
        keyframes: list[dict],
        screenshots_dir: str | Path,
    ) -> None:
        """Mark pending frames that are not keyframes as skipped and delete
        their JPEGs: they are never sent to the model."""
        keep = {e.get("id") for e in keyframes}
        dropped = 0
        for event in events:
            if event.get("id") in keep:
                continue
            if event.get("annotation_status", "pending") in _ANNOTATION_DONE:
                continue
            db.save_annotation(  # type: ignore[union-attr]
                event.get("id", ""), "", status="skipped",
            )
            self.annotator.discard_screenshot(event, screenshots_dir)
            event["annotation_status"] = "skipped"
            dropped += 1
        if dropped:
            logger.info(
                "Focus v2: %d keyframe(s) of %d frames; %d skipped",
                len(keyframes), len(events), dropped,
            )

    def annotate_remaining(self, db: object) -> dict:
        """Annotate the last session's keyframes that the SOP did not need.

        Called in the annotation phase of the tick after the session's
        Skill is written (or its generation has failed), so they never
        delay it and never sit between two generation calls.  Frames Ollama
        cannot take now stay pending with their screenshots.
        """
        rest, self._remaining = self._remaining, []
        if not rest or not self.annotator.ready():
            return {
                "annotated": 0, "skipped": 0, "failed": 0,
                "already_done": 0, "deferred": len(rest),
            }
        stats = self._ensure_annotations(db, rest, self._remaining_dir)
        logger.info(
            "Focus v2: %d remaining keyframe(s) annotated after the Skill",
            stats["annotated"],
        )
        return stats

    # ------------------------------------------------------------------
    # Step 1: Annotation
    # ------------------------------------------------------------------

    def _ensure_annotations(
        self,
        db: object,
        events: list[dict],
        screenshots_dir: str | Path,
    ) -> dict:
        """Annotate any un-annotated events in the focus session.

        Focus events are processed with priority — we annotate them
        immediately rather than waiting for the background loop.
        """
        stats = {
            "annotated": 0, "skipped": 0, "failed": 0,
            "already_done": 0, "deferred": 0,
        }

        for event in events:
            event_id = event.get("id", "unknown")
            status = event.get("annotation_status", "pending")

            # Already annotated
            if status in _ANNOTATION_DONE:
                stats["already_done"] += 1
                continue

            # Get sliding window context from recent annotations
            timestamp = event.get("timestamp", "")
            recent = db.get_recent_annotations(  # type: ignore[union-attr]
                before_timestamp=timestamp,
                limit=self.annotator.config.sliding_window_size,
                max_age_seconds=self.annotator.config.sliding_window_max_age_sec,
            )

            result = self.annotator.annotate_event(
                event,
                recent_annotations=recent,
                artifact_dir=str(screenshots_dir),
                page=page_for_event(db, event),
            )

            if result.status == "completed" and result.annotation:
                db.save_annotation(  # type: ignore[union-attr]
                    event_id,
                    json.dumps(result.annotation),
                    status="completed",
                )
                stats["annotated"] += 1
            elif result.status in ("skipped", "missing_screenshot"):
                db.save_annotation(  # type: ignore[union-attr]
                    event_id, "", status=result.status
                )
                stats["skipped"] += 1
            elif result.status == "deferred":
                # Ollama unreachable: leave this and the remaining rows
                # pending with their screenshots.
                stats["deferred"] += 1
                break
            else:
                db.save_annotation(  # type: ignore[union-attr]
                    event_id, "", status="failed"
                )
                stats["failed"] += 1

        return stats

    # ------------------------------------------------------------------
    # Step 2: Frame diffs
    # ------------------------------------------------------------------

    def _ensure_diffs(self, db: object, events: list[dict]) -> dict:
        """Compute frame diffs for annotated events that lack them."""
        stats = {"diffs": 0, "edge_cases": 0, "failed": 0, "already_done": 0}

        # Re-read events to get fresh annotation data
        annotated_events = []
        for event in events:
            event_id = event.get("id", "unknown")
            # Refresh from DB to get annotation
            fresh = db.get_event_by_id(event_id)  # type: ignore[union-attr]
            if fresh and fresh.get("annotation_status") == "completed":
                annotated_events.append(fresh)

        for i, event in enumerate(annotated_events):
            event_id = event.get("id", "unknown")

            # Already has diff
            if event.get("frame_diff_json"):
                stats["already_done"] += 1
                continue

            timestamp = event.get("timestamp", "")

            # Get previous annotated event
            if i == 0:
                # First frame in focus session — always use first_frame marker.
                # Do NOT look up a DB predecessor: it may come from unrelated
                # pre-session activity and would pollute the first step's diff.
                prev = None
            else:
                prev = annotated_events[i - 1]

            if prev is None:
                db.save_frame_diff(  # type: ignore[union-attr]
                    event_id, json.dumps({"diff_type": "first_frame"})
                )
                stats["edge_cases"] += 1
                continue

            result = self.differ.diff_pair(prev, event)
            db.save_frame_diff(  # type: ignore[union-attr]
                event_id, json.dumps(result.diff)
            )

            diff_type = result.diff.get("diff_type", "unknown")
            if diff_type == "action":
                stats["diffs"] += 1
            elif diff_type == "diff_failed":
                stats["failed"] += 1
            else:
                stats["edge_cases"] += 1

        return stats

    # ------------------------------------------------------------------
    # Step 3: Collect timeline
    # ------------------------------------------------------------------

    def _collect_timeline(
        self,
        db: object,
        events: list[dict],
    ) -> list[dict]:
        """Collect the annotated timeline for SOP generation.

        Returns a list of dicts, each containing:
        - annotation: parsed scene annotation dict
        - diff: parsed frame diff dict (or None)
        - timestamp: ISO timestamp string
        - app: application name
        - event_id: the frame's event id (the step evidence points at it)
        - host, dom_nodes, dom_snapshot_id: the DOM snapshot joined by host
        - click_targets: the Chrome clicks made on this frame's page
        - clipboard_context: the copy made from this frame, if any
        """
        timeline = []

        for event in events:
            event_id = event.get("id", "unknown")
            # Re-read to get latest annotation + diff data
            fresh = db.get_event_by_id(event_id)  # type: ignore[union-attr]
            if not fresh or fresh.get("annotation_status") != "completed":
                continue

            ann_json = fresh.get("scene_annotation_json", "")
            if not ann_json:
                continue

            try:
                annotation = json.loads(ann_json)
            except (json.JSONDecodeError, TypeError):
                continue

            diff = None
            diff_json = fresh.get("frame_diff_json", "")
            if diff_json:
                try:
                    diff = json.loads(diff_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            # The Chrome extension's DOM snapshot closest to this frame on
            # the same host.  The annotation's location may carry a scheme
            # or not ("news.ycombinator.com/item?id=1").
            dom_nodes = None
            dom_snapshot_id = None
            host = url_host(annotation.get("location", ""))
            if host:
                dom_snaps = db.get_dom_snapshots_near_timestamp(  # type: ignore[union-attr]
                    fresh.get("timestamp", ""),
                    host,
                    tolerance_sec=_DOM_TOLERANCE_SEC,
                )
                if dom_snaps:
                    dom_nodes = dom_snaps[0].get("nodes")
                    dom_snapshot_id = dom_snaps[0].get("event_id")

            # Look up clipboard context attached in Step 0b
            clipboard_ctx = (
                self._clipboard_map.get(event_id)
                if hasattr(self, "_clipboard_map")
                else None
            )

            timeline.append({
                "annotation": annotation,
                "diff": diff,
                "dom_nodes": dom_nodes,
                "dom_snapshot_id": dom_snapshot_id,
                "host": host,
                "click_targets": [],
                "timestamp": fresh.get("timestamp", ""),
                "app": annotation.get("app", ""),
                "event_id": event_id,
                "clipboard_context": clipboard_ctx,
                "focused_field": focused_field(fresh),
                "shell_commands": [],
            })

        self._attach_clicks(db, timeline)
        self._attach_shell_commands(db, timeline)

        # Safety net: the first frame in a focus session must never carry a
        # diff — it is the clean starting point of the SOP.  Any diff stored
        # for it (e.g. a "first_frame" marker or a stale pre-session diff)
        # would pollute the first step with irrelevant context.
        if timeline:
            timeline[0]["diff"] = None

        return timeline

    @staticmethod
    def _attach_shell_commands(db: object, timeline: list[dict]) -> None:
        """Give each frame the shell commands run since the previous frame.

        ``ShellCommand`` events carry no ``focus_session_id`` (the daemon
        ingests the hook's log in batches), so they are found by time: from
        the session's first frame (less the DOM tolerance, for a command
        typed just before the first capture) to its last frame.  A command
        belongs to the first frame at or after it, the screen that shows
        its result.
        """
        timed = [
            (when, frame) for frame in timeline
            if (when := parse_timestamp(frame.get("timestamp"))) is not None
        ]
        if not timed:
            return
        slack = timedelta(seconds=_DOM_TOLERANCE_SEC)
        commands = db.get_shell_commands(  # type: ignore[union-attr]
            (timed[0][0] - slack).isoformat(), timed[-1][0].isoformat(),
        )
        for command in commands:
            when = parse_timestamp(command.get("timestamp"))
            if when is None:
                continue
            owner = next((frame for t, frame in timed if t >= when), None)
            if owner is not None:
                owner["shell_commands"].append(command)

    @staticmethod
    def _attach_clicks(db: object, timeline: list[dict]) -> None:
        """Give each frame the Chrome clicks made on its page.

        Extension events carry no ``focus_session_id``, so they are found by
        time.  A click belongs to the last frame at or before it when that
        frame shows the click's host (the page the user clicked on);
        otherwise, when the page came on screen with the click itself (an
        app switch into Chrome), to the first frame after it on that host
        within the DOM tolerance.  A click no frame shows is left out.
        """
        timed = [
            (when, frame) for frame in timeline
            if (when := parse_timestamp(frame.get("timestamp"))) is not None
        ]
        if not timed:
            return
        slack = timedelta(seconds=_DOM_TOLERANCE_SEC)
        clicks = db.get_click_intents(  # type: ignore[union-attr]
            (timed[0][0] - slack).isoformat(), (timed[-1][0] + slack).isoformat(),
        )
        for click in clicks:
            when = parse_timestamp(click.get("timestamp"))
            host = url_host(click.get("url"))
            if when is None or not host:
                continue
            before = [frame for t, frame in timed if t <= when]
            owner = before[-1] if before and before[-1].get("host") == host else None
            if owner is None:
                owner = next(
                    (frame for t, frame in timed
                     if when < t <= when + slack and frame.get("host") == host),
                    None,
                )
            if owner is not None:
                owner["click_targets"].append(click)

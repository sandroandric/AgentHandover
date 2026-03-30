"""Episode Builder v1 — cluster events into episodes by app/URL/entities.

.. deprecated:: 0.2.0
    This module is part of the v1 pipeline (heuristic-based episode
    construction).  It is superseded by ``task_segmenter.py`` in the v2
    VLM-based pipeline, which uses semantic embeddings and VLM annotations
    to identify task boundaries instead of app/URL/timing heuristics.

    The v1 pipeline (episode_builder → translator → sop_inducer →
    sop_enhancer) remains functional for backward compatibility but will
    not receive new features.  Use v2 (scene_annotator → frame_differ →
    task_segmenter → sop_generator) for new deployments.

Implements section 8 of the AgentHandover spec: thread-multiplexed episode
construction with soft (time) and hard (event count) caps.

Thread Multiplexing Strategy:
- Cluster events by window/app identity (``app_id``) and URL domain
- Events with the same thread_id go to the same episode unless a cap is hit
- When a cap is exceeded the episode is split into linked segments

Episode Caps:
- Soft cap: 15 minutes duration — preferred split point
- Hard cap: 200 events — absolute maximum per segment
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from agenthandover_worker.clipboard_linker import ClipboardLink, ClipboardLinker

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A contiguous segment of events sharing a common thread."""

    episode_id: str
    segment_id: int = 0
    prev_segment_id: int | None = None
    thread_id: str = ""
    events: list[dict] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict = field(default_factory=dict)
    clipboard_links: list[ClipboardLink] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        """Elapsed wall-clock minutes between first and last event."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60.0
        return 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    def is_over_soft_cap(self) -> bool:
        """True when duration >= soft cap (15 min default)."""
        return self.duration_minutes >= 15.0

    def is_over_hard_cap(self) -> bool:
        """True when event count >= hard cap (200 default)."""
        return self.event_count >= 200

    def should_split(self) -> bool:
        """True when either cap is exceeded."""
        return self.is_over_soft_cap() or self.is_over_hard_cap()


class EpisodeBuilder:
    """Build episodes from a chronological stream of events.

    Parameters
    ----------
    soft_cap_minutes:
        Duration threshold in minutes that triggers a segment split.
    hard_cap_events:
        Maximum number of events in a single segment.
    """

    def __init__(
        self,
        soft_cap_minutes: float = 15.0,
        hard_cap_events: int = 200,
        clipboard_linker: ClipboardLinker | None = None,
    ) -> None:
        self.soft_cap_minutes = soft_cap_minutes
        self.hard_cap_events = hard_cap_events
        self._clipboard_linker = clipboard_linker or ClipboardLinker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_events(self, events: list[dict]) -> list[Episode]:
        """Process a batch of events and return completed episodes.

        Events are grouped by *thread_id* (derived from ``app_id`` and
        URL domain).  Within each thread the episode is split when the
        soft cap (duration) or hard cap (event count) is exceeded.
        """
        if not events:
            return []

        # Map thread_id -> current open Episode
        open_episodes: dict[str, Episode] = {}
        # Collect all completed (and still-open) episodes in order
        completed: list[Episode] = []

        for event in events:
            thread_id = self._get_thread_id(event)
            ts = self._parse_timestamp(event)

            current = open_episodes.get(thread_id)

            if current is None:
                # First event for this thread
                episode = self._new_episode(thread_id=thread_id)
                self._add_event(episode, event, ts)
                open_episodes[thread_id] = episode
            elif self._should_start_new_segment(event, current, ts):
                # Cap exceeded — finalise current, start a new segment
                completed.append(current)
                new_seg = self._split_episode(current)
                self._add_event(new_seg, event, ts)
                open_episodes[thread_id] = new_seg
            else:
                self._add_event(current, event, ts)

        # Flush remaining open episodes
        for ep in open_episodes.values():
            completed.append(ep)

        # Sort by start_time so output is deterministic
        completed.sort(key=lambda e: e.start_time or datetime.min.replace(tzinfo=timezone.utc))

        # Annotate episodes with clipboard copy-paste links
        self._annotate_clipboard_links(completed)

        return completed

    def _annotate_clipboard_links(self, episodes: list[Episode]) -> None:
        """Find clipboard copy-paste links across all episodes and annotate."""
        all_events: list[dict] = []
        for ep in episodes:
            all_events.extend(ep.events)

        if not all_events:
            return

        links = self._clipboard_linker.find_links(all_events)
        if not links:
            return

        # Build event_id -> episode mapping
        event_to_episode: dict[str, Episode] = {}
        for ep in episodes:
            for ev in ep.events:
                eid = ev.get("id", "")
                if eid:
                    event_to_episode[eid] = ep

        # Assign links to episodes containing the paste event.
        # Verify episode_id match to avoid wrong assignment on event_id collision.
        for link in links:
            paste_ep = event_to_episode.get(link.paste_event_id)
            copy_ep = event_to_episode.get(link.copy_event_id)
            if paste_ep is not None:
                # Only assign if the copy event is in the same episode or
                # no copy episode is found (cross-episode link is valid)
                if copy_ep is None or copy_ep.episode_id == paste_ep.episode_id:
                    paste_ep.clipboard_links.append(link)

    # ------------------------------------------------------------------
    # Thread identification
    # ------------------------------------------------------------------

    def _get_thread_id(self, event: dict) -> str:
        """Determine thread ID from event's app_id, window_id, URL domain, and entities.

        Entity-based clustering signals (extracted from window titles and URLs):
        - Ticket IDs: JIRA-123, PROJ-456, #789 patterns
        - Filenames: "document.pdf - Preview" patterns

        Thread ID format includes window_id to differentiate same-domain tabs:
        - ``{app_id}:{window_id}:{url_domain}:{entity}`` when all present
        - ``{app_id}:{window_id}:{url_domain}`` when URL present but no entity
        - ``{app_id}:{window_id}`` when window_id but no URL/entity
        - ``{app_id}`` when no URL or entity found
        - ``unknown`` when no app_id can be extracted
        """
        app_id = self._extract_app_id(event)
        url_domain = self._extract_url_domain(event)
        entity = self._extract_entity(event)
        window_id = self._extract_window_id(event)

        if not app_id:
            return "unknown"

        parts = [app_id]
        if window_id:
            parts.append(window_id)
        if url_domain:
            parts.append(url_domain)
        if entity:
            parts.append(entity)

        return ":".join(parts)

    def _extract_app_id(self, event: dict) -> str:
        """Extract app_id from the event's window_json field."""
        window_json = event.get("window_json")
        if not window_json:
            return ""

        try:
            window = json.loads(window_json) if isinstance(window_json, str) else window_json
        except (json.JSONDecodeError, TypeError):
            return ""

        return window.get("app_id", "")

    def _extract_window_id(self, event: dict) -> str:
        """Extract window_id from the event's window_json field."""
        window_json = event.get("window_json")
        if not window_json:
            return ""

        try:
            window = json.loads(window_json) if isinstance(window_json, str) else window_json
        except (json.JSONDecodeError, TypeError):
            return ""

        wid = window.get("window_id", "")
        return str(wid) if wid else ""

    def _extract_url_domain(self, event: dict) -> str:
        """Extract URL domain from the event's metadata_json field."""
        metadata_json = event.get("metadata_json")
        if not metadata_json:
            return ""

        try:
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except (json.JSONDecodeError, TypeError):
            return ""

        url = metadata.get("url", "")
        if not url:
            return ""

        try:
            parsed = urlparse(url)
            return parsed.netloc or ""
        except Exception:
            return ""

    # Regex patterns for entity extraction
    _TICKET_PATTERN = re.compile(r"([A-Z][A-Z0-9]+-\d+)")
    _ISSUE_PATTERN = re.compile(r"#(\d+)")
    _FILENAME_PATTERN = re.compile(r"([\w.-]+\.\w{1,10})\s*[-\u2014\u2013]")

    def _extract_entity(self, event: dict) -> str:
        """Extract entity identifiers from window title and URL.

        Looks for:
        - Ticket IDs like JIRA-123, PROJ-456
        - Issue numbers like #789
        - Filenames like document.pdf from window titles
        """
        window_json = event.get("window_json")
        title = ""
        if window_json:
            try:
                window = json.loads(window_json) if isinstance(window_json, str) else window_json
                title = window.get("title", "")
            except (json.JSONDecodeError, TypeError):
                pass

        metadata_json = event.get("metadata_json")
        url = ""
        if metadata_json:
            try:
                metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                url = metadata.get("url", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # Check both title and URL for entities
        combined = f"{title} {url}"

        # Try ticket ID first (highest priority)
        match = self._TICKET_PATTERN.search(combined)
        if match:
            return match.group(1)

        # Try issue number from URL (e.g. github.com/repo/issues/123)
        match = self._ISSUE_PATTERN.search(combined)
        if match:
            return f"#{match.group(1)}"

        # Try filename from window title
        if title:
            match = self._FILENAME_PATTERN.search(title)
            if match:
                return match.group(1)

        return ""

    # ------------------------------------------------------------------
    # Splitting logic
    # ------------------------------------------------------------------

    def _should_start_new_segment(
        self,
        event: dict,
        current: Episode,
        event_ts: datetime | None,
    ) -> bool:
        """Check if the current episode should be split before adding *event*.

        A split occurs when:
        - The event count would reach the hard cap, OR
        - The duration would reach the soft cap
        """
        # Hard cap: would this event push us to the limit?
        if current.event_count >= self.hard_cap_events:
            return True

        # Soft cap: would this event push duration past the threshold?
        if event_ts and current.start_time:
            prospective_minutes = (event_ts - current.start_time).total_seconds() / 60.0
            if prospective_minutes >= self.soft_cap_minutes:
                return True

        return False

    def _split_episode(self, current: Episode) -> Episode:
        """Create a new segment linked to *current*.

        Sets ``metadata["continuation_of"]`` to the previous episode's
        unique segment identifier so downstream consumers can reconstruct
        the full episode chain.  Carries forward clipboard_links and
        relevant metadata for continuity.
        """
        prev_id = f"{current.episode_id}:seg{current.segment_id}"
        new_segment = Episode(
            episode_id=current.episode_id,
            segment_id=current.segment_id + 1,
            prev_segment_id=current.segment_id,
            thread_id=current.thread_id,
            metadata={"continuation_of": prev_id},
            clipboard_links=list(current.clipboard_links),
        )
        return new_segment

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_episode(self, thread_id: str) -> Episode:
        """Create a brand-new episode (segment 0)."""
        return Episode(
            episode_id=str(uuid.uuid4()),
            segment_id=0,
            prev_segment_id=None,
            thread_id=thread_id,
        )

    def _add_event(
        self,
        episode: Episode,
        event: dict,
        ts: datetime | None,
    ) -> None:
        """Append *event* to *episode* and update time bookkeeping."""
        episode.events.append(event)
        if ts:
            if episode.start_time is None:
                episode.start_time = ts
            episode.end_time = ts

    @staticmethod
    def _parse_timestamp(event: dict) -> datetime | None:
        """Parse the ISO 8601 timestamp from an event dict."""
        raw = event.get("timestamp")
        if not raw:
            return None

        try:
            # Handle the 'Z' suffix and various ISO formats
            if isinstance(raw, str):
                raw = raw.replace("Z", "+00:00")
                return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return None
        return None

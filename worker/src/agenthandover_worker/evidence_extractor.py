"""Pre-expiry evidence extraction for AgentHandover.

Before raw annotations expire at 14 days, this module extracts everything
valuable and stores it permanently on the procedure:

- Content patterns: what the user typed, pasted, or created
- URL patterns: navigation patterns across sessions
- Timing patterns: dwell times, phase durations
- Selection signals: what the user engaged with vs skipped

Runs as a daily batch job alongside the staleness check.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB
    from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class EvidenceExtractor:
    """Extract valuable evidence from raw events before they expire.

    For each non-archived procedure, queries raw events that fall within
    the procedure's observation time windows, extracts content, selection
    signals, timing patterns, and URL patterns, then stores the extracted
    data on the procedure's evidence section.

    Args:
        kb: Knowledge base for reading/writing procedures.
        db: Worker database for querying raw events.
    """

    def __init__(self, kb: "KnowledgeBase", db: "WorkerDB") -> None:
        self._kb = kb
        self._db = db

    def extract_for_procedure(self, slug: str) -> dict:
        """Extract evidence for a single procedure before expiry.

        Loads all raw events that match the procedure's observation
        time windows, extracts patterns, and updates the procedure.

        Args:
            slug: Procedure identifier.

        Returns:
            Dict of extracted evidence (empty if nothing found).
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return {}

        evidence = proc.get("evidence", {})
        observations = evidence.get("observations", [])

        if not observations:
            return {}

        # Build a filter set from the procedure's known apps, domains,
        # and step keywords so we only collect events relevant to THIS
        # procedure — not unrelated same-app/same-domain activity.
        proc_apps = {a.lower() for a in proc.get("apps_involved", []) if a}
        proc_domains: set[str] = set()
        proc_keywords: set[str] = set()
        for step in proc.get("steps", []):
            loc = step.get("location", "")
            if loc:
                domain = _extract_domain(loc)
                if domain:
                    proc_domains.add(domain)
            # Collect keywords from step actions for what_doing matching
            action = step.get("action", step.get("step", ""))
            if action:
                for word in action.lower().split():
                    if len(word) > 3:  # skip short words
                        proc_keywords.add(word)
        # Also add keywords from procedure title/description
        for field in ("title", "description"):
            text = proc.get(field, "")
            if text:
                for word in text.lower().split():
                    if len(word) > 3:
                        proc_keywords.add(word)

        # Collect events from all observation windows
        all_events: list[dict] = []
        seen_event_ids: set[str] = set()
        for obs in observations:
            start_raw = obs.get("timestamp", obs.get("date", ""))
            if not start_raw:
                continue

            # Build time window: full day containing the observation
            try:
                if "T" in start_raw:
                    date_part = start_raw[:10]
                else:
                    date_part = start_raw[:10]
                window_start = f"{date_part}T00:00:00Z"
                window_end = f"{date_part}T23:59:59Z"
                events = self._db.get_events_for_procedure_window(
                    window_start, window_end,
                )
                # Deduplicate and filter to procedure-relevant events
                for ev in events:
                    eid = ev.get("event_id", ev.get("id", ""))
                    if eid and eid in seen_event_ids:
                        continue
                    if eid:
                        seen_event_ids.add(eid)

                    # Filter: only include events relevant to this procedure.
                    # Check app/domain match AND keyword overlap to exclude
                    # same-app noise (e.g., Chrome browsing news vs Chrome
                    # doing the actual workflow).
                    if proc_apps or proc_domains or proc_keywords:
                        ann = _parse_json_field(ev.get("scene_annotation_json"))
                        if ann and isinstance(ann, dict):
                            ev_app = (ann.get("app", "") or "").lower()
                            ev_loc = ann.get("location", "") or ""
                            ev_domain = _extract_domain(ev_loc)
                            ev_what = (
                                ann.get("task_context", {}).get("what_doing", "")
                                or ""
                            ).lower()

                            app_match = any(
                                pa in ev_app or ev_app in pa
                                for pa in proc_apps
                            ) if proc_apps else False
                            domain_match = (
                                ev_domain in proc_domains
                            ) if proc_domains and ev_domain else False

                            # For broad-surface apps (browsers, terminals),
                            # require keyword overlap too — app match alone
                            # isn't enough to confirm task relevance.
                            keyword_match = False
                            if proc_keywords and ev_what:
                                ev_words = {
                                    w for w in ev_what.split() if len(w) > 3
                                }
                                overlap = ev_words & proc_keywords
                                keyword_match = len(overlap) >= 1

                            # Accept if: (app OR domain matches) AND keywords
                            # match.  Both app and domain are broad surfaces
                            # (Chrome serves all sites, reddit.com serves all
                            # subreddits) so keyword overlap is always required
                            # when keywords are available.
                            if not proc_apps and not proc_domains:
                                pass  # No filter data, include everything
                            elif (app_match or domain_match) and keyword_match:
                                pass  # Surface + keyword confirms relevance
                            elif (app_match or domain_match) and not proc_keywords:
                                pass  # No keywords to check, surface is enough
                            else:
                                continue  # Skip unrelated event

                    all_events.append(ev)
            except Exception:
                logger.debug(
                    "Failed to query events for observation of '%s'",
                    slug, exc_info=True,
                )

        if not all_events:
            return {}

        # Extract patterns
        content = self.extract_content_produced(all_events)
        selection = self.extract_selection_signals(all_events)
        urls = self._extract_url_patterns(all_events)
        timing = self._extract_timing_patterns(all_events)

        extracted = {
            "content_produced": content,
            "selection_signals": selection,
            "url_patterns": urls,
            "timing_patterns": timing,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(all_events),
        }

        # Store on the procedure
        evidence["extracted_evidence"] = extracted
        proc["evidence"] = evidence
        self._kb.save_procedure(proc)

        logger.info(
            "Extracted evidence for '%s': %d events, %d content items, "
            "%d selection signals, %d URL patterns",
            slug,
            len(all_events),
            len(content),
            len(selection),
            len(urls),
        )

        return extracted

    def extract_all_pending(self, max_age_days: int = 12) -> int:
        """Extract evidence for all procedures with observations nearing expiry.

        Runs as part of the daily batch.  Targets procedures whose oldest
        un-extracted observation is within ``max_age_days`` of the 14-day
        expiry window.

        Args:
            max_age_days: Process observations older than this many days.
                Default 12 means we extract 2 days before the 14-day expiry.

        Returns:
            Number of procedures processed.
        """
        processed = 0
        for proc_summary in self._kb.list_procedures():
            slug = proc_summary.get("id", proc_summary.get("slug", ""))
            if not slug:
                continue

            proc = self._kb.get_procedure(slug)
            if proc is None:
                continue

            # Skip archived procedures
            if proc.get("lifecycle_state") == "archived":
                continue

            # Skip if extracted after the latest observation
            evidence = proc.get("evidence", {})
            extracted = evidence.get("extracted_evidence", {})
            extracted_at = extracted.get("extracted_at", "")
            if extracted_at:
                # Re-extract if there are observations newer than last extraction
                observations = evidence.get("observations", [])
                newest_obs = ""
                for obs in observations:
                    ts = obs.get("timestamp", obs.get("date", ""))
                    if ts > newest_obs:
                        newest_obs = ts
                if newest_obs and newest_obs <= extracted_at:
                    continue  # Already up-to-date

            # Check if there are observations to extract from
            observations = evidence.get("observations", [])
            if not observations:
                continue

            try:
                result = self.extract_for_procedure(slug)
                if result:
                    processed += 1
            except Exception:
                logger.debug(
                    "Evidence extraction failed for '%s'",
                    slug, exc_info=True,
                )

        return processed

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def extract_content_produced(self, events: list[dict]) -> list[dict]:
        """Extract content that the user produced during this workflow.

        Looks at:
        - Clipboard events (copy/paste)
        - Text input detected in frame diffs
        - DOM snapshots showing user-entered text

        Args:
            events: Raw event dicts from the database.

        Returns:
            List of content dicts with type, value preview, and context.
        """
        content: list[dict] = []

        for ev in events:
            # Check for clipboard data
            event_type = ev.get("event_type", "")
            if event_type == "ClipboardChange":
                meta = _parse_json_field(ev.get("metadata_json"))
                if meta:
                    byte_size = meta.get("byte_size", 0)
                    content_types = meta.get("content_types", [])
                    content.append({
                        "type": "clipboard",
                        "content_types": content_types,
                        "byte_size": byte_size,
                        "timestamp": ev.get("timestamp", ""),
                    })

            # Check frame diffs for text input
            diff_raw = ev.get("frame_diff_json")
            if diff_raw:
                diff = _parse_json_field(diff_raw)
                if diff and isinstance(diff, dict):
                    inputs = diff.get("inputs", [])
                    for inp in inputs:
                        field_name = inp.get("field", "")
                        value = inp.get("value", "")
                        if value and len(value) > 3:
                            content.append({
                                "type": "text_input",
                                "field": field_name,
                                "value_preview": value[:100],
                                "full_value": value[:10_000],
                                "timestamp": ev.get("timestamp", ""),
                            })

        return content

    # ------------------------------------------------------------------
    # Selection signal extraction
    # ------------------------------------------------------------------

    def extract_selection_signals(self, events: list[dict]) -> list[dict]:
        """Extract selection signals from navigation and dwell patterns.

        From dwell times and navigation patterns, infer what the user
        engaged with vs scrolled past.

        Args:
            events: Raw event dicts from the database.

        Returns:
            List of selection signal dicts.
        """
        signals: list[dict] = []

        # Track dwell time per URL/location
        location_dwells: dict[str, list[float]] = defaultdict(list)
        prev_timestamp: str | None = None
        prev_location: str | None = None

        for ev in events:
            ann_raw = ev.get("scene_annotation_json")
            if not ann_raw:
                continue
            ann = _parse_json_field(ann_raw)
            if not ann or not isinstance(ann, dict):
                continue

            location = ann.get("location", "")
            timestamp = ev.get("timestamp", "")

            if prev_location and prev_timestamp and timestamp:
                try:
                    t1 = datetime.fromisoformat(
                        prev_timestamp.replace("Z", "+00:00")
                    )
                    t2 = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                    dwell_seconds = (t2 - t1).total_seconds()
                    if 0 < dwell_seconds < 600:  # Cap at 10 min
                        location_dwells[prev_location].append(dwell_seconds)
                except (ValueError, TypeError):
                    pass

            prev_timestamp = timestamp
            prev_location = location

        # Classify locations by engagement level
        for loc, dwells in location_dwells.items():
            if not loc or not dwells:
                continue
            avg_dwell = sum(dwells) / len(dwells)
            visit_count = len(dwells)

            engagement = "high" if avg_dwell > 30 else "low" if avg_dwell < 5 else "medium"

            signals.append({
                "location": loc,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "visit_count": visit_count,
                "engagement": engagement,
            })

        # Sort by engagement (high first)
        engagement_order = {"high": 0, "medium": 1, "low": 2}
        signals.sort(key=lambda s: engagement_order.get(s["engagement"], 3))

        return signals

    # ------------------------------------------------------------------
    # URL pattern extraction
    # ------------------------------------------------------------------

    def _extract_url_patterns(self, events: list[dict]) -> list[dict]:
        """Extract URL navigation patterns from annotations."""
        url_counter: Counter[str] = Counter()
        url_sequence: list[str] = []

        for ev in events:
            ann_raw = ev.get("scene_annotation_json")
            if not ann_raw:
                continue
            ann = _parse_json_field(ann_raw)
            if not ann or not isinstance(ann, dict):
                continue

            location = ann.get("location", "").strip()
            if location and location.startswith(("http://", "https://")):
                url_counter[location] += 1
                if not url_sequence or url_sequence[-1] != location:
                    url_sequence.append(location)

        patterns: list[dict] = []
        for url, count in url_counter.most_common(20):
            patterns.append({
                "url": url,
                "visit_count": count,
                "domain": _extract_domain(url),
            })

        return patterns

    # ------------------------------------------------------------------
    # Timing pattern extraction
    # ------------------------------------------------------------------

    def _extract_timing_patterns(self, events: list[dict]) -> dict:
        """Extract timing patterns from event timestamps."""
        if not events:
            return {}

        timestamps: list[datetime] = []
        for ev in events:
            ts = ev.get("timestamp", "")
            if ts:
                try:
                    timestamps.append(
                        datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    )
                except (ValueError, TypeError):
                    pass

        if len(timestamps) < 2:
            return {}

        timestamps.sort()
        total_duration = (timestamps[-1] - timestamps[0]).total_seconds()

        # Detect gaps (potential phase boundaries)
        gaps: list[float] = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
            gaps.append(gap)

        avg_gap = sum(gaps) / len(gaps) if gaps else 0

        # Find significant pauses (> 2x average gap)
        pause_threshold = max(avg_gap * 2, 30)  # At least 30 seconds
        pauses = [g for g in gaps if g > pause_threshold]

        return {
            "total_duration_seconds": round(total_duration, 1),
            "event_count": len(timestamps),
            "avg_gap_seconds": round(avg_gap, 1),
            "significant_pauses": len(pauses),
            "start_time": timestamps[0].isoformat(),
            "end_time": timestamps[-1].isoformat(),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_json_field(raw: str | dict | None) -> dict | None:
    """Parse a JSON field that may be a string or already a dict."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_domain(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""

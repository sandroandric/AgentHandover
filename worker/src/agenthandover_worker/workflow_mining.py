"""Workflow mining: work sessions, workflow keys, the observation log,
recurrence, value and the review queue.

Passive discovery turns the annotated frames of its window into *work
sessions* (the non-noise frames, split wherever the user was idle for
:data:`IDLE_GAP_SECONDS` or the Mac slept) and records each finished session
as one ``workflow_observations`` row.  A session's *step signature* is its
frames' ``(app, host_or_doc, action_verb)`` triples, run-length collapsed;
URL paths are templated (``/reports/123`` -> ``/reports/{id}``), queries and
fragments dropped.  The ``workflow_key`` is the SHA-256 of that sequence.

Observations from the last :data:`OBSERVATION_WINDOW_DAYS` days are grouped:
two sessions are the same workflow when their keys are equal, their step
bags (the signature as a set) have a Jaccard of at least
:data:`BAG_THRESHOLD`, or the aligner (``VariantDetector.semantic_align``)
matches at least :data:`ALIGN_THRESHOLD` of the longer sequence's steps.
A group of :data:`MIN_OBSERVATIONS` or more that no Skill came from yet is a
*candidate*; the review queue admits at most :data:`WEEKLY_REVIEW_CAP`
candidates per seven days, highest value first, where value is the median
session duration times the occurrences per month.

The observation log outlives raw retention: the daemon purges events after
``retention_days_raw`` but never this table, and it holds off purging events
newer than :data:`DISTILLED_STATE_KEY` (the point up to which every
finished session has been recorded).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from agenthandover_worker import run_signals
from agenthandover_worker.dom import host_or_doc, template_path  # noqa: F401 (re-exported)
from agenthandover_worker.evidence_extractor import EvidenceExtractor
from agenthandover_worker.pattern_detector import OBSERVATION_WINDOW_DAYS, PatternDetector
from agenthandover_worker.scene_annotator import as_compact
from agenthandover_worker.sop_dedup import _normalize_app
from agenthandover_worker.task_segmenter import (
    AnnotatedFrame,
    _is_noise_frame,
    _sleep_between,
    _sleep_starts,
    _timestamp_to_epoch,
)
from agenthandover_worker.variant_detector import VariantDetector

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB
    from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# A pause this long between two work frames ends a session.
IDLE_GAP_SECONDS = 600
# A session needs this many work frames to be an observation.
MIN_SESSION_FRAMES = 2
ALIGN_THRESHOLD = 0.6
BAG_THRESHOLD = 0.7
# A workflow observed this often becomes a draft Skill.
MIN_OBSERVATIONS = 3
# Drafts the review queue admits per seven days.
WEEKLY_REVIEW_CAP = 3
# task_segments rows are kept this long.
SEGMENT_TTL_DAYS = 14
# Observations that never became part of a Skill are kept for two
# recurrence windows; promoted ones are a Skill's evidence and stay.
UNPROMOTED_TTL_DAYS = 2 * OBSERVATION_WINDOW_DAYS
# worker_state key: UTC time up to which every finished session is recorded.
DISTILLED_STATE_KEY = "observations_distilled_through"

# ---------------------------------------------------------------------------
# Step signatures and keys
# ---------------------------------------------------------------------------


def action_verb(annotation: dict) -> str:
    """What the user did on the frame, from the annotation's structure
    rather than its wording: ``enter`` (typed text), ``click`` (a chosen
    item or control), else ``view``."""
    ann = as_compact(annotation)
    if str((ann.get("key_text") or {}).get("typed_text") or "").strip():
        return "enter"
    if any(isinstance(i, dict) and i.get("chosen") for i in ann.get("visible_items") or []):
        return "click"
    return "view"


def step_signature(frame: AnnotatedFrame) -> tuple[str, str, str]:
    return (
        _normalize_app(frame.app or ""),
        host_or_doc(frame.location),
        action_verb(frame.annotation),
    )


def collapse(sigs: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Run-length collapse: consecutive equal signatures become one."""
    out: list[tuple[str, str, str]] = []
    for sig in sigs:
        if not out or out[-1] != sig:
            out.append(sig)
    return out


def workflow_key(steps: list[tuple[str, str, str]]) -> str:
    return hashlib.sha256(json.dumps([list(s) for s in steps]).encode()).hexdigest()


def bag_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


_ALIGNER = VariantDetector()


def align_score(a: list[tuple[str, str, str]], b: list[tuple[str, str, str]]) -> float:
    """Share of the longer sequence's steps the aligner matches.  A step
    is compared on its app and its host_or_doc (the verb is left to the
    key and the bag), so the same page reached in a different way still
    aligns and two different documents on one host do not."""
    if not a or not b:
        return 0.0

    def steps(seq):
        return [{"action": doc, "app": app, "location": doc} for app, doc, _ in seq]

    aligned = _ALIGNER.semantic_align(steps(a), steps(b))
    return sum(s.is_match for s in aligned) / max(len(a), len(b))


def same_workflow(a: list[tuple[str, str, str]], b: list[tuple[str, str, str]]) -> bool:
    """Two collapsed step signatures describe the same workflow."""
    if a == b:
        return True
    bag_a, bag_b = set(a), set(b)
    if bag_jaccard(bag_a, bag_b) >= BAG_THRESHOLD:
        return True
    # The aligner can only match steps that share an app or a place.
    if not ({s[0] for s in a} & {s[0] for s in b} or {s[1] for s in a} & {s[1] for s in b}):
        return False
    return align_score(a, b) >= ALIGN_THRESHOLD


# ---------------------------------------------------------------------------
# Sessions and the observation log
# ---------------------------------------------------------------------------


def split_sessions(
    events: list[dict], sleep_boundaries: list[tuple[str, str]] | None = None,
) -> list[list[AnnotatedFrame]]:
    """The work frames of *events* (noise frames dropped), oldest first,
    split into sessions at every idle gap of :data:`IDLE_GAP_SECONDS` or
    more and at every system sleep."""
    frames = [f for f in map(AnnotatedFrame.from_event, events) if f and not _is_noise_frame(f)]
    frames.sort(key=lambda f: f.timestamp)
    sleeps = _sleep_starts(sleep_boundaries)
    sessions: list[list[AnnotatedFrame]] = []
    for frame in frames:
        if sessions:
            prev = _timestamp_to_epoch(sessions[-1][-1].timestamp)
            cur = _timestamp_to_epoch(frame.timestamp)
            if cur - prev < IDLE_GAP_SECONDS and not _sleep_between(prev, cur, sleeps):
                sessions[-1].append(frame)
                continue
        sessions.append([frame])
    return sessions


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def observation_id(session: list[AnnotatedFrame]) -> str:
    """A session's observation id, from its first frame: re-recording the
    same session (a later run, a late annotation) updates its row."""
    return "obs-" + hashlib.sha256(session[0].event_id.encode()).hexdigest()[:16]


def build_observation(
    session: list[AnnotatedFrame], events: list[dict], task_label: str = "",
    signals: dict | None = None,
) -> dict:
    """The ``workflow_observations`` row of a finished *session* (its raw
    *events* feed the evidence extracted into ``evidence_json``, with the
    run's *signals*: lead-in, commit, checks, corrections, from
    ``run_signals``)."""
    steps = collapse([step_signature(f) for f in session])
    started = _parse(session[0].timestamp)
    local = started.astimezone()
    hosts: list[str] = []
    for f in session:
        if f.location.startswith(("http://", "https://")):
            host = (urlsplit(f.location).hostname or "").lower().removeprefix("www.")
            if host and host not in hosts:
                hosts.append(host)
    apps: list[str] = []
    for f in session:
        if f.app and f.app not in apps:
            apps.append(f.app)
    if not task_label:
        labels = Counter(f.what_doing for f in session if f.what_doing)
        task_label = labels.most_common(1)[0][0] if labels else ""
    evidence = EvidenceExtractor.extract_evidence(events)
    if signals is not None:
        evidence["signals"] = signals
    return {
        "obs_id": observation_id(session),
        "workflow_key": workflow_key(steps),
        "started_utc": started.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "started_local_date": local.date().isoformat(),
        "started_local_weekday": local.weekday(),
        "started_local_minute": local.hour * 60 + local.minute,
        "duration_s": round((_parse(session[-1].timestamp) - started).total_seconds(), 1),
        "apps_json": json.dumps(apps),
        "hosts_json": json.dumps(hosts),
        "step_sig_json": json.dumps([list(s) for s in steps]),
        "evidence_json": json.dumps(evidence, default=str),
        "source": "passive",
        "task_label": task_label,
        "event_ids_json": json.dumps([f.event_id for f in session]),
    }


def record_observations(
    db: "WorkerDB",
    events: list[dict],
    *,
    now: datetime,
    labels: dict[str, str] | None = None,
    sleep_boundaries: list[tuple[str, str]] | None = None,
) -> int:
    """Record every finished work session of *events* (a session is
    finished once the user has been idle in it for :data:`IDLE_GAP_SECONDS`
    before *now*) and advance :data:`DISTILLED_STATE_KEY`.  *labels* maps
    event ids to their segment's task label.  Returns the number of
    sessions recorded (an already recorded session is updated in place).

    Passive discovery runs every few minutes over a sliding window, so the
    window's start regularly falls inside a session recorded by an earlier
    run; such a truncated session holds frames of another observation and
    is skipped rather than recorded as a second one."""
    by_id = {e.get("id", ""): e for e in events}
    labels = labels or {}
    cutoff = now.timestamp() - IDLE_GAP_SECONDS
    distilled = now - timedelta(seconds=IDLE_GAP_SECONDS)
    recorded = 0
    sessions = split_sessions(events, sleep_boundaries)
    owner: dict[str, str] = {}  # event id -> the observation holding it
    if sessions:
        for obs in db.get_workflow_observations(
            since=_parse(sessions[0][0].timestamp) - timedelta(days=1),
        ):
            owner.update(dict.fromkeys(json.loads(obs["event_ids_json"]), obs["obs_id"]))
    for session in sessions:
        if _timestamp_to_epoch(session[-1].timestamp) > cutoff:
            distilled = min(distilled, _parse(session[0].timestamp))
            continue  # still in progress
        if len(session) < MIN_SESSION_FRAMES:
            continue
        obs_id = observation_id(session)
        if any(owner.get(f.event_id, obs_id) != obs_id for f in session):
            continue  # part of an observation already recorded
        seg_labels = Counter(labels[f.event_id] for f in session if labels.get(f.event_id))
        label = seg_labels.most_common(1)[0][0] if seg_labels else ""
        signals = run_signals.signals_from_db(
            db, _parse(session[0].timestamp), _parse(session[-1].timestamp),
        )
        obs = build_observation(session, [by_id[f.event_id] for f in session], label, signals)
        if db.save_workflow_observation(obs):
            recorded += 1
    db.set_state(
        DISTILLED_STATE_KEY,
        distilled.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    )
    return recorded


# ---------------------------------------------------------------------------
# Grouping, value, the review queue
# ---------------------------------------------------------------------------


def _steps(obs: dict) -> list[tuple[str, str, str]]:
    return [tuple(s) for s in json.loads(obs.get("step_sig_json") or "[]")]


@dataclass
class WorkflowGroup:
    """Observations of one workflow; the first is its representative."""

    observations: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> list[tuple[str, str, str]]:
        return _steps(self.observations[0])

    @property
    def source_id(self) -> str:
        return self.observations[0]["workflow_key"]

    @property
    def promoted_slug(self) -> str | None:
        return next((o["promoted_slug"] for o in self.observations if o.get("promoted_slug")), None)

    @property
    def task_label(self) -> str:
        labels = Counter(o["task_label"] for o in self.observations if o.get("task_label"))
        if not labels:
            return ""
        top = max(labels.values())
        # Ties go to the most recent observation's label.
        return next(o["task_label"] for o in reversed(self.observations)
                    if labels.get(o.get("task_label", "")) == top)


def group_observations(observations: list[dict]) -> list[WorkflowGroup]:
    """Greedy grouping, oldest first: an observation joins the first group
    whose representative (or any member's key) it matches, else starts one."""
    groups: list[WorkflowGroup] = []
    keys: list[set[str]] = []
    for obs in observations:
        steps = _steps(obs)
        for group, group_keys in zip(groups, keys):
            if obs["workflow_key"] in group_keys or same_workflow(steps, group.steps):
                group.observations.append(obs)
                group_keys.add(obs["workflow_key"])
                break
        else:
            groups.append(WorkflowGroup([obs]))
            keys.append({obs["workflow_key"]})
    return groups


def workflow_value(observations: list[dict], history_days: float) -> dict:
    """``value_score = median_duration_s x occurrences_per_month`` over the
    *history_days* the observations cover, and the share of the workflow's
    steps done in a browser (``automatable_fraction``)."""
    durations = sorted(float(o["duration_s"]) for o in observations)
    median = durations[len(durations) // 2] if durations else 0.0
    per_month = len(observations) * 30.0 / max(history_days, 1.0)
    steps = _steps(observations[0]) if observations else []
    hosts = set(json.loads(observations[0].get("hosts_json") or "[]")) if observations else set()
    browser = sum(1 for _, doc, _ in steps if doc.split("/")[0] in hosts)
    score = median * per_month
    return {
        "median_duration_s": round(median, 1),
        "occurrences_per_month": round(per_month, 1),
        "value_score": round(score, 1),
        "hours_per_month": round(score / 3600, 1),
        "automatable_fraction": round(browser / len(steps), 2) if steps else 0.0,
    }


def about_hours(value: dict) -> str:
    """``about N hours a month`` (minutes below one hour)."""
    hours = value.get("hours_per_month", 0.0)
    if hours >= 1:
        n = round(hours)
        return f"about {n} hour{'s' if n != 1 else ''} a month"
    minutes = max(1, round(hours * 60))
    return f"about {minutes} minute{'s' if minutes != 1 else ''} a month"


@dataclass
class Candidate:
    """A workflow ready to become a draft Skill."""

    source_id: str
    task_label: str
    observations: list[dict]
    recurrence: dict | None
    value: dict

    @property
    def obs_ids(self) -> list[str]:
        return [o["obs_id"] for o in self.observations]

    def template_evidence(self) -> dict:
        """What the draft's SOP template carries about its observations."""
        return {
            "episode_count": len(self.observations),
            "recurrence": self.recurrence,
            "value": self.value,
            "observations": observation_pointers(self.observations),
        }


def observation_pointers(observations: list[dict]) -> list[dict]:
    """A draft's evidence pointers into the observation log, each with the
    run's signals (``run_signals``) when the row carries them."""
    pointers = []
    for o in observations:
        pointer = {
            "obs_id": o["obs_id"], "timestamp": o["started_utc"],
            "date": o["started_local_date"], "duration_s": o["duration_s"],
            "source": "passive",
        }
        try:
            signals = json.loads(o.get("evidence_json") or "{}").get("signals")
        except (TypeError, ValueError, AttributeError):
            signals = None
        if isinstance(signals, dict):
            pointer["signals"] = signals
        pointers.append(pointer)
    return pointers


@dataclass
class WorkflowWindow:
    """The observation window at *now*: its groups and what recurrence
    and value are computed against."""

    now: datetime
    groups: list[WorkflowGroup]
    in_window: set[str]
    active_dates: set[str]
    history_days: float

    def window_observations(self, group: WorkflowGroup) -> list[dict]:
        return [o for o in group.observations if o["obs_id"] in self.in_window]

    def recurrence(self, group: WorkflowGroup, slug: str = "") -> dict | None:
        pattern = PatternDetector(None, min_observations=MIN_OBSERVATIONS).detect_from_observations(
            self.window_observations(group), today=self.now.astimezone().date(),
            active_dates=self.active_dates, slug=slug,
        )
        return pattern.as_recurrence() if pattern else None

    def value(self, group: WorkflowGroup) -> dict:
        return workflow_value(self.window_observations(group), self.history_days)


def load_window(db: "WorkerDB", now: datetime) -> WorkflowWindow:
    """Group the last :data:`OBSERVATION_WINDOW_DAYS` days of observations,
    plus the three most recent observations of every earlier promoted
    workflow (so a Skill's workflow is recognised after six weeks)."""
    since = now - timedelta(days=OBSERVATION_WINDOW_DAYS)
    window = db.get_workflow_observations(since=since, until=now)
    anchors: dict[str, list[dict]] = {}
    for obs in db.get_workflow_observations(until=since):
        if obs.get("promoted_slug"):
            anchors.setdefault(obs["promoted_slug"], []).append(obs)
    older = [o for rows in anchors.values() for o in rows[-3:]]
    older.sort(key=lambda o: o["started_utc"])
    first = _parse(window[0]["started_utc"]) if window else now
    return WorkflowWindow(
        now=now,
        groups=group_observations(older + window),
        in_window={o["obs_id"] for o in window},
        active_dates={o["started_local_date"] for o in window},
        history_days=min(max((now - first).total_seconds() / 86400, 7.0), OBSERVATION_WINDOW_DAYS),
    )


def _blocked(db: "WorkerDB", obs_ids: set[str]) -> bool:
    """A passive generation failed for exactly these observations (or a
    superset) and has not been retried: wait for new evidence or a retry."""
    for failure in db.get_failed_generations():
        if failure.get("source") != "passive":
            continue
        try:
            context = json.loads(failure.get("context_json") or "{}")
        except (TypeError, ValueError):
            continue
        if obs_ids <= set(context.get("obs_ids") or []):
            return True
    return False


def review_candidates(db: "WorkerDB", *, now: datetime) -> list[Candidate]:
    """The workflows the review queue admits at *now*, highest value first.

    Later observations of an already promoted workflow are attributed to
    its Skill here and are never candidates again.  A candidate has at
    least :data:`MIN_OBSERVATIONS` observations in the window and two
    distinct steps; at most :data:`WEEKLY_REVIEW_CAP` drafts are promoted
    per seven days.
    """
    win = load_window(db, now)
    eligible: list[Candidate] = []
    for group in win.groups:
        slug = group.promoted_slug
        if slug:
            fresh = [o["obs_id"] for o in group.observations if not o.get("promoted_slug")]
            db.set_observations_promoted(fresh, slug)
            continue
        observations = win.window_observations(group)
        if len(observations) < MIN_OBSERVATIONS or len(set(group.steps)) < 2:
            continue
        if _blocked(db, {o["obs_id"] for o in observations}):
            continue
        eligible.append(Candidate(
            source_id=group.source_id,
            task_label=group.task_label,
            observations=observations,
            recurrence=win.recurrence(group),
            value=win.value(group),
        ))
    slots = WEEKLY_REVIEW_CAP - db.count_promotions_since(now - timedelta(days=7))
    eligible.sort(key=lambda c: c.value["value_score"], reverse=True)
    if len(eligible) > max(slots, 0):
        logger.info(
            "Review queue: %d candidate(s) wait (cap %d per week, %d slot(s) left)",
            len(eligible) - max(slots, 0), WEEKLY_REVIEW_CAP, max(slots, 0),
        )
    return eligible[:max(slots, 0)]


def recurring_workflows(db: "WorkerDB", *, now: datetime, date: str | None = None) -> list[dict]:
    """Workflows seen at least :data:`MIN_OBSERVATIONS` times in the window
    (optionally only those observed on the local *date*), highest value
    first: ``{label, slug, observations, recurrence, value}``."""
    win = load_window(db, now)
    out = []
    for group in win.groups:
        observations = win.window_observations(group)
        if len(observations) < MIN_OBSERVATIONS:
            continue
        if date is not None and not any(o["started_local_date"] == date for o in observations):
            continue
        out.append({
            "label": group.task_label,
            "slug": group.promoted_slug,
            "observations": len(observations),
            "recurrence": win.recurrence(group, group.promoted_slug or ""),
            "value": win.value(group),
        })
    out.sort(key=lambda w: w["value"]["value_score"], reverse=True)
    return out


def refresh_skills(db: "WorkerDB", kb: "KnowledgeBase", *, now: datetime) -> int:
    """Write each promoted workflow's recurrence, value and observation
    count onto its procedure in the knowledge base.  Returns the number of
    procedures updated."""
    updated = 0
    for workflow in recurring_workflows(db, now=now):
        slug = workflow["slug"]
        proc = kb.get_procedure(slug) if slug else None
        if proc is None:
            continue
        if workflow["recurrence"]:
            proc["recurrence"] = workflow["recurrence"]
        proc["value"] = workflow["value"]
        evidence = proc.setdefault("evidence", {})
        evidence["total_observations"] = max(
            evidence.get("total_observations", 0), workflow["observations"],
        )
        kb.save_procedure(proc)
        updated += 1
    return updated


def run_daily(db: "WorkerDB", kb: "KnowledgeBase", *, now: datetime) -> None:
    """The daily batch's share: segment TTL, the unpromoted-observation
    TTL, and the promoted workflows' recurrence and value."""
    for name, job in (
        ("task segment TTL", lambda: db.purge_task_segments(SEGMENT_TTL_DAYS, now=now)),
        ("observation TTL", lambda: db.purge_unpromoted_observations(UNPROMOTED_TTL_DAYS, now=now)),
        ("skill recurrence refresh", lambda: refresh_skills(db, kb, now=now)),
    ):
        try:
            job()
        except Exception:
            logger.warning("Daily %s failed", name, exc_info=True)

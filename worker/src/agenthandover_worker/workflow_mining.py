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

Skills and their observations (``promoted_slug``):

- a Skill mined here is attributed the observations that promoted it, and
  every later observation grouped with them;
- a focus recording is recorded as an observation of its own
  (``source="focus"``) attributed to its Skill
  (:func:`record_focus_observation`), so later passive sessions of the
  same workflow group with it; a recording of a routine that already is a
  Skill (:func:`focus_skill`) is another run of that Skill, not a new
  one: the Skill keeps its steps (approved or draft, a later run never
  rewrites them, so no SOP is generated for the run: the match is made
  once its keyframes are annotated, before the SOP call) and the run's
  own way is divergence-mined into branch
  hypotheses and branches, like a passive run's;
- a Skill with no observation at all (recorded before the observation
  log, or imported) is linked by its steps' places: a workflow group no
  Skill holds is attributed to the Skill whose steps' ``(app,
  host_or_doc)`` sequence it aligns with (:func:`link_recorded_skills`);
- while a session is still in progress, :func:`recognize` matches its
  prefix against the Skills' observed signatures every
  :data:`RECOGNIZE_INTERVAL_SECONDS`; a match tags the session with the
  Skill (``worker_state`` :data:`LIVE_SESSION_STATE_KEY`) and the session
  is recorded as the Skill's observation as soon as it ends.

Every observation attached to an existing Skill is aligned to its steps
for divergence branches (:mod:`branch_mining`).
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
from agenthandover_worker.dom import host_or_doc, site_of
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
# A workflow the user labelled "not a routine" stays out of the review
# queue this long (``questions``).
SUPPRESS_DAYS = 42
# task_segments rows are kept this long.
SEGMENT_TTL_DAYS = 14
# Observations that never became part of a Skill are kept for two
# recurrence windows; promoted ones are a Skill's evidence and stay.
UNPROMOTED_TTL_DAYS = 2 * OBSERVATION_WINDOW_DAYS
# worker_state key: UTC time up to which every finished session is recorded.
DISTILLED_STATE_KEY = "observations_distilled_through"
# Live recognition: how often the session in progress is matched, over
# how many hours of frames, and how many collapsed steps it needs first.
RECOGNIZE_INTERVAL_SECONDS = 120
RECOGNIZE_WINDOW_HOURS = 4
MIN_PREFIX_STEPS = 3
# worker_state key: the session in progress tagged with a Skill,
# ``{"first_event_id", "attached_slug", "matched_at"}`` (empty when none).
LIVE_SESSION_STATE_KEY = "live_session"

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


def add_instances(proc: dict, observations: list[dict]) -> bool:
    """Add *observations* the procedure does not yet cite to
    ``proc["evidence"]["observations"]`` as :func:`observation_pointers`
    (with their run signals, which ``run_signals.learn`` reads when the
    procedure is saved), in place.  Returns whether any was added."""
    evidence = proc.setdefault("evidence", {})
    cited = evidence.setdefault("observations", [])
    held = {str(o.get("obs_id")) for o in cited if isinstance(o, dict) and o.get("obs_id")}
    new = observation_pointers([o for o in observations if o["obs_id"] not in held])
    if not new:
        return False
    cited.extend(new)
    evidence["total_observations"] = max(evidence.get("total_observations", 0), len(cited))
    return True


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


def _anchors(db: "WorkerDB", until: datetime) -> list[dict]:
    """The three most recent observations before *until* of every Skill,
    oldest first."""
    anchors: dict[str, list[dict]] = {}
    for obs in db.get_workflow_observations(until=until):
        if obs.get("promoted_slug"):
            anchors.setdefault(obs["promoted_slug"], []).append(obs)
    older = [o for rows in anchors.values() for o in rows[-3:]]
    older.sort(key=lambda o: o["started_utc"])
    return older


def load_window(db: "WorkerDB", now: datetime) -> WorkflowWindow:
    """Group the last :data:`OBSERVATION_WINDOW_DAYS` days of observations,
    plus the three most recent observations of every earlier promoted
    workflow (so a Skill's workflow is recognised after six weeks)."""
    since = now - timedelta(days=OBSERVATION_WINDOW_DAYS)
    window = db.get_workflow_observations(since=since, until=now)
    older = _anchors(db, since)
    first = _parse(window[0]["started_utc"]) if window else now
    return WorkflowWindow(
        now=now,
        groups=group_observations(older + window),
        in_window={o["obs_id"] for o in window},
        active_dates={o["started_local_date"] for o in window},
        history_days=min(max((now - first).total_seconds() / 86400, 7.0), OBSERVATION_WINDOW_DAYS),
    )


def suppressed(observations: list[dict], now: datetime) -> bool:
    """True when the user said, less than :data:`SUPPRESS_DAYS` days
    before *now*, that this workflow is not a routine they would hand off."""
    cutoff = now - timedelta(days=SUPPRESS_DAYS)
    return any(
        o.get("user_is_routine") == 0 and o.get("labelled_at")
        and _parse(o["labelled_at"]) >= cutoff
        for o in observations
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


# ---------------------------------------------------------------------------
# Skills and their observations
# ---------------------------------------------------------------------------


def attach_observations(
    db: "WorkerDB", kb: "KnowledgeBase | None", slug: str, observations: list[dict],
) -> None:
    """Attribute *observations* (new instances of an existing Skill) to
    *slug* and, with a knowledge base, add them to the procedure's
    evidence (:func:`add_instances`: what the runs teach, its learned
    trigger, done and verification, is recomputed from them) and record
    where they diverge from its steps
    (:func:`branch_mining.mine_observations`)."""
    if not observations:
        return
    db.set_observations_promoted([o["obs_id"] for o in observations], slug)
    for obs in observations:
        obs["promoted_slug"] = slug
    if kb is None:
        return
    proc = kb.get_procedure(slug)
    if proc is not None and add_instances(proc, observations):
        kb.save_procedure(proc)  # saving recomputes ``learned`` and the ledger
    from agenthandover_worker.branch_mining import mine_observations

    try:
        mine_observations(db, kb, slug, observations)
    except Exception:
        logger.warning("Branch mining failed for '%s'", slug, exc_info=True)


def skill_signature(proc: dict) -> list[tuple[str, str, str]]:
    """A Skill's steps as step signatures ``(app, host_or_doc, "")``,
    collapsed; steps without an app or a location are left out (the verb
    is not known from a written step, and :func:`align_score` does not
    compare it)."""
    return collapse([
        (_normalize_app(str(s.get("app") or "")), host_or_doc(str(s.get("location") or "")), "")
        for s in proc.get("steps") or []
        if isinstance(s, dict) and (s.get("app") or s.get("location"))
    ])


def _skill_score(signature: list[tuple[str, str, str]], steps: list[tuple[str, str, str]]) -> float:
    """How well a workflow's *steps* perform a Skill's *signature*: the
    aligner's score, 0 when they share no app and no place."""
    if not ({s[0] for s in signature} & {s[0] for s in steps}
            or {s[1] for s in signature} & {s[1] for s in steps}):
        return 0.0
    return align_score(signature, steps)


def link_recorded_skills(db: "WorkerDB", kb: "KnowledgeBase", win: WorkflowWindow) -> int:
    """Attribute each workflow group no Skill holds to the Skill without
    any observation whose steps' places it aligns with (at least
    :data:`ALIGN_THRESHOLD`, the best such Skill).  Returns the number of
    groups linked."""
    held = {o["promoted_slug"] for g in win.groups for o in g.observations if o.get("promoted_slug")}
    skills = []
    for proc in kb.list_procedures():
        slug = proc.get("id") or proc.get("slug")
        signature = skill_signature(proc)
        if slug and slug not in held and len(set(signature)) >= 2:
            skills.append((slug, signature))
    linked = 0
    for group in win.groups:
        if not skills or group.promoted_slug:
            continue
        score, slug = max((_skill_score(sig, group.steps), slug) for slug, sig in skills)
        if score < ALIGN_THRESHOLD:
            continue
        logger.info(
            "Workflow '%s' (%d observation(s)) is the Skill '%s'",
            group.task_label[:60], len(group.observations), slug,
        )
        attach_observations(db, kb, slug, group.observations)
        linked += 1
    return linked


def _focus_frame(frame: AnnotatedFrame | None) -> bool:
    """Whether *frame* is part of the focus recording's task.

    The user started the recording for the task, so every annotated frame
    of it is, except a skipped one (no app and no place) and one the older
    annotation schema explicitly labelled as not learnable
    (``learnability: ignore``) or as entertainment or dead time.  The
    compact schema's ``is_workflow`` does not decide: it is false for
    reading and browsing, which is what a routine like reading the news
    and sharing stories mostly is (the passive path's noise filter,
    ``_is_noise_frame``, falls back to it)."""
    if frame is None or not (frame.app or frame.location):
        return False
    if frame.learnability == "ignore":
        return False
    return frame.activity_type not in ("entertainment", "dead_time")


def _focus_session(events: list[dict]) -> list[AnnotatedFrame]:
    """A focus recording's annotated task frames (:func:`_focus_frame`),
    oldest first."""
    session = [f for f in map(AnnotatedFrame.from_event, events) if _focus_frame(f)]
    session.sort(key=lambda f: f.timestamp)
    return session


def site_steps(steps: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """*steps* with each place reduced to its site (``dom.site_of``) and
    no verb, collapsed: another story, another record on the same site is
    the same step (as in :mod:`branch_mining`)."""
    return collapse([(app, site_of(doc), "") for app, doc, _ in steps])


def _same_title(a: str, b: str) -> bool:
    return bool(a.strip()) and " ".join(a.lower().split()) == " ".join(b.lower().split())


def focus_skill(
    db: "WorkerDB", kb: "KnowledgeBase", events: list[dict], title: str, *, now: datetime,
) -> dict | None:
    """The existing Skill a focus recording is another run of, or None.

    The recording's steps (its annotated frames, as an observation's
    signature) are matched against every Skill's observations (the
    window's and each Skill's most recent older ones, as live
    :func:`recognize` does; the recording's own observation, on a retry,
    is left out) and, for a Skill with none, against its steps' places
    (:func:`skill_signature`, as :func:`link_recorded_skills` does):

    - the same workflow (:func:`same_workflow`: key, step bag or
      alignment), or
    - named as the Skill is (the user's title for the recording equals
      the Skill's title) and aligning with it by site
      (:func:`site_steps`, at least :data:`ALIGN_THRESHOLD`): a routine
      whose items change every run (the day's stories, this week's
      invoice) visits other pages of the same sites each time.

    The best-aligned match wins.  Returns the Skill's procedure (a copy
    the caller may edit and save)."""
    session = _focus_session(events)
    if len(session) < MIN_SESSION_FRAMES:
        return None
    steps = collapse([step_signature(f) for f in session])
    own = observation_id(session)
    since = now - timedelta(days=OBSERVATION_WINDOW_DAYS)
    signatures: dict[str, list[list[tuple[str, str, str]]]] = {}
    for obs in _anchors(db, since) + db.get_workflow_observations(since=since, until=now):
        if obs.get("promoted_slug") and obs["obs_id"] != own:
            signatures.setdefault(obs["promoted_slug"], []).append(_steps(obs))
    best: tuple[float, str] | None = None
    for proc in kb.list_procedures():
        slug = proc.get("id") or proc.get("slug")
        if not slug:
            continue
        for signature in signatures.get(slug) or [skill_signature(proc)]:
            if len(set(signature)) < 2:
                continue  # one place says nothing about the routine
            by_site = _skill_score(site_steps(signature), site_steps(steps))
            matched = same_workflow(steps, signature) or (
                _same_title(title, str(proc.get("title") or "")) and by_site >= ALIGN_THRESHOLD
            )
            if matched and (best is None or by_site > best[0]):
                best = (by_site, slug)
    # A copy to edit (``list_procedures`` shares its cache).
    return kb.get_procedure(best[1]) if best else None


def record_focus_observation(
    db: "WorkerDB", events: list[dict], slug: str, title: str = "",
    *, instance_of: dict | None = None, session_id: str = "",
) -> str | None:
    """Record a focus recording's annotated frames as one observation
    (``source="focus"``) of its Skill *slug*.  With *instance_of*, the
    existing Skill *slug* the recording is another run of
    (:func:`focus_skill`), the run's divergences from its steps are
    recorded on that procedure in place, under the recording's
    *session_id* (the id the ledger cites the run by), as
    :func:`attach_observations` does for passive runs.  Returns its
    ``obs_id``, or None when fewer than :data:`MIN_SESSION_FRAMES` frames
    are annotated."""
    session = _focus_session(events)
    if len(session) < MIN_SESSION_FRAMES:
        return None
    by_id = {e.get("id", ""): e for e in events}
    signals = run_signals.signals_from_db(
        db, _parse(session[0].timestamp), _parse(session[-1].timestamp),
    )
    obs = build_observation(session, [by_id[f.event_id] for f in session], title, signals)
    obs["source"] = "focus"
    if not db.save_workflow_observation(obs):
        return None
    db.set_observations_promoted([obs["obs_id"]], slug)
    if instance_of is not None:
        from agenthandover_worker.branch_mining import record_divergences

        record_divergences(instance_of, session_id or obs["obs_id"], session)
    return obs["obs_id"]


def _match_prefix(
    steps: list[tuple[str, str, str]], known: list[dict],
) -> str | None:
    """The Skill whose observed signature *steps* begin: an exact prefix
    first, else the first whose opening :func:`same_workflow` matches."""
    fallback = None
    for obs in known:
        head = _steps(obs)[:len(steps)]
        if head == steps:
            return obs["promoted_slug"]
        if fallback is None and same_workflow(steps, head):
            fallback = obs["promoted_slug"]
    return fallback


def _live_state(db: "WorkerDB") -> dict:
    try:
        state = json.loads(db.get_state(LIVE_SESSION_STATE_KEY) or "{}")
    except (TypeError, ValueError):
        return {}
    return state if isinstance(state, dict) and state.get("attached_slug") else {}


def recognize(db: "WorkerDB", kb: "KnowledgeBase | None", *, now: datetime) -> str | None:
    """Live recognition, model-free, every :data:`RECOGNIZE_INTERVAL_SECONDS`.

    The work session in progress (the last :data:`RECOGNIZE_WINDOW_HOURS`
    hours of annotated frames) is matched once it has
    :data:`MIN_PREFIX_STEPS` collapsed steps: its steps so far against the
    openings of every Skill's observations (the window's and each Skill's
    most recent older ones).  A match tags the session with the Skill.
    Once the tagged session ends (idle for :data:`IDLE_GAP_SECONDS`, the
    Mac slept, or a new session began) it is recorded as an observation
    attached to the Skill (:func:`attach_observations`).  Returns the slug
    the session in progress is tagged with, if any.
    """
    events = db.get_annotated_events_in_window(hours=RECOGNIZE_WINDOW_HOURS, now=now)
    window_start = now - timedelta(hours=RECOGNIZE_WINDOW_HOURS)
    sessions = split_sessions(events, db.get_sleep_boundaries(since=window_start.isoformat()))
    cutoff = now.timestamp() - IDLE_GAP_SECONDS
    live = sessions[-1] if sessions and _timestamp_to_epoch(sessions[-1][-1].timestamp) > cutoff else None

    state = _live_state(db)
    if state:
        tagged = next(
            (s for s in sessions if any(f.event_id == state["first_event_id"] for f in s)), None,
        )
        if tagged is not None and tagged is live:
            return state["attached_slug"]
        if tagged is not None and len(tagged) >= MIN_SESSION_FRAMES:
            by_id = {e.get("id", ""): e for e in events}
            signals = run_signals.signals_from_db(
                db, _parse(tagged[0].timestamp), _parse(tagged[-1].timestamp),
            )
            obs = build_observation(tagged, [by_id[f.event_id] for f in tagged], "", signals)
            if db.save_workflow_observation(obs):
                logger.info("Recognised session recorded for Skill '%s'", state["attached_slug"])
                attach_observations(db, kb, state["attached_slug"], [obs])
        db.set_state(LIVE_SESSION_STATE_KEY, "")

    if live is None:
        return None
    steps = collapse([step_signature(f) for f in live])
    if len(steps) < MIN_PREFIX_STEPS:
        return None
    since = now - timedelta(days=OBSERVATION_WINDOW_DAYS)
    known = _anchors(db, since) + [
        o for o in db.get_workflow_observations(since=since, until=now) if o.get("promoted_slug")
    ]
    slug = _match_prefix(steps, known)
    if slug:
        logger.info("Session in progress recognised as Skill '%s'", slug)
        db.set_state(LIVE_SESSION_STATE_KEY, json.dumps({
            "first_event_id": live[0].event_id,
            "attached_slug": slug,
            "matched_at": now.astimezone(timezone.utc).isoformat(),
        }))
    return slug


def review_candidates(
    db: "WorkerDB", *, now: datetime, kb: "KnowledgeBase | None" = None,
) -> list[Candidate]:
    """The workflows the review queue admits at *now*, highest value first.

    Later observations of an already promoted workflow are attributed to
    its Skill here and are never candidates again; with *kb*, so is a
    workflow that performs a Skill holding no observation yet
    (:func:`link_recorded_skills`), and the attributed observations are
    mined for divergence branches.  A candidate has at least
    :data:`MIN_OBSERVATIONS` observations in the window and two distinct
    steps, and the user has not said it is no routine (:func:`suppressed`); at most :data:`WEEKLY_REVIEW_CAP` drafts are promoted per seven
    days.
    """
    win = load_window(db, now)
    if kb is not None:
        link_recorded_skills(db, kb, win)
    eligible: list[Candidate] = []
    for group in win.groups:
        slug = group.promoted_slug
        if slug:
            attach_observations(
                db, kb, slug, [o for o in group.observations if not o.get("promoted_slug")],
            )
            continue
        observations = win.window_observations(group)
        if len(observations) < MIN_OBSERVATIONS or len(set(group.steps)) < 2:
            continue
        if suppressed(group.observations, now):
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


def recurring_workflows(
    db: "WorkerDB", *, now: datetime, date: str | None = None,
    kb: "KnowledgeBase | None" = None,
) -> list[dict]:
    """Workflows seen at least :data:`MIN_OBSERVATIONS` times in the window
    (optionally only those observed on the local *date*), highest value
    first: ``{label, slug, observations, recurrence, value}``.  With *kb*,
    workflows that perform a Skill holding no observation are linked to it
    first (:func:`link_recorded_skills`)."""
    win = load_window(db, now)
    if kb is not None:
        link_recorded_skills(db, kb, win)
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
    for workflow in recurring_workflows(db, now=now, kb=kb):
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

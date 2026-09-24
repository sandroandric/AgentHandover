"""When a procedure starts, when it is done and how the user checks it,
read from the events around each observed run.

Per run (an *instance*: a focus recording or a passively mined work
session), :func:`instance_signals` reads the raw events around it:

- ``lead_in``: where the user was in the :data:`LEAD_IN_SECONDS` before
  the run started (app, window title, templated URL, the subject of a
  message a mail window shows, the title of a calendar event on screen),
  and the run's local clock position (``local_weekday``,
  ``local_minute``, as ``workflow_observations`` stores them);
- ``done``: the run's first commit event (a form submitted; a Send, Save,
  Submit, Publish, Post or Confirm control clicked or pressed);
- ``verification``: the places the user went to in the
  :data:`VERIFY_SECONDS` after that commit (candidates for how they check
  the result);
- ``corrections``: the fields whose value the user corrected
  (``correction_detector.field_corrections``).

The signals are stored with the instance: ``evidence.observations[i].signals``
on the procedure, ``evidence_json["signals"]`` on a passive observation
row.  :func:`learn` turns the instances' signals into ``proc["learned"]``:

- ``trigger``: a lead-in item seen before at least :data:`MIN_INSTANCES`
  runs and :data:`MIN_SHARE` of them (``kind: "event"``); otherwise, when
  that many runs start at the same local time of day
  (±:data:`TIME_TOLERANCE_MINUTES`), the clock position (``kind: "time"``:
  "Most Mondays around 10:30", "Most weekdays around 09:15").  A calendar
  event title seen in a supporting run's lead-in is carried as
  ``calendar_event``.
- ``done``: the commit that ends at least :data:`MIN_SHARE` of the runs.
- ``verification``: ``checks`` (a place visited after the commit in at
  least :data:`MIN_INSTANCES` runs and :data:`MIN_SHARE` of them) and
  ``corrections`` (quality criteria: what the user fixed, from what to
  what).

Every learned item names the instances it rests on; ``ledger`` makes each
a ``hypothesis`` with those ``instance`` pointers as its evidence.  No
value here is a model's guess: all of it is read from recorded events.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Iterable

from agenthandover_worker.correction_detector import field_corrections
from agenthandover_worker.dom import host_or_doc, url_host
from agenthandover_worker.event_helpers import (
    display_name,
    extract_app,
    extract_location,
    is_own_event,
    parse_annotation,
    parse_json_field,
    parse_timestamp,
)
from agenthandover_worker.redaction import redact

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB

# The window before a run that may hold what started it.
LEAD_IN_SECONDS = 300
# The window after the commit in which the user checks the result.
VERIFY_SECONDS = 60
# A lead-in item, a start time or a check is recurring when at least this
# many runs, and this share of all runs, show it.
MIN_INSTANCES = 3
MIN_SHARE = 0.6
# Runs start "at the same time" within this many minutes of each other.
TIME_TOLERANCE_MINUTES = 30

_MAX_ITEMS = 6
_TEXT_MAX = 120
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Being in a browser says nothing on its own; the page does.
_BROWSERS = frozenset((
    "Google Chrome", "Google Chrome Canary", "Safari", "Arc", "Brave Browser",
    "Microsoft Edge", "Firefox",
))
_MAIL_APPS = frozenset(("Mail", "Microsoft Outlook", "Spark", "Airmail", "Mimestream"))
_MAIL_HOSTS = frozenset((
    "mail.google.com", "outlook.live.com", "outlook.office.com", "outlook.office365.com",
))
_CALENDAR_APPS = frozenset(("Calendar", "Fantastical", "Notion Calendar"))
_CALENDAR_HOSTS = frozenset(("calendar.google.com",))
# Titles of mailbox views, not of a message.
_MAILBOX_TITLES = re.compile(
    r"^(inbox|search results|sent|drafts|archive|all mail|starred|mail|gmail|outlook)\b", re.I,
)
# The label of a control that commits the work.
_COMMIT = re.compile(r"^(send|save|submit|publish|post|confirm)\b", re.I)
_COMMIT_LABEL_MAX = 40
# Title separators: "Subject - account - Gmail", "Report — Expensify".
_TITLE_PARTS = re.compile(r"\s+[-–—|]\s+")


def _clean(text: object) -> str:
    return " ".join(str(text or "").split())[:_TEXT_MAX]


def _norm(text: str) -> str:
    """A text as a recurrence key: lower case, digit runs as ``#``."""
    return re.sub(r"\d+", "#", _clean(text).lower())


def _first_part(title: str) -> str:
    return _TITLE_PARTS.split(title, maxsplit=1)[0].strip()


# ---------------------------------------------------------------------------
# One event, as a place the user was in
# ---------------------------------------------------------------------------


def _kind(event: dict) -> tuple[str, dict]:
    """The event's kind name and its payload (``kind_json`` is serde's
    internally tagged form, ``{"type": ..., ...}``)."""
    kind = parse_json_field(event, "kind_json")
    if "type" in kind:
        return str(kind["type"]), kind
    name = next(iter(kind), "")
    payload = kind.get(name) if name else None
    return str(name), payload if isinstance(payload, dict) else {}


def _view(event: dict) -> dict:
    """Where *event* happened: ``app``, ``title``, ``url``, ``place``."""
    window = parse_json_field(event, "window_json")
    metadata = parse_json_field(event, "metadata_json")
    element = parse_json_field(event, "focused_element_json")
    ann = parse_annotation(event) or {}
    _, payload = _kind(event)
    app = display_name(window.get("app_id")) or extract_app(ann)
    location = extract_location(ann)
    url = next(
        (str(u) for u in (
            payload.get("url"), metadata.get("url"), element.get("url"), location,
        ) if str(u or "").startswith(("http://", "https://"))),
        "",
    )
    document = str(element.get("document") or "")
    return {
        "app": app,
        "title": _clean(window.get("title") or payload.get("title")),
        "url": url,
        "place": host_or_doc(url) if url else host_or_doc(document) if document else "",
    }


def _subject(view: dict) -> str:
    """The subject of the message a mail window shows, else ""."""
    if view["app"] not in _MAIL_APPS and url_host(view["url"]) not in _MAIL_HOSTS:
        return ""
    if "/calendar" in view["url"]:
        return ""  # Outlook on the web's calendar
    subject = _first_part(view["title"])
    return "" if not subject or _MAILBOX_TITLES.match(subject) else subject


def _calendar_event(view: dict) -> str:
    """The title of a calendar event a calendar window shows, else ""."""
    host = url_host(view["url"])
    outlook_calendar = host in _MAIL_HOSTS and "/calendar" in view["url"]
    if view["app"] not in _CALENDAR_APPS and host not in _CALENDAR_HOSTS and not outlook_calendar:
        return ""
    title = _first_part(view["title"])
    # A view of the calendar itself ("Google Calendar - Week of ...").
    return "" if not title or "calendar" in title.lower() else title


def _lead_in_item(event: dict) -> dict | None:
    view = _view(event)
    if not view["app"] and not view["place"]:
        return None
    item = {
        "app": view["app"],
        "place": view["place"],
        "title": redact(view["title"]),
        "event_id": str(event.get("id") or ""),
    }
    subject = _subject(view)
    if subject:
        item["subject"] = redact(subject)
    calendar = _calendar_event(view)
    if calendar:
        item["calendar_event"] = redact(calendar)
    return item


def _commit(event: dict) -> dict | None:
    """The commit *event* is, if it is one: ``{"kind", "label"}``."""
    kind, payload = _kind(event)
    if kind == "FormSubmit":
        anchor = payload.get("form_anchor") if isinstance(payload.get("form_anchor"), dict) else {}
        submitter = anchor.get("submitter") if isinstance(anchor.get("submitter"), dict) else {}
        label = next((
            _clean(v) for v in (
                submitter.get("text"), submitter.get("ariaLabel"), anchor.get("ariaLabel"),
                anchor.get("name"), anchor.get("id"),
            ) if _clean(v)
        ), "form")
        return {"kind": "submit", "label": label}
    label = ""
    if kind == "ClickIntent":
        target = parse_json_field(event, "metadata_json").get("target")
        target = target if isinstance(target, dict) else {}
        label = _clean(
            target.get("innerText") or target.get("ariaLabel")
            or re.sub(r"<\w+>$", "", str(payload.get("target_description") or ""))
        )
    else:
        element = parse_json_field(event, "focused_element_json")
        if element.get("role") == "AXButton":
            label = _clean(element.get("title") or element.get("description"))
    match = _COMMIT.match(label) if len(label) <= _COMMIT_LABEL_MAX else None
    return {"kind": match.group(1).lower(), "label": label} if match else None


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def instance_signals(events: list[dict], *, start: datetime, end: datetime) -> dict:
    """The signals of the run from *start* to *end* (its first and last
    frame), read from *events*: every raw event from
    :data:`LEAD_IN_SECONDS` before *start* to :data:`VERIFY_SECONDS` after
    *end* (:meth:`WorkerDB.get_events_between`).  AgentHandover's own
    windows are left out."""
    timed = sorted(
        (
            (when, e) for e in events
            if (when := parse_timestamp(e.get("timestamp"))) is not None and not is_own_event(e)
        ),
        key=lambda pair: pair[0],
    )
    lead_from = start - timedelta(seconds=LEAD_IN_SECONDS)
    lead_in: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for when, event in reversed([p for p in timed if lead_from <= p[0] < start]):
        item = _lead_in_item(event)
        if item is None:
            continue
        key = (item["app"], item["place"], _norm(item["title"]))
        if key not in seen:
            seen.add(key)
            lead_in.insert(0, item)
        if len(lead_in) >= _MAX_ITEMS:
            break

    run = [(when, e) for when, e in timed if start <= when <= end]
    done = None
    for when, event in run:
        commit = _commit(event)
        if commit is not None:
            view = _view(event)
            done = {
                **commit, "app": view["app"], "place": view["place"],
                "event_id": str(event.get("id") or ""), "at": when.isoformat(),
            }
            break

    verification: list[dict] = []
    if done is not None:
        committed = parse_timestamp(done["at"])
        until = committed + timedelta(seconds=VERIFY_SECONDS)
        checked: set[tuple[str, str]] = {_check_key(done)}
        for when, event in timed:
            if not committed < when <= until:
                continue
            item = _lead_in_item(event)
            if item is None or _check_key(item) in checked:
                continue
            checked.add(_check_key(item))
            verification.append({k: item[k] for k in ("app", "place", "title", "event_id")})
            if len(verification) >= _MAX_ITEMS:
                break

    local = start.astimezone()
    return {
        "lead_in": lead_in,
        "local_weekday": local.weekday(),
        "local_minute": local.hour * 60 + local.minute,
        "done": done,
        "verification": verification,
        "corrections": field_corrections([e for _, e in run])[:_MAX_ITEMS],
    }


def _check_key(item: dict) -> tuple[str, str]:
    return (item.get("app", ""), item.get("place") or _norm(item.get("title", "")))


def signals_from_db(db: "WorkerDB", start: datetime, end: datetime) -> dict:
    """:func:`instance_signals` of the run from *start* to *end*, over the
    events *db* holds around it."""
    events = db.get_events_between(
        start - timedelta(seconds=LEAD_IN_SECONDS), end + timedelta(seconds=VERIFY_SECONDS),
    )
    return instance_signals(events, start=start, end=end)


def run_signals_from_db(db: "WorkerDB", run_events: list[dict]) -> dict | None:
    """:func:`signals_from_db` for the run made of *run_events* (a focus
    session's events), from its first to its last event; None when their
    timestamps cannot be read."""
    times = [t for e in run_events if (t := parse_timestamp(e.get("timestamp"))) is not None]
    if not times:
        return None
    return signals_from_db(db, min(times), max(times))


# ---------------------------------------------------------------------------
# Across runs
# ---------------------------------------------------------------------------


def _instances(proc: dict) -> dict[str, dict]:
    """``{instance id: signals}`` for the procedure's observed runs that
    carry signals, oldest first (the id the ledger's ``instance`` evidence
    uses: a focus ``session_id`` or a passive ``obs_id``)."""
    out: dict[str, dict] = {}
    for obs in (proc.get("evidence") or {}).get("observations") or []:
        if not isinstance(obs, dict) or not isinstance(obs.get("signals"), dict):
            continue
        iid = str(obs.get("session_id") or obs.get("obs_id") or "")
        if iid:
            out[iid] = obs["signals"]
    return out


def _recurring(
    instances: dict[str, dict],
    keys_of: Callable[[dict], Iterable[tuple[int, tuple, dict]]],
    *,
    min_instances: int = MIN_INSTANCES,
) -> list[tuple[int, tuple, list[str], dict]]:
    """The keys seen in at least *min_instances* runs and
    :data:`MIN_SHARE` of all runs, most specific (highest rank) and then
    most often seen first: ``(rank, key, instance ids, latest item)``."""
    ids: dict[tuple, list[str]] = {}
    items: dict[tuple, tuple[int, dict]] = {}
    for iid, signals in instances.items():
        for rank, key, item in keys_of(signals):
            if iid not in ids.setdefault(key, []):
                ids[key].append(iid)
            items[key] = (rank, item)
    found = [
        (items[key][0], key, members, items[key][1])
        for key, members in ids.items()
        if len(members) >= min_instances and len(members) / len(instances) >= MIN_SHARE
    ]
    found.sort(key=lambda f: (-f[0], -len(f[2])))
    return found


def _lead_in_keys(signals: dict) -> Iterable[tuple[int, tuple, dict]]:
    for item in signals.get("lead_in") or []:
        if not isinstance(item, dict):
            continue
        app = item.get("app", "")
        if item.get("calendar_event"):
            yield 4, ("calendar", _norm(item["calendar_event"])), item
        if item.get("subject"):
            yield 3, ("subject", app, _norm(item["subject"])), item
        if item.get("place"):
            yield 2, ("place", app, item["place"]), item
        elif item.get("title"):
            yield 2, ("title", app, _norm(item["title"])), item
        if app and app not in _BROWSERS:
            yield 1, ("app", app), item


def _event_trigger(kind: str, item: dict) -> str:
    app = item.get("app") or "another app"
    if kind == "calendar":
        return f"Usually starts after the calendar event \"{item['calendar_event']}\""
    if kind == "subject":
        return f"Usually starts after the message \"{item['subject']}\" in {app}"
    if kind == "place":
        return f"Usually starts after {item['place']} in {app}"
    if kind == "title":
        return f"Usually starts after \"{item['title']}\" in {app}"
    return f"Usually starts after using {app}"


def _clock_distance(a: int, b: int) -> int:
    d = abs(a - b) % 1440
    return min(d, 1440 - d)


def _time_trigger(instances: dict[str, dict]) -> dict | None:
    clocks = [
        (iid, s["local_weekday"], s["local_minute"]) for iid, s in instances.items()
        if isinstance(s.get("local_weekday"), int) and isinstance(s.get("local_minute"), int)
    ]
    best: list[tuple[str, int, int]] = []
    for _, _, center in clocks:
        members = [c for c in clocks if _clock_distance(c[2], center) <= TIME_TOLERANCE_MINUTES]
        if len(members) > len(best):
            best = members
    if len(best) < MIN_INSTANCES or len(best) / len(instances) < MIN_SHARE:
        return None
    weekday, on_day = Counter(w for _, w, _ in best).most_common(1)[0]
    if on_day >= MIN_INSTANCES and on_day / len(best) >= MIN_SHARE:
        best = [c for c in best if c[1] == weekday]
        when = f"Most {_WEEKDAYS[weekday]}s"
    else:
        weekday = None
        when = "Most weekdays" if all(w < 5 for _, w, _ in best) else "Most days"
    minutes = sorted(m for _, _, m in best)
    minute = round(minutes[len(minutes) // 2] / 15) * 15 % 1440
    time = f"{minute // 60:02d}:{minute % 60:02d}"
    return {
        "kind": "time",
        "weekday": weekday,
        "time": time,
        "description": f"{when} around {time}",
        "instances": [iid for iid, _, _ in best],
    }


def _trigger(instances: dict[str, dict]) -> dict | None:
    found = _recurring(instances, _lead_in_keys)
    if found:
        _, key, members, item = found[0]
        trigger = {
            "kind": "event",
            "description": _event_trigger(key[0], item),
            "app": item.get("app", ""),
            "place": item.get("place", ""),
            "title": item.get("title", ""),
            "instances": members,
        }
        if item.get("subject"):
            trigger["subject"] = item["subject"]
    else:
        trigger = _time_trigger(instances)
        if trigger is None:
            return None
    calendars = Counter(
        item["calendar_event"]
        for iid in trigger["instances"]
        for item in instances[iid].get("lead_in") or []
        if isinstance(item, dict) and item.get("calendar_event")
    )
    if calendars:
        trigger["calendar_event"] = calendars.most_common(1)[0][0]
    trigger["support"] = f"{len(trigger['instances'])} of {len(instances)} runs"
    return trigger


def _done(instances: dict[str, dict]) -> dict | None:
    def keys_of(signals: dict):
        done = signals.get("done")
        if isinstance(done, dict) and done.get("label"):
            yield 0, (done.get("kind"), _norm(done["label"]), done.get("place", "")), done

    found = _recurring(instances, keys_of, min_instances=1)
    if not found:
        return None
    _, _, members, done = found[0]
    where = f" on {done['place']}" if done.get("place") else (
        f" in {done['app']}" if done.get("app") else ""
    )
    if done.get("kind") == "submit":
        description = f"Done when the form is submitted (\"{done['label']}\"){where}"
    else:
        description = f"Done when \"{done['label']}\" is clicked{where}"
    event_ids = [
        instances[iid]["done"]["event_id"] for iid in members
        if instances[iid]["done"].get("event_id")
    ]
    return {
        "kind": done.get("kind"),
        "label": done["label"],
        "app": done.get("app", ""),
        "place": done.get("place", ""),
        "description": description,
        "instances": members,
        "event_ids": event_ids,
        "support": f"{len(members)} of {len(instances)} runs",
    }


def _verification(instances: dict[str, dict]) -> dict | None:
    def keys_of(signals: dict):
        for item in signals.get("verification") or []:
            if isinstance(item, dict):
                yield 0, _check_key(item), item

    checks = [
        {
            "app": item.get("app", ""),
            "place": item.get("place", ""),
            "title": item.get("title", ""),
            "description": (
                f"Within a minute of finishing, checks {item.get('place') or item.get('title')}"
                + (f" in {item['app']}" if item.get("app") else "")
            ),
            "instances": members,
            "support": f"{len(members)} of {len(instances)} runs",
        }
        for _, _, members, item in _recurring(instances, keys_of)
    ]
    corrections = [
        {**c, "instance": iid}
        for iid, signals in instances.items()
        for c in signals.get("corrections") or []
        if isinstance(c, dict) and c.get("after")
    ][-_MAX_ITEMS:]
    if not checks and not corrections:
        return None
    members = {iid for check in checks for iid in check["instances"]}
    members |= {c["instance"] for c in corrections}
    return {
        "checks": checks,
        "corrections": corrections,
        "instances": [iid for iid in instances if iid in members],
    }


def learn(proc: dict) -> dict:
    """Recompute ``proc["learned"]`` from its instances' signals, in place
    (removed when nothing recurs), and return it."""
    instances = _instances(proc)
    learned = {}
    if instances:
        for name, value in (
            ("trigger", _trigger(instances)),
            ("done", _done(instances)),
            ("verification", _verification(instances)),
        ):
            if value is not None:
                learned[name] = value
    if learned:
        proc["learned"] = learned
    else:
        proc.pop("learned", None)
    return learned

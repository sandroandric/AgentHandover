"""Replay a captured events database on a virtual clock and score what
passive discovery finds against labels.

The replay is the worker as ``main()`` runs it, driven by the journey driver
(``driver.JourneyWorker.run_tick``) in a throwaway ``HOME``:

* the source rows reach a fresh daemon-migrated ``events.db`` as the virtual
  clock passes their timestamps, so every stage sees only what had been
  captured by then (the daemon's columns plus the annotator's
  ``annotation_status`` / ``scene_annotation_json`` / ``frame_diff_json``);
* one ``run_tick()`` per ``--step`` of virtual time: the event pipeline, and
  passive discovery on its 2-hour cadence over the 24 hours before the
  virtual now;
* after each midnight, the daily batch's summary and digest for the day
  that ended (``DailyBatchProcessor.process_day`` with the production
  ``ProcedureMatcher``, then ``DigestGenerator``), as ``main()`` does;
* models answer from an Ollama cassette (``--cassette``, recorded once with
  ``--record``) or a live Ollama (``--ollama``).  Config is the production
  default apart from the host and any ``--vlm`` keys.

Scoring: every passive SOP generation is a *candidate*, made from the task
segments it marked generated.  Each labelled event belongs to one unit (a
routine, a one-off task, or noise).  A candidate is a *chimera* when its
events span two or more work units (routines or one-off tasks), a *true
positive* when they span exactly one and it is a routine.  Precision is true
positives over candidates, recall the share of routines with a true
positive, and days-to-discovery the time from a routine's first instance to
its first true positive.  Each day's digest must have content: active hours
above zero and close to the daily summary's, at least one highlight, task
minutes above zero.

Usage (from ``worker/tests``)::

    python -m journeys.replay --fixture passive_3weeks --cassette journeys/cassettes/passive_3weeks.jsonl
    python -m journeys.replay --db ~/events.db --labels labels.json --from 2026-08-31 --to 2026-09-21 --ollama http://localhost:11434

Exit status 1 when a threshold (``--min-recall``, ``--min-precision``,
``--max-chimeras``) or a digest check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # worker/tests

from daemon_db import create_daemon_db  # noqa: E402
from journeys.cassette import Cassette  # noqa: E402
from journeys.driver import Clock, JourneyWorker  # noqa: E402
from journeys.fixture_loader import DAEMON_COLUMNS, load_fixture, read_labels  # noqa: E402

ANNOTATOR_COLUMNS = ("annotation_status", "scene_annotation_json", "frame_diff_json")
# The digest's active hours sum task durations; the summary's count the
# spans between frames.  Both measure the same day, so they must agree
# within this fraction of the summary's value.
ACTIVE_HOURS_TOLERANCE = 0.5


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    sop_id: str
    slug: str
    title: str
    cluster_id: int
    generated_at: str
    segments: list[list[str]]
    units: dict[str, int] = field(default_factory=dict)  # work unit -> events
    verdict: str = ""  # "true_positive", "chimera" or "false_positive"
    routine: str | None = None


@dataclass
class DayCheck:
    date: str
    summary_active_hours: float
    digest_active_hours: float
    highlights: int
    task_minutes: float
    failures: list[str]


@dataclass
class ReplayReport:
    start: str
    end: str
    ticks: int
    passive_runs: int
    candidates: list[Candidate]
    failed_generations: int
    routines: dict[str, dict]
    precision: float
    recall: float
    chimeras: int
    days: list[DayCheck]
    model_seconds: float | None = None

    def failures(self, min_recall: float, min_precision: float, max_chimeras: int) -> list[str]:
        out = []
        if self.recall < min_recall:
            out.append(f"recall {self.recall:.2f} < {min_recall}")
        if self.precision < min_precision:
            out.append(f"precision {self.precision:.2f} < {min_precision}")
        if self.chimeras > max_chimeras:
            out.append(f"{self.chimeras} chimera candidate(s) > {max_chimeras}")
        if not self.days:
            out.append("no daily summary was produced")
        for day in self.days:
            out += [f"{day.date}: {msg}" for msg in day.failures]
        return out

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def unit_index(labels: dict) -> tuple[dict[str, str], set[str]]:
    """(event id -> unit, routine units).  Units: ``routine:<slug>``,
    ``task:<n>``, ``noise:<n>``."""
    unit_of: dict[str, str] = {}
    routines: set[str] = set()
    for routine in labels.get("routines", []):
        unit = f"routine:{routine['slug']}"
        routines.add(unit)
        for inst in routine["instances"]:
            for eid in inst["event_ids"] + inst.get("extra_event_ids", []):
                unit_of[eid] = unit
    for n, task in enumerate(labels.get("one_off_tasks", [])):
        for eid in task["event_ids"]:
            unit_of[eid] = f"task:{n}"
    for n, noise in enumerate(labels.get("noise", [])):
        for eid in noise["event_ids"]:
            unit_of[eid] = f"noise:{n}"
    return unit_of, routines


def score(candidates: list[Candidate], labels: dict, clock_start: datetime) -> tuple[float, float, int, dict]:
    unit_of, routine_units = unit_index(labels)
    for cand in candidates:
        # Work units only: noise frames and unlabelled events decide nothing.
        units = Counter(
            unit for seg in cand.segments for eid in seg
            if (unit := unit_of.get(eid, "")).startswith(("routine:", "task:"))
        )
        cand.units = dict(units)
        if len(units) > 1:
            cand.verdict = "chimera"
        elif len(units) == 1 and next(iter(units)) in routine_units:
            cand.verdict = "true_positive"
            cand.routine = next(iter(units)).split(":", 1)[1]
        else:
            cand.verdict = "false_positive"

    routines: dict[str, dict] = {}
    for routine in labels.get("routines", []):
        slug = routine["slug"]
        first = min(
            (datetime.fromisoformat(i["start"]) for i in routine["instances"]),
            default=clock_start,
        )
        hits = [c for c in candidates if c.routine == slug]
        found = min((datetime.fromisoformat(c.generated_at) for c in hits), default=None)
        routines[slug] = {
            "instances": len(routine["instances"]),
            "true_positives": len(hits),
            "days_to_discovery": (
                round((found - first).total_seconds() / 86400, 2) if found else None
            ),
        }
    tp = sum(c.verdict == "true_positive" for c in candidates)
    precision = tp / len(candidates) if candidates else 0.0
    recall = (
        sum(r["true_positives"] > 0 for r in routines.values()) / len(routines)
        if routines else 0.0
    )
    chimeras = sum(c.verdict == "chimera" for c in candidates)
    return precision, recall, chimeras, routines


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@contextmanager
def throwaway_home(ollama_host: str, vlm: dict | None = None) -> Iterator[Path]:
    """A temp ``HOME`` with the default config pointed at *ollama_host*
    (and ``OLLAMA_HOST``), plus any ``[vlm]`` keys in *vlm*; restored on exit."""
    from agenthandover_worker import main as wm

    saved = {k: os.environ.get(k) for k in ("HOME", "OLLAMA_HOST")}
    with tempfile.TemporaryDirectory(prefix="ah-replay-") as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        os.environ["HOME"] = str(home)
        os.environ["OLLAMA_HOST"] = ollama_host
        try:
            data_dir = wm._status_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            config = (
                data_dir / "config.toml" if wm._platform.system() == "Darwin"
                else home / ".config" / "agenthandover" / "config.toml"
            )
            config.parent.mkdir(parents=True, exist_ok=True)
            vlm_lines = "".join(f"{k} = {json.dumps(v)}\n" for k, v in (vlm or {}).items())
            config.write_text(
                f'[vlm]\nollama_host = "{ollama_host}"\n{vlm_lines}\n'
                "[knowledge]\nquery_api_enabled = false\n",
                encoding="utf-8",
            )
            yield home
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def read_source(db_path: Path) -> tuple[list[str], list[tuple]]:
    """Every row of the source ``events`` table, oldest first, in the
    columns the replay copies (daemon columns plus the annotator's)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        present = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        columns = [c for c in DAEMON_COLUMNS + ANNOTATOR_COLUMNS if c in present]
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM events ORDER BY timestamp ASC"
        ).fetchall()
    finally:
        conn.close()
    return columns, rows


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(
    *,
    source_db: Path | None = None,
    fixture: str | None = None,
    labels: dict,
    start: datetime,
    end: datetime,
    cassette: Path | None = None,
    record: bool | None = None,
    upstream: str | None = None,
    ollama: str | None = None,
    vlm: dict | None = None,
    step: timedelta = timedelta(minutes=15),
) -> ReplayReport:
    """Replay [start, end) and score it; see the module docstring."""
    if (source_db is None) == (fixture is None):
        raise ValueError("give exactly one of source_db and fixture")
    if (cassette is None) == (ollama is None):
        raise ValueError("give exactly one of cassette and ollama")

    with ExitStack() as stack:
        tape = None
        if cassette is not None:
            tape = stack.enter_context(Cassette(cassette, record=record, upstream=upstream))
            host = tape.host
        else:
            host = ollama
        home = stack.enter_context(throwaway_home(host, vlm))

        from agenthandover_worker import main as wm
        from agenthandover_worker.daily_digest import DigestGenerator
        from agenthandover_worker.daily_processor import DailyBatchProcessor

        data_dir = wm._status_dir()
        if fixture is not None:
            staged = load_fixture(fixture, home / "source")
            source_db = staged.db_path
        columns, rows = read_source(source_db)
        db_path = create_daemon_db(data_dir / "events.db")

        clock = Clock(start)
        worker = stack.enter_context(JourneyWorker(
            db_path=db_path,
            knowledge_dir=home / ".agenthandover" / "knowledge",
            sops_dir=home / ".openclaw" / "workspace" / "memory" / "apprentice",
            clock=clock,
        ))
        daily = DailyBatchProcessor(
            knowledge_base=worker.knowledge_base, matcher=worker.procedure_matcher,
        )
        digests = DigestGenerator(worker.knowledge_base)

        write = sqlite3.connect(str(db_path))
        stack.callback(write.close)
        insert = (
            f"INSERT INTO events ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )
        ts_col = columns.index("timestamp")
        rows = [r for r in rows if _parse(r[ts_col]) < end]
        pending = [r for r in rows if _parse(r[ts_col]) >= start]
        # Rows before the start are history the worker had already seen.
        history = [r for r in rows if _parse(r[ts_col]) < start]
        write.executemany(insert, history)
        write.execute("UPDATE events SET processed = 1")
        write.commit()

        candidates: list[Candidate] = []
        days: list[DayCheck] = []
        ticks = passive_runs = 0
        last_day = start.date()
        while clock.now() < end:
            now = clock.now()
            due = [r for r in pending if _parse(r[ts_col]) <= now]
            if due:
                write.executemany(insert, due)
                write.commit()
                pending = pending[len(due):]

            generated_before = _generated_segments(write)
            sops_before = _passive_sop_ids(write)
            result = worker.run_tick()
            ticks += 1
            if result["passive_sops"] is not None:
                passive_runs += 1
                candidates += _new_candidates(write, generated_before, sops_before, now)

            if now.date() != last_day:
                days += _daily_batch(worker.db, daily, digests, now.date() - timedelta(days=1))
                last_day = now.date()
            clock.advance(step)

        failed = _count(write, "SELECT COUNT(*) FROM failed_generations WHERE source = 'passive'")
        precision, recall, chimeras, routines = score(candidates, labels, start)
        return ReplayReport(
            start=start.isoformat(),
            end=end.isoformat(),
            ticks=ticks,
            passive_runs=passive_runs,
            candidates=candidates,
            failed_generations=failed,
            routines=routines,
            precision=round(precision, 4),
            recall=round(recall, 4),
            chimeras=chimeras,
            days=days,
            model_seconds=round(tape.model_seconds(), 1) if tape is not None else None,
        )


def _count(conn: sqlite3.Connection, sql: str) -> int:
    try:
        return conn.execute(sql).fetchone()[0]
    except sqlite3.OperationalError:  # table not created yet
        return 0


def _generated_segments(conn: sqlite3.Connection) -> set[str]:
    try:
        return {r[0] for r in conn.execute(
            "SELECT segment_id FROM task_segments WHERE sop_generated = 1"
        )}
    except sqlite3.OperationalError:
        return set()


def _passive_sop_ids(conn: sqlite3.Connection) -> set[str]:
    try:
        return {r[0] for r in conn.execute(
            "SELECT sop_id FROM generated_sops WHERE source = 'passive'"
        )}
    except sqlite3.OperationalError:
        return set()


def _new_candidates(
    conn: sqlite3.Connection, generated_before: set[str], sops_before: set[str], now: datetime,
) -> list[Candidate]:
    """The SOPs this passive run generated, each with the segments it used
    (the cluster's segments it marked generated in the same run)."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"task_segments", "generated_sops"} <= tables:
        return []
    by_cluster: dict[int, list[list[str]]] = {}
    for seg_id, cluster_id, event_ids in conn.execute(
        "SELECT segment_id, cluster_id, event_ids_json FROM task_segments "
        "WHERE sop_generated = 1 ORDER BY start_time"
    ):
        if seg_id not in generated_before:
            by_cluster.setdefault(int(cluster_id), []).append(json.loads(event_ids))
    out = []
    for sop_id, slug, title, source_id in conn.execute(
        "SELECT sop_id, slug, title, source_id FROM generated_sops "
        "WHERE source = 'passive' ORDER BY created_at"
    ):
        if sop_id in sops_before:
            continue
        cluster_id = int(source_id)
        out.append(Candidate(
            sop_id=sop_id, slug=slug, title=title, cluster_id=cluster_id,
            generated_at=now.isoformat(), segments=by_cluster.get(cluster_id, []),
        ))
    return out


def _daily_batch(db, daily, digests, day) -> list[DayCheck]:
    """``main()``'s daily batch for *day*: summary, then digest; the checks."""
    date = day.isoformat()
    events = db.get_annotated_events_for_date(date)
    if not events:
        return []
    summary = daily.process_day(date, events)
    digest = digests.generate(date)
    digests.save_digest(digest)
    tasks = [t if isinstance(t, dict) else asdict(t) for t in summary.tasks]
    task_minutes = sum(t.get("duration_minutes", 0) for t in tasks)
    failures = []
    if digest.active_hours <= 0:
        failures.append("digest active_hours is 0")
    elif abs(digest.active_hours - summary.active_hours) > ACTIVE_HOURS_TOLERANCE * summary.active_hours:
        failures.append(
            f"digest active_hours {digest.active_hours} vs summary {summary.active_hours}"
        )
    if not digest.highlights:
        failures.append("digest has no highlights")
    if task_minutes <= 0:
        failures.append("task durations sum to 0 minutes")
    return [DayCheck(
        date=date,
        summary_active_hours=summary.active_hours,
        digest_active_hours=digest.active_hours,
        highlights=len(digest.highlights),
        task_minutes=round(task_minutes, 1),
        failures=failures,
    )]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _day(text: str) -> datetime:
    return datetime.combine(datetime.strptime(text, "%Y-%m-%d").date(), time(0), timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay", description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", type=Path, help="events.db to replay")
    source.add_argument("--fixture", help="a journey fixture (with its labels.json)")
    parser.add_argument("--labels", type=Path, help="labels.json (default: the fixture's)")
    parser.add_argument("--from", dest="start", type=_day,
                        help="first day, YYYY-MM-DD (UTC; default: the first row's day)")
    parser.add_argument("--to", dest="end", type=_day,
                        help="day after the last, YYYY-MM-DD (default: the day after the last row)")
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--cassette", type=Path, help="Ollama cassette (.jsonl)")
    models.add_argument("--ollama", help="a live Ollama host")
    parser.add_argument("--record", action="store_true", default=None,
                        help="record cassette misses from --upstream")
    parser.add_argument("--upstream", help="Ollama to record from (with --record)")
    parser.add_argument("--vlm", action="append", default=[], metavar="KEY=VALUE",
                        help="a [vlm] config key to set (repeatable), e.g. sop_model=qwen3.5:4b")
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--min-recall", type=float, default=1.0)
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--max-chimeras", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    if args.labels is not None:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
    elif args.fixture is not None:
        labels = read_labels(args.fixture)
    else:
        parser.error("--labels is required with --db")
    start, end = args.start, args.end
    if start is None or end is None:
        first, last = _row_span(args.db, args.fixture)
        start = start or _day(first.date().isoformat())
        end = end or _day(last.date().isoformat()) + timedelta(days=1)

    report = replay(
        source_db=args.db, fixture=args.fixture, labels=labels, start=start, end=end,
        cassette=args.cassette, record=args.record, upstream=args.upstream,
        ollama=args.ollama, step=timedelta(minutes=args.step_minutes),
        vlm=dict(item.split("=", 1) for item in args.vlm),
    )
    if args.json is not None:
        args.json.write_text(json.dumps(report.to_json(), indent=1) + "\n", encoding="utf-8")
    print(render(report))
    failures = report.failures(args.min_recall, args.min_precision, args.max_chimeras)
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


def _row_span(db: Path | None, fixture: str | None) -> tuple[datetime, datetime]:
    if fixture is not None:
        from journeys.fixture_loader import read_manifest
        manifest = read_manifest(fixture)
        return _parse(manifest["recorded_start"]), _parse(manifest["recorded_end"])
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        first, last = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events").fetchone()
    finally:
        conn.close()
    return _parse(first), _parse(last)


def render(report: ReplayReport) -> str:
    lines = [
        f"replayed {report.start} .. {report.end}: {report.ticks} ticks, "
        f"{report.passive_runs} passive runs, {len(report.candidates)} candidates, "
        f"{report.failed_generations} failed generations",
        f"precision {report.precision:.2f}  recall {report.recall:.2f}  chimeras {report.chimeras}",
    ]
    for slug, r in report.routines.items():
        lines.append(
            f"  {slug}: {r['true_positives']} true positive(s) over {r['instances']} instances, "
            f"days to discovery {r['days_to_discovery']}"
        )
    for c in report.candidates:
        lines.append(f"  candidate {c.generated_at[:16]} {c.verdict:14} {c.title[:60]!r} {c.units}")
    bad_days = [d for d in report.days if d.failures]
    lines.append(f"daily digests: {len(report.days)} days, {len(bad_days)} failing")
    if report.model_seconds is not None:
        lines.append(f"model seconds served: {report.model_seconds}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

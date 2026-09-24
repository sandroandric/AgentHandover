"""Idle-Time Scheduler — gate heavy pipeline jobs on system conditions.

Implements sections 15 + 12.2 of the AgentHandover spec: runs the D/E/F
pipeline (episode builder, semantic translator) only when
system conditions allow.

Gate conditions (from config.example.toml [idle_jobs]):
- AC power required (optional)
- Minimum battery percentage
- Maximum CPU utilisation
- Maximum CPU temperature
- Time-of-day window (supports midnight-crossing windows)

The scheduler probes macOS system state using the same approach as the
Rust daemon's ``macos_power`` module: ``pmset -g batt`` for power state,
``ps -A -o %cpu`` divided by the core count for CPU usage, and
``pmset -g therm`` CPU_Speed_Limit for thermal proxy.  Passive annotation
has its own, lighter policy (:meth:`IdleJobGate.passive_annotation_allowed`):
AC power, a locked screen, or two minutes without input (``ioreg``).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import plistlib
import subprocess
import xml.parsers.expat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, time
from time import monotonic

from agenthandover_worker.ops_telemetry import MODEL_USAGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Configuration for the idle-time scheduler gate.

    Default values mirror ``config.example.toml`` [idle_jobs].
    """

    require_ac_power: bool = True
    min_battery_percent: int = 50
    max_cpu_percent: int = 30
    max_temp_c: int = 80
    run_window_start: time = field(default_factory=lambda: time(1, 0))   # 01:00
    run_window_end: time = field(default_factory=lambda: time(5, 0))     # 05:00
    check_interval_seconds: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# System conditions snapshot
# ---------------------------------------------------------------------------


@dataclass
class SystemConditions:
    """A point-in-time snapshot of system resource state.

    ``cpu_temp_c`` is None when the temperature cannot be read (Apple
    Silicon's ``pmset -g therm`` reports no ``CPU_Speed_Limit``); the gate
    then skips the temperature check instead of blocking forever.
    """

    on_ac_power: bool
    battery_percent: int
    cpu_percent: float
    cpu_temp_c: float | None
    current_time: time

    def to_dict(self) -> dict:
        """Serialise to a plain dict for logging / JSON export."""
        return {
            "on_ac_power": self.on_ac_power,
            "battery_percent": self.battery_percent,
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_temp_c": (
                None if self.cpu_temp_c is None else round(self.cpu_temp_c, 1)
            ),
            "current_time": self.current_time.isoformat(),
        }


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome of evaluating all gate conditions.

    ``conditions`` is None when the gate stopped at the time window and
    never probed the system.
    """

    can_run: bool
    blockers: list[str]
    conditions: SystemConditions | None


# ---------------------------------------------------------------------------
# IdleJobGate — evaluates whether heavy processing may run
# ---------------------------------------------------------------------------

# Each system probe (a subprocess) is reused for this long.
PROBE_TTL_SECONDS = 60.0

# Passive annotation on battery waits for this much user inactivity.
PASSIVE_IDLE_SECONDS = 120.0


class IdleJobGate:
    """Controls whether heavy processing modules are allowed to run.

    Evaluates AC power, battery level, CPU utilisation, CPU temperature,
    and time-of-day window against the supplied ``SchedulerConfig``.
    The time window is evaluated first, so outside it no subprocess is
    started; each probe's answer is cached for ``PROBE_TTL_SECONDS``.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config = config or SchedulerConfig()
        self._clock = clock
        self._probe_cache: dict[str, tuple[float, object]] = {}

    def check(self, conditions: SystemConditions | None = None) -> GateResult:
        """Evaluate all gate conditions.

        Parameters
        ----------
        conditions:
            Pre-built snapshot for deterministic testing.  When ``None``
            the gate probes the live system (after the time window passed).

        Returns
        -------
        GateResult
            ``can_run`` is ``True`` only when every condition passes.
            ``blockers`` lists human-readable reasons for each failure.
        """
        if conditions is None:
            now = datetime.now().time()
            if not self._in_time_window(now):
                return GateResult(
                    can_run=False,
                    blockers=[self._window_blocker(now)],
                    conditions=None,
                )
            conditions = self._probe_system(now)

        blockers: list[str] = []

        if self.config.require_ac_power and not conditions.on_ac_power:
            blockers.append("not_on_ac_power")

        if conditions.battery_percent < self.config.min_battery_percent:
            blockers.append(
                f"battery_low ({conditions.battery_percent}% < {self.config.min_battery_percent}%)"
            )

        if conditions.cpu_percent > self.config.max_cpu_percent:
            blockers.append(
                f"cpu_high ({conditions.cpu_percent:.1f}% > {self.config.max_cpu_percent}%)"
            )

        if (
            conditions.cpu_temp_c is not None
            and conditions.cpu_temp_c > self.config.max_temp_c
        ):
            blockers.append(
                f"temp_high ({conditions.cpu_temp_c:.1f}C > {self.config.max_temp_c}C)"
            )

        if not self._in_time_window(conditions.current_time):
            blockers.append(self._window_blocker(conditions.current_time))

        return GateResult(
            can_run=len(blockers) == 0,
            blockers=blockers,
            conditions=conditions,
        )

    def passive_annotation_allowed(self) -> tuple[bool, str]:
        """Whether passive (non-focus) annotation may run now, and why.

        It runs on AC power, while the screen is locked, or once the user
        has been idle for ``PASSIVE_IDLE_SECONDS``; on battery with the
        user at the keyboard it waits.  Focus processing does not ask.
        """
        on_ac, _battery = self._cached("power", self._get_power_state)
        if on_ac:
            return True, "on_ac_power"
        if self._cached("locked", self._get_screen_locked):
            return True, "screen_locked"
        idle = self._cached("idle", self._get_idle_seconds)
        if idle is not None and idle >= PASSIVE_IDLE_SECONDS:
            return True, "user_idle"
        return False, "on_battery_user_active"

    # ------------------------------------------------------------------
    # Time window
    # ------------------------------------------------------------------

    def _in_time_window(self, current: time) -> bool:
        """Check if *current* time falls within the configured run window.

        Handles midnight-crossing windows correctly.  For example, a
        window of 23:00-05:00 includes 02:00 but excludes 14:00.
        """
        start = self.config.run_window_start
        end = self.config.run_window_end

        if start <= end:
            # Normal window (e.g. 01:00-05:00)
            return start <= current <= end
        else:
            # Midnight-crossing window (e.g. 23:00-05:00)
            return current >= start or current <= end

    def _window_blocker(self, current: time) -> str:
        return (
            f"outside_time_window ({current.isoformat()} "
            f"not in {self.config.run_window_start.isoformat()}-{self.config.run_window_end.isoformat()})"
        )

    # ------------------------------------------------------------------
    # System probing (macOS)
    # ------------------------------------------------------------------

    def _cached(self, name: str, probe: Callable[[], object]):
        """*probe*'s answer, reused for ``PROBE_TTL_SECONDS``."""
        now = self._clock()
        hit = self._probe_cache.get(name)
        if hit is not None and now - hit[0] < PROBE_TTL_SECONDS:
            return hit[1]
        value = probe()
        self._probe_cache[name] = (now, value)
        return value

    def _probe_system(self, now: time) -> SystemConditions:
        """Probe live system conditions (macOS implementation).

        Falls back to safe defaults when a power or CPU probe fails so the
        gate errs on the side of *not* running heavy jobs.
        """
        on_ac, battery = self._cached("power", self._get_power_state)
        cpu = self._cached("cpu", self._get_cpu_usage)
        temp = self._cached("temp", self._get_cpu_temp)

        return SystemConditions(
            on_ac_power=on_ac,
            battery_percent=battery,
            cpu_percent=cpu,
            cpu_temp_c=temp,
            current_time=now,
        )

    @staticmethod
    def _get_power_state() -> tuple[bool, int]:
        """Get AC power status and battery percentage.

        Parses ``pmset -g batt`` output.  Returns ``(False, 0)`` on
        failure so the gate blocks heavy jobs when state is unknown.
        """
        if platform.system() != "Darwin":
            # Non-macOS: assume desktop on AC with full battery
            return (True, 100)

        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            stdout = result.stdout

            on_ac = "AC Power" in stdout

            # Parse battery percentage from lines containing '%'
            battery = 100
            for line in stdout.splitlines():
                if "%" in line:
                    # Format: "  ... <tab>NN%;..."
                    for part in line.split("\t"):
                        if "%" in part:
                            pct_str = part.strip().split("%")[0].strip()
                            try:
                                battery = int(pct_str)
                            except ValueError:
                                pass
                            break
                    break

            return (on_ac, battery)

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Failed to probe power state: %s", exc)
            return (False, 0)

    @staticmethod
    def _get_cpu_usage() -> float:
        """Get current aggregate CPU utilisation percentage.

        Sums per-process CPU from ``ps -A -o %cpu`` and divides by the
        number of logical cores.  Returns 100.0 on failure (blocks heavy
        jobs).
        """
        if platform.system() != "Darwin":
            try:
                # Linux fallback: read /proc/loadavg
                with open("/proc/loadavg") as f:
                    load_1m = float(f.read().split()[0])
                ncpu = os.cpu_count() or 1
                return min(100.0, (load_1m / ncpu) * 100.0)
            except (OSError, ValueError):
                return 100.0

        try:
            # Get per-process CPU totals
            ps_result = subprocess.run(
                ["ps", "-A", "-o", "%cpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total_cpu: float = 0.0
            for line in ps_result.stdout.splitlines()[1:]:  # skip header
                stripped = line.strip()
                if stripped:
                    try:
                        total_cpu += float(stripped)
                    except ValueError:
                        pass

            return total_cpu / max(os.cpu_count() or 1, 1)

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("Failed to probe CPU usage: %s", exc)
            return 100.0

    @staticmethod
    def _get_cpu_temp() -> float | None:
        """Get a CPU temperature estimate from thermal pressure.

        Parses ``pmset -g therm`` for ``CPU_Speed_Limit`` and maps it to an
        approximate temperature using the same linear model as the Rust
        daemon: ``temp = 50 + (100 - limit) * 0.5``.

        Returns None when there is no reading (no ``CPU_Speed_Limit`` line,
        as on Apple Silicon, or the probe failed): the gate then skips the
        temperature check rather than blocking every job for good.
        """
        if platform.system() != "Darwin":
            return None

        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Failed to probe CPU temperature: %s", exc)
            return None

        for line in result.stdout.splitlines():
            if "CPU_Speed_Limit" in line:
                tokens = line.split()
                if tokens:
                    try:
                        limit = int(tokens[-1])
                    except ValueError:
                        continue
                    # Same mapping as Rust daemon macos_power.rs:
                    # 100 = ~50C, 80 = ~70C, 50 = ~90C
                    return 50.0 + (100.0 - limit) * 0.5
        return None

    @staticmethod
    def _get_idle_seconds() -> float | None:
        """Seconds since the last keyboard or mouse input (``HIDIdleTime``).

        None when it cannot be read (non-macOS or a failed probe).
        """
        if platform.system() != "Darwin":
            return None
        try:
            result = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem", "-d", "4", "-r", "-k", "HIDIdleTime"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Failed to probe idle time: %s", exc)
            return None
        return parse_hid_idle_seconds(result.stdout)

    @staticmethod
    def _get_screen_locked() -> bool:
        """Whether the console session's screen is locked (``ioreg`` root)."""
        if platform.system() != "Darwin":
            return False
        try:
            result = subprocess.run(
                ["ioreg", "-n", "Root", "-d1", "-a"],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Failed to probe screen lock: %s", exc)
            return False
        return parse_screen_locked(result.stdout)


def parse_hid_idle_seconds(ioreg_output: str) -> float | None:
    """``"HIDIdleTime" = <ns>`` from ``ioreg`` output, in seconds."""
    for line in ioreg_output.splitlines():
        if '"HIDIdleTime"' in line and "=" in line:
            try:
                return int(line.rsplit("=", 1)[1].strip()) / 1e9
            except ValueError:
                return None
    return None


def parse_screen_locked(ioreg_plist: bytes) -> bool:
    """True when a console user's ``CGSSessionScreenIsLocked`` is set."""
    try:
        root = plistlib.loads(ioreg_plist)
    except (plistlib.InvalidFileException, ValueError, xml.parsers.expat.ExpatError):
        return False
    users = root.get("IOConsoleUsers", []) if isinstance(root, dict) else []
    return any(
        isinstance(user, dict) and user.get("CGSSessionScreenIsLocked") is True
        for user in users
    )


# ---------------------------------------------------------------------------
# AnnotationBudget — daily and hourly caps on passive model use
# ---------------------------------------------------------------------------

HOUR_SECONDS = 3600.0
# A passive backlog that needs this many hours of the hourly frame cap to
# clear asks the daemon to keep fewer passive frames (worker-status.json
# ``inference.lower_capture_cadence``).
CAPTURE_BACKLOG_HOURS = 4
DAILY_BUDGET_REASON = "annotation_budget_exhausted"
HOURLY_BUDGET_REASON = "annotation_hourly_budget_exhausted"


def model_compute_seconds() -> float:
    """Ollama's prompt + generation seconds over every model since start-up."""
    return sum(
        usage.get("prompt_eval_seconds", 0.0) + usage.get("eval_seconds", 0.0)
        for usage in MODEL_USAGE.snapshot().values()
    )


@dataclass
class AnnotationBudget:
    """What passive model use has spent in the local day and the last hour.

    Two caps per local day (``[vlm] annotation_max_frames_per_day`` and
    ``annotation_max_compute_minutes_per_day``) and two per rolling hour
    (``annotation_max_frames_per_hour`` and
    ``annotation_max_compute_minutes_per_hour``).  Both frame caps count
    frames sent to the model; a cache hit is free (WP-F6: it makes no
    model call, and the diff and classification calls that follow it are
    charged by the compute caps).  Hits in the hour are still counted for
    the status.  Compute is Ollama's own ``prompt_eval_duration +
    eval_duration`` of every model call made under :func:`charged_to`:
    annotation, activity classification, frame diffs, passive discovery.
    Focus recordings are not charged.

    :meth:`roll` sets the time: the day's counters start over on a new
    date, the hour's charges older than an hour drop out.
    """

    max_frames: int
    max_compute_seconds: float
    max_frames_per_hour: int = 60
    max_compute_seconds_per_hour: float = 600.0
    date: str = ""
    frames: int = 0
    compute_seconds: float = 0.0
    # Epoch seconds of the last roll; charges are stamped with it.
    now: float = 0.0
    # [time, model-annotated frames, compute seconds, cache hits] per
    # charge in the last hour.
    recent: list = field(default_factory=list)

    @classmethod
    def from_config(cls, vlm: object) -> "AnnotationBudget":
        """The budget ``[vlm]`` configures (a ``config.VlmConfig``)."""
        return cls(
            max_frames=vlm.annotation_max_frames_per_day,
            max_compute_seconds=vlm.annotation_max_compute_minutes_per_day * 60.0,
            max_frames_per_hour=vlm.annotation_max_frames_per_hour,
            max_compute_seconds_per_hour=vlm.annotation_max_compute_minutes_per_hour * 60.0,
        )

    def roll(self, now: datetime) -> None:
        """Move to *now* (local wall time for the date, as ``datetime.now()``)."""
        today = now.date().isoformat()
        if today != self.date:
            self.date, self.frames, self.compute_seconds = today, 0, 0.0
        self.now = now.timestamp()
        cutoff = self.now - HOUR_SECONDS
        self.recent = [entry for entry in self.recent if entry[0] > cutoff]

    def _live(self) -> list:
        cutoff = self.now - HOUR_SECONDS
        return [entry for entry in self.recent if entry[0] > cutoff]

    def hour(self) -> tuple[int, float]:
        """(frames sent to the model, compute seconds) in the hour before
        :attr:`now`: what the hourly caps compare."""
        live = self._live()
        return (
            int(sum(entry[1] for entry in live)),
            float(sum(entry[2] for entry in live)),
        )

    def hour_cache_hits(self) -> int:
        """Frames the annotation cache answered in the hour (not capped)."""
        return int(sum(entry[3] for entry in self._live()))

    def asks_lower_capture(self, backlog: int) -> bool:
        """True when *backlog* passive frames need
        :data:`CAPTURE_BACKLOG_HOURS` or more of the hourly frame cap to
        clear (never without an hourly cap)."""
        return (
            self.max_frames_per_hour > 0
            and backlog >= CAPTURE_BACKLOG_HOURS * self.max_frames_per_hour
        )

    def pause_reason(self) -> str | None:
        """Why passive model use must wait, or None while both caps allow it."""
        if (
            self.frames >= self.max_frames
            or self.compute_seconds >= self.max_compute_seconds
        ):
            return DAILY_BUDGET_REASON
        frames, seconds = self.hour()
        if frames >= self.max_frames_per_hour or seconds >= self.max_compute_seconds_per_hour:
            return HOURLY_BUDGET_REASON
        return None

    def exhausted(self) -> bool:
        return self.pause_reason() is not None

    def compute_exhausted(self) -> bool:
        """Whether the day's or the hour's model-compute cap is spent.  Frame
        diffs wait on this, not on the frame caps: they follow frames that
        were already annotated."""
        _, seconds = self.hour()
        return (
            self.compute_seconds >= self.max_compute_seconds
            or seconds >= self.max_compute_seconds_per_hour
        )

    def charge_frame(self, *, model_call: bool) -> None:
        """Count one annotated frame against the day's and the hour's frame
        caps when the model was called for it; a cache hit is only
        recorded as a hit."""
        if model_call:
            self.frames += 1
            self.recent.append([self.now, 1, 0.0, 0])
        else:
            self.recent.append([self.now, 0, 0.0, 1])

    def charge_compute(self, seconds: float) -> None:
        """Count Ollama compute (see :func:`charged_to`)."""
        seconds = max(seconds, 0.0)
        if seconds:
            self.compute_seconds += seconds
            self.recent.append([self.now, 0, seconds, 0])

    def to_json(self) -> str:
        return json.dumps({
            "date": self.date,
            "frames": self.frames,
            "compute_seconds": round(self.compute_seconds, 3),
            "recent": [[t, f, round(s, 3), h] for t, f, s, h in self.recent],
        })

    def load(self, raw: str | None) -> None:
        """Restore the counters persisted by :meth:`to_json` (if any)."""
        try:
            saved = json.loads(raw) if raw else {}
            self.date = str(saved.get("date", ""))
            self.frames = int(saved.get("frames", 0))
            self.compute_seconds = float(saved.get("compute_seconds", 0.0))
            # Entries saved before WP-F6 have no hit count; a frame there
            # may have been a hit, but counting it as a model call only
            # keeps the old (stricter) cap for at most an hour.
            self.recent = [
                [float(e[0]), int(e[1]), float(e[2]), int(e[3]) if len(e) > 3 else 0]
                for e in saved.get("recent", [])
            ]
        except (ValueError, TypeError, AttributeError, IndexError):
            self.date, self.frames, self.compute_seconds, self.recent = "", 0, 0.0, []

    def status(self) -> dict:
        """The counters as published in ``worker-status.json``."""
        hour_frames, hour_seconds = self.hour()
        return {
            "date": self.date,
            "frames": self.frames,
            "max_frames": self.max_frames,
            "compute_minutes": round(self.compute_seconds / 60.0, 2),
            "max_compute_minutes": round(self.max_compute_seconds / 60.0, 2),
            "hour_frames": hour_frames,
            "hour_cache_hits": self.hour_cache_hits(),
            "max_frames_per_hour": self.max_frames_per_hour,
            "hour_compute_minutes": round(hour_seconds / 60.0, 2),
            "max_compute_minutes_per_hour": round(
                self.max_compute_seconds_per_hour / 60.0, 2,
            ),
        }


@contextmanager
def charged_to(budget: AnnotationBudget | None) -> Iterator[None]:
    """Charge *budget* the Ollama compute of every model call made inside
    the block (nothing when *budget* is None, as for focus recordings).

    The one way passive model use is charged: wrap each passive step that
    can call a model, so its calls count against the same hourly and
    daily caps as annotation.  The worker's loop is one thread, so the
    difference in :data:`MODEL_USAGE` over the block is that block's.
    """
    if budget is None:
        yield
        return
    before = model_compute_seconds()
    try:
        yield
    finally:
        budget.charge_compute(model_compute_seconds() - before)


# ---------------------------------------------------------------------------
# IdleScheduler — top-level scheduler using the gate
# ---------------------------------------------------------------------------


class IdleScheduler:
    """Scheduler that gates heavy pipeline jobs on system idle conditions.

    Wraps ``IdleJobGate`` and provides time-window parsing for
    configuration loading from TOML strings.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self.gate = IdleJobGate(self.config)
        self._running = False

    def should_run_now(self, conditions: SystemConditions | None = None) -> GateResult:
        """Check if conditions allow running heavy jobs right now.

        Parameters
        ----------
        conditions:
            Optional pre-built snapshot for deterministic testing.
            When ``None`` the gate probes the live system.
        """
        return self.gate.check(conditions)

    @staticmethod
    def parse_time_window(window_str: str) -> tuple[time, time]:
        """Parse ``'HH:MM-HH:MM'`` format into ``(start, end)`` time objects.

        Parameters
        ----------
        window_str:
            Time window string, e.g. ``"01:00-05:00"`` or ``"23:00-05:00"``.

        Returns
        -------
        tuple[time, time]
            ``(start_time, end_time)``

        Raises
        ------
        ValueError
            If the format is invalid.
        """
        parts = window_str.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid time window format: {window_str!r} (expected 'HH:MM-HH:MM')"
            )

        start_str = parts[0].strip()
        end_str = parts[1].strip()

        start_parts = start_str.split(":")
        end_parts = end_str.split(":")

        if len(start_parts) != 2 or len(end_parts) != 2:
            raise ValueError(
                f"Invalid time window format: {window_str!r} (expected 'HH:MM-HH:MM')"
            )

        try:
            start_time = time(int(start_parts[0]), int(start_parts[1]))
            end_time = time(int(end_parts[0]), int(end_parts[1]))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid time window format: {window_str!r} — {exc}"
            ) from exc

        return (start_time, end_time)

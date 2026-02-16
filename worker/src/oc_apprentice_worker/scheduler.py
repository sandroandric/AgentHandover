"""Idle-Time Scheduler — gate heavy pipeline jobs on system conditions.

Implements sections 15 + 12.2 of the OpenMimic spec: runs the D/E/F
pipeline (episode builder, semantic translator, SOP inducer) only when
system conditions allow.

Gate conditions (from config.example.toml [idle_jobs]):
- AC power required (optional)
- Minimum battery percentage
- Maximum CPU utilisation
- Maximum CPU temperature
- Time-of-day window (supports midnight-crossing windows)

The scheduler probes macOS system state using the same approach as the
Rust daemon's ``macos_power`` module: ``pmset -g batt`` for power state,
``ps -A -o %cpu`` divided by ``hw.ncpu`` for CPU usage, and
``pmset -g therm`` CPU_Speed_Limit for thermal proxy.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, time

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
    """A point-in-time snapshot of system resource state."""

    on_ac_power: bool
    battery_percent: int
    cpu_percent: float
    cpu_temp_c: float
    current_time: time

    def to_dict(self) -> dict:
        """Serialise to a plain dict for logging / JSON export."""
        return {
            "on_ac_power": self.on_ac_power,
            "battery_percent": self.battery_percent,
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_temp_c": round(self.cpu_temp_c, 1),
            "current_time": self.current_time.isoformat(),
        }


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome of evaluating all gate conditions."""

    can_run: bool
    blockers: list[str]
    conditions: SystemConditions


# ---------------------------------------------------------------------------
# IdleJobGate — evaluates whether heavy processing may run
# ---------------------------------------------------------------------------


class IdleJobGate:
    """Controls whether heavy processing modules are allowed to run.

    Evaluates AC power, battery level, CPU utilisation, CPU temperature,
    and time-of-day window against the supplied ``SchedulerConfig``.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def check(self, conditions: SystemConditions | None = None) -> GateResult:
        """Evaluate all gate conditions.

        Parameters
        ----------
        conditions:
            Pre-built snapshot for deterministic testing.  When ``None``
            the gate probes the live system.

        Returns
        -------
        GateResult
            ``can_run`` is ``True`` only when every condition passes.
            ``blockers`` lists human-readable reasons for each failure.
        """
        if conditions is None:
            conditions = self._probe_system()

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

        if conditions.cpu_temp_c > self.config.max_temp_c:
            blockers.append(
                f"temp_high ({conditions.cpu_temp_c:.1f}C > {self.config.max_temp_c}C)"
            )

        if not self._in_time_window(conditions.current_time):
            blockers.append(
                f"outside_time_window ({conditions.current_time.isoformat()} "
                f"not in {self.config.run_window_start.isoformat()}-{self.config.run_window_end.isoformat()})"
            )

        return GateResult(
            can_run=len(blockers) == 0,
            blockers=blockers,
            conditions=conditions,
        )

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

    # ------------------------------------------------------------------
    # System probing (macOS)
    # ------------------------------------------------------------------

    def _probe_system(self) -> SystemConditions:
        """Probe live system conditions (macOS implementation).

        Falls back to safe defaults when a probe fails so the gate
        errs on the side of *not* running heavy jobs.
        """
        on_ac, battery = self._get_power_state()
        cpu = self._get_cpu_usage()
        temp = self._get_cpu_temp()
        now = datetime.now().time()

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

        Sums per-process CPU from ``ps -A -o %cpu`` and divides by
        the number of logical cores from ``sysctl -n hw.ncpu``.
        Returns 100.0 on failure (blocks heavy jobs).
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

            # Get number of logical cores
            ncpu_result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ncpu = int(ncpu_result.stdout.strip()) if ncpu_result.stdout.strip() else 1
            ncpu = max(ncpu, 1)

            return total_cpu / ncpu

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("Failed to probe CPU usage: %s", exc)
            return 100.0

    @staticmethod
    def _get_cpu_temp() -> float:
        """Get CPU temperature estimate from thermal pressure.

        Parses ``pmset -g therm`` for ``CPU_Speed_Limit`` and maps it
        to an approximate temperature using the same linear model as
        the Rust daemon: ``temp = 50 + (100 - limit) * 0.5``.

        Returns 50.0 (cool) on failure so temperature alone does not
        block when the probe is unavailable.
        """
        if platform.system() != "Darwin":
            return 50.0

        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            stdout = result.stdout

            for line in stdout.splitlines():
                if "CPU_Speed_Limit" in line:
                    tokens = line.split()
                    if tokens:
                        try:
                            limit = int(tokens[-1])
                            # Same mapping as Rust daemon macos_power.rs:
                            # 100 = ~50C, 80 = ~70C, 50 = ~90C
                            return 50.0 + (100.0 - limit) * 0.5
                        except ValueError:
                            pass

            # No CPU_Speed_Limit found — assume cool
            return 50.0

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Failed to probe CPU temperature: %s", exc)
            return 50.0


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

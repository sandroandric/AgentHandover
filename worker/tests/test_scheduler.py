"""Tests for agenthandover_worker.scheduler.

Covers the IdleJobGate condition checks, time-window logic (including
midnight-crossing), multiple-blocker accumulation, relaxed configs,
parse_time_window string parsing, and GateResult structure.
"""

from __future__ import annotations

from datetime import time

from agenthandover_worker.scheduler import (
    GateResult,
    IdleJobGate,
    IdleScheduler,
    SchedulerConfig,
    SystemConditions,
)


def _make_conditions(
    *,
    on_ac_power: bool = True,
    battery_percent: int = 100,
    cpu_percent: float = 10.0,
    cpu_temp_c: float = 55.0,
    current_time: time = time(3, 0),  # 03:00 — inside default window
) -> SystemConditions:
    """Build a ``SystemConditions`` snapshot with sensible defaults.

    Every field defaults to a value that passes the default
    ``SchedulerConfig`` gate, so individual tests can override just the
    field under test.
    """
    return SystemConditions(
        on_ac_power=on_ac_power,
        battery_percent=battery_percent,
        cpu_percent=cpu_percent,
        cpu_temp_c=cpu_temp_c,
        current_time=current_time,
    )


# ------------------------------------------------------------------
# 1. All conditions met → can_run = True
# ------------------------------------------------------------------


class TestAllConditionsMet:
    def test_all_conditions_met(self) -> None:
        """When every condition passes the gate allows running."""
        gate = IdleJobGate()
        conditions = _make_conditions()

        result = gate.check(conditions)

        assert result.can_run is True
        assert result.blockers == []
        assert result.conditions is conditions


# ------------------------------------------------------------------
# 2. Not on AC power → blocked
# ------------------------------------------------------------------


class TestNotOnAcPower:
    def test_not_on_ac_power(self) -> None:
        """AC required but on battery → blocked with descriptive reason."""
        gate = IdleJobGate()
        conditions = _make_conditions(on_ac_power=False)

        result = gate.check(conditions)

        assert result.can_run is False
        assert len(result.blockers) == 1
        assert "not_on_ac_power" in result.blockers[0]


# ------------------------------------------------------------------
# 3. Battery too low → blocked
# ------------------------------------------------------------------


class TestBatteryTooLow:
    def test_battery_too_low(self) -> None:
        """30% battery when 50% minimum is required → blocked."""
        gate = IdleJobGate()
        conditions = _make_conditions(battery_percent=30)

        result = gate.check(conditions)

        assert result.can_run is False
        assert any("battery_low" in b for b in result.blockers)
        assert any("30%" in b for b in result.blockers)
        assert any("50%" in b for b in result.blockers)


# ------------------------------------------------------------------
# 4. CPU too high → blocked
# ------------------------------------------------------------------


class TestCpuTooHigh:
    def test_cpu_too_high(self) -> None:
        """45% CPU when 30% max is configured → blocked."""
        gate = IdleJobGate()
        conditions = _make_conditions(cpu_percent=45.0)

        result = gate.check(conditions)

        assert result.can_run is False
        assert any("cpu_high" in b for b in result.blockers)
        assert any("45.0%" in b for b in result.blockers)
        assert any("30%" in b for b in result.blockers)


# ------------------------------------------------------------------
# 5. Temperature too high → blocked
# ------------------------------------------------------------------


class TestTempTooHigh:
    def test_temp_too_high(self) -> None:
        """85C when 80C max is configured → blocked."""
        gate = IdleJobGate()
        conditions = _make_conditions(cpu_temp_c=85.0)

        result = gate.check(conditions)

        assert result.can_run is False
        assert any("temp_high" in b for b in result.blockers)
        assert any("85.0C" in b for b in result.blockers)
        assert any("80C" in b for b in result.blockers)


# ------------------------------------------------------------------
# 6. Outside time window → blocked
# ------------------------------------------------------------------


class TestOutsideTimeWindow:
    def test_outside_time_window(self) -> None:
        """14:00 is not in the 01:00-05:00 window → blocked."""
        gate = IdleJobGate()
        conditions = _make_conditions(current_time=time(14, 0))

        result = gate.check(conditions)

        assert result.can_run is False
        assert any("outside_time_window" in b for b in result.blockers)


# ------------------------------------------------------------------
# 7. Inside time window → passes
# ------------------------------------------------------------------


class TestInsideTimeWindow:
    def test_inside_time_window(self) -> None:
        """03:00 is within the 01:00-05:00 window → passes."""
        gate = IdleJobGate()
        conditions = _make_conditions(current_time=time(3, 0))

        result = gate.check(conditions)

        assert result.can_run is True
        assert result.blockers == []

    def test_at_window_start_boundary(self) -> None:
        """Exactly 01:00 is within the 01:00-05:00 window (inclusive start)."""
        gate = IdleJobGate()
        conditions = _make_conditions(current_time=time(1, 0))

        result = gate.check(conditions)

        assert result.can_run is True

    def test_at_window_end_boundary(self) -> None:
        """Exactly 05:00 is within the 01:00-05:00 window (inclusive end)."""
        gate = IdleJobGate()
        conditions = _make_conditions(current_time=time(5, 0))

        result = gate.check(conditions)

        assert result.can_run is True


# ------------------------------------------------------------------
# 8. Midnight-crossing window
# ------------------------------------------------------------------


class TestMidnightCrossingWindow:
    def test_midnight_crossing_inside_after_midnight(self) -> None:
        """23:00-05:00 window, test 02:00 → inside (after midnight portion)."""
        config = SchedulerConfig(
            run_window_start=time(23, 0),
            run_window_end=time(5, 0),
        )
        gate = IdleJobGate(config)
        conditions = _make_conditions(current_time=time(2, 0))

        result = gate.check(conditions)

        assert result.can_run is True
        assert result.blockers == []

    def test_midnight_crossing_inside_before_midnight(self) -> None:
        """23:00-05:00 window, test 23:30 → inside (before midnight portion)."""
        config = SchedulerConfig(
            run_window_start=time(23, 0),
            run_window_end=time(5, 0),
        )
        gate = IdleJobGate(config)
        conditions = _make_conditions(current_time=time(23, 30))

        result = gate.check(conditions)

        assert result.can_run is True
        assert result.blockers == []

    def test_midnight_crossing_outside(self) -> None:
        """23:00-05:00 window, test 14:00 → outside."""
        config = SchedulerConfig(
            run_window_start=time(23, 0),
            run_window_end=time(5, 0),
        )
        gate = IdleJobGate(config)
        conditions = _make_conditions(current_time=time(14, 0))

        result = gate.check(conditions)

        assert result.can_run is False
        assert any("outside_time_window" in b for b in result.blockers)


# ------------------------------------------------------------------
# 9. Multiple blockers
# ------------------------------------------------------------------


class TestMultipleBlockers:
    def test_multiple_blockers(self) -> None:
        """Several conditions fail → all listed in blockers."""
        gate = IdleJobGate()
        conditions = _make_conditions(
            on_ac_power=False,
            battery_percent=20,
            cpu_percent=50.0,
            cpu_temp_c=90.0,
            current_time=time(12, 0),
        )

        result = gate.check(conditions)

        assert result.can_run is False
        assert len(result.blockers) == 5

        blocker_text = " ".join(result.blockers)
        assert "not_on_ac_power" in blocker_text
        assert "battery_low" in blocker_text
        assert "cpu_high" in blocker_text
        assert "temp_high" in blocker_text
        assert "outside_time_window" in blocker_text


# ------------------------------------------------------------------
# 10. Relaxed config
# ------------------------------------------------------------------


class TestRelaxedConfig:
    def test_relaxed_config_passes(self) -> None:
        """No AC required, low battery min, wide CPU/temp limits → passes."""
        config = SchedulerConfig(
            require_ac_power=False,
            min_battery_percent=10,
            max_cpu_percent=90,
            max_temp_c=100,
            run_window_start=time(0, 0),
            run_window_end=time(23, 59),
        )
        gate = IdleJobGate(config)
        conditions = _make_conditions(
            on_ac_power=False,
            battery_percent=15,
            cpu_percent=60.0,
            cpu_temp_c=85.0,
            current_time=time(14, 0),
        )

        result = gate.check(conditions)

        assert result.can_run is True
        assert result.blockers == []

    def test_ac_not_required_allows_battery(self) -> None:
        """When require_ac_power is False, being on battery is fine."""
        config = SchedulerConfig(require_ac_power=False)
        gate = IdleJobGate(config)
        conditions = _make_conditions(on_ac_power=False)

        result = gate.check(conditions)

        # Only the AC check is relaxed; other defaults still apply
        assert "not_on_ac_power" not in " ".join(result.blockers)


# ------------------------------------------------------------------
# 11. parse_time_window
# ------------------------------------------------------------------


class TestParseTimeWindow:
    def test_parse_normal_window(self) -> None:
        """'01:00-05:00' → (time(1, 0), time(5, 0))."""
        scheduler = IdleScheduler()
        start, end = scheduler.parse_time_window("01:00-05:00")

        assert start == time(1, 0)
        assert end == time(5, 0)

    def test_parse_midnight_crossing_window(self) -> None:
        """'23:00-05:00' → (time(23, 0), time(5, 0))."""
        scheduler = IdleScheduler()
        start, end = scheduler.parse_time_window("23:00-05:00")

        assert start == time(23, 0)
        assert end == time(5, 0)

    def test_parse_with_spaces(self) -> None:
        """Whitespace around the dash is tolerated."""
        scheduler = IdleScheduler()
        start, end = scheduler.parse_time_window("02:30 - 06:45")

        assert start == time(2, 30)
        assert end == time(6, 45)

    def test_parse_invalid_format_raises(self) -> None:
        """Malformed input raises ValueError."""
        scheduler = IdleScheduler()
        import pytest

        with pytest.raises(ValueError, match="Invalid time window format"):
            scheduler.parse_time_window("not-a-time-window")

    def test_parse_missing_dash_raises(self) -> None:
        """No dash separator raises ValueError."""
        scheduler = IdleScheduler()
        import pytest

        with pytest.raises(ValueError, match="Invalid time window format"):
            scheduler.parse_time_window("0100 0500")


# ------------------------------------------------------------------
# 12. GateResult structure
# ------------------------------------------------------------------


class TestGateResultStructure:
    def test_gate_result_fields_populated(self) -> None:
        """All fields on GateResult are populated correctly."""
        gate = IdleJobGate()
        conditions = _make_conditions()

        result = gate.check(conditions)

        assert isinstance(result, GateResult)
        assert isinstance(result.can_run, bool)
        assert isinstance(result.blockers, list)
        assert isinstance(result.conditions, SystemConditions)
        assert result.conditions is conditions

    def test_conditions_to_dict(self) -> None:
        """SystemConditions.to_dict() returns a well-formed dict."""
        conditions = _make_conditions(
            on_ac_power=True,
            battery_percent=75,
            cpu_percent=22.567,
            cpu_temp_c=61.234,
            current_time=time(3, 15),
        )

        d = conditions.to_dict()

        assert d["on_ac_power"] is True
        assert d["battery_percent"] == 75
        assert d["cpu_percent"] == 22.6  # rounded to 1 decimal
        assert d["cpu_temp_c"] == 61.2   # rounded to 1 decimal
        assert d["current_time"] == "03:15:00"

    def test_gate_result_blocked_has_conditions(self) -> None:
        """Even when blocked the conditions snapshot is attached."""
        gate = IdleJobGate()
        conditions = _make_conditions(on_ac_power=False)

        result = gate.check(conditions)

        assert result.can_run is False
        assert result.conditions is conditions
        assert result.conditions.on_ac_power is False


# ------------------------------------------------------------------
# 13. IdleScheduler.should_run_now delegates to gate
# ------------------------------------------------------------------


class TestIdleSchedulerShouldRunNow:
    def test_should_run_now_delegates(self) -> None:
        """IdleScheduler.should_run_now() returns the same result as the gate."""
        scheduler = IdleScheduler()
        conditions = _make_conditions()

        result = scheduler.should_run_now(conditions)

        assert result.can_run is True
        assert result.blockers == []

    def test_should_run_now_blocked(self) -> None:
        """IdleScheduler.should_run_now() reports blockers from the gate."""
        scheduler = IdleScheduler()
        conditions = _make_conditions(cpu_percent=99.0)

        result = scheduler.should_run_now(conditions)

        assert result.can_run is False
        assert any("cpu_high" in b for b in result.blockers)


# ------------------------------------------------------------------
# 14. Default config matches spec
# ------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_config_values(self) -> None:
        """Default SchedulerConfig matches config.example.toml [idle_jobs]."""
        config = SchedulerConfig()

        assert config.require_ac_power is True
        assert config.min_battery_percent == 50
        assert config.max_cpu_percent == 30
        assert config.max_temp_c == 80
        assert config.run_window_start == time(1, 0)
        assert config.run_window_end == time(5, 0)
        assert config.check_interval_seconds == 300


# ------------------------------------------------------------------
# 15. Edge: battery exactly at minimum
# ------------------------------------------------------------------


class TestBoundaryConditions:
    def test_battery_exactly_at_minimum(self) -> None:
        """Battery at exactly the minimum percentage passes."""
        gate = IdleJobGate()
        conditions = _make_conditions(battery_percent=50)

        result = gate.check(conditions)

        # 50 is NOT < 50, so this should pass the battery check
        assert "battery_low" not in " ".join(result.blockers)

    def test_cpu_exactly_at_maximum(self) -> None:
        """CPU at exactly the maximum percentage passes."""
        gate = IdleJobGate()
        conditions = _make_conditions(cpu_percent=30.0)

        result = gate.check(conditions)

        # 30.0 is NOT > 30, so this should pass the CPU check
        assert "cpu_high" not in " ".join(result.blockers)

    def test_temp_exactly_at_maximum(self) -> None:
        """Temperature at exactly the maximum passes."""
        gate = IdleJobGate()
        conditions = _make_conditions(cpu_temp_c=80.0)

        result = gate.check(conditions)

        # 80.0 is NOT > 80, so this should pass the temp check
        assert "temp_high" not in " ".join(result.blockers)

    def test_battery_one_below_minimum_blocks(self) -> None:
        """Battery one percent below the minimum blocks."""
        gate = IdleJobGate()
        conditions = _make_conditions(battery_percent=49)

        result = gate.check(conditions)

        assert any("battery_low" in b for b in result.blockers)

    def test_cpu_just_above_maximum_blocks(self) -> None:
        """CPU just above the maximum blocks."""
        gate = IdleJobGate()
        conditions = _make_conditions(cpu_percent=30.1)

        result = gate.check(conditions)

        assert any("cpu_high" in b for b in result.blockers)

    def test_temp_just_above_maximum_blocks(self) -> None:
        """Temperature just above the maximum blocks."""
        gate = IdleJobGate()
        conditions = _make_conditions(cpu_temp_c=80.1)

        result = gate.check(conditions)

        assert any("temp_high" in b for b in result.blockers)

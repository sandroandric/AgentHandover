"""Pins for the trigger spool (the one control channel into the worker)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agenthandover_worker import trigger_spool


def _age(path: Path, ns: int) -> None:
    os.utime(path, ns=(ns, ns))


class TestWriteAndDrain:
    def test_two_requests_of_one_kind_are_both_handled(self, tmp_path: Path) -> None:
        # The single-slot file lost the first click when a second one
        # landed before the worker's next tick.
        trigger_spool.write_trigger(tmp_path, "approve", {"sop_id": "a", "action": "approve"})
        trigger_spool.write_trigger(tmp_path, "approve", {"sop_id": "b", "action": "approve"})

        seen: list[str] = []
        consumed = trigger_spool.drain(tmp_path, {"approve": lambda t: seen.append(t["sop_id"])})

        assert consumed == 2
        assert sorted(seen) == ["a", "b"]
        assert trigger_spool.pending(tmp_path) == []

    def test_drains_oldest_first_across_kinds(self, tmp_path: Path) -> None:
        late = trigger_spool.write_trigger(tmp_path, "approve", {"n": 2})
        early = trigger_spool.write_trigger(tmp_path, "lifecycle-promote", {"n": 1})
        _age(early, 1_000_000_000)
        _age(late, 2_000_000_000)

        order: list[int] = []
        handlers = {
            "approve": lambda t: order.append(t["n"]),
            "lifecycle-promote": lambda t: order.append(t["n"]),
        }
        trigger_spool.drain(tmp_path, handlers)
        assert order == [1, 2]

    def test_filename_carries_the_kind(self, tmp_path: Path) -> None:
        path = trigger_spool.write_trigger(tmp_path, "staleness-reviewed", {})
        assert path.parent == tmp_path / "triggers"
        assert trigger_spool.kind_of(path) == "staleness-reviewed"
        # Swift's UUID().uuidString is upper case.
        swift_name = Path("lifecycle-promote-8C9E1A2B-3C4D-4E5F-8A9B-0C1D2E3F4A5B.json")
        assert trigger_spool.kind_of(swift_name) == "lifecycle-promote"
        assert trigger_spool.kind_of(Path("approve-trigger.json")) is None

    def test_tmp_files_are_not_consumed(self, tmp_path: Path) -> None:
        spool = tmp_path / "triggers"
        spool.mkdir()
        (spool / ".approve-x.json.tmp").write_text("{}")
        assert trigger_spool.drain(tmp_path, {"approve": lambda t: None}) == 0
        assert (spool / ".approve-x.json.tmp").exists()

    def test_bad_files_are_discarded_and_the_drain_continues(self, tmp_path: Path) -> None:
        spool = tmp_path / "triggers"
        spool.mkdir()
        (spool / "approve-00000000-0000-4000-8000-000000000001.json").write_text("{not json")
        (spool / "approve-00000000-0000-4000-8000-000000000002.json").write_text("[1, 2]")
        (spool / "nonsense-00000000-0000-4000-8000-000000000003.json").write_text("{}")
        good = trigger_spool.write_trigger(tmp_path, "approve", {"sop_id": "ok"})

        def boom(_: dict) -> None:
            raise RuntimeError("handler bug")

        seen: list[str] = []
        consumed = trigger_spool.drain(tmp_path, {
            "approve": lambda t: seen.append(t["sop_id"]),
            "merge": boom,
        })
        assert consumed == 4
        assert seen == ["ok"]
        assert not good.exists()
        assert list(spool.iterdir()) == []

    def test_a_failing_handler_does_not_block_later_triggers(self, tmp_path: Path) -> None:
        first = trigger_spool.write_trigger(tmp_path, "merge", {})
        second = trigger_spool.write_trigger(tmp_path, "approve", {"sop_id": "x"})
        _age(first, 1_000_000_000)
        _age(second, 2_000_000_000)

        def boom(_: dict) -> None:
            raise RuntimeError("handler bug")

        seen: list[str] = []
        trigger_spool.drain(tmp_path, {"merge": boom, "approve": lambda t: seen.append(t["sop_id"])})
        assert seen == ["x"]
        assert trigger_spool.pending(tmp_path) == []

    def test_empty_or_missing_spool(self, tmp_path: Path) -> None:
        assert trigger_spool.drain(tmp_path, {}) == 0


class TestMigrateLegacy:
    def test_single_slot_files_move_into_the_spool(self, tmp_path: Path) -> None:
        (tmp_path / "approve-trigger.json").write_text(json.dumps({"sop_id": "s", "action": "approve"}))
        (tmp_path / "export-trigger.json").write_text(json.dumps({"format": "all"}))
        (tmp_path / "search-query-result.json").write_text("{}")  # a result, not a trigger

        assert trigger_spool.migrate_legacy(tmp_path) == 2
        assert not (tmp_path / "approve-trigger.json").exists()
        assert (tmp_path / "search-query-result.json").exists()

        seen: dict[str, dict] = {}
        trigger_spool.drain(tmp_path, {
            "approve": lambda t: seen.setdefault("approve", t),
            "export": lambda t: seen.setdefault("export", t),
        })
        assert seen == {"approve": {"sop_id": "s", "action": "approve"}, "export": {"format": "all"}}

    def test_every_kind_has_a_worker_handler(self) -> None:
        # main() asserts the handler table covers KINDS exactly; this pins
        # the names the app and the CLI write.
        assert set(trigger_spool.KINDS) == {
            "retry", "approve", "failed-query", "search-query", "recall-query",
            "trust-accept", "trust-dismiss", "staleness-reviewed",
            "lifecycle-promote", "merge", "drift-reviewed", "lifecycle-dismiss",
            "export",
        }


class TestPollBackoff:
    def test_idle_backoff_is_capped_at_four_times_the_poll_interval(self) -> None:
        from agenthandover_worker.main import _next_poll_interval

        interval = 2.0
        for _ in range(10):
            interval = _next_poll_interval(interval, 2.0, busy=False, triggers_consumed=0)
        assert interval == 8.0

    def test_ceiling_never_below_five_seconds(self) -> None:
        from agenthandover_worker.main import _next_poll_interval

        interval = 0.5
        for _ in range(10):
            interval = _next_poll_interval(interval, 0.5, busy=False, triggers_consumed=0)
        assert interval == 5.0

    def test_a_consumed_trigger_resets_the_interval(self) -> None:
        from agenthandover_worker.main import _next_poll_interval

        assert _next_poll_interval(8.0, 2.0, busy=False, triggers_consumed=1) == 2.0
        assert _next_poll_interval(8.0, 2.0, busy=True, triggers_consumed=0) == 2.0

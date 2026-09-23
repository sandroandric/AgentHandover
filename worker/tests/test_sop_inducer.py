"""The v1 focus fallback's template helpers (sop_inducer), and the removal
of the v1 PrefixSpan miner (WP-F5)."""

from __future__ import annotations

from agenthandover_worker import sop_inducer
from agenthandover_worker.sop_inducer import detect_postconditions, detect_preconditions


class TestDetectPreconditions:
    def test_common_app_detected(self) -> None:
        """App appearing in 80%+ of instances is a precondition."""
        instances = [
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Firefox"}}],
        ]
        assert detect_preconditions(instances) == ["app_open:Chrome"]

    def test_no_precondition_when_diverse(self) -> None:
        """No precondition when apps are diverse (< 80%)."""
        instances = [
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Firefox"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Safari"}}],
        ]
        assert detect_preconditions(instances) == []

    def test_url_precondition(self) -> None:
        """Common URL in first step is a precondition."""
        instances = [
            [{"step": "click", "target": "Submit", "pre_state": {"url": "https://app.com/login"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"url": "https://app.com/login"}}],
        ]
        assert detect_preconditions(instances) == ["url_open:https://app.com/login"]


class TestDetectPostconditions:
    def test_common_final_action(self) -> None:
        """Common final intent and target are postconditions."""
        instances = [
            [{"step": "type", "target": "Email"}, {"step": "click", "target": "Save"}],
            [{"step": "type", "target": "Email"}, {"step": "click", "target": "Save"}],
        ]
        assert detect_postconditions(instances) == [
            "final_action:click", "final_target:Save",
        ]

    def test_empty_instances(self) -> None:
        assert detect_postconditions([]) == []


class TestPrefixSpanMinerIsGone:
    """On the live test Mac the v1 miner turned passive episodes into a junk
    template every tick (switch_app_textedit_navigate_textedit_read_…, no
    description, steps without apps) at two model calls per tick.  Passive
    recurrence belongs to workflow_mining; only the focus fallback's
    single-demonstration template is left here."""

    def test_only_the_focus_template_remains(self) -> None:
        assert not hasattr(sop_inducer, "SOPInducer")
        public = {n for n in vars(sop_inducer) if not n.startswith("_")}
        assert "induce_from_focus_session" in public
        assert not any("induce" in n and n != "induce_from_focus_session" for n in public)

    def test_worker_no_longer_declares_prefixspan(self) -> None:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
        assert not any(d.startswith("prefixspan") for d in deps)
